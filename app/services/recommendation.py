"""도서 추천.

기존에는 `ChromaManager.recommend_book`이 Chroma 검색 + MySQL 조회/저장을 한
클래스에서 처리했다. 여기서는 벡터 검색(`BookVectorStore`)과 DB(`BookRepository`)를
분리하고 이 서비스가 둘을 조합한다.

추천은 "가장 가까운 1권"이 아니라 **후보 풀을 재정렬한 뒤** 고른다.
벡터 유사도만 쓰면 진단에서 확인한 것처럼(책끼리 코사인 0.84~0.93) 상위 후보의
점수 차이가 작아 선호 카테고리가 거의 반영되지 않는다. 재정렬 규칙은
`app.services.reranking`에 따로 두어, 평가 하네스도 같은 함수를 쓴다.

제외(`exclude_isbns`)
- 이미 읽었거나 최근에 추천한 책은 후보에서 뺀다. 제외 목록은
  `BookRepository.get_recent_recommended_isbns`가 만든다(최근 N일 추천 이력).
- 저장소가 그 메서드를 제공하지 않으면(테스트 대역 등) 제외 없이 동작한다.
  제외 이력을 읽다가 실패해도 추천 자체는 계속한다(로그만 남긴다).
- **후보가 전부 제외되면** 제외를 풀고 같은 규칙으로 순위를 만든다. 카탈로그가
  30권이고 하루 1권을 추천하면 한 달이면 전부 제외될 수 있어, "추천 없음"보다
  "제외를 풀되 그 사실을 응답에 표시"하는 쪽을 택했다. `/recommendations` 응답의
  `exclusion_relaxed`가 이 경우를 알린다.

반환 메시지 문자열은 클라이언트(Spring 서버)가 그대로 보던 값이라 유지했다.
`/recommend`는 1권(`Recommended book ID: N`)만 응답하고, 상위 k권은
`/recommendations`가 응답한다.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from app.core.errors import AppError
from app.services.database import BookRepository
from app.services.llm import EmbeddingService
from app.services.reranking import (
    Candidate,
    RankingWeights,
    build_candidates,
    category_overlap,
    exclude_candidates,
    mmr_rank,
    relevance_score,
)
from app.services.vector_store import BookVectorStore

logger = logging.getLogger("arabook.recommendation")

NO_PREFERENCES = "No preferences found for the member."
NO_BOOK = "No book found for the given preferences."
NO_BOOK_ID = "No book ID found for the given ISBN."

#: 최근 추천 이력을 며칠치 제외할지. 0 이하면 제외하지 않는다.
DEFAULT_EXCLUDE_RECENT_DAYS = 30
#: `/recommendations`가 한 번에 돌려줄 수 있는 최대 권수(응답 크기 상한).
MAX_RECOMMEND_LIMIT = 20


@dataclass(frozen=True)
class RankingOutcome:
    """순위 결과와 제외 처리 상태."""

    candidates: list[Candidate]
    #: 요청된 제외 ISBN 수
    excluded_isbn_count: int = 0
    #: 제외로 후보에서 실제로 빠진 권수
    excluded_candidate_count: int = 0
    #: 후보가 전부 제외되어 제외를 풀고 순위를 만든 경우 True
    exclusion_relaxed: bool = False


@dataclass(frozen=True)
class RankedBook:
    """응답용 추천 1권. 점수를 구성 요소까지 남겨 순위 이유를 볼 수 있게 한다."""

    rank: int
    isbn: str
    title: str
    sub_categories: list[str] = field(default_factory=list)
    main_category: str | None = None
    author: str | None = None
    similarity: float = 0.0
    category_overlap: float = 0.0
    score: float = 0.0
    book_id: int | None = None


@dataclass(frozen=True)
class RecommendationList:
    """`/recommendations` 응답을 만드는 데 필요한 값 전부."""

    member_id: int | str
    limit: int
    preferences: list[str] = field(default_factory=list)
    items: list[RankedBook] = field(default_factory=list)
    excluded_isbn_count: int = 0
    excluded_candidate_count: int = 0
    exclusion_relaxed: bool = False
    message: str | None = None


class RecommendationService:
    def __init__(
        self,
        store: BookVectorStore,
        repository: BookRepository,
        embedder: EmbeddingService,
        weights: RankingWeights | None = None,
        exclude_recent_days: int = DEFAULT_EXCLUDE_RECENT_DAYS,
    ):
        self._store = store
        self._repository = repository
        self._embedder = embedder
        self._weights = weights or RankingWeights()
        self._exclude_recent_days = exclude_recent_days

    @property
    def weights(self) -> RankingWeights:
        return self._weights

    @property
    def exclude_recent_days(self) -> int:
        return self._exclude_recent_days

    async def recommend(self, member_id: int | str) -> str:
        """기존 `/recommend` 동작: 1권을 골라 이력을 남기고 메시지로 응답한다."""
        preferences = await asyncio.to_thread(self._repository.get_member_preferences, member_id)
        if not preferences:
            return NO_PREFERENCES

        outcome = await self._rank(
            member_id,
            preferences,
            limit=1,
            exclude_isbns=await self._excluded_isbns(member_id),
        )
        if not outcome.candidates:
            return NO_BOOK

        isbn = outcome.candidates[0].isbn
        book_id = await asyncio.to_thread(self._repository.get_book_id_by_isbn, isbn)
        if not book_id:
            return NO_BOOK_ID

        # 기존 동작 유지: '내일' 날짜로 추천 이력을 남긴다.
        recommendation_date = (datetime.now() + timedelta(days=1)).date()
        await asyncio.to_thread(
            self._repository.save_recommendation, member_id, book_id, recommendation_date
        )
        return f"Recommended book ID: {book_id}"

    async def rank(
        self,
        member_id: int | str,
        limit: int | None = None,
        exclude_isbns: Iterable[str] | None = None,
    ) -> list[Candidate]:
        """회원 취향으로 재정렬한 후보 목록.

        `/recommend`는 이 목록의 첫 1권만 응답하지만, 순위 전체가 필요할 때
        (평가·디버깅·top-k 확장) 같은 규칙을 그대로 쓸 수 있게 열어 둔다.

        `exclude_isbns`를 넘기지 않으면(`None`) 저장소에서 최근 추천 이력을 읽어
        제외한다. 빈 목록을 넘기면 제외하지 않는다.
        """
        preferences = await asyncio.to_thread(self._repository.get_member_preferences, member_id)
        if not preferences:
            return []
        excluded = (
            await self._excluded_isbns(member_id)
            if exclude_isbns is None
            else {str(isbn).strip() for isbn in exclude_isbns if str(isbn).strip()}
        )
        outcome = await self._rank(member_id, preferences, limit=limit, exclude_isbns=excluded)
        return outcome.candidates

    async def recommend_many(
        self,
        member_id: int | str,
        limit: int = 5,
        exclude_isbns: Iterable[str] | None = None,
    ) -> RecommendationList:
        """상위 k권 추천. MMR 다양성이 2순위 이하에서 실제로 드러나는 응답."""
        limit = max(1, min(int(limit), MAX_RECOMMEND_LIMIT))
        preferences = await asyncio.to_thread(self._repository.get_member_preferences, member_id)
        if not preferences:
            return RecommendationList(
                member_id=member_id, limit=limit, message=NO_PREFERENCES
            )

        excluded = (
            await self._excluded_isbns(member_id)
            if exclude_isbns is None
            else {str(isbn).strip() for isbn in exclude_isbns if str(isbn).strip()}
        )
        outcome = await self._rank(member_id, preferences, limit=limit, exclude_isbns=excluded)
        if not outcome.candidates:
            return RecommendationList(
                member_id=member_id,
                limit=limit,
                preferences=list(preferences),
                excluded_isbn_count=outcome.excluded_isbn_count,
                excluded_candidate_count=outcome.excluded_candidate_count,
                exclusion_relaxed=outcome.exclusion_relaxed,
                message=NO_BOOK,
            )

        items = await self._to_ranked_books(outcome.candidates, preferences)
        return RecommendationList(
            member_id=member_id,
            limit=limit,
            preferences=list(preferences),
            items=items,
            excluded_isbn_count=outcome.excluded_isbn_count,
            excluded_candidate_count=outcome.excluded_candidate_count,
            exclusion_relaxed=outcome.exclusion_relaxed,
        )

    async def rank_from_preferences(
        self,
        preferences: Sequence[str],
        limit: int | None = None,
        exclude_isbns: Iterable[str] | None = None,
    ) -> list[Candidate]:
        """선호 목록만으로 순위를 만든다(회원 조회 없이)."""
        outcome = await self._rank(
            None,
            preferences,
            limit=limit,
            exclude_isbns={str(i).strip() for i in (exclude_isbns or ()) if str(i).strip()},
        )
        return outcome.candidates

    # ------------------------------------------------------------------ 내부
    async def _rank(
        self,
        member_id: int | str | None,
        preferences: Sequence[str],
        limit: int | None,
        exclude_isbns: Iterable[str],
    ) -> RankingOutcome:
        embedding = await self._embedder.embed_query(" ".join(preferences))
        records = await self._store.find_candidates(embedding, n_results=self._weights.pool_size)
        candidates = build_candidates(records, query_embedding=embedding)
        if records and not candidates:
            logger.warning("벡터 검색 결과에 isbn 메타데이터가 없습니다: %d건", len(records))
        if not candidates:
            return RankingOutcome(candidates=[])

        excluded = {str(isbn).strip() for isbn in exclude_isbns if str(isbn).strip()}
        if not excluded:
            return RankingOutcome(
                candidates=mmr_rank(candidates, preferences, self._weights, limit=limit)
            )

        kept = exclude_candidates(candidates, excluded)
        removed = len(candidates) - len(kept)
        if not kept:
            # 후보가 전부 제외됨 → 제외를 풀고 같은 규칙으로 순위를 만든다(정책은 모듈 docstring).
            logger.warning(
                "후보 %d권이 모두 제외 목록(%d권)에 있어 제외를 풀고 추천합니다 (member=%s)",
                len(candidates),
                len(excluded),
                member_id,
            )
            return RankingOutcome(
                candidates=mmr_rank(candidates, preferences, self._weights, limit=limit),
                excluded_isbn_count=len(excluded),
                excluded_candidate_count=removed,
                exclusion_relaxed=True,
            )
        return RankingOutcome(
            candidates=mmr_rank(kept, preferences, self._weights, limit=limit),
            excluded_isbn_count=len(excluded),
            excluded_candidate_count=removed,
        )

    async def _excluded_isbns(self, member_id: int | str) -> set[str]:
        """이미 읽었거나 최근 추천한 ISBN. 읽지 못하면 빈 집합(추천은 계속)."""
        if self._exclude_recent_days <= 0:
            return set()
        getter = getattr(self._repository, "get_recent_recommended_isbns", None)
        if not callable(getter):
            return set()
        try:
            isbns = await asyncio.to_thread(getter, member_id, self._exclude_recent_days)
        except AppError:
            raise
        except Exception as exc:  # 스키마/권한 문제로 제외 목록만 못 읽는 경우
            logger.warning("최근 추천 이력을 읽지 못해 제외 없이 진행합니다: %s", exc)
            return set()
        return {str(isbn).strip() for isbn in (isbns or []) if str(isbn).strip()}

    async def _to_ranked_books(
        self, candidates: Sequence[Candidate], preferences: Sequence[str]
    ) -> list[RankedBook]:
        isbns = [candidate.isbn for candidate in candidates]
        book_ids = await self._book_ids(isbns)
        items: list[RankedBook] = []
        for rank, candidate in enumerate(candidates, start=1):
            metadata = candidate.metadata or {}
            items.append(
                RankedBook(
                    rank=rank,
                    isbn=candidate.isbn,
                    title=candidate.title,
                    sub_categories=sorted(candidate.sub_categories),
                    main_category=(str(metadata.get("mainCategory")) if metadata.get("mainCategory") else None),
                    author=(str(metadata.get("author")) if metadata.get("author") else None),
                    similarity=round(candidate.similarity, 6),
                    category_overlap=round(
                        category_overlap(preferences, candidate.sub_categories), 6
                    ),
                    score=round(relevance_score(candidate, preferences, self._weights), 6),
                    book_id=book_ids.get(candidate.isbn),
                )
            )
        return items

    async def _book_ids(self, isbns: Sequence[str]) -> dict[str, int | None]:
        """ISBN → book_id. 한 번에 조회하고, 저장소가 지원하지 않으면 한 권씩 조회한다."""
        if not isbns:
            return {}
        batch = getattr(self._repository, "get_book_ids_by_isbns", None)
        if callable(batch):
            try:
                mapping = await asyncio.to_thread(batch, list(isbns)) or {}
            except AppError:
                raise
            except Exception as exc:
                logger.warning("book_id 일괄 조회에 실패해 한 권씩 조회합니다: %s", exc)
            else:
                # MySQL에 없는 ISBN은 book_id가 None으로 남는다(응답은 계속 만든다).
                return {isbn: mapping.get(isbn) for isbn in isbns}
        result: dict[str, int | None] = {}
        for isbn in isbns:
            result[isbn] = await asyncio.to_thread(self._repository.get_book_id_by_isbn, isbn)
        return result
