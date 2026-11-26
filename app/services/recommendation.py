"""도서 추천.

기존에는 `ChromaManager.recommend_book`이 Chroma 검색 + MySQL 조회/저장을 한
클래스에서 처리했다. 여기서는 벡터 검색(`BookVectorStore`)과 DB(`BookRepository`)를
분리하고 이 서비스가 둘을 조합한다.

추천은 "가장 가까운 1권"이 아니라 **후보 풀을 재정렬한 뒤 1권**을 고른다.
벡터 유사도만 쓰면 진단에서 확인한 것처럼(책끼리 코사인 0.84~0.93) 상위 후보의
점수 차이가 작아 선호 카테고리가 거의 반영되지 않는다. 재정렬 규칙은
`app.services.reranking`에 따로 두어, 평가 하네스도 같은 함수를 쓴다.

반환 메시지 문자열은 클라이언트(Spring 서버)가 그대로 보던 값이라 유지했다.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Sequence

from app.services.database import BookRepository
from app.services.llm import EmbeddingService
from app.services.reranking import Candidate, RankingWeights, build_candidates, mmr_rank
from app.services.vector_store import BookVectorStore

logger = logging.getLogger("arabook.recommendation")

NO_PREFERENCES = "No preferences found for the member."
NO_BOOK = "No book found for the given preferences."
NO_BOOK_ID = "No book ID found for the given ISBN."


class RecommendationService:
    def __init__(
        self,
        store: BookVectorStore,
        repository: BookRepository,
        embedder: EmbeddingService,
        weights: RankingWeights | None = None,
    ):
        self._store = store
        self._repository = repository
        self._embedder = embedder
        self._weights = weights or RankingWeights()

    @property
    def weights(self) -> RankingWeights:
        return self._weights

    async def recommend(self, member_id: int | str) -> str:
        preferences = await asyncio.to_thread(self._repository.get_member_preferences, member_id)
        if not preferences:
            return NO_PREFERENCES

        ranking = await self.rank_from_preferences(preferences, limit=1)
        if not ranking:
            return NO_BOOK

        isbn = ranking[0].isbn
        book_id = await asyncio.to_thread(self._repository.get_book_id_by_isbn, isbn)
        if not book_id:
            return NO_BOOK_ID

        # 기존 동작 유지: '내일' 날짜로 추천 이력을 남긴다.
        recommendation_date = (datetime.now() + timedelta(days=1)).date()
        await asyncio.to_thread(
            self._repository.save_recommendation, member_id, book_id, recommendation_date
        )
        return f"Recommended book ID: {book_id}"

    async def rank(self, member_id: int | str, limit: int | None = None) -> list[Candidate]:
        """회원 취향으로 재정렬한 후보 목록.

        `/recommend`는 이 목록의 첫 1권만 응답하지만, 순위 전체가 필요할 때
        (평가·디버깅·top-k 확장) 같은 규칙을 그대로 쓸 수 있게 열어 둔다.
        """
        preferences = await asyncio.to_thread(self._repository.get_member_preferences, member_id)
        if not preferences:
            return []
        return await self.rank_from_preferences(preferences, limit=limit)

    async def rank_from_preferences(
        self, preferences: Sequence[str], limit: int | None = None
    ) -> list[Candidate]:
        embedding = await self._embedder.embed_query(" ".join(preferences))
        records = await self._store.find_candidates(embedding, n_results=self._weights.pool_size)
        candidates = build_candidates(records, query_embedding=embedding)
        if records and not candidates:
            logger.warning("벡터 검색 결과에 isbn 메타데이터가 없습니다: %d건", len(records))
        if not candidates:
            return []
        return mmr_rank(candidates, preferences, self._weights, limit=limit)
