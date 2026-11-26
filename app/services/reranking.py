"""추천 후보 재정렬 (하이브리드 점수 + 다양성).

배경
----
`scripts/catalog_diagnostics.py`로 실제 카탈로그를 재 보니, 책끼리의 최근접 코사인
유사도가 0.839~0.931에 몰려 있었다. 상위 후보 사이의 벡터 점수 차이가 작다는 뜻이라,
"가장 가까운 1권"만 고르면 선호 카테고리 신호가 거의 반영되지 않고 같은 군집의 책이
반복 선택되기 쉽다.

그래서 추천은 세 신호를 함께 쓴다.

1. 벡터 유사도 (코사인)
2. 선호 서브카테고리 일치율
   = |선호 ∩ 책의 subCategory| / |선호|
   (운영에서 선호는 `member_sub_category_selections`의 **서브카테고리 이름**이므로
   대분류가 아니라 서브카테고리로 비교한다.)
3. 다양성: MMR(maximal marginal relevance)
   = lambda * 관련도 - (1 - lambda) * max(이미 뽑힌 책과의 서브카테고리 Jaccard)
   이미 뽑힌 책과 카테고리가 겹치는 후보를 뒤로 미뤄 한 장르 쏠림을 줄인다.

왜 중복도로 임베딩 코사인을 쓰지 않는가
- 책끼리 코사인이 0.84~0.93 구간에 몰려 있어 (1 - lambda) * 0.9가 관련도 차이
  (0.01~0.05)를 덮어써 버린다. 서브카테고리 Jaccard(0, 1/3, 1/2, 1)는 이산적이라
  관련도와 스케일이 맞고, "같은 카테고리 중복 완화"라는 의미도 분명하다.

이 모듈은 numpy 없이 표준 라이브러리만 쓴다. 평가 하네스(`eval/`)가 같은 함수를
불러 운영과 동일한 순위를 재현할 수 있게 하기 위해서다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: 관련도에 더하는 카테고리 일치율 가중치. 평가셋으로 튜닝한 값이 아니라 사전 고정값이다.
DEFAULT_CATEGORY_WEIGHT = 0.15
#: MMR의 lambda. 1.0이면 다양성 항이 꺼지고(순수 하이브리드 점수), 0.5면 중복 페널티가
#: 관련도와 같은 무게가 된다. 0.85면 "완전히 같은 카테고리" 후보가 0.15만큼 뒤로 밀린다.
DEFAULT_MMR_LAMBDA = 0.85
#: 한 번에 가져와 재정렬할 후보 수. 적으면 다양성 항이 의미를 잃고, 크면 호출 비용이 는다.
DEFAULT_POOL_SIZE = 20


@dataclass(frozen=True)
class Candidate:
    """재정렬 대상 후보 1권."""

    isbn: str
    title: str
    sub_categories: frozenset[str]
    similarity: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.isbn:
            raise ValueError("isbn이 비어 있는 후보는 만들 수 없습니다.")


@dataclass(frozen=True)
class RankingWeights:
    """재정렬 설정. 값이 바뀌면 추천 결과가 바뀌므로 한 곳에서만 정의한다."""

    category_weight: float = DEFAULT_CATEGORY_WEIGHT
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    pool_size: int = DEFAULT_POOL_SIZE

    def __post_init__(self) -> None:
        if not 0.0 <= self.category_weight <= 1.0:
            raise ValueError(
                f"category_weight는 0~1이어야 합니다: {self.category_weight}"
            )
        if not 0.0 < self.mmr_lambda <= 1.0:
            raise ValueError(f"mmr_lambda는 0보다 크고 1 이하여야 합니다: {self.mmr_lambda}")
        if self.pool_size < 1:
            raise ValueError(f"pool_size는 1 이상이어야 합니다: {self.pool_size}")

    @property
    def diversity_weight(self) -> float:
        """중복 페널티의 무게 (1 - lambda)."""
        return 1.0 - self.mmr_lambda


def split_categories(value: Any) -> frozenset[str]:
    """`"추리/스릴러, SF"` 같은 쉼표 구분 문자열을 집합으로 바꾼다.

    Chroma 메타데이터는 리스트를 저장하지 못해 쉼표 문자열로 넣는다
    (`BookVectorStore.add_book`). 빈 값/공백은 버린다.
    """
    if not value:
        return frozenset()
    if isinstance(value, (list, tuple, set, frozenset)):
        parts: Iterable[str] = (str(part) for part in value)
    else:
        parts = str(value).split(",")
    return frozenset(part.strip() for part in parts if part and part.strip())


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """코사인 유사도. 0 벡터나 길이 불일치에는 0.0을 돌려준다(순위 계산을 멈추지 않는다)."""
    if len(left) != len(right) or not left:
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / ((left_norm**0.5) * (right_norm**0.5))


def similarity_from_l2_distance(distance: float) -> float:
    """Chroma `l2` 거리(제곱 유클리드)를 코사인 유사도로 바꾼다.

    저장된 임베딩(ada-002)은 단위 벡터이고(실측 노름 0.9999999~1.0000001),
    단위 벡터에서는 `d² = 2 - 2cos`이다. Chroma의 l2 space는 제곱 거리를 돌려주므로
    `cos = 1 - d²/2`. 임베딩을 함께 받아 오지 못한 경우의 대체 경로로만 쓴다.
    """
    if distance is None:
        return 0.0
    similarity = 1.0 - float(distance) / 2.0
    if similarity < 0.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return similarity


def category_overlap(preferences: Iterable[str], sub_categories: Iterable[str]) -> float:
    """선호 서브카테고리 중 책이 가진 비율(0~1). 선호가 없으면 0."""
    preferred = {str(piece).strip() for piece in preferences if str(piece).strip()}
    if not preferred:
        return 0.0
    owned = {str(piece).strip() for piece in sub_categories if str(piece).strip()}
    return len(preferred & owned) / len(preferred)


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    """두 카테고리 집합의 Jaccard 유사도. 둘 다 비어 있으면 0."""
    left_set = {piece for piece in left if piece}
    right_set = {piece for piece in right if piece}
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def build_candidates(
    records: Sequence[Mapping[str, Any]],
    query_embedding: Sequence[float] | None = None,
) -> list[Candidate]:
    """벡터 검색 레코드를 후보로 바꾼다.

    레코드 형태(운영: `BookVectorStore.find_candidates` 결과)::

        {"metadata": {...}, "distance": 0.12, "embedding": [...]}

    유사도는 가능하면 질의/도서 임베딩으로 직접 계산한 코사인을 쓰고, 임베딩이 없으면
    L2 거리를 코사인으로 환산한다. isbn이 없는 레코드는 순위를 매길 수 없어 버린다
    (서비스가 건수를 로그로 남긴다).
    """
    candidates: list[Candidate] = []
    for record in records:
        metadata = record.get("metadata") or {}
        isbn = str(metadata.get("isbn") or "").strip()
        if not isbn:
            continue
        embedding = record.get("embedding")
        if query_embedding is not None and embedding:
            similarity = cosine_similarity(query_embedding, list(embedding))
        else:
            similarity = similarity_from_l2_distance(record.get("distance"))
        candidates.append(
            Candidate(
                isbn=isbn,
                title=str(metadata.get("title") or ""),
                sub_categories=split_categories(metadata.get("subCategory")),
                similarity=similarity,
                metadata=metadata,
            )
        )
    return candidates


def relevance_score(
    candidate: Candidate,
    preferences: Iterable[str],
    weights: RankingWeights,
) -> float:
    """하이브리드 관련도 = 벡터 유사도 + 가중치 × 카테고리 일치율."""
    return candidate.similarity + weights.category_weight * category_overlap(
        preferences, candidate.sub_categories
    )


def mmr_rank(
    candidates: Sequence[Candidate],
    preferences: Iterable[str],
    weights: RankingWeights,
    limit: int | None = None,
) -> list[Candidate]:
    """MMR로 후보를 재정렬한다.

    첫 순위는 관련도가 가장 높은 후보(중복 페널티가 붙을 대상이 없으므로)이고,
    두 번째부터는 "관련도가 높으면서 이미 뽑힌 책과 카테고리가 덜 겹치는" 후보가 올라온다.
    동점이면 입력 순서(= 벡터 거리 순)를 그대로 지켜 결과가 결정적이다.
    """
    remaining = list(candidates)
    relevance = [relevance_score(candidate, preferences, weights) for candidate in remaining]
    selected: list[Candidate] = []
    target = len(remaining) if limit is None else max(0, min(limit, len(remaining)))

    while remaining and len(selected) < target:
        best_index = 0
        best_score = float("-inf")
        for index, candidate in enumerate(remaining):
            redundancy = max(
                (
                    jaccard(candidate.sub_categories, chosen.sub_categories)
                    for chosen in selected
                ),
                default=0.0,
            )
            score = weights.mmr_lambda * relevance[index] - weights.diversity_weight * redundancy
            if score > best_score:
                best_index = index
                best_score = score
        selected.append(remaining.pop(best_index))
        relevance.pop(best_index)

    return selected
