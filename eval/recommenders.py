"""평가에 쓰는 추천 변형(variant)들.

모든 변형은 같은 입력만 본다: 사용자가 고른 서브카테고리 목록(=운영에서
`member_sub_category_selections`로 들어오는 값)과 카탈로그. 정답 라벨은
순위를 만드는 데 쓰지 않는다(라벨 누설 방지).

0) `baseline_vector` — 현재 운영 로직. 선호 서브카테고리 문자열을 임베딩해
   Chroma에서 가장 가까운 책 1권을 고른다. 여기서는 재현을 위해 **L2 거리**
   오름차순으로 전체 카탈로그를 정렬한다(Chroma 컬렉션 기본 space가 l2).
1) `category_filter` — 후보를 선호 서브카테고리와 겹치는 책으로 제한한 뒤
   같은 L2 거리로 정렬. 겹치는 책이 없으면 전체 카탈로그로 되돌아간다.
2) `hybrid_category_boost` — 코사인 유사도에 카테고리 일치율을 가중치로 더한다.
   alpha는 평가셋을 보고 고른 값이 아니라 사전에 정한 값이다(과적합 방지).
   실제 튜닝은 별도 dev split에서 해야 한다는 점을 README에 적어 두었다.

주의: 1)과 2)는 "개선안"이지 검증된 개선이 아니다. 라벨 데이터로 측정하기
전까지는 어느 쪽이 나은지 알 수 없다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from eval.catalog import Catalog, CatalogError


@dataclass(frozen=True)
class QueryContext:
    member_id: str
    preferences: tuple[str, ...]
    query_text: str
    query_embedding: list[float] | None


def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1) * max(float(np.linalg.norm(query)), 1e-12)
    return (matrix @ query) / np.maximum(norms, 1e-12)


class VectorIndex:
    """카탈로그 임베딩을 한 번만 올려 두고 재사용한다."""

    def __init__(self, catalog: Catalog):
        if not catalog.has_embeddings:
            raise CatalogError(
                "카탈로그 임베딩이 없어 벡터 변형을 실행할 수 없습니다. "
                "chroma_db에서 읽거나 임베딩이 포함된 카탈로그 JSON을 쓰세요."
            )
        self.catalog = catalog
        self.isbns = catalog.isbns
        self.matrix = catalog.embedding_matrix()

    def l2_scores(self, query: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.matrix - query, axis=1)

    def cosine_scores(self, query: np.ndarray) -> np.ndarray:
        return _cosine_scores(self.matrix, query)

    def category_overlap(self, preferences: Sequence[str]) -> np.ndarray:
        pref = set(preferences)
        if not pref:
            return np.zeros(len(self.isbns), dtype="float32")
        return np.asarray(
            [
                len(pref & set(item.sub_categories)) / len(pref)
                for item in self.catalog.items
            ],
            dtype="float32",
        )

    def top_indices(self, scores: np.ndarray, limit: int, ascending: bool) -> list[int]:
        order = np.argsort(scores, kind="stable") if ascending else np.argsort(-scores, kind="stable")
        return [int(i) for i in order[:limit]]

    def as_isbns(self, indices: Sequence[int]) -> list[str]:
        return [self.isbns[i] for i in indices]


class Recommender:
    key: str = ""
    label: str = ""

    def __init__(self, index: VectorIndex):
        self.index = index

    def rank(self, context: QueryContext, limit: int) -> list[str]:
        raise NotImplementedError

    @property
    def notes(self) -> str:
        return ""


class BaselineVectorRecommender(Recommender):
    """현재 운영 추천 로직의 재현 (Chroma l2 거리 기준 최근접)."""

    key = "baseline_vector"
    label = "baseline (현재 운영: 벡터 최근접, L2)"

    def rank(self, context: QueryContext, limit: int) -> list[str]:
        if context.query_embedding is None:
            raise CatalogError("query embedding이 없습니다.")
        scores = self.index.l2_scores(np.asarray(context.query_embedding, dtype="float32"))
        return self.index.as_isbns(self.index.top_indices(scores, limit, ascending=True))


class CategoryFilterRecommender(Recommender):
    """선호 카테고리와 겹치는 책만 후보로 두고 벡터 유사도로 정렬."""

    key = "category_filter"
    label = "variant 1 (카테고리 필터 + 벡터, 후보 없으면 전체)"

    def rank(self, context: QueryContext, limit: int) -> list[str]:
        if context.query_embedding is None:
            raise CatalogError("query embedding이 없습니다.")
        query = np.asarray(context.query_embedding, dtype="float32")
        overlap = self.index.category_overlap(context.preferences)
        candidates = np.flatnonzero(overlap > 0)
        if candidates.size == 0:
            return BaselineVectorRecommender(self.index).rank(context, limit)
        scores = self.index.l2_scores(query)
        masked = np.full(len(self.index.isbns), np.inf, dtype="float32")
        masked[candidates] = scores[candidates]
        order = [int(i) for i in np.argsort(masked, kind="stable") if np.isfinite(masked[i])]
        return self.index.as_isbns(order[:limit])


class HybridCategoryBoostRecommender(Recommender):
    """코사인 유사도 + alpha * 카테고리 일치율."""

    key = "hybrid_category_boost"
    label = "variant 2 (코사인 + 카테고리 가중)"

    def __init__(self, index: VectorIndex, alpha: float = 0.15):
        super().__init__(index)
        self.alpha = alpha

    @property
    def notes(self) -> str:
        return f"alpha={self.alpha} (사전 고정값, 평가셋으로 튜닝하지 않음)"

    def rank(self, context: QueryContext, limit: int) -> list[str]:
        if context.query_embedding is None:
            raise CatalogError("query embedding이 없습니다.")
        query = np.asarray(context.query_embedding, dtype="float32")
        cosine = self.index.cosine_scores(query)
        overlap = self.index.category_overlap(context.preferences)
        scores = cosine + self.alpha * overlap
        return self.index.as_isbns(self.index.top_indices(scores, limit, ascending=False))


VARIANT_REGISTRY = {
    BaselineVectorRecommender.key: BaselineVectorRecommender,
    CategoryFilterRecommender.key: CategoryFilterRecommender,
    HybridCategoryBoostRecommender.key: HybridCategoryBoostRecommender,
}


def build_recommenders(index: VectorIndex, keys: Sequence[str], alpha: float = 0.15) -> list[Recommender]:
    recommenders: list[Recommender] = []
    for key in keys:
        if key not in VARIANT_REGISTRY:
            raise ValueError(f"알 수 없는 variant: {key} (가능: {', '.join(VARIANT_REGISTRY)})")
        if key == HybridCategoryBoostRecommender.key:
            recommenders.append(HybridCategoryBoostRecommender(index, alpha=alpha))
        else:
            recommenders.append(VARIANT_REGISTRY[key](index))
    return recommenders
