"""오프라인 추천 평가 지표.

이 모듈은 외부 의존성이 없는 순수 함수 모음이다. 실제 데이터(라벨, 카탈로그)가
없어도 지표 구현 자체를 단위 테스트로 검증할 수 있게 하려는 의도이며,
`tests/test_eval_metrics.py`가 작은 예제로 각 지표의 값을 직접 확인한다.

표기
----
* ranked    : 추천기가 반환한 정렬된 ISBN 목록 (상위부터)
* relevant  : 정답(관련 있음)으로 라벨링된 ISBN 집합
* k         : 상위 몇 개까지 볼지

정의상의 선택
-------------
* Recall@k  = |ranked[:k] ∩ relevant| / |relevant|
  - 분모를 |relevant|로 두는 표준 정의를 따른다. 따라서 |relevant| > k 이면
    이론적 상한이 k / |relevant| 이며, k가 작을수록 값이 낮게 나온다.
* Precision@k = |ranked[:k] ∩ relevant| / k
  - k를 분모로 쓴다. 추천기가 k개보다 적게 반환하면 그만큼 불리해진다
    (카탈로그가 k보다 작은 경우는 리포트에서 별도로 표시한다).
* MRR = 1 / (첫 관련 항목의 순위). 관련 항목이 없으면 0.
* Coverage@k = |모든 사용자의 top-k 합집합| / |카탈로그|
* 다양성 = 사용자별 top-k의 서브카테고리 분포 엔트로피(H)를, 그 top-k에 실제로
  등장한 서브카테고리 수 S의 최대 엔트로피 log(S)로 나눈 값(균등도)의 평균.
  모든 항목이 같은 카테고리면 0, 등장한 카테고리에 고르게 퍼져 있으면 1.
  (k로 나누면 한 책이 여러 서브카테고리를 가질 때 1을 넘을 수 있어 S로 나눈다.)
* nDCG@k = DCG@k / IDCG@k, 등급(graded relevance)이 주어진 경우에만 계산한다.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

#: 리포트에서 기본으로 사용하는 k 값
DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class UserResult:
    """한 사용자에 대한 지표 값. 계산 불가능한 지표는 None."""

    member_id: str
    n_relevant: int
    n_returned: int
    recall: Mapping[int, float]
    precision: Mapping[int, float]
    reciprocal_rank: float
    ndcg: Mapping[int, float] | None
    distinct_categories: Mapping[int, int]
    entropy: Mapping[int, float]


def _dedupe(items: Iterable[str]) -> list[str]:
    """순서를 유지한 채 중복 ISBN을 제거한다 (중복 추천은 한 번만 인정)."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """Recall@k. 정답이 비어 있으면 정의되지 않으므로 None."""
    relevant_set = set(relevant)
    if not relevant_set:
        return None
    if k <= 0:
        raise ValueError("k must be positive")
    hits = len(set(_dedupe(ranked)[:k]) & relevant_set)
    return hits / len(relevant_set)


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Precision@k. 정답이 비어 있어도 0.0으로 정의된다."""
    if k <= 0:
        raise ValueError("k must be positive")
    relevant_set = set(relevant)
    hits = len(set(_dedupe(ranked)[:k]) & relevant_set)
    return hits / k


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    """첫 관련 항목의 역순위. 없으면 0.0."""
    relevant_set = set(relevant)
    for index, isbn in enumerate(_dedupe(ranked), start=1):
        if isbn in relevant_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(
    ranked: Sequence[str],
    graded_relevance: Mapping[str, float],
    k: int,
) -> float | None:
    """등급 라벨이 있을 때의 nDCG@k. 등급 정보가 없으면 None."""
    if not graded_relevance:
        return None
    if k <= 0:
        raise ValueError("k must be positive")

    def dcg(gains: Sequence[float]) -> float:
        return sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))

    actual = dcg([float(graded_relevance.get(isbn, 0.0)) for isbn in _dedupe(ranked)[:k]])
    ideal_gains = sorted((float(v) for v in graded_relevance.values() if v > 0), reverse=True)[:k]
    ideal = dcg(ideal_gains)
    if ideal == 0:
        return None
    return actual / ideal


def coverage(ranked_lists: Sequence[Sequence[str]], catalog_isbns: Iterable[str]) -> float:
    """카탈로그 커버리지: 모든 사용자의 top-k에 한 번이라도 노출된 책의 비율."""
    catalog = set(catalog_isbns)
    if not catalog:
        return 0.0
    exposed: set[str] = set()
    for ranked in ranked_lists:
        exposed.update(ranked)
    return len(exposed & catalog) / len(catalog)


def distinct_category_count(
    ranked: Sequence[str],
    categories_by_isbn: Mapping[str, Sequence[str]],
) -> int:
    """top-k 안에 등장하는 서브카테고리 종류 수."""
    categories: set[str] = set()
    for isbn in ranked:
        categories.update(categories_by_isbn.get(isbn, ()))
    return len(categories)


def category_entropy(
    ranked: Sequence[str],
    categories_by_isbn: Mapping[str, Sequence[str]],
    k: int,
) -> float:
    """top-k 서브카테고리 분포의 정규화 엔트로피 (0~1).

    한 책이 여러 서브카테고리를 가지면 각 카테고리에 1/n씩 나눠 센다. 정규화
    분모는 top-k에 등장한 카테고리 수 S의 최대 엔트로피 log(S)다. 카테고리가
    1종뿐이거나 k가 1이면 정의상 0.
    """
    top = _dedupe(ranked)[:k]
    if k <= 1 or not top:
        return 0.0
    weights: Counter[str] = Counter()
    for isbn in top:
        cats = [c for c in categories_by_isbn.get(isbn, ()) if c]
        if not cats:
            continue
        share = 1.0 / len(cats)
        for category in cats:
            weights[category] += share
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    support = len(weights)
    if support <= 1:
        return 0.0
    entropy = -sum((w / total) * math.log(w / total) for w in weights.values())
    return entropy / math.log(support)


def evaluate_user(
    member_id: str,
    ranked: Sequence[str],
    relevant: Iterable[str],
    categories_by_isbn: Mapping[str, Sequence[str]],
    ks: Sequence[int] = DEFAULT_KS,
    graded_relevance: Mapping[str, float] | None = None,
) -> UserResult:
    """한 사용자에 대해 모든 지표를 계산한다."""
    ranked = _dedupe(ranked)
    relevant_set = set(relevant)
    return UserResult(
        member_id=str(member_id),
        n_relevant=len(relevant_set),
        n_returned=len(ranked),
        recall={k: recall_at_k(ranked, relevant_set, k) for k in ks},
        precision={k: precision_at_k(ranked, relevant_set, k) for k in ks},
        reciprocal_rank=reciprocal_rank(ranked, relevant_set),
        ndcg=(
            {k: ndcg_at_k(ranked, graded_relevance, k) for k in ks}
            if graded_relevance
            else None
        ),
        distinct_categories={k: distinct_category_count(ranked[:k], categories_by_isbn) for k in ks},
        entropy={k: category_entropy(ranked, categories_by_isbn, k) for k in ks},
    )


def _mean(values: Sequence[float | None]) -> tuple[float | None, int]:
    """None을 제외한 평균과 사용된 표본 수를 함께 반환한다."""
    usable = [v for v in values if v is not None]
    if not usable:
        return None, 0
    return sum(usable) / len(usable), len(usable)


def aggregate(
    results: Sequence[UserResult],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, object]:
    """사용자별 결과를 평균 낸다.

    반환 구조::

        {
          "n_users": 12,
          "recall": {1: {"value": 0.5, "n": 8}, ...},
          "precision": {...},
          "mrr": {"value": ..., "n": ...},
          "ndcg": {...},
          "distinct_categories": {...},
          "entropy": {...},   # 사용자 평균 정규화 엔트로피
        }

    `n`은 해당 지표가 정의된 사용자 수다. Recall은 정답이 없는 사용자에서
    정의되지 않으므로 n이 전체 사용자 수보다 작을 수 있다.
    """
    aggregated: dict[str, object] = {"n_users": len(results)}
    aggregated["recall"] = {
        k: _pair([r.recall.get(k) for r in results]) for k in ks
    }
    aggregated["precision"] = {
        k: _pair([r.precision.get(k) for r in results]) for k in ks
    }
    aggregated["mrr"] = _pair([r.reciprocal_rank for r in results])
    aggregated["ndcg"] = {
        k: _pair([(r.ndcg or {}).get(k) for r in results]) for k in ks
    }
    aggregated["distinct_categories"] = {
        k: _pair([float(r.distinct_categories.get(k, 0)) for r in results]) for k in ks
    }
    aggregated["entropy"] = {
        k: _pair([r.entropy.get(k, 0.0) for r in results]) for k in ks
    }
    return aggregated


def _pair(values: Sequence[float | None]) -> dict[str, float | int | None]:
    mean, n = _mean(values)
    return {"value": mean, "n": n}


def format_metric(value: float | None, digits: int = 3) -> str:
    """리포트 표에 쓰는 문자열. 값이 없으면 '미산출'."""
    if value is None:
        return "미산출"
    return f"{value:.{digits}f}"
