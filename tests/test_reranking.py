"""재정렬(`app/services/reranking.py`) 단위 테스트.

가짜 벡터 스토어 없이, 후보 레코드와 고정 수치만으로 계산을 검증한다.
"""

from __future__ import annotations

import pytest

from app.services.reranking import (
    Candidate,
    RankingWeights,
    build_candidates,
    category_overlap,
    cosine_similarity,
    exclude_candidates,
    jaccard,
    mmr_rank,
    relevance_score,
    similarity_from_l2_distance,
    split_categories,
)


def _candidate(isbn: str, similarity: float, subs=("추리/스릴러",)) -> Candidate:
    return Candidate(
        isbn=isbn,
        title=f"book-{isbn}",
        sub_categories=frozenset(subs),
        similarity=similarity,
        metadata={"isbn": isbn},
    )


# ------------------------------------------------------------------ 기본 계산
class TestSimilarity:
    def test_cosine_of_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)

    def test_cosine_of_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_ignores_scale(self):
        assert cosine_similarity([1.0, 0.0], [3.0, 0.0]) == pytest.approx(1.0)

    def test_cosine_of_zero_vector_is_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_cosine_with_different_dimensions_is_zero(self):
        assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_l2_distance_is_converted_to_cosine(self):
        # 단위 벡터에서 d² = 2 - 2cos → d²=0.2 이면 cos=0.9
        assert similarity_from_l2_distance(0.2) == pytest.approx(0.9)
        assert similarity_from_l2_distance(0.0) == pytest.approx(1.0)

    def test_l2_distance_is_clipped_to_valid_range(self):
        assert similarity_from_l2_distance(4.0) == 0.0
        assert similarity_from_l2_distance(None) == 0.0


class TestCategories:
    def test_split_csv_ignores_blanks(self):
        assert split_categories(" 추리/스릴러 ,, SF ") == frozenset({"추리/스릴러", "SF"})

    def test_split_accepts_missing_value(self):
        assert split_categories(None) == frozenset()

    def test_overlap_is_share_of_preferences(self):
        assert category_overlap(["추리/스릴러", "SF"], {"SF"}) == pytest.approx(0.5)
        assert category_overlap(["추리/스릴러"], {"SF"}) == 0.0

    def test_overlap_without_preferences_is_zero(self):
        assert category_overlap([], {"SF"}) == 0.0

    def test_jaccard(self):
        assert jaccard({"x", "y"}, {"y", "z"}) == pytest.approx(1 / 3)
        assert jaccard({"x"}, set()) == 0.0


# ------------------------------------------------------------------ 하이브리드
class TestHybridRelevance:
    def test_category_match_adds_weight(self):
        weights = RankingWeights(category_weight=0.15)
        matched = _candidate("a", 0.90, subs=("추리/스릴러",))
        unmatched = _candidate("b", 0.94, subs=("SF",))
        assert relevance_score(matched, ["추리/스릴러"], weights) == pytest.approx(1.05)
        assert relevance_score(unmatched, ["추리/스릴러"], weights) == pytest.approx(0.94)

    def test_partial_match_is_proportional(self):
        weights = RankingWeights(category_weight=0.20)
        candidate = _candidate("a", 0.80, subs=("SF",))
        assert relevance_score(candidate, ["SF", "천문학"], weights) == pytest.approx(0.90)


# ------------------------------------------------------------------- 다양성
class TestMmrRanking:
    def test_first_pick_is_highest_relevance(self):
        candidates = [_candidate("a", 0.90), _candidate("b", 0.95)]
        ranked = mmr_rank(candidates, ["추리/스릴러"], RankingWeights(), limit=1)
        assert [c.isbn for c in ranked] == ["b"]

    def test_same_category_duplicate_is_pushed_down(self):
        # 관련도: a 1.10, b 1.08, c 0.92
        # 2순위 계산: b = 0.85*1.08 - 0.15*1.0 = 0.768 / c = 0.85*0.92 - 0 = 0.782
        candidates = [
            _candidate("a", 0.95),
            _candidate("b", 0.93),
            _candidate("c", 0.92, subs=("SF",)),
        ]
        weights = RankingWeights()
        ranked = mmr_rank(candidates, ["추리/스릴러"], weights, limit=3)
        assert [c.isbn for c in ranked] == ["a", "c", "b"]

    def test_lambda_one_keeps_pure_relevance_order(self):
        candidates = [
            _candidate("a", 0.95),
            _candidate("b", 0.93),
            _candidate("c", 0.92, subs=("SF",)),
        ]
        ranked = mmr_rank(candidates, ["추리/스릴러"], RankingWeights(mmr_lambda=1.0))
        assert [c.isbn for c in ranked] == ["a", "b", "c"]

    def test_categories_are_not_penalized_against_unrelated_books(self):
        # 서로 다른 카테고리끼리는 페널티가 없어 관련도 순서가 유지된다
        candidates = [
            _candidate("a", 0.95, subs=("SF",)),
            _candidate("b", 0.93, subs=("역사",)),
            _candidate("c", 0.92, subs=("시/에세이",)),
        ]
        ranked = mmr_rank(candidates, ["SF"], RankingWeights(), limit=3)
        assert [c.isbn for c in ranked] == ["a", "b", "c"]

    def test_limit_bounds(self):
        candidates = [_candidate("a", 0.9), _candidate("b", 0.8)]
        assert mmr_rank(candidates, [], RankingWeights(), limit=0) == []
        assert len(mmr_rank(candidates, [], RankingWeights(), limit=5)) == 2
        assert len(mmr_rank(candidates, [], RankingWeights())) == 2

    def test_tie_keeps_vector_order(self):
        candidates = [_candidate("a", 0.9), _candidate("b", 0.9)]
        ranked = mmr_rank(candidates, ["추리/스릴러"], RankingWeights())
        assert [c.isbn for c in ranked] == ["a", "b"]


# ------------------------------------------------------------------- 레코드
class TestBuildCandidates:
    def test_uses_embedding_cosine_when_available(self):
        candidates = build_candidates(
            [
                {
                    "metadata": {"isbn": "a", "title": "A", "subCategory": "SF, 역사"},
                    "distance": 0.99,
                    "embedding": [1.0, 0.0],
                }
            ],
            query_embedding=[1.0, 0.0],
        )
        assert candidates[0].similarity == pytest.approx(1.0)
        assert candidates[0].sub_categories == frozenset({"SF", "역사"})

    def test_falls_back_to_distance_without_embedding(self):
        candidates = build_candidates(
            [{"metadata": {"isbn": "a"}, "distance": 0.2, "embedding": None}]
        )
        assert candidates[0].similarity == pytest.approx(0.9)

    def test_records_without_isbn_are_skipped(self):
        candidates = build_candidates(
            [
                {"metadata": {"title": "isbn 없음"}, "distance": 0.1},
                {"metadata": {"isbn": "b"}, "distance": 0.1},
            ]
        )
        assert [c.isbn for c in candidates] == ["b"]


class TestExcludeCandidates:
    def test_empty_exclusion_keeps_input_order(self):
        candidates = [_candidate("a", 0.9), _candidate("b", 0.8)]
        assert [c.isbn for c in exclude_candidates(candidates, [])] == ["a", "b"]
        assert [c.isbn for c in exclude_candidates(candidates, None)] == ["a", "b"]

    def test_excluded_isbns_are_removed(self):
        candidates = [_candidate("a", 0.9), _candidate("b", 0.8), _candidate("c", 0.7)]
        assert [c.isbn for c in exclude_candidates(candidates, {"a", "c"})] == ["b"]

    def test_blank_and_non_string_entries_are_ignored(self):
        candidates = [_candidate("a", 0.9)]
        assert len(exclude_candidates(candidates, ["", "   "])) == 1

    def test_returns_new_list(self):
        candidates = [_candidate("a", 0.9)]
        result = exclude_candidates(candidates, [])
        assert result is not candidates


class TestRankingWeights:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"category_weight": -0.1},
            {"category_weight": 1.1},
            {"mmr_lambda": 0.0},
            {"mmr_lambda": 1.5},
            {"pool_size": 0},
        ],
    )
    def test_rejects_out_of_range(self, kwargs):
        with pytest.raises(ValueError):
            RankingWeights(**kwargs)

    def test_diversity_weight_is_complement_of_lambda(self):
        assert RankingWeights(mmr_lambda=0.8).diversity_weight == pytest.approx(0.2)
