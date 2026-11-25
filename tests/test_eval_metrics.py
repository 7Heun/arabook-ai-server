"""추천 평가 지표 단위 테스트.

실제 라벨/임베딩 없이도 지표 구현이 맞는지 손으로 계산한 값으로 확인한다.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics import (
    aggregate,
    category_entropy,
    coverage,
    distinct_category_count,
    evaluate_user,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecall:
    def test_counts_hits_over_number_of_relevant(self):
        ranked = ["a", "b", "c"]
        relevant = {"b", "c", "d"}
        assert recall_at_k(ranked, relevant, 1) == 0.0
        assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)
        assert recall_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)

    def test_returns_none_without_relevant_items(self):
        assert recall_at_k(["a"], set(), 3) is None

    def test_duplicate_recommendations_count_once(self):
        assert recall_at_k(["a", "a", "b"], {"b"}, 2) == 1.0

    def test_k_larger_than_list_is_allowed(self):
        assert recall_at_k(["a"], {"a"}, 10) == 1.0

    def test_rejects_non_positive_k(self):
        with pytest.raises(ValueError):
            recall_at_k(["a"], {"a"}, 0)


class TestPrecision:
    def test_divides_by_k(self):
        assert precision_at_k(["a", "b"], {"a"}, 4) == pytest.approx(0.25)

    def test_zero_when_no_hits(self):
        assert precision_at_k(["a"], {"b"}, 1) == 0.0

    def test_empty_relevant_still_defined(self):
        assert precision_at_k(["a"], set(), 2) == 0.0


class TestMRR:
    def test_first_relevant_rank(self):
        assert reciprocal_rank(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_zero_without_relevant(self):
        assert reciprocal_rank(["a"], {"b"}) == 0.0

    def test_duplicates_do_not_shift_rank(self):
        assert reciprocal_rank(["a", "a", "b"], {"b"}) == pytest.approx(0.5)


class TestNDCG:
    def test_matches_hand_computation(self):
        ranked = ["a", "b"]
        graded = {"a": 1.0, "b": 2.0}
        dcg = 1.0 / math.log2(2) + 2.0 / math.log2(3)
        ideal = 2.0 / math.log2(2) + 1.0 / math.log2(3)
        assert ndcg_at_k(ranked, graded, 2) == pytest.approx(dcg / ideal)

    def test_perfect_ordering_is_one(self):
        assert ndcg_at_k(["b", "a"], {"a": 1.0, "b": 2.0}, 2) == pytest.approx(1.0)

    def test_none_without_graded_labels(self):
        assert ndcg_at_k(["a"], {}, 3) is None

    def test_zero_when_nothing_relevant_returned(self):
        assert ndcg_at_k(["a"], {"b": 1.0}, 1) == 0.0


class TestCoverage:
    def test_union_over_catalog(self):
        catalog = {"a", "b", "c", "d"}
        assert coverage([["a", "b"], ["b", "c"]], catalog) == pytest.approx(0.75)

    def test_empty_catalog(self):
        assert coverage([["a"]], set()) == 0.0

    def test_ignores_isbns_outside_catalog(self):
        assert coverage([["a", "x"]], {"a", "b"}) == pytest.approx(0.5)


class TestDiversity:
    def test_distinct_categories(self):
        categories = {"a": ["x"], "b": ["y"], "c": ["x", "y"]}
        assert distinct_category_count(["a", "b"], categories) == 2
        assert distinct_category_count(["a", "c"], categories) == 2

    def test_entropy_is_zero_for_single_category(self):
        assert category_entropy(["a", "b"], {"a": ["x"], "b": ["x"]}, 2) == 0.0

    def test_entropy_is_one_for_uniform_distribution(self):
        assert category_entropy(["a", "b"], {"a": ["x"], "b": ["y"]}, 2) == pytest.approx(1.0)

    def test_entropy_stays_bounded_with_multilabel_books(self):
        # 한 책이 두 카테고리를 가지면 등장 카테고리 수가 k보다 많아질 수 있다.
        # 정규화를 k가 아니라 등장 카테고리 수로 하므로 값은 0~1을 벗어나지 않는다.
        categories = {"a": ["x", "y"], "b": ["z"], "c": ["z"]}
        value = category_entropy(["a", "b", "c"], categories, 3)
        assert 0.0 <= value <= 1.0

    def test_entropy_zero_when_k_is_one(self):
        assert category_entropy(["a", "b"], {"a": ["x"], "b": ["y"]}, 1) == 0.0


class TestEvaluateUser:
    def test_collects_all_metrics(self):
        result = evaluate_user(
            member_id="1",
            ranked=["a", "b", "c"],
            relevant={"c"},
            categories_by_isbn={"a": ["x"], "b": ["y"], "c": ["z"]},
            ks=[1, 3],
            graded_relevance={"c": 2.0},
        )
        assert result.n_relevant == 1
        assert result.recall[1] == 0.0
        assert result.recall[3] == 1.0
        assert result.precision[3] == pytest.approx(1 / 3)
        assert result.reciprocal_rank == pytest.approx(1 / 3)
        assert result.ndcg is not None
        assert result.distinct_categories[3] == 3
        assert result.entropy[3] == pytest.approx(1.0)

    def test_ndcg_none_without_graded_labels(self):
        result = evaluate_user("1", ["a"], {"a"}, {}, ks=[1])
        assert result.ndcg is None


class TestAggregate:
    def test_excludes_undefined_recall_from_mean(self):
        undefined = evaluate_user("1", ["a"], set(), {}, ks=[1])
        defined = evaluate_user("2", ["a"], {"a"}, {}, ks=[1])
        result = aggregate([undefined, defined], ks=[1])
        assert result["n_users"] == 2
        assert result["recall"][1]["value"] == 1.0
        assert result["recall"][1]["n"] == 1
        assert result["precision"][1]["n"] == 2

    def test_empty_results(self):
        result = aggregate([], ks=[1])
        assert result["n_users"] == 0
        assert result["recall"][1]["value"] is None
