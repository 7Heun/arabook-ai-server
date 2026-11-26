"""다양성 실측 스크립트(`scripts/diversity_benchmark.py`) 테스트.

실제 `chroma_db` 없이 손으로 만든 임베딩으로 계산 경로를 확인한다.
수치의 타당성을 주장하는 테스트가 아니라, 질의 구성·후보 제외·지표 연결·
리포트 문구(정확도 아님 고지)가 깨지지 않는지 보는 회귀 테스트다.
"""

from __future__ import annotations

import math

import pytest

from eval.catalog import Catalog, CatalogItem
from eval.metrics import category_entropy, distinct_category_count
from scripts.diversity_benchmark import (
    QueryCase,
    VariantOutcome,
    _subset,
    build_queries,
    compare_variants,
    file_md5,
    render_markdown,
    run_variant,
)


def _item(isbn: str, title: str, subs: tuple[str, ...], embedding: tuple[float, ...]):
    return CatalogItem(
        isbn=isbn,
        title=title,
        author="저자",
        main_category="소설",
        sub_categories=subs,
        hashtags=(),
        description="설명",
        embedding=embedding,
    )


def _uniform_catalog() -> Catalog:
    """3권이 모두 같은 서브카테고리 → 다양성 지표가 정의상 최소."""
    return Catalog(
        items=[
            _item("a", "A", ("하나",), (1.0, 0.0)),
            _item("b", "B", ("하나",), (0.9, 0.1)),
            _item("c", "C", ("하나",), (0.9, 0.2)),
        ],
        source="test",
        dimension=2,
    )


def _two_category_catalog() -> Catalog:
    """2개 카테고리 × 2권. 카테고리 간 유사도도 높게(실측 0.84~0.93과 비슷한 압축)."""
    return Catalog(
        items=[
            _item("a1", "A1", ("가",), (1.0, 0.30, 0.00)),
            _item("a2", "A2", ("가",), (1.0, 0.32, 0.02)),
            _item("b1", "B1", ("나",), (1.0, 0.02, 0.30)),
            _item("b2", "B2", ("나",), (1.0, 0.00, 0.32)),
        ],
        source="test",
        dimension=3,
    )


class TestBuildQueries:
    def test_book_mode_excludes_only_its_own_book(self):
        catalog = _two_category_catalog()
        queries = build_queries(catalog, "book")
        assert len(queries) == 4
        first = queries[0]
        assert first.name == "book:a1"
        assert first.preferences == ("가",)
        assert first.allowed_isbns == frozenset({"a2", "b1", "b2"})
        assert first.embedding == pytest.approx((1.0, 0.30, 0.00))

    def test_category_mode_uses_unit_centroid(self):
        catalog = _two_category_catalog()
        queries = build_queries(catalog, "category", min_books=2)
        assert [q.preferences for q in queries] == [("가",), ("나",)]
        for query in queries:
            norm = math.sqrt(sum(value * value for value in query.embedding))
            assert norm == pytest.approx(1.0)
            assert query.allowed_isbns is None  # 카테고리 질의는 후보를 줄이지 않는다

    def test_category_mode_skips_small_labels(self):
        catalog = _uniform_catalog()
        assert len(build_queries(catalog, "category", min_books=2)) == 1
        assert build_queries(catalog, "category", min_books=4) == []

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(SystemExit):
            build_queries(_uniform_catalog(), "nope")

    def test_catalog_without_embeddings_is_rejected(self):
        catalog = Catalog(
            items=[CatalogItem("a", "A", "", "", (), (), "", None)],
            source="test",
        )
        with pytest.raises(SystemExit):
            build_queries(catalog, "book")


class TestRunVariant:
    def test_baseline_never_returns_the_query_book(self):
        catalog = _two_category_catalog()
        queries = build_queries(catalog, "book")
        outcome = run_variant(catalog, queries, "baseline_vector", (1, 3))
        for query in queries:
            assert query.name.split(":")[1] not in outcome.rankings[query.name]
            assert len(outcome.rankings[query.name]) == 3

    def test_hybrid_returns_ranked_isbns_within_pool(self):
        catalog = _two_category_catalog()
        queries = build_queries(catalog, "book")
        outcome = run_variant(catalog, queries, "production_hybrid", (1, 3))
        known = set(catalog.isbns)
        for query in queries:
            ranked = outcome.rankings[query.name]
            assert len(ranked) == 3
            assert ranked[0] != query.name.split(":")[1]
            assert set(ranked) <= known
            assert len(set(ranked)) == len(ranked)  # 중복 없음

    def test_metrics_match_metric_module_on_uniform_catalog(self):
        # 모든 책이 같은 카테고리 1종 → 고유 카테고리 1, 엔트로피 0.
        # 질의 3건 × top-2의 합집합은 3권 전부를 덮으므로 커버리지 1.0.
        catalog = _uniform_catalog()
        queries = build_queries(catalog, "book")
        for variant in ("baseline_vector", "production_hybrid"):
            outcome = run_variant(catalog, queries, variant, (2,))
            assert outcome.distinct[2] == pytest.approx(1.0)
            assert outcome.entropy[2] == pytest.approx(0.0)
            assert outcome.coverage[2] == pytest.approx(1.0)

    def test_coverage_is_never_above_one(self):
        catalog = _two_category_catalog()
        queries = build_queries(catalog, "category", min_books=2)
        outcome = run_variant(catalog, queries, "baseline_vector", (1, 3, 5))
        assert all(0.0 <= value <= 1.0 for value in outcome.coverage.values())

    def test_is_deterministic(self):
        catalog = _two_category_catalog()
        queries = build_queries(catalog, "book")
        first = run_variant(catalog, queries, "production_hybrid", (1, 3))
        second = run_variant(catalog, queries, "production_hybrid", (1, 3))
        assert first.rankings == second.rankings


class TestCompareVariants:
    def _outcome(self, key: str, rankings: dict[str, list[str]]) -> VariantOutcome:
        return VariantOutcome(
            key=key, label=key, rankings=rankings, distinct={}, entropy={}, coverage={}
        )

    def test_identical_rankings_overlap_completely(self):
        rankings = {"q1": ["a", "b"], "q2": ["c", "d"]}
        outcomes = {
            "baseline_vector": self._outcome("baseline_vector", rankings),
            "production_hybrid": self._outcome("production_hybrid", dict(rankings)),
        }
        queries = [QueryCase("q1", (), (1.0,)), QueryCase("q2", (), (1.0,))]
        result = compare_variants(queries, outcomes, k=2)
        assert result["same_top1_share"] == 1.0
        assert result["mean_topk_jaccard"] == 1.0

    def test_disjoint_rankings_overlap_zero(self):
        outcomes = {
            "baseline_vector": self._outcome("baseline_vector", {"q1": ["a", "b"]}),
            "production_hybrid": self._outcome("production_hybrid", {"q1": ["c", "d"]}),
        }
        result = compare_variants([QueryCase("q1", (), (1.0,))], outcomes, k=2)
        assert result["same_top1_share"] == 0.0
        assert result["mean_topk_jaccard"] == 0.0

    def test_partial_overlap(self):
        # top-1은 같고 top-2가 다름 → top-1 일치 1.0, Jaccard = 1/3
        outcomes = {
            "baseline_vector": self._outcome("baseline_vector", {"q1": ["a", "b"]}),
            "production_hybrid": self._outcome("production_hybrid", {"q1": ["a", "c"]}),
        }
        result = compare_variants([QueryCase("q1", (), (1.0,))], outcomes, k=2)
        assert result["same_top1_share"] == 1.0
        assert result["mean_topk_jaccard"] == pytest.approx(1 / 3)


class TestHelpers:
    def test_subset_keeps_requested_items(self):
        catalog = _two_category_catalog()
        subset = _subset(catalog, frozenset({"a1", "b1"}))
        assert subset.isbns == ["a1", "b1"]
        assert subset.source == catalog.source

    def test_subset_without_filter_returns_same_catalog(self):
        catalog = _two_category_catalog()
        assert _subset(catalog, None) is catalog

    def test_file_md5_changes_when_a_file_changes(self, tmp_path):
        (tmp_path / "header.bin").write_bytes(b"one")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "data.bin").write_bytes(b"two")
        before = file_md5(tmp_path)
        assert set(before) == {"header.bin", "nested/data.bin"}
        (tmp_path / "header.bin").write_bytes(b"changed")
        assert file_md5(tmp_path)["header.bin"] != before["header.bin"]


class TestReport:
    def _render(self, unchanged: bool = True) -> str:
        catalog = _two_category_catalog()
        queries_by_mode = {"book": build_queries(catalog, "book")}
        results = {
            mode: {
                key: run_variant(catalog, queries, key, (1, 3, 5))
                for key in ("baseline_vector", "production_hybrid")
            }
            for mode, queries in queries_by_mode.items()
        }
        before = {"chroma.sqlite3": "abc"}
        after = dict(before) if unchanged else {"chroma.sqlite3": "def"}
        return render_markdown(
            catalog=catalog,
            queries_by_mode=queries_by_mode,
            results=results,
            ks=(1, 3, 5),
            command="python scripts/diversity_benchmark.py",
            md5_before=before,
            md5_after=after,
            min_books=2,
        )

    def test_states_it_is_not_accuracy(self):
        text = self._render()
        assert "정확도가 아닙니다" in text
        assert "Recall" in text  # 왜 계산하지 않았는지 밝힌다
        assert "라벨" in text

    def test_records_md5_unchanged(self):
        assert "md5 불변: 예" in self._render(unchanged=True)
        assert "md5 불변: **아니오" in self._render(unchanged=False)

    def test_contains_both_variants_and_metric_tables(self):
        text = self._render()
        assert "`baseline_vector`" in text
        assert "`production_hybrid`" in text
        assert "고유 서브카테고리@5" in text
        assert "커버리지@5" in text
        assert "k=1" in text

    def test_k1_entropy_is_blank_not_zero(self):
        text = self._render()
        # k=1 엔트로피는 정의상 0이므로 0.000으로 단정하지 않고 –로 둔다
        assert "| `baseline_vector` | 정규화 엔트로피 | – |" in text


def test_metric_helpers_used_by_script_agree():
    """스크립트가 eval/metrics를 그대로 쓰는지(정의가 갈라지지 않는지) 확인."""
    catalog = _two_category_catalog()
    categories = catalog.categories_by_isbn()
    assert distinct_category_count(["a1", "b1"], categories) == 2
    assert category_entropy(["a1", "a2"], categories, 2) == pytest.approx(0.0)
