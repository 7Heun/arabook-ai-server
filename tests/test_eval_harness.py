"""평가 하네스/변형 테스트 (합성 카탈로그 사용, 외부 API 불필요)."""

from __future__ import annotations

import json

import pytest

from eval.catalog import Catalog, CatalogError, CatalogItem, load_catalog_from_json
from eval.harness import LabelError, evaluate, load_labels, validate_labels
from eval.recommenders import VectorIndex


def _item(isbn: str, subs: tuple[str, ...], embedding: tuple[float, ...]) -> CatalogItem:
    return CatalogItem(
        isbn=isbn,
        title=f"book-{isbn}",
        author="author",
        main_category="소설",
        sub_categories=subs,
        hashtags=(),
        description="desc",
        embedding=embedding,
    )


@pytest.fixture
def catalog() -> Catalog:
    return Catalog(
        items=[
            _item("a", ("x",), (1.0, 0.0)),
            _item("b", ("y",), (0.0, 1.0)),
            _item("c", ("x",), (0.995, 0.1)),
            _item("d", ("z",), (0.7, 0.7)),
        ],
        source="test",
        dimension=2,
    )


class FixedEmbedder:
    name = "fixed"

    def __init__(self, vector: list[float]):
        self.vector = vector

    def embed(self, texts):
        return [list(self.vector) for _ in texts]


def test_baseline_orders_by_l2_distance(catalog):
    results = evaluate(
        labels=_labels([{"member_id": 1, "preferences": ["x"], "relevant_isbns": ["c"]}]),
        catalog=catalog,
        embedder=FixedEmbedder([1.0, 0.0]),
        variant_keys=["baseline_vector"],
        ks=[1, 3],
    )
    ranking = results["baseline_vector"].ranked["1"]
    assert ranking[0] == "a"  # query와 거리 0
    assert ranking[1] == "c"  # x 카테고리, 두 번째로 가까움
    assert results["baseline_vector"].aggregate["recall"][1]["value"] == 0.0
    assert results["baseline_vector"].aggregate["recall"][3]["value"] == 1.0
    assert results["baseline_vector"].aggregate["mrr"]["value"] == pytest.approx(0.5)


def test_category_filter_restricts_candidates(catalog):
    results = evaluate(
        labels=_labels([{"member_id": 1, "preferences": ["y"], "relevant_isbns": ["b"]}]),
        catalog=catalog,
        embedder=FixedEmbedder([1.0, 0.0]),
        variant_keys=["category_filter"],
        ks=[3],
    )
    assert results["category_filter"].ranked["1"] == ["b"]


def test_category_filter_falls_back_when_no_overlap(catalog):
    results = evaluate(
        labels=_labels([{"member_id": 1, "preferences": ["없는카테고리"], "relevant_isbns": ["a"]}]),
        catalog=catalog,
        embedder=FixedEmbedder([1.0, 0.0]),
        variant_keys=["category_filter"],
        ks=[4],
    )
    assert len(results["category_filter"].ranked["1"]) == 4


def test_hybrid_boost_reorders_on_category_match(catalog):
    results = evaluate(
        labels=_labels([{"member_id": 1, "preferences": ["y"], "relevant_isbns": ["b"]}]),
        catalog=catalog,
        embedder=FixedEmbedder([1.0, 0.0]),
        variant_keys=["hybrid_category_boost"],
        ks=[4],
        alpha=1.0,
    )
    # 선호 카테고리(y)와 일치하는 b가 코사인 0이었는데도 2위로 올라온다.
    # (동률 1.0인 a와는 안정 정렬로 a가 먼저)
    ranking = results["hybrid_category_boost"].ranked["1"]
    assert ranking == ["a", "b", "c", "d"]


def test_production_hybrid_applies_mmr_diversity():
    """운영 변형은 카테고리 가중에 더해 MMR로 같은 카테고리 중복을 뒤로 미룬다.

    관련도(b2 1.05 > c1 0.88)만 보면 b2가 2순위지만, b2는 1순위와 서브카테고리가
    같아 중복 페널티(-0.15)를 받아 c1(0.748)에 밀린다(0.7425).
    """
    catalog = Catalog(
        items=[
            _item("b1", ("y",), (1.0, 0.0)),
            _item("b2", ("y",), (0.9, 0.43589)),
            _item("c1", ("z",), (0.88, 0.47497)),
        ],
        source="test",
        dimension=2,
    )
    results = evaluate(
        labels=_labels([{"member_id": 1, "preferences": ["y"], "relevant_isbns": ["b1"]}]),
        catalog=catalog,
        embedder=FixedEmbedder([1.0, 0.0]),
        variant_keys=["production_hybrid"],
        ks=[3],
    )
    assert results["production_hybrid"].ranked["1"] == ["b1", "c1", "b2"]


def test_vector_index_requires_embeddings():
    catalog_without_embeddings = Catalog(
        items=[
            CatalogItem(
                isbn="a",
                title="t",
                author="a",
                main_category="소설",
                sub_categories=("x",),
                hashtags=(),
            )
        ],
        source="test",
    )
    assert catalog_without_embeddings.has_embeddings is False
    with pytest.raises(CatalogError):
        VectorIndex(catalog_without_embeddings)


def test_validate_labels_reports_unknown_isbn_and_category(catalog):
    labels = _labels(
        [{"member_id": 7, "preferences": ["x", "없는카테고리"], "relevant_isbns": ["a", "zzz"]}]
    )
    report = validate_labels(labels, catalog)
    assert report.unknown_relevant_isbns == {"7": ["zzz"]}
    assert report.unknown_preference_labels == {"7": ["없는카테고리"]}
    assert report.has_blocking_issues is False


def test_load_labels_rejects_bad_shape(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"users": [{"preferences": []}]}), encoding="utf-8")
    with pytest.raises(LabelError):
        load_labels(path)


def test_load_labels_detects_synthetic_source(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(
        json.dumps({"meta": {"label_source": "example"}, "users": [{"member_id": 1}]}),
        encoding="utf-8",
    )
    assert load_labels(path).is_synthetic is True


def test_catalog_json_roundtrip(tmp_path, catalog):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(catalog.to_json()), encoding="utf-8")
    loaded = load_catalog_from_json(path)
    assert loaded.isbns == catalog.isbns
    assert loaded.has_embeddings is False
    assert loaded.get("a").sub_categories == ("x",)


def _labels(users):
    from eval.harness import EvalUser, LabelSet

    return LabelSet(
        path="memory",
        meta={"label_source": "unit-test"},
        users=[
            EvalUser(
                member_id=str(u["member_id"]),
                preferences=tuple(u.get("preferences", [])),
                relevant=tuple(u.get("relevant_isbns", [])),
            )
            for u in users
        ],
    )
