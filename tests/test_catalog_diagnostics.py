"""임베딩 공간 진단(scripts/catalog_diagnostics.py) 테스트."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from eval.catalog import Catalog, CatalogItem

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "catalog_diagnostics.py"
_spec = importlib.util.spec_from_file_location("catalog_diagnostics", _SCRIPT)
assert _spec and _spec.loader
diagnostics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnostics)


def _catalog(items):
    return Catalog(items=items, source="test", dimension=2)


def _item(isbn, subs, embedding):
    return CatalogItem(
        isbn=isbn,
        title=isbn,
        author="a",
        main_category="소설",
        sub_categories=subs,
        hashtags=(),
        embedding=embedding,
    )


def test_neighbors_share_category_when_clustered():
    catalog = _catalog(
        [
            _item("a", ("x",), (1.0, 0.0)),
            _item("b", ("x",), (0.99, 0.01)),
            _item("c", ("y",), (0.0, 1.0)),
        ]
    )
    result = diagnostics.diagnose(catalog)
    assert result["n_items"] == 3
    assert result["hit_at_1"] == 2  # a<->b 는 서로 같은 카테고리, c는 다름
    assert result["share_at_1"] == pytest.approx(2 / 3)
    assert result["lift_at_1"] > 1


def test_expected_rate_reflects_category_frequency():
    # 모두 같은 카테고리면 우연 기대값도 100% → lift 1
    catalog = _catalog(
        [
            _item("a", ("x",), (1.0, 0.0)),
            _item("b", ("x",), (0.0, 1.0)),
        ]
    )
    result = diagnostics.diagnose(catalog)
    assert result["expected_at_1"] == pytest.approx(1.0)
    assert result["lift_at_1"] == pytest.approx(1.0)


def test_lists_books_without_shared_category():
    catalog = _catalog(
        [
            _item("a", ("x",), (1.0, 0.0)),
            _item("b", ("y",), (0.99, 0.01)),
        ]
    )
    result = diagnostics.diagnose(catalog)
    assert result["hit_at_1"] == 0
    assert len(result["no_shared_category"]) == 2
