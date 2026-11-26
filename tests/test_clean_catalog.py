"""카탈로그 정제 스크립트(`scripts/clean_catalog.py`) 테스트.

- 정규화 규칙과 리포트는 합성 항목으로 검증한다.
- 실제 Chroma에 쓰는 경로는 임시 폴더에서 한 번만 확인하고, **원본이 바뀌지 않는지**
  파일 해시로 검사한다.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clean_catalog.py"
_spec = importlib.util.spec_from_file_location("clean_catalog", _SCRIPT)
assert _spec and _spec.loader
clean_catalog = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = clean_catalog
_spec.loader.exec_module(clean_catalog)


def _entry(isbn, main="소설", subs="사랑, 한국 근현대사", tags="노벨문학상", document="설명"):
    metadata = {"isbn": isbn, "title": f"book-{isbn}"}
    if main is not None:
        metadata["mainCategory"] = main
    if subs is not None:
        metadata["subCategory"] = subs
    if tags is not None:
        metadata["hashtags"] = tags
    return clean_catalog.CatalogEntry(id=isbn, metadata=metadata, document=document)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------ 정규화 규칙
class TestNormalization:
    def test_first_main_category_is_kept(self):
        assert clean_catalog.normalize_main_category("소설, 역사") == "소설"
        assert clean_catalog.normalize_main_category("  자기계발 ,사회 ") == "자기계발"

    def test_missing_main_category_is_empty(self):
        assert clean_catalog.normalize_main_category(None) == ""
        assert clean_catalog.normalize_main_category("  ,  ") == ""

    def test_list_values_are_trimmed_deduplicated(self):
        assert clean_catalog.split_values(" 사랑 ,, SF,사랑 , ") == ["사랑", "SF"]

    def test_list_values_accept_sequences(self):
        assert clean_catalog.split_values(["사랑", " 사랑", "SF"]) == ["사랑", "SF"]


# ---------------------------------------------------------------------- 계획
class TestPlanCleanup:
    def test_multi_valued_main_category_is_a_fix(self):
        plan = clean_catalog.plan_cleanup([_entry("1", main="소설, 역사")], source="t")
        assert plan.n_changed == 1
        fix = plan.fixes[0]
        assert fix.after["mainCategory"] == "소설"
        assert fix.changes == ("mainCategory: '소설, 역사' → '소설'",)
        assert plan.counts()["multi_valued_main_category"] == 1

    def test_list_formatting_is_a_fix(self):
        plan = clean_catalog.plan_cleanup(
            [_entry("1", subs="사랑,한국 근현대사 , 사랑", tags="a, b, a")], source="t"
        )
        fix = plan.fixes[0]
        assert fix.after["subCategory"] == "사랑, 한국 근현대사"
        assert fix.after["hashtags"] == "a, b"
        assert len(fix.changes) == 2
        assert plan.counts()["normalized_list"] == 1

    def test_clean_entry_is_not_a_fix(self):
        plan = clean_catalog.plan_cleanup([_entry("1")], source="t")
        assert plan.n_changed == 0
        assert plan.fixes == []

    def test_missing_fields_are_issues_not_fixes(self):
        plan = clean_catalog.plan_cleanup(
            [_entry("1", main=None, subs=None, tags=None, document="")], source="t"
        )
        assert plan.n_changed == 0
        assert sorted(plan.issues) == [
            "empty_description",
            "empty_hashtags",
            "empty_main_category",
            "empty_sub_category",
        ]

    def test_duplicate_isbn_is_reported(self):
        plan = clean_catalog.plan_cleanup([_entry("1"), _entry("1")], source="t")
        assert plan.issues["duplicate_isbn"] == ["1"]

    def test_json_report_shape(self):
        plan = clean_catalog.plan_cleanup([_entry("1", main="소설, 역사")], source="t")
        payload = json.loads(json.dumps(plan.to_json(), ensure_ascii=False))
        assert payload["n_items"] == 1
        assert payload["n_changed"] == 1
        assert payload["fixes"][0]["after"]["mainCategory"] == "소설"
        assert payload["applied"] is False

    def test_markdown_report_has_before_and_after(self):
        plan = clean_catalog.plan_cleanup([_entry("9788954682152", main="소설, 역사")], source="t")
        text = clean_catalog.format_markdown(plan)
        assert "dry-run" in text
        assert "'소설, 역사'" in text
        assert "mainCategory" in text


# ---------------------------------------------------------------------- 적용
class FakeCollection:
    def __init__(self):
        self.updated: dict | None = None

    def update(self, ids, metadatas):
        self.updated = {"ids": ids, "metadatas": metadatas}


class TestApply:
    def test_update_receives_normalized_metadata(self):
        plan = clean_catalog.plan_cleanup([_entry("1", main="소설, 역사")], source="t")
        collection = FakeCollection()
        written = clean_catalog.apply_plan(plan, "unused", collection=collection)
        assert written == 1
        assert collection.updated["ids"] == ["1"]
        assert collection.updated["metadatas"][0]["mainCategory"] == "소설"

    def test_no_fix_does_nothing(self):
        plan = clean_catalog.plan_cleanup([_entry("1")], source="t")
        assert clean_catalog.apply_plan(plan, "unused", collection=FakeCollection()) == 0

    def test_copy_refuses_same_path(self, tmp_path):
        source = tmp_path / "db"
        source.mkdir()
        (source / "chroma.sqlite3").write_bytes(b"x")
        with pytest.raises(clean_catalog.CatalogCleanError):
            clean_catalog.copy_database(source, source)

    def test_copy_refuses_existing_target_without_force(self, tmp_path):
        source = tmp_path / "db"
        source.mkdir()
        (source / "chroma.sqlite3").write_bytes(b"x")
        target = tmp_path / "out"
        target.mkdir()
        with pytest.raises(clean_catalog.CatalogCleanError):
            clean_catalog.copy_database(source, target)


class TestCli:
    def test_apply_requires_out_db(self):
        exit_code = clean_catalog.main(["--db", "없는폴더", "--apply"])
        # 폴더 검사보다 인자 검사가 먼저여야 원본 보호 의도가 분명해진다
        assert exit_code == 2

    def test_missing_database_is_reported(self, tmp_path, capsys):
        exit_code = clean_catalog.main(["--db", str(tmp_path / "nope")])
        assert exit_code == 2
        assert "Chroma 폴더가 없습니다" in capsys.readouterr().err


# ------------------------------------------------------- 실제 Chroma (원본 보존)
@pytest.fixture
def chroma_source(tmp_path):
    chromadb = pytest.importorskip("chromadb")
    path = tmp_path / "source"
    client = chromadb.PersistentClient(path=str(path))
    collection = client.create_collection("books")
    collection.add(
        ids=["9788954682152", "9788936434120"],
        documents=["설명1", "설명2"],
        metadatas=[
            {
                "isbn": "9788954682152",
                "title": "작별하지 않는다",
                "author": "한강",
                "mainCategory": "소설, 역사",
                "subCategory": "사랑, 한국 근현대사",
                "hashtags": "노벨문학상, 소설",
            },
            {
                "isbn": "9788936434120",
                "title": "소년이 온다",
                "author": "한강",
                "mainCategory": "소설",
                "subCategory": "한국 근현대사",
                "hashtags": "노벨문학상",
            },
        ],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
    del client
    return path


def test_apply_writes_copy_and_leaves_source_untouched(chroma_source, tmp_path):
    source_file = chroma_source / "chroma.sqlite3"
    before = _sha(source_file)
    target = tmp_path / "clean"

    exit_code = clean_catalog.main(
        ["--db", str(chroma_source), "--apply", "--out-db", str(target)]
    )
    assert exit_code == 0
    assert _sha(source_file) == before  # 원본은 그대로

    entries = {entry.isbn: entry for entry in clean_catalog.load_entries(target)}
    assert entries["9788954682152"].metadata["mainCategory"] == "소설"
    assert entries["9788936434120"].metadata["mainCategory"] == "소설"
    # 고칠 것이 없던 책의 메타데이터도 그대로 남아 있다
    assert entries["9788936434120"].metadata["subCategory"] == "한국 근현대사"


def test_dry_run_does_not_create_target(chroma_source, tmp_path):
    target = tmp_path / "clean"
    assert clean_catalog.main(["--db", str(chroma_source), "--out-db", str(target)]) == 0
    assert not target.exists()
