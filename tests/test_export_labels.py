"""라벨 추출 스크립트(`scripts/export_labels.py`) 테스트.

실제 MySQL 없이, 가짜 커서가 SQL과 파라미터를 기록하고 픽스처 행을 돌려준다.
결과 payload는 **실제 스키마 파일**(`eval/labels.schema.json`)로 검증한다.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_labels.py"
_spec = importlib.util.spec_from_file_location("export_labels", _SCRIPT)
assert _spec and _spec.loader
export_labels = importlib.util.module_from_spec(_spec)
# dataclass가 모듈을 sys.modules에서 찾으므로 등록한 뒤 실행한다.
sys.modules[_spec.name] = export_labels
_spec.loader.exec_module(export_labels)

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "eval" / "labels.schema.json").read_text(encoding="utf-8")
)
CATALOG = {"9791131963746", "9791158790745", "9788954782227"}


class FakeCursor:
    def __init__(self, parent: "FakeConnection"):
        self.parent = parent
        self._rows: list[dict] = []

    def execute(self, sql, params=()):
        self.parent.executed.append((sql, tuple(params)))
        self._rows = self.parent.rows_for(sql)

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, tables):
        self.tables = tables
        self.executed: list[tuple[str, tuple]] = []

    def rows_for(self, sql: str) -> list[dict]:
        for marker, rows in self.tables.items():
            if marker in sql:
                return rows
        return []

    def cursor(self):
        return FakeCursor(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeRepository:
    def __init__(self, tables):
        self.connection_obj = FakeConnection(tables)

    def connection(self):
        return self.connection_obj


PREFERENCES = [
    {"member_id": 1001, "sub_category_name": "추리/스릴러"},
    {"member_id": 1001, "sub_category_name": "SF"},
    {"member_id": 1002, "sub_category_name": "천문학/지구과학"},
    {"member_id": 1003, "sub_category_name": "청소년"},  # 행동 없음
    {"member_id": 1004, "sub_category_name": "역사"},
]

LOANS = [
    {"member_id": 1001, "isbn": "9791131963746", "acted_at": "2026-09-10"},
    {"member_id": 1001, "isbn": "979-11-5879-0745", "acted_at": "2026-09-11"},  # 하이픈
    {"member_id": 1002, "isbn": "9788954782227", "acted_at": "2026-10-01"},
    {"member_id": 1002, "isbn": "9780000000000", "acted_at": "2026-10-02"},  # 카탈로그 밖
    {"member_id": 1002, "isbn": "12345", "acted_at": "2026-10-03"},  # 형식 오류
]

LIKES = [
    {"member_id": 1001, "isbn": "9791131963746", "acted_at": "2026-09-12"},  # 중복(가중치 낮음)
    {"member_id": 1001, "isbn": "9788954782227", "acted_at": "2026-09-13"},
]

EXPOSED = [{"member_id": 1001, "isbn": "9788954782227"}]


def _tables(preferences=PREFERENCES, loans=LOANS, likes=LIKES, exposed=EXPOSED):
    return {
        "member_sub_category_selections": preferences,
        "FROM rentals": loans,
        "FROM book_likes": likes,
        "daily_book_id_recommendations": exposed,
    }


def _run(tables=None, **kwargs):
    repository = FakeRepository(tables or _tables())
    options = {
        "since": "2026-09-01",
        "until": "2026-12-01",
        "label_source": "mysql-actions-2026-09..11",
        "catalog_isbns": CATALOG,
    }
    options.update(kwargs)
    preferences = export_labels.load_preferences(repository)
    sources = [
        export_labels.parse_action_spec(
            "table=rentals,member=member_id,book=book_id,time=rented_at,weight=2"
        ),
        export_labels.parse_action_spec(
            "table=book_likes,member=member_id,book=book_id,time=created_at,weight=1"
        ),
    ]
    actions = export_labels.load_actions(repository, sources, options["since"], options["until"])
    excluded = (
        export_labels.load_exposed(repository, options["since"], options["until"])
        if options.pop("exclude_recommended", False)
        else []
    )
    options.setdefault("excluded", excluded)
    return repository, export_labels.build_labels(actions, preferences, **options)


# ------------------------------------------------------------------ 동작 정의
class TestActionSpec:
    def test_parses_book_id_source(self):
        source = export_labels.parse_action_spec(
            "table=rentals,member=member_id,book=book_id,time=rented_at,weight=2"
        )
        assert source.table == "rentals"
        assert source.weight == 2
        assert "JOIN books b ON b.book_id = a.book_id" in source.sql()
        assert "a.rented_at >= %s" in source.sql()

    def test_parses_isbn_source_without_join(self):
        source = export_labels.parse_action_spec(
            "table=reviews,member=member_id,isbn=book_isbn,time=created_at"
        )
        assert "JOIN" not in source.sql()
        assert "a.book_isbn AS isbn" in source.sql()

    @pytest.mark.parametrize(
        "spec",
        [
            "table=rentals,member=member_id,time=rented_at",  # book/isbn 없음
            "table=rentals,member=member_id,book=book_id,isbn=isbn,time=t",  # 둘 다
            "member=member_id,book=book_id,time=t",  # table 없음
            "table=rentals;DROP,member=m,book=b,time=t",  # 식별자 아님
            "table=rentals,member=m,book=b,time=t,weight=zero",
            "table=rentals,member=m,book=b,time=t,unknown=1",
        ],
    )
    def test_rejects_invalid_specs(self, spec):
        with pytest.raises(export_labels.LabelExportError):
            export_labels.parse_action_spec(spec)


# ------------------------------------------------------------------ 라벨 생성
class TestBuildLabels:
    def test_payload_passes_schema(self):
        _, result = _run()
        assert export_labels.validate_payload(result.payload, SCHEMA) == []

    def test_groups_actions_by_member_with_max_weight(self):
        _, result = _run()
        users = {str(user["member_id"]): user for user in result.payload["users"]}
        assert set(users) == {"1001", "1002"}  # 1003은 행동 없음, 1004는 행동 없음
        assert users["1001"]["preferences"] == ["추리/스릴러", "SF"]
        # 대여(2)가 좋아요(1)보다 우선한다
        assert users["1001"]["graded_relevance"]["9791131963746"] == 2
        assert users["1001"]["graded_relevance"]["9788954782227"] == 1

    def test_normalizes_isbn_and_drops_invalid(self):
        _, result = _run()
        users = {str(user["member_id"]): user for user in result.payload["users"]}
        assert "9791158790745" in users["1001"]["relevant_isbns"]
        assert result.stats["invalid_isbn_rows"] == 1
        assert result.stats["rows_outside_catalog"] == 1

    def test_excludes_exposed_books(self):
        _, result = _run(exclude_recommended=True)
        users = {str(user["member_id"]): user for user in result.payload["users"]}
        assert "9788954782227" not in users["1001"]["relevant_isbns"]
        assert result.stats["rows_excluded_as_exposed"] == 1  # 좋아요 1건(대여는 다른 회원)
        assert result.payload["meta"]["extraction"]["excluded_recommended"] is True

    def test_keeps_graded_only_when_weights_differ(self):
        _, result = _run()
        users = {str(user["member_id"]): user for user in result.payload["users"]}
        # 1002는 대여만 있어 모든 정답이 2 → graded_relevance는 1로 정규화되어 포함된다
        assert users["1002"]["graded_relevance"] == {"9788954782227": 2}

    def test_min_relevant_filters_users(self):
        _, result = _run(min_relevant=2)
        assert [str(u["member_id"]) for u in result.payload["users"]] == ["1001"]
        # 1002(정답 1건) + 행동이 없는 1003, 1004
        assert result.stats["users_without_relevant"] == 3

    def test_meta_records_extraction_method(self):
        _, result = _run()
        meta = result.payload["meta"]
        assert meta["format_version"] == 1
        assert meta["label_source"] == "mysql-actions-2026-09..11"
        assert meta["extraction"]["since"] == "2026-09-01"
        assert meta["extraction"]["actions"] == ["book_likes", "rentals"]

    def test_no_catalog_filter_keeps_unknown_isbn(self):
        _, filtered = _run()
        _, unfiltered = _run(catalog_isbns=None)
        unknown = {str(u["member_id"]): u for u in unfiltered.payload["users"]}["1002"]
        assert "9780000000000" in unknown["relevant_isbns"]
        assert unfiltered.stats["rows_outside_catalog"] == 0
        assert filtered.stats["rows_outside_catalog"] == 1

    def test_raises_when_no_user_survives(self):
        with pytest.raises(export_labels.LabelExportError):
            _run(min_relevant=99)


# -------------------------------------------------------------- 스키마 검증기
class TestValidatePayload:
    def test_detects_missing_required_key(self):
        errors = export_labels.validate_payload({"users": [{"member_id": 1}]}, SCHEMA)
        assert any("preferences" in error for error in errors)

    def test_detects_wrong_type(self):
        payload = {
            "users": [{"member_id": 1, "preferences": "SF", "relevant_isbns": []}]
        }
        errors = export_labels.validate_payload(payload, SCHEMA)
        assert any("preferences" in error for error in errors)

    def test_detects_extra_key_in_user(self):
        payload = {
            "users": [
                {
                    "member_id": 1,
                    "preferences": ["SF"],
                    "relevant_isbns": [],
                    "surprise": 1,
                }
            ]
        }
        errors = export_labels.validate_payload(payload, SCHEMA)
        assert any("surprise" in error for error in errors)

    def test_detects_wrong_format_version(self):
        payload = {"meta": {"format_version": 2}, "users": [], "x": None}
        errors = export_labels.validate_payload(payload, SCHEMA)
        assert any("format_version" in error for error in errors)


# ------------------------------------------------------------------ CLI 경로
class TestCli:
    def test_run_writes_file_and_validates(self, tmp_path, monkeypatch):
        repository = FakeRepository(_tables())
        monkeypatch.setattr(
            export_labels, "load_catalog_isbns", lambda path: CATALOG
        )
        monkeypatch.setattr(
            "app.services.database.BookRepository", lambda settings: repository
        )
        out = tmp_path / "labels.json"
        exit_code = export_labels.main(
            [
                "--since",
                "2026-09-01",
                "--until",
                "2026-12-01",
                "--action",
                "table=rentals,member=member_id,book=book_id,time=rented_at,weight=2",
                "--action",
                "table=book_likes,member=member_id,book=book_id,time=created_at",
                "--exclude-recommended",
                "--out",
                str(out),
            ]
        )
        assert exit_code == 0
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert export_labels.validate_payload(payload, SCHEMA) == []
        assert payload["meta"]["label_source"] == "mysql-actions-2026-09-01..2026-12-01"

    def test_run_rejects_missing_action(self, capsys):
        exit_code = export_labels.main(["--since", "2026-09-01", "--until", "2026-12-01"])
        assert exit_code == 2
        assert "--action" in capsys.readouterr().err

    def test_run_rejects_bad_date_range(self, capsys):
        exit_code = export_labels.main(
            [
                "--since",
                "2026-12-01",
                "--until",
                "2026-09-01",
                "--action",
                "table=t,member=m,book=b,time=t",
            ]
        )
        assert exit_code == 2
        assert "앞서야" in capsys.readouterr().err
