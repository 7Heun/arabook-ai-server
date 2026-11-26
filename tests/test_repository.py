"""`BookRepository` 조회 SQL 테스트 (실제 DB 없이).

새로 추가한 두 조회(최근 추천 ISBN, 여러 ISBN의 book_id)는 SQL과 바인딩 값이
곧 계약이다. SSH 터널을 열지 않고 `_connection`만 대역으로 바꿔 확인한다.
"""

from __future__ import annotations

from contextlib import contextmanager

from app.core.config import DatabaseSettings
from app.services.database import BookRepository


class FakeCursor:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.executed: list[tuple] = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True

    def commit(self):
        pass

    def rollback(self):
        pass


def _repository(cursor: FakeCursor) -> tuple[BookRepository, FakeConnection]:
    settings = DatabaseSettings(
        ssh_host=None,
        ssh_port=22,
        ssh_username=None,
        ssh_private_key=None,
        mysql_host=None,
        username=None,
        password=None,
        database=None,
    )
    repository = BookRepository(settings)
    connection = FakeConnection(cursor)

    @contextmanager
    def fake_connection():
        yield connection

    repository._connection = fake_connection  # type: ignore[method-assign]
    return repository, connection


class TestRecentRecommendedIsbns:
    def test_queries_recommendation_history_for_member_and_window(self):
        cursor = FakeCursor([{"isbn": "9791131963746"}, {"isbn": "9791158790745"}])
        repository, _ = _repository(cursor)

        assert repository.get_recent_recommended_isbns(7, days=30) == [
            "9791131963746",
            "9791158790745",
        ]
        sql, params = cursor.executed[0]
        assert "daily_book_id_recommendations" in sql
        assert "JOIN books" in sql
        assert "recommendation_date" in sql
        assert params == (7, 30)

    def test_rows_without_isbn_are_dropped(self):
        cursor = FakeCursor([{"isbn": "a"}, {"isbn": None}, {"isbn": "b"}])
        repository, _ = _repository(cursor)
        assert repository.get_recent_recommended_isbns(1) == ["a", "b"]

    def test_non_positive_days_skips_query(self):
        cursor = FakeCursor([{"isbn": "a"}])
        repository, _ = _repository(cursor)
        assert repository.get_recent_recommended_isbns(1, days=0) == []
        assert cursor.executed == []

    def test_no_history_returns_empty(self):
        repository, _ = _repository(FakeCursor([]))
        assert repository.get_recent_recommended_isbns(1) == []


class TestBookIdsByIsbns:
    def test_batches_into_single_in_query(self):
        cursor = FakeCursor(
            [{"book_id": 11, "isbn": "a"}, {"book_id": 22, "isbn": "b"}]
        )
        repository, _ = _repository(cursor)

        assert repository.get_book_ids_by_isbns(["a", "b"]) == {"a": 11, "b": 22}
        sql, params = cursor.executed[0]
        assert sql.count("%s") == 2
        assert "IN (%s, %s)" in sql
        assert params == ("a", "b")

    def test_missing_isbns_are_absent_from_mapping(self):
        repository, _ = _repository(FakeCursor([{"book_id": 11, "isbn": "a"}]))
        assert repository.get_book_ids_by_isbns(["a", "없는책"]) == {"a": 11}

    def test_blank_duplicates_are_removed(self):
        cursor = FakeCursor([])
        repository, _ = _repository(cursor)
        assert repository.get_book_ids_by_isbns([" a ", "a", "", "b"]) == {}
        _, params = cursor.executed[0]
        assert params == ("a", "b")

    def test_empty_input_skips_query(self):
        cursor = FakeCursor([])
        repository, _ = _repository(cursor)
        assert repository.get_book_ids_by_isbns([]) == {}
        assert cursor.executed == []
