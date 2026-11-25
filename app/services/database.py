"""MySQL 접근 계층.

기존 `database_conn.py`는 호출마다 SSH 터널을 열고, 실패를 `print`로 삼킨 뒤
`None`을 돌려줬다. 그 결과 상위 로직이 `None`을 상대로 `.cursor()`를 호출해
엉뚱한 500을 냈다. 여기서는 연결 실패를 `ServiceUnavailableError`로 올리고,
연결/터널 정리를 한 곳(`_connection`)에서 보장한다.

호출마다 터널을 새로 여는 구조는 그대로 두었다. 커넥션 풀 도입은 운영 환경
(워커 수, DB 방화벽)을 봐야 해서 이번 범위에서 바꾸지 않았다.
"""

from __future__ import annotations

import ast
import logging
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import pymysql
from sshtunnel import SSHTunnelForwarder

from app.core.config import DatabaseSettings
from app.core.errors import ConfigurationError, ServiceUnavailableError
from app.domain import Book

logger = logging.getLogger("arabook.database")


class BookRepository:
    def __init__(self, settings: DatabaseSettings):
        self._settings = settings

    @contextmanager
    def _connection(self) -> Iterator[pymysql.connections.Connection]:
        settings = self._settings
        if not settings.configured:
            raise ConfigurationError("데이터베이스 접속 환경 변수가 설정되어 있지 않습니다.")

        tunnel: SSHTunnelForwarder | None = None
        connection: pymysql.connections.Connection | None = None
        try:
            tunnel = SSHTunnelForwarder(
                (settings.ssh_host, settings.ssh_port),
                ssh_username=settings.ssh_username,
                ssh_pkey=settings.ssh_private_key,
                remote_bind_address=(settings.mysql_host, 3306),
            )
            tunnel.start()
            connection = pymysql.connect(
                host="127.0.0.1",
                user=settings.username,
                password=settings.password,
                database=settings.database,
                charset="utf8mb4",
                port=tunnel.local_bind_port,
                cursorclass=pymysql.cursors.DictCursor,
            )
        except Exception as exc:
            if tunnel is not None:
                tunnel.close()
            raise ServiceUnavailableError(f"데이터베이스에 연결하지 못했습니다: {exc}") from exc

        try:
            yield connection
        finally:
            connection.close()
            if tunnel is not None:
                tunnel.close()

    # ------------------------------------------------------------------ 조회
    def get_sub_category_id(self, category_name: str) -> int | None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT sub_category_id FROM sub_categories WHERE sub_category_name = %s",
                    (category_name,),
                )
                row = cursor.fetchone()
                return row["sub_category_id"] if row else None

    def get_book_id_by_isbn(self, isbn: str) -> int | None:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT book_id FROM books WHERE isbn = %s", (isbn,))
                row = cursor.fetchone()
                return row["book_id"] if row else None

    def get_member_preferences(self, member_id: int | str) -> list[str]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT sc.sub_category_name
                    FROM member_sub_category_selections ms
                    JOIN sub_categories sc ON ms.sub_category_id = sc.sub_category_id
                    WHERE ms.member_id = %s
                    """,
                    (member_id,),
                )
                rows = cursor.fetchall()
                return [row["sub_category_name"] for row in rows]

    def get_all_member_ids(self) -> list[int]:
        with self._connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT member_id FROM members")
                rows = cursor.fetchall()
                return [row["member_id"] for row in rows]

    # ------------------------------------------------------------------ 쓰기
    def insert_book_info(
        self, book: Book, category_names: Sequence[str] | str, hashtags: Sequence[str]
    ) -> int | None:
        """책 + 해시태그 + 카테고리 매핑을 저장하고 book_id를 돌려준다.

        기존 구현은 `ON DUPLICATE KEY UPDATE` 뒤의 `cursor.lastrowid`를 book_id로
        썼다. 이미 있는 책이면 lastrowid가 0이라 해시태그/카테고리 매핑이
        book_id=0으로 들어가는 버그가 있었다. 저장 후 ISBN으로 다시 조회해 고쳤다.
        """
        if isinstance(category_names, str):
            # 기존 호출부가 문자열로 넘기는 경우가 있어 호환을 유지한다.
            category_names = ast.literal_eval(category_names)

        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO books (isbn, title, author, description, publisher, cover_url, publish_year)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON DUPLICATE KEY UPDATE
                        title = VALUES(title), author = VALUES(author), description = VALUES(description),
                        publisher = VALUES(publisher), cover_url = VALUES(cover_url), publish_year = VALUES(publish_year)
                        """,
                        (
                            book.isbn,
                            book.title,
                            book.author,
                            book.description,
                            book.publisher,
                            book.cover_url,
                            book.publish_year,
                        ),
                    )
                    cursor.execute("SELECT book_id FROM books WHERE isbn = %s", (book.isbn,))
                    row = cursor.fetchone()
                    book_id = row["book_id"] if row else None
                    if book_id is None:
                        raise ServiceUnavailableError("책 정보를 저장했지만 book_id를 찾지 못했습니다.")

                    for hashtag in hashtags:
                        cursor.execute(
                            "INSERT INTO hashtags (name) VALUES (%s) ON DUPLICATE KEY UPDATE name = VALUES(name)",
                            (hashtag,),
                        )
                        cursor.execute("SELECT hashtag_id FROM hashtags WHERE name = %s", (hashtag,))
                        hashtag_row = cursor.fetchone()
                        if hashtag_row is None:
                            continue
                        cursor.execute(
                            """
                            INSERT INTO book_hashtag_mappings (book_id, hashtag_id)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE book_id = VALUES(book_id), hashtag_id = VALUES(hashtag_id)
                            """,
                            (book_id, hashtag_row["hashtag_id"]),
                        )

                    for category_name in category_names:
                        cursor.execute(
                            "SELECT sub_category_id FROM sub_categories WHERE sub_category_name = %s",
                            (category_name,),
                        )
                        sub_row = cursor.fetchone()
                        if sub_row is None:
                            logger.warning("DB에 없는 서브카테고리입니다: %s", category_name)
                            continue
                        cursor.execute(
                            """
                            INSERT INTO book_sub_category_mappings (book_id, sub_category_id)
                            VALUES (%s, %s)
                            ON DUPLICATE KEY UPDATE book_id = VALUES(book_id), sub_category_id = VALUES(sub_category_id)
                            """,
                            (book_id, sub_row["sub_category_id"]),
                        )

                connection.commit()
                return book_id
            except ServiceUnavailableError:
                connection.rollback()
                raise
            except Exception as exc:
                connection.rollback()
                raise ServiceUnavailableError(f"책 정보 저장에 실패했습니다: {exc}") from exc

    def save_recommendation(self, member_id: int | str, book_id: int, recommendation_date: Any) -> None:
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO daily_book_id_recommendations (member_id, book_id, recommendation_date)
                        VALUES (%s, %s, %s)
                        """,
                        (member_id, book_id, recommendation_date),
                    )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise ServiceUnavailableError(f"추천 이력 저장에 실패했습니다: {exc}") from exc

    def insert_best_sellers(self, best_sellers: Sequence[dict[str, Any]]) -> None:
        with self._connection() as connection:
            try:
                with connection.cursor() as cursor:
                    cursor.execute("DELETE FROM monthly_best_sellers")
                    cursor.executemany(
                        "INSERT INTO monthly_best_sellers (book_isbn, book_rank) VALUES (%s, %s)",
                        [(item["isbn"], item["best_rank"]) for item in best_sellers],
                    )
                connection.commit()
            except Exception as exc:
                connection.rollback()
                raise ServiceUnavailableError(f"베스트셀러 저장에 실패했습니다: {exc}") from exc
