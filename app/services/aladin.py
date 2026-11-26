"""알라딘 도서 API 클라이언트.

기존 `book_info.py`의 두 함수를 클래스로 옮기면서 다음을 바꿨다.
- 요청에 타임아웃을 준다(기존에는 무한 대기 가능).
- 실패를 `None`/`print`로 삼키지 않고 예외로 올린다(라우터에서 404/502로 매핑).
- 응답 항목 누락 시 KeyError 대신 도메인 예외를 던진다.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from app.core.errors import BookNotFoundError, UpstreamServiceError
from app.domain import BestSeller, Book

logger = logging.getLogger("arabook.aladin")

ITEM_LOOKUP_URL = "http://www.aladin.co.kr/ttb/api/ItemLookUp.aspx"
ITEM_LIST_URL = "http://www.aladin.co.kr/ttb/api/ItemList.aspx"


class AladinClient:
    def __init__(self, ttbkey: str | None, timeout: float = 10.0, session: requests.Session | None = None):
        self._ttbkey = ttbkey
        self._timeout = timeout
        self._session = session or requests.Session()

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self._ttbkey:
            raise UpstreamServiceError("알라딘 API 키(ALADIN_TTBKEY)가 설정되어 있지 않습니다.")
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise UpstreamServiceError(f"알라딘 API 요청에 실패했습니다: {exc}") from exc
        except ValueError as exc:  # JSON이 아닌 응답
            raise UpstreamServiceError("알라딘 API 응답을 해석할 수 없습니다.") from exc

        if not isinstance(data, dict) or "item" not in data:
            raise BookNotFoundError("알라딘에서 도서 정보를 찾지 못했습니다.")
        return data

    def get_book_by_isbn(self, isbn: str) -> Book:
        data = self._get(
            ITEM_LOOKUP_URL,
            {
                "ttbkey": self._ttbkey,
                "itemIdType": "ISBN",
                "ItemId": isbn,
                "output": "JS",
                "Version": "20131101",
            },
        )
        items = data.get("item") or []
        if not items:
            raise BookNotFoundError(f"ISBN {isbn}에 해당하는 도서를 찾지 못했습니다.")
        return self._to_book(items[0])

    def get_best_sellers(self, max_results: int = 18) -> list[BestSeller]:
        data = self._get(
            ITEM_LIST_URL,
            {
                "ttbkey": self._ttbkey,
                "QueryType": "Bestseller",
                "MaxResults": max_results,
                "start": 1,
                "SearchTarget": "Book",
                "output": "js",
                "Version": "20131101",
            },
        )
        best_sellers: list[BestSeller] = []
        for item in data.get("item") or []:
            try:
                best_sellers.append(
                    BestSeller(book=self._to_book(item), best_rank=int(item.get("bestRank", 0)))
                )
            except (UpstreamServiceError, KeyError, TypeError, ValueError) as exc:
                # 한 항목이 깨져도 전체 배치를 버리지 않는다.
                logger.warning("베스트셀러 항목을 건너뜁니다: %s", exc)
        return best_sellers

    @staticmethod
    def _to_book(item: dict[str, Any]) -> Book:
        try:
            return Book(
                isbn=item["isbn13"],
                title=item["title"],
                author=item["author"],
                publisher=item["publisher"],
                publish_year=item["pubDate"][:4],
                cover_url=item["cover"].replace("coversum", "cover500"),
                description=item.get("description") or "",
            )
        except KeyError as exc:
            raise UpstreamServiceError(f"알라딘 응답에 필수 항목이 없습니다: {exc}") from exc
