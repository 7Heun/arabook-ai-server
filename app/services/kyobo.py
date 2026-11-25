"""교보문고 검색 결과에서 해시태그를 수집한다.

기존 `crawling.py`와 동작은 같다(검색 페이지의 `a.tag` 파싱, 표준 카테고리 제외,
중복 제거). 달라진 점은 실패 시 `None` 대신 예외를 던진다는 것과, 타임아웃을
준다는 것뿐이다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from app.core.errors import UpstreamServiceError

logger = logging.getLogger("arabook.kyobo")

SEARCH_URL = "https://search.kyobobook.co.kr/search"


class KyoboHashtagCrawler:
    def __init__(
        self,
        categories_path: str | Path = "categories.txt",
        timeout: float = 10.0,
        session: requests.Session | None = None,
    ):
        self._categories_path = Path(categories_path)
        self._timeout = timeout
        self._session = session or requests.Session()
        self._categories: list[str] | None = None

    def load_categories(self) -> list[str]:
        if self._categories is not None:
            return self._categories
        try:
            self._categories = [
                line.strip()
                for line in self._categories_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except OSError as exc:
            # 카테고리 목록이 없으면 해시태그 필터만 건너뛴다(수집 자체는 계속).
            logger.warning("카테고리 파일을 읽지 못했습니다(%s): %s", self._categories_path, exc)
            self._categories = []
        return self._categories

    def fetch_hashtags(self, isbn: str) -> list[str]:
        try:
            response = self._session.get(
                SEARCH_URL, params={"keyword": isbn}, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise UpstreamServiceError(f"교보문고 검색 요청에 실패했습니다: {exc}") from exc

        soup = BeautifulSoup(response.text, "html.parser")
        hashtags = {tag.text.replace("#", "").strip() for tag in soup.select("a.tag")}
        standard_categories = set(self.load_categories())
        return sorted(tag for tag in hashtags if tag and tag not in standard_categories)
