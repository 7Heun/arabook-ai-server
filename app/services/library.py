"""도서 적재 파이프라인: ISBN → 메타데이터 → 해시태그 → 카테고리 → MySQL/Chroma.

기존 `/add-book` 핸들러 안에 있던 순서를 그대로 서비스로 옮겼다.
스케줄러(베스트셀러 갱신)도 같은 파이프라인을 쓰므로 중복 구현이 사라졌다.
"""

from __future__ import annotations

import asyncio
import logging

from app.services.aladin import AladinClient
from app.services.classifier import CategoryClassifier
from app.services.database import BookRepository
from app.services.kyobo import KyoboHashtagCrawler
from app.services.vector_store import BookVectorStore

logger = logging.getLogger("arabook.ingestion")


class BookIngestionService:
    def __init__(
        self,
        aladin: AladinClient,
        crawler: KyoboHashtagCrawler,
        classifier: CategoryClassifier,
        repository: BookRepository,
        store: BookVectorStore,
    ):
        self._aladin = aladin
        self._crawler = crawler
        self._classifier = classifier
        self._repository = repository
        self._store = store

    async def ensure_book(self, isbn: str) -> str:
        """ISBN으로 책을 조회해 분류하고 MySQL/Chroma에 저장한다.

        반환값은 기존 `/add-book` 응답의 message 문자열이다.
        """
        book = await asyncio.to_thread(self._aladin.get_book_by_isbn, isbn)
        hashtags = await asyncio.to_thread(self._crawler.fetch_hashtags, isbn)
        main_category, sub_categories = await self._classifier.classify(
            title=book.title,
            author=book.author,
            isbn=book.isbn,
            description=book.description,
            hashtags=hashtags,
        )

        book_id = await asyncio.to_thread(
            self._repository.insert_book_info, book, sub_categories, hashtags
        )
        logger.info("MySQL 저장 완료: isbn=%s, book_id=%s", book.isbn, book_id)

        message = await self._store.add_book(book, main_category, sub_categories, hashtags)
        logger.info("Chroma 저장 결과: isbn=%s, %s", book.isbn, message)
        return message
