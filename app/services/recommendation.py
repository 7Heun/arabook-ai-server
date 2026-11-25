"""도서 추천.

기존에는 `ChromaManager.recommend_book`이 Chroma 검색 + MySQL 조회/저장을 한
클래스에서 처리했다. 여기서는 벡터 검색(`BookVectorStore`)과 DB(`BookRepository`)를
분리하고 이 서비스가 둘을 조합한다.

반환 메시지 문자열은 클라이언트(Spring 서버)가 그대로 보던 값이라 유지했다.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from app.services.database import BookRepository
from app.services.llm import EmbeddingService
from app.services.vector_store import BookVectorStore

logger = logging.getLogger("arabook.recommendation")

NO_PREFERENCES = "No preferences found for the member."
NO_BOOK = "No book found for the given preferences."
NO_BOOK_ID = "No book ID found for the given ISBN."


class RecommendationService:
    def __init__(
        self,
        store: BookVectorStore,
        repository: BookRepository,
        embedder: EmbeddingService,
    ):
        self._store = store
        self._repository = repository
        self._embedder = embedder

    async def recommend(self, member_id: int | str) -> str:
        preferences = await asyncio.to_thread(self._repository.get_member_preferences, member_id)
        if not preferences:
            return NO_PREFERENCES

        embedding = await self._embedder.embed_query(" ".join(preferences))
        metadata = await self._store.find_nearest(embedding, n_results=1)
        if not metadata:
            return NO_BOOK

        isbn = metadata.get("isbn")
        if not isbn:
            logger.warning("벡터 검색 결과에 isbn 메타데이터가 없습니다: %s", metadata)
            return NO_BOOK

        book_id = await asyncio.to_thread(self._repository.get_book_id_by_isbn, isbn)
        if not book_id:
            return NO_BOOK_ID

        # 기존 동작 유지: '내일' 날짜로 추천 이력을 남긴다.
        recommendation_date = (datetime.now() + timedelta(days=1)).date()
        await asyncio.to_thread(
            self._repository.save_recommendation, member_id, book_id, recommendation_date
        )
        return f"Recommended book ID: {book_id}"
