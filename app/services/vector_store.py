"""Chroma 벡터 저장소.

`chromadb`는 동기 라이브러리이므로, async 핸들러에서 직접 부르면 이벤트 루프를
막는다. 여기서는 Chroma 호출을 `asyncio.to_thread`로 넘기고, 임베딩(네트워크)만
await 한다.

기존 코드와 달라진 점
- `get_collection` 대신 없으면 만드는 `get_or_create_collection`을 쓴다.
  `chroma_db/`를 git에서 빼면 새로 clone한 환경에는 컬렉션이 없어 기존 코드가
  바로 죽었다.
- `langchain_openai` 대신 `EmbeddingService`(OpenAI SDK)로 임베딩한다. 모델은
  기존과 같은 `text-embedding-ada-002`라 벡터 공간이 호환된다.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.errors import BookNotFoundError, InvalidInputError, ServiceUnavailableError
from app.domain import Book
from app.services.llm import EmbeddingService

logger = logging.getLogger("arabook.vector_store")

ADDED_MESSAGE = "책 정보가 성공적으로 추가되었습니다."
ALREADY_EXISTS_MESSAGE = "이미 존재하는 책 정보입니다."
NOT_FOUND_MESSAGE = "존재하지 않는 책 정보입니다."
DELETED_MESSAGE = "책 정보가 성공적으로 삭제되었습니다."


class BookVectorStore:
    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedder: EmbeddingService,
        collection: Any | None = None,
    ):
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embedder = embedder
        self._collection = collection

    @property
    def collection(self) -> Any:
        if self._collection is None:
            try:
                client = chromadb.PersistentClient(
                    path=self._persist_directory,
                    settings=ChromaSettings(anonymized_telemetry=False),
                )
                try:
                    self._collection = client.get_collection(self._collection_name)
                except Exception:
                    self._collection = client.get_or_create_collection(self._collection_name)
            except Exception as exc:
                raise ServiceUnavailableError(f"벡터 저장소를 열지 못했습니다: {exc}") from exc
        return self._collection

    @staticmethod
    def _embed_text(
        book: Book, main_category: str, sub_category: Iterable[str], hashtags: Iterable[str]
    ) -> str:
        """임베딩 대상 텍스트. 기존 chroma_manager와 같은 구성을 유지한다."""
        return (
            f"title: {book.title}\nauthor: {book.author}\ndescription: {book.description}\n"
            f"main_category: {main_category}\nsub_category: {', '.join(sub_category)}\n"
            f"hashtags: {', '.join(hashtags)}"
        )

    async def add_book(
        self,
        book: Book,
        main_category: str | None,
        sub_category: list[str] | None,
        hashtags: list[str] | None,
    ) -> str:
        if not main_category or not isinstance(main_category, str):
            raise InvalidInputError("main_category는 비어 있지 않은 문자열이어야 합니다.")
        if sub_category is None or not isinstance(sub_category, list):
            raise InvalidInputError("sub_category는 리스트여야 합니다.")
        if hashtags is None or not isinstance(hashtags, list):
            raise InvalidInputError("hashtags는 리스트여야 합니다.")

        existing = await asyncio.to_thread(self.collection.get, ids=[book.isbn])
        if existing.get("ids"):
            return ALREADY_EXISTS_MESSAGE

        embed_text = self._embed_text(book, main_category, sub_category, hashtags)
        embeddings = await self._embedder.embed_documents([embed_text])

        sub_category_str = ", ".join(sub_category)
        hashtags_str = ", ".join(hashtags)
        try:
            await asyncio.to_thread(
                self.collection.add,
                documents=[book.description],
                metadatas=[
                    {
                        "title": book.title,
                        "author": book.author,
                        "isbn": book.isbn,
                        "hashtags": hashtags_str,
                        "mainCategory": main_category,
                        "subCategory": sub_category_str,
                    }
                ],
                embeddings=[embeddings[0]],
                ids=[book.isbn],
            )
        except Exception as exc:
            raise ServiceUnavailableError(f"벡터 저장소에 책을 추가하지 못했습니다: {exc}") from exc
        return ADDED_MESSAGE

    async def delete_book(self, isbn: str) -> str:
        existing = await asyncio.to_thread(self.collection.get, ids=[isbn])
        if not existing.get("ids"):
            return NOT_FOUND_MESSAGE
        try:
            await asyncio.to_thread(self.collection.delete, ids=[isbn])
        except Exception as exc:
            raise ServiceUnavailableError(f"벡터 저장소에서 책을 삭제하지 못했습니다: {exc}") from exc
        return DELETED_MESSAGE

    async def get_book(self, isbn: str) -> dict[str, Any]:
        existing = await asyncio.to_thread(self.collection.get, ids=[isbn])
        if not existing.get("ids"):
            raise BookNotFoundError(NOT_FOUND_MESSAGE)
        metadata = (existing.get("metadatas") or [{}])[0] or {}
        documents = existing.get("documents") or [""]
        return {
            "title": metadata.get("title"),
            "author": metadata.get("author"),
            "isbn": metadata.get("isbn"),
            "description": documents[0],
            "hashtags": metadata.get("hashtags"),
            "mainCategory": metadata.get("mainCategory"),
            "subCategory": metadata.get("subCategory"),
        }

    async def find_candidates(
        self, embedding: list[float], n_results: int = 1
    ) -> list[dict[str, Any]]:
        """질의와 가까운 후보를 거리·임베딩과 함께 돌려준다.

        재정렬(`app.services.reranking`)이 후보들 사이의 순위를 다시 매기려면
        메타데이터만으로는 부족하다. 거리와 도서 임베딩을 함께 가져와, 임베딩이 있으면
        코사인을 직접 계산한다(거리 공간 설정에 의존하지 않기 위해).

        반환: `[{"metadata": {...}, "distance": float | None, "embedding": [...] | None}, ...]`
        """
        try:
            results = await asyncio.to_thread(
                self.collection.query,
                query_embeddings=[embedding],
                n_results=n_results,
                include=["metadatas", "distances", "embeddings"],
            )
        except Exception as exc:
            raise ServiceUnavailableError(f"벡터 검색에 실패했습니다: {exc}") from exc

        metadatas = (results.get("metadatas") or [[]])[0] or []
        distances = (results.get("distances") or [[]])[0] or []
        embeddings = (results.get("embeddings") or [[]])[0]

        records: list[dict[str, Any]] = []
        for index, metadata in enumerate(metadatas):
            distance = distances[index] if index < len(distances) else None
            vector = embeddings[index] if embeddings is not None and index < len(embeddings) else None
            records.append(
                {
                    "metadata": metadata or {},
                    "distance": float(distance) if distance is not None else None,
                    "embedding": [float(x) for x in vector] if vector is not None else None,
                }
            )
        return records

    async def find_nearest(self, embedding: list[float], n_results: int = 1) -> dict[str, Any] | None:
        """가장 가까운 책 1권의 메타데이터(기존 동작 유지)."""
        records = await self.find_candidates(embedding, n_results=n_results)
        if not records:
            return None
        return records[0]["metadata"]
