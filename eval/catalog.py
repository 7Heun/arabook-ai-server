"""평가용 카탈로그 로딩.

추천 품질 평가에는 (1) 도서 메타데이터와 (2) 도서 임베딩이 필요하다.
운영 데이터는 Chroma(`books` 컬렉션)와 MySQL에 나뉘어 있는데, 오프라인 평가는
읽기만 하면 되므로 Chroma 스냅샷에서 바로 읽는다.

중요: `chromadb.PersistentClient`는 폴더에 마이그레이션/락 파일을 쓸 수 있다.
그래서 원본 경로를 직접 열지 않고 **임시 폴더로 복사한 뒤** 연다. 커밋된
`chroma_db/`를 평가가 건드리지 않게 하려는 장치다.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class CatalogError(RuntimeError):
    """카탈로그를 읽을 수 없을 때."""


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


@dataclass(frozen=True)
class CatalogItem:
    isbn: str
    title: str
    author: str
    main_category: str
    sub_categories: tuple[str, ...]
    hashtags: tuple[str, ...]
    description: str = ""
    embedding: tuple[float, ...] | None = field(default=None, repr=False, compare=False)

    @classmethod
    def from_metadata(
        cls,
        isbn: str,
        metadata: dict[str, Any],
        document: str = "",
        embedding: tuple[float, ...] | None = None,
    ) -> "CatalogItem":
        return cls(
            isbn=str(metadata.get("isbn") or isbn),
            title=str(metadata.get("title") or ""),
            author=str(metadata.get("author") or ""),
            main_category=str(metadata.get("mainCategory") or ""),
            sub_categories=_split_csv(metadata.get("subCategory")),
            hashtags=_split_csv(metadata.get("hashtags")),
            description=document or "",
            embedding=embedding,
        )


@dataclass
class Catalog:
    items: list[CatalogItem]
    source: str
    collection: str | None = None
    dimension: int | None = None

    def __post_init__(self) -> None:
        self._by_isbn = {item.isbn: item for item in self.items}

    @property
    def isbns(self) -> list[str]:
        return [item.isbn for item in self.items]

    def get(self, isbn: str) -> CatalogItem | None:
        return self._by_isbn.get(isbn)

    def categories_by_isbn(self) -> dict[str, Sequence[str]]:
        return {item.isbn: item.sub_categories for item in self.items}

    @property
    def has_embeddings(self) -> bool:
        return bool(self.items) and all(item.embedding is not None for item in self.items)

    def embedding_matrix(self):
        """numpy 배열 (n_items x dim). 임베딩이 없으면 CatalogError."""
        import numpy as np

        if not self.has_embeddings:
            raise CatalogError(
                "카탈로그에 임베딩이 없습니다. chroma_db에서 읽거나 임베딩이 포함된 "
                "카탈로그 JSON을 사용하세요."
            )
        return np.asarray([item.embedding for item in self.items], dtype="float32")

    def to_json(self, include_embeddings: bool = False) -> dict[str, Any]:
        return {
            "meta": {
                "source": self.source,
                "collection": self.collection,
                "dimension": self.dimension,
                "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "includes_embeddings": include_embeddings,
                "n_items": len(self.items),
                "note": "메타데이터 스냅샷입니다. 임베딩은 기본적으로 포함하지 않습니다.",
            },
            "items": [
                {
                    "isbn": item.isbn,
                    "title": item.title,
                    "author": item.author,
                    "main_category": item.main_category,
                    "sub_categories": list(item.sub_categories),
                    "hashtags": list(item.hashtags),
                    "description": item.description,
                    **(
                        {"embedding": list(item.embedding)}
                        if include_embeddings and item.embedding is not None
                        else {}
                    ),
                }
                for item in self.items
            ],
        }


def load_catalog_from_json(path: str | Path) -> Catalog:
    path = Path(path)
    if not path.exists():
        raise CatalogError(f"카탈로그 파일이 없습니다: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = [
        CatalogItem(
            isbn=str(raw["isbn"]),
            title=raw.get("title", ""),
            author=raw.get("author", ""),
            main_category=raw.get("main_category", ""),
            sub_categories=tuple(raw.get("sub_categories") or ()),
            hashtags=tuple(raw.get("hashtags") or ()),
            description=raw.get("description", ""),
            embedding=tuple(raw["embedding"]) if raw.get("embedding") else None,
        )
        for raw in payload.get("items", [])
    ]
    if not items:
        raise CatalogError(f"카탈로그에 항목이 없습니다: {path}")
    meta = payload.get("meta", {})
    return Catalog(
        items=items,
        source=str(path),
        collection=meta.get("collection"),
        dimension=meta.get("dimension"),
    )


def load_catalog_from_chroma(
    chroma_path: str | Path,
    collection_name: str = "books",
    copy_before_open: bool = True,
) -> Catalog:
    """Chroma 폴더에서 카탈로그를 읽는다.

    copy_before_open=True(기본)이면 임시 폴더에 복사한 사본을 연다. 원본 폴더에
    아무것도 쓰지 않기 위한 기본값이다.
    """
    chroma_path = Path(chroma_path)
    if not chroma_path.exists():
        raise CatalogError(f"Chroma 폴더가 없습니다: {chroma_path}")
    if not (chroma_path / "chroma.sqlite3").exists():
        raise CatalogError(f"Chroma 폴더에 chroma.sqlite3가 없습니다: {chroma_path}")

    try:
        import chromadb  # 지역 import: 평가 CLI에서만 필요
    except ImportError as exc:  # pragma: no cover - 환경 의존
        raise CatalogError("chromadb가 설치되어 있지 않습니다: pip install -r requirements.txt") from exc

    temp_dir: str | None = None
    target = chroma_path
    if copy_before_open:
        temp_dir = tempfile.mkdtemp(prefix="arabook-chroma-")
        target = Path(temp_dir) / "chroma_db"
        shutil.copytree(chroma_path, target)

    client = chromadb.PersistentClient(path=str(target))
    try:
        try:
            collection = client.get_collection(collection_name)
        except Exception as exc:  # chroma는 컬렉션이 없으면 예외를 던진다
            raise CatalogError(
                f"컬렉션 '{collection_name}'을(를) 찾을 수 없습니다: {exc}"
            ) from exc

        payload = collection.get(include=["metadatas", "documents", "embeddings"])
        ids: list[str] = list(payload.get("ids") or [])
        metadatas: list[dict[str, Any]] = list(payload.get("metadatas") or [])
        documents: list[str] = list(payload.get("documents") or [])
        embeddings = payload.get("embeddings")
        if embeddings is None:
            embeddings = [None] * len(ids)

        items: list[CatalogItem] = []
        dimension: int | None = None
        for index, raw_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            document = documents[index] if index < len(documents) and documents[index] else ""
            embedding = embeddings[index] if index < len(embeddings) else None
            if embedding is not None:
                embedding = tuple(float(x) for x in embedding)
                dimension = dimension or len(embedding)
            items.append(CatalogItem.from_metadata(raw_id, metadata, document, embedding))
    finally:
        # sqlite 핸들이 닫히도록 참조를 버리고 임시 폴더를 정리한다
        del client
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if not items:
        raise CatalogError(f"컬렉션 '{collection_name}'이(가) 비어 있습니다: {chroma_path}")
    return Catalog(items=items, source=f"chroma:{chroma_path}", collection=collection_name, dimension=dimension)


def catalog_statistics(catalog: Catalog) -> dict[str, Any]:
    """카탈로그 구성 통계.

    실제 저장된 데이터에서 계산한 값이며, 추천 품질 지표가 아니라 "이 카탈로그로
    무엇을 평가할 수 있는가"를 보여주는 근거로 쓴다.
    """
    main_categories = Counter(item.main_category for item in catalog.items)
    sub_counter: Counter[str] = Counter()
    hashtag_counter: Counter[str] = Counter()
    for item in catalog.items:
        sub_counter.update(item.sub_categories)
        hashtag_counter.update(item.hashtags)

    multi_valued_main = [
        item.isbn for item in catalog.items if "," in item.main_category
    ]
    no_sub_category = [item.isbn for item in catalog.items if not item.sub_categories]
    no_description = [item.isbn for item in catalog.items if not item.description.strip()]
    duplicate_isbns = [isbn for isbn, count in Counter(catalog.isbns).items() if count > 1]

    return {
        "n_items": len(catalog.items),
        "dimension": catalog.dimension,
        "has_embeddings": catalog.has_embeddings,
        "main_categories": main_categories.most_common(),
        "n_main_category_labels": len(main_categories),
        "multi_valued_main_category": multi_valued_main,
        "n_sub_category_labels": len(sub_counter),
        "top_sub_categories": sub_counter.most_common(10),
        "n_hashtags": len(hashtag_counter),
        "top_hashtags": hashtag_counter.most_common(10),
        "no_sub_category": no_sub_category,
        "no_description": no_description,
        "duplicate_isbns": duplicate_isbns,
        "sub_categories_per_book_mean": (
            sum(len(i.sub_categories) for i in catalog.items) / len(catalog.items)
            if catalog.items
            else 0.0
        ),
    }
