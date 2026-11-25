"""도메인 모델."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    isbn: str
    title: str
    author: str
    publisher: str
    publish_year: str
    cover_url: str
    description: str


@dataclass(frozen=True)
class BestSeller:
    book: Book
    best_rank: int

    @property
    def isbn(self) -> str:
        return self.book.isbn
