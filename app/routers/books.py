"""도서 등록/삭제/조회 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import provide_ingestion_service, provide_vector_store
from app.schemas import AddBookRequest, BookResponse, MessageResponse, ProblemResponse
from app.services.library import BookIngestionService
from app.services.vector_store import BookVectorStore

router = APIRouter(tags=["books"])


@router.post(
    "/add-book",
    response_model=MessageResponse,
    responses={
        404: {"model": ProblemResponse, "description": "알라딘에서 도서를 찾지 못함"},
        502: {"model": ProblemResponse, "description": "외부 서비스 실패"},
    },
    summary="ISBN으로 도서를 조회·분류해 MySQL/Chroma에 저장",
)
async def add_book(
    request: AddBookRequest,
    ingestion: BookIngestionService = Depends(provide_ingestion_service),
) -> MessageResponse:
    return MessageResponse(message=await ingestion.ensure_book(request.isbn))


@router.post(
    "/delete-book",
    response_model=MessageResponse,
    responses={503: {"model": ProblemResponse, "description": "벡터 저장소 오류"}},
    summary="Chroma에서 도서 삭제",
)
async def delete_book(
    request: AddBookRequest,
    store: BookVectorStore = Depends(provide_vector_store),
) -> MessageResponse:
    # 기존 동작 유지: 없는 책이면 200 + "존재하지 않는 책 정보입니다."
    return MessageResponse(message=await store.delete_book(request.isbn))


@router.get(
    "/get-book",
    response_model=BookResponse,
    responses={404: {"model": ProblemResponse, "description": "도서 없음"}},
    summary="Chroma에서 도서 정보 조회",
)
async def get_book(
    isbn: str,
    store: BookVectorStore = Depends(provide_vector_store),
) -> BookResponse:
    book = await store.get_book(isbn)
    return BookResponse(**book)
