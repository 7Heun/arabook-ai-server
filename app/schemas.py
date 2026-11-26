"""요청/응답 스키마.

필드 이름과 기본값은 기존 `main.py`의 모델을 그대로 유지했다. Spring 서버가
이 API를 호출하므로 요청 필드를 바꾸면 안 된다. 검증(`min_length`, `ge`)만
추가해, 비어 있는 입력이 500 대신 422로 떨어지게 했다.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateMessageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="생성할 텍스트 지시문")
    model: str = Field("gpt-4", description="사용할 OpenAI 모델")
    max_tokens: int = Field(100, ge=1, le=4096)


class ClassifyMessageRequest(BaseModel):
    title: str = Field(..., min_length=1)
    author: str = Field("", description="저자 (없으면 빈 문자열)")
    isbn: str = Field(..., min_length=1)
    description: str = Field("", description="도서 설명")


class CrawlMessageRequest(BaseModel):
    isbn: str = Field(..., min_length=1)
    # 기존 클라이언트가 보내는 필드라 남겨 두었다. 현재 핸들러에서는 사용하지 않는다.
    max_tokens: int = Field(100, ge=1, le=4096)


class AddBookRequest(BaseModel):
    isbn: str = Field(..., min_length=1)


class RecommendRequest(BaseModel):
    member_id: int = Field(..., description="회원 ID")


class GenerateResponse(BaseModel):
    generated_text: str


class ClassifyResponse(BaseModel):
    title: str
    main_category: str
    sub_category: list[str]


class CrawlResponse(BaseModel):
    isbn: str
    hashtags: list[str]


class MessageResponse(BaseModel):
    message: str


class BookResponse(BaseModel):
    title: str | None = None
    author: str | None = None
    isbn: str | None = None
    description: str | None = None
    hashtags: str | None = Field(None, description="쉼표로 구분된 해시태그 문자열")
    mainCategory: str | None = None
    subCategory: str | None = None


class ProblemResponse(BaseModel):
    detail: str
