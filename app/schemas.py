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


class RecommendationListRequest(BaseModel):
    member_id: int = Field(..., description="회원 ID")
    limit: int = Field(5, ge=1, le=20, description="추천 권수 (1~20)")


class RecommendationItem(BaseModel):
    rank: int = Field(..., description="순위 (1부터)")
    book_id: int | None = Field(None, description="MySQL book_id (books에 없으면 null)")
    isbn: str
    title: str
    author: str | None = None
    main_category: str | None = None
    sub_categories: list[str] = Field(default_factory=list, description="서브카테고리 목록")
    score: float = Field(..., description="하이브리드 관련도 = 코사인 + 가중치 × 카테고리 일치율")
    similarity: float = Field(..., description="질의-도서 코사인 유사도")
    category_overlap: float = Field(..., description="선호 서브카테고리 일치율 (0~1)")


class RecommendationListResponse(BaseModel):
    member_id: int
    limit: int
    preferences: list[str] = Field(default_factory=list, description="회원이 고른 서브카테고리")
    recommendations: list[RecommendationItem] = Field(default_factory=list)
    excluded_isbn_count: int = Field(0, description="제외 목록 크기 (최근 추천 이력)")
    excluded_candidate_count: int = Field(0, description="제외로 후보에서 빠진 권수")
    exclusion_relaxed: bool = Field(
        False, description="후보가 전부 제외되어 제외를 풀고 추천한 경우 true"
    )
    message: str | None = Field(None, description="추천을 만들지 못했을 때의 사유")


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
