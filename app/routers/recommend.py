"""도서 추천 API.

`/recommend`는 Spring 서버가 그대로 보던 1권 응답(`Recommended book ID: N`)을
유지한다. 상위 k권이 필요한 클라이언트를 위해 `/recommendations`를 따로 두었고,
기존 응답 스키마는 건드리지 않았다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import provide_recommendation_service
from app.schemas import (
    MessageResponse,
    ProblemResponse,
    RecommendationItem,
    RecommendationListRequest,
    RecommendationListResponse,
    RecommendRequest,
)
from app.services.recommendation import RecommendationService

router = APIRouter(tags=["recommend"])

_ERRORS = {503: {"model": ProblemResponse, "description": "DB/벡터 저장소 오류"}}


@router.post(
    "/recommend",
    response_model=MessageResponse,
    responses=_ERRORS,
    summary="회원 선호 카테고리 기반 도서 추천 (1권, 기존 응답 유지)",
)
async def recommend(
    request: RecommendRequest,
    service: RecommendationService = Depends(provide_recommendation_service),
) -> MessageResponse:
    return MessageResponse(message=await service.recommend(request.member_id))


@router.post(
    "/recommendations",
    response_model=RecommendationListResponse,
    responses=_ERRORS,
    summary="회원 선호 카테고리 기반 상위 k권 추천 (하이브리드 + MMR 다양성)",
)
async def recommendations(
    request: RecommendationListRequest,
    service: RecommendationService = Depends(provide_recommendation_service),
) -> RecommendationListResponse:
    """상위 k권을 점수·순위와 함께 돌려준다.

    `/recommend`와 같은 재정렬 규칙을 쓰되 1권으로 자르지 않으므로, MMR 다양성
    (같은 서브카테고리 중복 완화)이 2순위 이하에서 드러난다. 추천 이력은 남기지
    않는다 — 하루 1권 이력(`daily_book_id_recommendations`)은 `/recommend`와
    스케줄러가 쓰는 계약이라, 목록 조회가 그 테이블을 채우면 Spring 서버가 보는
    \"오늘의 추천\"이 달라진다.
    """
    result = await service.recommend_many(request.member_id, limit=request.limit)
    return RecommendationListResponse(
        member_id=request.member_id,
        limit=result.limit,
        preferences=result.preferences,
        recommendations=[
            RecommendationItem(
                rank=item.rank,
                book_id=item.book_id,
                isbn=item.isbn,
                title=item.title,
                author=item.author,
                main_category=item.main_category,
                sub_categories=item.sub_categories,
                score=item.score,
                similarity=item.similarity,
                category_overlap=item.category_overlap,
            )
            for item in result.items
        ],
        excluded_isbn_count=result.excluded_isbn_count,
        excluded_candidate_count=result.excluded_candidate_count,
        exclusion_relaxed=result.exclusion_relaxed,
        message=result.message,
    )
