"""도서 추천 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.deps import provide_recommendation_service
from app.schemas import MessageResponse, ProblemResponse, RecommendRequest
from app.services.recommendation import RecommendationService

router = APIRouter(tags=["recommend"])


@router.post(
    "/recommend",
    response_model=MessageResponse,
    responses={503: {"model": ProblemResponse, "description": "DB/벡터 저장소 오류"}},
    summary="회원 선호 카테고리 기반 도서 추천",
)
async def recommend(
    request: RecommendRequest,
    service: RecommendationService = Depends(provide_recommendation_service),
) -> MessageResponse:
    return MessageResponse(message=await service.recommend(request.member_id))
