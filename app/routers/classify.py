"""카테고리 분류 API."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends

from app.core.errors import UpstreamServiceError
from app.deps import provide_classifier, provide_crawler
from app.schemas import ClassifyMessageRequest, ClassifyResponse, ProblemResponse
from app.services.classifier import CategoryClassifier
from app.services.kyobo import KyoboHashtagCrawler

logger = logging.getLogger("arabook.routers.classify")

router = APIRouter(tags=["classify"])


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    responses={502: {"model": ProblemResponse, "description": "LLM 호출 실패"}},
    summary="도서 정보로 대분류/소분류 분류",
)
async def classify(
    request: ClassifyMessageRequest,
    crawler: KyoboHashtagCrawler = Depends(provide_crawler),
    classifier: CategoryClassifier = Depends(provide_classifier),
) -> ClassifyResponse:
    # 해시태그는 분류 정확도를 돕는 참고 정보일 뿐이므로, 수집 실패가 분류 자체를
    # 막지는 않는다(기존 동작과 동일). 실패는 로그로 남긴다.
    try:
        hashtags = await asyncio.to_thread(crawler.fetch_hashtags, request.isbn)
    except UpstreamServiceError as exc:
        logger.warning("해시태그 수집 실패, 해시태그 없이 분류합니다: %s", exc)
        hashtags = []

    main_category, sub_categories = await classifier.classify(
        title=request.title,
        author=request.author,
        isbn=request.isbn,
        description=request.description,
        hashtags=hashtags,
    )
    return ClassifyResponse(
        title=request.title, main_category=main_category, sub_category=sub_categories
    )
