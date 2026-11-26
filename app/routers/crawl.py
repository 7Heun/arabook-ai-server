"""해시태그 크롤링 API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends

from app.deps import provide_crawler
from app.schemas import CrawlMessageRequest, CrawlResponse, ProblemResponse
from app.services.kyobo import KyoboHashtagCrawler

router = APIRouter(tags=["crawl"])


@router.post(
    "/crawl",
    response_model=CrawlResponse,
    responses={502: {"model": ProblemResponse, "description": "교보문고 요청 실패"}},
    summary="ISBN으로 해시태그 수집",
)
async def crawl(
    request: CrawlMessageRequest,
    crawler: KyoboHashtagCrawler = Depends(provide_crawler),
) -> CrawlResponse:
    hashtags = await asyncio.to_thread(crawler.fetch_hashtags, request.isbn)
    return CrawlResponse(isbn=request.isbn, hashtags=hashtags)
