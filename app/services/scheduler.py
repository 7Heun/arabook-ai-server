"""주기 작업 (매일 추천 생성, 매달 베스트셀러 갱신).

기존 `scheduler.py`는 모듈 import 시점에 ChromaManager를 만들고, lifespan에서
무조건 스케줄러를 시작했다. 워커를 2개 이상 띄우면 같은 작업이 워커 수만큼
중복 실행된다(추천 이력 중복 저장, 베스트셀러 테이블 delete/insert 경쟁).

바뀐 점
- `SCHEDULER_ENABLED=false`로 끌 수 있다(여러 워커를 띄우는 배포에서는 스케줄러를
  별도 프로세스로 두거나 한 워커에서만 켠다).
- `max_instances=1`, `coalesce=True`로 같은 작업의 겹침/몰림 실행을 막는다.
- 베스트셀러 조회가 실패하면 테이블을 비우지 않고 작업을 실패로 끝낸다
  (기존에는 빈 목록으로 delete만 수행해 데이터가 날아갈 수 있었다).
"""

from __future__ import annotations

import asyncio
import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.errors import AppError
from app.services.aladin import AladinClient
from app.services.database import BookRepository
from app.services.library import BookIngestionService
from app.services.recommendation import RecommendationService

logger = logging.getLogger("arabook.scheduler")


class ScheduledJobs:
    def __init__(
        self,
        repository: BookRepository,
        recommendation: RecommendationService,
        ingestion: BookIngestionService,
        aladin: AladinClient,
    ):
        self._repository = repository
        self._recommendation = recommendation
        self._ingestion = ingestion
        self._aladin = aladin

    async def recommend_for_all_members(self) -> None:
        member_ids = await asyncio.to_thread(self._repository.get_all_member_ids)
        logger.info("전체 회원 추천 시작: %d명", len(member_ids))
        succeeded = 0
        failed = 0
        for member_id in member_ids:
            try:
                await self._recommendation.recommend(member_id)
                succeeded += 1
            except AppError as exc:
                failed += 1
                logger.warning("회원 %s 추천 실패: %s", member_id, exc)
        logger.info("전체 회원 추천 완료: 성공 %d, 실패 %d", succeeded, failed)

    async def update_best_sellers(self) -> None:
        best_sellers = await asyncio.to_thread(self._aladin.get_best_sellers)
        if not best_sellers:
            logger.warning("베스트셀러 목록이 비어 있어 갱신을 건너뜁니다.")
            return

        rows: list[dict[str, object]] = []
        for best_seller in best_sellers:
            isbn = best_seller.isbn
            rows.append({"isbn": isbn, "best_rank": best_seller.best_rank})
            try:
                if await self._is_known(isbn):
                    logger.info("이미 저장된 책입니다: %s", isbn)
                    continue
                await self._ingestion.ensure_book(isbn)
            except AppError as exc:
                # 한 권이 실패해도 나머지 순위는 저장한다.
                logger.warning("베스트셀러 %s 적재 실패: %s", isbn, exc)

        await self._repository.insert_best_sellers(rows)
        logger.info("베스트셀러 %d건 갱신 완료", len(rows))

    async def _is_known(self, isbn: str) -> bool:
        return bool(await asyncio.to_thread(self._repository.get_book_id_by_isbn, isbn))


def build_scheduler(jobs: ScheduledJobs) -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        jobs.recommend_for_all_members,
        CronTrigger(hour=0, minute=1),
        id="daily_recommendations",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    scheduler.add_job(
        jobs.update_best_sellers,
        CronTrigger(day=1, hour=0, minute=0),
        id="monthly_best_sellers",
        max_instances=1,
        coalesce=True,
        misfire_grace_time=3600,
    )
    return scheduler
