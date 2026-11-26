"""FastAPI 앱 조립.

기존 `main.py` 한 파일(166줄)에 있던 엔드포인트를 라우터로, 서비스 로직을
`app/services/`로 나눴다. API 경로·요청 필드·응답 형태는 그대로 유지했다.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.routers import books, classify, crawl, generate, health, recommend
from app.services.scheduler import build_scheduler

logger = logging.getLogger("arabook.main")

DESCRIPTION = """
팀 프로젝트 **AraBook**의 AI 서버입니다.

- `/add-book`: ISBN → 알라딘 메타데이터 → 교보 해시태그 → LLM 카테고리 분류 →
  MySQL + Chroma 저장
- `/recommend`: 회원이 고른 서브카테고리로 Chroma에서 가장 가까운 책 1권 추천
- `/classify`, `/crawl`, `/generate`: 위 파이프라인의 개별 단계

추천 품질 측정 방법은 저장소의 `eval/README.md`를 참고하세요.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    scheduler = None
    if settings.scheduler_enabled:
        try:
            from app.deps import build_scheduler_jobs

            scheduler = build_scheduler(build_scheduler_jobs())
            scheduler.start()
            app.state.scheduler = scheduler
            logger.info("스케줄러 시작: 매일 00:01 추천, 매월 1일 00:00 베스트셀러 갱신")
            if not settings.database.configured:
                logger.warning(
                    "DB 접속 환경 변수가 없습니다. 스케줄러 작업은 실행 시점에 실패합니다."
                )
        except Exception:
            # 스케줄러가 없어도 API는 떠야 한다.
            logger.exception("스케줄러를 시작하지 못했습니다.")
    else:
        logger.info("SCHEDULER_ENABLED=false: 스케줄러를 시작하지 않습니다.")

    yield

    if scheduler is not None:
        scheduler.shutdown(wait=False)
        logger.info("스케줄러 종료")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="AraBook AI Server",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(generate.router)
    app.include_router(classify.router)
    app.include_router(crawl.router)
    app.include_router(books.router)
    app.include_router(recommend.router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
