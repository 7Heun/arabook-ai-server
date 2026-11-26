"""도메인 예외와 HTTP 매핑.

기존 코드는 모든 핸들러가 `except Exception`으로 잡아 500 + `str(e)`를 그대로
내보냈다. 그러면 ① 어떤 실패인지 구분할 수 없고 ② DB 접속 정보 같은 내부
문자열이 응답에 섞일 수 있다. 여기서는 실패를 종류별로 나누고, 클라이언트에는
사람이 읽을 수 있는 메시지만, 로그에는 원인(exception chain)을 남긴다.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("arabook.errors")


class AppError(Exception):
    """서비스 계층에서 던지는 모든 도메인 예외의 상위 타입."""

    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class BookNotFoundError(AppError):
    status_code = 404
    code = "book_not_found"


class InvalidInputError(AppError):
    status_code = 422
    code = "invalid_input"


class UpstreamServiceError(AppError):
    """알라딘/교보/OpenAI 등 외부 서비스 실패."""

    status_code = 502
    code = "upstream_error"


class LanguageModelError(UpstreamServiceError):
    code = "llm_error"


class ServiceUnavailableError(AppError):
    """DB/벡터 저장소 같은 내부 의존성 사용 불가."""

    status_code = 503
    code = "service_unavailable"


class ConfigurationError(AppError):
    status_code = 500
    code = "configuration_error"


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        # 5xx는 서버 문제이므로 원인을 로그에 남기고, 응답에는 노출하지 않는다.
        if exc.status_code >= 500:
            logger.error(
                "%s %s -> %s(%s): %s",
                request.method,
                request.url.path,
                exc.code,
                exc.status_code,
                exc.message,
                exc_info=exc.__cause__ is not None,
            )
        else:
            logger.info("%s %s -> %s(%s): %s", request.method, request.url.path, exc.code, exc.status_code, exc.message)
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("처리되지 않은 예외: %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "서버 내부 오류가 발생했습니다."},
        )
