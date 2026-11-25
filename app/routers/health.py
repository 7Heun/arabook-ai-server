"""헬스체크 (기존 API에 없던 추가 엔드포인트)."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health", summary="서버 상태 확인")
async def health() -> dict[str, str]:
    return {"status": "ok"}
