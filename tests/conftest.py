"""API 테스트 공통 설정.

OpenAI/Chroma/MySQL/크롤링은 모두 `app.deps`의 프로바이더를 통해 주입되므로,
테스트에서는 `app.dependency_overrides`로 가짜 구현을 넣는다. 따라서 API 키나
DB 없이 엔드포인트를 검증할 수 있다.
"""

from __future__ import annotations

import os

# app.main을 import하기 전에 환경을 고정한다 (스케줄러가 테스트 중 뜨지 않도록).
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ.pop("OPENAI_API_KEY", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402


@pytest.fixture
def app():
    from app.main import create_app

    get_settings.cache_clear()
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def override(app):
    """의존성 오버라이드를 등록하고 테스트 후 정리한다."""

    def _override(provider, value):
        app.dependency_overrides[provider] = lambda: value
        return value

    yield _override
    app.dependency_overrides.clear()
