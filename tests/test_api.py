"""엔드포인트 계약 테스트 (외부 의존성은 전부 가짜).

각 테스트는 ① 성공 응답의 키/형태가 기존 API와 같은지, ② 실패가 적절한 상태
코드로 매핑되는지를 확인한다.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.errors import (
    BookNotFoundError,
    LanguageModelError,
    ServiceUnavailableError,
    UpstreamServiceError,
)
from app.deps import (
    provide_classifier,
    provide_crawler,
    provide_ingestion_service,
    provide_llm,
    provide_recommendation_service,
    provide_vector_store,
)
from app.services.recommendation import NO_PREFERENCES, RecommendationService
from app.services.vector_store import ADDED_MESSAGE, DELETED_MESSAGE


class FakeLLM:
    def __init__(self, text: str = "생성된 텍스트", error: Exception | None = None):
        self.text = text
        self.error = error

    async def generate(self, prompt: str, model: str | None = None, max_tokens: int = 100) -> str:
        if self.error:
            raise self.error
        return self.text


class FakeCrawler:
    def __init__(self, hashtags: list[str] | None = None, error: Exception | None = None):
        self.hashtags = hashtags or ["스릴러", "장편소설"]
        self.error = error

    def fetch_hashtags(self, isbn: str) -> list[str]:
        if self.error:
            raise self.error
        return self.hashtags


class FakeClassifier:
    def __init__(self, result=("소설", ["추리/스릴러"]), error: Exception | None = None):
        self.result = result
        self.error = error

    async def classify(self, **kwargs):
        if self.error:
            raise self.error
        return self.result


class FakeStore:
    def __init__(self):
        self.stored: dict[str, dict] = {}

    async def add_book(self, book, main_category, sub_category, hashtags) -> str:
        self.stored[book.isbn] = {"book": book, "main": main_category, "subs": sub_category}
        return ADDED_MESSAGE

    async def delete_book(self, isbn: str) -> str:
        return DELETED_MESSAGE

    async def get_book(self, isbn: str) -> dict:
        if isbn not in BOOK_PAYLOAD:
            raise BookNotFoundError("존재하지 않는 책 정보입니다.")
        return BOOK_PAYLOAD[isbn]

    async def find_nearest(self, embedding, n_results: int = 1):
        return {"isbn": "9791131963746", "title": "유토피아"}


BOOK_PAYLOAD = {
    "9791131963746": {
        "title": "유토피아",
        "author": "김유진 (지은이)",
        "isbn": "9791131963746",
        "description": "설명",
        "hashtags": "스릴러, 장편소설",
        "mainCategory": "소설",
        "subCategory": "추리/스릴러",
    }
}


class FakeIngestion:
    def __init__(self, message: str = ADDED_MESSAGE, error: Exception | None = None):
        self.message = message
        self.error = error

    async def ensure_book(self, isbn: str) -> str:
        if self.error:
            raise self.error
        return self.message


class FakeRepository:
    def __init__(self, preferences=None, book_id=42, error: Exception | None = None):
        self.preferences = ["추리/스릴러"] if preferences is None else preferences
        self.book_id = book_id
        self.error = error
        self.saved: list[tuple] = []

    def get_member_preferences(self, member_id):
        if self.error:
            raise self.error
        return self.preferences

    def get_book_id_by_isbn(self, isbn):
        if self.error:
            raise self.error
        return self.book_id

    def save_recommendation(self, member_id, book_id, date):
        if self.error:
            raise self.error
        self.saved.append((member_id, book_id, date))


class FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


# --------------------------------------------------------------------- health
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ------------------------------------------------------------------- generate
def test_generate_success(client, override):
    override(provide_llm, FakeLLM("안녕하세요"))
    response = client.post(
        "/generate", json={"prompt": "인사말", "model": "gpt-4", "max_tokens": 50}
    )
    assert response.status_code == 200
    assert response.json() == {"generated_text": "안녕하세요"}


def test_generate_defaults_match_legacy_model(client, override):
    captured = {}

    class CapturingLLM(FakeLLM):
        async def generate(self, prompt, model=None, max_tokens=100):
            captured["model"] = model
            captured["max_tokens"] = max_tokens
            return "ok"

    override(provide_llm, CapturingLLM())
    response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 200
    # 기존 기본값 유지 확인: model=gpt-4, max_tokens=100
    assert captured == {"model": "gpt-4", "max_tokens": 100}


def test_generate_rejects_empty_prompt(client, override):
    override(provide_llm, FakeLLM())
    response = client.post("/generate", json={"prompt": ""})
    assert response.status_code == 422


def test_generate_llm_failure_is_502(client, override):
    override(provide_llm, FakeLLM(error=LanguageModelError("OpenAI 오류")))
    response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 502
    assert "detail" in response.json()


# ------------------------------------------------------------------- classify
def test_classify_success_shape(client, override):
    override(provide_crawler, FakeCrawler())
    override(provide_classifier, FakeClassifier())
    response = client.post(
        "/classify",
        json={"title": "유토피아", "author": "김유진", "isbn": "9791131963746", "description": "설명"},
    )
    assert response.status_code == 200
    assert response.json() == {
        "title": "유토피아",
        "main_category": "소설",
        "sub_category": ["추리/스릴러"],
    }


def test_classify_survives_hashtag_failure(client, override):
    override(provide_crawler, FakeCrawler(error=UpstreamServiceError("교보 오류")))
    override(provide_classifier, FakeClassifier())
    response = client.post(
        "/classify",
        json={"title": "유토피아", "author": "김유진", "isbn": "9791131963746", "description": "설명"},
    )
    assert response.status_code == 200


def test_classify_llm_failure_is_502(client, override):
    override(provide_crawler, FakeCrawler())
    override(provide_classifier, FakeClassifier(error=LanguageModelError("형식 오류")))
    response = client.post(
        "/classify",
        json={"title": "t", "author": "a", "isbn": "1", "description": "d"},
    )
    assert response.status_code == 502


# ---------------------------------------------------------------------- crawl
def test_crawl_success(client, override):
    override(provide_crawler, FakeCrawler(["스릴러"]))
    response = client.post("/crawl", json={"isbn": "9791131963746"})
    assert response.status_code == 200
    assert response.json() == {"isbn": "9791131963746", "hashtags": ["스릴러"]}


def test_crawl_upstream_failure_is_502(client, override):
    override(provide_crawler, FakeCrawler(error=UpstreamServiceError("교보 오류")))
    response = client.post("/crawl", json={"isbn": "9791131963746"})
    assert response.status_code == 502


# ------------------------------------------------------------------ add-book
def test_add_book_success(client, override):
    override(provide_ingestion_service, FakeIngestion())
    response = client.post("/add-book", json={"isbn": "9791131963746"})
    assert response.status_code == 200
    assert response.json() == {"message": ADDED_MESSAGE}


def test_add_book_not_found_is_404(client, override):
    override(provide_ingestion_service, FakeIngestion(error=BookNotFoundError("없음")))
    response = client.post("/add-book", json={"isbn": "000"})
    assert response.status_code == 404


def test_add_book_requires_isbn(client):
    response = client.post("/add-book", json={})
    assert response.status_code == 422


# ------------------------------------------------------- delete/get-book
def test_delete_book_message(client, override):
    override(provide_vector_store, FakeStore())
    response = client.post("/delete-book", json={"isbn": "9791131963746"})
    assert response.status_code == 200
    assert response.json() == {"message": DELETED_MESSAGE}


def test_get_book_shape_matches_legacy(client, override):
    override(provide_vector_store, FakeStore())
    response = client.get("/get-book", params={"isbn": "9791131963746"})
    assert response.status_code == 200
    # 기존 응답과 같은 키 집합이어야 한다 (Spring 서버가 이 형태를 읽는다)
    assert set(response.json()) == {
        "title",
        "author",
        "isbn",
        "description",
        "hashtags",
        "mainCategory",
        "subCategory",
    }


def test_get_book_missing_is_404(client, override):
    override(provide_vector_store, FakeStore())
    response = client.get("/get-book", params={"isbn": "없는책"})
    assert response.status_code == 404


# -------------------------------------------------------------------- recommend
def test_recommend_success(client, override):
    store = FakeStore()
    repository = FakeRepository()
    override(
        provide_recommendation_service,
        RecommendationService(store, repository, FakeEmbedder()),
    )
    response = client.post("/recommend", json={"member_id": 1})
    assert response.status_code == 200
    assert response.json() == {"message": "Recommended book ID: 42"}
    assert repository.saved and repository.saved[0][0] == 1


def test_recommend_without_preferences_keeps_legacy_message(client, override):
    override(
        provide_recommendation_service,
        RecommendationService(FakeStore(), FakeRepository(preferences=[]), FakeEmbedder()),
    )
    response = client.post("/recommend", json={"member_id": 1})
    assert response.status_code == 200
    assert response.json() == {"message": NO_PREFERENCES}


def test_recommend_db_failure_is_503(client, override):
    override(
        provide_recommendation_service,
        RecommendationService(
            FakeStore(),
            FakeRepository(error=ServiceUnavailableError("DB 연결 실패")),
            FakeEmbedder(),
        ),
    )
    response = client.post("/recommend", json={"member_id": 1})
    assert response.status_code == 503


def test_recommend_requires_member_id(client):
    response = client.post("/recommend", json={})
    assert response.status_code == 422


# ------------------------------------------------------------- 공통 오류 처리
def test_unexpected_error_does_not_leak_internals(app):
    from app.deps import provide_llm

    async def boom(*args, **kwargs):
        raise RuntimeError("password=hunter2")

    class BoomLLM:
        generate = staticmethod(boom)

    app.dependency_overrides[provide_llm] = lambda: BoomLLM()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/generate", json={"prompt": "hi"})
    assert response.status_code == 500
    assert "hunter2" not in response.text
    assert response.json() == {"detail": "서버 내부 오류가 발생했습니다."}


def test_camel_case_keys_preserved(app):
    """Pydantic 모델을 거쳐도 기존 camelCase 응답 키가 유지되는지 확인."""
    paths = app.openapi()["paths"]
    schema = app.openapi()["components"]["schemas"]["BookResponse"]["properties"]
    assert {"mainCategory", "subCategory"} <= set(schema)
    assert "/get-book" in paths
