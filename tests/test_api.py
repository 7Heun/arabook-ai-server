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
    def __init__(self, records: list[dict] | None = None):
        self.stored: dict[str, dict] = {}
        self._records = records

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

    async def find_candidates(self, embedding, n_results: int = 1):
        if self._records is not None:
            return self._records
        return [
            {
                "metadata": {
                    "isbn": "9791131963746",
                    "title": "유토피아",
                    "subCategory": "추리/스릴러",
                },
                "distance": 0.1,
                "embedding": None,
            }
        ]


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
    def __init__(
        self,
        preferences=None,
        book_id=42,
        error: Exception | None = None,
        excluded: list[str] | None = None,
        book_ids: dict[str, int] | None = None,
    ):
        self.preferences = ["추리/스릴러"] if preferences is None else preferences
        self.book_id = book_id
        self.error = error
        self.excluded = list(excluded or [])
        self.book_ids = dict(book_ids or {})
        self.saved: list[tuple] = []
        self.exclusion_lookups: list[tuple] = []

    def get_member_preferences(self, member_id):
        if self.error:
            raise self.error
        return self.preferences

    def get_book_id_by_isbn(self, isbn):
        if self.error:
            raise self.error
        return self.book_ids.get(isbn, self.book_id)

    def get_book_ids_by_isbns(self, isbns):
        if self.error:
            raise self.error
        return {
            isbn: self.book_ids.get(isbn, self.book_id)
            for isbn in isbns
            if self.book_ids.get(isbn, self.book_id) is not None
        }

    def get_recent_recommended_isbns(self, member_id, days=30):
        if self.error:
            raise self.error
        self.exclusion_lookups.append((member_id, days))
        return list(self.excluded)

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


# --------------------------------------------------------------- recommendations
RECOMMENDATION_RECORDS = [
    {
        "metadata": {
            "isbn": "a",
            "title": "A",
            "author": "저자A",
            "mainCategory": "소설",
            "subCategory": "추리/스릴러",
        },
        "distance": 0.10,
        "embedding": None,
    },
    {
        "metadata": {"isbn": "b", "title": "B", "subCategory": "추리/스릴러"},
        "distance": 0.14,
        "embedding": None,
    },
    {
        "metadata": {"isbn": "c", "title": "C", "mainCategory": "과학", "subCategory": "SF"},
        "distance": 0.16,
        "embedding": None,
    },
]


def _list_service(store=None, repository=None) -> RecommendationService:
    return RecommendationService(
        store or FakeStore(RECOMMENDATION_RECORDS),
        repository or FakeRepository(book_ids={"a": 1, "b": 2, "c": 3}),
        FakeEmbedder(),
    )


def test_recommendations_returns_ranked_list_shape(client, override):
    override(provide_recommendation_service, _list_service())
    response = client.post("/recommendations", json={"member_id": 1, "limit": 3})
    assert response.status_code == 200

    body = response.json()
    assert body["member_id"] == 1
    assert body["limit"] == 3
    assert body["preferences"] == ["추리/스릴러"]
    assert body["excluded_isbn_count"] == 0
    assert body["excluded_candidate_count"] == 0
    assert body["exclusion_relaxed"] is False
    assert body["message"] is None
    # MMR: 동일 카테고리 b가 3순위로 밀리고 c가 2순위
    assert [item["isbn"] for item in body["recommendations"]] == ["a", "c", "b"]
    assert [item["rank"] for item in body["recommendations"]] == [1, 2, 3]
    assert [item["book_id"] for item in body["recommendations"]] == [1, 3, 2]
    assert set(body["recommendations"][0]) == {
        "rank",
        "book_id",
        "isbn",
        "title",
        "author",
        "main_category",
        "sub_categories",
        "score",
        "similarity",
        "category_overlap",
    }
    first = body["recommendations"][0]
    assert first["title"] == "A"
    assert first["author"] == "저자A"
    assert first["main_category"] == "소설"
    assert first["sub_categories"] == ["추리/스릴러"]
    assert first["category_overlap"] == 1.0
    assert first["score"] > first["similarity"]  # 카테고리 가중이 더해진다


def test_recommendations_default_limit_is_five(client, override):
    override(provide_recommendation_service, _list_service())
    body = client.post("/recommendations", json={"member_id": 1}).json()
    assert body["limit"] == 5
    assert len(body["recommendations"]) == 3  # 후보가 3권뿐


def test_recommendations_excludes_recent_books(client, override):
    repository = FakeRepository(book_ids={"a": 1, "b": 2, "c": 3}, excluded=["a"])
    override(provide_recommendation_service, _list_service(repository=repository))
    body = client.post("/recommendations", json={"member_id": 1, "limit": 3}).json()
    assert [item["isbn"] for item in body["recommendations"]] == ["b", "c"]
    assert body["excluded_isbn_count"] == 1
    assert body["excluded_candidate_count"] == 1
    assert body["exclusion_relaxed"] is False


def test_recommendations_marks_relaxed_exclusion_when_all_candidates_excluded(client, override):
    repository = FakeRepository(book_ids={"a": 1, "b": 2, "c": 3}, excluded=["a", "b", "c"])
    override(provide_recommendation_service, _list_service(repository=repository))
    body = client.post("/recommendations", json={"member_id": 1, "limit": 3}).json()
    assert body["exclusion_relaxed"] is True
    assert body["excluded_candidate_count"] == 3
    assert len(body["recommendations"]) == 3


def test_recommendations_without_preferences_returns_message(client, override):
    override(
        provide_recommendation_service,
        _list_service(repository=FakeRepository(preferences=[])),
    )
    body = client.post("/recommendations", json={"member_id": 1}).json()
    assert body["recommendations"] == []
    assert body["message"] == NO_PREFERENCES


def test_recommendations_rejects_out_of_range_limit(client, override):
    override(provide_recommendation_service, _list_service())
    assert client.post("/recommendations", json={"member_id": 1, "limit": 0}).status_code == 422
    assert client.post("/recommendations", json={"member_id": 1, "limit": 21}).status_code == 422
    assert client.post("/recommendations", json={"limit": 3}).status_code == 422


def test_recommendations_db_failure_is_503(client, override):
    override(
        provide_recommendation_service,
        _list_service(repository=FakeRepository(error=ServiceUnavailableError("DB 연결 실패"))),
    )
    response = client.post("/recommendations", json={"member_id": 1})
    assert response.status_code == 503


def test_recommend_excludes_recently_recommended_book(client, override):
    store = FakeStore(RECOMMENDATION_RECORDS)
    repository = FakeRepository(book_ids={"a": 1, "b": 2, "c": 3}, excluded=["a"])
    override(provide_recommendation_service, RecommendationService(store, repository, FakeEmbedder()))
    response = client.post("/recommend", json={"member_id": 1})
    assert response.json() == {"message": "Recommended book ID: 2"}


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
