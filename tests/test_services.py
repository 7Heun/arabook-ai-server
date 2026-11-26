"""서비스 계층 테스트 (외부 연결 없이)."""

from __future__ import annotations

import pytest

from app.core.errors import BookNotFoundError, LanguageModelError, UpstreamServiceError
from app.domain import Book
from app.services.aladin import AladinClient
from app.services.classifier import CategoryClassifier
from app.services.kyobo import KyoboHashtagCrawler
from app.services.recommendation import NO_BOOK, NO_BOOK_ID, NO_PREFERENCES, RecommendationService
from app.services.reranking import RankingWeights
from app.services.scheduler import ScheduledJobs
from app.services.vector_store import (
    ADDED_MESSAGE,
    ALREADY_EXISTS_MESSAGE,
    BookVectorStore,
)

BOOK = Book(
    isbn="9791131963746",
    title="유토피아",
    author="김유진 (지은이)",
    publisher="출판사",
    publish_year="2024",
    cover_url="http://example.com/cover500.jpg",
    description="설명",
)


# ------------------------------------------------------------------ classifier
class TestClassifierParsing:
    @pytest.mark.parametrize(
        "text",
        [
            "소설 - 추리/스릴러, SF",
            "대분류 - 소설\n소분류 - 추리/스릴러, SF",
        ],
    )
    def test_parses_both_response_shapes(self, text):
        assert CategoryClassifier.parse(text) == ("소설", ["추리/스릴러", "SF"])

    def test_keeps_only_first_main_category(self):
        # 모델이 '소설, 역사'처럼 대분류를 두 개 주면 첫 번째만 남긴다.
        # (저장된 카탈로그 3권에서 실제로 발견된 문제)
        assert CategoryClassifier.parse("소설, 역사 - 사랑, 한국 근현대사") == (
            "소설",
            ["사랑", "한국 근현대사"],
        )

    def test_rejects_malformed_response(self):
        with pytest.raises(ValueError):
            CategoryClassifier.parse("카테고리를 알 수 없습니다")

    def test_empty_sub_categories_is_error(self):
        with pytest.raises(ValueError):
            CategoryClassifier.parse("소설 - ")


class TestClassifierService:
    async def test_classify_uses_llm_and_parses(self):
        class LLM:
            async def generate(self, prompt, model=None, max_tokens=100):
                assert "유토피아" in prompt
                return "소설 - 추리/스릴러"

        classifier = CategoryClassifier(LLM())
        assert await classifier.classify("유토피아", "김유진", "1", "설명", ["스릴러"]) == (
            "소설",
            ["추리/스릴러"],
        )

    async def test_bad_response_raises_language_model_error(self):
        class LLM:
            async def generate(self, prompt, model=None, max_tokens=100):
                return "모르겠습니다"

        with pytest.raises(LanguageModelError):
            await CategoryClassifier(LLM()).classify("t", "a", "1", "d", [])


# ---------------------------------------------------------------------- aladin
class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    def __init__(self, payload=None, error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.last_params: dict | None = None

    def get(self, url, params=None, timeout=None):
        self.last_params = params
        if self.error:
            raise self.error
        return FakeResponse(self.payload)


ALADIN_ITEM = {
    "isbn13": "9791131963746",
    "title": "유토피아",
    "author": "김유진 (지은이)",
    "publisher": "출판사",
    "pubDate": "2024-05-01",
    "cover": "http://example.com/coversum.jpg",
    "description": "설명",
    "bestRank": 3,
}


class TestAladinClient:
    def test_maps_item_fields(self):
        client = AladinClient("key", session=FakeSession({"item": [ALADIN_ITEM]}))
        book = client.get_book_by_isbn("9791131963746")
        assert book.publish_year == "2024"
        assert book.cover_url == "http://example.com/cover500.jpg"

    def test_missing_item_raises_not_found(self):
        client = AladinClient("key", session=FakeSession({"item": []}))
        with pytest.raises(BookNotFoundError):
            client.get_book_by_isbn("000")

    def test_request_error_raises_upstream(self):
        import requests

        client = AladinClient("key", session=FakeSession(error=requests.ConnectionError("down")))
        with pytest.raises(UpstreamServiceError):
            client.get_book_by_isbn("9791131963746")

    def test_missing_key_raises_upstream(self):
        client = AladinClient(None, session=FakeSession({"item": [ALADIN_ITEM]}))
        with pytest.raises(UpstreamServiceError):
            client.get_book_by_isbn("9791131963746")

    def test_best_sellers_skips_broken_items(self):
        broken = {"title": "깨진 항목"}
        client = AladinClient("key", session=FakeSession({"item": [ALADIN_ITEM, broken]}))
        result = client.get_best_sellers()
        assert len(result) == 1
        assert result[0].best_rank == 3


# ---------------------------------------------------------------------- kyobo
class TestKyoboHashtagCrawler:
    def test_parses_and_filters_standard_categories(self, tmp_path):
        categories = tmp_path / "categories.txt"
        categories.write_text("소설\n", encoding="utf-8")
        html = '<a class="tag">#스릴러</a><a class="tag">#소설</a><a class="tag">#스릴러</a>'

        class Response:
            text = html

            def raise_for_status(self):
                pass

        class Session:
            def get(self, url, params=None, timeout=None):
                return Response()

        crawler = KyoboHashtagCrawler(categories, session=Session())
        assert crawler.fetch_hashtags("1") == ["스릴러"]

    def test_missing_categories_file_does_not_fail(self, tmp_path):
        class Response:
            text = '<a class="tag">#스릴러</a>'

            def raise_for_status(self):
                pass

        class Session:
            def get(self, url, params=None, timeout=None):
                return Response()

        crawler = KyoboHashtagCrawler(tmp_path / "none.txt", session=Session())
        assert crawler.fetch_hashtags("1") == ["스릴러"]

    def test_request_error_raises_upstream(self):
        import requests

        class Session:
            def get(self, url, params=None, timeout=None):
                raise requests.ConnectionError("down")

        crawler = KyoboHashtagCrawler("categories.txt", session=Session())
        with pytest.raises(UpstreamServiceError):
            crawler.fetch_hashtags("1")


# ---------------------------------------------------------------- vector store
class FakeCollection:
    def __init__(self):
        self.records: dict[str, dict] = {}
        self.added: dict | None = None

    def get(self, ids):
        return {"ids": [i for i in ids if i in self.records]}

    def add(self, documents, metadatas, embeddings, ids):
        self.added = {"documents": documents, "metadatas": metadatas, "ids": ids}
        self.records[ids[0]] = metadatas[0]

    def delete(self, ids):
        for i in ids:
            self.records.pop(i, None)

    def query(self, query_embeddings, n_results):
        return {"metadatas": [[self.records[next(iter(self.records))]]], "documents": [["d"]]}


class FakeEmbedder:
    async def embed_documents(self, texts):
        assert texts and "title: 유토피아" in texts[0]
        return [[0.1, 0.2]]

    async def embed_query(self, text):
        return [0.1, 0.2]


class TestBookVectorStore:
    async def test_add_book_stores_legacy_metadata(self):
        collection = FakeCollection()
        store = BookVectorStore("unused", "books", FakeEmbedder(), collection=collection)
        message = await store.add_book(BOOK, "소설", ["추리/스릴러"], ["스릴러"])
        assert message == ADDED_MESSAGE
        metadata = collection.added["metadatas"][0]
        assert metadata["mainCategory"] == "소설"
        assert metadata["subCategory"] == "추리/스릴러"
        assert metadata["hashtags"] == "스릴러"
        assert collection.added["documents"] == ["설명"]

    async def test_duplicate_returns_existing_message(self):
        collection = FakeCollection()
        store = BookVectorStore("unused", "books", FakeEmbedder(), collection=collection)
        await store.add_book(BOOK, "소설", ["추리/스릴러"], [])
        message = await store.add_book(BOOK, "소설", ["추리/스릴러"], [])
        assert message == ALREADY_EXISTS_MESSAGE

    async def test_add_book_rejects_invalid_category(self):
        store = BookVectorStore("unused", "books", FakeEmbedder(), collection=FakeCollection())
        from app.core.errors import InvalidInputError

        with pytest.raises(InvalidInputError):
            await store.add_book(BOOK, None, [], [])

    async def test_get_book_missing_raises_404(self):
        store = BookVectorStore("unused", "books", FakeEmbedder(), collection=FakeCollection())
        with pytest.raises(BookNotFoundError):
            await store.get_book("없음")


# --------------------------------------------------------------- recommendation
class Repo:
    def __init__(self, preferences=None, book_id=7):
        self.preferences = ["추리/스릴러"] if preferences is None else preferences
        self.book_id = book_id
        self.saved = []

    def get_member_preferences(self, member_id):
        return self.preferences

    def get_book_id_by_isbn(self, isbn):
        return self.book_id

    def save_recommendation(self, member_id, book_id, date):
        self.saved.append((member_id, book_id))


_UNSET = object()


def _record(isbn, distance, sub_categories=("추리/스릴러",), embedding=None):
    return {
        "metadata": {
            "isbn": isbn,
            "title": f"book-{isbn}",
            "subCategory": ", ".join(sub_categories),
        },
        "distance": distance,
        "embedding": embedding,
    }


class Store:
    """`BookVectorStore.find_candidates`의 가짜 구현.

    거리(제곱 L2)만 주면 재정렬이 코사인으로 환산해 쓴다 (d = 2 - 2cos).
    """

    def __init__(self, metadata=_UNSET, records=None):
        self.metadata = {"isbn": "9791131963746"} if metadata is _UNSET else metadata
        self.records = records
        self.requested: list[int] = []

    async def find_candidates(self, embedding, n_results=1):
        self.requested.append(n_results)
        if self.records is not None:
            return self.records
        if self.metadata is None:
            return []
        return [{"metadata": self.metadata, "distance": 0.1, "embedding": None}]

    async def find_nearest(self, embedding, n_results=1):
        records = await self.find_candidates(embedding, n_results=n_results)
        return records[0]["metadata"] if records else None


class TestRecommendationService:
    async def test_returns_book_id_message(self):
        repo = Repo()
        service = RecommendationService(Store(), repo, FakeEmbedder())
        assert await service.recommend(1) == "Recommended book ID: 7"
        assert repo.saved == [(1, 7)]

    async def test_no_preferences_message(self):
        service = RecommendationService(Store(), Repo(preferences=[]), FakeEmbedder())
        assert await service.recommend(1) == NO_PREFERENCES

    async def test_no_book_found_message(self):
        service = RecommendationService(Store(metadata=None), Repo(), FakeEmbedder())
        assert await service.recommend(1) == NO_BOOK

    async def test_no_book_id_message(self):
        service = RecommendationService(Store(), Repo(book_id=None), FakeEmbedder())
        assert await service.recommend(1) == NO_BOOK_ID


class IsbnRepository(Repo):
    """ISBN별로 다른 book_id를 돌려주는 저장소."""

    def __init__(self, mapping, preferences=None):
        super().__init__(preferences=preferences)
        self.mapping = mapping

    def get_book_id_by_isbn(self, isbn):
        return self.mapping.get(isbn)


class TestRecommendationReranking:
    """벡터 순위 → 하이브리드 재정렬이 실제 응답을 바꾸는지 확인한다."""

    async def test_category_weight_changes_the_picked_book(self):
        # a: 벡터 1위(cos 0.94)지만 선호 카테고리가 아님
        # b: 벡터 2위(cos 0.90)이고 선호 카테고리 일치 → 0.90 + 0.15 = 1.05 로 역전
        store = Store(
            records=[
                _record("a", distance=0.12, sub_categories=("SF",)),
                _record("b", distance=0.20, sub_categories=("추리/스릴러",)),
            ]
        )
        repo = IsbnRepository({"a": 1, "b": 2})
        service = RecommendationService(store, repo, FakeEmbedder())
        assert await service.recommend(1) == "Recommended book ID: 2"
        assert repo.saved == [(1, 2)]

    async def test_without_category_weight_vector_order_is_kept(self):
        store = Store(
            records=[
                _record("a", distance=0.12, sub_categories=("SF",)),
                _record("b", distance=0.20, sub_categories=("추리/스릴러",)),
            ]
        )
        repo = IsbnRepository({"a": 1, "b": 2})
        service = RecommendationService(
            store, repo, FakeEmbedder(), weights=RankingWeights(category_weight=0.0)
        )
        assert await service.recommend(1) == "Recommended book ID: 1"

    async def test_diversity_pushes_same_category_duplicate_down(self):
        # a, b는 같은 카테고리, c는 다른 카테고리. 벡터로는 b(0.93) > c(0.92)지만
        # MMR이 b에 중복 페널티를 줘서 2순위가 c가 된다.
        store = Store(
            records=[
                _record("a", distance=0.10),  # cos 0.95
                _record("b", distance=0.14),  # cos 0.93
                _record("c", distance=0.16, sub_categories=("SF",)),  # cos 0.92
            ]
        )
        service = RecommendationService(store, IsbnRepository({}), FakeEmbedder())
        ranking = [candidate.isbn for candidate in await service.rank(1, limit=3)]
        assert ranking == ["a", "c", "b"]

    async def test_diversity_can_be_turned_off_with_mmr_lambda(self):
        store = Store(
            records=[
                _record("a", distance=0.10),
                _record("b", distance=0.14),
                _record("c", distance=0.16, sub_categories=("SF",)),
            ]
        )
        service = RecommendationService(
            store,
            IsbnRepository({}),
            FakeEmbedder(),
            weights=RankingWeights(mmr_lambda=1.0),
        )
        ranking = [candidate.isbn for candidate in await service.rank(1, limit=3)]
        assert ranking == ["a", "b", "c"]

    async def test_pool_size_is_used_for_vector_query(self):
        store = Store()
        service = RecommendationService(
            store, Repo(), FakeEmbedder(), weights=RankingWeights(pool_size=5)
        )
        await service.rank(1)
        assert store.requested == [5]

    async def test_rank_without_preferences_is_empty(self):
        service = RecommendationService(Store(), Repo(preferences=[]), FakeEmbedder())
        assert await service.rank(1) == []

    async def test_records_without_isbn_are_dropped(self):
        store = Store(records=[{"metadata": {"title": "isbn 없음"}, "distance": 0.1}])
        service = RecommendationService(store, Repo(), FakeEmbedder())
        assert await service.rank(1) == []
        assert await service.recommend(1) == NO_BOOK


# ------------------------------------------------------------------- scheduler
class SchedulerRepo(Repo):
    def __init__(self, member_ids=(1, 2)):
        super().__init__()
        self.member_ids = list(member_ids)
        self.best_sellers = []

    def get_all_member_ids(self):
        return self.member_ids

    def insert_best_sellers(self, rows):
        self.best_sellers.append(rows)


class SchedulerRecommendation:
    def __init__(self, failing=()):
        self.failing = set(failing)
        self.called = []

    async def recommend(self, member_id):
        self.called.append(member_id)
        if member_id in self.failing:
            raise UpstreamServiceError("실패")


class TestScheduledJobs:
    async def test_recommendations_continue_after_one_failure(self):
        repo = SchedulerRepo([1, 2, 3])
        recommendation = SchedulerRecommendation(failing=[2])
        jobs = ScheduledJobs(repo, recommendation, ingestion=None, aladin=None)  # type: ignore[arg-type]
        await jobs.recommend_for_all_members()
        assert recommendation.called == [1, 2, 3]

    async def test_empty_best_sellers_does_not_clear_table(self):
        """알라딘 조회가 빈 목록이면 테이블을 비우지 않는다(기존 버그 방지)."""
        repo = SchedulerRepo()

        class Aladin:
            def get_best_sellers(self):
                return []

        jobs = ScheduledJobs(repo, None, ingestion=None, aladin=Aladin())  # type: ignore[arg-type]
        await jobs.update_best_sellers()
        assert repo.best_sellers == []
