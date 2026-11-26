"""의존성 조립.

FastAPI의 `Depends`로 서비스를 주입하면 테스트에서 `app.dependency_overrides`로
외부 의존성(OpenAI, Chroma, MySQL, 크롤링)을 통째로 바꿀 수 있다. API 키나 DB
없이 엔드포인트를 테스트하려는 이유가 여기에 있다.

무거운 객체(Chroma 클라이언트, OpenAI 클라이언트)는 `lru_cache`로 한 번만 만든다.
테스트에서는 `reset_singletons()`으로 초기화한다.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from openai import AsyncOpenAI

from app.core.config import Settings, get_settings
from app.core.errors import ConfigurationError
from app.services.aladin import AladinClient
from app.services.classifier import CategoryClassifier
from app.services.database import BookRepository
from app.services.kyobo import KyoboHashtagCrawler
from app.services.library import BookIngestionService
from app.services.llm import EmbeddingService, LLMService
from app.services.recommendation import RecommendationService
from app.services.reranking import RankingWeights
from app.services.scheduler import ScheduledJobs
from app.services.vector_store import BookVectorStore


@lru_cache(maxsize=1)
def _openai_client() -> AsyncOpenAI | None:
    api_key = get_settings().openai_api_key
    if not api_key:
        return None
    return AsyncOpenAI(api_key=api_key)


@lru_cache(maxsize=1)
def _llm_service() -> LLMService:
    return LLMService(_openai_client(), default_model=get_settings().openai_chat_model)


@lru_cache(maxsize=1)
def _embedder() -> EmbeddingService:
    return EmbeddingService(_openai_client(), model=get_settings().openai_embedding_model)


@lru_cache(maxsize=1)
def _vector_store() -> BookVectorStore:
    settings = get_settings()
    return BookVectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        embedder=_embedder(),
    )


@lru_cache(maxsize=1)
def _repository() -> BookRepository:
    return BookRepository(get_settings().database)


@lru_cache(maxsize=1)
def _aladin() -> AladinClient:
    settings = get_settings()
    return AladinClient(settings.aladin_ttbkey, timeout=settings.http_timeout_seconds)


@lru_cache(maxsize=1)
def _crawler() -> KyoboHashtagCrawler:
    settings = get_settings()
    return KyoboHashtagCrawler(settings.categories_file, timeout=settings.http_timeout_seconds)


@lru_cache(maxsize=1)
def _classifier() -> CategoryClassifier:
    return CategoryClassifier(_llm_service(), model=get_settings().openai_chat_model)


def reset_singletons() -> None:
    for cached in (
        _openai_client,
        _llm_service,
        _embedder,
        _vector_store,
        _repository,
        _aladin,
        _crawler,
        _classifier,
    ):
        cached.cache_clear()


# ---------------------------------------------------------------- FastAPI용
def provide_settings() -> Settings:
    return get_settings()


def provide_llm() -> LLMService:
    return _llm_service()


def provide_embedder() -> EmbeddingService:
    return _embedder()


def provide_vector_store() -> BookVectorStore:
    return _vector_store()


def provide_repository() -> BookRepository:
    return _repository()


def provide_aladin() -> AladinClient:
    return _aladin()


def provide_crawler() -> KyoboHashtagCrawler:
    return _crawler()


def provide_classifier() -> CategoryClassifier:
    return _classifier()


def build_ranking_weights() -> RankingWeights:
    """환경 변수에서 추천 재정렬 가중치를 만든다.

    범위를 벗어난 값이 들어오면 요청마다 500이 나는 대신, 설정 오류로 바로 드러나게 한다.
    """
    settings = get_settings()
    try:
        return RankingWeights(
            category_weight=settings.recommend_category_weight,
            mmr_lambda=settings.recommend_mmr_lambda,
            pool_size=settings.recommend_pool_size,
        )
    except ValueError as exc:
        raise ConfigurationError(f"추천 재정렬 설정이 올바르지 않습니다: {exc}") from exc


def provide_recommendation_service(
    store: BookVectorStore = Depends(provide_vector_store),
    repository: BookRepository = Depends(provide_repository),
    embedder: EmbeddingService = Depends(provide_embedder),
) -> RecommendationService:
    return RecommendationService(
        store,
        repository,
        embedder,
        weights=build_ranking_weights(),
        exclude_recent_days=get_settings().recommend_exclude_recent_days,
    )


def provide_ingestion_service(
    aladin: AladinClient = Depends(provide_aladin),
    crawler: KyoboHashtagCrawler = Depends(provide_crawler),
    classifier: CategoryClassifier = Depends(provide_classifier),
    repository: BookRepository = Depends(provide_repository),
    store: BookVectorStore = Depends(provide_vector_store),
) -> BookIngestionService:
    return BookIngestionService(aladin, crawler, classifier, repository, store)


def build_scheduler_jobs() -> ScheduledJobs:
    return ScheduledJobs(
        repository=_repository(),
        recommendation=RecommendationService(
            _vector_store(),
            _repository(),
            _embedder(),
            weights=build_ranking_weights(),
            exclude_recent_days=get_settings().recommend_exclude_recent_days,
        ),
        ingestion=BookIngestionService(_aladin(), _crawler(), _classifier(), _repository(), _vector_store()),
        aladin=_aladin(),
    )
