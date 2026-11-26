"""OpenAI 호출 (텍스트 생성 · 임베딩).

기존 코드는 `openai.ChatCompletion.create`(v0.28 이하 API)를 동기로 호출했다.
1.x SDK의 `AsyncOpenAI` + `chat.completions.create`로 바꾸고, 모델/토큰 수 같은
파라미터는 그대로 유지했다(`/generate` 기본 모델 "gpt-4", max_tokens 100).
"""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.core.errors import ConfigurationError, LanguageModelError

logger = logging.getLogger("arabook.llm")


class LLMService:
    def __init__(self, client: AsyncOpenAI | None, default_model: str = "gpt-4"):
        self._client = client
        self._default_model = default_model

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            raise ConfigurationError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return self._client

    async def generate(self, prompt: str, model: str | None = None, max_tokens: int = 100) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=model or self._default_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
        except Exception as exc:  # openai.APIError 계열을 하나로 묶는다
            raise LanguageModelError(f"OpenAI 텍스트 생성에 실패했습니다: {exc}") from exc
        content = response.choices[0].message.content if response.choices else None
        if content is None:
            raise LanguageModelError("OpenAI가 빈 응답을 반환했습니다.")
        return content


class EmbeddingService:
    """Chroma에 넣고 질의할 임베딩을 만든다.

    기존에는 `langchain_openai.OpenAIEmbeddings`를 썼고 기본 모델은
    `text-embedding-ada-002`였다. 같은 모델을 OpenAI SDK로 직접 호출해
    벡터 공간을 그대로 유지하면서 langchain 의존성을 없앴다.
    """

    def __init__(self, client: AsyncOpenAI | None, model: str = "text-embedding-ada-002"):
        self._client = client
        self.model = model

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            raise ConfigurationError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return self._client

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._embed(texts)

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self._embed([text])
        return vectors[0]

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            response = await self.client.embeddings.create(model=self.model, input=texts)
        except Exception as exc:
            raise LanguageModelError(f"OpenAI 임베딩 생성에 실패했습니다: {exc}") from exc
        data: list[Any] = response.data
        return [list(item.embedding) for item in data]
