"""평가용 임베딩 제공자.

운영 코드는 OpenAI 임베딩을 쓴다. 오프라인 평가도 같은 벡터 공간을 써야
비교가 의미 있으므로 기본값은 OpenAI `text-embedding-ada-002`(운영과 동일)다.

`fake` 임베더는 API 키 없이 **파이프라인이 도는지** 확인하는 용도다. 의미 있는
유사도가 아니므로 이 임베더로 낸 수치는 품질 지표로 해석하면 안 된다
(`--embedder fake` 사용 시 하네스가 리포트에 경고를 남긴다).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Protocol, Sequence


class EmbedderError(RuntimeError):
    """임베딩을 만들 수 없을 때."""


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


class OpenAIEmbedder:
    """운영과 동일한 OpenAI 임베딩 모델을 사용한다."""

    def __init__(self, model: str = "text-embedding-ada-002", api_key: str | None = None, client=None):
        self.model = model
        self.name = f"openai:{model}"
        if client is not None:
            self._client = client
            return
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbedderError(
                "OPENAI_API_KEY가 없어 OpenAI 임베딩을 만들 수 없습니다. "
                "실제 품질 측정에는 키가 필요합니다(파이프라인 점검은 --embedder fake)."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise EmbedderError("openai 패키지가 설치되어 있지 않습니다.") from exc
        self._client = OpenAI(api_key=api_key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [list(item.embedding) for item in response.data]


class FakeHashEmbedder:
    """API 키 없이 도는 결정적 해시 임베더. **품질 측정용이 아니다.**

    토큰을 해시해 고정 차원에 흩뿌리고 L2 정규화한다. 문자열이 겹치면 벡터도
    겹치는 정도의 의미만 있어, 의미 기반 유사도와는 무관하다. 파이프라인/지표
    계산이 정상 동작하는지 확인하는 스모크 테스트 전용.
    """

    name = "fake-hash"

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


class CachedEmbedder:
    """임베딩 결과를 JSON에 캐시해 API 재호출을 줄인다."""

    def __init__(self, inner: Embedder, cache_path: str | Path):
        self._inner = inner
        self._path = Path(cache_path)
        self.name = inner.name
        self._cache: dict[str, list[float]] = {}
        if self._path.exists():
            try:
                self._cache = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = [text for text in dict.fromkeys(texts) if text not in self._cache]
        if missing:
            for text, embedding in zip(missing, self._inner.embed(missing)):
                self._cache[text] = embedding
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
            )
        return [self._cache[text] for text in texts]


def build_embedder(name: str, model: str, cache_path: str | Path | None = None) -> Embedder:
    if name == "openai":
        embedder: Embedder = OpenAIEmbedder(model=model)
    elif name == "fake":
        embedder = FakeHashEmbedder()
    else:
        raise EmbedderError(f"알 수 없는 임베더: {name} (openai | fake)")
    if cache_path and name == "openai":
        return CachedEmbedder(embedder, cache_path)
    return embedder
