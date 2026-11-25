"""환경 설정.

기존 코드는 모듈을 import하는 순간 `os.getenv`를 읽고 OpenAI 클라이언트를 만들어
API 키가 없으면 import 자체가 실패했다. 여기서는 설정을 한 곳에서 **호출 시점에**
읽어, 테스트나 문서 생성처럼 외부 연결이 필요 없는 작업이 키 없이도 돌게 한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# .env 파일이 있으면 로드한다 (없으면 조용히 넘어간다).
load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DatabaseSettings:
    ssh_host: str | None
    ssh_port: int
    ssh_username: str | None
    ssh_private_key: str | None
    mysql_host: str | None
    username: str | None
    password: str | None
    database: str | None

    @property
    def configured(self) -> bool:
        return all(
            [
                self.ssh_host,
                self.ssh_username,
                self.ssh_private_key,
                self.mysql_host,
                self.username,
                self.database,
            ]
        )


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4"
    openai_embedding_model: str = "text-embedding-ada-002"
    aladin_ttbkey: str | None = None
    chroma_persist_directory: str = "./chroma_db"
    chroma_collection_name: str = "books"
    categories_path: str = "categories.txt"
    http_timeout_seconds: float = 10.0
    scheduler_enabled: bool = True
    log_level: str = "INFO"
    database: DatabaseSettings = DatabaseSettings(None, 22, None, None, None, None, None, None)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4"),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            aladin_ttbkey=os.getenv("ALADIN_TTBKEY"),
            chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
            chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "books"),
            categories_path=os.getenv("CATEGORIES_PATH", "categories.txt"),
            http_timeout_seconds=float(os.getenv("HTTP_TIMEOUT_SECONDS", "10")),
            scheduler_enabled=_env_bool("SCHEDULER_ENABLED", True),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            database=DatabaseSettings(
                ssh_host=os.getenv("SSH_TUNNEL_HOST_ADDRESS"),
                ssh_port=int(os.getenv("SSH_TUNNEL_HOST_PORT", "22") or 22),
                ssh_username=os.getenv("SSH_USERNAME"),
                ssh_private_key=os.getenv("SSH_PRIVATE_KEY"),
                mysql_host=os.getenv("MYSQL_SERVER_HOST"),
                username=os.getenv("DB_USERNAME"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_DATABASE_NAME"),
            ),
        )

    @property
    def categories_file(self) -> Path:
        return Path(self.categories_path)

    def require_openai_key(self) -> str:
        if not self.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        return self.openai_api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()


def reset_settings_cache() -> None:
    """테스트에서 환경 변수를 바꾼 뒤 설정을 다시 읽기 위한 헬퍼."""
    get_settings.cache_clear()
