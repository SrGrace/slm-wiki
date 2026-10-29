from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SLM_WIKI_",
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: Literal["local", "mga"] = "local"
    llm_base_url: str = "http://127.0.0.1:18434/v1"
    llm_model: str = "Qwen_Qwen3-8B-Q4_K_M"
    llm_api_key: str = "no-key"
    mga_base_url: str = "https://chat.int.bayer.com/api/v2"
    mga_model: str = "claude-sonnet-4.6"
    mga_api_key: str = ""
    llm_timeout_s: float = 600.0
    llm_temperature: float = 0.2
    llm_max_tokens: int = 8192

    chunk_chars: int = 4_000
    page_char_budget: int = 12_000
    max_pages_per_ingest: int = 20
    query_top_k: int = 8
    query_char_budget: int = 24_000

    vault: Path = Path("vault")  # relative to repo root

    @field_validator("vault", mode="before")
    @classmethod
    def _vault_path(cls, v: object) -> Path:
        p = Path(str(v))
        return p if p.is_absolute() else (ROOT / p).resolve()

    @model_validator(mode="after")
    def _provider(self) -> Settings:
        if self.llm_provider == "mga" and not self.mga_api_key:
            raise ValueError("SLM_WIKI_LLM_PROVIDER=mga needs SLM_WIKI_MGA_API_KEY")
        return self

    # llm_base_url/model/api_key always hold the local llama.cpp triple; these pick
    # local vs MGA per llm_provider, which the UI/server can flip at runtime.
    @property
    def effective_base_url(self) -> str:
        return self.mga_base_url if self.llm_provider == "mga" else self.llm_base_url

    @property
    def effective_model(self) -> str:
        return self.mga_model if self.llm_provider == "mga" else self.llm_model

    @property
    def effective_api_key(self) -> str:
        return self.mga_api_key if self.llm_provider == "mga" else self.llm_api_key

    @property
    def raw_dir(self) -> Path:
        return self.vault / "raw"

    @property
    def wiki_dir(self) -> Path:
        return self.vault / "wiki"

    @property
    def schema_dir(self) -> Path:
        return self.vault / "schema"

    @property
    def logs_dir(self) -> Path:
        return self.vault / "logs"

    @property
    def index_path(self) -> Path:
        return self.wiki_dir / "index.md"

    @property
    def log_path(self) -> Path:
        return self.wiki_dir / "log.md"

    @property
    def events_path(self) -> Path:
        return self.logs_dir / "events.jsonl"


def get_settings() -> Settings:
    return Settings()
