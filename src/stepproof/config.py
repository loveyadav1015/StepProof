"""Configuration loader using environment variables and Pydantic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field


class Settings(BaseModel):
    """Application settings driven by environment variables."""

    # Provider selection
    stepproof_llm: Literal["mock", "openai_compat"] = Field(
        default_factory=lambda: os.getenv("STEPPROOF_LLM", "mock")  # type: ignore
    )
    stepproof_seg: Literal["mock", "local_sam", "meta_api"] = Field(
        default_factory=lambda: os.getenv("STEPPROOF_SEG", "mock")  # type: ignore
    )

    # Execution mode
    stepproof_mode: Literal["live", "cache_only", "refresh"] = Field(
        default_factory=lambda: os.getenv("STEPPROOF_MODE", "live")  # type: ignore
    )

    # Meta Compliance Guard
    stepproof_require_meta: bool = Field(
        default_factory=lambda: os.getenv("STEPPROOF_REQUIRE_META", "0").lower()
        in ("1", "true", "yes")
    )

    # Cache Path
    cache_db_path: Path = Field(
        default_factory=lambda: Path(
            os.getenv("STEPPROOF_CACHE_PATH", "data/cache/stepproof_cache.db")
        )
    )

    # LLM Settings (OpenAI-compatible)
    llm_base_url: str = Field(
        default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    )
    llm_model: str = Field(
        default_factory=lambda: os.getenv("LLM_MODEL", "meta/muse-glimmer-30b")
    )
    llm_api_key: str = Field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_api_style: Literal["chat", "responses"] = Field(
        default_factory=lambda: os.getenv("LLM_API_STYLE", "chat")  # type: ignore
    )
    llm_temperature: Optional[float] = Field(
        default_factory=lambda: (
            float(os.getenv("LLM_TEMPERATURE"))
            if os.getenv("LLM_TEMPERATURE")
            else None
        )
    )
    llm_reasoning_effort: Optional[str] = Field(
        default_factory=lambda: os.getenv("LLM_REASONING_EFFORT") or None
    )

    @classmethod
    def load(cls) -> Settings:
        """Create a Settings instance from current environment variables."""
        return cls()


# Global default settings instance
settings = Settings.load()
