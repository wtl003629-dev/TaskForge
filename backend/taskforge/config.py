"""Application configuration for the local TaskForge API.

Only host configuration selects durable storage and filesystem roots.  Task
metadata is deliberately not consulted for any of these authority-bearing
paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Environment-backed settings using the ``TASKFORGE_`` prefix."""

    model_config = SettingsConfigDict(
        env_prefix="TASKFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sqlite_path: Path = _PROJECT_ROOT / ".taskforge" / "taskforge.sqlite3"
    context_sqlite_path: Path = _PROJECT_ROOT / ".taskforge" / "context.sqlite3"
    operations_sqlite_path: Path = _PROJECT_ROOT / ".taskforge" / "operations.sqlite3"
    orchestration_sqlite_path: Path = _PROJECT_ROOT / ".taskforge" / "orchestration.sqlite3"
    review_case_sqlite_path: Path = _PROJECT_ROOT / ".taskforge" / "review-cases.sqlite3"
    verification_sqlite_path: Path = _PROJECT_ROOT / ".taskforge" / "verification.sqlite3"
    literature_sqlite_path: Path = _PROJECT_ROOT / ".taskforge" / "literature.sqlite3"
    literature_cache_path: Path = _PROJECT_ROOT / ".taskforge" / "literature-cache.sqlite3"
    workspace_root: Path = _PROJECT_ROOT
    artifact_root: Path = _PROJECT_ROOT / ".taskforge" / "artifacts"
    context_backend: Literal["memory", "sqlite"] = "sqlite"
    retrieval_routing: Literal["profile", "lexical"] = "profile"
    # Recall-first default for long-form research. The lexical path remains
    # available as an explicit degraded/diagnostic profile.
    general_text_backend: Literal["bm25", "fastembed"] = "fastembed"
    semantic_model: str = "BAAI/bge-small-en-v1.5"
    semantic_cache_path: Path = _PROJECT_ROOT / ".taskforge" / "embeddings.sqlite3"
    research_graph_enabled: bool = False
    research_reranker_backend: Literal["fastembed", "flagembedding", "fastembed_ensemble", "transformers"] = "fastembed_ensemble"
    research_reranker_model: str | None = "jinaai/jina-reranker-v1-tiny-en,Xenova/ms-marco-MiniLM-L-6-v2"
    research_reranker_device: Literal["auto", "cpu", "cuda"] = "auto"
    research_reranker_batch_size: int = Field(default=32, ge=1, le=512)
    research_feature_ranker_path: Path | None = None
    research_structure_fusion_enabled: bool = False
    research_structure_section_weight: float = Field(default=0.5, ge=0.0, le=2.0)
    research_structure_query_coverage_weight: float = Field(default=0.1, ge=0.0, le=2.0)
    research_preserve_head_k: int = Field(default=0, ge=0, le=10)
    research_reranker_context_window: int = Field(default=0, ge=0, le=2)
    research_lexical_fusion_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    research_intent_section_fusion_enabled: bool = False
    research_intent_section_fusion_weight: float = Field(default=0.1, ge=0.0, le=2.0)
    research_intent_query_overlap_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    research_intent_rank_fusion_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    research_rewrite_enabled: bool = False
    literature_cache_ttl_seconds: int = Field(default=86_400, ge=0, le=2_592_000)
    literature_provider_timeout_seconds: float = Field(default=15.0, gt=0, le=120)
    literature_provider_max_retries: int = Field(default=2, ge=0, le=5)
    literature_results_per_query: int = Field(default=50, ge=1, le=100)
    literature_contact_email: str | None = None
    semantic_scholar_api_key: SecretStr | None = None
    openalex_api_key: SecretStr | None = None

    worker_lease_seconds: int = Field(default=30, ge=5, le=3_600)
    worker_max_attempts: int = Field(default=3, ge=1, le=20)
    mcp_config_path: Path | None = None

    provider: Literal["demo", "openai", "deepseek"] = "demo"
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    deepseek_api_key: SecretStr | None = None
    deepseek_model: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    @field_validator(
        "mcp_config_path",
        "literature_contact_email",
        "semantic_scholar_api_key",
        "openalex_api_key",
        "research_reranker_model",
        mode="before",
    )
    @classmethod
    def blank_mcp_path_is_disabled(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


__all__ = ["Settings"]
