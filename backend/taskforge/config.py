"""Application configuration for the local TaskForge API.

Only host configuration selects durable storage and filesystem roots.  Task
metadata is deliberately not consulted for any of these authority-bearing
paths.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .rag_experiment_profile import validate_optimized_promotion_manifest

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
    # Durable backend selection is deliberately independent from the legacy
    # context_backend switch. PostgreSQL is the production default; SQLite is
    # retained only as an explicit compatibility/test backend.
    database_backend: Literal["sqlite", "postgres"] = "postgres"
    database_url: str | None = None
    postgres_pool_min_size: int = Field(default=1, ge=1, le=32)
    postgres_pool_max_size: int = Field(default=8, ge=1, le=64)
    postgres_connect_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
    # Legacy SQLite-branch selector; ignored when database_backend=postgres,
    # which always uses the PostgreSQL context stores.
    context_backend: Literal["memory", "sqlite"] = "sqlite"
    # The live application stays on the original chain until an offline gate
    # explicitly promotes optimized.  The experiment selector is consumed by
    # evaluation tooling and never changes live routing on its own.
    rag_active_profile: Literal["current", "optimized"] = "current"
    rag_experiment_profile: Literal["current", "optimized"] = "current"
    rag_optimized_ablation: Literal["a", "b", "c", "d", "e"] = "e"
    rag_evaluation_mode: bool = False
    rag_optimized_promotion_manifest: Path | None = None
    retrieval_routing: Literal["profile", "lexical"] = "profile"
    # Keep the evaluated control route as the safe default. BGE-M3 is an
    # explicit candidate selected by TASKFORGE_GENERAL_TEXT_BACKEND after its
    # promotion gates pass; the lexical path remains diagnostic only.
    general_text_backend: Literal[
        "bm25", "fastembed", "flagembedding", "bailian"
    ] = "fastembed"
    semantic_model: str = "BAAI/bge-small-en-v1.5"
    semantic_model_path: Path | None = None
    semantic_batch_size: int = Field(default=64, ge=1, le=256)
    semantic_device: Literal["auto", "cpu", "cuda"] = "auto"
    # Optional research-only multilingual route. Keeping these opt-in avoids
    # downloading a 3+ GB model pair for existing English deployments; when
    # configured, the research retriever selects it for CJK/cross-lingual
    # queries while leaving the English route untouched.
    research_multilingual_semantic_model: str | None = None
    semantic_cache_path: Path = _PROJECT_ROOT / ".taskforge" / "embeddings.sqlite3"
    bailian_api_key: SecretStr | None = None
    bailian_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    bailian_chat_model: str = "qwen-plus"
    bailian_chat_timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    # Bailian calls should not inherit a workstation proxy implicitly. This
    # keeps generation consistent with the existing embedding/rerank clients.
    bailian_chat_trust_env: bool = False
    bailian_model: str = "text-embedding-v4"
    bailian_embedding_dimension: int = Field(default=1_024, ge=1, le=65_536)
    bailian_batch_size: int = Field(default=10, ge=1, le=10)
    bailian_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    bailian_max_retries: int = Field(default=3, ge=0, le=10)
    bailian_cache_path: Path = (
        _PROJECT_ROOT / ".taskforge" / "embeddings-bailian-v4-1024.sqlite3"
    )
    bailian_index_name: str = "knowledge-bailian-text-embedding-v4-1024-v1"
    fastembed_model_cache_root: Path = (
        _PROJECT_ROOT / ".taskforge" / "model-cache" / "fastembed"
    )
    research_graph_enabled: bool = False
    research_reranker_backend: Literal["fastembed", "flagembedding", "fastembed_ensemble", "transformers", "bailian"] = "fastembed_ensemble"
    research_reranker_model: str | None = "jinaai/jina-reranker-v1-tiny-en,Xenova/ms-marco-MiniLM-L-6-v2"
    research_multilingual_reranker_backend: Literal["fastembed", "flagembedding", "fastembed_ensemble", "transformers", "bailian"] = "fastembed"
    research_multilingual_reranker_model: str | None = None
    research_reranker_device: Literal["auto", "cpu", "cuda"] = "auto"
    research_reranker_batch_size: int = Field(default=32, ge=1, le=512)
    # Optional Alibaba Cloud qwen3-rerank route. It reuses bailian_api_key;
    # keeping the endpoint separate from the embedding endpoint prevents an
    # OpenAI-compatible /embeddings URL from being used for reranking.
    bailian_rerank_base_url: str = (
        "https://dashscope.aliyuncs.com/compatible-api/v1"
    )
    bailian_rerank_model: str = "qwen3-rerank"
    bailian_rerank_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    bailian_rerank_max_retries: int = Field(default=2, ge=0, le=10)
    research_feature_ranker_path: Path | None = None
    research_structure_fusion_enabled: bool = False
    research_structure_section_weight: float = Field(default=0.5, ge=0.0, le=2.0)
    research_structure_query_coverage_weight: float = Field(default=0.1, ge=0.0, le=2.0)
    research_preserve_head_k: int = Field(default=0, ge=0, le=10)
    research_reranker_context_window: int = Field(default=0, ge=0, le=2)
    # Experimental single-pass rerank view: raw Child plus bounded same-Parent
    # neighbours. It is gated to the optimized profile in the application, so
    # setting it cannot mutate the current route.
    research_contextual_child_rerank_enabled: bool = False
    research_contextual_child_neighbor_tokens: int = Field(default=120, ge=16, le=240)
    research_contextual_child_max_tokens: int = Field(default=500, ge=256, le=1_024)
    research_lexical_fusion_weight: float = Field(default=0.0, ge=0.0, le=1.0)
    research_parent_aware_rerank_enabled: bool = True
    research_parent_referential_guard_enabled: bool = True
    research_parent_aware_candidate_k: int = Field(default=20, ge=1, le=100)
    research_parent_context_max_tokens: int = Field(default=800, ge=128, le=3_000)
    research_parent_include_document_title: bool = True
    research_parent_include_heading_path: bool = True
    research_parent_include_neighbor_chunks: bool = True
    research_parent_child_score_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    research_parent_context_score_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    research_parent_retrieval_score_weight: float = Field(default=0.10, ge=0.0, le=1.0)
    research_lineage_diversity_enabled: bool = True
    research_lineage_preferred_children_per_parent: int = Field(default=2, ge=1, le=10)
    research_lineage_parent_penalty: float = Field(default=0.08, ge=0.0, le=1.0)
    research_lineage_overlap_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    research_intent_section_fusion_enabled: bool = False
    research_intent_section_fusion_weight: float = Field(default=0.1, ge=0.0, le=2.0)
    research_intent_query_overlap_weight: float = Field(default=0.05, ge=0.0, le=1.0)
    research_intent_rank_fusion_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    research_rewrite_enabled: bool = False
    research_query_expansion_mode: Literal["original", "keyword", "synonym", "full"] = "original"
    research_operator_budget_standard: int = Field(default=1, ge=0, le=4)
    research_operator_budget_rigorous: int = Field(default=2, ge=0, le=4)
    # Optional dual-lane retrieval. It is disabled by default so the current
    # Flat/Parent-Child route remains byte-for-byte selectable as the control.
    research_dual_route_enabled: bool = False
    research_dual_route_flat_candidate_k: int = Field(default=30, ge=1, le=100)
    research_dual_route_child_candidate_k: int = Field(default=20, ge=1, le=100)
    research_dual_route_flat_head_k: int = Field(default=2, ge=0, le=10)
    research_dual_route_rerank_candidate_k: int = Field(default=10, ge=1, le=100)
    research_dual_route_tail_rerank_candidate_k: int = Field(default=0, ge=0, le=100)
    research_dual_route_min_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    pdf_parser_backend: Literal["auto", "native", "mineru"] = "auto"
    mineru_base_url: str | None = None
    mineru_expected_version: str | None = None
    mineru_cache_root: Path | None = None
    # The default D-drive deployment downloads MinerU's pipeline bundle only.
    # VLM-based parsing remains an explicit opt-in because it requires the
    # separate, substantially larger model bundle.
    mineru_backend: str = "pipeline"
    mineru_parse_method: Literal["auto", "txt", "ocr"] = "auto"
    mineru_effort: Literal["medium", "high"] = "high"
    mineru_timeout_seconds: float = Field(default=300.0, gt=0, le=1_800)
    mineru_max_retries: int = Field(default=2, ge=0, le=5)
    mineru_concurrency: int = Field(default=2, ge=1, le=16)
    visual_extractor_base_url: str | None = None
    visual_extractor_api_key: SecretStr | None = None
    visual_extractor_model: str | None = None
    visual_extractor_timeout_seconds: float = Field(default=120.0, gt=0, le=600)
    visual_extractor_concurrency: int = Field(default=2, ge=1, le=8)
    pdf_parent_target_tokens: int = Field(default=2_000, ge=500, le=8_000)
    pdf_parent_max_tokens: int = Field(default=3_000, ge=500, le=8_000)
    pdf_child_target_tokens: int = Field(default=400, ge=100, le=2_000)
    pdf_child_max_tokens: int = Field(default=500, ge=100, le=3_000)
    pdf_child_overlap_tokens: int = Field(default=60, ge=0, le=200)
    # Parent-Child is the single application path. Flat/sliding remain host-side
    # diagnostic profiles and are not exposed as separate Agent modes.
    pdf_chunking_mode: Literal["flat", "parent_child", "hybrid", "sliding"] = "parent_child"
    pdf_flat_chunk_chars: int = Field(default=2_000, ge=256, le=50_000)
    pdf_flat_overlap_chars: int = Field(default=0, ge=0, le=50_000)
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

    provider: Literal["demo", "openai", "deepseek", "bailian"] = "demo"
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    deepseek_api_key: SecretStr | None = None
    deepseek_model: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    deepseek_trust_env: bool = True

    @field_validator(
        "mcp_config_path",
        "database_url",
        "literature_contact_email",
        "semantic_scholar_api_key",
        "openalex_api_key",
        "bailian_api_key",
        "research_reranker_model",
        "research_multilingual_semantic_model",
        "research_multilingual_reranker_model",
        "semantic_model_path",
        "mineru_base_url",
        "mineru_expected_version",
        "mineru_cache_root",
        "visual_extractor_base_url",
        "visual_extractor_api_key",
        "visual_extractor_model",
        "rag_optimized_promotion_manifest",
        mode="before",
    )
    @classmethod
    def blank_mcp_path_is_disabled(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def pdf_parser_settings_are_consistent(self) -> Settings:
        if self.postgres_pool_max_size < self.postgres_pool_min_size:
            raise ValueError("postgres_pool_max_size must be >= postgres_pool_min_size")
        if self.database_backend == "postgres" and not (self.database_url or "").strip():
            raise ValueError(
                "database_backend=postgres requires TASKFORGE_DATABASE_URL"
            )
        if self.general_text_backend == "bailian":
            if self.bailian_api_key is None:
                raise ValueError(
                    "general_text_backend=bailian requires bailian_api_key"
                )
            if not self.bailian_base_url.strip().startswith("https://"):
                raise ValueError("bailian_base_url must use HTTPS")
            if self.bailian_model.casefold() != "text-embedding-v4":
                raise ValueError(
                    "the initial Bailian route only supports text-embedding-v4"
                )
            if self.bailian_embedding_dimension != 1_024:
                raise ValueError(
                    "the Bailian text-embedding-v4 route requires 1024 dimensions"
                )
            if not self.bailian_index_name.strip():
                raise ValueError("bailian_index_name must be non-empty")
        if self.provider == "bailian":
            if self.bailian_api_key is None:
                raise ValueError("provider=bailian requires bailian_api_key")
            if not self.bailian_base_url.strip().startswith("https://"):
                raise ValueError("bailian_base_url must use HTTPS")
            if not self.bailian_chat_model.strip():
                raise ValueError("bailian_chat_model must be non-empty")
        if (
            self.research_reranker_backend == "bailian"
            or self.research_multilingual_reranker_backend == "bailian"
        ):
            if self.bailian_api_key is None:
                raise ValueError(
                    "Bailian research reranker requires bailian_api_key"
                )
            if not self.bailian_rerank_base_url.strip().startswith("https://"):
                raise ValueError("bailian_rerank_base_url must use HTTPS")
            if self.bailian_rerank_model.casefold() != "qwen3-rerank":
                raise ValueError(
                    "the Bailian research reranker currently supports qwen3-rerank"
                )
        if self.rag_active_profile == "optimized" and not self.rag_evaluation_mode:
            if self.rag_optimized_promotion_manifest is None:
                raise ValueError(
                    "rag_active_profile=optimized requires a passed promotion manifest"
                )
            validate_optimized_promotion_manifest(
                self.rag_optimized_promotion_manifest
            )
        if self.rag_evaluation_mode and (
            self.rag_active_profile != self.rag_experiment_profile
        ):
            raise ValueError(
                "RAG evaluation mode requires active and experiment profiles to match"
            )
        if self.pdf_parser_backend == "mineru" and self.mineru_base_url is None:
            raise ValueError("pdf_parser_backend=mineru requires mineru_base_url")
        if self.mineru_base_url is not None and self.mineru_expected_version is None:
            raise ValueError("configured MinerU service requires mineru_expected_version")
        visual_values = (
            self.visual_extractor_base_url,
            self.visual_extractor_api_key,
            self.visual_extractor_model,
        )
        if any(value is not None for value in visual_values) and not all(
            value is not None for value in visual_values
        ):
            raise ValueError(
                "visual extractor requires base URL, API key, and exact model ID"
            )
        if self.pdf_parent_target_tokens > self.pdf_parent_max_tokens:
            raise ValueError("PDF Parent target cannot exceed Parent maximum")
        if self.pdf_child_target_tokens > self.pdf_child_max_tokens:
            raise ValueError("PDF Child target cannot exceed Child maximum")
        if self.pdf_child_max_tokens > self.pdf_parent_max_tokens:
            raise ValueError("PDF Child maximum cannot exceed Parent maximum")
        if self.pdf_child_overlap_tokens >= self.pdf_child_target_tokens:
            raise ValueError("PDF Child overlap must be smaller than Child target")
        if self.pdf_flat_overlap_chars >= self.pdf_flat_chunk_chars:
            raise ValueError("PDF flat overlap must be smaller than flat chunk target")
        if self.research_dual_route_enabled and self.pdf_chunking_mode != "hybrid":
            raise ValueError(
                "research_dual_route_enabled requires pdf_chunking_mode=hybrid"
            )
        if self.pdf_chunking_mode == "hybrid" and not self.research_dual_route_enabled:
            raise ValueError(
                "pdf_chunking_mode=hybrid requires research_dual_route_enabled=true"
            )
        if (
            self.research_contextual_child_neighbor_tokens * 2
            >= self.research_contextual_child_max_tokens
        ):
            raise ValueError(
                "contextual Child neighbour budgets must leave room for the target Child"
            )
        if (
            self.research_parent_child_score_weight
            + self.research_parent_context_score_weight
            + self.research_parent_retrieval_score_weight
            <= 0
        ):
            raise ValueError("Parent-aware rerank score weights must have a positive sum")
        return self


__all__ = ["Settings"]
