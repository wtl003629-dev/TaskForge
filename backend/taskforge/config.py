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
    workspace_root: Path = _PROJECT_ROOT
    artifact_root: Path = _PROJECT_ROOT / ".taskforge" / "artifacts"
    context_backend: Literal["memory", "sqlite"] = "sqlite"

    worker_lease_seconds: int = Field(default=30, ge=5, le=3_600)
    worker_max_attempts: int = Field(default=3, ge=1, le=20)
    mcp_config_path: Path | None = None

    provider: Literal["demo", "openai"] = "demo"
    openai_api_key: SecretStr | None = None
    openai_model: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_timeout_seconds: float = Field(default=30.0, gt=0, le=300)

    @field_validator("mcp_config_path", mode="before")
    @classmethod
    def blank_mcp_path_is_disabled(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


__all__ = ["Settings"]
