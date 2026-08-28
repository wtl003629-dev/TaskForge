"""Host-owned RAG experiment profiles and isolated index identities.

``current`` deliberately keeps the legacy, untagged document and knowledge
base identities so existing indexes remain readable.  Every ``optimized``
ablation uses an explicit suffix, which allows experimental chunks to coexist
without replacing the current document version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

RAGProfileName = Literal["current", "optimized"]
RAGAblationStage = Literal["a", "b", "c", "d", "e"]


@dataclass(frozen=True, slots=True)
class RAGExperimentProfile:
    """Resolved behavior for one controlled RAG ablation."""

    name: RAGProfileName
    ablation: RAGAblationStage
    retrieval_text_enabled: bool
    parent_aware_rerank_enabled: bool
    lineage_diversity_enabled: bool
    structure_aware_chunking_enabled: bool

    @property
    def label(self) -> str:
        return "current-a" if self.name == "current" else f"optimized-{self.ablation}"

    def document_id(self, current_document_id: str) -> str:
        if self.name == "current":
            return current_document_id
        return f"{current_document_id}:rag:{self.label}"

    def knowledge_base_id(self, current_knowledge_base_id: str) -> str:
        if self.name == "current":
            return current_knowledge_base_id
        return f"{current_knowledge_base_id}:rag:{self.label}"

    def metadata(self) -> dict[str, str]:
        # Do not tag current chunks: old indexes have no profile field and must
        # remain byte-for-byte addressable through their existing identity.
        if self.name == "current":
            return {}
        return {
            "rag_profile": self.name,
            "rag_ablation": self.ablation,
            "rag_profile_label": self.label,
        }

    def matches(self, metadata: Mapping[str, object]) -> bool:
        tagged_profile = str(metadata.get("rag_profile") or "").strip().casefold()
        if self.name == "current":
            return tagged_profile in {"", "current"}
        tagged_ablation = str(metadata.get("rag_ablation") or "").strip().casefold()
        return tagged_profile == "optimized" and tagged_ablation == self.ablation


def resolve_rag_experiment_profile(
    name: RAGProfileName | str,
    ablation: RAGAblationStage | str = "e",
) -> RAGExperimentProfile:
    """Resolve A-E without letting an experiment mutate current behavior."""

    normalized_name = str(name).strip().casefold()
    normalized_ablation = str(ablation).strip().casefold()
    if normalized_name not in {"current", "optimized"}:
        raise ValueError("RAG profile must be current or optimized")
    if normalized_ablation not in {"a", "b", "c", "d", "e"}:
        raise ValueError("RAG ablation must be one of a, b, c, d, or e")
    # A is the original chain.  B-E add exactly one family at a time.
    stage_order = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
    effective_ablation = "a" if normalized_name == "current" else normalized_ablation
    level = stage_order[effective_ablation]
    return RAGExperimentProfile(
        name=normalized_name,  # type: ignore[arg-type]
        ablation=effective_ablation,  # type: ignore[arg-type]
        retrieval_text_enabled=level >= 1,
        parent_aware_rerank_enabled=level >= 2,
        lineage_diversity_enabled=level >= 3,
        structure_aware_chunking_enabled=level >= 4,
    )


def validate_optimized_promotion_manifest(path: str | Path) -> dict[str, object]:
    """Require a completed A/B gate before live optimized routing."""

    manifest_path = Path(path).resolve(strict=True)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "taskforge.rag_profile_ab.v1":
        raise ValueError("optimized promotion manifest has an unsupported schema")
    if raw.get("status") != "complete":
        raise ValueError("optimized promotion manifest is not complete")
    decision = raw.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("optimized promotion manifest has no decision")
    if decision.get("outcome") != "eligible_for_canary":
        raise ValueError("optimized RAG has not passed the canary promotion gate")
    if decision.get("retrieval_gate_passed") is not True:
        raise ValueError("optimized RAG retrieval gate did not pass")
    answer_gate = decision.get("answer_and_citation_gate")
    if not isinstance(answer_gate, dict) or answer_gate.get("passed") is not True:
        raise ValueError("optimized RAG answer/citation gate did not pass")
    return raw


__all__ = [
    "RAGAblationStage",
    "RAGExperimentProfile",
    "RAGProfileName",
    "resolve_rag_experiment_profile",
    "validate_optimized_promotion_manifest",
]
