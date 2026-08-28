"""Deterministic structure profiler and hybrid chunk-policy resolver."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ParsedDocument
from .hierarchy import (
    HierarchicalUnit,
    build_flat_units,
    build_parent_child_units,
)

_STRUCTURED_TYPES = frozenset(
    {"list", "table", "chart", "equation", "image", "code", "algorithm"}
)


@dataclass(frozen=True, slots=True)
class StructureProfile:
    indexable_blocks: int
    title_blocks: int
    structured_blocks: int
    pages: int
    usable_hierarchy: bool

    def as_metadata(self) -> dict[str, int | bool]:
        return {
            "indexable_blocks": self.indexable_blocks,
            "title_blocks": self.title_blocks,
            "structured_blocks": self.structured_blocks,
            "pages": self.pages,
            "usable_hierarchy": self.usable_hierarchy,
        }


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    name: str
    reason: str
    hierarchical: bool


@dataclass(frozen=True, slots=True)
class StructureAwareChunkingResult:
    units: tuple[HierarchicalUnit, ...]
    profile: StructureProfile
    policy: ChunkPolicy


def profile_document_structure(document: ParsedDocument) -> StructureProfile:
    blocks = [block for block in document.blocks if block.indexable]
    titles = sum(block.block_type == "title" for block in blocks)
    structured = sum(block.block_type in _STRUCTURED_TYPES for block in blocks)
    return StructureProfile(
        indexable_blocks=len(blocks),
        title_blocks=titles,
        structured_blocks=structured,
        pages=document.page_count,
        usable_hierarchy=bool(titles or structured),
    )


def resolve_chunk_policy(profile: StructureProfile) -> ChunkPolicy:
    if profile.usable_hierarchy:
        return ChunkPolicy(
            name="structured_parent_child_v1",
            reason="titles or structured regions are available",
            hierarchical=True,
        )
    return ChunkPolicy(
        name="flat_fallback_v1",
        reason="parser produced no usable titles or structured regions",
        hierarchical=False,
    )


def build_structure_aware_units(
    document: ParsedDocument,
    *,
    parent_target_tokens: int = 2_000,
    parent_max_tokens: int = 3_000,
    child_target_tokens: int = 400,
    child_max_tokens: int = 500,
    child_overlap_tokens: int = 60,
    fallback_target_chars: int = 2_000,
    fallback_overlap_chars: int = 0,
) -> StructureAwareChunkingResult:
    """Choose a region-safe hierarchy, with a fixed-length degraded path."""

    profile = profile_document_structure(document)
    policy = resolve_chunk_policy(profile)
    if policy.hierarchical:
        units = build_parent_child_units(
            document,
            parent_target_tokens=parent_target_tokens,
            parent_max_tokens=parent_max_tokens,
            child_target_tokens=child_target_tokens,
            child_max_tokens=child_max_tokens,
            child_overlap_tokens=child_overlap_tokens,
            hybrid_policy=True,
        )
    else:
        units = build_flat_units(
            document,
            target_chars=fallback_target_chars,
            overlap_chars=fallback_overlap_chars,
        )
    return StructureAwareChunkingResult(
        units=units,
        profile=profile,
        policy=policy,
    )


__all__ = [
    "ChunkPolicy",
    "StructureAwareChunkingResult",
    "StructureProfile",
    "build_structure_aware_units",
    "profile_document_structure",
    "resolve_chunk_policy",
]
