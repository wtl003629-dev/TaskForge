"""Run locked QASPER questions through the direct PDF upload RAG path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
import tempfile
import textwrap
import unicodedata
from collections import defaultdict
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app  # noqa: E402
from taskforge.config import Settings  # noqa: E402
from taskforge.knowledge import AccessContext, tokenise  # noqa: E402
from taskforge.qasper_alignment import (  # noqa: E402
    AlignmentChunk,
    align_qasper_gold,
    aligned_recall_at_k,
    alignment_diagnostics,
)
from taskforge.rag_evaluation import (  # noqa: E402
    EvalCorpusDocument,
    load_qasper_dataset,
)
from taskforge.rag_experiment_profile import (  # noqa: E402
    resolve_rag_experiment_profile,
)

RECALL_KS = (1, 5, 10, 50)
DEFAULT_AGENT_VISIBLE_K = 8
REPORT_EVIDENCE_TEXT_CHARS = 3_000
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_CJK_FONT = "TaskForge-CJK-Regular"
_CJK_BOLD_FONT = "TaskForge-CJK-Bold"
_CJK_FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
_CJK_BOLD_FONT_PATH = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def _nearest_rank(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _aligned_ranking_metrics(
    gold: Any,
    alignments: dict[str, Any],
    ranked_child_ids: list[str],
    *,
    ndcg_k: int = 8,
) -> tuple[float, float]:
    """Return best-annotation MRR and evidence-unit NDCG over aligned Child IDs.

    A single annotated evidence unit can align to both the Flat and Child
    lanes in a hybrid index.  MRR already takes the earliest aligned Child,
    but the old NDCG denominator counted every aligned Child as an independent
    relevant item.  That made a correct rank-1 hit score only 0.613 when the
    same Gold unit had two lane alignments.  Collapse aligned Child IDs back to
    one gain per Gold unit while still using the earliest rank of that unit.
    """

    best_mrr = 0.0
    best_ndcg = 0.0
    for evidence_set in gold.evidence_sets:
        unit_relevant_ids = {
            unit.unit_id: {
                span.child_id
                for span in alignments[unit.unit_id].aligned_child_spans
            }
            for unit in evidence_set.units
            if alignments[unit.unit_id].status in {"exact", "fuzzy"}
        }
        unit_relevant_ids = {
            unit_id: child_ids
            for unit_id, child_ids in unit_relevant_ids.items()
            if child_ids
        }
        if not unit_relevant_ids:
            continue
        unit_first_ranks = {
            unit_id: next(
                (
                    rank
                    for rank, child_id in enumerate(ranked_child_ids, start=1)
                    if child_id in child_ids
                ),
                None,
            )
            for unit_id, child_ids in unit_relevant_ids.items()
        }
        first_rank = min(
            (rank for rank in unit_first_ranks.values() if rank is not None),
            default=None,
        )
        if first_rank is not None:
            best_mrr = max(best_mrr, 1.0 / first_rank)
        gains_by_rank: dict[int, float] = {}
        for rank in unit_first_ranks.values():
            if rank is not None and rank <= ndcg_k:
                gains_by_rank[rank] = gains_by_rank.get(rank, 0.0) + 1.0
        gains = [gains_by_rank.get(rank, 0.0) for rank in range(1, ndcg_k + 1)]
        dcg = sum(
            gain / math.log2(rank + 1)
            for rank, gain in enumerate(gains, start=1)
        )
        ideal_hits = min(len(unit_relevant_ids), ndcg_k)
        idcg = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, ideal_hits + 1)
        )
        if idcg:
            best_ndcg = max(best_ndcg, dcg / idcg)
    return best_mrr, best_ndcg


def _settings(
    state: Path,
    *,
    backend: str,
    semantic_model: str | None = None,
    semantic_model_path: Path | None = None,
    semantic_batch_size: int = 8,
    semantic_device: str = "auto",
    graph_enabled: bool,
    reranker_model: str | None,
    multilingual_semantic_model: str | None = None,
    multilingual_reranker_model: str | None = None,
    multilingual_reranker_backend: str = "fastembed",
    reranker_backend: str = "fastembed",
    query_expansion_mode: str = "original",
    feature_ranker_path: Path | None = None,
    structure_fusion_enabled: bool = False,
    structure_section_weight: float = 0.5,
    structure_query_coverage_weight: float = 0.1,
    preserve_head_k: int = 0,
    reranker_context_window: int = 0,
    contextual_child_rerank_enabled: bool = False,
    contextual_child_neighbor_tokens: int = 120,
    contextual_child_max_tokens: int = 500,
    lexical_fusion_weight: float = 0.0,
    intent_section_fusion_enabled: bool = False,
    intent_section_fusion_weight: float = 0.1,
    intent_query_overlap_weight: float = 0.05,
    intent_rank_fusion_weight: float = 0.45,
    pdf_parser_backend: str = "native",
    mineru_base_url: str | None = None,
    mineru_expected_version: str | None = None,
    mineru_cache_root: Path | None = None,
    pdf_chunking_mode: str = "flat",
    pdf_flat_chunk_chars: int = 2_000,
    pdf_flat_overlap_chars: int = 0,
    pdf_parent_target_tokens: int = 2_000,
    pdf_parent_max_tokens: int = 3_000,
    pdf_child_target_tokens: int = 400,
    pdf_child_max_tokens: int = 500,
    pdf_child_overlap_tokens: int = 60,
    operator_budget: int = 2,
    visual_extractor_enabled: bool = False,
    rag_profile: str = "current",
    rag_ablation: str = "e",
    parent_aware_candidate_k: int = 20,
    parent_context_max_tokens: int = 800,
    parent_child_score_weight: float = 0.55,
    parent_context_score_weight: float = 0.35,
    parent_retrieval_score_weight: float = 0.10,
    dual_route_enabled: bool = False,
    dual_route_flat_candidate_k: int = 30,
    dual_route_child_candidate_k: int = 20,
    dual_route_flat_head_k: int = 2,
    dual_route_rerank_candidate_k: int = 10,
    dual_route_tail_rerank_candidate_k: int = 0,
    dual_route_min_confidence: float = 0.35,
    embedding_cache_path: Path | None = None,
) -> Settings:
    host_settings = (
        Settings(_env_file=PROJECT_ROOT / ".env")
        if (
            visual_extractor_enabled
            or backend == "bailian"
            or reranker_backend == "bailian"
            or multilingual_reranker_backend == "bailian"
        )
        else None
    )
    if visual_extractor_enabled and host_settings is not None and (
        host_settings.visual_extractor_base_url is None
        or host_settings.visual_extractor_api_key is None
        or host_settings.visual_extractor_model is None
    ):
        raise ValueError(
            "visual ablation requires the configured base URL, API key, and model ID"
        )
    if backend == "bailian" and (
        host_settings is None or host_settings.bailian_api_key is None
    ):
        raise ValueError(
            "Bailian evaluation requires TASKFORGE_BAILIAN_API_KEY in .env"
        )
    resolved_semantic_model = semantic_model or (
        "BAAI/bge-m3"
        if backend == "flagembedding"
        else "text-embedding-v4"
        if backend == "bailian"
        else "BAAI/bge-small-en-v1.5"
    )
    return Settings(
        _env_file=None,
        database_backend="sqlite",
        sqlite_path=state / "taskforge.sqlite3",
        context_sqlite_path=state / "context.sqlite3",
        rag_active_profile=rag_profile,
        rag_experiment_profile=rag_profile,
        rag_optimized_ablation=rag_ablation,
        rag_evaluation_mode=True,
        operations_sqlite_path=state / "operations.sqlite3",
        orchestration_sqlite_path=state / "orchestration.sqlite3",
        review_case_sqlite_path=state / "review.sqlite3",
        verification_sqlite_path=state / "verification.sqlite3",
        literature_sqlite_path=state / "literature.sqlite3",
        literature_cache_path=state / "literature-cache.sqlite3",
        workspace_root=PROJECT_ROOT,
        artifact_root=state / "artifacts",
        retrieval_routing=(
            "profile"
            if backend in {"fastembed", "flagembedding", "bailian"}
            else "lexical"
        ),
        general_text_backend=backend,
        semantic_model=resolved_semantic_model,
        semantic_model_path=semantic_model_path,
        semantic_batch_size=semantic_batch_size,
        semantic_device=semantic_device,
        semantic_cache_path=(
            embedding_cache_path
            or state
            / (
                "embeddings-bge-m3-v1.sqlite3"
                if backend == "flagembedding"
                else "embeddings-bailian-v4-1024.sqlite3"
                if backend == "bailian"
                else "embeddings.sqlite3"
            )
        ),
        bailian_api_key=(
            host_settings.bailian_api_key.get_secret_value()
            if host_settings is not None
            and host_settings.bailian_api_key is not None
            else None
        ),
        bailian_base_url=(
            host_settings.bailian_base_url
            if host_settings is not None
            else "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        bailian_model=(
            host_settings.bailian_model
            if host_settings is not None
            else "text-embedding-v4"
        ),
        bailian_embedding_dimension=(
            host_settings.bailian_embedding_dimension
            if host_settings is not None
            else 1_024
        ),
        bailian_batch_size=(
            host_settings.bailian_batch_size
            if host_settings is not None
            else 10
        ),
        bailian_timeout_seconds=(
            host_settings.bailian_timeout_seconds
            if host_settings is not None
            else 30.0
        ),
        bailian_max_retries=(
            host_settings.bailian_max_retries
            if host_settings is not None
            else 3
        ),
        bailian_cache_path=(
            embedding_cache_path
            or state / "embeddings-bailian-v4-1024.sqlite3"
        ),
        bailian_index_name=(
            host_settings.bailian_index_name
            if host_settings is not None
            else "knowledge-bailian-text-embedding-v4-1024-v1"
        ),
        fastembed_model_cache_root=(
            PROJECT_ROOT / ".taskforge" / "model-cache" / "fastembed"
        ),
        research_graph_enabled=graph_enabled,
        research_reranker_model=reranker_model,
        research_reranker_backend=reranker_backend,
        research_multilingual_semantic_model=multilingual_semantic_model,
        research_multilingual_reranker_model=multilingual_reranker_model,
        research_multilingual_reranker_backend=multilingual_reranker_backend,
        research_feature_ranker_path=feature_ranker_path,
        research_structure_fusion_enabled=structure_fusion_enabled,
        research_structure_section_weight=structure_section_weight,
        research_structure_query_coverage_weight=structure_query_coverage_weight,
        research_preserve_head_k=preserve_head_k,
        research_reranker_context_window=reranker_context_window,
        research_contextual_child_rerank_enabled=(
            contextual_child_rerank_enabled
        ),
        research_contextual_child_neighbor_tokens=(
            contextual_child_neighbor_tokens
        ),
        research_contextual_child_max_tokens=contextual_child_max_tokens,
        research_lexical_fusion_weight=lexical_fusion_weight,
        research_intent_section_fusion_enabled=intent_section_fusion_enabled,
        research_intent_section_fusion_weight=intent_section_fusion_weight,
        research_intent_query_overlap_weight=intent_query_overlap_weight,
        research_intent_rank_fusion_weight=intent_rank_fusion_weight,
        research_rewrite_enabled=False,
        research_query_expansion_mode=query_expansion_mode,
        provider="demo",
        pdf_parser_backend=pdf_parser_backend,
        mineru_base_url=mineru_base_url,
        mineru_expected_version=mineru_expected_version,
        mineru_cache_root=mineru_cache_root,
        pdf_chunking_mode=pdf_chunking_mode,
        pdf_flat_chunk_chars=pdf_flat_chunk_chars,
        pdf_flat_overlap_chars=pdf_flat_overlap_chars,
        pdf_parent_target_tokens=pdf_parent_target_tokens,
        pdf_parent_max_tokens=pdf_parent_max_tokens,
        pdf_child_target_tokens=pdf_child_target_tokens,
        pdf_child_max_tokens=pdf_child_max_tokens,
        pdf_child_overlap_tokens=pdf_child_overlap_tokens,
        research_operator_budget_standard=operator_budget,
        research_operator_budget_rigorous=operator_budget,
        research_parent_aware_candidate_k=parent_aware_candidate_k,
        research_parent_context_max_tokens=parent_context_max_tokens,
        research_parent_child_score_weight=parent_child_score_weight,
        research_parent_context_score_weight=parent_context_score_weight,
        research_parent_retrieval_score_weight=parent_retrieval_score_weight,
        research_dual_route_enabled=dual_route_enabled,
        research_dual_route_flat_candidate_k=dual_route_flat_candidate_k,
        research_dual_route_child_candidate_k=dual_route_child_candidate_k,
        research_dual_route_flat_head_k=dual_route_flat_head_k,
        research_dual_route_rerank_candidate_k=dual_route_rerank_candidate_k,
        research_dual_route_tail_rerank_candidate_k=dual_route_tail_rerank_candidate_k,
        research_dual_route_min_confidence=dual_route_min_confidence,
        visual_extractor_base_url=(
            host_settings.visual_extractor_base_url if host_settings else None
        ),
        visual_extractor_api_key=(
            host_settings.visual_extractor_api_key if host_settings else None
        ),
        visual_extractor_model=(
            host_settings.visual_extractor_model if host_settings else None
        ),
    )


def _printable(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    replacements = {
        "–": "-",
        "—": "-",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    normalized = "".join(replacements.get(character, character) for character in normalized)
    return normalized.encode("latin-1", "replace").decode("latin-1")


def _render_paper(
    target: Path,
    paper_title: str,
    documents: list[EvalCorpusDocument],
    *,
    compact: bool = False,
) -> dict[str, list[int]]:
    if compact:
        return _render_compact_paper(target, paper_title, documents)
    pdf = canvas.Canvas(str(target), pagesize=A4, pageCompression=1, invariant=1)
    width, height = A4
    page_number = 0
    mapping: dict[str, list[int]] = {}
    for document_index, document in enumerate(documents):
        section = str(
            document.metadata.get("section_title")
            or document.metadata.get("section")
            or "Document"
        )
        lines = textwrap.wrap(
            _printable(document.text),
            width=96,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        first_part = True
        while lines:
            page_number += 1
            mapping.setdefault(document.document_id, []).append(page_number)
            y = height - 48
            if document_index == 0 and first_part:
                pdf.setFont("Helvetica-Bold", 14)
                for title_line in textwrap.wrap(_printable(paper_title), width=72)[:3]:
                    pdf.drawString(48, y, title_line)
                    y -= 18
                y -= 8
            pdf.setFont("Helvetica-Bold", 11)
            suffix = " (continued)" if not first_part else ""
            pdf.drawString(48, y, _printable(section + suffix)[:110])
            y -= 22
            pdf.setFont("Helvetica", 9)
            max_lines = max(1, int((y - 42) // 12))
            current, lines = lines[:max_lines], lines[max_lines:]
            for line in current:
                pdf.drawString(48, y, line)
                y -= 12
            pdf.setFont("Helvetica", 7)
            pdf.drawRightString(width - 48, 24, f"page {page_number}")
            pdf.showPage()
            first_part = False
    pdf.save()
    return mapping


def _render_compact_paper(
    target: Path,
    paper_title: str,
    documents: list[EvalCorpusDocument],
) -> dict[str, list[int]]:
    """Render consecutive source paragraphs into realistic multi-paragraph pages."""

    uses_cjk = bool(
        _CJK_RE.search(paper_title)
        or any(_CJK_RE.search(document.text) for document in documents)
    )
    body_font = "Helvetica"
    heading_font = "Helvetica-Bold"
    title_font = "Helvetica-Bold"
    if uses_cjk:
        if not _CJK_FONT_PATH.is_file() or not _CJK_BOLD_FONT_PATH.is_file():
            raise RuntimeError("Chinese PDF fixture fonts are unavailable")
        registered = pdfmetrics.getRegisteredFontNames()
        if _CJK_FONT not in registered:
            pdfmetrics.registerFont(TTFont(_CJK_FONT, _CJK_FONT_PATH, subfontIndex=0))
        if _CJK_BOLD_FONT not in registered:
            pdfmetrics.registerFont(
                TTFont(_CJK_BOLD_FONT, _CJK_BOLD_FONT_PATH, subfontIndex=0)
            )
        body_font = _CJK_FONT
        heading_font = title_font = _CJK_BOLD_FONT

    def printable(value: str) -> str:
        return value if uses_cjk else _printable(value)

    def wrap_lines(value: str, *, latin_width: int, cjk_width: int) -> list[str]:
        lines: list[str] = []
        for raw_line in value.splitlines() or [""]:
            lines.extend(
                textwrap.wrap(
                    printable(raw_line),
                    width=(cjk_width if _CJK_RE.search(raw_line) else latin_width),
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                or [""]
            )
        return lines

    pdf = canvas.Canvas(str(target), pagesize=A4, pageCompression=1, invariant=1)
    width, height = A4
    page_number = 0
    y = 0.0
    mapping: dict[str, list[int]] = {}
    current_section: str | None = None

    def finish_page() -> None:
        if page_number <= 0:
            return
        pdf.setFont("Helvetica", 7)
        pdf.drawRightString(width - 48, 24, f"page {page_number}")
        pdf.showPage()

    def start_page() -> None:
        nonlocal page_number, y
        page_number += 1
        y = height - 48
        if page_number == 1:
            pdf.setFont(title_font, 14)
            for title_line in wrap_lines(
                paper_title, latin_width=72, cjk_width=35
            )[:3]:
                pdf.drawString(48, y, title_line)
                y -= 18
            y -= 8

    start_page()
    for document in documents:
        section = str(
            document.metadata.get("section_title")
            or document.metadata.get("section")
            or "Document"
        )
        if section != current_section:
            if y < 90:
                finish_page()
                start_page()
            pdf.setFont(heading_font, 11)
            pdf.drawString(48, y, printable(section)[:110])
            y -= 20
            current_section = section
        lines = wrap_lines(document.text, latin_width=96, cjk_width=55)
        mapping.setdefault(document.document_id, [])
        pdf.setFont(body_font, 9)
        while lines:
            if y < 48:
                finish_page()
                start_page()
                pdf.setFont(body_font, 9)
            if page_number not in mapping[document.document_id]:
                mapping[document.document_id].append(page_number)
            pdf.drawString(48, y, lines.pop(0))
            y -= 12
        y -= 8
    finish_page()
    pdf.save()
    return mapping


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def _paper_id(document: EvalCorpusDocument) -> str:
    value = str(document.metadata.get("paper_id") or "").strip()
    if not value:
        raise ValueError("QASPER document is missing paper_id")
    return value


def _gold_evidence_type(text: str) -> str:
    value = str(text).strip()
    if not value.casefold().startswith("float selected:"):
        return "paragraph"
    caption = value.split(":", 1)[-1].strip().casefold()
    if caption.startswith(("table", "tab.")):
        return "table"
    if caption.startswith(("figure", "fig.")):
        return "figure"
    return "visual"


def _alignment_by_evidence_type(
    labels: Any,
    alignments: dict[str, Any],
) -> dict[str, dict[str, int | float]]:
    units: dict[str, Any] = {}
    for evidence_set in labels.evidence_sets:
        for unit in evidence_set.units:
            units.setdefault(unit.unit_id, unit)
    counts: dict[str, dict[str, int]] = {}
    for unit_id, unit in units.items():
        kind = _gold_evidence_type(unit.text)
        status = alignments[unit_id].status
        values = counts.setdefault(
            kind,
            {"total": 0, "exact": 0, "fuzzy": 0, "ambiguous": 0, "unaligned": 0},
        )
        values["total"] += 1
        values[status] += 1
    return {
        kind: {
            **values,
            "alignment_coverage": (
                (values["exact"] + values["fuzzy"]) / values["total"]
                if values["total"]
                else 0.0
            ),
        }
        for kind, values in counts.items()
    }


def _retrieval_failure_stage(
    *,
    candidate_recall: float,
    reranked_top_10_recall: float,
    presented_top_10_recall: float,
) -> str:
    """Attribute the first stage that lost any legal Gold evidence.

    A pending visual that happens to share the returned head is a separate
    parse diagnostic; it is not a retrieval failure when the legal Gold set
    has already been fully presented.
    """

    epsilon = 1e-12
    if candidate_recall < 1.0 - epsilon:
        return "candidate_missing"
    if reranked_top_10_recall < candidate_recall - epsilon:
        return "rerank_top10_missing"
    if presented_top_10_recall < reranked_top_10_recall - epsilon:
        return "presentation_window_missing"
    return "retrieval_success"


def _retrieval_failure_stage_for_visible_k(
    *,
    candidate_recall: float,
    reranked_top_10_recall: float,
    reranked_visible_recall: float,
    presented_visible_recall: float,
) -> str:
    """Attribute loss while keeping the production visible-card budget explicit.

    ``Recall@1/5/10/50`` is measured on the complete reranked list.  The
    production Agent receives only ``agent_visible_k`` query-centred windows,
    so presentation loss must compare the same rank budget rather than
    comparing Top-10 retrieval with an unrelated visible-card count.
    """

    epsilon = 1e-12
    if candidate_recall < 1.0 - epsilon:
        return "candidate_missing"
    if reranked_top_10_recall < candidate_recall - epsilon:
        return "rerank_top10_missing"
    if presented_visible_recall < reranked_visible_recall - epsilon:
        return "presentation_window_missing"
    return "retrieval_success"


def _trace_ranked_ids(
    traces: list[dict[str, Any]],
    field: str,
) -> list[str]:
    """Return the final non-empty ranked ID list emitted by a retrieval trace."""

    for trace in reversed(traces):
        values = trace.get(field, [])
        if not isinstance(values, list):
            continue
        ids = [
            str(item["chunk_id"])
            for item in values
            if isinstance(item, dict) and item.get("chunk_id")
        ]
        if ids:
            return list(dict.fromkeys(ids))
    return []


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_real_pdf_manifest(
    manifest_path: Path,
    required_paper_ids: set[str],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    """Load and verify a pre-registered real-PDF cohort.

    Every selected paper must be present before evaluation begins.  Paths may
    be relative to the manifest, but content is identified by its pinned
    SHA-256 rather than by a mutable filename or URL.
    """

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0" or str(raw.get("dataset") or "").strip() not in {
        "QASPER",
        "TaskForge Paper RAG",
    }:
        raise ValueError(
            "real-PDF manifest must use schema_version 1.0 and a supported paper dataset"
        )
    paper_rows = raw.get("papers")
    if not isinstance(paper_rows, list):
        raise ValueError("real-PDF manifest papers must be a list")
    by_id: dict[str, dict[str, object]] = {}
    for raw_row in paper_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("real-PDF manifest paper rows must be objects")
        paper_id = str(raw_row.get("paper_id") or "").strip()
        if not paper_id or paper_id in by_id:
            raise ValueError(f"invalid or duplicate real-PDF paper_id: {paper_id!r}")
        relative_path = Path(str(raw_row.get("path") or ""))
        path = (
            relative_path
            if relative_path.is_absolute()
            else manifest_path.parent / relative_path
        ).resolve()
        expected_sha256 = str(raw_row.get("sha256") or "").strip().casefold()
        source_url = str(raw_row.get("source_url") or "").strip()
        acquired_at = str(raw_row.get("acquired_at") or "").strip()
        if not _SHA256_RE.fullmatch(expected_sha256):
            raise ValueError(f"invalid SHA-256 for real PDF {paper_id}")
        if not source_url.startswith("https://") or not acquired_at:
            raise ValueError(
                f"real PDF {paper_id} requires HTTPS source_url and acquired_at"
            )
        if not path.is_file():
            raise ValueError(f"real PDF is missing for {paper_id}: {path}")
        with path.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise ValueError(f"real PDF has an invalid header for {paper_id}")
        actual_sha256 = _file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"real PDF checksum mismatch for {paper_id}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        page_count = raw_row.get("page_count")
        if page_count is not None and (
            not isinstance(page_count, int) or isinstance(page_count, bool) or page_count < 1
        ):
            raise ValueError(f"invalid page_count for real PDF {paper_id}")
        by_id[paper_id] = {
            "paper_id": paper_id,
            "path": path,
            "sha256": actual_sha256,
            "source_url": source_url,
            "acquired_at": acquired_at,
            "page_count": page_count,
        }
    missing = sorted(required_paper_ids - by_id.keys())
    if missing:
        raise ValueError(
            "real-PDF manifest is incomplete for the locked cohort: "
            + ", ".join(missing[:5])
        )
    return {paper_id: by_id[paper_id] for paper_id in required_paper_ids}, raw


def _load_query_variants(
    path: Path,
    *,
    cases: list[Any],
    split_sha256: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "1.0":
        raise ValueError("query variant manifest schema must be 1.0")
    if raw.get("split_sha256") != split_sha256:
        raise ValueError("query variant manifest does not match the locked split")
    values = raw.get("variants")
    if not isinstance(values, list):
        raise ValueError("query variant manifest must contain a variants array")
    by_case: dict[str, dict[str, object]] = {}
    for item in values:
        if not isinstance(item, dict):
            raise ValueError("query variant rows must be objects")
        case_id = str(item.get("case_id") or "").strip()
        if not case_id or case_id in by_case:
            raise ValueError("query variant case IDs must be unique and non-empty")
        by_case[case_id] = item
    variants: dict[str, tuple[str, str]] = {}
    for case in cases:
        item = by_case.get(case.case_id)
        if item is None:
            raise ValueError(f"query variants are missing locked case {case.case_id}")
        if str(item.get("query") or "") != case.query:
            raise ValueError(f"query variant text mismatch for {case.case_id}")
        synonym = " ".join(str(item.get("synonym_query") or "").split())
        keyword = " ".join(str(item.get("keyword_query") or "").split())
        if not synonym or not keyword or synonym.casefold() == keyword.casefold():
            raise ValueError(f"query variants are invalid for {case.case_id}")
        current = variants.get(case.query)
        if current is not None and current != (synonym, keyword):
            raise ValueError("duplicate query text has inconsistent frozen variants")
        variants[case.query] = (synonym, keyword)
    return variants, raw


class _FrozenQueryExpander:
    def __init__(self, variants: dict[str, tuple[str, str]]) -> None:
        self.variants = variants

    async def expand(self, query: str, intent: str) -> tuple[str, str]:
        try:
            return self.variants[query]
        except KeyError as exc:
            raise ValueError("locked query has no frozen expansion") from exc


def run(
    dataset_path: Path,
    split_path: Path,
    output: Path,
    *,
    limit: int,
    offset: int = 0,
    backend: str = "bm25",
    semantic_model: str | None = None,
    semantic_model_path: Path | None = None,
    semantic_batch_size: int = 8,
    semantic_device: str = "auto",
    graph_enabled: bool = True,
    reranker_model: str | None = None,
    multilingual_semantic_model: str | None = None,
    multilingual_reranker_model: str | None = None,
    multilingual_reranker_backend: str = "fastembed",
    reranker_backend: str = "fastembed",
    query_expansion_mode: str = "original",
    query_variants_path: Path | None = None,
    minimum_alignment_coverage: float = 0.90,
    minimum_alignment_eligible_case_ratio: float = 0.90,
    candidate_k: int = 50,
    feature_ranker_path: Path | None = None,
    structure_fusion_enabled: bool = False,
    structure_section_weight: float = 0.5,
    structure_query_coverage_weight: float = 0.1,
    preserve_head_k: int = 0,
    reranker_context_window: int = 0,
    contextual_child_rerank_enabled: bool = False,
    contextual_child_neighbor_tokens: int = 120,
    contextual_child_max_tokens: int = 500,
    lexical_fusion_weight: float = 0.0,
    intent_section_fusion_enabled: bool = False,
    intent_section_fusion_weight: float = 0.1,
    intent_query_overlap_weight: float = 0.05,
    intent_rank_fusion_weight: float = 0.45,
    agent_visible_k: int = DEFAULT_AGENT_VISIBLE_K,
    pdf_manifest_path: Path | None = None,
    pdf_parser_backend: str = "native",
    mineru_base_url: str | None = None,
    mineru_expected_version: str | None = None,
    mineru_cache_root: Path | None = None,
    pdf_chunking_mode: str = "flat",
    pdf_flat_chunk_chars: int = 2_000,
    pdf_flat_overlap_chars: int = 0,
    pdf_parent_target_tokens: int = 2_000,
    pdf_parent_max_tokens: int = 3_000,
    pdf_child_target_tokens: int = 400,
    pdf_child_max_tokens: int = 500,
    pdf_child_overlap_tokens: int = 60,
    operator_budget: int = 2,
    visual_extractor_enabled: bool = False,
    rag_profile: str = "current",
    rag_ablation: str = "e",
    parent_aware_candidate_k: int = 20,
    parent_context_max_tokens: int = 800,
    parent_child_score_weight: float = 0.55,
    parent_context_score_weight: float = 0.35,
    parent_retrieval_score_weight: float = 0.10,
    dual_route_enabled: bool = False,
    dual_route_flat_candidate_k: int = 30,
    dual_route_child_candidate_k: int = 20,
    dual_route_flat_head_k: int = 2,
    dual_route_rerank_candidate_k: int = 10,
    dual_route_tail_rerank_candidate_k: int = 0,
    dual_route_min_confidence: float = 0.35,
    embedding_cache_path: Path | None = None,
    state_dir: Path | None = None,
) -> dict[str, object]:
    if candidate_k < 10 or candidate_k > 100:
        raise ValueError("candidate_k must be between 10 and 100")
    if not 1 <= agent_visible_k <= candidate_k:
        raise ValueError("agent_visible_k must be between 1 and candidate_k")
    if query_expansion_mode not in {"original", "keyword", "synonym", "full"}:
        raise ValueError("query_expansion_mode must be original, keyword, synonym, or full")
    if query_expansion_mode != "original" and query_variants_path is None:
        raise ValueError("expanded query ablations require a frozen variant manifest")
    if rag_profile not in {"current", "optimized"}:
        raise ValueError("rag_profile must be current or optimized")
    if rag_ablation not in {"a", "b", "c", "d", "e"}:
        raise ValueError("rag_ablation must be one of a, b, c, d, or e")
    if not 1 <= parent_aware_candidate_k <= 100:
        raise ValueError("parent_aware_candidate_k must be between 1 and 100")
    if not 1 <= dual_route_flat_candidate_k <= 100:
        raise ValueError("dual_route_flat_candidate_k must be between 1 and 100")
    if not 1 <= dual_route_child_candidate_k <= 100:
        raise ValueError("dual_route_child_candidate_k must be between 1 and 100")
    if not 0 <= dual_route_flat_head_k <= 10:
        raise ValueError("dual_route_flat_head_k must be between 0 and 10")
    if not 1 <= dual_route_rerank_candidate_k <= 100:
        raise ValueError("dual_route_rerank_candidate_k must be between 1 and 100")
    if not 0 <= dual_route_tail_rerank_candidate_k <= 100:
        raise ValueError("dual_route_tail_rerank_candidate_k must be between 0 and 100")
    if not 0.0 <= dual_route_min_confidence <= 1.0:
        raise ValueError("dual_route_min_confidence must be between 0 and 1")
    if not 16 <= contextual_child_neighbor_tokens <= 240:
        raise ValueError(
            "contextual_child_neighbor_tokens must be between 16 and 240"
        )
    if not 256 <= contextual_child_max_tokens <= 1_024:
        raise ValueError(
            "contextual_child_max_tokens must be between 256 and 1024"
        )
    if contextual_child_neighbor_tokens * 2 >= contextual_child_max_tokens:
        raise ValueError(
            "contextual Child neighbour budgets must leave room for the target"
        )
    if not 128 <= parent_context_max_tokens <= 3_000:
        raise ValueError("parent_context_max_tokens must be between 128 and 3000")
    if min(
        parent_child_score_weight,
        parent_context_score_weight,
        parent_retrieval_score_weight,
    ) < 0 or (
        parent_child_score_weight
        + parent_context_score_weight
        + parent_retrieval_score_weight
        <= 0
    ):
        raise ValueError("parent-aware weights must be non-negative with positive sum")
    if pdf_parser_backend not in {"native", "mineru"}:
        raise ValueError("scored runs must freeze the parser as native or mineru")
    if pdf_parser_backend == "mineru" and (
        not mineru_base_url or not mineru_expected_version
    ):
        raise ValueError("MinerU scored runs require URL and exact expected version")
    if pdf_chunking_mode not in {"flat", "parent_child", "hybrid", "sliding"}:
        raise ValueError(
            "PDF chunking mode must be flat, parent_child, hybrid, or sliding"
        )
    if dual_route_enabled and pdf_chunking_mode != "hybrid":
        raise ValueError("dual-route evaluation requires pdf_chunking_mode=hybrid")
    if not 256 <= pdf_flat_chunk_chars <= 50_000:
        raise ValueError("pdf_flat_chunk_chars must be between 256 and 50000")
    if not 0 <= pdf_flat_overlap_chars < pdf_flat_chunk_chars:
        raise ValueError("pdf_flat_overlap_chars must be smaller than chunk size")
    if not 500 <= pdf_parent_target_tokens <= pdf_parent_max_tokens <= 8_000:
        raise ValueError("invalid Parent token budgets")
    if not 100 <= pdf_child_target_tokens <= pdf_child_max_tokens <= pdf_parent_max_tokens:
        raise ValueError("invalid Child token budgets")
    if not 0 <= pdf_child_overlap_tokens < pdf_child_target_tokens:
        raise ValueError("invalid Child overlap budget")
    if not 0 <= operator_budget <= 2:
        raise ValueError("operator budget must be between zero and two")
    if not 0.0 < minimum_alignment_coverage <= 1.0:
        raise ValueError("minimum_alignment_coverage must be in (0, 1]")
    if not 0.0 < minimum_alignment_eligible_case_ratio <= 1.0:
        raise ValueError("minimum_alignment_eligible_case_ratio must be in (0, 1]")
    dataset = load_qasper_dataset(dataset_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    locked_ids = [str(item) for item in split["case_ids"]][offset : offset + limit]
    case_by_id = {case.case_id: case for case in dataset.cases}
    missing = [case_id for case_id in locked_ids if case_id not in case_by_id]
    if missing:
        raise ValueError(f"locked QASPER cases are missing: {missing[:3]}")
    cases = [case_by_id[case_id] for case_id in locked_ids]
    query_variants: dict[str, tuple[str, str]] = {}
    query_variants_raw: dict[str, object] | None = None
    if query_variants_path is not None:
        query_variants, query_variants_raw = _load_query_variants(
            query_variants_path,
            cases=cases,
            split_sha256=_file_sha256(split_path),
        )
    paper_ids = {str(case.metadata["paper_id"]) for case in cases}
    documents_by_paper: dict[str, list[EvalCorpusDocument]] = defaultdict(list)
    for document in dataset.documents:
        paper_id = _paper_id(document)
        if paper_id in paper_ids:
            documents_by_paper[paper_id].append(document)

    started = perf_counter()
    if state_dir is not None:
        state_dir = state_dir.resolve()
        if state_dir.exists() and any(state_dir.iterdir()):
            raise ValueError(f"refusing to reuse non-empty evaluation state: {state_dir}")
        state_dir.mkdir(parents=True, exist_ok=True)
    state_context = (
        nullcontext(str(state_dir))
        if state_dir is not None
        else tempfile.TemporaryDirectory(
            prefix="taskforge-qasper-upload-eval-",
            ignore_cleanup_errors=True,
        )
    )
    with state_context as raw:
        state = Path(raw)
        rendered: dict[str, dict[str, Any]] = {}
        manifest_raw: dict[str, object] | None = None
        if pdf_manifest_path is not None:
            real_pdfs, manifest_raw = _load_real_pdf_manifest(
                pdf_manifest_path,
                paper_ids,
            )
            for paper_id, documents in documents_by_paper.items():
                source = real_pdfs[paper_id]
                rendered[paper_id] = {
                    **source,
                    "title": str(
                        documents[0].metadata.get("paper_title") or paper_id
                    ),
                    "evidence_units": len(documents),
                }
        else:
            pdf_root = state / "pdfs"
            pdf_root.mkdir()
            for paper_id, documents in documents_by_paper.items():
                title = str(documents[0].metadata.get("paper_title") or paper_id)
                path = pdf_root / f"{paper_id}.pdf"
                evidence_pages = _render_paper(
                    path,
                    title,
                    documents,
                    compact=(
                        split.get("synthetic_pdf_layout")
                        == "compact_scientific_paper_v1"
                    ),
                )
                rendered[paper_id] = {
                    "path": path,
                    "title": title,
                    "page_count": max(
                        page for pages in evidence_pages.values() for page in pages
                    ),
                    "evidence_units": len(documents),
                }

        application_settings = _settings(
            state,
            backend=backend,
            semantic_model=semantic_model,
            semantic_model_path=semantic_model_path,
            semantic_batch_size=semantic_batch_size,
            semantic_device=semantic_device,
            graph_enabled=graph_enabled,
            reranker_model=reranker_model,
            multilingual_semantic_model=multilingual_semantic_model,
            multilingual_reranker_model=multilingual_reranker_model,
            multilingual_reranker_backend=multilingual_reranker_backend,
            reranker_backend=reranker_backend,
            query_expansion_mode=query_expansion_mode,
            feature_ranker_path=feature_ranker_path,
            structure_fusion_enabled=structure_fusion_enabled,
            structure_section_weight=structure_section_weight,
            structure_query_coverage_weight=structure_query_coverage_weight,
            preserve_head_k=preserve_head_k,
            reranker_context_window=reranker_context_window,
            contextual_child_rerank_enabled=(
                contextual_child_rerank_enabled
            ),
            contextual_child_neighbor_tokens=(
                contextual_child_neighbor_tokens
            ),
            contextual_child_max_tokens=contextual_child_max_tokens,
            lexical_fusion_weight=lexical_fusion_weight,
            intent_section_fusion_enabled=intent_section_fusion_enabled,
            intent_section_fusion_weight=intent_section_fusion_weight,
            intent_query_overlap_weight=intent_query_overlap_weight,
            intent_rank_fusion_weight=intent_rank_fusion_weight,
            pdf_parser_backend=pdf_parser_backend,
            mineru_base_url=mineru_base_url,
            mineru_expected_version=mineru_expected_version,
            mineru_cache_root=mineru_cache_root,
            pdf_chunking_mode=pdf_chunking_mode,
            pdf_flat_chunk_chars=pdf_flat_chunk_chars,
            pdf_flat_overlap_chars=pdf_flat_overlap_chars,
            pdf_parent_target_tokens=pdf_parent_target_tokens,
            pdf_parent_max_tokens=pdf_parent_max_tokens,
            pdf_child_target_tokens=pdf_child_target_tokens,
            pdf_child_max_tokens=pdf_child_max_tokens,
            pdf_child_overlap_tokens=pdf_child_overlap_tokens,
            operator_budget=operator_budget,
            visual_extractor_enabled=visual_extractor_enabled,
            rag_profile=rag_profile,
            rag_ablation=rag_ablation,
            parent_aware_candidate_k=parent_aware_candidate_k,
            parent_context_max_tokens=parent_context_max_tokens,
            parent_child_score_weight=parent_child_score_weight,
            parent_context_score_weight=parent_context_score_weight,
            parent_retrieval_score_weight=parent_retrieval_score_weight,
            dual_route_enabled=dual_route_enabled,
            dual_route_flat_candidate_k=dual_route_flat_candidate_k,
            dual_route_child_candidate_k=dual_route_child_candidate_k,
            dual_route_flat_head_k=dual_route_flat_head_k,
            dual_route_rerank_candidate_k=dual_route_rerank_candidate_k,
            dual_route_tail_rerank_candidate_k=dual_route_tail_rerank_candidate_k,
            dual_route_min_confidence=dual_route_min_confidence,
            embedding_cache_path=embedding_cache_path,
        )
        active_profile = resolve_rag_experiment_profile(
            rag_profile,
            rag_ablation,
        )
        app = create_app(application_settings)
        if query_variants:
            app.state.container.scope_evidence.query_expander = _FrozenQueryExpander(
                query_variants
            )
        auth = {
            "X-TaskForge-Tenant": "qasper-upload-eval",
            "X-TaskForge-User": "evaluator",
        }
        principal = AccessContext(
            tenant_id="qasper-upload-eval",
            user_id="evaluator",
        )
        scopes: dict[str, dict[str, object]] = {}
        chunks_by_paper: dict[str, list[AlignmentChunk]] = {}
        ingestion: list[dict[str, object]] = []
        parser_failures: dict[str, str] = {}
        rows: list[dict[str, object]] = []
        with TestClient(app) as client:
            for paper_id, item in rendered.items():
                path = Path(item["path"])
                uploaded = client.post(
                    "/api/research/uploads",
                    headers={
                        **auth,
                        "Content-Type": "application/pdf",
                        "X-Filename": path.name,
                    },
                    params={
                        "conversation_id": f"qasper-{paper_id}",
                        "user_intent": "Answer questions from this uploaded paper.",
                        "title": str(item["title"]),
                    },
                    content=path.read_bytes(),
                )
                uploaded.raise_for_status()
                scope = uploaded.json()["scope"]
                indexed = client.post(
                    f"/api/research/scopes/{scope['scope_id']}/ingest",
                    headers=auth,
                )
                if not indexed.is_success:
                    parser_failures[paper_id] = (
                        f"HTTP {indexed.status_code}: {indexed.text[:500]}"
                    )
                    chunks_by_paper[paper_id] = []
                    ingestion.append(
                        {
                            "paper_id": paper_id,
                            "pages": item.get("page_count"),
                            "pdf_sha256": item.get("sha256"),
                            "gold_evidence_units": item["evidence_units"],
                            "indexed_chunks": 0,
                            "status": "failed",
                            "failure": parser_failures[paper_id],
                        }
                    )
                    continue
                status = indexed.json()[0]
                if status["status"] != "indexed":
                    parser_failures[paper_id] = str(
                        status.get("error") or status
                    )[:1_000]
                    chunks_by_paper[paper_id] = []
                    ingestion.append(
                        {
                            "paper_id": paper_id,
                            "pages": item.get("page_count"),
                            "pdf_sha256": item.get("sha256"),
                            "gold_evidence_units": item["evidence_units"],
                            "indexed_chunks": 0,
                            "status": "failed",
                            "failure": parser_failures[paper_id],
                        }
                    )
                    continue
                scopes[paper_id] = scope
                knowledge_base_id = (
                    f"research-scope:{scope['scope_id']}:v{scope['scope_version']}"
                )
                knowledge_base_id = active_profile.knowledge_base_id(
                    knowledge_base_id
                )
                indexed_chunks = app.state.container.knowledge_store.visible_chunks(
                    principal,
                    knowledge_base_ids=(knowledge_base_id,),
                    latest_only=True,
                )
                chunks_by_paper[paper_id] = [
                    AlignmentChunk(
                        child_id=chunk.chunk_id,
                        text=chunk.text,
                        order=int(chunk.metadata.get("chunk_index", index)),
                        section=(
                            str(chunk.metadata["heading"])
                            if chunk.metadata.get("heading")
                            else None
                        ),
                    )
                    for index, chunk in enumerate(indexed_chunks)
                    if chunk.metadata.get("retrieval_role") != "parent"
                ]
                parse_quality = next(
                    (
                        chunk.metadata.get("parse_quality")
                        for chunk in indexed_chunks
                        if isinstance(chunk.metadata.get("parse_quality"), dict)
                    ),
                    {},
                )
                parser_name = next(
                    (
                        str(chunk.metadata.get("parser"))
                        for chunk in indexed_chunks
                        if chunk.metadata.get("parser")
                    ),
                    None,
                )
                parser_version = next(
                    (
                        str(chunk.metadata.get("parser_version"))
                        for chunk in indexed_chunks
                        if chunk.metadata.get("parser_version")
                    ),
                    None,
                )
                parser_attempts = next(
                    (
                        chunk.metadata.get("parser_attempts")
                        for chunk in indexed_chunks
                        if isinstance(chunk.metadata.get("parser_attempts"), list)
                    ),
                    [],
                )
                child_chunks = [
                    chunk
                    for chunk in indexed_chunks
                    if chunk.metadata.get("retrieval_role") != "parent"
                ]
                flat_primary_chunks = [
                    chunk
                    for chunk in child_chunks
                    if chunk.metadata.get("hybrid_route") == "flat_primary"
                ]
                child_aux_chunks = [
                    chunk
                    for chunk in child_chunks
                    if chunk.metadata.get("hybrid_route") == "child_aux"
                ]
                parent_chunks = [
                    chunk
                    for chunk in indexed_chunks
                    if chunk.metadata.get("retrieval_role") == "parent"
                ]
                child_token_lengths = [
                    len(tokenise(chunk.text)) for chunk in child_chunks
                ]
                ingestion.append(
                    {
                        "paper_id": paper_id,
                        "pages": item.get("page_count"),
                        "pdf_sha256": item.get("sha256"),
                        "gold_evidence_units": item["evidence_units"],
                        "indexed_chunks": status["evidence_count"],
                        "status": "indexed",
                        "parser": parser_name,
                        "parser_version": parser_version,
                        "parse_quality": parse_quality,
                        "parser_attempts": parser_attempts,
                        "pdf_bytes": path.stat().st_size,
                        "indexed_characters": sum(
                            len(chunk.text)
                            for chunk in indexed_chunks
                            if chunk.metadata.get("retrieval_role") != "parent"
                        ),
                        "index_statistics": {
                            "knowledge_base_id": knowledge_base_id,
                            "child_count": len(child_chunks),
                            "parent_count": len(parent_chunks),
                            "flat_primary_count": len(flat_primary_chunks),
                            "child_aux_count": len(child_aux_chunks),
                            "child_token_lengths": child_token_lengths,
                            "flat_fallback": any(
                                chunk.metadata.get("chunk_policy")
                                == "flat_fallback_v1"
                                for chunk in child_chunks
                            ),
                            "structured_parent_child": any(
                                chunk.metadata.get("chunk_policy")
                                == "structured_parent_child_v1"
                                for chunk in child_chunks
                            ),
                            "table_children": sum(
                                "table"
                                in {
                                    str(value).casefold()
                                    for value in chunk.metadata.get(
                                        "block_types", ()
                                    )
                                }
                                for chunk in child_chunks
                            ),
                            "list_children": sum(
                                "list"
                                in {
                                    str(value).casefold()
                                    for value in chunk.metadata.get(
                                        "block_types", ()
                                    )
                                }
                                for chunk in child_chunks
                            ),
                            "cross_page_children": sum(
                                len(chunk.metadata.get("pages", ())) > 1
                                for chunk in child_chunks
                            ),
                            "heading_children": sum(
                                bool(chunk.metadata.get("heading_path"))
                                for chunk in child_chunks
                            ),
                        },
                    }
                )

            for case in cases:
                paper_id = str(case.metadata["paper_id"])
                if paper_id in parser_failures:
                    alignments = align_qasper_gold(case.qasper_gold, []) if case.qasper_gold else {}
                    alignment = alignment_diagnostics(alignments)
                    rows.append(
                        {
                            "case_id": case.case_id,
                            "paper_id": paper_id,
                            "query": case.query,
                            "gold_annotation_count": (
                                len(case.qasper_gold.evidence_sets)
                                if case.qasper_gold is not None
                                else 0
                            ),
                            "gold_evidence_set_sizes": (
                                [len(item.units) for item in case.qasper_gold.evidence_sets]
                                if case.qasper_gold is not None
                                else []
                            ),
                            "recall_at_k": {str(k): 0.0 for k in RECALL_KS},
                            "candidate_child_recall_at_k": {
                                str(k): 0.0 for k in RECALL_KS
                            },
                            "agent_visible_recall_at_8": 0.0,
                            "reranked_visible_recall_at_8": 0.0,
                            "mrr": 0.0,
                            "ndcg_at_8": 0.0,
                            "citation_localization_hit_at_8": 0.0,
                            "citation_precision_at_8": 0.0,
                            "citation_roundtrip_verified": 0,
                            "citation_roundtrip_checked": 0,
                            "selected_annotation_at_k": {},
                            "candidate_child_selected_annotation_at_k": {},
                            "alignment": alignment.model_dump(mode="json"),
                            "alignment_by_evidence_type": (
                                _alignment_by_evidence_type(case.qasper_gold, alignments)
                                if case.qasper_gold is not None
                                else {}
                            ),
                            "gold_alignments": {
                                unit_id: item.model_dump(mode="json")
                                for unit_id, item in alignments.items()
                            },
                            "alignment_eligible": False,
                            "latency_ms": 0.0,
                            "retrieved_count": 0,
                            "retrieved_child_ids": [],
                            "retrieval_rounds": 0,
                            "retrieval_route": "english",
                            "retrieval_traces": [],
                            "stage_recall": {
                                "candidate_pool": 0.0,
                                "reranked_top_10": 0.0,
                                "reranked_top_8": 0.0,
                                "agent_visible_top_8": 0.0,
                            },
                            "failure_stage": "parser_alignment_failure",
                            "parser_error": parser_failures[paper_id],
                            "retrieved_evidence": [],
                        }
                    )
                    continue
                scope = scopes[paper_id]
                query_started = perf_counter()
                response = client.post(
                    "/api/research/evidence/search",
                    headers=auth,
                    json={
                        "scope_id": scope["scope_id"],
                        "scope_version": scope["scope_version"],
                        "query": case.query,
                        "intent": "general_fact",
                        # Keep the scored request identical to production: the
                        # service reranks the full Candidate@50 pool but only
                        # exposes the bounded Agent-visible head.
                        "top_k": agent_visible_k,
                        "candidate_k": candidate_k,
                        "mode": "rigorous",
                    },
                )
                latency_ms = (perf_counter() - query_started) * 1_000
                response.raise_for_status()
                result = response.json()
                if case.qasper_gold is None:
                    raise RuntimeError(
                        f"QASPER case lacks multi-annotation gold labels: {case.case_id}"
                    )
                alignments = align_qasper_gold(
                    case.qasper_gold,
                    chunks_by_paper[paper_id],
                )
                alignment = alignment_diagnostics(alignments)
                retrieved_child_ids = [
                    str(item["chunk_id"])
                    for item in result["evidence"]
                    if item.get("chunk_id")
                ]
                traces = [
                    trace
                    for trace in result.get("retrieval_traces", [])
                    if isinstance(trace, dict)
                ]
                reranked_child_ids = _trace_ranked_ids(traces, "reranked_hits")
                if not reranked_child_ids:
                    # Defensive fallback for old service adapters that do not
                    # serialize a reranked trace.  New scored runs always use
                    # the complete trace, never the visible prefix, for the
                    # headline retrieval metric.
                    reranked_child_ids = list(retrieved_child_ids)
                recall_results = {
                    str(k): aligned_recall_at_k(
                        case.qasper_gold,
                        alignments,
                        reranked_child_ids,
                        k,
                    )
                    for k in RECALL_KS
                }
                recalls = {
                    key: value.recall for key, value in recall_results.items()
                }
                candidate_child_ids = list(
                    dict.fromkeys(
                        str(hit["chunk_id"])
                        for trace in traces
                        for hit in trace.get("candidate_hits", [])
                        if isinstance(hit, dict) and hit.get("chunk_id")
                    )
                )
                candidate_recall_results = {
                    str(k): aligned_recall_at_k(
                        case.qasper_gold,
                        alignments,
                        candidate_child_ids,
                        k,
                    )
                    for k in RECALL_KS
                }
                candidate_recalls = {
                    key: value.recall
                    for key, value in candidate_recall_results.items()
                }
                candidate_recall = aligned_recall_at_k(
                    case.qasper_gold,
                    alignments,
                    candidate_child_ids,
                    max(1, len(candidate_child_ids)),
                )
                presented_chunks = [
                    AlignmentChunk(
                        child_id=str(item["chunk_id"]),
                        text=str(item.get("snippet") or item.get("text") or ""),
                        order=index,
                    )
                    for index, item in enumerate(result["evidence"])
                    if item.get("chunk_id")
                    and str(item.get("snippet") or item.get("text") or "").strip()
                ]
                presented_alignments = align_qasper_gold(
                    case.qasper_gold,
                    presented_chunks,
                )
                presented_recall_results = {
                    str(k): aligned_recall_at_k(
                        case.qasper_gold,
                        presented_alignments,
                        [item.child_id for item in presented_chunks],
                        k,
                    )
                    for k in RECALL_KS
                }
                presented_recalls = {
                    key: value.recall
                    for key, value in presented_recall_results.items()
                }
                visible_k = min(agent_visible_k, len(retrieved_child_ids))
                reranked_visible_recall = aligned_recall_at_k(
                    case.qasper_gold,
                    alignments,
                    reranked_child_ids,
                    visible_k,
                )
                presented_visible_recall = aligned_recall_at_k(
                    case.qasper_gold,
                    presented_alignments,
                    retrieved_child_ids,
                    visible_k,
                )
                selected_annotation = next(
                    evidence_set
                    for evidence_set in case.qasper_gold.evidence_sets
                    if evidence_set.annotation_id
                    == presented_visible_recall.selected_annotation_id
                )
                relevant_visible_child_ids = {
                    span.child_id
                    for unit in selected_annotation.units
                    for span in alignments[unit.unit_id].aligned_child_spans
                    if alignments[unit.unit_id].status in {"exact", "fuzzy"}
                }
                visible_child_ids = retrieved_child_ids[:visible_k]
                citation_hits = sum(
                    child_id in relevant_visible_child_ids
                    for child_id in visible_child_ids
                )
                citation_roundtrip_verified = 0
                citation_roundtrip_checked = 0
                for evidence_index, item in enumerate(
                    result["evidence"][:visible_k]
                ):
                    claim = str(
                        item.get("snippet") or item.get("text") or ""
                    ).strip()[:450]
                    evidence_id = str(item.get("evidence_id") or "").strip()
                    if not claim or not evidence_id:
                        continue
                    verification = client.post(
                        f"/api/research/scopes/{scope['scope_id']}/claims/verify",
                        headers=auth,
                        json={
                            "claim_id": (
                                f"eval-{hashlib.sha256(f'{case.case_id}:{evidence_index}'.encode()).hexdigest()[:24]}"
                            ),
                            "claim": claim,
                            "evidence_ids": [evidence_id],
                            "risk_level": "low",
                        },
                    )
                    verification.raise_for_status()
                    citation_roundtrip_checked += 1
                    citation_roundtrip_verified += bool(
                        verification.json().get("verified")
                    )
                mrr, ndcg_at_8 = _aligned_ranking_metrics(
                    case.qasper_gold,
                    alignments,
                    reranked_child_ids,
                )
                unresolved_visual_in_top10 = any(
                    bool(item.get("visual_pending"))
                    for item in result["evidence"][:10]
                )
                failure_stage = _retrieval_failure_stage_for_visible_k(
                    candidate_recall=candidate_recall.recall,
                    reranked_top_10_recall=recall_results["10"].recall,
                    reranked_visible_recall=reranked_visible_recall.recall,
                    presented_visible_recall=presented_visible_recall.recall,
                )
                fully_aligned = any(
                    all(
                        alignments[unit.unit_id].status in {"exact", "fuzzy"}
                        for unit in evidence_set.units
                    )
                    for evidence_set in case.qasper_gold.evidence_sets
                )
                rows.append(
                    {
                        "case_id": case.case_id,
                        "paper_id": paper_id,
                        "query": case.query,
                        "gold_annotation_count": len(
                            case.qasper_gold.evidence_sets
                        ),
                        "gold_evidence_set_sizes": [
                            len(item.units)
                            for item in case.qasper_gold.evidence_sets
                        ],
                        # Headline Recall is strict paragraph Recall over the
                        # complete Cross-Encoder-ranked Candidate@50 list.
                        # Agent-visible Recall@8 is kept separately because the
                        # endpoint exposes only query-centred windows.
                        "recall_at_k": recalls if reranked_child_ids else presented_recalls,
                        "candidate_child_recall_at_k": candidate_recalls,
                        "agent_visible_recall_at_8": presented_visible_recall.recall,
                        "reranked_visible_recall_at_8": reranked_visible_recall.recall,
                        "mrr": mrr,
                        "ndcg_at_8": ndcg_at_8,
                        "citation_localization_hit_at_8": float(
                            citation_hits > 0
                        ),
                        "citation_precision_at_8": (
                            citation_hits / len(visible_child_ids)
                            if visible_child_ids
                            else 0.0
                        ),
                        "citation_roundtrip_verified": (
                            citation_roundtrip_verified
                        ),
                        "citation_roundtrip_checked": citation_roundtrip_checked,
                        "selected_annotation_at_k": {
                            key: value.selected_annotation_id
                            for key, value in recall_results.items()
                        },
                        "candidate_child_selected_annotation_at_k": {
                            key: value.selected_annotation_id
                            for key, value in candidate_recall_results.items()
                        },
                        "alignment": alignment.model_dump(mode="json"),
                        "alignment_by_evidence_type": _alignment_by_evidence_type(
                            case.qasper_gold,
                            alignments,
                        ),
                        "gold_alignments": {
                            unit_id: item.model_dump(mode="json")
                            for unit_id, item in alignments.items()
                        },
                        "alignment_eligible": fully_aligned,
                        "latency_ms": latency_ms,
                        "retrieved_count": len(retrieved_child_ids),
                        "retrieved_child_ids": retrieved_child_ids,
                        "retrieval_rounds": result["retrieval_rounds"],
                        "retrieval_route": result.get("retrieval_route", "english"),
                        "activated_operators": result.get(
                            "activated_operators", []
                        ),
                        "retrieval_traces": traces,
                        "stage_recall": {
                            "candidate_pool": candidate_recall.recall,
                            "reranked_top_10": recall_results["10"].recall,
                            "reranked_top_8": reranked_visible_recall.recall,
                            "agent_visible_top_8": presented_visible_recall.recall,
                        },
                        "failure_stage": failure_stage,
                        "unresolved_visual_in_top10": unresolved_visual_in_top10,
                        "retrieved_evidence": [
                            {
                                "evidence_id": item.get("evidence_id"),
                                "chunk_id": item.get("chunk_id"),
                                "page": item.get("page"),
                                "score": item.get("score"),
                                "text": str(
                                    item.get("snippet") or item.get("text") or ""
                                )[:REPORT_EVIDENCE_TEXT_CHARS],
                                "metadata": item.get("metadata", {}),
                                "text_start": item.get("text_start"),
                                "text_end": item.get("text_end"),
                                "presentation_strategy": item.get(
                                    "presentation_strategy"
                                ),
                                "visual_pending": item.get("visual_pending", False),
                                "visual_artifact_ids": item.get(
                                    "visual_artifact_ids", []
                                ),
                                "retrieval_sources": item.get("retrieval_sources", []),
                            }
                            for item in result["evidence"]
                        ],
                    }
                )

    diagnostic_metrics = {
        f"recall_at_{k}": statistics.fmean(
            float(row["recall_at_k"][str(k)])  # type: ignore[index]
            for row in rows
        )
        for k in RECALL_KS
    }
    diagnostic_metrics.update(
        {
            "mrr": statistics.fmean(float(row.get("mrr") or 0.0) for row in rows)
            if rows
            else 0.0,
            "ndcg_at_8": statistics.fmean(
                float(row.get("ndcg_at_8") or 0.0) for row in rows
            )
            if rows
            else 0.0,
        }
    )
    candidate_child_metrics = {
        f"recall_at_{k}": statistics.fmean(
            float(row["candidate_child_recall_at_k"][str(k)])  # type: ignore[index]
            for row in rows
        )
        for k in RECALL_KS
    }
    agent_visible_metrics = {
        "recall_at_8": statistics.fmean(
            float(row.get("agent_visible_recall_at_8") or 0.0)
            for row in rows
        )
        if rows
        else 0.0,
    }
    reranked_visible_metrics = {
        "recall_at_8": statistics.fmean(
            float(row.get("reranked_visible_recall_at_8") or 0.0)
            for row in rows
        )
        if rows
        else 0.0,
    }
    citation_checked = sum(
        int(row.get("citation_roundtrip_checked") or 0) for row in rows
    )
    citation_metrics = {
        "localization_hit_rate_at_8": statistics.fmean(
            float(row.get("citation_localization_hit_at_8") or 0.0)
            for row in rows
        )
        if rows
        else 0.0,
        "precision_at_8": statistics.fmean(
            float(row.get("citation_precision_at_8") or 0.0)
            for row in rows
        )
        if rows
        else 0.0,
        "roundtrip_verification_accuracy": (
            sum(int(row.get("citation_roundtrip_verified") or 0) for row in rows)
            / citation_checked
            if citation_checked
            else 0.0
        ),
        "roundtrip_checked": citation_checked,
        "definition": (
            "Localization/precision use the best valid Gold annotation over the "
            "Agent-visible Child head. Roundtrip verification submits each returned "
            "excerpt against its authoritative Child evidence ID."
        ),
    }
    retrieval_route_counts: dict[str, int] = {}
    for row in rows:
        route = str(row.get("retrieval_route") or "english")
        retrieval_route_counts[route] = retrieval_route_counts.get(route, 0) + 1
    latencies = [float(row["latency_ms"]) for row in rows]
    alignment_eligible_rows = [
        row for row in rows if bool(row["alignment_eligible"])
    ]
    alignment_totals = {
        key: sum(int(row["alignment"][key]) for row in rows)  # type: ignore[index]
        for key in (
            "total_units",
            "exact_units",
            "fuzzy_units",
            "ambiguous_units",
            "unaligned_units",
        )
    }
    alignment_by_type: dict[str, dict[str, int | float]] = {}
    for row in rows:
        for kind, raw_values in row.get("alignment_by_evidence_type", {}).items():  # type: ignore[union-attr]
            values = alignment_by_type.setdefault(
                str(kind),
                {"total": 0, "exact": 0, "fuzzy": 0, "ambiguous": 0, "unaligned": 0},
            )
            for key in ("total", "exact", "fuzzy", "ambiguous", "unaligned"):
                values[key] = int(values[key]) + int(raw_values[key])
    for values in alignment_by_type.values():
        total = int(values["total"])
        values["alignment_coverage"] = (
            (int(values["exact"]) + int(values["fuzzy"])) / total
            if total
            else 0.0
        )
    diagnostic_metrics.update(
        {
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
        }
    )
    real_pdf_track = pdf_manifest_path is not None
    unit_alignment_coverage = (
        (
            alignment_totals["exact_units"]
            + alignment_totals["fuzzy_units"]
        )
        / alignment_totals["total_units"]
        if alignment_totals["total_units"]
        else 0.0
    )
    eligible_case_ratio = len(alignment_eligible_rows) / len(rows) if rows else 0.0
    alignment_gate_passed = (
        unit_alignment_coverage >= minimum_alignment_coverage
        and eligible_case_ratio >= minimum_alignment_eligible_case_ratio
    )
    paper_index_statistics = [
        value
        for item in ingestion
        if isinstance((value := item.get("index_statistics")), dict)
    ]
    child_token_lengths = [
        int(length)
        for item in paper_index_statistics
        for length in item.get("child_token_lengths", [])
    ]
    total_children = sum(
        int(item.get("child_count", 0)) for item in paper_index_statistics
    )
    total_parents = sum(
        int(item.get("parent_count", 0)) for item in paper_index_statistics
    )
    total_flat_primary = sum(
        int(item.get("flat_primary_count", 0))
        for item in paper_index_statistics
    )
    total_child_aux = sum(
        int(item.get("child_aux_count", 0)) for item in paper_index_statistics
    )
    index_statistics = {
        "profile_label": active_profile.label,
        "knowledge_base_identity": (
            "current legacy identity"
            if active_profile.name == "current"
            else "independent optimized identity with profile/ablation suffix"
        ),
        "papers_indexed": len(paper_index_statistics),
        "child_count": total_children,
        "parent_count": total_parents,
        "flat_primary_count": total_flat_primary,
        "child_aux_count": total_child_aux,
        "parent_child_ratio": (
            total_parents / total_children if total_children else 0.0
        ),
        "child_tokens": {
            "minimum": min(child_token_lengths) if child_token_lengths else 0,
            "p50": _nearest_rank(child_token_lengths, 0.50),
            "p95": _nearest_rank(child_token_lengths, 0.95),
            "maximum": max(child_token_lengths) if child_token_lengths else 0,
            "mean": (
                statistics.fmean(child_token_lengths)
                if child_token_lengths
                else 0.0
            ),
        },
        "flat_fallback_papers": sum(
            bool(item.get("flat_fallback")) for item in paper_index_statistics
        ),
        "flat_fallback_ratio": (
            sum(bool(item.get("flat_fallback")) for item in paper_index_statistics)
            / len(paper_index_statistics)
            if paper_index_statistics
            else 0.0
        ),
        "structured_parent_child_papers": sum(
            bool(item.get("structured_parent_child"))
            for item in paper_index_statistics
        ),
        "table_children": sum(
            int(item.get("table_children", 0)) for item in paper_index_statistics
        ),
        "list_children": sum(
            int(item.get("list_children", 0)) for item in paper_index_statistics
        ),
        "cross_page_children": sum(
            int(item.get("cross_page_children", 0))
            for item in paper_index_statistics
        ),
        "heading_children": sum(
            int(item.get("heading_children", 0)) for item in paper_index_statistics
        ),
    }
    metrics = (
        diagnostic_metrics
        if alignment_gate_passed
        else {
            "recall_at_1": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "recall_at_50": None,
            "mrr": None,
            "ndcg_at_8": None,
            "p50_ms": diagnostic_metrics["p50_ms"],
            "p95_ms": diagnostic_metrics["p95_ms"],
        }
    )
    raw_report_metadata = split.get("report_metadata", {})
    report_metadata = (
        raw_report_metadata if isinstance(raw_report_metadata, dict) else {}
    )
    default_evaluation_type = (
        "qasper_real_pdf_upload_retrieval"
        if real_pdf_track
        else "qasper_synthetic_pdf_parser_regression"
    )
    default_benchmark_track = (
        "real_pdf_upload_retrieval"
        if real_pdf_track
        else "synthetic_pdf_parser_regression"
    )
    report: dict[str, object] = {
        "schema_version": "2.3",
        "status": "complete" if alignment_gate_passed else "alignment_gate_failed",
        "evaluation_type": str(
            report_metadata.get("evaluation_type") or default_evaluation_type
        ),
        "benchmark_track": str(
            report_metadata.get("benchmark_track") or default_benchmark_track
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "state": {
            "persistent": state_dir is not None,
            "path": str(state_dir) if state_dir is not None else None,
        },
        "dataset": str(
            report_metadata.get("dataset")
            or f"QASPER v0.3 {split.get('source_split', 'locked')} split"
        ),
        "license": str(report_metadata.get("license") or "CC BY 4.0"),
        "source_dataset": str(dataset_path),
        "source_dataset_sha256": _file_sha256(dataset_path),
        "split": str(split_path),
        "split_sha256": _file_sha256(split_path),
        "scenario": split.get("scenario"),
        "metric_interpretation": split.get("metric_interpretation"),
        "pdf_manifest": (
            {
                "path": str(pdf_manifest_path),
                "sha256": _file_sha256(pdf_manifest_path),
                "cohort_id": manifest_raw.get("cohort_id") if manifest_raw else None,
                "papers": [
                    {
                        key: item.get(key)
                        for key in (
                            "paper_id",
                            "sha256",
                            "source_url",
                            "acquired_at",
                            "page_count",
                        )
                    }
                    for item in rendered.values()
                ],
            }
            if real_pdf_track
            else None
        ),
        "papers": len(rendered),
        "cases": len(rows),
        "case_offset": offset,
        "pipeline": (
            ["locked_real_pdf", "direct_upload", "parse", "chunk", "index", "search"]
            if real_pdf_track
            else ["render_pdf", "direct_upload", "parse", "chunk", "index", "search"]
        ),
        "rag_profile": {
            "name": rag_profile,
            "ablation": "a" if rag_profile == "current" else rag_ablation,
        },
        "retrieval": {
            "backend": backend,
            "semantic_model": semantic_model
            or (
                "BAAI/bge-m3"
                if backend == "flagembedding"
                else "text-embedding-v4"
                if backend == "bailian"
                else "BAAI/bge-small-en-v1.5"
            ),
            "semantic_model_path": (
                str(semantic_model_path) if semantic_model_path is not None else None
            ),
            "embedding_cache_path": (
                str(embedding_cache_path) if embedding_cache_path is not None else None
            ),
            "semantic_batch_size": semantic_batch_size,
            "semantic_device": semantic_device,
            "graph_enabled": graph_enabled,
            "reranker_backend": reranker_backend,
            "reranker_model": reranker_model,
            "multilingual_semantic_model": multilingual_semantic_model,
            "multilingual_reranker_backend": multilingual_reranker_backend,
            "multilingual_reranker_model": multilingual_reranker_model,
            "query_expansion_mode": query_expansion_mode,
            "query_variants": (
                {
                    "path": str(query_variants_path),
                    "sha256": _file_sha256(query_variants_path),
                    "generator": query_variants_raw.get("generator")
                    if query_variants_raw
                    else None,
                }
                if query_variants_path is not None
                else None
            ),
            "candidate_k": candidate_k,
            "agent_visible_k": agent_visible_k,
            "operator_budget": operator_budget,
            "contextual_child_rerank_enabled": (
                contextual_child_rerank_enabled
            ),
            "contextual_child_neighbor_tokens": (
                contextual_child_neighbor_tokens
            ),
            "contextual_child_max_tokens": contextual_child_max_tokens,
            "parent_aware_candidate_k": parent_aware_candidate_k,
            "parent_context_max_tokens": parent_context_max_tokens,
            "parent_score_weights": {
                "child": parent_child_score_weight,
                "context": parent_context_score_weight,
                "retrieval": parent_retrieval_score_weight,
            },
            "dual_route_enabled": dual_route_enabled,
            "dual_route_flat_candidate_k": dual_route_flat_candidate_k,
            "dual_route_child_candidate_k": dual_route_child_candidate_k,
            "dual_route_flat_head_k": dual_route_flat_head_k,
            "dual_route_rerank_candidate_k": dual_route_rerank_candidate_k,
            "dual_route_tail_rerank_candidate_k": dual_route_tail_rerank_candidate_k,
            "dual_route_min_confidence": dual_route_min_confidence,
            "retrieval_route_counts": retrieval_route_counts,
        },
        "parser": {
            "backend": pdf_parser_backend,
            "synthetic_pdf_layout": split.get("synthetic_pdf_layout"),
            "mineru_base_url_configured": bool(mineru_base_url),
            "mineru_expected_version": mineru_expected_version,
            "chunking_mode": pdf_chunking_mode,
            "flat_chunk_chars": pdf_flat_chunk_chars,
            "flat_overlap_chars": pdf_flat_overlap_chars,
            "parent_target_tokens": pdf_parent_target_tokens,
            "parent_max_tokens": pdf_parent_max_tokens,
            "child_target_tokens": pdf_child_target_tokens,
            "child_max_tokens": pdf_child_max_tokens,
            "child_overlap_tokens": pdf_child_overlap_tokens,
            "visual_extractor_enabled": visual_extractor_enabled,
            "visual_extractor_model": (
                application_settings.visual_extractor_model
                if visual_extractor_enabled
                else None
            ),
        },
        "metrics": metrics,
        "index_statistics": index_statistics,
        "ranking_metric_definition": (
            "MRR uses the earliest Gold-aligned Child ID; NDCG@8 uses one binary "
            "gain per Gold evidence unit at the earliest aligned Child rank, "
            "and both take the best valid QASPER annotation, matching Recall's "
            "multi-reference policy."
        ),
        "candidate_child_metrics": {
            **candidate_child_metrics,
            "definition": (
                "Gold paragraph coverage inside complete retrieved Child chunks; "
                "diagnostic only, not Agent-visible Recall."
            ),
        },
        "agent_visible_metrics": agent_visible_metrics,
        "citation_metrics": citation_metrics,
        "reranked_visible_metrics": reranked_visible_metrics,
        "diagnostic_lower_bound_metrics": diagnostic_metrics,
        "conditional_retrieval_metrics": {
            f"recall_at_{k}": statistics.fmean(
                float(row["recall_at_k"][str(k)])  # type: ignore[index]
                for row in alignment_eligible_rows
            )
            if alignment_eligible_rows
            else 0.0
            for k in RECALL_KS
        },
        "alignment_diagnostics": {
            **alignment_totals,
            "alignment_coverage": unit_alignment_coverage,
            "fully_aligned_cases": len(alignment_eligible_rows),
            "alignment_eligible_case_ratio": eligible_case_ratio,
            "by_evidence_type": alignment_by_type,
        },
        "alignment_gate": {
            "passed": alignment_gate_passed,
            "minimum_unit_coverage": minimum_alignment_coverage,
            "minimum_eligible_case_ratio": minimum_alignment_eligible_case_ratio,
            "reason": (
                None
                if alignment_gate_passed
                else (
                    "Gold-to-Child alignment is below the frozen quality gate; "
                    "headline Recall is suppressed until parser/alignment repair."
                )
            ),
        },
        "ingestion": ingestion,
        "parser_diagnostics": {
            "papers": len(ingestion),
            "failed_papers": sum(item.get("status") == "failed" for item in ingestion),
            "parser_failure_rate": (
                sum(item.get("status") == "failed" for item in ingestion)
                / len(ingestion)
                if ingestion
                else 0.0
            ),
            "ocr_used_papers": sum(
                bool(item.get("parse_quality", {}).get("ocr_used"))  # type: ignore[union-attr]
                for item in ingestion
            ),
            "visual_pending_blocks": sum(
                int(item.get("parse_quality", {}).get("visual_unparsed_count") or 0)  # type: ignore[union-attr]
                for item in ingestion
            ),
            "mean_page_coverage": (
                statistics.fmean(
                    float(item.get("parse_quality", {}).get("text_coverage") or 0.0)  # type: ignore[union-attr]
                    for item in ingestion
                )
                if ingestion
                else 0.0
            ),
            "pdf_bytes": sum(int(item.get("pdf_bytes") or 0) for item in ingestion),
            "indexed_characters": sum(
                int(item.get("indexed_characters") or 0) for item in ingestion
            ),
        },
        "rows": rows,
        "elapsed_ms": (perf_counter() - started) * 1_000,
        "limitations": [
            *(
                [
                    "The PDF cohort is pre-registered and checksum-pinned; parser failures remain scored rows and cannot be dropped after results are observed."
                ]
                if real_pdf_track
                else [
                    str(
                        report_metadata.get("synthetic_layout_limitation")
                        or "QASPER text and labels are real; the PDF layout is "
                        "generated locally and this is not a real-PDF benchmark."
                    )
                ]
            ),
            "Headline Recall is strict Gold paragraph Recall over the complete Cross-Encoder-ranked Candidate@50 list; page overlap is never scored.",
            f"The production paper_search response exposes {agent_visible_k} query-centred windows. Agent-visible Recall@{agent_visible_k} is diagnostic and is scored separately from complete-Child retrieval Recall.",
            *(
                [
                    "Alignment quality failed the frozen gate, so diagnostic retrieval values are not publishable Recall metrics."
                ]
                if not alignment_gate_passed
                else []
            ),
            f"The run uses the {backend} product retrieval path with graph_enabled={graph_enabled}.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT
        / "eval"
        / "splits"
        / "qasper-dev-clean-holdout-100-v2.json",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--rag-profile",
        choices=("current", "optimized"),
        default="current",
        help="Isolated RAG chain to index and evaluate.",
    )
    parser.add_argument(
        "--rag-ablation",
        choices=("a", "b", "c", "d", "e"),
        default="e",
        help="Optimized A-E stage; current always resolves to A.",
    )
    parser.add_argument(
        "--backend",
        choices=("bm25", "fastembed", "flagembedding", "bailian"),
        default="bm25",
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help=(
            "Dense embedding model; defaults to bge-small-en for fastembed "
            "BAAI/bge-m3 for flagembedding, and text-embedding-v4 for Bailian."
        ),
    )
    parser.add_argument(
        "--semantic-model-path",
        type=Path,
        default=None,
        help="Local BGE-M3 model directory for the flagembedding backend.",
    )
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="Optional shared text-keyed embedding cache for paired eval runs.",
    )
    parser.add_argument(
        "--semantic-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--reranker-model",
        default=None,
        help="Optional local fastembed cross-encoder model.",
    )
    parser.add_argument(
        "--multilingual-semantic-model",
        default=None,
        help="Optional local multilingual embedding model for CJK/cross-lingual cases.",
    )
    parser.add_argument(
        "--multilingual-reranker-model",
        default=None,
        help="Optional local multilingual cross-encoder for CJK/cross-lingual cases.",
    )
    parser.add_argument(
        "--multilingual-reranker-backend",
        choices=("fastembed", "fastembed_ensemble", "flagembedding", "transformers", "bailian"),
        default="fastembed",
        help="Adapter for --multilingual-reranker-model.",
    )
    parser.add_argument(
        "--reranker-backend",
        choices=("fastembed", "fastembed_ensemble", "flagembedding", "transformers", "bailian"),
        default="fastembed",
        help="Cross-encoder adapter; ensemble model names are comma-separated. Bailian uses qwen3-rerank.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Disable graph feature reranking for an ablation run.",
    )
    parser.add_argument(
        "--query-expansion-mode",
        choices=("original", "keyword", "synonym", "full"),
        default="original",
        help="Locked ablation: original; +keyword; +synonym; or +synonym+keyword.",
    )
    parser.add_argument(
        "--query-variants",
        type=Path,
        default=None,
        help="Checksum-recorded frozen query variant manifest required by expanded modes.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Number of candidates retrieved and reranked (10-100).",
    )
    parser.add_argument(
        "--agent-visible-k",
        type=int,
        default=DEFAULT_AGENT_VISIBLE_K,
        help="Number of query-centred evidence windows exposed to the Agent (default: 8).",
    )
    parser.add_argument("--parent-aware-candidate-k", type=int, default=20)
    parser.add_argument("--parent-context-max-tokens", type=int, default=800)
    parser.add_argument("--parent-child-score-weight", type=float, default=0.55)
    parser.add_argument("--parent-context-score-weight", type=float, default=0.35)
    parser.add_argument("--parent-retrieval-score-weight", type=float, default=0.10)
    parser.add_argument(
        "--dual-route",
        action="store_true",
        help=(
            "Use an explicit Flat-primary + Child-auxiliary hybrid index. "
            "Requires --pdf-chunking-mode hybrid."
        ),
    )
    parser.add_argument("--dual-route-flat-candidate-k", type=int, default=30)
    parser.add_argument("--dual-route-child-candidate-k", type=int, default=20)
    parser.add_argument("--dual-route-flat-head-k", type=int, default=2)
    parser.add_argument("--dual-route-rerank-candidate-k", type=int, default=10)
    parser.add_argument("--dual-route-tail-rerank-candidate-k", type=int, default=0)
    parser.add_argument("--dual-route-min-confidence", type=float, default=0.35)
    parser.add_argument("--minimum-alignment-coverage", type=float, default=0.90)
    parser.add_argument(
        "--minimum-alignment-eligible-case-ratio",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--feature-ranker-path",
        type=Path,
        default=None,
        help="Optional serialized supervised feature-ranker report.",
    )
    parser.add_argument("--structure-fusion", action="store_true")
    parser.add_argument("--structure-section-weight", type=float, default=0.5)
    parser.add_argument("--structure-query-coverage-weight", type=float, default=0.1)
    parser.add_argument("--preserve-head-k", type=int, default=0)
    parser.add_argument("--reranker-context-window", type=int, default=0)
    parser.add_argument(
        "--contextual-child-rerank",
        action="store_true",
        help=(
            "Rerank optimized Child candidates with bounded same-Parent "
            "previous/target/next context; citations remain the target Child."
        ),
    )
    parser.add_argument(
        "--contextual-child-neighbor-tokens",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--contextual-child-max-tokens",
        type=int,
        default=500,
    )
    parser.add_argument("--lexical-fusion-weight", type=float, default=0.0)
    parser.add_argument("--intent-section-fusion", action="store_true")
    parser.add_argument("--intent-section-fusion-weight", type=float, default=0.1)
    parser.add_argument("--intent-query-overlap-weight", type=float, default=0.05)
    parser.add_argument("--intent-rank-fusion-weight", type=float, default=0.45)
    parser.add_argument(
        "--pdf-parser-backend",
        choices=("native", "mineru"),
        default="native",
        help="Freeze one parser backend for a scored ablation; auto is forbidden.",
    )
    parser.add_argument("--mineru-base-url", default=None)
    parser.add_argument("--mineru-expected-version", default=None)
    parser.add_argument(
        "--mineru-cache-root",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "mineru-shared-v1",
        help="Persistent SHA-keyed MinerU response cache; keep on the D drive.",
    )
    parser.add_argument(
        "--pdf-chunking-mode",
        choices=("flat", "parent_child", "hybrid", "sliding"),
        default="flat",
    )
    parser.add_argument(
        "--pdf-flat-chunk-chars",
        type=int,
        default=2_000,
        help="Flat PDF target size in characters; complete blocks remain atomic.",
    )
    parser.add_argument(
        "--pdf-flat-overlap-chars",
        type=int,
        default=0,
        help="Flat PDF same-page overlap in characters, applied by whole blocks.",
    )
    parser.add_argument("--pdf-parent-target-tokens", type=int, default=2_000)
    parser.add_argument("--pdf-parent-max-tokens", type=int, default=3_000)
    parser.add_argument("--pdf-child-target-tokens", type=int, default=400)
    parser.add_argument("--pdf-child-max-tokens", type=int, default=500)
    parser.add_argument("--pdf-child-overlap-tokens", type=int, default=60)
    parser.add_argument(
        "--operator-budget",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Zero disables directed supplementation; nonzero allows one second round.",
    )
    parser.add_argument(
        "--enable-visual-extractor",
        action="store_true",
        help=(
            "Use the separately configured VLM for pending image/chart blocks. "
            "This may make billable external calls."
        ),
    )
    parser.add_argument(
        "--confirm-visual-calls",
        action="store_true",
        help="Required acknowledgement for --enable-visual-extractor.",
    )
    parser.add_argument(
        "--pdf-manifest",
        type=Path,
        default=None,
        help=(
            "Optional checksum-pinned QASPER real-PDF manifest. Without it, "
            "the command runs only the synthetic parser regression track."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "qasper-direct-upload.json",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Persist this run's isolated SQLite index; must be empty or absent.",
    )
    args = parser.parse_args()
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")
    split_case_count = len(json.loads(args.split.read_text(encoding="utf-8"))["case_ids"])
    if args.offset < 0 or args.offset + args.limit > split_case_count:
        raise SystemExit(
            f"--offset + --limit must select cases within the split ({split_case_count})"
        )
    if args.enable_visual_extractor and not args.confirm_visual_calls:
        raise SystemExit(
            "refusing external visual-model calls without --confirm-visual-calls"
        )
    report = run(
        args.dataset,
        args.split,
        args.output,
        limit=args.limit,
        offset=args.offset,
        backend=args.backend,
        semantic_model=args.semantic_model,
        semantic_model_path=args.semantic_model_path,
        semantic_batch_size=args.semantic_batch_size,
        semantic_device=args.semantic_device,
        graph_enabled=not args.no_graph,
        reranker_model=args.reranker_model,
        multilingual_semantic_model=args.multilingual_semantic_model,
        multilingual_reranker_model=args.multilingual_reranker_model,
        multilingual_reranker_backend=args.multilingual_reranker_backend,
        reranker_backend=args.reranker_backend,
        query_expansion_mode=args.query_expansion_mode,
        query_variants_path=args.query_variants,
        minimum_alignment_coverage=args.minimum_alignment_coverage,
        minimum_alignment_eligible_case_ratio=(
            args.minimum_alignment_eligible_case_ratio
        ),
        candidate_k=args.candidate_k,
        feature_ranker_path=args.feature_ranker_path,
        structure_fusion_enabled=args.structure_fusion,
        structure_section_weight=args.structure_section_weight,
        structure_query_coverage_weight=args.structure_query_coverage_weight,
        preserve_head_k=args.preserve_head_k,
        reranker_context_window=args.reranker_context_window,
        contextual_child_rerank_enabled=args.contextual_child_rerank,
        contextual_child_neighbor_tokens=args.contextual_child_neighbor_tokens,
        contextual_child_max_tokens=args.contextual_child_max_tokens,
        lexical_fusion_weight=args.lexical_fusion_weight,
        intent_section_fusion_enabled=args.intent_section_fusion,
        intent_section_fusion_weight=args.intent_section_fusion_weight,
        intent_query_overlap_weight=args.intent_query_overlap_weight,
        intent_rank_fusion_weight=args.intent_rank_fusion_weight,
        agent_visible_k=args.agent_visible_k,
        pdf_manifest_path=args.pdf_manifest,
        pdf_parser_backend=args.pdf_parser_backend,
        mineru_base_url=args.mineru_base_url,
        mineru_expected_version=args.mineru_expected_version,
        mineru_cache_root=args.mineru_cache_root,
        pdf_chunking_mode=args.pdf_chunking_mode,
        pdf_flat_chunk_chars=args.pdf_flat_chunk_chars,
        pdf_flat_overlap_chars=args.pdf_flat_overlap_chars,
        pdf_parent_target_tokens=args.pdf_parent_target_tokens,
        pdf_parent_max_tokens=args.pdf_parent_max_tokens,
        pdf_child_target_tokens=args.pdf_child_target_tokens,
        pdf_child_max_tokens=args.pdf_child_max_tokens,
        pdf_child_overlap_tokens=args.pdf_child_overlap_tokens,
        operator_budget=args.operator_budget,
        visual_extractor_enabled=args.enable_visual_extractor,
        rag_profile=args.rag_profile,
        rag_ablation=args.rag_ablation,
        parent_aware_candidate_k=args.parent_aware_candidate_k,
        parent_context_max_tokens=args.parent_context_max_tokens,
        parent_child_score_weight=args.parent_child_score_weight,
        parent_context_score_weight=args.parent_context_score_weight,
        parent_retrieval_score_weight=args.parent_retrieval_score_weight,
        dual_route_enabled=args.dual_route,
        dual_route_flat_candidate_k=args.dual_route_flat_candidate_k,
        dual_route_child_candidate_k=args.dual_route_child_candidate_k,
        dual_route_flat_head_k=args.dual_route_flat_head_k,
        dual_route_rerank_candidate_k=args.dual_route_rerank_candidate_k,
        dual_route_tail_rerank_candidate_k=args.dual_route_tail_rerank_candidate_k,
        dual_route_min_confidence=args.dual_route_min_confidence,
        embedding_cache_path=args.embedding_cache,
        state_dir=args.state_dir,
    )
    # Keep the console output ASCII-safe on Windows; the complete Unicode
    # report is already persisted at --output.
    print(json.dumps({"output": str(args.output), "metrics": report["metrics"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
