"""Evaluate the real TaskForge four-Agent chain on frozen QASPER retrieval.

The script replays the first eight evidence cards from a hashed, strict
paragraph-level PDF retrieval report through the production
Planner -> Evaluator -> Writer -> Critic orchestration.  Retrieval is frozen so
that answer-quality changes cannot be confused with a different PDF parse,
query rewrite, or reranker result.

This is billable and refuses to run without ``--confirm-live-call``.  Agent
outputs and scored predictions are checkpointed separately, so an interrupted
semantic-judge pass never causes completed four-Agent cases to run again.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
SCRIPTS_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_qasper_answer_e2e import (  # noqa: E402
    SemanticJudgement,
    _judge_answer,
    _merge_usage,
    _sha256,
    _validate_retrieval_report,
    citation_metrics,
    qasper_answer_references,
    qasper_gold_evidence_texts,
    qasper_reference_answer,
)

from taskforge.app import create_app  # noqa: E402
from taskforge.config import Settings  # noqa: E402
from taskforge.domain import utc_now  # noqa: E402
from taskforge.knowledge import tokenise  # noqa: E402
from taskforge.literature.evidence import ScopeBoundEvidenceService  # noqa: E402
from taskforge.literature.repository import (  # noqa: E402
    LiteratureAccess,
    LiteratureConflictError,
)
from taskforge.openai_provider import OpenAIChatCompletionsProvider  # noqa: E402
from taskforge.qasper_alignment import (  # noqa: E402
    AlignmentChunk,
    GoldAlignment,
    align_gold_unit,
    alignment_coverage_for_children,
)
from taskforge.rag_evaluation import (  # noqa: E402
    RAGEvalCase,
    answer_exact_match,
    answer_token_f1,
    load_qasper_dataset,
)
from taskforge.research_protocol import (  # noqa: E402
    CriticHandoff,
    EvidenceCard,
    EvidenceSearchRequest,
    LiteratureRequest,
    PaperCard,
    ResearchScope,
    ScopeEvidenceResult,
    WriterHandoff,
    project_final_research_answer,
)
from taskforge.research_retrieval import (  # noqa: E402
    CitationVerification,
    ResearchEvidence,
)

PROTOCOLS_BY_ROLE = {
    "retrieval_planner": "research.planner_handoff.v1",
    "source_evaluator": "research.evaluator_handoff.v1",
    "synthesis_writer": "research.writer_handoff.v1",
    "critical_reviewer": "research.critic_handoff.v1",
}
ROLE_ORDER = tuple(PROTOCOLS_BY_ROLE)
FROZEN_TOP_K = 8
WRITER_SELECTED_TOTAL_SNIPPET_CHARS = 7_200
WRITER_SELECTED_MAX_SNIPPET_CHARS = 2_600
WRITER_SELECTED_MIN_SNIPPET_CHARS = 800
WRITER_EMPTY_SELECTION_EVIDENCE_COUNT = 3
WRITER_FALLBACK_EVIDENCE_COUNT = 0
WRITER_FALLBACK_SNIPPET_CHARS = 400
EVALUATOR_VERSION = "v4"


def _request(
    client: TestClient,
    method: str,
    path: str,
    **kwargs: object,
) -> dict[str, Any]:
    response = client.request(method, path, **kwargs)
    if not response.is_success:
        raise RuntimeError(
            f"{method} {path} failed ({response.status_code}): {response.text[:2_000]}"
        )
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError(f"{method} {path} returned a non-object response")
    return value


def _load_jsonl(path: Path, *, require_success: bool = False) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if require_success and not value.get("orchestration_complete"):
            continue
        rows[str(value["case_id"])] = value
    return rows


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def _raw_questions(raw_dataset: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for paper_id, paper in raw_dataset.items():
        qas = paper.get("qas") if isinstance(paper, Mapping) else None
        if not isinstance(qas, list):
            continue
        for question in qas:
            if not isinstance(question, Mapping):
                continue
            question_id = str(question.get("question_id") or "").strip()
            if question_id:
                output[f"qasper:{paper_id}:{question_id}"] = question
    return output


def _paper_titles(raw_dataset: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(paper_id): str(paper.get("title") or f"QASPER paper {paper_id}").strip()
        for paper_id, paper in raw_dataset.items()
        if isinstance(paper, Mapping)
    }


def _frozen_cards(
    retrieval_row: Mapping[str, Any],
    *,
    scope_id: str,
    scope_version: int,
    paper_id: str,
    title: str,
) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for item in retrieval_row.get("retrieved_evidence", [])[:FROZEN_TOP_K]:
        if not isinstance(item, Mapping):
            continue
        text = str(item.get("text") or "").strip()[:3_000]
        evidence_id = str(item.get("evidence_id") or "").strip()
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not text or not evidence_id or not chunk_id:
            continue
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        text_start = max(0, int(item.get("text_start") or 0))
        text_end = item.get("text_end")
        if not isinstance(text_end, int) or text_end <= text_start:
            text_end = text_start + len(text)
        cards.append(
            EvidenceCard(
                evidence_id=evidence_id,
                scope_id=scope_id,
                scope_version=scope_version,
                paper_id=paper_id,
                chunk_id=chunk_id,
                source=f"paper://{paper_id}",
                title=title[:500],
                section=(
                    str(metadata.get("section") or metadata.get("heading"))[:500]
                    if metadata.get("section") or metadata.get("heading")
                    else None
                ),
                page=(str(item["page"]) if item.get("page") is not None else None),
                evidence_type=str(metadata.get("kind") or "paragraph")[:128],
                visual_artifact_ids=[
                    str(value)
                    for value in item.get("visual_artifact_ids", [])
                    if str(value).strip()
                ][:32],
                visual_pending=bool(item.get("visual_pending")),
                snippet=text,
                text_start=text_start,
                text_end=text_end,
                presentation_strategy=str(
                    item.get("presentation_strategy") or "full_child"
                )[:128],
                score=max(0.0, float(item.get("score") or 0.0)),
                retrieval_sources=[
                    str(value)
                    for value in item.get("retrieval_sources", [])
                    if str(value).strip()
                ][:16],
                verification_status="read",
            )
        )
    if not cards:
        raise ValueError(f"frozen report row has no usable evidence: {retrieval_row.get('case_id')}")
    return cards


class FrozenEvidenceReplay:
    """Scope-safe replay implementation for the production research tools."""

    def __init__(self, repository: Any) -> None:
        self.repository = repository
        self._cards_by_scope: dict[str, list[EvidenceCard]] = {}

    def register(self, scope_id: str, cards: Sequence[EvidenceCard]) -> None:
        self._cards_by_scope[scope_id] = [card.model_copy(deep=True) for card in cards]

    def _cards(self, access: LiteratureAccess, scope_id: str, version: int | None) -> list[EvidenceCard]:
        scope = self.repository.get_scope(access, scope_id, version=version)
        if scope.status != "ready":
            raise ValueError("research scope must be ready before frozen evidence replay")
        return self._cards_by_scope.get(scope.scope_id) or self.repository.list_evidence(
            access,
            scope.scope_id,
            version=scope.scope_version,
        )

    async def search(
        self,
        access: LiteratureAccess,
        request: EvidenceSearchRequest,
    ) -> ScopeEvidenceResult:
        scope = self.repository.get_scope(
            access,
            request.scope_id,
            version=request.scope_version,
        )
        cards = self._cards(access, scope.scope_id, scope.scope_version)[
            : min(request.top_k, FROZEN_TOP_K)
        ]
        confidence = ScopeBoundEvidenceService._confidence(
            request.query,
            cards,
            selected_papers=scope.selected_paper_ids,
            intent=request.intent,
        )
        return ScopeEvidenceResult(
            scope_id=scope.scope_id,
            scope_version=scope.scope_version,
            query=request.query,
            query_variants=[request.query],
            routed_intent=request.intent,
            retrieval_rounds=1,
            activated_operators=[],
            evidence=cards,
            confidence=confidence,
            retrieval_traces=[],
        )

    def read_evidence(
        self,
        access: LiteratureAccess,
        scope_id: str,
        evidence_id: str,
        *,
        scope_version: int | None = None,
    ) -> ResearchEvidence:
        cards = {
            card.evidence_id: card
            for card in self._cards(access, scope_id, scope_version)
        }
        card = cards.get(evidence_id)
        if card is None or not card.chunk_id:
            raise KeyError("evidence_id is outside the frozen research scope")
        return ResearchEvidence(
            evidence_id=card.evidence_id,
            chunk_id=card.chunk_id,
            title=card.title,
            source=card.source,
            section=card.section,
            page=card.page,
            version="frozen-a5-v4",
            text=card.snippet,
            evidence_type=card.evidence_type,
            visual_artifact_ids=tuple(card.visual_artifact_ids),
            visual_pending=card.visual_pending,
            text_start=card.text_start,
            text_end=card.text_end,
            presentation_strategy=card.presentation_strategy,
            score=card.score,
            retrieval_sources=tuple(card.retrieval_sources),
        )

    def verify_citation(
        self,
        access: LiteratureAccess,
        scope_id: str,
        claim: str,
        evidence_ids: Sequence[str],
        *,
        scope_version: int | None = None,
    ) -> CitationVerification:
        cards = {
            card.evidence_id: card
            for card in self._cards(access, scope_id, scope_version)
        }
        ids = tuple(dict.fromkeys(str(value) for value in evidence_ids))
        resolved = tuple(value for value in ids if value in cards)
        missing = tuple(value for value in ids if value not in cards)
        claim_tokens = set(tokenise(claim))
        support_tokens = set(
            tokenise(" ".join(cards[value].snippet for value in resolved))
        )
        coverage = (
            len(claim_tokens & support_tokens) / len(claim_tokens)
            if claim_tokens
            else 0.0
        )
        return CitationVerification(
            verified=not missing and coverage >= 0.55,
            claim=claim,
            evidence_ids=ids,
            resolved_evidence_ids=resolved,
            missing_evidence_ids=missing,
            token_coverage=coverage,
        )


def _scope_for_case(
    app: Any,
    replay: FrozenEvidenceReplay,
    *,
    case_id: str,
    query: str,
    paper_id: str,
    title: str,
    retrieval_row: Mapping[str, Any],
    tenant_id: str,
    user_id: str,
) -> ResearchScope:
    digest = hashlib.sha256(case_id.encode()).hexdigest()[:24]
    request_id = f"qasper-four-agent-request-{digest}"
    scope_id = f"qasper-four-agent-scope-{digest}"
    conversation_id = f"qasper-four-agent-{digest}"
    access = LiteratureAccess(tenant_id=tenant_id, user_id=user_id)
    repository = app.state.container.literature_repository
    request = LiteratureRequest(request_id=request_id, query=query)
    try:
        repository.save_request(access, request)
    except LiteratureConflictError:
        existing = repository.get_request(access, request_id)
        if existing.query != query:
            raise ValueError("resumed QASPER request disagrees with frozen query")
    repository.upsert_paper(
        access,
        PaperCard(
            paper_id=paper_id,
            canonical_title=title,
            source_urls=[f"https://arxiv.org/abs/{paper_id}"],
            arxiv_id=paper_id,
            verification_status="provider_verified",
            full_text_status="ingested",
        ),
    )
    scope = ResearchScope(
        scope_id=scope_id,
        tenant_id=tenant_id,
        owner_user_id=user_id,
        conversation_id=conversation_id,
        request_id=request_id,
        selected_paper_ids=[paper_id],
        selected_source_uris=[f"paper://{paper_id}"],
        user_intent=query,
        allowed_expansion=False,
        status="ready",
        confirmed_at=utc_now(),
    )
    try:
        scope = repository.create_scope(access, scope)
    except LiteratureConflictError:
        scope = repository.get_scope(access, scope_id)
        if scope.user_intent != query or scope.selected_paper_ids != [paper_id]:
            raise ValueError("resumed QASPER scope disagrees with frozen case")
    cards = _frozen_cards(
        retrieval_row,
        scope_id=scope.scope_id,
        scope_version=scope.scope_version,
        paper_id=paper_id,
        title=title,
    )
    repository.save_evidence(access, cards)
    replay.register(scope.scope_id, cards)
    return scope


def _run_agent_case(
    app: Any,
    client: TestClient,
    replay: FrozenEvidenceReplay,
    *,
    case_id: str,
    query: str,
    paper_id: str,
    title: str,
    retrieval_row: Mapping[str, Any],
    tenant_id: str,
    user_id: str,
    run_version: str,
) -> dict[str, Any]:
    scope = _scope_for_case(
        app,
        replay,
        case_id=case_id,
        query=query,
        paper_id=paper_id,
        title=title,
        retrieval_row=retrieval_row,
        tenant_id=tenant_id,
        user_id=user_id,
    )
    digest = hashlib.sha256(case_id.encode()).hexdigest()[:24]
    headers = {
        "X-TaskForge-Tenant": tenant_id,
        "X-TaskForge-User": user_id,
        "Idempotency-Key": f"qasper-four-agent-run-{run_version}-{digest}",
    }
    created = _request(
        client,
        "POST",
        f"/api/research/scopes/{scope.scope_id}/agent-run",
        headers=headers,
        json={
            "title": query[:500],
            "context": (
                "Frozen QASPER single-paper QA evaluation. The research question "
                "is the exact requested answer target; use only replayed A5 evidence."
            ),
            "survey_depth": "rigorous",
        },
    )
    runtime_case_id = str(created["case"]["case_id"])
    finished = _request(
        client,
        "POST",
        f"/api/review-cases/{runtime_case_id}/run-until-review",
        headers={
            "X-TaskForge-Tenant": tenant_id,
            "X-TaskForge-User": user_id,
        },
        json={"max_iterations": 12},
    )
    raw_roles = [
        item for item in finished.get("role_runs", []) if isinstance(item, Mapping)
    ]
    latest_by_role: dict[str, Mapping[str, Any]] = {}
    for role in raw_roles:
        role_id = str(role.get("role_id") or "")
        current = latest_by_role.get(role_id)
        if current is None or int(role.get("attempt") or 0) > int(
            current.get("attempt") or 0
        ):
            latest_by_role[role_id] = role
    role_rows: list[dict[str, Any]] = []
    protocols: dict[str, str | None] = {}
    total_usage: list[Mapping[str, int]] = []
    for role_id in ROLE_ORDER:
        role = latest_by_role.get(role_id, {})
        result = role.get("role_result")
        payload = result.get("research_payload") if isinstance(result, Mapping) else None
        metrics = role.get("runtime_metrics")
        usage = metrics.get("usage") if isinstance(metrics, Mapping) else None
        if isinstance(usage, Mapping):
            total_usage.append(usage)
        protocol = payload.get("protocol") if isinstance(payload, Mapping) else None
        protocols[role_id] = str(protocol) if protocol is not None else None
        role_rows.append(
            {
                "role_id": role_id,
                "status": role.get("status"),
                "attempt": role.get("attempt"),
                "error": role.get("error"),
                "runtime_metrics": metrics,
                "retrieved_evidence_refs": role.get("retrieved_evidence_refs", []),
                "role_result": result,
                "research_protocol": protocol,
            }
        )
    orchestration_complete = (
        finished.get("case", {}).get("status") == "waiting_human_review"
        and all(
            latest_by_role.get(role_id, {}).get("status") == "succeeded"
            and protocols.get(role_id) == expected
            for role_id, expected in PROTOCOLS_BY_ROLE.items()
        )
    )
    return {
        "case_id": case_id,
        "runtime_case_id": runtime_case_id,
        "paper_id": paper_id,
        "query": query,
        "scope_id": scope.scope_id,
        "scope_version": scope.scope_version,
        "frozen_top_k": FROZEN_TOP_K,
        "case_status": finished.get("case", {}).get("status"),
        "orchestration_complete": orchestration_complete,
        "protocols": protocols,
        "role_runs": role_rows,
        "agent_usage": _merge_usage(*total_usage),
    }


def _writer_and_final(agent_row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_role = {
        str(role.get("role_id")): role
        for role in agent_row.get("role_runs", [])
        if isinstance(role, Mapping)
    }
    writer_result = by_role.get("synthesis_writer", {}).get("role_result")
    critic_result = by_role.get("critical_reviewer", {}).get("role_result")
    writer_payload = (
        writer_result.get("research_payload")
        if isinstance(writer_result, Mapping)
        else None
    )
    critic_payload = (
        critic_result.get("research_payload")
        if isinstance(critic_result, Mapping)
        else None
    )
    writer = WriterHandoff.model_validate(writer_payload)
    writer_answer = "\n\n".join(
        claim.claim_text.strip()
        for claim in writer.claim_manifest
        if claim.claim_text.strip()
    )
    writer_citations = list(
        dict.fromkeys(
            evidence_id
            for claim in writer.claim_manifest
            for evidence_id in claim.evidence_ids
        )
    )
    writer_raw = {
        "answer": writer_answer,
        "direct_answer": writer.direct_answer.strip(),
        "citation_ids": writer_citations,
        "claim_ids": [claim.claim_id for claim in writer.claim_manifest],
    }

    def fallback_final(*, verdict: str, reason: str) -> dict[str, Any]:
        # A malformed or over-aggressive Critic must not erase the only valid
        # Writer answer.  ``more_evidence`` is the one safety gate where the
        # host deliberately returns Unanswerable instead of bypassing the
        # Critic's evidence objection.
        if verdict == "more_evidence":
            return {
                "answer": "Insufficient evidence in the selected papers.",
                "direct_answer": "Unanswerable",
                "evidence_ids": [],
                "included_claim_ids": [],
                "removed_claim_ids": [claim.claim_id for claim in writer.claim_manifest],
                "unresolved_claim_ids": [claim.claim_id for claim in writer.claim_manifest],
                "critic_verdict": verdict,
                "projection_fallback": True,
                "projection_fallback_reason": reason,
            }
        return {
            "answer": writer_answer or "Insufficient evidence in the selected papers.",
            "direct_answer": writer.direct_answer.strip() or "Unanswerable",
            "evidence_ids": writer_citations,
            "included_claim_ids": [claim.claim_id for claim in writer.claim_manifest],
            "removed_claim_ids": [],
            "unresolved_claim_ids": [],
            "critic_verdict": verdict,
            "projection_fallback": True,
            "projection_fallback_reason": reason,
        }

    try:
        critic = CriticHandoff.model_validate(critic_payload)
    except Exception as exc:
        critic = None
        final_raw = fallback_final(
            verdict="needs_revision",
            reason=f"critic_payload_invalid:{type(exc).__name__}",
        )
    else:
        try:
            projected = project_final_research_answer(writer, critic)
        except Exception as exc:
            final_raw = fallback_final(
                verdict=critic.verdict,
                reason=f"critic_projection_failed:{type(exc).__name__}",
            )
        else:
            final_raw = projected.model_dump(mode="json")
            final_raw["projection_fallback"] = False
    return (
        writer_raw,
        final_raw,
    )


def _gold_alignments(retrieval_row: Mapping[str, Any]) -> dict[str, GoldAlignment]:
    raw = retrieval_row.get("gold_alignments")
    if not isinstance(raw, Mapping):
        raise ValueError("strict retrieval row has no Gold-to-Child alignment")
    return {
        str(unit_id): GoldAlignment.model_validate(value)
        for unit_id, value in raw.items()
    }


def _candidate_child_recall(
    case: RAGEvalCase,
    retrieval_row: Mapping[str, Any],
    *,
    top_k: int,
) -> float:
    if case.qasper_gold is None:
        raise ValueError(f"QASPER case has no Gold labels: {case.case_id}")
    child_ids = {
        str(item.get("chunk_id"))
        for item in retrieval_row.get("retrieved_evidence", [])[:top_k]
        if isinstance(item, Mapping) and item.get("chunk_id")
    }
    alignments = _gold_alignments(retrieval_row)
    recalls: list[float] = []
    for evidence_set in case.qasper_gold.evidence_sets:
        hit_count = sum(
            alignment_coverage_for_children(alignments[unit.unit_id], child_ids) >= 0.80
            for unit in evidence_set.units
            if unit.unit_id in alignments
        )
        recalls.append(hit_count / len(evidence_set.units))
    return max(recalls, default=0.0)


def _presented_window_alignments(
    case: RAGEvalCase,
    presented: Sequence[Mapping[str, Any]],
    *,
    text_chars: int | None = None,
) -> dict[str, GoldAlignment]:
    if case.qasper_gold is None:
        raise ValueError(f"QASPER case has no Gold labels: {case.case_id}")
    chunks: list[AlignmentChunk] = []
    for index, item in enumerate(presented):
        evidence_id = str(item.get("evidence_id") or "").strip()
        text = str(item.get("text") or item.get("snippet") or "").strip()
        if text_chars is not None:
            text = text[:text_chars]
        if evidence_id and text:
            chunks.append(
                AlignmentChunk(child_id=evidence_id, text=text, order=index)
            )
    units = {
        unit.unit_id: unit
        for evidence_set in case.qasper_gold.evidence_sets
        for unit in evidence_set.units
    }
    return {
        unit_id: align_gold_unit(unit, chunks)
        for unit_id, unit in units.items()
    }


def _recall_from_window_alignments(
    case: RAGEvalCase,
    alignments: Mapping[str, GoldAlignment],
) -> float:
    if case.qasper_gold is None:
        raise ValueError(f"QASPER case has no Gold labels: {case.case_id}")
    recalls = [
        sum(
            (
                unit.unit_id in alignments
                and alignments[unit.unit_id].status in {"exact", "fuzzy"}
                and alignments[unit.unit_id].normalized_coverage >= 0.80
            )
            for unit in evidence_set.units
        )
        / len(evidence_set.units)
        for evidence_set in case.qasper_gold.evidence_sets
    ]
    return max(recalls, default=0.0)


def _presented_window_recall(
    case: RAGEvalCase,
    retrieval_row: Mapping[str, Any],
    *,
    top_k: int,
    text_chars: int | None = None,
) -> float:
    presented = [
        item
        for item in retrieval_row.get("retrieved_evidence", [])[:top_k]
        if isinstance(item, Mapping)
    ]
    return _recall_from_window_alignments(
        case,
        _presented_window_alignments(case, presented, text_chars=text_chars),
    )


def _writer_selected_evidence(
    agent_row: Mapping[str, Any],
    presented: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reproduce the Host-verified Evaluator-to-Writer evidence projection."""

    requested_ids: list[str] = []
    role_runs = agent_row.get("role_runs", [])
    if isinstance(role_runs, list):
        for role in role_runs:
            if not isinstance(role, Mapping) or role.get("role_id") != "source_evaluator":
                continue
            result = role.get("role_result")
            payload = result.get("research_payload") if isinstance(result, Mapping) else None
            ledger = payload.get("ledger") if isinstance(payload, Mapping) else None
            raw_ids = ledger.get("evidence_ids", []) if isinstance(ledger, Mapping) else []
            if isinstance(raw_ids, list):
                requested_ids = [str(value) for value in raw_ids]
            break
    by_id = {
        str(item.get("evidence_id")): item
        for item in presented
        if isinstance(item.get("evidence_id"), str)
        and str(item.get("evidence_id")).strip()
    }
    selected_ids = [
        value
        for value in dict.fromkeys(requested_ids)
        if value in by_id
    ]
    ranked_ids = [
        str(item.get("evidence_id"))
        for item in presented
        if isinstance(item.get("evidence_id"), str)
        and str(item.get("evidence_id")).strip()
    ]
    if not selected_ids:
        selected_ids = ranked_ids[:WRITER_EMPTY_SELECTION_EVIDENCE_COUNT]
    if not selected_ids:
        return []
    snippet_budget = min(
        WRITER_SELECTED_MAX_SNIPPET_CHARS,
        max(
            WRITER_SELECTED_MIN_SNIPPET_CHARS,
            WRITER_SELECTED_TOTAL_SNIPPET_CHARS // len(selected_ids),
        ),
    )
    projected = [
        {
            **dict(by_id[evidence_id]),
            "text": str(
                by_id[evidence_id].get("text")
                or by_id[evidence_id].get("snippet")
                or ""
            )[:snippet_budget],
        }
        for evidence_id in selected_ids
    ]
    selected_set = set(selected_ids)
    fallback_ids = [
        evidence_id
        for evidence_id in ranked_ids[:WRITER_FALLBACK_EVIDENCE_COUNT]
        if evidence_id not in selected_set
    ]
    projected.extend(
        {
            **dict(by_id[evidence_id]),
            "text": str(
                by_id[evidence_id].get("text")
                or by_id[evidence_id].get("snippet")
                or ""
            )[:WRITER_FALLBACK_SNIPPET_CHARS],
        }
        for evidence_id in fallback_ids
    )
    return projected


def _writer_selected_context_recall(
    case: RAGEvalCase,
    agent_row: Mapping[str, Any],
    presented: Sequence[Mapping[str, Any]],
) -> float:
    selected = _writer_selected_evidence(agent_row, presented)
    return _recall_from_window_alignments(
        case,
        _presented_window_alignments(case, selected),
    )


def _window_citation_metrics(
    citation_ids: Sequence[str],
    presented_evidence: Sequence[Mapping[str, Any]],
    case: RAGEvalCase,
) -> dict[str, Any]:
    """Score citations against text the frozen replay actually exposed."""

    if case.qasper_gold is None:
        raise ValueError(f"QASPER case has no Gold labels: {case.case_id}")
    presented = {
        str(item.get("evidence_id")): item
        for item in presented_evidence
        if isinstance(item.get("evidence_id"), str)
        and str(item.get("evidence_id")).strip()
    }
    unique_citations = list(dict.fromkeys(str(value) for value in citation_ids))
    valid_ids = [value for value in unique_citations if value in presented]
    cited = [presented[value] for value in valid_ids]
    alignments = _presented_window_alignments(case, cited)
    scored_sets: list[tuple[float, int, str, set[str]]] = []
    for evidence_set in case.qasper_gold.evidence_sets:
        hit_units = {
            unit.unit_id
            for unit in evidence_set.units
            if unit.unit_id in alignments
            and alignments[unit.unit_id].status in {"exact", "fuzzy"}
            and alignments[unit.unit_id].normalized_coverage >= 0.80
        }
        scored_sets.append(
            (
                len(hit_units) / len(evidence_set.units),
                len(evidence_set.units),
                evidence_set.annotation_id,
                hit_units,
            )
        )
    coverage, _, annotation_id, covered_unit_ids = max(
        scored_sets,
        key=lambda item: (item[0], -item[1], item[2]),
    )
    supporting_ids = {
        span.child_id
        for unit_id in covered_unit_ids
        for span in alignments[unit_id].aligned_child_spans
    }
    total = len(unique_citations)
    return {
        "citation_count": total,
        "valid_citation_count": len(valid_ids),
        "gold_supported_citation_count": sum(
            value in supporting_ids for value in valid_ids
        ),
        "invalid_citation_ids": [
            value for value in unique_citations if value not in presented
        ],
        "citation_validity": len(valid_ids) / total if total else 0.0,
        "gold_content_citation_precision": (
            sum(value in supporting_ids for value in valid_ids) / total
            if total
            else 0.0
        ),
        "gold_evidence_unit_coverage": coverage,
        "covered_gold_unit_ids": sorted(covered_unit_ids),
        "selected_gold_annotation_id": annotation_id,
        "support_basis": "frozen_presented_text_window",
    }


def _deterministic_candidate_metrics(
    *,
    answer: str,
    direct_answer: str = "",
    citations: Sequence[str],
    references: Sequence[str],
    presented: Sequence[Mapping[str, Any]],
    case: RAGEvalCase,
    alignments: Mapping[str, GoldAlignment],
) -> dict[str, Any]:
    if case.qasper_gold is None:
        raise ValueError(f"QASPER case has no Gold labels: {case.case_id}")
    scoring_answer = direct_answer.strip() or answer.strip()
    exact = max(
        (answer_exact_match(scoring_answer, reference) for reference in references),
        default=0.0,
    )
    token_f1 = max(
        (answer_token_f1(scoring_answer, reference) for reference in references),
        default=0.0,
    )
    child_citation_metrics = citation_metrics(
        citations,
        presented,
        case.qasper_gold,
        alignments,
    )
    window_citation_metrics = _window_citation_metrics(
        citations,
        presented,
        case,
    )
    strict_exact_and_gold_citation = float(
        exact == 1.0
        and window_citation_metrics["citation_validity"] == 1.0
        and window_citation_metrics["gold_content_citation_precision"] == 1.0
        and window_citation_metrics["gold_evidence_unit_coverage"] > 0.0
    )
    return {
        "answer": answer,
        "direct_answer": direct_answer.strip(),
        "scoring_answer": scoring_answer,
        "citation_ids": list(citations),
        "exact_match": exact,
        "token_f1": token_f1,
        "strict_exact_and_gold_citation": strict_exact_and_gold_citation,
        "citation_metrics": window_citation_metrics,
        "candidate_child_citation_metrics": child_citation_metrics,
    }


async def _judge_candidate(
    provider: OpenAIChatCompletionsProvider,
    *,
    model: str,
    case: RAGEvalCase,
    references: Sequence[str],
    raw_question: Mapping[str, Any],
    candidate: Mapping[str, Any],
    presented: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], Mapping[str, int] | None, str | None]:
    answer = str(candidate.get("answer") or "").strip()
    citations = [str(value) for value in candidate.get("citation_ids", [])]
    if not answer:
        judgement = SemanticJudgement(
            answer_verdict="incorrect",
            citation_verdict="no_citation",
            critical_error=True,
            rationale="The four-Agent chain produced no projectable answer.",
        )
        return judgement.model_dump(mode="json"), None, None
    by_id = {
        str(item.get("evidence_id")): item
        for item in presented
        if isinstance(item.get("evidence_id"), str)
    }
    cited_text = [
        str(by_id[value].get("text") or "") for value in citations if value in by_id
    ]
    judgement, usage, error = await _judge_answer(
        provider,
        model=model,
        case=case,
        reference_answers=references,
        candidate_answer=answer,
        gold_evidence=qasper_gold_evidence_texts(raw_question),
        cited_evidence=cited_text,
    )
    return (
        judgement.model_dump(mode="json") if judgement is not None else {},
        usage,
        error,
    )


async def _score_case(
    *,
    case_id: str,
    agent_row: Mapping[str, Any],
    retrieval_row: Mapping[str, Any],
    case: RAGEvalCase,
    raw_question: Mapping[str, Any],
    provider: OpenAIChatCompletionsProvider | None,
    model: str,
    judge_writer: bool,
) -> dict[str, Any]:
    references = qasper_answer_references(raw_question)
    if not references:
        references = [str(qasper_reference_answer(case.answer))]
    presented = [
        item
        for item in retrieval_row.get("retrieved_evidence", [])[:FROZEN_TOP_K]
        if isinstance(item, Mapping)
    ]
    alignments = _gold_alignments(retrieval_row)
    projection_error: str | None = None
    try:
        writer_raw, final_raw = _writer_and_final(agent_row)
    except Exception as exc:
        projection_error = f"{type(exc).__name__}: {str(exc)[:1_000]}"
        writer_raw = {"answer": "", "citation_ids": [], "claim_ids": []}
        final_raw = {
            "answer": "",
            "direct_answer": "",
            "evidence_ids": [],
            "included_claim_ids": [],
            "removed_claim_ids": [],
            "unresolved_claim_ids": [],
            "critic_verdict": None,
        }
    writer = _deterministic_candidate_metrics(
        answer=str(writer_raw["answer"]),
        direct_answer=str(writer_raw.get("direct_answer") or ""),
        citations=writer_raw["citation_ids"],
        references=references,
        presented=presented,
        case=case,
        alignments=alignments,
    )
    final = _deterministic_candidate_metrics(
        answer=str(final_raw["answer"]),
        direct_answer=str(final_raw.get("direct_answer") or ""),
        citations=final_raw.get("evidence_ids", []),
        references=references,
        presented=presented,
        case=case,
        alignments=alignments,
    )
    judge_usage: list[Mapping[str, int] | None] = []
    judge_errors: dict[str, str | None] = {"writer": None, "final": None}
    if provider is not None:
        if judge_writer:
            judgement, usage, error = await _judge_candidate(
                provider,
                model=model,
                case=case,
                references=references,
                raw_question=raw_question,
                candidate=writer,
                presented=presented,
            )
            writer["semantic_judgement"] = judgement
            judge_usage.append(usage)
            judge_errors["writer"] = error
        judgement, usage, error = await _judge_candidate(
            provider,
            model=model,
            case=case,
            references=references,
            raw_question=raw_question,
            candidate=final,
            presented=presented,
        )
        final["semantic_judgement"] = judgement
        judge_usage.append(usage)
        judge_errors["final"] = error
    candidate_child_recall = _candidate_child_recall(
        case,
        retrieval_row,
        top_k=FROZEN_TOP_K,
    )
    frozen_window_recall = _presented_window_recall(
        case,
        retrieval_row,
        top_k=FROZEN_TOP_K,
    )
    writer_selected_context_recall = _writer_selected_context_recall(
        case,
        agent_row,
        presented,
    )
    final_judgement = final.get("semantic_judgement")
    final_correct = bool(
        isinstance(final_judgement, Mapping)
        and final_judgement.get("answer_verdict") == "correct"
        and not final_judgement.get("critical_error")
    )
    final_grounded = bool(
        final_correct
        and isinstance(final_judgement, Mapping)
        and final_judgement.get("citation_verdict") == "fully_supported"
    )
    if not agent_row.get("orchestration_complete") or projection_error:
        failure_stage = "orchestration_or_projection_failure"
    elif final_correct and final_grounded:
        failure_stage = "success"
    elif final_correct:
        failure_stage = "citation_failure"
    elif frozen_window_recall == 0.0 and candidate_child_recall > 0.0:
        failure_stage = "presentation_window_failure"
    elif candidate_child_recall == 0.0:
        failure_stage = "retrieval_failure"
    else:
        failure_stage = "answer_reasoning_failure"
    return {
        "case_id": case_id,
        "paper_id": str(case.metadata["paper_id"]),
        "query": case.query,
        "gold_answers": references,
        f"candidate_child_recall_at_{FROZEN_TOP_K}": candidate_child_recall,
        f"frozen_window_recall_at_{FROZEN_TOP_K}": frozen_window_recall,
        f"writer_selected_context_recall_at_{FROZEN_TOP_K}": writer_selected_context_recall,
        # Backward-compatible alias. It is deliberately the actual replay
        # window recall, not the full-Child diagnostic.
        "recall_at_k": frozen_window_recall,
        "retrieval_recall_at_10": float(retrieval_row["recall_at_k"]["10"]),
        "presented_evidence_ids": [str(item.get("evidence_id")) for item in presented],
        "orchestration_complete": bool(agent_row.get("orchestration_complete")),
        "projection_error": projection_error,
        "critic": {
            "verdict": final_raw.get("critic_verdict"),
            "projection_fallback": bool(final_raw.get("projection_fallback")),
            "projection_fallback_reason": final_raw.get("projection_fallback_reason"),
            "included_claim_ids": final_raw.get("included_claim_ids", []),
            "removed_claim_ids": final_raw.get("removed_claim_ids", []),
            "unresolved_claim_ids": final_raw.get("unresolved_claim_ids", []),
            "changed_answer": writer["answer"] != final["answer"],
        },
        "writer": writer,
        "final": final,
        "failure_stage": failure_stage,
        "judge_errors": judge_errors,
        "usage": {
            "agents": agent_row.get("agent_usage", {}),
            "judge": _merge_usage(*judge_usage),
        },
    }


def _with_visibility_audit(
    row: Mapping[str, Any],
    *,
    agent_row: Mapping[str, Any],
    retrieval_row: Mapping[str, Any],
    case: RAGEvalCase,
) -> dict[str, Any]:
    """Upgrade a scored checkpoint without repeating semantic model calls."""

    upgraded = deepcopy(dict(row))
    presented = [
        item
        for item in retrieval_row.get("retrieved_evidence", [])[:FROZEN_TOP_K]
        if isinstance(item, Mapping)
    ]
    alignments = _gold_alignments(retrieval_row)
    references = [str(value) for value in upgraded.get("gold_answers", [])]
    for name in ("writer", "final"):
        prior = upgraded.get(name)
        if not isinstance(prior, Mapping):
            continue
        rescored = _deterministic_candidate_metrics(
            answer=str(prior.get("answer") or ""),
            direct_answer=str(prior.get("direct_answer") or ""),
            citations=[str(value) for value in prior.get("citation_ids", [])],
            references=references,
            presented=presented,
            case=case,
            alignments=alignments,
        )
        if isinstance(prior.get("semantic_judgement"), Mapping):
            rescored["semantic_judgement"] = deepcopy(prior["semantic_judgement"])
        upgraded[name] = rescored

    candidate_recall = _candidate_child_recall(
        case,
        retrieval_row,
        top_k=FROZEN_TOP_K,
    )
    window_recall = _presented_window_recall(
        case,
        retrieval_row,
        top_k=FROZEN_TOP_K,
    )
    writer_recall = _writer_selected_context_recall(
        case,
        agent_row,
        presented,
    )
    upgraded[f"candidate_child_recall_at_{FROZEN_TOP_K}"] = candidate_recall
    upgraded[f"frozen_window_recall_at_{FROZEN_TOP_K}"] = window_recall
    upgraded[f"writer_selected_context_recall_at_{FROZEN_TOP_K}"] = writer_recall
    upgraded.pop("writer_initial_context_recall_at_8", None)
    upgraded.pop("recall_at_8", None)
    upgraded["recall_at_k"] = window_recall

    final_judgement = upgraded.get("final", {}).get("semantic_judgement")
    final_correct = bool(
        isinstance(final_judgement, Mapping)
        and final_judgement.get("answer_verdict") == "correct"
        and not final_judgement.get("critical_error")
    )
    final_grounded = bool(
        final_correct
        and isinstance(final_judgement, Mapping)
        and final_judgement.get("citation_verdict") == "fully_supported"
    )
    if not upgraded.get("orchestration_complete") or upgraded.get("projection_error"):
        failure_stage = "orchestration_or_projection_failure"
    elif final_correct and final_grounded:
        failure_stage = "success"
    elif final_correct:
        failure_stage = "citation_failure"
    elif window_recall == 0.0 and candidate_recall > 0.0:
        failure_stage = "presentation_window_failure"
    elif candidate_recall == 0.0:
        failure_stage = "retrieval_failure"
    else:
        failure_stage = "answer_reasoning_failure"
    upgraded["failure_stage"] = failure_stage
    return upgraded


def _candidate_aggregate(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    candidates = [row[name] for row in rows if isinstance(row.get(name), Mapping)]
    judged = [
        item
        for item in candidates
        if isinstance(item.get("semantic_judgement"), Mapping)
        and item["semantic_judgement"].get("answer_verdict")
        in {"correct", "partially_correct", "incorrect"}
    ]
    correct = [
        item
        for item in judged
        if item["semantic_judgement"].get("answer_verdict") == "correct"
        and not item["semantic_judgement"].get("critical_error")
    ]
    strict = [
        item
        for item in correct
        if item["semantic_judgement"].get("citation_verdict") == "fully_supported"
    ]
    weighted = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}

    def strict_exact_and_gold_citation(item: Mapping[str, Any]) -> float:
        stored = item.get("strict_exact_and_gold_citation")
        if stored is not None:
            return float(stored)
        citation = item.get("citation_metrics")
        if not isinstance(citation, Mapping):
            return 0.0
        return float(
            float(item.get("exact_match") or 0.0) == 1.0
            and float(citation.get("citation_validity") or 0.0) == 1.0
            and float(citation.get("gold_content_citation_precision") or 0.0)
            == 1.0
            and float(citation.get("gold_evidence_unit_coverage") or 0.0) > 0.0
        )

    return {
        "exact_match_accuracy": statistics.fmean(
            float(item.get("exact_match") or 0.0) for item in candidates
        ),
        "avg_token_f1": statistics.fmean(
            float(item.get("token_f1") or 0.0) for item in candidates
        ),
        "strict_exact_and_gold_citation_accuracy": statistics.fmean(
            strict_exact_and_gold_citation(item) for item in candidates
        ),
        "avg_citation_validity": statistics.fmean(
            float(item["citation_metrics"]["citation_validity"]) for item in candidates
        ),
        "avg_gold_content_citation_precision": statistics.fmean(
            float(item["citation_metrics"]["gold_content_citation_precision"])
            for item in candidates
        ),
        "avg_gold_evidence_unit_coverage": statistics.fmean(
            float(item["citation_metrics"]["gold_evidence_unit_coverage"])
            for item in candidates
        ),
        "semantic_judged_cases": len(judged),
        "semantic_answer_accuracy": len(correct) / len(judged) if judged else None,
        "semantic_strict_grounded_accuracy": len(strict) / len(judged) if judged else None,
        "semantic_weighted_accuracy": (
            statistics.fmean(
                weighted[item["semantic_judgement"]["answer_verdict"]]
                for item in judged
            )
            if judged
            else None
        ),
    }


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty four-Agent evaluation")
    writer_correct: list[bool] = []
    final_correct: list[bool] = []
    for row in rows:
        writer_judgement = row.get("writer", {}).get("semantic_judgement")
        final_judgement = row.get("final", {}).get("semantic_judgement")
        if not all(
            isinstance(value, Mapping)
            and value.get("answer_verdict")
            in {"correct", "partially_correct", "incorrect"}
            for value in (writer_judgement, final_judgement)
        ):
            continue
        writer_correct.append(
            writer_judgement.get("answer_verdict") == "correct"
            and not writer_judgement.get("critical_error")
        )
        final_correct.append(
            final_judgement.get("answer_verdict") == "correct"
            and not final_judgement.get("critical_error")
        )
    paired = len(writer_correct)
    usage = {
        "agents": _merge_usage(
            *(row.get("usage", {}).get("agents") for row in rows)
        ),
        "judge": _merge_usage(
            *(row.get("usage", {}).get("judge") for row in rows)
        ),
    }
    return {
        "total_cases": len(rows),
        "orchestration_completion_rate": sum(
            bool(row.get("orchestration_complete")) for row in rows
        )
        / len(rows),
        "four_protocol_completion_rate": sum(
            bool(row.get("orchestration_complete")) for row in rows
        )
        / len(rows),
        f"candidate_child_recall_at_{FROZEN_TOP_K}": statistics.fmean(
            float(row.get(f"candidate_child_recall_at_{FROZEN_TOP_K}") or 0.0)
            for row in rows
        ),
        f"frozen_window_recall_at_{FROZEN_TOP_K}": statistics.fmean(
            float(row.get(f"frozen_window_recall_at_{FROZEN_TOP_K}") or 0.0)
            for row in rows
        ),
        f"writer_selected_context_recall_at_{FROZEN_TOP_K}": statistics.fmean(
            float(row.get(f"writer_selected_context_recall_at_{FROZEN_TOP_K}") or 0.0)
            for row in rows
        ),
        "writer": _candidate_aggregate(rows, "writer"),
        "critic_projected_final": _candidate_aggregate(rows, "final"),
        "critic_effect": {
            "paired_semantic_cases": paired,
            "improved": sum(
                not writer_correct[index] and final_correct[index]
                for index in range(paired)
            ),
            "regressed": sum(
                writer_correct[index] and not final_correct[index]
                for index in range(paired)
            ),
            "answer_text_changed": sum(
                bool(row.get("critic", {}).get("changed_answer")) for row in rows
            ),
        },
        "failure_stage_counts": dict(Counter(str(row["failure_stage"]) for row in rows)),
        "failure_counts": {
            "projection_error": sum(bool(row.get("projection_error")) for row in rows),
            "critic_projection_fallback": sum(
                bool(row.get("critic", {}).get("projection_fallback"))
                for row in rows
            ),
            "judge_error": sum(
                bool(error)
                for row in rows
                for error in row.get("judge_errors", {}).values()
            ),
        },
        "usage": usage,
    }


def _settings(configured: Settings, state_root: Path) -> Settings:
    state_root.mkdir(parents=True, exist_ok=True)
    state = state_root / "state"
    return configured.model_copy(
        update={
            "sqlite_path": state / "taskforge.sqlite3",
            "context_sqlite_path": state / "context.sqlite3",
            "operations_sqlite_path": state / "operations.sqlite3",
            "orchestration_sqlite_path": state / "orchestration.sqlite3",
            "review_case_sqlite_path": state / "review.sqlite3",
            "verification_sqlite_path": state / "verification.sqlite3",
            "literature_sqlite_path": state / "literature.sqlite3",
            "literature_cache_path": state / "literature-cache.sqlite3",
            "workspace_root": PROJECT_ROOT,
            "artifact_root": state_root / "artifacts",
            "context_backend": "memory",
            "retrieval_routing": "lexical",
            "research_reranker_model": None,
            "research_rewrite_enabled": False,
            "research_query_expansion_mode": "original",
            "mineru_base_url": None,
            "mineru_expected_version": None,
            "visual_extractor_base_url": None,
            "visual_extractor_api_key": None,
            "visual_extractor_model": None,
            "mcp_config_path": None,
            "deepseek_timeout_seconds": 120.0,
        }
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    retrieval_path = args.retrieval_report.resolve()
    dataset_path = args.dataset.resolve()
    split_path = args.split.resolve()
    output_path = args.output.resolve()
    state_root = args.state_root.resolve()
    retrieval_report = json.loads(retrieval_path.read_text(encoding="utf-8"))
    _validate_retrieval_report(retrieval_report, evidence_source="retrieved")
    frozen_report_text_chars = (
        3_000
        if retrieval_report.get("schema_version") in {"2.2", "2.3"}
        else 1_000
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    case_ids = [str(value) for value in split.get("case_ids", [])]
    if args.max_cases is not None:
        case_ids = case_ids[: args.max_cases]
    report_rows = {
        str(row["case_id"]): row
        for row in retrieval_report.get("rows", [])
        if isinstance(row, Mapping) and row.get("case_id")
    }
    missing = [case_id for case_id in case_ids if case_id not in report_rows]
    if missing:
        raise ValueError(f"retrieval report is missing locked cases: {missing[:3]}")
    dataset = load_qasper_dataset(dataset_path)
    cases = {case.case_id: case for case in dataset.cases}
    raw_dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    raw_questions = _raw_questions(raw_dataset)
    titles = _paper_titles(raw_dataset)
    missing = [case_id for case_id in case_ids if case_id not in cases or case_id not in raw_questions]
    if missing:
        raise ValueError(f"QASPER dataset is missing locked cases: {missing[:3]}")

    agent_checkpoint = output_path.with_suffix(".agent-runs.jsonl")
    prediction_checkpoint = output_path.with_suffix(".predictions.jsonl")
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if (agent_checkpoint.exists() or prediction_checkpoint.exists()) and not args.resume:
        raise FileExistsError("checkpoints exist; pass --resume or choose another --output")
    prior_agent_rows = _load_jsonl(agent_checkpoint)
    completed_agents = _load_jsonl(agent_checkpoint, require_success=True)
    prior_predictions = _load_jsonl(prediction_checkpoint)
    for case_id, prediction in prior_predictions.items():
        if prediction.get("projection_error"):
            completed_agents.pop(case_id, None)
    attempt_counts: Counter[str] = Counter()
    if agent_checkpoint.exists():
        for line in agent_checkpoint.read_text(encoding="utf-8").splitlines():
            if line.strip():
                attempt_counts[str(json.loads(line)["case_id"])] += 1
    configured = Settings(_env_file=PROJECT_ROOT / ".env")
    if configured.provider != "deepseek":
        raise ValueError("the four-Agent QASPER evaluator currently requires provider=deepseek")
    if configured.deepseek_api_key is None:
        raise ValueError("TASKFORGE_DEEPSEEK_API_KEY is required")
    model = args.model or configured.deepseek_model or "deepseek-v4-flash"
    app = create_app(
        _settings(configured, state_root).model_copy(update={"deepseek_model": model})
    )
    replay = FrozenEvidenceReplay(app.state.container.literature_repository)
    # create_tool_registry captured this service object. Replacing its methods
    # retains all production scope checks and only freezes the retrieval source.
    live_service = app.state.container.scope_evidence
    live_service.search = replay.search  # type: ignore[method-assign]
    live_service.read_evidence = replay.read_evidence  # type: ignore[method-assign]
    live_service.verify_citation = replay.verify_citation  # type: ignore[method-assign]
    tenant_id = "qasper-four-agent-eval"
    user_id = "qasper-evaluator"
    with TestClient(app) as client:
        for index, case_id in enumerate(case_ids, start=1):
            if case_id in completed_agents:
                continue
            case = cases[case_id]
            paper_id = str(case.metadata["paper_id"])
            previous = prior_agent_rows.get(case_id)
            run_version = (
                f"{EVALUATOR_VERSION}-retry-{attempt_counts[case_id] + 1}"
                if previous is not None and not previous.get("orchestration_complete")
                else EVALUATOR_VERSION
            )
            try:
                row = _run_agent_case(
                    app,
                    client,
                    replay,
                    case_id=case_id,
                    query=case.query,
                    paper_id=paper_id,
                    title=titles.get(paper_id, f"QASPER paper {paper_id}"),
                    retrieval_row=report_rows[case_id],
                    tenant_id=tenant_id,
                    user_id=user_id,
                    run_version=run_version,
                )
            except Exception as exc:
                row = {
                    "case_id": case_id,
                    "paper_id": paper_id,
                    "query": case.query,
                    "orchestration_complete": False,
                    "execution_error": f"{type(exc).__name__}: {str(exc)[:2_000]}",
                    "role_runs": [],
                    "agent_usage": {},
                }
            _append_jsonl(agent_checkpoint, row)
            prior_agent_rows[case_id] = row
            attempt_counts[case_id] += 1
            if row.get("orchestration_complete"):
                completed_agents[case_id] = row
            print(
                json.dumps(
                    {
                        "phase": "four_agents",
                        "case": index,
                        "total": len(case_ids),
                        "case_id": case_id,
                        "complete": bool(row.get("orchestration_complete")),
                        "usage": row.get("agent_usage", {}),
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

    # Include failed cases from their latest checkpoint too; a resume retries
    # them because only successful Agent rows enter completed_agents.
    all_agent_rows = _load_jsonl(agent_checkpoint)
    completed_predictions = {
        case_id: row
        for case_id, row in _load_jsonl(prediction_checkpoint).items()
        if row.get("orchestration_complete")
        and not row.get("projection_error")
        and not any(row.get("judge_errors", {}).values())
    }

    async def score_pending() -> None:
        judge: OpenAIChatCompletionsProvider | None = None
        if not args.no_semantic_judge:
            judge = OpenAIChatCompletionsProvider(
                api_key=configured.deepseek_api_key.get_secret_value(),
                enabled=True,
                model=model,
                base_url=configured.deepseek_base_url,
                timeout_seconds=120,
                thinking_mode="disabled",
                json_mode=True,
            )
        try:
            for index, case_id in enumerate(case_ids, start=1):
                if case_id in completed_predictions:
                    continue
                agent_row = all_agent_rows[case_id]
                row = await _score_case(
                    case_id=case_id,
                    agent_row=agent_row,
                    retrieval_row=report_rows[case_id],
                    case=cases[case_id],
                    raw_question=raw_questions[case_id],
                    provider=judge,
                    model=model,
                    judge_writer=args.judge_writer,
                )
                _append_jsonl(prediction_checkpoint, row)
                completed_predictions[case_id] = row
                print(
                    json.dumps(
                        {
                            "phase": "scoring",
                            "case": index,
                            "total": len(case_ids),
                            "case_id": case_id,
                            "failure_stage": row["failure_stage"],
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        finally:
            if judge is not None:
                await judge.aclose()

    asyncio.run(score_pending())
    rows = [
        _with_visibility_audit(
            completed_predictions[case_id],
            agent_row=all_agent_rows[case_id],
            retrieval_row=report_rows[case_id],
            case=cases[case_id],
        )
        for case_id in case_ids
    ]
    report = {
        "schema_version": "1.0",
        "evaluation_type": "qasper_frozen_retrieval_four_agent_answer_e2e_live",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": "QASPER v0.3 official dev clean locked split",
        "model": model,
        "live_model_calls": True,
        "semantic_judge": {
            "enabled": not args.no_semantic_judge,
            "writer_judged": bool(args.judge_writer and not args.no_semantic_judge),
            "model": model if not args.no_semantic_judge else None,
            "independent_from_agents": False if not args.no_semantic_judge else None,
        },
        "inputs": {
            "retrieval_report": str(retrieval_path),
            "retrieval_report_sha256": _sha256(retrieval_path),
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "split": str(split_path),
            "split_sha256": _sha256(split_path),
            "frozen_evidence_top_k": FROZEN_TOP_K,
            "frozen_report_text_chars": frozen_report_text_chars,
            "writer_selected_total_snippet_chars": (
                WRITER_SELECTED_TOTAL_SNIPPET_CHARS
            ),
        },
        "pipeline": [
            "persisted_real_pdf_a5_retrieval",
            f"frozen_top_{FROZEN_TOP_K}_scope_bound_paper_search_replay",
            "retrieval_planner",
            "source_evaluator",
            "synthesis_writer",
            "critical_reviewer",
            "deterministic_critic_patch_projection",
            *( ["live_semantic_judge"] if not args.no_semantic_judge else [] ),
            "deterministic_multireference_and_gold_citation_scoring",
        ],
        "retrieval_metrics": retrieval_report.get("metrics"),
        "metrics": _metrics(rows),
        "rows": rows,
        "checkpoints": {
            "agent_runs": str(agent_checkpoint),
            "predictions": str(prediction_checkpoint),
        },
        "limitations": [
            "The retrieval stage is replayed from the hashed A5 real-PDF report rather than rerun inside each Agent case.",
            (
                f"The schema 2.2/2.3 retrieval report preserves paper_search text windows up to 3,000 characters; frozen_window_recall_at_{FROZEN_TOP_K} measures that exact replay text, while candidate_child_recall_at_{FROZEN_TOP_K} measures complete retrieved Child chunks."
                if frozen_report_text_chars == 3_000
                else f"The legacy retrieval report serializes only the first 1,000 characters of each returned evidence window; frozen_window_recall_at_{FROZEN_TOP_K} therefore measures a conservative replay rather than the full live window."
            ),
            f"writer_selected_context_recall_at_{FROZEN_TOP_K} reproduces the Host-verified Evaluator selection and preserves its 7,200-character evidence budget; ranked fallback is used only when the Evaluator returns no valid selection.",
            f"The production Evaluator contract exposes at most eight evidence cards, so this is an A5 Top-{FROZEN_TOP_K} four-Agent answer evaluation.",
            "QASPER answers are scored against all distinct answerable annotations for each locked question.",
            "The semantic judge uses the same configured model family as the four research Agents and requires human calibration.",
            "Exact match and token F1 undercount valid free-form paraphrases.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--retrieval-report",
        type=Path,
        default=PROJECT_ROOT
        / ".taskforge"
        / "eval-runs"
        / "qasper-pdf-final-a5-v4"
        / "a5.json",
    )
    value.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    value.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "eval" / "splits" / "qasper-dev-clean-holdout-100-v2.json",
    )
    value.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "qasper-four-agent-e2e-live.json",
    )
    value.add_argument(
        "--state-root",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-runs" / "qasper-four-agent-state",
    )
    value.add_argument("--model", default=None)
    value.add_argument("--max-cases", type=int, default=None)
    value.add_argument("--judge-writer", action="store_true")
    value.add_argument("--no-semantic-judge", action="store_true")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--confirm-live-call", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.confirm_live_call:
        raise SystemExit("refusing billable four-Agent eval without --confirm-live-call")
    if args.max_cases is not None and not 1 <= args.max_cases <= 100:
        raise SystemExit("--max-cases must be between 1 and 100")
    report = run(args)
    print(
        json.dumps(
            {"output": str(args.output.resolve()), "metrics": report["metrics"]},
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
