"""Live smoke for discovery -> user PDF upload -> bounded retrieval."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import httpx
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app  # noqa: E402
from taskforge.config import Settings  # noqa: E402

DISCOVERY_QUERY = (
    "retrieval augmented generation for technical documentation assistants"
)


def arxiv_pdf_url(paper: dict[str, object]) -> str | None:
    for raw_url in paper["source_urls"]:
        url = str(raw_url)
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#]+)", url)
        if match:
            arxiv_id = re.sub(r"\.pdf$", "", match.group(1))
            return f"https://arxiv.org/pdf/{arxiv_id}"
    return None


def run(output: Path) -> dict[str, object]:
    started = perf_counter()
    with tempfile.TemporaryDirectory(
        prefix="taskforge-manual-upload-smoke-",
        ignore_cleanup_errors=True,
    ) as raw:
        state = Path(raw)
        settings = Settings(
            sqlite_path=state / "taskforge.sqlite3",
            context_sqlite_path=state / "context.sqlite3",
            operations_sqlite_path=state / "operations.sqlite3",
            orchestration_sqlite_path=state / "orchestration.sqlite3",
            review_case_sqlite_path=state / "review.sqlite3",
            verification_sqlite_path=state / "verification.sqlite3",
            literature_sqlite_path=state / "literature.sqlite3",
            literature_cache_path=state / "literature-cache.sqlite3",
            workspace_root=PROJECT_ROOT,
            artifact_root=state / "artifacts",
            retrieval_routing="lexical",
            general_text_backend="bm25",
        )
        app = create_app(settings)
        headers = {
            "X-TaskForge-Tenant": "live-smoke",
            "X-TaskForge-User": "researcher",
        }
        with TestClient(app) as client:
            search = client.post(
                "/api/literature/search",
                headers=headers,
                json={
                    "conversation_id": "manual-upload-live-smoke",
                    "request": {
                        "request_id": "manual-upload-live-smoke-request",
                        "query": DISCOVERY_QUERY,
                        "research_questions": [
                            "How does retrieval augmented generation combine parametric and non-parametric memory?"
                        ],
                        "result_limit": 20,
                    },
                },
            )
            search.raise_for_status()
            discovery = search.json()
            papers = discovery["papers"]
            if not papers:
                raise RuntimeError("live discovery returned no recommendations")
            if any("abstract" in paper for paper in papers):
                raise RuntimeError("discovery response leaked full abstracts")
            selected = next(
                (paper for paper in papers if arxiv_pdf_url(paper) is not None),
                None,
            )
            if selected is None:
                raise RuntimeError("discovery returned no recommendation with a user-downloadable arXiv link")
            selected_pdf_url = arxiv_pdf_url(selected)
            if selected_pdf_url is None:
                raise RuntimeError("selected recommendation has no arXiv PDF URL")

            scope_response = client.post(
                "/api/research/scopes",
                headers=headers,
                json={
                    "request_id": discovery["request_id"],
                    "conversation_id": "manual-upload-live-smoke",
                    "selected_paper_ids": [selected["paper_id"]],
                    "excluded_paper_ids": [
                        paper["paper_id"]
                        for paper in papers
                        if paper["paper_id"] != selected["paper_id"]
                    ],
                    "user_intent": "Explain how the selected RAG paper combines memory types.",
                    "allowed_expansion": False,
                    "confirm": True,
                },
            )
            scope_response.raise_for_status()
            scope = scope_response.json()

            pre_upload_ingestion = client.post(
                f"/api/research/scopes/{scope['scope_id']}/ingest",
                headers=headers,
            )
            pre_upload_ingestion.raise_for_status()
            pre_upload_rows = pre_upload_ingestion.json()
            if [row["status"] for row in pre_upload_rows] != ["failed"]:
                raise RuntimeError(
                    "ingestion unexpectedly succeeded before the user uploaded a PDF: "
                    f"{pre_upload_rows}"
                )
            if "user-uploaded PDF" not in (pre_upload_rows[0].get("error") or ""):
                raise RuntimeError(
                    "pre-upload ingestion did not fail for the expected boundary: "
                    f"{pre_upload_rows}"
                )

            with httpx.Client(follow_redirects=True, timeout=60.0) as downloader:
                downloaded = downloader.get(selected_pdf_url)
                downloaded.raise_for_status()
                pdf = downloaded.content
            if not pdf.startswith(b"%PDF-"):
                raise RuntimeError("simulated user download did not return a PDF")

            upload = client.put(
                f"/api/research/scopes/{scope['scope_id']}/papers/{selected['paper_id']}/pdf",
                headers={
                    **headers,
                    "Content-Type": "application/pdf",
                    "X-Filename": "rag-paper.pdf",
                },
                content=pdf,
            )
            upload.raise_for_status()
            if upload.json()["status"] != "uploaded":
                raise RuntimeError("PDF upload was not acknowledged")

            ingestion = client.post(
                f"/api/research/scopes/{scope['scope_id']}/ingest",
                headers=headers,
            )
            ingestion.raise_for_status()
            ingestion_rows = ingestion.json()
            if [row["status"] for row in ingestion_rows] != ["indexed"]:
                raise RuntimeError(f"uploaded PDF was not indexed: {ingestion_rows}")

            current_scope = client.get(
                f"/api/research/scopes/{scope['scope_id']}",
                headers=headers,
            )
            current_scope.raise_for_status()
            if current_scope.json()["status"] != "ready":
                raise RuntimeError("Scope did not become ready after successful indexing")

            evidence = client.post(
                "/api/research/evidence/search",
                headers=headers,
                json={
                    "scope_id": scope["scope_id"],
                    "scope_version": scope["scope_version"],
                    "query": "What parametric and non-parametric memory components does RAG combine?",
                    "intent": "method_definition",
                    "top_k": 5,
                    "candidate_k": 20,
                    "mode": "rigorous",
                },
            )
            evidence.raise_for_status()
            evidence_payload = evidence.json()
            if not evidence_payload["evidence"]:
                raise RuntimeError("bounded retrieval returned no PDF evidence")

            provider_reports = discovery["provider_reports"]
            report: dict[str, object] = {
                "schema_version": "1.0",
                "evaluation_type": "manual_upload_live_smoke",
                "created_at": datetime.now(UTC).isoformat(),
                "passed": True,
                "live_external_requests": True,
                "simulated_user_download": selected_pdf_url,
                "discovery": {
                    "recommendation_count": len(papers),
                    "selected_title": selected["title"],
                    "selected_source_urls": selected["source_urls"],
                    "short_description": selected["short_description"],
                    "full_abstract_returned": False,
                    "selection_policy": (
                        "highest-ranked recommendation with a user-downloadable arXiv link"
                    ),
                    "query_rewrite_applied": discovery["query_rewrite_applied"],
                    "provider_reports": provider_reports,
                },
                "upload": {
                    "bytes": len(pdf),
                    "status": upload.json()["status"],
                    "pre_upload_ingestion": pre_upload_rows,
                },
                "ingestion": ingestion_rows,
                "scope_status": current_scope.json()["status"],
                "bounded_retrieval": {
                    "evidence_count": len(evidence_payload["evidence"]),
                    "retrieval_rounds": evidence_payload["retrieval_rounds"],
                    "sufficient": evidence_payload["confidence"]["sufficient"],
                    "top_evidence": [
                        {
                            "section": item.get("section"),
                            "page": item.get("page"),
                            "snippet": item["snippet"],
                            "score": item["score"],
                        }
                        for item in evidence_payload["evidence"][:3]
                    ],
                },
                "elapsed_ms": (perf_counter() - started) * 1_000,
            }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "reports" / "manual-upload-live-smoke.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
