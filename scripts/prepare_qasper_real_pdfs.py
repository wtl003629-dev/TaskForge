"""Acquire and checksum-pin the real PDFs for a locked QASPER split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MAX_PDF_BYTES = 100 * 1024 * 1024
_ARXIV_VERSION_RE = re.compile(r"^v[1-9][0-9]*$")


def _paper_ids(split_path: Path) -> list[str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    paper_ids: list[str] = []
    for raw_case_id in split.get("case_ids", []):
        parts = str(raw_case_id).split(":", 2)
        if len(parts) != 3 or parts[0] != "qasper" or not parts[1]:
            raise ValueError(f"invalid QASPER case ID: {raw_case_id!r}")
        if parts[1] not in paper_ids:
            paper_ids.append(parts[1])
    if not paper_ids:
        raise ValueError("locked QASPER split is empty")
    return paper_ids


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _arxiv_pdf_url(paper_id: str, version: str | None = None) -> str:
    if version is not None and not _ARXIV_VERSION_RE.fullmatch(version):
        raise ValueError(f"invalid arXiv version for {paper_id}: {version!r}")
    suffix = version or ""
    return f"https://arxiv.org/pdf/{paper_id}{suffix}.pdf"


def _load_version_overrides(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("arXiv version overrides must be a JSON object")
    values: dict[str, str] = {}
    for raw_paper_id, raw_version in raw.items():
        paper_id = str(raw_paper_id).strip()
        version = str(raw_version).strip()
        if not paper_id or not _ARXIV_VERSION_RE.fullmatch(version):
            raise ValueError(
                f"invalid arXiv version override: {raw_paper_id!r}={raw_version!r}"
            )
        values[paper_id] = version
    return values


def _download_arxiv_pdf(
    paper_id: str,
    *,
    timeout_seconds: float,
    version: str | None = None,
) -> bytes:
    url = _arxiv_pdf_url(paper_id, version)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TaskForge-QASPER-evaluator/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        final_url = response.geturl()
        if not final_url.startswith(("https://arxiv.org/", "https://export.arxiv.org/")):
            raise ValueError(f"unexpected PDF redirect for {paper_id}: {final_url}")
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > _MAX_PDF_BYTES:
            raise ValueError(f"PDF is too large for {paper_id}: {declared} bytes")
        body = response.read(_MAX_PDF_BYTES + 1)
    if len(body) > _MAX_PDF_BYTES:
        raise ValueError(f"PDF is too large for {paper_id}")
    if not body.startswith(b"%PDF-"):
        raise ValueError(f"downloaded body is not a PDF for {paper_id}")
    return body


def prepare(
    split_path: Path,
    pdf_root: Path,
    manifest_path: Path,
    *,
    timeout_seconds: float = 60.0,
    delay_seconds: float = 3.0,
    version_overrides: dict[str, str] | None = None,
    cohort_version: str = "v1",
) -> dict[str, object]:
    if not _ARXIV_VERSION_RE.fullmatch(cohort_version):
        raise ValueError("cohort_version must use the form v1, v2, ...")
    paper_ids = _paper_ids(split_path)
    selected_versions = version_overrides or {}
    unknown_overrides = sorted(set(selected_versions).difference(paper_ids))
    if unknown_overrides:
        raise ValueError(
            f"arXiv version overrides are outside the split: {unknown_overrides}"
        )
    pdf_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, paper_id in enumerate(paper_ids):
        arxiv_version = selected_versions.get(paper_id)
        source_url = _arxiv_pdf_url(paper_id, arxiv_version)
        # Versioned filenames ensure a stale latest-version file can never be
        # silently relabelled as a historical arXiv revision on a resumed run.
        target = (
            pdf_root / f"{paper_id}{arxiv_version or ''}.pdf"
        ).resolve()
        if target.is_file():
            body = target.read_bytes()
            if not body.startswith(b"%PDF-"):
                raise ValueError(f"existing file is not a PDF for {paper_id}: {target}")
        else:
            if index and delay_seconds:
                time.sleep(delay_seconds)
            body = _download_arxiv_pdf(
                paper_id,
                timeout_seconds=timeout_seconds,
                version=arxiv_version,
            )
            temporary = target.with_suffix(".pdf.part")
            temporary.write_bytes(body)
            os.replace(temporary, target)
        rows.append(
            {
                "paper_id": paper_id,
                "path": os.path.relpath(target, manifest_path.parent.resolve()),
                "sha256": _sha256(body),
                "source_url": source_url,
                "arxiv_version": arxiv_version or "latest_at_acquisition",
                "acquired_at": datetime.now(UTC).isoformat(),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "dataset": "QASPER",
        "cohort_id": (
            f"{json.loads(split_path.read_text(encoding='utf-8')).get('split_id')}"
            f"-real-pdf-{cohort_version}"
        ),
        "split_path": str(split_path),
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "source": "arXiv PDF",
        "arxiv_version_overrides": selected_versions,
        "papers": rows,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "eval" / "splits" / "qasper-validation-upload-50-v2.json",
    )
    parser.add_argument(
        "--pdf-root",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-real-pdfs-v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-real-pdfs-v1.json",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--delay-seconds", type=float, default=3.0)
    parser.add_argument(
        "--arxiv-version-overrides",
        type=Path,
        default=None,
        help='Optional JSON object such as {"1902.09666": "v1"}.',
    )
    parser.add_argument("--cohort-version", default="v1")
    args = parser.parse_args()
    report = prepare(
        args.split,
        args.pdf_root,
        args.manifest,
        timeout_seconds=args.timeout_seconds,
        delay_seconds=args.delay_seconds,
        version_overrides=_load_version_overrides(args.arxiv_version_overrides),
        cohort_version=args.cohort_version,
    )
    print(
        json.dumps(
            {"manifest": str(args.manifest), "papers": len(report["papers"])},
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
