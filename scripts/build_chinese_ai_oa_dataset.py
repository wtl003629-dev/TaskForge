"""Build a quality-gated Chinese AI open-access full-text corpus.

The builder uses OpenAlex only for discovery and OA metadata.  It downloads
only an explicitly exposed PDF URL, records the source/license/DOI, extracts
text with pypdf, and keeps a paper only when the downloaded file looks like a
Chinese full paper.  The output is intentionally split into journal OA and
curated preprint records; preprints are not silently presented as peer-
reviewed publications.

This script is resumable at the paper level.  Re-running it reuses existing
PDFs and metadata files, which makes a later expansion from a pilot to a
larger corpus inexpensive.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
from pypdf import PdfReader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / ".taskforge" / "datasets" / "chinese-ai-oa-v1"
OPENALEX_URL = "https://api.openalex.org/works"
USER_AGENT = "TaskForge-Chinese-AI-OA-Dataset/1.0 (openalex discovery; polite client)"
TERMS = (
    "机器学习",
    "深度学习",
    "大语言模型",
    "自然语言处理",
    "计算机视觉",
    "强化学习",
    "生成式人工智能",
    "知识图谱",
    "多模态",
    "检索增强生成",
)
AI_TOKENS = (
    "人工智能",
    "机器学习",
    "深度学习",
    "大语言模型",
    "语言模型",
    "自然语言处理",
    "计算机视觉",
    "强化学习",
    "生成式人工智能",
    "生成式",
    "知识图谱",
    "多模态",
    "神经网络",
    "卷积神经",
    "Transformer",
    "检索增强",
    "智能体",
    "深度神经",
)
KNOWN_SOURCES = (
    "Journal of Software",
    "软件学报",
    "Acta Automatica Sinica",
    "自动化学报",
    "Journal of Computer Research and Development",
    "计算机研究与发展",
    "Journal of Chinese Information Processing",
    "中文信息学报",
    "Pattern Recognition and Artificial Intelligence",
    "模式识别与人工智能",
    "Science China Information Sciences",
    "中国科学：信息科学",
    "Chinese Journal of Computers",
    "计算机学报",
)
ALLOWED_LICENSES = {"cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "cc0", "public-domain"}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _abstract_text(inverted: Any) -> str:
    if not isinstance(inverted, dict):
        return ""
    positions: dict[int, str] = {}
    for token, offsets in inverted.items():
        if not isinstance(offsets, list):
            continue
        for offset in offsets:
            try:
                positions[int(offset)] = str(token)
            except (TypeError, ValueError):
                continue
    return " ".join(positions[index] for index in sorted(positions))


def _zh_ratio(text: str) -> float:
    if not text:
        return 0.0
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    letters = len(re.findall(r"[A-Za-z\u3400-\u9fff0-9]", text))
    return chinese / max(letters, 1)


def _topic_score(title: str, abstract: str) -> float:
    text = f"{title} {abstract}".lower()
    hits = sum(1 for token in AI_TOKENS if token.lower() in text)
    return min(1.0, hits / 4.0)


def _source_name(work: dict[str, Any]) -> str:
    location = work.get("best_oa_location") or work.get("primary_location") or {}
    source = location.get("source") or {}
    return str(source.get("display_name") or location.get("raw_source_name") or "Unknown")


def _source_info(work: dict[str, Any]) -> dict[str, Any]:
    location = work.get("best_oa_location") or work.get("primary_location") or {}
    source = location.get("source") or {}
    return {
        "name": _source_name(work),
        "id": source.get("id"),
        "issn_l": source.get("issn_l"),
        "is_core": bool(source.get("is_core")),
        "is_in_doaj": bool(source.get("is_in_doaj")),
        "source_is_oa": bool(source.get("is_oa")),
        "type": source.get("type"),
    }


def _pdf_url(work: dict[str, Any]) -> str | None:
    locations: list[dict[str, Any]] = []
    best = work.get("best_oa_location")
    if isinstance(best, dict):
        locations.append(best)
    for location in work.get("locations") or []:
        if isinstance(location, dict):
            locations.append(location)
    content = work.get("content_urls") or {}
    if isinstance(content, dict) and content.get("pdf"):
        locations.append({"pdf_url": content["pdf"], "is_oa": True, "license": None})
    for location in locations:
        url = location.get("pdf_url")
        if not url or not location.get("is_oa", True):
            continue
        value = str(url)
        if value.lower().startswith(("http://", "https://")):
            return value
    return None


def _quality_score(work: dict[str, Any], title: str, abstract: str) -> tuple[float, str]:
    source = _source_info(work)
    location = work.get("best_oa_location") or {}
    license_name = str(location.get("license") or "").lower()
    score = 0.0
    if source["name"] in KNOWN_SOURCES or any(
        known.lower() in source["name"].lower() for known in KNOWN_SOURCES
    ):
        score += 3.0
    if source["is_core"]:
        score += 2.5
    if source["is_in_doaj"]:
        score += 2.0
    if license_name in ALLOWED_LICENSES:
        score += 1.5
    if location.get("version") == "publishedVersion":
        score += 1.0
    if work.get("type") == "article":
        score += 0.5
    if work.get("cited_by_count", 0) >= 3:
        score += min(1.5, math.log1p(int(work.get("cited_by_count", 0))) / 3.0)
    if len(abstract) >= 160:
        score += 0.5
    if _topic_score(title, abstract) >= 0.5:
        score += 1.0
    bucket = "journal_oa" if location.get("version") == "publishedVersion" and source["type"] == "journal" else "curated_preprints"
    return round(score, 3), bucket


def _candidate(work: dict[str, Any], matched_term: str) -> dict[str, Any] | None:
    title = str(work.get("title") or work.get("display_name") or "").strip()
    abstract = _abstract_text(work.get("abstract_inverted_index"))
    if not title or not abstract or _topic_score(title, abstract) < 0.5:
        return None
    if work.get("language") != "zh" or not (work.get("open_access") or {}).get("is_oa"):
        return None
    if work.get("is_retracted") or work.get("is_paratext"):
        return None
    url = _pdf_url(work)
    if not url:
        return None
    if _zh_ratio(f"{title} {abstract}") < 0.20:
        return None
    score, bucket = _quality_score(work, title, abstract)
    # A published/core/DOAJ paper must score at least 4; a preprint must score
    # at least 3 and is kept in a separate bucket.  The thresholds are
    # deliberately explicit and can be tightened without changing ingestion.
    if score < (4.0 if bucket == "journal_oa" else 3.0):
        return None
    paper_id = str(work.get("doi") or work.get("id") or "").strip()
    if not paper_id:
        return None
    return {
        "paper_id": paper_id,
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": title,
        "abstract": abstract,
        "language": work.get("language"),
        "publication_year": work.get("publication_year"),
        "publication_date": work.get("publication_date"),
        "authors": [
            str((row.get("author") or {}).get("display_name") or "").strip()
            for row in work.get("authorships") or []
            if isinstance(row, dict) and (row.get("author") or {}).get("display_name")
        ],
        "source": _source_info(work),
        "license": (work.get("best_oa_location") or {}).get("license"),
        "oa_status": (work.get("open_access") or {}).get("oa_status"),
        "landing_page_url": (work.get("best_oa_location") or {}).get("landing_page_url"),
        "pdf_url": url,
        "cited_by_count": int(work.get("cited_by_count") or 0),
        "quality_score": score,
        "quality_bucket": bucket,
        "topic_score": round(_topic_score(title, abstract), 3),
        "matched_terms": [matched_term],
        "retrieved_at": datetime.now(UTC).isoformat(),
    }


def _merge_candidate(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    terms = sorted(set(existing.get("matched_terms", [])) | set(candidate.get("matched_terms", [])))
    if candidate.get("quality_score", 0) > existing.get("quality_score", 0):
        chosen = {**candidate}
    else:
        chosen = {**existing}
    chosen["matched_terms"] = terms
    return chosen


def _fetch_candidates(
    client: httpx.Client,
    *,
    per_page: int,
    max_pages_per_term: int,
    min_year: int,
    max_year: int,
    mailto: str | None,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    filt = f"language:zh,is_oa:true,type:article,from_publication_date:{min_year}-01-01,to_publication_date:{max_year}-12-31"
    for term in TERMS:
        cursor = "*"
        for _page in range(max_pages_per_term):
            params = {
                "search": term,
                "filter": filt,
                "sort": "cited_by_count:desc",
                "per-page": per_page,
                "cursor": cursor,
            }
            if mailto:
                params["mailto"] = mailto
            # OpenAlex applies a shared public-pool rate limit.  Respect an
            # explicit Retry-After header and back off before giving up so a
            # long collection can be resumed instead of failing mid-run.
            for attempt in range(6):
                response = client.get(OPENALEX_URL, params=params)
                if response.status_code != 429:
                    break
                retry_after = response.headers.get("retry-after")
                try:
                    delay = max(1.0, min(60.0, float(retry_after or "5")))
                except ValueError:
                    delay = 5.0
                time.sleep(delay * (attempt + 1))
            response.raise_for_status()
            payload = response.json()
            for work in payload.get("results") or []:
                if not isinstance(work, dict):
                    continue
                row = _candidate(work, term)
                if row is not None:
                    existing = by_id.get(row["paper_id"])
                    by_id[row["paper_id"]] = row if existing is None else _merge_candidate(existing, row)
            cursor = str((payload.get("meta") or {}).get("next_cursor") or "")
            if not cursor or not (payload.get("results") or []):
                break
            time.sleep(0.2)
    rows = list(by_id.values())
    rows.sort(key=lambda row: (-float(row.get("quality_score", 0)), -int(row.get("cited_by_count", 0)), row["paper_id"]))
    return rows


def _extract_pdf(path: Path) -> tuple[str, int, float]:
    reader = PdfReader(str(path), strict=False)
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(str(page.extract_text() or ""))
        except Exception:
            chunks.append("")
    text = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
    return text, len(reader.pages), _zh_ratio(text)


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _download_one(client: httpx.Client, candidate: dict[str, Any], pdf_dir: Path, min_bytes: int) -> dict[str, Any]:
    key = hashlib.sha256(candidate["paper_id"].encode("utf-8")).hexdigest()[:24]
    target = pdf_dir / f"{key}.pdf"
    result = {**candidate, "local_pdf": str(target), "status": "pending"}
    try:
        if not target.exists() or target.stat().st_size < min_bytes:
            response = client.get(candidate["pdf_url"], follow_redirects=True, timeout=60)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            data = response.content
            if len(data) < min_bytes or ("pdf" not in content_type and not data.startswith(b"%PDF")):
                result.update({"status": "rejected_download", "error": f"not a PDF or too small ({len(data)} bytes)"})
                return result
            target.write_bytes(data)
        text, pages, zh_ratio = _extract_pdf(target)
        result.update({
            "status": "accepted" if pages >= 4 and len(text) >= 6000 and zh_ratio >= 0.12 else "rejected_quality",
            "sha256": _sha256_file(target),
            "bytes": target.stat().st_size,
            "pages": pages,
            "text_chars": len(text),
            "zh_ratio": round(zh_ratio, 4),
            "text": text if pages >= 4 and len(text) >= 6000 and zh_ratio >= 0.12 else "",
        })
    except Exception as exc:  # network and malformed-PDF boundary
        result.update({"status": "download_error", "error": str(exc)[:500]})
    return result


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidates.json"
    records_path = output / "records.jsonl"
    candidates = _json_load(candidates_path, [])
    if not candidates:
        with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
            candidates = _fetch_candidates(
                client,
                per_page=args.per_page,
                max_pages_per_term=args.max_pages_per_term,
                min_year=args.min_year,
                max_year=args.max_year,
                mailto=args.mailto,
            )
        _json_dump(candidates_path, candidates)
    candidates = candidates[: args.candidate_limit]

    records: dict[str, dict[str, Any]] = {}
    if records_path.exists():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("paper_id"):
                    records[str(row["paper_id"])] = row
            except ValueError:
                continue
    pdf_dir = output / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    with httpx.Client(headers={"User-Agent": USER_AGENT}) as client:
        for index, candidate in enumerate(candidates, 1):
            if candidate["paper_id"] in records and records[candidate["paper_id"]].get("status") == "accepted":
                continue
            row = _download_one(client, candidate, pdf_dir, args.min_bytes)
            records[candidate["paper_id"]] = row
            if index % 10 == 0 or index == len(candidates):
                records_path.write_text(
                    "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records.values()),
                    encoding="utf-8",
                )
            time.sleep(args.download_interval)

    accepted = [row for row in records.values() if row.get("status") == "accepted"]
    accepted.sort(key=lambda row: (-float(row.get("quality_score", 0)), row["paper_id"]))
    accepted = accepted[: args.limit]
    paper_rows = [{key: value for key, value in row.items() if key != "text"} for row in accepted]
    corpus_rows = [
        {
            "document_id": row["paper_id"],
            "paper_id": row["paper_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "text": row["text"],
            "language": "zh",
            "source_dataset": "chinese-ai-oa-v1",
            "source": row["source"],
            "source_url": row.get("landing_page_url") or row["pdf_url"],
            "pdf_url": row["pdf_url"],
            "license": row.get("license"),
            "quality_bucket": row["quality_bucket"],
            "quality_score": row["quality_score"],
            "metadata": {
                "doi": row.get("doi"),
                "openalex_id": row.get("openalex_id"),
                "publication_year": row.get("publication_year"),
                "authors": row.get("authors", []),
                "pages": row.get("pages"),
                "text_chars": row.get("text_chars"),
                "zh_ratio": row.get("zh_ratio"),
                "sha256": row.get("sha256"),
            },
        }
        for row in accepted
    ]
    _write_jsonl_gz(output / "papers.jsonl.gz", paper_rows)
    _write_jsonl_gz(output / "corpus.jsonl.gz", corpus_rows)
    by_bucket = defaultdict(int)
    for row in accepted:
        by_bucket[row["quality_bucket"]] += 1
    manifest = {
        "dataset_id": "chinese-ai-oa-v1",
        "built_at": datetime.now(UTC).isoformat(),
        "requested_limit": args.limit,
        "candidate_count": len(candidates),
        "download_record_count": len(records),
        "accepted_count": len(accepted),
        "accepted_by_bucket": dict(sorted(by_bucket.items())),
        "source": "OpenAlex discovery plus directly exposed OA PDF URLs",
        "terms": list(TERMS),
        "quality_gate": {
            "language": "OpenAlex zh and extracted Chinese ratio >= 0.12",
            "topic_score": ">= 0.5 over title and abstract controlled vocabulary",
            "pdf": ">= 4 pages and >= 6000 extracted characters",
            "quality_score": "journal >= 4.0; preprint >= 3.0; preprints are kept separate",
            "deduplication": "paper identifier and downloaded PDF SHA-256",
        },
        "license_note": "The corpus records license metadata per work. Redistribution must follow each source's license and terms.",
        "files": {
            "papers": "papers.jsonl.gz",
            "corpus": "corpus.jsonl.gz",
            "candidates": "candidates.json",
            "records": "records.jsonl",
        },
    }
    _json_dump(output / "manifest.json", manifest)
    quality_rows = [
        {key: value for key, value in row.items() if key != "text"}
        for row in sorted(records.values(), key=lambda value: (value.get("status", ""), value.get("paper_id", "")))
    ]
    _json_dump(output / "quality_report.json", {"summary": manifest, "rows": quality_rows})
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000, help="accepted papers to keep")
    parser.add_argument("--candidate-limit", type=int, default=3000, help="maximum candidates to download")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-year", type=int, default=2015)
    parser.add_argument("--max-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--per-page", type=int, default=200)
    parser.add_argument("--max-pages-per-term", type=int, default=5)
    parser.add_argument("--min-bytes", type=int, default=30_000)
    parser.add_argument("--download-interval", type=float, default=0.4)
    parser.add_argument("--mailto", default="", help="optional OpenAlex polite-pool email")
    args = parser.parse_args()
    if args.limit <= 0 or args.candidate_limit <= 0:
        parser.error("--limit and --candidate-limit must be positive")
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
