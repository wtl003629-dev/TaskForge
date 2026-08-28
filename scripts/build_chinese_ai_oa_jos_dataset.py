"""Collect quality-gated Chinese AI papers from the Journal of Software.

The Journal of Software (软件学报) exposes issue pages and article PDF links
without a login.  This collector uses those public pages as the source of
truth, keeps only research articles whose title/abstract is AI-related, then
downloads and text-checks the PDF.  It records source URLs and a conservative
rights note: public download does not imply permission to redistribute the
PDFs.

The resulting corpus is suitable for a local RAG experiment.  Journal papers
and full-text quality failures are auditable in ``quality_report.json``; no
abstract-only record is promoted to the corpus.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import html
import json
import re
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import httpx
from pypdf import PdfReader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / ".taskforge" / "datasets" / "chinese-ai-oa-jos-v2"
BASE = "https://www.jos.org.cn"
USER_AGENT = "TaskForge-Chinese-AI-OA-JOS-Dataset/1.0 (polite local research client)"
AI_TERMS = (
    "人工智能",
    "机器学习",
    "深度学习",
    "大语言模型",
    "语言模型",
    "自然语言处理",
    "计算机视觉",
    "强化学习",
    "知识图谱",
    "多模态",
    "生成式",
    "生成模型",
    "神经网络",
    "卷积神经",
    "Transformer",
    "检索增强",
    "智能体",
    "语义分割",
    "目标检测",
    "推荐系统",
    "迁移学习",
    "聚类算法",
    "分类算法",
    "表示学习",
    "异常检测",
    "图神经",
)
EXCLUDED_TITLE = re.compile(r"前言|编者按|征稿|通知|启事|声明|更正|勘误|目录|致谢|悼念")


def _clean(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _zh_ratio(text: str) -> float:
    if not text:
        return 0.0
    zh = len(re.findall(r"[\u3400-\u9fff]", text))
    alnum = len(re.findall(r"[A-Za-z0-9\u3400-\u9fff]", text))
    return zh / max(alnum, 1)


def _topic_score(title: str, abstract: str) -> float:
    text = f"{title} {abstract}".lower()
    return min(1.0, sum(token.lower() in text for token in AI_TERMS) / 3.0)


def _parse_years(client: httpx.Client) -> list[tuple[int, int]]:
    response = client.get(f"{BASE}/jos/issue/get_year_info", timeout=30)
    response.raise_for_status()
    rows: list[tuple[int, int]] = []
    for part in response.text.split("#"):
        try:
            year, volume = part.split(",", 1)
            if int(year) > 0:
                rows.append((int(year), int(volume)))
        except (TypeError, ValueError):
            continue
    return rows


def _issue_numbers(client: httpx.Client, year: int) -> list[str]:
    response = client.get(f"{BASE}/jos/issue/get_all_issue_info", timeout=30)
    response.raise_for_status()
    match = re.search(rf"<{year}>(.*?)</{year}>", response.text)
    if not match:
        return []
    return [part.split(",", 1)[0] for part in match.group(1).split(":") if part]


def _parse_issue(text: str, year: int, issue: str) -> list[dict[str, Any]]:
    blocks = re.findall(r'<li class="article_line">(.*?)</li>\s*', text, re.S | re.I)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        file_no = re.search(r"FileNO(\d+)", block, re.I)
        title_match = re.search(
            r"href=['\"](?:/)?jos/article/abstract/(\d+)['\"][^>]*>(.*?)</a>",
            block,
            re.S | re.I,
        )
        if not file_no or not title_match:
            continue
        paper_id = file_no.group(1)
        title = _clean(title_match.group(2))
        author_match = re.search(r"article_author.*?<a[^>]*>(.*?)</a>", block, re.S | re.I)
        abstract_match = re.search(
            r'<div class="abstract_body">.*?<p><b>摘要:</b>(.*?)</p>',
            block,
            re.S | re.I,
        )
        position_match = re.search(r'<p class="article_position">(.*?)</p>', block, re.S | re.I)
        doi_match = re.search(r"id=['\"]doi['\"][^>]*>(.*?)</a>", block, re.S | re.I)
        cstr_match = re.search(r"id=['\"]cstr['\"][^>]*>(.*?)</a>", block, re.S | re.I)
        abstract = _clean(abstract_match.group(1)) if abstract_match else ""
        if not abstract or EXCLUDED_TITLE.search(title):
            continue
        topic = _topic_score(title, abstract)
        # A single strong AI term (for example "机器学习" or "大语言模型")
        # is sufficient because the source journal is already a computer
        # science venue.  Requiring two terms would silently drop legitimate
        # single-topic papers such as a focused semantic-segmentation study.
        if topic < 0.30:
            continue
        rows.append(
            {
                "paper_id": f"jos:{paper_id}",
                "jos_id": paper_id,
                "title": title,
                "abstract": abstract,
                "authors": _clean(author_match.group(1)).split("，") if author_match else [],
                "bibliography": _clean(position_match.group(1)) if position_match else "",
                "doi": _clean(doi_match.group(1)) if doi_match else None,
                "cstr": _clean(cstr_match.group(1)) if cstr_match else None,
                "year": year,
                "issue": issue,
                "source": {
                    "name": "Journal of Software / 软件学报",
                    "url": f"{BASE}/jos/article/abstract/{paper_id}",
                },
                "landing_page_url": f"{BASE}/jos/article/abstract/{paper_id}",
                "pdf_url": f"{BASE}/jos/article/pdf/{paper_id}",
                "quality_bucket": "journal_oa",
                "quality_score": round(7.0 + (0.5 if doi_match else 0.0) + min(len(abstract), 600) / 1200, 3),
                "topic_score": round(topic, 3),
                "retrieved_at": datetime.now(UTC).isoformat(),
            }
        )
    return rows


def _crawl_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        available = _parse_years(client)
        selected = [(year, volume) for year, volume in available if args.min_year <= year <= args.max_year]
        for index, (year, volume) in enumerate(selected, 1):
            issues = _issue_numbers(client, year)
            for issue_index, issue in enumerate(issues):
                url = f"{BASE}/jos/article/issue/{year}_{volume}_{issue}"
                response = client.get(url, timeout=45)
                if response.status_code != 200:
                    continue
                for row in _parse_issue(response.text, year, issue):
                    rows[row["paper_id"]] = row
                if (index % 5 == 0 and issue_index == len(issues) - 1) or index == len(selected):
                    print(f"crawled {year}: {len(rows)} AI candidates", flush=True)
                time.sleep(args.page_interval)
    result = list(rows.values())
    result.sort(key=lambda row: (-float(row["quality_score"]), -int(row["year"]), row["paper_id"]))
    return result


def _extract_pdf(path: Path) -> tuple[str, int, float]:
    reader = PdfReader(str(path), strict=False)
    pages: list[str] = []
    for page in reader.pages:
        try:
            value = str(page.extract_text() or "").strip()
        except Exception:
            value = ""
        if value:
            pages.append(value)
    text = "\n\n".join(pages)
    return text, len(reader.pages), _zh_ratio(text)


def _download_one(candidate: dict[str, Any], pdf_dir: Path, min_bytes: int) -> dict[str, Any]:
    key = hashlib.sha256(candidate["paper_id"].encode("utf-8")).hexdigest()[:24]
    target = pdf_dir / f"{key}.pdf"
    result = {**candidate, "local_pdf": str(target), "status": "pending"}
    try:
        if not target.exists() or target.stat().st_size < min_bytes:
            with httpx.Client(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
                response = client.get(candidate["pdf_url"], timeout=90)
                response.raise_for_status()
                data = response.content
            content_type = response.headers.get("content-type", "").lower()
            if len(data) < min_bytes or ("pdf" not in content_type and not data.startswith(b"%PDF")):
                result.update({"status": "rejected_download", "error": f"not PDF or too small ({len(data)} bytes)"})
                return result
            target.write_bytes(data)
        text, pages, zh_ratio = _extract_pdf(target)
        accepted = pages >= 4 and len(text) >= 6000 and zh_ratio >= 0.12
        result.update(
            {
                "status": "accepted" if accepted else "rejected_quality",
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
                "pages": pages,
                "text_chars": len(text),
                "zh_ratio": round(zh_ratio, 4),
                "text": text if accepted else "",
            }
        )
    except Exception as exc:
        result.update({"status": "download_error", "error": str(exc)[:500]})
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "candidates.json"
    candidates = json.loads(candidates_path.read_text(encoding="utf-8")) if candidates_path.exists() else []
    if not candidates:
        candidates = _crawl_candidates(args)
        candidates_path.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    candidates = candidates[: args.candidate_limit]
    pdf_dir = output / "pdfs"
    pdf_dir.mkdir(exist_ok=True)
    records_path = output / "records.jsonl"
    records: dict[str, dict[str, Any]] = {}
    if records_path.exists():
        # ``str.splitlines`` also splits on U+2028/U+2029, which can occur in
        # extracted PDF text.  JSONL records are delimited by a literal LF,
        # so split only on that byte to preserve valid records on resume.
        for line in records_path.read_text(encoding="utf-8").split("\n"):
            try:
                row = json.loads(line)
                if row.get("paper_id"):
                    records[str(row["paper_id"])] = row
            except ValueError:
                continue
    pending = [row for row in candidates if records.get(row["paper_id"], {}).get("status") != "accepted"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_download_one, row, pdf_dir, args.min_bytes): row["paper_id"] for row in pending}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = future.result()
            records[row["paper_id"]] = row
            if index % 10 == 0 or index == len(pending):
                records_path.write_text(
                    "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records.values()),
                    encoding="utf-8",
                )
                accepted_count = sum(value.get("status") == "accepted" for value in records.values())
                print(f"downloaded {index}/{len(pending)}; accepted {accepted_count}", flush=True)
            time.sleep(args.download_interval)

    accepted = [row for row in records.values() if row.get("status") == "accepted"]
    accepted.sort(key=lambda row: (-float(row.get("quality_score", 0)), -int(row.get("year", 0)), row["paper_id"]))
    accepted = accepted[: args.limit]
    papers = [{key: value for key, value in row.items() if key not in {"text"}} for row in accepted]
    corpus = [
        {
            "document_id": row["paper_id"],
            "paper_id": row["paper_id"],
            "title": row["title"],
            "abstract": row["abstract"],
            "text": row["text"],
            "language": "zh",
            "source_dataset": "chinese-ai-oa-jos-v2",
            "source": row["source"],
            "source_url": row["landing_page_url"],
            "pdf_url": row["pdf_url"],
            "license": "website-download; redistribution not assumed",
            "quality_bucket": row["quality_bucket"],
            "quality_score": row["quality_score"],
            "metadata": {
                "doi": row.get("doi"),
                "cstr": row.get("cstr"),
                "authors": row.get("authors", []),
                "year": row.get("year"),
                "issue": row.get("issue"),
                "pages": row.get("pages"),
                "text_chars": row.get("text_chars"),
                "zh_ratio": row.get("zh_ratio"),
                "sha256": row.get("sha256"),
            },
        }
        for row in accepted
    ]
    _write_jsonl_gz(output / "papers.jsonl.gz", papers)
    _write_jsonl_gz(output / "corpus.jsonl.gz", corpus)
    status_counts = Counter(row.get("status") for row in records.values())
    manifest = {
        "dataset_id": "chinese-ai-oa-jos-v2",
        "built_at": datetime.now(UTC).isoformat(),
        "requested_limit": args.limit,
        "candidate_count": len(candidates),
        "download_record_count": len(records),
        "accepted_count": len(accepted),
        "status_counts": dict(status_counts),
        "source": "Journal of Software / 软件学报 public issue and article PDF pages",
        "year_range": [args.min_year, args.max_year],
        "quality_gate": {
            "topic": "title/abstract contains at least one controlled AI term",
            "pdf": ">= 4 pages, >= 6000 extracted chars, Chinese ratio >= 0.12",
            "exclusions": "front matter, notices, corrections, directories and similar titles",
            "deduplication": "JOS article id and downloaded PDF SHA-256",
        },
        "rights_note": "The website exposes the PDFs for reading/download. Each record retains the source URL; redistribution rights are not inferred.",
        "files": {
            "papers": "papers.jsonl.gz",
            "corpus": "corpus.jsonl.gz",
            "candidates": "candidates.json",
            "records": "records.jsonl",
            "pdfs": "pdfs/",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    quality_rows = [{key: value for key, value in row.items() if key != "text"} for row in records.values()]
    (output / "quality_report.json").write_text(
        json.dumps({"summary": manifest, "rows": quality_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--candidate-limit", type=int, default=1400)
    parser.add_argument("--min-year", type=int, default=2000)
    parser.add_argument("--max-year", type=int, default=datetime.now(UTC).year)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-bytes", type=int, default=30_000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--page-interval", type=float, default=0.15)
    parser.add_argument("--download-interval", type=float, default=0.5)
    args = parser.parse_args()
    if args.limit <= 0 or args.candidate_limit <= 0 or args.workers <= 0:
        parser.error("limit, candidate-limit and workers must be positive")
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
