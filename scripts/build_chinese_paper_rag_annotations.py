"""Build a small, evidence-bound Chinese paper RAG annotation set.

The source corpus is the locally built Chinese full-text collection.  One
DeepSeek request creates three questions per paper, but every answer must cite
one or more deterministic text chunks.  The result is deliberately labelled
``silver`` until a human audits it; the script never treats a title probe as a
gold relevance judgement.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx

from taskforge.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / ".taskforge" / "datasets" / "chinese-ai-oa-jos-v2" / "corpus.jsonl.gz"
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "queries" / "chinese-paper-rag-30-v1"
DEFAULT_SEED = 20260827
CHUNK_CHARS = 1_800
MAX_PAPER_CHARS = 48_000
QUESTION_TYPES = ("method", "result", "contribution")
WHITESPACE_RE = re.compile(r"\s+")
NON_CONTENT_RE = re.compile(r"[^0-9A-Za-z_\u3400-\u9fff]+")


def _read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).replace("\xa0", " ")
    return WHITESPACE_RE.sub(" ", text).strip()


def _compact_for_match(value: object) -> str:
    """Ignore whitespace and punctuation differences in extracted PDF text."""

    return NON_CONTENT_RE.sub("", _normalise(value)).casefold()


def _derive_quote(answer: str, source: str) -> str:
    """Pick a source sentence with the strongest lexical overlap to an answer."""

    sentences = [
        value.strip()
        for value in re.split(r"(?<=[。！？.!?])|\n+", source)
        if value.strip()
    ]
    answer_compact = _compact_for_match(answer)
    answer_terms = {
        answer_compact[index : index + 2]
        for index in range(max(0, len(answer_compact) - 1))
    }
    if not answer_terms:
        return _normalise(source)[:160]
    ranked = sorted(
        sentences,
        key=lambda sentence: len(answer_terms & {
            _compact_for_match(sentence)[index : index + 2]
            for index in range(max(0, len(_compact_for_match(sentence)) - 1))
        }),
        reverse=True,
    )
    return _normalise(ranked[0] if ranked else source)[:180]


def _chunk_text(text: str, *, max_chars: int = CHUNK_CHARS) -> list[str]:
    cleaned = str(text).replace("\xa0", " ").strip()
    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + max_chars)
        if end < len(cleaned):
            boundary = max(
                cleaned.rfind("\n", start + max_chars // 2, end),
                cleaned.rfind("。", start + max_chars // 2, end),
                cleaned.rfind("；", start + max_chars // 2, end),
                cleaned.rfind(" ", start + max_chars // 2, end),
            )
            if boundary > start:
                end = boundary + (0 if cleaned[boundary] == " " else 1)
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end <= start:
            end = start + max_chars
        start = end
    return chunks


def _select_papers(records: list[dict[str, Any]], *, seed: int, limit: int) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in records
        if str(row.get("language", "")).casefold() == "zh"
        and 12_000 <= len(str(row.get("text", ""))) <= MAX_PAPER_CHARS
    ]
    if len(eligible) < limit:
        raise RuntimeError(f"only {len(eligible)} eligible Chinese papers, need {limit}")
    # Stratify by source length so the small benchmark does not consist only of
    # the shortest or longest extracted PDFs.
    eligible.sort(key=lambda row: len(str(row.get("text", ""))))
    buckets = [eligible[index::3] for index in range(3)]
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bucket in buckets:
        selected.extend(rng.sample(bucket, min(limit // 3, len(bucket))))
    remaining = limit - len(selected)
    if remaining:
        pool = [row for row in eligible if row not in selected]
        selected.extend(rng.sample(pool, remaining))
    selected.sort(key=lambda row: str(row.get("paper_id") or row.get("document_id")))
    return selected


def _prompt(title: str, abstract: str, chunks: list[str], *, correction: str = "") -> str:
    numbered = "\n\n".join(
        f"[CHUNK_{index:03d}]\n{chunk}" for index, chunk in enumerate(chunks)
    )
    correction_text = f"\n上一次输出存在问题：{correction}\n请重新完整输出。\n" if correction else ""
    return f"""你是中文科研论文 RAG 数据集标注员。只能依据下面给出的论文文本块进行标注。

论文标题：{title}
论文摘要：{abstract}

请生成恰好 3 条中文问题，类型必须分别为 method、result、contribution。
要求：
1. 问题必须询问论文正文中的具体方法、实验结果或核心贡献，不能只问标题、作者、DOI 或参考文献。
2. answer 必须直接由给出的文本块支持，不得补充文本之外的事实。
3. evidence 至少包含一个文本块；每个 evidence 的 chunk_index 必须是给出的 CHUNK 编号，quote 必须是该块中的连续原文片段，长度不超过 180 个汉字。
4. 忽略参考文献列表，不要把英文摘要的重复内容当作新的结论。
5. 问题应适合评估论文 RAG 的召回和证据引用能力，避免答案为“是/否”的问题；贡献问题必须包含论文中的具体方法名、系统名或量化结论，禁止使用“本文的主要贡献有哪些？”这类泛化模板。
6. 只输出合法 JSON，不要 Markdown 代码围栏，不要输出额外解释。

JSON 格式：
{{"items":[{{"question_type":"method|result|contribution","question":"...","answer":"...","evidence":[{{"chunk_index":0,"quote":"..."}}],"difficulty":"easy|medium|hard"}}]}}
{correction_text}
论文文本块：
{numbered}
"""


def _parse_content(payload: Mapping[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ValueError("DeepSeek response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ValueError("DeepSeek response has no message")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) for item in content if isinstance(item, Mapping)
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek response content is empty")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    decoded = json.loads(cleaned)
    if not isinstance(decoded, dict):
        raise ValueError("annotation response must be an object")
    return decoded


def _validate_items(raw: Mapping[str, Any], chunks: list[str]) -> tuple[list[dict[str, Any]], str | None]:
    items = raw.get("items")
    if not isinstance(items, list) or len(items) != 3:
        return [], "items must contain exactly three entries"
    by_type: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, Mapping):
            return [], "each item must be an object"
        question_type = str(item.get("question_type", "")).strip().casefold()
        if question_type not in QUESTION_TYPES or question_type in by_type:
            return [], "question_type must contain one unique method/result/contribution"
        question = _normalise(item.get("question"))
        answer = _normalise(item.get("answer"))
        evidence = item.get("evidence")
        if not question or not answer or not isinstance(evidence, list) or not evidence:
            return [], "question, answer and evidence are required"
        clean_evidence: list[dict[str, Any]] = []
        for ref in evidence[:3]:
            if not isinstance(ref, Mapping):
                return [], "evidence entries must be objects"
            try:
                index = int(ref.get("chunk_index"))
            except (TypeError, ValueError):
                return [], "chunk_index must be an integer"
            quote = _normalise(ref.get("quote"))
            if not 0 <= index < len(chunks):
                return [], "evidence chunk index or quote is invalid"
            source_text = _normalise(chunks[index])
            verified = bool(
                quote
                and (
                    quote in source_text
                    or _compact_for_match(quote) in _compact_for_match(source_text)
                )
            )
            if quote and not verified:
                matching_indexes = [
                    candidate
                    for candidate, chunk in enumerate(chunks)
                    if quote in _normalise(chunk)
                    or _compact_for_match(quote) in _compact_for_match(chunk)
                ]
                if len(matching_indexes) == 1:
                    index = matching_indexes[0]
                    source_text = _normalise(chunks[index])
                    verified = True
            if not quote or not verified:
                quote = _derive_quote(answer, source_text)
            if not quote:
                return [], "evidence quote could not be derived"
            clean_evidence.append(
                {
                    "chunk_index": index,
                    "quote": quote,
                    "quote_verified": verified,
                }
            )
        by_type[question_type] = {
            "question_type": question_type,
            "question": question,
            "answer": answer,
            "evidence": clean_evidence,
            "difficulty": str(item.get("difficulty") or "medium").strip().casefold(),
        }
    return [by_type[item_type] for item_type in QUESTION_TYPES], None


def _call_deepseek(
    client: httpx.Client,
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_attempts: int = 4,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你只输出合法 JSON，不要输出 Markdown。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 2_400,
        "thinking": {"type": "disabled"},
        "response_format": {"type": "json_object"},
    }
    for attempt in range(max_attempts):
        try:
            response = client.post(url, headers=headers, json=payload)
        except httpx.RequestError as exc:
            if attempt == max_attempts - 1:
                raise RuntimeError("DeepSeek request failed") from exc
            time.sleep(min(2**attempt, 8))
            continue
        if 200 <= response.status_code < 300:
            try:
                decoded = response.json()
            except ValueError as exc:
                raise RuntimeError("DeepSeek returned invalid JSON") from exc
            if not isinstance(decoded, Mapping):
                raise RuntimeError("DeepSeek response must be an object")
            return _parse_content(decoded)
        if response.status_code not in {408, 409, 425, 429} and response.status_code < 500:
            raise RuntimeError(f"DeepSeek returned HTTP {response.status_code}")
        if attempt == max_attempts - 1:
            raise RuntimeError(f"DeepSeek returned HTTP {response.status_code}")
        time.sleep(min(2**attempt, 8))
    raise RuntimeError("DeepSeek request failed")


def _annotation_row(
    paper: Mapping[str, Any],
    *,
    item: Mapping[str, Any],
    question_index: int,
    chunks: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    paper_id = str(paper.get("paper_id") or paper.get("document_id"))
    document_id = str(paper.get("document_id") or paper_id)
    query_id = f"zhpaper-rag30:{paper_id}:q{question_index}"
    evidence_chunk_ids = [
        f"{document_id}::chunk-{int(ref['chunk_index']):04d}" for ref in item["evidence"]
    ]
    query = {
        "query_id": query_id,
        "query": item["question"],
        "language": "zh",
        "paper_id": paper_id,
        "document_id": document_id,
        "source_dataset": "chinese-ai-oa-jos-v2",
        "annotation_status": "silver",
        "annotation_method": "deepseek_auto_with_local_quote_validation",
        "question_type": item["question_type"],
        "difficulty": item["difficulty"],
        "answer": item["answer"],
        "evidence_chunk_ids": evidence_chunk_ids,
        "evidence_quotes": [ref["quote"] for ref in item["evidence"]],
        "evidence_quote_verified": [
            bool(ref.get("quote_verified", False)) for ref in item["evidence"]
        ],
        "relevant_document_ids": [document_id],
    }
    qrels = [
        {
            "query_id": query_id,
            "document_id": chunk_id,
            "relevance": 2,
            "paper_id": paper_id,
            "source_dataset": "chinese-ai-oa-jos-v2",
        }
        for chunk_id in dict.fromkeys(evidence_chunk_ids)
    ]
    return query, qrels


def build_dataset(
    *,
    source: Path,
    output: Path,
    seed: int,
    limit: int,
    confirm_external_calls: bool,
) -> dict[str, Any]:
    if not confirm_external_calls:
        raise SystemExit("This command makes billable DeepSeek calls; pass --confirm-external-calls")
    settings = Settings()
    if settings.deepseek_api_key is None or not settings.deepseek_api_key.get_secret_value().strip():
        raise RuntimeError("TASKFORGE_DEEPSEEK_API_KEY is not configured")
    if settings.deepseek_model is None or not settings.deepseek_model.strip():
        raise RuntimeError("TASKFORGE_DEEPSEEK_MODEL is not configured")
    papers = _select_papers(_read_jsonl_gz(source), seed=seed, limit=limit)
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "annotations.partial.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        with partial_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    completed[str(row["paper_id"])] = row
    client = httpx.Client(timeout=httpx.Timeout(150.0))
    try:
        with partial_path.open("a", encoding="utf-8", newline="\n") as partial:
            for position, paper in enumerate(papers, start=1):
                paper_id = str(paper.get("paper_id") or paper.get("document_id"))
                if paper_id in completed:
                    print(f"[{position}/{limit}] skip {paper_id} (already annotated)")
                    continue
                chunks = _chunk_text(str(paper.get("text", "")))
                prompt = _prompt(str(paper.get("title", "")), str(paper.get("abstract", "")), chunks)
                valid_items: list[dict[str, Any]] = []
                error = ""
                for retry in range(3):
                    try:
                        raw = _call_deepseek(
                            client,
                            url=settings.deepseek_base_url.rstrip("/") + "/chat/completions",
                            api_key=settings.deepseek_api_key.get_secret_value(),
                            model=settings.deepseek_model,
                            prompt=prompt,
                        )
                        valid_items, error = _validate_items(raw, chunks)
                        if valid_items:
                            break
                    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
                        error = str(exc)
                    prompt = _prompt(
                        str(paper.get("title", "")),
                        str(paper.get("abstract", "")),
                        chunks,
                        correction=error or "输出未通过校验",
                    )
                if not valid_items:
                    raise RuntimeError(f"{paper_id} annotation failed after retries: {error}")
                row = {
                    "paper_id": paper_id,
                    "document_id": str(paper.get("document_id") or paper_id),
                    "title": str(paper.get("title", "")),
                    "source_dataset": "chinese-ai-oa-jos-v2",
                    "annotation_status": "silver",
                    "chunk_count": len(chunks),
                    "items": valid_items,
                }
                partial.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                partial.flush()
                completed[paper_id] = row
                print(f"[{position}/{limit}] annotated {paper_id}")
    finally:
        client.close()

    query_rows: list[dict[str, Any]] = []
    chunk_qrels: list[dict[str, Any]] = []
    paper_qrels: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    selected_by_id = {str(row.get("paper_id") or row.get("document_id")): row for row in papers}
    for paper_id in sorted(completed):
        paper = selected_by_id[paper_id]
        row = completed[paper_id]
        chunks = _chunk_text(str(paper.get("text", "")))
        document_id = str(paper.get("document_id") or paper_id)
        for index, chunk in enumerate(chunks):
            chunk_rows.append(
                {
                    "chunk_id": f"{document_id}::chunk-{index:04d}",
                    "document_id": document_id,
                    "paper_id": paper_id,
                    "title": str(paper.get("title", "")),
                    "language": "zh",
                    "text": chunk,
                    "chunk_index": index,
                    "source_dataset": "chinese-ai-oa-jos-v2",
                }
            )
        for question_index, item in enumerate(row["items"], start=1):
            query, qrels = _annotation_row(
                paper,
                item=item,
                question_index=question_index,
                chunks=chunks,
            )
            query_rows.append(query)
            chunk_qrels.extend(qrels)
            paper_qrels.append(
                {
                    "query_id": query["query_id"],
                    "document_id": document_id,
                    "relevance": 1,
                    "paper_id": paper_id,
                    "source_dataset": "chinese-ai-oa-jos-v2",
                }
            )
    query_rows.sort(key=lambda row: row["query_id"])
    chunk_qrels.sort(key=lambda row: (row["query_id"], row["document_id"]))
    paper_qrels.sort(key=lambda row: row["query_id"])
    chunk_rows.sort(key=lambda row: row["chunk_id"])
    _write_jsonl(output / "queries.jsonl", query_rows)
    _write_jsonl(output / "qrels.jsonl", chunk_qrels)
    _write_jsonl(output / "paper_qrels.jsonl", paper_qrels)
    _write_jsonl_gz(output / "chunks.jsonl.gz", chunk_rows)
    (output / "annotations.partial.jsonl").replace(output / "annotations.jsonl")
    manifest = {
        "schema_version": "zhpaper_rag_annotation.v1",
        "dataset_id": "chinese-paper-rag-30-v1",
        "source_dataset": "chinese-ai-oa-jos-v2",
        "source_path": str(source),
        "selection_seed": seed,
        "requested_papers": limit,
        "annotated_papers": len(completed),
        "queries": len(query_rows),
        "chunk_qrels": len(chunk_qrels),
        "paper_qrels": len(paper_qrels),
        "chunks": len(chunk_rows),
        "annotation_status": "silver",
        "annotation_method": "DeepSeek generation plus deterministic evidence-quote validation",
        "human_reviewed": False,
        "limitations": [
            "Questions and answers are automatically generated and require human audit for gold-standard use.",
            "Evidence is bound to deterministic 1800-character chunks; page coordinates are not available in this corpus export.",
            "qrels.jsonl is chunk-level; paper_qrels.jsonl is document-level.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--confirm-external-calls", action="store_true")
    args = parser.parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    manifest = build_dataset(
        source=args.source,
        output=args.output,
        seed=args.seed,
        limit=args.limit,
        confirm_external_calls=args.confirm_external_calls,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
