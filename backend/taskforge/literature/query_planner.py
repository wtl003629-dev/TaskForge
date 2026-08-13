"""Bounded, durable query plans for open scholarly discovery."""

from __future__ import annotations

import re

from ..research_protocol import LiteratureRequest, SearchQuery

_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "all",
        "about",
        "address",
        "apply",
        "are",
        "based",
        "can",
        "could",
        "discuss",
        "find",
        "for",
        "from",
        "give",
        "how",
        "in",
        "insights",
        "is",
        "list",
        "looking",
        "me",
        "of",
        "on",
        "or",
        "paper",
        "papers",
        "provide",
        "propose",
        "research",
        "result",
        "results",
        "share",
        "show",
        "some",
        "that",
        "than",
        "the",
        "to",
        "use",
        "using",
        "want",
        "would",
        "what",
        "which",
        "with",
        "work",
        "works",
        "you",
        "know",
    }
)

# A conservative terminology bridge is used only to add an English scholarly
# query for CJK input. It is not presented as general machine translation.
_ZH_ACADEMIC_TERMS: tuple[tuple[str, str], ...] = (
    ("检索增强生成", "retrieval augmented generation"),
    ("参数记忆", "parametric memory"),
    ("非参数记忆", "non-parametric memory"),
    ("开放域问答", "open-domain question answering"),
    ("双编码器", "dual encoder"),
    ("稠密段落检索", "dense passage retrieval"),
    ("推理轨迹", "reasoning traces"),
    ("外部动作", "external actions"),
    ("自主判断检索", "retrieve on demand"),
    ("反思", "self-reflection critique"),
    ("注意力机制", "attention mechanism"),
    ("循环", "recurrence"),
    ("卷积", "convolution"),
    ("参数高效微调", "parameter-efficient fine-tuning"),
    ("低秩矩阵", "low-rank adaptation"),
    ("稀疏检索", "sparse retrieval"),
    ("稠密检索", "dense retrieval"),
    ("late interaction", "late interaction"),
    ("长上下文", "long context"),
    ("输入中间", "middle of the input"),
    ("自然语言监督", "natural language supervision"),
    ("视觉表示", "visual representations"),
    ("图像与文本编码器", "image and text encoders"),
    ("深层视觉网络", "deep visual networks"),
    ("退化问题", "degradation problem"),
    ("残差学习", "residual learning"),
    ("去噪扩散概率模型", "denoising diffusion probabilistic models"),
    ("强化学习", "reinforcement learning"),
    ("自我博弈", "self-play"),
    ("国际象棋", "chess"),
    ("将棋", "shogi"),
    ("围棋", "Go"),
    ("语言模型", "language models"),
    ("论文", "paper"),
)


def _terms(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\w-]+", value.casefold(), re.UNICODE)
        if len(token) > 1 and token not in _STOP
    ]


def english_academic_bridge(value: str) -> str | None:
    if re.search(r"[\u3400-\u9fff]", value) is None:
        return None
    terms = [english for chinese, english in _ZH_ACADEMIC_TERMS if chinese in value]
    return " ".join(dict.fromkeys(terms)) or None


def plan_literature_queries(request: LiteratureRequest) -> list[SearchQuery]:
    """Produce three to six complementary searches without retaining reasoning."""

    filters: dict[str, object] = {}
    if request.year_from is not None:
        filters["year_from"] = request.year_from
    if request.year_to is not None:
        filters["year_to"] = request.year_to
    if request.venues:
        filters["venues"] = list(request.venues)
    if request.authors:
        filters["authors"] = list(request.authors)

    raw: list[tuple[str, str, int]] = [(request.query, "topic", 1)]
    bridge = english_academic_bridge(request.query)
    if bridge:
        raw.append((bridge, "method", 2))
    for index, question in enumerate(request.research_questions[:2], start=2):
        raw.append((question, "method", index))

    core = list(
        dict.fromkeys(
            [*request.required_terms, *_terms(request.query)[:10]]
        )
    )
    if core:
        raw.append((" ".join(core), "method", 4))
        raw.append((f"{' '.join(core[:8])} review survey", "foundational", 5))
        raw.append((f"{' '.join(core[:8])} recent advances", "recent", 6))

    queries: list[SearchQuery] = []
    seen: set[str] = set()
    for text, intent, priority in raw:
        cleaned = " ".join(text.split())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        queries.append(
            SearchQuery(
                text=cleaned,
                intent=intent,  # type: ignore[arg-type]
                priority=priority,
                provider_filters=filters,
            )
        )
        if len(queries) == 6:
            break

    # Very short requests can collapse the variants. Keep at least three
    # searches because provider tokenisation and scholarly terminology differ.
    suffixes = (("literature", "topic"), ("methods", "method"), ("survey", "foundational"))
    for suffix, intent in suffixes:
        if len(queries) >= 3:
            break
        text = f"{request.query} {suffix}"
        if text.casefold() in seen:
            continue
        queries.append(
            SearchQuery(
                text=text,
                intent=intent,  # type: ignore[arg-type]
                priority=len(queries) + 1,
                provider_filters=filters,
            )
        )
        seen.add(text.casefold())
    return queries


__all__ = ["english_academic_bridge", "plan_literature_queries"]
