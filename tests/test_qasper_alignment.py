from __future__ import annotations

from taskforge.qasper_alignment import (
    AlignmentChunk,
    align_gold_unit,
    align_qasper_gold,
    aligned_recall_at_k,
    alignment_diagnostics,
    alignment_tokens,
    normalize_alignment_text,
    paragraph_recall_at_k,
)
from taskforge.rag_evaluation import (
    GoldEvidenceSet,
    GoldEvidenceUnit,
    QasperGoldLabels,
)


def _unit(name: str, text: str, *locations: str) -> GoldEvidenceUnit:
    return GoldEvidenceUnit(
        unit_id=name,
        text=text,
        alternative_paragraph_ids=list(locations),
    )


def test_paragraph_recall_uses_best_legal_annotation_set() -> None:
    labels = QasperGoldLabels(
        evidence_sets=[
            GoldEvidenceSet(
                annotation_id="worker-a",
                units=[
                    _unit("a", "Evidence A", "p-a"),
                    _unit("b", "Evidence B", "p-b"),
                ],
            ),
            GoldEvidenceSet(
                annotation_id="worker-b",
                units=[_unit("c", "Alternative evidence", "p-c")],
            ),
        ]
    )

    result = paragraph_recall_at_k(labels, ["p-c"], 1)

    assert result.recall == 1.0
    assert result.selected_annotation_id == "worker-b"
    assert result.hit_unit_ids == ["c"]


def test_duplicate_paragraph_locations_remain_one_denominator_unit() -> None:
    labels = QasperGoldLabels(
        evidence_sets=[
            GoldEvidenceSet(
                annotation_id="worker",
                units=[
                    _unit(
                        "duplicate",
                        "Repeated evidence",
                        "paragraph-1",
                        "paragraph-9",
                    )
                ],
            )
        ]
    )

    result = paragraph_recall_at_k(labels, ["paragraph-9"], 1)

    assert result.recall == 1.0
    assert result.total_units == 1


def test_alignment_normalizes_ligatures_whitespace_and_line_hyphenation() -> None:
    assert normalize_alignment_text("Ef\ufb01cient retriev-\n al") == (
        "efficient retrieval"
    )
    assert normalize_alignment_text(
        "FLOAT SELECTED: Table 5: Results"
    ) == "table: results"


def test_alignment_normalizes_corpus_reference_placeholders() -> None:
    gold = (
        "Table TABREF44 supports the result BIBREF18. "
        "See Fig. FIGREF16 and Section SECREF17."
    )
    pdf = (
        "Table 4 supports the result (Smith et al., 2018). "
        "See Fig. 6 and Section 7."
    )

    assert alignment_tokens(gold) == alignment_tokens(pdf)


def test_alignment_normalizes_pdf_markup_and_source_formula_placeholders() -> None:
    gold = (
        "For side INLINEFORM0, non-targeted posts use German "
        "$\\rightarrow$ English BIBREF27."
    )
    pdf = (
        "For side X,<sup>12</sup> nontargeted posts use German → English "
        "(Smith et al., 2019)."
    )

    assert alignment_tokens(gold) == (
        "for",
        "side",
        "nontargeted",
        "posts",
        "use",
        "german",
        "english",
    )
    assert alignment_tokens(pdf) == (
        "for",
        "side",
        "x",
        "nontargeted",
        "posts",
        "use",
        "german",
        "english",
    )


def test_short_gold_can_align_across_an_inserted_citation() -> None:
    unit = _unit(
        "short",
        "IWSLT 2017 German to English 200K sentence pairs",
        "official-paragraph",
    )
    chunk = AlignmentChunk(
        child_id="datasets",
        text=(
            "IWSLT 2017 German to English (Cettolo et al., 2017): "
            "200K sentence pairs."
        ),
    )

    aligned = align_gold_unit(unit, [chunk])

    assert aligned.status in {"exact", "fuzzy"}
    assert aligned.normalized_coverage == 1.0


def test_split_gold_paragraph_requires_enough_retrieved_child_coverage() -> None:
    text = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda mu nu xi"
    )
    labels = QasperGoldLabels(
        evidence_sets=[
            GoldEvidenceSet(
                annotation_id="worker",
                units=[_unit("gold", text, "official-paragraph")],
            )
        ]
    )
    chunks = [
        AlignmentChunk(
            child_id="first",
            text="alpha beta gamma delta epsilon zeta eta",
            order=0,
        ),
        AlignmentChunk(
            child_id="second",
            text="theta iota kappa lambda mu nu xi",
            order=1,
        ),
    ]

    alignments = align_qasper_gold(labels, chunks)

    assert alignments["gold"].status == "fuzzy"
    assert aligned_recall_at_k(labels, alignments, ["first"], 1).recall == 0.0
    assert aligned_recall_at_k(
        labels, alignments, ["first", "second"], 2
    ).recall == 1.0
    diagnostics = alignment_diagnostics(alignments)
    assert diagnostics.fuzzy_units == 1
    assert diagnostics.alignment_coverage == 1.0


def test_partial_ambiguous_alignment_never_counts_as_a_hit() -> None:
    labels = QasperGoldLabels(
        evidence_sets=[
            GoldEvidenceSet(
                annotation_id="worker",
                units=[
                    _unit(
                        "gold",
                        "alpha beta gamma delta epsilon zeta eta theta iota kappa",
                        "official-paragraph",
                    )
                ],
            )
        ]
    )
    chunks = [
        AlignmentChunk(
            child_id="partial",
            text="alpha beta gamma delta epsilon zeta unrelated unrelated",
        )
    ]

    alignments = align_qasper_gold(labels, chunks)

    assert alignments["gold"].status == "ambiguous"
    assert aligned_recall_at_k(labels, alignments, ["partial"], 1).recall == 0.0


def test_alignment_ignores_child_context_outside_gold_local_window() -> None:
    unit = GoldEvidenceUnit(
        unit_id="gold-context",
        text=(
            "The final tweets were forwarded to three annotators. Approximately "
            "60 hours were spent tagging the tweets for humor."
        ),
        alternative_paragraph_ids=["paragraph-1"],
    )
    chunks = [
        AlignmentChunk(
            child_id="child-context",
            order=0,
            text=(
                ("unrelated introduction context " * 50)
                + "The final tweets were forwarded to three annotators. "
                + "Approximately footnote URL 60 hours were spent tagging the "
                + "tweets for humor. "
                + ("unrelated conclusion context " * 50)
            ),
        )
    ]

    aligned = align_gold_unit(unit, chunks)

    assert aligned.status == "fuzzy"
    assert aligned.normalized_coverage >= 0.8
