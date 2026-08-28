from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPTS_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_qasper_direct_upload import _aligned_ranking_metrics  # noqa: E402

from taskforge.rag_evaluation import (  # noqa: E402
    GoldEvidenceSet,
    GoldEvidenceUnit,
    QasperGoldLabels,
)


def test_ndcg_counts_one_gain_for_flat_and_child_alignment_of_one_unit() -> None:
    gold = QasperGoldLabels(
        evidence_sets=[
            GoldEvidenceSet(
                annotation_id="worker",
                units=[
                    GoldEvidenceUnit(
                        unit_id="unit-1",
                        text="shared evidence",
                        alternative_paragraph_ids=["paragraph-1"],
                    )
                ],
            )
        ]
    )
    alignments = {
        "unit-1": SimpleNamespace(
            status="exact",
            aligned_child_spans=[
                SimpleNamespace(child_id="flat-1"),
                SimpleNamespace(child_id="child-1"),
            ],
        )
    }

    mrr, ndcg = _aligned_ranking_metrics(
        gold,
        alignments,
        ["child-1", "unrelated"],
    )

    assert mrr == 1.0
    assert ndcg == 1.0
