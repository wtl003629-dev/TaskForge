"""Evaluate a retrieval artifact against pinned TagOp heuristic mappings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.tatqa_mapping_eval import (  # noqa: E402
    TAGOP_TRAIN_SHA256,
    TATQAMappingDiagnosticError,
    evaluate_tagop_mapping_retrieval,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--annotations", type=Path, required=True)
    value.add_argument("--predictions", type=Path, required=True)
    value.add_argument("--stage")
    value.add_argument("--document-k", type=int, default=10)
    value.add_argument("--emitted-unit-k", type=int, default=10)
    value.add_argument("--evidence-hit-k", type=int, default=10)
    value.add_argument("--query-slot-k", type=int, default=10)
    value.add_argument("--output", type=Path)
    value.add_argument(
        "--allow-unpinned-annotations",
        action="store_true",
        help="Disable the pinned annotation SHA check for local fixture development.",
    )
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = evaluate_tagop_mapping_retrieval(
            args.annotations,
            args.predictions,
            stage=args.stage,
            document_k=args.document_k,
            emitted_unit_k=args.emitted_unit_k,
            evidence_hit_k=args.evidence_hit_k,
            query_slot_k=args.query_slot_k,
            expected_annotation_sha256=(
                None if args.allow_unpinned_annotations else TAGOP_TRAIN_SHA256
            ),
        )
        payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        sys.stdout.write(payload)
        return 0
    except (OSError, json.JSONDecodeError, TATQAMappingDiagnosticError) as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}, ensure_ascii=False) + "\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
