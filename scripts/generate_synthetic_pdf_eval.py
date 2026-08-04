from __future__ import annotations

import argparse
import json
from pathlib import Path

from taskforge.synthetic_pdf_eval import generate_synthetic_pdfs


ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Generate the self-authored TaskForge PDF eval suite.")
    value.add_argument(
        "--suite",
        type=Path,
        default=ROOT / "eval" / "synthetic_pdf_suite.json",
    )
    value.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / ".taskforge" / "eval-cache" / "synthetic-pdf-v1",
    )
    return value


def main() -> int:
    args = parser().parse_args()
    manifest = generate_synthetic_pdfs(args.suite, args.output_dir)
    print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
