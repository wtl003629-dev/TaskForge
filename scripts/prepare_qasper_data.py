from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.qasper_data import prepare_qasper_data  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Convert pinned QASPER train/validation Parquet files to v0.3 JSON."
    )
    cache = REPOSITORY_ROOT / ".taskforge" / "eval-cache"
    value.add_argument(
        "--train-parquet",
        type=Path,
        default=cache / "qasper-train-v0.3.parquet",
    )
    value.add_argument(
        "--validation-parquet",
        type=Path,
        default=cache / "qasper-validation-v0.3.parquet",
    )
    value.add_argument(
        "--train-json",
        type=Path,
        default=cache / "qasper-train-v0.3.json",
    )
    value.add_argument(
        "--validation-json",
        type=Path,
        default=cache / "qasper-validation-v0.3.json",
    )
    value.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "eval" / "qasper-v0.3-data-manifest.json",
    )
    return value


def _inside_repository(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise ValueError("QASPER preparation paths must stay inside the repository") from exc
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = prepare_qasper_data(
            train_parquet=_inside_repository(args.train_parquet),
            validation_parquet=_inside_repository(args.validation_parquet),
            train_json=_inside_repository(args.train_json),
            validation_json=_inside_repository(args.validation_json),
            manifest_path=_inside_repository(args.manifest),
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
