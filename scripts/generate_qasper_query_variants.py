"""Generate a frozen, split-bound QASPER retrieval-query variant manifest."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings  # noqa: E402
from taskforge.literature.evidence import route_evidence_intent  # noqa: E402
from taskforge.literature.evidence_query_expander import (  # noqa: E402
    OpenAICompatibleEvidenceQueryExpander,
)
from taskforge.literature.rule_query_expander import (  # noqa: E402
    RuleEvidenceQueryExpander,
)
from taskforge.rag_evaluation import load_qasper_dataset  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


async def generate(
    dataset_path: Path,
    split_path: Path,
    output: Path,
    *,
    provider: str,
    model: str | None,
) -> dict[str, object]:
    config = Settings(_env_file=PROJECT_ROOT / ".env")
    if provider == "rule":
        secret = None
        selected_model = "rule-keyword-v1"
        base_url = "local://rule-keyword-v1"
        timeout = 1.0
    elif provider == "deepseek":
        secret = config.deepseek_api_key
        selected_model = model or config.deepseek_model
        base_url = config.deepseek_base_url
        timeout = config.deepseek_timeout_seconds
    else:
        secret = config.openai_api_key
        selected_model = model or config.openai_model
        base_url = config.openai_base_url
        timeout = config.openai_timeout_seconds
    if provider != "rule" and (secret is None or selected_model is None):
        raise ValueError(f"{provider} API key and model are required")

    dataset = load_qasper_dataset(dataset_path)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    case_by_id = {case.case_id: case for case in dataset.cases}
    locked_ids = [str(value) for value in split["case_ids"]]
    missing = [case_id for case_id in locked_ids if case_id not in case_by_id]
    if missing:
        raise ValueError(f"locked cases are missing from QASPER: {missing[:3]}")
    existing: dict[str, dict[str, object]] = {}
    if output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if (
            prior.get("schema_version") != "1.0"
            or prior.get("split_sha256") != _sha256(split_path)
            or prior.get("dataset_sha256") != _sha256(dataset_path)
        ):
            raise ValueError("existing query variant manifest is incompatible")
        existing = {
            str(item["case_id"]): item
            for item in prior.get("variants", [])
            if isinstance(item, dict) and item.get("case_id")
        }
    expander = (
        RuleEvidenceQueryExpander()
        if provider == "rule"
        else OpenAICompatibleEvidenceQueryExpander(
            api_key=secret.get_secret_value(),  # type: ignore[union-attr]
            model=selected_model,  # type: ignore[arg-type]
            base_url=base_url,
            timeout_seconds=timeout,
        )
    )
    try:
        for case_id in locked_ids:
            case = case_by_id[case_id]
            current = existing.get(case_id)
            if current is not None and current.get("query") == case.query:
                continue
            intent = route_evidence_intent(case.query, "general_fact")
            synonym, keyword = await expander.expand(case.query, intent)
            existing[case_id] = {
                "case_id": case_id,
                "query": case.query,
                "intent": intent,
                "synonym_query": synonym,
                "keyword_query": keyword,
            }
            payload: dict[str, object] = {
                "schema_version": "1.0",
                "created_at": datetime.now(UTC).isoformat(),
                "dataset": str(dataset_path),
                "dataset_sha256": _sha256(dataset_path),
                "split": str(split_path),
                "split_sha256": _sha256(split_path),
                "generator": {
                    "provider": provider,
                    "model": selected_model,
                    "temperature": 0,
                },
                "variants": [existing[value] for value in locked_ids if value in existing],
            }
            _write_atomic(output, payload)
    finally:
        await expander.aclose()
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / ".taskforge" / "eval-cache" / "qasper-dev-v0.3.json",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT
        / "eval"
        / "splits"
        / "qasper-dev-clean-holdout-100-v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "eval" / "queries" / "qasper-clean-holdout-100-v2.json",
    )
    parser.add_argument(
        "--provider",
        choices=("rule", "deepseek", "openai"),
        default="rule",
        help="rule is local/no-API; deepseek/openai make billable calls.",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--confirm-live-call",
        action="store_true",
        help="Required acknowledgement that this command makes billable model calls.",
    )
    args = parser.parse_args()
    if args.provider != "rule" and not args.confirm_live_call:
        raise SystemExit("--confirm-live-call is required")
    result = asyncio.run(
        generate(
            args.dataset,
            args.split,
            args.output,
            provider=args.provider,
            model=args.model,
        )
    )
    print(
        json.dumps(
            {"output": str(args.output), "variants": len(result["variants"])},
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
