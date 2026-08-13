"""Run the enterprise-review single-Agent versus multi-Agent benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.config import Settings
from taskforge.openai_provider import (
    OpenAIChatCompletionsProvider,
    OpenAIResponsesProvider,
)
from taskforge.review_ab import (
    ArmExecution,
    build_report,
    load_review_benchmark,
    multi_business_e2e_passed,
    run_multi_arm,
    run_single_arm,
)
from taskforge.verification import SQLiteVerificationStore, VerificationRecord


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one bounded single Agent with TaskForge's fixed four-role "
            "enterprise-review DAG on identical evidence."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "eval" / "review_ab_cases.json",
    )
    parser.add_argument("--case", action="append", dest="case_ids", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--arms", choices=["single", "multi", "both"], default="both"
    )
    parser.add_argument(
        "--provider", choices=["deepseek", "openai"], default=None
    )
    parser.add_argument(
        "--confirm-live-call",
        action="store_true",
        help="Required acknowledgement that the run makes billable network calls.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize the dataset without calling a provider.",
    )
    parser.add_argument(
        "--record-business-e2e",
        action="store_true",
        help=(
            "Persist a signed business_e2e record only when every selected multi "
            "run reaches human review with four successful roles and zero safety violations."
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _select_cases(args: argparse.Namespace):
    benchmark = load_review_benchmark(args.dataset)
    selected = benchmark.cases
    if args.case_ids:
        wanted = set(args.case_ids)
        selected = [case for case in selected if case.id in wanted]
        missing = sorted(wanted - {case.id for case in selected})
        if missing:
            raise RuntimeError(f"unknown benchmark case IDs: {missing}")
    if args.limit is not None:
        if args.limit < 1:
            raise RuntimeError("--limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise RuntimeError("no benchmark cases selected")
    return benchmark, selected


def _provider(settings: Settings, requested: str | None):
    name = requested or settings.provider
    if name == "deepseek":
        if settings.deepseek_api_key is None or not settings.deepseek_model:
            raise RuntimeError("DeepSeek key and model are required")
        model = settings.deepseek_model
        provider = OpenAIChatCompletionsProvider(
            api_key=settings.deepseek_api_key.get_secret_value(),
            enabled=True,
            model=model,
            base_url=settings.deepseek_base_url,
            timeout_seconds=max(60.0, settings.deepseek_timeout_seconds),
        )
        return name, model, provider
    if name == "openai":
        if settings.openai_api_key is None or not settings.openai_model:
            raise RuntimeError("OpenAI key and model are required")
        model = settings.openai_model
        provider = OpenAIResponsesProvider(
            api_key=settings.openai_api_key.get_secret_value(),
            enabled=True,
            model=model,
            base_url=settings.openai_base_url,
            timeout_seconds=max(60.0, settings.openai_timeout_seconds),
        )
        return name, model, provider
    raise RuntimeError("live review A/B supports only deepseek or openai")


async def _run(args: argparse.Namespace) -> int:
    benchmark, cases = _select_cases(args)
    category_counts: dict[str, int] = {}
    for case in cases:
        category_counts[case.category] = category_counts.get(case.category, 0) + 1
    if args.validate_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": benchmark.schema_version,
                    "case_count": len(cases),
                    "categories": category_counts,
                    "gold_is_host_only": True,
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if not args.confirm_live_call:
        raise RuntimeError("refusing billable network use without --confirm-live-call")

    settings = Settings(_env_file=REPOSITORY_ROOT / ".env")
    provider_name, model, provider = _provider(settings, args.provider)
    selected_arms = ["single", "multi"] if args.arms == "both" else [args.arms]
    executions: dict[str, dict[str, ArmExecution]] = {}
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPOSITORY_ROOT
        / ".taskforge"
        / "eval-runs"
        / f"review-ab-{provider_name}-{timestamp}.json"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        with TemporaryDirectory(
            prefix="taskforge-review-ab-", ignore_cleanup_errors=True
        ) as temporary:
            root = Path(temporary)
            for index, case in enumerate(cases, start=1):
                executions[case.id] = {}
                print(
                    f"[{index}/{len(cases)}] {case.id}: {', '.join(selected_arms)}",
                    file=sys.stderr,
                    flush=True,
                )
                case_root = root / case.id
                case_root.mkdir(parents=True, exist_ok=True)
                if "single" in selected_arms:
                    executions[case.id]["single"] = await run_single_arm(
                        case,
                        provider=provider,
                        model=model,
                        workdir=case_root / "single",
                    )
                if "multi" in selected_arms:
                    executions[case.id]["multi"] = await run_multi_arm(
                        case,
                        provider=provider,
                        model=model,
                        workdir=case_root / "multi",
                    )
    finally:
        await provider.aclose()

    report = build_report(
        provider=provider_name,
        model=model,
        dataset=str(args.dataset.resolve()),
        cases=cases,
        executions=executions,
    )
    output.write_text(report.to_json() + "\n", encoding="utf-8")

    recorded = False
    multi_executions = [
        item["multi"] for item in executions.values() if "multi" in item
    ]
    if args.record_business_e2e:
        if not multi_executions:
            raise RuntimeError("--record-business-e2e requires the multi arm")
        if not all(multi_business_e2e_passed(item) for item in multi_executions):
            raise RuntimeError(
                "refusing business_e2e record: a multi run did not complete all four roles safely"
            )
        representative_run = multi_executions[-1].run_ids[-1]
        verification = VerificationRecord.signed(
            kind="business_e2e",
            provider=provider_name,
            model=model,
            run_id=representative_run,
            evidence={
                "report_path": str(output),
                "report_sha256": _sha256(output),
                "dataset": str(args.dataset.resolve()),
                "dataset_sha256": _sha256(args.dataset.resolve()),
                "case_ids": [case.id for case in cases],
                "multi_business_e2e_passed": True,
                "quality_summary": report.summary["arms"].get("multi"),
            },
        )
        SQLiteVerificationStore(settings.verification_sqlite_path).save(verification)
        recorded = True

    print(
        json.dumps(
            {
                "provider": provider_name,
                "model": model,
                "case_count": len(cases),
                "arms": selected_arms,
                "summary": report.summary,
                "report": str(output),
                "business_e2e_recorded": recorded,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(_run(args))
    except RuntimeError as exc:
        print(f"review A/B refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
