"""Opt-in real DeepSeek Chat Completions smoke test for TaskForge.

This script is intentionally excluded from the offline test suite.  It makes
billable network requests only when the caller supplies both credentials and
the explicit ``--confirm-live-call`` flag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.builtins import create_tool_registry
from taskforge.checkpoints import SQLiteCheckpointStore
from taskforge.config import Settings
from taskforge.context import ContextAssembler
from taskforge.domain import AgentProfile, RunStatus, Task
from taskforge.knowledge import InMemoryKnowledgeStore
from taskforge.memory import InMemoryMemoryStore
from taskforge.openai_provider import OpenAIChatCompletionsProvider
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy
from taskforge.verification import SQLiteVerificationStore, VerificationRecord


def _required_setting(name: str, fallback: str) -> str:
    """Resolve from the process environment, then fall back to ``.env``."""

    value = os.environ.get(name, "").strip()
    if value:
        return value
    if fallback.strip():
        return fallback
    raise RuntimeError(f"{name} is required for the live smoke test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make a real, potentially billable DeepSeek call through "
            "TaskForge's native function-calling AgentRuntime."
        )
    )
    parser.add_argument(
        "--confirm-live-call",
        action="store_true",
        help="Required acknowledgement that this performs billable network calls.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help=(
            "After a successful smoke, persist a signed live_smoke verification "
            "record so the review API can disclose live_smoke_verified=True."
        ),
    )
    return parser.parse_args()


async def run_live_smoke(*, confirmed: bool, record: bool = False) -> dict[str, object]:
    if not confirmed:
        raise RuntimeError("refusing live API use without --confirm-live-call")
    settings = Settings(_env_file=REPOSITORY_ROOT / ".env")
    api_key = _required_setting(
        "TASKFORGE_DEEPSEEK_API_KEY",
        settings.deepseek_api_key.get_secret_value()
        if settings.deepseek_api_key is not None
        else "",
    )
    model = _required_setting(
        "TASKFORGE_DEEPSEEK_MODEL", settings.deepseek_model or ""
    )
    base_url = os.environ.get(
        "TASKFORGE_DEEPSEEK_BASE_URL", settings.deepseek_base_url
    ).strip()

    knowledge = InMemoryKnowledgeStore()
    memory = InMemoryMemoryStore()
    with tempfile.TemporaryDirectory(
        prefix="taskforge-deepseek-smoke-",
        # Windows may briefly hold SQLite file handles during teardown; a
        # leftover temp dir must not mask the real smoke result or error.
        ignore_cleanup_errors=True,
    ) as temp:
        root = Path(temp)
        registry = create_tool_registry(
            workspace_root=root,
            artifact_root=root / "artifacts",
            knowledge_store=knowledge,
            memory_store=memory,
        )
        provider = OpenAIChatCompletionsProvider(
            api_key=api_key,
            enabled=True,
            model=model,
            base_url=base_url,
            timeout_seconds=60,
        )
        runtime = AgentRuntime(
            provider=provider,
            registry=registry,
            policy=CapabilityPolicy(registry),
            checkpoint=SQLiteCheckpointStore(root / "checkpoint.sqlite3"),
            context=ContextAssembler(knowledge, memory),
        )
        task = Task(
            tenant_id="live-smoke-tenant",
            user_id="live-smoke-user",
            goal=(
                "Call the calculator exactly once with the expression (6 * 7) + 1. "
                "Then report the host tool receipt and the numeric result."
            ),
        )
        profile = AgentProfile(
            id="live-calculator-agent",
            name="Live calculator smoke Agent",
            instructions=(
                "You must use the calculator tool exactly once before answering. "
                "Do not estimate the result yourself."
            ),
            model=model,
            allowed_tools=["calculator"],
            max_steps=4,
        )
        try:
            state = await runtime.run(task, profile)
        finally:
            await provider.aclose()

    calculator_receipts = [
        receipt
        for receipt in state.receipts.values()
        if receipt.metadata.get("tool") == "calculator" and receipt.ok
    ]
    passed = (
        state.status is RunStatus.COMPLETED
        and len(calculator_receipts) == 1
        and calculator_receipts[0].output == {"value": 43}
    )
    report: dict[str, object] = {
        "test_mode": "live_deepseek_chat_completions",
        "real_network_call": True,
        "model": model,
        "run_id": state.run_id,
        "status": state.status.value,
        "calculator_receipt_count": len(calculator_receipts),
        "calculator_output": (
            calculator_receipts[0].output if calculator_receipts else None
        ),
        "final_answer": state.final_answer,
        "passed": passed,
    }
    if record and passed:
        store = SQLiteVerificationStore(settings.verification_sqlite_path)
        store.save(
            VerificationRecord.signed(
                kind="live_smoke",
                provider="deepseek",
                model=model,
                run_id=state.run_id,
                evidence=report,
            )
        )
        report["recorded"] = True
    return report


async def _main() -> int:
    args = parse_args()
    try:
        report = await run_live_smoke(
            confirmed=args.confirm_live_call, record=args.record
        )
    except RuntimeError as exc:
        print(f"live smoke refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
