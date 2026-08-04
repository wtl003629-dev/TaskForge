"""Opt-in real OpenAI Responses API smoke test for TaskForge.

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
from taskforge.context import ContextAssembler
from taskforge.domain import AgentProfile, RunStatus, Task
from taskforge.knowledge import InMemoryKnowledgeStore
from taskforge.memory import InMemoryMemoryStore
from taskforge.openai_provider import OpenAIResponsesProvider
from taskforge.runtime import AgentRuntime
from taskforge.tooling import CapabilityPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Make a real, potentially billable OpenAI call through TaskForge's "
            "native function-calling AgentRuntime."
        )
    )
    parser.add_argument(
        "--confirm-live-call",
        action="store_true",
        help="Required acknowledgement that this performs billable network calls.",
    )
    return parser.parse_args()


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required for the live smoke test")
    return value


async def run_live_smoke(*, confirmed: bool) -> dict[str, object]:
    if not confirmed:
        raise RuntimeError("refusing live API use without --confirm-live-call")
    api_key = _required_environment("TASKFORGE_OPENAI_API_KEY")
    model = _required_environment("TASKFORGE_OPENAI_MODEL")
    base_url = os.environ.get(
        "TASKFORGE_OPENAI_BASE_URL", "https://api.openai.com/v1"
    ).strip()

    knowledge = InMemoryKnowledgeStore()
    memory = InMemoryMemoryStore()
    with tempfile.TemporaryDirectory(prefix="taskforge-live-smoke-") as temp:
        root = Path(temp)
        registry = create_tool_registry(
            workspace_root=root,
            artifact_root=root / "artifacts",
            knowledge_store=knowledge,
            memory_store=memory,
        )
        provider = OpenAIResponsesProvider(
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
    return {
        "test_mode": "live_openai_responses",
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


async def _main() -> int:
    args = parse_args()
    try:
        report = await run_live_smoke(confirmed=args.confirm_live_call)
    except RuntimeError as exc:
        print(f"live smoke refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
