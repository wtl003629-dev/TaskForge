"""Run TaskForge's configured durable PostgreSQL/SQLite queue worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from taskforge.app import create_app
from taskforge.worker import DurableWorker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Claim and execute checkpointed TaskForge queue jobs."
    )
    parser.add_argument(
        "--owner",
        default=f"{socket.gethostname()}:{os.getpid()}",
        help="Stable worker identity used by the lease CAS.",
    )
    parser.add_argument("--tenant", help="Optionally restrict claims to one tenant.")
    parser.add_argument("--once", action="store_true", help="Process at most one job.")
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    return parser.parse_args()


async def run() -> int:
    args = parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    app = create_app()
    async with app.router.lifespan_context(app):
        container = app.state.container
        worker = DurableWorker(
            owner=args.owner,
            operations=container.operations,
            checkpoints=container.store,
            runtime=container.runtime,
            lease_seconds=container.settings.worker_lease_seconds,
            tenant_id=args.tenant,
        )
        if args.once:
            outcome = await worker.run_once()
            if outcome is None:
                print(json.dumps({"claimed": False}))
                return 0
            print(
                json.dumps(
                    {
                        "claimed": True,
                        "run_id": outcome.job.run_id,
                        "job_status": outcome.job.status.value,
                        "result_status": outcome.job.result_status,
                        "attempt": outcome.job.attempt,
                        "outcome": outcome.outcome,
                        "error": outcome.error,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            return 0 if outcome.error is None else 1

        stop = asyncio.Event()
        try:
            await worker.run_forever(stop, poll_interval=args.poll_seconds)
        except asyncio.CancelledError:
            stop.set()
            raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
