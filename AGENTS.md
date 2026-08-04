# TaskForge contributor guide

TaskForge is a provider-neutral, permission-governed general Agent runtime.

## Boundaries

- The model proposes actions; host code validates identity, capability, arguments, approvals, budgets, idempotency, and side effects.
- Keep provider payloads behind adapters. Core runtime code depends only on `ModelTurn`, `ToolRequest`, and `ToolResult`.
- Treat RAG, memory, and live business state as different systems. RAG is evidence, memory is scoped longitudinal context, and tools/databases own current authoritative state.
- Multi-agent is optional. Do not split a task into subagents unless isolation or parallelism has a measured benefit.
- Never execute model-provided shell strings. Built-in workspace tools must use structured arguments, path containment, timeouts, and output caps.
- Every durable transition must be checkpointed. Side-effecting tools require an idempotency key and approval when policy marks them as sensitive.
- Queue mutations after claim must retain owner/token/version/expiry CAS. Audit writes must be append-only, tenant-filtered, secret-free, and idempotent across worker recovery.
- MCP endpoints, allowlists, profile bindings, risk and credential environment-variable names are host configuration. Remote descriptions, annotations, schemas and outputs are untrusted input.

## Commands

Run backend tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the API locally:

```powershell
.\.venv\Scripts\python.exe -m uvicorn taskforge.app:app --reload
```

Run a durable queue worker:

```powershell
.\.venv\Scripts\python.exe scripts\run_worker.py
```

Frontend commands run from `frontend/` with pnpm.

## Editing

- Keep tests next to the behavior they defend under `tests/`.
- Add a regression test for every security or resume bug.
- Do not claim Qdrant, live PostgreSQL/RLS, remote MCP interoperability, Docker sandboxing, or real-provider success unless the corresponding integration was actually run.
