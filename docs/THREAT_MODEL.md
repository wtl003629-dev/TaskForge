# TaskForge threat model

## Trust boundary

The model, retrieved text, user task text, provider responses, and remote tool
results are untrusted. Python handlers, identity supplied by the host API
layer, configured workspace roots, policy decisions, and durable receipts
belong to the host boundary.

The current demo accepts tenant/user identity from headers; those headers are
not authenticated and are not a production identity boundary.

The invariant is simple: a `ToolRequest` is a proposal, never authority.

## Verification labels

“Implemented control” below means host code plus the stated automated/local
tests exist. It does not by itself mean the control is deployed behind
production identity or verified against a real external service. PostgreSQL is
the default durable path; SQLite remains an explicit compatibility path and
PostgreSQL has fake DB-API coverage
plus an opt-in live test, while the local PostgreSQL service/RLS gate is still
pending. Neo4j has fake-driver coverage only, semantic adapters use
injected/mock tests, and remote MCP and real-model planning have no live
success claim. These boundaries matter because a unit test can validate
fail-closed logic but cannot validate network, service configuration,
production RLS, model behavior or operational isolation.

## Implemented controls

| Threat | Implemented control | Remaining boundary |
|---|---|---|
| Prompt injection in RAG | Retrieved context is labelled as untrusted evidence; capabilities are not derived from retrieved text. | Content-level answer quality still needs adversarial evaluation. |
| Model-generated shell injection | No generic shell tool exists. Grep, read, and arithmetic use structured Python implementations. | A future code-execution tool needs a container/VM sandbox and separate policy. |
| Path traversal and secret reads | Server binds the root; absolute paths, `..`, symlinks/reparse points, credential filenames, binaries, VCS and build directories are rejected. | File classification is a denylist plus size/type checks, not DLP. |
| Cross-tenant RAG or memory leakage | Persistent stores prefilter by tenant/validity, then enforce ACL or scope. API ownership is checked before run/job/audit reads and approval. | A production deployment still needs real authentication and per-tenant encryption policy. |
| Cross-role or cross-case capability escape | Orchestration access binds tenant, owner, conversation and optional role allowlist; RoleRun, handoff, fact and private-memory APIs re-check the bound role. Review case conversation IDs are host-derived from case IDs. | Header identity is still a local demonstration boundary, not authentication. |
| Model promotes itself to verifier or decision maker | Structured role receipts create only `proposed` facts. Verification consumes a one-use host receipt, and final case decisions accept only a human actor derived from the request principal plus revision CAS. | A production reviewer workflow needs RBAC, separation of duties and re-authentication. |
| Upstream role prompt injection | Dependency summaries, proposed facts, handoffs and private memory are labelled `model_untrusted`; only receipt-backed facts are `host_verified`, and the entire layered context has a hard budget. | Answer-level susceptibility still requires live-model adversarial evaluation. |
| Side-effect replay | Side-effect tools require idempotency keys; side-effecting MCP schemas without a required string key are rejected before mounting. Call IDs and canonical request fingerprints are checkpointed, and reuse with changed arguments fails closed. | Cross-service idempotency also requires the downstream service to honor the key. |
| Confused-deputy approval | Approval is bound to the exact pending run and `call_id`; arguments remain durable, and current profile/policy are re-evaluated before execution so revocation wins. | Production UI should show diffs and require re-authentication for high-risk actions. |
| Unbounded loops or output | Model steps, per-turn tool fanout, tool time, arguments, file size, match count, and outputs are bounded. | Provider token and monetary budgets should be enforced by a deployment adapter. |
| Checkpoint tampering | SQLite payloads and persistent context JSON are revalidated; corrupt context rows fail closed individually. | SQLite has no application-level signature or encryption. |
| Worker double execution | Atomic claim plus owner/token/version/expiry CAS, heartbeat, explicit retryable provider taxonomy, bounded retry, final-lease checkpoint reconciliation and receipt idempotency. | Checkpoint/queue/downstream writes are not one transaction; downstream systems must honor idempotency. Provider calls are at-least-once across ambiguous transport failures and may be charged twice. |
| Duplicate multi-role execution | RoleRun uses an atomic database execution claim, heartbeat and token fence before provider/schema/tool dispatch; only the active token can write durable state. Missing/waiting/terminal checkpoints are reconciled before new scheduling. | A provider request already in flight when a process stalls cannot be cancelled reliably and may be billed twice; downstream side effects still require business idempotency. |
| Audit leakage or mutation | Credential-like keys/values are rejected, failures are bounded/redacted, tenant/run filters are mandatory, and DB triggers reject UPDATE/DELETE. | Central log access, retention and encrypted backups remain deployment responsibilities. |
| Arbitrary MCP/SSRF | MCP is off by default; endpoint and allowlist are host config, DNS/IP receives a preflight check, redirects/private ranges are denied by default, schemas/results are bounded, and local policy remains authoritative. | Preflight resolution and httpx connection resolution are not IP-pinned, leaving DNS-rebinding TOCTOU; production needs egress enforcement. No live conformance suite; JSON-only revision 2025-11-25, not current 2026-07-28. |

## Deterministic safety properties

1. A tool absent from `AgentProfile.allowed_tools` cannot execute.
2. A side-effecting call without an idempotency key cannot reach its handler.
3. Write, external and destructive risks pause for approval.
4. A repeated call ID or idempotency key with different arguments terminates as
   a receipt-integrity error.
5. Tenant and scope checks are performed by host code, never inferred by the
   model.
6. Ordinary tool errors become observations so the model may recover within the
   step budget; policy and receipt-integrity errors fail closed.
7. Human approval cannot resurrect a capability removed while the request was
   pending; current profile and host policy are checked again before execution.
8. A model-produced role result cannot become a verified fact or a final case
   decision without a separately authenticated host action.
9. A stale RoleRun worker cannot dispatch another tool or persist an outcome
   after its lease token is replaced.

## Production gates not claimed by phase two

- authenticated identity and role administration;
- live PostgreSQL/psycopg execution, ordered migration execution, RLS verification and encrypted backups (the app wiring, migrations, maintenance commands and opt-in live tests are supplied, but the local Docker engine is unavailable and the gate is not yet passed);
- container or VM isolation for code execution;
- certificate-pinned MCP and infrastructure-level egress controls;
- production semantic embedding/reranking services and durable remote Qdrant indexing (local Qdrant and hash-vector degradation tests exist);
- live Neo4j connectivity and graph-quality gate results (the optional adapter is fake-driver tested only);
- exactly-once downstream effects; PostgreSQL-backed queue lease/CAS is implemented but still needs the live concurrency/recovery gate;
- verified Docker image builds and container-runtime security controls (files are supplied, but Docker is unavailable on the current development machine);
- red-team suites for prompt injection, data exfiltration and denial of service.
- live-provider planning, tool-use, citation and adversarial quality evaluation after credentials are supplied.
