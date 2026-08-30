-- TaskForge runtime schema for PostgreSQL 16 + pgvector.
-- Apply as migration_admin after 001_roles.sh. The application role must not
-- receive CREATE, ALTER, TRUNCATE, or schema ownership privileges.

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS core;
-- The existing PostgresContextRepository uses `taskforge` for its context
-- tables. Keep that name as the compatibility context schema.
CREATE SCHEMA IF NOT EXISTS taskforge;
CREATE SCHEMA IF NOT EXISTS operations;
CREATE SCHEMA IF NOT EXISTS orchestration;
CREATE SCHEMA IF NOT EXISTS review;
CREATE SCHEMA IF NOT EXISTS verification;
CREATE SCHEMA IF NOT EXISTS literature;
CREATE SCHEMA IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS core.tasks (
    tenant_id   text NOT NULL,
    task_id     text NOT NULL,
    task_json   jsonb NOT NULL CHECK (jsonb_typeof(task_json) = 'object'),
    created_at  timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, task_id)
);

CREATE TABLE IF NOT EXISTS core.profiles (
    tenant_id   text NOT NULL,
    profile_id  text NOT NULL,
    profile_json jsonb NOT NULL CHECK (jsonb_typeof(profile_json) = 'object'),
    updated_at  timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, profile_id)
);

CREATE TABLE IF NOT EXISTS core.runs (
    tenant_id   text NOT NULL,
    run_id      text NOT NULL,
    task_id     text NOT NULL,
    profile_id  text NOT NULL,
    state_json  jsonb NOT NULL,
    version     integer NOT NULL CHECK (version >= 1),
    updated_at  timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, run_id),
    FOREIGN KEY (tenant_id, task_id) REFERENCES core.tasks(tenant_id, task_id),
    FOREIGN KEY (tenant_id, profile_id) REFERENCES core.profiles(tenant_id, profile_id),
    CHECK (
        jsonb_typeof(state_json) = 'object'
        AND state_json->>'status' IN (
            'pending', 'running', 'waiting_approval', 'completed', 'failed', 'step_limit'
        )
    )
);
CREATE INDEX IF NOT EXISTS runs_task_idx ON core.runs (tenant_id, task_id);
CREATE INDEX IF NOT EXISTS runs_updated_idx ON core.runs (tenant_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS taskforge.knowledge_chunks (
    tenant_id       text NOT NULL,
    chunk_id        text NOT NULL,
    text_content    text NOT NULL,
    source_uri      text NOT NULL,
    document_id     text,
    version         text NOT NULL,
    version_order   integer NOT NULL CHECK (version_order >= 0),
    acl_json        jsonb NOT NULL,
    valid_from      timestamptz,
    valid_until     timestamptz,
    created_at      timestamptz NOT NULL,
    metadata_json   jsonb NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(metadata_json) = 'object'),
    PRIMARY KEY (tenant_id, chunk_id),
    CHECK (jsonb_typeof(acl_json) = 'array'),
    CHECK (valid_from IS NULL OR valid_until IS NULL OR valid_from < valid_until)
);
CREATE INDEX IF NOT EXISTS knowledge_tenant_idx
    ON taskforge.knowledge_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS knowledge_version_idx
    ON taskforge.knowledge_chunks (tenant_id, document_id, version_order DESC, version);
CREATE INDEX IF NOT EXISTS knowledge_acl_idx
    ON taskforge.knowledge_chunks USING gin (acl_json jsonb_ops);

CREATE TABLE IF NOT EXISTS taskforge.memory_items (
    tenant_id       text NOT NULL,
    memory_id       text NOT NULL,
    content         text NOT NULL,
    scope           text NOT NULL CHECK (scope IN ('tenant', 'org', 'user', 'agent', 'task')),
    scope_id        text NOT NULL,
    provenance_json jsonb NOT NULL CHECK (jsonb_typeof(provenance_json) = 'object'),
    importance      double precision NOT NULL CHECK (importance >= 0.0 AND importance <= 1.0),
    created_at      timestamptz NOT NULL,
    updated_at      timestamptz NOT NULL,
    expires_at      timestamptz,
    tags_json       jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata_json   jsonb NOT NULL DEFAULT '{}'::jsonb
        CHECK (jsonb_typeof(metadata_json) = 'object'),
    PRIMARY KEY (tenant_id, memory_id),
    CHECK (jsonb_typeof(provenance_json) = 'object'),
    CHECK (jsonb_typeof(tags_json) = 'array')
);
CREATE INDEX IF NOT EXISTS memory_expiry_idx
    ON taskforge.memory_items (tenant_id, expires_at);
CREATE INDEX IF NOT EXISTS memory_scope_idx
    ON taskforge.memory_items (tenant_id, scope, scope_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS operations.operation_jobs (
    tenant_id       text NOT NULL,
    run_id          text NOT NULL,
    status          text NOT NULL CHECK (status IN ('queued', 'leased', 'completed', 'dead_letter')),
    priority        integer NOT NULL DEFAULT 0,
    attempt         integer NOT NULL DEFAULT 0 CHECK (attempt >= 0),
    max_attempts    integer NOT NULL CHECK (max_attempts >= 1),
    available_at    timestamptz NOT NULL,
    owner           text,
    lease_token     text,
    lease_version   bigint NOT NULL DEFAULT 0 CHECK (lease_version >= 0),
    lease_expires_at timestamptz,
    result_status   text,
    last_error      text,
    created_at      timestamptz NOT NULL,
    updated_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, run_id),
    FOREIGN KEY (tenant_id, run_id) REFERENCES core.runs(tenant_id, run_id),
    CHECK (
        (status = 'leased' AND owner IS NOT NULL AND lease_token IS NOT NULL AND lease_expires_at IS NOT NULL)
        OR (status <> 'leased' AND owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)
    )
);
CREATE INDEX IF NOT EXISTS operation_claim_idx
    ON operations.operation_jobs (tenant_id, status, available_at, priority DESC);

CREATE TABLE IF NOT EXISTS operations.audit_events (
    sequence        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    tenant_id       text NOT NULL,
    event_id        text NOT NULL UNIQUE,
    run_id          text NOT NULL,
    action          text NOT NULL,
    outcome         text NOT NULL,
    duration_ms     double precision,
    tool            text,
    provider        text,
    input_tokens    integer,
    output_tokens   integer,
    total_tokens    integer,
    cost_usd        double precision,
    safety_violation boolean NOT NULL DEFAULT false,
    metadata_json   jsonb NOT NULL CHECK (jsonb_typeof(metadata_json) = 'object'),
    occurred_at     timestamptz NOT NULL,
    FOREIGN KEY (tenant_id, run_id) REFERENCES core.runs(tenant_id, run_id)
);
CREATE INDEX IF NOT EXISTS audit_events_order_idx
    ON operations.audit_events (tenant_id, occurred_at, sequence);

CREATE TABLE IF NOT EXISTS orchestration.speaker_plans (
    tenant_id               text NOT NULL,
    plan_id                 text NOT NULL,
    owner_user_id           text NOT NULL,
    conversation_id         text NOT NULL,
    client_idempotency_key  text NOT NULL,
    request_hash            text NOT NULL CHECK (request_hash ~ '^[0-9a-f]{64}$'),
    status                  text NOT NULL CHECK (status IN (
        'ready', 'running', 'waiting_approval', 'completed', 'degraded', 'failed', 'cancelled'
    )),
    version                 integer NOT NULL CHECK (version >= 1),
    plan_json               jsonb NOT NULL CHECK (jsonb_typeof(plan_json) = 'object'),
    created_at              timestamptz NOT NULL,
    updated_at              timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, plan_id),
    UNIQUE (tenant_id, owner_user_id, client_idempotency_key)
);

CREATE TABLE IF NOT EXISTS orchestration.role_runs (
    tenant_id       text NOT NULL,
    role_run_id     text NOT NULL,
    run_id          text NOT NULL,
    conversation_id text NOT NULL,
    plan_id         text NOT NULL,
    slot_id         text NOT NULL,
    role_id         text NOT NULL,
    attempt         integer NOT NULL CHECK (attempt >= 1),
    status          text NOT NULL CHECK (status IN (
        'pending', 'queued', 'running', 'waiting_approval', 'succeeded', 'failed', 'cancelled'
    )),
    version         integer NOT NULL CHECK (version >= 1),
    role_run_json   jsonb NOT NULL CHECK (jsonb_typeof(role_run_json) = 'object'),
    created_at      timestamptz NOT NULL,
    updated_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, role_run_id),
    UNIQUE (tenant_id, run_id),
    UNIQUE (tenant_id, plan_id, slot_id, attempt),
    FOREIGN KEY (tenant_id, plan_id) REFERENCES orchestration.speaker_plans(tenant_id, plan_id)
);

CREATE TABLE IF NOT EXISTS orchestration.handoffs (
    tenant_id       text NOT NULL,
    handoff_id      text NOT NULL,
    conversation_id text NOT NULL,
    plan_id         text NOT NULL,
    from_role_run_id text NOT NULL,
    to_slot_id      text NOT NULL,
    payload_hash    text NOT NULL CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
    handoff_json    jsonb NOT NULL CHECK (jsonb_typeof(handoff_json) = 'object'),
    created_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, handoff_id),
    UNIQUE (tenant_id, from_role_run_id, to_slot_id),
    FOREIGN KEY (tenant_id, plan_id) REFERENCES orchestration.speaker_plans(tenant_id, plan_id),
    FOREIGN KEY (tenant_id, from_role_run_id) REFERENCES orchestration.role_runs(tenant_id, role_run_id)
);

CREATE TABLE IF NOT EXISTS orchestration.shared_facts (
    tenant_id       text NOT NULL,
    fact_id         text NOT NULL,
    owner_user_id   text NOT NULL,
    conversation_id text NOT NULL,
    fact_key        text NOT NULL,
    version         integer NOT NULL CHECK (version >= 1),
    status          text NOT NULL CHECK (status IN ('proposed', 'verified')),
    fact_json       jsonb NOT NULL CHECK (jsonb_typeof(fact_json) = 'object'),
    created_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, fact_id),
    UNIQUE (tenant_id, owner_user_id, conversation_id, fact_key, version)
);

CREATE TABLE IF NOT EXISTS orchestration.fact_verification_receipts (
    tenant_id          text NOT NULL,
    receipt_id         text NOT NULL,
    owner_user_id      text NOT NULL,
    conversation_id    text NOT NULL,
    fact_key            text NOT NULL,
    authority           text NOT NULL CHECK (authority IN ('tool', 'user', 'system')),
    value_hash          text NOT NULL CHECK (value_hash ~ '^[0-9a-f]{64}$'),
    evidence_ref        text NOT NULL,
    receipt_json        jsonb NOT NULL CHECK (jsonb_typeof(receipt_json) = 'object'),
    consumed_by_fact_id text,
    created_at          timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, receipt_id)
);

CREATE TABLE IF NOT EXISTS orchestration.private_role_memories (
    tenant_id       text NOT NULL,
    memory_id       text NOT NULL,
    owner_user_id   text NOT NULL,
    conversation_id text NOT NULL,
    role_id         text NOT NULL,
    provenance_key  text NOT NULL,
    content_hash    text NOT NULL CHECK (content_hash ~ '^[0-9a-f]{64}$'),
    memory_json     jsonb NOT NULL CHECK (jsonb_typeof(memory_json) = 'object'),
    expires_at      timestamptz,
    created_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, memory_id),
    UNIQUE (tenant_id, owner_user_id, conversation_id, role_id, provenance_key, content_hash)
);

CREATE TABLE IF NOT EXISTS orchestration.role_run_execution_claims (
    tenant_id       text NOT NULL,
    role_run_id     text NOT NULL,
    owner_user_id   text NOT NULL,
    conversation_id text NOT NULL,
    claim_token     text NOT NULL UNIQUE,
    expires_at      timestamptz NOT NULL,
    claim_json      jsonb NOT NULL CHECK (jsonb_typeof(claim_json) = 'object'),
    PRIMARY KEY (tenant_id, role_run_id),
    FOREIGN KEY (tenant_id, role_run_id) REFERENCES orchestration.role_runs(tenant_id, role_run_id)
);

CREATE TABLE IF NOT EXISTS review.review_cases (
    tenant_id       text NOT NULL,
    case_id         text NOT NULL,
    owner_user_id   text NOT NULL,
    conversation_id text NOT NULL,
    status          text NOT NULL CHECK (status IN ('draft', 'submitted', 'running', 'waiting_human_review', 'approved', 'rejected', 'failed')),
    revision        integer NOT NULL CHECK (revision >= 1),
    case_json       jsonb NOT NULL CHECK (jsonb_typeof(case_json) = 'object'),
    created_at      timestamptz NOT NULL,
    updated_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, case_id)
);

CREATE TABLE IF NOT EXISTS review.review_case_audit_events (
    tenant_id       text NOT NULL,
    event_id        text NOT NULL,
    owner_user_id   text NOT NULL,
    conversation_id text NOT NULL,
    case_id         text NOT NULL,
    revision        integer NOT NULL,
    event_type      text NOT NULL CHECK (event_type IN (
        'case_created', 'draft_updated', 'case_submitted', 'review_started',
        'model_recommendation_recorded', 'case_approved', 'case_rejected', 'case_failed'
    )),
    event_json      jsonb NOT NULL CHECK (jsonb_typeof(event_json) = 'object'),
    created_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, event_id),
    UNIQUE (tenant_id, case_id, revision),
    FOREIGN KEY (tenant_id, case_id) REFERENCES review.review_cases(tenant_id, case_id)
);

CREATE TABLE IF NOT EXISTS review.review_case_commands (
    tenant_id       text NOT NULL,
    owner_user_id   text NOT NULL,
    conversation_id text NOT NULL,
    idempotency_key text NOT NULL,
    command_type    text NOT NULL,
    request_hash    text NOT NULL CHECK (request_hash ~ '^[0-9a-f]{64}$'),
    case_id         text NOT NULL,
    result_revision integer NOT NULL,
    result_case_json jsonb NOT NULL CHECK (jsonb_typeof(result_case_json) = 'object'),
    created_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, owner_user_id, conversation_id, idempotency_key),
    FOREIGN KEY (tenant_id, case_id) REFERENCES review.review_cases(tenant_id, case_id)
);

CREATE TABLE IF NOT EXISTS verification.verification_records (
    tenant_id   text NOT NULL,
    record_id   text NOT NULL,
    record_json jsonb NOT NULL CHECK (jsonb_typeof(record_json) = 'object'),
    produced_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, record_id)
);

CREATE TABLE IF NOT EXISTS literature.literature_requests (
    tenant_id       text NOT NULL,
    request_id      text NOT NULL,
    user_id         text NOT NULL,
    conversation_id text,
    request_json    jsonb NOT NULL CHECK (jsonb_typeof(request_json) = 'object'),
    created_at      timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, request_id)
);

CREATE TABLE IF NOT EXISTS literature.literature_queries (
    tenant_id  text NOT NULL,
    request_id text NOT NULL,
    query_id   text NOT NULL,
    priority   integer NOT NULL,
    query_json jsonb NOT NULL CHECK (jsonb_typeof(query_json) = 'object'),
    created_at timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, request_id, query_id),
    FOREIGN KEY (tenant_id, request_id) REFERENCES literature.literature_requests(tenant_id, request_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS literature.paper_catalog (
    tenant_id          text NOT NULL,
    paper_id           text NOT NULL,
    card_json          jsonb NOT NULL CHECK (jsonb_typeof(card_json) = 'object'),
    verification_status text NOT NULL CHECK (verification_status IN (
        'provider_verified', 'cross_source_verified', 'metadata_partial', 'unverified'
    )),
    full_text_status    text NOT NULL CHECK (full_text_status IN (
        'not_requested', 'available', 'abstract_only', 'ingested', 'failed'
    )),
    updated_at         timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, paper_id)
);

CREATE TABLE IF NOT EXISTS literature.paper_identifiers (
    tenant_id       text NOT NULL,
    identifier_type text NOT NULL,
    identifier_value text NOT NULL,
    paper_id        text NOT NULL,
    PRIMARY KEY (tenant_id, identifier_type, identifier_value),
    FOREIGN KEY (tenant_id, paper_id) REFERENCES literature.paper_catalog(tenant_id, paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS literature.research_scopes (
    tenant_id       text NOT NULL,
    scope_id        text NOT NULL,
    scope_version   integer NOT NULL,
    owner_user_id   text NOT NULL,
    conversation_id text NOT NULL,
    request_id      text NOT NULL,
    status          text NOT NULL CHECK (status IN (
        'draft', 'confirmed', 'ingesting', 'ready', 'expansion_requested', 'closed'
    )),
    scope_json      jsonb NOT NULL CHECK (jsonb_typeof(scope_json) = 'object'),
    created_at      timestamptz NOT NULL,
    confirmed_at    timestamptz,
    PRIMARY KEY (tenant_id, scope_id, scope_version),
    FOREIGN KEY (tenant_id, request_id) REFERENCES literature.literature_requests(tenant_id, request_id)
);

CREATE TABLE IF NOT EXISTS literature.research_scope_papers (
    tenant_id     text NOT NULL,
    scope_id      text NOT NULL,
    scope_version integer NOT NULL,
    paper_id      text NOT NULL,
    selection_status text NOT NULL CHECK (selection_status IN ('selected', 'excluded')),
    PRIMARY KEY (tenant_id, scope_id, scope_version, paper_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version) REFERENCES literature.research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, paper_id) REFERENCES literature.paper_catalog(tenant_id, paper_id)
);

CREATE TABLE IF NOT EXISTS literature.paper_ingestion_jobs (
    tenant_id     text NOT NULL,
    scope_id      text NOT NULL,
    scope_version integer NOT NULL,
    paper_id      text NOT NULL,
    job_id        text NOT NULL,
    status_json   jsonb NOT NULL CHECK (
        jsonb_typeof(status_json) = 'object'
        AND status_json->>'status' IN (
            'queued', 'uploaded', 'fetching', 'parsing', 'indexed', 'abstract_only', 'failed'
        )
    ),
    updated_at    timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, scope_id, scope_version, paper_id),
    UNIQUE (tenant_id, job_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version) REFERENCES literature.research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS literature.paper_search_results (
    tenant_id           text NOT NULL,
    request_id          text NOT NULL,
    paper_id            text NOT NULL,
    rank                integer NOT NULL,
    relevance_score     double precision NOT NULL,
    matched_queries_json jsonb NOT NULL,
    created_at          timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, request_id, paper_id),
    FOREIGN KEY (tenant_id, request_id) REFERENCES literature.literature_requests(tenant_id, request_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, paper_id) REFERENCES literature.paper_catalog(tenant_id, paper_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS literature.evidence_cards (
    tenant_id     text NOT NULL,
    evidence_id   text NOT NULL,
    scope_id      text NOT NULL,
    scope_version integer NOT NULL,
    paper_id      text NOT NULL,
    card_json     jsonb NOT NULL CHECK (
        jsonb_typeof(card_json) = 'object'
        AND card_json->>'verification_status' IN ('unread', 'read', 'verified', 'unsupported')
    ),
    created_at    timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, evidence_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version) REFERENCES literature.research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id, paper_id) REFERENCES literature.paper_catalog(tenant_id, paper_id)
);

CREATE TABLE IF NOT EXISTS literature.claim_records (
    tenant_id     text NOT NULL,
    claim_id      text NOT NULL,
    scope_id      text NOT NULL,
    scope_version integer NOT NULL,
    claim_json    jsonb NOT NULL CHECK (
        jsonb_typeof(claim_json) = 'object'
        AND claim_json->>'citation_status' IN ('unverified', 'verified', 'unsupported', 'scope_mismatch')
        AND claim_json->>'verification_status' IN ('unverified', 'verified', 'needs_review')
    ),
    created_at    timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, claim_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version) REFERENCES literature.research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS literature.scope_expansion_requests (
    tenant_id     text NOT NULL,
    expansion_id  text NOT NULL,
    scope_id      text NOT NULL,
    scope_version integer NOT NULL,
    request_json  jsonb NOT NULL CHECK (
        jsonb_typeof(request_json) = 'object'
        AND request_json->>'status' IN ('pending', 'approved', 'rejected')
    ),
    created_at    timestamptz NOT NULL,
    decided_at    timestamptz,
    PRIMARY KEY (tenant_id, expansion_id),
    FOREIGN KEY (tenant_id, scope_id, scope_version) REFERENCES literature.research_scopes(tenant_id, scope_id, scope_version) ON DELETE CASCADE
);

CREATE SEQUENCE IF NOT EXISTS literature.audit_event_id_seq;
CREATE TABLE IF NOT EXISTS literature.audit_events (
    event_id      bigint PRIMARY KEY DEFAULT nextval('literature.audit_event_id_seq'::regclass),
    tenant_id     text NOT NULL,
    user_id       text NOT NULL,
    action        text NOT NULL,
    resource_type text NOT NULL,
    resource_id   text NOT NULL,
    details_json  jsonb NOT NULL CHECK (jsonb_typeof(details_json) = 'object'),
    created_at    timestamptz NOT NULL
);
ALTER SEQUENCE literature.audit_event_id_seq
    OWNED BY literature.audit_events.event_id;

CREATE TABLE IF NOT EXISTS literature.provider_cache (
    tenant_id   text NOT NULL,
    cache_key   text NOT NULL,
    provider    text NOT NULL,
    payload_json jsonb NOT NULL,
    status_code integer NOT NULL,
    created_at  timestamptz NOT NULL,
    expires_at  timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, cache_key)
);

CREATE TABLE IF NOT EXISTS vector.embedding_cache (
    tenant_id      text NOT NULL,
    cache_key      text NOT NULL,
    model_name     text NOT NULL,
    embedding_kind text NOT NULL CHECK (embedding_kind IN ('document', 'query')),
    text_sha256    text NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    dimension      integer NOT NULL CHECK (dimension > 0),
    -- Provider caches may contain the unchanged BGE (384d) route as well as
    -- the 1024d Bailian route.  The migrated knowledge corpus below remains
    -- fixed at vector(1024); this cache column intentionally uses pgvector's
    -- unbounded vector type and stores the explicit dimension beside it.
    embedding      vector NOT NULL,
    created_at     timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tenant_id, cache_key),
    CHECK (dimension > 0),
    CHECK (vector_dims(embedding) = dimension)
);
CREATE INDEX IF NOT EXISTS embedding_cache_hash_idx
    ON vector.embedding_cache (tenant_id, model_name, embedding_kind, text_sha256);

CREATE TABLE IF NOT EXISTS vector.knowledge_embeddings (
    tenant_id   text NOT NULL,
    chunk_id    text NOT NULL,
    model       text NOT NULL,
    text_sha256 text NOT NULL CHECK (text_sha256 ~ '^[0-9a-f]{64}$'),
    dimension   integer NOT NULL CHECK (dimension = 1024),
    embedding   vector(1024) NOT NULL,
    created_at  timestamptz NOT NULL,
    PRIMARY KEY (tenant_id, chunk_id, model),
    FOREIGN KEY (tenant_id, chunk_id)
        REFERENCES taskforge.knowledge_chunks(tenant_id, chunk_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS knowledge_embeddings_hash_idx
    ON vector.knowledge_embeddings (tenant_id, model, text_sha256);
-- The application correctness path uses the exact cosine expression
-- (`embedding <=> query_vector`) over the already-authorized tenant/chunk
-- scope. HNSW is an optional acceleration index for the fixed-dimension
-- knowledge corpus; it does not move tenant/ACL filtering out of
-- host/database authorization.  The provider cache intentionally has mixed
-- dimensions (for example BGE 384d and Bailian 1024d), so it uses the hash
-- index above rather than an invalid mixed-dimension vector index.
-- The HNSW acceleration index is deliberately applied only after the exact
-- SQLite/NumPy and PostgreSQL consistency gate passes. Run
-- migrations/postgres/003_taskforge_hnsw.sql at that point.

-- Default-deny tenant RLS. The application must set the trusted tenant in a
-- transaction-local setting before each operation; request/model text never
-- controls this setting. FORCE also protects against accidental owner use.
DO $rls$
DECLARE
    item record;
BEGIN
    FOR item IN
        SELECT * FROM (VALUES
            ('core', 'tasks'), ('core', 'profiles'), ('core', 'runs'),
            ('taskforge', 'knowledge_chunks'), ('taskforge', 'memory_items'),
            ('operations', 'operation_jobs'), ('operations', 'audit_events'),
            ('orchestration', 'speaker_plans'), ('orchestration', 'role_runs'),
            ('orchestration', 'handoffs'), ('orchestration', 'shared_facts'),
            ('orchestration', 'fact_verification_receipts'),
            ('orchestration', 'private_role_memories'),
            ('orchestration', 'role_run_execution_claims'),
            ('review', 'review_cases'), ('review', 'review_case_audit_events'),
            ('review', 'review_case_commands'),
            ('verification', 'verification_records'),
            ('literature', 'literature_requests'), ('literature', 'literature_queries'),
            ('literature', 'paper_catalog'), ('literature', 'paper_identifiers'),
            ('literature', 'research_scopes'), ('literature', 'research_scope_papers'),
            ('literature', 'paper_ingestion_jobs'), ('literature', 'paper_search_results'),
            ('literature', 'evidence_cards'), ('literature', 'claim_records'),
            ('literature', 'scope_expansion_requests'), ('literature', 'audit_events'),
            ('literature', 'provider_cache'),
            ('vector', 'embedding_cache'), ('vector', 'knowledge_embeddings')
        ) AS tables(schema_name, table_name)
    LOOP
        EXECUTE format('ALTER TABLE %I.%I ENABLE ROW LEVEL SECURITY', item.schema_name, item.table_name);
        EXECUTE format('ALTER TABLE %I.%I FORCE ROW LEVEL SECURITY', item.schema_name, item.table_name);
        EXECUTE format('DROP POLICY IF EXISTS tenant_isolation ON %I.%I', item.schema_name, item.table_name);
        EXECUTE format(
            'CREATE POLICY tenant_isolation ON %I.%I USING (tenant_id = NULLIF(current_setting(''taskforge.tenant_id'', true), '''')) WITH CHECK (tenant_id = NULLIF(current_setting(''taskforge.tenant_id'', true), ''''))',
            item.schema_name, item.table_name
        );
    END LOOP;
END;
$rls$;

REVOKE ALL ON ALL TABLES IN SCHEMA core, taskforge, operations, orchestration, review, verification, literature, vector FROM PUBLIC;
GRANT USAGE ON SCHEMA core, taskforge, operations, orchestration, review, verification, literature, vector TO taskforge_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA core, taskforge, operations, orchestration, review, verification, literature, vector TO taskforge_app;
GRANT DELETE ON taskforge.knowledge_chunks TO taskforge_app;
GRANT DELETE ON taskforge.memory_items TO taskforge_app;
GRANT DELETE ON literature.evidence_cards TO taskforge_app;
-- Role execution claims are released after every role run.  Keep the
-- application role unable to delete other append-only/runtime records while
-- allowing this lease cleanup path to complete.
GRANT DELETE ON orchestration.role_run_execution_claims TO taskforge_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA operations, literature TO taskforge_app;
GRANT UPDATE ON SEQUENCE literature.audit_event_id_seq TO taskforge_app;

-- Audit and idempotency receipts are append-only from the application role.
REVOKE UPDATE, DELETE ON operations.audit_events FROM taskforge_app;
REVOKE UPDATE, DELETE ON literature.audit_events FROM taskforge_app;
REVOKE UPDATE, DELETE ON review.review_case_audit_events FROM taskforge_app;
REVOKE UPDATE, DELETE ON review.review_case_commands FROM taskforge_app;

COMMIT;
