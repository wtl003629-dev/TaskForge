-- TaskForge persistent context schema for PostgreSQL.
-- This migration is supplied for deployment; it is not exercised by the
-- local SQLite test suite.

CREATE SCHEMA IF NOT EXISTS taskforge;

CREATE TABLE IF NOT EXISTS taskforge.knowledge_chunks (
    tenant_id       text        NOT NULL,
    chunk_id        text        NOT NULL,
    text_content    text        NOT NULL,
    source_uri      text        NOT NULL,
    document_id     text,
    version         text        NOT NULL,
    version_order   integer     NOT NULL CHECK (version_order >= 0),
    acl_json        jsonb       NOT NULL,
    valid_from      timestamptz,
    valid_until     timestamptz,
    created_at      timestamptz NOT NULL,
    metadata_json   jsonb       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, chunk_id),
    CHECK (jsonb_typeof(acl_json) = 'array'),
    CHECK (valid_from IS NULL OR valid_until IS NULL OR valid_from < valid_until)
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_tenant
    ON taskforge.knowledge_chunks (tenant_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_tenant_validity
    ON taskforge.knowledge_chunks (tenant_id, valid_from, valid_until);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_tenant_version
    ON taskforge.knowledge_chunks (tenant_id, document_id, version_order DESC, version);

CREATE TABLE IF NOT EXISTS taskforge.memory_items (
    tenant_id       text        NOT NULL,
    memory_id       text        NOT NULL,
    content         text        NOT NULL,
    scope           text        NOT NULL CHECK (scope IN ('tenant', 'org', 'user', 'agent', 'task')),
    scope_id        text        NOT NULL,
    provenance_json jsonb       NOT NULL,
    importance      double precision NOT NULL CHECK (importance >= 0.0 AND importance <= 1.0),
    created_at      timestamptz NOT NULL,
    updated_at      timestamptz NOT NULL,
    expires_at      timestamptz,
    tags_json       jsonb       NOT NULL DEFAULT '[]'::jsonb,
    metadata_json   jsonb       NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (tenant_id, memory_id),
    CHECK (jsonb_typeof(provenance_json) = 'object'),
    CHECK (jsonb_typeof(tags_json) = 'array')
);

CREATE INDEX IF NOT EXISTS idx_memory_items_tenant
    ON taskforge.memory_items (tenant_id);
CREATE INDEX IF NOT EXISTS idx_memory_items_tenant_expiry
    ON taskforge.memory_items (tenant_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_memory_items_tenant_scope
    ON taskforge.memory_items (tenant_id, scope, scope_id, updated_at DESC);

ALTER TABLE taskforge.knowledge_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE taskforge.knowledge_chunks FORCE ROW LEVEL SECURITY;
ALTER TABLE taskforge.memory_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE taskforge.memory_items FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS knowledge_tenant_isolation ON taskforge.knowledge_chunks;
CREATE POLICY knowledge_tenant_isolation ON taskforge.knowledge_chunks
    USING (
        tenant_id = NULLIF(current_setting('taskforge.tenant_id', true), '')
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('taskforge.tenant_id', true), '')
    );

DROP POLICY IF EXISTS memory_tenant_isolation ON taskforge.memory_items;
CREATE POLICY memory_tenant_isolation ON taskforge.memory_items
    USING (
        tenant_id = NULLIF(current_setting('taskforge.tenant_id', true), '')
    )
    WITH CHECK (
        tenant_id = NULLIF(current_setting('taskforge.tenant_id', true), '')
    );

REVOKE ALL ON SCHEMA taskforge FROM PUBLIC;
REVOKE ALL ON taskforge.knowledge_chunks FROM PUBLIC;
REVOKE ALL ON taskforge.memory_items FROM PUBLIC;

COMMENT ON SCHEMA taskforge IS
    'Grant USAGE only to the runtime role; migration ownership must remain separate.';
COMMENT ON TABLE taskforge.knowledge_chunks IS
    'Runtime role should receive only SELECT, INSERT, UPDATE. Set taskforge.tenant_id from trusted authenticated host state at transaction checkout, never from model or request text.';
COMMENT ON TABLE taskforge.memory_items IS
    'Runtime role should receive only SELECT, INSERT, UPDATE. DELETE requires a separate retention capability; taskforge.tenant_id must be a trusted session setting.';
