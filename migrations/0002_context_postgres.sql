-- PostgreSQL hardening for the optional TaskForge context repository.
--
-- The runtime role must not own these tables and must not be SUPERUSER or
-- carry BYPASSRLS.  The host sets taskforge.tenant_id transaction-locally via
-- parameterised set_config(..., true); model/request text must never set it.

BEGIN;

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

-- JSONB GIN is a candidate-filter accelerator only.  It does not turn this
-- repository into a lexical or semantic search implementation.
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_acl_gin
    ON taskforge.knowledge_chunks USING gin (acl_json jsonb_ops);
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_kb
    ON taskforge.knowledge_chunks (
        tenant_id,
        (metadata_json ->> 'knowledge_base_id')
    );
CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_latest
    ON taskforge.knowledge_chunks (
        tenant_id,
        (COALESCE(document_id, source_uri)),
        version_order DESC,
        created_at DESC
    );

REVOKE ALL ON SCHEMA taskforge FROM PUBLIC;
REVOKE ALL ON taskforge.knowledge_chunks FROM PUBLIC;
REVOKE ALL ON taskforge.memory_items FROM PUBLIC;

COMMENT ON POLICY knowledge_tenant_isolation
    ON taskforge.knowledge_chunks IS
    'Default-deny tenant RLS. Host must call set_config(taskforge.tenant_id, trusted_tenant, true) inside every transaction.';
COMMENT ON POLICY memory_tenant_isolation
    ON taskforge.memory_items IS
    'Default-deny tenant RLS. User/conversation memory scope is additionally filtered by parameterised repository SQL.';

COMMIT;
