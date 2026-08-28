-- Apply only after TaskForge migration and exact SQLite/NumPy retrieval gates.
BEGIN;

CREATE INDEX IF NOT EXISTS knowledge_embeddings_hnsw_cosine_idx
    ON vector.knowledge_embeddings USING hnsw (embedding vector_cosine_ops);

COMMIT;
