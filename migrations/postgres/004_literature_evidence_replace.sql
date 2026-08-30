-- Allow the application role to replace one paper's derived evidence cards.
-- Audit tables remain append-only and are intentionally not covered.

GRANT DELETE ON literature.evidence_cards TO taskforge_app;
