"""Open literature discovery separated from bounded evidence retrieval."""

from .evidence import ScopeBoundEvidenceService, route_evidence_intent
from .evidence_query_expander import OpenAICompatibleEvidenceQueryExpander
from .ingestion import PaperIngestionService, SafePDFDownloader
from .models import DiscoveryResult, ProviderPaper, ProviderReport
from .postgres_repository import PostgresLiteratureRepository
from .query_planner import plan_literature_queries
from .query_rewriter import OpenAICompatibleQueryRewriter
from .repository import LiteratureAccess, SQLiteLiteratureRepository
from .rule_query_expander import RuleEvidenceQueryExpander
from .service import LiteratureDiscoveryService
from .zotero_mcp import (
    ZoteroItem,
    ZoteroMCPError,
    ZoteroMCPService,
)

__all__ = [
    "DiscoveryResult",
    "LiteratureAccess",
    "LiteratureDiscoveryService",
    "ProviderPaper",
    "ProviderReport",
    "PaperIngestionService",
    "OpenAICompatibleQueryRewriter",
    "OpenAICompatibleEvidenceQueryExpander",
    "RuleEvidenceQueryExpander",
    "SafePDFDownloader",
    "ScopeBoundEvidenceService",
    "SQLiteLiteratureRepository",
    "PostgresLiteratureRepository",
    "plan_literature_queries",
    "route_evidence_intent",
    "ZoteroItem",
    "ZoteroMCPError",
    "ZoteroMCPService",
]
