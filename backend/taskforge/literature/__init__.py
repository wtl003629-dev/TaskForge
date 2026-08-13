"""Open literature discovery separated from bounded evidence retrieval."""

from .evidence import ScopeBoundEvidenceService, route_evidence_intent
from .ingestion import PaperIngestionService, SafePDFDownloader
from .models import DiscoveryResult, ProviderPaper, ProviderReport
from .query_planner import plan_literature_queries
from .query_rewriter import OpenAICompatibleQueryRewriter
from .repository import LiteratureAccess, SQLiteLiteratureRepository
from .service import LiteratureDiscoveryService

__all__ = [
    "DiscoveryResult",
    "LiteratureAccess",
    "LiteratureDiscoveryService",
    "ProviderPaper",
    "ProviderReport",
    "PaperIngestionService",
    "OpenAICompatibleQueryRewriter",
    "SafePDFDownloader",
    "ScopeBoundEvidenceService",
    "SQLiteLiteratureRepository",
    "plan_literature_queries",
    "route_evidence_intent",
]
