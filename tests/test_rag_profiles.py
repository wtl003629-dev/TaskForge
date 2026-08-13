from taskforge.rag_evaluation import EvalCorpusDocument
from taskforge.rag_profiles import (
    CorpusMetadata,
    corpus_metadata,
    profile_metadata,
    query_features,
    select_retrieval_profile,
)


def test_profile_selection_uses_query_and_corpus_signals_not_dataset_name() -> None:
    corpus = CorpusMetadata(
        document_count=10,
        table_count=2,
        page_count=0,
        source_count=1,
        has_page_coordinates=False,
        has_table_structure=True,
    )
    assert select_retrieval_profile("What was the percentage growth in 2021?", corpus) == (
        "table_numeric"
    )
    assert select_retrieval_profile("Summarize the approval policy.", corpus) == (
        "general_text"
    )


def test_cross_document_and_pdf_profiles_require_matching_corpus_capabilities() -> None:
    cross_document = CorpusMetadata(
        document_count=4,
        table_count=0,
        page_count=0,
        source_count=4,
        has_page_coordinates=False,
        has_table_structure=False,
    )
    assert (
        select_retrieval_profile(
            "According to both reports, which source changed?", cross_document
        )
        == "cross_document"
    )
    pdf = CorpusMetadata(
        document_count=4,
        table_count=1,
        page_count=2,
        source_count=1,
        has_page_coordinates=True,
        has_table_structure=True,
    )
    assert select_retrieval_profile("Compare the table on page 2 with page 1.", pdf) == (
        "pdf_layout"
    )
    pdf_without_layout_words = CorpusMetadata(
        document_count=4,
        table_count=0,
        page_count=2,
        source_count=1,
        has_page_coordinates=True,
        has_table_structure=False,
    )
    assert select_retrieval_profile("Who approved the change?", pdf_without_layout_words) == (
        "pdf_layout"
    )


def test_cross_document_profile_matches_multiple_real_source_labels() -> None:
    corpus = CorpusMetadata(
        document_count=10,
        table_count=0,
        page_count=0,
        source_count=3,
        has_page_coordinates=False,
        has_table_structure=False,
        source_labels=("techcrunch", "the verge", "fortune"),
    )
    assert (
        select_retrieval_profile(
            "Was the report from TechCrunch consistent with The Verge article?",
            corpus,
        )
        == "cross_document"
    )


def test_metadata_is_derived_from_documents_and_is_auditable() -> None:
    documents = [
        EvalCorpusDocument(
            document_id="table",
            text="Year | Revenue",
            source_uri="file://a",
            metadata={"kind": "table", "table_rows": [["Year", "Revenue"]], "page": 1, "source": "a"},
        ),
        EvalCorpusDocument(
            document_id="text",
            text="Policy",
            source_uri="file://b",
            metadata={"kind": "paragraph", "page": 2, "source": "b", "bbox": [0, 0, 1, 1]},
        ),
    ]
    metadata = corpus_metadata(documents)
    features = query_features("What was the total revenue in 2021?")
    payload = profile_metadata(
        select_retrieval_profile("What was the total revenue in 2021?", metadata),
        metadata,
        features,
    )
    assert metadata.table_count == 1
    assert metadata.source_count == 2
    assert metadata.source_labels == ("a", "b")
    assert payload["name"] == "table_numeric"
    assert payload["selection"]["corpus"]["source_count"] == 2
