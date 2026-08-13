from taskforge.research_feature_reranker import (
    feature_vector,
    train_pairwise_feature_reranker,
)


def test_feature_vector_contains_numeric_and_section_signals() -> None:
    values = feature_vector(
        "Which dataset was evaluated in 2020?",
        "We evaluate on the benchmark dataset in 2020.",
        {"section_title": "Evaluation", "node_type": "paragraph"},
        base_rank=2,
        base_score=0.4,
        reranker_score=1.2,
    )
    assert len(values) == 12
    assert values[0] == 1.2
    assert values[5] == 1.0
    assert values[8] == 1.0


def test_pairwise_feature_training_preserves_model_contract() -> None:
    documents = {
        "positive": {"text": "The experiments use the benchmark dataset.", "metadata": {"section_title": "Experiments"}},
        "negative": {"text": "We discuss future limitations.", "metadata": {"section_title": "Discussion"}},
    }
    rows = [
        {
            "query": "Which dataset is used?",
            "retrieved_ids": ["positive", "negative"],
            "relevant_ids": ["positive"],
            "base_scores": [0.4, 0.3],
            "reranker_scores": [0.2, 0.1],
        }
    ]
    model = train_pairwise_feature_reranker(rows, documents, epochs=2)
    assert model.training_cases == 1
    assert model.positive_pairs == 1
    assert model.rerank([feature_vector("Which dataset is used?", documents[key]["text"], documents[key]["metadata"], base_rank=i + 1, base_score=rows[0]["base_scores"][i], reranker_score=rows[0]["reranker_scores"][i]) for i, key in enumerate(("positive", "negative"))]) == [0, 1]
