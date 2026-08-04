from __future__ import annotations

import json
import math

import httpx
import pytest

from taskforge.hybrid_retrieval import (
    EmbeddingContractError,
    QdrantHybridIndex,
    RerankerContractError,
)
from taskforge.semantic_providers import (
    FastEmbedCrossEncoderReranker,
    FastEmbedDenseEmbedder,
    OpenAIEmbeddingsDenseEmbedder,
    SemanticProviderConfigurationError,
    SemanticProviderHTTPError,
)


class FakeDenseModel:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.inputs.append(list(texts))
        for text in texts:
            yield [float(len(text)), 1.0, 0.0]


class FakeReranker:
    def rerank(self, query: str, documents: list[str]):
        return [float(document.count(query)) for document in documents]


def test_openai_embedding_adapter_batches_reorders_and_integrates_qdrant() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-secret"
        payload = json.loads(request.content)
        requests.append(payload)
        data = [
            {"index": index, "embedding": [float(len(text)), 1.0, 0.0, 0.5]}
            for index, text in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(data))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = OpenAIEmbeddingsDenseEmbedder(
        api_key="test-secret",
        model="embedding-model",
        dimension=4,
        enabled=True,
        batch_size=2,
        base_url="https://embedding.example/v1",
        client=client,
    )
    vectors = embedder.embed_documents(["one", "two", "three"])
    assert len(requests) == 2
    assert vectors == [
        [3.0, 1.0, 0.0, 0.5],
        [3.0, 1.0, 0.0, 0.5],
        [5.0, 1.0, 0.0, 0.5],
    ]
    index = QdrantHybridIndex.in_memory(
        collection_name="mocked-real-embedding-contract",
        embedder=embedder,
    )
    assert index is not None


def test_openai_embedding_adapter_fails_closed_and_sanitizes_http_errors() -> None:
    with pytest.raises(SemanticProviderConfigurationError, match="disabled"):
        OpenAIEmbeddingsDenseEmbedder(
            api_key="key", model="model", dimension=3
        )

    malformed = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
            )
        )
    )
    embedder = OpenAIEmbeddingsDenseEmbedder(
        api_key="key",
        model="model",
        dimension=3,
        enabled=True,
        client=malformed,
    )
    with pytest.raises(EmbeddingContractError, match="dimension"):
        embedder.embed_query("query")

    failed = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(401, text="secret response body")
        )
    )
    denied = OpenAIEmbeddingsDenseEmbedder(
        api_key="key",
        model="model",
        dimension=3,
        enabled=True,
        client=failed,
    )
    with pytest.raises(SemanticProviderHTTPError, match="HTTP 401") as error:
        denied.embed_query("query")
    assert "secret response body" not in str(error.value)


def test_fastembed_wrappers_use_prefixes_and_validate_contracts() -> None:
    dense_model = FakeDenseModel()
    embedder = FastEmbedDenseEmbedder(
        model_name="test-model",
        dimension=3,
        model=dense_model,
    )
    assert embedder.embed_query("needle")[0] == len("query: needle")
    assert embedder.embed_documents(["haystack"])[0][0] == len("passage: haystack")
    assert dense_model.inputs == [["query: needle"], ["passage: haystack"]]

    reranker = FastEmbedCrossEncoderReranker(model=FakeReranker())
    assert reranker.score("x", ["x", "xx", "none"]) == [1.0, 2.0, 0.0]


def test_fastembed_wrappers_reject_wrong_count_and_nonfinite_scores() -> None:
    class MissingDense:
        def embed(self, texts: list[str]):
            return []

    with pytest.raises(EmbeddingContractError, match="one vector"):
        FastEmbedDenseEmbedder(
            model_name="missing", dimension=3, model=MissingDense()
        ).embed_query("query")

    class BadScores:
        def rerank(self, query: str, documents: list[str]):
            return [math.nan for _ in documents]

    with pytest.raises(RerankerContractError, match="non-finite"):
        FastEmbedCrossEncoderReranker(model=BadScores()).score("q", ["doc"])
