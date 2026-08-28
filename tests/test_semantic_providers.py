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
    BailianDenseEmbedder,
    BgeM3DenseEmbedder,
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


def test_bailian_dense_adapter_batches_caches_and_sends_v4_contract(tmp_path) -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-secret"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == "text-embedding-v4"
        assert payload["dimensions"] == 1_024
        assert payload["encoding_format"] == "float"
        values = [
            {
                "index": index,
                "embedding": [float(len(text)), 1.0, *([0.0] * 1_022)],
            }
            for index, text in enumerate(payload["input"])
        ]
        return httpx.Response(200, json={"data": list(reversed(values))})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    embedder = BailianDenseEmbedder(
        api_key="test-secret",
        base_url="https://bailian.example/compatible-mode/v1",
        cache_path=tmp_path / "embeddings-bailian-v4-1024.sqlite3",
        batch_size=2,
        client=client,
    )
    first = embedder.embed_documents(["一", "two", "three"])
    assert [vector[0] for vector in first] == [1.0, 3.0, 5.0]
    assert len(requests) == 2
    assert [len(request["input"]) for request in requests] == [2, 1]
    assert embedder.index_name == "knowledge-bailian-text-embedding-v4-1024-v1"

    second = embedder.embed_documents(["一", "two", "three"])
    assert second == first
    assert len(requests) == 2
    assert len(embedder.embed_query("跨语言问题")) == 1_024
    assert len(requests) == 3
    embedder.embed_query("跨语言问题")
    assert len(requests) == 3


def test_bailian_dense_adapter_retries_transient_errors_and_sanitizes() -> None:
    attempts = 0
    sleeps: list[float] = []

    def retry_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "data": [
                    {
                        "index": 0,
                        "embedding": [1.0, *([0.0] * 1_023)],
                    }
                ]
            },
        )

    retrying = BailianDenseEmbedder(
        api_key="test-secret",
        base_url="https://bailian.example/compatible-mode/v1",
        max_retries=1,
        client=httpx.Client(transport=httpx.MockTransport(retry_handler)),
        sleeper=sleeps.append,
    )
    assert len(retrying.embed_query("query")) == 1_024
    assert attempts == 2
    assert sleeps == [0.0]

    denied = BailianDenseEmbedder(
        api_key="test-secret",
        base_url="https://bailian.example/compatible-mode/v1",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(401, text="private provider body")
            )
        ),
    )
    with pytest.raises(SemanticProviderHTTPError, match="HTTP 401") as error:
        denied.embed_query("query")
    assert "private provider body" not in str(error.value)

    with pytest.raises(SemanticProviderConfigurationError, match="dimension 1024"):
        BailianDenseEmbedder(
            api_key="test-secret",
            base_url="https://bailian.example/compatible-mode/v1",
            dimension=512,
        )


def test_bge_m3_dense_adapter_isolated_cache_and_contract(tmp_path) -> None:
    class FakeBgeM3:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []
            self.kwargs: list[dict[str, object]] = []
            self.tokenizer = self

        def encode(self, sentences, **kwargs):  # type: ignore[no-untyped-def]
            values = list(sentences)
            self.calls.append(values)
            self.kwargs.append(dict(kwargs))
            return {
                "dense_vecs": [
                    [float(len(value)), 1.0, *([0.0] * 1_022)] for value in values
                ]
            }

        def __call__(self, sentences, **kwargs):  # type: ignore[no-untyped-def]
            return {"length": [len(str(value)) for value in sentences]}

    fake = FakeBgeM3()
    cache = tmp_path / "embeddings-bge-m3-v1.sqlite3"
    embedder = BgeM3DenseEmbedder(
        model_name="BAAI/bge-m3",
        model=fake,
        cache_path=cache,
        batch_size=2,
    )
    assert len(embedder.embed_documents(["中文段落"])[0]) == 1_024
    assert fake.kwargs[0]["max_length"] == 128
    assert len(embedder.embed_query("中文问题")) == 1_024
    first_call_count = len(fake.calls)
    assert embedder.embed_documents(["中文段落"])[0][0] == float(len("中文段落"))
    assert embedder.embed_query("中文问题")[0] == float(len("中文问题"))
    assert len(fake.calls) == first_call_count

    with pytest.raises(SemanticProviderConfigurationError, match="model_path"):
        BgeM3DenseEmbedder(model_path=tmp_path / "missing")
