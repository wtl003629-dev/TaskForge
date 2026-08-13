from __future__ import annotations

import pytest

from taskforge import research_reranking
from taskforge.hybrid_retrieval import RerankerContractError


class _FakeFlagReranker:
    def __init__(self, model: str, *, use_fp16: bool, devices: str) -> None:
        self.model = model
        self.use_fp16 = use_fp16
        self.devices = devices

    def compute_score(self, pairs, *, batch_size: int, normalize: bool):
        assert batch_size == 2
        assert normalize is True
        return [float(len(pair[1])) for pair in pairs]


def test_bge_adapter_preserves_candidate_cardinality(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(research_reranking, "FlagReranker", _FakeFlagReranker)
    reranker = research_reranking.BGEV2M3Reranker(
        batch_size=2,
        device="cpu",
    )
    assert reranker.score("q", ["a", "abcd"]) == [1.0, 4.0]
    assert reranker.telemetry()["scored_pairs"] == 2


def test_bge_adapter_fails_explicitly_when_extra_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(research_reranking, "FlagReranker", None)
    with pytest.raises(RerankerContractError, match="FlagEmbedding"):
        research_reranking.BGEV2M3Reranker()


def test_fastembed_ensemble_normalizes_each_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeFastEmbed:
        def __init__(self, model_name: str, *, batch_size: int) -> None:
            self.model_name = model_name

        def score(self, query, documents):
            return ([1.0, 2.0] if self.model_name == "a" else [10.0, 10.0])

        def telemetry(self):
            return {"model": self.model_name}

    monkeypatch.setattr(research_reranking, "FastEmbedCrossEncoderReranker", _FakeFastEmbed)
    reranker = research_reranking.FastEmbedEnsembleReranker(("a", "b"), batch_size=2)
    assert reranker.score("q", ["one", "two"]) == [0.0, 0.5]
    assert reranker.telemetry()["backend"] == "fastembed_ensemble"
