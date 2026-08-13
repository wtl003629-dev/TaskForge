from __future__ import annotations

from pathlib import Path

from taskforge.tatqa_reranker import TATQADomainReranker


def test_domain_reranker_fit_prefers_positive_and_round_trips(tmp_path: Path) -> None:
    examples = [
        ("what was revenue in 2019", "Revenue | 2019 | $100", 1),
        ("what was revenue in 2019", "Operating expenses | 2017 | $40", 0),
        ("how many years exceeded 5 million", "Table row: Other | 2019=6 | 2018=8", 1),
        ("how many years exceeded 5 million", "Narrative about debt", 0),
    ]
    model = TATQADomainReranker.fit(examples, epochs=80)
    scores = model.score(
        "what was revenue in 2019",
        ["Revenue | 2019 | $100", "Operating expenses | 2017 | $40"],
    )

    assert scores[0] > scores[1]
    path = tmp_path / "reranker.json"
    model.save(path)
    restored = TATQADomainReranker.load(path)
    assert restored.model_dump() == model.model_dump()


def test_domain_reranker_rejects_incompatible_feature_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"feature_names": []}', encoding="utf-8")

    try:
        TATQADomainReranker.load(path)
    except ValueError as exc:
        assert "feature schema" in str(exc)
    else:  # pragma: no cover - assertion documents the contract.
        raise AssertionError("incompatible schema must be rejected")
