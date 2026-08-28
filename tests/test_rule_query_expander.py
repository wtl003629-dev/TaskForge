from __future__ import annotations

import pytest

from taskforge.literature.rule_query_expander import RuleEvidenceQueryExpander


@pytest.mark.asyncio
async def test_rule_expander_preserves_entities_numbers_and_negation() -> None:
    expander = RuleEvidenceQueryExpander()
    synonym, keyword = await expander.expand(
        "How does BERT compare with RoBERTa on Recall@10, not Recall@50?",
        "general_fact",
    )

    assert synonym.startswith("How does BERT")
    assert "BERT" in keyword
    assert "RoBERTa" in keyword
    assert "10" in keyword and "50" in keyword
    assert "not" in keyword.casefold()


@pytest.mark.asyncio
async def test_rule_expander_is_local_and_closable() -> None:
    expander = RuleEvidenceQueryExpander()
    _, keyword = await expander.expand("What method was used?", "general_fact")
    assert keyword
    await expander.aclose()
