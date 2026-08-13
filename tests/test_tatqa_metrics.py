from taskforge.tatqa_metrics import tatqa_answer_metrics


def test_tatqa_metric_handles_scale_and_percent() -> None:
    result = tatqa_answer_metrics(
        "4.6%",
        4.6,
        answer_type="arithmetic",
        gold_scale="percent",
    )
    assert result["exact_match"] == 1.0
    assert result["f1"] == 1.0
    assert result["scale_match"] == 1.0


def test_tatqa_metric_uses_one_to_one_alignment_for_multi_span() -> None:
    result = tatqa_answer_metrics(
        ["2019", "2018"],
        ["2018", "2019"],
        answer_type="multi-span",
    )
    assert result["exact_match"] == 1.0
    assert result["f1"] == 1.0


def test_tatqa_metric_treats_count_f1_as_exact_match() -> None:
    result = tatqa_answer_metrics("3", 3, answer_type="count")
    assert result["exact_match"] == 1.0
    assert result["f1"] == 1.0

    wrong = tatqa_answer_metrics("4", 3, answer_type="count")
    assert wrong["exact_match"] == 0.0
    assert wrong["f1"] == 0.0


def test_tatqa_metric_infers_word_scale_from_answer() -> None:
    result = tatqa_answer_metrics("2 million", 2, gold_scale="million")
    assert result["exact_match"] == 1.0
    assert result["f1"] == 1.0
    assert result["prediction_scale"] == "million"
