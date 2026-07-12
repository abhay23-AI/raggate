"""Scorer behavior on known inputs, in heuristic mode (Judge with no API key).

These are the credibility core — they assert the heuristics behave sanely and
that the robustness fixes hold (string/int inputs, unicode, empties)."""

from raggate.judge import Judge
from raggate.scorers import (
    answer_correctness,
    answer_relevancy,
    citation_support,
    context_coverage,
    faithfulness,
)

J = Judge()  # no OPENAI_API_KEY in CI -> heuristic mode


def test_faithfulness_grounded_vs_hallucinated():
    ctx = {"contexts": ["Employees get 20 vacation days per year."]}
    grounded = faithfulness({}, {"answer": "Employees get 20 vacation days.", **ctx}, J)
    hallucinated = faithfulness({}, {"answer": "The capital of France is Berlin.", **ctx}, J)
    assert grounded > hallucinated
    assert hallucinated == 0.0


def test_faithfulness_empty_answer_is_excluded_not_perfect():
    # empty answer must never score a vacuous 1.0 on the hallucination metric
    assert faithfulness({}, {"answer": "  ", "contexts": ["x"]}, J) is None


def test_unicode_answer_does_not_score_vacuous_one():
    # non-Latin hallucination against unrelated English context -> low, not 1.0
    out = {"answer": "これは完全な幻覚です", "contexts": ["totally unrelated"]}
    assert faithfulness({}, out, J) == 0.0


def test_context_coverage_reflects_retrieval():
    case = {"expected": "20 vacation days per year"}
    good = {"answer": "", "contexts": ["Employees get 20 vacation days per year."]}
    bad = {"answer": "", "contexts": ["The office is open on weekdays."]}
    assert context_coverage(case, good, J) == 1.0
    assert context_coverage(case, bad, J) < 1.0


def test_citation_support_handles_str_dict_and_int():
    ctx = ["The retention period is seven years."]
    assert citation_support({}, {"answer": "", "contexts": ctx,
                                 "citations": ["retention period seven years"]}, J) == 1.0
    assert citation_support({}, {"answer": "", "contexts": ctx,
                                 "citations": [{"text": "retention period seven years"}]}, J) == 1.0
    # int index citation must not crash; just scores 0 (no overlap)
    assert citation_support({}, {"answer": "", "contexts": ctx, "citations": [3]}, J) == 0.0


def test_citation_support_none_when_no_citations():
    assert citation_support({}, {"answer": "x", "contexts": ["y"], "citations": []}, J) is None


def test_citation_support_dict_citation_via_runner():
    # a dict citation that survives normalization must score by its text, not its repr
    from raggate.runner import _normalize_output

    out = _normalize_output(
        {"answer": "", "contexts": ["The capital of France is Paris."],
         "citations": [{"text": "Paris"}]},
        "c",
    )
    assert citation_support({}, out, J) == 1.0


def test_faithfulness_ignores_content_free_filler_sentence():
    ctx = {"contexts": ["The Eiffel Tower is in Paris."]}
    grounded = faithfulness({}, {"answer": "The Eiffel Tower is in Paris.", **ctx}, J)
    filler_answer = "The Eiffel Tower is in Paris. It is what it is."
    with_filler = faithfulness({}, {"answer": filler_answer, **ctx}, J)
    assert grounded == 1.0
    assert with_filler == 1.0  # the filler sentence must not deflate the score
    # an entirely content-free answer is nothing-to-measure -> None
    assert faithfulness({}, {"answer": "It is. They are.", **ctx}, J) is None


def test_answer_relevancy_discriminates_on_topic_from_off_topic():
    case = {"question": "How many vacation days do employees get?"}
    on_topic = {"answer": "Employees get 20 vacation days per year.", "contexts": []}
    off_topic = {"answer": "The weather in Paris is mild.", "contexts": []}
    assert answer_relevancy(case, on_topic, J) > answer_relevancy(case, off_topic, J)


def test_answer_correctness_range_and_empty():
    case = {"question": "How many vacation days?", "expected": "20 days"}
    out = {"answer": "Employees get 20 vacation days per year.", "contexts": []}
    assert answer_correctness(case, out, J) > 0.0
    assert answer_correctness(case, {"answer": "", "contexts": []}, J) == 0.0


def test_scorers_see_single_digit_numbers():
    # single digits carry facts — they must not be dropped as noise, or a wrong
    # number scores as perfect on the token-level metrics.
    from raggate.text import content_tokens

    assert "5" in content_tokens("Up to 5 days")
    case = {"expected": "Up to 5 days."}
    right = answer_correctness(case, {"answer": "Up to 5 days.", "contexts": []}, J)
    wrong = answer_correctness(case, {"answer": "Up to 8 days.", "contexts": []}, J)
    assert right == 1.0
    assert wrong < 1.0
