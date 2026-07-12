"""End-to-end smoke test on the sample evals in heuristic mode (no API key)."""

from pathlib import Path

import pytest

from raggate.runner import TargetError, _as_str_list, _normalize_output, run_suite
from raggate.scorers import SCORERS

EVALS = Path(__file__).resolve().parent.parent / "evals"


def test_suite_runs_on_sample():
    suite = run_suite(EVALS)
    assert suite.n_cases == 6
    assert suite.backend == "heuristic"
    for metric in ("faithfulness", "context_coverage", "answer_correctness"):
        assert suite.scores[metric] is not None
        assert 0.0 <= suite.scores[metric] <= 1.0


def test_sample_target_retrieves_relevant_context():
    suite = run_suite(EVALS)
    assert suite.scores["context_coverage"] >= 0.5


def test_all_scorers_registered():
    assert set(SCORERS) == {
        "faithfulness",
        "answer_relevancy",
        "citation_support",
        "context_coverage",
        "answer_correctness",
    }


def test_as_str_list_does_not_explode_strings():
    assert _as_str_list("source-A") == ["source-A"]      # not ['s','o','u',...]
    assert _as_str_list(["a", "b"]) == ["a", "b"]
    assert _as_str_list([3, 7]) == ["3", "7"]
    assert _as_str_list(None) == []


def test_normalize_output_requires_answer():
    with pytest.raises(TargetError, match="must return a dict"):
        _normalize_output(["not", "a", "dict"], "case-1")


def test_normalize_output_coerces_types():
    out = _normalize_output({"answer": 42, "contexts": "one", "citations": [1, 2]}, "c")
    assert out["answer"] == "42"
    assert out["contexts"] == ["one"]
    assert out["citations"] == ["1", "2"]
