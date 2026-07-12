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


def _mk_evals(tmp_path, target_src):
    (tmp_path / "golden.json").write_text(
        '{"version":"1","cases":[{"id":"a","question":"q?","expected":"e"}]}'
    )
    (tmp_path / "target.py").write_text(target_src)
    return tmp_path


def test_missing_target_raises_file_not_found(tmp_path):
    (tmp_path / "golden.json").write_text(
        '{"version":"1","cases":[{"id":"a","question":"q?","expected":"e"}]}'
    )
    with pytest.raises(FileNotFoundError, match="raggate init"):
        run_suite(tmp_path)


def test_target_without_run_raises(tmp_path):
    _mk_evals(tmp_path, "x = 1\n")
    with pytest.raises(TargetError, match="must define a callable"):
        run_suite(tmp_path)


def test_target_run_exception_names_the_case(tmp_path):
    _mk_evals(tmp_path, "def run(q):\n    raise ValueError('boom')\n")
    with pytest.raises(TargetError, match="case 'a'"):
        run_suite(tmp_path)
