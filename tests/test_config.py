"""Config loading + validation (the merge logic the review flagged as untested)."""

import pytest

from raggate.config import load_gates, load_judge


def _write(tmp_path, text):
    p = tmp_path / "gates.yaml"
    p.write_text(text)
    return p


def test_missing_file_returns_defaults(tmp_path):
    gates = load_gates(tmp_path / "nope.yaml")
    assert gates["faithfulness"]["fail"] == 0.75  # the default


def test_override_merges_over_defaults(tmp_path):
    p = _write(tmp_path, "metrics:\n  faithfulness:\n    fail: 0.80\n")
    gates = load_gates(p)
    assert gates["faithfulness"]["fail"] == 0.80          # overridden
    assert gates["faithfulness"]["target"] == 0.90        # default preserved
    assert gates["answer_relevancy"]["fail"] == 0.65      # untouched metric intact


def test_inverted_thresholds_rejected(tmp_path):
    p = _write(tmp_path, "metrics:\n  faithfulness: {fail: 0.9, warn: 0.5, target: 0.8}\n")
    with pytest.raises(ValueError, match="fail <= warn <= target"):
        load_gates(p)


def test_bad_metrics_type_rejected(tmp_path):
    p = _write(tmp_path, "metrics: not-a-mapping\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_gates(p)


def test_load_judge_defaults_and_override(tmp_path):
    defaults = {"model": "gpt-4.1-mini", "runs": 1, "temperature": 0.0}
    assert load_judge(tmp_path / "nope.yaml") == defaults
    p = _write(tmp_path, "judge:\n  model: gpt-4o\n  runs: 3\n")
    j = load_judge(p)
    assert j["model"] == "gpt-4o" and j["runs"] == 3 and j["temperature"] == 0.0


def test_load_judge_rejects_zero_runs(tmp_path):
    p = _write(tmp_path, "judge:\n  runs: 0\n")
    with pytest.raises(ValueError, match="runs must be >= 1"):
        load_judge(p)
