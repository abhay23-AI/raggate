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


def test_empty_metrics_block_falls_back_to_defaults(tmp_path):
    # an explicitly empty `metrics:` (YAML -> None) means "no overrides"
    gates = load_gates(_write(tmp_path, "metrics:\n"))
    assert gates["faithfulness"]["fail"] == 0.75


def test_custom_metric_merges_over_fallback_base(tmp_path):
    p = _write(tmp_path, "metrics:\n  my_metric: {kpi: true, fail: 0.5, warn: 0.6, target: 0.7}\n")
    gates = load_gates(p)
    assert gates["my_metric"] == {"kpi": True, "fail": 0.5, "warn": 0.6, "target": 0.7}


def test_non_numeric_band_rejected(tmp_path):
    p = _write(tmp_path, "metrics:\n  faithfulness: {fail: high, warn: 0.8, target: 0.9}\n")
    with pytest.raises(ValueError, match="needs numeric"):
        load_gates(p)


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
