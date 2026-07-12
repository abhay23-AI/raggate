"""Gate band logic — pure functions, no API key needed."""

from raggate.gates import (
    DEFAULT_GATES,
    FAIL,
    PASS,
    WARN,
    band_for,
    evaluate,
    has_kpi_failure,
)


def test_band_for_boundaries():
    spec = {"fail": 0.75, "warn": 0.85, "target": 0.90}
    assert band_for(0.74, spec) == FAIL
    assert band_for(0.75, spec) == WARN   # exactly at fail threshold -> not failing
    assert band_for(0.84, spec) == WARN
    assert band_for(0.85, spec) == PASS   # exactly at warn threshold -> passing
    assert band_for(0.99, spec) == PASS


def test_evaluate_marks_missing_as_na_and_carries_target():
    gates = {"faithfulness": DEFAULT_GATES["faithfulness"]}
    results = evaluate({"faithfulness": None}, gates)
    assert results[0].band == "N/A"
    assert results[0].value is None
    assert results[0].target == 0.90  # target is threaded through for the report


def test_kpi_failure_detection():
    gates = {
        "faithfulness": {"kpi": True, "fail": 0.75, "warn": 0.85, "target": 0.90},
        "extra": {"kpi": False, "fail": 0.90, "warn": 0.95, "target": 0.99},
    }
    # diagnostic metric fails, KPI passes -> overall not a failure
    assert has_kpi_failure(evaluate({"faithfulness": 0.92, "extra": 0.10}, gates)) is False
    # KPI fails -> overall failure
    assert has_kpi_failure(evaluate({"faithfulness": 0.50, "extra": 0.99}, gates)) is True


def test_defaults_are_ordered_and_faithfulness_is_strictest():
    for spec in DEFAULT_GATES.values():
        assert spec["fail"] <= spec["warn"] <= spec["target"]
    # faithfulness (hallucination guardrail) must carry the highest floor
    floors = {m: s["fail"] for m, s in DEFAULT_GATES.items()}
    assert floors["faithfulness"] == max(floors.values())
