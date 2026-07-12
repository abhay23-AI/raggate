"""Band-based quality gates: FAIL / WARN / PASS.

Why bands instead of a single threshold? LLM-as-judge scores are
non-deterministic — the same output can score a few points differently run to
run. A lone `>= 0.80` gate flaps red/green on that noise and teams learn to
ignore it. Bands encode intent:

    value <  fail   -> FAIL   (blocks the merge, if the metric is a KPI)
    value <  warn   -> WARN   (comment only, still passes)
    else            -> PASS

Only KPI metrics can block. Diagnostic metrics inform without gating.

Bands ABSORB variance, they don't remove it — pair them with a pinned judge
(recorded in the report) and, for high-variance metrics, multi-run averaging
(`judge.runs > 1`). These defaults are starting points for general-purpose RAG;
tune them to your domain. For legal/medical, see evals/gates.high-stakes.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass

FAIL, WARN, PASS = "FAIL", "WARN", "PASS"

# Starting-point defaults for general-purpose RAG — NOT domain-calibrated.
# faithfulness (the hallucination guardrail) carries the highest bar.
# answer_correctness keeps a wide band because it is a known-noisy metric.
DEFAULT_GATES: dict[str, dict] = {
    "faithfulness":       {"kpi": True,  "fail": 0.75, "warn": 0.85, "target": 0.90},
    "answer_relevancy":   {"kpi": True,  "fail": 0.65, "warn": 0.75, "target": 0.80},
    "citation_support":   {"kpi": True,  "fail": 0.65, "warn": 0.75, "target": 0.85},
    "context_coverage":   {"kpi": True,  "fail": 0.60, "warn": 0.70, "target": 0.80},
    "answer_correctness": {"kpi": True,  "fail": 0.60, "warn": 0.75, "target": 0.85},
}


@dataclass(frozen=True)
class GateResult:
    metric: str
    value: float | None
    band: str
    is_kpi: bool
    target: float | None = None


def band_for(value: float, spec: dict) -> str:
    if value < spec["fail"]:
        return FAIL
    if value < spec["warn"]:
        return WARN
    return PASS


def evaluate(scores: dict[str, float | None], gates: dict[str, dict]) -> list[GateResult]:
    """Turn aggregate scores into banded results, in gate-config order."""
    results: list[GateResult] = []
    for metric, spec in gates.items():
        value = scores.get(metric)
        is_kpi = bool(spec.get("kpi"))
        target = spec.get("target")
        if value is None:
            results.append(GateResult(metric, None, "N/A", is_kpi, target))
        else:
            results.append(GateResult(metric, value, band_for(value, spec), is_kpi, target))
    return results


def has_kpi_failure(results: list[GateResult]) -> bool:
    return any(r.is_kpi and r.band == FAIL for r in results)
