"""Orchestration: load the golden set + your target, score every case, aggregate.

Your system stays yours — the runner only calls the `run(question) -> dict`
function you expose in evals/target.py.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

from .dataset import load_dataset
from .judge import Judge, JudgeError
from .scorers import SCORERS


@dataclass(frozen=True)
class SuiteResult:
    scores: dict[str, float | None]          # aggregate per metric
    per_case: list[dict] = field(default_factory=list)
    backend: str = "heuristic"
    judge_desc: str = "heuristic"
    n_cases: int = 0


class TargetError(RuntimeError):
    """Raised when the user's evals/target.py fails to load or run."""


def _load_target(evals_dir: Path):
    """Import evals/target.py and return its `run` callable.

    Cleans up sys.path afterwards so repeated runs don't accumulate entries or
    let an evals/ module (e.g. evals/json.py) shadow stdlib.
    """
    target_path = evals_dir / "target.py"
    if not target_path.exists():
        raise FileNotFoundError(f"{target_path} not found (run `raggate init`)")

    added = str(evals_dir.resolve())
    sys.path.insert(0, added)
    try:
        spec = importlib.util.spec_from_file_location("raggate_user_target", target_path)
        if spec is None or spec.loader is None:
            raise TargetError(f"could not import {target_path}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:  # noqa: BLE001 — surface any user import-time error cleanly
            raise TargetError(f"{target_path} failed to import: {e}") from e
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(added)

    run = getattr(module, "run", None)
    if not callable(run):
        raise TargetError(f"{target_path} must define a callable `run(question) -> dict`")
    return run


def _as_str_list(value) -> list[str]:
    """Coerce to a list of strings. A bare string becomes a one-element list —
    NOT list('abc') == ['a','b','c'], which silently corrupted scoring."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v if isinstance(v, str) else str(v) for v in value]
    return [str(value)]


def _as_list(value) -> list:
    """Coerce to a list WITHOUT stringifying elements — citations may be dicts
    or indices, and citation_support coerces each itself. A bare string/dict
    becomes a one-element list (not exploded)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _normalize_output(raw, case_id: str) -> dict:
    if not isinstance(raw, dict) or "answer" not in raw:
        raise TargetError(
            f"target.run() must return a dict with an 'answer' key "
            f"(case '{case_id}' got {type(raw).__name__})"
        )
    answer = raw.get("answer")
    return {
        "answer": answer if isinstance(answer, str) else ("" if answer is None else str(answer)),
        "contexts": _as_str_list(raw.get("contexts")),
        # citations keep their structure (dict/int/str) — the scorer coerces.
        "citations": _as_list(raw.get("citations")),
    }


def run_suite(
    evals_dir: str | Path,
    metrics: list[str] | None = None,
    judge_config: dict | None = None,
) -> SuiteResult:
    evals_dir = Path(evals_dir)
    cases = load_dataset(evals_dir / "golden.json")
    run = _load_target(evals_dir)
    judge = Judge(**(judge_config or {}))
    active = list(metrics or SCORERS.keys())

    per_case: list[dict] = []
    collected: dict[str, list[float]] = {m: [] for m in active}

    for case in cases:
        try:
            raw = run(case["question"])
        except Exception as e:  # noqa: BLE001 — user code; report which case broke
            raise TargetError(f"target.run() raised on case '{case['id']}': {e}") from e
        output = _normalize_output(raw, case["id"])
        row = {"id": case["id"], "scores": {}}
        for m in active:
            value = SCORERS[m](case, output, judge)
            row["scores"][m] = value
            if value is not None:
                collected[m].append(value)
        per_case.append(row)

    # A judge configured but 0/N calls succeeded is a broken configuration, not
    # a heuristic run — fail loudly rather than silently gating on lexical proxies.
    if judge.degraded:
        raise JudgeError(
            f"OpenAI judge '{judge.model}' was configured but all {judge.calls} "
            f"rating calls failed (check OPENAI_API_KEY, connectivity, and quota). "
            f"Refusing to gate on the lexical fallback."
        )

    aggregate = {m: (round(mean(v), 4) if v else None) for m, v in collected.items()}
    return SuiteResult(
        scores=aggregate,
        per_case=per_case,
        backend=judge.backend,
        judge_desc=judge.describe(),
        n_cases=len(cases),
    )
