"""Load gate + judge configuration, merging a YAML file over built-in defaults."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .gates import DEFAULT_GATES

_JUDGE_DEFAULTS = {"model": "gpt-4.1-mini", "runs": 1, "temperature": 0.0}


def _read(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    return data


def load_gates(path: str | Path) -> dict[str, dict]:
    """Return the gate config. Missing file -> defaults. Per-metric keys are
    merged over the defaults, so you can tweak one threshold without restating
    the rest. Raises ValueError on a malformed file or inverted thresholds."""
    gates = copy.deepcopy(DEFAULT_GATES)
    metrics = _read(path).get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"{path}: 'metrics' must be a mapping of metric -> thresholds")

    for name, spec in metrics.items():
        if not isinstance(spec, dict):
            raise ValueError(f"{path}: metric '{name}' must be a mapping")
        base = gates.get(name, {"kpi": False, "fail": 0.60, "warn": 0.75, "target": 0.85})
        merged = {**base, **spec}
        _validate_bands(path, name, merged)
        gates[name] = merged
    return gates


def load_judge(path: str | Path) -> dict:
    """Judge settings (model / runs / temperature). File `judge:` block over
    defaults; environment variables (RAGGATE_JUDGE_*) win at construction time."""
    judge = {**_JUDGE_DEFAULTS, **(_read(path).get("judge") or {})}
    if int(judge["runs"]) < 1:
        raise ValueError(f"{path}: judge.runs must be >= 1")
    return judge


def _validate_bands(path: str | Path, name: str, spec: dict) -> None:
    try:
        fail, warn, target = float(spec["fail"]), float(spec["warn"]), float(spec["target"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: metric '{name}' needs numeric fail/warn/target") from e
    if not (fail <= warn <= target):
        raise ValueError(
            f"{path}: metric '{name}' thresholds must satisfy fail <= warn <= target "
            f"(got fail={fail}, warn={warn}, target={target})"
        )
