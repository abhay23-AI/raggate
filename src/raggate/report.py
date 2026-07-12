"""Console report — a banded table of aggregate scores."""

from __future__ import annotations

import os
import sys

from .gates import FAIL, PASS, WARN, GateResult

_C = {
    FAIL: "\033[31m", WARN: "\033[33m", PASS: "\033[32m",
    "N/A": "\033[90m", "dim": "\033[90m", "bold": "\033[1m", "reset": "\033[0m",
}


def _use_color() -> bool:
    # Evaluated per-render (not at import) so NO_COLOR set later is honored.
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(key: str, text: str) -> str:
    return f"{_C[key]}{text}{_C['reset']}" if _use_color() else text


def render(results: list[GateResult], judge_desc: str, n_cases: int) -> str:
    heuristic = judge_desc == "heuristic"
    lines = [
        "",
        _c("bold", f"  raggate — {n_cases} case(s) · judge: {judge_desc}"),
        "  " + "─" * 62,
        f"  {'METRIC':<22}{'SCORE':>7}  {'TARGET':>7}   BAND",
        "  " + "─" * 62,
    ]
    for r in results:
        value = "    n/a" if r.value is None else f"{r.value:7.3f}"
        target = "      —" if r.target is None else f"{r.target:7.3f}"
        kpi_tag = "" if r.is_kpi else _c("dim", " (diagnostic)")
        band = _c(r.band, f"{r.band:<5}")
        lines.append(f"  {r.metric:<22}{value}  {target}   {band}{kpi_tag}")
    lines.append("  " + "─" * 62)

    if heuristic:
        note = [
            "  heuristic mode — lexical proxies, informational only (never blocks).",
            "  Set OPENAI_API_KEY + `pip install raggate[openai]` to enforce gates.",
        ]
    else:
        note = ["  scores are judge-specific — comparable only against the judge above."]
    lines.extend(_c("dim", n) for n in note)
    lines.append("")
    return "\n".join(lines)
