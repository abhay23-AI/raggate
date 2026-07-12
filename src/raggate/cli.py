"""Command line: `raggate init | run | gate | version`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__, templates
from .config import load_gates, load_judge
from .gates import evaluate, has_kpi_failure
from .judge import JudgeError
from .report import markdown, render
from .runner import TargetError, run_suite


def _gate_passes(suite, results) -> bool:
    """The gate blocks only on a KPI failure in a real (openai) judge run."""
    return not (has_kpi_failure(results) and suite.backend == "openai")

_SCAFFOLD = {
    "golden.json": templates.GOLDEN_JSON,
    "gates.yaml": templates.GATES_YAML,
    "gates.high-stakes.yaml": templates.GATES_HIGH_STAKES_YAML,
    "target.py": templates.TARGET_PY,
}


def _cmd_init(args) -> int:
    evals = Path(args.dir)
    evals.mkdir(parents=True, exist_ok=True)
    for name, content in _SCAFFOLD.items():
        dest = evals / name
        if dest.exists() and not args.force:
            print(f"  skip  {dest} (exists — use --force to overwrite)")
            continue
        dest.write_text(content)
        print(f"  write {dest}")
    print(f"\nScaffolded {evals}/. Edit target.py to call your system, then run `raggate run`.")
    return 0


def _run(args):
    gates_path = Path(args.gates) if args.gates else Path(args.dir) / "gates.yaml"
    gates = load_gates(gates_path)
    judge_config = load_judge(gates_path)
    suite = run_suite(args.dir, judge_config=judge_config)
    results = evaluate(suite.scores, gates)
    print(render(results, suite.judge_desc, suite.n_cases))
    if args.json:
        payload = {
            "judge": suite.judge_desc,
            "backend": suite.backend,
            "n_cases": suite.n_cases,
            "scores": suite.scores,
            "gates": [
                {"metric": r.metric, "value": r.value, "target": r.target,
                 "band": r.band, "kpi": r.is_kpi}
                for r in results
            ],
            "per_case": suite.per_case,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"  wrote {args.json}")

    # Native GitHub Actions reporting: append the score table to the run summary.
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        md = markdown(results, suite.judge_desc, suite.n_cases, _gate_passes(suite, results))
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(md)
        except OSError:
            pass  # never let reporting break the run
    return suite, results


def _cmd_run(args) -> int:
    _run(args)
    return 0


def _cmd_gate(args) -> int:
    suite, results = _run(args)
    if has_kpi_failure(results):
        if suite.backend == "openai":
            print("  ❌ GATE FAILED — a KPI metric is in the FAIL band.\n")
            return 1
        print("  ⚠️  KPI in FAIL band, but running in heuristic mode — not blocking.")
        print("      Set OPENAI_API_KEY to enforce this gate in CI.\n")
        return 0
    print("  ✅ GATE PASSED — no KPI failures.\n")
    return 0


def _add_eval_args(p) -> None:
    p.add_argument("--dir", default="evals", help="evals directory (default: evals)")
    p.add_argument("--gates", metavar="PATH", help="gate config file (default: <dir>/gates.yaml)")
    p.add_argument("--json", metavar="PATH", help="also write full results as JSON")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="raggate", description="CI-gated RAG/LLM evaluation.")
    parser.add_argument("-v", "--version", action="version", version=f"raggate {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="scaffold ./evals (golden set, gates, target)")
    p_init.add_argument("--dir", default="evals")
    p_init.add_argument("--force", action="store_true", help="overwrite existing files")
    p_init.set_defaults(func=_cmd_init)

    p_run = sub.add_parser("run", help="score the golden set and print a report")
    _add_eval_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    p_gate = sub.add_parser("gate", help="score, then exit non-zero on a KPI failure")
    _add_eval_args(p_gate)
    p_gate.set_defaults(func=_cmd_gate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, TargetError, JudgeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
