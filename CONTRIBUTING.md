# Contributing to raggate

Thanks for considering a contribution! This project aims to be the smallest
sharp tool for gating RAG/LLM quality in CI — contributions that keep it
simple, honest, and dependency-light are especially welcome.

## Getting started

```bash
git clone git@github.com:abhay23-AI/raggate.git
cd raggate
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,openai]"
pytest -q
raggate run          # runs the sample suite in heuristic mode
```

## Ways to help

- **New scorers.** Add a function to `src/raggate/scorers.py` with the
  signature `scorer(case, output, judge) -> float | None`, register it in the
  `SCORERS` dict, and give it default bands in `gates.py` + `templates.py`.
  Support both modes: use `judge.rate(...)` when available, fall back to a
  lexical heuristic when it returns `None`. Name the metric for what it actually
  computes — do not borrow a RAGAS name for a weaker proxy.
- **Golden-set examples** from real domains (anonymized).
- **Docs, bug fixes, and reduced dependencies.**

## Ground rules

- **Every metric must run in heuristic mode** (no API key). The kit must never
  hard-require a paid API to execute — that's what keeps CI green for forks.
- **Only the LLM path gates.** Heuristics are informational smoke signals; never
  wire a gate to a heuristic score.
- **Keep files small and cohesive** (< ~200 lines) and prefer pure functions.
- **Add a test.** Scorer/gate/parse logic must be covered by a unit test that
  runs without network access. After editing templates, run
  `python scripts/sync_evals.py` to refresh the committed `evals/` sample.
- **No breaking changes to the `target.run()` contract** without discussion.

## Pull requests

1. Branch from `main`.
2. Keep the PR focused; describe the *why*, not just the *what*.
3. Ensure `pytest -q` passes, `ruff check .` is clean, and `raggate run` still
   works on the sample.

## Reporting issues

Open an issue with a minimal repro. For scoring disagreements, include the
`case`, the `output` your system returned, and the score you expected vs. got.
