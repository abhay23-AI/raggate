# Good first issues

Genuinely-scoped starter tasks. Comment on the tracking issue (or open one) and
I'll help scope it. Each should be a small, self-contained PR with a test.

1. **`--report junit.xml`** — emit a JUnit report from `raggate gate` so CI systems
   show per-metric results as test cases. (Touches `cli.py`, add a `report.py` writer.)
2. **`raggate import --from ragas|promptfoo`** — map an existing dataset into raggate's
   golden.json so users don't rebuild their eval set.
3. **New heuristic scorer: answer length / refusal detection** — flag empty or
   "I can't answer" responses. (Add to `scorers.py` + `SCORERS`, give it bands.)
4. **Reranker-lift metric** — score retrieval quality before vs after a rerank step.
5. **`docs/recipes/`: pre-commit hook recipe** — run the gate on staged changes.
6. **HTML report** — a static `report.html` from the `--json` output.
7. **Multi-run *majority* aggregation** — currently the judge averages N runs;
   add a `median`/`majority` option in `gates.yaml`.
8. **pytest plugin (`pytest-raggate`)** — `assert_gate(case, bands=...)` as a
   pytest-native assertion. (New package; happy to pair on this.)
