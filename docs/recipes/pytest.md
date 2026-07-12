# Recipe: run raggate inside pytest

Until the dedicated `pytest-raggate` plugin lands, gate from a plain test:

```python
# tests/test_rag_quality.py
from raggate import run_suite, evaluate, has_kpi_failure
from raggate.config import load_gates

def test_rag_meets_quality_bar():
    suite = run_suite("evals")               # calls your evals/target.py
    results = evaluate(suite.scores, load_gates("evals/gates.yaml"))
    assert not (has_kpi_failure(results) and suite.backend == "openai"), \
        {r.metric: (r.value, r.band) for r in results}
```

Run it in CI like any other test. With no `OPENAI_API_KEY` it scores in heuristic
mode (informational); set the key to make the assertion bite.
