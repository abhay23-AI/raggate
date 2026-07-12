# raggate quickstart example

A self-contained sample that runs offline (heuristic mode, no API key):

```bash
pip install raggate
raggate run  --dir examples/quickstart     # see the banded report
raggate gate --dir examples/quickstart     # exit non-zero on a KPI FAIL (LLM mode)
```

`target.py` is a tiny keyword retriever over an inline corpus — replace its
`run(question)` with a call into your real RAG/LLM pipeline. See the top-level
README for the contract.
