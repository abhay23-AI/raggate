# raggate

A thin, CI-gated evaluation gate for RAG & LLM systems. Point it at a golden set, pick pass/warn/fail bands, and it fails your build when answer quality regresses. Runs with **no API key** (lexical heuristic scorers) and upgrades to **LLM-as-judge** scoring when you set `OPENAI_API_KEY`.

[![eval-gate](https://github.com/abhay23-AI/raggate/actions/workflows/eval.yml/badge.svg)](https://github.com/abhay23-AI/raggate/actions/workflows/eval.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/abhay23-AI/raggate/blob/main/LICENSE)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/abhay23-AI/raggate/blob/main/pyproject.toml)

It is deliberately small. It does not replace RAGAS or DeepEval — it is the drop-in gate you wire into GitHub Actions in five minutes, and it composes with them (see [Prior art](#prior-art)).

## Why

Most teams test their prompts by eyeballing a few answers and shipping. Then real users arrive and quality quietly regresses with no alarm. `raggate` treats RAG quality like any other production concern: measured against a versioned golden set, and gated in CI.

## Quickstart

```bash
pip install raggate            # core (heuristic mode, no key needed)
# or, for LLM-as-judge scoring:
pip install "raggate[openai]"

raggate init          # scaffolds ./evals (golden set, gates.yaml, target.py)
raggate run           # scores the golden set, prints a banded report
raggate gate          # same, but exits non-zero when a KPI metric FAILs
```

`raggate run` scores a tiny built-in example immediately:

```
  raggate — 6 case(s) · judge: heuristic
  ──────────────────────────────────────────────────────────────
  METRIC                  SCORE   TARGET   BAND
  ──────────────────────────────────────────────────────────────
  faithfulness            1.000    0.900   PASS
  answer_relevancy        0.715    0.800   WARN
  citation_support        1.000    0.850   PASS
  context_coverage        1.000    0.800   PASS
  answer_correctness      0.627    0.850   WARN
  ──────────────────────────────────────────────────────────────
  heuristic mode — lexical proxies, informational only (never blocks).
```

## Point it at your system

`raggate` calls one function you own. Edit `evals/target.py`:

```python
from my_app.rag import answer   # your existing RAG entrypoint

def run(question: str) -> dict:
    result = answer(question)
    return {
        "answer": result.text,
        "contexts": result.retrieved_chunks,   # list[str] — for coverage & faithfulness
        "citations": result.citations,         # optional
    }
```

## The metrics (and what they actually measure)

Two metric names are chosen to match **what this kit computes**, which is a cheaper proxy than the same-sounding RAGAS metric — so the name does not overpromise. The LLM-judge path is a pointwise rubric score; the heuristic path is a lexical proxy and is **informational only** (never gates).

| Metric | Question | LLM-judge path | Heuristic (no key) | Standard it approximates |
|---|---|---|---|---|
| `faithfulness` | Are the answer's claims grounded in the context? (hallucination) | rubric 0–1 | fraction of sentences covered by context | [RAGAS faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) (claim-level NLI) |
| `answer_relevancy` | Does the answer address the question? | rubric 0–1 | question-token coverage (weak — can be fooled by echoing) | [RAGAS response relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) (gen-question + cosine) |
| `citation_support` | Do citations overlap the cited context? | — (overlap only) | citation↔context token coverage | [ALCE citation precision/recall](https://arxiv.org/abs/2305.14627) (NLI) — this is a weaker overlap proxy |
| `context_coverage` | Did retrieval surface the needed tokens? | — (deterministic) | reference-token coverage by retrieved context | [RAGAS context recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) (claim attribution) — **not the same**; renamed to be honest |
| `answer_correctness` | Does the answer match ground truth? | rubric 0–1 | token-F1 vs expected | [RAGAS answer correctness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_correctness/) (claim-F1 + cosine) |

Verify a metric name against its source before quoting a number — `context_coverage` and `citation_support` are intentionally *not* the RAGAS metrics.

## Bands, not a single threshold

LLM-as-judge scores are non-deterministic (sampling, position bias, judge-model dependence). A lone `>= 0.80` gate flaps on that noise and teams learn to ignore it. Bands encode intent — edit `evals/gates.yaml`:

```yaml
judge:              # pin the judge — scores are only comparable against the same one
  model: gpt-4.1-mini
  runs: 1           # >1 averages several calls per rating to damp variance
  temperature: 0.0

metrics:
  faithfulness:     { kpi: true, fail: 0.75, warn: 0.85, target: 0.90 }
  context_coverage: { kpi: true, fail: 0.60, warn: 0.70, target: 0.80 }
```

`FAIL` blocks the merge (KPI metrics only). `WARN` reports but passes. Diagnostic (non-KPI) metrics inform without gating. Gating is only enforced in **openai mode** (a judge is configured); **heuristic mode never blocks**. Bands *absorb* variance — they don't remove it — so pin the judge and raise `runs` for high-variance metrics.

**Defaults are starting points for general-purpose RAG, not domain-calibrated.** `faithfulness` carries the highest bar because it is the hallucination guardrail. For regulated domains, use the shipped high-stakes profile — Stanford RegLab found commercial legal RAG tools hallucinate [17–33%](https://reglab.stanford.edu/publications/hallucination-free-assessing-the-reliability-of-leading-ai-legal-research-tools/) of the time, so general-purpose faithfulness/citation floors are unsafe there (note: raggate's `faithfulness` measures grounding in retrieved context, not legal correctness — a distinct concern, but the direction holds):

```bash
raggate gate --gates evals/gates.high-stakes.yaml   # faithfulness/citation floors ≥ 0.90
```

## CI gate (GitHub Actions)

```yaml
name: eval-gate
on: [pull_request]
jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install "raggate[openai]"
      - run: raggate gate
        env: { OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }} }
```

Without the `OPENAI_API_KEY` secret the gate runs in heuristic mode (informational, never blocks). Add the secret to make it a hard quality gate.

## Prior art

`raggate` is inspired by these; if you want deep metrics or dashboards, use them — raggate is the thin gate that composes with them.

- **[RAGAS](https://github.com/explodinggradients/ragas)** (Apache-2.0) — the reference library for RAG-specific metrics. raggate's LLM metrics approximate its constructs with pointwise judges; its faithfulness/context-recall definitions are the standard.
- **[DeepEval](https://github.com/confident-ai/deepeval)** (Apache-2.0) — the broadest metric set and pytest-native gating. If you want assertions inside your test suite, use it.
- **[promptfoo](https://github.com/promptfoo/promptfoo)** (MIT) — declarative eval + CI action with RAG assertions.
- **[TruLens](https://github.com/truera/trulens)**, **[Arize Phoenix](https://github.com/Arize-ai/phoenix)** (Elastic-2.0, source-available), **[MLflow LLM eval](https://mlflow.org/docs/latest/genai/eval-monitor/)** — tracing/observability platforms.

## Design notes

1. Retrieval quality (`context_coverage`) is measured separately from generation — most "the LLM is dumb" bugs are retrieval misses.
2. The golden set is versioned and small. Ten sharp cases you trust beat a thousand you don't; grow it every time a bug reaches production.
3. Gating is enforced only in openai mode; heuristic mode (no key) is a smoke test and never blocks. Note `citation_support` and `context_coverage` are deterministic overlap metrics with no LLM path — they still gate in openai mode.

## Roadmap

- [ ] Reranker-lift metric (retrieval score before/after rerank)
- [ ] LLM/NLI path for `citation_support` (toward ALCE-style precision/recall)
- [ ] Adversarial / prompt-injection test pack
- [ ] Multi-run majority (not just mean) aggregation
- [ ] HTML report + PR comment bot

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](https://github.com/abhay23-AI/raggate/blob/main/CONTRIBUTING.md). Every metric must run in heuristic mode (no API key) and ship with a test.

## License

MIT © Abhay Trivedi. See [LICENSE](https://github.com/abhay23-AI/raggate/blob/main/LICENSE).
