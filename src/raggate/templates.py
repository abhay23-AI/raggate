"""Canonical scaffold files written by `raggate init`.

The repo also ships a committed copy of these under ./evals so the kit runs
(and its own CI is green) straight after clone. Keep the two in sync
(scripts/sync_evals.py regenerates the committed copies from here).
"""

from __future__ import annotations

GOLDEN_JSON = r"""{
  "version": "1.0.0",
  "description": "Sample golden set over a tiny fake company handbook. Replace with 10+ real cases from your own domain.",
  "cases": [
    {
      "id": "vacation-days",
      "question": "How many vacation days do employees get per year?",
      "expected": "20 days of paid vacation per year.",
      "category": "numerical-extraction",
      "difficulty": "easy"
    },
    {
      "id": "vacation-rollover",
      "question": "How many unused vacation days can roll over to next year?",
      "expected": "Up to 5 days.",
      "category": "numerical-extraction",
      "difficulty": "easy"
    },
    {
      "id": "core-hours",
      "question": "What are the core collaboration hours?",
      "expected": "10am to 4pm.",
      "category": "metadata-query",
      "difficulty": "easy"
    },
    {
      "id": "expense-window",
      "question": "Within how many days must expense reports be submitted?",
      "expected": "Within 30 days of purchase.",
      "category": "policy",
      "difficulty": "easy"
    },
    {
      "id": "insurance-employee",
      "question": "What percentage of the health insurance premium does the company cover for employees?",
      "expected": "80 percent for employees.",
      "category": "numerical-extraction",
      "difficulty": "medium"
    },
    {
      "id": "probation-period",
      "question": "How long is the probation period and what notice applies?",
      "expected": "90 days, during which either party may end employment with two weeks notice.",
      "category": "policy-recommendation",
      "difficulty": "medium"
    }
  ]
}
"""

GATES_YAML = r"""# Band-based quality gates (a metric FAILs below `fail`, WARNs below `warn`,
# else PASSes). Only metrics with `kpi: true` can block a deploy. These are
# starting points for general-purpose RAG — tune them to your domain.
#
# The `judge` block pins the LLM used to score in openai mode. Scores are only
# comparable against the same judge; changing it invalidates your thresholds.
# `runs` > 1 averages several judge calls per rating to damp variance.
judge:
  model: gpt-4.1-mini
  runs: 1
  temperature: 0.0

metrics:
  faithfulness:        # hallucination guardrail — held to the highest bar
    kpi: true
    fail: 0.75
    warn: 0.85
    target: 0.90
  answer_relevancy:
    kpi: true
    fail: 0.65
    warn: 0.75
    target: 0.80
  citation_support:    # project convention — no industry-standard number
    kpi: true
    fail: 0.65
    warn: 0.75
    target: 0.85
  context_coverage:
    kpi: true
    fail: 0.60
    warn: 0.70
    target: 0.80
  answer_correctness:  # wide band on purpose — a known-noisy metric
    kpi: true
    fail: 0.60
    warn: 0.75
    target: 0.85
"""

GATES_HIGH_STAKES_YAML = r"""# High-stakes profile for regulated / safety-critical domains (legal, medical,
# finance). Use with: `raggate gate --gates evals/gates.high-stakes.yaml`
#
# Rationale: Stanford RegLab found commercial RAG-based legal tools hallucinate
# 17-33% of the time — a 0.70 faithfulness floor would let a system that is
# wrong a third of the time PASS. Here faithfulness and citation floors sit at
# 0.90+, and a human-in-the-loop is still recommended above these gates.
judge:
  model: gpt-4.1-mini
  runs: 3            # average 3 calls to reduce judge variance on hard cases
  temperature: 0.0

metrics:
  faithfulness:
    kpi: true
    fail: 0.90
    warn: 0.95
    target: 0.98
  citation_support:
    kpi: true
    fail: 0.90
    warn: 0.95
    target: 0.99
  context_coverage:
    kpi: true
    fail: 0.75
    warn: 0.85
    target: 0.90
  answer_relevancy:
    kpi: true
    fail: 0.75
    warn: 0.85
    target: 0.90
  answer_correctness:
    kpi: true
    fail: 0.75
    warn: 0.85
    target: 0.90
"""

TARGET_PY = r'''"""Adapter between raggate and YOUR system.

Replace the body of `run()` with a call into your real RAG/LLM pipeline. The
only contract: given a question string, return a dict with:

    {
      "answer":    str,          # required
      "contexts":  list[str],    # the chunks you retrieved (for recall/faithfulness)
      "citations": list[str],    # optional: sources you cited
    }

The sample below is a tiny keyword retriever over an inline corpus, so the kit
runs end-to-end out of the box. Delete it and wire in your own.
"""

from __future__ import annotations

import re

# --- a stand-in corpus (replace with your retriever) ---
CORPUS = [
    "Employees accrue 20 days of paid vacation per year. Unused vacation rolls "
    "over up to a maximum of 5 days into the next year.",
    "The standard work week is 40 hours. Core collaboration hours are 10am to "
    "4pm, and the rest of the schedule is flexible.",
    "Expense reports must be submitted within 30 days of purchase. "
    "Reimbursements are paid on the next payroll cycle after approval.",
    "New hires are enrolled in health insurance on their first day. The company "
    "covers 80 percent of the premium for employees and 50 percent for dependents.",
    "The probation period for new employees is 90 days, during which either "
    "party may end employment with two weeks notice.",
]

_WORD = re.compile(r"[a-z0-9]+")
_STOP = set("a an the of to in is are do does how many what within per and or for".split())


def _tokens(text):
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1}


def _sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def run(question: str) -> dict:
    q = _tokens(question)

    # naive retrieval: rank documents by content-word overlap, take top 2
    ranked = sorted(CORPUS, key=lambda d: len(q & _tokens(d)), reverse=True)
    contexts = [d for d in ranked[:2] if q & _tokens(d)] or ranked[:1]

    # naive synthesis: the single sentence best matching the question
    best, best_score = "", -1
    for doc in contexts:
        for sent in _sentences(doc):
            score = len(q & _tokens(sent))
            if score > best_score:
                best, best_score = sent, score

    return {"answer": best, "contexts": contexts, "citations": [best]}
'''
