"""Adapter between raggate and YOUR system.

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
_STOP = {"a", "an", "the", "of", "to", "in", "is", "are", "do", "does",
         "how", "many", "what", "within", "per", "and", "or", "for"}


def _tokens(text):
    return {w for w in _WORD.findall(text.lower())
            if w not in _STOP and (len(w) > 1 or w.isdigit())}


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
