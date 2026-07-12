"""The five built-in scorers (plus room to add your own).

Every scorer has the same signature:

    scorer(case: dict, output: dict, judge: Judge) -> float | None

`case`   — one golden test case: {question, expected, contexts?, ...}
`output` — what your system returned: {answer, contexts, citations?}
`judge`  — the LLM judge; use it when available, else fall back to a heuristic.

Return a score in [0, 1], or None to exclude this metric for this case
(e.g. citation support when the answer has no citations).

Naming honesty: two metrics are named for what THIS kit actually computes,
which differs from the same-sounding RAGAS metrics:
  * `context_coverage` — token coverage of the reference by the retrieved
    context. RAGAS `context_recall` is LLM claim-attribution; this is a cheaper
    proxy, so it carries a different name.
  * `citation_support` — token overlap of each citation with the context. The
    academic standard (ALCE) is NLI-based citation precision/recall; "support"
    signals the weaker overlap check, not verified attribution.
See the README metric table for the full mapping.
"""

from __future__ import annotations

from .judge import Judge
from .text import coverage, sentences, token_f1

# Heuristic mode only: a claim/citation counts as "supported" at >=60% token
# coverage of the context. A lexical proxy, not entailment.
_SUPPORT = 0.6


def _contexts_text(output: dict) -> str:
    return "\n".join(output.get("contexts") or [])


def faithfulness(case: dict, output: dict, judge: Judge) -> float | None:
    """Are the answer's claims grounded in the retrieved context? (hallucination)"""
    answer = output.get("answer", "")
    ctx = _contexts_text(output)
    if not answer.strip():
        return None  # nothing to judge; relevancy/correctness will catch emptiness
    score = judge.rate(
        "You are a strict RAG evaluator. Output only a number between 0 and 1.",
        f"Rate the fraction of claims in the ANSWER that are directly "
        f"supported by the CONTEXT (0 = none, 1 = all).\n\n"
        f"CONTEXT:\n{ctx}\n\nANSWER:\n{answer}",
    )
    if score is not None:
        return score
    claims = sentences(answer)
    if not claims:
        return None
    supported = sum(1 for c in claims if (cov := coverage(c, ctx)) is not None and cov >= _SUPPORT)
    return round(supported / len(claims), 4)


def answer_relevancy(case: dict, output: dict, judge: Judge) -> float | None:
    """Does the answer address the question that was asked?

    Heuristic caveat: the lexical fallback measures how much of the question's
    content appears in the answer — it can be fooled by an answer that merely
    echoes the question. Trust the LLM path here; treat the heuristic as a
    rough smoke signal only.
    """
    q, answer = case.get("question", ""), output.get("answer", "")
    if not answer.strip():
        return 0.0
    score = judge.rate(
        "You rate how well an answer addresses a question, ignoring factual "
        "accuracy. Output only a number between 0 and 1.",
        f"How well does the ANSWER address the QUESTION "
        f"(0 = off-topic, 1 = fully on-topic)?\n\n"
        f"QUESTION:\n{q}\n\nANSWER:\n{answer}",
    )
    if score is not None:
        return score
    cov = coverage(q, answer)
    return round(cov, 4) if cov is not None else None


def citation_support(case: dict, output: dict, judge: Judge) -> float | None:
    """Fraction of citations whose text overlaps the retrieved context.
    None when the answer has no citations. (Overlap proxy, not NLI attribution.)"""
    citations = output.get("citations") or []
    if not citations:
        return None
    ctx = _contexts_text(output)
    hits = 0
    for cite in citations:
        text = _citation_text(cite)
        cov = coverage(text, ctx)
        if cov is not None and cov >= _SUPPORT:
            hits += 1
    return round(hits / len(citations), 4)


def _citation_text(cite: object) -> str:
    """Coerce any citation representation (str, dict, int index, …) to text."""
    if isinstance(cite, str):
        return cite
    if isinstance(cite, dict):
        return str(cite.get("text") or cite.get("source") or cite.get("quote") or cite)
    return str(cite)


def context_coverage(case: dict, output: dict, judge: Judge) -> float | None:
    """Token coverage of the reference (expected answer + gold contexts) by the
    retrieved context. A deterministic proxy for "did retrieval surface the
    evidence", measured separately from generation. Not RAGAS context_recall."""
    expected = case.get("expected", "")
    gold_ctx = "\n".join(case.get("contexts") or [])
    target = (expected + "\n" + gold_ctx).strip()
    cov = coverage(target, _contexts_text(output))
    return round(cov, 4) if cov is not None else None


def answer_correctness(case: dict, output: dict, judge: Judge) -> float | None:
    """Does the final answer agree with the ground-truth answer?
    Heuristic = token-F1 (surface), not RAGAS's claim-level F1 + semantic blend."""
    expected, answer = case.get("expected", ""), output.get("answer", "")
    if not expected:
        return None
    if not answer.strip():
        return 0.0
    score = judge.rate(
        "You grade answer correctness against a reference. "
        "Output only a number between 0 and 1.",
        f"How correct is the ANSWER compared to the REFERENCE "
        f"(0 = wrong, 1 = fully correct)?\n\n"
        f"REFERENCE:\n{expected}\n\nANSWER:\n{answer}",
    )
    if score is not None:
        return score
    f1 = token_f1(answer, expected)
    return round(f1, 4) if f1 is not None else None


# name -> scorer. Add your own here (and give it bands in gates.yaml).
SCORERS = {
    "faithfulness": faithfulness,
    "answer_relevancy": answer_relevancy,
    "citation_support": citation_support,
    "context_coverage": context_coverage,
    "answer_correctness": answer_correctness,
}
