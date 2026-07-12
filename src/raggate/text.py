"""Small, dependency-free text helpers used by the heuristic scorers.

These are deliberately simple lexical proxies. They exist so the kit runs
end-to-end with no API key (heuristic mode). They are NOT validated against
human judgment and must never be treated as equivalent to the LLM-judge path —
token overlap cannot tell a correct paraphrase from a fluent hallucination.
When OPENAI_API_KEY is set, the LLM judge replaces them for the
language-understanding metrics.
"""

from __future__ import annotations

import re

# Unicode-aware: match runs of word characters in any script, excluding "_".
# (ASCII-only matching silently returned zero tokens for non-Latin text, which
# made coverage() vacuously score 1.0 — a real hazard for a hallucination tool.)
_WORD = re.compile(r"[^\W_]+", re.UNICODE)

# English stopwords stripped before overlap so scores reflect content words.
_STOP = frozenset(
    """a an and are as at be by for from has have in is it its of on or that the
    to was were will with what which who whom how why when where do does did your
    you we they this these those i me my our their them he she his her not no if
    then than so such can could should would may might must into over under about""".split()  # noqa: SIM905
)


def tokenize(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def content_tokens(text: str) -> set[str]:
    """Meaning-bearing tokens: stopwords removed, single-letter noise dropped —
    but single digits kept (0-9 carry facts the numeric-extraction cases test)."""
    return {t for t in tokenize(text) if t not in _STOP and (len(t) > 1 or t.isdigit())}


def coverage(needle: str, haystack: str) -> float | None:
    """Fraction of `needle`'s content tokens present in `haystack` (0..1).

    Returns None when `needle` has no content tokens — there is nothing to
    measure, and returning a number would be misleading.
    """
    need = content_tokens(needle)
    if not need:
        return None
    hay = content_tokens(haystack)
    return sum(1 for t in need if t in hay) / len(need)


def token_f1(pred: str, gold: str) -> float | None:
    """Token-overlap F1 between two strings (0..1). None if `gold` is empty."""
    g = content_tokens(gold)
    if not g:
        return None
    p = content_tokens(pred)
    if not p:
        return 0.0
    overlap = len(p & g)
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def sentences(text: str) -> list[str]:
    """Naive sentence split — good enough for per-claim heuristics."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [s.strip() for s in parts if s.strip()]
