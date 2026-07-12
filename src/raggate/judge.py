"""LLM-as-judge backend.

Two modes, chosen automatically:
  * "openai"    — used when the `openai` package is installed AND OPENAI_API_KEY
                  is set. Scorers ask the judge to rate 0..1 against a rubric.
  * "heuristic" — the fallback. `rate()` returns None and each scorer uses its
                  own lexical heuristic instead.

Keeping the judge behind one tiny interface means the scorers don't care which
mode is active, and the kit always runs — even in CI with no secret.

LLM-as-judge scores are non-deterministic (sampling, position bias, and they
vary by judge model/version). Two things keep that honest here:
  * the judge model is pinned and recorded in the report — a score is only
    comparable against the same judge;
  * `runs > 1` averages several judge calls per rating to damp variance
    (recommended for high-variance metrics; costs N× the calls).
"""

from __future__ import annotations

import os
import re
from statistics import mean


class Judge:
    def __init__(
        self,
        model: str | None = None,
        runs: int = 1,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("RAGGATE_JUDGE_MODEL", "gpt-4.1-mini")
        self.runs = max(1, int(os.environ.get("RAGGATE_JUDGE_RUNS", runs)))
        self.temperature = float(os.environ.get("RAGGATE_JUDGE_TEMPERATURE", temperature))
        self._client = self._make_client()

    @property
    def backend(self) -> str:
        return "openai" if self._client is not None else "heuristic"

    @property
    def available(self) -> bool:
        return self._client is not None

    def describe(self) -> str:
        if self._client is None:
            return "heuristic"
        runs = f" ×{self.runs}" if self.runs > 1 else ""
        return f"openai:{self.model}{runs}@t{self.temperature:g}"

    def _make_client(self):
        if not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            from openai import OpenAI  # lazy: optional dependency
        except ImportError:
            return None
        try:
            return OpenAI()
        except Exception:
            return None

    def rate(self, system: str, user: str) -> float | None:
        """Ask the judge for a 0..1 score, averaged over `self.runs` calls.
        Returns None in heuristic mode or if every call fails/parses to None."""
        if self._client is None:
            return None
        scores = [s for _ in range(self.runs) if (s := self._one_call(system, user)) is not None]
        return round(mean(scores), 4) if scores else None

    def _one_call(self, system: str, user: str) -> float | None:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return parse_score(resp.choices[0].message.content or "")
        except Exception:
            return None


_FRACTION = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*(\d+(?:\.\d+)?)")
# A decimal already in [0,1], not embedded in a larger number (so "1.5" is not
# read as ".5", and "0.67" inside "1.5x" is not mismatched).
_DECIMAL_0_1 = re.compile(r"(?<![\d.])(?:0?\.\d+|0|1(?:\.0+)?)(?![\d.])")
_ANY_NUM = re.compile(r"\d+(?:\.\d+)?")


def parse_score(text: str) -> float | None:
    """Extract a 0..1 score from a judge reply, robustly.

    Handles the ways models actually answer: "0.67", "2/3", "9 out of 10",
    "Score: 8/10", and prose that contains an incidental count before the
    score ("Based on 3 claims, 2 supported: 0.67" -> 0.67). Order of attempts:
      1. an explicit fraction "a/b" or "a out of b"
      2. a decimal already in [0, 1] (prefer the LAST one — usually the verdict)
      3. otherwise the last number, rescaled from a 0..10 / 0..100 rating
    """
    if not text:
        return None

    frac = _FRACTION.search(text)
    if frac:
        num, den = float(frac.group(1)), float(frac.group(2))
        if den > 0:
            return _clamp(num / den)

    decimals = _DECIMAL_0_1.findall(text)
    if decimals:
        return _clamp(float(decimals[-1]))

    nums = _ANY_NUM.findall(text)
    if not nums:
        return None
    val = float(nums[-1])
    if val <= 1.0:
        return _clamp(val)
    if val.is_integer():  # a 0..10 or 0..100 rating like "8" or "85"
        if val <= 10:
            return val / 10
        if val <= 100:
            return val / 100
    return 1.0  # non-integer out of range (e.g. "1.5") -> clamp


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))
