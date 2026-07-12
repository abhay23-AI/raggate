"""Load and validate a golden dataset.

Format (evals/golden.json):

    {
      "version": "1.0.0",
      "cases": [
        {
          "id": "unique-id",
          "question": "…",
          "expected": "ground-truth answer",
          "contexts": ["optional gold source chunk", …],
          "category": "optional label",
          "difficulty": "easy | medium | hard"
        }
      ]
    }

Start small — 10 sharp cases you trust beat 1000 you don't.
"""

from __future__ import annotations

import json
from pathlib import Path

_REQUIRED = ("id", "question", "expected")


def load_dataset(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"golden dataset not found: {p} (run `raggate init`)")

    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{p}: invalid JSON — {e}") from e

    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{p}: expected a non-empty 'cases' array")

    seen: set[str] = set()
    for i, case in enumerate(cases):
        for field in _REQUIRED:
            if not case.get(field):
                raise ValueError(f"{p}: case #{i} is missing required field '{field}'")
        cid = case["id"]
        if cid in seen:
            raise ValueError(f"{p}: duplicate case id '{cid}'")
        seen.add(cid)
    return cases
