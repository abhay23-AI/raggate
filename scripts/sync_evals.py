"""Regenerate the committed evals/ sample from raggate.templates.

Keeps the shipped example in sync with what `raggate init` scaffolds.
Run: python scripts/sync_evals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from raggate import templates  # noqa: E402

FILES = {
    "golden.json": templates.GOLDEN_JSON,
    "gates.yaml": templates.GATES_YAML,
    "gates.high-stakes.yaml": templates.GATES_HIGH_STAKES_YAML,
    "target.py": templates.TARGET_PY,
}


def main() -> None:
    evals = Path(__file__).resolve().parent.parent / "evals"
    evals.mkdir(exist_ok=True)
    for name, content in FILES.items():
        (evals / name).write_text(content)
        print(f"wrote evals/{name}")


if __name__ == "__main__":
    main()
