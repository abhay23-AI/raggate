"""Test fixtures.

Force heuristic mode for every test so the suite is hermetic: without this, a
developer who has both the `openai` package installed and OPENAI_API_KEY set
would run tests against the live judge, making the deterministic score
assertions flap and the `backend == "heuristic"` checks fail.
"""

import pytest


@pytest.fixture(autouse=True)
def _force_heuristic(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
