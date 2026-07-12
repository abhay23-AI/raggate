"""load_dataset validation — the user-input boundary parser for golden.json."""

import json
from pathlib import Path

import pytest

from raggate.dataset import load_dataset

VALID = {
    "version": "1.0.0",
    "cases": [{"id": "a", "question": "q?", "expected": "yes"}],
}


def _write(tmp_path, data) -> Path:
    p = tmp_path / "golden.json"
    p.write_text(data if isinstance(data, str) else json.dumps(data))
    return p


def test_loads_valid_dataset(tmp_path):
    cases = load_dataset(_write(tmp_path, VALID))
    assert len(cases) == 1 and cases[0]["id"] == "a"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.json")


def test_invalid_json_raises(tmp_path):
    with pytest.raises(ValueError, match="invalid JSON"):
        load_dataset(_write(tmp_path, "{not json"))


def test_empty_cases_raises(tmp_path):
    with pytest.raises(ValueError, match="non-empty 'cases'"):
        load_dataset(_write(tmp_path, {"version": "1.0.0", "cases": []}))


@pytest.mark.parametrize("field", ["id", "question", "expected"])
def test_missing_required_field_raises(tmp_path, field):
    case = {"id": "a", "question": "q?", "expected": "yes"}
    del case[field]
    with pytest.raises(ValueError, match=f"missing required field '{field}'"):
        load_dataset(_write(tmp_path, {"cases": [case]}))


def test_duplicate_id_raises(tmp_path):
    dupe = {"cases": [
        {"id": "x", "question": "q1", "expected": "e1"},
        {"id": "x", "question": "q2", "expected": "e2"},
    ]}
    with pytest.raises(ValueError, match="duplicate case id 'x'"):
        load_dataset(_write(tmp_path, dupe))
