"""Markdown summary (written to $GITHUB_STEP_SUMMARY) + CLI wiring."""

from pathlib import Path

from raggate.cli import main
from raggate.gates import FAIL, PASS, GateResult
from raggate.report import markdown

EVALS = Path(__file__).resolve().parent.parent / "evals"


def test_markdown_summary_table_and_status():
    results = [
        GateResult("faithfulness", 0.91, PASS, True, 0.90),
        GateResult("citation_support", None, "N/A", True, 0.85),
        GateResult("answer_relevancy", 0.40, FAIL, True, 0.80),
    ]
    md = markdown(results, "openai:gpt-4.1-mini@t0", 6, passed=True)
    assert "✅ **passed**" in md
    assert "| `faithfulness` |" in md and "0.910" in md
    assert "🟢" in md and "🔴" in md

    heur = markdown(results, "heuristic", 6, passed=False)
    assert "❌ **failed**" in heur
    assert "informational" in heur


def test_cli_writes_github_step_summary(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rc = main(["run", "--dir", str(EVALS)])
    assert rc == 0
    text = summary.read_text()
    assert "raggate — RAG/LLM eval gate" in text
    assert "| `faithfulness` |" in text
