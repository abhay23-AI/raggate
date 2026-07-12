"""raggate — a thin, CI-gated evaluation gate for RAG & LLM systems."""

from importlib.metadata import PackageNotFoundError, version

from .gates import GateResult, evaluate, has_kpi_failure
from .judge import JudgeError
from .runner import SuiteResult, TargetError, run_suite

try:
    # single source of truth: the installed package metadata (pyproject version)
    __version__ = version("raggate")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0+source"

__all__ = [
    "run_suite",
    "SuiteResult",
    "TargetError",
    "JudgeError",
    "evaluate",
    "has_kpi_failure",
    "GateResult",
    "__version__",
]
