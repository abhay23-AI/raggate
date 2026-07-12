"""raggate — a thin, CI-gated evaluation gate for RAG & LLM systems."""

from .gates import GateResult, evaluate, has_kpi_failure
from .runner import SuiteResult, TargetError, run_suite

__version__ = "0.1.0"

__all__ = [
    "run_suite",
    "SuiteResult",
    "TargetError",
    "evaluate",
    "has_kpi_failure",
    "GateResult",
    "__version__",
]
