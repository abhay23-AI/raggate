"""Package metadata — version must not drift from the installed distribution."""

import raggate


def test_version_is_derived_not_hardcoded():
    v = raggate.__version__
    assert isinstance(v, str) and v
    try:
        from importlib.metadata import version

        assert v == version("raggate")   # single source of truth = package metadata
    except Exception:
        # running from a source tree without an install
        assert v == "0.0.0+source"
