"""parse_score — the score extractor that feeds the gate. Bugs here silently
produce wrong scores, so it gets thorough coverage."""

import pytest

from raggate.judge import Judge, parse_score


class _Boom:
    """A stand-in OpenAI client whose every call raises (bad key / no network)."""

    def __getattr__(self, _name):
        return self

    def create(self, **_kwargs):
        raise RuntimeError("boom")


def test_judge_tracks_degradation_when_all_calls_fail():
    j = Judge()
    j._client = _Boom()          # simulate a configured-but-broken judge
    assert j.rate("system", "user") is None
    assert j.calls >= 1
    assert j.errors == j.calls
    assert j.degraded is True


def test_judge_not_degraded_in_heuristic_mode():
    j = Judge()                  # no OPENAI_API_KEY -> heuristic
    assert j.rate("s", "u") is None
    assert j.degraded is False   # nothing was attempted; not a failure


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("0.67", 0.67),
        ("0.8", 0.8),
        ("1", 1.0),
        ("0", 0.0),
        # a decimal verdict after an incidental count must win (the C2 bug)
        ("Based on 3 claims, 2 supported: 0.67", 0.67),
        # explicit fractions
        ("1/10", 0.1),
        ("2/3", pytest.approx(0.6667, abs=1e-3)),
        ("Score: 8/10", 0.8),
        ("9 out of 10", 0.9),
        # bare integers on a 0..10 scale
        ("8", 0.8),
        ("Rating: 7", 0.7),
        # bare integers on a 0..100 scale
        ("85", 0.85),
        ("Rating: 42", 0.42),
        # percentages (a failing 10% must NOT read as a perfect 1.0)
        ("10%", 0.10),
        ("5%", 0.05),
        ("85%", 0.85),
        ("85.5%", pytest.approx(0.855)),
        ("100%", 1.0),
        # European decimal comma
        ("0,85", 0.85),
        ("1,0", 1.0),
        # clamping out-of-range
        ("1.5", 1.0),
        ("-0.5", 0.0),
        ("-1", 0.0),
        # no number
        ("the answer is fully correct", None),
        ("", None),
    ],
)
def test_parse_score(reply, expected):
    assert parse_score(reply) == expected


def test_parse_score_prefers_last_decimal_in_range():
    # reasoning that ends on the verdict
    assert parse_score("could be 0.9 but really it's 0.4") == 0.4
