"""The shared timing check itself (tests/_timing.py;, the lead's
ruling, option (a)): each round's larger size is judged against the smaller
timed just before it, and the best round counts. A load spike in one sample,
or a quiet small paired with a busy large, does not fail it; a quadratic
term in every round does. A fake clock gives each call its scripted time.
What the check returns is plain built-in types, which xdist can send.
"""

from __future__ import annotations

import pytest

from tests import _timing


def _judged(durations: list[float], warm: bool = False) -> list[list[float]]:
    """The rounds of a call whose successive times are *durations* (small,
    large, small, large, ...), on a fake clock; with *warm*, the first two
    are the untimed warm-up calls."""
    now = [0.0]
    script = iter(durations)

    def clock() -> float:
        return now[0]

    def call(text: str) -> str:
        now[0] += next(script)
        return text

    samples, _ = _timing.linear_rounds(
        call, lambda size: "x" * size, (1, 4), len(durations) // 2 - warm, clock, warm
    )
    return samples


def _judge(samples: list[list[float]]) -> None:
    _timing.assert_linear(_timing.best_round(samples))


def test_a_load_spike_in_one_sample_does_not_fail_the_check():
    """A spike on one large sample; and the 18-of-20 shape: a quiet small in
    a round whose large is busy, every other round busy in both. The best
    of each size taken apart reads 8.0 against 1.0 there; each round's own
    pair reads 8 once and 4 elsewhere."""
    _judge(_judged([1.0, 9.0] + [1.0, 4.0] * 6))
    samples = _judged([1.0, 8.0] + [2.0, 8.0] * 6)
    apart = [min(column) for column in zip(*samples, strict=True)]
    assert apart == [1.0, 8.0]
    assert apart[1] > _timing.RATIO * apart[0] + _timing.FLOOR
    _judge(samples)
    assert _timing.best_round(samples) == [2.0, 8.0] and len(samples) == 7


def test_a_quadratic_term_in_every_round_fails_the_check():
    """Sixteen times in every round, with load on either sample of some
    rounds (a busy small, a busy large, both): no round passes. Every sample
    is under the ceiling, so the ratio alone fails it (review R192B: with
    samples of 16-32 s the ceiling did, and a ratio of 50 still failed)."""
    samples = _judged(
        [0.1, 1.6, 0.13, 1.7, 0.2, 3.2, 0.11, 2.1, 0.1, 1.65, 0.12, 1.9, 0.1, 1.6]
    )
    assert max(large for _, large in samples) < _timing.CEILING
    with pytest.raises(AssertionError):
        _judge(samples)


def test_a_first_use_in_round_one_cannot_hide_a_quadratic():
    """Review R192B, L-2: a check that is the first call in its worker pays
    for patterns compiled on demand in its first small sample (R192B's run 1:
    rounds [0.299, 0.051], [0.015, 0.079], [0.02, 0.075], ...), and the best
    round was then that one. With the larger size on a sixteen-times slope
    (four times those ratios), the first round passes it; after one untimed
    call per size, no round does."""
    cold = [0.299, 0.051 * 4]
    warm_rounds = [0.015, 0.079 * 4, 0.02, 0.075 * 4] * 3 + [0.015, 0.08 * 4]
    without = _judged(cold + warm_rounds)
    assert _timing.best_round(without) == pytest.approx(cold)
    _judge(without)
    with pytest.raises(AssertionError):
        _judge(_judged(cold + warm_rounds, warm=True))


def test_the_check_fails_a_real_quadratic_call_and_passes_a_linear_one():
    """On real calls: a scan of the rest of the text from every 32nd
    character fails it at 50,000 and 200,000 characters, in every round; a
    single scan passes; and the positive control finds the quadratic term
    it adds."""

    def quadratic(text: str) -> str:
        for start in range(0, len(text), 32):
            text.find("\x00\x01", start)
        return text

    def make(size: int) -> str:
        return "a_key_2_b_token_v2_" * (size // 19)

    samples, _ = _timing.linear_rounds(quadratic, make)
    assert all(large > _timing.RATIO * small for small, large in samples)
    seconds, _ = _timing.linear_seconds(quadratic, make)
    with pytest.raises(AssertionError):
        _timing.assert_linear(seconds)
    linear, _ = _timing.linear_seconds(lambda text: text.count("\x00"), make)
    _timing.assert_linear(linear)
    assert _timing.catches_quadratic(lambda text: text.count("\x00"), make)


def test_the_ceiling_still_holds():
    """A larger size past 10 s fails, however the ratio reads."""
    with pytest.raises(AssertionError):
        _judge(_judged([3.0, 11.0] * 7))


def _plain(value: object) -> bool:
    """Whether *value* is made only of list, tuple, dict, float, int, str,
    bool and None (the exact types, no subclass)."""
    if value is None or type(value) in (bool, int, float, str):
        return True
    if isinstance(value, list | tuple) and type(value) in (list, tuple):
        return all(_plain(item) for item in value)
    if isinstance(value, dict) and type(value) is dict:
        return all(_plain(k) and _plain(v) for k, v in value.items())
    return False


def _sendable(value: object) -> bool:
    """Whether xdist can send *value* from a worker (execnet's dumps takes
    it), or, without execnet, whether it is plain."""
    try:
        from execnet.gateway_base import dumps
    except ImportError:
        return _plain(value)
    try:
        dumps(value)
    except Exception:
        return False
    return True


def test_what_a_timing_test_records_can_be_sent_by_xdist():
    """F2b, 11:1xZ: a list subclass recorded with record_property killed an
    xdist session. What the check returns, and so what a timing test
    records, is plain built-in types that execnet sends; a list subclass is
    not."""
    seconds, _ = _timing.linear_seconds(
        lambda text: text, lambda size: "x" * size, (1, 4)
    )
    samples, _ = _timing.linear_rounds(
        lambda text: text, lambda size: "x" * size, (1, 4)
    )
    assert type(seconds) is list and all(type(value) is float for value in seconds)
    assert type(samples) is list and all(type(pair) is list for pair in samples)
    for value in (seconds, samples, _timing.best_round(samples)):
        assert _plain(value) and _sendable(value), value

    class Subclass(list):
        pass

    assert not _plain(Subclass(seconds)) and not _sendable(Subclass(seconds))
