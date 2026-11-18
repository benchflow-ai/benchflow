"""The one timing check of the test suite (the lead's ruling,
2026-11-14).

Four times the text takes about four times as long where matching is
linear, and sixteen times where it is quadratic. A timing test times its
call on two sizes with linear_seconds: this process's CPU time (waiting for
a core on a busy machine does not count), the sizes taken in turn, seven
rounds (linear_rounds). Each round is judged on its own, the larger size
against the smaller timed just before it, and the best round counts
(linear_seconds gives its pair, best_round): assert_linear allows the larger at
most five times the smaller, plus a 0.05 s floor for times too small to
measure on a busy machine, and never 10 s. Two samples side by side see the
same load; the best of each size taken apart could pair a quiet small with
a busy large (the lead's ruling, option (a), after 18 of 20 loaded runs;
retrying a failed check stays rejected). No tighter bound in seconds, as a
loaded machine is slower, not worse at scaling. A test that times one call holds it to the ceiling
(within_ceiling). catches_quadratic is the positive control: the same check
on the same inputs, with a term quadratic in the text's length added to the
call, must fail.

Every timing test calls these; no test file asserts on a timer of its own
(tests/test_timing_helper.py).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

ROUNDS = 7
RATIO = 5
FLOOR = 0.05
CEILING = 10.0


def linear_rounds(
    call: Callable[[Any], Any],
    make: Callable[[int], Any],
    sizes: Sequence[int] = (50_000, 200_000),
    rounds: int = ROUNDS,
    clock: Callable[[], float] = time.process_time,
    warm: bool = True,
) -> tuple[list[list[float]], list[Any]]:
    """*rounds* CPU timings of *call* on ``make(size)`` for each size, the
    sizes in turn every round: one plain list of floats per round, the
    sizes in order; and the outputs of the last round. First, unless
    *warm* is false, one untimed call per size: a first use (patterns
    compiled on demand, the first call in a worker) is no round's, so the
    best round cannot be one whose smaller size carried it (review R192B,
    L-2). *clock* is for this module's own tests."""
    texts = [make(size) for size in sizes]
    if warm:
        for text in texts:
            call(text)
    samples: list[list[float]] = []
    outs: list[Any] = [None] * len(texts)
    for _ in range(rounds):
        row: list[float] = []
        for index, text in enumerate(texts):
            start = clock()
            outs[index] = call(text)
            row.append(float(clock() - start))
        samples.append(row)
    return samples, outs


def best_round(samples: list[list[float]]) -> list[float]:
    """The round whose larger size reads best against the smaller timed
    just before it (the most room under the ratio), as a plain list."""
    return list(min(samples, key=lambda pair: pair[1] - RATIO * pair[0]))


def linear_seconds(
    call: Callable[[Any], Any],
    make: Callable[[int], Any],
    sizes: Sequence[int] = (50_000, 200_000),
    rounds: int = ROUNDS,
    clock: Callable[[], float] = time.process_time,
    warm: bool = True,
) -> tuple[list[float], list[Any]]:
    """The best round of linear_rounds, its smaller and larger size timed
    side by side, as a plain list of floats (a test may record it: xdist
    sends only built-in types); and the outputs of the last round."""
    samples, outs = linear_rounds(call, make, sizes, rounds, clock, warm)
    return best_round(samples), outs


def assert_linear(seconds: Sequence[float]) -> None:
    """The larger size, four times the smaller and timed beside it, takes at
    most five times as long, plus the floor, and less than the ceiling."""
    small, large = seconds
    assert large <= RATIO * small + FLOOR, seconds
    assert large < CEILING, seconds


def within_ceiling(call: Callable[..., Any], *args: Any) -> Any:
    """*call* on *args*, which must take less than the ceiling in CPU time;
    its result. Never under xdist, where a neighbour on the same core inflates
    CPU time (the lead's ruling, 2026-11-18): a test that calls it carries the
    ``ceiling`` mark, which conftest leaves out of an xdist run and CI runs
    with -n 0."""
    import os

    if os.environ.get("PYTEST_XDIST_WORKER"):
        raise AssertionError(
            "within_ceiling ran under xdist: mark the test `ceiling` and run it with -n 0"
        )
    start = time.process_time()
    out = call(*args)
    took = time.process_time() - start
    assert took < CEILING, took
    return out


def _burn(seconds: float) -> None:
    """Spend *seconds* of this process's CPU time."""
    end = time.process_time() + seconds
    while time.process_time() < end:
        pass


def catches_quadratic(
    call: Callable[[Any], Any],
    make: Callable[[int], Any],
    sizes: Sequence[int] = (50_000, 200_000),
) -> bool:
    """The positive control: whether assert_linear fails on *call* once a
    term quadratic in the text's length is added to it, as large at the
    smaller size as *call*'s own cost there plus 0.01 s (CPU time spent, so
    a busy machine cannot hide it). Linear, the larger takes four times the
    smaller; with the term about ten times. One round: noise only adds.
    The call before the term is measured is its warm-up (a pattern compiled
    on demand is compiled once, whatever the size)."""
    small = make(sizes[0])
    start = time.process_time()
    call(small)
    term = time.process_time() - start + 0.01

    def slowed(text: Any) -> Any:
        out = call(text)
        _burn(term * (len(text) / len(small)) ** 2)
        return out

    seconds, _ = linear_seconds(slowed, make, sizes, rounds=1, warm=False)
    try:
        assert_linear(seconds)
    except AssertionError:
        return True
    return False
