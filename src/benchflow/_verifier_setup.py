"""How long each command that sets up the verifier may take.

Hardening the sandbox and preparing a separate verifier's runtime run their
commands under one budget per trial: the run's ``--verifier-setup-timeout``,
else the task's ``verifier.setup_timeout_sec``, else
:data:`VERIFIER_SETUP_TIMEOUT_SEC`. A busy machine took 208 s to set up
Terminal-Bench 4.1's cumulative-layout-shift separate verifier, past the
fixed 180 s before. Kept out of :mod:`benchflow.sandbox.lockdown` so the
rollout kernel, which imports no concrete plane, can settle it.
"""

from __future__ import annotations

import contextvars
import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

#: The budget when neither the run nor the task names one.
VERIFIER_SETUP_TIMEOUT_SEC = 180
#: The least budget a run or a task may name: the pytest plugin guard keeps
#: 60 s of each setup command for its install after the input proof
#: (lockdown._INPUT_PROOF_RESERVE_SEC), so 60 or less would leave the proof
#: no time at all.
MIN_VERIFIER_SETUP_TIMEOUT_SEC = 61

_verifier_setup_timeout: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "benchflow_verifier_setup_timeout", default=None
)


def verifier_setup_timeout_sec() -> int:
    """Seconds each verifier-setup command may take in this trial."""
    return _verifier_setup_timeout.get() or VERIFIER_SETUP_TIMEOUT_SEC


def task_verifier_setup_timeout(task: Any) -> int | None:
    """The task's ``verifier.setup_timeout_sec`` in whole seconds, or None."""
    verifier = getattr(getattr(task, "config", None), "verifier", None)
    value = getattr(verifier, "setup_timeout_sec", None)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return None
    return math.ceil(value)


def validate_verifier_setup_timeout(value: Any, *, where: str) -> int | None:
    """*value* as a run's verifier-setup budget: None, or a whole number of
    seconds of at least :data:`MIN_VERIFIER_SETUP_TIMEOUT_SEC`. Anything else
    (``0``, ``-5``, ``"900"``, ``900.5``, infinity) raises ValueError naming
    *where*, instead of running at the default."""
    if value is None:
        return None
    whole = (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else int(value)
        if isinstance(value, float) and math.isfinite(value) and value.is_integer()
        else None
    )
    if whole is None or whole < MIN_VERIFIER_SETUP_TIMEOUT_SEC:
        raise ValueError(
            f"{where} must be a whole number of seconds, at least "
            f"{MIN_VERIFIER_SETUP_TIMEOUT_SEC} (the pytest plugin guard keeps 60 s "
            f"of each verifier-setup command), not {value!r}"
        )
    return whole


def recorded_verifier_setup_timeout(config: Any) -> int | None:
    """The run's own value a trial's config.json records
    (``verifier_setup_timeout_run``), for a regrade or a recovery to pass on
    as the run's; None when the run named none, so the task's value, as it
    is now, applies. Never the budget used (``verifier_setup_timeout``),
    which may have been the task's or the default."""
    value = (
        config.get("verifier_setup_timeout_run") if isinstance(config, dict) else None
    )
    try:
        return validate_verifier_setup_timeout(value, where="verifier_setup_timeout")
    except ValueError:
        return None


def apply_verifier_setup_timeout(task: Any, run_seconds: int | None) -> int:
    """Settle a trial's verifier-setup budget, the run's *run_seconds* over
    the task's ``verifier.setup_timeout_sec`` over the default, and return it.

    The run's value is written into the task's config, the view every verifier
    step reads (a separate verifier's included), and the budget holds for the
    rest of the calling task: the rollout calls this once, at setup.
    """
    if run_seconds is not None:
        task.config.verifier.setup_timeout_sec = run_seconds
        # A multi-step task's step views take their verifier settings over
        # the task's: the run's value stays the one that wins (task/steps.py).
        task._benchflow_run_verifier_setup_timeout = run_seconds
    _verifier_setup_timeout.set(task_verifier_setup_timeout(task))
    return verifier_setup_timeout_sec()


@contextmanager
def verifier_setup_timeout_for(task: Any) -> Iterator[None]:
    """While the block runs, the task's own verifier-setup budget when it
    names one; otherwise whatever the trial set."""
    seconds = task_verifier_setup_timeout(task)
    if seconds is None:
        yield
        return
    token = _verifier_setup_timeout.set(seconds)
    try:
        yield
    finally:
        _verifier_setup_timeout.reset(token)
