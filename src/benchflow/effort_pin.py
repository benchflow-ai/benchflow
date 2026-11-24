"""The opt-in effort pin (benchflow#1246).

A sub-agent can run at another reasoning effort than the run's: Claude Code's
Agent tool takes an ``effort`` argument, Codex's ``spawn_agent`` a
``reasoning_effort``. With the pin on, the model proxy's request guard
(``providers/litellm_request_guard.py``) rewrites each generation request's
effort to the run's, and says on the request's ``request_end`` record what it
asked for and what went upstream; ``model-calls.json`` counts the rewrites.
Off (the default), nothing is rewritten and the requests whose effort differs
from the run's are counted.

The run turns it on with ``--effort-pin`` or ``effort_pin: true`` in a run
config; either sets :data:`EFFORT_PIN_ENV` to ``1`` in the run's agent env,
which every launch path and the proxy's own environment carry. Before the
proxy starts, :func:`benchflow.effort_request.with_route_effort` replaces the
``1`` with the run's level, the value the guard pins to.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from benchflow._utils.config import EFFORT_LEVELS

EFFORT_PIN_ENV = "BENCHFLOW_EFFORT_PIN"
#: The run's switch, before the proxy is given the level.
SWITCH_ON = "1"
_ON = frozenset({"1", "true", "yes", "on"})
_OFF = frozenset({"", "0", "false", "no", "off"})


def effort_pin_setting(env: Mapping[str, str]) -> str | None:
    """What the run's env says of the pin: None when off, :data:`SWITCH_ON`
    when the run turned it on, or the level the proxy is to pin to.
    Raises ValueError for any other value, so a typo never runs unpinned."""
    raw = env.get(EFFORT_PIN_ENV)
    value = str(raw).strip().lower() if raw is not None else ""
    if value in _OFF:
        return None
    if value in _ON:
        return SWITCH_ON
    if value in EFFORT_LEVELS:
        return value
    raise ValueError(
        f"{EFFORT_PIN_ENV}={raw!r}: use 1 (or --effort-pin) to pin each model "
        "call's reasoning effort to the run's, or 0 for no pin"
    )


def effort_pin_on(env: Mapping[str, str]) -> bool:
    """Whether the run pins its model calls' effort."""
    return effort_pin_setting(env) is not None


def trial_run_effort(trial: Path) -> str | None:
    """The reasoning effort a trial's run asked for, as its config.json
    recorded it: what ``model-calls.json`` compares each request's effort
    with (``effort_mismatches``). None when it asked for none."""
    try:
        config = json.loads((trial / "config.json").read_text())
    except (OSError, ValueError):
        return None
    effort = config.get("reasoning_effort") if isinstance(config, dict) else None
    return effort if isinstance(effort, str) else None
