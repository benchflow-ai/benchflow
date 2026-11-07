"""Live utilization of a running job: the slots in use and the model's load.

Every :data:`SAMPLE_SEC` the watcher looks at the job's running trials (the
live-activity registry the dashboard reads): which phase each is in, and, for a
trial whose model proxy runs on this machine, how many model requests are in
flight. It logs one line after the first :data:`FIRST_REPORT_SEC`, then every
:data:`REPORT_SEC` and at the job's end, and warns when the run is underused:
slots empty while tasks wait, or agents that leave the model idle most of the
time. The second is the case of eight sessions on four GPUs with
one GPU busy, the agents spending their time in their own tools.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

from benchflow._utils import live_activity

logger = logging.getLogger("benchflow.utilization")

SAMPLE_SEC = 15.0
FIRST_REPORT_SEC = 120.0
REPORT_SEC = 300.0
# A job's last stretch gets a line when it held this much sampling.
_LAST_LINE_SEC = 60.0
# Below this share of agent time with a model request in flight, a model
# server sized for the sessions is mostly idle.
MODEL_BUSY_FLOOR = 0.4
# Below this share of the slots running while tasks wait, slots sit empty.
SLOT_FLOOR = 0.75

_SETUP_PHASES = frozenset({"created", "setup", "started"})
# The agent at work: an ACP agent connected, or an oracle, nop or script agent
# running in its installed sandbox (it never connects).
_AGENT_PHASES = frozenset({"installed", "connected"})
# Where a model request can be in flight: an ACP agent's session.
_MODEL_PHASES = frozenset({"connected"})
# The last report, for the TTY dashboard, which holds log lines back until the
# job ends: (line, warnings).
_latest: tuple[str, list[str]] | None = None


@dataclass
class Window:
    """Sums over the samples since the last report."""

    samples: int = 0
    running: int = 0
    setup: int = 0
    agent: int = 0
    # Running but neither: grading, or cleaning up.
    other: int = 0
    connected: int = 0
    waiting: int = 0
    # Connected trials whose proxy's in-flight count could be read, those
    # with a request in flight, and the requests in flight.
    readable: int = 0
    busy: int = 0
    in_flight: int = 0

    def add(self, trials: list[tuple[str, int | None]], waiting: int) -> None:
        self.samples += 1
        self.running += len(trials)
        self.waiting += waiting
        for phase, in_flight in trials:
            if phase in _SETUP_PHASES or phase.startswith("image-"):
                self.setup += 1
            if phase in _AGENT_PHASES:
                self.agent += 1
            elif not (phase in _SETUP_PHASES or phase.startswith("image-")):
                self.other += 1
            if phase not in _MODEL_PHASES:
                continue
            self.connected += 1
            if in_flight is not None:
                self.readable += 1
                self.in_flight += in_flight
                self.busy += in_flight > 0


def summarize(
    window: Window, slots: int, minutes: float, *, first: bool = False
) -> tuple[str, list[str]]:
    """The report line, and the warnings it carries (empty when all is well).
    ``first``: the job's first window, which setup fills by nature."""
    n = max(1, window.samples)
    running, waiting = window.running / n, window.waiting / n
    setup, agent, other = window.setup / n, window.agent / n, window.other / n
    phases = f"{setup:.1f} in setup, {agent:.1f} with the agent"
    if other >= 0.05:
        phases += f", {other:.1f} grading or cleaning up"
    parts = [
        f"{running:.1f} of {slots} slot{'' if slots == 1 else 's'} running ({phases})",
        f"{waiting:.0f} task{'' if round(waiting) == 1 else 's'} waiting",
    ]
    warnings: list[str] = []
    setup_bound = running >= 1 and setup >= 0.5 * running
    if window.readable:
        share = window.busy / window.readable
        parts.append(
            f"a model request in flight {share:.0%} of the agents' time "
            f"(averaging {window.in_flight / n:.1f} at once over the window)"
        )
        if share < MODEL_BUSY_FLOOR:
            warnings.append(
                f"the agents had a model request in flight only {share:.0%} of their "
                "time, so the model server waits on them (their own tools, or the "
                "client between calls)"
                + (
                    ""
                    if setup_bound
                    else "; if the server is yours and has room, more sessions "
                    "(--concurrency) would use it"
                )
            )
    elif window.connected:
        parts.append(
            "model load not visible (no BenchFlow model proxy on this machine)"
        )
    if setup_bound and not first:
        warnings.append(
            f"{setup:.1f} of the {running:.1f} running slots were in setup: image "
            "builds, agent installs or a busy Docker daemon hold the run back"
        )
    if waiting >= 1 and running < SLOT_FLOOR * slots:
        warnings.append(
            f"only {running:.1f} of {slots} slots ran while tasks waited: look at "
            "what holds starts back (image builds, a budget or the start pacer)"
        )
    line = (
        f"Utilization, last {minutes:.0f} min ({window.samples} samples): "
        + "; ".join(parts)
    )
    return line, warnings


def latest() -> tuple[str, list[str]] | None:
    """The last report line and its warnings, or None before the first."""
    return _latest


def _report(window: Window, slots: int, minutes: float, *, first: bool = False) -> None:
    global _latest
    line, warnings = summarize(window, slots, minutes, first=first)
    _latest = (line, warnings)
    if warnings:
        logger.warning(line + ". Underused: " + "; ".join(warnings))
    else:
        logger.info(line)


async def watch(
    slots: int,
    waiting: Callable[[], int],
    *,
    sample_sec: float = SAMPLE_SEC,
    first_report_sec: float = FIRST_REPORT_SEC,
    report_sec: float = REPORT_SEC,
    card: dict | None = None,
) -> None:
    """Sample and report until cancelled (the job's end cancels it); a job's
    last stretch since the previous line gets a line of its own when it held
    a minute of samples. At the first report with model requests on record,
    one line gives the sampling they carried against the run card's Vendor
    line and reference (:func:`benchflow.run_card.sampling_check`)."""
    global _latest
    _latest = None
    window, since, due = Window(), time.monotonic(), first_report_sec
    first, sampling_said = True, False
    try:
        while True:
            await asyncio.sleep(sample_sec)
            try:
                window.add(live_activity.utilization_sample(), max(0, waiting()))
            except Exception as exc:  # a report is never worth a failed run
                logger.debug("utilization sample failed: %s", exc)
                continue
            elapsed = time.monotonic() - since
            if elapsed < due or not window.samples:
                continue
            _report(window, slots, elapsed / 60, first=first)
            if not sampling_said:
                sampling_said = _say_sampling(card)
            window, since, due = Window(), time.monotonic(), report_sec
            first = False
    except asyncio.CancelledError:
        if window.samples * sample_sec >= _LAST_LINE_SEC:
            _report(window, slots, (time.monotonic() - since) / 60, first=first)
        raise


def _say_sampling(card: dict | None) -> bool:
    """Log the sampling the live trials' model requests carried so far, against
    the card; False while no request is on record yet."""
    from benchflow.report import _sampling
    from benchflow.run_card import sampling_check

    try:
        seen = _sampling(live_activity.trial_dirs())
        if not seen:
            return False
        line, differs = sampling_check(card or {}, seen)
    except Exception as exc:  # a report is never worth a failed run
        logger.debug("sampling check failed: %s", exc)
        return False
    (logger.warning if differs else logger.info)(line)
    return True
