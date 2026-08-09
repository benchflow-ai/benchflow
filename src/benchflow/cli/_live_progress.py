"""Live terminal dashboard for ``bench eval run`` runs.

Renders a single Rich :class:`~rich.live.Live` panel — a progress bar with ETA,
queued/running/passed/failed/errored counts, a "running now" table, and running
token / cost / pass-rate totals — fed by the :class:`~benchflow.evaluation.Evaluation`
engine's ``on_plan`` / ``on_task_start`` / ``on_result`` hooks, plus render-time
polls of the live-activity registry for the running tasks' in-flight session
counters (activity cells and the footer's live token sum).

TTY-only by contract: the CLI keeps its plain ``logger.info`` lines when stdout
isn't a terminal (CI, pipes, parity files), so machine-readable output is never
polluted with cursor escapes. The display is purely additive — every mutator is
cheap and lock-guarded, and the engine fires the hooks best-effort, so a render
bug can never perturb or abort a run.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.markup import escape
from rich.table import Table
from rich.text import Text

from benchflow._utils import live_activity
from benchflow.cli._shared import console
from benchflow.usage_tracking import is_trusted_usage_source

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.console import Console, RenderableType

    from benchflow.models import RunResult

_BAR_WIDTH = 30
_MAX_RUNNING_ROWS = 12
_DISABLE_ENV = "BENCHFLOW_NO_PROGRESS"


def progress_enabled(console: Console) -> bool:
    """Live dashboard only on a real TTY, and only when not opted out.

    Non-terminal stdout (CI, pipes, parity files) keeps the plain ``logger.info``
    stream so machine-readable output is never polluted with cursor escapes;
    ``BENCHFLOW_NO_PROGRESS=1`` forces that path too.
    """
    if os.environ.get(_DISABLE_ENV, "").strip() not in ("", "0", "false", "False"):
        return False
    return bool(getattr(console, "is_terminal", False))


@contextlib.contextmanager
def quiet_root_logging() -> Iterator[None]:
    """Mute INFO chatter during a live display, but buffer + replay WARNING+.

    The engine streams ``logger.info`` lines to stderr during a run; a Live panel
    repainting stdout would be shredded by them, so INFO/DEBUG are dropped while
    the dashboard owns the screen. But the engine's *batch-level reliability
    verdicts* (">20% verifier errors — results may be unreliable", the
    verifier-error summary, circuit-breaker trips) are WARNING/ERROR and must NOT
    vanish — a 100%-verifier-error run looking like a normal red score line is a
    correctness-of-conclusions hazard. So WARNING+ records are captured and
    replayed to stderr after the Live exits. Handlers are restored even on raise.
    """
    root = logging.getLogger()
    saved = root.handlers[:]
    buffer = _WarningBuffer()
    root.handlers = [buffer]
    try:
        yield
    finally:
        root.handlers = saved
        buffer.replay()


class _WarningBuffer(logging.Handler):
    """Capture WARNING+ records during a Live; drop INFO/DEBUG; replay on exit."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self._records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self._records.append(record)

    def replay(self) -> None:
        for record in self._records:
            style = "red" if record.levelno >= logging.ERROR else "yellow"
            # escape(): buffered warnings carry user-derived text (a malformed
            # task dir name, a resume agent-mismatch from config.json) that can
            # contain Rich markup — an unescaped replay raises MarkupError and
            # crashes the CLI on live-context exit (and silently swallows
            # balanced [tokens]), defeating the buffer's "warnings must survive
            # the Live" purpose.
            console.print(f"[{style}]{escape(record.getMessage())}[/{style}]")


@contextlib.contextmanager
def live_session(live: LiveEvalProgress) -> Iterator[None]:
    """Run a block under the live dashboard with logging quieted.

    Combines :func:`quiet_root_logging` and the ``Live`` so callers have a single
    context to wrap the blocking ``Evaluation.run()`` — and a single, non-
    duplicated ``Evaluation(...)`` construction at the call site.
    """
    with quiet_root_logging(), live:
        yield


def _fmt_dur(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


# Rollout lifecycle phase (Rollout._phase — the completed phase, except
# "verifying" which verify() marks at entry) -> what is running now. Shown
# while the session counters are unavailable — before the agent session
# exists and after it is torn down — so sandbox create, agent install, and
# the verifier read as work, not as a hang.
_PHASE_LABELS = {
    "created": "creating sandbox…",
    "setup": "creating sandbox…",
    "started": "installing agent…",
    "installed": "running agent…",
    "connected": "running agent…",
    "executed": "verifying…",
    "verifying": "verifying…",
    "verified": "cleaning up…",
    "cleaned": "cleaning up…",
}
# Unknown phases (e.g. "branched") and pre-register/teardown races: still
# never blank — a blank cell on a live row reads as a hang.
_PHASE_FALLBACK = "starting…"


def _activity_cell(snap: live_activity.ActivitySnapshot | None) -> str:
    """Per-task activity for the "running now" table, e.g.
    ``38 calls · last: file_editor``.

    Pure formatter over one polled :class:`~benchflow._utils.live_activity.
    ActivitySnapshot` — ``__rich__`` polls the registry exactly once per
    running task per frame and shares the snapshots with the footer's live
    token sum. The counters are the same ones the console heartbeat logs,
    which the Live display otherwise mutes. Until the agent session exists
    (and after it is closed for the verifier) the cell shows the rollout's
    lifecycle phase as a label instead, and a registry miss (None) shows the
    fallback label, so the row is never blank. Tokens appear as soon as
    either live signal exists — the gateway's live callback capture (per
    completed LLM request, i.e. mid-prompt) or the ACP usage snapshot (per
    completed prompt); ``Rollout.activity_snapshot`` reconciles the two as
    max(). Once tokens are present, a single-tool agent's cell is
    ``38 calls · 412.0k tok`` (no ``last:``).
    """
    if snap is None:
        return _PHASE_FALLBACK
    counters = snap.counters
    if counters is None:
        return _PHASE_LABELS.get(snap.phase, _PHASE_FALLBACK)
    cell = f"{counters.tool_calls} calls"
    if counters.total_tokens:
        cell += f" · {_fmt_tokens(counters.total_tokens)} tok"
    # A single-tool agent (prime-agent funnels every call through one IPython
    # tool) makes "last: <name>" a constant: once tokens carry the signal,
    # drop it. Varied tool names — and unknown distinct counts (0), and a
    # first lone call, where the name is still news — keep the last: suffix.
    constant_tool = counters.tool_calls > 1 and counters.distinct_tools == 1
    if counters.last_tool and not (constant_tool and counters.total_tokens):
        cell += f" · last: {counters.last_tool[:30]}"
    return cell


class LiveEvalProgress:
    """A live ``Live``-rendered dashboard, driven by the engine's progress hooks.

    Use as a context manager around the blocking ``Evaluation.run()`` call and
    pass the three bound methods as the engine's ``on_plan`` / ``on_task_start`` /
    ``on_result`` callbacks. The panel re-renders on a timer (so elapsed/ETA tick
    between events) by reading lock-guarded state in :meth:`__rich__`.
    """

    def __init__(
        self,
        console: Console,
        *,
        label: str,
        agent: str,
        model: str | None,
        sandbox: str,
    ) -> None:
        self._console = console
        self._label = label
        self._agent = agent
        self._model = model or "(default)"
        self._sandbox = sandbox
        self._lock = threading.Lock()

        self._total = 0
        self._resumed = 0  # already-complete on resume; counted as done, not run
        self._to_run = 0
        self._run_start = time.monotonic()

        self._passed = 0
        self._failed = 0
        self._errored = 0
        self._running: dict[str, float] = {}  # name -> monotonic start

        self._tokens = 0
        self._cost = 0.0
        self._completed = 0  # finished this run (for telemetry coverage)
        self._covered = 0  # finished with trusted token telemetry

        self._live: Live | None = None

    # -- engine hooks -------------------------------------------------------

    def on_plan(
        self,
        total: int,
        done: int,
        remaining: int,
        resumed_outcomes: tuple[int, int, int] = (0, 0, 0),
    ) -> None:
        # Seed the pass/fail/errored counters with the RESUMED tasks' outcomes so
        # the counts row and pass-rate footer are correct over the whole job, not
        # just this process's new tasks. resumed_outcomes = (passed, failed,
        # errored). _completed stays this-run-only (drives the ETA rate).
        with self._lock:
            self._total = total
            self._resumed = done
            self._to_run = remaining
            self._run_start = time.monotonic()
            self._passed, self._failed, self._errored = resumed_outcomes

    def on_task_start(self, name: str) -> None:
        with self._lock:
            self._running[name] = time.monotonic()

    def on_result(self, name: str, result: RunResult) -> None:
        with self._lock:
            self._running.pop(name, None)
            # Mirror Evaluation._log_and_report exactly: reward==1 -> PASS,
            # reward not None -> FAIL, else ERR (no reward reached).
            rewards = getattr(result, "rewards", None)
            reward = rewards.get("reward") if rewards else None
            if reward == 1:
                self._passed += 1
            elif reward is not None:
                self._failed += 1
            else:
                self._errored += 1

            self._completed += 1
            tokens = getattr(result, "total_tokens", None)
            cost = getattr(result, "cost_usd", None)
            if is_trusted_usage_source(getattr(result, "usage_source", None)):
                self._covered += 1
                if tokens:
                    self._tokens += int(tokens)
                if cost:
                    self._cost += float(cost)

    # -- context manager ----------------------------------------------------

    def __enter__(self) -> LiveEvalProgress:
        self._live = Live(
            self,
            console=self._console,
            auto_refresh=True,
            refresh_per_second=4,
            transient=False,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)
            self._live = None

    # -- rendering ----------------------------------------------------------

    def __rich__(self) -> Group:
        with self._lock:
            total = self._total
            resumed = self._resumed
            to_run = self._to_run
            passed, failed, errored = self._passed, self._failed, self._errored
            running = dict(self._running)
            tokens, cost = self._tokens, self._cost
            completed, covered = self._completed, self._covered
            elapsed = time.monotonic() - self._run_start

        # passed/failed/errored already include the resumed-seeded outcomes, so
        # "done" is their sum — adding `resumed` again would double-count.
        done = passed + failed + errored
        queued = max(total - done - len(running), 0)

        header = Text()
        header.append("benchflow", style="bold cyan")
        header.append(f"  ·  {self._label}  ·  {self._agent}", style="dim")
        header.append(f"  ·  {self._model}  ·  {self._sandbox}", style="dim")

        # Progress bar + ETA (computed from this run's finish rate).
        frac = (done / total) if total else 0.0
        filled = int(frac * _BAR_WIDTH)
        bar = Text()
        bar.append("━" * filled, style="green")
        bar.append("━" * (_BAR_WIDTH - filled), style="grey37")
        # ETA from THIS run's finish rate (completed excludes instant resumed).
        rate = completed / elapsed if elapsed > 0 and completed else 0.0
        eta = (to_run - completed) / rate if rate > 0 else None
        bar.append(f"  {done}/{total}", style="bold")
        bar.append(f" · {frac * 100:.0f}%", style="dim")
        bar.append(f" · {_fmt_dur(elapsed)}", style="dim")
        if eta is not None:
            bar.append(f" · ETA {_fmt_dur(eta)}", style="dim")
        if resumed:
            bar.append(f" · {resumed} resumed", style="dim")

        counts = Text()
        counts.append(f"✓ {passed} passed", style="green")
        counts.append("   ")
        counts.append(f"✗ {failed} failed", style="red" if failed else "dim")
        counts.append("   ")
        counts.append(f"⚠ {errored} errored", style="yellow" if errored else "dim")
        counts.append("   ")
        counts.append(f"◷ {len(running)} running", style="cyan")
        counts.append("   ")
        counts.append(f"⋯ {queued} queued", style="dim")

        parts: list[RenderableType] = [header, bar, counts]

        # ONE registry poll per running task per frame (each poll an O(1)
        # counter read; see live_activity), shared by the running-now cells
        # and the footer's live token sum — displayed rows are never polled
        # twice, and cell vs footer can't read different snapshots within a
        # frame.
        snaps = {name: live_activity.activity(name) for name in running}

        # "Running now" — cap rows so short terminals don't overflow.
        if running:
            tbl = Table(
                show_edge=True, show_header=True, header_style="dim", expand=False
            )
            tbl.add_column("running now", no_wrap=True)
            tbl.add_column("elapsed", justify="right")
            tbl.add_column("activity", no_wrap=True)
            now = time.monotonic()
            for name in sorted(running, key=lambda n: running[n])[:_MAX_RUNNING_ROWS]:
                # Text() so a task name containing Rich markup (`[` is legal in
                # SkillsBench dir names) is rendered literally, not parsed as
                # markup — a MarkupError here escapes __rich__ and aborts the
                # CLI on live-context exit, violating this module's "a render
                # bug can never perturb a run" contract. Same for the activity
                # cell, which carries agent-authored tool titles.
                tbl.add_row(
                    Text(name),
                    _fmt_dur(now - running[name]),
                    Text(_activity_cell(snaps.get(name)), style="dim"),
                )
            extra = len(running) - _MAX_RUNNING_ROWS
            if extra > 0:
                tbl.add_row(f"… {extra} more", "", "")
            parts.append(tbl)

        # Footer: pass-rate (excl errors) + token/cost economics. Tokens sum
        # the completed tasks' trusted telemetry PLUS the running rollouts'
        # live usage — per running task, max(ACP prompt-completion snapshot,
        # gateway live-capture total): the gateway side steps forward per
        # completed LLM *request*, so a single-prompt agent phase shows its
        # spend while the prompt runs, not only at scoring (the ACP side
        # alone stays empty until the prompt completes). Reconciliation
        # (documented at Rollout.activity_snapshot): max() keeps the running
        # figure monotonic whichever signal leads, and degrades to the
        # ACP-only value when the gateway capture is unavailable. The total
        # is still NOT monotonic across a task's completion boundary: scoring
        # swaps the live figure for the trusted gateway import (which can be
        # lower than either live signal — untracked discards, cache
        # accounting, ACP over-report), and a task completing with
        # untrusted/absent telemetry drops its live contribution entirely —
        # a sole-task footer can collapse back to "— tokens". Clamping would
        # falsify the trusted sum, so the dip is intended. Show "—" (not 0 /
        # $0.00) when neither signal exists, so a coverage-0 run reads
        # broken, not free — matching the summary.json contract. Cost stays
        # completed-tasks-only: $ comes from the sandbox LiteLLM gateway log
        # imported at scoring time (BenchFlow computes no prices of its own),
        # so there is no live $ to show.
        live_tokens = sum(
            snap.counters.total_tokens or 0
            for snap in snaps.values()
            if snap is not None and snap.counters is not None
        )

        footer = Text()
        scored = passed + failed
        if scored:
            footer.append(f"pass-rate {passed / scored * 100:.1f}% (excl-err)", "bold")
        else:
            footer.append("pass-rate —", style="dim")
        footer.append(" · ")
        footer.append(
            f"{_fmt_tokens(tokens + live_tokens)} tokens"
            if covered or live_tokens
            else "— tokens",
            "dim",
        )
        footer.append(" · ")
        footer.append(f"${cost:.2f}" if covered else "$—", style="dim")
        if completed:
            footer.append(f" · telemetry {covered / completed * 100:.0f}%", style="dim")
        parts.append(footer)

        return Group(*parts)
