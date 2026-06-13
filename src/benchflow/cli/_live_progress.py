"""Live terminal dashboard for ``bench eval create`` runs.

Renders a single Rich :class:`~rich.live.Live` panel — a progress bar with ETA,
queued/running/passed/failed/errored counts, a "running now" table, and running
token / cost / pass-rate totals — fed by the :class:`~benchflow.evaluation.Evaluation`
engine's ``on_plan`` / ``on_task_start`` / ``on_result`` hooks.

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
from rich.table import Table
from rich.text import Text

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
            console.print(f"[{style}]{record.getMessage()}[/{style}]")


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

        # "Running now" — cap rows so short terminals don't overflow.
        if running:
            tbl = Table(
                show_edge=True, show_header=True, header_style="dim", expand=False
            )
            tbl.add_column("running now", no_wrap=True)
            tbl.add_column("elapsed", justify="right")
            now = time.monotonic()
            for name in sorted(running, key=lambda n: running[n])[:_MAX_RUNNING_ROWS]:
                # Text() so a task name containing Rich markup (`[` is legal in
                # SkillsBench dir names) is rendered literally, not parsed as
                # markup — a MarkupError here escapes __rich__ and aborts the
                # CLI on live-context exit, violating this module's "a render
                # bug can never perturb a run" contract.
                tbl.add_row(Text(name), _fmt_dur(now - running[name]))
            extra = len(running) - _MAX_RUNNING_ROWS
            if extra > 0:
                tbl.add_row(f"… {extra} more", "")
            parts.append(tbl)

        # Footer: pass-rate (excl errors) + token/cost economics. Show "—" (not
        # 0 / $0.00) when no trusted telemetry, so a coverage-0 run reads broken,
        # not free — matching the summary.json contract.
        footer = Text()
        scored = passed + failed
        if scored:
            footer.append(f"pass-rate {passed / scored * 100:.1f}% (excl-err)", "bold")
        else:
            footer.append("pass-rate —", style="dim")
        footer.append(" · ")
        footer.append(f"{_fmt_tokens(tokens)} tokens" if covered else "— tokens", "dim")
        footer.append(" · ")
        footer.append(f"${cost:.2f}" if covered else "$—", style="dim")
        if completed:
            footer.append(f" · telemetry {covered / completed * 100:.0f}%", style="dim")
        parts.append(footer)

        return Group(*parts)
