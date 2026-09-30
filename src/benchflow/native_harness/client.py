"""Drive a native CLI harness with the verbs the kernel calls on an ACP client.

The rollout kernel runs an ACP agent through ``ACPClient`` (``prompt``,
``cancel``, ``close``, ``session``, ``on_ask_user``, ``expect_silence``) and
:func:`benchflow.acp.runtime.execute_prompts`, which owns the idle watchdog,
the wall-clock budget and the bounded cancel. :class:`NativeCLIClient` offers
the same verbs, so both harnesses share that loop. Each :meth:`prompt` starts
the CLI in the sandbox for one turn, as the sandbox user, through the same
live-process transport ACP agents use; reads its JSON lines; and applies the
parser's ACP updates to an :class:`~benchflow.acp.session.ACPSession`, so the
trajectory, the dashboards, the watchdog and the training exports read the
same object they read for an ACP agent.

Every process of a turn carries ``BENCHFLOW_NATIVE_RUN=<id>`` in its
environment and the CLI leads its own process group (``set -m``), so
:meth:`cancel` can signal the CLI's group and then kill every marked process,
including tool subprocesses that left the group.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import shlex
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from benchflow.acp.session import ACPSession
from benchflow.acp.types import McpServerSpec, PromptResult, StopReason
from benchflow.diagnostics import TransportClosedError
from benchflow.native_harness.spec import (
    NativeHarness,
    NativeHarnessError,
    NativeParser,
    NativeTurn,
    NativeTurnOutcome,
)
from benchflow.sandbox.lockdown import build_priv_drop_cmd
from benchflow.sandbox.process._acp_lines import line_limit, shrink_line
from benchflow.sandbox.process._base import _ANSI_CSI_RE, _ANSI_OSC_RE
from benchflow.trajectories.types import redact_trajectory_obj, redact_trajectory_text

logger = logging.getLogger(__name__)

RUN_ENV = "BENCHFLOW_NATIVE_RUN"
# The launch script's last line: the CLI's exit status.
EXIT_KEY = "benchflow_native_exit"
# How long a cancelled CLI gets to exit on SIGINT (writing its result and
# session log) before its process group and every marked process are killed.
CANCEL_GRACE_SEC = 2
# How long a cancelled turn's prompt waits for the CLI's stream to end and for
# the kill, counted from the cancel. The kernel gives a timed-out prompt 5 s
# to return on its own (acp/timeout_cleanup.py) before it cancels the task,
# which loses the turn's diagnostics; a kill still running then goes on, and
# close() and the next turn wait for it.
CANCEL_RETURN_SEC = 3.5
# How long a CLI may keep running after it reported the end of its turn.
EXIT_GRACE_SEC = 15
_KILL_TIMEOUT_SEC = 20
_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cached_read_tokens",
    "cached_write_tokens",
    "thought_tokens",
    "total_tokens",
)


def launch_script(executable: str, argv: tuple[str, ...], run_id: str) -> str:
    """The in-sandbox script that runs one turn (before the privilege drop).

    The prompt arrives as one base64 line on the transport's stdin, as an ACP
    prompt does in a JSON-RPC line, so its size is not bounded by the
    command-line and environment limits of ``execve``. It becomes the CLI's
    stdin through a private file that is unlinked (its descriptor kept) before
    the CLI starts; the redirection is explicit because a background job's
    default stdin is /dev/null when the shell has no terminal. ``set -m``
    starts the CLI as a job in its own process group (and is switched off
    again so bash prints no job notice); the shell waits for it and reports
    its exit status as a last JSON line, :data:`EXIT_KEY`, which reaches the
    client on every transport.

    The CLI writes to the shell's original stderr (kept as fd 4), and the
    shell's own stderr is /dev/null while it waits: bash reports a job a
    signal killed with its whole command line ("line 1: 479 Killed ..."),
    which would put the arguments, MCP settings included, into the error.
    """
    work = shlex.quote(f"/tmp/benchflow-native-{run_id}")
    command = " ".join(shlex.quote(part) for part in (executable, *argv))
    return (
        f"umask 077; mkdir -p {work} || exit 1; "
        "IFS= read -r p || exit 1; "
        f"printf '%s' \"$p\" | base64 -d > {work}/prompt || exit 1; "
        f"unset p; exec 3<{work}/prompt 4>&2; rm -rf {work}; "
        f"set -m; {command} <&3 3<&- 2>&4 4>&- & set +m; "
        "exec 2>/dev/null; wait $!; rc=$?; "
        f'printf \'{{"{EXIT_KEY}": %d}}\\n\' "$rc"; exit $rc'
    )


def encode_prompt(text: str) -> str:
    """The stdin line :func:`launch_script` turns back into the prompt."""
    return base64.b64encode(text.encode()).decode()


def _process_functions(run_id: str) -> str:
    """Shell functions over this turn's marked processes (read from ``/proc``).

    Marked processes are those whose initial environment holds this turn's
    ``BENCHFLOW_NATIVE_RUN``: the CLI, the shells around it and every tool
    subprocess the CLI starts. One ``grep`` reads every environment (the run
    id is a unique 32-digit hex string), so the scan costs one process
    however many run in the sandbox. Needs ``/proc``, ``grep``, ``sed`` and
    ``cut``.
    """
    marker = shlex.quote(f"{RUN_ENV}={run_id}")
    return (
        f"marked() {{ grep -lF {marker} /proc/[0-9]*/environ 2>/dev/null | "
        "sed -n 's#^/proc/\\([0-9][0-9]*\\)/environ$#\\1#p'; }; "
        "leaders() { for p in $(marked); do "
        "g=$(sed 's/^.*) //' /proc/$p/stat 2>/dev/null | cut -d' ' -f3); "
        '[ "$g" = "$p" ] && echo "$p"; done; }; '
        'running() { for p in "$@"; do '
        "s=$(sed 's/^.*) //' /proc/$p/stat 2>/dev/null | cut -d' ' -f1); "
        '[ -n "$s" ] && [ "$s" != Z ] && echo "$p"; done; }; '
        "alive() { running $(marked); }; "
    )


# Polls every 0.2 s (every second where sleep takes whole seconds only).
_POLL = "sleep 0.2 2>/dev/null || { sleep 1; i=$((i+4)); }; i=$((i+1))"


def kill_script(run_id: str, *, grace_sec: int) -> str:
    """Root script: SIGINT the turn's process groups, then kill what is left.

    The grace lasts until the signalled group leaders (the CLI) have exited,
    at most ``grace_sec``: the CLI gets to write its result and session log,
    and the tool processes it leaves are killed at once. Only groups a marked
    process leads are signalled as groups, so a group shared with anything
    else (a CLI started without job control) is never hit; every marked
    process is then killed by pid. Prints ``gone`` once no marked process
    runs, else ``left`` and the pids.
    """
    return (
        _process_functions(run_id)
        + "gs=$(leaders); for g in $gs; do kill -INT -- -$g 2>/dev/null; done; "
        f'i=0; while [ $i -lt {int(grace_sec) * 5} ] && [ -n "$(running $gs)" ]; '
        f"do {_POLL}; done; "
        "for g in $gs $(leaders); do kill -KILL -- -$g 2>/dev/null; done; "
        "for p in $(marked); do kill -KILL $p 2>/dev/null; done; "
        f'i=0; while [ $i -lt 10 ] && [ -n "$(alive)" ]; do {_POLL}; done; '
        'left=$(alive); [ -z "$left" ] && echo gone || echo left $left'
    )


def alive_script(run_id: str) -> str:
    """Root script printing ``alive`` while any marked process runs, else ``gone``."""
    return _process_functions(run_id) + '[ -n "$(alive)" ] && echo alive || echo gone'


def recorded_argv(argv: tuple[str, ...]) -> list[str]:
    """A turn's arguments for the evidence files, MCP server settings left out.

    Task MCP servers can carry credentials in their environment or headers;
    the logs name the setting, not its value (Claude Code's ``--mcp-config``
    document, Codex's ``-c mcp_servers.*`` overrides).
    """
    shown: list[str] = []
    for index, arg in enumerate(argv):
        if index and argv[index - 1] == "--mcp-config":
            shown.append("<omitted>")
        elif arg.startswith("mcp_servers."):
            shown.append(arg.split("=", 1)[0] + "=<omitted>")
        else:
            shown.append(arg)
    return shown


def _consume(task: asyncio.Future[Any]) -> None:
    """Retrieve an abandoned read's outcome so asyncio does not report it."""
    if not task.cancelled():
        task.exception()


def _decode_event(text: str, *, lenient: bool) -> dict[str, Any] | None:
    """One JSON object from a line; ``lenient`` tolerates PTY noise around it."""
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict):
        return value
    if not lenient or "{" not in text:
        return None
    cleaned = _ANSI_CSI_RE.sub("", _ANSI_OSC_RE.sub("", text))
    if "{" not in cleaned:
        return None
    try:
        value = json.loads(cleaned[cleaned.index("{") :])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


class NativeCLIClient:
    """One native-harness conversation: a CLI session continued turn by turn.

    ``resume_id`` opens an existing CLI session (a branch child resuming its
    parent). Otherwise the first turn creates the session: under
    ``new_session_id`` when the CLI accepts a caller-chosen id (Claude Code),
    else under the id the CLI reports.
    """

    def __init__(
        self,
        *,
        env: Any,
        harness: NativeHarness,
        agent: str,
        launch_env: dict[str, str],
        sandbox_user: str | None,
        cwd: str,
        rollout_dir: Path,
        model: str | None = None,
        reasoning_effort: str | None = None,
        mcp_servers: tuple[McpServerSpec, ...] = (),
        resume_id: str | None = None,
    ) -> None:
        self._env = env
        self._harness = harness
        self._agent = agent
        self._launch_env = dict(launch_env)
        self._sandbox_user = sandbox_user
        self._cwd = cwd
        self._model = model
        self._effort = reasoning_effort
        self._mcp_servers = tuple(mcp_servers)
        self._resume_id = resume_id
        self._new_session_id = (
            str(uuid4()) if resume_id is None and harness.accepts_session_id else None
        )
        # Until a CLI that picks its own id reports it, the session is named
        # after this client (the id a later turn or a branch child resumes is
        # set once known, see prompt()).
        self._session = ACPSession(
            resume_id or self._new_session_id or f"native-{uuid4().hex[:12]}"
        )
        self._turn = 0
        self._silence_sec: float | None = None
        self._process: Any = None
        self._run_id: str | None = None
        self._cancel_requested = False
        self._cancel_at: float | None = None
        self._cancel_event: asyncio.Event | None = None
        self._totals: dict[str, int] = dict.fromkeys(_USAGE_FIELDS, 0)
        self._last_total: dict[str, int] | None = None
        self._exit_code: int | None = None
        # A cancel's kill (it can outlive the prompt, see CANCEL_RETURN_SEC)
        # and the run it stops.
        self._kill_task: asyncio.Future[bool] | None = None
        self._kill_run_id: str | None = None
        self._kill_record: dict[str, Any] | None = None
        # The running turn's evidence record
        self._record: dict[str, Any] | None = None
        self._closed = False
        agent_dir = rollout_dir / "agent"
        self._stream_path = agent_dir / f"{harness.cli}.jsonl"
        self._log_path = agent_dir / f"{agent.replace('-', '_')}.txt"
        self._turns_path = agent_dir / "native-turns.json"
        self._log_file: TextIO | None = None
        self.turns: list[dict[str, Any]] = []

    # -- the ACPClient surface the kernel uses --------------------------------

    @property
    def session(self) -> ACPSession:
        return self._session

    @property
    def harness(self) -> NativeHarness:
        return self._harness

    @property
    def cli_session_id(self) -> str | None:
        """The CLI's own session id (what ``--resume`` takes), once known."""
        return self._resume_id or self._new_session_id

    def expect_silence(self, seconds: float) -> None:
        """Let the live process's read guard wait at least ``seconds`` (#1143)."""
        self._silence_sec = seconds
        if self._process is not None:
            self._process.expect_silence(seconds)

    def on_ask_user(self, handler: Any) -> None:
        """The native CLIs run with their approvals off and no permission channel.

        Clearing (``None``) is a no-op. A handler would never be called, and a
        run that registers one (an ``on_ask_user`` hook, or a task whose
        ``confirmation_policy`` is ``human``) relies on permission requests
        being routed; it is refused rather than run with every tool approved.
        """
        if handler is not None:
            raise NativeHarnessError(
                self._harness.cli,
                "the native harness has no permission channel, so an ask_user "
                "handler (or confirmation_policy: human) cannot be honored; "
                "run this with harness='acp'",
            )

    async def prompt(self, text: str) -> PromptResult:
        """Run one turn: start the CLI, stream its events, return how it ended."""
        if self._closed:
            raise RuntimeError("native harness client is closed")
        if self._process is not None:
            raise RuntimeError("a native harness turn is already running")
        # The last turn's processes are gone before this one starts: a
        # cancel's kill may still run, and a turn whose prompt task was
        # cancelled outright may have left its processes.
        await self._finish_kill()
        if self._run_id is not None:
            await self._kill(self._run_id, grace_sec=0)
            self._run_id = None
        self._turn += 1
        self._cancel_requested = False
        self._cancel_at = None
        self._cancel_event = asyncio.Event()
        self._exit_code = None
        run_id = uuid4().hex
        resume_id = self._resume_id
        turn = NativeTurn(
            cwd=self._cwd,
            resume_id=resume_id,
            new_session_id=None if resume_id else self._new_session_id,
            model=self._model,
            reasoning_effort=self._effort,
            mcp_servers=self._mcp_servers,
        )
        launch = self._harness.build_launch(turn, self._launch_env)
        command = launch_script(self._harness.executable, launch.argv, run_id)
        if self._sandbox_user:
            command = build_priv_drop_cmd(command, self._sandbox_user)
        env = {**self._launch_env, **launch.env, RUN_ENV: run_id}
        parser = self._harness.new_parser(self._cwd, self._turn)
        started = datetime.now(UTC)
        shown_argv = recorded_argv(launch.argv)
        record: dict[str, Any] = {
            "turn": self._turn,
            "cli": self._harness.cli,
            "version": self._harness.version,
            "resumed": resume_id,
            "argv": shown_argv,
            "started_at": started.isoformat(),
        }
        self._record = record
        self._write_stream({"benchflow_native_turn": self._turn, "argv": shown_argv})
        process = await self._env.live_process(agent=self._agent)
        self._process = process
        self._run_id = run_id
        end: TransportClosedError | None = None
        try:
            try:
                await process.start(command=command, env=env, cwd=self._cwd)
                await process.writeline(encode_prompt(text))
                if self._cancel_requested:
                    # cancel() ran while the process was starting, before its
                    # processes existed to be killed.
                    await self._kill(run_id, grace_sec=0)
                if self._silence_sec is not None:
                    process.expect_silence(self._silence_sec)
                end = await self._read_turn(process, parser, run_id)
            finally:
                self._process = None
                with contextlib.suppress(Exception):
                    await process.close()
                stderr = getattr(process, "stderr_tail", "")
                if isinstance(stderr, str) and stderr.strip():
                    self._log(redact_trajectory_text(stderr.rstrip()))
        except asyncio.CancelledError:
            # The caller stopped waiting (the kernel does 5 s after a
            # timeout): keep the evidence; close() or the next turn kills
            # what still runs.
            record.update(
                ended_at=datetime.now(UTC).isoformat(),
                cancelled=True,
                failure="the prompt was cancelled before the turn ended",
            )
            self._save_turn(record)
            raise
        kill = self._kill_task
        if self._cancel_requested and kill is not None:
            # Return once the cancel's kill is done, or at its deadline.
            cancel_at = self._cancel_at or asyncio.get_running_loop().time()
            remaining = (
                cancel_at + CANCEL_RETURN_SEC - asyncio.get_running_loop().time()
            )
            await asyncio.wait({kill}, timeout=max(remaining, 0))
            record["stopped"] = self._kill_outcome()
        outcome = parser.outcome()
        exit_code = self._exit_code
        if exit_code is None and end is not None:
            exit_code = end.diagnostic.process_exit_code
        if outcome.session_id and not self._resume_id:
            self._resume_id = outcome.session_id
        elif self._resume_id is None and self._new_session_id and outcome.completed:
            self._resume_id = self._new_session_id
        if self._resume_id:
            # What a branch child or a checkpoint retry resumes (rollout_branch
            # reads the rollout session's id).
            self._session.session_id = self._resume_id
        turn_usage = self._turn_usage(outcome)
        self._record_usage(turn_usage)
        record.update(
            ended_at=datetime.now(UTC).isoformat(),
            session_id=outcome.session_id,
            exit_code=exit_code,
            usage=turn_usage,
            cli_cost_usd=outcome.cost_usd,
            error=outcome.error,
            cancelled=self._cancel_requested,
        )
        try:
            stop = await self._stop_reason(outcome, end, run_id, exit_code)
        except BaseException as exc:
            record["failure"] = str(exc)
            self._save_turn(record)
            raise
        finally:
            self._run_id = None
        record["stop_reason"] = stop.value
        self._save_turn(record)
        self._session.stop_reason = stop
        usage = self._session.latest_usage_totals()
        return PromptResult.model_validate(
            {
                "stopReason": stop.value,
                "usage": {
                    "inputTokens": usage.get("input_tokens") or 0,
                    "outputTokens": usage.get("output_tokens") or 0,
                    "totalTokens": usage.get("total_tokens") or 0,
                    "cachedReadTokens": usage.get("cached_read_tokens"),
                    "cachedWriteTokens": usage.get("cached_write_tokens"),
                    "thoughtTokens": usage.get("thought_tokens"),
                }
                if usage
                else None,
            }
        )

    async def cancel(self) -> None:
        """Stop the running turn: SIGINT the CLI's group, then kill what is left.

        The trajectory stops at the cancel: events the CLI writes while it
        winds down (Claude Code reports the interrupted tool call as rejected)
        are kept in the stream log but not applied to the session, as an ACP
        adapter's cancelled turn leaves its pending calls pending. The prompt
        returns once the kill is done or :data:`CANCEL_RETURN_SEC` after the
        cancel, whichever is first; a kill still running then goes on, and
        :meth:`close` and the next turn wait for it.
        """
        run_id = self._run_id
        if run_id is None:
            return
        if not self._cancel_requested:
            self._cancel_requested = True
            self._cancel_at = asyncio.get_running_loop().time()
            if self._cancel_event is not None:
                self._cancel_event.set()
        if self._kill_task is None or self._kill_run_id != run_id:
            self._kill_task = asyncio.ensure_future(
                self._kill(run_id, grace_sec=CANCEL_GRACE_SEC)
            )
            self._kill_run_id = run_id
            self._kill_record = self._record
        await asyncio.shield(self._kill_task)

    async def close(self) -> None:
        """Kill what the turns left running and release the transport and logs."""
        self._closed = True
        await self._finish_kill()
        run_id = self._run_id
        if run_id is not None:
            # A turn still running, or one whose prompt task was cancelled
            # outright.
            self._cancel_requested = True
            with contextlib.suppress(Exception):
                await self._kill(run_id, grace_sec=0)
        process = self._process
        if process is not None:
            with contextlib.suppress(Exception):
                await process.close()
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    # -- internals -------------------------------------------------------------

    async def _read_turn(
        self, process: Any, parser: NativeParser, run_id: str
    ) -> TransportClosedError | None:
        """Apply the CLI's output lines until its stream ends or a deadline passes.

        Returns the transport's end-of-stream error, or None when a deadline
        ended the read: a CLI that reported the end of its turn gets
        :data:`EXIT_GRACE_SEC` to exit before it is killed, and after a
        cancel the read stops :data:`CANCEL_RETURN_SEC` after it. One read
        is outstanding at a time and is never abandoned mid-line while the
        turn runs, so no output is lost to a deadline check.
        """
        loop = asyncio.get_running_loop()
        lenient = True
        finished_at: float | None = None
        event = self._cancel_event or asyncio.Event()
        cancelled = asyncio.ensure_future(event.wait())
        read: asyncio.Future[bytes] | None = None
        try:
            while True:
                if read is None:
                    read = asyncio.ensure_future(process.readline())
                if self._cancel_at is not None:
                    deadline: float | None = self._cancel_at + CANCEL_RETURN_SEC
                elif finished_at is not None:
                    deadline = finished_at + EXIT_GRACE_SEC
                else:
                    deadline = None
                done, _ = await asyncio.wait(
                    {read} if cancelled.done() else {read, cancelled},
                    timeout=None
                    if deadline is None
                    else max(deadline - loop.time(), 0),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if read in done:
                    finished, read = read, None
                    try:
                        line = finished.result()
                    except TransportClosedError as exc:
                        return exc
                    if self._handle_line(line, parser, lenient=lenient):
                        lenient = False
                        if finished_at is None and parser.outcome().completed:
                            finished_at = loop.time()
                    continue
                if done:
                    continue  # a cancel: its deadline applies from now on
                if self._cancel_requested:
                    return None
                logger.warning(
                    "%s did not exit %ss after the end of its turn; stopping it",
                    self._harness.cli,
                    EXIT_GRACE_SEC,
                )
                await self._kill(run_id, grace_sec=0)
                return None
        finally:
            cancelled.cancel()
            if read is not None:
                if not read.done():
                    read.cancel()
                    await asyncio.wait({read}, timeout=1)
                read.add_done_callback(_consume)

    def _kill_outcome(self) -> bool | None:
        """The cancel's kill: True (nothing left), False, or None (running)."""
        task = self._kill_task
        if task is None or not task.done():
            return None
        if task.cancelled() or task.exception() is not None:
            return False
        return bool(task.result())

    async def _finish_kill(self) -> None:
        """Wait for a cancel's kill, and kill again if it left processes."""
        task = self._kill_task
        if task is None:
            return
        try:
            with contextlib.suppress(Exception):
                await asyncio.shield(task)
            stopped = self._kill_outcome()
            if not stopped and self._kill_run_id is not None:
                try:
                    stopped = await self._kill(self._kill_run_id, grace_sec=0)
                except Exception:
                    logger.warning("Native harness kill failed", exc_info=True)
            record = self._kill_record
            if record is not None and record.get("stopped") is not True:
                record["stopped"] = bool(stopped)
                self._write_turns()
        finally:
            self._kill_task = None
            self._kill_run_id = None
            self._kill_record = None

    async def _kill(self, run_id: str, *, grace_sec: int) -> bool:
        """Run :func:`kill_script`; True when no process of the run is left."""
        result = await self._env.exec(
            kill_script(run_id, grace_sec=grace_sec),
            user="root",
            timeout_sec=_KILL_TIMEOUT_SEC + grace_sec,
        )
        out = (getattr(result, "stdout", "") or "").strip()
        if getattr(result, "return_code", 0) == 0 and out.endswith("gone"):
            return True
        logger.warning(
            "Native harness kill of turn %s exited %s: %s %s",
            run_id,
            getattr(result, "return_code", None),
            out[-200:],
            (getattr(result, "stderr", "") or "")[:300],
        )
        return False

    async def _cli_alive(self, run_id: str) -> bool:
        result = await self._env.exec(
            alive_script(run_id), user="root", timeout_sec=_KILL_TIMEOUT_SEC
        )
        return (getattr(result, "stdout", "") or "").strip().endswith("alive")

    async def _stop_reason(
        self,
        outcome: NativeTurnOutcome,
        end: TransportClosedError | None,
        run_id: str,
        exit_code: int | None,
    ) -> StopReason:
        cli = self._harness.cli
        if self._cancel_requested:
            return StopReason.CANCELLED
        if outcome.error is not None:
            raise NativeHarnessError(cli, outcome.error)
        if outcome.completed and outcome.stop_reason is not None:
            return outcome.stop_reason
        # The stream ended with no final event. If the sandbox answers, the
        # CLI died (killed, crashed); otherwise the transport or the sandbox
        # went away, which is reported as the transport failure it is.
        try:
            alive = await self._cli_alive(run_id)
        except Exception:
            if end is not None:
                raise end from None
            raise
        if alive:
            with contextlib.suppress(Exception):
                await self._kill(run_id, grace_sec=0)
            if end is not None:
                raise end
        detail = (
            f"the CLI exited (exit code {exit_code}) before it reported the end of the turn"
            if exit_code is not None
            else "the CLI stopped before it reported the end of the turn"
        )
        stderr = end.diagnostic.stderr_snippet if end is not None else None
        if stderr:
            detail += f": {stderr.strip()[-500:]}"
        raise NativeHarnessError(cli, detail, exit_code=exit_code)

    def _handle_line(self, line: bytes, parser: NativeParser, *, lenient: bool) -> bool:
        """Apply one output line; True once it was a JSON event."""
        limit = line_limit()
        if limit is not None and len(line) > limit:
            line = shrink_line(line, limit)
        text = line.decode(errors="replace").strip()
        if not text:
            return False
        event = _decode_event(text, lenient=lenient)
        if event is None:
            self._log(redact_trajectory_text(text))
            return False
        if set(event) == {EXIT_KEY} and isinstance(event[EXIT_KEY], int):
            self._exit_code = event[EXIT_KEY]
            return False
        self._write_stream(event)
        try:
            updates = parser.feed(event)
        except Exception:
            logger.warning("Native harness parser failed on an event", exc_info=True)
            self._log(f"parser failed on event type {event.get('type')!r}")
            return True
        if self._cancel_requested:
            return True
        for update in updates:
            self._session.handle_update(update)
        return True

    def _turn_usage(self, outcome: NativeTurnOutcome) -> dict[str, int] | None:
        """This turn's usage, from the turn's own report or consecutive totals.

        A client that resumed a session it did not start (a branch child)
        has no earlier total, so its first turn counts the resumed history
        too; with BenchFlow's proxy the proxy's usage is the run's anyway.
        """
        if outcome.usage is not None:
            return outcome.usage
        total = outcome.usage_total
        if total is None:
            return None
        previous = self._last_total or {}
        self._last_total = dict(total)
        return {
            field: max(int(total.get(field) or 0) - int(previous.get(field) or 0), 0)
            for field in _USAGE_FIELDS
        }

    def _record_usage(self, usage: dict[str, int] | None) -> None:
        """Add this turn's usage to the running totals the session reports.

        ``ACPSession.record_prompt_usage`` keeps cumulative snapshots (the
        rollout takes deltas between them), so the per-turn usage is summed
        here first.
        """
        if not usage:
            return
        for field in _USAGE_FIELDS:
            self._totals[field] += int(usage.get(field) or 0)
        self._session.record_prompt_usage(dict(self._totals))

    def _write_stream(self, event: dict[str, Any]) -> None:
        self._stream_path.parent.mkdir(parents=True, exist_ok=True)
        with self._stream_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(redact_trajectory_obj(event)) + "\n")

    def _log(self, text: str) -> None:
        if self._log_file is None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = self._log_path.open("a", encoding="utf-8")
        self._log_file.write(text + "\n")
        self._log_file.flush()

    def _save_turn(self, record: dict[str, Any]) -> None:
        self.turns.append(record)
        self._write_turns()

    def _write_turns(self) -> None:
        self._turns_path.parent.mkdir(parents=True, exist_ok=True)
        self._turns_path.write_text(
            json.dumps(redact_trajectory_obj(self.turns), indent=2) + "\n"
        )


__all__ = [
    "CANCEL_GRACE_SEC",
    "EXIT_KEY",
    "RUN_ENV",
    "NativeCLIClient",
    "kill_script",
    "alive_script",
    "encode_prompt",
    "launch_script",
]
