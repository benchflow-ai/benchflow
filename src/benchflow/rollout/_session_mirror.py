"""Copy the agent's session log out of the sandbox while its turn runs.

A sandbox lost before a turn's report took the turn's tokens with it on the
ACP path: Claude Code's session log stays in the sandbox, and the ACP
adapter's updates carry how full the context is, not each call's tokens
(Codex's rollouts likewise stay in the sandbox). While a turn runs, the
log's new bytes are copied to the trial folder every
:data:`MIRROR_INTERVAL_SEC` and when the turn ends, redacted line by line as
they arrive. At teardown, when the full copy (``_session_log``) could not be
made, this copy takes its place in ``agent/`` and the salvage and the Codex
count read it (``usage_source: session_log``).

Cheap: one listing a tick, then only the bytes each file grew by. It never
blocks the turn: it is a task beside it, its redaction runs in a worker
thread, and a failure is a warning.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import logging
import shlex
import shutil
from pathlib import Path
from typing import Any

from benchflow._utils.reachable_write import private_folder
from benchflow.sandbox._exec_output import exec_raw

logger = logging.getLogger(__name__)

MIRROR_INTERVAL_SEC = 30.0
# Under the trial folder, beside agent/ and not in it: the sandbox shares
# agent/, so the copy is placed there only once, at teardown, by one rename.
MIRROR_DIR = ".session-mirror"
_EXEC_TIMEOUT_SEC = 15
_TURN_END_TIMEOUT_SEC = 20.0
# The most of one file read in one tick; the rest comes in the next.
_READ_CAP = 4 * 1024 * 1024


def _shares_another_rollouts_turns(rollout: Any) -> bool:
    return bool(
        getattr(rollout, "_branch_child_active", False)
        or getattr(rollout, "_branch_forks", None)
        or getattr(rollout, "_from_branch_snapshot", False)
    )


def _plan(rollout: Any) -> tuple[str, str] | None:
    """The agent's session-log folder in the sandbox and its name under
    ``agent/``, when the log is the trial's only record of its calls: no
    proxy saw them, one agent made them all, no other rollout's turns."""
    from benchflow.rollout._session_log import (
        CLAUDE_SESSION_LOGS,
        CODEX_SESSION_DIR,
        CODEX_SESSION_LOGS,
        SESSION_DIR,
        _contributors,
        _home,
        _runs_claude_code,
        _runs_codex,
    )

    if (
        getattr(rollout, "_env", None) is None
        or getattr(rollout, "_rollout_paths", None) is None
        or getattr(rollout, "_native_usage_proxy_covered", False)
        or _shares_another_rollouts_turns(rollout)
    ):
        return None
    role = getattr(rollout, "_active_role", None)
    agent = (getattr(role, "agent", None) if role is not None else None) or (
        rollout._config.primary_agent or ""
    )
    agents = _contributors(rollout) | {agent}
    cfg = rollout._config
    if all(_runs_claude_code(name) for name in agents):
        return f"{_home(cfg)}/{CLAUDE_SESSION_LOGS}", SESSION_DIR
    if all(_runs_codex(name) for name in agents):
        env = getattr(rollout, "_agent_env", None) or {}
        home = env.get("CODEX_HOME") or f"{_home(cfg)}/.codex"
        return f"{home}/{CODEX_SESSION_LOGS}", CODEX_SESSION_DIR
    return None


def _clean(line: str, secrets: list[str]) -> str:
    """One log line, redacted as the teardown copy is (``_session_log.redact_line``)."""
    from benchflow.rollout._session_log import redact_line

    return redact_line(line, secrets)


def _render(held: bytes, chunk: bytes, secrets: list[str]) -> tuple[str, bytes]:
    """What *chunk* adds to a file's copy, after the unended line *held*:
    its ended lines, redacted (:func:`_clean`), and the line still unended.
    It changes nothing, so it runs in a worker thread. A redaction that
    raises is ``RedactionFailed``, named by the exception's class."""
    from benchflow.rollout._session_log import RedactionFailed

    ended, newline, rest = (held + chunk).rpartition(b"\n")
    if not newline:
        return "", rest
    lines = ended.decode("utf-8", errors="replace").split("\n")
    try:
        return "".join(_clean(line, secrets) + "\n" for line in lines), rest
    except Exception as exc:
        raise RedactionFailed(type(exc).__name__) from None


class SessionMirror:
    """The trial's copy of one agent's session-log folder, kept while it runs."""

    def __init__(self, rollout: Any, source: str, name: str) -> None:
        self._rollout = rollout
        self.source = source
        self.name = name
        self.folder = Path(rollout._rollout_paths.rollout_dir) / MIRROR_DIR / name
        # Bytes copied of each sandbox file, and its last line not yet ended.
        self._offsets: dict[str, int] = {}
        self._partial: dict[str, bytes] = {}
        self._task: asyncio.Task[None] | None = None
        self._warned = False
        # The redaction raised: the copy stops where it was (and says so).
        self._failed = False
        # Whether the copy had caught up with the sandbox's log when the last
        # turn ended: its final copy ran, read every file to its end and left
        # no line half-written. A copy placed at teardown that had not is a
        # count of some of the turn's calls only.
        self.current = False

    async def sync(self) -> bool:
        """Copy what each log file gained since the last sync; True when
        every file was copied to its end and no line is half-written."""
        from benchflow.rollout._session_log import (
            RedactionFailed,
            _secrets,
            note_redaction_failure,
            off_loop,
        )

        env = self._rollout._env
        if env is None or self._failed:
            return False
        source = shlex.quote(self.source)
        listing = await exec_raw(
            env.exec,
            f"cd {source} 2>/dev/null || exit 3; "
            "find . -type f -name '*.jsonl' -exec wc -c {} +",
            user="root",
            timeout_sec=_EXEC_TIMEOUT_SEC,
        )
        if listing.return_code != 0:
            return True  # no log yet: nothing to copy
        secrets = _secrets(self._rollout)
        complete = True
        for row in str(listing.stdout or "").splitlines():
            size, _, name = row.strip().partition(" ")
            name = name.strip()
            if not size.isdigit() or not name.startswith("./"):
                continue  # wc's "total" line
            if ".." in Path(name).parts:
                continue
            offset = self._offsets.get(name, 0)
            if int(size) <= offset:
                continue
            read = await exec_raw(
                env.exec,
                f"cd {source} && tail -c +{offset + 1} {shlex.quote(name)}"
                f" | head -c {_READ_CAP} | base64",
                user="root",
                timeout_sec=_EXEC_TIMEOUT_SEC,
            )
            if read.return_code != 0:
                complete = False
                continue
            chunk = base64.b64decode(str(read.stdout or ""))
            if not chunk:
                complete = False
                continue
            if offset + len(chunk) < int(size):
                complete = False  # the rest comes in the next tick
            # Redacted in a worker thread and appended on the loop; a cancel
            # (the turn ended, a timeout) takes effect once the chunk is.
            try:
                await off_loop(
                    _render,
                    self._partial.get(name, b""),
                    chunk,
                    secrets,
                    then=functools.partial(self._commit, name, offset + len(chunk)),
                )
            except RedactionFailed as exc:
                self._failed = True
                note_redaction_failure(self._rollout, exc.error, copy="mirror")
                return False
        return complete and not self._partial

    def _commit(self, name: str, end: int, out: tuple[str, bytes]) -> None:
        self._offsets[name] = end
        self._place(name, *out)

    def _append(self, name: str, chunk: bytes, secrets: list[str]) -> None:
        """Add *chunk* to the copy of *name*, on the calling thread."""
        self._place(name, *_render(self._partial.get(name, b""), chunk, secrets))

    def _place(self, name: str, text: str, rest: bytes) -> None:
        """Append the redacted *text* to the copy of *name*, and hold *rest*
        (its last line, not yet ended)."""
        if rest:
            self._partial[name] = rest
        else:
            self._partial.pop(name, None)
        if not text:
            return
        path = self.folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)

    async def _sync_quietly(self) -> bool:
        try:
            return await asyncio.wait_for(self.sync(), timeout=_TURN_END_TIMEOUT_SEC)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the sandbox may be gone: keep what was copied
            if not self._warned:
                self._warned = True
                logger.warning(
                    "Could not copy the agent's session log while its turn ran: %s",
                    exc,
                )
            return False

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(MIRROR_INTERVAL_SEC)
            await self._sync_quietly()

    def start_turn(self) -> None:
        self.current = False
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def end_turn(self, *, sync: bool = True) -> None:
        """Stop the ticks; copy once more unless the turn was cancelled."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            # Its cancellation, or a failed tick: the copy so far stands.
            with contextlib.suppress(BaseException):
                await task
        self.current = await self._sync_quietly() if sync else False


def start_session_mirror(rollout: Any) -> SessionMirror | None:
    """Start copying the agent's session log for this turn; None when the
    trial needs no copy (see :func:`_plan`). Never fails the turn."""
    try:
        plan = _plan(rollout)
        if plan is None:
            return None
        mirror = getattr(rollout, "_session_mirror", None)
        if mirror is None or (mirror.source, mirror.name) != plan:
            mirror = SessionMirror(rollout, *plan)
            rollout._session_mirror = mirror
        mirror.start_turn()
        return mirror
    except Exception as exc:
        logger.warning("Could not start copying the agent's session log: %s", exc)
        return None


def place_session_mirror(rollout: Any) -> None:
    """At teardown: where the full copy of the agent's session log could not
    be made (a lost sandbox), the copy made while its turns ran takes its
    place under ``agent/``. The copy is removed either way."""
    paths = getattr(rollout, "_rollout_paths", None)
    if paths is None:
        return
    root = Path(paths.rollout_dir) / MIRROR_DIR
    if not root.is_dir():
        return
    try:
        if _shares_another_rollouts_turns(rollout):
            return  # teardown keeps no log for these (see keep_session_logs)
        mirror = getattr(rollout, "_session_mirror", None)
        placed = getattr(rollout, "_mirror_placed", None)
        if not isinstance(placed, dict):
            placed = rollout._mirror_placed = {}
        snapshot = getattr(rollout, "_session_log_snapshot", None)
        for folder in sorted(root.iterdir()):
            target = paths.agent_dir / folder.name
            if not any(folder.rglob("*.jsonl")):
                continue
            if snapshot is not None:
                # After the copy point: the held copy is the log;
                # the mirror stands in only where none was made, and what
                # agent/ holds is not the trial's.
                held = snapshot.view / "agent" / folder.name
                if held.exists():
                    continue
                shutil.copytree(folder, held, symlinks=True)
                from benchflow.rollout._session_log import publish_session_log_copy

                publish_session_log_copy(rollout, folder.name, held)
            else:
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with private_folder(target.parent) as temp:
                    staged = Path(temp) / "tree"
                    shutil.copytree(folder, staged, symlinks=True)
                    target.publish(staged, refuse_special=True)
            # A count read from it misses what the copy had not reached.
            placed[folder.name] = bool(
                mirror is not None
                and getattr(mirror, "name", None) == folder.name
                and getattr(mirror, "current", False)
            )
            logger.warning(
                "The agent's session log could not be copied at teardown; "
                "the trial keeps the copy made while its turns ran (agent/%s)",
                folder.name,
            )
    except Exception as exc:
        logger.warning("Could not keep the agent's session-log copy: %s", exc)
    finally:
        shutil.rmtree(root, ignore_errors=True)
