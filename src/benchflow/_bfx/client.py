"""The host's side of bfx's protocol.

Protocol version 1 (bfx/src/serve.rs).

One-shot: ``bfx <cmd> --json '<args>'`` prints one JSON line,
``{"ok": true, "result": {...}}`` or ``{"ok": false, "error": {...}}``
(:func:`one_shot_argv`, :func:`parse_answer`). With ``--json -`` the
arguments are stdin's first line, so secrets (an exec's environment) never
appear in a process's argv.

``bfx serve`` takes one JSON object per line, ``{"id": n, "cmd", "args",
"v": 1, "r": seconds left}``, writes ``{"id": n, "started": true}`` (after
its journal lists the request) before a command runs, then the answer
(:class:`BfxServe`). A request the host gives up on before its started line
is withdrawn with a cancel line, and bfx refuses one whose time ran out; both
are *not started* (:class:`BfxNotStarted`). When the channel dies after a
request was sent, it may or may not have run (:class:`BfxAmbiguous`): a
root mutation is never sent a second time. Byte payloads travel as base64
(``data_b64``, ``stdin_b64``, ``stdout_b64``).

The decisions stay here, on the host: bfx only carries out primitives and
reports facts (TRUSTED-RUNTIME.md §4.1).
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import os
import signal
from collections.abc import Mapping, Sequence
from typing import Any

from benchflow._bfx.locate import SANDBOX_PATH

#: The longest answer line the host reads (a walk of a large tree, a
#: tar-out of a workspace).
MAX_LINE = 256 << 20


class BfxError(RuntimeError):
    """bfx refused or failed a command; *kind* is stable (``link``,
    ``mount``, ``missing``, ``survivors``, ``bad-request``, ...)."""

    def __init__(
        self,
        kind: str,
        message: str,
        *,
        errno: int | None = None,
        detail: Any = None,
    ) -> None:
        super().__init__(f"bfx {kind}: {message}")
        self.kind = kind
        self.message = message
        self.errno = errno
        self.detail = detail


class BfxProtocolError(RuntimeError):
    """bfx's output was not an answer (it was not bfx, or it died)."""


class BfxNotStarted(BfxError):
    """bfx did not start the request (withdrawn, past its deadline, or its
    start could not be recorded): it is safe to run it another way."""


class BfxAmbiguous(BfxProtocolError):
    """The channel died after the request was sent: it may or may not have
    run, and is never sent again. *started* says whether its started line
    arrived."""

    def __init__(self, message: str, ident: int, started: bool) -> None:
        super().__init__(message)
        self.ident = ident
        self.started = started


PROTOCOL = 1


def _error(body: Any) -> BfxError:
    if not isinstance(body, dict):
        return BfxError("protocol", f"unreadable error {body!r}")
    return BfxError(
        str(body.get("kind", "unknown")),
        str(body.get("message", "")),
        errno=body.get("errno") if isinstance(body.get("errno"), int) else None,
        detail=body.get("detail"),
    )


def one_shot_argv(
    cmd: str,
    args: Mapping[str, Any] | None = None,
    *,
    bfx: str = SANDBOX_PATH,
    args_on_stdin: bool = False,
) -> list[str]:
    """The argv of one bfx command, started by absolute path. With
    *args_on_stdin* the arguments are not in argv: send
    :func:`stdin_args_line` first on the command's stdin."""
    if not bfx.startswith("/"):
        raise ValueError("bfx is started by absolute path")
    if args_on_stdin:
        return [bfx, cmd, "--json", "-"]
    return [bfx, cmd, "--json", json.dumps(dict(args or {}), separators=(",", ":"))]


def tar_in_args(
    root: str,
    *,
    uid: int,
    gid: int,
    agent_derived: bool = True,
    **options: Any,
) -> dict[str, Any]:
    """``tar-in``'s arguments for extracting beneath *root* as *uid*:*gid*.

    Agent-derived bytes (a workspace restore, a regrade's or a separate
    verifier's transfer, anything taken from an agent-touched sandbox) are
    never given root ownership (TRUSTED-RUNTIME v3, review #12): the request
    carries ``refuse_root_owner``, so bfx itself refuses it unless both ids
    are non-root, and a root id here raises before any request is made. Only
    bytes BenchFlow made itself pass ``agent_derived=False``. Set-id bits
    never come from an archive (bfx masks them and refuses a mask that keeps
    one)."""
    if agent_derived and (uid == 0 or gid == 0):
        raise ValueError("agent-derived bytes are never extracted as root")
    if "refuse_root_owner" in options:
        raise ValueError("refuse_root_owner follows agent_derived")
    args: dict[str, Any] = {"root": root, "uid": uid, "gid": gid, **options}
    if agent_derived:
        args["refuse_root_owner"] = True
    return args


def stdin_args_line(args: Mapping[str, Any]) -> bytes:
    """The first stdin line for ``--json -``."""
    return json.dumps(dict(args), separators=(",", ":")).encode() + b"\n"


def parse_answer(output: bytes | str | None) -> dict[str, Any]:
    """The result of a one-shot answer; raises :class:`BfxError` when bfx
    refused, :class:`BfxProtocolError` when the output is not an answer."""
    if isinstance(output, bytes):
        output = output.decode("utf-8", "replace")
    lines = [line for line in (output or "").splitlines() if line.strip()]
    if len(lines) != 1:
        raise BfxProtocolError(f"expected one answer line, got {len(lines)}")
    try:
        answer = json.loads(lines[0])
    except ValueError as exc:
        raise BfxProtocolError(f"not a JSON answer: {exc}") from None
    if not isinstance(answer, dict) or not isinstance(answer.get("ok"), bool):
        raise BfxProtocolError("not a bfx answer")
    if not answer["ok"]:
        raise _error(answer.get("error"))
    result = answer.get("result")
    if not isinstance(result, dict):
        raise BfxProtocolError("an answer without a result")
    return result


class BfxServe:
    """A running ``bfx serve``: *argv* starts it (for a sandbox, the backend's
    exec of ``/.bfx/bfx serve`` as root). Calls are answered in order; one
    call at a time is on the wire."""

    def __init__(
        self, argv: Sequence[str], env: Mapping[str, str] | None = None
    ) -> None:
        # *argv* ends with ``serve`` or ``serve --journal <path>``.
        self._argv = list(argv)
        self._env = dict(env) if env is not None else None
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._ids = itertools.count(1)
        #: Requests the host gave up on; their late lines are skipped.
        self._stale: set[int] = set()
        self.version: dict[str, Any] | None = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def start(self, timeout: float = 15.0) -> dict[str, Any]:
        """Start bfx and read its version (which names its own SHA-256)."""
        self._process = await asyncio.create_subprocess_exec(
            *self._argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=self._env,
            limit=MAX_LINE,
            start_new_session=os.name == "posix",
        )
        self.version = await self.call("version", timeout=timeout)
        if self.version.get("protocol") != PROTOCOL:
            await self.close()
            raise BfxProtocolError(
                f"bfx speaks protocol {self.version.get('protocol')!r}, not {PROTOCOL}"
            )
        return self.version

    async def call(
        self,
        cmd: str,
        args: Mapping[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Run one command. Raises :class:`BfxError` when bfx refused or the
        command failed, :class:`BfxNotStarted` when it never started, and
        :class:`BfxAmbiguous` when the channel died after it was sent."""
        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise BfxProtocolError("bfx serve is not running")
        ident = next(self._ids)
        request: dict[str, Any] = {
            "id": ident,
            "v": PROTOCOL,
            "cmd": cmd,
            "args": dict(args or {}),
        }
        if timeout is not None:
            request["r"] = timeout
        async with self._lock:
            started = False
            sent = False
            answer: dict[str, Any] = {}
            try:
                process.stdin.write(json.dumps(request).encode() + b"\n")
                await process.stdin.drain()
                sent = True
                deadline = (
                    None
                    if timeout is None
                    else asyncio.get_running_loop().time() + timeout
                )
                while True:
                    left = (
                        None
                        if deadline is None
                        else max(0.0, deadline - asyncio.get_running_loop().time())
                    )
                    raw = await asyncio.wait_for(process.stdout.readline(), left)
                    if not raw:
                        raise ConnectionError("bfx serve closed its output")
                    line = json.loads(raw)
                    if not isinstance(line, dict) or line.get("id") in self._stale:
                        continue
                    if line.get("id") != ident:
                        raise BfxProtocolError("an answer to another call")
                    if line.get("started") is True and "ok" not in line:
                        started = True
                        continue
                    answer = line
                    break
            except TimeoutError as exc:
                if not started:
                    # Withdraw it: a request bfx has not started never starts.
                    with contextlib.suppress(Exception):
                        process.stdin.write(
                            json.dumps({"cancel": ident}).encode() + b"\n"
                        )
                self._stale.add(ident)
                # The cancel may arrive after it started: either way the host
                # cannot know, and the request is never sent again.
                raise BfxAmbiguous(
                    f"{cmd!r} timed out after {timeout} s; it may or may not have run",
                    ident,
                    started,
                ) from exc
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    if not started:
                        process.stdin.write(
                            json.dumps({"cancel": ident}).encode() + b"\n"
                        )
                self._stale.add(ident)
                raise
            except (
                ConnectionError,
                ValueError,
                asyncio.LimitOverrunError,
                BfxProtocolError,
            ) as exc:
                await self.close()
                if not sent:
                    raise BfxNotStarted(
                        "channel",
                        f"bfx serve stopped before the request was sent: {exc}",
                    ) from exc
                raise BfxAmbiguous(
                    f"bfx serve stopped during {cmd!r}: it may or may not have run ({exc})",
                    ident,
                    started,
                ) from exc
        if not answer.get("ok"):
            error = _error(answer.get("error"))
            if answer.get("started") is False:
                raise BfxNotStarted(
                    error.kind, error.message, errno=error.errno, detail=error.detail
                )
            raise error
        result = answer.get("result")
        if not isinstance(result, dict):
            raise BfxProtocolError("an answer without a result")
        return result

    async def close(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            with contextlib.suppress(Exception):
                process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if os.name == "posix" and process.pid > 1:
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=2.0)
