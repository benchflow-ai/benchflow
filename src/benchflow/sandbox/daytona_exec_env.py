"""An exec's env on Daytona and its DinD VM: a file root uploads, never command text.

Daytona keeps each session command's text in its API and logs, and every
process in the sandbox can read a running shell's ``/proc/<pid>/cmdline``, so
no env value is written into a command. Each distinct env (the ``export``
lines of :func:`~benchflow.sandbox._base.env_stdin_body`) is a file the
toolbox (root) uploads once per box, into one folder of root's staging folder
(:data:`~benchflow.sandbox._transport.STAGE_ROOT`), under a random name. The
key, the SHA-256 of the file's bytes, stays on the host: a command names a
file and says nothing about what is in it. Before the box's first upload, one
root exec prepares and proves the staging folder and makes the env folder
root's, mode 700 (:func:`~benchflow.sandbox._transport.prepare_stage_root`),
so the toolbox's upload cannot follow an image's link out of it.

Each exec that carries an env has root's session shell prove the file
root's alone (:func:`proof`), make it 0600 and open it before anything in
the command runs (:func:`opened`). The
command reads it from that descriptor
(:func:`~benchflow.sandbox._base.env_from_descriptor`). Processes of the
selected account hold the descriptor while the command runs (``su``'s shell,
``timeout``); mode 0600 keeps that account from opening the file again
through their ``/proc`` entries.

A failed preparation or upload raises: the exec is refused and nothing of it
is sent; there is no inline form to fall back to. A file that fails its proof
retires its folder: the next exec uploads into a new one. The folders are
removed at stop and before a snapshot (a snapshot keeps the filesystem);
while it is taken no upload is made, and after it the next exec uploads
again. A new provider sandbox (a start, a restore) starts with no files.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import secrets
import shlex
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from benchflow._root_only import ROOT_CHAIN_FUNCS
from benchflow.sandbox._base import (
    ENV_FD,
    SYSTEM_PATH,
    SYSTEM_SHELL,
    ExecRoute,
    env_from_descriptor,
    system_shell_prefix,
    system_shell_route,
)
from benchflow.sandbox._transport import READ_FAILED, STAGE_ROOT, prepare_stage_root

logger = logging.getLogger(__name__)


class ExecEnvUnavailable(RuntimeError):
    """An exec's env file could not be made or removed: the step is refused."""


#: The words of :func:`opened`'s refusal that :meth:`ExecEnvFiles.refused`
#: is called on.
REFUSED = "an exec's environment file"


def refuse_nul(env: dict[str, str]) -> None:
    """Refuse an env with a NUL byte in a value: no process environment can
    hold one, and a shell reading the env's file would drop it silently, so
    the command would run with a value that is not the one asked for."""
    named = sorted(k for k, v in env.items() if "\0" in v)
    if named:
        raise ValueError(
            f"benchflow: the env value of {', '.join(named)} holds a NUL byte, "
            "which no process environment can carry; nothing of the command ran"
        )


@dataclass
class _Box:
    provider: Any
    folder: str
    paths: dict[str, str] = field(default_factory=dict)
    prepared: bool = False


class ExecEnvFiles:
    """The env files in one ``DaytonaSandbox``'s current provider sandbox."""

    def __init__(self) -> None:
        self._box: _Box | None = None
        # Folders of this provider sandbox no exec opens any more (one whose
        # file failed its proof), removed with the current one.
        self._stale: list[tuple[Any, str]] = []
        self._lock = asyncio.Lock()
        self._snapshotting = False

    def _current(self, provider: Any) -> _Box:
        if self._box is None or self._box.provider is not provider:
            self._box = _Box(provider, f"{STAGE_ROOT}/env-{secrets.token_hex(16)}")
        return self._box

    async def path(self, sandbox: Any, body: bytes, route: ExecRoute) -> str:
        """The box's file holding *body*, uploaded now if this box has none.

        *route*: the exec's own, which names the box (the DinD VM's for
        BenchFlow's own VM). Refused while a snapshot is being taken
        (:meth:`snapshotting`)."""
        provider = sandbox._require_sandbox()
        key = hashlib.sha256(body).hexdigest()
        async with self._lock:
            if self._snapshotting:
                raise ExecEnvUnavailable(
                    "benchflow: an exec with an environment was asked for while "
                    "the sandbox's snapshot is being taken; it is refused, so no "
                    "environment file is in the snapshot"
                )
            box = self._current(provider)
            found = box.paths.get(key)
            if found is not None:
                return found
            if not box.prepared:
                await sandbox._prepare_exec_env_folder(route, box.folder)
                box.prepared = True
            path = f"{box.folder}/{secrets.token_hex(16)}"
            await sandbox._sdk_upload_bytes(body, path)
            box.paths[key] = path
            return path

    def refused(self) -> None:
        """An exec's env file failed its proof (:func:`opened`): no exec
        opens that folder again. The next one uploads its env into a new
        folder; the old one is removed with it."""
        self._retire()

    def _retire(self) -> None:
        box, self._box = self._box, None
        if box is not None and box.prepared:
            self._stale.append((box.provider, box.folder))

    async def remove(self, sandbox: Any, *, required: bool = False) -> None:
        """Remove the box's env folders and forget their files.

        *required*: raise :class:`ExecEnvUnavailable` when a folder may still
        be there (before a snapshot, which would keep it); otherwise a
        failure is logged and the folder goes with the sandbox."""
        self._retire()
        stale, self._stale = self._stale, []
        provider = getattr(sandbox, "_sandbox", None)
        # Another provider sandbox's folders went with it.
        for folder in [f for p, f in stale if p is provider]:
            try:
                await sandbox._sdk_delete_tree(folder)
            except Exception as exc:
                if required:
                    raise ExecEnvUnavailable(
                        f"benchflow: could not remove the execs' environment "
                        f"files ({folder}): {exc}; the snapshot is not taken, so "
                        "it holds none of them"
                    ) from exc
                logger.warning(
                    "Could not remove the execs' environment files (%s): %s; "
                    "they go with the sandbox",
                    folder,
                    exc,
                )

    @contextlib.asynccontextmanager
    async def snapshotting(self, sandbox: Any) -> AsyncIterator[None]:
        """Around a snapshot of *sandbox*: its env files removed first (a
        failure refuses the snapshot), no upload while it is taken, and the
        cache empty after it, so the next exec uploads its env again."""
        await self.remove(sandbox, required=True)
        self._snapshotting = True
        try:
            yield
        finally:
            self._snapshotting = False
            self._box = None


#: Where Daytona's daemon (root in a task image, and on the DinD VM) keeps
#: each session command it ran: ``<session>/<command>/cmd.sh``, the command's
#: text, beside its output and status, for the sandbox's life (BenchFlow keeps
#: its sessions, so detached children live on); a stop and start keep them,
#: and a snapshot holds them.
SESSIONS_DIR = "/root/.daytona/sessions"


async def scrub_sessions(
    sandbox: Any, route: ExecRoute, *, required: bool, ran_sessions: bool = False
) -> None:
    """Remove every session's files from the box *route* runs on.

    One toolbox exec (``process.exec``: no session of its own, so it writes
    nothing there itself) with root's tools. The daemon's finished commands
    need nothing in those folders, and the processes they started keep their
    own descriptors. *required*: raise :class:`ExecEnvUnavailable` when it
    fails (before a snapshot, which would hold them); otherwise log it.
    *ran_sessions*: this sandbox sent session commands, so the daemon keeps
    them there: with *required*, no folder there is refused (a daemon that
    keeps them elsewhere), never taken as clean."""
    shell, path, _ = await root_tools(sandbox, route)
    missing = (
        f"{{ echo 'no {SESSIONS_DIR} on this box' >&2; exit 3; }}"
        if required and ran_sessions
        else "exit 0"
    )
    # An attach's env folder that ssh never reached goes too (the remote
    # shell removes it once it opened it).
    attach = f"{STAGE_ROOT}/attach-"
    body = (
        f"{system_shell_prefix(path)}; "
        f'for d in {attach}*; do [ ! -e "$d" ] || rm -rf -- "$d"; done; '
        f"[ -d {SESSIONS_DIR} ] || {missing}; "
        f"find {SESSIONS_DIR} -mindepth 1 -maxdepth 1 -exec rm -rf -- {{}} +"
    )
    command = f"{system_shell_prefix(path)}; {shell} -c {shlex.quote(body)}"
    try:
        response = await sandbox._require_sandbox().process.exec(command, timeout=60)
        code = getattr(response, "exit_code", 1)
        if code != 0:
            output = str(getattr(response, "result", "") or "").strip()
            raise RuntimeError(f"status {code}: {output[-300:]}")
    except Exception as exc:
        if required:
            raise ExecEnvUnavailable(
                f"benchflow: could not remove Daytona's session files "
                f"({SESSIONS_DIR}): {exc}; the snapshot is not taken, so it holds "
                "none of them"
            ) from exc
        logger.warning(
            "Could not remove Daytona's session files (%s): %s; they go with the "
            "sandbox",
            SESSIONS_DIR,
            exc,
        )


async def remove_exec_env_files(sandbox: Any) -> None:
    """Remove *sandbox*'s exec env files now (a Daytona sandbox's; nothing on
    another backend): at the end of the verifier's command, so the file of
    its ``[verifier.env]`` is not kept for the rest of the trial, where any
    root process could read it. The next exec uploads its env again. A
    failure is logged; stop removes the folder too."""
    files = getattr(sandbox, "_exec_env_files", None)
    if isinstance(files, ExecEnvFiles):
        await files.remove(sandbox)


async def root_tools(sandbox: Any, route: ExecRoute) -> tuple[str, str, ExecRoute]:
    """The shell, PATH and root route of the box *route* runs on, as
    ``DaytonaSandbox._stage_proof`` picks them: the DinD VM's system tools
    for BenchFlow's own VM, else root's system-shell route here (the
    protected kit's once the sandbox has one)."""
    if route.own_vm:
        return SYSTEM_SHELL, SYSTEM_PATH, route
    root = await system_shell_route(sandbox)
    return root.shell, root.pin or SYSTEM_PATH, root


def preparation(folder: str, *, shell: str, path: str) -> str:
    """sh: root's staging folder prepared and proved, then *folder* made in
    it, root's and 700 (a new name: ``mkdir`` fails on anything there)."""
    make = shlex.quote(f"umask 077 && mkdir -- {shlex.quote(folder)}")
    return (
        f"{prepare_stage_root(shell=shell, path=path)} && "
        f"( {system_shell_prefix(path)}; {shell} -c {make} )"
    )


async def prepare_folder(sandbox: Any, folder: str, route: ExecRoute) -> None:
    """One root exec of :func:`preparation` on the box *route* runs on,
    before anything is uploaded into *folder*. Raises
    :class:`ExecEnvUnavailable` when that fails."""
    shell, path, root = await root_tools(sandbox, route)
    result = await sandbox._sandbox_exec(
        preparation(folder, shell=shell, path=path),
        shell="/bin/sh -c",
        timeout_sec=60,
        route=root,
    )
    if result.return_code != 0:
        detail = " ".join(
            (text or "").strip() for text in (result.stdout, result.stderr)
        ).strip()
        raise ExecEnvUnavailable(
            f"benchflow: root's staging folder ({STAGE_ROOT}) could not be "
            f"prepared for an exec's environment (status {result.return_code}"
            f"{': ' + detail[-400:] if detail else ''}); nothing of the "
            "command ran"
        )


def proof(names: list[str], *, shell: str, path: str) -> str:
    """sh: status 0 only when the env files *names* (one folder of root's
    staging folder) are root's alone, as
    :func:`~benchflow.sandbox._transport.stage_proof` proves a staged
    payload, and then each is made 0600. One subshell on *shell* by absolute
    path with PATH *path*: every folder and link on the way to the files'
    folder root's (``bfc_chain``), that folder and the staging folder root's
    and nobody else's to write, no link, and each file root's, no link,
    nobody else's to write. The staging folder was prepared once
    (:func:`prepare_folder`); this checks, it repairs nothing, and a file
    every exec opens costs a few ``find`` calls, not a staged payload's
    full proof."""
    folder = names[0].rsplit("/", 1)[0]
    if folder.rsplit("/", 1)[0] != STAGE_ROOT or any(
        n.rsplit("/", 1)[0] != folder for n in names
    ):
        raise ValueError(f"env files outside one folder of {STAGE_ROOT}: {names}")
    checks = " && ".join(
        [
            f"bfc_chain {shlex.quote(folder)} 0",
            f"bfc_dir {shlex.quote(STAGE_ROOT)}",
            *(f"bfc_file {shlex.quote(n)}" for n in names),
            f"chmod 600 -- {shlex.join(names)}",
        ]
    )
    body = f"{system_shell_prefix(path)}; {ROOT_CHAIN_FUNCS}{checks}"
    return f"( {system_shell_prefix(path)}; {shell} -c {shlex.quote(body)} )"


def opened(command: str, files: dict[int, str], proven: str) -> str:
    """Root's shell text that runs *command* with each env file of *files*
    (descriptor to path, all in one folder) open on its descriptor, after
    *proven* (:func:`proof` of those files) passed; otherwise a
    ``benchflow:`` line and status 126, and nothing of *command* runs."""
    names = sorted(set(files.values()))
    redirects = " ".join(
        f"{'' if fd == 0 else fd}<{shlex.quote(p)}" for fd, p in sorted(files.items())
    )
    refuse = shlex.quote(
        f"benchflow: {REFUSED} ({' '.join(names)}) is not root's alone; "
        "nothing of the command ran"
    )
    return (
        f"if {proven}; "
        f"then {{ {command}\n}} {redirects}; "
        f"else echo {refuse} >&2; (exit {READ_FAILED}); fi"
    )


async def opening(
    sandbox: Any, files: dict[int, str], route: ExecRoute
) -> Callable[[str], str]:
    """:func:`opened` with ``sandbox._exec_env_proof`` of *files* on the box
    *route* runs on."""
    proven = await sandbox._exec_env_proof(route, sorted(set(files.values())))
    return lambda command: opened(command, files, proven)


async def attach_env_command(sandbox: Any, body: bytes, command: str) -> str:
    """The DinD VM's shell text for an operator's ``attach``: *body* (the
    compose env) in a fresh root-only 0600 file the text proves and opens,
    removes at once (the descriptor still reads it) and reads before it
    execs *command*. Only the file's path is in the text: the process that
    runs it (``ssh``) replaces BenchFlow's, so nothing comes back to remove
    the file later."""
    from benchflow.sandbox.daytona_dind import VM_ROUTE

    folder = f"{STAGE_ROOT}/attach-{secrets.token_hex(16)}"
    path = f"{folder}/env"
    await sandbox._prepare_exec_env_folder(VM_ROUTE, folder)
    await sandbox._sdk_upload_bytes(body, path)
    return opened(
        f"/bin/rm -rf -- {shlex.quote(folder)}; "
        f"{env_from_descriptor('/bin/cat', ENV_FD)}exec {command}",
        {ENV_FD: path},
        await sandbox._exec_env_proof(VM_ROUTE, [path]),
    )
