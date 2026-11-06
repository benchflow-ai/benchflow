"""ManifestEnvironment — the default Environment-plane adapter.

Implements the ``Environment`` protocol from an ``EnvironmentManifest``,
running over any ``Sandbox`` provider. This is the "zero-modification
adoption" path: a benchmark writes a manifest, this class runs it.

**Topology — in-sandbox (the architecture's core).** The environment's
services run *inside the rollout's own sandbox*, sharing its network
namespace, so the agent reaches them on ``localhost``. This adapter starts
the manifest's ``[[services]]`` and health-checks them with ``sandbox.exec``
— it is the declarative replacement for the hard-coded ``SERVICES`` dict +
``build_service_hooks`` in ``benchflow.sandbox.services``.

One manifest serves a whole benchmark even when, as with smolclaws, each
task is a per-task image carrying only a subset of the services: ``provision``
probes ``command -v`` and starts only the services actually installed (the
same idea as ``detect_services_from_dockerfile``).

Environment-state ``snapshot``/``restore`` are real here — roll-back over the
SQLite files an ``[environment.state]`` table declares. ``reset`` returns the
environment to the per-task baseline by stopping the services the framework
started, restoring the baseline snapshot captured during ``provision`` (when
``[environment.state]`` is declared), and restarting the services. The sidecar
/ shared-fleet topology (host-exposed ports, ``wait_for_readiness`` over
httpx) remains the platform layer.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import shlex
import time
from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from benchflow.environment.manifest import EnvironmentManifest, ServiceSpec
from benchflow.environment.protocol import (
    EnvHandle,
    EnvState,
    ReadinessProbe,
    StateSnapshot,
)
from benchflow.sandbox._exec_output import diagnostic_result, exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from benchflow.sandbox._task_route import as_task_command

# Bounded guard for the detached-start exec. The start command backgrounds the
# service and returns immediately, so this only bounds sandbox transport
# stalls — readiness() owns how long the service may take to become healthy.
_SERVICE_START_TIMEOUT_SEC = 15


def _state_command(commands: Iterable[str]) -> str:
    """Complete shell statements as one ``&&`` chain. Its size is the
    backend's to carry (the transport contract,
    :mod:`benchflow.sandbox._transport`), never a reason to split it."""
    return " && ".join(commands)


# The time one state chain gets: 120 s per 8 KiB of it, the budget each
# 8 KiB batch had when the chain was split (DX-1 review A L5), so a snapshot
# of many databases has as long as it had before.
_STATE_SECONDS_PER_UNIT = 120
_STATE_UNIT_BYTES = 8 * 1024
# The most of a chain an error carries (it names paths only).
_STATE_ERROR_COMMAND_MAX = 4096


def _state_timeout(command: str) -> int:
    units = -(-len(command.encode("utf-8")) // _STATE_UNIT_BYTES)
    return _STATE_SECONDS_PER_UNIT * max(1, units)


def _state_error_command(command: str) -> str:
    if len(command) <= _STATE_ERROR_COMMAND_MAX:
        return command
    return command[:_STATE_ERROR_COMMAND_MAX] + f" ... ({len(command)} characters)"


def _shell_path_match(path: str, pattern: str) -> str:
    """Test a captured path with the sandbox shell's pathname-pattern syntax.

    ``case`` uses the same bracket expressions (including locale character
    classes) as capture's pathname expansion. Match one component at a time:
    wildcards cannot cross slashes or select an implicit leading dot. Quoting
    assignments keeps every pattern and filename data, without eval or a new
    filesystem expansion that could select files created since capture.
    """
    if not glob.has_magic(pattern):
        return "true" if path == pattern else "false"
    parts, patterns = path.split("/"), pattern.split("/")
    if len(parts) != len(patterns):
        return "false"
    if any(
        part.startswith(".") and not component.startswith((".", r"\."))
        for part, component in zip(parts, patterns, strict=True)
    ):
        return "false"
    # A fixed loop keeps validation syntax independent of path depth. Unrolling
    # one case per component can exceed the exec bound for a short captured path
    # (150 one-character directories already did). Parameter expansions preserve
    # empty components and literal whitespace/backslashes without word splitting.
    return (
        f"(path={shlex.quote(path)}; pattern={shlex.quote(pattern)}; "
        "while :; do part=${path%%/*}; component=${pattern%%/*}; "
        'case "$part" in $component) : ;; *) exit 1 ;; esac; '
        'case "$path" in */*) path=${path#*/}; pattern=${pattern#*/} ;; '
        "*) break ;; esac; done)"
    )


def service_log_path(service_name: str) -> str:
    """In-sandbox path collecting a started service's stdout+stderr."""
    return f"/tmp/benchflow-env-{service_name}.log"


def service_start_command(svc: ServiceSpec) -> str:
    """Render the fully detached background start for one manifest service.

    Full fd detachment — ``nohup {cmd} </dev/null >{log} 2>&1 &`` — is
    load-bearing on Daytona (BF-9, the same failure class as BF-6/W9):
    Daytona runs each exec as a *session command*, and a backgrounded service
    that inherits ANY of the session's std streams keeps the session command
    "running", wedging the exec until its timeout — which the bounded start
    timeout turns into a hard provision failure on Daytona while the same
    manifest passes on Docker. The three redirections sever every inherited
    stream and ``nohup`` survives the session's terminal HUP. No ``disown``:
    that is a bash/zsh builtin, absent from the ``sh``/dash shell sandbox
    execs run under. The manifest's ``command`` is a single command line and
    is used verbatim, matching clawbench's ``_build_service_hooks`` shape.
    """
    return f"nohup {svc.command} </dev/null >{service_log_path(svc.name)} 2>&1 &"


class EnvironmentSnapshotError(RuntimeError):
    """Raised when a snapshot or restore sandbox command fails (issue #387).

    Snapshot/restore are the substrate ``Rollout.branch()`` builds rollback on
    — a silent infra failure here lets a Branch record a checkpoint that never
    existed (or restore from a copy that did not happen), then score children
    from corrupted state. Bubbling a typed error keeps that failure visible to
    the caller so child rewards / estimated V(parent) never leak into release
    evidence as if they were real.
    """

    def __init__(
        self, message: str, *, command: str, exit_code: int, stdout: str, stderr: str
    ) -> None:
        super().__init__(message)
        self.command = command
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class ManifestEnvironment:
    """Runs a manifest-declared stateful environment over a Sandbox.

    The Sandbox is injected so this class is provider-agnostic and
    unit-testable. ``provision`` starts the declared services that are
    present in the image; ``readiness`` gates on their health checks —
    both inside the sandbox.
    """

    def __init__(self, manifest: EnvironmentManifest, *, sandbox: Any) -> None:
        self._manifest = manifest
        self._sandbox = sandbox
        self._handle: EnvHandle | None = None
        self._started: list[ServiceSpec] = []
        # Baseline captured during ``provision`` for stateful manifests so
        # ``reset`` can return the environment to its initial per-task state
        # (distinct from arbitrary snapshot roll-back).
        self._baseline: StateSnapshot | None = None

    async def provision(self, ctx: Any) -> EnvHandle:
        """Start the manifest's services inside the sandbox.

        When ``owns_lifecycle`` is true the image's entrypoint already
        started the services, so this only builds the handle. Otherwise
        each service whose entry point actually runs is started in the
        background; services not installed in this per-task image are skipped.
        """
        m = self._manifest
        self._started = []
        if not m.owns_lifecycle:
            for svc in m.services:
                binary = svc.command.split()[0]
                # Detection probe: the service's entry point must actually
                # *run*, not merely exist on PATH. smolclaws-style base images
                # ship a console-script stub for every claw-* service, so
                # `command -v` over-detects — it sees the stub of a service
                # whose package this per-task image never installed. `--help`
                # runs the entry point and exits non-zero (ModuleNotFoundError)
                # for such a stub, while a real, installed service exits 0.
                # The manifest's binaries are the task's (C8's task route).
                with as_task_command(self._sandbox):
                    probe = await exec_raw(
                        self._sandbox.exec,
                        f"{shlex.quote(binary)} --help >/dev/null 2>&1",
                        timeout_sec=15,
                    )
                if probe.return_code != 0:
                    continue  # binary missing or its package absent — skip
                await self._start_service(svc)
                self._started.append(svc)
            if m.services and not self._started:
                # Per-task images legitimately carry a SUBSET of the declared
                # services — but ZERO startable services means the manifest and
                # the image disagree entirely (stale CLI names, wrong image),
                # and with no explicit readiness URLs the readiness gate then
                # passes vacuously. That exact shape shipped a drifted
                # env0@prod pin whose every rollout burned its budget and died
                # at the verifier with connection-refused (2026-08-09). Fail
                # here, loudly, instead.
                declared = ", ".join(s.command.split()[0] for s in m.services)
                raise RuntimeError(
                    f"Environment manifest '{m.name}' declares "
                    f"{len(m.services)} service(s) but NONE are startable in "
                    f"this image (probed: {declared}). The manifest's service "
                    "commands likely no longer match the image — verify the "
                    "binary names against the image or update the manifest."
                )
        endpoints = {p: f"http://localhost:{p}" for p in m.all_ports}
        self._handle = EnvHandle(name=m.name, endpoints=endpoints)
        # Capture a baseline of the declared state so ``reset`` can return the
        # environment to its per-task initial state. Stateless manifests have
        # no [environment.state] table — ``reset`` then just restarts services.
        if m.state is not None:
            self._baseline = await self._capture_baseline()
        return self._handle

    async def _start_service(self, svc: ServiceSpec) -> None:
        """Start one service fully detached, logging to its per-service file.

        Shared by ``provision`` and ``reset`` so the (Daytona-critical)
        detachment shape lives in exactly one place — see
        :func:`service_start_command`. The service's output is retrievable at
        :func:`service_log_path` for its name.
        """
        with as_task_command(self._sandbox):
            await exec_raw(
                self._sandbox.exec,
                service_start_command(svc),
                timeout_sec=_SERVICE_START_TIMEOUT_SEC,
            )

    async def readiness(self) -> ReadinessProbe:
        """Poll each service's health endpoint from inside the sandbox.

        Explicit HTTP and TCP probes apply in both lifecycle modes. Without
        explicit probes, only framework services actually started are probed.
        """
        m = self._manifest
        timeout = m.readiness.timeout_sec
        urls = m.readiness.http
        if not urls and not m.readiness.tcp and not m.owns_lifecycle:
            urls = [f"http://localhost:{s.port}{s.health_path}" for s in self._started]

        probes: list[tuple[str, int | None]] = [(url, None) for url in urls]
        probes.extend((f"tcp://localhost:{port}", port) for port in m.readiness.tcp)
        deadline = time.monotonic() + timeout
        checked: list[str] = []
        for url, port in probes:
            checked.append(url)
            while True:
                remaining = deadline - time.monotonic()
                # A zero timeout retains the one-shot probe contract. Positive
                # deadlines bound the whole sequence, including each request.
                if timeout > 0 and remaining <= 0:
                    return ReadinessProbe(
                        ready=False,
                        checked=checked,
                        error=f"environment not ready: {url} never responded",
                    )
                request_timeout = remaining if timeout > 0 else 1
                if port is None:
                    command = (
                        f"curl -sf --connect-timeout {request_timeout} "
                        f"--max-time {request_timeout} {shlex.quote(url)} "
                        ">/dev/null 2>&1"
                    )
                else:
                    command = f"timeout {request_timeout} bash -c " + shlex.quote(
                        f"exec 3<>/dev/tcp/127.0.0.1/{port}"
                    )
                result = await exec_raw(
                    self._sandbox.exec,
                    command,
                    timeout_sec=request_timeout + 1,
                )
                if result.return_code == 0:
                    break
                if time.monotonic() >= deadline:
                    return ReadinessProbe(
                        ready=False,
                        checked=checked,
                        error=f"environment not ready: {url} never responded",
                    )
                await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))
        return ReadinessProbe(ready=True, checked=checked, error=None)

    async def query(self) -> EnvState:
        """Return environment state for the verifier.

        The slice's verifier reads state through the task's own test.sh
        inside the sandbox, so this returns an empty state.
        """
        return EnvState(data={})

    async def teardown(self) -> None:
        """Stop the service processes (best-effort).

        In the in-sandbox topology the Sandbox plane owns container
        teardown; this only stops the processes this adapter started.
        """
        for svc in self._started:
            binary = svc.command.split()[0]
            with contextlib.suppress(Exception):
                await exec_raw(
                    self._sandbox.exec,
                    f"pkill -f {shlex.quote(binary)}",
                    timeout_sec=10,
                )

    # roll-back (the substrate branching runs on)

    async def snapshot(self) -> StateSnapshot:
        """Capture the environment's declared state — Han's roll-back.

        For ``kind = "sqlite"`` each declared DB file is copied with
        ``sqlite3 .backup`` (a consistent online backup) into a per-snapshot
        directory inside the sandbox.
        """
        if self._manifest.state is None:
            raise RuntimeError(
                f"environment '{self._manifest.name}' declares no "
                "[environment.state]; snapshot/restore are unsupported for a "
                "stateless environment"
            )
        return await self._capture_baseline()

    async def _capture_baseline(self) -> StateSnapshot:
        """Copy the declared state files into a fresh in-sandbox snapshot dir.

        Caller guarantees ``self._manifest.state`` is not None.

        Issue #387: if the sandbox command fails (sqlite3 missing, db file
        absent, copy permission denied), raise ``EnvironmentSnapshotError``
        rather than recording a checkpoint that never existed — Branch and
        ``reset()`` depend on this being a real backup.
        """
        spec = self._manifest.state
        assert spec is not None  # caller checks
        snap_id = uuid4().hex[:12]
        snap_dir = f"/tmp/benchflow-snapshots/{snap_id}"
        cmds = [f"mkdir -p {shlex.quote(snap_dir)}"]
        paths: list[str] = []
        for pattern in spec.paths:
            if not glob.has_magic(pattern):
                paths.append(pattern)
                continue
            # Expand in the sandbox, without eval or word splitting: shell
            # metacharacters in a path stay data, and NUL frames every filename.
            command = (
                f"pattern={shlex.quote(pattern)}; IFS=; "
                'for src in $pattern; do [ -f "$src" ] || exit 1; '
                'printf "%s\\0" "$src"; done'
            )
            # Through the receipt: NUL frames the names, and a session
            # command's output on Daytona loses every NUL.
            expanded = await receipt_exec(self._sandbox, command, timeout_sec=120)
            if expanded.return_code != 0 or not expanded.stdout:
                raise EnvironmentSnapshotError(
                    "snapshot pattern matched no database files",
                    command=command,
                    exit_code=expanded.return_code or 1,
                    stdout=expanded.stdout or "",
                    stderr=expanded.stderr or "",
                )
            paths.extend(expanded.stdout.rstrip("\0").split("\0"))
        files = {
            src: f"{index}.sqlite" for index, src in enumerate(dict.fromkeys(paths))
        }
        for src, filename in files.items():
            dest = f"{snap_dir}/{filename}"
            cmds.append(
                f'[ -f {shlex.quote(src)} ] && sqlite3 -readonly {shlex.quote(src)} ".backup {shlex.quote(dest)}"'
            )
        command = _state_command(cmds)
        result = await exec_raw(
            self._sandbox.exec, command, timeout_sec=_state_timeout(command)
        )
        if result.return_code != 0:
            raise EnvironmentSnapshotError(
                f"snapshot failed for environment '{self._manifest.name}' "
                f"(exit_code={result.return_code}): "
                f"{(diagnostic_result(result).stderr.render() or diagnostic_result(result).stdout.render() or '').strip()[:500]}",
                command=_state_error_command(command),
                exit_code=result.return_code,
                stdout=diagnostic_result(result).stdout.render() or "",
                stderr=diagnostic_result(result).stderr.render() or "",
            )
        return StateSnapshot(id=snap_id, path=snap_dir, files=files)

    async def restore(self, snap: StateSnapshot) -> None:
        await self._restore(snap, clear_sqlite_sidecars=False)

    async def restore_after_sandbox_restore(self, snap: StateSnapshot) -> None:
        """Install standalone backups before framework services are restarted.

        A container image may contain WAL/journal files newer than the online
        backup. They must not replay over that backup. This entry point requires
        successful sandbox recreation with no database processes running; ordinary
        environment-only restoration cannot safely remove live SQLite sidecars.
        """
        self.validate_sandbox_restore()
        await self._restore(snap, clear_sqlite_sidecars=True)

    async def _restore(
        self, snap: StateSnapshot, *, clear_sqlite_sidecars: bool
    ) -> None:
        """Roll the environment's state back to a snapshot.

        Copies each captured DB file from the snapshot directory back over
        its live path. The caller quiesces the agent and services first
        (the Branch lifecycle) so the copy is consistent.

        Issue #387: if the sandbox ``cp`` command fails (snapshot directory
        missing, destination not writable), raise
        ``EnvironmentSnapshotError`` rather than silently returning success
        — earlier copies may already have changed live state, so the caller
        must treat it as an infra failure, not a successful rollback.
        """
        spec = self._manifest.state
        if spec is None:
            raise RuntimeError(
                f"environment '{self._manifest.name}' declares no "
                "[environment.state]; snapshot/restore are unsupported for a "
                "stateless environment"
            )
        from benchflow.trust.state import IngressKind, admit

        admit(self._sandbox, "main", IngressKind.SNAPSHOT_RESTORE)
        files = snap.files
        if not files:
            if any(glob.has_magic(path) for path in spec.paths):
                raise ValueError(
                    "legacy snapshot has no captured database paths for its patterns"
                )
            # Legacy snapshots used basenames. A collision has already destroyed
            # one backup; refuse to restore silently corrupted state.
            files = {dst: PurePosixPath(dst).name for dst in spec.paths}
            if len(set(files.values())) != len(files):
                raise ValueError("legacy snapshot has ambiguous database basenames")
        if len(set(files.values())) != len(files):
            raise ValueError("snapshot maps multiple databases to the same backup")
        if any(
            PurePosixPath(name).name != name or name in {"", ".", ".."}
            for name in files.values()
        ):
            raise ValueError("snapshot contains an invalid backup filename")
        mismatch = "snapshot database paths do not match the environment manifest"
        if not any(glob.has_magic(pattern) for pattern in spec.paths):
            if set(files) != set(spec.paths):
                raise ValueError(mismatch)
        else:
            # Both sides must be covered, under the same shell syntax used to
            # capture. Validate before any copy or sidecar removal.
            # Each matched pair proves coverage of its path and pattern, in
            # one exec; no live state changes until all are proven.
            checks = (
                f"if {_shell_path_match(path, pattern)}; "
                f"then printf '%s\\n' p{i} g{j}; fi"
                for i, path in enumerate(files)
                for j, pattern in enumerate(spec.paths)
            )
            check = _state_command(checks)
            result = await exec_raw(
                self._sandbox.exec, check, timeout_sec=_state_timeout(check)
            )
            if result.return_code != 0:
                raise ValueError(mismatch)
            covered = set((result.stdout or "").splitlines())
            required = {f"p{i}" for i in range(len(files))} | {
                f"g{j}" for j in range(len(spec.paths))
            }
            if covered != required:
                raise ValueError(mismatch)
        cmds = []
        for dst in files:
            src = f"{snap.path}/{files[dst]}"
            if clear_sqlite_sidecars:
                sidecars = " ".join(
                    shlex.quote(dst + suffix) for suffix in ("-wal", "-shm", "-journal")
                )
                cmds.append(f"rm -f -- {sidecars}")
            cmds.append(f"cp {shlex.quote(src)} {shlex.quote(dst)}")
        command = _state_command(cmds)
        result = await exec_raw(
            self._sandbox.exec, command, timeout_sec=_state_timeout(command)
        )
        if result.return_code != 0:
            raise EnvironmentSnapshotError(
                f"restore failed for environment '{self._manifest.name}' "
                f"snapshot {snap.id!r} (exit_code={result.return_code}): "
                f"{(diagnostic_result(result).stderr.render() or diagnostic_result(result).stdout.render() or '').strip()[:500]}",
                command=_state_error_command(command),
                exit_code=result.return_code,
                stdout=diagnostic_result(result).stdout.render() or "",
                stderr=diagnostic_result(result).stderr.render() or "",
            )

    def validate_sandbox_restore(self) -> None:
        """Reject service lifecycles a recreated container cannot reproduce."""
        m = self._manifest
        has_services = bool(
            m.services or m.all_ports or m.effective_http or m.readiness.tcp
        )
        if m.owns_lifecycle and has_services:
            raise RuntimeError(
                f"environment '{m.name}' has entrypoint-owned services; sandbox "
                "branching has no validated service restart contract"
            )
        if m.services and not self._started:
            raise RuntimeError(
                f"environment '{m.name}' must be provisioned before sandbox branching"
            )

    def prepare_sandbox_restore(self) -> None:
        """Invalidate the old handle before replacing its container/processes."""
        self.validate_sandbox_restore()
        self._handle = None

    async def resume_after_sandbox_restore(self) -> None:
        """Recreate framework services after container and database restoration.

        Unlike provision(), this preserves the original reset baseline and starts
        only the previously installed services. A handle is valid again only once
        readiness succeeds. The engine also calls this for sandbox-only branches.
        """
        self.validate_sandbox_restore()
        self._handle = None
        for svc in self._started:
            await self._start_service(svc)
        probe = await self.readiness()
        if not probe.ready:
            raise RuntimeError(
                probe.error or "environment not ready after sandbox restore"
            )
        self._handle = EnvHandle(
            name=self._manifest.name,
            endpoints={p: f"http://localhost:{p}" for p in self._manifest.all_ports},
        )

    async def reset(self) -> None:
        """Return the environment to its per-task initial state.

        Distinct from ``restore`` (which rolls back to an arbitrary snapshot):
        ``reset`` returns to the baseline captured during ``provision`` so the
        environment can be reused for a fresh episode without tearing down the
        sandbox. The sequence is the inverse of ``provision`` for the same
        manifest:

        1. Stop the services the framework started (best-effort ``pkill``).
        2. Restore the baseline state snapshot, if a baseline exists. A
           stateless manifest skips this step.
        3. Restart the previously-started services so the environment is ready
           for the next episode.

        When the image entrypoint owns the lifecycle (``owns_lifecycle = true``)
        the framework cannot restart services itself; ``reset`` then only
        restores baseline state (if declared) and leaves the entrypoint-owned
        services running.
        """
        if self._handle is None:
            raise RuntimeError(
                f"environment '{self._manifest.name}' has not been provisioned; "
                "call provision() before reset()"
            )
        # 1. Stop framework-started services so the SQLite restore is consistent.
        if self._started:
            for svc in self._started:
                binary = svc.command.split()[0]
                with contextlib.suppress(Exception):
                    await exec_raw(
                        self._sandbox.exec,
                        f"pkill -f {shlex.quote(binary)}",
                        timeout_sec=10,
                    )
        # 2. Restore the per-task baseline state, if any.
        if self._baseline is not None:
            await self.restore(self._baseline)
        # 3. Restart the same services we previously started, in the same
        #    detached background-with-log shape provision() uses.
        #    owns_lifecycle = true manifests reach this with an empty
        #    self._started — nothing to do.
        for svc in self._started:
            await self._start_service(svc)
