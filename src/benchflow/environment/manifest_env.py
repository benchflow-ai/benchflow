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

import contextlib
import shlex
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
                probe = await self._sandbox.exec(
                    f"{shlex.quote(binary)} --help >/dev/null 2>&1",
                    timeout_sec=15,
                )
                if probe.return_code != 0:
                    continue  # binary missing or its package absent — skip
                log = f"/tmp/benchflow-env-{svc.name}.log"
                await self._sandbox.exec(
                    f"{svc.command} > {log} 2>&1 &",
                    timeout_sec=15,
                )
                self._started.append(svc)
        endpoints = {p: f"http://localhost:{p}" for p in m.all_ports}
        self._handle = EnvHandle(name=m.name, endpoints=endpoints)
        # Capture a baseline of the declared state so ``reset`` can return the
        # environment to its per-task initial state. Stateless manifests have
        # no [environment.state] table — ``reset`` then just restarts services.
        if m.state is not None:
            self._baseline = await self._capture_baseline()
        return self._handle

    async def readiness(self) -> ReadinessProbe:
        """Poll each service's health endpoint from inside the sandbox.

        For framework-started environments only the services that were
        actually started are probed; for entrypoint-owned environments the
        manifest's declared HTTP probes are used.
        """
        m = self._manifest
        timeout = m.readiness.timeout_sec
        if m.owns_lifecycle:
            urls = m.effective_http
        else:
            urls = [f"http://localhost:{s.port}{s.health_path}" for s in self._started]

        checked: list[str] = []
        for url in urls:
            checked.append(url)
            result = await self._sandbox.exec(
                f"for i in $(seq 1 {timeout}); do "
                f"curl -sf {shlex.quote(url)} >/dev/null 2>&1 && exit 0; "
                f"sleep 1; done; exit 1",
                timeout_sec=timeout + 10,
            )
            if result.return_code != 0:
                return ReadinessProbe(
                    ready=False,
                    checked=checked,
                    error=f"environment not ready: {url} never responded",
                )
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
                await self._sandbox.exec(
                    f"pkill -f {shlex.quote(binary)}", timeout_sec=10
                )

    # --- roll-back (the substrate branching runs on) ---

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
        for src in spec.paths:
            dest = f"{snap_dir}/{PurePosixPath(src).name}"
            cmds.append(f'sqlite3 {shlex.quote(src)} ".backup {shlex.quote(dest)}"')
        command = " && ".join(cmds)
        result = await self._sandbox.exec(command, timeout_sec=120)
        if result.return_code != 0:
            raise EnvironmentSnapshotError(
                f"snapshot failed for environment '{self._manifest.name}' "
                f"(exit_code={result.return_code}): "
                f"{(result.stderr or result.stdout or '').strip()[:500]}",
                command=command,
                exit_code=result.return_code,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
            )
        return StateSnapshot(id=snap_id, path=snap_dir)

    async def restore(self, snap: StateSnapshot) -> None:
        """Roll the environment's state back to a snapshot.

        Copies each captured DB file from the snapshot directory back over
        its live path. The caller quiesces the agent and services first
        (the Branch lifecycle) so the copy is consistent.

        Issue #387: if the sandbox ``cp`` command fails (snapshot directory
        missing, destination not writable), raise
        ``EnvironmentSnapshotError`` rather than silently returning success
        — the live state is then unchanged and the caller must treat it as
        an infra failure, not a successful rollback.
        """
        spec = self._manifest.state
        if spec is None:
            raise RuntimeError(
                f"environment '{self._manifest.name}' declares no "
                "[environment.state]; snapshot/restore are unsupported for a "
                "stateless environment"
            )
        cmds = []
        for dst in spec.paths:
            src = f"{snap.path}/{PurePosixPath(dst).name}"
            cmds.append(f"cp {shlex.quote(src)} {shlex.quote(dst)}")
        command = " && ".join(cmds)
        result = await self._sandbox.exec(command, timeout_sec=120)
        if result.return_code != 0:
            raise EnvironmentSnapshotError(
                f"restore failed for environment '{self._manifest.name}' "
                f"snapshot {snap.id!r} (exit_code={result.return_code}): "
                f"{(result.stderr or result.stdout or '').strip()[:500]}",
                command=command,
                exit_code=result.return_code,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
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
                    await self._sandbox.exec(
                        f"pkill -f {shlex.quote(binary)}", timeout_sec=10
                    )
        # 2. Restore the per-task baseline state, if any.
        if self._baseline is not None:
            await self.restore(self._baseline)
        # 3. Restart the same services we previously started, in the same
        #    background-with-log shape provision() uses. owns_lifecycle = true
        #    manifests reach this with an empty self._started — nothing to do.
        for svc in self._started:
            log = f"/tmp/benchflow-env-{svc.name}.log"
            await self._sandbox.exec(
                f"{svc.command} > {log} 2>&1 &",
                timeout_sec=15,
            )
