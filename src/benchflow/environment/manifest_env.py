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

The sidecar / shared-fleet topology (host-exposed ports, ``wait_for_readiness``
over httpx) and environment-state ``snapshot``/``restore`` are the platform
layer; this adapter raises ``NotImplementedError`` for the latter.
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

    async def provision(self, ctx: Any) -> EnvHandle:
        """Start the manifest's services inside the sandbox.

        When ``owns_lifecycle`` is true the image's entrypoint already
        started the services, so this only builds the handle. Otherwise
        each service whose binary is on ``PATH`` is started in the
        background; services absent from a per-task image are skipped.
        """
        m = self._manifest
        self._started = []
        if not m.owns_lifecycle:
            for svc in m.services:
                binary = svc.command.split()[0]
                present = await self._sandbox.exec(
                    f"command -v {shlex.quote(binary)} >/dev/null 2>&1",
                    timeout_sec=10,
                )
                if present.return_code != 0:
                    continue  # not in this per-task image — skip
                log = f"/tmp/benchflow-env-{svc.name}.log"
                await self._sandbox.exec(
                    f"{svc.command} > {log} 2>&1 &",
                    timeout_sec=15,
                )
                self._started.append(svc)
        endpoints = {p: f"http://localhost:{p}" for p in m.all_ports}
        self._handle = EnvHandle(name=m.name, endpoints=endpoints)
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
            urls = [
                f"http://localhost:{s.port}{s.health_path}"
                for s in self._started
            ]

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
        spec = self._manifest.state
        if spec is None:
            raise RuntimeError(
                f"environment '{self._manifest.name}' declares no "
                "[environment.state]; snapshot/restore are unsupported for a "
                "stateless environment"
            )
        snap_id = uuid4().hex[:12]
        snap_dir = f"/tmp/benchflow-snapshots/{snap_id}"
        cmds = [f"mkdir -p {shlex.quote(snap_dir)}"]
        for src in spec.paths:
            dest = f"{snap_dir}/{PurePosixPath(src).name}"
            cmds.append(
                f'sqlite3 {shlex.quote(src)} ".backup {shlex.quote(dest)}"'
            )
        await self._sandbox.exec(" && ".join(cmds), timeout_sec=120)
        return StateSnapshot(id=snap_id, path=snap_dir)

    async def restore(self, snap: StateSnapshot) -> None:
        """Roll the environment's state back to a snapshot.

        Copies each captured DB file from the snapshot directory back over
        its live path. The caller quiesces the agent and services first
        (the Branch lifecycle) so the copy is consistent.
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
        await self._sandbox.exec(" && ".join(cmds), timeout_sec=120)

    async def reset(self) -> None:
        raise NotImplementedError("environment reset is not yet implemented")
