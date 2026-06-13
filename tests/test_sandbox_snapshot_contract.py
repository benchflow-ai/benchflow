"""Sandbox snapshot/restore contract conformance (#384).

Guards the fix from PR for issue #384 against a regression where
``snapshot``/``restore`` were free functions over ``env.exec()`` rather
than methods on the Sandbox contract the kernel uses.

The contract surface (``docs/architecture.md``, "The four contracts"):

* ``Sandbox.snapshot()`` / ``Sandbox.restore(image)`` are real methods.
* ``Sandbox.supports_snapshot`` is the capability gate.
* Providers that cannot snapshot the container layer raise
  :class:`SandboxSnapshotNotSupported` from both methods.
* ``Rollout.branch(require_sandbox_snapshot=True)`` fails closed on those
  providers with a clear diagnostic.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchflow.sandbox.protocol import (
    Sandbox,
    SandboxImage,
    SandboxSnapshotNotSupported,
)

# 1. Protocol surface


class TestSandboxProtocolHasSnapshot:
    """The Sandbox Protocol exposes snapshot/restore — the kernel can rely on it."""

    def test_protocol_has_snapshot(self):
        assert hasattr(Sandbox, "snapshot")

    def test_protocol_has_restore(self):
        assert hasattr(Sandbox, "restore")

    def test_protocol_has_supports_snapshot(self):
        assert hasattr(Sandbox, "supports_snapshot")

    def test_sandbox_image_is_provider_scoped(self):
        img = SandboxImage(
            provider="docker", ref="bf-snap-foo", meta={"digest": "sha256:abc"}
        )
        assert img.provider == "docker"
        assert img.ref == "bf-snap-foo"
        assert img.meta == {"digest": "sha256:abc"}

    def test_sandbox_image_is_frozen(self):
        img = SandboxImage(provider="docker", ref="x")
        with pytest.raises((AttributeError, TypeError)):
            img.ref = "y"  # type: ignore[misc]


# 2. Capability declarations on concrete backends


class TestDockerSnapshotCapability:
    """DockerSandbox declares snapshot support — implemented via ``docker commit``."""

    def test_docker_supports_snapshot(self):
        from benchflow.sandbox.docker import DockerSandbox

        # Class-level capability — true for every DockerSandbox instance.
        assert DockerSandbox.supports_snapshot is not False  # property descriptor
        # Build a minimal instance and probe the property.
        # Avoid touching the real constructor: just call the property descriptor.
        prop = DockerSandbox.__dict__["supports_snapshot"]
        result = prop.fget(None)  # type: ignore[arg-type]
        assert result is True

    def test_docker_snapshot_is_async(self):
        from benchflow.sandbox.docker import DockerSandbox

        assert asyncio.iscoroutinefunction(DockerSandbox.snapshot)
        assert asyncio.iscoroutinefunction(DockerSandbox.restore)


_daytona_available = True
try:
    import daytona as _daytona_mod  # noqa: F401
except ImportError:
    _daytona_available = False


@pytest.mark.skipif(not _daytona_available, reason="daytona not installed")
class TestDaytonaSnapshotCapability:
    """Daytona direct = supported (native API); DinD = unsupported."""

    def test_daytona_direct_supports_snapshot(self):
        from benchflow.sandbox.daytona import _DaytonaDirect

        assert _DaytonaDirect.supports_snapshot is True

    def test_daytona_dind_does_not_support_snapshot(self):
        from benchflow.sandbox.daytona import _DaytonaDinD

        # DinD/compose cannot satisfy provider-level snapshot today (#384).
        assert _DaytonaDinD.supports_snapshot is False

    def test_daytona_sandbox_snapshot_is_async(self):
        from benchflow.sandbox.daytona import DaytonaSandbox

        assert asyncio.iscoroutinefunction(DaytonaSandbox.snapshot)
        assert asyncio.iscoroutinefunction(DaytonaSandbox.restore)


_modal_available = True
try:
    import modal as _modal_mod  # noqa: F401
except ImportError:
    _modal_available = False


@pytest.mark.skipif(not _modal_available, reason="modal not installed")
class TestModalSnapshotCapability:
    """Modal has no provider-level snapshot — declares unsupported."""

    def test_modal_does_not_support_snapshot(self):
        from benchflow.sandbox.modal_impl import ModalSandbox

        # ModalSandbox inherits the default (False) from BaseSandbox.
        prop = ModalSandbox.__dict__.get("supports_snapshot") or ModalSandbox.__mro__[
            1
        ].__dict__.get("supports_snapshot")
        result = prop.fget(None)  # type: ignore[arg-type, union-attr]
        assert result is False


# 3. Default fails closed with SandboxSnapshotNotSupported


class _UnsupportedSandbox:
    """Bare object exercising BaseSandbox's default snapshot/restore."""

    # Inherit the methods directly off BaseSandbox so the default path is
    # what's under test — without dragging the whole __init__ in.
    pass


async def test_base_sandbox_default_snapshot_raises_unsupported():
    """The default ``snapshot`` raises so naive new backends fail closed."""
    from benchflow.sandbox._base import BaseSandbox

    # Bind the unbound method to a sentinel object; the body never touches
    # ``self`` state, only the type name in the error message.
    class _Dummy:
        __class__ = type("FakeSandbox", (), {})  # for the error message

    with pytest.raises(SandboxSnapshotNotSupported, match="does not support"):
        await BaseSandbox.snapshot(_Dummy())  # type: ignore[arg-type]


async def test_base_sandbox_default_restore_raises_unsupported():
    """The default ``restore`` raises so naive new backends fail closed."""
    from benchflow.sandbox._base import BaseSandbox

    class _Dummy:
        pass

    img = SandboxImage(provider="docker", ref="x")
    with pytest.raises(SandboxSnapshotNotSupported, match="does not support"):
        await BaseSandbox.restore(_Dummy(), img)  # type: ignore[arg-type]


# 4. Branch fails closed on unsupported providers


class _FakeRolloutEnv:
    """Minimal Environment for the Branch engine — records snapshot calls."""

    async def snapshot(self):
        from benchflow.environment.protocol import StateSnapshot

        return StateSnapshot(id="snap-1", path="/tmp/x")

    async def restore(self, snap):
        pass


class _FakeRollout:
    """A stand-in just rich enough for the require_sandbox_snapshot gate."""

    def __init__(self, sandbox_supports: bool):
        from benchflow.trajectories.tree import RolloutTree

        self._tree = RolloutTree()
        self._cursor = self._tree.root
        self._environment = _FakeRolloutEnv()

        class _FakeSandbox:
            supports_snapshot = sandbox_supports

        self._env = _FakeSandbox()
        self._trajectory: list = []
        self._n_tool_calls = 0
        self._phase = "ready"
        self._rewards = None
        self._trajectory_source = None
        self._partial_trajectory = False
        self._session_tool_count = 0
        self._session_traj_count = 0
        self._executed_prompts: list[str] = []

    async def disconnect(self):
        pass


async def test_branch_fails_closed_when_sandbox_snapshot_required_but_unsupported():
    """``require_sandbox_snapshot=True`` rejects providers without snapshot."""
    from benchflow.rollout_branch import branch

    rollout = _FakeRollout(sandbox_supports=False)
    with pytest.raises(RuntimeError, match="container-level snapshot/restore"):
        await branch(rollout, n=2, require_sandbox_snapshot=True)  # type: ignore[arg-type]


async def test_branch_does_not_require_sandbox_snapshot_by_default():
    """Backwards-compat: the existing Environment-only path is unchanged."""
    from benchflow.rollout_branch import branch

    rollout = _FakeRollout(sandbox_supports=False)

    async def _runner(child):
        return 0.5

    # Without the flag, the engine continues — the env-only path still works.
    # Children run via the injected runner so this exercises only the gate.
    value = await branch(rollout, n=2, run_child=_runner)  # type: ignore[arg-type]
    assert value == pytest.approx(0.5)


# 4b. Docker snapshot images are labelled + reaped (no bf-snap-* leak)


class TestDockerSnapshotImageLifecycle:
    """``docker commit`` snapshots must be ownership-labelled and reaped.

    Regression for the ``bf-snap-*`` image leak: every snapshot image is
    stamped ``benchflow.owned=true`` (so the label-scoped reaper can find it)
    and ``stop(delete=True)`` removes the tags this instance committed.
    """

    def _make_sandbox(self):
        import logging

        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox._keep_containers = False
        sandbox._snapshot_tags = []
        sandbox.environment_name = "demo-env"
        sandbox.logger = logging.getLogger("benchflow.test")
        return sandbox

    @pytest.mark.asyncio
    async def test_snapshot_commit_carries_owned_label_and_tracks_tag(
        self, monkeypatch
    ):
        from benchflow.sandbox import docker as docker_mod

        sandbox = self._make_sandbox()

        async def fake_main_container_id():
            return "container-123"

        sandbox._main_container_id = fake_main_container_id  # type: ignore[method-assign]

        captured: list[list[str]] = []

        class _FakeProc:
            returncode = 0

            async def communicate(self):
                return (b"sha256:deadbeef\n", b"")

        async def fake_exec(*args, **_kwargs):
            captured.append(list(args))
            return _FakeProc()

        monkeypatch.setattr(docker_mod.asyncio, "create_subprocess_exec", fake_exec)

        image = await sandbox.snapshot(name="snap1")

        # The commit invocation stamps the ownership label via --change so the
        # label-scoped reaper can reclaim a leaked snapshot image.
        assert captured, "docker commit was never invoked"
        commit_cmd = captured[0]
        assert commit_cmd[:2] == ["docker", "commit"]
        assert "--change" in commit_cmd
        idx = commit_cmd.index("--change")
        assert commit_cmd[idx + 1] == "LABEL benchflow.owned=true"

        # The tag is tracked so stop(delete=True) can remove it later.
        assert image.ref in sandbox._snapshot_tags
        assert image.ref.startswith("bf-snap-demo-env-snap1")

    @pytest.mark.asyncio
    async def test_remove_snapshot_images_rms_tracked_tags(self):
        sandbox = self._make_sandbox()
        sandbox._snapshot_tags = ["bf-snap-a", "bf-snap-b"]

        removed: list[list[str]] = []

        async def fake_docker_cli(args, check=True):
            from benchflow.sandbox._base import ExecResult

            removed.append(args)
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._docker_cli = fake_docker_cli  # type: ignore[method-assign]

        await sandbox._remove_snapshot_images()

        assert removed == [
            ["image", "rm", "-f", "bf-snap-a"],
            ["image", "rm", "-f", "bf-snap-b"],
        ]
        # Tracking list is cleared so a later teardown is a no-op.
        assert sandbox._snapshot_tags == []

    @pytest.mark.asyncio
    async def test_remove_snapshot_images_respects_keep_containers(self):
        sandbox = self._make_sandbox()
        sandbox._keep_containers = True
        sandbox._snapshot_tags = ["bf-snap-keep"]

        removed: list[list[str]] = []

        async def fake_docker_cli(args, check=True):
            removed.append(args)

        sandbox._docker_cli = fake_docker_cli  # type: ignore[method-assign]

        await sandbox._remove_snapshot_images()

        # keep_containers => keep the snapshot image too; nothing removed.
        assert removed == []
        assert sandbox._snapshot_tags == ["bf-snap-keep"]


# 4c. Docker restore is spec-faithful + reaper-visible

# The runtime spec a live ``main`` container would report from
# ``docker inspect`` — env, cwd, user, cpu/memory limits, volume binds.
_RESTORE_INSPECT = [
    {
        "Config": {
            "Env": ["PATH=/usr/bin", "FOO=bar", "BENCHFLOW_TASK=demo"],
            "WorkingDir": "/testbed",
            "User": "agent",
            "Cmd": ["sh", "-c", "sleep infinity"],
        },
        "HostConfig": {
            "NetworkMode": "sess-id-123_default",
            "NanoCpus": 2_000_000_000,
            "Memory": 2147483648,
            "Binds": ["/host/rollout:/sandbox/rollout"],
        },
    }
]


class TestDockerRestoreFidelity:
    """``restore()`` must recreate ``main`` faithfully and ownership-labelled.

    Regression for two defects in the prior ``docker run ... sleep infinity``
    restore path: it carried NONE of the original container spec (env, cwd,
    user, cpu/memory limits, volume binds) and never stamped
    ``benchflow.owned=true`` — so a restored container silently changed the
    agent's environment AND was invisible to the label-scoped reaper /
    ``bench environment cleanup`` (both filter on that label), leaking.
    """

    def _make_sandbox(self):
        import logging

        from benchflow.sandbox.docker import DockerSandbox

        sandbox = DockerSandbox.__new__(DockerSandbox)
        sandbox.session_id = "Sess-ID/123"  # exercises project-name sanitization
        sandbox.environment_name = "demo-env"
        sandbox.logger = logging.getLogger("benchflow.test")
        return sandbox

    async def _run_restore(self, sandbox):
        """Drive ``restore`` with stubbed docker calls; return the run argv."""
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.protocol import SandboxImage

        async def fake_main_container_id():
            return "container-abc"

        sandbox._main_container_id = fake_main_container_id  # type: ignore[method-assign]

        calls: list[list[str]] = []

        async def fake_docker_cli(args, check=True):
            calls.append(list(args))
            if args[:1] == ["inspect"]:
                return ExecResult(
                    stdout=json.dumps(_RESTORE_INSPECT), stderr="", return_code=0
                )
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._docker_cli = fake_docker_cli  # type: ignore[method-assign]

        image = SandboxImage(provider="docker", ref="bf-snap-demo-env-snap1")
        await sandbox.restore(image)

        run_cmd = next(c for c in calls if c[:1] == ["run"])
        return calls, run_cmd

    @pytest.mark.asyncio
    async def test_restore_run_carries_owned_label(self):
        sandbox = self._make_sandbox()
        _calls, run_cmd = await self._run_restore(sandbox)

        # (1) Ownership label so the reaper / cleanup can reclaim it.
        idxs = [i for i, a in enumerate(run_cmd) if a == "--label"]
        labels = {run_cmd[i + 1] for i in idxs}
        assert "benchflow.owned=true" in labels
        # Compose project/service labels remain so it joins the project view.
        assert "com.docker.compose.service=main" in labels
        assert any(v.startswith("com.docker.compose.project=") for v in labels)

    @pytest.mark.asyncio
    async def test_restore_run_preserves_original_spec(self):
        sandbox = self._make_sandbox()
        _calls, run_cmd = await self._run_restore(sandbox)

        # (2) env / cwd / user / limits / binds carried from the inspected
        # container — NOT a bare ``sleep infinity`` with nothing else.
        def _flag_values(flag: str) -> list[str]:
            return [run_cmd[i + 1] for i, a in enumerate(run_cmd) if a == flag]

        env_values = _flag_values("--env")
        assert "FOO=bar" in env_values
        assert "BENCHFLOW_TASK=demo" in env_values

        assert "--workdir" in run_cmd
        assert _flag_values("--workdir") == ["/testbed"]

        assert "--user" in run_cmd
        assert _flag_values("--user") == ["agent"]

        # NanoCpus 2e9 -> --cpus 2 ; Memory bytes preserved verbatim.
        assert _flag_values("--cpus") == ["2"]
        assert _flag_values("--memory") == ["2147483648"]

        # Volume bind preserved.
        assert "/host/rollout:/sandbox/rollout" in _flag_values("--volume")

        # Network from the inspected NetworkMode.
        assert "sess-id-123_default" in _flag_values("--network")

        # Original command preserved verbatim (still stays up for exec).
        assert run_cmd[-3:] == ["sh", "-c", "sleep infinity"]

        # The image ref appears after the run flags and before the command.
        assert "bf-snap-demo-env-snap1" in run_cmd
        assert run_cmd.index("bf-snap-demo-env-snap1") == len(run_cmd) - 4

    @pytest.mark.asyncio
    async def test_restore_stops_and_removes_old_container_before_run(self):
        sandbox = self._make_sandbox()
        calls, _run_cmd = await self._run_restore(sandbox)

        verbs = [c[0] for c in calls]
        # inspect (capture spec) -> stop -> rm -> run, in that order.
        assert verbs.index("inspect") < verbs.index("stop")
        assert verbs.index("stop") < verbs.index("rm")
        assert verbs.index("rm") < verbs.index("run")

    @pytest.mark.asyncio
    async def test_restore_falls_back_when_inspect_unavailable(self):
        """No live container / failed inspect => still labelled + stays up."""
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.protocol import SandboxImage

        sandbox = self._make_sandbox()

        async def no_container():
            return None

        sandbox._main_container_id = no_container  # type: ignore[method-assign]

        calls: list[list[str]] = []

        async def fake_docker_cli(args, check=True):
            calls.append(list(args))
            return ExecResult(stdout="", stderr="", return_code=0)

        sandbox._docker_cli = fake_docker_cli  # type: ignore[method-assign]

        await sandbox.restore(SandboxImage(provider="docker", ref="bf-snap-x"))

        run_cmd = next(c for c in calls if c[:1] == ["run"])
        labels = {run_cmd[i + 1] for i, a in enumerate(run_cmd) if a == "--label"}
        assert "benchflow.owned=true" in labels
        # Falls back to the compose project network + sleep-infinity command.
        assert run_cmd[-3:] == ["sh", "-c", "sleep infinity"]
        assert any("_default" in v for v in run_cmd)


# 5. Workspace helper is scope-renamed, alias preserved


class TestWorkspaceHelperScoped:
    """``benchflow.sandbox.snapshot`` is now explicitly workspace-only (#384)."""

    def test_new_workspace_names_exported(self):
        import benchflow

        assert hasattr(benchflow, "workspace_snapshot")
        assert hasattr(benchflow, "workspace_restore")
        assert hasattr(benchflow, "list_workspace_snapshots")

    def test_legacy_aliases_preserved(self):
        import benchflow
        from benchflow.sandbox.snapshot import (
            workspace_restore,
            workspace_snapshot,
        )

        # Pre-#384 names stay as aliases so the proof script and downstream
        # callers keep working.
        assert benchflow.snapshot is workspace_snapshot
        assert benchflow.restore is workspace_restore


# 6. Daytona cloud-snapshot leak: track on create, delete on teardown
#
# Regression for the ``bf-snap-*`` *cloud* snapshot leak: ``DaytonaSandbox``
# creates Daytona snapshots that the sandbox reaper never touches, so an
# untracked / undeleted one leaks against the account quota indefinitely. The
# tests below mock the Daytona SDK (the async client + sandbox handle) — they
# are SDK-gated like the rest of this module and never reach live cloud.


@pytest.mark.skipif(not _daytona_available, reason="daytona not installed")
class TestDaytonaSnapshotLifecycle:
    """Daytona snapshots are tracked on create and deleted on teardown."""

    def _make_strategy(self, *, with_rollout_dir=None):
        import logging
        from types import SimpleNamespace

        from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDirect

        env = DaytonaSandbox.__new__(DaytonaSandbox)
        env.environment_name = "demo-env"
        env.logger = logging.getLogger("benchflow.test")
        env._sandbox = None
        env._client_manager = None
        env.rollout_paths = (
            SimpleNamespace(rollout_dir=with_rollout_dir)
            if with_rollout_dir is not None
            else None
        )
        strategy = _DaytonaDirect(env)
        env._strategy = strategy
        return env, strategy

    @pytest.mark.asyncio
    async def test_snapshot_records_created_name(self):
        env, strategy = self._make_strategy()

        created: list[str] = []

        class _FakeSandbox:
            id = "sb-1"

            async def _experimental_create_snapshot(self, name):
                created.append(name)

        env._sandbox = _FakeSandbox()

        image = await strategy.snapshot(name="snap1")

        # The create call was issued and the name tracked for teardown deletion.
        assert created == ["snap1"]
        assert image.ref == "snap1"
        assert strategy._created_snapshots == ["snap1"]

    @pytest.mark.asyncio
    async def test_default_snapshot_name_uses_prefix(self):
        env, strategy = self._make_strategy()

        class _FakeSandbox:
            id = "sb-1"

            async def _experimental_create_snapshot(self, name):
                pass

        env._sandbox = _FakeSandbox()

        image = await strategy.snapshot()

        assert image.ref.startswith("bf-snap-demo-env-")
        assert strategy._created_snapshots == [image.ref]

    @pytest.mark.asyncio
    async def test_stop_deletes_tracked_snapshots(self):
        env, strategy = self._make_strategy()
        strategy._created_snapshots = ["bf-snap-a", "bf-snap-b"]

        got: list[str] = []
        deleted: list[str] = []

        class _FakeSnapshotService:
            async def get(self, name):
                got.append(name)
                return SimpleNamespaceSnapshot(name)

            async def delete(self, snap):
                deleted.append(snap.name)

        class _FakeClient:
            snapshot = _FakeSnapshotService()

        class _FakeManager:
            async def get_client(self):
                return _FakeClient()

        env._client_manager = _FakeManager()

        # stop() with no live sandbox still runs the snapshot-delete tier.
        await strategy.stop(delete=True)

        assert got == ["bf-snap-a", "bf-snap-b"]
        assert deleted == ["bf-snap-a", "bf-snap-b"]
        # Successful delete clears tracking so a later teardown is a no-op.
        assert strategy._created_snapshots == []

    @pytest.mark.asyncio
    async def test_stop_already_gone_snapshot_is_not_a_leak(self):
        env, strategy = self._make_strategy()
        strategy._created_snapshots = ["bf-snap-gone"]

        class _FakeSnapshotService:
            async def get(self, name):
                raise RuntimeError("404 not found")

            async def delete(self, snap):  # pragma: no cover - never reached
                raise AssertionError("delete should not run when get() fails")

        class _FakeClient:
            snapshot = _FakeSnapshotService()

        class _FakeManager:
            async def get_client(self):
                return _FakeClient()

        env._client_manager = _FakeManager()

        await strategy.stop(delete=True)

        # A snapshot the server no longer has is treated as already-reaped.
        assert strategy._created_snapshots == []

    @pytest.mark.asyncio
    async def test_failed_delete_is_recorded_and_kept(self, tmp_path):
        env, strategy = self._make_strategy(with_rollout_dir=tmp_path)
        strategy._created_snapshots = ["bf-snap-stuck"]

        class _FakeSnapshotService:
            async def get(self, name):
                return SimpleNamespaceSnapshot(name)

            async def delete(self, snap):
                raise RuntimeError("api 500")

        class _FakeClient:
            snapshot = _FakeSnapshotService()

        class _FakeManager:
            async def get_client(self):
                return _FakeClient()

        env._client_manager = _FakeManager()

        await strategy.stop(delete=True)

        # An undeletable snapshot stays tracked AND is recorded for post-mortem.
        assert strategy._created_snapshots == ["bf-snap-stuck"]
        leak_report = tmp_path / "snapshot_leaks.json"
        assert leak_report.is_file()
        data = json.loads(leak_report.read_text())
        assert data[-1]["snapshot_names"] == ["bf-snap-stuck"]
        assert data[-1]["provider"] == "DaytonaSandbox"


class SimpleNamespaceSnapshot:
    """Minimal stand-in for the SDK ``Snapshot`` (carries ``name``/``id``)."""

    def __init__(self, name):
        self.name = name
        self.id = f"id-{name}"
        self.state = "active"


@pytest.mark.skipif(not _daytona_available, reason="daytona not installed")
class TestDaytonaSnapshotReaper:
    """The snapshot reaper deletes only ``bf-snap-*`` (no labels exist)."""

    def _client(self, names):
        from types import SimpleNamespace

        snaps = [SimpleNamespaceSnapshot(n) for n in names]
        deleted: list[str] = []

        class _SnapshotService:
            def list(self, page=None, limit=None):
                # Single page; mirrors PaginatedSnapshots shape.
                return SimpleNamespace(
                    items=list(snaps), total=len(snaps), page=1, total_pages=1
                )

            def delete(self, snap):
                deleted.append(snap.name)

        client = SimpleNamespace(snapshot=_SnapshotService())
        return client, deleted

    def test_reaper_deletes_only_owned_prefix(self):
        from benchflow.sandbox.daytona import reap_leaked_snapshots

        client, deleted = self._client(
            ["bf-snap-x", "someone-elses-snapshot", "bf-snap-y"]
        )
        counts = reap_leaked_snapshots(client)

        # Foreign snapshot is never deleted; both benchflow-owned ones are.
        assert sorted(deleted) == ["bf-snap-x", "bf-snap-y"]
        assert counts["found"] == 3
        assert counts["deleted"] == 2
        assert counts["skipped"] == 1
        assert counts["failed"] == 0

    def test_reaper_dry_run_deletes_nothing(self):
        from benchflow.sandbox.daytona import reap_leaked_snapshots

        client, deleted = self._client(["bf-snap-x"])
        counts = reap_leaked_snapshots(client, dry_run=True)

        assert deleted == []
        assert counts["deleted"] == 1  # counted as would-delete
