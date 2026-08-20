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


# 4b. Docker restore re-creates an equivalent container


_LIVE_CONTAINER = {
    "Id": "abc123",
    "Config": {"WorkingDir": "/app", "User": "agent", "Env": ["FOO=bar"]},
    "HostConfig": {
        "NetworkMode": "bf-proj_default",
        "NanoCpus": 2_000_000_000,
        "Memory": 4294967296,
    },
    "Mounts": [
        {
            "Type": "bind",
            "Source": "/host/jobs/run/verifier",
            "Destination": "/logs/verifier",
            "RW": True,
        },
        {
            "Type": "bind",
            "Source": "/host/jobs/run/agent",
            "Destination": "/logs/agent",
            "RW": True,
        },
        {
            "Type": "bind",
            "Source": "/host/readonly",
            "Destination": "/opt/fixtures",
            "RW": False,
        },
        {
            "Type": "volume",
            "Name": "pgdata",
            "Source": "/var/lib/docker/volumes/pgdata/_data",
            "Destination": "/var/lib/postgresql/data",
            "RW": True,
        },
    ],
}


def _docker_sandbox(tmp_path):
    """A DockerSandbox instance — construction only, no daemon contact."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig
    from benchflow.task.paths import RolloutPaths

    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text("FROM alpine:3.20\n")
    rollout_paths = RolloutPaths(rollout_dir=tmp_path / "run")
    rollout_paths.mkdir()
    return DockerSandbox(
        environment_dir=env_dir,
        environment_name="snapshot-contract",
        session_id="bf-snapshot-contract",
        rollout_paths=rollout_paths,
        task_env_config=SandboxConfig(),
    )


class TestReplayedRunArgs:
    """The pure reconstruction: inspect output -> ``docker run`` flags."""

    def test_bind_mounts_are_replayed_with_their_host_paths(self):
        from benchflow.sandbox.docker import _replayed_run_args

        args = _replayed_run_args(_LIVE_CONTAINER, default_network="bf-proj_default")

        assert "--mount" in args
        specs = [args[i + 1] for i, a in enumerate(args) if a == "--mount"]
        assert "type=bind,src=/host/jobs/run/verifier,dst=/logs/verifier" in specs
        assert "type=bind,src=/host/jobs/run/agent,dst=/logs/agent" in specs

    def test_a_read_only_bind_stays_read_only(self):
        from benchflow.sandbox.docker import _replayed_run_args

        args = _replayed_run_args(_LIVE_CONTAINER, default_network="bf-proj_default")
        specs = [args[i + 1] for i, a in enumerate(args) if a == "--mount"]

        assert "type=bind,src=/host/readonly,dst=/opt/fixtures,readonly" in specs

    def test_a_named_volume_is_replayed_by_name_not_by_its_data_path(self):
        from benchflow.sandbox.docker import _replayed_run_args

        args = _replayed_run_args(_LIVE_CONTAINER, default_network="bf-proj_default")
        specs = [args[i + 1] for i, a in enumerate(args) if a == "--mount"]

        assert "type=volume,src=pgdata,dst=/var/lib/postgresql/data" in specs

    def test_resource_limits_and_the_project_network_are_replayed(self):
        from benchflow.sandbox.docker import _replayed_run_args

        args = _replayed_run_args(_LIVE_CONTAINER, default_network="bf-proj_default")

        assert args[:2] == ["--network", "bf-proj_default"]
        assert args[args.index("--cpus") + 1] == "2"
        assert args[args.index("--memory") + 1] == "4294967296"

    def test_a_container_with_no_network_does_not_get_one_back(self):
        """A task that opted out of networking must stay opted out."""
        from benchflow.sandbox.docker import _replayed_run_args

        container = {**_LIVE_CONTAINER, "HostConfig": {"NetworkMode": "none"}}

        args = _replayed_run_args(container, default_network="bf-proj_default")

        assert args[:2] == ["--network", "none"]
        assert "bf-proj_default" not in args

    def test_an_empty_host_config_replays_only_the_default_network(self):
        from benchflow.sandbox.docker import _replayed_run_args

        args = _replayed_run_args({}, default_network="bf-proj_default")

        assert args == ["--network", "bf-proj_default"]

    def test_a_tmpfs_mount_is_replayed_as_tmpfs(self):
        from benchflow.sandbox.docker import _replayed_run_args

        container = {"Mounts": [{"Type": "tmpfs", "Destination": "/scratch"}]}

        args = _replayed_run_args(container, default_network="net")

        assert args[-2:] == ["--tmpfs", "/scratch"]


class TestDockerRestoreRebuildsTheContainer:
    """``restore()`` must produce a container equivalent to the one it replaces.

    The live regression: the replacement was created with only ``--network``
    and two labels, so the rollout's ``/logs`` bind mounts were dropped. The
    verifier then wrote ``reward.txt`` into a container-local directory, the
    host saw nothing, and the branch child was reported as ``0.00``.
    """

    async def _restore_with(
        self, tmp_path, monkeypatch, inspect_result, container_id="abc123"
    ):
        import json as _json

        from benchflow.sandbox._base import ExecResult

        sandbox = _docker_sandbox(tmp_path)
        calls: list[list[str]] = []

        async def fake_main_container_id():
            return container_id

        async def fake_docker_cli(args, check=True):
            calls.append(list(args))
            if args[0] == "inspect":
                if isinstance(inspect_result, int):
                    return ExecResult(stdout="", stderr="boom", return_code=1)
                return ExecResult(
                    stdout=_json.dumps(inspect_result), stderr="", return_code=0
                )
            return ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(sandbox, "_main_container_id", fake_main_container_id)
        monkeypatch.setattr(sandbox, "_docker_cli", fake_docker_cli)
        await sandbox.restore(SandboxImage(provider="docker", ref="bf-snap-x"))
        return calls

    async def test_restore_replays_the_original_bind_mounts(
        self, tmp_path, monkeypatch
    ):
        calls = await self._restore_with(tmp_path, monkeypatch, [_LIVE_CONTAINER])

        run_cmd = next(call for call in calls if call[0] == "run")
        specs = [run_cmd[i + 1] for i, a in enumerate(run_cmd) if a == "--mount"]
        assert "type=bind,src=/host/jobs/run/verifier,dst=/logs/verifier" in specs
        assert "type=bind,src=/host/jobs/run/agent,dst=/logs/agent" in specs
        # the compose identity the engine relies on is still there
        assert "com.docker.compose.service=main" in run_cmd
        assert run_cmd[-2:] == ["sleep", "infinity"]

    async def test_the_container_is_inspected_before_it_is_removed(
        self, tmp_path, monkeypatch
    ):
        """Ordering is the whole trick: a removed container has no host config."""
        calls = await self._restore_with(tmp_path, monkeypatch, [_LIVE_CONTAINER])

        verbs = [call[0] for call in calls]
        assert verbs.index("inspect") < verbs.index("stop") < verbs.index("rm")

    async def test_restore_fails_closed_when_the_container_cannot_be_inspected(
        self, tmp_path, monkeypatch
    ):
        """Unknown host config is not a licence to create a container without one."""
        with pytest.raises(RuntimeError, match="docker inspect"):
            await self._restore_with(tmp_path, monkeypatch, 1)

    async def test_restore_fails_closed_when_there_is_no_container_to_inspect(
        self, tmp_path, monkeypatch
    ):
        """The other half of "fails closed" — no ``main`` container at all.

        Guards the fix from "fix(sandbox): restore fails closed when the
        container cannot be inspected". The inspect-failure path above raised,
        but the *unresolvable* path logged a warning and fell through to
        ``replayed = ["--network", default_network]`` — a container with no
        bind mounts, which is precisely the regression the mount replay was
        added to fix: the verifier writes ``reward.txt`` into a container-local
        ``/logs``, the host sees nothing, and the branch child is reported as
        ``0.00``. A restore that cannot read the host config cannot reproduce
        it, so it must not create the replacement.
        """
        from benchflow.sandbox.protocol import SandboxRestoreHostConfigUnavailable

        with pytest.raises(SandboxRestoreHostConfigUnavailable) as excinfo:
            await self._restore_with(
                tmp_path, monkeypatch, [_LIVE_CONTAINER], container_id=None
            )

        # names what could not be resolved, not just that something failed
        assert "'main'" in str(excinfo.value)
        assert "bf-snap-x" in str(excinfo.value)
        assert isinstance(excinfo.value, RuntimeError)

    async def test_a_restore_that_fails_closed_leaves_no_container_behind(
        self, tmp_path, monkeypatch
    ):
        """Failing closed means nothing was created *or* destroyed.

        Guards the fix from "fix(sandbox): restore fails closed when the
        container cannot be inspected". A raise that had already run ``rm -f``
        would trade a mountless container for no container at all.
        """
        from benchflow.sandbox._base import ExecResult
        from benchflow.sandbox.protocol import SandboxRestoreHostConfigUnavailable

        sandbox = _docker_sandbox(tmp_path)
        calls: list[list[str]] = []

        async def fake_main_container_id():
            return None

        async def fake_docker_cli(args, check=True):
            calls.append(list(args))
            return ExecResult(stdout="", stderr="", return_code=0)

        monkeypatch.setattr(sandbox, "_main_container_id", fake_main_container_id)
        monkeypatch.setattr(sandbox, "_docker_cli", fake_docker_cli)

        with pytest.raises(SandboxRestoreHostConfigUnavailable):
            await sandbox.restore(SandboxImage(provider="docker", ref="bf-snap-x"))

        assert calls == []


class TestVerifierMountDecisionIsLive:
    """The "are the outputs already on the host" question is asked, not assumed."""

    async def test_a_stale_is_mounted_declaration_falls_back_to_downloading(
        self, tmp_path
    ):
        """``is_mounted`` is static; ``restore()`` can invalidate it mid-run."""
        from benchflow.task.paths import RolloutPaths
        from benchflow.task.verifier import Verifier

        class _Sandbox:
            is_mounted = True

            async def has_host_mount(self, *, host_dir, container_dir) -> bool:
                return False

        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "run")
        rollout_paths.mkdir()
        verifier = Verifier(object(), rollout_paths, _Sandbox())

        assert await verifier._verifier_outputs_are_mounted("main") is False

    async def test_a_live_mount_still_skips_the_download(self, tmp_path):
        from benchflow.task.paths import RolloutPaths
        from benchflow.task.verifier import Verifier

        class _Sandbox:
            is_mounted = True

            async def has_host_mount(self, *, host_dir, container_dir) -> bool:
                return True

        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "run")
        rollout_paths.mkdir()
        verifier = Verifier(object(), rollout_paths, _Sandbox())

        assert await verifier._verifier_outputs_are_mounted("main") is True

    async def test_a_live_check_that_raises_falls_back_to_downloading(self, tmp_path):
        """Fail toward the download: redundant is cheap, skipped loses the score."""
        from benchflow.task.paths import RolloutPaths
        from benchflow.task.verifier import Verifier

        class _Sandbox:
            is_mounted = True

            async def has_host_mount(self, *, host_dir, container_dir) -> bool:
                raise RuntimeError("daemon unreachable")

        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "run")
        rollout_paths.mkdir()
        verifier = Verifier(object(), rollout_paths, _Sandbox())

        assert await verifier._verifier_outputs_are_mounted("main") is False

    async def test_a_backend_without_a_live_check_keeps_its_declaration(self, tmp_path):
        from benchflow.task.paths import RolloutPaths
        from benchflow.task.verifier import Verifier

        class _Sandbox:
            is_mounted = True

        rollout_paths = RolloutPaths(rollout_dir=tmp_path / "run")
        rollout_paths.mkdir()
        verifier = Verifier(object(), rollout_paths, _Sandbox())

        assert await verifier._verifier_outputs_are_mounted("main") is True
        # a target service is never host-mounted, live check or not
        assert await verifier._verifier_outputs_are_mounted("target") is False

    async def test_docker_has_host_mount_reads_the_live_container(
        self, tmp_path, monkeypatch
    ):
        import json as _json

        from benchflow.sandbox._base import ExecResult

        sandbox = _docker_sandbox(tmp_path)
        mounted = tmp_path / "run" / "verifier"
        container = {
            "Mounts": [
                {
                    "Type": "bind",
                    "Source": str(mounted),
                    "Destination": "/logs/verifier",
                    "RW": True,
                }
            ]
        }

        async def fake_main_container_id():
            return "abc123"

        async def fake_docker_cli(args, check=True):
            return ExecResult(stdout=_json.dumps([container]), stderr="", return_code=0)

        monkeypatch.setattr(sandbox, "_main_container_id", fake_main_container_id)
        monkeypatch.setattr(sandbox, "_docker_cli", fake_docker_cli)

        assert (
            await sandbox.has_host_mount(
                host_dir=mounted, container_dir="/logs/verifier"
            )
            is True
        )
        # same destination, a different host directory: not our mount
        assert (
            await sandbox.has_host_mount(
                host_dir=tmp_path / "elsewhere", container_dir="/logs/verifier"
            )
            is False
        )


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
