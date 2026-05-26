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

# ── 1. Protocol surface ──────────────────────────────────────────────────


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


# ── 2. Capability declarations on concrete backends ──────────────────────


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


# ── 3. Default fails closed with SandboxSnapshotNotSupported ─────────────


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


# ── 4. Branch fails closed on unsupported providers ──────────────────────


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


# ── 5. Workspace helper is scope-renamed, alias preserved ─────────────────


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
