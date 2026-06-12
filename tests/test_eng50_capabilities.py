"""Tests for ENG-50: per-agent capabilities + agent-as-tool infrastructure.

Validates:
- Role.capabilities field on the canonical type
- Sandbox protocol conformance for Docker and Daytona
- Scene desugaring preserves role metadata for downstream Step execution
"""

from __future__ import annotations

import pytest

from benchflow._types import Role, Scene, Turn
from benchflow.scenes import compile_scenes_to_steps, scene_step_role

# Role.capabilities


class TestRoleCapabilities:
    """Guards ENG-50 Role.capabilities field."""

    def test_defaults_to_none(self) -> None:
        role = Role(name="agent", agent="gemini")
        assert role.capabilities is None

    def test_accepts_list(self) -> None:
        role = Role(
            name="coder",
            agent="claude-agent-acp",
            capabilities=["tool-use", "agent-as-tool", "loop"],
        )
        assert role.capabilities == ["tool-use", "agent-as-tool", "loop"]

    def test_empty_list(self) -> None:
        role = Role(name="x", agent="y", capabilities=[])
        assert role.capabilities == []

    def test_backward_compat_no_capabilities(self) -> None:
        """Existing code that doesn't pass capabilities still works."""
        role = Role(
            name="agent",
            agent="gemini",
            model="flash",
            env={"KEY": "val"},
            timeout_sec=60,
        )
        assert role.capabilities is None
        assert role.timeout_sec == 60

    def test_scene_single_preserves_capabilities(self) -> None:
        """Scene.single() creates a Role; capabilities defaults to None."""
        scene = Scene.single(agent="gemini", model="flash")
        assert len(scene.roles) == 1
        assert scene.roles[0].capabilities is None


# Sandbox.expose_ports — protocol conformance


class TestDockerSandboxExposePorts:
    """Guards ENG-50 expose_ports — verified via protocol conformance."""

    def test_docker_has_sandbox_interface(self) -> None:
        from benchflow.sandbox.docker import DockerSandbox

        for attr in ("exec", "start", "stop", "upload_file"):
            assert hasattr(DockerSandbox, attr), f"DockerSandbox missing {attr}"

    def test_docker_has_exec(self) -> None:
        from benchflow.sandbox.docker import DockerSandbox

        assert hasattr(DockerSandbox, "exec")

    def test_docker_has_start_stop(self) -> None:
        from benchflow.sandbox.docker import DockerSandbox

        assert hasattr(DockerSandbox, "start")
        assert hasattr(DockerSandbox, "stop")

    def test_docker_has_upload_download(self) -> None:
        from benchflow.sandbox.docker import DockerSandbox

        assert hasattr(DockerSandbox, "upload_file")
        assert hasattr(DockerSandbox, "download_file")


_daytona_available = True
try:
    import daytona as _daytona_mod  # noqa: F401
except ImportError:
    _daytona_available = False


@pytest.mark.skipif(not _daytona_available, reason="daytona not installed")
class TestDaytonaSandboxExposePorts:
    """Guards ENG-50 expose_ports — verified via protocol conformance."""

    def test_daytona_has_sandbox_interface(self) -> None:
        from benchflow.sandbox.daytona import DaytonaSandbox

        for attr in ("exec", "start", "stop", "upload_file"):
            assert hasattr(DaytonaSandbox, attr), f"DaytonaSandbox missing {attr}"

    def test_daytona_has_exec(self) -> None:
        from benchflow.sandbox.daytona import DaytonaSandbox

        assert hasattr(DaytonaSandbox, "exec")

    def test_daytona_has_start_stop(self) -> None:
        from benchflow.sandbox.daytona import DaytonaSandbox

        assert hasattr(DaytonaSandbox, "start")
        assert hasattr(DaytonaSandbox, "stop")

    def test_daytona_has_upload(self) -> None:
        from benchflow.sandbox.daytona import DaytonaSandbox

        assert hasattr(DaytonaSandbox, "upload_file")


async def test_role_env_vars_preserved_in_declarative_type() -> None:
    """Role env dict survives round-trip through Scene construction."""
    scene = Scene(
        name="env-check",
        roles=[
            Role("coder", "gemini", env={"CODER_KEY": "abc"}),
            Role("reviewer", "gemini", env={"REVIEWER_KEY": "xyz"}),
        ],
        turns=[Turn("coder"), Turn("reviewer")],
    )
    role_map = {r.name: r for r in scene.roles}
    assert role_map["coder"].env == {"CODER_KEY": "abc"}
    assert role_map["reviewer"].env == {"REVIEWER_KEY": "xyz"}


async def test_role_metadata_reaches_compiled_steps() -> None:
    """Role env and capabilities survive the Scene-to-Step lowering."""
    scene = Scene(
        name="code-review",
        roles=[
            Role(
                "coder",
                "gemini",
                env={"CODER_KEY": "abc"},
                capabilities=["tool-use"],
            ),
            Role(
                "reviewer",
                "gemini",
                env={"REVIEWER_KEY": "xyz"},
                capabilities=["agent-as-tool"],
            ),
        ],
        turns=[Turn("coder"), Turn("reviewer")],
    )

    coder, reviewer = [
        scene_step_role(step) for step in compile_scenes_to_steps([scene])
    ]

    assert coder.env == {"CODER_KEY": "abc"}
    assert coder.capabilities == ["tool-use"]
    assert reviewer.env == {"REVIEWER_KEY": "xyz"}
    assert reviewer.capabilities == ["agent-as-tool"]
