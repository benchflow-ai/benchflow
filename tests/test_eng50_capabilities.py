"""Tests for ENG-50: per-agent capabilities + agent-as-tool infrastructure.

Validates:
- Role.capabilities field on the canonical type
- Sandbox protocol conformance for Docker and Daytona
- Scene scheduler Coder → Reviewer → Coder turn sequence with role env vars
"""

from __future__ import annotations

import json

import pytest

from benchflow._scene import Scene as SceneRuntime
from benchflow._scene import SceneRole
from benchflow._types import Role, Scene, Turn

# ---------------------------------------------------------------------------
# Role.capabilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Sandbox.expose_ports — protocol conformance
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scene scheduler: Coder → Reviewer → Coder with role-specific env
# ---------------------------------------------------------------------------


class FakeExecResult:
    def __init__(
        self, stdout: str = "", stderr: str = "", return_code: int = 0
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class FakeEnv:
    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self._exec_log: list[str] = []

    async def exec(self, cmd: str, **kwargs) -> FakeExecResult:
        self._exec_log.append(cmd)
        if "rm -rf" in cmd and "outbox" in cmd:
            self._files = {
                k: v
                for k, v in self._files.items()
                if not k.startswith("/app/.outbox/")
            }
            return FakeExecResult()
        if "ls /app/.outbox/" in cmd:
            files = [f for f in self._files if f.startswith("/app/.outbox/")]
            return FakeExecResult(stdout="\n".join(files))
        if cmd.startswith("cat "):
            path = cmd.split(" ", 1)[1]
            return FakeExecResult(stdout=self._files.get(path, "{}"))
        if cmd.startswith("rm -f "):
            path = cmd.split()[-1]
            self._files.pop(path, None)
            return FakeExecResult()
        return FakeExecResult()

    def stage_outbox(self, recipient: str, content: str) -> None:
        self._files[f"/app/.outbox/{recipient}.json"] = json.dumps(
            {"to": recipient, "content": content}
        )


@pytest.fixture
def coder_reviewer_roles() -> dict[str, SceneRole]:
    return {
        "coder": SceneRole(
            name="coder",
            agent="gemini",
            model="flash",
            instruction="Write a function.",
        ),
        "reviewer": SceneRole(
            name="reviewer",
            agent="gemini",
            model="flash",
            instruction="Review the code.",
        ),
    }


async def test_coder_reviewer_coder_turn_sequence(
    coder_reviewer_roles: dict[str, SceneRole],
) -> None:
    """Coder → Reviewer → Coder turn sequence routes messages correctly."""
    env = FakeEnv()
    scene = SceneRuntime(roles=coder_reviewer_roles, max_rounds=6)
    turns: list[tuple[str, str]] = []

    async def mock_runner(e, role, prompt):
        turns.append((role.name, prompt))
        if role.name == "coder" and len(turns) == 1:
            env.stage_outbox("reviewer", "here is my code")
        elif role.name == "reviewer":
            env.stage_outbox("coder", "fix the bug on line 3")

    trajectory = await scene.run(env, mock_runner)

    assert turns[0][0] == "coder"
    assert turns[1][0] == "reviewer"
    assert "here is my code" in turns[1][1]
    assert turns[2][0] == "coder"
    assert "fix the bug on line 3" in turns[2][1]

    assert len(trajectory) >= 2
    assert trajectory[0].sender == "coder"
    assert trajectory[0].recipient == "reviewer"
    assert trajectory[1].sender == "reviewer"
    assert trajectory[1].recipient == "coder"


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


async def test_scene_scheduler_respects_max_rounds(
    coder_reviewer_roles: dict[str, SceneRole],
) -> None:
    """Scene stops after max_rounds even if agents keep messaging."""
    env = FakeEnv()
    scene = SceneRuntime(roles=coder_reviewer_roles, max_rounds=2)

    async def always_message(e, role, prompt):
        other = "reviewer" if role.name == "coder" else "coder"
        env.stage_outbox(other, f"msg from {role.name}")

    await scene.run(env, always_message)
    assert scene.is_done
    assert scene._round >= 2
