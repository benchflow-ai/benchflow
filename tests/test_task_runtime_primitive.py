"""Tests for the reusable BenchFlow task runtime primitive."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from benchflow.rollout import Rollout, TaskRuntime, TaskRuntimeConfig
from benchflow.skill_policy import SKILL_MODE_SELF_GEN
from benchflow.task import VerifierResult

TASK_PATH = Path(__file__).parent / "examples" / "hello-world-task"


@dataclass
class _ExecResult:
    return_code: int = 0
    stdout: str = ""
    stderr: str = ""


class _FakeSandbox:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.exec_calls: list[dict[str, Any]] = []
        self.uploads: list[tuple[str, str]] = []
        self.files: dict[str, str] = {}
        self.sandbox_id = "synthetic-sandbox"

    async def start(self, *args: Any, **kwargs: Any) -> None:
        self.started += 1

    async def stop(self, *args: Any, **kwargs: Any) -> None:
        self.stopped += 1

    async def exec(self, cmd: str, **kwargs: Any) -> _ExecResult:
        self.exec_calls.append({"cmd": cmd, **kwargs})
        if cmd == "pwd":
            return _ExecResult(stdout="/app\n")
        if cmd.startswith("bash -lc "):
            tokens = shlex.split(cmd)
            script = tokens[2] if len(tokens) > 2 else ""
            if "hello.txt" in script and "Hello, world!" in script:
                self.files["/app/hello.txt"] = "Hello, world!"
            if "exit 7" in script:
                return _ExecResult(return_code=7, stderr="bad command\n")
        return _ExecResult()

    async def upload_file(self, src: str | Path, dst: str) -> None:
        self.uploads.append((str(src), dst))
        self.files[dst] = Path(src).read_text()

    async def upload_dir(
        self, src: str | Path, dst: str, service: str = "main"
    ) -> None:
        self.uploads.append((str(src), dst))


class _FakeVerifier:
    def __init__(self, sandbox: _FakeSandbox, rollout_paths: Any) -> None:
        self.sandbox = sandbox
        self.rollout_paths = rollout_paths

    async def verify(self) -> VerifierResult:
        reward = (
            1.0 if self.sandbox.files.get("/app/hello.txt") == "Hello, world!" else 0.0
        )
        self.rollout_paths.reward_text_path.write_text(str(reward))
        return VerifierResult(rewards={"reward": reward})


class _FakePlanes:
    def __init__(self) -> None:
        self.sandbox = _FakeSandbox()
        self.created: list[dict[str, Any]] = []
        self.locked_paths: list[str] = []

    def install_docker_compat(self) -> None:
        return None

    def extract_usage(self, runtime: Any) -> dict[str, Any]:
        return {"usage_source": "unavailable"}

    def agent_launch(self, agent: str, *, disallow_web_tools: bool) -> str:
        return agent

    def agent_config(self, agent: str) -> Any:
        return None

    def resolve_agent_env(
        self,
        agent: str,
        model: str | None,
        agent_env: dict[str, str] | None,
    ) -> dict[str, str]:
        return {}

    def resolve_locked_paths(
        self, sandbox_user: str | None, locked_paths: list[str] | None
    ) -> list[str]:
        if locked_paths is not None:
            return locked_paths
        if sandbox_user is None:
            return []
        return ["/oracle", "/solution", "/verifier", "/tests", "/testbed_verify"]

    def stage_dockerfile_deps(self, task_path: Path, context_root: Path) -> None:
        return None

    def override_dockerfile_base_image(self, task_path: Path, base_image: str) -> int:
        return 0

    def inject_skills_into_dockerfile(
        self, task_path: Path, skills_dir: Path, *, sandbox_dir: str = "/skills"
    ) -> None:
        return None

    def create_environment(
        self,
        environment: str,
        task: Any,
        task_path: Path,
        rollout_name: str | None,
        rollout_paths: Any,
        *,
        preserve_agent_network: bool,
        environment_manifest: Any,
    ) -> _FakeSandbox:
        self.created.append(
            {
                "environment": environment,
                "task_path": task_path,
                "rollout_name": rollout_name,
                "preserve_agent_network": preserve_agent_network,
            }
        )
        return self.sandbox

    def manifest_environment(self, manifest: Any, *, sandbox: Any) -> Any:
        raise AssertionError("environment manifests are not used in this test")

    async def setup_sandbox_user(
        self,
        env: Any,
        sandbox_user: str,
        *,
        workspace: str,
        timeout_sec: int = 120,
    ) -> str:
        return workspace

    async def snapshot_build_config(self, env: Any, *, workspace: str) -> None:
        return None

    async def seed_verifier_workspace(
        self, env: Any, *, workspace: str, sandbox_user: str | None
    ) -> None:
        return None

    async def deploy_skills(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def lockdown_paths(self, env: Any, locked_paths: list[str]) -> None:
        self.locked_paths = list(locked_paths)

    async def install_agent(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TaskRuntime must not install an ACP agent")

    async def write_credential_files(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def upload_subscription_auth(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def apply_web_tool_policy(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def link_skill_paths(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def ensure_litellm_runtime(self, *args: Any, **kwargs: Any) -> Any:
        return {}, None

    async def stop_provider_runtime(self, runtime: Any) -> None:
        return None

    async def connect_acp(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TaskRuntime must not connect ACP")

    async def execute_prompts(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TaskRuntime must not execute ACP prompts")

    async def connect_session_factory(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TaskRuntime must not connect session factories")

    async def execute_prompts_session_factory(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("TaskRuntime must not execute session-factory prompts")

    async def harden_before_verify(self, *args: Any, **kwargs: Any) -> None:
        return None

    def verifier(self, *args: Any, **kwargs: Any) -> _FakeVerifier:
        return _FakeVerifier(kwargs["sandbox"], kwargs["rollout_paths"])

    async def clear_verifier_output_dir(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def ensure_legacy_app_dir(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def cleanup_verifier_python_hooks(self, *args: Any, **kwargs: Any) -> None:
        return None


@pytest.mark.asyncio
async def test_task_runtime_bash_verify_writes_rollout_artifacts(
    tmp_path: Path,
) -> None:
    """Guards PR #902: training loops can exec, verify, and keep artifacts."""

    planes = _FakePlanes()
    config = TaskRuntimeConfig(
        task_path=TASK_PATH,
        environment="synthetic",
        jobs_dir=tmp_path / "jobs",
        rollout_name="train-loop",
        planes=planes,
    )

    async with TaskRuntime(config) as runtime:
        tool = await runtime.bash("printf 'Hello, world!' > hello.txt", timeout_sec=5)
        failed_tool = await runtime.bash("printf bad >&2; exit 7", timeout_sec=5)
        result = await runtime.verify()

    assert tool.return_code == 0
    assert failed_tool.return_code == 7
    assert result.reward == 1.0
    assert result.rewards == {"reward": 1.0}
    assert planes.sandbox.started == 1
    assert planes.sandbox.stopped == 1
    assert planes.locked_paths == [
        "/oracle",
        "/solution",
        "/verifier",
        "/tests",
        "/testbed_verify",
    ]

    bash_call = next(
        call for call in planes.sandbox.exec_calls if "bash -lc" in call["cmd"]
    )
    assert bash_call["user"] == "agent"
    assert "cd /app && printf" in shlex.split(bash_call["cmd"])[2]

    rollout_dir = result.rollout_dir
    assert (rollout_dir / "config.json").is_file()
    assert (rollout_dir / "result.json").is_file()
    assert (rollout_dir / "rewards.jsonl").is_file()
    assert (rollout_dir / "trajectory" / "acp_trajectory.jsonl").is_file()
    assert planes.sandbox.files["/logs/agent/acp_trajectory.jsonl"]
    trajectory_events = [
        json.loads(line)
        for line in (rollout_dir / "trajectory" / "acp_trajectory.jsonl")
        .read_text()
        .splitlines()
    ]
    assert [
        (event["type"], event["tool_name"], event["return_code"], event["status"])
        for event in trajectory_events
    ] == [
        ("tool_call", "bash", 0, "completed"),
        ("tool_call", "bash", 7, "failed"),
    ]

    result_json = json.loads((rollout_dir / "result.json").read_text())
    assert result_json["agent"] == "task-runtime"
    assert result_json["n_tool_calls"] == 2
    assert result_json["rewards"] == {"reward": 1.0}
    assert result_json["sandbox_id"] == "synthetic-sandbox"


@pytest.mark.asyncio
async def test_task_runtime_returns_zero_reward_from_verifier(tmp_path: Path) -> None:
    """Guards PR #902: scalar rewards come from the verifier, not bash exit."""

    planes = _FakePlanes()
    runtime = await TaskRuntime.create(
        TaskRuntimeConfig(
            task_path=TASK_PATH,
            environment="synthetic",
            jobs_dir=tmp_path / "jobs",
            rollout_name="unanswered",
            planes=planes,
        )
    )
    try:
        result = await runtime.verify()
    finally:
        await runtime.close()

    assert result.reward == 0.0
    assert result.rewards == {"reward": 0.0}


def test_task_runtime_rejects_self_gen_skill_mode() -> None:
    """Guards PR #902: this primitive does not run skill-generation flows."""

    with pytest.raises(ValueError, match="no-skill and with-skill"):
        TaskRuntimeConfig(task_path=TASK_PATH, skill_mode=SKILL_MODE_SELF_GEN)


def test_external_tool_call_rejects_reserved_event_fields(tmp_path: Path) -> None:
    """Guards PR #902: external tool events cannot override canonical fields."""

    rollout = Rollout(
        TaskRuntimeConfig(
            task_path=TASK_PATH,
            environment="synthetic",
            jobs_dir=tmp_path / "jobs",
            planes=_FakePlanes(),
        ).to_rollout_config()
    )

    with pytest.raises(ValueError, match="reserved fields: tool_name, type"):
        rollout.record_external_tool_call(
            tool_name="bash",
            event={"type": "agent_message", "tool_name": "python"},
        )
