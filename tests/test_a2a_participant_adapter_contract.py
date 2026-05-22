"""Contract tests and pending runtime tests for AgentBeats A2A participants."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from benchflow.agents.a2a import (
    A2AArtifactRef,
    A2AParticipantRequest,
    A2AParticipantResult,
    A2ATaskHandle,
    A2ATrajectoryEvent,
)
from benchflow.rollout import Role, Rollout, RolloutConfig, Scene, Turn
from benchflow.scenes import compile_scenes_to_steps
from benchflow.skill_policy import resolve_task_skill_policy


@dataclass
class FakeExecResult:
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0


class FakeEnv:
    def __init__(self) -> None:
        self.exec_log: list[str] = []
        self.uploads: dict[str, str] = {}

    async def exec(self, cmd: str, **kwargs: Any) -> FakeExecResult:
        self.exec_log.append(cmd)
        return FakeExecResult()

    async def upload_file(self, src: str | Path, dst: str) -> None:
        self.uploads[dst] = Path(src).read_text()


@dataclass
class FakeRolloutPaths:
    verifier_dir: Path


class FakeA2AAdapter:
    def __init__(
        self,
        result: A2AParticipantResult | None = None,
        *,
        block_wait: bool = False,
    ) -> None:
        self.result = result or A2AParticipantResult(status="completed")
        self.block_wait = block_wait
        self.requests: list[A2AParticipantRequest] = []
        self.waited: list[A2ATaskHandle] = []
        self.cancelled: list[tuple[A2ATaskHandle, str]] = []

    async def start(self, request: A2AParticipantRequest) -> A2ATaskHandle:
        self.requests.append(request)
        return A2ATaskHandle(
            task_id=f"a2a-task-{len(self.requests)}",
            endpoint_url=request.endpoint_url,
            role_name=request.role_name,
        )

    async def wait(self, handle: A2ATaskHandle) -> A2AParticipantResult:
        self.waited.append(handle)
        if self.block_wait:
            await asyncio.sleep(60)
        return self.result

    async def cancel(self, handle: A2ATaskHandle, reason: str) -> None:
        self.cancelled.append((handle, reason))


def _a2a_role(**overrides: Any) -> Role:
    data: dict[str, Any] = {
        "name": "agent",
        "agent": "registered-purple-agent",
        "transport": "a2a",
        "endpoint_url": "http://purple.example/a2a",
    }
    data.update(overrides)
    return Role(**data)


def _make_rollout(
    tmp_path: Path,
    adapter: FakeA2AAdapter,
    scene: Scene,
) -> Rollout:
    task_path = tmp_path / "tasks" / "citation-check"
    task_path.mkdir(parents=True)
    (task_path / "instruction.md").write_text("Solve the task.")
    rollout_dir = tmp_path / "jobs" / "run" / "trial"
    (rollout_dir / "artifacts").mkdir(parents=True)
    (rollout_dir / "trajectory").mkdir(parents=True)

    trial = Rollout(
        RolloutConfig(
            task_path=task_path,
            scenes=[scene],
            a2a_adapter=adapter,
        )
    )
    trial._env = FakeEnv()
    trial._rollout_dir = rollout_dir
    trial._rollout_paths = FakeRolloutPaths(verifier_dir=rollout_dir / "verifier")
    trial._rollout_name = "trial"
    trial._started_at = datetime.now()
    trial._agent_cwd = "/app"
    trial._timeout = 5
    trial._task = object()
    trial._resolved_prompts = ["Solve the task."]
    return trial


def test_a2a_contract_result_done_statuses() -> None:
    assert A2AParticipantResult(status="running").done is False
    for status in ("completed", "failed", "cancelled", "timeout"):
        assert A2AParticipantResult(status=status).done is True


def test_a2a_contract_carries_endpoint_visible_prompt_and_redactable_refs() -> None:
    request = A2AParticipantRequest(
        endpoint_url="https://purple.example/a2a",
        role_name="agent",
        prompt="Solve the task visible in /app.",
        skills_dir="/skills",
        timeout_sec=300,
        metadata={"task_id": "citation-check", "condition": "with_skills"},
    )
    handle = A2ATaskHandle(
        task_id="a2a-task-1",
        endpoint_url=request.endpoint_url,
        role_name=request.role_name,
    )
    result = A2AParticipantResult(
        status="completed",
        trajectory=(
            A2ATrajectoryEvent(kind="task_update", payload={"state": "working"}),
            A2ATrajectoryEvent(kind="final_response", payload={"ok": True}),
        ),
        artifacts=(
            A2AArtifactRef(
                name="answer",
                uri="benchflow-private://rollout/artifacts/answer.json",
                media_type="application/json",
                digest="sha256:abc123",
            ),
        ),
        final_response={"done": True},
    )

    assert handle.endpoint_url == "https://purple.example/a2a"
    assert request.skills_dir == "/skills"
    assert result.status == "completed"
    assert result.artifacts[0].uri.startswith("benchflow-private://")


async def test_a2a_endpoint_role_skips_acp_install_and_invokes_endpoint(
    tmp_path: Path,
) -> None:
    adapter = FakeA2AAdapter(
        A2AParticipantResult(
            status="completed",
            trajectory=(
                A2ATrajectoryEvent(
                    kind="task_update",
                    payload={"state": "completed"},
                ),
            ),
            final_response={"message": "done"},
        )
    )
    scene = Scene(
        roles=[_a2a_role()],
        turns=[Turn("agent", "Solve visible workspace task.")],
    )
    trial = _make_rollout(tmp_path, adapter, scene)

    async def fail_acp(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("A2A roles must not use ACP execution")

    trial.connect_as = fail_acp  # type: ignore[method-assign]
    trial.execute = fail_acp  # type: ignore[method-assign]

    await trial._run_steps(compile_scenes_to_steps([scene]))

    assert len(adapter.requests) == 1
    request = adapter.requests[0]
    assert request.endpoint_url == "http://purple.example/a2a"
    assert request.prompt == "Solve visible workspace task."
    # Default config runs skill-free: the participant gets no skills path.
    assert request.skills_dir is None
    assert adapter.waited[0].role_name == "agent"


async def test_a2a_request_points_at_task_skill_policy_sandbox_dir(
    tmp_path: Path,
) -> None:
    """When task skills are deployed, the A2A request carries their sandbox path."""
    adapter = FakeA2AAdapter()
    scene = Scene(
        roles=[_a2a_role()],
        turns=[Turn("agent", "Solve visible workspace task.")],
    )
    trial = _make_rollout(tmp_path, adapter, scene)
    skills_dir = trial._config.task_path / "environment" / "skills" / "demo"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("# demo skill")
    trial._task_skill_policy = resolve_task_skill_policy(
        task_path=trial._config.task_path,
        skill_mode="with-skill",
        runtime_skills_dir=None,
        declared_sandbox_skills_dir=None,
    )

    await trial._run_steps(compile_scenes_to_steps([scene]))

    assert adapter.requests[0].skills_dir == "/skills"


async def test_a2a_timeout_cancels_task_and_records_timeout_result(
    tmp_path: Path,
) -> None:
    adapter = FakeA2AAdapter(block_wait=True)
    scene = Scene(
        roles=[_a2a_role(timeout_sec=0)],
        turns=[Turn("agent", "Solve visible workspace task.")],
    )
    trial = _make_rollout(tmp_path, adapter, scene)

    await trial._run_steps(compile_scenes_to_steps([scene]))

    assert adapter.cancelled
    assert adapter.cancelled[0][1] == "timeout"
    assert trial._error == "a2a_timeout"
    assert any(
        event["payload"].get("error_type") == "a2a_timeout"
        for event in trial.trajectory
    )


async def test_a2a_artifacts_are_persisted_before_verifier_handoff(
    tmp_path: Path,
) -> None:
    adapter = FakeA2AAdapter(
        A2AParticipantResult(
            status="completed",
            artifacts=(
                A2AArtifactRef(
                    name="answer",
                    uri="a2a://task/artifacts/0",
                    media_type="application/json",
                    digest="sha256:abc",
                ),
            ),
        )
    )
    scene = Scene(roles=[_a2a_role()], turns=[Turn("agent", "Solve.")])
    trial = _make_rollout(tmp_path, adapter, scene)

    await trial._run_steps(compile_scenes_to_steps([scene]))

    artifact_path = trial._rollout_dir / "artifacts" / "a2a_artifacts.json"
    artifacts = json.loads(artifact_path.read_text())
    assert artifacts == [
        {
            "name": "answer",
            "uri": "a2a://task/artifacts/0",
            "media_type": "application/json",
            "digest": "sha256:abc",
            "protocol": "a2a",
            "task_id": "a2a-task-1",
            "scene": "default",
            "role": "agent",
        }
    ]


async def test_a2a_file_artifacts_are_materialized_under_workspace(
    tmp_path: Path,
) -> None:
    adapter = FakeA2AAdapter(
        A2AParticipantResult(
            status="completed",
            final_response={
                "files": [
                    {
                        "path": "reports/answer.txt",
                        "content": "materialized answer",
                        "media_type": "text/plain",
                    }
                ]
            },
        )
    )
    scene = Scene(roles=[_a2a_role()], turns=[Turn("agent", "Solve.")])
    trial = _make_rollout(tmp_path, adapter, scene)

    await trial._run_steps(compile_scenes_to_steps([scene]))

    assert trial._env.uploads["/app/reports/answer.txt"] == "materialized answer"
    artifact_path = trial._rollout_dir / "artifacts" / "a2a_artifacts.json"
    artifacts = json.loads(artifact_path.read_text())
    assert artifacts[0]["uri"] == "sandbox:///app/reports/answer.txt"
    assert artifacts[0]["materialized"] is True


async def test_a2a_successful_done_signal_runs_existing_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = FakeA2AAdapter(
        A2AParticipantResult(
            status="completed",
            trajectory=(A2ATrajectoryEvent(kind="task_update", payload={}),),
        )
    )
    scene = Scene(roles=[_a2a_role()], turns=[Turn("agent", "Solve.")])
    trial = _make_rollout(tmp_path, adapter, scene)
    calls: dict[str, Any] = {}

    async def fake_verify_rollout(*args: Any, **kwargs: Any):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return {"reward": 1.0}, None, None

    monkeypatch.setattr("benchflow.rollout._verify_rollout", fake_verify_rollout)

    await trial._run_steps(compile_scenes_to_steps([scene]))
    rewards = await trial.verify()

    assert rewards == {"reward": 1.0}
    assert calls["kwargs"]["workspace"] == "/app"
    assert "/logs/agent/a2a_trajectory.jsonl" in trial._env.uploads


async def test_a2a_updates_are_written_to_a2a_trajectory_jsonl(
    tmp_path: Path,
) -> None:
    adapter = FakeA2AAdapter(
        A2AParticipantResult(
            status="completed",
            trajectory=(
                A2ATrajectoryEvent(
                    kind="task_update",
                    payload={"state": "working"},
                ),
            ),
        )
    )
    scene = Scene(roles=[_a2a_role()], turns=[Turn("agent", "Solve.")])
    trial = _make_rollout(tmp_path, adapter, scene)

    await trial._run_steps(compile_scenes_to_steps([scene]))
    trial._build_result()

    trajectory_path = trial._rollout_dir / "trajectory" / "a2a_trajectory.jsonl"
    assert trajectory_path.exists()
    lines = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    assert all(line["protocol"] == "a2a" for line in lines)
    assert not (trial._rollout_dir / "trajectory" / "acp_trajectory.jsonl").exists()
