"""Exercise the real SDK config and hook with fake sandbox/provider/hardware."""

import asyncio
import json
import urllib.request
from types import SimpleNamespace

import pytest

import benchflow
from benchflow.models import RolloutResult
from benchflow.robotics import runner
from benchflow.robotics.tasks import build_tasks
from benchflow.sandbox.docker import DockerSandbox


@pytest.mark.parametrize(
    "agent,model", [("codex", "gpt-6-astra"), ("claude", "claude-fable-5-1")]
)
@pytest.mark.parametrize("probe", [True, False])
def test_sdk_hook_and_metrics(tmp_path, monkeypatch, agent, model, probe):
    frames = tmp_path / "frames"
    frames.mkdir()
    images = []
    for camera in ("wrist", "side"):
        path = frames / f"001_after_{camera}.jpg"
        path.write_bytes(b"\xff\xd8image\xff\xd9")
        images.append(str(path))
    episode = tmp_path / "episode.jsonl"
    episode.touch()
    setup = tmp_path / "setup.json"
    setup.write_text(
        json.dumps(
            {
                "setup_id": "fixture",
                "robot_id": "fixture-metal",
                "arm_type": "metal",
                "socket_path": "unused",
                "frames_root": str(frames),
                "episode_log": str(episode),
                "cameras": {"wrist": "unused", "side": "unused"},
                "public_facts": {"table_z_m": 0.01},
            }
        )
    )
    calls, uploaded = [], {}

    def transport(command, args):
        calls.append(command)
        return {"ok": True, "armed": True, "arm": "metal", "frames": images}

    async def upload(source, destination):
        uploaded[destination] = json.loads(source.read_text())

    async def sandbox_exec(command, **kwargs):
        return SimpleNamespace(exit_code=0)

    monkeypatch.setattr(DockerSandbox, "preflight", lambda: None)
    monkeypatch.setattr(runner, "HarnessTransport", lambda _: transport)
    monkeypatch.setattr(runner, "provider_environment", lambda _: {})
    monkeypatch.setattr(
        runner,
        "RecordingSidecar",
        lambda *args: SimpleNamespace(
            start=lambda: None,
            healthy=lambda: True,
            stop=lambda: {
                "complete": True,
                "exports": {camera: {"ok": True} for camera in ("wrist", "side")},
            },
        ),
    )

    async def sdk_run(config):
        assert config.model == model and config.reasoning_effort == "max"
        assert config.environment == "docker" and config.skip_verify
        assert config.usage_tracking.mode == "required"
        assert config.context_root is None and config.skills_dir is None
        assert config.prompts == ([runner.PROBE_PROMPT] if probe else None)
        for hook in config.pre_agent_hooks:
            await hook(SimpleNamespace(upload_file=upload, exec=sandbox_exec))
        assert set(uploaded) == {"/app/robot-connection.json", "/app/setup.json"}
        connection = uploaded["/app/robot-connection.json"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

        def request(command, args):
            with opener.open(
                urllib.request.Request(
                    connection["url"] + "/command",
                    json.dumps(
                        {"request_id": command, "command": command, "args": args}
                    ).encode(),
                    {"Authorization": "Bearer " + connection["token"]},
                ),
                timeout=3,
            ) as response:
                return json.load(response)

        assert request("observe", [])["ok"]
        assert request("tip", [0.2, 0, 0.2, -75])["ok"] is (not probe)
        assert request("finish", [])["ok"]
        return RolloutResult(
            task_name="fixture",
            agent=agent,
            model=model,
            n_input_tokens=101,
            n_output_tokens=11,
            cost_usd=None,
            usage_source="provider_response",
        )

    monkeypatch.setattr(benchflow, "run", sdk_run)
    output = asyncio.run(
        runner.run_trial(
            setup_path=setup,
            task_path=build_tasks(tmp_path / "tasks")[0],
            output_root=tmp_path / "trials",
            agent=agent,
            model=model,
            reasoning_effort="max",
            reset_id="fixture",
            operator="test",
            probe=probe,
            allow_motion=not probe,
            bind="0.0.0.0",
            advertised_host="127.0.0.1",
            lock_root=tmp_path / "locks",
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["kind"] == ("agent_probe" if probe else "physical_trial")
    assert manifest["status"] == ("probe_completed" if probe else "awaiting_assessment")
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["n_input_tokens"] == 101 and metrics["cost_usd"] is None
    assert ("tip" in calls) is (not probe)
    assert not (output / ".connection.json").exists()
    assert not (output / "reward.txt").exists()
