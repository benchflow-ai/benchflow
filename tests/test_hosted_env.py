from __future__ import annotations

import json
from types import SimpleNamespace

from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.hosted_env import (
    HostedEnvRef,
    HostedEnvRunConfig,
    HostedEnvRunResult,
    normalize_verifiers_model,
    parse_sampling_args,
    parse_source_env_args,
    run_hosted_env,
)


def test_hosted_env_ref_keeps_prime_identity():
    ref = HostedEnvRef.parse(
        "primeintellect:primeintellect/general-agent@0.1.1"
    )

    assert ref.provider == "primeintellect"
    assert ref.env_id == "primeintellect/general-agent"
    assert ref.versioned_env_id == "primeintellect/general-agent@0.1.1"
    assert ref.env_uid == "primeintellect:primeintellect/general-agent@0.1.1"
    assert (
        ref.hub_url
        == "https://app.primeintellect.ai/dashboard/environments/primeintellect/general-agent"
    )
    assert ref.python_package == "general_agent"
    assert ref.verifiers_env_id == "general-agent"


def test_source_env_args_parse_json_scalars():
    assert parse_source_env_args(["task=calendar_scheduling_t0", "n=1"]) == {
        "task": "calendar_scheduling_t0",
        "n": 1,
    }
    assert parse_sampling_args(["reasoning_effort=minimal"]) == {
        "reasoning_effort": "minimal"
    }


def test_normalize_verifiers_model_for_prime_registry():
    assert (
        normalize_verifiers_model("gemini-3.1-flash-lite-preview")
        == "google/gemini-3.1-flash-lite-preview"
    )
    assert normalize_verifiers_model("openai/gpt-5-mini") == "openai/gpt-5-mini"


def test_run_hosted_env_uses_controlled_verifiers_venv(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_which(binary: str) -> str:
        return f"/bin/{binary}"

    def fake_run(cmd, **kwargs):
        calls.append([str(c) for c in cmd])
        if str(cmd[0]).endswith("vf-eval"):
            return SimpleNamespace(
                returncode=0,
                stdout="reward: avg - 1.000\ntotal_tool_calls: avg - 2.000\n",
                stderr="",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchflow.hosted_env.shutil.which", fake_which)
    monkeypatch.setattr("benchflow.hosted_env.subprocess.run", fake_run)

    result = run_hosted_env(
        HostedEnvRunConfig(
            source_env=HostedEnvRef.parse(
                "primeintellect/general-agent", version="0.1.1"
            ),
            model="gemini-3.1-flash-lite-preview",
            env_args={"task": "calendar_scheduling_t0"},
            agent="gemini",
            jobs_dir=tmp_path,
        )
    )

    assert result.returncode == 0
    assert result.reward == 1.0
    assert result.total_tool_calls == 2
    assert calls[0][:3] == ["/bin/uv", "venv", "--python"]
    assert calls[1][:4] == ["/bin/uv", "pip", "install", "--python"]
    assert "general_agent==0.1.1" in calls[1]
    assert calls[2][0].endswith("/bin/vf-eval")
    assert "general-agent" in calls[2]
    assert "google/gemini-3.1-flash-lite-preview" in calls[2]

    payload = json.loads((result.run_dir / "result.json").read_text())
    assert payload["env_uid"] == "primeintellect:primeintellect/general-agent@0.1.1"
    assert payload["rewards"] == {"reward": 1.0}


def test_run_hosted_env_classifies_verifiers_model_errors(tmp_path, monkeypatch):
    def fake_which(binary: str) -> str:
        return f"/bin/{binary}"

    def fake_run(cmd, **kwargs):
        if str(cmd[0]).endswith("vf-eval"):
            return SimpleNamespace(
                returncode=0,
                stdout="reward: avg - 0.000\n",
                stderr="ERROR - Aborted rollout due to ModelError() -> NotFoundError('model_not_found')\n",
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchflow.hosted_env.shutil.which", fake_which)
    monkeypatch.setattr("benchflow.hosted_env.subprocess.run", fake_run)

    result = run_hosted_env(
        HostedEnvRunConfig(
            source_env=HostedEnvRef.parse(
                "primeintellect/general-agent", version="0.1.1"
            ),
            model="gemini-3.1-flash-lite-preview",
            jobs_dir=tmp_path,
        )
    )

    assert result.returncode == 0
    assert result.reward == 0.0
    assert result.error == "ModelError() -> NotFoundError('model_not_found')"


def test_eval_create_source_env_routes_to_hosted_runner(tmp_path, monkeypatch):
    seen: dict[str, object] = {}

    def fake_run_hosted_env(config: HostedEnvRunConfig) -> HostedEnvRunResult:
        seen["config"] = config
        return HostedEnvRunResult(
            source_env=config.source_env,
            run_dir=tmp_path / "run",
            command=["vf-eval"],
            returncode=0,
            stdout="",
            stderr="",
            model=config.model,
            normalized_model=normalize_verifiers_model(config.model),
            reward=1.0,
            total_tool_calls=2,
        )

    monkeypatch.setattr("benchflow.hosted_env.run_hosted_env", fake_run_hosted_env)

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "create",
            "--source-env",
            "primeintellect/general-agent",
            "--source-env-version",
            "0.1.1",
            "--source-env-arg",
            "task=calendar_scheduling_t0",
            "--agent",
            "gemini",
            "--model",
            "gemini-3.1-flash-lite-preview",
            "--sandbox",
            "daytona",
            "--jobs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    config = seen["config"]
    assert isinstance(config, HostedEnvRunConfig)
    assert config.source_env.env_uid == "primeintellect:primeintellect/general-agent@0.1.1"
    assert config.env_args == {"task": "calendar_scheduling_t0"}
    assert config.agent == "gemini"
    assert config.model == "gemini-3.1-flash-lite-preview"
    assert "not used by source-env runs" in result.output
