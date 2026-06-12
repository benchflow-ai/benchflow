"""Hosted-env runs must produce the same artifact contract as native rollouts.

Guards PR #419: hosted-env evaluations previously bypassed the rollout
reward and trajectory engine and wrote a custom ``result.json``, so they
could not be compared to native rollouts without a schema rewrite.

Each test pins one slice of the contract that downstream tools (dashboards,
release checks, ``rewards.jsonl`` consumers, trajectory readers) depend on.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.hosted_env import (
    HostedEnvRef,
    HostedEnvRunConfig,
    run_hosted_env,
)

VF_EVAL_STDOUT = "reward: avg - 1.000\ntotal_tool_calls: avg - 2.000\n"


def _patch_run(monkeypatch: pytest.MonkeyPatch, *, results_jsonl: str = "") -> Path:
    """Patch shutil.which + subprocess.run; capture the output_dir vf-eval sees."""
    captured: dict[str, Path] = {}

    def fake_which(binary: str) -> str:
        return f"/bin/{binary}"

    def fake_run(cmd, **kwargs):
        if str(cmd[0]).endswith("vf-eval"):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            captured["output_dir"] = output_dir
            if results_jsonl:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "results.jsonl").write_text(results_jsonl)
            return SimpleNamespace(returncode=0, stdout=VF_EVAL_STDOUT, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchflow.hosted_env.shutil.which", fake_which)
    monkeypatch.setattr("benchflow.hosted_env.subprocess.run", fake_run)
    return Path(captured.get("output_dir", Path()))


def _patch_run_custom(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stdout: str,
    stderr: str = "",
    returncode: int = 0,
    results_jsonl: str = "",
) -> None:
    """Patch shutil.which + subprocess.run with caller-controlled vf-eval output."""

    def fake_which(binary: str) -> str:
        return f"/bin/{binary}"

    def fake_run(cmd, **kwargs):
        if str(cmd[0]).endswith("vf-eval"):
            output_dir = Path(cmd[cmd.index("--output-dir") + 1])
            if results_jsonl:
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "results.jsonl").write_text(results_jsonl)
            return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("benchflow.hosted_env.shutil.which", fake_which)
    monkeypatch.setattr("benchflow.hosted_env.subprocess.run", fake_run)


def _run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, results_jsonl: str = ""):
    _patch_run(monkeypatch, results_jsonl=results_jsonl)
    return _invoke(tmp_path)


def _invoke(tmp_path: Path):
    return run_hosted_env(
        HostedEnvRunConfig(
            source_env=HostedEnvRef.parse(
                "primeintellect/general-agent", version="0.1.1"
            ),
            model="gemini-3.1-flash-lite-preview",
            env_args={"task": "calendar_scheduling_t0"},
            agent="gemini",
            jobs_dir=tmp_path,
            num_examples=2,
        )
    )


def _classify(result) -> str:
    from benchflow._utils.scoring import classify_score_outcome

    payload = json.loads((result.run_dir / "result.json").read_text())
    return classify_score_outcome(payload)


def test_hosted_env_writes_contract_result_json(tmp_path, monkeypatch):
    """result.json carries the rollout-contract field set, not a custom shape."""
    result = _run(tmp_path, monkeypatch)
    payload = json.loads((result.run_dir / "result.json").read_text())

    # Required rollout-contract keys consumed by EvaluationResult /
    # summary.json / dashboards. Missing any of these would silently
    # downgrade hosted evidence to second-class status — the original bug.
    for key in (
        "task_name",
        "rollout_name",
        "rewards",
        "agent",
        "agent_name",
        "model",
        "n_tool_calls",
        "n_prompts",
        "agent_result",
        "final_metrics",
        "trajectory_summary",
        "error",
        "verifier_error",
        "partial_trajectory",
        "trajectory_source",
        "started_at",
        "finished_at",
        "timing",
        "source",
    ):
        assert key in payload, f"missing contract key: {key}"

    agent_result = payload["agent_result"]
    for key in (
        "n_tool_calls",
        "n_prompts",
        "n_input_tokens",
        "n_output_tokens",
        "total_tokens",
        "cost_usd",
        "usage_source",
        "price_source",
    ):
        assert key in agent_result, f"missing agent_result key: {key}"
    assert set(payload["final_metrics"]) == {
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_cached_tokens",
        "total_cost_usd",
    }
    assert payload["trajectory_summary"]["steps"] >= 0
    assert payload["trajectory_summary"]["tool_call_steps"] >= 0


def test_hosted_env_writes_rewards_jsonl(tmp_path, monkeypatch):
    """rewards.jsonl is required by the dense-reward / metrics pipeline."""
    result = _run(tmp_path, monkeypatch)

    rewards_path = result.run_dir / "rewards.jsonl"
    assert rewards_path.exists(), "rewards.jsonl must be written"
    events = [
        json.loads(line) for line in rewards_path.read_text().splitlines() if line
    ]
    # At least one terminal verifier reward event.
    terminal = [e for e in events if e["type"] == "terminal" and e["tag"] == "reward"]
    assert len(terminal) == 1
    assert terminal[0]["value"] == 1.0
    assert terminal[0]["source"] == "verifier"


def test_hosted_env_writes_trajectory_dir(tmp_path, monkeypatch):
    """trajectory/acp_trajectory.jsonl must exist (empty is fine, missing is not)."""
    results_jsonl = (
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "schedule a meeting"}],
                "completion": [{"role": "assistant", "content": "10am works"}],
                "reward": 1.0,
            }
        )
        + "\n"
    )
    result = _run(tmp_path, monkeypatch, results_jsonl=results_jsonl)

    traj_path = result.run_dir / "trajectory" / "acp_trajectory.jsonl"
    assert traj_path.exists()
    events = [json.loads(line) for line in traj_path.read_text().splitlines() if line]
    kinds = [e["type"] for e in events]
    assert "user_message" in kinds
    assert "agent_message" in kinds
    assert "reward" in kinds


def test_hosted_env_marks_trajectory_source_as_imported(tmp_path, monkeypatch):
    """trajectory_source must signal the lineage so consumers can downweight."""
    results_jsonl = (
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "hi"}],
                "completion": [{"role": "assistant", "content": "hello"}],
                "reward": 1.0,
            }
        )
        + "\n"
    )
    result = _run(tmp_path, monkeypatch, results_jsonl=results_jsonl)
    payload = json.loads((result.run_dir / "result.json").read_text())
    assert payload["trajectory_source"] == "hosted_env"


def test_hosted_env_records_imported_source_provenance(tmp_path, monkeypatch):
    """source provenance must declare ``type=hosted_env`` (not ``github``)."""
    result = _run(tmp_path, monkeypatch)
    payload = json.loads((result.run_dir / "result.json").read_text())

    source = payload["source"]
    assert source["type"] == "hosted_env"
    assert source["provider"] == "primeintellect"
    assert source["env_uid"] == "primeintellect:primeintellect/general-agent@0.1.1"
    assert source["version"] == "0.1.1"


def test_hosted_env_writes_config_and_timing_and_prompts(tmp_path, monkeypatch):
    """The supporting trio (config.json, timing.json, prompts.json) is part of the contract."""
    result = _run(tmp_path, monkeypatch)
    assert (result.run_dir / "config.json").exists()
    assert (result.run_dir / "timing.json").exists()
    assert (result.run_dir / "prompts.json").exists()

    config = json.loads((result.run_dir / "config.json").read_text())
    assert config["environment"] == "hosted_env"
    assert config["hosted_env"]["env_uid"] == (
        "primeintellect:primeintellect/general-agent@0.1.1"
    )


def test_hosted_env_keeps_raw_evidence_for_forensics(tmp_path, monkeypatch):
    """Raw vf-eval evidence moves to hosted_env/ so it does not clobber the contract."""
    result = _run(tmp_path, monkeypatch)
    hosted_dir = result.run_dir / "hosted_env"
    assert (hosted_dir / "hosted_run.json").exists()
    assert (hosted_dir / "stdout.log").exists()
    assert (hosted_dir / "stderr.log").exists()


# ── HOE-1: reward bounds gate (hosted evidence == native contract) ────


def test_hosted_env_out_of_range_headline_reward_is_not_persisted(
    tmp_path, monkeypatch
):
    """A reward outside [0,1] must not land as a terminal reward (native parity)."""
    _patch_run_custom(monkeypatch, stdout="reward: avg - 4.000\n")
    result = _invoke(tmp_path)
    payload = json.loads((result.run_dir / "result.json").read_text())

    assert payload["rewards"] is None
    assert payload["error"] and "out of [0,1]" in payload["error"]
    # The raw value is kept for forensics but never as a terminal event.
    assert result.raw_reward == 4.0
    rewards_path = result.run_dir / "rewards.jsonl"
    if rewards_path.exists():
        values = [
            json.loads(line)["value"]
            for line in rewards_path.read_text().splitlines()
            if line
        ]
        assert all(v is None or v <= 1.0 for v in values)


def test_hosted_env_out_of_range_rubric_score_is_not_written(tmp_path, monkeypatch):
    """An out-of-contract per-example score (1e9) must not become a process event."""
    results_jsonl = json.dumps({"prompt": [], "completion": [], "reward": 1e9}) + "\n"
    _patch_run_custom(
        monkeypatch, stdout="reward: avg - 1.000\n", results_jsonl=results_jsonl
    )
    result = _invoke(tmp_path)
    rewards_path = result.run_dir / "rewards.jsonl"
    process_values = [
        json.loads(line)["value"]
        for line in rewards_path.read_text().splitlines()
        if line and json.loads(line)["type"] == "process"
    ]
    assert all(v <= 1.0 for v in process_values), "out-of-range score leaked"


# ── HOE-2: aborts fail closed (errored, not failed 0.0) ───────────────


def test_hosted_env_aborted_run_is_errored_not_zero_reward(tmp_path, monkeypatch):
    """An all-aborted vf-eval (reward 0.0 + abort + rc 0) must classify errored."""
    _patch_run_custom(
        monkeypatch,
        stdout="reward: avg - 0.000\n",
        stderr="Aborted rollout due to NotFoundError: model not found",
        returncode=0,
    )
    result = _invoke(tmp_path)
    payload = json.loads((result.run_dir / "result.json").read_text())

    assert payload["rewards"] is None
    assert payload["error"] and "NotFoundError" in payload["error"]
    assert payload["verifier_error"] and "NotFoundError" in payload["verifier_error"]
    # No misleading terminal reward=0.0 event.
    rewards_path = result.run_dir / "rewards.jsonl"
    assert not rewards_path.exists()
    assert _classify(result) == "errored"


def test_hosted_env_genuine_zero_reward_still_fails(tmp_path, monkeypatch):
    """A genuine 0.0 episode (no abort) keeps its terminal reward and is 'failed'."""
    _patch_run_custom(monkeypatch, stdout="reward: avg - 0.000\n", returncode=0)
    result = _invoke(tmp_path)
    payload = json.loads((result.run_dir / "result.json").read_text())

    assert payload["rewards"] == {"reward": 0.0}
    rewards_path = result.run_dir / "rewards.jsonl"
    terminal = [
        json.loads(line)
        for line in rewards_path.read_text().splitlines()
        if line and json.loads(line)["type"] == "terminal"
    ]
    assert terminal and terminal[0]["value"] == 0.0
    assert _classify(result) == "failed"
    # The abort path and the genuine-zero path must not collapse to one bucket.
    assert _classify(result) != "errored"


# ── HOE-3: unparseable vf-eval output fails closed ────────────────────


def test_hosted_env_unparseable_reward_is_flagged_as_error(tmp_path, monkeypatch):
    """rc 0 with no reward line and no abort marker must surface a parse error."""
    _patch_run_custom(
        monkeypatch,
        stdout="some new summary format with no reward line\n",
        returncode=0,
    )
    result = _invoke(tmp_path)
    payload = json.loads((result.run_dir / "result.json").read_text())

    assert payload["error"] is not None
    assert "could not parse reward" in payload["error"]
    assert payload["rewards"] is None
    assert _classify(result) == "errored"


# ── HOE-4: rewards.jsonl carries space/granularity ────────────────────


def test_hosted_env_rewards_carry_space_and_granularity(tmp_path, monkeypatch):
    """Hosted terminal/rubric events must carry the (space, granularity) tags."""
    results_jsonl = json.dumps({"prompt": [], "completion": [], "reward": 0.5}) + "\n"
    _patch_run_custom(
        monkeypatch, stdout="reward: avg - 1.000\n", results_jsonl=results_jsonl
    )
    result = _invoke(tmp_path)
    events = [
        json.loads(line)
        for line in (result.run_dir / "rewards.jsonl").read_text().splitlines()
        if line
    ]
    terminal = next(e for e in events if e["type"] == "terminal")
    assert terminal["space"] == "output"
    assert terminal["granularity"] == "terminal"
    rubric = [e for e in events if e["type"] == "process"]
    assert rubric, "expected a per-example rubric process event"
    assert all(e["space"] == "output" for e in rubric)
    assert all(e["granularity"] == "step" for e in rubric)


def test_hosted_env_promotes_verifier_supplied_space_out_of_meta(tmp_path, monkeypatch):
    """A verifier-supplied space/granularity is promoted to first-class fields."""
    results_jsonl = (
        json.dumps(
            {
                "prompt": [],
                "completion": [],
                "reward": 0.5,
                "space": "memory",
                "granularity": "step",
            }
        )
        + "\n"
    )
    _patch_run_custom(
        monkeypatch, stdout="reward: avg - 1.000\n", results_jsonl=results_jsonl
    )
    result = _invoke(tmp_path)
    process = [
        json.loads(line)
        for line in (result.run_dir / "rewards.jsonl").read_text().splitlines()
        if line and json.loads(line)["type"] == "process"
    ]
    assert process, "expected a per-example rubric process event"
    # The space tag is promoted onto the event, not buried in meta.
    assert process[0]["space"] == "memory"
    assert "space" not in process[0]["meta"]
