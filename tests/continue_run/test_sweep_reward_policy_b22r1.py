"""Guards FF-sw-evaluation-C4 against d97bf9f6 reward-policy drift."""

import pytest

from benchflow.continue_run.orchestrator import build_rollout_config
from benchflow.continue_run.run_folder import RunFolder, RunFolderError


@pytest.mark.parametrize("environment", ["docker", "remote-docker", "daytona", "modal"])
@pytest.mark.parametrize(
    "recorded,current", [(True, None), (True, "0"), (False, "1"), (False, "true")]
)
def test_continuation_refuses_reward_policy_change(
    tmp_path, monkeypatch, environment, recorded, current
):
    if current is None:
        monkeypatch.delenv("BENCHFLOW_REWARD_LENIENT", raising=False)
    else:
        monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", current)
    run = RunFolder(
        tmp_path,
        {"environment": environment, "reward_lenient": recorded},
        {},
        ["go"],
        [],
    )
    with pytest.raises(RunFolderError, match="BENCHFLOW_REWARD_LENIENT"):
        build_rollout_config(
            run,
            task_path=tmp_path,
            live_model="test",
            agent_env={},
            timeout=30,
            output_dir=tmp_path,
            rollout_name="continued",
        )


@pytest.mark.parametrize(
    "recorded,current",
    [(True, "1"), (True, "TRUE"), (True, "yes"), (False, "0"), (False, "")],
)
def test_continuation_accepts_matching_reward_policy(
    tmp_path, monkeypatch, recorded, current
):
    monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", current)
    run = RunFolder(tmp_path, {"reward_lenient": recorded}, {}, ["go"], [])
    assert build_rollout_config(
        run,
        task_path=tmp_path,
        live_model="test",
        agent_env={},
        timeout=30,
        output_dir=tmp_path,
        rollout_name="continued",
    )


@pytest.mark.parametrize("recorded", [None, "true", "false", 0, 1, [], {}])
def test_continuation_refuses_invalid_recorded_reward_policy(
    tmp_path, monkeypatch, recorded
):
    monkeypatch.delenv("BENCHFLOW_REWARD_LENIENT", raising=False)
    run = RunFolder(tmp_path, {"reward_lenient": recorded}, {}, ["go"], [])
    with pytest.raises(RunFolderError, match="reward_lenient"):
        build_rollout_config(
            run,
            task_path=tmp_path,
            live_model="test",
            agent_env={},
            timeout=30,
            output_dir=tmp_path,
            rollout_name="continued",
        )


@pytest.mark.parametrize("proxy_mode", ["host", "sandbox", "auto"])
@pytest.mark.parametrize("environment", ["docker", "remote-docker", "daytona", "modal"])
async def test_reward_policy_refused_before_proxy_or_credentials(
    tmp_path, monkeypatch, proxy_mode, environment
):
    from benchflow.continue_run import orchestrator

    monkeypatch.delenv("BENCHFLOW_REWARD_LENIENT", raising=False)
    run = RunFolder(
        tmp_path,
        {"environment": environment, "reward_lenient": True, "model": "test"},
        {},
        ["go"],
        [],
    )
    monkeypatch.setattr(orchestrator, "load_run_folder", lambda *a, **kw: run)
    monkeypatch.setattr(orchestrator, "resolve_task_path", lambda *a: tmp_path)
    monkeypatch.setattr(orchestrator, "require_replayable_task", lambda *a: tmp_path)

    def no_side_effects(*args, **kwargs):
        pytest.fail("policy mismatch must refuse before credentials or proxy setup")

    monkeypatch.setattr(orchestrator, "LiteLLMLiveForwarder", no_side_effects)
    monkeypatch.setattr(orchestrator, "ReplayProxy", no_side_effects)
    with pytest.raises(RunFolderError, match="BENCHFLOW_REWARD_LENIENT=1"):
        await orchestrator.continue_run(tmp_path, proxy_mode=proxy_mode)
