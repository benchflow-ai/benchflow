"""trainer/verifiers.jsonl writes an unscored trial as unscored
(audit RECORDS-4).

The Verifiers/ORS record wrote ``"reward": 0.0`` and ``"is_completed":
true`` for every errored and unscored trial (a verifier that crashed, a lost
sandbox, a reward of NaN), the rule results.jsonl was changed to forbid: a
consumer training on the file read them as completed episodes that scored
0. Now such a trial's reward is null and it is not completed; an agent that
failed or was cut off is not completed either, whatever the verifier scored,
as in results.jsonl.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.trajectories.export import (
    ROLLOUT_ARTIFACT_RELPATH,
    write_rollout_verifiers_jsonl,
)

TRAJECTORY = [
    {"type": "user_message", "text": "Fit the model."},
    {"type": "agent_message", "text": "Done."},
]


def _record(path: Path) -> dict:
    return json.loads((path / ROLLOUT_ARTIFACT_RELPATH).read_text().strip())


@pytest.mark.parametrize(
    "rewards,error",
    [
        (None, "verifier crashed: exit 3"),
        (None, None),
        ({"reward": float("nan")}, None),
        ({"reward": 1.0}, "verifier timed out"),
    ],
)
def test_an_unscored_trial_has_a_null_reward_and_is_not_completed(
    tmp_path, rewards, error
):
    write_rollout_verifiers_jsonl(
        tmp_path,
        task_id="t",
        prompts=["Fit the model."],
        trajectory=TRAJECTORY,
        rewards=rewards,
        model="claude-haiku-5-5",
        environment="t",
        error=error,
    )
    record = _record(tmp_path)
    assert record["reward"] is None
    assert record["is_completed"] is False
    assert record["info"]["reward_valid"] is False or error is not None


def test_a_scored_zero_stays_a_zero(tmp_path):
    write_rollout_verifiers_jsonl(
        tmp_path,
        task_id="t",
        prompts=["Fit the model."],
        trajectory=TRAJECTORY,
        rewards={"reward": 0.0},
        model="claude-haiku-5-5",
        environment="t",
    )
    record = _record(tmp_path)
    assert record["reward"] == 0.0 and record["is_completed"] is True


@pytest.mark.asyncio
async def test_a_rollouts_trainer_record_agrees_with_its_results_row(
    tmp_path, monkeypatch
):
    """A trial whose agent timed out and that the verifier never scored:
    the rollout's trainer/verifiers.jsonl and results.jsonl both say
    unscored and not completed."""
    from tests.test_en374_salvage import _rollout

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    rollout._error = "Agent timed out after 14400s"
    rollout._verifier_error = "verifier crashed: the sandbox is gone"
    rollout._build_result()

    record = _record(trial)
    assert record["reward"] is None
    assert record["is_completed"] is False
    row = json.loads((trial / "results.jsonl").read_text().splitlines()[0])
    assert row["reward"] is None and row["is_completed"] is False


@pytest.mark.asyncio
async def test_a_scored_trial_whose_agent_failed_is_not_completed(
    tmp_path, monkeypatch
):
    from tests.test_en374_salvage import _rollout

    home = tmp_path / "sandbox-home"
    home.mkdir()
    rollout, trial = _rollout(
        tmp_path, monkeypatch, agent="claude-agent-acp", harness="native", home=home
    )
    rollout._error = "Agent timed out after 900s"
    rollout._rewards = {"reward": 0.0}
    rollout._build_result()
    record = _record(trial)
    assert record["reward"] == 0.0
    assert record["is_completed"] is False


def test_bench_train_convert_skips_an_unscored_verifiers_row(tmp_path):
    """``bench train convert --format prime-sft <file>.jsonl`` wrote an
    unscored trial out as SFT data with its reward dropped, unless
    --min-reward was set: one scored and one crashed row gave 2 rows."""
    from benchflow.trajectories.export_prime_sft import export_prime_sft_jsonl

    rows = []
    for name, rewards, error in (
        ("scored", {"reward": 1.0}, None),
        ("crashed", None, "verifier crashed: exit 3"),
    ):
        folder = tmp_path / name
        folder.mkdir()
        rows.append(
            write_rollout_verifiers_jsonl(
                folder,
                task_id=name,
                prompts=["Fit the model."],
                trajectory=TRAJECTORY,
                rewards=rewards,
                model="claude-haiku-5-5",
                environment=name,
                error=error,
            )
        )
    source = tmp_path / "verifiers.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "prime-sft.jsonl"

    stats = export_prime_sft_jsonl(source, out)

    assert stats.rollouts_seen == 2
    assert stats.rows_written == 1
    assert len(out.read_text().splitlines()) == 1


def test_an_older_verifiers_rows_unscored_zero_reads_as_unscored(tmp_path):
    """Before this release verifiers.jsonl wrote 0.0 with
    ``info.reward_valid`` false for an unscored trial; bf.load_job reads it
    as unscored. A scored 0.0 stays a 0."""
    from benchflow.jobs import load_results_jsonl

    def row(task: str, reward_valid: bool) -> dict:
        return {
            "example_id": 0,
            "prompt": [{"role": "user", "content": "Fit the model."}],
            "completion": [{"role": "assistant", "content": "Done."}],
            "reward": 0.0,
            "metrics": {},
            "is_completed": True,
            "is_truncated": False,
            "info": {
                "task_id": task,
                "environment": task,
                "model": "claude-haiku-5-5",
                "reward_valid": reward_valid,
                "reward_metadata": {"items": {}, "events": [], "error": None},
            },
        }

    path = tmp_path / "verifiers.jsonl"
    path.write_text(
        json.dumps(row("crashed", False))
        + "\n"
        + json.dumps(row("failed", True))
        + "\n"
    )
    rewards = {t.task_name: t.result.reward for t in load_results_jsonl(path)}
    assert rewards == {"crashed": None, "failed": 0.0}


def test_a_reward_outside_0_1_the_task_allows_is_written_as_scored(tmp_path):
    """The review's F3: a task scoring in [-1, 1] scored -0.5; verifiers.jsonl
    wrote it as unscored (null) while results.jsonl wrote -0.5, and prime-sft
    dropped it. The verifier already held it to the task's range."""
    write_rollout_verifiers_jsonl(
        tmp_path,
        task_id="t",
        prompts=["Fit the model."],
        trajectory=TRAJECTORY,
        rewards={"reward": -0.5},
        model="claude-haiku-5-5",
        environment="t",
    )
    record = _record(tmp_path)
    assert record["reward"] == -0.5 and record["is_completed"] is True


def test_bench_train_convert_skips_an_older_unscored_verifiers_row(tmp_path):
    """The review's F4: a row from before this release wrote an unscored
    trial as 0.0 beside info.reward_valid false; bf.load_job reads it as
    unscored, and the converter now skips it too."""
    from benchflow.trajectories.export_prime_sft import export_prime_sft_jsonl

    folder = tmp_path / "scored"
    folder.mkdir()
    scored = write_rollout_verifiers_jsonl(
        folder,
        task_id="scored",
        prompts=["Fit the model."],
        trajectory=TRAJECTORY,
        rewards={"reward": 1.0},
        model="claude-haiku-5-5",
        environment="scored",
    )
    older = {
        **scored,
        "reward": 0.0,
        "is_completed": True,
        "info": {**scored["info"], "task_id": "crashed", "reward_valid": False},
    }
    source = tmp_path / "verifiers.jsonl"
    source.write_text(json.dumps(scored) + "\n" + json.dumps(older) + "\n")
    stats = export_prime_sft_jsonl(source, tmp_path / "prime-sft.jsonl")
    assert stats.rollouts_seen == 2 and stats.rows_written == 1
