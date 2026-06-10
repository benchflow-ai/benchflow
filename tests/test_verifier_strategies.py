"""End-to-end ``Verifier.verify()`` coverage for native verifier.md strategies.

``verifier/verifier.md`` can select reward-kit, agent-judge, or ors-episode
strategies; ``verify()`` dispatches on the selected strategy type. These tests
drive the full ``verify()`` path over a fake sandbox and a real rollout
directory — no network, no containers, no provider SDKs.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchflow.task import RolloutPaths, Verifier
from benchflow.task.config import TaskConfig
from benchflow.task.verifier import (
    AgentJudgeInputError,
    ORSEpisodeInputError,
    VerifierOutputParseError,
)


def _make_task(verifier_dir: Path, instruction: str = "Do the task.") -> MagicMock:
    """Task stub with real config defaults and a real verifier package dir."""
    task = MagicMock()
    task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
    task.instruction = instruction
    task.paths.tests_dir = verifier_dir
    return task


def _make_rollout(tmp_path: Path) -> RolloutPaths:
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    return rollout_paths


def _write_verifier_package(
    tmp_path: Path,
    verifier_md: str,
    files: dict[str, str] | None = None,
) -> Path:
    verifier_dir = tmp_path / "task" / "verifier"
    verifier_dir.mkdir(parents=True)
    (verifier_dir / "verifier.md").write_text(dedent(verifier_md))
    for relative, content in (files or {}).items():
        path = verifier_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return verifier_dir


# ---------------------------------------------------------------------------
# reward-kit strategy
# ---------------------------------------------------------------------------

_REWARD_KIT_VERIFIER_MD = """\
---
verifier:
  default_strategy: rewardkit
  strategies:
    rewardkit:
      type: reward-kit
      root: reward_kit
      criteria: rubrics/criteria.toml
---
"""


def _criteria_toml(
    aggregation: str,
    weights: dict[str, float],
    threshold: float = 0.75,
) -> str:
    lines = [
        "[scoring]",
        f'aggregation = "{aggregation}"',
        f"threshold = {threshold}",
        "",
    ]
    for name, weight in weights.items():
        lines.extend(
            [
                "[[criterion]]",
                f'name = "{name}"',
                f'description = "{name} check"',
                f"weight = {weight}",
                "",
            ]
        )
    return "\n".join(lines)


def _reward_kit_verifier(
    tmp_path: Path,
    *,
    criteria_toml: str,
    reward_payload: dict,
    calls: dict | None = None,
) -> tuple[Verifier, RolloutPaths]:
    """Verifier over a mounted sandbox whose runner exec writes reward.json."""
    verifier_dir = _write_verifier_package(
        tmp_path,
        _REWARD_KIT_VERIFIER_MD,
        files={
            "reward_kit/reward.py": "raise SystemExit(0)\n",
            "rubrics/criteria.toml": criteria_toml,
        },
    )
    rollout_paths = _make_rollout(tmp_path)

    sandbox = MagicMock()
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()

    async def run_runner(*args, **kwargs):
        if calls is not None:
            calls["command"] = kwargs.get("command")
            calls["env"] = kwargs.get("env")
        rollout_paths.reward_json_path.write_text(json.dumps(reward_payload))
        return MagicMock(exit_code=0, returncode=0)

    sandbox.exec = AsyncMock(side_effect=run_runner)
    return Verifier(_make_task(verifier_dir), rollout_paths, sandbox), rollout_paths


class TestRewardKitStrategy:
    @pytest.mark.parametrize(
        ("aggregation", "weights", "metrics", "expected_reward"),
        [
            ("weighted_mean", {"c1": 3.0, "c2": 1.0}, {"c1": 1.0, "c2": 0.5}, 0.875),
            ("weighted_sum", {"c1": 0.5, "c2": 0.25}, {"c1": 1.0, "c2": 1.0}, 0.75),
            ("all_pass", {"c1": 1.0, "c2": 1.0}, {"c1": 1.0, "c2": 0.5}, 1.0),
            ("all_pass", {"c1": 1.0, "c2": 1.0}, {"c1": 1.0, "c2": 0.4}, 0.0),
            ("any_pass", {"c1": 1.0, "c2": 1.0}, {"c1": 0.5, "c2": 0.0}, 1.0),
            ("any_pass", {"c1": 1.0, "c2": 1.0}, {"c1": 0.4, "c2": 0.0}, 0.0),
            ("threshold", {"c1": 1.0, "c2": 1.0}, {"c1": 1.0, "c2": 0.5}, 1.0),
            ("threshold", {"c1": 1.0, "c2": 1.0}, {"c1": 0.9, "c2": 0.5}, 0.0),
        ],
    )
    @pytest.mark.asyncio
    async def test_aggregation_policies_compute_reward(
        self,
        tmp_path: Path,
        aggregation: str,
        weights: dict[str, float],
        metrics: dict[str, float],
        expected_reward: float,
    ) -> None:
        """Each declared criteria aggregation produces its exact reward."""
        verifier, rollout_paths = _reward_kit_verifier(
            tmp_path,
            criteria_toml=_criteria_toml(aggregation, weights),
            reward_payload={"metrics": metrics},
        )

        result = await verifier.verify()

        assert result.rewards["reward"] == expected_reward
        assert result.rewards["metrics"] == metrics
        persisted = json.loads(rollout_paths.reward_json_path.read_text())
        assert persisted["reward"] == expected_reward

    @pytest.mark.asyncio
    async def test_runner_invocation_contract(self, tmp_path: Path) -> None:
        """The sandbox runner command, env, and manifest carry the package paths."""
        calls: dict = {}
        verifier, rollout_paths = _reward_kit_verifier(
            tmp_path,
            criteria_toml=_criteria_toml("weighted_mean", {"c1": 3.0, "c2": 1.0}),
            reward_payload={"metrics": {"c1": 1.0, "c2": 0.5}},
            calls=calls,
        )

        result = await verifier.verify()

        assert result.rewards["reward"] == 0.875
        assert calls["command"] == (
            "cd /verifier && python reward_kit/reward.py "
            "> /logs/verifier/test-stdout.txt 2>&1"
        )
        env = calls["env"]
        assert env["BENCHFLOW_VERIFIER_DIR"] == "/verifier"
        assert env["BENCHFLOW_REWARD_KIT_ROOT"] == "/verifier/reward_kit"
        assert env["BENCHFLOW_REWARD_KIT_CRITERIA"] == "/verifier/rubrics/criteria.toml"
        assert env["BENCHFLOW_REWARD_JSON"] == "/logs/verifier/reward.json"
        assert (
            env["BENCHFLOW_REWARD_KIT_MANIFEST"]
            == "/logs/verifier/reward-kit-manifest.json"
        )
        manifest = json.loads(rollout_paths.reward_kit_manifest_path.read_text())
        assert manifest["strategy"]["root"] == "reward_kit"
        assert manifest["strategy"]["entrypoint"] == "reward.py"
        assert manifest["criteria_policy"]["method"] == "weighted_mean"
        assert manifest["criteria_policy"]["criteria"] == ["c1", "c2"]
        assert manifest["criteria_policy"]["weights"] == {"c1": 3.0, "c2": 1.0}

    @pytest.mark.asyncio
    async def test_metrics_must_match_declared_criteria(self, tmp_path: Path) -> None:
        """A runner that drops a declared criterion is a verifier output error."""
        verifier, _ = _reward_kit_verifier(
            tmp_path,
            criteria_toml=_criteria_toml("weighted_mean", {"c1": 1.0, "c2": 1.0}),
            reward_payload={"metrics": {"c1": 1.0}},
        )

        with pytest.raises(
            VerifierOutputParseError, match="must match declared criteria"
        ):
            await verifier.verify()


# ---------------------------------------------------------------------------
# agent-judge strategy
# ---------------------------------------------------------------------------

_AGENT_JUDGE_VERIFIER_MD = """\
---
verifier:
  default_strategy: judge
  strategies:
    judge:
      type: agent-judge
      role: adjudicator
      model: stub-judge-1
      isolation: verifier-only
      inputs: [artifacts/answer.txt]
---

## role:adjudicator

Score the declared answer evidence.
"""


def _agent_judge_verifier(
    tmp_path: Path,
    *,
    answer: str | None = "the final answer is 42\n",
) -> tuple[Verifier, RolloutPaths]:
    verifier_dir = _write_verifier_package(tmp_path, _AGENT_JUDGE_VERIFIER_MD)
    rollout_paths = _make_rollout(tmp_path)
    if answer is not None:
        (rollout_paths.artifacts_dir / "answer.txt").write_text(answer)
    return Verifier(_make_task(verifier_dir), rollout_paths, MagicMock()), rollout_paths


class TestAgentJudgeStrategy:
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_pass_verdict_scores_full_reward(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = '{"verdict": "pass", "reasoning": "good"}'
        verifier, rollout_paths = _agent_judge_verifier(tmp_path)

        result = await verifier.verify()

        assert result.rewards["reward"] == 1.0
        assert result.rewards["metadata"] == {
            "source": "agent-judge",
            "strategy": "judge",
            "role": "adjudicator",
            "model": "stub-judge-1",
        }
        model, prompt = mock_judge.await_args.args
        assert model == "stub-judge-1"
        assert "Score the declared answer evidence." in prompt
        assert "--- artifacts/answer.txt ---" in prompt
        assert "the final answer is 42" in prompt
        persisted = json.loads(rollout_paths.reward_json_path.read_text())
        assert persisted["reward"] == 1.0
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["verdict"] == {"verdict": "pass", "reasoning": "good"}
        assert details["aggregate"] == {"reward": 1.0, "method": "agent-judge-score"}
        assert details["inputs"] == [
            {"path": "artifacts/answer.txt", "chars": 23, "truncated": False}
        ]

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_fail_verdict_scores_zero(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = '{"verdict": "fail", "reasoning": "bad"}'
        verifier, rollout_paths = _agent_judge_verifier(tmp_path)

        result = await verifier.verify()

        assert result.rewards["reward"] == 0.0
        persisted = json.loads(rollout_paths.reward_json_path.read_text())
        assert persisted["reward"] == 0.0

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_fractional_score_verdict_is_preserved(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        mock_judge.return_value = '{"score": 0.25}'
        verifier, _ = _agent_judge_verifier(tmp_path)

        result = await verifier.verify()

        assert result.rewards["reward"] == 0.25

    @pytest.mark.parametrize(
        "judge_response",
        [
            "definitely a pass, great work",
            '{"verdict": "maybe"}',
            '{"score": 1.5}',
        ],
    )
    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_malformed_verdict_fails_closed(
        self, mock_judge: AsyncMock, tmp_path: Path, judge_response: str
    ) -> None:
        """An unusable verdict raises instead of scoring, and emits no reward."""
        mock_judge.return_value = judge_response
        verifier, rollout_paths = _agent_judge_verifier(tmp_path)
        rollout_paths.reward_json_path.write_text('{"reward": 1.0}')

        with pytest.raises(VerifierOutputParseError, match="invalid verdict"):
            await verifier.verify()

        assert not rollout_paths.reward_json_path.exists()
        assert not rollout_paths.reward_text_path.exists()

    @patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_missing_declared_input_is_input_error(
        self, mock_judge: AsyncMock, tmp_path: Path
    ) -> None:
        """Evidence collection fails before the judge model is ever called."""
        verifier, _ = _agent_judge_verifier(tmp_path, answer=None)

        with pytest.raises(
            AgentJudgeInputError, match="was not found under the rollout"
        ):
            await verifier.verify()

        mock_judge.assert_not_awaited()


# ---------------------------------------------------------------------------
# ors-episode strategy
# ---------------------------------------------------------------------------

_ORS_VERIFIER_MD = """\
---
verifier:
  default_strategy: ors
  strategies:
    ors:
      type: ors-episode
      inputs: [trajectory/ors-rewards.jsonl]
---
"""


def _ors_verifier(
    tmp_path: Path,
    records: list[dict] | None,
) -> tuple[Verifier, RolloutPaths]:
    verifier_dir = _write_verifier_package(tmp_path, _ORS_VERIFIER_MD)
    rollout_paths = _make_rollout(tmp_path)
    if records is not None:
        evidence = rollout_paths.rollout_dir / "trajectory" / "ors-rewards.jsonl"
        evidence.parent.mkdir(parents=True)
        evidence.write_text("".join(json.dumps(r) + "\n" for r in records))
    return Verifier(_make_task(verifier_dir), rollout_paths, MagicMock()), rollout_paths


class TestORSEpisodeStrategy:
    @pytest.mark.asyncio
    async def test_terminal_reward_extracted_dense_steps_retained(
        self, tmp_path: Path
    ) -> None:
        """The terminal event sets the reward; dense step events are kept as
        evidence but never folded into the headline value."""
        records = [
            {
                "type": "dense",
                "reward": 0.2,
                "step": 1,
                "space": "action",
                "granularity": "step",
            },
            {"type": "terminal", "reward": 0.6},
            {
                "type": "dense",
                "reward": 0.9,
                "step": 2,
                "space": "action",
                "granularity": "step",
            },
        ]
        verifier, rollout_paths = _ors_verifier(tmp_path, records)

        result = await verifier.verify()

        assert result.rewards["reward"] == 0.6
        assert result.rewards["metadata"]["source"] == "ors-episode"
        assert result.rewards["metadata"]["strategy"] == "ors"
        ors = result.rewards["metadata"]["ors"]
        assert ors["is_valid"] is True
        assert ors["reward"] == 0.6
        assert ors["metadata"]["items"] == {"ors": 0.6}
        events = ors["metadata"]["events"]
        assert [event["reward"] for event in events] == [0.2, 0.6, 0.9]
        assert [event["granularity"] for event in events] == [
            "step",
            "terminal",
            "step",
        ]
        persisted = json.loads(rollout_paths.reward_json_path.read_text())
        assert persisted["reward"] == 0.6
        details = json.loads(rollout_paths.reward_details_json_path.read_text())
        assert details["aggregate"] == {"reward": 0.6, "method": "ors-terminal"}
        assert details["inputs"][0]["path"] == "trajectory/ors-rewards.jsonl"
        assert details["inputs"][0]["records"] == 3

    @pytest.mark.asyncio
    async def test_dense_only_episode_is_a_parse_error(self, tmp_path: Path) -> None:
        """Step events alone never aggregate into a reward — that is an error."""
        records = [
            {"type": "dense", "reward": 0.2, "step": 1, "granularity": "step"},
            {"type": "dense", "reward": 0.9, "step": 2, "granularity": "step"},
        ]
        verifier, rollout_paths = _ors_verifier(tmp_path, records)

        with pytest.raises(
            VerifierOutputParseError, match="did not include a terminal reward"
        ):
            await verifier.verify()

        assert not rollout_paths.reward_json_path.exists()

    @pytest.mark.asyncio
    async def test_out_of_range_terminal_reward_is_rejected(
        self, tmp_path: Path
    ) -> None:
        records = [{"type": "terminal", "reward": 1.5}]
        verifier, _ = _ors_verifier(tmp_path, records)

        with pytest.raises(VerifierOutputParseError, match=r"between 0\.0 and 1\.0"):
            await verifier.verify()

    @pytest.mark.asyncio
    async def test_missing_declared_evidence_is_input_error(
        self, tmp_path: Path
    ) -> None:
        verifier, _ = _ors_verifier(tmp_path, records=None)

        with pytest.raises(
            ORSEpisodeInputError, match="was not found under the rollout"
        ):
            await verifier.verify()
