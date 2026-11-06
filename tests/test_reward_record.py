"""Reward files and recording what ran (#1178, PLAN P6; round PY8 target V4).

- A reward file with extra fields beside its numeric ``reward`` scores, and
  the fields are kept (#1178: CooperBench's ``feature1_passed``, KUMO's
  ``answer``), and so does a details-only reward.json beside reward.txt's
  score (#1192).
- A reward outside the task's range is never clamped and never thrown away
  with the trial: the trial is unscored with a named verifier problem
  (``reward out of range:``, category ``reward_out_of_range``) and the raw
  number is kept in ``reward_out_of_range_info`` (Rust 0.8.2's R302).
- ``config.json`` records the lenient mode and the models that answered.
- Result readers count a scored timeout (an agent timeout the verifier still
  scored) as scored.
- ``bench eval run`` with no agent warns once, naming the default agent and
  model, and the job records it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from benchflow.acp.session import ACPSession
from benchflow.rewards.validation import validate_reward_map
from benchflow.rollout import _ensure_canonical_rewards
from benchflow.task.verifier import Verifier, VerifierOutputParseError

COOPERBENCH = {
    "reward": 1,
    "feature1_passed": True,
    "merge_ok": True,
    "strategy": "oracle",
}
KUMO = {"reward": 1.0, "answer": "Empowering", "is_exact_match": True}


def _verifier(tmp_path: Path, *, reward_range: list[float] | None = None) -> Verifier:
    from benchflow.task import TaskConfig

    task = MagicMock()
    verifier: dict = {} if reward_range is None else {"reward_range": reward_range}
    task.config = TaskConfig.model_validate({"verifier": verifier})
    paths = MagicMock()
    paths.reward_text_path = tmp_path / "reward.txt"
    paths.reward_json_path = tmp_path / "reward.json"
    return Verifier(task=task, rollout_paths=paths, sandbox=MagicMock())


# ---------------------------------------------------------------------------
# Extra fields are accepted and kept
# ---------------------------------------------------------------------------


class TestExtraFields:
    @pytest.mark.parametrize("payload", [COOPERBENCH, KUMO])
    def test_extra_fields_beside_the_reward_are_kept(self, payload: dict) -> None:
        assert validate_reward_map(payload, source="reward JSON") == payload

    @pytest.mark.parametrize("payload", [COOPERBENCH, KUMO])
    def test_reward_json_with_extra_fields_scores(
        self, tmp_path: Path, payload: dict
    ) -> None:
        verifier = _verifier(tmp_path)
        (tmp_path / "reward.json").write_text(json.dumps(payload))
        rewards = verifier._parse_reward_json()
        assert rewards == payload
        # The final gate before scoring keeps them too.
        assert _ensure_canonical_rewards(rewards) == payload

    def test_a_reward_txt_score_keeps_a_details_only_reward_json(
        self, tmp_path: Path
    ) -> None:
        # #1192: the score in reward.txt, KUMO's details (no reward) in
        # reward.json. Both used to fail the trial on "answer".
        (tmp_path / "reward.txt").write_text("1\n")
        details = {"answer": "Empowering", "is_exact_match": True}
        (tmp_path / "reward.json").write_text(json.dumps(details))
        rewards = _verifier(tmp_path)._parse_reward_json_with_text_compat()
        assert rewards == {"reward": 1.0, **details}
        assert _ensure_canonical_rewards(rewards) == rewards

    def test_a_number_in_the_details_is_a_metric(self, tmp_path: Path) -> None:
        # KUMO's real reward.json has "seed": 3 (#1192): a top-level number
        # is a metric, held to the range, and the warning says where details
        # go instead.
        (tmp_path / "reward.txt").write_text("1")
        (tmp_path / "reward.json").write_text(json.dumps({"answer": "x", "seed": 3}))
        with pytest.raises(VerifierOutputParseError) as caught:
            _verifier(tmp_path)._parse_reward_json_with_text_compat()
        assert _raw(caught.value) == ("seed", 3, "verifier/reward.json")
        assert "reward-details.json" in str(caught.value)

    def test_details_only_reward_json_still_needs_a_score(self, tmp_path: Path) -> None:
        (tmp_path / "reward.json").write_text(json.dumps({"answer": "Empowering"}))
        with pytest.raises(VerifierOutputParseError):
            _verifier(tmp_path)._parse_reward_json_with_text_compat()
        (tmp_path / "reward.txt").write_text("yes")
        with pytest.raises(VerifierOutputParseError, match="text file"):
            _verifier(tmp_path)._parse_reward_json_with_text_compat()

    def test_reward_json_and_reward_txt_must_still_agree(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("1")
        (tmp_path / "reward.json").write_text(json.dumps({**KUMO, "reward": 0.0}))
        with pytest.raises(VerifierOutputParseError, match="does not match"):
            _verifier(tmp_path)._parse_reward_json_with_text_compat()

    def test_extra_fields_survive_lenient_mode(self) -> None:
        assert validate_reward_map(COOPERBENCH, lenient=True) == COOPERBENCH

    def test_a_flat_map_without_a_reward_still_needs_numbers(self) -> None:
        # Without a numeric ``reward`` there is no envelope to keep the field
        # beside: still refused, as in Rust's bench.
        with pytest.raises(ValueError):
            validate_reward_map({"feature1_passed": True, "acc": 1.0})

    def test_a_non_numeric_metric_is_still_refused(self) -> None:
        with pytest.raises(ValueError, match="invalid metric value"):
            validate_reward_map({"reward": 1.0, "metrics": {"label": "good"}})


# ---------------------------------------------------------------------------
# Out-of-range rewards: unscored, named, the raw value kept
# ---------------------------------------------------------------------------


def _raw(exc: BaseException) -> tuple:
    return (
        getattr(exc, "key", None),
        getattr(exc, "value", None),
        getattr(exc, "file", None),
    )


class TestOutOfRange:
    def test_reward_txt_above_one_keeps_its_raw_value(self, tmp_path: Path) -> None:
        # AlgoTune's speedup reward (#1178).
        (tmp_path / "reward.txt").write_text("1.0002\n")
        with pytest.raises(VerifierOutputParseError) as caught:
            _verifier(tmp_path)._parse_reward_text()
        assert _raw(caught.value) == ("reward", 1.0002, "verifier/reward.txt")
        assert "outside [0.0, 1.0]" in str(caught.value)

    def test_reward_json_above_one_keeps_its_raw_value(self, tmp_path: Path) -> None:
        (tmp_path / "reward.json").write_text(json.dumps({"reward": 1.5, "n": "x"}))
        with pytest.raises(VerifierOutputParseError) as caught:
            _verifier(tmp_path)._parse_reward_json()
        assert _raw(caught.value) == ("reward", 1.5, "verifier/reward.json")

    def test_an_out_of_range_metric_is_named(self, tmp_path: Path) -> None:
        (tmp_path / "reward.json").write_text(
            json.dumps({"reward": 1.0, "metrics": {"speedup": 2.5}})
        )
        with pytest.raises(VerifierOutputParseError) as caught:
            _verifier(tmp_path)._parse_reward_json()
        assert _raw(caught.value) == (
            "metrics.speedup",
            2.5,
            "verifier/reward.json",
        )

    def test_a_declared_range_bounds_the_reward(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("-1.5")
        with pytest.raises(VerifierOutputParseError) as caught:
            _verifier(tmp_path, reward_range=[-1.0, 1.0])._parse_reward_text()
        assert _raw(caught.value) == ("reward", -1.5, "verifier/reward.txt")
        assert "(the task's reward_range)" in str(caught.value)

    def test_a_declared_range_still_scores_inside_it(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("-1.0")
        verifier = _verifier(tmp_path, reward_range=[-1.0, 1.0])
        assert verifier._parse_reward_text() == {"reward": -1.0}

    def test_lenient_mode_does_not_swap_in_an_alias(self) -> None:
        # Before: lenient mode dropped the out-of-range reward and scored the
        # ``score`` alias instead, a score the verifier did not give.
        with pytest.raises(ValueError) as caught:
            validate_reward_map({"reward": 5.0, "score": 0.9}, lenient=True)
        assert _raw(caught.value)[:2] == ("reward", 5.0)

    def test_nan_is_no_reward_not_an_out_of_range_one(self, tmp_path: Path) -> None:
        (tmp_path / "reward.txt").write_text("nan")
        with pytest.raises(VerifierOutputParseError) as caught:
            _verifier(tmp_path)._parse_reward_text()
        assert _raw(caught.value) == (None, None, None)


class TestOutOfRangeOutcome:
    @pytest.fixture
    def verify_harness(self, tmp_path):
        from benchflow.sdk import SDK

        task = MagicMock()
        task.config.verifier.timeout_sec = 5
        task.config.verifier.env = None
        task.config.verifier.reward_range = None
        tp = MagicMock()
        tp.verifier_dir = tmp_path / "verifier"
        env = MagicMock()
        env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", exit_code=0))
        return SDK(), env, task, tp

    async def _verify(self, verify_harness, rewards: dict):
        sdk, env, task, tp = verify_harness
        verifier = MagicMock()
        verifier.verify = AsyncMock(return_value=MagicMock(rewards=rewards))
        with (
            patch("benchflow.task.Verifier", return_value=verifier),
            patch(
                "benchflow.sandbox.lockdown.harden_before_verify",
                new_callable=AsyncMock,
            ),
        ):
            return await sdk._verify(env, task, tp, {})

    @pytest.mark.asyncio
    async def test_an_out_of_range_reward_is_zero_with_a_named_warning(
        self, verify_harness, tmp_path: Path, caplog
    ) -> None:
        from benchflow.diagnostics import RolloutDiagnostics
        from benchflow.failures import cause_of
        from benchflow.rollout._results import _build_rollout_result

        rewards, verifier_error, diagnostic = await self._verify(
            verify_harness, {"reward": 1.0002}
        )
        assert rewards == {"reward": 0.0}
        assert verifier_error is None
        assert "reward out of range:" in caplog.text
        assert "1.0002" in caplog.text
        assert diagnostic is not None
        assert diagnostic.to_dict()["value"] == 1.0002

        diagnostics = RolloutDiagnostics()
        diagnostics.set(diagnostic)
        rollout_dir = tmp_path / "trial"
        rollout_dir.mkdir()
        from datetime import datetime

        _build_rollout_result(
            rollout_dir,
            task_name="t",
            rollout_name="t__1",
            agent="oracle",
            agent_name="oracle",
            model=None,
            n_tool_calls=0,
            prompts=[],
            error=None,
            verifier_error=verifier_error,
            trajectory=[],
            partial_trajectory=False,
            rewards=rewards,
            started_at=datetime.now(),
            timing={},
            diagnostics=diagnostics,
        )
        saved = json.loads((rollout_dir / "result.json").read_text())
        assert saved["rewards"] == {"reward": 0.0}
        assert saved.get("verifier_error_category") is None
        assert saved["reward_out_of_range_info"]["value"] == 1.0002
        assert saved["reward_out_of_range_info"]["reward_range"] == [0.0, 1.0]
        cause = cause_of(saved)
        assert cause is None

    @pytest.mark.asyncio
    async def test_extra_fields_score_through_the_final_gate(
        self, verify_harness
    ) -> None:
        rewards, verifier_error, diagnostic = await self._verify(
            verify_harness, dict(COOPERBENCH)
        )
        assert (rewards, verifier_error, diagnostic) == (COOPERBENCH, None, None)


# ---------------------------------------------------------------------------
# config.json: lenient mode and the models that answered
# ---------------------------------------------------------------------------


def _config(tmp_path: Path) -> dict:
    from datetime import datetime

    from benchflow.rollout._results import _write_config
    from benchflow.skill_policy import resolve_task_skill_policy

    _write_config(
        tmp_path,
        task_path=tmp_path / "task",
        agent="claude-agent-acp",
        model="sonnet",
        environment="docker",
        skill_policy=resolve_task_skill_policy(
            task_path=tmp_path / "task",
            skill_mode="no-skill",
            runtime_skills_dir=None,
            declared_sandbox_skills_dir=None,
        ),
        sandbox_user="agent",
        context_root=None,
        sandbox_locked_paths=None,
        sandbox_setup_timeout=120,
        skip_agent_install=True,
        timeout=60,
        started_at=datetime.now(),
        agent_env={},
    )
    return json.loads((tmp_path / "config.json").read_text())


class TestConfigRecordsLenientMode:
    def test_strict_by_default(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("BENCHFLOW_REWARD_LENIENT", raising=False)
        assert _config(tmp_path).get("reward_lenient") is False

    def test_lenient_when_the_operator_opts_in(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", "1")
        assert _config(tmp_path).get("reward_lenient") is True


def _callback_log(*responses: dict) -> str:
    lines = []
    for response in responses:
        lines.append(
            {
                "event": "success",
                "request_model": "sonnet",
                "provider_model": "anthropic/sonnet",
                "request": {"method": "POST", "path": "/v1/messages", "body": {}},
                "response": response,
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "start_time": "2026-10-15T00:00:00+00:00",
                "end_time": "2026-10-15T00:00:01+00:00",
            }
        )
    return "".join(json.dumps(line) + "\n" for line in lines)


class TestReportedModels:
    def test_the_proxy_log_names_the_models_the_provider_reported(self) -> None:
        from benchflow.providers.litellm_logging import (
            trajectory_from_litellm_callback_log,
        )

        text = _callback_log(
            {"model": "claude-sonnet-5-20261001", "content": []},
            {"content": []},  # no model: the exchange body falls back, we do not
            {"model": "claude-sonnet-5-20261001", "content": []},
            {"model": "claude-haiku-4-5-20251001", "content": []},
            {"model": "bad\nmodel\x1b[31m", "content": []},
        )
        trajectory = trajectory_from_litellm_callback_log(
            text, session_id="s", agent_name="claude-agent-acp"
        )
        assert trajectory.metadata.get("reported_models") == [
            "claude-sonnet-5-20261001",
            "claude-haiku-4-5-20251001",
        ]

    @pytest.mark.asyncio
    async def test_cleanup_records_the_proxys_models_in_config_json(
        self, tmp_path, monkeypatch
    ) -> None:
        from benchflow.providers.litellm_logging import (
            trajectory_from_litellm_callback_log,
        )
        from tests.test_session_cost import _unpriced_claude_rollout

        monkeypatch.setattr(
            "benchflow.rollout_planes.stop_provider_runtime", AsyncMock()
        )
        rollout, trial = _unpriced_claude_rollout(
            tmp_path, monkeypatch, tmp_path / "home", {}
        )
        (trial / "config.json").write_text(json.dumps({"model": "sonnet"}))
        trajectory = trajectory_from_litellm_callback_log(
            _callback_log({"model": "claude-sonnet-5-20261001", "content": []}),
            session_id="s",
            agent_name="claude-agent-acp",
        )
        rollout._usage_runtime = SimpleNamespace(
            kind="litellm",
            server=SimpleNamespace(trajectory=trajectory),
            backend_model="sonnet",
        )

        await rollout.cleanup()

        config = json.loads((trial / "config.json").read_text())
        assert config["model"] == "sonnet"
        assert config.get("reported_models") == ["claude-sonnet-5-20261001"]
        assert config.get("reported_models_source") == "provider_response"

    @pytest.mark.asyncio
    async def test_cleanup_records_the_session_logs_models_in_config_json(
        self, tmp_path, monkeypatch
    ) -> None:
        # A subscription run (G-28): the run asked for the alias ``sonnet``;
        # only Claude Code's session log names the model that answered.
        from tests.test_session_cost import (
            _log,
            _response,
            _state,
            _unpriced_claude_rollout,
        )

        home = tmp_path / "home"
        synthetic = {
            "type": "assistant",
            "isApiErrorMessage": True,
            "message": {"id": "e1", "model": "<synthetic>", "usage": {}},
        }
        _log(
            home / ".claude" / "projects",
            [
                _response("msg_1", model="claude-sonnet-5[1m]"),
                synthetic,
                _response("msg_2", model="claude-sonnet-5"),
                _state(0.00125, input_tokens=1000, output_tokens=50),
            ],
        )
        rollout, trial = _unpriced_claude_rollout(tmp_path, monkeypatch, home, {})
        (trial / "config.json").write_text(json.dumps({"model": "sonnet"}))

        await rollout.cleanup()

        config = json.loads((trial / "config.json").read_text())
        assert config["model"] == "sonnet"
        assert config.get("reported_models") == ["claude-sonnet-5"]
        assert config.get("reported_models_source") == "agent_session_log"

    @pytest.mark.asyncio
    async def test_no_answer_records_no_model(self, tmp_path, monkeypatch) -> None:
        from tests.test_session_cost import _unpriced_claude_rollout

        rollout, trial = _unpriced_claude_rollout(
            tmp_path, monkeypatch, tmp_path / "home", {}
        )
        (trial / "config.json").write_text(json.dumps({"model": "sonnet"}))

        await rollout.cleanup()

        config = json.loads((trial / "config.json").read_text())
        assert "reported_models" not in config

    def test_session_log_reader_is_bounded_to_model_ids(self, tmp_path) -> None:
        from benchflow._utils.reported_models import (
            MAX_REPORTED_MODELS,
            session_log_models,
        )

        project = tmp_path / ".claude" / "projects" / "-app"
        project.mkdir(parents=True)
        lines = [
            {"type": "assistant", "message": {"model": f"model-{i}"}}
            for i in range(MAX_REPORTED_MODELS + 3)
        ]
        lines.insert(0, {"type": "assistant", "message": {"model": "x" * 300}})
        lines.insert(0, {"type": "user", "message": {"model": "not-an-answer"}})
        (project / "s.jsonl").write_text(
            "not json\n" + "".join(json.dumps(line) + "\n" for line in lines)
        )
        models = session_log_models(tmp_path)
        assert models == [f"model-{i}" for i in range(MAX_REPORTED_MODELS)]


# ---------------------------------------------------------------------------
# A scored timeout is scored, in every reader
# ---------------------------------------------------------------------------

SCORED_TIMEOUT = {
    "task_name": "hello",
    "rollout_name": "hello__1",
    "agent": "claude-agent-acp",
    "agent_name": "claude-agent-acp",
    "model": "claude-haiku-4-5-20251001",
    "rewards": {"reward": 1.0},
    "error": "Agent timed out after 45s (prompt exceeded wall-clock budget)",
    "error_category": "timeout",
    "verifier_error": None,
    "n_tool_calls": 2,
    "n_prompts": 1,
    "started_at": "2026-10-15 00:00:00",
    "finished_at": "2026-10-15 00:01:00",
    "agent_result": {
        "n_input_tokens": 2000,
        "n_output_tokens": 100,
        "total_tokens": 2100,
        "cost_usd": 0.0025,
        "usage_source": "provider_response",
    },
}
ERRORED = {
    **SCORED_TIMEOUT,
    "task_name": "crash",
    "rollout_name": "crash__1",
    "rewards": None,
    "error": "ACP error -32603: Internal error",
    "error_category": "acp_error",
}


class TestScoredTimeoutIsScored:
    def test_summary_usage_counts_its_tokens_and_cost(self) -> None:
        from benchflow._utils.evaluation_results import usage_summary

        usage = usage_summary({"hello": SCORED_TIMEOUT, "crash": ERRORED})
        assert usage["total_tokens"] == 2100
        assert usage["total_cost_usd"] == pytest.approx(0.0025)
        assert usage["telemetry_coverage"] == 1.0

    def test_score_summary_and_solve_rates(self) -> None:
        from benchflow._utils.evaluation_results import solve_rate_summary
        from benchflow._utils.scoring import score_summary_fields

        rows = [SCORED_TIMEOUT, ERRORED]
        summary = score_summary_fields(rows)
        assert (summary["passed"], summary["errored"]) == (1, 1)
        assert summary["score_excl_errors_ratio"] == 1.0
        rates = solve_rate_summary({"hello": SCORED_TIMEOUT, "crash": ERRORED})
        assert rates["solve_rates"]["unscored"] == 1

    def test_job_readers(self, tmp_path: Path) -> None:
        import benchflow as bf
        from benchflow.failures import cause_of
        from benchflow.metrics import collect_metrics

        job = tmp_path / "job"
        for row in (SCORED_TIMEOUT, ERRORED):
            trial = job / row["rollout_name"]
            trial.mkdir(parents=True)
            (trial / "result.json").write_text(json.dumps(row))
        d = bf.load_job(job).denominators()
        assert (d.scored, d.unscored, d.passed) == (1, 1, 1)
        assert d.execution_errors == 2  # the timeout still shows as one
        metrics = collect_metrics(job)
        assert (metrics.passed, metrics.errored) == (1, 1)
        assert cause_of(SCORED_TIMEOUT) is None

    def test_viewer_status(self) -> None:
        from benchflow.trajectories.viewer.payload import _run_status

        status = _run_status(SCORED_TIMEOUT, 1.0)
        assert (status.execution, status.assessment) == ("timed_out", "scored")


# ---------------------------------------------------------------------------
# bench eval run with no agent named
# ---------------------------------------------------------------------------


def _task_dir(tmp_path: Path) -> Path:
    tasks = tmp_path / "tasks"
    task = tasks / "hello"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.0"\n')
    (task / "instruction.md").write_text("Say hi.")
    return tasks


class TestNoAgentNamed:
    def _run(self, tmp_path: Path, *extra: str):
        from benchflow.cli.main import app

        return CliRunner().invoke(
            app,
            [
                "eval",
                "run",
                "--tasks-dir",
                str(_task_dir(tmp_path)),
                "--jobs-dir",
                str(tmp_path / "jobs"),
                "--dry-run",
                *extra,
            ],
        )

    def test_warns_once_naming_the_default_agent_and_model(self, tmp_path) -> None:
        res = self._run(tmp_path)
        assert res.exit_code == 0, res.output
        assert res.output.count("No agent named") == 1, res.output
        assert "claude-agent-acp" in res.output
        assert "claude-haiku-4-5-20251001" in res.output

    def test_names_the_model_given(self, tmp_path) -> None:
        res = self._run(tmp_path, "--model", "claude-sonnet-4-5")
        assert res.output.count("No agent named") == 1, res.output
        assert "claude-agent-acp with claude-sonnet-4-5" in res.output

    @pytest.mark.parametrize("agent", ["oracle", "claude-agent-acp"])
    def test_a_named_agent_is_not_warned_about(self, tmp_path, agent) -> None:
        res = self._run(tmp_path, "--agent", agent)
        assert res.exit_code == 0, res.output
        assert "No agent named" not in res.output

    def test_the_job_records_it(self, tmp_path) -> None:
        from benchflow.eval_plan import EvalCreateRequest, build_eval_plan
        from benchflow.evaluation import _config_to_record

        tasks = _task_dir(tmp_path)
        plan = build_eval_plan(EvalCreateRequest(tasks_dir=tasks))
        record = _config_to_record(plan.make_eval_config())
        assert record.get("agent_defaulted") is True
        assert record["agent"] == "claude-agent-acp"
        assert record["model"] == "claude-haiku-4-5-20251001"
        named = build_eval_plan(EvalCreateRequest(tasks_dir=tasks, agent="oracle"))
        assert (
            _config_to_record(named.make_eval_config()).get("agent_defaulted") is False
        )

    def test_a_config_file_without_an_agent_warns_and_records(
        self, tmp_path, caplog
    ) -> None:
        from benchflow.evaluation import Evaluation

        tasks = _task_dir(tmp_path)
        config = tmp_path / "run.yaml"
        config.write_text(f"tasks_dir: {tasks}\njobs_dir: {tmp_path / 'jobs'}\n")
        with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
            evaluation = Evaluation.from_yaml(config, preflight=False)
        assert getattr(evaluation._config, "agent_defaulted", None) is True
        warned = [r for r in caplog.records if "No agent named" in r.getMessage()]
        assert len(warned) == 1
        assert "claude-agent-acp" in warned[0].getMessage()

    def _run_config(self, tmp_path: Path, caplog, *extra: str):
        """``bench eval run --config`` on a YAML with no ``agent`` key.

        The job's ``run`` is stubbed (it records the config it got), so this
        reaches ``_run_config_file_eval``'s overrides, which ``--dry-run``
        returns before.
        """
        from benchflow.cli.main import app
        from benchflow.evaluation import Evaluation

        tasks = _task_dir(tmp_path)
        config = tmp_path / "run.yaml"
        config.write_text(f"tasks_dir: {tasks}\njobs_dir: {tmp_path / 'jobs'}\n")
        seen: dict = {}

        async def fake_run(self: Evaluation):
            seen["config"] = self._config
            return SimpleNamespace(
                passed=1, total=1, score=1.0, errored=0, verifier_errored=0
            )

        with (
            patch.object(Evaluation, "run", new=fake_run),
            caplog.at_level(logging.WARNING, logger="benchflow.evaluation"),
        ):
            res = CliRunner().invoke(
                app, ["eval", "run", "--config", str(config), *extra]
            )
        logged = [r for r in caplog.records if "No agent named" in r.getMessage()]
        return res, seen.get("config"), logged

    def test_a_config_file_with_agent_on_the_command_line_is_not_defaulted(
        self, tmp_path, caplog
    ) -> None:
        """V4-R1-A-1: --agent names the agent even when the YAML does not."""
        res, config, logged = self._run_config(tmp_path, caplog, "--agent", "oracle")
        assert res.exit_code == 0, res.output
        assert "No agent named" not in res.output
        assert logged == []
        assert config.agent == "oracle"
        assert config.agent_defaulted is False

    def test_a_config_file_without_an_agent_warns_once_with_the_model_run(
        self, tmp_path, caplog
    ) -> None:
        """V4-R1-A-1: the warning names --model, the model that runs."""
        res, config, logged = self._run_config(
            tmp_path, caplog, "--model", "claude-sonnet-4-5"
        )
        assert res.exit_code == 0, res.output
        assert logged == []
        assert res.output.count("No agent named") == 1, res.output
        assert "claude-agent-acp with claude-sonnet-4-5" in res.output
        assert config.agent_defaulted is True
        assert config.model == "claude-sonnet-4-5"

    def test_from_yaml_can_leave_the_warning_to_its_caller(
        self, tmp_path, caplog
    ) -> None:
        from benchflow.evaluation import Evaluation

        tasks = _task_dir(tmp_path)
        config = tmp_path / "run.yaml"
        config.write_text(f"tasks_dir: {tasks}\njobs_dir: {tmp_path / 'jobs'}\n")
        with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
            evaluation = Evaluation.from_yaml(
                config, preflight=False, warn_default_agent=False
            )
        assert evaluation._config.agent_defaulted is True
        assert not [r for r in caplog.records if "No agent named" in r.getMessage()]


# ---------------------------------------------------------------------------
# Codex findings fix round
# ---------------------------------------------------------------------------


class TestKeptFieldsAreFinite:
    """A kept field holds no NaN or infinity (the trial's JSON
    records refuse them), so it fails at verification, not at publication."""

    @pytest.mark.parametrize(
        "text",
        [
            '{"reward": 1, "answer": {"confidence": NaN}}',
            '{"reward": 1.0, "features": [Infinity]}',
            '{"reward": 1.0, "features": [1, {"deep": [-Infinity]}]}',
            '{"reward": 1.0, "answer": {"confidence": 1e400}}',
            '{"reward": 1.0, "details": {"x": NaN}}',
        ],
    )
    def test_reward_json_with_a_non_finite_kept_field_is_a_verifier_error(
        self, tmp_path: Path, text: str
    ) -> None:
        (tmp_path / "reward.json").write_text(text)
        with pytest.raises(VerifierOutputParseError, match="non-finite"):
            _verifier(tmp_path)._parse_reward_json()
        with pytest.raises(ValueError, match="non-finite"):
            _ensure_canonical_rewards(json.loads(text))

    def test_lenient_mode_drops_a_non_finite_kept_field(self) -> None:
        with pytest.warns(UserWarning):
            kept = validate_reward_map(
                {"reward": 1.0, "answer": {"c": float("nan")}, "ok": True},
                lenient=True,
            )
        assert kept == {"reward": 1.0, "ok": True}

    def test_finite_kept_fields_still_pass(self) -> None:
        payload = {"reward": 1.0, "answer": {"c": 0.5, "n": [1, 2]}, "ok": True}
        assert validate_reward_map(payload) == payload


class TestStructuredMetadataIsFinite:
    """Guards CONF-V4-1 at 1008b372: structured metadata must be writable."""

    @pytest.mark.parametrize("key", ["rubric", "aggregate"])
    @pytest.mark.parametrize(
        "number", ["NaN", "Infinity", "-Infinity", "1e400", "9" * 400]
    )
    def test_strict_rejects_non_finite_metadata(self, tmp_path, key, number):
        metadata = json.loads('{"nested": [{"confidence": ' + number + "}]}")
        value = (
            [{"score": 1, "evidence": metadata}]
            if key == "rubric"
            else {"method": "mean", "metadata": metadata}
        )
        payload = {"reward": 1, key: value}
        (tmp_path / "reward.json").write_text(json.dumps(payload))
        with pytest.raises(VerifierOutputParseError, match="non-finite"):
            _verifier(tmp_path)._parse_reward_json()
        with pytest.raises(ValueError, match="non-finite"):
            _ensure_canonical_rewards(payload)

    @pytest.mark.parametrize("key", ["rubric", "aggregate"])
    def test_lenient_drops_the_whole_bad_structured_field(
        self, tmp_path, monkeypatch, key
    ):
        value = (
            [{"score": 1, "evidence": {"confidence": float("nan")}}]
            if key == "rubric"
            else {"method": "mean", "metadata": {"x": float("inf")}}
        )
        payload = {"reward": 1, key: value, "answer": {"confidence": 0.5}}
        expected = {"reward": 1, "answer": {"confidence": 0.5}}
        with pytest.warns(UserWarning, match=key):
            assert validate_reward_map(payload, lenient=True) == expected
        monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", "1")
        (tmp_path / "reward.json").write_text(json.dumps(payload))
        with pytest.warns(UserWarning, match=key):
            assert _verifier(tmp_path)._parse_reward_json() == expected
        json.dumps(expected, allow_nan=False)

    def test_finite_metadata_is_preserved(self):
        payload = {
            "reward": 1,
            "rubric": [{"score": 1, "evidence": {"confidence": 0.5}}],
            "aggregate": {"method": "mean", "metadata": {"x": [2, True, None]}},
        }
        assert validate_reward_map(payload) == payload


class TestLenientNonFiniteReward:
    """An explicit NaN or infinite ``reward`` is not replaced by a
    ``score`` alias in lenient mode."""

    @pytest.mark.parametrize(
        "text",
        [
            '{"reward": NaN, "score": 1}',
            '{"reward": Infinity, "score": 1}',
            '{"reward": -Infinity, "score": 1}',
            '{"reward": 1e400, "score": 1}',
        ],
    )
    def test_lenient_refuses_a_non_finite_reward(
        self, tmp_path: Path, monkeypatch, text: str
    ) -> None:
        monkeypatch.setenv("BENCHFLOW_REWARD_LENIENT", "1")
        (tmp_path / "reward.json").write_text(text)
        with pytest.raises(VerifierOutputParseError, match="not a finite number"):
            _verifier(tmp_path)._parse_reward_json()
        with pytest.raises(ValueError, match="not a finite number"):
            validate_reward_map(json.loads(text), lenient=True)

    def test_lenient_still_reads_an_alias_for_a_non_numeric_reward(self) -> None:
        with pytest.warns(UserWarning):
            kept = validate_reward_map({"reward": "pass", "score": 1}, lenient=True)
        assert kept == {"reward": 1}


def _role_rollout(tmp_path: Path, runtimes: list):
    """A rollout whose fake planes hand out ``runtimes`` one per connect."""
    from benchflow.rollout import Rollout, RolloutConfig

    rollout = Rollout(
        RolloutConfig(
            task_path=tmp_path / "task",
            agent="claude-agent-acp",
            model="claude-sonnet-4-6",
            environment="docker",
            agent_env={"ANTHROPIC_API_KEY": "sk-ant"},
        )
    )
    rollout._rollout_dir = tmp_path
    rollout._rollout_name = "rollout"
    rollout._agent_cwd = "/workspace"
    rollout._env = SimpleNamespace()
    rollout._agent_cfg = SimpleNamespace()
    rollout._reapply_ask_user_handler = lambda: None
    rollout._attach_trajectory_writer = lambda _rollout_dir: None
    queue = list(runtimes)

    async def fake_litellm(**kwargs):
        return dict(kwargs["agent_env"]), queue.pop(0)

    async def fake_connect_acp(**kwargs):
        # C26 snapshots synchronous session state while closing the prior role.
        session = ACPSession("dummy-session")
        return (AsyncMock(session=session), session, MagicMock(), "claude-agent-acp")

    rollout._planes = SimpleNamespace(
        agent_launch=lambda agent, disallow_web_tools, disallow_hosted_search=False: (
            agent
        ),
        resolve_agent_env=lambda agent, model, env: dict(env or {}),
        ensure_litellm_runtime=fake_litellm,
        install_agent=AsyncMock(return_value=SimpleNamespace()),
        write_credential_files=AsyncMock(),
        upload_subscription_auth=AsyncMock(),
        apply_web_tool_policy=AsyncMock(),
        # A role transition stops the previous agent (C26).
        quiesce_agent=AsyncMock(),
        connect_acp=fake_connect_acp,
    )
    return rollout


def _runtime_reporting(*models: str):
    return SimpleNamespace(
        kind="litellm",
        server=SimpleNamespace(
            trajectory=SimpleNamespace(metadata={"reported_models": list(models)})
        ),
    )


class TestReportedModelsAcrossRoles:
    """A role on another model replaces the proxy; the replaced
    proxy's reported models are kept."""

    @pytest.mark.asyncio
    async def test_a_replaced_proxy_keeps_its_models(self, tmp_path: Path) -> None:
        from benchflow.rollout import Role

        first, second = _runtime_reporting("model-a-1"), _runtime_reporting("model-b-1")
        rollout = _role_rollout(tmp_path, [first, second])
        await rollout.connect_as(Role(name="a", agent="claude-agent-acp", model="a"))
        await rollout.connect_as(Role(name="b", agent="claude-agent-acp", model="b"))
        assert rollout._usage_runtime is second
        assert rollout._proxy_reported_models == ["model-a-1"]

    @pytest.mark.asyncio
    async def test_a_final_unproxied_role_keeps_the_earlier_models(
        self, tmp_path: Path
    ) -> None:
        from benchflow.rollout import Role

        rollout = _role_rollout(tmp_path, [_runtime_reporting("model-a-1"), None])
        await rollout.connect_as(Role(name="a", agent="claude-agent-acp", model="a"))
        await rollout.connect_as(Role(name="b", agent="claude-agent-acp", model="b"))
        assert rollout._usage_runtime is None
        assert rollout._proxy_reported_models == ["model-a-1"]

    @pytest.mark.asyncio
    async def test_a_reused_proxy_is_not_counted_twice(self, tmp_path: Path) -> None:
        from benchflow.rollout import Role

        same = _runtime_reporting("model-a-1")
        rollout = _role_rollout(tmp_path, [same, same])
        await rollout.connect_as(Role(name="a", agent="claude-agent-acp", model="a"))
        await rollout.connect_as(Role(name="a2", agent="claude-agent-acp", model="a"))
        assert rollout._proxy_reported_models == []

    @pytest.mark.asyncio
    async def test_cleanup_merges_the_last_proxy_with_the_replaced_ones(
        self, tmp_path, monkeypatch
    ) -> None:
        from benchflow.providers.litellm_logging import (
            trajectory_from_litellm_callback_log,
        )
        from tests.test_session_cost import _unpriced_claude_rollout

        monkeypatch.setattr(
            "benchflow.rollout_planes.stop_provider_runtime", AsyncMock()
        )
        rollout, trial = _unpriced_claude_rollout(
            tmp_path, monkeypatch, tmp_path / "home", {}
        )
        (trial / "config.json").write_text(json.dumps({"model": "sonnet"}))
        rollout._proxy_reported_models = ["model-a-1"]
        trajectory = trajectory_from_litellm_callback_log(
            _callback_log({"model": "model-b-1", "content": []}),
            session_id="s",
            agent_name="claude-agent-acp",
        )
        rollout._usage_runtime = SimpleNamespace(
            kind="litellm",
            server=SimpleNamespace(trajectory=trajectory),
            backend_model="sonnet",
        )

        await rollout.cleanup()

        config = json.loads((trial / "config.json").read_text())
        assert config.get("reported_models") == ["model-a-1", "model-b-1"]
        assert config.get("reported_models_source") == "provider_response"


class TestLegacyYamlNoAgentNamed:
    """The legacy (agents + datasets) format records and warns
    about the default agent as the native format does, for an agents entry
    with no name. A legacy (task.toml) file that names no agent at all, no
    `agents:` or `agents: []`, runs the oracle with one note: a run that
    needs no model never starts one."""

    def _legacy(self, tmp_path: Path, agents: str) -> Path:
        tasks = _task_dir(tmp_path)
        config = tmp_path / "legacy.yaml"
        config.write_text(
            f"{agents}datasets:\n  - path: {tasks}\njobs_dir: {tmp_path / 'jobs'}\n"
        )
        return config

    def test_legacy_config_with_a_nameless_agent_warns_and_records(
        self, tmp_path, caplog
    ) -> None:
        from benchflow.evaluation import Evaluation

        config = self._legacy(tmp_path, "agents:\n  - {}\n")
        with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
            evaluation = Evaluation.from_yaml(config, preflight=False)
        assert evaluation._config.agent == "claude-agent-acp"
        assert evaluation._config.agent_defaulted is True
        warned = [r for r in caplog.records if "No agent named" in r.getMessage()]
        assert len(warned) == 1

    @pytest.mark.parametrize("agents", ["", "agents: []\n"])
    def test_legacy_config_naming_no_agent_runs_the_oracle(
        self, tmp_path, caplog, agents: str
    ) -> None:
        from benchflow.evaluation import Evaluation

        config = self._legacy(tmp_path, agents)
        with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
            evaluation = Evaluation.from_yaml(config, preflight=False)
        assert evaluation._config.agent == "oracle"
        assert evaluation._config.agent_defaulted is False
        messages = [r.getMessage() for r in caplog.records]
        assert not [m for m in messages if "No agent named" in m]
        assert len([m for m in messages if "names no agent" in m]) == 1

    def test_legacy_config_naming_an_agent_is_not_defaulted(
        self, tmp_path, caplog
    ) -> None:
        from benchflow.evaluation import Evaluation

        config = self._legacy(tmp_path, "agents:\n  - name: oracle\n")
        with caplog.at_level(logging.WARNING, logger="benchflow.evaluation"):
            evaluation = Evaluation.from_yaml(config, preflight=False)
        assert evaluation._config.agent_defaulted is False
        assert not [r for r in caplog.records if "No agent named" in r.getMessage()]

    def test_legacy_config_through_the_cli_notes_the_oracle_once(
        self, tmp_path, caplog
    ) -> None:
        from benchflow.cli.main import app
        from benchflow.evaluation import Evaluation

        config = self._legacy(tmp_path, "")
        seen: dict = {}

        async def fake_run(self: Evaluation):
            seen["config"] = self._config
            return SimpleNamespace(
                passed=1, total=1, score=1.0, errored=0, verifier_errored=0
            )

        with (
            patch.object(Evaluation, "run", new=fake_run),
            caplog.at_level(logging.WARNING, logger="benchflow.evaluation"),
        ):
            res = CliRunner().invoke(app, ["eval", "run", "--config", str(config)])
        assert res.exit_code == 0, res.output
        assert "No agent named" not in res.output
        assert not [r for r in caplog.records if "names no agent" in r.getMessage()]
        assert res.output.count("names no agent, so the oracle runs") == 1, res.output
        assert (seen["config"].agent, seen["config"].agent_defaulted) == (
            "oracle",
            False,
        )
