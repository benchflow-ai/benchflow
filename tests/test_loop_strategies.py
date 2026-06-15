"""Tests for harness-level loop strategies — spec parsing, feedback filter, VerifyRetryUser."""

from __future__ import annotations

import pytest

from benchflow.contracts import RoundResult
from benchflow.loop_strategies import (
    FeedbackLevel,
    LoopStrategySpec,
    SelfReviewUser,
    VerifyRetryUser,
    build_loop_user,
    collect_loop_metadata,
    filter_verifier_feedback,
    loop_block,
    parse_loop_strategy_spec,
)

SECRET = "expected 'hunter2-ground-truth'"
PYTEST_OUTPUT = f"""\
============================= test session starts ==============================
collected 3 items

tests/test_alpha.py::test_one PASSED
tests/test_alpha.py::test_two FAILED
tests/test_beta.py::test_three FAILED

=================================== FAILURES ===================================
FAILED tests/test_alpha.py::test_two - AssertionError: {SECRET}
FAILED tests/test_beta.py::test_three - ValueError: bad input
========================= 2 failed, 1 passed in 0.12s =========================
"""


class TestParseLoopStrategySpec:
    def test_full_spec(self):
        spec = parse_loop_strategy_spec("verify-retry:k=3,feedback=names")
        assert spec.name == "verify-retry"
        assert spec.params == {"k": 3, "feedback": "names"}

    def test_bare_verify_retry_takes_defaults(self):
        spec = parse_loop_strategy_spec("verify-retry")
        assert spec.params == {"k": 3, "feedback": "names"}

    def test_bare_single_shot(self):
        spec = parse_loop_strategy_spec("single-shot")
        assert spec.name == "single-shot"
        assert spec.params == {}

    def test_whitespace_tolerated(self):
        spec = parse_loop_strategy_spec(" verify-retry: k=2 , feedback=raw ")
        assert spec.params == {"k": 2, "feedback": "raw"}

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "not-a-strategy",
            "verify-retry:k",
            "verify-retry:k=",
            "verify-retry:k=zero",
            "verify-retry:k=0",
            "verify-retry:k=-1",
            "verify-retry:feedback=verbose",
            "verify-retry:retries=3",
            "verify-retry:k=2,k=3",
            "single-shot:k=3",
            "self-review:k=0",
            "self-review:feedback=names",
        ],
    )
    def test_bad_specs_raise(self, bad: str):
        with pytest.raises(ValueError):
            parse_loop_strategy_spec(bad)

    def test_mapping_round_trip(self):
        spec = parse_loop_strategy_spec("verify-retry:k=2,feedback=raw")
        assert LoopStrategySpec.from_mapping(spec.to_mapping()) == spec

    def test_single_shot_mapping_omits_params(self):
        assert parse_loop_strategy_spec("single-shot").to_mapping() == {
            "strategy": "single-shot"
        }

    def test_from_mapping_rejects_missing_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            LoopStrategySpec.from_mapping({"params": {"k": 3}})

    def test_from_mapping_validates_params(self):
        with pytest.raises(ValueError, match="positive integer"):
            LoopStrategySpec.from_mapping(
                {"strategy": "verify-retry", "params": {"k": 0}}
            )


class TestFilterVerifierFeedback:
    def test_none_is_empty(self):
        assert filter_verifier_feedback(PYTEST_OUTPUT, FeedbackLevel.NONE) == ""

    def test_counts_has_summary_only(self):
        result = filter_verifier_feedback(PYTEST_OUTPUT, FeedbackLevel.COUNTS)
        assert "2 failed, 1 passed" in result
        assert "test_two" not in result
        assert SECRET not in result

    def test_names_has_node_ids_only(self):
        result = filter_verifier_feedback(PYTEST_OUTPUT, FeedbackLevel.NAMES)
        assert "tests/test_alpha.py::test_two" in result
        assert "tests/test_beta.py::test_three" in result
        assert SECRET not in result
        assert "bad input" not in result

    def test_names_fails_closed_on_unrecognized_output(self):
        assert filter_verifier_feedback("the answer is 42", FeedbackLevel.NAMES) == ""

    def test_names_strips_parametrized_id_suffixes(self):
        """Parametrized ids embed the expected values NAMES exists to
        withhold — strip them and dedupe what remains."""
        out = (
            "FAILED tests/test_alpha.py::test_answer[expected-42] - AssertionError\n"
            "FAILED tests/test_alpha.py::test_answer[expected-43] - AssertionError"
        )
        result = filter_verifier_feedback(out, FeedbackLevel.NAMES)
        assert result == "tests/test_alpha.py::test_answer"

    def test_names_ignores_captured_log_error_lines(self):
        """A captured-log line that happens to start with ERROR is not a
        pytest result line — its payload must never reach the agent."""
        out = "ERROR    root:app.py:7 the answer is 42"
        assert filter_verifier_feedback(out, FeedbackLevel.NAMES) == ""
        assert filter_verifier_feedback(out, FeedbackLevel.COUNTS) == ""

    def test_names_keeps_collection_error_file_tokens(self):
        out = "ERROR tests/test_broken.py - ImportError: nope"
        result = filter_verifier_feedback(out, FeedbackLevel.NAMES)
        assert result == "tests/test_broken.py"

    def test_raw_truncates_to_max_chars(self):
        result = filter_verifier_feedback(PYTEST_OUTPUT, "raw", max_chars=50)
        assert len(result) == 50
        assert PYTEST_OUTPUT.startswith(result)

    def test_empty_output(self):
        for level in FeedbackLevel:
            assert filter_verifier_feedback(None, level) == ""
            assert filter_verifier_feedback("   ", level) == ""


def _failing_round(round: int) -> RoundResult:
    return RoundResult(
        round=round,
        rewards={"reward": 0.0},
        verifier_output=PYTEST_OUTPUT,
    )


class TestVerifyRetryUser:
    @pytest.mark.asyncio
    async def test_passes_first_try_stops_after_one_round(self):
        user = VerifyRetryUser(k=3)
        assert await user.run(0, "Fix the bug") == "Fix the bug"
        passing = RoundResult(round=0, rewards={"reward": 1.0})
        assert await user.run(1, "Fix the bug", passing) is None

    @pytest.mark.asyncio
    async def test_keeps_retrying_while_failing_engine_owns_budget(self):
        """The user never self-terminates on a failing reward — the retry
        budget belongs to the engine: build_loop_user caps the loop at
        max_user_rounds=k+1 and the engine derives stop_reason from the
        round log."""
        k = 2
        user = VerifyRetryUser(k=k, feedback=FeedbackLevel.NAMES)
        assert await user.run(0, "Fix the bug") == "Fix the bug"
        for n in range(1, k + 3):
            prompt = await user.run(n, "Fix the bug", _failing_round(n - 1))
            assert prompt is not None

    @pytest.mark.asyncio
    async def test_retry_prompts_carry_filtered_feedback_only(self):
        user = VerifyRetryUser(k=2, feedback=FeedbackLevel.NAMES)
        await user.run(0, "Fix the bug")
        prompt = await user.run(1, "Fix the bug", _failing_round(0))
        assert prompt is not None
        assert "tests/test_alpha.py::test_two" in prompt
        assert SECRET not in prompt

    @pytest.mark.asyncio
    async def test_feedback_none_prompt_has_no_verifier_output(self):
        user = VerifyRetryUser(k=1, feedback=FeedbackLevel.NONE)
        await user.run(0, "Fix the bug")
        prompt = await user.run(1, "Fix the bug", _failing_round(0))
        assert prompt is not None
        assert "test_alpha" not in prompt
        assert SECRET not in prompt

    @pytest.mark.asyncio
    async def test_verifier_error_round_still_retries(self):
        user = VerifyRetryUser(k=2)
        await user.run(0, "Fix the bug")
        errored = RoundResult(round=0, rewards=None, verifier_error="boom")
        prompt = await user.run(1, "Fix the bug", errored)
        assert prompt is not None
        assert "boom" not in prompt

    def test_rejects_nonpositive_k(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            VerifyRetryUser(k=0)


class TestSelfReviewUser:
    @pytest.mark.asyncio
    async def test_round0_instruction_then_self_review(self):
        user = SelfReviewUser(k=2)
        assert await user.run(0, "Fix the bug") == "Fix the bug"
        prompt = await user.run(1, "Fix the bug", _failing_round(0))
        assert prompt is not None
        assert "review" in prompt.lower()

    @pytest.mark.asyncio
    async def test_self_review_never_leaks_verifier_output(self):
        """The whole point: self-review uses NO verifier signal, so neither the
        failing test names nor any ground truth may reach the agent."""
        user = SelfReviewUser(k=2)
        await user.run(0, "Fix the bug")
        prompt = await user.run(1, "Fix the bug", _failing_round(0))
        assert "tests/test_alpha.py::test_two" not in prompt
        assert SECRET not in prompt
        assert user.feedback_level is FeedbackLevel.NONE

    @pytest.mark.asyncio
    async def test_stops_when_soft_reward_passes(self):
        user = SelfReviewUser(k=3)
        passing = RoundResult(round=0, rewards={"reward": 1.0})
        assert await user.run(1, "Fix the bug", passing) is None

    def test_rejects_nonpositive_k(self):
        with pytest.raises(ValueError, match="k must be >= 1"):
            SelfReviewUser(k=0)


def _rounds(*rewards: float | None) -> list[dict]:
    return [
        {"round": i, "rewards": None if r is None else {"reward": r}}
        for i, r in enumerate(rewards)
    ]


class TestCollectLoopMetadata:
    def test_pass_on_final_round_is_passed_bar(self):
        """k=2 → budget 3; trajectory [0, 0, 1] must report passed_bar with
        first_pass_iteration=2, not fall through to max_iterations."""
        meta = collect_loop_metadata(
            VerifyRetryUser(k=2), _rounds(0.0, 0.0, 1.0), max_rounds=3, error=None
        )
        assert meta == {
            "iterations_run": 3,
            "stop_reason": "passed_bar",
            "reward_trajectory": [0.0, 0.0, 1.0],
            "tokens_trajectory": [None, None, None],
            "first_pass_iteration": 2,
            "tokens_to_pass": None,
        }

    def test_exhausted_budget_is_max_iterations(self):
        meta = collect_loop_metadata(
            VerifyRetryUser(k=1), _rounds(0.0, 0.0), max_rounds=2, error=None
        )
        assert meta["stop_reason"] == "max_iterations"
        assert meta["first_pass_iteration"] is None

    def test_error_wins_even_with_completed_rounds(self):
        meta = collect_loop_metadata(
            VerifyRetryUser(k=2), _rounds(0.0), max_rounds=3, error="Agent timed out"
        )
        assert meta == {
            "iterations_run": 1,
            "stop_reason": "error",
            "reward_trajectory": [0.0],
            "tokens_trajectory": [None],
            "first_pass_iteration": None,
            "tokens_to_pass": None,
        }

    def test_token_trajectory_and_cost_to_pass(self):
        """Cumulative tokens per round form the cost-curve x-axis; tokens_to_pass
        is the cumulative spend at the first passing round."""
        rounds = [
            {"round": 0, "rewards": {"reward": 0.0}, "tokens": 1000},
            {"round": 1, "rewards": {"reward": 1.0}, "tokens": 2500},
        ]
        meta = collect_loop_metadata(
            VerifyRetryUser(k=2), rounds, max_rounds=3, error=None
        )
        assert meta["tokens_trajectory"] == [1000, 2500]
        assert meta["first_pass_iteration"] == 1
        assert meta["tokens_to_pass"] == 2500

    def test_unscored_rounds_keep_none_in_trajectory(self):
        meta = collect_loop_metadata(
            VerifyRetryUser(k=2), _rounds(None, 1.0), max_rounds=3, error=None
        )
        assert meta["reward_trajectory"] == [None, 1.0]
        assert meta["stop_reason"] == "passed_bar"
        assert meta["first_pass_iteration"] == 1


class TestLoopBlock:
    def test_no_spec_stamps_single_shot(self):
        assert loop_block(None) == {"strategy": "single-shot"}

    def test_metadata_merges_into_spec_block(self):
        spec = parse_loop_strategy_spec("verify-retry:k=2")
        block = loop_block(spec, {"iterations_run": 1, "stop_reason": "error"})
        assert block["strategy"] == "verify-retry"
        assert block["params"] == {"k": 2, "feedback": "names"}
        assert block["iterations_run"] == 1
        assert block["stop_reason"] == "error"


class TestBuildLoopUser:
    def test_single_shot_is_none(self):
        assert build_loop_user(parse_loop_strategy_spec("single-shot")) is None

    def test_verify_retry_materializes_user_and_rounds(self):
        built = build_loop_user(parse_loop_strategy_spec("verify-retry:k=2"))
        assert built is not None
        user, max_rounds = built
        assert isinstance(user, VerifyRetryUser)
        assert user.k == 2
        assert user.feedback_level is FeedbackLevel.NAMES
        assert max_rounds == 3

    def test_self_review_materializes_user_and_rounds(self):
        built = build_loop_user(parse_loop_strategy_spec("self-review:k=2"))
        assert built is not None
        user, max_rounds = built
        assert isinstance(user, SelfReviewUser)
        assert user.k == 2
        assert user.feedback_level is FeedbackLevel.NONE
        assert max_rounds == 3

    def test_self_review_rejects_feedback_param(self):
        with pytest.raises(ValueError, match="unknown self-review param"):
            parse_loop_strategy_spec("self-review:feedback=names")

    def test_self_review_round_trips(self):
        spec = parse_loop_strategy_spec("self-review:k=4")
        assert LoopStrategySpec.from_mapping(spec.to_mapping()) == spec


class TestRolloutConfigLoopStrategy:
    def test_materializes_user_and_max_rounds(self, tmp_path):
        from benchflow.rollout import RolloutConfig

        config = RolloutConfig(
            task_path=tmp_path,
            loop_strategy="verify-retry:k=2,feedback=counts",
        )
        assert isinstance(config.user, VerifyRetryUser)
        assert config.user.k == 2
        assert config.max_user_rounds == 3
        assert config.loop_strategy_spec == LoopStrategySpec(
            "verify-retry", {"k": 2, "feedback": "counts"}
        )

    def test_single_shot_leaves_user_unset(self, tmp_path):
        from benchflow.rollout import RolloutConfig

        config = RolloutConfig(task_path=tmp_path, loop_strategy="single-shot")
        assert config.user is None
        assert config.loop_strategy_spec == LoopStrategySpec("single-shot")

    def test_explicit_user_conflicts(self, tmp_path):
        from benchflow.contracts import PassthroughUser
        from benchflow.rollout import RolloutConfig

        with pytest.raises(ValueError, match="mutually exclusive"):
            RolloutConfig(
                task_path=tmp_path,
                user=PassthroughUser(),
                loop_strategy="verify-retry:k=1",
            )

    def test_from_legacy_overrides_document_user_with_warning(
        self, tmp_path, monkeypatch, caplog
    ):
        from benchflow.contracts import PassthroughUser
        from benchflow.rollout import RolloutConfig
        from benchflow.rollout import _config as rollout_config

        monkeypatch.setattr(
            rollout_config,
            "_task_document_user_runtime",
            lambda *a, **kw: (PassthroughUser(), 4),
        )
        with caplog.at_level("WARNING"):
            config = RolloutConfig.from_legacy(
                task_path=tmp_path,
                loop_strategy="verify-retry:k=1",
            )
        assert isinstance(config.user, VerifyRetryUser)
        assert config.max_user_rounds == 2
        assert any("overrides the user declared" in r.message for r in caplog.records)

    def test_single_shot_suppresses_document_user_with_warning_on_both_paths(
        self, tmp_path, monkeypatch, caplog
    ):
        """Explicit single-shot forces a true single-shot run: the
        task-document user is suppressed (with a warning) identically for
        direct construction and from_legacy — the B2 semantics fork."""
        from benchflow.contracts import PassthroughUser
        from benchflow.rollout import RolloutConfig
        from benchflow.rollout import _config as rollout_config

        monkeypatch.setattr(
            rollout_config,
            "_task_document_user_runtime",
            lambda *a, **kw: (PassthroughUser(), 4),
        )
        with caplog.at_level("WARNING"):
            direct = RolloutConfig(task_path=tmp_path, loop_strategy="single-shot")
            legacy = RolloutConfig.from_legacy(
                task_path=tmp_path, loop_strategy="single-shot"
            )
        assert direct.user is None
        assert legacy.user is None
        suppressions = [
            r for r in caplog.records if "suppresses the user declared" in r.message
        ]
        assert len(suppressions) == 2

    def test_document_user_fallback_identical_on_both_paths(
        self, tmp_path, monkeypatch
    ):
        """No strategy → both construction paths adopt the document user
        and its round budget."""
        from benchflow.contracts import PassthroughUser
        from benchflow.rollout import RolloutConfig
        from benchflow.rollout import _config as rollout_config

        doc_user = PassthroughUser()
        monkeypatch.setattr(
            rollout_config,
            "_task_document_user_runtime",
            lambda *a, **kw: (doc_user, 4),
        )
        direct = RolloutConfig(task_path=tmp_path)
        legacy = RolloutConfig.from_legacy(task_path=tmp_path)
        assert direct.user is doc_user
        assert legacy.user is doc_user
        assert direct.max_user_rounds == legacy.max_user_rounds == 4

    def test_verify_retry_rejects_more_scene_turns_than_budget(self, tmp_path):
        """Multi-turn scenes beyond k+1 rounds fail at construction, before
        any sandbox is provisioned."""
        from benchflow._types import Role, Scene, Turn
        from benchflow.rollout import RolloutConfig

        scene = Scene(
            name="multi",
            roles=[Role(name="agent", agent="codex")],
            turns=[Turn(role="agent", prompt=f"t{i}") for i in range(3)],
        )
        with pytest.raises(ValueError, match="scene turns"):
            RolloutConfig(
                task_path=tmp_path,
                scenes=[scene],
                loop_strategy="verify-retry:k=1",
            )

    def test_dict_form_materializes_user(self, tmp_path):
        """Guard the confirmed-high: a to_mapping()-shaped dict (round-tripped
        --config YAML / SDK kwargs) must materialize, not silently drop the
        strategy and mislabel the run single-shot."""
        from benchflow.rollout import RolloutConfig

        config = RolloutConfig(
            task_path=tmp_path,
            loop_strategy={"strategy": "verify-retry", "params": {"k": 2}},
        )
        assert isinstance(config.user, VerifyRetryUser)
        assert config.loop_strategy_spec == LoopStrategySpec("verify-retry", {"k": 2})

    def test_to_mapping_round_trip_materializes(self, tmp_path):
        """The exact stamped shape (to_mapping) fed back in must rebuild."""
        from benchflow.rollout import RolloutConfig

        spec = LoopStrategySpec("verify-retry", {"k": 3, "feedback": "names"})
        config = RolloutConfig(task_path=tmp_path, loop_strategy=spec.to_mapping())
        assert config.loop_strategy_spec == spec
        assert isinstance(config.user, VerifyRetryUser)

    def test_bad_loop_strategy_type_raises(self, tmp_path):
        from benchflow.rollout import RolloutConfig

        with pytest.raises(
            ValueError, match="spec string, mapping, or LoopStrategySpec"
        ):
            RolloutConfig(task_path=tmp_path, loop_strategy=5)


class TestEvaluationConfigLoopStrategy:
    """The non-sharded --config / SDK path — the one the sharding guard never
    protects, so the dict coercion must live in EvaluationConfig too."""

    def test_dict_form_materializes(self):
        from benchflow.evaluation import EvaluationConfig

        cfg = EvaluationConfig(
            loop_strategy={"strategy": "verify-retry", "params": {"k": 2}}
        )
        assert cfg.loop_strategy == LoopStrategySpec("verify-retry", {"k": 2})

    def test_string_form_still_materializes(self):
        from benchflow.evaluation import EvaluationConfig

        cfg = EvaluationConfig(loop_strategy="verify-retry:k=2")
        assert cfg.loop_strategy == LoopStrategySpec("verify-retry", {"k": 2})

    def test_bad_type_raises(self):
        from benchflow.evaluation import EvaluationConfig

        with pytest.raises(
            ValueError, match="spec string, mapping, or LoopStrategySpec"
        ):
            EvaluationConfig(loop_strategy=5)


class TestResumeLoopMismatchWarning:
    def _job_dir(self, tmp_path, config_data: dict):
        import json

        task_dir = tmp_path / "job" / "task-a"
        task_dir.mkdir(parents=True)
        (task_dir / "config.json").write_text(json.dumps(config_data))
        return tmp_path / "job"

    def test_pre_feature_config_without_loop_key_warns(self, tmp_path, caplog):
        """config.json written before loop strategies existed has no "loop"
        key — it ran single-shot, so resuming with a strategy must warn."""
        from benchflow.evaluation import EvaluationConfig, _check_resume_mismatch

        job_dir = self._job_dir(tmp_path, {"agent": "claude-agent-acp"})
        config = EvaluationConfig(loop_strategy="verify-retry:k=2")
        with caplog.at_level("WARNING"):
            _check_resume_mismatch(job_dir, config)
        assert any(
            "loop_strategy" in r.message and "single-shot" in r.message
            for r in caplog.records
        )

    def test_matching_single_shot_does_not_warn(self, tmp_path, caplog):
        from benchflow.evaluation import EvaluationConfig, _check_resume_mismatch

        job_dir = self._job_dir(tmp_path, {"agent": "claude-agent-acp"})
        with caplog.at_level("WARNING"):
            _check_resume_mismatch(job_dir, EvaluationConfig())
        assert not [r for r in caplog.records if "loop_strategy" in r.message]

    def test_agent_mismatch_refuses_instead_of_blending_scores(self, tmp_path):
        # The reported bug: resuming a jobs_dir with a different agent silently
        # folded the prior agent's cached rollouts into the new run's summary,
        # publishing a blended Score: X/N. An agent mismatch must refuse (a
        # loop-only mismatch still just warns), preserving the prior results.
        from benchflow.evaluation import (
            EvaluationConfig,
            ResumeMismatchError,
            _check_resume_mismatch,
        )

        job_dir = self._job_dir(tmp_path, {"agent": "oracle"})
        config = EvaluationConfig(agent="codex-acp")
        with pytest.raises(ResumeMismatchError, match="codex-acp"):
            _check_resume_mismatch(job_dir, config)
