"""Regression tests for the oracle agent + DEFAULT_MODEL chokepoint.

Pins the fix from this branch (post-PR #173 follow-up):

  Layer 1 — restore `agent != "oracle"` guard in resolve_agent_env so the
            chokepoint defends against any caller that forwards a model.
  Layer 2 — keep `bench eval create` routed through the live CLI entry point.
  Layer 3 — funnel all CLI/YAML-loader sites through effective_model() so
            oracle gets honest model=None end-to-end.

The classes below pin each layer at the right altitude:
- TestEvalCreateRouting   — proves `bench eval create` dispatches to
                            cli/main.py.
- TestEffectiveModel      — unit tests for the helper.
- TestOracleYamlLoaders   — Evaluation.from_yaml(oracle config) → model is None.
- TestEvalCreateOracleCLI — end-to-end: invoke `bench eval create --agent oracle`
                            and assert no API-key validation error.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _plain_cli_output(output: str) -> str:
    return ANSI_ESCAPE_RE.sub("", output)


class TestEvalCreateRouting:
    """`bench eval create` must dispatch to cli/main.py:eval_create.

    The pyproject.toml `bench`/`benchflow` scripts both resolve to
    `benchflow.cli.main:app`; these tests pin the live command callback.
    """

    def test_entry_point_app_is_cli_main(self):
        from benchflow.cli.main import app

        assert app.info.name == "benchflow"

    def test_eval_create_callback_lives_in_cli_main(self):
        from benchflow.cli.main import eval_app

        create_cmds = [c for c in eval_app.registered_commands if c.name == "create"]
        assert len(create_cmds) == 1
        assert create_cmds[0].callback.__module__ == "benchflow.cli.main"

    def test_bench_eval_create_help_resolves(self):
        """Smoke test: `bench eval create --help` reaches a real callback."""
        from benchflow.cli.main import app

        result = CliRunner().invoke(app, ["eval", "create", "--help"])
        assert result.exit_code == 0
        assert "tasks-dir" in result.stdout or "task" in result.stdout.lower()

    @pytest.mark.parametrize(
        "command",
        [
            pytest.param(["eval", "create", "--help"], id="eval-create"),
            pytest.param(["environment", "create", "--help"], id="environment-create"),
        ],
    )
    def test_sandbox_help_matches_v04_supported_backends(self, command):
        """Guards ENG-92 CLI help does not advertise future sandbox backends."""
        from benchflow.cli.main import app

        result = CliRunner().invoke(app, command)

        assert result.exit_code == 0
        assert "Sandbox: docker, daytona, or modal" in result.stdout
        assert "firecracker" not in result.stdout.lower()
        assert "kubernetes" not in result.stdout.lower()
        assert "k8s" not in result.stdout.lower()

    def test_tasks_generate_help_resolves(self):
        """Guards ENG-65: `bench tasks generate` is registered on live CLI."""
        from benchflow.cli.main import app

        result = CliRunner().invoke(app, ["tasks", "generate", "--help"])
        assert result.exit_code == 0
        assert "--from-local" in _plain_cli_output(result.stdout)

    def test_eval_create_normalizes_agent_alias(self, tmp_path: Path):
        """Guards ENG-86: eval create normalizes aliases before launch."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('schema_version = "1.1"\n')
        (task / "instruction.md").write_text("solve\n")
        captured = {}

        async def fake_run(self):
            captured["agent"] = self._config.agent
            return SimpleNamespace(
                passed=1, total=1, score=1.0, errored=0, verifier_errored=0
            )

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(
                config_file=None,
                tasks_dir=task,
                source_repo=None,
                source_path=None,
                source_ref=None,
                agent="codex",
                model="gpt-4o",
                environment="docker",
                concurrency=1,
                jobs_dir=str(tmp_path / "jobs"),
                sandbox_user="agent",
                sandbox_setup_timeout=120,
                skills_dir=None,
                skill_mode="default",
                skill_creator_dir=None,
                self_gen_no_internet=False,
                agent_env=None,
            )

        assert captured["agent"] == "codex-acp"

    def test_eval_create_normalizes_sandbox_user_none(self, tmp_path: Path):
        """Guards ENG-91 P0 dogfood sandbox-user CLI regression."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('schema_version = "1.1"\n')
        (task / "instruction.md").write_text("solve\n")
        captured = {}

        async def fake_run(self):
            captured["sandbox_user"] = self._config.sandbox_user
            return SimpleNamespace(
                passed=1, total=1, score=1.0, errored=0, verifier_errored=0
            )

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(
                config_file=None,
                tasks_dir=task,
                source_repo=None,
                source_path=None,
                source_ref=None,
                agent="oracle",
                model=None,
                environment="docker",
                concurrency=1,
                jobs_dir=str(tmp_path / "jobs"),
                sandbox_user="none",
                sandbox_setup_timeout=120,
                skills_dir=None,
                skill_mode="default",
                skill_creator_dir=None,
                self_gen_no_internet=False,
                agent_env=None,
            )

        assert captured["sandbox_user"] is None

    def test_eval_create_config_rejects_unsupported_agent_protocol(
        self, tmp_path: Path
    ):
        """Guards the ENG-86 fix from commit b69bdf4 against config bypass."""
        from benchflow.cli.main import app

        tasks = tmp_path / "tasks"
        tasks.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(f"tasks_dir: {tasks}\nagent: openai/codex\n")

        result = CliRunner().invoke(app, ["eval", "create", "--config", str(config)])

        assert result.exit_code == 1
        assert "Unsupported eval agent protocol: openai" in result.stdout

    def test_eval_create_config_recomputes_model_after_agent_normalization(
        self, tmp_path: Path
    ):
        """Guards the ENG-86 fix from commit b69bdf4 for config-file agents."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        tasks = tmp_path / "tasks"
        tasks.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(f"tasks_dir: {tasks}\nagent: acp/oracle\n")
        captured = {}

        async def fake_run(self):
            captured["agent"] = self._config.agent
            captured["model"] = self._config.model
            return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(config_file=config)

        assert captured == {"agent": "oracle", "model": None}

    def test_eval_create_inherits_host_provider_key_without_agent_env(
        self, tmp_path: Path, monkeypatch
    ):
        """Guards ENG-78: CLI runs inherit provider keys without --agent-env."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('schema_version = "1.1"\n')
        (task / "instruction.md").write_text("solve\n")
        monkeypatch.setenv("GEMINI_API_KEY", "from-host")
        captured = {}

        async def fake_run(self):
            from benchflow.agents.env import resolve_agent_env

            captured["agent_env"] = dict(self._config.agent_env)
            captured["resolved_agent_env"] = resolve_agent_env(
                self._config.agent,
                self._config.model,
                self._config.agent_env,
            )
            return SimpleNamespace(
                passed=1, total=1, score=1.0, errored=0, verifier_errored=0
            )

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(
                config_file=None,
                tasks_dir=task,
                source_repo=None,
                source_path=None,
                source_ref=None,
                agent="gemini",
                model="gemini-3.1-flash-lite-preview",
                environment="docker",
                concurrency=1,
                jobs_dir=str(tmp_path / "jobs"),
                sandbox_user="agent",
                sandbox_setup_timeout=120,
                skills_dir=None,
                skill_mode="default",
                skill_creator_dir=None,
                self_gen_no_internet=False,
                agent_env=None,
            )

        assert captured["agent_env"] == {}
        assert captured["resolved_agent_env"]["GEMINI_API_KEY"] == "from-host"
        assert captured["resolved_agent_env"]["GOOGLE_API_KEY"] == "from-host"

    def test_eval_create_loads_dotenv_for_sandbox_provider_auth(
        self, tmp_path: Path, monkeypatch
    ):
        """Guards release smokes: .env sandbox credentials reach provider SDKs."""
        import os
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('schema_version = "1.1"\n')
        (task / "instruction.md").write_text("solve\n")
        env_file = tmp_path / ".env"
        env_file.write_text(
            "DAYTONA_API_KEY=from-dotenv\n"
            "MODAL_TOKEN_ID=modal-id\n"
            "MODAL_TOKEN_SECRET=modal-secret\n"
        )
        monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(env_file))
        monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
        monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
        captured = {}

        async def fake_run(self):
            captured["daytona"] = os.environ.get("DAYTONA_API_KEY")
            captured["modal_id"] = os.environ.get("MODAL_TOKEN_ID")
            captured["modal_secret"] = os.environ.get("MODAL_TOKEN_SECRET")
            return SimpleNamespace(
                passed=1, total=1, score=1.0, errored=0, verifier_errored=0
            )

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(
                config_file=None,
                tasks_dir=task,
                source_repo=None,
                source_path=None,
                source_ref=None,
                agent="oracle",
                model=None,
                environment="daytona",
                concurrency=1,
                jobs_dir=str(tmp_path / "jobs"),
                sandbox_user="agent",
                sandbox_setup_timeout=120,
                skills_dir=None,
                skill_mode="default",
                skill_creator_dir=None,
                self_gen_no_internet=False,
                agent_env=None,
            )

        assert captured == {
            "daytona": "from-dotenv",
            "modal_id": "modal-id",
            "modal_secret": "modal-secret",
        }

    def test_eval_create_exits_nonzero_when_single_task_errors(self, tmp_path: Path):
        """Guards ENG-93 release smoke evidence against false-green CLI exits."""
        from types import SimpleNamespace

        from benchflow.cli.main import app
        from benchflow.evaluation import Evaluation

        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('schema_version = "1.1"\n')
        (task / "instruction.md").write_text("solve\n")

        async def fake_run(self):
            return SimpleNamespace(
                passed=0,
                total=1,
                score=0.0,
                errored=1,
                verifier_errored=0,
            )

        with patch.object(Evaluation, "run", new=fake_run):
            result = CliRunner().invoke(
                app,
                [
                    "eval",
                    "create",
                    "--tasks-dir",
                    str(task),
                    "--agent",
                    "oracle",
                    "--sandbox",
                    "modal",
                    "--jobs-dir",
                    str(tmp_path / "jobs"),
                ],
            )

        assert result.exit_code == 1
        assert "Score: 0/1" in result.stdout

    def test_eval_create_exits_nonzero_when_single_task_verifier_errors(
        self, tmp_path: Path
    ):
        """Guards ENG-93 release smoke evidence against hidden verifier errors."""
        from types import SimpleNamespace

        from benchflow.cli.main import app
        from benchflow.evaluation import Evaluation

        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('schema_version = "1.1"\n')
        (task / "instruction.md").write_text("solve\n")

        async def fake_run(self):
            return SimpleNamespace(
                passed=0,
                total=1,
                score=0.0,
                errored=0,
                verifier_errored=1,
            )

        with patch.object(Evaluation, "run", new=fake_run):
            result = CliRunner().invoke(
                app,
                [
                    "eval",
                    "create",
                    "--tasks-dir",
                    str(task),
                    "--agent",
                    "oracle",
                    "--sandbox",
                    "docker",
                    "--jobs-dir",
                    str(tmp_path / "jobs"),
                ],
            )

        assert result.exit_code == 1
        assert "Score: 0/1" in result.stdout

    def test_eval_create_exits_nonzero_when_batch_has_harness_errors(
        self, tmp_path: Path
    ):
        """Guards ENG-93 release smoke evidence against false-green batch runs."""
        from types import SimpleNamespace

        from benchflow.cli.main import app
        from benchflow.evaluation import Evaluation

        tasks = tmp_path / "tasks"
        tasks.mkdir()

        async def fake_run(self):
            return SimpleNamespace(
                passed=0,
                total=1,
                score=0.0,
                errored=1,
                verifier_errored=0,
            )

        with patch.object(Evaluation, "run", new=fake_run):
            result = CliRunner().invoke(
                app,
                [
                    "eval",
                    "create",
                    "--tasks-dir",
                    str(tasks),
                    "--agent",
                    "oracle",
                    "--sandbox",
                    "docker",
                    "--jobs-dir",
                    str(tmp_path / "jobs"),
                ],
            )

        assert result.exit_code == 1
        assert "Score: 0/1" in result.stdout

    def test_eval_list_reads_root_summary(self, tmp_path: Path):
        """Guards ENG-83: root summary.json is treated as the eval summary."""
        import json

        from benchflow.cli.main import app

        jobs = tmp_path / "jobs-batch-oracle"
        jobs.mkdir()
        (jobs / "summary.json").write_text(
            json.dumps({"total": 2, "passed": 1, "score": "50.0%"})
        )
        child = jobs / "2026-05-18__11-26-49"
        child.mkdir()

        result = CliRunner().invoke(app, ["eval", "list", str(jobs)])
        assert result.exit_code == 0
        assert "1/2" in result.stdout
        assert "no summary" not in result.stdout


class TestEvalCreateIncludeExclude:
    """Guards ENG-159: --include and --exclude flags wire through to EvaluationConfig."""

    def test_eval_create_include_via_tasks_dir(self, tmp_path: Path):
        """Guards ENG-159: --include reaches EvaluationConfig for --tasks-dir batch runs."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        tasks = tmp_path / "tasks"
        tasks.mkdir()
        (tasks / "task-a").mkdir()
        (tasks / "task-a" / "task.toml").write_text('schema_version = "1.1"\n')
        (tasks / "task-b").mkdir()
        (tasks / "task-b" / "task.toml").write_text('schema_version = "1.1"\n')
        captured = {}

        async def fake_run(self):
            captured["include"] = self._config.include_tasks
            captured["exclude"] = self._config.exclude_tasks
            return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(
                config_file=None,
                tasks_dir=tasks,
                source_repo=None,
                source_path=None,
                source_ref=None,
                agent="oracle",
                model=None,
                environment="docker",
                concurrency=1,
                jobs_dir=str(tmp_path / "jobs"),
                sandbox_user="agent",
                sandbox_setup_timeout=120,
                skills_dir=None,
                skill_mode="default",
                skill_creator_dir=None,
                self_gen_no_internet=False,
                agent_env=None,
                include=["task-a"],
                exclude=["task-b"],
            )

        assert captured["include"] == {"task-a"}
        assert captured["exclude"] == {"task-b"}

    def test_eval_create_include_overrides_yaml(self, tmp_path: Path):
        """Guards ENG-159: CLI --include overrides YAML include list."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        tasks = tmp_path / "tasks"
        tasks.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"tasks_dir: {tasks}\nagent: oracle\ninclude:\n  - yaml-task\n"
        )
        captured = {}

        async def fake_run(self):
            captured["include"] = self._config.include_tasks
            return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(config_file=config, include=["cli-task"])

        assert captured["include"] == {"cli-task"}

    def test_eval_create_no_include_preserves_yaml(self, tmp_path: Path):
        """Guards ENG-159: without CLI --include, YAML include list is preserved."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.evaluation import Evaluation

        tasks = tmp_path / "tasks"
        tasks.mkdir()
        config = tmp_path / "config.yaml"
        config.write_text(
            f"tasks_dir: {tasks}\nagent: oracle\ninclude:\n  - yaml-task\n"
        )
        captured = {}

        async def fake_run(self):
            captured["include"] = self._config.include_tasks
            return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

        with patch.object(Evaluation, "run", new=fake_run):
            eval_create(config_file=config)

        assert captured["include"] == {"yaml-task"}

    def test_eval_create_cli_runner_include_flag(self, tmp_path: Path):
        """Guards ENG-159: --include flag works via CliRunner end-to-end."""
        from types import SimpleNamespace

        from benchflow.cli.main import app
        from benchflow.evaluation import Evaluation

        tasks = tmp_path / "tasks"
        tasks.mkdir()
        (tasks / "task-a").mkdir()
        (tasks / "task-a" / "task.toml").write_text('schema_version = "1.1"\n')
        captured = {}

        async def fake_run(self):
            captured["include"] = self._config.include_tasks
            captured["exclude"] = self._config.exclude_tasks
            return SimpleNamespace(passed=0, total=0, score=0.0, errored=0)

        with patch.object(Evaluation, "run", new=fake_run):
            result = CliRunner().invoke(
                app,
                [
                    "eval", "create",
                    "--tasks-dir", str(tasks),
                    "--include", "task-a",
                    "--exclude", "task-b",
                ],
            )

        assert result.exit_code == 0
        assert captured["include"] == {"task-a"}
        assert captured["exclude"] == {"task-b"}


class TestVerifierNonzeroExitRewardAcceptance:
    """Guards ENG-150: verifiers exiting nonzero after reward 0 must be
    classified as 'failed' (honest model failure), not 'verifier_errored'."""

    @pytest.mark.asyncio
    async def test_nonzero_exit_with_reward_zero_classified_as_failed(
        self, tmp_path: Path
    ):
        """Guards ENG-150: rc!=0 + reward=0.0 → 'failed', not 'verifier_errored'."""
        from benchflow._utils.scoring import classify_result

        outcome = classify_result(reward=0.0, error=None, verifier_error=None)
        assert outcome == "failed"

    @pytest.mark.asyncio
    async def test_nonzero_exit_reward_zero_accepted_by_verifier(
        self, tmp_path: Path
    ):
        """Guards ENG-150: Verifier accepts reward file when script exits nonzero."""
        from unittest.mock import AsyncMock, MagicMock

        from benchflow.task import RolloutPaths, Verifier
        from benchflow.task.config import TaskConfig

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        tests_dir = task_dir / "tests"
        tests_dir.mkdir()
        test_sh = tests_dir / "test.sh"
        test_sh.write_text("#!/bin/bash\nexit 1\n")

        task = MagicMock()
        task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = tests_dir
        task.paths.test_path = test_sh

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_writes_reward_zero(
            *_args: object, **_kwargs: object
        ) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_text_path.write_text("0.0")
            return MagicMock(return_code=1, stdout="tests failed")

        sandbox.exec = AsyncMock(side_effect=exec_writes_reward_zero)

        result = await Verifier(task, rollout_paths, sandbox).verify()
        assert result.rewards is not None
        assert result.rewards["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_nonzero_exit_reward_json_zero_accepted(self, tmp_path: Path):
        """Guards ENG-150: reward.json with reward=0 accepted despite rc!=0."""
        import json
        from unittest.mock import AsyncMock, MagicMock

        from benchflow.task import RolloutPaths, Verifier
        from benchflow.task.config import TaskConfig

        task_dir = tmp_path / "task"
        task_dir.mkdir()
        tests_dir = task_dir / "tests"
        tests_dir.mkdir()
        test_sh = tests_dir / "test.sh"
        test_sh.write_text("#!/bin/bash\nexit 1\n")

        task = MagicMock()
        task.config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
        task.paths.tests_dir = tests_dir
        task.paths.test_path = test_sh

        sandbox = MagicMock()
        sandbox.upload_dir = AsyncMock()
        sandbox.is_mounted = True

        rollout_paths = RolloutPaths(tmp_path / "rollout")
        rollout_paths.mkdir()

        async def exec_writes_reward_json(
            *_args: object, **_kwargs: object
        ) -> MagicMock:
            if sandbox.exec.await_count == 1:
                return MagicMock(return_code=0, stdout="")
            rollout_paths.reward_json_path.write_text(
                json.dumps({"reward": 0.0})
            )
            return MagicMock(return_code=1, stdout="tests failed")

        sandbox.exec = AsyncMock(side_effect=exec_writes_reward_json)

        result = await Verifier(task, rollout_paths, sandbox).verify()
        assert result.rewards is not None
        assert result.rewards["reward"] == 0.0

    def test_end_to_end_scoring_nonzero_verifier_reward_zero(self):
        """Guards ENG-150: full classify_score_outcome pipeline gives 'failed'."""
        from benchflow._utils.scoring import classify_score_outcome

        result = {
            "rewards": {"reward": 0.0},
            "error": None,
            "verifier_error": None,
        }
        assert classify_score_outcome(result) == "failed"

    def test_nonzero_exit_no_reward_still_verifier_error(self, tmp_path: Path):
        """Guards ENG-150 doesn't regress: rc!=0 with NO reward file is still an error."""
        from benchflow._utils.scoring import classify_result

        outcome = classify_result(reward=None, error=None, verifier_error="verifier crashed: rc=7")
        assert outcome == "verifier_errored"


class TestEffectiveModel:
    """The helper introduced in Layer 3 — single source of truth for the rule
    "oracle never gets a model; non-oracle agents fall back to DEFAULT_MODEL"."""

    def test_oracle_with_no_model_returns_none(self):
        from benchflow.evaluation import effective_model

        assert effective_model("oracle", None) is None

    def test_oracle_ignores_explicit_model(self):
        """Even if a caller forwards a model for oracle, the helper drops it."""
        from benchflow.evaluation import effective_model

        assert effective_model("oracle", "claude-haiku-4-5-20251001") is None

    def test_non_oracle_with_no_model_returns_default(self):
        from benchflow.evaluation import DEFAULT_MODEL, effective_model

        assert effective_model("claude-agent-acp", None) == DEFAULT_MODEL

    def test_non_oracle_explicit_model_passes_through(self):
        from benchflow.evaluation import effective_model

        assert effective_model("codex-acp", "gpt-5") == "gpt-5"

    def test_non_oracle_empty_model_falls_back_to_default(self):
        """Empty string == "no model" — legacy YAML can produce this shape."""
        from benchflow.evaluation import DEFAULT_MODEL, effective_model

        assert effective_model("claude-agent-acp", "") == DEFAULT_MODEL


class TestOracleYamlLoaders:
    """YAML configs for oracle must produce EvaluationConfig.model is None.

    Both loader paths (_from_native_yaml, _from_legacy_yaml) previously
    coalesced missing model to DEFAULT_MODEL unconditionally — Layer 3
    routes them through effective_model() so oracle drops the default.
    """

    def _make_task(self, tmp_path: Path) -> Path:
        tasks = tmp_path / "tasks" / "task-a"
        tasks.mkdir(parents=True)
        (tasks / "task.toml").write_text('schema_version = "1.1"\n')
        return tmp_path / "tasks"

    def test_native_yaml_oracle_no_model(self, tmp_path: Path):
        from benchflow.evaluation import Evaluation

        self._make_task(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("tasks_dir: tasks\nagent: oracle\n")
        job = Evaluation.from_yaml(config)
        assert job._config.agent == "oracle"
        assert job._config.model is None

    def test_legacy_yaml_oracle_no_model(self, tmp_path: Path):
        from benchflow.evaluation import Evaluation

        self._make_task(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("agents:\n  - name: oracle\ndatasets:\n  - path: tasks\n")
        job = Evaluation.from_yaml(config)
        assert job._config.agent == "oracle"
        assert job._config.model is None

    def test_native_yaml_non_oracle_keeps_default_when_omitted(self, tmp_path: Path):
        """Backwards-compat: omitting model for an LLM agent still gets DEFAULT_MODEL."""
        from benchflow.evaluation import DEFAULT_MODEL, Evaluation

        self._make_task(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("tasks_dir: tasks\nagent: claude-agent-acp\n")
        job = Evaluation.from_yaml(config)
        assert job._config.model == DEFAULT_MODEL


class TestEvalCreateOracleCLI:
    """End-to-end: `bench eval create --agent oracle` must not trip API key validation.

    This is the user-visible bug the chokepoint test guards against at the
    unit level. Here we call the live handler (cli/main.py:eval_create)
    directly rather than through Typer's CliRunner; the handler still runs
    asyncio.run() internally, exercising the full
    CLI → effective_model → RolloutConfig → resolve_agent_env path the bug
    originally lived in.
    """

    def _make_task(self, tmp_path: Path) -> Path:
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_text('schema_version = "1.1"\n')
        (task / "instruction.md").write_text("solve\n")
        return task

    def _strip_api_keys(self, monkeypatch):
        for k in (
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
            "LLM_API_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

    def test_oracle_single_task_no_api_key_no_error(self, tmp_path: Path, monkeypatch):
        """The bug: oracle + missing API key → ANTHROPIC_API_KEY ValueError."""
        from types import SimpleNamespace

        from benchflow.cli.main import eval_create
        from benchflow.models import RunResult

        self._strip_api_keys(monkeypatch)
        task = self._make_task(tmp_path)
        captured: dict = {}

        async def fake_create(config):
            captured["config"] = config
            # Exercise the real chokepoint with the config the CLI built —
            # that's the specific call site the bug manifested at.
            from benchflow.agents.env import resolve_agent_env

            captured["agent_env"] = resolve_agent_env(
                config.primary_agent, config.primary_model, config.agent_env
            )
            trial = SimpleNamespace(
                run=AsyncMock(
                    return_value=RunResult(
                        task_name="task",
                        agent_name="oracle",
                        rewards={"reward": 1.0},
                        n_tool_calls=0,
                    )
                )
            )
            return trial

        with patch("benchflow.rollout.Rollout.create", new=fake_create):
            eval_create(
                config_file=None,
                tasks_dir=task,
                agent="oracle",
                model=None,
                environment="docker",
                concurrency=1,
                jobs_dir=str(tmp_path / "jobs"),
                sandbox_user="agent",
                skills_dir=None,
            )

        cfg = captured["config"]
        # Layer 3: oracle never receives a model, even when CLI defaults exist.
        assert cfg.primary_agent == "oracle"
        assert cfg.primary_model is None
        # Layer 1: chokepoint did not inject provider env or raise.
        assert "BENCHFLOW_PROVIDER_MODEL" not in captured["agent_env"]


# ── ENG-149: idle timeout diagnostics in result.json ──


class TestIdleTimeoutResultDiagnostics:
    """Guards ENG-149: result.json must record structured idle timeout diagnostics
    so check_results/dashboard can identify invalidated measurements."""

    def test_error_category_persisted_for_idle_timeout(self, tmp_path):
        """Guards ENG-149: result.json contains error_category='idle_timeout'."""
        from benchflow._utils.scoring import classify_error
        from benchflow.rollout import _build_rollout_result

        error = "Agent idle for 600s with no new tool call, message, or thought (last activity 602s ago, 3 tool calls so far)"
        idle_info = {
            "reason": "idle_timeout",
            "idle_timeout_sec": 600,
            "idle_duration_sec": 602,
            "wall_clock_elapsed_sec": 605,
            "n_tool_calls": 3,
            "n_message_chunks": 0,
            "n_thought_chunks": 1,
            "last_activity_at": "2026-05-23T16:00:00+00:00",
        }
        from datetime import datetime

        result = _build_rollout_result(
            tmp_path,
            task_name="court-form-filling",
            rollout_name="trial-0",
            agent="gemini",
            agent_name="gemini-agent",
            model="google/gemini-3.1-flash-lite-preview",
            n_tool_calls=3,
            prompts=["solve"],
            error=error,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=True,
            rewards=None,
            started_at=datetime.now(),
            timing={},
            idle_timeout_info=idle_info,
        )
        assert result.error == error
        result_json = json.loads((tmp_path / "result.json").read_text())
        assert result_json["error_category"] == "idle_timeout"
        assert result_json["idle_timeout_info"] == idle_info
        assert result_json["idle_timeout_info"]["n_tool_calls"] == 3
        assert result_json["idle_timeout_info"]["idle_duration_sec"] == 602
        # Also verify classify_error agrees
        assert classify_error(error) == "idle_timeout"

    def test_error_category_persisted_for_non_idle_errors(self, tmp_path):
        """Guards ENG-149: error_category is set for all error types."""
        from datetime import datetime

        from benchflow.rollout import _build_rollout_result

        _build_rollout_result(
            tmp_path,
            task_name="some-task",
            rollout_name="trial-0",
            agent="gemini",
            agent_name="gemini-agent",
            model="google/gemini-3.1-flash-lite-preview",
            n_tool_calls=0,
            prompts=["solve"],
            error="install failed: npm ERR!",
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards=None,
            started_at=datetime.now(),
            timing={},
        )
        result_json = json.loads((tmp_path / "result.json").read_text())
        assert result_json["error_category"] == "install_failure"
        assert result_json["idle_timeout_info"] is None

    def test_error_category_null_on_success(self, tmp_path):
        """Guards ENG-149: error_category is null when no error."""
        from datetime import datetime

        from benchflow.rollout import _build_rollout_result

        _build_rollout_result(
            tmp_path,
            task_name="passing-task",
            rollout_name="trial-0",
            agent="gemini",
            agent_name="gemini-agent",
            model="google/gemini-3.1-flash-lite-preview",
            n_tool_calls=5,
            prompts=["solve"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
        )
        result_json = json.loads((tmp_path / "result.json").read_text())
        assert result_json["error_category"] is None
        assert result_json["idle_timeout_info"] is None

    def test_classify_error_categories_comprehensive(self):
        """Guards ENG-149: all expected error categories are classified."""
        from benchflow._utils.scoring import classify_error

        assert classify_error("Agent idle for 600s with no new tool call") == "idle_timeout"
        assert classify_error("install failed: pip error") == "install_failure"
        assert classify_error("Agent closed stdout") == "pipe_closed"
        assert classify_error("ACP error: session closed") == "acp_error"
        assert classify_error("prompt exceeded wall-clock budget 300s") == "timeout"
        assert classify_error("connection lost to sandbox") == "infra_failure"
        assert classify_error(None) is None
        assert classify_error("") is None
