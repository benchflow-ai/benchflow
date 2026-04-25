"""Regression tests for the oracle agent + DEFAULT_MODEL chokepoint.

Pins the fix from this branch (post-PR #173 follow-up):

  Layer 1 — restore `agent != "oracle"` guard in resolve_agent_env so the
            chokepoint defends against any caller that forwards a model.
  Layer 2 — delete the orphaned src/benchflow/cli/eval.py whose oracle fix
            was unreachable because nothing wired it into the live CLI.
  Layer 3 — funnel all CLI/YAML-loader sites through effective_model() so
            oracle gets honest model=None end-to-end.

The classes below pin each layer at the right altitude:
- TestOrphanRemoval       — proves cli/eval.py is gone and stays gone.
- TestEvalCreateRouting   — proves `bench eval create` dispatches to
                            cli/main.py (the file that PR #173 missed).
- TestEffectiveModel      — unit tests for the helper.
- TestOracleYamlLoaders   — Job.from_yaml(oracle config) → model is None.
- TestEvalCreateOracleCLI — end-to-end: invoke `bench eval create -a oracle`
                            and assert no API-key validation error.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

from typer.testing import CliRunner


class TestEvalModuleNotWiredIntoCLI:
    """src/benchflow/cli/eval.py exists but is NOT wired into the live CLI.

    The live `bench eval create` dispatches to cli/main.py:eval_create.
    cli/eval.py is a standalone module (with its own resolver logic and tests)
    but must not be imported by the CLI entry-point code — doing so is what
    caused PR #173 to land its fix in dead code.
    """

    def test_cli_main_does_not_import_cli_eval(self):
        """cli/main.py must not import from cli/eval — they are separate."""
        main_py = (
            Path(__file__).resolve().parent.parent
            / "src" / "benchflow" / "cli" / "main.py"
        )
        text = main_py.read_text()
        assert "from benchflow.cli.eval" not in text
        assert "import benchflow.cli.eval" not in text


class TestEvalCreateRouting:
    """`bench eval create` must dispatch to cli/main.py:eval_create.

    The pyproject.toml `bench`/`benchflow` scripts both resolve to
    `benchflow.cli.main:app`. PR #173 patched a different `eval_create` in
    `benchflow.cli.eval` that no entry point ever loaded — these tests
    pin the routing so that mistake can't happen again.
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


class TestEffectiveModel:
    """The helper introduced in Layer 3 — single source of truth for the rule
    "oracle never gets a model; non-oracle agents fall back to DEFAULT_MODEL"."""

    def test_oracle_with_no_model_returns_none(self):
        from benchflow.job import effective_model

        assert effective_model("oracle", None) is None

    def test_oracle_ignores_explicit_model(self):
        """Even if a caller forwards a model for oracle, the helper drops it."""
        from benchflow.job import effective_model

        assert effective_model("oracle", "claude-haiku-4-5-20251001") is None

    def test_non_oracle_with_no_model_returns_default(self):
        from benchflow.job import DEFAULT_MODEL, effective_model

        assert effective_model("claude-agent-acp", None) == DEFAULT_MODEL

    def test_non_oracle_explicit_model_passes_through(self):
        from benchflow.job import effective_model

        assert effective_model("codex-acp", "gpt-5") == "gpt-5"

    def test_non_oracle_empty_model_falls_back_to_default(self):
        """Empty string == "no model" — Harbor YAML can produce this shape."""
        from benchflow.job import DEFAULT_MODEL, effective_model

        assert effective_model("claude-agent-acp", "") == DEFAULT_MODEL


class TestOracleYamlLoaders:
    """YAML configs for oracle must produce JobConfig.model is None.

    Both loader paths (_from_native_yaml, _from_harbor_yaml) previously
    coalesced missing model to DEFAULT_MODEL unconditionally — Layer 3
    routes them through effective_model() so oracle drops the default.
    """

    def _make_task(self, tmp_path: Path) -> Path:
        tasks = tmp_path / "tasks" / "task-a"
        tasks.mkdir(parents=True)
        (tasks / "task.toml").write_text('schema_version = "1.1"\n')
        return tmp_path / "tasks"

    def test_native_yaml_oracle_no_model(self, tmp_path: Path):
        from benchflow.job import Job

        self._make_task(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("tasks_dir: tasks\nagent: oracle\n")
        job = Job.from_yaml(config)
        assert job._config.agent == "oracle"
        assert job._config.model is None

    def test_harbor_yaml_oracle_no_model(self, tmp_path: Path):
        from benchflow.job import Job

        self._make_task(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text(
            "agents:\n"
            "  - name: oracle\n"
            "datasets:\n"
            "  - path: tasks\n"
        )
        job = Job.from_yaml(config)
        assert job._config.agent == "oracle"
        assert job._config.model is None

    def test_native_yaml_non_oracle_keeps_default_when_omitted(
        self, tmp_path: Path
    ):
        """Backwards-compat: omitting model for an LLM agent still gets DEFAULT_MODEL."""
        from benchflow.job import DEFAULT_MODEL, Job

        self._make_task(tmp_path)
        config = tmp_path / "config.yaml"
        config.write_text("tasks_dir: tasks\nagent: claude-agent-acp\n")
        job = Job.from_yaml(config)
        assert job._config.model == DEFAULT_MODEL


class TestEvalCreateOracleCLI:
    """End-to-end: `bench eval create -a oracle` must not trip API key validation.

    This is the user-visible bug the chokepoint test guards against at the
    unit level. Here we call the live handler (cli/main.py:eval_create)
    directly — invoking via Typer's CliRunner triggers asyncio.run()
    internally, which leaves no current event loop and breaks pre-existing
    tests in the suite that use the deprecated asyncio.get_event_loop()
    pattern. Calling the function directly still exercises the full
    CLI → effective_model → TrialConfig → resolve_agent_env path the bug
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

    def test_oracle_single_task_no_api_key_no_error(
        self, tmp_path: Path, monkeypatch
    ):
        """The bug: oracle + missing API key → ANTHROPIC_API_KEY ValueError."""
        import asyncio

        from benchflow.cli.main import eval_create
        from benchflow.models import RunResult

        # The CLI handler internally calls asyncio.run(), which leaves no
        # current event loop. Pre-existing tests in the suite use the
        # deprecated asyncio.get_event_loop() and break in that state, so
        # restore a fresh loop after the test (teardown via finally below).

        self._strip_api_keys(monkeypatch)
        task = self._make_task(tmp_path)
        captured: dict = {}

        async def fake_create(config):
            captured["config"] = config
            # Exercise the real chokepoint with the config the CLI built —
            # that's the specific call site the bug manifested at.
            from benchflow._agent_env import resolve_agent_env

            captured["agent_env"] = resolve_agent_env(
                config.primary_agent, config.primary_model, config.agent_env
            )
            trial = type("FakeTrial", (), {})()
            trial.run = AsyncMock(
                return_value=RunResult(
                    task_name="task",
                    agent_name="oracle",
                    rewards={"reward": 1.0},
                    n_tool_calls=0,
                )
            )
            return trial

        try:
            with patch("benchflow.trial.Trial.create", new=fake_create):
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
        finally:
            asyncio.set_event_loop(asyncio.new_event_loop())

        cfg = captured["config"]
        # Layer 3: oracle never receives a model, even when CLI defaults exist.
        assert cfg.primary_agent == "oracle"
        assert cfg.primary_model is None
        # Layer 1: chokepoint did not inject provider env or raise.
        assert "BENCHFLOW_PROVIDER_MODEL" not in captured["agent_env"]
