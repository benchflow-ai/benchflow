"""Model defaults, and the model a run card says a run uses and where from."""

import inspect
import json
import re

import pytest
import yaml
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import DEFAULT_AGENT, DEFAULT_MODEL, effective_model
from benchflow.rewards.builtins import LLMJudgeRewardFunc
from benchflow.rewards.rubric_config import JudgeConfig, _parse_judge
from benchflow.run_card import build_run_card, card_lines
from benchflow.task.config import JudgeVerifierConfig
from tests.test_run_card import _evaluation, _tasks


def test_current_default_models():
    assert effective_model(DEFAULT_AGENT, None) == "claude-haiku-5-5"
    assert JudgeVerifierConfig().model == "claude-sonnet-5-5"
    assert JudgeConfig().model == "claude-sonnet-5-5"
    assert _parse_judge({}).model == "claude-sonnet-5-5"
    assert (
        inspect.signature(LLMJudgeRewardFunc).parameters["model"].default
        == "claude-sonnet-5-5"
    )


@pytest.mark.parametrize("key", ["ANTHROPIC_MODEL", "BENCHFLOW_PROVIDER_MODEL"])
def test_card_names_env_model_when_no_model_argument(tmp_path, key):
    job = _evaluation(
        tmp_path, _tasks(tmp_path), model=None, agent_env={key: "mimo-v2.6-flash"}
    )
    card = build_run_card(job, job._get_task_dirs())
    assert card["model"]["id"] == "mimo-v2.6-flash"
    assert card["model"]["source"] == key
    assert key in "\n".join(card_lines(card, tmp_path))


def test_card_explicit_model_and_controls_take_precedence(tmp_path):
    job = _evaluation(
        tmp_path,
        _tasks(tmp_path),
        model="explicit-model",
        agent_env={"ANTHROPIC_MODEL": "env-model"},
    )
    assert build_run_card(job, job._get_task_dirs())["model"]["id"] == "explicit-model"
    job._config.agent = "oracle"
    assert build_run_card(job, job._get_task_dirs())["model"]["id"] is None


def test_assess_help_has_no_internal_issue_key():
    result = CliRunner().invoke(app, ["assess", "--help"], color=False)
    assert result.exit_code == 0
    assert not re.search(r"\b(?:EN|BEN3|BD|ENG)-\d+\b", result.output)
    assert "collector" in result.output


def test_cap_and_modal_help_already_fixed(tmp_path):
    job = _evaluation(tmp_path, _tasks(tmp_path))
    text = "\n".join(card_lines(build_run_card(job, job._get_task_dirs()), tmp_path))
    assert "128000" in text
    assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" in text
    result = CliRunner().invoke(
        app, ["run", "--help-all"], color=False, terminal_width=240
    )
    assert result.exit_code == 0
    separate = result.output.split("--separate-verifier ", 1)[1].split(
        "--separate-verifier-include", 1
    )[0]
    assert "Modal" in separate


def test_cli_plan_preserves_environment_model(tmp_path):
    from benchflow.eval_plan import EvalCreateRequest, build_eval_plan
    from benchflow.evaluation import Evaluation

    plan = build_eval_plan(
        EvalCreateRequest(
            tasks_dir=_tasks(tmp_path),
            agent="claude-agent-acp",
            agent_env={"ANTHROPIC_MODEL": "mimo-v2.6-flash"},
        )
    )
    config = plan.make_eval_config()
    assert config.model == "mimo-v2.6-flash"
    assert config.model_given is False
    job = Evaluation(
        tasks_dir=str(tmp_path / "tasks"),
        jobs_dir=str(tmp_path / "jobs"),
        config=config,
        preflight=False,
    )
    card = build_run_card(job, job._get_task_dirs())
    assert card["model"]["id"] == "mimo-v2.6-flash"
    assert card["model"]["source"] == "ANTHROPIC_MODEL"
    assert card["model"]["given"] is False


def test_native_env_wins_generic_but_explicit_model_wins_both():
    env = {
        "ANTHROPIC_MODEL": "native-model",
        "BENCHFLOW_PROVIDER_MODEL": "generic-model",
    }
    assert effective_model("claude-agent-acp", None, env) == "native-model"
    assert (
        effective_model("claude-agent-acp", "explicit-model", env) == "explicit-model"
    )
    assert effective_model("oracle", None, env) is None


def test_preflight_checks_env_model(monkeypatch):
    from benchflow.agents import env as agent_env
    from benchflow.cli.doctor import eval_preflight

    seen = []
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    monkeypatch.setattr(
        agent_env, "unroutable_agent_model", lambda agent, model: seen.append(model)
    )
    eval_preflight(
        sandbox="docker",
        agent="claude-agent-acp",
        agent_env={"ANTHROPIC_MODEL": "native-model"},
    )
    assert seen == ["native-model"]


def test_yaml_load_resolves_environment_model(tmp_path):
    import yaml

    from benchflow.evaluation import Evaluation

    tasks = _tasks(tmp_path)
    config = tmp_path / "eval.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "tasks_dir": str(tasks),
                "agent": "claude-agent-acp",
                "agent_env": {"ANTHROPIC_MODEL": "native-model"},
            }
        )
    )
    job = Evaluation.from_yaml(config, preflight=False)
    assert job._config.model == "native-model"
    assert build_run_card(job, job._get_task_dirs())["model"]["id"] == "native-model"


OPUS = "claude-opus-5-5"


@pytest.fixture
def no_host_model(monkeypatch):
    """No model variable from the shell running the tests."""
    for key in ("ANTHROPIC_MODEL", "BENCHFLOW_PROVIDER_MODEL"):
        monkeypatch.delenv(key, raising=False)


def _dry_run(tmp_path, *args):
    """``bench run --dry-run``: the plan (stdout JSON), its card's model, and
    the card's printed Model line."""
    argv = ["run", *args, "--sandbox", "docker"]
    argv += ["--jobs-dir", str(tmp_path / "jobs"), "--dry-run"]
    result = CliRunner().invoke(app, argv, env={"COLUMNS": "400"})
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    line = next(x for x in result.stderr.splitlines() if x.startswith("  Model:"))
    return plan, plan["run_card"]["model"], line


def _config(tmp_path, **fields):
    config = tmp_path / "eval.yaml"
    config.write_text(yaml.safe_dump({"tasks_dir": str(_tasks(tmp_path)), **fields}))
    return config


def test_a_run_naming_no_model_says_it_runs_the_default(tmp_path, no_host_model):
    """The plan fills in the default model before the card is built, so the
    card read the default as a model the run named."""
    plan, model, line = _dry_run(
        tmp_path, str(_tasks(tmp_path)), "--agent", "claude-agent-acp"
    )
    assert plan["model"] == model["id"] == DEFAULT_MODEL
    assert (model["given"], model["source"]) == (False, "default")
    assert f"{DEFAULT_MODEL} (the agent's default)" in line


def test_a_model_variable_on_the_command_line_reaches_a_config_file_run(
    tmp_path, no_host_model
):
    """--config with --agent-env ANTHROPIC_MODEL=...: the model was resolved
    from the file before the command line's variables were merged, so the
    run stayed on the default model."""
    config = _config(tmp_path, agent="claude-agent-acp")
    plan, model, line = _dry_run(
        tmp_path, "--config", str(config), "--agent-env", f"ANTHROPIC_MODEL={OPUS}"
    )
    assert plan["model"] == model["id"] == OPUS
    assert (model["given"], model["source"]) == (False, "ANTHROPIC_MODEL")
    assert f"{OPUS} (from ANTHROPIC_MODEL)" in line


def test_a_config_files_own_model_wins_over_a_command_line_variable(
    tmp_path, no_host_model
):
    config = _config(tmp_path, agent="claude-agent-acp", model="named-model")
    plan, model, _ = _dry_run(
        tmp_path, "--config", str(config), "--agent-env", f"ANTHROPIC_MODEL={OPUS}"
    )
    assert plan["model"] == model["id"] == "named-model"
    assert (model["given"], model["source"]) == (True, "model")


def test_a_config_file_run_with_another_agent_takes_that_agents_default(
    tmp_path, no_host_model
):
    """A file that names no model: --agent picks the agent and so its
    default model, not the one resolved for the file's own agent."""
    from benchflow.agents.registry import AGENTS

    config = _config(tmp_path, agent="claude-agent-acp")
    plan, model, _ = _dry_run(tmp_path, "--config", str(config), "--agent", "gemini")
    assert plan["model"] == model["id"] == AGENTS["gemini"].default_model
    assert (model["given"], model["source"]) == (False, "default")


def test_a_model_argument_is_given_even_when_the_variable_matches(
    tmp_path, no_host_model
):
    """--model X --agent-env ANTHROPIC_MODEL=X was reported as taken from the
    variable, not given."""
    _, model, line = _dry_run(
        tmp_path,
        str(_tasks(tmp_path)),
        "--agent",
        "claude-agent-acp",
        "--model",
        OPUS,
        "--agent-env",
        f"ANTHROPIC_MODEL={OPUS}",
    )
    assert (model["id"], model["given"], model["source"]) == (OPUS, True, "model")
    assert "ANTHROPIC_MODEL" not in line


@pytest.mark.parametrize(
    ("entry", "given"),
    [
        ({"name": "claude-agent-acp"}, False),
        ({"name": "claude-agent-acp", "model_name": "anthropic/named"}, True),
    ],
)
def test_a_job_config_records_whether_its_agent_names_a_model(tmp_path, entry, given):
    from benchflow.evaluation import Evaluation

    config = tmp_path / "job.yaml"
    config.write_text(
        yaml.safe_dump(
            {"agents": [entry], "datasets": [{"path": str(_tasks(tmp_path))}]}
        )
    )
    job = Evaluation.from_yaml(config, preflight=False)
    assert job._config.model_given is given
    assert build_run_card(job, job._get_task_dirs())["model"]["given"] is given


def test_a_config_file_agent_with_no_default_model_needs_one(tmp_path, no_host_model):
    """`agent: codex` named no model and got the Claude default: the file's
    alias was resolved before it was normalized. --agent codex is refused
    without --model; a config file is now refused the same way."""
    config = _config(tmp_path, agent="codex")
    argv = ["run", "--config", str(config), "--sandbox", "docker"]
    argv += ["--jobs-dir", str(tmp_path / "jobs"), "--dry-run"]
    result = CliRunner().invoke(app, argv, env={"COLUMNS": "400"})
    assert result.exit_code != 0
    assert "has no default model; pass --model" in result.output
    assert DEFAULT_MODEL not in result.stdout
