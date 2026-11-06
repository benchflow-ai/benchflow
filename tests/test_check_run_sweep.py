"""Guards the eight check/run regressions reviewed at PY8 D3 tip b9f248b4."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.runtime import check_agent_routes
from tests._taskmd_helpers import EXAMPLES
from tests.test_check_run_plan import _run_refusal, _task


@pytest.fixture(autouse=True)
def isolated_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "dummy.env"))
    for key in (
        "BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "BENCHFLOW_TASKMD_JUDGE_MODEL",
    ):
        # Register restoration even when dotenv later inserts an absent key.
        monkeypatch.setenv(key, "dummy-cleared")
        monkeypatch.delenv(key)


@pytest.mark.parametrize("command", [["check"], ["tasks", "check"]])
@pytest.mark.parametrize("agent", [None, "claude"])
def test_default_model_validates_proxy_extras(tmp_path, monkeypatch, command, agent):
    """Guards b9f248b4: omitted models and aliases use run's effective model."""
    task = _task(tmp_path / "default-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-d3-key")
    monkeypatch.setenv("BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS", "litellm==1.80.0")
    args = [*command, str(task)] + (["--agent", agent] if agent else [])
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1, result.output
    assert "conflicts with the locked" in result.stderr


def test_dotenv_proxy_extras_reach_the_validator(tmp_path):
    """Guards b9f248b4: all validators see the run's dotenv environment."""
    task = _task(tmp_path / "dotenv-extra")
    (tmp_path / "dummy.env").write_text(
        "ANTHROPIC_API_KEY=dummy-d3-key\n"
        "BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS=litellm==1.80.0\n"
    )
    result = CliRunner().invoke(
        app, ["check", str(task), "--model", "anthropic/claude-haiku-4-5"]
    )
    assert result.exit_code == 1, result.output
    assert "conflicts with the locked" in result.stderr


def test_dotenv_judge_credentials_reach_launch_preflight(tmp_path):
    """Guards b9f248b4: dotenv-only judge credentials permit a valid plan."""
    task = tmp_path / "judge"
    shutil.copytree(EXAMPLES / "analysis-judge", task)
    (tmp_path / "dummy.env").write_text("ANTHROPIC_API_KEY=dummy-d3-key\n")
    result = CliRunner().invoke(app, ["check", str(task), "--agent", "oracle"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("mode", ["allowlist", "denylist"])
@pytest.mark.parametrize("user", ["null", "none", "NULL"])
def test_network_policy_requires_an_agent_user(tmp_path, mode, user):
    """Guards b9f248b4: run's egress UID prerequisite also gates check."""
    hosts = (
        'allowed_hosts = ["example.invalid"]'
        if mode == "allowlist"
        else 'blocked_hosts = ["example.invalid"]'
    )
    task = _task(tmp_path / "egress", f'[sandbox]\nnetwork_mode = "{mode}"\n{hosts}\n')
    reason = _run_refusal(task, tmp_path, sandbox_user=user)
    assert reason == f"network_mode={mode!r} requires a sandbox_user"
    result = CliRunner().invoke(app, ["check", str(task), "--sandbox-user", user])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


def test_agent_allowlist_override_uses_the_same_user_gate(tmp_path):
    """Guards b9f248b4's sibling path: agent policy overrides sandbox policy."""
    task = _task(
        tmp_path / "agent-policy",
        '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.invalid"]\n',
    )
    reason = _run_refusal(task, tmp_path, sandbox_user="null")
    assert reason == "network_mode='allowlist' requires a sandbox_user"
    result = CliRunner().invoke(app, ["check", str(task), "--sandbox-user", "null"])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


@pytest.mark.parametrize("agent", ["oracle", "nop"])
def test_scripted_seats_keep_the_root_egress_exemption(tmp_path, agent):
    """Guards the scripted-agent exception while fixing b9f248b4."""
    task = _task(
        tmp_path / "scripted",
        '[sandbox]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.invalid"]\n',
    )
    result = CliRunner().invoke(
        app, ["check", str(task), "--sandbox-user", "null", "--agent", agent]
    )
    assert result.exit_code == 0, result.output


def test_unroutable_agent_model_matches_run(tmp_path):
    """Guards b9f248b4: check applies the same pure route refusal as run."""
    task = _task(tmp_path / "route")
    config = SimpleNamespace(agent="claude-agent-acp", model="openai/gpt-4.1")
    with pytest.raises(ValueError) as refusal:
        check_agent_routes([config])
    result = CliRunner().invoke(
        app, ["check", str(task), "--agent", config.agent, "--model", config.model]
    )
    assert result.exit_code == 1, result.output
    assert str(refusal.value) in result.stderr


@pytest.mark.parametrize("user", ["null", "none", "NULL", "root"])
def test_declared_root_accepts_run_user_spellings(tmp_path, user):
    """Guards b9f248b4: the run's root sentinel spellings agree in check."""
    task = tmp_path / "root-user"
    shutil.copytree(EXAMPLES / "hello-world", task)
    path = task / "task.md"
    path.write_text(path.read_text().replace("[agent]", '[agent]\nuser = "root"'))
    result = CliRunner().invoke(app, ["check", str(task), "--sandbox-user", user])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("draft2", [False, True])
def test_schema_level_does_not_apply_launch_refusals(tmp_path, draft2):
    """Guards b9f248b4: schema-only authoring checks remain parse-only."""
    if draft2:
        task = tmp_path / "schema-draft2"
        shutil.copytree(EXAMPLES / "calc-quarterly", task)
    else:
        task = (
            Path(__file__).resolve().parents[1]
            / "docs/examples/task-md/task-toml-parity"
        )
    result = CliRunner().invoke(app, ["tasks", "check", str(task), "--level", "schema"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("filtered", [False, True])
def test_batch_root_error_and_excluded_symlink_keep_healthy_tasks(
    tmp_path, caplog, filtered
):
    """Guards b9f248b4: an excluded link cannot suppress a healthy batch."""
    suite = tmp_path / "suite"
    good = _task(suite / "good")
    (suite / "task.toml").write_text("version = [\n")
    excluded = suite / "excluded"
    excluded.mkdir()
    (excluded / "link").symlink_to("dummy-missing")
    job = Evaluation(
        tasks_dir=suite,
        jobs_dir=tmp_path / "jobs",
        preflight=False,
        config=EvaluationConfig(
            agent="oracle", include_tasks={"good"} if filtered else None
        ),
    )
    assert job._get_task_dirs() == [good]
    assert "Ignoring malformed task.toml" in caplog.text
    assert "excluded" not in caplog.text


@pytest.mark.parametrize(
    "level", [None, "schema", "runtime-capability", "publication-grade", "acceptance"]
)
@pytest.mark.parametrize("reference_probe", [True, False])
def test_default_family_check_never_runs_a_generator(
    tmp_path, monkeypatch, level, reference_probe
):
    """Guards b9f248b4: neither reference nor Docker generators run by default."""
    from benchflow.taskmd import family
    from benchflow.taskmd._vendor import taskmd as ref

    task = tmp_path / "family"
    shutil.copytree(EXAMPLES / "sql-family", task)
    calls = []

    def generated(*args, **kwargs):
        calls.append("generator")
        raise AssertionError("ordinary check executed a family generator")

    monkeypatch.setattr(
        ref,
        "run_generator",
        generated
        if reference_probe
        else lambda *args: (False, "dummy probe unavailable"),
    )
    monkeypatch.setattr(family, "generate", generated)
    monkeypatch.setattr(family.shutil, "which", lambda name: "/usr/bin/docker")
    args = ["check", str(task), "--agent", "oracle"]
    if level:
        args += ["--level", level, "--sandbox", "docker"]
    result = CliRunner().invoke(app, args)
    assert calls == [], result.output
    if level in (None, "schema", "runtime-capability"):
        assert result.exit_code == 0, result.output
    if level != "schema":
        assert "source plan" in result.output


@pytest.mark.parametrize("level", ["equivalence", "acceptance-live"])
def test_family_execution_requires_an_explicit_execution_level(
    tmp_path, monkeypatch, level
):
    """Guards b9f248b4's fix: execution levels still reach the real generator boundary."""
    from benchflow.taskmd import family
    from benchflow.taskmd._vendor import taskmd as ref

    task = tmp_path / "family-execution"
    shutil.copytree(EXAMPLES / "sql-family", task)
    calls = []

    def generated(*args, **kwargs):
        calls.append("generator")
        raise family.FamilyError("dummy generator boundary")

    monkeypatch.setattr(ref, "run_generator", lambda *args: (False, "dummy probe"))
    monkeypatch.setattr(family, "generate", generated)
    monkeypatch.setattr(family.shutil, "which", lambda name: "/usr/bin/docker")
    result = CliRunner().invoke(
        app,
        [
            "check",
            str(task),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--level",
            level,
        ],
    )
    assert calls == ["generator"], result.output
    assert result.exit_code == 1, result.output
    assert "dummy generator boundary" in result.output
