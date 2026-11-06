"""A task's ``${VAR}`` env values read only variables the operator named.

task.toml env tables (``[environment] env``, setup commands, ``[verifier]
env``, ``[solution] env``) were filled from the operator's environment and
``.env`` with no allowlist, so a registry task could ask for
``ANTHROPIC_API_KEY`` and hand it to its own scripts. Now an unnamed
``${VAR}`` is refused before any sandbox starts, an unnamed
``${VAR:-default}`` gets its default, and ``--allow-task-env NAME`` or
``BENCHFLOW_ALLOW_TASK_ENV`` names a variable. One test per place a task's
env table is filled.

The names under test are read as strings where they can be, so each test
fails on the code before the change by what it does, not by an import.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

ALLOW = "BENCHFLOW_ALLOW_TASK_ENV"
SECRET = "s1-operator-value-never-in-a-task"
REFUSED = r"--allow-task-env S1_SECRET"


@pytest.fixture(autouse=True)
def _operator_env(monkeypatch, tmp_path):
    """The operator has S1_SECRET set, names nothing, and has no .env."""
    monkeypatch.setenv(ALLOW, "")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(tmp_path / "absent.env"))
    monkeypatch.setenv("S1_SECRET", SECRET)
    monkeypatch.delenv("S1_UNSET", raising=False)


def _name(monkeypatch, *names: str) -> None:
    monkeypatch.setenv(ALLOW, ",".join(names))


def _resolve(env: dict[str, str]) -> dict[str, str]:
    from benchflow.task.env import resolve_env_vars

    return resolve_env_vars(env)


def _write_task(task_dir: Path, toml_body: str = "") -> Path:
    """A split-layout task that passes ``bench tasks check``."""
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (task_dir / "tests").mkdir(exist_ok=True)
    (task_dir / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    (task_dir / "solution").mkdir(exist_ok=True)
    (task_dir / "solution" / "solve.sh").write_text("#!/bin/sh\necho ok\n")
    (task_dir / "instruction.md").write_text("Write ok.\n")
    (task_dir / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 60\n\n'
        f"[verifier]\ntimeout_sec = 60\n\n{toml_body}"
    )
    return task_dir


# The resolver


def test_unnamed_variable_is_refused_naming_it_and_the_opt_in():
    with pytest.raises(ValueError, match=REFUSED) as raised:
        _resolve({"KEY": "${S1_SECRET}"})
    message = str(raised.value)
    assert "S1_SECRET" in message
    assert f"{ALLOW}=S1_SECRET" in message
    assert SECRET not in message


def test_unnamed_default_gets_the_default_not_the_operators_value(
    monkeypatch, tmp_path
):
    dotenv = tmp_path / "operator.env"
    dotenv.write_text("S1_DOTENV=from-dotenv\n")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))

    assert _resolve(
        {"A": "${S1_SECRET:-fallback}", "B": "${S1_DOTENV:-d}", "C": "${S1_SECRET:-}"}
    ) == {"A": "fallback", "B": "d", "C": ""}


def test_named_variable_is_filled_from_the_environment_then_dotenv(
    monkeypatch, tmp_path
):
    dotenv = tmp_path / "operator.env"
    dotenv.write_text("S1_DOTENV=from-dotenv\n")
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))
    _name(monkeypatch, "OTHER", "S1_SECRET", "S1_DOTENV")

    assert _resolve(
        {"A": "${S1_SECRET}", "B": "${S1_SECRET:-fallback}", "C": "${S1_DOTENV}"}
    ) == {"A": SECRET, "B": SECRET, "C": "from-dotenv"}


def test_named_but_unset_variable_without_default_is_still_an_error(monkeypatch):
    _name(monkeypatch, "S1_UNSET")
    with pytest.raises(ValueError, match="not found") as raised:
        _resolve({"KEY": "${S1_UNSET}"})
    assert "--allow-task-env" not in str(raised.value)
    assert _resolve({"KEY": "${S1_UNSET:-d}"}) == {"KEY": "d"}


def test_a_template_inside_a_longer_value_stays_literal():
    assert _resolve({"AUTH": "Bearer ${S1_SECRET}"}) == {"AUTH": "Bearer ${S1_SECRET}"}


def test_the_env_list_takes_commas_and_spaces(monkeypatch):
    _name(monkeypatch, " S1_A ,S1_SECRET  S1_B")
    assert _resolve({"K": "${S1_SECRET}"}) == {"K": SECRET}


def test_allow_task_env_names_variables_for_a_block():
    from benchflow.task.env import allow_task_env

    with allow_task_env(["S1_SECRET"]):
        assert _resolve({"K": "${S1_SECRET}"}) == {"K": SECRET}
    with pytest.raises(ValueError, match=REFUSED):
        _resolve({"K": "${S1_SECRET}"})


# Every place a task env table is filled


def test_sandbox_env_is_refused_when_the_sandbox_is_made(monkeypatch, tmp_path):
    """``[environment] env`` without Compose: BaseSandbox's persistent env."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")

    def make(env: dict[str, str]) -> DockerSandbox:
        return DockerSandbox(environment, "t", "t__1", None, SandboxConfig(env=env))

    with pytest.raises(ValueError, match=REFUSED):
        make({"K": "${S1_SECRET}"})
    assert make({"K": "${S1_SECRET:-d}"})._persistent_env["K"] == "d"
    _name(monkeypatch, "S1_SECRET")
    assert make({"K": "${S1_SECRET}"})._persistent_env["K"] == SECRET


def test_compose_task_env_is_refused_when_the_sandbox_is_made(monkeypatch, tmp_path):
    """``[environment] env`` with a docker-compose.yaml: the Compose env."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    environment = tmp_path / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM alpine:3.20\n")
    (environment / "docker-compose.yaml").write_text("services:\n  main: {}\n")

    def make(env: dict[str, str]) -> DockerSandbox:
        return DockerSandbox(environment, "t", "t__1", None, SandboxConfig(env=env))

    with pytest.raises(ValueError, match=REFUSED):
        make({"K": "${S1_SECRET}"})
    assert make({"K": "${S1_SECRET:-d}"})._compose_task_env == {"K": "d"}
    _name(monkeypatch, "S1_SECRET")
    assert make({"K": "${S1_SECRET}"})._compose_task_env == {"K": SECRET}


def test_daytona_compose_task_env_is_refused(monkeypatch):
    from benchflow.sandbox.daytona_dind import _DaytonaDinD
    from benchflow.task.config import SandboxConfig

    def make(env: dict[str, str]) -> _DaytonaDinD:
        sandbox = SimpleNamespace(
            task_env_config=SandboxConfig(env=env),
            _persistent_env={},
            environment_name="t",
            logger=logging.getLogger("s1-test"),
        )
        return _DaytonaDinD(sandbox)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=REFUSED):
        make({"K": "${S1_SECRET}"})
    assert make({"K": "${S1_SECRET:-d}"})._resolved_task_env == {"K": "d"}
    _name(monkeypatch, "S1_SECRET")
    assert make({"K": "${S1_SECRET}"})._resolved_task_env == {"K": SECRET}


class _SetupEnv:
    def __init__(self) -> None:
        self.exec_calls: list[dict] = []

    async def exec(self, command: str, **kwargs):
        self.exec_calls.append({"command": command, **kwargs})
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _setup_task(env: dict[str, str]):
    from benchflow.task.config import SetupCommandConfig

    return SimpleNamespace(
        config=SimpleNamespace(
            sandbox=SimpleNamespace(
                setup_commands=[SetupCommandConfig(command="true", env=env)]
            )
        )
    )


async def test_setup_command_env_is_refused_before_it_runs(monkeypatch):
    from benchflow.rollout import _run_environment_setup_commands

    sandbox = _SetupEnv()
    with pytest.raises(ValueError, match=REFUSED):
        await _run_environment_setup_commands(
            sandbox, _setup_task({"K": "${S1_SECRET}"})
        )
    assert sandbox.exec_calls == []
    _name(monkeypatch, "S1_SECRET")
    await _run_environment_setup_commands(sandbox, _setup_task({"K": "${S1_SECRET}"}))
    assert sandbox.exec_calls[0]["env"] == {"K": SECRET}


def _oracle_sandbox() -> MagicMock:
    sandbox = MagicMock()

    async def exec_side_effect(cmd, **kwargs):
        return MagicMock(return_code=0, stdout="")

    sandbox.exec = AsyncMock(side_effect=exec_side_effect)
    return sandbox


async def test_oracle_env_is_refused_before_solve_runs(monkeypatch, tmp_path):
    from benchflow.rollout._setup import _run_oracle

    task = _write_task(tmp_path / "task", '[solution.env]\nTOKEN = "${S1_SECRET}"\n')
    sandbox = _oracle_sandbox()
    with pytest.raises(ValueError, match=REFUSED):
        await _run_oracle(sandbox, task, timeout=60)
    assert not any("solve.sh" in c.args[0] for c in sandbox.exec.call_args_list)

    _name(monkeypatch, "S1_SECRET")
    sandbox = _oracle_sandbox()
    await _run_oracle(sandbox, task, timeout=60)
    solve = next(c for c in sandbox.exec.call_args_list if "solve.sh" in c.args[0])
    assert solve.kwargs["env"]["TOKEN"] == SECRET


async def test_test_script_verifier_env_is_refused(monkeypatch, tmp_path):
    from benchflow.task import RolloutPaths, Task, Verifier

    task = Task(_write_task(tmp_path / "task", '[verifier.env]\nK = "${S1_SECRET}"\n'))
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()
    sandbox = MagicMock()
    sandbox.is_mounted = True
    sandbox.upload_dir = AsyncMock()
    sandbox.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout=""))

    with pytest.raises(ValueError, match=REFUSED):
        await Verifier(task, rollout_paths, sandbox)._verify_test_script()
    assert not any(
        "test.sh" in str(call.args[0] if call.args else "")
        and "chmod" not in str(call.args[0])
        for call in sandbox.exec.call_args_list
    )

    _name(monkeypatch, "S1_SECRET")
    # The fake sandbox writes no reward file; what matters is test.sh's env.
    with pytest.raises(Exception, match="reward"):
        await Verifier(task, rollout_paths, sandbox)._verify_test_script()
    envs = [c.kwargs.get("env") or {} for c in sandbox.exec.call_args_list]
    assert any(env.get("K") == SECRET for env in envs)


@patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
async def test_llm_judge_env_is_refused(mock_judge, tmp_path):
    from benchflow.task import RolloutPaths, Verifier
    from tests.test_llm_judge_verifier import _judge_toml, _make_sandbox, _make_task

    toml = _judge_toml() + '\n[verifier.env]\nRESOLVED_KEY = "${S1_SECRET}"\n'
    task = _make_task(tmp_path, toml)
    (task.task_dir / "tests").mkdir()
    (task.task_dir / "tests" / "rubric.json").write_text(
        json.dumps({"criteria": [{"id": "c1", "match_criteria": "ok"}]})
    )
    rollout_paths = RolloutPaths(tmp_path / "rollout")
    rollout_paths.mkdir()

    with pytest.raises(ValueError, match=REFUSED):
        await Verifier(task, rollout_paths, _make_sandbox({"a": "b"})).verify()
    mock_judge.assert_not_awaited()


async def test_criteria_runner_env_is_refused(monkeypatch, tmp_path):
    from tests.test_verifier_strategies import _criteria_runner_verifier, _criteria_toml

    calls: dict = {}
    verifier, _ = _criteria_runner_verifier(
        tmp_path,
        criteria_toml=_criteria_toml("weighted_mean", {"c1": 1.0}),
        reward_payload={"metrics": {"c1": 1.0}},
        calls=calls,
    )
    verifier._task.config.verifier.env = {"K": "${S1_SECRET}"}
    with pytest.raises(ValueError, match=REFUSED):
        await verifier.verify()
    assert calls.get("env") is None  # the runner never ran with an env

    _name(monkeypatch, "S1_SECRET")
    await verifier.verify()
    assert calls["env"]["K"] == SECRET


@patch("benchflow.rewards.llm.call_judge", new_callable=AsyncMock)
async def test_agent_judge_env_is_refused(mock_judge, monkeypatch, tmp_path):
    from tests.test_verifier_strategies import _agent_judge_verifier

    mock_judge.return_value = '{"verdict": "pass", "reasoning": "ok"}'
    verifier, _ = _agent_judge_verifier(tmp_path)
    verifier._task.config.verifier.env = {"K": "${S1_SECRET}"}
    with pytest.raises(ValueError, match=REFUSED):
        await verifier.verify()
    mock_judge.assert_not_awaited()

    _name(monkeypatch, "S1_SECRET")
    await verifier.verify()
    assert mock_judge.await_args.kwargs["env"] == {"K": SECRET}


class _StopAfterEnv(Exception):
    pass


async def test_hardened_verifier_env_is_filled_by_the_allowlist(monkeypatch, tmp_path):
    """harden_before_verify's env merge used os.path.expandvars, which
    filled any $VAR of the operator's environment, inside longer values too."""
    from benchflow._utils.trusted_path import TrustedPathReport
    from benchflow.sandbox import lockdown
    from benchflow.task.config import TaskConfig

    async def path(*args, **kwargs):
        return "/usr/bin:/bin"

    async def no_distro_env(*args, **kwargs):
        return {}

    seen: dict[str, str] = {}

    async def stop(env, task, verifier_env, *args, **kwargs):
        seen.update(verifier_env)
        raise _StopAfterEnv

    async def judged(*args, **kwargs):
        return TrustedPathReport(path="/usr/bin:/bin")

    async def same_report(env, report, *args):
        return report

    # V3 judges the PATH in /bin/sh into a report.
    monkeypatch.setattr(lockdown, "_judge_verifier_path", judged)
    monkeypatch.setattr(lockdown, "_check_required_path_tools", same_report)
    monkeypatch.setattr(lockdown, "_install_refusal_stubs", same_report)
    monkeypatch.setattr(lockdown, "_trusted_verifier_pythonpath", path)
    monkeypatch.setattr(lockdown, "_distro_pip_env", no_distro_env)
    monkeypatch.setattr(lockdown, "_isolate_verifier_tool_state", stop)

    def task(env: dict[str, str]):
        config = TaskConfig.model_validate_toml('version = "1.0"\n[verifier]\n')
        config.verifier.env = env
        return SimpleNamespace(config=config)

    with pytest.raises(ValueError, match=REFUSED):
        await lockdown._build_verifier_env(
            MagicMock(), task({"K": "${S1_SECRET}"}), "agent", "/app"
        )
    with pytest.raises(_StopAfterEnv):
        await lockdown._build_verifier_env(
            MagicMock(),
            task({"K": "x-$S1_SECRET", "L": "${S1_SECRET:-d}"}),
            "agent",
            "/app",
        )
    assert seen["K"] == "x-$S1_SECRET"
    assert seen["L"] == "d"
    _name(monkeypatch, "S1_SECRET")
    with pytest.raises(_StopAfterEnv):
        await lockdown._build_verifier_env(
            MagicMock(), task({"K": "${S1_SECRET}"}), "agent", "/app"
        )
    assert seen["K"] == SECRET


# The task's docker-compose.yaml: Compose's environment (the launch gate's
# `docker compose config` check is in test_compose_oracle.py)


def _compose_task(root: Path, compose: str, toml_body: str = "") -> Path:
    task = _write_task(root, toml_body)
    (task / "environment" / "docker-compose.yaml").write_text(compose)
    return task


_SMALL_COMPOSE = (
    "services:\n  main:\n    environment:\n"
    "      - KEY=${S1_SECRET}\n      - OPT=${S1_DEFAULTED:-x}\n"
    "      - TASK_VALUE=${TASK_KEY}\n"
)


def test_compose_runs_without_unnamed_variables_it_reads(monkeypatch, tmp_path):
    """Compose fills ${VAR:-x} from its own environment, which holds only the
    docker CLI's variables and the named ones, so the default applies."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    task = _compose_task(tmp_path / "task", _SMALL_COMPOSE)
    monkeypatch.setenv("S1_DEFAULTED", "operator-value")
    monkeypatch.setenv("S1_UNREAD", "kept")

    def compose_env() -> dict[str, str]:
        sandbox = DockerSandbox(
            task / "environment",
            "t",
            "t__1",
            None,
            SandboxConfig(env={"TASK_KEY": "from-task"}),
        )
        return sandbox._docker_compose_env()

    env = compose_env()
    assert "S1_DEFAULTED" not in env
    assert "S1_SECRET" not in env
    assert env["TASK_KEY"] == "from-task"
    # Not read by docker-compose.yaml, but a file it points at could read it.
    assert "S1_UNREAD" not in env
    assert env["MAIN_IMAGE_NAME"]
    assert "PATH" in env

    _name(monkeypatch, "S1_DEFAULTED", "S1_SECRET")
    env = compose_env()
    assert env["S1_DEFAULTED"] == "operator-value"
    assert env["S1_SECRET"] == SECRET


# Before any sandbox: the launch gate, the job, the manifest


def test_launch_gate_refuses_before_the_sandbox_exists(tmp_path):
    from benchflow.sandbox.setup import _validate_task_runtime_for_launch
    from benchflow.task import Task

    for index, table in enumerate(
        (
            '[environment.env]\nK = "${S1_SECRET}"\n',
            '[verifier.env]\nK = "${S1_SECRET}"\n',
            '[solution.env]\nK = "${S1_SECRET}"\n',
            '[[environment.setup_commands]]\ncommand = "true"\nenv = { K = "${S1_SECRET}" }\n',
        )
    ):
        path = _write_task(tmp_path / f"task-{index}", table)
        with pytest.raises(ValueError, match=REFUSED):
            _validate_task_runtime_for_launch(
                Task(path), sandbox_type="docker", task_path=path
            )


def _two_task_dir(root: Path) -> Path:
    _write_task(root / "plain")
    _write_task(root / "asks", '[verifier.env]\nJUDGE_KEY = "${S1_SECRET}"\n')
    return root


def _evaluation(tasks: Path, jobs: Path):
    from benchflow.evaluation import Evaluation, EvaluationConfig

    return Evaluation(
        tasks_dir=str(tasks),
        jobs_dir=str(jobs),
        config=EvaluationConfig(agent="oracle", environment="docker"),
        preflight=False,
    )


def test_a_job_is_refused_before_any_trial_or_review_starts(monkeypatch, tmp_path):
    from benchflow import evaluation as evaluation_mod

    started: list[str] = []

    async def no_trial(self, task_dir, cfg):
        started.append(task_dir.name)
        raise AssertionError("a trial started")

    async def no_reviews(*args, **kwargs):
        started.append("reviews")

    monkeypatch.setattr(evaluation_mod.Evaluation, "_run_single_task", no_trial)
    monkeypatch.setattr("benchflow.review.resume.resume_pending_reviews", no_reviews)
    job = _evaluation(_two_task_dir(tmp_path / "tasks"), tmp_path / "jobs")

    with pytest.raises(ValueError, match=REFUSED) as raised:
        asyncio.run(job.run())
    assert "asks: verifier.env.JUDGE_KEY" in str(raised.value)
    assert "plain" not in str(raised.value)
    assert started == []


def test_a_job_naming_the_variable_passes_the_gate(monkeypatch, tmp_path):
    from benchflow.evaluation import _refuse_unnamed_task_env

    tasks = _two_task_dir(tmp_path / "tasks")
    task_dirs = [tasks / "plain", tasks / "asks"]
    with pytest.raises(ValueError, match=REFUSED):
        _refuse_unnamed_task_env(task_dirs, manifest=None, sandbox="docker")
    _name(monkeypatch, "S1_SECRET")
    _refuse_unnamed_task_env(task_dirs, manifest=None, sandbox="docker")


def test_a_task_manifest_forwards_only_named_variables(monkeypatch, tmp_path):
    from benchflow.evaluation import _environment_manifest_from_task_document

    task = tmp_path / "task"
    task.mkdir()
    (task / "environment.toml").write_text(
        '[environment]\nname = "svc"\nimage = "svc:1"\nowns_lifecycle = true\n\n'
        '[environment.forward_env]\nkeys = ["S1_SECRET"]\n'
    )
    (task / "task.md").write_text(
        "---\nschema_version: '1.0'\nbenchflow:\n  environment:\n"
        "    manifest: environment.toml\n---\n\n## prompt\n\nDo it.\n"
    )
    with pytest.raises(ValueError, match=REFUSED):
        _environment_manifest_from_task_document(task)
    _name(monkeypatch, "S1_SECRET")
    manifest = _environment_manifest_from_task_document(task)
    assert manifest is not None
    assert manifest.forward_env.keys == ["S1_SECRET"]


# The CLI


def test_tasks_check_refuses_what_run_refuses_and_the_flag_names_it(tmp_path):
    from benchflow.cli.main import app

    task = _write_task(tmp_path / "task", '[verifier.env]\nK = "${S1_SECRET}"\n')
    refused = CliRunner().invoke(app, ["tasks", "check", str(task)])
    assert refused.exit_code == 1
    flat = " ".join(refused.output.split())
    assert "verifier.env.K" in flat
    assert "--allow-task-env S1_SECRET" in flat
    assert SECRET not in refused.output

    named = CliRunner().invoke(
        app, ["tasks", "check", str(task), "--allow-task-env", "S1_SECRET"]
    )
    assert named.exit_code == 0, named.output


def test_eval_run_refuses_before_any_sandbox(monkeypatch, tmp_path):
    from benchflow import evaluation as evaluation_mod
    from benchflow.cli.main import app

    async def no_trial(self, task_dir, cfg):
        raise AssertionError("a trial started")

    monkeypatch.setattr(evaluation_mod.Evaluation, "_run_single_task", no_trial)
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(_two_task_dir(tmp_path / "tasks")),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ],
    )
    assert result.exit_code == 1
    flat = " ".join(result.output.split())
    assert "S1_SECRET" in flat
    assert "--allow-task-env S1_SECRET" in flat
    assert SECRET not in result.output


def test_eval_run_flag_names_the_variable_for_the_run(monkeypatch, tmp_path):
    import os

    from benchflow.cli import main as cli_main

    seen: list[str | None] = []
    monkeypatch.setattr(
        cli_main,
        "run_batch_eval",
        lambda *args, **kwargs: seen.append(os.environ.get(ALLOW)),
    )
    _name(monkeypatch, "S1_FROM_ENV")
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(_two_task_dir(tmp_path / "tasks")),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--allow-task-env",
            "S1_SECRET",
            "--allow-task-env",
            "S1_OTHER",
        ],
    )
    assert result.exit_code == 0, result.output
    assert seen == ["S1_FROM_ENV,S1_OTHER,S1_SECRET"]


def test_the_flag_takes_a_name_and_never_echoes_a_value(tmp_path):
    """Guards S1's structured next-step contract after integration 7a2387ac."""
    from benchflow.cli.main import app

    result = CliRunner().invoke(
        app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(_two_task_dir(tmp_path / "tasks")),
            "--agent",
            "oracle",
            "--allow-task-env",
            f"S1_SECRET={SECRET}",
        ],
    )
    assert result.exit_code == 2
    assert "takes one variable name" in " ".join(result.output.split())
    assert "next:" in result.output
    assert "repeat the option" in " ".join(result.output.split())
    assert SECRET not in result.output


# Where the names travel


async def test_run_blocking_keeps_the_callers_names():
    """Inside a running loop run_blocking hops to a thread, which starts with
    an empty context; the names a with-block set must still hold there."""
    from benchflow.batch import run_blocking
    from benchflow.task.env import allow_task_env, allowed_task_env

    async def names():
        return allowed_task_env()

    with allow_task_env(["S1_SECRET"]):
        assert "S1_SECRET" in run_blocking(names)


async def test_worker_processes_get_the_names_of_the_callers_block(
    monkeypatch, tmp_path
):
    from benchflow import eval_sharding
    from benchflow.task.env import allow_task_env

    seen: dict[str, str] = {}

    async def fake_exec(*args, env, **kwargs):
        seen.update(env)

        async def no_lines():
            return
            yield

        async def wait():
            return 0

        return SimpleNamespace(stdout=no_lines(), wait=wait)

    monkeypatch.setattr(eval_sharding.asyncio, "create_subprocess_exec", fake_exec)
    with allow_task_env(["S1_SECRET"]):
        await eval_sharding._run_worker_once(tmp_path / "p.json", tmp_path / "w.log")
    assert seen[ALLOW] == "S1_SECRET"


async def test_skill_eval_names_the_judge_keys_its_own_tasks_read(
    monkeypatch, tmp_path
):
    """``bench skills eval`` writes ``${GEMINI_API_KEY}`` and the like into the
    tasks it generates; its run names them, so they still reach the judge."""
    from benchflow.evaluation import Evaluation
    from benchflow.skill_eval._core import JUDGE_API_ENV_KEYS, SkillEvaluator
    from benchflow.task.env import allowed_task_env

    seen: list[frozenset[str]] = []

    async def fake_run(self):
        seen.append(allowed_task_env())

    monkeypatch.setattr(Evaluation, "run", fake_run)
    evaluator = SkillEvaluator.__new__(SkillEvaluator)
    evaluator.dataset = SimpleNamespace(cases=[])
    await evaluator._run_job(
        tmp_path / "tasks",
        "oracle",
        None,
        "docker",
        str(tmp_path / "jobs"),
        1,
        with_skill=False,
    )
    assert set(JUDGE_API_ENV_KEYS) <= seen[0]


# The pre-compose hook, and the judge's endpoint

HOST_ACCESS = "BENCHFLOW_ALLOW_TASK_HOST_ACCESS"


def _gate(path: Path, sandbox_type: str = "docker") -> None:
    from benchflow.sandbox.setup import _validate_task_runtime_for_launch
    from benchflow.task import Task

    _validate_task_runtime_for_launch(
        Task(path), sandbox_type=sandbox_type, task_path=path
    )


def test_compose_gets_only_docker_cli_and_named_variables(monkeypatch):
    from benchflow.task.compose_env import compose_process_env

    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/s1-docker.sock")
    env = compose_process_env()
    assert "S1_SECRET" not in env
    assert "DOCKER_HOST" not in env  # on the command line instead
    assert "PATH" in env
    _name(monkeypatch, "S1_SECRET")
    assert compose_process_env()["S1_SECRET"] == SECRET


def test_a_pre_compose_hook_is_refused_unless_host_access_is_allowed(
    monkeypatch, tmp_path
):
    task = _write_task(tmp_path / "task")
    (task / "environment" / "benchflow-pre-compose.sh").write_text("true\n")
    with pytest.raises(ValueError, match=f"{HOST_ACCESS}=1") as raised:
        _gate(task)
    assert "benchflow-pre-compose.sh runs on this machine" in str(raised.value)
    _gate(task, "daytona")  # it runs inside the sandbox there
    monkeypatch.setenv(HOST_ACCESS, "1")
    _gate(task)


async def test_the_pre_compose_hook_runs_only_when_allowed_without_unnamed_variables(
    monkeypatch, tmp_path
):
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.task.config import SandboxConfig

    task = _write_task(tmp_path / "task")
    environment = task / "environment"
    (environment / "benchflow-pre-compose.sh").write_text(
        'printf "%s|%s" "${S1_SECRET:-unset}" "${S1_NAMED:-unset}" > hook.out\n'
    )
    sandbox = DockerSandbox(environment, "t", "t__1", None, SandboxConfig())
    with pytest.raises(RuntimeError, match=HOST_ACCESS):
        await sandbox._run_pre_compose_hook()
    assert not (environment / "hook.out").exists()

    monkeypatch.setenv(HOST_ACCESS, "1")
    monkeypatch.setenv("S1_NAMED", "named-value")
    _name(monkeypatch, "S1_NAMED")
    await sandbox._run_pre_compose_hook()
    assert (environment / "hook.out").read_text() == "unset|named-value"


_CHAT_COMPLETION = {
    "id": "s1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
}


async def test_a_task_judge_endpoint_never_gets_the_operators_openai_key(
    monkeypatch,
):
    """[verifier] env's literal OPENAI_BASE_URL is the task's endpoint: the
    judge call there carries the task's own key or none, never the
    operator's OPENAI_API_KEY."""
    import httpx
    import openai

    from benchflow.rewards.llm import call_judge

    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization")))
        return httpx.Response(200, json=_CHAT_COMPLETION)

    real = openai.AsyncOpenAI

    def client(**kwargs):
        transport = httpx.MockTransport(handler)
        return real(http_client=httpx.AsyncClient(transport=transport), **kwargs)

    monkeypatch.setattr(openai, "AsyncOpenAI", client)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-operator-s1")
    monkeypatch.setenv("OPENAI_ADMIN_KEY", "sk-operator-admin-s1")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    async def judge(env: dict[str, str]) -> tuple[str, str | None]:
        await call_judge("gpt-4o", "score", retries=1, env=env)
        return seen[-1]

    url, auth = await judge({"OPENAI_BASE_URL": "http://judge.invalid/v1"})
    assert url == "http://judge.invalid/v1/chat/completions"
    assert "sk-operator" not in (auth or "")

    # The task's own key goes to its own endpoint.
    _, auth = await judge(
        {"OPENAI_BASE_URL": "http://judge.invalid/v1", "OPENAI_API_KEY": "sk-task"}
    )
    assert auth == "Bearer sk-task"

    # The operator's own endpoint, or none named, keeps the operator's key.
    monkeypatch.setenv("OPENAI_BASE_URL", "http://mine.invalid/v1")
    assert await judge({"OPENAI_BASE_URL": "http://mine.invalid/v1"}) == (
        "http://mine.invalid/v1/chat/completions",
        "Bearer sk-operator-s1",
    )
    assert await judge({}) == (
        "http://mine.invalid/v1/chat/completions",
        "Bearer sk-operator-s1",
    )

    # OpenAI's own endpoint, pinned by the task, is the provider's, not the
    # task's: the operator's key goes there (S1-R2-A-4).
    monkeypatch.delenv("OPENAI_BASE_URL")
    for pinned in ("https://api.openai.com/v1", "https://api.openai.com/v1/"):
        url, auth = await judge({"OPENAI_BASE_URL": pinned})
        assert url == "https://api.openai.com/v1/chat/completions"
        assert auth == "Bearer sk-operator-s1"


def test_a_dotenv_in_the_working_folder_opts_into_nothing(monkeypatch, tmp_path):
    """``bench`` copies the working folder's .env into its environment; a
    benchmark's own .env must not name variables or allow host access."""
    import os

    from benchflow.cli._shared import _apply_dotenv_to_process_env

    dotenv = tmp_path / "benchmark.env"
    dotenv.write_text(
        f"{ALLOW}=S1_SECRET\n{HOST_ACCESS}=1\nS1_FROM_DOTENV=from-dotenv\n"
    )
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(dotenv))
    for name in (ALLOW, HOST_ACCESS, "S1_FROM_DOTENV"):
        monkeypatch.setenv(name, "x")  # so teardown restores each
        monkeypatch.delenv(name)
    _apply_dotenv_to_process_env()
    assert os.environ["S1_FROM_DOTENV"] == "from-dotenv"
    assert ALLOW not in os.environ
    assert HOST_ACCESS not in os.environ


def _toolathlon_task(task_dir: Path, mcp_env: dict[str, str]) -> Path:
    from benchflow.adapters._toolathlon import _toolathlon_task_toml, _write_toml

    _write_task(task_dir)
    payload = _toolathlon_task_toml(
        SimpleNamespace(repo="hkust-nlp/Toolathlon", resolved_sha="d" * 40),
        task_name="s1-github",
        mcp_servers=[
            {
                "name": "github",
                "transport": "stdio",
                "command": "/usr/bin/python3",
                "args": ["/workspace/.toolathlon/toolathlon_container.py", "launch"],
                "env": mcp_env,
            }
        ],
        variant="official",
        credential_refs=set(),
    )
    _write_toml(task_dir / "task.toml", payload)
    return task_dir


def test_toolathlon_tasks_are_refused_until_the_tokens_they_need_are_named(
    monkeypatch, tmp_path, caplog
):
    """The adapter's token-setup env reads the tokens a task's MCP servers use
    with no default, so the gate names them instead of the task running with
    empty credentials; the other tokens keep a default and warn when set."""
    from benchflow.evaluation import _refuse_unnamed_task_env
    from benchflow.task import Task

    task = _toolathlon_task(
        tmp_path / "task", {"GITHUB_PERSONAL_ACCESS_TOKEN": "${token.github_token}"}
    )
    monkeypatch.setenv("TOOLATHLON_GITHUB_TOKEN", "s1-fake-github")
    monkeypatch.setenv("TOOLATHLON_HF_TOKEN", "s1-fake-hf")
    with pytest.raises(
        ValueError, match="--allow-task-env TOOLATHLON_GITHUB_TOKEN"
    ) as raised:
        _refuse_unnamed_task_env([task], manifest=None, sandbox="docker")
    assert "TOOLATHLON_HF_TOKEN" not in str(raised.value)
    assert "s1-fake-github" not in str(raised.value)

    _name(monkeypatch, "TOOLATHLON_GITHUB_TOKEN")
    caplog.set_level(logging.WARNING, logger="benchflow.evaluation")
    _refuse_unnamed_task_env([task], manifest=None, sandbox="docker")
    assert "TOOLATHLON_HF_TOKEN is set in your environment" in caplog.text
    assert "s1-fake-hf" not in caplog.text
    setup_env = Task(task).config.sandbox.setup_commands[0].env
    filled = _resolve(setup_env)
    assert filled["TOOLATHLON_GITHUB_TOKEN"] == "s1-fake-github"
    assert filled["TOOLATHLON_HF_TOKEN"] == ""

    plain = _toolathlon_task(tmp_path / "plain", {"S1": "literal"})
    _refuse_unnamed_task_env([plain], manifest=None, sandbox="docker")


def test_toolathlons_required_tokens_are_the_containers_placeholder_ones():
    """The host's list of tokens a task cannot work without matches the
    container helper's tokens whose default is a placeholder."""
    import importlib.util

    import benchflow.adapters as adapters
    from benchflow.adapters._toolathlon import _TOOLATHLON_REQUIRED_TOKEN_ENVS

    path = Path(adapters.__file__).parent / "_toolathlon_container.py"
    spec = importlib.util.spec_from_file_location("s1_toolathlon_container", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    placeholders = {
        key: env
        for key, env, default in module._GLOBAL_TOKENS
        if env and default in ("", "XX")
    }
    assert placeholders == _TOOLATHLON_REQUIRED_TOKEN_ENVS


def _toolathlon_upstream(root: Path) -> Path:
    """A minimal upstream Toolathlon checkout; returns tasks/finalpool."""
    (root / ".git").mkdir(parents=True)
    (root / "global_preparation").mkdir()
    mcp_dir = root / "configs" / "mcp_servers"
    mcp_dir.mkdir(parents=True)
    (root / "configs" / "users_data.json").write_text("{}")
    # The notion server's config reads no ${token.notion_integration_key_eval}
    # and no page URL: preprocess and evaluation read those.
    (mcp_dir / "notion.yaml").write_text(
        "type: stdio\nname: notion\nparams:\n  command: npx\n"
        '  args: ["@notionhq/notion-mcp-server"]\n'
    )
    (mcp_dir / "filesystem.yaml").write_text(
        "type: stdio\nname: filesystem\nparams:\n  command: npx\n"
        '  args: ["@modelcontextprotocol/server-filesystem"]\n'
    )
    return root / "tasks" / "finalpool"


def _toolathlon_upstream_task(finalpool: Path, name: str, servers: list[str]) -> Path:
    task_dir = finalpool / name
    (task_dir / "docs").mkdir(parents=True)
    (task_dir / "docs" / "agent_system_prompt.md").write_text("Workspace.\n")
    (task_dir / "docs" / "task.md").write_text(f"Do {name}.\n")
    (task_dir / "task_config.json").write_text(
        json.dumps({"needed_mcp_servers": servers, "needed_local_tools": []})
    )
    return task_dir


def test_toolathlon_tokens_a_tasks_own_code_reads_are_required(monkeypatch, tmp_path):
    """A token read only by a task's preprocess or evaluation code (or by the
    Notion page duplicator every notion task runs) is required too, so the
    task is refused naming it instead of scoring 0 on the XX placeholder."""
    from benchflow._utils.benchmark_repos import ResolvedSource
    from benchflow.adapters.source import adapt_resolved_source_if_needed
    from benchflow.evaluation import _refuse_unnamed_task_env

    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "Toolathlon"
    finalpool = _toolathlon_upstream(repo)
    _toolathlon_upstream_task(finalpool, "s1-notion", ["notion"])
    report = _toolathlon_upstream_task(finalpool, "s1-report", ["filesystem"])
    (report / "evaluation").mkdir()
    (report / "evaluation" / "main.py").write_text(
        "from configs.token_key_session import all_token_key_session as tokens\n"
        "account = tokens.snowflake_account\n"
    )
    _toolathlon_upstream_task(finalpool, "s1-plain", ["filesystem"])
    adapted = adapt_resolved_source_if_needed(
        ResolvedSource(
            path=finalpool,
            provenance={
                "type": "github",
                "repo": "hkust-nlp/Toolathlon",
                "requested_ref": None,
                "resolved_sha": "e" * 40,
                "path": "tasks/finalpool",
                "local_path": str(finalpool),
                "dirty": False,
                "file_hashes": {},
            },
        )
    )

    def refused(name: str) -> str:
        with pytest.raises(ValueError) as raised:
            _refuse_unnamed_task_env(
                [adapted.path / name], manifest=None, sandbox="docker"
            )
        return " ".join(str(raised.value).split())

    notion = refused("s1-notion")
    for variable in (
        "TOOLATHLON_NOTION_KEY",
        "TOOLATHLON_NOTION_KEY_EVAL",
        "TOOLATHLON_NOTION_SOURCE_PAGE_URL",
        "TOOLATHLON_NOTION_EVAL_PAGE_URL",
    ):
        assert f"--allow-task-env {variable}" in notion
    assert "SNOWFLAKE" not in notion
    report_message = refused("s1-report")
    assert "--allow-task-env TOOLATHLON_SNOWFLAKE_ACCOUNT" in report_message
    assert "NOTION" not in report_message
    _refuse_unnamed_task_env(
        [adapted.path / "s1-plain"], manifest=None, sandbox="docker"
    )
