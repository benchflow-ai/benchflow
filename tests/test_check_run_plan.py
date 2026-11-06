"""Guards PY8 D3 (base 89c99021): check must not bless a refused run plan.

Pending K2/S1/J1 cases are active contracts, not skips: when the run acquires
a refusal, check must immediately print that same exception's message.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app
from benchflow.evaluation import Evaluation, EvaluationConfig
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.sandbox.setup import _validate_task_runtime_for_launch
from benchflow.task import Task
from benchflow.task.formats import materialize_task_dir
from tests._taskmd_helpers import EXAMPLES


def _task(root: Path, config: str = "") -> Path:
    root.mkdir(parents=True)
    for folder in ("environment", "verifier", "oracle"):
        (root / folder).mkdir()
    (root / "task.toml").write_text('version = "1.0"\n' + config)
    (root / "instruction.md").write_text("Write the answer.\n")
    (root / "environment" / "Dockerfile").write_text("FROM alpine:3.20\n")
    (root / "verifier" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    (root / "oracle" / "solve.sh").write_text("#!/bin/sh\nexit 0\n")
    return root


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_TASK_FORMAT_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS", raising=False)
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_ENV", raising=False)
    monkeypatch.delenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", raising=False)
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_OAUTH_TOKEN",
        "BENCHFLOW_TASKMD_JUDGE_MODEL",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


class _SandboxBoundary(Exception):
    pass


def _run_refusal(
    task_dir: Path,
    tmp_path: Path,
    *,
    model=None,
    resolved_env=None,
    agent="claude-agent-acp",
    sandbox_user="agent",
) -> str | None:
    """Run the real setup; stop at the factory, without constructing a sandbox."""
    planes = MagicMock()
    planes.resolve_locked_paths.return_value = []
    planes.resolve_agent_env.return_value = resolved_env or {}
    planes.agent_launch.return_value = "claude-agent-acp"

    def boundary(*args, **kwargs):
        task = Task(task_dir)
        _validate_task_runtime_for_launch(
            task, sandbox_type="docker", task_path=task_dir
        )
        # S1's existing run gate is absent at the fixed D3 base. No skips or
        # expected failures: its landing changes the required result below.
        from benchflow import evaluation

        gate = getattr(evaluation, "_refuse_unnamed_task_env", None)
        if gate is not None:
            gate([task_dir], manifest=None, sandbox="docker")
        raise _SandboxBoundary

    planes.create_environment.side_effect = boundary
    try:
        native = materialize_task_dir(task_dir)
        rollout = Rollout(
            RolloutConfig.from_legacy(
                task_path=native,
                agent=agent,
                model=model,
                sandbox_user=sandbox_user,
                jobs_dir=tmp_path / "jobs",
                planes=planes,
            )
        )
        asyncio.run(rollout.setup())
    except _SandboxBoundary:
        return None
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    raise AssertionError("setup bypassed the sandbox boundary")


@pytest.mark.parametrize("command", [["tasks", "check"], ["check"]])
@pytest.mark.parametrize(
    "config, marker",
    [
        ('[[steps]]\nname = "first"\n', "steps:"),
        (
            '[verifier]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.invalid"]\n',
            "verifier.network_mode:",
        ),
    ],
)
def test_runtime_refusal_is_verbatim_without_a_backend_flag(
    tmp_path, command, config, marker
):
    """Guards D3 against structural check passing a parsed runtime refusal."""
    task = _task(tmp_path / "refused", config)
    reason = _run_refusal(task, tmp_path)
    assert reason is not None and marker in reason
    result = CliRunner().invoke(app, [*command, str(task)])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


def test_judge_launch_refusal_is_verbatim(tmp_path):
    """Guards D3 against passing a rubric whose judge cannot be served."""
    task = tmp_path / "judge"
    shutil.copytree(EXAMPLES / "analysis-judge", task)
    reason = _run_refusal(task, tmp_path)
    assert reason is not None and "[verifier.judges]" in reason
    result = CliRunner().invoke(app, ["check", str(task)])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


def test_proxy_extra_refusal_is_verbatim(tmp_path, monkeypatch):
    """Guards D3/U2a against check accepting a conflicting proxy extra."""
    task = _task(tmp_path / "extra")
    monkeypatch.setenv("BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS", "litellm==1.80.0")
    model = "anthropic/claude-haiku-4-5"
    reason = _run_refusal(task, tmp_path, model=model)
    assert reason is not None and "conflicts with the locked" in reason
    result = CliRunner().invoke(app, ["check", str(task), "--model", model])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


def test_native_login_does_not_validate_unused_proxy_extras(tmp_path, monkeypatch):
    """Guards D3 against validating proxy extras before the auth route is resolved."""
    task = _task(tmp_path / "native-login")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "dummy-native-login")
    monkeypatch.setenv("BENCHFLOW_LITELLM_EXTRA_REQUIREMENTS", "litellm==1.80.0")
    model = "anthropic/claude-haiku-4-5"
    assert (
        _run_refusal(
            task,
            tmp_path,
            model=model,
            resolved_env={"ANTHROPIC_AUTH_TOKEN": "dummy-native-login"},
        )
        is None
    )
    result = CliRunner().invoke(app, ["check", str(task), "--model", model])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "config",
    [
        '[sandbox.env]\nD3_INPUT = "${D3_UNNAMED}"\n',
        '[metadata]\nx-keep-services = "not-a-list"\n',
        '[metadata]\nx-keep-services = [{ port = 0, name = "bad" }]\n',
    ],
)
def test_pending_branch_refusals_activate_without_skips(tmp_path, config):
    """Guards D3's S1/K2 merge: a new run refusal can never leave check green."""
    task = _task(tmp_path / "pending", config)
    reason = _run_refusal(task, tmp_path)
    result = CliRunner().invoke(app, ["check", str(task)])
    if reason is None:
        assert result.exit_code == 0, result.output
    else:
        assert result.exit_code == 1, result.output
        assert reason in result.stderr


def test_s1_job_gate_is_shared_when_it_lands(tmp_path, monkeypatch):
    """Guards D3's S1 bridge even before S1's parser and policy have landed."""
    from benchflow import evaluation

    task = _task(tmp_path / "s1-gate")
    reason = "refused: the task asks for D3_UNNAMED from your environment"

    def refuse(task_dirs, *, manifest, sandbox):
        assert task_dirs == [task]
        assert sandbox == "docker"
        raise ValueError(reason)

    monkeypatch.setattr(evaluation, "_refuse_unnamed_task_env", refuse, raising=False)
    assert _run_refusal(task, tmp_path) == reason
    result = CliRunner().invoke(app, ["check", str(task)])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


def test_check_uses_the_selected_agent_context(tmp_path):
    """Guards D3: agent-only refusals apply unless check names an oracle seat."""
    task = tmp_path / "skills"
    shutil.copytree(EXAMPLES / "hello-world", task)
    path = task / "task.md"
    path.write_text(
        path.read_text().replace(
            'timeout = "2m"', 'timeout = "2m"\n\n[sandbox]\nskills = "/skills"'
        )
    )
    reason = _run_refusal(task, tmp_path)
    assert reason is not None and "[sandbox] skills" in reason
    result = CliRunner().invoke(app, ["check", str(task)])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr
    result = CliRunner().invoke(app, ["check", str(task), "--agent", "oracle"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "isolation, network", [("shared", "none"), ("separate", "open")]
)
def test_pending_j1_judge_plan_refusal_is_verbatim(tmp_path, isolation, network):
    """Guards D3/J1's pending isolation/network plan refusals with real packages."""
    from tests._taskmd_helpers import FIXTURES

    task = tmp_path / "judge-plan"
    shutil.copytree(FIXTURES / "vectors/judge-prompt-1/agent-basic/task", task)
    path = task / "task.md"
    path.write_text(
        path.read_text().replace(
            "[verifier.judges.agent]",
            f'[verifier]\nisolation = "{isolation}"\nnetwork = "{network}"\n\n[verifier.judges.agent]',
        )
    )
    reason = _run_refusal(task, tmp_path)
    assert reason is not None
    result = CliRunner().invoke(app, ["check", str(task)])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


def test_j1_agent_judge_plan_stays_fail_closed(tmp_path):
    """Guards D3/J1: source planning and the judge launch gate are both used."""
    from tests._taskmd_helpers import FIXTURES

    task = tmp_path / "agent-judge"
    shutil.copytree(FIXTURES / "vectors/judge-prompt-1/agent-basic/task", task)
    reason = _run_refusal(task, tmp_path)
    assert reason is not None
    result = CliRunner().invoke(app, ["check", str(task)])
    assert result.exit_code == 1, result.output
    assert reason in result.stderr


@pytest.mark.parametrize("draft2", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_symlinked_source_is_reported_and_skipped_before_materializing(
    tmp_path,
    caplog,
    draft2,
    batch,
):
    """Guards D3/REPORT B9: do not copy through a task's linked oracle."""
    root = tmp_path / "tasks"
    task = root / "linked"
    if draft2:
        shutil.copytree(EXAMPLES / "hello-world", task)
    else:
        _task(task)
    solve = task / "oracle" / "solve.sh"
    solve.rename(solve.with_name("real.sh"))
    solve.symlink_to("real.sh")
    source = root if batch else task
    result = CliRunner().invoke(app, ["check", str(source)])
    assert result.exit_code == 1, result.output
    assert "symlinks" in result.stderr and "oracle/solve.sh" in result.stderr
    job = Evaluation(
        tasks_dir=source,
        jobs_dir=tmp_path / "jobs",
        preflight=False,
        config=EvaluationConfig(agent="oracle"),
    )
    assert job._get_task_dirs() == []
    assert "Skipping task 'linked'" in caplog.text
    assert "oracle/solve.sh" in caplog.text
    assert not list((tmp_path / "cache").rglob("task.toml"))


def test_task_folder_checks_every_child_and_fails_if_any_is_bad(tmp_path):
    """Guards D3/REPORT B18: continue past one refusal to check its siblings."""
    root = tmp_path / "tasks"
    _task(root / "a-good")
    _task(root / "b-refused", '[[steps]]\nname = "first"\n')
    _task(root / "c-good")
    result = CliRunner().invoke(app, ["check", str(root)])
    assert result.exit_code == 1, result.output
    assert "a-good — valid" in result.stdout
    assert "c-good — valid" in result.stdout
    assert "steps:" in result.stderr
    assert "Missing required file: task.toml" not in result.output


def test_folder_of_good_tasks_passes(tmp_path):
    root = tmp_path / "tasks"
    _task(root / "one")
    _task(root / "two")
    result = CliRunner().invoke(app, ["tasks", "check", str(root)])
    assert result.exit_code == 0, result.output
    assert result.stdout.count("valid (structural)") == 2


@pytest.mark.parametrize(
    "pin, ignored, expected",
    [
        (">=0,<999", False, 0),
        ("<0.0.1", False, 1),
        ("<0.0.1", True, 0),
    ],
)
def test_dry_run_reports_registry_pin_without_fetching_tasks(
    tmp_path,
    monkeypatch,
    pin,
    ignored,
    expected,
):
    """Guards D3/REPORT B1: dry-run cannot hide the dataset version gate."""
    from benchflow._utils import benchmark_repos

    def no_clone(*args, **kwargs):
        raise AssertionError("dry-run cloned the task source")

    monkeypatch.setattr(benchmark_repos, "resolve_source_with_metadata", no_clone)
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "name": "d3",
                    "version": "1",
                    "bench_version": pin,
                    "tasks": [{"path": "tasks/example"}],
                }
            ]
        )
    )
    args = [
        "eval",
        "run",
        "--dataset",
        "d3@1",
        "--registry",
        str(registry),
        "--agent",
        "oracle",
        "--dry-run",
    ]
    if ignored:
        args.append("--ignore-bench-version")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == expected, result.output
    output = json.loads(result.stdout)
    assert output["bench_version"] == pin
    assert output["ignore_bench_version"] is ignored
    assert bool(output["bench_version_issue"]) == (pin == "<0.0.1")
    if expected:
        assert output["bench_version_issue"] in result.stderr
        assert "--ignore-bench-version" in result.stderr
