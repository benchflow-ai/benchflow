"""Guards PARITY-3 (gap 3, rows A4, H5, B22).

At b1e738d29 a YAML job config ran a fraction of itself and said nothing:

* only ``agents[0]`` and ``datasets[0]`` ran (``evaluation.py:3201``, ``3280``);
* ``n_attempts: 3`` became one trial with two retries, not three trials;
* the verifier, setup and build timeout multipliers, ``metrics:``, an agent's
  ``kwargs`` and the rest were ignored with a warning, or not read at all
  (``environment.*``, ``agents[].*``, ``datasets[].task_names``).

A YAML job config now runs all of ``agents`` x ``datasets`` x ``n_attempts``,
one job per cell and trial in the ``--matrix --trials`` layout, or is refused
when it loads, by key.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from benchflow.cli.main import app


def _bench(*args: str):
    return CliRunner().invoke(app, list(args), prog_name="bench")


def _plan(result) -> dict[str, Any]:
    text = result.stdout
    return json.loads(text[text.index("{") :])


def _task(root: Path, name: str) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "oracle").mkdir()
    (task / "oracle" / "solve.sh").write_text("#!/bin/bash\necho hi > /app/o\n")
    (task / "verifier").mkdir()
    (task / "verifier" / "test.sh").write_text(
        "#!/bin/bash\necho 1.0 > /logs/verifier/reward.txt\n"
    )
    (task / "task.md").write_text(
        "---\nschema_version: '1.3'\n"
        f"name: {name}\nagent:\n  timeout_sec: 300\n---\nWrite hi to /app/o.\n"
    )
    return task


@pytest.fixture
def project(tmp_path, monkeypatch) -> Path:
    """Two datasets: ds-a (alpha, beta, gamma) and ds-b (delta)."""
    monkeypatch.chdir(tmp_path)
    for name in ("alpha", "beta", "gamma"):
        _task(tmp_path / "ds-a", name)
    _task(tmp_path / "ds-b", "delta")
    return tmp_path


def _config(project: Path, **keys: Any) -> Path:
    job = {
        "agents": [{"name": "oracle"}],
        "datasets": [{"path": "ds-a"}],
        **keys,
    }
    path = project / "config.json"
    path.write_text(json.dumps(job, indent=2))
    return path


@pytest.fixture
def ran(monkeypatch) -> list[dict[str, Any]]:
    """Each job the run starts: its agent, model, tasks, folder and retries.
    The job "runs" by recording itself and passing every task."""
    from benchflow.evaluation import Evaluation, EvaluationResult

    jobs: list[dict[str, Any]] = []

    async def run(self):
        tasks = [d.name for d in self._get_task_dirs()]
        jobs.append(
            {
                "agent": self._config.agent,
                "model": self._config.model,
                "tasks_dir": Path(self._tasks_dir).name,
                "tasks": tasks,
                "jobs_dir": Path(self._jobs_dir).as_posix(),
                "job": self._job_name,
                "max_retries": self._config.retry.max_retries,
                "timeout_multiplier": self._config.timeout_multiplier,
                "config_override": self._config.config_override,
                "checkpoints": self._config.checkpoints,
                "checkpoint_keep": self._config.checkpoint_keep,
                "retry_from_checkpoint": self._config.retry_from_checkpoint,
                "sample_seed": self._config.sample_seed,
                "freeze_workspace": self._config.freeze_workspace,
                "retry_prompt": self._config.retry_prompt,
                "retry_resume_session": self._config.retry_resume_session,
                "sandbox_setup_timeout": self._config.sandbox_setup_timeout,
                "base_image_override": self._config.base_image_override,
                "concurrency": self._config.concurrency,
            }
        )
        return EvaluationResult(
            job_name=self._job_name,
            config=self._config,
            total=len(tasks),
            passed=len(tasks),
        )

    monkeypatch.setattr(Evaluation, "run", run)
    return jobs


# --- agents x datasets ---------------------------------------------------------


def test_two_agents_by_two_datasets_plan_four_job_sets(project) -> None:
    from benchflow.job_config import job_plan

    path = _config(
        project,
        agents=[{"name": "oracle"}, {"name": "nop"}],
        datasets=[{"path": "ds-a"}, {"path": "ds-b"}],
    )
    plan = job_plan(path)
    assert plan is not None and plan.is_matrix
    assert [(c.agent, c.dataset, c.alias) for c in plan.cells] == [
        ("oracle", "ds-a", "oracle__ds-a"),
        ("oracle", "ds-b", "oracle__ds-b"),
        ("nop", "ds-a", "nop__ds-a"),
        ("nop", "ds-b", "nop__ds-b"),
    ]
    assert plan.attempts == 1


def test_two_agents_by_two_datasets_run_four_job_sets(project, ran) -> None:
    path = _config(
        project,
        agents=[{"name": "oracle"}, {"name": "nop"}],
        datasets=[{"path": "ds-a"}, {"path": "ds-b"}],
    )
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    got = sorted((j["agent"], j["tasks_dir"], tuple(j["tasks"])) for j in ran)
    assert got == [
        ("nop", "ds-a", ("alpha", "beta", "gamma")),
        ("nop", "ds-b", ("delta",)),
        ("oracle", "ds-a", ("alpha", "beta", "gamma")),
        ("oracle", "ds-b", ("delta",)),
    ]
    # Each combination has its own job records, named for it.
    assert sorted(j["jobs_dir"] for j in ran) == [
        "jobs/nop__ds-a/trial-01",
        "jobs/nop__ds-b/trial-01",
        "jobs/oracle__ds-a/trial-01",
        "jobs/oracle__ds-b/trial-01",
    ]
    summary = json.loads((project / "jobs" / "matrix-summary.json").read_text())
    assert summary["complete"] is True and summary["trials"] == 1
    assert [c["alias"] for c in summary["job_cells"]] == [
        "oracle__ds-a", "oracle__ds-b", "nop__ds-a", "nop__ds-b",
    ]  # fmt: skip
    assert sorted((r["alias"], r["total"]) for r in summary["runs"]) == [
        ("nop__ds-a", 3), ("nop__ds-b", 1), ("oracle__ds-a", 3), ("oracle__ds-b", 1),
    ]  # fmt: skip
    assert "Matrix: 4 trial job(s), 8/8 trials passed" in " ".join(
        result.output.split()
    )


def test_the_dry_run_of_a_matrix_lists_every_job(project, ran) -> None:
    path = _config(
        project,
        agents=[{"name": "oracle"}, {"name": "nop"}],
        datasets=[{"path": "ds-a"}, {"path": "ds-b"}],
        n_attempts=2,
    )
    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 0, result.output
    plan = _plan(result)
    assert [(j["cell"], j["agent"], j["tasks"]) for j in plan["jobs"]] == [
        ("oracle__ds-a", "oracle", ["alpha", "beta", "gamma"]),
        ("oracle__ds-b", "oracle", ["delta"]),
        ("nop__ds-a", "nop", ["alpha", "beta", "gamma"]),
        ("nop__ds-b", "nop", ["delta"]),
    ]
    assert (plan["trials"], plan["trial_jobs"], plan["rollouts"]) == (2, 8, 16)
    assert ran == []  # a dry run runs nothing


def test_a_model_agent_names_its_model_in_the_cell(project) -> None:
    from benchflow.job_config import job_plan

    path = _config(
        project,
        agents=[
            {"name": "codex", "model_name": "openai/gpt-a"},
            {"name": "codex", "model_name": "openai/gpt-b"},
        ],
    )
    plan = job_plan(path)
    assert plan is not None
    assert [c.alias for c in plan.cells] == [
        "codex__openai-gpt-a__ds-a",
        "codex__openai-gpt-b__ds-a",
    ]


def test_several_agents_refuse_an_agent_override(project, ran) -> None:
    path = _config(project, agents=[{"name": "oracle"}, {"name": "nop"}])
    for flags in (["-a", "oracle"], ["--model", "openai/gpt-a"]):
        result = _bench("run", "--config", str(path), *flags)
        assert result.exit_code == 2, result.output
        text = " ".join(result.output.split())
        assert "its agents: list names 2 agents" in text, text
    assert ran == []


def test_one_agent_one_dataset_one_attempt_is_one_job_as_before(project, ran) -> None:
    path = _config(project, job_name="solo")
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    assert [(j["jobs_dir"], j["job"], j["tasks"]) for j in ran] == [
        ("jobs", "solo", ["alpha", "beta", "gamma"])
    ]
    assert not (project / "jobs" / "matrix-summary.json").exists()


def test_from_python_one_evaluation_never_runs_the_first_of_several(project) -> None:
    from benchflow.evaluation import Evaluation

    path = _config(project, datasets=[{"path": "ds-a"}, {"path": "ds-b"}])
    # By behaviour first (review L-A2): at b1e738d29 this returned the first
    # cell's Evaluation, and ds-b never ran.
    try:
        loaded = Evaluation.from_yaml(path, preflight=False)
    except Exception as exc:
        refused = exc
    else:
        pytest.fail(f"from_yaml returned one job of two: {loaded._tasks_dir}")
    assert "runs 2 job(s)" in str(refused), refused
    from benchflow.evaluation import JobMatrixError

    assert isinstance(refused, JobMatrixError)
    assert "job_cell" in (refused.hint or "")
    second = Evaluation.from_yaml(path, preflight=False, job_cell=1)
    assert Path(second._tasks_dir).name == "ds-b"
    with pytest.raises(ValueError, match="job_cell 2: the config has 2 cell"):
        Evaluation.from_yaml(path, preflight=False, job_cell=2)


# --- n_attempts ------------------------------------------------------------------


def test_n_attempts_three_is_three_trials_per_task(project, ran) -> None:
    path = _config(project, n_attempts=3, job_name="pass3")
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    assert [(j["jobs_dir"], j["job"]) for j in ran] == [
        ("jobs/oracle__ds-a/trial-01", "pass3"),
        ("jobs/oracle__ds-a/trial-02", "pass3"),
        ("jobs/oracle__ds-a/trial-03", "pass3"),
    ]
    # Three trials of every task, and never a retry.
    assert all(j["tasks"] == ["alpha", "beta", "gamma"] for j in ran)
    assert {j["max_retries"] for j in ran} == {0}
    summary = json.loads((project / "jobs" / "matrix-summary.json").read_text())
    assert summary["trials"] == 3
    assert [r["trial"] for r in summary["runs"]] == [1, 2, 3]

    from benchflow.job_config import job_plan

    plan = job_plan(path)
    assert plan is not None and plan.attempts == 3 and len(plan.cells) == 1


def test_a_job_config_never_retries(project) -> None:
    from benchflow.evaluation import Evaluation

    for keys in ({}, {"n_attempts": 1}):
        j = Evaluation.from_yaml(_config(project, **keys), preflight=False)
        assert j._config.retry.max_retries == 0


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "3"])
def test_a_bad_n_attempts_is_refused(project, ran, value) -> None:
    result = _bench("run", "--config", str(_config(project, n_attempts=value)))
    assert result.exit_code == 2, result.output
    assert "n_attempts" in result.output and "whole number" in result.output
    assert ran == []


# --- what is honoured exactly ----------------------------------------------------


def test_the_agent_timeout_multipliers_are_job_config(project) -> None:
    from benchflow.evaluation import Evaluation

    def multiplier(**keys: Any) -> float | None:
        path = _config(project, **keys)
        return Evaluation.from_yaml(path, preflight=False)._config.timeout_multiplier

    assert multiplier(agent_timeout_multiplier=2.0) == 2.0
    ones = {
        "verifier_timeout_multiplier": 1,
        "agent_setup_timeout_multiplier": 1.0,
        "environment_build_timeout_multiplier": 1,
    }
    assert (
        multiplier(timeout_multiplier=2.0, agent_timeout_multiplier=3.0, **ones) == 3.0
    )
    assert multiplier(timeout_multiplier=2.5, **ones) == 2.5
    assert multiplier(timeout_multiplier=1.0) == 1.0


def test_a_datasets_own_task_selection_is_applied(project, ran) -> None:
    path = _config(
        project,
        datasets=[
            {"path": "ds-a", "task_names": ["*a"], "exclude_task_names": ["gamma"]},
            {"path": "ds-a", "n_tasks": 2},
        ],
    )
    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 0, result.output
    assert [(j["cell"], j["tasks"]) for j in _plan(result)["jobs"]] == [
        ("oracle__ds-a", ["alpha", "beta"]),
        ("oracle__ds-a-2", ["alpha", "beta"]),
    ]
    one = _config(project, datasets=[{"path": "ds-a", "task_names": ["beta"]}])
    result = _bench("run", "--config", str(one), "--dry-run")
    assert _plan(result)["tasks"] == ["beta"]


def test_an_agents_env_and_mean_metric_and_extra_files_run(
    project, monkeypatch
) -> None:
    from benchflow.evaluation import Evaluation

    monkeypatch.setenv("PARITY3_TOKEN", "from-shell")
    monkeypatch.delenv("PARITY3_UNSET", raising=False)
    (project / "extra.md").write_text("Read the README first.")
    path = _config(
        project,
        agents=[
            {
                "name": "oracle",
                "env": {
                    "A": "${PARITY3_TOKEN}",
                    "B": "${PARITY3_UNSET:-fallback}",
                    "C": "plain",
                },
            }
        ],
        metrics=[{"type": "mean"}],
        extra_instruction_paths=["extra.md"],
        extra_instructions=["Be brief."],
        quiet=True,
    )
    j = Evaluation.from_yaml(path, preflight=False)
    assert {k: j._config.agent_env[k] for k in "ABC"} == {
        "A": "from-shell",
        "B": "fallback",
        "C": "plain",
    }
    assert j._config.extra_instruction == "Read the README first.\n\nBe brief."

    missing = _config(
        project, agents=[{"name": "oracle", "env": {"A": "${PARITY3_UNSET}"}}]
    )
    with pytest.raises(
        ValueError, match=r"agents\[0\]\.env\.A: \$\{PARITY3_UNSET\} is not set"
    ):
        Evaluation.from_yaml(missing, preflight=False)


def test_job_config_spelled_out_defaults_still_load(project) -> None:
    """Fully expanded default blocks must load."""
    from benchflow.evaluation import Evaluation
    from benchflow.job_config import JOB_DEFAULTS

    path = _config(
        project,
        **JOB_DEFAULTS,
        timeout_multiplier=1.0,
        agent_timeout_multiplier=None,
        verifier_timeout_multiplier=None,
        agent_setup_timeout_multiplier=None,
        environment_build_timeout_multiplier=None,
        metrics=[],
        quiet=False,
        extra_instruction_paths=[],
        extra_instructions=[],
        n_concurrent_trials=4,
        environment={"type": "docker", "delete": True, "force_build": False, "env": {}},
        orchestrator={
            "type": "local",
            "n_concurrent_trials": 4,
            "retry": {"max_retries": 0},
        },
    )
    Evaluation.from_yaml(path, preflight=False)


# --- each setting that cannot be honoured is refused, by name -------------------

REFUSED = [
    ({"timeout_multiplier": 2.0}, "timeout_multiplier: 2.0: BenchFlow scales the agent, verifier and environment build timeouts with it, but not the agent setup timeout"),
    ({"agent_setup_timeout_multiplier": 3}, "agent_setup_timeout_multiplier: 3: BenchFlow installs agents with its own time budget"),
    ({"agents": [{"name": "oracle", "n_concurrent": 5}]}, "agents[0].n_concurrent: 5: cannot exceed n_concurrent_trials (4)"),
    ({"agents": [{"name": "oracle", "concurrency_group": "g"}]}, "agents[0].concurrency_group: 'g': an agent in a concurrency group must set n_concurrent"),
    ({"agents": [{"name": "oracle", "n_concurrent": 1, "concurrency_group": "g"}, {"name": "nop", "n_concurrent": 2, "concurrency_group": "g"}]}, "agents[1].n_concurrent: 2: agents in concurrency_group 'g' must use the same n_concurrent (1)"),
    ({"metrics": [{"type": "sum"}]}, "metrics[0]:"),
    ({"metrics": [{"type": "uv-script", "kwargs": {"script_path": "m.py"}}]}, "metrics[0]:"),
    ({"metrics": [{"type": "mean", "kwargs": {"x": 1}}]}, "metrics[0]:"),
    ({"retry": {"max_retries": 2}}, "retry: {'max_retries': 2}"),
    ({"orchestrator": {"retry": {"max_retries": 1}}}, "orchestrator.retry:"),
    ({"orchestrator": {"type": "ray"}}, "orchestrator.type: 'ray'"),
    ({"debug": True}, "debug: True"),
    ({"install_only": True}, "install_only: True"),
    ({"verifier": {"disable": True}}, "verifier: {'disable': True}"),
    ({"verifier": {"override_timeout_sec": 30}}, "verifier:"),
    ({"artifacts": ["/app/out.txt"]}, "artifacts:"),
    ({"user_agent": {"name": "oracle", "bridge": {}}}, "user_agent:"),
    ({"source_jobs": [{"action": "regrade", "type": "local", "path": "jobs/x"}]}, "source_jobs:"),
    ({"extra_instruction_paths": ["missing.md"]}, "extra_instruction_paths[0]: 'missing.md': file not found"),
    ({"environment": {"type": "docker", "env": {"K": "v"}}}, "environment.env:"),
    ({"environment": {"type": "docker", "env": ["K=v"]}}, "environment.env:"),
    ({"environment": {"type": "docker", "override_cpus": 4}}, "environment.override_cpus: 4"),
    ({"environment": {"type": "docker", "mounts": [{"type": "bind", "source": "/tmp/victim-a", "target": "/x"}]}}, "environment.mounts:"),
    ({"environment": {"type": "docker", "cpu_count": 4}}, "environment.cpu_count: unknown key"),
    ({"agents": [{"name": "oracle", "kwargs": {"max_turns": 5}}]}, "agents[0].kwargs:"),
    ({"agents": [{"name": "oracle", "override_timeout_sec": 60}]}, "agents[0].override_timeout_sec: 60"),
    ({"agents": [{"name": "oracle", "max_timeout_sec": 60}]}, "agents[0].max_timeout_sec: 60"),
    ({"agents": [{"name": "oracle", "skills": ["skills/x"]}]}, "agents[0].skills:"),
    ({"agents": [{"name": "oracle", "import_path": "my.agent:Agent"}]}, "agents[0].import_path:"),
    ({"agents": [{"name": "oracle", "mcp_servers": [{"name": "m"}]}]}, "agents[0].mcp_servers:"),
    ({"agents": [{"name": "oracle"}, {"name": "nop", "temperature": 0}]}, "agents[1].temperature: unknown key"),
    ({"agents": [{"name": "oracle", "env": ["K=v"]}]}, "agents[0].env: must be a mapping"),
    ({"datasets": [{"path": "ds-a", "overwrite": True}]}, "datasets[0].overwrite: True"),
    ({"datasets": [{"path": "ds-a", "download_dir": "/tmp/victim-a"}]}, "datasets[0].download_dir:"),
    ({"datasets": [{"path": "ds-a", "task_name": ["alpha"]}]}, "datasets[0].task_name: unknown key (did you mean 'task_names'?)"),
    ({"datasets": [{"path": "ds-a", "n_tasks": 0}]}, "datasets[0].n_tasks: 0: must be a whole number"),
]  # fmt: skip


@pytest.mark.parametrize(
    ("keys", "named"), REFUSED, ids=[text.split(":")[0] for _, text in REFUSED]
)
def test_each_setting_benchflow_cannot_honour_is_refused_by_name(
    project, ran, keys, named
) -> None:
    path = _config(project, **keys)
    result = _bench("run", "--config", str(path))
    text = " ".join(result.output.split())
    assert result.exit_code == 2, result.output
    assert named in text, text
    assert "config.json:" in text  # by file and line
    assert "Traceback" not in result.output
    assert ran == []  # refused before anything ran
    # The dry run and the Python loader refuse it too.
    assert _bench("run", "--config", str(path), "--dry-run").exit_code == 2
    from benchflow._utils.config_keys import ConfigKeyError
    from benchflow.evaluation import Evaluation

    with pytest.raises(ConfigKeyError):
        Evaluation.from_yaml(path, preflight=False)


@pytest.mark.parametrize(
    ("dataset", "named"),
    [
        ({"name": "terminal-bench", "version": "2.0"}, "datasets[0] has no path"),
        ({"path": "ds-a", "repo": "org/tasks"}, "datasets[0].repo: names a remote dataset"),
        ({"path": "ds-a", "registry_url": "https://example.invalid/r.json"}, "datasets[0].registry_url: names a remote dataset"),
    ],
)  # fmt: skip
def test_a_remote_dataset_is_refused_with_the_way_to_run_it(
    project, ran, dataset, named
) -> None:
    path = _config(project, datasets=[{"path": "ds-b"}, dataset])
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 1, result.output
    assert named.replace("[0]", "[1]") in " ".join(result.output.split())
    assert ran == []


def test_every_problem_is_named_at_once(project, ran) -> None:
    path = _config(
        project,
        agent_setup_timeout_multiplier=2,
        metrics=[{"type": "max"}],
        agents=[{"name": "oracle", "kwargs": {"a": 1}}],
    )
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    for key in ("agent_setup_timeout_multiplier:", "metrics[0]:", "agents[0].kwargs:"):
        assert key in text, (key, text)
    assert "next: change or drop each setting named" in text


# --- round 2 -------------------------------------------------------------------


def test_three_cells_have_three_unique_job_paths(project, ran) -> None:
    """Review B-H1: datasets ds-a, ds-a and ds-a-2 named two cells
    oracle__ds-a-2, which shared one job folder."""
    _task(project / "ds-a-2", "epsilon")
    path = _config(
        project,
        job_name="collision",
        datasets=[{"path": "ds-a"}, {"path": "ds-a"}, {"path": "ds-a-2"}],
    )
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    first = [(j["jobs_dir"], j["tasks_dir"]) for j in ran]
    assert len({d for d, _ in first}) == 3, first
    # The natural -2 name stays the ds-a-2 cell's; the repeat takes -3.
    assert first == [
        ("jobs/oracle__ds-a/trial-01", "ds-a"),
        ("jobs/oracle__ds-a-3/trial-01", "ds-a"),
        ("jobs/oracle__ds-a-2/trial-01", "ds-a-2"),
    ]
    # A rerun of the same file resumes each cell in its own folder.
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    assert [(j["jobs_dir"], j["tasks_dir"]) for j in ran[3:]] == first


def test_a_yaml_matrix_keeps_every_cell_and_attempt(project, ran) -> None:
    """Review B's probe: 2 agents x 2 datasets x n_attempts 3 is 12 jobs."""
    path = project / "config.yaml"
    path.write_text(
        "agents:\n  - name: oracle\n  - name: nop\n"
        "datasets:\n  - path: ds-a\n  - path: ds-b\nn_attempts: 3\n"
    )
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    assert len(ran) == 12 and len({j["jobs_dir"] for j in ran}) == 12


def test_a_stopped_matrix_says_how_to_resume_the_whole_matrix(
    project, monkeypatch
) -> None:
    """Review A M-A1 / B-M2: a stop printed `bench eval resume <one trial
    job>`, which finished that job and left the rest of the matrix unrun."""
    import shlex

    from benchflow.cli._termination import RunStopped
    from benchflow.evaluation import Evaluation, EvaluationResult

    calls: list[str] = []

    async def run(self):
        done = self.job_dir / "done"
        calls.append(str(self.job_dir))
        if not done.exists():
            self.job_dir.mkdir(parents=True, exist_ok=True)
            if len(calls) == 2:
                raise RunStopped("Terminated (SIGTERM)", 143)
            done.write_text("")
        return EvaluationResult(
            job_name=self._job_name, config=self._config, total=1, passed=1
        )

    monkeypatch.setattr(Evaluation, "run", run)
    path = _config(
        project,
        job_name="stop",
        agents=[{"name": "oracle"}, {"name": "nop"}],
        datasets=[{"path": "ds-a"}, {"path": "ds-b"}],
    )
    stopped = _bench("run", "--config", str(path))
    assert stopped.exit_code == 143, stopped.output
    text = " ".join(stopped.output.split())
    assert "bench eval resume" not in text, text
    hint = "run the same command again to finish the whole matrix (3 of 4 trial jobs not finished; finished ones are kept): "
    assert hint in text, text
    summary_path = project / "jobs" / "matrix-summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["complete"] is False and len(summary["runs"]) == 1
    command = text.split(hint, 1)[1].split(" Matrix summary")[0].strip()
    assert command == f"bench run --config {path}", command

    resumed = _bench(*shlex.split(command)[1:])
    assert resumed.exit_code == 0, resumed.output
    summary = json.loads(summary_path.read_text())
    assert summary["complete"] is True and summary["resume"] is None
    assert sorted(r["alias"] for r in summary["runs"]) == [
        "nop__ds-a", "nop__ds-b", "oracle__ds-a", "oracle__ds-b",
    ]  # fmt: skip
    # Every cell ran to its end: the stopped one resumed, the rest started.
    assert len(set(calls)) == 4 and all((Path(c) / "done").exists() for c in set(calls))


def test_an_agents_own_concurrency_cap_runs(project, ran) -> None:
    """Review B-M3: the documented agents[].n_concurrent was refused."""
    path = _config(
        project,
        n_concurrent_trials=4,
        agents=[
            {"name": "oracle", "n_concurrent": 2},
            {"name": "nop", "n_concurrent": 2, "concurrency_group": "g"},
        ],
    )
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    assert [j["concurrency"] for j in ran] == [2, 2]
    one = _config(project, agents=[{"name": "oracle", "n_concurrent": 2}])
    dry = _bench("run", "--config", str(one), "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert _plan(dry)["concurrency"] == 2


@pytest.mark.parametrize(
    ("agent", "flags", "want"),
    [
        ({"name": "oracle"}, ["--agent", "nop"], ("nop", None, "nop__ds-a")),
        (
            {"name": "codex", "model_name": "openai/gpt-a"},
            ["--model", "openai/gpt-b"],
            ("codex-acp", "openai/gpt-b", "codex__openai-gpt-b__ds-a"),
        ),
    ],
)
def test_an_allowed_override_is_what_the_matrix_records(
    project, ran, agent, flags, want
) -> None:
    """Review B-M4: --agent nop on a one-agent file ran nop, while
    matrix-summary.json recorded an oracle cell."""
    path = _config(project, agents=[agent], n_attempts=2)
    result = _bench("run", "--config", str(path), *flags)
    assert result.exit_code == 0, result.output
    assert {(j["agent"], j["model"]) for j in ran} == {want[:2]}
    cells = json.loads((project / "jobs" / "matrix-summary.json").read_text())[
        "job_cells"
    ]
    assert [(c["agent"], c["model"], c["alias"]) for c in cells] == [want]
    assert {j["jobs_dir"].split("/")[1] for j in ran} == {want[2]}
    dry = _bench("run", "--config", str(path), *flags, "--dry-run")
    assert [(j["agent"], j["model"], j["cell"]) for j in _plan(dry)["jobs"]] == [want]


def test_every_cell_is_scored_at_k_n_attempts_controls_included(
    project, ran, monkeypatch
) -> None:
    """Review B-M1 and B-L1: control cells (oracle, nop) had no pass@k, and
    n_attempts 3 gave k = 1 and 2 only."""
    from benchflow.cli import eval_artifacts

    asked: list[dict[str, Any]] = []

    def rates(root, aliases, *, include_controls=False, ks=None):
        asked.append({"aliases": aliases, "controls": include_controls, "ks": ks})
        return {alias: {"ks": ks} for alias in aliases}

    monkeypatch.setattr(eval_artifacts, "matrix_solve_rates", rates)
    path = _config(project, agents=[{"name": "oracle"}, {"name": "nop"}], n_attempts=3)
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    assert asked == [
        {"aliases": ["oracle__ds-a", "nop__ds-a"], "controls": True, "ks": [1, 2, 3]}
    ]
    summary = json.loads((project / "jobs" / "matrix-summary.json").read_text())
    assert summary["solve_rates"]["nop__ds-a"]["ks"] == [1, 2, 3]


def test_the_solve_rates_count_controls_at_the_asked_ks(tmp_path) -> None:
    """The emitted pass@k itself, from real job records: an oracle job of
    3 trials of one task scores pass@3 = 1 with controls included."""
    from benchflow.cli.eval_artifacts import matrix_solve_rates
    from benchflow.evaluation import Evaluation, EvaluationConfig

    root = tmp_path / "jobs"
    for trial in (1, 2, 3):
        job = root / "oracle__ds" / f"trial-{trial:02d}" / "j"
        rollout = job / "alpha__1"
        rollout.mkdir(parents=True)
        (rollout / "config.json").write_text(json.dumps({"agent": "oracle"}))
        (rollout / "result.json").write_text(
            json.dumps({"task_name": "alpha", "rewards": {"reward": 1.0}})
        )
        Evaluation(
            tmp_path, jobs_dir=job.parent, job_name="j",
            config=EvaluationConfig(agent="oracle"), preflight=False,
        )._write_evaluation_record()  # fmt: skip
    got = matrix_solve_rates(root, ["oracle__ds"], include_controls=True, ks=[1, 2, 3])
    assert got["oracle__ds"]["ks"] == [1, 2, 3], got
    assert got["oracle__ds"]["pass_at_k"]["3"] == 1.0, got
    left_out = matrix_solve_rates(root, ["oracle__ds"])
    assert left_out["oracle__ds"]["pass_at_k"] != got["oracle__ds"]["pass_at_k"]


def test_a_cell_that_selects_no_task_stops_the_run_before_it_starts(
    project, ran
) -> None:
    """Review L-A3: --include alpha on two datasets planned the ds-b cells
    with 0 rollouts and said nothing."""
    path = _config(
        project,
        agents=[{"name": "oracle"}, {"name": "nop"}],
        datasets=[{"path": "ds-a"}, {"path": "ds-b"}],
    )
    for extra in (["--dry-run"], []):
        result = _bench("run", "--config", str(path), "--include", "alpha", *extra)
        assert result.exit_code == 1, result.output
        text = " ".join(result.output.split())
        assert "2 of 4 cells select no task (oracle__ds-b, nop__ds-b)" in text, text
    assert ran == []


def test_job_config_timeout_multiplier_scales_agent_verifier_and_build(project) -> None:
    """Ruling on review A L-A1: map timeout_multiplier where it maps exactly
    (agent, verifier, build) and refuse only the agent setup phase."""
    from benchflow._utils.config_override import apply_config_override
    from benchflow.evaluation import Evaluation
    from benchflow.task import Task

    path = _config(project, timeout_multiplier=1.5, agent_setup_timeout_multiplier=1)
    j = Evaluation.from_yaml(path, preflight=False)
    assert j._config.timeout_multiplier == 1.5
    overlay = j._config.config_override
    assert overlay == {"timeout_multipliers": {"verifier": 1.5, "build": 1.5}}
    task = Task(project / "ds-a" / "alpha").config
    scaled = apply_config_override(task, overlay)
    assert scaled.verifier.timeout_sec == task.verifier.timeout_sec * 1.5
    assert scaled.sandbox.build_timeout_sec == task.sandbox.build_timeout_sec * 1.5

    own = _config(
        project,
        verifier_timeout_multiplier=2,
        environment_build_timeout_multiplier=0.5,
        agent_timeout_multiplier=3,
    )
    j = Evaluation.from_yaml(own, preflight=False)
    assert j._config.timeout_multiplier == 3
    assert j._config.config_override == {
        "timeout_multipliers": {"verifier": 2.0, "build": 0.5}
    }
    # The build phase is always there: it bounds the environment start (B2-M1).
    plain = Evaluation.from_yaml(_config(project), preflight=False)._config
    assert plain.config_override == {"timeout_multipliers": {"build": 1.0}}


def test_a_cli_config_override_keeps_the_files_timeout_multipliers(
    project, ran
) -> None:
    path = _config(project, verifier_timeout_multiplier=2)
    result = _bench(
        "run", "--config", str(path), "--config-override", '{"metadata": {"k": "v"}}'
    )
    assert result.exit_code == 0, result.output
    assert ran[0]["config_override"] == {
        "metadata": {"k": "v"},
        "timeout_multipliers": {"verifier": 2.0, "build": 1.0},
    }
    clash = _bench(
        "run", "--config", str(path),
        "--config-override", '{"timeout_multipliers": {"verifier": 3}}',
    )  # fmt: skip
    assert clash.exit_code == 1, clash.output
    assert "both scale the timeouts" in " ".join(clash.output.split())


@pytest.mark.parametrize("layout", ["toml", "native"])
def test_config_runs_take_n_tasks_sample_seed_and_retries_from_the_line(
    project, ran, layout
) -> None:
    """Review A's pre-existing medium: --n-tasks, --sample-seed and
    --retry-attempts were dropped on any --config run (it ran every task)."""
    if layout == "toml":
        path = _config(project)
    else:
        path = project / "native.yaml"
        path.write_text("tasks_dir: ds-a\nagent: oracle\n")
    dry = _bench("run", "--config", str(path), "--n-tasks", "1", "--dry-run")
    assert dry.exit_code == 0, dry.output
    assert (_plan(dry)["n_tasks"], _plan(dry)["tasks"]) == (1, ["alpha"])
    result = _bench(
        "run", "--config", str(path), "--n-tasks", "2", "--sample-seed", "7",
        "--retry-attempts", "5",
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert len(ran[0]["tasks"]) == 2 and ran[0]["max_retries"] == 5


# --- round 3 -------------------------------------------------------------------


async def test_environment_start_obeys_scaled_build_phase_budget(project) -> None:
    """Review B2-M1 (B's probe): the whole environment start is bounded by
    the scaled build budget; the scaled budget reached only the image build.
    The probe asked for a TimeoutError; the error is a sandbox startup error
    instead, because Rollout.run reads a TimeoutError as the agent's timeout."""
    import asyncio
    import time
    from types import SimpleNamespace

    from benchflow._utils.config_override import apply_config_override
    from benchflow.rollout._setup import _start_env_and_upload
    from benchflow.task.config import TaskConfig

    cfg = apply_config_override(
        TaskConfig(), {"timeout_multipliers": {"build": 0.00005}}
    )

    async def start(**kwargs):
        await asyncio.sleep(0.10)

    task = project / "ds-a" / "alpha"
    env = SimpleNamespace(start=start, task_env_config=cfg.sandbox)
    then = time.monotonic()
    # By behaviour first: at db8fd328a the 100 ms start ran to its end.
    try:
        await _start_env_and_upload(env, task, {}, upload_task_files=False)
    except Exception as exc:
        stopped = exc
    else:
        pytest.fail("the environment start ran past its 0.03 s build budget")
    assert time.monotonic() - then < 0.09
    from benchflow.sandbox.protocol import (
        EnvironmentStartTimeoutError,
        SandboxStartupError,
    )

    assert isinstance(stopped, EnvironmentStartTimeoutError), stopped
    assert "did not start within" in str(stopped)
    assert isinstance(stopped, SandboxStartupError)
    assert not isinstance(stopped, TimeoutError)
    assert cfg.sandbox.start_deadline_sec == pytest.approx(0.03)

    # No build multiplier in the overlay: no bound of its own (as before).
    free = SimpleNamespace(start=start, task_env_config=TaskConfig().sandbox)
    await _start_env_and_upload(free, task, {}, upload_task_files=False)


def test_a_job_configs_start_is_bounded_by_its_build_budget(project) -> None:
    """Every job cell's sandbox carries the bound, multiplier 1 included;
    the image build keeps its own budget."""
    from benchflow._utils.config_override import apply_config_override
    from benchflow.evaluation import Evaluation
    from benchflow.task import Task

    task = Task(project / "ds-a" / "alpha").config
    for keys, factor in (
        ({}, 1.0),
        ({"environment_build_timeout_multiplier": 0.5}, 0.5),
    ):
        j = Evaluation.from_yaml(_config(project, **keys), preflight=False)
        sandbox = apply_config_override(task, j._config.config_override).sandbox
        assert sandbox.build_timeout_sec == task.sandbox.build_timeout_sec * factor
        assert sandbox.start_deadline_sec == sandbox.build_timeout_sec


def _native(project: Path, extra: str) -> Path:
    path = project / "native.yaml"
    path.write_text("tasks_dir: ds-a\nagent: oracle\n" + extra)
    return path


def test_checkpoint_keep_overrides_file_without_repeating_policy(project, ran) -> None:
    """Review B2-M2 (B's probe): --checkpoint-keep 7 kept the file's 2."""
    path = _native(project, "checkpoints: every-prompt\ncheckpoint_keep: 2\n")
    result = _bench("run", "--config", str(path), "--checkpoint-keep", "7")
    assert result.exit_code == 0, result.output
    assert (ran[0]["checkpoints"], ran[0]["checkpoint_keep"]) == ("every-prompt", 7)


def test_repeating_checkpoint_policy_keeps_unspecified_file_retention(
    project, ran
) -> None:
    """Review B2-M2 (B's probe; a round-2 regression): --checkpoints alone
    reset the file's checkpoint_keep 7 to the flag default 3."""
    path = _native(project, "checkpoints: every-prompt\ncheckpoint_keep: 7\n")
    result = _bench("run", "--config", str(path), "--checkpoints", "prompt:1")
    assert result.exit_code == 0, result.output
    assert (ran[0]["checkpoints"], ran[0]["checkpoint_keep"]) == ("prompt:1", 7)


def test_retry_uses_checkpoint_policy_from_file(project, ran) -> None:
    """Review B2-M2: --retry-from-checkpoint was refused ("needs
    --checkpoints") before the file that sets checkpoints was read."""
    path = _native(project, "checkpoints: every-prompt\n")
    result = _bench(
        "run", "--config", str(path), "--retry-from-checkpoint", "on-failure"
    )
    assert result.exit_code == 0, result.output
    assert ran[0]["retry_from_checkpoint"] == "on-failure"
    # With checkpoints nowhere, it is still refused, before anything runs.
    bare = _native(project, "")
    refused = _bench(
        "run", "--config", str(bare), "--retry-from-checkpoint", "on-failure"
    )
    assert refused.exit_code == 1, refused.output
    assert "needs checkpoints" in " ".join(refused.output.split())
    assert len(ran) == 1
    folder = _bench(
        "run", "ds-a", "-a", "oracle", "--retry-from-checkpoint", "on-failure"
    )
    assert folder.exit_code == 2, folder.output
    assert "needs --checkpoints" in " ".join(folder.output.split())


def test_a_flag_left_out_keeps_the_files_value(project, ran) -> None:
    """Every flag round 2 applied to --config runs (B2-M2's audit): left out,
    it keeps the file's value; only a given flag wins. PARITY-3 r4 B3-L2
    guards the four missing assertions at 74aebf8a2."""
    path = _native(
        project,
        "checkpoints: every-prompt\ncheckpoint_keep: 5\nn_tasks: 2\nmax_retries: 4\n"
        "freeze_workspace: true\nretry_from_checkpoint: on-failure\n"
        "retry_prompt: again\nretry_resume_session: true\nsample_seed: 19\n",
    )
    result = _bench("run", "--config", str(path))
    assert result.exit_code == 0, result.output
    got = ran[0]
    assert got["checkpoints"] == "every-prompt"
    assert got["checkpoint_keep"] == 5
    assert len(got["tasks"]) == 2
    assert got["max_retries"] == 4
    assert got["retry_from_checkpoint"] == "on-failure"
    assert got["sample_seed"] == 19
    assert got["freeze_workspace"] is True
    assert got["retry_prompt"] == "again"
    assert got["retry_resume_session"] is True


def test_the_same_agent_twice_must_share_one_cap(project, ran) -> None:
    """Review B2-L1: an implicit pool gives the same agent one cap."""
    path = _config(
        project,
        n_concurrent_trials=4,
        agents=[
            {"name": "oracle", "n_concurrent": 1},
            {"name": "oracle", "n_concurrent": 2},
        ],
    )
    result = _bench("run", "--config", str(path), "--dry-run")
    assert result.exit_code == 2, result.output
    text = " ".join(result.output.split())
    assert (
        "agents[1].n_concurrent: 2: agents[0] is the same agent with n_concurrent 1"
        in text
    ), text
    for agents in (
        [{"name": "oracle", "n_concurrent": 2}, {"name": "oracle", "n_concurrent": 2}],
        [{"name": "oracle", "n_concurrent": 1}, {"name": "nop", "n_concurrent": 2}],
        [{"name": "oracle"}, {"name": "oracle", "kwargs": {}}],
    ):
        ok = _bench(
            "run", "--config", str(_config(project, agents=agents)), "--dry-run"
        )
        assert ok.exit_code == 0, (agents, ok.output)
    unset = _config(
        project, agents=[{"name": "oracle"}, {"name": "oracle", "n_concurrent": 2}]
    )
    assert _bench("run", "--config", str(unset), "--dry-run").exit_code == 2
    assert ran == []
