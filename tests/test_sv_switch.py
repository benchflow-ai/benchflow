"""One switch: ``--verifier-sandbox separate`` for any run, refused at plan time.

- The switch (flag, evaluation YAML, rollout YAML, ``EvaluationConfig``,
  ``RolloutConfig``, the shard payload) moves a shared task's verifier into the
  separate verifier sandbox, and the trial records it: ``config.json``
  ``verifier_sandbox``, ``result.json`` ``verifier_isolation`` and its source.
  It writes no integrity audit; ``--integrity strict`` implies it.
- There is no ``shared`` value, and no setting turns a declared separate task
  shared.
- A run that forces separate checks every task before a job folder, a shard
  plan or any sandbox exists, and names every refused task with its reason and
  next step; ``bench tasks check --verifier-sandbox separate`` prints the same
  lines.
- A resume compares the switch (strict counts as separate), and the job
  summary counts trials per mode and warns about run-flag tasks with no
  declared artifacts.

The new names are imported inside the tests, so this file still loads on a
tree without them and each test fails there on behavior.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import benchflow as bf
from benchflow.cli import main as cli_main
from benchflow.evaluation import Evaluation, EvaluationConfig, ResumeMismatchError
from benchflow.models import RolloutResult
from benchflow.rollout import Rollout, RolloutConfig

# --- fixtures ------------------------------------------------------------------

_TEST_SH = "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"


def _task(
    root: Path,
    name: str,
    verifier: str = "",
    *,
    tail: str = "",
    tests_dockerfile: bool = True,
    artifacts: str = "",
) -> Path:
    """A task folder: ``[verifier]`` gets ``verifier``; ``tail`` goes last."""
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "solution").mkdir()
    (task / "environment" / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /app\n"
    )
    (task / "tests" / "test.sh").write_text(_TEST_SH)
    if tests_dockerfile:
        (task / "tests" / "Dockerfile").write_text(
            "FROM ubuntu:24.04\nCOPY test.sh /tests/test.sh\n"
        )
    (task / "solution" / "solve.sh").write_text("#!/bin/sh\ntrue\n")
    (task / "instruction.md").write_text("Write the answer.\n")
    (task / "task.toml").write_text(
        'version = "1.0"\n'
        + artifacts
        + "[agent]\ntimeout_sec = 60\n"
        + "[verifier]\ntimeout_sec = 60\n"
        + verifier
        + "[environment]\n"
        + tail
    )
    return task


#: One task of each kind the plan gate refuses on Docker (the Modal backend is
#: its own case below), keyed by task name, with a phrase of its reason.
_REFUSED: dict[str, tuple[dict[str, Any], str]] = {
    "llm-judge": ({"verifier": 'type = "llm-judge"\n'}, "test-script verifiers only"),
    "side-service": (
        {"verifier": 'service = "grader"\n'},
        "verifier.service must be 'main'",
    ),
    "allowlist-verifier": (
        {
            "verifier": '[verifier.sandbox]\ndocker_image = "ubuntu:24.04"\n'
            'network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]\n'
        },
        "not enforced in a separate verifier sandbox",
    ),
    "no-image": (
        {"verifier": "[verifier.sandbox]\ncpus = 1\n", "tests_dockerfile": False},
        "no verifier image",
    ),
    "multi-step": (
        {"tail": '[[steps]]\nname = "one"\n[[steps]]\nname = "two"\n'},
        "cannot use a separate verifier sandbox",
    ),
    "tests-artifact": (
        {"artifacts": 'artifacts = ["/tests/conftest.py"]\n'},
        "overlaps verifier-owned '/tests'",
    ),
    # DF-4: the verifier sandbox has none of the agent's kept services.
    "kept-services": (
        {
            "artifacts": "[metadata]\n"
            'x-keep-services = [{ port = 8000, name = "wordfreq-http" }]\n'
        },
        "port 8000 (wordfreq-http) (from the task), and a separate verifier "
        "sandbox has none of them",
    ),
}


def _refused_set(root: Path) -> Path:
    """A tasks folder with every refused kind and one task the switch can run."""
    tasks = root / "tasks"
    tasks.mkdir()
    for name, (kwargs, _) in _REFUSED.items():
        _task(tasks, name, **kwargs)
    _task(tasks, "fine")
    return tasks


def _no_sandbox(monkeypatch) -> list[Any]:
    """Record any rollout or sandbox that starts (none may)."""
    started: list[Any] = []

    async def create(config):
        started.append(config)
        raise AssertionError("a rollout started")

    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    return started


class _AgentEnv:
    """A stand-in agent sandbox that records what is uploaded into it."""

    def __init__(self) -> None:
        self.uploads: list[str] = []

    async def upload_dir(self, source, target, *args, **kwargs):
        self.uploads.append(str(target))

    async def upload_file(self, source, target, *args, **kwargs):
        self.uploads.append(str(target))

    async def exec(self, *args, **kwargs):
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _patch_verify(monkeypatch) -> list[str]:
    """verify() with the capture and both verifier paths faked; returns the
    path taken. The shared path uploads tests/ into the agent's sandbox, as
    the real ``_verify_rollout`` does."""
    import benchflow.rollout as rollout_mod

    calls: list[str] = []

    async def fake_separate(rollout):
        calls.append("separate")
        return {"reward": 1.0}, None

    async def fake_shared(env, task, paths, *args, **kwargs):
        calls.append("shared")
        await env.upload_dir(Path(task.paths.tests_dir), "/tests")
        return {"reward": 1.0}, None, None

    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(rollout_mod, "run_separate_verifier", fake_separate)
    monkeypatch.setattr(rollout_mod, "_verify_rollout", fake_shared)
    monkeypatch.setattr(rollout_mod, "capture_terminal_workspace", noop)
    monkeypatch.setattr(rollout_mod, "collect_rollout_artifacts", noop)
    monkeypatch.setattr(rollout_mod, "_publish_trajectory_for_verifier", noop)
    monkeypatch.setattr(rollout_mod, "require_safe_branch_world", lambda r: None)
    monkeypatch.setattr(
        "benchflow.rollout._verifier_recovery.mark_solver_complete", lambda r: None
    )
    return calls


def _trial(config: RolloutConfig, monkeypatch) -> tuple[Rollout, _AgentEnv, list[str]]:
    """Set up and verify one trial with a fake agent sandbox; write result.json."""
    calls = _patch_verify(monkeypatch)
    rollout = Rollout(config)
    env = _AgentEnv()
    rollout._env = env  # a caller-supplied sandbox: setup() builds none
    asyncio.run(rollout.setup())
    rollout._agent_cwd = "/app"
    rollout._trajectory = [{"type": "oracle"}]
    asyncio.run(Rollout.verify(rollout))
    result = rollout._build_result()
    rollout._write_integrity(result)
    return rollout, env, calls


def _records(rollout: Rollout) -> tuple[Any, Any, Any]:
    trial = Path(rollout._rollout_dir)
    config = json.loads((trial / "config.json").read_text())
    result = json.loads((trial / "result.json").read_text())
    return (
        config.get("verifier_sandbox"),
        result.get("verifier_isolation"),
        result.get("verifier_isolation_source"),
    )


def _rollout_config(task: Path, jobs: Path, **fields: Any) -> RolloutConfig:
    """A RolloutConfig with ``fields`` set after construction, as the
    evaluation sets integrity and the switch on the config it builds."""
    config = RolloutConfig(
        task_path=task, agent="oracle", environment="docker", jobs_dir=jobs
    )
    for key, value in fields.items():
        setattr(config, key, value)
    return config


# --- the switch takes the separate path, and the records say so ---------------------


def test_the_switch_moves_a_shared_task_and_records_why(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path, "shared-task")
    rollout, env, calls = _trial(
        _rollout_config(task, tmp_path / "jobs", verifier_sandbox="separate"),
        monkeypatch,
    )
    assert calls == ["separate"]
    assert env.uploads == []  # tests/ never enters the agent's sandbox
    assert _records(rollout) == ("separate", "separate", "run-flag")
    # The switch is not the audit: no integrity record is written.
    assert not (Path(rollout._rollout_dir) / "integrity").exists()
    config = json.loads((Path(rollout._rollout_dir) / "config.json").read_text())
    assert "integrity" not in config


def test_strict_integrity_is_the_source_when_it_moves_the_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path, "shared-task")
    rollout, env, calls = _trial(
        _rollout_config(task, tmp_path / "jobs", integrity="strict"), monkeypatch
    )
    assert calls == ["separate"]
    assert env.uploads == []
    assert _records(rollout) == ("task", "separate", "integrity-strict")
    # Both strict and the switch: strict is named, since it adds the audit.
    task2 = _task(tmp_path, "shared-task-2")
    rollout2, _, _ = _trial(
        _rollout_config(
            task2, tmp_path / "jobs2", integrity="strict", verifier_sandbox="separate"
        ),
        monkeypatch,
    )
    assert _records(rollout2) == ("separate", "separate", "integrity-strict")


def test_task_setting_leaves_every_task_as_declared(
    tmp_path: Path, monkeypatch
) -> None:
    shared = _task(tmp_path, "shared-task")
    rollout, env, calls = _trial(
        _rollout_config(shared, tmp_path / "jobs", verifier_sandbox="task"),
        monkeypatch,
    )
    assert calls == ["shared"]
    assert env.uploads == ["/tests"]  # the shared verifier, as before
    assert _records(rollout) == ("task", "shared", "task")

    declared = _task(tmp_path, "separate-task", 'environment_mode = "separate"\n')
    rollout, env, calls = _trial(
        _rollout_config(declared, tmp_path / "jobs2", verifier_sandbox="task"),
        monkeypatch,
    )
    assert calls == ["separate"]
    assert _records(rollout) == ("task", "separate", "task")


# --- every way to set it gives the same records -------------------------------------


def _evaluation_rollout_config(
    evaluation: Evaluation, task: Path, monkeypatch
) -> RolloutConfig:
    """The RolloutConfig the evaluation builds for ``task``."""
    seen: list[RolloutConfig] = []

    class _Stop(Exception):
        pass

    async def create(config):
        seen.append(config)
        raise _Stop

    monkeypatch.setattr("benchflow.rollout.Rollout.create", create)
    with pytest.raises(_Stop):
        asyncio.run(evaluation._run_single_task(task, evaluation._config))
    monkeypatch.undo()
    return seen[0]


def test_flag_yaml_rollout_yaml_and_config_give_the_same_records(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    task = _task(tasks, "shared-task")
    configs: dict[str, EvaluationConfig] = {}

    async def fake_run(self):
        configs["flag"] = self._config
        raise SystemExit(0)

    monkeypatch.setattr(bf.Evaluation, "run", fake_run)
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs-flag"),
            "--verifier-sandbox",
            "separate",
        ],
        env={"BENCHFLOW_SKIP_PREFLIGHT": "1"},
    )
    assert "flag" in configs, result.output
    flag_config = configs["flag"]
    assert flag_config.verifier_sandbox == "separate"

    # bench run (D1's verb) takes the same flag.
    configs.clear()
    CliRunner().invoke(
        cli_main.app,
        [
            "run",
            str(tasks),
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(tmp_path / "jobs-run"),
            "--verifier-sandbox",
            "separate",
        ],
        env={"BENCHFLOW_SKIP_PREFLIGHT": "1"},
    )
    assert configs["flag"].verifier_sandbox == "separate"

    # --verifier-sandbox on top of --config wins over the file.
    configs.clear()
    plain = Evaluation(
        tasks_dir=tasks, jobs_dir=tmp_path / "jobs-x", config=EvaluationConfig()
    ).to_yaml(tmp_path / "plain.yaml")
    CliRunner().invoke(
        cli_main.app,
        ["eval", "run", "--config", str(plain), "--verifier-sandbox", "separate"],
        env={"BENCHFLOW_SKIP_PREFLIGHT": "1"},
    )
    assert configs["flag"].verifier_sandbox == "separate"
    monkeypatch.undo()

    # The evaluation YAML (_from_native_yaml), and to_dict's round trip.
    direct = EvaluationConfig(agent="oracle")
    direct.verifier_sandbox = "separate"
    evaluation = Evaluation(
        tasks_dir=tasks, jobs_dir=tmp_path / "jobs-yaml", config=direct
    )
    assert evaluation.to_dict()["verifier_sandbox"] == "separate"
    path = evaluation.to_yaml(tmp_path / "job.yaml")
    from_yaml = Evaluation.from_yaml(path)
    assert from_yaml._config.verifier_sandbox == "separate"
    assert from_yaml.to_dict() == evaluation.to_dict() | {
        "jobs_dir": from_yaml.to_dict()["jobs_dir"]
    }

    # The rollout YAML (yaml_loader), and RolloutConfig.to_dict's round trip.
    rollout_cfg = RolloutConfig.from_dict(
        {
            "task_path": str(task),
            "agent": "oracle",
            "environment": "docker",
            "verifier_sandbox": "separate",
        }
    )
    assert rollout_cfg.verifier_sandbox == "separate"
    assert RolloutConfig.from_dict(rollout_cfg.to_dict()).verifier_sandbox == (
        "separate"
    )

    records = {}
    for source, cfg in (
        ("flag", flag_config),
        ("yaml", from_yaml._config),
        ("config", direct),
    ):
        evaluation = Evaluation(
            tasks_dir=tasks, jobs_dir=tmp_path / f"jobs-{source}", config=cfg
        )
        built = _evaluation_rollout_config(evaluation, task, monkeypatch)
        built.jobs_dir = tmp_path / f"trials-{source}"
        rollout, _, _ = _trial(built, monkeypatch)
        monkeypatch.undo()
        records[source] = _records(rollout)
    rollout_cfg.jobs_dir = tmp_path / "trials-rollout-yaml"
    rollout, _, _ = _trial(rollout_cfg, monkeypatch)
    records["rollout-yaml"] = _records(rollout)
    assert set(records.values()) == {("separate", "separate", "run-flag")}, records


def test_the_task_runtime_passes_the_switch_to_its_rollout(tmp_path: Path) -> None:
    from benchflow.rollout.task_runtime import TaskRuntimeConfig

    config = TaskRuntimeConfig(task_path=_task(tmp_path, "shared-task"))
    assert config.to_rollout_config().verifier_sandbox == "task"
    config.verifier_sandbox = "separate"
    assert config.to_rollout_config().verifier_sandbox == "separate"


# --- sharded runs ---------------------------------------------------------------------


def test_the_shard_payload_carries_the_switch() -> None:
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config

    config = EvaluationConfig(agent="oracle")
    config.verifier_sandbox = "separate"
    shard = EvalShard(index=0, task_names=("task",), concurrency=1)
    payload = _config_payload(config, shard=shard)
    assert payload.get("verifier_sandbox") == "separate"
    assert _evaluation_config(payload).verifier_sandbox == "separate"


def test_the_run_card_fields_reach_sharded_workers() -> None:
    """A sharded worker builds its own run card: without the accepted reason
    a strict reference refused every worker of a run that had given one."""
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config

    config = EvaluationConfig(
        agent="oracle",
        accept_deviations="a harness check",
        planned_trials=4,
        trial_number=2,
    )
    shard = EvalShard(index=0, task_names=("task",), concurrency=1)
    worker = _evaluation_config(_config_payload(config, shard=shard))
    assert (worker.accept_deviations, worker.planned_trials, worker.trial_number) == (
        "a harness check",
        4,
        2,
    )


#: EvaluationConfig fields that do not cross the shard payload, each reviewed:
#: a new field fails the test below until it crosses or is listed here.
_NOT_IN_SHARD_PAYLOAD = {
    # Per process: each worker sets its own Docker build limit.
    "build_concurrency": "a per-process Docker build limit",
    # Task selection is resolved in the parent: a shard gets its task names.
    "n_tasks": "the parent samples; the shard's include_tasks are the result",
    "sample_seed": "the parent samples; the shard's include_tasks are the result",
    # Python callables cannot cross a process boundary; refused before sharding.
    "pre_agent_hooks": "refused by run_sharded_evaluation",
    # Recorded by the parent only.
    "agent_defaulted": "the CLI's own bookkeeping",
    # Known gaps, reported as findings (SV2.md, "Left"): not fixed here.
    "budget": "known gap: a per-job cap cannot be split across workers",
    "seeds": "known gap: workers do not re-materialize seeded variants",
}


def _payload_keys() -> set[str]:
    from benchflow.eval_sharding import EvalShard, _config_payload

    shard = EvalShard(index=0, task_names=("task",), concurrency=1)
    return set(_config_payload(EvaluationConfig(agent="oracle"), shard=shard))


def _unreviewed(config_type: type, payload_keys: set[str]) -> set[str]:
    fields = {f.name for f in dataclasses.fields(config_type)}
    # usage_tracking crosses flattened (UsageTrackingConfig.to_mapping()).
    crossed = payload_keys | {"usage_tracking"}
    return fields - crossed - set(_NOT_IN_SHARD_PAYLOAD)


def test_every_config_field_crosses_the_shard_payload_or_is_reviewed() -> None:
    keys = _payload_keys()
    assert "verifier_sandbox" in keys
    unreviewed = _unreviewed(EvaluationConfig, keys)
    assert not unreviewed, (
        f"EvaluationConfig field(s) {sorted(unreviewed)} do not reach sharded "
        "workers: add each to eval_sharding._config_payload and "
        "eval_worker._evaluation_config, or to _NOT_IN_SHARD_PAYLOAD with why"
    )
    # A field added without review is caught.
    extended = dataclasses.make_dataclass(
        "Extended", [("new_setting", str, "x")], bases=(EvaluationConfig,)
    )
    assert _unreviewed(extended, keys) == {"new_setting"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("timeout_multiplier", 2.0),
        ("extra_instruction", "Answer in French."),
        ("skip_agent_install", True),
        ("dataset_name", "bench"),
        ("dataset_version", "1.0"),
        ("dataset_task_digests", {"task": "sha256:ab"}),
    ],
)
def test_settings_that_change_a_trial_reach_sharded_workers(
    field: str, value: Any
) -> None:
    from benchflow.eval_sharding import EvalShard, _config_payload
    from benchflow.eval_worker import _evaluation_config

    config = EvaluationConfig(agent="oracle")
    setattr(config, field, value)
    shard = EvalShard(index=0, task_names=("task",), concurrency=1)
    payload = json.loads(json.dumps(_config_payload(config, shard=shard)))
    assert getattr(_evaluation_config(payload), field) == value


# --- no shared value, and no setting weakens a task -----------------------------------


@pytest.mark.parametrize("value", ["shared", "SHARED", "off", "strict", "yes"])
def test_shared_or_any_other_value_is_refused_before_a_job_folder(
    tmp_path: Path, value: str
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "shared-task")
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--verifier-sandbox",
            value,
        ],
        env={"BENCHFLOW_SKIP_PREFLIGHT": "1"},
    )
    assert result.exit_code == 2, result.output
    assert "verifier_sandbox" in result.output
    assert "Traceback" not in result.output
    assert not (tmp_path / "jobs").exists()
    with pytest.raises(ValueError, match="verifier_sandbox"):
        EvaluationConfig(**{"agent": "oracle", "verifier_sandbox": value})
    with pytest.raises(ValueError, match="verifier_sandbox"):
        RolloutConfig(
            task_path=tasks / "shared-task", agent="oracle", verifier_sandbox=value
        )


def test_a_shared_value_in_a_run_config_is_refused_before_a_job_folder(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "shared-task")
    path = tmp_path / "job.yaml"
    path.write_text(
        f"tasks_dir: {tasks}\njobs_dir: {tmp_path / 'jobs'}\nagent: oracle\n"
        "verifier_sandbox: shared\n"
    )
    result = CliRunner().invoke(
        cli_main.app,
        ["eval", "run", "--config", str(path)],
        env={"BENCHFLOW_SKIP_PREFLIGHT": "1"},
    )
    assert result.exit_code == 1, result.output
    assert "no 'shared' value" in result.output
    assert not (tmp_path / "jobs").exists()


def test_no_setting_turns_a_declared_separate_task_shared(tmp_path: Path) -> None:
    from benchflow._utils.config_override import validate_overlay
    from benchflow.integrity.trial import resolve_verifier_isolation
    from benchflow.task import Task

    # The run overlay still refuses verifier keys.
    for overlay in (
        {"verifier": {"sandbox_mode": "shared"}},
        {"verifier": {"environment_mode": "shared"}},
    ):
        with pytest.raises(ValueError, match="disallowed section"):
            validate_overlay(overlay)
    declared = Task(
        _task(tmp_path, "separate-task", 'environment_mode = "separate"\n')
    ).config
    shared = Task(_task(tmp_path, "shared-task")).config
    for setting in ("task", "separate", None):
        for integrity in ("off", "audit", "strict"):
            assert resolve_verifier_isolation(declared, setting, integrity) == (
                "separate",
                "task",
            )
    assert resolve_verifier_isolation(shared, "task", "audit") == ("shared", "task")
    assert resolve_verifier_isolation(shared, None, "off") == ("shared", "task")


# --- plan-time refusal -------------------------------------------------------------------


def _assert_names_every_refusal(message: str, *, switch: str, next_step: str) -> None:
    for name, (_, phrase) in _REFUSED.items():
        lines = [line for line in message.splitlines() if line.startswith(f"- {name}:")]
        assert lines, f"{name} not named in:\n{message}"
        assert any(phrase in line for line in lines), (name, lines)
    assert "- fine:" not in message
    assert message.startswith(switch), message
    assert f"{len(_REFUSED)} of {len(_REFUSED) + 1} selected task(s)" in message
    excludes = " ".join(f"--exclude {name}" for name in sorted(_REFUSED))
    assert f"next: drop those tasks ({excludes}), or {next_step}" in message


@pytest.mark.parametrize(
    "fields,switch,next_step",
    [
        (
            {"verifier_sandbox": "separate"},
            "--verifier-sandbox separate",
            "run without --verifier-sandbox separate",
        ),
        ({"integrity": "strict"}, "--integrity strict", "run with --integrity audit"),
    ],
)
@pytest.mark.parametrize("preflight", [True, False])
def test_a_forced_run_refuses_unsupported_tasks_before_a_job_folder(
    tmp_path: Path, monkeypatch, fields, switch, next_step, preflight
) -> None:
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    started = _no_sandbox(monkeypatch)
    tasks = _refused_set(tmp_path)
    config = EvaluationConfig(agent="oracle", environment="docker")
    for key, value in fields.items():
        setattr(config, key, value)
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=config,
        job_name="forced",
        preflight=preflight,
    )
    with pytest.raises(ValueError) as caught:
        asyncio.run(job.run())
    _assert_names_every_refusal(str(caught.value), switch=switch, next_step=next_step)
    assert not (tmp_path / "jobs").exists()
    assert started == []


@pytest.mark.parametrize(
    "flags,switch,next_step",
    [
        (
            ["--verifier-sandbox", "separate"],
            "--verifier-sandbox separate",
            "run without --verifier-sandbox separate",
        ),
        (["--integrity", "strict"], "--integrity strict", "run with --integrity audit"),
    ],
)
@pytest.mark.parametrize("sharded", [False, True])
def test_the_cli_refuses_the_set_before_a_job_folder_or_shard_plan(
    tmp_path: Path, monkeypatch, flags, switch, next_step, sharded
) -> None:
    started = _no_sandbox(monkeypatch)
    tasks = _refused_set(tmp_path)
    jobs = tmp_path / "jobs"
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            "oracle",
            "--sandbox",
            "docker",
            "--jobs-dir",
            str(jobs),
            *(["--worker-concurrency", "1"] if sharded else []),
            *flags,
        ],
        env={"BENCHFLOW_SKIP_PREFLIGHT": "1", "COLUMNS": "400"},
    )
    assert result.exit_code == 1, result.output
    output = " ".join(result.output.split())
    for name, (_, phrase) in _REFUSED.items():
        assert f"- {name}:" in output and phrase in output, name
    assert switch in output and next_step in output
    assert not (jobs / "worker-shards" / "plan.json").exists()
    assert not jobs.exists()
    assert started == []


def test_a_sharded_run_refuses_before_its_plan_exists(
    tmp_path: Path, monkeypatch
) -> None:
    from benchflow.eval_sharding import run_sharded_evaluation

    started = _no_sandbox(monkeypatch)
    tasks = _refused_set(tmp_path)
    config = EvaluationConfig(agent="oracle", environment="docker")
    config.verifier_sandbox = "separate"
    with pytest.raises(ValueError) as caught:
        asyncio.run(
            run_sharded_evaluation(
                tasks_dir=tasks,
                jobs_dir=tmp_path / "jobs",
                config=config,
                worker_concurrency=1,
                worker_retries=0,
                worker_start_stagger_sec=0,
            )
        )
    _assert_names_every_refusal(
        str(caught.value),
        switch="--verifier-sandbox separate",
        next_step="run without --verifier-sandbox separate",
    )
    assert not (tmp_path / "jobs" / "worker-shards" / "plan.json").exists()
    assert started == []


def test_a_backend_without_separate_verifiers_refuses_every_task(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    started = _no_sandbox(monkeypatch)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "fine")
    # Modal runs separate verifiers now; Apple Container does not.
    config = EvaluationConfig(agent="oracle", environment="apple-container")
    config.verifier_sandbox = "separate"
    job = Evaluation(tasks_dir=tasks, jobs_dir=tmp_path / "jobs", config=config)
    with pytest.raises(ValueError) as caught:
        asyncio.run(job.run())
    message = str(caught.value)
    assert "- fine: verifier.sandbox_mode: separate verifier sandboxes are " in message
    assert "(sandbox=apple-container)" in message and "on apple-container" in message
    assert not (tmp_path / "jobs").exists()
    assert started == []


def test_the_overlay_is_applied_before_the_plan_gate(
    tmp_path: Path, monkeypatch
) -> None:
    """An artifact a run overlay adds under /tests is refused at plan time,
    as Rollout.setup (which applies the overlay first) would refuse it."""
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    started = _no_sandbox(monkeypatch)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "fine")
    config = EvaluationConfig(
        agent="oracle",
        environment="docker",
        config_override={"artifacts": ["/tests/conftest.py"]},
    )
    config.verifier_sandbox = "separate"
    job = Evaluation(tasks_dir=tasks, jobs_dir=tmp_path / "jobs", config=config)
    with pytest.raises(ValueError, match="overlaps verifier-owned '/tests'"):
        asyncio.run(job.run())
    assert not (tmp_path / "jobs").exists()
    assert started == []


def test_the_task_setting_never_refuses_at_plan_time(
    tmp_path: Path, monkeypatch
) -> None:
    """Without the switch or strict, the plan gate checks nothing: tasks
    (judge tasks included) reach their trials as before."""
    tasks = _refused_set(tmp_path)
    ran: list[str] = []

    async def fake_single(self, task_dir, cfg):
        ran.append(task_dir.name)
        return RolloutResult(task_name=task_dir.name, rewards={"reward": 1.0})

    monkeypatch.setattr(Evaluation, "_run_single_task", fake_single)
    job = Evaluation(
        tasks_dir=tasks,
        jobs_dir=tmp_path / "jobs",
        config=EvaluationConfig(agent="oracle", environment="docker"),
        job_name="plain",
        preflight=False,
    )
    asyncio.run(job.run())
    assert sorted(ran) == sorted([*_REFUSED, "fine"])


# --- bench tasks check prints the same lines ----------------------------------------------


def _check(task: Path, *extra: str) -> Any:
    return CliRunner().invoke(
        cli_main.app,
        ["tasks", "check", str(task), *extra],
        env={"COLUMNS": "400"},
    )


@pytest.mark.parametrize("name", sorted(_REFUSED))
def test_tasks_check_prints_the_plan_gates_text(
    tmp_path: Path, monkeypatch, name: str
) -> None:
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    _no_sandbox(monkeypatch)
    kwargs, phrase = _REFUSED[name]
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    task = _task(tasks, name, **kwargs)
    config = EvaluationConfig(agent="oracle", environment="docker")
    config.verifier_sandbox = "separate"
    job = Evaluation(tasks_dir=tasks, jobs_dir=tmp_path / "jobs", config=config)
    with pytest.raises(ValueError) as caught:
        asyncio.run(job.run())
    gate_lines = str(caught.value).splitlines()[1:]

    result = _check(task, "--sandbox", "docker", "--verifier-sandbox", "separate")
    assert result.exit_code == 1, result.output
    shown = [line.strip() for line in result.output.splitlines()]
    for line in gate_lines:
        assert line in shown, (line, result.output)
    assert any(phrase in line for line in gate_lines)


def test_tasks_check_names_the_verifier_image_source(tmp_path: Path) -> None:
    built = _task(tmp_path, "built")
    fallback = _task(tmp_path, "fallback", tests_dockerfile=False)
    result = _check(built, "--verifier-sandbox", "separate")
    assert result.exit_code == 0, result.output
    assert "verifier image from tests/Dockerfile" in result.output
    assert "on docker" in result.output
    result = _check(fallback, "--sandbox", "docker", "--verifier-sandbox", "separate")
    assert result.exit_code == 0, result.output
    assert "verifier image from environment/Dockerfile" in result.output
    # Without the switch nothing changes.
    assert "verifier image" not in _check(built).output
    bad = _check(built, "--verifier-sandbox", "shared")
    assert bad.exit_code == 2 and "no 'shared' value" in bad.output


# --- resume ----------------------------------------------------------------------------------


def test_resume_compares_the_switch_as_p2_compares_settings(tmp_path: Path) -> None:
    from benchflow.evaluation import _recorded_run_differs

    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "shared-task")

    def recorded(**fields: Any) -> Path:
        config = EvaluationConfig(agent="oracle")
        for key, value in fields.items():
            setattr(config, key, value)
        job = Evaluation(
            tasks_dir=tasks,
            jobs_dir=tmp_path / f"jobs-{len(list(tmp_path.iterdir()))}",
            config=config,
            job_name="named",
        )
        job._write_evaluation_record()
        return job.job_dir

    def now(**fields: Any) -> EvaluationConfig:
        config = EvaluationConfig(agent="oracle")
        for key, value in fields.items():
            setattr(config, key, value)
        return config

    plain = recorded()
    assert _recorded_run_differs(plain, now(), tasks) is None
    assert _recorded_run_differs(plain, now(verifier_sandbox="separate"), tasks) == (
        "verifier_sandbox='task' (this run: 'separate')"
    )
    # Strict implies the switch: strict then the switch alone is no change of
    # where the verifiers ran (the audit itself is not compared).
    strict = recorded(integrity="strict")
    assert (
        _recorded_run_differs(
            strict, now(verifier_sandbox="separate", integrity="audit"), tasks
        )
        is None
    )
    assert "verifier_sandbox='separate' (this run: 'task')" in (
        _recorded_run_differs(strict, now(), tasks) or ""
    )
    # A record from before the switch ran "task" (or "separate" under strict).
    old = recorded()
    record = json.loads((old / "evaluation.json").read_text())
    del record["config"]["verifier_sandbox"]
    (old / "evaluation.json").write_text(json.dumps(record))
    assert _recorded_run_differs(old, now(), tasks) is None
    assert _recorded_run_differs(old, now(verifier_sandbox="separate"), tasks)


def test_a_named_job_sets_aside_trials_of_the_other_setting(
    tmp_path: Path, monkeypatch
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "shared-task")
    ran: list[str] = []

    async def fake_single(self, task_dir, cfg):
        ran.append(cfg.verifier_sandbox)
        trial = self._jobs_dir / self._job_name / f"{task_dir.name}__{len(ran)}"
        trial.mkdir(parents=True)
        payload = {
            "task_name": task_dir.name,
            "rollout_name": trial.name,
            "rewards": {"reward": 1.0},
        }
        (trial / "result.json").write_text(json.dumps(payload))
        (trial / "config.json").write_text(json.dumps({"agent": "oracle"}))
        return RolloutResult(
            task_name=task_dir.name, rollout_name=trial.name, rewards={"reward": 1.0}
        )

    monkeypatch.setattr(Evaluation, "_run_single_task", fake_single)

    def run(setting: str, restart_changed: bool = False) -> None:
        config = EvaluationConfig(agent="oracle", environment="docker")
        config.verifier_sandbox = setting
        asyncio.run(
            Evaluation(
                tasks_dir=tasks,
                jobs_dir=tmp_path / "jobs",
                config=config,
                job_name="named",
                preflight=False,
                restart_changed=restart_changed,
            ).run()
        )

    run("task")
    run("task")  # unchanged: the finished trial is reused
    assert ran == ["task"]
    # Changed: the finished trial does not count for this run, which is
    # refused unless it restarts the job's trials.
    with pytest.raises(ResumeMismatchError, match="verifier_sandbox='task'"):
        run("separate")
    assert ran == ["task"]
    run("separate", restart_changed=True)
    assert ran == ["task", "separate"]


# --- the job summary ---------------------------------------------------------------------------


def test_the_summary_counts_modes_and_warns_about_run_flag_tasks_without_artifacts(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "bare")
    _task(tasks, "declares", artifacts='artifacts = ["/app/out.txt"]\n')
    _task(tasks, "own-separate", 'environment_mode = "separate"\n')
    isolation = {
        "bare": ("separate", "run-flag"),
        "declares": ("separate", "run-flag"),
        "own-separate": ("separate", "task"),
    }

    async def fake_single(self, task_dir, cfg):
        trial = self._jobs_dir / self._job_name / f"{task_dir.name}__1"
        trial.mkdir(parents=True)
        mode, source = isolation[task_dir.name]
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task_dir.name,
                    "rollout_name": trial.name,
                    "rewards": {"reward": 0.0},
                    "verifier_isolation": mode,
                    "verifier_isolation_source": source,
                }
            )
        )
        return RolloutResult(
            task_name=task_dir.name, rollout_name=trial.name, rewards={"reward": 0.0}
        )

    monkeypatch.setattr(Evaluation, "_run_single_task", fake_single)
    config = EvaluationConfig(agent="oracle", environment="docker")
    config.verifier_sandbox = "separate"
    caplog.set_level(logging.INFO, logger="benchflow.evaluation")
    asyncio.run(
        Evaluation(
            tasks_dir=tasks,
            jobs_dir=tmp_path / "jobs",
            config=config,
            job_name="j",
            preflight=False,
        ).run()
    )
    summary = json.loads((tmp_path / "jobs" / "j" / "summary.json").read_text())
    assert summary["verifier_isolation"] == {
        "modes": {"separate": 3},
        "sources": {"run-flag": 2, "task": 1},
        "run_flag_without_artifacts": ["bare"],
    }
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    named = [w for w in warnings if "declare no artifacts" in w]
    assert len(named) == 1 and "bare" in named[0]
    assert "declares" not in named[0] and "own-separate" not in named[0]
    infos = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert "Verifier: 3 separate" in infos


# --- the separate verifier's image matches the agent's environment (R1-A-1) --------


@pytest.mark.asyncio
async def test_the_fallback_verifier_image_gets_the_runs_base_and_context_staging(
    tmp_path: Path,
) -> None:
    """A task with neither tests/Dockerfile nor a docker_image is verified in
    an image built from its environment/Dockerfile. --base-image-override and
    --context-root change the agent's build; the verifier's build gets the
    same, or its COPY sources are missing and it starts FROM another base."""
    from benchflow.rollout._separate_verifier import run_separate_verifier
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from tests.test_separate_verifier import (
        FakeVerifierSandbox,
        _agent_trial,
        _fake_rollout,
        _task_dir,
    )

    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(
        tmp_path,
        '[verifier]\nenvironment_mode = "separate"\n',
        tests_dockerfile=False,
    )
    original = "FROM registry.invalid/base:1\nCOPY packages/lib /opt/lib\n"
    (task / "environment" / "Dockerfile").write_text(original)
    repo = tmp_path / "repo"
    (repo / "packages" / "lib").mkdir(parents=True)
    (repo / "packages" / "lib" / "mod.txt").write_text("dep\n")
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._config.context_root = repo
    rollout._config.base_image_override = "mirror.invalid/base:1"
    rollout._planes = DefaultRolloutPlanes()
    created: list[Path] = []

    def create(vtask, context_path, vpaths):
        created.append(context_path)
        return FakeVerifierSandbox(tmp_path / "verifier-box")

    async def verify(env, vtask, vpaths, timing, *, workspace, agent_paths):
        vpaths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (vpaths.verifier_dir / "reward.txt").write_text("1\n")
        return {"reward": 1.0}, None

    rewards, error = await run_separate_verifier(
        rollout, create_environment=create, verify=verify
    )

    assert (rewards, error) == ({"reward": 1.0}, None)
    environment = created[0] / "environment"
    dockerfile = (environment / "Dockerfile").read_text()
    assert "FROM mirror.invalid/base:1" in dockerfile
    assert "registry.invalid" not in dockerfile
    assert "COPY packages/lib " not in dockerfile and "COPY _deps/" in dockerfile
    staged = [p for p in (environment / "_deps").rglob("mod.txt")]
    assert [p.read_text() for p in staged] == ["dep\n"]
    # The task's own folder is never rewritten.
    assert (task / "environment" / "Dockerfile").read_text() == original
    record = json.loads(
        (trial / "verifier-sandbox" / "verifier-sandbox.json").read_text()
    )
    assert record["image_source"] == "environment/Dockerfile"


@pytest.mark.asyncio
async def test_a_tests_dockerfile_verifier_image_is_built_as_the_task_declares(
    tmp_path: Path,
) -> None:
    from benchflow.rollout._separate_verifier import run_separate_verifier
    from benchflow.rollout_planes import DefaultRolloutPlanes
    from tests.test_separate_verifier import (
        FakeVerifierSandbox,
        _agent_trial,
        _fake_rollout,
        _task_dir,
    )

    _, workspace, trial = await _agent_trial(tmp_path)
    task = _task_dir(tmp_path, '[verifier]\nenvironment_mode = "separate"\n')
    rollout = _fake_rollout(tmp_path, trial, task, str(workspace))
    rollout._config.context_root = tmp_path
    rollout._config.base_image_override = "mirror.invalid/base:1"
    rollout._planes = DefaultRolloutPlanes()
    created: list[Path] = []

    def create(vtask, context_path, vpaths):
        created.append(context_path)
        return FakeVerifierSandbox(tmp_path / "verifier-box")

    async def verify(env, vtask, vpaths, timing, *, workspace, agent_paths):
        return {"reward": 1.0}, None

    await run_separate_verifier(rollout, create_environment=create, verify=verify)
    dockerfile = (created[0] / "environment" / "Dockerfile").read_text()
    assert dockerfile == (task / "tests" / "Dockerfile").read_text()


def _manifest() -> Any:
    from benchflow.environment.manifest import EnvironmentManifest

    return EnvironmentManifest(name="crm", image="registry.invalid/crm:1")


def _manifest_set(root: Path) -> Path:
    """Two tasks whose verifier image only mirrors the agent's environment,
    and one with its own tests/Dockerfile."""
    tasks = root / "tasks"
    tasks.mkdir()
    _task(tasks, "env-dockerfile", tests_dockerfile=False)
    _task(
        tasks,
        "env-image",
        tail='docker_image = "ubuntu:24.04"\n',
        tests_dockerfile=False,
    )
    _task(tasks, "own-verifier-image")
    return tasks


@pytest.mark.parametrize("sharded", [False, True])
def test_a_manifest_run_refuses_verifier_images_that_mirror_the_agents(
    tmp_path: Path, monkeypatch, sharded
) -> None:
    """The manifest's image and state reach only the agent's sandbox, so a
    verifier image that stands for "the agent's environment" would score in
    another one: refused at plan time, before a job folder or shard plan."""
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    started = _no_sandbox(monkeypatch)
    tasks = _manifest_set(tmp_path)
    config = EvaluationConfig(agent="oracle", environment="docker")
    config.verifier_sandbox = "separate"
    config.environment_manifest = _manifest()
    jobs = tmp_path / "jobs"
    with pytest.raises(ValueError) as caught:
        if sharded:
            from benchflow.eval_sharding import run_sharded_evaluation

            asyncio.run(
                run_sharded_evaluation(
                    config=config,
                    tasks_dir=tasks,
                    jobs_dir=jobs,
                    worker_concurrency=1,
                    worker_retries=0,
                    worker_start_stagger_sec=0,
                )
            )
        else:
            asyncio.run(
                Evaluation(
                    tasks_dir=tasks,
                    jobs_dir=jobs,
                    config=config,
                    job_name="m",
                    preflight=False,
                ).run()
            )
    message = str(caught.value)
    for name, source in (
        ("env-dockerfile", "environment/Dockerfile"),
        ("env-image", "sandbox.docker_image"),
    ):
        lines = [line for line in message.splitlines() if line.startswith(f"- {name}:")]
        assert len(lines) == 1, message
        assert "environment manifest 'crm'" in lines[0] and source in lines[0]
        assert "add tests/Dockerfile" in lines[0]
    assert "- own-verifier-image:" not in message
    assert not (jobs / "worker-shards" / "plan.json").exists()
    assert not jobs.exists()
    assert started == []


@pytest.mark.parametrize("with_base_image_manifest", [False, True])
def test_the_mirrored_images_run_without_a_manifest_image(
    tmp_path: Path, with_base_image_manifest
) -> None:
    """No manifest, or one with only a base_image and services (the agent's
    image stays the task's): nothing to refuse. Its services are outside
    the transfer like any other, which the docs name as the switch's limit."""
    from benchflow.environment.manifest import EnvironmentManifest
    from benchflow.integrity.trial import check_separate_plan

    tasks = _manifest_set(tmp_path)
    check_separate_plan(
        sorted(tasks.iterdir()),
        sandbox="docker",
        run_setting="separate",
        integrity="off",
        environment_manifest=EnvironmentManifest(
            name="fake-llm",
            base_image="ubuntu:24.04",
            owns_lifecycle=False,
            services=[{"name": "svc", "command": "true", "port": 8911}],
        )
        if with_base_image_manifest
        else None,
    )


def test_the_rollout_backstop_refuses_a_manifest_with_a_mirrored_image(
    tmp_path: Path, monkeypatch
) -> None:
    task = _task(tmp_path, "env-dockerfile", tests_dockerfile=False)
    config = _rollout_config(
        task,
        tmp_path / "jobs",
        verifier_sandbox="separate",
        environment_manifest=_manifest(),
    )
    with pytest.raises(ValueError, match="environment manifest 'crm'"):
        _trial(config, monkeypatch)


# --- a hosted environment cannot honour the switch (R1-A-2) -------------------------


@pytest.mark.parametrize(
    "flags,named",
    [
        (["--verifier-sandbox", "separate"], "--verifier-sandbox separate is"),
        (["--integrity", "strict"], "--integrity strict is"),
        (
            ["--verifier-sandbox", "separate", "--integrity", "strict"],
            "--verifier-sandbox separate and --integrity strict are",
        ),
    ],
)
def test_source_env_refuses_the_switch_instead_of_dropping_it(
    tmp_path: Path, monkeypatch, flags, named
) -> None:
    ran: list[Any] = []

    def fake_run_hosted_env(config):
        ran.append(config)
        raise AssertionError("the hosted run started")

    monkeypatch.setattr("benchflow.hosted_env.run_hosted_env", fake_run_hosted_env)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--source-env",
            "primeintellect/general-agent",
            "--model",
            "gemini-3.1-flash-lite-preview",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            *flags,
        ],
        env={"COLUMNS": "400"},
    )
    assert result.exit_code == 2, result.output
    output = " ".join(result.output.split())
    assert f"{named} not supported with --source-env" in output
    assert "separate verifier sandbox" in output
    assert ran == []
    assert not (tmp_path / "jobs").exists()


def test_source_env_keeps_the_task_setting(tmp_path: Path) -> None:
    from benchflow.eval_plan import EvalCreateRequest, build_eval_plan

    build_eval_plan(
        EvalCreateRequest(
            source_env="primeintellect/general-agent",
            model="m",
            jobs_dir=str(tmp_path / "jobs"),
            verifier_sandbox="task",
            integrity="audit",
        )
    )


def test_a_task_documents_own_manifest_image_is_checked_too(tmp_path: Path) -> None:
    """A task.md task's own manifest reaches its trial when the run names
    none, so the plan gate (and bench tasks check) reads it the same way."""
    import shutil

    from benchflow.integrity.trial import (
        SeparateVerifierRefusedError,
        check_separate_plan,
    )

    template = Path(__file__).parent / "integration" / "deterministic" / "task"
    task = tmp_path / "doc-manifest"
    shutil.copytree(template, task)
    manifest = task / "environment.toml"
    text = manifest.read_text()
    assert 'base_image     = "ubuntu:24.04"' in text
    manifest.write_text(
        text.replace(
            'base_image     = "ubuntu:24.04"', 'image = "registry.invalid/crm:1"'
        )
    )
    with pytest.raises(SeparateVerifierRefusedError) as caught:
        check_separate_plan(
            [task], sandbox="docker", run_setting="separate", integrity="off"
        )
    assert "environment manifest 'fake-llm'" in str(caught.value)
    assert "environment/Dockerfile" in str(caught.value)


# --- the legacy (agents + datasets) run YAML (SV2-codex-1) ---------------------------


def _legacy_yaml(tmp_path: Path, tasks: Path, extra: str, env: str = "docker") -> Path:
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "agents:\n  - name: oracle\n"
        f"datasets:\n  - path: {tasks}\n"
        f"environment:\n  type: {env}\n"
        f"jobs_dir: {tmp_path / 'jobs'}\n" + extra
    )
    return path


@pytest.mark.parametrize(
    ("extra", "mode", "integrity"),
    [
        ("verifier_sandbox: separate\n", "separate", "off"),
        ("integrity: strict\n", "task", "strict"),
        ("", "task", "off"),
    ],
)
def test_a_legacy_run_yaml_keeps_the_switch_and_integrity(
    tmp_path: Path, extra: str, mode: str, integrity: str
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "shared-task")
    evaluation = Evaluation.from_yaml(_legacy_yaml(tmp_path, tasks, extra))
    assert evaluation._config.verifier_sandbox == mode
    assert evaluation._config.integrity == integrity


def test_a_legacy_run_yaml_moves_the_verifier(tmp_path: Path, monkeypatch) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    task = _task(tasks, "shared-task")
    evaluation = Evaluation.from_yaml(
        _legacy_yaml(tmp_path, tasks, "verifier_sandbox: separate\n")
    )
    built = _evaluation_rollout_config(evaluation, task, monkeypatch)
    built.jobs_dir = tmp_path / "trials"
    rollout, _, _ = _trial(built, monkeypatch)
    assert _records(rollout) == ("separate", "separate", "run-flag")


def test_a_shared_value_in_a_legacy_run_yaml_is_refused_before_a_job_folder(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "shared-task")
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "--config",
            str(_legacy_yaml(tmp_path, tasks, "verifier_sandbox: shared\n")),
        ],
        env={"BENCHFLOW_SKIP_PREFLIGHT": "1"},
    )
    assert result.exit_code == 1, result.output
    assert "no 'shared' value" in result.output
    assert not (tmp_path / "jobs").exists()


def test_a_legacy_run_yaml_on_a_backend_without_separate_verifiers_refuses(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BENCHFLOW_SKIP_PREFLIGHT", "1")
    started = _no_sandbox(monkeypatch)
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    _task(tasks, "fine")
    job = Evaluation.from_yaml(
        _legacy_yaml(
            tmp_path, tasks, "verifier_sandbox: separate\n", env="apple-container"
        )
    )
    with pytest.raises(ValueError) as caught:
        asyncio.run(job.run())
    assert "(sandbox=apple-container)" in str(caught.value)
    assert not (tmp_path / "jobs").exists()
    assert started == []
