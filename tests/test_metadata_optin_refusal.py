"""A task's cloud-metadata opt-out needs the run's flag: the unit,
SDK and CLI rows that need no sandbox.

On 0.10.0 a task's ``allow_cloud_metadata = true`` (``[environment]`` in
task.toml, ``sandbox:`` in task.md) or ``metadata.x-allow-cloud-metadata =
true`` lifted the block by itself, with nothing printed. Now the key is the
task's request and ``--allow-cloud-metadata`` (the run config's
``allow_cloud_metadata``, ``EvaluationConfig`` / ``RolloutConfig`` /
``RuntimeConfig(allow_cloud_metadata=True)``) the operator's permission. A task
that asks without the permission is refused before anything starts, the
reason naming the flag; with it the run card names the task, its key and
the flag. The Docker rows are in ``test_metadata_optin_docker.py``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

FLAG = "--allow-cloud-metadata"


class _Started(RuntimeError):
    """What the patched job or sandbox raises when the run gets past the
    point where a task with an ungranted key must already be refused."""


def _task(root: Path, name: str = "md-key", *, key: str | None = "environment") -> Path:
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    environment = 'docker_image = "python:3.12-slim"\n'
    metadata = ""
    if key == "environment":
        environment += "allow_cloud_metadata = true\n"
    elif key == "metadata":
        metadata = '[metadata]\n"x-allow-cloud-metadata" = true\n\n'
    elif key == "metadata-yes":
        metadata = '[metadata]\n"x-allow-cloud-metadata" = "yes"\n\n'
    elif key == "metadata-one":
        metadata = '[metadata]\n"x-allow-cloud-metadata" = 1\n\n'
    (task / "task.toml").write_text(
        'version = "1.0"\n\n'
        + metadata
        + "[agent]\ntimeout_sec = 60.0\n\n[verifier]\ntimeout_sec = 60.0\n\n"
        f"[{'sandbox' if key == 'sandbox-table' else 'environment'}]\n"
        + environment
        + ("allow_cloud_metadata = true\n" if key == "sandbox-table" else "")
    )
    (task / "instruction.md").write_text("Do nothing.\n")
    for path in (task / "tests" / "test.sh", task / "solution" / "solve.sh"):
        path.write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
        path.chmod(0o755)
    return task


def _taskmd(root: Path, name: str = "md-key-taskmd") -> Path:
    task = root / name
    (task / "verifier").mkdir(parents=True)
    (task / "oracle").mkdir()
    (task / "task.md").write_text(
        "---\n"
        'schema_version: "1.3"\n'
        "agent:\n  timeout_sec: 60\n"
        "verifier:\n  timeout_sec: 60\n"
        "sandbox:\n  docker_image: python:3.12-slim\n  allow_cloud_metadata: true\n"
        "---\n# md-key-taskmd\n\n## prompt\n\nDo nothing.\n"
    )
    (task / "verifier" / "verifier.md").write_text(
        "---\n"
        'document_version: "0.3"\n'
        "verifier:\n  name: mdkey\n  default_strategy: deterministic\n"
        "  strategies:\n    deterministic:\n      type: script\n"
        "      command: ./test.sh\n"
        "---\n\n## verifier intent\n\nPass.\n"
    )
    for path in (task / "verifier" / "test.sh", task / "oracle" / "solve.sh"):
        path.write_text("#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n")
        path.chmod(0o755)
    return task


SPELLINGS = {
    "toml-environment": lambda root: _task(root, key="environment"),
    "toml-sandbox-table": lambda root: _task(root, key="sandbox-table"),
    "toml-metadata": lambda root: _task(root, key="metadata"),
    "toml-metadata-yes": lambda root: _task(root, key="metadata-yes"),
    "toml-metadata-one": lambda root: _task(root, key="metadata-one"),
    "taskmd-sandbox": _taskmd,
}


def _says_why(text: str) -> bool:
    """The refusal names the flag and the key it is about."""
    return FLAG in text and (
        "allow_cloud_metadata" in text or "x-allow-cloud-metadata" in text
    )


# ----------------------------------------------- the refusal: the unit rows


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_the_sandbox_factory_refuses_an_ungranted_key(tmp_path, spelling):
    """Every sandbox BenchFlow makes for a task goes through the factory
    (rollouts, separate verifiers, ``Environment.from_task``, the task
    tools): a key the caller did not grant is refused there, before any
    sandbox exists. On 0.10.0 the factory honoured it (main's every-user rule
    was left off)."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.task import RolloutPaths, Task

    task_dir = SPELLINGS[spelling](tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "trial")
    paths.mkdir()
    try:
        sandbox = default_rollout_planes().create_environment(
            "docker",
            task=Task(task_dir),
            task_path=task_dir,
            rollout_name="md-key",
            rollout_paths=paths,
            preserve_agent_network=False,
            environment_manifest=None,
        )
    except Exception as exc:
        assert _says_why(str(exc)), f"refused for another reason: {exc!r}"
        return
    pytest.fail(
        "the factory made the sandbox; its main metadata rule is "
        f"{getattr(sandbox, '_main_metadata_rule', 'unset')!r} (False: the key lifted it)"
    )


def test_the_factory_takes_the_key_once_the_run_grants_it(tmp_path):
    """The grant is the factory's ``allow_cloud_metadata``: then the key does
    what it did on 0.10.0 (main's every-user rule off)."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.task import RolloutPaths, Task

    task_dir = _task(tmp_path)
    paths = RolloutPaths(rollout_dir=tmp_path / "trial")
    paths.mkdir()
    sandbox = default_rollout_planes().create_environment(
        "docker",
        task=Task(task_dir),
        task_path=task_dir,
        rollout_name="md-key",
        rollout_paths=paths,
        preserve_agent_network=False,
        environment_manifest=None,
        allow_cloud_metadata=True,
    )
    assert getattr(sandbox, "_main_metadata_rule", None) is False


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_the_rollout_plan_gate_refuses_an_ungranted_key(tmp_path, spelling):
    """The rollout's own checks before any sandbox (``bench run --dry-run``
    runs the same): the task's key without ``allow_cloud_metadata`` on the
    RolloutConfig is refused with the reason."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import RolloutConfig, dry_run_plan_gate

    task_dir = SPELLINGS[spelling](tmp_path)
    cfg = RolloutConfig(task_path=task_dir, agent="oracle", environment="docker")
    with pytest.raises(Exception) as caught:
        dry_run_plan_gate(cfg, default_rollout_planes())
    assert _says_why(str(caught.value)), caught.value


def test_the_rollout_plan_gate_admits_a_granted_key(tmp_path):
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import RolloutConfig, dry_run_plan_gate

    task_dir = _task(tmp_path)
    cfg = RolloutConfig(
        task_path=task_dir,
        agent="oracle",
        environment="docker",
        allow_cloud_metadata=True,
    )
    dry_run_plan_gate(cfg, default_rollout_planes())


def test_a_task_without_the_key_needs_no_flag(tmp_path):
    """Guard: nothing changes for a task that does not ask."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import RolloutConfig, dry_run_plan_gate

    task_dir = _task(tmp_path, key=None)
    dry_run_plan_gate(
        RolloutConfig(task_path=task_dir, agent="oracle", environment="docker"),
        default_rollout_planes(),
    )


# ------------------------------------------------------ the refusal: the SDK


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_sdk_rollout_is_refused_before_any_sandbox(tmp_path, monkeypatch, spelling):
    """``benchflow.run_sync(RolloutConfig(...))`` without the grant: refused
    before the sandbox factory is called."""
    import benchflow as bf
    from benchflow import rollout_planes

    def started(*_a, **_k):
        raise _Started("the sandbox factory was called")

    monkeypatch.setattr(rollout_planes, "_create_environment", started)
    task_dir = SPELLINGS[spelling](tmp_path)
    try:
        result = bf.run_sync(
            bf.RolloutConfig(
                task_path=task_dir,
                agent="oracle",
                environment="docker",
                jobs_dir=tmp_path / "jobs",
            )
        )
    except _Started as exc:
        pytest.fail(f"not refused before the sandbox: {exc}")
    except Exception as exc:
        assert _says_why(str(exc)), exc
        return
    error = str(getattr(result, "error", "") or "")
    assert "the sandbox factory was called" not in error, error
    assert _says_why(error), error


def _job(tmp_path: Path, tasks: Path, **config):
    from benchflow.evaluation import Evaluation, EvaluationConfig

    base = {"agent": "oracle", "environment": "docker", "concurrency": 1}
    base.update(config)
    return Evaluation(
        tasks_dir=str(tasks),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(**base),
        preflight=False,
    )


def _two_tasks(tmp_path: Path) -> Path:
    root = tmp_path / "tasks"
    _task(root, "md-key")
    _task(root, "plain", key=None)
    return root


def test_sdk_job_is_refused_before_its_folder_exists(tmp_path, monkeypatch):
    """``Evaluation(...).run_sync()`` with a task that asks and no grant:
    refused before the job folder exists, naming the task and the flag."""
    from benchflow.evaluation import Evaluation

    async def started(self):
        raise _Started("the job started")

    monkeypatch.setattr(Evaluation, "_run_unlocked", started)
    job = _job(tmp_path, _two_tasks(tmp_path))
    with pytest.raises(Exception) as caught:
        job.run_sync()
    assert not isinstance(caught.value, _Started), "the job started"
    assert _says_why(str(caught.value)) and "md-key" in str(caught.value), caught.value
    assert not (tmp_path / "jobs").exists() or not any((tmp_path / "jobs").iterdir())


def test_sdk_job_with_the_grant_starts(tmp_path, monkeypatch):
    from benchflow.evaluation import Evaluation

    async def started(self):
        raise _Started("the job started")

    monkeypatch.setattr(Evaluation, "_run_unlocked", started)
    job = _job(tmp_path, _two_tasks(tmp_path), allow_cloud_metadata=True)
    with pytest.raises(_Started):
        job.run_sync()


def test_the_run_config_key_is_the_grant(tmp_path):
    """The run config's ``allow_cloud_metadata: true`` loads as the grant
    (the YAML a ``bench eval run --config`` reads)."""
    from benchflow._utils.yaml_loader import rollout_config_from_dict

    task_dir = _task(tmp_path)
    cfg = rollout_config_from_dict(
        {"task_path": str(task_dir), "agent": "oracle", "allow_cloud_metadata": True}
    )
    assert cfg.allow_cloud_metadata is True


# ------------------------------------------------------ the refusal: the CLI


def _bench(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    bench = Path(sys.executable).parent / "bench"
    if not bench.exists():
        pytest.skip("no bench script beside this Python")
    env = {k: v for k, v in os.environ.items() if k in ("USER", "LANG", "HOME", "PATH")}
    env.update(
        BENCHFLOW_SKIP_PREFLIGHT="1",
        HARBOR_TELEMETRY="off",
        DO_NOT_TRACK="1",
        COLUMNS="200",
    )
    return subprocess.run(
        [str(bench), *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=300,
    )


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_cli_dry_run_refuses_the_key_without_the_flag(tmp_path, spelling):
    """``bench run <task> --dry-run``: a non-zero exit, the reason naming the
    flag. On 0.10.0 the dry run passed."""
    task_dir = SPELLINGS[spelling](tmp_path)
    done = _bench(
        "run",
        str(task_dir),
        "--agent",
        "oracle",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--dry-run",
        cwd=tmp_path,
    )
    said = done.stdout + done.stderr
    assert done.returncode != 0 and _says_why(said), (done.returncode, said[-1200:])


def test_cli_dry_run_passes_with_the_flag(tmp_path):
    task_dir = _task(tmp_path)
    done = _bench(
        "run",
        str(task_dir),
        "--agent",
        "oracle",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--dry-run",
        FLAG,
        cwd=tmp_path,
    )
    assert done.returncode == 0, (done.stdout + done.stderr)[-1200:]


# --------------------------------------------- with the grant: the run card


def test_the_run_card_names_the_tasks_whose_key_the_flag_grants(tmp_path):
    from benchflow.run_card import build_run_card

    job = _job(tmp_path, _two_tasks(tmp_path), allow_cloud_metadata=True)
    card = build_run_card(job, job._get_task_dirs())
    block = card.get("cloud_metadata")
    assert isinstance(block, dict), card.keys()
    assert block.get("run_flag") == FLAG
    assert [t.get("task") for t in block.get("lifted_for") or []] == ["md-key"]
    assert "allow_cloud_metadata" in str(block["lifted_for"][0].get("key"))


def test_the_run_says_so_at_start(tmp_path, caplog):
    job = _job(tmp_path, _two_tasks(tmp_path), allow_cloud_metadata=True)
    job._prepare_run_card()
    (tmp_path / "jobs" / job._job_name).mkdir(parents=True, exist_ok=True)
    with caplog.at_level(logging.WARNING):
        job._log_run_card(done=0)
    warned = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and FLAG in r.getMessage()
    ]
    assert any("md-key" in w and "allow_cloud_metadata" in w for w in warned), warned


# ------------------------------------------------ the plan, behind the refusals


@pytest.mark.parametrize("spelling", sorted(SPELLINGS))
def test_the_plan_keeps_the_block_for_an_ungranted_key(tmp_path, spelling):
    """Behind the refusals: a task whose key the run did not grant still
    gets the agent's block armed (0.10.0 disarmed it for the key alone)."""
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan
    from benchflow.task import Task

    task = Task(SPELLINGS[spelling](tmp_path))
    plan = cloud_metadata_plan(
        task.config,
        sandbox_user="agent",
        launches_agent=True,
        allow_cloud_metadata=False,
    )
    assert plan is not None and plan.armed, plan
    granted = cloud_metadata_plan(
        task.config,
        sandbox_user="agent",
        launches_agent=True,
        allow_cloud_metadata=True,
    )
    assert granted is not None and not granted.armed
    assert granted.lifted_by.get("run") == FLAG and granted.lifted_by.get("task"), (
        granted
    )


# ---------------------------------------------- hints that name the task key


HINT_MODULES = (
    "src/benchflow/sandbox/cloud_metadata.py",
    "src/benchflow/sandbox/lockdown.py",
    "src/benchflow/sandbox/service_cover.py",
    "src/benchflow/sandbox/docker.py",
    "src/benchflow/sandbox/setup.py",
    "src/benchflow/cli/_options.py",
    "src/benchflow/task/config.py",
)


def test_every_message_that_names_the_task_key_names_the_flag_too():
    """A user told to set ``allow_cloud_metadata = true`` would meet the
    refusal next: every message (not a docstring) that names the task key
    also names ``--allow-cloud-metadata``."""
    import ast

    root = Path(__file__).resolve().parents[1]
    missing = []
    for name in HINT_MODULES:
        tree = ast.parse((root / name).read_text())
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.JoinedStr):
                text = "".join(
                    v.value for v in node.values if isinstance(v, ast.Constant)
                )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                if id(node) in docstrings:
                    continue
                text = node.value
            else:
                continue
            if (
                "allow_cloud_metadata = true" in text
                and "--allow-cloud-metadata" not in text
            ):
                if (
                    text.strip()
                    == "allow_cloud_metadata = true in the task's environment"
                ):
                    continue  # the record's name for the key, not a hint
                missing.append(f"{name}:{node.lineno}: {text[:90]!r}")
    assert not missing, missing


# --------------------------------------------- the recorded grant, strictly


@pytest.mark.parametrize(
    ("value", "granted"),
    [
        (True, True),
        ("true", True),
        ("yes", True),
        (1, True),
        (False, False),
        ("false", False),
        ("no", False),
        (0, False),
        (None, False),
    ],
)
def test_a_recorded_grant_is_read_strictly(value, granted):
    """A shard's or a resumed job's config carries the grant: a quoted
    "false" is no permission (bool("false") was)."""
    from benchflow.eval_worker import _evaluation_config

    assert (
        _evaluation_config({"allow_cloud_metadata": value}).allow_cloud_metadata
        is granted
    )


def test_a_recorded_grant_that_is_not_a_boolean_is_refused():
    from benchflow.eval_worker import _evaluation_config

    with pytest.raises(ValueError, match="allow_cloud_metadata must be true or false"):
        _evaluation_config({"allow_cloud_metadata": "maybe"})


# --------------------------------------------------------- the other entry paths


def test_environment_from_task_refuses_an_ungranted_key_and_leaves_nothing(tmp_path):
    from benchflow.runtime import Environment

    task_dir = _task(tmp_path)
    jobs = tmp_path / "jobs"
    with pytest.raises(Exception) as caught:
        Environment.from_task(task_dir, sandbox="docker", jobs_dir=jobs)
    assert _says_why(str(caught.value)), caught.value
    left = (
        list((jobs / "environment").glob("*"))
        if (jobs / "environment").exists()
        else []
    )
    assert not left, left


def test_environment_from_task_takes_the_grant(tmp_path):
    from benchflow.runtime import Environment

    env = Environment.from_task(
        _task(tmp_path),
        sandbox="docker",
        jobs_dir=tmp_path / "jobs",
        allow_cloud_metadata=True,
    )
    assert getattr(env.inner, "_main_metadata_rule", None) is False


async def test_a_granted_environment_with_an_ungranted_run_is_refused_before_it_starts(
    tmp_path, monkeypatch
):
    from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

    env = Environment.from_task(
        _task(tmp_path),
        sandbox="docker",
        jobs_dir=tmp_path / "jobs",
        allow_cloud_metadata=True,
    )

    async def started(self):
        raise _Started("the environment started")

    monkeypatch.setattr(Environment, "start", started)
    runtime = Runtime(
        env, Agent("oracle", model=""), RuntimeConfig(jobs_dir=tmp_path / "jobs")
    )
    with pytest.raises(Exception) as caught:
        await runtime.execute()
    assert not isinstance(caught.value, _Started), "the environment started"
    assert _says_why(str(caught.value)), caught.value


def test_bench_eval_run_dry_run_refuses_the_key_without_the_flag(tmp_path):
    task_dir = _task(tmp_path)
    done = _bench(
        "eval",
        "run",
        "--tasks-dir",
        str(task_dir),
        "--agent",
        "oracle",
        "--sandbox",
        "docker",
        "--jobs-dir",
        str(tmp_path / "jobs"),
        "--dry-run",
        cwd=tmp_path,
    )
    said = done.stdout + done.stderr
    assert done.returncode != 0 and _says_why(said), (done.returncode, said[-1200:])


async def test_the_bench_tasks_checks_say_they_cannot_run_such_a_task(tmp_path):
    from benchflow.task.equivalence import check_equivalence_async

    with pytest.raises(Exception) as caught:
        await check_equivalence_async(_task(tmp_path), sandbox_type="docker")
    hint = str(getattr(caught.value, "hint", "") or "")
    assert _says_why(str(caught.value)) and "bench tasks" in hint, (caught.value, hint)


# ------------------------------- a falsy grant, on every way a run is built
# (rollout_config_from_dict and the readers once took bool("false") as a grant)

FALSY = ["false", "no", "off", "0", "False", " Off "]


def _rollout_cfg(entry: str, task_dir: Path, value, tmp_path: Path):
    import yaml

    import benchflow as bf
    from benchflow._utils.yaml_loader import rollout_config_from_dict
    from benchflow.rollout import RolloutConfig

    raw = {
        "task_path": str(task_dir),
        "agent": "oracle",
        "environment": "docker",
        "jobs_dir": str(tmp_path / "jobs"),
        "allow_cloud_metadata": value,
    }
    if entry == "rollout_config_from_dict":
        return rollout_config_from_dict(raw)
    if entry == "RolloutConfig.from_dict":
        return RolloutConfig.from_dict(raw)
    if entry == "rollout_config_from_yaml":
        path = tmp_path / "rollout.yaml"
        path.write_text(yaml.safe_dump(raw))
        return bf.rollout_config_from_yaml(path)
    if entry == "RolloutConfig":
        return RolloutConfig(
            task_path=task_dir,
            agent="oracle",
            environment="docker",
            jobs_dir=tmp_path / "jobs",
            allow_cloud_metadata=value,
        )
    if entry == "replace":
        import dataclasses

        return dataclasses.replace(
            RolloutConfig(task_path=task_dir, agent="oracle", environment="docker"),
            allow_cloud_metadata=value,
        )
    if entry == "set-after":
        cfg = RolloutConfig(task_path=task_dir, agent="oracle", environment="docker")
        cfg.allow_cloud_metadata = value
        return cfg
    raise AssertionError(entry)


ROLLOUT_ENTRIES = [
    "rollout_config_from_dict",
    "RolloutConfig.from_dict",
    "rollout_config_from_yaml",
    "RolloutConfig",
    "replace",
    "set-after",
]


@pytest.mark.parametrize("value", FALSY)
@pytest.mark.parametrize("entry", ROLLOUT_ENTRIES)
def test_a_falsy_grant_on_a_rollout_config_is_no_grant(tmp_path, entry, value):
    """A quoted ``"false"`` (``no``, ``off``, ``0``) is no permission on any
    way a RolloutConfig is built: the key task is refused at the plan gate
    (``bench run --dry-run``'s check), the reason naming the flag."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import dry_run_plan_gate

    cfg = _rollout_cfg(entry, _task(tmp_path), value, tmp_path)
    if entry != "set-after":
        assert cfg.allow_cloud_metadata is False, cfg.allow_cloud_metadata
    with pytest.raises(Exception) as caught:
        dry_run_plan_gate(cfg, default_rollout_planes())
    assert _says_why(str(caught.value)), caught.value


@pytest.mark.parametrize("value", FALSY)
def test_a_falsy_grant_on_an_sdk_rollout_is_refused_before_any_sandbox(
    tmp_path, monkeypatch, value
):
    import benchflow as bf
    from benchflow import rollout_planes

    def started(*_a, **_k):
        raise _Started("the sandbox factory was called")

    monkeypatch.setattr(rollout_planes, "_create_environment", started)
    cfg = _rollout_cfg("set-after", _task(tmp_path), value, tmp_path)
    cfg.jobs_dir = tmp_path / "jobs"
    try:
        result = bf.run_sync(cfg)
    except _Started as exc:
        pytest.fail(f"not refused before the sandbox: {exc}")
    except Exception as exc:
        assert _says_why(str(exc)), exc
        return
    error = str(getattr(result, "error", "") or "")
    assert "the sandbox factory was called" not in error, error
    assert _says_why(error), error


@pytest.mark.parametrize("value", FALSY)
def test_a_falsy_grant_on_a_job_is_refused_before_its_folder_exists(
    tmp_path, monkeypatch, value
):
    from benchflow.evaluation import Evaluation

    async def started(self):
        raise _Started("the job started")

    monkeypatch.setattr(Evaluation, "_run_unlocked", started)
    job = _job(tmp_path, _two_tasks(tmp_path), allow_cloud_metadata=value)
    assert job._config.allow_cloud_metadata is False
    with pytest.raises(Exception) as caught:
        job.run_sync()
    assert not isinstance(caught.value, _Started), "the job started"
    assert _says_why(str(caught.value)), caught.value
    job._config.allow_cloud_metadata = value  # set after: read strictly too
    with pytest.raises(Exception) as caught:
        job.run_sync()
    assert not isinstance(caught.value, _Started), "the job started"


@pytest.mark.parametrize("value", FALSY)
def test_a_falsy_grant_in_a_run_config_file_is_refused(tmp_path, monkeypatch, value):
    """``bench eval run --config`` (``Evaluation.from_yaml``)."""
    import yaml

    from benchflow.evaluation import Evaluation

    async def started(self):
        raise _Started("the job started")

    monkeypatch.setattr(Evaluation, "_run_unlocked", started)
    path = tmp_path / "run.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tasks_dir": str(_two_tasks(tmp_path)),
                "jobs_dir": str(tmp_path / "jobs"),
                "agent": "oracle",
                "environment": "docker",
                "allow_cloud_metadata": value,
            }
        )
    )
    job = Evaluation.from_yaml(path)
    assert job._config.allow_cloud_metadata is False
    with pytest.raises(Exception) as caught:
        job.run_sync()
    assert not isinstance(caught.value, _Started), "the job started"
    assert _says_why(str(caught.value)), caught.value


@pytest.mark.parametrize("value", FALSY)
async def test_a_falsy_grant_on_the_runtime_is_refused(tmp_path, monkeypatch, value):
    """``Environment.from_task(allow_cloud_metadata=...)`` and
    ``RuntimeConfig(allow_cloud_metadata=...)``."""
    from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

    with pytest.raises(Exception) as caught:
        Environment.from_task(
            _task(tmp_path / "a"),
            sandbox="docker",
            jobs_dir=tmp_path / "jobs",
            allow_cloud_metadata=value,
        )
    assert _says_why(str(caught.value)), caught.value

    env = Environment.from_task(
        _task(tmp_path / "b"),
        sandbox="docker",
        jobs_dir=tmp_path / "jobs",
        allow_cloud_metadata=True,
    )

    async def started(self):
        raise _Started("the environment started")

    monkeypatch.setattr(Environment, "start", started)
    config = RuntimeConfig(jobs_dir=tmp_path / "jobs", allow_cloud_metadata=value)
    assert config.allow_cloud_metadata is False
    runtime = Runtime(env, Agent("oracle", model=""), config)
    with pytest.raises(Exception) as caught:
        await runtime.execute()
    assert not isinstance(caught.value, _Started), "the environment started"
    assert _says_why(str(caught.value)), caught.value


@pytest.mark.parametrize("value", FALSY)
def test_a_falsy_grant_at_the_sandbox_factory_is_refused(tmp_path, value):
    from benchflow.sandbox.cloud_metadata import refuse_ungranted_opt_in
    from benchflow.task import Task

    task = Task(_task(tmp_path))
    with pytest.raises(Exception) as caught:
        refuse_ungranted_opt_in(
            task.config, task_name="md-key", allow_cloud_metadata=value
        )
    assert _says_why(str(caught.value)), caught.value


@pytest.mark.parametrize("value", FALSY)
def test_the_shared_helpers_read_a_falsy_grant_as_none(tmp_path, value):
    """The helpers every path calls take the grant as given, so each reads
    it strictly: no record, no run-card grant, the agent's block armed."""
    from benchflow.sandbox.cloud_metadata import (
        cloud_metadata_plan,
        opt_in_records,
        run_card_block,
    )
    from benchflow.task import Task

    task_dir = _task(tmp_path)
    config = Task(task_dir).config
    assert opt_in_records(config, allow_cloud_metadata=value) == []
    card = run_card_block([task_dir], allow_cloud_metadata=value)
    assert card == {"run_flag": None, "lifted_for": []}, card
    plan = cloud_metadata_plan(
        config, sandbox_user="agent", launches_agent=True, allow_cloud_metadata=value
    )
    assert plan is not None and plan.armed, plan


def _not_a_boolean(entry: str, tmp_path: Path):
    from benchflow.evaluation import EvaluationConfig
    from benchflow.runtime import Environment, RuntimeConfig

    if entry in ROLLOUT_ENTRIES and entry != "set-after":
        return _rollout_cfg(entry, _task(tmp_path), "nope", tmp_path)
    if entry == "EvaluationConfig":
        return EvaluationConfig(agent="oracle", allow_cloud_metadata="nope")
    if entry == "RuntimeConfig":
        return RuntimeConfig(allow_cloud_metadata="nope")
    if entry == "Environment.from_task":
        return Environment.from_task(
            _task(tmp_path),
            sandbox="docker",
            jobs_dir=tmp_path / "jobs",
            allow_cloud_metadata="nope",
        )
    raise AssertionError(entry)


@pytest.mark.parametrize(
    "entry",
    [e for e in ROLLOUT_ENTRIES if e != "set-after"]
    + ["EvaluationConfig", "RuntimeConfig", "Environment.from_task"],
)
def test_a_grant_that_is_not_a_boolean_is_refused_on_every_entry(tmp_path, entry):
    with pytest.raises(ValueError, match="allow_cloud_metadata must be true or false"):
        _not_a_boolean(entry, tmp_path)
    left = list((tmp_path / "jobs").rglob("*")) if (tmp_path / "jobs").exists() else []
    assert not left, left


@pytest.mark.parametrize("value", [True, "true", "yes", "on", "1", 1])
@pytest.mark.parametrize("entry", ROLLOUT_ENTRIES)
def test_a_true_grant_on_a_rollout_config_still_grants(tmp_path, entry, value):
    """Guard: the spellings of true still grant (the plan gate admits)."""
    from benchflow.contracts import default_rollout_planes
    from benchflow.rollout import dry_run_plan_gate

    cfg = _rollout_cfg(entry, _task(tmp_path), value, tmp_path)
    dry_run_plan_gate(cfg, default_rollout_planes())


# ------------------------------------------- continue and branch take no grant


@pytest.mark.parametrize("command", ["bench eval continue", "bench eval branch"])
def test_a_command_with_no_grant_says_so_and_names_the_one_that_can(tmp_path, command):
    from benchflow.sandbox.cloud_metadata import refuse_opt_in_for_command

    with pytest.raises(Exception) as caught:
        refuse_opt_in_for_command([_task(tmp_path)], command=command)
    said = str(caught.value)
    hint = str(getattr(caught.value, "hint", ""))
    assert f"{command} takes no such flag, so it can't lift the block" in said, said
    assert "bench eval run --allow-cloud-metadata" in said, said
    assert hint.startswith(f"{command} can't lift the cloud metadata block"), hint
    assert "run with --allow-cloud-metadata" not in hint, hint
    # A task without the key, and a folder that is not a task, pass.
    refuse_opt_in_for_command([_task(tmp_path, "plain", key=None)], command=command)
    refuse_opt_in_for_command([tmp_path / "not-a-task"], command=command)


def test_bench_eval_continue_refuses_a_key_task_with_its_own_reason(tmp_path):
    from benchflow.continue_run.orchestrator import (
        build_agent_env,
        build_rollout_config,
    )
    from benchflow.continue_run.run_folder import RunFolder

    run = RunFolder(tmp_path, {"environment": "docker"}, {}, ["go"], [])
    with pytest.raises(Exception) as caught:
        build_rollout_config(
            run,
            task_path=_task(tmp_path),
            live_model="test",
            agent_env=build_agent_env("http://host:1/v1"),
            timeout=30,
            output_dir=tmp_path / "out",
            rollout_name="continued",
        )
    assert "bench eval continue takes no such flag" in str(caught.value), caught.value


async def test_bench_eval_branch_refuses_a_key_task_before_any_rollout(tmp_path):
    from benchflow.branch_run import BranchPlan, ChildSpec, run_branch_trial

    task = _task(tmp_path)
    plan = BranchPlan(
        task_paths=[task],
        agent="oracle",
        model=None,
        sandbox="docker",
        children=[ChildSpec("baseline")],
        checkpoint_after=0,
        jobs_dir=tmp_path / "jobs",
        job_name="job",
        isolate_children=False,
    )

    def factory(_config):
        raise _Started("a rollout was made")

    outcome = await run_branch_trial(plan, task, rollout_factory=factory)
    assert "a rollout was made" not in str(outcome.error), outcome.error
    assert "bench eval branch takes no such flag" in str(outcome.error), outcome.error
