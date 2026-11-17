"""The task's own harness user beside BenchFlow's (sandbox/agent_user.py).

A task's own harness runs the agent as ``[agent].user``, else as main's image
``USER``; BenchFlow runs it as ``--sandbox-user`` (``agent`` by default). The
run says so once per task where they differ, and every result.json and the
run card record both with the flag that matches. Nothing changes who runs the
agent: these tests read the record and the line; the deterministic tier's
``tests/test_agent_user_docker.py`` checks the agent's uid on real Docker.
"""

from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow.sandbox import agent_user
from benchflow.task.runtime_view import TaskRuntimeView

FIELDS = {
    "task_harness",
    "task_harness_source",
    "ran_as",
    "image_workdir",
    "working_tree",
    "same",
    "matching_flag",
}
ROOT_IMAGE = "FROM python:3.12-slim\nWORKDIR /app\n"
NOBODY_IMAGE = "FROM python:3.12-slim\nWORKDIR /app\nUSER nobody\n"
# A base this host does not hold and that is not a Docker official image.
UNKNOWN_IMAGE = "FROM ghcr.io/example/never-pulled:1\nWORKDIR /app\n"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """No line reported yet, and no local image store: a base image's USER is
    known only for official images, as on a host that pulled nothing."""
    monkeypatch.setattr(agent_user, "_reported", {})
    monkeypatch.setattr(agent_user, "_tree_reported", set())
    for module in ("benchflow.sandbox.pid1_ptrace", "benchflow.sandbox.agent_user"):
        monkeypatch.setattr(
            f"{module}._local_image_user", lambda ref: None, raising=False
        )


def _task(
    root: Path,
    name: str,
    dockerfile: str = ROOT_IMAGE,
    *,
    user: str | None = None,
    compose: str | None = None,
) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    agent = f'user = "{user}"\n' if user is not None else ""
    (task / "task.toml").write_text(
        'version = "1.0"\n'
        f"[agent]\ntimeout_sec = 600\n{agent}"
        "[verifier]\ntimeout_sec = 120\n"
        "[environment]\ncpus = 1\nmemory_mb = 1024\n"
    )
    (task / "instruction.md").write_text("Write ok to /app/answer.txt.\n")
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    if compose is not None:
        (task / "environment" / "docker-compose.yaml").write_text(compose)
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    return task


def _record(task: Path, agent: str = "pi-acp", sandbox_user="agent", image_user=None):
    config = TaskRuntimeView.from_task_dir(task).config
    return agent_user.record(task, config, agent, sandbox_user, image_user)


ROOT_LINE = (
    "This task's own harness runs the agent as root; BenchFlow runs it as `agent`. "
    "To match it, pass `--sandbox-user null`."
)


def test_a_root_image_without_agent_user_gets_the_line_and_the_record(tmp_path):
    task = _task(tmp_path, "root-image")
    entry = _record(task)
    assert entry == {
        "task_harness": "root",
        "task_harness_source": "image USER",
        "ran_as": "agent",
        "image_workdir": None,
        "working_tree": None,
        "same": False,
        "matching_flag": "--sandbox-user null",
    }
    assert agent_user.start_lines({task.name: entry}, "pi-acp") == [ROOT_LINE]
    # Once per task: the trial does not say it again.
    assert agent_user.trial_line(task.name, entry, "pi-acp") is None


@pytest.mark.parametrize("user", ["", "root", "0", "0:0", "root:root"])
def test_an_empty_user_root_and_0_all_mean_root(tmp_path, user):
    task = _task(
        tmp_path, "t", f"FROM python:3.12-slim\nUSER {user}\n" if user else ROOT_IMAGE
    )
    assert _record(task)["task_harness"] == "root"
    assert _record(task, image_user=user)["task_harness"] == "root"


def test_a_user_nobody_image_records_nobody(tmp_path):
    task = _task(tmp_path, "nobody-image", NOBODY_IMAGE)
    entry = _record(task)
    assert entry["task_harness"] == "nobody"
    assert entry["task_harness_source"] == "image USER"
    assert entry["ran_as"] == "agent"
    assert entry["same"] is False
    assert entry["matching_flag"] == "--sandbox-user nobody"
    assert agent_user.line(entry, "pi-acp") == (
        "This task's own harness runs the agent as `nobody`; BenchFlow runs it as "
        "`agent`. To match it, pass `--sandbox-user nobody`."
    )
    # The started image's USER, read on the host, says the same.
    assert (
        _record(_task(tmp_path, "built", ROOT_IMAGE), image_user="nobody") == entry | {}
    )


def test_agent_user_set_and_matching_prints_no_line(tmp_path):
    task = _task(tmp_path, "declared", user="agent")
    entry = _record(task)
    assert entry == {
        "task_harness": "agent",
        "task_harness_source": "task.toml [agent].user",
        "ran_as": "agent",
        "image_workdir": None,
        "working_tree": None,
        "same": True,
        "matching_flag": "--sandbox-user agent",
    }
    assert agent_user.start_lines({task.name: entry}, "pi-acp") == []
    assert agent_user.trial_line(task.name, entry, "pi-acp") is None


@pytest.mark.parametrize("flag", [None, "null", "none", "root", "0"])
def test_sandbox_user_null_on_a_root_image_prints_no_line(tmp_path, flag):
    task = _task(tmp_path, "root-image")
    entry = _record(task, sandbox_user=flag)
    assert entry["ran_as"] == "root"
    assert entry["same"] is True
    assert entry["matching_flag"] == "--sandbox-user null"
    assert agent_user.start_lines({task.name: entry}, "pi-acp") == []
    assert agent_user.trial_line(task.name, entry, "pi-acp") is None


def test_the_record_fields_are_there_in_every_case(tmp_path):
    cases = [
        (_task(tmp_path, "a"), "pi-acp", "agent"),
        (_task(tmp_path, "b", NOBODY_IMAGE), "pi-acp", "agent"),
        (_task(tmp_path, "c", user="agent"), "pi-acp", "agent"),
        (_task(tmp_path, "d"), "pi-acp", None),
        (_task(tmp_path, "e", UNKNOWN_IMAGE), "pi-acp", "agent"),
        (_task(tmp_path, "f", NOBODY_IMAGE), "oracle", "agent"),
        (_task(tmp_path, "g"), "script:/tmp/x.sh", "agent"),
        (_task(tmp_path, "h"), "nop", "agent"),
    ]
    for task, agent, sandbox_user in cases:
        assert set(_record(task, agent, sandbox_user)) == FIELDS, task.name


def test_an_image_user_not_known_yet_is_said_by_the_trial(tmp_path):
    task = _task(tmp_path, "later", UNKNOWN_IMAGE)
    entry = _record(task)
    assert entry["task_harness"] is None and entry["same"] is None
    assert entry["matching_flag"] is None
    # Not known at the run's start: nothing then, the trial says it once.
    assert agent_user.start_lines({task.name: entry}, "pi-acp") == []
    started = _record(task, image_user="")
    assert agent_user.trial_line(task.name, started, "pi-acp") == (
        "later: its own harness runs the agent as root; BenchFlow runs it as "
        "`agent`. To match it, pass `--sandbox-user null`."
    )
    assert agent_user.trial_line(task.name, started, "pi-acp") is None


def test_compose_user_for_main_comes_before_the_image(tmp_path):
    compose = "services:\n  main:\n    build: .\n    user: '1000:1000'\n"
    entry = _record(_task(tmp_path, "compose", ROOT_IMAGE, compose=compose))
    assert entry["task_harness"] == "1000"
    assert entry["task_harness_source"] == "compose main user"
    # A uid beside a name cannot be compared on the host, and no flag names it.
    assert entry["same"] is None and entry["matching_flag"] is None


def test_the_oracle_runs_as_root_or_its_declared_user(tmp_path):
    nobody = _record(_task(tmp_path, "nobody", NOBODY_IMAGE), "oracle", "agent")
    assert nobody["ran_as"] == "root" and nobody["same"] is False
    assert nobody["matching_flag"] is None
    assert agent_user.line(nobody, "oracle") == (
        "This task's own harness runs the agent as `nobody`; BenchFlow's oracle "
        "runs solve.sh as root. No --sandbox-user changes the oracle's user."
    )
    declared = _record(_task(tmp_path, "declared", user="agent"), "oracle", "agent")
    assert declared["ran_as"] == "agent" and declared["same"] is True
    assert _record(_task(tmp_path, "root"), "oracle", "agent")["same"] is True


REFUSES_ROOT = {
    "script:/tmp/solve.sh": ("agents/scripts.py", "requires a non-root --sandbox-user"),
    "muse-code": (
        "agents/muse_config.py",
        "muse-code requires a non-root sandbox_user",
    ),
    "codex-acp": (
        "agents/codex_connector_policy.py",
        "Disabled Codex Apps requires a non-root sandbox user",
    ),
    "claude-agent-acp": (
        "agents/claude_connector_policy.py",
        "Claude connector restrictions require a non-root sandbox user",
    ),
}


@pytest.mark.parametrize("agent", list(REFUSES_ROOT))
def test_an_agent_that_refuses_root_gets_no_null_flag(tmp_path, agent):
    """--sandbox-user null would only get the run refused: the record has no
    flag, and the line says why. Each refusal is still in its module."""
    import benchflow

    module, refusal = REFUSES_ROOT[agent]
    source = (Path(benchflow.__file__).parent / module).read_text()
    assert refusal in source
    entry = _record(_task(tmp_path, "root"), agent, "agent")
    assert entry["same"] is False and entry["matching_flag"] is None
    assert "No --sandbox-user matches it: " in agent_user.line(entry, agent)
    # A run that matches, as such an agent can under its own settings
    # (codex-acp with --codex-apps-policy inherit), names the flag it used.
    if not agent.startswith(("script:", "muse")):
        ran_root = _record(_task(tmp_path, "ran-root"), agent, None)
        assert ran_root["same"] is True
        assert ran_root["matching_flag"] == "--sandbox-user null"
    # A non-root task's user is still a flag it takes.
    nobody = _record(_task(tmp_path, "nobody", NOBODY_IMAGE), agent, "agent")
    assert nobody["matching_flag"] == "--sandbox-user nobody"


def test_claude_agent_acp_on_a_root_task(tmp_path):
    entry = _record(_task(tmp_path, "root"), "claude-agent-acp", "agent")
    assert entry["matching_flag"] is None


def test_a_script_agent_on_a_root_image_has_no_matching_flag(tmp_path):
    entry = _record(_task(tmp_path, "root"), "script:/tmp/solve.sh", "agent")
    assert entry["same"] is False and entry["matching_flag"] is None
    assert agent_user.line(entry, "script:/tmp/solve.sh").endswith(
        "No --sandbox-user matches it: a script: agent needs a non-root sandbox user."
    )


def test_a_batch_gets_one_line_with_the_count(tmp_path):
    tasks = [
        _task(tmp_path, "r1"),
        _task(tmp_path, "r2"),
        _task(tmp_path, "same", user="agent"),
        _task(tmp_path, "later", UNKNOWN_IMAGE),
    ]
    entries = {task.name: _record(task) for task in tasks}
    assert agent_user.start_lines(entries, "pi-acp") == [
        "2 of 4 tasks' own harnesses run the agent as another user than BenchFlow "
        "does (`agent`); each trial's result.json names both under agent_user. "
        "To match them, pass `--sandbox-user null`."
    ]
    # The tasks the line covered say nothing again; the one known only once
    # its image is ready is said by its trial.
    assert agent_user.trial_line("r1", entries["r1"], "pi-acp") is None
    later = _record(tasks[3], image_user="root")
    assert agent_user.trial_line("later", later, "pi-acp").startswith("later: ")


def test_every_result_records_agent_user(tmp_path):
    from benchflow.rollout._results import _build_rollout_result

    entry = _record(_task(tmp_path / "tasks", "t"))
    for folder, kwargs in (("given", {"agent_user": entry}), ("default", {})):
        out = tmp_path / folder
        out.mkdir()
        _build_rollout_result(
            out,
            task_name="t",
            rollout_name=folder,
            agent="pi-acp",
            agent_name="pi-acp",
            model=None,
            n_tool_calls=0,
            prompts=["task"],
            error=None,
            verifier_error=None,
            trajectory=[],
            partial_trajectory=False,
            rewards={"reward": 1.0},
            started_at=datetime.now(),
            timing={},
            **kwargs,
        )
        recorded = json.loads((out / "result.json").read_text())
        assert "agent_user" in recorded
        assert recorded["agent_user"] == kwargs.get("agent_user")


class _Env:
    def __init__(self, user: str | None, pwd: str | None = None) -> None:
        self.user = user
        self.pwd = pwd
        self.reads = 0
        self.execs = 0

    def kept_identity(self):
        """The start probe's answer (sandbox/_exec_merge.py), as Docker keeps it."""
        from benchflow.sandbox._exec_merge import KeptIdentity

        if self.pwd is None:
            return None
        return KeptIdentity(
            generation=0, user=None, env_key=None, pwd=self.pwd, uid="0"
        )

    async def task_image_user(self) -> str | None:
        self.reads += 1
        return self.user

    async def exec(self, *args, **kwargs):  # pragma: no cover - must not run
        self.execs += 1
        raise AssertionError("nothing new may run in the sandbox")


def _rollout(task: Path, env: _Env, sandbox_user="agent", agent="pi-acp"):
    from benchflow.rollout import Rollout

    rollout = Rollout.__new__(Rollout)
    rollout._env = env
    rollout._task = SimpleNamespace(config=TaskRuntimeView.from_task_dir(task).config)
    rollout._config = SimpleNamespace(
        task_path=task, primary_agent=agent, sandbox_user=sandbox_user
    )
    return rollout


async def test_the_trial_reads_the_started_image_on_the_host_and_says_it_once(
    tmp_path, caplog
):
    task = _task(tmp_path, "later", UNKNOWN_IMAGE)
    env = _Env("nobody")
    rollout = _rollout(task, env)
    rollout._agent_user = await rollout._agent_user_record()
    assert rollout._agent_user["task_harness"] is None
    with caplog.at_level(logging.WARNING, logger="benchflow.rollout"):
        await rollout._note_agent_user()
        await _rollout(task, env)._note_agent_user()
    assert rollout._agent_user["task_harness"] == "nobody"
    assert env.reads == 2 and env.execs == 0
    said = [r.getMessage() for r in caplog.records if "own harness" in r.getMessage()]
    assert said == [
        "later: its own harness runs the agent as `nobody`; BenchFlow runs it as "
        "`agent`. To match it, pass `--sandbox-user nobody`."
    ]


async def test_a_matching_trial_says_nothing(tmp_path, caplog):
    task = _task(tmp_path, "root-image")
    rollout = _rollout(task, _Env(""), sandbox_user=None)
    with caplog.at_level(logging.WARNING, logger="benchflow.rollout"):
        await rollout._note_agent_user()
    assert rollout._agent_user["same"] is True
    assert not [r for r in caplog.records if "own harness" in r.getMessage()]


def test_nothing_here_changes_who_runs_the_agent():
    """The record is made beside the start and the result, from the config the
    rollout already has: it never reassigns the sandbox user or the task's
    [agent].user (the deterministic tier checks the uid itself)."""
    from benchflow.rollout import Rollout

    for method in (Rollout._agent_user_record, Rollout._note_agent_user):
        source = inspect.getsource(method)
        assert "sandbox_user =" not in source
        assert ".user =" not in source
    start = inspect.getsource(Rollout.start)
    assert start.index("check_at_start") < start.index("self._note_agent_user()")
    assert start.index("self._note_agent_user()") < start.index("pre_agent_hooks")
    assert "agent_user=self._agent_user" in inspect.getsource(Rollout._build_result)


def test_the_run_card_records_each_task_and_logs_the_line(tmp_path, caplog):
    from benchflow.evaluation import Evaluation, EvaluationConfig
    from benchflow.run_card import build_run_card

    root = tmp_path / "tasks"
    _task(root, "root-image")
    job = Evaluation(
        tasks_dir=str(root),
        jobs_dir=str(tmp_path / "jobs"),
        config=EvaluationConfig(agent="pi-acp", model="claude-haiku-4-5"),
        preflight=False,
    )
    card = build_run_card(job, job._get_task_dirs())
    assert card["agent_user"] == {"root-image": _record(root / "root-image")}
    job._run_card = card
    (tmp_path / "jobs" / job._job_name).mkdir(parents=True)
    with caplog.at_level(logging.INFO, logger="benchflow.evaluation"):
        job._log_run_card(done=0)
    assert ROOT_LINE in [r.getMessage() for r in caplog.records]
    written = json.loads(
        (tmp_path / "jobs" / job._job_name / "run-card.json").read_text()
    )
    assert set(written["agent_user"]["root-image"]) == FIELDS


def test_an_image_that_works_in_slash_hands_the_agent_no_tree(tmp_path):
    """FeatureBench's shape: the image works in / and the task names no
    [environment].workdir, so BenchFlow starts the agent user in /root and
    hands it none of the task's files; the trial says so outright, once."""
    task = _task(tmp_path, "featurebench-like")
    config = TaskRuntimeView.from_task_dir(task).config
    entry = agent_user.record(task, config, "pi-acp", "agent", None, None, "/")
    assert entry["image_workdir"] == "/"
    assert entry["working_tree"] == "not handed"
    text = agent_user.tree_line(task.name, entry, "pi-acp")
    assert text == (
        "featurebench-like: BenchFlow hands `agent` no working tree for this "
        "task. Its image works in / and the task names no [environment].workdir, "
        "so `agent` starts in /root and cannot write the task's own files (such "
        "as a repository under /testbed) unless the image lets every user write "
        "them: read its 0s under `agent` as that, not as the model's. To let the "
        "agent write them, pass `--sandbox-user null`, or set "
        "[environment].workdir to the task's working folder."
    )
    assert agent_user.tree_line(task.name, entry, "pi-acp") is None
    # An agent that refuses root gets the workdir way only.
    refusing = agent_user.record(task, config, "codex-acp", "agent", None, None, "/")
    assert agent_user.tree_line("other", refusing, "codex-acp").endswith(
        "Set [environment].workdir to the task's working folder to hand it over."
    )


@pytest.mark.parametrize(
    ("sandbox_user", "workdir", "pwd", "expected"),
    [
        ("agent", "/testbed", "/", "handed"),
        ("agent", None, "/app", "handed"),
        (None, None, "/", "root"),
        ("agent", None, None, None),
    ],
)
def test_the_working_tree_follows_the_workspace_benchflow_hands_over(
    tmp_path, sandbox_user, workdir, pwd, expected
):
    task = _task(tmp_path, "t")
    config = TaskRuntimeView.from_task_dir(task).config
    entry = agent_user.record(task, config, "pi-acp", sandbox_user, None, workdir, pwd)
    assert entry["working_tree"] == expected
    assert agent_user.tree_line("t", entry, "pi-acp") is None


async def test_the_trial_says_it_from_the_start_probe(tmp_path, caplog):
    """The hook reads the start probe's kept pwd (nothing new runs in the
    sandbox) and prints the no-working-tree line before the agent starts."""
    task = _task(tmp_path, "slash")
    env = _Env("", pwd="/")
    rollout = _rollout(task, env)
    with caplog.at_level(logging.WARNING, logger="benchflow.rollout"):
        await rollout._note_agent_user()
        await _rollout(task, env)._note_agent_user()
    assert rollout._agent_user["working_tree"] == "not handed"
    assert rollout._agent_user["image_workdir"] == "/"
    said = [r.getMessage() for r in caplog.records]
    assert sum("no working tree" in s for s in said) == 1
    assert env.execs == 0
