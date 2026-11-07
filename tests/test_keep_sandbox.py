"""Guards --keep-sandbox: a trial's sandbox kept for `bench sandbox
attach`, never deleted by a later sweep, and deleted by `bench sandbox release`.

Docs rounds 2 and 3 scored "a failed sandbox cannot be kept or
attached to" against BenchFlow: every sandbox was deleted at teardown.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

from benchflow.cli import sandbox as sandbox_cli
from benchflow.evaluation import EvaluationConfig
from benchflow.rollout import Rollout, RolloutConfig
from benchflow.rollout import _keep_sandbox as keep
from benchflow.sandbox import _docker_sweep

TOML = (
    'version = "1.0"\n[agent]\ntimeout_sec = 60\n[verifier]\ntimeout_sec = 60\n'
    '[environment]\ndocker_image = "python:3.12-slim"\n'
)


def _task(root: Path) -> Path:
    task = root / "task"
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text(TOML)
    (task / "instruction.md").write_text("Do nothing.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    return task


# The Docker daemon's kept markers, before the fixture below stands in for them.
REAL_MARK = _docker_sweep._mark_on_daemon
REAL_UNMARK = _docker_sweep._unmark_on_daemon
REAL_MARKED = _docker_sweep._marked_on_daemon


@pytest.fixture(autouse=True)
def kept_dir(tmp_path, monkeypatch) -> Path:
    folder = tmp_path / "kept-registry"
    monkeypatch.setenv("BENCHFLOW_KEPT_SANDBOXES_DIR", str(folder))
    # No test here touches the real daemon: its markers are kept in memory.
    marked: set[str] = set()
    monkeypatch.setattr(
        _docker_sweep, "_mark_on_daemon", lambda project: marked.add(project) or True
    )
    monkeypatch.setattr(_docker_sweep, "_unmark_on_daemon", marked.discard)
    monkeypatch.setattr(_docker_sweep, "_marked_on_daemon", lambda: frozenset(marked))
    return folder


@pytest.mark.parametrize(
    ("policy", "rewards", "error", "verifier_error", "kept"),
    [
        ("never", None, "boom", None, False),
        ("on-failure", {"reward": 1.0}, None, None, False),
        ("on-failure", {"reward": 0.0}, None, None, True),
        ("on-failure", {"reward": 0.5}, None, None, True),
        ("on-failure", None, "agent error", None, True),
        ("on-failure", None, None, "verifier error", True),
        ("on-failure", None, None, None, True),
        ("always", {"reward": 1.0}, None, None, True),
    ],
)
def test_the_keep_rule_is_the_summarys_pass_rule(
    policy, rewards, error, verifier_error, kept
):
    assert (
        keep.wants_keep(
            policy, rewards=rewards, error=error, verifier_error=verifier_error
        )
        is kept
    )


def test_policies_and_backends_are_checked_before_a_run(tmp_path):
    task = _task(tmp_path)
    with pytest.raises(ValueError, match="never, on-failure or always"):
        RolloutConfig(task_path=task, keep_sandbox="sometimes")
    with pytest.raises(ValueError, match="docker or daytona"):
        RolloutConfig(task_path=task, environment="modal", keep_sandbox="on-failure")
    with pytest.raises(ValueError, match="docker or daytona"):
        EvaluationConfig(agent="oracle", environment="modal", keep_sandbox="always")
    assert RolloutConfig(task_path=task, keep_sandbox="ON-FAILURE").keep_sandbox == (
        "on-failure"
    )
    assert RolloutConfig(task_path=task).keep_sandbox == "never"


class _DockerEnv:
    compose_project_name = "bf-keep-test-0123abcd"

    def __init__(self) -> None:
        self.stops = 0
        self.detaches = 0

    async def stop(self, delete: bool = True) -> None:
        self.stops += 1

    async def detach(self) -> None:
        self.detaches += 1


def _rollout(tmp_path: Path, policy: str, rewards: dict | None) -> Rollout:
    rollout = Rollout(
        RolloutConfig.from_legacy(
            task_path=_task(tmp_path / "tasks"),
            agent="nop",
            environment="docker",
            jobs_dir=str(tmp_path / "jobs"),
            job_name="keep",
        )
    )
    rollout._config.keep_sandbox = policy
    rollout._rollout_dir = tmp_path / "trial"
    rollout._rollout_dir.mkdir()
    rollout._rewards = rewards
    return rollout


@pytest.mark.asyncio
async def test_a_failed_trial_keeps_its_sandbox_and_writes_the_record(tmp_path):
    rollout = _rollout(tmp_path, "on-failure", {"reward": 0.0})
    env = _DockerEnv()
    rollout._env = env

    await rollout._teardown_sandbox()

    assert env.stops == 0
    # Its exec channels close, so no `docker exec` outlives the event loop.
    assert env.detaches == 1
    record = json.loads((tmp_path / "trial" / keep.RECORD_NAME).read_text())
    assert record["backend"] == "docker"
    assert record["project"] == env.compose_project_name
    assert record["state"] == "kept"
    assert record["attach"] == f"bench sandbox attach {tmp_path / 'trial'}"
    assert env.compose_project_name in _docker_sweep.kept_projects()


@pytest.mark.asyncio
async def test_a_passed_trial_is_deleted_as_before(tmp_path):
    rollout = _rollout(tmp_path, "on-failure", {"reward": 1.0})
    env = _DockerEnv()
    rollout._env = env

    await rollout._teardown_sandbox()

    assert env.stops == 1
    assert env.detaches == 0
    assert not (tmp_path / "trial" / keep.RECORD_NAME).exists()
    assert _docker_sweep.kept_projects() == frozenset()


class _DaytonaSandbox:
    id = "sb-1234"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.intervals: list[int] = []
        self.labels = {"benchflow.managed": "owner"}

    async def set_auto_delete_interval(self, interval: int) -> None:
        if self.fail:
            raise RuntimeError("daytona down")
        self.intervals.append(interval)

    async def set_labels(self, labels: dict[str, str]) -> dict[str, str]:
        self.labels = labels
        return labels


@pytest.mark.asyncio
async def test_a_kept_daytona_sandbox_outlives_its_auto_stop(tmp_path):
    sandbox = _DaytonaSandbox()
    env = SimpleNamespace(_sandbox=sandbox, _uses_compose=False)
    record = await keep.keep(env, tmp_path, environment="daytona", task_copy=None)
    assert record is not None and record["sandbox_id"] == "sb-1234"
    # Daytona deletes a stopped sandbox at once when the interval is 0.
    assert sandbox.intervals == [keep.DAYTONA_KEPT_DELETE_MINUTES]
    # The reapers skip the kept label; the ownership label stays.
    assert sandbox.labels == {"benchflow.managed": "owner", "benchflow.kept": "1"}
    failing = SimpleNamespace(_sandbox=_DaytonaSandbox(fail=True))
    assert (
        await keep.keep(failing, tmp_path, environment="daytona", task_copy=None)
        is None
    )


def _rows(*rows: tuple[str, ...]) -> str:
    return "".join("\t".join(row) + "\n" for row in rows)


def test_no_sweep_removes_a_kept_project(monkeypatch):
    """A kept sandbox outlives its process, so its label says "gone": the
    sweep would otherwise force-remove the running container and its network."""
    gone = "otherhost:1:boot:1:ns"
    monkeypatch.setattr(_docker_sweep, "owner_state", lambda token: "gone")
    monkeypatch.setattr(_docker_sweep, "is_dead_running", lambda token: True)
    listings = {
        "network": _rows(("net-kept", gone, "kept"), ("net-old", gone, "old")),
        "stopped": _rows(("c-kept-stopped", gone, "kept"), ("c-old", gone, "old")),
        "running": _rows(("c-kept", gone, "kept", "running", "kept-main-1")),
    }

    def docker(args: list[str]) -> str | None:
        return listings["network"] if args[0] == "network" else None

    def listing(owned, states, fmt=None):
        return listings["running" if "running" in states else "stopped"]

    monkeypatch.setattr(_docker_sweep, "_docker", docker)
    monkeypatch.setattr(_docker_sweep, "_container_listing", listing)
    _docker_sweep.keep_project("kept")

    result = _docker_sweep.sweep_leftovers(dry_run=True)

    assert result.containers == ["c-old"]
    assert result.networks == ["net-old"]
    assert result.running == []
    _docker_sweep.unkeep_project("kept")
    assert _docker_sweep.kept_projects() == frozenset()


def test_a_sweep_with_another_home_spares_a_project_kept_on_the_daemon(
    monkeypatch, tmp_path
):
    """A sweep run with another HOME (CI, another user, a test run)
    cannot read this user's kept list, found the kept sandbox's process gone,
    and removed it. The marker on the Docker daemon keeps it."""
    gone = "otherhost:1:boot:1:ns"
    monkeypatch.setattr(_docker_sweep, "owner_state", lambda token: "gone")
    monkeypatch.setattr(_docker_sweep, "is_dead_running", lambda token: True)
    monkeypatch.setattr(_docker_sweep, "_docker", lambda args: "")
    monkeypatch.setattr(
        _docker_sweep,
        "_container_listing",
        lambda owned, states, fmt=None: (
            _rows(("c-kept", gone, "kept", "running", "kept-main-1"))
            if "running" in states
            else ""
        ),
    )
    _docker_sweep.keep_project("kept")
    # The other HOME: an empty kept list of its own.
    monkeypatch.setenv("BENCHFLOW_KEPT_SANDBOXES_DIR", str(tmp_path / "elsewhere"))

    assert _docker_sweep.kept_projects() == frozenset({"kept"})
    assert _docker_sweep.sweep_leftovers(dry_run=True).running == []
    _docker_sweep.unkeep_project("kept")
    assert _docker_sweep.sweep_leftovers(dry_run=True).running == ["c-kept"]


def test_the_daemon_marker_is_a_labelled_volume(monkeypatch):
    calls: list[list[str]] = []

    def docker(args: list[str]) -> str | None:
        calls.append(args)
        return (
            "kept-a\n\nnot a project!\nkept-b\n" if args[:2] == ["volume", "ls"] else ""
        )

    monkeypatch.setattr(_docker_sweep, "_docker", docker)
    assert REAL_MARK("kept-a") is True
    REAL_UNMARK("kept-a")
    assert REAL_MARKED() == frozenset({"kept-a", "kept-b"})
    assert calls == [
        ["volume", "create", "--label", "benchflow.kept=kept-a", "bf-kept-kept-a"],
        ["volume", "rm", "bf-kept-kept-a"],
        [
            "volume", "ls", "--filter", "label=benchflow.kept",
            "--format", '{{.Label "benchflow.kept"}}',
        ],
    ]  # fmt: skip
    monkeypatch.setattr(_docker_sweep, "_docker", lambda args: None)
    assert REAL_MARK("kept-a") is False  # no daemon: the user's own list only
    assert REAL_MARKED() == frozenset()


def _record(tmp_path: Path, **extra: Any) -> Path:
    trial = tmp_path / "trial"
    trial.mkdir(exist_ok=True)
    (trial / keep.RECORD_NAME).write_text(
        json.dumps({"version": 1, "state": "kept", **extra})
    )
    return trial


def test_attach_runs_a_command_in_the_main_container(tmp_path, monkeypatch):
    trial = _record(tmp_path, backend="docker", project="bf-p", service="main")
    calls: list[list[str]] = []

    def run(argv, **kwargs):
        calls.append(argv)
        out = "abc123\n" if argv[:2] == ["docker", "ps"] else ""
        return subprocess.CompletedProcess(argv, 7 if argv[1] == "exec" else 0, out, "")

    monkeypatch.setattr(sandbox_cli.subprocess, "run", run)
    with pytest.raises(typer.Exit) as exited:
        sandbox_cli.sandbox_attach(trial, ["cat", "/app/out.txt"])
    assert exited.value.exit_code == 7
    assert calls[0][:3] == ["docker", "ps", "-aq"]
    assert "label=com.docker.compose.project=bf-p" in calls[0]
    assert "label=com.docker.compose.service=main" in calls[0]
    assert calls[-1] == ["docker", "exec", "-u", "0", "abc123", "cat", "/app/out.txt"]


def test_release_removes_the_project_and_marks_the_record(tmp_path, monkeypatch):
    copy = tmp_path / "benchflow-task-abc"
    copy.mkdir()
    trial = _record(
        tmp_path, backend="docker", project="bf-p", service="main", task_copy=str(copy)
    )
    _docker_sweep.keep_project("bf-p")
    calls: list[list[str]] = []

    removed: list[str] = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "rm"]:
            removed.extend(argv[4:])
        listed = {
            "ps": "" if removed else "c1\nc2\n",
            "network": "n1\n",
            "volume": "",
        }
        out = listed.get(argv[1], "") if "ls" in argv or argv[1] == "ps" else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(sandbox_cli.subprocess, "run", run)
    sandbox_cli.sandbox_release(trial)

    assert ["docker", "rm", "-f", "-v", "c1", "c2"] in calls
    assert ["docker", "network", "rm", "n1"] in calls
    assert not copy.exists()
    assert _docker_sweep.kept_projects() == frozenset()
    record = json.loads((trial / keep.RECORD_NAME).read_text())
    assert record["state"] == "released" and record["released_at"]
    with pytest.raises(typer.Exit) as exited:
        sandbox_cli.sandbox_attach(trial, ["true"])
    assert exited.value.exit_code == 1


def test_a_release_docker_refuses_says_so_and_stays_kept(tmp_path, monkeypatch):
    """Round 4 of the release review: a network or volume Docker would not
    remove was ignored, and the sandbox was reported released."""
    trial = _record(tmp_path, backend="docker", project="bf-p", service="main")
    _docker_sweep.keep_project("bf-p")

    def run(argv, **kwargs):
        if argv[:3] == ["docker", "network", "rm"]:
            return subprocess.CompletedProcess(
                argv, 1, "", "network n1 has active endpoints"
            )
        listed = {"ps": "", "network": "n1\n", "volume": ""}
        out = listed.get(argv[1], "") if "ls" in argv or argv[1] == "ps" else ""
        return subprocess.CompletedProcess(argv, 0, out, "")

    monkeypatch.setattr(sandbox_cli.subprocess, "run", run)
    with pytest.raises(typer.Exit) as exited:
        sandbox_cli.sandbox_release(trial)
    assert exited.value.exit_code == 1
    record = json.loads((trial / keep.RECORD_NAME).read_text())
    assert record["state"] == "kept" and "released_at" not in record
    assert _docker_sweep.kept_projects() == frozenset({"bf-p"})


def test_attach_starts_a_stopped_daytona_sandbox_and_revokes_its_access(
    tmp_path, monkeypatch
):
    trial = _record(tmp_path, backend="daytona", sandbox_id="sb-9")
    events: list[Any] = []

    class Sandbox:
        state = "stopped"

        def start(self, timeout=None):
            events.append("start")

        def create_ssh_access(self, expires_in_minutes=None):
            events.append(("ssh", expires_in_minutes))
            return SimpleNamespace(token="tok-1")

        def revoke_ssh_access(self, token):
            events.append(("revoke", token))

    client = SimpleNamespace(
        get=lambda sandbox_id: events.append(sandbox_id) or Sandbox()
    )
    from benchflow.cli import main as cli_main

    monkeypatch.setattr(cli_main, "_daytona_client_or_exit", lambda **_: client)
    seen: list[list[str]] = []

    def run(argv, **kwargs):
        seen.append(argv)
        config = Path(argv[argv.index("-F") + 1]).read_text()
        assert "User tok-1" in config and "ssh.app.daytona.io" in config
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(sandbox_cli.subprocess, "run", run)
    with pytest.raises(typer.Exit) as exited:
        sandbox_cli.sandbox_attach(trial, ["ls", "/app"])
    assert exited.value.exit_code == 0
    assert events == ["sb-9", "start", ("ssh", 60), ("revoke", "tok-1")]
    assert seen[0][-2:] == ["benchflow-kept", "ls /app"]


def test_the_daytona_reaper_never_deletes_a_kept_sandbox(monkeypatch):
    """A kept sandbox stops itself when idle and its run's lease lapses: the
    reaper would delete it then. --id still does (the operator's decision)."""
    from benchflow.sandbox import daytona_reaper

    owned = daytona_reaper._BENCHFLOW_MANAGED_LABEL
    value = daytona_reaper._benchflow_managed_value()
    deleted: list[str] = []

    def sb(name: str, kept: bool) -> SimpleNamespace:
        labels = {owned: value, **({daytona_reaper.KEPT_LABEL: "1"} if kept else {})}
        return SimpleNamespace(
            id=name, labels=labels, state="stopped", created_at="2020-01-01T00:00:00Z"
        )

    sandboxes = [sb("kept", True), sb("old", False)]
    client = SimpleNamespace(
        list=lambda: iter(sandboxes), delete=lambda s, **kw: deleted.append(s.id)
    )
    reasons: list[str] = []
    daytona_reaper.reap_stale_sandboxes(
        client, max_age_minutes=1, on_skip=lambda s, why: reasons.append(why)
    )
    assert deleted == ["old"] and reasons == ["kept"]
    daytona_reaper.reap_stale_sandboxes(
        client, max_age_minutes=1, names=frozenset({"kept"})
    )
    # By --id the kept one goes too (the fake list still holds "old").
    assert deleted == ["old", "kept", "old"]


def test_doctor_does_not_count_a_kept_sandbox_as_left_behind(monkeypatch):
    """`bench doctor` advised `docker rm -f` for every container of an ended
    run, a kept sandbox included (a false warning once a sandbox can be kept)."""
    from benchflow import doctor

    monkeypatch.setattr(_docker_sweep, "owner_state", lambda token: "gone")
    listing = "kept000000001\ttok\tkept-project\trunning\nleft000000002\ttok\told-run\texited\n"
    probes = SimpleNamespace(
        run=lambda argv, timeout: SimpleNamespace(returncode=0, stdout=listing)
    )
    _docker_sweep.keep_project("kept-project")
    (row,) = doctor.check_docker_leftovers(probes)
    assert row.summary.startswith("1 container left by 1 ended run")
    assert "left00000000" in row.fix and "kept00000000" not in row.fix
    _docker_sweep.unkeep_project("kept-project")
    assert len(doctor.check_docker_leftovers(probes)) == 1


def test_the_run_summary_names_the_kept_trials(tmp_path):
    from benchflow.cli._shared import _kept_trials

    job = tmp_path / "job"
    for name, state in (("t-kept", "kept"), ("t-released", "released")):
        (job / name).mkdir(parents=True)
        (job / name / keep.RECORD_NAME).write_text(
            json.dumps({"state": state, "backend": "docker"})
        )
    (job / "t-plain").mkdir()
    assert _kept_trials(job) == [("t-kept", "docker")]
    assert _kept_trials(None) == []
