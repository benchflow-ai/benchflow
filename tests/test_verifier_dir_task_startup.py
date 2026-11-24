"""A task's own start-up code that gives /logs/verifier to another user, on real Docker.

Every service a task runs, main included, starts its own start-up code when BenchFlow releases it, just before the
start's mount probe closes /logs/verifier. An image whose entrypoint, as root, chowns the folder to the image's user
(the agent's, often) then raced that probe: a folder another uid owned was made root's, and the host could not remove
its own probe file, so the trial ended with no score; or the chown landed after the probe and the folder stayed the
agent's for its whole turn. Neither is the agent's doing: the trial is scored, with the folder closed again right
before the turn. A chown the turn itself makes keeps the close before test.sh: scored.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from benchflow.sandbox.docker import DockerSandbox
from benchflow.sandbox.protocol import ExecResult
from benchflow.task import RolloutPaths
from tests.integration.deterministic import harness as h

SANDBOX, SKIP_REASON = h.select_sandbox()
# The image's own user: neither root nor the host's.
OTHER = 4242
CHOWN = f"chown {OTHER}:{OTHER} /logs/verifier"


def _docker() -> None:
    if (
        shutil.which("docker") is None
        or subprocess.run(
            ["docker", "info"], capture_output=True, timeout=20
        ).returncode
        != 0
    ):
        pytest.skip("docker required")
    if os.getuid() in (0, OTHER):
        pytest.skip("the host's uid must be neither root nor the image user's")


@contextlib.contextmanager
def _mounted(host_dir: Path):
    """A container with *host_dir* bind-mounted at /logs/verifier, as Docker's start leaves main."""
    _docker()
    name = "bf-verifier-start-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "-v",
            f"{host_dir}:/logs/verifier",
            "python:3.12-slim",
            "sleep",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"cannot start a container: {run.stderr.strip()}")
    try:
        yield name
    finally:
        # Back to the host user, so the test's folder can be removed.
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0",
                name,
                "chown",
                "-R",
                f"{os.getuid()}:{os.getgid()}",
                "/logs/verifier",
            ],
            capture_output=True,
        )
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@contextlib.contextmanager
def _plain():
    """A container with nothing mounted: Daytona's Docker-in-Docker VM, whose commands run as root."""
    _docker()
    name = "bf-verifier-vm-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "none",
            "python:3.12-slim",
            "sleep",
            "600",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"cannot start a container: {run.stderr.strip()}")
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _root(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", "0", name, "sh", "-c", command],
        capture_output=True,
        text=True,
    )


def _exec_in(
    name: str,
    *,
    before: str | None = None,
    after: str | None = None,
    inside: str | None = None,
    inside_every: str | None = None,
):
    """The sandbox's exec, run as root in container *name*; *before* and *after* run there around it, as the task's
    start-up code landing just before or just after BenchFlow's command; *inside* runs in the first command between
    its close and its judgment of the folder, *inside_every* in each one (a process that keeps giving it away)."""
    pending = [inside] if inside else []

    async def exec_(command, *args, **kwargs):
        if before:
            await asyncio.to_thread(_root, name, before)
        command = str(command)
        if pending and "bf_vd=$(stat" in command:
            command = command.replace(
                "bf_vd=$(stat", f"{pending.pop()}; bf_vd=$(stat", 1
            )
        if inside_every:
            command = command.replace("bf_vd=$(stat", f"{inside_every}; bf_vd=$(stat")
        done = await asyncio.to_thread(_root, name, command)
        if after:
            await asyncio.to_thread(_root, name, after)
        return ExecResult(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    return exec_


def _sandbox(tmp_path: Path) -> DockerSandbox:
    sandbox = object.__new__(DockerSandbox)
    paths = RolloutPaths(rollout_dir=tmp_path / "run")
    paths.mkdir()
    sandbox.rollout_paths = paths
    sandbox.logger = logging.getLogger("test")
    sandbox._verifier_service = "main"
    return sandbox


def _host_writes(folder: Path) -> bool:
    record = folder / f"record-{uuid.uuid4().hex[:6]}"
    try:
        record.write_text("")
    except PermissionError:
        return False
    record.unlink()
    return True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "when", ["before-write", "before-exec", "inside-exec", "after-exec"]
)
async def test_the_probe_survives_the_task_giving_the_folder_to_another_uid(
    tmp_path, when
):
    """The task's start-up code gives the host's folder to uid 4242 before the mount probe's file is written, between
    that write and the probe's exec, inside the exec between its close and its judgment, or right after the exec: the
    probe raises nothing, finds the mount and leaves no file; then the close before the turn leaves the folder the
    host user's, 0755, which the host can write and uid 4242 can't.

    Mutants: the probe's file removed by the host only (``after-exec`` raises); a folder the probe's own judgment
    found open refused, not closed again (``inside-exec`` raises)."""
    sandbox = _sandbox(tmp_path)
    folder = tmp_path / "run" / "verifier"
    with _mounted(folder) as name:
        if when == "before-write":
            assert _root(name, CHOWN).returncode == 0
        sandbox.exec = _exec_in(
            name,
            before=CHOWN if when == "before-exec" else None,
            after=CHOWN if when == "after-exec" else None,
            inside=CHOWN if when == "inside-exec" else None,
        )
        await sandbox._probe_verifier_log_mount()
        assert sandbox.is_mounted
        assert not list(folder.glob(".runtime-mount-probe-*"))
        sandbox.exec = _exec_in(name)
        await sandbox.close_verifier_dir_before_turn()
        st = folder.stat()
        assert (st.st_uid, st.st_mode & 0o777) == (os.getuid(), 0o755), (
            st.st_uid,
            oct(st.st_mode),
        )
        assert _host_writes(folder), (
            "BenchFlow on the host can't write its records there"
        )
        planted = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                str(OTHER),
                name,
                "touch",
                "/logs/verifier/planted",
            ],
            capture_output=True,
        )
        assert planted.returncode != 0, "uid 4242 wrote in /logs/verifier"


AS_OTHER = f"setpriv --reuid={OTHER} --regid={OTHER} --clear-groups"


def _rollout(sandbox, *, earlier_turns: int = 0):
    """A rollout around *sandbox* that has closed the folder before *earlier_turns* turns."""
    from benchflow.rollout import Rollout

    rollout = object.__new__(Rollout)
    rollout._env = sandbox
    if earlier_turns:
        rollout._turn_closes = earlier_turns
    return rollout


@pytest.mark.asyncio
async def test_a_dind_main_that_gives_the_folder_away_finds_it_closed_at_the_turn():
    """Daytona with Compose (Docker-in-Docker), its VM a container whose commands run as root. Setup closes the VM
    folder Compose bind-mounts at /logs/verifier; main's entrypoint, released after that, gives it to uid 4242;
    the close before the turn, through the rollout's hook, leaves it root's 0755, which uid 4242 can't write.

    Mutant: no close before the turn on Docker-in-Docker."""
    from types import SimpleNamespace

    from benchflow.sandbox.daytona import DaytonaSandbox, _DaytonaDinD
    from benchflow.sandbox.daytona_dind import dind_log_dirs_command

    folder = f"{_DaytonaDinD._LOGS_DIR}/verifier"
    with _plain() as name:
        setup = _root(name, dind_log_dirs_command(_DaytonaDinD._LOGS_DIR, []))
        assert setup.returncode == 0 and setup.stdout == "", setup
        assert _root(name, f"chown {OTHER}:{OTHER} {folder}").returncode == 0
        strategy = _DaytonaDinD.__new__(_DaytonaDinD)
        strategy._env = SimpleNamespace(logger=logging.getLogger("test"))
        strategy._vm_exec = _exec_in(name)
        sandbox = object.__new__(DaytonaSandbox)
        sandbox._strategy = strategy
        await _rollout(sandbox)._close_verifier_dir_before_turn()
        shown = _root(
            name,
            f"stat -c '%u %a' {folder}; "
            f"{AS_OTHER} touch {folder}/planted 2>/dev/null && echo planted || echo refused",
        ).stdout.split()
    assert shown == ["0", "755", "refused"], shown


@pytest.mark.asyncio
@pytest.mark.parametrize("owned", [False, True], ids=["own", "caller-owned"])
async def test_a_caller_owned_sandbox_never_counts_as_a_first_turn(owned):
    """A rollout on a live sandbox the caller owns (Rollout.use_prebuilt_env) may follow another rollout's agent
    there, whose root process may still run: its first close is a later turn's, warned and never refused.

    Mutant: a caller-owned sandbox counted as a first turn."""
    seen: list[bool] = []

    class _Box:
        async def close_verifier_dir_before_turn(self, *, first_turn):
            seen.append(first_turn)

    rollout = _rollout(_Box())
    rollout._env_externally_owned = owned
    await rollout._close_verifier_dir_before_turn()
    assert seen == [not owned]


@pytest.mark.asyncio
@pytest.mark.parametrize("turn", ["later", "first"])
async def test_a_folder_kept_open_is_refused_only_before_the_first_turn(tmp_path, turn):
    """A process gives /logs/verifier to uid 4242 again inside every close: a root one left from an earlier turn,
    or, before the first, the task's own start-up loop. Before a later turn the trial is not refused: it goes on,
    with a warning in its result naming the folder and its owner (the close before test.sh still holds the score).
    Before the first turn, where only the task's code has run, it is refused as at setup.

    Mutant: refuse before a later turn too (``later`` raises)."""
    from benchflow._utils.scoring import INTEGRITY_UNAVAILABLE, classify_error

    sandbox = _sandbox(tmp_path)
    folder = tmp_path / "run" / "verifier"
    with _mounted(folder) as name:
        sandbox._logs_are_mounted = True
        sandbox._host_owner_in_sandbox = (os.getuid(), os.getgid())
        sandbox.exec = _exec_in(name, inside_every=CHOWN)
        rollout = _rollout(sandbox, earlier_turns=1 if turn == "later" else 0)
        if turn == "later":
            await rollout._close_verifier_dir_before_turn()
            warned = [
                w for w in vars(rollout).get("_warnings", []) if "/logs/verifier" in w
            ]
            assert len(warned) == 1 and f"uid {OTHER}'s" in warned[0], warned
            return
        with pytest.raises(Exception) as caught:
            await rollout._close_verifier_dir_before_turn()
    assert classify_error(str(caught.value)) == INTEGRITY_UNAVAILABLE, str(caught.value)


# -- Whole trials: a real `bench` on Docker ----------------------------------------------------------------------

TASK_TOML = """version = "1.0"

[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "sanity"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 120.0
{agent_user}
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 512
"""

IMAGE = f"""FROM ubuntu:24.04
RUN useradd -m -u {OTHER} app && mkdir -p /app && chown app /app
WORKDIR /app
"""

# As root at its release: the folder goes to the image's user, then the service says it is up.
ENTRY_READY = f"""#!/bin/bash
mkdir -p /logs/verifier
{CHOWN}
echo ready > /tmp/start-chowned
exec "$@"
"""

# As root at its release: the folder goes to the image's user right after the start's mount probe is gone (no
# healthcheck waits for it).
ENTRY_LATE = f"""#!/bin/bash
mkdir -p /logs/verifier
(
  shopt -s nullglob
  seen=
  for _ in $(seq 1 24000); do
    probes=(/logs/verifier/.runtime-mount-probe-*)
    if [ ${{#probes[@]}} -gt 0 ]; then seen=1; elif [ -n "$seen" ]; then break; fi
    sleep 0.005
  done
  {CHOWN}
  echo "${{seen:+after-probe}}" > /tmp/start-chowned
) &
exec "$@"
"""

# solve.sh, as the task's [agent] user (uid 4242): may it make a name in /logs/verifier?
SOLVE_AS_APP = """#!/bin/bash
if touch /logs/verifier/planted 2>/dev/null; then s=open; else s=closed; fi
echo "$s $(cat /tmp/start-chowned 2>/dev/null || echo none)" > /tmp/state.txt
"""

TEST_SH = """#!/bin/bash
cat /tmp/state.txt
read -r s how < /tmp/state.txt
if [ "$s" = closed ] && [ "$how" = {expect} ]; then echo 1; else echo 0; fi > /logs/verifier/reward.txt
"""

# The turn, as root (solve.sh with no [agent] user): the folder goes to uid 4242, which plants a reward there.
TURN_AS_ROOT = f"""#!/bin/bash
{CHOWN}
setpriv --reuid={OTHER} --regid={OTHER} --clear-groups sh -c 'echo 1 > /logs/verifier/reward.txt'
"""

TEST_SH_TURN = """#!/bin/bash
echo 0 > /logs/verifier/reward.txt
"""


def _write(
    root: Path,
    name: str,
    *,
    image: str,
    test_sh: str,
    solve: str = "",
    entry: str = "",
    agent_user: str = "",
    healthy: bool = False,
) -> None:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(TASK_TOML.format(agent_user=agent_user))
    (task / "instruction.md").write_text("Write /tmp/state.txt.\n")
    dockerfile = image
    if entry:
        (task / "environment" / "entry.sh").write_text(entry)
        dockerfile += "COPY entry.sh /entry.sh\nRUN chmod 755 /entry.sh\n"
        if healthy:
            dockerfile += "HEALTHCHECK --interval=1s --timeout=2s --retries=60 CMD test -f /tmp/start-chowned\n"
        dockerfile += 'ENTRYPOINT ["/entry.sh"]\n'
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    for path, text in (
        (task / "solution" / "solve.sh", solve or "#!/bin/bash\n"),
        (task / "tests" / "test.sh", test_sh),
    ):
        path.write_text(text)
        path.chmod(0o755)


def _bench(tasks: Path, jobs: Path, agent: str, *extra: str) -> str:
    env = {
        **os.environ,
        "BENCHFLOW_SKIP_UPDATE_CHECK": "1",
        "HARBOR_TELEMETRY": "off",
        "DO_NOT_TRACK": "1",
    }
    done = subprocess.run(
        [
            h.bench_executable(),
            "eval",
            "run",
            "--tasks-dir",
            str(tasks),
            "--agent",
            agent,
            "--sandbox",
            "docker",
            "--concurrency",
            "2",
            "--retry-attempts",
            "0",
            "--jobs-dir",
            str(jobs),
            *extra,
        ],
        cwd=h.REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=1800,
    )
    return done.stdout


def _trials(jobs: Path) -> dict[str, dict]:
    return {
        result.parent.name.rsplit("__", 1)[0]: {
            **json.loads(result.read_text()),
            "_dir": str(result.parent),
        }
        for result in jobs.glob("*/*__*/result.json")
    }


@pytest.fixture(scope="module")
def trials(tmp_path_factory) -> dict[str, dict]:
    if SANDBOX != "docker":
        pytest.skip(SKIP_REASON if SANDBOX is None else "Docker only")
    root = tmp_path_factory.mktemp("verifier-dir-start")
    started = root / "started"
    app = 'user = "app"\n'
    _write(
        started,
        "start-chown-ready",
        image=IMAGE,
        entry=ENTRY_READY,
        solve=SOLVE_AS_APP,
        test_sh=TEST_SH.format(expect="ready"),
        agent_user=app,
        healthy=True,
    )
    _write(
        started,
        "start-chown-late",
        image=IMAGE,
        entry=ENTRY_LATE,
        solve=SOLVE_AS_APP,
        test_sh=TEST_SH.format(expect="after-probe"),
        agent_user=app,
    )
    _write(
        started,
        "turn-chown-root",
        image=IMAGE,
        solve=TURN_AS_ROOT,
        test_sh=TEST_SH_TURN,
    )
    out = _bench(started, root / "jobs", "oracle")
    found = _trials(root / "jobs")
    assert sorted(found) == sorted(NAMES), out[-4000:]
    return found


NAMES = ("start-chown-ready", "start-chown-late", "turn-chown-root")


def _scored(trial: dict) -> float:
    reward = (trial.get("rewards") or {}).get("reward")
    assert trial.get("error") is None and reward is not None, (
        trial.get("error"),
        trial.get("_dir"),
    )
    return reward


@pytest.mark.deterministic
@pytest.mark.parametrize("name", ["start-chown-ready", "start-chown-late"])
def test_a_task_that_gives_the_folder_to_its_user_at_start_is_scored_and_closed(
    trials, name
):
    """The task's entrypoint, as root, gives /logs/verifier to the image's user (the task's [agent] user), before the
    service reports up or right after the mount probe: the trial is scored, and solve.sh, as that user, finds the
    folder closed (reward 1). Was unscored (ready) and open to the agent (late).

    Mutant: no close before the turn (late scores 0)."""
    assert _scored(trials[name]) == 1.0, trials[name].get("_dir")


@pytest.mark.deterministic
def test_a_chown_in_the_turn_keeps_the_close_before_test_sh(trials):
    """The turn, as root, gives /logs/verifier to uid 4242, which plants reward.txt there: the close before test.sh
    sweeps it (test.sh's 0 is the score) and the trial is scored, never unscored; the close before the turn, which
    ran before it, changes nothing here."""
    trial = trials["turn-chown-root"]
    assert _scored(trial) == 0.0, trial.get("_dir")
