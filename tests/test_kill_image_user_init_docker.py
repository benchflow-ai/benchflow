"""The kill before grading in an image whose USER is the sandbox user.

Guards the kill's PID 1 rule and the keep-alive as PID 1 against the
regression of the kill before grading (c5ab45031, 74c0e84eb) meeting the
compose keep-alive (7a479e9f4, #294). On Docker PID 1 runs as the image's
user; Terminal-Bench 4.1's rs-archive-clone builds ``USER agent``, the
account BenchFlow's sandbox user is named for, so PID 1 and its keep-alive
were the agent's by UID. The kill ended the keep-alive's ``sleep``, PID 1 (a
shell waiting on it) exited, and the container went down with every process
in it, BenchFlow's root command helper included: the trial errored with "the
sandbox's command channel stopped: the helper closed its output", in the
quiesce of a separate verifier's capture and in the hardening of a shared one.
Under a user that is not root the inert bootstrap now makes the keep-alive
PID 1 itself; a root PID 1 stays the shell, which reaps orphans.

Deterministic tier (``tests/integration/README.md``): real Docker, no model
(the oracle runs the solution as the agent user). The solution leaves
processes behind, among them a writer that would keep changing the workspace
during grading and one whose command line is the keep-alive's. Each trial
scores 1 only when the kill left PID 1 running and every other process of
the agent's user dead: the shared verifier's test.sh checks both, the separate
one gets the answer only if the agent's container survived its quiesce.

The kill may leave that PID 1 alone only where nothing of the agent's could
have traced it: Yama's ptrace_scope 1 to 3. At 0 (Debian 12's default) or on a
kernel without Yama the sandbox is refused when it starts, before the oracle
runs (sandbox/pid1_ptrace.py), so these trials assert the refusal there
and the kill everywhere else; this host's scope is read from the kernel the
local Docker daemon shares. An image that ends as root is graded under any
scope. Every trial records the scope (result.json's host_ptrace_scope).

The kill before grading is strict's (the faithful default, 09:58Z), so these
trials run under ``--integrity strict``. In audit the agent's processes live
through grading and end after it (S2, 12:36Z) under the same PID 1 rule: the
audit trial scores what its test.sh says of them, and does not error.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic
# Whole trials on real Docker: the exec channel and the side-by-side steps
# before the verifier stay on, as in a run (tests/conftest.py).
DOCKER_RUN_DEFAULTS = True

SANDBOX, SKIP_REASON = h.select_sandbox()
UID = 10001


def _host_scope() -> int | str | None:
    """This kernel's Yama ptrace_scope, which a local daemon's containers read."""
    from benchflow.sandbox import pid1_ptrace

    path = Path(pid1_ptrace.PTRACE_SCOPE_FILE)
    if not path.exists():
        return pid1_ptrace.NO_YAMA
    return pid1_ptrace.parse_scope(path.read_text())


HOST_SCOPE = _host_scope()

TASK_TOML = """version = "1.0"
{artifacts}
[metadata]
author_name = "benchflow"
difficulty = "easy"
category = "fixture"

[verifier]
timeout_sec = 120.0
user = "root"
{mode}
[agent]
timeout_sec = 120.0
user = "agent"

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
"""
INSTRUCTION = "Write ok to /app/answer.txt.\n"
DOCKERFILE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    f"RUN useradd --create-home --uid {UID} agent && mkdir -p /app /logs/verifier "
    "/logs/agent /logs/artifacts && chmod 1777 /app\n"
    "WORKDIR /app\n"
    "USER agent\n"
)
SOLVE_SH = """#!/bin/bash
echo ok > /app/answer.txt
nohup sleep 3600 >/dev/null 2>&1 </dev/null &
nohup sleep infinity >/dev/null 2>&1 </dev/null &
# Reparented when its parent exits at once.
setsid bash -c 'sleep 3600 >/dev/null 2>&1 </dev/null & exit 0'
nohup bash -c 'while :; do date +%s%N > /app/clock.txt; sleep 0.1; done' \
    >/dev/null 2>&1 </dev/null &
sleep 0.5
"""
SHARED_TEST_SH = f"""#!/bin/bash
fail() {{ echo "FAIL: $*"; echo 0 > /logs/verifier/reward.txt; exit 0; }}
[ "$(cat /app/answer.txt 2>/dev/null)" = ok ] || fail "no answer"
init=$(tr '\\000' ' ' < /proc/1/cmdline)
[ "$init" = "sleep infinity " ] || fail "PID 1 is $init"
left=
for d in /proc/[0-9]*; do
    p=${{d#/proc/}}
    [ "$p" = 1 ] && continue
    # A zombie of one thread is dead: PID 1, a sleep, never reaps the killed.
    zombie=$(awk '/^State:/{{s=$2}} /^Threads:/{{t=$2}} END{{print s t}}' "$d/status" 2>/dev/null)
    [ "$zombie" = Z1 ] && continue
    ids=$(awk '/^Uid:/{{print " "$2" "$3" "$4" "}}' "$d/status" 2>/dev/null) || continue
    case "$ids" in *" {UID} "*) left="$left $p:$(tr '\\000' ' ' < "$d/cmdline")" ;; esac
done
[ -z "$left" ] || fail "processes of the agent's user left:$left"
a=$(cat /app/clock.txt 2>/dev/null); sleep 1; b=$(cat /app/clock.txt 2>/dev/null)
[ -n "$a" ] && [ "$a" = "$b" ] || fail "the workspace changed during grading"
echo PASS
echo 1 > /logs/verifier/reward.txt
"""
SEPARATE_TEST_SH = """#!/bin/bash
if [ "$(cat /app/answer.txt 2>/dev/null)" = ok ]; then r=1; else r=0; fi
echo "$r" > /logs/verifier/reward.txt
"""
# An image that ends as root (most tasks'): PID 1 is root's shell, the agent
# user is BenchFlow's own account, and the kill ends every process of it.
ROOT_DOCKERFILE = (
    "FROM python:3.12-slim@sha256:"
    "a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1\n"
    "RUN mkdir -p /app /logs/verifier /logs/agent /logs/artifacts && chmod 1777 /app\n"
    "WORKDIR /app\n"
)
ROOT_TEST_SH = """#!/bin/bash
fail() { echo "FAIL: $*"; echo 0 > /logs/verifier/reward.txt; exit 0; }
[ "$(cat /app/answer.txt 2>/dev/null)" = ok ] || fail "no answer"
uid=$(id -u agent) || fail "no agent account"
[ "$(awk '/^Uid:/{print $2}' /proc/1/status)" = 0 ] || fail "PID 1 is not root's"
left=
for d in /proc/[0-9]*; do
    p=${d#/proc/}
    zombie=$(awk '/^State:/{s=$2} /^Threads:/{t=$2} END{print s t}' "$d/status" 2>/dev/null)
    [ "$zombie" = Z1 ] && continue
    ids=$(awk '/^Uid:/{print " "$2" "$3" "$4" "}' "$d/status" 2>/dev/null) || continue
    case "$ids" in *" $uid "*) left="$left $p:$(tr '\\000' ' ' < "$d/cmdline")" ;; esac
done
[ -z "$left" ] || fail "processes of the agent's user left:$left"
a=$(cat /app/clock.txt 2>/dev/null); sleep 1; b=$(cat /app/clock.txt 2>/dev/null)
[ -n "$a" ] && [ "$a" = "$b" ] || fail "the workspace changed during grading"
echo PASS
echo 1 > /logs/verifier/reward.txt
"""
MODES = {
    "shared": ("", "", SHARED_TEST_SH),
    "separate": (
        'artifacts = ["/app/answer.txt"]\n',
        'environment_mode = "separate"\n',
        SEPARATE_TEST_SH,
    ),
}


@pytest.fixture
def require_docker() -> None:
    if SANDBOX != "docker":
        reason = SKIP_REASON if SANDBOX is None else f"Docker-only scenario ({SANDBOX})"
        pytest.skip(reason)


def _task(root: Path, mode: str, *, root_image: bool = False) -> Path:
    artifacts, environment_mode, test_sh = MODES[mode]
    if root_image:
        test_sh = ROOT_TEST_SH
    task = root / f"{'root-image' if root_image else 'image-user'}-{mode}"
    for sub in ("environment", "solution", "tests"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(
        TASK_TOML.format(artifacts=artifacts, mode=environment_mode)
    )
    (task / "instruction.md").write_text(INSTRUCTION)
    (task / "environment" / "Dockerfile").write_text(
        ROOT_DOCKERFILE if root_image else DOCKERFILE
    )
    for path, text in (
        (task / "solution" / "solve.sh", SOLVE_SH),
        (task / "tests" / "test.sh", test_sh),
    ):
        path.write_text(text)
        path.chmod(0o755)
    return task


@pytest.mark.parametrize(
    ("user", "pid_1"),
    [
        ("agent", "exec sleep infinity"),
        ("10001:10001", "exec sleep infinity"),
        ("", "sleep infinity"),
        ("root", "sleep infinity"),
        ("0:0", "sleep infinity"),
    ],
)
def test_the_keep_alive_is_pid_1_itself_only_under_a_user_that_is_not_root(user, pid_1):
    """No shell PID 1 waits on a sleep the kill before grading may end; a
    root PID 1 keeps the shell, which reaps the orphans the kernel gives it
    (a sleep reaps none). A task's own command is never changed."""
    from benchflow.sandbox import _inert_bootstrap as bootstrap

    def argv(cmd: list[str]) -> list[str]:
        config = {"Cmd": cmd, "Env": ["PATH=/usr/bin:/bin"], "User": user}
        return bootstrap.entrypoint(config)[5:]

    assert argv(["sh", "-c", "sleep infinity"]) == ["sh", "-c", pid_1]
    assert argv(["sh", "-c", "sleep 99"]) == ["sh", "-c", "sleep 99"]
    assert argv(["sleep", "infinity"]) == ["sleep", "infinity"]


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("mode", list(MODES))
async def test_the_kill_spares_pid_1_and_ends_every_agent_process(tmp_path, mode):
    import benchflow as bf
    from benchflow.sandbox.docker import DockerSandbox

    assert DockerSandbox.exec_channel is True
    task = _task(tmp_path / "tasks", mode)
    jobs = tmp_path / "jobs" / mode
    # Strict: the kill before grading is strict's alone since the faithful
    # default (the lead, 09:58Z); audit and off grade beside the
    # agent's processes and end them after (S2, 12:36Z), which
    # test_audit_ends_the_agent_processes_after_grading_and_spares_pid_1 checks.
    result = await bf.run(
        bf.RolloutConfig(
            task_path=task,
            agent="oracle",
            jobs_dir=jobs,
            environment="docker",
            integrity="strict",
        )
    )
    outputs = sorted(jobs.glob("**/verifier/test-stdout.txt"))
    said = "\n".join(path.read_text() for path in outputs)
    assert _recorded_scope(jobs) == HOST_SCOPE
    from benchflow.sandbox import pid1_ptrace

    if pid1_ptrace.unprotected(HOST_SCOPE):
        # Refused when the sandbox started: nothing of the agent's ran, so
        # there was nothing to kill, and the trial is never retried.
        refused = str(pid1_ptrace.refusal("agent", HOST_SCOPE))
        assert result.error is not None and refused in result.error, result.error
        assert result.rewards is None
        assert not list(jobs.glob("**/kill-before-grading.json"))
        assert not outputs
        return
    assert result.error is None, result.error
    assert result.rewards == {"reward": 1.0}, said
    (record,) = jobs.glob("**/kill-before-grading.json")
    # The turn-end kill and the one before grading: the four leftovers (and
    # the clock's sleep, when it ran), never PID 1.
    assert json.loads(record.read_text())["killed_total"] >= 4, record.read_text()


def _recorded_scope(jobs: Path) -> int | str | None:
    """The trial's result.json's host_ptrace_scope (every trial records it)."""
    (result,) = [json.loads(path.read_text()) for path in jobs.glob("*/*/result.json")]
    assert "host_ptrace_scope" in result, sorted(result)
    return result["host_ptrace_scope"]


@pytest.mark.usefixtures("require_docker")
async def test_an_image_that_ends_as_root_is_graded_under_any_scope(tmp_path):
    """Most tasks' images end as root: PID 1 is root's, no process of the
    agent's can trace it whatever the scope, and nothing is refused. Audit
    grades beside the agent's leftovers (S2, 12:36Z): test.sh sees them and
    says 0, and no kill before grading is recorded. Strict's separate
    verifier has no agent account for this test.sh to check."""
    import benchflow as bf

    task = _task(tmp_path / "tasks", "shared", root_image=True)
    jobs = tmp_path / "jobs" / "root-image"
    result = await bf.run(
        bf.RolloutConfig(
            task_path=task,
            agent="oracle",
            jobs_dir=jobs,
            environment="docker",
            integrity="audit",
        )
    )
    said = "\n".join(
        path.read_text() for path in sorted(jobs.glob("**/verifier/test-stdout.txt"))
    )
    assert result.error is None, result.error
    assert result.rewards == {"reward": 0.0}, said
    assert "processes of the agent's user left" in said, said
    assert _recorded_scope(jobs) == HOST_SCOPE
    assert not list(jobs.glob("**/kill-before-grading.json"))


@pytest.mark.usefixtures("require_docker")
async def test_audit_ends_the_agent_processes_after_grading_and_spares_pid_1(tmp_path):
    """Audit grades beside the agent's leftovers and ends them only after
    test.sh (S2, 12:36Z): the shared test.sh sees them and says 0, no kill
    before grading is recorded, and the kill after grading, under the same PID
    1 rule, leaves the container up for the trial's own end (no error)."""
    import benchflow as bf
    from benchflow.sandbox import pid1_ptrace

    if pid1_ptrace.unprotected(HOST_SCOPE):
        pytest.skip("this host's ptrace scope refuses the image-user sandbox at start")
    task = _task(tmp_path / "tasks", "shared")
    jobs = tmp_path / "jobs" / "audit"
    result = await bf.run(
        bf.RolloutConfig(
            task_path=task,
            agent="oracle",
            jobs_dir=jobs,
            environment="docker",
            integrity="audit",
        )
    )
    said = "\n".join(
        path.read_text() for path in sorted(jobs.glob("**/verifier/test-stdout.txt"))
    )
    assert result.error is None, result.error
    assert result.verifier_error is None, result.verifier_error
    assert result.rewards == {"reward": 0.0}, said
    assert "processes of the agent's user left" in said, said
    assert not list(jobs.glob("**/kill-before-grading.json"))
