"""K1-R15-2 in real containers (VM Docker): two accounts restarting each
other across the per-account clean-up.

The b133 review's input: a task lets ``agent`` switch to ``worker`` and
back. The agent leaves one detached script per account; each catches
SIGTERM, starts the other account's script through the setuid switch
program, then exits. The kills before grading clean one account, then the
other, and each per-account check passes while a writer survives.

``su`` with ``pam_wheel trust`` stands in for the task's ``sudo -u`` rule
(the image has no sudo and the test needs no network): both are setuid-root
programs, which no_new_privs disarms. The containers:

- no container no_new_privs, the agent's commands as integ's native route
  sent them under the opt-in (bare): the hazard itself, why such a task is
  now refused on those backends before start;
- no container no_new_privs, the agent's commands through this branch's
  generic exec (what a native provider runs): the barrier holds, nothing
  survives. At 6c7d11fa5 the opt-in returned the bare command and this
  failed;
- the Docker family's creation-time no_new_privs, where the
  task still runs: nothing survives, in both kill orders;
- the kept-service control: the same input with a kept listener on the
  Docker family; the listener is held and released, every other process of
  both accounts is gone;
- the kill before grading as ``bfx kill`` (TRUSTED-RUNTIME v2 §4, K1):
  the pinned bfx bound read-only at ``/.bfx`` and the accounts' pre-agent
  numbers; the same outcome, from bfx's receipt.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import subprocess
import time
import uuid
from types import SimpleNamespace

import pytest

from tests.test_c8_protected_runtime_container import _docker_ready, _sandbox

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGE = "python:3.12-slim"
LABEL = os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "target=K1-R15-2")
PREFIX = os.environ.get("BENCHFLOW_TEST_CONTAINER_PREFIX", "k1r152-")
ACCOUNTS = ("agent", "worker")
PORT = 5000

# Root's set-up, as the task's image would have it: two accounts that may
# switch to each other without a password, and the agent's script.
_PREP = r"""
set -e
groupadd switch
useradd -m -G switch agent
useradd -m -G switch worker
sed -i '1i auth sufficient pam_wheel.so trust use_uid group=switch' /etc/pam.d/su
cat > /opt/pp.sh <<'EOF'
#!/bin/sh
# Asked to stop, start the other account's copy, then go.
me=$(id -un); other=$1
trap 'su -s /bin/sh "$other" -c "setsid /opt/pp.sh $me </dev/null >/dev/null 2>&1 &" </dev/null >/dev/null 2>&1; exit 0' TERM
while :; do sleep 0.1; done
EOF
chmod 755 /opt/pp.sh
mkdir -p /logs/verifier
"""

# The agent's command: its own copy, and the worker's through the switch.
AGENT_COMMAND = (
    "setsid /opt/pp.sh worker </dev/null >/dev/null 2>&1 & "
    "su -s /bin/sh worker -c "
    "'setsid /opt/pp.sh agent </dev/null >/dev/null 2>&1 &' "
    "</dev/null >/dev/null 2>&1; "
    "sleep 0.5"
)
LISTENER = (
    f"setsid python3 -m http.server {PORT} --bind 127.0.0.1 "
    "</dev/null >/dev/null 2>&1 &"
)


def _dx(name: str, argv: list[str], user: str = "0") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", user, name, *argv],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _bfx_mount() -> list[str]:
    from benchflow._bfx import delivery
    from benchflow._bfx.locate import BfxUnavailable

    try:
        folder = delivery.host_dir("x86_64")
    except BfxUnavailable as exc:
        pytest.skip(f"no pinned bfx in this checkout: {exc}")
    return delivery.run_args(delivery.BIND, source=str(folder))


def _box(*, no_new_privs: bool, freezer: bool = False, bfx: bool = False):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = PREFIX + uuid.uuid4().hex[:8]
    flags = ["--network", "none"]
    if bfx:
        flags += _bfx_mount()
    if no_new_privs:
        flags += ["--security-opt", "no-new-privileges:true"]
    if freezer:
        # As docker-compose-kept-freezer.yaml stacks it for a keeping task.
        flags += ["--cgroupns", "private", "--security-opt", "writable-cgroups=true"]
    key, _, value = LABEL.partition("=")
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", f"{key}={value}",
         *flags, IMAGE, "sleep", "600"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {IMAGE}: {run.stderr.strip()}")
    made = _dx(name, ["/bin/sh", "-c", _PREP])
    if made.returncode != 0:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
        raise AssertionError(made.stderr)
    return name


@pytest.fixture
def boxes():
    made: list[str] = []

    def make(**kwargs):
        made.append(_box(**kwargs))
        return made[-1]

    try:
        yield make
    finally:
        for name in made:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _env(name: str):
    sandbox = _sandbox(name)
    sandbox.agent_identity_switch = ("worker",)
    return sandbox


async def _bfx_env(name: str):
    """_env with bfx delivered and the accounts measured before the agent,
    counting the bfx commands it runs."""
    from benchflow.sandbox import _identities
    from benchflow.sandbox.protocol import ExecResult

    sandbox = _env(name)
    sandbox.bfx_delivery = "bind"

    async def root_exec(command: str, **kwargs):
        done = _dx(name, ["/bin/sh", "-c", command])
        return ExecResult(done.returncode, done.stdout, done.stderr)

    await _identities.snapshot(sandbox, root_exec=root_exec, config={"User": ""})
    sandbox.bfx_commands = []
    call = type(sandbox).bfx_call

    async def counted(cmd, args, **kwargs):
        sandbox.bfx_commands.append(cmd)
        return await call(sandbox, cmd, args, **kwargs)

    sandbox.bfx_call = counted
    return sandbox


def _run_as_agent(name: str, command: str, *, route: str) -> None:
    """Start the agent's command as a provider would run it."""
    from benchflow.sandbox.agent_exec import (
        AgentExecCommand,
        agent_exec_call,
        native_agent_argv,
    )

    if route == "bare":
        argv = ["/bin/sh", "-c", command]
    else:
        # A native provider (native Daytona, Modal, AgentCore, Apple): the
        # account it selects, then the whole dispatch in native_agent_argv
        # when generic exec wrapped the command.
        sandbox = SimpleNamespace(
            agent_exec_shell="/bin/sh", agent_container_no_new_privileges=False
        )
        wrapped, user = agent_exec_call(
            command, "agent", identity_switch=("worker",), sandbox=sandbox
        )
        assert user == "agent"
        argv = (
            native_agent_argv(wrapped)
            if isinstance(wrapped, AgentExecCommand)
            else ["/bin/sh", "-c", wrapped]
        )
    done = _dx(name, argv, user="agent")
    assert done.returncode == 0, (done.stdout, done.stderr)


def _population(name: str) -> dict[str, list[str]]:
    done = _dx(name, ["ps", "-eo", "user=,pid=,args="])
    if done.returncode != 0:
        # python:3.12-slim has no procps: read /proc as root.
        done = _dx(
            name,
            [
                "python3",
                "-c",
                "import os,pwd\n"
                "for p in os.listdir('/proc'):\n"
                " if not p.isdigit(): continue\n"
                " try:\n"
                "  u=pwd.getpwuid(os.stat('/proc/'+p).st_uid).pw_name\n"
                "  a=open('/proc/'+p+'/cmdline','rb').read().replace(b'\\0',b' ').decode().strip()\n"
                " except (OSError, KeyError): continue\n"
                " if a: print(u,p,a)",
            ],
        )
    assert done.returncode == 0, done.stderr
    found: dict[str, list[str]] = {account: [] for account in ACCOUNTS}
    for line in done.stdout.splitlines():
        user, _, rest = line.strip().partition(" ")
        if user in found:
            found[user].append(rest)
    return found


def _settle(name: str) -> dict[str, list[str]]:
    # A grace handler's restart lands within a second or two.
    time.sleep(2)
    return _population(name)


async def _before_grading(env) -> None:
    """harden_before_verify's order: the agent user, then the accounts."""
    from benchflow.sandbox.lockdown import (
        _kill_sandbox_user_procs,
        kill_switched_accounts,
    )

    await _kill_sandbox_user_procs(env, "agent", record=False)
    await kill_switched_accounts(env)


async def _turn_end(env) -> None:
    """quiesce_agent's order: the accounts, then the agent user."""
    from benchflow.sandbox.lockdown import (
        _kill_sandbox_user_procs,
        kill_switched_accounts,
    )

    await kill_switched_accounts(env)
    await _kill_sandbox_user_procs(env, "agent", record=False)


ORDERS = {"before-grading": _before_grading, "turn-end": _turn_end}


@pytest.mark.asyncio
@pytest.mark.parametrize("order", sorted(ORDERS))
async def test_without_the_barrier_one_account_restarts_the_other(boxes, order):
    """The hazard: why an identity-switch task is refused, before start, on
    every backend without the container's no_new_privs."""
    box = boxes(no_new_privs=False)
    _run_as_agent(box, AGENT_COMMAND, route="bare")
    started = _population(box)
    assert started["agent"] and started["worker"], started
    # A survivor found by the second check is the same hazard.
    with contextlib.suppress(RuntimeError):
        await ORDERS[order](_env(box))
    left = _settle(box)
    assert left["agent"] or left["worker"], left


@pytest.mark.asyncio
@pytest.mark.parametrize("order", sorted(ORDERS))
async def test_native_generic_exec_keeps_the_barrier_under_the_opt_in(boxes, order):
    """Fails at 6c7d11fa5: the opt-in exempted the command, the switch
    worked, and a writer survived the clean-up."""
    box = boxes(no_new_privs=False)
    _run_as_agent(box, AGENT_COMMAND, route="native")
    assert _population(box)["worker"] == []
    await ORDERS[order](_env(box))
    assert _settle(box) == {"agent": [], "worker": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("order", sorted(ORDERS))
async def test_docker_creation_barrier_leaves_no_writer(boxes, order):
    """Where the task still runs: Docker's generic exec keeps the command
    unchanged, and the container's no_new_privs disarms the switch."""
    from benchflow.sandbox.agent_exec import agent_exec_call

    box = boxes(no_new_privs=True)
    command, user = agent_exec_call(
        AGENT_COMMAND,
        "agent",
        identity_switch=("worker",),
        sandbox=SimpleNamespace(agent_container_no_new_privileges=True),
    )
    assert (command, user) == (AGENT_COMMAND, "agent")
    _run_as_agent(box, command, route="bare")
    assert _population(box)["worker"] == []
    await ORDERS[order](_env(box))
    assert _settle(box) == {"agent": [], "worker": []}


@pytest.mark.asyncio
async def test_kept_service_survives_and_nothing_else_does(boxes):
    """K2 preservation control on the Docker family: the turn-end kill keeps
    the agent's listener frozen, the accounts and the agent's other
    processes are gone, the kill before grading keeps it again, and the
    release lets it answer."""
    from benchflow.sandbox.lockdown import (
        _kill_sandbox_user_procs_keeping,
        kill_switched_accounts,
        release_kept_services,
    )

    box = boxes(no_new_privs=True, freezer=True)
    _run_as_agent(box, LISTENER + " " + AGENT_COMMAND, route="bare")
    env = _env(box)
    await kill_switched_accounts(env)
    first = await _kill_sandbox_user_procs_keeping(env, "agent", (PORT,), wait=True)
    assert len(first.kept) == 1
    await kill_switched_accounts(env)
    again = await _kill_sandbox_user_procs_keeping(env, "agent", (PORT,), wait=False)
    assert again.kept == first.kept
    await kill_switched_accounts(env)
    left = _settle(box)
    assert left["worker"] == []
    assert len(left["agent"]) == 1 and "http.server" in left["agent"][0], left
    pids = tuple(k.pid for k in again.kept)
    await release_kept_services(env, "agent", pids)
    probe = (
        "import urllib.request;"
        f"print(urllib.request.urlopen('http://127.0.0.1:{PORT}/', timeout=5).status)"
    )
    for _ in range(20):
        answered = _dx(box, ["python3", "-c", probe])
        if answered.returncode == 0:
            break
        await asyncio.sleep(0.25)
    assert answered.stdout.strip() == "200", (answered.stdout, answered.stderr)


@pytest.mark.asyncio
@pytest.mark.parametrize("order", sorted(ORDERS))
async def test_bfx_kill_leaves_no_process_of_either_account(boxes, order):
    """The kill before grading on bfx: no process of either account remains,
    and each kill was bfx's."""
    box = boxes(no_new_privs=True, bfx=True)
    _run_as_agent(box, AGENT_COMMAND, route="bare")
    env = await _bfx_env(box)
    await ORDERS[order](env)
    assert _settle(box) == {"agent": [], "worker": []}
    assert env.bfx_commands == ["kill", "kill"]


@pytest.mark.asyncio
async def test_bfx_kill_receipt_counts_what_it_killed(boxes):
    from benchflow.sandbox.lockdown import _kill_sandbox_user_procs

    box = boxes(no_new_privs=True, bfx=True)
    _run_as_agent(box, LISTENER + " " + AGENT_COMMAND, route="bare")
    env = await _bfx_env(box)
    assert _population(box)["agent"]
    killed = await _kill_sandbox_user_procs(env, "agent", record=False)
    # The listener and the respawning script at least.
    assert killed >= 2
    assert _population(box)["agent"] == []
    assert env.bfx_commands == ["kill"]


def test_the_respawn_input_switches_where_nothing_holds_it(boxes):
    """Control for the controls: the image's switch works without any
    barrier, so 'nothing survived' above is the barrier's doing."""
    box = boxes(no_new_privs=False)
    done = _dx(
        box,
        ["/bin/sh", "-c", "su -s /bin/sh worker -c 'id -un' </dev/null"],
        user="agent",
    )
    assert done.stdout.strip() == "worker", (done.stdout, done.stderr)
    held = boxes(no_new_privs=True)
    done = _dx(
        held,
        ["/bin/sh", "-c", "su -s /bin/sh worker -c 'id -un' </dev/null"],
        user="agent",
    )
    assert done.stdout.strip() != "worker"
