"""Images that hand the agent the firewall are disarmed or refused.

Guards PY8 S3: wherever BenchFlow holds the agent user with iptables rules in
the sandbox (``denylist``, ``allowlist``, a no-network task kept open for an
LLM agent, an open network with the cloud metadata block), ``main`` has
NET_ADMIN, and nothing checked file capabilities or setuid firewall tools,
or took NET_ADMIN out of the agent's bounding set: an image with
``setcap cap_net_admin+ep`` on iptables, or a setuid-root firewall helper,
let the agent remove the rules. Now the agent's commands start without
NET_ADMIN where setpriv can drop it (disarmed), and the trial stops before
the agent where it cannot (refused); ``firewall-power.json`` says which.

The rollout-level tests answer the check's commands with what the two
fixture images printed in a real container on a build VM (results/S3 in
the round's notes); the shell-level tests run the scripts in a real POSIX
shell against stub tools in ``tmp_path``; the deterministic tier runs both
fixtures in real sandboxes (tests/test_deterministic_firewall_power.py).
Module-level code here imports nothing new, so on a tree without the
check every test fails on what it checks.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests._firewall_tools import audit_image_tools  # noqa: F401
from tests._receipt_fakes import receipt_aware

pytestmark = pytest.mark.usefixtures("audit_image_tools")

RECORD = "firewall-power.json"
DROP = "--bounding-set=-net_admin,-net_raw --inh-caps=-all --ambient-caps=-all"
# The sets the two fixtures showed for the agent user: the container's
# bounding set with NET_ADMIN (bit 12), and the same after the drop.
BND_NET_ADMIN = "00000000a80435fb"
BND_DROPPED = "00000000a80425fb"
ZERO = "0" * 16

STANDARD_SETUID = "".join(
    f"BF fwpower-found setuid /usr/bin/{name}\n"
    for name in ("chfn", "chsh", "gpasswd", "mount", "newgrp", "passwd", "su", "umount")
)
CAP_LINE = "BF fwpower-found cap /usr/sbin/xtables-nft-multi cap_net_admin=ep\n"
HELPER_LINE = "BF fwpower-found setuid /usr/local/sbin/fwhelper\n"


def _agent_line(bnd: str, power: str = "", nnp: str = "0", prm: str = ZERO) -> str:
    return (
        f"BF fwpower-agent bnd={bnd} inh={ZERO} amb={ZERO} prm={prm} eff={prm} "
        f"nnp={nnp} power={power}\n"
    )


def _root_out(setpriv: str, scan: str, caps: str, scope: str) -> str:
    """Root's step; ``caps``: what read the file capabilities (getcap,
    python3, unread or none)."""
    return (
        f"BF fwpower-root net_admin=1 setpriv={setpriv}\n{scan}"
        f"BF fwpower-scan scope={scope} caps={caps}\n"
    )


# What each fixture printed (Ubuntu 24.04, NET_ADMIN in the container):
# "cap" has cap_net_admin+ep on xtables-nft-multi, "setuid" a setuid-root
# helper /usr/local/sbin/fwhelper that runs iptables as root; with setpriv
# kept (root scans the PATH, the agent's check runs with the drop) and
# removed from the image (root scans the whole file system, no drop).
FIXTURES = {
    "cap": {
        "root": _root_out("ok", STANDARD_SETUID + CAP_LINE, "getcap", "path"),
        "root_nosetpriv": _root_out(
            "none", STANDARD_SETUID + CAP_LINE, "getcap", "all"
        ),
        "drop": "BF fwpower-x /usr/sbin/xtables-nft-multi\n"
        + _agent_line(BND_DROPPED, nnp="1"),
        "nodrop": "BF fwpower-x /usr/sbin/xtables-nft-multi\n"
        + _agent_line(BND_NET_ADMIN, "iptables,iptables-nft,ip6tables,ip6tables-nft"),
        "file": "/usr/sbin/xtables-nft-multi",
    },
    "setuid": {
        "root": _root_out("ok", STANDARD_SETUID + HELPER_LINE, "none", "path"),
        "root_nosetpriv": _root_out(
            "none", HELPER_LINE + STANDARD_SETUID, "none", "all"
        ),
        "drop": "BF fwpower-x /usr/local/sbin/fwhelper\n"
        + _agent_line(BND_DROPPED, nnp="1"),
        "nodrop": "BF fwpower-x /usr/local/sbin/fwhelper\n"
        + _agent_line(BND_NET_ADMIN),
        "file": "/usr/local/sbin/fwhelper",
    },
}

# A clean Alpine image with bash (node:22-alpine plus bash): BusyBox's
# setpriv (no --reuid or --bounding-set, but --nnp), no getcap, no python3,
# only BusyBox's own setuid helper. no_new_privs set by root before su is the
# drop: NET_ADMIN stays in the bounding set, the agent's other sets are empty.
FIXTURES["busybox"] = {
    "root": "BF fwpower-root net_admin=1 setpriv=no-reuid nnp=1\n"
    "BF fwpower-found setuid /bin/bbsuid\n"
    "BF fwpower-scan scope=path caps=none\n",
    "root_nosetpriv": "BF fwpower-root net_admin=1 setpriv=none nnp=0\n"
    "BF fwpower-found setuid /bin/bbsuid\n"
    "BF fwpower-scan scope=all caps=none\n",
    "drop": _agent_line(BND_NET_ADMIN, nnp="1"),
    "nodrop": _agent_line(BND_NET_ADMIN),
}

METADATA_ROOT_OK = (
    "BF metadata status=0 fw4=iptables fw6=ip6tables v6=local install=none "
    "install_t=- hits=0\n"
)


class FixtureEnv:
    """A sandbox that answers the check's commands as a fixture image did.

    Every other command (the metadata block's, a clean-up) gets what a sound
    image answers. ``setpriv`` False: the image has none, so a launch line
    that asks for the drop is refused with exit 126, as the real one is.
    """

    supports_system_shell = True

    def __init__(self, fixture: str, *, setpriv: bool = True, mode: str = "denylist"):
        self.fixture = FIXTURES[fixture]
        self.setpriv = setpriv
        self.calls: list[tuple[str, str | None]] = []
        self.task_env_config = _sandbox_config(mode)

    @receipt_aware
    async def exec(self, command: str, user: str | None = None, **_: object):
        self.calls.append((command, user))
        out, code = "", 0
        if "BF metadata-construction" in command and "run() {" in command:
            # These legacy inventory fixtures do not model the live proof.
            code = 7
        elif "BF fwpower-root" in command:
            out = self.fixture["root" if self.setpriv else "root_nosetpriv"]
        elif "BF fwpower-agent" in command and DROP in command:
            if self.setpriv and self.fixture is not FIXTURES["busybox"]:
                out = self.fixture["drop"]
            else:
                code = 126
        elif "BF fwpower-agent" in command:
            out = self.fixture["nodrop"]
        elif 'echo "BF killed $n end"' in command:
            # K1's kill before a reconnect: nothing of the agent's was left.
            out = "BF killed 0 end\n"
        elif "BF metadata" in command:
            from tests.test_cloud_metadata_block import AGENT_REFUSED, ROOT_OK, FakeEnv

            # Each block (every connection) gets the answers of its own.
            if not hasattr(self, "metadata") or "BF metadata status=" in command:
                self.metadata = FakeEnv(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 2\n")
            return await self.metadata.exec(command, user=user)
        return SimpleNamespace(stdout=out, stderr="", return_code=code)

    def commands(self, marker: str) -> list[tuple[str, str | None]]:
        return [(c, u) for c, u in self.calls if marker in c]


def _sandbox_config(mode: str):
    from benchflow.task.config import NetworkMode, SandboxConfig

    if mode == "denylist":
        return SandboxConfig(
            network_mode=NetworkMode.DENYLIST, blocked_hosts=["example.org"]
        )
    if mode == "allowlist":
        return SandboxConfig(
            network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["example.com"]
        )
    if mode == "no-network":
        config = SandboxConfig(network_mode=NetworkMode.NO_NETWORK)
        config.allow_internet = True  # kept open for the LLM agent
        return config
    assert mode == "open"
    return SandboxConfig(network_mode=NetworkMode.PUBLIC, allow_internet=True)


# The agent env each mode's rollout has when the agent connects.
MODE_AGENT_ENV = {
    "denylist": {"BENCHFLOW_EGRESS_DENYLIST": "1"},
    "allowlist": {
        "BENCHFLOW_EGRESS_DENYLIST": "1",
        "BENCHFLOW_EGRESS_ALLOWLIST": "1",
    },
    "no-network": {"BENCHFLOW_DISALLOW_WEB_TOOLS": "1"},
    "open": {},
}
MODES = tuple(MODE_AGENT_ENV)


def _rollout(tmp_path: Path, env: FixtureEnv, mode: str, agent="claude-agent-acp"):
    from benchflow.rollout import Role, Rollout, RolloutConfig, Scene

    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        scenes=[Scene(roles=[Role(name="agent", agent=agent, model="m")])],
    )
    trial = Rollout(cfg)
    trial._env = env
    trial._rollout_dir = tmp_path
    trial._phase = "idle"
    trial._agent_env = dict(MODE_AGENT_ENV[mode])
    # Under open the cloud metadata block is armed (PY8 S2); the other modes
    # confine the agent user with the uid firewall.
    trial._cloud_metadata = (
        SimpleNamespace(armed=True, uncovered=(), why="", state="on")
        if mode == "open"
        else None
    )
    return trial


async def _connect(trial) -> list[str]:
    """Rollout.connect with the agent's launch replaced by a recorder of the
    user its commands drop NET_ADMIN for (what the launch line reads)."""
    started: list[str] = []

    async def connect(**kwargs):
        dropped = getattr(kwargs["env"], "_benchflow_net_admin_dropped_for", None)
        started.append(f"agent drop={dropped}")
        return (MagicMock(), MagicMock(), MagicMock(), "agent")

    with (
        patch.object(trial, "_connect_live_agent", side_effect=connect),
        patch.object(
            trial._planes,
            "ensure_litellm_runtime",
            AsyncMock(side_effect=lambda **kw: (kw["agent_env"], None)),
        ),
        patch(
            "benchflow.rollout.enforce_codex_apps_policy",
            AsyncMock(side_effect=lambda env, **kw: kw["agent_env"]),
        ),
    ):
        await trial.connect()
    return started


# --- the trial: disarmed, or refused before the agent, in each mode -------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", ["cap", "setuid"])
@pytest.mark.parametrize("mode", MODES)
async def test_a_fixture_image_is_disarmed_before_the_agent_starts(
    tmp_path, mode, fixture
):
    env = FixtureEnv(fixture, mode=mode)
    trial = _rollout(tmp_path, env, mode)

    started = await _connect(trial)

    record = json.loads((tmp_path / RECORD).read_text())
    assert record["state"] == "disarmed", record
    assert record["drop"] == "setpriv"
    assert [e["path"] for e in record["found"]] == [FIXTURES[fixture]["file"]]
    assert FIXTURES[fixture]["file"] in record["why"]
    # The agent's commands start without NET_ADMIN.
    assert started == ["agent drop=agent"]
    # The check ran through the agent's own launch line with the drop, as root.
    (check,) = env.commands("BF fwpower-agent")
    assert DROP in check[0] and check[1] == "root"
    assert "firewall_power" in trial._timing


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", ["cap", "setuid"])
@pytest.mark.parametrize("mode", MODES)
async def test_without_setpriv_a_fixture_image_is_refused_before_the_agent(
    tmp_path, mode, fixture
):
    env = FixtureEnv(fixture, setpriv=False, mode=mode)
    trial = _rollout(tmp_path, env, mode)

    with pytest.raises(RuntimeError, match="sandbox startup refused") as caught:
        await _connect(trial)

    message = str(caught.value)
    assert "restricted agent launch could not be verified" in message
    assert "the image has no setpriv" in message
    record = json.loads((tmp_path / RECORD).read_text())
    assert record["state"] == "refused", record
    assert record["drop"] is None
    assert record["found"] == (
        [
            {
                "kind": "file capability",
                "path": "/usr/sbin/xtables-nft-multi",
                "capabilities": "cap_net_admin=ep",
            }
        ]
        if fixture == "cap"
        else [{"kind": "setuid root", "path": "/usr/local/sbin/fwhelper"}]
    )  # root inventory remains evidence when the mandatory launch is refused
    # The cloud metadata block never ran: nothing past the refusal did.
    assert not env.commands("BF metadata status=")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_a_clean_busybox_image_needs_the_full_launch_barrier(tmp_path, mode):
    """Guards S2-CONF-9 / 70400f55 in all four of S3's network modes."""
    env = FixtureEnv("busybox", mode=mode)
    trial = _rollout(tmp_path, env, mode)

    with pytest.raises(RuntimeError, match="sandbox startup refused"):
        await _connect(trial)

    record = json.loads((tmp_path / RECORD).read_text())
    assert record["state"] == "refused", record
    assert record["drop"] is None
    assert record["found"] == []
    assert record["file_capabilities_checked"] is False
    assert "could not be started" in record["no_drop"]
    (check,) = env.commands("BF fwpower-agent")
    assert DROP in check[0] and check[1] == "root"


@pytest.mark.asyncio
async def test_the_refusal_is_a_sandbox_setup_failure_a_retry_would_repeat(tmp_path):
    from benchflow._utils.scoring import (
        SANDBOX_SETUP,
        classify_error,
        is_unrecoverable_startup_error,
    )

    trial = _rollout(tmp_path, FixtureEnv("cap", setpriv=False), "denylist")
    with pytest.raises(RuntimeError) as caught:
        await _connect(trial)

    assert classify_error(str(caught.value)) == SANDBOX_SETUP
    # Batch22 classifies a missing mandatory launcher as an image repair,
    # so a retry cannot start the same task safely.
    assert is_unrecoverable_startup_error(str(caught.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_the_check_runs_before_the_cloud_metadata_block_and_the_agent(
    tmp_path, mode
):
    env = FixtureEnv("cap", mode=mode)
    trial = _rollout(tmp_path, env, mode)

    await _connect(trial)

    order = [
        "check" if "BF fwpower" in c else "metadata"
        for c, _ in env.calls
        if "BF fwpower" in c
        or ("BF metadata" in c and "BF metadata-construction" not in c)
    ]
    assert order[0] == "check"
    if mode == "open":
        assert "metadata" in order
        # The metadata block's own agent check starts the way the agent does.
        (agent_check,) = env.commands("BF metadata-agent")
        assert DROP in agent_check[0] and agent_check[1] == "root"


@pytest.mark.asyncio
async def test_the_check_runs_before_the_codex_apps_probe(tmp_path):
    """The probe is the first command BenchFlow runs as the agent user."""
    env = FixtureEnv("cap", mode="denylist")
    trial = _rollout(tmp_path, env, "denylist")
    seen: list[object] = []

    async def codex_policy(env_, **kwargs):
        seen.append(getattr(env_, "_benchflow_net_admin_dropped_for", None))
        return kwargs["agent_env"]

    with (
        patch.object(
            trial,
            "_connect_live_agent",
            AsyncMock(return_value=(MagicMock(), MagicMock(), MagicMock(), "a")),
        ),
        patch.object(
            trial._planes,
            "ensure_litellm_runtime",
            AsyncMock(side_effect=lambda **kw: (kw["agent_env"], None)),
        ),
        patch("benchflow.rollout.enforce_codex_apps_policy", side_effect=codex_policy),
    ):
        await trial.connect()

    assert seen == ["agent"]


@pytest.mark.asyncio
async def test_no_rule_of_benchflows_means_no_check(tmp_path):
    """An open network without the block (an opt-out): main has no
    NET_ADMIN, so nothing is checked and the agent starts as before."""
    from benchflow.task.config import NetworkMode, SandboxConfig

    env = FixtureEnv("cap", mode="open")
    env.task_env_config = SandboxConfig(network_mode=NetworkMode.PUBLIC)
    trial = _rollout(tmp_path, env, "open")
    trial._cloud_metadata = SimpleNamespace(
        armed=False,
        uncovered=(),
        why="the run opts out (k = true)",
        state="off",
        opt_out="k = true",
    )

    started = await _connect(trial)

    assert not env.commands("BF fwpower")
    assert not (tmp_path / RECORD).exists()
    assert started == ["agent drop=None"]


@pytest.mark.asyncio
async def test_a_root_agent_is_not_checked(tmp_path):
    """A guard: root holds every power anyway (passes before the fix too)."""
    env = FixtureEnv("cap", mode="allowlist")
    trial = _rollout(tmp_path, env, "allowlist")
    trial._config.sandbox_user = None

    started = await _connect(trial)

    assert not env.commands("BF fwpower")
    assert not (tmp_path / RECORD).exists()
    assert started == ["agent drop=None"]


# --- the launch lines read the drop ----------------------------------------------


class LaunchEnv:
    """An env whose agent user's commands start without NET_ADMIN."""

    _benchflow_net_admin_dropped_for = "agent"

    @receipt_aware
    async def exec(self, command: str, **_: object):
        return SimpleNamespace(stdout="", stderr="", return_code=1)

    async def live_process(self, agent: str | None = None, **_kwargs):
        return MagicMock()


class _Started(Exception):
    pass


@pytest.mark.asyncio
async def test_the_acp_agent_launch_drops_net_admin(tmp_path):
    from benchflow.acp import runtime

    seen: list[str] = []

    def transport(*, command, **_):
        seen.append(command)
        raise _Started

    with (
        patch.object(runtime, "ContainerTransport", side_effect=transport),
        pytest.raises(_Started),
    ):
        await runtime.connect_acp(
            env=LaunchEnv(),
            agent="claude-agent-acp",
            agent_launch="claude-agent-acp --stdio",
            agent_env={},
            sandbox_user="agent",
            model=None,
            rollout_dir=tmp_path,
            environment="docker",
            agent_cwd="/app",
        )

    (command,) = seen
    assert DROP in command
    # su only ever starts under no_new_privs set by root (BusyBox's setpriv).
    assert command.count(" su -m") == command.count(
        "setpriv --no-new-privs " + DROP + " -- su -m"
    )


@pytest.mark.asyncio
async def test_the_native_harness_launch_drops_net_admin(tmp_path):
    from benchflow.native_harness import client as native_client
    from benchflow.native_harness.harnesses import CLAUDE_CODE

    harness = CLAUDE_CODE
    client = native_client.NativeCLIClient(
        env=LaunchEnv(),
        harness=harness,
        agent="claude",
        launch_env={},
        sandbox_user="agent",
        cwd="/app",
        rollout_dir=tmp_path,
    )
    seen: list[str] = []

    async def live_process(agent=None, **_kwargs):
        raise _Started

    def launch_script(executable, argv, run_id):
        return "native-cli"

    original = native_client.build_priv_drop_cmd

    def record(command, user, **kwargs):
        built = original(command, user, **kwargs)
        seen.append(built)
        return built

    client._env.live_process = live_process
    with (
        patch.object(native_client, "launch_script", side_effect=launch_script),
        patch.object(native_client, "build_priv_drop_cmd", side_effect=record),
        pytest.raises(_Started),
    ):
        await client.prompt("hello")

    assert seen and DROP in seen[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_drop", [False, True])
async def test_the_metadata_blocks_agent_check_starts_without_net_admin(saved_drop):
    """Guards the S2 e81b486c merge: every probe keeps the shared barrier."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    env = FixtureEnv("cap", mode="open")
    if saved_drop:
        env._benchflow_net_admin_dropped_for = "agent"

    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "on", record
    (check,) = env.commands("BF metadata-agent")
    assert DROP in check[0] and check[1] == "root"


# --- the launch line in a real shell -----------------------------------------------

STUB_SETPRIV = """#!/bin/sh
if [ "$1" = --help ]; then printf '%s\\n' "$SETPRIV_HELP"; exit 0; fi
case $SETPRIV_HELP in *no-new-privs*) ;; *) exit 1;; esac
printf '%s\\n' "$*" >> "$SETPRIV_LOG"
while [ "$#" -gt 0 ] && [ "$1" != -- ]; do shift; done
shift
exec "$@"
"""
UTIL_LINUX_HELP = (
    " --no-new-privs              disallow granting new privileges\n"
    " --inh-caps <caps,...>       set inheritable capabilities\n"
    " --bounding-set <caps>       set capability bounding set\n"
    " --reuid <uid|user>          set real and effective uid\n"
    " --init-groups               initialize supplementary groups"
)
# util-linux's setpriv before --no-new-privs (it came with --bounding-set).
OLD_UTIL_LINUX_HELP = "\n".join(UTIL_LINUX_HELP.splitlines()[1:])
# BusyBox v1.36.1's (alpine:3.20, node:22-alpine): no --reuid, no
# --bounding-set, but --nnp.
BUSYBOX_HELP = (
    "Usage: setpriv [OPTIONS] PROG ARGS\n\n"
    "Run PROG with different privilege settings\n\n"
    "-d,--dump\t\tShow current capabilities\n"
    "--nnp,--no-new-privs\tIgnore setuid/setgid bits and file capabilities\n"
    "--inh-caps CAP,CAP\tSet inheritable capabilities\n"
    "--ambient-caps CAP,CAP\tSet ambient capabilities"
)


def _shell_launch(tmp_path: Path, help_text: str | None, *, drop: bool):
    from benchflow.sandbox.lockdown import build_priv_drop_cmd

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("bash", "sh", "grep", "cat", "printf", "touch"):
        found = shutil.which(name)
        if found and not (bin_dir / name).exists():
            (bin_dir / name).symlink_to(found)
    if help_text is not None:
        (bin_dir / "setpriv").write_text(STUB_SETPRIV)
        (bin_dir / "setpriv").chmod(0o755)
    (bin_dir / "su").write_text(
        '#!/bin/sh\necho "su $*" >> "$SETPRIV_LOG"\n'
        '[ "$1 $2 $4 $6" = "-m -s -c agent" ] && exec "$3" -c "$5"\n'
    )
    (bin_dir / "su").chmod(0o755)
    ran = tmp_path / "ran"
    launch = f"touch {shlex.quote(str(ran))}"
    # Without the drop, the call is the one every launch made before (a guard
    # that passes on a tree without the drop too).
    from unittest.mock import patch

    with patch("benchflow.sandbox._root_launch.AGENT_LAUNCH_PATH", str(bin_dir)):
        cmd = (
            build_priv_drop_cmd(launch, "agent", drop_net_admin=True)
            if drop
            else build_priv_drop_cmd(launch, "agent")
        )
    log = tmp_path / "setpriv.log"
    result = subprocess.run(
        ["/bin/sh", "-c", cmd],
        capture_output=True,
        text=True,
        env={
            "PATH": str(bin_dir),
            "SETPRIV_HELP": help_text or "",
            "SETPRIV_LOG": str(log),
        },
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls, ran.exists()


def test_the_launch_line_drops_net_admin_from_both_sets(tmp_path):
    result, calls, ran = _shell_launch(tmp_path, UTIL_LINUX_HELP, drop=True)

    assert result.returncode == 0, result.stderr
    assert ran
    (call,) = calls
    assert "--reuid=agent --regid=agent --init-groups" in call
    assert DROP in call
    # S3-R1-A-1: no setuid program or file capability makes the agent uid 0.
    assert "--no-new-privs" in call


@pytest.mark.parametrize("help_text", [None, OLD_UTIL_LINUX_HELP])
def test_a_launch_that_must_drop_never_starts_without_it(tmp_path, help_text):
    result, calls, ran = _shell_launch(tmp_path, help_text, drop=True)

    assert result.returncode != 0
    assert not ran
    assert not any(c.startswith("su ") for c in calls)


def test_busybox_without_bounding_set_support_cannot_start_the_agent(tmp_path):
    """Guards S2-CONF-9 / 70400f55 in S3's explicitly checked launch."""
    result, calls, ran = _shell_launch(tmp_path, BUSYBOX_HELP, drop=True)

    assert result.returncode != 0
    assert "requires util-linux setpriv" in result.stderr
    assert not ran
    assert not calls


def test_without_a_checked_drop_the_launch_still_has_the_barrier(tmp_path):
    """Guards S2-CONF-9 / 70400f55 without S3's in-memory check state."""
    result, calls, ran = _shell_launch(tmp_path, UTIL_LINUX_HELP, drop=False)

    assert result.returncode == 0 and ran
    (call,) = calls
    assert DROP in call and "--no-new-privs" in call
    assert DROP in call and "--ambient-caps=-all" in call


# --- the check's scripts in a real shell -------------------------------------------

SHELLS = [s for s in ("/bin/sh", "/bin/dash", "/bin/bash") if Path(s).exists()]

STUB_FIREWALL = """#!/bin/sh
# Makes the probe's chain only when FW_POWER names this tool.
name=$(basename "$0")
echo "$name $*" >> "$FW_LOG"
case " $FW_POWER " in *" $name "*) exit 0;; esac
exit 1
"""
STUB_GETCAP = """#!/bin/sh
# Prints the stub capabilities of the files asked about (GETCAP_FILES:
# "path=caps" pairs), as libcap 2.x does.
for f in "$@"; do
  for pair in $GETCAP_FILES; do
    [ "${pair%%=*}" = "$f" ] && echo "$f ${pair#*=}"
  done
done
exit 0
"""


class Stub:
    """A stub sandbox: /proc/self/status, firewall tools, getcap, files."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.proc = tmp_path / "proc"
        (self.proc / "self").mkdir(parents=True)
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.fs = tmp_path / "fs"
        (self.fs / "usr" / "bin").mkdir(parents=True)
        self.sys_bin = tmp_path / "sysbin"
        self.sys_bin.mkdir()
        hidden = {"getcap", "setpriv", "nft"}
        for d in ("/usr/bin", "/bin"):
            for src in Path(d).iterdir():
                name = src.name
                if name in hidden or name.startswith(
                    ("iptables", "ip6tables", "python")
                ):
                    continue
                if not (self.sys_bin / name).exists():
                    (self.sys_bin / name).symlink_to(src)
        self.status(BND_NET_ADMIN)
        self.env = {
            "FW_LOG": str(tmp_path / "fw.log"),
            "FW_POWER": "",
            "GETCAP_FILES": "",
        }

    def status(
        self, bnd: str, prm: str = ZERO, amb: str = ZERO, nnp: str = "0"
    ) -> Stub:
        (self.proc / "self" / "status").write_text(
            f"Name:\tsh\nCapInh:\t{ZERO}\nCapPrm:\t{prm}\nCapEff:\t{prm}\n"
            f"CapBnd:\t{bnd}\nCapAmb:\t{amb}\nNoNewPrivs:\t{nnp}\n"
        )
        return self

    def tool(self, name: str, body: str, folder: Path | None = None) -> Path:
        path = (folder or self.bin) / name
        path.write_text(body)
        path.chmod(0o755)
        return path

    def run(self, cmd: str, shell: str = "/bin/sh", path_first: str = "") -> str:
        path = f"{path_first + ':' if path_first else ''}{self.bin}:{self.sys_bin}"
        # Script-unit fixtures supply the absolute bindings the production
        # root route now validates. The route itself has separate tests.
        tools = ("awk", "find", "head", "sort", "getcap", "python3", "setpriv")
        bindings = "".join(f"bf_{name}=$(command -v {name} || :); " for name in tools)
        result = subprocess.run(
            [shell, "-c", bindings + cmd],
            capture_output=True,
            text=True,
            env={"PATH": path, **self.env},
            check=True,
        )
        return result.stdout

    def scan(self, *, whole_fs: bool, shell: str = "/bin/sh", path_first: str = ""):
        import os

        from benchflow.sandbox.firewall_power import _scan_cmd

        cmd = _scan_cmd(
            whole_fs=whole_fs,
            fs_root=str(self.fs),
            setuid_owner=str(os.getuid()),
            python_command=str(self.bin / "python3"),
        )
        return self.run(cmd, shell, path_first)

    def check(self, candidates=(), shell: str = "/bin/sh") -> str:
        from benchflow.sandbox.firewall_power import _agent_check_cmd

        return self.run(
            _agent_check_cmd("BFFWtest", candidates, proc=str(self.proc)), shell
        )


@pytest.fixture
def stub(tmp_path) -> Stub:
    return Stub(tmp_path)


def _last(out: str, prefix: str = "BF fwpower-agent ") -> dict[str, str]:
    (line,) = [ln for ln in out.splitlines() if ln.startswith(prefix)]
    return dict(f.split("=", 1) for f in line.split()[2:])


@pytest.mark.parametrize("shell", SHELLS)
def test_the_root_step_reads_net_admin_and_what_setpriv_can_do(stub, shell):
    import os

    from benchflow.sandbox.firewall_power import _root_cmd

    def root(help_text: str | None, **kwargs) -> tuple[str, str]:
        (stub.bin / "setpriv").unlink(missing_ok=True)
        if help_text is not None:
            stub.tool("setpriv", f"#!/bin/sh\nprintf '%s\\n' '{help_text}'\n")
        cmd = _root_cmd(
            proc=str(stub.proc),
            fs_root=str(stub.fs),
            setuid_owner=str(os.getuid()),
            **kwargs,
        )
        out = stub.run(cmd, shell)
        (line,) = [ln for ln in out.splitlines() if ln.startswith("BF fwpower-root ")]
        scans = [ln for ln in out.splitlines() if ln.startswith("BF fwpower-scan ")]
        return line, scans[0].split()[2] if scans else ""

    # Where the drop applies root lists the PATH; where it cannot, everything.
    assert root(UTIL_LINUX_HELP) == (
        "BF fwpower-root net_admin=1 setpriv=ok nnp=1",
        "scope=path",
    )
    assert root(UTIL_LINUX_HELP, scan_all=True)[1] == "scope=all"
    assert root(OLD_UTIL_LINUX_HELP) == (
        "BF fwpower-root net_admin=1 setpriv=no-nnp nnp=0",
        "scope=all",
    )
    # BusyBox's sets no_new_privs, a drop of its own (S3-R2-A-1).
    assert root(BUSYBOX_HELP) == (
        "BF fwpower-root net_admin=1 setpriv=no-reuid nnp=1",
        "scope=path",
    )
    assert root(None) == (
        "BF fwpower-root net_admin=1 setpriv=none nnp=0",
        "scope=all",
    )
    # Without NET_ADMIN in the sandbox nothing is scanned.
    stub.status(BND_DROPPED)
    assert root(UTIL_LINUX_HELP) == (
        "BF fwpower-root net_admin=0 setpriv=ok nnp=1",
        "",
    )


@pytest.mark.parametrize("shell", SHELLS)
def test_the_probe_makes_and_deletes_only_an_empty_chain_of_its_own(stub, shell):
    for name in ("iptables", "ip6tables", "iptables-legacy"):
        stub.tool(name, STUB_FIREWALL)
    stub.env["FW_POWER"] = "iptables-legacy"

    fields = _last(stub.check(shell=shell))

    assert fields["power"] == "iptables-legacy"
    assert fields["bnd"] == BND_NET_ADMIN
    calls = (stub.root / "fw.log").read_text().splitlines()
    assert calls, calls
    for call in calls:
        _tool, *args = call.split()
        assert args[0] == "-w" and args[1] in ("-N", "-X") and args[2] == "BFFWtest", (
            call
        )
    assert "iptables-legacy -w -X BFFWtest" in calls


@pytest.mark.parametrize("shell", SHELLS)
def test_the_agent_check_reads_no_new_privs(stub, shell):
    assert _last(stub.check(shell=shell))["nnp"] == "0"
    stub.status(BND_DROPPED, nnp="1")
    assert _last(stub.check(shell=shell))["nnp"] == "1"


@pytest.mark.parametrize("shell", SHELLS)
def test_the_agent_user_says_which_of_roots_files_it_can_run(stub, shell):
    runs = stub.tool("fwhelper", "#!/bin/sh\n", stub.fs / "usr" / "bin")
    locked = stub.tool("locked", "#!/bin/sh\n", stub.fs / "usr" / "bin")
    locked.chmod(0o644)
    odd = stub.fs / "usr" / "bin" / "it's here"
    odd.write_text("#!/bin/sh\n")
    odd.chmod(0o755)

    out = stub.check([str(runs), str(locked), str(odd), "/nonexistent/x"], shell)

    assert [ln for ln in out.splitlines() if ln.startswith("BF fwpower-xhex ")] == [
        f"BF fwpower-xhex {str(runs).encode().hex()}",
        f"BF fwpower-xhex {str(odd).encode().hex()}",
    ]


@pytest.mark.parametrize("shell", SHELLS)
def test_the_path_scan_reads_the_last_path_entry_on_every_shell(stub, shell):
    """dash splits only what an expansion gives: "$PATH:/usr/sbin" glued
    PATH's last entry to the system folders in the Rust bench's first scan."""
    tools = stub.root / "opt-tools"
    tools.mkdir()
    capped = stub.tool("netctl", "#!/bin/sh\n", tools)
    stub.tool("getcap", STUB_GETCAP)
    stub.env["GETCAP_FILES"] = f"{capped}=cap_net_admin+ep"

    out = stub.scan(whole_fs=False, shell=shell, path_first=f"{stub.bin}:{tools}")

    assert f"BF fwpower-caphex {str(capped).encode().hex()} cap_net_admin+ep" in out
    assert _last(out, "BF fwpower-scan ") == {"scope": "path", "caps": "getcap"}


@pytest.mark.parametrize("shell", SHELLS)
def test_the_whole_file_system_scan_finds_setuid_files_and_capabilities(stub, shell):
    helper = stub.tool("fwhelper", "#!/bin/sh\n", stub.fs / "usr" / "bin")
    helper.chmod(0o4755)
    plain = stub.tool("plain", "#!/bin/sh\n", stub.fs / "usr" / "bin")
    deep = stub.fs / "opt" / "x" / "bin"
    deep.mkdir(parents=True)
    python = stub.tool("python3", "#!/bin/sh\n", deep)
    stub.tool("getcap", STUB_GETCAP)
    stub.env["GETCAP_FILES"] = f"{python}=cap_setuid=ep {plain}=cap_net_raw+ep"

    out = stub.scan(whole_fs=True, shell=shell)

    assert f"BF fwpower-setuidhex {str(helper).encode().hex()}" in out
    assert f"BF fwpower-caphex {str(python).encode().hex()} cap_setuid=ep" in out
    from benchflow.sandbox.firewall_power import _scanned

    assert [e["path"] for e in _scanned(out)] == [str(helper), str(python)]
    assert _last(out, "BF fwpower-scan ") == {"scope": "all", "caps": "getcap"}


@pytest.mark.parametrize("shell", SHELLS)
def test_without_getcap_the_scan_says_file_capabilities_were_not_read(stub, shell):
    out = stub.scan(whole_fs=True, shell=shell)
    assert _last(out, "BF fwpower-scan ")["caps"] == "none"


STUB_PYTHON3 = """#!/bin/sh
# Stands in for the protected Python: says it was run isolated with the
# reader and its limit, and prints a capability for every file it is given.
[ "$1" = -I ] && [ "$2" = -S ] && [ "$3" = -B ] && [ "$4" = -c ] && [ "$6" = 1000 ] || exit 2
[ -n "$PY_FAIL" ] && exit 1
tr '\\000' '\\n' | while IFS= read -r f; do
  [ -n "$f" ] && echo "BF fwpower-found cap $f cap_setuid=ep"
done
exit 0
"""


@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("whole_fs", [True, False])
def test_without_getcap_root_reads_file_capabilities_with_python3(
    stub, shell, whole_fs
):
    """S3-R1-A-2: Debian, Ubuntu and Alpine bases ship no getcap."""
    tools = stub.fs / "usr" / "bin"
    python = stub.tool("python3.12", "#!/bin/sh\n", tools)
    plain = tools / "data.txt"
    plain.write_text("not executable\n")
    stub.tool("python3", STUB_PYTHON3)

    out = stub.scan(whole_fs=whole_fs, shell=shell, path_first=str(tools))

    assert f"BF fwpower-found cap {python} cap_setuid=ep" in out
    assert str(plain) not in out
    assert _last(out, "BF fwpower-scan ")["caps"] == "python3"
    stub.env["PY_FAIL"] = "1"
    out = stub.scan(whole_fs=whole_fs, shell=shell, path_first=str(tools))
    assert _last(out, "BF fwpower-scan ")["caps"] == "unread"


def _read_caps(files: dict[bytes, bytes | None], limit: int = 1000) -> list[str]:
    """Run root's python3 reader in this process on made-up attributes."""
    import io
    import os

    from benchflow.sandbox.firewall_power import _CAP_READER

    def getxattr(path, name, *, follow_symlinks=True):
        assert name == "security.capability" and follow_symlinks is False
        value = files[path]
        if value is None:
            raise OSError(61, "No data available")
        return value

    out = io.BytesIO()
    stdin = SimpleNamespace(buffer=io.BytesIO(b"\0".join(files) + b"\0"))
    with (
        patch.object(os, "getxattr", getxattr, create=True),
        patch.object(sys, "argv", ["-c", str(limit)]),
        patch.object(sys, "stdin", stdin),
        patch.object(sys, "stdout", SimpleNamespace(buffer=out)),
    ):
        exec(compile(_CAP_READER, "<cap reader>", "exec"), {"__name__": "__main__"})
    return out.getvalue().decode().splitlines()


def _vfs_cap(revision: int, effective: bool, permitted: int, inheritable: int = 0):
    magic = revision | (1 if effective else 0)
    words = [magic, permitted & 0xFFFFFFFF, inheritable & 0xFFFFFFFF]
    if revision != 0x01000000:
        words += [permitted >> 32, inheritable >> 32]
    if revision == 0x03000000:
        words.append(0)  # the namespace's root uid
    return b"".join(w.to_bytes(4, "little") for w in words)


def test_the_python3_reader_prints_capabilities_as_getcap_does():
    from benchflow.sandbox.firewall_power import _scanned

    net_admin, setuid, net_raw = 1 << 12, 1 << 7, 1 << 13
    lines = _read_caps(
        {
            b"/opt/py/python3": _vfs_cap(0x02000000, True, net_admin),
            b"/usr/local/bin/tool": _vfs_cap(0x03000000, True, setuid | net_raw),
            b"/usr/local/bin/inh": _vfs_cap(0x02000000, False, 0, 1 << 40),
            b"/usr/local/bin/odd": b"\x07\x00\x00\x09",
            b"/usr/local/bin/none": None,
            b"/usr/local/bin/empty": _vfs_cap(0x02000000, True, 0),
            b"/usr/bin/old": _vfs_cap(0x01000000, True, 1 << 21),
        }
    )
    assert lines == [
        f"BF fwpower-caphex {path.encode().hex()} {caps}"
        for path, caps in [
            ("/opt/py/python3", "cap_net_admin=ep"),
            ("/usr/local/bin/tool", "cap_setuid,cap_net_raw=ep"),
            ("/usr/local/bin/inh", "cap_checkpoint_restore=i"),
            ("/usr/local/bin/odd", "=ep"),
            ("/usr/bin/old", "cap_sys_admin=ep"),
        ]
    ]
    counted = [e["path"] for e in _scanned("\n".join(lines))]
    assert "/opt/py/python3" in counted and "/usr/local/bin/odd" in counted
    assert (
        len(
            _read_caps(
                {b"/a": _vfs_cap(0x02000000, True, setuid)}
                | {
                    f"/b{i}".encode(): _vfs_cap(0x02000000, True, setuid)
                    for i in range(5)
                },
                limit=3,
            )
        )
        == 4  # three records and a fail-closed truncation marker
    )


# --- what counts ----------------------------------------------------------------


def test_only_files_that_give_net_admin_count():
    from benchflow.sandbox.firewall_power import _runnable, _scanned

    out = (
        STANDARD_SETUID
        + "BF fwpower-found setuid /usr/lib/openssh/ssh-keysign\n"
        + "BF fwpower-found setuid /usr/lib/chromium/chrome-sandbox\n"
        + "BF fwpower-found cap /usr/lib/x86_64-linux-gnu/gstreamer1.0/gstreamer-1.0/"
        "gst-ptp-helper cap_net_bind_service,cap_net_admin=ep\n"
        + "BF fwpower-found setuid /opt/tools/passwd\n"
        + "BF fwpower-found setuid /usr/local/sbin/fwhelper\n"
        + "BF fwpower-found cap /usr/bin/ping cap_net_raw=ep\n"
        + "BF fwpower-found cap /usr/bin/dumpcap cap_net_admin,cap_net_raw=eip\n"
        + "BF fwpower-found cap /usr/local/bin/locked cap_net_admin=ep\n"
        + "BF fwpower-found cap /usr/local/bin/everything =ep\n"
        + "BF fwpower-found cap /usr/local/bin/old = cap_net_admin+ep\n"
        + "BF fwpower-found cap /opt/py/python3 cap_sys_admin,cap_net_bind_service+ep\n"
    )
    scanned = _scanned(out)
    assert [e["path"] for e in scanned] == [
        "/opt/tools/passwd",
        "/usr/local/sbin/fwhelper",
        "/usr/local/bin/locked",
        "/usr/local/bin/everything",
        "/usr/local/bin/old",
        "/opt/py/python3",
    ]
    # What the agent user cannot run does not count.
    agent = "".join(
        f"BF fwpower-x {e['path']}\n"
        for e in scanned
        if e["path"] != "/usr/local/bin/locked"
    )
    assert "/usr/local/bin/locked" not in [e["path"] for e in _runnable(scanned, agent)]
    assert len(_runnable(scanned, agent)) == 5


# --- check_firewall_power's decisions --------------------------------------------


class ScriptedEnv:
    """Answers root's step, the drop check, root's rescan and the plain check."""

    supports_system_shell = True

    def __init__(
        self,
        root: str,
        drop: str | None = None,
        nodrop: str = "",
        rescan: str = "",
        barrier: str = "",
    ) -> None:
        self.root, self.drop, self.nodrop, self.rescan = root, drop, nodrop, rescan
        # What S2's construction proof prints (none by default: unverified).
        self.barrier = barrier
        self.calls: list[str] = []
        self.drop_pending = False

    @receipt_aware
    async def exec(self, command: str, user: str | None = None, **_: object):
        self.calls.append(command)
        if "BF metadata-construction" in command:
            out = self.barrier
        elif "BF fwpower-root" in command:
            out = self.root
            self.drop_pending = "all=1;" not in command and (
                "setpriv=ok" in out or "nnp=1" in out
            )
        elif "BF fwpower-agent" in command and self.drop_pending:
            self.drop_pending = False
            out = self.drop or ""
        elif "BF fwpower-agent" in command:
            out = self.nodrop
        elif "BF fwpower-scan" in command:
            out = self.rescan
        else:
            out = ""
        return SimpleNamespace(stdout=out, stderr="", return_code=0)


ROOT_OK = _root_out("ok", STANDARD_SETUID, "getcap", "path")
ROOT_NO_SETPRIV = _root_out("none", STANDARD_SETUID, "getcap", "all")
HOLDS = ("the denylist's uid firewall",)


async def _check(env, **kwargs):
    from benchflow.sandbox.firewall_power import check_firewall_power

    return await check_firewall_power(env, "agent", holds=HOLDS, **kwargs)


@pytest.mark.asyncio
async def test_a_sandbox_without_net_admin_is_clear_and_runs_nothing_more():
    env = ScriptedEnv("BF fwpower-root net_admin=0 setpriv=ok\n")
    record, refusal = await _check(env)
    assert refusal is None
    assert record["state"] == "clear"
    assert len(env.calls) == 1  # root's read, first


SAFE_BARRIER = (
    "BF metadata-construction uid=1000 nnp=1 bnd=0000000000000000 "
    "amb=0000000000000000\n"
)


@pytest.mark.asyncio
async def test_without_net_admin_the_check_ends_at_roots_read_even_with_a_barrier():
    """Now main starts without NET_ADMIN where the metadata
    helper adds the block's rules. Root's read finds none, so the check is
    clear there: no construction proof and no probe through the launch,
    whose barrier every agent launch takes anyway. Before, a launch that
    passed the proof was probed and recorded as disarmed (three execs)."""
    env = ScriptedEnv(
        "BF fwpower-root net_admin=0 setpriv=ok nnp=1\n",
        drop=_agent_line(ZERO, nnp="1"),
        barrier=SAFE_BARRIER,
    )
    record, refusal = await _check(env)
    assert refusal is None
    assert record["state"] == "clear", record
    assert "not in root's bounding set" in record["why"]
    assert len(env.calls) == 1 and "BF fwpower-root" in env.calls[0]


@pytest.mark.asyncio
async def test_with_net_admin_a_verified_barrier_is_still_proven_and_probed():
    """The other way round: with NET_ADMIN in root's bounding set the
    barrier's proof and the probe through the launch run, as before."""
    env = ScriptedEnv(
        ROOT_OK,
        drop=_agent_line(ZERO, nnp="1"),
        barrier=SAFE_BARRIER,
    )
    record, refusal = await _check(env)
    assert refusal is None
    assert record["state"] == "disarmed", record
    assert record["mandatory_barrier"] == "verified"
    order = [
        next(
            m
            for m in ("fwpower-root", "metadata-construction", "fwpower-agent")
            if m in c
        )
        for c in env.calls
    ]
    assert order == ["fwpower-root", "metadata-construction", "fwpower-agent"]


@pytest.mark.asyncio
async def test_a_drop_that_still_leaves_net_admin_is_refused_without_retry():
    """Guards S3-CONF-1: a failed restricted launch never enables a bare one."""
    from benchflow.sandbox.firewall_power import net_admin_dropped

    env = ScriptedEnv(
        FIXTURES["cap"]["root"],
        drop=_agent_line(BND_NET_ADMIN, nnp="1", prm=BND_NET_ADMIN),
        nodrop=FIXTURES["cap"]["nodrop"],
        rescan=STANDARD_SETUID + CAP_LINE + "BF fwpower-scan scope=all caps=getcap\n",
    )
    record, refusal = await _check(env)
    assert record["state"] == "refused"
    assert "still hold NET_ADMIN" in record["no_drop"]
    assert refusal and "xtables-nft-multi" in refusal
    assert not net_admin_dropped(env, "agent")
    assert len(env.calls) == 4  # root read, construction, restricted probe, rescan
    probes = [command for command in env.calls if "BF fwpower-agent" in command]
    assert len(probes) == 1 and DROP in probes[0]


@pytest.mark.asyncio
async def test_a_drop_whose_check_did_not_start_is_refused_without_retry():
    """Guards S3-CONF-1 / e4108744: missing output cannot downgrade the audit."""
    env = ScriptedEnv(
        ROOT_OK,
        drop="",
        nodrop=_agent_line(BND_NET_ADMIN),
        rescan="BF fwpower-scan scope=all caps=getcap\n",
    )
    record, refusal = await _check(env)
    assert refusal and "util-linux setpriv" in refusal
    assert record["state"] == "refused"
    assert "could not be started with no_new_privs" in record["no_drop"]
    assert len(env.calls) == 4  # root read, construction, restricted probe, rescan
    probes = [command for command in env.calls if "BF fwpower-agent" in command]
    assert len(probes) == 1 and DROP in probes[0]


@pytest.mark.asyncio
async def test_with_no_way_to_the_firewall_the_trial_runs():
    env = ScriptedEnv(ROOT_NO_SETPRIV, nodrop=_agent_line(BND_NET_ADMIN))
    record, refusal = await _check(env, can_drop=False)
    assert refusal is None
    assert record["state"] == "clear"
    assert record["file_capabilities_checked"] is True
    assert record["found"] == []


@pytest.mark.asyncio
async def test_file_capabilities_read_by_python3_count_like_getcaps():
    """An image without getcap: root's python3 read the attributes."""
    env = ScriptedEnv(
        _root_out(
            "none",
            STANDARD_SETUID
            + "BF fwpower-found cap /usr/bin/python3.12 cap_setuid=ep\n",
            "python3",
            "all",
        ),
        nodrop="BF fwpower-x /usr/bin/python3.12\n" + _agent_line(BND_NET_ADMIN),
    )
    record, refusal = await _check(env)
    assert record["state"] == "refused"
    assert record["file_capabilities_checked"] is True
    assert (
        refusal and "/usr/bin/python3.12 (a file capability: cap_setuid=ep)" in refusal
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("caps", "words"),
    [
        ("none", "the image has no getcap (libcap) or python3"),
        ("unread", "python3 could not read the image's file capabilities"),
    ],
)
@pytest.mark.parametrize("can_drop", [True, False])
async def test_without_a_drop_unread_file_capabilities_stop_the_trial(
    caps, words, can_drop
):
    """S3-R1-A-2: cap_net_admin or cap_setuid on python3 is invisible to the
    agent user's probe, so a scan that read no file capabilities cannot say
    clear where no drop holds (BusyBox's setpriv, a session-factory agent)."""
    setpriv = "ok" if not can_drop else "no-bounding-set"
    env = ScriptedEnv(
        _root_out(setpriv, STANDARD_SETUID, caps, "all"),
        nodrop=_agent_line(BND_NET_ADMIN),
    )
    record, refusal = await _check(env, can_drop=can_drop)
    assert record["state"] == "refused", record
    assert record["file_capabilities_checked"] is False
    assert refusal and refusal.startswith("sandbox startup refused")
    if can_drop:
        assert "util-linux setpriv" in refusal
        assert not any("BF fwpower-agent" in c for c in env.calls)
    else:
        assert words in refusal
        assert "libcap2-bin" in refusal
    # A session-factory agent's commands never go through setpriv, so its
    # refusal does not suggest one (S3-R2-A-1).
    assert ("setpriv" in refusal) == can_drop


@pytest.mark.asyncio
async def test_with_the_drop_unread_file_capabilities_are_disarmed_anyway():
    """Under no_new_privs a file capability gives nothing, read or not."""
    env = ScriptedEnv(
        _root_out("ok", STANDARD_SETUID, "none", "path"),
        drop=_agent_line(BND_DROPPED, nnp="1"),
    )
    record, refusal = await _check(env)
    assert refusal is None
    assert record["state"] == "disarmed"
    assert record["file_capabilities_checked"] is False


@pytest.mark.asyncio
async def test_a_drop_without_no_new_privs_does_not_disarm_a_setuid_root_shell():
    """S3-R1-A-1: with NET_ADMIN out of the bounding set but no no_new_privs,
    a setuid-root shell still makes the agent uid 0, whose traffic none of
    the uid rules match: not disarmed, and refused where no drop holds."""
    from benchflow.sandbox.firewall_power import net_admin_dropped

    rootsh = "BF fwpower-found setuid /usr/local/bin/rootsh\n"
    env = ScriptedEnv(
        _root_out("ok", STANDARD_SETUID + rootsh, "getcap", "path"),
        drop="BF fwpower-x /usr/local/bin/rootsh\n" + _agent_line(BND_DROPPED),
        nodrop="BF fwpower-x /usr/local/bin/rootsh\n" + _agent_line(BND_NET_ADMIN),
        rescan=STANDARD_SETUID + rootsh + "BF fwpower-scan scope=all caps=getcap\n",
    )
    record, refusal = await _check(env)
    assert record["state"] == "refused", record
    assert "NoNewPrivs 0" in record["no_drop"]
    assert refusal and "/usr/local/bin/rootsh (setuid root)" in refusal
    assert not net_admin_dropped(env, "agent")


@pytest.mark.asyncio
async def test_the_drop_disarms_a_setuid_root_shell_under_no_new_privs():
    env = ScriptedEnv(
        _root_out(
            "ok",
            STANDARD_SETUID + "BF fwpower-found setuid /usr/local/bin/rootsh\n",
            "getcap",
            "path",
        ),
        drop="BF fwpower-x /usr/local/bin/rootsh\n" + _agent_line(BND_DROPPED, nnp="1"),
    )
    record, refusal = await _check(env)
    assert refusal is None
    assert record["state"] == "disarmed"
    assert "no_new_privs" in record["why"] and "uid 0" in record["why"]
    assert [e["path"] for e in record["found"]] == ["/usr/local/bin/rootsh"]


ROOT_BUSYBOX = (
    "BF fwpower-root net_admin=1 setpriv=no-reuid nnp=1\n"
    "BF fwpower-found setuid /usr/local/bin/rootsh\n"
    "BF fwpower-scan scope=path caps=none\n"
)
ROOTSH_X = "BF fwpower-x /usr/local/bin/rootsh\n"


@pytest.mark.asyncio
async def test_busyboxs_no_new_privs_disarms_with_net_admin_left_in_the_bounding_set():
    """S3-R2-A-1: after su the agent's permitted, effective and ambient sets
    are empty, and under no_new_privs no exec adds to them: NET_ADMIN in
    the bounding set gives it nothing, read capabilities or not."""
    from benchflow.sandbox.firewall_power import net_admin_dropped

    env = ScriptedEnv(ROOT_BUSYBOX, drop=ROOTSH_X + _agent_line(BND_NET_ADMIN, nnp="1"))
    record, refusal = await _check(env)
    assert refusal is None, refusal
    assert record["state"] == "disarmed", record
    assert record["drop"] == "nnp"
    assert "before su" in record["why"] and "uid 0" in record["why"]
    assert record["agent_capabilities"]["bounding"] == BND_NET_ADMIN
    assert record["agent_capabilities"]["permitted"] == ZERO
    assert [e["path"] for e in record["found"]] == ["/usr/local/bin/rootsh"]
    assert net_admin_dropped(env, "agent")
    # Root scanned only the PATH: a drop was expected.
    (root,) = [c for c in env.calls if "BF fwpower-root" in c]
    assert '[ "$nnp" != 1 ]' in root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("drop", "words"),
    [
        (_agent_line(BND_NET_ADMIN, nnp="1", prm=BND_NET_ADMIN), "still hold"),
        (_agent_line(BND_NET_ADMIN, nnp="0"), "NoNewPrivs 0"),
    ],
)
async def test_busyboxs_drop_needs_no_new_privs_and_empty_sets(drop, words):
    from benchflow.sandbox.firewall_power import net_admin_dropped

    env = ScriptedEnv(
        ROOT_BUSYBOX,
        drop=ROOTSH_X + drop,
        nodrop=ROOTSH_X + _agent_line(BND_NET_ADMIN),
        rescan="BF fwpower-found setuid /usr/local/bin/rootsh\n"
        "BF fwpower-scan scope=all caps=none\n",
    )
    record, refusal = await _check(env)
    assert record["state"] == "refused", record
    assert words in record["no_drop"]
    assert refusal and "/usr/local/bin/rootsh (setuid root)" in refusal
    assert not net_admin_dropped(env, "agent")


@pytest.mark.asyncio
async def test_a_file_root_found_but_the_agent_user_cannot_run_does_not_count():
    env = ScriptedEnv(
        _root_out("none", HELPER_LINE + STANDARD_SETUID, "getcap", "all"),
        nodrop=_agent_line(BND_NET_ADMIN),
    )
    record, refusal = await _check(env, can_drop=False)
    assert refusal is None
    assert record["state"] == "clear"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("line", "words"),
    [
        (
            f"BF fwpower-agent bnd={BND_NET_ADMIN} inh={ZERO} amb={ZERO} "
            f"prm={BND_NET_ADMIN} eff={BND_NET_ADMIN} power=\n",
            "its own process",
        ),
        (_agent_line(BND_NET_ADMIN, "nft"), "a firewall tool (nft)"),
    ],
)
async def test_an_agent_user_holding_the_power_is_refused(line, words):
    env = ScriptedEnv(ROOT_NO_SETPRIV, nodrop=line)
    record, refusal = await _check(env, can_drop=False)
    assert record["state"] == "refused"
    assert refusal and words in refusal


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["agent", "scan"])
async def test_a_check_that_did_not_report_stops_the_trial(missing):
    root = (
        "BF fwpower-root net_admin=1 setpriv=none\n"
        if missing == "scan"
        else ROOT_NO_SETPRIV
    )
    env = ScriptedEnv(
        root, nodrop="" if missing == "agent" else _agent_line(BND_NET_ADMIN)
    )
    record, refusal = await _check(env, can_drop=False)
    assert record["state"] == "refused"
    assert refusal and refusal.startswith("sandbox startup refused")
    assert "did not report" in refusal


@pytest.mark.asyncio
async def test_a_session_factory_agent_gets_no_drop_and_root_scans_everything():
    """Guards the S3/S2 merge at 8eb4d7b3: audit actual session privileges."""
    env = ScriptedEnv(
        _root_out("ok", STANDARD_SETUID + CAP_LINE, "getcap", "all"),
        drop=FIXTURES["cap"]["drop"],
        nodrop=FIXTURES["cap"]["nodrop"],
    )
    record, refusal = await _check(env, can_drop=False)
    assert record["state"] == "refused"
    assert refusal is not None
    assert "setpriv" not in refusal
    assert "one by one" in record["no_drop"]
    # A session transport must be audited with the privileges it actually has.
    assert all(DROP not in c for c in env.calls if "BF fwpower-agent" in c)
    (root,) = [c for c in env.calls if "BF fwpower-root" in c]
    assert "all=1;" in root


@pytest.mark.asyncio
async def test_a_session_audit_uses_the_agent_identity_without_a_restricted_wrapper():
    """Guards the S3/S2 merge at 8eb4d7b3 against hiding session power."""
    env = FixtureEnv("cap")
    from benchflow.sandbox.firewall_power import check_firewall_power

    record, refusal = await check_firewall_power(
        env, "agent", holds=HOLDS, can_drop=False
    )
    assert record["state"] == "refused" and refusal
    (check,) = env.commands("BF fwpower-agent")
    command, user = check
    assert user == "agent"
    assert DROP not in command
    assert "a firewall tool" in refusal


@pytest.mark.asyncio
async def test_a_probe_that_worked_is_cleaned_up_by_root():
    env = ScriptedEnv(ROOT_NO_SETPRIV, nodrop=FIXTURES["cap"]["nodrop"])
    await _check(env, can_drop=False)
    assert any(
        "-w -X" in c and "BF fwpower-root" not in c and "BF fwpower-agent" not in c
        for c in env.calls
    )


def test_the_check_scripts_are_posix_sh():
    from benchflow.sandbox.firewall_power import (
        _agent_check_cmd,
        _cleanup_cmd,
        _root_cmd,
        _scan_cmd,
    )

    for shell in SHELLS:
        for cmd in (
            _root_cmd(),
            _root_cmd(scan_all=True),
            _scan_cmd(whole_fs=True),
            _scan_cmd(whole_fs=False),
            _agent_check_cmd("BFFWx", ["/usr/bin/a b", "/x'y"]),
            _cleanup_cmd("BFFWx"),
        ):
            subprocess.run([shell, "-n", "-c", cmd], check=True)
    assert sys.executable  # the suite's own interpreter ran these
