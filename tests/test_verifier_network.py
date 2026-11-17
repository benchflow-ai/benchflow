"""test.sh runs held to the verifier's network, root included (EN-609).

Root's test.sh runs the agent's code. It starts through a bounded launch (no
NET_ADMIN, NET_RAW, SETUID or SETGID, the verifier's group, no new
privileges) that records its capabilities; the verifier's network is
``verifier.network_mode``, else the sandbox's; a failed test.sh the network
refused packets ends "verifier network refused", unscored, and every reader
says so. The real-Docker matrix (four policies x oracle, script and model
agents) ran on a GCE VM; see the PR.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchflow._utils.scoring import (
    VERIFIER_ERROR_CATEGORIES,
    classify_score_outcome,
    classify_verifier_error,
    score_summary_fields,
)
from benchflow.sandbox import verifier_network as vn
from benchflow.task.config import NetworkMode, TaskConfig
from benchflow.task.verifier_errors import (
    VerifierLaunchRefusedError,
    VerifierNetworkRefusedError,
)

# The record test.sh writes as root on Docker (fx609, every policy): Docker's
# default set without NET_RAW; SETUID and SETGID kept (apt drops to _apt).
ROOT_HELD = (
    "BF verifier-launch uid=0 gid=0 CapBnd=00000000a80405fb "
    "CapEff=00000000a80405fb NoNewPrivs=1"
)


def _config(toml: str) -> TaskConfig:
    return TaskConfig.model_validate_toml('version = "1.0"\n' + toml)


# --- which network -------------------------------------------------------------


@pytest.mark.parametrize(
    ("toml", "mode"),
    [
        ("", "public"),
        ("[environment]\nallow_internet = false\n", "no-network"),
        (
            '[environment]\nnetwork_mode = "denylist"\nblocked_hosts = ["example.com"]\n',
            "denylist",
        ),
        (
            '[environment]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
            "allowlist",
        ),
        # The task's own verifier setting wins.
        (
            '[environment]\nallow_internet = false\n[verifier]\nnetwork_mode = "public"\n',
            "public",
        ),
        ('[verifier]\nnetwork_mode = "no-network"\n', "no-network"),
        # An [agent] override binds the agent alone, as documented.
        ('[agent]\nnetwork_mode = "no-network"\n', "public"),
        (
            '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
            "public",
        ),
    ],
)
def test_the_verifier_network_is_its_own_else_the_sandbox_s(toml, mode):
    assert vn.verifier_network_mode(_config(toml)) == mode


def test_a_separate_verifier_sandbox_keeps_the_network_it_declares():
    """WeightBench Qwen: an offline main, a separate verifier declaring public."""
    config = _config(
        "[environment]\nallow_internet = false\n"
        '[verifier]\nenvironment_mode = "separate"\n'
        '[verifier.environment]\nnetwork_mode = "public"\n'
    )
    assert vn.verifier_network_mode(config) == "no-network"
    assert vn.separate_verifier_network(config) == "public"


# --- the ending's rule -------------------------------------------------------------


@pytest.mark.parametrize(
    ("refused", "exit_code", "reward", "ends"),
    [
        (3, 1, 0.0, True),  # a non-zero exit (the exit form)
        (3, 1, 1.0, True),  # a non-zero exit, whatever reward it wrote
        (3, 0, 0.0, True),  # a 0 written with exit 0 (ProgramBench's)
        (3, 0, 0.5, True),  # a partial reward
        (3, 0, None, True),  # no reward
        (3, 0, 1.0, False),  # a pass despite refusals: the reward stands
        (0, 1, 0.0, False),  # nothing refused: the tests' own verdict
    ],
)
def test_refused_packets_and_no_pass_end_the_run(refused, exit_code, reward, ends):
    assert vn.network_refusal_ends(refused, exit_code, reward) is ends


# --- the launch -------------------------------------------------------------------


def test_the_launch_is_valid_sh_and_runs_the_command_held(tmp_path):
    record = tmp_path / "record"
    command = vn.verifier_launch("echo ran", record=str(record))
    assert subprocess.run(["bash", "-n", "-c", command]).returncode == 0
    done = subprocess.run(["bash", "-c", command], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ran"
    # A non-root user: no new privileges, no NET_ADMIN or NET_RAW in effect.
    text = record.read_text()
    assert "NoNewPrivs=1" in text and vn.launch_record_problem(text) is None


def test_root_s_launch_drops_only_the_network_and_tracing_capabilities():
    """SETUID and SETGID stay (apt drops to _apt; NETSCAN 11-14): the rules
    match every process, so changing user or group escapes nothing."""
    command = vn.verifier_launch("true")
    assert (
        "--no-new-privs --bounding-set=-net_admin,-net_raw,-sys_ptrace --inh-caps=-all"
    ) in command
    assert "--regid" not in command and "setuid" not in command
    # setpriv only from the system folders, never test.sh's PATH.
    assert "for bf_c in /usr/bin/setpriv /bin/setpriv" in command
    assert "exit 97" in command


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (ROOT_HELD, None),
        (None, "left no launch record"),
        ("garbage", "unreadable"),
        # Root with NET_ADMIN (bit 12) in its bounding set.
        (
            ROOT_HELD.replace("CapBnd=00000000a80405fb", "CapBnd=00000000a80c15fb"),
            "NET_ADMIN",
        ),
        # NET_RAW (bit 13) in effect.
        (
            ROOT_HELD.replace("CapEff=00000000a80405fb", "CapEff=00000000a80425fb"),
            "NET_RAW",
        ),
        # SYS_PTRACE (bit 19) in the bounding set: it could reach the proxy.
        (
            ROOT_HELD.replace("CapBnd=00000000a80405fb", "CapBnd=00000000a80805fb"),
            "SYS_PTRACE",
        ),
        # A non-root user with a full bounding set but no new privileges: held.
        (
            "BF verifier-launch uid=1000 gid=1000 CapBnd=000001ffffffffff "
            "CapEff=0000000000000000 NoNewPrivs=1",
            None,
        ),
        (
            "BF verifier-launch uid=1000 gid=1000 CapBnd=000001ffffffffff "
            "CapEff=0000000000000000 NoNewPrivs=0",
            "NET_ADMIN, NET_RAW",
        ),
    ],
)
def test_the_record_shows_whether_test_sh_ran_held(text, problem):
    found = vn.launch_record_problem(text)
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


# --- the rules -------------------------------------------------------------------


def test_every_process_is_held_and_only_the_proxy_s_marked_sockets_leave():
    from benchflow.sandbox.egress_denylist import DNS_PORT, EGRESS_MARK

    v4, v6, nat = vn.everyone_rules(
        "allowlist",
        subnets=("172.18.0.0/16",),
        gateways=("172.18.0.1",),
        allow_networks=("10.9.0.0/24",),
    )
    # No owner match anywhere: a uid or group change escapes nothing.
    assert not any("--uid-owner" in r or "--gid-owner" in r for r in v4 + v6 + nat)
    # Metadata and link-local first, whatever a route says.
    assert (
        v4[0]
        == "-d 169.254.169.254 -p tcp ! --dport 53 -j REJECT --reject-with tcp-reset"
    )
    assert (
        "-d 169.254.0.0/16 -p tcp ! --dport 53 -j REJECT --reject-with tcp-reset" in v4
    )
    mark = v4.index(f"-m mark --mark {EGRESS_MARK} -j ACCEPT")
    # The gateway (the host) is refused before its network is admitted.
    assert v4.index("-d 172.18.0.1 -j REJECT") < v4.index("-d 172.18.0.0/16 -j ACCEPT")
    assert mark < v4.index("-o lo -j ACCEPT")
    assert "-d 10.9.0.0/24 -j ACCEPT" in v4
    assert v4[-1] == "-j REJECT" and v6[-1] == "-j REJECT"
    assert nat == [
        f"-p {p} --dport 53 -m mark ! --mark {EGRESS_MARK} -j REDIRECT --to-ports {DNS_PORT}"
        for p in ("udp", "tcp")
    ]
    off, _, nat_off = vn.everyone_rules("no-network", subnets=("172.18.0.0/16",))
    assert nat_off == [] and "-d 10.9.0.0/24 -j ACCEPT" not in off
    assert not any("--dport 53 -j ACCEPT" in r for r in off)  # offline: no DNS out
    deny, _, _ = vn.everyone_rules("denylist")
    assert "-p udp --dport 53 -j ACCEPT" in deny


def test_routes_to_the_host_and_link_local_are_never_sandbox_networks():
    out = (
        "gw 0100800A\n"  # 10.128.0.1, the default gateway
        "net 0100800A FFFFFFFF\n"  # a host route to it
        "net FEA9FEA9 FFFFFFFF\n"  # 169.254.169.254/32 (GCE puts one in)
        "net 000012AC 0000FFFF\n"  # 172.18.0.0/16, the compose network
        "BF verifier-network routes 4\n"
    )
    assert vn.route_facts(out) == (4, ("172.18.0.0/16",), ("10.128.0.1",))


def test_the_rule_script_and_the_probes_are_valid_sh():
    v4, v6, nat = vn.everyone_rules(
        "allowlist", subnets=("172.18.0.0/16",), allow_networks=("10.0.0.0/8",)
    )
    for script in (
        vn.rules_cmd(v4, v6, nat),
        vn.count_cmd(),
        vn._route_facts_cmd(),
        vn._proof_cmd(True),
    ):
        assert subprocess.run(["sh", "-n", "-c", script]).returncode == 0


# --- the proxy -------------------------------------------------------------------


def test_the_proxy_never_reaches_metadata_or_link_local_even_by_a_listed_name(
    monkeypatch,
):
    """An allowlisted name that resolves to 169.254.169.254 (rebinding, a CNAME
    to metadata.google.internal) is refused, and so is Azure's WireServer,
    which is a global address."""
    from benchflow.sandbox import _egress_denylist_proxy as proxy
    from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4, CLOUD_METADATA_V6

    listed = {str(n.network_address) for n in proxy._NEVER if n.num_addresses == 1}
    assert listed == set(CLOUD_METADATA_V4) | set(CLOUD_METADATA_V6)
    policy = proxy.Policy([], [], allowed_hosts=["files.test", "169.254.0.0/16"])
    for address in (
        "169.254.169.254",
        "168.63.129.16",
        "169.254.1.1",
        "::ffff:169.254.169.254",
    ):
        assert not policy.address_allowed(address, 80), address
        assert not proxy._upstream_allowed(address, 80), address
    monkeypatch.setattr(proxy, "_resolve", lambda host, port: ["169.254.169.254"])
    with pytest.raises(proxy._PrivateDestination):
        proxy._connect_upstream(
            "files.test", 80, address_allowed=policy.address_allowed
        )


def test_the_proxy_refuses_an_ip_literal_the_list_does_not_name():
    from benchflow.sandbox import _egress_denylist_proxy as proxy

    policy = proxy.Policy([], [], allowed_hosts=["files.test"])
    assert policy._allow_rule("203.0.113.9", 443) == "not-allowlisted"
    assert policy._allow_rule("files.test", 443) is None


def test_only_a_process_holding_net_admin_can_mark_a_socket():
    """The bounded test.sh (no NET_ADMIN) cannot forge the proxy's mark."""
    import os
    import socket

    from benchflow.sandbox import _egress_denylist_proxy as proxy

    if os.geteuid() == 0:
        pytest.skip("root holds NET_ADMIN here")
    proxy._UPSTREAM_MARK[0] = 0xBF609
    try:
        with socket.socket() as sock, pytest.raises(PermissionError):
            proxy._marked(sock)
    finally:
        proxy._UPSTREAM_MARK[0] = 0


# --- the verifier -----------------------------------------------------------------


def _verifier(
    tmp_path: Path,
    armed: vn.VerifierNetwork,
    *,
    reward: str,
    exit_code: int = 0,
    record: str | None = ROOT_HELD,
    refused_now: int = 0,
):
    from benchflow.task import RolloutPaths, Verifier
    from tests.test_llm_judge_verifier import _make_task

    task = _make_task(tmp_path, 'version = "1.0"\n[verifier]\n')
    task.paths.tests_dir = task.task_dir / "tests"
    task.paths.test_path = task.task_dir / "tests" / "test.sh"
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    sandbox = MagicMock()
    sandbox.upload_dir = AsyncMock()
    sandbox.is_mounted = True
    sandbox._benchflow_verifier_network = armed
    commands: list[str] = []
    written: dict[str, str] = {}

    async def exec_(command, *args, **kwargs):
        commands.append(command)
        if "BF verifier-network refused" in command:
            return MagicMock(
                return_code=0, stdout=f"BF verifier-network refused {refused_now}\n"
            )
        if "IFS= read -r bf_l" in command and "verifier-launch-" in command:
            # Root reads the record from beside the receipt, in the sandbox,
            # with builtins (no program of the image runs for it).
            path = shlex.split(command.split("< ", 1)[1].split(";", 1)[0])[0]
            return MagicMock(return_code=0, stdout=written.get(path, ""))
        if "test-stdout" in command:
            # test.sh's command: the bounded launch's record, then its reward.
            if record is not None and "BF verifier-launch" in command:
                path = re.search(r"> (/run/benchflow/\S+?) 2>/dev/null", command)[1]
                written[path.strip("'")] = record + "\n"
            if reward is not None:
                paths.reward_text_path.write_text(reward)
            return MagicMock(return_code=exit_code, stdout="")
        return MagicMock(return_code=0, stdout="")

    sandbox.exec = AsyncMock(side_effect=exec_)
    return Verifier(task, paths, sandbox), paths, commands


@pytest.mark.asyncio
async def test_a_hardened_run_whose_test_sh_skipped_the_launch_is_refused(tmp_path):
    verifier, _, _ = _verifier(
        tmp_path, vn.VerifierNetwork("public", "open"), reward="1", record=None
    )
    with pytest.raises(VerifierLaunchRefusedError, match="left no launch record"):
        await verifier.verify()


@pytest.mark.asyncio
async def test_test_sh_starts_through_the_launch(tmp_path):
    verifier, _, commands = _verifier(
        tmp_path, vn.VerifierNetwork("public", "open"), reward="1"
    )
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    (test_command,) = [c for c in commands if "BF verifier-launch" in c]
    assert "--bounding-set=-net_admin,-net_raw,-sys_ptrace" in test_command
    # The record is in the sandbox's receipt folder, never /logs/verifier
    # (task preflights require it to hold only what test.sh writes).
    record = re.search(r"> (/run/benchflow/\S+?) 2>/dev/null", test_command)[1]
    assert record.startswith("/run/benchflow/verifier-launch-")
    # No program from the image's PATH runs as root for the record.
    assert not any("rm -f" in c for c in commands if "verifier-launch" in c)


def _held(baseline: int = 4) -> vn.VerifierNetwork:
    return vn.VerifierNetwork(
        "no-network",
        "held",
        match="every process",
        baseline=baseline,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("reward", "exit_code"), [("0", 1), ("0", 0), ("0.5", 0)])
async def test_a_failed_test_sh_the_network_refused_ends_verifier_network_refused(
    tmp_path, reward, exit_code
):
    verifier, paths, _ = _verifier(
        tmp_path, _held(), reward=reward, exit_code=exit_code, refused_now=7
    )
    with pytest.raises(VerifierNetworkRefusedError, match="refused it 3 packets"):
        await verifier.verify()
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record == {
        "where": "in-container",
        "mode": "no-network",
        "state": "held",
        "refused_packets": 3,
        "launch": ROOT_HELD,
    }


@pytest.mark.asyncio
async def test_a_pass_despite_refusals_keeps_its_reward_and_records_the_count(tmp_path):
    verifier, paths, _ = _verifier(tmp_path, _held(), reward="1", refused_now=6)
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record["refused_packets"] == 2


@pytest.mark.asyncio
async def test_a_separate_verifier_sandbox_runs_test_sh_under_its_own_network(tmp_path):
    armed = vn.VerifierNetwork(
        "public", "its own", where="separate sandbox", launch=False
    )
    verifier, paths, commands = _verifier(tmp_path, armed, reward="1", record=None)
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    assert not any("BF verifier-launch" in c for c in commands)
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record["where"] == "separate sandbox" and record["mode"] == "public"


@pytest.mark.asyncio
async def test_no_usable_reward_after_refusals_ends_verifier_network_refused(tmp_path):
    """FXPB: a verify.py that exits 3 with no reward after refused fetches ends
    "verifier network refused", not "verifier wrote no usable reward"."""
    verifier, _, _ = _verifier(
        tmp_path, _held(), reward=None, exit_code=3, refused_now=7
    )
    with pytest.raises(VerifierNetworkRefusedError, match="left no usable reward"):
        await verifier.verify()


@pytest.mark.asyncio
async def test_no_usable_reward_and_nothing_refused_keeps_its_ending(tmp_path):
    from benchflow.task.verifier_errors import RewardFileNotFoundError

    verifier, _, _ = _verifier(
        tmp_path, _held(), reward=None, exit_code=3, refused_now=4
    )
    with pytest.raises(RewardFileNotFoundError):
        await verifier.verify()


@pytest.mark.asyncio
async def test_an_offline_grader_that_fails_on_its_own_keeps_its_ending(tmp_path):
    """network: none, counted: a grader that never tries the network counts
    nothing, so its 0 is its own."""
    armed = vn.VerifierNetwork(
        "no-network",
        "no route, counted",
        match="every process",
        baseline=4,
        via_helper=True,
    )
    verifier, paths, _ = _verifier(tmp_path, armed, reward="0", refused_now=4)
    verifier._sandbox.metadata_rules_install = AsyncMock(
        return_value=(MagicMock(stdout="BF verifier-network refused 4\n"), "")
    )
    result = await verifier.verify()
    assert result.rewards == {"reward": 0.0}
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record["refused_packets"] == 0 and record["state"] == "no route, counted"


def test_the_no_route_counting_rules_are_valid_sh():
    script = vn._no_route_rules_cmd()
    # A dummy interface, not loopback: a connect is refused at once (a route
    # to loopback counted it but the refusal never came back; fx609 matrix).
    assert f"ip route replace default dev {vn.NO_ROUTE_LINK}" in script
    assert "dev lo" not in script
    assert subprocess.run(["sh", "-n", "-c", script]).returncode == 0


# --- every reader -------------------------------------------------------------------

REFUSED = (
    vn.VERIFIER_NETWORK_REFUSED
    + "test.sh exited 0 with reward 0.0 after the verifier's network (in-container "
    "under no-network: verifier.network_mode, else the sandbox's policy) refused it 3 "
    "packets, so the run is not scored."
)


def _rows() -> list[dict]:
    return [
        {
            "task_name": "a",
            "rewards": {"reward": 1.0},
            "error": None,
            "verifier_error": None,
        },
        {
            "task_name": "b",
            "rewards": None,
            "error": None,
            "verifier_error": REFUSED,
            # As the rollout's writer stores it (rollout/_results.py).
            "verifier_error_category": classify_verifier_error(REFUSED),
        },
    ]


def test_the_ending_is_unscored_and_counts_0_in_mean_reward():
    rows = _rows()
    assert classify_verifier_error(REFUSED) == "verifier_network_refused"
    assert "verifier_network_refused" in VERIFIER_ERROR_CATEGORIES
    assert classify_score_outcome(rows[1]) == "verifier_errored"
    summary = score_summary_fields(rows)
    assert summary["verifier_network_refusals"] == 1
    assert summary["verifier_errored"] == 1
    assert summary["mean_reward"] == pytest.approx(
        0.5
    )  # every trial, the refused one as 0
    assert summary["score_ratio"] == pytest.approx(0.5)
    assert summary["verifier_error_categories"] == {"verifier_network_refused": 1}


def test_explain_names_the_cause_and_the_remedy():
    from benchflow.failures import cause_of

    cause = cause_of(_rows()[1])
    assert cause is not None and cause.key == "verifier_network_refused"
    assert "verifier.network_mode" in cause.next_step


def test_the_results_jsonl_catalog_keeps_the_category():
    from benchflow.diagnostics import RolloutDiagnostics

    block = RolloutDiagnostics().to_results_jsonl_block(
        error_category=None, verifier_error_category="verifier_network_refused"
    )
    assert block == {
        "schema_version": 1,
        "verifier_error_category": "verifier_network_refused",
    }


def test_load_job_load_trial_and_the_export_read_the_ending(tmp_path):
    import benchflow as bf
    from benchflow.job_export import job_export

    job = tmp_path / "job"
    for i, row in enumerate(_rows()):
        trial = job / f"{row['task_name']}__{i:08x}"
        trial.mkdir(parents=True)
        trial.joinpath("result.json").write_text(
            json.dumps({**row, "rollout_name": trial.name, "agent": "claude-agent-acp"})
        )
    loaded = bf.load_job(job)
    refused = next(t for t in loaded.trials if t.raw.get("verifier_error"))
    assert refused.reward is None  # unscored, never a 0
    assert (
        bf.load_trial(refused.path)
        .raw["verifier_error"]
        .startswith(vn.VERIFIER_NETWORK_REFUSED)
    )
    exported = job_export(loaded, include_verifier=False)
    (row,) = [t for t in exported.trials if t.verifier_error]
    assert row.verifier_error_category == "verifier_network_refused"
    assert row.score_outcome == "verifier_errored"


def test_the_rollout_records_the_ending_with_its_words():
    """_verify_rollout writes BenchFlow's own opening words before the reason."""
    import inspect

    from benchflow.rollout import _setup

    source = inspect.getsource(_setup)
    assert 'verifier_error = f"{VERIFIER_NETWORK_REFUSED}{e}"' in source
    assert 'verifier_error = f"{VERIFIER_LAUNCH_REFUSED_ERROR}{e}"' in source
    assert 'verifier_error = f"{VERIFIER_NETWORK_UNHELD}{e}"' in source
    assert isinstance(NetworkMode.PUBLIC.value, str) and SimpleNamespace is not None


@pytest.mark.asyncio
async def test_an_empty_record_read_is_read_again_before_it_refuses(tmp_path):
    """RCI's L-1 (1 launch in 603): a read of the launch record that comes
    back empty is made again before the run is refused; a record that is
    really missing still refuses (test above)."""
    verifier, _, _ = _verifier(
        tmp_path, vn.VerifierNetwork("public", "open"), reward="1"
    )
    inner = verifier._sandbox.exec.side_effect
    misses: list[str] = []

    async def flaky(command, *args, **kwargs):
        record_read = "IFS= read -r bf_l" in command and "verifier-launch-" in command
        if record_read and not misses:
            misses.append(command)
            return MagicMock(return_code=0, stdout="")
        return await inner(command, *args, **kwargs)

    verifier._sandbox.exec.side_effect = flaky
    result = await verifier.verify()
    assert result.rewards == {"reward": 1.0}
    assert len(misses) == 1


@pytest.mark.parametrize(
    ("status", "refused"),
    [
        (
            "Uid:\t0\t0\t0\t0\nGid:\t0\t0\t0\t0\nCapEff:\t00000000a80405fb\n"
            "CapBnd:\t00000000a80405fb\nNoNewPrivs:\t1\n",
            False,
        ),
        # Still empty after its reads: refused, as before (fail closed).
        ("", True),
    ],
    ids=["held", "unreadable"],
)
def test_the_launch_check_reads_its_status_again_and_still_fails_closed(
    tmp_path, status, refused
):
    """RCI's L-1: the launch's own check reads /proc/self/status again while
    a field came back empty; one that stays empty refuses (exit 97)."""
    import subprocess

    status_file = tmp_path / "status"
    status_file.write_text(status)
    record = tmp_path / "record"
    script = (
        vn._caps_check(str(record)).replace("/proc/self/status", str(status_file))
        + "echo RAN"
    )
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    if refused:
        assert done.returncode == vn.VERIFIER_LAUNCH_REFUSED, done
        assert "RAN" not in done.stdout
    else:
        assert done.returncode == 0 and "RAN" in done.stdout, done
        assert "CapBnd=00000000a80405fb" in record.read_text()


@pytest.mark.parametrize(
    ("ending", "category"),
    [
        (vn.VERIFIER_NETWORK_REFUSED + "test.sh exited 1", "verifier_network_refused"),
        (vn.VERIFIER_LAUNCH_REFUSED_ERROR + "no setpriv", "verifier_launch_refused"),
    ],
)
def test_a_solver_preserved_ending_keeps_its_category(ending, category):
    """A recovery-eligible task tags the ending ``[solver-preserved]``
    (rollout._verifier_recovery.PRESERVED_SOLVER); its category is the
    ending's, not verifier_other (#234's 8fc85a736)."""
    assert classify_verifier_error(ending) == category
    assert classify_verifier_error("[solver-preserved] " + ending) == category
