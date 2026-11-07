"""The cloud metadata endpoints are refused to the agent under an open network.

Guards PY8 S2: under the default open network nothing refused 169.254.169.254
and the other cloud metadata endpoints, so an agent in a sandbox on a cloud VM
could read the VM's instance credentials. Root's firewall script (iptables,
ip6tables, the legacy backend, an iptables install), the agent user's check
(no NET_ADMIN, no firewall tool that grants it, a refused test connect), the
record, the opt-out key and the NET_ADMIN rule are tested here; the
deterministic tier runs the whole thing in a real sandbox.

The scripts run in a real POSIX shell against stub firewall tools, a stub
``/proc`` and a stub apt list folder in ``tmp_path``. The agent check's test
connect goes to a stub ``timeout`` that never connects anywhere.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests._receipt_fakes import inner_command, receipt_answer


@pytest.fixture(autouse=True)
def _isolate_metadata_policy_from_compose_admission(monkeypatch):
    """a4d08b3d: this mocked-backend matrix tests the metadata capability policy."""
    from benchflow.sandbox import setup

    validate = setup._validate_task_runtime_for_launch

    def without_compose(*args, **kwargs):
        kwargs["check_compose"] = False
        return validate(*args, **kwargs)

    monkeypatch.setattr(setup, "_validate_task_runtime_for_launch", without_compose)
    monkeypatch.setattr(setup, "_freeze", lambda sandbox: None)
    monkeypatch.setattr(
        "benchflow.evaluation._refuse_unnamed_task_env", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        setup, "_preflight_daytona_compose", lambda *args, **kwargs: None
    )


V4 = (
    "169.254.169.254",
    "169.254.170.2",
    "169.254.170.23",
    "100.100.100.200",
    "168.63.129.16",
    "169.254.0.23",
)
V6 = ("fd00:ec2::254", "fd00:ec2::23", "fd20:ce::254", "fd00:c1::a9fe:a9fe")

# Root's CapEff with and without CAP_NET_ADMIN (bit 12): Docker's default set,
# and the same with NET_ADMIN added.
CAPS_DEFAULT = "00000000a80425fb"
CAPS_NET_ADMIN = "00000000a80435fb"

FAKE_FIREWALL = r'''#!{python}
"""A stateful iptables or ip6tables: OUTPUT rules per tool in a JSON file."""
import json, os, sys
name = os.path.basename(sys.argv[0])
args = [a for a in sys.argv[1:] if a != "-w"]
with open(os.environ["FW_LOG"], "a") as log:
    log.write(name + " " + " ".join(sys.argv[1:]) + "\n")
if name in os.environ.get("FW_BROKEN", "").split():
    print(name + ": cannot initialize table: Permission denied", file=sys.stderr)
    sys.exit(3)
path = os.environ["FW_STATE"]
state = json.load(open(path)) if os.path.exists(path) else {{}}
rules = state.setdefault(name, [])
op = args[0]
if op == "-S":
    hits = os.environ.get("FW_HITS", "0")
    for spec in rules if "-v" in args else []:
        t = spec.split()
        t[1] += "/128" if ":" in t[1] else "/32"
        t.insert(4, "-m tcp")
        j = t.index("-j")
        t[j:j] = ["-c", hits, "60"]
        print("-A OUTPUT " + " ".join(t))
    sys.exit(0)
if op in ("-N", "-X"):
    sys.exit(0 if os.environ.get("FW_AGENT_POWER") == name else 1)
spec = " ".join(args[2:] if op == "-C" else args[3:])
if op == "-C":
    sys.exit(0 if spec in rules else 1)
if op == "-I":
    rules.insert(0, spec)
json.dump(state, open(path, "w"))
'''

FAKE_APT_GET = r"""#!/bin/sh
# Records each call; "update" rewrites the lists, "install" then puts the
# firewall tools on PATH (unless APT_FAILS is set).
echo "apt-get $*" >> "$APT_LOG"
case "$1" in
  update) rm -f "$APT_LISTS"/*; echo fresh > "$APT_LISTS/fresh-list"; exit 0;;
  install)
    [ -n "$APT_FAILS" ] && { echo "E: Unable to locate package iptables"; exit 100; }
    [ -e "$APT_LISTS/fresh-list" ] || { echo "E: Unable to locate package iptables"; exit 100; }
    cp "$APT_STUB" "$APT_BIN/iptables"; cp "$APT_STUB" "$APT_BIN/ip6tables"; exit 0;;
esac
"""

FAKE_GETCAP = """#!/bin/sh
# getcap -v emits the canonical path even when it has no file capabilities.
case "${CAP_READ:-ok}" in
  failed) exit 1;;
  empty) exit 0;;
  malformed) echo incomplete; exit 0;;
esac
printf '%s%s\\n' "$2" "${FILE_CAPS:+ $FILE_CAPS}"
"""


def _rule(address: str) -> str:
    return (
        f"-d {address} -p tcp ! --dport 53 -m owner ! --uid-owner 0 "
        "-j REJECT --reject-with tcp-reset"
    )


class Sandbox:
    """A stub sandbox in ``tmp_path``: firewall tools, /proc, apt lists."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        # Root runs a tool only from folders no one else can write (umask 002
        # would leave them group-writable).
        self.bin.chmod(0o755)
        self.proc = tmp_path / "proc"
        (self.proc / "self").mkdir(parents=True)
        (self.proc / "net").mkdir()
        (self.proc / "uptime").write_text("100.25 400.00\n")
        self.lists = tmp_path / "apt-lists"
        self.lists.mkdir()
        (self.lists / "image-list").write_text("from the image\n")
        self.stub = tmp_path / "fake-firewall"
        self.stub.write_text(FAKE_FIREWALL.format(python=sys.executable))
        self.stub.chmod(0o755)
        self.env = {
            "FW_STATE": str(tmp_path / "fw.json"),
            "FW_LOG": str(tmp_path / "fw.log"),
            "APT_LOG": str(tmp_path / "apt.log"),
            "APT_LISTS": str(self.lists),
            "APT_STUB": str(self.stub),
            "APT_BIN": str(self.bin),
        }
        self.caps(CAPS_NET_ADMIN)
        self.ipv6("local")
        self.script("getcap", FAKE_GETCAP)

    def tools(self, *names: str) -> Sandbox:
        for name in names:
            (self.bin / name).write_bytes(self.stub.read_bytes())
            (self.bin / name).chmod(0o755)
        return self

    def script(self, name: str, body: str) -> Sandbox:
        p = self.bin / name
        p.write_text(body)
        p.chmod(0o755)
        return self

    def caps(self, eff: str, prm: str | None = None, amb: str = "0" * 16) -> Sandbox:
        (self.proc / "self" / "status").write_text(
            f"Name:\tsh\nCapInh:\t{'0' * 16}\nCapPrm:\t{prm or eff}\n"
            f"CapEff:\t{eff}\nCapBnd:\t{CAPS_NET_ADMIN}\nCapAmb:\t{amb}\n"
        )
        return self

    def ipv6(self, kind: str) -> Sandbox:
        """``absent`` (no IPv6 in the kernel), ``local`` (loopback only, IPv6
        off on eth0, as Docker leaves a network without IPv6), ``enabled``
        (loopback only so far, IPv6 on for eth0) or ``global`` (eth0 has an
        address that routes out)."""
        path = self.proc / "net" / "if_inet6"
        lo = "00000000000000000000000000000001 01 80 10 80       lo\n"
        lines = {
            "absent": None,
            "local": lo,
            "enabled": lo,
            "global": lo + "fd000000000000000000000000000002 02 40 00 80     eth0\n",
        }[kind]
        if lines is None:
            path.unlink(missing_ok=True)
            shutil.rmtree(self.proc / "sys" / "net" / "ipv6", ignore_errors=True)
            return self
        else:
            path.write_text(lines)
        conf = self.proc / "sys" / "net" / "ipv6" / "conf"
        eth0_off = "1" if kind == "local" else "0"
        for name, value in (("all", "0"), ("default", "0"), ("lo", "0")):
            (conf / name).mkdir(parents=True, exist_ok=True)
            (conf / name / "disable_ipv6").write_text(value + "\n")
        (conf / "eth0").mkdir(parents=True, exist_ok=True)
        (conf / "eth0" / "disable_ipv6").write_text(eth0_off + "\n")
        return self

    def _system_bin(self) -> Path:
        """/usr/bin and /bin as links, without the host's package managers,
        firewall tools, bash and timeout: the script would find those first
        (a Linux CI runner has apt-get) instead of this sandbox's stubs."""
        sys_bin = self.root / "sysbin"
        if sys_bin.exists():
            return sys_bin
        sys_bin.mkdir()
        sys_bin.chmod(0o755)
        hidden = {"apk", "apt", "apt-get", "dnf", "microdnf", "yum", "nft"}
        hidden |= {"bash", "timeout"}
        for d in ("/usr/bin", "/bin"):
            for src in Path(d).iterdir():
                name = src.name
                if name in hidden or name.startswith(("iptables", "ip6tables")):
                    continue
                if not (sys_bin / name).is_symlink():
                    (sys_bin / name).symlink_to(src)
        return sys_bin

    def run(self, cmd: str, shell: str = "/bin/sh", **env: str) -> str:
        result = subprocess.run(
            [shell, "-c", cmd],
            capture_output=True,
            text=True,
            env={
                "PATH": f"{self.bin}:{self._system_bin()}",
                **self.env,
                **env,
            },
            check=True,
        )
        return result.stdout

    def block(self, shell: str = "/bin/sh", **env: str) -> dict[str, str]:
        from benchflow.sandbox.lockdown import _cloud_metadata_block_cmd

        out = self.run(
            _cloud_metadata_block_cmd(proc=str(self.proc), apt_lists=str(self.lists)),
            shell,
            **env,
        )
        (line,) = [ln for ln in out.splitlines() if ln.startswith("BF metadata ")]
        fields = dict(f.split("=", 1) for f in line.split()[2:])
        fields["_out"] = out
        return fields

    def rules(self, tool: str) -> list[str]:
        path = self.root / "fw.json"
        return json.loads(path.read_text()).get(tool, []) if path.exists() else []

    def calls(self, name: str = "fw.log") -> list[str]:
        path = self.root / name
        return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def sandbox(tmp_path, monkeypatch) -> Sandbox:
    from benchflow.sandbox import lockdown

    box = Sandbox(tmp_path)
    # Root's scripts search only their own PATH, and run a firewall tool only
    # when every folder and link on its way is root's alone: here the stub
    # folders, which belong to the test's user.
    monkeypatch.setattr(
        lockdown,
        "_CLOUD_METADATA_ROOT_PATH",
        f"{box.bin}:{box._system_bin()}",
        raising=False,
    )
    monkeypatch.setattr(
        lockdown,
        "_CLOUD_METADATA_TRUSTED_UIDS",
        ("0", str(os.getuid())),
        raising=False,
    )
    return box


SHELLS = ["/bin/sh", "/bin/bash"]


def test_the_endpoint_list_is_the_rust_benchs():
    from benchflow.sandbox.cloud_metadata import CLOUD_METADATA_V4, CLOUD_METADATA_V6

    assert (CLOUD_METADATA_V4, CLOUD_METADATA_V6) == (V4, V6)


# --- root's rules -------------------------------------------------------------


@pytest.mark.parametrize("shell", SHELLS)
def test_iptables_and_ip6tables_refuse_every_endpoint_for_non_root_users(
    sandbox, shell
):
    sandbox.tools("iptables", "ip6tables")
    report = sandbox.block(shell)

    assert report["status"] == "0", report["_out"]
    assert (report["fw4"], report["fw6"], report["v6"]) == (
        "iptables",
        "ip6tables",
        "enabled",
    )
    assert report["install"] == "none"
    assert sorted(sandbox.rules("iptables")) == sorted(_rule(a) for a in V4)
    assert sorted(sandbox.rules("ip6tables")) == sorted(_rule(a) for a in V6)
    # Each rule goes in at the top of OUTPUT, above any ACCEPT.
    assert f"iptables -w -I OUTPUT 1 {_rule(V4[0])}" in sandbox.calls()


def test_running_it_again_adds_nothing(sandbox):
    sandbox.tools("iptables", "ip6tables")
    sandbox.block()
    sandbox.block()

    assert len(sandbox.rules("iptables")) == len(V4)
    assert len(sandbox.rules("ip6tables")) == len(V6)


def test_the_legacy_backend_is_used_where_nf_tables_does_not_work(sandbox):
    sandbox.tools("iptables", "ip6tables", "iptables-legacy", "ip6tables-legacy")
    report = sandbox.block(FW_BROKEN="iptables ip6tables")

    assert report["status"] == "0", report["_out"]
    assert (report["fw4"], report["fw6"]) == ("iptables-legacy", "ip6tables-legacy")
    assert sorted(sandbox.rules("iptables-legacy")) == sorted(_rule(a) for a in V4)
    assert sorted(sandbox.rules("ip6tables-legacy")) == sorted(_rule(a) for a in V6)
    assert sandbox.rules("iptables") == []


def test_iptables_nft_alone_is_used(sandbox):
    sandbox.tools("iptables-nft", "ip6tables-nft")
    report = sandbox.block()

    assert (report["status"], report["fw4"], report["fw6"]) == (
        "0",
        "iptables-nft",
        "ip6tables-nft",
    )
    assert len(sandbox.rules("iptables-nft")) == len(V4)


def test_no_ipv6_in_the_kernel_needs_no_ip6tables(sandbox):
    """Guards S2-CONF-4 at 84b4ab01: absent proc entries prove no lifetime exemption."""
    sandbox.tools("iptables").ipv6("absent")
    report = sandbox.block()

    assert (report["status"], report["v6"], report["fw6"]) == ("3", "absent", "-")
    assert not any(c.startswith("ip6tables") for c in sandbox.calls())


def test_ipv6_with_a_route_out_and_no_ip6tables_is_status_3(sandbox):
    sandbox.tools("iptables").ipv6("global")
    report = sandbox.block()

    # No package manager to install ip6tables with: the IPv4 rules hold.
    assert (report["status"], report["v6"]) == ("3", "global"), report["_out"]
    assert len(sandbox.rules("iptables")) == len(V4)


def test_loopback_only_ipv6_without_ip6tables_is_enough(sandbox):
    """Guards S2-CONF-4 at a070066d: loopback alone no longer proves safety."""
    sandbox.tools("iptables").ipv6("local")
    report = sandbox.block()

    assert (report["status"], report["install"]) == ("3", "unavailable")


def test_ipv6_enabled_on_an_interface_without_an_address_yet_needs_ip6tables(
    sandbox,
):
    # Only loopback had an address when the block ran, so no ip6tables was
    # needed and the record said on; a service that finished configuring
    # IPv6 later gave the agent a route to fd00:ec2::254.
    sandbox.tools("iptables").ipv6("enabled")
    report = sandbox.block()

    assert (report["status"], report["v6"]) == ("3", "enabled"), report["_out"]
    assert len(sandbox.rules("iptables")) == len(V4)


def test_ipv6_enabled_and_a_failed_ip6tables_rule_is_a_failed_block(sandbox):
    sandbox.tools("iptables", "ip6tables").ipv6("enabled")
    report = sandbox.block(FW_BROKEN="ip6tables")

    assert report["status"] != "0", report["_out"]


def test_unreadable_ipv6_settings_count_as_enabled(sandbox):
    sandbox.tools("iptables").ipv6("local")
    shutil.rmtree(sandbox.proc / "sys")
    report = sandbox.block()

    assert (report["status"], report["v6"]) == ("3", "enabled"), report["_out"]


@pytest.mark.parametrize("settings", ["future-enabled", "all-disabled", "unreadable"])
@pytest.mark.parametrize("broken_tool", [False, True])
def test_future_interfaces_need_ipv6_rules(sandbox, settings, broken_tool):
    """Guards S2-CONF-4 at a070066d: a snapshot cannot constrain later links."""
    sandbox.tools("iptables").ipv6("local")
    conf = sandbox.proc / "sys" / "net" / "ipv6" / "conf"
    if settings == "all-disabled":
        for name in ("all", "default", "lo", "eth0"):
            (conf / name / "disable_ipv6").write_text("1\n")
    elif settings == "unreadable":
        (conf / "default" / "disable_ipv6").unlink()
    if broken_tool:
        sandbox.tools("ip6tables")
    report = sandbox.block(FW_BROKEN="ip6tables" if broken_tool else "")

    assert report["status"] == "3", report["_out"]
    assert report["v6"] == "enabled"
    assert len(sandbox.rules("iptables")) == len(V4)


def test_ipv6_sysctls_without_an_address_list_still_need_rules(sandbox):
    """Guards S2-CONF-4 at a070066d when only kernel IPv6 sysctls are visible."""
    sandbox.tools("iptables").ipv6("local")
    (sandbox.proc / "net" / "if_inet6").unlink()
    report = sandbox.block()

    assert (report["status"], report["v6"]) == ("3", "enabled"), report["_out"]


# --- root runs only firewall tools the agent cannot change --------------------

SHIM = """#!/bin/sh
echo "$0 $*" >> "$SHIM_LOG"
exec "$SHIM_REAL" "$@"
"""


def _shim_dir(sandbox: Sandbox, name: str, mode: int = 0o755) -> Path:
    d = sandbox.root / name
    d.mkdir()
    for tool in ("iptables", "ip6tables"):
        (d / tool).write_text(SHIM)
        (d / tool).chmod(0o755)
    d.chmod(mode)
    return d


def test_root_never_runs_a_firewall_tool_from_the_agents_path(sandbox):
    # The image's PATH put an agent-owned folder first (/app/bin); between
    # two scenes the agent left an iptables there, which root's next block
    # ran with NET_ADMIN.
    sandbox.tools("iptables", "ip6tables")
    agent_bin = _shim_dir(sandbox, "agent-bin")
    hostile = {
        "PATH": f"{agent_bin}:{sandbox.bin}:{sandbox._system_bin()}",
        "SHIM_LOG": str(sandbox.root / "shim.log"),
        "SHIM_REAL": str(sandbox.stub),
    }
    report = sandbox.block(**hostile)

    assert report["status"] == "0", report["_out"]
    assert sandbox.calls("shim.log") == []
    assert len(sandbox.rules("iptables")) == len(V4)

    from benchflow.sandbox.lockdown import _cloud_metadata_after_cmd

    out = sandbox.run(_cloud_metadata_after_cmd("iptables", "BFMDtest"), **hostile)
    assert "BF metadata-hits 0" in out
    assert sandbox.calls("shim.log") == []


@pytest.mark.parametrize("how", ["file", "link"])
def test_a_firewall_tool_others_can_change_is_never_run_as_root(
    sandbox, monkeypatch, how
):
    from benchflow.sandbox import lockdown

    # A folder anyone can write (no sticky bit), on root's PATH or on the
    # way to the tool through a link.
    open_dir = _shim_dir(sandbox, "open", mode=0o777)
    if how == "file":
        path = f"{open_dir}:{sandbox._system_bin()}"
    else:
        sandbox.bin.joinpath("iptables").symlink_to(open_dir / "iptables")
        path = f"{sandbox.bin}:{sandbox._system_bin()}"
    monkeypatch.setattr(lockdown, "_CLOUD_METADATA_ROOT_PATH", path, raising=False)
    report = sandbox.block(
        SHIM_LOG=str(sandbox.root / "shim.log"), SHIM_REAL=str(sandbox.stub)
    )

    assert sandbox.calls("shim.log") == []
    assert (report["status"], report["fw4"]) == ("6", "-"), report["_out"]


def test_tools_root_cannot_use_say_no_net_admin_or_a_failed_rule(sandbox):
    sandbox.tools("iptables", "ip6tables").caps(CAPS_DEFAULT)
    assert sandbox.block(FW_BROKEN="iptables ip6tables")["status"] == "4"
    sandbox.caps(CAPS_NET_ADMIN)
    assert sandbox.block(FW_BROKEN="iptables ip6tables")["status"] == "1"


# --- no iptables in the image -------------------------------------------------


def test_iptables_is_installed_and_apts_lists_are_left_as_found(sandbox):
    sandbox.script("apt-get", FAKE_APT_GET)
    report = sandbox.block()

    assert report["status"] == "0", report["_out"]
    assert report["install"] == "ok"
    start, end = report["install_t"].split(",")
    assert float(start) == float(end) == 100.25
    assert len(sandbox.rules("iptables")) == len(V4)
    # Installed without an update first; the update's fresh lists are gone
    # again and the image's own are back.
    assert sandbox.calls("apt.log")[0].startswith("apt-get install")
    assert "apt-get update -qq" in sandbox.calls("apt.log")
    assert sorted(p.name for p in sandbox.lists.iterdir()) == ["image-list"]
    assert (sandbox.lists / "image-list").read_text() == "from the image\n"


def test_the_lists_are_never_restored_from_an_incomplete_copy(sandbox):
    """A copy that failed (here: a lists folder root cannot read) must not
    replace what apt left with nothing."""
    if os.geteuid() == 0:
        pytest.skip("root reads a folder whatever its mode")
    sandbox.script("apt-get", FAKE_APT_GET)
    sandbox.lists.chmod(0o300)
    try:
        report = sandbox.block()
    finally:
        sandbox.lists.chmod(0o755)

    assert report["status"] == "0", report["_out"]
    assert (sandbox.lists / "fresh-list").exists()


def test_no_iptables_and_no_package_manager_is_status_2(sandbox):
    report = sandbox.block()

    assert (report["status"], report["install"]) == ("2", "unavailable")


def test_a_failed_install_is_status_5_with_its_last_lines(sandbox):
    sandbox.script("apt-get", FAKE_APT_GET)
    report = sandbox.block(APT_FAILS="1")

    assert (report["status"], report["install"]) == ("5", "failed")
    assert "BF metadata-install: " in report["_out"]
    assert "Unable to locate package iptables" in report["_out"]
    assert sorted(p.name for p in sandbox.lists.iterdir()) == ["image-list"]


def test_without_net_admin_nothing_is_installed(sandbox):
    sandbox.script("apt-get", FAKE_APT_GET).caps(CAPS_DEFAULT)
    report = sandbox.block()

    assert (report["status"], report["install"]) == ("4", "no-net-admin")
    assert sandbox.calls("apt.log") == []


def test_apk_installs_both_families(sandbox):
    sandbox.script(
        "apk",
        '#!/bin/sh\necho "apk $*" >> "$APT_LOG"\n'
        'cp "$APT_STUB" "$APT_BIN/iptables"; cp "$APT_STUB" "$APT_BIN/ip6tables"\n',
    )
    report = sandbox.block()

    assert (report["status"], report["install"]) == ("0", "ok")
    assert sandbox.calls("apt.log") == ["apk add --no-cache iptables ip6tables"]


def test_the_hit_counter_is_read_from_the_test_endpoints_rule(sandbox):
    sandbox.tools("iptables", "ip6tables")
    assert sandbox.block()["hits"] == "0"
    assert sandbox.block(FW_HITS="3")["hits"] == "3"


# --- the agent user's check ----------------------------------------------------

FAKE_TIMEOUT = """#!/bin/sh
# Stands in for `timeout 5 bash -c 'exec 3<>/dev/tcp/...'`: never connects.
echo "$*" >> "$CONNECT_LOG"
[ "${CONNECT_RC:-1}" = 1 ] && echo "bash: connect: Connection refused" >&2
exit "${CONNECT_RC:-1}"
"""


def _agent(
    sandbox: Sandbox, *, connect_probe: bool = True, **env: str
) -> dict[str, str]:
    from benchflow.sandbox.lockdown import _cloud_metadata_agent_check_cmd

    out = sandbox.run(
        _cloud_metadata_agent_check_cmd(
            "BFMDtest", proc=str(sandbox.proc), connect_probe=connect_probe
        ),
        CONNECT_LOG=str(sandbox.root / "connect.log"),
        **env,
    )
    (line,) = [ln for ln in out.splitlines() if ln.startswith("BF metadata-agent ")]
    return dict(f.split("=", 1) for f in line.split()[2:])


def test_the_agent_check_sees_no_net_admin_and_a_refused_connect(sandbox):
    sandbox.tools("iptables", "ip6tables").caps("0" * 16)
    sandbox.script("bash", "#!/bin/sh\nexit 0\n").script("timeout", FAKE_TIMEOUT)

    assert _agent(sandbox) == {"held": "0", "power": "", "connect": "refused"}
    (call,) = sandbox.calls("connect.log")
    assert call == "5 bash -c exec 3<>/dev/tcp/169.254.169.254/80"


@pytest.mark.parametrize(
    ("rc", "connect"), [("0", "reached"), ("124", "timeout"), ("7", "failed")]
)
def test_the_agent_checks_connect_outcomes(sandbox, rc, connect):
    sandbox.caps("0" * 16)
    sandbox.script("bash", "#!/bin/sh\nexit 0\n").script("timeout", FAKE_TIMEOUT)

    assert _agent(sandbox, CONNECT_RC=rc)["connect"] == connect


def test_without_bash_or_timeout_the_connect_is_skipped(sandbox):
    sandbox.caps("0" * 16)
    assert _agent(sandbox)["connect"] == "skipped"


@pytest.mark.parametrize(
    "caps",
    [
        {"eff": CAPS_NET_ADMIN},
        {"eff": "0" * 16, "prm": "0000000000001000"},
        {"eff": "0" * 16, "prm": "0" * 16, "amb": "0000000000001000"},
    ],
    ids=["effective", "permitted", "ambient"],
)
def test_the_agent_check_sees_net_admin_in_any_set(sandbox, caps):
    sandbox.caps(**caps)
    assert _agent(sandbox)["held"] == "1"


def test_the_agent_check_finds_a_firewall_tool_that_works_for_it(sandbox):
    """A setcap'd or setuid iptables would let the agent undo the rules."""
    sandbox.tools("iptables", "ip6tables").caps("0" * 16)
    report = _agent(sandbox, FW_AGENT_POWER="iptables")

    assert report["power"] == "iptables"
    assert "iptables -w -N BFMDtest" in sandbox.calls()
    assert "iptables -w -X BFMDtest" in sandbox.calls()


def test_privilege_only_check_never_connects(sandbox):
    """Guards S2-CONF-3 at a070066d: failed installs need only a power audit."""
    sandbox.tools("iptables").caps("0" * 16)
    sandbox.script("bash", "#!/bin/sh\nexit 0\n").script("timeout", FAKE_TIMEOUT)
    report = _agent(sandbox, connect_probe=False, FW_AGENT_POWER="iptables")

    assert report == {"held": "0", "power": "iptables", "connect": "skipped"}
    assert sandbox.calls("connect.log") == []


@pytest.mark.parametrize(
    "tool",
    [
        "iptables",
        "iptables-nft",
        "iptables-legacy",
        "ip6tables",
        "ip6tables-nft",
        "ip6tables-legacy",
        "nft",
    ],
)
@pytest.mark.parametrize(
    "directory",
    ["usr/local/sbin", "usr/local/bin", "usr/sbin", "usr/bin", "sbin", "bin"],
)
@pytest.mark.parametrize("shadowed", [False, True])
def test_power_audit_checks_every_system_candidate(
    sandbox, monkeypatch, tool, directory, shadowed
):
    """Guards S2-CONF-3A at 84b4ab01: PATH cannot hide a capable tool."""
    from benchflow.sandbox import lockdown

    system = sandbox.root / "system"
    directories = [
        system / d
        for d in (
            "usr/local/sbin",
            "usr/local/bin",
            "usr/sbin",
            "usr/bin",
            "sbin",
            "bin",
        )
    ]
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
    candidate = system / directory / tool
    candidate.write_text(
        "#!/bin/sh\nexit 0\n" if tool == "nft" else sandbox.stub.read_text()
    )
    candidate.chmod(0o775)
    monkeypatch.setattr(
        lockdown,
        "_CLOUD_METADATA_ROOT_PATH",
        ":".join(map(str, [*directories, sandbox.bin, sandbox._system_bin()])),
    )
    if shadowed:
        sandbox.script(tool, "#!/bin/sh\nexit 1\n")
    sandbox.caps("0" * 16)

    report = _agent(sandbox, connect_probe=False, FW_AGENT_POWER=tool)

    assert tool in report["power"].split(",")
    assert report["connect"] == "skipped"


@pytest.mark.parametrize(
    "tool",
    [
        "iptables",
        "iptables-nft",
        "iptables-legacy",
        "ip6tables",
        "ip6tables-nft",
        "ip6tables-legacy",
        "nft",
    ],
)
@pytest.mark.parametrize(
    "directory",
    ["usr/local/sbin", "usr/local/bin", "usr/sbin", "usr/bin", "sbin", "bin"],
)
@pytest.mark.parametrize(
    "privilege", ["cap_net_admin=ep", "cap_net_admin=p", "setuid", "setgid"]
)
def test_file_privilege_cannot_be_cleared_by_a_failed_mutation(
    sandbox, monkeypatch, tool, directory, privilege
):
    """Guards S2-CONF-5 at bd223d11: unavailable families cannot clear privilege."""
    from benchflow.sandbox import lockdown

    folder = sandbox.root / "system" / directory
    folder.mkdir(parents=True)
    candidate = folder / tool
    # Models an ordinary nft on a kernel with ip/ip6 but no inet support.
    # No real firewall or metadata server is touched.
    candidate.write_text(
        '#!/bin/sh\ncase "$*" in "add table inet "*) exit 1;; '
        '"add table ip "*|"add table ip6 "*|"flush ruleset") exit 0;; esac\nexit 1\n'
    )
    candidate.chmod({"setuid": 0o4755, "setgid": 0o2755}.get(privilege, 0o755))
    monkeypatch.setattr(
        lockdown,
        "_CLOUD_METADATA_ROOT_PATH",
        f"{folder}:{sandbox.bin}:{sandbox._system_bin()}",
    )
    # The first PATH match is harmless; the capable system candidate still counts.
    sandbox.script(tool, "#!/bin/sh\nexit 1\n").caps("0" * 16)
    report = _agent(
        sandbox,
        connect_probe=False,
        FILE_CAPS=privilege if privilege.startswith("cap_") else "",
    )
    assert tool in report["power"].split(",")


@pytest.mark.parametrize(
    "tool",
    [
        "iptables",
        "iptables-nft",
        "iptables-legacy",
        "ip6tables",
        "ip6tables-nft",
        "ip6tables-legacy",
        "nft",
    ],
)
@pytest.mark.parametrize(
    "placement", ["absolute", "relative", "empty", "symlink", "spaces"]
)
@pytest.mark.parametrize(
    "capabilities",
    [
        "",
        "cap_net_admin=ep",
        "cap_net_admin=i",
        "cap_setuid=ep",
        "cap_net_bind_service=ep",
    ],
)
def test_file_privilege_inspects_resolved_candidates_on_every_path(
    sandbox, tool, placement, capabilities
):
    """Guards S2-CONF-5 at bd223d11: inspect every PATH entry and link target."""
    folder = sandbox.root / ("folder with spaces" if placement == "spaces" else "extra")
    folder.mkdir()
    candidate = folder / tool
    if placement == "symlink":
        target = sandbox.root / "actual-tool"
        target.write_text("#!/bin/sh\nexit 1\n")
        target.chmod(0o755)
        candidate.symlink_to(target)
    else:
        candidate.write_text("#!/bin/sh\nexit 1\n")
        candidate.chmod(0o755)
    sandbox.caps("0" * 16)
    path = str(folder)
    if placement in ("relative", "empty"):
        # Use an isolated shell cwd; never change the test runner's cwd.
        path = os.path.relpath(folder) if placement == "relative" else ""
    from benchflow.sandbox.lockdown import _cloud_metadata_agent_check_cmd

    command = _cloud_metadata_agent_check_cmd(
        "BFMDtest", proc=str(sandbox.proc), connect_probe=False
    )
    if placement == "empty":
        import shlex

        command = f"cd {shlex.quote(str(folder))} && {command}"
    out = sandbox.run(
        command,
        PATH=f"{sandbox.bin}:{path}:{sandbox._system_bin()}",
        FILE_CAPS=capabilities,
    )
    assert f"power={tool if capabilities else ''} connect=skipped" in out


@pytest.mark.parametrize(
    "inspection",
    ["missing", "writable", "failed", "empty", "malformed", "link-reader-failed"],
)
def test_incomplete_file_privilege_inspection_refuses_a_clean_report(
    sandbox, inspection
):
    """Guards S2-CONF-5 at bd223d11: missing inspection is not no privilege."""
    sandbox.script("nft", "#!/bin/sh\nexit 1\n").caps("0" * 16)
    if inspection == "missing":
        (sandbox.bin / "getcap").unlink()
    elif inspection == "writable":
        (sandbox.bin / "getcap").chmod(0o777)
    elif inspection == "link-reader-failed":
        sandbox.script("readlink", "#!/bin/sh\nexit 1\n")
    with pytest.raises(subprocess.CalledProcessError):
        _agent(sandbox, connect_probe=False, CAP_READ=inspection)


@pytest.mark.parametrize("helper", ["getcap", "readlink", "ls"])
def test_file_privilege_inspection_never_uses_agent_path_helpers(sandbox, helper):
    """Guards S2-CONF-5 at bd223d11: agent helpers cannot clear file privilege."""
    from benchflow.sandbox.lockdown import _cloud_metadata_agent_check_cmd

    folder = sandbox.root / "agent-bin"
    folder.mkdir()
    fake = folder / helper
    fake.write_text('#!/bin/sh\necho used > "$HELPER_MARKER"\nexit 0\n')
    fake.chmod(0o755)
    marker = sandbox.root / "helper-marker"
    sandbox.script("nft", "#!/bin/sh\nexit 1\n").caps("0" * 16)
    out = sandbox.run(
        _cloud_metadata_agent_check_cmd(
            "BFMDtest", proc=str(sandbox.proc), connect_probe=False
        ),
        PATH=f"{folder}:{sandbox.bin}:{sandbox._system_bin()}",
        HELPER_MARKER=str(marker),
        FILE_CAPS="cap_net_admin=ep",
    )
    assert "power=nft connect=skipped" in out
    assert not marker.exists()


@pytest.mark.parametrize(
    "tool",
    [
        "iptables",
        "iptables-nft",
        "iptables-legacy",
        "ip6tables",
        "ip6tables-nft",
        "ip6tables-legacy",
        "nft",
    ],
)
@pytest.mark.parametrize(
    "placement",
    [
        "usr/local/sbin",
        "usr/local/bin",
        "usr/sbin",
        "usr/bin",
        "sbin",
        "bin",
        "absolute",
        "relative",
        "empty",
        "symlink",
        "spaces",
        "shadowed",
    ],
)
@pytest.mark.parametrize("shebang", ["#!/bin/sh", "#!/usr/bin/env sh"])
@pytest.mark.parametrize("shell", SHELLS)
@pytest.mark.parametrize("can_mutate", [True, False])
def test_delegated_firewall_uses_agent_path_without_changing_inspection(
    sandbox, monkeypatch, tool, placement, shebang, shell, can_mutate
):
    """Guards S2-CONF-6 at dae01349: wrappers must keep their agent PATH."""
    from benchflow.sandbox import lockdown

    folder = sandbox.root / ("folder with spaces" if placement == "spaces" else "extra")
    folder.mkdir()
    candidate_dir = folder
    if placement in (
        "usr/local/sbin",
        "usr/local/bin",
        "usr/sbin",
        "usr/bin",
        "sbin",
        "bin",
    ):
        candidate_dir = sandbox.root / "system" / placement
        candidate_dir.mkdir(parents=True)
        monkeypatch.setattr(
            lockdown,
            "_CLOUD_METADATA_ROOT_PATH",
            f"{candidate_dir}:{sandbox.bin}:{sandbox._system_bin()}",
        )
    candidate = candidate_dir / tool
    if placement == "symlink":
        target = sandbox.root / "wrapper-target"
        candidate.symlink_to(target)
    else:
        target = candidate
    target.write_text(shebang + '\nexec firewall.real "$@"\n')
    target.chmod(0o755)
    delegate = folder / "firewall.real"
    # Only dummy markers: no firewall, capabilities or metadata connection.
    delegate.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$PATH" >> "$DELEGATE_PATHS"\n'
        'printf "%s\\n" "$*" >> "$DELEGATE_CALLS"\n'
        'case "$*" in\n'
        '  "add table inet BFMDtest"|"-w -N BFMDtest")\n'
        '    [ "$CAN_MUTATE" = yes ] || exit 1\n'
        '    printf created > "$DELEGATE_STATE";;\n'
        '  "delete table inet BFMDtest"|"-w -X BFMDtest")\n'
        '    printf cleaned > "$DELEGATE_STATE";;\n'
        "  *) exit 1;;\nesac\n"
    )
    delegate.chmod(0o755)
    # Agent PATH helpers must never supply inspection or cleanup facts,
    # including after a successful candidate invocation restores its PATH.
    for helper in ("getcap", "readlink", "ls", "rm", "mktemp", "timeout"):
        fake = folder / helper
        fake.write_text('#!/bin/sh\nprintf used > "$HELPER_MARKER"\nexit 0\n')
        fake.chmod(0o755)
    if placement == "shadowed":
        sandbox.script(tool, "#!/bin/sh\nexit 1\n")
    sandbox.caps("0" * 16)
    path = str(folder)
    if placement == "relative":
        path = os.path.relpath(folder)
    elif placement == "empty":
        path = ""
    path = f"{path}:{sandbox.bin}:{sandbox._system_bin()}"
    if placement == "shadowed":
        path = f"{sandbox.bin}:{path}"
    command = lockdown._cloud_metadata_agent_check_cmd(
        "BFMDtest", proc=str(sandbox.proc), connect_probe=False
    )
    if placement == "empty":
        command = f"cd {shlex.quote(str(folder))} && {command}"
    marker = sandbox.root / "helper-marker"
    state = sandbox.root / "delegate-state"
    calls = sandbox.root / "delegate-calls"
    paths = sandbox.root / "delegate-paths"
    out = sandbox.run(
        command,
        shell=shell,
        PATH=path,
        CAN_MUTATE="yes" if can_mutate else "no",
        HELPER_MARKER=str(marker),
        DELEGATE_STATE=str(state),
        DELEGATE_CALLS=str(calls),
        DELEGATE_PATHS=str(paths),
    )
    assert f"held=0 power={tool if can_mutate else ''} connect=skipped" in out
    assert not marker.exists(), "agent PATH escaped into inspection/cleanup"
    assert paths.exists(), "the wrapper never reached its delegated executable"
    assert set(paths.read_text().splitlines()) == {path}
    create, cleanup = (
        ("add table inet BFMDtest", "delete table inet BFMDtest")
        if tool == "nft"
        else ("-w -N BFMDtest", "-w -X BFMDtest")
    )
    assert set(calls.read_text().splitlines()) == (
        {create, cleanup} if can_mutate else {create}
    )
    if can_mutate:
        assert state.read_text() == "cleaned"
    else:
        assert not state.exists()


@pytest.mark.parametrize(
    "broken", ["missing-file", "missing-set", "invalid-hex", "empty", "duplicate"]
)
def test_incomplete_capability_read_cannot_report_a_clean_audit(sandbox, broken):
    """Guards S2-CONF-3B at 84b4ab01: unread capabilities are not zero."""
    from benchflow.sandbox.lockdown import _cloud_metadata_agent_check_cmd

    sandbox.caps("0" * 16)
    status = sandbox.proc / "self" / "status"
    if broken == "missing-file":
        status.unlink()
    elif broken == "missing-set":
        status.write_text("CapEff:\t0000000000000000\n")
    elif broken == "invalid-hex":
        status.write_text(
            status.read_text().replace(
                "CapEff:\t0000000000000000", "CapEff:\tbad-value"
            )
        )
    elif broken == "empty":
        status.write_text("")
    else:
        status.write_text(status.read_text() + "CapEff:\t0000000000000000\n")
    with pytest.raises(subprocess.CalledProcessError):
        sandbox.run(
            _cloud_metadata_agent_check_cmd(
                "BFMDtest", proc=str(sandbox.proc), connect_probe=False
            )
        )


@pytest.mark.parametrize("tool", ["ip6tables", "ip6tables-nft", "ip6tables-legacy"])
@pytest.mark.parametrize("available", ["missing", "broken", "working"])
def test_missing_ipv6_proc_entries_never_exempt_ipv6_rules(sandbox, tool, available):
    """Guards S2-CONF-4 at 84b4ab01: IPv6 may appear after this snapshot."""
    sandbox.tools("iptables").ipv6("absent")
    if available != "missing":
        sandbox.tools(tool)
    report = sandbox.block(FW_BROKEN=tool if available == "broken" else "")
    if available == "working":
        assert report["status"] == "0"
        assert sandbox.rules(tool) == list(reversed([_rule(a) for a in V6]))
    else:
        assert report["status"] == "3"


# --- block_cloud_metadata: the record, and refusing an agent with the power ----


class FakeEnv:
    """Answers root's script, the agent check and the counter read in turn."""

    def __init__(self, *outputs: str) -> None:
        self.outputs = list(outputs)
        self.calls: list[tuple[str, str | None]] = []

    @staticmethod
    def agent_command(command: str, user: str | None) -> tuple[str, str | None]:
        """Model the carrier's uid transition; its real flags have own tests."""
        from benchflow.sandbox._root_launch import save_task_env

        command = inner_command(command)
        if user == "root" and command.startswith(("if setpriv", save_task_env())):
            # The launch keeps the task's values and pins PATH first (C8's
            # AgentLaunch), then drops to ``bash --noprofile --norc -c``,
            # whose text ends by exec'ing the agent's own command.
            words = shlex.split(command[command.index("if setpriv") :])
            inner = words[words.index("--norc") + 2]
            parts = shlex.split(inner)
            for i, word in enumerate(parts):
                if word == "exec" and parts[i + 1 : i + 3] == ["/bin/bash", "-c"]:
                    return parts[i + 3], "agent"
            raise AssertionError(f"no agent command in {inner!r}")
        return command, user

    async def exec(self, command: str, user: str | None = None, **_: object):
        # Lockdown reads these through a receipt: answer the command it
        # wraps, then print the receipt for that answer.
        return receipt_answer(command, self.answer(command, user))

    def answer(self, command: str, user: str | None):
        command = inner_command(command)
        if "BF metadata-construction" in command:
            assert user == "root"
            return SimpleNamespace(
                stdout="BF metadata-construction uid=1000 nnp=1 "
                "bnd=0000000000000000 amb=0000000000000000\n",
                return_code=0,
            )
        if "BF delegate-python" in command:
            # Root's look for BenchFlow's own Python: none here, so
            # the delegate inventory runs as the shell (find and getcap).
            assert user == "root"
            return SimpleNamespace(stdout=NO_DELEGATE_PYTHON, stderr="", return_code=0)
        command, user = self.agent_command(command, user)
        self.calls.append((command, user))
        if "BF metadata-connect" in command:
            return SimpleNamespace(
                stdout="BF metadata-connect failed\n", stderr="", return_code=0
            )
        if "BF metadata-delegates" in command:
            return SimpleNamespace(stdout=DELEGATES_CLEAN, stderr="", return_code=0)
        out = self.outputs.pop(0) if self.outputs else ""
        return SimpleNamespace(stdout=out, stderr="", return_code=0)


# Root's look for BenchFlow's own Python finding none, on a machine with no
# build of it (lockdown._cloud_metadata_delegate_audit, _delegate_disarm).
NO_DELEGATE_PYTHON = (
    "BF delegate-python managed=- staged=- arch=other libc=none tar=0\n"
)
# The delegate inventory's report for an image with nothing that counts,
# read with getcap (lockdown._cloud_metadata_delegate_check_cmd).
DELEGATES_CLEAN = (
    "BF metadata-delegates-scan reader=getcap complete=1 files=- vanished=- "
    "errors=- why=-\n"
)
ROOT_OK = (
    "BF metadata status=0 fw4=iptables fw6=ip6tables v6=local install=none "
    "install_t=- hits=0\n"
)
AGENT_REFUSED = "BF metadata-agent held=0 power= connect=refused\n"


async def _block(*outputs: str) -> tuple[dict, FakeEnv]:
    from benchflow.sandbox.lockdown import block_cloud_metadata

    env = FakeEnv(*outputs)
    return await block_cloud_metadata(env, "agent"), env


@pytest.mark.asyncio
async def test_on_needs_the_agents_refused_connect_counted_by_the_rule():
    record, env = await _block(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 2\n")

    assert record["state"] == "on", record
    assert record["defense"] == "cloud_metadata"
    assert record["addresses"] == list(V4 + V6)
    assert "refused and the rule counted 2 packets" in record["why"]
    assert (record["iptables"], record["ip6tables"]) == ("iptables", "ip6tables")
    assert [user for _, user in env.calls] == ["root", "agent", "root", "root"]


@pytest.mark.asyncio
async def test_a_refusal_the_rule_did_not_count_is_unproven():
    """Something else refused it (a host's own block): not this rule's proof."""
    record, _ = await _block(
        ROOT_OK.replace("hits=0", "hits=5"), AGENT_REFUSED, "BF metadata-hits 5\n"
    )

    assert record["state"] == "unproven"
    assert "its packet counter did not rise" in record["why"]


@pytest.mark.asyncio
@pytest.mark.parametrize("connect", ["reached", "timeout", "skipped"])
async def test_a_connect_that_was_not_refused_is_unproven(connect):
    record, _ = await _block(
        ROOT_OK, AGENT_REFUSED.replace("refused", connect), "BF metadata-hits 0\n"
    )

    assert record["state"] == "unproven", record


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "words"),
    [
        ("1", "could not be added"),
        ("2", "no package manager"),
        ("4", "no NET_ADMIN"),
        ("5", "installing iptables failed (E: no mirror)"),
        ("6", "a folder or link a non-root user can change"),
    ],
)
async def test_a_block_that_could_not_be_installed_says_why_and_the_trial_goes_on(
    status, words
):
    root = (
        f"BF metadata status={status} fw4=- fw6=- v6=local install=failed "
        "install_t=10.00,22.50 hits=0\nBF metadata-install: E: no mirror\n"
    )
    record, env = await _block(root, AGENT_REFUSED.replace("refused", "skipped"))

    assert record["state"] == "not_installed"
    assert words in record["why"], record["why"]
    assert record["why"].endswith(
        ", 168.63.129.16, 169.254.0.23, fd00:ec2::254, "
        "fd00:ec2::23, fd20:ce::254, fd00:c1::a9fe:a9fe"
    )
    assert record["install_seconds"] == 12.5
    assert [user for _, user in env.calls] == ["root", "agent", "root", "root"]


@pytest.mark.asyncio
async def test_ipv6_out_without_ip6tables_is_not_installed_for_ipv6():
    record, _ = await _block(
        ROOT_OK.replace("status=0", "status=3").replace("fw6=ip6tables", "fw6=-"),
        AGENT_REFUSED,
        "BF metadata-hits 1\n",
    )

    assert record["state"] == "not_installed"
    assert "fd00:ec2::254" in record["why"]


@pytest.mark.asyncio
async def test_an_installed_iptables_and_its_seconds_are_recorded():
    record, _ = await _block(
        ROOT_OK.replace("install=none install_t=-", "install=ok install_t=5.10,17.35"),
        AGENT_REFUSED,
        "BF metadata-hits 1\n",
    )

    assert record["state"] == "on"
    assert record["iptables_installed"] is True
    assert record["install_seconds"] == 12.25


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "agent",
    [
        "BF metadata-agent held=1 power= connect=refused",
        "BF metadata-agent held=0 power=iptables,ip6tables connect=refused",
    ],
    ids=["holds-net-admin", "firewall-tool"],
)
async def test_an_agent_user_that_can_change_the_firewall_stops_the_trial(agent):
    """The agent user's processes never hold NET_ADMIN: BenchFlow gives main
    NET_ADMIN under an open network only for these rules, so an image that
    hands it on to the agent user is refused before the agent starts."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError

    with pytest.raises(CloudMetadataPowerError, match=r"^sandbox startup refused"):
        await _block(ROOT_OK, agent, "BF metadata-hits 1\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["1", "2", "3", "4", "5", "6", "missing", "timeout"])
@pytest.mark.parametrize("power", ["held=1 power=", "held=0 power=iptables"])
async def test_failed_install_still_refuses_agent_firewall_power(status, power):
    """Guards S2-CONF-3 at a070066d, including every installation failure exit."""
    from benchflow.sandbox.lockdown import (
        CloudMetadataPowerError,
        block_cloud_metadata,
    )

    root = (
        "" if status == "missing" else ROOT_OK.replace("status=0", f"status={status}")
    )
    outputs = (
        root,
        f"BF metadata-agent {power} connect=skipped\n",
        "BF metadata-hits 0\n",
    )
    env = RaisingEnv(0, *outputs[1:]) if status == "timeout" else FakeEnv(*outputs)
    with pytest.raises(CloudMetadataPowerError, match=r"^sandbox startup refused"):
        await block_cloud_metadata(env, "agent")
    assert [user for _, user in env.calls] == ["root", "agent", "root"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", ["0", "1", "2", "3", "4", "5", "6", "missing", "timeout"]
)
@pytest.mark.parametrize(
    "broken",
    [
        "timeout",
        "nonzero",
        "missing-exit",
        "boolean-exit",
        "string-exit",
        "float-exit",
        "empty",
        "missing-field",
        "invalid-held",
        "invalid-connect",
        "duplicate-field",
        "duplicate-report",
        "unknown-field",
    ],
)
async def test_incomplete_agent_audit_always_refuses_startup(status, broken):
    """Guards batch 8 at 70400f55: missing optional tools cannot refuse startup."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    root = (
        "" if status == "missing" else ROOT_OK.replace("status=0", f"status={status}")
    )
    reports = {
        "empty": "",
        "missing-field": "BF metadata-agent held=0 connect=refused\n",
        "invalid-held": AGENT_REFUSED.replace("held=0", "held=unknown"),
        "invalid-connect": AGENT_REFUSED.replace("refused", "unknown"),
        "duplicate-field": AGENT_REFUSED.replace("held=0", "held=1 held=0"),
        "duplicate-report": AGENT_REFUSED * 2,
        "unknown-field": AGENT_REFUSED.strip() + " ignored=1\n",
    }

    class AuditEnv(FakeEnv):
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-construction" in command:
                return await super().exec(command, user=user, **kwargs)
            if "BF metadata-connect" in command:
                return receipt_answer(
                    command,
                    SimpleNamespace(
                        stdout="BF metadata-connect refused\n", return_code=0
                    ),
                )
            call = len(self.calls)
            result = await super().exec(command, user=user, **kwargs)
            if (call == 0 and status == "timeout") or (
                call == 1 and broken == "timeout"
            ):
                raise TimeoutError("dummy exec timeout")
            if call == 1 and broken == "nonzero":
                result.return_code = 124
            if call == 1 and broken == "missing-exit":
                del result.return_code
            if call == 1 and broken in ("boolean-exit", "string-exit", "float-exit"):
                result.return_code = {
                    "boolean-exit": False,
                    "string-exit": "0",
                    "float-exit": 0.0,
                }[broken]
            return result

    env = AuditEnv(root, reports.get(broken, AGENT_REFUSED), "BF metadata-hits 0\n")
    record = await block_cloud_metadata(env, "agent")
    assert record["agent_privilege_audit"] == "unavailable"


@pytest.mark.asyncio
async def test_writable_firewall_tool_does_not_bypass_power_refusal(sandbox):
    """Guards S2-CONF-3 at a070066d with the real scripts and dummy firewall."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError

    sandbox.tools("iptables").ipv6("absent")
    (sandbox.bin / "iptables").chmod(0o775)
    root = sandbox.block()["_out"]
    assert "status=6" in root
    agent = _agent(sandbox.caps("0" * 16), FW_AGENT_POWER="iptables")
    assert agent["power"] == "iptables"
    report = "BF metadata-agent " + " ".join(f"{k}={v}" for k, v in agent.items())
    with pytest.raises(CloudMetadataPowerError, match=r"^sandbox startup refused"):
        await _block(root, report, "BF metadata-hits 0\n")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool",
    [
        "iptables",
        "iptables-nft",
        "iptables-legacy",
        "ip6tables",
        "ip6tables-nft",
        "ip6tables-legacy",
        "nft",
    ],
)
@pytest.mark.parametrize(
    "status", ["0", "1", "2", "3", "4", "5", "6", "missing", "timeout"]
)
@pytest.mark.parametrize("supports_system_shell", [True, False])
async def test_delegated_firewall_refuses_every_install_result_and_reconnect(
    sandbox, tool, status, supports_system_shell
):
    """Guards S2-CONF-6 at dae01349 through both backend exec contracts."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    folder = sandbox.root / "delegates"
    folder.mkdir()
    delegate = folder / "firewall.real"
    delegate.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DELEGATE_CALLS"\n')
    delegate.chmod(0o755)
    sandbox.script(tool, '#!/bin/sh\nexec firewall.real "$@"\n')
    # The connect remains a dummy refused marker; no live endpoint is used.
    sandbox.script("bash", "#!/bin/sh\nexit 0\n").script("timeout", FAKE_TIMEOUT)

    class DelegatedEnv(FakeEnv):
        async def exec(self, command, user=None, **kwargs):
            inner, as_user = self.agent_command(command, user)
            result = await super().exec(command, user=user, **kwargs)
            if len(self.calls) == 1 and status == "timeout":
                raise TimeoutError("dummy root exec timeout")
            if as_user == "agent":
                result = receipt_answer(
                    command,
                    SimpleNamespace(
                        stdout=sandbox.run(
                            inner,
                            PATH=f"{folder}:{sandbox.bin}:{sandbox._system_bin()}",
                            DELEGATE_CALLS=str(sandbox.root / "delegate-calls"),
                            CONNECT_LOG=str(sandbox.root / "connect.log"),
                        ),
                        stderr="",
                        return_code=0,
                    ),
                )
            else:
                assert kwargs.get("system_shell", False) is supports_system_shell
            return result

    root = (
        "" if status == "missing" else ROOT_OK.replace("status=0", f"status={status}")
    )
    for _ in range(2):  # Initial connection and reconnect use the same refusal.
        env = DelegatedEnv(root, "", "BF metadata-hits 1\n")
        env.supports_system_shell = supports_system_shell
        with pytest.raises(CloudMetadataPowerError, match="can change the firewall"):
            await block_cloud_metadata(env, "agent")
        assert [user for _, user in env.calls] == ["root", "agent", "root"]
    assert sandbox.calls("delegate-calls")


@pytest.mark.asyncio
async def test_the_refusal_is_a_sandbox_setup_failure_a_retry_would_repeat():
    from benchflow._utils.scoring import (
        SANDBOX_SETUP,
        classify_error,
        is_unrecoverable_startup_error,
    )
    from benchflow.sandbox.lockdown import CloudMetadataPowerError

    with pytest.raises(CloudMetadataPowerError) as refused:
        await _block(
            ROOT_OK,
            "BF metadata-agent held=0 power=iptables connect=refused",
            "BF metadata-hits 1\n",
        )
    assert classify_error(str(refused.value)) == SANDBOX_SETUP
    assert is_unrecoverable_startup_error(str(refused.value))


@pytest.mark.asyncio
async def test_a_tool_name_root_reports_is_never_run_unless_it_is_known():
    _, env = await _block(
        ROOT_OK.replace("fw4=iptables", "fw4=/tmp/evil"),
        AGENT_REFUSED,
        "BF metadata-hits 1\n",
    )

    assert "n4=iptables;" in env.calls[2][0]
    assert "/tmp/evil" not in env.calls[2][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("supports", [True, False])
async def test_root_starts_the_systems_shell_where_the_sandbox_can(supports):
    # Where a sandbox can start /bin/sh by path (exec's system_shell), root's
    # scripts never take their shell from the image's PATH, which the agent
    # may have changed by a later scene.
    from benchflow.sandbox.lockdown import block_cloud_metadata

    seen: list[tuple[str | None, object]] = []

    class Env(FakeEnv):
        supports_system_shell = supports

        async def exec(self, command: str, user: str | None = None, **kw: object):
            seen.append((user, kw.get("system_shell")))
            return await super().exec(command, user=user)

    await block_cloud_metadata(
        Env(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n"), "agent"
    )

    root = True if supports else None
    # The construction check, root's rules, the agent's check (through
    # root's launch), the counter, root's look for BenchFlow's own Python
    # and the delegate inventory.
    assert seen == [("root", root)] * 6


# --- the plan: when the block is armed ----------------------------------------


def _config(text: str):
    from benchflow.task.config import TaskConfig

    return TaskConfig.model_validate_toml(text)


OPEN = 'version = "1.0"\n'


@pytest.mark.parametrize(
    ("toml", "user", "launches", "expect"),
    [
        (OPEN, "agent", True, "armed"),
        (OPEN + "[metadata]\nx-allow-cloud-metadata = true\n", "agent", True, "off"),
        (OPEN + "[environment]\nallow_cloud_metadata = true\n", "agent", True, "off"),
        (OPEN + "[environment]\nallow_cloud_metadata = true\n", None, True, "off"),
        (OPEN, None, True, "root"),
        (OPEN, "root", True, "root"),
        (OPEN, "agent", False, None),
        (OPEN + '[environment]\nnetwork_mode = "no-network"\n', "agent", True, None),
        (
            OPEN + '[environment]\nnetwork_mode = "denylist"\n'
            'blocked_hosts = ["example.com"]\n',
            "agent",
            True,
            None,
        ),
        (
            OPEN
            + '[agent]\nnetwork_mode = "allowlist"\nallowed_hosts = ["example.com"]\n',
            "agent",
            True,
            None,
        ),
    ],
    ids=[
        "open",
        "opt-out",
        "environment-opt-out",
        "root-opt-out",
        "root-agent",
        "root-named",
        "oracle",
        "no-network",
        "denylist",
        "agent-allowlist",
    ],
)
def test_the_block_is_armed_only_for_an_open_networks_agent_user(
    toml, user, launches, expect
):
    """An agent that runs as root is held too (``root``); only an
    explicit opt-out leaves the block off."""
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    plan = cloud_metadata_plan(
        _config(toml), sandbox_user=user, launches_agent=launches
    )
    if expect is None:
        assert plan is None
    elif expect == "off":
        assert plan is not None and not plan.armed
        assert plan.opt_out and plan.opt_out in plan.why
    else:
        assert plan is not None and plan.armed
        assert plan.root_agent is (expect == "root")


@pytest.mark.parametrize(
    ("toml", "run_opt_out", "named"),
    [
        (
            OPEN + "[metadata]\nx-allow-cloud-metadata = true\n",
            False,
            "metadata.x-allow-cloud-metadata = true",
        ),
        (
            OPEN + "[environment]\nallow_cloud_metadata = true\n",
            False,
            "allow_cloud_metadata = true in the task's environment",
        ),
        (OPEN, True, "--allow-cloud-metadata"),
    ],
    ids=["metadata-key", "environment-key", "run-flag"],
)
def test_the_opt_out_record_names_the_key(toml, run_opt_out, named):
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    plan = cloud_metadata_plan(
        _config(toml),
        sandbox_user="agent",
        launches_agent=True,
        allow_cloud_metadata=run_opt_out,
    )
    assert plan is not None and not plan.armed
    assert plan.opt_out == named
    assert plan.why == f"the run opts out ({named})"


# --- the opt-out key -----------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "allowed"),
    [("true", True), ("false", False), ('"yes"', True), ('"off"', False), ("1", True)],
)
def test_the_opt_out_reads_a_boolean_as_the_rust_bench_does(value, allowed):
    config = _config(OPEN + f"[metadata]\nx-allow-cloud-metadata = {value}\n")
    assert config.allows_cloud_metadata is allowed


def test_no_opt_out_means_the_endpoints_are_refused():
    assert _config(OPEN).allows_cloud_metadata is False


@pytest.mark.parametrize("value", ['"maybe"', "2", "[true]"])
def test_an_opt_out_that_is_not_a_boolean_is_refused(value):
    with pytest.raises(
        ValueError, match="x-allow-cloud-metadata must be true or false"
    ):
        _config(OPEN + f"[metadata]\nx-allow-cloud-metadata = {value}\n")


def test_the_opt_out_is_read_from_task_md_metadata(tmp_path):
    from benchflow.task import Task

    task = tmp_path / "t"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task / "tests").mkdir()
    (task / "tests" / "test.sh").write_text("#!/bin/sh\n")
    (task / "task.md").write_text(
        "---\nschema_version: '1.0'\nmetadata:\n  x-allow-cloud-metadata: true\n"
        "---\n\n## prompt\n\nDo it.\n"
    )
    assert Task(task).config.allows_cloud_metadata is True


# --- NET_ADMIN for main under an open network ------------------------------------


MAIN_ONLY = "services:\n  main: {}\n"


def _task_dir(root: Path, toml: str = OPEN, compose: str | None = MAIN_ONLY) -> Path:
    """A task folder: task.toml, a Dockerfile and (unless None) a compose file."""
    task_dir = root / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Do it.\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\n")
    (task_dir / "task.toml").write_text(
        toml + "[agent]\ntimeout_sec = 1\n[verifier]\ntimeout_sec = 1\n"
    )
    environment = task_dir / "environment"
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    if compose is not None:
        (environment / "docker-compose.yaml").write_text(compose)
    return task_dir


def _sandbox_for(
    tmp_path: Path,
    toml: str,
    *,
    block: bool,
    backend: str = "docker",
    compose: str = MAIN_ONLY,
):
    from benchflow.sandbox.setup import _create_sandbox_environment
    from benchflow.task import RolloutPaths, Task

    task_dir = _task_dir(tmp_path / backend, toml, compose)
    with (
        patch("benchflow.sandbox.daytona._load_daytona_sdk"),
        patch("benchflow.sandbox._sdk_ops.apply"),
        # This constructor matrix inspects metadata capability selection.
        # Compose admission is covered by the launch-policy tests.
        patch("benchflow.sandbox.setup._freeze"),
        # Batch 23: this fake constructor matrix inspects capability selection;
        # real Daytona Compose admission is checked independently.
        patch("benchflow.sandbox.setup._preflight_daytona_compose"),
    ):
        return _create_sandbox_environment(
            backend,
            Task(task_dir),
            task_dir,
            "rollout",
            RolloutPaths(rollout_dir=tmp_path / backend / "rollout"),
            block_cloud_metadata=block,
        )


@pytest.mark.parametrize(
    ("toml", "block", "net_admin"),
    [
        (OPEN, True, True),
        (OPEN, False, False),
        (OPEN + "[metadata]\nx-allow-cloud-metadata = true\n", True, False),
        (OPEN + '[environment]\nnetwork_mode = "no-network"\n', True, False),
    ],
    ids=["armed", "not-armed", "opt-out", "no-network-oracle"],
)
def test_main_gets_net_admin_under_open_only_while_the_block_is_armed(
    tmp_path, toml, block, net_admin
):
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        compose_needs_net_admin,
    )
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    docker = _sandbox_for(tmp_path, toml, block=block)
    assert compose_needs_net_admin(docker.task_env_config) is net_admin
    assert (COMPOSE_NET_ADMIN_PATH in docker._docker_compose_paths) is net_admin
    daytona = _sandbox_for(tmp_path, toml, block=block, backend="daytona")
    assert isinstance(daytona._strategy, _DaytonaDinD)
    flags = daytona._strategy._compose_file_flags()
    dind = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_NET_ADMIN_PATH.name}"
    assert (dind in flags) is net_admin


def test_arming_never_changes_the_tasks_own_config(tmp_path):
    docker = _sandbox_for(tmp_path, OPEN, block=True)
    assert docker.task_env_config.cloud_metadata_block is True
    from benchflow.task import Task

    assert (
        Task(tmp_path / "docker" / "task").config.sandbox.cloud_metadata_block is False
    )


# --- the rollout: before the agent starts, and the record -------------------------


def _rollout(tmp_path: Path, plan):
    from benchflow.rollout import Role, Rollout, RolloutConfig, Scene

    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        scenes=[Scene(roles=[Role(name="agent", agent="claude-agent-acp", model="m")])],
    )
    trial = Rollout(cfg)
    trial._env = SimpleNamespace()
    trial._rollout_dir = tmp_path
    trial._phase = "idle"
    trial._cloud_metadata = plan
    return trial


@pytest.mark.asyncio
async def test_the_block_runs_before_the_agent_starts_and_is_recorded(tmp_path):
    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        cloud_metadata_record,
    )

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True))
    order: list[str] = []

    async def block(env, user):
        order.append(f"block {user}")
        return cloud_metadata_record("on", "proof")

    async def connect(**_):
        order.append("agent")
        return (MagicMock(), MagicMock(), MagicMock(), "agent")

    async def firewall_power(env, user, **_):
        # PY8 S3's check (tests/test_firewall_power.py) runs first.
        order.append(f"firewall power {user}")
        return {"defense": "firewall_power", "state": "disarmed", "why": ""}, None

    with (
        patch("benchflow.rollout.check_firewall_power", side_effect=firewall_power),
        patch.object(trial._planes, "block_cloud_metadata", side_effect=block),
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

    assert order == ["firewall power agent", "block agent", "agent"]
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "on"
    assert "cloud_metadata_block" in trial._timing


@pytest.mark.asyncio
async def test_an_opted_out_trial_records_off_and_installs_nothing(tmp_path):
    from benchflow.sandbox.cloud_metadata import CloudMetadataPlan

    trial = _rollout(
        tmp_path,
        CloudMetadataPlan(
            armed=False, why="the run opts out (k = true)", opt_out="k = true"
        ),
    )
    block = AsyncMock()
    with patch.object(trial._planes, "block_cloud_metadata", block):
        await trial._apply_cloud_metadata_block(tmp_path)

    block.assert_not_awaited()
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "off"
    assert record["why"].startswith("the run opts out (k = true)")
    assert record["opt_out"] == "k = true"
    # result.json's cloud_metadata.
    assert trial._cloud_metadata_result == {"state": "off", "opt_out": "k = true"}


def _connect_patches(trial, order, block):
    """What trial.connect() needs around the block, recording the order."""

    async def connect(**_):
        order.append("agent")
        return (MagicMock(), MagicMock(), MagicMock(), "agent")

    async def firewall_power(env, user, **_):
        order.append(f"firewall power {user}")
        return {"defense": "firewall_power", "state": "disarmed", "why": ""}, None

    return (
        patch("benchflow.rollout.check_firewall_power", side_effect=firewall_power),
        patch.object(trial._planes, "block_cloud_metadata", side_effect=block),
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
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["not_installed", "unproven"])
async def test_a_block_that_is_not_on_stops_the_trial_before_the_agent(tmp_path, state):
    """GAP-METADATA-FAILOPEN: the agent never starts with the
    endpoints open to it unless the run opted out. The trial stops as a
    sandbox-setup error that says why and how to opt out."""
    import contextlib

    from benchflow._utils.scoring import SANDBOX_SETUP, classify_error
    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        CloudMetadataUnprotectedError,
        cloud_metadata_record,
    )

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True))
    order: list[str] = []

    async def block(env, user):
        order.append(f"block {user}")
        return cloud_metadata_record(state, "the rules could not go in")

    with contextlib.ExitStack() as stack:
        for p in _connect_patches(trial, order, block):
            stack.enter_context(p)
        with pytest.raises(CloudMetadataUnprotectedError) as refused:
            await trial.connect()

    assert order == ["firewall power agent", "block agent"]
    message = str(refused.value)
    assert message.startswith("sandbox startup refused: ")
    assert f"the block is {state}: the rules could not go in" in message
    assert "allow_cloud_metadata = true" in message
    assert "--allow-cloud-metadata" in message
    assert classify_error(message) == SANDBOX_SETUP
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == state
    assert trial._cloud_metadata_result == {"state": state}


@pytest.mark.asyncio
async def test_main_on_the_hosts_network_stops_the_trial_unless_opted_out(tmp_path):
    """The block cannot go in where main shares the host's network; that is
    no longer a trial that runs on with the endpoints open."""
    from benchflow._utils.scoring import is_unrecoverable_startup_error
    from benchflow.sandbox.cloud_metadata import (
        MAIN_SHARES_HOST_NETWORK,
        CloudMetadataPlan,
        CloudMetadataUnprotectedError,
    )

    why = MAIN_SHARES_HOST_NETWORK.format(mode="host")
    trial = _rollout(
        tmp_path, CloudMetadataPlan(armed=False, state="not_installed", why=why)
    )
    with pytest.raises(CloudMetadataUnprotectedError) as refused:
        await trial._apply_cloud_metadata_block(tmp_path)
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "not_installed"
    # The task's own file decides it: another attempt would only repeat it.
    assert is_unrecoverable_startup_error(str(refused.value))


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["on", "not_installed"])
async def test_a_root_agent_is_held_and_starts_only_on_a_proven_block(tmp_path, state):
    """An agent that runs as root gets the block too (the plan
    arms it, the block holds root); it starts only when the block is on.
    The firewall-power check is not run for root, as before."""
    import contextlib

    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        CloudMetadataUnprotectedError,
        cloud_metadata_record,
    )

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True, root_agent=True))
    trial._config.sandbox_user = None
    order: list[str] = []

    async def block(env, user):
        order.append(f"block {user}")
        return cloud_metadata_record(state, "root's connect was refused")

    with contextlib.ExitStack() as stack:
        for p in _connect_patches(trial, order, block):
            stack.enter_context(p)
        if state == "on":
            await trial.connect()
        else:
            with pytest.raises(CloudMetadataUnprotectedError):
                await trial.connect()

    assert order == (["block None", "agent"] if state == "on" else ["block None"])


# --- review round 1 -------------------------------------------------------------

HOST_MAIN = "services:\n  main:\n    network_mode: host\n"
OUTSIDE_NETWORKS = [
    "host",
    '"container:other"',
    '"service:db"',
    '"${MAIN_NET:-host}"',
]


@pytest.mark.parametrize("mode", OUTSIDE_NETWORKS)
def test_main_on_the_hosts_network_gets_no_net_admin(tmp_path, mode):
    """S2-R1-A-1: NET_ADMIN in a main that shares the host's network is power
    over the host's own firewall; the block's rules would land there and
    outlast the trial, so main gets no capability for them."""
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        compose_needs_net_admin,
    )
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    compose = f"services:\n  main:\n    network_mode: {mode}\n  db: {{}}\n"
    docker = _sandbox_for(tmp_path, OPEN, block=True, compose=compose)
    assert compose_needs_net_admin(docker.task_env_config) is False
    assert COMPOSE_NET_ADMIN_PATH not in docker._docker_compose_paths
    daytona = _sandbox_for(
        tmp_path, OPEN, block=True, backend="daytona", compose=compose
    )
    dind = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_NET_ADMIN_PATH.name}"
    assert dind not in daytona._strategy._compose_file_flags()


@pytest.mark.parametrize("mode", OUTSIDE_NETWORKS)
def test_main_on_the_hosts_network_is_not_installed_and_says_why(tmp_path, mode):
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    task_dir = _task_dir(
        tmp_path, compose=f"services:\n  main:\n    network_mode: {mode}\n"
    )
    plan = cloud_metadata_plan(
        _config(OPEN),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=task_dir / "environment",
    )
    assert plan is not None and not plan.armed
    assert plan.state == "not_installed"
    assert "main shares the host's network" in plan.why
    assert "block the endpoints at the host" in plan.why


@pytest.mark.parametrize(
    "compose",
    [
        None,
        MAIN_ONLY,
        "services:\n  main:\n    network_mode: bridge\n",
        # The embodied template's form: compose's own tag on another key.
        "services:\n  main:\n    volumes: !override\n      - rpc:/rpc:ro\n",
        "services:\n  main:\n    network_mode: !reset host\n",
        "services:\n  main:\n    networks: [inner]\nnetworks:\n  inner: {}\n",
    ],
    ids=["no-compose", "main-only", "bridge", "override-tag", "reset-mode", "own-net"],
)
def test_main_on_its_own_network_is_still_armed(tmp_path, compose):
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    task_dir = _task_dir(tmp_path, compose=compose)
    plan = cloud_metadata_plan(
        _config(OPEN),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=task_dir / "environment",
    )
    assert plan is not None and plan.armed and plan.uncovered == ()


@pytest.mark.asyncio
async def test_the_rollout_reads_mains_network_from_the_tasks_compose(
    tmp_path, monkeypatch
):
    """The rollout's own plan (not only the factory) sees the host network,
    so root's script never runs in the host's namespace even when main has
    NET_ADMIN from elsewhere (the task's own cap_add)."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox.cloud_metadata import CloudMetadataPowerError

    task_dir = _task_dir(
        tmp_path, compose=HOST_MAIN + "    cap_add:\n      - NET_ADMIN\n"
    )
    monkeypatch.chdir(tmp_path)
    trial = Rollout(
        RolloutConfig(task_path=task_dir, environment="docker", sandbox_user="agent")
    )
    trial.use_prebuilt_env(SimpleNamespace())
    await trial.setup()

    plan = trial._cloud_metadata
    assert plan is not None and not plan.armed
    assert plan.state == "not_installed"
    block = AsyncMock()
    with (
        patch.object(trial._planes, "block_cloud_metadata", block),
        # The agent does not start with the endpoints open.
        pytest.raises(CloudMetadataPowerError, match="sandbox startup refused"),
    ):
        await trial._apply_cloud_metadata_block(tmp_path)
    block.assert_not_awaited()
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "not_installed"
    assert "main shares the host's network" in record["why"]


class RaisingEnv(FakeEnv):
    """Raises, as Docker's and Daytona's exec do on a timeout, on call
    number ``fail_at`` (0 = root's script)."""

    def __init__(self, fail_at: int, *outputs: str) -> None:
        super().__init__(*outputs)
        self.fail_at = fail_at

    async def exec(self, command: str, user: str | None = None, **kw: object):
        if "BF metadata-construction" in command:
            return await super().exec(command, user=user, **kw)
        inner, as_user = self.agent_command(command, user)
        if len(self.calls) == self.fail_at:
            self.calls.append((inner, as_user))
            raise RuntimeError("Command timed out after 300 seconds")
        return await super().exec(command, user=user, **kw)


@pytest.mark.asyncio
async def test_root_script_that_times_out_is_not_installed_and_the_trial_goes_on():
    """S2-R1-A-2: the exec's timeout no longer escapes and fails the trial."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    env = RaisingEnv(0, AGENT_REFUSED.replace("refused", "skipped"))
    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "not_installed", record
    assert "did not finish in 300 s" in record["why"]
    assert "Command timed out after 300 seconds" in record["why"]
    assert record["iptables_installed"] is False
    assert [user for _, user in env.calls] == ["root", "agent", "root", "root"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fail_at", "words"),
    [(1, "the agent user's check did not finish"), (2, "reading the rule's counter")],
    ids=["agent-check", "after"],
)
async def test_a_check_that_times_out_is_unproven(fail_at, words):
    """Guards batch 8 at 70400f55: optional evidence timeouts stay unproven."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    env = RaisingEnv(fail_at, ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")
    if fail_at == 1:
        record = await block_cloud_metadata(env, "agent")
        assert record["state"] == "unproven"
        assert record["agent_privilege_audit"] == "unavailable"
        assert len(env.calls) == 5
        return
    record = await block_cloud_metadata(env, "agent")

    assert record["state"] == "unproven", record
    assert words in record["why"]
    assert len(env.calls) == 4


FAKE_SLOW_TIMEOUT = """#!/bin/sh
# Stands in for coreutils timeout: answers the probe (timeout -k 5 5 true),
# and reports every bounded call as run out of time.
echo "timeout $*" >> "$TIMEOUT_LOG"
[ "$4" = true ] && exit 0
exit 124
"""


def test_the_install_is_bounded_and_a_timeout_is_status_5(sandbox):
    sandbox.script("apt-get", FAKE_APT_GET).script("timeout", FAKE_SLOW_TIMEOUT)
    report = sandbox.block(TIMEOUT_LOG=str(sandbox.root / "timeout.log"))

    assert (report["status"], report["install"]) == ("5", "timeout"), report["_out"]
    calls = sandbox.calls("timeout.log")
    assert calls[0] == "timeout -k 5 5 true"
    # One 240 s deadline for the whole install (the stub /proc/uptime does not
    # move, so every call gets all of it).
    assert calls[1].startswith(
        "timeout -k 10 240 env DEBIAN_FRONTEND=noninteractive apt-get install"
    )
    assert sorted(p.name for p in sandbox.lists.iterdir()) == ["image-list"]


@pytest.mark.asyncio
async def test_an_install_that_ran_out_of_time_says_so():
    record, _ = await _block(
        "BF metadata status=5 fw4=- fw6=- v6=local install=timeout "
        "install_t=10.00,250.00 hits=0\nBF metadata-install: Get:1 slow\n",
        AGENT_REFUSED.replace("refused", "skipped"),
    )

    assert record["state"] == "not_installed"
    assert record["why"].startswith(
        "installing iptables did not finish in 240 s and was stopped (Get:1 slow)"
    )


def test_an_ip6tables_install_that_failed_is_not_reported_as_installed(sandbox):
    """S2-R1-B-6: IPv4 iptables was there already; the missing ip6tables was
    not installed, so the install failed."""
    sandbox.tools("iptables").ipv6("global").script("apt-get", FAKE_APT_GET)
    report = sandbox.block(APT_FAILS="1")

    assert (report["status"], report["install"]) == ("3", "failed"), report["_out"]


def test_other_compose_services_are_named_as_uncovered(tmp_path):
    """S2-R1-B-3: the rules go in main only."""
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    task_dir = _task_dir(
        tmp_path, compose="services:\n  main: {}\n  target: {}\n  browser: {}\n"
    )
    plan = cloud_metadata_plan(
        _config(OPEN),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=task_dir / "environment",
    )
    assert plan is not None and plan.armed
    assert plan.uncovered == ("browser", "target")


def test_a_service_with_no_network_is_not_named_as_uncovered(tmp_path):
    """An embodied task's simulator has network_mode none, yet the trial's record
    said it reaches the endpoints. A service that extends another or joins networks still counts."""
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    task_dir = _task_dir(
        tmp_path,
        compose=(
            "services:\n  main: {}\n  simulator:\n    network_mode: none\n  target: {}\n"
            "  joined:\n    network_mode: none\n    extends: {service: target}\n"
            "  listed:\n    network_mode: none\n    networks: [default]\n"
        ),
    )
    plan = cloud_metadata_plan(
        _config(OPEN),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=task_dir / "environment",
    )
    assert plan is not None and plan.armed
    assert plan.uncovered == ("joined", "listed", "target")


@pytest.mark.asyncio
async def test_the_record_says_other_services_reach_the_endpoints(tmp_path):
    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        cloud_metadata_record,
    )

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True, uncovered=("target",)))
    block = AsyncMock(return_value=cloud_metadata_record("on", "proof"))
    with patch.object(trial._planes, "block_cloud_metadata", block):
        await trial._apply_cloud_metadata_block(tmp_path)

    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "on"
    assert record["uncovered_services"] == ["target"]
    assert "the rules cover main only" in record["why"]
    assert "(target) reach the endpoints" in record["why"]


@pytest.mark.asyncio
async def test_an_sdk_environment_gets_net_admin_for_the_block(tmp_path, monkeypatch):
    """S2-R1-B-1: Environment.from_task makes the sandbox before the agent is
    known and Runtime.execute hands it over as it is; the rollout then arms
    the block, so that sandbox must give main the NET_ADMIN it needs."""
    from benchflow.rollout import Rollout
    from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        compose_needs_net_admin,
    )

    task_dir = _task_dir(tmp_path)
    monkeypatch.chdir(tmp_path)
    env = Environment.from_task(task_dir, sandbox="docker")
    monkeypatch.setattr(type(env.inner), "start", AsyncMock())
    seen: dict = {}

    async def fake_run(self: Rollout):
        await self.setup()
        seen["plan"] = self._cloud_metadata
        seen["config"] = self._env.task_env_config
        seen["paths"] = self._env._docker_compose_paths
        raise RuntimeError("stop after setup")

    monkeypatch.setattr(Rollout, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop after setup"):
        await Runtime(
            env,
            Agent("claude-agent-acp", "m"),
            RuntimeConfig(sandbox_user="agent", timeout=60),
        ).execute()

    assert seen["plan"] is not None and seen["plan"].armed
    assert compose_needs_net_admin(seen["config"])
    assert COMPOSE_NET_ADMIN_PATH in seen["paths"]


def test_an_sdk_environment_never_arms_under_the_opt_out_or_the_hosts_network(
    tmp_path, monkeypatch
):
    from benchflow.runtime import Environment
    from benchflow.sandbox._compose import compose_needs_net_admin

    monkeypatch.chdir(tmp_path)
    opted = _task_dir(
        tmp_path / "opt-out", OPEN + "[metadata]\nx-allow-cloud-metadata = true\n"
    )
    host = _task_dir(tmp_path / "host", compose=HOST_MAIN)
    for task_dir in (opted, host):
        env = Environment.from_task(task_dir, sandbox="docker")
        assert not compose_needs_net_admin(env.inner.task_env_config)


# --- review round 2 -------------------------------------------------------------

#: Forms compose accepts that put main on the host's network without a plain
#: ``network_mode: host`` on main, or that cannot be read (S2-R2-B-1,
#: S2-R2-A-2).
UNREAD_OR_HIDDEN_HOST = {
    "extends": (
        "services:\n  base:\n    network_mode: host\n"
        "  main:\n    extends:\n      service: base\n"
    ),
    "reset-tag": "services:\n  main:\n    network_mode: host\n    ports: !reset []\n",
    "override-tag": "services:\n  main:\n    network_mode: !override host\n",
    "unknown-tag": "services:\n  main:\n    ports: !bogus []\n",
    "unparsable": "a: [b",
    "include": "include:\n  - base.yaml\nservices:\n  db: {}\n",
    "external-host": (
        "services:\n  main:\n    networks: [hostnet]\n"
        "networks:\n  hostnet:\n    external: true\n    name: host\n"
    ),
    "external-var": (
        "services:\n  main:\n    networks:\n      outer: {}\n"
        'networks:\n  outer:\n    external: true\n    name: "${NET}"\n'
    ),
}


@pytest.mark.parametrize(
    "compose", list(UNREAD_OR_HIDDEN_HOST.values()), ids=list(UNREAD_OR_HIDDEN_HOST)
)
def test_main_whose_network_is_hidden_or_unread_gets_no_net_admin(tmp_path, compose):
    """S2-R2-B-1: the check fails closed. A main that extends a host-network
    service, compose's !reset/!override tags, a file that does not parse, or
    an external network named host: no NET_ADMIN, no rule, not_installed."""
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        compose_needs_net_admin,
    )
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    docker = _sandbox_for(tmp_path, OPEN, block=True, compose=compose)
    assert compose_needs_net_admin(docker.task_env_config) is False
    assert COMPOSE_NET_ADMIN_PATH not in docker._docker_compose_paths

    task_dir = _task_dir(tmp_path / "plan", compose=compose)
    plan = cloud_metadata_plan(
        _config(OPEN),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=task_dir / "environment",
    )
    assert plan is not None and not plan.armed
    assert plan.state == "not_installed"
    assert "host's network" in plan.why
    assert "block the endpoints at the host" in plan.why


def test_the_record_says_why_mains_network_was_not_read(tmp_path):
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    for name, words in [
        ("extends", "main uses extends"),
        ("unparsable", "does not parse"),
        ("external-host", "external network 'hostnet' (name: host)"),
    ]:
        task_dir = _task_dir(tmp_path / name, compose=UNREAD_OR_HIDDEN_HOST[name])
        plan = cloud_metadata_plan(
            _config(OPEN),
            sandbox_user="agent",
            launches_agent=True,
            environment_dir=task_dir / "environment",
        )
        assert plan is not None and words in plan.why, (name, plan)


# --- review round 4: Compose network resolution --------------------------------

# main's attributes, top-level networks, whether the block must stay off.
# Also consumed by the VM's real-Compose parity check, not a second parser.
COMPOSE_NETWORK_CASES = {
    "implicit-default-external-host": (
        "",
        "default: {name: host, external: true}",
        True,
    ),
    "implicit-default-host": ("", "default: {name: host}", True),
    "implicit-default-legacy-host": ("", "default: {external: {name: host}}", True),
    "empty-list": ("networks: []", "default: {name: host, external: true}", True),
    "empty-map": ("networks: {}", "default: {name: host, external: true}", True),
    "reset-networks": ("networks: !reset [inner]", "default: {name: host}", True),
    "override-empty": ("networks: !override []", "default: {name: host}", True),
    "reset-mode": ("network_mode: !reset host", "default: {name: host}", True),
    "list-external-host": (
        "networks: [outer]",
        "outer: {name: host, external: true}",
        True,
    ),
    "map-host": ("networks: {outer: null}", "outer: {name: host}", True),
    "list-legacy-host": ("networks: [outer]", "outer: {external: {name: host}}", True),
    "map-legacy-host": (
        "networks: {outer: {}}",
        "outer: {external: {name: host}}",
        True,
    ),
    "external-host-key": ("networks: [host]", "host: {external: true}", True),
    "host-key": ("networks: [host]", "host: {}", False),
    "implicit-default-var": ("", 'default: {name: "${NET}"}', True),
    "external-var": (
        "networks: [outer]",
        'outer: {name: "${NET}", external: true}',
        True,
    ),
    "legacy-var": ("networks: [outer]", 'outer: {external: {name: "${NET}"}}', True),
    "key-var": ('networks: {"${NET}": {}}', '"${NET}": {external: true}', True),
    "escaped-dollar": ("networks: [outer]", 'outer: {name: "$$NET"}', True),
    "second-network": (
        "networks: [inner, outer]",
        "inner: {}\n  outer: {name: host}",
        True,
    ),
    "implicit-undeclared": ("", "", False),
    "empty-mode": ('network_mode: ""', "default: {name: host}", False),
    "implicit-inner": ("", "default: {name: inner}", False),
    "implicit-external-inner": ("", "default: {name: inner, external: true}", False),
    "unjoined-default": (
        "networks: [inner]",
        "inner: {}\n  default: {name: host}",
        False,
    ),
    "bridge-no-default": ("network_mode: bridge", "default: {name: host}", False),
    "none-no-default": ("network_mode: none", "default: {name: host}", False),
    "name-before-key": (
        "networks: [host]",
        "host: {name: inner, external: true}",
        False,
    ),
    "legacy-before-key": ("networks: [host]", "host: {external: {name: inner}}", False),
    "name-and-legacy": (
        "networks: [outer]",
        "outer: {name: inner, external: {name: inner}}",
        False,
    ),
    "inherited-main-host": (
        "@no-services",
        "default: {name: host, external: true}",
        True,
    ),
    "inherited-main-legacy-host": (
        "@sidecar",
        "default: {external: {name: host}}",
        True,
    ),
    "inherited-main-safe": ("@no-services", "default: {name: inner}", False),
}


def _network_case_compose(attributes: str, networks: str) -> str:
    # The VM oracle stacks this override on a base that defines main, as
    # BenchFlow does, including when the override has only networks/sidecars.
    if attributes.startswith("@"):
        services = (
            "services:\n  db: {image: busybox:1.37}\n"
            if attributes == "@sidecar"
            else ""
        )
        return services + f"networks:\n  {networks}\n"
    return (
        "services:\n  main:\n    image: busybox:1.37\n"
        + (f"    {attributes}\n" if attributes else "")
        + (f"networks:\n  {networks}\n" if networks else "")
    )


@pytest.mark.parametrize("case", COMPOSE_NETWORK_CASES)
def test_main_resolved_networks_control_the_block_and_capability(tmp_path, case):
    """Guards S2-R3-B-1 at 44bc1262: implicit default and external.name
    must not arm root's firewall or grant NET_ADMIN on an outside network.
    Safe explicit networks and network_mode must not inherit default.
    """
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        compose_needs_net_admin,
    )
    from benchflow.sandbox.cloud_metadata import (
        cloud_metadata_plan,
        task_compose_topology,
    )
    from benchflow.sandbox.daytona_dind import _DaytonaDinD

    attributes, networks, outside = COMPOSE_NETWORK_CASES[case]
    compose = _network_case_compose(attributes, networks)
    task_dir = _task_dir(tmp_path / "plan", compose=compose)
    topology = task_compose_topology(task_dir / "environment")
    assert topology.main_shares_outside_network is outside
    plan = cloud_metadata_plan(
        _config(OPEN),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=task_dir / "environment",
    )
    assert plan is not None and plan.armed is not outside
    if outside:
        assert plan.state == "not_installed"
        assert "block the endpoints at the host" in plan.why

    docker = _sandbox_for(tmp_path / "docker", OPEN, block=True, compose=compose)
    assert compose_needs_net_admin(docker.task_env_config) is not outside
    assert (COMPOSE_NET_ADMIN_PATH in docker._docker_compose_paths) is not outside
    daytona = _sandbox_for(
        tmp_path / "daytona", OPEN, block=True, backend="daytona", compose=compose
    )
    dind = f"{_DaytonaDinD._COMPOSE_DIR}/{COMPOSE_NET_ADMIN_PATH.name}"
    assert (dind in daytona._strategy._compose_file_flags()) is not outside


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["implicit-default-external-host", "map-legacy-host", "inherited-main-host"]
)
async def test_resolved_host_network_never_runs_the_metadata_script(
    tmp_path, monkeypatch, case
):
    """Guards S2-R3-B-1 at 44bc1262 through the rollout and on-disk record."""
    from benchflow.rollout import Rollout, RolloutConfig
    from benchflow.sandbox.cloud_metadata import CloudMetadataPowerError

    attributes, networks, _ = COMPOSE_NETWORK_CASES[case]
    task_dir = _task_dir(tmp_path, compose=_network_case_compose(attributes, networks))
    monkeypatch.chdir(tmp_path)
    trial = Rollout(
        RolloutConfig(task_path=task_dir, environment="docker", sandbox_user="agent")
    )
    trial.use_prebuilt_env(SimpleNamespace())
    await trial.setup()
    block = AsyncMock()
    with (
        patch.object(trial._planes, "block_cloud_metadata", block),
        # The agent does not start with the endpoints open.
        pytest.raises(CloudMetadataPowerError, match="sandbox startup refused"),
    ):
        await trial._apply_cloud_metadata_block(tmp_path)
    block.assert_not_awaited()
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "not_installed"
    assert "(name: host)" in record["why"]


async def _sdk_compose_paths(tmp_path, monkeypatch, agent: str, user: str | None):
    """The compose files main starts with when Runtime.execute runs an
    Environment.from_task sandbox (start mocked, stopped before the run)."""
    from benchflow.rollout import Rollout
    from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig

    task_dir = _task_dir(tmp_path / f"{agent}-{user}")
    monkeypatch.chdir(tmp_path)
    env = Environment.from_task(task_dir, sandbox="docker")
    seen: dict = {}

    async def fake_start(self, force_build: bool = False):
        seen["paths"] = self._docker_compose_paths

    async def fake_run(self: Rollout):
        raise RuntimeError("stop before the run")

    monkeypatch.setattr(type(env.inner), "start", fake_start)
    monkeypatch.setattr(Rollout, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop before the run"):
        await Runtime(
            env, Agent(agent, "m"), RuntimeConfig(sandbox_user=user, timeout=60)
        ).execute()
    return seen["paths"]


@pytest.mark.asyncio
async def test_an_sdk_run_without_the_block_gives_main_no_net_admin(
    tmp_path, monkeypatch
):
    """S2-R2-B-2: an oracle (no plan) on an Environment.from_task sandbox
    starts main without NET_ADMIN."""
    from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH

    paths = await _sdk_compose_paths(tmp_path, monkeypatch, "oracle", "agent")
    assert COMPOSE_NET_ADMIN_PATH not in paths


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [None, "root"], ids=["root-agent", "root-by-name"])
async def test_an_sdk_run_with_a_root_agent_holds_root_without_net_admin_or_net_raw(
    tmp_path, monkeypatch, user
):
    """A root agent on an Environment.from_task sandbox is held
    too (it was not before). With this process's helper image built, main
    starts with neither NET_ADMIN, which could remove the rules, nor
    NET_RAW, whose packet sockets go around them."""
    from benchflow.rollout import Rollout
    from benchflow.runtime import Agent, Environment, Runtime, RuntimeConfig
    from benchflow.sandbox import _metadata_helper as helper_mod
    from benchflow.sandbox._compose import (
        COMPOSE_NET_ADMIN_PATH,
        COMPOSE_NO_NET_RAW_PATH,
    )

    task_dir = _task_dir(tmp_path / f"root-{user}")
    monkeypatch.chdir(tmp_path)
    env = Environment.from_task(task_dir, sandbox="docker")
    daemon = " ".join(env.inner._docker_compose_global_args())
    monkeypatch.setattr(helper_mod, "_READY", {daemon: "sha256:" + "ab" * 32})
    seen: dict = {}

    async def fake_start(self, force_build: bool = False):
        await self._decide_metadata_net_admin()
        seen["paths"] = self._docker_compose_paths
        seen["config"] = self.task_env_config

    async def fake_run(self: Rollout):
        raise RuntimeError("stop before the run")

    monkeypatch.setattr(type(env.inner), "start", fake_start)
    monkeypatch.setattr(Rollout, "run", fake_run)
    with pytest.raises(RuntimeError, match="stop before the run"):
        await Runtime(
            env,
            Agent("claude-agent-acp", "m"),
            RuntimeConfig(sandbox_user=user, timeout=60),
        ).execute()
    assert seen["config"].cloud_metadata_root_agent is True
    assert COMPOSE_NET_ADMIN_PATH not in seen["paths"]
    assert COMPOSE_NO_NET_RAW_PATH in seen["paths"]


@pytest.mark.asyncio
async def test_an_sdk_run_with_the_block_starts_main_with_net_admin(
    tmp_path, monkeypatch
):
    from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH

    paths = await _sdk_compose_paths(tmp_path, monkeypatch, "claude-agent-acp", "agent")
    assert COMPOSE_NET_ADMIN_PATH in paths


def test_a_bare_sdk_environment_has_no_net_admin(tmp_path, monkeypatch):
    """bench sandbox and an SDK caller that starts the environment itself:
    no agent is known, so no NET_ADMIN."""
    from benchflow.runtime import Environment
    from benchflow.sandbox._compose import COMPOSE_NET_ADMIN_PATH

    monkeypatch.chdir(tmp_path)
    env = Environment.from_task(_task_dir(tmp_path), sandbox="docker")
    assert COMPOSE_NET_ADMIN_PATH not in env.inner._docker_compose_paths


# --- Root agents on providers without BenchFlow's helper ------------


class _ProviderSandbox:
    """A sandbox provider's environment (Daytona, Modal, AgentCore, Apple):
    no ``metadata_rules_install``, so no helper can add the rules."""

    cloud_metadata_provider_managed = True

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, command, **kwargs):
        self.commands.append(command)
        raise AssertionError(f"nothing runs for a provider-managed root: {command}")


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [None, "root"])
async def test_a_root_agent_on_a_provider_is_provider_managed(user):
    """The host and its metadata service are the provider's, so a
    root agent there gets the provider-managed record and BenchFlow runs
    nothing in the sandbox for it (a non-root agent keeps the block)."""
    from benchflow.sandbox.cloud_metadata import PROVIDER_MANAGED
    from benchflow.sandbox.lockdown import block_cloud_metadata

    env = _ProviderSandbox()
    record = await block_cloud_metadata(env, user)
    assert record["state"] == PROVIDER_MANAGED == "provider-managed"
    assert record["root_agent"] is True
    assert "provider" in record["why"] and "policy applies" in record["why"]
    assert env.commands == []


@pytest.mark.asyncio
async def test_a_provider_managed_root_agent_starts_with_one_warning(tmp_path, caplog):
    """The agent starts, result.json says provider-managed, and one
    warning line says the provider's own metadata policy applies."""
    import contextlib
    import logging

    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        cloud_metadata_record,
    )

    trial = _rollout(tmp_path, CloudMetadataPlan(armed=True, root_agent=True))
    trial._config.sandbox_user = None
    order: list[str] = []

    async def block(env, user):
        order.append(f"block {user}")
        return cloud_metadata_record("provider-managed", "the provider's host")

    with contextlib.ExitStack() as stack:
        for p in _connect_patches(trial, order, block):
            stack.enter_context(p)
        with caplog.at_level(logging.WARNING):
            await trial.connect()

    assert order == ["block None", "agent"]
    assert trial._cloud_metadata_result == {"state": "provider-managed"}
    warnings = [
        r.getMessage() for r in caplog.records if "metadata policy" in r.getMessage()
    ]
    assert warnings == [
        "Cloud metadata: the agent runs as root on a sandbox provider; the "
        "provider's own metadata policy applies (no BenchFlow block)"
    ]
    record = json.loads((tmp_path / "cloud-metadata.json").read_text())
    assert record["state"] == "provider-managed"


@pytest.mark.asyncio
async def test_the_strict_opt_in_refuses_a_provider_managed_root_agent(tmp_path):
    """require_cloud_metadata_block = true refuses the root agent
    on a provider instead of running it under the provider's policy."""
    import contextlib

    from benchflow.sandbox.cloud_metadata import (
        CloudMetadataPlan,
        CloudMetadataUnprotectedError,
        cloud_metadata_record,
    )

    trial = _rollout(
        tmp_path, CloudMetadataPlan(armed=True, root_agent=True, require_block=True)
    )
    trial._config.sandbox_user = None
    order: list[str] = []

    async def block(env, user):
        order.append(f"block {user}")
        return cloud_metadata_record("provider-managed", "the provider's host")

    with contextlib.ExitStack() as stack:
        for p in _connect_patches(trial, order, block):
            stack.enter_context(p)
        with pytest.raises(CloudMetadataUnprotectedError) as refused:
            await trial.connect()

    assert order == ["block None"]
    message = str(refused.value)
    assert "the block is provider-managed" in message
    assert "require_cloud_metadata_block = true" in message


def test_the_strict_opt_in_is_read_from_the_tasks_environment():
    from benchflow.sandbox.cloud_metadata import cloud_metadata_plan

    strict = cloud_metadata_plan(
        _config(OPEN + "[environment]\nrequire_cloud_metadata_block = true\n"),
        sandbox_user=None,
        launches_agent=True,
    )
    plain = cloud_metadata_plan(_config(OPEN), sandbox_user=None, launches_agent=True)
    assert strict is not None and strict.armed and strict.require_block
    assert plain is not None and plain.armed and not plain.require_block


def test_a_root_agents_main_never_gets_net_admin_for_the_block():
    """A root agent is held only where root has no NET_ADMIN, so the block
    never gives its main one (a provider's root agent keeps what it had
    before); a non-root agent's main still gets it without the helper."""
    from benchflow.sandbox._compose import compose_needs_net_admin
    from benchflow.task.config import SandboxConfig

    root = SandboxConfig(allow_internet=True)
    root.arm_cloud_metadata_block(root_agent=True)
    agent = SandboxConfig(allow_internet=True)
    agent.arm_cloud_metadata_block()
    assert not compose_needs_net_admin(root)
    assert compose_needs_net_admin(agent)


@pytest.mark.parametrize("spelled", ["root", "0", None, "null", "None"])
def test_root_is_one_case_at_config_time(spelled):
    """``--sandbox-user root`` and ``0`` are the root agent, as ``null`` is:
    the run's config holds None for each, so every later check sees one case."""
    from benchflow._utils.config import normalize_sandbox_user
    from benchflow.rollout import Role, RolloutConfig, Scene

    assert normalize_sandbox_user(spelled) is None
    cfg = RolloutConfig(
        task_path=Path("task"),
        scenes=[Scene(roles=[Role(name="agent", agent="claude-agent-acp", model="m")])],
        sandbox_user=spelled,
    )
    assert cfg.sandbox_user is None


@pytest.mark.parametrize("user", ["agent", "Root", "root2", "00", "1000"])
def test_other_users_are_kept_as_given(user):
    from benchflow._utils.config import normalize_sandbox_user

    assert normalize_sandbox_user(user) == user


@pytest.mark.asyncio
@pytest.mark.parametrize("spelled", ["root", "0"])
async def test_a_root_agent_spelled_root_on_a_provider_runs_no_firewall_power_check(
    tmp_path, spelled
):
    """A ``--sandbox-user root`` (or ``0``) agent on a provider without
    BenchFlow's helper starts provider-managed, as ``null`` does: the
    firewall-power check does not run (no firewall-power.json), so it cannot
    disarm or refuse root (on 55c66cf4 it ran and could do either)."""
    import contextlib

    from benchflow.rollout import Role, Rollout, RolloutConfig, Scene
    from benchflow.sandbox.cloud_metadata import CloudMetadataPlan
    from benchflow.sandbox.lockdown import block_cloud_metadata

    cfg = RolloutConfig(
        task_path=tmp_path / "task",
        scenes=[Scene(roles=[Role(name="agent", agent="claude-agent-acp", model="m")])],
        sandbox_user=spelled,
    )
    trial = Rollout(cfg)
    trial._env = _ProviderSandbox()
    trial._rollout_dir = tmp_path
    trial._phase = "idle"
    trial._cloud_metadata = CloudMetadataPlan(armed=True, root_agent=True)
    order: list[str] = []

    async def block(env, user):
        order.append(f"block {user}")
        return await block_cloud_metadata(env, user)

    with contextlib.ExitStack() as stack:
        for p in _connect_patches(trial, order, block):
            stack.enter_context(p)
        await trial.connect()

    assert order == ["block None", "agent"]
    assert not (tmp_path / "firewall-power.json").exists()
    assert trial._cloud_metadata_result == {"state": "provider-managed"}
    assert trial._env.commands == []


@pytest.mark.asyncio
async def test_a_root_agent_on_an_unknown_sandbox_class_is_not_provider_managed():
    """provider-managed is for the four providers marked on their
    classes. Any other sandbox without BenchFlow's helper (an SDK
    caller's own class) gets not_installed for a root agent, so the trial
    stops unless the run opts out. On 8a0d5b250 it was provider-managed."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    class OwnSandbox:
        async def exec(self, command, **kwargs):
            raise AssertionError(command)

    record = await block_cloud_metadata(OwnSandbox(), None)
    assert record["state"] == "not_installed", record
    assert "OwnSandbox" in record["why"] and "not a known provider" in record["why"]


def test_the_four_providers_are_marked_provider_managed():
    from benchflow.sandbox.agentcore import AgentCoreSandbox
    from benchflow.sandbox.apple_container import AppleContainerSandbox
    from benchflow.sandbox.daytona import DaytonaSandbox
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.modal_impl import ModalSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    for cls in (DaytonaSandbox, ModalSandbox, AgentCoreSandbox, AppleContainerSandbox):
        assert cls.cloud_metadata_provider_managed is True, cls
    for cls in (DockerSandbox, RemoteDockerSandbox):
        assert getattr(cls, "cloud_metadata_provider_managed", False) is False, cls
