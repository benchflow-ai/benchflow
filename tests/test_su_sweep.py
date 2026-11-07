"""Guards SU sweep fixes for 5a175946."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._firewall_tools import audit_image_tools  # noqa: F401
from tests._receipt_fakes import receipt_aware

pytestmark = pytest.mark.usefixtures("audit_image_tools")


def _tool(folder: Path, name: str, body: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return path


def test_whole_scan_visits_a_separate_executable_mount(tmp_path):
    """Model find's device boundary without mounting."""
    from benchflow.sandbox.firewall_power import _scan_cmd, _scanned

    root = tmp_path / "fs"
    mounted = root / "opt" / "task tools"
    marker = _tool(mounted, "root-helper", "exit 0\n")
    marker.chmod(0o4755)
    mountinfo = tmp_path / "mountinfo"
    escaped = str(mounted).replace(" ", r"\040")
    mountinfo.write_text(
        f"1 0 0:1 / {root} rw - overlay overlay rw\n"
        f"2 1 0:2 / {escaped} rw - tmpfs tmpfs rw\n"
    )
    tools = tmp_path / "bin"
    # This is a device-boundary stand-in, not a stand-in for the detector:
    # find on the outer root with -xdev cannot enumerate the mounted file.
    _tool(
        tools,
        "find",
        f'if [ "$1" = {shlex.quote(str(root))} ]; then\n'
        '  case " $* " in *" -xdev "*) exit 0;; esac\nfi\n'
        f'exec {shlex.quote(shutil.which("find") or "/usr/bin/find")} "$@"\n',
    )
    _tool(tools, "getcap", "exit 0\n")
    command = _scan_cmd(
        whole_fs=True, fs_root=str(root), setuid_owner=str(os.getuid())
    ).replace("/proc/self/mountinfo", str(mountinfo))
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env={**os.environ, "PATH": f"{tools}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(marker) in [entry["path"] for entry in _scanned(result.stdout)]


@pytest.mark.parametrize("failure", ["find-error", "truncated", "mountinfo"])
def test_incomplete_whole_scan_is_reported(tmp_path, failure):
    """A partial traversal cannot authorize launch."""
    from benchflow.sandbox.firewall_power import _scan_cmd

    root = tmp_path / "fs"
    root.mkdir()
    tools = tmp_path / "bin"
    if failure == "find-error":
        _tool(tools, "find", "exit 1\n")
    elif failure == "truncated":
        _tool(
            tools,
            "find",
            "i=0; while [ $i -lt 1001 ]; do echo /tmp/dummy-$i; i=$((i+1)); done\n",
        )
    _tool(tools, "getcap", "exit 0\n")
    command = _scan_cmd(whole_fs=True, fs_root=str(root))
    if failure == "mountinfo":
        command = command.replace("/proc/self/mountinfo", str(tmp_path / "absent"))
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        env={**os.environ, "PATH": f"{tools}:/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "complete=0" in result.stdout


class _Env:
    def __init__(self, scan: str, agent: str):
        self.scan = scan
        self.agent = agent

    @receipt_aware
    async def exec(self, command, **kwargs):
        out = self.scan if "BF fwpower-root" in command else self.agent
        return SimpleNamespace(stdout=out, stderr="", return_code=0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,caps",
    [
        ("/opt/py helper", "cap_setuid=ep"),
        ("/opt/fw tools/iptables", "cap_net_admin=ep"),
        ("/opt/leading and trailing ", "cap_net_admin=ep cap_setuid=ep"),
    ],
)
async def test_capability_paths_keep_spaces_and_refuse_startup(path, caps):
    """Guards the fix through the startup decision."""
    from benchflow.sandbox.firewall_power import check_firewall_power

    scan = (
        "BF fwpower-root net_admin=1 setpriv=none nnp=0\n"
        f"BF fwpower-found cap {path} {caps}\n"
        "BF fwpower-scan scope=all caps=getcap\n"
    )
    agent = (
        f"BF fwpower-x {path}\n"
        "BF fwpower-agent bnd=1000 inh=0 amb=0 prm=0 eff=0 nnp=0 power=\n"
    )
    record, refusal = await check_firewall_power(
        _Env(scan, agent), "agent", holds=("dummy uid firewall",)
    )
    assert refusal and record["state"] == "refused", record
    assert record["found"][0]["path"] == path


@pytest.mark.asyncio
async def test_incomplete_scan_refuses_even_when_capabilities_were_read():
    """Guards the fail-closed startup decision."""
    from benchflow.sandbox.firewall_power import check_firewall_power

    scan = (
        "BF fwpower-root net_admin=1 setpriv=none nnp=0\n"
        "BF fwpower-scan scope=all caps=getcap complete=0\n"
    )
    agent = "BF fwpower-agent bnd=1000 inh=0 amb=0 prm=0 eff=0 nnp=0 power=\n"
    record, refusal = await check_firewall_power(
        _Env(scan, agent), "agent", holds=("dummy uid firewall",)
    )
    assert refusal and record["state"] == "refused", record


@pytest.mark.parametrize(
    "name",
    ["py helper", "tab\thelper", "line\nhelper", "helper cap_setuid=ep", "helper "],
)
@pytest.mark.parametrize("kind", ["cap", "setuid"])
def test_scan_and_runnable_check_preserve_exact_filename_bytes(tmp_path, name, kind):
    """Guards the transport fix, including delimiter-shaped names."""
    from benchflow.sandbox.firewall_power import (
        _agent_check_cmd,
        _runnable,
        _scan_cmd,
        _scanned,
    )

    root = tmp_path / "fs"
    marker = _tool(root, name, "exit 0\n")
    if kind == "setuid":
        marker.chmod(0o4755)
    tools = tmp_path / "bin"
    _tool(
        tools,
        "getcap",
        'for f do printf "%s cap_setuid=ep\\n" "$f"; done\n'
        if kind == "cap"
        else "exit 0\n",
    )
    env = {**os.environ, "PATH": f"{tools}:/usr/bin:/bin"}
    scanned_out = subprocess.run(
        [
            "/bin/sh",
            "-c",
            _scan_cmd(whole_fs=True, fs_root=str(root), setuid_owner=str(os.getuid())),
        ],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    scanned = _scanned(scanned_out)
    assert [entry["path"] for entry in scanned] == [str(marker)]
    agent_out = subprocess.run(
        ["/bin/sh", "-c", _agent_check_cmd("BFFWdummy", [str(marker)])],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout
    assert _runnable(scanned, agent_out) == scanned


@pytest.mark.parametrize("main", ["image: dummy", "networks: []", "networks: {}", ""])
@pytest.mark.parametrize(
    "network",
    [
        "external: true\n    name: host",
        "external:\n      name: host",
        "name: host",
        "external: true\n    name: ${DUMMY_NETWORK}",
    ],
)
def test_implicit_default_host_network_is_not_armed(tmp_path, main, network):
    """Guards the fix, including a main inherited from the base file."""
    from benchflow.sandbox.cloud_metadata import (
        cloud_metadata_plan,
        task_compose_topology,
    )
    from benchflow.task.config import TaskConfig

    services = f"services:\n  main:\n    {main}\n" if main else ""
    (tmp_path / "docker-compose.yaml").write_text(
        services + f"networks:\n  default:\n    {network}\n"
    )
    topology = task_compose_topology(tmp_path)
    assert topology.main_shares_outside_network, topology
    plan = cloud_metadata_plan(
        TaskConfig(),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=tmp_path,
    )
    assert plan is not None and not plan.armed and plan.state == "not_installed"


@pytest.mark.parametrize("mode", ["", "network_mode: none", "networks: [private]"])
def test_default_bridge_and_explicit_other_network_are_safe_controls(tmp_path, mode):
    """Guards the fix against refusing a private network unnecessarily."""
    from benchflow.sandbox.cloud_metadata import task_compose_topology

    default = "name: dummy-bridge" if not mode else "external: true\n    name: host"
    (tmp_path / "docker-compose.yaml").write_text(
        f"services:\n  main:\n    image: dummy\n    {mode}\n"
        f"networks:\n  default:\n    {default}\n  private: {{}}\n"
    )
    assert not task_compose_topology(tmp_path).main_shares_outside_network
