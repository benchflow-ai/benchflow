"""Guard SU-CONFIRM-1/2 against regressions in commit 0ef0e209."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests._firewall_tools import audit_image_tools  # noqa: F401
from tests._receipt_fakes import receipt_aware

pytestmark = pytest.mark.usefixtures("audit_image_tools")


def _tool(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)


def _scan(tmp_path, reader, tool=None, failure="error", blocked=None, shell="/bin/sh"):
    from benchflow.sandbox.firewall_power import _scan_cmd

    root = tmp_path / "fs"
    root.mkdir()
    marker = root / "dummy root helper"
    marker.write_text("dummy marker, never executed\n")
    marker.chmod(0o4755)
    tools = tmp_path / "tools"
    tools.mkdir()
    # An explicit tool set models a minimal trusted image; no host PATH
    # fallback can accidentally rescue the command whose failure we test.
    names = [
        "sh",
        "mktemp",
        "awk",
        "sort",
        "find",
        "wc",
        "head",
        "od",
        "tr",
        "rm",
        "touch",
    ]
    if reader == "python3":
        names.append("python3")
    else:
        _tool(tools, "getcap", 'for f do printf "%s cap_setuid=ep\\n" "$f"; done\n')
    for name in names:
        actual = shutil.which(name)
        assert actual, name
        if name != tool or failure != "missing":
            (tools / name).symlink_to(actual)
    if tool and failure != "missing":
        (tools / tool).unlink()
        actual = shutil.which(tool)
        assert actual, tool
        actual = shlex.quote(actual)
        body = f'{actual} "$@"\nexit 1\n' if failure == "partial" else "exit 1\n"
        _tool(tools, tool, body)
    if blocked:
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        (scratch / blocked).mkdir()
        (tools / "mktemp").unlink()
        _tool(tools, "mktemp", f"echo {shlex.quote(str(scratch))}\n")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 0 0:1 / {root} rw - overlay overlay rw\n")
    command = _scan_cmd(
        whole_fs=True,
        fs_root=str(root),
        setuid_owner=str(os.getuid()),
        python_command=str(tools / "python3"),
    )
    command = command.replace("/proc/self/mountinfo", str(mountinfo))
    return subprocess.run(
        [shell, "-c", command],
        env={**os.environ, "PATH": str(tools)},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    ).stdout


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
@pytest.mark.parametrize("reader", ["getcap", "python3"])
@pytest.mark.parametrize("failure", ["missing", "error", "partial"])
@pytest.mark.parametrize(
    "tool", ["mktemp", "awk", "sort", "find", "wc", "head", "od", "tr", "sh"]
)
def test_scan_command_failure_cannot_report_complete(
    tmp_path, reader, failure, tool, shell
):
    """Guards SU-CONFIRM-1: every scan stage must finish successfully."""
    from benchflow.sandbox.firewall_power import _scan_report

    report = _scan_report(_scan(tmp_path, reader, tool, failure, shell=shell))
    assert report.get("complete") == "0", report


@pytest.mark.parametrize("reader", ["getcap", "python3"])
@pytest.mark.parametrize(
    "blocked", ["roots", "mounts", "sorted", "setuid", "caps", "errors"]
)
def test_scan_output_failure_cannot_report_complete(tmp_path, reader, blocked):
    """Guards SU-CONFIRM-1: failed redirections cannot publish success."""
    from benchflow.sandbox.firewall_power import _scan_report

    report = _scan_report(_scan(tmp_path, reader, blocked=blocked))
    assert report.get("complete") == "0", report


@pytest.mark.parametrize("reader", ["getcap", "python3"])
def test_successful_scan_is_still_complete(tmp_path, reader):
    """Guards SU-CONFIRM-1 with a complete-scan positive control."""
    from benchflow.sandbox.firewall_power import _scan_report, _scanned

    output = _scan(tmp_path, reader)
    assert _scan_report(output).get("complete") != "0", output
    assert _scanned(output)[0]["path"] == str(tmp_path / "fs" / "dummy root helper")


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["no-setpriv", "session-factory", "failed-drop"])
@pytest.mark.parametrize("reader", ["getcap", "python3"])
async def test_failed_preparation_refuses_every_no_drop_route(tmp_path, route, reader):
    """Guards SU-CONFIRM-1 through initial and fallback startup scans."""
    from benchflow.sandbox.firewall_power import check_firewall_power

    scan = _scan(tmp_path, reader, "sort", "missing")

    class Env:
        @receipt_aware
        async def exec(self, command, **kwargs):
            if "BF fwpower-root" in command:
                output = (
                    "BF fwpower-root net_admin=1 setpriv=ok nnp=1\n"
                    "BF fwpower-scan scope=path caps=getcap\n"
                    if route == "failed-drop"
                    else "BF fwpower-root net_admin=1 setpriv=none nnp=0\n" + scan
                )
            elif "BF fwpower-scan" in command:
                output = scan
            else:
                output = (
                    "BF fwpower-agent bnd=1000 inh=0 amb=0 prm=0 eff=0 nnp=0 power=\n"
                )
            return SimpleNamespace(stdout=output, return_code=0)

    record, refusal = await check_firewall_power(
        Env(),
        "agent",
        holds=("dummy uid firewall",),
        can_drop=route != "session-factory",
    )
    assert refusal and record["state"] == "refused", record
    assert "incomplete" in record["why"]


@pytest.mark.parametrize("failure", ["missing", "error", "partial", "empty", "invalid"])
def test_root_status_failure_is_unknown_not_no_capability(tmp_path, failure):
    """Guards SU-CONFIRM-1's sibling root capability-reader failure."""
    from benchflow.sandbox.firewall_power import _root_cmd

    if failure != "missing":
        _tool(
            tmp_path,
            "awk",
            {
                "error": "exit 1\n",
                "partial": "echo 0000000000000000; exit 1\n",
                "empty": "exit 0\n",
                "invalid": "echo unknown\n",
            }[failure],
        )
    result = subprocess.run(
        ["/bin/sh", "-c", _root_cmd(proc="/tmp/dummy-proc")],
        env={"PATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "net_admin=unknown" in result.stdout


@pytest.mark.parametrize(
    "joined", [["host"], {"host": None}, {"host": {"aliases": ["dummy"]}}]
)
@pytest.mark.parametrize(
    "spec",
    [
        None,
        {},
        {"driver": "bridge"},
        {"external": False},
        {"external": False, "driver": "bridge"},
    ],
)
def test_project_scoped_host_key_keeps_metadata_protection(tmp_path, joined, spec):
    """Guards SU-CONFIRM-2: a private key is project-prefixed by Compose."""
    from benchflow.sandbox.cloud_metadata import (
        cloud_metadata_plan,
        task_compose_topology,
    )
    from benchflow.task.config import TaskConfig

    (tmp_path / "docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {"main": {"image": "dummy", "networks": joined}},
                "networks": {"host": spec},
            }
        )
    )
    assert not task_compose_topology(tmp_path).main_shares_outside_network
    plan = cloud_metadata_plan(
        TaskConfig(),
        sandbox_user="agent",
        launches_agent=True,
        environment_dir=tmp_path,
    )
    assert plan is not None and plan.armed


@pytest.mark.parametrize(
    "key,spec,outside",
    [
        ("host", {"external": True}, True),
        ("private", {"name": "host"}, True),
        ("private", {"external": {"name": "host"}}, True),
        ("private", {"name": "${DUMMY_NETWORK}"}, True),
        ("private", {"external": {"name": "${DUMMY_NETWORK}"}}, True),
        ("host", {"name": "dummy-bridge"}, False),
        ("host", {"external": True, "name": "dummy-bridge"}, False),
        ("host", {"external": {"name": "dummy-bridge"}}, False),
        ("host", {"name": "dummy-bridge", "external": {"name": "host"}}, False),
    ],
)
def test_literal_network_names_keep_their_meaning(tmp_path, key, spec, outside):
    """Guards SU-CONFIRM-2 with host, variable and explicit-name controls."""
    from benchflow.sandbox.cloud_metadata import task_compose_topology

    (tmp_path / "docker-compose.yaml").write_text(
        yaml.safe_dump(
            {
                "services": {"main": {"image": "dummy", "networks": [key]}},
                "networks": {key: spec},
            }
        )
    )
    assert task_compose_topology(tmp_path).main_shares_outside_network == outside
