"""Guard SU-INTEG-C1/C2 in the integration review of 9043c877."""

from __future__ import annotations

import os
import shlex
import subprocess
from types import SimpleNamespace

import pytest

TOOLS = (
    "id",
    "mkfifo",
    "rmdir",
    "ls",
    "readlink",
    "awk",
    "setpriv",
    "getcap",
    "sh",
    "od",
    "tr",
    "find",
    "sort",
    "head",
    "wc",
    "mktemp",
    "rm",
    "grep",
    "bash",
    "env",
    "sed",
    "su",
    "iptables",
    "iptables-nft",
    "iptables-legacy",
    "ip6tables",
    "ip6tables-nft",
    "ip6tables-legacy",
    "nft",
)
BACKENDS = (
    "docker",
    "remote_docker",
    "daytona",
    "daytona_dind",
    "modal",
    "agentcore",
    "apple",
)


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "stage", ["root", "path", "all", "agent", "launch", "cleanup", "construction"]
)
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
async def test_no_audit_stage_executes_an_earlier_path_tool(
    tmp_path, monkeypatch, backend, stage, shell
):
    """Guards 9043c877: environment overlays cannot supply any audit tool."""
    from benchflow.sandbox import firewall_power as fp
    from benchflow.sandbox import lockdown

    hostile = tmp_path / "agent-bin"
    hostile.mkdir()
    marker = tmp_path / "planted-tools-ran"
    for tool in TOOLS:
        path = hostile / tool
        path.write_text(
            f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(tool)} >> {shlex.quote(str(marker))}\nexit 0\n"
        )
        path.chmod(0o755)
    root = tmp_path / "fs"
    root.mkdir()
    sample = root / "dummy-carrier"
    sample.write_text("not executed\n")
    sample.chmod(0o4755)
    proc = tmp_path / "proc" / "self"
    proc.mkdir(parents=True)
    (proc / "status").write_text("CapBnd:\t0000000000001000\n")
    mounts = tmp_path / "mountinfo"
    mounts.write_text(f"1 0 0:1 / {root} rw - overlay overlay rw\n")
    monkeypatch.setattr(fp, "_SYSTEM_DIRS", (str(root),))
    commands = {
        "construction": lambda: lockdown._cloud_metadata_construction_cmd(
            "ubuntu", protected_tools=True
        ),
        "root": lambda: fp._root_cmd(proc=str(proc.parent), fs_root=str(root)),
        "path": lambda: fp._scan_cmd(whole_fs=False, fs_root=str(root)),
        "all": lambda: fp._scan_cmd(whole_fs=True, fs_root=str(root)),
        "agent": lambda: fp._agent_check_cmd("BFFWdummy", proc=str(proc.parent)),
        "launch": lambda: fp._agent_cmd("true", "nobody"),
        "cleanup": lambda: fp._cleanup_cmd("BFFWdummy"),
    }
    # The old tip has no adapter: exercise its actual launch expression,
    # not an import/attribute failure, for the failing-first comparison.
    if stage == "launch" and not hasattr(fp, "_agent_cmd"):
        from benchflow.sandbox.lockdown import build_priv_drop_cmd

        commands[stage] = lambda: build_priv_drop_cmd("true", "nobody")
    command = commands[stage]().replace("/proc/self/mountinfo", str(mounts))

    class Env:
        supports_system_shell = backend in (
            "docker",
            "remote_docker",
            "daytona",
            "daytona_dind",
        )

        async def exec(self, wrapped, **kwargs):
            assert kwargs["user"] == "root"
            assert kwargs.get("system_shell", False) == self.supports_system_shell
            # Every provider sees this overlay after its own initial pin;
            # the shared audit must defend itself after materialization.
            result = subprocess.run(
                [shell, "-c", wrapped],
                env={**os.environ, "PATH": str(hostile)},
                capture_output=True,
                text=True,
                timeout=30,
            )
            return SimpleNamespace(
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
            )

    await fp._exec(Env(), command, 30)
    assert not marker.exists(), marker.read_text() if marker.exists() else ""


@pytest.mark.parametrize("name", TOOLS)
@pytest.mark.parametrize(
    "route", ["writable-leaf", "writable-parent", "symlink-parent"]
)
def test_audit_tools_require_protected_executables_and_ancestors(
    tmp_path, monkeypatch, name, route
):
    """Guards 9043c877's sibling: a root-path spelling is not custody."""
    from benchflow.sandbox import firewall_power as fp
    from benchflow.sandbox import lockdown

    folder = tmp_path / "trusted"
    folder.mkdir()
    marker = tmp_path / "unsafe-ran"
    executable = folder / name
    executable.write_text(f"#!/bin/sh\nprintf unsafe > {shlex.quote(str(marker))}\n")
    executable.chmod(0o777 if route == "writable-leaf" else 0o755)
    if route != "writable-leaf":
        folder.chmod(0o777)
    if route == "symlink-parent":
        link = tmp_path / "alias"
        link.symlink_to(folder, target_is_directory=True)
        folder = link
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_TRUSTED_UIDS", (str(os.getuid()), "0")
    )
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_ROOT_PATH", f"{folder}:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    subprocess.run(
        ["/bin/sh", "-c", fp._audit_prelude() + f"run {shlex.quote(name)} --help"],
        capture_output=True,
        timeout=10,
    )
    assert not marker.exists()


SAFE_BARRIER = "BF metadata-construction uid=1000 nnp=1 bnd=0000000000000000 amb=0000000000000000\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "inventory", ["missing-tools", "root-timeout", "agent-timeout", "incomplete"]
)
@pytest.mark.parametrize(
    "positive",
    [
        "",
        "power=iptables",
        "prm=0000000000001000",
        "eff=0000000000001000",
        "amb=0000000000001000",
    ],
)
async def test_verified_barrier_survives_missing_inventory_but_not_unsafe_evidence(
    inventory, positive
):
    """Guards SU-INTEG-C2 at 9043c877, including late failing audit commands."""
    from benchflow.sandbox import firewall_power as fp
    from tests._receipt_fakes import receipt_aware

    class Env:
        supports_system_shell = True

        def __init__(self):
            self.calls = []

        @receipt_aware
        async def exec(self, command, **kwargs):
            self.calls.append(command)
            if "BF metadata-construction" in command:
                out, code = SAFE_BARRIER, 0
            elif "BF fwpower-root" in command:
                out = "BF fwpower-root net_admin=1 setpriv=ok nnp=1\nBF fwpower-scan scope=all caps=none complete=0\n"
                code = 7 if inventory == "root-timeout" else 0
            elif "BF fwpower-agent" in command:
                out, code = "", 7 if inventory == "agent-timeout" else 0
                if positive:
                    fields = dict(
                        bnd="0000000000000000",
                        inh="0",
                        amb="0",
                        prm="0",
                        eff="0",
                        nnp="1",
                        power="",
                    )
                    key, value = positive.split("=")
                    fields[key] = value
                    out = (
                        "BF fwpower-agent "
                        + " ".join(f"{k}={v}" for k, v in fields.items())
                        + "\n"
                    )
            else:
                out, code = "", 0
            return SimpleNamespace(stdout=out, stderr="", return_code=code)

    env = Env()
    record, refusal = await fp.check_firewall_power(
        env, "agent", holds=("dummy firewall",)
    )
    assert bool(refusal) == bool(positive), record
    assert record["state"] == ("refused" if positive else "disarmed")
    assert record["mandatory_barrier"] == "verified"
    assert record["supplemental_inventory"] == "unavailable"
    assert fp.net_admin_dropped(env, "agent")
    if "power=" in positive:
        assert any("-X BFFW" in command for command in env.calls)


@pytest.mark.parametrize(
    "bad",
    [
        "empty",
        "duplicate",
        "root",
        "nnp",
        "net-admin",
        "net-raw",
        "ambient",
        "malformed",
        "nonzero",
        "boolean",
    ],
)
def test_supplemental_and_s2_share_the_exact_barrier_acceptance_rule(bad):
    """Guards SU-INTEG-C2: inventory availability never manufactures a proof."""
    from benchflow.sandbox.lockdown import _cloud_metadata_barrier_verified

    output, code = SAFE_BARRIER, 0
    output = {
        "empty": "",
        "duplicate": output * 2,
        "root": output.replace("uid=1000", "uid=0"),
        "nnp": output.replace("nnp=1", "nnp=0"),
        "net-admin": output.replace("bnd=0000000000000000", "bnd=0000000000001000"),
        "net-raw": output.replace("bnd=0000000000000000", "bnd=0000000000002000"),
        "ambient": output.replace("amb=0000000000000000", "amb=0000000000000001"),
        "malformed": output.replace("bnd=0000000000000000", "bnd=unknown"),
    }.get(bad, output)
    if bad == "nonzero":
        code = 7
    if bad == "boolean":
        code = False
    assert not _cloud_metadata_barrier_verified(output, code)
