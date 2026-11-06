"""Guard the root audit requirement left incomplete at d4cbbc3c.

Root inventory, the launcher and both cleanups share the metadata root
route. Agent PATH entries remain data and never choose a root executable.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from importlib import import_module
from types import SimpleNamespace

import pytest

from benchflow.sandbox import firewall_power as fp
from benchflow.sandbox import lockdown
from tests._receipt_fakes import receipt_aware
from tests.test_firewall_power_retry import OUTCOMES, AuditEnv


@pytest.mark.parametrize(
    "module,name,supported",
    [
        ("docker", "DockerSandbox", True),
        ("remote_docker", "RemoteDockerSandbox", True),
        ("daytona", "DaytonaSandbox", True),
        ("modal_impl", "ModalSandbox", True),
        ("agentcore", "AgentCoreSandbox", True),
        ("apple_container", "AppleContainerSandbox", True),
    ],
)
async def test_backend_contract_applies_on_first_connect_and_reconnect(
    module, name, supported
):
    """Guard d4cbbc3c on real backend capability declarations, without providers."""
    cls = getattr(import_module(f"benchflow.sandbox.{module}"), name)
    env = object.__new__(cls)
    captured = AuditEnv(OUTCOMES["safe"])

    async def execute(command, *, user, **kwargs):
        assert kwargs.get("system_shell") is True
        return await captured.exec(command, user=user, **kwargs)

    env.exec = execute
    assert env.supports_system_shell is supported
    for _ in range(2):
        record, refusal = await fp.check_firewall_power(
            env, "agent", holds=("denylist",)
        )
        assert (refusal is None) is supported
        assert record["state"] == ("disarmed" if supported else "refused")
    assert len(captured.calls) == (6 if supported else 0)


@pytest.mark.parametrize("can_drop", [True, False])
@pytest.mark.parametrize("outcome", ["safe", "power", "empty", "exception"])
async def test_every_audit_root_step_uses_the_protected_route(can_drop, outcome):
    """Guard d4cbbc3c's inventory, launcher and two cleanup call sites."""

    class Env(AuditEnv):
        supports_system_shell = True

        @receipt_aware
        async def exec(self, command, *, user, **kwargs):
            if user == "root":
                assert kwargs.get("system_shell") is True
                assert '"$bf_env" -i PATH="$PATH"' in command
                assert '"$bf_sh" -c ' in command
                assert "$(tool " in command
            return await super().exec(command, user=user, **kwargs)

    env = Env(OUTCOMES[outcome])
    for _ in range(2):
        await fp.check_firewall_power(
            env, "agent", holds=("denylist",), can_drop=can_drop
        )
    roots = [c for c, u in env.calls if u == "root"]
    # _exec records assertion errors as failed probes: also assert coverage
    # outside the fake transport so a failure cannot turn this test green.
    assert len(roots) == 2 * (
        1 + 2 * can_drop + (outcome == "power") + (can_drop and outcome != "safe")
    )
    if outcome == "power":
        assert sum("delete table inet" in c for c in roots) >= 2


@pytest.mark.parametrize("support", [False, None, 1, "yes"])
@pytest.mark.parametrize("can_drop", [True, False])
async def test_missing_protected_backend_refuses_before_any_exec(support, can_drop):
    """Guard d4cbbc3c: do not fall back to a PATH-started root shell."""
    env = AuditEnv(OUTCOMES["safe"])
    env.supports_system_shell = support
    record, refusal = await fp.check_firewall_power(
        env, "agent", holds=("denylist",), can_drop=can_drop
    )
    assert not env.calls
    assert record["state"] == "refused"
    assert refusal and "backend" in refusal and "system-shell" in refusal


@pytest.mark.parametrize(
    "tool", ["awk", "find", "head", "sort", "setpriv", "grep", "bash"]
)
async def test_required_tool_refusal_names_the_repair(tool):
    """Guard d4cbbc3c: an unavailable absolute tool has an actionable refusal."""

    class Env:
        supports_system_shell = True

        @receipt_aware
        async def exec(self, *args, **kwargs):
            return SimpleNamespace(
                stdout=f"BF fwpower-error tool={tool}\n", return_code=126
            )

    record, refusal = await fp.check_firewall_power(Env(), "agent", holds=("denylist",))
    assert record["state"] == "refused"
    assert refusal and f"absolute path for {tool}" in refusal
    assert "install it" in refusal and "symlink targets" in refusal


def _route(command, **kwargs):
    # The pre-fix archive lacks the wrapper. Run its actual command anyway
    # so fails-before demonstrates execution behavior, not a missing import.
    route = getattr(fp, "_root_route", None)
    return route(command, **kwargs) if route else command


def _run(command, path, **env):
    return subprocess.run(
        ["/bin/sh", "-c", command],
        env={"PATH": path, **env},
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize("position", ["first", "last", "relative", "empty"])
@pytest.mark.parametrize(
    "tool", ["setpriv", "awk", "getcap", "python3", "find", "head", "sort"]
)
def test_inventory_never_executes_tools_from_the_image_path(
    tmp_path, monkeypatch, tool, position
):
    """Guard d4cbbc3c: root never starts a later agent-written PATH tool."""
    folder = tmp_path / "bin"
    folder.mkdir()
    marker = tmp_path / "executed"
    planted = folder / tool
    planted.write_text(
        f"#!/bin/sh\nprintf planted > {shlex.quote(str(marker))}\nexit 1\n"
    )
    planted.chmod(0o755)
    proc = tmp_path / "proc" / "self"
    proc.mkdir(parents=True)
    (proc / "status").write_text("CapBnd:\t0000000000001000\n")
    fs = tmp_path / "fs"
    fs.mkdir()
    (fs / "candidate").write_text("dummy")
    (fs / "candidate").chmod(0o755)
    monkeypatch.chdir(folder)
    # Omit system tools from the inherited PATH for the missing-tool and
    # relative spellings. The protected route uses its own fixed candidates.
    path = {
        "first": f"{folder}:/usr/bin:/bin",
        "last": f"/usr/bin:/bin:{folder}",
        "relative": ".",
        "empty": "",
    }[position]
    cmd = fp._root_cmd(proc=str(proc.parent), scan_all=True, fs_root=str(fs))
    result = _run(
        _route(
            cmd,
            required=("awk", "find", "head", "sort"),
            optional=("setpriv", "getcap", "python3"),
        ),
        path,
    )
    assert not marker.exists(), (tool, position, result.stdout, result.stderr)
    assert result.returncode == 0 and "BF fwpower-root net_admin=1" in result.stdout


@pytest.mark.parametrize("tool", [*fp._PROBE_TOOLS, "nft"])
@pytest.mark.parametrize("position", ["first", "last", "relative", "empty"])
def test_cleanup_never_executes_a_tool_supplied_by_the_probe(
    tmp_path, monkeypatch, tool, position
):
    """Guard d4cbbc3c for every IPv4, IPv6, nft and legacy spelling."""
    marker = tmp_path / "executed"
    planted = tmp_path / tool
    planted.write_text(f"#!/bin/sh\nprintf planted > {shlex.quote(str(marker))}\n")
    planted.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    path = {
        "first": f"{tmp_path}:/usr/bin:/bin",
        "last": f"/usr/bin:/bin:{tmp_path}",
        "relative": ".",
        "empty": "",
    }[position]
    result = _run(
        _route(fp._cleanup_cmd("BFFWtest"), optional=(*fp._PROBE_TOOLS, "nft")), path
    )
    assert not marker.exists(), (tool, position, result.stderr)
    assert result.returncode == 0


@pytest.mark.parametrize(
    "kind", ["writable-file", "writable-parent", "symlink", "relative-symlink", "hop"]
)
def test_root_tool_validation_covers_the_whole_path(tmp_path, monkeypatch, kind):
    """Guard d4cbbc3c with the shared metadata lexical/link validator."""
    system = tmp_path / "system"
    system.mkdir()
    writable = tmp_path / "writable"
    writable.mkdir(mode=0o777)
    writable.chmod(0o777)
    marker = tmp_path / "executed"
    tool = system / "getcap"
    target = writable / "payload"
    target.write_text(f"#!/bin/sh\nprintf planted > {shlex.quote(str(marker))}\n")
    target.chmod(0o755)
    if kind == "writable-file":
        tool.write_text(target.read_text())
        tool.chmod(0o777)
    elif kind == "writable-parent":
        tool.write_text(target.read_text())
        tool.chmod(0o755)
        system.chmod(0o777)
    elif kind == "relative-symlink":
        tool.symlink_to("../writable/payload")
    elif kind == "hop":
        (writable / "hop").symlink_to(target)
        tool.symlink_to(writable / "hop")
    else:
        tool.symlink_to(target)
    monkeypatch.setattr(lockdown, "_CLOUD_METADATA_ROOT_PATH", str(system))
    from benchflow.sandbox import _base

    monkeypatch.setattr(_base, "SYSTEM_PATH", f"{system}:/usr/bin:/bin")
    # PT's route pins an explicit PATH; put the planted tool on that route.
    original_route = _base.with_system_path
    monkeypatch.setattr(
        _base,
        "with_system_path",
        lambda command: original_route(command, f"{system}:/usr/bin:/bin"),
    )
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_TRUSTED_UIDS", ("0", str(os.getuid()))
    )
    command = '"$bf_getcap"' if hasattr(fp, "_root_route") else "getcap"
    result = _run(_route(command, required=("getcap",)), str(system))
    assert not marker.exists()
    assert result.returncode == 126
    assert "BF fwpower-error tool=getcap" in result.stdout


def test_root_tool_environment_is_built_from_scratch():
    """Guard d4cbbc3c against unenumerated tool-specific startup variables."""
    command = 'printf "%s:%s:%s" "${S3_DUMMY-unset}" "${PYTHONPATH-unset}" "$HOME"'
    result = _run(
        _route(command), "/app/bin", S3_DUMMY="present", PYTHONPATH="/app", HOME="/app"
    )
    assert result.returncode == 0
    assert result.stdout == "unset:unset:/app"


def test_root_route_preserves_the_backend_environment_cleanup(tmp_path):
    """Guard 4f5190af: the root route must not replace the backend trap owner."""
    from benchflow.sandbox._base import wrap_command_with_env_file

    command = wrap_command_with_env_file(
        {"S3_DUMMY": "fixture"},
        _route("printf done"),
        env_path_prefix=str(tmp_path / "environment-"),
    )
    result = _run(command, "/usr/bin:/bin")
    assert result.returncode == 0 and result.stdout == "done"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("reader", ["getcap", "python3", "none"])
async def test_honest_shared_launch_does_not_require_capability_readers(reader):
    """Guard d4cbbc3c's minimal-image compatibility on first/reconnect."""

    class Env(AuditEnv):
        supports_system_shell = True

        @receipt_aware
        async def exec(self, command, **kwargs):
            result = await super().exec(command, **kwargs)
            result.stdout = result.stdout.replace("caps=getcap", f"caps={reader}")
            return result

    env = Env(OUTCOMES["safe"])
    for _ in range(2):
        record, refusal = await fp.check_firewall_power(
            env, "agent", holds=("denylist",)
        )
        assert refusal is None and record["state"] == "disarmed"
        assert record["file_capabilities_checked"] is (reader != "none")
