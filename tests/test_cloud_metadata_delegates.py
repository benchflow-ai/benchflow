"""S2-CONF-7 regressions: failed operations cannot clear delegate privilege."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from tests._receipt_fakes import receipt_aware
from tests.test_cloud_metadata_block import (
    AGENT_REFUSED,
    DELEGATES_CLEAN,
    ROOT_OK,
    FakeEnv,
    Sandbox,
)

TOOLS = (
    "iptables",
    "iptables-nft",
    "iptables-legacy",
    "ip6tables",
    "ip6tables-nft",
    "ip6tables-legacy",
    "nft",
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    from benchflow.sandbox import lockdown

    box = Sandbox(tmp_path)
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_ROOT_PATH", f"{box.bin}:{box._system_bin()}"
    )
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_TRUSTED_UIDS", ("0", str(os.getuid()))
    )
    return box


def _isolate_inventory(box, monkeypatch):
    from benchflow.sandbox import lockdown

    inventory = getattr(lockdown, "_cloud_metadata_delegate_check_cmd", None)
    if inventory is not None:
        monkeypatch.setattr(
            lockdown,
            "_cloud_metadata_delegate_check_cmd",
            lambda **kwargs: inventory(str(box.root), **kwargs),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TOOLS)
@pytest.mark.parametrize(
    "placement",
    [
        "absolute",
        "relative",
        "empty",
        "spaces",
        "symlink",
        "chain",
        "conditional",
        "off-path",
        "usr/local/sbin",
        "usr/local/bin",
        "usr/sbin",
        "usr/bin",
        "sbin",
        "bin",
        "dev/shm",
        "run/private",
    ],
)
@pytest.mark.parametrize(
    "privilege",
    [
        "cap_net_admin=ep",
        "cap_net_admin=p",
        "cap_net_admin=i",
        "cap_setuid=ep",
        "cap_setpcap=ep",
        "cap_sys_admin=ep",
        "=ep",
        "cap_dac_override=ep",
        "cap_fowner=ep",
        "cap_net_bind_service=ep",
        "setuid",
        "setgid",
        "",
    ],
)
async def test_failed_wrapper_probe_cannot_clear_delegate_privilege(
    sandbox, monkeypatch, tool, placement, privilege
):
    """Guards S2-CONF-7 at 705249aa, including off-PATH/unprobed delegates."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    folder = sandbox.root / ("with spaces" if placement == "spaces" else "delegates")
    if "/" in placement or placement in ("sbin", "bin"):
        folder = sandbox.root / "system" / placement
    folder.mkdir(parents=True)
    delegate = folder / "ordinary.real"
    delegate.write_text(
        '#!/bin/sh\ncase "$*" in "flush ruleset"|"add table ip "*|"add table ip6 "*) '
        'printf dummy > "$DELEGATE_MARKER"; exit 0;; esac\nexit 1\n'
    )
    delegate.chmod({"setuid": 0o4755, "setgid": 0o2755}.get(privilege, 0o755))
    target = delegate
    if placement == "symlink":
        target = folder / "ordinary.link"
        target.symlink_to(delegate)
    if placement == "chain":
        target = folder / "second-wrapper"
        target.write_text(f'#!/bin/sh\nexec {shlex.quote(str(delegate))} "$@"\n')
        target.chmod(0o755)
    invocation = shlex.quote(str(target))
    if placement in ("relative", "empty"):
        invocation = "ordinary.real"
    elif placement == "off-path":
        folder.chmod(0o711)
    conditional = '[ "$1" = flush ] || exit 1\n' if placement == "conditional" else ""
    sandbox.script(tool, f'#!/bin/sh\n{conditional}exec {invocation} "$@"\n')
    # Fake xattrs only; real set-ID mode bits are inert on these dummy scripts.
    sandbox.script(
        "getcap",
        '#!/bin/sh\nverbose=; [ "$1" != -v ] || { verbose=1; shift; }; '
        'for f do if [ "$f" = "$PRIVILEGED_FILE" ] && [ -n "$DELEGATE_CAPS" ]; then '
        'printf "%s %s\\n" "$f" "$DELEGATE_CAPS"; '
        'elif [ -n "$verbose" ]; then printf "%s\\n" "$f"; fi; done\n',
    ).caps("0" * 16)
    _isolate_inventory(sandbox, monkeypatch)
    path = f"{folder}:{sandbox.bin}:{sandbox._system_bin()}"
    if placement == "relative":
        path = f".:{sandbox.bin}:{sandbox._system_bin()}"
    if placement == "empty":
        path = f":{sandbox.bin}:{sandbox._system_bin()}"
    marker = sandbox.root / "dummy-marker"
    environment = dict(
        PATH=path,
        PRIVILEGED_FILE=str(delegate),
        DELEGATE_CAPS=privilege if privilege not in ("setuid", "setgid", "") else "",
        DELEGATE_MARKER=str(marker),
    )

    class WrappedEnv(FakeEnv):
        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            command, user = self.agent_command(command, user)
            if "BF metadata-delegates" in command:
                self.calls.append((command, user))
                return SimpleNamespace(
                    stdout=sandbox.run(command, **environment), return_code=0
                )
            result = await super().exec(command, user=user, **kwargs)
            if user == "agent":
                result.stdout = sandbox.run(
                    f"cd {shlex.quote(str(folder))} && {command}", **environment
                )
                # The original named-tool audit really has no positive evidence.
                assert "held=0 power= connect=" in result.stdout
            return result

    env = WrappedEnv(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n")
    if privilege:
        with pytest.raises(CloudMetadataPowerError, match="delegate"):
            await block_cloud_metadata(env, "agent")
    else:
        record = await block_cloud_metadata(env, "agent")
        assert record["state"] == "unproven"  # no bash/timeout in the fixture
    assert not marker.exists(), "inventory must not execute delegates"
    # This is only a dummy marker, never an actual firewall or metadata call.
    sandbox.run(f"{shlex.quote(str(delegate))} flush ruleset", **environment)
    assert marker.read_text() == "dummy"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", ["0", "1", "2", "3", "4", "5", "6", "missing", "timeout"]
)
@pytest.mark.parametrize("supports_system_shell", [True, False])
@pytest.mark.parametrize(
    "broken", ["power", "empty", "duplicate", "nonzero", "boolean", "timeout"]
)
async def test_delegate_inventory_is_required_on_every_return_and_reconnect(
    status, supports_system_shell, broken
):
    """Guards S2-CONF-7 at 705249aa across backend contracts and failures.

    The inventory is required in fact: a timed-out, empty,
    repeated or failed report used to record "unavailable" and start the
    agent; now each refuses startup like positive evidence does, and the
    refusal carries its record."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    su = "BF metadata-delegates-setid u-0 " + b"/usr/bin/su".hex() + "\n"

    class InventoryEnv(FakeEnv):
        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-delegates" in command:
                assert user == "root"
                assert kwargs.get("system_shell", False) is supports_system_shell
                if broken == "timeout":
                    raise TimeoutError("dummy inventory timeout")
                return SimpleNamespace(
                    stdout={
                        "power": su + DELEGATES_CLEAN,
                        "empty": "",
                        "duplicate": DELEGATES_CLEAN * 2,
                    }.get(broken, DELEGATES_CLEAN),
                    return_code={"nonzero": 1, "boolean": False}.get(broken, 0),
                )
            result = await super().exec(command, user=user, **kwargs)
            if len(self.calls) == 1 and status == "timeout":
                raise TimeoutError("dummy install timeout")
            return result

    root = (
        "" if status == "missing" else ROOT_OK.replace("status=0", f"status={status}")
    )
    for _ in range(2):
        env = InventoryEnv(root, AGENT_REFUSED, "BF metadata-hits 1\n")
        env.supports_system_shell = supports_system_shell
        with pytest.raises(CloudMetadataPowerError, match="delegate") as refused:
            await block_cloud_metadata(env, "agent")
        record = refused.value.record
        assert record["state"] == "refused"
        if broken == "power":
            assert "/usr/bin/su (setuid root)" in str(refused.value)
            assert record["delegates"] == [
                {"path": "/usr/bin/su", "why": "setuid root"}
            ]
            assert record["delegate_privilege_audit"] == "complete"
        else:
            assert record["delegate_privilege_audit"] == "incomplete"
            assert "could not finish the file privilege inventory" in str(refused.value)


@pytest.mark.parametrize("helper", ["find", "getcap"])
@pytest.mark.parametrize("failure", ["missing", "nonzero"])
def test_delegate_enumeration_failure_cannot_report_clean(
    sandbox, monkeypatch, helper, failure
):
    """Guards S2-CONF-7 at 705249aa: incomplete inventory fails closed.

    The shell inventory now also needs sh, od, tr, mktemp and rm
    (it reports paths in hex); they are kept here, so the helper under test
    is what fails the inventory, as its report's reason says."""
    from benchflow.sandbox import lockdown

    isolated = sandbox.root / "inspection"
    isolated.mkdir()
    for name in ("find", "getcap"):
        sandbox.script(name, "#!/bin/sh\nexit 0\n")
        (isolated / name).write_bytes((sandbox.bin / name).read_bytes())
        (isolated / name).chmod(0o755)
    (isolated / "find").unlink()
    (isolated / "find").symlink_to(shutil.which("find"))
    # Keep inspection bootstrap utilities without the helper under test.
    for name in ("ls", "readlink", "sh", "od", "tr", "mktemp", "rm"):
        (isolated / name).symlink_to(shutil.which(name))
    (isolated / helper).unlink()
    if failure == "nonzero":
        (isolated / helper).write_text("#!/bin/sh\nexit 1\n")
        (isolated / helper).chmod(0o755)
    monkeypatch.setattr(lockdown, "_CLOUD_METADATA_ROOT_PATH", str(isolated))
    with pytest.raises(subprocess.CalledProcessError) as failed:
        sandbox.run(lockdown._cloud_metadata_delegate_check_cmd(str(sandbox.root)))
    assert failed.value.returncode == 7
    why = {
        ("find", "missing"): "why=tools",
        ("find", "nonzero"): "why=setid,caps",
        ("getcap", "missing"): "why=reader",
        ("getcap", "nonzero"): "why=caps",
    }[helper, failure]
    assert failed.value.stdout.rstrip("\n").endswith(why), failed.value.stdout
