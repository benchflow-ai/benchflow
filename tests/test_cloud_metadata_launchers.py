"""S2-CONF-8: legitimate policy cannot clear a general privilege launcher."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests._receipt_fakes import receipt_answer
from tests.test_cloud_metadata_block import (
    AGENT_REFUSED,
    DELEGATES_CLEAN,
    ROOT_OK,
    FakeEnv,
)

LAUNCHERS = ("sudo", "sudoedit", "doas", "pkexec", "su", "ksu", "bbsuid")
SYSTEM_DIRS = ("/usr/bin", "/usr/sbin", "/bin", "/sbin")


def _inventory_report(path, privilege):
    # What root's inventory prints for one root-owned file at *path* with
    # *privilege* (its set-ID line, then a complete scan), without installing
    # fixtures into the runner's system directories. Now the
    # inventory only reports, and the host classifies (_delegate_report);
    # tests/test_cloud_metadata_delegate_inventory.py runs the reader itself.
    flags = {"setuid": "u-0", "setgid": "-g0", "both": "ug0", "": ""}[privilege]
    line = f"BF metadata-delegates-setid {flags} {path.encode().hex()}\n"
    return (line if flags else "") + DELEGATES_CLEAN


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize(
    "directory",
    (*SYSTEM_DIRS, "/usr/local/bin", "/usr/local/sbin", "/opt/tools", "/tmp/tools"),
)
@pytest.mark.parametrize("privilege", ["setuid", "setgid", "both", ""])
def test_policy_launchers_never_receive_a_fixed_path_exemption(
    launcher, directory, privilege
):
    """Guards S2-CONF-8 at 4745b42d across names, paths and set-ID modes."""
    from benchflow.sandbox.lockdown import _delegate_report

    path = f"{directory}/{launcher}"
    report = _delegate_report(_inventory_report(path, privilege), 0)
    assert report.complete, report.problems
    assert bool(report.findings) is bool(privilege)
    assert all(found["path"] == path for found in report.findings)


@pytest.mark.parametrize("helper", ["passwd", "chage", "unix_chkpwd", "wall", "ping"])
@pytest.mark.parametrize("directory", SYSTEM_DIRS)
def test_narrow_distribution_helpers_keep_their_exemption(helper, directory):
    """Guards S2-CONF-8 at 4745b42d without banning narrow system helpers."""
    from benchflow.sandbox.lockdown import _delegate_report

    report = _delegate_report(_inventory_report(f"{directory}/{helper}", "setuid"), 0)
    assert report.complete, report.problems
    assert report.findings == []


@pytest.mark.asyncio
@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("directory", SYSTEM_DIRS)
@pytest.mark.parametrize(
    "status", ["0", "1", "2", "3", "4", "5", "6", "missing", "timeout"]
)
@pytest.mark.parametrize("supports_system_shell", [True, False])
async def test_policy_launchers_refuse_every_install_outcome_and_reconnect(
    launcher, directory, status, supports_system_shell
):
    """Guards S2-CONF-8 at 4745b42d on both shared backend exec contracts."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    report = _inventory_report(f"{directory}/{launcher}", "setuid")

    class LauncherEnv(FakeEnv):
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-delegates" in command:
                assert user == "root"
                assert kwargs.get("system_shell", False) is supports_system_shell
                self.calls.append((command, user))
                return receipt_answer(
                    command, SimpleNamespace(stdout=report, return_code=0)
                )
            result = await super().exec(command, user=user, **kwargs)
            if len(self.calls) == 1 and status == "timeout":
                raise TimeoutError("dummy install timeout")
            return result

    root = (
        "" if status == "missing" else ROOT_OK.replace("status=0", f"status={status}")
    )
    # Initial setup and reconnect both invoke the shared coordinator anew.
    for _ in range(2):
        env = LauncherEnv(root, AGENT_REFUSED, "BF metadata-hits 1\n")
        env.supports_system_shell = supports_system_shell
        with pytest.raises(CloudMetadataPowerError, match="delegate") as refused:
            await block_cloud_metadata(env, "agent")
        # Positive evidence, named, not a failed inventory.
        assert f"{directory}/{launcher} (setuid root)" in str(refused.value)
