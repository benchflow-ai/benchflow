"""S2-CONF-8: legitimate policy cannot clear a general privilege launcher."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.test_cloud_metadata_block import AGENT_REFUSED, ROOT_OK, FakeEnv
from tests.test_cloud_metadata_delegates import sandbox as sandbox

LAUNCHERS = ("sudo", "sudoedit", "doas", "pkexec", "su", "ksu", "bbsuid")
SYSTEM_DIRS = ("/usr/bin", "/usr/sbin", "/bin", "/sbin")


def _inventory_file(box, path, privilege):
    # Supply find's file arguments to the real generated classifier, without
    # installing fixtures into the runner's system directories. getcap sees
    # no xattrs. Both commands remain inspection-only, never running the file.
    box.script("getcap", "#!/bin/sh\nexit 0\n")
    box.script(
        "find",
        '#!/bin/sh\n[ -n "$INVENTORY_PRIVILEGE" ] || exit 0\n'
        'case " $* " in *" -perm "*) ;; *) exit 0;; esac\n'
        'while [ "$1" != -exec ]; do shift; done\nshift\n'
        'exec "$1" "$2" "$3" "$4" "$INVENTORY_FILE"\n',
    )
    return {"INVENTORY_FILE": path, "INVENTORY_PRIVILEGE": privilege}


@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize(
    "directory",
    (*SYSTEM_DIRS, "/usr/local/bin", "/usr/local/sbin", "/opt/tools", "/tmp/tools"),
)
@pytest.mark.parametrize("privilege", ["setuid", "setgid", "both", ""])
@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
def test_policy_launchers_never_receive_a_fixed_path_exemption(
    sandbox, launcher, directory, privilege, shell
):
    """Guards S2-CONF-8 at 4745b42d across names, paths and set-ID modes."""
    from benchflow.sandbox.lockdown import _cloud_metadata_delegate_check_cmd

    environment = _inventory_file(sandbox, f"{directory}/{launcher}", privilege)
    report = sandbox.run(_cloud_metadata_delegate_check_cmd(), shell, **environment)
    assert report == f"BF metadata-delegates power={int(bool(privilege))}\n"


@pytest.mark.parametrize("helper", ["passwd", "chage", "unix_chkpwd", "wall", "ping"])
@pytest.mark.parametrize("directory", SYSTEM_DIRS)
def test_narrow_distribution_helpers_keep_their_exemption(sandbox, helper, directory):
    """Guards S2-CONF-8 at 4745b42d without banning narrow system helpers."""
    from benchflow.sandbox.lockdown import _cloud_metadata_delegate_check_cmd

    environment = _inventory_file(sandbox, f"{directory}/{helper}", "setuid")
    assert sandbox.run(_cloud_metadata_delegate_check_cmd(), **environment) == (
        "BF metadata-delegates power=0\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("launcher", LAUNCHERS)
@pytest.mark.parametrize("directory", SYSTEM_DIRS)
@pytest.mark.parametrize(
    "status", ["0", "1", "2", "3", "4", "5", "6", "missing", "timeout"]
)
@pytest.mark.parametrize("supports_system_shell", [True, False])
async def test_policy_launchers_refuse_every_install_outcome_and_reconnect(
    sandbox, launcher, directory, status, supports_system_shell
):
    """Guards S2-CONF-8 at 4745b42d on both shared backend exec contracts."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    environment = _inventory_file(sandbox, f"{directory}/{launcher}", "setuid")

    class LauncherEnv(FakeEnv):
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-delegates" in command:
                assert user == "root"
                assert kwargs.get("system_shell", False) is supports_system_shell
                self.calls.append((command, user))
                return SimpleNamespace(
                    stdout=sandbox.run(command, **environment), return_code=0
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
        with pytest.raises(CloudMetadataPowerError, match="delegate"):
            await block_cloud_metadata(env, "agent")
