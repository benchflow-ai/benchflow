"""Batch 8 regressions for ordinary images and the kernel exec barrier."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests._receipt_fakes import receipt_aware
from tests.test_cloud_metadata_block import AGENT_REFUSED, ROOT_OK, FakeEnv

# The delegate inventory's report from PT's protected Python on a clean image.
PYTHON_CLEAN = (
    "BF metadata-delegates-scan reader=python3 complete=1 files=812 vanished=0 "
    "errors=0 why=-\n"
)

SAFE = (
    "BF metadata-construction uid=1000 nnp=1 "
    "bnd=0000000000000000 amb=0000000000000000\n"
)


@pytest.mark.parametrize("shell", ["/bin/sh", "/bin/bash"])
def test_root_reader_observes_the_live_shared_launch(monkeypatch, shell):
    """Guards batch 8 at 70400f55: /proc is read outside the dropped child.

    Model the uid-drop only, retaining the real FIFO lifetime and /proc
    observation. Full Docker startup exercises the genuine setpriv launch.
    """
    from benchflow.sandbox import lockdown

    monkeypatch.setattr(
        lockdown,
        "build_priv_drop_cmd",
        lambda command, user: "exec /bin/sh -c " + shlex.quote(command),
    )
    result = subprocess.run(
        [shell, "-c", lockdown._cloud_metadata_construction_cmd(str(os.getuid()))],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    fields = lockdown._report_fields(result.stdout)
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    assert fields["uid"] == str(os.getuid())
    assert fields["nnp"] == status["NoNewPrivs"].strip()
    assert fields["bnd"] == status["CapBnd"].strip()
    assert fields["amb"] == status["CapAmb"].strip()


@pytest.mark.asyncio
@pytest.mark.parametrize("system_shell", [True, False])
@pytest.mark.parametrize(
    "status", ["0", "1", "2", "3", "4", "5", "6", "missing", "timeout"]
)
@pytest.mark.parametrize(
    "broken",
    [
        "nnp",
        "net-admin",
        "net-raw",
        "ambient",
        "root",
        "empty",
        "duplicate",
        "extra",
        "malformed",
        "nonzero",
        "boolean",
        "timeout",
    ],
)
async def test_construction_is_mandatory_before_every_install_and_reconnect(
    system_shell,
    status,
    broken,
):
    """Guards batch 8 at 70400f55 for both backend exec contracts and retries."""
    from benchflow.sandbox.lockdown import CloudMetadataPowerError, block_cloud_metadata

    outputs = {
        "nnp": SAFE.replace("nnp=1", "nnp=0"),
        "net-admin": SAFE.replace("bnd=0000000000000000", "bnd=0000000000001000"),
        "net-raw": SAFE.replace("bnd=0000000000000000", "bnd=0000000000002000"),
        "ambient": SAFE.replace("amb=0000000000000000", "amb=0000000000000001"),
        "root": SAFE.replace("uid=1000", "uid=0"),
        "empty": "",
        "duplicate": SAFE * 2,
        "extra": SAFE + "ignored\n",
        "malformed": SAFE.replace("bnd=0000000000000000", "bnd=unknown"),
    }

    class Env(FakeEnv):
        supports_system_shell = system_shell

        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-construction" in command:
                assert user == "root"
                assert kwargs.get("system_shell", False) is system_shell
                if broken == "timeout":
                    raise TimeoutError("dummy construction timeout")
                return SimpleNamespace(
                    stdout=outputs.get(broken, SAFE),
                    return_code={"nonzero": 1, "boolean": False}.get(broken, 0),
                )
            return await super().exec(command, user=user, **kwargs)

    for _ in range(2):
        env = Env(ROOT_OK.replace("status=0", f"status={status}"), AGENT_REFUSED)
        with pytest.raises(CloudMetadataPowerError, match="construction"):
            await block_cloud_metadata(env, "agent")
        assert not env.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("system_shell", [True, False])
@pytest.mark.parametrize("missing", ["getcap", "readlink", "find", "all"])
async def test_missing_optional_tools_keep_tcp_proof_and_startup(system_shell, missing):
    """Guards batch 8 at 70400f55: the agent check's missing tools keep the
    TCP proof and startup. Now the separate delegate inventory must
    still finish: here PT's protected Python reads the file capabilities of
    an image without getcap, readlink or find, and finds nothing."""
    from benchflow.sandbox.lockdown import block_cloud_metadata

    class Env(FakeEnv):
        supports_system_shell = system_shell

        @receipt_aware
        async def exec(self, command, user=None, **kwargs):
            if "BF metadata-construction" in command:
                assert user == "root"
                assert kwargs.get("system_shell", False) is system_shell
                return SimpleNamespace(stdout=SAFE, return_code=0)
            if "BF metadata-delegates" in command:
                return SimpleNamespace(stdout=PYTHON_CLEAN, return_code=0)
            if "BF metadata-connect" in command:
                assert user == "root"
                assert "--no-new-privs" in command
                assert "--bounding-set=-net_admin,-net_raw" in command
                return SimpleNamespace(
                    stdout="BF metadata-connect refused\n", return_code=0
                )
            normalize = getattr(
                self, "agent_command", lambda command, user: (command, user)
            )
            command, user = normalize(command, user)
            result = await super().exec(command, user=user, **kwargs)
            if user == "agent":
                result.stdout = ""
                result.return_code = 7
                result.stderr = f"{missing}: unavailable"
            return result

    for _ in range(2):
        record = await block_cloud_metadata(
            Env(ROOT_OK, AGENT_REFUSED, "BF metadata-hits 1\n"), "agent"
        )
        assert record["state"] == "on"
        assert record["agent_privilege_audit"] == "unavailable"
        assert record["delegate_privilege_audit"] == "complete"
        assert record["delegate_inventory"]["reader"] == "python3"
