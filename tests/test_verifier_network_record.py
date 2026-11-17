"""The network verdict's record survives a test.sh that takes /logs/verifier.

FDB's test.sh opens with ``chown root:root /logs/verifier && chmod 755
/logs/verifier``. On Docker that folder is the trial's bind-mounted
``verifier/``: the host could still read the reward, so the verifier never
handed the folder back, and ``verifier-network.json`` (the network verdict and
test.sh's launch line, CapBnd/CapEff/NoNewPrivs) failed to write without a
word. The verifier now hands the mounted outputs to the host user when the host
cannot add files there either. Here the folder's mode stands in for the root
ownership a test user cannot give away; the real-Docker run is in the lane's
report.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from benchflow.sandbox import verifier_network as vn
from tests.test_policy_reproof import (
    ROOT_HELD,
    _held,
    _install,
    _sandbox,
    _start,
    _verifier,
    _verifier_entry,
)

pytestmark = pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0,
    reason="root writes through any folder mode",
)


def _run(tmp_path: Path, *, takes_the_folder: bool):
    box = _sandbox(tmp_path)
    _install(box, _start())
    _install(box, _verifier_entry())
    folder = tmp_path / "rollout" / "verifier"
    handed: list[tuple[str, bool]] = []

    async def chown_to_host_user(path, *, recursive=False, service="main"):
        # The sandbox's chown to the host user: the folder is the host's again.
        handed.append((path, recursive))
        folder.chmod(0o755)

    box._chown_to_host_user = chown_to_host_user

    def as_fdb() -> None:
        # The host may read the reward, but not add a file.
        folder.chmod(0o555)

    verifier, paths = _verifier(
        tmp_path, box, _held(), during_test=as_fdb if takes_the_folder else None
    )
    return verifier, paths, handed, folder


@pytest.mark.asyncio
async def test_a_test_sh_that_takes_the_outputs_folder_still_gets_its_record(
    tmp_path,
):
    verifier, paths, handed, folder = _run(tmp_path, takes_the_folder=True)
    try:
        result = await verifier.verify()
    finally:
        folder.chmod(0o755)

    assert result.rewards == {"reward": 1.0}
    assert handed == [("/logs/verifier", True)]
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    # test.sh's launch: no NET_ADMIN, NET_RAW or SYS_PTRACE, no new privileges.
    assert record["launch"] == ROOT_HELD
    assert "CapBnd=00000000a80405fb CapEff=00000000a80405fb" in record["launch"]
    assert record["state"] == "held" and record["policy"]["state"] == "held"


@pytest.mark.asyncio
async def test_a_plain_test_sh_keeps_its_folder_and_its_record(tmp_path):
    verifier, paths, handed, _folder = _run(tmp_path, takes_the_folder=False)

    result = await verifier.verify()

    assert result.rewards == {"reward": 1.0}
    # Nothing changes owner where the host reads and writes already.
    assert handed == []
    record = json.loads((paths.verifier_dir / vn.VERIFIER_NETWORK_RECORD).read_text())
    assert record["launch"] == ROOT_HELD
    assert record["state"] == "held" and record["policy"]["state"] == "held"
