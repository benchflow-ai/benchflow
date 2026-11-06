"""Real mount regression inputs from R10/R11, exercised on the original plain API.

Run only inside the disposable topology container prepared by the R13 receipt.
This tests controller publication against real mounts, not Docker launch policy.
"""

import os
from pathlib import Path

import pytest

from benchflow._utils import reachable_write as rw
from benchflow.sandbox._download import private_download


class Receiver:
    _mount_inventory_error = None

    @private_download(directory=False)
    async def download_file(self, source_path, target_path):
        target_path.write_bytes(b"downloaded")


@pytest.fixture
def topology():
    if os.environ.get("BENCHFLOW_R13_TOPOLOGY") != "/topo":
        pytest.skip("requires owned real-mount topology container")
    return Path("/topo")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case,source,planted,target",
    [
        ("R10-2", "srv/shared", "work/output", "exports/subtree/results/marker.json"),
        (
            "REVIEW-1",
            "srv/shared",
            "work/data/output",
            "exports/nested/results/marker.json",
        ),
        (
            "REVIEW-2",
            "dataroot/volumes/v/_data",
            "exports/volume",
            "exports/volume/results/marker.json",
        ),
        (
            "REVIEW-3",
            "srv/shared",
            "work/output",
            "exports/ci/OUTPUT/results/marker.json",
        ),
        (
            "R11-1",
            "dataroot/volumes/v/_data",
            "exports/enclosing/_data",
            "exports/enclosing/_data/results/marker.json",
        ),
        (
            "R11-3",
            "srv/data/volumes/v/_data",
            "exports/case-sensitive",
            "exports/case-sensitive/results/marker.json",
        ),
        ("R10-3", "dataroot/volumes/b/_data", "work", "srv/shared/results/marker.json"),
        ("R10-4", None, "work", "srv/shared/results/marker.json"),
        ("R11-2", "dataroot/volumes/b/_data", "work", "srv/shared/results/marker.json"),
        ("R9-2", "srv/shared", "work", "srv/shared/results/marker.json"),
    ],
)
async def test_original_plain_mount_alias_input(
    topology, case, source, planted, target
):
    """Guards the release blockers at b9d614004 and the R9-2 regression at 7c24f4d88."""
    marker = topology / "private/marker.json"
    marker.write_bytes(b"original")
    # The old boundary registers the original mount, never its later alias.
    register = getattr(rw, "register_mount_source", None)
    if register is not None:
        rw._REACHABLE_IDS.clear()
        if source is not None:
            register(topology / source)
    link = topology / planted / "results"
    link.unlink(missing_ok=True)
    link.symlink_to(topology / "private", target_is_directory=True)
    refused = False
    try:
        await Receiver().download_file("/answer", topology / target)
    except (OSError, ValueError):
        refused = True
    assert marker.read_bytes() == b"original", f"{case}: controller marker overwritten"
    assert refused, f"{case}: original unsafe destination was not refused"


@pytest.mark.asyncio
async def test_honest_other_filesystem_with_preagent_root(topology):
    """Caller-root control: a mount ABOVE the destination root is permitted."""
    root = topology / "exports/honest"
    if hasattr(rw, "pin_destination"):
        _pinned = rw.pin_destination(root)
    target = root / "output.json"
    await Receiver().download_file("/answer", target)
    assert target.read_bytes() == b"downloaded"
