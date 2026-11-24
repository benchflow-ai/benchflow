"""the pinned bfx under gVisor, through Modal's route.

Modal runs sandboxes on gVisor. Here gVisor is ``runsc do`` on this machine
(``BENCHFLOW_RUNSC``: the command that starts runsc as root, for example
``sudo -n /opt/gvisor/runsc``; unset, the row is skipped). Each bfx call
goes through ``ModalSandbox.bfx_call`` and starts the pinned binary in a
fresh gVisor sandbox: its own report, the mount table it reads, a walk that
finds a planted link, a removal, and a walk of /usr with no errors. With
``BENCHFLOW_M6_OUT`` set, what it measured is written there as JSON.
"""

from __future__ import annotations

import json
import os
import shlex

import pytest

from benchflow._bfx import pins

RUNSC = os.environ.get("BENCHFLOW_RUNSC", "").strip()

pytestmark = [
    pytest.mark.bfx_refusal,
    pytest.mark.skipif(not RUNSC, reason="BENCHFLOW_RUNSC is not set"),
]


async def test_m6_the_pinned_bfx_walks_under_gvisor(tmp_path):
    """M6. Mutant: none in code (a measurement of the platform); the row
    is red wherever Modal has no bfx route."""
    from tests.test_modal_bfx_route import local_box

    root = tmp_path / "runsc-root"
    root.mkdir()
    prefix = [*shlex.split(RUNSC), "--root", str(root), "--network=none", "do"]
    box, starts = local_box(prefix)
    found: dict = {}

    version = await box.bfx_call("version", {}, timeout_sec=120)
    found["version"] = version
    assert version.get("sha256") == pins.SHA256["x86_64"], version
    assert version.get("procfs") is True, version

    mounts = await box.bfx_call("scan", {"mounts": True, "acl": False}, timeout_sec=120)
    found["mounts"] = [m.get("mount_point") for m in mounts.get("mounts") or ()]
    assert "/" in found["mounts"], found["mounts"]

    tree = tmp_path / "tree"
    (tree / "app").mkdir(parents=True)
    os.symlink("/logs/verifier/reward.txt", tree / "app" / "grade.log")
    walked = await box.bfx_call(
        "scan",
        {"walks": [{"root": str(tree), "depth": 64}], "mounts": False, "acl": False},
        timeout_sec=120,
    )
    links = {
        e.get("path"): e.get("link")
        for e in walked.get("entries") or ()
        if e.get("type") == "link"
    }
    found["links"] = links
    assert links == {str(tree / "app" / "grade.log"): "/logs/verifier/reward.txt"}

    removed = await box.bfx_call(
        "remove", {"root": str(tree), "path": "app/grade.log"}, timeout_sec=120
    )
    found["remove"] = removed

    usr = await box.bfx_call(
        "scan",
        {
            "walks": [{"root": "/usr", "depth": 4096}],
            "mounts": False,
            "acl": False,
            "max_entries": 4_000_000,
            "max_bytes": 512 << 20,
        },
        timeout_sec=600,
    )
    found["usr_entries"] = len(usr.get("entries") or ())
    found["usr_errors"] = (usr.get("errors") or [])[:5]
    found["starts"] = len(starts)
    out = os.environ.get("BENCHFLOW_M6_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(found, f, indent=1, default=str)
    assert found["usr_entries"] > 1000, found["usr_entries"]
    assert not found["usr_errors"], found["usr_errors"]
