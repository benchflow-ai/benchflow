"""Guards the fix against 800ecffd1, which applied the Win32 name rules on
every controller: one ``aux.py`` or ``A:1022428818870.json`` in a Linux
workspace failed the evidence capture, so the trial lost its review and its
score, on every retry. On a POSIX controller such a workspace is captured,
bundled and installed for the reviewer whole, as in 0.7.11."""

import hashlib
import os
import subprocess
import sys

import pytest

from benchflow.review.evidence import (
    _ADMISSION_SCRIPT,
    _UNPACK_SCRIPT,
    capture_workspace,
    prepare_workspace_upload,
    validate_workspace,
)
from tests.test_separate_verifier import LocalTransport

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Windows reserves these names")

# Names live rollouts of one benchmark hold that the 0.9 candidate refused,
# that benchmark's own unicode fixture, and a Next.js build chunk.
NAMES = [
    "work/aux.py",
    "work/aux.json",
    "work/test_aux/aux.in",
    "output/aux",
    "research/literature/A:1022428818870.json",
    "research/ref_0004-6361:20041453.json",
    'research/ti_"randomization.xml',
    "notes/con.txt",
    "web/_next/static/chunks/0jr8wo_7ak~7n.js",
    'emoji 🚀 dir/it\'s "quoted" & spaced.txt',
]


async def test_windows_reserved_names_reach_the_reviewer(tmp_path):
    workspace = tmp_path / "box" / "work-root"
    for name in NAMES:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name + "\n")

    bundle = tmp_path / "evidence"
    manifest = await capture_workspace(LocalTransport(), str(workspace), bundle)
    files = {entry.path: entry for entry in manifest.entries if entry.kind == "file"}
    assert sorted(files) == sorted(NAMES)
    for name in NAMES:
        data = (name + "\n").encode()
        assert (bundle / "workspace" / name).read_bytes() == data
        assert files[name].sha256 == hashlib.sha256(data).hexdigest()
    validate_workspace(bundle / "workspace", manifest)

    # What the reviewer sandbox gets: the upload archive, unpacked and then
    # admitted by the same stdlib scripts the review runs there.
    upload = tmp_path / "upload"
    prepare_workspace_upload(bundle, upload / "workspace.tar")
    manifest_path = upload / "workspace-manifest.json"
    manifest_path.write_bytes((bundle / "manifest.json").read_bytes())
    installed = tmp_path / "reviewer" / "workspace"
    for script, args in (
        (_UNPACK_SCRIPT, [upload / "workspace.tar", installed, manifest_path]),
        (_ADMISSION_SCRIPT, [installed, manifest_path]),
    ):
        subprocess.run(
            [sys.executable, "-I", "-c", script, *map(str, args)],
            check=True,
            capture_output=True,
        )
    for name in NAMES:
        assert (installed / name).read_text() == name + "\n"
