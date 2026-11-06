"""Derive cached-tree pins from the verified CPython archives (GNU tar/Linux).

Run with --work-dir pointing at a private scratch directory. Prints the
archive-digest to tree-digest mapping for _staged_python_layer.TREE_SHA256.
No code from the archives is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from benchflow.sandbox import _delegate_disarm as stage
from benchflow.sandbox._staged_python_layer import TAR_OPTIONS, TREE_SUFFIX


def derive(work_dir: Path) -> dict[str, str]:
    result = {}
    env = dict(os.environ, LC_ALL="C")
    for name in ("TAR_OPTIONS", "POSIXLY_CORRECT"):
        env.pop(name, None)
    for arch, libc in stage.STAGED_PYTHON_SHA256:
        archive = stage.host_tarball(arch, libc)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                path = Path(member.name)
                if (
                    not path.parts
                    or path.parts[0] != "python"
                    or ".." in path.parts
                    or not (member.isfile() or member.isdir() or member.issym())
                    or member.uid != 0
                    or member.gid != 0
                ):
                    raise ValueError(f"unexpected archive entry: {member.name}")
        with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
            subprocess.run(
                [
                    "tar",
                    "--no-same-owner",
                    "--no-same-permissions",
                    "-xzf",
                    str(archive),
                    "-C",
                    tmp,
                ],
                check=True,
                env=env,
                umask=0o022,
            )
            for path in Path(tmp).rglob("*"):
                if path.is_symlink() and not path.resolve().is_relative_to(tmp):
                    raise ValueError(f"link leaves the pinned tree: {path}")
            stream = subprocess.run(
                [
                    "tar",
                    *TAR_OPTIONS.split(),
                    "--owner=0",
                    "--group=0",
                    "-C",
                    tmp,
                    "-cf",
                    "-",
                    ".",
                ],
                check=True,
                env=env,
                capture_output=True,
            )
            if stream.stderr:
                raise ValueError(stream.stderr.decode())
            result[stage.STAGED_PYTHON_SHA256[arch, libc]] = hashlib.sha256(
                stream.stdout + TREE_SUFFIX.encode()
            ).hexdigest()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(derive(args.work_dir), indent=2))
