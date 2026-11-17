"""The firewall tests' /usr/bin link folders (tests/_system_bin.py)."""

from __future__ import annotations

import os
from pathlib import Path

from tests import (
    test_c8_review_capture,
    test_cloud_metadata_block,
    test_firewall_power,
    test_sdk_lockdown,
)
from tests._system_bin import SOURCES, system_bin

# One folder per filter, however many tests the worker runs:
# test_cloud_metadata_block's, test_firewall_power's, test_sdk_lockdown's and
# test_c8_review_capture's. (test_verifier_script_shell's _bin_dir stays per
# test: its tests remove and plant tools in that folder.)
FILTERS = 4


def _link_folders(root: Path) -> list[Path]:
    """The folders under ``root`` holding links into the host's bin folders."""
    found = []
    for folder, dirs, files in os.walk(root):
        for name in dirs + files:
            path = os.path.join(folder, name)
            if os.path.islink(path) and os.readlink(path).startswith(
                tuple(f"{d}/" for d in SOURCES)
            ):
                found.append(Path(folder))
                break
    return found


def test_a_worker_builds_one_link_folder_per_filter(tmp_path):
    """Guards PR #190: each test built its own folder of links to every file
    in /usr/bin and /bin (about 1,150 on Ubuntu 24.04, more on GitHub's
    runner image), and pytest keeps every test's folder, so the unit suite
    ran CI's runner out of inodes. Forty tests of one worker, ten for each
    of the four helpers, now share one folder per filter."""
    worker = tmp_path / "popen-gw0"
    for i in range(10):
        dirs = {kind: worker / f"test_{kind}{i}" for kind in "bplc"}
        for test_dir in dirs.values():
            test_dir.mkdir(parents=True)
        test_cloud_metadata_block.Sandbox(dirs["b"])._system_bin()
        test_firewall_power.Stub(dirs["p"])
        test_sdk_lockdown.TestSandboxDaemonGuard._system_bin(dirs["l"])
        test_c8_review_capture._system_without_python(dirs["c"])
    folders = _link_folders(worker)
    assert len(folders) <= FILTERS, sorted(str(f.relative_to(worker)) for f in folders)
    tools = {p.name for d in SOURCES for p in Path(d).iterdir()}
    links = sum(1 for f in folders for p in f.iterdir() if p.is_symlink())
    assert links <= FILTERS * len(tools)


def test_the_link_folder_holds_no_link_to_a_folder(tmp_path):
    """Guards PR #190: Debian's /usr/bin/X11 is a link to "." (GitHub's runner
    image has one), and a link to it in the folder let any walk that follows
    links go X11/X11/... until the paths were too long. Links to folders
    are left out; files, links to files and dangling links stay."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "tool").write_text("#!/bin/sh\n")
    (bin_dir / "tool").chmod(0o755)
    (bin_dir / "alias").symlink_to("tool")
    (bin_dir / "gone").symlink_to("/nonexistent/gone")
    (bin_dir / "X11").symlink_to(".")
    test_dir = tmp_path / "popen-gw0" / "test_0"
    test_dir.mkdir(parents=True)
    folder = system_bin(test_dir, set(), (), sources=(str(bin_dir),))
    assert sorted(p.name for p in folder.iterdir()) == ["alias", "gone", "tool"]
    assert [d for d, _, _ in os.walk(folder, followlinks=True)] == [str(folder)]
