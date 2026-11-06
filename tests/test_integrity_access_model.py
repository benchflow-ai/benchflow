"""The access assessment against a brute-force reference, on generated trees.

``tests/integrity_fs_model.py`` resolves paths as the kernel does and decides,
from the kernel's permission rules alone, whether the agent user can change or
read what root finds at a protected path. Each generated tree of folders,
files and links (owners, groups, modes, sticky bits, relative and absolute
targets, loops) is printed as the probe prints it, and the real host-side
assessment must never certify what the reference shows open. Where the
reference is exact (the path resolves to a file, is missing, or does not
resolve) the two must agree; for a folder the assessment may only be more
cautious. ``tests/test_integrity_access_real_probe.py`` checks that the real
probe script prints the same facts for these trees.
"""

from __future__ import annotations

import random

import pytest

from benchflow.integrity.access import parse_probe, path_state
from tests.integrity_fs_model import random_case

SEED = 20261018
CASES = 4000


def _assess(tree, path, write):
    probe = parse_probe(tree.probe_output([path]))
    return path_state(probe, path, write=write) == "open"


@pytest.mark.parametrize("write", [False, True], ids=["read", "write"])
def test_the_assessment_never_certifies_what_the_agent_can_change(write):
    rng = random.Random(SEED + write)
    exact = cautious = 0
    for case in range(CASES):
        tree, path = random_case(rng)
        reference = (
            tree.reference_write_open(path) if write else tree.reference_read_open(path)
        )
        assessed = _assess(tree, path, write)
        context = f"case {case}: {path} in {sorted(tree.nodes.items())}"
        if reference:
            assert assessed, f"certified an exposure, {context}"
        if tree.final_kind(path) in {"file", "absent", "unresolved"}:
            assert assessed == reference, f"not exact, {context}"
            exact += 1
        elif assessed and not reference:
            cautious += 1
    # The generator reaches every kind of case.
    assert exact > CASES // 4
    assert cautious < CASES // 2


def test_the_generator_covers_every_awkward_name():
    """Folders, files and link targets all meet every name a shell or a
    line-based record can lose (newlines, tabs, spaces, backslashes, a
    leading dash, a non-UTF-8 byte)."""
    from tests.integrity_fs_model import _NAMES

    rng = random.Random(SEED)
    seen: dict[str, set[str]] = {name: set() for name in _NAMES}
    for _ in range(CASES):
        tree, _path = random_case(rng)
        for path, node in tree.nodes.items():
            for name in _NAMES:
                if path.rsplit("/", 1)[-1] == name:
                    seen[name].add(node.kind)
                if node.kind == "link" and name in node.target.split("/"):
                    seen[name].add("target")
    for name, kinds in seen.items():
        assert {"dir", "file", "link", "target"} <= kinds, (name, kinds)
