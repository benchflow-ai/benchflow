"""A bind-mounted namespace in mountinfo is not a malformed record.

Docker's netns files and ``ip netns`` mounts show as ``nsfs`` records whose
root is ``net:[<inode>]``. They hold no files, so they cannot alias a
workspace, and a sandbox that has one must still verify. Every other record
keeps the strict grammar.
"""

import pytest

from benchflow.sandbox._mount_aliases import blocked_mount_aliases

ROOT = "1 0 0:1 / / rw - overlay overlay rw\n"
WORKSPACE = "2 1 0:2 / /app rw - ext4 /dev/sda1 rw\n"


@pytest.mark.parametrize("kind", ["net", "mnt", "pid_for_children", "cgroup"])
def test_namespace_mounts_are_not_aliases_and_do_not_refuse(kind):
    info = (
        ROOT
        + WORKSPACE
        + (
            f"80 1 0:5 {kind}:[4026532524] /run/docker/netns/7c1e769077a0 rw "
            "shared:221 - nsfs nsfs rw\n"
        )
    )
    assert blocked_mount_aliases(info, ("/app",)) == ("/app",)


@pytest.mark.parametrize(
    "record",
    [
        "80 1 0:5 net:[4026532524] /run/x rw - ext4 nsfs rw\n",
        "80 1 0:5 net:[x] /run/x rw - nsfs nsfs rw\n",
        "80 1 0:5 net:[4026532524] run/x rw - nsfs nsfs rw\n",
        "80 1 0:5 relative /run/x rw - tmpfs tmpfs rw\n",
    ],
    ids=["not-nsfs", "bad-inode", "relative-target", "relative-root"],
)
def test_other_non_path_roots_still_refuse(record):
    with pytest.raises(ValueError):
        blocked_mount_aliases(ROOT + WORKSPACE + record, ("/app",))


def test_a_path_rooted_nsfs_record_is_still_an_alias():
    info = ROOT + WORKSPACE + "80 1 0:2 / /view rw - nsfs nsfs rw\n"
    assert "/view" in blocked_mount_aliases(info, ("/app",))
