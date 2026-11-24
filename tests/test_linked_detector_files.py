"""A linked conftest.py, pytest configuration or start-up file is seen.

pytest and Python read through a symlink, but the files walk opened only
regular files: a conftest.py planted as a link (``/app/tests/conftest.py ->
/tmp/c.py``) read ``checked``, with no flag and no change, where the same file
as a regular or hard-linked file was flagged. Now each walk (Python's, and the
shell walk on an image without one) keeps a wanted name that is a link by its
target, so one planted or retargeted in the turn is a flag whatever it points
at, and never a package's (dpkg, a pip or uv RECORD). The same link before and
after the turn is unchanged. The tree sections (``.pth`` and customize files in
a site, pytest's own files, PATH tools) already compare a link's target; the
guards below keep them so.
"""

from __future__ import annotations

import pytest

from benchflow._uv_pin import UV_CACHE_DIR
from tests import test_image_trees
from tests.test_image_trees import SITE, TEST_SH, _detect, _flags, _record

box = test_image_trees.box
bare = test_image_trees.bare

ARCHIVE = f"{UV_CACHE_DIR}/archive-v0"


@pytest.mark.parametrize(
    ("plant", "path", "category"),
    [
        (
            "printf x > /tmp/c.py && ln -s /tmp/c.py /app/tests/conftest.py",
            "/app/tests/conftest.py",
            "conftest",
        ),
        (
            "printf x > /tmp/c.py && ln /tmp/c.py /app/tests/conftest.py",
            "/app/tests/conftest.py",
            "conftest",
        ),
        (
            "printf '[pytest]\\naddopts = -q\\n' > /tmp/p.ini && ln -s /tmp/p.ini /app/pytest.ini",
            "/app/pytest.ini",
            "pytest-config",
        ),
        (
            "printf 'import os\\n' > /tmp/s.py && ln -s /tmp/s.py /app/sitecustomize.py",
            "/app/sitecustomize.py",
            "startup",
        ),
    ],
    ids=[
        "symlinked-conftest",
        "hard-linked-conftest",
        "symlinked-pytest-ini",
        "symlinked-pth",
    ],
)
def test_a_linked_file_planted_in_the_turn_is_flagged(box, plant, path, category):
    _record(box)
    box.sh("mkdir -p /app/tests && " + plant)
    assert (path, "planted", category) in _flags(_detect(box, test_script=TEST_SH))


def test_the_same_link_before_and_after_the_turn_is_unchanged(box):
    box.sh(
        "mkdir -p /app/tests && printf x > /tmp/c.py && ln -s /tmp/c.py /app/tests/conftest.py"
    )
    _record(box)
    record = _detect(box, test_script=TEST_SH)
    assert record["status"] == "checked", _flags(record)
    assert not [c for c in record["changes"] if c["path"] == "/app/tests/conftest.py"]


def test_a_link_retargeted_in_the_turn_is_edited(box):
    box.sh(
        "mkdir -p /app/tests && printf x > /tmp/a.py && printf y > /tmp/b.py "
        "&& ln -s /tmp/a.py /app/tests/conftest.py"
    )
    _record(box)
    box.sh("ln -sfn /tmp/b.py /app/tests/conftest.py")
    assert ("/app/tests/conftest.py", "edited", "conftest") in _flags(
        _detect(box, test_script=TEST_SH)
    )


def test_a_linked_conftest_in_the_uv_cache_is_never_a_wheels(box):
    """Its target's bytes match the entry's RECORD; a link still flags."""
    import base64
    import hashlib

    body = "import pytest\n"
    digest = base64.urlsafe_b64encode(hashlib.sha256(body.encode()).digest())
    line = f"numpy/conftest.py,sha256={digest.rstrip(b'=').decode()},{len(body)}"
    entry = f"{ARCHIVE}/LINKEDFILE000000"
    _record(box)
    box.sh(
        f"mkdir -p '{entry}/numpy' '{entry}/numpy-2.4.6.dist-info' && "
        f"printf '%s' '{body}' > /tmp/c.py && "
        f"printf '%s\\n' '{line}' > '{entry}/numpy-2.4.6.dist-info/RECORD' && "
        f"ln -s /tmp/c.py '{entry}/numpy/conftest.py'"
    )
    assert (f"{entry}/numpy/conftest.py", "planted", "conftest") in _flags(
        _detect(box, test_script=TEST_SH)
    )


def test_the_shell_walk_flags_a_linked_conftest(bare):
    _record(bare)
    bare.sh(
        "mkdir -p /app/tests && printf x > /tmp/c.py && ln -s /tmp/c.py /app/tests/conftest.py"
    )
    assert ("/app/tests/conftest.py", "planted", "conftest") in _flags(_detect(bare))


# Guards: the tree sections already compare a link's target.
@pytest.mark.parametrize(
    ("plant", "path", "kind", "category"),
    [
        (
            f"printf x > /tmp/z.pth && ln -s /tmp/z.pth {SITE}/zz.pth",
            f"{SITE}/zz.pth",
            "planted",
            "startup",
        ),
        (
            f"printf x > /tmp/s.py && ln -s /tmp/s.py {SITE}/sitecustomize.py",
            f"{SITE}/sitecustomize.py",
            "planted",
            "startup",
        ),
        (
            f"printf x > /tmp/p.py && rm -f {SITE}/_pytest/__init__.py && ln -s /tmp/p.py {SITE}/_pytest/__init__.py",
            f"{SITE}/_pytest/__init__.py",
            "edited",
            "pytest-own",
        ),
        (
            "printf x > /tmp/pip && chmod 755 /tmp/pip && rm -f /usr/local/bin/pip && ln -s /tmp/pip /usr/local/bin/pip",
            "/usr/local/bin/pip",
            "edited",
            "path-tool",
        ),
    ],
    ids=["site-pth", "site-sitecustomize", "pytest-own", "path-tool"],
)
def test_the_tree_sections_flag_a_link_too(box, plant, path, kind, category):
    _record(box)
    box.sh(plant)
    assert (path, kind, category) in _flags(_detect(box, test_script=TEST_SH))


# The host: a link is never a package's, whatever owner the walk gave its path.
@pytest.mark.parametrize(
    ("path", "owner", "category"),
    [
        ("/etc/python3.6/sitecustomize.py", "python3.6-minimal", "startup"),
        (f"{ARCHIVE}/E/pkg/pytest.ini", "uv:pkg-1.0.dist-info", "pytest-config"),
        (f"{ARCHIVE}/E/numpy/conftest.py", "uv:numpy-2.4.6.dist-info", "conftest"),
    ],
    ids=["dpkg-startup", "uv-pytest-config", "uv-conftest"],
)
def test_the_host_takes_no_owner_for_a_link(path, owner, category):
    from benchflow.sandbox import _image_manifest as im
    from tests.test_image_trees import _manifest

    before = im.ImageManifest(taken="pre-agent", files={}, trees=_manifest().trees)
    trees = dict(_manifest().trees)
    trees["owned"] = {path: owner}
    after = im.ImageManifest(
        taken="post-turn",
        files={path: {"sha256": "link:/tmp/x", "size": 0, "link": "/tmp/x"}},
        trees=trees,
    )
    flags, _, _ = im._diff_files(before, after)
    assert [(f["path"], f["category"]) for f in flags] == [(path, category)]


def test_the_loader_keeps_a_link_only_with_its_own_target():
    import json

    from benchflow.sandbox import _image_manifest as im

    def loaded(entry):
        text = json.dumps(
            {
                "version": im.VERSION,
                "taken": "post-turn",
                "files": {"/app/conftest.py": entry},
            }
        )
        return dict(im.load(text).files)

    assert loaded({"sha256": "link:/tmp/x", "size": 0, "link": "/tmp/x"}) == {
        "/app/conftest.py": {"sha256": "link:/tmp/x", "size": 0, "link": "/tmp/x"}
    }
    assert loaded({"sha256": "link:/tmp/y", "size": 0, "link": "/tmp/x"}) == {}
    assert loaded({"sha256": "link:/tmp/x", "size": 0}) == {}
