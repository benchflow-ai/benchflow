"""A conftest.py or pytest configuration file a wheel ships, as BenchFlow's own uv cache unpacked it, is an install.

Every trial where BenchFlow's own uv fills its cache after the record from before the turn read flagged: the model
proxy BenchFlow installs in the sandbox after that record (measured on Daytona; Modal and the other providers that
install it there by the code), and agents BenchFlow installs with its uv. On Docker the proxy's venv and the cache are
already in place at the record, so Docker never flagged (measured). uv unpacks each wheel into BenchFlow's cache
(root's alone), and the proxy's numpy and fsspec each ship a conftest.py, so the walk after the turn found two
"planted" conftests in ``/opt/runtime/cache/uv/archive-v0/<entry>/``. Such a file is the wheel's own when a dist RECORD at its archive entry's
root lists it with its sha256, as the walk already treats a .pth a pip dist's RECORD lists. The folder is never left
out wholesale: a file there that matches no RECORD, and one anywhere else (a RECORD beside it or not), stay flagged.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from benchflow._uv_pin import UV_CACHE_DIR
from benchflow.sandbox import _image_manifest as im
from benchflow.sandbox import _image_trees as trees
from tests import test_image_trees
from tests.test_image_trees import _detect, _flags, _manifest, _record

# The container the image-trees rows walk (docker run, network none).
box = test_image_trees.box

UV_ARCHIVE = f"{UV_CACHE_DIR}/archive-v0"
ENTRY = f"{UV_ARCHIVE}/HdYdRPPRqZ5ENUvc"
CONFTEST = f"{ENTRY}/numpy/conftest.py"
BODY = "import pytest\n\n\ndef pytest_configure(config):\n    pass\n"


def _record_line(path: str, body: str) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(body.encode()).digest())
    return f"{path},sha256={digest.rstrip(b'=').decode()},{len(body.encode())}"


def _files(**entries: str) -> dict:
    return {
        path: {"sha256": hashlib.sha256(body.encode()).hexdigest(), "size": len(body)}
        for path, body in entries.items()
    }


# -- the rule, on the host's copies ---------------------------------------------


@pytest.mark.parametrize(
    ("owner", "flagged"),
    [
        ("uv:numpy-2.4.6.dist-info", False),
        (None, True),
        ("pip:numpy-2.4.6.dist-info", True),
    ],
    ids=["uv-record", "no-record", "not-the-uv-cache"],
)
def test_a_cache_conftest_is_flagged_unless_its_uv_record_lists_it(owner, flagged):
    before = im.ImageManifest(taken="pre-agent", files={}, trees=_manifest().trees)
    after_trees = dict(_manifest().trees)
    after_trees["owned"] = {CONFTEST: owner} if owner else {}
    after = im.ImageManifest(
        taken="post-turn", files=_files(**{CONFTEST: BODY}), trees=after_trees
    )
    flags, changes, _ = im._diff_files(before, after)
    if flagged:
        assert [(f["path"], f["kind"], f["category"]) for f in flags] == [
            (CONFTEST, "planted", "conftest")
        ]
    else:
        assert flags == []
        assert [(c["path"], c["category"]) for c in changes] == [
            (CONFTEST, "conftest-owned")
        ]


def test_the_walk_after_the_turn_is_told_where_the_uv_cache_is():
    assert trees.after_options(_manifest()).get("uv_archive") == UV_ARCHIVE


# -- end to end, the record script in a container --------------------------------


def _plant(box, path: str, body: str, record_lines: list[str] | None) -> None:
    folder = path.rsplit("/", 1)[0]
    box.sh(f"mkdir -p {folder} && cat > {path} <<'EOF'\n{body}EOF")
    if record_lines is not None:
        dist = f"{ENTRY}/numpy-2.4.6.dist-info"
        text = "\n".join(record_lines) + "\n"
        box.sh(f"mkdir -p {dist} && cat > {dist}/RECORD <<'EOF'\n{text}EOF")


def test_the_proxys_numpy_conftest_in_the_uv_cache_reads_checked(box):
    """The primes case: the model proxy's numpy wheel unpacked after the record
    from before the turn, its conftest.py listed in its RECORD."""
    _record(box)
    _plant(box, CONFTEST, BODY, [_record_line("numpy/conftest.py", BODY)])
    record = _detect(box)
    assert record["status"] == "checked", _flags(record)
    owned = [c for c in record["changes"] if c["path"] == CONFTEST]
    assert [c["category"] for c in owned] == ["conftest-owned"]


@pytest.mark.parametrize(
    "record_lines",
    [None, [_record_line("numpy/conftest.py", "something else\n")]],
    ids=["no-record", "record-for-other-bytes"],
)
def test_a_conftest_planted_in_the_uv_cache_stays_flagged(box, record_lines):
    _record(box)
    _plant(box, CONFTEST, BODY, record_lines)
    record = _detect(box)
    assert (CONFTEST, "planted", "conftest") in _flags(record)


def test_a_conftest_in_the_test_hierarchy_stays_flagged(box):
    _record(box)
    box.sh(f"cat > /app/conftest.py <<'EOF'\n{BODY}EOF")
    record = _detect(box)
    assert ("/app/conftest.py", "planted", "conftest") in _flags(record)


# -- the boundary: a RECORD owns only its own archive entry's files ----------------


def _flagged(record, path: str) -> bool:
    return any(
        f["path"] == path and f["category"] in ("conftest", "pytest-config")
        for f in record["flags"]
    )


def _write(box, path: str, text: str) -> None:
    folder = path.rsplit("/", 1)[0]
    box.sh(f"mkdir -p '{folder}' && cat > '{path}' <<'EOF'\n{text}EOF")


def _record_file(
    box, entry: str, lines: list[str], dist: str = "numpy-2.4.6.dist-info"
):
    _write(box, f"{entry}/{dist}/RECORD", "\n".join(lines) + "\n")


def test_a_record_in_another_entry_owns_nothing(box):
    _record(box)
    planted = f"{UV_ARCHIVE}/AAAAAAAAAAAAAAAA/numpy/conftest.py"
    _write(box, planted, BODY)
    _record_file(
        box, f"{UV_ARCHIVE}/BBBBBBBBBBBBBBBB", [_record_line("numpy/conftest.py", BODY)]
    )
    assert _flagged(_detect(box), planted)


@pytest.mark.parametrize(
    ("rel", "planted"),
    [
        ("../../../../../../../app/conftest.py", "/app/conftest.py"),
        (
            "../DDDDDDDDDDDDDDDD/numpy/conftest.py",
            f"{UV_ARCHIVE}/DDDDDDDDDDDDDDDD/numpy/conftest.py",
        ),
        (
            "numpy/../../DDDDDDDDDDDDDDDD/numpy/conftest.py",
            f"{UV_ARCHIVE}/DDDDDDDDDDDDDDDD/numpy/conftest.py",
        ),
        ("/app/conftest.py", "/app/conftest.py"),
    ],
    ids=["to-app", "to-sibling-entry", "inner-dotdot", "absolute"],
)
def test_a_record_line_outside_its_entry_owns_nothing(box, rel, planted):
    _record(box)
    _write(box, planted, BODY)
    _record_file(box, f"{UV_ARCHIVE}/CCCCCCCCCCCCCCCC", [_record_line(rel, BODY)])
    assert _flagged(_detect(box), planted)


@pytest.mark.parametrize(
    "entry",
    [
        "/app",
        "/app/vendor/archive-v0/EEEEEEEEEEEEEEEE",
        "/var/cache/uv/archive-v0/EEEEEEEEEEEEEEEE",
        f"{UV_ARCHIVE}-x/EEEEEEEEEEEEEEEE",
        f"{UV_CACHE_DIR}/wheels-v5/EEEEEEEEEEEEEEEE",
    ],
    ids=[
        "test-hierarchy",
        "workspace-archive-shaped",
        "other-uv-cache",
        "prefix-collision",
        "cache-not-archive",
    ],
)
def test_a_record_listed_conftest_outside_the_archive_stays_flagged(box, entry):
    """A dist RECORD that lists the file, beside it, owns nothing outside
    BenchFlow's own archive-v0 entries (the test hierarchy among them)."""
    _record(box)
    planted = f"{entry}/numpy/conftest.py"
    _write(box, planted, BODY)
    _record_file(box, entry, [_record_line("numpy/conftest.py", BODY)])
    assert _flagged(_detect(box), planted)


def test_a_dist_info_below_the_entry_root_owns_nothing(box):
    _record(box)
    entry = f"{UV_ARCHIVE}/HHHHHHHHHHHHHHHH"
    planted = f"{entry}/numpy/conftest.py"
    _write(box, planted, BODY)
    _record_file(box, f"{entry}/numpy", [_record_line("conftest.py", BODY)])
    assert _flagged(_detect(box), planted)


def test_a_cached_conftest_edited_in_the_turn_is_flagged(box):
    entry = f"{UV_ARCHIVE}/GGGGGGGGGGGGGGGG"
    path = f"{entry}/numpy/conftest.py"
    _write(box, path, BODY)
    _record_file(box, entry, [_record_line("numpy/conftest.py", BODY)])
    _record(box)
    box.sh(f"printf 'import os\\n' >> '{path}'")
    assert (path, "edited", "conftest") in _flags(_detect(box))


def test_a_pytest_config_a_cached_wheel_ships_is_an_install(box):
    _record(box)
    entry = f"{UV_ARCHIVE}/FFFFFFFFFFFFFFFF"
    ini = "[pytest]\naddopts = -p no:cacheprovider\n"
    _write(box, f"{entry}/pkg/pytest.ini", ini)
    _record_file(
        box, entry, [_record_line("pkg/pytest.ini", ini)], dist="pkg-1.0.dist-info"
    )
    record = _detect(box)
    assert not _flagged(record, f"{entry}/pkg/pytest.ini"), _flags(record)
    assert record["status"] == "checked", _flags(record)


@pytest.mark.parametrize(
    "path",
    [
        "/app/conftest.py",
        f"{UV_ARCHIVE}-x/E/numpy/conftest.py",
        "/opt/runtime/conftest.py",
        # Paths an honest walk never gives: not in normal form.
        f"{UV_ARCHIVE}/../../../../app/conftest.py",
        f"{UV_ARCHIVE}/E/../../../../../app/conftest.py",
        f"{UV_ARCHIVE}/./E/numpy/conftest.py",
        f"{UV_ARCHIVE}//E/numpy/conftest.py",
    ],
    ids=[
        "app",
        "archive-lookalike",
        "runtime",
        "dotdot-top",
        "dotdot-inner",
        "dot",
        "double-slash",
    ],
)
def test_the_host_takes_a_uv_owner_only_inside_the_cache(path):
    before = im.ImageManifest(taken="pre-agent", files={}, trees=_manifest().trees)
    after_trees = dict(_manifest().trees)
    after_trees["owned"] = {path: "uv:numpy-2.4.6.dist-info"}
    after = im.ImageManifest(
        taken="post-turn", files=_files(**{path: BODY}), trees=after_trees
    )
    flags, _, _ = im._diff_files(before, after)
    assert [(f["path"], f["category"]) for f in flags] == [(path, "conftest")]


def test_the_record_names_the_rules_trust():
    record = im.detector_record("audit", _manifest(), None)
    assert any(
        "BenchFlow's own uv cache" in n and "root only" in n for n in record["notes"]
    )


# The shell walk (an image without a Python) has no RECORD reader: a cached
# wheel's own conftest.py still reads planted there (a known issue, no reward
# effect). An unlisted one is flagged either way.
bare = test_image_trees.bare


def test_the_shell_walk_flags_an_unlisted_cache_conftest(bare):
    _record(bare)
    planted = f"{ENTRY}/numpy/conftest.py"
    _write(bare, planted, BODY)
    assert _flagged(_detect(bare), planted)


# A linked dist-info or RECORD inside a real entry: the RECORD's bytes would
# live elsewhere (an agent-writable /tmp). Making the link needs a write into
# the root-only cache, but the rule reads through no link.
@pytest.mark.parametrize("what", ["dist-info", "RECORD"])
def test_a_linked_dist_info_or_record_owns_nothing(box, what):
    _record(box)
    entry = f"{UV_ARCHIVE}/LINKEDDIST000000"
    planted = f"{entry}/numpy/conftest.py"
    dist = "numpy-2.4.6.dist-info"
    _write(box, planted, BODY)
    _write(
        box,
        f"/tmp/linked/{dist}/RECORD",
        _record_line("numpy/conftest.py", BODY) + "\n",
    )
    if what == "dist-info":
        box.sh(f"ln -s /tmp/linked/{dist} '{entry}/{dist}'")
    else:
        box.sh(
            f"mkdir -p '{entry}/{dist}' && "
            f"ln -s /tmp/linked/{dist}/RECORD '{entry}/{dist}/RECORD'"
        )
    assert _flagged(_detect(box), planted)


# The pytest-config side keeps its guards: a config no RECORD lists, or one
# whose RECORD hash is for other bytes, stays pytest-config.
@pytest.mark.parametrize(
    "record_lines",
    [None, [_record_line("pkg/pytest.ini", "[pytest]\naddopts = -q\n")]],
    ids=["no-record", "record-for-other-bytes"],
)
def test_an_unlisted_pytest_config_in_the_cache_stays_flagged(box, record_lines):
    _record(box)
    entry = f"{UV_ARCHIVE}/IIIIIIIIIIIIIIII"
    path = f"{entry}/pkg/pytest.ini"
    _write(box, path, "[pytest]\naddopts = -p no:cacheprovider\n")
    if record_lines is not None:
        _record_file(box, entry, record_lines, dist="pkg-1.0.dist-info")
    assert (path, "planted", "pytest-config") in _flags(_detect(box))


def test_the_shell_walk_still_flags_a_record_listed_cache_conftest(bare):
    """The known issue as it stands (0.10.1 notes): the shell walk has no RECORD
    reader, so a cached wheel's own conftest.py still reads planted there."""
    _record(bare)
    planted = f"{ENTRY}/numpy/conftest.py"
    _write(bare, planted, BODY)
    _record_file(bare, ENTRY, [_record_line("numpy/conftest.py", BODY)])
    assert _flagged(_detect(bare), planted)
