"""``--judge``: the host's re-judgment on the pristine image (R).

BenchFlow runs the armed source it installed, as ``__main__`` with
``--judge``, in a fresh container of the task image: as root, with no network
and nothing mounted, through a bootstrap that takes the source and the request
on stdin (FX687's host side). The image there is as it was before any agent,
so whatever the guard refuses is its own policy refusing the image's code,
never the trial's doing. Nothing the trial wrote reaches the request.

These tests drive that bootstrap in a subprocess over a folder standing in
for the image, and in a real ``python:3.6-slim`` container started exactly as
the host starts it.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.sandbox import _image_manifest, lockdown
from benchflow.sandbox import _pytest_plugin_guard as guard
from tests.test_pytest_plugin_guard_floor import floor_python  # noqa: F401

# FX687's bootstrap, safe on Python 3.6.
BOOTSTRAP = """
import io, json, os, runpy, sys, tempfile
data = json.loads(sys.stdin.read())
fd, path = tempfile.mkstemp(suffix=".py")
with os.fdopen(fd, "w") as handle:
    handle.write(data["guard"])
sys.argv = [path, "--judge"]
sys.stdin = io.StringIO(json.dumps(data["request"]))
runpy.run_path(path, run_name="__main__")
"""
NAME = "_bf_guard_judgetest"
PYTHON = ".".join(str(part) for part in sys.version_info[:3])


def _manifest(*paths: Path) -> str:
    return json.dumps(
        {
            "version": 1,
            "taken": "2026-11-16T00:00:00Z",
            "files": {
                str(p): {
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    "size": p.stat().st_size,
                }
                for p in paths
            },
        }
    )


def _armed(tmp_path: Path, blocked: tuple[str, ...], extra: str = "", **policy) -> str:
    source = lockdown._pytest_plugin_guard_source(
        NAME,
        blocked,
        list(policy.pop("requested", [])),
        markers_dir=str(tmp_path / "markers"),
        key_file=str(tmp_path / "key"),
        manifest=str(tmp_path / "manifest.json"),
        **policy,
    )
    source = lockdown.pytest_plugin_guard_judge_source(source)
    entry = lockdown.PYTEST_PLUGIN_GUARD_JUDGE_ENTRY
    assert source.endswith(entry)
    # The stand-in image lives under the test's /tmp, a tree the manifest
    # leaves out in a real image: let it speak for this one.
    return source[: -len(entry)] + "_MANIFEST_PRUNED = ()\n" + extra + entry


def _judge(source: str, request, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BOOTSTRAP],
        input=json.dumps({"guard": source, "request": request}),
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=300,
    )


@pytest.fixture
def image(tmp_path):
    """A workspace standing in for the image: a package and a conftest the
    manifest records, a file changed since, and one it never had. The
    markers folder and key a trial would have exist, so a marker could be
    written if the judge tried."""
    ws = tmp_path / "testbed"
    (ws / "imgpkg").mkdir(parents=True)
    (ws / "imgpkg" / "__init__.py").write_text("")
    (ws / "imgpkg" / "fixtures.py").write_text("FIXTURE = 1\n")
    (ws / "edited.py").write_text("x = 1\n")
    (ws / "conftest.py").write_text("pytest_plugins = ['imgpkg.fixtures']\n")
    manifest = _manifest(
        ws / "imgpkg" / "__init__.py",
        ws / "imgpkg" / "fixtures.py",
        ws / "edited.py",
        ws / "conftest.py",
    )
    (ws / "edited.py").write_text("x = 2\n")
    (ws / "planted.py").write_text("y = 1\n")
    (ws / "pytest.ini").write_text(
        "[pytest]\naddopts = -p imgpkg.fixtures -pno:cacheprovider -p nosuch_bf\n"
    )
    (tmp_path / "markers").mkdir()
    (tmp_path / "key").write_text("00" * 32)
    return ws, manifest


def _rows(done: subprocess.CompletedProcess) -> list[dict]:
    assert done.returncode == 0, done.stderr
    rows = json.loads(done.stdout)
    assert isinstance(rows, list)
    return rows


def test_the_judge_judges_the_request_and_what_the_image_registers(tmp_path, image):
    ws, manifest = image
    request = {
        "paths": [
            str(ws / "edited.py"),
            str(ws / "planted.py"),
            str(ws / "imgpkg" / "fixtures.py"),
            "/nonexistent/x.py",
        ],
        "modules": ["imgpkg.fixtures", "nosuchmodule_bf"],
        "manifest": manifest,
    }
    # ownership=False (a separate verifier's policy): this test's Python and
    # stand-in image are not root's, which only the agent's sandbox demands.
    source = _armed(tmp_path, (str(ws),), ownership=False)
    rows = _rows(_judge(source, request, ws))
    by = {(row.get("path") or row.get("module"), row["from"]): row for row in rows}

    def kind(item, origin="request"):
        return by[(str(item), origin)]["kind"]

    assert kind(ws / "edited.py") == "rejected"
    assert f"is under {ws}" in by[(str(ws / "edited.py"), "request")]["why"]
    assert kind(ws / "planted.py") == "rejected"
    assert kind(ws / "imgpkg" / "fixtures.py") == "trusted"
    assert kind("/nonexistent/x.py") == "absent"
    assert kind("imgpkg.fixtures") == "trusted"
    assert kind("nosuchmodule_bf") == "absent"
    # The image's conftest names its plugin: judged with no request for it.
    assert kind("imgpkg.fixtures", f"conftest:{ws / 'conftest.py'}") == "trusted"
    # And the image's ini names its -p plugins (never a no: one).
    ini = f"ini:{ws / 'pytest.ini'}"
    assert kind("imgpkg.fixtures", ini) == "trusted"
    assert kind("nosuch_bf", ini) == "absent"
    assert sorted(r.get("module") for r in rows if r["from"] == ini) == [
        "imgpkg.fixtures",
        "nosuch_bf",
    ]
    # This Python's pytest11 entry points (pytest-xdist at least), all the
    # image's own and unchanged.
    entry_points = [row for row in rows if row["from"].startswith("entry_point:")]
    assert entry_points, rows
    assert [row for row in entry_points if row["kind"] != "trusted"] == []
    assert {row["python"] for row in rows} == {PYTHON}
    for row in rows:
        assert len(set(row) & {"path", "module"}) == 1, row
        assert set(row) - {"path", "module"} == {"kind", "why", "python", "from"}, row
    # No marker, though the folder and key were there.
    assert list((tmp_path / "markers").iterdir()) == []


def test_the_guards_own_failure_on_an_item_is_that_items_row(tmp_path, image):
    ws, manifest = image
    source = _armed(
        tmp_path,
        (str(ws),),
        extra="def _untrusted_reason(path, blocked):\n"
        "    raise TypeError('forced by the test')\n",
    )
    request = {"paths": [str(ws / "edited.py")], "modules": [], "manifest": manifest}
    rows = _rows(_judge(source, request, ws))
    row = next(r for r in rows if r.get("path") == str(ws / "edited.py"))
    assert row["kind"] == "guard_error"
    assert row["why"] == (
        f"guard_error: TypeError: forced by the test while judging {ws / 'edited.py'}"
    )
    assert list((tmp_path / "markers").iterdir()) == []


@pytest.mark.parametrize(
    ("same_code", "kind", "why"),
    [
        (
            "    raise TypeError(\"zip() got an unexpected keyword argument 'strict'\")\n",
            "guard_error",
            "guard_error: TypeError: zip() got an unexpected keyword argument "
            "'strict' while judging /",
        ),
        (
            "    return False\n",
            "rejected",
            "the guard does not take its own compilation",
        ),
    ],
    ids=["raises", "refuses"],
)
def test_the_judge_reaches_the_cache_check_without_caches(
    tmp_path, image, same_code, kind, why
):
    """The stand-in image ships no caches, as a pristine image need not. The
    trial's caches (test.sh's install writes them) took astropy-7166's guard
    into the byte-code comparison, where 3.6 raised; the judge compiles each
    source itself, so a broken comparison shows here too."""
    ws, manifest = image
    source = _armed(
        tmp_path,
        (str(ws),),
        extra="def _same_code(cached, compiled):\n" + same_code,
        ownership=False,
    )
    request = {"paths": [], "modules": ["imgpkg.fixtures"], "manifest": manifest}
    rows = _rows(_judge(source, request, ws))
    row = next(r for r in rows if r["from"] == "request")
    assert not list(ws.rglob("*.pyc"))
    assert row["kind"] == kind, row
    assert why in row["why"], row


@pytest.mark.parametrize(
    "request_",
    [
        {"paths": "x", "modules": []},
        {"paths": [1], "modules": []},
        {"paths": [], "modules": [], "manifest": json.dumps({"version": 2})},
        {"paths": [], "modules": [], "manifest": "{not json"},
    ],
    ids=["paths-not-a-list", "path-not-a-string", "manifest-v2", "manifest-not-json"],
)
def test_a_request_the_judge_cannot_read_exits_2(tmp_path, image, request_):
    ws, _ = image
    done = _judge(_armed(tmp_path, (str(ws),)), request_, ws)
    assert done.returncode == 2, (done.stdout, done.stderr)
    assert done.stdout == ""
    assert "the plugin guard's judge refuses its request" in done.stderr


def test_the_discovery_entry_point_is_unchanged_by_the_judge(tmp_path):
    """The pre-verifier discovery (``python3 -I guard.py <prefixes> ...``)
    still answers; only ``--judge`` reaches the judge."""
    script = tmp_path / "guard.py"
    script.write_text(lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT)
    done = subprocess.run(
        [sys.executable, "-I", str(script), json.dumps([str(tmp_path / "ws")]), "[]"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert set(json.loads(done.stdout)) == {"plugins", "rejected"}
    judged = subprocess.run(
        [sys.executable, "-I", str(script), "--judge"],
        input="{}",
        capture_output=True,
        text=True,
        timeout=300,
    )
    # The unarmed script has no judge entry: it neither judges nor fails.
    assert (judged.returncode, judged.stdout) == (0, ""), judged.stderr


def test_the_judge_never_goes_into_the_sandbox():
    """Discovery's script and the armed guard go into the sandbox inside one
    ``sh -c`` argument (``printf %s <source> > guard``; discovery's
    ``python -c``), which Linux caps at 128 KiB. With the judge inside, rel's
    guard passed that (``Argument list too long``), so the judge is the
    host's alone: only the source built for the pristine image carries it."""
    armed = lockdown._pytest_plugin_guard_source(NAME, ("/testbed",), ["pytester"])
    for text in (lockdown._DISCOVER_PYTEST_PLUGINS_SCRIPT, armed):
        assert "def _judge_main" not in text
        assert lockdown.PYTEST_PLUGIN_GUARD_JUDGE_ENTRY not in text
    judge = lockdown.pytest_plugin_guard_judge_source(armed)
    assert judge.startswith(armed)
    assert "def _judge_main" in judge
    assert judge.endswith(lockdown.PYTEST_PLUGIN_GUARD_JUDGE_ENTRY)
    # The armed guard, quoted as the install writes it, with room to spare.
    assert len(shlex.quote(armed).encode()) < 120_000


@pytest.mark.parametrize(
    ("text", "names"),
    [
        ("pytest_plugins = 'a.b, c'\n", ["a.b", "c"]),
        ("pytest_plugins = ('a', 'b.c')\n", ["a", "b.c"]),
        ("pytest_plugins: list = ['a']\n", ["a"]),
        ("pytest_plugins = make()\n", []),
        ("pytest_plugins = [name]\n", []),
        ("def (\n", []),
        ("x = 'pytest_plugins'\n", []),
    ],
    ids=["comma-string", "tuple", "annotated", "call", "name", "syntax-error", "other"],
)
def test_conftest_plugins_reads_only_literals(tmp_path, text, names):
    conftest = tmp_path / "conftest.py"
    conftest.write_text(text)
    assert guard._conftest_plugins(str(conftest)) == names


def test_entry_points_are_read_without_importlib_metadata(tmp_path, monkeypatch):
    """pytest 3 on Python 3.6 has neither importlib.metadata nor its backport
    (astropy 1.3's testbed): the judge reads entry_points.txt itself."""
    dist = tmp_path / "foo-1.0.dist-info"
    dist.mkdir()
    (dist / "entry_points.txt").write_text(
        "[console_scripts]\nfoo = foo:main\n\n[pytest11]\nFoo_Plugin = foo.plugin\n"
    )
    egg = tmp_path / "bar.egg-info"
    egg.mkdir()
    (egg / "entry_points.txt").write_text(
        "[pytest11]\n# a comment\nbar=bar.pytest_bar\n"
    )
    assert [ep.name for ep, _ in guard._scanned_registrations([str(tmp_path)])] == [
        "bar",
        "Foo_Plugin",
    ]
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    found = [(ep.name, ep.value, dist) for ep, dist in guard._scanned_registrations()]
    assert found == [
        ("bar", "bar.pytest_bar", str(egg)),
        ("Foo_Plugin", "foo.plugin", str(dist)),
    ]

    # registrations() itself, in discovery (a search path) and in the armed
    # guard (sys.path), where neither library imports.
    monkeypatch.setitem(sys.modules, "importlib.metadata", None)
    monkeypatch.setitem(sys.modules, "importlib_metadata", None)
    monkeypatch.setattr(guard, "_BENCHFLOW_SEARCH_PATH", [str(tmp_path)])
    assert [ep.name for ep, _ in guard.registrations()] == ["bar", "Foo_Plugin"]
    monkeypatch.setattr(guard, "_BENCHFLOW_SEARCH_PATH", None)
    assert [ep.name for ep, _ in guard.registrations()] == ["bar", "Foo_Plugin"]

    def no_metadata():
        raise ImportError("No module named 'importlib_metadata'")

    monkeypatch.setattr(guard, "registrations", no_metadata)
    rows = guard._judge_discovered((), None, NAME)
    assert {row["from"] for row in rows} == {
        "entry_point:bar",
        "entry_point:Foo_Plugin",
    }


def test_the_judge_runs_on_python_3_6_as_the_host_starts_it(floor_python):  # noqa: F811
    """FX687's exact start: ``docker run --rm -i --pull never --network none
    --user 0:0 --entrypoint <python> <image> -c <bootstrap>``, nothing
    mounted. python:3.6-slim has no pytest and no importlib_metadata, so the
    entry points come from the scan, and the requested short name is
    absent."""
    source = lockdown.pytest_plugin_guard_judge_source(
        lockdown._pytest_plugin_guard_source(
            NAME,
            ("/testbed",),
            ["pytester"],
            manifest=_image_manifest.MANIFEST_PATH,
        )
    )
    request = {
        "paths": [
            "/usr/local/lib/python3.6/site-packages/pip/__init__.py",
            "/testbed/x.py",
        ],
        "modules": ["pip._internal"],
        "manifest": json.dumps({"version": 1, "taken": "t", "files": {}}),
    }
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--pull",
            "never",
            "--network",
            "none",
            "--user",
            "0:0",
            "--entrypoint",
            "/usr/local/bin/python",
            floor_python,
            "-c",
            BOOTSTRAP,
        ],
        input=json.dumps({"guard": source, "request": request}),
        capture_output=True,
        text=True,
        timeout=300,
    )
    rows = _rows(done)
    by = {
        (row.get("path") or row.get("module")): row
        for row in rows
        if row["from"] in ("request", "requested")
    }
    assert (
        by["/usr/local/lib/python3.6/site-packages/pip/__init__.py"]["kind"]
        == "trusted"
    ), by
    assert by["/testbed/x.py"]["kind"] == "absent"
    assert by["pip._internal"]["kind"] == "trusted", by
    assert by["pytester"] == {
        "module": "pytester",
        "kind": "absent",
        "why": "pytester is not on the search path",
        "python": by["pytester"]["python"],
        "from": "requested",
    }
    assert {row["python"][:4] for row in rows} == {"3.6."}
    assert [row for row in rows if row["kind"] == "guard_error"] == []
    # The 3.6 bug the guard had (zip(strict=True) in _same_code), put back:
    # the judge reaches it on the image's own sources and says guard_error.
    anchor = "def _same_code(cached, compiled):\n"
    assert source.count(anchor) == 1
    replay = source.replace(anchor, anchor + "    list(zip((), (), strict=True))\n")
    done = subprocess.run(
        done.args,
        input=json.dumps({"guard": replay, "request": request}),
        capture_output=True,
        text=True,
        timeout=300,
    )
    replayed = {
        (row.get("path") or row.get("module")): row
        for row in _rows(done)
        if row["from"] == "request"
    }
    for item in (
        "/usr/local/lib/python3.6/site-packages/pip/__init__.py",
        "pip._internal",
    ):
        assert replayed[item]["kind"] == "guard_error", replayed[item]
        assert "TypeError: zip" in replayed[item]["why"], replayed[item]


@pytest.mark.parametrize("tomllib", [True, False], ids=["tomllib", "no-tomllib"])
@pytest.mark.parametrize(
    ("name", "text", "names"),
    [
        ("setup.cfg", "[tool:pytest]\naddopts = -p a -pb.c\n", ["a", "b.c"]),
        ("tox.ini", "[pytest]\naddopts = -p no:x -p y\n", ["y"]),
        ("pytest.ini", "[pytest]\naddopts = -p 'a'\n[other]\naddopts = -p z\n", ["a"]),
        ("setup.cfg", "[pytest]\naddopts = -p wrong-section\n", []),
        ("pytest.ini", "[pytest\naddopts = -p a\n", []),
        ("pyproject.toml", '[tool.pytest.ini_options]\naddopts = "-p a -v"\n', ["a"]),
        (
            "pyproject.toml",
            '[tool.black]\naddopts = "-p no"\n\n[tool.pytest.ini_options]\n'
            'addopts = [\n  "-p", "a",\n  "-pb",\n]\n',
            ["a", "b"],
        ),
        ("pyproject.toml", "[tool.pytest.ini_options]\naddopts = [\n", []),
    ],
    ids=[
        "setup.cfg",
        "tox.ini",
        "pytest.ini-quoted",
        "setup.cfg-wrong-section",
        "broken-ini",
        "toml-string",
        "toml-array",
        "toml-broken",
    ],
)
def test_ini_plugins_reads_the_addopts_p_names(
    tmp_path, monkeypatch, tomllib, name, text, names
):
    if not tomllib:
        real = guard._stdlib_import

        def without_tomllib(module):
            if module == "tomllib":
                raise ImportError("No module named 'tomllib'")
            return real(module)

        monkeypatch.setattr(guard, "_stdlib_import", without_tomllib)
    ini = tmp_path / name
    ini.write_text(text)
    assert guard._ini_plugins(str(ini)) == names


# What discovery finds in python:3.6-slim: a plugin distribution the image
# installed (root's, entry_points.txt only), and no importlib_metadata.
_FAKE_PLUGIN = (
    "set -e; s=/usr/local/lib/python3.6/site-packages; "
    "mkdir -p $s/fakeplug-1.0.dist-info; "
    "printf '[pytest11]\\nfakeplug = fakeplug\\n' > $s/fakeplug-1.0.dist-info/entry_points.txt; "
    "printf 'Metadata-Version: 2.1\\nName: fakeplug\\nVersion: 1.0\\n' > $s/fakeplug-1.0.dist-info/METADATA; "
    "echo 'x = 1' > $s/fakeplug.py; "
    "python3 -c 'import importlib_metadata' 2>/dev/null && exit 9; "
)


def test_discovery_reads_entry_points_on_python_3_6(floor_python):  # noqa: F811
    """Pre-verifier discovery on an image whose only Python is 3.6 failed
    with "No module named 'importlib.metadata'" (an unscored error; FX609R).
    It now reads the image's entry_points.txt itself and proposes the
    plugin, as on a newer Python."""
    command = lockdown._discover_pytest_plugins_cmd(("/testbed",))
    done = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--network",
            "none",
            "--user",
            "0:0",
            floor_python,
            "sh",
            "-c",
            _FAKE_PLUGIN + command,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout.strip().splitlines()[-1]) == {
        "plugins": ["fakeplug"],
        "rejected": [],
    }
