"""The detector beside the faithful default.

In the default integrity mode test.sh runs as its authors wrote it, and
nothing BenchFlow finds changes the reward. What it finds is a diff of the
sandbox after the agent's turn against the image manifest recorded before it
(the host's copy): a conftest.py, a pytest configuration file, a start-up
module, bytecode without a source or a pytest plugin module that was edited,
planted or removed is a flag; any other change (the solution's own source
edits) is a change. Both go in the trial's ``integrity/verifier-inputs.json``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import uuid
from types import SimpleNamespace

import pytest

from benchflow.sandbox import _image_manifest as im


def _entry(text: str) -> dict:
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "size": len(text)}


def _walk(taken: str, files: dict[str, str]) -> im.ImageManifest:
    return im.ImageManifest(
        taken=taken, files={path: _entry(text) for path, text in files.items()}
    )


IMAGE = {
    "/testbed/tox.ini": "[pytest]\naddopts = -ra\n",
    "/testbed/tests/conftest.py": "import pytest\n",
    "/testbed/pkg/mod.py": "def f():\n    return 1\n",
    "/testbed/pkg/__pycache__/mod.cpython-312.pyc": "bytes",
}


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/testbed/conftest.py", "conftest"),
        ("/testbed/tox.ini", "pytest-config"),
        ("/testbed/.pytest.ini", "pytest-config"),
        ("/testbed/pytest.toml", "pytest-config"),
        ("/testbed/pyproject.toml", "pytest-config"),
        ("/testbed/setup.cfg", "pytest-config"),
        ("/opt/x/sitecustomize.py", "startup"),
        ("/opt/x/usercustomize.py", "startup"),
        ("/testbed/pkg/mod.pyc", "bytecode"),
        ("/testbed/pkg/__pycache__/mod.cpython-312.pyc", "bytecode-cache"),
        ("/testbed/pytest_evil.py", "pytest-plugin"),
        ("/testbed/pkg/mod.py", "python"),
    ],
)
def test_each_path_has_its_category(path, expected):
    assert im.category(path) == expected


def test_an_untouched_sandbox_has_no_flags_and_no_changes():
    record = im.detector_record(
        "audit", _walk("pre-agent", IMAGE), _walk("post-turn", IMAGE)
    )

    assert record["status"] == "checked"
    assert record["flags"] == [] and record["changes"] == []
    # These walks carry no trees, interpreters, PATH tools or start-up files
    # (_image_trees): nothing of theirs is compared.
    nothing = {"checked": 0, "flags": 0, "changes": 0}
    assert record["sections"] == {
        "files": {"checked": 4, "flags": 0, "changes": 0},
        "python_trees": nothing,
        "interpreters": nothing,
        "path_tools": nothing,
        "startup": nothing,
        "processes": nothing,
    }
    assert record["notes"] and record["changes_by_category"] == {}
    assert record["reward_changed"] is False


def test_the_test_machinery_is_flagged_and_everything_else_is_a_change():
    """The two tiers (the lead's 10:30Z scope): a conftest and a pytest ini
    section are flags; a pytest_*.py module and a sourceless .pyc are now
    changes, with the solution's own edits."""
    after = dict(IMAGE)
    after["/testbed/tox.ini"] = "[pytest]\naddopts = -ra -p evil\n"  # edited ini
    after["/testbed/conftest.py"] = "def pytest_runtest_call(): pass\n"  # planted
    after["/testbed/pytest_evil.py"] = "def pytest_configure(config): pass\n"
    after["/testbed/pkg/forged.pyc"] = "bytes"  # bytecode without a source
    del after["/testbed/tests/conftest.py"]  # removed
    after["/testbed/pkg/mod.py"] = "def f():\n    return 2\n"  # the solution
    after["/testbed/pkg/__pycache__/mod.cpython-312.pyc"] = "new bytes"

    record = im.detector_record(
        "audit", _walk("pre-agent", IMAGE), _walk("post-turn", after)
    )

    flags = {(f["path"], f["kind"], f["category"]) for f in record["flags"]}
    assert flags == {
        ("/testbed/tox.ini", "edited", "pytest-config"),
        ("/testbed/conftest.py", "planted", "conftest"),
        ("/testbed/tests/conftest.py", "removed", "conftest"),
    }
    changes = {(c["path"], c["kind"], c["category"]) for c in record["changes"]}
    assert changes == {
        ("/testbed/pytest_evil.py", "planted", "pytest-plugin"),
        ("/testbed/pkg/forged.pyc", "planted", "bytecode"),
        ("/testbed/pkg/mod.py", "edited", "python"),
        ("/testbed/pkg/__pycache__/mod.cpython-312.pyc", "edited", "bytecode-cache"),
    }
    assert record["changes_by_category"] == {
        "bytecode": 1,
        "bytecode-cache": 1,
        "pytest-plugin": 1,
        "python": 1,
    }
    assert record["status"] == "flagged"
    assert record["reward_changed"] is False
    edited = next(f for f in record["flags"] if f["path"] == "/testbed/tox.ini")
    assert edited["where"] == "files"
    assert edited["before"] == _entry(IMAGE["/testbed/tox.ini"])
    assert edited["after"] == _entry(after["/testbed/tox.ini"])
    removed = next(f for f in record["flags"] if f["kind"] == "removed")
    assert removed["after"] is None


@pytest.mark.parametrize(
    ("mode", "before", "after", "status", "because"),
    [
        ("off", True, True, "off", "integrity is off"),
        ("audit", False, True, "unchecked", "no record of the image"),
        ("strict", True, False, "unchecked", "the walk after the turn failed"),
    ],
)
def test_what_could_not_be_checked_says_why(mode, before, after, status, because):
    record = im.detector_record(
        mode,
        _walk("pre-agent", IMAGE) if before else None,
        _walk("post-turn", IMAGE) if after else None,
    )

    assert record["status"] == status
    assert because in record["unchecked_because"]
    assert record["flags"] == [] and record["reward_changed"] is False


def test_changes_are_capped_and_counted():
    after = dict(
        IMAGE, **{f"/testbed/gen/m{i}.py": "x" for i in range(im.CHANGES_MAX + 5)}
    )

    record = im.detector_record(
        "audit", _walk("pre-agent", IMAGE), _walk("post-turn", after)
    )

    assert len(record["changes"]) == im.CHANGES_MAX
    assert record["changes_truncated"] == 5
    assert record["sections"]["files"]["changes"] == im.CHANGES_MAX + 5


def test_off_runs_no_walk():
    calls = []

    class Env:
        _benchflow_image_manifest: im.ImageManifest | None = None
        _benchflow_verifier_inputs: dict | None = None

        async def exec(self, *args, **kwargs):
            calls.append(args)
            raise AssertionError("no exec in off")

    env = Env()
    env._benchflow_image_manifest = _walk("pre-agent", IMAGE)

    record = asyncio.run(im.detect(env, "off", timeout_sec=10))

    assert record["status"] == "off" and calls == []
    assert env._benchflow_verifier_inputs is record


def test_the_record_is_written_to_the_trials_integrity_folder(tmp_path):
    record = im.detector_record(
        "audit", _walk("pre-agent", IMAGE), _walk("post-turn", IMAGE)
    )

    im.write_detector_record(tmp_path, record)

    written = json.loads((tmp_path / "integrity" / im.DETECTOR_RECORD).read_text())
    assert written == record
    assert written["schema"] == "benchflow.verifier_inputs.v1"


def test_a_post_turn_walk_is_not_a_record_of_the_image():
    """The sweep and the guard read only a pre-turn record."""
    text = _walk("post-turn", IMAGE).to_json()

    walked = im.load(text)
    assert walked is not None and walked.taken == "post-turn"
    assert im.load(text.replace("post-turn", "after")) is None


# -- in a real container (skipped without Docker) -----------------------------


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10
            ).returncode
            == 0
        )
    except Exception:
        return False


class _Box:
    def __init__(self, name: str) -> None:
        self.name = name

    async def exec(self, command: str, user=None, **kwargs):
        done = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                str(user or "root"),
                self.name,
                "sh",
                "-c",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )

    def sh(self, command: str) -> str:
        done = subprocess.run(
            ["docker", "exec", self.name, "sh", "-c", command],
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        return done.stdout


@pytest.fixture
def box():
    if not _docker_ready():
        pytest.skip("docker required")
    name = f"bf-detector-{uuid.uuid4().hex[:8]}"
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--network",
            "none",
            "--name",
            name,
            "python:3.12-slim",
            "sleep",
            "900",
        ],
        capture_output=True,
        text=True,
    )
    if run.returncode != 0:
        pytest.skip(f"could not start the container: {run.stderr[:200]}")
    b = _Box(name)
    b.sh(
        "mkdir -p /testbed/tests /testbed/pkg && printf '[pytest]\\naddopts = -ra\\n' > /testbed/tox.ini"
        " && printf 'import pytest\\n' > /testbed/tests/conftest.py"
        " && printf 'def f():\\n    return 1\\n' > /testbed/pkg/mod.py"
    )
    try:
        yield b
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@pytest.mark.parametrize(
    ("agent", "flags", "changes"),
    [
        ("true", set(), set()),
        (
            "printf 'addopts = -p evil\\n' >> /testbed/tox.ini"
            " && printf 'def pytest_runtest_call(): pass\\n' > /testbed/conftest.py"
            " && printf 'def pytest_configure(config): pass\\n' > /testbed/pytest_evil.py"
            " && printf 'def f():\\n    return 2\\n' > /testbed/pkg/mod.py",
            {
                ("/testbed/tox.ini", "edited"),
                ("/testbed/conftest.py", "planted"),
            },
            # A pytest_*.py module is a change (the lead's 10:30Z tiers).
            {("/testbed/pkg/mod.py", "edited"), ("/testbed/pytest_evil.py", "planted")},
        ),
    ],
    ids=["untouched", "ini-conftest-plugin-and-a-source-edit"],
)
def test_the_detector_in_a_container(box, agent, flags, changes):
    manifest = asyncio.run(im.record(box, "pre-agent", timeout_sec=300))
    assert manifest is not None
    on_disk = box.sh(f"sha256sum {im.MANIFEST_PATH}")
    box.sh(agent)

    record = asyncio.run(im.detect(box, "audit", timeout_sec=300))

    assert {(f["path"], f["kind"]) for f in record["flags"]} == flags
    assert {(c["path"], c["kind"]) for c in record["changes"]} == changes
    assert record["status"] == ("flagged" if flags else "checked")
    # The walk after the turn writes nothing in the sandbox.
    assert box.sh(f"sha256sum {im.MANIFEST_PATH}") == on_disk
