"""The conftest sweep keeps the image's own conftest.py wherever it sits.

The sweep removed every conftest.py outside /tests, and kept only a workspace
file byte for byte the pre-agent copy's. An image whose repository is not the
workspace (featurebench: ``WORKDIR /``, so the workspace is /root, the repo
/testbed) or that keeps a second copy of it outside the workspace (SWT-bench's
/root/tmp_repo, which test.sh copies back) lost the repository's conftest.py
files, and a correct solution scored 0.

The image manifest records the sha256 of every conftest.py and pytest
configuration file in the image, as root, before any agent process exists;
the host keeps that copy and writes it back before hardening, so a copy the
agent rewrote, even as root, changes nothing. The sweep keeps a conftest.py
whose bytes match the manifest at the same absolute path, and removes every
other one; without a manifest it runs as before.
"""

from __future__ import annotations

import asyncio
import hashlib
import json

import pytest

from benchflow.sandbox import _hook_cleanup, _image_manifest, lockdown
from tests.test_c8_protected_runtime_container import _dx, _sandbox, box  # noqa: F401
from tests.test_conftest_sweep_repository_files import PLANTED, REPO, _RootFs
from tests.test_hook_cleanup_bfx import INPUTS

MANIFEST = _image_manifest.MANIFEST_PATH


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- the host's copy ---------------------------------------------------------


def test_a_manifest_reads_back_and_names_its_files():
    text = json.dumps(
        {
            "version": 1,
            "taken": "pre-agent",
            "files": {
                "/testbed/tests/conftest.py": {"sha256": _digest(REPO), "size": 1},
                "/testbed/tox.ini": {"sha256": _digest(b"[pytest]\n"), "size": 1},
                "relative/conftest.py": {"sha256": _digest(REPO), "size": 1},
                "/bad/conftest.py": {"sha256": "not-a-digest", "size": 1},
            },
        }
    )

    manifest = _image_manifest.load(text)

    assert manifest is not None
    assert manifest.digest("/testbed/tests/conftest.py") == _digest(REPO)
    assert manifest.digest("/srv/conftest.py") is None
    assert manifest.named("conftest.py") == {
        "/testbed/tests/conftest.py": _digest(REPO)
    }
    assert _image_manifest.load(manifest.to_json()) == manifest


@pytest.mark.parametrize(
    "text",
    [
        "",
        "{",
        "[]",
        '{"version": 2, "taken": "pre-agent", "files": {}}',
        '{"version": 1, "taken": "later", "files": {}}',
    ],
    ids=["empty", "broken", "list", "version", "taken"],
)
def test_anything_else_is_no_manifest(text):
    assert _image_manifest.load(text) is None


@pytest.mark.parametrize(
    ("written", "reaches"),
    [
        (("/app",), False),
        (("/",), True),
        ((_image_manifest.MANIFEST_DIR,), True),
        ((f"{MANIFEST}",), True),
        (("/var/lib/benchflow-image2",), False),
        (("/var/lib/benchflow",), False),
    ],
)
def test_a_manifest_the_transfer_could_have_written_is_never_used(written, reaches):
    assert lockdown._transfer_reaches(written, _image_manifest.MANIFEST_DIR) is reaches


def test_the_manifest_has_a_folder_of_its_own():
    """Not /var/lib/benchflow, which holds the verifier env on some lines: the
    write-back makes its folder root's 0700 and the walk leaves it out."""
    from benchflow.task import verifier_core

    assert _image_manifest.MANIFEST_DIR == "/var/lib/benchflow-image"
    env_parent = getattr(verifier_core, "_VERIFIER_ENV_PARENT", "/var/lib/benchflow")
    assert not env_parent.startswith(_image_manifest.MANIFEST_DIR + "/")
    assert not _image_manifest.MANIFEST_DIR.startswith(env_parent + "/")


# -- the bfx form ------------------------------------------------------------


def _tree() -> dict[bytes, bytes]:
    return {
        b"/testbed/tests/conftest.py": REPO,
        b"/root/tmp_repo/tests/conftest.py": REPO,
        b"/testbed/pkg/conftest.py": PLANTED,
        b"/testbed/new/conftest.py": PLANTED,
        b"/srv/conftest.py": PLANTED,
    }


IMAGE = {
    "/testbed/tests/conftest.py": _digest(REPO),
    "/root/tmp_repo/tests/conftest.py": _digest(REPO),
    "/testbed/pkg/conftest.py": _digest(b"# the image's\n"),
}


async def _bfx_sweep(fs, manifest):
    loop = asyncio.get_running_loop()
    await _hook_cleanup.run(
        fs.call,
        _hook_cleanup.parse_inputs(INPUTS.replace("a:b", "")),
        cleanup_conftests=True,
        workspace="/root",
        workspace_aliases=(),
        verifier_dir="/opt/runtime/verifier",
        deadline=loop.time() + 30,
        agent_user=False,
        baseline="/testbed_verify",
        conftest_manifest=manifest,
    )


@pytest.mark.asyncio
async def test_bfx_form_keeps_the_images_conftests_outside_the_workspace():
    """The workspace is /root: /testbed's and /root/tmp_repo's unchanged
    conftest.py stay; an edited one and two planted ones go."""
    fs = _RootFs(_tree(), {})
    await _bfx_sweep(fs, IMAGE)
    assert fs.tree[b"/testbed/tests/conftest.py"] == REPO
    assert fs.tree[b"/root/tmp_repo/tests/conftest.py"] == REPO
    for gone in (
        b"/testbed/pkg/conftest.py",
        b"/testbed/new/conftest.py",
        b"/srv/conftest.py",
    ):
        assert gone not in fs.tree, gone


@pytest.mark.asyncio
async def test_bfx_form_without_a_manifest_keeps_nothing_outside_the_workspace():
    fs = _RootFs(_tree(), {})
    await _bfx_sweep(fs, None)
    assert b"/testbed/tests/conftest.py" not in fs.tree
    assert b"/root/tmp_repo/tests/conftest.py" not in fs.tree


# -- in a real container: record, write back, sweep ----------------------------


IMAGE_FILES = (
    "mkdir -p /testbed/tests /testbed/pkg /root/tmp_repo/tests /srv && "
    "printf 'repo\\n' > /testbed/tests/conftest.py && "
    "printf 'image\\n' > /testbed/pkg/conftest.py && "
    "printf 'repo\\n' > /root/tmp_repo/tests/conftest.py && "
    "printf '[pytest]\\naddopts = -p pytester\\n' > /testbed/tox.ini && "
    "ln -s /etc/hostname /testbed/linked_conftest_target && "
    "mkdir -p /testbed/link && ln -s /etc/hostname /testbed/link/conftest.py"
)
AGENT_CHANGES = (
    "printf 'edited\\n' > /testbed/pkg/conftest.py && "
    "mkdir -p /testbed/new /opt/planted && "
    "printf 'planted\\n' > /testbed/new/conftest.py && "
    "printf 'planted\\n' > /opt/planted/conftest.py && "
    "printf 'planted\\n' > /root/tmp_repo/conftest.py"
)
KEPT = ("/testbed/tests/conftest.py", "/root/tmp_repo/tests/conftest.py")
GONE = (
    "/testbed/pkg/conftest.py",
    "/testbed/new/conftest.py",
    "/opt/planted/conftest.py",
    "/root/tmp_repo/conftest.py",
    "/testbed/link/conftest.py",
)


def _sweep_in(box, manifest_path):  # noqa: F811
    sweep = lockdown._conftest_sweep_cmd(workspace="/root", manifest=manifest_path)
    ran = _dx(box, "__bf_root_python=/usr/local/bin/python3; " + sweep)
    assert ran.returncode == 0, ran.stderr


def _assert_swept(box):  # noqa: F811
    for path in KEPT:
        assert _dx(box, f"cat {path}").stdout == "repo\n", path
    for path in GONE:
        probe = _dx(box, f"[ -e {path} ] || [ -L {path} ]")
        assert probe.returncode == 1, path


def test_the_recorded_manifest_holds_the_images_files(box):  # noqa: F811
    assert _dx(box, IMAGE_FILES).returncode == 0
    env = _sandbox(box)

    manifest = asyncio.run(_image_manifest.record(env, "pre-agent", timeout_sec=120))

    assert manifest is not None and manifest.taken == "pre-agent"
    assert manifest.digest("/testbed/tests/conftest.py") == _digest(b"repo\n")
    assert manifest.digest("/root/tmp_repo/tests/conftest.py") == _digest(b"repo\n")
    assert manifest.digest("/testbed/tox.ini") is not None
    assert manifest.digest("/testbed/link/conftest.py") is None
    assert _image_manifest.kept(env) is manifest
    assert _dx(
        box, f"stat -c '%u %a' {MANIFEST} {_image_manifest.MANIFEST_DIR}"
    ).stdout.split() == ["0", "600", "0", "700"]


def test_the_shell_sweep_keeps_the_images_conftests_anywhere(box):  # noqa: F811
    """featurebench and SWT-bench: the workspace is /root, the repository's
    conftest.py files are in /testbed and /root/tmp_repo. Unchanged ones stay
    as fresh root-owned files; edited, planted and linked ones go."""
    assert _dx(box, IMAGE_FILES).returncode == 0
    env = _sandbox(box)
    assert asyncio.run(_image_manifest.record(env, "pre-agent", timeout_sec=120))
    assert _dx(box, AGENT_CHANGES).returncode == 0

    path = asyncio.run(lockdown._use_image_manifest(env, None))

    assert path == MANIFEST
    _sweep_in(box, path)
    _assert_swept(box)


def test_a_root_agent_rewriting_the_manifest_changes_nothing(box):  # noqa: F811
    """The agent, as root, lists its planted conftest.py in the sandbox's
    copy: the host's copy is written back over it before the sweep reads it."""
    assert _dx(box, IMAGE_FILES).returncode == 0
    env = _sandbox(box)
    assert asyncio.run(_image_manifest.record(env, "pre-agent", timeout_sec=120))
    assert _dx(box, AGENT_CHANGES).returncode == 0
    forged = json.dumps(
        {
            "version": 1,
            "taken": "pre-agent",
            "files": {
                path: {"sha256": _digest(b"planted\n"), "size": 8}
                for path in ("/testbed/new/conftest.py", "/opt/planted/conftest.py")
            },
        }
    )
    assert _dx(box, f"printf '%s' '{forged}' > {MANIFEST}").returncode == 0

    path = asyncio.run(lockdown._use_image_manifest(env, None))

    _sweep_in(box, path)
    _assert_swept(box)


def test_without_a_manifest_the_sweep_runs_as_before(box):  # noqa: F811
    assert _dx(box, IMAGE_FILES).returncode == 0
    env = _sandbox(box)

    path = asyncio.run(lockdown._use_image_manifest(env, None))

    assert path == ""
    _sweep_in(box, path or None)
    for gone in (*KEPT, "/testbed/pkg/conftest.py"):
        assert _dx(box, f"test -e {gone}").returncode == 1, gone


def test_the_transfer_reaching_the_manifest_turns_it_off(box):  # noqa: F811
    assert _dx(box, IMAGE_FILES).returncode == 0
    env = _sandbox(box)
    assert asyncio.run(_image_manifest.record(env, "pre-transfer", timeout_sec=120))

    path = asyncio.run(
        lockdown._use_image_manifest(env, ("/app", _image_manifest.MANIFEST_DIR))
    )

    assert path == ""


# -- the write-back: inline when small, an admitted upload when large ------------


class _Env:
    """An env that records its execs and uploads."""

    def __init__(self):
        self.execs: list[str] = []
        self.uploads: list[tuple[str, str, object]] = []

    async def exec(self, command, **kwargs):
        from types import SimpleNamespace

        self.execs.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")

    async def upload_file(self, source, target, *, admission=None):
        from pathlib import Path

        self.uploads.append((Path(source).read_text(), target, admission))


def _manifest(entries: int) -> _image_manifest.ImageManifest:
    files = {
        f"/testbed/pkg{i:06d}/module.py": {"sha256": _digest(b"%d" % i), "size": 1}
        for i in range(entries)
    }
    return _image_manifest.ImageManifest(taken="pre-agent", files=files)


def test_a_small_manifest_is_written_back_in_the_execs_own_text():
    env = _Env()
    manifest = _manifest(3)

    assert asyncio.run(_image_manifest.write_back(env, manifest, timeout_sec=30))

    assert env.uploads == []
    assert len(env.execs) == 1 and manifest.to_json() in env.execs[0]
    assert "mkdir -m 700" in env.execs[0] and "chmod 600" in env.execs[0]


def test_a_large_manifest_is_uploaded_as_a_trusted_transfer(monkeypatch):
    """The upload carries BenchFlow's own admission, as the verifier
    runtime's tree does; without one the sandbox refuses it."""
    from benchflow.trust import state

    seen = []

    def admit(env, service, kind, *, reason=None):
        seen.append((service, kind, reason))
        return "admitted"

    monkeypatch.setattr(state, "admit", admit)
    env = _Env()
    manifest = _manifest(4000)
    assert len(manifest.to_json()) > _image_manifest._INLINE_MAX

    assert asyncio.run(_image_manifest.write_back(env, manifest, timeout_sec=30))

    [(text, target, admission)] = env.uploads
    assert text == manifest.to_json() and target == MANIFEST
    assert admission == "admitted"
    assert seen == [("main", state.TRUSTED_TRANSFER, "BenchFlow's image manifest")]
    assert "chown 0:0" in env.execs[-1] and "chmod 600" in env.execs[-1]


def test_the_manifest_records_each_pytests_version(box):  # noqa: F811
    """Read from the installs' metadata names, nothing of the image's run:
    pip's dist-info folder and conda's or setuptools' egg-info file (:
    strict passes no --rootdir to a pytest before 3.5)."""
    made = _dx(
        box,
        "mkdir -p /usr/local/lib/python3.12/site-packages/pytest-8.4.2.dist-info "
        "/opt/conda/envs/testbed/lib/python3.6/site-packages && "
        "touch /opt/conda/envs/testbed/lib/python3.6/site-packages/pytest-3.3.1-py3.6.egg-info",
    )
    assert made.returncode == 0, made.stderr
    env = _sandbox(box)

    manifest = asyncio.run(_image_manifest.record(env, "pre-agent", timeout_sec=120))

    assert manifest is not None
    assert manifest.pytests == {
        "/usr/local/lib/python3.12/site-packages/pytest-8.4.2.dist-info": "8.4.2",
        "/opt/conda/envs/testbed/lib/python3.6/site-packages/pytest-3.3.1-py3.6.egg-info": "3.3.1",
    }
    # The host's copy (the write-back) keeps them.
    assert _image_manifest.load(manifest.to_json()).pytests == manifest.pytests
