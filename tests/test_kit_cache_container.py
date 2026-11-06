"""PERF-SBX: a kit restored from this host's kit cache is the kit a build
makes, in a real container.

- A fresh container's built kit, exported as the cache keeps it, placed in a
  second fresh container of the same image by ``kit_prepare_command(
  "restore")``: every entry of the kit (type, mode, owner, link target,
  bytes) is the same after the restore as after the build, and after the
  seal RECORD, MANIFEST and every entry are the same again; on a glibc and a
  musl Python image and on an image without Python.
- A stage that is not root's alone is refused, removed, and leaves no kit;
  a sandbox that already holds a kit keeps it and drops the stage.
- The daemon's own diff of a container decides the cache: BenchFlow's
  uploads alone pass, a file the image's entrypoint (or anything) wrote in a
  system folder does not.

Skipped without a Docker daemon.
"""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import shutil
import subprocess
import uuid

import pytest

from benchflow._runtime_tree import KIT, kit_prepare_command
from benchflow.sandbox import _kit_cache

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

IMAGES = ("python:3.12-slim", "python:3.12-alpine", "debian:bookworm-slim")
# Every entry of the kit, by the image's own find (GNU, BusyBox) and
# sha256sum; sorted.
KIT_TREE = (
    f"cd {KIT} && find . -print | sort | while IFS= read -r p; do "
    'if [ -L "$p" ]; then echo "l $(stat -c %a:%u:%g "$p") $p $(readlink "$p")"; '
    'elif [ -d "$p" ]; then echo "d $(stat -c %a:%u:%g "$p") $p"; '
    'else echo "f $(stat -c %a:%u:%g "$p") $p $(sha256sum < "$p")"; fi; done'
)
KIT_FILES = (
    f"cd {KIT} && for f in RECORD MANIFEST IMAGE_ENV_OK IMAGE_PYTHON_PATHS.json "
    'IMAGE_PYTHON_PATHS.jsonl; do echo "== $f"; cat "$f" 2>/dev/null; done'
)


@pytest.fixture(autouse=True)
def enable_kit_cache_mechanism(monkeypatch):
    """Keep the deferred 0.8.4 mechanism covered while production disables it."""
    monkeypatch.setattr(_kit_cache, "ENABLED", True)


def _docker_ready() -> bool:
    try:
        probe = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except Exception:
        return False
    return probe.returncode == 0


def _dx(name: str, command: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "exec", "-u", "0", name, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


@contextlib.contextmanager
def _fresh(image: str):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "psbx-kc-" + uuid.uuid4().hex[:8]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "perf-sbx=test",
         "--network", "none", image, "sleep", "900"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {image}: {run.stderr.strip()}")
    try:
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _export(name: str) -> bytes:
    """The kit as the cache keeps it: the daemon's archive of the kit folder."""
    out = subprocess.run(["docker", "cp", f"{name}:{KIT}", "-"], capture_output=True)
    assert out.returncode == 0, out.stderr
    return out.stdout


def _stage(name: str, data: bytes, *, to_root: bool = False, tmp=None) -> str:
    """The kept kit in a fresh stage: a folder made first, or (*to_root*, as
    the sandbox does beside its start) one archive to ``/`` that makes it."""
    stage = f"/.runtime-kit-{uuid.uuid4().hex}"
    if to_root:
        kept = tmp / "kept.tar"
        kept.write_bytes(data)
        data = _kit_cache.staged_tar(kept, stage[1:])
        target = f"{name}:/"
    else:
        assert _dx(name, f"mkdir -m 700 {stage}").returncode == 0
        target = f"{name}:{stage}"
    put = subprocess.run(["docker", "cp", "-", target], input=data, capture_output=True)
    assert put.returncode == 0, put.stderr
    if to_root:
        mode = _dx(name, f"stat -c %a:%u:%g {stage}").stdout.strip()
        assert mode == "700:0:0", mode
    return stage


@pytest.mark.parametrize("to_root", [False, True])
@pytest.mark.parametrize("image", IMAGES)
def test_a_restored_kit_is_the_built_kit_before_and_after_the_seal(
    image, to_root, tmp_path
):
    with _fresh(image) as built, _fresh(image) as restored:
        made = _dx(built, kit_prepare_command("build"))
        assert made.returncode == 0, made.stderr
        assert "BF kit-built" in made.stdout.splitlines()
        tree = _dx(built, KIT_TREE)
        assert tree.returncode == 0 and tree.stdout, tree.stderr

        stage = _stage(restored, _export(built), to_root=to_root, tmp=tmp_path)
        placed = _dx(restored, kit_prepare_command("restore", stage=stage))
        assert placed.returncode == 0, placed.stderr
        assert "BF kit-built" in placed.stdout.splitlines()
        assert _dx(restored, f"[ ! -e {stage} ]").returncode == 0
        assert _dx(restored, KIT_TREE).stdout == tree.stdout

        sealed = [_dx(c, kit_prepare_command("seal")) for c in (built, restored)]
        assert [s.returncode for s in sealed] == [0, 0], [s.stderr for s in sealed]
        assert sealed[0].stdout == sealed[1].stdout
        assert _dx(built, KIT_FILES).stdout == _dx(restored, KIT_FILES).stdout
        assert _dx(built, KIT_TREE).stdout == _dx(restored, KIT_TREE).stdout


def test_a_stage_not_roots_alone_is_refused_and_removed():
    with _fresh(IMAGES[0]) as built, _fresh(IMAGES[0]) as restored:
        assert _dx(built, kit_prepare_command("build")).returncode == 0
        stage = _stage(restored, _export(built))
        assert _dx(restored, f"chown 1000 {stage}/rt/libexec/python3").returncode == 0
        placed = _dx(restored, kit_prepare_command("restore", stage=stage))
        assert placed.returncode != 0
        assert "not root's alone" in placed.stderr
        assert (
            _dx(restored, f"[ ! -e {stage} ] && [ ! -e {KIT}/.built ]").returncode == 0
        )


def test_a_sandbox_that_has_a_kit_keeps_it_and_drops_the_stage():
    with _fresh(IMAGES[0]) as name:
        assert _dx(name, kit_prepare_command("all")).returncode == 0
        before = _dx(name, KIT_TREE).stdout
        stage = _stage(name, _export(name))
        again = _dx(name, kit_prepare_command("restore", stage=stage))
        assert again.returncode == 0, again.stderr
        assert "BF kit-record" in again.stdout
        assert _dx(name, f"[ ! -e {stage} ]").returncode == 0
        assert _dx(name, KIT_TREE).stdout == before


def _changes(name: str):
    from benchflow.sandbox._engine_api import EngineClient

    async def get():
        client = EngineClient("/var/run/docker.sock")
        return (await client.call("GET", f"/containers/{name}/changes")).json()

    return asyncio.run(get())


@pytest.mark.parametrize(
    "write, decides",
    [
        (
            "mkdir -p /logs/agent /solution && : > /instruction.md && : > /solution/solve.sh",
            None,
        ),
        ("echo planted > /usr/local/bin/python3", "/usr"),
        ("rm -f /usr/bin/sha256sum", "/usr"),
        ("mkdir -p /opt/runtime", "/opt"),
    ],
)
def test_the_daemons_diff_decides_whether_the_cache_is_used(write, decides):
    with _fresh(IMAGES[2]) as name:
        assert _dx(name, write).returncode == 0
        why = _kit_cache.unchanged(_changes(name))
        if decides is None:
            assert why is None
        else:
            assert why is not None and why.startswith(decides), why


def test_the_recipe_restore_refuses_a_stage_name_it_did_not_make():
    for stage in ("/tmp/x", "/.runtime-kit-../etc", "/.runtime-kit-" + "g" * 32):
        with pytest.raises(ValueError):
            kit_prepare_command("restore", stage=stage)
    assert shlex.quote("/.runtime-kit-" + "a" * 32) in kit_prepare_command(
        "restore", stage="/.runtime-kit-" + "a" * 32
    )
