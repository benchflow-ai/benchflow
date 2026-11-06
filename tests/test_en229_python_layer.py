"""Guards 3617eec1: authenticate cached Python before reuse."""

from __future__ import annotations

import hashlib
import os
import shlex
import socket
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from benchflow.sandbox import _delegate_disarm as stage
from benchflow.sandbox import _staged_python_layer as cached
from benchflow.sandbox import daytona_layers as layers
from benchflow.sandbox import lockdown
from tests._root_stand_in import run_sh

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="GNU tar/Linux runtime")


def _digest(path, *extra):
    result = subprocess.run(
        ["tar", *cached.TAR_OPTIONS.split(), *extra, "-C", str(path), "-cf", "-", "."],
        check=True,
        capture_output=True,
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
    )
    assert not result.stderr
    return hashlib.sha256(result.stdout + cached.TREE_SUFFIX.encode()).hexdigest()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    root.mkdir(mode=0o700)
    folder = root / "python" / "bin"
    folder.mkdir(parents=True)
    folder.chmod(0o755)
    folder.parent.chmod(0o755)
    binary = folder / "python3.12"
    binary.write_text(f"#!/bin/sh\ntouch {tmp_path}/executed\n")
    binary.chmod(0o755)
    (folder / "python3").symlink_to("python3.12")
    monkeypatch.setattr(stage, "STAGED_ROOT", str(tmp_path))
    monkeypatch.setattr(stage, "staged_python_dir", lambda *_: str(root))
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_TRUSTED_UIDS", ("0", str(os.getuid()))
    )
    pin = stage.STAGED_PYTHON_SHA256["x86_64", "gnu"]
    monkeypatch.setitem(cached.TREE_SHA256, pin, _digest(root))
    return root


def _check(tree, *, prefix="", env=None):
    return run_sh(
        prefix
        + lockdown._cloud_metadata_root_prelude()
        + cached.tree_check("x86_64", "gnu"),
        capture_output=True,
        text=True,
        timeout=15,
        env=env or {"PATH": "/usr/bin:/bin"},
    )


def test_cached_tree_is_checked_without_starting_python(tree):
    """Guards 3617eec1: the content proof executes only image tools."""
    result = _check(tree)
    assert result.returncode == 0, result.stderr
    assert not (tree.parent / "executed").exists()
    # Timestamps are not executable content or custody; snapshot copies may differ.
    os.utime(tree / "python/bin/python3.12", (1, 1))
    assert _check(tree).returncode == 0


@pytest.mark.parametrize(
    "change", ["bytes", "extra", "missing", "link", "mode", "parent", "socket", "fifo"]
)
def test_cached_tree_rejects_changes_before_execution(tree, change):
    """Guards 3617eec1: contents, topology and custody all count."""
    binary = tree / "python/bin/python3.12"
    held = None
    if change == "bytes":
        binary.write_text("#!/bin/sh\nexit 0\n")
    elif change == "extra":
        (tree / "python/extra.py").write_text("pass")
    elif change == "missing":
        binary.unlink()
    elif change == "link":
        link = tree / "python/bin/python3"
        link.unlink()
        link.symlink_to("/usr/bin/python3")
    elif change == "mode":
        binary.chmod(0o777)
    elif change == "parent":
        tree.parent.chmod(0o777)
    elif change == "socket":
        held = socket.socket(socket.AF_UNIX)
        held.bind(str(tree / "socket"))
    else:
        os.mkfifo(tree / "fifo")
    try:
        assert _check(tree).returncode != 0
        assert not (tree.parent / "executed").exists()
    finally:
        if held:
            held.close()


def test_cached_tree_hash_keeps_numeric_owners(tree, monkeypatch):
    """Guards 3617eec1: read-only bytes do not hide another owner."""
    pin = stage.STAGED_PYTHON_SHA256["x86_64", "gnu"]
    monkeypatch.setitem(
        cached.TREE_SHA256, pin, _digest(tree, f"--owner={os.getuid() + 1}")
    )
    assert _check(tree).returncode != 0


@pytest.mark.parametrize("tool", ["tar", "sha256sum"])
def test_cached_tree_needs_both_checked_tools(tree, tool):
    """Guards 3617eec1: missing tools force ordinary staging."""
    command = (
        lockdown._cloud_metadata_root_prelude()
        + f'tool() {{ [ "$1" != {tool} ] && command -v "$1"; }}; '
    )
    result = run_sh(command + cached.tree_check("x86_64", "gnu"), capture_output=True)
    assert result.returncode != 0


def test_cached_tree_ignores_tar_environment_and_rejects_partial_stream(
    tree, monkeypatch
):
    """Guards 3617eec1: no injected exclusions or successful partial hash."""
    assert (
        _check(
            tree,
            env={
                "PATH": "/usr/bin:/bin",
                "TAR_OPTIONS": "--exclude=*",
                "POSIXLY_CORRECT": "1",
            },
        ).returncode
        == 0
    )
    # A tar that emits the exact bytes then fails must not get the success suffix.
    wrapper = tree.parent / "failing-tar"
    wrapper.write_text('#!/bin/sh\n/usr/bin/tar "$@"\nexit 1\n')
    wrapper.chmod(0o755)
    command = (
        lockdown._cloud_metadata_root_prelude()
        + f'tool() {{ case $1 in tar) echo {shlex.quote(str(wrapper))};; *) command -v "$1";; esac; }}; '
    )
    assert (
        run_sh(
            command + cached.tree_check("x86_64", "gnu"), capture_output=True
        ).returncode
        != 0
    )


def test_new_pin_without_tree_digest_is_not_reused(tree, monkeypatch):
    """Guards 3617eec1: pin updates cannot inherit old content proof."""
    monkeypatch.setattr(stage, "STAGED_PYTHON_SHA256", {("x86_64", "gnu"): "1" * 64})
    assert _check(tree).returncode != 0


@pytest.mark.parametrize("tampered", [False, True])
async def test_probe_reuses_only_authenticated_tree_else_stages(
    tree, monkeypatch, tampered
):
    """Guards 3617eec1: a missing or failed cache keeps the upload path."""
    from tests.test_cloud_metadata_delegate_disarm import _StageEnv

    monkeypatch.setattr(
        stage,
        "STAGED_PYTHON_SHA256",
        {("x86_64", "gnu"): stage.STAGED_PYTHON_SHA256["x86_64", "gnu"]},
    )
    monkeypatch.setattr(stage, "PYTHON_INSTALL_DIR", str(tree.parent / "managed"))
    monkeypatch.setattr(
        stage, "host_tarball", lambda *_: tree.parent / "archive.tar.gz"
    )
    if tampered:
        (tree / "python/bin/python3.12").write_text("changed")
    report = run_sh(stage.probe_cmd(), capture_output=True, text=True)
    assert report.returncode == 0, report.stderr
    env = _StageEnv(report.stdout)
    python, why = await stage.find_python(env)
    assert why == "" and python is not None and python.source == "staged"
    assert bool(env.uploads) is tampered
    assert not (tree.parent / "executed").exists()


def test_layer_identity_opt_out_and_disk(monkeypatch):
    """Guards 3617eec1: recipes are stable, pin-dependent and optional."""
    plan = layers.plan_install_layers(agent=None, model=None, stage_python=True)
    layer = plan.python
    assert layer is not None
    recipe = layers.layered_recipe(plan, "FROM task\n")
    assert list(recipe.keys) == ["python-delegate"]
    assert "ADD https://" in "\n".join(recipe.lines)
    assert "sha256sum" in layer.script and "--no-same-owner" in layer.script
    assert stage.STAGED_PYTHON_SHA256["x86_64", "gnu"] in layer.script
    assert recipe == layers.layered_recipe(plan, "FROM task\n")
    assert plan.footprint_mb == cached.FOOTPRINT_MB
    assert layers.disk_gib(10240, plan.footprint_mb, 10240) == 10
    assert layers.plan_install_layers(agent=None, model=None).layers == ()
    fresh = layers.plan_install_layers(
        agent=None, model=None, stage_python=True, fresh=True
    )
    assert fresh.layers == () and fresh.footprint_mb == cached.FOOTPRINT_MB
    monkeypatch.setattr(stage, "STAGED_PYTHON_SHA256", {("x86_64", "gnu"): "1" * 64})
    changed = layers.plan_install_layers(agent=None, model=None, stage_python=True)
    assert changed.python.key("FROM task\n") != layer.key("FROM task\n")


@pytest.mark.parametrize(
    "outcome",
    [
        "ok",
        "checksum",
        "missing-downloader",
        "image-python",
        "unsupported-tar",
        "bundled-archive",
        "no-libc",
        "unpack-failure",
        "timeout",
        "nonroot",
    ],
)
def test_layer_build_is_pinned_and_discards_failed_attempts(
    tmp_path, monkeypatch, outcome
):
    """Review F1/F2 against 9ceade61's committed partial Python."""
    root = tmp_path / "runtime"
    source = tmp_path / "source"
    (source / "python/bin").mkdir(parents=True)
    binary = source / "python/bin/python3.12"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o775)
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(source / "python", arcname="python")
    pin = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(stage, "RUNTIME_ROOT", str(root))
    monkeypatch.setattr(stage, "STAGED_ROOT", str(root / "staged"))
    monkeypatch.setattr(stage, "PYTHON_INSTALL_DIR", str(root / "managed"))
    monkeypatch.setattr(stage, "STAGED_PYTHON_SHA256", {("x86_64", "gnu"): pin})
    monkeypatch.setattr(cached, "TREE_SHA256", {pin: "0" * 64})
    bundled = tmp_path / "bundled.tar.gz"
    if outcome == "bundled-archive":
        bundled.write_bytes(archive.read_bytes())
    monkeypatch.setattr(cached, "BUNDLED_ARCHIVE", str(bundled))
    monkeypatch.setattr(
        cached, "IMAGE_PYTHONS", (str(binary),) if outcome == "image-python" else ()
    )
    monkeypatch.setattr(layers, "LAYER_KEY_DIR", str(tmp_path / "keys"))
    monkeypatch.setattr(
        lockdown, "_CLOUD_METADATA_TRUSTED_UIDS", ("0", str(os.getuid()))
    )
    plan = layers.plan_install_layers(agent=None, model=None, stage_python=True)
    assert plan.python is not None
    command = layers.layer_run_line(plan.python, "key").split("\nRUN ", 1)[1]
    # Stand in for root's chown on the local test user's files.
    command = command.replace("chown 0:0", f"chown {os.getuid()}:{os.getgid()}")
    command = command.replace("a=$(uname -m)", "a=x86_64")
    downloader = tmp_path / "download"
    downloader.write_text(f'#!/bin/sh\ncp {shlex.quote(str(archive))} "$5"\n')
    downloader.chmod(0o755)
    # Inject only the downloader lookup, leaving actual tar/hash and custody intact.
    command = command.replace(
        "bf_curl=$(tool curl)", "bf_curl=" + shlex.quote(str(downloader))
    )
    command = command.replace("bf_wget=$(tool wget)", "bf_wget=")
    if outcome in {"missing-downloader", "bundled-archive"}:
        command = command.replace("bf_curl=" + shlex.quote(str(downloader)), "bf_curl=")
    if outcome == "checksum":
        archive.write_bytes(b"not the pinned archive")
    if outcome == "unsupported-tar":
        command = command.replace("bf_tar=$(tool tar)", "bf_tar=/usr/bin/false")
    if outcome == "no-libc":
        command = command.replace("/lib/ld-", str(tmp_path / "missing/ld-"))
        command = command.replace("/lib64/ld-", str(tmp_path / "missing64/ld-"))
    if outcome in {"unpack-failure", "timeout"}:
        wrapper = tmp_path / "partial-tar"
        wrapper.write_text(
            '#!/bin/sh\ncase "$1" in --version) exec /usr/bin/tar "$@";; esac\n'
            '/usr/bin/tar "$@" || exit 1\n'
            + ("sleep 10\n" if outcome == "timeout" else "exit 1\n")
        )
        wrapper.chmod(0o755)
        command = command.replace(
            "bf_tar=$(tool tar)", "bf_tar=" + shlex.quote(str(wrapper))
        )
        command = command.replace("timeout 300 ", "timeout 1 ")
    if outcome == "nonroot":
        command = command.replace('[ "$(id -u)" = 0 ]', '[ "$(id -u)" = 99999 ]')
    result = run_sh(command, capture_output=True, text=True, timeout=30)
    failed = outcome in {"checksum", "unpack-failure", "timeout", "nonroot"}
    # Only successful or unattemptable installs may commit a layer.
    assert (result.returncode != 0) is failed, result.stderr
    installed = Path(stage.staged_python_bin("x86_64", "gnu"))
    assert installed.exists() is (
        outcome in {"ok", "bundled-archive", "unpack-failure", "timeout"}
    )
    assert Path(plan.python.marker).exists() is (outcome in {"ok", "bundled-archive"})
    assert not bundled.exists()
    if installed.exists():
        assert installed.stat().st_mode & 0o022 == 0


async def test_failed_python_recipe_stays_plain_in_this_process(tmp_path, monkeypatch):
    """Review: Python-only failures use the shared failure memory."""
    from tests.test_daytona_install_layers import FakeDaytona, _start
    from tests.test_dfx_scale_r3 import _fake_client

    client = _fake_client(monkeypatch, FakeDaytona, fail_layered_builds=True)
    for trial in ("one", "two"):
        sandbox = await _start(
            client, tmp_path / trial, agent=None, model=None, stage_python=True
        )
        assert sandbox._install_layer_keys == {}
    assert len(client.requests) == 3
    assert "ADD " in client.requests[0].image.dockerfile()
    assert (
        client.requests[1].image.dockerfile() == client.requests[2].image.dockerfile()
    )


async def test_single_trial_keeps_python_layer(tmp_path, monkeypatch):
    """Guards 3617eec1: the small layer survives the optional reuse gate."""
    pytest.importorskip("daytona")
    from tests.test_daytona_install_layers import FakeDaytona, _start
    from tests.test_dfx_scale_r3 import _fake_client

    monkeypatch.setenv(layers.LAYER_REUSE_ENV, "0")
    monkeypatch.setattr(layers, "_BUILT", set())
    client = _fake_client(monkeypatch, FakeDaytona)
    sandbox = await _start(client, tmp_path, stage_python=True, only_if_reused=True)
    assert sandbox.cached_install_layer("python", "delegate") is not None
    assert "proxy-litellm" not in sandbox._install_layer_keys


async def test_archive_build_failure_uses_plain_image(tmp_path, monkeypatch):
    """Guards b0c64310: a failed ADD keeps the original image fallback."""
    pytest.importorskip("daytona")
    from tests.test_daytona_install_layers import FakeDaytona, _start
    from tests.test_dfx_scale_r3 import _fake_client

    client = _fake_client(monkeypatch, FakeDaytona, fail_layered_builds=True)
    sandbox = await _start(client, tmp_path, stage_python=True)
    assert sandbox._install_layer_keys == {}
    assert "ADD https://" not in client.requests[-1].image.dockerfile()


@pytest.mark.parametrize("opt_out", ["fresh", "persistent-env", "compose"])
def test_python_layer_uses_existing_opt_outs(tmp_path, opt_out):
    """Guards 3617eec1: mutable and compose runs stage normally."""
    pytest.importorskip("daytona")
    from tests.test_daytona_install_layers import FakeDaytona, _sandbox

    sandbox = _sandbox(FakeDaytona(), tmp_path)
    sandbox._install_layers = None
    sandbox._compose_mode = opt_out == "compose"
    sandbox._persistent_env = (
        {"example": "value"} if opt_out == "persistent-env" else {}
    )
    sandbox.configure_install_layers(
        agent=None, model=None, stage_python=True, fresh_install=opt_out == "fresh"
    )
    assert sandbox._install_layers is None or sandbox._install_layers.layers == ()


@pytest.mark.parametrize(
    "user,agent,verifier,expected",
    [
        ("agent", "claude-agent-acp", False, True),
        (None, "claude-agent-acp", False, False),
        ("agent", "oracle", False, False),
        ("agent", "claude-agent-acp", True, False),
    ],
)
def test_rollout_plans_python_only_for_guarded_user(
    tmp_path, user, agent, verifier, expected
):
    """Guards 3617eec1: no-user and verifier-only recipes are unchanged."""
    from benchflow.rollout import Rollout, RolloutConfig
    from tests.test_daytona_install_layers import _PlannedSandbox

    rollout = Rollout(RolloutConfig(task_path=tmp_path, agent=agent, sandbox_user=user))
    rollout._env = _PlannedSandbox()
    rollout._verifier_only = verifier
    rollout._configure_daytona_layers()
    assert rollout._env.plans[0]["stage_python"] is expected
