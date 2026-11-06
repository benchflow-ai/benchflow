"""Old/new kit receipts and tamper refusals in real root containers."""

from __future__ import annotations

import shutil
import uuid

import pytest

from benchflow._runtime_tree import (
    KIT,
    KIT_CHANNEL_PYTHON,
    kit_load_command,
    kit_prepare_command,
)
from benchflow.sandbox import _channel_python as channel
from benchflow.sandbox import _delegate_disarm as stage
from benchflow.sandbox import _staged_python_layer as cached
from tests.test_en229_layer_build_failures_docker import _docker
from tests.test_en229_python_layer import _digest

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.timeout(120),
]


@pytest.mark.parametrize("state", ["valid", "absent", "tampered", "image-python"])
def test_old_new_kit_records_and_manifest_refusals(tmp_path, monkeypatch, state):
    """Copying channel Python preserves receipts; its mutation fails kit load."""
    image = "python:3.12-slim" if state == "image-python" else "ubuntu:24.04"
    inspected = _docker("image", "inspect", "--format", "{{.Id}}", image, check=False)
    if inspected.returncode:
        pytest.skip("cached fixture image required; this test never pulls")
    image = inspected.stdout.strip()
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    (source / "python/bin").mkdir(parents=True)
    (source / "python").chmod(0o755)
    (source / "python/bin").chmod(0o755)
    (source / "python/bin/python3.12").write_text("#!/bin/sh\ntouch /executed\n")
    (source / "python/bin/python3.12").chmod(0o755)
    pin = stage.STAGED_PYTHON_SHA256["x86_64", "gnu"]
    monkeypatch.setattr(stage, "STAGED_PYTHON_SHA256", {("x86_64", "gnu"): pin})
    monkeypatch.setitem(
        cached.TREE_SHA256, pin, _digest(source, "--owner=0", "--group=0")
    )
    new = kit_prepare_command()
    with monkeypatch.context() as patch:
        # The only added step: omitting it reconstructs the original boundary.
        patch.setattr(channel, "prepare_command", lambda: "")
        old = kit_prepare_command()
    assert new.replace(channel.prepare_command(), "") == old
    src = stage.staged_python_dir("x86_64", "gnu")
    setup = ""
    if state != "absent":
        setup = (
            f"mkdir -p {src}; cp -r /input/source/. {src}; chmod 700 {src}; "
            f"chown -R 0:0 {src}; "
        )
    if state == "tampered":
        setup += f"echo changed >> {stage.staged_python_bin('x86_64', 'gnu')}; "
    answers = []
    for name, command in (("old", old), ("new", new)):
        has_channel = name == "new" and state == "valid"
        changed = channel.CHANNEL_BINARY if has_channel else f"{KIT}/IMAGE_PYTHON_GUARD"
        tail = (
            f"\ntest ! -e /executed || exit 90\n"
            f"test {'-x' if has_channel else '! -e'} {KIT_CHANNEL_PYTHON} || exit 91\n"
            f"echo changed >> {changed}\n"
            # The same kit-load failure message must reach the host.
            f"{kit_load_command()}\n"
        )
        script = tmp_path / f"{name}.sh"
        script.write_text(setup + "\n" + command + tail)
        result = _docker(
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "root",
            "--name",
            f"en229-l13-kit-{uuid.uuid4().hex[:12]}",
            "-v",
            f"{tmp_path}:/input:ro",
            image,
            "sh",
            f"/input/{name}.sh",
            check=False,
        )
        assert result.returncode not in (0, 90, 91), result.stdout + result.stderr
        assert "changed" in result.stderr, result.stdout + result.stderr
        answers.append((result.returncode, result.stdout, result.stderr))
    assert answers[0] == answers[1]
