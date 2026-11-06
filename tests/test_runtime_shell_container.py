"""Guards FIX-RUNTIME-SH against the runtime parse failure at 1f02fbc4c.

Run with a Docker daemon. Images must be available or pullable; a failed pull
or setup is a failure. Set BENCHFLOW_TEST_SURICATA_IMAGE to the image built
from SkillsBench f860ecf02ef53412a030f8661c83d644b739ff01's
suricata-custom-exfil/environment to include that full task image.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid

import pytest

from benchflow._runtime_tree import (
    kit_load_command,
    kit_prepare_command,
    parse_kit_record,
)
from benchflow.sandbox._base import ExecResult
from benchflow.sandbox._exec_receipt import (
    new_receipt_nonce,
    read_receipt,
    receipt_command,
)

BASH_VERSIONS = ("3.2", "4.0", "4.1", "4.2", "4.3", "4.4", "5.0", "5.1", "5.2", "5.3")
SHELLS = [
    pytest.param((f"bash:{version}", mode), id=f"bash-{version}-{mode}")
    for version in BASH_VERSIONS
    for mode in ("sh", "posix")
] + [
    pytest.param(("debian:bookworm-slim", "sh"), id="dash"),
    pytest.param(("busybox:1.37", "sh"), id="busybox-ash"),
    pytest.param(("alpine:3.20", "sh"), id="alpine-ash"),
    pytest.param(
        (os.environ.get("BENCHFLOW_TEST_SURICATA_IMAGE"), "sh"), id="suricata"
    ),
]


def docker(*args, script=None, timeout=300):
    return subprocess.run(
        ["docker", *args], input=script, text=True, capture_output=True, timeout=timeout
    )


@pytest.fixture(params=SHELLS)
def shell_box(request):
    if shutil.which("docker") is None or docker("info", timeout=10).returncode:
        pytest.skip("Docker daemon required")
    image, mode = request.param
    if image is None:
        pytest.skip(
            "BENCHFLOW_TEST_SURICATA_IMAGE must name the built pinned task image"
        )
    name = "bf-runtime-sh-" + uuid.uuid4().hex[:12]
    started = docker(
        "run",
        "-d",
        "--name",
        name,
        "--cpus=1",
        "--memory=1g",
        "--network=none",
        "--label",
        os.environ.get("BENCHFLOW_TEST_CONTAINER_LABEL", "runtime-shell=1"),
        "--entrypoint",
        "/bin/sh",
        image,
        "-c",
        "sleep 1800",
    )
    try:
        assert started.returncode == 0, started.stderr
        if image.startswith("bash:"):
            linked = docker(
                "exec", name, "/bin/sh", "-c", "ln -sf /usr/local/bin/bash /bin/sh"
            )
            assert linked.returncode == 0, linked.stderr
        shell = ["/usr/local/bin/bash", "--posix"] if mode == "posix" else ["/bin/sh"]
        yield name, shell
    finally:
        docker("rm", "-f", name)


def run_receipt(box, command):
    name, shell = box
    nonce = new_receipt_nonce()
    script = receipt_command(command, nonce)
    # The transport also evals the submitted text. Exercise that extra parse
    # without putting the large payload in argv or changing its bytes.
    done = docker("exec", "-i", name, *shell, "-c", 'eval "$(cat)"', script=script)
    result = read_receipt(
        ExecResult(stdout=done.stdout, stderr=done.stderr, return_code=done.returncode),
        nonce,
    )
    return result


def test_full_runtime_build_seal_and_tamper_refusal(shell_box):
    """1f02fbc4c fails before build on old Bash; the full checks must still run."""
    built = run_receipt(shell_box, kit_prepare_command("build"))
    assert built.return_code == 0, built.stderr
    assert "BF kit-built" in (built.stdout or "").splitlines()
    sealed = run_receipt(shell_box, kit_prepare_command("seal"))
    assert sealed.return_code == 0, sealed.stderr
    assert parse_kit_record(sealed.stdout) is not None
    loaded = run_receipt(shell_box, kit_load_command())
    assert loaded.return_code == 0, loaded.stderr
    assert parse_kit_record(loaded.stdout) == parse_kit_record(sealed.stdout)
    tamper = docker(
        "exec",
        shell_box[0],
        "/bin/sh",
        "-c",
        "echo tampered >> /opt/benchflow/rt/RECORD",
    )
    assert tamper.returncode == 0, tamper.stderr
    refused = run_receipt(shell_box, kit_prepare_command("seal"))
    assert refused.return_code != 0
    assert "changed since BenchFlow made it" in (refused.stderr or "")
