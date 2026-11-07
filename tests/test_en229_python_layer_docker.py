"""Failed Python extraction cannot become a cached image."""

from __future__ import annotations

import hashlib
import shlex
import shutil
import tarfile
from dataclasses import replace

import pytest

from benchflow.sandbox import _delegate_disarm as stage
from benchflow.sandbox import _staged_python_layer as cached
from benchflow.sandbox import daytona_layers as layers
from tests.test_en229_layer_build_failures_docker import _docker
from tests.test_en229_layer_build_failures_docker import offline_image as offline_image

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker required"
)


@pytest.mark.parametrize("failure", ["timeout", "nonroot"])
def test_python_attempt_is_discarded_and_plain_staging_works(
    offline_image, monkeypatch, failure
):
    """Review F1/F2 against 9ceade61's masked unpack/user failures."""
    tag, context = offline_image
    source = context / "python"
    (source / "bin").mkdir(parents=True)
    (source / "lib").mkdir()
    (source / "bin/python3.12").write_text("#!/bin/sh\nexit 0\n")
    (source / "bin/python3.12").chmod(0o755)
    (source / "lib/fixture.py").write_text("# offline fixture\n")
    archive = context / "python.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname="python")
    pin = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(stage, "STAGED_PYTHON_SHA256", {("x86_64", "gnu"): pin})
    monkeypatch.setattr(stage, "PYTHON_INSTALL_DIR", "/no-managed-python")
    monkeypatch.setattr(cached, "TREE_SHA256", {pin: "0" * 64})
    monkeypatch.setattr(cached, "IMAGE_PYTHONS", ())
    monkeypatch.setattr(
        cached,
        "archive_add_line",
        lambda: f"COPY python.tar.gz {cached.BUNDLED_ARCHIVE}",
    )
    path = stage.staged_python_dir("x86_64", "gnu")
    (context / "slow-tar").write_text(
        '#!/bin/sh\ncase "$1" in --version) exec /usr/bin/tar "$@";; esac\n'
        '/usr/bin/tar "$@" python/bin/python3.12 || exit 1\n'
        f"test -f {path}/python/bin/python3.12 && test ! -e {path}/python/lib/fixture.py || exit 1\n"
        "echo 'EN229 partial Python unpack'\nsleep 195\n"
    )
    (context / "slow-tar").chmod(0o755)
    user = "USER nobody\n" if failure == "nonroot" else ""
    (context / "Dockerfile.user").write_text(f"FROM {tag}:base\n{user}")
    _docker(
        "build",
        "--network",
        "none",
        "-f",
        str(context / "Dockerfile.user"),
        "-t",
        tag + ":nonroot",
        str(context),
    )
    plan = layers.plan_install_layers(agent=None, model=None, stage_python=True)
    assert plan.python is not None
    layer = replace(
        plan.python,
        timeout_sec=5,
        script=plan.python.script.replace("bf_tar=$(tool tar)", "bf_tar=/w/slow-tar"),
    )
    (context / "Dockerfile.layer").write_text(
        f"FROM {tag}:nonroot\nCOPY slow-tar /w/slow-tar\n"
        + layers.layer_run_line(layer, "en229-python-review")
        + "\n"
    )
    result = _docker(
        "build",
        "--network",
        "none",
        "-f",
        str(context / "Dockerfile.layer"),
        "-t",
        tag + ":candidate",
        str(context),
        check=False,
    )
    assert result.returncode != 0, "failed Python attempt was committed"
    if failure == "timeout":
        assert "EN229 partial Python unpack" in result.stdout + result.stderr
    assert _docker("image", "inspect", tag + ":candidate", check=False).returncode != 0
    command = (
        f"test ! -e {shlex.quote(path)} && "
        + stage.stage_prepare_cmd("x86_64", "gnu")
        + f" && cp /archive.tar.gz {shlex.quote(path)}/python.tar.gz && "
        + stage.stage_unpack_cmd("x86_64", "gnu")
    )
    fallback = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        "root",
        "-v",
        f"{archive}:/archive.tar.gz:ro",
        tag + ":nonroot",
        "sh",
        "-c",
        command,
    )
    assert "BF delegate-python staged" in fallback.stdout, fallback.stdout
    print(
        f"EN229 Python {failure}: candidate rejected; plain runtime staging completed"
    )
