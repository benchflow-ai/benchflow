"""a fixture image is built under a tag of its own, and a build that
fails is an error, never a skip.

The pt-r16 module fixtures built fixed tags. Under xdist two workers built the
same tag at once, Docker's containerd image store refused the second ("already
exists"), and ``_build`` turned that into a skip: CI's sharded unit legs
skipped tests the release gate ran. Every case here fails on bd2912111.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from tests.test_pt_r16_hooks_container import _build, _docker_ready

ROOT = Path(__file__).resolve().parents[1]

# No shell in scratch, so the RUN fails at once and nothing is pulled.
BROKEN = "FROM scratch\nRUN false\n"
EMPTY = 'FROM scratch\nLABEL target="fixture"\n'


def test_a_fixture_image_that_fails_to_build_is_an_error_not_a_skip(
    tmp_path_factory,
):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    with pytest.raises(BaseException) as caught:
        _build(tmp_path_factory, "bf-en676-broken:test", BROKEN)
    assert isinstance(caught.value, pytest.fail.Exception), type(caught.value)
    assert not isinstance(caught.value, pytest.skip.Exception)


def test_each_fixture_build_gets_a_tag_of_its_own(tmp_path_factory):
    if not _docker_ready():
        pytest.skip("docker daemon not running")
    first = _build(tmp_path_factory, "bf-en676-own:test", EMPTY)
    try:
        second = _build(tmp_path_factory, "bf-en676-own:test", EMPTY)
        try:
            assert first != second
            assert first.startswith("bf-en676-own-") and first.endswith(":test")
            assert second.startswith("bf-en676-own-") and second.endswith(":test")
        finally:
            _rmi(second)
    finally:
        _rmi(first)


# One test per fixture that builds an image. With `docker build` failing, each must
# ERROR, never skip (R632 M1: noweb_image still skipped in round 1).
BUILDERS = [
    "tests/test_pt_r16_hooks_container.py::test_debian_system_pythons_sitecustomize_is_kept",
    "tests/test_pt_integ_r2_container.py::test_image_pythonpath_app_hooks_go_through_the_docker_backend",
    "tests/test_hook_cleanup_bfx_container.py::test_pythonpath_folder_matches_today",
    "tests/test_c8_r14_container.py::test_the_guard_is_importable_by_a_debian_venvs_own_pytest",
    "tests/test_c8_r15_container.py::test_a_uv_project_venv_in_the_workspace_keeps_its_editable_pth",
    "tests/test_cloud_metadata_failclosed_docker.py::test_the_agent_user_runs_only_behind_the_block",
    "tests/test_cloud_metadata_failclosed_docker.py::test_an_image_without_timeout_starts_behind_a_block_the_helper_proves",
    "tests/test_cloud_metadata_failclosed_docker.py::test_a_uid_0_agent_on_a_no_web_task_is_held_to_loopback",
    "tests/test_sandbox_user_uid.py::test_the_agent_user_owns_no_file_of_the_image",
    "tests/test_verifier_trusted_path.py::test_the_path_check_needs_no_python_and_keeps_closed_folders",
]
# Fails `docker build` only; every other docker command reaches the real CLI.
SHIM = """#!/bin/sh
if [ "$1" = build ]; then echo "PLANTED fixture build failure" >&2; exit 1; fi
exec "__DOCKER__" "$@"
"""


@pytest.mark.parametrize(
    "node", BUILDERS, ids=[n.split("::")[0][len("tests/test_") : -3] for n in BUILDERS]
)
def test_a_planted_build_failure_errors_every_fixture_that_builds(tmp_path, node):
    docker = shutil.which("docker")
    if docker is None or not _docker_ready():
        pytest.skip("docker daemon not running")
    if "c8_r15" in node and shutil.which("uv") is None:
        pytest.skip("its fixture needs uv on the test host")
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "docker").write_text(SHIM.replace("__DOCKER__", docker))
    (shim / "docker").chmod(0o755)
    junit = tmp_path / "junit.xml"
    env = {**os.environ, "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}"}
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            node,
            "-n",
            "0",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--basetemp={tmp_path / 'bt'}",
            f"--junitxml={junit}",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    cases = list(ET.parse(junit).getroot().iter("testcase")) if junit.exists() else []
    assert cases, done.stdout[-2000:]
    for case in cases:
        found = {child.tag: child for child in case}
        assert "skipped" not in found, (
            case.get("name"),
            found["skipped"].get("message"),
        )
        outcome = found.get("error", found.get("failure"))
        assert outcome is not None, case.get("name")
        assert "PLANTED fixture" in (outcome.get("message") or "") + (
            outcome.text or ""
        )


def _rmi(tag: str) -> None:
    subprocess.run(["docker", "rmi", "--force", tag], capture_output=True, timeout=120)
