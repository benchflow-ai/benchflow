"""Releases ship both pinned bfx binaries (TRUSTED-RUNTIME.md §6).

The publishing workflows build bfx with the pin required before
``uv build``, and check every distribution against the pins before
publishing; ``tools/check_dist_bfx.py`` accepts only a distribution whose
two binaries are regular executable files with the pinned SHA-256.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from benchflow._bfx import pins
from tools import check_dist_bfx

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
PUBLISHING = ("public-release.yml", "internal-preview-release.yml")

PINS = {
    "x86_64": hashlib.sha256(b"x86").hexdigest(),
    "aarch64": hashlib.sha256(b"arm").hexdigest(),
}
GOOD = {"bfx-x86_64": b"x86", "bfx-aarch64": b"arm"}


def _steps(name: str) -> list[dict]:
    loaded = yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)
    return loaded["jobs"]["publish"]["steps"]


def _index(steps: list[dict], name: str) -> int:
    return next(i for i, step in enumerate(steps) if step.get("name") == name)


@pytest.mark.parametrize("workflow", PUBLISHING)
def test_publishing_builds_pinned_bfx_before_the_distributions(workflow):
    steps = _steps(workflow)
    build = _index(steps, "Build bfx (pinned)")
    assert steps[build]["run"] == "BFX_REQUIRE_PIN=1 scripts/build_bfx.sh"
    assert _index(steps, "Install the bfx toolchain") < build
    assert build < _index(steps, "Build distributions")


@pytest.mark.parametrize("workflow", PUBLISHING)
def test_publishing_checks_bfx_in_every_distribution_before_publishing(workflow):
    steps = _steps(workflow)
    check = _index(steps, "Check bfx in distributions")
    assert "tools/check_dist_bfx.py dist" in steps[check]["run"]
    assert _index(steps, "Build distributions") < check
    publish = next(
        i for i, step in enumerate(steps) if step.get("run", "") == "uv publish"
    )
    assert check < publish
    # The preview publishes only some runs; its bfx steps run on the same ones.
    gate = steps[publish].get("if")
    for name in ("Install the bfx toolchain", "Build bfx (pinned)"):
        assert steps[_index(steps, name)].get("if") == gate
    assert steps[check].get("if") == gate


def test_the_tool_reads_the_package_pins():
    assert check_dist_bfx.read_pins() == pins.SHA256


def test_the_build_script_refuses_to_install_an_unpinned_binary():
    script = (WORKFLOWS.parent.parent / "scripts" / "build_bfx.sh").read_text()
    require = script.index('"${BFX_REQUIRE_PIN:-0}" = 1')
    assert require < script.index('install -m 0555 "$built"')
    assert "exit 1" in script[require : script.index("fi", require)]


def _wheel(path: Path, files: dict[str, bytes], mode: int = 0o100555) -> Path:
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("benchflow/__init__.py", "")
        for name, data in files.items():
            info = zipfile.ZipInfo(f"benchflow/_bfx/bin/{name}")
            info.external_attr = mode << 16
            wheel.writestr(info, data)
    return path


def _sdist(path: Path, files: dict[str, bytes], mode: int = 0o555) -> Path:
    with tarfile.open(path, "w:gz") as sdist:
        for name, data in files.items():
            info = tarfile.TarInfo(f"benchflow-0.8.3/src/benchflow/_bfx/bin/{name}")
            info.size = len(data)
            info.mode = mode
            sdist.addfile(info, io.BytesIO(data))
    return path


def test_distributions_with_both_pinned_binaries_pass(tmp_path):
    _wheel(tmp_path / "benchflow-0.8.3-py3-none-any.whl", GOOD)
    _sdist(tmp_path / "benchflow-0.8.3.tar.gz", GOOD)
    problems = check_dist_bfx.dist_problems(tmp_path, PINS)
    assert problems == {
        "benchflow-0.8.3-py3-none-any.whl": [],
        "benchflow-0.8.3.tar.gz": [],
    }


@pytest.mark.parametrize(
    "files,mode,expected",
    [
        ({"bfx-x86_64": b"x86"}, 0o100555, "bfx-aarch64 is missing"),
        ({**GOOD, "bfx-aarch64": b"other"}, 0o100555, "not the pinned"),
        (GOOD, 0o100444, "not a regular executable file"),
        (GOOD, 0o120777, "not a regular executable file"),
    ],
)
def test_a_wheel_without_both_pinned_executables_fails(tmp_path, files, mode, expected):
    _wheel(tmp_path / "benchflow-0.8.3-py3-none-any.whl", files, mode)
    [problems] = check_dist_bfx.dist_problems(tmp_path, PINS).values()
    assert any(expected in p for p in problems), problems


def test_an_sdist_without_the_binaries_fails(tmp_path):
    _sdist(tmp_path / "benchflow-0.8.3.tar.gz", {"bfx-x86_64": b"x86"})
    [problems] = check_dist_bfx.dist_problems(tmp_path, PINS).values()
    assert problems == ["bfx-aarch64 is missing"]


def test_an_empty_dist_folder_fails(tmp_path):
    assert check_dist_bfx.main([str(tmp_path)]) == 1
