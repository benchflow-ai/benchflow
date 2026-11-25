"""The metadata helper still builds on a fresh host after Alpine drops a pinned
build (real Docker, no build cache, the base pulled again).

Alpine's repository serves one build of each package, its newest, so a
recipe whose exact pins name a build it has replaced must build from the
newest build of each package's minor series. Each row reads what the image
holds: the tier its /etc/apk/world names (``apk add`` writes the world only
when it installs) and the builds ``apk info -v`` lists.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from benchflow.sandbox import _metadata_helper as helper_mod

# Builds Alpine 3.24 served and has since replaced: python3 3.14.10-r1 by
# 3.14.10-r2 (a rebuild, 23 November 2026), and openssl 3.5.8-r0 by 3.5.9-r0
# (a patch release, 30 September 2026).
DROPPED_BUILD = "python3=3.14.10-r1"
DROPPED_VERSION = "openssl=3.5.8-r0"


def _docker() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.fixture(scope="module", autouse=True)
def docker_host():
    if not _docker():
        pytest.skip("Docker required")


def _series(pin: str) -> str:
    name, version = pin.split("=", 1)
    return f"{name}~{'.'.join(version.split('.')[:2])}"


def _with(pin: str) -> tuple[tuple[str, ...], str]:
    """The exact pins with ``pin`` in place of its package's, and the
    helper's Dockerfile with that one pin changed."""
    name = pin.split("=", 1)[0]
    (current,) = [p for p in helper_mod.HELPER_PACKAGES if p.startswith(f"{name}=")]
    assert helper_mod.HELPER_DOCKERFILE.count(f" {current} ") == 1
    pins = tuple(pin if p == current else p for p in helper_mod.HELPER_PACKAGES)
    return pins, helper_mod.HELPER_DOCKERFILE.replace(f" {current} ", f" {pin} ")


def _resolves(*pins: str) -> bool:
    """Whether Alpine serves these builds now (apk's own answer, in the base)."""
    return (
        subprocess.run(
            [
                *("docker", "run", "--rm", "--pull", "missing"),
                helper_mod.HELPER_BASE,
                *("apk", "add", "--no-cache", "--simulate", *pins),
            ],
            capture_output=True,
            timeout=300,
        ).returncode
        == 0
    )


def _build(dockerfile: str) -> tuple[list[str], dict[str, str]]:
    """Build with no cache; the helper packages' world lines and every
    package's build, read from the image, which is then removed."""
    with tempfile.TemporaryDirectory() as tmp:
        iid = Path(tmp) / "iid"
        built = subprocess.run(
            ["docker", "build", "--no-cache", "--pull", "--iidfile", str(iid), "-"],
            input=dockerfile,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert built.returncode == 0, (built.stdout + built.stderr)[-3000:]
        image = iid.read_text().strip()
    try:
        read = subprocess.run(
            [
                *("docker", "run", "--rm", "--network", "none"),
                *("--entrypoint", "/bin/sh", image),
                *("-c", "cat /etc/apk/world; echo ---; apk info -v"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    finally:
        subprocess.run(["docker", "image", "rm", "--force", image], capture_output=True)
    world, info = read.split("---\n")
    names = {p.split("=", 1)[0] for p in helper_mod.HELPER_PACKAGES}
    ours = sorted(w for w in world.split() if w.split("=")[0].split("~")[0] in names)
    builds = {}
    for line in info.split():
        name, version, release = line.rsplit("-", 2)
        builds[name] = f"{version}-{release}"
    return ours, builds


def _in_series(builds: dict[str, str], pins: tuple[str, ...]) -> None:
    for pin in pins:
        name, version = pin.split("=", 1)
        assert builds[name].startswith(".".join(version.split(".")[:2]) + "."), (
            name,
            builds[name],
        )


def test_a_pin_alpine_rebuilt_builds_from_its_minor_series():
    """python3's pin names a build Alpine replaced by a rebuild of the same
    version: the exact pins fail whole and the next tier installs."""
    assert not _resolves(DROPPED_BUILD), "Alpine serves the dropped build again"
    pins, dockerfile = _with(DROPPED_BUILD)
    world, builds = _build(dockerfile)
    assert world == sorted(_series(p) for p in pins)
    assert builds["python3"] != DROPPED_BUILD.split("=")[1]
    _in_series(builds, pins)


def test_a_pin_alpine_moved_past_builds_from_its_minor_series():
    """openssl's pin names a version Alpine no longer serves (a patch release
    replaced it): the minor series still installs, at a later version."""
    assert not _resolves(DROPPED_VERSION), "Alpine serves the dropped version again"
    pins, dockerfile = _with(DROPPED_VERSION)
    world, builds = _build(dockerfile)
    assert world == sorted(_series(p) for p in pins)
    assert builds["openssl"].split("-r")[0] != "3.5.8"
    _in_series(builds, pins)


def test_the_exact_pins_install_whenever_alpine_serves_them():
    """The recipe as it is: while Alpine serves every pinned build (apk says
    so in the base), the image holds exactly those builds, from tier 1."""
    if not _resolves(*helper_mod.HELPER_PACKAGES):
        pytest.skip(
            "Alpine no longer serves every pinned build: fresh hosts build "
            "from the fallback tier; move HELPER_PACKAGES (the daily "
            "helper-fresh-build job says to which)"
        )
    world, builds = _build(helper_mod.HELPER_DOCKERFILE)
    assert world == sorted(helper_mod.HELPER_PACKAGES)
    for pin in helper_mod.HELPER_PACKAGES:
        name, version = pin.split("=", 1)
        assert builds[name] == version, (name, builds[name])
