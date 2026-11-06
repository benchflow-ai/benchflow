"""Layer build failures with real offline apt/dpkg and inherited image USER.

The fixture's firewall commands are harmless stubs: this checks package state
and fallback, not network enforcement. Set EN229_LAYER_TEST_TIMEOUT=180 to
repeat the review's production-budget interruption; ordinary tests use 5 s.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from dataclasses import replace

import pytest

from benchflow.sandbox import daytona_layers as layers
from benchflow.sandbox.lockdown import _daemon_guard_cmd

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="docker required"
)


def _docker(*args, check=True):
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=600
    )
    if check:
        assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture
def offline_image(tmp_path):
    if _docker("info", check=False).returncode:
        pytest.skip("Docker daemon unavailable")
    for image in ("python:3.12-slim", "debian:bookworm-slim"):
        found = _docker("image", "inspect", "--format", "{{.Id}}", image, check=False)
        if found.returncode == 0:
            base = found.stdout.strip()
            break
    else:
        pytest.skip("cached Debian-based image required; this test never pulls")
    tag = f"bf-en229-review-{uuid.uuid4().hex[:12]}"
    control = tmp_path / "pkg/DEBIAN"
    control.mkdir(parents=True)
    # dpkg requires public package directories, even under a private test umask.
    control.chmod(0o755)
    control.parent.chmod(0o755)
    (control / "control").write_text(
        "Package: iptables\nVersion: 999\nArchitecture: all\n"
        "Maintainer: Test <test@example.invalid>\nDescription: offline fixture\n"
    )
    (control / "control").chmod(0o644)
    (control / "preinst").write_text(
        "#!/bin/sh\nset -e\nif [ -f /build-phase ]; then\n"
        'test -n "$(ls -A /var/lib/dpkg/updates)"\n'
        "echo 'EN229 dpkg journal started'\nsleep 195\nfi\n"
    )
    (control / "preinst").chmod(0o755)
    tools = tmp_path / "pkg/usr/sbin"
    tools.mkdir(parents=True)
    tools.chmod(0o755)
    tools.parent.chmod(0o755)
    for name in ("iptables", "ip6tables"):
        (tools / name).write_text("#!/bin/sh\nexit 0\n")
        (tools / name).chmod(0o755)
    (tmp_path / "guard.sh").write_text(
        _daemon_guard_cmd("root", proc="/fake-proc", log="/tmp/guard.log")
    )
    (tmp_path / "Dockerfile").write_text(
        f"FROM {tag}:source\nCOPY pkg /pkg\nCOPY guard.sh /w/guard.sh\n"
        "RUN mkdir /repo && dpkg-deb --build /pkg /repo/iptables.deb && "
        "{ cat /pkg/DEBIAN/control; printf 'Filename: ./iptables.deb\\nSize: '; "
        "stat -c %s /repo/iptables.deb; printf 'SHA256: '; "
        "sha256sum /repo/iptables.deb | cut -d' ' -f1; } > /repo/Packages && "
        "rm -f /etc/apt/sources.list.d/* && "
        "printf 'deb [trusted=yes] file:/repo ./\\n' > /etc/apt/sources.list && "
        "apt-get update -q\n"
    )
    try:
        _docker("image", "tag", base, tag + ":source")
        _docker("build", "--network", "none", "-t", tag + ":base", str(tmp_path))
        yield tag, tmp_path
    finally:
        for suffix in ("candidate", "nonroot", "base", "source"):
            _docker("image", "rm", tag + ":" + suffix, check=False)


def _plain_guard(tag):
    result = _docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        "root",
        tag,
        "sh",
        "-c",
        'test -z "$(ls -A /var/lib/dpkg/updates)" && '
        "sh /w/guard.sh && cat /tmp/guard.log",
    )
    assert " status=0" in result.stdout, result.stdout
    assert "install iptables: apt-get install" in result.stdout
    assert "dpkg was interrupted" not in result.stdout + result.stderr
    return result


def test_interrupted_dpkg_layer_is_discarded_before_runtime(offline_image):
    """Review F1 against 1887d869's committed dpkg journal."""
    tag, context = offline_image
    layer = layers.plan_install_layers(agent=None, model=None, guard_daemon=True).guard
    assert layer is not None
    budget = int(os.environ.get("EN229_LAYER_TEST_TIMEOUT", "5"))
    layer = replace(layer, timeout_sec=budget)
    (context / "Dockerfile.layer").write_text(
        f"FROM {tag}:base\nRUN touch /build-phase\n"
        + layers.layer_run_line(layer, "en229-review")
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
    (context / "failed-build.log").write_text(result.stdout + result.stderr)
    assert "EN229 dpkg journal started" in result.stdout + result.stderr
    assert result.returncode != 0, "interrupted transaction was committed"
    assert _docker("image", "inspect", tag + ":candidate", check=False).returncode != 0
    for _ in range(2):
        _plain_guard(tag + ":base")
    print(
        f"EN229 F1: build failed after {budget}s; no candidate image; two clean runtime guards status=0"
    )


def test_inherited_nonroot_user_rejects_layer_and_keeps_plain_image(offline_image):
    """Review F2 against 1887d869's silent non-root install."""
    tag, context = offline_image
    (context / "Dockerfile.user").write_text(f"FROM {tag}:base\nUSER nobody\n")
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
    base = f"FROM {tag}:nonroot\n"
    plan = layers.plan_install_layers(agent=None, model=None, guard_daemon=True)
    recipe = layers.layered_recipe(plan, base)
    assert not any(line.startswith("USER ") for line in recipe.lines)
    (context / "Dockerfile.layer").write_text(base + "\n".join(recipe.lines) + "\n")
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
    assert result.returncode != 0, "non-root build was accepted"
    assert _docker("image", "inspect", tag + ":candidate", check=False).returncode != 0
    assert (
        _docker(
            "run", "--rm", "--network", "none", tag + ":nonroot", "id", "-u"
        ).stdout.strip()
        == "65534"
    )
    _plain_guard(tag + ":nonroot")
    print(
        "EN229 F2: inherited nobody build failed; plain USER retained; root runtime guard status=0"
    )
