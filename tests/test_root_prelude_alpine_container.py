"""DET-ALPINE-SU, in real containers (VM Docker): root's tool custody walk
follows links on Alpine.

``_cloud_metadata_root_prelude`` read each link's target with
``/bin/readlink`` (b3eb0315). Alpine ships ``readlink`` only as
``/usr/bin/readlink``, so every link on the way to a tool failed: ``tool sh``
(``/bin/sh -> /bin/busybox``) returned 2 and the firewall audit's root step
exited 7, refusing every Alpine task whose agent the firewall holds. The
walk now reads targets from the ``/bin/ls`` it already trusts.

The controls keep the walk's refusals: a link in a folder others may write,
a link to a file someone else owns, and a crafted ``TIME_STYLE``. Skipped
without a Docker daemon.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from tests.test_c8_protected_runtime_container import _docker_ready

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
]

ALPINE = "alpine:3.21"
DEBIAN = "debian:12"

# Root's links for the controls, made as root before the walk runs.
_PREP = (
    "mkdir -p /opt/open /opt/ok /opt/theirs && chmod 777 /opt/open "
    "&& ln -s /bin/sh /opt/open/sh && ln -s ../../bin/sh /opt/ok/rel "
    "&& ln -s /bin/sh '/opt/ok/a -> b' && ln -s /opt/ok/rel /opt/ok/two "
    "&& printf '#!/bin/sh\\n' > /opt/theirs/x && chmod 755 /opt/theirs/x "
    "&& chown 1000 /opt/theirs/x && ln -s /opt/theirs/x /opt/ok/theirs"
)


def _walk(image: str, script: str, env: tuple[str, ...] = ()) -> str:
    from benchflow.sandbox.lockdown import _cloud_metadata_root_prelude

    if not _docker_ready():
        pytest.skip("docker daemon not running")
    name = "DET-ALPINE-SU-" + uuid.uuid4().hex[:8]
    flags = [f for value in env for f in ("-e", value)]
    run = subprocess.run(
        ["docker", "run", "-d", "--name", name, "--label", "target=DET-ALPINE-SU",
         "--network", "none", *flags, image, "sleep", "300"],
        capture_output=True,
        text=True,
    )  # fmt: skip
    if run.returncode != 0:
        pytest.skip(f"cannot start {image}: {run.stderr.strip()}")
    try:
        made = subprocess.run(
            ["docker", "exec", name, "sh", "-c", _PREP],
            capture_output=True,
            text=True,
        )
        assert made.returncode == 0, made.stderr
        done = subprocess.run(
            ["docker", "exec", name, "/bin/sh", "-c",
             _cloud_metadata_root_prelude() + script],
            capture_output=True,
            text=True,
            timeout=120,
        )  # fmt: skip
        assert done.returncode == 0, (done.stdout, done.stderr)
        return done.stdout
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


_PROBE = (
    'for t in sh od tr ls; do printf \'tool %s %s rc=\' "$t" "$(tool $t)"; '
    "tool $t >/dev/null; echo $?; done; "
    "for p in /opt/ok/rel /opt/ok/two '/opt/ok/a -> b' /opt/open/sh "
    '/opt/ok/theirs; do if walk "$p"; then echo "walk $p W=$W"; '
    'else echo "walk $p refused"; fi; done'
)


@pytest.mark.parametrize("image", [ALPINE, DEBIAN])
def test_root_tools_behind_links_pass_custody(image):
    """The tools the audit's root step binds first (``scan_sh``, ``scan_od``,
    ``scan_tr``) resolve through their links on Alpine as on Debian."""
    out = _walk(image, _PROBE)
    for tool in ("sh", "od", "tr", "ls"):
        assert any(
            line.startswith(f"tool {tool} /") and line.endswith(" rc=0")
            for line in out.splitlines()
        ), out
    # A relative link, a two-hop chain and a name holding " -> " end at the
    # same shell (BusyBox on Alpine, dash on Debian).
    shell = next(
        line.removeprefix("walk /opt/ok/rel W=")
        for line in out.splitlines()
        if line.startswith("walk /opt/ok/rel W=")
    )
    assert shell.startswith("/"), out
    assert f"walk /opt/ok/two W={shell}" in out, out
    assert f"walk /opt/ok/a -> b W={shell}" in out, out


@pytest.mark.parametrize("image", [ALPINE, DEBIAN])
def test_links_others_could_change_are_still_refused(image):
    out = _walk(image, _PROBE)
    assert "walk /opt/open/sh refused" in out, out
    assert "walk /opt/ok/theirs refused" in out, out


def test_a_crafted_time_style_cannot_redirect_a_link():
    """GNU ls prints TIME_STYLE's text before the name; the prelude unsets it,
    so a date that spells " <link> -> <elsewhere>" never becomes the target."""
    out = _walk(
        DEBIAN,
        _PROBE,
        env=("TIME_STYLE=+ /opt/ok/theirs -> /bin/sh /opt/ok/rel -> /opt/open/sh",),
    )
    assert "walk /opt/ok/theirs refused" in out, out
    assert "walk /opt/ok/rel W=/usr/bin/dash" in out, out
