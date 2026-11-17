"""The verifier's bounded launch on every image family (EN-609, 0.9.x).

test.sh runs the agent's code as root, so it starts without NET_ADMIN,
NET_RAW and SYS_PTRACE (benchflow.sandbox.verifier_network.verifier_launch).
On the 0.9.x line the launch is util-linux setpriv, else the kit Python, else
it refuses (exit 97) and names why: the pinned bfx of this line has no
``launch``, so a delivered bfx never changes the path. NETSCAN found one
shipped task with neither setpriv nor a Python (cyberthon fastify,
node:20-alpine), and Alpine's /bin/setpriv is BusyBox's, without
--bounding-set: that task gets the named exit 97 (a 0.9.3 leftover).

Each case runs the real launch command as root in a real image that holds
all three capabilities at the start, then reads the started shell's
/proc/self/status and the launch's record.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import uuid

import pytest

from benchflow._bfx.locate import SANDBOX_PATH, BfxUnavailable, verified_binary
from benchflow.sandbox.verifier_network import (
    VERIFIER_LAUNCH_REFUSED,
    launch_record_problem,
    verifier_launch,
)

HELD = (1 << 12) | (1 << 13) | (1 << 19)
# SETGID (6) and SETUID (7) stay on every path: apt drops to _apt with them,
# and without them apt-get fails and apt-using oracles score 0 (the lead's
# ruling). The rules match every process, so a uid or group change escapes
# nothing.
KEPT = (1 << 6) | (1 << 7)
PROBE = 'while read -r k v r; do case $k in CapBnd:|CapEff:|NoNewPrivs:) echo "$k $v";; esac; done < /proc/self/status'
RECORD = "/tmp/bf-launch-record"
# The image families: (image, Dockerfile lines over it or None).
ALPINE = "alpine:3.20"
BUSYBOX = "busybox:1.36"
NODE_ALPINE = "node:20-alpine"
DEBIAN = "debian:12-slim"
UBUNTU = "ubuntu:24.04"


def _docker() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.fixture(scope="module")
def bfx():
    if not _docker():
        pytest.skip("Docker required")
    try:
        return verified_binary(platform.machine())
    except BfxUnavailable as exc:
        pytest.skip(str(exc))


def _image(base: str, lines: str = "") -> str:
    if not lines:
        assert (
            subprocess.run(
                ["docker", "pull", "-q", base], capture_output=True
            ).returncode
            == 0
        )
        return base
    tag = f"bf-launch-{uuid.uuid4().hex[:10]}"
    made = subprocess.run(
        ["docker", "build", "-q", "-t", tag, "-"],
        input=f"FROM {base}\n{lines}\n",
        capture_output=True,
        text=True,
    )
    assert made.returncode == 0, made.stderr
    return tag


def _run(
    image: str, bfx, args: list[str], *, entrypoint: str | None = None
) -> subprocess.CompletedProcess:
    mount = ["-v", f"{bfx}:{SANDBOX_PATH}:ro"] if bfx is not None else []
    return subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "0:0",
            "--network",
            "none",
            # All three at the start, so the drop is what removes them.
            "--cap-add",
            "NET_ADMIN",
            "--cap-add",
            "NET_RAW",
            "--cap-add",
            "SYS_PTRACE",
            *mount,
            *(["--entrypoint", entrypoint] if entrypoint else []),
            image,
            *args,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )


def _launched(image: str, bfx, shell: str = "/bin/sh") -> tuple[dict[str, str], str]:
    command = (
        verifier_launch(PROBE, record=RECORD)
        + f"; echo RC=$?; cat {RECORD} 2>/dev/null"
    )
    done = _run(image, bfx, ["-c", command], entrypoint=shell)
    status = dict(re.findall(r"(?m)^(CapBnd:|CapEff:|NoNewPrivs:) (\S+)$", done.stdout))
    return status, done.stdout + done.stderr


def _assert_held(status: dict[str, str], out: str) -> None:
    assert status, out
    assert int(status["CapBnd:"], 16) & HELD == 0, out
    assert int(status["CapEff:"], 16) & HELD == 0, out
    assert int(status["CapBnd:"], 16) & KEPT == KEPT, out
    assert int(status["CapEff:"], 16) & KEPT == KEPT, out
    assert status["NoNewPrivs:"] == "1", out
    record = next(
        ln for ln in out.splitlines() if ln.startswith("BF verifier-launch uid=")
    )
    assert launch_record_problem(record) is None, record


@pytest.mark.parametrize("base", [DEBIAN, UBUNTU], ids=["debian", "ubuntu"])
def test_a_delivered_bfx_leaves_the_launch_on_setpriv(bfx, base):
    """This line's pinned bfx has no ``launch``: with it mounted read-only at
    /.bfx/bfx, as BenchFlow delivers it, test.sh still starts through the
    image's util-linux setpriv, held, and the launch never execs bfx."""
    status, out = _launched(_image(base), bfx)
    _assert_held(status, out)
    assert "RC=0" in out, out
    assert SANDBOX_PATH not in verifier_launch("true"), out


@pytest.mark.parametrize(
    "base", [ALPINE, BUSYBOX, NODE_ALPINE], ids=["alpine", "busybox", "node-alpine"]
)
def test_an_image_with_neither_setpriv_nor_python_is_refused_with_exit_97(bfx, base):
    """The named 0.9.3 leftover: no util-linux setpriv and no kit Python, so
    test.sh never runs unheld; the launch exits 97 and says why, bfx
    delivered or not."""
    status, out = _launched(_image(base), bfx)
    assert not status, out
    assert f"RC={VERIFIER_LAUNCH_REFUSED}" in out
    assert "test.sh needs util-linux setpriv" in out


@pytest.mark.parametrize("base", [DEBIAN, UBUNTU], ids=["debian", "ubuntu"])
def test_debian_and_ubuntu_hold_test_sh_through_util_linux_setpriv(bfx, base):
    status, out = _launched(_image(base), None)
    _assert_held(status, out)


def test_the_kit_python_holds_test_sh_on_musl(bfx):
    """NETSCAN: the kit-Python fallback on musl was unproven. An Alpine image
    with bash and python3 (the cyberthon bolo family), the image's python3
    standing in for the kit's at its path, no util-linux setpriv, no bfx."""
    image = _image(
        NODE_ALPINE,
        "RUN apk add --no-cache bash python3 && mkdir -p /opt/benchflow/rt/py "
        '&& ln -s "$(command -v python3)" /opt/benchflow/rt/py/python3',
    )
    status, out = _launched(image, None, shell="/bin/bash")
    assert "RC=0" in out, out
    _assert_held(status, out)


# apt drops to _apt (setgroups, setgid, setuid) for what it fetches, a local
# .deb's copy included, so the probe needs no network: a package built in the
# image with dpkg-deb.
APT_DEB = (
    "RUN mkdir -p /opt/bf-apt/bf-apt-probe/DEBIAN "
    "&& printf 'Package: bf-apt-probe\\nVersion: 1\\nArchitecture: all\\n"
    "Maintainer: BenchFlow\\nDescription: apt drop probe\\n' "
    "> /opt/bf-apt/bf-apt-probe/DEBIAN/control "
    "&& dpkg-deb --build /opt/bf-apt/bf-apt-probe /opt/bf-apt/bf-apt-probe.deb "
    "&& chmod 755 /opt/bf-apt && chmod 644 /opt/bf-apt/bf-apt-probe.deb"
)
APT_INSTALL = "apt-get install -y /opt/bf-apt/bf-apt-probe.deb"
# What apt prints when it cannot drop to _apt.
APT_DROP_FAILED = re.compile(
    r"setgroups|setegid|seteuid|unsandboxed|Operation not permitted"
)
# Each launch path, forced by what verifier_launch reads: util-linux setpriv
# in the system folders (bfx delivered or not: this line's bfx has no
# launch); else the kit Python (the image's python3 at the kit's path,
# setpriv moved away).
APT_PATHS = {
    "setpriv": (APT_DEB, False),
    "setpriv-bfx-delivered": (APT_DEB, True),
    "kit-python": (
        "RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends "
        "python3 && rm -rf /var/lib/apt/lists/*\n"
        + APT_DEB
        + " && mkdir -p /usr/local/lib/bf-ctl /opt/benchflow/rt/py "
        "&& mv /usr/bin/setpriv /usr/local/lib/bf-ctl/setpriv "
        "&& ln -s /usr/bin/python3 /opt/benchflow/rt/py/python3",
        False,
    ),
}


@pytest.mark.parametrize("path", list(APT_PATHS))
def test_apt_in_test_sh_drops_to_apt_on_every_launch_path(bfx, path):
    """test.sh's apt-get, as root through the launch: its drop to _apt works
    and the install succeeds, on each path (the lead's ruling: test.sh keeps
    SETUID and SETGID). The verifier's policy changes only its network rules,
    never the launch, so a local install covers every policy here; fx609's
    trial rows ran apt-get update and a network install under each."""
    lines, with_bfx = APT_PATHS[path]
    image = _image(UBUNTU, lines)
    command = (
        verifier_launch(
            f"{PROBE}; [ -x {SANDBOX_PATH} ] && echo BFX-MOUNTED; "
            "[ -e /usr/bin/setpriv ] && echo PATH-SETPRIV; "
            f"{APT_INSTALL} > /tmp/apt.log 2>&1; echo APT=$?; cat /tmp/apt.log",
            record=RECORD,
        )
        + f"; echo RC=$?; cat {RECORD} 2>/dev/null"
    )
    done = _run(
        image, bfx if with_bfx else None, ["-c", command], entrypoint="/bin/bash"
    )
    out = done.stdout + done.stderr
    status = dict(re.findall(r"(?m)^(CapBnd:|CapEff:|NoNewPrivs:) (\S+)$", done.stdout))
    _assert_held(status, out)
    assert ("BFX-MOUNTED" in out) is with_bfx, out
    assert ("PATH-SETPRIV" in out) is (path != "kit-python"), out
    assert "APT=0" in out, out
    assert not APT_DROP_FAILED.search(out), out
    assert "RC=0" in out, out


def test_without_setuid_and_setgid_apt_cannot_drop_to_apt(bfx):
    """The control: the same install with SETUID and SETGID out of the
    bounding set fails at apt's drop, so the test above would see a launch
    that took them."""
    done = _run(
        _image(UBUNTU, APT_DEB),
        None,
        [
            "-c",
            "setpriv --bounding-set=-setuid,-setgid --inh-caps=-all -- "
            f"{APT_INSTALL}; echo APT=$?",
        ],
        entrypoint="/bin/bash",
    )
    out = done.stdout + done.stderr
    assert "APT=0" not in out, out
    assert re.search(r"setgroups \d+ failed", out), out
