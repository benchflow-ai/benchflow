"""RCI's wrapped-binary shape against BenchFlow's post-agent root steps.

RCI's run on #234 (86fd158d6): a root oracle's solve.sh moved each of the
image's ``cat``, ``find``, ``rm``, ``chown``, ``chmod``, ``stat``, ``tar``,
``ls``, ``mkdir``, ``cp``, ``mv``, ``readlink`` and ``sha256sum`` aside and put
a wrapper in its place; every root step BenchFlow ran in main afterwards ran
the wrappers with NET_ADMIN. Here a real container holds NET_ADMIN (main's
worst case before the helper served), the wrappers log each call's CapEff
and NoNewPrivs, and BenchFlow's own step commands run through the real
``docker exec`` argv builder:

- a root step that still uses the image's tools runs them bounded: no
  NET_ADMIN, NET_RAW or SYS_PTRACE, no_new_privs set;
- the steps moved to bfx (the receipt folders, test.sh's chmod, the test
  upload's folder, the runtime-log publish, the proxy stop) run none of them.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import uuid
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from benchflow._bfx.locate import SANDBOX_PATH, BfxUnavailable, verified_binary
from benchflow.rollout._setup import (
    _publish_modes_command,
    _publish_rm_command,
    _publish_stage_command,
)
from benchflow.sandbox._base import ROOT_EXEC_PATH, ExecRoute
from benchflow.sandbox._bounded_exec import BFX_LAUNCH, bfx_sh
from benchflow.sandbox.docker import route_exec_argv
from benchflow.sandbox.egress_denylist import (
    _PID_PATH,
    RUNTIME_DIR,
    _runtime_dir_cmd,
    _stop_cmd,
)
from benchflow.task.verifier_core import _receipt_command

WRAPPED = [
    "cat",
    "find",
    "rm",
    "chown",
    "chmod",
    "stat",
    "tar",
    "ls",
    "mkdir",
    "cp",
    "mv",
    "readlink",
    "sha256sum",
]
HELD = (1 << 12) | (1 << 13) | (1 << 19)
LOG = "/var/tmp/rci-wrapper.log"
# The root route of a sandbox without the kit (a root agent: no sandbox user).
ROUTE = ExecRoute(pin=ROOT_EXEC_PATH, task_root=False, explicit_root=True)
WITH_BFX = SimpleNamespace(bfx_delivery="bind")

_WRAPPER = (
    "#!/bin/sh\n"
    "e=; n=; while read -r k v r; do case $k in CapEff:) e=$v;; "
    "NoNewPrivs:) n=$v;; esac; done < /proc/self/status\n"
    f'printf \'%s %s %s\\n\' "${{0##*/}}" "$e" "$n" >> {LOG}\n'
    'exec /usr/local/lib/rci-real/"${0##*/}" "$@"\n'
)


def _docker() -> bool:
    return shutil.which("docker") is not None and (
        subprocess.run(["docker", "info"], capture_output=True).returncode == 0
    )


@pytest.fixture(scope="module")
def main():
    if not _docker():
        pytest.skip("Docker required")
    try:
        bfx = verified_binary(platform.machine())
    except BfxUnavailable as exc:
        pytest.skip(str(exc))
    name = f"bf-rci-{uuid.uuid4().hex[:10]}"
    started = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--cap-add",
            "NET_ADMIN",
            "--mount",
            f"type=bind,src={bfx},dst={SANDBOX_PATH},readonly",
            "ubuntu:24.04",
            "sleep",
            "infinity",
        ],
        capture_output=True,
        text=True,
    )
    assert started.returncode == 0, started.stderr
    try:
        # The root agent's turn: every wrapped tool moved aside and replaced.
        plant = "mkdir -p /usr/local/lib/rci-real; " + "".join(
            f"cp /usr/bin/{n} /usr/local/lib/rci-real/{n}; " for n in WRAPPED
        )
        for n in WRAPPED:
            plant += (
                f"printf '%s' {_quote(_WRAPPER)} > /usr/local/lib/rci-real/.w; "
                f"/usr/local/lib/rci-real/cp /usr/local/lib/rci-real/.w /usr/bin/{n}; "
                f"/usr/local/lib/rci-real/chmod 755 /usr/bin/{n}; "
            )
        planted = _sh(name, plant)
        assert planted.returncode == 0, planted.stderr
        _sh(name, f": > {LOG}")
        yield name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def _sh(container: str, command: str) -> subprocess.CompletedProcess:
    """A setup command with builtins and the moved-aside tools only."""
    return subprocess.run(
        ["docker", "exec", "-u", "0", container, "/bin/sh", "-c", command],
        capture_output=True,
        text=True,
    )


def _calls(container: str) -> list[tuple[str, int, str]]:
    out = _sh(container, f'while IFS= read -r l; do echo "$l"; done < {LOG}').stdout
    calls = []
    for line in out.splitlines():
        tool, eff, nnp = [*line.split(), "", "", ""][:3]
        calls.append((tool, int(eff or "0", 16), nnp))
    return calls


def _step(container: str, command: str, *, bounded: bool = True):
    """BenchFlow's root step as its exec layer sends it into main."""
    argv = route_exec_argv(
        ROUTE,
        command,
        container=container,
        user="0",
        launch=BFX_LAUNCH if bounded else None,
    )
    before = len(_calls(container))
    ran = subprocess.run(["docker", *argv], capture_output=True, text=True)
    return ran, _calls(container)[before:]


def test_without_the_bound_the_planted_tools_run_with_net_admin(main):
    ran, calls = _step(
        main, "mkdir -p /tmp/bf-x && chmod 700 /tmp/bf-x && ls /tmp", bounded=False
    )
    assert ran.returncode == 0, ran.stderr
    assert {c[0] for c in calls} >= {"mkdir", "chmod", "ls"}
    assert all(c[1] & (1 << 12) for c in calls), calls


def test_a_root_step_on_the_image_tools_runs_them_bounded(main):
    ran, calls = _step(
        main,
        "mkdir -p /tmp/bf-y && chmod 700 /tmp/bf-y && ls /tmp >/dev/null && "
        "cat /etc/hostname >/dev/null && rm -rf /tmp/bf-y",
    )
    assert ran.returncode == 0, ran.stderr
    assert {c[0] for c in calls} >= {"mkdir", "chmod", "ls", "cat", "rm"}
    for tool, eff, nnp in calls:
        assert eff & HELD == 0, (tool, hex(eff))
        assert nnp == "1", tool


def test_the_receipt_folders_run_no_image_tool(main):
    shared = _receipt_command(WITH_BFX, PurePosixPath("/run/runtime"), False, "main")
    ran, calls = _step(main, shared)
    assert ran.returncode == 0, ran.stderr + ran.stdout
    assert calls == []
    folder = PurePosixPath("/run/runtime/verifier-abc")
    ran, calls = _step(main, _receipt_command(WITH_BFX, folder, True, "main"))
    assert ran.returncode == 0, ran.stderr + ran.stdout
    assert calls == []
    modes = _sh(main, "/usr/local/lib/rci-real/stat -c %a /run/runtime " + str(folder))
    assert modes.stdout.split() == ["1777", "755"]


def test_the_test_upload_folder_and_chmod_run_no_image_tool(main):
    ran, calls = _step(main, bfx_sh("mkdir", "/tests", parents=True, mode=0o755))
    assert ran.returncode == 0, ran.stderr
    assert _sh(main, "printf '#!/bin/sh\\nexit 0\\n' > /tests/test.sh").returncode == 0
    ran2, calls2 = _step(main, bfx_sh("chmod", "/tests/test.sh", set=0o111))
    assert ran2.returncode == 0, ran2.stderr
    assert calls == calls2 == []
    assert _sh(main, "[ -x /tests/test.sh ]").returncode == 0


def test_the_runtime_log_publish_runs_no_image_tool(main):
    stage = f"/logs/.runtime-publish-{uuid.uuid4().hex}"
    ran, calls = _step(main, _publish_stage_command(WITH_BFX, stage, ""))
    assert ran.returncode == 0, ran.stderr
    assert (
        _sh(main, f"printf x > {stage}/a.txt; printf y > {stage}/b.txt").returncode == 0
    )
    ran2, calls2 = _step(
        main, _publish_modes_command(WITH_BFX, stage, {"a.txt": "", "b.txt": ""})
    )
    assert ran2.returncode == 0, ran2.stderr
    modes = _sh(main, f"/usr/local/lib/rci-real/stat -c %a {stage}/a.txt {stage}/b.txt")
    assert modes.stdout.split() == ["644", "644"]
    ran3, calls3 = _step(main, _publish_rm_command(WITH_BFX, stage))
    assert ran3.returncode == 0, ran3.stderr
    assert calls == calls2 == calls3 == []
    assert _sh(main, f"[ ! -e {stage} ] && [ -d /logs/agent ]").returncode == 0


def test_the_proxy_stop_runs_no_image_tool(main):
    started = _sh(
        main,
        f"/usr/local/lib/rci-real/mkdir -p {RUNTIME_DIR}; "
        f"sleep 1000 >/dev/null 2>&1 & echo $! > {_PID_PATH}",
    )
    assert started.returncode == 0, started.stderr
    pid = _sh(main, f"IFS= read -r p < {_PID_PATH}; echo $p").stdout.strip()
    ran, calls = _step(main, _stop_cmd(WITH_BFX))
    assert ran.returncode == 0, ran.stderr
    assert calls == []
    gone = _sh(
        main,
        f"for i in 1 2 3 4 5 6 7 8 9 10; do [ -d /proc/{pid} ] || break; "
        "sleep 0.2; done; "
        f"s=; [ -r /proc/{pid}/status ] && while read -r k v r; do "
        '[ "$k" = State: ] && s=$v; done < /proc/' + pid + "/status; "
        f'[ ! -d /proc/{pid} ] || [ "$s" = Z ]',
    )
    assert gone.returncode == 0
    assert _sh(main, f"[ ! -e {RUNTIME_DIR} ]").returncode == 0


def test_the_egress_runtime_folder_runs_no_image_tool(main):
    ran, calls = _step(main, _runtime_dir_cmd(WITH_BFX, RUNTIME_DIR + "/certs"))
    assert ran.returncode == 0, ran.stderr
    assert calls == []
    modes = _sh(main, f"/usr/local/lib/rci-real/stat -c %a {RUNTIME_DIR}")
    assert modes.stdout.split() == ["700"]
    _sh(main, f"/usr/local/lib/rci-real/rm -rf {RUNTIME_DIR}")
