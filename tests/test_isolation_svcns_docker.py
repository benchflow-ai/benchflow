"""A task's services sharing namespaces with main: what holds and what is
refused (from the reviewer's measurements on 0.10.0).

- c1: the sidecar joins main's network (``network_mode: service:main``);
- c2: main joins the sidecar's network (``network_mode: service:side``);
- c3b: main joins the sidecar's pid namespace, with ``SYS_PTRACE`` on main;
- c3e: main joins the sidecar's pid namespace, with ``SYS_PTRACE`` on the
  sidecar;
- c3f: main and the sidecar both join a third service's pid namespace, with
  ``SYS_PTRACE`` on the sidecar.

c1 and c2, under ``public`` and ``no-network``, with the oracle (root) and a
``script:`` agent (the agent user): no process (main's, test.sh's, the
sidecar's) gets an answer from the metadata server. Under ``no-network`` the
sidecar reaches no outside host unless the task names it in
``metadata.x-service-egress`` (and c1's joined sidecar not even then), and in
c2 main does not share the sidecar's network namespace (BenchFlow puts main on
``none``).

c3b, c3e and c3f share a pid namespace between main and another service,
whichever way the join goes: refused before anything starts without host
access, under every policy (``network_mode: service:`` sharing stays as it is).
The metadata asks are for the instance id only; the rows run only where the
host gets that answer (a GCE VM, meant to have no service account).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tests.test_metadata_optin_docker import (
    _docker_ready,
    _host_reaches_metadata,
    _probe_rows,
    _reached,
    _refusal_text,
    _sdk_run,
    _task,
    build_probe_image,
)

pytestmark = [
    pytest.mark.skipif(shutil.which("docker") is None, reason="docker required"),
    pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon answers"),
    pytest.mark.skipif(
        not _host_reaches_metadata(),
        reason="the host gets no answer from a GCE metadata server",
    ),
]


@pytest.fixture(scope="module")
def probe_image(tmp_path_factory):
    yield from build_probe_image(tmp_path_factory)


# The sidecar: the metadata probe, one outside connect, its network namespace,
# and (c3b) nothing else; rows go to the shared volume for test.sh to carry.
SIDE_PY = r"""
import os, socket, subprocess, time
out = open("/shared/side.txt", "a")
def outside():
    s = socket.socket(); s.settimeout(5)
    try:
        s.connect(("1.1.1.1", 80)); return "connected"
    except Exception as e:
        return "refused " + type(e).__name__
    finally:
        s.close()
for phase in ("side-start", "side-late"):
    probe = subprocess.run(["python3", "/opt/mdprobe/probe.py", phase], capture_output=True, text=True)
    out.write(probe.stdout)
    out.write(f"SIDE-OUTSIDE {phase} {outside()}\n")
    out.write(f"SIDE-NETNS {os.readlink('/proc/self/ns/net')}\n")
    out.flush()
    time.sleep(20)
time.sleep(3600)
"""

# test.sh: carry the sidecar's rows, main's own netns, and (c3b) main's attach
# check on the sidecar's processes, as root and as the agent user.
ATTACH_PY = r"""
import errno, os, sys
found = {}
me = os.getpid()
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        continue
    if b"/side.py" not in cmd:
        continue
    try:
        fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)  # the attach check only; nothing is read
        os.close(fd)
        found[pid] = "attach"
    except OSError as e:
        found[pid] = errno.errorcode.get(e.errno, str(e.errno))
print(f"ATTACH {sys.argv[1]} " + ",".join(sorted(set(found.values()))))
"""

TEST_SH = (
    "#!/bin/bash\n"
    "/opt/mdprobe/probe.sh verifier /logs/verifier/mdprobe-verifier.txt\n"
    "cp /tmp/mdprobe-agent.txt /logs/verifier/mdprobe-agent.txt 2>/dev/null\n"
    "for _ in $(seq 1 60); do grep -q side-late /shared/side.txt 2>/dev/null && break; sleep 1; done\n"
    "cp /shared/side.txt /logs/verifier/side.txt 2>/dev/null\n"
    'echo "MAIN-NETNS $(readlink /proc/self/ns/net)" > /logs/verifier/main.txt\n'
    "if [ -f /opt/attach.py ]; then\n"
    "  python3 /opt/attach.py root >> /logs/verifier/main.txt 2>&1\n"
    "  setpriv --reuid=$(id -u agent) --regid=$(id -g agent) --clear-groups "
    "python3 /opt/attach.py agent >> /logs/verifier/main.txt 2>&1\n"
    "fi\n"
    "echo 1 > /logs/verifier/reward.txt\n"
)

CASES = {
    "c0-plain": {"main": {}, "side": {}},
    "c1-side-joins-main-network": {
        "main": {},
        "side": {"network_mode": "service:main"},
    },
    "c2-main-joins-side-network": {
        "main": {"network_mode": "service:side"},
        "side": {},
    },
}

# Main shares a pid namespace with another service: refused without host
# access. (shown: the services the refusal names.)
SHARED_PID = {
    "c3b-main-joins-side-pid": {
        "main": {"pid": "service:side", "cap_add": ["SYS_PTRACE"]},
        "side": {},
        "shown": ["side"],
    },
    "c3e-main-joins-side-pid-ptrace-on-side": {
        "main": {"pid": "service:side"},
        "side": {"cap_add": ["SYS_PTRACE"]},
        "shown": ["side"],
    },
    "c3f-main-and-side-join-anchor-pid": {
        "main": {"pid": "service:anchor"},
        "side": {"pid": "service:anchor", "cap_add": ["SYS_PTRACE"]},
        "anchor": {},
        "shown": ["anchor", "side"],
    },
}


def _svc_task(root: Path, image: str, case: str, policy: str) -> Path:
    task = _task(root, image, policy=policy)
    env = task / "environment"
    (env / "side.py").write_text(SIDE_PY)
    (env / "attach.py").write_text(ATTACH_PY)
    spec = CASES.get(case) or SHARED_PID[case]
    services = {
        "main": {
            "volumes": ["shared:/shared", "./attach.py:/opt/attach.py:ro"],
            **spec["main"],
        },
        "side": {
            "image": image,
            "volumes": ["shared:/shared", "./side.py:/side.py:ro"],
            "command": ["python3", "/side.py"],
            **spec["side"],
        },
    }
    if "anchor" in spec:
        services["anchor"] = {
            "image": image,
            "command": ["sleep", "3600"],
            **spec["anchor"],
        }
    (env / "docker-compose.yaml").write_text(
        json.dumps({"services": services, "volumes": {"shared": {}}})
    )
    (task / "tests" / "test.sh").write_text(TEST_SH)
    (task / "tests" / "test.sh").chmod(0o755)
    return task


def _lines(trial: Path, name: str) -> list[str]:
    path = trial / "verifier" / name
    return path.read_text(errors="replace").splitlines() if path.is_file() else []


def _side_rows(trial: Path) -> list[dict]:
    return [
        json.loads(line.split(" ", 1)[1])
        for line in _lines(trial, "side.txt")
        if line.startswith("BF-MDPROBE ")
    ]


@pytest.mark.parametrize("runner", ["oracle", "script"])
@pytest.mark.parametrize("policy", ["public", "no-network"])
@pytest.mark.parametrize("case", sorted(set(CASES) - {"c0-plain"}))
def test_a_service_sharing_main_s_namespaces_holds_the_block_and_the_policy(
    tmp_path, probe_image, case, policy, runner
):
    task = _svc_task(tmp_path, probe_image, case, policy)
    outcome, trial = _sdk_run(tmp_path, task, runner, sandbox_user="agent", flag=False)
    assert trial is not None, _refusal_text(outcome)
    main_rows, side_rows = _probe_rows(trial), _side_rows(trial)
    assert main_rows and side_rows, (_refusal_text(outcome), main_rows, side_rows)
    assert not _reached(main_rows + side_rows), _reached(main_rows + side_rows)
    side = _lines(trial, "side.txt")
    main = _lines(trial, "main.txt")
    if policy == "no-network":
        outside = [line for line in side if line.startswith("SIDE-OUTSIDE")]
        assert outside and all("refused" in line for line in outside), outside
        if case.startswith("c2"):
            side_ns = {
                line.split()[1] for line in side if line.startswith("SIDE-NETNS")
            }
            main_ns = {
                line.split()[1] for line in main if line.startswith("MAIN-NETNS")
            }
            assert main_ns and side_ns and not main_ns & side_ns, (main_ns, side_ns)


@pytest.mark.parametrize("runner", ["oracle", "script"])
@pytest.mark.parametrize("policy", ["public", "no-network"])
@pytest.mark.parametrize("case", sorted(SHARED_PID))
def test_main_sharing_a_pid_namespace_with_a_service_is_refused_before_start(
    tmp_path, probe_image, case, policy, runner
):
    task = _svc_task(tmp_path, probe_image, case, policy)
    outcome, trial = _sdk_run(tmp_path, task, runner, sandbox_user="agent", flag=False)
    said = " ".join(_refusal_text(outcome).split())
    for other in SHARED_PID[case]["shown"]:
        assert f"shares main's pid namespace with service {other}" in said, said
    assert "BENCHFLOW_ALLOW_TASK_HOST_ACCESS" in said, said
    ran = trial is not None and (
        _probe_rows(trial) or _side_rows(trial) or _lines(trial, "main.txt")
    )
    assert not ran, (said, trial)


def test_a_joined_sidecar_named_in_service_egress_still_gets_no_egress(
    tmp_path, probe_image
):
    """c1 under no-network with the sidecar in ``x-service-egress``: the
    joined sidecar is main's namespace, so it still reaches nothing outside."""
    task = _svc_task(tmp_path, probe_image, "c1-side-joins-main-network", "no-network")
    toml = (task / "task.toml").read_text()
    (task / "task.toml").write_text(
        toml.replace(
            'version = "1.0"\n',
            'version = "1.0"\n\n[metadata]\n"x-service-egress" = ["side"]\n',
            1,
        )
    )
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    assert trial is not None, _refusal_text(outcome)
    outside = [
        line for line in _lines(trial, "side.txt") if line.startswith("SIDE-OUTSIDE")
    ]
    assert outside and all("refused" in line for line in outside), outside


@pytest.mark.parametrize("case", ["c0-plain", "c1-side-joins-main-network"])
def test_under_an_allowlist_a_sidecar_that_can_reach_outside_is_refused_at_start(
    tmp_path, probe_image, case
):
    task = _svc_task(tmp_path, probe_image, case, "allowlist")
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    said = _refusal_text(outcome)
    assert "can reach outside" in said and not (trial and _side_rows(trial)), said
