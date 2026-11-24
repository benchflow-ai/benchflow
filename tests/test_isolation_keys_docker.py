"""Task-file settings the census suspected of weakening isolation by
themselves, checked on a real Docker daemon (guard rows: each holds on 0.10.0
and must keep holding; the cloud metadata key's own rows are in
``test_metadata_optin_docker.py``).

- An ``allowlist`` entry naming a cloud metadata address (or a range holding
  one) gives the agent's user a direct route by its own rule, but main's rule
  for every user, added from BenchFlow's helper before the task's processes
  start, still refuses it: no phase gets an answer from the VM's metadata
  server, as root or not, with or without the proxy.
- A Compose ``security_opt`` naming the task's own seccomp profile never runs
  a container under that profile: the profile's path is not in the staged
  project, so the start fails with the reason (``unconfined`` itself is
  refused by the host-access gate).

The metadata rows ask GCE's server for the instance id only and run only
where the host gets that answer (a GCE VM, meant to have no service account).
"""

from __future__ import annotations

import json
import shutil
import uuid
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
]


@pytest.fixture(scope="module")
def probe_image(tmp_path_factory):
    yield from build_probe_image(tmp_path_factory)


METADATA_ENTRIES = {
    "address": "169.254.169.254",
    "range": "169.254.0.0/16",
    "everything": "0.0.0.0/0",
}


def _allowlist_task(root: Path, image: str, entry: str) -> Path:
    task = _task(root, image, policy="allowlist")
    toml = (task / "task.toml").read_text()
    toml = toml.replace(
        'allowed_hosts = ["example.com"]', f'allowed_hosts = ["example.com", "{entry}"]'
    )
    assert entry in toml
    (task / "task.toml").write_text(toml)
    return task


@pytest.mark.skipif(
    not _host_reaches_metadata(),
    reason="the host gets no answer from a GCE metadata server",
)
@pytest.mark.parametrize("runner", ["oracle", "script"])
@pytest.mark.parametrize("entry", sorted(METADATA_ENTRIES))
def test_an_allowlist_entry_for_the_metadata_server_reaches_nothing(
    tmp_path, probe_image, entry, runner
):
    """``allowed_hosts`` naming 169.254.169.254, 169.254.0.0/16 or
    0.0.0.0/0, no flag: neither the turn nor test.sh gets an answer from the
    metadata server (0.0.0.0/0 is refused at start: the policy can't hold
    every process)."""
    task = _allowlist_task(tmp_path, probe_image, METADATA_ENTRIES[entry])
    outcome, trial = _sdk_run(tmp_path, task, runner, sandbox_user="agent", flag=False)
    assert trial is not None, _refusal_text(outcome)
    rows = _probe_rows(trial)
    if entry == "everything":
        said = _refusal_text(outcome)
        assert not rows and "sandbox startup refused" in said, (rows, said)
        return
    phases = {r["phase"].split("-")[0] for r in rows}
    assert "verifier" in phases and phases & {"oracle", "agent"}, rows
    assert not _reached(rows), f"reached the metadata server: {_reached(rows)}"


ALLOW_ALL_SECCOMP = {"defaultAction": "SCMP_ACT_ALLOW"}
# keyctl(2) (x86_64 250): Docker's default seccomp profile refuses it with
# EPERM; under an allow-all profile the kernel answers it.
KEYCTL_PROBE = (
    'python3 -c "import ctypes; c = ctypes.CDLL(None, use_errno=True); '
    "r = c.syscall(250, 0, 0, 0, 0, 0); "
    "print('keyctl', r, ctypes.get_errno())\""
)


def _security_opt_task(root: Path, image: str, option: str) -> Path:
    task = root / f"secopt-{uuid.uuid4().hex[:6]}"
    for sub in ("environment", "tests", "solution"):
        (task / sub).mkdir(parents=True)
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 120.0\n\n'
        "[verifier]\ntimeout_sec = 120.0\n\n"
        f'[environment]\ndocker_image = "{image}"\n'
    )
    (task / "instruction.md").write_text("Nothing to do.\n")
    (task / "environment" / "allow-all.json").write_text(json.dumps(ALLOW_ALL_SECCOMP))
    (task / "environment" / "docker-compose.yaml").write_text(
        json.dumps({"services": {"main": {"image": image, "security_opt": [option]}}})
    )
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        f"{KEYCTL_PROBE} > /logs/verifier/keyctl.txt 2>&1\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    for path in (task / "solution" / "solve.sh", task / "tests" / "test.sh"):
        path.chmod(0o755)
    return task


@pytest.mark.parametrize("option", ["seccomp=allow-all.json", "seccomp:allow-all.json"])
def test_a_task_s_own_seccomp_profile_never_confines_a_container(
    tmp_path, probe_image, option
):
    """A Compose ``security_opt`` naming the task's own seccomp profile (one
    that allows every syscall): the start fails with the reason, and nothing
    ran under the task's profile."""
    task = _security_opt_task(tmp_path, probe_image, option)
    outcome, trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    said = _refusal_text(outcome)
    probe = trial / "verifier" / "keyctl.txt" if trial is not None else None
    ran = probe.read_text().strip() if probe is not None and probe.is_file() else ""
    assert not ran, f"a container ran under the task's profile ({ran!r})"
    assert "allow-all.json" in said, said


def _capture_task(root: Path, image: str, name: str, file_var: str | None) -> Path:
    task = root / f"capture-{uuid.uuid4().hex[:6]}"
    for sub in ("environment", "tests", "solution"):
        (task / sub).mkdir(parents=True)
    line = f'capture_dir_b64_env_file_var = "{file_var}"\n' if file_var else ""
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[agent]\ntimeout_sec = 120.0\n\n'
        "[verifier]\ntimeout_sec = 120.0\n\n"
        f'[environment]\ndocker_image = "{image}"\n\n'
        "[[environment.setup_commands]]\n"
        'command = "mkdir -p /tmp/cap && echo hi > /tmp/cap/x"\n'
        'capture_dir = "/tmp/cap"\n'
        f'capture_dir_b64_env = "{name}"\n' + line
    )
    (task / "instruction.md").write_text("Nothing to do.\n")
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\ntrue\n")
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n"
    )
    for path in (task / "solution" / "solve.sh", task / "tests" / "test.sh"):
        path.chmod(0o755)
    return task


@pytest.mark.parametrize("listed", ["none", "name", "name-and-file"])
def test_a_setup_capture_writes_benchflow_s_environment_only_when_listed(
    tmp_path, probe_image, monkeypatch, listed
):
    """A setup command's capture writes the variable the task names into
    BenchFlow's own environment, and upserts it into the dotenv file an
    operator variable names: only for names the operator lists with
    --allow-task-env. Unlisted: refused before setup, nothing written."""
    name = f"S1_CAPTURE_{uuid.uuid4().hex[:8].upper()}"
    file_var = f"S1_DOTENV_{uuid.uuid4().hex[:8].upper()}"
    dotenv = tmp_path / "operator.env"
    dotenv.write_text("KEEP=1\n")
    monkeypatch.setenv(file_var, str(dotenv))
    monkeypatch.setenv(name, "before")
    allowed = {"none": "", "name": name, "name-and-file": f"{name},{file_var}"}[listed]
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_ENV", allowed)
    task = _capture_task(tmp_path, probe_image, name, file_var)
    outcome, _trial = _sdk_run(
        tmp_path, task, "oracle", sandbox_user="agent", flag=False
    )
    import os

    written = os.environ.get(name)
    in_file = name in dotenv.read_text()
    said = _refusal_text(outcome)
    if listed == "name-and-file":
        assert written not in (None, "before") and written.startswith("H4sI"), said
        assert in_file, dotenv.read_text()
        return
    assert written == "before" and not in_file, (
        f"the capture wrote {name} ({'changed' if written != 'before' else 'kept'}), "
        f"dotenv {'written' if in_file else 'kept'}; outcome: {said!r}"
    )
    assert "--allow-task-env" in said and "capture" in said, said


EGRESS_TEST_SH = """#!/bin/bash
python3 - > /logs/verifier/egress.txt 2>&1 <<'PY'
import socket
s = socket.socket()
s.settimeout(5)
try:
    s.connect(("1.1.1.1", 443))
    print("egress connected")
except Exception as e:
    print("egress refused", type(e).__name__)
PY
echo 1 > /logs/verifier/reward.txt
"""


def test_an_operator_no_network_overlay_holds_test_sh(tmp_path, probe_image):
    """The operator's --config-override sets no-network; the task's own
    [verifier] network_mode = "public" no longer lets test.sh out, and
    result.json names what the overlay replaced."""
    import benchflow as bf

    task = _task(tmp_path, probe_image)
    toml = (
        (task / "task.toml")
        .read_text()
        .replace("[verifier]\n", '[verifier]\nnetwork_mode = "public"\n')
    )
    (task / "task.toml").write_text(toml)
    (task / "tests" / "test.sh").write_text(EGRESS_TEST_SH)
    result = bf.run_sync(
        bf.RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            jobs_dir=tmp_path / "jobs",
            config_override={"sandbox": {"network_mode": "no-network"}},
        )
    )
    trial = Path(result.rollout_dir)
    said = (trial / "verifier" / "egress.txt").read_text()
    assert "egress refused" in said and "egress connected" not in said, said
    recorded = json.loads((trial / "result.json").read_text()).get("operator_network")
    assert recorded and "verifier.network_mode = public" in recorded["replaced"], (
        recorded
    )


SIDECAR_ATTACH_PY = r"""
import errno, json, os, time
found = {}
deadline = time.time() + 90
while not found and time.time() < deadline:
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/status") as f:
                uid = next(l.split()[1] for l in f if l.startswith("Uid:"))
        except Exception:
            continue
        if uid == "0":
            continue  # root: main's and the sidecar's own
        try:
            fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)  # the attach check only; nothing is read
            os.close(fd)
            found[pid] = "attach"
        except OSError as e:
            found[pid] = errno.errorcode.get(e.errno, str(e.errno))
    time.sleep(0.5)
with open("/shared/side.json", "w") as f:
    json.dump(found, f)
time.sleep(3600)
"""


def test_a_sidecar_in_main_s_pid_namespace_without_ptrace_cannot_attach(
    tmp_path, probe_image, monkeypatch
):
    """Allowed with host access, a sidecar in main's pid namespace with no
    SYS_PTRACE gets EACCES on the agent's own process (the attach check
    only: nothing is read)."""
    monkeypatch.setenv("BENCHFLOW_ALLOW_TASK_HOST_ACCESS", "1")
    monkeypatch.setenv("BENCHFLOW_ALLOW_HOST_ACCESS_ON_CLOUD", "1")
    task = _task(tmp_path, probe_image)
    (task / "environment" / "side.py").write_text(SIDECAR_ATTACH_PY)
    (task / "environment" / "docker-compose.yaml").write_text(
        json.dumps(
            {
                "services": {
                    "main": {"volumes": ["shared:/shared"]},
                    "side": {
                        "image": probe_image,
                        "pid": "service:main",
                        "volumes": ["shared:/shared", "./side.py:/side.py:ro"],
                        "command": ["python3", "/side.py"],
                    },
                },
                "volumes": {"shared": {}},
            }
        )
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/bash\n"
        "for _ in $(seq 1 60); do [ -s /shared/side.json ] && break; sleep 1; done\n"
        "cp /shared/side.json /logs/verifier/side.json 2>/dev/null\n"
        "echo 1 > /logs/verifier/reward.txt\n"
    )
    script = tmp_path / "agent-waits.sh"
    script.write_text("sleep 30\n")
    import benchflow as bf

    result = bf.run_sync(
        bf.RolloutConfig(
            task_path=task,
            agent=f"script:{script}",
            environment="docker",
            sandbox_user="agent",
            jobs_dir=tmp_path / "jobs",
        )
    )
    trial = Path(result.rollout_dir)
    seen = trial / "verifier" / "side.json"
    assert seen.is_file(), (getattr(result, "error", None), list(trial.rglob("*"))[:20])
    found = json.loads(seen.read_text())
    assert found and set(found.values()) == {"EACCES"}, found
