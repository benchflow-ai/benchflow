"""A trial that keeps services keeps its start receipt where they cannot write, on real Docker.

Deterministic tier (``tests/integration/README.md``): whole oracle trials,
``[agent] user = "agent"``, the host user not root. The kept service is
``receipt_probe.py``: while test.sh runs (as root), it looks at
``/run/runtime`` as the agent's uid, tries to create a file in, and to move,
each attempt's receipt folder, and to create the receipt's ``.exit`` there
first; and it writes "0" as the status of every receipt it finds under the
receipt's name in ``/run/runtime`` itself, as a service that stalls the tests
past their deadline would to have them scored. With ``fs.protected_regular``
2 a status created first there stood: root's own write of it later failed.

The trials run in audit, the default: nothing is killed or frozen
before grading, so the service simply runs on beside the verifier, as every
process of the agent's does, and no kill record is written; the receipt's
folder is root's alone all the same ((c)2).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests import test_keep_services_docker as kept

pytestmark = pytest.mark.deterministic
# Whole trials on real Docker: the exec channel and the side-by-side steps
# before the verifier stay on, as in a run (tests/conftest.py).
DOCKER_RUN_DEFAULTS = True

PROBE_PY = '''"""GET / answers "kept"; GET /probe looks at /run/runtime; ``python3 receipt_probe.py <port>``."""

import errno
import http.server
import json
import os
import sys

RUN = "/run/runtime"


def why(exc):
    return errno.errorcode.get(exc.errno, str(exc.errno))


def receipts(names):
    return [n for n in names if n.startswith("verifier-started-") and "." not in n]


def probe():
    out = {"uid": os.getuid(), "folders": [], "old_exit": [], "found": 0}
    names = sorted(os.listdir(RUN))
    found = receipts(names)
    for name in names:
        path = os.path.join(RUN, name)
        if name.startswith("verifier-started-") or not os.path.isdir(path):
            continue
        st = os.lstat(path)
        row = {"uid": st.st_uid, "mode": oct(st.st_mode & 0o7777)}
        row["w_ok"] = os.access(path, os.W_OK)
        inner = receipts(sorted(os.listdir(path)))
        found += inner
        row["exit"] = []
        for name in inner:
            try:
                with open(os.path.join(path, name + ".exit"), "x") as f:
                    f.write("0")
                row["exit"].append("ok")
            except OSError as exc:
                row["exit"].append(why(exc))
        try:
            os.close(os.open(os.path.join(path, "planted"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            row["create"] = "ok"
        except OSError as exc:
            row["create"] = why(exc)
        try:
            os.rename(path, path + "-moved")
            row["move"] = "ok"
        except OSError as exc:
            row["move"] = why(exc)
        out["folders"].append(row)
    st = os.lstat(RUN)
    out["parent"] = {"uid": st.st_uid, "mode": oct(st.st_mode & 0o7777)}
    out["found"] = len(found)
    for name in found:
        try:
            with open(os.path.join(RUN, name + ".exit"), "x") as f:
                f.write("0")
            out["old_exit"].append("ok")
        except OSError as exc:
            out["old_exit"].append(why(exc))
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = (json.dumps(probe()) if self.path == "/probe" else "kept") + "\\n"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *args):
        pass


http.server.ThreadingHTTPServer(("0.0.0.0", int(sys.argv[1])), Handler).serve_forever()
'''
SOLVE = (
    "#!/bin/bash\nnohup python3 /opt/svc/receipt_probe.py 5000 >/tmp/service.out 2>&1 "
    "</dev/null &\n" + kept.WAIT_FOR_SERVICE
)
# test.sh: the service's look first, then a passing reward, then __TAIL__
# (a stall keeps the tests running past their deadline; an exit status).
TEST_SH = """#!/bin/bash
python3 -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/probe', timeout=10).read().decode().strip())" > /logs/verifier/probe.json 2>&1
echo 1 > /logs/verifier/reward.txt
__TAIL__
"""
BUDGET = "[verifier]\ntimeout_sec = 120.0\n"


@pytest.fixture
def require_docker() -> None:
    if kept.SANDBOX != "docker":
        reason = (
            kept.SKIP_REASON
            if kept.SANDBOX is None
            else f"Docker-only scenario ({kept.SANDBOX})"
        )
        pytest.skip(reason)


def _task(
    root: Path,
    name: str,
    *,
    stall: bool = False,
    same_uid: bool = False,
    tail: str = "",
):
    task = kept._task(root / "tasks", name, SOLVE)
    (task / "environment" / "receipt_probe.py").write_text(PROBE_PY)
    dockerfile = task / "environment" / "Dockerfile"
    text = dockerfile.read_text() + "COPY receipt_probe.py /opt/svc/\n"
    if same_uid:
        text += f"RUN useradd -m -u {os.getuid()} -s /bin/bash agent\n"
    dockerfile.write_text(text)
    test = task / "tests" / "test.sh"
    test.write_text(TEST_SH.replace("__TAIL__", "sleep 300" if stall else tail))
    if stall:
        toml = task / "task.toml"
        assert BUDGET in toml.read_text()
        toml.write_text(
            toml.read_text().replace(BUDGET, "[verifier]\ntimeout_sec = 20.0\n")
        )
    return task


def _past_the_host_uid_refusal(monkeypatch) -> None:
    """A kept-service trial whose agent has the host user's uid is refused
    before it starts on a mounted sandbox (until that change); its
    receipt folder is checked past that refusal."""
    from benchflow.task import keep_services

    monkeypatch.setattr(
        keep_services, "refuse_host_uid", lambda *a, **k: None, raising=False
    )


async def _run(tmp_path: Path, name: str, task: Path) -> dict | None:
    """One oracle trial in audit, the default: its rewards. Audit kills
    nothing before grading, so it writes no kill record."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sdk import SDK

    assert DockerSandbox.exec_channel is True
    assert DockerSandbox.concurrent_steps is True
    jobs = tmp_path / "jobs" / name
    result = await SDK().run(
        task_path=task, agent="oracle", jobs_dir=jobs, environment="docker"
    )
    assert not list(jobs.glob("**/kill-before-grading.json"))
    return result.rewards


def _rollout(tmp_path: Path, name: str) -> tuple[dict, dict]:
    (path,) = (tmp_path / "jobs" / name).glob("**/result.json")
    probe = json.loads((path.parent / "verifier" / "probe.json").read_text())
    return json.loads(path.read_text()), probe


def _root_only(probe: dict) -> None:
    """One receipt folder, root's and 0755, in a root-owned sticky parent:
    the agent's uid can neither write in it nor move it, and the status it
    wrote under the receipt's name in /run/runtime is beside it, unread."""
    assert probe["parent"] == {"uid": 0, "mode": "0o1777"}, probe
    assert len(probe["folders"]) == 1, probe
    (folder,) = probe["folders"]
    assert folder == {
        "uid": 0,
        "mode": "0o755",
        "w_ok": False,
        "create": "EACCES",
        "exit": ["EACCES"],
        "move": "EPERM",
    }, probe
    assert probe["found"] == 1 and probe["old_exit"] == ["ok"], probe


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("same_uid", [False, True], ids=["other-uid", "same-uid"])
async def test_the_receipt_folder_is_root_s_alone(tmp_path, monkeypatch, same_uid):
    if same_uid:
        if not 1000 <= os.getuid() < 60000:
            pytest.skip(f"the image's agent user cannot take host uid {os.getuid()}")
        _past_the_host_uid_refusal(monkeypatch)
    task = _task(tmp_path, "receipt", same_uid=same_uid)
    rewards = await _run(tmp_path, "receipt", task)
    assert rewards == {"reward": 1.0}
    result, probe = _rollout(tmp_path, "receipt")
    assert (probe["uid"] == os.getuid()) is same_uid, probe
    assert probe["uid"] != 0, probe
    _root_only(probe)
    assert result.get("verifier_timeout_info") is None, result


@pytest.mark.usefixtures("require_docker")
async def test_a_status_the_kept_service_wrote_does_not_score_stalled_tests(tmp_path):
    """The tests wrote a passing reward, then still ran at their deadline;
    the kept service wrote "0" as their status under the receipt's name in
    /run/runtime. That file is not read: the timeout scores 0."""
    task = _task(tmp_path, "stall", stall=True)
    rewards = await _run(tmp_path, "stall", task)
    result, probe = _rollout(tmp_path, "stall")
    assert rewards == {"reward": 0.0}, (result.get("warnings"), probe)
    _root_only(probe)
    info = result.get("verifier_timeout_info")
    assert info and info["scored"] is True and info["timeout_budget_sec"] == 20.0, info


@pytest.mark.usefixtures("require_docker")
@pytest.mark.parametrize("status", [0, 3])
async def test_tests_that_ended_behind_a_lost_exec_are_scored(
    tmp_path, monkeypatch, caplog, status
):
    """The test command's exec never answers, though its tests ended: the
    status the command recorded in the attempt's folder scores the trial
    from its outputs, as for any trial. It is the tests' own (3), not the
    "0" the kept service created first under the receipt's name."""
    import asyncio

    from benchflow._verifier_env import VERIFIER_ATTEMPT_ENV
    from benchflow.task import verifier_core

    real = verifier_core.exec_raw

    async def lose_the_test_command(run, *args, **kwargs):
        result = await real(run, *args, **kwargs)
        if f"export {VERIFIER_ATTEMPT_ENV}=" in str(kwargs.get("command", "")):
            await asyncio.Event().wait()  # its answer never comes back
        return result

    monkeypatch.setattr(verifier_core, "exec_raw", lose_the_test_command)
    task = _task(tmp_path, "lost", tail=f"exit {status}")
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace(BUDGET, "[verifier]\ntimeout_sec = 20.0\n")
    )
    caplog.set_level("WARNING")
    rewards = await _run(tmp_path, "lost", task)
    result, probe = _rollout(tmp_path, "lost")
    assert rewards == {"reward": 1.0}, (result.get("warnings"), probe)
    ended = f"ended (exit status {status}) but its exec did not return"
    assert ended in caplog.text, (caplog.text[-2000:], probe)
    _root_only(probe)
    assert result.get("verifier_timeout_info") is None, result
