"""The agent can neither forge nor plant the test command's receipt files ((c)2), on real Docker.

Deterministic tier: whole oracle trials, the sandbox user ``agent``. solve.sh
leaves a process of the agent's uid running (in integrity off and audit the
agent's processes live through grading). For every attempt's receipt folder
it sees it forges the four files root reads back: the start receipt
("started"), ``.abandoned`` (test.sh then runs no tests), ``.exit`` ("0":
tests still running read as ended) and the launch record (a bad line refuses
the run, unscored). It tries each in the attempt's folder (a file renamed
over the name) and under the same name in the 1777 folder (created first).
test.sh writes a passing reward, then still runs at its deadline.

Before its reward provenance the four were in the 1777 folder in every
trial that keeps no service: the planted ``.exit`` scored verifier/'s early
reward 1.0. Each attempt's files are now in a folder only root can write, in
every mode: the forger is refused there, what it wrote in the 1777 folder is
never read, and the timeout scores 0.
"""

from __future__ import annotations

import json

import pytest

from benchflow.task.verifier_core import _EXECUTION_RECEIPT_DIR
from tests.exec_diet_fakes import docker_available
from tests.test_acc_run2_docker import BASE, _run, _task

pytestmark = [pytest.mark.deterministic, pytest.mark.verifier_runtime]

RUN = str(_EXECUTION_RECEIPT_DIR)
# name suffix -> what the forger writes there.
FORGED = {"": "started", ".abandoned": "", ".exit": "0", ".launch": "bad\\n"}
FORGE_PY = f"""import errno, json, os, pwd, re, time
agent = pwd.getpwnam("agent")
os.setgroups([])
os.setgid(agent.pw_gid)
os.setuid(agent.pw_uid)
RUN = {RUN!r}
FORGED = {FORGED!r}
why = lambda exc: errno.errorcode.get(exc.errno, str(exc.errno))
tried, listed = {{}}, {{}}
end = time.time() + 240
n = 0
while time.time() < end:
    try:
        names = os.listdir(RUN)
    except OSError:
        names = []
    for folder in names:
        found = re.fullmatch(r"verifier-([0-9a-f]{{32}})", folder)
        if not found:
            continue
        attempt = found.group(1)
        inside = os.path.join(RUN, folder)
        try:
            listed[folder] = sorted(os.listdir(inside))
        except OSError as exc:
            listed[folder] = why(exc)
        forged = {{f"verifier-started-{{attempt}}" + s: body for s, body in FORGED.items()}}
        forged[f"verifier-launch-{{attempt}}"] = "bad\\n"
        for name, body in forged.items():
            # In the attempt's folder: a file of the agent's renamed over it.
            path = os.path.join(inside, name)
            if path not in tried:
                n += 1
                stage = os.path.join(RUN, f".forge-{{n}}")
                try:
                    with open(stage, "w") as f:
                        f.write(body)
                    os.rename(stage, path)
                    tried[path] = "ok"
                except OSError as exc:
                    tried[path] = why(exc)
                    try:
                        os.unlink(stage)
                    except OSError:
                        pass
            # In the 1777 folder: the same name, created first.
            path = os.path.join(RUN, name)
            if path not in tried:
                try:
                    with open(path, "x") as f:
                        f.write(body)
                    tried[path] = "ok"
                except OSError as exc:
                    tried[path] = why(exc)
    with open("/tmp/forge.tmp", "w") as f:
        json.dump({{"uid": os.getuid(), "tried": tried, "listed": listed}}, f)
    os.chmod("/tmp/forge.tmp", 0o644)
    os.replace("/tmp/forge.tmp", "/tmp/forge.json")
    time.sleep(0.02)
"""
SOLVE = (
    "cat > /opt/forge.py <<'PY'\n" + FORGE_PY + "PY\n"
    "setsid nohup python3 /opt/forge.py </dev/null >/dev/null 2>&1 &\n"
    "sleep 1\n"
)
# A passing reward first, the forger's look at the receipts, then a stall
# past the 20 s budget.
TEST_SH = """#!/bin/bash
echo 1 > /logs/verifier/reward.txt
sleep 8
cp /tmp/forge.json /logs/verifier/forge.json 2>/dev/null
sleep 300
"""


@pytest.fixture(autouse=True)
def _needs_docker():
    if not docker_available():
        pytest.skip("needs a Docker daemon")


@pytest.mark.parametrize("mode", ["audit", "off", "strict"])
async def test_a_forged_receipt_file_never_scores_stalled_tests(tmp_path, mode):
    task = _task(tmp_path, f"forge-{mode}", dockerfile=BASE, solve=SOLVE, test=TEST_SH)
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace(
            "[verifier]\ntimeout_sec = 300.0", "[verifier]\ntimeout_sec = 20.0"
        )
    )
    result = await _run(task, "oracle", integrity=mode)
    record = json.loads((result.rollout_dir / "result.json").read_text())
    forge = result.rollout_dir / "verifier" / "forge.json"
    seen = json.loads(forge.read_text()) if forge.is_file() else None
    assert result.verifier_error is None, (result.verifier_error, seen)
    assert result.rewards == {"reward": 0.0}, (record.get("warnings"), seen)
    info = record.get("verifier_timeout_info")
    assert info and info["scored"] is True, info
    if mode == "strict":
        return  # the agent's processes end before grading
    # The forger, of the agent's uid, saw the attempt's folder and the
    # receipt in it, and was refused each of the four names there.
    assert seen is not None and seen["uid"] != 0, seen
    (folder,) = seen["listed"]
    attempt = folder.removeprefix("verifier-")
    assert f"verifier-started-{attempt}" in seen["listed"][folder], seen
    inside = {p: r for p, r in seen["tried"].items() if f"/{folder}/" in p}
    want = {f"verifier-started-{attempt}{s}" for s in FORGED}
    want.add(f"verifier-launch-{attempt}")
    assert {p.rsplit("/", 1)[-1] for p in inside} == want, seen
    assert set(inside.values()) == {"EACCES"}, seen
    # What it planted in the 1777 folder first stood, and was never read.
    above = {p: r for p, r in seen["tried"].items() if p not in inside}
    assert set(above.values()) == {"ok"}, seen


# test.sh passes and exits after the forger's look: its launch record is
# read (_check_launch_record), and a bad line planted there would refuse the
# run, unscored.
PASS_SH = """#!/bin/bash
sleep 8
cp /tmp/forge.json /logs/verifier/forge.json 2>/dev/null
echo 1 > /logs/verifier/reward.txt
"""


@pytest.mark.parametrize("network", ["public", "no-network"])
@pytest.mark.parametrize("mode", ["audit", "off"])
async def test_a_forged_launch_record_never_refuses_a_passing_run(
    tmp_path, mode, network
):
    """With a restricted verifier network the launch is always held and its
    record read (``_check_launch_record``): the root-owned record beside the
    receipt is there, the forger was refused it, and the run passes. A bad
    line planted where the record is read refuses it (the mutant with the
    record in the 1777 folder: "launch record is unreadable")."""
    task = _task(
        tmp_path, f"launch-{mode}-{network}", dockerfile=BASE, solve=SOLVE, test=PASS_SH
    )
    toml = task / "task.toml"
    toml.write_text(
        toml.read_text().replace(
            "[verifier]\ntimeout_sec = 300.0",
            f'[verifier]\ntimeout_sec = 300.0\nnetwork_mode = "{network}"',
        )
    )
    result = await _run(task, "oracle", integrity=mode)
    forge = result.rollout_dir / "verifier" / "forge.json"
    seen = json.loads(forge.read_text()) if forge.is_file() else None
    assert result.verifier_error is None, (result.verifier_error, seen)
    assert result.rewards == {"reward": 1.0}, seen
    assert seen is not None and seen["uid"] != 0, seen
    (folder,) = seen["listed"]
    attempt = folder.removeprefix("verifier-")
    if network == "no-network":
        # The held launch wrote its record beside the receipt.
        assert f"verifier-started-{attempt}.launch" in seen["listed"][folder], seen
    inside = {p: r for p, r in seen["tried"].items() if f"/{folder}/" in p}
    launch = [p for p in inside if p.endswith(".launch") or "/verifier-launch-" in p]
    assert launch and {inside[p] for p in launch} == {"EACCES"}, seen
