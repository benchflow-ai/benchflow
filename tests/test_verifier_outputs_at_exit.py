"""test.sh's outputs as they were the moment it exited (CIR's L14).

In integrity off and audit the agent's processes live through grading, and a
mounted /logs/verifier stays writable from the sandbox until it is gone. The
verifier copies the folder into the trial (``verifier-at-exit/``) the moment
test.sh has ended and reads the rewards from that copy; the rollout puts it
back over ``verifier/`` once nothing in the sandbox can write there, and each
difference is a flag in ``integrity/verifier-inputs.json``. While test.sh
runs, only root (and the host user BenchFlow writes as) may write the folder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchflow.task import verifier_outputs as vo
from tests.exec_diet_fakes import docker_available

# -- the copy and the restore --------------------------------------------------


def _trial(tmp_path: Path, files: dict[str, str]) -> Path:
    trial = tmp_path / "trial"
    (trial / "verifier").mkdir(parents=True)
    for rel, text in files.items():
        path = trial / "verifier" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (trial / "integrity").mkdir()
    (trial / "integrity" / "verifier-inputs.json").write_text(
        json.dumps({"status": "checked", "flags": [], "sections": {}})
    )
    return trial


def test_the_copy_at_exit_goes_back_and_each_change_is_a_flag(tmp_path):
    trial = _trial(tmp_path, {"reward.txt": "0\n", "logs/out.txt": "ran\n"})
    taken = vo.take(trial)
    assert set(taken["files"]) == {"reward.txt", "logs/out.txt"}
    assert (trial / vo.AT_EXIT / "reward.txt").read_text() == "0\n"

    # A process left running rewrites the reward, removes a log and plants a
    # reward.json; BenchFlow writes its own record after the exit.
    (trial / "verifier" / "reward.txt").write_text("1\n")
    (trial / "verifier" / "logs" / "out.txt").unlink()
    (trial / "verifier" / "reward.json").write_text('{"reward": 1}')
    (trial / "verifier" / "verifier-network.json").write_text("{}")
    written = {"verifier-network.json": b"{}"}

    # The pass after the sandbox's stop: the first one only names a plant.
    flags = vo.put_back(trial, taken, written, remove=True)
    vo.record_flags(trial, flags)

    assert (trial / "verifier" / "reward.txt").read_text() == "0\n"
    assert (trial / "verifier" / "logs" / "out.txt").read_text() == "ran\n"
    assert not (trial / "verifier" / "reward.json").exists()
    assert (trial / vo.AFTER_EXIT / "reward.json").read_text() == '{"reward": 1}'
    assert (trial / "verifier" / "verifier-network.json").read_text() == "{}"
    assert {(f["path"], f["kind"]) for f in flags} == {
        ("/logs/verifier/reward.txt", "edited"),
        ("/logs/verifier/logs/out.txt", "removed"),
        ("/logs/verifier/reward.json", "planted"),
    }
    record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    assert record["status"] == "flagged"
    assert record["sections"]["verifier_output"]["flags"] == 3
    # Restoring again finds nothing and adds no flag twice.
    assert vo.put_back(trial, taken, written, remove=True) == []
    vo.record_flags(trial, flags)
    again = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    assert len(again["flags"]) == 3


def test_the_put_back_moves_nothing_into_a_sandbox():
    """put_back copies test.sh's outputs from the trial's verifier-at-exit/
    back over its verifier/ on the host, after the rewards were read from
    the copy: it calls no transfer primitive and admits nothing, so it is no
    ingress (benchflow.trust.ingress). A put-back that moved a file into a
    sandbox without an admission fails tests/test_trust_ingress_registry.py
    (rule 1), as one named ``restore`` that neither admits nor inherits
    lineage does (rule 3)."""
    import ast

    from benchflow.trust.ingress import SELF_ADMITTING, TRANSFER_PRIMITIVES

    tree = ast.parse(Path(vo.__file__).read_text())
    names = {
        node.attr if isinstance(node, ast.Attribute) else node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute | ast.Name)
    }
    assert not names & (TRANSFER_PRIMITIVES | {"admit", "restore_from"})
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert not defined & SELF_ADMITTING


def test_benchflows_own_later_record_is_kept_as_written(tmp_path):
    trial = _trial(tmp_path, {"reward.json": '{"reward": 0.5}'})
    taken = vo.take(trial)
    canonical = b'{\n  "reward": 0.5\n}'
    (trial / "verifier" / "reward.json").write_bytes(canonical)

    assert vo.put_back(trial, taken, {"reward.json": canonical}) == []
    assert (trial / "verifier" / "reward.json").read_bytes() == canonical


def test_integrity_off_records_no_flag(tmp_path):
    trial = _trial(tmp_path, {"reward.txt": "0\n"})
    (trial / "integrity" / "verifier-inputs.json").write_text(
        json.dumps({"status": "off", "flags": []})
    )
    taken = vo.take(trial)
    (trial / "verifier" / "reward.txt").write_text("1\n")

    flags = vo.put_back(trial, taken, {})
    vo.record_flags(trial, flags)

    assert (trial / "verifier" / "reward.txt").read_text() == "0\n"  # restored
    record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
    assert record == {"status": "off", "flags": []}


def test_the_rewards_are_read_from_the_copy(tmp_path):
    from unittest.mock import MagicMock

    from benchflow.task.paths import RolloutPaths
    from benchflow.task.verifier_core import Verifier

    paths = RolloutPaths(rollout_dir=tmp_path / "trial")
    paths.mkdir()
    (tmp_path / "trial" / "verifier" / "reward.txt").write_text("0")
    verifier = Verifier.__new__(Verifier)
    verifier._rollout_paths = paths
    verifier._sandbox = MagicMock()
    verifier._reward_range = None
    verifier.outputs_at_exit = None
    verifier._written_after_exit = {}

    verifier._take_outputs_at_exit()
    (tmp_path / "trial" / "verifier" / "reward.txt").write_text("1")

    assert verifier._parse_reward_text() == {"reward": 0.0}
    assert verifier._sandbox._benchflow_verifier_outputs_at_exit["files"]["reward.txt"]


@pytest.mark.asyncio
@pytest.mark.parametrize("integrity", [None, "strict"])
async def test_audit_grades_as_beside_live_agent_processes(integrity, tmp_path):
    """With S2 the agent's processes are alive during grading in audit: the
    verifier env goes by a root-only file, never a command line."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from benchflow.sdk import SDK
    from tests.test_sandbox_hardening import _make_env, _make_task

    env = _make_env()
    task = _make_task()
    task.config.verifier.service = "main"
    paths = MagicMock()
    paths.verifier_dir = tmp_path / "verifier"
    verifier = MagicMock()
    verifier.agent_processes_alive = False
    verifier.verify = AsyncMock(return_value=MagicMock(rewards={"reward": 1.0}))
    with patch("benchflow.task.Verifier", return_value=verifier):
        await SDK()._verify(
            env,
            task,
            paths,
            {},
            sandbox_user="agent",
            workspace="/app",
            integrity=integrity,
        )

    assert verifier.agent_processes_alive is (integrity is None)


# -- real trials (Docker) --------------------------------------------------------

DOCKERFILE = """\
FROM python:3.12-slim
WORKDIR /app
RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
"""

# CIR's writer: a root script the turn leaves (`sh /root/.w.sh`, cwd /, no
# descriptor held, the reward's path built at run time), which rewrites the
# reward once test.sh is done.
ROOT_WRITER = (
    "cat > /root/.w.sh <<'W'\n"
    "d=/logs; f=$d/verifier/reward.txt\n"
    "while [ ! -f /tmp/bf-test-done ]; do sleep 0.1; done; sleep 1\n"
    'while :; do echo 1 > "$f"; sleep 0.2; done\n'
    "W\n"
    "(cd / && nohup sh /root/.w.sh >/dev/null 2>&1 &)\nsleep 1\n"
)
# The agent's own process, trying all through test.sh.
AGENT_WRITER = (
    "CMD='while :; do echo 1 > /logs/verifier/reward.txt 2>/dev/null; sleep 0.05; done'\n"
    'nohup runuser -u agent -- sh -c "$CMD" >/dev/null 2>&1 &\nsleep 1\n'
)
WRONG_ANSWER_TEST = """\
#!/bin/bash
sleep 2
echo 0 > /logs/verifier/reward.txt
sleep 1
touch /tmp/bf-test-done
"""


def _task(tmp_path: Path, solve: str, name: str, test: str = WRONG_ANSWER_TEST) -> Path:
    task = tmp_path / name
    (task / "environment").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "tests").mkdir()
    (task / "task.toml").write_text(
        'version = "1.0"\n\n[verifier]\ntimeout_sec = 300.0\n\n'
        "[agent]\ntimeout_sec = 300.0\n\n[environment]\nbuild_timeout_sec = 900.0\n"
    )
    (task / "instruction.md").write_text("Nothing to do.\n")
    (task / "environment" / "Dockerfile").write_text(DOCKERFILE)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\nset -e\n" + solve)
    (task / "tests" / "test.sh").write_text(test)
    return task


async def _oracle(task: Path, **isolation):
    from benchflow.rollout import Rollout, RolloutConfig

    return await Rollout(
        RolloutConfig(
            task_path=task,
            agent="oracle",
            environment="docker",
            sandbox_user="agent",
            jobs_dir=task.parent / "jobs",
            job_name=f"{task.name}-oracle",
            rollout_name=f"{task.name}__oracle",
            **isolation,
        )
    ).run()


def _trial_dir(task: Path) -> Path:
    (found,) = [
        p.parent
        for p in (task.parent / "jobs").rglob("result.json")
        if "verifier-sandbox" not in p.parts
    ]
    return found


@pytest.mark.deterministic
@pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon")
@pytest.mark.parametrize("mode", ["audit", "strict"])
@pytest.mark.parametrize(
    ("writer", "name"), [(ROOT_WRITER, "root"), (AGENT_WRITER, "agent")]
)
async def test_a_leftover_writer_changes_neither_the_reward_nor_the_saved_outputs(
    tmp_path, mode, writer, name
):
    task = _task(tmp_path, writer, f"writer-{name}-{mode}")
    kwargs = {} if mode == "audit" else {"integrity": "strict"}

    result = await _oracle(task, **kwargs)

    assert result.error is None and result.verifier_error is None, result.error
    assert result.rewards == {"reward": 0.0}, (mode, name, result.rewards)
    trial = _trial_dir(task)
    assert (trial / "verifier" / "reward.txt").read_text().strip() == "0"
    if mode == "audit" and name == "root":
        # A root process can still write once test.sh is done: put back, and
        # flagged.
        assert (trial / vo.AT_EXIT / "reward.txt").read_text().strip() == "0"
        record = json.loads((trial / "integrity" / "verifier-inputs.json").read_text())
        assert ("/logs/verifier/reward.txt", "edited") in {
            (f["path"], f["kind"]) for f in record["flags"]
        }


# TB4.1 wal-recovery-ordering's own check (GATE1N): test.sh refuses a reward
# folder that is not root's. It also leaves a file only root can read, as a
# reward written through mkstemp is (#1202).
ROOT_FOLDER_TEST = """\
#!/bin/bash
if [ "$(id -u)" -ne 0 ] || [ -L /logs/verifier ] || { [ -e /logs/verifier ] && [ ! -O /logs/verifier ]; }; then
  echo "verifier requires root and a root-owned reward directory"
  exit 2
fi
stat -c '%u %a' /logs/verifier > /logs/verifier/folder.txt
( umask 077; echo private > /logs/verifier/private.txt )
echo 1 > /logs/verifier/reward.txt
"""


@pytest.mark.deterministic
@pytest.mark.skipif(not docker_available(), reason="needs a Docker daemon")
@pytest.mark.parametrize(
    ("mode", "sandbox"),
    [("audit", "task"), ("audit", "separate"), ("strict", "separate")],
)
async def test_test_sh_finds_its_reward_folder_roots(tmp_path, mode, sandbox):
    """While test.sh runs /logs/verifier is root's, 0755, as before the gate,
    also where it is the host's mount and the host user is not root (this
    test's, on Linux Docker); BenchFlow's own records go there once test.sh
    has exited."""
    task = _task(tmp_path, "true\n", f"rootdir-{mode}-{sandbox}", test=ROOT_FOLDER_TEST)
    kwargs = {} if mode == "audit" else {"integrity": "strict"}
    if sandbox == "separate":
        kwargs["verifier_sandbox"] = "separate"

    result = await _oracle(task, **kwargs)

    assert result.error is None and result.verifier_error is None, (
        result.error,
        result.verifier_error,
    )
    assert result.rewards == {"reward": 1.0}, (mode, sandbox, result.rewards)
    trial = _trial_dir(task)
    assert (trial / "verifier" / "folder.txt").read_text().split() == ["0", "755"]
    assert (trial / "verifier" / "test-stdout.txt").exists()
