"""Merged execs give what the separate execs gave (PY8 U12, research brief Part 3 A2).

U12 runs some setup and verifier commands in one exec instead of several
(``sandbox/_exec_merge.py``) and sends short Daytona execs with
``run_async=False``. Each test here runs the real BenchFlow function against a
real container (``tests/exec_diet_fakes.py``'s ``BoxSandbox``) and compares
what it produced with a golden file in ``tests/fixtures/exec_diet/``. The
goldens were made on the base (``integ/py-0.8`` fc2b9e9a plus U5b's
``py8/daytona-poll``, commit bc0c8529) with ``BENCHFLOW_UPDATE_GOLDEN=1``,
before any of U12's code existed, so a pass on the branch means the merged
execs left every result as it was: the snapshot manifest's bytes, the seeded
workspace, the verifier environment, the oracle's trajectory, the verifier
start's errors and receipt, and the Daytona exec results.

Faults keyed on ``MERGED_MARK`` (the marker of a merged exec) do nothing on
the base; on the branch they make the merged exec fail or answer garbage, and
the result must still match.

Needs Docker (and for the Daytona part the ``sandbox-daytona`` extra).
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from benchflow.sandbox import lockdown
from benchflow.task.config import TaskConfig
from tests._trust_admission import admitted
from tests.exec_diet_fakes import (
    MERGED_MARK,
    PYTHON,
    UBUNTU,
    BoxSandbox,
    Faults,
    as_root,
    box_factory,  # noqa: F401  (fixture)
    check_golden,
)
from tests.integration.deterministic import harness as h

pytestmark = pytest.mark.deterministic

_HEX = re.compile(r"[0-9a-f]{32}")


def _normalized(value):
    if isinstance(value, dict):
        return {k: _normalized(v) for k, v in value.items()}
    if isinstance(value, str):
        return _HEX.sub("<hex>", value)
    return value


# --- the build-config snapshot ---------------------------------------------

_SNAPSHOT_CASES = {
    "all_present": " && ".join(
        f"echo {name} > /app/{name}" for name in lockdown._BUILD_CONFIG_FILES
    ),
    "all_absent": "true",
    "some_present": "echo s > /app/setup.py && echo m > /app/Makefile",
    # cp cannot read the link's target (another process's memory), so it
    # fails for setup.py; pyproject.toml copies.
    "copy_fails": "ln -s /proc/1/mem /app/setup.py && echo p > /app/pyproject.toml",
}


@pytest.mark.parametrize("case", sorted(_SNAPSHOT_CASES))
async def test_snapshot_manifest_bytes(box_factory, case):  # noqa: F811
    box = box_factory(UBUNTU)
    as_root(box, f"mkdir -p /app && {_SNAPSHOT_CASES[case]}")
    await lockdown._snapshot_build_config(BoxSandbox(box), "/app")
    snapshot = as_root(
        box,
        f"cd {lockdown._SNAPSHOT_DIR} && stat -c '%a %U' . && "
        "ls -A | sort && echo --- && cat manifest.json",
    )
    check_golden(f"snapshot-{case}", snapshot)


# --- the verifier workspace seed -------------------------------------------

_SEED_STATE = (
    "for p in /logs /logs/agent /logs/artifacts /logs/verifier /testbed_verify "
    "/testbed_verify/a.txt /testbed_verify/sub /testbed_verify/sub/b.txt; do "
    "stat -c '%n %U:%G %a' $p 2>&1; done"
)


@pytest.mark.parametrize("case", ["with_user", "without_user", "first_fails"])
async def test_seed_verifier_workspace(box_factory, case):  # noqa: F811
    box = box_factory(UBUNTU)
    as_root(
        box,
        "useradd -m agent && mkdir -p /app/sub && echo a > /app/a.txt && "
        "echo b > /app/sub/b.txt && chmod 700 /app/sub && chown -R agent /app",
    )
    if case != "first_fails":
        # Without /logs the first command (chown /logs) fails; the copy after
        # it must still run.
        as_root(box, "mkdir -p /logs/agent /logs/verifier /logs/artifacts")
    await lockdown._seed_verifier_workspace(
        BoxSandbox(box),
        workspace="/app",
        sandbox_user=None if case == "without_user" else "agent",
    )
    check_golden(f"seed-{case}", as_root(box, _SEED_STATE))


# --- the verifier environment ------------------------------------------------

_DEBIAN_PATH = "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin"
_FEDORA_OS_RELEASE = (
    'NAME="Fedora Linux"\nVERSION="40 (Container Image)"\nID=fedora\n'
    'VERSION_ID=40\nPRETTY_NAME="Fedora Linux 40 (Container Image)"\n'
)
# The merged probe's first step (its begin marker included): a separate
# read wraps the same command in its receipt's subshell.
_PROBE_STEP = ":0:begin'\n(\nprintenv PATH\n)"
_ENV_CASES: dict[str, dict] = {
    "debian": {},
    "fedora": {"setup": f"printf %s '{_FEDORA_OS_RELEASE}' > /etc/os-release"},
    # The image's printenv says PATH is unset (a container needs a PATH to
    # start at all).
    "empty_path": {
        "setup": "printf '%s\\n' '#!/bin/sh' '[ \"$1\" = PATH ] && exit 1' "
        "'exec /usr/bin/printenv \"$@\"' > /usr/local/bin/printenv && "
        "chmod 755 /usr/local/bin/printenv"
    },
    "path_extras": {
        "run": [
            "-e",
            f"PATH=/opt/trusted/bin:{_DEBIAN_PATH}:/tmp/untrusted:/app/bin",
        ],
        "setup": "mkdir -p /opt/trusted/bin /tmp/untrusted /app/bin && "
        "chmod 755 /opt/trusted/bin && chmod 1777 /tmp/untrusted",
        # A PATH with entries left out needs the task's tool contract
        # (c84e6eb8c, merged after this golden was made): the empty one.
        "toml": "required_tools = []\n",
    },
    "pythonpath_set": {
        "run": ["-e", "PYTHONPATH=/opt/pylib:/tmp/evil"],
        "setup": "mkdir -p /opt/pylib /tmp/evil && chmod 777 /tmp/evil",
    },
    "pythonpath_empty": {"run": ["-e", "PYTHONPATH="]},
    "python3_absent": {"image": UBUNTU},
    "probe_fails": {"faults": Faults(raise_on={_PROBE_STEP: RuntimeError("lost")})},
    "probe_malformed": {
        "faults": Faults(status_on={_PROBE_STEP: (0, f"{MERGED_MARK}0:begin\nx")})
    },
    "separate_verifier": {"agent_paths": ("/app",)},
}


def _verifier_task(tmp_path: Path, toml: str = "") -> MagicMock:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    task = MagicMock()
    task.name = "exec-diet"
    task.task_dir = tests_dir.parent
    task.paths.task_dir = tests_dir.parent
    task.paths.tests_dir = tests_dir
    task.paths.test_path = tests_dir / "test.sh"
    task.paths.uses_native_verifier_dir = False
    task.config = TaskConfig.model_validate_toml(
        'version = "1.0"\n[verifier]\ntimeout_sec = 60\n' + toml
    )
    return task


# The verifier-env-*.json goldens were made again on git archive of
# integ/py-0.8 363d815b (without U12): V7 names BASH_ENV and ENV (both empty
# here) in every verifier env, so each golden gained those two keys and
# nothing else changed.
# Regenerated after ccc67ac5's integration merge: CONFIRM-lockdown-1/2
# intentionally add PIP_CONFIG_FILE=/dev/null in every case. All other
# recorded values remain identical to the integration-base goldens.
# path_extras declares the empty tool contract its left-out PATH entries
# need (c84e6eb8c); without one it was a refusal, so it alone missed the
# runner stub's BENCHFLOW_PYTEST (96155b555) every other case gained.
@pytest.mark.parametrize("case", sorted(_ENV_CASES))
async def test_build_verifier_env(box_factory, tmp_path, case):  # noqa: F811
    spec = _ENV_CASES[case]
    box = box_factory(spec.get("image", PYTHON), *spec.get("run", []))
    as_root(box, "useradd -m agent; mkdir -p /app /logs/verifier")
    if "setup" in spec:
        as_root(box, spec["setup"])
    sandbox = BoxSandbox(box, faults=spec.get("faults"))
    try:
        built: dict = await lockdown._build_verifier_env(
            sandbox,
            _verifier_task(tmp_path, spec.get("toml", "")),
            "agent",
            "/app",
            agent_paths=spec.get("agent_paths"),
        )
    except Exception as exc:
        built = {"raised": type(exc).__name__, "message": str(exc)}
    check_golden(f"verifier-env-{case}", _normalized(built))


# --- the oracle ----------------------------------------------------------------

_ORACLE_CASES = {
    "rc0": ("echo solved; echo to-stderr >&2", 60, None),
    "rc3": ("echo partial; exit 3", 60, None),
    "timeout": ("sleep 20", 2, None),
    "raises": ("echo never", 60, ConnectionResetError("session lost")),
}


@pytest.mark.parametrize("case", sorted(_ORACLE_CASES))
async def test_run_oracle_trajectory(box_factory, tmp_path, case):  # noqa: F811
    from benchflow.rollout._setup import _run_oracle

    script, timeout, error = _ORACLE_CASES[case]
    task = tmp_path / "task"
    shutil.copytree(h.TEMPLATE_TASK, task)
    (task / "solution" / "solve.sh").write_text(f"#!/bin/bash\n{script}\n")
    box = box_factory(UBUNTU)
    as_root(box, "mkdir -p /logs/agent /solution")
    sandbox = BoxSandbox(
        box,
        faults=Faults(raise_on={"solve.sh": error}) if error is not None else None,
    )
    await sandbox.upload_file(
        task / "solution" / "solve.sh",
        "/solution/solve.sh",
        admission=admitted(sandbox),
    )
    try:
        trajectory, name = await _run_oracle(sandbox, task, timeout)
        outcome = {"trajectory": trajectory, "name": name}
    except Exception as exc:
        outcome = {"raised": type(exc).__name__, "message": str(exc)}
    # What ran after solve.sh: on a timeout or an exception, no preview.
    outcome["preview_ran"] = any(
        "tail -c" in e.command and "oracle.txt" in e.command for e in sandbox.execs
    )
    check_golden(f"oracle-{case}", outcome)


# --- the verifier start (chmod of test.sh and the start-receipt folder) -------

# A chmod in front of the real one that fails for test.sh while the counter
# file says so, and counts each run for test.sh.
_FAKE_CHMOD = """#!/bin/sh
case "$*" in *test.sh*)
  n=$(cat /tmp/chmod-tries 2>/dev/null || echo 0); n=$((n + 1)); echo $n > /tmp/chmod-tries
  fail=$(cat /tmp/chmod-fail 2>/dev/null || echo 0)
  [ "$n" -le "$fail" ] && exit 7
  [ -f /tmp/chmod-sleep ] && [ "$n" -eq 1 ] && sleep "$(cat /tmp/chmod-sleep)";;
esac
exec /usr/bin/chmod "$@"
"""

_START_CASES: dict[str, dict] = {
    "ok": {},
    "chmod_fails": {"setup": "echo 9 > /tmp/chmod-fail"},
    "chmod_fails_twice": {"setup": "echo 2 > /tmp/chmod-fail"},
    "receipt_dir_fails": {"setup": "mkdir -p /run && touch /run/benchflow"},
    "receipt_dir_times_out": {
        "faults": Faults(
            raise_on={
                "mkdir -p /run/benchflow": RuntimeError(
                    "Command timed out after 10 seconds"
                )
            }
        )
    },
    "receipt_disabled": {"receipt": False},
    "sidecar_service": {"toml": 'service = "target"\n'},
    "start_exec_lost": {
        "faults": Faults(raise_on={MERGED_MARK: ConnectionResetError("lost")})
    },
    # chmod's first run takes 12 s: past the merged exec's 10 s, within
    # chmod's own 60 s.
    "chmod_slow": {"setup": "echo 12 > /tmp/chmod-sleep"},
    # A script strategy whose command names no path runs no chmod.
    "no_chmod": {"strategy": "sh test.sh"},
}


@pytest.mark.parametrize("case", sorted(_START_CASES))
async def test_verifier_start(box_factory, tmp_path, monkeypatch, case):  # noqa: F811
    from benchflow.task import verifier_core
    from benchflow.task.paths import RolloutPaths
    from benchflow.task.verifier_core import Verifier

    monkeypatch.setattr(verifier_core, "_SETUP_RETRY_DELAYS_SEC", (0.0, 0.0))
    spec = _START_CASES[case]
    box = box_factory(UBUNTU)
    as_root(box, "mkdir -p /logs/verifier /usr/local/sbin")
    fake = tmp_path / "chmod"
    fake.write_text(_FAKE_CHMOD)
    sandbox = BoxSandbox(box, faults=spec.get("faults"))
    await sandbox.upload_file(
        fake, "/usr/local/sbin/chmod", admission=admitted(sandbox)
    )
    as_root(box, "/usr/bin/chmod 755 /usr/local/sbin/chmod")
    if "setup" in spec:
        as_root(box, spec["setup"])
    paths = RolloutPaths(tmp_path / "rollout")
    paths.mkdir()
    verifier = Verifier(
        _verifier_task(tmp_path, spec.get("toml", "")),
        paths,
        sandbox,
        execution_receipt=spec.get("receipt", True),
    )
    try:
        if "strategy" in spec:
            from benchflow.task.verifier_document import VerifierStrategy

            result = await verifier._verify_test_script(
                strategy=VerifierStrategy(
                    name="script", type="script", config={"command": spec["strategy"]}
                )
            )
        else:
            result = await verifier.verify()
        outcome: dict = {"rewards": result.rewards}
    except Exception as exc:
        outcome = {"raised": type(exc).__name__, "message": str(exc)}
    outcome["receipt"] = (
        None
        if verifier.execution_receipt is None
        else [_normalized(verifier.execution_receipt[0]), verifier.execution_receipt[1]]
    )
    outcome["receipt_dir"] = as_root(
        box, "stat -c '%a %F' /run/benchflow 2>&1 || true"
    ).strip()
    if case != "chmod_slow":
        # How often chmod ran for test.sh (the slow case runs it once more on
        # the branch: the merged exec ran out of time and chmod ran again).
        outcome["chmod_tries"] = as_root(
            box, "cat /tmp/chmod-tries 2>/dev/null || echo 0"
        ).strip()
    check_golden(f"verifier-start-{case}", outcome)


# --- Daytona: synchronous and asynchronous execs ----------------------------


class _LocalProcess:
    """``sandbox.process`` running each session command in a local ``sh``
    (only the commands below, which print, sleep and exit). With
    ``run_async=False`` it answers when the command is done, as Daytona's
    toolbox does; otherwise it answers at once and reports the exit code to
    status requests once the command has finished."""

    def __init__(self, execute_error: BaseException | None = None) -> None:
        self.execute_error = execute_error
        self.commands: dict[str, dict] = {}
        self.requests: list[str] = []

    async def create_session(self, session_id):
        self.requests.append("create")

    async def delete_session(self, session_id):
        self.requests.append("delete")

    async def _run(self, command: str, record: dict) -> None:
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        record.update(
            exit_code=proc.returncode, stdout=out.decode(), stderr=err.decode()
        )

    async def execute_session_command(self, session_id, request, timeout=None):
        self.requests.append("execute")
        if self.execute_error is not None:
            raise self.execute_error
        record: dict = {"exit_code": None, "stdout": "", "stderr": ""}
        self.commands[session_id] = record
        if request.run_async is False:
            await self._run(request.command, record)
            return SimpleNamespace(cmd_id=session_id, output=None, **record)
        task = asyncio.ensure_future(self._run(request.command, record))
        record["task"] = task
        return SimpleNamespace(cmd_id=session_id, exit_code=None)

    async def get_session_command(self, session_id, command_id):
        self.requests.append("status")
        return SimpleNamespace(
            id=command_id, exit_code=self.commands[command_id]["exit_code"]
        )

    async def get_session_command_logs(self, session_id, command_id):
        self.requests.append("logs")
        record = self.commands[command_id]
        return SimpleNamespace(stdout=record["stdout"], stderr=record["stderr"])

    async def get_session(self, session_id):
        self.requests.append("session")
        record = self.commands.get(session_id)
        return SimpleNamespace(
            commands=[]
            if record is None
            else [SimpleNamespace(id=session_id, exit_code=record["exit_code"])]
        )


def _daytona(process) -> object:
    pytest.importorskip("daytona")
    from benchflow.sandbox.daytona import DaytonaSandbox, _load_daytona_sdk
    from benchflow.sandbox.daytona_strategies import _DaytonaDirect

    _load_daytona_sdk()
    sandbox = DaytonaSandbox.__new__(DaytonaSandbox)
    sandbox.default_user = None
    sandbox._persistent_env = {}
    sandbox._sandbox = SimpleNamespace(process=process)
    sandbox._strategy = _DaytonaDirect(sandbox)
    sandbox.rollout_paths = None
    return sandbox


_DAYTONA_CASES = {
    "rc0_streams": ("echo out; echo err >&2", 10),
    "rc3": ("echo partial; exit 3", 10),
    "rc124_itself": ("echo early; exit 124", 10),
    "shell_timeout": ("sleep 4; echo late", 1),
    "no_budget": ("echo none", None),
    "long_budget": ("echo long", 60),
    "zero_budget": ("echo zero", 0),
    "negative_budget": ("echo negative", -1),
    "transient_execute_error": ("echo never", 10),
}


@pytest.mark.parametrize("case", sorted(_DAYTONA_CASES))
async def test_daytona_exec_result(case):
    command, budget = _DAYTONA_CASES[case]
    process = _LocalProcess(
        execute_error=TimeoutError("read timed out")
        if case == "transient_execute_error"
        else None
    )
    sandbox = _daytona(process)
    try:
        result = await sandbox.exec(command, timeout_sec=budget)
        outcome: dict = {
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except Exception as exc:
        outcome = {"raised": type(exc).__name__, "message": str(exc)}
    check_golden(f"daytona-{case}", outcome)


def test_golden_files_are_valid_json():
    for path in sorted((Path(__file__).parent / "fixtures" / "exec_diet").glob("*")):
        json.loads(path.read_text())
