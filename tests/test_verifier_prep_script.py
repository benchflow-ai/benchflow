"""Golden tests for the verifier preparation in ``harden_before_verify``.

Before the verifier runs, ``harden_before_verify`` kills the sandbox user's
processes, clears ``/logs/verifier``, prepares ``/app``, reclaims caches,
freezes the workspace and removes injected Python hooks. Each of those steps
used to be its own exec, and on Daytona every exec is a polled round trip.
On the backends that set ``verifier_prep_script`` they now run as one script
in one exec, which prints a status line per step.

``tests/fixtures/verifier_prep_golden.json`` is what the step-by-step code
did on integ/py-0.8 (85b87bc9, regenerated on V3's merge of 66471e95 for
its pre-freeze probes and the clean-up's interpreter; and after
two later merges, whose freeze copies a regular file it may not
chown, in the walker and in its shell fallback, and whose clear gives
/logs/verifier back to root: two commands changed, no scenario record did;
and for the kill's PID 1 keep-alive rule: one command changed; and for the
freeze's git hook and config purge and the verifier env's GIT_CONFIG keys) for success
and for each step failing or timing out: which steps ran, the kill defense row, as which user and
with which budget, each step's exact command, the error hardening raised, the
verifier env's keys, and the ``harden_before_verify`` sub-keys the trial's
verifier phase writes to timing.json. The
tests hold the step-by-step path and the one-script path to that record, the
script run by a real shell with each step's command swapped for a stub that
gives the scenario's outcome. To regenerate the record on a base that runs
each step in its own exec, run this file there with BENCHFLOW_UPDATE_GOLDEN=1
and review the diff.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from benchflow._utils import kill_record, timing_spans
from benchflow.sandbox import lockdown
from tests._receipt_fakes import receipt_answer, receipt_aware
from tests.sandbox_mounts import ROOT_MOUNTINFO

GOLDEN = Path(__file__).parent / "fixtures" / "verifier_prep_golden.json"
UPDATE_GOLDEN_ENV = "BENCHFLOW_UPDATE_GOLDEN"
# lockdown._PREP_MARK; spelled out so this file imports on a base without it.
MARK = "BENCHFLOW-PREP"

CHECKED = (
    "kill_sandbox_user",
    "clear_verifier_output",
    "prepare_app_dir",
    "chown_workspace",
    # V8's freeze: these fail hardening when their Python fails
    # (V8-R1-B-1), and the workspace venv's startup hooks are purged.
    "purge_external_symlinks",
    "purge_workspace_python_hooks",
    "purge_git_controls",
)
UNCHECKED = (
    "reclaim_disk",
    "purge_pycache",
    "cleanup_conftests",
)
STEPS = CHECKED + UNCHECKED
# The harden_before_verify timing sub-key each step's seconds add to.
SPAN = {
    "kill_sandbox_user": "kill_sandbox_user_procs",
    "purge_external_symlinks": "freeze_workspace",
    "purge_pycache": "freeze_workspace",
    "purge_workspace_python_hooks": "freeze_workspace",
    "purge_git_controls": "freeze_workspace",
    "chown_workspace": "freeze_workspace",
}


def _scenarios() -> dict[str, dict[str, Any]]:
    """Inputs and per-step outcomes: ``("exit", rc, stdout, stderr)`` or
    ``("timeout",)``; a step with no outcome exits 0 with no output."""
    base = {"sandbox_user": "agent", "workspace": "/app", "restore": False}
    scenarios: dict[str, dict[str, Any]] = {
        "success": {**base, "outcomes": {}},
        "no sandbox user": {**base, "sandbox_user": None, "outcomes": {}},
        "no workspace": {**base, "workspace": None, "outcomes": {}},
        "restore workspace": {**base, "restore": True, "outcomes": {}},
        "restore workspace, chown fails": {
            **base,
            "restore": True,
            "outcomes": {
                "chown_workspace": ("exit", 1, "", "chown: operation not permitted\n")
            },
        },
        "chown fails with stdout and stderr": {
            **base,
            "outcomes": {
                "chown_workspace": (
                    "exit",
                    1,
                    "changed ownership of '/app/a'\n",
                    "chown: cannot access '/app/b': No such file or directory\n",
                )
            },
        },
    }
    for rc, name in enumerate(CHECKED, start=1):
        scenarios[f"{name} fails"] = {
            **base,
            "outcomes": {name: ("exit", rc, "", f"{name}: it went wrong\n")},
        }
    for name in UNCHECKED:
        scenarios[f"{name} exits 1"] = {
            **base,
            "outcomes": {name: ("exit", 1, "ignored\n", "ignored too\n")},
        }
    for name in STEPS:
        scenarios[f"{name} times out"] = {**base, "outcomes": {name: ("timeout",)}}
    return scenarios


SCENARIOS = _scenarios()


def step_name(command: str, workspace: str | None) -> str | None:
    """Which preparation step *command* is, from base-era builders only."""
    if "BF killed" in command:
        return "kill_sandbox_user"
    known = {
        lockdown._CLEAR_VERIFIER_DIR_CMD: "clear_verifier_output",
        lockdown._ENSURE_APP_DIR_CMD: "prepare_app_dir",
        lockdown.build_reclaim_caches_cmd(workspace): "reclaim_disk",
        lockdown._build_cleanup_cmd(workspace=workspace): "cleanup_conftests",
        # With a sandbox user, the hook clean-up's root-made rule (ACC-RUN2).
        lockdown._build_cleanup_cmd(
            workspace=workspace, agent_user=True
        ): "cleanup_conftests",
    }
    if workspace:
        known[lockdown._purge_external_symlinks_cmd(workspace)] = (
            "purge_external_symlinks"
        )
        known[lockdown._purge_pycache_cmd(workspace)] = "purge_pycache"
        known[lockdown._purge_workspace_python_hooks_cmd(workspace)] = (
            "purge_workspace_python_hooks"
        )
        for agent_user in (False, True):
            known[
                lockdown._purge_git_controls_cmd(workspace, agent_user=agent_user)
            ] = "purge_git_controls"
        known[lockdown._chown_workspace_cmd(workspace)] = "chown_workspace"
    return known.get(command)


# Every root preparation command runs on V3's root route: the
# step-by-step helpers prefix it to each command, the script's launch to the
# whole script. The record is of the commands themselves.
_ROUTE_LINES = (lockdown._prep_on_path(""), lockdown._on_path(""))


def off_route(command: str) -> str:
    """*command* without the root route's launch line, if it has one."""
    for line in _ROUTE_LINES:
        if command.startswith(line):
            return command[len(line) :]
    return command


def _manifest() -> str:
    return json.dumps({name: False for name in lockdown._BUILD_CONFIG_FILES})


def _other_result(command: str) -> SimpleNamespace:
    """What a plain image answers to the verifier-env probes (step 7) and the
    workspace restore."""
    if "/proc/self/mountinfo" in command:
        stdout = ROOT_MOUNTINFO
    elif (
        "find -H" in command
        and "printf 'complete\\n'" in command
        and "captur" not in command
    ):
        # V3's pre-freeze inode identities: none under the blocked trees.
        stdout = "complete\n"
    elif command in ("printenv PATH", lockdown._IMAGE_PATH_CMD):
        # integ's read and PT's (the route keeps the image PATH as data).
        stdout = "/usr/local/bin:/usr/bin"
    elif command.startswith("printenv PYTHONPATH"):
        stdout = ""
    elif command.startswith("cat /etc/os-release"):
        stdout = "ID=ubuntu\n"
    elif command == f"cat {lockdown._SNAPSHOT_MANIFEST}":
        stdout = _manifest()
    elif "from importlib.metadata import entry_points" in command:
        stdout = json.dumps({"plugins": [], "rejected": []})
    elif command.startswith("python3 -c"):
        stdout = "[]"
    else:
        stdout = ""
    return SimpleNamespace(return_code=0, stdout=stdout, stderr="")


def _exit_result(rc: int, stdout: str, stderr: str, merge: bool) -> SimpleNamespace:
    # Docker's exec merges stderr into stdout; Daytona's keeps them apart.
    if merge:
        return SimpleNamespace(
            return_code=rc, stdout=(stdout + stderr) or None, stderr=None
        )
    return SimpleNamespace(return_code=rc, stdout=stdout, stderr=stderr)


class OneExecEachSandbox:
    """Answers each preparation step's own exec with the scenario's outcome."""

    def __init__(self, scenario: dict[str, Any], merge: bool) -> None:
        # The pre-agent PATH inspector snapshot_build_config leaves behind,
        # which V3's pre-freeze exclusions require.
        self._benchflow_path_inspectors = {"main": "/etc/fake-inspector"}
        self.outcomes = scenario["outcomes"]
        self.workspace = scenario["workspace"]
        self.merge = merge
        self.steps: list[list[Any]] = []
        self.commands: dict[str, str] = {}
        self.other_calls = 0
        self.execs = 0

    @receipt_aware
    async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.execs += 1
        command = off_route(command)
        name = step_name(command, self.workspace)
        if name is None:
            self.other_calls += 1
            return _other_result(command)
        return self.answer(name, command, kwargs.get("user"), kwargs.get("timeout_sec"))

    def answer(
        self, name: str, command: str, user: Any, timeout: Any
    ) -> SimpleNamespace:
        self.steps.append([name, user, timeout])
        self.commands[name] = command
        outcome = self.outcomes.get(name)
        if outcome is None:
            return SimpleNamespace(
                return_code=0,
                stdout="BF killed 0 end\n" if name == "kill_sandbox_user" else "",
                stderr="",
            )
        if outcome[0] == "timeout":
            raise RuntimeError(f"Command timed out after {timeout} seconds")
        _, rc, stdout, stderr = outcome
        return _exit_result(rc, stdout, stderr, self.merge)


# One step of the script: the step's command, verbatim, in a subshell.
_STEP_BLOCK = re.compile(
    r"__bf_now; __bf_t0=\$__bf_t\n"
    r'\(\n(?:trap - EXIT\n)?(?P<command>.*?)\n\) </dev/null >"\$__bf_d/o(?P<index>\d+)" '
    r'2>"\$__bf_d/e\d+" 3>&- &\n__bf_p=\$!\n\(\n'
    r"(?:trap - EXIT\n)?(?:__bf_s=; __bf_e=\n)?trap '[^\n]*' TERM\n"
    r'PATH="\$__bf_path" sleep (?P<budget>\d+) &\n',
    re.DOTALL,
)
_WORK_DIR_LINE = re.compile(r"^__bf_d=/tmp/\.runtime_prep_[0-9a-f]{16}$", re.MULTILINE)


def _work_dir(tmp: Path, name: str = "prep") -> Path:
    """Where a test puts the script's work folder: in a sticky folder the
    test's user owns, which the script accepts as it accepts a sticky /tmp
    that root owns."""
    parent = tmp / "sticky"
    parent.mkdir(exist_ok=True)
    parent.chmod(0o1700)
    return parent / name


# Seconds the stub of the script's k-th step (from 0) takes on the fake clock.
STEP_SECONDS = 0.25


def _prep_transport(exec_fn):
    """The golden fake models receipt payloads, not transport launch text.

    Real builder replays in test_verifier_prep_compat cover the prelude itself.
    Strip only that builtin line before the existing receipt-aware fake, so
    its historical merged/separate-stream contract remains under test.
    """

    @functools.wraps(exec_fn)
    async def execute(self, command, **kwargs):
        stripped = off_route(command)
        if f"{MARK}-START" in stripped:
            # The script launches on the preparation route.
            assert command.startswith(lockdown._prep_on_path(""))
        command = stripped
        return await exec_fn(self, command, **kwargs)

    return execute


class OneScriptSandbox(OneExecEachSandbox):
    """Runs the preparation script in a real shell, each step's command
    swapped for a stub that logs the step, moves a fake clock on and gives
    the scenario's outcome. The script's own lines run unchanged."""

    verifier_prep_script = True

    def __init__(
        self, scenario: dict[str, Any], merge: bool, shell: list[str], tmp: Path
    ) -> None:
        super().__init__(scenario, merge)
        self.shell = shell
        self.tmp = tmp
        self.scripts = 0
        self.script_results = []

    @_prep_transport
    @receipt_aware
    async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
        if f"{MARK}-START" not in command:
            return await super().exec(command, **kwargs)
        self.execs += 1
        self.scripts += 1
        assert kwargs.get("user") == "root"
        return await asyncio.to_thread(
            self.run_script, command, kwargs.get("timeout_sec")
        )

    def run_script(self, script: str, exec_timeout: int) -> SimpleNamespace:
        ran = self.tmp / f"ran{self.scripts}"
        clock = self.tmp / "uptime"
        clock.write_text("100.00 0.00\n")
        assert len(_WORK_DIR_LINE.findall(script)) == 1, script[:400]
        script = _WORK_DIR_LINE.sub(
            f"__bf_d={shlex.quote(str(_work_dir(self.tmp, f'prep{self.scripts}')))}",
            script,
        )
        assert script.count("</proc/uptime") == 1
        script = script.replace("</proc/uptime", f"<{shlex.quote(str(clock))}")
        blocks = list(_STEP_BLOCK.finditer(script))
        assert [int(b["index"]) for b in blocks] == list(range(len(blocks)))
        embedded: dict[str, tuple[str, int]] = {}
        pieces, last, now = [], 0, 100.0
        for k, block in enumerate(blocks):
            command = block["command"]
            name = step_name(command, self.workspace)
            assert name is not None, f"unknown step command: {command[:200]}"
            embedded[name] = (command, int(block["budget"]))
            now += STEP_SECONDS * (k + 1)
            outcome = self.outcomes.get(name)
            stub = [
                f"echo {name} >> {shlex.quote(str(ran))}",
                f"printf '%.2f 0.00\\n' {now} > {shlex.quote(str(clock))}",
            ]
            budget = block["budget"]
            if outcome and outcome[0] == "timeout":
                stub.append("exec sleep 30")
                budget = "0.2"  # GNU, busybox and BSD sleep take fractions
            elif outcome:
                _, rc, stdout, stderr = outcome
                stub += [
                    f"printf %s {shlex.quote(stdout)}",
                    f"printf %s {shlex.quote(stderr)} >&2",
                    f"exit {rc}",
                ]
            text = block.group(0)
            needle = f"{command}\n) </dev/null"
            assert needle in text
            text = text.replace(needle, "\n".join(stub) + "\n) </dev/null", 1)
            text = text.replace(
                f"sleep {block['budget']} &\n", f"sleep {budget} &\n", 1
            )
            pieces += [script[last : block.start()], text]
            last = block.end()
        pieces.append(script[last:])
        assert sum(b for _, b in embedded.values()) + 30 == exec_timeout
        done = subprocess.run(
            [*self.shell, "".join(pieces)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT if self.merge else subprocess.PIPE,
            text=True,
            timeout=60,
        )
        self.script_results.append((done.returncode, done.stdout, done.stderr))
        for name in ran.read_text().split() if ran.exists() else []:
            command, budget = embedded[name]
            self.steps.append([name, "root", budget])
            self.commands[name] = command
        return SimpleNamespace(
            return_code=done.returncode,
            stdout=done.stdout,
            stderr=None if self.merge else done.stderr,
        )


def _task() -> Any:
    # As tests/test_sandbox_hardening.py builds its task.
    task = MagicMock()
    task.config.verifier.env = None
    task.config.verifier.user = None
    task.config.verifier.pytest_plugins = []
    task.config.verifier.service = "main"
    task.task_dir = None
    task.paths.uses_native_verifier_dir = False
    return task


_HEX = re.compile(r"[0-9a-f]{8,}")


async def _harden(env: Any, scenario: dict[str, Any], tmp: Path) -> dict[str, Any]:
    """Record exactly the hardening span and defense row the rollout publishes."""
    task = _task()
    timing: dict[str, Any] = {}
    defense: dict[str, Any] = {}
    error = None
    seconds = None
    with (
        timing_spans.recording_into(timing),
        timing_spans.span(timing_spans.HARDEN_BEFORE_VERIFY, "total"),
        kill_record.recording_into(defense),
    ):
        try:
            seconds = await lockdown.harden_before_verify(
                env,
                task,
                scenario["sandbox_user"],
                scenario["workspace"],
                restore_workspace=scenario["restore"],
            )
        except RuntimeError as exc:
            error = str(exc)
    steps = timing_spans.rounded(timing).get(timing_spans.HARDEN_BEFORE_VERIFY, {})
    verifier_env = task.config.verifier.env
    return {
        "record": {
            "steps": env.steps,
            "error": error,
            "defense": defense,
            "env_keys": sorted(verifier_env) if verifier_env else None,
            "other_calls": env.other_calls,
            "timing_keys": list(steps),
        },
        "commands": env.commands,
        "seconds": seconds,
        "timing": steps,
        "env": {k: _HEX.sub("<hex>", v) for k, v in (verifier_env or {}).items()},
    }


def _golden() -> dict[str, Any]:
    return json.loads(GOLDEN.read_text())


def _shells() -> list[Any]:
    params = []
    for shell in (["sh", "-c"], ["bash", "-c"], ["busybox", "sh", "-c"]):
        params.append(
            pytest.param(
                shell,
                id=shell[0],
                marks=pytest.mark.skipif(
                    shutil.which(shell[0]) is None, reason=f"no {shell[0]} here"
                ),
            )
        )
    return params


@pytest.mark.asyncio
async def test_one_exec_per_step_matches_golden(tmp_path):
    """The step-by-step path (every backend without ``verifier_prep_script``)
    runs the same steps, commands, budgets and errors as integ/py-0.8."""
    observed: dict[str, Any] = {"scenarios": {}, "commands": {}}
    for title, scenario in SCENARIOS.items():
        for merge in (True, False):
            result = await _harden(
                OneExecEachSandbox(scenario, merge), scenario, tmp_path
            )
            key = f"{title} ({'merged' if merge else 'separate'} streams)"
            observed["scenarios"][key] = result["record"]
            if title == "success":
                observed["commands"] = result["commands"]
    if os.environ.get(UPDATE_GOLDEN_ENV) == "1":
        GOLDEN.write_text(json.dumps(observed, indent=1, sort_keys=True) + "\n")
    golden = _golden()
    assert observed["commands"] == golden["commands"]
    assert observed["scenarios"] == golden["scenarios"]


@pytest.mark.asyncio
@pytest.mark.parametrize("shell", _shells())
@pytest.mark.parametrize("merge", [True, False], ids=["merged", "separate"])
@pytest.mark.parametrize("title", list(SCENARIOS))
async def test_one_script_matches_golden(title, merge, shell, tmp_path):
    """One script gives the golden record: the same steps in the same order,
    each step's command byte for byte with its own budget, and the same error
    for the step that failed or timed out; and the pure-command group takes one exec (two around
    the opt-in workspace restore), after the kill and pre-freeze proofs."""
    scenario = SCENARIOS[title]
    golden = _golden()
    key = f"{title} ({'merged' if merge else 'separate'} streams)"
    env = OneScriptSandbox(scenario, merge, shell, tmp_path)
    result = await _harden(env, scenario, tmp_path)
    assert result["record"] == golden["scenarios"][key], env.script_results
    for name, command in result["commands"].items():
        if scenario["sandbox_user"] == "agent" and scenario["workspace"] == "/app":
            assert command == golden["commands"][name], name
    batched = [row for row in env.steps if row[0] != "kill_sandbox_user"]
    assert env.scripts == (
        0
        if not batched
        else 2
        if scenario["restore"]
        and any(row[0] == "purge_external_symlinks" for row in batched)
        else 1
    )
    assert env.execs == env.scripts + env.other_calls + sum(
        row[0] == "kill_sandbox_user" for row in env.steps
    ), env.script_results


@pytest.mark.asyncio
@pytest.mark.parametrize("title", list(SCENARIOS))
async def test_step_seconds_have_the_same_keys_filled_from_status_lines(
    title, tmp_path
):
    """Both paths fill the same timing.json sub-keys of
    ``harden_before_verify`` (the golden holds the step-by-step code's keys,
    ``total`` included), for exactly the steps that ran, on failure too; the
    script's seconds are its status lines' (the stubs move a fake
    /proc/uptime on by a known amount per step)."""
    scenario = SCENARIOS[title]
    one_each = await _harden(OneExecEachSandbox(scenario, False), scenario, tmp_path)
    script = await _harden(
        OneScriptSandbox(scenario, False, ["sh", "-c"], tmp_path), scenario, tmp_path
    )
    key = f"{title} (separate streams)"
    golden_keys = _golden()["scenarios"][key]["timing_keys"]
    assert script["record"]["timing_keys"] == golden_keys
    assert one_each["record"]["timing_keys"] == golden_keys
    ran = [
        name for name, _, _ in script["record"]["steps"] if name != "kill_sandbox_user"
    ]
    assert {SPAN.get(name, name) for name in ran} <= set(script["timing"])
    # The fake clock restarts with each script; step k of a script takes
    # STEP_SECONDS * (k + 1).
    first = (
        ran[: ran.index("purge_external_symlinks")]
        if scenario["restore"] and "purge_external_symlinks" in ran
        else ran
    )
    expected: dict[str, float] = {}
    for part in (first, ran[len(first) :]):
        for k, name in enumerate(part):
            span = SPAN.get(name, name)
            expected[span] = expected.get(span, 0.0) + STEP_SECONDS * (k + 1)
    assert {span: script["timing"][span] for span in expected} == expected
    if one_each["record"]["error"] is not None:
        # A failed hardening returns nothing; its seconds are in timing.json.
        assert one_each["seconds"] is None and script["seconds"] is None
        return
    assert set(script["seconds"]) == set(one_each["seconds"])
    assert {span: script["seconds"][span] for span in expected} == expected
    assert script["env"] == one_each["env"]


@pytest.mark.asyncio
async def test_failed_step_seconds_come_from_its_status_line(tmp_path):
    """A failing step still has its status line, so its time is known."""
    scenario = SCENARIOS["chown_workspace fails"]
    seconds: dict[str, float] = {}
    env = OneScriptSandbox(scenario, False, ["sh", "-c"], tmp_path)
    steps = [
        *lockdown._verifier_output_steps(),
        lockdown._reclaim_step("/app"),
        *lockdown._freeze_steps("/app"),
        lockdown._cleanup_step(lockdown.HARDENING_DEFAULTS, "/app"),
    ]
    with pytest.raises(
        RuntimeError, match="freezing workspace ownership exited with rc=4"
    ):
        await lockdown._run_prep_steps(env, steps, seconds)
    assert list(seconds) == [
        "clear_verifier_output",
        "prepare_app_dir",
        "reclaim_disk",
        "freeze_workspace",
    ]
    # purge_external_symlinks, purge_pycache, the workspace venv hooks purge
    # (V8), the git purge and chown_workspace: steps 3-7.
    assert seconds["freeze_workspace"] == STEP_SECONDS * (4 + 5 + 6 + 7 + 8)


def _step(name: str, command: str, timeout: int = 5, **kwargs: Any) -> Any:
    return lockdown._PrepStep(name, name, command, timeout, **kwargs)


class ShellSandbox:
    """Runs whatever it is given in ``sh -c``, as Daytona would (separate
    streams). For the script's own lines; the steps here are harmless."""

    verifier_prep_script = True

    def __init__(
        self,
        tmp: Path,
        rewrite_dir: bool = True,
        shell: tuple[str, ...] = ("sh", "-c"),
        work_dir: Path | None = None,
    ) -> None:
        self.tmp = tmp
        self.rewrite_dir = rewrite_dir
        self.work_dir = work_dir
        self.shell = list(shell)
        self.commands: list[str] = []

    async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
        self.commands.append(command)
        if self.rewrite_dir:
            command = _WORK_DIR_LINE.sub(
                f"__bf_d={shlex.quote(str(self.work_dir or _work_dir(self.tmp)))}",
                command,
            )
        done = await asyncio.to_thread(
            subprocess.run,
            [*self.shell, command],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return SimpleNamespace(
            return_code=done.returncode, stdout=done.stdout, stderr=done.stderr
        )


@pytest.mark.asyncio
async def test_unusable_work_dir_runs_the_steps_one_by_one(tmp_path):
    """A work folder that cannot be made (here: it exists, as one an agent
    planted would) means no step ran: the steps run one exec each, and the
    planted folder is left alone."""
    planted = _work_dir(tmp_path)
    planted.mkdir()
    (planted / "keep").write_text("x")
    log = tmp_path / "log"
    env = ShellSandbox(tmp_path)
    steps = [
        _step("one", f"echo one >> {shlex.quote(str(log))}", label="one failed"),
        _step("two", f"echo two >> {shlex.quote(str(log))}"),
    ]
    await lockdown._run_prep_steps(env, steps)
    assert log.read_text().split() == ["one", "two"]
    assert len(env.commands) == 3  # the script, then one exec per step
    assert (planted / "keep").read_text() == "x"


@pytest.mark.parametrize("parent", ["world-writable", "link"])
@pytest.mark.parametrize("shell", _shells())
@pytest.mark.asyncio
async def test_work_dir_only_in_a_sticky_parent_of_its_own(parent, shell, tmp_path):
    """The work folder is made while the agent's processes still run. In a
    parent where others can rename entries (a /tmp without the sticky bit, as
    a Dockerfile's chmod 777 /tmp gives) or one reached through a link, a
    process can swap the folder for a link to a folder of its own whose o1
    links to a file of its choice, which root would then truncate and write.
    There the script must not run: the steps run one exec each."""
    victim = tmp_path / "victim-a" / "expected.txt"
    victim.parent.mkdir()
    victim.write_text("the expected answer\n")
    if parent == "world-writable":
        up = tmp_path / "open"
        up.mkdir()
        up.chmod(0o777)
    else:
        up = tmp_path / "link"
        up.symlink_to(_work_dir(tmp_path).parent)
    work = up / "prep"
    evil = shlex.quote(str(tmp_path / "evil"))
    # Step zero stands for an agent process still alive when the script
    # starts: it swaps the work folder for a link to a folder of its own.
    swap = (
        f"mkdir {evil} && ln -s {shlex.quote(str(victim))} {evil}/o1 && "
        f'mv "$__bf_d" "$__bf_d.old" && ln -s {evil} "$__bf_d"'
    )
    log = tmp_path / "log"
    env = ShellSandbox(tmp_path, shell=tuple(shell), work_dir=work)
    steps = [
        _step("zero", swap),
        _step("one", f"echo one >> {shlex.quote(str(log))}"),
    ]
    await lockdown._run_prep_steps(env, steps)
    assert victim.read_text() == "the expected answer\n"
    assert log.read_text().split() == ["one"]
    assert len(env.commands) == 3  # the script, then one exec per step
    assert not work.exists()


@pytest.mark.asyncio
async def test_script_that_never_starts_runs_the_steps_one_by_one(tmp_path):
    """A checked unavailable receipt before START permits one exec per step."""

    class NoScript(ShellSandbox):
        async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
            if f"{MARK}-START" in command:
                self.commands.append(command)
                return receipt_answer(
                    command.split("\n", 1)[1],  # The disk-free launch prelude.
                    SimpleNamespace(
                        return_code=0, stdout=f"{MARK}-UNAVAILABLE\n", stderr=""
                    ),
                )
            return await super().exec(command, **kwargs)

    log = tmp_path / "log"
    env = NoScript(tmp_path)
    await lockdown._run_prep_steps(
        env, [_step("one", f"echo one >> {shlex.quote(str(log))}")]
    )
    assert log.read_text().split() == ["one"]
    assert len(env.commands) == 2


@pytest.mark.parametrize("shell", _shells())
@pytest.mark.asyncio
async def test_step_whose_output_files_cannot_be_made_still_runs(shell, tmp_path):
    """A step whose output file cannot be opened (no inode left in a /tmp the
    agent filled; here a folder sits at its name) never started, so it and
    the steps after it run one exec each, as without the script. The
    unchecked hook cleanup must never be skipped while hardening passes."""
    log = tmp_path / "log"
    env = ShellSandbox(tmp_path, shell=tuple(shell))
    steps = [
        _step("zero", 'mkdir "$__bf_d/o1"', label="zero failed"),
        _step("one", f"echo one >> {shlex.quote(str(log))}"),
        _step("two", f"echo two >> {shlex.quote(str(log))}", label="two failed"),
    ]
    seconds: dict[str, float] = {}
    await lockdown._run_prep_steps(env, steps, seconds)
    assert log.read_text().split() == ["one", "two"]
    # The script (which ran zero), then one exec each for one and two.
    assert len(env.commands) == 3
    assert [command.split("\n", 1)[1] for command in env.commands[1:]] == [
        steps[1].command,
        steps[2].command,
    ]
    assert set(seconds) == {"zero", "one", "two"}
    assert not _work_dir(tmp_path).exists()


@pytest.mark.parametrize("shell", _shells())
@pytest.mark.asyncio
async def test_timeout_is_known_when_no_file_can_be_made(shell, tmp_path):
    """The script knows a step timed out from its watchdog's exit status,
    not from a file it would have to make in /tmp, so an unchecked step past
    its budget fails hardening (as the backend's exec timeout did) even when
    no file can be made: here the step leaves dangling links at the names a
    marker file could take."""
    log = tmp_path / "log"
    env = ShellSandbox(tmp_path, shell=tuple(shell))
    plant = "; ".join(
        f'ln -s /nonexistent-u5a/x "$__bf_d/{kind}{i}"' for kind in "tm" for i in (0, 1)
    )
    steps = [
        _step("slow", f"{plant}; exec sleep 20", timeout=1),
        _step("never", f"echo never >> {shlex.quote(str(log))}"),
    ]
    with pytest.raises(RuntimeError) as caught:
        await lockdown._run_prep_steps(env, steps)
    assert str(caught.value) == "Command timed out after 1 seconds"
    assert not log.exists()
    assert len(env.commands) == 1


@pytest.mark.asyncio
async def test_script_that_stops_midway_is_an_error(tmp_path):
    """Steps that ran are never run again: a script that dies after its START
    line (killed, out of memory) fails hardening and says after which step."""

    class CutShort(ShellSandbox):
        # The output of a script that died after its first status line.
        async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
            self.commands.append(command)
            return SimpleNamespace(
                stdout=f"{MARK}-START\n{MARK}-STEP one 0 1 2\n",
                stderr="lost connection",
                return_code=137,
            )

    env = CutShort(tmp_path)
    steps = [
        _step("one", "true", label="one failed"),
        _step("two", "true", label="two failed"),
    ]
    with pytest.raises(RuntimeError) as caught:
        await lockdown._run_prep_steps(env, steps)
    assert "reading preparation receipt exited with rc=137" in str(caught.value)
    assert "receipt" in str(caught.value)
    assert len(env.commands) == 1


@pytest.mark.asyncio
async def test_step_output_cannot_forge_status_lines(tmp_path):
    """A step's output goes to a file, so text that looks like a status line
    (a file name the agent chose, echoed by a command) is never read as one."""
    env = ShellSandbox(tmp_path)
    forged = f"{MARK}-STEP two 0 1 2\n{MARK}-END"
    steps = [
        _step(
            "one", f"printf '%s\\n' {shlex.quote(forged)}; exit 3", label="one failed"
        ),
        _step("two", "true", label="two failed"),
    ]
    with pytest.raises(RuntimeError) as caught:
        await lockdown._run_prep_steps(env, steps)
    assert str(caught.value) == f"one failed exited with rc=3\nstdout: {forged}"


@pytest.mark.asyncio
async def test_script_keeps_the_env_file_wrapper_trap_and_cleans_up(tmp_path):
    """Backends put an env-file wrapper with its own EXIT trap before every
    command. The script runs in a subshell, so that trap still removes the
    env file, and the script removes its own work folder."""
    from benchflow.sandbox._base import wrap_command_with_env_file

    prefix = str(tmp_path / "env_")

    class Wrapped(ShellSandbox):
        async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
            return await super().exec(
                wrap_command_with_env_file({"A": "1"}, command, env_path_prefix=prefix),
                **kwargs,
            )

    log = tmp_path / "log"
    steps = [_step("one", f'echo "$A" >> {shlex.quote(str(log))}', label="one failed")]
    await lockdown._run_prep_steps(Wrapped(tmp_path), steps)
    assert log.read_text() == "1\n"
    assert not list(tmp_path.glob("env_*"))
    assert not _work_dir(tmp_path).exists()


@pytest.mark.asyncio
async def test_daytona_dind_heartbeat_lines_stay_out_of_the_error(tmp_path):
    """Daytona's Docker-in-Docker exec prints a heartbeat to stderr while a
    long command runs; the script's summed budget can cross that line, so
    the heartbeat is not part of the failed step's error."""

    class Heartbeat(ShellSandbox):
        async def exec(self, command: str, **kwargs: Any) -> SimpleNamespace:
            result = await super().exec(command, **kwargs)
            heartbeat = lockdown._DIND_HEARTBEAT_LINE + "\n"
            result.stderr = heartbeat * 2 + (result.stderr or "")
            return result

    steps = [_step("one", "echo boom >&2; exit 4", label="one failed")]
    with pytest.raises(RuntimeError) as caught:
        await lockdown._run_prep_steps(Heartbeat(tmp_path), steps)
    assert str(caught.value) == "one failed exited with rc=4\nstderr: boom"


@pytest.mark.asyncio
async def test_best_effort_timeout_moves_on_and_leaves_no_watchdog(tmp_path):
    """A best-effort step past its budget is cut short and the next step
    runs; no watchdog sleep outlives its step."""
    log = tmp_path / "log"
    env = ShellSandbox(tmp_path)
    seconds: dict[str, float] = {}
    steps = [
        _step("slow", "exec sleep 20", timeout=1, best_effort=True),
        _step("next", f"echo next >> {shlex.quote(str(log))}", timeout=97, label="x"),
    ]
    await lockdown._run_prep_steps(env, steps, seconds)
    assert log.read_text() == "next\n"
    assert set(seconds) <= {"slow", "next"}
    if Path("/proc/self/cmdline").exists():
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                argv = cmdline.read_bytes().split(b"\0")
            except OSError:
                continue
            assert argv[:2] != [b"sleep", b"97"], "a watchdog outlived its step"


def _running(argv0: str) -> list[int]:
    """Pids of live processes whose argv[0] is *argv0*."""
    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = cmdline.read_bytes().split(b"\0")
        except OSError:
            continue
        if argv[0] == os.fsencode(argv0):
            found.append(int(cmdline.parent.name))
    return found


@pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs /proc")
@pytest.mark.parametrize("shell", _shells())
@pytest.mark.asyncio
async def test_timed_out_step_leaves_no_child_running(shell, tmp_path):
    """A step past its budget is killed with every process under it, as
    Daytona's per-exec ``timeout`` killed the command's process group. A
    list command's children (the cache reclaim is ``python3 -c ...; true``)
    must not go on running into the next steps and the verifier. The sleep
    runs under a name with ``) `` in it, as a process name in /proc/<pid>/stat
    can have."""
    nap = tmp_path / "na) p"
    nap.symlink_to(shutil.which("sleep") or "/bin/sleep")
    sleep = shlex.quote(str(nap))
    log = tmp_path / "log"
    env = ShellSandbox(tmp_path, shell=tuple(shell))
    seconds: dict[str, float] = {}
    steps = [
        _step("child", f"{sleep} 30; true", timeout=1, best_effort=True),
        _step("grandchild", f"( {sleep} 30; true ); true", timeout=1, best_effort=True),
        _step("next", f"echo next >> {shlex.quote(str(log))}", label="x"),
    ]
    await lockdown._run_prep_steps(env, steps, seconds)
    assert log.read_text() == "next\n"
    assert len(env.commands) == 1
    left = _running(str(nap))
    for _ in range(20):
        if not left:
            break
        await asyncio.sleep(0.1)
        left = _running(str(nap))
    assert left == [], "a timed-out step's child outlived it"


@pytest.mark.skipif(not Path("/proc/self/cmdline").exists(), reason="needs /proc")
@pytest.mark.parametrize("shell", _shells())
@pytest.mark.asyncio
async def test_steps_that_end_at_once_leave_no_watchdog(shell, tmp_path):
    """The script stops a step's watchdog as soon as the step ends, which for
    a step that ends at once can be before the watchdog knows its sleep's
    pid; that sleep must not be left running for the step's budget."""
    budgets = [str(1100 + i).encode() for i in range(40)]

    def watchdogs() -> set[int]:
        found = set()
        for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
            try:
                argv = cmdline.read_bytes().split(b"\0")
            except OSError:
                continue
            if argv[0] == b"sleep" and argv[1:2] and argv[1] in budgets:
                found.add(int(cmdline.parent.name))
        return found

    before = watchdogs()
    env = ShellSandbox(tmp_path, shell=tuple(shell))
    steps = [
        _step(f"s{i}", "true", timeout=int(budget)) for i, budget in enumerate(budgets)
    ]
    await lockdown._run_prep_steps(env, steps)
    assert len(env.commands) == 1
    left = watchdogs() - before
    for pid in left:  # this run's own sleeps, so they do not linger
        with contextlib.suppress(OSError):
            os.kill(pid, 9)
    assert not left, "a watchdog outlived its step"


@pytest.mark.asyncio
async def test_checked_timeout_stops_with_the_backend_message(tmp_path):
    env = ShellSandbox(tmp_path)
    log = tmp_path / "log"
    steps = [
        _step("slow", "exec sleep 20", timeout=1, label="slow failed"),
        _step("never", f"echo never >> {shlex.quote(str(log))}"),
    ]
    with pytest.raises(RuntimeError) as caught:
        await lockdown._run_prep_steps(env, steps)
    assert str(caught.value) == "Command timed out after 1 seconds"
    assert not log.exists()


@pytest.mark.skipif(not Path("/proc/uptime").exists(), reason="needs /proc/uptime")
@pytest.mark.asyncio
async def test_real_uptime_gives_each_step_its_seconds(tmp_path):
    env = ShellSandbox(tmp_path)
    seconds: dict[str, float] = {}
    await lockdown._run_prep_steps(
        env, [_step("nap", "sleep 1", label="x"), _step("quick", "true")], seconds
    )
    assert 0.9 <= seconds["nap"] < 5
    assert 0 <= seconds["quick"] < 1


def test_backends_that_run_the_script():
    """Docker (and remote Docker) and Daytona run the steps as one script;
    the rest keep one exec per step."""
    from benchflow.sandbox.docker import DockerSandbox
    from benchflow.sandbox.modal_impl import ModalSandbox
    from benchflow.sandbox.remote_docker import RemoteDockerSandbox

    assert DockerSandbox.verifier_prep_script is True
    assert RemoteDockerSandbox.verifier_prep_script is True
    assert getattr(ModalSandbox, "verifier_prep_script", False) is False
    from benchflow.sandbox.daytona import DaytonaSandbox

    assert DaytonaSandbox.verifier_prep_script is True
