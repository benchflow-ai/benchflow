"""Verifier ($V$) — maps agent completion to a reward signal.

Supports two verification methods,
selected by ``[verifier].type`` in ``task.toml``:

- ``"test-script"`` (default): run ``tests/test.sh`` inside the sandbox and
  parse ``reward.txt`` / ``reward.json``.
- ``"llm-judge"``: download the agent's deliverables and grade them against a
  human-authored rubric using an LLM judge (see #270).

The ``Verifier`` class is kept whole here; its self-coupled ``_verify_*`` /
``_parse_*`` methods share ``_task`` / ``_rollout_paths`` / ``_sandbox`` /
``_logger`` state. The pure free-function leaf clusters live in sibling
``verifier_*`` modules and are imported below.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import functools
import json
import logging
import math
import os
import shlex
import stat
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from benchflow._runtime_tree import KIT_BIN, RuntimeKit
from benchflow._utils.command_hints import command_arg
from benchflow._utils.reachable_write import (
    PrivateStorageError,
    ReachablePath,
    ReachableWriteError,
    UnsafeDownloadEntryError,
    destination,
    private_folder,
    private_temporary_directory,
)
from benchflow._utils.scoring import VERIFIER_OUTPUT_UNWRITABLE
from benchflow._utils.text import describe_exception
from benchflow._verifier_env import (
    VERIFIER_ATTEMPT_ENV,
    pinned_shell_prefix,
    sandbox_verifier_env,
)
from benchflow.rewards.validation import (
    RewardOutOfRange,
    apply_aggregate_policy,
    declared_reward_range,
    is_out_of_range_reward,
    is_valid_reward_number,
    keeps_out_of_range_reward,
    reward_lenient_from_env,
    reward_range_phrase,
    validate_reward_map,
)
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from benchflow.sandbox._guard_reach import PytestReach
from benchflow.sandbox._guard_reach import task_reach as pytest_reach_of_task
from benchflow.sandbox._guard_rejudge import nominate as nominate_guard_rejudgment
from benchflow.sandbox._guard_verdicts import agents_rejection
from benchflow.sandbox._guard_verdicts import read_verdicts as read_guard_verdicts
from benchflow.sandbox._guard_verdicts import unconfirmed as unconfirmed_guard_verdicts
from benchflow.sandbox._task_route import as_task_command
from benchflow.sandbox._transport import READ_FAILED
from benchflow.sandbox.daytona_exec_env import remove_exec_env_files
from benchflow.sandbox.lockdown import (
    GuardMarker,
    _exec_return_code,
    clear_verifier_output_dir,
    describe_installed_marker,
    installed_marker_plugins,
    parse_pytest_plugin_guard_markers,
    pytest_plugin_guard_markers_cmd,
    pytest_plugin_guard_markers_dir,
    pytest_plugin_guard_markers_problem,
    pytest_plugin_guard_name,
    with_verifier_umask,
)
from benchflow.sandbox.protocol import ExecTimeoutError
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths
from benchflow.task.verifier_criteria_runner import (
    _criteria_runner_criteria,
    _criteria_runner_criteria_policy,
    _criteria_runner_manifest_json,
    _criteria_runner_root,
    _criteria_runner_runner,
    _llm_judge_input_dir,
    _safe_strategy_relative_path,
)
from benchflow.task.verifier_document import (
    VerifierDocument,
    VerifierStrategy,
    load_verifier_document,
)
from benchflow.task.verifier_errors import (
    AddTestsDirError,
    AgentJudgeInputError,
    DownloadVerifierDirError,
    ORSEpisodeInputError,
    PluginGuardLoadError,
    PluginGuardMarkersUnreadableError,
    PluginGuardRejectedError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    RewardFileParseError,
    RewardOutOfRangeError,
    RubricNotFoundError,
    UnsupportedVerifierStrategyError,
    VerifierExecutionError,
    VerifierInfrastructureError,
    VerifierLaunchRefusedError,
    VerifierOutputParseError,
    VerifierResult,
    VerifierSetupError,
    verifier_failure_reason,
    verifier_infrastructure_failure,
)
from benchflow.task.verifier_judge_inputs import (
    _agent_judge_score,
    _local_rollout_input_path,
    _read_agent_judge_input,
    _safe_input_filename,
)
from benchflow.task.verifier_ors_episode import (
    _count_ors_episode_records,
    _load_ors_episode_records,
    _ors_records_to_verify_result,
)
from benchflow.task.verifier_scan import (
    _DEP_INSTALL_DIAGNOSTIC,
    _has_dep_install_failure,
    _has_guard_load_failure,
)
from benchflow.task.verifier_script_strategy import (
    _has_aggregate_declaration,
    _script_strategy_chmod_command,
    _script_strategy_command,
    script_run_command,
)
from benchflow.trust.state import TRUSTED_TRANSFER, admit

logger = logging.getLogger(__name__)

# Where the test command records that it started (#1136): a BenchFlow-owned
# directory, never the verifier output dir test.sh inspects.
_EXECUTION_RECEIPT_DIR = PurePosixPath("/run/runtime")
# At most this many of test.sh's outputs are made readable to the host before
# the at-exit copy (:meth:`Verifier._let_host_read_outputs`), the names the
# host reads to score first; the shell route takes them this many at a time.
_HOST_READ_MAX = 512
_HOST_READ_CHUNK = 64

# The verifier's setup commands (clearing the output dir, chmod, the target
# verifier dir, the criteria-runner manifest) run after the agent has finished. On
# a slow or stalled exec layer (a loaded Docker host, a Daytona API stall) a
# 10 s exec failed them, and the finished rollout's verification with them;
# #1043 made the publish-path execs non-fatal for the same reason (#948). Each
# setup command now gets 60 s, and a failed one two more tries.
_SETUP_FAILED = "Verifier setup failed:"
# The answers of the check before the test command.
_STDOUT_WRITABLE = "BF stdout-writable"
_STDOUT_UNWRITABLE = "BF stdout-unwritable"


def stdout_probe_command(test_stdout_path: str) -> str:
    """Open *test_stdout_path* (already quoted) for appending, writing
    nothing, and say whether that worked. ``true`` in a subshell: a refused
    redirection on ``:``, a special built-in, would end the shell itself
    (exit 2) before it could answer."""
    return (
        f"if ( true >> {test_stdout_path} ) 2>/dev/null; "
        f"then echo {_STDOUT_WRITABLE}; else echo {_STDOUT_UNWRITABLE}; fi"
    )


_SETUP_EXEC_TIMEOUT_SEC = 60
# The start-receipt folder (see Verifier._prepare_execution_receipt).
_RECEIPT_DIR_COMMAND = (
    f"mkdir -p {shlex.quote(str(_EXECUTION_RECEIPT_DIR))} && "
    f"chmod 1777 {shlex.quote(str(_EXECUTION_RECEIPT_DIR))}"
)
_RECEIPT_DIR_BUDGET_SEC = 10


def _root_receipt_dir_command(folder: PurePosixPath) -> str:
    """The receipt folder of one attempt of the test command.

    The agent's processes can run beside the verifier (kept services, and
    every process of the agent's in integrity off and audit), and
    can create names in the 1777 folder, the receipt's ``.exit`` status
    included: an ``.exit`` "0" written first for tests still
    running at the deadline had them scored from whatever ``verifier/``
    held. So the receipt and its ``.exit`` and ``.abandoned`` go in a new
    folder only root can write, in a parent that is root's, sticky and not a
    link (no one else can move the folder out of it); otherwise the command
    fails before changing anything, and startup stays unknown, as without a
    receipt folder.
    """
    parent = shlex.quote(str(folder.parent))
    return (
        f"[ ! -L {parent} ] && mkdir -p {parent} && [ -O {parent} ] && "
        f"chmod 1777 {parent} && mkdir -m 0755 {shlex.quote(str(folder))}"
    )


def _receipt_command(
    sandbox: Any, folder: PurePosixPath, root_only: bool, service: str
) -> str:
    """The receipt folder's root command: :data:`_RECEIPT_DIR_COMMAND` or
    :func:`_root_receipt_dir_command`, made with bfx's mkdir and chmod where
    bfx is delivered, so no program of the image runs for it after the
    agent's turn (bfx follows no link, which ``[ ! -L ]`` checked)."""
    from benchflow.sandbox._bounded_exec import bfx_delivered, bfx_sh

    if service != "main" or not bfx_delivered(sandbox):
        return _root_receipt_dir_command(folder) if root_only else _RECEIPT_DIR_COMMAND
    if not root_only:
        top = str(_EXECUTION_RECEIPT_DIR)
        return (
            f"{bfx_sh('mkdir', top, parents=True, mode=0o1777)} && "
            f"{bfx_sh('chmod', top, mode=0o1777)}"
        )
    parent = str(folder.parent)
    return (
        f"{bfx_sh('mkdir', parent, parents=True, mode=0o1777)} && "
        f"[ -O {shlex.quote(parent)} ] && {bfx_sh('chmod', parent, mode=0o1777)} && "
        f"{bfx_sh('mkdir', str(folder), mode=0o755)}"
    )


_SETUP_RETRY_DELAYS_SEC = (5.0, 15.0)
# A read of the plugin guard's markers that did not come back whole is tried
# again after these waits (Verifier._plugin_guard_markers).
_GUARD_MARKER_RETRY_DELAYS_SEC = (2.0, 5.0)
# The uid the verifier's commands run as, with shell builtins only: the shell
# reads its own status file (no program on a PATH, no task env in this exec).
_VERIFIER_UID_CMD = (
    'while read -r k v r; do case $k in Uid:) echo "$v"; exit 0;; esac; '
    "done < /proc/self/status; exit 1"
)


async def user_uid(sandbox: Any, user: str, *, service: str = "main") -> str | None:
    """*user*'s uid in *sandbox*, read by a shell started as that user (its
    own /proc/self/status); None when it can't be read. Before the turn, for
    RB2: two names for one uid are one user."""
    kwargs = (
        {"system_shell": True}
        if getattr(sandbox, "supports_system_shell", False) is True
        else {}
    )
    try:
        result = await exec_raw(
            sandbox.exec,
            _VERIFIER_UID_CMD,
            user=user,
            service=service,
            timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            **kwargs,
        )
    except Exception:
        return None
    uid = (getattr(result, "stdout", "") or "").strip()
    return uid if _exec_return_code(result) == 0 and uid.isdigit() else None


# What the plugin guard's ``unrecorded`` marker says on its first line
# (``_pytest_plugin_guard._unrecorded``).
_UNRECORDED_WHY = ("none was taken", "the record did not finish")
_REWARD_FILE_MAX_BYTES = 16 * 1024 * 1024


def _hardened_network(sandbox: Any) -> bool:
    """Whether hardening armed the verifier's network on *sandbox*."""
    from benchflow.sandbox.verifier_network import VerifierNetwork

    armed = getattr(sandbox, "_benchflow_verifier_network", None)
    return isinstance(armed, VerifierNetwork) and armed.launch


def _strings(value: Any) -> tuple[str, ...]:
    return (
        tuple(v for v in value if isinstance(v, str))
        if isinstance(value, (tuple, list))
        else ()
    )


def _addopts_value(addopts: str | None, option: str) -> str:
    """The last ``option=VALUE`` in BenchFlow's PYTEST_ADDOPTS, ``""``
    without one."""
    try:
        words = shlex.split(addopts or "")
    except ValueError:
        return ""
    found = ""
    for word in words:
        if word.startswith(option + "="):
            found = word[len(option) + 1 :]
    return found


def _pytest_collects_from(
    reach: PytestReach | None, ini: Any, addopts: str | None
) -> tuple[tuple[str, ...], str]:
    """Where the trial's pytest collects from, by the host's facts: the
    paths test.sh gives it, and for a pytest given none, the folder it runs
    in, or the ``testpaths`` of strict's ``-c`` ini when that folder is the
    rootdir; with BenchFlow's ``--confcutdir`` (R334's delta)."""
    if reach is None:
        return (), ""
    rootdir = _addopts_value(addopts, "--rootdir").rstrip("/")
    testpaths = _strings(getattr(ini, "testpaths", ()))
    anchors = list(reach.paths)
    for folder in reach.bare:
        if not folder:
            continue
        if testpaths and folder.rstrip("/") == rootdir:
            anchors.extend(testpaths)
        else:
            anchors.append(folder)
    return tuple(dict.fromkeys(anchors)), _addopts_value(addopts, "--confcutdir")


def _passes(rewards: dict[str, Any] | None) -> bool:
    """Whether *rewards* give the trial anything above 0."""
    return any(
        isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0
        for value in (rewards or {}).values()
    )


def _guard_end_problem(
    guard: str, markers: dict[str, list[GuardMarker]], reach: PytestReach | None
) -> str | None:
    """Why the plugin guard's end-of-session records do not back a pass, or
    None ((g2)).

    Every pytest the guard loads in leaves a ``finished`` record at pytest's
    own last cleanup, with its exit status and counts, and takes its
    ``registered`` marker away; one left is a pytest that ended before then
    (an exit mid-collection, a kill). A record of exit status 0 that counts a
    failed test, or none collected, is not the status pytest gives for its
    counts. And where test.sh starts only pytests the guard reaches
    (sandbox/_guard_reach.py), a pass needs a session that passed a test.
    The records are written inside the test process: they stop the
    accidental case and the mid-collection exit, not a forgery (the
    lead's 15:41Z ruling).
    """
    if markers.get("registered"):
        return (
            f"a pytest the plugin guard {guard} loaded in ended before pytest's "
            "own cleanup and left no end-of-session record"
        )
    finished = markers.get("finished", [])
    if not finished:
        return (
            f"no pytest the plugin guard {guard} loaded in left its "
            "end-of-session record"
        )
    sessions = []
    for marker in finished:
        try:
            record = json.loads(marker.detail)
        except ValueError:
            record = None
        if not isinstance(record, dict):
            return (
                f"the plugin guard's end-of-session record verifier/{marker.name} "
                "cannot be read"
            )
        if record.get("session") is True:
            sessions.append((marker.name, record))
    for name, record in sessions:
        failed, collected = record.get("failed"), record.get("collected")
        if record.get("exitstatus") == 0 and not (
            isinstance(failed, int)
            and failed == 0
            and isinstance(collected, int)
            and collected > 0
        ):
            return (
                f"pytest's end-of-session record verifier/{name} gives exit "
                f"status 0 with {failed} failed of {collected} collected, which "
                "is not the status pytest gives for those counts"
            )
    if (
        reach is not None
        and reach.expects_record
        and not any(
            # None: a pytest with no terminal reporter, which cannot count.
            record.get("passed") is None
            or (isinstance(record["passed"], int) and record["passed"] > 0)
            for _, record in sessions
        )
    ):
        return (
            "no pytest session of this run passed a test, by the plugin guard's "
            "end-of-session records"
        )
    return None


def _marker_tokens(markers: list[GuardMarker]) -> set[str]:
    """The pytest processes (``<guard>.<pid>-<random>``) that left *markers*."""
    return {marker.name.rpartition(".")[0] for marker in markers}


# Where the verifier's env file goes while services are kept: a folder only
# root can enter, never /tmp. A kept service that can write a folder's parent
# can rename root's folder away and put its own in its place before root
# sources it, and /tmp is writable by everyone (without its sticky bit, as
# some images leave it, that rename is allowed: K2-R6-A-1). So /, /var and
# /var/lib must be root's real folders with no group or other write, and
# /var/lib/runtime (made root's 0700 before the agent user exists) and this
# folder are made root's 0700; anything else fails the step.
_VERIFIER_ENV_PARENT = "/var/lib/runtime/verifier-env"
_VERIFIER_ENV_PARENT_CMD = (
    "for d in / /var /var/lib; do "
    '[ -d "$d" ] && [ ! -h "$d" ] && [ "$(stat -c %u "$d")" = 0 ] && '
    'case "$(stat -c %a "$d")" in *[2367]?|*[2367]) false ;; esac || '
    '{ echo "Unsafe folder for the verifier env: $d" >&2; exit 1; }; done; '
    f"for d in /var/lib/runtime {_VERIFIER_ENV_PARENT}; do "
    '[ ! -h "$d" ] || { echo "Unsafe folder for the verifier env: $d" >&2; '
    "exit 1; }; done; "
    f"umask 077 && mkdir -p {_VERIFIER_ENV_PARENT} && "
    f"for d in /var/lib/runtime {_VERIFIER_ENV_PARENT}; do "
    '[ ! -h "$d" ] && chown 0:0 "$d" && chmod 700 "$d" || '
    '{ echo "Unsafe folder for the verifier env: $d" >&2; exit 1; }; done'
)


def _root_file_env_prefix(folder: str) -> str:
    """What sets the env from ``<folder>/env`` before the verifier's command.

    No trap: an EXIT trap here would replace one the command sets
    (K2-R5-A-3). The env's last line clears the pending parameter, so a file
    that was not sourced to its end stops the command; ``rm`` is the
    system's, by path.
    """
    return (
        f"set -- benchflow-env-pending; if . {folder}/env; then "
        f"/bin/rm -rf {folder}; else /bin/rm -rf {folder}; exit 1; fi; "
        "case $# in 0) ;; *) exit 1;; esac\n"
    )


def _set_aside(path: ReachablePath) -> None:
    """Rename a reward file a timed-out test command left to ``timeout-<name>``.

    The test command's code could have put something in the way (a folder of
    that name); then the file takes a name of its own, or goes, so that no
    reward stays under its real name beside a score of 0.
    """
    for name in (f"timeout-{path.name}", f"timeout-{uuid4().hex}-{path.name}"):
        try:
            path.rename_entry(name)
            return
        except FileNotFoundError:
            return
        except OSError:
            continue
    path.unlink(missing_ok=True)


# Printed by the owner rule's command once the outputs folder is root's
# (Verifier._close_output_dir).
_OUTPUTS_CLOSED = "benchflow-outputs-folder-closed"
# Each top-most entry another user owns (-prune: one command per foreign
# subtree, however many files its writer makes) is taken from it before it
# goes: root's, not
# writable by others, then removed. A process of that user still writing
# there (the agent's, alive through grading in integrity audit and off) can
# then no longer keep the removal from finishing. BusyBox's tools take every
# flag here.
_SWEEP_FOREIGN = (
    'for e; do if [ -L "$e" ]; then rm -f -- "$e" || exit 1; continue; fi; '
    '[ -e "$e" ] || continue; '
    'chown -R -h 0:0 -- "$e" && chmod -R go-w -- "$e" && rm -rf -- "$e" '
    "|| exit 1; done"
)
# Why the outputs folder could not be closed: only another user's process can
# keep the sweep from finishing once the folder is root's. The trial's
# verifier failed (verifier_failed), never infrastructure, never retried.
_OUTPUTS_HELD = (
    "agent processes kept writing in /logs/verifier while it was being closed"
)


class VerifierOutputsHeldError(RuntimeError):
    """Another user's processes kept writing in the outputs folder while
    the owner rule closed it (:data:`_OUTPUTS_HELD`)."""


def _guard_nomination(
    task: Any, sandbox: Any, guard: str, pytest_reach: Any, pytest_addopts: Any
) -> Any:
    """What the pristine re-judgment of a stop judges (R): the host's own
    facts, never the trial's record (sandbox/_guard_rejudge.py). The plugins
    test.sh's pytest commands name with ``-p`` and the ``-c`` ini's, less the
    ones test.sh turns off with ``-p no:``, beside the task's own."""
    pythons = getattr(sandbox, "_benchflow_guard_pythons", ())
    ini = getattr(sandbox, "_benchflow_pytest_ini", None)
    anchors, confcutdir = _pytest_collects_from(pytest_reach, ini, pytest_addopts)
    return nominate_guard_rejudgment(
        getattr(getattr(task, "paths", None), "tests_dir", None),
        pythons if isinstance(pythons, (tuple, list)) else (),
        _task_addopts(task),
        guard,
        (
            *(pytest_reach.plugins if pytest_reach is not None else ()),
            *_strings(getattr(ini, "plugins", ())),
        ),
        pytest_reach.disabled if pytest_reach is not None else (),
        anchors,
        confcutdir,
    )


def _task_addopts(task: Any) -> str:
    """The task's own ``[verifier] env`` PYTEST_ADDOPTS (its config, on the
    host), or "": the plugins it names are put to the pristine guard (
    (g2) R)."""
    env = getattr(getattr(getattr(task, "config", None), "verifier", None), "env", None)
    value = env.get("PYTEST_ADDOPTS") if isinstance(env, dict) else None
    return value if isinstance(value, str) else ""


class Verifier:
    """Runs the task's verifier and parses rewards.

    Two verification methods are supported (selected by ``verifier.type``):

    1. ``test-script`` — uploads the task's ``tests/`` directory into the
       sandbox at ``/tests``, runs ``test.sh`` with configured env vars/user,
       and parses the reward from ``reward.txt`` or ``reward.json``.
    2. ``llm-judge`` — downloads the agent's deliverables from the sandbox and
       grades them against a rubric using an LLM judge, writing the aggregate
       reward to ``reward.json``.
    """

    # What a Verifier built without __init__ reads: no copy of test.sh's
    # outputs taken at its exit, no outputs folder closed to the host.
    outputs_at_exit: dict[str, Any] | None = None
    _closed_to_host = False

    # The verifier user's uid, once read (:meth:`_verifier_uid`).
    _verifier_uid_read: str | None = None
    # Whether test.sh's output folder was closed to the agent while it ran
    # (:meth:`_close_output_dir`; None: no close was asked), and why not.
    outputs_closed: bool | None = None
    outputs_open_why: str | None = None
    # Whether a close was due at all: a verify that runs a test command asks
    # for one (even where it then returns unset, kept services); a verify with
    # none (llm-judge, agent-judge, an ORS episode) grades a reward test.sh
    # never wrote, and the close is no check of it.
    outputs_close_due: bool = False
    # What the read before the copy at test.sh's exit granted, skipped and
    # left out (:meth:`_let_host_read_outputs`; None: nothing to grant).
    read_grant: dict[str, Any] | None = None
    # Each workspace the grant made sticky, with the mode the step before a
    # root test.sh left it (:meth:`_check_sticky_closed`; None: no such step).
    workspace_modes_before_test: dict[str, str | None] | None = None

    def __init__(
        self,
        task: Any,
        rollout_paths: RolloutPaths,
        sandbox: Any,
        _logger: logging.Logger | None = None,
        *,
        execution_receipt: bool = False,
    ) -> None:
        self._task = task
        self._rollout_paths = rollout_paths
        self._sandbox = sandbox
        # Recovery-eligible rollouts (#1136) ask the test command to prove it
        # started; ``execution_receipt`` then names the proof for the prober.
        self._write_execution_receipt = execution_receipt
        self.execution_receipt: tuple[str, str] | None = None
        # Set when kept services run beside the verifier, whose verifier is
        # root: each attempt's receipt folder is then root's without the
        # uid the outputs folder's close reads. Every attempt's receipt is
        # in a folder of its own that the agent cannot write.
        self.receipt_root_only = False
        self._logger = (_logger or logger).getChild("verifier")
        # Task-declared ``[verifier] reward_range`` (BF-8); None keeps the
        # canonical strict [0, 1]. Applies to the test-script reward contract
        # (reward.txt / reward.json) — judge and ORS scores stay [0, 1].
        self._reward_range = declared_reward_range(task)
        # The last test script's exit status (``_verify_test_script``).
        self.test_return_code: int | None = None
        # Scores the issued test command's outputs once it has ended, for a
        # command whose exec never returned (finish_test_command).
        self._finish_test: Callable[[int], Awaitable[VerifierResult]] | None = None
        # Brings in the issued test command's outputs; kept for a command still
        # running at its deadline (save_timed_out_outputs).
        self._save_outputs: Callable[..., Awaitable[None]] | None = None
        # Host state, unlike the receipt test.sh's user can delete or forge.
        self.test_command_issued = False
        # The mounted outputs folder is root's until test.sh exits and the
        # host user cannot write there (_close_output_dir).
        self._closed_to_host = False
        # The service the issued test command runs in, for the network
        # verdict of one killed at its deadline (timed_out_network_verdict).
        self._issued_service: str | None = None
        # What the verifier's network refused the last test command, when it
        # refused anything (_end_of_test_proof): the trial's read flag and
        # result.json's ``verifier_network_refused``. Never a change of score.
        self.network_refusal: dict[str, Any] | None = None
        # Awaited with the verifier's code folder in the sandbox just before
        # the verifier's first command there (test.sh, a script strategy, a
        # criteria-runner runner, a task.md rubric's judge runner), after its
        # upload: the scoring path lets the services a task keeps
        # (x-keep-services) go on there, with that folder closed to them.
        self.before_test: Callable[[str], Awaitable[None]] | None = None
        # Soft verification sets this: the agent's own processes stay alive
        # beside every command of the verifier, so its env goes to them as
        # it does beside kept services (b156 PY8-MAIN-30).
        self.agent_processes_alive = False
        # Off and audit: the sandbox is held still from test.sh's exit until
        # its outputs are copied (Q3, verifier_freeze), recorded here.
        self.freeze_at_exit = False
        self.freeze_record: dict[str, Any] | None = None
        self._freeze_state: tuple[str, str] | None = None
        # test.sh's outputs at its exit (verifier_outputs) and what
        # BenchFlow itself wrote to verifier/ after them.
        self.outputs_at_exit: dict[str, Any] | None = None
        self._written_after_exit: dict[str, bytes] = {}
        # The BASH_ENV test.sh's shell may read: the value a trust check
        # validated (hardening's, recorded on the config; soft verification
        # sets its own, same policy), else none. The task's raw
        # [verifier.env] value never is (b143 PY8-MAIN-28).
        trusted = getattr(
            getattr(getattr(task, "config", None), "verifier", None),
            "_trusted_bash_env",
            None,
        )
        self.bash_env = trusted if isinstance(trusted, str) else ""

    @contextmanager
    def _empty_reward_named(self, runner: str, return_code: int) -> Iterator[None]:
        """Name an empty reward file for what left it so (DF-p2-task-author-11).

        ``test.sh exited 1 and wrote an empty verifier/reward.txt``; or, when
        the disk that holds the job folder is full (the sandbox writes the
        reward into the host's folder), that, with what is free (DF-p5-5).
        """
        try:
            yield
        except RewardFileEmptyError as e:
            from benchflow._utils.disk_space import (
                LOW_SPACE_BYTES,
                disk_full_text,
                free_bytes,
                size_text,
            )

            name = e.file or "the reward file"
            free = None
            if e.path is not None:
                with contextlib.suppress(Exception), e.path.open("rb") as held:
                    free = free_bytes(held)
            if free is not None and free < LOW_SPACE_BYTES:
                full = disk_full_text().rstrip()
                raise RewardFileEmptyError(
                    f"{name} is empty: {full} ({size_text(free)} free), so "
                    f"{runner} could not write its reward",
                    file=e.file,
                    path=e.path,
                ) from None
            exited = f" exited {return_code} and" if return_code else ""
            raise RewardFileEmptyError(
                f"{runner}{exited} wrote an empty {name}",
                file=e.file,
                path=e.path,
            ) from None
        except VerifierOutputParseError:
            raise
        except (OSError, ValueError, RecursionError, OverflowError) as e:
            raise RewardFileParseError(
                "Cannot read verifier/reward.txt or verifier/reward.json: "
                f"{type(e).__name__} (invalid contents or unsafe file)"
            ) from e

    def _read_reward_file(self, path: ReachablePath) -> str:
        # The verifier user can grow a reward file while it is being read.
        # Bound the read itself, not just a preceding stat (FIX-49 r4).
        with destination(path).open("rb") as stream:
            data = stream.read(_REWARD_FILE_MAX_BYTES + 1)
        if len(data) > _REWARD_FILE_MAX_BYTES:
            raise RewardFileParseError(
                f"verifier/{path.name} exceeds the 16 MiB reward file limit"
            )
        return data.decode("utf-8")

    def _parse_reward_text(self) -> dict[str, float | int]:
        if self._reward_file("reward.txt").stat().st_size == 0:
            raise RewardFileEmptyError(
                "verifier/reward.txt is empty",
                file="verifier/reward.txt",
                path=self._reward_file("reward.txt"),
            )
        try:
            reward = float(self._read_reward_file(self._reward_file("reward.txt")))
        except (ValueError, TypeError) as e:
            raise RewardFileParseError(
                "Failed to parse rewards from text file verifier/reward.txt"
            ) from e
        kept = is_out_of_range_reward(reward, reward_range=self._reward_range)
        if kept and not keeps_out_of_range_reward():
            raise RewardOutOfRangeError(
                RewardOutOfRange("reward", reward, self._reward_range),
                file="verifier/reward.txt",
            )
        if not kept and not is_valid_reward_number(
            reward, reward_range=self._reward_range
        ):
            raise RewardFileParseError(
                f"Reward text file verifier/reward.txt "
                "must contain a finite numeric reward "
                f"{reward_range_phrase(self._reward_range)}"
            )
        return {"reward": reward}

    def _parse_reward_json(
        self,
        *,
        aggregate_policy: dict[str, Any] | None = None,
        with_text: bool = False,
    ) -> dict[str, Any]:
        if self._reward_file("reward.json").stat().st_size == 0:
            raise RewardFileEmptyError(
                "verifier/reward.json is empty",
                file="verifier/reward.json",
                path=self._reward_file("reward.json"),
            )
        try:
            rewards = json.loads(
                self._read_reward_file(self._reward_file("reward.json"))
            )
        except (ValueError, TypeError) as e:
            raise RewardFileParseError(
                "Failed to parse rewards from JSON file verifier/reward.json"
            ) from e

        if not isinstance(rewards, dict):
            raise RewardFileParseError(
                "Reward JSON file verifier/reward.json "
                "must contain an object with numeric rewards"
            )
        if (
            with_text
            and "reward" not in rewards
            and not _has_aggregate_declaration(rewards, aggregate_policy)
        ):
            # The score is reward.txt's and reward.json holds the details
            # (#1192): checked as one map, a detail that is no number
            # ("answer": "Empowering") is kept beside the reward, as it is
            # beside reward.json's own (#1178), instead of failing the trial.
            rewards = {"reward": self._parse_reward_text()["reward"], **rewards}

        try:
            return validate_reward_map(
                rewards,
                source="reward JSON",
                aggregate_policy=aggregate_policy,
                lenient=reward_lenient_from_env(),
                reward_range=self._reward_range,
                keep_out_of_range_reward=keeps_out_of_range_reward(),
            )
        except RewardOutOfRange as e:
            error = RewardOutOfRangeError(e, file="verifier/reward.json")
            if e.key != "reward" and "." not in e.key:
                # Beside a reward such a number is a detail; with
                # none, an aggregate makes the reward from it.
                error.args = (
                    f"{error}; with no 'reward', a top-level number in "
                    "reward.json is a metric the reward is made from, held to "
                    "the reward range: write details that are not scores to "
                    "reward-details.json",
                )
            raise error from e
        except ValueError as e:
            raise RewardFileParseError(
                f"Reward JSON file verifier/reward.json {e}"
            ) from e

    def _parse_reward_json_with_text_compat(
        self,
        *,
        aggregate_policy: dict[str, Any] | None = None,
        strict_aggregate: bool = False,
    ) -> dict[str, Any]:
        rewards = self._parse_reward_json(
            aggregate_policy=aggregate_policy,
            with_text=not strict_aggregate and self._reward_file("reward.txt").exists(),
        )
        if strict_aggregate:
            try:
                rewards = apply_aggregate_policy(
                    rewards,
                    aggregate_policy=aggregate_policy,
                    source="reward JSON",
                    strict=True,
                    reward_range=self._reward_range,
                )
            except ValueError as e:
                raise RewardFileParseError(
                    f"Reward JSON file verifier/reward.json {e}"
                ) from e
            self._write_verifier_file(
                self._rollout_paths.reward_json_path,
                json.dumps(rewards, indent=2, allow_nan=False),
            )

        if not self._reward_file("reward.txt").exists():
            if "reward" not in rewards:
                try:
                    rewards = apply_aggregate_policy(
                        rewards,
                        aggregate_policy=aggregate_policy,
                        source="reward JSON",
                        reward_range=self._reward_range,
                    )
                except ValueError as e:
                    raise RewardFileParseError(
                        f"Reward JSON file verifier/reward.json {e}"
                    ) from e
                self._write_verifier_file(
                    self._rollout_paths.reward_json_path,
                    json.dumps(rewards, indent=2, allow_nan=False),
                )
            return rewards

        text_reward = self._parse_reward_text()["reward"]
        json_reward = rewards.get("reward")
        if json_reward is None:
            if _has_aggregate_declaration(rewards, aggregate_policy):
                try:
                    rewards = apply_aggregate_policy(
                        rewards,
                        aggregate_policy=aggregate_policy,
                        source="reward JSON",
                        reward_range=self._reward_range,
                    )
                except ValueError as e:
                    raise RewardFileParseError(
                        f"Reward JSON file verifier/reward.json {e}"
                    ) from e
                json_reward = rewards["reward"]
                if not math.isclose(
                    float(json_reward), float(text_reward), abs_tol=1e-9
                ):
                    raise RewardFileParseError(
                        f"Reward JSON file verifier/reward.json "
                        f"aggregates to reward={json_reward}, which does not match "
                        f"scalar reward.txt value {text_reward}"
                    )
                self._write_verifier_file(
                    self._rollout_paths.reward_json_path,
                    json.dumps(rewards, indent=2, allow_nan=False),
                )
                return rewards
            return {"reward": text_reward, **rewards}
        if not math.isclose(float(json_reward), float(text_reward), abs_tol=1e-9):
            raise RewardFileParseError(
                f"Reward JSON file verifier/reward.json "
                f"has reward={json_reward}, which does not match scalar "
                f"reward.txt value {text_reward}"
            )
        return rewards

    def _clear_reward_outputs(self) -> None:
        for path in (
            self._rollout_paths.reward_text_path,
            self._rollout_paths.reward_json_path,
            self._rollout_paths.reward_details_json_path,
            self._rollout_paths.criteria_runner_manifest_path,
        ):
            path.unlink(missing_ok=True)

    # The rollout's verifier folder is /logs/verifier in a Docker sandbox, and
    # an agent process can still be running while a judge answers: every file
    # BenchFlow writes or downloads there goes through reachable_write, never
    # a path the sandbox could have turned into a link (R4-1).

    def _write_verifier_file(self, path: ReachablePath, text: str) -> None:
        path.write(text)
        if self.outputs_at_exit is not None:
            # BenchFlow's own record after test.sh's exit: kept as written
            # when the folder is put back (verifier_outputs.put_back).
            try:
                rel = path.relative_to(self._rollout_paths.verifier_dir).as_posix()
            except ValueError:
                return
            self._written_after_exit[rel] = (
                text.encode() if isinstance(text, str) else text
            )

    def _reward_file(self, name: str) -> ReachablePath:
        """Where a reward file is read: test.sh's own copy at its exit once
        there is one (``verifier-at-exit/``), else ``verifier/``."""
        if self.outputs_at_exit is None or name in (
            self.outputs_at_exit.get("special") or ()
        ):
            if name == "reward.txt":
                return self._rollout_paths.reward_text_path
            return self._rollout_paths.reward_json_path
        from benchflow.task.verifier_outputs import AT_EXIT

        return (
            ReachablePath.directory(Path(self._rollout_paths.rollout_dir) / AT_EXIT)
            / name
        )

    def output_at_exit(self, name: str) -> Any:
        """test.sh's output *name* as it was at its exit
        (``verifier-at-exit/``) once that copy is taken:
        a name it does not hold was not there; one too large to copy is
        read from ``verifier/``, as is every name without a copy."""
        taken = self.outputs_at_exit
        entry = (taken.get("files") or {}).get(name) if taken else None
        if taken is None or (isinstance(entry, dict) and not entry.get("copied")):
            return self._rollout_paths.verifier_dir / name
        from benchflow.task.verifier_outputs import AT_EXIT

        return (
            ReachablePath.directory(Path(self._rollout_paths.rollout_dir) / AT_EXIT)
            / name
        )

    async def _freeze_step(self, service: str) -> str:
        """The stop the test command runs after test.sh's exit
        (:func:`verifier_freeze.freeze_step`), or ``""`` with the record of
        why there is none: only a root verifier can stop the agent's
        processes, and only beside them (main)."""
        from benchflow.task import verifier_freeze

        self._freeze_state = None
        self.freeze_record = None
        if not self.freeze_at_exit:
            return ""
        if service != "main":
            self.freeze_record = verifier_freeze.not_made(
                f"test.sh runs in service {service!r}, not beside the agent's processes"
            )
            return ""
        try:
            uid = await self._verifier_uid(service)
        except Exception as exc:
            self.freeze_record = verifier_freeze.not_made(
                f"the verifier's uid could not be read ({describe_exception(exc)})"
            )
            return ""
        if uid != "0":
            self.freeze_record = verifier_freeze.not_made(
                f"the verifier runs as uid {uid}, which cannot stop the agent's "
                "processes"
            )
            return ""
        state = verifier_freeze.state_path()
        self._freeze_state = (state, service)
        # The channels the stop leaves running follow the channel in use:
        # ``bfx serve`` only where bfx sits at /.bfx (bfx_delivered); per-call
        # (Daytona) and bfx off run the kit's helpers (map-H2 §1.2f). A name
        # alone exempts nobody: the stop also asks for a real uid of 0, so the
        # kit's patterns hold whether or not Q4 refused model agents here.
        from benchflow.sandbox._bounded_exec import bfx_delivered

        return verifier_freeze.freeze_step(
            state, kit_channels=not bfx_delivered(self._sandbox)
        )

    async def _cancel_freeze(self) -> None:
        """No stop after this test command's exit once its exec outlived
        its budget; a stop already made stays until :meth:`thaw_after_test`."""
        freeze_state = getattr(self, "_freeze_state", None)
        if freeze_state is None:
            return
        state, service = freeze_state
        try:
            await exec_raw(
                self._sandbox.exec,
                f": > {shlex.quote(state + '.cancel')}",
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            )
        except Exception as exc:
            self._logger.warning("the stop after test.sh was not cancelled: %s", exc)

    async def thaw_after_test(self) -> None:
        """Let the sandbox go on after the stop at test.sh's exit and record
        the freeze (``freeze_record``); a no-op once done, or when none was
        made."""
        pending, self._freeze_state = getattr(self, "_freeze_state", None), None
        if pending is None:
            return
        from benchflow.task import verifier_freeze

        state, service = pending
        record = verifier_freeze.not_made("the thaw did not run")
        for _ in range(2):
            try:
                result = await exec_raw(
                    self._sandbox.exec,
                    verifier_freeze.thaw_command(state),
                    user="root",
                    service=service,
                    timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                )
            except Exception as exc:
                record = verifier_freeze.not_made(
                    f"the thaw failed: {describe_exception(exc)}"
                )
                self._logger.warning(
                    "The sandbox's processes may still be stopped: %s", exc
                )
                continue
            record = verifier_freeze.parse(
                getattr(result, "stdout", "") or "", _exec_return_code(result)
            )
            break
        self.freeze_record = record
        with contextlib.suppress(AttributeError):
            self._sandbox._benchflow_verifier_freeze = record

    def _take_outputs_at_exit(self) -> None:
        """test.sh's outputs as they were the moment it exited (CIR's
        L14): the rewards are read from this copy, and the rollout puts it
        back over ``verifier/`` once nothing in the sandbox can write there,
        so a process left running changes neither the score nor the saved
        outputs."""
        from benchflow.task import verifier_outputs

        rollout_dir = getattr(self._rollout_paths, "rollout_dir", None)
        if not isinstance(rollout_dir, (str, Path)):
            return
        try:
            taken = verifier_outputs.take(Path(rollout_dir))
        except OSError as exc:
            # The rewards are then read from verifier/ as before.
            self._logger.warning("test.sh's outputs not copied at its exit: %s", exc)
            return
        self.outputs_at_exit = taken
        self._written_after_exit = {}
        with contextlib.suppress(AttributeError):
            self._sandbox._benchflow_verifier_outputs_at_exit = self.outputs_at_exit
            self._sandbox._benchflow_verifier_written_after_exit = (
                self._written_after_exit
            )

    def _write_reward_files(
        self, rewards: dict[str, Any], details: dict[str, Any]
    ) -> None:
        self._write_verifier_file(
            self._rollout_paths.reward_json_path,
            json.dumps(rewards, indent=2, allow_nan=False),
        )
        self._write_verifier_file(
            self._rollout_paths.reward_details_json_path,
            json.dumps(details, indent=2, allow_nan=False),
        )

    async def _download_verifier_dir(
        self, sandbox_paths: SandboxPaths, service: str
    ) -> None:
        """Download the sandbox's verifier folder privately, then place each entry."""
        verifier_dir = self._rollout_paths.verifier_dir
        await self._read_back(str(sandbox_paths.verifier_dir), service)
        with private_folder(verifier_dir) as private:
            landing = private / "verifier"
            destination(landing).mkdir()
            await self._sandbox.download_dir(
                source_dir=str(sandbox_paths.verifier_dir),
                target_dir=landing,
                service=service,
            )
            verifier_dir.publish_entries(landing)

    async def _bundled_tests(
        self, verifier_code_dir: PurePosixPath, service: str
    ) -> bool:
        """Whether this sandbox's image already holds the task's tests.

        A task.toml task's separate verifier built from ``tests/Dockerfile``
        carries its tests as that Dockerfile left them (the format's bundled
        tests; every Terminal-Bench 4.1 verifier image copies them in). Its
        ``/tests`` is used as built when the test script is there: uploading
        ``tests/`` over it brought back files the Dockerfile removed on
        purpose (risk-scorer-replay deletes its private scorer's source, then
        checks that nobody can read one). Any other sandbox, or an image
        without its test script, gets the upload as before.
        """
        if getattr(self._task, "_benchflow_bundled_tests", False) is not True:
            return False
        script = (
            verifier_code_dir
            / self._task.paths.test_path.relative_to(
                self._task.paths.tests_dir
            ).as_posix()
        )
        found = await exec_raw(
            self._sandbox.exec,
            f"test -f {shlex.quote(str(script))}",
            user="root",
            service=service,
            timeout_sec=30,
        )
        return getattr(found, "return_code", 1) == 0

    async def verify(self) -> VerifierResult:
        """Run the configured verifier and return the reward result."""
        self._save_outputs = None
        self._clear_reward_outputs()
        verifier_document = self._selected_verifier_document()
        if verifier_document is not None:
            strategy = verifier_document.selected_strategy
            if strategy.type == "script":
                return await self._verify_test_script(
                    strategy=strategy,
                    aggregate_policy=verifier_document.outputs.aggregate_policy,
                )
            if strategy.type == "llm-judge":
                return await self._verify_llm_judge(strategy=strategy)
            if strategy.type == "criteria-runner":
                return await self._verify_criteria_runner(
                    strategy=strategy,
                    document=verifier_document,
                    aggregate_policy=verifier_document.outputs.aggregate_policy,
                )
            if strategy.type == "agent-judge":
                return await self._verify_agent_judge(
                    strategy=strategy,
                    document=verifier_document,
                )
            if strategy.type == "ors-episode":
                return await self._verify_ors_episode(strategy=strategy)
            if strategy.type == "taskmd":
                from benchflow.taskmd.verify import verify_taskmd

                return await verify_taskmd(self, strategy)
            raise UnsupportedVerifierStrategyError(
                f"verifier strategy {strategy.name!r} has type {strategy.type!r}, "
                "which is parsed but not executable yet"
            )
        if self._task.config.verifier.type == "llm-judge":
            return await self._verify_llm_judge()
        return await self._verify_test_script()

    def _selected_verifier_document(self) -> VerifierDocument | None:
        paths = getattr(self._task, "paths", None)
        verifier_dir = getattr(paths, "tests_dir", None)
        if not isinstance(verifier_dir, str | Path):
            return None
        return load_verifier_document(verifier_dir)

    async def _make_verifier_code_roots(
        self, verifier_code_dir: PurePosixPath, service: str
    ) -> None:
        """The task's tests, as uploaded, are root's alone before anything
        reads them: ``docker cp`` writes the host's uid:gid, which an account
        of the image can share. Every agent-uid process is gone by then
        (quiesce), so this removes that dependency rather than a live route
        (RUSAGE on). It closes no open route, so a failed chown leaves
        the files as uploaded (as before) and is a warning in the trial's
        record, never the trial's end."""
        from benchflow._utils.trial_warnings import report_trial_warning

        path = shlex.quote(str(verifier_code_dir))
        try:
            result = await exec_raw(
                self._sandbox.exec,
                f"chown -R 0:0 {path}",
                user="root",
                service=service,
                timeout_sec=60,
            )
            detail = f"rc={_exec_return_code(result)}"
            failed = _exec_return_code(result) != 0
        except Exception as e:
            detail, failed = type(e).__name__, True
        if failed:
            self._logger.warning(
                "Could not make the uploaded tests root's alone (chown -R 0:0): %s",
                detail,
            )
            report_trial_warning(
                f"the uploaded tests at {verifier_code_dir} stay as uploaded, not "
                f"root's alone (chown -R 0:0: {detail})"
            )

    async def _link_legacy_verifier_mount(
        self,
        *,
        verifier_code_dir: PurePosixPath,
        sandbox_paths: SandboxPaths,
        service: str,
    ) -> None:
        """Alias the legacy ``/tests`` mount onto a native verifier dir.

        ``bench tasks migrate`` mounts a task.md (native) verifier at
        ``/verifier`` rather than the legacy ``/tests``. Verifier scripts carried
        over from a legacy benchmark commonly hardcode ``/tests/...`` paths
        (e.g. ``python3 /tests/evaluate.py``), so a faithful conversion must keep
        those resolvable. When the native mount differs from ``/tests``, symlink
        ``/tests`` → the native dir — but only when ``/tests`` is absent, so real
        content is never clobbered. No-op for legacy tasks (already mounted at
        ``/tests``) and harmless for native verifiers that reference
        ``$BENCHFLOW_VERIFIER_DIR`` instead of a hardcoded path.
        """
        legacy_dir = sandbox_paths.tests_dir
        if str(verifier_code_dir) == str(legacy_dir):
            return
        src = shlex.quote(str(verifier_code_dir))
        dst = shlex.quote(str(legacy_dir))
        try:
            result = await exec_raw(
                self._sandbox.exec,
                f"[ -e {dst} ] || ln -s {src} {dst}",
                user="root",
                service=service,
                timeout_sec=10,
            )
        except Exception as e:  # best-effort; native verifiers may not need it
            self._logger.debug("legacy verifier-mount symlink skipped: %s", e)
            return
        if _exec_return_code(result) != 0:
            self._logger.debug(
                "legacy verifier-mount symlink (%s -> %s) returned nonzero; "
                "verifier scripts hardcoding %s may fail",
                legacy_dir,
                verifier_code_dir,
                legacy_dir,
            )

    async def _setup_step(self, what: str, run: Callable[[], Awaitable[Any]]) -> None:
        """Run one setup exec (``run``), retrying a failed or lost one.

        Raises ``Verifier setup failed: ...`` (verifier infrastructure) once
        every try failed; ``run`` either returns the exec result or raises.
        """
        failure = f"{_SETUP_FAILED} {what} did not run"
        cause: Exception | None = None
        for delay in (0.0, *_SETUP_RETRY_DELAYS_SEC):
            if delay:
                self._logger.warning("%s; retrying in %.0fs", failure, delay)
                await asyncio.sleep(delay)
            try:
                result = await run()
            except Exception as e:
                cause = e
                text = str(e)
                failure = (
                    text
                    if text.startswith(_SETUP_FAILED)
                    else f"{_SETUP_FAILED} {what} did not run: {describe_exception(e)}"
                )
                continue
            return_code = _exec_return_code(result)
            if return_code == 0:
                return
            failure = f"{_SETUP_FAILED} {what} exited with rc={return_code}"
            cause = None
        if cause is not None and not isinstance(
            cause, (TimeoutError, ExecTimeoutError)
        ):
            raise VerifierExecutionError(failure) from cause
        raise VerifierSetupError(failure) from cause

    def _keeps_services(self, service: str) -> bool:
        """Whether services the task keeps run beside this verifier's
        commands: the scoring path sets ``before_test`` only then, and they
        live in the agent's container (``main``). Setup refuses kept services
        beside a verifier in any other service (``keep_services._refusal``):
        its env would go on a command line they may see (DT2-CONF-10)."""
        if self.before_test is not None and service != "main":
            raise VerifierSetupError(
                f"{_SETUP_FAILED} kept services cannot run beside a verifier "
                f"in service {service!r}; run it in main"
            )
        return self.before_test is not None

    def _agent_may_run(self, service: str) -> bool:
        """Whether agent code may be alive beside this verifier's commands in
        *service*: services the task keeps, or the agent's own processes in
        soft verification. Both live in the agent's container (``main``)."""
        return service == "main" and (
            self.before_test is not None or self.agent_processes_alive
        )

    def _host_reads_rewards(self) -> bool:
        """Whether this host can open every canonical reward file there is."""
        for path in (
            self._rollout_paths.reward_json_path,
            self._rollout_paths.reward_text_path,
        ):
            try:
                if path.exists():
                    with destination(path).open("rb"):
                        pass
            except PermissionError:
                return False
            except OSError:
                continue  # Not a permission problem: the parse reports it.
        return True

    async def _close_output_dir(self, service: str, *, mounted: bool) -> None:
        """While test.sh runs, its outputs folder is root's, mode 0755, with
        whatever another user made there since the clear removed:
        the agent's processes, alive through grading in integrity off and
        audit, can no longer race the reward, and a test.sh that refuses a
        reward folder not root's runs as it did before the gate (TB4.1
        wal-recovery-ordering). Where the folder is the host's mount and the
        host user can then not write there, root also makes test-stdout.txt,
        and the folder goes to the host user once test.sh has exited
        (:meth:`_open_output_dir_to_host`). A verifier that runs as a user
        of its own keeps the folder as it was, as do kept services, whose own
        rule applies.

        A root test.sh in integrity audit and off also meets each workspace
        the sandbox user's grant made root's and 1777 (a daemon folder or a
        reserved folder in it) without group or other write, in this same
        exec: see :meth:`_sticky_workspaces` (#1249)."""
        self._closed_to_host = False
        self.outputs_close_due = True
        if self.before_test is not None:
            return
        try:
            uid = await self._verifier_uid(service)
        except VerifierSetupError:
            # Not known: the folder stays as the clear left it, open to the
            # agent; the reward's provenance says so.
            self.outputs_closed = False
            self.outputs_open_why = "the verifier user's uid could not be read"
            return
        owner = uid if uid.isdigit() else "0"
        sticky = self._sticky_workspaces(service, owner)
        owner_spec = "0:0" if owner == "0" else owner
        # Root's own files stay (BenchFlow's launch record, test-stdout.txt);
        # the hard-link rule holds for root's and the verifier user's alike.
        owned = "-user 0" if owner == "0" else f"\\( -user 0 -o -user {owner} \\)"
        foreign = "! -user 0" if owner == "0" else f"! -user 0 ! -user {owner}"
        folder = shlex.quote(str(SandboxPaths().verifier_dir))
        stdout = shlex.quote(str(SandboxPaths().verifier_dir / "test-stdout.txt"))
        # Whatever another user made there since the clear goes (a file the
        # agent owns, or holds open, would stay its own to write). `-user 0`,
        # not `-uid 0`: BusyBox's find (Alpine) has no -uid, and GNU find
        # takes a numeric -user. The marker says the folder is root's from
        # here on, whatever fails after it.
        # The folder is the verifier's own user's: root's, or a verifier user
        # of its own (read-back #2), never the agent's.
        command = (
            f"chown {owner_spec} {folder} && chmod 755 {folder} && echo {_OUTPUTS_CLOSED} && "
            "if command -v find >/dev/null 2>&1; then "
            f"find {folder} -mindepth 1 -type d -user {owner} -exec chmod go-w {{}} + && "
            # A root-owned name with another link (a hard link to a root file
            # the agent may write) is unlinked; the file it names elsewhere
            # stays as it was (read-back #5).
            f"find {folder} -mindepth 1 {owned} ! -type d -links +1 "
            "-exec rm -f -- {} + && "
            f"find {folder} -mindepth 1 {foreign} -prune -exec sh -c "
            f"{shlex.quote(_SWEEP_FOREIGN)} sh {{}} +; "
            # An image without find: the shell sweeps each top-level link and
            # each name not root's.
            + ("" if owner == "0" else "else exit 1; fi")
            + (
                ""
                if owner != "0"
                else f"else set --; for e in {folder}/* {folder}/.[!.]* {folder}/..?*; do "
                'if [ -L "$e" ] || { [ -e "$e" ] && [ ! -O "$e" ]; }; then '
                'set -- "$@" "$e"; fi; done; '
                f'[ $# -eq 0 ] || sh -c {shlex.quote(_SWEEP_FOREIGN)} sh "$@"; fi'
            )
        )
        if mounted:
            command += f" && : > {stdout} && chmod 644 {stdout}"
            if owner != "0":
                # test.sh's own user writes its output there.
                command += f" && chown {owner} {stdout}"
        if sticky:
            from benchflow._runtime_tree import (
                WORKSPACE_CLOSED_ATTR,
                close_granted_command,
            )

            # Its status is not the close's: what it shows is judged below.
            command = f"{command} && {{ {close_granted_command(sticky)}; }}"
            with contextlib.suppress(AttributeError):
                # Re-opened before another turn whatever the exec answers (a
                # timeout included).
                setattr(self._sandbox, WORKSPACE_CLOSED_ATTR, dict(sticky))
        with contextlib.suppress(AttributeError):
            self._sandbox._benchflow_outputs_clearing = "the close"
        try:
            result = await exec_raw(
                self._sandbox.exec,
                command,
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                **self._setup_exec_kwargs(),
            )
        except (TimeoutError, ExecTimeoutError):
            with contextlib.suppress(AttributeError):
                self._sandbox._benchflow_outputs_clearing = None
            self.outputs_closed = False
            self.outputs_open_why = "the close did not finish"
            raise VerifierOutputsHeldError(
                f"{_OUTPUTS_HELD} (the close did not finish in "
                f"{_SETUP_EXEC_TIMEOUT_SEC} s)"
            ) from None
        except asyncio.CancelledError:
            # The verifier's deadline: the close stays named for its handler.
            raise
        except BaseException:
            with contextlib.suppress(AttributeError):
                self._sandbox._benchflow_outputs_clearing = None
            raise
        with contextlib.suppress(AttributeError):
            self._sandbox._benchflow_outputs_clearing = None
        closed = _OUTPUTS_CLOSED in str(getattr(result, "stdout", "") or "").split()
        if _exec_return_code(result) != 0:
            said = str(getattr(result, "stderr", "") or "").strip().splitlines()
            detail = f"rc={_exec_return_code(result)}" + (
                f": {said[0][:200]}" if said else ""
            )
            self.outputs_closed = False
            if closed:
                # The folder is root's, but other users' files may still be
                # in it (or root's test-stdout.txt is not): grading would not
                # hold the rule it states, so it does not start, and the
                # verifier failed.
                self.outputs_open_why = f"{_OUTPUTS_HELD} ({detail})"
                raise VerifierOutputsHeldError(f"{_OUTPUTS_HELD} ({detail})")
            # The folder never became root's: test.sh does not run with it
            # open to the agent (read-back #6), the same ending.
            self.outputs_open_why = f"the close failed ({detail})"
            raise VerifierOutputsHeldError(
                "the outputs folder could not be made root's, so test.sh does "
                f"not run with it open to the agent's processes ({detail})"
            )
        self.outputs_closed = True
        # Root's from the marker on: the host writes test-stdout.txt only to a
        # folder it still owns.
        self._closed_to_host = mounted and closed and not self._host_writes_outputs()
        if sticky:
            self._check_sticky_closed(result, sticky)

    def _sticky_workspaces(self, service: str, owner: str) -> dict[str, str]:
        """The workspaces whose group and other write a root test.sh's close
        takes (#1249 (b), (c)), with the modes the grant recorded
        (``_runtime_tree.WORKSPACE_MODE_ATTR``); empty when none applies.

        Only for test.sh as root in the agent's container (``main``), in
        integrity audit and off (test.sh as written). Strict gives each
        workspace its image mode back in the freeze before (and its separate
        verifier is a fresh sandbox); a verifier user of the task's own owns
        what it writes and may need to create files in the workspace, so the
        grant stays as it is for it."""
        if service != "main" or owner != "0" or not self._runs_as_written():
            return {}
        from benchflow._runtime_tree import WORKSPACE_MODE_ATTR

        modes = getattr(self._sandbox, WORKSPACE_MODE_ATTR, None)
        if not isinstance(modes, dict):
            return {}
        return {
            path: mode
            for path, mode in modes.items()
            if isinstance(path, str) and isinstance(mode, str)
        }

    def _check_sticky_closed(self, result: Any, sticky: dict[str, str]) -> None:
        """Judge what the workspace step showed: each workspace without group
        or other write, else a verifier error before test.sh (never a quiet 0
        from a write test.sh could not make). The modes go in the verifier's
        record, and the next agent turn gets the grant's mode back first
        (``lockdown.reopen_granted_workspaces``)."""
        from benchflow._runtime_tree import (
            granted_closed,
            shown_modes,
            shown_modes_problem,
        )

        stdout = getattr(result, "stdout", "")
        stdout = stdout if isinstance(stdout, str) else ""
        shown = shown_modes(stdout)
        self.workspace_modes_before_test = {
            path: f"{shown[path][0]:o}" if path in shown else None
            for path in sorted(sticky)
        }
        problem = shown_modes_problem(stdout, sticky, granted_closed)
        if problem is not None:
            raise VerifierSetupError(
                "Verifier hardening failed: taking group and other write from "
                f"the workspace before test.sh ({problem})"
            )

    async def _open_output_dir_to_host(self, service: str) -> None:
        """Once test.sh has exited (and its outputs are copied, see
        :meth:`_take_outputs_at_exit`), give the mounted folder the
        :meth:`_close_output_dir` closed to the host user, so BenchFlow can
        add its records there: the folder only, test.sh's files staying
        root's as before the gate; failing that, the trial stops (never 0777,
        #1247)."""
        if not getattr(self, "_closed_to_host", False):
            return
        self._closed_to_host = False
        if self._host_writes_outputs():
            return  # the whole tree went to the host already
        folder = shlex.quote(str(SandboxPaths().verifier_dir))
        if hasattr(os, "getuid"):
            await exec_raw(
                self._sandbox.exec,
                f"chown {os.getuid()}:{os.getgid()} {folder}",
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                **self._setup_exec_kwargs(),
            )
            if self._host_writes_outputs():
                return
        # Never opened to every user (#1247): the agent's processes are alive
        # after test.sh in audit and off. By the code this is not reached: a
        # root host, and rootless Docker (container root is the host user),
        # can write the folder after the close, so ``_closed_to_host`` stays
        # False; with the host uid's chown done, a host that still can't
        # write is one whose container root isn't host root (userns-remap),
        # where the close's own chown of the host's folder fails first.
        # Reached anyway, the trial stops here instead.
        raise VerifierInfrastructureError(
            "the verifier outputs folder could not be given back to the host user "
            "after test.sh, and BenchFlow will not open it to every user while "
            "the agent's processes may run; its records for this trial can't be "
            "written"
        )

    def _host_writes_outputs(self) -> bool:
        """Whether this host can add files to the rollout's ``verifier/``.

        BenchFlow's own records go there after the tests: the network
        verdict's ``verifier-network.json`` (the launch's CapBnd/CapEff
        among it), a breach's ``network-policy.json``. A test.sh that makes
        the mounted ``/logs/verifier`` root's, mode 0755 (FDB's does), left
        the host able to read the reward but not to write a record, and the
        record was lost without a word.
        """
        geteuid = getattr(os, "geteuid", None)
        if geteuid is None or geteuid() == 0:
            return True
        try:
            st = self._rollout_paths.verifier_dir.stat()
        except OSError:
            return True  # Not a permission problem: the writes report it.
        if st.st_uid == geteuid():
            need = stat.S_IWUSR | stat.S_IXUSR
        elif st.st_gid == os.getegid() or st.st_gid in os.getgroups():
            need = stat.S_IWGRP | stat.S_IXGRP
        else:
            need = stat.S_IWOTH | stat.S_IXOTH
        return st.st_mode & need == need

    async def _let_host_read_outputs(
        self, sandbox_paths: SandboxPaths, service: str
    ) -> None:
        """Let the host read what the test command left on the mounted outputs
        before they are copied (:meth:`_take_outputs_at_exit`).

        A test.sh that writes through ``mkstemp`` and a rename, or under
        ``umask 077``, leaves root's 0600 files there (#1202). The host reads
        them as its own user, so the copy skipped the reward and the trial
        lost it (a shared reward script writes every reward this way); the
        hand-over to the host user (:meth:`_hand_mounted_outputs_to_host`)
        never runs while the agent's processes may live. Root adds read for
        group and others, nothing else:

        - only names directly in the folder (a subfolder may be another
          user's to change: REV2 R2-1), the names the host reads to score
          first (:data:`verifier_outputs.PRIORITY`), the rest after them,
          capped (R2-4);
        - through bfx's ``chmod``, which acts on the descriptor it opened,
          never follows a link and skips a hardlinked file, and only once
          bfx's ``stat`` shows the folder root's (or the verifier user's)
          with no group or other write: no other user can then swap a name in
          it;
        - the reward names on that alone; the rest only while the freeze is
          proven held (its state file) or where no agent code can run.

        Without bfx (it isn't delivered: only the oracle runs in off and
        audit, Q4; strict has no agent code left) a root shell does the same
        name by name, a regular file and no link. The record is
        ``read_grant``.
        """
        from benchflow.task import verifier_outputs

        rollout_dir = getattr(self._rollout_paths, "rollout_dir", None)
        if (
            not isinstance(rollout_dir, (str, Path))
            or getattr(os, "getuid", None) is None
        ):
            return
        if os.getuid() == 0:
            return
        try:
            names = verifier_outputs.unreadable_top(Path(rollout_dir))
        except OSError:
            return
        if not names:
            return
        priority = [n for n in names if n in verifier_outputs.PRIORITY]
        rest = [n for n in names if n not in verifier_outputs.PRIORITY]
        grant: dict[str, Any] = {
            "granted": [],
            "skipped": {},
            "not_held": 0,
            "not_granted": 0,
        }
        self.read_grant = grant
        folder = str(sandbox_paths.verifier_dir)
        if self._bfx_grants():
            why = await self._outputs_folder_open(folder)
            if why is not None:
                grant["why"] = why
                self._logger.warning(
                    "test.sh's unreadable outputs were not made readable: %s", why
                )
                return
            if rest and self._agent_may_run(service) and not await self._freeze_held():
                grant["not_held"] = len(rest)
                rest = []
        room = max(0, _HOST_READ_MAX - len(priority))
        grant["not_granted"] = max(0, len(rest) - room)
        targets = priority + rest[:room]
        if self._bfx_grants():
            await self._bfx_grant(folder, targets, grant)
        else:
            await self._shell_grant(folder, targets, grant, service)
        try:
            left = [
                n
                for n in verifier_outputs.unreadable_top(Path(rollout_dir))
                if n in priority
            ]
        except OSError:
            left = []
        if left:
            grant["still_unreadable"] = left
            self._logger.warning(
                "test.sh's %s could not be made readable to the host: %s",
                ", ".join(left),
                grant["skipped"] or "unknown why",
            )

    def _bfx_grants(self) -> bool:
        """Whether the read grant goes through bfx (it is delivered here)."""
        return isinstance(
            getattr(self._sandbox, "bfx_delivery", None), str
        ) and callable(getattr(self._sandbox, "bfx_call", None))

    async def _outputs_folder_open(self, folder: str) -> str | None:
        """Why no name in *folder* may be granted (another user could swap
        one), from bfx's ``stat`` of it; None when it is root's or the
        verifier user's, with no group or other write."""
        parent, _, name = folder.rstrip("/").rpartition("/")
        try:
            answer = await self._sandbox.bfx_call(
                "stat",
                {"root": parent or "/", "paths": [name]},
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            )
        except Exception as exc:
            return (
                f"the output folder could not be looked at: {describe_exception(exc)}"
            )
        entries = [e for e in answer.get("entries") or () if isinstance(e, dict)]
        facts = entries[0] if entries else {}
        if "error" in facts or facts.get("type") != "dir":
            return f"the output folder could not be looked at ({facts})"
        uid = self._verifier_uid_read
        owners = {0} | ({int(uid)} if isinstance(uid, str) and uid.isdigit() else set())
        if facts.get("uid") not in owners:
            return f"the output folder is uid {facts.get('uid')}'s"
        mode = facts.get("mode")
        if not isinstance(mode, int) or mode & 0o022:
            return "the output folder is writable by its group or others"
        return None

    async def _freeze_held(self) -> bool:
        """Whether the stop after test.sh's exit holds the sandbox still now:
        its state file, root's and written only by the stop, says it was made
        and every process stopped (:func:`verifier_freeze.held`)."""
        import base64

        from benchflow.task import verifier_freeze

        pending = getattr(self, "_freeze_state", None)
        if pending is None:
            return False
        state, _service = pending
        try:
            answer = await self._sandbox.bfx_call(
                "scan",
                {"paths": [state], "contents": [state], "mounts": False, "acl": False},
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            )
        except Exception:
            return False
        facts = next(
            (
                e
                for e in answer.get("entries") or ()
                if isinstance(e, dict) and e.get("path") == state
            ),
            {},
        )
        mode = facts.get("mode")
        if (
            facts.get("type") != "file"
            or facts.get("uid") != 0
            or not isinstance(mode, int)
            or mode & 0o022
        ):
            return False
        content = (answer.get("contents") or {}).get(state) or {}
        try:
            text = base64.b64decode(content.get("b64") or "").decode("utf-8", "replace")
        except (ValueError, TypeError):
            return False
        return verifier_freeze.held(text)

    async def _bfx_grant(
        self, folder: str, names: list[str], grant: dict[str, Any]
    ) -> None:
        """bfx ``chmod +044`` on each of *names* in *folder*: the descriptor
        it opened, never a link, never a hardlinked file. One ``bfx serve``
        for many names where the sandbox has it."""
        from benchflow.sandbox._verifier_links import _wire

        def args(name: str) -> dict[str, Any]:
            return _wire({"root": folder, "set": "044"}, "path", name)

        def record(name: str, answer: Any) -> None:
            answer = answer if isinstance(answer, dict) else {}
            if answer.get("skipped_links"):
                grant["skipped"][name] = "a link"
            elif answer.get("skipped_hardlinks"):
                grant["skipped"][name] = "a file with another name elsewhere"
            elif answer.get("skipped_mounts"):
                grant["skipped"][name] = "a mount point"
            else:
                grant["granted"].append(name)

        session = getattr(self._sandbox, "bfx_session", None)
        if len(names) > 8 and callable(session):
            try:
                async with session(timeout_sec=_SETUP_EXEC_TIMEOUT_SEC) as serve:
                    for name in names:
                        try:
                            record(
                                name,
                                await serve.call(
                                    "chmod", args(name), timeout=_SETUP_EXEC_TIMEOUT_SEC
                                ),
                            )
                        except Exception as exc:
                            grant["skipped"][name] = describe_exception(exc)
                return
            except Exception as exc:
                self._logger.warning(
                    "bfx serve for the read grant did not start (%s); one call "
                    "per name instead",
                    describe_exception(exc),
                )
                names = [n for n in names if n not in grant["granted"]]
        for name in names:
            try:
                record(
                    name,
                    await self._sandbox.bfx_call(
                        "chmod", args(name), timeout_sec=_SETUP_EXEC_TIMEOUT_SEC
                    ),
                )
            except Exception as exc:
                grant["skipped"][name] = describe_exception(exc)

    async def _shell_grant(
        self, folder: str, names: list[str], grant: dict[str, Any], service: str
    ) -> None:
        """Without bfx: a root shell adds read to each name directly in
        *folder* that is still a regular file and no link, a few at a time
        (never one long argv)."""
        for i in range(0, len(names), _HOST_READ_CHUNK):
            chunk = names[i : i + _HOST_READ_CHUNK]
            targets = " ".join(shlex.quote(f"{folder}/{name}") for name in chunk)
            try:
                result = await exec_raw(
                    self._sandbox.exec,
                    f'for p in {targets}; do [ -f "$p" ] && [ ! -L "$p" ] && '
                    'chmod go+r -- "$p"; done; true',
                    user="root",
                    service=service,
                    timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                    **self._setup_exec_kwargs(),
                )
            except Exception as e:
                for name in chunk:
                    grant["skipped"][name] = describe_exception(e)
                continue
            if _exec_return_code(result) != 0:
                for name in chunk:
                    grant["skipped"][name] = f"rc={_exec_return_code(result)}"
                continue
            grant["granted"].extend(chunk)

    async def _hand_mounted_outputs_to_host(
        self, sandbox_paths: SandboxPaths, service: str
    ) -> None:
        """Let the host read a reward the test command left on the mounted outputs.

        On Linux rootful docker the rollout's verifier dir is bind-mounted and
        the test command runs as container root, so a reward written through
        ``mkstemp`` (mode 0600) is root's and the host cannot parse it (#1202);
        the download that chowns outputs is skipped when they are mounted.
        Only then, or when the host cannot add BenchFlow's records to the
        folder (a test.sh made it root's 0755), hand the folder to the host
        user, as the sandbox's teardown does: where the host reads it and
        writes there already (rootless docker maps container root to the host
        user) nothing changes owner. Never while agent code may still run
        beside the verifier, since its uid can be the host user's.
        """
        chown = getattr(self._sandbox, "_chown_to_host_user", None)
        if (
            chown is None
            or self._agent_may_run(service)
            or (self._host_reads_rewards() and self._host_writes_outputs())
        ):
            return
        try:
            await asyncio.wait_for(
                chown(str(sandbox_paths.verifier_dir), recursive=True, service=service),
                timeout=30,
            )
        except Exception as e:
            self._logger.warning(
                "Could not hand the mounted verifier outputs to the host user: %s",
                describe_exception(e),
            )

    def _setup_exec_kwargs(self) -> dict[str, Any]:
        """Root's own setup execs start /bin/sh by path with PATH the system
        folders where the sandbox can (``system_shell``); never a tool from
        the image's PATH, which can lead into a folder the agent wrote."""
        if getattr(self._sandbox, "supports_system_shell", False) is True:
            return {"system_shell": True}
        return {}

    async def _read_back(self, path: str, service: str) -> None:
        """Root reads *path* back only when nothing on it is the agent's to
        change (:func:`benchflow.task._read_back.read_back`), where the
        agent's processes are alive through grading; else the outputs are
        held (counted, never retried)."""
        if not getattr(self, "agent_processes_alive", False):
            return
        from benchflow.task._read_back import ReadBackRefused, read_back

        owners = (0,)
        if (
            isinstance(self._verifier_uid_read, str)
            and self._verifier_uid_read.isdigit()
        ):
            owners = (0, int(self._verifier_uid_read))
        try:
            await read_back(self._sandbox, path, owners=owners, service=service)
        except ReadBackRefused as exc:
            raise VerifierOutputsHeldError(
                f"{_OUTPUTS_HELD} (not read back: {exc})"
            ) from None

    async def _verifier_uid(self, service: str) -> str:
        """The uid the verifier's commands run as (``[verifier] user``, else
        the sandbox's default user), read by the shell itself."""
        user = self._task.config.verifier.user
        if user in ("root", 0, "0"):
            return "0"
        cached = self._verifier_uid_read
        if isinstance(cached, str):
            return cached  # read once per verify (the close reads it too)
        result = await exec_raw(
            self._sandbox.exec,
            _VERIFIER_UID_CMD,
            user=user,
            service=service,
            timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            **self._setup_exec_kwargs(),
        )
        uid = (getattr(result, "stdout", "") or "").strip()
        if _exec_return_code(result) != 0 or not uid.isdigit():
            raise VerifierSetupError(
                f"{_SETUP_FAILED} the verifier user's uid could not be read "
                f"(rc={_exec_return_code(result)})"
            )
        self._verifier_uid_read = uid
        return uid

    async def _require_writable_test_stdout(
        self, test_stdout_path: str, service: str
    ) -> None:
        """Refuse, before the test command, a verifier that cannot write the
        file its output goes to.

        Where the verifier folder is the host's own (Docker's bind mount of
        the trial's verifier/), BenchFlow makes test-stdout.txt there before
        the test command, as the host's user, mode 0644. A verifier that runs
        as another user, not root (``[verifier] user``, else the image's own
        USER), cannot open it: the shell's redirect fails (exit 2), test.sh
        never runs and no reward is written, which would count as a failure
        in every score. So the verifier's user opens it first, without
        writing; when that open is refused, the trial ends as a verifier setup
        failure (infrastructure, unscored) naming the file, its owner and the
        uid.

        Checked where hardening asked for the verifier's uid (every trial
        hardened in this sandbox) and did not hear 0, so a root image adds no
        exec; a sandbox that was never hardened (a stand-in) is left alone.
        Hardening's read takes a receipt, which a user other than root cannot
        give, so for the image's own non-root user it has no answer and this
        check asks. Only the probe's own refusal decides; a probe that cannot
        run or answers anything else leaves the test command to run as before.
        """
        user = self._task.config.verifier.user
        if user in ("root", 0, "0"):
            return
        known = getattr(self._sandbox, "_benchflow_verifier_uid", None)
        if not isinstance(known, str) or known == "0":
            return
        uid: str | None = known if known.isdigit() else None
        try:
            probe = await exec_raw(
                self._sandbox.exec,
                stdout_probe_command(test_stdout_path),
                user=user,
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            )
        except Exception:
            return
        answer = getattr(probe, "stdout", None)
        if not isinstance(answer, str) or answer.strip() != _STDOUT_UNWRITABLE:
            return
        if uid is None:
            with contextlib.suppress(Exception):
                uid = await self._verifier_uid(service)
        owner = "of an owner BenchFlow could not read"
        with contextlib.suppress(OSError):
            st = self._rollout_paths.test_stdout_path.stat()
            owner = f"uid {st.st_uid}'s, mode {st.st_mode & 0o7777:04o}"
        raise VerifierSetupError(
            f"{_SETUP_FAILED} {VERIFIER_OUTPUT_UNWRITABLE}{test_stdout_path}: "
            f"the verifier runs as uid {uid or 'unknown'}, and the file is "
            f"{owner}, made by BenchFlow on the host before the test command. "
            'Run the verifier as root ([verifier] user = "root") until '
            "BenchFlow leaves that file to the test command"
        )

    async def _env_by_root_file(
        self, command: str, env: dict[str, str] | None, service: str
    ) -> tuple[str, dict[str, str] | None]:
        """Hand *env* to *command* through a file only root can read.

        A backend's exec may carry an env in the shell's command line (Daytona's
        inline form, ``wrap_command_with_env_file``), and any process in the
        sandbox can read /proc/<pid>/cmdline for as long as that shell runs:
        the whole test command. While agent code is alive during grading (the
        services a task keeps; the agent's own processes in soft verification,
        b156 PY8-MAIN-30), the verifier's env (``[verifier.env]``, an LLM
        judge's key) is uploaded instead, which puts it on no command line,
        into a folder root makes fresh, 0700, under a new name on each try
        (never one that already exists, K2-R5-A-1), inside a parent only root
        can enter (K2-R6-A-1); the test command sources
        it and removes the folder at once (K2-R4-B-1). The process environment of
        root's commands is not readable by the agent user. The sandbox's
        persistent env goes into the same file: while services are kept, exec
        adds it no more (``BaseSandbox.services_kept``), since each backend
        would source it from a file under /tmp. Returns the
        command to run and the env to pass to exec (None).

        Only a verifier that runs as root reads the file; one that runs as
        another user gets the env from exec, as before (its own processes'
        environment is that user's to read anyway). The setup commands find
        nothing on the image's PATH, and the command sets nothing but the
        env: its only scratch state is the positional parameters, which no
        task variable can name (b156 PY8-MAIN-32/33).
        """
        persistent = getattr(self._sandbox, "_persistent_env", None)
        merged = {**(persistent if isinstance(persistent, dict) else {}), **(env or {})}
        if not merged:
            return command, env
        uid = await self._verifier_uid(service)
        if uid != "0":
            # Another user's process environment is not the agent's to read,
            # but a command line is: only a backend whose exec passes the env
            # privately may carry it (b162 PY8-MAIN-36).
            if getattr(self._sandbox, "exec_env_is_private", False) is not True:
                raise VerifierSetupError(
                    f"{_SETUP_FAILED} the verifier runs as uid {uid} while agent "
                    f"code is alive, and {type(self._sandbox).__name__} can pass "
                    "its environment only on a command line the agent can read. "
                    "Run the verifier as root ([verifier] user), or use a sandbox "
                    "whose exec passes the environment on stdin (Docker, Apple "
                    "container)."
                )
            return command, merged
        from benchflow.sandbox._base import env_stdin_body

        body = env_stdin_body(merged)
        setup = self._setup_exec_kwargs()
        chosen: list[str] = []

        async def write() -> Any:
            # A fresh name on every try, and a plain mkdir, which fails if the
            # path exists: on a verifier retried in place the kept services
            # are running and can read this command's arguments in /proc, so
            # a folder (and an open env file) one of them made first under the
            # name is never reused (K2-R5-A-1). As root under umask 077 the
            # new folder is root's 0700 from the start. Its parent is root's
            # alone, so no kept service can rename it away (K2-R6-A-1).
            parent = await exec_raw(
                self._sandbox.exec,
                _VERIFIER_ENV_PARENT_CMD,
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                **setup,
            )
            if _exec_return_code(parent) != 0:
                return parent
            folder = shlex.quote(
                f"{_VERIFIER_ENV_PARENT}/.runtime-verifier-env-{uuid4().hex}"
            )
            made = await exec_raw(
                self._sandbox.exec,
                f"umask 077 && mkdir {folder}",
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                **setup,
            )
            if _exec_return_code(made) != 0:
                return made
            chosen[:] = [folder]
            path = f"{folder}/env"
            with tempfile.TemporaryDirectory() as tmp:
                local = Path(tmp) / "env"
                local.write_bytes(body)
                local.chmod(0o600)
                await self._sandbox.upload_file(
                    local,
                    path,
                    mode="600",
                    admission=admit(
                        self._sandbox,
                        "main",
                        TRUSTED_TRANSFER,
                        reason="the verifier environment BenchFlow computes",
                    ),
                )
            return await exec_raw(
                self._sandbox.exec,
                f"chown 0:0 {path} && chmod 600 {path}",
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                **setup,
            )

        await self._setup_step("writing the verifier env", write)
        (folder,) = chosen
        # The command removes it when it starts; one that never starts
        # leaves it to _remove_unread_root_env.
        self._unread_root_env = (folder, service)
        return _root_file_env_prefix(folder) + command, None

    async def _remove_unread_root_env(self, ran: bool) -> None:
        """Remove the root-only env folder of :meth:`_env_by_root_file` when
        its test command did not start (refused before it ran, or its exec
        failed), so the env is not kept for the rest of the trial or in a
        snapshot. A command that ran removed it itself."""
        pending = getattr(self, "_unread_root_env", None)
        self._unread_root_env = None
        if pending is None or ran:
            return
        folder, service = pending
        try:
            await exec_raw(
                self._sandbox.exec,
                f"/bin/rm -rf -- {folder}",
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                **self._setup_exec_kwargs(),
            )
        except Exception as exc:
            logger.warning(
                "Could not remove the verifier env's folder %s: %s",
                folder,
                describe_exception(exc),
            )

    async def _clear_output_dir(self, service: str) -> None:
        # While services are kept the folder stays closed to them: on a
        # verifier retried in place they are running by now (K2-R4-A-1).
        closed = self._keeps_services(service)
        await self._setup_step(
            "clearing verifier output directory",
            lambda: clear_verifier_output_dir(
                self._sandbox,
                closed=closed,
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            ),
        )

    async def _prepare_execution_receipt(self, service: str) -> str | None:
        """Return a fresh start-receipt path for the test command, if requested.

        An empty stdout file does not prove an exec failed to start: valid
        verifiers can be quiet for their entire budget, so the recovery path
        (#1136) has the command itself write a receipt. It lives outside
        ``/logs/verifier``, which task preflights require to hold only what
        test.sh writes. A receipt directory BenchFlow cannot create leaves
        startup unknown, as a failed probe does, rather than failing the run.
        """
        self.execution_receipt = None
        if not self._write_execution_receipt:
            return None
        attempt = uuid4().hex
        folder = _EXECUTION_RECEIPT_DIR / f"verifier-{attempt}"
        command = self._attempt_folder_command(folder, service)
        try:
            result = await exec_raw(
                self._sandbox.exec,
                command,
                user="root",
                service=service,
                timeout_sec=_RECEIPT_DIR_BUDGET_SEC,
            )
            return_code = _exec_return_code(result)
        except Exception as e:
            return_code = None
            self._logger.debug("verifier start receipt dir failed: %s", e)
        return self._receipt_after(
            return_code, service, folder / f"verifier-started-{attempt}"
        )

    def _attempt_folder_command(self, folder: PurePosixPath, service: str) -> str:
        """Root's command for one attempt's receipt folder *folder*
        (:func:`_root_receipt_dir_command`), in every integrity mode: root's
        and 0755, or, for a verifier with a user of its own whose uid the
        outputs folder's close read (:meth:`_close_output_dir`), that uid's
        (``chown``, the image's, on the verifier's PATH). A uid not read
        leaves the folder root's: a verifier that is not root then cannot
        write its receipt, a start never confirmed, never a success. A
        verifier that runs as the agent's own uid writes nothing the agent
        could not, as for its outputs folder."""
        command = _receipt_command(self._sandbox, folder, True, service)
        uid = None if self.receipt_root_only else self._verifier_uid_read
        if isinstance(uid, str) and uid.isdigit() and uid != "0":
            command = f"{command} && chown {uid} {shlex.quote(str(folder))}"
        return command

    def _receipt_after(
        self, return_code: int | None, service: str, receipt: PurePosixPath
    ) -> str | None:
        """The start-receipt path *receipt* once its folder's exec answered
        *return_code* (``None``: it raised); a warning and no receipt unless
        it is 0."""
        if return_code != 0:
            self._logger.warning(
                "Cannot create %s for the verifier start receipt; startup of "
                "this verifier cannot be confirmed",
                receipt.parent,
            )
            return None
        self.execution_receipt = (str(receipt), service)
        return str(receipt)

    async def _chmod_and_prepare_receipt(
        self, chmod_command: str, service: str
    ) -> str | None:
        """Chmod, then prepare the receipt, with their separate execs.

        A merged 10 s exec cannot keep chmod's 60 s deadline, and a lost
        answer must count toward its three tries. Keep the
        original retry loop and reach the best-effort receipt only on success.
        """
        await self._setup_step(
            "chmod",
            lambda: exec_raw(
                self._sandbox.exec,
                chmod_command,
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            ),
        )
        return await self._prepare_execution_receipt(service)

    # test-script verifier (default)

    async def _verify_test_script(
        self,
        *,
        strategy: VerifierStrategy | None = None,
        aggregate_policy: dict[str, Any] | None = None,
        cwd: str | None = None,
        script_timeout_sec: float | None = None,
        parse_rewards: bool = True,
    ) -> VerifierResult:
        """Run the task's ``test.sh`` verifier and return the reward result.

        ``[verifier].service`` selects which compose service ``test.sh`` runs
        in. The default ``"main"`` is the agent container.
        Multi-container (vulhub-style) tasks set it to a target/database
        service so the verifier can inspect *target-side* state — RCE markers,
        DB modifications — instead of only the agent workspace (#248).

        ``cwd`` runs the script in that folder, ``script_timeout_sec`` bounds it
        instead of ``[verifier].timeout_sec``, and ``parse_rewards=False`` stops
        after the script and its outputs are in (``test_return_code`` holds its
        exit status): a task.md draft 2 rubric is scored from the script's
        report, not from a reward file (``benchflow.taskmd.verify``).
        """
        self._finish_test = None
        self._save_outputs = None
        service = self._task.config.verifier.service
        verifier_outputs_are_mounted = service == "main" and getattr(
            self._sandbox, "is_mounted", False
        )
        if not verifier_outputs_are_mounted:
            await self._clear_output_dir(service)
        await self._close_output_dir(service, mounted=verifier_outputs_are_mounted)

        sandbox_paths = SandboxPaths()
        uses_native_verifier_dir = (
            getattr(self._task.paths, "uses_native_verifier_dir", False) is True
        )
        verifier_code_dir = (
            sandbox_paths.verifier_code_dir
            if uses_native_verifier_dir
            else sandbox_paths.tests_dir
        )
        if not await self._bundled_tests(verifier_code_dir, service):
            try:
                await self._sandbox.upload_dir(
                    source_dir=self._task.paths.tests_dir,
                    target_dir=str(verifier_code_dir),
                    service=service,
                    admission=admit(
                        self._sandbox,
                        service,
                        TRUSTED_TRANSFER,
                        reason="the task's tests",
                    ),
                )
            except VerifierOutputParseError:
                raise
            except Exception as e:
                raise AddTestsDirError(
                    "Failed to add verifier directory to sandbox."
                ) from e
            await self._make_verifier_code_roots(verifier_code_dir, service)

        await self._link_legacy_verifier_mount(
            verifier_code_dir=verifier_code_dir,
            sandbox_paths=sandbox_paths,
            service=service,
        )

        if not self._closed_to_host:
            # Root made it in the folder closed to the host (_close_output_dir).
            self._write_verifier_file(self._rollout_paths.test_stdout_path, "")

        for key in self._task.config.verifier.env or {}:
            if "api_key" in key.lower():
                self._logger.debug(
                    "The verifier.env contains an API key (often the case for LLM-"
                    "based verifiers). You will incur costs associated with the "
                    "API calls."
                )
        env = sandbox_verifier_env(self._task)
        # Registry test.sh scripts find their files through $TEST_DIR, the
        # folder the verifier package is mounted at: /tests for a tests/
        # package, /verifier for a native verifier/ one (#1179). A task's
        # [verifier.env] still sets its own. BenchFlow's default rides in the
        # command with the pins, not in the exec env: it is no secret, and an
        # exec env makes a non-root verifier need a private channel
        # (V3-INTEG-2).
        test_dir = (
            ""
            if env and "TEST_DIR" in env
            else f"export TEST_DIR={shlex.quote(str(verifier_code_dir))}; "
        )

        if strategy is not None:
            test_command = _script_strategy_command(strategy, verifier_code_dir)
            chmod_command = _script_strategy_chmod_command(strategy, verifier_code_dir)
        else:
            test_script_path = shlex.quote(
                str(
                    verifier_code_dir
                    / self._task.paths.test_path.relative_to(
                        self._task.paths.tests_dir
                    ).as_posix()
                )
            )
            # Under its first-line shebang, else bash, else sh: the shell the
            # task was written for, on every sandbox kind (script_run_command).
            test_command = script_run_command(test_script_path)
            chmod_command = f"chmod +x {test_script_path}"
            from benchflow.sandbox._bounded_exec import bfx_delivered, bfx_sh

            script_path = (
                verifier_code_dir
                / self._task.paths.test_path.relative_to(
                    self._task.paths.tests_dir
                ).as_posix()
            )
            if (
                service == "main"
                and bfx_delivered(self._sandbox)
                and script_path.is_absolute()
            ):
                # No program of the image runs for it.
                chmod_command = bfx_sh("chmod", str(script_path), set=0o111)
        if cwd is not None:
            test_command = f"cd {shlex.quote(cwd)} && {test_command}"
        # The verifier's pins as builtins, before test.sh's shell starts and
        # after whatever env the exec set up (benchflow._verifier_env).
        test_command = (
            pinned_shell_prefix(
                env, bash_env=self.bash_env, faithful=self._runs_as_written()
            )
            + test_dir
            + test_command
        )
        self._launch_record = self._launch_text = None
        launched = self._held_launch(service)
        test_stdout_path = shlex.quote(
            str(
                sandbox_paths.verifier_dir
                / self._rollout_paths.test_stdout_path.relative_to(
                    self._rollout_paths.verifier_dir
                ).as_posix()
            )
        )
        # Prepare the receipt after chmod; sidecars first need their log dir.
        receipt_prepared = (
            chmod_command is not None
            and service == "main"
            and self._write_execution_receipt
        )
        receipt: str | None = None
        if receipt_prepared:
            assert chmod_command is not None
            receipt = await self._chmod_and_prepare_receipt(chmod_command, service)
        elif chmod_command is not None:
            await self._setup_step(
                "chmod",
                lambda: exec_raw(
                    self._sandbox.exec,
                    chmod_command,
                    user="root",
                    service=service,
                    timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                ),
            )

        if service != "main":
            verifier_dir = shlex.quote(str(sandbox_paths.verifier_dir))
            await self._setup_step(
                "target verifier dir",
                lambda: exec_raw(
                    self._sandbox.exec,
                    f"mkdir -p {verifier_dir} && chmod 777 {verifier_dir}",
                    user="root",
                    service=service,
                    timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                ),
            )

        # test.sh runs the agent's code: when launched, it starts held, without
        # NET_ADMIN, NET_RAW or SYS_PTRACE. Its record goes beside the
        # start receipt, never in /logs/verifier, which task preflights require
        # to hold only what test.sh writes.
        if not receipt_prepared:
            receipt = await self._prepare_execution_receipt(service)
        if launched:
            test_command = await self._held_command(test_command, service, receipt)
        freeze = await self._freeze_step(service)
        command = f"{{ {test_command}; }} > {test_stdout_path} 2>&1"
        if receipt is not None:
            # A probe that found no receipt marks it abandoned; a command that
            # starts after that must not run the tests (_verifier_started).
            # Once the tests end, the command records their exit status beside
            # the receipt (``.exit``): a lost exec whose tests had finished is
            # scored from their outputs, not timed out (finish_test_command).
            # The receipt's name in the environment of every process the
            # command starts, so the kill of a timed-out attempt finds them
            # by it (VERIFIER_ATTEMPT_ENV).
            ended = shlex.quote(receipt + ".exit")
            attempt = shlex.quote(PurePosixPath(receipt).name)
            command = (
                f"export {VERIFIER_ATTEMPT_ENV}={attempt} && "
                f"printf started > {shlex.quote(receipt)} && "
                f"[ ! -e {shlex.quote(receipt + '.abandoned')} ] && "
                f"{{ {command}; _benchflow_status=$?; {freeze}"
                f"printf '%s' \"$_benchflow_status\" > {ended}; "
                "(exit $_benchflow_status); }"
            )
        elif freeze:
            command = (
                f"{{ {command}; _benchflow_status=$?; {freeze}"
                "(exit $_benchflow_status); }"
            )
        guard = pytest_plugin_guard_name((env or {}).get("PYTEST_ADDOPTS"))
        # What the host's copy of test.sh says of the pytest it starts
        # (sandbox/_guard_reach.py): whether the guard reaches every one, and
        # the -p names of each.
        pytest_reach = (
            pytest_reach_of_task(
                self._task, strategy.command if strategy is not None else None, cwd
            )
            if guard is not None
            else None
        )
        if guard is not None:
            # A previous attempt's markers are not this run's.
            markers_dir = shlex.quote(pytest_plugin_guard_markers_dir(guard))
            clear = f"rm -f {markers_dir}/* 2>/dev/null"
            if service == "main" and isinstance(
                getattr(self._sandbox, "runtime_kit", None), RuntimeKit
            ):
                # main's protected kit (every user may run it) holds the rm,
                # not test.sh's PATH; its own rm only if the kit has none
                # (merge wave 32b round-2 review A, L-2).
                clear = f"{{ {KIT_BIN}/{clear} || {clear}; }}"
            command = f"{clear}; {command}"
        # Whatever the runtime's mask, test.sh creates files 0644 and
        # directories 0755, which the plugin guard's ownership rule needs.
        command = with_verifier_umask(command)
        if self._agent_may_run(service):
            command, env = await self._env_by_root_file(command, env, service)
        if verifier_outputs_are_mounted:
            await self._require_writable_test_stdout(test_stdout_path, service)
        if self.before_test is not None:
            await self.before_test(str(verifier_code_dir))
        # test.sh is the task's own command (C8's task route): its shell
        # starts without the image's loader variables, which come back just
        # before it, under the verifier env's.
        after_test = functools.partial(
            self._after_test_command,
            sandbox_paths=sandbox_paths,
            service=service,
            guard=guard,
            pytest_reach=pytest_reach,
            pytest_addopts=(env or {}).get("PYTEST_ADDOPTS"),
            verifier_outputs_are_mounted=verifier_outputs_are_mounted,
            aggregate_policy=aggregate_policy,
        )
        if parse_rewards and receipt is not None:
            self._finish_test = functools.partial(after_test, parse_rewards=True)
        self._save_outputs = functools.partial(
            self._bring_in_outputs,
            sandbox_paths=sandbox_paths,
            service=service,
            verifier_outputs_are_mounted=verifier_outputs_are_mounted,
        )
        # An earlier test command's copy at its exit (a multi-step task's
        # earlier verify) is not this one's outputs: never put back over them.
        from benchflow.task import verifier_outputs

        verifier_outputs.forget(self._sandbox)
        with as_task_command(self._sandbox):
            self.test_command_issued = True
            self._issued_service = service
            self._sandbox._benchflow_test_command_issued = True
            ran = False
            try:
                test_result = await exec_raw(
                    self._sandbox.exec,
                    command=command,
                    env=env,
                    user=self._task.config.verifier.user,
                    service=service,
                    timeout_sec=(
                        script_timeout_sec
                        if script_timeout_sec is not None
                        else self._task.config.verifier.timeout_sec
                    ),
                )
                ran = _exec_return_code(test_result) != READ_FAILED
            except (TimeoutError, ExecTimeoutError):
                await self._cancel_freeze()
                raise
            except Exception as e:
                raise VerifierExecutionError(
                    f"Sandbox test-command exec failed: {describe_exception(e)}"
                ) from e
            finally:
                # The verifier env's file is not kept past its command.
                if env:
                    await remove_exec_env_files(self._sandbox)
                await self._remove_unread_root_env(ran)
        return await self._scored_under_network(
            after_test(_exec_return_code(test_result), parse_rewards=parse_rewards),
            service,
        )

    def _runs_as_written(self) -> bool:
        """Hardening ran test.sh's inputs as its authors wrote them
        (integrity off or audit): no pins in its command."""
        return getattr(self._sandbox, "_benchflow_verifier_faithful", False) is True

    def _held_launch(self, service: str) -> bool:
        """Whether the verifier's command in *service* starts through the
        bounded launch: main, or the service hardening held, unless
        hardening said this sandbox keeps its own network (a provider's
        separate verifier)."""
        from benchflow.sandbox.verifier_network import VerifierNetwork

        armed = getattr(self._sandbox, "_benchflow_verifier_network", None)
        if not isinstance(armed, VerifierNetwork) or armed.launch is False:
            # Hardening did not arm this sandbox's verifier network (a
            # verifier driven outside a rollout), or it keeps its own.
            return False
        return service == "main" or armed.where == f"service {service}"

    async def _held_command(
        self, command: str, service: str, receipt: str | None
    ) -> str:
        """*command* through :func:`verifier_launch`. Its record goes
        beside the start receipt *receipt*, else, on a hardened sandbox, in a
        fresh receipt folder of its own (one root exec); never in
        /logs/verifier, which task preflights require to hold only what
        test.sh writes. Where no folder can be made the launch's own check in
        the shell still refuses a held capability (exit 97)."""
        from benchflow.sandbox.verifier_network import verifier_launch

        record = f"{receipt}.launch" if receipt else None
        if record is None and _hardened_network(self._sandbox):
            record = await self._launch_record_folder(service)
        self._launch_record = record
        self._launch_text = None
        if record is None:
            return verifier_launch(command)
        # The record's name is new for each attempt: nothing to clear first,
        # and no program from the image's PATH runs as root for it.
        return verifier_launch(command, record=record)

    async def _launch_record_folder(self, service: str) -> str | None:
        """A launch record's path in its own receipt folder (no receipt
        state), or None when the folder's exec fails."""
        attempt = uuid4().hex
        folder = _EXECUTION_RECEIPT_DIR / f"verifier-{attempt}"
        command = self._attempt_folder_command(folder, service)
        try:
            result = await exec_raw(
                self._sandbox.exec,
                command,
                user="root",
                service=service,
                timeout_sec=_RECEIPT_DIR_BUDGET_SEC,
            )
            return_code = _exec_return_code(result)
        except Exception as e:
            return_code = None
            self._logger.debug("verifier launch record dir failed: %s", e)
        if return_code != 0:
            self._logger.warning(
                "No folder for test.sh's launch record (exit %s): the launch's "
                "own check is the only proof of this run",
                return_code,
            )
            return None
        return str(folder / f"verifier-launch-{attempt}")

    async def _check_launch_record(self, service: str) -> None:
        """Refuse a hardened run whose command did not start held: no record
        beside its receipt (a path that skipped the launch) or one showing
        NET_ADMIN, NET_RAW or SYS_PTRACE."""
        record = getattr(self, "_launch_record", None)
        if not isinstance(record, str) or not _hardened_network(self._sandbox):
            return
        from benchflow.sandbox.verifier_network import launch_record_problem

        async def read_record() -> str:
            try:
                # Builtins only: the record is one line.
                read = await exec_raw(
                    self._sandbox.exec,
                    f"bf_l=; IFS= read -r bf_l 2>/dev/null < {shlex.quote(record)}; "
                    'printf "%s\\n" "$bf_l"; true',
                    user="root",
                    service=service,
                    timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                )
                text = getattr(read, "stdout", "") or ""
            except Exception:
                text = ""
            return text if isinstance(text, str) else ""

        text = await read_record()
        for _ in range(2):
            # An empty answer is read again before it refuses: a
            # record really missing still reads empty and refuses.
            if text.strip():
                break
            await asyncio.sleep(0.5)
            text = await read_record()
        problem = launch_record_problem(text)
        if problem and not text.strip():
            # A launch that refused before test.sh (no setpriv with
            # --bounding-set and no kit Python) says why first in its output.
            with contextlib.suppress(Exception):
                with destination(self._rollout_paths.test_stdout_path).open("rb") as f:
                    head = f.read(4096).decode("utf-8", "replace")
                said = next(
                    (
                        ln.partition("refused: ")[2]
                        for ln in head.splitlines()
                        if ln.startswith("BF verifier-launch refused: ")
                    ),
                    "",
                )
                if said:
                    problem = f"the bounded launch refused: {said}"
        if problem:
            raise VerifierLaunchRefusedError(problem)
        self._launch_text = text.strip().splitlines()[-1]

    async def _scored_under_network(
        self, scoring: Awaitable[VerifierResult], service: str
    ) -> VerifierResult:
        """test.sh's scoring, then its network's verdict (:meth:`_network_verdict`).
        A test.sh that left no usable reward after refused packets keeps that
        ending ("verifier wrote no usable reward", a failure), with the
        refusals recorded."""
        try:
            result = await scoring
        except (
            RewardFileNotFoundError,
            RewardFileEmptyError,
            RewardFileParseError,
            VerifierOutputParseError,
        ) as exc:
            await self._network_verdict(
                VerifierResult(rewards=None), service, cause=exc
            )
            raise
        return await self._network_verdict(result, service)

    async def _network_verdict(
        self,
        result: VerifierResult,
        service: str,
        *,
        cause: BaseException | None = None,
    ) -> VerifierResult:
        """test.sh's reward under the verifier's network: its refused
        packets go to ``verifier/verifier-network.json``, and any refusal to
        :attr:`network_refusal` with test.sh's exit and whether it passed.
        The reward stands as test.sh gave it. A refused packet can come from
        test.sh's own code or from code the agent left that test.sh runs, as
        root in the same phase, and BenchFlow can't tell them apart: the
        trial is scored and flagged for a read, never unscored for it."""
        armed_here = getattr(self._sandbox, "_benchflow_verifier_network", None)
        if not self._network_judged(service, armed_here):
            return result
        from benchflow.sandbox.verifier_network import network_refusal_ends

        armed, refused = await self._end_of_test_proof(service, armed_here)
        if armed is None or refused <= 0 or self.network_refusal is None:
            return result
        reward = None if cause is not None else (result.rewards or {}).get("reward")
        exit_code = getattr(self, "test_return_code", None)
        self.network_refusal.update(
            exit_code=exit_code,
            passed=not network_refusal_ends(refused, exit_code, reward),
        )
        if cause is not None:
            self.network_refusal["no_usable_reward"] = True
        return result

    @staticmethod
    def _network_judged(service: str, armed: Any) -> bool:
        """Whether a test command in *service* gets a network verdict: in
        main, or in the ``verifier.service`` hardening armed."""
        return service == "main" or getattr(armed, "where", "") == f"service {service}"

    async def _end_of_test_proof(
        self, service: str, armed_here: Any, *, timed_out: bool = False
    ) -> tuple[Any, int]:
        """BenchFlow's rules proven again at the end of the test command, around the count of its network's refusals, and the
        verdict's record, ``verifier/verifier-network.json``. A rule gone, a
        counter back, a check not made or a connect that got through ends the
        run "network policy breached" (:meth:`_policy_breached`), never
        "held, refused 0". Returns what hardening armed (None: no record) and
        the refusals counted since the arm."""
        from benchflow.sandbox import policy_reproof
        from benchflow.sandbox.verifier_network import (
            VERIFIER_NETWORK_RECORD,
            VerifierNetwork,
            refused_since_arm,
        )

        proof = await policy_reproof.reprove(
            self._sandbox,
            policy_reproof.END_OF_TEST,
            armed=armed_here if isinstance(armed_here, VerifierNetwork) else None,
        )
        if proof is not None and not proof.proven:
            await self._policy_breached(proof, armed_here)
        armed, refused = await refused_since_arm(self._sandbox)
        if proof is not None:
            # After the count: a refused TEST-NET connect counts at the final
            # REJECT, and it is not test.sh's.
            await policy_reproof.connect_from_launch(self._sandbox, proof, service)
            if not proof.proven:
                await self._policy_breached(proof, armed)
        if armed is None:
            return None, refused
        if refused > 0:
            self.network_refusal = {
                "phase": "test.sh",
                "where": armed.where,
                "mode": armed.mode,
                "refused": refused,
                "refused_words": armed.refusal_words(refused),
                **({"timed_out": True} if timed_out else {}),
            }
        try:
            record = armed.record(refused)
            launch = getattr(self, "_launch_text", None)
            if isinstance(launch, str) and launch:
                record["launch"] = launch
            if proof is not None:
                record["policy"] = {
                    # Unproven only for held services that run nothing: one
                    # restarted, or one exited (policy_reproof.with_held_restarts).
                    "state": "held"
                    if proof.proven
                    else "not proven, held after a restart or exited",
                    "proven_at": proof.checked_at,
                    "rules": proof.record()["rules"],
                }
            if timed_out:
                record["timed_out"] = True
            self._write_verifier_file(
                self._rollout_paths.verifier_dir / VERIFIER_NETWORK_RECORD,
                json.dumps(record, sort_keys=True) + "\n",
            )
        except OSError as e:
            self._logger.warning(
                "could not write verifier/verifier-network.json: %s",
                describe_exception(e),
            )
        return armed, refused

    async def timed_out_network_verdict(self) -> None:
        """The end-of-test re-proof of a test command killed at its deadline.

        A scored timeout used to skip it: test.sh could write reward
        1, remove a rule and outlive its timeout, and the trial scored a plain
        0 with no word of the breach. The rollout calls this once it has killed the
        tests and kept their outputs: the rules are proven again as at the end
        of a test command that returned (:meth:`_end_of_test_proof`). A breach
        raises :class:`NetworkPolicyBreachedError`, the ending "network policy
        breached" (unscored). A clean proof leaves the timeout's 0 as it was,
        whatever the network refused: its record says ``timed_out``, and a
        refusal is flagged as after a returned test command.
        """
        service = self._issued_service
        armed = getattr(self._sandbox, "_benchflow_verifier_network", None)
        if service is None or not self._network_judged(service, armed):
            return
        await self._end_of_test_proof(service, armed, timed_out=True)

    async def _policy_breached(self, proof: Any, armed: Any) -> None:
        """Record the breach in ``verifier/verifier-network.json`` (never
        "held") and end the run "network policy breached"."""
        from benchflow.sandbox import policy_reproof
        from benchflow.sandbox.verifier_network import (
            VERIFIER_NETWORK_RECORD,
            VerifierNetwork,
        )

        policy = proof.record()
        record: dict[str, Any] = {"state": policy["state"], "policy": policy}
        if isinstance(armed, VerifierNetwork):
            record = {"where": armed.where, "mode": armed.mode, **record}
        with contextlib.suppress(OSError):
            self._write_verifier_file(
                self._rollout_paths.verifier_dir / VERIFIER_NETWORK_RECORD,
                json.dumps(record, sort_keys=True) + "\n",
            )
        error = await policy_reproof.with_held_restarts(self._sandbox, proof)
        if error is not None:
            raise error

    @property
    def test_command_finishable(self) -> bool:
        """Whether :meth:`finish_test_command` can score the issued test
        command: a script verifier whose command records its end."""
        return self._finish_test is not None

    async def finish_test_command(self, return_code: int) -> VerifierResult:
        """Score a test command that ended although its exec never returned.

        Once its tests end, the command records their exit status beside its
        start receipt; the rollout reads that record when the exec outlives
        the verifier's budget (``_verifier_finished``), cancels it, and calls
        this, which runs what verify() runs after the exec: the output
        download, the plugin guard's markers and the reward rules.
        test.sh's user can write that record, as it can end test.sh after
        writing a reward; either way the reward files are scored by the same
        rules.
        """
        finish = self._finish_test
        if finish is None:
            raise VerifierExecutionError("no test command of this run to score")
        return await self._scored_under_network(finish(return_code), "main")

    @property
    def test_command_savable(self) -> bool:
        """Whether :meth:`save_timed_out_outputs` has an issued test command
        whose outputs it can keep."""
        return self._save_outputs is not None

    async def save_timed_out_outputs(self) -> None:
        """Keep what a test command still running at its deadline wrote.

        The rollout calls this once it has killed the tests (a scored
        timeout): the outputs come into ``verifier/`` as a finished command's
        do, file by file when the folder's download fails, and each reward
        file there is renamed ``timeout-<name>``, since a reward written
        before the deadline is no verdict.
        """
        save = self._save_outputs
        if save is None:
            raise VerifierExecutionError("no test command of this run to save")
        failure: OSError | None = None
        try:
            await save(recover_all=True)
        finally:
            for path in (
                self._rollout_paths.reward_text_path,
                self._rollout_paths.reward_json_path,
                self._rollout_paths.reward_details_json_path,
                self._rollout_paths.criteria_runner_manifest_path,
            ):
                try:
                    _set_aside(path)
                except OSError as exc:
                    failure = failure or exc
        if failure is not None:
            raise failure

    async def _bring_in_outputs(
        self,
        *,
        sandbox_paths: SandboxPaths,
        service: str,
        verifier_outputs_are_mounted: bool,
        recover_all: bool = False,
    ) -> None:
        """Bring the test command's outputs into the rollout's ``verifier/``.

        Mounted outputs are there already and only change owner; others are
        downloaded. When the download fails, ``main``'s canonical outputs are
        copied one by one: the run needs a reward file among them, unless
        *recover_all* takes whatever is there.
        """
        if verifier_outputs_are_mounted:
            await self._hand_mounted_outputs_to_host(sandbox_paths, service)
            await self._open_output_dir_to_host(service)
            return
        try:
            await self._download_verifier_dir(sandbox_paths, service)
        except Exception as e:
            if isinstance(
                e, (ReachableWriteError, UnsafeDownloadEntryError)
            ) and not verifier_infrastructure_failure(e):
                raise VerifierOutputParseError(
                    f"Unsafe verifier output capture: {e}"
                ) from e
            if service != "main" or not (
                await self._recover_main_verifier_outputs(sandbox_paths) or recover_all
            ):
                raise DownloadVerifierDirError(
                    "Failed to download verifier directory from sandbox: "
                    + verifier_failure_reason(e)
                ) from e
            self._logger.warning(
                "Verifier directory download failed; recovered canonical "
                "verifier output files individually: %s",
                e,
            )

    async def _after_test_command(
        self,
        test_return_code: int,
        *,
        sandbox_paths: SandboxPaths,
        service: str,
        guard: str | None,
        verifier_outputs_are_mounted: bool,
        parse_rewards: bool,
        aggregate_policy: dict[str, Any] | None,
        pytest_reach: PytestReach | None = None,
        pytest_addopts: str | None = None,
    ) -> VerifierResult:
        """The test command has ended with *test_return_code*: bring in its
        outputs, read the plugin guard's markers and score the rewards.
        *pytest_reach* is what the host's copy of test.sh says of the pytest
        it starts (sandbox/_guard_reach.py)."""
        self.test_return_code = test_return_code

        # Download verifier output if it is not host-mounted. Only the agent's
        # ``main`` container has the rollout dir bind-mounted; a target service
        # never does, so target-side rewards (#248) are always downloaded.
        if verifier_outputs_are_mounted:
            # Mounted outputs are on the host already: copied before the
            # folder goes to the host user (_open_output_dir_to_host), and
            # the sandbox goes on once they are (verifier_freeze).
            try:
                await self._let_host_read_outputs(sandbox_paths, service)
                self._take_outputs_at_exit()
            finally:
                await self.thaw_after_test()
            await self._bring_in_outputs(
                sandbox_paths=sandbox_paths,
                service=service,
                verifier_outputs_are_mounted=verifier_outputs_are_mounted,
            )
        else:
            # Downloaded, then copied; the sandbox goes on once they are.
            try:
                await self._bring_in_outputs(
                    sandbox_paths=sandbox_paths,
                    service=service,
                    verifier_outputs_are_mounted=verifier_outputs_are_mounted,
                )
                self._take_outputs_at_exit()
            finally:
                await self.thaw_after_test()
        await self._check_launch_record(service)

        markers = (
            await self._plugin_guard_markers(guard, service)
            if guard is not None
            else {}
        )
        # The guard's signed verdicts record ((g2)).
        verdicts, unread = (
            read_guard_verdicts(markers.get("verdicts", []), guard)
            if guard is not None
            else ([], None)
        )
        self._guard_nominated = (
            (
                guard,
                _guard_nomination(
                    self._task, self._sandbox, guard, pytest_reach, pytest_addopts
                ),
            )
            if guard is not None
            else None
        )
        checking = markers.get("checking", [])
        if (
            checking
            and not markers.get("refused")
            and not markers.get("crashed")
            # A verdict written means it decided, though pytest stopped
            # before its refused marker ((g2)).
            and not verdicts
        ):
            # A guard stopped while it checked plugins (killed, or blocked on a
            # FIFO until test.sh's timeout) recorded no verdict.
            raise PluginGuardLoadError(
                f"pytest stopped while the plugin guard {guard} was still "
                "checking pytest plugins, so it never decided whether to trust "
                f"them; listed in verifier/{checking[0].name}; this run is not "
                "scored",
                record="the guard stopped while checking pytest plugins and left "
                "no verdict",
                rejudge=True,
            )
        if guard is not None and (
            # Left by a guard pytest imported but never registered, whether
            # or not test.sh kept pytest's output.
            markers.get("loading")
            # Both the message and a signed registered marker can be forged
            # by test.sh's user, which can read the signing key. A marker
            # therefore cannot cancel a reported load failure (FIX-49 r4).
            or _has_guard_load_failure(self._rollout_paths.verifier_dir, guard)
        ):
            # pytest aborted before running a test, so a reward test.sh derived
            # from its exit status would score the harness, not the solution.
            # Hardening names the Pythons that took no copy: not in the
            # pre-agent record, or the agent's files (PT-C8-CONF-2b round 3).
            unrecorded = getattr(self._sandbox, "_runtime_guard_unrecorded", ())
            # Why each took none, from hardening (its log line is DEBUG).
            why = getattr(self._sandbox, "_runtime_guard_why", None)
            why = why if isinstance(why, dict) else {}
            unrecorded_note = (
                "; these Pythons took no guard copy and reach it through "
                "PYTHONPATH only: "
                + ", ".join(
                    f"{python} ({why[python]})" if why.get(python) else python
                    for python in unrecorded
                )
                if isinstance(unrecorded, tuple) and unrecorded
                else ""
            )
            raise PluginGuardLoadError(
                f"pytest could not load the plugin guard {guard}: the verifier "
                "ran pytest in a Python that cannot import or register it (for "
                "example `python -I` or a replaced PYTHONPATH)"
                f"{unrecorded_note}; its reward is not scored",
                guard_copy_diagnostic=unrecorded_note,
                rejudge=True,
            )
        if (
            guard is not None
            and not markers
            and pytest_reach is not None
            and pytest_reach.expects_record
        ):
            # test.sh starts pytest where BenchFlow's PYTEST_ADDOPTS reaches it
            # (the host's copy of its text, which neither the agent nor
            # test.sh's user can change), yet no pytest of this run left a
            # marker: code test.sh ran stopped pytest, or kept the guard out,
            # before it loaded. Its output proves nothing either way. The
            # reward is not test.sh's: a recorded 0 that says why, until the
            # host re-judges the stop on the pristine image, where a guard
            # that gives no verdict either leaves the trial unscored (the
            # lead, 2026-11-16 13:57Z).
            started = pytest_reach.direct[0]
            self._guard_stop_has_no_record()
            raise PluginGuardLoadError(
                f"no pytest of this run reached the pytest plugin guard {guard}, "
                f"though test.sh starts pytest where it reaches (`{started}`): "
                "the reward test.sh wrote does not stand",
                record=(
                    "the guard left no record, though test.sh starts pytest "
                    f"(`{started}`)"
                ),
                rejudge=True,
            )
        if guard is not None and not markers:
            # No pytest of this run reached the guard: test.sh ended before
            # pytest (a failed install), checked without pytest, or ran it
            # where BenchFlow's environment does not reach (ssh, a nested VM,
            # another container or user). That is no evidence against the
            # solution, so test.sh's reward is scored as it wrote it
            # (GUARD-LOAD). Code test.sh runs before pytest could switch the
            # guard off, but it could as well keep the guard and set the
            # result; the task's script is trusted (contract B). DEBUG: a
            # task without pytest gets it on every run (dogfood p1-5). A
            # test.sh that starts pytest where the guard reaches it is not
            # here (above).
            self._logger.debug(
                "No marker for the pytest plugin guard %s was read; test.sh's "
                "reward is scored as it wrote it (guard loading is not confirmed)",
                guard,
            )
        crashed = markers.get("crashed", [])
        if crashed:
            # The guard failed for its own reasons (a refused plugin leaves a
            # `refused` marker instead), so pytest's exit status says nothing
            # about the solution.
            lines = crashed[0].detail.strip().splitlines()
            raise PluginGuardLoadError(
                f"the pytest plugin guard {guard} crashed inside pytest"
                + (f" ({lines[-1].strip()[:300]})" if lines else "")
                + f"; traceback in verifier/{crashed[0].name}; its reward is not "
                "scored",
                record=unconfirmed_guard_verdicts(verdicts, unread)
                if verdicts or unread is not None
                else "",
                rejudge=True,
            )
        installed = markers.get("installed", [])
        if installed:
            # The guard refused a plugin the verifier installed after the agent
            # stopped: the task's problem, not the solution's 0.
            detail = installed[0].detail
            plugins, environments = installed_marker_plugins(detail)
            what = "pytest plugin " + (", ".join(plugins) if plugins else "code")
            where = f" into {', '.join(environments)}" if environments else ""
            task_dir = getattr(getattr(self._task, "paths", None), "task_dir", None)
            check = f"bench check {command_arg(task_dir, '<task folder>')}"
            example = plugins[0].replace(" ", "==") if plugins else "<plugin>"
            raise PluginGuardLoadError(
                f"{what} was installed after the agent stopped, by the "
                f"verifier{where}, where the agent could write; the pytest "
                f"plugin guard {guard} refused it, so this run is not scored, "
                "and neither is any run of this task, the reference solution's "
                f"included. This is a task problem: `{check}` flags it. Install "
                "the plugin into the image, or pin its release in [verifier] "
                f'pytest_plugin_releases (for example `["{example}"]`), '
                "which BenchFlow installs into its protected verifier runtime "
                "before the agent's turn. Refused: "
                f"{describe_installed_marker(detail)} (listed in "
                f"verifier/{installed[0].name})"
            )
        refused = markers.get("refused", [])
        unlisted = markers.get("unlisted", [])
        unproven = markers.get("unproven", [])
        if (unlisted or unproven) and _marker_tokens(refused) <= _marker_tokens(
            [*unlisted, *unproven]
        ):
            # Every refusal was of a plugin test.sh installed after the agent
            # stopped, from inputs the agent changed (V2-R1-B-2), or of a
            # vouched release the host's table lacks (V2-R1-B-1): the guard
            # cannot vouch for it, so it still counts as a failure, but its
            # opening words keep it from being the agent's. Only the guard's
            # fixed words decide which.
            detail = (unproven or unlisted)[0].detail
            plugins, _ = installed_marker_plugins(detail)
            what = ", ".join(plugins) if plugins else "a pytest plugin"
            lines = refused[0].detail.strip().splitlines() if refused else []
            where = f" (listed in verifier/{(unproven or unlisted)[0].name})"
            if unproven:
                example = plugins[0].replace(" ", "==") if plugins else "<plugin>"
                raise PluginGuardLoadError(
                    f"the pytest plugin guard {guard} refused a pytest plugin it "
                    "could not trust, which the verifier installed after the "
                    "agent stopped from inputs the agent could have changed, so "
                    "pytest stopped and this run is not scored"
                    + (f" ({lines[0].strip()[:500]})" if lines else "")
                    + f"; {what}{where}. A task problem: BenchFlow trusts such "
                    "an install only as a release it can check byte for byte. "
                    "Pin it in [verifier] pytest_plugin_releases (for example "
                    f'`pytest_plugin_releases = ["{example}"]`) or install it '
                    "into the image"
                )
            raise PluginGuardLoadError(
                f"the pytest plugin guard {guard} refused a pytest plugin it "
                "could not trust, a release of a plugin BenchFlow vouches for "
                "that its table of published releases lacks, so pytest stopped "
                "and this run is not scored"
                + (f" ({lines[0].strip()[:500]})" if lines else "")
                + f"; {what}{where} was installed by the verifier after the "
                "agent stopped, and this machine could not read that release "
                "from PyPI before the verifier ran, or PyPI published it since"
            )
        unrecorded = markers.get("unrecorded", [])
        if unrecorded:
            # A refusal with no record to tell test.sh's own install from a
            # file the agent left (V2-R2-A-1): it counts as a failure, as any
            # refusal, but its opening words keep it from being the agent's.
            # Only the guard's fixed words for why go into it.
            why = unrecorded[0].detail.partition("\n")[0]
            if why not in _UNRECORDED_WHY:
                why = "it could not be read"
            lines = refused[0].detail.strip().splitlines() if refused else []
            raise PluginGuardLoadError(
                f"the pytest plugin guard {guard} refused a pytest plugin it "
                "could not trust, with no complete record of the agent's files "
                f"({why}) to tell whether the verifier installed it after the "
                "agent stopped or the agent left it, so pytest stopped and this "
                "run is not scored"
                + (f" ({lines[0].strip()[:500]})" if lines else "")
                + f"; files listed in verifier/{unrecorded[0].name}. When "
                "test.sh installs pytest plugins, pin their releases in "
                "[verifier] pytest_plugin_releases or install them into the "
                "image (published pytest-json-ctrf releases load wherever "
                "installed)"
            )
        rejection = agents_rejection(verdicts)
        if rejection is not None:
            # The guard's verdicts record names a file it refused that the
            # image does not have as it is ((g2)): the agent's doing, so
            # a 0 that names it, whatever pytest's exit (an INTERNALERROR
            # included, which can leave no refused marker). The record is
            # test.sh's user's to forge, and a forged one only lowers that
            # trial's own score.
            raise PluginGuardRejectedError(rejection)
        if refused or verdicts or unread is not None:
            # The guard refused a plugin it could not trust: pytest
            # stopped before any test ran, so test.sh's 0 scores the guard, not
            # the solution, and a guard false positive would pass as a fair 0.
            # Its record names no file the image lacks (the image's own file,
            # the guard's own error, or no verdict): a recorded 0 until the
            # host re-judges the files on the pristine image ((g2)).
            lines = refused[0].detail.strip().splitlines() if refused else []
            raise PluginGuardLoadError(
                f"the pytest plugin guard {guard} "
                + (
                    "refused a pytest plugin it could not trust"
                    if refused
                    else "left verdicts without a refusal"
                )
                + ", so pytest stopped and this run is not scored"
                + (f" ({lines[0].strip()[:500]})" if lines else "")
                + (f"; listed in verifier/{refused[0].name}" if refused else "")
                + f"; {unconfirmed_guard_verdicts(verdicts, unread)}",
                record=unconfirmed_guard_verdicts(verdicts, unread),
                rejudge=True,
            )

        if not parse_rewards:
            return VerifierResult(rewards=None)

        with self._empty_reward_named("test.sh", test_return_code):
            if self._reward_file("reward.json").exists():
                rewards = self._parse_reward_json_with_text_compat(
                    aggregate_policy=aggregate_policy,
                )
            elif self._reward_file("reward.txt").exists():
                rewards = self._parse_reward_text()
            else:
                rewards = None
        if rewards is not None and test_return_code != 0:
            # Verifier produced a reward file but exited nonzero.
            # This is common when test frameworks exit with the count of
            # failures (e.g. pytest exits 1 when all tests fail → reward 0).
            # Since _clear_reward_outputs() wiped stale files before this run,
            # any reward file present was written by THIS invocation. Accept
            # the reward so the result is classified as "failed" (honest model
            # failure) instead of "verifier_errored" (infrastructure problem).
            # Said only once the reward has parsed (DF-p2-task-author-11).
            self._logger.warning(
                "Verifier exited with rc=%d after writing its reward; the reward "
                "is scored (reward files were cleared before this run)",
                test_return_code,
            )
        if rewards is None:
            if test_return_code != 0:
                msg = (
                    f"verifier exited with rc={test_return_code}; no reward file "
                    f"found at verifier/reward.txt or "
                    f"verifier/reward.json"
                )
            else:
                msg = (
                    "No reward file found at verifier/reward.txt or "
                    "verifier/reward.json"
                )
            # Surface ONLY a fixed, secret-free dep-install diagnostic (never
            # any scanned stdout) so classify_verifier_error can return
            # VERIFIER_DEP_INSTALL without leaking untrusted subprocess output
            # into result metadata (#572). Raw resolver output remains in the
            # downloaded verifier/test-stdout.txt artifact.
            if _has_dep_install_failure(self._rollout_paths.test_stdout_path):
                msg += f"\n{_DEP_INSTALL_DIAGNOSTIC}"
            raise RewardFileNotFoundError(msg)

        if guard is not None and markers and _passes(rewards):
            # In strict a pass stands only on the guard's end-of-session
            # records ((g2), the lead's 15:41Z ruling): every
            # pytest the guard loaded in left one, and they agree with it.
            why = _guard_end_problem(guard, markers, pytest_reach)
            if why is not None:
                self._guard_stop_has_no_record()
                raise PluginGuardLoadError(
                    f"{why}, so the reward test.sh wrote does not stand",
                    record=why,
                    rejudge=True,
                )
        return VerifierResult(rewards=rewards)

    def _guard_stop_has_no_record(self) -> None:
        """The pristine re-judgment judges a run with no whole record of the
        guard (sandbox/_guard_rejudge.py, ``no_record``)."""
        nominated = self._guard_nominated
        if nominated is not None:
            self._guard_nominated = (
                nominated[0],
                dataclasses.replace(nominated[1], no_record=True),
            )

    async def _plugin_guard_markers(
        self, guard: str, service: str
    ) -> dict[str, list[GuardMarker]]:
        """The markers the plugin guard *guard* signed during this run, by kind.

        They are read from the guard's own directory in the sandbox, and
        copied to the trial's ``verifier/`` for reference. When they cannot
        all be read or validated, a refusal may be among those missed, so
        the scoring boundary gives 0 with a neutral warning. An adverse
        marker that was read still prevents a pass. They are read from a
        receipt (``_exec_receipt.py``): a backend that rewrites exec output
        (remote Docker's redaction) could otherwise change a refusal marker's
        body, which then fails its signature and is dropped.

        A read whose output did not come back whole (its exec raised, or ended
        with a failed status: a timeout or a lost exec on a loaded host) is
        tried again, up to three tries in all. It only reads, so a passing
        reward is never lost to one failed exec, which said nothing about the
        markers. Output that did come back is decided as it is, a missing or
        overfull marker folder included (test.sh's user can make those). When
        no try brought output back and the last failed in transport or in the
        controller, the verifier errors as it does for any of its steps; any
        other failure scores 0 as unreadable markers, naming the cause.
        """
        problem = None
        output = ""
        failure: Exception | None = None
        tries = 1 + len(_GUARD_MARKER_RETRY_DELAYS_SEC)
        for delay in (0.0, *_GUARD_MARKER_RETRY_DELAYS_SEC):
            if delay:
                self._logger.warning(
                    "Reading the pytest plugin guard %s's markers failed (%s); "
                    "retrying in %.0fs",
                    guard,
                    str(problem)[:300],
                    delay,
                )
                await asyncio.sleep(delay)
            failure = None
            try:
                result = await receipt_exec(
                    self._sandbox,
                    pytest_plugin_guard_markers_cmd(guard),
                    user="root",
                    service=service,
                    timeout_sec=30,
                )
            except Exception as e:
                failure = e
                problem = f"{type(e).__name__}: {e}"
                continue
            # A failed status's output still counts when no try does better:
            # a verdict read from it decides, as before.
            output = result.stdout or ""
            if _exec_return_code(result) != 0:
                problem = f"exit {_exec_return_code(result)}"
                continue
            problem = pytest_plugin_guard_markers_problem(output)
            break
        else:
            if (
                not output
                and failure is not None
                and verifier_infrastructure_failure(failure)
            ):
                raise VerifierInfrastructureError(
                    f"the pytest plugin guard {guard}'s markers could not be read "
                    f"in {tries} tries"
                ) from failure
            problem = f"{tries} tries, the last: {problem}"
        by_kind: dict[str, list[GuardMarker]] = {}
        markers = parse_pytest_plugin_guard_markers(output, guard)
        names = {marker.name for marker in markers}
        if any(
            line.startswith(guard + ".") and line.partition("\t")[0] not in names
            for line in output.splitlines()
        ):
            problem = "a guard marker failed validation"
        for marker in markers:
            by_kind.setdefault(marker.kind, []).append(marker)
            if marker.kind != "registered":
                try:
                    self._write_verifier_file(
                        self._rollout_paths.verifier_dir / marker.name, marker.detail
                    )
                except (OSError, ValueError) as e:
                    self._logger.debug(
                        "Cannot copy guard marker %s: %s", marker.name, e
                    )
        if problem is not None:
            if any(kind != "registered" for kind in by_kind):
                self._logger.warning(
                    "Plugin guard %s's markers were not all read (%s); "
                    "deciding on the verdicts that were",
                    guard,
                    str(problem)[:300],
                )
                return by_kind
            raise PluginGuardMarkersUnreadableError(
                f"BenchFlow could not read back the markers of the pytest "
                f"plugin guard {guard} ({str(problem)[:300]}), so a plugin it "
                "refused may be among those missed"
            )
        return by_kind

    async def _recover_main_verifier_outputs(
        self,
        sandbox_paths: SandboxPaths,
    ) -> bool:
        candidates = [
            (sandbox_paths.reward_json_path, self._rollout_paths.reward_json_path),
            (sandbox_paths.reward_text_path, self._rollout_paths.reward_text_path),
            (
                sandbox_paths.reward_details_json_path,
                self._rollout_paths.reward_details_json_path,
            ),
            (
                sandbox_paths.criteria_runner_manifest_path,
                self._rollout_paths.criteria_runner_manifest_path,
            ),
            (
                sandbox_paths.verifier_dir
                / self._rollout_paths.test_stdout_path.relative_to(
                    self._rollout_paths.verifier_dir
                ).as_posix(),
                self._rollout_paths.test_stdout_path,
            ),
            (
                sandbox_paths.verifier_dir
                / self._rollout_paths.test_stderr_path.relative_to(
                    self._rollout_paths.verifier_dir
                ).as_posix(),
                self._rollout_paths.test_stderr_path,
            ),
            (
                sandbox_paths.verifier_dir / "ctrf.json",
                self._rollout_paths.verifier_dir / "ctrf.json",
            ),
        ]
        recovered: list[ReachablePath] = []
        verifier_dir = self._rollout_paths.verifier_dir
        await self._read_back(str(sandbox_paths.verifier_dir), "main")
        with private_folder(verifier_dir) as private:
            for index, (remote_path, local_path) in enumerate(candidates):
                landing = private / f"{index}"
                try:
                    try:
                        await self._sandbox.download_file(str(remote_path), landing)
                    except FileNotFoundError:
                        # Only an explicitly absent remote companion may be
                        # omitted, not a failed landing or publication.
                        continue
                    if not landing.is_file() or landing.is_symlink():
                        raise ReachableWriteError(
                            "downloaded entry is not a regular file"
                        )
                    local_path.publish(landing)
                except Exception as exc:
                    if index < 2:
                        if isinstance(
                            exc,
                            (
                                ReachableWriteError,
                                UnsafeDownloadEntryError,
                                PermissionError,
                                IsADirectoryError,
                                NotADirectoryError,
                            ),
                        ) and not verifier_infrastructure_failure(exc):
                            raise RewardFileParseError(
                                f"Unsafe or unreadable reward entry {remote_path}: {exc}"
                            ) from exc
                        raise DownloadVerifierDirError(
                            f"Failed to capture reward entry {remote_path}: "
                            + verifier_failure_reason(exc)
                        ) from exc
                    self._logger.debug(
                        "Could not recover verifier output %s: %s",
                        remote_path,
                        exc,
                    )
                    continue
                recovered.append(local_path)

        return (
            self._rollout_paths.reward_json_path in recovered
            or self._rollout_paths.reward_text_path in recovered
        )

    # llm-judge verifier (#270)

    def _resolve_rubric_path(
        self,
        strategy: VerifierStrategy | None = None,
    ) -> Path:
        """Locate the rubric file relative to the task directory."""
        judge = self._task.config.verifier.judge
        raw_rubric_path = strategy.rubric_path if strategy is not None else None
        rubric_path = Path(raw_rubric_path or judge.rubric_path)
        if not rubric_path.is_absolute():
            base_dir = (
                Path(self._task.paths.tests_dir)
                if strategy is not None
                else Path(self._task.paths.task_dir)
            )
            rubric_path = base_dir / rubric_path
        if not rubric_path.exists():
            raise RubricNotFoundError(
                f"llm-judge rubric not found at {rubric_path}. Set "
                "verifier.strategies.<name>.rubric in verifier.md or "
                "[verifier.judge].rubric_path in task.toml."
            )
        return rubric_path

    @asynccontextmanager
    async def _download_deliverables(
        self,
        strategy: VerifierStrategy | None = None,
    ) -> AsyncIterator[Path]:
        """Capture native judge evidence through the bounded archive protocol.

        Keep the capture private until the caller finishes judging. The
        verifier output directory is mounted into the live sandbox, so neither
        extraction nor subsequent reads may use it as a custody root.
        """
        input_dir = _llm_judge_input_dir(strategy, self._task.config.verifier.judge)
        from benchflow.taskmd.verify import copy_kept

        remote = PurePosixPath(input_dir)
        if not remote.is_absolute() or ".." in remote.parts or str(remote) == "/":
            raise DownloadVerifierDirError(
                "llm-judge input must be an absolute directory below the sandbox root"
            )
        with private_temporary_directory(prefix="benchflow-native-judge-") as tmp:
            dest = Path(tmp).resolve()
            if dest.is_relative_to(self._rollout_paths.rollout_dir.resolve()):
                raise DownloadVerifierDirError(
                    "Judge temporary directory is inside the sandbox mount; "
                    "set TMPDIR to a controller-private directory outside the job"
                )
            try:
                refused = await copy_kept(self._sandbox, [str(remote)], dest)
            except (TimeoutError, ExecTimeoutError):
                raise
            except VerifierOutputParseError:
                raise
            except Exception as e:
                raise DownloadVerifierDirError(
                    f"Failed to download llm-judge input from {input_dir}: "
                    + verifier_failure_reason(e)
                ) from e
            captured = dest.joinpath(*remote.parts[1:])
            if str(remote) in refused or not captured.is_dir():
                raise VerifierOutputParseError(
                    f"llm-judge input {input_dir} is missing, unsafe, or exceeds capture limits"
                )
            yield captured

    def _resolve_llm_judge_context(
        self,
        strategy: VerifierStrategy | None,
    ) -> str:
        judge = self._task.config.verifier.judge
        if strategy is not None:
            if strategy.context_file is not None:
                relative = _safe_strategy_relative_path(
                    strategy.context_file,
                    strategy=strategy,
                    field="context_file",
                )
                context_path = Path(self._task.paths.tests_dir) / Path(*relative.parts)
                if not context_path.is_file():
                    raise VerifierOutputParseError(
                        f"llm-judge strategy {strategy.name!r} expected "
                        f"context_file at {context_path}"
                    )
                return context_path.read_text()
            if strategy.context is not None:
                return strategy.context
        return judge.context or self._task.instruction or ""

    async def _verify_llm_judge(
        self,
        *,
        strategy: VerifierStrategy | None = None,
    ) -> VerifierResult:
        """Score agent deliverables against a rubric with an LLM judge.

        A missing provider SDK raises ``JudgeEnvironmentError``, which is left
        to propagate: the judge could not run, so the rollout is marked as a
        verifier error rather than silently scored ``0.0`` (which would be
        indistinguishable from a genuine judge verdict of fail).
        """
        from benchflow._utils.reachable_write import require_descriptor_io
        from benchflow.rewards.builtins import JudgeScoringError, LLMJudgeRewardFunc
        from benchflow.rewards.llm import JudgeEnvironmentError

        try:
            require_descriptor_io()
        except RuntimeError as exc:
            raise VerifierInfrastructureError(
                "llm-judge capture capability unavailable: "
                + verifier_failure_reason(exc)
            ) from exc

        judge = self._task.config.verifier.judge

        # API keys for the judge come from [verifier.env]. They are threaded
        # explicitly into the judge call (and on into the provider clients)
        # rather than written to the process-global ``os.environ``. Mutating
        # the shared environment is not concurrency-safe: ``evaluation.py``
        # runs verifications via ``asyncio.gather`` with concurrency > 1, so
        # two judge runs in the same process would race on the env and see
        # each other's (or missing) credentials.
        judge_env: dict[str, str] | None = {}
        if self._task.config.verifier.env:
            judge_env = resolve_env_vars(self._task.config.verifier.env)

        rubric_path = self._resolve_rubric_path(strategy)
        context = self._resolve_llm_judge_context(strategy)
        judge_model = (
            strategy.model if strategy is not None and strategy.model else judge.model
        )

        reward_func = LLMJudgeRewardFunc(
            prompt=context,
            rubric_path=rubric_path,
            judge_model=judge_model,
            judge_env=judge_env,
            judge_errors_are_infra=True,
        )
        async with self._download_deliverables(strategy) as deliverables_dir:
            try:
                score = await reward_func.score(deliverables_dir)
            except (JudgeScoringError, JudgeEnvironmentError, PrivateStorageError):
                # Preserve sanitized judge errors and typed private storage
                # causes, including after an earlier script has run.
                raise
            except Exception as exc:
                raise VerifierInfrastructureError(
                    # The judge's public exception deliberately omits raw
                    # provider text, which can contain credentials.
                    "llm-judge provider failed: " + describe_exception(exc)
                ) from None

        self._logger.info(
            "llm-judge verifier: %d criteria → reward %.4f",
            len(reward_func.events),
            score,
        )

        # Persist the reward in the standard location for downstream tooling.
        self._write_reward_files(
            {"reward": score},
            {
                "criteria": [asdict(event) for event in reward_func.events],
                "aggregate": {"reward": score, "method": "mean"},
                "source": "llm-judge",
            },
        )
        return VerifierResult(rewards={"reward": score})

    # criteria-runner verifier

    async def _verify_criteria_runner(
        self,
        *,
        strategy: VerifierStrategy,
        document: VerifierDocument,
        aggregate_policy: dict[str, Any] | None = None,
    ) -> VerifierResult:
        """Run a verifier-scoped criteria runner inside the sandbox."""
        service = self._task.config.verifier.service
        verifier_outputs_are_mounted = service == "main" and getattr(
            self._sandbox, "is_mounted", False
        )
        if not verifier_outputs_are_mounted:
            await self._clear_output_dir(service)

        runner = _criteria_runner_runner(
            strategy, verifier_dir=self._task.paths.tests_dir
        )
        criteria = _criteria_runner_criteria(
            strategy,
            verifier_dir=self._task.paths.tests_dir,
        )

        sandbox_paths = SandboxPaths()
        verifier_code_dir = sandbox_paths.verifier_code_dir
        try:
            await self._sandbox.upload_dir(
                source_dir=self._task.paths.tests_dir,
                target_dir=str(verifier_code_dir),
                service=service,
                admission=admit(
                    self._sandbox, service, TRUSTED_TRANSFER, reason="the task's tests"
                ),
            )
        except VerifierOutputParseError:
            raise
        except Exception as e:
            raise AddTestsDirError(
                "Failed to add verifier directory to sandbox."
            ) from e
        await self._make_verifier_code_roots(verifier_code_dir, service)

        await self._link_legacy_verifier_mount(
            verifier_code_dir=verifier_code_dir,
            sandbox_paths=sandbox_paths,
            service=service,
        )

        self._write_verifier_file(self._rollout_paths.test_stdout_path, "")
        if service != "main":
            verifier_dir = shlex.quote(str(sandbox_paths.verifier_dir))
            await self._setup_step(
                "target verifier dir",
                lambda: exec_raw(
                    self._sandbox.exec,
                    f"mkdir -p {verifier_dir} && chmod 777 {verifier_dir}",
                    user="root",
                    service=service,
                    timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
                ),
            )

        env = dict(sandbox_verifier_env(self._task) or {})
        root = _criteria_runner_root(strategy)
        env.update(
            {
                "BENCHFLOW_VERIFIER_DIR": str(verifier_code_dir),
                "BENCHFLOW_CRITERIA_RUNNER_ROOT": str(verifier_code_dir / root),
                "BENCHFLOW_REWARD_TEXT": str(sandbox_paths.reward_text_path),
                "BENCHFLOW_REWARD_JSON": str(sandbox_paths.reward_json_path),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(
                    sandbox_paths.reward_details_json_path
                ),
            }
        )
        if criteria is not None:
            criteria_policy = _criteria_runner_criteria_policy(
                criteria,
                strategy=strategy,
                verifier_dir=self._task.paths.tests_dir,
            )
            env["BENCHFLOW_CRITERIA_RUNNER_CRITERIA"] = str(
                verifier_code_dir
                / _safe_strategy_relative_path(
                    criteria,
                    strategy=strategy,
                    field="criteria",
                )
            )
        else:
            criteria_policy = None
        env["BENCHFLOW_CRITERIA_RUNNER_MANIFEST"] = str(
            sandbox_paths.criteria_runner_manifest_path
        )

        manifest_json = _criteria_runner_manifest_json(
            document=document,
            strategy=strategy,
            root=root,
            criteria=criteria,
            criteria_policy=criteria_policy,
            sandbox_paths=sandbox_paths,
            verifier_code_dir=verifier_code_dir,
        )
        # The runner's reward folder is closed to the agent while it runs, as
        # test.sh's is (read-back #1); root writes its manifest there.
        await self._close_output_dir(service, mounted=verifier_outputs_are_mounted)
        manifest_path = shlex.quote(str(sandbox_paths.criteria_runner_manifest_path))
        await self._setup_step(
            "criteria-runner manifest write",
            lambda: exec_raw(
                self._sandbox.exec,
                f"printf %s {shlex.quote(manifest_json)} > {manifest_path}",
                user="root",
                service=service,
                timeout_sec=_SETUP_EXEC_TIMEOUT_SEC,
            ),
        )

        runner_path = verifier_code_dir / _safe_strategy_relative_path(
            runner,
            strategy=strategy,
            field="runner",
        )
        test_stdout_path = shlex.quote(
            str(
                sandbox_paths.verifier_dir
                / self._rollout_paths.test_stdout_path.relative_to(
                    self._rollout_paths.verifier_dir
                ).as_posix()
            )
        )
        runner_command = (
            pinned_shell_prefix(
                env, bash_env=self.bash_env, faithful=self._runs_as_written()
            )
            + f"cd {shlex.quote(str(verifier_code_dir))} && "
            f"python {shlex.quote(str(runner_path.relative_to(verifier_code_dir)))}"
        )
        self._launch_record = self._launch_text = None
        if self._held_launch(service):
            # The runner is the task's verifier code, which may import the
            # agent's: it starts held, as test.sh does.
            held = await self._held_command(runner_command, service, None)
            runner_command = f"{{ {held}; }}"
        command = with_verifier_umask(f"{runner_command} > {test_stdout_path} 2>&1")
        run_env: dict[str, str] | None = env
        if self._agent_may_run(service):
            command, run_env = await self._env_by_root_file(command, env, service)
        if verifier_outputs_are_mounted:
            await self._require_writable_test_stdout(test_stdout_path, service)
        if self.before_test is not None:
            await self.before_test(str(verifier_code_dir))
        # An earlier test command's copy at its exit (a multi-step task's
        # earlier verify) is not this one's outputs: never put back over them.
        from benchflow.task import verifier_outputs

        verifier_outputs.forget(self._sandbox)
        with as_task_command(self._sandbox):
            self.test_command_issued = True
            self._issued_service = service
            self._sandbox._benchflow_test_command_issued = True
            ran = False
            try:
                result = await exec_raw(
                    self._sandbox.exec,
                    command=command,
                    env=run_env,
                    user=self._task.config.verifier.user,
                    service=service,
                    timeout_sec=self._task.config.verifier.timeout_sec,
                )
                ran = _exec_return_code(result) != READ_FAILED
            except (TimeoutError, ExecTimeoutError):
                raise
            except Exception as e:
                raise VerifierExecutionError(
                    f"Sandbox test-command exec failed: {describe_exception(e)}"
                ) from e
            finally:
                # The verifier env's file is not kept past its command.
                if run_env:
                    await remove_exec_env_files(self._sandbox)
                await self._remove_unread_root_env(ran)
        return_code = _exec_return_code(result)

        if verifier_outputs_are_mounted:
            # Its outputs as they were at its exit, before the folder goes to
            # the host user: the rewards are read from that copy (
            # read-back #1, as for test.sh).
            await self._let_host_read_outputs(sandbox_paths, service)
            self._take_outputs_at_exit()
            await self._hand_mounted_outputs_to_host(sandbox_paths, service)
            await self._open_output_dir_to_host(service)
        else:
            try:
                await self._download_verifier_dir(sandbox_paths, service)
            except Exception as e:
                if isinstance(
                    e, (ReachableWriteError, UnsafeDownloadEntryError)
                ) and not verifier_infrastructure_failure(e):
                    raise VerifierOutputParseError(
                        f"Unsafe verifier output capture: {e}"
                    ) from e
                if service != "main" or not await self._recover_main_verifier_outputs(
                    sandbox_paths
                ):
                    raise DownloadVerifierDirError(
                        "Failed to download verifier directory from sandbox: "
                        + verifier_failure_reason(e)
                    ) from e
                self._logger.warning(
                    "Criteria-runner verifier directory download failed; recovered "
                    "canonical verifier output files individually: %s",
                    e,
                )

        if not verifier_outputs_are_mounted:
            self._take_outputs_at_exit()
        self.test_return_code = return_code
        await self._check_launch_record(service)
        return await self._scored_under_network(
            self._criteria_runner_rewards(
                return_code, criteria_policy, aggregate_policy
            ),
            service,
        )

    async def _criteria_runner_rewards(
        self,
        return_code: int,
        criteria_policy: Any,
        aggregate_policy: dict[str, Any] | None,
    ) -> VerifierResult:
        """The criteria runner's reward files, as it wrote them."""
        with self._empty_reward_named("criteria runner", return_code):
            if self._reward_file("reward.json").exists():
                effective_policy = criteria_policy or aggregate_policy
                rewards = self._parse_reward_json_with_text_compat(
                    aggregate_policy=effective_policy,
                    strict_aggregate=criteria_policy is not None,
                )
            elif self._reward_file("reward.txt").exists():
                if criteria_policy is not None:
                    raise VerifierOutputParseError(
                        "criteria-runner strategy with declared criteria must write "
                        "reward.json metrics"
                    )
                rewards = self._parse_reward_text()
            else:
                rewards = None
        if rewards is not None and return_code != 0:
            self._logger.warning(
                "Criteria runner exited with rc=%d after writing its reward; the "
                "reward is scored (reward files were cleared before this run)",
                return_code,
            )
        if rewards is None:
            if return_code != 0:
                raise RewardFileNotFoundError(
                    f"criteria-runner exited with rc={return_code}; no reward file "
                    f"found at verifier/reward.txt or "
                    f"verifier/reward.json"
                )
            raise RewardFileNotFoundError(
                "No reward file found at verifier/reward.txt or verifier/reward.json"
            )

        return VerifierResult(rewards=rewards)

    # ors-episode verifier

    async def _verify_ors_episode(
        self,
        *,
        strategy: VerifierStrategy,
    ) -> VerifierResult:
        """Normalize declared ORS episode reward evidence into BenchFlow rewards."""
        from benchflow.adapters.ors import ORSAdapter

        # Sandbox inputs download into private custody and are read there; the
        # kept copy is placed in the verifier folder afterwards (R4-1).
        verifier_dir = self._rollout_paths.verifier_dir
        kept = verifier_dir / "ors-episode-inputs"
        with private_folder(verifier_dir) as private:
            landing = private / kept.name
            destination(landing).mkdir()
            inputs = await self._collect_ors_episode_inputs(strategy, landing)
            records: list[dict[str, Any]] = []
            for item in inputs:
                records.extend(
                    _load_ors_episode_records(
                        item["local_path"],
                        strategy=strategy,
                        declared_path=item["path"],
                    )
                )
            kept.publish(landing)
        for item in inputs:
            if item["local_path"].parent == landing:
                item["local_path"] = kept / item["local_path"].name
        verify_result = _ors_records_to_verify_result(records, strategy=strategy)
        ors_response = ORSAdapter.verify_result_to_ors(verify_result)
        if not ors_response.get("is_valid"):
            metadata = ors_response.get("metadata", {})
            error = metadata.get("error") if isinstance(metadata, dict) else None
            raise VerifierOutputParseError(
                f"ors-episode strategy {strategy.name!r} produced invalid ORS "
                f"reward evidence: {error or 'invalid reward'}"
            )

        rewards: dict[str, Any] = {
            "reward": ors_response["reward"],
            "metadata": {
                "source": "ors-episode",
                "strategy": strategy.name,
                "ors": ors_response,
            },
        }
        details = {
            "source": "ors-episode",
            "strategy": strategy.name,
            "inputs": [
                {
                    "path": item["path"],
                    "local_path": str(item["local_path"]),
                    "records": item["records"],
                }
                for item in inputs
            ],
            "ors_response": ors_response,
            "aggregate": {
                "reward": ors_response["reward"],
                "method": "ors-terminal",
            },
        }
        self._write_reward_files(rewards, details)
        return VerifierResult(rewards=rewards)

    async def _collect_ors_episode_inputs(
        self,
        strategy: VerifierStrategy,
        inputs_dir: Path,
    ) -> list[dict[str, Any]]:
        if not strategy.inputs:
            raise ORSEpisodeInputError(
                f"ors-episode strategy {strategy.name!r} must declare inputs"
            )

        collected: list[dict[str, Any]] = []
        for index, declared_path in enumerate(strategy.inputs):
            local_path = await self._resolve_ors_episode_input(
                declared_path,
                inputs_dir=inputs_dir,
                index=index,
            )
            collected.append(
                {
                    "path": declared_path,
                    "local_path": local_path,
                    "records": _count_ors_episode_records(
                        local_path,
                        strategy=strategy,
                        declared_path=declared_path,
                    ),
                }
            )
        return collected

    async def _resolve_ors_episode_input(
        self,
        declared_path: str,
        *,
        inputs_dir: Path,
        index: int,
    ) -> Path | ReachablePath:
        try:
            local_candidate = _local_rollout_input_path(
                declared_path,
                rollout_dir=self._rollout_paths.rollout_dir,
            )
            if local_candidate is not None and local_candidate.exists():
                return local_candidate
        except (OSError, ValueError) as exc:
            raise ORSEpisodeInputError(
                f"ors-episode input {declared_path!r} has an unsafe local path"
            ) from exc

        posix_path = PurePosixPath(declared_path)
        if not posix_path.is_absolute():
            raise ORSEpisodeInputError(
                f"ors-episode input {declared_path!r} was not found under the "
                "rollout directory"
            )

        download_target = inputs_dir / f"{index}-{_safe_input_filename(posix_path)}"
        try:
            await self._sandbox.download_file(declared_path, download_target)
        except Exception as e:
            raise ORSEpisodeInputError(
                f"failed to download ors-episode input {declared_path!r}"
            ) from e
        if not download_target.exists():
            raise ORSEpisodeInputError(
                f"ors-episode input {declared_path!r} did not download"
            )
        return download_target

    # agent-judge verifier

    async def _verify_agent_judge(
        self,
        *,
        strategy: VerifierStrategy,
        document: VerifierDocument,
    ) -> VerifierResult:
        """Run a verifier-scoped judge role over declared evidence inputs."""
        from benchflow.rewards.builtins import JudgeScoringError
        from benchflow.rewards.llm import call_judge, parse_verdict

        model = strategy.model or self._task.config.verifier.judge.model
        judge_env: dict[str, str] | None = {}
        if self._task.config.verifier.env:
            judge_env = resolve_env_vars(self._task.config.verifier.env)

        inputs = await self._collect_agent_judge_inputs(strategy)
        prompt = self._agent_judge_prompt(
            strategy=strategy,
            document=document,
            inputs=inputs,
        )
        try:
            raw_response = await call_judge(model, prompt, env=judge_env)
        except Exception as exc:
            raise VerifierInfrastructureError(
                "agent-judge provider failed: " + verifier_failure_reason(exc)
            ) from exc
        try:
            verdict = parse_verdict(raw_response)
            score = _agent_judge_score(verdict)
        except Exception as exc:
            # Provider bytes are controller output. Keep the cause for debugging,
            # but the public error boundary must not expose response text.
            raise JudgeScoringError(
                f"agent-judge strategy {strategy.name!r} returned an invalid verdict "
                f"({type(exc).__name__})"
            ) from exc

        rewards: dict[str, Any] = {
            "reward": score,
            "metadata": {
                "source": "agent-judge",
                "strategy": strategy.name,
                "role": strategy.role,
                "model": model,
            },
        }
        details = {
            "source": "agent-judge",
            "strategy": strategy.name,
            "role": strategy.role,
            "model": model,
            "inputs": [
                {
                    "path": item["path"],
                    "chars": len(item["content"]),
                    "truncated": item["truncated"],
                }
                for item in inputs
            ],
            "verdict": verdict,
            "aggregate": {"reward": score, "method": "agent-judge-score"},
        }
        self._write_reward_files(rewards, details)
        return VerifierResult(rewards=rewards)

    async def _collect_agent_judge_inputs(
        self,
        strategy: VerifierStrategy,
    ) -> list[dict[str, Any]]:
        if not strategy.inputs:
            raise AgentJudgeInputError(
                f"agent-judge strategy {strategy.name!r} must declare inputs"
            )
        collected: list[dict[str, Any]] = []
        # Remote files are extracted outside every rollout mount. Local inputs
        # still use the rollout's no-follow reader; after reading either kind,
        # only immutable bounded text is passed to the judge, never a live path.
        with private_temporary_directory(prefix="benchflow-agent-judge-") as tmp:
            inputs_dir = Path(tmp)
            for index, declared_path in enumerate(strategy.inputs):
                local_path = await self._resolve_agent_judge_input(
                    declared_path,
                    inputs_dir=inputs_dir,
                    index=index,
                )
                evidence_root = (
                    inputs_dir
                    if local_path.is_relative_to(inputs_dir)
                    else self._rollout_paths.rollout_dir
                )
                content, truncated = _read_agent_judge_input(
                    local_path, evidence_root=evidence_root
                )
                collected.append(
                    {
                        "path": declared_path,
                        "content": content,
                        "truncated": truncated,
                    }
                )
        return collected

    async def _resolve_agent_judge_input(
        self,
        declared_path: str,
        *,
        inputs_dir: Path,
        index: int,
    ) -> Path | ReachablePath:
        try:
            local_candidate = _local_rollout_input_path(
                declared_path,
                rollout_dir=self._rollout_paths.rollout_dir,
            )
            if local_candidate is not None and local_candidate.exists():
                return local_candidate
        except (OSError, ValueError) as exc:
            if verifier_infrastructure_failure(exc):
                raise DownloadVerifierDirError(
                    f"Failed to read agent-judge input {declared_path!r}: "
                    + verifier_failure_reason(exc)
                ) from exc
            raise AgentJudgeInputError(
                f"agent-judge input {declared_path!r} has an unsafe local path"
            ) from exc

        posix_path = PurePosixPath(declared_path)
        if not posix_path.is_absolute():
            raise AgentJudgeInputError(
                f"agent-judge input {declared_path!r} was not found under the "
                "rollout directory"
            )

        from benchflow.taskmd.verify import copy_kept

        if ".." in posix_path.parts or str(posix_path) == "/":
            raise AgentJudgeInputError(
                "agent-judge input must stay below the sandbox root"
            )
        capture_root = inputs_dir / f"{index}-capture"
        try:
            refused = await copy_kept(self._sandbox, [str(posix_path)], capture_root)
        except (TimeoutError, ExecTimeoutError):
            raise
        except VerifierOutputParseError as exc:
            raise AgentJudgeInputError(
                f"unsafe agent-judge input {declared_path!r}: {exc}"
            ) from exc
        except Exception as e:
            raise DownloadVerifierDirError(
                f"failed to download agent-judge input {declared_path!r} with bounded capture: "
                + verifier_failure_reason(e)
            ) from e
        download_target = capture_root.joinpath(*posix_path.parts[1:])
        if str(posix_path) in refused or not download_target.is_file():
            raise AgentJudgeInputError(
                f"agent-judge input {declared_path!r} is missing, unsafe, or exceeds capture limits"
            )
        return download_target

    def _agent_judge_prompt(
        self,
        *,
        strategy: VerifierStrategy,
        document: VerifierDocument,
        inputs: list[dict[str, Any]],
    ) -> str:
        role = strategy.role
        role_prompt = document.role_prompts.get(role or "", "")
        task_prompt = (getattr(self._task, "instruction", "") or "").strip()
        evidence = "\n\n".join(
            f"--- {item['path']} ---\n{item['content']}" for item in inputs
        )
        return (
            "You are running as a verifier-scoped agent judge. You are not the "
            "solver and you must not assume access to hidden oracle files or "
            "undeclared verifier fixtures.\n\n"
            f"Verifier role: {role or strategy.name}\n\n"
            f"{role_prompt}\n\n"
            "Public task prompt:\n"
            f"{task_prompt or '(none)'}\n\n"
            "Declared verifier inputs follow. Judge only this evidence.\n\n"
            f"{evidence or '(no declared input content)'}\n\n"
            'Return only JSON: {"score": <number from 0.0 to 1.0>, '
            '"reasoning": "<brief reason>"}'
        )
