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
import functools
import json
import logging
import math
import shlex
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
from benchflow._utils.text import describe_exception
from benchflow._verifier_env import pinned_shell_prefix, sandbox_verifier_env
from benchflow.rewards.validation import (
    RewardOutOfRange,
    apply_aggregate_policy,
    declared_reward_range,
    is_out_of_range_reward,
    is_valid_reward_number,
    reward_lenient_from_env,
    reward_range_phrase,
    validate_reward_map,
)
from benchflow.sandbox._exec_output import exec_raw
from benchflow.sandbox._exec_receipt import receipt_exec
from benchflow.sandbox._task_route import as_task_command
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
    RewardFileEmptyError,
    RewardFileNotFoundError,
    RewardOutOfRangeError,
    RubricNotFoundError,
    UnsupportedVerifierStrategyError,
    VerifierExecutionError,
    VerifierInfrastructureError,
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
_EXECUTION_RECEIPT_DIR = PurePosixPath("/run/benchflow")

# The verifier's setup commands (clearing the output dir, chmod, the target
# verifier dir, the criteria-runner manifest) run after the agent has finished. On
# a slow or stalled exec layer (a loaded Docker host, a Daytona API stall) a
# 10 s exec failed them, and the finished rollout's verification with them;
# #1043 made the publish-path execs non-fatal for the same reason (#948). Each
# setup command now gets 60 s, and a failed one two more tries.
_SETUP_FAILED = "Verifier setup failed:"
_SETUP_EXEC_TIMEOUT_SEC = 60
# The start-receipt folder (see Verifier._prepare_execution_receipt).
_RECEIPT_DIR_COMMAND = (
    f"mkdir -p {shlex.quote(str(_EXECUTION_RECEIPT_DIR))} && "
    f"chmod 1777 {shlex.quote(str(_EXECUTION_RECEIPT_DIR))}"
)
_RECEIPT_DIR_BUDGET_SEC = 10
_SETUP_RETRY_DELAYS_SEC = (5.0, 15.0)
# The uid the verifier's commands run as, with shell builtins only: the shell
# reads its own status file (no program on a PATH, no task env in this exec).
_VERIFIER_UID_CMD = (
    'while read -r k v r; do case $k in Uid:) echo "$v"; exit 0;; esac; '
    "done < /proc/self/status; exit 1"
)

# What the plugin guard's ``unrecorded`` marker says on its first line
# (``_pytest_plugin_guard._unrecorded``).
_UNRECORDED_WHY = ("none was taken", "the record did not finish")
_REWARD_FILE_MAX_BYTES = 16 * 1024 * 1024


def _marker_tokens(markers: list[GuardMarker]) -> set[str]:
    """The pytest processes (``<guard>.<pid>-<random>``) that left *markers*."""
    return {marker.name.rpartition(".")[0] for marker in markers}


# Where the verifier's env file goes while services are kept: a folder only
# root can enter, never /tmp. A kept service that can write a folder's parent
# can rename root's folder away and put its own in its place before root
# sources it, and /tmp is writable by everyone (without its sticky bit, as
# some images leave it, that rename is allowed: K2-R6-A-1). So /, /var and
# /var/lib must be root's real folders with no group or other write, and
# /var/lib/benchflow (made root's 0700 before the agent user exists) and this
# folder are made root's 0700; anything else fails the step.
_VERIFIER_ENV_PARENT = "/var/lib/benchflow/verifier-env"
_VERIFIER_ENV_PARENT_CMD = (
    "for d in / /var /var/lib; do "
    '[ -d "$d" ] && [ ! -h "$d" ] && [ "$(stat -c %u "$d")" = 0 ] && '
    'case "$(stat -c %a "$d")" in *[2367]?|*[2367]) false ;; esac || '
    '{ echo "Unsafe folder for the verifier env: $d" >&2; exit 1; }; done; '
    f"for d in /var/lib/benchflow {_VERIFIER_ENV_PARENT}; do "
    '[ ! -h "$d" ] || { echo "Unsafe folder for the verifier env: $d" >&2; '
    "exit 1; }; done; "
    f"umask 077 && mkdir -p {_VERIFIER_ENV_PARENT} && "
    f"for d in /var/lib/benchflow {_VERIFIER_ENV_PARENT}; do "
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
        # Host state, unlike the receipt test.sh's user can delete or forge.
        self.test_command_issued = False
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
            raise VerifierOutputParseError(
                "Cannot read verifier/reward.txt or verifier/reward.json: "
                f"{type(e).__name__} (invalid contents or unsafe file)"
            ) from e

    def _read_reward_file(self, path: ReachablePath) -> str:
        # The verifier user can grow a reward file while it is being read.
        # Bound the read itself, not just a preceding stat (FIX-49 r4).
        with destination(path).open("rb") as stream:
            data = stream.read(_REWARD_FILE_MAX_BYTES + 1)
        if len(data) > _REWARD_FILE_MAX_BYTES:
            raise VerifierOutputParseError(
                f"verifier/{path.name} exceeds the 16 MiB reward file limit"
            )
        return data.decode("utf-8")

    def _parse_reward_text(self) -> dict[str, float | int]:
        if self._rollout_paths.reward_text_path.stat().st_size == 0:
            raise RewardFileEmptyError(
                "verifier/reward.txt is empty",
                file="verifier/reward.txt",
                path=self._rollout_paths.reward_text_path,
            )
        try:
            reward = float(self._read_reward_file(self._rollout_paths.reward_text_path))
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from text file {self._rollout_paths.reward_text_path}"
            ) from e
        if is_out_of_range_reward(reward, reward_range=self._reward_range):
            raise RewardOutOfRangeError(
                RewardOutOfRange("reward", reward, self._reward_range),
                file="verifier/reward.txt",
            )
        if not is_valid_reward_number(reward, reward_range=self._reward_range):
            raise VerifierOutputParseError(
                f"Reward text file {self._rollout_paths.reward_text_path} "
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
        if self._rollout_paths.reward_json_path.stat().st_size == 0:
            raise RewardFileEmptyError(
                "verifier/reward.json is empty",
                file="verifier/reward.json",
                path=self._rollout_paths.reward_json_path,
            )
        try:
            rewards = json.loads(
                self._read_reward_file(self._rollout_paths.reward_json_path)
            )
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from JSON file {self._rollout_paths.reward_json_path}"
            ) from e

        if not isinstance(rewards, dict):
            raise VerifierOutputParseError(
                f"Reward JSON file {self._rollout_paths.reward_json_path} "
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
            )
        except RewardOutOfRange as e:
            error = RewardOutOfRangeError(e, file="verifier/reward.json")
            if e.key != "reward" and "." not in e.key:
                # KUMO's "seed": 3 (#1192): a number at the top is a metric.
                error.args = (
                    f"{error}; a top-level number in reward.json is a metric, "
                    "held to the reward range: write details that are not "
                    "scores to reward-details.json",
                )
            raise error from e
        except ValueError as e:
            raise VerifierOutputParseError(
                f"Reward JSON file {self._rollout_paths.reward_json_path} {e}"
            ) from e

    def _parse_reward_json_with_text_compat(
        self,
        *,
        aggregate_policy: dict[str, Any] | None = None,
        strict_aggregate: bool = False,
    ) -> dict[str, Any]:
        rewards = self._parse_reward_json(
            aggregate_policy=aggregate_policy,
            with_text=not strict_aggregate
            and self._rollout_paths.reward_text_path.exists(),
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
                raise VerifierOutputParseError(
                    f"Reward JSON file {self._rollout_paths.reward_json_path} {e}"
                ) from e
            self._write_verifier_file(
                self._rollout_paths.reward_json_path,
                json.dumps(rewards, indent=2, allow_nan=False),
            )

        if not self._rollout_paths.reward_text_path.exists():
            if "reward" not in rewards:
                try:
                    rewards = apply_aggregate_policy(
                        rewards,
                        aggregate_policy=aggregate_policy,
                        source="reward JSON",
                        reward_range=self._reward_range,
                    )
                except ValueError as e:
                    raise VerifierOutputParseError(
                        f"Reward JSON file {self._rollout_paths.reward_json_path} {e}"
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
                    raise VerifierOutputParseError(
                        f"Reward JSON file {self._rollout_paths.reward_json_path} {e}"
                    ) from e
                json_reward = rewards["reward"]
                if not math.isclose(
                    float(json_reward), float(text_reward), abs_tol=1e-9
                ):
                    raise VerifierOutputParseError(
                        f"Reward JSON file {self._rollout_paths.reward_json_path} "
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
            raise VerifierOutputParseError(
                f"Reward JSON file {self._rollout_paths.reward_json_path} "
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
        with private_folder(verifier_dir) as private:
            landing = private / "verifier"
            destination(landing).mkdir()
            await self._sandbox.download_dir(
                source_dir=str(sandbox_paths.verifier_dir),
                target_dir=landing,
                service=service,
            )
            verifier_dir.publish_entries(landing)

    async def verify(self) -> VerifierResult:
        """Run the configured verifier and return the reward result."""
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

    async def _hand_mounted_outputs_to_host(
        self, sandbox_paths: SandboxPaths, service: str
    ) -> None:
        """Let the host read a reward the test command left on the mounted outputs.

        On Linux rootful docker the rollout's verifier dir is bind-mounted and
        the test command runs as container root, so a reward written through
        ``mkstemp`` (mode 0600) is root's and the host cannot parse it (#1202);
        the download that chowns outputs is skipped when they are mounted.
        Only then, hand the folder to the host user, as the sandbox's teardown
        does: where the host reads it already (rootless docker maps container
        root to the host user) nothing changes owner. Never while agent code
        may still run beside the verifier, since its uid can be the host
        user's.
        """
        chown = getattr(self._sandbox, "_chown_to_host_user", None)
        if chown is None or self._agent_may_run(service) or self._host_reads_rewards():
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

    async def _verifier_uid(self, service: str) -> str:
        """The uid the verifier's commands run as (``[verifier] user``, else
        the sandbox's default user), read by the shell itself."""
        user = self._task.config.verifier.user
        if user in ("root", 0, "0"):
            return "0"
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
        return uid

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
                f"{_VERIFIER_ENV_PARENT}/.benchflow-verifier-env-{uuid4().hex}"
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
        return _root_file_env_prefix(folder) + command, None

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
        try:
            result = await exec_raw(
                self._sandbox.exec,
                _RECEIPT_DIR_COMMAND,
                user="root",
                service=service,
                timeout_sec=_RECEIPT_DIR_BUDGET_SEC,
            )
            return_code = _exec_return_code(result)
        except Exception as e:
            return_code = None
            self._logger.debug("verifier start receipt dir failed: %s", e)
        return self._receipt_after(return_code, service)

    def _receipt_after(self, return_code: int | None, service: str) -> str | None:
        """The start-receipt path once the receipt folder's exec answered
        *return_code* (``None``: it raised); a warning and no receipt unless
        it is 0."""
        if return_code != 0:
            self._logger.warning(
                "Cannot create %s for the verifier start receipt; startup of "
                "this verifier cannot be confirmed",
                _EXECUTION_RECEIPT_DIR,
            )
            return None
        receipt = str(_EXECUTION_RECEIPT_DIR / f"verifier-started-{uuid4().hex}")
        self.execution_receipt = (receipt, service)
        return receipt

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
        service = self._task.config.verifier.service
        verifier_outputs_are_mounted = service == "main" and getattr(
            self._sandbox, "is_mounted", False
        )
        if not verifier_outputs_are_mounted:
            await self._clear_output_dir(service)

        sandbox_paths = SandboxPaths()
        uses_native_verifier_dir = (
            getattr(self._task.paths, "uses_native_verifier_dir", False) is True
        )
        verifier_code_dir = (
            sandbox_paths.verifier_code_dir
            if uses_native_verifier_dir
            else sandbox_paths.tests_dir
        )
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

        await self._link_legacy_verifier_mount(
            verifier_code_dir=verifier_code_dir,
            sandbox_paths=sandbox_paths,
            service=service,
        )

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
        if cwd is not None:
            test_command = f"cd {shlex.quote(cwd)} && {test_command}"
        # The verifier's pins as builtins, before test.sh's shell starts and
        # after whatever env the exec set up (benchflow._verifier_env).
        test_command = (
            pinned_shell_prefix(env, bash_env=self.bash_env) + test_dir + test_command
        )
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

        command = f"{{ {test_command}; }} > {test_stdout_path} 2>&1"
        if not receipt_prepared:
            receipt = await self._prepare_execution_receipt(service)
        if receipt is not None:
            # A probe that found no receipt marks it abandoned; a command that
            # starts after that must not run the tests (_verifier_started).
            # Once the tests end, the command records their exit status beside
            # the receipt (``.exit``): a lost exec whose tests had finished is
            # scored from their outputs, not timed out (finish_test_command).
            ended = shlex.quote(receipt + ".exit")
            command = (
                f"printf started > {shlex.quote(receipt)} && "
                f"[ ! -e {shlex.quote(receipt + '.abandoned')} ] && "
                f"{{ {command}; _benchflow_status=$?; "
                f"printf '%s' \"$_benchflow_status\" > {ended}; "
                "(exit $_benchflow_status); }"
            )
        guard = pytest_plugin_guard_name((env or {}).get("PYTEST_ADDOPTS"))
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
            verifier_outputs_are_mounted=verifier_outputs_are_mounted,
            aggregate_policy=aggregate_policy,
        )
        if parse_rewards and receipt is not None:
            self._finish_test = functools.partial(after_test, parse_rewards=True)
        with as_task_command(self._sandbox):
            self.test_command_issued = True
            self._sandbox._benchflow_test_command_issued = True
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
            except (TimeoutError, ExecTimeoutError):
                raise
            except Exception as e:
                raise VerifierExecutionError(
                    f"Sandbox test-command exec failed: {describe_exception(e)}"
                ) from e
        return await after_test(
            _exec_return_code(test_result), parse_rewards=parse_rewards
        )

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
        return await finish(return_code)

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
    ) -> VerifierResult:
        """The test command has ended with *test_return_code*: bring in its
        outputs, read the plugin guard's markers and score the rewards."""
        self.test_return_code = test_return_code

        # Download verifier output if it is not host-mounted. Only the agent's
        # ``main`` container has the rollout dir bind-mounted; a target service
        # never does, so target-side rewards (#248) are always downloaded.
        if verifier_outputs_are_mounted:
            await self._hand_mounted_outputs_to_host(sandbox_paths, service)
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
                    "Verifier directory download failed; recovered canonical "
                    "verifier output files individually: %s",
                    e,
                )

        markers = (
            await self._plugin_guard_markers(guard, service)
            if guard is not None
            else {}
        )
        checking = markers.get("checking", [])
        if checking and not markers.get("refused") and not markers.get("crashed"):
            # A guard stopped while it checked plugins (killed, or blocked on a
            # FIFO until test.sh's timeout) recorded no verdict.
            raise PluginGuardLoadError(
                f"pytest stopped while the plugin guard {guard} was still "
                "checking pytest plugins, so it never decided whether to trust "
                f"them; listed in verifier/{checking[0].name}; this run is not "
                "scored"
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
            unrecorded = getattr(self._sandbox, "_benchflow_guard_unrecorded", ())
            # Why each took none, from hardening (its log line is DEBUG).
            why = getattr(self._sandbox, "_benchflow_guard_why", None)
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
            # task without pytest gets it on every run (dogfood p1-5).
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
                "scored"
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
        if refused:
            # The guard refused a plugin it could not trust: pytest
            # stopped before any test ran, so test.sh's 0 scores the guard, not
            # the solution, and a guard false positive would pass as a fair 0.
            lines = refused[0].detail.strip().splitlines()
            raise PluginGuardLoadError(
                f"the pytest plugin guard {guard} refused a pytest plugin it "
                "could not trust, so pytest stopped and this run is not scored"
                + (f" ({lines[0].strip()[:500]})" if lines else "")
                + f"; listed in verifier/{refused[0].name}"
            )

        if not parse_rewards:
            return VerifierResult(rewards=None)

        with self._empty_reward_named("test.sh", test_return_code):
            if self._rollout_paths.reward_json_path.exists():
                rewards = self._parse_reward_json_with_text_compat(
                    aggregate_policy=aggregate_policy,
                )
            elif self._rollout_paths.reward_text_path.exists():
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
                    f"found at {self._rollout_paths.reward_text_path} or "
                    f"{self._rollout_paths.reward_json_path}"
                )
            else:
                msg = (
                    f"No reward file found at {self._rollout_paths.reward_text_path} or "
                    f"{self._rollout_paths.reward_json_path}"
                )
            # Surface ONLY a fixed, secret-free dep-install diagnostic (never
            # any scanned stdout) so classify_verifier_error can return
            # VERIFIER_DEP_INSTALL without leaking untrusted subprocess output
            # into result metadata (#572). Raw resolver output remains in the
            # downloaded verifier/test-stdout.txt artifact.
            if _has_dep_install_failure(self._rollout_paths.test_stdout_path):
                msg += f"\n{_DEP_INSTALL_DIAGNOSTIC}"
            raise RewardFileNotFoundError(msg)

        return VerifierResult(rewards=rewards)

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
        """
        problem = None
        output = ""
        try:
            result = await receipt_exec(
                self._sandbox,
                pytest_plugin_guard_markers_cmd(guard),
                user="root",
                service=service,
                timeout_sec=30,
            )
        except Exception as e:
            problem = f"{type(e).__name__}: {e}"
        else:
            output = result.stdout or ""
            if _exec_return_code(result) != 0:
                problem = f"exit {_exec_return_code(result)}"
            else:
                problem = pytest_plugin_guard_markers_problem(output)
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
                f"the pytest plugin guard {guard}'s markers could not all be "
                f"read ({str(problem)[:300]}), so a refusal may be among those "
                "missed; guard loading is not confirmed"
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
                            raise VerifierOutputParseError(
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
        if verifier_outputs_are_mounted:
            self._write_verifier_file(
                self._rollout_paths.criteria_runner_manifest_path, manifest_json
            )
        else:
            manifest_path = shlex.quote(
                str(sandbox_paths.criteria_runner_manifest_path)
            )
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
        command = with_verifier_umask(
            pinned_shell_prefix(env, bash_env=self.bash_env)
            + f"cd {shlex.quote(str(verifier_code_dir))} && "
            f"python {shlex.quote(str(runner_path.relative_to(verifier_code_dir)))} "
            f"> {test_stdout_path} 2>&1"
        )
        run_env: dict[str, str] | None = env
        if self._agent_may_run(service):
            command, run_env = await self._env_by_root_file(command, env, service)
        if self.before_test is not None:
            await self.before_test(str(verifier_code_dir))
        with as_task_command(self._sandbox):
            self.test_command_issued = True
            self._sandbox._benchflow_test_command_issued = True
            try:
                result = await exec_raw(
                    self._sandbox.exec,
                    command=command,
                    env=run_env,
                    user=self._task.config.verifier.user,
                    service=service,
                    timeout_sec=self._task.config.verifier.timeout_sec,
                )
            except (TimeoutError, ExecTimeoutError):
                raise
            except Exception as e:
                raise VerifierExecutionError(
                    f"Sandbox test-command exec failed: {describe_exception(e)}"
                ) from e
        return_code = _exec_return_code(result)

        if verifier_outputs_are_mounted:
            await self._hand_mounted_outputs_to_host(sandbox_paths, service)
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

        with self._empty_reward_named("criteria runner", return_code):
            if self._rollout_paths.reward_json_path.exists():
                effective_policy = criteria_policy or aggregate_policy
                rewards = self._parse_reward_json_with_text_compat(
                    aggregate_policy=effective_policy,
                    strict_aggregate=criteria_policy is not None,
                )
            elif self._rollout_paths.reward_text_path.exists():
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
                    f"found at {self._rollout_paths.reward_text_path} or "
                    f"{self._rollout_paths.reward_json_path}"
                )
            raise RewardFileNotFoundError(
                f"No reward file found at {self._rollout_paths.reward_text_path} or "
                f"{self._rollout_paths.reward_json_path}"
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
