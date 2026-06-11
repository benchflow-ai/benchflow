"""Verifier ($V$) — maps agent completion to a reward signal.

Internalized from Harbor's Verifier class. Supports two verification methods,
selected by ``[verifier].type`` in ``task.toml``:

- ``"test-script"`` (default): run ``tests/test.sh`` inside the sandbox and
  parse ``reward.txt`` / ``reward.json``.
- ``"llm-judge"``: download the agent's deliverables and grade them against a
  human-authored rubric using an LLM judge (see #270).
"""

from __future__ import annotations

import json
import logging
import math
import shlex
import shutil
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pydantic import BaseModel

from benchflow._utils.scoring import (
    VERIFIER_DEP_INSTALL_MARKERS,
    contains_verifier_dep_install_marker,
)
from benchflow.rewards.events import Granularity, RewardEvent, Space
from benchflow.rewards.protocol import VerifyResult
from benchflow.rewards.rubric_config import criteria_aggregate_policy_from_rubric
from benchflow.rewards.validation import (
    apply_aggregate_policy,
    declared_reward_range,
    is_valid_reward_number,
    reward_lenient_from_env,
    reward_range_phrase,
    validate_reward_map,
)
from benchflow.sandbox.lockdown import _exec_return_code, clear_verifier_output_dir
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths
from benchflow.task.verifier_document import (
    VerifierDocument,
    VerifierStrategy,
    load_verifier_document,
)

logger = logging.getLogger(__name__)


class VerifierResult(BaseModel):
    """Result from the verifier — reward dict."""

    model_config = {"strict": True}

    rewards: dict[str, Any] | None = None


class RewardFileEmptyError(Exception):
    pass


class RewardFileNotFoundError(Exception):
    pass


class VerifierOutputParseError(Exception):
    pass


class AddTestsDirError(Exception):
    pass


class DownloadVerifierDirError(Exception):
    pass


class RubricNotFoundError(Exception):
    """Raised when an llm-judge verifier cannot locate its rubric file."""


# The safe, fixed diagnostic surfaced into ``verifier_error`` on a detected
# dep-install failure. Contains a marker the classifier recognises; carries no
# stdout content. Points operators at the (private) log artifact for detail.
_DEP_INSTALL_DIAGNOSTIC = (
    "dependency install failed (see verifier/test-stdout.txt in the run "
    "artifacts for resolver output)"
)

# Size of each fixed chunk streamed off disk while scanning for markers. The
# whole file is scanned (dep-install runs at the START of test.sh, so the marker
# can be buried under arbitrarily many trailing lines — see PR #572), but only
# one bounded chunk is ever held in memory at a time, so a verifier emitting a
# single huge line can't bloat memory. We never persist any of the scanned text.
_SCAN_CHUNK_BYTES = 64 * 1024


def _has_dep_install_failure(path: Path) -> bool:
    """True if *path* (test-stdout.txt) shows a dependency-install failure.

    Only the boolean verdict leaves this function — the scanned text is never
    returned or persisted, so no secret-bearing stdout can reach result
    metadata (PR #572).

    The ENTIRE file is scanned from the start in fixed ``_SCAN_CHUNK_BYTES``
    chunks (with a small overlap so a marker straddling a chunk boundary is
    still caught), short-circuiting on the first marker. uv/pip install runs at
    the START of ``test.sh``, so its failure marker may be followed by many
    lines of trailing output (fallback attempts, cleanup, partial tests); a
    tail-only scan would silently drop it (PR #572). Memory stays bounded — at
    most one chunk plus the overlap is held at a time, regardless of file size.
    """
    # Longest marker minus one byte: enough overlap to catch a marker split
    # across two reads without rescanning whole chunks.
    overlap = max(len(m) for m in VERIFIER_DEP_INSTALL_MARKERS) - 1
    try:
        with path.open(errors="replace") as f:
            carry = ""
            while True:
                chunk = f.read(_SCAN_CHUNK_BYTES)
                if not chunk:
                    return False
                window = carry + chunk
                if contains_verifier_dep_install_marker(window):
                    return True
                # Keep the tail of this chunk so a boundary-spanning marker is
                # found on the next read; bound it to the overlap size.
                carry = chunk[-overlap:] if overlap else ""
    except OSError:
        return False


class UnsupportedVerifierStrategyError(VerifierOutputParseError):
    """Raised when ``verifier/verifier.md`` selects a non-executable strategy."""


class AgentJudgeInputError(VerifierOutputParseError):
    """Raised when an agent-judge strategy cannot read declared inputs."""


class ORSEpisodeInputError(VerifierOutputParseError):
    """Raised when an ors-episode strategy cannot read declared reward evidence."""


def _script_strategy_command(
    strategy: VerifierStrategy,
    verifier_code_dir: PurePosixPath,
) -> str:
    command = strategy.command
    if command is None:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} is missing command"
        )
    _script_strategy_first_token(strategy)
    return f"cd {shlex.quote(str(verifier_code_dir))} && {command}"


def _script_strategy_chmod_command(
    strategy: VerifierStrategy,
    verifier_code_dir: PurePosixPath,
) -> str | None:
    first = _script_strategy_first_token(strategy)
    if "/" not in first:
        return None
    script_path = _relative_posix_path(first, strategy=strategy)
    return f"chmod +x {shlex.quote(str(verifier_code_dir / script_path))}"


def _script_strategy_first_token(strategy: VerifierStrategy) -> str:
    command = strategy.command
    if command is None:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} is missing command"
        )
    try:
        tokens = shlex.split(command)
    except ValueError as e:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} has invalid command"
        ) from e
    if not tokens:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} has empty command"
        )
    first = tokens[0]
    if first.startswith("/"):
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} command must be relative "
            "to the verifier directory"
        )
    if "/" in first:
        _relative_posix_path(first, strategy=strategy)
    return first


def _relative_posix_path(
    value: str,
    *,
    strategy: VerifierStrategy,
) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise VerifierOutputParseError(
            f"script verifier strategy {strategy.name!r} command must stay inside "
            "the verifier directory"
        )
    return path


def _has_aggregate_declaration(
    rewards: dict[str, Any],
    aggregate_policy: dict[str, Any] | None,
) -> bool:
    return isinstance(rewards.get("aggregate"), dict) or bool(aggregate_policy)


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
    ) -> None:
        self._task = task
        self._rollout_paths = rollout_paths
        self._sandbox = sandbox
        self._logger = (_logger or logger).getChild("verifier")
        # Task-declared ``[verifier] reward_range`` (BF-8); None keeps the
        # canonical strict [0, 1]. Applies to the test-script reward contract
        # (reward.txt / reward.json) — judge and ORS scores stay [0, 1].
        self._reward_range = declared_reward_range(task)

    def _parse_reward_text(self) -> dict[str, float | int]:
        if self._rollout_paths.reward_text_path.stat().st_size == 0:
            raise RewardFileEmptyError(
                f"Reward file is empty at {self._rollout_paths.reward_text_path}"
            )
        try:
            reward = float(self._rollout_paths.reward_text_path.read_text())
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from text file {self._rollout_paths.reward_text_path}"
            ) from e
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
    ) -> dict[str, Any]:
        if self._rollout_paths.reward_json_path.stat().st_size == 0:
            raise RewardFileEmptyError(
                f"Reward file is empty at {self._rollout_paths.reward_json_path}"
            )
        try:
            rewards = json.loads(self._rollout_paths.reward_json_path.read_text())
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from JSON file {self._rollout_paths.reward_json_path}"
            ) from e

        if not isinstance(rewards, dict):
            raise VerifierOutputParseError(
                f"Reward JSON file {self._rollout_paths.reward_json_path} "
                "must contain an object with numeric rewards"
            )

        try:
            return validate_reward_map(
                rewards,
                source="reward JSON",
                aggregate_policy=aggregate_policy,
                lenient=reward_lenient_from_env(),
                reward_range=self._reward_range,
            )
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
        rewards = self._parse_reward_json(aggregate_policy=aggregate_policy)
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
            self._rollout_paths.reward_json_path.write_text(
                json.dumps(rewards, indent=2, allow_nan=False)
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
                self._rollout_paths.reward_json_path.write_text(
                    json.dumps(rewards, indent=2, allow_nan=False)
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
                self._rollout_paths.reward_json_path.write_text(
                    json.dumps(rewards, indent=2, allow_nan=False)
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
            self._rollout_paths.reward_kit_manifest_path,
        ):
            path.unlink(missing_ok=True)

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
            if strategy.type == "reward-kit":
                return await self._verify_reward_kit(
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
            raise UnsupportedVerifierStrategyError(
                f"verifier strategy {strategy.name!r} has type {strategy.type!r}, "
                "which is parsed but not executable yet"
            )
        if self._task.config.verifier.type == "llm-judge":
            return await self._verify_llm_judge()
        return await self._verify_test_script()

    def _selected_verifier_strategy(self) -> VerifierStrategy | None:
        document = self._selected_verifier_document()
        return None if document is None else document.selected_strategy

    def _selected_verifier_document(self) -> VerifierDocument | None:
        paths = getattr(self._task, "paths", None)
        verifier_dir = getattr(paths, "tests_dir", None)
        if not isinstance(verifier_dir, str | Path):
            return None
        return load_verifier_document(verifier_dir)

    # test-script verifier (default — Harbor-compatible)

    async def _verify_test_script(
        self,
        *,
        strategy: VerifierStrategy | None = None,
        aggregate_policy: dict[str, Any] | None = None,
    ) -> VerifierResult:
        """Run the task's ``test.sh`` verifier and return the reward result.

        ``[verifier].service`` selects which compose service ``test.sh`` runs
        in. The default ``"main"`` is the agent container (Harbor-compatible).
        Multi-container (vulhub-style) tasks set it to a target/database
        service so the verifier can inspect *target-side* state — RCE markers,
        DB modifications — instead of only the agent workspace (#248).
        """
        service = self._task.config.verifier.service
        verifier_outputs_are_mounted = service == "main" and getattr(
            self._sandbox, "is_mounted", False
        )
        if not verifier_outputs_are_mounted:
            try:
                await clear_verifier_output_dir(
                    self._sandbox,
                    user="root",
                    service=service,
                    timeout_sec=10,
                )
            except RuntimeError as e:
                raise VerifierOutputParseError(str(e)) from e

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
            )
        except Exception as e:
            raise AddTestsDirError(
                "Failed to add verifier directory to sandbox."
            ) from e

        self._rollout_paths.test_stdout_path.touch()

        env = None
        if self._task.config.verifier.env:
            for key in self._task.config.verifier.env:
                if "api_key" in key.lower():
                    self._logger.debug(
                        "The verifier.env contains an API key (often the case for LLM-"
                        "based verifiers). You will incur costs associated with the "
                        "API calls."
                    )
            env = resolve_env_vars(self._task.config.verifier.env)

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
            test_command = test_script_path
            chmod_command = f"chmod +x {test_script_path}"
        test_stdout_path = shlex.quote(
            str(
                sandbox_paths.verifier_dir
                / self._rollout_paths.test_stdout_path.relative_to(
                    self._rollout_paths.verifier_dir
                ).as_posix()
            )
        )
        if chmod_command is not None:
            chmod_result = await self._sandbox.exec(
                chmod_command,
                user="root",
                service=service,
                timeout_sec=10,
            )
            chmod_return_code = _exec_return_code(chmod_result)
            if chmod_return_code != 0:
                raise VerifierOutputParseError(
                    f"Verifier setup failed: chmod exited with rc={chmod_return_code}"
                )

        if service != "main":
            verifier_dir = shlex.quote(str(sandbox_paths.verifier_dir))
            mkdir_result = await self._sandbox.exec(
                f"mkdir -p {verifier_dir} && chmod 777 {verifier_dir}",
                user="root",
                service=service,
                timeout_sec=10,
            )
            mkdir_return_code = _exec_return_code(mkdir_result)
            if mkdir_return_code != 0:
                raise VerifierOutputParseError(
                    "Verifier setup failed: target verifier dir exited with "
                    f"rc={mkdir_return_code}"
                )

        test_result = await self._sandbox.exec(
            command=f"{test_command} > {test_stdout_path} 2>&1",
            env=env,
            user=self._task.config.verifier.user,
            service=service,
            timeout_sec=self._task.config.verifier.timeout_sec,
        )
        test_return_code = _exec_return_code(test_result)

        # Download verifier output if it is not host-mounted. Only the agent's
        # ``main`` container has the rollout dir bind-mounted; a target service
        # never does, so target-side rewards (#248) are always downloaded.
        if not verifier_outputs_are_mounted:
            try:
                await self._sandbox.download_dir(
                    source_dir=str(sandbox_paths.verifier_dir),
                    target_dir=self._rollout_paths.verifier_dir,
                    service=service,
                )
            except Exception as e:
                if service != "main" or not await self._recover_main_verifier_outputs(
                    sandbox_paths
                ):
                    raise DownloadVerifierDirError(
                        "Failed to download verifier directory from sandbox"
                    ) from e
                self._logger.warning(
                    "Verifier directory download failed; recovered canonical "
                    "verifier output files individually: %s",
                    e,
                )

        if test_return_code != 0 and (
            self._rollout_paths.reward_text_path.exists()
            or self._rollout_paths.reward_json_path.exists()
        ):
            # ENG-150: Verifier produced a reward file but exited nonzero.
            # This is common when test frameworks exit with the count of
            # failures (e.g. pytest exits 1 when all tests fail → reward 0).
            # Since _clear_reward_outputs() wiped stale files before this run,
            # any reward file present was written by THIS invocation. Accept
            # the reward so the result is classified as "failed" (honest model
            # failure) instead of "verifier_errored" (infrastructure problem).
            self._logger.warning(
                "Verifier exited with rc=%d but produced reward output; "
                "accepting reward (reward files were cleared before this run)",
                test_return_code,
            )

        if self._rollout_paths.reward_json_path.exists():
            rewards = self._parse_reward_json_with_text_compat(
                aggregate_policy=aggregate_policy,
            )
        elif self._rollout_paths.reward_text_path.exists():
            rewards = self._parse_reward_text()
        else:
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
                sandbox_paths.reward_kit_manifest_path,
                self._rollout_paths.reward_kit_manifest_path,
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
        recovered: list[Path] = []
        for remote_path, local_path in candidates:
            try:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                await self._sandbox.download_file(str(remote_path), local_path)
            except Exception as exc:
                self._logger.debug(
                    "Could not recover verifier output %s: %s",
                    remote_path,
                    exc,
                )
                continue
            if local_path.exists():
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

    async def _download_deliverables(
        self,
        strategy: VerifierStrategy | None = None,
    ) -> Path:
        """Download the agent's deliverables from the sandbox.

        Returns the local directory the judge should read from.
        """
        input_dir = _llm_judge_input_dir(strategy, self._task.config.verifier.judge)
        dest = self._rollout_paths.verifier_dir / "deliverables"
        if dest.exists():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.mkdir(parents=True, exist_ok=True)
        try:
            await self._sandbox.download_dir(
                source_dir=input_dir,
                target_dir=dest,
            )
        except Exception as e:
            raise DownloadVerifierDirError(
                f"Failed to download llm-judge input from {input_dir}"
            ) from e
        return dest

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
        from benchflow.rewards.builtins import LLMJudgeRewardFunc

        judge = self._task.config.verifier.judge

        # API keys for the judge come from [verifier.env]. They are threaded
        # explicitly into the judge call (and on into the provider clients)
        # rather than written to the process-global ``os.environ``. Mutating
        # the shared environment is not concurrency-safe: ``evaluation.py``
        # runs verifications via ``asyncio.gather`` with concurrency > 1, so
        # two judge runs in the same process would race on the env and see
        # each other's (or missing) credentials.
        judge_env: dict[str, str] = {}
        if self._task.config.verifier.env:
            judge_env = resolve_env_vars(self._task.config.verifier.env)

        rubric_path = self._resolve_rubric_path(strategy)
        deliverables_dir = await self._download_deliverables(strategy)
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
        score = await reward_func.score(deliverables_dir)

        self._logger.info(
            "llm-judge verifier: %d criteria → reward %.4f",
            len(reward_func.events),
            score,
        )

        # Persist the reward in the standard location for downstream tooling.
        self._rollout_paths.reward_json_path.write_text(
            json.dumps({"reward": score}, indent=2, allow_nan=False)
        )
        self._rollout_paths.reward_details_json_path.write_text(
            json.dumps(
                {
                    "criteria": [asdict(event) for event in reward_func.events],
                    "aggregate": {"reward": score, "method": "mean"},
                    "source": "llm-judge",
                },
                indent=2,
                allow_nan=False,
            )
        )
        return VerifierResult(rewards={"reward": score})

    # reward-kit verifier

    async def _verify_reward_kit(
        self,
        *,
        strategy: VerifierStrategy,
        document: VerifierDocument,
        aggregate_policy: dict[str, Any] | None = None,
    ) -> VerifierResult:
        """Run a verifier-scoped Reward Kit runner inside the sandbox."""
        service = self._task.config.verifier.service
        verifier_outputs_are_mounted = service == "main" and getattr(
            self._sandbox, "is_mounted", False
        )
        if not verifier_outputs_are_mounted:
            try:
                await clear_verifier_output_dir(
                    self._sandbox,
                    user="root",
                    service=service,
                    timeout_sec=10,
                )
            except RuntimeError as e:
                raise VerifierOutputParseError(str(e)) from e

        runner = _reward_kit_runner(strategy, verifier_dir=self._task.paths.tests_dir)
        criteria = _reward_kit_criteria(
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
            )
        except Exception as e:
            raise AddTestsDirError(
                "Failed to add verifier directory to sandbox."
            ) from e

        self._rollout_paths.test_stdout_path.touch()
        if service != "main":
            verifier_dir = shlex.quote(str(sandbox_paths.verifier_dir))
            mkdir_result = await self._sandbox.exec(
                f"mkdir -p {verifier_dir} && chmod 777 {verifier_dir}",
                user="root",
                service=service,
                timeout_sec=10,
            )
            mkdir_return_code = _exec_return_code(mkdir_result)
            if mkdir_return_code != 0:
                raise VerifierOutputParseError(
                    "Verifier setup failed: target verifier dir exited with "
                    f"rc={mkdir_return_code}"
                )

        env = dict(resolve_env_vars(self._task.config.verifier.env))
        root = _reward_kit_root(strategy)
        env.update(
            {
                "BENCHFLOW_VERIFIER_DIR": str(verifier_code_dir),
                "BENCHFLOW_REWARD_KIT_ROOT": str(verifier_code_dir / root),
                "BENCHFLOW_REWARD_TEXT": str(sandbox_paths.reward_text_path),
                "BENCHFLOW_REWARD_JSON": str(sandbox_paths.reward_json_path),
                "BENCHFLOW_REWARD_DETAILS_JSON": str(
                    sandbox_paths.reward_details_json_path
                ),
            }
        )
        if criteria is not None:
            criteria_policy = _reward_kit_criteria_policy(
                criteria,
                strategy=strategy,
                verifier_dir=self._task.paths.tests_dir,
            )
            env["BENCHFLOW_REWARD_KIT_CRITERIA"] = str(
                verifier_code_dir
                / _safe_strategy_relative_path(
                    criteria,
                    strategy=strategy,
                    field="criteria",
                )
            )
        else:
            criteria_policy = None
        env["BENCHFLOW_REWARD_KIT_MANIFEST"] = str(
            sandbox_paths.reward_kit_manifest_path
        )

        manifest_json = _reward_kit_manifest_json(
            document=document,
            strategy=strategy,
            root=root,
            criteria=criteria,
            criteria_policy=criteria_policy,
            sandbox_paths=sandbox_paths,
            verifier_code_dir=verifier_code_dir,
        )
        if verifier_outputs_are_mounted:
            self._rollout_paths.reward_kit_manifest_path.write_text(manifest_json)
        else:
            manifest_path = shlex.quote(str(sandbox_paths.reward_kit_manifest_path))
            write_manifest = await self._sandbox.exec(
                f"printf %s {shlex.quote(manifest_json)} > {manifest_path}",
                user="root",
                service=service,
                timeout_sec=10,
            )
            manifest_return_code = _exec_return_code(write_manifest)
            if manifest_return_code != 0:
                raise VerifierOutputParseError(
                    "Verifier setup failed: reward-kit manifest write exited with "
                    f"rc={manifest_return_code}"
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
        command = (
            f"cd {shlex.quote(str(verifier_code_dir))} && "
            f"python {shlex.quote(str(runner_path.relative_to(verifier_code_dir)))} "
            f"> {test_stdout_path} 2>&1"
        )
        result = await self._sandbox.exec(
            command=command,
            env=env,
            user=self._task.config.verifier.user,
            service=service,
            timeout_sec=self._task.config.verifier.timeout_sec,
        )
        return_code = _exec_return_code(result)

        if not verifier_outputs_are_mounted:
            try:
                await self._sandbox.download_dir(
                    source_dir=str(sandbox_paths.verifier_dir),
                    target_dir=self._rollout_paths.verifier_dir,
                    service=service,
                )
            except Exception as e:
                if service != "main" or not await self._recover_main_verifier_outputs(
                    sandbox_paths
                ):
                    raise DownloadVerifierDirError(
                        "Failed to download verifier directory from sandbox"
                    ) from e
                self._logger.warning(
                    "Reward Kit verifier directory download failed; recovered "
                    "canonical verifier output files individually: %s",
                    e,
                )

        if return_code != 0 and (
            self._rollout_paths.reward_text_path.exists()
            or self._rollout_paths.reward_json_path.exists()
        ):
            self._logger.warning(
                "Reward Kit exited with rc=%d but produced reward output; "
                "accepting reward (reward files were cleared before this run)",
                return_code,
            )

        if self._rollout_paths.reward_json_path.exists():
            effective_policy = criteria_policy or aggregate_policy
            rewards = self._parse_reward_json_with_text_compat(
                aggregate_policy=effective_policy,
                strict_aggregate=criteria_policy is not None,
            )
        elif self._rollout_paths.reward_text_path.exists():
            if criteria_policy is not None:
                raise VerifierOutputParseError(
                    "reward-kit strategy with declared criteria must write "
                    "reward.json metrics"
                )
            rewards = self._parse_reward_text()
        else:
            if return_code != 0:
                raise RewardFileNotFoundError(
                    f"reward-kit exited with rc={return_code}; no reward file "
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

        inputs = await self._collect_ors_episode_inputs(strategy)
        records: list[dict[str, Any]] = []
        for item in inputs:
            records.extend(
                _load_ors_episode_records(
                    item["local_path"],
                    strategy=strategy,
                    declared_path=item["path"],
                )
            )
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
        self._rollout_paths.reward_json_path.write_text(
            json.dumps(rewards, indent=2, allow_nan=False)
        )
        self._rollout_paths.reward_details_json_path.write_text(
            json.dumps(details, indent=2, allow_nan=False)
        )
        return VerifierResult(rewards=rewards)

    async def _collect_ors_episode_inputs(
        self,
        strategy: VerifierStrategy,
    ) -> list[dict[str, Any]]:
        if not strategy.inputs:
            raise ORSEpisodeInputError(
                f"ors-episode strategy {strategy.name!r} must declare inputs"
            )
        inputs_dir = self._rollout_paths.verifier_dir / "ors-episode-inputs"
        if inputs_dir.exists():
            if inputs_dir.is_dir() and not inputs_dir.is_symlink():
                shutil.rmtree(inputs_dir)
            else:
                inputs_dir.unlink()
        inputs_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> Path:
        local_candidate = _local_rollout_input_path(
            declared_path,
            rollout_dir=self._rollout_paths.rollout_dir,
        )
        if local_candidate is not None and local_candidate.exists():
            return local_candidate

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
        from benchflow.rewards.llm import call_judge, parse_verdict

        model = strategy.model or self._task.config.verifier.judge.model
        judge_env: dict[str, str] = {}
        if self._task.config.verifier.env:
            judge_env = resolve_env_vars(self._task.config.verifier.env)

        inputs = await self._collect_agent_judge_inputs(strategy)
        prompt = self._agent_judge_prompt(
            strategy=strategy,
            document=document,
            inputs=inputs,
        )
        raw_response = await call_judge(model, prompt, env=judge_env)
        try:
            verdict = parse_verdict(raw_response)
            score = _agent_judge_score(verdict)
        except Exception as e:
            raise VerifierOutputParseError(
                f"agent-judge strategy {strategy.name!r} returned an invalid verdict"
            ) from e

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
        self._rollout_paths.reward_json_path.write_text(
            json.dumps(rewards, indent=2, allow_nan=False)
        )
        self._rollout_paths.reward_details_json_path.write_text(
            json.dumps(details, indent=2, allow_nan=False)
        )
        return VerifierResult(rewards=rewards)

    async def _collect_agent_judge_inputs(
        self,
        strategy: VerifierStrategy,
    ) -> list[dict[str, Any]]:
        inputs_dir = self._rollout_paths.verifier_dir / "agent-judge-inputs"
        if inputs_dir.exists():
            if inputs_dir.is_dir() and not inputs_dir.is_symlink():
                shutil.rmtree(inputs_dir)
            else:
                inputs_dir.unlink()
        inputs_dir.mkdir(parents=True, exist_ok=True)

        collected: list[dict[str, Any]] = []
        for index, declared_path in enumerate(strategy.inputs):
            local_path = await self._resolve_agent_judge_input(
                declared_path,
                inputs_dir=inputs_dir,
                index=index,
            )
            content, truncated = _read_agent_judge_input(local_path)
            collected.append(
                {
                    "path": declared_path,
                    "local_path": str(local_path),
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
    ) -> Path:
        local_candidate = _local_rollout_input_path(
            declared_path,
            rollout_dir=self._rollout_paths.rollout_dir,
        )
        if local_candidate is not None and local_candidate.exists():
            return local_candidate

        posix_path = PurePosixPath(declared_path)
        if not posix_path.is_absolute():
            raise AgentJudgeInputError(
                f"agent-judge input {declared_path!r} was not found under the "
                "rollout directory"
            )

        download_target = inputs_dir / f"{index}-{_safe_input_filename(posix_path)}"
        try:
            await self._sandbox.download_file(declared_path, download_target)
        except Exception as e:
            raise AgentJudgeInputError(
                f"failed to download agent-judge input {declared_path!r}"
            ) from e
        if not download_target.exists():
            raise AgentJudgeInputError(
                f"agent-judge input {declared_path!r} did not download"
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


_AGENT_JUDGE_INPUT_CHAR_LIMIT = 50_000


def _agent_judge_score(verdict: dict[str, Any]) -> float:
    raw_score = verdict.get("score", verdict.get("reward"))
    if raw_score is not None:
        score = float(raw_score)
        if not is_valid_reward_number(score):
            raise ValueError("agent-judge score must be between 0.0 and 1.0")
        return score

    raw_verdict = verdict.get("verdict")
    if isinstance(raw_verdict, str):
        normalized = raw_verdict.strip().lower()
        if normalized in {"pass", "passed", "yes", "true"}:
            return 1.0
        if normalized in {"fail", "failed", "no", "false"}:
            return 0.0

    raise ValueError("agent-judge verdict must include score/reward or pass/fail")


def _load_ors_episode_records(
    path: Path,
    *,
    strategy: VerifierStrategy,
    declared_path: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ORSEpisodeInputError(
            f"ors-episode input {declared_path!r} must resolve to a regular file"
        )
    text = path.read_text(errors="replace")
    try:
        if path.suffix == ".jsonl":
            records: list[dict[str, Any]] = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ORSEpisodeInputError(
                        f"ors-episode input {declared_path!r} line {line_number} "
                        "must be a JSON object"
                    )
                records.append(cast(dict[str, Any], loaded))
            if not records:
                raise ORSEpisodeInputError(
                    f"ors-episode input {declared_path!r} has no JSON records"
                )
            return records

        loaded = json.loads(text)
    except json.JSONDecodeError as e:
        raise ORSEpisodeInputError(
            f"ors-episode input {declared_path!r} is not valid JSON"
        ) from e

    if isinstance(loaded, dict):
        return [loaded]
    if isinstance(loaded, list) and all(isinstance(item, dict) for item in loaded):
        if not loaded:
            raise ORSEpisodeInputError(
                f"ors-episode input {declared_path!r} has no JSON records"
            )
        return cast(list[dict[str, Any]], loaded)
    raise ORSEpisodeInputError(
        f"ors-episode input {declared_path!r} must be a JSON object or list of objects"
    )


def _count_ors_episode_records(
    path: Path,
    *,
    strategy: VerifierStrategy,
    declared_path: str,
) -> int:
    return len(
        _load_ors_episode_records(
            path,
            strategy=strategy,
            declared_path=declared_path,
        )
    )


def _ors_records_to_verify_result(
    records: list[dict[str, Any]],
    *,
    strategy: VerifierStrategy,
) -> VerifyResult:
    events: list[RewardEvent] = []
    items: dict[str, float] = {}
    reward: float | None = None
    error: str | None = None
    headline_space: Space = "output"
    headline_granularity: Granularity = "terminal"

    for index, record in enumerate(records):
        path = f"records[{index}]"
        if _is_ors_response(record):
            if record.get("is_valid") is False:
                metadata = record.get("metadata")
                message = metadata.get("error") if isinstance(metadata, dict) else None
                raise VerifierOutputParseError(
                    f"ors-episode strategy {strategy.name!r} contains invalid "
                    f"ORS response at {path}: {message or 'is_valid=false'}"
                )
            reward = _bounded_ors_reward(record.get("reward"), path=f"{path}.reward")
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                items.update(
                    _ors_items(metadata.get("items"), path=f"{path}.metadata.items")
                )
                events.extend(
                    _ors_events(
                        metadata.get("events"),
                        strategy=strategy,
                        path=f"{path}.metadata.events",
                    )
                )
                raw_error = metadata.get("error")
                if isinstance(raw_error, str) and raw_error:
                    error = raw_error
                if isinstance(metadata.get("space"), str):
                    headline_space = _ors_space(
                        metadata["space"],
                        path=f"{path}.metadata.space",
                    )
                if isinstance(metadata.get("granularity"), str):
                    headline_granularity = _ors_granularity(
                        metadata["granularity"],
                        path=f"{path}.metadata.granularity",
                    )
            continue

        if "events" in record:
            events.extend(
                _ors_events(
                    record.get("events"),
                    strategy=strategy,
                    path=f"{path}.events",
                )
            )
            if "reward" in record:
                reward = _bounded_ors_reward(
                    record.get("reward"),
                    path=f"{path}.reward",
                )
            items.update(_ors_items(record.get("items"), path=f"{path}.items"))
            metadata = record.get("metadata", {})
            if isinstance(metadata, dict):
                items.update(
                    _ors_items(metadata.get("items"), path=f"{path}.metadata.items")
                )
            continue

        event = _ors_event(record, strategy=strategy, path=path)
        events.append(event)
        if (
            event.type == "terminal"
            or event.granularity == "terminal"
            or record.get("finished") is True
            or record.get("done") is True
        ):
            reward = event.reward
            headline_space = event.space
            headline_granularity = event.granularity

    if reward is None:
        terminal_events = [
            event
            for event in events
            if event.type == "terminal" or event.granularity == "terminal"
        ]
        if terminal_events:
            last_terminal = terminal_events[-1]
            reward = last_terminal.reward
            headline_space = last_terminal.space
            headline_granularity = last_terminal.granularity
    if reward is None:
        raise VerifierOutputParseError(
            f"ors-episode strategy {strategy.name!r} did not include a terminal reward"
        )
    if not items:
        items[strategy.name] = reward

    return VerifyResult(
        reward=reward,
        items=items,
        events=events,
        error=error,
        space=headline_space,
        granularity=headline_granularity,
    )


def _is_ors_response(record: dict[str, Any]) -> bool:
    return "reward" in record and (
        "is_valid" in record
        or (
            isinstance(record.get("metadata"), dict)
            and (
                "items" in record["metadata"]
                or "events" in record["metadata"]
                or "space" in record["metadata"]
            )
        )
    )


def _ors_events(
    value: Any,
    *,
    strategy: VerifierStrategy,
    path: str,
) -> list[RewardEvent]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise VerifierOutputParseError(f"{path} must be a list")
    events: list[RewardEvent] = []
    for index, raw_event in enumerate(value):
        if not isinstance(raw_event, dict):
            raise VerifierOutputParseError(f"{path}[{index}] must be an object")
        events.append(
            _ors_event(
                cast(dict[str, Any], raw_event),
                strategy=strategy,
                path=f"{path}[{index}]",
            )
        )
    return events


def _ors_event(
    record: dict[str, Any],
    *,
    strategy: VerifierStrategy,
    path: str,
) -> RewardEvent:
    reward = _bounded_ors_reward(record.get("reward"), path=f"{path}.reward")
    raw_step = record.get("step")
    if raw_step is None:
        step = None
    elif isinstance(raw_step, int):
        step = raw_step
    else:
        raise VerifierOutputParseError(f"{path}.step must be an integer or null")
    return RewardEvent(
        type=str(record.get("type") or "terminal"),
        reward=reward,
        source=str(record.get("source") or strategy.name),
        step=step,
        space=_ors_space(record.get("space", "output"), path=f"{path}.space"),
        granularity=_ors_granularity(
            record.get("granularity", "terminal"),
            path=f"{path}.granularity",
        ),
        ts=str(record.get("timestamp") or record.get("ts") or ""),
    )


def _bounded_ors_reward(value: Any, *, path: str) -> float:
    try:
        reward = float(value)
    except (TypeError, ValueError) as e:
        raise VerifierOutputParseError(f"{path} must be a numeric reward") from e
    if not is_valid_reward_number(reward):
        raise VerifierOutputParseError(
            f"{path} must be a finite numeric reward between 0.0 and 1.0"
        )
    return reward


def _ors_items(value: Any, *, path: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise VerifierOutputParseError(f"{path} must be a mapping")
    items: dict[str, float] = {}
    for name, raw_score in value.items():
        items[str(name)] = _bounded_ors_reward(raw_score, path=f"{path}.{name}")
    return items


def _ors_space(value: Any, *, path: str) -> Space:
    if value not in {"output", "action", "reasoning", "memory", "latent"}:
        raise VerifierOutputParseError(f"{path} must be a valid reward space")
    return cast(Space, value)


def _ors_granularity(value: Any, *, path: str) -> Granularity:
    if value not in {"terminal", "step"}:
        raise VerifierOutputParseError(f"{path} must be terminal or step granularity")
    return cast(Granularity, value)


def _reward_kit_root(strategy: VerifierStrategy) -> PurePosixPath:
    root = strategy.root_path
    if root is None:
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} is missing root"
        )
    return _safe_strategy_relative_path(root, strategy=strategy, field="root")


def _llm_judge_input_dir(strategy: VerifierStrategy | None, judge: Any) -> str:
    if strategy is not None and strategy.input_dir is not None:
        return strategy.input_dir
    return str(judge.input_dir)


def _reward_kit_runner(
    strategy: VerifierStrategy,
    *,
    verifier_dir: Path,
) -> str:
    root = _reward_kit_root(strategy)
    entrypoint = strategy.entrypoint or "reward.py"
    runner = root / _safe_strategy_relative_path(
        entrypoint,
        strategy=strategy,
        field="entrypoint",
    )
    local_runner = verifier_dir / Path(*runner.parts)
    if not local_runner.is_file():
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} expected runner at {local_runner}"
        )
    return str(runner)


def _reward_kit_criteria(
    strategy: VerifierStrategy,
    *,
    verifier_dir: Path,
) -> str | None:
    criteria = strategy.criteria_path
    if criteria is None:
        return None
    path = _safe_strategy_relative_path(criteria, strategy=strategy, field="criteria")
    local_criteria = verifier_dir / Path(*path.parts)
    if not local_criteria.is_file():
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} expected criteria at "
            f"{local_criteria}"
        )
    return str(path)


def _reward_kit_criteria_policy(
    criteria: str,
    *,
    strategy: VerifierStrategy,
    verifier_dir: Path,
) -> dict[str, Any]:
    path = _safe_strategy_relative_path(criteria, strategy=strategy, field="criteria")
    local_criteria = verifier_dir / Path(*path.parts)
    try:
        return criteria_aggregate_policy_from_rubric(local_criteria)
    except ValueError as e:
        raise VerifierOutputParseError(
            f"reward-kit strategy {strategy.name!r} has invalid criteria: {e}"
        ) from e


def _reward_kit_manifest_json(
    *,
    document: VerifierDocument,
    strategy: VerifierStrategy,
    root: PurePosixPath,
    criteria: str | None,
    criteria_policy: dict[str, Any] | None,
    sandbox_paths: SandboxPaths,
    verifier_code_dir: PurePosixPath,
) -> str:
    """Build the verifier-scoped manifest passed to Reward Kit runners."""

    criteria_path = (
        str(
            verifier_code_dir
            / _safe_strategy_relative_path(
                criteria,
                strategy=strategy,
                field="criteria",
            )
        )
        if criteria is not None
        else None
    )
    payload = {
        "version": "benchflow.reward-kit.v1",
        "verifier": {
            "name": document.name,
            "document_version": document.document_version,
            "default_strategy": document.default_strategy,
        },
        "strategy": {
            "name": strategy.name,
            "type": strategy.type,
            "root": str(root),
            "entrypoint": strategy.entrypoint or "reward.py",
            "criteria": criteria,
            "config": _jsonable(strategy.config),
        },
        "paths": {
            "verifier_dir": str(verifier_code_dir),
            "reward_kit_root": str(verifier_code_dir / root),
            "criteria": criteria_path,
            "reward_text": str(sandbox_paths.reward_text_path),
            "reward_json": str(sandbox_paths.reward_json_path),
            "reward_details_json": str(sandbox_paths.reward_details_json_path),
        },
        "rubric": _jsonable(document.rubric),
        "criteria_policy": _jsonable(criteria_policy),
        "outputs": _jsonable(asdict(document.outputs)),
    }
    return json.dumps(payload, indent=2, allow_nan=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return str(value)


def _safe_strategy_relative_path(
    value: str,
    *,
    strategy: VerifierStrategy,
    field: str,
) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise VerifierOutputParseError(
            f"verifier strategy {strategy.name!r} {field} must be a safe relative path"
        )
    return path


def _read_agent_judge_input(path: Path) -> tuple[str, bool]:
    if not path.is_file():
        raise AgentJudgeInputError(
            f"agent-judge input {path} must resolve to a regular file"
        )
    content = path.read_text(errors="replace")
    if len(content) <= _AGENT_JUDGE_INPUT_CHAR_LIMIT:
        return content, False
    truncated = (
        content[:_AGENT_JUDGE_INPUT_CHAR_LIMIT]
        + f"\n[TRUNCATED: {len(content)} chars total]"
    )
    return truncated, True


def _local_rollout_input_path(
    declared_path: str,
    *,
    rollout_dir: Path,
) -> Path | None:
    path = PurePosixPath(declared_path)
    if path.is_absolute():
        logs_prefix = PurePosixPath("/logs")
        try:
            relative = path.relative_to(logs_prefix)
        except ValueError:
            return None
    else:
        relative = path

    if not relative.parts or ".." in relative.parts:
        raise AgentJudgeInputError(
            f"agent-judge input {declared_path!r} must stay inside rollout evidence"
        )
    return rollout_dir / Path(*relative.parts)


def _safe_input_filename(path: PurePosixPath) -> str:
    name = "__".join(part for part in path.parts if part != "/")
    return name or "input"
