"""Verifier ($V$) — maps agent completion to a reward signal.

Internalized from Harbor's Verifier class. Supports two verification methods,
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

import json
import logging
import math
import shlex
import shutil
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from benchflow.rewards.validation import (
    apply_aggregate_policy,
    is_valid_reward_number,
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
from benchflow.task.verifier_errors import (
    AddTestsDirError,
    AgentJudgeInputError,
    DownloadVerifierDirError,
    ORSEpisodeInputError,
    RewardFileEmptyError,
    RewardFileNotFoundError,
    RubricNotFoundError,
    UnsupportedVerifierStrategyError,
    VerifierOutputParseError,
    VerifierResult,
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
from benchflow.task.verifier_reward_kit import (
    _llm_judge_input_dir,
    _reward_kit_criteria,
    _reward_kit_criteria_policy,
    _reward_kit_manifest_json,
    _reward_kit_root,
    _reward_kit_runner,
    _safe_strategy_relative_path,
)
from benchflow.task.verifier_scan import (
    _DEP_INSTALL_DIAGNOSTIC,
    _has_dep_install_failure,
)
from benchflow.task.verifier_script_strategy import (
    _has_aggregate_declaration,
    _script_strategy_chmod_command,
    _script_strategy_command,
)

logger = logging.getLogger(__name__)


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
        if not is_valid_reward_number(reward):
            raise VerifierOutputParseError(
                f"Reward text file {self._rollout_paths.reward_text_path} "
                "must contain a finite numeric reward between 0.0 and 1.0"
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
            result = await self._sandbox.exec(
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

        await self._link_legacy_verifier_mount(
            verifier_code_dir=verifier_code_dir,
            sandbox_paths=sandbox_paths,
            service=service,
        )

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

        await self._link_legacy_verifier_mount(
            verifier_code_dir=verifier_code_dir,
            sandbox_paths=sandbox_paths,
            service=service,
        )

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
