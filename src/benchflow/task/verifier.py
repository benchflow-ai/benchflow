"""Verifier ($V$) — maps agent completion to a reward signal.

Internalized from Harbor's Verifier class. Supports two verification methods,
selected by ``[verifier].type`` in ``task.toml``:

- ``"test-script"`` (default): run ``tests/test.sh`` inside the sandbox and
  parse ``reward.json`` / ``reward.txt`` (JSON is authoritative when present).
- ``"llm-judge"``: download the agent's deliverables and grade them against a
  human-authored rubric using an LLM judge (see #270).
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchflow.rewards.validation import (
    RewardFileParseError,
    parse_verifier_reward_files,
)
from benchflow.sandbox.lockdown import _exec_return_code, clear_verifier_output_dir
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths
from benchflow.task.verifier_document import (
    VerifierDocument,
    is_executable_agent_judge_strategy,
    is_executable_reward_kit_strategy,
    is_executable_script_strategy,
    resolve_agent_judge_role_prompt,
    resolve_default_strategy,
    resolve_structured_rubric_path,
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


class UnsupportedVerifierStrategyError(ValueError):
    """Raised when a verifier document selects a strategy the runtime cannot execute."""

    def __init__(
        self,
        *,
        strategy_name: str,
        strategy_type: str | None,
        task_path: Path | str | None = None,
    ) -> None:
        self.strategy_name = strategy_name
        self.strategy_type = strategy_type
        self.task_path = Path(task_path) if task_path is not None else None
        type_label = strategy_type or "<missing>"
        location = f" at {self.task_path}" if self.task_path is not None else ""
        message = (
            f"Verifier strategy {strategy_name!r} (type={type_label!r}){location} "
            "is parsed but not executable by the runtime yet"
        )
        super().__init__(message)


class Verifier:
    """Runs the task's verifier and parses rewards.

    Two verification methods are supported (selected by ``verifier.type``):

    1. ``test-script`` — uploads the task's ``tests/`` directory into the
       sandbox at ``/tests``, runs ``test.sh`` with configured env vars/user,
       and parses the reward from ``reward.json`` or ``reward.txt``.
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

    def _parse_rewards(self, *, test_return_code: int) -> dict[str, Any]:
        """Parse verifier reward outputs with JSON-first precedence."""
        try:
            return parse_verifier_reward_files(
                reward_text_path=self._rollout_paths.reward_text_path,
                reward_json_path=self._rollout_paths.reward_json_path,
                source="reward JSON",
            )
        except RewardFileParseError as exc:
            message = str(exc)
            if "No reward file found" in message:
                if test_return_code != 0:
                    raise RewardFileNotFoundError(
                        f"verifier exited with rc={test_return_code}; {message}"
                    ) from exc
                raise RewardFileNotFoundError(message) from exc
            if "empty" in message:
                raise RewardFileEmptyError(message) from exc
            raise VerifierOutputParseError(message) from exc

    def _clear_reward_outputs(self) -> None:
        for path in (
            self._rollout_paths.reward_text_path,
            self._rollout_paths.reward_json_path,
            self._rollout_paths.reward_details_path,
        ):
            path.unlink(missing_ok=True)

    def _verifier_dir(self) -> Path:
        return Path(self._task.paths.tests_dir)

    def _route_verifier_document_strategy(self) -> str | None:
        """Log and validate the selected ``verifier/verifier.md`` strategy.

        Returns ``"test-script"``, ``"reward-kit"``, ``"agent-judge"``, or
        ``None`` when no document strategy is selected.
        """
        document = getattr(self._task, "verifier_document", None)
        if not isinstance(document, VerifierDocument) or not document.default_strategy:
            return None

        strategy_name, strategy = resolve_default_strategy(document)
        strategy_type = strategy.get("type")
        verifier_dir = self._verifier_dir()
        self._logger.info(
            "Selected verifier document strategy %r (type=%r)",
            strategy_name,
            strategy_type,
        )
        if is_executable_script_strategy(strategy):
            return "test-script"
        if is_executable_reward_kit_strategy(strategy, verifier_dir):
            return "reward-kit"
        if is_executable_agent_judge_strategy(strategy, document, verifier_dir):
            return "agent-judge"

        task_path = getattr(self._task, "task_dir", None)
        raise UnsupportedVerifierStrategyError(
            strategy_name=strategy_name,
            strategy_type=strategy_type if isinstance(strategy_type, str) else None,
            task_path=task_path,
        )

    async def verify(self) -> VerifierResult:
        """Run the configured verifier and return the reward result."""
        self._clear_reward_outputs()
        document_route = self._route_verifier_document_strategy()
        if document_route in {"test-script", "reward-kit"}:
            return await self._verify_test_script()
        if document_route == "agent-judge":
            return await self._verify_agent_judge_document()
        if self._task.config.verifier.type == "llm-judge":
            return await self._verify_llm_judge()
        return await self._verify_test_script()

    # ------------------------------------------------------------------
    # test-script verifier (default — Harbor-compatible)
    # ------------------------------------------------------------------

    async def _verify_test_script(self) -> VerifierResult:
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

        test_script_path = shlex.quote(
            str(
                verifier_code_dir
                / self._task.paths.test_path.relative_to(
                    self._task.paths.tests_dir
                ).as_posix()
            )
        )
        test_stdout_path = shlex.quote(
            str(
                sandbox_paths.verifier_dir
                / self._rollout_paths.test_stdout_path.relative_to(
                    self._rollout_paths.verifier_dir
                ).as_posix()
            )
        )
        chmod_result = await self._sandbox.exec(
            f"chmod +x {test_script_path}",
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
            command=f"{test_script_path} > {test_stdout_path} 2>&1",
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
                raise DownloadVerifierDirError(
                    "Failed to download verifier directory from sandbox"
                ) from e

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

        rewards = self._parse_rewards(test_return_code=test_return_code)
        return VerifierResult(rewards=rewards)

    # ------------------------------------------------------------------
    # llm-judge verifier (#270)
    # ------------------------------------------------------------------

    def _resolve_rubric_path(self) -> Path:
        """Locate the rubric file relative to the task directory."""
        judge = self._task.config.verifier.judge
        rubric_path = Path(judge.rubric_path)
        if not rubric_path.is_absolute():
            rubric_path = Path(self._task.paths.task_dir) / rubric_path
        if not rubric_path.exists():
            raise RubricNotFoundError(
                f"llm-judge rubric not found at {rubric_path}. Set "
                f"[verifier.judge].rubric_path in task.toml."
            )
        return rubric_path

    async def _download_deliverables(self) -> Path:
        """Download the agent's deliverables from the sandbox.

        Returns the local directory the judge should read from.
        """
        judge = self._task.config.verifier.judge
        dest = self._rollout_paths.verifier_dir / "deliverables"
        if dest.exists():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.mkdir(parents=True, exist_ok=True)
        try:
            await self._sandbox.download_dir(
                source_dir=judge.input_dir,
                target_dir=dest,
            )
        except Exception as e:
            raise DownloadVerifierDirError(
                f"Failed to download llm-judge input from {judge.input_dir}"
            ) from e
        return dest

    async def _verify_agent_judge_document(self) -> VerifierResult:
        """Score declared verifier inputs with a document agent-judge strategy."""
        document = self._task.verifier_document
        if not isinstance(document, VerifierDocument):
            raise VerifierOutputParseError(
                "agent-judge verifier requires a parsed verifier document"
            )

        strategy_name, strategy = resolve_default_strategy(document)
        verifier_dir = self._verifier_dir()
        role_prompt = resolve_agent_judge_role_prompt(strategy, document, verifier_dir)
        rubric_path = resolve_structured_rubric_path(document, verifier_dir)
        if role_prompt is None or rubric_path is None:
            raise VerifierOutputParseError(
                f"agent-judge strategy {strategy_name!r} is missing role prompt "
                "or structured rubric"
            )

        deliverables_dir = await self._gather_agent_judge_inputs(strategy)
        judge_env: dict[str, str] = {}
        if self._task.config.verifier.env:
            judge_env = resolve_env_vars(self._task.config.verifier.env)

        from benchflow.rewards.builtins import LLMJudgeRewardFunc

        reward_func = LLMJudgeRewardFunc(
            prompt=role_prompt,
            rubric_path=rubric_path,
            judge_model=strategy.get("model")
            if isinstance(strategy.get("model"), str)
            else None,
            judge_env=judge_env,
            judge_errors_are_infra=True,
        )
        score = await reward_func.score(deliverables_dir)

        self._logger.info(
            "agent-judge verifier strategy %r: %d criteria → reward %.4f",
            strategy_name,
            len(reward_func.events),
            score,
        )

        rewards = {"reward": score}
        self._rollout_paths.reward_json_path.write_text(
            json.dumps(rewards, indent=2, allow_nan=False)
        )
        if reward_func.events:
            details = {
                "strategy": strategy_name,
                "criteria": [
                    {
                        "source": event.source,
                        "score": event.reward,
                    }
                    for event in reward_func.events
                ],
            }
            self._rollout_paths.reward_details_path.write_text(
                json.dumps(details, indent=2, allow_nan=False)
            )
        return VerifierResult(rewards=rewards)

    async def _gather_agent_judge_inputs(self, strategy: dict[str, Any]) -> Path:
        """Collect verifier-scoped judge inputs into a local deliverables dir."""
        dest = self._rollout_paths.verifier_dir / "agent_judge_inputs"
        if dest.exists():
            if dest.is_dir() and not dest.is_symlink():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        dest.mkdir(parents=True, exist_ok=True)

        raw_inputs = strategy.get("inputs")
        inputs = (
            [item for item in raw_inputs if isinstance(item, str) and item.strip()]
            if isinstance(raw_inputs, list)
            else []
        )
        if not inputs:
            return dest

        rollout_dir = self._rollout_paths.rollout_dir
        for index, input_path in enumerate(inputs):
            stripped = input_path.strip()
            target_name = f"input-{index}-{Path(stripped).name}"
            target = dest / target_name

            if stripped.startswith("trajectory/"):
                host_path = rollout_dir / stripped
                if host_path.is_file():
                    target.write_bytes(host_path.read_bytes())
                continue

            if stripped.startswith("/logs/"):
                sandbox_source = stripped
            else:
                sandbox_source = stripped if stripped.startswith("/") else f"/{stripped}"

            try:
                await self._sandbox.download_file(sandbox_source, target)
            except Exception as exc:
                self._logger.warning(
                    "agent-judge input %r unavailable: %s", stripped, exc
                )
        return dest

    async def _verify_llm_judge(self) -> VerifierResult:
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

        rubric_path = self._resolve_rubric_path()
        deliverables_dir = await self._download_deliverables()

        context = judge.context or self._task.instruction or ""

        reward_func = LLMJudgeRewardFunc(
            prompt=context,
            rubric_path=rubric_path,
            judge_model=judge.model,
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
        return VerifierResult(rewards={"reward": score})
