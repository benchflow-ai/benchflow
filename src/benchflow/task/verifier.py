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
import shlex
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths

logger = logging.getLogger(__name__)


class VerifierResult(BaseModel):
    """Result from the verifier — reward dict."""

    rewards: dict[str, float | int] | None = None


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
            return {"reward": float(self._rollout_paths.reward_text_path.read_text())}
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from text file {self._rollout_paths.reward_text_path}"
            ) from e

    def _parse_reward_json(self) -> dict[str, float | int]:
        if self._rollout_paths.reward_json_path.stat().st_size == 0:
            raise RewardFileEmptyError(
                f"Reward file is empty at {self._rollout_paths.reward_json_path}"
            )
        try:
            return json.loads(self._rollout_paths.reward_json_path.read_text())
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from JSON file {self._rollout_paths.reward_json_path}"
            ) from e

    async def verify(self) -> VerifierResult:
        """Run the configured verifier and return the reward result."""
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
        try:
            await self._sandbox.upload_dir(
                source_dir=self._task.paths.tests_dir,
                target_dir="/tests",
                service=service,
            )
        except Exception as e:
            raise AddTestsDirError("Failed to add tests directory to sandbox.") from e

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

        sandbox_paths = SandboxPaths()
        test_script_path = shlex.quote(
            str(
                sandbox_paths.tests_dir
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
        await self._sandbox.exec(
            f"chmod +x {test_script_path}",
            user="root",
            service=service,
        )
        await self._sandbox.exec(
            command=f"{test_script_path} > {test_stdout_path} 2>&1",
            env=env,
            user=self._task.config.verifier.user,
            service=service,
        )

        # Download verifier output if it is not host-mounted. Only the agent's
        # ``main`` container has the rollout dir bind-mounted; a target service
        # never does, so target-side rewards (#248) are always downloaded.
        is_mounted = service == "main" and self._sandbox.is_mounted
        if not is_mounted:
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

        if self._rollout_paths.reward_text_path.exists():
            rewards = self._parse_reward_text()
        elif self._rollout_paths.reward_json_path.exists():
            rewards = self._parse_reward_json()
        else:
            raise RewardFileNotFoundError(
                f"No reward file found at {self._rollout_paths.reward_text_path} or "
                f"{self._rollout_paths.reward_json_path}"
            )

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
        dest.mkdir(parents=True, exist_ok=True)
        try:
            await self._sandbox.download_dir(
                source_dir=judge.input_dir,
                target_dir=dest,
            )
        except Exception as e:
            self._logger.warning(
                "Failed to download deliverables from %s: %s", judge.input_dir, e
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
        )
        score = await reward_func.score(deliverables_dir)

        self._logger.info(
            "llm-judge verifier: %d criteria → reward %.4f",
            len(reward_func.events),
            score,
        )

        # Persist the reward in the standard location for downstream tooling.
        self._rollout_paths.reward_json_path.write_text(
            json.dumps({"reward": score}, indent=2)
        )
        return VerifierResult(rewards={"reward": score})
