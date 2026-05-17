"""Verifier ($V$) — maps agent completion to a reward signal.

Internalized from Harbor's Verifier class. Runs test.sh inside the sandbox
and parses reward files.
"""

from __future__ import annotations

import json
import logging
import shlex
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


class Verifier:
    """Runs the task's test.sh verifier inside the sandbox and parses rewards.

    The verifier:
    1. Uploads the task's tests/ directory into the sandbox at /tests
    2. Runs test.sh with configured env vars and user
    3. Parses reward from reward.txt or reward.json
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
        """Run the verifier and return the reward result."""
        try:
            await self._sandbox.upload_dir(
                source_dir=self._task.paths.tests_dir,
                target_dir="/tests",
            )
        except Exception as e:
            raise AddTestsDirError(
                "Failed to add tests directory to sandbox."
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
        )
        await self._sandbox.exec(
            command=f"{test_script_path} > {test_stdout_path} 2>&1",
            env=env,
            user=self._task.config.verifier.user,
        )

        # Download verifier output if sandbox doesn't mount locally
        is_mounted = getattr(self._sandbox, "is_mounted", False)
        if not is_mounted:
            try:
                await self._sandbox.download_dir(
                    source_dir=str(sandbox_paths.verifier_dir),
                    target_dir=self._rollout_paths.verifier_dir,
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
