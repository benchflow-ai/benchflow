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
import re
import shlex
import shutil
from collections import deque
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from benchflow.rewards.validation import is_valid_reward_number, validate_reward_map
from benchflow.sandbox.lockdown import _exec_return_code, clear_verifier_output_dir
from benchflow.task.env import resolve_env_vars
from benchflow.task.paths import RolloutPaths, SandboxPaths

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


_TAIL_LINES = 30

# Secret patterns redacted from verifier stdout before it is surfaced in an
# exception message (and from there into ``verifier_error`` / summaries /
# dashboards). ``test-stdout.txt`` is untrusted subprocess output that can
# contain env dumps, install URLs, or tokens — see PR #572 review. Mirrors the
# trajectory redaction set (#537); kept local so this module has no dependency
# on the trajectories package.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(sk-ant-[a-zA-Z0-9_-]{10})[a-zA-Z0-9_-]+"), r"\1***REDACTED***"),
    (re.compile(r"(sk-proj-[a-zA-Z0-9_-]{10})[a-zA-Z0-9_-]+"), r"\1***REDACTED***"),
    (re.compile(r"(sk-[a-zA-Z0-9]{10})[a-zA-Z0-9]+"), r"\1***REDACTED***"),
    (re.compile(r"(AIzaSy[A-Za-z0-9_-]{4})[A-Za-z0-9_-]{20,}"), r"\1***REDACTED***"),
    (
        re.compile(r"((?:AKIA|ASIA)[A-Z0-9]{4})[A-Z0-9]{12}(?![A-Z0-9])"),
        r"\1***REDACTED***",
    ),
    (re.compile(r"(dtn_[A-Za-z0-9_]{4})[A-Za-z0-9_]{16,}"), r"\1***REDACTED***"),
    (
        re.compile(r'("authorization"\s*:\s*"Bearer\s+)[^"]+(")', re.IGNORECASE),
        r"\1***REDACTED***\2",
    ),
    (
        re.compile(r"(Bearer\s+)[A-Za-z0-9._-]{12,}", re.IGNORECASE),
        r"\1***REDACTED***",
    ),
    (
        re.compile(r"((?:x-api-key|api-key)\s*[:=]\s*)\S+", re.IGNORECASE),
        r"\1***REDACTED***",
    ),
)


def _redact_secrets(text: str) -> str:
    """Strip well-known credential patterns from untrusted subprocess output."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _tail_file(path: Path, n: int = _TAIL_LINES) -> str:
    """Return the last *n* lines of *path*, redacted, or "" if unreadable.

    Streams the file with a bounded ``deque`` so a large ``test-stdout.txt``
    is never fully materialized. Output is redacted because it is untrusted
    subprocess output surfaced into ``verifier_error`` (PR #572 review).
    """
    try:
        with path.open(errors="replace") as f:
            lines = deque(f, maxlen=n)
    except OSError:
        return ""
    return _redact_secrets("".join(lines).rstrip("\n"))


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

    def _parse_reward_json(self) -> dict[str, Any]:
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

        canonical_reward = rewards.get("reward")
        if not is_valid_reward_number(canonical_reward):
            raise VerifierOutputParseError(
                f"Reward JSON file {self._rollout_paths.reward_json_path} "
                "is missing numeric 'reward' between 0.0 and 1.0"
            )

        try:
            return validate_reward_map(rewards, source="reward JSON")
        except ValueError as e:
            raise VerifierOutputParseError(
                f"Reward JSON file {self._rollout_paths.reward_json_path} {e}"
            ) from e

    def _clear_reward_outputs(self) -> None:
        for path in (
            self._rollout_paths.reward_text_path,
            self._rollout_paths.reward_json_path,
        ):
            path.unlink(missing_ok=True)

    async def verify(self) -> VerifierResult:
        """Run the configured verifier and return the reward result."""
        self._clear_reward_outputs()
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

        if self._rollout_paths.reward_text_path.exists():
            rewards = self._parse_reward_text()
        elif self._rollout_paths.reward_json_path.exists():
            rewards = self._parse_reward_json()
        else:
            stdout_tail = _tail_file(self._rollout_paths.test_stdout_path)
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
            if stdout_tail:
                msg += f"\n--- test-stdout.txt (last {_TAIL_LINES} lines) ---\n{stdout_tail}"
            raise RewardFileNotFoundError(msg)

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
