"""Automatic rubric-review state owned outside the general rollout engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from benchflow.models import RolloutResult
from benchflow.review.config import find_task_rubric
from benchflow.review.policy import RubricReviewConfig
from benchflow.review.workspace import export_workspace_delta, record_workspace_baseline

logger = logging.getLogger(__name__)


@dataclass
class AutomaticRubricReview:
    """State machine for one source rollout's automatic review phase."""

    task_path: Path
    rubric_path: Path
    policy: RubricReviewConfig
    _baseline_ready: bool = False
    _workspace_exported: bool = False
    _workspace_error: str | None = None

    @classmethod
    def discover(
        cls, task_path: Path, policy: RubricReviewConfig
    ) -> AutomaticRubricReview | None:
        """Create the lifecycle only when an enabled task ships a review rubric."""

        if not policy.enabled:
            return None
        rubric_path = find_task_rubric(task_path)
        return (
            cls(task_path=task_path, rubric_path=rubric_path, policy=policy)
            if rubric_path is not None
            else None
        )

    async def record_baseline(self, sandbox: Any, workspace: str) -> float:
        """Record the pre-solver workspace without failing the source rollout."""

        started = datetime.now()
        try:
            await record_workspace_baseline(sandbox, workspace)
            self._baseline_ready = True
        except Exception as exc:
            self._workspace_error = str(exc)
            logger.error("Automatic rubric-review workspace setup failed: %s", exc)
        return (datetime.now() - started).total_seconds()

    async def export_workspace(
        self,
        sandbox: Any,
        workspace: str,
        *,
        artifacts_dir: Path,
    ) -> float | None:
        """Capture solver outputs once, permitting a cleanup-phase retry."""

        if not self._baseline_ready or self._workspace_exported:
            return None
        started = datetime.now()
        try:
            await export_workspace_delta(
                sandbox,
                workspace,
                host_artifacts_dir=artifacts_dir,
            )
            self._workspace_exported = True
            self._workspace_error = None
        except Exception as exc:
            self._workspace_error = str(exc)
            logger.error("Automatic rubric-review workspace export failed: %s", exc)
        return (datetime.now() - started).total_seconds()

    async def integrate(
        self,
        result: RolloutResult,
        *,
        rollout_dir: Path,
        source_environment: str,
    ) -> RolloutResult:
        """Run the isolated reviewer and fail closed on orchestration defects."""

        from benchflow.review.integration import (
            integrate_task_rubric_review,
            mark_task_rubric_review_error,
        )

        try:
            return await integrate_task_rubric_review(
                result,
                rollout_dir=rollout_dir,
                task_path=self.task_path,
                rubric_path=self.rubric_path,
                source_environment=source_environment,
                policy=self.policy,
                workspace_error=self._workspace_error,
            )
        except Exception as exc:
            logger.error("Automatic rubric review failed unexpectedly", exc_info=True)
            return mark_task_rubric_review_error(
                result,
                rollout_dir=rollout_dir,
                rubric_path=self.rubric_path,
                policy=self.policy,
                message=str(exc),
            )
