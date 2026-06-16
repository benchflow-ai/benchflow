"""Task package boundary for native and compatibility task layouts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from benchflow.task.config import TaskConfig
from benchflow.task.document import TaskDocument
from benchflow.task.prompts import TaskPromptPlan, compile_task_prompt_plan
from benchflow.task.runtime_capabilities import (
    UnsupportedTaskFeature,
    validate_task_runtime_support,
)
from benchflow.task.runtime_view import TaskRuntimeView

CompatibilityTarget = Literal["harbor"]


@dataclass(frozen=True)
class TaskPackage:
    """One authoritative package-level view of a task directory.

    ``TaskRuntimeView`` selects the executable config, prompt, oracle, verifier,
    and source hashes. ``TaskPackage`` wraps that selected view with the parsed
    native document, selected verifier package, and sandbox capability evidence
    so callers do not rebuild those decisions independently.
    """

    view: TaskRuntimeView
    document: TaskDocument | None = None
    prompt_plan: TaskPromptPlan | None = None
    runtime_issues: tuple[UnsupportedTaskFeature, ...] = field(default_factory=tuple)

    @classmethod
    def from_task_dir(
        cls,
        task_dir: str | Path,
        *,
        sandbox: str | None = None,
    ) -> TaskPackage:
        view = TaskRuntimeView.from_task_dir(task_dir)
        document = _load_document(view)
        return cls._from_view(view, document=document, sandbox=sandbox)

    @classmethod
    def from_task(
        cls,
        task: Any,
        *,
        task_dir: str | Path | None = None,
        sandbox: str | None = None,
    ) -> TaskPackage:
        view = TaskRuntimeView.from_task(task, task_dir=task_dir)
        document = getattr(task, "document", None)
        return cls._from_view(
            view,
            document=document if isinstance(document, TaskDocument) else None,
            sandbox=sandbox,
        )

    @classmethod
    def _from_view(
        cls,
        view: TaskRuntimeView,
        *,
        document: TaskDocument | None,
        sandbox: str | None,
    ) -> TaskPackage:
        prompt_plan = compile_task_prompt_plan(
            document,
            fallback_prompt=view.prompt,
            scenes=view.scenes,
        )
        runtime_issues: tuple[UnsupportedTaskFeature, ...] = ()
        if sandbox is not None:
            runtime_subject: TaskDocument | TaskConfig = document or view.config
            runtime_issues = tuple(
                validate_task_runtime_support(
                    runtime_subject,
                    sandbox=sandbox,
                    task_dir=view.task_dir,
                )
            )
        return cls(
            view=view,
            document=document,
            prompt_plan=prompt_plan,
            runtime_issues=runtime_issues,
        )

    @property
    def task_dir(self) -> Path:
        return self.view.task_dir

    @property
    def runtime_supported(self) -> bool:
        return not self.runtime_issues

    def compatibility_export_report(
        self,
        *,
        target: CompatibilityTarget = "harbor",
        output_dir: str | Path | None = None,
    ) -> Any:
        """Build the target split-layout export report for this package."""

        from benchflow.task.export import build_compatibility_export_report

        return build_compatibility_export_report(
            self.task_dir,
            target=target,
            output_dir=output_dir,
        )

    def to_dict(self) -> dict[str, Any]:
        verifier_document = self.view.verifier_document
        selected_strategy = (
            verifier_document.selected_strategy
            if verifier_document is not None
            else None
        )
        return {
            "task_dir": str(self.task_dir),
            "entrypoint": self.view.entrypoint,
            "prompt_path": (
                "task.md" if self.view.entrypoint == "task.md" else "instruction.md"
            ),
            "selected_oracle_dir": _rel_or_none(self.task_dir, self.view.oracle_dir),
            "selected_verifier_dir": _rel_or_none(
                self.task_dir, self.view.verifier_dir
            ),
            "verifier_document": (
                {
                    "name": verifier_document.name,
                    "default_strategy": verifier_document.default_strategy,
                    "selected_strategy": (
                        {
                            "name": selected_strategy.name,
                            "type": selected_strategy.type,
                        }
                        if selected_strategy is not None
                        else None
                    ),
                }
                if verifier_document is not None
                else None
            ),
            "runtime_supported": self.runtime_supported,
            "runtime_issues": [asdict(issue) for issue in self.runtime_issues],
            "prompt_plan": (
                self.prompt_plan.to_dict() if self.prompt_plan is not None else None
            ),
            "compatibility": asdict(self.view.compatibility),
            "source_hashes": dict(self.view.source_hashes),
        }


def _load_document(view: TaskRuntimeView) -> TaskDocument | None:
    path = view.task_dir / "task.md"
    if not path.exists():
        return None
    return TaskDocument.from_path(path)


def _rel_or_none(root: Path, path: Path) -> str | None:
    if not path.exists():
        return None
    return path.relative_to(root).as_posix()
