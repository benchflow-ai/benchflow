"""Launch checks for a materialized task.md draft 2 package, before any sandbox starts.

Some fields BenchFlow honors for a scripted seat but not for an agent: an
agent's tool-call or token budget, which a script never approaches, and
``[agent] user``, which the oracle runs as while an agent runs as the run's
``--sandbox-user``. The package records them (``metadata.taskmd``), and
:func:`check_launch` refuses an agent run of such a task by name. It also
refuses a run whose model judges could not be served, before the solver
spends anything (docs/runtime/judging.md, "Preflight").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchflow.taskmd.materialize import taskmd_metadata

SCRIPTED_SEATS = ("oracle", "nop")


class TaskMdLaunchRefused(ValueError):
    """This run cannot honor the task; the message names each field."""


def check_launch(
    task_path: Path, *, primary_agent: str, sandbox_user: str | None
) -> None:
    """Raise :class:`TaskMdLaunchRefused` when this run cannot honor the package."""
    meta = taskmd_metadata(Path(task_path))
    if meta is None:
        return
    reasons: list[str] = []
    scripted = primary_agent in SCRIPTED_SEATS
    if not scripted:
        for item in meta.get("agent_refused") or []:
            reasons.append(f"{item.get('field')}: {item.get('reason')}")
        user = meta.get("agent_user")
        if user is not None and not _same_user(user, sandbox_user):
            reasons.append(
                f"[agent] user: the task's agent runs as {user!r}, and this run's agent runs as "
                f"{sandbox_user or 'root'!r}; pass --sandbox-user {user}"
            )
    reasons += judge_preflight(Path(task_path), meta)
    if reasons:
        raise TaskMdLaunchRefused(
            f"{Path(task_path).name}: this run cannot honor these task.md draft 2 fields, so it does not start:\n"
            + "\n".join(f"  - {r}" for r in reasons)
        )


def _same_user(declared: Any, sandbox_user: str | None) -> bool:
    if str(declared) in ("root", "0"):
        return sandbox_user in (None, "root")
    return sandbox_user is not None and str(declared) == sandbox_user


def judge_preflight(task_path: Path, meta: dict[str, Any]) -> list[str]:
    """Why the package's model judges could not be served by this run, if they could not."""
    if meta.get("grading") != "rubric":
        return []
    from benchflow.taskmd import judging
    from benchflow.taskmd._vendor import judgeprompt as jp
    from benchflow.taskmd._vendor import taskmd as ref
    from benchflow.taskmd.materialize import judge_package_dir, shared_rubrics

    pkg = judge_package_dir(task_path)
    document = ref.parse(pkg)
    rubric = document.rubric or {}
    try:
        criteria = jp.merged_criteria(rubric, shared_rubrics(task_path, meta))
    except jp.JudgePromptError as exc:
        return [f"verifier/rubric.json: {exc}"]
    roles = sorted({c.get("judge") for c in criteria} & {"llm", "agent"})
    if not roles:
        return []
    reasons = []
    if judging.credentials_from_env() is None:
        reasons.append(
            f"[verifier.judges]: the {', '.join(roles)} judge needs ANTHROPIC_API_KEY, or a Claude Code "
            "OAuth token in CLAUDE_CODE_OAUTH_TOKEN, in BenchFlow's environment"
        )
    for role in roles:
        settings, _ = ref.resolve_judge(document.config, role, None, pkg)
        model = judging.model_for(role, settings)
        if model is None:
            reasons.append(
                f"[verifier.judges.{role}] model: none is named; set one, or {judging.MODEL_ENV}"
            )
        elif not model.startswith(("claude-", "anthropic/")):
            reasons.append(
                f"[verifier.judges.{role}] model: {model!r}; BenchFlow runs judge-loop@1 over the "
                f"Anthropic Messages API only (substitute one with {judging.MODEL_ENV})"
            )
    return reasons
