"""Reviewer flags for evaluation and scoring commands."""

from typing import Annotated

import typer

from benchflow._utils.config import normalize_agent_idle_timeout
from benchflow.review.options import ReviewerConfig

ReviewerAgentOption = Annotated[
    str | None, typer.Option("--reviewer-agent", help="Rubric reviewer harness")
]
ReviewerModelOption = Annotated[
    str | None, typer.Option("--reviewer-model", help="Rubric reviewer model")
]
ReviewerEffortOption = Annotated[
    str | None,
    typer.Option(
        "--reviewer-reasoning-effort", help="Rubric reviewer reasoning effort"
    ),
]
ReviewerSandboxOption = Annotated[
    str | None,
    typer.Option("--reviewer-sandbox", help="Rubric reviewer sandbox backend"),
]
ReviewerTimeoutOption = Annotated[
    int | None,
    typer.Option(
        "--reviewer-timeout-sec", min=1, help="Reviewer execution budget in seconds"
    ),
]
ReviewerIdleTimeoutOption = Annotated[
    str | None,
    typer.Option(
        "--reviewer-idle-timeout",
        help=(
            "Abort a reviewer prompt after this many idle seconds (default: 600). "
            "Pass 0 or 'none' to disable the idle watchdog and leave "
            "--reviewer-timeout-sec in charge."
        ),
    ),
]
ReviewerConcurrencyOption = Annotated[
    int | None,
    typer.Option(
        "--reviewer-concurrency", min=1, help="Max concurrent rubric reviewers"
    ),
]
ReviewerImageOption = Annotated[
    str | None, typer.Option("--reviewer-image", help="Rubric reviewer container image")
]
ReviewerEnvOption = Annotated[
    list[str] | None,
    typer.Option(
        "--reviewer-agent-env", help="Reviewer environment variable (KEY=VALUE)"
    ),
]
ReviewerNetworkOption = Annotated[
    bool | None,
    typer.Option(
        "--reviewer-open-network", help="Allow unrestricted reviewer network access"
    ),
]


def reviewer_idle_timeout(value: str | None) -> int | None:
    """``--reviewer-idle-timeout`` as ReviewerConfig.idle_timeout_sec: None when
    the flag is absent, 0 when it disables the watchdog."""
    if value is None:
        return None
    try:
        return normalize_agent_idle_timeout(value) or 0
    except ValueError as exc:
        raise typer.BadParameter(
            "expected whole seconds, 0 or 'none'", param_hint="--reviewer-idle-timeout"
        ) from exc


def reviewer_from_cli(**options: object) -> ReviewerConfig | None:
    """Retain only explicit overrides, preserving a YAML reviewer's other fields."""
    supplied = {key: value for key, value in options.items() if value is not None}
    if not supplied:
        return None
    try:
        return ReviewerConfig.model_validate(supplied)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="reviewer options") from exc
