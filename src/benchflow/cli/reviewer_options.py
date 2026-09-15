"""Reviewer flags for evaluation and scoring commands."""

from typing import Annotated

import typer

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


def reviewer_from_cli(**options: object) -> ReviewerConfig | None:
    """Retain only explicit overrides, preserving a YAML reviewer's other fields."""
    supplied = {key: value for key, value in options.items() if value is not None}
    if not supplied:
        return None
    try:
        return ReviewerConfig.model_validate(supplied)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="reviewer options") from exc
