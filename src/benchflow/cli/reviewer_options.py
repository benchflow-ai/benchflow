"""Reviewer flags for evaluation and scoring commands.

The flags are defined at import, so `bench eval run` can declare them; what
they are checked against (the reviewer config model) is imported only when a
command reads them, which keeps `bench --help` from loading the review stack.
"""

from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
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
ReviewerStorageOption = Annotated[
    int | None,
    typer.Option(
        "--reviewer-storage-mb",
        min=1,
        help=(
            "Reviewer sandbox disk in MB (default: the evidence it holds plus "
            "2048, at least 4096, at most the sandbox backend's limit)"
        ),
    ),
]
ReviewerMemoryOption = Annotated[
    int | None,
    typer.Option(
        "--reviewer-memory-mb",
        min=1,
        help="Reviewer sandbox memory in MB (default: 2048)",
    ),
]
ReviewerSetupTimeoutOption = Annotated[
    int | None,
    typer.Option(
        "--reviewer-setup-timeout",
        min=1,
        help=(
            "Seconds for the reviewer sandbox's setup, its model proxy install "
            "among it (default: the run's --sandbox-setup-timeout)"
        ),
    ),
]
ReviewerEvidenceTimeoutOption = Annotated[
    int | None,
    typer.Option(
        "--reviewer-evidence-timeout",
        min=1,
        help=(
            "Seconds to unpack, and again to check, the workspace evidence in "
            "the reviewer sandbox (default: 1 per MiB, at least 600)"
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
ReviewerPdfToolsOption = Annotated[
    bool | None,
    typer.Option(
        "--reviewer-pdf-tools",
        help=(
            "Install PyMuPDF beside numpy and pypdf in the reviewer sandbox, to "
            "read and render a submitted PDF (default: off)"
        ),
    ),
]
ReviewerSubmittedFileRuleOption = Annotated[
    bool | None,
    typer.Option(
        "--reviewer-submitted-file-rule",
        help=(
            "Tell the reviewer to judge a written deliverable by the file the "
            "agent submitted: read it first, cite pages, and take a compiled "
            "PDF over its sources (default: off)"
        ),
    ),
]


def reviewer_idle_timeout(value: str | None) -> int | None:
    """``--reviewer-idle-timeout`` as ReviewerConfig.idle_timeout_sec: None when
    the flag is absent, 0 when it disables the watchdog."""
    from benchflow._utils.config import normalize_agent_idle_timeout

    if value is None:
        return None
    try:
        return normalize_agent_idle_timeout(value) or 0
    except ValueError as exc:
        raise typer.BadParameter(
            "expected whole seconds, 0 or 'none'", param_hint="--reviewer-idle-timeout"
        ) from exc


def reviewer_from_cli(**options: object) -> "ReviewerConfig | None":
    """Retain only explicit overrides, preserving a YAML reviewer's other fields."""
    from benchflow.review.options import ReviewerConfig

    supplied = {key: value for key, value in options.items() if value is not None}
    if not supplied:
        return None
    try:
        return ReviewerConfig.model_validate(supplied)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="reviewer options") from exc
