"""``bench traj upload`` / ``bench traj setup`` — contribute trajectory captures."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

import click
import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error
from benchflow.cli._traj_upload_ui import (
    UploadProgressHooks,
    format_bytes,
    render_trajectory_report,
    upload_progress,
)
from benchflow.publish.traj_capture import (
    StagedCapture,
    default_source_id,
    finalize_trajectory_capture,
    stage_trajectory_artifacts,
    validate_email,
    validate_github_id,
)
from benchflow.publish.traj_report import (
    DEFAULT_PREVIEW_STEPS,
    MAX_PREVIEW_STEPS,
    build_trajectory_report,
)

# The environment variable remains an override for development and disaster
# recovery.
DEFAULT_TRAJ_BROKER_URL: str | None = (
    "https://tasksminer-traj-broker.nicewave-c3abaecf.westus2.azurecontainerapps.io"
)

_MISSING_CONTRIBUTOR = (
    "need a GitHub username and email so we can credit you. Re-run:\n"
    "  bench traj upload PATH --github-id YOUR_ID --email YOU@example.com"
)

SKILL_RAW_URL = (
    "https://raw.githubusercontent.com/benchflow-ai/benchflow/main"
    "/.agents/skills/benchflow-traj-upload/SKILL.md"
)

CONTRIBUTOR_PROMPT = (
    "Submit my best local Claude Code or Codex session to the BenchFlow eval "
    "prize. Read "
    f"{SKILL_RAW_URL} "
    "and follow it: find a session, open the viewer, upload only after I review it."
)


@dataclass(frozen=True)
class _UploadOptions:
    path: Path | None
    github_id: str | None
    email: str | None
    source_id: str | None
    direct: bool
    container_url: str | None
    dry_run: bool
    preview_steps: int


@dataclass(frozen=True)
class _UploadDestination:
    url: str
    direct: bool


class _PublishResult(Protocol):
    @property
    def url(self) -> str: ...

    @property
    def uploaded(self) -> tuple[str, ...]: ...

    @property
    def skipped(self) -> tuple[str, ...]: ...


def register_traj(app: typer.Typer) -> None:
    """Attach the trajectory contribution group to the top-level app."""
    traj_app = typer.Typer(help="Trajectory commands.")
    app.add_typer(traj_app, name="traj", rich_help_panel="Core")

    @traj_app.command("setup")
    def setup(
        yes: Annotated[
            bool,
            typer.Option("--yes", "-y", help="Install the skill without prompts"),
        ] = False,
        prompt_only: Annotated[
            bool,
            typer.Option("--prompt", help="Print the copy-paste agent line and exit"),
        ] = False,
        list_sessions: Annotated[
            bool,
            typer.Option("--list", help="List recent local sessions and exit"),
        ] = False,
    ) -> None:
        """Install the submit skill, or print the line to paste into an agent."""
        if prompt_only:
            _print_contributor_prompt()
            return
        if list_sessions:
            _print_session_hits()
            return
        interactive = sys.stdin.isatty() and not yes
        if interactive:
            if typer.confirm(
                "Install the trajectory skill into this project?", default=True
            ):
                _install_project_skill(Path.cwd())
            if shutil.which("npx") and typer.confirm(
                "Also install for Claude / Codex / Cursor on this machine?",
                default=False,
            ):
                _run_npx_skill_install()
        else:
            _install_project_skill(Path.cwd())
        console.print("Paste this to your agent:")
        _print_contributor_prompt()
        if interactive and typer.confirm(
            "List recent sessions and open the viewer now?", default=False
        ):
            _interactive_view()

    @traj_app.command("upload")
    def upload(
        path: Annotated[
            Path | None,
            typer.Argument(help="Trajectory JSONL file, directory, or trial directory"),
        ] = None,
        github_id: Annotated[
            str | None,
            typer.Option(
                "--github-id",
                help="Contributor GitHub username (inferred from gh/git when omitted)",
            ),
        ] = None,
        email: Annotated[
            str | None,
            typer.Option(
                "--email",
                help="Contributor email stored in the manifest "
                "(inferred from git when omitted)",
            ),
        ] = None,
        source_id: Annotated[
            str | None,
            typer.Option("--source-id", help="Stable contributor source identifier"),
        ] = None,
        direct: Annotated[
            bool,
            typer.Option("--direct", help="Upload with local Azure credentials"),
        ] = False,
        container_url: Annotated[
            str | None,
            typer.Option("--container-url", help="Azure container URL for --direct"),
        ] = None,
        dry_run: Annotated[
            bool,
            typer.Option("--dry-run", help="Validate and stage without uploading"),
        ] = False,
        preview_steps: Annotated[
            int,
            typer.Option(
                "--preview-steps",
                min=0,
                max=MAX_PREVIEW_STEPS,
                help="Number of redacted trajectory steps to preview",
            ),
        ] = DEFAULT_PREVIEW_STEPS,
    ) -> None:
        """Inspect, redact, confirm, and upload trajectory JSONL."""
        try:
            _run_upload(
                _UploadOptions(
                    path=path,
                    github_id=github_id,
                    email=email,
                    source_id=source_id,
                    direct=direct,
                    container_url=container_url,
                    dry_run=dry_run,
                    preview_steps=preview_steps,
                )
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None


def _run_upload(options: _UploadOptions) -> None:
    prompted = options.path is None
    path = options.path or _prompt_for_path()
    source_id = options.source_id or default_source_id(path)
    destination = _resolve_destination(options)

    with (
        console.status(
            "[bold cyan]Inspecting trajectory and masking key values…"
        ) as status,
        stage_trajectory_artifacts(path, source_id=source_id) as artifacts,
    ):
        report = build_trajectory_report(
            artifacts.files,
            masked_values=artifacts.redaction_replacements,
            preview_steps=options.preview_steps,
        )
        status.stop()
        render_trajectory_report(report, console=console)

        github_id, email, identity_prompted = _resolve_contributor(
            options.github_id, options.email
        )
        prompted = prompted or identity_prompted
        staged = finalize_trajectory_capture(
            artifacts,
            uploaded_by=os.environ.get("BENCHFLOW_TRAJ_UPLOADED_BY"),
            github_id=github_id,
            email=email,
            trajectory_report=report.as_manifest_metadata(),
        )
        if options.dry_run:
            _print_dry_run(staged)
            return
        # Confirm only when this session actually prompted a human. Fully
        # resolved invocations (flags or gh/git inference) stay
        # non-interactive so agents driving the CLI never hang on a TTY
        # prompt — their confirmation happens in chat, before this command.
        if prompted and not typer.confirm(
            "Upload this trajectory?",
            default=False,
        ):
            console.print("[yellow]Upload cancelled.[/yellow]")
            return
        if not destination.direct:
            console.print(
                "Uploading… the first request can take a minute "
                "while the service wakes up."
            )
        with upload_progress(staged.files, console=console) as hooks:
            result = _publish(staged, destination=destination, hooks=hooks)
        _print_upload_result(staged, result, direct=destination.direct)


def _resolve_contributor(
    github_id: str | None,
    email: str | None,
) -> tuple[str, str, bool]:
    """Resolve contributor identity: flags, then env/gh/git, then a prompt.

    Returns ``(github_id, email, prompted)``. When a prompt is needed but no
    input is available (an agent piping the command), the click abort becomes
    the one-line ``--github-id`` / ``--email`` fallback instead of a bare
    ``Aborted.``.
    """
    # Explicit flags are validated as given (a malformed flag is an error,
    # never silently replaced); only absent values fall through to inference.
    resolved_github = (github_id or "").strip()
    resolved_github = (
        validate_github_id(resolved_github) if resolved_github else _infer_github_id()
    )
    resolved_email = (email or "").strip()
    resolved_email = (
        validate_email(resolved_email) if resolved_email else _infer_email()
    )
    if resolved_github and resolved_email:
        return resolved_github, resolved_email, False
    try:
        return (
            resolved_github or _prompt_valid("GitHub ID", validate_github_id),
            resolved_email or _prompt_valid("Email", validate_email),
            True,
        )
    except click.exceptions.Abort:
        raise ValueError(_MISSING_CONTRIBUTOR) from None


def _infer_github_id() -> str:
    for candidate in (
        os.environ.get("BENCHFLOW_GITHUB_ID", "").strip(),
        _command_stdout("gh", "api", "user", "--jq", ".login") or "",
        _git_config("github.user") or "",
    ):
        if not candidate:
            continue
        try:
            return validate_github_id(candidate)
        except ValueError:
            continue
    return ""


def _infer_email() -> str:
    for candidate in (
        os.environ.get("BENCHFLOW_EMAIL", "").strip(),
        _git_config("user.email") or "",
    ):
        if not candidate:
            continue
        try:
            return validate_email(candidate)
        except ValueError:
            continue
    return ""


def _git_config(key: str) -> str | None:
    return _command_stdout("git", "config", "--get", key)


def _command_stdout(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value or None


def _prompt_for_path() -> Path:
    while True:
        raw = typer.prompt("Trajectory JSONL file or trial directory").strip()
        # Shells wrap dragged-in paths in quotes; accept them as typed.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1]
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print_error(f"path not found: {path}")


def _prompt_valid(label: str, validate: Callable[[str], str]) -> str:
    while True:
        try:
            return validate(typer.prompt(label))
        except ValueError as exc:
            print_error(str(exc))


def _resolve_destination(options: _UploadOptions) -> _UploadDestination:
    if options.direct:
        destination = options.container_url or os.environ.get(
            "BENCHFLOW_AZURE_CONTAINER_URL"
        )
        if not destination:
            raise ValueError(
                "--direct requires --container-url or BENCHFLOW_AZURE_CONTAINER_URL"
            )
        return _UploadDestination(url=destination, direct=True)
    if options.container_url:
        raise ValueError("--container-url is only valid with --direct")
    destination = os.environ.get("BENCHFLOW_TRAJ_BROKER_URL") or DEFAULT_TRAJ_BROKER_URL
    if not destination:
        raise ValueError(
            "no trajectory broker is configured; set BENCHFLOW_TRAJ_BROKER_URL, "
            "or use --direct with --container-url/BENCHFLOW_AZURE_CONTAINER_URL "
            "if you have Azure credentials"
        )
    return _UploadDestination(url=destination, direct=False)


def _publish(
    staged: StagedCapture,
    *,
    destination: _UploadDestination,
    hooks: UploadProgressHooks,
) -> _PublishResult:
    if destination.direct:
        from benchflow.publish.azure_blob import upload_capture_direct

        return upload_capture_direct(
            staged,
            container_url=destination.url,
            on_file_complete=hooks.on_file_complete,
            on_bytes=hooks.on_bytes,
        )
    from benchflow.publish.broker import upload_capture_via_broker

    return upload_capture_via_broker(
        staged,
        broker_url=destination.url,
        on_file_complete=hooks.on_file_complete,
        on_bytes=hooks.on_bytes,
    )


def _print_upload_result(
    staged: StagedCapture, result: _PublishResult, *, direct: bool
) -> None:
    # Public success copy never includes the destination URL: broker uploads
    # land in a private quarantine inbox nobody can open, and printing it
    # invites people to share a link that 403s. Direct mode is a trusted
    # operator route where the destination is the point.
    if direct:
        if not result.uploaded:
            console.print(
                f"[green]Already uploaded:[/green] {escape(result.url)} (no-op)"
            )
            return
        size = sum(item.size_bytes for item in staged.files)
        console.print(
            "[green]Uploaded trajectory:[/green] "
            f"{escape(result.url)} "
            f"({len(result.uploaded)} uploaded, {len(result.skipped)} skipped, "
            f"{format_bytes(size)}, {staged.redaction_replacements} redactions)"
        )
        return
    digest = f"sha256:{staged.traj_digest}"
    if result.uploaded:
        size = sum(item.size_bytes for item in staged.files)
        console.print("[green]Submitted.[/green] We'll review this trajectory.")
        console.print(
            f"Digest: {digest}\n"
            f"Files: {len(staged.files)} ({format_bytes(size)}, "
            f"{staged.redaction_replacements} redactions)"
        )
    else:
        console.print(
            "[green]Already submitted.[/green] Same trajectory, nothing else to do."
        )
        console.print(f"Digest: {digest}")


def _print_dry_run(staged: StagedCapture) -> None:
    console.print("[bold]Dry run[/bold] — no files uploaded")
    console.print(f"Digest: sha256:{staged.traj_digest}")
    for staged_file in staged.files:
        console.print(
            f"  {escape(staged_file.relname)} ({format_bytes(staged_file.size_bytes)})"
        )
    if staged.ignored:
        console.print(f"Ignored: {escape(', '.join(staged.ignored))}")
    console.print(f"Redactions: {staged.redaction_replacements}")


def _print_contributor_prompt() -> None:
    # One physical line so people can copy it. Rich wrapping would break that.
    print(CONTRIBUTOR_PROMPT)


def _install_project_skill(project_root: Path) -> Path:
    dest_dir = project_root / ".agents" / "skills" / "benchflow-traj-upload"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "SKILL.md"
    dest.write_text(_skill_markdown(), encoding="utf-8")
    console.print(f"Installed {dest}")
    return dest


def _skill_markdown() -> str:
    local = _local_skill_md()
    if local is not None:
        return local.read_text(encoding="utf-8")
    import httpx

    response = httpx.get(SKILL_RAW_URL, timeout=20.0, follow_redirects=True)
    response.raise_for_status()
    return response.text


def _local_skill_md() -> Path | None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".agents" / "skills" / "benchflow-traj-upload" / "SKILL.md"
        if candidate.is_file():
            return candidate
    return None


def _run_npx_skill_install() -> None:
    completed = subprocess.run(
        [
            "npx",
            "--yes",
            "skills",
            "add",
            "benchflow-ai/benchflow",
            "--skill",
            "benchflow-traj-upload",
        ],
        check=False,
    )
    if completed.returncode != 0:
        print_error("npx skills add failed; the copy-paste line below still works.")


def _print_session_hits() -> None:
    from benchflow.trajectories.sessions import list_recent_sessions

    hits = list_recent_sessions()
    if not hits:
        console.print("No recent Claude Code, Codex, or trial sessions found.")
        return
    for index, hit in enumerate(hits, start=1):
        snippet = hit.snippet or "(no prompt yet)"
        console.print(f"{index}. [{hit.source}] {hit.when}  {hit.path}\n   {snippet}")


def _interactive_view() -> None:
    from benchflow.trajectories.sessions import list_recent_sessions
    from benchflow.trajectories.viewer import serve

    hits = list_recent_sessions()
    if not hits:
        console.print("No recent sessions found.")
        return
    _print_session_hits()
    choice = typer.prompt("Which number?", default="1")
    try:
        index = int(choice)
    except ValueError:
        print_error("Need a session number.")
        raise typer.Exit(1) from None
    if index < 1 or index > len(hits):
        print_error("That number is not in the list.")
        raise typer.Exit(1)
    serve(str(hits[index - 1].path))
