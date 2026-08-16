"""``bench traj upload`` / ``bench traj setup`` — contribute trajectory captures."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error

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
            Path,
            typer.Argument(help="Trajectory JSONL file, directory, or trial directory"),
        ],
        github_id: Annotated[
            str | None,
            typer.Option(
                "--github-id",
                help="GitHub username to credit (inferred from gh/git when omitted)",
            ),
        ] = None,
        email: Annotated[
            str | None,
            typer.Option(
                "--email",
                help="Email stored in the manifest (inferred from git when omitted)",
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
    ) -> None:
        """Validate, redact, and upload trajectory JSONL."""
        from benchflow.publish.azure_blob import upload_capture_direct
        from benchflow.publish.broker import upload_capture_via_broker
        from benchflow.publish.traj_capture import (
            default_source_id,
            stage_trajectory_capture,
        )

        try:
            resolved_github_id, resolved_email = resolve_contributor(
                github_id=github_id, email=email
            )
            selected_source_id = source_id or default_source_id(path)
            if direct:
                destination = container_url or os.environ.get(
                    "BENCHFLOW_AZURE_CONTAINER_URL"
                )
                if not destination:
                    raise ValueError(
                        "--direct requires --container-url or "
                        "BENCHFLOW_AZURE_CONTAINER_URL"
                    )
            else:
                if container_url:
                    raise ValueError("--container-url is only valid with --direct")
                destination = (
                    os.environ.get("BENCHFLOW_TRAJ_BROKER_URL")
                    or DEFAULT_TRAJ_BROKER_URL
                )
                if not destination:
                    raise ValueError(
                        "no trajectory broker is configured; set "
                        "BENCHFLOW_TRAJ_BROKER_URL, or use --direct with "
                        "--container-url/BENCHFLOW_AZURE_CONTAINER_URL if you have "
                        "Azure credentials"
                    )

            with stage_trajectory_capture(
                path,
                source_id=selected_source_id,
                uploaded_by=os.environ.get("BENCHFLOW_TRAJ_UPLOADED_BY"),
                github_id=resolved_github_id,
                email=resolved_email,
            ) as staged:
                if dry_run:
                    _print_dry_run(staged)
                    return
                if not direct:
                    console.print(
                        "Uploading… the first request can take a minute "
                        "while the service wakes up."
                    )
                if direct:
                    result = upload_capture_direct(
                        staged,
                        container_url=destination,
                    )
                else:
                    result = upload_capture_via_broker(
                        staged,
                        broker_url=destination,
                    )
                _print_submit_result(staged, result)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None


def resolve_contributor(*, github_id: str | None, email: str | None) -> tuple[str, str]:
    """Fill missing contributor fields from the local git/gh identity."""
    resolved_github = (
        (github_id or "").strip()
        or os.environ.get("BENCHFLOW_GITHUB_ID", "").strip()
        or _command_stdout("gh", "api", "user", "--jq", ".login")
        or _git_config("github.user")
    )
    resolved_email = (
        (email or "").strip()
        or os.environ.get("BENCHFLOW_EMAIL", "").strip()
        or _git_config("user.email")
    )
    if not resolved_github or not resolved_email:
        raise ValueError(_MISSING_CONTRIBUTOR)
    return resolved_github, resolved_email


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


def _print_dry_run(staged) -> None:
    console.print("[bold]Looks good.[/bold] Nothing was uploaded.")
    console.print(f"Digest: sha256:{staged.traj_digest}")
    for staged_file in staged.files:
        console.print(
            f"  {escape(staged_file.relname)} ({_format_bytes(staged_file.size_bytes)})"
        )
    if staged.ignored:
        console.print(f"Ignored: {escape(', '.join(staged.ignored))}")
    if staged.redaction_replacements:
        console.print(f"Redactions: {staged.redaction_replacements}")


def _print_submit_result(staged, result) -> None:
    digest = f"sha256:{staged.traj_digest}"
    if result.uploaded:
        console.print("[green]Submitted.[/green] We'll review this trajectory.")
    else:
        console.print(
            "[green]Already submitted.[/green] Same trajectory, nothing else to do."
        )
    console.print(f"Digest: {digest}")
    console.print(f"Files: {len(staged.files)}")


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


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
