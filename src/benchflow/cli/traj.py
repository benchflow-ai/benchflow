"""``bench traj upload`` / ``setup`` / ``status`` — contribute trajectory captures."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

import click
import typer
from rich.markup import escape

import benchflow.cli._traj_tui as tui
from benchflow.cli._shared import console, print_error
from benchflow.cli._traj_upload_ui import (
    UploadProgressHooks,
    format_bytes,
    render_trajectory_report,
    upload_progress,
)
from benchflow.publish.redact import format_redaction_breakdown
from benchflow.publish.traj_capture import (
    StagedCapture,
    attach_workspace_archive,
    default_source_id,
    finalize_trajectory_capture,
    stage_trajectory_artifacts,
    validate_email,
    validate_github_id,
    validate_source_id,
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

UPGRADE_COMMAND = "uv tool install --python 3.12 --upgrade --force benchflow"

# Three unwrapped lines separated by blank lines: each logical unit is one
# full physical line (no hard wraps mid-sentence), so the URL and the upgrade
# command stay selectable in agents and terminals.
CONTRIBUTOR_PROMPT = (
    "Submit my relevant local Claude Code, Codex, OpenCode, or Cursor session "
    "from the re:Agent e2e agentic science / ai4bio hackathon (last 72 hours, "
    "on this laptop) to the BenchFlow eval prize."
    "\n\n"
    f"1. First make sure the latest benchflow CLI is installed: {UPGRADE_COMMAND}"
    "\n\n"
    f"2. Then read {SKILL_RAW_URL} "
    "and follow it: find a session, open the viewer, and upload only after "
    "I (the human) review it."
)

CONTRIBUTOR_PROMPT_FRAMING = (
    "Send this to your coding agent (it's a prompt for the agent, not steps for you):"
)


def _fetch_latest_version() -> str | None:
    """Return the latest release on PyPI, or ``None`` when unavailable.

    Module-level so tests monkeypatch it and never touch the network.
    """
    import logging

    import httpx

    # The CLI configures INFO logging; keep httpx's "HTTP Request" line for
    # this background check out of the user's terminal.
    logger = logging.getLogger("httpx")
    previous_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        response = httpx.get("https://pypi.org/pypi/benchflow/json", timeout=2.0)
        response.raise_for_status()
        version = response.json()["info"]["version"]
    finally:
        logger.setLevel(previous_level)
    return version if isinstance(version, str) else None


def _installed_version() -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version("benchflow")
    except importlib.metadata.PackageNotFoundError:
        return None


def _maybe_print_update_hint() -> None:
    """Print a one-line upgrade hint when a newer release exists on PyPI.

    Best-effort only: any network, parse, or metadata failure is silent and
    never blocks the command. ``BENCHFLOW_SKIP_UPDATE_CHECK`` disables the
    check entirely (the test suite sets it for hermeticity).
    """
    if os.environ.get("BENCHFLOW_SKIP_UPDATE_CHECK"):
        return
    try:
        installed = _installed_version()
        latest = _fetch_latest_version()
        if installed is None or latest is None:
            return
        from packaging.version import Version

        # Compare release tuples so a dev/prerelease of a newer (or equal)
        # base — e.g. 0.7.1.dev0 against PyPI 0.7.0 — never counts as
        # outdated.
        if Version(installed).release >= Version(latest).release:
            return
    except Exception:
        return
    # Plain print keeps the hint one physical line (no Rich wrapping).
    print(f"A newer BenchFlow ({latest}) is available — run: {UPGRADE_COMMAND}")


# Post-upload storage verification: how long the CLI polls the broker for the
# validator's verdict before handing off to ``bench traj status``. The env
# variable overrides the budget; ``0`` disables waiting entirely (the test
# suite sets it for hermeticity).
DEFAULT_WAIT_SECONDS = 240.0
_WAIT_INITIAL_DELAY = 2.0
_WAIT_MAX_DELAY = 10.0
_WAIT_BACKOFF = 1.5
_WAIT_TRANSIENT_LIMIT = 3

_WAIT_STATE_COPY = {
    "pending": "queued for validation",
    "validating": "validator is checking this capture",
    "unknown": "waiting for the ledger entry",
    "throttled": "status endpoint is busy — backing off",
}

# Module-level indirections so the wait-loop tests control time without
# patching the global ``time`` module.
_sleep = time.sleep
_monotonic = time.monotonic


@dataclass(frozen=True)
class _UploadOptions:
    path: Path | None
    github_id: str | None
    email: str | None
    source_id: str | None
    repo: bool
    direct: bool
    container_url: str | None
    dry_run: bool
    preview_steps: int
    wait: bool = True
    workspace: bool = True
    workspace_dir: Path | None = None


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
            typer.Option("--prompt", help="Print the copy-paste agent prompt and exit"),
        ] = False,
        list_sessions: Annotated[
            bool,
            typer.Option("--list", help="List recent local sessions and exit"),
        ] = False,
    ) -> None:
        """Install the submit skill, or print the prompt to send to an agent."""
        _maybe_print_update_hint()
        if prompt_only:
            _print_contributor_prompt()
            return
        if list_sessions:
            _print_session_hits()
            return
        tui.banner(
            console,
            "traj setup",
            "Install the submit skill and hand your coding agent the prompt",
        )
        interactive = sys.stdin.isatty() and not yes
        if interactive:
            if typer.confirm(
                tui.question_label("Install the trajectory skill into this project?"),
                default=True,
            ):
                _install_project_skill(Path.cwd())
            if shutil.which("npx") and typer.confirm(
                tui.question_label(
                    "Also install for Claude / Codex / Cursor on this machine?"
                ),
                default=False,
            ):
                _run_npx_skill_install()
        else:
            _install_project_skill(Path.cwd())
        _print_contributor_prompt()
        if interactive and typer.confirm(
            tui.question_label("List recent sessions and open the viewer now?"),
            default=False,
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
        repo: Annotated[
            bool,
            typer.Option(
                "--repo/--no-repo",
                help="Tag the upload with the session's repository "
                "(owner/name from its git remote) as the source id",
            ),
        ] = True,
        workspace: Annotated[
            bool,
            typer.Option(
                "--workspace/--no-workspace",
                help="Attach the session's workspace folder as a zip "
                "(auto-detected from the session's recorded cwd; skipped "
                "over 1 GiB)",
            ),
        ] = True,
        workspace_dir: Annotated[
            Path | None,
            typer.Option(
                "--workspace-dir",
                help="Workspace folder to attach instead of auto-detection",
            ),
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
        wait: Annotated[
            bool,
            typer.Option(
                "--wait/--no-wait",
                help="After uploading, wait until the capture is verified "
                "in storage (BENCHFLOW_TRAJ_WAIT_SECONDS overrides the budget)",
            ),
        ] = True,
    ) -> None:
        """Inspect, redact, confirm, and upload trajectory JSONL."""
        _maybe_print_update_hint()
        try:
            _run_upload(
                _UploadOptions(
                    path=path,
                    github_id=github_id,
                    email=email,
                    source_id=source_id,
                    repo=repo,
                    workspace=workspace,
                    workspace_dir=workspace_dir,
                    direct=direct,
                    container_url=container_url,
                    dry_run=dry_run,
                    preview_steps=preview_steps,
                    wait=wait,
                )
            )
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None

    @traj_app.command("status")
    def status(
        digest: Annotated[
            str | None,
            typer.Argument(
                help="Capture digest printed by bench traj upload (sha256:…)"
            ),
        ] = None,
    ) -> None:
        """Check whether an uploaded trajectory is verified in storage."""
        _maybe_print_update_hint()
        try:
            _run_status(digest)
        except ValueError as exc:
            print_error(str(exc))
            raise typer.Exit(1) from None


def _run_upload(options: _UploadOptions) -> None:
    tui.banner(
        console,
        "traj upload",
        "Inspect, redact, confirm — nothing leaves this machine unreviewed",
    )
    prompted = options.path is None
    path = options.path or _prompt_for_path()
    detected: _DetectedRepo | None = None
    if options.source_id is not None:
        source_id = options.source_id
    else:
        if options.repo:
            detected = _detect_repo_slug(path)
        source_id = f"repo/{detected.slug}" if detected else default_source_id(path)
    if detected:
        # Contributor-visible metadata: surface the tag and where it came
        # from so private-repo sessions can opt out before anything leaves
        # the machine. The local path stays terminal-only; the uploaded
        # source id is just repo/<owner>/<name>. Plain print keeps the line
        # one physical line (no Rich wrapping) so the path stays selectable.
        print(
            f"Repo: {detected.slug} "
            f"(from session cwd {detected.session_cwd}; use --no-repo to omit)"
        )
    workspace_dir, workspace_prompted = _resolve_workspace_dir(options, path)
    prompted = prompted or workspace_prompted
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
            masked_categories=artifacts.redaction_categories,
        )
        status.stop()
        render_trajectory_report(report, console=console)

        if workspace_dir is not None:
            with console.status("[bold cyan]Archiving workspace folder…"):
                attach = attach_workspace_archive(artifacts, workspace_dir)
            artifacts = attach.artifacts
            if attach.attached is not None:
                # Plain print: one physical line per machine-readable fact.
                print(
                    f"Workspace attached: {attach.attached.relname} "
                    f"({format_bytes(attach.attached.size_bytes)}, "
                    f"{attach.file_count} files, {attach.excluded_count} "
                    "excluded) — archived as-is; VCS internals and "
                    "secret-like filenames stay local"
                )
            else:
                print(f"Workspace skipped: {attach.skipped_reason}")

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
            tui.question_label("Upload this trajectory?"),
            default=False,
        ):
            console.print("[yellow]Upload cancelled.[/yellow]")
            return
        if not destination.direct:
            # Honest on both cold and warm runs: the broker may already be
            # up. Don't claim it's waking when a retry is just transferring.
            console.print("Uploading… this can take up to a minute; retries are safe.")
        with upload_progress(staged.files, console=console) as hooks:
            result = _publish(staged, destination=destination, hooks=hooks)
        _print_upload_result(staged, result, direct=destination.direct)
        if not destination.direct:
            _confirm_storage(
                result,
                broker_url=destination.url,
                traj_digest=staged.traj_digest,
                wait=options.wait,
            )


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


_SESSION_CWD_SCAN_LINES = 50


@dataclass(frozen=True)
class _DetectedRepo:
    slug: str
    session_cwd: Path


def _detect_repo_slug(path: Path) -> _DetectedRepo | None:
    """Best-effort ``owner/name`` for the repository the session was about.

    Reads the working directory the session recorded (Claude events carry a
    ``cwd`` field; Codex ``session_meta`` payloads do too) and asks that
    directory's git for the ``origin`` remote. The trajectory's own recorded
    cwd is the ONLY provenance source: there is deliberately no fallback to
    the upload invocation directory, which mis-attributed sessions recorded
    outside a repo to whatever checkout the contributor happened to upload
    from. No session cwd, a missing directory, or no GitHub remote all mean
    no tag. Every failure is silent — repo tagging must never break an
    upload — and local-path remotes never produce a tag, so no local
    absolute path can leak into the manifest.
    """
    session_cwd = _session_cwd(path)
    if session_cwd is None or not session_cwd.is_dir():
        return None
    remote = _command_stdout(
        "git", "-C", str(session_cwd), "remote", "get-url", "origin"
    )
    slug = _repo_slug_from_remote(remote) if remote else None
    if slug:
        return _DetectedRepo(slug=slug, session_cwd=session_cwd)
    return None


def _resolve_workspace_dir(
    options: _UploadOptions, session_path: Path
) -> tuple[Path | None, bool]:
    """Pick the workspace folder to attach, mirroring identity resolution.

    Auto-detection reads only the session's recorded cwd (Claude ``cwd``
    events, Codex ``session_meta``) — the same provenance rule as repo
    tagging. When detection fails on a real terminal, one optional prompt
    lets the contributor point at a folder; Enter skips. Returns the folder
    (or ``None``) and whether a human was prompted.
    """
    if not options.workspace:
        return None, False
    if options.workspace_dir is not None:
        explicit = options.workspace_dir.expanduser()
        if not explicit.is_dir():
            raise ValueError(f"workspace folder not found: {explicit}")
        print(f"Workspace: {explicit} (from --workspace-dir)")
        return explicit, False
    recorded = _session_cwd(session_path)
    if recorded is not None and recorded.is_dir():
        print(f"Workspace: {recorded} (from session cwd; use --no-workspace to omit)")
        return recorded, False
    if not sys.stdin.isatty():
        return None, False
    while True:
        raw = typer.prompt(
            tui.field_label("Workspace folder to attach (optional, Enter to skip)"),
            default="",
            show_default=False,
        ).strip()
        if not raw:
            return None, True
        candidate = Path(raw.strip("'\"")).expanduser()
        if candidate.is_dir():
            return candidate, True
        print_error(f"not a folder: {candidate}")


def _session_cwd(path: Path) -> Path | None:
    """Working directory recorded by the first session event that has one."""
    session = path.expanduser()
    if not session.is_file():
        return None
    try:
        with session.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream):
                if line_number >= _SESSION_CWD_SCAN_LINES:
                    break
                body = line.strip()
                if not body:
                    continue
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                cwd = event.get("cwd")
                if isinstance(cwd, str) and cwd:
                    return Path(cwd)
                payload = event.get("payload")
                if (
                    event.get("type") == "session_meta"
                    and isinstance(payload, dict)
                    and isinstance(payload.get("cwd"), str)
                    and payload["cwd"]
                ):
                    return Path(payload["cwd"])
    except OSError:
        return None
    return None


def _repo_slug_from_remote(remote: str) -> str | None:
    """Normalize an https/ssh git remote URL to ``owner/name``.

    Only URL-shaped remotes qualify; a filesystem-path remote returns
    ``None`` so local paths never enter the uploaded source id.
    """
    value = remote.strip()
    if "://" in value:
        _, _, rest = value.partition("://")
        _, _, repo_path = rest.partition("/")
    elif ":" in value and "@" in value.partition(":")[0]:
        repo_path = value.partition(":")[2]
    else:
        return None
    repo_path = repo_path.strip("/").removesuffix(".git")
    segments = [segment for segment in repo_path.split("/") if segment]
    if len(segments) < 2:
        return None
    slug = f"{segments[-2]}/{segments[-1]}"
    try:
        validate_source_id(f"repo/{slug}")
    except ValueError:
        return None
    return slug


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
    # On a real terminal, offer the recent-session picker first; typing a
    # path stays one keystroke away and is the only path everywhere else
    # (agents, pipes, CI), preserving the prompt-driven contract.
    if tui.interactive_terminal():
        selected = _pick_recent_session()
        if selected is not None:
            return selected
    while True:
        raw = typer.prompt(
            tui.field_label("Trajectory JSONL file or trial directory")
        ).strip()
        # Shells wrap dragged-in paths in quotes; accept them as typed.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
            raw = raw[1:-1]
        path = Path(raw).expanduser()
        if path.exists():
            return path
        print_error(f"path not found: {path}")


def _pick_recent_session() -> Path | None:
    """Arrow-key picker over recent sessions; ``None`` falls back to typing."""
    from benchflow.trajectories.sessions import list_recent_sessions

    try:
        hits = list_recent_sessions()
    except OSError:
        return None
    if not hits:
        return None
    options = [
        *_session_options(hits),
        tui.SelectOption(label="Enter a path manually…"),
    ]
    choice = tui.select(console, title="Pick a session to upload", options=options)
    if choice is None or choice >= len(hits):
        return None
    return hits[choice].path


def _session_options(hits) -> list[tui.SelectOption]:
    return [
        tui.SelectOption(
            label=f"{hit.when}  {hit.source:<6} {tui.compact_path(hit.path)}",
            hint=hit.snippet or "(no prompt yet)",
        )
        for hit in hits
    ]


def _prompt_valid(label: str, validate: Callable[[str], str]) -> str:
    while True:
        try:
            return validate(typer.prompt(tui.field_label(label)))
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
    return _UploadDestination(url=_resolve_broker_url(), direct=False)


def _resolve_broker_url() -> str:
    destination = os.environ.get("BENCHFLOW_TRAJ_BROKER_URL") or DEFAULT_TRAJ_BROKER_URL
    if not destination:
        raise ValueError(
            "no trajectory broker is configured; set BENCHFLOW_TRAJ_BROKER_URL, "
            "or use --direct with --container-url/BENCHFLOW_AZURE_CONTAINER_URL "
            "if you have Azure credentials"
        )
    return destination


def _wait_budget_seconds() -> float:
    raw = os.environ.get("BENCHFLOW_TRAJ_WAIT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_WAIT_SECONDS
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return DEFAULT_WAIT_SECONDS


def _confirm_storage(
    result: _PublishResult,
    *,
    broker_url: str,
    traj_digest: str,
    wait: bool,
) -> None:
    """Confirm the capture reached durable storage, or say how to check later.

    The broker's status endpoint reads the validation ledger; ``ingested`` is
    recorded only after the validator promotes every file to
    ``sources/community/<digest>/`` in the Azure container, so a verified
    result here is proof the files are in final storage — not just accepted
    into quarantine.
    """
    if getattr(result, "already_ingested", False):
        _print_verified()
        return
    if not wait:
        return
    budget = _wait_budget_seconds()
    if budget <= 0:
        return
    _wait_for_validation(broker_url=broker_url, traj_digest=traj_digest, budget=budget)


def _print_verified() -> None:
    console.print(
        f"[bold green]{tui.GLYPH_OK} Verified in Azure storage[/] "
        "[dim]— the validator accepted and promoted this capture.[/]"
    )


def _print_status_hint(traj_digest: str) -> None:
    # Plain print keeps the command selectable in agents and terminals.
    print(f"  bench traj status sha256:{traj_digest}")


def _wait_for_validation(*, broker_url: str, traj_digest: str, budget: float) -> None:
    """Poll the broker until the validator's verdict, a timeout, or a 404.

    Success and every soft outcome return normally; only a validator
    rejection raises ``ValueError`` so the command exits 1 — the capture is
    not in the dataset and the detail says what to fix.
    """
    from benchflow.publish.broker import fetch_capture_status

    started = _monotonic()
    delay = _WAIT_INITIAL_DELAY
    transient_failures = 0
    state_copy = _WAIT_STATE_COPY["pending"]
    with console.status(
        f"[bold {tui.ACCENT}]Verifying in storage[/] [dim]— {state_copy}[/]"
    ) as live_status:
        while True:
            elapsed = _monotonic() - started
            remaining = budget - elapsed
            if remaining <= 0:
                live_status.stop()
                console.print(
                    f"[yellow]Still validating after {int(elapsed)}s.[/] "
                    "Your upload is safely queued; check on it any time:"
                )
                _print_status_hint(traj_digest)
                return
            try:
                snapshot = fetch_capture_status(
                    broker_url=broker_url, traj_digest=traj_digest
                )
                transient_failures = 0
            except ValueError:
                transient_failures += 1
                if transient_failures >= _WAIT_TRANSIENT_LIMIT:
                    live_status.stop()
                    console.print(
                        "[yellow]Couldn't confirm the storage status "
                        "(network hiccup).[/] The upload itself succeeded; "
                        "check on it any time:"
                    )
                    _print_status_hint(traj_digest)
                    return
                snapshot = None
            if snapshot is not None:
                if snapshot.status == "ingested":
                    live_status.stop()
                    _print_verified()
                    return
                if snapshot.status == "rejected":
                    live_status.stop()
                    detail = snapshot.detail or "the validator declined this capture"
                    raise ValueError(
                        f"the validator rejected this capture: {detail}. "
                        "Fix the input and run the upload again."
                    )
                if snapshot.status == "unsupported":
                    # Deployed broker predates the status endpoint: keep the
                    # pre-verification behavior without any noise.
                    return
                if snapshot.status == "throttled" and snapshot.retry_after:
                    delay = max(delay, min(snapshot.retry_after, remaining))
                state_copy = _WAIT_STATE_COPY.get(snapshot.status, state_copy)
                live_status.update(
                    f"[bold {tui.ACCENT}]Verifying in storage[/] "
                    f"[dim]— {state_copy} ({int(elapsed)}s)[/]"
                )
            _sleep(min(delay, max(remaining, 0.1)))
            delay = min(delay * _WAIT_BACKOFF, _WAIT_MAX_DELAY)


def _run_status(digest_argument: str | None) -> None:
    from benchflow.publish.broker import fetch_capture_status

    tui.banner(
        console,
        "traj status",
        "Ask the contribution service about an uploaded capture",
    )
    raw = digest_argument or typer.prompt(tui.field_label("Digest (sha256:…)"))
    traj_digest = _normalize_digest(raw)
    broker_url = _resolve_broker_url()
    # Plain print, and before the fetch: the digest stays selectable and
    # visible even when the check errors (stderr interleaves with stdout).
    print(f"Digest: sha256:{traj_digest}")
    with console.status(f"[bold {tui.ACCENT}]Checking capture status…[/]"):
        snapshot = fetch_capture_status(broker_url=broker_url, traj_digest=traj_digest)
    if snapshot.status == "ingested":
        _print_verified()
        return
    if snapshot.status == "pending":
        console.print(
            f"[{tui.ACCENT}]● Queued[/] — uploaded and waiting for the validator. "
            "Check again in a minute."
        )
        return
    if snapshot.status == "validating":
        console.print(
            f"[{tui.ACCENT}]● Validating[/] — the validator is checking this "
            "capture right now. Check again shortly."
        )
        return
    if snapshot.status == "rejected":
        detail = snapshot.detail or "the validator declined this capture"
        raise ValueError(f"the validator rejected this capture: {detail}")
    if snapshot.status == "throttled":
        suffix = (
            f"; retry after {int(snapshot.retry_after)}s"
            if snapshot.retry_after
            else ""
        )
        raise ValueError(f"status checks are rate limited right now{suffix}")
    if snapshot.status == "unsupported":
        raise ValueError(
            "the deployed contribution service does not report capture status "
            "yet; uploads still work, and this command will once the latest "
            "service is deployed"
        )
    raise ValueError(
        "the service has no record of this digest. Check the digest, or run "
        "bench traj upload first."
    )


def _normalize_digest(value: str) -> str:
    body = value.strip().strip("'\"").removeprefix("sha256:")
    if len(body) == 64 and all(char in "0123456789abcdef" for char in body):
        return body
    raise ValueError(
        "digest must be sha256:<64 hex characters>, as printed by bench traj upload"
    )


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
    # One plain, greppable line so the upload skill can lift the breakdown into
    # the viewer's confirm bar (bench eval view --redaction-summary "...").
    breakdown = format_redaction_breakdown(dict(staged.artifact_redaction_categories))
    if breakdown:
        print(f"Masked for you: {breakdown}")
    else:
        print("Masked for you: nothing — no secrets detected")


def _print_contributor_prompt() -> None:
    # Plain print: Rich wrapping would break the unbroken URL line and make
    # the block awkward to copy.
    print(CONTRIBUTOR_PROMPT_FRAMING)
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
        print_error("npx skills add failed; the copy-paste prompt below still works.")


def _print_session_hits() -> None:
    from benchflow.trajectories.sessions import list_recent_sessions

    hits = list_recent_sessions()
    if not hits:
        print("No recent Claude Code, Codex, or trial sessions found.")
        return
    for index, hit in enumerate(hits, start=1):
        snippet = hit.snippet or "(no prompt yet)"
        # Styled facts first, then the path via plain print on its own line:
        # Rich wrapping used to split long session paths mid-token and make
        # them unselectable.
        console.print(
            f"[bold {tui.ACCENT}]{index:>2}[/] [magenta]{hit.source:<6}[/] "
            f"[dim]{hit.when}[/]"
        )
        print(f"   {hit.path}")
        console.print(f"   [dim]└ {escape(snippet)}[/]")


def _interactive_view() -> None:
    from benchflow.trajectories.sessions import list_recent_sessions
    from benchflow.trajectories.viewer import serve

    hits = list_recent_sessions()
    if not hits:
        console.print("No recent sessions found.")
        return
    choice = tui.select(
        console, title="Recent sessions", options=_session_options(hits)
    )
    if choice is None:
        console.print("[yellow]Cancelled.[/]")
        return
    serve(str(hits[choice].path))
