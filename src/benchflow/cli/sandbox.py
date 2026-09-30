"""``bench sandbox`` — local sandbox lifecycle (create / list / cleanup).

This is the local execution side of the framework: provision a task as a
runnable environment on a docker/daytona/modal **sandbox** backend, list active
sandboxes, and reap stale ones. It was previously ``bench environment``; that
name now reads as a misnomer (hosted-environment browsing moved to
``bench hub``), so the group is renamed to ``sandbox`` — ``bench
environment`` stays as a hidden deprecated alias group through 0.6.

The command bodies live here as plain functions so the deprecated
``bench environment`` aliases (``cli/environment.py``) can delegate to the same
logic without a fork. The Daytona client + reaper deliberately resolve through
``benchflow.cli.main`` so tests that monkeypatch those names keep working.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._options import SandboxOption
from benchflow.cli._shared import console, print_error


def sandbox_create(task_dir: Path, sandbox: str) -> None:
    """Create an environment object from a task directory (does not start it)."""
    from benchflow.runtime import Environment

    if not task_dir.is_dir():
        print_error(f"Not a directory: {task_dir}")
        raise typer.Exit(1)
    try:
        env = Environment.from_task(task_dir, sandbox=sandbox)
    except (OSError, ValueError) as e:
        # An existing dir with no task document — or one where task.md/task.toml
        # is itself a directory — reaches Task's unguarded read_text(), raising
        # FileNotFoundError / IsADirectoryError (both OSError). Surface a clean
        # error instead of a raw traceback.
        print_error(f"Not a valid task directory {task_dir}: {e}")
        raise typer.Exit(1) from None
    except RuntimeError as e:
        # An unknown --sandbox backend (UnsupportedTaskFeatureError, a RuntimeError
        # subclass) and a missing optional sandbox dependency both raise a
        # RuntimeError carrying a clean, user-facing message. Surface it without a
        # traceback, matching how `sandbox list`/`cleanup` handle the same cases.
        print_error(str(e))
        raise typer.Exit(1) from None
    console.print(f"[green]Environment created:[/green] {escape(str(env))}")
    console.print(f"  Task:    {env.task_path}")
    console.print(f"  Sandbox: {env.sandbox}")
    console.print(
        "  Use [cyan]bench eval run[/cyan] for CLI runs, or pass to [cyan]bf.run()[/cyan]"
    )


def _daytona_sdk_available() -> bool:
    """True if the optional Daytona SDK can be imported.

    A plain import rather than ``importlib.util.find_spec``: the test suite
    injects a fake ``daytona`` module into ``sys.modules`` that has no
    ``__spec__``, which makes ``find_spec`` raise/return None. An import sees the
    fake (and a real install) alike.

    The anyio compat shim is applied first — the Daytona sync client imports
    ``anyio.AsyncContextManagerMixin`` at import time, which the pinned anyio may
    not expose. Without it a *real* install would look absent here (and so would
    ``build_sync_client`` further down). Mirrors that bootstrap.
    """
    try:
        from benchflow.sandbox.daytona import _ensure_daytona_anyio_compat

        _ensure_daytona_anyio_compat()
        import daytona  # noqa: F401

        return True
    except ImportError:
        return False


def sandbox_list_local(*, show_all: bool = False, as_json: bool = False) -> None:
    """List active off-box sandboxes (Daytona).

    Daytona is the only backend with persistent, listable sandboxes; Docker
    sandboxes are ephemeral (built and torn down per run). When the optional
    Daytona SDK is not installed there is nothing to list — an empty result, not
    an error (mirroring how ``sandbox create`` degrades on a missing extra).
    With ``BENCHFLOW_DAYTONA_OWNER`` set, only that owner's sandboxes are listed
    unless ``show_all`` (the shared-key case).
    """
    if not _daytona_sdk_available():
        if as_json:
            typer.echo("[]")
            return
        console.print(
            "No active sandboxes. Daytona is the only backend with persistent, "
            "listable sandboxes, and its SDK is not installed "
            "([cyan]uv sync --extra sandbox-daytona[/cyan]). Docker sandboxes are "
            "ephemeral and created per run."
        )
        return
    import json

    from benchflow.cli import main as cli_main
    from benchflow.sandbox.daytona_reaper import (
        _BENCHFLOW_OWNER_LABEL,
        _benchflow_owner_scope,
    )

    d = cli_main._daytona_client_or_exit()
    owner = None if show_all else _benchflow_owner_scope()
    now = datetime.now(UTC)
    rows = []
    # daytona SDK >=0.18: ``list()`` yields an auto-paginating Iterator[Sandbox].
    for sb in d.list():
        labels = getattr(sb, "labels", None)
        sb_owner = (
            labels.get(_BENCHFLOW_OWNER_LABEL) if isinstance(labels, dict) else None
        )
        if owner is not None and sb_owner != owner:
            continue
        age = None
        if sb.created_at:
            created = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
            age = round((now - created).total_seconds() / 60, 1)
        rows.append(
            {
                "id": sb.id,
                "state": str(sb.state).rsplit(".", 1)[-1],
                "age_minutes": age,
                "owner": sb_owner,
                "target": str(getattr(sb, "target", "") or ""),
            }
        )
    if as_json:
        typer.echo(json.dumps(rows, indent=2))
        return
    title = f"Sandboxes of owner {owner}" if owner else "Active Sandboxes"
    table = Table(title=title)
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("State", style="green")
    table.add_column("Age")
    table.add_column("Owner")
    table.add_column("Target")
    for row in rows:
        table.add_row(
            row["id"],
            row["state"],
            "" if row["age_minutes"] is None else f"{row['age_minutes']:.0f}m",
            row["owner"] or "",
            row["target"][:40],
        )
    console.print(table)
    note = f" (owner {owner}; --all for every sandbox on the account)" if owner else ""
    console.print(f"\n[bold]{len(rows)} sandbox(es)[/bold]{note}", soft_wrap=True)


def _cleanup_agentcore_runtimes(*, dry_run: bool, max_age_minutes: int) -> bool:
    """Reap stale AgentCore runtimes. False when AgentCore is not in play.

    AgentCore runtimes are shared across rollouts and intentionally outlive the
    run that created them, so unlike Docker they accumulate — and *Total Agents
    per Account* defaults to only 100.

    Gated on ``BENCHFLOW_AGENTCORE_ROLE_ARN`` rather than merely on an
    importable SDK. ``boto3`` arrives with several unrelated extras, so without
    this gate ``bench sandbox cleanup`` would reach out to AWS on any machine
    that happens to have credentials configured — including during tests.
    """
    import os

    if not os.environ.get("BENCHFLOW_AGENTCORE_ROLE_ARN"):
        return False
    try:
        import boto3

        from benchflow.sandbox.agentcore_reaper import reap_stale_runtimes
    except ImportError:
        return False

    region = os.environ.get("BENCHFLOW_AGENTCORE_REGION") or "us-west-2"
    try:
        control = boto3.Session(region_name=region).client("bedrock-agentcore-control")
        report = reap_stale_runtimes(
            control, max_age_minutes=max_age_minutes, dry_run=dry_run
        )
    except Exception as exc:
        # Missing/expired credentials are a "nothing to do here", not a crash
        # in the middle of a cleanup that may still have Daytona work to do.
        print_error(f"Skipping AgentCore cleanup: {exc}")
        return False

    verb = "Would delete" if dry_run else "Deleted"
    console.print(f"AgentCore ({region}): {report.summary()}")
    if report.deleted:
        console.print(f"  {verb}: {', '.join(report.deleted)}")
    return True


def cleanup_docker_snapshots(
    *, dry_run: bool, max_age_minutes: int
) -> dict[str, int] | None:
    """Remove ``bf-snap-*`` images (branch snapshots and automatic
    checkpoints) older than ``max_age_minutes``; None without a Docker CLI
    or daemon. An image a container still uses is refused by ``docker rmi``
    and counted as failed."""
    if shutil.which("docker") is None:
        return None
    try:
        listing = subprocess.run(
            [
                "docker",
                "images",
                "--filter",
                "reference=bf-snap-*",
                "--format",
                "{{.ID}}\t{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if listing.returncode != 0:
        return None
    counts = {"found": 0, "deleted": 0, "skipped": 0, "failed": 0}
    now = datetime.now(UTC)
    for line in listing.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        _, name, created = parts
        try:
            stamp = datetime.strptime(
                " ".join(created.split()[:3]), "%Y-%m-%d %H:%M:%S %z"
            )
        except ValueError:
            continue
        counts["found"] += 1
        age = (now - stamp).total_seconds() / 60
        if age < max_age_minutes:
            counts["skipped"] += 1
            continue
        if dry_run:
            console.print(
                f"  [dim]{escape(name)}[/dim] age={age:.0f}m [red](delete)[/red]"
            )
            counts["deleted"] += 1
            continue
        removed = subprocess.run(
            ["docker", "rmi", name], capture_output=True, text=True, timeout=120
        )
        counts["deleted" if removed.returncode == 0 else "failed"] += 1
    return counts


def sandbox_cleanup(
    *, dry_run: bool, max_age_minutes: int, all_mine: bool = False
) -> None:
    """Clean up orphaned Daytona sandboxes and stale AgentCore runtimes.

    Both are opt-in extras, so a missing SDK is a no-op rather than an error;
    only these two backends leave anything behind between runs. Docker
    sandboxes are torn down per run. ``all_mine`` deletes every sandbox of
    this ``BENCHFLOW_DAYTONA_OWNER`` whatever its age or activity. Exits 1
    when an installed backend could not be listed or cleaned.
    """
    from benchflow.sandbox.daytona_reaper import _benchflow_owner_scope

    if all_mine and _benchflow_owner_scope() is None:
        print_error(
            "--all deletes every sandbox of one owner: set BENCHFLOW_DAYTONA_OWNER "
            "(it never touches sandboxes without your owner label)"
        )
        raise typer.Exit(2)
    cleaned_any = False
    backend_failed = False

    # Sandboxes (Docker projects, Daytona sandboxes) that a BenchFlow process
    # on this machine started and was killed before tearing down: its lease
    # file still lists them (benchflow.sandbox.leases).
    from benchflow.sandbox.leases import lease_dir, lease_state

    dead = []
    for path in sorted(lease_dir().glob("*.json")) if lease_dir().is_dir() else []:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and lease_state(data.get("process")) == "gone":
            dead.append(data)
    if dead:
        cleaned_any = True
        listed = [
            r for d in dead for r in d.get("resources") or [] if isinstance(r, dict)
        ]
        if dry_run:
            for resource in listed:
                console.print(
                    f"  [dim]{escape(str(resource.get('provider')))}:"
                    f"{escape(str(resource.get('id')))}[/dim] "
                    "(left by a killed process) [red](delete)[/red]"
                )
            console.print(
                f"Sandboxes left by killed processes: {len(listed)} would be deleted"
            )
        else:
            from benchflow.sandbox.leases import reap_dead_leases

            reaped = reap_dead_leases()
            console.print(
                f"Sandboxes left by killed processes: {len(reaped['deleted'])} deleted"
                + (f", {len(reaped['failed'])} failed" if reaped["failed"] else "")
            )
            backend_failed = backend_failed or bool(reaped["failed"])

    if _daytona_sdk_available():
        from benchflow.cli import main as cli_main

        try:
            cli_main._cleanup_daytona_sandboxes(
                dry_run=dry_run, max_age_minutes=max_age_minutes, all_mine=all_mine
            )
            cleaned_any = True
        except (Exception, typer.Exit) as exc:
            # The Daytona SDK is installed but unusable (typically no or a bad
            # DAYTONA_API_KEY). Report and continue: a second backend may still
            # have resources to reclaim — AgentCore runtimes consume a
            # 100-per-account quota — then exit 1 so a teardown step fails.
            print_error(f"Daytona cleanup failed: {exc}")
            backend_failed = True

    if _cleanup_agentcore_runtimes(dry_run=dry_run, max_age_minutes=max_age_minutes):
        cleaned_any = True

    docker = cleanup_docker_snapshots(dry_run=dry_run, max_age_minutes=max_age_minutes)
    if docker is not None:
        cleaned_any = True
        verb = "would delete" if dry_run else "deleted"
        console.print(
            f"Docker bf-snap images: {docker['found']} found, {docker['deleted']} "
            f"{verb}, {docker['skipped']} younger than {max_age_minutes}m"
            + (f", {docker['failed']} still in use" if docker["failed"] else "")
        )

    if backend_failed:
        raise typer.Exit(1)
    if not cleaned_any:
        console.print(
            "Nothing to clean up. Neither the Daytona SDK "
            "([cyan]uv sync --extra sandbox-daytona[/cyan]) nor the AgentCore "
            "extra ([cyan]uv sync --extra sandbox-agentcore[/cyan]) is "
            "installed; only those backends leave resources between runs."
        )


def register_sandbox(app: typer.Typer) -> None:
    """Attach the ``sandbox`` command group to the top-level benchflow app."""
    sandbox_app = typer.Typer(help="Local sandbox lifecycle (create / list / cleanup).")
    app.add_typer(sandbox_app, name="sandbox", rich_help_panel="Environments")

    @sandbox_app.command("create")
    def sandbox_create_cmd(
        task_dir: Annotated[
            Path,
            typer.Argument(
                help="Task directory with task.md or task.toml + Dockerfile"
            ),
        ],
        sandbox: SandboxOption = "daytona",
    ) -> None:
        """Create an environment from a task directory (does not start it)."""
        sandbox_create(task_dir, sandbox)

    @sandbox_app.command("list")
    def sandbox_list_cmd(
        show_all: Annotated[
            bool,
            typer.Option(
                "--all",
                help="Every sandbox on the account, not only BENCHFLOW_DAYTONA_OWNER's",
            ),
        ] = False,
        as_json: Annotated[
            bool, typer.Option("--json", help="Print the sandboxes as JSON")
        ] = False,
    ) -> None:
        """List active sandboxes (Daytona; Docker sandboxes are ephemeral)."""
        sandbox_list_local(show_all=show_all, as_json=as_json)

    @sandbox_app.command("cleanup")
    def sandbox_cleanup_cmd(
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="List sandboxes without deleting")
        ] = False,
        max_age_minutes: Annotated[
            int,
            typer.Option(
                "--max-age",
                min=0,
                help="Delete sandboxes older than N minutes",
            ),
        ] = 1440,
        all_mine: Annotated[
            bool,
            typer.Option(
                "--all",
                help="Delete every sandbox of BENCHFLOW_DAYTONA_OWNER, whatever its "
                "age or activity (a CI teardown step); others are never touched",
            ),
        ] = False,
    ) -> None:
        """Clean up orphaned sandboxes and stale shared runtimes."""
        sandbox_cleanup(
            dry_run=dry_run, max_age_minutes=max_age_minutes, all_mine=all_mine
        )
