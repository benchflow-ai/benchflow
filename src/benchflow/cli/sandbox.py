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

import contextlib
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape
from rich.table import Table

from benchflow._utils.sandbox_providers import extra_install_hint
from benchflow.cli._options import SandboxOption
from benchflow.cli._shared import DaytonaUnconfigured, console, print_error

# `bench sandbox`'s help (cli/main.py's layout copies it). Only Daytona's
# sandboxes outlive a run; Docker's are made and removed per trial.
SANDBOX_HELP = (
    "Daytona sandboxes: create, list and clean up (cleanup also removes "
    "leftover Docker containers)."
)

# What `bench sandbox create` takes.
_TASK_DIR_NEXT = "pass a task folder (it holds task.md or task.toml)"


def sandbox_create(task_dir: Path, sandbox: str) -> None:
    """Create an environment object from a task directory (does not start it)."""
    from benchflow.runtime import Environment

    if not task_dir.is_dir():
        print_error(f"Not a directory: {task_dir}", next_step=_TASK_DIR_NEXT)
        raise typer.Exit(2)
    try:
        env = Environment.from_task(task_dir, sandbox=sandbox)
    except (OSError, ValueError) as e:
        # An existing dir with no task document — or one where task.md/task.toml
        # is itself a directory — reaches Task's unguarded read_text(), raising
        # FileNotFoundError / IsADirectoryError (both OSError). Surface a clean
        # error instead of a raw traceback.
        print_error(
            f"Not a valid task directory {task_dir}: {e}", next_step=_TASK_DIR_NEXT
        )
        raise typer.Exit(1) from None
    except RuntimeError as e:
        # An unknown --sandbox backend (UnsupportedTaskFeatureError, a RuntimeError
        # subclass) and a missing optional sandbox dependency both raise a
        # RuntimeError carrying a clean, user-facing message. Surface it without a
        # traceback, matching how `sandbox list`/`cleanup` handle the same cases.
        print_error(
            str(e), next_step="`bench doctor --sandbox <name>` checks a sandbox"
        )
        raise typer.Exit(1) from None
    console.print(f"[green]Environment created:[/green] {escape(str(env))}")
    console.print(f"  Task:    {env.task_path}")
    console.print(f"  Sandbox: {env.sandbox}")
    console.print(
        "  Use [cyan]bench run[/cyan] for CLI runs, or pass to [cyan]bf.run()[/cyan]"
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

    Daytona is the only backend with persistent, listable sandboxes; a Docker
    sandbox is torn down after its trial unless --keep-sandbox kept it (its trial
    folder then holds kept-sandbox.json). When the optional
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
            f"([cyan]{escape(extra_install_hint('sandbox-daytona'))}[/cyan]). "
            "A Docker sandbox is deleted after its trial, unless --keep-sandbox "
            "kept it: its trial folder then holds kept-sandbox.json (the run's summary "
            "names those trials). [cyan]bench sandbox cleanup[/cyan] "
            "removes what a killed run left behind."
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
        print_error(
            f"Skipping AgentCore cleanup: {exc}",
            next_step="check this machine's AWS credentials and "
            "BENCHFLOW_AGENTCORE_REGION",
        )
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
            ["docker", "rmi", "--no-prune", name],
            capture_output=True,
            text=True,
            timeout=120,
        )
        counts["deleted" if removed.returncode == 0 else "failed"] += 1
    return counts


def cleanup_docker_leftovers(*, dry_run: bool) -> bool:
    """Remove (or list) the Docker containers and networks of BenchFlow
    processes that died without tearing them down: a ``kill -9``, an OOM
    kill, a preempted VM (DF-p3-eval-runner-9). The rule is the one every
    Docker job sweeps with when it starts and ends
    (:func:`benchflow.sandbox._docker_sweep.sweep_leftovers`): only resources
    labelled with a process of this host that is provably gone; a live run's,
    another host's or an unlabelled one is never touched. False without a
    Docker CLI."""
    if shutil.which("docker") is None:
        return False
    from benchflow.sandbox._docker_sweep import sweep_leftovers

    result = sweep_leftovers(dry_run=dry_run)
    verb = "would remove" if dry_run else "removed"
    # A dry run counts what it would ask for; a real one what Docker removed.
    containers = result.containers if dry_run else result.removed_containers
    networks = result.networks if dry_run else result.removed_networks
    running = f" ({len(result.running)} running)" if result.running else ""
    projects = f": {', '.join(result.projects)}" if result.projects else ""
    console.print(
        "Docker leftovers of exited BenchFlow processes: "
        f"{len(containers)} container(s){running}, "
        f"{len(networks)} network(s) {verb}{escape(projects)}"
    )
    if result.orphans:
        console.print(
            "Still running, not removed: "
            f"{escape(', '.join(result.orphans))} (`docker rm -f` removes them)"
        )
    return True


def sandbox_cleanup(
    *,
    dry_run: bool,
    max_age_minutes: int,
    all_mine: bool = False,
    names: tuple[str, ...] | list[str] = (),
) -> None:
    """Clean up orphaned Daytona sandboxes, stale AgentCore runtimes, and
    the Docker leftovers of BenchFlow processes that died.

    Daytona and AgentCore are opt-in extras, so a missing SDK is a no-op
    rather than an error. Docker sandboxes are torn down per run; a run that
    was killed leaves its containers behind, and those are removed here
    whatever their age (their process is gone). ``all_mine`` deletes every sandbox of
    this ``BENCHFLOW_DAYTONA_OWNER`` whatever its age or activity, except a
    STARTED one or one held by a live run lease; ``names`` (``--id``) deletes those sandboxes of
    this owner scope whatever their lease. Exits 1
    when an installed backend could not be listed or cleaned.
    """
    from benchflow.sandbox.daytona_reaper import _benchflow_owner_scope

    if all_mine and _benchflow_owner_scope() is None:
        print_error(
            "--all deletes every sandbox of one owner, and no owner is set",
            next_step="set BENCHFLOW_DAYTONA_OWNER (cleanup never touches "
            "sandboxes without your owner label)",
        )
        raise typer.Exit(2)
    cleaned_any = False
    backend_failed = False

    if _daytona_sdk_available():
        from benchflow.cli import main as cli_main

        try:
            cli_main._cleanup_daytona_sandboxes(
                dry_run=dry_run,
                max_age_minutes=max_age_minutes,
                all_mine=all_mine,
                names=frozenset(names),
            )
            cleaned_any = True
        except DaytonaUnconfigured:
            # The SDK is installed (an extra) but no key is set: no Daytona
            # account holds sandboxes of this machine's runs.
            console.print(
                "Daytona: no key set (DAYTONA_API_KEY), so there is nothing to clean there."
            )
        except (Exception, typer.Exit) as exc:
            # The Daytona SDK is installed but unusable (a bad DAYTONA_API_KEY,
            # an API error). Report and continue: a second backend may still
            # have resources to reclaim — AgentCore runtimes consume a
            # 100-per-account quota — then exit 1 so a teardown step fails.
            print_error(
                "Daytona cleanup failed"
                + ("" if isinstance(exc, typer.Exit) else f": {exc}"),
                next_step="`bench doctor --sandbox daytona` checks the Daytona key",
            )
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

    if cleanup_docker_leftovers(dry_run=dry_run):
        cleaned_any = True

    if backend_failed:
        raise typer.Exit(1)
    if not cleaned_any:
        console.print(
            "Nothing to clean up. Neither the Daytona SDK "
            f"([cyan]{escape(extra_install_hint('sandbox-daytona'))}[/cyan]) "
            "nor the AgentCore extra "
            f"([cyan]{escape(extra_install_hint('sandbox-agentcore'))}[/cyan]) "
            "is installed; only those backends leave resources between runs."
        )


# --- a sandbox kept with --keep-sandbox --------------------------------------

_KEPT_NEXT = (
    "pass the trial folder of a run with --keep-sandbox (it holds kept-sandbox.json)"
)
# The shell attach opens when no command is given: bash where the image has it.
_ATTACH_SHELL = [
    "/bin/sh",
    "-c",
    "command -v bash >/dev/null 2>&1 && exec bash -l || exec sh -l",
]


def _kept_record(target: Path) -> tuple[Path, dict]:
    import json

    from benchflow.rollout._keep_sandbox import RECORD_NAME

    path = target if target.name == RECORD_NAME else target / RECORD_NAME
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print_error(f"No kept sandbox at {target}: {exc}", next_step=_KEPT_NEXT)
        raise typer.Exit(2) from None
    if not isinstance(record, dict) or record.get("backend") not in (
        "docker",
        "daytona",
    ):
        print_error(f"{path} is not a kept-sandbox record", next_step=_KEPT_NEXT)
        raise typer.Exit(2)
    if record.get("state") == "released":
        print_error(
            f"The sandbox in {path} was released at {record.get('released_at')}",
            next_step="run the task again with --keep-sandbox to get a new one",
        )
        raise typer.Exit(1)
    return path, record


def _docker_env(record: dict) -> dict[str, str]:
    import os

    env = dict(os.environ)
    for key, name in (
        ("docker_host", "DOCKER_HOST"),
        ("docker_context", "DOCKER_CONTEXT"),
    ):
        if record.get(key) and not env.get(name):
            env[name] = str(record[key])
    return env


def _kept_docker_ids(record: dict, *kinds: str) -> list[str]:
    """The ids of the kept Compose project's resources (``ps -a``, or
    ``network ls`` / ``volume ls``), by Compose's own project label."""
    label = f"label=com.docker.compose.project={record['project']}"
    if kinds[0] == "container":
        argv = ["docker", "ps", "-aq", "--filter", label, *kinds[1:]]
    else:
        argv = ["docker", kinds[0], "ls", "-q", "--filter", label]
    out = subprocess.run(
        argv, capture_output=True, text=True, timeout=60, env=_docker_env(record)
    )
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip() or f"{' '.join(argv)} failed")
    return [line for line in out.stdout.split() if line]


def _attach_docker(record: dict, command: list[str]) -> int:
    import sys

    ids = _kept_docker_ids(
        record,
        "container",
        "--filter",
        f"label=com.docker.compose.service={record.get('service') or 'main'}",
    )
    if not ids:
        print_error(
            f"The kept sandbox {record['project']} has no main container any more",
            next_step="`bench sandbox release <trial>` clears the record",
        )
        return 1
    env = _docker_env(record)
    subprocess.run(["docker", "start", ids[0]], capture_output=True, env=env)
    argv = ["docker", "exec", "-u", "0"]
    if not command:
        tty = sys.stdin.isatty() and sys.stdout.isatty()
        argv.append("-it" if tty else "-i")
    argv += [ids[0], *(command or _ATTACH_SHELL)]
    return subprocess.run(argv, env=env).returncode


def _attach_daytona(record: dict, command: list[str]) -> int:
    import os
    import shlex
    import sys
    import tempfile

    from benchflow.cli import main as cli_main

    client = cli_main._daytona_client_or_exit()
    sandbox = client.get(record["sandbox_id"])
    if "started" not in str(getattr(sandbox, "state", "")).lower():
        # BenchFlow's auto-stop stopped it while it sat idle.
        sandbox.start(timeout=180)
    access = sandbox.create_ssh_access(expires_in_minutes=60)
    fd, config = tempfile.mkstemp(prefix="benchflow_attach_", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(
                "Host benchflow-kept\n  HostName ssh.app.daytona.io\n"
                f"  User {access.token}\n  StrictHostKeyChecking no\n"
                "  UserKnownHostsFile /dev/null\n  LogLevel ERROR\n"
            )
        os.chmod(config, 0o600)
        argv = ["ssh", "-F", config]
        if not command and sys.stdin.isatty() and sys.stdout.isatty():
            argv.append("-t")
        argv.append("benchflow-kept")
        if command:
            argv.append(shlex.join(command))
        return subprocess.run(argv).returncode
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(config)
        with contextlib.suppress(Exception):
            sandbox.revoke_ssh_access(access.token)


def sandbox_attach(target: Path, command: list[str] | None = None) -> None:
    """Open a root shell in a kept sandbox, or run *command* there."""
    _path, record = _kept_record(target)
    try:
        if record["backend"] == "docker":
            rc = _attach_docker(record, list(command or ()))
        else:
            rc = _attach_daytona(record, list(command or ()))
    except Exception as exc:
        print_error(
            f"Could not attach to the kept sandbox: {exc}",
            next_step="`bench sandbox release <trial>` deletes it if it is broken",
        )
        raise typer.Exit(1) from None
    raise typer.Exit(rc)


def sandbox_release(target: Path) -> None:
    """Delete a kept sandbox and mark its record released."""
    from benchflow.review.persistence import write_json_atomic

    path, record = _kept_record(target)
    if record["backend"] == "docker":
        from benchflow.sandbox._docker_sweep import unkeep_project

        env = _docker_env(record)
        containers = _kept_docker_ids(record, "container")
        if containers:
            subprocess.run(
                ["docker", "rm", "-f", "-v", *containers], capture_output=True, env=env
            )
        # Docker can still list a container it is removing; the networks
        # and volumes go only once their containers have.
        for _ in range(20):
            if not _kept_docker_ids(record, "container"):
                break
            time.sleep(0.5)
        else:
            print_error(
                f"Docker still lists containers of {record['project']}",
                next_step="`docker ps -a --filter "
                f"label=com.docker.compose.project={record['project']}` shows them",
            )
            raise typer.Exit(1)
        left: list[tuple[str, str]] = []
        for kind in ("network", "volume"):
            ids = _kept_docker_ids(record, kind)
            if not ids:
                continue
            run = subprocess.run(
                ["docker", kind, "rm", *ids], capture_output=True, text=True, env=env
            )
            if run.returncode != 0 and _kept_docker_ids(record, kind):
                left.append((kind, (run.stderr or "").strip()))
        if left:
            # The record stays kept, so the release can be run again.
            kind, why = left[0]
            print_error(
                f"Docker did not remove the {' and '.join(k + 's' for k, _ in left)} "
                f"of {record['project']}"
                + (f": {why}" if why else "")
                + "; the sandbox is still marked kept",
                next_step=f"`docker {kind} ls --filter label=com.docker.compose."
                f"project={record['project']}` shows what is left; run `bench sandbox "
                "release` again once it is free",
            )
            raise typer.Exit(1)
        unkeep_project(record["project"])
        copy = record.get("task_copy")
        # Only BenchFlow's own per-trial task copy (tempfile prefix).
        if (
            copy
            and Path(copy).name.startswith("benchflow-task-")
            and Path(copy).is_dir()
        ):
            shutil.rmtree(copy, ignore_errors=True)
        what = f"Docker project {record['project']}"
    else:
        from benchflow.cli import main as cli_main

        client = cli_main._daytona_client_or_exit()
        try:
            client.delete(client.get(record["sandbox_id"]))
        except Exception as exc:
            if "not found" not in str(exc).lower() and "404" not in str(exc):
                print_error(
                    f"Could not delete Daytona sandbox {record['sandbox_id']}: {exc}",
                    next_step="`bench sandbox cleanup --id <sandbox id>` deletes it",
                )
                raise typer.Exit(1) from None
        what = f"Daytona sandbox {record['sandbox_id']}"
    record.update(state="released", released_at=datetime.now(UTC).isoformat())
    write_json_atomic(path, record)
    console.print(f"Released the kept sandbox ({escape(what)}).")


def register_sandbox(app: typer.Typer) -> None:
    """Attach the ``sandbox`` command group to the top-level benchflow app."""
    sandbox_app = typer.Typer(help=SANDBOX_HELP)
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
        """List active Daytona sandboxes (kept Docker sandboxes: their trial folders hold kept-sandbox.json)."""
        sandbox_list_local(show_all=show_all, as_json=as_json)

    @sandbox_app.command(
        "attach",
        context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    )
    def sandbox_attach_cmd(
        ctx: typer.Context,
        trial: Annotated[
            Path,
            typer.Argument(
                help="The trial folder of a run with --keep-sandbox, or its kept-sandbox.json"
            ),
        ],
    ) -> None:
        """Open a root shell in a kept sandbox; with a command after --, run it there and exit with its status."""
        sandbox_attach(trial, list(ctx.args))

    @sandbox_app.command("release")
    def sandbox_release_cmd(
        trial: Annotated[
            Path,
            typer.Argument(
                help="The trial folder of a run with --keep-sandbox, or its kept-sandbox.json"
            ),
        ],
    ) -> None:
        """Delete a sandbox kept with --keep-sandbox."""
        sandbox_release(trial)

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
                "age or activity, except a STARTED one or one a live run still "
                "holds (a CI teardown step); others are never touched",
            ),
        ] = False,
        names: Annotated[
            list[str] | None,
            typer.Option(
                "--id",
                help="Delete this Daytona sandbox (repeatable), even when a live "
                "run lease holds it; only sandboxes of your owner scope",
            ),
        ] = None,
    ) -> None:
        """Clean up orphaned sandboxes and stale shared runtimes."""
        sandbox_cleanup(
            dry_run=dry_run,
            max_age_minutes=max_age_minutes,
            all_mine=all_mine,
            names=tuple(names or ()),
        )
