"""``bench sandbox`` — local sandbox lifecycle (create / check / list / cleanup).

This is the local execution side of the framework: provision a task as a
runnable environment on a docker/daytona/modal/cua **sandbox** backend, validate
task/runtime/provider readiness, list active sandboxes, and reap stale ones. It
was previously ``bench environment``; that name now reads as a misnomer
(hosted-environment browsing moved to ``bench hub env``), so the group is renamed
to ``sandbox`` — ``bench environment`` stays as a hidden deprecated alias group
through 0.6.

The command bodies live here as plain functions so the deprecated
``bench environment`` aliases (``cli/environment.py``) can delegate to the same
logic without a fork. The Daytona client + reaper deliberately resolve through
``benchflow.cli.main`` so tests that monkeypatch those names keep working.

The adapter-aware create/check/list/cleanup flows (adapter detection, ``--json``
machine-readable reports, the Cua runtime probe, and the docker/cua backends for
``list``/``cleanup``) are the 0.7 universal-environment surface, re-homed here
onto the canonical ``bench sandbox`` group.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Coroutine, Mapping
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.adapters.inbound import UnsupportedInboundTaskError
from benchflow.cli._adapter_reporting import unsupported_adapter_task_or_exit
from benchflow.cli._options import SandboxOption
from benchflow.cli._shared import console
from benchflow.environment.adapters import environment_adapter_report
from benchflow.sandbox.providers import SANDBOX_PROVIDER_SET

# Sandboxes `bench sandbox create/check/list` can target: the canonical
# docker/daytona/modal registry plus the 0.7 `cua` universal-environment
# backend. Derived from the registry so it can't drift from the provider set.
# (cua-cloud / macos-ios-simulator are reachable via `bench eval create`, but
# the sandbox create/check provider-readiness path only models cua today.)
_SUPPORTED_SANDBOXES = SANDBOX_PROVIDER_SET | {"cua"}
_DOCKER_OWNED_LABEL = "benchflow.owned=true"


def sandbox_create(
    task_dir: Path,
    sandbox: str,
    *,
    dry_run: bool = False,
    output_json: bool = False,
) -> None:
    """Create an environment object from a task directory (does not start it).

    Validates the task (and, for inbound adapter formats, materializes a native
    task package) before constructing the environment. ``--dry-run`` validates
    without creating; ``--json`` emits a machine-readable create report.
    """
    from benchflow.runtime import Environment

    if not task_dir.is_dir():
        console.print(f"[red]Not a directory: {escape(str(task_dir))}[/red]")
        raise typer.Exit(1)
    _validate_sandbox_or_exit(sandbox)
    if dry_run:
        adapter_source = _check_task_runtime_or_exit(
            task_dir,
            sandbox=sandbox,
            output_json=output_json,
        )
        environment_adapter = environment_adapter_report(
            benchmark_adapter=adapter_source,
            sandbox=sandbox,
            provider_mode=_environment_adapter_provider_mode(sandbox),
        )
        if output_json:
            typer.echo(
                json.dumps(
                    {
                        "status": "dry-run",
                        "task": str(task_dir),
                        "task_name": task_dir.name,
                        "adapter": adapter_source,
                        "environment_adapter": environment_adapter.to_dict(),
                        "sandbox": sandbox,
                        "created": False,
                    }
                )
            )
            return
        console.print("[green]Environment dry-run passed[/green]")
        console.print(f"  Task:    {task_dir}")
        if adapter_source:
            console.print(f"  Adapter: {adapter_source}")
        console.print(f"  Environment adapter: {environment_adapter.name}")
        console.print(f"  Sandbox: {sandbox}")
        console.print("  Created: no")
        return

    from benchflow.cli._inbound_task_target import native_task_target

    try:
        with native_task_target(task_dir) as target:
            env = Environment.from_task(target.path, sandbox=sandbox)
            adapter_source = target.adapter_source
    except UnsupportedInboundTaskError as e:
        unsupported_adapter_task_or_exit(task_dir, e, output_json=output_json)
    except (FileNotFoundError, NotADirectoryError, ValueError) as e:
        # An existing dir with no task document reaches Task.__init__'s
        # unguarded read_text() — surface a clean error instead of a raw
        # FileNotFoundError traceback ending at instruction.md/task.md.
        if output_json:
            typer.echo(
                json.dumps(
                    {
                        "status": "invalid",
                        "task": str(task_dir),
                        "task_name": task_dir.name,
                        "sandbox": sandbox,
                        "error": str(e),
                    }
                )
            )
            raise typer.Exit(1) from None
        console.print(
            f"[red]Not a valid task directory {escape(str(task_dir))}:[/red] "
            f"{escape(str(e))}"
        )
        raise typer.Exit(1) from None

    environment_adapter = environment_adapter_report(
        benchmark_adapter=adapter_source,
        sandbox=sandbox,
        provider_mode=_environment_adapter_provider_mode(sandbox),
    )
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": "created",
                    "task": str(task_dir),
                    "task_name": task_dir.name,
                    "adapter": adapter_source,
                    "environment_adapter": environment_adapter.to_dict(),
                    "sandbox": env.sandbox,
                    "created": True,
                    "native": (
                        "materialized-temporary" if adapter_source else "native-task"
                    ),
                }
            )
        )
        return
    console.print(f"[green]Environment created:[/green] {escape(str(env))}")
    console.print(f"  Task:    {escape(str(task_dir))}")
    if adapter_source:
        console.print(f"  Adapter: {escape(str(adapter_source))}")
        console.print("  Native:  materialized task.md package (temporary)")
    console.print(f"  Environment adapter: {escape(str(environment_adapter.name))}")
    console.print(f"  Sandbox: {env.sandbox}")
    console.print(
        "  Use [cyan]bench eval create[/cyan] for CLI runs, or pass to [cyan]bf.run()[/cyan]"
    )


def sandbox_check(
    task_or_ref: Path,
    sandbox: str,
    *,
    probe_runtime: bool = False,
    output_json: bool = False,
) -> None:
    """Check task/runtime/provider readiness without starting a sandbox."""
    _validate_sandbox_or_exit(sandbox)
    if not task_or_ref.exists():
        if output_json:
            typer.echo(
                json.dumps(
                    {
                        "status": "error",
                        "task": str(task_or_ref),
                        "task_name": task_or_ref.name,
                        "sandbox": sandbox,
                        "reason": "task directory not found",
                    }
                )
            )
            raise typer.Exit(1)
        console.print(f"[red]Task directory not found: {task_or_ref}[/red]")
        raise typer.Exit(1)
    adapter_source = _check_task_runtime_or_exit(
        task_or_ref,
        sandbox=sandbox,
        output_json=output_json,
    )
    provider = _check_provider_or_exit(
        sandbox,
        quiet=output_json,
        runtime_probe=probe_runtime,
        output_json=output_json,
    )
    environment_adapter = environment_adapter_report(
        benchmark_adapter=adapter_source,
        sandbox=sandbox,
        provider_mode=_environment_adapter_provider_mode(sandbox),
        runtime_probe_ready=_provider_runtime_probe_ready(provider),
    )
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": "ready",
                    "task": str(task_or_ref),
                    "task_name": task_or_ref.name,
                    "adapter": adapter_source,
                    "environment_adapter": environment_adapter.to_dict(),
                    "sandbox": sandbox,
                    "provider": provider,
                }
            )
        )
        return
    console.print("[green]Environment check passed[/green]")
    console.print(f"  Task:    {task_or_ref}")
    if adapter_source:
        console.print(f"  Adapter: {adapter_source}")
    console.print(f"  Environment adapter: {environment_adapter.name}")
    console.print(f"  Sandbox: {sandbox}")


def sandbox_list_local(sandbox: str = "daytona", *, output_json: bool = False) -> None:
    """List active local sandboxes for the selected ``--sandbox`` backend."""
    from benchflow.cli import main as cli_main

    _validate_sandbox_or_exit(sandbox)
    if sandbox == "cua":
        _list_cua_environments(output_json=output_json)
        return
    if sandbox == "docker":
        _list_docker_environments(output_json=output_json)
        return
    if sandbox != "daytona":
        console.print(
            f"[red]sandbox list is only supported for docker, daytona, and cua today; got {sandbox!r}[/red]"
        )
        raise typer.Exit(1)

    d = cli_main._daytona_client_or_exit()
    table = Table(title="Active Sandboxes")
    table.add_column("ID", style="cyan")
    table.add_column("State", style="green")
    table.add_column("Age")
    table.add_column("Target")

    now = datetime.now(UTC)
    total = 0
    # daytona SDK >=0.18: ``list()`` yields an auto-paginating
    # Iterator[Sandbox] (was a paged ``list(page=, limit=)`` -> page object
    # with ``.items``).
    for sb in d.list():
        total += 1
        age = ""
        if sb.created_at:
            created = datetime.fromisoformat(sb.created_at.replace("Z", "+00:00"))
            mins = (now - created).total_seconds() / 60
            age = f"{mins:.0f}m"
        target = getattr(sb, "target", "") or ""
        table.add_row(sb.id[:12] + "…", str(sb.state), age, str(target)[:40])

    console.print(table)
    console.print(f"\n[bold]{total} sandbox(es)[/bold]")


def sandbox_cleanup(
    *,
    sandbox: str = "daytona",
    dry_run: bool,
    max_age_minutes: int,
    output_json: bool = False,
) -> None:
    """Clean up orphaned provider sandboxes for the selected backend."""
    from benchflow.cli import main as cli_main

    _validate_sandbox_or_exit(sandbox)
    if sandbox == "cua":
        _cleanup_cua_environments(
            dry_run=dry_run,
            max_age_minutes=max_age_minutes,
            output_json=output_json,
        )
        return
    if sandbox == "docker":
        _cleanup_docker_environments(
            dry_run=dry_run,
            max_age_minutes=max_age_minutes,
            output_json=output_json,
        )
        return
    if sandbox != "daytona":
        console.print(
            f"[red]sandbox cleanup is only supported for docker, daytona, and cua today; got {sandbox!r}[/red]"
        )
        raise typer.Exit(1)
    if output_json:
        console.print("[red]--json cleanup is only supported for cua today[/red]")
        raise typer.Exit(1)

    cli_main._cleanup_daytona_sandboxes(
        dry_run=dry_run, max_age_minutes=max_age_minutes
    )
    _cleanup_daytona_snapshots(dry_run=dry_run)


def register_sandbox(app: typer.Typer) -> None:
    """Attach the ``sandbox`` command group to the top-level benchflow app."""
    sandbox_app = typer.Typer(
        help="Local sandbox lifecycle (create / check / list / cleanup)."
    )
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
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run", help="Validate without creating an environment object"
            ),
        ] = False,
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Emit a machine-readable create report"),
        ] = False,
    ) -> None:
        """Create an environment from a task directory (does not start it)."""
        sandbox_create(task_dir, sandbox, dry_run=dry_run, output_json=output_json)

    @sandbox_app.command("check")
    def sandbox_check_cmd(
        task_or_ref: Annotated[
            Path,
            typer.Argument(help="Task directory to validate for an environment"),
        ],
        sandbox: SandboxOption = "daytona",
        probe_runtime: Annotated[
            bool,
            typer.Option(
                "--probe-runtime",
                help=(
                    "Start a bounded provider runtime probe for sandboxes that "
                    "support it. For Cua this creates/connects, runs shell, "
                    "file transfer, screenshot, dimensions, display metadata, "
                    "then cleans up."
                ),
            ),
        ] = False,
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Emit a machine-readable readiness report"),
        ] = False,
    ) -> None:
        """Check task/runtime/provider readiness without starting a sandbox."""
        sandbox_check(
            task_or_ref,
            sandbox,
            probe_runtime=probe_runtime,
            output_json=output_json,
        )

    @sandbox_app.command("list")
    def sandbox_list_cmd(
        sandbox: SandboxOption = "daytona",
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Emit raw JSON for list results"),
        ] = False,
    ) -> None:
        """List active local sandboxes."""
        sandbox_list_local(sandbox, output_json=output_json)

    @sandbox_app.command("cleanup")
    def sandbox_cleanup_cmd(
        sandbox: SandboxOption = "daytona",
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="List sandboxes without deleting")
        ] = False,
        max_age_minutes: Annotated[
            int, typer.Option("--max-age", help="Delete sandboxes older than N minutes")
        ] = 1440,
        output_json: Annotated[
            bool,
            typer.Option("--json", help="Emit a machine-readable cleanup report"),
        ] = False,
    ) -> None:
        """Clean up orphaned provider sandboxes."""
        sandbox_cleanup(
            sandbox=sandbox,
            dry_run=dry_run,
            max_age_minutes=max_age_minutes,
            output_json=output_json,
        )


def _validate_sandbox_or_exit(sandbox: str) -> None:
    if sandbox not in _SUPPORTED_SANDBOXES:
        console.print(
            f"[red]Invalid --sandbox {sandbox!r}: choose docker, daytona, modal, or cua[/red]"
        )
        raise typer.Exit(1)


def _cleanup_daytona_snapshots(dry_run: bool) -> None:
    """Reap leaked ``bf-snap-*`` Daytona snapshots (display wrapper).

    The sandbox reaper never touches snapshots, so a snapshot whose owning
    sandbox's ``stop()`` never ran leaks against the account. Scoped by the
    ``bf-snap-`` name prefix benchflow stamps — Daytona snapshots have no labels.
    """
    from benchflow.cli import main as cli_main
    from benchflow.sandbox.daytona import reap_leaked_snapshots

    d = cli_main._daytona_client_or_exit()

    def _show(snap, will_delete):
        verdict = "[red](delete)[/red]" if will_delete else "[green](skip)[/green]"
        if dry_run or not will_delete:
            console.print(
                f"  [dim]{getattr(snap, 'name', '?')}[/dim] "
                f"state={getattr(snap, 'state', '?')} {verdict}"
            )

    counts = reap_leaked_snapshots(d, dry_run=dry_run, on_decision=_show)
    if dry_run:
        console.print(
            f"\n[bold]{counts['found']} snapshots found, "
            f"{counts['deleted']} benchflow-owned[/bold] "
            "(use without --dry-run to delete)"
        )
    else:
        console.print(
            f"\n[bold green]{counts['deleted']} snapshots deleted[/bold green] "
            f"({counts['skipped']} skipped, not benchflow-owned)"
        )


def _environment_adapter_provider_mode(sandbox: str) -> str | None:
    if sandbox != "cua":
        return sandbox
    return (
        "local"
        if os.environ.get("BENCHFLOW_CUA_LOCAL") in {"1", "true", "yes"}
        else "cloud"
    )


def _provider_runtime_probe_ready(provider: object) -> bool:
    if not isinstance(provider, Mapping):
        return False
    typed = cast("Mapping[str, object]", provider)
    runtime_probe = typed.get("runtime_probe")
    if not isinstance(runtime_probe, Mapping):
        return False
    probe = cast("Mapping[str, object]", runtime_probe)
    return probe.get("status") == "ready"


def _check_task_runtime_or_exit(
    task_dir: Path,
    *,
    sandbox: str,
    output_json: bool = False,
) -> str | None:
    from benchflow._utils.task_authoring import check_task
    from benchflow.cli._inbound_task_target import native_task_target

    try:
        with native_task_target(task_dir) as target:
            issues = check_task(
                target.path,
                sandbox_type=sandbox,
                validation_level="runtime-capability",
            )
            adapter_source = target.adapter_source
    except UnsupportedInboundTaskError as e:
        unsupported_adapter_task_or_exit(task_dir, e, output_json=output_json)

    if issues:
        if output_json:
            typer.echo(
                json.dumps(
                    {
                        "status": "invalid",
                        "task": str(task_dir),
                        "task_name": task_dir.name,
                        "adapter": adapter_source,
                        "validation_level": "runtime-capability",
                        "sandbox": sandbox,
                        "issues": issues,
                    }
                )
            )
            raise typer.Exit(1)
        label = task_dir.name
        if adapter_source:
            label = f"{label} ({adapter_source})"
        console.print(f"[red]✗[/red] {label} — {len(issues)} issue(s):")
        for issue in issues:
            console.print(f"  [yellow]→[/yellow] {escape(issue)}")
        raise typer.Exit(1)
    return adapter_source


def _check_provider_or_exit(
    sandbox: str,
    *,
    quiet: bool = False,
    runtime_probe: bool = False,
    output_json: bool = False,
) -> dict[str, object]:
    if sandbox == "docker":
        if shutil.which("docker") is None:
            console.print("[red]docker executable not found[/red]")
            raise typer.Exit(1)
        try:
            result = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired:
            console.print("[red]docker did not respond within 10s[/red]")
            raise typer.Exit(1) from None
        if result.returncode != 0:
            console.print(f"[red]docker is not ready: {result.stdout.strip()}[/red]")
            raise typer.Exit(1)
        version = result.stdout.strip()
        if not quiet:
            console.print(f"[green]✓[/green] docker ready ({version})")
        return {"provider": "docker", "status": "ready", "server_version": version}

    if sandbox == "cua":
        from benchflow.sandbox.cua import CuaSandbox

        try:
            CuaSandbox.preflight()
        except SystemExit as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
        except Exception as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from None
        payload: dict[str, object] = {
            "provider": "cua",
            "status": "ready",
            "sdk_import": True,
            "local": os.environ.get("BENCHFLOW_CUA_LOCAL") in {"1", "true", "yes"},
            "api_key_set": bool(os.environ.get("CUA_API_KEY")),
        }
        if not quiet:
            console.print("[green]✓[/green] Cua SDK import ok")
        if os.environ.get("BENCHFLOW_CUA_LOCAL") in {"1", "true", "yes"}:
            if not quiet:
                console.print("[green]✓[/green] Cua local mode selected")
        elif os.environ.get("CUA_API_KEY"):
            if not quiet:
                console.print("[green]✓[/green] CUA_API_KEY is set")
        else:
            payload["warning"] = (
                "CUA_API_KEY is not set; cloud create may rely on SDK auth state or fail"
            )
            if not quiet:
                console.print(
                    "[yellow]CUA_API_KEY is not set; cloud create may rely on SDK auth state or fail[/yellow]"
                )
        if runtime_probe:
            try:
                runtime = _probe_cua_runtime()
            except Exception as exc:
                _provider_error_or_exit(
                    "cua",
                    f"Cua runtime probe failed: {_short_error(exc)}",
                    output_json=output_json,
                    payload={**payload, "runtime_probe": {"status": "error"}},
                )
            payload["runtime_probe"] = runtime
            if runtime.get("status") != "ready":
                _provider_error_or_exit(
                    "cua",
                    "Cua runtime probe did not become ready",
                    output_json=output_json,
                    payload=payload,
                )
            if not quiet:
                console.print("[green]✓[/green] Cua runtime probe ready")
        return payload

    if sandbox == "daytona":
        from benchflow.cli import main as cli_main

        cli_main._daytona_client_or_exit()
        if not quiet:
            console.print("[green]✓[/green] Daytona client ready")
        return {"provider": "daytona", "status": "ready"}

    if sandbox == "modal":
        try:
            import modal  # noqa: F401
        except ModuleNotFoundError:
            console.print(
                "[red]modal SDK not installed[/red]\n"
                "Install it with [cyan]uv sync --extra sandbox-modal[/cyan]."
            )
            raise typer.Exit(1) from None
        if not quiet:
            console.print("[green]✓[/green] Modal SDK import ok")
        return {"provider": "modal", "status": "ready", "sdk_import": True}

    raise typer.Exit(1)


def _provider_error_or_exit(
    provider: str,
    reason: str,
    *,
    output_json: bool,
    payload: dict[str, object] | None = None,
) -> None:
    if output_json:
        typer.echo(
            json.dumps(
                {
                    "status": "error",
                    "provider": provider,
                    "reason": reason,
                    "provider_check": payload,
                }
            )
        )
    else:
        console.print(f"[red]{reason}[/red]")
    raise typer.Exit(1)


def _probe_cua_runtime() -> dict[str, object]:
    """Run a bounded Cua runtime smoke and return non-sensitive diagnostics."""
    with tempfile.TemporaryDirectory(prefix="benchflow-cua-probe-") as tmp:
        root = Path(tmp)
        env_dir = root / "environment"
        env_dir.mkdir()
        payload, background_errors = _run_async_with_background_error_capture(
            _probe_cua_runtime_async(env_dir)
        )
        if background_errors:
            payload["background_errors"] = background_errors
        failure_class = _classify_cua_probe_failure(payload)
        if failure_class:
            payload["failure_class"] = failure_class
        return payload


def _run_async_with_background_error_capture(
    coro: Coroutine[object, object, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Run a probe coroutine and keep loop-level task errors out of stderr.

    Cua's cloud transport may leave a readiness task whose exception would
    otherwise be printed as "Task exception was never retrieved" after cleanup.
    Capture that as structured diagnostics so `--json` remains machine-readable.
    """
    background_errors: list[dict[str, object]] = []
    loop = asyncio.new_event_loop()

    def _capture(_loop, context: dict[str, object]) -> None:
        message = str(context.get("message") or "background task error")
        exc = context.get("exception")
        item: dict[str, object] = {"message": message[:200]}
        if isinstance(exc, BaseException):
            item["error_type"] = type(exc).__name__
            item["reason"] = _short_error(exc)
        background_errors.append(item)

    loop.set_exception_handler(_capture)
    try:
        asyncio.set_event_loop(loop)
        payload = loop.run_until_complete(coro)
        # Give task finalizers a chance to report exceptions through the custom
        # handler before the loop closes.
        loop.run_until_complete(asyncio.sleep(0))
        return payload, background_errors
    finally:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        asyncio.set_event_loop(None)
        loop.close()


async def _probe_cua_runtime_async(env_dir: Path) -> dict[str, object]:
    from benchflow.sandbox.cua import CuaSandbox
    from benchflow.task.config import SandboxConfig

    config = SandboxConfig()
    sandbox = CuaSandbox(
        environment_dir=env_dir,
        environment_name="cua-runtime-probe",
        session_id="probe",
        rollout_paths=None,
        task_env_config=config,
    )
    checks: dict[str, object] = {}
    cleanup: dict[str, object] = {"attempted": False, "ok": False}
    sandbox_id: str | None = None
    start_error: str | None = None
    start_error_type: str | None = None
    try:
        await sandbox.start()
        sandbox_id = sandbox.sandbox_id
        checks["shell"] = await _probe_cua_shell(sandbox)
        checks["file_transfer"] = await _probe_cua_file_transfer(
            sandbox, env_dir.parent
        )
        checks["dimensions"] = await _probe_cua_dimensions(sandbox)
        checks["screenshot"] = await _probe_cua_screenshot(sandbox)
        checks["display_url"] = await _probe_cua_display_url(sandbox)
    except Exception as exc:
        sandbox_id = sandbox.sandbox_id
        start_error = _short_error(exc)
        start_error_type = type(exc).__name__
    finally:
        cleanup["attempted"] = True
        try:
            await sandbox.stop(delete=True)
            cleanup["ok"] = True
        except Exception as exc:  # pragma: no cover - defensive cleanup evidence
            cleanup["ok"] = False
            cleanup["reason"] = _short_error(exc)
    required_capabilities = ("shell", "file_transfer", "dimensions", "screenshot")
    failed_capabilities = [
        name for name in required_capabilities if not _probe_check_ok(checks.get(name))
    ]
    if start_error is not None:
        failed_capabilities.insert(0, "startup")
    required_ok = not failed_capabilities
    cleanup_ok = cleanup.get("ok") is True
    payload: dict[str, object] = {
        "status": "ready"
        if start_error is None and required_ok and cleanup_ok
        else "not-ready",
        "local": os.environ.get("BENCHFLOW_CUA_LOCAL") in {"1", "true", "yes"},
        "sdk": _cua_probe_sdk_metadata(),
        "request": _cua_probe_request_metadata(config),
        "sandbox_id": sandbox_id,
        "required_capabilities": list(required_capabilities),
        "failed_capabilities": failed_capabilities,
        "checks": checks,
        "cleanup": cleanup,
    }
    if start_error is not None:
        payload["reason"] = start_error
        payload["error_type"] = start_error_type or "Exception"
    return payload


def _cua_probe_sdk_metadata() -> dict[str, object]:
    try:
        import cua_sandbox

        import_name = "cua_sandbox"
        module_file = str(getattr(cua_sandbox, "__file__", "") or "")
        Sandbox = cua_sandbox.Sandbox
    except Exception:
        try:
            import cua

            import_name = "cua"
            module_file = str(getattr(cua, "__file__", "") or "")
            Sandbox = cua.Sandbox
        except Exception as exc:  # pragma: no cover - provider check covers import
            return {"available": False, "reason": _short_error(exc)}
    version = None
    package = "cua-sandbox" if import_name == "cua_sandbox" else "cua"
    try:
        version = importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        version = None
    return {
        "available": True,
        "package": package,
        "import_name": import_name,
        "version": version,
        "module_file": module_file,
        "supports_ephemeral": hasattr(Sandbox, "ephemeral"),
    }


def _cua_probe_request_metadata(config) -> dict[str, object]:
    local = os.environ.get("BENCHFLOW_CUA_LOCAL") in {"1", "true", "yes"}
    linux_kind = os.environ.get("BENCHFLOW_CUA_LINUX_KIND", "vm")
    payload: dict[str, object] = {
        "local": local,
        "region": os.environ.get("BENCHFLOW_CUA_REGION", "us-east-1"),
        "linux_kind": linux_kind,
        "linux_distro": os.environ.get("BENCHFLOW_CUA_LINUX_DISTRO", "ubuntu"),
        "linux_version": os.environ.get("BENCHFLOW_CUA_LINUX_VERSION", "24.04"),
        "named_create": local
        or os.environ.get("BENCHFLOW_CUA_NAMED_CREATE", "").lower()
        in {"1", "true", "yes", "on"},
        "time_to_start_sec": os.environ.get("BENCHFLOW_CUA_TIME_TO_START_SEC"),
        "request_timeout_sec": os.environ.get("BENCHFLOW_CUA_REQUEST_TIMEOUT_SEC"),
        "cpu": config.cpus,
        "memory_mb": config.memory_mb,
        "disk_gb": config.storage_mb // 1024 if config.storage_mb else None,
    }
    if config.docker_image:
        payload["registry_image"] = config.docker_image
    return payload


def _classify_cua_probe_failure(payload: Mapping[str, object]) -> str | None:
    if payload.get("status") == "ready":
        return None
    text_parts: list[str] = []
    reason = payload.get("reason")
    if isinstance(reason, str):
        text_parts.append(reason)
    background_errors = payload.get("background_errors")
    if isinstance(background_errors, list):
        for item in background_errors:
            if isinstance(item, dict):
                typed_item = cast("dict[str, object]", item)
                for key in ("reason", "message"):
                    value = typed_item.get(key)
                    if isinstance(value, str):
                        text_parts.append(value)
    text = "\n".join(text_parts).lower()
    if "/cmd" in text and "404" in text:
        return "cloud-computer-server-cmd-404"
    if "did not become running" in text:
        return "cloud-vm-running-timeout"
    if "not reachable" in text or "timed out" in text:
        return "cloud-runtime-readiness-timeout"
    if payload.get("failed_capabilities"):
        return "runtime-capability-failure"
    return None


async def _probe_cua_shell(sandbox) -> dict[str, object]:
    result = await sandbox.exec("printf benchflow-cua-probe", timeout_sec=30)
    stdout = (result.stdout or "").strip()
    return {
        "ok": result.return_code == 0 and stdout == "benchflow-cua-probe",
        "return_code": result.return_code,
        "stdout_preview": stdout[:80],
    }


async def _probe_cua_dimensions(sandbox) -> dict[str, object]:
    try:
        width, height = await sandbox.get_dimensions()
    except Exception as exc:
        return {"ok": False, "reason": _short_error(exc)}
    return {"ok": width > 0 and height > 0, "width": width, "height": height}


async def _probe_cua_file_transfer(sandbox, root: Path) -> dict[str, object]:
    source = root / "cua-probe-upload.txt"
    dest = root / "cua-probe-download.txt"
    expected = b"benchflow-cua-file-transfer"
    source.write_bytes(expected)
    target = "/tmp/benchflow-cua-file-transfer.txt"
    try:
        await sandbox.upload_file(source, target)
        await sandbox.download_file(target, dest)
    except Exception as exc:
        return {"ok": False, "reason": _short_error(exc)}
    actual = dest.read_bytes() if dest.exists() else b""
    return {"ok": actual == expected, "bytes": len(actual)}


async def _probe_cua_screenshot(sandbox) -> dict[str, object]:
    try:
        data = await sandbox.screenshot(format="png", quality=50)
    except Exception as exc:
        return {"ok": False, "reason": _short_error(exc)}
    return {"ok": bool(data), "bytes": len(data)}


async def _probe_cua_display_url(sandbox) -> dict[str, object]:
    try:
        url = await sandbox.get_display_url(share=False)
    except Exception as exc:
        return {"ok": False, "reason": _short_error(exc)}
    scheme = url.split(":", 1)[0] if ":" in url else ""
    return {
        "ok": bool(url),
        "available": bool(url),
        "scheme": scheme,
        "length": len(url),
    }


def _short_error(exc: BaseException) -> str:
    text = str(exc) or type(exc).__name__
    return text[:500]


def _probe_check_ok(item: object) -> bool:
    if not isinstance(item, Mapping):
        return False
    typed = cast("Mapping[str, object]", item)
    return typed.get("ok") is True


def _list_docker_environments(*, output_json: bool) -> None:
    containers = _docker_owned_resources(
        "container",
        [
            "docker",
            "container",
            "ls",
            "-a",
            "--filter",
            f"label={_DOCKER_OWNED_LABEL}",
        ],
    )
    networks = _docker_owned_resources(
        "network",
        [
            "docker",
            "network",
            "ls",
            "--filter",
            f"label={_DOCKER_OWNED_LABEL}",
        ],
    )
    images = _docker_owned_resources(
        "image",
        [
            "docker",
            "image",
            "ls",
            "--filter",
            f"label={_DOCKER_OWNED_LABEL}",
        ],
    )
    payload = {
        "provider": "docker",
        "ownership_label": _DOCKER_OWNED_LABEL,
        "containers": containers,
        "networks": networks,
        "images": images,
    }
    if output_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Docker Environments")
    table.add_column("Type")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Status", style="green")
    table.add_column("Project")
    resources = [*containers, *networks, *images]
    for item in resources:
        table.add_row(
            item["type"],
            item["id"][:12],
            item["name"],
            item.get("status", ""),
            item.get("project", ""),
        )
    console.print(table)
    console.print(f"\n[bold]{len(resources)} resource(s)[/bold]")


def _cleanup_docker_environments(
    *,
    dry_run: bool,
    max_age_minutes: int,
    output_json: bool = False,
) -> None:
    now = datetime.now(UTC)
    containers = _docker_owned_resources(
        "container",
        [
            "docker",
            "container",
            "ls",
            "-a",
            "--filter",
            f"label={_DOCKER_OWNED_LABEL}",
        ],
    )
    networks = _docker_owned_resources(
        "network",
        [
            "docker",
            "network",
            "ls",
            "--filter",
            f"label={_DOCKER_OWNED_LABEL}",
        ],
    )
    # Snapshot images (``bf-snap-*`` from ``docker commit``) carry the same
    # ownership label; without this pass they leak forever (tagged images are
    # never reaped by ``image prune``, which only touches dangling layers).
    images = _docker_owned_resources(
        "image",
        [
            "docker",
            "image",
            "ls",
            "--filter",
            f"label={_DOCKER_OWNED_LABEL}",
        ],
    )
    resources = [*containers, *networks, *images]
    candidates: list[dict[str, object]] = []
    skipped = 0
    for item in resources:
        age = _docker_age_minutes(item.get("created_at"), now=now)
        if age is None or age < max_age_minutes:
            skipped += 1
            continue
        candidates.append({**item, "age_minutes": round(age, 1)})

    deleted: list[dict[str, str]] = []
    container_ids = [
        str(item["id"]) for item in candidates if item["type"] == "container"
    ]
    network_ids = [str(item["id"]) for item in candidates if item["type"] == "network"]
    image_ids = [str(item["id"]) for item in candidates if item["type"] == "image"]
    if not dry_run:
        if container_ids:
            _run_docker_delete(["docker", "rm", "-f", *container_ids])
            deleted.extend(
                {
                    "type": "container",
                    "id": str(item["id"]),
                    "name": str(item.get("name") or ""),
                }
                for item in candidates
                if item["type"] == "container"
            )
        if network_ids:
            _run_docker_delete(["docker", "network", "rm", *network_ids])
            deleted.extend(
                {
                    "type": "network",
                    "id": str(item["id"]),
                    "name": str(item.get("name") or ""),
                }
                for item in candidates
                if item["type"] == "network"
            )
        if image_ids:
            _run_docker_delete(["docker", "image", "rm", "-f", *image_ids])
            deleted.extend(
                {
                    "type": "image",
                    "id": str(item["id"]),
                    "name": str(item.get("name") or ""),
                }
                for item in candidates
                if item["type"] == "image"
            )

    if output_json:
        typer.echo(
            json.dumps(
                {
                    "provider": "docker",
                    "status": "dry-run" if dry_run else "deleted",
                    "dry_run": dry_run,
                    "ownership_label": _DOCKER_OWNED_LABEL,
                    "max_age_minutes": max_age_minutes,
                    "found": len(resources),
                    "matched": len(candidates),
                    "skipped": skipped,
                    "deleted": deleted,
                    "candidates": [
                        {**item, "would_delete": dry_run} for item in candidates
                    ],
                },
                indent=2,
            )
        )
        return

    for item in candidates:
        verdict = "[red](delete)[/red]" if not dry_run else "[yellow](dry-run)[/yellow]"
        age = _coerce_age_for_display(item.get("age_minutes"))
        console.print(
            f"  [dim]{item['type']}:{str(item['id'])[:12]}[/dim] "
            f"name={item.get('name', '')} age={age:.0f}m {verdict}"
        )
    if dry_run:
        console.print(
            f"\n[bold]{len(resources)} Docker resource(s) found, {len(candidates)} "
            f"owned resources older than {max_age_minutes}m[/bold] "
            "(use without --dry-run to delete)"
        )
    else:
        console.print(
            f"\n[bold green]{len(deleted)} Docker resource(s) deleted[/bold green] "
            f"({skipped} skipped)"
        )


def _docker_owned_resources(kind: str, base_cmd: list[str]) -> list[dict[str, str]]:
    if shutil.which("docker") is None:
        console.print("[red]docker executable not found[/red]")
        raise typer.Exit(1)
    result = subprocess.run(
        [*base_cmd, "--format", "{{json .}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        console.print(f"[red]docker {kind} list failed: {result.stderr.strip()}[/red]")
        raise typer.Exit(1)
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            console.print(f"[red]docker {kind} list emitted invalid JSON: {exc}[/red]")
            raise typer.Exit(1) from exc
        rows.append(_docker_resource_info(kind, raw))
    return rows


def _docker_resource_info(kind: str, raw: Mapping[str, object]) -> dict[str, str]:
    # ``docker image ls --format {{json .}}`` has no ``Names``/``Name`` field —
    # the human-facing name is ``Repository:Tag`` (e.g. ``bf-snap-foo:latest``).
    if kind == "image":
        repository = str(raw.get("Repository") or "")
        tag = str(raw.get("Tag") or "")
        name = f"{repository}:{tag}" if repository else tag
    else:
        name = str(raw.get("Names") or raw.get("Name") or "")
    resource_id = str(raw.get("ID") or "")
    labels = str(raw.get("Labels") or "")
    project = _docker_label_value(labels, "com.docker.compose.project")
    status = str(raw.get("Status") or raw.get("Driver") or raw.get("Size") or "")
    return {
        "type": kind,
        "id": resource_id,
        "name": name,
        "status": status,
        "project": project,
        "created_at": str(raw.get("CreatedAt") or raw.get("Created") or ""),
    }


def _docker_label_value(labels: str, key: str) -> str:
    for part in labels.split(","):
        if "=" not in part:
            continue
        label_key, label_value = part.split("=", 1)
        if label_key.strip() == key:
            return label_value.strip()
    return ""


def _run_docker_delete(cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        console.print(f"[red]{' '.join(cmd[:3])} failed: {result.stderr.strip()}[/red]")
        raise typer.Exit(1)


def _docker_age_minutes(value: object, *, now: datetime) -> float | None:
    if not value:
        return None
    text = str(value)
    created = _parse_docker_datetime(text)
    if created is None:
        return None
    return (now - created).total_seconds() / 60


def _coerce_age_for_display(value: object) -> float:
    return (
        value
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )


def _parse_docker_datetime(text: str) -> datetime | None:
    try:
        value = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    except ValueError:
        pass
    parts = text.split()
    if len(parts) >= 3:
        trimmed = " ".join(parts[:3])
        if "." in trimmed:
            before_fraction, after_fraction = trimmed.split(".", 1)
            fraction, rest = after_fraction.split(" ", 1)
            trimmed = f"{before_fraction}.{fraction[:6]} {rest}"
        for fmt in ("%Y-%m-%d %H:%M:%S %z", "%Y-%m-%d %H:%M:%S.%f %z"):
            try:
                return datetime.strptime(trimmed, fmt)
            except ValueError:
                continue
    return None


def _list_cua_environments(*, output_json: bool) -> None:
    from benchflow.sandbox.cua import list_cua_sandboxes

    try:
        sandboxes = asyncio.run(list_cua_sandboxes())
    except Exception as exc:
        console.print(f"[red]Cua list failed: {exc}[/red]")
        raise typer.Exit(1) from None

    rows = [_cua_info_dict(sb) for sb in sandboxes]
    if output_json:
        typer.echo(json.dumps(rows, indent=2))
        return

    table = Table(title="Cua Environments")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Source")
    table.add_column("OS")
    table.add_column("Created", style="dim")
    for item in rows:
        table.add_row(
            item.get("name", ""),
            item.get("status", ""),
            item.get("source", ""),
            item.get("os_type", ""),
            item.get("created_at", ""),
        )
    console.print(table)
    console.print(f"\n[bold]{len(rows)} environment(s)[/bold]")


def _cleanup_cua_environments(
    *,
    dry_run: bool,
    max_age_minutes: int,
    output_json: bool = False,
) -> None:
    from datetime import datetime

    from benchflow.sandbox.cua import delete_cua_sandbox, list_cua_sandboxes

    prefix = os.environ.get("BENCHFLOW_CUA_CLEANUP_PREFIX", "benchflow-")
    now = datetime.now(UTC)
    try:
        sandboxes = asyncio.run(list_cua_sandboxes())
    except Exception as exc:
        console.print(f"[red]Cua cleanup list failed: {exc}[/red]")
        raise typer.Exit(1) from None

    candidates: list[tuple[dict[str, str], float]] = []
    skipped = 0
    for raw in sandboxes:
        item = _cua_info_dict(raw)
        name = item.get("name", "")
        age = _age_minutes(item.get("created_at"), now=now)
        if not name.startswith(prefix) or age is None or age < max_age_minutes:
            skipped += 1
            continue
        candidates.append((item, age))

    deleted: list[str] = []
    for item, age in candidates:
        if not output_json:
            console.print(
                f"  [dim]{item['name']}[/dim] status={item.get('status', '')} "
                f"age={age:.0f}m [red](delete)[/red]"
            )
        if not dry_run:
            try:
                asyncio.run(delete_cua_sandbox(item["name"]))
                deleted.append(item["name"])
            except Exception as exc:
                console.print(f"[red]Cua delete failed for {item['name']}: {exc}[/red]")
                raise typer.Exit(1) from None

    if output_json:
        typer.echo(
            json.dumps(
                {
                    "provider": "cua",
                    "status": "dry-run" if dry_run else "deleted",
                    "dry_run": dry_run,
                    "cleanup_prefix": prefix,
                    "max_age_minutes": max_age_minutes,
                    "found": len(sandboxes),
                    "matched": len(candidates),
                    "skipped": skipped,
                    "deleted": deleted,
                    "candidates": [
                        {
                            **item,
                            "age_minutes": round(age, 1),
                            "would_delete": dry_run,
                        }
                        for item, age in candidates
                    ],
                }
            )
        )
        return

    if dry_run:
        console.print(
            f"\n[bold]{len(sandboxes)} environment(s) found, {len(candidates)} "
            f"matching prefix {prefix!r} older than {max_age_minutes}m[/bold] "
            "(use without --dry-run to delete)"
        )
    else:
        console.print(
            f"\n[bold green]{len(candidates)} Cua environment(s) deleted[/bold green] "
            f"({skipped} skipped)"
        )


def _cua_info_dict(sb: object) -> dict[str, str]:
    model_dump = getattr(sb, "model_dump", None)
    if callable(model_dump):
        raw = model_dump()
    elif hasattr(sb, "__dict__"):
        raw = vars(sb)
    else:
        raw = {}
    return {
        key: str(raw.get(key) or "")
        for key in (
            "name",
            "status",
            "source",
            "os_type",
            "host",
            "vnc_url",
            "api_url",
            "created_at",
        )
    }


def _age_minutes(created_at: str | None, *, now) -> float | None:
    if not created_at:
        return None
    try:
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - created).total_seconds() / 60
