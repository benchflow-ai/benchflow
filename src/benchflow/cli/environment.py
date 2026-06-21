"""``bench environment`` — DEPRECATED alias group (removed in 0.7).

The local sandbox lifecycle moved to ``bench sandbox`` (create / list / cleanup;
see :mod:`benchflow.cli.sandbox`) and hosted-provider browsing to ``bench hub
env`` (see :mod:`benchflow.cli._hosted_env`). Every command here is a hidden
deprecated alias that emits a one-line stderr notice and delegates to the new
home, so existing scripts keep working through 0.6.

Registered onto the top-level app by :func:`register_environment`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from benchflow.cli._hosted_env import (
    hosted_env_inspect,
    hosted_env_list,
    hosted_env_show,
)
from benchflow.cli._options import SandboxOption
from benchflow.cli._shared import warn_deprecated
from benchflow.cli.sandbox import sandbox_cleanup, sandbox_create, sandbox_list_local


def register_environment(app: typer.Typer) -> None:
    """Attach the deprecated ``environment`` alias group (hidden from help).

    Each command uses ``hidden=True`` only — NOT Typer's ``deprecated=True``.
    ``deprecated=True`` would (a) print its own generic ``DeprecationWarning: The
    command 'X' is deprecated.`` line that omits the canonical replacement and
    re-fires every invocation, doubling up with our ``warn_deprecated`` one-liner,
    and (b) surface the aliased verbs in ``environment --help``. ``hidden=True``
    alone matches the ``adopt`` / ``agent`` alias families: exactly one
    once-per-process stderr notice, verbs hidden from help.
    """
    env_app = typer.Typer(help="Deprecated; use `bench sandbox` / `bench hub`.")
    app.add_typer(env_app, name="environment", hidden=True)

    @env_app.command("create", hidden=True)
    def environment_create(
        task_dir: Annotated[
            Path,
            typer.Argument(
                help="Task directory with task.md or task.toml + Dockerfile"
            ),
        ],
        sandbox: SandboxOption = "daytona",
    ) -> None:
        """Deprecated; use `bench sandbox create`."""
        warn_deprecated("bench environment create", "bench sandbox create")
        sandbox_create(task_dir, sandbox)

    @env_app.command("list", hidden=True)
    def environment_list(
        provider: Annotated[
            str | None,
            typer.Option(
                "--provider", hidden=True, help="Deprecated; use `bench hub list`"
            ),
        ] = None,
        hub: Annotated[
            str | None,
            typer.Option("--hub", hidden=True, help="Deprecated; use `bench hub list`"),
        ] = None,
        owner: Annotated[
            str | None,
            typer.Option("--owner", hidden=True, help="Hosted provider owner filter"),
        ] = None,
        search: Annotated[
            str | None,
            typer.Option("--search", hidden=True, help="Hosted provider search query"),
        ] = None,
        limit: Annotated[
            int | None,
            typer.Option("--limit", hidden=True, help="Maximum hosted results"),
        ] = None,
        output_json: Annotated[
            bool,
            typer.Option(
                "--json", hidden=True, help="Emit raw JSON for hosted results"
            ),
        ] = False,
    ) -> None:
        """Deprecated; use `bench sandbox list` (local) or `bench hub list` (hosted)."""
        provider = provider or hub
        if provider:
            warn_deprecated(
                "bench environment list --provider", "bench hub list --provider"
            )
            hosted_env_list(
                provider=provider,
                owner=owner,
                search=search,
                limit=limit,
                output_json=output_json,
            )
            return
        warn_deprecated("bench environment list", "bench sandbox list")
        sandbox_list_local()

    @env_app.command("show", hidden=True)
    def environment_show(
        source_env: Annotated[
            str,
            typer.Argument(
                help="Hosted environment (e.g. primeintellect/general-agent)"
            ),
        ],
        version: Annotated[
            str | None, typer.Option("--version", help="Hosted environment version")
        ] = None,
    ) -> None:
        """Deprecated; use `bench hub show`."""
        warn_deprecated("bench environment show", "bench hub show")
        hosted_env_show(source_env=source_env, version=version)

    @env_app.command("inspect", hidden=True)
    def environment_inspect(
        source_env: Annotated[
            str,
            typer.Argument(
                help="Hosted environment (e.g. primeintellect/general-agent)"
            ),
        ],
        version: Annotated[
            str | None, typer.Option("--version", help="Hosted environment version")
        ] = None,
        path: Annotated[
            str,
            typer.Option("--path", help="File inside the hosted environment package"),
        ] = "README.md",
    ) -> None:
        """Deprecated; use `bench hub inspect`."""
        warn_deprecated("bench environment inspect", "bench hub inspect")
        hosted_env_inspect(source_env=source_env, version=version, path=path)

    @env_app.command("cleanup", hidden=True)
    def environment_cleanup(
        dry_run: Annotated[
            bool, typer.Option("--dry-run", help="List sandboxes without deleting")
        ] = False,
        max_age_minutes: Annotated[
            int, typer.Option("--max-age", help="Delete sandboxes older than N minutes")
        ] = 1440,
    ) -> None:
        """Deprecated; use `bench sandbox cleanup`."""
        warn_deprecated("bench environment cleanup", "bench sandbox cleanup")
        sandbox_cleanup(dry_run=dry_run, max_age_minutes=max_age_minutes)
