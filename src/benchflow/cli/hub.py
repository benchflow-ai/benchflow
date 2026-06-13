"""``bench hub`` — compatibility checks for external environment hubs.

Registered onto the top-level app by :func:`register_hub`; ``cli/main.py``
only wires the call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._hosted_env import (
    hosted_env_inspect,
    hosted_env_list,
    hosted_env_show,
)
from benchflow.cli._shared import console
from benchflow.hub.harbor_registry import DEFAULT_HARBOR_REGISTRY_URL


def register_hub(app: typer.Typer) -> None:
    """Attach the ``hub`` command group to the top-level benchflow app."""
    hub_app = typer.Typer(help="Compatibility checks for external environment hubs.")
    app.add_typer(hub_app, name="hub", rich_help_panel="Environments")
    _register_hub_env(hub_app)

    @hub_app.command("check")
    def hub_check(
        registry: Annotated[
            str,
            typer.Option(
                "--registry",
                help="Harbor registry JSON URL or local file.",
            ),
        ] = DEFAULT_HARBOR_REGISTRY_URL,
        tasks_per_dataset: Annotated[
            int,
            typer.Option(
                "--tasks-per-dataset",
                help="Number of representative tasks to select per registry dataset.",
                min=1,
            ),
        ] = 2,
        level: Annotated[
            str,
            typer.Option(
                "--level",
                help="Compatibility level to run: inventory or check.",
            ),
        ] = "inventory",
        out: Annotated[
            Path | None,
            typer.Option("--out", help="Optional JSONL output path."),
        ] = None,
        cache_dir: Annotated[
            Path,
            typer.Option("--cache-dir", help="Cache directory for sparse clones."),
        ] = Path(".cache/hub/harbor"),
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Optional cap on selected task refs.", min=1),
        ] = None,
    ) -> None:
        """Inventory or structurally check representative Harbor registry tasks."""
        from benchflow.hub.harbor_registry import (
            check_harbor_registry,
            records_summary,
        )

        try:
            records = check_harbor_registry(
                registry,
                tasks_per_dataset=tasks_per_dataset,
                level=level,
                out=out,
                cache_dir=cache_dir,
                limit=limit,
            )
        except Exception as exc:
            # escape(): the message echoes the user-supplied --registry path,
            # which can contain Rich markup (`[`, `[/red]`) — an unescaped
            # interpolation makes the error handler itself raise MarkupError.
            console.print(
                f"[red]Harbor compatibility check failed:[/red] {escape(str(exc))}"
            )
            raise typer.Exit(1) from exc

        summary = records_summary(records)
        console.print(
            "[bold]Harbor compatibility:[/bold] "
            f"{summary['total']} task refs, "
            f"{summary['pass']} pass, {summary['fail']} fail, "
            f"{summary['blocked']} blocked"
        )
        if out is not None:
            console.print(f"[green]Wrote JSONL report:[/green] {escape(str(out))}")


def _register_hub_env(hub_app: typer.Typer) -> None:
    """Attach ``bench hub env`` — read-only browsing of a hosted provider's
    environments (PrimeIntellect "Environments"). The canonical home for what
    used to be ``bench environment list --provider`` / ``show`` / ``inspect``;
    the run path stays on ``bench eval create --source-env``."""
    env_app = typer.Typer(
        help="Browse hosted environments from an external provider (e.g. primeintellect)."
    )
    hub_app.add_typer(env_app, name="env")

    @env_app.command("list")
    def hub_env_list(
        provider: Annotated[
            str,
            typer.Option("--provider", help="Hosted environment provider"),
        ] = "primeintellect",
        owner: Annotated[
            str | None, typer.Option("--owner", help="Owner/namespace filter")
        ] = None,
        search: Annotated[
            str | None, typer.Option("--search", help="Search query")
        ] = None,
        limit: Annotated[
            int | None, typer.Option("--limit", help="Maximum results")
        ] = None,
        output_json: Annotated[
            bool, typer.Option("--json", help="Emit raw JSON")
        ] = False,
    ) -> None:
        """List a hosted provider's environments."""
        hosted_env_list(
            provider=provider,
            owner=owner,
            search=search,
            limit=limit,
            output_json=output_json,
        )

    @env_app.command("show")
    def hub_env_show(
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
        """Show hosted environment metadata."""
        hosted_env_show(source_env=source_env, version=version)

    @env_app.command("inspect")
    def hub_env_inspect(
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
        """Inspect a file from a hosted environment package."""
        hosted_env_inspect(source_env=source_env, version=version, path=path)
