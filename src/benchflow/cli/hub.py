"""``bench hub`` — browse external environment hubs + check compatibility.

Registered onto the top-level app by :func:`register_hub`; ``cli/main.py`` only
wires the call. The browsing verbs (``list``/``show``/``inspect``) live directly
under ``hub`` — the old ``hub env`` nesting was redundant (the whole group is
about environment hubs), so ``hub env *`` is kept only as a hidden back-compat
alias.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._hosted_env import (
    hosted_env_inspect,
    hosted_env_list,
    hosted_env_show,
)
from benchflow.cli._shared import console, print_error
from benchflow.hub.harbor_registry import DEFAULT_HARBOR_REGISTRY_URL


def register_hub(app: typer.Typer) -> None:
    """Attach the ``hub`` command group to the top-level benchflow app."""
    hub_app = typer.Typer(
        help=(
            "External environment hubs: browse a hub's environments "
            "(list/show/inspect) and check Harbor compatibility (check)."
        )
    )
    app.add_typer(hub_app, name="hub", rich_help_panel="Environments")

    # Canonical browsing verbs live directly under `hub` (the `env` level was
    # redundant — `hub` already means environment hubs).
    _register_env_verbs(hub_app)

    # Back-compat: `bench hub env list|show|inspect` still resolves, as a hidden
    # alias of the flattened commands.
    env_alias = typer.Typer(
        help="Alias of `bench hub` — use `bench hub list/show/inspect`."
    )
    hub_app.add_typer(env_alias, name="env", hidden=True)
    _register_env_verbs(env_alias)

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
        # Surface a user-meaningful message instead of a raw stdlib repr: a bad
        # --registry otherwise leaks `Expecting value: line 1 column 1 (char 0)`
        # (JSONDecodeError) or `[Errno 2] ...` (OSError). print_error escapes the
        # path + routes to stderr.
        except json.JSONDecodeError:
            print_error(f"--registry {registry} is not valid JSON")
            raise typer.Exit(1) from None
        except OSError as exc:
            print_error(f"--registry {registry}: {exc.strerror or exc}")
            raise typer.Exit(1) from None
        except Exception as exc:
            print_error(f"Harbor compatibility check failed: {exc}")
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


def _register_env_verbs(target: typer.Typer) -> None:
    """Register the hub-environment browsing verbs (list/show/inspect) on ``target``.

    Called once for the canonical ``bench hub`` group and once for the hidden
    ``bench hub env`` back-compat alias, so both expose identical commands.
    """

    @target.command("list")
    def hub_list(
        provider: Annotated[
            str,
            typer.Option(
                "--provider",
                help="Hub to browse: 'primeintellect' (hosted envs) or "
                "'harbor' (benchmark registry)",
            ),
        ] = "primeintellect",
        owner: Annotated[
            str | None,
            typer.Option("--owner", help="Owner/namespace filter (primeintellect)"),
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
        """List a hub's environments (primeintellect or harbor)."""
        hosted_env_list(
            provider=provider,
            owner=owner,
            search=search,
            limit=limit,
            output_json=output_json,
        )

    @target.command("show")
    def hub_show(
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
        """Show a hub environment's metadata."""
        hosted_env_show(source_env=source_env, version=version)

    @target.command("inspect")
    def hub_inspect(
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
        """Inspect a file from a hub environment package."""
        hosted_env_inspect(source_env=source_env, version=version, path=path)
