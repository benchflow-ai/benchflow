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
from benchflow.cli._shared import console, exit_if_missing, print_error


def register_hub(app: typer.Typer) -> None:
    """Attach the ``hub`` command group to the top-level benchflow app."""
    hub_app = typer.Typer(
        help=(
            "Environment hubs: browse a hub's environments (list/show/inspect), "
            "check registry tasks (check), and publish datasets and push runs "
            "to a BenchFlow Hub (publish/push)."
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

    @hub_app.command("publish")
    def hub_publish(
        tasks_dir: Annotated[
            Path, typer.Argument(help="Folder holding one folder per task")
        ],
        name: Annotated[
            str,
            typer.Option(
                "--name",
                help="The version to publish, as org/name@version (e.g. robouse/metaworld-sample@0.6.1)",
            ),
        ],
        hub: Annotated[
            str | None,
            typer.Option("--hub", help="Hub address (default: $BENCHFLOW_HUB_URL)"),
        ] = None,
        description: Annotated[
            str | None, typer.Option("--description", help="One line on what it is")
        ] = None,
        license: Annotated[
            str | None,
            typer.Option(
                "--license",
                help="The tasks' license, as an SPDX expression (e.g. Apache-2.0 AND MIT)",
            ),
        ] = None,
        kind: Annotated[
            str | None,
            typer.Option("--kind", help="What its tasks are, e.g. embodied"),
        ] = None,
        source_repo: Annotated[
            str | None,
            typer.Option(
                "--source-repo", help="Where the tasks came from (a repository URL)"
            ),
        ] = None,
        source_commit: Annotated[
            str | None,
            typer.Option("--source-commit", help="The commit they came from"),
        ] = None,
        exporter: Annotated[
            str | None,
            typer.Option(
                "--exporter",
                help="What wrote the packages, e.g. 'robouse-export@d05027fb' (the tool and its commit)",
            ),
        ] = None,
    ) -> None:
        """Publish a folder of tasks as an immutable dataset version on a BenchFlow Hub."""
        from benchflow.cli import _hub_client

        exit_if_missing(
            tasks_dir, next_step="pass a folder that holds one folder per task"
        )
        _hub_client.publish(
            tasks_dir,
            name,
            hub,
            description=description,
            license=license,
            kind=kind,
            source_repo=source_repo,
            source_commit=source_commit,
            exporter=exporter,
        )

    @hub_app.command("push")
    def hub_push(
        job_dir: Annotated[
            Path,
            typer.Argument(
                help="A job folder (jobs/<job>) holding one finished run, or the run (jobs/<job>/<time>)"
            ),
        ],
        hub: Annotated[
            str | None,
            typer.Option("--hub", help="Hub address (default: $BENCHFLOW_HUB_URL)"),
        ] = None,
        dataset: Annotated[
            str | None,
            typer.Option(
                "--dataset",
                help="The dataset version the job ran, org/name@version (default: what the job recorded)",
            ),
        ] = None,
        label: Annotated[
            str | None, typer.Option("--label", help="A label for the run on the Hub")
        ] = None,
    ) -> None:
        """Push a finished run (traces, results, an embodied trial's record and video) to a BenchFlow Hub."""
        from benchflow.cli import _hub_client

        exit_if_missing(
            job_dir,
            next_step="pass a job folder under ./jobs (or --jobs-dir) that bench run wrote",
        )
        _hub_client.push(job_dir, hub, dataset=dataset, label=label)

    @hub_app.command("check")
    def hub_check(
        registry: Annotated[
            str | None,
            typer.Option(
                "--registry",
                help="Registry JSON URL or local file (default: BenchFlow's registry).",
            ),
        ] = None,
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
        ] = Path(".cache/hub/registry"),
        limit: Annotated[
            int | None,
            typer.Option("--limit", help="Optional cap on selected task refs.", min=1),
        ] = None,
    ) -> None:
        """Inventory or structurally check representative registry tasks."""
        from benchflow.hub.registry import check_registry_tasks, records_summary

        shown = registry or "BenchFlow's registry"
        where = f"--registry {registry}" if registry else shown
        if registry and not registry.startswith(("http://", "https://")):
            exit_if_missing(
                Path(registry).expanduser(),
                next_step="pass an existing registry JSON file or URL to --registry",
            )
        try:
            records = check_registry_tasks(
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
            print_error(
                f"{where} is not valid JSON",
                next_step="pass a registry JSON file or URL to --registry",
            )
            raise typer.Exit(1) from None
        except OSError as exc:
            from urllib.error import HTTPError, URLError

            offline = isinstance(exc, URLError) and not isinstance(exc, HTTPError)
            print_error(
                f"{where}: {exc.strerror or exc}",
                next_step="check this machine's network (or its proxy), then run "
                "the command again"
                if offline
                else "pass a registry JSON file or URL to --registry",
            )
            raise typer.Exit(1) from None
        except Exception as exc:
            print_error(
                f"Registry check failed: {exc}",
                next_step="check --registry and this machine's network, then run "
                "the check again",
            )
            raise typer.Exit(1) from exc

        summary = records_summary(records)
        console.print(
            f"[bold]Registry check ({escape(shown)}):[/bold] "
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
                help="Hub to browse: 'benchflow' (BenchFlow's dataset registry) "
                "or 'primeintellect' (hosted envs; needs the `prime` CLI)",
            ),
        ] = "benchflow",
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
        """List a hub's datasets or environments (benchflow or primeintellect)."""
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
                help=(
                    "A dataset on a BenchFlow Hub as org/name@version (e.g. "
                    "robouse/metaworld-sample@0.6.1; read from --hub, or from "
                    "$BENCHFLOW_HUB_URL when no --provider is given), a dataset of "
                    "BenchFlow's registry as name@version (e.g. terminal-bench@2.0), "
                    "or a hosted environment (e.g. primeintellect/general-agent)"
                )
            ),
        ],
        version: Annotated[
            str | None,
            typer.Option(
                "--version", help="Version (of the dataset or hosted environment)"
            ),
        ] = None,
        provider: Annotated[
            str | None,
            typer.Option(
                "--provider",
                help="Where to read it without a Hub: 'benchflow' (BenchFlow's "
                "registry; lists the dataset's tasks) or 'primeintellect' (default: "
                "benchflow for name@version, primeintellect for owner/name)",
            ),
        ] = None,
        output_json: Annotated[
            bool,
            typer.Option(
                "--json",
                help="Print the dataset's entry as JSON (for a Hub version, the version and its controls)",
            ),
        ] = False,
        hub: Annotated[
            str | None,
            typer.Option(
                "--hub",
                help="A BenchFlow Hub to read org/name@version from, with its control "
                "results (default: $BENCHFLOW_HUB_URL when set)",
            ),
        ] = None,
    ) -> None:
        """Show a dataset's tasks, or a hosted environment's metadata."""
        import os

        from benchflow.hub.client import HUB_URL_ENV

        spec = (
            f"{source_env}@{version}"
            if version and "@" not in source_env
            else source_env
        )
        # As `bench run -d`: --hub reads that Hub; $BENCHFLOW_HUB_URL alone serves org/name datasets, never
        # BenchFlow's own registry names.
        name = spec.partition("@")[0]
        on_hub = hub or (
            provider is None and "/" in name and os.environ.get(HUB_URL_ENV)
        )
        if on_hub and "@" in spec:
            from benchflow.cli import _hub_client

            _hub_client.show(spec, hub, output_json=output_json)
            return
        hosted_env_show(
            source_env=source_env,
            version=version,
            provider=provider,
            output_json=output_json,
        )

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
