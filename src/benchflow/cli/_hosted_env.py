"""Shared hosted-environment read commands.

Canonically reached via ``bench hub list|show|inspect`` (external
environment-hub browsing). ``list`` browses multiple hubs — PrimeIntellect
hosted "Environments" and BenchFlow's dataset registry — dispatched on
``--provider``. ``show`` lists a BenchFlow dataset's tasks or shows a
PrimeIntellect environment; ``inspect`` targets the PrimeIntellect CLI.

The deprecated ``bench environment list --provider`` / ``environment show`` /
``environment inspect`` aliases delegate here so there is exactly one copy of
the logic. The actual API/registry helpers live in ``benchflow.hosted_env`` and
``benchflow.hub.registry`` and are untouched.
"""

from __future__ import annotations

import json

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import console, print_error

#: Hubs that ``bench hub list`` can browse.
LIST_PROVIDERS = ("primeintellect", "benchflow")
# What to check when the PrimeIntellect hub cannot be read.
_PRIME_NEXT = (
    "references are provider:owner/name; the PrimeIntellect hub needs the "
    "`prime` CLI installed and signed in"
)


def hosted_env_list(
    *,
    provider: str,
    owner: str | None,
    search: str | None,
    limit: int | None,
    output_json: bool,
) -> None:
    """List a hub's environments (table, or raw JSON to stdout)."""
    if provider == "primeintellect":
        _list_primeintellect(
            owner=owner, search=search, limit=limit, output_json=output_json
        )
    elif provider == "benchflow":
        if owner:
            print_error(
                "--owner is not supported for the benchflow hub",
                next_step="drop --owner, or narrow the list with --search",
            )
            raise typer.Exit(2)
        _list_benchflow(search=search, limit=limit, output_json=output_json)
    else:
        print_error(
            f"--provider {provider!r} is an unknown value; choose one of: "
            f"{', '.join(LIST_PROVIDERS)}",
            next_step="pass one of those to --provider",
        )
        raise typer.Exit(2)


def _list_primeintellect(
    *, owner: str | None, search: str | None, limit: int | None, output_json: bool
) -> None:
    from benchflow.hosted_env import HostedEnvError, prime_env_list

    try:
        raw = prime_env_list(owner=owner, search=search, limit=limit)
    except HostedEnvError as e:
        print_error(str(e), next_step=_PRIME_NEXT)
        raise typer.Exit(1) from None
    if output_json:
        # typer.echo, NOT console.print: Rich's console soft-wraps long lines to
        # the terminal width and injects a literal newline mid-string, which
        # turns the upstream's valid JSON into unparseable output. Write the
        # payload verbatim so `--json | jq` works at any width.
        typer.echo(raw)
        return
    data = json.loads(raw)
    rows = (
        data
        if isinstance(data, list)
        else data.get("environments", data.get("items", []))
    )
    table = Table(title="PrimeIntellect Environments")
    table.add_column("Environment", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Visibility")
    table.add_column("Updated", style="dim")
    for item in rows:
        name = (
            item.get("environment")
            or item.get("fullName")
            or item.get("name")
            or item.get("id")
            or ""
        )
        version = str(item.get("version") or item.get("latestVersion") or "")
        visibility = str(item.get("visibility") or item.get("private") or "")
        updated = str(item.get("updated_at") or item.get("updatedAt") or "")
        # escape(): cells are external-API strings that may contain Rich markup.
        table.add_row(
            escape(str(name)), escape(version), escape(visibility), escape(updated)
        )
    console.print(table)
    # The table is a (often small) page of a much larger catalog; without this
    # footer a user reasonably concludes the provider has only these few.
    total = (
        data.get("total") or data.get("totalCount") if isinstance(data, dict) else None
    )
    suffix = f" of {total}" if isinstance(total, int) and total > len(rows) else ""
    console.print(
        f"[dim]Showing {len(rows)}{suffix} environment(s). Refine with "
        "--search/--owner, raise --limit, or use --json for the full payload.[/dim]"
    )


def _list_benchflow(
    *, search: str | None, limit: int | None, output_json: bool
) -> None:
    from benchflow.hub.registry import load_hub_registry

    try:
        datasets = load_hub_registry()
    except (OSError, ValueError) as e:
        # A network/file failure or malformed registry must not dump a traceback.
        print_error(
            f"could not load BenchFlow's registry: {e}",
            next_step="check this machine's network, then run the command again",
        )
        raise typer.Exit(1) from None

    rows = datasets
    if search:
        q = search.lower()
        rows = [
            d
            for d in rows
            if q in str(d.get("name", "")).lower()
            or q in str(d.get("description", "")).lower()
        ]
    matched = len(rows)
    if limit is not None:
        rows = rows[:limit]

    if output_json:
        typer.echo(json.dumps(rows))
        return

    table = Table(title="BenchFlow Registry")
    table.add_column("Dataset", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Tasks", justify="right")
    table.add_column("Description", style="dim")
    for d in rows:
        tasks = d.get("tasks") or []
        n_tasks = str(len(tasks)) if isinstance(tasks, list) else "0"
        desc = str(d.get("description", "") or "")
        if len(desc) > 80:
            desc = desc[:79] + "…"
        table.add_row(
            escape(str(d.get("name", ""))),
            escape(str(d.get("version", "") or "")),
            n_tasks,
            escape(desc),
        )
    console.print(table)
    suffix = f" of {matched}" if matched > len(rows) else ""
    console.print(
        f"[dim]Showing {len(rows)}{suffix} dataset(s). Refine with "
        "--search, raise --limit, or use --json for the full registry.[/dim]"
    )


def hosted_env_show(
    *,
    source_env: str,
    version: str | None,
    provider: str | None = None,
    output_json: bool = False,
) -> None:
    """Show a BenchFlow dataset's tasks, or hosted environment metadata.

    Without ``provider``, ``name@version`` is a dataset of BenchFlow's registry
    and ``owner/name`` a PrimeIntellect environment (dogfood p1-6).
    """
    if source_env.startswith("benchflow:"):
        source_env = source_env.removeprefix("benchflow:")
        provider = provider or "benchflow"
    if provider is None:
        provider = "primeintellect" if "/" in source_env else "benchflow"
    if provider == "benchflow":
        _show_benchflow(source_env, version, output_json=output_json)
        return
    if provider != "primeintellect":
        print_error(
            f"--provider {provider!r} is an unknown value; choose one of: "
            f"{', '.join(LIST_PROVIDERS)}",
            next_step="pass one of those to --provider",
        )
        raise typer.Exit(2)
    if output_json:
        # `prime env info` prints text only.
        print_error(
            "--json shows a dataset of BenchFlow's registry; a hosted "
            "environment's metadata is text",
            next_step="drop --json, or pass a dataset as name@version",
        )
        raise typer.Exit(2)
    from benchflow.hosted_env import HostedEnvError, HostedEnvRef, prime_env_info

    try:
        ref = HostedEnvRef.parse(source_env, version=version)
        console.print(prime_env_info(ref))
    except HostedEnvError as e:
        print_error(str(e), next_step=_PRIME_NEXT)
        raise typer.Exit(1) from None


def _show_benchflow(spec: str, version: str | None, *, output_json: bool) -> None:
    """The tasks of one dataset of BenchFlow's registry, one row each."""
    from benchflow._utils.dataset_registry import (
        DatasetResolutionError,
        parse_dataset_spec,
    )
    from benchflow.hub.registry import load_hub_registry

    if version and "@" not in spec:
        spec = f"{spec}@{version}"
    try:
        name, version = parse_dataset_spec(spec)
    except DatasetResolutionError as e:
        print_error(
            str(e),
            next_step="name the dataset as name@version; `bench hub list` lists them",
        )
        raise typer.Exit(2) from None
    try:
        datasets = load_hub_registry()
    except (OSError, ValueError) as e:
        print_error(
            f"could not load BenchFlow's registry: {e}",
            next_step="check this machine's network, then run the command again",
        )
        raise typer.Exit(1) from None
    entry = next(
        (
            d
            for d in datasets
            if d.get("name") == name and str(d.get("version", "")) == version
        ),
        None,
    )
    if entry is None:
        versions = sorted(
            {str(d.get("version", "")) for d in datasets if d.get("name") == name}
        )
        found = f" (its versions there: {', '.join(versions)})" if versions else ""
        print_error(
            f"Dataset {spec!r} is not in BenchFlow's registry{found}",
            next_step="`bench hub list` lists the datasets and their versions",
        )
        raise typer.Exit(2)
    if output_json:
        # The registry's entry as it stands, as `bench hub list --json` lists it.
        typer.echo(json.dumps(entry))
        return
    tasks = sorted(
        str(t.get("name", "")) for t in entry.get("tasks") or [] if isinstance(t, dict)
    )
    console.print(
        f"[bold]{escape(name)}@{escape(version)}[/bold]: {len(tasks)} task(s)"
    )
    table = Table()
    table.add_column("Task", style="cyan")
    for task in tasks:
        table.add_row(escape(task))
    console.print(table)
    description = str(entry.get("description", "") or "")
    if description:
        console.print(f"[dim]{escape(description)}[/dim]")
    if tasks:
        console.print(
            f"[dim]Run one: bench run -d {escape(name)}@{escape(version)} "
            f"-i {escape(tasks[0])}[/dim]"
        )


def hosted_env_inspect(*, source_env: str, version: str | None, path: str) -> None:
    """Inspect a file from a hosted environment package."""
    from benchflow.hosted_env import HostedEnvError, HostedEnvRef, prime_env_inspect

    try:
        ref = HostedEnvRef.parse(source_env, version=version)
        console.print(prime_env_inspect(ref, path=path))
    except HostedEnvError as e:
        print_error(str(e), next_step=_PRIME_NEXT)
        raise typer.Exit(1) from None
