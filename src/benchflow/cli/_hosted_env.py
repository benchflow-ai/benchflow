"""Shared hosted-environment read commands.

Canonically reached via ``bench hub list|show|inspect`` (external
environment-hub browsing). ``list`` browses multiple hubs — PrimeIntellect
hosted "Environments" and the Harbor benchmark registry — dispatched on
``--provider``. ``show``/``inspect`` currently target the PrimeIntellect CLI.

The deprecated ``bench environment list --provider`` / ``environment show`` /
``environment inspect`` aliases delegate here so there is exactly one copy of
the logic. The actual API/registry helpers live in ``benchflow.hosted_env`` and
``benchflow.hub.harbor_registry`` and are untouched.
"""

from __future__ import annotations

import json

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import console, print_error

#: Hubs that ``bench hub list`` can browse.
LIST_PROVIDERS = ("primeintellect", "harbor")


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
    elif provider == "harbor":
        if owner:
            print_error("--owner is not supported for the harbor hub")
            raise typer.Exit(1)
        _list_harbor(search=search, limit=limit, output_json=output_json)
    else:
        print_error(
            f"Unknown --provider {provider!r}; supported: {', '.join(LIST_PROVIDERS)}"
        )
        raise typer.Exit(1)


def _list_primeintellect(
    *, owner: str | None, search: str | None, limit: int | None, output_json: bool
) -> None:
    from benchflow.hosted_env import HostedEnvError, prime_env_list

    try:
        raw = prime_env_list(owner=owner, search=search, limit=limit)
    except HostedEnvError as e:
        print_error(str(e))
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


def _list_harbor(*, search: str | None, limit: int | None, output_json: bool) -> None:
    from benchflow.hub.harbor_registry import (
        DEFAULT_HARBOR_REGISTRY_URL,
        load_harbor_registry,
    )

    try:
        datasets = load_harbor_registry(DEFAULT_HARBOR_REGISTRY_URL)
    except (OSError, ValueError) as e:
        # A network/file failure or malformed registry must not dump a traceback.
        print_error(f"could not load the Harbor registry: {e}")
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

    table = Table(title="Harbor Environments")
    table.add_column("Environment", style="cyan")
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
        f"[dim]Showing {len(rows)}{suffix} Harbor environment(s). Refine with "
        "--search, raise --limit, or use --json for the full registry.[/dim]"
    )


def hosted_env_show(*, source_env: str, version: str | None) -> None:
    """Show hosted environment metadata."""
    from benchflow.hosted_env import HostedEnvError, HostedEnvRef, prime_env_info

    try:
        ref = HostedEnvRef.parse(source_env, version=version)
        console.print(prime_env_info(ref))
    except HostedEnvError as e:
        print_error(str(e))
        raise typer.Exit(1) from None


def hosted_env_inspect(*, source_env: str, version: str | None, path: str) -> None:
    """Inspect a file from a hosted environment package."""
    from benchflow.hosted_env import HostedEnvError, HostedEnvRef, prime_env_inspect

    try:
        ref = HostedEnvRef.parse(source_env, version=version)
        console.print(prime_env_inspect(ref, path=path))
    except HostedEnvError as e:
        print_error(str(e))
        raise typer.Exit(1) from None
