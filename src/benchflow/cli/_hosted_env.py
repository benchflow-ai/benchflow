"""Shared hosted-environment read commands (PrimeIntellect "Environments").

Canonically reached via ``bench hub env list|show|inspect`` (external
environment-hub browsing); the deprecated ``bench environment list --provider``
/ ``environment show`` / ``environment inspect`` aliases delegate here so there
is exactly one copy of the logic. The actual API helpers live in
``benchflow.hosted_env`` and are untouched.
"""

from __future__ import annotations

import json

import typer
from rich.markup import escape
from rich.table import Table

from benchflow.cli._shared import console, print_error

_SUPPORTED_PROVIDER = "primeintellect"


def hosted_env_list(
    *,
    provider: str,
    owner: str | None,
    search: str | None,
    limit: int | None,
    output_json: bool,
) -> None:
    """List a hosted provider's environments (table, or raw JSON to stdout)."""
    if provider != _SUPPORTED_PROVIDER:
        print_error(
            f"Only --provider {_SUPPORTED_PROVIDER} is supported today (got {provider!r})"
        )
        raise typer.Exit(1)
    from benchflow.hosted_env import HostedEnvError, prime_env_list

    try:
        raw = prime_env_list(owner=owner, search=search, limit=limit)
    except HostedEnvError as e:
        print_error(str(e))
        raise typer.Exit(1) from None
    if output_json:
        console.print(raw)
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
