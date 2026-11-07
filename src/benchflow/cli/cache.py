"""``bench cache``: what BenchFlow keeps on this machine between runs.

``bench cache list`` (the default) shows it; ``bench cache clean`` prints
what it would remove, and removes it only with ``--remove``. The spelling is
Rust 0.8.2's (``list``, ``clean``, ``--only``, ``-s``, ``--older-than``);
what Python keeps differs, and so do the kinds:

- ``images``: the cached layer images (``benchflow-layer:<kind>-<key>``,
  agent and proxy, label-checked), the task-image pins a killed layer build
  left behind, and with ``-s daytona`` this owner's content snapshots;
- ``build-cache``: the BuildKit records of those layers, found by their
  marker;
- ``pulled-images``: task images BenchFlow itself pulled (only when named
  with ``--only``; an image that was there before is never one).

``--older-than`` is the creation age (``age_basis: created``), not Rust's
last use. Checkpoints and sandboxes are ``bench sandbox``'s; sources,
registries and logs are not kept by Python BenchFlow. List and a dry clean
never remove or tag anything.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, NoReturn

import typer

IMAGES = "images"
BUILD_CACHE = "build-cache"
PULLED = "pulled-images"
KINDS = (IMAGES, BUILD_CACHE, PULLED)
DEFAULT_KINDS = (IMAGES, BUILD_CACHE)
# Rust's kinds Python keeps no cache of, and where to look instead.
OTHER_KINDS = {
    "checkpoints": "Python keeps checkpoints as sandbox snapshots: run "
    "`bench sandbox list` or `bench sandbox cleanup`",
    "sandboxes": "run `bench sandbox list` or `bench sandbox cleanup`",
    "sources": "Python BenchFlow keeps no cache of task sources",
    "registries": "Python BenchFlow keeps no cache of registry indexes",
    "logs": "Python BenchFlow keeps no logs outside the job folders",
}
BACKENDS = ("docker", "daytona")
OTHER_BACKENDS = {
    "podman": "Python BenchFlow builds no cached layers on Podman",
    "remote-docker": "run `bench cache` with DOCKER_HOST (or a Docker context) "
    "pointed at that daemon, as `-s docker`",
}
AGE_BASIS = "created"


@dataclass
class Item:
    """One row: what it is, its size and age, and what happens to it."""

    backend: str
    kind: str
    type: str  # layer, records, pin, snapshot, pulled
    name: str
    bytes: int | None
    created: datetime | None
    detail: dict[str, Any] = field(default_factory=dict)
    kept: str | None = None
    action: str = "listed"
    target: Any = None

    def as_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "kind": self.kind,
            "type": self.type,
            "name": self.name,
            "bytes": self.bytes,
            "created": self.created.isoformat() if self.created else None,
            "kept": self.kept,
            "action": self.action,
            **self.detail,
        }


class UsageProblem(Exception):
    """A flag value this command cannot take; exit 2 with the message."""


# The next step of a UsageProblem.
_USAGE_NEXT = "`bench cache --help` lists the kinds, backends and ages it takes"


def parse_age(text: str) -> timedelta:
    match = re.fullmatch(r"\s*(\d+)\s*([smhdw])\s*", text or "")
    if match is None:
        raise UsageProblem(
            f"--older-than {text!r}: use a number and a unit, e.g. 30d, 12h, 45m"
        )
    n = int(match.group(1))
    unit = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    return timedelta(**{unit[match.group(2)]: n})


def _split(values: list[str] | None) -> list[str]:
    out: list[str] = []
    for value in values or []:
        out += [v.strip() for v in value.split(",") if v.strip()]
    return out


def resolve_kinds(only: list[str] | None) -> tuple[str, ...]:
    asked = _split(only)
    if not asked:
        return DEFAULT_KINDS
    for kind in asked:
        if kind in OTHER_KINDS:
            raise UsageProblem(f"--only {kind}: {OTHER_KINDS[kind]}")
        if kind not in KINDS:
            raise UsageProblem(f"--only {kind}: not a kind; use {', '.join(KINDS)}")
    return tuple(k for k in KINDS if k in asked)


def resolve_backends(sandbox: list[str] | None) -> tuple[str, ...]:
    asked = _split(sandbox)
    for backend in asked:
        if backend in OTHER_BACKENDS:
            raise UsageProblem(f"-s {backend}: {OTHER_BACKENDS[backend]}")
        if backend not in BACKENDS:
            raise UsageProblem(f"-s {backend}: use docker or daytona")
    if asked:
        return tuple(b for b in BACKENDS if b in asked)
    found = []
    if shutil.which("docker"):
        found.append("docker")
    if os.environ.get("DAYTONA_API_KEY"):
        found.append("daytona")
    return tuple(found)


# -- Docker ------------------------------------------------------------


def _short(key: str) -> str:
    return key[:12]


async def _docker_items(kinds: tuple[str, ...], notes: list[str]) -> list[Item]:
    from benchflow.sandbox import env_layers

    layers = env_layers.DockerLayers()
    items: list[Item] = []
    if IMAGES in kinds or BUILD_CACHE in kinds:
        inv = await layers.inventory()
        notes.extend(f"docker: {n}" for n in inv.notes)
        if IMAGES in kinds:
            for layer in inv.layers:
                if layer.kind == env_layers.AGENT:
                    what = f"{layer.agent} {layer.agent_version}".strip()
                elif layer.kind == env_layers.VERIFIER:
                    what = layer.verifier
                else:
                    what = layer.proxy_group
                item = Item(
                    backend="docker",
                    kind=IMAGES,
                    type="layer",
                    name=layer.tag,
                    bytes=layer.size,
                    created=layer.created,
                    detail={
                        "layer": layer.kind,
                        "key": layer.key,
                        "what": what,
                        "base": layer.base,
                        "libc": layer.libc,
                        "held_by": list(layer.holders),
                        "dead_aliases": list(layer.dead_aliases),
                    },
                    target=(layer.kind, layer.key),
                )
                if layer.holders:
                    item.kept = "in use: " + "; ".join(layer.holders)
                items.append(item)
            for pin in inv.pins:
                items.append(
                    Item(
                        backend="docker",
                        kind=IMAGES,
                        type="pin",
                        name=pin.ref,
                        bytes=None,
                        created=pin.created,
                        detail={"layer": pin.kind, "key": pin.key, "owner": pin.owner},
                        kept=None if pin.removable else pin.reason,
                        target=pin.ref,
                    )
                )
        if BUILD_CACHE in kinds:
            for record in inv.records:
                items.append(
                    Item(
                        backend="docker",
                        kind=BUILD_CACHE,
                        type="records",
                        name=f"{record.kind} layer {_short(record.key)}",
                        bytes=record.size,
                        created=record.created,
                        detail={
                            "layer": record.kind,
                            "key": record.key,
                            "records": len(record.ids),
                            "image_present": record.image_present,
                        },
                        target=(record.kind, record.key),
                    )
                )
    if PULLED in kinds:
        pulled, problem = await env_layers.pulled_inventory(layers)
        if problem:
            notes.append(f"docker: pulled images were not listed: {problem}")
        for entry in pulled:
            items.append(
                Item(
                    backend="docker",
                    kind=PULLED,
                    type="pulled",
                    name=entry.record.ref,
                    bytes=entry.size if not entry.gone else 0,
                    created=entry.record.pulled,
                    detail={
                        "image_id": entry.record.image_id,
                        "gone": entry.gone,
                    },
                    kept=None if (entry.removable or entry.gone) else entry.reason,
                    target=entry,
                )
            )
    return items


async def _docker_remove(chosen: list[Item]) -> None:
    from benchflow.sandbox import env_layers

    layers = env_layers.DockerLayers()
    images = [i for i in chosen if i.type == "layer"]
    records = [i for i in chosen if i.type == "records"]
    pins = [i for i in chosen if i.type == "pin"]
    report = await layers.remove_selected(
        layers=[i.target for i in images],
        records=[i.target for i in records],
        pins=[i.target for i in pins],
    )
    removed_keys = set(report.removed)
    pruned = set(report.records_pruned)
    for item in images:
        _kind, key = item.target
        _settle(
            item, key in removed_keys, report.skipped.get(key), key in report.in_use
        )
    for item in records:
        _kind, key = item.target
        _settle(item, key in pruned, report.skipped.get(key), False)
    for item in pins:
        _settle(
            item,
            item.target in removed_keys,
            report.skipped.get(item.target),
            item.target in report.in_use,
        )
    pulled = [i for i in chosen if i.type == "pulled"]
    if pulled:
        gone, kept = await env_layers.remove_pulled(layers, [i.target for i in pulled])
        for item in pulled:
            ref = item.target.record.ref
            if item.target.gone:
                item.action = "forgotten"
            else:
                _settle(item, ref in gone, kept.get(ref), False)


def _settle(item: Item, done: bool, skipped: str | None, in_use: bool) -> None:
    if done:
        item.action = "removed"
    elif skipped or in_use:
        item.action = "kept"
        item.kept = skipped or "in use (looked at again before removing)"
    else:
        item.action = "failed"
        item.kept = "Docker did not remove it (see the log)"


# -- Daytona -----------------------------------------------------------


def _daytona_client() -> Any:
    from benchflow.sandbox.daytona import build_sync_client

    return build_sync_client()


def _daytona_items(kinds: tuple[str, ...], notes: list[str]) -> list[Item]:
    if IMAGES not in kinds:
        return []
    from benchflow.sandbox import daytona_content_snapshots as content
    from benchflow.sandbox.daytona_reaper import _parse_sandbox_timestamp

    if content.owner_prefix() is None:
        notes.append(
            "daytona: BENCHFLOW_DAYTONA_OWNER is not set, so no snapshot can be "
            "proved to be yours; none are listed"
        )
        return []
    snapshots, problem = content.list_own(_daytona_client())
    if problem:
        notes.append(f"daytona: {problem}")
    items = []
    for snap in snapshots:
        state = getattr(
            getattr(snap, "state", None), "value", getattr(snap, "state", "")
        )
        size = getattr(snap, "size", None)
        items.append(
            Item(
                backend="daytona",
                kind=IMAGES,
                type="snapshot",
                name=str(snap.name),
                bytes=int(float(size) * 1024**3)
                if isinstance(size, (int, float))
                else None,
                created=_parse_sandbox_timestamp(getattr(snap, "created_at", None)),
                detail={"state": str(state or "").lower()},
                target=snap,
            )
        )
    return items


def _daytona_remove(chosen: list[Item]) -> None:
    from benchflow.sandbox import daytona_content_snapshots as content

    client = _daytona_client()
    for item in chosen:
        try:
            why = content.delete_own(client, item.target)
        except Exception as e:
            item.action = "failed"
            item.kept = f"delete failed: {e}"
            continue
        if why is None:
            item.action = "removed"
        else:
            item.action = "kept"
            item.kept = why


# -- the command -------------------------------------------------------


def _human_size(n: int | None) -> str:
    if n is None:
        return "-"
    for unit, size in (("GB", 1000**3), ("MB", 1000**2), ("kB", 1000)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def _note(message: str, json_out: bool) -> None:
    print(message, file=sys.stderr if json_out else sys.stdout)


def run_cache(
    *,
    cleaning: bool,
    remove: bool,
    only: list[str] | None,
    sandbox: list[str] | None,
    older_than: str | None,
    json_out: bool,
) -> int:
    """List, dry-clean or clean; returns the exit code."""
    try:
        kinds = resolve_kinds(only)
        backends = resolve_backends(sandbox)
        age = parse_age(older_than) if older_than else None
    except UsageProblem as e:
        from benchflow.cli._shared import print_error

        print_error(str(e), next_step=_USAGE_NEXT)
        return 2
    notes: list[str] = []
    items: list[Item] = []
    failed = False
    if not backends:
        notes.append(
            "no backend found: Docker is not on PATH and DAYTONA_API_KEY is unset"
        )
    for backend in backends:
        try:
            if backend == "docker":
                items += asyncio.run(_docker_items(kinds, notes))
            else:
                items += _daytona_items(kinds, notes)
        except Exception as e:
            notes.append(f"{backend}: not listed: {e}")
            failed = True
    now = datetime.now(UTC)
    if age is not None:
        for item in items:
            if item.kept is None and (item.created is None or now - item.created < age):
                item.kept = (
                    "its creation time is unknown"
                    if item.created is None
                    else f"created less than {older_than} ago"
                )
    if cleaning:
        chosen = [i for i in items if i.kept is None]
        for item in items:
            item.action = "kept" if item.kept is not None else "would remove"
        if remove:
            docker_chosen = [i for i in chosen if i.backend == "docker"]
            daytona_chosen = [i for i in chosen if i.backend == "daytona"]
            if docker_chosen:
                asyncio.run(_docker_remove(docker_chosen))
            if daytona_chosen:
                _daytona_remove(daytona_chosen)
            failed = failed or any(i.action == "failed" for i in chosen)
    if json_out:
        payload = {
            "age_basis": AGE_BASIS,
            "kinds": list(kinds),
            "backends": list(backends),
            "items": [i.as_json() for i in items],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_table(items, cleaning=cleaning, remove=remove)
    for message in notes:
        _note(message, json_out)
    if failed:
        _note(
            "  next: `bench doctor` checks this machine's Docker and Daytona",
            json_out,
        )
    return 1 if failed else 0


def _print_table(items: list[Item], *, cleaning: bool, remove: bool) -> None:
    if not items:
        print("Nothing cached.")
        return
    for item in items:
        created = item.created.strftime("%Y-%m-%d %H:%M") if item.created else "-"
        what = item.detail.get("what") or item.detail.get("state") or ""
        head = f"{item.backend:<7} {item.kind:<13} {item.name}"
        line = f"{head}  {what}  {_human_size(item.bytes)}  created {created}".rstrip()
        if cleaning:
            line = f"{item.action:<12} {line}"
        if item.kept:
            line += f"  ({item.kept})"
        print(line)
    if cleaning and not remove:
        print(
            "Nothing was removed; add --remove to remove what is listed as 'would remove'."
        )


OnlyOption = Annotated[
    list[str] | None,
    typer.Option(
        "--only",
        metavar="KINDS",
        help="Only these kinds: images, build-cache, pulled-images "
        "(comma-separated or repeated; default: images and build-cache)",
    ),
]
SandboxBackendOption = Annotated[
    list[str] | None,
    typer.Option(
        "-s",
        "--sandbox",
        metavar="BACKEND",
        help="docker or daytona (default: docker when installed, daytona when "
        "DAYTONA_API_KEY is set); daytona lists this BENCHFLOW_DAYTONA_OWNER's "
        "content snapshots",
    ),
]
OlderThanOption = Annotated[
    str | None,
    typer.Option(
        "--older-than",
        metavar="DURATION",
        help="Only what was created longer ago than this (e.g. 30d, 12h); "
        "the age is the creation time, not the last use",
    ),
]
JsonOption = Annotated[
    bool, typer.Option("--json", help="Print JSON on stdout; notes go to stderr")
]


# Where `bench cache`'s own options (given before `list` or `clean`) wait
# for the command; click shares ctx.meta with the nested contexts.
_GROUP_OPTIONS = "benchflow.cache.group-options"
_FLAGS = {
    "only": "--only",
    "sandbox": "-s",
    "older_than": "--older-than",
    "json_out": "--json",
}


def _with_group_options(ctx: typer.Context, **own: Any) -> dict[str, Any]:
    """The command's options, with those given before it filled in.

    `bench cache --older-than 30d clean --remove` must remove what is 30 days
    old, never everything: an option given before the command counts as if
    given after it. The same option given both ways with different values
    raises UsageProblem rather than letting one of them silently win.
    """
    merged = dict(own)
    for name, value in (ctx.meta.get(_GROUP_OPTIONS) or {}).items():
        if not value:  # None, [] or False: not given before the command
            continue
        mine = own.get(name)
        if mine and mine != value:
            raise UsageProblem(
                f"{_FLAGS[name]} is given before and after `{ctx.info_name}` "
                "with different values; give it once"
            )
        merged[name] = value
    return merged


def _run_command(
    ctx: typer.Context, *, cleaning: bool, remove: bool, **own: Any
) -> NoReturn:
    try:
        options = _with_group_options(ctx, **own)
    except UsageProblem as e:
        from benchflow.cli._shared import print_error

        print_error(str(e), next_step=_USAGE_NEXT)
        raise typer.Exit(2) from None
    raise typer.Exit(run_cache(cleaning=cleaning, remove=remove, **options))


def register_cache(app: typer.Typer) -> None:
    cache_app = typer.Typer(invoke_without_command=True)

    @cache_app.callback(invoke_without_command=True)
    def cache_root(
        ctx: typer.Context,
        only: OnlyOption = None,
        sandbox: SandboxBackendOption = None,
        older_than: OlderThanOption = None,
        json_out: JsonOption = False,
    ) -> None:
        """List or clean what BenchFlow keeps: cached layers, build records, snapshots.

        Without a command, lists (as `bench cache list`).

        Examples:
          bench cache list --only images -s docker   # cached layers on Docker
          bench cache clean --older-than 30d         # what would go
        """
        if ctx.invoked_subcommand is not None:
            ctx.meta[_GROUP_OPTIONS] = {
                "only": only,
                "sandbox": sandbox,
                "older_than": older_than,
                "json_out": json_out,
            }
            return
        raise typer.Exit(
            run_cache(
                cleaning=False,
                remove=False,
                only=only,
                sandbox=sandbox,
                older_than=older_than,
                json_out=json_out,
            )
        )

    @cache_app.command("list")
    def cache_list(
        ctx: typer.Context,
        only: OnlyOption = None,
        sandbox: SandboxBackendOption = None,
        older_than: OlderThanOption = None,
        json_out: JsonOption = False,
    ) -> None:
        """List what BenchFlow keeps, with sizes and creation times (the default).

        Nothing is removed or tagged.

        Examples:
          bench cache list --only images -s docker   # cached layers on Docker
          bench cache list --older-than 30d --json   # made 30+ days ago, as JSON
        """
        _run_command(
            ctx,
            cleaning=False,
            remove=False,
            only=only,
            sandbox=sandbox,
            older_than=older_than,
            json_out=json_out,
        )

    @cache_app.command("clean")
    def cache_clean(
        ctx: typer.Context,
        only: OnlyOption = None,
        sandbox: SandboxBackendOption = None,
        older_than: OlderThanOption = None,
        json_out: JsonOption = False,
        remove: Annotated[
            bool,
            typer.Option(
                "--remove", help="Remove what is listed; without it, nothing is removed"
            ),
        ] = False,
    ) -> None:
        """Print what would be removed; remove it with --remove.

        What something still uses stays, with the reason.

        Examples:
          bench cache clean                            # what would be removed
          bench cache clean --only images --remove     # remove idle cached layers
          bench cache clean --only pulled-images --remove
        """
        _run_command(
            ctx,
            cleaning=True,
            remove=remove,
            only=only,
            sandbox=sandbox,
            older_than=older_than,
            json_out=json_out,
        )

    app.add_typer(cache_app, name="cache")
