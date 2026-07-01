"""The native concurrent multi-agent floor — standard `bench eval run` shape.

N agents share ONE task + its ONE IN-SANDBOX service, each in /work/<seat>, with
per-agent ACP + raw-LLM trajectories in separate files. The roster (`--agents`) is
the file form of repeated `--agent/--model`; everything else (task/service/sandbox)
follows single-agent `eval run`. Exposed as `bench eval run --agents …` (see
``run_floor_from_cli``); ``arena run`` is the deprecated alias.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from benchflow.cli._shared import console, print_error


def run_floor_from_cli(
    *,
    agents: Path,
    environment_manifest: Path,
    out: Path,
    game: str | None = None,
    sandbox: str = "docker",
    drive: str = "auto-loop",
    prompt: str | None = None,
    url_env: str | None = None,
    seat_env: str | None = None,
    standings_path: str | None = None,
    events_path: str | None = None,
    service_env: list[str] | None = None,
    deadline_s: int = 1200,
) -> None:
    """Shared entry for `eval run --agents` and the `arena run` alias."""
    from benchflow.arena.bootstrap import run_native_floor
    from benchflow.arena.concurrent_floor import FloorConfig
    from benchflow.arena.roster import Roster

    roster = Roster.from_yaml(agents)
    seats = roster.seats()
    body = prompt
    if body and body.startswith("@"):
        body = Path(body[1:]).read_text()
    cfg = FloorConfig(
        out=str(out), drive=drive, prompt=body, deadline_s=deadline_s,
        url_env=url_env, seat_env=seat_env, standings_path=standings_path,
        events_path=events_path, environment=sandbox,
    )
    console.print(
        f"[bold]floor[/bold]: {len(seats)} seats · drive={drive} · sandbox={sandbox} · "
        f"{', '.join(s.seat_id for s in seats)}"
    )
    try:
        svc_env = dict(kv.split("=", 1) for kv in (service_env or []) if "=" in kv) or None
        summary = asyncio.run(run_native_floor(
            roster, environment_manifest=environment_manifest, config=cfg, game=game,
            service_env=svc_env,
        ))
    except (SystemExit, RuntimeError) as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc

    console.print("\n[bold]floor results[/bold]")
    played = 0
    for r in summary["results"]:
        played += r["acp_tool_calls"] > 0
        raw = "raw+acp" if r["raw"] else "acp-only"
        console.print(
            f"  {r['seat']:<22} {r['status']:<28} "
            f"acp={r['acp_tool_calls']} llm={r['llm_calls']} [{raw}]"
        )
    console.print(
        f"\n{played}/{len(summary['results'])} seats played · "
        f"per-seat trajectories under {out}"
    )
    if not played:
        raise typer.Exit(1)


def register_arena(app: typer.Typer) -> None:
    """Attach the deprecated ``arena`` alias. The supported entry is
    `bench eval run --agents` (see ``cli.main``)."""
    arena_app = typer.Typer(help="[deprecated] use `bench eval run --agents roster.yaml`.")
    app.add_typer(arena_app, name="arena", hidden=True)

    @arena_app.command("run")
    def arena_run(
        agents: Annotated[Path, typer.Option("--agents", help="Roster file (agents only).")],
        environment_manifest: Annotated[
            Path, typer.Option("--environment-manifest", help="environment.toml (in-sandbox service).")
        ],
        out: Annotated[Path, typer.Option("--out", help="Output dir.")] = Path("out/native-floor"),
        game: Annotated[str | None, typer.Option("--game", help="task_selection value (e.g. game id).")] = None,
        sandbox: Annotated[str, typer.Option("--sandbox", help="docker | daytona.")] = "docker",
        drive: Annotated[str, typer.Option("--drive", help="auto-loop | service-rounds.")] = "auto-loop",
        prompt: Annotated[str | None, typer.Option("--prompt", help="Shared prompt (or @file).")] = None,
        url_env: Annotated[str | None, typer.Option("--url-env")] = None,
        seat_env: Annotated[str | None, typer.Option("--seat-env")] = None,
        standings_path: Annotated[str | None, typer.Option("--standings-path")] = None,
        events_path: Annotated[str | None, typer.Option("--events-path")] = None,
        multiplayer: Annotated[bool, typer.Option("--multiplayer")] = False,
    ) -> None:
        """[deprecated] Alias of `bench eval run --agents`."""
        console.print("[yellow]`arena run` is deprecated — use `bench eval run --agents`.[/yellow]")
        run_floor_from_cli(
            agents=agents, environment_manifest=environment_manifest, out=out, game=game,
            sandbox=sandbox, drive=drive, prompt=prompt, url_env=url_env, seat_env=seat_env,
            standings_path=standings_path, events_path=events_path, multiplayer=multiplayer,
        )
