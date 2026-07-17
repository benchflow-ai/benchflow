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

_SUPPORTED_FLOOR_SANDBOXES = ("docker", "daytona")


def _validate_floor_sandbox(sandbox: str) -> None:
    if sandbox not in _SUPPORTED_FLOOR_SANDBOXES:
        choices = " or ".join(_SUPPORTED_FLOOR_SANDBOXES)
        print_error(f"Invalid --sandbox {sandbox!r}: choose {choices}")
        raise typer.Exit(1)


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
    reasoning_effort: str | None = None,
    usage_tracking: str | None = None,
    agent_idle_timeout: str | None = None,
) -> None:
    """Shared entry for `eval run --agents` and the `arena run` alias."""
    from benchflow._utils.config import (
        normalize_agent_idle_timeout,
        normalize_reasoning_effort,
    )
    from benchflow.arena.bootstrap import run_native_floor
    from benchflow.arena.concurrent_floor import FloorConfig
    from benchflow.arena.roster import Roster
    from benchflow.usage_tracking import UsageTrackingConfig

    _validate_floor_sandbox(sandbox)
    roster = Roster.from_yaml(agents)
    seats = roster.seats()
    if deadline_s <= 0:
        deadline_s = 86400  # "no deadline" — capped at 24h so a wedged run still ends
    body = prompt
    if body and body.startswith("@"):
        body = Path(body[1:]).read_text()
    try:
        normalized_reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        normalized_idle_timeout = (
            normalize_agent_idle_timeout(agent_idle_timeout)
            if agent_idle_timeout is not None
            else 300
        )
        usage_cfg = (
            UsageTrackingConfig(mode=usage_tracking)
            if usage_tracking is not None
            else None
        )
    except ValueError as exc:
        print_error(str(exc))
        raise typer.Exit(1) from exc
    cfg = FloorConfig(
        out=str(out),
        drive=drive,
        prompt=body,
        deadline_s=deadline_s,
        idle_timeout_s=normalized_idle_timeout,
        url_env=url_env,
        seat_env=seat_env,
        standings_path=standings_path,
        events_path=events_path,
        environment=sandbox,
        reasoning_effort=normalized_reasoning_effort,
        usage_tracking=usage_cfg,
    )
    console.print(
        f"[bold]floor[/bold]: {len(seats)} seats · drive={drive} · sandbox={sandbox} · "
        f"{', '.join(s.seat_id for s in seats)}"
    )
    try:
        svc_env = (
            dict(kv.split("=", 1) for kv in (service_env or []) if "=" in kv) or None
        )
        summary = asyncio.run(
            run_native_floor(
                roster,
                environment_manifest=environment_manifest,
                config=cfg,
                game=game,
                service_env=svc_env,
            )
        )
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
    arena_app = typer.Typer(
        help="[deprecated] use `bench eval run --agents roster.yaml`."
    )
    app.add_typer(arena_app, name="arena", hidden=True)

    @arena_app.command("run")
    def arena_run(
        agents: Annotated[
            Path, typer.Option("--agents", help="Roster file (agents only).")
        ],
        environment_manifest: Annotated[
            Path,
            typer.Option(
                "--environment-manifest", help="environment.toml (in-sandbox service)."
            ),
        ],
        out: Annotated[Path, typer.Option("--out", help="Output dir.")] = Path(
            "out/native-floor"
        ),
        game: Annotated[
            str | None,
            typer.Option("--game", help="task_selection value (e.g. game id)."),
        ] = None,
        sandbox: Annotated[
            str, typer.Option("--sandbox", help="docker | daytona.")
        ] = "docker",
        drive: Annotated[
            str, typer.Option("--drive", help="auto-loop | service-rounds.")
        ] = "auto-loop",
        prompt: Annotated[
            str | None, typer.Option("--prompt", help="Shared prompt (or @file).")
        ] = None,
        url_env: Annotated[str | None, typer.Option("--url-env")] = None,
        seat_env: Annotated[str | None, typer.Option("--seat-env")] = None,
        standings_path: Annotated[str | None, typer.Option("--standings-path")] = None,
        events_path: Annotated[str | None, typer.Option("--events-path")] = None,
        service_env: Annotated[
            list[str] | None,
            typer.Option(
                "--service-env",
                help="Extra KEY=VALUE env for the in-sandbox service.",
            ),
        ] = None,
        deadline: Annotated[
            int,
            typer.Option(
                "--deadline",
                help="Soft deadline in seconds (0 = no deadline, capped at 24h).",
            ),
        ] = 1200,
        reasoning_effort: Annotated[
            str | None, typer.Option("--reasoning-effort")
        ] = None,
        usage_tracking: Annotated[
            str | None, typer.Option("--usage-tracking")
        ] = None,
        agent_idle_timeout: Annotated[
            str | None, typer.Option("--agent-idle-timeout")
        ] = None,
        multiplayer: Annotated[bool, typer.Option("--multiplayer")] = False,
    ) -> None:
        """[deprecated] Alias of `bench eval run --agents`."""
        console.print(
            "[yellow]`arena run` is deprecated — use `bench eval run --agents`.[/yellow]"
        )
        run_floor_from_cli(
            agents=agents,
            environment_manifest=environment_manifest,
            out=out,
            game=game,
            sandbox=sandbox,
            drive=drive,
            prompt=prompt,
            url_env=url_env,
            seat_env=seat_env,
            standings_path=standings_path,
            events_path=events_path,
            service_env=service_env,
            deadline_s=deadline,
            reasoning_effort=reasoning_effort,
            usage_tracking=usage_tracking,
            agent_idle_timeout=agent_idle_timeout,
        )
