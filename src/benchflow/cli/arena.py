"""``benchflow arena run --agents agents.yaml`` — the native concurrent floor.

Runs N agents on ONE shared task + its ONE service CONCURRENTLY in ONE shared
sandbox (each agent in /work/<seat>), with per-agent ACP + raw-LLM trajectories
in separate files. Registered onto the top-level app by :func:`register_arena`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from benchflow.cli._shared import console, print_error


def register_arena(app: typer.Typer) -> None:
    """Attach the ``arena`` command group to the top-level benchflow app."""
    arena_app = typer.Typer(help="Native concurrent multi-agent floor (--agents agents.yaml).")
    app.add_typer(arena_app, name="arena")

    @arena_app.command("run")
    def arena_run(
        agents: Annotated[
            Path,
            typer.Option("--agents", help="Path to agents.yaml (seats + shared task)."),
        ],
        out: Annotated[
            Path | None,
            typer.Option("--out", help="Output dir (default: the manifest's `out:`)."),
        ] = None,
        environment: Annotated[
            str, typer.Option("--environment", help="docker (daytona parity deferred)."),
        ] = "docker",
    ) -> None:
        """Run every seat in agents.yaml concurrently against one shared service."""
        from benchflow.arena.agents_manifest import AgentsManifest
        from benchflow.arena.bootstrap import run_native_floor

        if environment != "docker":
            print_error(
                f"--environment {environment!r} not supported yet; only 'docker' "
                "(shared-sandbox daytona parity is deferred)."
            )
            raise typer.Exit(1)

        manifest = AgentsManifest.from_yaml(agents)
        if out is not None:
            manifest.out = str(out)
        seats = manifest.seats()
        console.print(
            f"[bold]arena[/bold]: {len(seats)} seats · drive={manifest.drive} · "
            f"{', '.join(s.seat_id for s in seats)}"
        )

        try:
            summary = asyncio.run(run_native_floor(manifest, environment=environment))
        except (SystemExit, RuntimeError) as exc:
            print_error(str(exc))
            raise typer.Exit(1) from exc

        run_dir = Path(manifest.out)
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
            f"per-seat trajectories under {run_dir}"
        )
        if not played:
            raise typer.Exit(1)
