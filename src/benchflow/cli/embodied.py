"""``bench embodied`` — embodied rollouts: seeded-rollout reports, training export, spec checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.markup import escape

from benchflow.cli._shared import console, print_error


def register_embodied(app: typer.Typer) -> None:
    """Attach the ``embodied`` command group to the top-level benchflow app."""
    emb_app = typer.Typer(
        help="Embodied rollouts (robots and simulators): reports, export, specs."
    )
    app.add_typer(emb_app, name="embodied", rich_help_panel="Core")

    @emb_app.command("report")
    def embodied_report(
        job_dir: Annotated[
            Path, typer.Argument(help="BenchFlow job directory (<jobs>/<job>)")
        ],
        as_json: Annotated[
            bool, typer.Option("--json", help="Print the report as JSON")
        ] = False,
    ) -> None:
        """Per task: rewards by seed, mean, std, pass@k and reset reproducibility."""
        from benchflow.embodied.rollouts import format_report, seed_report

        if not job_dir.is_dir():
            print_error(f"no job directory {job_dir}")
            raise typer.Exit(1)
        report = seed_report(job_dir)
        if as_json:
            print(json.dumps(report, indent=2))
            return
        if not report["tasks"]:
            print_error(f"no trials in {job_dir}")
            raise typer.Exit(1)
        print(format_report(report))  # a fixed-width table: no console wrapping

    @emb_app.command("export")
    def embodied_export(
        job_dir: Annotated[
            Path, typer.Argument(help="BenchFlow job directory (<jobs>/<job>)")
        ],
        out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")],
    ) -> None:
        """Write every trial's per-step transitions and episode index (steps.jsonl, episodes.jsonl)."""
        from benchflow.embodied.export import export_job

        if not job_dir.is_dir():
            print_error(f"no job directory {job_dir}")
            raise typer.Exit(1)
        res = export_job(job_dir, out)
        if not res["episodes"]:
            print_error(f"no embodied episodes in {job_dir}")
            raise typer.Exit(1)
        console.print(
            f"{res['episodes']} episode(s), {res['steps']} step(s) -> {escape(res['out'])}"
        )

    @emb_app.command("check-spec")
    def embodied_check_spec(
        spec: Annotated[
            Path,
            typer.Argument(
                help="Embodiment spec JSON (e.g. the `embodiment` of `robo info --json`)"
            ),
        ],
    ) -> None:
        """Validate an embodiment spec."""
        from benchflow.embodied.spec import Embodiment, SpecError

        data = json.loads(spec.read_text())
        if isinstance(data.get("result"), dict):  # a raw `robo info --json` response
            data = data["result"]
        if isinstance(data.get("embodiment"), dict):
            data = data["embodiment"]
        try:
            e = Embodiment.from_dict(data).validate()
        except (SpecError, KeyError, TypeError) as exc:
            print_error(f"invalid embodiment spec: {exc}")
            raise typer.Exit(1) from None
        console.print(
            f"ok: {escape(e.name)} ({e.kind}), {len(e.action_groups)} action group(s), dim {e.dim}, "
            f"{len(e.skills)} skill(s)"
        )

    @emb_app.command("robo-path")
    def embodied_robo_path() -> None:
        """Print the path of the agent-side `robo` script (standard library only)."""
        from benchflow.embodied import robo

        print(Path(robo.__file__).resolve())
