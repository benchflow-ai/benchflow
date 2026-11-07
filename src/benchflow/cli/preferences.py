"""``bench preferences``: blind pairs of trial outputs, the human votes on
them, and ratings from those votes (:mod:`benchflow.preferences`).

Refusals write nothing: exit 3 for a vote id already recorded with other
content, exit 1 for any other refusal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

Store = Annotated[
    Path, typer.Argument(help="The preference store `bench preferences pairs` made.")
]


def _refused(exc: Exception) -> NoReturn:
    from benchflow.cli._shared import print_error
    from benchflow.preferences import Conflict

    print_error(
        str(exc),
        next_step=getattr(exc, "hint", None)
        or "`bench preferences --help` lists each command's inputs",
    )
    raise typer.Exit(3 if isinstance(exc, Conflict) else 1)


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}{'' if n == 1 else 's'}"


def _left_out_text(left_out: list[dict[str, Any]]) -> str:
    """``2 trials: arm: 1 (why); ...`` for the trials pairs left out."""
    from collections import Counter

    counts = Counter(
        (f"{row.get('agent') or '-'} / {row.get('model') or '-'}", row["reason"])
        for row in left_out
    )
    return f"{_count(len(left_out), 'trial')}: " + "; ".join(
        f"{arm}: {n} ({why})" for (arm, why), n in sorted(counts.items())
    )


def _rater_checks(result: dict[str, Any]) -> list[str]:
    """The lines to read before the ratings: what was counted, and whether
    the raters can be trusted (side bias, agreement against chance)."""
    left_out = [
        _count(result["abstentions"], "abstention"),
        f"{_count(result['replaced_votes'], 'vote')} a rater replaced later",
    ]
    if result.get("same_arm_votes"):
        left_out.append(
            f"{_count(result['same_arm_votes'], 'vote')} between outputs that "
            f"--by {result['by']} puts in one arm"
        )
    lines = [
        f"{result['effective_votes']} votes counted ({result['pairwise_votes']} "
        f"pairwise, {result['rubric_scores']} rubric scores) of {result['votes']} "
        f"recorded; left out: {', '.join(left_out[:-1])} and {left_out[-1]}. From "
        f"{len(result['raters'])} raters: "
        + ", ".join(f"{name} {n}" for name, n in result["raters"].items())
        + "."
    ]
    if result.get("left_out"):
        lines.append(
            f"Not in any pair: {_left_out_text(result['left_out'])}. Their arms are "
            "rated on their other outputs."
        )
    left = result["left_share"]
    if left["decisive"] and result.get("sides") == "fixed":
        lines.append(
            f"Left side chosen in {left['left']} of {left['decisive']} decisive "
            "overall votes; the sides were not shuffled (randomize_sides = false), "
            "so this says nothing about the raters."
        )
    elif left["decisive"]:
        z = left["z"]
        bias = (
            f"no side bias large enough to show in {left['decisive']} votes"
            if abs(z) < 2
            else f"more {'left' if z > 0 else 'right'} choices than chance gives, a "
            "possible side bias"
        )
        # z treats votes as independent, but raters of one pair saw the same sides.
        lines.append(
            f"Left side chosen in {left['left']} of {left['decisive']} decisive "
            f"overall votes (z = {z}, counting each vote apart although raters of a "
            f"pair saw the same sides: {bias})."
        )
    agreed = result["agreement"]
    if agreed["agreement"] is not None:
        line = (
            f"Two raters on the same pair gave the same answer {agreed['agreeing']} "
            f"times in {agreed['rater_pairs']} ({agreed['agreement']:.0%}), over the "
            f"{agreed['targets']} pairs judged by two or more raters"
        )
        if agreed.get("chance_agreement") is not None:
            line += (
                f"; their own choice rates would agree {agreed['chance_agreement']:.0%} "
                "of the time by chance"
            )
            if agreed.get("kappa") is not None:
                kappa = agreed["kappa"]
                line += f" (kappa {kappa:.2f}" + (
                    ": below chance)" if kappa < 0 else ")"
                )
        if agreed.get("met") is not None:
            line += (
                f"; the minimum raw agreement of {agreed['min_agreement']:.0%} is "
                + ("met" if agreed["met"] else "NOT met")
            )
        lines.append(line + ".")
    elif agreed.get("min_agreement") is not None:
        lines.append(
            "No pair was judged by two raters, so the minimum agreement of "
            f"{agreed['min_agreement']:.0%} can't be checked."
        )
    for aspect, raters in sorted((result.get("aspect_raters") or {}).items()):
        if raters < 2:
            lines.append(
                f"The {aspect} votes come from {raters} rater: no agreement check for them."
            )
    if result["coverage"]:
        c = result["coverage"]
        lines.append(
            f"{c['covered']} of {c['pairs']} pairs have {c['raters_per_pair']} or more raters."
        )
    return lines


def _print_ratings(result: dict[str, Any]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    # The checks come first: they say whether the tables below can be trusted.
    for line in _rater_checks(result):
        console.print(line, highlight=False, soft_wrap=True)
    # What the intervals are, before the numbers they qualify.
    console.print(
        "Ratings: Bradley-Terry on the Elo scale (1000 is a fixed reference; 400 "
        "points means preferred 10 to 1). Intervals: 95%, from resampling these "
        "votes as if they were independent, with these outputs and raters held "
        "fixed (pairwise), or the items (rubric, shown from 3 items up). They say "
        "nothing about other outputs of the same agents.",
        highlight=False,
        soft_wrap=True,
    )
    if not result["pairwise"]:
        console.print("No pairwise votes: no pairwise ratings.", highlight=False)
    for name, rows in result["pairwise"].items():
        table = Table(title=f"Pairwise ({name})", title_justify="left")
        for column in ("arm", "rating", "95% interval", "wins", "ties", "losses"):
            table.add_column(column, justify="left" if column == "arm" else "right")
        for row in rows:
            interval = row.get("interval")
            table.add_row(
                row["arm"],
                "unranked" if row["rating"] is None else f"{row['rating']:.0f}",
                f"{interval[0]:.0f} to {interval[1]:.0f}" if interval else "-",
                str(row["wins"]),
                str(row["ties"]),
                str(row["losses"]),
            )
        console.print(table)
    if result["rubric"]:
        table = Table(title="Rubric", title_justify="left")
        for column in ("arm", "criterion", "mean", "95% interval", "items"):
            table.add_column(
                column, justify="left" if column in ("arm", "criterion") else "right"
            )
        for row in result["rubric"]:
            interval = row.get("interval")
            table.add_row(
                row["arm"],
                row["criterion"],
                f"{row['mean']:.2f}",
                f"{interval[0]:.2f} to {interval[1]:.2f}" if interval else "-",
                str(row["items"]),
            )
        console.print(table)


def register_preferences(app: typer.Typer) -> None:
    group = typer.Typer(
        help=(
            "Human preferences and rubric scores on finished trials.\n\n"
            "`pairs` makes blind pairs of trial outputs (same task version, different "
            "agent or model) for a rater's tool, `import` takes the votes and rubric "
            "scores it collected, and `rank` gives Bradley-Terry ratings on the Elo "
            "scale with rater checks. Votes are never rewards: no trial file is "
            "changed. See docs/tasks/human-preferences.md."
        ),
        no_args_is_help=True,
    )
    app.add_typer(group, name="preferences")

    @group.command("pairs")
    def pairs_cmd(
        jobs: Annotated[
            list[Path],
            typer.Argument(help="Job or trial folders whose outputs to pair."),
        ],
        store: Annotated[
            Path, typer.Option("--store", help="A new folder for the preference store.")
        ],
        out: Annotated[
            Path,
            typer.Option(
                "--out",
                help="The file to write the blind pairs to, for the raters' tool.",
            ),
        ],
        output: Annotated[
            str,
            typer.Option(
                "--output",
                help="Which folder of each trial the raters compare, relative to "
                "the trial (artifacts, or evidence/workspace with "
                "--freeze-workspace).",
            ),
        ] = "artifacts",
        per_task: Annotated[
            int | None,
            typer.Option("--per-task", help="At most this many pairs per task."),
        ] = None,
        seed: Annotated[
            int | None,
            typer.Option(
                "--seed",
                help="Seed for sampling and sides (default: random, recorded).",
            ),
        ] = None,
        rubric: Annotated[
            Path | None,
            typer.Option(
                "--rubric", help="A rubric JSON file to freeze for aspects and scores."
            ),
        ] = None,
        bundle: Annotated[
            Path | None,
            typer.Option(
                "--bundle",
                help="Where to copy the outputs under their opaque ids for the "
                "rating tool (default: <store>-outputs beside the store).",
            ),
        ] = None,
        keep_sides: Annotated[
            bool,
            typer.Option(
                "--keep-sides", help="Do not shuffle which output is shown on the left."
            ),
        ] = False,
    ) -> None:
        """Make the store and the blind assignment."""
        from benchflow.preferences import PreferenceError, issue

        try:
            assignment = issue(
                jobs,
                store,
                output=output,
                per_task=per_task,
                seed=seed,
                rubric=rubric,
                bundle=bundle,
                randomize_sides=False if keep_sides else None,
            )
        except (PreferenceError, OSError) as exc:
            _refused(exc)
        out.write_text(json.dumps(assignment, indent=1) + "\n")
        manifest = json.loads((store / "store.v1.json").read_text())
        typer.echo(
            f"{len(assignment['pairs'])} pairs of {len(assignment['items'])} outputs."
        )
        left_out = manifest.get("left_out") or []
        if left_out:
            typer.echo(f"Left out {_left_out_text(left_out)}.")
        typer.echo(
            f"For the rating tool: {out}, and the outputs under their opaque ids in "
            f"{manifest['bundle']}."
        )
        typer.echo(f"Store (import and rank read it): {store}")

    @group.command("import")
    def import_cmd(
        store: Store,
        votes: Annotated[
            Path,
            typer.Option(
                "--votes",
                help="Votes as JSON lines or CSV (see the docs for the fields).",
            ),
        ],
        source: Annotated[
            str | None,
            typer.Option(
                "--source", help="The tool that collected them, recorded on each vote."
            ),
        ] = None,
    ) -> None:
        """Check every vote in the file, then add the new ones to the store."""
        from benchflow.preferences import PreferenceError, import_votes

        try:
            counts = import_votes(store, votes, source=source)
        except (PreferenceError, OSError, ValueError) as exc:
            _refused(exc)
        added, again = counts["added"], counts["already_recorded"]
        typer.echo(
            f"{added} vote{'s' if added != 1 else ''} added, {again} already recorded."
        )
        if counts.get("head_sha256"):
            typer.echo(
                f"Log head: {counts['head_sha256']} (keep a copy elsewhere: the log "
                "alone cannot show that its last lines were cut)."
            )

    @group.command("rank")
    def rank_cmd(
        store: Store,
        by: Annotated[
            str, typer.Option("--by", help="Rate each agent, model or agent+model.")
        ] = "agent+model",
        bootstrap: Annotated[
            int,
            typer.Option(
                "--bootstrap", help="Resamples for the 95% intervals (0: none)."
            ),
        ] = 1000,
        seed: Annotated[
            int, typer.Option("--seed", help="Seed for the resampling.")
        ] = 0,
        min_agreement: Annotated[
            float | None,
            typer.Option(
                "--min-agreement",
                help="Raw agreement (not kappa) to check the raters against (0 to"
                " 1); rank says whether it is met.",
            ),
        ] = None,
        raters_per_pair: Annotated[
            int | None,
            typer.Option("--raters-per-pair", help="Raters each pair should have."),
        ] = None,
        as_json: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
    ) -> None:
        """Ratings from the store's votes (also written to ratings.v1.json)."""
        from benchflow.preferences import PreferenceError, rank

        try:
            result = rank(
                store,
                by=by,
                bootstrap=bootstrap,
                seed=seed,
                min_agreement=min_agreement,
                raters_per_pair=raters_per_pair,
            )
        except (PreferenceError, OSError) as exc:
            _refused(exc)
        if as_json:
            typer.echo(json.dumps(result, indent=1))
        else:
            _print_ratings(result)
