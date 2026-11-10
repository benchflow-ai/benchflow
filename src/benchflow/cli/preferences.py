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
    """``2 trials: arm: 1 (why); ...`` for the trials (and listed outputs)
    pairs left out."""
    from collections import Counter

    counts = Counter(
        (f"{row.get('agent') or '-'} / {row.get('model') or '-'}", row["reason"])
        for row in left_out
    )
    trials = sum("trial" in row for row in left_out)
    noun = (
        _count(trials, "trial")
        if trials == len(left_out)
        else _count(len(left_out), "output")
    )
    return f"{noun}: " + "; ".join(
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
    lines += _kind_lines(result)
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
    c = result.get("coverage")
    if c and c["raters_per_pair"] > 1:
        lines.append(
            f"{c['covered']} of {c['pairs']} pairs have {c['raters_per_pair']} or more raters."
        )
    if c and result.get("ranked") is False:
        missing = c["pairs"] - result.get("voted_pairs", 0)
        lines.append(
            f"Not ranked yet: {missing} of {c['pairs']} pairs "
            f"{'has' if missing == 1 else 'have'} no vote from {_whose(result)}; until "
            "every pair has one, the arms are listed by name without ratings "
            "(--provisional shows them)."
        )
    return lines


_KIND_NOUN = {
    "person": ("person", "people"),
    "guest": ("guest", "guests"),
    "model": ("model judge", "model judges"),
}


def _whose(result: dict[str, Any]) -> str:
    """Whose votes the ratings are: people, unless no person voted."""
    kinds = (result.get("headline") or {}).get("kinds") or ["person"]
    return " or ".join(_KIND_NOUN[k][1] for k in kinds)


def _kind_lines(result: dict[str, Any]) -> list[str]:
    """Whose votes the ratings are, and the votes rated apart from them."""
    lines = []
    note = (result.get("headline") or {}).get("note")
    if note:
        lines.append(note)
    for kind, part in (result.get("by_kind") or {}).items():
        one, many = _KIND_NOUN[kind]
        lines.append(
            f"Rated apart: {_count(part['votes'], 'vote')} from {part['raters']} "
            f"{one if part['raters'] == 1 else many}, in tables of their own below."
        )
    return lines


_RESAMPLED = {
    "vote": "resampling these votes as if they were independent, with these outputs "
    "and raters held fixed",
    "rater": "resampling whole raters, since one rater's votes share that rater's "
    "taste, with these outputs held fixed",
    "task": "resampling whole tasks, since one task's votes share that task, with "
    "these outputs and raters held fixed",
    "rater+task": "resampling raters and tasks together, since a vote shares its "
    "rater's taste and its task, with these outputs held fixed",
}


def _pairwise_tables(
    console: Any, tables: dict[str, Any], suffix: str, *, ranked: bool
) -> None:
    from rich.table import Table

    for name, rows in tables.items():
        table = Table(title=f"Pairwise ({name}){suffix}", title_justify="left")
        for column in ("arm", "rating", "95% interval", "wins", "ties", "losses"):
            table.add_column(column, justify="left" if column == "arm" else "right")
        # Until every pair has a vote the arms are listed by name, unrated.
        for row in rows if ranked else sorted(rows, key=lambda r: r["arm"]):
            interval = row.get("interval")
            table.add_row(
                row["arm"],
                "not yet"
                if not ranked
                else "unranked"
                if row["rating"] is None
                else f"{row['rating']:.0f}",
                f"{interval[0]:.0f} to {interval[1]:.0f}"
                if interval and ranked
                else "-",
                str(row["wins"]),
                str(row["ties"]),
                str(row["losses"]),
            )
        console.print(table)
        for row in rows:
            if ranked and row.get("interval") and row.get("resamples") is not None:
                console.print(
                    f"{row['arm']}: its interval is from the {row['resamples']} "
                    "resamples that drew any of its votes; the others drew none.",
                    highlight=False,
                    soft_wrap=True,
                )


def _rubric_table(console: Any, title: str, rows: list[dict[str, Any]]) -> None:
    from rich.table import Table

    table = Table(title=title, title_justify="left")
    rater = any("rater" in row for row in rows)
    columns = (("rater",) if rater else ()) + (
        "arm",
        "criterion",
        "mean",
        "95% interval",
        "items",
    )
    for column in columns:
        table.add_column(
            column,
            justify="left" if column in ("rater", "arm", "criterion") else "right",
        )
    for row in rows:
        interval = row.get("interval")
        table.add_row(
            *((row["rater"],) if rater else ()),
            row["arm"],
            row["criterion"],
            f"{row['mean']:.2f}",
            f"{interval[0]:.2f} to {interval[1]:.2f}" if interval else "-",
            str(row["items"]),
        )
    console.print(table)


def _print_ratings(result: dict[str, Any], *, provisional: bool = False) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    # The checks come first: they say whether the tables below can be trusted.
    for line in _rater_checks(result):
        console.print(line, highlight=False, soft_wrap=True)
    ranked = provisional or result.get("ranked") is not False
    # What the intervals are, before the numbers they qualify.
    cluster = result.get("cluster") or "vote"
    text = (
        "Ratings: Bradley-Terry on the Elo scale (1000 is a fixed reference; 400 "
        f"points means preferred 10 to 1). Intervals: 95%, from {_RESAMPLED[cluster]} "
        "(pairwise), or the items (rubric, shown from 3 items up); none where every "
        "resample gave the same rating. They say nothing about other outputs of the "
        "same agents."
    )
    effect = result.get("design_effect")
    if effect and effect > 1:
        text += (
            f" Here a vote counts for about 1/{effect:.1f} of an independent one "
            f"(design effect {effect:.1f})."
        )
    console.print(text, highlight=False, soft_wrap=True)
    kinds = (result.get("headline") or {}).get("kinds") or ["person"]
    suffix = "" if kinds == ["person"] else f", {_whose(result)}"
    if not result["pairwise"]:
        console.print("No pairwise votes: no pairwise ratings.", highlight=False)
    _pairwise_tables(console, result["pairwise"], suffix, ranked=ranked)
    if ranked and result.get("head_to_head"):
        table = Table(title=f"Head to head{suffix}", title_justify="left")
        for column in (
            "first",
            "second",
            "votes",
            "first's share",
            "95% interval",
            "smallest gap",
        ):
            table.add_column(
                column, justify="left" if column in ("first", "second") else "right"
            )
        for row in result["head_to_head"]:
            interval = row.get("interval")
            table.add_row(
                *row["arms"],
                str(row["votes"]),
                f"{row['share']:.0%}",
                f"{interval[0]:.0%} to {interval[1]:.0%}" if interval else "-",
                f"±{row['smallest_gap']:.0%}" if interval else "-",
            )
        console.print(table)
        note = result["head_to_head"][0].get("interval_note")
        console.print(
            f"No intervals or smallest gaps: {note}."
            if note
            else "The smallest gap is how far from an even split the first's share "
            "must be for these votes to show it 4 times in 5 (a two-sided test at 95%).",
            highlight=False,
            soft_wrap=True,
        )
    if result["rubric"]:
        _rubric_table(console, "Rubric", result["rubric"])
    for kind, part in (result.get("by_kind") or {}).items():
        _pairwise_tables(
            console, part["pairwise"], f", {_KIND_NOUN[kind][1]}", ranked=True
        )
        if part.get("rubric"):
            _rubric_table(console, f"Rubric, {_KIND_NOUN[kind][1]}", part["rubric"])
    if result.get("judges"):
        base = result.get("people_baseline")
        if base:
            low, high = base["interval"]
            console.print(
                f"One person against the others: {base['agree']} of {base['votes']} "
                f"({base['agreement']:.0%}, 95% {low:.0%} to {high:.0%}); a coin would "
                "agree about half the time.",
                highlight=False,
                soft_wrap=True,
            )
        table = Table(title="Judges against people", title_justify="left")
        for column in (
            "judge",
            "from",
            "pairs",
            "agreement",
            "95% interval",
            "both ways round",
        ):
            table.add_column(
                column, justify="left" if column in ("judge", "from") else "right"
            )
        for row in result["judges"]:
            low, high = row["interval"]
            both = row.get("consistency")
            table.add_row(
                row["rater"],
                "its votes"
                if row["how"] == "pairwise"
                else f"its {row['criterion']} scores",
                str(row["pairs"]),
                f"{row['agree']} ({row['agreement']:.0%})",
                f"{low:.0%} to {high:.0%}",
                f"{both['same']} of {both['pairs']} the same" if both else "-",
            )
        console.print(table)
    if result.get("judge_scores"):
        _rubric_table(
            console, "Rubric scores from model judges", result["judge_scores"]
        )


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
        jobs: Annotated[
            list[Path] | None,
            typer.Argument(
                help="Job or trial folders whose outputs to pair (none with only --items)."
            ),
        ] = None,
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
        items: Annotated[
            Path | None,
            typer.Option(
                "--items",
                help="A JSON-lines file of outputs made outside BenchFlow (a golden, "
                "a remix), each with its task, output folder and arm labels; they "
                "pair with the trials of their task's version.",
            ),
        ] = None,
        condition: Annotated[
            str | None,
            typer.Option(
                "--condition",
                help="A condition to label every trial's output with (one-shot, "
                "say); --items rows carry their own.",
            ),
        ] = None,
        entry: Annotated[
            str | None,
            typer.Option(
                "--entry",
                help="The page to open in each output, such as index.html; an "
                "output without it is left out.",
            ),
        ] = None,
        golden: Annotated[
            str,
            typer.Option(
                "--golden",
                help="Whether a golden from --items is a side in pairs (side) or "
                "only the bar a rater's tool shows (bar).",
            ),
        ] = "side",
    ) -> None:
        """Make the store and the blind assignment."""
        from benchflow.preferences import PreferenceError, issue

        try:
            assignment = issue(
                jobs or [],
                store,
                output=output,
                per_task=per_task,
                seed=seed,
                rubric=rubric,
                bundle=bundle,
                randomize_sides=False if keep_sides else None,
                items=items,
                condition=condition,
                entry=entry,
                golden=golden,
            )
        except (PreferenceError, OSError) as exc:
            _refused(exc)
        out.write_text(json.dumps(assignment, indent=1) + "\n")
        manifest = json.loads((store / "store.v1.json").read_text())
        origins = manifest.get("origins") or {}
        from_where = (
            f" ({_count(origins.get('trial', 0), 'trial')}, "
            f"{origins.get('outside', 0)} from --items)"
            if items is not None
            else ""
        )
        typer.echo(
            f"{len(assignment['pairs'])} pairs of {len(assignment['items'])} "
            f"outputs{from_where}."
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
                help="Votes as JSON lines or CSV, or the FDB vote site's export "
                "(see the docs for the fields).",
            ),
        ],
        source: Annotated[
            str | None,
            typer.Option(
                "--source", help="The tool that collected them, recorded on each vote."
            ),
        ] = None,
        skip_unknown: Annotated[
            bool,
            typer.Option(
                "--skip-unknown",
                help="Leave out votes on outputs the store doesn't have, instead "
                "of refusing the file.",
            ),
        ] = False,
        dry_run: Annotated[
            bool,
            typer.Option(
                "--dry-run",
                help="Check every vote and say what would change; write nothing.",
            ),
        ] = False,
    ) -> None:
        """Check every vote in the file, then add the new ones to the store."""
        from benchflow.preferences import PreferenceError, import_votes

        try:
            counts = import_votes(
                store, votes, source=source, skip_unknown=skip_unknown, dry_run=dry_run
            )
        except (PreferenceError, OSError, ValueError) as exc:
            _refused(exc)
        added, again = counts["added"], counts["already_recorded"]
        if dry_run:
            typer.echo(
                f"Dry run: {_count(added, 'vote')} would be added, {again} already "
                "recorded; nothing was written."
            )
        else:
            typer.echo(
                f"{added} vote{'s' if added != 1 else ''} added, {again} already recorded."
            )
        if counts.get("withdrawn"):
            n = counts["withdrawn"]
            typer.echo(
                f"{_count(n, 'vote')} the export no longer has "
                f"{'would be' if dry_run else 'is' if n == 1 else 'are'} recorded as "
                "withdrawn (an abstention, so it stops counting)."
            )
        if counts.get("skipped_unknown"):
            n = counts["skipped_unknown"]
            typer.echo(
                f"{_count(n, 'vote')} {'names' if n == 1 else 'name'} an output the "
                "store doesn't have, left out (--skip-unknown)."
            )
        if counts.get("head_sha256") and not dry_run:
            typer.echo(
                f"Log head: {counts['head_sha256']} (keep a copy elsewhere: the log "
                "alone cannot show that its last lines were cut)."
            )

    @group.command("rank")
    def rank_cmd(
        store: Store,
        by: Annotated[
            str,
            typer.Option(
                "--by", help="Rate each agent, model, agent+model or condition."
            ),
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
        raters: Annotated[
            str | None,
            typer.Option(
                "--raters",
                help="Whose votes to rate together: person, guest, model, or several "
                "joined by commas (default: people; the others get tables of their "
                "own).",
            ),
        ] = None,
        cluster: Annotated[
            str,
            typer.Option(
                "--cluster",
                help="What the intervals resample: vote, rater, task or rater+task "
                "(auto: raters and tasks, when there are several of each).",
            ),
        ] = "auto",
        provisional: Annotated[
            bool,
            typer.Option(
                "--provisional",
                help="Show the ratings before every pair has a vote.",
            ),
        ] = False,
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
                raters=[k.strip() for k in raters.split(",") if k.strip()]
                if raters is not None
                else None,
                cluster=cluster,
            )
        except (PreferenceError, OSError) as exc:
            _refused(exc)
        if as_json:
            typer.echo(json.dumps(result, indent=1))
        else:
            _print_ratings(result, provisional=provisional)
