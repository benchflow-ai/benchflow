"""Ratings from human votes: Bradley-Terry over pairwise votes, means over
rubric scores, and the checks that say how far to trust them.

Bradley-Terry gives each arm a strength ``s`` such that arm ``a`` is preferred
to arm ``b`` with probability ``s_a / (s_a + s_b)``. It is fitted by Hunter's
MM algorithm (Ann. Statist. 32(1), 2004): a tie counts half a win to each
side, and every arm also holds :data:`PRIOR_TIES` virtual ties against a
reference arm whose strength is fixed at 1, so an arm that won or lost every
vote still gets a finite rating and the scale has a fixed zero. Ratings are on
the Elo scale: ``1000 + 400 log10(s)``, the reference at 1000.

The intervals resample what the votes are not independent of. One rater's
votes share that rater's taste and one task's votes share that task, so a
bootstrap can draw whole raters, whole tasks, or both at once: raters and
tasks drawn apart, a vote counted once for each pairing of its rater's and its
task's draws (Owen's pigeonhole bootstrap, Ann. Appl. Stat. 1(2), 2007).
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

REFERENCE = 1000.0
SCALE = 400.0
PRIOR_TIES = 1.0
_ITERATIONS = 100_000
_TOLERANCE = 1e-10
# What a bootstrap draw resamples: single votes, whole raters, whole tasks,
# or raters and tasks at once.
CLUSTERS = ("vote", "rater", "task", "rater+task")
_Z95 = 1.959964
_Z80 = 0.841621


@dataclass(frozen=True)
class Outcome:
    """One effective pairwise vote between two arms: ``score`` is 1 when the
    left arm was preferred, 0 for the right, 0.5 for a tie."""

    left: str
    right: str
    score: float


def strengths(outcomes: Sequence[Outcome], arms: Iterable[str]) -> dict[str, float]:
    """Each arm's Bradley-Terry strength (the reference's is 1)."""
    names = sorted(set(arms))
    wins = dict.fromkeys(names, PRIOR_TIES / 2)
    games: dict[str, dict[str, float]] = {name: defaultdict(float) for name in names}
    for outcome in outcomes:
        if outcome.left == outcome.right:
            continue
        wins[outcome.left] += outcome.score
        wins[outcome.right] += 1.0 - outcome.score
        games[outcome.left][outcome.right] += 1.0
        games[outcome.right][outcome.left] += 1.0
    strength = dict.fromkeys(names, 1.0)
    for _ in range(_ITERATIONS):
        new = {}
        for name in names:
            denominator = PRIOR_TIES / (strength[name] + 1.0)
            for other, count in games[name].items():
                denominator += count / (strength[name] + strength[other])
            new[name] = wins[name] / denominator
        change = max(
            (abs(math.log(new[n]) - math.log(strength[n])) for n in names), default=0.0
        )
        strength = new
        if change < _TOLERANCE:
            break
    return strength


def elo(strength: float) -> float:
    return REFERENCE + SCALE * math.log10(strength)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def draw(
    clusters: Sequence[tuple[str, str]], method: str, rng: random.Random
) -> tuple[list[int], Counter[str], Counter[str]]:
    """One bootstrap sample of votes, as indices into ``clusters`` (each
    vote's rater and task), with how many times each rater and each task was
    drawn. ``vote`` draws single votes; ``rater`` and ``task`` draw whole
    raters or tasks, each bringing all its votes; ``rater+task`` draws raters
    and tasks apart and repeats a vote once per pairing of its rater's and its
    task's draws."""
    n = len(clusters)
    if method == "vote":
        return [rng.randrange(n) for _ in range(n)], Counter(), Counter()
    raters = sorted({c[0] for c in clusters})
    tasks = sorted({c[1] for c in clusters})
    if method == "rater":
        drawn_raters = Counter(raters[rng.randrange(len(raters))] for _ in raters)
        drawn_tasks = Counter(dict.fromkeys(tasks, 1))
    elif method == "task":
        drawn_raters = Counter(dict.fromkeys(raters, 1))
        drawn_tasks = Counter(tasks[rng.randrange(len(tasks))] for _ in tasks)
    elif method == "rater+task":
        drawn_raters = Counter(raters[rng.randrange(len(raters))] for _ in raters)
        drawn_tasks = Counter(tasks[rng.randrange(len(tasks))] for _ in tasks)
    else:
        raise ValueError(
            f"cluster must be one of {', '.join(CLUSTERS)}, not {method!r}"
        )
    indices = [
        i
        for i, (rater, task) in enumerate(clusters)
        for _ in range(drawn_raters[rater] * drawn_tasks[task])
    ]
    return indices, drawn_raters, drawn_tasks


def _interval(samples: list[float]) -> list[float]:
    return [
        round(_percentile(samples, 0.025), 1),
        round(_percentile(samples, 0.975), 1),
    ]


def pairwise_table(
    outcomes: Sequence[Outcome],
    arms: Iterable[str],
    *,
    bootstrap: int,
    seed: int,
    clusters: Sequence[tuple[str, str]] | None = None,
    cluster: str = "vote",
) -> list[dict[str, Any]]:
    """One row per arm, best first: rating, 95% interval over ``bootstrap``
    resamples (none when 0), and its wins, ties and losses.

    ``cluster`` says what a resample draws (:func:`draw`), from ``clusters``,
    one (rater, task) per outcome. With anything but ``vote``, each row also
    has ``interval_votes``, the interval from redrawing single votes, so the
    two can be compared. An interval every resample collapses to one value
    (all the votes went one way) is no interval: the row says so instead. A
    resample that drew none of an arm's votes is left out of that arm's
    interval, and ``resamples`` then says how many counted."""
    if cluster not in CLUSTERS:
        raise ValueError(
            f"cluster must be one of {', '.join(CLUSTERS)}, not {cluster!r}"
        )
    if cluster != "vote" and (clusters is None or len(clusters) != len(outcomes)):
        raise ValueError("a clustered bootstrap needs one (rater, task) per outcome")
    names = sorted(set(arms))
    rating = {name: elo(s) for name, s in strengths(outcomes, names).items()}
    record = {name: {"wins": 0, "ties": 0, "losses": 0} for name in names}
    for outcome in outcomes:
        if outcome.score == 0.5:
            record[outcome.left]["ties"] += 1
            record[outcome.right]["ties"] += 1
        else:
            winner, loser = (
                (outcome.left, outcome.right)
                if outcome.score == 1.0
                else (outcome.right, outcome.left)
            )
            record[winner]["wins"] += 1
            record[loser]["losses"] += 1
    samples: dict[str, list[float]] = {name: [] for name in names}
    by_vote: dict[str, list[float]] = {name: [] for name in names}

    def resampled(drawn: list[Outcome], into: dict[str, list[float]]) -> None:
        # A resample that drew none of an arm's votes says nothing about it:
        # its strength there would be the prior's, the reference's 1000.
        present = {o.left for o in drawn} | {o.right for o in drawn}
        for name, s in strengths(drawn, names).items():
            if name in present:
                into[name].append(elo(s))

    if bootstrap and outcomes:
        # Redrawing single votes, with the draws 0.10.0rc2 made.
        rng = random.Random(seed)
        for _ in range(bootstrap):
            resampled([rng.choice(outcomes) for _ in outcomes], by_vote)
        if cluster == "vote":
            samples = by_vote
        else:
            assert clusters is not None
            rng = random.Random(seed)
            for _ in range(bootstrap):
                indices, _raters, _tasks = draw(clusters, cluster, rng)
                resampled([outcomes[i] for i in indices], samples)
    rows = []
    for name in names:
        votes = sum(record[name].values())
        row: dict[str, Any] = {
            "arm": name,
            # An arm with no comparison has only the prior: no rating to show.
            "rating": round(rating[name], 1) if votes else None,
            **record[name],
            "votes": votes,
        }
        if samples[name] and votes:
            interval = _interval(samples[name])
            if interval[0] == interval[1]:
                row["interval_note"] = "every resample gave the same rating"
            else:
                row["interval"] = interval
            if len(samples[name]) < bootstrap:
                row["resamples"] = len(samples[name])
            if cluster != "vote" and by_vote[name]:
                row["interval_votes"] = _interval(by_vote[name])
        rows.append(row)
    rows.sort(key=lambda r: (r["rating"] is None, -(r["rating"] or 0), r["arm"]))
    return rows


def rubric_table(
    scores: dict[tuple[str, str], dict[str, list[float]]],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """``scores[(arm, criterion)][item]`` holds each rater's score of one
    item. An item's score is its raters' mean, and an arm's is the mean over
    its items, each item counted once; the interval resamples items."""
    rows = []
    rng = random.Random(seed)
    for (arm, criterion), by_item in sorted(scores.items()):
        means = [sum(v) / len(v) for _, v in sorted(by_item.items())]
        row: dict[str, Any] = {
            "arm": arm,
            "criterion": criterion,
            "mean": round(sum(means) / len(means), 4),
            "items": len(means),
            "scores": sum(len(v) for v in by_item.values()),
        }
        # Fewer than three items give no interval worth printing.
        if bootstrap and len(means) >= 3:
            draws = [
                sum(rng.choice(means) for _ in means) / len(means)
                for _ in range(bootstrap)
            ]
            row["interval"] = [
                round(_percentile(draws, 0.025), 4),
                round(_percentile(draws, 0.975), 4),
            ]
        rows.append(row)
    return rows


def agreement(choices: dict[Any, dict[str, str]]) -> dict[str, Any]:
    """Share of rater pairs that gave the same answer, over the targets two
    or more raters judged (``choices[target][rater]`` is left, right or tie),
    and the share chance alone would give from each rater's own choice
    rates, with Cohen's kappa against it."""
    rates: dict[str, dict[str, float]] = {}
    by_rater_choices: dict[str, list[str]] = defaultdict(list)
    for by_rater in choices.values():
        for rater, choice in by_rater.items():
            by_rater_choices[rater].append(choice)
    for rater, made in by_rater_choices.items():
        rates[rater] = {c: made.count(c) / len(made) for c in set(made)}
    agreeing = total = targets = 0
    chance = 0.0
    for by_rater in choices.values():
        raters = list(by_rater.items())
        if len(raters) < 2:
            continue
        targets += 1
        for i in range(len(raters)):
            for j in range(i + 1, len(raters)):
                (r1, a1), (r2, a2) = raters[i], raters[j]
                total += 1
                agreeing += a1 == a2
                chance += sum(p * rates[r2].get(c, 0.0) for c, p in rates[r1].items())
    observed = agreeing / total if total else None
    expected = chance / total if total else None
    kappa = (
        (observed - expected) / (1 - expected)
        if observed is not None and expected is not None and expected < 1
        else None
    )
    return {
        "targets": targets,
        "rater_pairs": total,
        "agreeing": agreeing,
        "agreement": round(observed, 4) if observed is not None else None,
        "chance_agreement": round(expected, 4) if expected is not None else None,
        "kappa": round(kappa, 4) if kappa is not None else None,
    }


def left_share(choices: Iterable[str]) -> dict[str, Any]:
    """How often a decisive vote chose the side shown on the left; with
    sides shuffled it should sit near one half (``z`` against that)."""
    decisive = [c for c in choices if c in ("left", "right")]
    n = len(decisive)
    left = sum(c == "left" for c in decisive)
    return {
        "decisive": n,
        "left": left,
        "share": round(left / n, 4) if n else None,
        "z": round((left - n / 2) / math.sqrt(n / 4), 2) if n else None,
    }


def wilson(successes: float, n: float, z: float = _Z95) -> tuple[float, float]:
    """Wilson's 95% interval for a share of ``successes`` in ``n`` (``n`` may
    be an effective count, not a whole number)."""
    if n <= 0:
        raise ValueError("a share needs at least one vote")
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def smallest_gap(n: float) -> float:
    """How far from one half a share of ``n`` independent votes must be for a
    two-sided test at 95% to find it 80% of the time."""
    return (_Z95 + _Z80) * 0.5 / math.sqrt(n)


def head_to_head(
    outcomes: Sequence[Outcome], *, design_effect: float | None = 1.0
) -> list[dict[str, Any]]:
    """Each pair of arms that met: the votes between them, the share the
    first (by name) won with ties as halves, its Wilson interval and the
    smallest gap from one half these votes could show, both on the votes
    divided by the design effect (``effective_votes``). With no design effect
    (clustered votes it could not be measured on) a row has neither, and its
    ``interval_note`` says why."""
    met: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for outcome in outcomes:
        if outcome.left == outcome.right:
            continue
        first, second = sorted((outcome.left, outcome.right))
        points = outcome.score if outcome.left == first else 1.0 - outcome.score
        met[(first, second)][0] += points
        met[(first, second)][1] += 1
    rows = []
    for (first, second), (points, votes) in sorted(met.items()):
        row: dict[str, Any] = {
            "arms": [first, second],
            "votes": int(votes),
            "share": round(points / votes, 4),
        }
        if design_effect is None:
            row["interval_note"] = (
                "these votes share raters or tasks, and the design effect"
                " could not be measured"
            )
        else:
            effective = votes / max(1.0, design_effect)
            low, high = wilson(points / votes * effective, effective)
            row["interval"] = [round(low, 4), round(high, 4)]
            row["effective_votes"] = round(effective, 1)
            row["smallest_gap"] = round(smallest_gap(effective), 4)
        rows.append(row)
    return rows
