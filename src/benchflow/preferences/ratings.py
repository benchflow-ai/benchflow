"""Ratings from human votes: Bradley-Terry over pairwise votes, means over
rubric scores, and the checks that say how far to trust them.

Bradley-Terry gives each arm a strength ``s`` such that arm ``a`` is preferred
to arm ``b`` with probability ``s_a / (s_a + s_b)``. It is fitted by Hunter's
MM algorithm (Ann. Statist. 32(1), 2004): a tie counts half a win to each
side, and every arm also holds :data:`PRIOR_TIES` virtual ties against a
reference arm whose strength is fixed at 1, so an arm that won or lost every
vote still gets a finite rating and the scale has a fixed zero. Ratings are on
the Elo scale: ``1000 + 400 log10(s)``, the reference at 1000.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

REFERENCE = 1000.0
SCALE = 400.0
PRIOR_TIES = 1.0
_ITERATIONS = 100_000
_TOLERANCE = 1e-10


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


def pairwise_table(
    outcomes: Sequence[Outcome],
    arms: Iterable[str],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """One row per arm, best first: rating, 95% interval over votes
    resampled ``bootstrap`` times (none when 0), and its wins, ties and
    losses."""
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
    if bootstrap and outcomes:
        rng = random.Random(seed)
        for _ in range(bootstrap):
            drawn = [rng.choice(outcomes) for _ in outcomes]
            for name, s in strengths(drawn, names).items():
                samples[name].append(elo(s))
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
            row["interval"] = [
                round(_percentile(samples[name], 0.025), 1),
                round(_percentile(samples[name], 0.975), 1),
            ]
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
