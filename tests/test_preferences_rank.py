"""Guards how `bench preferences rank` weighs votes: intervals that
resample raters and tasks instead of treating every vote as independent, model
judges and guests rated apart from people, each judge's agreement with people
beside one person's, ranks withheld until every pair has a vote, and the
smallest difference the votes could show. The gap it closes: on 0.10.0rc2 a
model judge imported as a rater was fitted like a person, agreement pooled
every pair of raters, and the bootstrap redrew single votes."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.preferences import import_votes, issue, rank, ratings
from benchflow.preferences import store as store_module

DIGEST = "sha256:" + "1" * 64
LATER = "2099-01-01T00:00:00Z"
ARMS = {"a": ("openhands", "m1"), "b": ("pi-acp", "m1"), "c": ("openhands", "m2")}


def _job(root: Path, *, tasks=("landing",)) -> Path:
    job = root / "job"
    for task in tasks:
        for arm, (agent, model) in ARMS.items():
            trial = job / f"{task}__{arm}"
            (trial / "artifacts").mkdir(parents=True)
            (trial / "artifacts" / "index.html").write_text(f"<h1>{task} {arm}</h1>\n")
            (trial / "config.json").write_text("{}")
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": task,
                        "task_digest": DIGEST,
                        "agent_name": agent,
                        "model": model,
                        "finished_at": "2026-11-04 01:00:00",
                        "rewards": {"reward": 1.0},
                    }
                )
            )
    return job


def _store(tmp_path: Path, **kwargs) -> tuple[Path, dict]:
    store = tmp_path / "store"
    assignment = issue([_job(tmp_path, **kwargs)], store, seed=11)
    return store, assignment


def _arm_letter(store: Path, item_id: str) -> str:
    items = {
        row["item_id"]: row
        for row in map(
            json.loads, (store / store_module.ITEMS).read_text().splitlines()
        )
    }
    return Path(items[item_id]["trial"]).name.split("__")[1]


def _vote_for(store: Path, pair: dict, arm: str, rater: str, **extra) -> dict:
    left = _arm_letter(store, pair["left"])
    right = _arm_letter(store, pair["right"])
    choice = "left" if left == arm else "right" if right == arm else "tie"
    return {
        "pair_id": pair["pair_id"],
        "choice": choice,
        "rater": rater,
        "rated_at": LATER,
        **extra,
    }


def _import(store: Path, tmp_path: Path, rows: list[dict], name: str = "v") -> None:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    import_votes(store, path)


def _arms(result: dict, table: str = "overall") -> list[str]:
    return [row["arm"] for row in result["pairwise"][table]]


# --- intervals that don't assume independent votes ---------------------------


def _dominated(seed: int = 4) -> tuple[list[ratings.Outcome], list[tuple[str, str]]]:
    """Nine raters split evenly; a tenth casts most of the votes, all for x."""
    rng = random.Random(seed)
    outcomes, clusters = [], []
    for _ in range(120):
        outcomes.append(ratings.Outcome("x", "y", 1.0))
        clusters.append(("loud", "t0"))
    for r in range(9):
        for k in range(8):
            outcomes.append(
                ratings.Outcome("x", "y", 1.0 if rng.random() < 0.5 else 0.0)
            )
            clusters.append((f"r{r}", f"t{k % 4}"))
    return outcomes, clusters


def _width(row: dict, key: str = "interval") -> float:
    low, high = row[key]
    return high - low


def test_resampling_raters_widens_the_interval_one_rater_dominates():
    outcomes, clusters = _dominated()
    votes = ratings.pairwise_table(outcomes, "xy", bootstrap=300, seed=1)
    raters = ratings.pairwise_table(
        outcomes, "xy", bootstrap=300, seed=1, clusters=clusters, cluster="rater"
    )
    x_votes = next(r for r in votes if r["arm"] == "x")
    x_raters = next(r for r in raters if r["arm"] == "x")
    assert _width(x_raters) > 2 * _width(x_votes)
    # Beside the clustered interval, the one that redraws single votes.
    assert x_raters["interval_votes"] == x_votes["interval"]


@pytest.mark.parametrize("cluster", ["rater", "task", "rater+task"])
def test_a_clustered_bootstrap_is_seeded(cluster):
    outcomes, clusters = _dominated()
    first = ratings.pairwise_table(
        outcomes, "xy", bootstrap=40, seed=3, clusters=clusters, cluster=cluster
    )
    again = ratings.pairwise_table(
        outcomes, "xy", bootstrap=40, seed=3, clusters=clusters, cluster=cluster
    )
    assert first == again


def test_the_pigeonhole_draw_repeats_a_vote_once_per_pairing_of_its_draws():
    clusters = [("r1", "t1"), ("r1", "t2"), ("r2", "t1")]
    rng = random.Random(0)
    drawn, raters, tasks = ratings.draw(clusters, "rater+task", rng)
    for index, (rater, task) in enumerate(clusters):
        assert drawn.count(index) == raters[rater] * tasks[task]
    assert sum(raters.values()) == 2 and sum(tasks.values()) == 2


def test_a_vote_bootstrap_is_unchanged():
    """Guards the intervals 0.10.0rc2 printed: with no clusters the bootstrap
    redraws single votes exactly as before."""
    outcomes = [
        ratings.Outcome("x", "y", s) for s in (1.0, 0.0, 1.0, 0.5, 1.0, 1.0, 0.0)
    ]
    rows = ratings.pairwise_table(outcomes, "xy", bootstrap=50, seed=2)
    rng = random.Random(2)
    samples = []
    for _ in range(50):
        drawn = [rng.choice(outcomes) for _ in outcomes]
        samples.append(ratings.elo(ratings.strengths(drawn, "xy")["x"]))
    x = next(r for r in rows if r["arm"] == "x")
    assert x["interval"] == [
        round(ratings._percentile(samples, 0.025), 1),
        round(ratings._percentile(samples, 0.975), 1),
    ]


@pytest.mark.parametrize("cluster", ["vote", "task"])
def test_a_resample_without_an_arm_s_votes_leaves_the_arm_out(cluster):
    """An arm compared in one task of three is missing from about three task
    resamples in ten, and an arm with one vote of 41 from about four vote
    resamples in ten. Those resamples say nothing about it, so they don't pull
    its interval to the reference's 1000; the row says how many counted."""
    outcomes, clusters = [], []
    rng = random.Random(4)
    for task in ("t1", "t2", "t3"):
        for r in range(4):
            for _ in range(3):
                outcomes.append(
                    ratings.Outcome("x", "y", 1.0 if rng.random() < 0.5 else 0.0)
                )
                clusters.append((f"r{r}", task))
    beats = 1 if cluster == "vote" else 4
    for r in range(beats):
        outcomes.append(ratings.Outcome("z", "x", 1.0))
        clusters.append((f"r{r}", "t1"))
    rows = ratings.pairwise_table(
        outcomes, "xyz", bootstrap=300, seed=5, clusters=clusters, cluster=cluster
    )
    z = next(r for r in rows if r["arm"] == "z")
    assert 150 < z["resamples"] < 260
    assert z["interval"][0] > 1050
    # An arm in every resample keeps its row as it was.
    assert "resamples" not in next(r for r in rows if r["arm"] == "y")


def test_votes_that_all_went_one_way_get_no_interval():
    outcomes = [ratings.Outcome("x", "y", 1.0)] * 8
    rows = ratings.pairwise_table(outcomes, "xy", bootstrap=50, seed=1)
    for row in rows:
        assert "interval" not in row
        assert row["interval_note"] == "every resample gave the same rating"


def test_rank_picks_the_clusters_the_votes_have(tmp_path):
    store, assignment = _store(tmp_path, tasks=("landing", "pricing"))
    one_rater = [_vote_for(store, p, "a", "r1") for p in assignment["pairs"]]
    _import(store, tmp_path, one_rater, "one")
    assert rank(store, bootstrap=10)["cluster"] == "task"
    two = [_vote_for(store, p, "b", "r2") for p in assignment["pairs"]]
    _import(store, tmp_path, two, "two")
    result = rank(store, bootstrap=10)
    assert result["cluster"] == "rater+task"
    assert rank(store, bootstrap=10, cluster="vote")["cluster"] == "vote"
    with pytest.raises(store_module.PreferenceError, match="cluster"):
        rank(store, cluster="judges")


# --- model judges and guests apart from people -------------------------------


def test_model_judges_are_rated_apart_from_people(tmp_path):
    store, assignment = _store(tmp_path)
    people = [
        _vote_for(store, p, "a", rater)
        for p in assignment["pairs"]
        for rater in ("ann", "bo")
    ]
    judges = [
        _vote_for(store, p, "c", judge, rater_kind="model")
        for p in assignment["pairs"]
        for judge in ("judge-1", "judge-2", "judge-3")
    ]
    # A judge from an older votes file says only "model:" in its name.
    legacy = [_vote_for(store, p, "c", "model:old-judge") for p in assignment["pairs"]]
    _import(store, tmp_path, people + judges + legacy)
    result = rank(store, bootstrap=0)
    assert result["headline"]["kinds"] == ["person"]
    assert _arms(result)[0] == "openhands / m1"  # arm a, the people's choice
    assert result["raters"] == {"ann": 3, "bo": 3}
    models = result["by_kind"]["model"]
    assert models["raters"] == 4
    assert models["pairwise"]["overall"][0]["arm"] == "openhands / m2"  # arm c
    # Asked for, every kind is fitted together.
    pooled = rank(store, bootstrap=0, raters=["person", "model"])
    assert pooled["effective_votes"] == len(people + judges + legacy)


def test_guests_are_rated_in_a_table_of_their_own(tmp_path):
    store, assignment = _store(tmp_path)
    rows = [_vote_for(store, p, "a", "ann") for p in assignment["pairs"]] + [
        _vote_for(store, p, "b", "g-1", rater_kind="guest") for p in assignment["pairs"]
    ]
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0)
    assert result["raters"] == {"ann": 3}
    assert result["by_kind"]["guest"]["pairwise"]["overall"][0]["arm"] == "pi-acp / m1"


def test_with_no_person_voting_the_ratings_say_whose_they_are(tmp_path):
    from benchflow.cli.preferences import _rater_checks

    store, assignment = _store(tmp_path)
    rows = [
        _vote_for(store, p, "a", "model:mimo", rater_kind="model")
        for p in assignment["pairs"]
    ]
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0)
    assert result["headline"]["kinds"] == ["model"]
    assert result["pairwise"]["overall"]
    text = " ".join(_rater_checks(result))
    assert "No person voted: the ratings below are the model judges'." in text


# --- each judge against people ----------------------------------------------


def _three_people_and_a_judge(store: Path, assignment: dict) -> tuple[list[dict], dict]:
    """People prefer a over b over c on every pair; the judge agrees on the
    pairs with a, and picks c over b."""
    rows, truth = [], {}
    for pair in assignment["pairs"]:
        arms = {_arm_letter(store, pair["left"]), _arm_letter(store, pair["right"])}
        best = min(arms)
        truth[pair["pair_id"]] = best
        for rater in ("ann", "bo", "cy"):
            rows.append(_vote_for(store, pair, best, rater))
        judge_pick = best if "a" in arms else "c"
        rows.append(
            _vote_for(store, pair, judge_pick, "model:judge", rater_kind="model")
        )
    return rows, truth


def test_a_judge_s_agreement_with_people_is_counted_against_one_person_s(tmp_path):
    store, assignment = _store(tmp_path)
    rows, _ = _three_people_and_a_judge(store, assignment)
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0)
    (judge,) = result["judges"]
    assert (judge["rater"], judge["how"]) == ("model:judge", "pairwise")
    assert (judge["pairs"], judge["agree"]) == (3, 2)
    low, high = ratings.wilson(2, 3)
    assert judge["interval"] == [round(low, 4), round(high, 4)]
    # Each person agrees with the other two on every pair.
    baseline = result["people_baseline"]
    assert (baseline["votes"], baseline["agree"]) == (9, 9)
    # Agreement on the pairs each arm is in: the judge misses on b-vs-c.
    by_arm = judge["by_arm"]
    assert by_arm["pi-acp / m1"] == {"pairs": 2, "agree": 1}
    assert by_arm["openhands / m1"] == {"pairs": 2, "agree": 2}


def test_a_judge_asked_both_ways_round_keeps_both_answers(tmp_path):
    store, assignment = _store(tmp_path)
    rows, _ = _three_people_and_a_judge(store, assignment)
    pair = assignment["pairs"][0]
    # The judge sees the first pair the other way round and changes its mind.
    first_pick = next(
        r["choice"]
        for r in rows
        if r["rater"] == "model:judge" and r["pair_id"] == pair["pair_id"]
    )
    _import(store, tmp_path, rows)
    (before,) = rank(store, bootstrap=0)["judges"]
    rows.append(
        {
            "left_item": pair["right"],
            "right_item": pair["left"],
            "choice": first_pick,  # the same side, so the other output
            "rater": "model:judge",
            "rater_kind": "model",
            "rated_at": "2099-01-02T00:00:00Z",
        }
    )
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0)
    (judge,) = result["judges"]
    assert judge["consistency"] == {"pairs": 1, "same": 0}
    # A pair seen both ways is one pair, scored by the share of its looks
    # that agree: here one of two, so the judge's count moves by a half.
    assert judge["pairs"] == before["pairs"] == 3
    assert abs(judge["agree"] - before["agree"]) == 0.5


def test_scores_a_judge_gave_imply_its_preference_on_each_pair(tmp_path):
    store = tmp_path / "store"
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "id": "r",
                "version": "1",
                "criteria": [{"id": "overall", "scale": [0, 1]}],
            }
        )
    )
    assignment = issue([_job(tmp_path)], store, seed=11, rubric=rubric)
    rows, _ = _three_people_and_a_judge(store, assignment)
    rows = [r for r in rows if r["rater"] != "model:judge"]
    score = {"a": 0.9, "b": 0.2, "c": 0.5}  # agrees on a, not on b-vs-c
    for item in assignment["items"]:
        rows.append(
            {
                "item_id": item["item_id"],
                "criterion": "overall",
                "score": score[_arm_letter(store, item["item_id"])],
                "rater": "rubric-v2",
                "rater_kind": "model",
                "rated_at": LATER,
            }
        )
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0)
    (judge,) = result["judges"]
    assert (judge["rater"], judge["how"], judge["criterion"]) == (
        "rubric-v2",
        "scores",
        "overall",
    )
    assert (judge["pairs"], judge["agree"]) == (3, 2)
    # The judge's scores are not averaged into the people's rubric table.
    assert result["rubric"] == []
    assert {row["rater"] for row in result["judge_scores"]} == {"rubric-v2"}
    # Asked to rate judges with people, their scores are the table's too.
    together = rank(store, bootstrap=0, raters=["person", "model"])
    assert together["rubric"] and together["judge_scores"] == []


def test_guests_scores_are_a_rubric_table_of_their_own(tmp_path):
    store = tmp_path / "store"
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "id": "r",
                "version": "1",
                "criteria": [{"id": "overall", "scale": [0, 1]}],
            }
        )
    )
    assignment = issue([_job(tmp_path)], store, seed=11, rubric=rubric)
    rows = [
        {
            "item_id": item["item_id"],
            "criterion": "overall",
            "score": 0.5,
            "rater": rater,
            "rater_kind": kind,
            "rated_at": LATER,
        }
        for item in assignment["items"]
        for rater, kind in (("p-1", "person"), ("g-1", "guest"))
    ]
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0)
    assert result["rubric"] and result["by_kind"]["guest"]["rubric"]


# --- ranks wait for every pair; the smallest difference ---------------------


def test_ranks_wait_until_every_pair_has_a_vote(tmp_path):
    from benchflow.cli.preferences import _rater_checks

    store, assignment = _store(tmp_path)
    first_two = [_vote_for(store, p, "a", "ann") for p in assignment["pairs"][:2]]
    _import(store, tmp_path, first_two, "first")
    result = rank(store, bootstrap=0)
    assert result["coverage"] == {"raters_per_pair": 1, "pairs": 3, "covered": 2}
    assert result["ranked"] is False
    text = " ".join(_rater_checks(result))
    assert "Not ranked yet: 1 of 3 pairs has no vote from people" in text
    last = [_vote_for(store, assignment["pairs"][2], "a", "ann")]
    _import(store, tmp_path, last, "last")
    assert rank(store, bootstrap=0)["ranked"] is True


def test_head_to_head_gives_the_smallest_difference_the_votes_could_show(tmp_path):
    store, assignment = _store(tmp_path)
    rows = [
        _vote_for(store, p, "a", rater)
        for p in assignment["pairs"]
        for rater in ("ann", "bo", "cy", "dee")
    ]
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0, cluster="vote")
    assert result["design_effect"] == 1.0
    ab = next(
        row
        for row in result["head_to_head"]
        if set(row["arms"]) == {"openhands / m1", "pi-acp / m1"}
    )
    assert ab["votes"] == 4
    gap = (1.959964 + 0.841621) * 0.5 / math.sqrt(4)
    assert ab["smallest_gap"] == round(gap, 4)
    assert ab["share"] == (1.0 if ab["arms"][0] == "openhands / m1" else 0.0)


def test_the_design_effect_shrinks_the_votes_that_count(tmp_path):
    store, assignment = _store(tmp_path, tasks=("landing", "pricing", "faq", "about"))
    rng = random.Random(5)
    rows = []
    for rater in ("ann", "bo", "cy", "dee", "eve"):
        lean = rng.choice("abc")
        for pair in assignment["pairs"]:
            rows.append(
                _vote_for(store, pair, lean if rng.random() < 0.8 else "x", rater)
            )
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=200)
    assert result["cluster"] == "rater+task"
    assert result["design_effect"] >= 1.0
    for row in result["head_to_head"]:
        assert row["effective_votes"] == pytest.approx(
            row["votes"] / result["design_effect"], abs=0.1
        )


def test_head_to_head_has_no_interval_when_the_design_effect_is_unknown(tmp_path):
    """Guards the fix for an unknown design effect in the head-to-head table:
    with votes that share raters
    or tasks and no measured design effect, its intervals and smallest gaps
    were computed as if every vote were independent, and printed with no
    caveat. The effect is unknown with --bootstrap 0, and when no arm gets
    both a clustered and a single-vote interval (here every vote goes one
    way, so every resample gives one rating)."""
    from benchflow.cli.main import app

    store, assignment = _store(tmp_path, tasks=("landing", "pricing"))
    rows = [
        _vote_for(store, p, "a", rater)
        for p in assignment["pairs"]
        for rater in ("ann", "bo", "cy")
    ]
    _import(store, tmp_path, rows)
    result = rank(store, bootstrap=0)
    assert (result["cluster"], result["design_effect"]) == ("rater+task", None)
    assert result["head_to_head"]
    for row in result["head_to_head"]:
        assert "interval" not in row and "smallest_gap" not in row
        assert "design effect" in row["interval_note"]
    resampled = rank(store, bootstrap=50)
    assert (resampled["cluster"], resampled["design_effect"]) == ("rater+task", None)
    assert all("interval" not in row for row in resampled["head_to_head"])
    shown = CliRunner().invoke(
        app, ["preferences", "rank", str(store), "--bootstrap", "0"]
    )
    assert shown.exit_code == 0, shown.output
    text = " ".join(shown.output.split())
    assert "No intervals or smallest gaps" in text
    assert "The smallest gap is how far" not in text


# --- the command line ---------------------------------------------------------


def test_the_cli_takes_raters_cluster_and_provisional(tmp_path):
    from benchflow.cli.main import app

    store, assignment = _store(tmp_path)
    rows = [_vote_for(store, p, "a", "ann") for p in assignment["pairs"][:2]] + [
        _vote_for(store, p, "c", "model:j", rater_kind="model")
        for p in assignment["pairs"]
    ]
    _import(store, tmp_path, rows)
    runner = CliRunner()
    shown = runner.invoke(app, ["preferences", "rank", str(store), "--bootstrap", "20"])
    assert shown.exit_code == 0, shown.output
    text = " ".join(shown.output.split())
    assert "Not ranked yet" in text
    assert "Rated apart: 3 votes from 1 model judge" in text
    assert "Judges against people" in text
    early = runner.invoke(
        app,
        [
            "preferences",
            "rank",
            str(store),
            "--provisional",
            "--cluster",
            "vote",
            "--json",
        ],
    )
    assert early.exit_code == 0, early.output
    data = json.loads(early.output)
    assert data["cluster"] == "vote" and data["ranked"] is False
    pooled = runner.invoke(
        app,
        ["preferences", "rank", str(store), "--raters", "person,model", "--json"],
    )
    assert json.loads(pooled.output)["headline"]["kinds"] == ["person", "model"]
