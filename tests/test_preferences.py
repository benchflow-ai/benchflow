"""Guards human preferences (benchflow.preferences): blind pairs of
finished trials, votes from any rater's tool, and Bradley-Terry ratings. The
gap it closes: task.md's [preference] was refused and the only preference data
BenchFlow held was one machine-checked pair at a time.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.preferences import (
    Conflict,
    PreferenceError,
    import_votes,
    issue,
    rank,
    ratings,
)
from benchflow.preferences import store as store_module

ARMS = {"a": ("openhands", "m1"), "b": ("pi-acp", "m1"), "c": ("openhands", "m2")}


def _job(root: Path, *, per_arm: int = 2, task: str = "landing") -> Path:
    job = root / "job"
    for arm, (agent, model) in ARMS.items():
        for n in range(per_arm):
            trial = job / f"{task}__{arm}{n}"
            (trial / "artifacts").mkdir(parents=True)
            (trial / "artifacts" / "index.html").write_text(f"<h1>{arm}{n}</h1>\n")
            (trial / "config.json").write_text("{}")
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": task,
                        "task_digest": "sha256:" + "1" * 64,
                        "agent_name": agent,
                        "model": model,
                        "finished_at": "2026-11-04 01:00:00",
                        "rewards": {"reward": 1.0},
                    }
                )
            )
    return job


def _issue(tmp_path: Path, **kwargs) -> tuple[Path, dict]:
    store = tmp_path / "store"
    assignment = issue([_job(tmp_path)], store, seed=7, **kwargs)
    return store, assignment


def _arm_of(store: Path, item_id: str) -> str:
    items = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in (store / store_module.ITEMS).read_text().splitlines()
    }
    return Path(items[item_id]["trial"]).name.split("__")[1][0]


def _votes(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


LATER = "2099-01-01T00:00:00Z"


# --- ratings --------------------------------------------------------------


def _random_outcomes(seed: int, n: int = 400) -> list[ratings.Outcome]:
    rng = random.Random(seed)
    arms = ["x", "y", "z", "w"]
    out = []
    for _ in range(n):
        left, right = rng.sample(arms, 2)
        out.append(ratings.Outcome(left, right, rng.choice([1.0, 0.0, 0.5, 1.0])))
    return out


def test_the_fit_solves_the_bradley_terry_likelihood_equations():
    # At the maximum, each arm's observed wins (ties as halves, plus its
    # virtual half win) equal the wins the model expects of it.
    outcomes = _random_outcomes(3)
    s = ratings.strengths(outcomes, ["x", "y", "z", "w"])
    for arm in s:
        observed = ratings.PRIOR_TIES / 2
        expected = ratings.PRIOR_TIES * s[arm] / (s[arm] + 1.0)
        for o in outcomes:
            if arm == o.left:
                observed += o.score
                expected += s[arm] / (s[arm] + s[o.right])
            elif arm == o.right:
                observed += 1.0 - o.score
                expected += s[arm] / (s[arm] + s[o.left])
        assert math.isclose(observed, expected, rel_tol=1e-7)


def test_known_strengths_are_recovered_with_intervals_around_them():
    true = {"strong": 1200.0, "mid": 1000.0, "weak": 800.0}
    strength = {k: 10 ** ((v - 1000) / 400) for k, v in true.items()}
    rng = random.Random(11)
    outcomes = []
    for _ in range(3000):
        a, b = rng.sample(sorted(true), 2)
        p = strength[a] / (strength[a] + strength[b])
        outcomes.append(ratings.Outcome(a, b, 1.0 if rng.random() < p else 0.0))
    rows = ratings.pairwise_table(outcomes, true, bootstrap=200, seed=1)
    assert [r["arm"] for r in rows] == ["strong", "mid", "weak"]
    for row in rows:
        assert abs(row["rating"] - true[row["arm"]]) < 40
        low, high = row["interval"]
        # The interval brackets the estimate, and the truth is within 1.5 of
        # its half-widths (the prior pulls each rating a little toward 1000).
        assert low < row["rating"] < high
        assert abs(row["rating"] - true[row["arm"]]) < 1.5 * (high - low) / 2


def test_an_arm_that_won_every_vote_gets_a_finite_rating():
    outcomes = [ratings.Outcome("a", "b", 1.0)] * 5
    s = ratings.strengths(outcomes, ["a", "b"])
    assert math.isfinite(ratings.elo(s["a"])) and s["a"] > 1 > s["b"]


def test_only_ties_leave_equal_arms_at_the_reference():
    s = ratings.strengths([ratings.Outcome("a", "b", 0.5)] * 4, ["a", "b"])
    assert math.isclose(s["a"], s["b"]) and math.isclose(ratings.elo(s["a"]), 1000.0)


def test_the_bootstrap_is_seeded():
    outcomes = _random_outcomes(5, 60)
    first = ratings.pairwise_table(outcomes, "xyzw", bootstrap=50, seed=2)
    assert first == ratings.pairwise_table(outcomes, "xyzw", bootstrap=50, seed=2)


# --- the store ------------------------------------------------------------


def test_pairs_join_different_arms_of_one_task_version_and_hide_them(tmp_path):
    store, assignment = _issue(tmp_path, bundle=tmp_path / "bundle")
    # 6 trials, 3 arms of 2: every pair of trials from different arms.
    assert len(assignment["items"]) == 6 and len(assignment["pairs"]) == 12
    for pair in assignment["pairs"]:
        assert _arm_of(store, pair["left"]) != _arm_of(store, pair["right"])
    shown = json.dumps(assignment)
    for agent, model in ARMS.values():
        assert agent not in shown and model not in shown
    assert "landing__" not in shown  # no trial names, bundle paths only
    for item in assignment["items"]:
        assert (Path(item["path"]) / "index.html").is_file()
    assert (store / store_module.STORE).is_file()
    with pytest.raises(PreferenceError, match="already holds"):
        issue([tmp_path / "job"], store, seed=7)


def test_pairs_are_blind_by_default_and_say_what_was_left_out(tmp_path):
    job = _job(tmp_path)
    # An unfinished trial and one whose output folder is empty (another model,
    # so they are not taken for retries a scored trial replaced).
    unfinished = job / "landing__a9"
    (unfinished / "artifacts").mkdir(parents=True)
    (unfinished / "artifacts" / "index.html").write_text("<h1>late</h1>\n")
    (unfinished / "config.json").write_text("{}")
    (unfinished / "result.json").write_text(
        json.dumps({"task_name": "landing", "agent_name": "openhands", "model": "m9"})
    )
    empty = job / "landing__b9"
    (empty / "artifacts").mkdir(parents=True)
    (empty / "config.json").write_text("{}")
    (empty / "result.json").write_text(
        json.dumps(
            {
                "task_name": "landing",
                "agent_name": "pi-acp",
                "model": "m9",
                "finished_at": "2026-11-04 01:00:00",
            }
        )
    )
    store = tmp_path / "store"
    assignment = issue([job], store, seed=7)
    shown = json.dumps(assignment)
    assert "landing__" not in shown and str(job) not in shown
    for item in assignment["items"]:
        # Beside the store, never in it: the store maps items to agents.
        assert Path(item["path"]).parent == tmp_path / f"store-{store_module.BUNDLE}"
        assert (Path(item["path"]) / "index.html").is_file()
    manifest = json.loads((store / store_module.STORE).read_text())
    assert sorted((row["agent"], row["reason"]) for row in manifest["left_out"]) == [
        ("openhands", "unfinished"),
        ("pi-acp", "no files in artifacts/"),
    ]


def test_a_trial_without_a_task_digest_is_left_out(tmp_path):
    job = _job(tmp_path)
    loose = job / "landing__d0"
    (loose / "artifacts").mkdir(parents=True)
    (loose / "artifacts" / "index.html").write_text("<h1>d0</h1>\n")
    (loose / "config.json").write_text("{}")
    (loose / "result.json").write_text(
        json.dumps(
            {
                "task_name": "landing",
                "agent_name": "codex-acp",
                "model": "m3",
                "finished_at": "2026-11-04 01:00:00",
                "rewards": {"reward": 1.0},
            }
        )
    )
    store = tmp_path / "store"
    issue([job], store, seed=7)
    manifest = json.loads((store / store_module.STORE).read_text())
    assert [(r["agent"], r["reason"]) for r in manifest["left_out"]] == [
        ("codex-acp", "no task digest recorded")
    ]


def test_votes_between_outputs_of_one_arm_are_not_counted(tmp_path):
    # Arms a and c are both openhands: --by agent makes them one arm, so a
    # vote between them says nothing and is neither fitted nor counted.
    store, assignment = _issue(tmp_path)
    rows = [
        {"pair_id": p["pair_id"], "choice": "left", "rater": "r1", "rated_at": LATER}
        for p in assignment["pairs"]
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    same = sum(
        {_arm_of(store, p["left"]), _arm_of(store, p["right"])} == {"a", "c"}
        for p in assignment["pairs"]
    )
    by_agent = rank(store, by="agent", bootstrap=0)
    assert by_agent["same_arm_votes"] == same > 0
    games = sum(
        r["wins"] + r["ties"] + r["losses"] for r in by_agent["pairwise"]["overall"]
    )
    assert games == 2 * (len(rows) - same)
    assert rank(store, by="agent+model", bootstrap=0)["same_arm_votes"] == 0


def test_a_task_asking_for_another_protocol_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        store_module,
        "_task_preference",
        lambda trial, task, digest: ({"protocol": "best-of-n"}, None),
    )
    with pytest.raises(PreferenceError, match="best-of-n"):
        issue([_job(tmp_path)], tmp_path / "store", seed=1)
    assert not (tmp_path / "store").exists()


def _pair(assignment: dict, left: str, right: str, store: Path) -> dict:
    return next(
        p
        for p in assignment["pairs"]
        if {_arm_of(store, p["left"]), _arm_of(store, p["right"])} == {left, right}
    )


def _prefer(store: Path, pair: dict, arm: str) -> str:
    return "left" if _arm_of(store, pair["left"]) == arm else "right"


def test_votes_are_imported_once_ranked_and_replaced_by_a_later_vote(tmp_path):
    store, assignment = _issue(tmp_path)
    ab = _pair(assignment, "a", "b", store)
    ac = _pair(assignment, "a", "c", store)
    bc = _pair(assignment, "b", "c", store)
    rows = [
        {
            "pair_id": ab["pair_id"],
            "choice": _prefer(store, ab, "a"),
            "rater": "r1",
            "rated_at": LATER,
        },
        {
            "pair_id": ac["pair_id"],
            "choice": _prefer(store, ac, "a"),
            "rater": "r1",
            "rated_at": LATER,
        },
        {
            "pair_id": bc["pair_id"],
            "choice": _prefer(store, bc, "b"),
            "rater": "r1",
            "rated_at": LATER,
        },
        {
            "pair_id": ab["pair_id"],
            "choice": _prefer(store, ab, "a"),
            "rater": "r2",
            "rated_at": LATER,
        },
        {
            "pair_id": bc["pair_id"],
            "choice": "abstain",
            "rater": "r2",
            "rated_at": LATER,
        },
    ]
    votes = _votes(tmp_path / "votes.jsonl", rows)
    first = import_votes(store, votes)
    assert (first["added"], first["already_recorded"]) == (5, 0)
    again = import_votes(store, votes)
    assert (again["added"], again["already_recorded"]) == (0, 5)
    # The chain head is the last vote's hash, and re-importing leaves it.
    assert first["head_sha256"] == again["head_sha256"]
    assert first["head_sha256"].startswith("sha256:")
    result = rank(store, by="agent+model", bootstrap=20)
    order = [r["arm"] for r in result["pairwise"]["overall"]]
    assert order == ["openhands / m1", "pi-acp / m1", "openhands / m2"]
    agreed = result["agreement"]
    assert (agreed["targets"], agreed["rater_pairs"], agreed["agreeing"]) == (1, 1, 1)
    assert agreed["agreement"] == 1.0
    # r2's abstention on b-vs-c is recorded but not counted.
    assert (result["effective_votes"], result["abstentions"]) == (4, 1)
    # r1 changes their mind on a-vs-b later: only the later vote counts.
    flip = {
        **rows[0],
        "choice": _prefer(store, ab, "b"),
        "rated_at": "2099-01-02T00:00:00Z",
    }
    import_votes(store, _votes(tmp_path / "flip.jsonl", [flip]))
    again = rank(store, bootstrap=0)
    assert again["effective_votes"] == 4 and again["votes"] == 6
    assert again["replaced_votes"] == 1
    assert again["agreement"]["agreement"] == 0.0
    assert json.loads((store / store_module.RATINGS).read_text())["votes"] == 6


def test_agreement_is_set_against_the_raters_own_choice_rates():
    # Two raters who each split left and right half and half agree half the
    # time by chance; agreeing on both pairs is kappa 1, on neither kappa -1.
    same = ratings.agreement(
        {"p1": {"a": "left", "b": "left"}, "p2": {"a": "right", "b": "right"}}
    )
    assert (same["agreement"], same["chance_agreement"], same["kappa"]) == (
        1.0,
        0.5,
        1.0,
    )
    apart = ratings.agreement(
        {"p1": {"a": "left", "b": "right"}, "p2": {"a": "right", "b": "left"}}
    )
    assert (apart["agreement"], apart["chance_agreement"], apart["kappa"]) == (
        0.0,
        0.5,
        -1.0,
    )
    alone = ratings.agreement({"p1": {"a": "left"}})
    assert alone["agreement"] is None and alone["kappa"] is None


def test_abstentions_and_replaced_votes_are_not_counted(tmp_path):
    store, assignment = _issue(tmp_path)
    pairs = assignment["pairs"]
    rows = [
        {
            "pair_id": pairs[0]["pair_id"],
            "choice": "left",
            "rater": "r1",
            "rated_at": LATER,
        },
        {
            "pair_id": pairs[1]["pair_id"],
            "choice": "abstain",
            "rater": "r1",
            "rated_at": LATER,
        },
        {
            "pair_id": pairs[2]["pair_id"],
            "choice": "right",
            "rater": "r1",
            "rated_at": LATER,
        },
        {
            "pair_id": pairs[2]["pair_id"],
            "choice": "left",
            "rater": "r1",
            "rated_at": "2099-01-02T00:00:00Z",
        },
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    result = rank(store, bootstrap=0, min_agreement=0.6)
    assert (result["votes"], result["effective_votes"]) == (4, 2)
    assert (result["abstentions"], result["replaced_votes"]) == (1, 1)
    # One rater: no pair to check the minimum on, and it says so.
    assert result["agreement"]["met"] is None


def test_a_changed_copy_that_raters_see_is_refused(tmp_path):
    store, assignment = _issue(tmp_path)
    pair = assignment["pairs"][0]
    shown = Path(
        next(i["path"] for i in assignment["items"] if i["item_id"] == pair["left"])
    )
    (shown / "index.html").write_text("<h1>not what was issued</h1>\n")
    vote = {
        "pair_id": pair["pair_id"],
        "choice": "left",
        "rater": "r1",
        "rated_at": LATER,
    }
    with pytest.raises(PreferenceError, match="that raters see is missing or changed"):
        import_votes(store, _votes(tmp_path / "v.jsonl", [vote]))


def test_a_moved_output_is_missing_not_changed(tmp_path):
    store, assignment = _issue(tmp_path)
    pair = assignment["pairs"][0]
    items = [
        json.loads(line)
        for line in (store / store_module.ITEMS).read_text().splitlines()
    ]
    left = next(i for i in items if i["item_id"] == pair["left"])
    Path(left["output"]).rename(Path(left["output"]).with_name("moved"))
    vote = {
        "pair_id": pair["pair_id"],
        "choice": "left",
        "rater": "r1",
        "rated_at": LATER,
    }
    with pytest.raises(PreferenceError, match="is missing") as refused:
        import_votes(store, _votes(tmp_path / "v.jsonl", [vote]))
    assert "put the jobs folder back" in refused.value.hint


def test_a_vote_id_recorded_with_other_content_is_a_conflict(tmp_path):
    store, assignment = _issue(tmp_path)
    pair = assignment["pairs"][0]
    vote = {
        "pair_id": pair["pair_id"],
        "choice": "left",
        "rater": "r1",
        "rated_at": LATER,
    }
    import_votes(store, _votes(tmp_path / "v1.jsonl", [vote]))
    with pytest.raises(Conflict, match=r"v2.jsonl vote 1 \(rater r1, p-"):
        import_votes(
            store, _votes(tmp_path / "v2.jsonl", [{**vote, "choice": "right"}])
        )


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"pair_id": "p-nope", "choice": "left"}, "was not issued"),
        ({"choice": "sideways"}, "choice must be"),
        ({"choice": "left", "rated_at": "2001-01-01T00:00:00Z"}, "before the pair"),
        ({"choice": "left", "rated_at": "2099-01-01T00:00:00"}, "no time zone"),
        ({"choice": "left", "rater": ""}, "no rater"),
        ({"choice": "left", "aspect": "layout"}, "not a criterion"),
    ],
)
def test_a_bad_vote_refuses_the_whole_file(tmp_path, row, message):
    store, assignment = _issue(tmp_path)
    pair_id = assignment["pairs"][0]["pair_id"]
    good = {"pair_id": pair_id, "choice": "left", "rater": "r1", "rated_at": LATER}
    bad = {**good, **row}
    with pytest.raises(PreferenceError, match=message):
        import_votes(store, _votes(tmp_path / "votes.jsonl", [good, bad]))
    assert not (store / store_module.VOTES).exists()


def test_rubric_scores_are_checked_against_the_frozen_rubric(tmp_path):
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "id": "pdc",
                "version": "1",
                "criteria": [{"id": "layout", "scale": [1, 5]}],
            }
        )
    )
    store, assignment = _issue(tmp_path, rubric=rubric)
    assert assignment["rubric"]["criteria"][0]["id"] == "layout"
    items = [i["item_id"] for i in assignment["items"]]
    base = {"criterion": "layout", "rater": "r1", "rated_at": LATER}
    with pytest.raises(PreferenceError, match="outside"):
        import_votes(
            store,
            _votes(tmp_path / "v.jsonl", [{**base, "item_id": items[0], "score": 6}]),
        )
    with pytest.raises(PreferenceError, match="not in the rubric"):
        import_votes(
            store,
            _votes(
                tmp_path / "v.jsonl",
                [{**base, "item_id": items[0], "score": 3, "criterion": "x"}],
            ),
        )
    rows = [
        {**base, "item_id": i, "score": 4 if _arm_of(store, i) == "a" else 2}
        for i in items
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    table = {r["arm"]: r["mean"] for r in rank(store, bootstrap=10)["rubric"]}
    assert table == {"openhands / m1": 4.0, "openhands / m2": 2.0, "pi-acp / m1": 2.0}
    # The frozen rubric cannot change under the votes.
    (store / store_module.RUBRIC).write_text("{}")
    with pytest.raises(PreferenceError, match="changed after"):
        rank(store)


def test_the_rater_checks_count_each_rater_once_per_pair(tmp_path):
    # A rater's aspect vote on a pair mostly repeats their overall vote:
    # counting both would double the left-side evidence and the agreement.
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "id": "pdc",
                "version": "1",
                "criteria": [{"id": "layout", "scale": [1, 5]}],
            }
        )
    )
    store, assignment = _issue(tmp_path, rubric=rubric)
    rows = []
    for pair in assignment["pairs"]:
        for rater in ("r1", "r2"):
            base = {"pair_id": pair["pair_id"], "choice": "left", "rater": rater}
            rows += [
                {**base, "rated_at": LATER},
                {**base, "aspect": "layout", "rated_at": LATER},
            ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    result = rank(store, bootstrap=0)
    n = len(assignment["pairs"])
    assert result["left_share"]["decisive"] == 2 * n
    assert result["agreement"]["targets"] == n
    assert result["agreement"]["rater_pairs"] == n


def test_an_output_changed_after_issue_refuses_its_votes(tmp_path):
    store, assignment = _issue(tmp_path)
    pair = assignment["pairs"][0]
    trial = json.loads((store / store_module.ITEMS).read_text().splitlines()[0])
    for line in (store / store_module.ITEMS).read_text().splitlines():
        item = json.loads(line)
        if item["item_id"] == pair["left"]:
            trial = item
    (Path(trial["output"]) / "index.html").write_text("<h1>edited</h1>\n")
    vote = {
        "pair_id": pair["pair_id"],
        "choice": "left",
        "rater": "r1",
        "rated_at": LATER,
    }
    with pytest.raises(PreferenceError, match="changed after it was issued") as refused:
        import_votes(store, _votes(tmp_path / "v.jsonl", [vote]))
    assert "make new pairs into a new store" in refused.value.hint


def test_an_edited_vote_log_is_refused(tmp_path):
    store, assignment = _issue(tmp_path)
    rows = [
        {"pair_id": p["pair_id"], "choice": "left", "rater": "r1", "rated_at": LATER}
        for p in assignment["pairs"][:2]
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    log = store / store_module.VOTES
    log.write_text(log.read_text().replace('"choice":"left"', '"choice":"right"', 1))
    with pytest.raises(PreferenceError, match="hash does not match") as refused:
        rank(store)
    assert refused.value.hint == (
        "votes.v1.jsonl was changed outside `bench preferences import`:"
        " restore it from a copy"
    )


def test_csv_votes_and_the_shared_min_agreement(tmp_path, monkeypatch):
    monkeypatch.setattr(
        store_module,
        "_task_preference",
        lambda trial, task, digest: (
            {
                "protocol": "pairwise",
                "min_agreement": {"metric": "percent", "value": 0.8},
            },
            None,
        ),
    )
    store, assignment = _issue(tmp_path)
    pair = assignment["pairs"][0]["pair_id"]
    csv = tmp_path / "votes.csv"
    csv.write_text(
        "pair_id,choice,rater,rated_at\n"
        f"{pair},left,r1,{LATER}\n"
        f"{pair},right,r2,{LATER}\n"
    )
    assert import_votes(store, csv, source="sheet")["added"] == 2
    agreed = rank(store, bootstrap=0)["agreement"]
    assert agreed["min_agreement"] == 0.8 and agreed["met"] is False


def test_the_cli_runs_pairs_import_and_rank(tmp_path):
    from benchflow.cli.main import app

    job = _job(tmp_path)
    runner = CliRunner()
    made = runner.invoke(
        app,
        [
            "preferences",
            "pairs",
            str(job),
            "--store",
            str(tmp_path / "s"),
            "--out",
            str(tmp_path / "a.json"),
            "--seed",
            "3",
            "--per-task",
            "4",
        ],
    )
    assert made.exit_code == 0, made.output
    assert "4 pairs of 6 outputs." in made.output
    assert f"outputs under their opaque ids in {tmp_path / 's-outputs'}" in " ".join(
        made.output.split()
    )
    assignment = json.loads((tmp_path / "a.json").read_text())
    assert len(assignment["pairs"]) == 4
    votes = _votes(
        tmp_path / "v.jsonl",
        [
            {"pair_id": p["pair_id"], "choice": "left", "rater": "r", "rated_at": LATER}
            for p in assignment["pairs"]
        ],
    )
    imported = runner.invoke(
        app, ["preferences", "import", str(tmp_path / "s"), "--votes", str(votes)]
    )
    assert imported.exit_code == 0 and "4 votes added" in imported.output
    assert "Log head: sha256:" in imported.output
    again = runner.invoke(
        app, ["preferences", "import", str(tmp_path / "s"), "--votes", str(votes)]
    )
    assert "0 votes added, 4 already recorded" in again.output
    ranked = runner.invoke(
        app, ["preferences", "rank", str(tmp_path / "s"), "--json", "--bootstrap", "5"]
    )
    assert ranked.exit_code == 0, ranked.output
    assert json.loads(ranked.output)["votes"] == 4
    table = runner.invoke(
        app, ["preferences", "rank", str(tmp_path / "s"), "--bootstrap", "5"]
    )
    assert table.exit_code == 0, table.output
    # Every vote chose the left side: the line says what that z means.
    shown = " ".join(table.output.split())
    assert (
        "(z = 2.0, counting each vote apart although raters of a pair saw the same "
        "sides: more left choices than chance gives, a possible side bias)."
    ) in shown
    # The checks come before the ratings, with singular and plural counts.
    assert shown.index("votes counted") < shown.index("Pairwise (overall)")
    assert "left out: 0 abstentions and 0 votes a rater replaced later" in shown
    refused = runner.invoke(app, ["preferences", "rank", str(tmp_path / "nowhere")])
    assert refused.exit_code == 1


# --- a 0.10.0rc1 acceptance run: the task's own preference table ----------


def _task_folder(root: Path, name: str, preference: str) -> Path:
    """A task.md task (YAML front matter) that declares preference:."""
    task = root / name
    (task / "tests").mkdir(parents=True)
    (task / "task.md").write_text(
        "---\n"
        "schema_version: '1.3'\n"
        "sandbox: {docker_image: 'python:3.12-slim', workdir: /app}\n"
        f"preference: {preference}\n"
        "---\n\n## prompt\n\nMake a page.\n"
    )
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\necho 1 > /logs/verifier/reward.txt\n"
    )
    return task


def _run_of(root: Path, tasks_dir: Path, task: Path, *, arms=("a", "b")) -> Path:
    """A job whose evaluation.json records tasks_dir, as `bench run` writes it."""
    from benchflow._utils.task_authoring import task_digest

    job = root / "job"
    job.mkdir(parents=True)
    (job / "evaluation.json").write_text(json.dumps({"tasks_dir": str(tasks_dir)}))
    for arm in arms:
        agent, model = ARMS[arm]
        trial = job / f"{task.name}__{arm}"
        (trial / "artifacts").mkdir(parents=True)
        (trial / "artifacts" / "index.html").write_text(f"<h1>{arm}</h1>\n")
        (trial / "config.json").write_text("{}")
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "task_name": task.name,
                    "task_digest": task_digest(task),
                    "agent": agent,
                    "model": model,
                    "finished_at": "2026-11-04 01:00:00",
                    "rewards": {"reward": 1.0},
                }
            )
        )
    return job


FIXED = "{protocol: pairwise, randomize_sides: false, min_agreement: 0.8}"


@pytest.mark.parametrize("one_task_run", [True, False])
def test_a_yaml_task_s_table_is_used_from_a_folder_or_a_one_task_run(
    tmp_path, one_task_run
):
    task = _task_folder(tmp_path / "tasks", "landing", FIXED)
    job = _run_of(tmp_path, task if one_task_run else task.parent, task)
    issue([job], tmp_path / "store", seed=3)
    manifest = json.loads((tmp_path / "store" / store_module.STORE).read_text())
    (table,) = manifest["protocol"].values()
    assert table["randomize_sides"] is False and table["min_agreement"] == 0.8
    assert manifest["sides"] == "fixed"
    assert manifest["protocol_notes"] == []


def test_a_task_edited_after_the_trials_does_not_lend_them_its_table(tmp_path):
    task = _task_folder(tmp_path / "tasks", "landing", FIXED)
    job = _run_of(tmp_path, task.parent, task)
    (task / "task.md").write_text((task / "task.md").read_text() + "Edited.\n")
    issue([job], tmp_path / "store", seed=3)
    manifest = json.loads((tmp_path / "store" / store_module.STORE).read_text())
    assert manifest["protocol"] == {}
    assert "changed after the trial ran" in manifest["protocol_notes"][0]


def test_a_rubric_task_gets_items_to_score_and_no_pairs(tmp_path):
    task = _task_folder(tmp_path / "tasks", "landing", "{protocol: rubric}")
    job = _run_of(tmp_path, task.parent, task)
    assignment = issue([job], tmp_path / "store", seed=3)
    assert assignment["pairs"] == []
    assert len(assignment["items"]) == 2
    manifest = json.loads((tmp_path / "store" / store_module.STORE).read_text())
    assert manifest["rubric_only_tasks"] == ["landing"]
    assert manifest["left_out"] == []


def test_an_errored_trial_and_an_empty_output_are_left_out(tmp_path):
    # One trial per arm: an errored attempt beside a scored one of its arm is
    # a replaced attempt, and trial discovery keeps the scored one.
    job = _job(tmp_path, per_arm=1)
    errored = job / "landing__a0"
    data = json.loads((errored / "result.json").read_text())
    data.update(rewards=None, error="RuntimeError: sandbox setup failed")
    (errored / "result.json").write_text(json.dumps(data))
    empty = job / "landing__b0" / "artifacts"
    (empty / "index.html").unlink()
    (empty / "sub").mkdir()
    issue([job], tmp_path / "store", seed=3)
    manifest = json.loads((tmp_path / "store" / store_module.STORE).read_text())
    reasons = {Path(row["trial"]).name: row["reason"] for row in manifest["left_out"]}
    assert reasons["landing__a0"] == "errored before a reward"
    assert reasons["landing__b0"] == "no files in artifacts/"


def test_rank_says_which_trials_the_pairs_left_out(tmp_path):
    """Round 4 of the release review: an arm whose second trial made no page
    was rated on its one page, and rank never said so."""
    from benchflow.cli.preferences import _rater_checks

    job = _job(tmp_path)
    (job / "landing__b0" / "artifacts" / "index.html").unlink()
    issue([job], tmp_path / "store", seed=3)
    line = next(
        line
        for line in _rater_checks(rank(tmp_path / "store"))
        if line.startswith("Not in any pair")
    )
    assert line.startswith("Not in any pair: 1 trial: ")
    assert line.endswith(
        ": 1 (no files in artifacts/). Their arms are rated on their other outputs."
    )


def test_the_raters_copies_carry_one_time_and_paths_that_resolve(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    job = _job(tmp_path)
    os.utime(job / "landing__a0" / "artifacts" / "index.html", (1e9, 1e9))
    assignment = issue([job], Path("store"), seed=3)
    times = set()
    for item in assignment["items"]:
        path = Path(item["path"])
        assert path.is_absolute() and path.is_dir()
        times |= {p.stat().st_mtime for p in [path, *path.rglob("*")]}
    assert len(times) == 1


def test_fixed_sides_say_nothing_about_raters(tmp_path):
    from benchflow.cli.preferences import _rater_checks

    task = _task_folder(tmp_path / "tasks", "landing", FIXED)
    job = _run_of(tmp_path, task.parent, task, arms=("a", "b", "c"))
    store = tmp_path / "store"
    assignment = issue([job], store, seed=3)
    rows = [
        {
            "pair_id": pair["pair_id"],
            "rater": rater,
            "choice": "left",
            "rated_at": LATER,
        }
        for pair in assignment["pairs"]
        for rater in ("r1", "r2")
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    result = rank(store, bootstrap=0)
    assert result["sides"] == "fixed"
    text = "\n".join(_rater_checks(result))
    assert "the sides were not shuffled" in text
    assert "raters favour" not in text


@pytest.mark.parametrize(
    "preference,message",
    [
        ("{protocol: tournament}", "preference.protocol must be"),
        ("{randomize_sides: maybe}", "randomize_sides must be true or false"),
        ("{raters_per_pair: 0}", "raters_per_pair must be a whole number"),
        ("{min_agreement: 1.5}", "min_agreement must be a number from 0 to 1"),
    ],
)
def test_a_task_s_preference_table_is_checked(tmp_path, preference, message):
    from benchflow.task.runtime_view import TaskRuntimeView

    task = _task_folder(tmp_path, "landing", preference)
    with pytest.raises(Exception, match=message):
        TaskRuntimeView.from_task_dir(task)


# --- what the counts and tables claim -------------------------------------


def test_counted_votes_leave_out_the_ones_by_merges_into_one_arm(tmp_path):
    from benchflow.cli.preferences import _rater_checks

    store, assignment = _issue(tmp_path)
    rows = [
        {"pair_id": p["pair_id"], "rater": "r1", "choice": "left", "rated_at": LATER}
        for p in assignment["pairs"]
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    by_agent = rank(store, by="agent", bootstrap=0)
    same = by_agent["same_arm_votes"]
    assert same > 0
    assert by_agent["effective_votes"] == len(rows) - same
    assert by_agent["pairwise_votes"] == len(rows) - same
    first = _rater_checks(by_agent)[0]
    assert first.startswith(f"{len(rows) - same} votes counted")
    assert "between outputs that --by agent puts in one arm" in first


def test_a_rubric_only_store_shows_no_pairwise_ratings(tmp_path):
    task = _task_folder(tmp_path / "tasks", "landing", "{protocol: rubric}")
    job = _run_of(tmp_path, task.parent, task)
    store = tmp_path / "store"
    rubric = tmp_path / "rubric.json"
    rubric.write_text(
        json.dumps(
            {
                "id": "pdc",
                "version": "1",
                "criteria": [{"id": "layout", "scale": [1, 5]}],
            }
        )
    )
    assignment = issue([job], store, seed=3, rubric=rubric)
    rows = [
        {
            "item_id": item["item_id"],
            "criterion": "layout",
            "score": 4,
            "rater": "r1",
            "rated_at": LATER,
        }
        for item in assignment["items"]
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    result = rank(store, bootstrap=0)
    assert result["pairwise"] == {}
    assert result["rubric"]


def test_an_arm_with_no_comparison_is_unranked(tmp_path):
    outcomes = [ratings.Outcome("a", "b", 1.0), ratings.Outcome("a", "b", 0.0)]
    rows = ratings.pairwise_table(outcomes, ["a", "b", "c"], bootstrap=50, seed=1)
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["c"]["rating"] is None and "interval" not in by_arm["c"]
    assert rows[-1]["arm"] == "c"
    assert by_arm["a"]["rating"] is not None
