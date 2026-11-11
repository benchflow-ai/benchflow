"""Guards outside items and votes as shown: a golden, a remix or any
output made outside BenchFlow can be a side beside trials of its task version,
votes can name the two outputs a rater saw instead of a pair BenchFlow issued,
and the FDB vote site's export is read as it comes. The gap it closes: on
0.10.0rc2 only BenchFlow trials could be paired, and nothing read the vote
site's export (``fdb-votes/v1``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.preferences import PreferenceError, import_votes, issue, rank
from benchflow.preferences import store as store_module

DIGEST = "sha256:" + "1" * 64
OTHER = "sha256:" + "2" * 64
LATER = "2099-01-01T00:00:00Z"
ARMS = {"a": ("openhands", "m1"), "b": ("pi-acp", "m1")}


def _trials(root: Path, *, task: str = "landing", digests=(DIGEST,)) -> Path:
    job = root / "job"
    for d, digest in enumerate(digests):
        for arm, (agent, model) in ARMS.items():
            trial = job / f"{task}__{arm}{d}"
            (trial / "artifacts").mkdir(parents=True)
            (trial / "artifacts" / "index.html").write_text(f"<h1>{arm}{d}</h1>\n")
            (trial / "config.json").write_text("{}")
            (trial / "result.json").write_text(
                json.dumps(
                    {
                        "task_name": task,
                        "task_digest": digest,
                        "agent_name": agent,
                        "model": model,
                        "finished_at": "2026-11-04 01:00:00",
                        "rewards": {"reward": 1.0},
                    }
                )
            )
    return job


def _site(root: Path, name: str, *, page: str = "index.html") -> Path:
    folder = root / "outside" / name
    folder.mkdir(parents=True)
    (folder / page).write_text(f"<h1>{name}</h1>\n")
    return folder


def _items(root: Path, rows: list[dict]) -> Path:
    path = root / "items.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _golden_and_remix(root: Path, task: str = "landing") -> list[dict]:
    return [
        {
            "task": task,
            "output": str(_site(root, f"golden-{task}")),
            "kind": "golden",
            "condition": "golden",
            "agent": "golden",
            "model": "person-picked",
            "external_id": f"golden-{task}-R1",
            "source": "goldens/README.md",
        },
        {
            "task": task,
            "output": str(_site(root, f"remix-{task}-1")),
            "condition": "enriched",
            "agent": "remix",
            "model": "claude-opus-5-5",
            "external_id": f"remix-{task}-1",
        },
    ]


def _store_items(store: Path) -> dict[str, dict]:
    return {
        row["item_id"]: row
        for row in map(
            json.loads, (store / store_module.ITEMS).read_text().splitlines()
        )
    }


def _by_external(store: Path) -> dict[str, str]:
    return {
        row["external_id"]: item_id
        for item_id, row in _store_items(store).items()
        if row.get("external_id")
    }


def _votes(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# --- outside items --------------------------------------------------------


def test_outside_items_are_sides_beside_the_trials_of_their_task_version(tmp_path):
    items = _items(tmp_path, _golden_and_remix(tmp_path))
    store = tmp_path / "store"
    assignment = issue([_trials(tmp_path)], store, seed=3, items=items)
    stored = _store_items(store)
    assert len(stored) == 4  # two trials, the golden and the remix
    outside = [i for i in stored.values() if i["origin"] == "outside"]
    assert {i["external_id"] for i in outside} == {
        "golden-landing-R1",
        "remix-landing-1",
    }
    # An outside item takes the task version of the trials of its task.
    assert {i["task_digest"] for i in stored.values()} == {DIGEST}
    assert all(i["trial"] is None for i in outside)
    # Every pair of different arms: 4 arms, 6 pairs.
    assert len(assignment["pairs"]) == 6
    shown = json.dumps(assignment)
    for word in ("golden", "remix", "enriched", "openhands", "person-picked"):
        assert word not in shown, word
    # The raters' copies of an outside item are its folder's files.
    for item in assignment["items"]:
        assert (Path(item["path"]) / "index.html").is_file()


def test_an_outside_item_names_its_version_when_the_task_has_two(tmp_path):
    job = _trials(tmp_path, digests=(DIGEST, OTHER))
    rows = _golden_and_remix(tmp_path)
    with pytest.raises(PreferenceError, match="task_digest"):
        issue([job], tmp_path / "s1", seed=1, items=_items(tmp_path, rows))
    pinned = [{**row, "task_digest": OTHER} for row in rows]
    (tmp_path / "items.jsonl").unlink()
    issue([job], tmp_path / "s2", seed=1, items=_items(tmp_path, pinned))
    stored = _store_items(tmp_path / "s2")
    outside = [i for i in stored.values() if i["origin"] == "outside"]
    assert {i["task_digest"] for i in outside} == {OTHER}


def test_outside_items_of_a_task_with_no_trials_are_paired_together(tmp_path):
    rows = _golden_and_remix(tmp_path, task="awning")
    rows.append(
        {
            "task": "awning",
            "output": str(_site(tmp_path, "remix-awning-2")),
            "condition": "enriched",
            "agent": "remix-v7",
            "model": "claude-opus-5-5",
        }
    )
    assignment = issue(
        [_trials(tmp_path)], tmp_path / "s", seed=2, items=_items(tmp_path, rows)
    )
    stored = _store_items(tmp_path / "s")
    awning = {i for i, row in stored.items() if row["task"] == "awning"}
    assert len(awning) == 3
    pairs = [p for p in assignment["pairs"] if {p["left"], p["right"]} <= awning]
    assert len(pairs) == 3


def test_a_golden_as_the_bar_is_an_item_but_never_a_side(tmp_path):
    items = _items(tmp_path, _golden_and_remix(tmp_path))
    assignment = issue(
        [_trials(tmp_path)], tmp_path / "s", seed=4, items=items, golden="bar"
    )
    stored = _store_items(tmp_path / "s")
    golden = next(i for i, row in stored.items() if row.get("kind") == "golden")
    assert golden in {item["item_id"] for item in assignment["items"]}
    assert all(golden not in (p["left"], p["right"]) for p in assignment["pairs"])
    # Three arms are left: the two trials and the remix.
    assert len(assignment["pairs"]) == 3
    manifest = json.loads((tmp_path / "s" / store_module.STORE).read_text())
    assert manifest["golden"] == "bar"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"output": "/nonexistent/folder"}, "no folder"),
        ({"task": ""}, "needs a task"),
        ({"kind": "bronze"}, "kind"),
        ({"agent": ""}, "agent"),
        ({"external_id": "remix-landing-1"}, "external_id"),
        ({"rater": "someone"}, "unknown key"),
    ],
)
def test_a_bad_outside_item_refuses_the_whole_file(tmp_path, change, message):
    rows = _golden_and_remix(tmp_path)
    rows[0] = {**rows[0], **change}
    with pytest.raises(PreferenceError, match=message):
        issue([_trials(tmp_path)], tmp_path / "s", seed=1, items=_items(tmp_path, rows))
    assert not (tmp_path / "s").exists()


def test_an_outside_folder_holding_a_link_is_refused(tmp_path):
    rows = _golden_and_remix(tmp_path)
    (Path(rows[1]["output"]) / "elsewhere").symlink_to(tmp_path)
    with pytest.raises(PreferenceError, match="link"):
        issue([_trials(tmp_path)], tmp_path / "s", seed=1, items=_items(tmp_path, rows))


def test_entry_names_the_page_and_leaves_out_outputs_without_it(tmp_path):
    rows = _golden_and_remix(tmp_path)
    rows.append(
        {
            "task": "landing",
            "output": str(_site(tmp_path, "no-index", page="other.html")),
            "agent": "remix-v7",
            "model": "claude-opus-5-5",
        }
    )
    assignment = issue(
        [_trials(tmp_path)],
        tmp_path / "s",
        seed=5,
        items=_items(tmp_path, rows),
        entry="index.html",
    )
    assert len(assignment["items"]) == 4
    for item in assignment["items"]:
        assert item["entry"] == "index.html"
        assert Path(item["page"]) == Path(item["path"]) / "index.html"
        assert Path(item["page"]).is_file()
    manifest = json.loads((tmp_path / "s" / store_module.STORE).read_text())
    reasons = [row["reason"] for row in manifest["left_out"]]
    assert reasons == ["no index.html in the output"]


def test_conditions_label_the_items_and_rank_by_condition(tmp_path):
    store = tmp_path / "s"
    assignment = issue(
        [_trials(tmp_path)],
        store,
        seed=6,
        items=_items(tmp_path, _golden_and_remix(tmp_path)),
        condition="one-shot",
    )
    stored = _store_items(store)
    assert {row["condition"] for row in stored.values()} == {
        "one-shot",
        "golden",
        "enriched",
    }
    condition = {i: row["condition"] for i, row in stored.items()}
    rows = [
        {
            "pair_id": p["pair_id"],
            "choice": "left"
            if condition[p["left"]] == "golden"
            else "right"
            if condition[p["right"]] == "golden"
            else "tie",
            "rater": "r1",
            "rated_at": LATER,
        }
        for p in assignment["pairs"]
    ]
    import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    result = rank(store, by="condition", bootstrap=0)
    order = [row["arm"] for row in result["pairwise"]["overall"]]
    assert order[0] == "golden"
    assert set(order) == {"golden", "enriched", "one-shot"}


def test_a_condition_tells_two_otherwise_equal_arms_apart(tmp_path):
    rows = [
        {
            "task": "landing",
            "output": str(_site(tmp_path, "same-arm-enriched")),
            "condition": "enriched",
            "agent": "openhands",
            "model": "m1",
        }
    ]
    assignment = issue(
        [_trials(tmp_path)],
        tmp_path / "s",
        seed=7,
        items=_items(tmp_path, rows),
        condition="one-shot",
    )
    # The outside output shares the trial's agent and model, but not its
    # condition, so the two are different arms and get a pair.
    stored = _store_items(tmp_path / "s")
    enriched = next(i for i, row in stored.items() if row["origin"] == "outside")
    trial_a = next(
        i
        for i, row in stored.items()
        if row["origin"] == "trial" and row["agent"] == "openhands"
    )
    assert any(
        {p["left"], p["right"]} == {enriched, trial_a} for p in assignment["pairs"]
    )


# --- votes as shown ---------------------------------------------------------


def _store_with_outside(tmp_path: Path, **kwargs) -> tuple[Path, dict]:
    store = tmp_path / "store"
    assignment = issue(
        [_trials(tmp_path)],
        store,
        seed=8,
        items=_items(tmp_path, _golden_and_remix(tmp_path)),
        **kwargs,
    )
    return store, assignment


def test_a_vote_names_the_two_outputs_as_the_rater_saw_them(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    ids = _by_external(store)
    golden, remix = ids["golden-landing-R1"], ids["remix-landing-1"]
    rows = [
        # The golden on the left, chosen: a win for the golden.
        {"left_item": golden, "right_item": remix, "choice": "left", "rater": "p1"},
        # The same two the other way round, the left chosen: a win for the remix.
        {"left_item": remix, "right_item": golden, "choice": "left", "rater": "p2"},
        # External ids resolve to the items they name.
        {
            "left_item": "remix-landing-1",
            "right_item": "golden-landing-R1",
            "choice": "right",
            "rater": "p3",
        },
    ]
    for row in rows:
        row["rated_at"] = LATER
    added = import_votes(store, _votes(tmp_path / "v.jsonl", rows))
    assert added["added"] == 3
    log = [
        json.loads(line)
        for line in (store / store_module.VOTES).read_text().splitlines()
    ]
    assert [(v["shown_left"], v["shown_right"]) for v in log] == [
        (golden, remix),
        (remix, golden),
        (remix, golden),
    ]
    result = rank(store, by="agent+model", bootstrap=0)
    rows_out = {r["arm"]: r for r in result["pairwise"]["overall"]}
    golden_arm = next(a for a in rows_out if a.startswith("golden"))
    assert (rows_out[golden_arm]["wins"], rows_out[golden_arm]["losses"]) == (2, 1)
    # A later vote by the same rater on the same two outputs, in either
    # order, replaces the earlier one.
    flip = {
        "left_item": golden,
        "right_item": remix,
        "choice": "left",
        "rater": "p2",
        "rated_at": "2099-01-02T00:00:00Z",
    }
    import_votes(store, _votes(tmp_path / "flip.jsonl", [flip]))
    again = rank(store, bootstrap=0)
    rows_out = {r["arm"]: r for r in again["pairwise"]["overall"]}
    assert (rows_out[golden_arm]["wins"], rows_out[golden_arm]["losses"]) == (3, 0)
    assert again["replaced_votes"] == 1


def test_a_vote_on_an_issued_pair_and_one_as_shown_are_the_same_pair(tmp_path):
    store, assignment = _store_with_outside(tmp_path)
    pair = assignment["pairs"][0]
    first = {
        "pair_id": pair["pair_id"],
        "choice": "left",
        "rater": "p1",
        "rated_at": LATER,
    }
    later = {
        "left_item": pair["right"],
        "right_item": pair["left"],
        "choice": "left",
        "rater": "p1",
        "rated_at": "2099-01-02T00:00:00Z",
    }
    import_votes(store, _votes(tmp_path / "v.jsonl", [first, later]))
    result = rank(store, bootstrap=0)
    assert result["replaced_votes"] == 1 and result["pairwise_votes"] == 1


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"left_item": "i-nope"}, "was not issued"),
        ({"right_item": "LEFT"}, "itself"),
        ({"rated_at": "2001-01-01T00:00:00Z"}, "before the items"),
        ({"rater_kind": "robot"}, "rater_kind"),
        ({"against": "mood"}, "against"),
        ({"invited": "yes"}, "invited"),
        ({"pair_id": "p-x"}, "either a pair_id or"),
    ],
)
def test_a_bad_vote_as_shown_refuses_the_whole_file(tmp_path, change, message):
    store, _ = _store_with_outside(tmp_path)
    ids = _by_external(store)
    good = {
        "left_item": ids["golden-landing-R1"],
        "right_item": ids["remix-landing-1"],
        "choice": "left",
        "rater": "p1",
        "rated_at": LATER,
    }
    bad = {**good, **change}
    if bad.get("right_item") == "LEFT":
        bad["right_item"] = bad["left_item"]
    with pytest.raises(PreferenceError, match=message):
        import_votes(store, _votes(tmp_path / "v.jsonl", [good, bad]))
    assert not (store / store_module.VOTES).exists()


def test_two_outputs_of_different_task_versions_are_not_a_pair(tmp_path):
    rows = _golden_and_remix(tmp_path) + _golden_and_remix(tmp_path, task="awning")
    store = tmp_path / "s"
    issue([_trials(tmp_path)], store, seed=9, items=_items(tmp_path, rows))
    ids = _by_external(store)
    vote = {
        "left_item": ids["golden-landing-R1"],
        "right_item": ids["golden-awning-R1"],
        "choice": "left",
        "rater": "p1",
        "rated_at": LATER,
    }
    with pytest.raises(PreferenceError, match="task version"):
        import_votes(store, _votes(tmp_path / "v.jsonl", [vote]))


def test_a_pair_id_vote_keeps_the_vote_id_it_had_before(tmp_path):
    """Guards the vote ids of 0.10.0rc2's votes files: re-importing one after
    this change must add nothing, so a vote's id is made from the same fields
    as before (kind, rater, rated_at, pair_id, aspect)."""
    store, assignment = _store_with_outside(tmp_path)
    pair = assignment["pairs"][0]
    row = {
        "pair_id": pair["pair_id"],
        "choice": "left",
        "rater": "r1",
        "rated_at": LATER,
    }
    import_votes(store, _votes(tmp_path / "v.jsonl", [row]))
    vote = json.loads((store / store_module.VOTES).read_text())
    identity = {
        "kind": "pairwise",
        "rater": "r1",
        "rated_at": "2099-01-01T00:00:00.000000Z",
        "pair_id": pair["pair_id"],
        "aspect": None,
    }
    expected = store_module._sha256(store_module._canonical(identity))[7:23]
    assert vote["vote_id"] == expected


def test_rater_kind_invited_and_the_bar_are_recorded(tmp_path):
    store, assignment = _store_with_outside(tmp_path)
    pair = assignment["pairs"][0]
    row = {
        "pair_id": pair["pair_id"],
        "choice": "right",
        "rater": "model:judge",
        "rater_kind": "model",
        "invited": True,
        "against": "golden",
        "tags": ["hierarchy"],
        "rated_at": LATER,
    }
    import_votes(store, _votes(tmp_path / "v.jsonl", [row]))
    vote = json.loads((store / store_module.VOTES).read_text())
    assert (vote["rater_kind"], vote["invited"], vote["against"], vote["tags"]) == (
        "model",
        True,
        "golden",
        ["hierarchy"],
    )


# --- the vote site's export -------------------------------------------------


def _export(
    path: Path, votes: list[dict], *, schema: str = "fdb-votes/v1", at: str = LATER
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": schema,
                "exportedAt": at,
                "source": {"site": "https://votes.example", "surface": "vote"},
                "people": [],
                "counts": {"votes": len(votes)},
                "collector": {"cases": [{"caseId": "landing", "votes": votes}]},
                "assessment": {"answers": []},
            }
        )
    )
    return path


def _site_vote(
    left: str, right: str, choice: str, by: str, *, at: str = LATER, **extra
):
    return {
        "id": f"vote-{by}-{left}-{right}",
        "reviewer": "Guest" if by.startswith("g-") else "A Person",
        "reviewerId": by,
        "invited": not by.startswith("g-"),
        "choice": choice,
        "reason": "",
        "tags": [],
        "createdAt": at,
        "phase": "evaluation",
        "leftId": left,
        "rightId": right,
        "taskId": None,
        "judgeAgainst": "brief",
        **extra,
    }


def test_the_vote_site_export_is_read_as_it_comes(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    votes = [
        _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann"),
        _site_vote("remix-landing-1", "golden-landing-R1", "skip", "p-bo"),
        _site_vote(
            "golden-landing-R1",
            "remix-landing-1",
            "right",
            "g-0123456789abcdef01234567",
            judgeAgainst="golden",
            reason="the type is cleaner",
        ),
    ]
    export = _export(tmp_path / "export.json", votes)
    first = import_votes(store, export)
    assert first["added"] == 3
    log = [
        json.loads(line)
        for line in (store / store_module.VOTES).read_text().splitlines()
    ]
    by_rater = {v["rater"]: v for v in log}
    assert by_rater["p-ann"]["rater_kind"] == "person"
    assert by_rater["p-bo"]["choice"] == "abstain"
    guest = by_rater["g-0123456789abcdef01234567"]
    assert (guest["rater_kind"], guest["invited"], guest["against"]) == (
        "guest",
        False,
        "golden",
    )
    assert guest["reason"] == "the type is cleaner"
    assert guest["source"] == "https://votes.example vote"
    # The same export again adds nothing.
    assert import_votes(store, export)["added"] == 0


def test_a_changed_vote_on_the_site_replaces_the_earlier_one(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    before = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    import_votes(store, _export(tmp_path / "e1.json", [before]))
    # The site keeps one vote per voter and pair: a changed vote has the same
    # id and a later time.
    after = {**before, "choice": "right", "createdAt": "2099-01-03T00:00:00Z"}
    later = _export(tmp_path / "e2.json", [after], at="2099-01-03T00:00:00Z")
    assert import_votes(store, later)["added"] == 1
    result = rank(store, bootstrap=0)
    rows = {r["arm"]: r for r in result["pairwise"]["overall"]}
    golden_arm = next(a for a in rows if a.startswith("golden"))
    assert (rows[golden_arm]["wins"], rows[golden_arm]["losses"]) == (0, 1)
    assert result["replaced_votes"] == 1


def test_a_vote_taken_back_on_the_site_stops_counting(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    kept = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    undone = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-bo")
    import_votes(store, _export(tmp_path / "e1.json", [kept, undone]))
    assert rank(store, bootstrap=0)["pairwise_votes"] == 2
    # p-bo pressed Undo: the next export no longer has the vote.
    later = _export(tmp_path / "e2.json", [kept], at="2099-01-05T00:00:00Z")
    counts = import_votes(store, later)
    # The withdrawal is counted as such, not as a vote added.
    assert (counts["added"], counts["already_recorded"], counts["withdrawn"]) == (
        0,
        1,
        1,
    )
    result = rank(store, bootstrap=0)
    assert result["pairwise_votes"] == 1
    log = [
        json.loads(line)
        for line in (store / store_module.VOTES).read_text().splitlines()
    ]
    assert log[-1]["choice"] == "abstain" and log[-1]["rater"] == "p-bo"
    assert "withdrawn" in log[-1]["reason"]
    # Importing the later export again adds nothing more.
    assert import_votes(store, later)["added"] == 0


def test_an_export_withdraws_only_votes_cast_before_it_was_made(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    early = _site_vote(
        "golden-landing-R1", "remix-landing-1", "left", "p-ann", at=LATER
    )
    late = _site_vote(
        "golden-landing-R1",
        "remix-landing-1",
        "right",
        "p-bo",
        at="2099-01-01T00:15:00Z",
    )
    first = _export(tmp_path / "e1.json", [early], at="2099-01-01T00:10:00Z")
    second = _export(tmp_path / "e2.json", [early, late], at="2099-01-01T00:30:00Z")
    import_votes(store, first)
    import_votes(store, second)
    # The older export, imported again, can't speak for a vote cast after it.
    again = import_votes(store, first)
    assert (again["added"], again["withdrawn"]) == (0, 0)
    assert rank(store, bootstrap=0)["pairwise_votes"] == 2


def test_an_export_with_no_votes_or_no_time_withdraws_nothing(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    vote = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    import_votes(store, _export(tmp_path / "e1.json", [vote]))
    # A site whose store was emptied or replaced exports no votes at all.
    empty = import_votes(
        store, _export(tmp_path / "e2.json", [], at="2099-01-05T00:00:00Z")
    )
    assert empty["withdrawn"] == 0
    # Without exportedAt nothing says which votes the export could have held.
    undated = json.loads((tmp_path / "e1.json").read_text())
    undated.pop("exportedAt")
    undated["collector"]["cases"][0]["votes"] = [
        _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-cy")
    ]
    (tmp_path / "e3.json").write_text(json.dumps(undated))
    assert import_votes(store, tmp_path / "e3.json")["withdrawn"] == 0
    assert rank(store, bootstrap=0)["pairwise_votes"] == 2


@pytest.mark.parametrize(
    "at", ["2099-01-05 00:00:00", 4070908800, "2099-13-05T00:00:00Z", 0, ""]
)
@pytest.mark.parametrize("holds", ["a new vote", "no vote"])
def test_an_export_whose_time_cannot_be_read_is_refused(tmp_path, at, holds):
    """Guards the fix for an unreadable exportedAt: an export
    whose exportedAt has no zone, is a number, is empty or is not a date
    withdrew nothing and said nothing, so votes their voters undid kept
    counting. A refused file writes nothing, a new vote in it included."""
    store, _ = _store_with_outside(tmp_path)
    kept = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    undone = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-bo")
    import_votes(store, _export(tmp_path / "e1.json", [kept, undone]))
    before = (store / store_module.VOTES).read_bytes()
    new = _site_vote("golden-landing-R1", "remix-landing-1", "right", "p-cy")
    later = _export(tmp_path / "e2.json", [kept, new] if holds == "a new vote" else [])
    document = json.loads(later.read_text())
    document["exportedAt"] = at
    later.write_text(json.dumps(document))
    with pytest.raises(PreferenceError, match="exportedAt"):
        import_votes(store, later)
    assert (store / store_module.VOTES).read_bytes() == before
    assert rank(store, bootstrap=0)["pairwise_votes"] == 2


def test_the_same_site_vote_from_another_export_label_is_already_recorded(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    vote = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    import_votes(store, _export(tmp_path / "e1.json", [vote]))
    # The same site, named with a trailing slash by another export tool.
    other = json.loads((tmp_path / "e1.json").read_text())
    other["source"]["site"] = "https://votes.example/"
    (tmp_path / "e2.json").write_text(json.dumps(other))
    counts = import_votes(store, tmp_path / "e2.json")
    assert (counts["added"], counts["already_recorded"]) == (0, 1)


def test_a_vote_changed_within_the_same_second_is_not_a_conflict(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    vote = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    import_votes(store, _export(tmp_path / "e1.json", [vote]))
    # Undo and a new answer in the same second keep the id and the time.
    changed = {**vote, "choice": "right"}
    later = _export(tmp_path / "e2.json", [changed], at="2099-01-05T00:00:00Z")
    assert import_votes(store, later)["added"] == 1
    rows = {r["arm"]: r for r in rank(store, bootstrap=0)["pairwise"]["overall"]}
    golden_arm = next(a for a in rows if a.startswith("golden"))
    assert (rows[golden_arm]["wins"], rows[golden_arm]["losses"]) == (0, 1)


def test_the_cli_reports_a_withdrawal_apart_from_the_votes_added(tmp_path):
    from benchflow.cli.main import app

    store, _ = _store_with_outside(tmp_path)
    kept = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    undone = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-bo")
    import_votes(store, _export(tmp_path / "e1.json", [kept, undone]))
    later = _export(tmp_path / "e2.json", [kept], at="2099-01-05T00:00:00Z")
    runner = CliRunner()
    argv = ["preferences", "import", str(store), "--votes", str(later)]
    dry = runner.invoke(app, [*argv, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    shown = " ".join(dry.output.split())
    assert "Dry run: 0 votes would be added, 1 already recorded" in shown
    assert "1 vote the export no longer has would be recorded as withdrawn" in shown
    done = runner.invoke(app, argv)
    assert done.exit_code == 0, done.output
    shown = " ".join(done.output.split())
    assert "0 votes added, 1 already recorded." in shown
    assert "1 vote the export no longer has is recorded as withdrawn" in shown


def test_the_older_export_name_is_read_and_its_voters_are_people(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    vote = _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")
    del vote["invited"]
    import_votes(
        store, _export(tmp_path / "old.json", [vote], schema="fdb-site-votes/v1")
    )
    logged = json.loads((store / store_module.VOTES).read_text())
    assert logged["rater_kind"] == "person" and "invited" not in logged


def test_an_export_naming_outputs_the_store_lacks(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    votes = [
        _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann"),
        _site_vote("art-legacy-1", "art-legacy-2", "left", "p-ann"),
    ]
    export = _export(tmp_path / "e.json", votes)
    with pytest.raises(PreferenceError, match="art-legacy-1"):
        import_votes(store, export)
    assert not (store / store_module.VOTES).exists()
    counts = import_votes(store, export, skip_unknown=True)
    assert (counts["added"], counts["skipped_unknown"]) == (1, 1)


def test_a_dry_run_checks_everything_and_writes_nothing(tmp_path):
    store, _ = _store_with_outside(tmp_path)
    export = _export(
        tmp_path / "e.json",
        [_site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann")],
    )
    counts = import_votes(store, export, dry_run=True)
    assert counts["added"] == 1 and counts["dry_run"] is True
    assert not (store / store_module.VOTES).exists()


def test_an_export_of_no_votes_adds_nothing_and_is_not_an_error(tmp_path):
    """The live site's export on 2026-11-06 held 0 votes: reading it must say
    so rather than refuse."""
    store, _ = _store_with_outside(tmp_path)
    counts = import_votes(store, _export(tmp_path / "e.json", []))
    assert (counts["added"], counts["already_recorded"]) == (0, 0)


# --- the command line ---------------------------------------------------------


def test_the_cli_takes_outside_items_and_an_export(tmp_path):
    from benchflow.cli.main import app

    runner = CliRunner()
    items = _items(tmp_path, _golden_and_remix(tmp_path))
    made = runner.invoke(
        app,
        [
            "preferences",
            "pairs",
            str(_trials(tmp_path)),
            "--items",
            str(items),
            "--condition",
            "one-shot",
            "--entry",
            "index.html",
            "--golden",
            "side",
            "--store",
            str(tmp_path / "s"),
            "--out",
            str(tmp_path / "a.json"),
            "--seed",
            "3",
        ],
    )
    assert made.exit_code == 0, made.output
    assert "6 pairs of 4 outputs (2 trials, 2 from --items)." in made.output
    export = _export(
        tmp_path / "e.json",
        [
            _site_vote("golden-landing-R1", "remix-landing-1", "left", "p-ann"),
            _site_vote("art-legacy-1", "art-legacy-2", "left", "p-ann"),
        ],
    )
    checked = runner.invoke(
        app,
        [
            "preferences",
            "import",
            str(tmp_path / "s"),
            "--votes",
            str(export),
            "--skip-unknown",
            "--dry-run",
        ],
    )
    assert checked.exit_code == 0, checked.output
    shown = " ".join(checked.output.split())
    assert "Dry run: 1 vote would be added" in shown
    assert "1 vote names an output the store doesn't have" in shown
    assert not (tmp_path / "s" / store_module.VOTES).exists()
    ranked = runner.invoke(
        app,
        ["preferences", "rank", str(tmp_path / "s"), "--by", "condition", "--json"],
    )
    assert ranked.exit_code == 0, ranked.output
    assert json.loads(ranked.output)["by"] == "condition"
