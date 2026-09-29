"""The train/test split of ``bench hillclimb``: seeded, stratified, or from a file."""

from __future__ import annotations

import json

import pytest

from benchflow.hillclimbing.split import (
    SplitError,
    load_split_file,
    make_split,
    task_strata,
)
from tests._hillclimb_fakes import make_tasks


def _strata(n_a: int, n_b: int) -> dict[str, str]:
    out = {f"a{i}": "a" for i in range(n_a)}
    out.update({f"b{i}": "b" for i in range(n_b)})
    return out


def test_the_same_seed_gives_the_same_split_and_another_seed_another():
    strata = _strata(10, 10)
    one = make_split(strata, test_frac=0.3, seed=7, stratify_by="category")
    two = make_split(strata, test_frac=0.3, seed=7, stratify_by="category")
    other = make_split(strata, test_frac=0.3, seed=8, stratify_by="category")
    assert one == two
    assert one.test != other.test


def test_a_split_is_a_partition_with_the_requested_size():
    strata = _strata(7, 5)
    split = make_split(strata, test_frac=0.25, seed=0, stratify_by="category")
    assert sorted(split.train + split.test) == sorted(strata)
    assert not set(split.train) & set(split.test)
    assert len(split.test) == 3  # round(12 * 0.25)


def test_each_stratum_is_represented_in_proportion():
    strata = _strata(20, 10)
    split = make_split(strata, test_frac=0.3, seed=3, stratify_by="category")
    test_a = sum(1 for t in split.test if t.startswith("a"))
    test_b = sum(1 for t in split.test if t.startswith("b"))
    assert (test_a, test_b) == (6, 3)
    assert split.method == "stratified" and split.stratify_by == "category"
    assert split.strata["a0"] == "a"


def test_without_strata_the_split_is_plain_random():
    split = make_split(dict.fromkeys(["x", "y", "z", "w"]), test_frac=0.5, seed=1)
    assert split.method == "random" and split.stratify_by is None
    assert len(split.test) == 2


def test_a_tiny_task_set_still_gets_one_task_on_each_side():
    split = make_split(dict.fromkeys(["x", "y"]), test_frac=0.1, seed=0)
    assert len(split.train) == 1 and len(split.test) == 1
    with pytest.raises(SplitError, match="at least 2 tasks"):
        make_split({"x": None}, test_frac=0.3, seed=0)


def test_a_split_file_must_place_exactly_the_tasks(tmp_path):
    path = tmp_path / "split.json"
    path.write_text(json.dumps({"train": ["a", "b"], "test": ["c"]}))
    split = load_split_file(path, ["a", "b", "c"])
    assert split.train == ("a", "b") and split.test == ("c",)
    assert split.method == "file" and split.source == str(path)
    with pytest.raises(SplitError, match="does not place"):
        load_split_file(path, ["a", "b", "c", "d"])
    with pytest.raises(SplitError, match="not in the task set"):
        load_split_file(path, ["a", "b"])
    path.write_text(json.dumps({"train": ["a", "c"], "test": ["c", "b"]}))
    with pytest.raises(SplitError, match="both train and test"):
        load_split_file(path, ["a", "b", "c"])


def test_a_saved_split_reads_back_as_a_split_file(tmp_path):
    split = make_split(_strata(4, 4), test_frac=0.5, seed=2, stratify_by="category")
    saved = split.save(tmp_path / "split.json")
    again = load_split_file(saved, [*split.train, *split.test])
    assert (again.train, again.test) == (split.train, split.test)


def test_strata_come_from_task_metadata(tmp_path):
    root = make_tasks(
        tmp_path, ["p", "q", "r"], category=lambda i: "hydro" if i else ""
    )
    dirs = {d.name: d for d in root.iterdir()}
    assert task_strata(dirs, "category") == {"p": None, "q": "hydro", "r": "hydro"}
    assert task_strata(dirs, None) == {"p": None, "q": None, "r": None}
