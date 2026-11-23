"""The oracle-parity sample a pull request runs (tools/parity/sample.py).

Gate 1 compares BenchFlow's oracle with Harbor 0.26.0's on 404 tasks; that
takes about 20 task-hours. A pull request runs 12 of them: four fixed
sentinels and eight draws from the cheap tasks CI can fetch without our
storage, one family each, seeded by the tree hash so a rerun of the same tree
draws the same tasks and other trees draw others.
"""

from __future__ import annotations

from pathlib import Path

from tools.parity import sample as S

CORPUS = Path(__file__).resolve().parents[1] / "tools" / "parity" / "corpus-g1.tsv"


def test_the_corpus_holds_only_tasks_ci_can_fetch():
    rows = S.load_corpus(CORPUS)
    assert len(rows) >= 120
    for row in rows:
        assert row.repo.startswith(
            ("https://github.com/", "https://huggingface.co/")
        ), row
        assert len(row.commit) == 40 and row.path, row
        assert "/" in row.key and row.family == row.key.split("/")[0], row


def test_the_same_tree_draws_the_same_tasks():
    rows = S.load_corpus(CORPUS)
    a = S.select(rows, seed="a" * 40, k=12)
    assert a == S.select(rows, seed="a" * 40, k=12)
    assert len(a) == 12 and len(set(a)) == 12
    assert a != S.select(rows, seed="b" * 40, k=12)


def test_the_sentinels_are_always_in():
    rows = S.load_corpus(CORPUS)
    for seed in ("1" * 40, "2" * 40, "3" * 40):
        picked = S.select(rows, seed=seed, k=12)
        assert set(S.SENTINELS) <= set(picked), (seed, picked)


def test_draws_come_from_distinct_families_of_cheap_github_tasks():
    rows = {r.key: r for r in S.load_corpus(CORPUS)}
    for seed in ("4" * 40, "5" * 40, "6" * 40, "7" * 40):
        draws = [
            k for k in S.select(rows.values(), seed=seed, k=12) if k not in S.SENTINELS
        ]
        assert len(draws) == 8
        families = [rows[k].family for k in draws]
        assert len(set(families)) == len(families), families
        for k in draws:
            assert rows[k].repo.startswith("https://github.com/"), k
            assert rows[k].rc4_secs <= S.MAX_SECS, (k, rows[k].rc4_secs)


def test_a_change_in_an_adapter_forces_its_family_in():
    rows = list(S.load_corpus(CORPUS))
    forced = S.forced_families(["src/benchflow/adapters/sldbench.py"], rows)
    assert forced == {"sldbench"}
    picked = S.select(rows, seed="8" * 40, k=12, forced=forced)
    assert any(k.startswith("sldbench/") and k not in S.SENTINELS for k in picked)


def test_a_change_to_grading_widens_the_sample():
    rows = list(S.load_corpus(CORPUS))
    assert S.sample_size(["src/benchflow/task/verifier_core.py"], 12) == 20
    assert S.sample_size(["docs/start/install.md"], 12) == 12
    assert len(S.select(rows, seed="9" * 40, k=20)) == 20


def test_a_task_broken_upstream_is_never_drawn():
    """A task whose image no longer builds a working verifier (an unpinned
    dependency moved; reasoning-gym's drawsvg on 2026-11-18) says nothing
    about BenchFlow: the ledger excludes it, with its evidence, until it is fixed."""
    rows = list(S.load_corpus(CORPUS))
    excluded = S.load_excluded()
    assert (
        "reasoning-gym/reasoning-gym-algorithmic-letter-jumble-hard0292e21f-seed45-2"
        in excluded
    )
    for seed in ("a" * 40, "b" * 40, "c" * 40, "d" * 40, "e" * 40):
        assert not set(S.select(rows, seed=seed, k=40)) & set(excluded)
