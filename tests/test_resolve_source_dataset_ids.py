"""`bf.resolve_source` takes BenchFlow's dataset IDs as well as `org/repo`.

``resolve_source("terminal-bench-2", path="log-summary-date-ranges")`` reads
the repository, pin and folder BenchFlow's ID table holds, so docs and
scripts name the dataset, not where its upstream copy lives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br


@pytest.fixture
def clones(monkeypatch, tmp_path):
    calls: list[tuple] = []

    def fake_clone(org, repo, ref, path):
        calls.append((f"{org}/{repo}", ref, path))
        return tmp_path

    monkeypatch.setattr(br, "_clone_repo", fake_clone)
    monkeypatch.setattr(br, "_resolve_repo_path", lambda root, path, label: root / path)
    return calls


def test_terminal_bench_2_is_a_dataset_id(clones, tmp_path):
    got = br.resolve_source("terminal-bench-2", path="log-summary-date-ranges")
    assert got == tmp_path / "log-summary-date-ranges"
    repo, ref, _folder = br.TASK_ALIASES["terminal-bench-2"]
    assert clones == [(repo, ref, "log-summary-date-ranges")]
    # Pinned to the same commit as BenchFlow's registry entry.
    assert ref == "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"


def test_a_dataset_id_folder_prefixes_the_path(clones, tmp_path):
    got = br.resolve_source("skillsbench", path="edit-pdf")
    assert got == tmp_path / "tasks" / "edit-pdf"
    assert clones == [("benchflow-ai/skillsbench", "main", "tasks/edit-pdf")]


def test_an_explicit_ref_overrides_the_pin(clones):
    br.resolve_source("terminal-bench-2", path="x", ref="main")
    assert clones[0][1] == "main"


def test_org_repo_is_unchanged(clones, tmp_path):
    br.resolve_source("acme/bench", path="tasks/a")
    assert clones == [("acme/bench", None, "tasks/a")]


def test_an_unknown_bare_name_names_both_forms():
    with pytest.raises(ValueError, match="or a BenchFlow dataset ID"):
        br.resolve_source("no-such-dataset")


def test_the_id_table_has_no_path_outside_its_repo():
    for org_repo, _ref, folder in br.TASK_ALIASES.values():
        assert org_repo.count("/") == 1
        assert folder is None or not Path(folder).is_absolute()
