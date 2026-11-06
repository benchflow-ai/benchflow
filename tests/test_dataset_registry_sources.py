"""`-d name@version` reads BenchFlow's registry and nothing else.

BenchFlow's registry is its published list (``registry.json`` in the
skillsbench repo) and the entries the package ships for datasets that list
does not hold (Terminal-Bench 2.0, every task pinned by commit and digest).
A dataset it lacks is an error that names BenchFlow's registry and says how
to name another one; no other registry is read on its own. A registry named
with ``--registry`` is read alone: tasks it gives no digest for run
unchecked and the run says so, and a ``HEAD`` commit runs the default
branch.
"""

from __future__ import annotations

import io
import json
import re
from hashlib import sha1
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchflow._utils import dataset_registry
from benchflow._utils.dataset_registry import (
    DEFAULT_REGISTRY_SOURCE,
    PACKAGED_REGISTRY_SOURCE,
    DatasetDigestMismatchError,
    DatasetResolutionError,
    dataset_registry_entry,
    resolve_dataset,
)
from benchflow._utils.task_authoring import task_digest

COMMIT = "c" * 40
# Terminal-Bench 2.0 as BenchFlow pins it (the commit its keep-services list
# and the round's controls use).
TB2_COMMIT = "69671fbaac6d67a7ef0dfec016cc38a64ef7a77c"
TB2_GIT_URL = "https://github.com/laude-institute/terminal-bench-2.git"
BENCHFLOW_ENTRIES = [
    {
        "name": "skillsbench",
        "version": "1.1",
        "tasks": [
            {
                "name": "x",
                "git_url": "https://github.com/benchflow-ai/skillsbench.git",
                "git_commit_id": "a" * 40,
                "path": "tasks/x",
                "digest": "sha256:" + "0" * 64,
            }
        ],
    }
]


def _unpinned_entry(name: str, version: str, commit: str = COMMIT) -> dict:
    # A registry entry that pins no digest, tasks at the repository root.
    return {
        "name": name,
        "version": version,
        "description": "another benchmark",
        "tasks": [
            {
                "name": task,
                "git_url": "https://github.com/acme/other-bench.git",
                "git_commit_id": commit,
                "path": task,
            }
            for task in ("task-a", "task-b")
        ],
        "metrics": [],
    }


class FakeRegistries:
    """Serves registry URLs; records which were read."""

    def __init__(self, served: dict[str, object]) -> None:
        self.served = served
        self.read: list[str] = []

    def __call__(self, url: str, timeout: float = 30):
        self.read.append(url)
        body = self.served.get(url)
        if body is None:
            raise OSError("HTTP Error 503: unavailable")
        return io.BytesIO(json.dumps(body).encode())


def _make_task(parent: Path, name: str) -> None:
    task = parent / name
    (task / "tests").mkdir(parents=True)
    (task / "task.toml").write_text('version = "1.1"\n')
    (task / "instruction.md").write_text(f"Solve {name}.\n")
    (task / "tests" / "test.sh").write_text("exit 0\n")


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    _make_task(root, "task-a")
    _make_task(root, "task-b")
    calls: dict = {}

    def fake_resolve(repo, path=None, ref=None):
        calls.update(repo=repo, path=path, ref=ref)
        return SimpleNamespace(
            path=root,
            provenance={"type": "github", "repo": repo, "resolved_sha": COMMIT},
        )

    monkeypatch.setattr(
        "benchflow._utils.benchmark_repos.resolve_source_with_metadata", fake_resolve
    )
    trees = {}
    for task in root.iterdir():
        records = []
        for path in sorted(task.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                oid = sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
                records.append(
                    f"100644 blob {oid}\t{path.relative_to(task).as_posix()}\0"
                )
        trees[task] = "".join(records)

    def fake_tree(candidate, *args):
        if args[0] == "rev-parse":
            return None  # This fixture has no Git object store for conversions.
        assert args == (
            "--no-replace-objects",
            "ls-tree",
            "-r",
            "-z",
            COMMIT,
            "--",
            ".",
        )
        return trees[candidate]

    monkeypatch.setattr("benchflow._utils.benchmark_repos._git_stdout", fake_tree)
    return SimpleNamespace(root=root, calls=calls)


def _serve(monkeypatch, served: dict[str, object]) -> FakeRegistries:
    fake = FakeRegistries(served)
    monkeypatch.setattr(dataset_registry, "urlopen", fake)
    return fake


def _packaged(tmp_path: Path, monkeypatch, entries: list[dict]) -> str:
    """Stand in for the packaged entries file; returns its path."""
    path = tmp_path / "packaged.json"
    path.write_text(json.dumps(entries))
    monkeypatch.setattr(
        dataset_registry,
        "DEFAULT_REGISTRY_SOURCES",
        (DEFAULT_REGISTRY_SOURCE, str(path)),
    )
    return str(path)


def test_benchflows_registry_is_its_list_and_its_packaged_entries():
    assert dataset_registry.DEFAULT_REGISTRY_SOURCES == (
        DEFAULT_REGISTRY_SOURCE,
        PACKAGED_REGISTRY_SOURCE,
    )
    assert Path(PACKAGED_REGISTRY_SOURCE).is_file()
    assert dataset_registry.registry_label(PACKAGED_REGISTRY_SOURCE) == (
        "BenchFlow's registry"
    )


def test_a_dataset_benchflows_registry_lacks_is_an_error_naming_it(
    snapshot, monkeypatch
):
    fake = _serve(monkeypatch, {DEFAULT_REGISTRY_SOURCE: BENCHFLOW_ENTRIES})
    with pytest.raises(DatasetResolutionError) as caught:
        resolve_dataset("other-bench@2.0")
    assert str(caught.value) == (
        "Dataset 'other-bench@2.0' not found in BenchFlow's registry"
    )
    assert "--registry" in (caught.value.hint or "")
    # Only BenchFlow's published list goes over the network; nothing else
    # is tried and nothing is cloned.
    assert fake.read == [DEFAULT_REGISTRY_SOURCE]
    assert snapshot.calls == {}
    with pytest.raises(DatasetResolutionError, match="BenchFlow's registry"):
        dataset_registry_entry("other-bench@2.0")


def test_a_packaged_entry_resolves_digest_checked(tmp_path, snapshot, monkeypatch):
    pinned = {
        "name": "other-bench",
        "version": "2.0",
        "tasks": [
            {
                "name": name,
                "git_url": "https://github.com/acme/other-bench.git",
                "git_commit_id": COMMIT,
                "path": name,
                "digest": task_digest(snapshot.root / name),
            }
            for name in ("task-a", "task-b")
        ],
    }
    packaged = _packaged(tmp_path, monkeypatch, [pinned])
    fake = _serve(monkeypatch, {DEFAULT_REGISTRY_SOURCE: BENCHFLOW_ENTRIES})
    resolved = resolve_dataset("other-bench@2.0")
    assert fake.read == [DEFAULT_REGISTRY_SOURCE]
    assert resolved.registry == packaged
    assert set(resolved.task_digests) == {"task-a", "task-b"}
    assert resolved.unpinned_tasks == set()
    assert snapshot.calls == {"repo": "acme/other-bench", "path": None, "ref": COMMIT}
    (snapshot.root / "task-a" / "instruction.md").write_text("tampered\n")
    with pytest.raises(DatasetDigestMismatchError, match="task-a"):
        resolve_dataset("other-bench@2.0")


def test_the_published_list_wins_over_a_packaged_entry(tmp_path, snapshot, monkeypatch):
    _packaged(tmp_path, monkeypatch, [_unpinned_entry("skillsbench", "1.1")])
    _serve(monkeypatch, {DEFAULT_REGISTRY_SOURCE: BENCHFLOW_ENTRIES})
    entry = dataset_registry_entry("skillsbench@1.1")
    assert entry["tasks"][0]["path"] == "tasks/x"


def test_packaged_entries_need_no_network(tmp_path, snapshot, monkeypatch):
    _packaged(tmp_path, monkeypatch, [_unpinned_entry("other-bench", "2.0")])
    _serve(monkeypatch, {})  # the published list is unreachable
    entry = dataset_registry_entry("other-bench@2.0")
    assert entry["name"] == "other-bench"
    with pytest.raises(DatasetResolutionError) as caught:
        dataset_registry_entry("skillsbench@1.1")
    text = str(caught.value)
    assert text.startswith(
        "Dataset 'skillsbench@1.1' could not be looked up in BenchFlow's registry"
    )
    assert "HTTP Error 503" in text
    assert "network" in (caught.value.hint or "")


def test_a_missing_version_lists_the_versions_benchflow_has(snapshot, monkeypatch):
    _serve(monkeypatch, {DEFAULT_REGISTRY_SOURCE: BENCHFLOW_ENTRIES})
    with pytest.raises(DatasetResolutionError) as caught:
        resolve_dataset("skillsbench@9.9")
    assert "not found in BenchFlow's registry (its versions there: 1.1)" in str(
        caught.value
    )


def test_a_head_commit_runs_the_default_branch(tmp_path, snapshot, monkeypatch):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([_unpinned_entry("ds", "head", commit="HEAD")]))
    resolved = resolve_dataset("ds@head", registry=str(registry))
    assert snapshot.calls["ref"] == "HEAD"
    assert resolved.provenance["resolved_sha"] == COMMIT


def test_the_packaged_terminal_bench_entry_pins_every_task():
    entries = json.loads(Path(PACKAGED_REGISTRY_SOURCE).read_text())
    (entry,) = [e for e in entries if e["name"] == "terminal-bench"]
    assert entry["version"] == "2.0"
    tasks = entry["tasks"]
    assert len(tasks) == 89
    assert len({t["name"] for t in tasks}) == 89
    for task in tasks:
        assert task["git_url"] == TB2_GIT_URL
        assert task["git_commit_id"] == TB2_COMMIT
        assert task["path"] == task["name"]
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", task["digest"])
    # The tasks BenchFlow's keep-services list pins are the same folders.
    keep = json.loads(
        (
            Path(dataset_registry.__file__).parents[1]
            / "task"
            / "tb2_keep_services.json"
        ).read_text()
    )
    assert keep["dataset"] == "terminal-bench-2"
    names = {t["name"] for t in tasks}
    assert names >= {t["task"] for t in keep["tasks"]}


def test_a_named_registry_is_the_only_one_read(tmp_path, snapshot, monkeypatch):
    fake = _serve(monkeypatch, {})
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([_unpinned_entry("other-bench", "2.0")]))
    resolved = resolve_dataset("other-bench@2.0", registry=str(registry))
    assert fake.read == []
    assert resolved.registry == str(registry)
    assert resolved.unpinned_tasks == {"task-a", "task-b"}
    with pytest.raises(DatasetResolutionError, match="not found in registry"):
        resolve_dataset("skillsbench@1.1", registry=str(registry))


def test_pinned_tasks_are_still_checked_beside_unpinned_ones(
    tmp_path, snapshot, monkeypatch
):
    entry = _unpinned_entry("mixed", "1.0")
    entry["tasks"][0]["digest"] = task_digest(snapshot.root / "task-a")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps([entry]))
    resolved = resolve_dataset("mixed@1.0", registry=str(registry))
    assert resolved.task_digests == {"task-a": entry["tasks"][0]["digest"]}
    assert resolved.unpinned_tasks == {"task-b"}
    (snapshot.root / "task-a" / "instruction.md").write_text("tampered\n")
    with pytest.raises(DatasetDigestMismatchError, match="task-a"):
        resolve_dataset("mixed@1.0", registry=str(registry))
    # An unpinned task that is missing from the checkout is still an error.
    entry["tasks"][0].pop("digest")
    entry["tasks"].append(dict(entry["tasks"][1], name="task-c", path="task-c"))
    registry.write_text(json.dumps([entry]))
    with pytest.raises(DatasetDigestMismatchError, match="task-c: missing"):
        resolve_dataset("mixed@1.0", registry=str(registry))
