"""A path-scoped source fetches that path only.

Guards the dx/first-run sparse source fetch. Before it, `bench eval run
--source-repo benchflow-ai/skillsbench --source-path tasks/citation-check`
cloned the whole repository to run one task: on 2026-09-30 that was a
1.1 GB checkout (a 447 MB pack) plus a 644 MB per-commit snapshot, about
21 s on a GCP VM. These tests run real git against a local repository served
over ``file://``, and check which file contents were fetched at all.
"""

from __future__ import annotations

import os
import re
import select
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path

import pytest

from benchflow._utils import benchmark_repos as br

_IDENTITY = ("-c", "user.name=BenchFlow Test", "-c", "user.email=test@example.com")


def _git_version() -> tuple[int, int]:
    # Runs at import time for the module-wide skip below, so a machine with no
    # git has to skip, not fail collection (the production twin
    # br._git_can_sparse guards the same call the same way).
    try:
        out = subprocess.run(["git", "version"], capture_output=True, text=True).stdout
    except (OSError, ValueError):
        return (0, 0)
    match = re.search(r"(\d+)\.(\d+)", out)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


# `git init -b` needs 2.28 and `sparse-checkout add` 2.26.
pytestmark = pytest.mark.skipif(
    _git_version() < (2, 28), reason="needs git 2.28 or newer"
)


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _remote(tmp_path: Path) -> Path:
    """A bare repository with two tasks, docs, a non-task folder and a symlink."""
    src = tmp_path / "src"
    _write(src / "README.md", "# bench\n")
    _write(src / "tasks/citation-check/task.md", "---\n---\nCheck the citations.\n")
    _write(src / "tasks/citation-check/environment/Dockerfile", "FROM ubuntu:24.04\n")
    _write(src / "tasks/citation-check/verifier/test.sh", "exit 0\n")
    _write(src / "tasks/court-form/task.md", "---\n---\nFill the form.\n")
    _write(src / "tasks/court-form/big.bin", "x" * 100_000)
    _write(src / "docs/guide.md", "guide\n")
    _write(src / "data/foreign/items.csv", "id,prompt\n1,hi\n")
    (src / "link").symlink_to("tasks", target_is_directory=True)
    _git(tmp_path, "init", "-q", "-b", "main", str(src))
    _git(src, "add", "-A")
    _git(src, *_IDENTITY, "commit", "-q", "-m", "one")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(bare))
    _git(bare, "config", "uploadpack.allowFilter", "true")
    _git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")
    return bare


@pytest.fixture
def bench(tmp_path, monkeypatch) -> Path:
    """Serve the repository as acme/bench; return the cache checkout's path."""
    bare = _remote(tmp_path)
    monkeypatch.setattr(br, "_repo_url", lambda org, repo: f"file://{bare}")
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path / "cache")
    return tmp_path / "cache" / "acme" / "bench"


def _missing_blobs(repo: Path) -> set[str]:
    """Objects HEAD needs that were never fetched (listing them fetches nothing)."""
    listing = _git(repo, "rev-list", "--objects", "--missing=print", "HEAD")
    return {line[1:] for line in listing.splitlines() if line.startswith("?")}


def _blob(repo: Path, path: str) -> str:
    return _git(repo, "ls-tree", "HEAD", "--", path).split()[2]


def test_source_path_fetches_and_checks_out_only_that_path(bench):
    resolved = br.resolve_source_with_metadata(
        "acme/bench", path="tasks/citation-check"
    )

    snapshot = resolved.path.parents[1]
    assert (resolved.path / "task.md").read_text().endswith("Check the citations.\n")
    for checkout in (bench, snapshot):
        assert (checkout / "tasks/citation-check/environment/Dockerfile").is_file()
        assert not (checkout / "tasks/court-form").exists()
        assert not (checkout / "docs").exists()
    missing = _missing_blobs(bench)
    assert _blob(bench, "tasks/court-form/big.bin") in missing
    assert _blob(bench, "docs/guide.md") in missing
    assert _blob(bench, "tasks/citation-check/task.md") not in missing
    assert resolved.provenance["path"] == "tasks/citation-check"
    assert resolved.provenance["dirty"] is False
    assert set(resolved.provenance["file_hashes"]) == {
        "task.md",
        "environment/Dockerfile",
        "verifier/test.sh",
    }


def test_a_second_path_joins_the_same_checkout_and_snapshot(bench):
    first = br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")
    second = br.resolve_source_with_metadata("acme/bench", path="tasks/court-form")

    assert first.path.parents[1] == second.path.parents[1]
    assert (first.path / "task.md").is_file()
    assert (second.path / "big.bin").stat().st_size == 100_000
    assert (bench / "tasks/court-form/big.bin").is_file()
    assert not (second.path.parents[1] / "docs").exists()
    assert _blob(bench, "docs/guide.md") in _missing_blobs(bench)


def test_plain_resolve_source_is_sparse_too(bench):
    path = br.resolve_source("acme/bench", path="tasks/court-form")

    assert path == bench / "tasks" / "court-form"
    assert (path / "task.md").is_file()
    assert not (bench / "tasks/citation-check").exists()


def test_a_request_for_the_whole_repository_turns_sparse_off(bench):
    br.resolve_source("acme/bench", path="tasks/citation-check")

    root = br.resolve_source("acme/bench")

    assert root == bench
    assert (root / "docs/guide.md").read_text() == "guide\n"
    assert (root / "tasks/court-form/big.bin").is_file()
    assert not br._is_sparse_checkout(root)


def test_a_folder_without_benchflow_tasks_gets_the_whole_repository(bench):
    """Foreign benchmarks keep working: their source adapters (Toolathlon,
    MCP-Atlas) read files outside the path, such as db/ or configs/."""
    path = br.resolve_source("acme/bench", path="data/foreign")

    assert (path / "items.csv").is_file()
    assert (bench / "docs/guide.md").is_file()
    assert not br._is_sparse_checkout(bench)


def test_a_path_through_a_symlink_gets_the_whole_repository(bench):
    resolved = br.resolve_source_with_metadata("acme/bench", path="link/court-form")

    assert resolved.provenance["path"] == "tasks/court-form"
    assert (resolved.path / "big.bin").is_file()
    assert not br._is_sparse_checkout(bench)


def test_a_missing_path_lists_its_parent_and_suggests_a_match(bench):
    with pytest.raises(FileNotFoundError) as caught:
        br.resolve_source_with_metadata("acme/bench", path="tasks/citaton-check")

    message = str(caught.value)
    assert message == (
        "Path 'tasks/citaton-check' not found in acme/bench. Did you mean "
        "'tasks/citation-check'? Available: ['tasks/citation-check', "
        "'tasks/court-form']"
    )
    # Reporting the typo fetched no task's files.
    assert _blob(bench, "tasks/court-form/big.bin") in _missing_blobs(bench)


def test_a_file_path_is_refused_as_not_a_directory(bench):
    with pytest.raises(ValueError, match="must resolve to a directory"):
        br.resolve_source_with_metadata("acme/bench", path="docs/guide.md")


def test_a_commit_ref_checks_out_that_commit(bench, tmp_path):
    src = tmp_path / "src"
    first = _git(src, "rev-parse", "HEAD")
    _write(src / "tasks/court-form/task.md", "---\n---\nFill the new form.\n")
    _git(src, *_IDENTITY, "commit", "-qam", "two")
    _git(src, "push", "-q", str(tmp_path / "remote.git"), "main")

    resolved = br.resolve_source_with_metadata(
        "acme/bench", path="tasks/court-form", ref=first
    )

    assert resolved.provenance["resolved_sha"] == first
    assert (resolved.path / "task.md").read_text().endswith("Fill the form.\n")
    assert not (resolved.path.parents[1] / "tasks/citation-check").exists()


def test_a_full_clone_made_before_sparse_fetches_is_reused_as_is(bench, tmp_path):
    bench.parent.mkdir(parents=True)
    _git(tmp_path, "clone", "-q", f"file://{tmp_path / 'remote.git'}", str(bench))

    resolved = br.resolve_source_with_metadata("acme/bench", path="tasks/court-form")

    assert (resolved.path / "big.bin").is_file()
    assert (resolved.path.parents[1] / "docs/guide.md").is_file()
    assert not br._is_sparse_checkout(bench)


def test_the_fetch_writes_nothing_to_a_pipe(bench, capfd):
    """Nothing reaches stdout or stderr when they are pipes (CI logs). Git
    shows fetch progress only on a terminal; the pseudo-terminal test below
    covers that."""
    capfd.readouterr()

    br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")
    br.resolve_source_with_metadata("acme/bench", path="tasks/court-form")

    out, err = capfd.readouterr()
    assert out == ""
    assert err == ""


def test_git_older_than_2_26_clones_the_whole_repository(bench, monkeypatch):
    """git before 2.26 has no `sparse-checkout add`: keep the full clone."""
    monkeypatch.setattr(br, "_git_can_sparse", lambda: False)

    path = br.resolve_source("acme/bench", path="tasks/court-form")

    assert (path / "big.bin").is_file()
    assert (bench / "docs/guide.md").is_file()
    assert not br._is_sparse_checkout(bench)
    assert not _missing_blobs(bench)


@pytest.mark.parametrize(
    ("version", "can_sparse"),
    [
        ("git version 2.25.1\n", False),
        ("git version 2.26.0\n", True),
        ("git version 2.34.1\n", True),
        ("git version 2.50.1 (Apple Git-155)\n", True),
        ("", True),
    ],
)
def test_git_can_sparse_reads_the_version(monkeypatch, version, can_sparse):
    def fake_run(cmd, **kwargs):
        assert cmd == ["git", "version"]
        return subprocess.CompletedProcess(cmd, 0, stdout=version, stderr="")

    br._git_can_sparse.cache_clear()
    monkeypatch.setattr(br.subprocess, "run", fake_run)
    try:
        assert br._git_can_sparse() is can_sparse
    finally:
        br._git_can_sparse.cache_clear()


def _read_until_exit(pid: int, fd: int, timeout: float = 120.0) -> tuple[int, bytes]:
    """The child's terminal output and exit code; kill it past the timeout."""
    output = b""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail(f"no exit within {timeout:.0f} s: {output!r}")
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            continue
        try:
            chunk = os.read(fd, 4096)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), output


def _read_until_exit_closing(pid: int, fd: int) -> tuple[int, bytes]:
    """`_read_until_exit`, always closing the pty master it was handed."""
    try:
        return _read_until_exit(pid, fd)
    finally:
        with suppress(OSError):
            os.close(fd)


@pytest.mark.skipif(sys.platform == "win32", reason="needs a pseudo-terminal")
@pytest.mark.parametrize("ref", [None, "main"])
def test_the_fetch_prints_no_progress_on_a_terminal(bench, tmp_path, ref):
    """git shows an on-demand blob fetch's progress ("Receiving objects: ...")
    when stderr is a terminal, whatever the command's own flags: in the
    2026-09-30 first-run walk it printed two progress blocks before the
    dashboard. Run resolves on a pseudo-terminal, then, as a control, one
    unquieted git command that fetches, and check only the control shows it.
    With a ref, the cache is warm and the ref moved, so the ref's checkout
    fetches the changed file on demand too."""
    import pty

    if ref:
        br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")
        src = tmp_path / "src"
        _write(src / "tasks/citation-check/task.md", "---\n---\nCheck them all.\n")
        _git(src, *_IDENTITY, "commit", "-qam", "two")
        _git(src, "push", "-q", str(tmp_path / "remote.git"), "main")

    pid, fd = pty.fork()
    if pid == 0:  # the child: stdout and stderr are the terminal
        code = 1
        try:
            os.write(2, b"tty=%d\n" % os.isatty(2))
            for path in ("tasks/citation-check", "tasks/court-form"):
                br.resolve_source_with_metadata("acme/bench", path=path, ref=ref)
            os.write(2, b"CONTROL\n")
            subprocess.run(["git", "-C", str(bench), "sparse-checkout", "add", "docs"])
            code = 0
        except BaseException:
            # os._exit below discards the traceback, and this child does a
            # fetch, a detached checkout and a read-tree: without this a
            # failure reads as a bare code=1.
            with suppress(OSError):
                os.write(2, traceback.format_exc().encode("utf-8", "replace"))
        finally:
            os._exit(code)
    code, output = _read_until_exit_closing(pid, fd)
    assert code == 0, output
    ours, _, control = output.partition(b"CONTROL")
    assert b"tty=1" in ours
    if b"Receiving objects" not in control:
        pytest.skip("this git prints no fetch progress on a terminal")
    assert b"Receiving objects" not in ours
    assert b"remote:" not in ours
    if ref:
        task_md = (bench / "tasks/citation-check/task.md").read_text()
        assert task_md.endswith("Check them all.\n")


def test_dot_is_the_whole_repository(bench):
    """`--source-path .` names the repository root, as before sparse fetches."""
    assert br.resolve_source("acme/bench", path=".") == bench
    assert (bench / "docs/guide.md").is_file()
    assert not br._is_sparse_checkout(bench)


def test_dot_on_a_sparse_cache_checks_out_everything(bench):
    br.resolve_source("acme/bench", path="tasks/citation-check")

    root = br.resolve_source("acme/bench", path="./")

    assert (root / "tasks/court-form/big.bin").is_file()
    assert not br._is_sparse_checkout(root)


def test_a_path_through_a_skipped_folder_resolves(bench):
    """`docs/../tasks/x` names tasks/x; docs/ is not on disk in a sparse checkout."""
    resolved = br.resolve_source_with_metadata(
        "acme/bench", path="docs/../tasks/court-form"
    )

    assert resolved.provenance["path"] == "tasks/court-form"
    assert (resolved.path / "big.bin").is_file()
    assert not (resolved.path.parents[1] / "docs").exists()


def test_a_foreign_folder_widens_an_existing_sparse_snapshot(bench):
    first = br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")
    snapshot = first.path.parents[1]
    assert not (snapshot / "docs").exists()

    foreign = br.resolve_source_with_metadata("acme/bench", path="data/foreign")

    assert foreign.path == snapshot / "data" / "foreign"
    assert (snapshot / "docs/guide.md").is_file()
    assert not br._is_sparse_checkout(snapshot)
    assert foreign.provenance["dirty"] is False


def test_a_foreign_folder_under_an_already_fetched_pattern_still_widens(
    tmp_path, monkeypatch
):
    """A foreign benchmark needs the whole repository even when already fetched.

    Regression guard. The "leave an already fetched path alone" check ran
    before the native-task verdict, so a source that first fetched `tasks`
    (native, stays sparse) and then asked for `tasks/harbor` matched the
    existing `tasks` pattern, returned early, and never widened: the foreign
    adapter's files outside the path (here `configs/`) stayed absent and the
    adapter failed on a missing file. Only the two calls that write to the
    working tree may be skipped, not the verdict.
    """
    src = tmp_path / "src"
    _write(src / "tasks/citation-check/task.md", "---\n---\nCheck the citations.\n")
    _write(src / "tasks/harbor/config.yaml", "name: harbor\n")
    _write(src / "configs/harbor.yaml", "rows: 3\n")
    _git(tmp_path, "init", "-q", "-b", "main", str(src))
    _git(src, "add", "-A")
    _git(src, *_IDENTITY, "commit", "-q", "-m", "one")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(bare))
    _git(bare, "config", "uploadpack.allowFilter", "true")
    _git(bare, "config", "uploadpack.allowAnySHA1InWant", "true")
    monkeypatch.setattr(br, "_repo_url", lambda org, repo: f"file://{bare}")
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path / "cache")
    checkout = tmp_path / "cache" / "acme" / "mixed"

    native = br.resolve_source_with_metadata("acme/mixed", path="tasks")
    snapshot = native.path.parent
    assert br._is_sparse_checkout(checkout)
    assert not (checkout / "configs").exists()

    foreign = br.resolve_source_with_metadata("acme/mixed", path="tasks/harbor")

    assert foreign.path == snapshot / "tasks" / "harbor"
    assert (checkout / "configs/harbor.yaml").is_file()
    assert (snapshot / "configs/harbor.yaml").is_file()
    assert not br._is_sparse_checkout(checkout)


def test_a_path_git_would_quote_is_found(tmp_path, monkeypatch):
    """Non-ASCII names come out of `git ls-tree` quoted unless -z is used."""
    src = tmp_path / "src"
    _write(src / "tasks/café/task.md", "---\n---\nOrder a coffee.\n")
    _write(src / "docs/guide.md", "guide\n")
    _git(tmp_path, "init", "-q", "-b", "main", str(src))
    _git(src, "add", "-A")
    _git(src, *_IDENTITY, "commit", "-q", "-m", "one")
    bare = tmp_path / "remote.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(bare))
    _git(bare, "config", "uploadpack.allowFilter", "true")
    monkeypatch.setattr(br, "_repo_url", lambda org, repo: f"file://{bare}")
    monkeypatch.setattr(br, "_cache_dir", lambda: tmp_path / "cache")

    resolved = br.resolve_source_with_metadata("acme/cafe", path="tasks/café")

    assert (resolved.path / "task.md").read_text().endswith("Order a coffee.\n")
    assert not (resolved.path.parents[1] / "docs").exists()


def test_a_path_already_fetched_leaves_the_checkout_alone(bench, monkeypatch):
    """A resolve of a fetched path runs no sparse-checkout or read-tree: another
    process may be reading that checkout."""
    br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")
    ran: list[tuple[str, ...]] = []
    monkeypatch.setattr(br, "_git_quiet", lambda root, *args: ran.append(args))

    br.resolve_source_with_metadata("acme/bench", path="tasks/citation-check")

    assert ran == []


def test_a_long_listing_is_cut_and_counted(tmp_path):
    """SkillsBench's tasks/ holds 87 folders; the message lists 12 of them."""
    root = tmp_path / "repo"
    for index in range(15):
        (root / "tasks" / f"task-{index:02d}").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as caught:
        br._resolve_repo_path(root, "tasks/nothing-like-it", "acme/bench")

    message = str(caught.value)
    assert message.startswith(
        "Path 'tasks/nothing-like-it' not found in acme/bench. Available: "
        "['tasks/task-00', "
    )
    assert "'tasks/task-11']" in message
    assert message.endswith(" and 3 more")
