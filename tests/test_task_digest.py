"""Tests for `bench tasks digest` and task_digest().

The digest pins a task's content independent of git and must byte-match the
reference digests published in the skillsbench dataset registry
(registry.json / docs/dataset-versioning.md, skillsbench PR #922):

    task_digest = sha256( for each regular file under the task directory,
                          sorted by POSIX relative path:
                            update( path_utf8 + b"\\x00"
                                    + sha256(file_bytes).digest() ) )

Symlinks and file modes are excluded; the hex digest is prefixed "sha256:".

The implementation was cross-validated against all 87 published digests of
the skillsbench v1.1 registry entry (git tag v1.1, commit 14f33967) — e.g.
tasks/3d-scan-calc reproduces
sha256:587a79f13a31098b5f7b4d1f99582ac6d4184e2cca1f02e2679fc98b81861220.
The golden values below freeze the same algorithm onto an offline fixture.
"""

import hashlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow._utils.task_authoring import task_digest
from benchflow.cli.main import app

FIXTURES = Path(__file__).parent / "fixtures" / "digest"

# Frozen against the checked-in fixture tasks. If these fail without a
# deliberate fixture change, the algorithm drifted from the published
# skillsbench registry digests — fix the code, not the constants.
ALPHA_DIGEST = "sha256:6a0c023a27c1dd1b68b16ed23a41cdd9b3ea3401599bc6abbb95ed77d41e08f9"
BETA_DIGEST = "sha256:8f5a4f868cef8c5c50b700183d0b9bd7405d92245eed79a0d50494b7daa0de16"


def _spec_digest(file_entries: list[tuple[str, bytes]]) -> str:
    """The registry construction, written out literally, in the given order."""
    h = hashlib.sha256()
    for rel_posix, content in file_entries:
        h.update(rel_posix.encode("utf-8"))
        h.update(b"\x00")
        h.update(hashlib.sha256(content).digest())
    return f"sha256:{h.hexdigest()}"


class TestTaskDigest:
    def test_golden_fixture_digests(self):
        assert task_digest(FIXTURES / "alpha-task") == ALPHA_DIGEST
        assert task_digest(FIXTURES / "beta-task") == BETA_DIGEST

    def test_matches_spec_construction(self, tmp_path):
        task = tmp_path / "task"
        (task / "environment").mkdir(parents=True)
        (task / "task.toml").write_bytes(b'version = "1.1"\n')
        (task / "instruction.md").write_bytes(b"do the thing\n")
        (task / "environment" / "Dockerfile").write_bytes(b"FROM scratch\n")
        expected = _spec_digest(
            [
                ("environment/Dockerfile", b"FROM scratch\n"),
                ("instruction.md", b"do the thing\n"),
                ("task.toml", b'version = "1.1"\n'),
            ]
        )
        assert task_digest(task) == expected

    def test_sorts_by_posix_path_string_not_path_parts(self, tmp_path):
        """'a.b/c' sorts before 'a/c' as a string ('.' < '/'), after it as
        Path parts — the registry orders by the POSIX path string."""
        task = tmp_path / "task"
        (task / "a").mkdir(parents=True)
        (task / "a.b").mkdir()
        (task / "a" / "c").write_bytes(b"slash")
        (task / "a.b" / "c").write_bytes(b"dot")
        string_order = _spec_digest([("a.b/c", b"dot"), ("a/c", b"slash")])
        parts_order = _spec_digest([("a/c", b"slash"), ("a.b/c", b"dot")])
        assert string_order != parts_order
        assert task_digest(task) == string_order

    def test_symlinks_excluded(self, tmp_path):
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_bytes(b'version = "1.1"\n')
        before = task_digest(task)

        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "payload.txt").write_bytes(b"must not be hashed")
        (task / "file-link.toml").symlink_to(task / "task.toml")
        (task / "dir-link").symlink_to(outside, target_is_directory=True)
        (task / "broken-link").symlink_to(task / "does-not-exist")

        assert task_digest(task) == before

    def test_file_modes_excluded(self, tmp_path):
        task = tmp_path / "task"
        task.mkdir()
        script = task / "solve.sh"
        script.write_bytes(b"#!/bin/bash\n")
        before = task_digest(task)
        script.chmod(0o755)
        assert task_digest(task) == before

    def test_empty_directories_excluded(self, tmp_path):
        task = tmp_path / "task"
        task.mkdir()
        (task / "task.toml").write_bytes(b'version = "1.1"\n')
        before = task_digest(task)
        (task / "empty" / "nested").mkdir(parents=True)
        assert task_digest(task) == before

    def test_sensitive_to_content_and_path(self, tmp_path):
        task = tmp_path / "task"
        task.mkdir()
        f = task / "task.toml"
        f.write_bytes(b'version = "1.1"\n')
        original = task_digest(task)

        f.write_bytes(b'version = "1.2"\n')
        changed_content = task_digest(task)
        assert changed_content != original

        f.write_bytes(b'version = "1.1"\n')
        f.rename(task / "renamed.toml")
        assert task_digest(task) not in (original, changed_content)

    def test_not_a_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            task_digest(tmp_path / "missing")


class TestTasksDigestCli:
    def test_single_task_prints_digest(self):
        result = CliRunner().invoke(
            app, ["tasks", "digest", str(FIXTURES / "alpha-task")]
        )
        assert result.exit_code == 0
        assert result.output == f"{ALPHA_DIGEST}\n"

    def test_directory_of_tasks_prints_name_digest_lines(self):
        """One '<name> <digest>' line per task, sorted; non-task entries
        (not-a-task/, README.md) are skipped."""
        result = CliRunner().invoke(app, ["tasks", "digest", str(FIXTURES)])
        assert result.exit_code == 0
        assert result.output.splitlines() == [
            f"alpha-task {ALPHA_DIGEST}",
            f"beta-task {BETA_DIGEST}",
        ]

    def test_missing_path_fails(self, tmp_path):
        result = CliRunner().invoke(app, ["tasks", "digest", str(tmp_path / "nope")])
        assert result.exit_code == 1

    def test_directory_without_tasks_fails(self):
        result = CliRunner().invoke(
            app, ["tasks", "digest", str(FIXTURES / "not-a-task")]
        )
        assert result.exit_code == 1
        assert "No tasks under" in result.output

    def test_recognizes_native_task_md_task(self, tmp_path):
        """The universal-adapter native format is task.md (no task.toml). The
        digest CLI must recognize it and emit the same digest the run-time
        stamper uses, or `bench tasks digest` is unusable on converted tasks."""
        task = tmp_path / "md-task"
        task.mkdir()
        (task / "task.md").write_text("# A native task\n")
        (task / "input.txt").write_bytes(b"payload")
        result = CliRunner().invoke(app, ["tasks", "digest", str(task)])
        assert result.exit_code == 0, result.output
        assert result.output.strip() == task_digest(task)

    def test_directory_of_task_md_tasks(self, tmp_path):
        for name in ("a-md", "b-md"):
            t = tmp_path / name
            t.mkdir()
            (t / "task.md").write_text(f"# {name}\n")
        result = CliRunner().invoke(app, ["tasks", "digest", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert [line.split()[0] for line in result.output.splitlines()] == [
            "a-md",
            "b-md",
        ]
