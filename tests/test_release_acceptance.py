"""Guards the fix for unbound release artifacts at commit 1cdbbf96d3."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from tools.check_release_acceptance import check_release_acceptance, main


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=BenchFlow Test",
            "-c",
            "user.email=test@example.com",
            *args,
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


@pytest.fixture
def accepted(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init")
    (repo / "pyproject.toml").write_text('[project]\nversion = "0.9.0"\n')
    (repo / "README.md").write_text("Tested release source.\n")
    (repo / ".gitignore").write_text(
        "/dist/\n/src/benchflow/_bfx/bin/bfx-x86_64\n"
        "/src/benchflow/_bfx/bin/bfx-aarch64\n"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Tested source")
    source = _git(repo, "rev-parse", "HEAD")
    dist = tmp_path / "dist"
    dist.mkdir()
    contents = {
        "benchflow-0.9.0-py3-none-any.whl": b"accepted wheel bytes",
        "benchflow-0.9.0.tar.gz": b"accepted source archive bytes",
    }
    for name, data in contents.items():
        (dist / name).write_bytes(data)
    record = {
        "schema_version": 1,
        "version": "0.9.0",
        "status": "accepted",
        "source_commit": source,
        "acceptance_receipt_sha256": hashlib.sha256(b"reviewed receipt").hexdigest(),
        "artifacts": {
            name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
        },
    }
    path = repo / ".github" / "release-acceptance" / "0.9.0.json"
    path.parent.mkdir(parents=True)
    _record(repo, path, record)
    return repo, dist, path, record


def _record(repo: Path, path: Path, record: dict) -> None:
    path.write_text(json.dumps(record, indent=2) + "\n")
    _git(repo, "add", str(path))
    _git(repo, "commit", "-m", "Record release acceptance")


def test_accepts_only_recorded_source_and_distribution_bytes(accepted: tuple) -> None:
    repo, dist, _, record = accepted
    messages = check_release_acceptance(repo, dist)
    assert record["source_commit"] in messages[0]
    assert len(messages) == 3
    assert main([str(dist), "--repo", str(repo)]) == 0


def test_accepts_uv_generated_distribution_ignore_file(accepted: tuple) -> None:
    """Guards PR #13: uv 0.13.4 adds a one-byte '*' alongside built artifacts."""
    repo, dist, _, record = accepted
    (dist / ".gitignore").write_bytes(b"*")
    assert record["source_commit"] in check_release_acceptance(repo, dist)[0]


@pytest.mark.parametrize(
    "change", ["empty", "changed", "symlink", "directory", "fifo", "other-hidden"]
)
def test_rejects_substituted_distribution_metadata(
    accepted: tuple, change: str
) -> None:
    """Guards PR #13: only the exact regular uv marker may accompany artifacts."""
    repo, dist, _, _ = accepted
    marker = dist / ".gitignore"
    if change == "empty":
        marker.write_bytes(b"")
    elif change == "changed":
        marker.write_bytes(b"*\n")
    elif change == "symlink":
        outside = dist.parent / "marker"
        outside.write_bytes(b"*")
        marker.symlink_to(outside)
    elif change == "directory":
        marker.mkdir()
    elif change == "fifo":
        os.mkfifo(marker)
    else:
        (dist / ".extra").write_bytes(b"*")
    with pytest.raises(ValueError, match="artifact set"):
        check_release_acceptance(repo, dist)


@pytest.mark.parametrize("suffix", [".whl", ".tar.gz"])
def test_rejects_rebuilt_artifact_with_different_bytes(
    accepted: tuple, suffix: str
) -> None:
    repo, dist, _, _ = accepted
    artifact = next(path for path in dist.iterdir() if path.name.endswith(suffix))
    artifact.write_bytes(artifact.read_bytes() + b"changed after acceptance")
    with pytest.raises(ValueError, match="differs from the accepted SHA-256"):
        check_release_acceptance(repo, dist)


@pytest.mark.parametrize("change", ["missing", "extra", "symlink", "directory", "fifo"])
def test_rejects_incomplete_or_substituted_distribution_set(
    accepted: tuple, change: str
) -> None:
    repo, dist, _, _ = accepted
    wheel = dist / "benchflow-0.9.0-py3-none-any.whl"
    if change == "missing":
        wheel.unlink()
    elif change == "extra":
        (dist / "benchflow-0.9.1-py3-none-any.whl").write_bytes(b"untested")
    elif change == "symlink":
        outside = dist.parent / "outside.whl"
        wheel.rename(outside)
        wheel.symlink_to(outside)
    else:
        wheel.unlink()
        if change == "directory":
            wheel.mkdir()
        else:
            os.mkfifo(wheel)
    with pytest.raises(ValueError, match=r"artifact set|regular file"):
        check_release_acceptance(repo, dist)


def test_rejects_source_changed_after_acceptance(accepted: tuple) -> None:
    repo, dist, _, _ = accepted
    (repo / "README.md").write_text("Unreviewed release claims.\n")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "Change after acceptance")
    with pytest.raises(
        ValueError, match=r"source changed after acceptance: README\.md"
    ):
        check_release_acceptance(repo, dist)


def test_rejects_uncommitted_acceptance_replacement(accepted: tuple) -> None:
    repo, dist, path, record = accepted
    record["acceptance_receipt_sha256"] = "a" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="uncommitted changes"):
        check_release_acceptance(repo, dist)


@pytest.mark.parametrize(
    "name", ["src/benchflow/release_probe.py", "tests/test_probe.py"]
)
def test_rejects_untracked_release_source(accepted: tuple, name: str) -> None:
    """Guards R1-M1: 1fdf142f ignores untracked package and test inputs."""
    repo, dist, _, _ = accepted
    path = repo / name
    path.parent.mkdir(parents=True)
    path.write_text("unexpected_source = True\n")
    with pytest.raises(ValueError, match="uncommitted changes"):
        check_release_acceptance(repo, dist)


def test_ignored_build_outputs_do_not_change_source_identity(accepted: tuple) -> None:
    """Guards R1-M1 while retaining separately checked build artifacts."""
    repo, dist, _, record = accepted
    inside = repo / "dist"
    dist.rename(inside)
    binaries = repo / "src" / "benchflow" / "_bfx" / "bin"
    binaries.mkdir(parents=True)
    for arch in ("x86_64", "aarch64"):
        (binaries / f"bfx-{arch}").write_bytes(b"checked separately by check_dist_bfx")
    assert record["source_commit"] in check_release_acceptance(repo, inside)[0]


@pytest.mark.parametrize("case", ["duplicate-top", "duplicate-artifact", "truncated"])
def test_rejects_malformed_or_duplicate_json(accepted: tuple, case: str) -> None:
    """Guards R1-L1 using raw JSON, which a dict serializer cannot emit."""
    repo, dist, path, record = accepted
    raw = json.dumps(record)
    if case == "duplicate-top":
        raw = raw.replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
        )
    elif case == "duplicate-artifact":
        name, digest = next(iter(record["artifacts"].items()))
        field = f"{json.dumps(name)}: {json.dumps(digest)}"
        raw = raw.replace(field, f"{field}, {field}")
    else:
        raw = raw[:-1]
    path.write_text(raw)
    _git(repo, "add", str(path))
    _git(repo, "commit", "-m", "Malformed acceptance input")
    with pytest.raises(ValueError):
        check_release_acceptance(repo, dist)


@pytest.mark.parametrize(
    "name", ["../benchflow-0.9.0.tar.gz", "benchflow-0.8.3.tar.gz"]
)
def test_rejects_artifact_names_outside_this_release(
    accepted: tuple, name: str
) -> None:
    """Guards R1-L1's required version and artifact-name boundary."""
    repo, dist, path, record = accepted
    digest = record["artifacts"].pop("benchflow-0.9.0.tar.gz")
    record["artifacts"][name] = digest
    _record(repo, path, record)
    with pytest.raises(ValueError, match="names do not match"):
        check_release_acceptance(repo, dist)


def test_rejects_unrelated_accepted_source(accepted: tuple) -> None:
    repo, dist, path, record = accepted
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    record["source_commit"] = _git(repo, "commit-tree", tree, "-m", "Other history")
    _record(repo, path, record)
    with pytest.raises(ValueError, match="git validation failed"):
        check_release_acceptance(repo, dist)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "pending", "not complete"),
        ("version", "0.8.3", "version does not match"),
        ("schema_version", True, "schema_version 1"),
        ("acceptance_receipt_sha256", None, "receipt SHA-256"),
        ("source_commit", "main", "full commit SHA"),
        ("artifacts", {}, "exactly one wheel and one sdist"),
    ],
)
def test_rejects_incomplete_acceptance(
    accepted: tuple, field: str, value: object, message: str
) -> None:
    repo, dist, path, record = accepted
    record[field] = value
    _record(repo, path, record)
    with pytest.raises(ValueError, match=message):
        check_release_acceptance(repo, dist)


def test_missing_record_fails_cli_without_publication(accepted: tuple) -> None:
    repo, dist, path, _ = accepted
    path.unlink()
    assert main([str(dist), "--repo", str(repo)]) == 1


def test_publication_checks_acceptance_before_upload() -> None:
    root = Path(__file__).resolve().parent.parent
    workflow = yaml.load(
        (root / ".github/workflows/public-release.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    commands = [step.get("run", "") for step in workflow["jobs"]["publish"]["steps"]]
    check = next(
        i for i, cmd in enumerate(commands) if "check_release_acceptance.py" in cmd
    )
    publish = next(i for i, cmd in enumerate(commands) if "uv publish" in cmd)
    helper = next(i for i, cmd in enumerate(commands) if "check_dist_bfx.py" in cmd)
    assert helper < check < publish
