"""Guards FF-sw-tests-ci's job upload boundary against 363d815b."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import yaml

from tests.test_ci_artifact_checkout import HELPER_REV, _git

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github/scripts/job_artifacts.py"
LANES = [
    ("light", "rollout-smoke", "light"),
    ("scope", "run-matrix", "scope"),
    ("final-review", "run-matrix", "final"),
]


@pytest.mark.parametrize("lane,job,folder", LANES)
@pytest.mark.parametrize("kind", ["absolute", "relative", "directory", "chain"])
def test_workflow_upload_never_reads_host_only_file(tmp_path, lane, job, folder, kind):
    """Guards the fix with the actual upload inputs and a dummy marker."""
    workflow = yaml.safe_load(
        (ROOT / f".github/workflows/integration-{lane}.yml").read_text()
    )
    steps = workflow["jobs"][job]["steps"]
    upload = next(
        s
        for s in steps
        if s.get("uses", "").startswith("actions/upload-artifact")
        and s.get("name") in {"Upload cell jobs", "Upload smoke rollout artifacts"}
    )
    secret = tmp_path / "host-only" / "marker.txt"
    secret.parent.mkdir()
    secret.write_bytes(b"HOST_ONLY_DUMMY_MARKER")
    artifacts = (
        tmp_path / "jobs" / f"integration-{folder}" / "cell" / "trial" / "artifacts"
    )
    artifacts.mkdir(parents=True)
    (artifacts / "answer.txt").write_text("ordinary artifact")
    target = secret if kind != "directory" else secret.parent
    if kind == "relative":
        target = Path(os.path.relpath(secret, artifacts))
    if kind == "chain":
        (artifacts / "hop").symlink_to(secret)
        target = Path("hop")
    (artifacts / "leak").symlink_to(target)
    runner = tmp_path / "runner"
    runner.mkdir()
    if SCRIPT.exists():
        scripts = tmp_path / ".github/scripts"
        scripts.mkdir(parents=True)
        shutil.copyfile(SCRIPT, scripts / SCRIPT.name)
        _git(tmp_path, "init")
        _git(tmp_path, "add", ".github/scripts/job_artifacts.py")
        _git(tmp_path, "commit", "-m", "trusted helper fixture")
        pinned = _git(tmp_path, "rev-parse", "HEAD")
        _git(tmp_path, "remote", "add", "origin", str(tmp_path))
    for step in steps:
        if "job_artifacts.py" in step.get("run", "") and " pack " in step["run"]:
            assert step.get("if") == "always()"
            subprocess.run(
                [
                    "bash",
                    "-eu",
                    "-c",
                    step["run"]
                    .replace("${{ strategy.job-index }}", "0")
                    .replace(HELPER_REV, pinned),
                ],
                cwd=tmp_path,
                env={**os.environ, "RUNNER_TEMP": str(runner)},
                check=True,
            )
    upload_path = upload["with"]["path"].replace("${{ runner.temp }}", str(runner))
    upload_path = upload_path.replace("${{ strategy.job-index }}", "0")
    source = tmp_path / upload_path
    # upload-artifact follows symbolic links. Read the same selected inputs;
    # safe workflows hand it only a host-created regular archive.
    if source.is_dir():
        bodies = [
            Path(parent, name).read_bytes()
            for parent, _, names in os.walk(source, followlinks=True)
            for name in names
        ]
    else:
        assert source.is_file() and not source.is_symlink()
        with tarfile.open(source) as archive:
            assert all(m.isfile() or m.isdir() for m in archive)
            bodies = [archive.extractfile(m).read() for m in archive if m.isfile()]
    assert b"ordinary artifact" in bodies
    assert all(b"HOST_ONLY_DUMMY_MARKER" not in body for body in bodies)


def _script(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=10,
    )


@pytest.mark.parametrize("backend", ["docker", "remote-docker", "daytona", "modal"])
def test_archive_roundtrip_keeps_regular_evidence_only(tmp_path, backend):
    """Guards FF-sw-tests-ci: files round-trip on every backend's job layout."""
    jobs = tmp_path / "jobs"
    trial = jobs / backend / "trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text('{"reward": 0}')
    (trial / "empty-dir").mkdir()
    (trial / "empty").touch()
    (trial / ".private").write_text("HOST_ONLY_DUMMY_MARKER")
    (trial / ".hidden-directory").mkdir()
    (trial / ".hidden-directory/marker").write_text("HOST_ONLY_DUMMY_MARKER")
    (trial / "dangling").symlink_to("missing")
    (trial / "internal-link").symlink_to("result.json")
    (trial / "loop").symlink_to(".")
    os.mkfifo(trial / "fifo")
    outside = tmp_path / "marker"
    outside.write_text("HOST_ONLY_DUMMY_MARKER")
    os.link(outside, trial / "hardlink")
    archives = tmp_path / "archives"
    assert _script("pack", jobs, archives / "jobs.tar.gz").returncode == 0
    destination = tmp_path / "restored"
    result = _script("unpack", archives, destination)
    assert result.returncode == 0, result.stderr
    restored = destination / backend / "trial"
    assert sorted(p.name for p in restored.iterdir()) == [
        "empty",
        "empty-dir",
        "result.json",
    ]
    assert (restored / "result.json").read_text() == '{"reward": 0}'
    assert (restored / "empty-dir").is_dir()


@pytest.mark.parametrize("kind", ["root-link", "ancestor-link", "output-link"])
def test_pack_refuses_links_in_its_own_paths(tmp_path, kind):
    """Guards FF-sw-tests-ci: the root and output obey the same rule as leaves."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "jobs").mkdir()
    (real / "jobs/marker").write_text("HOST_ONLY_DUMMY_MARKER")
    link = tmp_path / "link"
    link.symlink_to(real if kind == "ancestor-link" else real / "jobs")
    source = link / "jobs" if kind == "ancestor-link" else link
    output = tmp_path / "jobs.tar.gz"
    if kind == "output-link":
        source = real / "jobs"
        output.symlink_to(real / "jobs/marker")
    result = _script("pack", source, output)
    assert result.returncode != 0
    assert (real / "jobs/marker").read_text() == "HOST_ONLY_DUMMY_MARKER"


@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "fifo", "absolute", "parent", "duplicate"]
)
def test_unpack_refuses_non_files_and_escaping_paths(tmp_path, kind):
    """Guards FF-sw-tests-ci: downloaded archives cannot reintroduce links."""
    import io

    archives = tmp_path / "archives"
    archives.mkdir()
    with tarfile.open(archives / "jobs.tar.gz", "w:gz") as tar:
        member = tarfile.TarInfo("file")
        if kind in {"symlink", "hardlink", "fifo"}:
            member.type = {
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
            }[kind]
            member.linkname = "../marker"
        elif kind == "absolute":
            member.name = str(tmp_path / "marker")
        elif kind == "parent":
            member.name = "../marker"
        tar.addfile(member, io.BytesIO())
        if kind == "duplicate":
            tar.addfile(member, io.BytesIO())
    marker = tmp_path / "marker"
    marker.write_text("untouched")
    assert _script("unpack", archives, tmp_path / "restored").returncode != 0
    assert marker.read_text() == "untouched"


@pytest.mark.parametrize("lane,job,folder", LANES[1:])
def test_downloaded_archives_are_restored_before_grading(lane, job, folder):
    """Guards FF-sw-tests-ci: archive transport preserves the grader layout."""
    workflow = yaml.safe_load(
        (ROOT / f".github/workflows/integration-{lane}.yml").read_text()
    )
    steps = workflow["jobs"]["review-pack"]["steps"]
    download = next(s for s in steps if s.get("name") == "Download all cell jobs")
    restore = next(s for s in steps if s.get("name") == "Restore regular job files")
    grade = next(
        s for s in steps if "build_integration_review_pack.py" in s.get("run", "")
    )
    assert steps.index(download) < steps.index(restore) < steps.index(grade)
    assert (
        restore["run"]
        .rstrip()
        .endswith(
            f'python -I "$RUNNER_TEMP/job_artifacts.py" unpack {download["with"]["path"]} jobs/integration-{folder}'
        )
    )


@pytest.mark.parametrize("kind", ["leaf", "directory"])
def test_replacement_between_listing_and_open_never_follows_link(
    tmp_path, monkeypatch, kind
):
    """Guards FF-sw-tests-ci against the next path spelling: a substituted inode."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("job_artifacts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    entry = jobs / "entry"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "marker").write_bytes(b"HOST_ONLY_DUMMY_MARKER")
    if kind == "directory":
        entry.mkdir()
        (entry / "answer").write_text("safe")
    else:
        entry.write_text("safe")
    original = os.open
    swapped = False

    def swap(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "entry" and not swapped:
            swapped = True
            entry.rename(jobs / "old-entry")
            entry.symlink_to(outside if kind == "directory" else outside / "marker")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", swap)
    output = tmp_path / "archive.tar.gz"
    module.pack(jobs, output)
    assert swapped
    with tarfile.open(output) as archive:
        assert all(
            b"HOST_ONLY_DUMMY_MARKER" not in archive.extractfile(m).read()
            for m in archive
            if m.isfile()
        )
