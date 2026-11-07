"""Guards FF-sw-tests-ci-confirm-1 against the checkout skew at 1a32bcbe."""

import hashlib
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HELPER_REV = "1a32bcbee558377912ae01c6d15767ef88d981f5"
HELPER = ".github/scripts/job_artifacts.py"
# The local fixture must be the pinned release's bytes, not a future PR helper.
HELPER_SHA256 = "45c9acfd858dcd698f375e30fd85c9ce99948684f8f9d363f18491b0ac06b7a2"
LANES = [
    ("light", "rollout-smoke", "light"),
    ("scope", "run-matrix", "scope"),
    ("final-review", "run-matrix", "final"),
]


def _git(cwd, *args):
    return subprocess.check_output(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            *args,
        ],
        cwd=cwd,
        text=True,
        stderr=subprocess.PIPE,
    ).strip()


@pytest.mark.parametrize("lane,job,folder", LANES)
@pytest.mark.parametrize("base_has_helper", [False, True])
def test_archive_commands_use_pinned_helper_on_older_checkout(
    tmp_path, lane, job, folder, base_has_helper
):
    """Run pack/restore on base + src-only overlay; candidate infra stays unused."""
    assert hashlib.sha256((ROOT / HELPER).read_bytes()).hexdigest() == HELPER_SHA256
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init")
    scripts = origin / ".github/scripts"
    scripts.mkdir(parents=True)
    (origin / "base.txt").write_text("trusted base")
    if base_has_helper:
        (origin / HELPER).write_text("raise RuntimeError('old helper must not run')")
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "base without the new helper")
    base = _git(origin, "rev-parse", "HEAD")
    (origin / HELPER).write_text((ROOT / HELPER).read_text())
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "trusted helper release")
    pinned = _git(origin, "rev-parse", "HEAD")
    (origin / HELPER).write_text("raise RuntimeError('candidate helper must not run')")
    src = origin / "src/benchflow"
    src.mkdir(parents=True)
    (src / "candidate.txt").write_text("code under test")
    _git(origin, "add", ".")
    _git(origin, "commit", "-m", "candidate")
    candidate = _git(origin, "rev-parse", "HEAD")

    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", str(origin), str(checkout))
    _git(checkout, "checkout", base)
    _git(checkout, "checkout", candidate, "--", "src/benchflow")
    workflow = yaml.safe_load(
        (ROOT / f".github/workflows/integration-{lane}.yml").read_text()
    )
    steps = workflow["jobs"][job]["steps"]
    pack = next(s for s in steps if s.get("name", "").startswith("Archive "))
    runner = tmp_path / "runner"
    runner.mkdir()
    evidence = checkout / "jobs" / f"integration-{folder}" / "cell/trial/result.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text('{"reward": 0}')
    (evidence.parent / "link").symlink_to(tmp_path / "host-marker")
    (tmp_path / "host-marker").write_text("HOST_ONLY_DUMMY_MARKER")

    def execute(step, cwd):
        command = step["run"]
        # Substitute only the pinned revision with its local fixture equivalent;
        # fetch/show and the workflow's actual paths run unchanged, without a key.
        command = command.replace(HELPER_REV, pinned)
        command = command.replace("${{ strategy.job-index }}", "0")
        result = subprocess.run(
            ["bash", "-euo", "pipefail", "-c", command],
            cwd=cwd,
            env={**os.environ, "RUNNER_TEMP": str(runner)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, result.stderr
        assert f"git fetch --no-tags origin {HELPER_REV}" in step["run"]
        assert f"git show {HELPER_REV}:{HELPER}" in step["run"]
        assert re.search(r'python -I "\$RUNNER_TEMP/job_artifacts.py"', step["run"])

    execute(pack, checkout)
    if lane == "light":
        return
    # Review jobs have a fresh base checkout and no source overlay at all.
    review = tmp_path / "review"
    _git(tmp_path, "clone", str(origin), str(review))
    _git(review, "checkout", base)
    (review / "job-archives").mkdir()
    for archive in (runner / "job-archives").glob("*.tar.gz"):
        (review / "job-archives" / archive.name).write_bytes(archive.read_bytes())
    restore = next(
        s
        for s in workflow["jobs"]["review-pack"]["steps"]
        if s.get("name") == "Restore regular job files"
    )
    execute(restore, review)
    restored = review / evidence.relative_to(checkout)
    assert restored.read_text() == '{"reward": 0}'
    assert not (restored.parent / "link").exists()
