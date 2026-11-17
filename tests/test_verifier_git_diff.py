"""test.sh's own ``git diff`` works under the verifier's environment.

``lockdown.VERIFIER_ENV`` pinned ``diff.external`` to an empty value at git's
command-line scope. git then ran "" for every diff and died, so SWT-bench's
test.sh, whose ``git diff`` builds the patch it grades, scored the correct
oracle 0 on astropy-14309 (Harbor 0.26 scored it 1). The pin is gone; an
external diff the agent names in a repository's config is still removed by the
freeze's purge (``_PURGE_GIT_CONTROLS_SCRIPT``) before test.sh runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from benchflow.sandbox.lockdown import _PURGE_GIT_CONTROLS_SCRIPT, VERIFIER_ENV

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is not installed")


def _plain_env(home):
    return {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "LC_ALL": "C",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }


def _verifier_git_env(home):
    """The verifier's git settings (every GIT_* of VERIFIER_ENV) on a plain PATH."""
    env = _plain_env(home)
    env.pop("GIT_CONFIG_NOSYSTEM")
    env.update({k: v for k, v in VERIFIER_ENV.items() if k.startswith("GIT_")})
    return env


def _repo_with_a_change(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(
            [GIT, *args],
            cwd=repo,
            check=True,
            capture_output=True,
            env=_plain_env(tmp_path),
        )

    git("init", "-q")
    (repo / "f.py").write_text("a = 1\n")
    git("add", "f.py")
    git("-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "i")
    (repo / "f.py").write_text("a = 2\n")
    return repo


def _diff(repo, home):
    return subprocess.run(
        [GIT, "diff"],
        cwd=repo,
        env=_verifier_git_env(home),
        capture_output=True,
        text=True,
    )


def test_git_diff_under_the_verifier_env_prints_the_patch(tmp_path):
    """SWT-bench's patch: an empty diff.external made git run "" and die."""
    repo = _repo_with_a_change(tmp_path)

    out = _diff(repo, tmp_path)

    assert out.returncode == 0, out.stderr
    assert "-a = 1" in out.stdout and "+a = 2" in out.stdout, out.stdout


def test_the_verifier_env_never_pins_diff_external():
    """No command-line value unsets it; an empty one is a broken git diff."""
    count = int(VERIFIER_ENV["GIT_CONFIG_COUNT"])
    keys = [VERIFIER_ENV[f"GIT_CONFIG_KEY_{i}"] for i in range(count)]

    assert "diff.external" not in keys
    assert {"core.hooksPath", "core.fsmonitor"} <= set(keys)
    assert not any(
        k.startswith("GIT_CONFIG_KEY_") and int(k.rsplit("_", 1)[1]) >= count
        for k in VERIFIER_ENV
    )


def test_an_external_diff_the_agent_named_is_purged_and_never_runs(tmp_path):
    """The freeze's purge, not the pin, keeps an agent's external diff out."""
    repo = _repo_with_a_change(tmp_path)
    proof = tmp_path / "agent-diff-ran"
    tool = tmp_path / "agent-diff"
    tool.write_text(f"#!/bin/sh\ntouch {proof}\n")
    tool.chmod(0o755)
    subprocess.run(
        [GIT, "config", "diff.external", str(tool)],
        cwd=repo,
        check=True,
        env=_plain_env(tmp_path),
    )

    # No pre-agent copy: every such entry goes (a separate verifier's rule).
    purge = subprocess.run(
        [sys.executable, "-c", _PURGE_GIT_CONTROLS_SCRIPT, str(tmp_path), ""],
        capture_output=True,
        text=True,
    )
    assert purge.returncode == 0, purge.stderr

    out = _diff(repo, tmp_path)

    assert out.returncode == 0, out.stderr
    assert "+a = 2" in out.stdout
    assert not proof.exists()
