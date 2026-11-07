"""The freeze's git purge (``lockdown._PURGE_GIT_CONTROLS_SCRIPT``) and the
verifier env's git settings, on a real repository.

Fails on 42bb217b, where root's git in test.sh ran the hooks, fsmonitor, drivers
and includes the agent left in the workspace's repository.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchflow.sandbox import lockdown

pytestmark = pytest.mark.skipif(
    not os.path.isfile("/usr/bin/git"),
    reason="the purge uses a root-owned /usr/bin/git",
)

AGENT_CONFIG = (
    ("core.fsmonitor", "sh -c :"),
    ("core.hooksPath", "/tmp"),
    ("diff.external", "sh -c :"),
    ("filter.gi.clean", "sh -c :"),
    ("diff.gi.textconv", "sh -c :"),
    ("merge.gi.driver", "sh -c :"),
    ("include.path", "/tmp/elsewhere"),
    ("alias.lfs", "!sh -c :"),
    ("core.alternateRefsCommand", "sh -c :"),
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "HOME": str(repo.parent),
            "GIT_CONFIG_NOSYSTEM": "1",
        },
    ).stdout


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "app"
    workspace.mkdir()
    _git(workspace, "init", "-q")
    hook = workspace / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 0\n")
    hook.chmod(0o755)
    _git(workspace, "config", "filter.lfs.clean", "git-lfs clean -- %f")
    return workspace


def _baseline(tmp_path: Path, workspace: Path) -> Path:
    baseline = tmp_path / "pre-agent"
    shutil.copytree(workspace, baseline, symlinks=True)
    baseline.chmod(0o700)
    return baseline


def _plant(workspace: Path) -> None:
    hooks = workspace / ".git" / "hooks"
    (hooks / "post-checkout").write_text("#!/bin/sh\n: > /tmp/gi-marker\n")
    (hooks / "post-checkout").chmod(0o755)
    (hooks / "pre-commit").write_text("#!/bin/sh\n: > /tmp/gi-marker\n")
    for key, value in AGENT_CONFIG:
        _git(workspace, "config", key, value)
    _git(workspace, "config", "user.name", "agent")


def _purge(workspace: Path, baseline: Path | str = "") -> None:
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            lockdown._PURGE_GIT_CONTROLS_SCRIPT,
            str(workspace),
            str(baseline),
        ],
        check=True,
    )


def _hooks(workspace: Path) -> list[str]:
    folder = workspace / ".git" / "hooks"
    return sorted(p.name for p in folder.iterdir() if not p.name.endswith(".sample"))


def test_the_agents_hooks_and_commands_go_and_the_repositorys_own_stay(tmp_path):
    """New and changed hooks and every command entry the agent added go; the
    pre-agent filter and the agent's harmless settings stay."""
    workspace = _workspace(tmp_path)
    baseline = _baseline(tmp_path, workspace)
    _plant(workspace)
    _purge(workspace, baseline)
    assert _hooks(workspace) == []
    config = _git(workspace, "config", "--list", "--no-includes")
    for key, _ in AGENT_CONFIG:
        assert key.lower() + "=" not in config.lower(), (key, config)
    assert "filter.lfs.clean=git-lfs clean -- %f" in config
    assert "user.name=agent" in config


def test_an_unchanged_repository_is_left_as_it_was(tmp_path):
    """The repository's own hook and config, untouched by the agent, stay byte for byte."""
    workspace = _workspace(tmp_path)
    baseline = _baseline(tmp_path, workspace)
    before = (workspace / ".git" / "config").read_bytes()
    _purge(workspace, baseline)
    assert _hooks(workspace) == ["pre-commit"]
    assert (workspace / ".git" / "config").read_bytes() == before


def test_without_a_trusted_baseline_every_such_entry_goes(tmp_path):
    """A separate verifier has no pre-agent copy: nothing there is the repository's own."""
    workspace = _workspace(tmp_path)
    baseline = _baseline(tmp_path, workspace)
    baseline.chmod(0o755)  # open to others: not trusted
    _plant(workspace)
    _purge(workspace, baseline)
    assert _hooks(workspace) == []
    config = _git(workspace, "config", "--list", "--no-includes")
    assert "filter.lfs" not in config and "user.name=agent" in config


def test_a_linked_config_and_a_nested_repository_are_purged(tmp_path):
    """A config that is a link is put back from the copy (git would follow
    it); a repository the agent made inside the workspace is purged too."""
    workspace = _workspace(tmp_path)
    baseline = _baseline(tmp_path, workspace)
    elsewhere = tmp_path / "elsewhere.cfg"
    elsewhere.write_text('[filter "gi"]\n\tclean = sh -c :\n')
    config = workspace / ".git" / "config"
    config.unlink()
    config.symlink_to(elsewhere)
    nested = workspace / "vendor" / "lib"
    nested.mkdir(parents=True)
    _git(nested, "init", "-q")
    _git(nested, "config", "core.fsmonitor", "sh -c :")
    _purge(workspace, baseline)
    assert not config.is_symlink()
    assert config.read_bytes() == (baseline / ".git" / "config").read_bytes()
    assert "fsmonitor" not in _git(nested, "config", "--list", "--no-includes")
    _git(workspace, "status", "--porcelain")


def _common_folder(workspace: Path, folder: Path) -> None:
    """Move the repository's shared part to *folder*: git then reads its
    config and hooks there (``commondir``)."""
    shutil.copytree(workspace / ".git", folder, symlinks=True)
    (folder / "HEAD").unlink()
    _git(
        workspace,
        "config",
        "--file",
        str(folder / "config"),
        "filter.gi.clean",
        "sh -c :",
    )
    (folder / "hooks" / "post-checkout").write_text("#!/bin/sh\n: > /tmp/gi-marker\n")
    (folder / "hooks" / "post-checkout").chmod(0o755)
    (workspace / ".git" / "commondir").write_text(
        os.path.relpath(folder, workspace / ".git") + "\n"
    )
    assert "filter.gi.clean" in _git(workspace, "config", "--list")


@pytest.mark.parametrize("agent_ran_here", [True, False])
def test_a_common_folder_in_the_workspace_is_purged_as_the_repository(
    tmp_path, agent_ran_here
):
    """A commondir naming a workspace folder without HEAD: its config and hooks go."""
    workspace = _workspace(tmp_path)
    baseline = _baseline(tmp_path, workspace) if agent_ran_here else ""
    common = workspace / "vendor" / "shared"
    common.parent.mkdir()
    _common_folder(workspace, common)
    _purge(workspace, baseline)
    assert "filter.gi" not in _git(workspace, "config", "--list", "--no-includes")
    left = [
        p.name for p in (common / "hooks").iterdir() if not p.name.endswith(".sample")
    ]
    assert left == []
    _git(workspace, "status", "--porcelain")


def test_pointers_out_of_the_workspace_go_only_where_the_agent_ran(tmp_path):
    """A commondir or .git file naming a folder outside goes when the agent
    ran in this sandbox; a separate verifier, where nothing outside came
    from the agent, keeps both."""
    for verifier_baseline in ("baseline", ""):
        root = tmp_path / (verifier_baseline or "separate")
        root.mkdir()
        workspace = _workspace(root)
        baseline = _baseline(root, workspace) if verifier_baseline else ""
        _common_folder(workspace, root / "outside")
        sub = workspace / "sub"
        sub.mkdir()
        (sub / ".git").write_text(f"gitdir: {root / 'outside-gitdir'}\n")
        _purge(workspace, baseline)
        moved = (workspace / ".git" / "commondir").exists(), (sub / ".git").exists()
        assert moved == ((False, False) if verifier_baseline else (True, True))
    _git(tmp_path / "baseline" / "app", "status", "--porcelain")


def test_the_verifier_env_alone_runs_no_hook_fsmonitor_or_external_diff(tmp_path):
    """Beside the purge: git under the verifier's env runs none of them; without it, all three run."""
    workspace = _workspace(tmp_path)
    (workspace / "tracked.txt").write_text("base\n")
    _git(workspace, "add", "tracked.txt")
    _git(
        workspace,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-qm",
        "base",
    )
    marker = tmp_path / "ran"
    hook = workspace / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\n: > {marker}-hook\n")
    hook.chmod(0o755)
    _git(workspace, "config", "core.fsmonitor", f"sh -c ': > {marker}-fsmonitor'")
    _git(workspace, "config", "diff.external", f"sh -c ': > {marker}-diff'")
    verifier = {k: v for k, v in lockdown.VERIFIER_ENV.items() if k.startswith("GIT_")}

    def grade(extra: dict[str, str], answer: str) -> list[str]:
        for leftover in tmp_path.glob("ran-*"):
            leftover.unlink()
        (workspace / "tracked.txt").write_text(answer)
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(tmp_path),
            "GIT_CONFIG_NOSYSTEM": "1",
            **extra,
        }
        for args in (
            ["status"],
            ["diff"],
            [
                "-c",
                "user.name=g",
                "-c",
                "user.email=g@example.invalid",
                "commit",
                "-qam",
                "g",
            ],
        ):
            subprocess.run(
                ["/usr/bin/git", "-C", str(workspace), *args],
                env=env,
                capture_output=True,
                check=False,
            )
        return sorted(p.name for p in tmp_path.glob("ran-*"))

    assert grade({}, "first\n") == ["ran-diff", "ran-fsmonitor", "ran-hook"]
    assert grade(verifier, "second\n") == []
