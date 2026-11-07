"""Which paths the oracle's git trusts, and for whom (DF-p3-eval-runner-2).

The Docker test (tests/test_oracle_git_ownership_docker.py) shows the
ownership failure itself; these check, with a real git and shell, that the
trust step lists the workspace and each repository in it and nothing outside.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from benchflow.rollout._setup import trust_workspace_repos, trust_workspace_repos_cmd

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(*args: str, env: dict[str, str]) -> None:
    subprocess.run(["git", *args], check=True, env=env, capture_output=True)


def test_trusts_the_workspace_and_its_repositories_only(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1"}
    env.pop("GIT_CONFIG_GLOBAL", None)
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    _git("init", "-q", str(ws / "repo"), env=env)
    _git("init", "-q", str(ws / "a" / "b" / "deep"), env=env)
    _git("init", "-q", "--bare", str(ws / "remote.git"), env=env)
    # DX-G-ASTRA-2: git needs no .git suffix on a bare repository.
    _git("init", "-q", "--bare", str(ws / "remote"), env=env)
    # A HEAD file alone is not a repository.
    (ws / "notes").mkdir()
    (ws / "notes" / "HEAD").write_text("ref: refs/heads/main\n")
    _git("init", "-q", str(outside), env=env)
    outside_bare = tmp_path / "outside-bare"
    _git("init", "-q", "--bare", str(outside_bare), env=env)
    # A link into another repository is not followed.
    (ws / "link").symlink_to(outside)
    (ws / "link-bare").symlink_to(outside_bare)

    for _ in range(2):  # a second run adds nothing twice
        subprocess.run(
            ["sh", "-c", trust_workspace_repos_cmd(str(ws))], check=True, env=env
        )

    listed = subprocess.run(
        ["git", "config", "--global", "--get-all", "safe.directory"],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    real = os.path.realpath(ws)
    assert sorted(listed) == sorted(
        {
            str(ws),
            real,
            f"{real}/repo",
            f"{real}/a/b/deep",
            f"{real}/remote.git",
            f"{real}/remote",
        }
    )
    assert str(outside) not in listed
    assert str(outside_bare) not in listed


def test_a_missing_workspace_or_root_adds_no_repository(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1"}
    env.pop("GIT_CONFIG_GLOBAL", None)
    subprocess.run(
        ["sh", "-c", trust_workspace_repos_cmd(str(tmp_path / "gone"))],
        check=True,
        env=env,
    )
    assert not (home / ".gitconfig").exists()


class _Env:
    def __init__(self) -> None:
        self.users: list[str | None] = []

    async def exec(self, command, user=None, timeout_sec=None, **_kwargs):
        self.users.append(user)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def _task(user):
    return SimpleNamespace(config=SimpleNamespace(agent=SimpleNamespace(user=user)))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("declared", "users"),
    [
        (None, ["root"]),
        ("root", ["root"]),
        ("builder", ["root", "builder"]),
        # An integer [agent].user is left to the image user, as in _run_oracle.
        (1000, ["root"]),
    ],
)
async def test_runs_for_root_and_the_tasks_named_oracle_user(declared, users):
    env = _Env()
    await trust_workspace_repos(env, "/app", _task(declared))
    assert env.users == users
