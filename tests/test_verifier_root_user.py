"""One answer to whether test.sh runs as root.

Hardening asked twice: the uv and pip state move read the task's spelling of
``[verifier] user`` and counted only ``root`` and ``0``, while the plugin
guard's policy read ``id -u``. Docker takes ``user[:group]``, so ``0:0`` was
not root to the move (uv kept reading the workspace's configuration) but was
to the guard. ``id -u`` as the verifier's user now decides both, and the
spelling only when that cannot be told.

The scenarios use the harness of test_verifier_uv_state.py: the real
hardening (``_build_verifier_env``) against a directory tree standing in for
the image.
"""

import os

import pytest

from benchflow.sandbox import lockdown
from tests.test_verifier_uv_state import Layout, make_task


def guard_policy(layout: Layout) -> str:
    (guard,) = layout.fs.glob("_runtime_guard_*/_runtime_guard_*.py")
    return guard.read_text()


@pytest.mark.parametrize("user", ["0:0", "root:root", "root:0", "0:1000", 0])
async def test_every_spelling_of_root_gets_the_protected_uv_configuration(
    tmp_path, monkeypatch, user
):
    """``user = "0:0"`` read the agent's uv.toml.

    Docker takes ``user[:group]``. The uv and pip state move counted only
    ``root`` and ``0`` as root, so with ``0:0`` uv kept reading the
    workspace's configuration (a ``find-links`` to the agent's wheel), while
    the guard, from ``id -u``, applied its rules for a root test.sh. Both now
    take the same answer: the move happens and the guard's policy says root.
    """
    layout = Layout(tmp_path, monkeypatch, workspace_name="app")
    task = make_task(layout)
    task.config.verifier.user = user

    env = await lockdown._build_verifier_env(
        layout, task, "agent", str(layout.workspace)
    )

    (directory,) = layout.fs.glob("_runtime_verifier_*")
    assert env.get("UV_CONFIG_FILE") == str(directory / "uv.toml")
    policy = guard_policy(layout)
    assert "_BENCHFLOW_VERIFIER_ROOT = True" in policy
    assert "_BENCHFLOW_RECORD = ''" not in policy


async def test_root_spelling_decides_when_id_cannot_be_told(tmp_path, monkeypatch):
    """Without ``id -u`` the task's spelling is the one answer."""

    async def unknown(env, user):
        return None

    monkeypatch.setattr(lockdown, "_verifier_uid", unknown)
    layout = Layout(tmp_path, monkeypatch, workspace_name="app")
    task = make_task(layout)
    task.config.verifier.user = "0:0"

    env = await lockdown._build_verifier_env(
        layout, task, "agent", str(layout.workspace)
    )

    assert "UV_CONFIG_FILE" in env
    assert "_BENCHFLOW_VERIFIER_ROOT = True" in guard_policy(layout)


async def test_a_verifier_id_says_is_not_root_gets_neither(tmp_path, monkeypatch):
    """An image whose default user is not root.

    ``user`` unset runs test.sh as the image's user; ``id -u`` says it is not 0.
    The move (a root-owned directory test.sh could not write) and the
    guard's root rules both stay off, and no record makes anything test.sh
    installs in the agent's trees its own.
    """

    async def not_root(env, user):
        return str(os.getuid() or 1000)

    monkeypatch.setattr(lockdown, "_verifier_uid", not_root)
    layout = Layout(tmp_path, monkeypatch, workspace_name="app")
    task = make_task(layout)

    env = await lockdown._build_verifier_env(
        layout, task, "agent", str(layout.workspace)
    )

    assert "UV_CONFIG_FILE" not in env
    assert not list(layout.fs.glob("_runtime_verifier_*"))
    policy = guard_policy(layout)
    assert "_BENCHFLOW_VERIFIER_ROOT = False" in policy
    assert "_BENCHFLOW_RECORD = ''" in policy


@pytest.mark.parametrize(
    ("user", "root"),
    [
        (None, True),
        ("root", True),
        ("0", True),
        (0, True),
        ("0:0", True),
        ("root:root", True),
        (" root : staff", True),
        ("1000:0", False),
        ("agent", False),
        ("rooted", False),
        ("10", False),
    ],
)
def test_root_spellings(user, root):
    """uid 0 whatever the group, as Docker reads ``user[:group]``."""
    assert lockdown._verifier_runs_as_root(user) is root
    assert lockdown._verifier_is_root(None, user) is root
    # ``id -u`` wins over the spelling.
    assert lockdown._verifier_is_root("0", user) is True
    assert lockdown._verifier_is_root("1000", user) is False
