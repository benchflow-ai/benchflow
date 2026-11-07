"""A review's own rollout picks its Daytona ACP transport apart from the solver's.

0.9 picks one transport per process for every Daytona rollout
(``BENCHFLOW_DAYTONA_ACP_TRANSPORT``, PTY unless set), so an in-run reviewer
rode its solver's PTY websocket, where a burst of agent output can end the
session ("PTY closed by the peer: websocket closed (close_code=1006 ...)").
A review's rollout now talks ssh on Daytona unless
``BENCHFLOW_REVIEWER_ACP_TRANSPORT=pty``, records it in its ``config.json``,
and its ssh launch carries what the PTY session gave the agent process
(``TERM``, ``SHELL``). The solver and every other rollout keep 0.9's choice.

Guards the fix for the gap a benchmark's own infrastructure works around as
a patch. No sandbox is created: the reviewer's rollout is a fake that
reports what the rollout would select.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import benchflow
from benchflow.acp.selection import selected_acp_transport
from benchflow.review.config import load_rubric
from benchflow.review.options import ReviewerConfig
from benchflow.review.runner import run_review
from benchflow.sandbox import process as process_pkg
from benchflow.sandbox._base import SYSTEM_SHELL_EMPTY_VARS
from benchflow.sandbox.daytona import DaytonaSandbox
from benchflow.sandbox.process.daytona import DaytonaProcess
from tests._trust_admission import admitted
from tests.test_review_runtime import (
    WEIGHTED_RUBRIC,
    FakeRun,
    good_weighted_review,
    make_rollout,
    make_task,
)

_PTY_SESSION_ENV = {"TERM": "xterm-256color", "SHELL": "/bin/bash"}


@pytest.fixture(autouse=True)
def _defaults(monkeypatch):
    monkeypatch.delenv("BENCHFLOW_DAYTONA_ACP_TRANSPORT", raising=False)
    monkeypatch.delenv("BENCHFLOW_REVIEWER_ACP_TRANSPORT", raising=False)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name, *a, **kw: "/usr/bin/ssh" if name == "ssh" else real_which(name),
    )


class _Observed(FakeRun):
    """The reviewer's rollout: what it would select, its config.json, and
    the live process its Daytona sandbox would open."""

    def __init__(self) -> None:
        super().__init__(review_payload=good_weighted_review())
        self.selected: list[str] = []
        self.recorded: list[str] = []
        self.opened: list[tuple[str, dict]] = []

    async def __call__(self, config):
        from benchflow.rollout._results import _write_config
        from benchflow.skill_policy import (
            SKILL_MODE_NO_SKILL,
            resolve_task_skill_policy,
        )

        self.selected.append(
            selected_acp_transport(agent=config.agent, environment=config.environment)
        )
        leaf = Path(config.jobs_dir) / "provenance"
        leaf.mkdir(parents=True)
        _write_config(
            leaf,
            task_path=config.task_path,
            agent=config.agent,
            model=config.model,
            environment=config.environment,
            skill_policy=resolve_task_skill_policy(
                task_path=config.task_path,
                skill_mode=SKILL_MODE_NO_SKILL,
                runtime_skills_dir=None,
                declared_sandbox_skills_dir=None,
            ),
            sandbox_user=None,
            context_root=None,
            timeout=300,
            started_at=datetime(2026, 11, 2),
            agent_env={},
            purpose="reviewer",
        )
        self.recorded.append(
            json.loads((leaf / "config.json").read_text())["acp_transport"]
        )
        env = SimpleNamespace()
        with (
            patch.object(
                process_pkg.DaytonaPtyProcess,
                "from_sandbox_env",
                new_callable=AsyncMock,
            ) as pty,
            patch.object(
                process_pkg.DaytonaProcess, "from_sandbox_env", new_callable=AsyncMock
            ) as ssh,
        ):
            await DaytonaSandbox.live_process(
                env, agent=config.agent, admission=admitted(env)
            )
        for name, opened in (("pty", pty), ("ssh", ssh)):
            if opened.await_count:
                self.opened.append((name, dict(opened.await_args.kwargs)))
        return await super().__call__(config)


async def _review(tmp_path: Path, monkeypatch) -> _Observed:
    observed = _Observed()
    monkeypatch.setattr(benchflow, "run", observed)
    task = make_task(tmp_path, with_rubric=True, rubric_data=WEIGHTED_RUBRIC)
    source = make_rollout(tmp_path / "jobs", "rollout-a", task_path=task)
    rubric_path = task / "verifier/rubric.json"
    trial = await run_review(
        source,
        task,
        load_rubric(rubric_path),
        rubric_path,
        ReviewerConfig(
            agent="codex-acp", model="azure/gpt-5.6-sol", environment="daytona"
        ),
        tmp_path / "review-output",
        deterministic_pass=True,
    )
    assert trial.review_valid, trial.error
    return observed


async def test_a_review_talks_ssh_on_daytona_and_says_so(tmp_path, monkeypatch):
    """The reviewer's rollout selects ssh, opens DaytonaProcess with the PTY
    session's terminal values, and its config.json records ssh; the solver
    of the same process still selects the PTY."""
    observed = await _review(tmp_path, monkeypatch)

    assert observed.selected == ["ssh"]
    assert observed.recorded == ["ssh"]
    assert observed.opened == [("ssh", {"terminal_env": _PTY_SESSION_ENV})]
    assert selected_acp_transport(agent="codex-acp", environment="daytona") == "pty"


async def test_the_reviewer_can_keep_the_pty(tmp_path, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_REVIEWER_ACP_TRANSPORT", "pty")
    observed = await _review(tmp_path, monkeypatch)

    assert observed.selected == observed.recorded == ["pty"]
    assert [name for name, _ in observed.opened] == ["pty"]


async def test_a_process_already_on_ssh_keeps_it_for_both(tmp_path, monkeypatch):
    """The solver's own setting is untouched by the reviewer's."""
    monkeypatch.setenv("BENCHFLOW_DAYTONA_ACP_TRANSPORT", "ssh")
    observed = await _review(tmp_path, monkeypatch)

    assert observed.selected == ["ssh"]
    assert selected_acp_transport(agent="codex-acp", environment="daytona") == "ssh"


def test_an_invalid_reviewer_transport_is_ssh(monkeypatch, caplog):
    from benchflow.acp import selection

    monkeypatch.setenv("BENCHFLOW_REVIEWER_ACP_TRANSPORT", "websocket")
    with caplog.at_level(logging.WARNING), selection.review_rollout():
        assert selected_acp_transport(agent="opencode", environment="daytona") == "ssh"
        assert selected_acp_transport(agent="opencode", environment="docker") == (
            "docker-stdio"
        )
    assert "BENCHFLOW_REVIEWER_ACP_TRANSPORT" in caplog.text
    assert selected_acp_transport(agent="opencode", environment="daytona") == "pty"


def test_without_an_ssh_client_the_reviewer_keeps_the_pty(monkeypatch, caplog):
    """The ssh launch runs the machine's ssh client: without one, an unset
    switch keeps 0.9's PTY (said once); an explicit ssh is honoured."""
    from benchflow.acp import selection

    monkeypatch.setattr(shutil, "which", lambda *_a, **_kw: None)
    monkeypatch.setattr(selection, "_no_ssh_client_logged", False)
    with caplog.at_level(logging.WARNING), selection.review_rollout():
        assert selected_acp_transport(agent="opencode", environment="daytona") == "pty"
        assert selected_acp_transport(agent="opencode", environment="daytona") == "pty"
        monkeypatch.setenv("BENCHFLOW_REVIEWER_ACP_TRANSPORT", "ssh")
        assert selected_acp_transport(agent="opencode", environment="daytona") == "ssh"
    assert caplog.text.count("No ssh client") == 1


def _sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox.create_ssh_access = AsyncMock(return_value=MagicMock(token="t"))
    sandbox.process.exec = AsyncMock(
        return_value=MagicMock(exit_code=0, result="__BENCHFLOW_BOOTSTRAP_DONE__\n")
    )
    return sandbox


async def _launched_env(**process_kwargs) -> dict[str, str]:
    """The values the ssh launch's env file is written with."""
    sandbox = _sandbox()
    process = DaytonaProcess(sandbox=sandbox, **process_kwargs)
    live = SimpleNamespace(
        pid=1,
        returncode=None,
        stdin=AsyncMock(),
        stdout=AsyncMock(),
        stderr=AsyncMock(),
    )
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=live)):
        await process.start(
            "codex-acp", env={"SHELL": "/usr/bin/zsh", "OPENAI_API_KEY": "sk-x"}
        )
    env = dict(sandbox.process.exec.await_args.kwargs["env"])
    for name in SYSTEM_SHELL_EMPTY_VARS:
        env.pop(name, None)
    return env


async def test_the_reviewers_ssh_launch_carries_the_pty_sessions_terminal():
    """TERM and SHELL as the PTY session sets them, under the agent env's own
    values; neither on a launch that never ran on the PTY, nor on a compose
    task, where compose's exec gives the agent no terminal either way."""
    moved = await _launched_env(is_dind=False, terminal_env=_PTY_SESSION_ENV)
    assert moved == {
        "TERM": "xterm-256color",
        "SHELL": "/usr/bin/zsh",
        "OPENAI_API_KEY": "sk-x",
    }
    assert await _launched_env(is_dind=False) == {
        "SHELL": "/usr/bin/zsh",
        "OPENAI_API_KEY": "sk-x",
    }
    compose = await _launched_env(
        is_dind=True,
        compose_cmd_base="docker compose -p t",
        terminal_env=_PTY_SESSION_ENV,
    )
    assert "TERM" not in compose
