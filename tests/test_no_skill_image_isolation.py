"""Regression tests for no-skill rollout isolation.

A with-skill build rewrites the task Dockerfile in place (``COPY _deps/skills``
plus symlinks into every agent skill path). While the image tag was derived from
the task name alone, that build and a no-skill build of the same task shared one
tag, so whichever ran last decided what both started from — and a no-skill
rollout could silently run inside an image containing ``/skills``.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from benchflow.agents.install import assert_no_skill_isolation
from benchflow.sandbox.docker import _build_context_fingerprint


def _write_env(tmp_path, dockerfile_body: str, skills: dict[str, str] | None = None):
    env_dir = tmp_path / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "Dockerfile").write_text(dockerfile_body)
    if skills:
        for rel, content in skills.items():
            target = env_dir / "_deps" / "skills" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content)
    return env_dir


BASE_DOCKERFILE = "FROM python:3.12-slim\nCOPY payload.stl /root/payload.stl\n"
SKILL_INJECTED = BASE_DOCKERFILE + (
    "\n# Skills directory (injected by benchflow --skills-dir)\n"
    "COPY _deps/skills /skills/\n"
    "RUN mkdir -p /home/agent/.opencode && ln -sf /skills /home/agent/.opencode/skills\n"
)


def test_fingerprint_is_stable_for_identical_context(tmp_path):
    a = _write_env(tmp_path / "a", BASE_DOCKERFILE)
    b = _write_env(tmp_path / "b", BASE_DOCKERFILE)
    assert _build_context_fingerprint(a) == _build_context_fingerprint(b)


def test_skill_injection_changes_the_fingerprint(tmp_path):
    """The core regression: the two builds must not share an image tag."""
    clean = _write_env(tmp_path / "clean", BASE_DOCKERFILE)
    injected = _write_env(
        tmp_path / "injected",
        SKILL_INJECTED,
        skills={"mentor/SKILL.md": "# mentor\n"},
    )
    assert _build_context_fingerprint(clean) != _build_context_fingerprint(injected)


def test_changing_skill_payload_changes_the_fingerprint(tmp_path):
    one = _write_env(
        tmp_path / "one", SKILL_INJECTED, skills={"mentor/SKILL.md": "# mentor\n"}
    )
    two = _write_env(
        tmp_path / "two",
        SKILL_INJECTED,
        skills={"mentor/SKILL.md": "# mentor\n", "second/SKILL.md": "# second\n"},
    )
    assert _build_context_fingerprint(one) != _build_context_fingerprint(two)


class _FakeEnv:
    """Minimal env double returning canned `find` output."""

    def __init__(self, stdout: str):
        self._stdout = stdout
        self.commands: list[str] = []

    async def exec(self, cmd: str, timeout_sec: int = 20):
        self.commands.append(cmd)

        class _Result:
            stdout = self._stdout
            stderr = ""
            return_code = 0

        return _Result()


class _FakeAgentCfg:
    skill_paths: ClassVar[list[str]] = ["$HOME/.opencode/skills"]


@pytest.mark.asyncio
async def test_isolation_check_passes_when_no_skills_present():
    env = _FakeEnv("")
    await assert_no_skill_isolation(env, _FakeAgentCfg(), "agent")
    assert "/home/agent/.opencode/skills" in env.commands[0]
    assert "/skills" in env.commands[0]


@pytest.mark.asyncio
async def test_isolation_check_raises_when_skills_are_reachable():
    env = _FakeEnv("/skills/ion-shuttling-mentor/SKILL.md\n")
    with pytest.raises(RuntimeError, match="no-skill rollout is contaminated"):
        await assert_no_skill_isolation(env, _FakeAgentCfg(), "agent")


@pytest.mark.asyncio
async def test_isolation_check_handles_missing_agent_config():
    """The oracle path passes no agent config; the mount point is still checked."""
    env = _FakeEnv("")
    await assert_no_skill_isolation(env, None, None)
    assert "/skills" in env.commands[0]


@pytest.mark.asyncio
async def test_isolation_check_ignores_non_skill_output():
    """Shell noise on stdout must not be mistaken for a contaminated sandbox."""
    env = _FakeEnv("/workspace\nsome unrelated line\n")
    await assert_no_skill_isolation(env, _FakeAgentCfg(), "agent")
