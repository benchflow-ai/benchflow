"""PERF-D: one Compose check per set of inputs in a process.

A job's admission, the trial factory and the trial's own admission ask
``check_compose`` the same question of the same task; the answer is kept
under a key over every input it reads, so ``docker compose config`` runs once,
and any changed input (a file's bytes, mode or link, a value, the sandbox, the
process environment) checks again. A refusal stays a refusal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchflow.task import compose_env

MAIN = "services:\n  main:\n    image: fixture-main:latest\n"


def _task(root: Path, compose: str = MAIN) -> Path:
    environment = root / "task" / "environment"
    environment.mkdir(parents=True)
    (environment / "docker-compose.yaml").write_text(compose)
    (environment / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    return environment


@pytest.fixture
def resolutions(monkeypatch) -> list[list[str]]:
    """Every ``docker compose ... config`` the check runs (faked: no daemon)."""
    seen: list[list[str]] = []

    def fake(command, env, *, project_only=False):
        seen.append(list(command))
        return compose_env._Resolved(
            project={"services": {"main": {"image": "fixture-main:latest"}}},
            variables={},
            environment=frozenset(),
        )

    monkeypatch.setattr(compose_env, "_run_config", fake)
    return seen


def test_the_same_compose_check_resolves_once(tmp_path, resolutions):
    environment = _task(tmp_path)

    first = compose_env.check_compose(environment, {"A": "1"})
    second = compose_env.check_compose(environment, {"A": "1"})
    third = compose_env.check_compose(environment, {"A": "1"})

    assert len(resolutions) == 1
    assert first.refused == second.refused == third.refused
    assert first.defaulted == second.defaulted == third.defaulted
    # Each caller gets its own answer: one caller's edits never reach another.
    assert first is not second and first.refused is not second.refused
    first.refused.append("caller's own")
    assert (
        "caller's own" not in compose_env.check_compose(environment, {"A": "1"}).refused
    )


def test_any_changed_input_checks_again(tmp_path, resolutions, monkeypatch):
    environment = _task(tmp_path)
    compose_env.check_compose(environment, {"A": "1"})
    assert len(resolutions) == 1

    # The same file with other bytes (same size, same name).
    (environment / "docker-compose.yaml").write_text(MAIN.replace("latest", "lates2"))
    compose_env.check_compose(environment, {"A": "1"})
    assert len(resolutions) == 2

    # A file added anywhere in the folder.
    (environment / "extra.env").write_text("X=1\n")
    compose_env.check_compose(environment, {"A": "1"})
    assert len(resolutions) == 3

    # Another value from the task.
    compose_env.check_compose(environment, {"A": "2"})
    assert len(resolutions) == 4

    # Another sandbox.
    compose_env.check_compose(environment, {"A": "2"}, sandbox_type="remote-docker")
    assert len(resolutions) == 5

    # The process environment (it carries the Docker connection).
    monkeypatch.setenv("DOCKER_HOST", "unix:///tmp/another.sock")
    compose_env.check_compose(environment, {"A": "2"}, sandbox_type="remote-docker")
    assert len(resolutions) == 6


def test_a_changed_mode_or_link_checks_again(tmp_path, resolutions):
    environment = _task(tmp_path)
    compose_env.check_compose(environment)
    (environment / "Dockerfile").chmod(0o755)
    compose_env.check_compose(environment)
    (environment / "link").symlink_to("Dockerfile")
    compose_env.check_compose(environment)
    (environment / "link").unlink()
    (environment / "link").symlink_to("docker-compose.yaml")
    compose_env.check_compose(environment)
    assert len(resolutions) == 4


def test_host_access_never_keeps_an_answer(tmp_path, resolutions, monkeypatch):
    """With host access the check may read files outside the folder, which the
    key does not cover: every call checks."""
    monkeypatch.setenv(compose_env.ALLOW_HOST_ACCESS_VAR, "1")
    environment = _task(tmp_path)
    compose_env.check_compose(environment)
    compose_env.check_compose(environment)
    assert len(resolutions) == 2


def test_a_refusal_is_kept_as_refused(tmp_path, monkeypatch):
    """A refused project stays refused for every caller (nothing is relaxed by
    keeping the answer)."""
    calls: list[int] = []

    def refuse(command, env, *, project_only=False):
        calls.append(1)
        return compose_env._Resolved(error="no such image")

    monkeypatch.setattr(compose_env, "_run_config", refuse)
    environment = _task(tmp_path)
    first = compose_env.check_compose(environment)
    second = compose_env.check_compose(environment)
    assert first.refused and second.refused
    assert [r.shown for r in first.refused] == [r.shown for r in second.refused]
    assert len(calls) == 1
