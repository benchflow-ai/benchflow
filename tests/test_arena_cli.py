"""Multi-agent floor is the standard `bench eval run --agents` (plural of --agent)."""

from __future__ import annotations

import textwrap

import pytest
import typer
from typer.testing import CliRunner

from benchflow.cli.main import app

runner = CliRunner()


def _roster(tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text(
        textwrap.dedent("""
        agents:
          - { name: codex, agent: codex-acp }
    """)
    )
    return p


def test_eval_run_accepts_agents_and_drive_options(tmp_path):
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agents",
            str(_roster(tmp_path)),
            "--drive",
            "service-rounds",
        ],
    )
    assert r.exit_code == 1
    assert "--agents requires --environment-manifest" in r.output


def test_agents_requires_environment_manifest(tmp_path):
    # --agents without --environment-manifest fails closed (no docker touched).
    r = runner.invoke(app, ["eval", "run", "--agents", str(_roster(tmp_path))])
    assert r.exit_code == 1


def test_agents_mutually_exclusive_with_agent(tmp_path):
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agents",
            str(_roster(tmp_path)),
            "--agent",
            "codex-acp",
            "--environment-manifest",
            "x.toml",
        ],
    )
    assert r.exit_code == 1


def test_arena_run_is_deprecated_alias(tmp_path):
    # The old `arena run` still works (hidden alias) and requires the manifest.
    r = runner.invoke(app, ["arena", "run", "--agents", str(_roster(tmp_path))])
    assert r.exit_code != 0  # missing required --environment-manifest


def test_eval_run_agents_threads_floor_flags(tmp_path, monkeypatch):
    # The floor-shape flags fold onto the standard `eval run` and reach the runner.
    from benchflow.cli import arena as arena_mod

    captured = {}
    monkeypatch.setattr(
        arena_mod, "run_floor_from_cli", lambda **kw: captured.update(kw)
    )
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agents",
            str(_roster(tmp_path)),
            "--environment-manifest",
            "benchmarks/casinobench/environment.toml",
            "--game",
            "six-deck-blackjack-s17",
            "--service-env",
            "CASINO_MULTIPLAYER=1",
            "--url-env",
            "CASINO_URL",
            "--seat-env",
            "CASINOBENCH_SEAT_ID",
            "--standings-path",
            "/_admin/standings",
            "--events-path",
            "/_admin/events",
            "--reasoning-effort",
            "MAX",
            "--usage-tracking",
            "required",
            "--agent-idle-timeout",
            "0",
        ],
    )
    assert r.exit_code == 0, r.output
    assert captured["service_env"] == ["CASINO_MULTIPLAYER=1"]
    assert captured["game"] == "six-deck-blackjack-s17"
    assert captured["url_env"] == "CASINO_URL"
    assert captured["seat_env"] == "CASINOBENCH_SEAT_ID"
    assert captured["standings_path"] == "/_admin/standings"
    assert captured["events_path"] == "/_admin/events"
    assert captured["reasoning_effort"] == "MAX"
    assert captured["usage_tracking"] == "required"
    assert captured["agent_idle_timeout"] == "0"


def test_arena_alias_does_not_pass_removed_multiplayer_kwarg(tmp_path, monkeypatch):
    """Guards PR #846 against reintroducing the stale multiplayer kwarg alias."""
    from benchflow.cli import arena as arena_mod

    captured = {}
    monkeypatch.setattr(
        arena_mod, "run_floor_from_cli", lambda **kw: captured.update(kw)
    )
    r = runner.invoke(
        app,
        [
            "arena",
            "run",
            "--agents",
            str(_roster(tmp_path)),
            "--environment-manifest",
            "benchmarks/casinobench/environment.toml",
            "--multiplayer",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "multiplayer" not in captured


def test_arena_alias_threads_current_floor_controls(tmp_path, monkeypatch):
    """Guards PR #846 so the deprecated alias stays wired to the floor surface."""
    from benchflow.cli import arena as arena_mod

    captured = {}
    monkeypatch.setattr(
        arena_mod, "run_floor_from_cli", lambda **kw: captured.update(kw)
    )
    r = runner.invoke(
        app,
        [
            "arena",
            "run",
            "--agents",
            str(_roster(tmp_path)),
            "--environment-manifest",
            "benchmarks/casinobench/environment.toml",
            "--service-env",
            "CASINO_MULTIPLAYER=1",
            "--deadline",
            "0",
            "--reasoning-effort",
            "max",
            "--usage-tracking",
            "required",
            "--agent-idle-timeout",
            "none",
        ],
    )
    assert r.exit_code == 0, r.output
    assert captured["service_env"] == ["CASINO_MULTIPLAYER=1"]
    assert captured["deadline_s"] == 0
    assert captured["reasoning_effort"] == "max"
    assert captured["usage_tracking"] == "required"
    assert captured["agent_idle_timeout"] == "none"


def test_arena_run_accepts_current_floor_controls(tmp_path):
    """Guards PR #846 against hidden alias option drift."""
    r = runner.invoke(
        app,
        [
            "arena",
            "run",
            "--agents",
            str(_roster(tmp_path)),
            "--environment-manifest",
            str(tmp_path / "environment.toml"),
            "--sandbox",
            "fake",
            "--deadline",
            "0",
            "--service-env",
            "CASINO_MULTIPLAYER=1",
            "--reasoning-effort",
            "max",
            "--usage-tracking",
            "required",
            "--agent-idle-timeout",
            "none",
        ],
    )
    assert r.exit_code == 1
    assert "Invalid --sandbox 'fake': choose docker or daytona" in r.output


def test_floor_rejects_unknown_sandbox_before_bootstrap(tmp_path, monkeypatch):
    """Guards PR #846 so typoed floor sandboxes fail before Docker/Daytona work."""
    from benchflow.arena import bootstrap
    from benchflow.cli.arena import run_floor_from_cli

    async def fail_run_native_floor(*args, **kwargs):
        raise AssertionError("run_native_floor should not be reached")

    monkeypatch.setattr(bootstrap, "run_native_floor", fail_run_native_floor)
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--agents",
            str(_roster(tmp_path)),
            "--environment-manifest",
            "benchmarks/casinobench/environment.toml",
            "--sandbox",
            "fake",
        ],
    )
    assert r.exit_code == 1
    assert "Invalid --sandbox 'fake': choose docker or daytona" in r.output

    with pytest.raises(typer.Exit):
        run_floor_from_cli(
            agents=_roster(tmp_path),
            environment_manifest=tmp_path / "environment.toml",
            out=tmp_path / "out",
            sandbox="fake",
        )


def test_run_floor_from_cli_normalizes_run_level_knobs(tmp_path, monkeypatch):
    """Guards PR #846 run-level controls through the native floor entrypoint."""
    from benchflow.arena import bootstrap
    from benchflow.cli.arena import run_floor_from_cli

    captured = {}

    async def fake_run_native_floor(
        roster, *, environment_manifest, config, game=None, service_env=None
    ):
        captured["config"] = config
        captured["service_env"] = service_env
        return {
            "results": [
                {
                    "seat": "codex",
                    "status": "ok",
                    "acp_tool_calls": 1,
                    "llm_calls": 0,
                    "raw": False,
                }
            ]
        }

    monkeypatch.setattr(bootstrap, "run_native_floor", fake_run_native_floor)
    run_floor_from_cli(
        agents=_roster(tmp_path),
        environment_manifest=tmp_path / "environment.toml",
        out=tmp_path / "out",
        reasoning_effort="MAX",
        usage_tracking="required",
        agent_idle_timeout="0",
        service_env=["CASINO_MULTIPLAYER=1"],
        prompt="play",
    )
    cfg = captured["config"]
    assert cfg.reasoning_effort == "max"
    assert cfg.usage_tracking.mode == "required"
    assert cfg.idle_timeout_s is None
    assert captured["service_env"] == {"CASINO_MULTIPLAYER": "1"}
