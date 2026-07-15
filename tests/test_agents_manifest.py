"""Parse + resolve `--agents agents.yaml` into concurrent-floor seats."""

from __future__ import annotations

import textwrap

import pytest

from benchflow.arena.agents_manifest import AgentsManifest


def _write(tmp_path, body: str, name: str = "agents.yaml"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def test_minimal_prebuilt_parses_with_defaults(tmp_path):
    p = _write(
        tmp_path,
        """
        task: casino
        prompt: play the game
        agents:
          - { name: codex, agent: codex-acp, model: gpt-5.5 }
        """,
    )
    m = AgentsManifest.from_yaml(p)
    assert m.task == "casino"
    assert m.drive == "auto-loop"  # default
    assert m.deadline_s == 1200  # default
    seats = m.seats()
    assert [s.seat_id for s in seats] == ["codex"]
    assert seats[0].config.name == "codex-acp"  # resolved AgentConfig
    assert seats[0].agent_cwd == "/work/codex"


def test_count_fans_out_seats(tmp_path):
    p = _write(
        tmp_path,
        """
        agents:
          - { name: cx, agent: codex-acp, count: 3 }
          - { name: solo, agent: claude-agent-acp }
        """,
    )
    seats = AgentsManifest.from_yaml(p).seats()
    assert [s.seat_id for s in seats] == ["cx-0", "cx-1", "cx-2", "solo"]
    # claude family instruction file rides on the resolved config (P2)
    assert seats[-1].config.name == "claude-agent-acp"


def test_missing_name_defaults_to_agent_model_id(tmp_path):
    p = _write(
        tmp_path,
        """
        agents:
          - { agent: codex-acp, model: gpt-5.5 }
          - { agent: deepagents, model: deepseek/deepseek-v4-pro }
        """,
    )
    seats = AgentsManifest.from_yaml(p).seats()
    assert [s.seat_id for s in seats] == [
        "codex-acp-gpt-5.5",
        "deepagents-deepseek-v4-pro",
    ]


def test_exactly_one_of_agent_or_manifest(tmp_path):
    both = _write(
        tmp_path,
        """
        agents:
          - { name: x, agent: codex-acp, manifest: m.toml }
        """,
    )
    with pytest.raises(ValueError, match="exactly one"):
        AgentsManifest.from_yaml(both)

    neither = _write(
        tmp_path,
        """
        agents:
          - { name: x, model: gpt-5.5 }
        """,
    )
    with pytest.raises(ValueError, match="exactly one"):
        AgentsManifest.from_yaml(neither)


def test_duplicate_seat_ids_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        agents:
          - { name: dup, agent: codex-acp }
          - { name: dup, agent: claude-agent-acp }
        """,
    )
    with pytest.raises(ValueError, match="duplicate seat id"):
        AgentsManifest.from_yaml(p).seats()


def test_unknown_top_level_key_rejected(tmp_path):
    p = _write(
        tmp_path,
        """
        nonsense: 1
        agents:
          - { name: x, agent: codex-acp }
        """,
    )
    with pytest.raises(ValueError):
        AgentsManifest.from_yaml(p)


def test_byoa_manifest_resolves_and_registers(tmp_path):
    (tmp_path / "my.toml").write_text(
        textwrap.dedent("""
        contract_version = "1.0"
        name = "my-byoa-agent"
        install_cmd = "true"
        launch_cmd = "my-agent --acp"
        default_model = "gpt-5.5"
    """)
    )
    p = _write(
        tmp_path,
        """
        agents:
          - { name: mine, manifest: my.toml }
    """,
    )
    seats = AgentsManifest.from_yaml(p).seats()
    cfg = seats[0].config
    assert cfg.name == "my-byoa-agent"
    assert cfg.launch_cmd == "my-agent --acp"
    assert cfg.protocol == "acp"
    # and it is now resolvable by name from the registry
    from benchflow.agents.registry import resolve_agent

    assert resolve_agent("my-byoa-agent").launch_cmd == "my-agent --acp"


def test_byoa_manifest_rejects_unknown_field(tmp_path):
    (tmp_path / "bad.toml").write_text(
        textwrap.dedent("""
        contract_version = "1.0"
        name = "bad"
        install_cmd = "true"
        launch_cmd = "x"
        bogus_field = "nope"
    """)
    )
    p = _write(
        tmp_path,
        """
        agents:
          - { name: bad, manifest: bad.toml }
    """,
    )
    with pytest.raises(ValueError):
        AgentsManifest.from_yaml(p).seats()


def test_paths_resolved_relative_to_yaml(tmp_path):
    sub = tmp_path / "cfg"
    sub.mkdir()
    (sub / "my.toml").write_text(
        textwrap.dedent("""
        contract_version = "1.0"
        name = "rel-agent"
        install_cmd = "true"
        launch_cmd = "x"
    """)
    )
    (sub / "instr.md").write_text("be bold")
    p = _write(
        sub,
        """
        agents:
          - { name: a, manifest: my.toml, instructions: instr.md }
    """,
    )
    m = AgentsManifest.from_yaml(p)
    seats = m.seats()
    assert seats[0].config.name == "rel-agent"
    # instructions path resolves against the yaml's directory
    assert m.instructions_path(seats[0].spec).read_text() == "be bold"
