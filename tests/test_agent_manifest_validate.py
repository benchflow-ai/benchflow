"""Custom-agent manifests: a strict check, one discovery path, the handshake.

A file study of the CLI scored the custom-agent manifest 4/10, BenchFlow's
lowest (run/hub/pm-views-20261117/G-cli-ux-vs-harbor.md, "Three format
changes"). Its three injected mistakes got no field or file diagnosis:
``modle_route`` was dropped and the run asked for a model, ``model_route =
7`` made the agent vanish ("Unknown agent"), and a missing manifest was
"Unknown agent" too. ``.env``'s ``BENCHFLOW_AGENTS_DIR`` picked the agent
for ``bench run`` but not for ``bench agents``. Guards rc6's change 1
(``bench agents validate``, invalid manifests visible in discovery) and
change 3 (one discovery path and dotenv precedence, a manifest named by its
bytes, the zero-event handshake). No Docker: the fixture is
tests/fixtures/custom_agent, the study's marker-file agent and marker task.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from benchflow.agents.manifest_check import check_manifest, manifest_identity
from benchflow.cli.main import app

FIXTURE = Path(__file__).parent / "fixtures" / "custom_agent"
MANIFEST = Path("agents") / "marker-file" / "manifest.toml"


@pytest.fixture
def project(tmp_path, monkeypatch):
    """The fixture copied to a fresh folder, run from there, with no shell
    export of BENCHFLOW_AGENTS_DIR: only its .env names the agents folder."""
    from benchflow.agents import registry, remote_manifests

    root = tmp_path / "project"
    shutil.copytree(FIXTURE, root)
    (root / ".env").write_text("BENCHFLOW_AGENTS_DIR=agents\n")
    monkeypatch.chdir(root)
    # The .env bench reads from the folder it runs in (conftest points the
    # lookup at a per-test file; here that file is the project's own).
    monkeypatch.setenv("BENCHFLOW_DOTENV_PATH", str(root / ".env"))
    monkeypatch.delenv("BENCHFLOW_AGENTS_DIR", raising=False)
    monkeypatch.setenv("BENCHFLOW_AGENTS_SOURCE", "off")
    remote_manifests._reset_for_tests()
    maps = (
        registry.AGENTS,
        registry.AGENT_ALIASES,
        registry.AGENT_INSTALLERS,
        registry.AGENT_LAUNCH,
    )
    snapshots = tuple(mapping.copy() for mapping in maps)
    yield root
    remote_manifests._reset_for_tests()
    for mapping, snapshot in zip(maps, snapshots, strict=True):
        mapping.clear()
        mapping.update(snapshot)


def _edit(root: Path, old: str, new: str) -> None:
    path = root / MANIFEST
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))


def _run(*args: str):
    result = CliRunner().invoke(app, list(args))
    return result.exit_code, " ".join(result.output.split())


# Change 1: bench agents validate -------------------------------------------


def test_a_valid_manifest_passes_and_is_named_by_its_bytes(project):
    code, out = _run("agents", "validate", str(MANIFEST))
    assert code == 0, out
    assert "marker-file" in out and "sha256:" in out


def test_a_misspelled_key_names_its_line_and_the_key_it_meant(project):
    _edit(project, 'model_route = "agent"', 'modle_route = "agent"')
    code, out = _run("agents", "validate", str(MANIFEST))
    assert code == 1, out
    assert (
        "manifest.toml:2: modle_route: unknown key (did you mean 'model_route'?)" in out
    )


def test_a_wrong_type_names_its_line_and_what_it_must_be(project):
    _edit(project, 'model_route = "agent"', "model_route = 7")
    code, out = _run("agents", "validate", str(MANIFEST))
    assert code == 1, out
    assert (
        "manifest.toml:2: model_route: expected one of 'proxy', 'agent'; got 7" in out
    )


def test_a_missing_manifest_names_the_path(project):
    (project / MANIFEST).unlink()
    code, out = _run("agents", "validate", str(MANIFEST))
    assert code == 2, out
    assert "agents/marker-file/manifest.toml" in out and "no manifest" in out


def test_an_unknown_contract_version_is_refused(project):
    _edit(project, 'contract_version = "1.1"', 'contract_version = "1.9"')
    problems = check_manifest(project / MANIFEST)
    assert [p.key for p in problems] == ["contract_version"]
    assert problems[0].line == 1 and "1.0, 1.1" in problems[0].message


def test_lines_inside_an_inline_script_are_not_keys():
    # install_cmd's Python has "result = {}" and "method = ..." lines.
    assert check_manifest(FIXTURE / MANIFEST) == []


# Discovery: invalid manifests stay visible; one path for agents and run ----


def test_dotenv_names_the_agents_folder_for_bench_agents_too(project):
    code, out = _run("agents", "marker-file")
    assert code == 0, out
    assert "Model route: agent" in out


def test_an_invalid_manifest_is_listed_as_invalid_with_its_problem(project):
    _edit(project, 'model_route = "agent"', "model_route = 7")
    _code, out = _run("agents")
    assert "marker-file" in out and "invalid" in out
    assert "model_route" in out


@pytest.mark.parametrize(
    ("old", "new", "said"),
    [
        ('model_route = "agent"', 'modle_route = "agent"', "modle_route: unknown key"),
        ('model_route = "agent"', "model_route = 7", "model_route: expected one of"),
    ],
)
def test_a_run_of_an_invalid_manifest_says_which_field(project, old, new, said):
    _edit(project, old, new)
    code, out = _run("run", "marker", "-a", "marker-file", "--dry-run")
    assert code != 0, out
    assert said in out and "agents/marker-file/manifest.toml" in out
    assert "bench agents validate" in out


def test_a_missing_agents_folder_is_named(project):
    shutil.rmtree(project / "agents")
    code, out = _run("run", "marker", "-a", "marker-file", "--dry-run")
    assert code != 0, out
    assert "BENCHFLOW_AGENTS_DIR" in out and "agents" in out and "no folder" in out


# Change 3: the same manifest moved is the same agent -----------------------


def test_a_moved_fixture_keeps_its_manifest_identity(project, tmp_path):
    first = manifest_identity(project / MANIFEST, project / "agents")
    moved = tmp_path / "elsewhere" / "fresh"
    shutil.copytree(project, moved)
    again = manifest_identity(moved / MANIFEST, moved / "agents")
    assert first == again
    assert first.relative_path == "marker-file/manifest.toml"


# The handshake: a turn with zero ACP events is an integration failure -------


def test_a_manifest_agent_s_silent_turn_is_judged(monkeypatch):
    from benchflow import integration_health
    from benchflow.agents import remote_manifests

    monkeypatch.setattr(
        remote_manifests, "manifest_agent_names", lambda: frozenset({"marker-file"})
    )
    failure = integration_health.diagnose(
        [],
        agent="marker-file",
        n_tool_calls=0,
        output_tokens=None,
        logs={},
        agent_seconds=4.0,
        prompt_sent=True,
    )
    assert failure is not None and failure.cause == "empty_trajectory"


def test_an_empty_turn_says_how_to_fix_it():
    from benchflow.failures import _INTEGRATION

    headline, fault, step = _INTEGRATION["empty_trajectory"]
    assert "no ACP event" in headline and fault == "agent"
    assert "session/update" in step


def test_an_agent_folder_without_its_manifest_is_named(project):
    (project / MANIFEST).unlink()
    code, out = _run("run", "marker", "-a", "marker-file", "--dry-run")
    assert code != 0, out
    assert "agents/marker-file/manifest.toml does not exist" in out


# A catalog manifest never silently changes a core agent ------------

_CORE_NAMED = """contract_version = "1.1"
name = "claude-agent-acp"
install_cmd = "true"
launch_cmd = "some-other-launcher --acp"
"""


@pytest.fixture
def core_named(project, monkeypatch):
    folder = project / "agents" / "claude-agent-acp"
    folder.mkdir()
    (folder / "manifest.toml").write_text(_CORE_NAMED)
    monkeypatch.setenv("BENCHFLOW_AGENTS_DIR", "agents")
    return project


def test_a_manifest_that_differs_from_a_core_agent_does_not_replace_it(core_named):
    from benchflow.agents import registry, remote_manifests

    core = registry._CORE_AGENT_CONFIGS["claude-agent-acp"]
    assert registry.resolve_agent("claude-agent-acp") == core
    warnings = " ".join(remote_manifests.ensure_manifest_catalog().warnings)
    assert "names the core agent 'claude-agent-acp' but differs from it in" in warnings
    assert "launch_cmd" in warnings and "BENCHFLOW_AGENTS_OVERRIDE_CORE" in warnings


def test_the_override_runs_the_manifest_by_name_and_says_so(core_named, monkeypatch):
    from benchflow.agents import registry, remote_manifests

    monkeypatch.setenv("BENCHFLOW_AGENTS_OVERRIDE_CORE", "1")
    assert registry.resolve_agent("claude-agent-acp").launch_cmd == (
        "some-other-launcher --acp"
    )
    warnings = " ".join(remote_manifests.ensure_manifest_catalog().warnings)
    assert "replaces the core agent 'claude-agent-acp'" in warnings


def test_a_core_agent_kept_has_no_agent_source(core_named):
    from benchflow.agents import registry, remote_manifests

    registry.resolve_agent("claude-agent-acp")
    assert remote_manifests.agent_source("claude-agent-acp") is None


def test_the_override_is_said_at_start_and_recorded_in_every_record(
    core_named, monkeypatch, tmp_path
):
    """B3 of the part-2 review: under the opt-in, a run of a core agent's name
    that a local manifest replaced says so, and config.json, result.json and
    the run card name the manifest and its bytes."""
    import hashlib
    import json

    from benchflow.agents import registry, remote_manifests
    from benchflow.rollout import _build_rollout_result, _write_config
    from benchflow.run_card import build_run_card, card_lines
    from tests.test_config_redaction import _no_skill_policy
    from tests.test_run_card import _evaluation, _tasks

    monkeypatch.setenv("BENCHFLOW_AGENTS_OVERRIDE_CORE", "1")
    registry.resolve_agent("claude-agent-acp")
    manifest = core_named / "agents" / "claude-agent-acp" / "manifest.toml"
    expected = {
        "manifest": "agents/claude-agent-acp/manifest.toml",
        "sha256": "sha256:" + hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "overrides_core": True,
    }
    source = remote_manifests.agent_source("claude-agent-acp")
    assert source is not None and "launch_cmd" in source.pop("differs_in")
    assert source == expected
    said = remote_manifests.core_override("claude-agent-acp").text()
    assert said.startswith(
        "claude-agent-acp runs from agents/claude-agent-acp/manifest.toml (sha256:"
    )
    assert "not BenchFlow's core claude-agent-acp" in said

    trial = tmp_path / "trial"
    trial.mkdir()
    _write_config(
        trial,
        task_path=tmp_path / "task",
        agent="claude-agent-acp",
        model="deepseek/deepseek-flash",
        environment="docker",
        skill_policy=_no_skill_policy(tmp_path / "task"),
        sandbox_user=None,
        context_root=None,
        timeout=300,
        started_at=__import__("datetime").datetime(2026, 11, 18),
        agent_env={},
    )
    config = json.loads((trial / "config.json").read_text())
    assert config["agent_source"]["sha256"] == expected["sha256"]
    _build_rollout_result(
        trial,
        task_name="state",
        rollout_name="state__1",
        agent="claude-agent-acp",
        agent_name="claude-agent-acp",
        model="deepseek/deepseek-flash",
        n_tool_calls=0,
        prompts=["solve"],
        error=None,
        verifier_error=None,
        trajectory=[],
        partial_trajectory=False,
        rewards={"reward": 1.0},
        started_at=__import__("datetime").datetime.now(),
        timing={"agent": 1.0},
    )
    result = json.loads((trial / "result.json").read_text())
    assert result["agent_source"]["overrides_core"] is True
    assert result["agent_source"]["manifest"] == expected["manifest"]

    job = _evaluation(tmp_path, _tasks(tmp_path, 1), agent="claude-agent-acp")
    card = build_run_card(job, job._get_task_dirs())
    assert card["agent"]["source"]["sha256"] == expected["sha256"]
    agent_line = next(
        line for line in card_lines(card, tmp_path / "j") if "Agent:" in line
    )
    assert (
        "from the local manifest agents/claude-agent-acp/manifest.toml, which "
        "replaces the core agent"
    ) in agent_line


def test_bench_agents_shows_the_override(core_named, monkeypatch):
    monkeypatch.setenv("BENCHFLOW_AGENTS_OVERRIDE_CORE", "1")
    code, out = _run("agents", "claude-agent-acp")
    assert code == 0, out
    assert (
        "Source: agents/claude-agent-acp/manifest.toml (sha256:" in out
        and "replacing the core agent" in out
    )
