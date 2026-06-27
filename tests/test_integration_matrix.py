"""Tests for the deterministic integration-test matrix planner (7-set taxonomy).

Stdlib-only (no ``benchflow`` import). Exercises the real ``scope_map.yml`` /
``scope_defaults.yml`` data files and the public planner functions/CLI against
the Default-config-rules table, the MATRIX CELL SCHEMA, and the SPEC top-level
output. Every Default-config-rules row is covered by name.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "integration_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("integration_matrix", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mx = _load_module()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _maps():
    return mx.load_maps()


def _plan(files: Sequence[str], **kw):
    maps = _maps()
    return mx.build_plan(
        list(files),
        maps,
        base_ref=kw.pop("base_ref", "main"),
        head_sha=kw.pop("head_sha", "deadbeefcafe"),
        override=kw.pop("override", "auto"),
        custom_tasks=kw.pop("custom_tasks", ()),
        sandboxes_override=kw.pop("sandboxes_override", ()),
        skill_modes_override=kw.pop("skill_modes_override", ()),
    )


def _ids(plan) -> list[str]:
    return [c.id for c in plan.matrix]


def _tasks(plan) -> set[str]:
    return {c.task for c in plan.matrix}


def _agents(plan) -> set[str]:
    return {c.agent for c in plan.matrix}


def _skill_modes(plan) -> set[str]:
    return {c.skill_mode for c in plan.matrix}


def _sandboxes(plan) -> set[str]:
    return {c.sandbox for c in plan.matrix}


# ------------------------------------------------------------------
# Map / defaults loading
# ------------------------------------------------------------------


def test_maps_load_and_caps_self_consistent():
    maps = _maps()
    assert maps.caps.max_cells >= maps.caps.max_agents * maps.caps.max_tasks
    assert maps.caps.aggregate_concurrency <= 24
    assert maps.caps.agent_idle_timeout == 240
    # Roster = 5 DeepSeek agents + 2 gated natives (claude-agent-acp, codex-acp).
    # gemini and harvey-lab-harness were dropped (cannot run on DeepSeek).
    assert len(maps.agents) == 7
    assert "gemini" not in maps.agents
    assert "harvey-lab-harness" not in maps.agents
    assert maps.baseline_agent == "openhands"
    assert maps.baseline_model == "deepseek/deepseek-v4-flash"
    assert maps.canonical_high_task == "weighted-gdp-calc"
    assert maps.citation_vehicle == "citation-check"


def test_task_sets_taxonomy():
    maps = _maps()
    ts = maps.task_sets
    assert ts["citation"] == ["citation-check"]
    assert ts["low-smoke"] == ["jax-computing-basics"]
    assert ts["low-3"] == [
        "jax-computing-basics",
        "python-scala-translation",
        "jpg-ocr-stat",
    ]
    assert ts["medium-3"] == [
        "grid-dispatch-operator",
        "threejs-to-obj",
        "data-to-d3",
    ]
    assert ts["high-3"] == [
        "lake-warming-attribution",
        "weighted-gdp-calc",
        "shock-analysis-supply",
    ]
    # nine = low-3 + medium-3 + high-3
    assert ts["nine"] == ts["low-3"] + ts["medium-3"] + ts["high-3"]


def test_miniyaml_fallback_matches_pyyaml():
    # The planner must run under bare python3 without pyyaml.
    import yaml  # pyyaml available in the test env

    map_text = (REPO_ROOT / ".github/integration/scope_map.yml").read_text()
    def_text = (REPO_ROOT / ".github/integration/scope_defaults.yml").read_text()

    py_map = yaml.safe_load(map_text)
    py_def = yaml.safe_load(def_text)
    mini_map = mx._MiniYaml(map_text).parse()
    mini_def = mx._MiniYaml(def_text).parse()

    def norm(obj) -> str:
        return json.dumps(obj, sort_keys=True, default=str)

    assert norm(py_map) == norm(mini_map)
    assert norm(py_def) == norm(mini_def)


# ------------------------------------------------------------------
# Default-config-rules table — one test per row
# ------------------------------------------------------------------


def test_docs_only_nonruntime_no_rollout():
    plan = _plan(["docs/guide.md", "README.md"])
    assert plan.matrix == []
    assert plan.scope == "none"
    # level reported as L0 for the no-op lane.
    assert plan.buckets == ["docs-nonruntime"]
    assert plan.rejected_overflow is None


def test_citation_evidence_schema_docs():
    plan = _plan(["docs/examples/task-md/real-skillsbench/citation-check/SKILL.md"])
    assert plan.scope == "citation"
    assert _tasks(plan) == {"citation-check"}
    assert _sandboxes(plan) == {"docker"}
    assert _skill_modes(plan) == {"no-skill"}


def test_eval_rollout_schema_is_nine():
    plan = _plan(["src/benchflow/eval_runner.py"])
    assert plan.scope == "nine"
    assert _tasks(plan) == set(_maps().task_sets["nine"])
    # judge axis present (extra carries 'judge').
    assert all(c.judge_model for c in plan.matrix)
    assert _sandboxes(plan) == {"docker"}


def test_rollout_lifecycle_is_nine():
    plan = _plan(["src/benchflow/rollout/engine.py"])
    assert plan.scope == "nine"
    assert plan.trust_boundary is True


def test_artifact_schema_trajectories_is_nine():
    plan = _plan(["src/benchflow/trajectories/atif.py"])
    assert plan.scope == "nine"


def test_agents_rule_low3_plus_one_high_affected_and_baseline():
    maps = _maps()
    plan = _plan(["src/benchflow/agents/codex_config.py"])
    assert plan.scope == "low-3"
    # low-3 + the canonical one-high task.
    assert _tasks(plan) == set(maps.task_sets["low-3"]) | {maps.canonical_high_task}
    # affected agent (codex-acp) + baseline agent (openhands), no full 9.
    assert _agents(plan) == {"codex-acp", maps.baseline_agent}
    # no-skill AND with-skill.
    assert _skill_modes(plan) == {"no-skill", "with-skill"}
    assert _sandboxes(plan) == {"daytona"}
    # baseline agent carries the baseline (flash) model, except the one high
    # task (weighted-gdp-calc), which DeepSeek tiering promotes to the pro model.
    baseline_cells = [c for c in plan.matrix if c.agent == maps.baseline_agent]
    for c in baseline_cells:
        expected = maps.pro_model if c.task in maps.pro_tasks else maps.baseline_model
        assert c.model == expected


def test_deepseek_tiering_flash_for_low_pro_for_high():
    maps = _maps()
    # Tiering data: pro_tasks defaults to the high-3 set; pro model is wired.
    assert maps.pro_model == "deepseek/deepseek-v4-pro"
    assert maps.pro_tasks == frozenset(maps.task_sets["high-3"])
    # high-3: every baseline (openhands/flash-lane) cell is promoted to pro.
    high = _plan([], override="high-3")
    high_models = {c.model for c in high.matrix if c.agent == maps.baseline_agent}
    assert high_models == {maps.pro_model}
    # low-3: baseline cells keep the cheap flash model.
    low = _plan([], override="low-3")
    low_models = {c.model for c in low.matrix if c.agent == maps.baseline_agent}
    assert low_models == {maps.baseline_model}


def test_agents_rule_acp_shim_path_map():
    maps = _maps()
    plan = _plan(["src/benchflow/agents/openclaw_acp_shim.py"])
    assert "openclaw" in _agents(plan)
    assert maps.baseline_agent in _agents(plan)


def test_agent_runtime_infra_fans_roster_subset():
    # Changes to the registry / shared ACP infra affect EVERY agent, but at L2
    # (auto-on-push) we fan only the representative SUBSET — one agent per
    # credential/launcher family — not the full 9-agent roster. The FULL roster
    # is reachable at L3/expanded (scope=nine/expanded).
    maps = _maps()
    expected = set(maps.roster_subset) | {maps.baseline_agent}
    for f in (
        "src/benchflow/agents/registry.py",
        "src/benchflow/agents/protocol.py",
        "src/benchflow/acp/session.py",
    ):
        plan = _plan([f])
        assert _agents(plan) == expected, f
        # The L2 subset is strictly smaller than the full roster.
        assert _agents(plan) != set(maps.agents), f


def test_provider_change_fans_roster_subset():
    # A provider / LLM-proxy / routing change affects every agent's model calls,
    # but at L2 it runs the representative SUBSET, not the whole roster.
    maps = _maps()
    plan = _plan(["src/benchflow/providers/litellm_runtime.py"])
    assert _agents(plan) == set(maps.roster_subset) | {maps.baseline_agent}
    assert _agents(plan) != set(maps.agents)
    assert plan.network_lane is True


def test_expanded_fans_deepseek_roster_only():
    # At L3/expanded the broad fan is the DeepSeek roster ONLY (the agent IS the
    # varying axis via _FULL_ROSTER_SCOPES). The gated native agents
    # (claude-agent-acp, codex-acp) are "blocked" from the default fan and never
    # appear here — they run only via affected-agent.
    maps = _maps()
    plan = _plan([], override="expanded")
    assert _agents(plan) == set(maps.deepseek_roster)
    assert "claude-agent-acp" not in _agents(plan)
    assert "codex-acp" not in _agents(plan)


def test_expanded_override_beats_roster_subset_extra():
    # Regression (_matrix_agents ordering): an agent-runtime-infra file like
    # registry.py carries ``extra: all-agents-subset`` (the L2 breadth tier), but
    # when promoted to L3/expanded the FULL-roster scope MUST win and fan the
    # whole DeepSeek roster -- not the smaller L2 subset. Before the ordering fix
    # the all-agents-subset branch was evaluated first and silently capped
    # expanded back to the breadth subset.
    maps = _maps()
    plan = _plan(["src/benchflow/agents/registry.py"], override="expanded")
    assert plan.scope == "expanded"
    assert _agents(plan) == set(maps.deepseek_roster)
    # ... strictly larger than the L2 breadth subset it would get on auto-push.
    assert _agents(plan) != set(maps.roster_subset) | {maps.baseline_agent}
    # The gating still holds: the heavy lane does not pull in the natives.
    assert "claude-agent-acp" not in _agents(plan)


def test_model_policy_open_agents_on_deepseek():
    # The openai-completions-family agents that proxy cleanly through the
    # LiteLLM usage proxy run on deepseek/deepseek-v4-flash: openclaw, opencode,
    # pi-acp, mimo (mimo on deepseek REPLACES xiaomi). openhands keeps its
    # baseline deepseek model too.
    maps = _maps()
    for agent in ("openclaw", "opencode", "pi-acp", "mimo"):
        assert maps.model_for(agent) == "deepseek/deepseek-v4-flash", agent
    # mimo is no longer on xiaomi.
    assert "xiaomi" not in maps.model_for("mimo")
    # The gated native agent keeps its native model (OpenAI Responses).
    assert maps.model_for("codex-acp") == "gpt-5.4-nano"
    # gemini & harvey are dropped from the roster entirely (cannot use DeepSeek).
    assert "gemini" not in maps.agents
    assert "harvey-lab-harness" not in maps.agents


def test_claude_on_bedrock():
    # claude-agent-acp routes through Bedrock's anthropic-messages surface, not
    # the bare claude-haiku native id.
    maps = _maps()
    assert maps.model_for("claude-agent-acp").startswith("aws-bedrock/")
    assert (
        maps.model_for("claude-agent-acp")
        == "aws-bedrock/us.anthropic.claude-haiku-4-5-20251001"
    )


def test_specific_agent_change_does_not_fan_full_roster():
    # Regression: a SPECIFIC agent file stays affected-agent + baseline (NOT the
    # whole roster), so per-agent PRs remain cheap and targeted.
    maps = _maps()
    plan = _plan(["src/benchflow/agents/codex_config.py"])
    assert _agents(plan) == {"codex-acp", maps.baseline_agent}
    assert _agents(plan) != set(maps.agents)


def test_gated_native_runs_only_via_affected_agent():
    # The gated natives (claude-agent-acp, codex-acp) cannot use DeepSeek, so
    # they are blocked from the broad fan and run ONLY when their own adapter
    # changes. This is the one path "using other models as needed to test that
    # agent": a claude*.py change DOES run claude-agent-acp (on its Bedrock
    # model) + the DeepSeek baseline for before/after comparison...
    maps = _maps()
    plan = _plan(["src/benchflow/agents/claude_agent_acp.py"])
    assert _agents(plan) == {"claude-agent-acp", maps.baseline_agent}
    # ...while the broad DeepSeek lanes never pull the natives in.
    assert "claude-agent-acp" not in maps.deepseek_roster
    assert "codex-acp" not in maps.deepseek_roster
    assert set(maps.deepseek_roster) <= set(maps.agents)


def test_co_changed_gated_native_survives_broad_fan():
    # Regression: a gated native reaches the matrix ONLY via affected-agent, so a
    # PR that co-changes its adapter AND a broad-fan trigger must STILL run it —
    # the broad branch must not win first and silently drop the changed native.
    maps = _maps()
    # registry.py (agent-runtime-infra -> all-agents-subset) + codex_config.py
    # (agents-adapters -> affected-agent=codex-acp): subset + baseline + codex.
    sub = _plan(
        [
            "src/benchflow/agents/registry.py",
            "src/benchflow/agents/codex_config.py",
        ]
    )
    assert "codex-acp" in _agents(sub), "co-changed native dropped from subset lane"
    assert set(maps.roster_subset) | {maps.baseline_agent} <= _agents(sub)
    # pyproject.toml (release-critical -> expanded) + claude*.py (affected-agent):
    # full DeepSeek roster + the changed native, even at expanded scope.
    exp = _plan(["pyproject.toml", "src/benchflow/agents/claude_agent_acp.py"])
    assert exp.scope == "expanded"
    assert "claude-agent-acp" in _agents(exp), "co-changed native dropped at expanded"
    assert set(maps.deepseek_roster) <= _agents(exp)


def test_all_agents_extra_fans_deepseek_roster_only():
    # The manual/heavy ``all-agents`` fan is DeepSeek-only: the gated natives
    # are excluded by policy (blocked in the real workflow currently).
    maps = _maps()
    plan = _plan([], override="expanded")
    # expanded carries the broad fan; confirm it equals the DeepSeek roster.
    assert _agents(plan) == set(maps.deepseek_roster)
    assert all(
        maps.model_for(a).startswith("deepseek/") or a == maps.baseline_agent
        for a in _agents(plan)
    )


def test_skill_loading_rule_low3_medium3_both_skill_modes_audit():
    maps = _maps()
    plan = _plan([".agents/skills/some-skill/SKILL.md"])
    assert plan.scope == "medium-3"
    # low-3 + medium-3 (low-3-plus extra).
    expected = set(maps.task_sets["low-3"]) | set(maps.task_sets["medium-3"])
    assert _tasks(plan) == expected
    assert _skill_modes(plan) == {"no-skill", "with-skill"}
    # skill-catalog extraction requested via audit_skills.
    assert all(c.audit_skills for c in plan.matrix)


def test_sandbox_root_path_rule_low3_medium3_docker_daytona_parity():
    maps = _maps()
    plan = _plan(["src/benchflow/sandbox/docker.py"])
    assert plan.scope == "medium-3"
    expected = set(maps.task_sets["low-3"]) | set(maps.task_sets["medium-3"])
    assert _tasks(plan) == expected
    assert _sandboxes(plan) == {"docker", "daytona"}
    assert plan.trust_boundary is True


def test_verifier_rewards_judge_rule_custom_with_cheat():
    plan = _plan(["src/benchflow/verifier/score.py"])
    assert plan.scope == "custom"
    assert _tasks(plan) == {
        "citation-check",
        "weighted-gdp-calc",
        "shock-analysis-supply",
    }
    assert plan.cheat is True
    assert any(c.id.endswith("-cheat") for c in plan.matrix)
    assert any(c.expect_reward == "<1.0" for c in plan.matrix)


def test_network_package_rule_allowlist_variant_network_lane():
    maps = _maps()
    plan = _plan(["src/benchflow/providers/openai.py"])
    assert plan.scope == "custom"
    assert plan.network_lane is True
    # jax + data-to-d3 + one high.
    assert {"jax-computing-basics", "data-to-d3", "weighted-gdp-calc"} <= _tasks(plan)
    # the allowlist VARIANT cell carries network_mode=allowlist (EXPECTED only).
    allowlist = [c for c in plan.matrix if c.network_mode == "allowlist"]
    assert len(allowlist) == 1
    assert allowlist[0].task == maps.citation_vehicle
    # default cells are network-off.
    assert all(
        c.network_mode == "default-off"
        for c in plan.matrix
        if not c.id.endswith("-allowlist")
    )


def test_network_lockdown_glob_triggers_lane():
    plan = _plan(["src/benchflow/sandbox/lockdown.py"])
    assert plan.network_lane is True


def test_release_critical_refactor_is_expanded_rerun_base():
    plan = _plan(["src/benchflow/cli/main.py"])
    assert plan.scope == "expanded"
    assert plan.baseline == "rerun-base"
    assert plan.trust_boundary is True


# ------------------------------------------------------------------
# MATRIX CELL SCHEMA + top-level output shape
# ------------------------------------------------------------------


def test_plan_json_schema_shape():
    plan = _plan(["src/benchflow/sandbox/docker.py"])
    data = plan.to_json()
    assert data["schema_version"] == "1"
    for key in (
        "head_sha",
        "base_ref",
        "scope",
        "buckets",
        "trust_boundary",
        "cheat",
        "network_lane",
        "baseline",
        "caps",
        "matrix",
        "residual_risk",
        "rejected_overflow",
    ):
        assert key in data
    for cell in data["matrix"]:
        for key in (
            "id",
            "level",
            "task",
            "agent",
            "model",
            "judge_model",
            "sandbox",
            "skill_mode",
            "network_mode",
            "timeout_minutes",
            "agent_idle_timeout",
            "audit_skills",
            "expect_reward",
        ):
            assert key in cell
        assert cell["sandbox"] in ("docker", "daytona")
        assert cell["skill_mode"] in ("no-skill", "with-skill", "self-gen")
        assert cell["network_mode"] in ("default-off", "allowlist")
        assert cell["level"] in ("light", "scope", "final")
        assert cell["agent_idle_timeout"] == 240
        assert cell["expect_reward"] in ("==1.0", "<1.0", "any")


# ------------------------------------------------------------------
# Overflow / hard ceiling
# ------------------------------------------------------------------


def test_overflow_sets_rejected_and_keeps_cells():
    maps = _maps()
    maps.caps = mx.Caps(
        max_cells=10,
        max_agents=9,
        max_tasks=9,
        per_agent_concurrency=2,
        aggregate_concurrency=24,
        agent_idle_timeout=240,
        comment_trials_cap=3,
    )
    plan = mx.build_plan(
        ["src/benchflow/eval_runner.py"],
        maps,
        base_ref="main",
        head_sha="h",
    )
    assert plan.rejected_overflow is not None
    # No silent drop.
    assert len(plan.matrix) > 10


def test_cli_overflow_exits_two(tmp_path, monkeypatch):
    real_load = mx.load_maps

    def small_load(*a, **k):
        maps = real_load(*a, **k)
        maps.caps = mx.Caps(
            max_cells=5,
            max_agents=9,
            max_tasks=9,
            per_agent_concurrency=2,
            aggregate_concurrency=24,
            agent_idle_timeout=240,
            comment_trials_cap=3,
        )
        return maps

    monkeypatch.setattr(mx, "load_maps", small_load)
    out = tmp_path / "matrix.json"
    code = mx.main(
        [
            "--base-ref",
            "main",
            "--head-sha",
            "h",
            "--changed-file",
            "src/benchflow/eval_runner.py",
            "--out",
            str(out),
        ]
    )
    assert code == mx.EXIT_OVERFLOW
    data = json.loads(out.read_text())
    assert data["rejected_overflow"]


# ------------------------------------------------------------------
# Aggregate concurrency ceiling
# ------------------------------------------------------------------


def test_aggregate_concurrency_under_ceiling_for_daytona_matrix():
    plan = _plan(["src/benchflow/sandbox/docker.py"])
    distinct = {c.agent for c in plan.matrix if c.sandbox == "daytona" and c.agent}
    aggregate = plan.caps.per_agent_concurrency * max(1, len(distinct))
    assert aggregate <= plan.caps.aggregate_concurrency
    assert aggregate <= 24


def test_per_agent_concurrency_clamped_for_full_daytona_roster():
    # sandbox rule runs both sandboxes with the full 9-agent roster on daytona.
    plan = _plan(["src/benchflow/sandbox/docker.py"])
    distinct = {c.agent for c in plan.matrix if c.sandbox == "daytona"}
    assert plan.caps.per_agent_concurrency * len(distinct) <= 24


# ------------------------------------------------------------------
# affected-agent path map
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,agent",
    [
        ("src/benchflow/agents/codex_config.py", "codex-acp"),
        ("src/benchflow/agents/openclaw_acp_shim.py", "openclaw"),
        ("src/benchflow/agents/claude_agent_acp.py", "claude-agent-acp"),
        ("src/benchflow/agents/pi_acp_launcher.py", "pi-acp"),
    ],
)
def test_affected_agent_path_map(path, agent):
    maps = _maps()
    assert mx.affected_agents([path], maps) == [agent]


def test_affected_agents_empty_when_no_agent_path():
    maps = _maps()
    assert mx.affected_agents(["src/benchflow/sandbox/docker.py"], maps) == []


# ------------------------------------------------------------------
# Explicit --scope override
# ------------------------------------------------------------------


def test_scope_override_forces_set_on_docs():
    plan = _plan(["docs/x.md"], override="low-3")
    assert plan.scope == "low-3"
    assert _tasks(plan) == set(_maps().task_sets["low-3"])


def test_scope_override_custom_uses_custom_tasks():
    plan = _plan(
        ["docs/x.md"],
        override="custom",
        custom_tasks=("weighted-gdp-calc", "data-to-d3"),
    )
    assert plan.scope == "custom"
    assert _tasks(plan) == {"weighted-gdp-calc", "data-to-d3"}


def test_scope_override_nine_on_unmapped_path():
    plan = _plan(["some/unmapped/path.txt"], override="nine")
    assert plan.scope == "nine"
    assert _tasks(plan) == set(_maps().task_sets["nine"])


def test_sandbox_and_skill_mode_overrides():
    plan = _plan(
        ["src/benchflow/agents/codex_config.py"],
        sandboxes_override=("docker",),
        skill_modes_override=("no-skill",),
    )
    assert _sandboxes(plan) == {"docker"}
    assert _skill_modes(plan) == {"no-skill"}


# ------------------------------------------------------------------
# Cell id stability + uniqueness
# ------------------------------------------------------------------


def test_cell_ids_unique_across_all_scopes():
    for files in (
        ["src/benchflow/eval_runner.py"],
        ["src/benchflow/agents/codex_config.py"],
        ["src/benchflow/sandbox/docker.py"],
        ["src/benchflow/verifier/score.py"],
        ["src/benchflow/providers/openai.py"],
    ):
        plan = _plan(files)
        ids = _ids(plan)
        assert len(ids) == len(set(ids)), files


def test_cell_ids_stable_across_runs():
    a = _ids(_plan(["src/benchflow/sandbox/docker.py"]))
    b = _ids(_plan(["src/benchflow/sandbox/docker.py"]))
    assert a == b


def test_no_match_diff_degrades_to_citation():
    plan = _plan(["some/unmapped/path.txt"])
    assert plan.scope == "citation"
    assert plan.buckets == []
    assert len(plan.matrix) == 1


# ------------------------------------------------------------------
# CLI GITHUB_OUTPUT emission
# ------------------------------------------------------------------


def test_cli_writes_matrix_json(tmp_path):
    out = tmp_path / "matrix.json"
    code = mx.main(
        [
            "--base-ref",
            "main",
            "--head-sha",
            "abc123",
            "--changed-file",
            "src/benchflow/agents/codex_config.py",
            "--out",
            str(out),
        ]
    )
    assert code == mx.EXIT_OK
    data = json.loads(out.read_text())
    assert data["head_sha"] == "abc123"
    assert data["scope"] == "low-3"
    assert data["matrix"]


def test_cli_github_output_emits_matrix_and_head_sha(tmp_path, monkeypatch):
    gh_out = tmp_path / "gh_output"
    gh_out.write_text("")
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
    code = mx.main(
        [
            "--base-ref",
            "main",
            "--head-sha",
            "feedface",
            "--changed-file",
            "src/benchflow/agents/codex_config.py",
            "--out",
            str(gh_out),
        ]
    )
    assert code == mx.EXIT_OK
    text = gh_out.read_text()
    assert "head_sha=feedface" in text
    assert "matrix=" in text
    # the matrix value parses as JSON.
    line = next(ln for ln in text.splitlines() if ln.startswith("matrix="))
    json.loads(line[len("matrix=") :])


def test_caps_inconsistency_fails_closed(monkeypatch):
    real_load_yaml = mx._load_yaml

    def patched(path):
        data = real_load_yaml(path)
        if path == mx.SCOPE_DEFAULTS_PATH:
            data = json.loads(json.dumps(data))  # deep copy
            data["caps"]["max_cells"] = 2  # < max_agents*max_tasks
        return data

    monkeypatch.setattr(mx, "_load_yaml", patched)
    with pytest.raises(mx.ScopeError):
        mx.load_maps()
