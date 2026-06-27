"""Byte-identical parity gate: every core agent reproduced by a benchflow-ai/agents
manifest must match its in-core ``registry.AGENTS`` entry, field for field.

The agent-decoupling plan (decision #4) makes each agent ship as *data* — a
``manifest.toml`` in ``benchflow-ai/agents`` — instead of a hand-authored entry in
``registry.AGENTS``. The future "step 6" flips the source of truth (default to the
manifests, then delete the in-core defs). That flip is only safe if every manifest
that reproduces an existing core agent is **byte-identical** to the core entry —
otherwise deleting the core def would silently change behaviour. This file is that
gate: it must be green for all manifest-backed core agents before step 6 proceeds.

Why a pinned-ref clone (the most CI-robust shape)
-------------------------------------------------
A session-scoped fixture clones ``benchflow-ai/agents`` at a single pinned commit
(``AGENTS_REPO_PIN``) into a throwaway dir, then loads the manifests with the
production ``load_agents_from_dir``. Pinning the exact SHA makes the gate:

* deterministic — the exact commit yields the exact manifests, reproducible run
  to run (a moving branch tip could flip the result between identical pushes);
* offline after the one clone — no network mid-test, no flaky live API;
* hermetic — no ``$BENCHFLOW_AGENTS_DIR`` / env setup; the pin lives in source;
* auditable + updatable — the commit is visible here; when the agents repo
  advances you bump ``AGENTS_REPO_PIN`` and re-run, and any drift surfaces at
  once as a named per-field failure telling on-call to sync core or the manifest.

When the clone is impossible (no ``git`` / offline dev box) the live checks SKIP
with a clear reason rather than fail; the hermetic synthetic-drift test below
still exercises the comparison logic on every run, so the gate is never a no-op.

Scope
-----
Only agents that have BOTH a ``manifest.toml`` AND a core ``AGENTS`` entry are
compared. ``omnigent-pi`` (a session-factory package agent wired through the #825
seam) and the ai-sdk / register.py package agents ship no manifest and are out of
scope by construction. Only the 14 manifest-owned data fields (== ``_FIELD_MAP``)
are compared; the remaining ``AgentConfig`` fields are ``_SHIM_ONLY`` host/credential
concerns a data manifest never carries.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchflow.agents.manifest import (
    _FIELD_MAP,
    MANIFEST_DIR_ENV,
    load_agents_from_dir,
)
from benchflow.agents.registry import AGENTS, AgentConfig

# Pinned ref of benchflow-ai/agents this gate compares against. eb6c60d == agents
# PR #24 "restore chmod 600 on auth.json in launch_cmd (parity with core #825)".
# Bump this when the agents repo advances and re-run locally; any new drift fails
# loudly with the agent + field named.
AGENTS_REPO_URL = "https://github.com/benchflow-ai/agents.git"
AGENTS_REPO_PIN = "eb6c60da6de3cad331e79424d4048048d5399a2e"

# Core agents whose data ALSO lives in a benchflow-ai/agents manifest.toml.
# Hardcoded (not discovered) so a vanished/renamed manifest, or a core agent that
# loses its manifest, fails THIS gate loudly instead of silently shrinking the set.
# omnigent-pi is the documented exception: a session-factory package agent (#825
# seam), not a manifest — see test_parity_set_is_exactly_the_core_manifest_agents.
MANIFEST_CORE_AGENTS = (
    "claude-agent-acp",
    "codex-acp",
    "deepagents",
    "gemini",
    "harvey-lab-harness",
    "mimo",
    "openclaw",
    "opencode",
    "openhands",
    "pi-acp",
)

# The 14 manifest-owned data fields (== _FIELD_MAP.values()). The remaining
# AgentConfig fields are _SHIM_ONLY host/credential concerns the manifest never
# carries, so they are out of scope for byte-parity. test_parity_fields_track_
# the_loader_contract pins this tuple to _FIELD_MAP so a newly-added contract data
# field can never silently escape the gate.
PARITY_FIELDS = (
    "name",
    "description",
    "install_cmd",
    "launch_cmd",
    "protocol",
    "api_protocol",
    "acp_model_format",
    "supports_acp_set_model",
    "requires_env",
    "install_timeout",
    "env_mapping",
    "default_model",
    "skill_paths",
    "home_dirs",
)


def _clone_pinned(dest: Path) -> None:
    """init + fetch the single pinned commit by SHA, then check it out.

    ``git fetch --depth 1 origin <sha>`` pulls exactly the pinned tree (GitHub
    allows fetching a reachable SHA), so nothing but the pin is downloaded — more
    deterministic than ``clone --depth 1`` of a moving branch tip.
    """
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(
        ["git", "-C", str(dest), "remote", "add", "origin", AGENTS_REPO_URL],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(dest),
            "fetch",
            "-q",
            "--depth",
            "1",
            "origin",
            AGENTS_REPO_PIN,
        ],
        check=True,
        timeout=180,
    )
    subprocess.run(
        ["git", "-C", str(dest), "checkout", "-q", AGENTS_REPO_PIN],
        check=True,
    )


@pytest.fixture(scope="session")
def manifest_agents(tmp_path_factory: pytest.TempPathFactory) -> dict[str, AgentConfig]:
    """``{name: AgentConfig}`` loaded once from a pinned, hermetic agents clone.

    Clones ``AGENTS_REPO_PIN`` into a throwaway session dir and runs the production
    ``load_agents_from_dir(repo/"acp")``. Skips (does not fail) when the comparison
    cannot be trusted or performed: ``$BENCHFLOW_AGENTS_DIR`` set (then in-process
    AGENTS is already manifest-merged and cannot be the pure-core baseline), no
    ``git``, or the clone fails (offline).
    """
    if os.environ.get(MANIFEST_DIR_ENV):
        pytest.skip(
            f"${MANIFEST_DIR_ENV} is set: registry.AGENTS is already merged with "
            "that manifest dir at import, so it cannot serve as the pure-core "
            "baseline for byte-parity. Unset it to run this gate."
        )
    if shutil.which("git") is None:
        pytest.skip("git unavailable; cannot clone the pinned agents repo")
    repo = tmp_path_factory.mktemp("agents-repo")
    try:
        _clone_pinned(repo)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        pytest.skip(
            f"could not clone benchflow-ai/agents@{AGENTS_REPO_PIN[:12]}: {exc}"
        )
    return {name: lm.config for name, lm in load_agents_from_dir(repo / "acp").items()}


@pytest.mark.parametrize("agent", MANIFEST_CORE_AGENTS)
def test_manifest_byte_matches_core(
    agent: str, manifest_agents: dict[str, AgentConfig]
) -> None:
    """Each core agent is byte-identical to its pinned manifest on the 14
    manifest-owned data fields. Fails loudly naming the agent + drifted fields so
    on-call knows exactly what to reconcile before the source-of-truth flip."""
    assert agent in AGENTS, f"{agent!r} is not in core registry.AGENTS"
    assert agent in manifest_agents, (
        f"core agent {agent!r} has NO manifest at benchflow-ai/agents@"
        f"{AGENTS_REPO_PIN[:12]}; the gate requires one (or drop it from "
        "MANIFEST_CORE_AGENTS if the manifest was removed on purpose)."
    )
    core_cfg = AGENTS[agent]
    man_cfg = manifest_agents[agent]
    drift = {
        field: {"manifest": getattr(man_cfg, field), "core": getattr(core_cfg, field)}
        for field in PARITY_FIELDS
        if getattr(man_cfg, field) != getattr(core_cfg, field)
    }
    assert not drift, (
        f"PARITY DRIFT for agent {agent!r}: its manifest (benchflow-ai/agents@"
        f"{AGENTS_REPO_PIN[:12]}) and core registry.AGENTS disagree on field(s) "
        f"{sorted(drift)} -> {drift}. The manifest is the source of truth for the "
        "agent decouple; reconcile registry.AGENTS with the manifest (or bump "
        "AGENTS_REPO_PIN if core is intentionally ahead) before the step-6 flip."
    )


def test_parity_set_is_exactly_the_core_manifest_agents(
    manifest_agents: dict[str, AgentConfig],
) -> None:
    """The core<->manifest intersection is EXACTLY ``MANIFEST_CORE_AGENTS``.

    Adding a manifest-backed core agent (or removing one) must update this gate
    deliberately rather than silently widen/shrink coverage. Together with the
    per-agent test this keeps ``omnigent-pi`` the sole core agent without a manifest.
    """
    core_with_manifest = {name for name in AGENTS if name in manifest_agents}
    assert core_with_manifest == set(MANIFEST_CORE_AGENTS), (
        "core<->manifest intersection drifted from the gated set: unexpected="
        f"{sorted(core_with_manifest - set(MANIFEST_CORE_AGENTS))}, missing="
        f"{sorted(set(MANIFEST_CORE_AGENTS) - core_with_manifest)}. Update "
        "MANIFEST_CORE_AGENTS deliberately when adding/removing a manifest-backed "
        "core agent (omnigent-pi stays manifest-less by design)."
    )


def test_parity_fields_track_the_loader_contract() -> None:
    """``PARITY_FIELDS`` must equal the manifest-owned data fields the loader maps
    (``_FIELD_MAP``). If the contract gains a data field, this gate must compare it
    — fail here rather than let a new, uncompared field drift silently. Hermetic."""
    assert set(PARITY_FIELDS) == set(_FIELD_MAP.values()), (
        "PARITY_FIELDS is out of sync with manifest._FIELD_MAP: missing="
        f"{sorted(set(_FIELD_MAP.values()) - set(PARITY_FIELDS))}, extra="
        f"{sorted(set(PARITY_FIELDS) - set(_FIELD_MAP.values()))}."
    )


def test_parity_assertion_catches_synthetic_drift() -> None:
    """Load-bearing + hermetic: prove the comparison REJECTS a perturbed field.

    A gate that silently passes on everything is indistinguishable from a no-op.
    Build a core-like AgentConfig, perturb exactly one data field, and assert the
    same per-field comparison the live test performs flags exactly that field.
    No clone / no network, so the gate logic runs even when the live fixture skips.
    """
    core_cfg = AgentConfig(name="drift-probe", install_cmd="echo i", launch_cmd="REAL")
    drifted = dataclasses.replace(core_cfg, launch_cmd="DRIFTED")
    flagged = {
        field
        for field in PARITY_FIELDS
        if getattr(drifted, field) != getattr(core_cfg, field)
    }
    assert flagged == {"launch_cmd"}, flagged
