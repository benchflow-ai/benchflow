"""Core <-> manifest byte-identical parity gate (the pre-flip source-of-truth lock).

The agent-decoupling plan moves each agent's *data* out of ``registry.AGENTS``
and into a ``manifest.toml`` in ``benchflow-ai/agents`` (decision #4). Before the
source-of-truth can be flipped (the future "step 6"), every manifest that
reproduces an existing core agent must merge back to a config that is
byte-identical to the hand-authored core entry — otherwise the flip would
silently change behaviour. This test is that gate.

Why a child interpreter for "pure core"
---------------------------------------
``registry.py`` calls ``register_env_manifest_agents()`` at import time, so when
``$BENCHFLOW_AGENTS_DIR`` is set (as the dedicated CI job sets it) the in-process
``AGENTS`` is *already* merged with those manifests. Comparing a manifest against
that merged ``AGENTS`` would compare it against *itself* — a silent false pass
that no drift could ever fail. So we recover the un-merged, hand-authored core
configs by re-importing ``registry`` in a child interpreter with
``$BENCHFLOW_AGENTS_DIR`` *unset* (the proven parity recipe), and compare the
live manifests against that.

What "byte-identical" means here
--------------------------------
A data-only manifest cannot carry the ``_SHIM_ONLY`` AgentConfig fields
(``subscription_auth``, ``credential_files``, ``acp_model_config_id``,
``disallow_web_tools_*``, ...); ``_merge_core_shim_only`` takes exactly those
from the core entry and everything else from the manifest. The merged result
must then equal the core entry on *every* dataclass field — i.e. the manifest's
14 data fields reproduce core exactly.

Lanes
-----
* Default ``pytest tests/`` (env unset): the live checks skip with a clear
  reason; the synthetic-drift test still runs (it is hermetic) so the comparison
  logic is exercised on every push.
* The dedicated parity-gate CI job clones ``benchflow-ai/agents@main``, sets
  ``$BENCHFLOW_AGENTS_DIR``, and runs this file — turning the live checks on.

Relationship to the sibling manifest tests: ``test_manifest_loader.py`` covers
parsing -> AgentConfig; ``test_manifest_dirscan.py`` / ``test_manifest_wiring.py``
cover discovery + registry merge mechanics. This file is orthogonal: it asserts
the *content* parity of the merge's output against the real agents repo.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import benchflow
from benchflow.agents.manifest import (
    MANIFEST_DIR_ENV,
    LoadedManifest,
    _merge_core_shim_only,
    load_agents_from_dir,
)
from benchflow.agents.registry import AgentConfig

# The documented exception: the only core agent that ships without a manifest.
# It is a session-factory agent wired through the #825 omnigent seam (a
# "module:callable" entrypoint), not data expressible as an acp-only manifest.
_CORE_ONLY_EXCEPTION = "omnigent-pi"

# Import roots so the child interpreter finds benchflow whether it is installed
# (CI) or only on PYTHONPATH from a worktree (dev VM) — mirrors the approach in
# test_manifest_wiring.py.
_SRC = Path(benchflow.__file__).resolve().parents[1]
_ROOT = _SRC.parent

_DUMP_PURE_CORE = (
    "import json, dataclasses;"
    "from benchflow.agents.registry import AGENTS;"
    "print(json.dumps({n: dataclasses.asdict(c) for n, c in AGENTS.items()}))"
)

_SKIP_REASON = (
    f"${MANIFEST_DIR_ENV} unset; the dedicated parity-gate CI job sets it to a "
    "shallow clone of benchflow-ai/agents@main to run the live parity checks"
)


@functools.cache
def _pure_core_agents() -> dict[str, AgentConfig]:
    """``registry.AGENTS`` as authored in core, with the manifest plane gated off.

    Re-imports ``registry`` in a child interpreter with ``$BENCHFLOW_AGENTS_DIR``
    unset so the import-time manifest merge is a no-op, then ships the configs
    back as JSON (``dataclasses.asdict``) and rebuilds them. The ``_SHIM_ONLY``
    fields (credential_files / subscription_auth) round-trip as plain dicts, but
    they are taken from this same core entry during the merge, so they compare
    equal by construction — only the data fields are meaningfully compared.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), str(_ROOT)])
    env.pop(MANIFEST_DIR_ENV, None)
    out = subprocess.run(
        [sys.executable, "-c", _DUMP_PURE_CORE],
        env=env,
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    raw = json.loads(out.stdout.strip().splitlines()[-1])
    return {name: AgentConfig(**cfg) for name, cfg in raw.items()}


@functools.cache
def _live_manifests() -> dict[str, LoadedManifest]:
    """Manifests loaded from ``$BENCHFLOW_AGENTS_DIR`` (the agents-repo clone)."""
    root = os.environ[MANIFEST_DIR_ENV]
    return load_agents_from_dir(root)


def _parity_param_names() -> list[str]:
    """Core agents that also have a manifest — the set to check for parity.

    Evaluated at collection time. Empty when ``$BENCHFLOW_AGENTS_DIR`` is unset
    (no child interpreter is spawned in that case)."""
    if not os.environ.get(MANIFEST_DIR_ENV):
        return []
    core = _pure_core_agents()
    loaded = _live_manifests()
    return [name for name in sorted(loaded) if name in core]


def _parametrization() -> list:
    names = _parity_param_names()
    if names:
        return names
    # Keep a visible, clearly-skipped item when the gate env is unset.
    return [pytest.param(None, marks=pytest.mark.skip(reason=_SKIP_REASON))]


@pytest.mark.parametrize("agent_name", _parametrization())
def test_manifest_byte_identical_to_core(agent_name: str | None) -> None:
    """Every core agent's manifest merges back to a byte-identical AgentConfig.

    The manifest owns the 14 data fields; ``_merge_core_shim_only`` restores the
    host-side ``_SHIM_ONLY`` bits from core; the merged result must equal core on
    every dataclass field. Any drift fails loudly with a per-field diff so the
    flip is never made on a manifest that changed behaviour.
    """
    core = _pure_core_agents()
    loaded = _live_manifests()
    core_cfg = core[agent_name]
    manifest_cfg = loaded[agent_name].config
    merged = _merge_core_shim_only(manifest_cfg, core_cfg)

    diffs = {
        f.name: (getattr(merged, f.name), getattr(core_cfg, f.name))
        for f in dataclasses.fields(core_cfg)
        if getattr(merged, f.name) != getattr(core_cfg, f.name)
    }
    assert not diffs, (
        f"manifest {agent_name!r} is NOT byte-identical to its core AgentConfig "
        f"after _merge_core_shim_only; drifted fields (merged vs core): {diffs}. "
        "The agents repo owns the data fields; core keeps the _SHIM_ONLY bits. "
        "Reconcile the manifest with registry.AGENTS before the source-of-truth flip."
    )


@pytest.mark.skipif(not os.environ.get(MANIFEST_DIR_ENV), reason=_SKIP_REASON)
def test_omnigent_pi_is_sole_core_unmanifested_agent() -> None:
    """omnigent-pi is the ONLY core agent without a manifest.

    Computed dynamically (core names minus manifest names) rather than hardcoded,
    so adding a new core agent without shipping its manifest fails this gate
    instead of silently widening the exception set.
    """
    core = _pure_core_agents()
    loaded = _live_manifests()
    unmanifested = set(core) - set(loaded)
    assert unmanifested == {_CORE_ONLY_EXCEPTION}, (
        "unmanifested core agents mismatch: expected only "
        f"{{{_CORE_ONLY_EXCEPTION!r}}}, got {sorted(unmanifested)}. "
        f"{_CORE_ONLY_EXCEPTION!r} is the documented exception — a session-factory "
        "agent registered via the #825 omnigent seam, not a manifest.toml. Every "
        "other core agent must ship a manifest in benchflow-ai/agents; a new "
        "unmanifested core agent must add one (or extend this exception on purpose)."
    )


def test_parity_assertion_catches_synthetic_drift(tmp_path: Path) -> None:
    """Load-bearing: prove the byte-identical check FAILS on a drifted manifest.

    A parity test that silently passes on everything is indistinguishable from
    one that does nothing. We hand-build a core AgentConfig and a manifest that
    reproduces it EXCEPT for a perturbed ``launch_cmd``, then assert the exact
    comparison the live test performs (``merged == core``) rejects it. Hermetic
    (no agents-repo clone, no child interpreter) so it runs on every push.
    """
    core_cfg = AgentConfig(
        name="drift-probe",
        install_cmd="echo install",
        launch_cmd="REAL_LAUNCH_CMD",
        acp_model_config_id="core-owned-shim",  # a _SHIM_ONLY field
    )
    agent_dir = tmp_path / "drift-probe"
    agent_dir.mkdir()
    (agent_dir / "manifest.toml").write_text(
        'contract_version = "1.0"\n'
        'name = "drift-probe"\n'
        'install_cmd = "echo install"\n'
        'launch_cmd = "DRIFTED_LAUNCH_CMD"\n'  # << intentional delta vs core
    )

    drifted = load_agents_from_dir(tmp_path)["drift-probe"].config
    merged = _merge_core_shim_only(drifted, core_cfg)

    # _SHIM_ONLY is restored from core even under drift (manifest can't carry it).
    assert merged.acp_model_config_id == "core-owned-shim"
    # The data-field drift survives the merge ...
    assert merged.launch_cmd == "DRIFTED_LAUNCH_CMD"
    assert merged.launch_cmd != core_cfg.launch_cmd
    # ... so the byte-identical assertion the live parity test makes MUST reject it.
    assert merged != core_cfg
