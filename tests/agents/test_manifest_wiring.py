"""Import-time activation of the dual-source registry (decision #7 go-live).

registry.py calls register_env_manifest_agents() at end-of-module. These tests
spawn a *fresh* interpreter so the import-time merge runs cleanly (an in-process
reload would mutate the shared registry the rest of the suite uses). They pin the
gated-off guarantee — a default import is unchanged — and the opt-in behaviour —
$BENCHFLOW_AGENTS_DIR merges a manifest agent into every name-keyed map.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import benchflow

# Resolve absolute import roots so the spawned interpreter finds benchflow
# whether it is installed (CI) or on PYTHONPATH from a worktree (dev VM).
_SRC = Path(benchflow.__file__).resolve().parents[1]
_ROOT = _SRC.parent

_MANIFEST = """contract_version = "1.0"
name = "probe-agent"
install_cmd = "echo install"
launch_cmd = "echo launch"
aliases = ["probe-alias"]
"""

_PROBE = (
    "import json;"
    "from benchflow.agents.registry import ("
    "AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH, AGENT_ALIASES);"
    "print(json.dumps({"
    "'has': 'probe-agent' in AGENTS,"
    "'installer': AGENT_INSTALLERS.get('probe-agent'),"
    "'launch': AGENT_LAUNCH.get('probe-agent'),"
    "'alias': AGENT_ALIASES.get('probe-alias'),"
    "}))"
)


def _probe(manifest_root: Path, *, set_env: bool) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), str(_ROOT)])
    if set_env:
        env["BENCHFLOW_AGENTS_DIR"] = str(manifest_root)
    else:
        env.pop("BENCHFLOW_AGENTS_DIR", None)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _fixture(tmp_path: Path) -> Path:
    d = tmp_path / "probe-agent-dir"  # dir name deliberately != agent name
    d.mkdir()
    (d / "manifest.toml").write_text(_MANIFEST)
    return tmp_path


def test_default_import_is_unchanged_when_env_unset(tmp_path: Path):
    result = _probe(_fixture(tmp_path), set_env=False)
    assert result == {
        "has": False,
        "installer": None,
        "launch": None,
        "alias": None,
    }


def test_import_merges_manifest_agent_when_env_set(tmp_path: Path):
    result = _probe(_fixture(tmp_path), set_env=True)
    assert result["has"] is True
    assert result["installer"] == "echo install"
    assert result["launch"] == "echo launch"
    assert result["alias"] == "probe-agent"


_FULL_SNAP = (
    "import json;"
    "from benchflow.agents.registry import ("
    "AGENTS, AGENT_INSTALLERS, AGENT_LAUNCH, AGENT_ALIASES);"
    "print(json.dumps({"
    "'agents': sorted(AGENTS),"
    "'aliases': dict(sorted(AGENT_ALIASES.items())),"
    "'installers': dict(sorted(AGENT_INSTALLERS.items())),"
    "'launch': dict(sorted(AGENT_LAUNCH.items())),"
    "}, sort_keys=True))"
)


def _snapshot(manifest_root: Path, *, set_env: bool) -> dict:
    """Full canonical snapshot of all four name-keyed registry maps (fresh interp)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), str(_ROOT)])
    if set_env:
        env["BENCHFLOW_AGENTS_DIR"] = str(manifest_root)
    else:
        env.pop("BENCHFLOW_AGENTS_DIR", None)
    out = subprocess.run(
        [sys.executable, "-c", _FULL_SNAP],
        env=env,
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _drop_probe(snap: dict) -> dict:
    return {
        "agents": [a for a in snap["agents"] if a != "probe-agent"],
        "aliases": {k: v for k, v in snap["aliases"].items() if k != "probe-alias"},
        "installers": {
            k: v for k, v in snap["installers"].items() if k != "probe-agent"
        },
        "launch": {k: v for k, v in snap["launch"].items() if k != "probe-agent"},
    }


def test_default_import_byte_identical_full_registry(tmp_path: Path):
    """Airtight 'byte-identical when env unset' guarantee (concern #12).

    Strengthens test_default_import_is_unchanged_when_env_unset from asserting only
    the probe agent's ABSENCE to asserting FULL-dict equality of all four name-keyed
    registry maps: the sole delta the opt-in manifest plane introduces is the probe
    agent itself; every pre-existing agent / alias / installer / launch entry is
    byte-for-byte unchanged. Catches accidental mutation of an *existing* entry,
    which the absence-only assertion cannot.
    """
    fixture = _fixture(tmp_path)
    unset = _snapshot(tmp_path, set_env=False)
    activated = _snapshot(fixture, set_env=True)

    # gated off: zero probe footprint anywhere in the four maps
    assert "probe-agent" not in unset["agents"]
    assert "probe-agent" not in unset["installers"]
    assert "probe-agent" not in unset["launch"]
    assert "probe-alias" not in unset["aliases"]

    # opt-in adds EXACTLY the probe agent — nothing else appears
    assert set(activated["agents"]) - set(unset["agents"]) == {"probe-agent"}
    assert activated["aliases"].get("probe-alias") == "probe-agent"

    # full byte-identical equality of all four maps, modulo the probe agent
    assert unset == _drop_probe(activated)
