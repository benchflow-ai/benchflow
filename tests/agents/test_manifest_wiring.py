"""Lazy local catalog activation through one ingestion entrypoint."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import benchflow

_SRC = Path(benchflow.__file__).resolve().parents[1]
_ROOT = _SRC.parent
_MANIFEST = """contract_version = "1.0"
name = "probe-agent"
install_cmd = "echo install"
launch_cmd = "echo launch"
aliases = ["probe-alias"]
"""


def _fixture(tmp_path: Path) -> Path:
    target = tmp_path / "probe"
    target.mkdir()
    (target / "manifest.toml").write_text(_MANIFEST)
    return tmp_path


def _probe(root: Path, *, activate: bool) -> dict:
    action = (
        "from benchflow.agents.registry import resolve_agent;"
        "resolve_agent('probe-agent');"
        if activate
        else ""
    )
    code = (
        "import json;"
        "from benchflow.agents.registry import (AGENTS, AGENT_INSTALLERS, "
        "AGENT_LAUNCH, AGENT_ALIASES);"
        + action
        + "print(json.dumps({'has':'probe-agent' in AGENTS,"
        "'installer':AGENT_INSTALLERS.get('probe-agent'),"
        "'launch':AGENT_LAUNCH.get('probe-agent'),"
        "'alias':AGENT_ALIASES.get('probe-alias')}))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_SRC), str(_ROOT)])
    env["BENCHFLOW_AGENTS_DIR"] = str(root)
    env["BENCHFLOW_AGENTS_SOURCE"] = "off"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_directory_env_does_not_mutate_registry_during_import(tmp_path: Path):
    """Guards PR #1090 against a second import-time ingestion plane."""
    assert _probe(_fixture(tmp_path), activate=False) == {
        "has": False,
        "installer": None,
        "launch": None,
        "alias": None,
    }


def test_directory_env_activates_all_registry_maps_lazily(tmp_path: Path):
    """Guards PR #1090 lazy local catalog registration projections."""
    assert _probe(_fixture(tmp_path), activate=True) == {
        "has": True,
        "installer": "echo install",
        "launch": "echo launch",
        "alias": "probe-agent",
    }
