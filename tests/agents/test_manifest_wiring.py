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
