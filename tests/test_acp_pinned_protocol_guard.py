"""Gated guard: pinned ACP agents resolve to the protocol our registry targets.

Skipped by default. Run with ``RUN_ACP_DEP_GUARD=1`` (needs ``npm`` + network):

    RUN_ACP_DEP_GUARD=1 uv run --extra dev python -m pytest \
        tests/test_acp_pinned_protocol_guard.py -q

This codifies the manual check behind the ``claude-agent-acp@0.40.0`` pin in
``benchflow.agents.registry``: its ACP SDK exposes ``session/set_config_option``
(the model/effort path the registry wires up) and no longer exposes
``session/set_model``. Re-run this when bumping an ``@agentclientprotocol`` pin —
if it fails, the registry's config-option wiring is stale for the new version.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    __import__("os").environ.get("RUN_ACP_DEP_GUARD") != "1",
    reason="ACP dependency guard is gated; set RUN_ACP_DEP_GUARD=1 (needs npm + network)",
)


def _npm() -> str:
    npm = shutil.which("npm")
    if not npm:
        pytest.skip("npm not available")
    return npm


def _pack_and_extract(npm: str, spec: str, workdir: Path) -> Path:
    """``npm pack`` ``spec`` into ``workdir`` and return the extracted package dir."""
    subprocess.run(
        [npm, "pack", spec],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    tgz = next(workdir.glob("*.tgz"))
    subprocess.run(["tar", "xzf", tgz.name], cwd=workdir, check=True, timeout=120)
    return workdir / "package"


def _source_blob(pkg_dir: Path) -> str:
    return "\n".join(
        p.read_text(errors="ignore")
        for p in pkg_dir.rglob("*")
        if p.suffix in {".js", ".cjs", ".mjs", ".ts"}
    )


def test_pinned_claude_acp_uses_config_option_protocol(tmp_path):
    npm = _npm()
    agent_pkg = _pack_and_extract(
        npm, "@agentclientprotocol/claude-agent-acp@0.40.0", tmp_path
    )
    sdk_spec = json.loads((agent_pkg / "package.json").read_text())["dependencies"][
        "@agentclientprotocol/sdk"
    ]

    sdk_dir = tmp_path / "sdk"
    sdk_dir.mkdir()
    sdk_pkg = _pack_and_extract(npm, f"@agentclientprotocol/sdk@{sdk_spec}", sdk_dir)
    blob = _source_blob(sdk_pkg)

    assert "session/set_config_option" in blob, (
        "pinned claude-agent-acp ACP SDK no longer exposes session/set_config_option "
        "— the registry's acp_model_config_id/acp_effort_config_id wiring is stale"
    )
    assert "session/set_model" not in blob, (
        "pinned claude-agent-acp ACP SDK still exposes session/set_model — the pin "
        "may be honoring a pre-migration version the registry wiring does not target"
    )
