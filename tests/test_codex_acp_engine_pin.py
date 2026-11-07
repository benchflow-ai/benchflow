"""Guards #1204: the ACP engine is pinned independently of its adapter."""

import base64
import re

from benchflow.agents.registry import AGENTS, pinned_npm_package


def test_acp_install_pins_and_checks_the_engine_it_launches():
    command = AGENTS["codex-acp"].install_cmd
    package, version = pinned_npm_package("codex")
    assert f"{package}@{version}" in command
    # Install into the adapter's own dependency tree, which both its CLI probe
    # and app-server wrapper resolve; a separate global install is insufficient.
    assert (
        "--prefix /opt/runtime/js-agents/lib/node_modules/@agentclientprotocol/codex-acp"
        in command
    )
    assert "/opt/runtime/bin/codex-acp cli -V" in command
    assert f"codex-cli {version}" in command
    match = re.search(
        r"printf '%s' '?([A-Za-z0-9+/=]+)'? \| base64 -d > \S*/codex-app-server",
        command,
    )
    assert match
    wrapper = base64.b64decode(match[1]).decode()
    assert "/lib/node_modules/@agentclientprotocol/codex-acp" in wrapper
