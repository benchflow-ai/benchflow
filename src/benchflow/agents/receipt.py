"""Installed harness versions captured before agent code runs."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shlex
from typing import Any

from benchflow.agents import registry

logger = logging.getLogger(__name__)

ROLES = ("adapter", "cli", "engine")


def install_identity(agent: str, harness: str = "acp") -> dict[str, str]:
    """Identity of the requested installer, including transitive pin changes."""
    name = agent.split()[0] if agent else ""
    recipe = registry.AGENT_INSTALLERS.get(name, "")
    if harness == "native":
        from benchflow.native_harness.harnesses import NATIVE_HARNESSES

        native = NATIVE_HARNESSES.get(name)
        if native:
            recipe += native.package
    return {
        "agent": name,
        "harness": harness,
        "install_sha256": hashlib.sha256(recipe.encode()).hexdigest(),
    }


def probe_command(agent: str, harness: str = "acp") -> str | None:
    """Read installed metadata with the managed Node, never the task's Node."""
    packages = {
        "claude-agent-acp": [
            "@agentclientprotocol/claude-agent-acp",
            "@anthropic-ai/claude-code",
            "@anthropic-ai/claude-code",
        ],
        "codex-acp": [
            "@agentclientprotocol/codex-acp",
            "@openai/codex",
            "@openai/codex",
        ],
        "gemini": [
            "@google/gemini-cli",
            "@google/gemini-cli",
            "@google/gemini-cli-core",
        ],
        "opencode": ["opencode-ai", "opencode-ai", "opencode-ai"],
        "mimo": ["@mimo-ai/cli", "@mimo-ai/cli", "@mimo-ai/cli"],
        "pi-acp": [
            "pi-acp",
            "@earendil-works/pi-coding-agent",
            "@earendil-works/pi-coding-agent",
        ],
    }.get(agent.split()[0])
    if packages is None:
        return None
    root = registry._BENCHFLOW_JS_AGENT_PREFIX + "/lib/node_modules"
    # Resolve companions from the adapter first: its install puts the engine
    # it runs in its own node_modules (codex-acp's pinned Codex CLI among
    # them). Claude's CLI and a native harness's CLI are separate installs.
    script = r"""
const fs = require('fs'), path = require('path');
const [root, packages, agent, harness] = JSON.parse(process.argv[1]);
const roots = [root];
if (agent === 'mimo') roots.unshift(path.dirname(path.dirname(root)) + '/mimo-acp/node_modules');
const adapter = path.join(roots[0], packages[0]);
function version(name, role) {
  const dirs = harness === 'native' || role === 'adapter' || agent === 'claude-agent-acp'
    ? roots : [path.join(adapter, 'node_modules'), ...roots];
  for (const dir of dirs) {
    try {
      const p = JSON.parse(fs.readFileSync(path.join(dir, name, 'package.json'), 'utf8'));
      if (p.name === name && typeof p.version === 'string')
        return {package: name, version: p.version, source: 'installed-package'};
    } catch (_) {}
  }
  return {package: name, version: null, source: 'unavailable'};
}
console.log(JSON.stringify(Object.fromEntries(['adapter', 'cli', 'engine'].map((r,i) => [r, version(packages[i],r)]))));
"""
    return (
        f"{registry._BENCHFLOW_NODE_PREFIX}/bin/node -e {shlex.quote(script)} "
        + shlex.quote(json.dumps([root, packages, agent.split()[0], harness]))
    )


async def probe_receipt(env: Any, agent: str, harness: str) -> dict[str, Any]:
    """The installed versions, read in the sandbox before the agent starts
    (cached and skip-install sandboxes too).

    The probe is evidence, never a reason to fail the trial: a probe that
    cannot run, or answers anything but its JSON, leaves the versions
    ``unavailable``. Nothing is written here; the rollout keeps the receipt
    and records it in config.json and result.json.
    """
    from benchflow.sandbox._exec_output import exec_raw

    receipt: dict[str, Any] = {
        "schema": 1,
        "identity": install_identity(agent, harness),
        **{role: {"version": None, "source": "unavailable"} for role in ROLES},
    }
    if registry.is_scripted_agent(agent):
        for role in ROLES:
            receipt[role] = {"version": None, "source": "not-applicable"}
    command = probe_command(agent, harness)
    if command:
        try:
            result = await exec_raw(env.exec, command, timeout_sec=30)
            observed = json.loads(result.stdout) if result.return_code == 0 else {}
            for role in ROLES:
                value = observed.get(role)
                if isinstance(value, dict) and value.get("source") in (
                    "installed-package",
                    "unavailable",
                ):
                    receipt[role] = value
        except Exception as exc:
            logger.warning("Could not read %s's installed versions: %s", agent, exc)
    if harness == "native":
        receipt["adapter"] = {"version": None, "source": "not-applicable"}
    return receipt


def _pinned(package: str, recipe: str) -> list[str]:
    """Every version *recipe* installs *package* at (``package@version``)."""
    return re.findall(rf"(?<![\w@/.-]){re.escape(package)}@([\w.+-]+)", recipe)


def receipt_matches_install(receipt: dict[str, Any], agent: str, harness: str) -> bool:
    """Whether a trial's recorded versions can stand for what the installer
    would install now.

    The installer must be the same (its identity), and every version the
    probe read for a package the installer pins must be that pin. A package
    it does not pin (an engine the adapter brings with it, such as gemini's
    ``@google/gemini-cli-core``) is not compared, and a version the probe
    could not read is unknown, not a difference.
    """
    if receipt.get("identity") != install_identity(agent, harness):
        return False
    name = agent.split()[0] if agent else ""
    recipe = registry.AGENT_INSTALLERS.get(name, "")
    if harness == "native":
        from benchflow.native_harness.harnesses import NATIVE_HARNESSES

        native = NATIVE_HARNESSES.get(name)
        if native:
            recipe += " " + native.package
    for role in ROLES:
        value = receipt.get(role)
        if not isinstance(value, dict) or value.get("source") != "installed-package":
            continue
        package, version = value.get("package"), value.get("version")
        if not isinstance(package, str) or not isinstance(version, str):
            continue
        pins = _pinned(package, recipe)
        if pins and version not in pins:
            return False
    return True
