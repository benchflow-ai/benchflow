"""OpenCode proxy-boundary configuration hardening."""

from __future__ import annotations

import shlex

OPENCODE_CONFIG_RELATIVE_PATH = ".config/opencode/opencode.json"
_OPENCODE_NODE = "/opt/benchflow/node/bin/node"


def opencode_provider_reset_source() -> str:
    """Return stdlib Node.js that removes every pre-existing provider."""

    return "\n".join(
        [
            'const fs = require("fs");',
            'const os = require("os");',
            'const path = require("path");',
            'const home = (process.env.BENCHFLOW_AGENT_HOME || "").trim() || os.homedir();',
            f"const p = path.join(home, {OPENCODE_CONFIG_RELATIVE_PATH!r});",
            "fs.mkdirSync(path.dirname(p), { recursive: true });",
            'const text = fs.existsSync(p) ? fs.readFileSync(p, "utf8").trim() : "";',
            "const d = text ? JSON.parse(text) : {};",
            # The manifest-owned wrapper adds the one BenchFlow gateway provider
            # immediately after this pre-launch boundary. Removing the full map
            # here prevents an image-baked literal key/endpoint from surviving.
            "d.provider = {};",
            'const temporary = p + ".benchflow-" + process.pid + ".tmp";',
            'fs.writeFileSync(temporary, JSON.stringify(d, null, 2) + "\\n", { mode: 0o600 });',
            "fs.renameSync(temporary, p);",
        ]
    )


def opencode_provider_reset_command() -> str:
    """Return the shell-safe reset command run immediately before OpenCode."""

    return f"{_OPENCODE_NODE} -e {shlex.quote(opencode_provider_reset_source())}"
