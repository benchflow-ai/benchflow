"""Per-sandbox Codex Apps requirements; no account or hosted catalog operations."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from benchflow.agents.codex_config import disable_codex_apps
from benchflow.agents.registry import CODEX_ACP_BUILTIN_LAUNCH, pinned_npm_package
from benchflow.sandbox.lockdown import build_priv_drop_cmd

_ADAPTER = "/opt/benchflow/bin/codex-acp"
_NODE = "/opt/benchflow/node/bin/node"
_NODE_MODULES = "/opt/benchflow/js-agents/lib/node_modules"
# The adapter must be the registry pin. Native Codex is whatever that adapter
# resolves within its declared `@openai/codex` range (^0.156.1 for 1.13.1); the
# `features list` probe below is the behavioral check for every admitted build.
_MANIFEST = """const m=JSON.parse(require('fs').readFileSync(%s,'utf8'));
console.log(JSON.stringify({name:m.name,version:m.version,codex:(m.dependencies||{})['@openai/codex']}));"""
_SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")

# Runs as root with the managed Node runtime; no Python dependency in task images.
# Preserve existing requirements byte-for-byte. Native config loading below is
# the parser and conflict check; we do not implement a partial TOML merger.
_INSTALL = r"""
const fs = require('node:fs');
const crypto = require('node:crypto');
function trusted(p, directory) {
  const s = fs.lstatSync(p);
  if (s.isSymbolicLink() || s.uid !== 0 || (s.mode & 0o022) ||
      (directory ? !s.isDirectory() : !s.isFile())) throw Error('unsafe policy path');
  return s;
}
try {
  trusted('/', true); trusted('/etc', true);
  try { fs.mkdirSync('/etc/codex', {mode:0o755}); }
  catch (e) { if(e.code !== 'EEXIST') throw e; }
  trusted('/etc/codex', true);
  const p = '/etc/codex/requirements.toml';
  let created = false;
  try {
    const fd = fs.openSync(p, fs.constants.O_WRONLY | fs.constants.O_CREAT |
      fs.constants.O_EXCL | fs.constants.O_NOFOLLOW, 0o644);
    try { fs.writeFileSync(fd, '[features]\napps = false\n'); fs.fsyncSync(fd); }
    finally { fs.closeSync(fd); }
    created = true;
  } catch (e) { if(e.code !== 'EEXIST') throw e; }
  const s = trusted(p, false);
  if(s.size > 1048576) throw Error('oversized policy');
  const fd = fs.openSync(p, fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW);
  let data;
  try { data = fs.readFileSync(fd); } finally { fs.closeSync(fd); }
  console.log(JSON.stringify({created, requirements_sha256:crypto.createHash('sha256').update(data).digest('hex')}));
} catch (_) { console.error('Codex Apps managed requirements installation failed'); process.exit(1); }
"""


class CodexAppsPolicyRefused(RuntimeError):
    """The disabled Apps policy could not be verified; names the opt-out."""

    def __init__(self, reason: str):
        super().__init__(
            f"{reason}. To run with account Codex Apps left as configured, pass "
            "--codex-apps-policy inherit (config: codex_apps_policy: inherit)."
        )


def _satisfies(version: str, declared: str) -> bool:
    """npm exact or caret range over release versions; other forms fail closed."""
    floor = _SEMVER.fullmatch(declared.removeprefix("^"))
    found = _SEMVER.fullmatch(version)
    if floor is None or found is None:
        return False
    low = tuple(int(part) for part in floor.groups())
    got = tuple(int(part) for part in found.groups())
    if not declared.startswith("^"):
        return got == low
    # A caret range keeps the first nonzero component fixed.
    fixed = next((i for i, part in enumerate(low) if part), 2)
    return low <= got and got[: fixed + 1] == low[: fixed + 1]


def effective_apps_policy(
    override: str | None, *, purpose: str, skip_verify: bool
) -> str:
    if override not in {None, "disabled", "inherit"}:
        raise ValueError("codex_apps_policy must be disabled, inherit, or None")
    return override or (
        "disabled" if purpose == "task" and not skip_verify else "inherit"
    )


async def enforce_codex_apps_policy(
    env: Any,
    *,
    agent: str,
    agent_launch: str,
    agent_env: dict[str, str],
    sandbox_user: str | None,
    policy: str,
    requested: str | None,
    rollout_dir: Path,
    timeout_sec: int = 60,
) -> dict[str, str]:
    """Fail before ACP initialization unless the supported native policy holds.

    `inherit` leaves any existing stricter managed policy intact. The receipt
    proves configuration enforcement only, never authenticated tool absence.
    """
    if agent != "codex-acp":
        return agent_env
    # A failed reconnect must not leave a previous successful receipt current.
    (rollout_dir / "codex_apps_policy.json").unlink(missing_ok=True)
    receipt: dict[str, Any] = {
        "requested": requested or "automatic",
        "applied": policy,
        "mechanism": "inherited" if policy == "inherit" else "managed_requirements",
        "authenticated_tool_absence_verified": False,
    }
    if policy == "inherit":
        updated = agent_env
    else:
        if policy != "disabled":
            raise ValueError("Unsupported Codex Apps policy")
        if not sandbox_user or sandbox_user in {"root", "0"}:
            raise CodexAppsPolicyRefused(
                "Disabled Codex Apps requires a non-root sandbox user"
            )
        if agent_launch != CODEX_ACP_BUILTIN_LAUNCH:
            raise CodexAppsPolicyRefused(
                "Codex Apps policy requires the managed Codex ACP launcher"
            )
        if any(agent_env.get(k) for k in ("CODEX_PATH", "NODE_OPTIONS", "NODE_PATH")):
            raise CodexAppsPolicyRefused(
                "Codex Apps policy does not support executable injection overrides"
            )
        updated = disable_codex_apps(agent_env)
        # Force supported bundled resolution even if the image supplied CODEX_PATH.
        updated.update(CODEX_PATH="", NODE_OPTIONS="", NODE_PATH="")

        async def probe_native(command: str):
            # Reuse the actual launch privilege/home wrapper and environment.
            return await env.exec(
                build_priv_drop_cmd(command, sandbox_user),
                user="root",
                env=updated,
                timeout_sec=timeout_sec,
            )

        async def root_node(script: str):
            return await env.exec(
                f"/usr/bin/env -u NODE_OPTIONS -u NODE_PATH {_NODE} -e {shlex.quote(script)}",
                user="root",
                timeout_sec=timeout_sec,
            )

        identity = await probe_native("/usr/bin/id -u")
        uid = (identity.stdout or "").strip()
        if identity.return_code != 0 or not uid.isdigit() or int(uid) == 0:
            raise CodexAppsPolicyRefused(
                "Disabled Codex Apps requires a verified non-root UID"
            )
        package, pinned = pinned_npm_package("codex-acp")
        adapter = await probe_native(f"{_ADAPTER} --version")
        if (
            adapter.return_code != 0
            or (adapter.stdout or "").strip() != f"{package} {pinned}"
        ):
            raise CodexAppsPolicyRefused(
                f"Codex Apps policy requires verified Codex ACP {pinned}"
            )
        manifest = await root_node(
            _MANIFEST % json.dumps(f"{_NODE_MODULES}/{package}/package.json")
        )
        try:
            declared = json.loads(manifest.stdout or "")
        except ValueError:
            declared = None
        if (
            manifest.return_code != 0
            or not isinstance(declared, dict)
            or (declared.get("name"), declared.get("version")) != (package, pinned)
            or not isinstance(declared.get("codex"), str)
        ):
            raise CodexAppsPolicyRefused(
                f"Codex Apps policy requires the native Codex range declared by Codex ACP {pinned}"
            )
        version = await probe_native(f"{_ADAPTER} cli -V")
        reported = re.fullmatch(r"codex-cli (\S+)", (version.stdout or "").strip())
        native = reported.group(1) if reported else ""
        if version.return_code != 0 or not _satisfies(native, declared["codex"]):
            raise CodexAppsPolicyRefused(
                f"Codex Apps policy requires verified native Codex {declared['codex']}"
                f" as declared by Codex ACP {pinned}"
            )
        installed = await root_node(_INSTALL)
        if installed.return_code != 0:
            raise CodexAppsPolicyRefused(
                "Codex Apps managed requirements installation failed"
            )
        try:
            installation = json.loads(installed.stdout or "")
            digest = installation["requirements_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError("bad digest")
        except (ValueError, KeyError, TypeError) as exc:
            raise CodexAppsPolicyRefused(
                "Invalid Codex Apps policy installation receipt"
            ) from exc
        # An explicit CLI enable must remain false under requirements. This also
        # rejects an existing valid policy which lacks the required restriction.
        probe = await probe_native(
            f"{_ADAPTER} cli -c features.apps=true features list"
        )
        apps_rows = [
            line.split()
            for line in (probe.stdout or "").splitlines()
            if line.split()[:1] == ["apps"]
        ]
        if (
            probe.return_code != 0
            or len(apps_rows) != 1
            or apps_rows[0] != ["apps", "stable", "false"]
        ):
            raise CodexAppsPolicyRefused(
                "Codex Apps managed requirements conflict or enforcement probe failed"
            )
        receipt.update(
            adapter_version=pinned,
            native_version=native,
            requirements_sha256=digest,
            effective_apps=False,
            override_probe="cli_true_remains_false",
        )
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "codex_apps_policy.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return updated
