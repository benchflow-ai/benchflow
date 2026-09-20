"""Google Antigravity CLI (``agy``) install and policy wiring for the registry.

The Antigravity CLI replaced the hosted Gemini CLI in mid-2026. It ships as a
single native binary (no npm package) and has no ACP mode, so BenchFlow drives
it through ``antigravity_acp_shim.py`` (an ACP server over agy's headless
``stream-json`` protocol). Model and effort selection, the settings file and
everything else that happens *inside* the sandbox live in that shim; this
module holds only what the registry entry needs on the host:

- the pinned agy release (version, per-architecture download URL + SHA-512)
  and the POSIX ``sh`` snippet that installs it under ``/opt/benchflow``;
- the discovery paths agy reads (skills, hooks, MCP config);
- the ``hooks.json`` mutator behind BenchFlow's web-tool policies. agy has no
  tool-exclusion setting; its documented switch is a ``PreToolUse`` lifecycle
  hook whose command answers ``{"decision": "deny"}``.
"""

from __future__ import annotations

import json
import shlex

AGY_VERSION = "1.2.7"
AGY_BIN = "/opt/benchflow/antigravity/agy"
_AGY_RELEASE_BASE = (
    "https://storage.googleapis.com/antigravity-public/antigravity-cli/"
    "1.2.7-6731160148115456"
)
# uname -m -> (tarball URL, SHA-512). Taken from the installer manifests at
# https://antigravity-cli-auto-updater-974169037036.us-central1.run.app/manifests/
# (linux_amd64.json / linux_arm64.json) on 2026-09-19. Each tarball holds one
# file, ``antigravity``, which is the CLI binary. Bump all three together.
AGY_RELEASES: dict[str, tuple[str, str]] = {
    "x86_64": (
        f"{_AGY_RELEASE_BASE}/linux-x64/cli_linux_x64.tar.gz",
        "fec769d611c4afdf0ae72d38bdb2652c8e2c8e71e4f6de97a27b80dda3c50429"
        "160d9e03776a36a59b8857c20e76783c4a49cb0feb8b2f5c3bf925b0cc03bb77",
    ),
    "aarch64": (
        f"{_AGY_RELEASE_BASE}/linux-arm/cli_linux_arm64.tar.gz",
        "d39f939ffc80776bfd2dc10db7b9a1a1b58650f08115fa21065c11d10882c712"
        "10f368c2e6060f26966d1a33d6b2dd2bc19bfd641062305af6546c51d494511a",
    ),
}

ANTIGRAVITY_HOME_DIR = ".gemini"
ANTIGRAVITY_HOOKS_RELPATH = ".gemini/antigravity-cli/hooks.json"
ANTIGRAVITY_MCP_CONFIG_RELPATH = ".gemini/config/mcp_config.json"
ANTIGRAVITY_GLOBAL_SKILLS_PATH = "$HOME/.gemini/config/skills"
ANTIGRAVITY_WORKSPACE_SKILLS_PATH = "$WORKSPACE/.agents/skills"

# agy tools that reach the web. ``search_web`` is Google's hosted search and
# ``read_url_content`` fetches server-side, so neither passes through
# BenchFlow's egress proxy; the browser tools drive a local Chrome (absent in
# task images) but are switched off with the rest under the no-web policy.
ANTIGRAVITY_HOSTED_WEB_TOOL_MATCHERS = ("search_web", "read_url_content")
ANTIGRAVITY_WEB_TOOL_MATCHERS = (
    *ANTIGRAVITY_HOSTED_WEB_TOOL_MATCHERS,
    "google_search",
    "open_browser_url",
    ".*browser.*",
)


def agy_install_cmd(apt_install_curl: str) -> str:
    """POSIX ``sh`` snippet installing the pinned agy binary under /opt/benchflow.

    Idempotent: a binary already reporting ``AGY_VERSION`` is kept. The
    download is checksum-verified against the manifest SHA-512 before the
    binary is moved into place, and ``agy --version`` at the end proves the
    binary actually runs on the task image (glibc / architecture mismatch
    surfaces at install time, not as a silent ACP handshake failure).
    ``apt_install_curl`` is the registry's retrying apt snippet for
    ``curl`` + ``ca-certificates``. ``DEBIAN_FRONTEND`` is exported for the
    whole install line: on bare images the python3 install that follows pulls
    ``tzdata``, whose configuration prompt otherwise blocks apt under a TTY.
    """
    q_bin = shlex.quote(AGY_BIN)
    q_dir = shlex.quote(AGY_BIN.rsplit("/", 1)[0])
    x86_url, x86_sha = AGY_RELEASES["x86_64"]
    arm_url, arm_sha = AGY_RELEASES["aarch64"]
    return (
        "export DEBIAN_FRONTEND=noninteractive; "
        f'( [ -x {q_bin} ] && [ "$({q_bin} --version 2>/dev/null | head -n1 | '
        f"tr -d '[:space:]')\" = {shlex.quote(AGY_VERSION)} ] ) || ( "
        "set -e; "
        "( command -v curl >/dev/null 2>&1 && [ -e /etc/ssl/certs/ca-certificates.crt ] || "
        f"{apt_install_curl} ); "
        'arch="$(uname -m)"; '
        'case "$arch" in '
        f"x86_64|amd64) url={shlex.quote(x86_url)}; sha={x86_sha} ;; "
        f"aarch64|arm64) url={shlex.quote(arm_url)}; sha={arm_sha} ;; "
        '*) echo "Unsupported architecture for the Antigravity CLI: $arch" >&2; exit 1 ;; '
        "esac; "
        'tmp="$(mktemp -d)"; '
        'curl -fsSLo "$tmp/agy.tar.gz" "$url"; '
        "if command -v sha512sum >/dev/null 2>&1; then "
        'echo "$sha  $tmp/agy.tar.gz" | sha512sum -c - >/dev/null; '
        "elif command -v shasum >/dev/null 2>&1; then "
        'echo "$sha  $tmp/agy.tar.gz" | shasum -a 512 -c - >/dev/null; '
        "else echo 'sha512sum or shasum is required to verify the Antigravity CLI download' >&2; exit 1; fi; "
        'tar -xzf "$tmp/agy.tar.gz" -C "$tmp"; '
        f"mkdir -p {q_dir}; "
        f'mv "$tmp/antigravity" {q_bin}; '
        f"chmod 755 {q_bin}; "
        'rm -rf "$tmp" ) && '
        f"chmod -R a+rX {q_dir} && "
        f"{q_bin} --version"
    )


def hooks_deny_mutator(hook_name: str, matchers: tuple[str, ...], reason: str) -> str:
    """Python mutator (for ``_json_settings_merge``) adding a deny hook set.

    Produces ``hooks.json`` content of the documented shape::

        {"<hook_name>": {"PreToolUse": [{"matcher": "<tool regex>",
                                          "hooks": [{"type": "command",
                                                     "command": "printf ...",
                                                     "timeout": 5}]}, ...]}}

    Each matched tool call is answered with ``{"decision": "deny"}`` before it
    runs, independent of the permission mode agy is launched in. Named hook
    entries merge with any hooks the task or image already configured.
    """
    verdict = json.dumps({"decision": "deny", "reason": reason})
    command = f"printf '%s' {shlex.quote(verdict)}"
    spec = {
        "PreToolUse": [
            {
                "matcher": matcher,
                "hooks": [{"type": "command", "command": command, "timeout": 5}],
            }
            for matcher in matchers
        ]
    }
    return f"d[{hook_name!r}]={spec!r}"
