"""Meta Muse Code (``muse``) install wiring for the registry.

Muse Code ships as a single statically linked binary and has no ACP mode, so
BenchFlow drives it through ``muse_acp_shim.py`` (an ACP server over the
headless ``muse exec --json`` mode). Everything that happens *inside* the
sandbox (credentials, model and effort, the web policy flags, the session log
that becomes the trajectory) lives in that shim; this module holds only what
the registry entry needs on the host:

- the pinned Muse Code release (version, per-architecture download URL,
  SHA-256 and size) and the POSIX ``sh`` snippet that installs it under
  ``/opt/benchflow``. Meta's own installer follows the ``muse-stable`` channel
  and has no version pin, so the release manifest is read once here instead;
- the credential paths the agent uses.
"""

from __future__ import annotations

import shlex

MUSE_VERSION = "1.4.7-R6101.1"
MUSE_DIR = "/opt/benchflow/muse"
MUSE_BIN = f"{MUSE_DIR}/muse"
_MUSE_DOWNLOAD = (
    "https://lookaside.facebook.com/lookaside/muse/download/"
    f"?channel=muse&version={MUSE_VERSION}&file="
)
# uname -m -> (URL, SHA-256, size in bytes). Taken from the release manifest
# that https://api.meta.ai/muse-code/channels/muse-stable named on 2026-10-13
# (``manifest_url``, ``artifacts.x86_linux`` / ``artifacts.aarch64_linux``).
# Each artifact is the CLI binary itself. Bump all of them together.
MUSE_RELEASES: dict[str, tuple[str, str, int]] = {
    "x86_64": (
        f"{_MUSE_DOWNLOAD}muse-x86-linux",
        "36e3dac91f56c15572b95be55d70ee9fd9685cd9ab49f747bab4746c8d7c2518",
        363518168,
    ),
    "aarch64": (
        f"{_MUSE_DOWNLOAD}muse-aarch64-linux",
        "48db3c7468cd4f401eff9b415fb7e0cdfe0351fb1730649ceae2a0f766e23b85",
        325916776,
    ),
}

# The subscription login that ``TBH_CREDENTIAL_BACKEND=file muse login``
# writes. MUSE_AUTH_JSON carries its contents into the sandbox; META_API_KEY is
# the pay-as-you-go alternative, which Muse itself prefers when both are set.
MUSE_AUTH_RELPATH = ".config/muse/auth.json"
MUSE_AUTH_JSON_ENV = "MUSE_AUTH_JSON"
META_API_KEY_ENV = "META_API_KEY"
MUSE_SKILLS_PATH = "$HOME/.config/muse/skills"


def muse_install_cmd(apt_install_curl: str) -> str:
    """POSIX ``sh`` snippet installing the pinned Muse Code binary under /opt/benchflow.

    Idempotent: a binary already reporting ``MUSE_VERSION`` is kept. The
    download is checked against the manifest SHA-256 before it is moved into
    place, and ``muse --version`` at the end proves the binary runs on the task
    image. ``MUSE_NO_AUTO_UPDATE`` keeps the binary from replacing itself.
    ``apt_install_curl`` is the registry's retrying apt snippet for ``curl`` +
    ``ca-certificates``.
    """
    q_bin = shlex.quote(MUSE_BIN)
    q_dir = shlex.quote(MUSE_DIR)
    q_target = shlex.quote(f"{MUSE_DIR}/muse-bin-{MUSE_VERSION}")
    x86_url, x86_sha, _ = MUSE_RELEASES["x86_64"]
    arm_url, arm_sha, _ = MUSE_RELEASES["aarch64"]
    return (
        "export DEBIAN_FRONTEND=noninteractive; "
        f"( [ -x {q_bin} ] && MUSE_NO_AUTO_UPDATE=1 {q_bin} --version 2>/dev/null | "
        f"grep -qF {shlex.quote(MUSE_VERSION)} ) || ( "
        "set -e; "
        "( command -v curl >/dev/null 2>&1 && [ -e /etc/ssl/certs/ca-certificates.crt ] || "
        f"{apt_install_curl} ); "
        'arch="$(uname -m)"; '
        'case "$arch" in '
        f"x86_64|amd64) url={shlex.quote(x86_url)}; sha={x86_sha} ;; "
        f"aarch64|arm64) url={shlex.quote(arm_url)}; sha={arm_sha} ;; "
        '*) echo "Unsupported architecture for Muse Code: $arch" >&2; exit 1 ;; '
        "esac; "
        'tmp="$(mktemp -d)"; '
        'curl -fsSL --retry 3 -o "$tmp/muse" "$url"; '
        "if command -v sha256sum >/dev/null 2>&1; then "
        'echo "$sha  $tmp/muse" | sha256sum -c - >/dev/null; '
        "elif command -v shasum >/dev/null 2>&1; then "
        'echo "$sha  $tmp/muse" | shasum -a 256 -c - >/dev/null; '
        "else echo 'sha256sum or shasum is required to verify the Muse Code download' >&2; exit 1; fi; "
        f"mkdir -p {q_dir}; "
        f'mv "$tmp/muse" {q_target}; '
        f"chmod 755 {q_target}; "
        f"ln -sf {q_target} {q_bin}; "
        'rm -rf "$tmp" ) && '
        f"chmod -R a+rX {q_dir} && "
        f"MUSE_NO_AUTO_UPDATE=1 {q_bin} --version"
    )
