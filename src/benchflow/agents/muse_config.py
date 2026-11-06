"""Meta Muse Code (``muse``) install wiring for the registry.

Muse Code ships as a single statically linked binary and has no ACP mode, so
BenchFlow drives it through ``muse_acp_shim.py`` (an ACP server over the
headless ``muse exec --json`` mode). Everything that happens *inside* the
sandbox (credentials, model and effort, the web policy flags, the session log
that becomes the trajectory) lives in that shim; this module holds only what
the registry entry needs on the host:

- the pinned Muse Code release (version, per-architecture download URL,
  SHA-256 and size) and the POSIX ``sh`` snippet that installs it under
  ``/opt/runtime``. Meta's own installer follows the ``muse-stable`` channel
  and has no version pin, so the release manifest is read once here instead;
- the credential paths the agent uses.
"""

from __future__ import annotations

import shlex

MUSE_VERSION = "1.4.7-R6101.1"
MUSE_DIR = "/opt/runtime/muse"
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


def validate_muse_model(model: str | None) -> None:
    """Refuse unsupported routes before any provider or login discovery.

    Muse has its own subscription transport, not a general provider client.
    Prefixes here match the shim's models.dev aliases; contributor selections
    retain the shim's explicit refusal.
    """
    bare = (model or "").strip()
    for prefix in ("meta/", "muse/"):
        if bare.startswith(prefix):
            bare = bare[len(prefix) :]
            break
    if bare.endswith("-contributor"):
        raise ValueError("muse-code refuses contributor model selections")
    if bare and (not bare.startswith("muse-") or "/" in bare):
        raise ValueError(
            "muse-code requires a Muse model (muse-*, meta/muse-* or muse/muse-*); "
            "other providers are unsupported"
        )


def validate_muse_roles(
    targets: list[tuple[str, str | None]], *, sandbox_user: str | None = "agent"
) -> None:
    """Fail closed for scenes whose shared UID cannot isolate the Muse login."""
    if not any(agent == "muse-code" for agent, _ in targets):
        return
    for agent, model in targets:
        if agent == "muse-code":
            validate_muse_model(model)
    if any(agent != "muse-code" for agent, _ in targets):
        raise ValueError(
            "mixed scenes with muse-code are unsupported: roles share a sandbox UID"
        )
    if not sandbox_user or sandbox_user == "root":
        raise ValueError(
            "muse-code requires a non-root sandbox_user for credential cleanup"
        )


def muse_install_cmd(apt_install_curl: str) -> str:
    """POSIX ``sh`` snippet installing the pinned Muse Code binary under /opt/runtime.

    Idempotent: a binary already there is kept only when its bytes match the
    manifest SHA-256 (AG-R2-2: a version string is no proof, and an installer
    before the checksum fix could leave bad bytes that print the right one).
    The download is checked against the same digest before it is moved into
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
        "export DEBIAN_FRONTEND=noninteractive; ( "
        'arch="$(uname -m)" || exit 1; '
        'case "$arch" in '
        f"x86_64|amd64) url={shlex.quote(x86_url)}; sha={x86_sha} ;; "
        f"aarch64|arm64) url={shlex.quote(arm_url)}; sha={arm_sha} ;; "
        '*) echo "Unsupported architecture for Muse Code: $arch" >&2; exit 1 ;; '
        "esac; "
        # Exit 2: no tool to check a digest with.
        "bf_sha256() { if command -v sha256sum >/dev/null 2>&1; then "
        'echo "$sha  $1" | sha256sum -c - >/dev/null; '
        "elif command -v shasum >/dev/null 2>&1; then "
        'echo "$sha  $1" | shasum -a 256 -c - >/dev/null; '
        "else return 2; fi; }; "
        f"[ -x {q_bin} ] && bf_sha256 {q_bin} 2>/dev/null && "
        f"MUSE_NO_AUTO_UPDATE=1 {q_bin} --version 2>/dev/null | "
        f"grep -qF {shlex.quote(MUSE_VERSION)} && exit 0; "
        # The rest runs only without a verified binary; errexit cannot guard it.
        "( command -v curl >/dev/null 2>&1 && [ -e /etc/ssl/certs/ca-certificates.crt ] || "
        f"{apt_install_curl} ) || exit 1; "
        'tmp="$(mktemp -d)" || exit 1; '
        "trap 'rm -rf \"$tmp\"' 0; "
        'curl -fsSL --retry 3 -o "$tmp/muse" "$url" || exit 1; '
        'bf_sha256 "$tmp/muse" || { [ $? -eq 2 ] && '
        "echo 'sha256sum or shasum is required to verify the Muse Code download' >&2; exit 1; } && "
        f"mkdir -p {q_dir} && "
        f'mv "$tmp/muse" {q_target} && '
        f"chmod 755 {q_target} && "
        f"ln -sf {q_target} {q_bin} ) && "
        f"chmod -R a+rX {q_dir} && "
        f"MUSE_NO_AUTO_UPDATE=1 {q_bin} --version"
    )
