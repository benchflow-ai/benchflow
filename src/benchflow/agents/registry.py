"""Agent registry — supported agents and their configurations.

Adding a new agent is a registry-only change: append one entry to ``AGENTS``
below. The SDK reads everything it needs about the agent from this dict, so no
``sdk.py`` edits are required. ``tests/test_registry_invariants.py`` runs
contract checks against every entry — read it for the executable schema.

Required fields
---------------
- ``name``               Must equal the dict key.
- ``install_cmd``        Bash command run inside the sandbox to install the
                         agent. Idempotent (use ``command -v ... ||`` guards).
- ``launch_cmd``         Command that starts the agent process. The SDK pipes
                         ACP messages to its stdin / reads from its stdout.

Common optional fields
----------------------
- ``protocol``           "acp" (default) or "session-factory" (an ``acpx/``
                         spec wraps an entry to launch through acpx). Almost
                         always "acp". Running an agent's own CLI in headless
                         JSON mode is not a protocol value: it is the run
                         option ``harness="native"``, available for the entries
                         ``benchflow.native_harness.harnesses`` lists.
- ``session_factory``    Non-ACP "module:callable" entrypoint used only when
                         ``protocol="session-factory"``.
- ``requires_env``       List of env var names the SDK must propagate into the
                         sandbox (e.g. ``["ANTHROPIC_API_KEY"]``). Validated at
                         run start; missing keys raise before the container
                         spins up.
- ``api_protocol``       "anthropic-messages" | "openai-completions" |
                         "openai-responses" | "" — the
                         LLM API the agent natively speaks. Used to pick the
                         right endpoint when a provider exposes multiple
                         (e.g. zai). Empty means "agent infers from model name".
- ``env_mapping``        ``BENCHFLOW_PROVIDER_*`` → agent-native env var.
                         Applied by the SDK after provider resolution. Keys
                         **must** start with ``BENCHFLOW_PROVIDER_``.
- ``skill_paths``        Sandbox paths where benchflow should mount skill
                         content. Must start with ``$HOME/`` or ``$WORKSPACE/``.
- ``credential_files``   ``CredentialFile`` entries for files written into the
                         container before launch (e.g. ``~/.codex/auth.json``).
- ``home_dirs``          Extra dot-dirs under ``$HOME`` to copy to the sandbox
                         user (for dirs not derivable from ``skill_paths`` /
                         ``credential_files``).
- ``temp_dirs``          Names of the agent tool's folders under the sandbox's
                         temp folder (``/tmp/opencode``: ``["opencode"]``),
                         made the sandbox user's before launch.
- ``subscription_auth``  ``SubscriptionAuth`` describing host CLI login files
                         (e.g. ``claude login`` credentials) that can stand in
                         for an API key. API keys still take precedence.

Look at the existing entries below for worked examples:
``claude-agent-acp`` (subscription auth + env_mapping), ``codex-acp``
(credential_files), and
``antigravity`` (pinned native binary + Python ACP shim, hook-based web
policy, native MCP config).
"""

import base64
import json
import shlex
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from benchflow._root_only import root_only_tree_check
from benchflow._root_python import root_python_sh
from benchflow._utils.text import describe_exception
from benchflow._uv_pin import (
    BENCHFLOW_ROOT,
    PYTHON_INSTALL_DIR,
    PYTHON_VERSION,
    UV_BIN,
    UV_DIR,
    managed_python_exports,
    managed_python_install_command,
    private_tmpdir_exports,
    uv_install_command,
    uv_lock_file_command,
    uv_lock_seal_command,
)
from benchflow.agents._python_agent_pins import (
    DEEPAGENTS_REQUIREMENTS,
    HARVEY_LAB_COMMIT,
    HARVEY_LAB_REPO,
    HARVEY_LAB_REQUIREMENTS,
    HARVEY_LAB_SPARSE_PATHS,
)
from benchflow.agents.antigravity_config import (
    AGY_VERSION,
    ANTIGRAVITY_GLOBAL_SKILLS_PATH,
    ANTIGRAVITY_HOME_DIR,
    ANTIGRAVITY_HOOKS_RELPATH,
    ANTIGRAVITY_HOSTED_WEB_TOOL_MATCHERS,
    ANTIGRAVITY_MCP_CONFIG_RELPATH,
    ANTIGRAVITY_WEB_TOOL_MATCHERS,
    ANTIGRAVITY_WORKSPACE_SKILLS_PATH,
    agy_install_cmd,
    hooks_deny_mutator,
)
from benchflow.agents.muse_config import (
    MUSE_AUTH_JSON_ENV,
    MUSE_AUTH_RELPATH,
    MUSE_SKILLS_PATH,
    muse_install_cmd,
)
from benchflow.agents.visible_runtime import visible_script


def _install_python_script(
    container_path: str, source: str, *, python: str | None = None
) -> str:
    """Write a shim using its existing private or fixed image interpreter.

    Base64 transport — makes the install shell line content-agnostic so a line
    like `SHIMEOF` or `LAUNCHEREOF` inside the Python source can't collide with
    a heredoc terminator.

    Used by pi-acp, harvey-lab-harness, deepagents, antigravity and muse-code —
    each ships a Python launcher/shim baked into install_cmd. Semantics differ
    intentionally: pi bridges BENCHFLOW_PROVIDER_* env vars to agent-native
    config; harvey-lab delegates to Harvey LAB's own model adapters which read
    provider env vars directly; antigravity bridges ACP to agy's headless
    stream-json protocol; muse-code runs one `muse exec --json` per prompt.
    A shared base is not yet justified —
    divergence is cheap, premature abstraction isn't.
    """
    encoded = base64.b64encode(visible_script(source).encode()).decode()
    parent = shlex.quote(str(Path(container_path).parent))
    q_path = shlex.quote(container_path)
    ensure_python = (
        f"[ -x {shlex.quote(python)} ]"
        if python is not None
        else "( [ -x /usr/bin/python3 ] || [ -x /usr/local/bin/python3 ] || "
        "(apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1) )"
    )
    return (
        # Never consult an earlier role's PATH. Venv-backed shims already
        # installed their exact interpreter and need no system Python or apt.
        f"{ensure_python} && "
        f"mkdir -p {parent} && "
        f"echo {encoded} | base64 -d > {q_path} && "
        f"chmod +x {q_path}"
    )


def _python_shim_launch(path: str) -> str:
    """Use the same fixed image locations admitted by the shim installer."""
    return (
        "bf_shim_python=/usr/bin/python3; "
        '[ -x "$bf_shim_python" ] || bf_shim_python=/usr/local/bin/python3; '
        f'exec "$bf_shim_python" -I -S {shlex.quote(path)}'
    )


def _write_requirements(
    container_path: str, requirements: tuple[str, ...], *, python: str = "python3"
) -> str:
    """Shell snippet that writes pinned requirement lines to container_path.

    Compress hash locks to stay below the shell's per-argument size limit.
    The already-created agent venv supplies the decoder, not the image Python.
    Isolate root helpers from the workspace, user site and PYTHON* settings;
    an earlier role can write those before a later role installs its agent.
    """
    encoded = base64.b64encode(
        zlib.compress(("\n".join(requirements) + "\n").encode())
    ).decode()
    q_path = shlex.quote(container_path)
    return (
        f"mkdir -p {shlex.quote(str(Path(container_path).parent))} && "
        f"printf '%s' {shlex.quote(encoded)} | "
        # Root's decoder on the protected Python (C8, PT-C8-R8-2): the kit's
        # once the sandbox has one, else this venv's, isolated, from /.
        + root_python_sh(
            "import base64, sys, zlib; "
            "sys.stdout.buffer.write(zlib.decompress(base64.b64decode("
            "sys.stdin.buffer.read())))",
            fallback=shlex.quote(python),
        )
        + f" > {q_path}"
    )


def _apt_install(*packages: str) -> str:
    """POSIX shell snippet for apt installs with transient mirror recovery."""
    package_args = " ".join(shlex.quote(package) for package in packages)
    return (
        "( attempt=1; "
        'while [ "$attempt" -le 3 ]; do '
        "rm -rf /var/lib/apt/lists/*; "
        "apt-get clean; "
        "if apt-get -o Acquire::Retries=3 update -qq && "
        f"apt-get -o Acquire::Retries=3 install -y -qq {package_args}; then "
        "exit 0; "
        "fi; "
        'case "$attempt" in 1) sleep 2; attempt=2 ;; '
        "2) sleep 4; attempt=3 ;; "
        "*) sleep 6; attempt=4 ;; esac; "
        "done; "
        "exit 1 )"
    )


@dataclass(frozen=True)
class RequirementsFile:
    """One host-authored lock, with a self-contained image-build recipe.

    Live installs upload its bytes and replace exactly this recipe. A later
    role can therefore install even when the sealed kit has no Python.
    """

    path: str
    lines: tuple[str, ...]
    python: str

    @property
    def command(self) -> str:
        return _write_requirements(self.path, self.lines, python=self.python)

    @property
    def content(self) -> str:
        return "\n".join(self.lines) + "\n"


# Isolated Node.js bootstrap for JavaScript-based ACP agents.
#
# Keep this out of system prefixes. Task images may need their own Node/npm
# versions, so BenchFlow installs agent runtime bits under /opt/runtime.
# Install commands can put that prefix on PATH, but launch wrappers call the
# private Node binary explicitly so task subprocesses keep the task's PATH.
_BENCHFLOW_NODE_PREFIX = "/opt/runtime/node"
_BENCHFLOW_JS_AGENT_PREFIX = "/opt/runtime/js-agents"
_BENCHFLOW_BIN_PREFIX = "/opt/runtime/bin"
_AGENT_UV_ROOT = f"{BENCHFLOW_ROOT}/agents"
_AGENT_UV_BIN = f"{_AGENT_UV_ROOT}/bin"
# OpenHands' uv tool (its venv and the entry point uv links into
# _BENCHFLOW_BIN_PREFIX), in BenchFlow's protected tree (C8). The direct
# execution patch (acp.runtime) checks and runs it from there.
_OPENHANDS_TOOL_DIR = f"{BENCHFLOW_ROOT}/uv-tools"
_DEEPAGENTS_VENV = f"{_AGENT_UV_ROOT}/deepagents-venv"
_HARVEY_LAB_VENV = f"{_AGENT_UV_ROOT}/harvey-lab-venv"
_HARVEY_LAB_ROOT = f"{_AGENT_UV_ROOT}/harvey-labs"
_HARVEY_LAB_REQUIREMENTS_PATH = f"{_AGENT_UV_ROOT}/harvey-lab-requirements.txt"
_DEEPAGENTS_REQUIREMENTS_PATH = f"{_AGENT_UV_ROOT}/deepagents-requirements.txt"
# Root uv's TMPDIR during agent installs (see private_tmpdir_exports).
_AGENT_UV_TMP = f"{_AGENT_UV_ROOT}/uv-tmp"


# Git before 2.32 ignores GIT_CONFIG_GLOBAL and reads $HOME/.gitconfig and
# $XDG_CONFIG_HOME/git/config, which an earlier role may have written as
# root's HOME (FF-install-C13). /dev/null holds no file, so every Git reads
# none; git treats the ENOTDIR as a missing file. Set on each installer Git
# command and on the uv call that runs Git, not exported: later steps of a
# recipe still write the agent's settings under the real HOME.
_AGENT_GIT_HOME = "HOME=/dev/null XDG_CONFIG_HOME=/dev/null"
_AGENT_GIT = f"{_AGENT_GIT_HOME} git -c core.hooksPath=/dev/null"


def _agent_git_exports() -> str:
    """Give installer Git, including uv's children, only fixed settings.

    A pinned checkout still runs hooks and reads includes or URL rewrites
    from HOME/XDG config. Earlier roles may own those directories. Clear
    the entire GIT_* namespace, then disable ambient config, hooks and init
    templates. Command-scope settings also override a checkout's config.
    """
    return (
        "for v in $(env | sed -n 's/^\\(GIT_[A-Za-z0-9_]*\\)=.*/\\1/p'); "
        'do unset "$v"; done; '
        "export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null "
        "GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_COUNT=2 "
        "GIT_CONFIG_KEY_0=core.hooksPath GIT_CONFIG_VALUE_0=/dev/null "
        "GIT_CONFIG_KEY_1=init.templateDir GIT_CONFIG_VALUE_1=''"
    )


def _agent_uv_setup() -> str:
    """Every root-side agent uv call shares the proxy's isolation policy.

    Braces preserve exports in the caller while keeping each required step
    in the recipe's && chain. The agent's cwd and old role's UV_* variables
    must never select a Python catalog, tool environment or cache.
    """
    # Root runs and replaces what these folders hold (the venvs, the
    # harness checkout, the tool and its entry point): each is shown root's
    # alone first (C8), as root; tests run the recipe as their own user.
    folders = (_AGENT_UV_ROOT, _OPENHANDS_TOOL_DIR, _BENCHFLOW_BIN_PREFIX)
    checks = " && ".join(
        root_only_tree_check(folder, "BenchFlow's agent uv tools") for folder in folders
    )
    guard = (
        f'mkdir -p {" ".join(folders)} && {{ [ "$(id -u)" != 0 ] || {{ {checks}; }}; }}'
    )
    return (
        f"{{ {_agent_git_exports()}; }} && "
        f"{uv_install_command()} && "
        f"{{ {managed_python_exports()}; }} && "
        f"{guard} && "
        # Before the first uv call that downloads: uv unpacks hash-checked
        # files in TMPDIR, which must not be /tmp (FF-install-C12).
        f"{private_tmpdir_exports(_AGENT_UV_TMP)} && "
        f"{{ {managed_python_install_command(shlex.quote(UV_BIN))}; }} && "
        f"export UV_TOOL_DIR={_OPENHANDS_TOOL_DIR} "
        f"UV_TOOL_BIN_DIR={_BENCHFLOW_BIN_PREFIX} && "
        f'export PATH="{UV_DIR}:{_BENCHFLOW_BIN_PREFIX}:{_AGENT_UV_BIN}:$PATH"'
    )


# OpenCode-family proxy provider id. OpenCode hard-codes the OpenAI *Responses*
# API for the built-in ``openai`` provider id (its ``getModel`` calls
# ``provider.responses(id)``), which the LiteLLM gateway/DeepSeek cannot serve —
# so the gateway alias must be registered under a *separate* provider id that
# OpenCode routes through the chat-completions path. Shared with
# ``benchflow.acp.runtime._format_acp_model`` so set_model targets the same id.
OPENCODE_PROXY_PROVIDER_ID = "runtime"
# Gemini models skip that chat-completions alias: OpenCode keeps its native
# ``google`` provider (``@ai-sdk/google``, which carries Gemini's thought
# signatures and thinking config) and the ``-proxy`` wrapper points it at
# LiteLLM's byte-preserving Gemini pass-through, the route Gemini CLI uses.
# Set by ``litellm_runtime._wire_litellm_agent_env``; read by the wrapper.
OPENCODE_GOOGLE_BASE_URL_ENV = "BENCHFLOW_OPENCODE_GOOGLE_BASE_URL"
# Exact pins. The Codex Apps and native Claude no-web admission gates verify
# installed clients against these through pinned_npm_package(), so a bump here
# retargets them; re-run their credential-free conformance fixtures when bumping.
# Stay on the 0.81 line: 0.82.0 changed the tool-call contract (AIR).
_CLAUDE_AGENT_ACP_PACKAGE = "@agentclientprotocol/claude-agent-acp@0.81.2"
# The Claude Code CLI that claude-agent-acp drives, pinned apart from the
# adapter. Models set a minimum CLI version
# (claude-opus-5-5 refuses Claude Code older than 2.1.280, #1137), and the CLI
# the adapter's SDK bundles only moves with an adapter release, which can
# break the ACP contract. The launcher hands this CLI to the adapter through
# CLAUDE_CODE_EXECUTABLE, which claude-agent-acp reads before looking for its
# SDK's bundled binary. That binary is still installed (npm keeps a global
# package's optional dependencies even with --omit=optional) but never runs;
# the native no-web gate verifies the launcher.
# 2.1.316 (2026-11-04): 2.1.280 predates claude-sonnet-5-5 (2.1.284) and
# claude-haiku-5-5 (2.1.293, with its token counts behind a gateway fixed in
# 2.1.296), and its "haiku" alias still meant Haiku 4.5, so 0.81.2 refused
# claude-haiku-5-5 at connect. 2.1.316 is the CLI claude-agent-acp 0.100.0's
# SDK was built against; it is past 2.1.304-2.1.313 (tool search turned on
# behind a custom ANTHROPIC_BASE_URL, which failed requests until 2.1.305 and
# repeated refusals until 2.1.314) and before 2.1.317's change to Read.
_CLAUDE_CODE_PACKAGE = "@anthropic-ai/claude-code@2.1.316"
_CODEX_ACP_PACKAGE = "@agentclientprotocol/codex-acp@2.0.1"
# Pi moved to @earendil-works (same maintainers); the old package stopped at
# 0.73.1, deprecated, and pi-acp 0.0.34 needs pi 0.81.0 or later: with 0.73.1
# session/new never answered, so every pi-acp trial failed at connect.
# 1.0.0 (2026-11-04): its catalog has Opus 5.5, Sonnet 5.5, Fable 5.1 and
# gpt-6.1-sol, which 0.81.1's lacks; it is the last release that ships
# npm-shrinkwrap.json (1.0.1 dropped it, so later installs float their
# dependencies).
_PI_PACKAGE = "@earendil-works/pi-coding-agent@1.0.0"
_PI_ACP_PACKAGE = "pi-acp@0.0.34"
# Both harnesses use this exact engine. The ACP install replaces its adapter's
# caret dependency with the pin in the same tree the launcher and policy probe
# resolve, then verifies the running CLI before accepting the installation.
_CODEX_CLI_PACKAGE = "@openai/codex@0.159.3"
CODEX_APP_SERVER_EXECUTABLE = f"{_BENCHFLOW_BIN_PREFIX}/codex-app-server"


# OpenHands-CLI's last commit (2026-10-22; unmaintained since 2026-08-11):
# only workflows, README and pyproject/uv.lock, raising certifi to
# 2026.10.19, which drops the eMudhra and emSign roots. Its main still pins
# openhands-sdk 1.28.1, so the SDK stays there.
_OPENHANDS_CLI_GIT_REV = "283c6ac3610f4299f405f7389f65766dcb83e12a"
_OPENHANDS_SDK_VERSION = "1.28.1"
_OPENHANDS_TOOLS_VERSION = "1.28.1"
_JS_AGENT_PATH = (
    f"{_BENCHFLOW_BIN_PREFIX}:{_BENCHFLOW_JS_AGENT_PREFIX}/bin:"
    f"{_BENCHFLOW_NODE_PREFIX}/bin:$PATH"
)
# Shared JavaScript-agent runtime pin. Root installs into and runs from
# /opt/runtime only while it is root's alone (C8): a workspace grant
# (``WORKDIR /opt``) can hand it to the agent user, whose Node or npm root
# would run next. The install then stops instead.
_NODE_INSTALL = (
    "mkdir -p /opt/runtime && "
    + root_only_tree_check("/opt/runtime", "BenchFlow's agent runtime folder")
    + " || exit 1; "
    "export DEBIAN_FRONTEND=noninteractive; "
    f"BF_NODE_DIR={_BENCHFLOW_NODE_PREFIX}; "
    "BF_NODE_VERSION=22.20.0; "
    'if [ ! -x "$BF_NODE_DIR/bin/node" ]; then '
    "  if ! command -v curl >/dev/null 2>&1 || "
    "     ! command -v tar >/dev/null 2>&1 || "
    "     ! command -v xz >/dev/null 2>&1; then "
    "    if command -v apt-get >/dev/null 2>&1; then "
    "      apt-get update -qq && "
    "      apt-get install -y -qq curl ca-certificates tar xz-utils; "
    "    elif command -v dnf >/dev/null 2>&1; then "
    "      dnf -y install curl ca-certificates tar xz; "
    "    elif command -v apk >/dev/null 2>&1; then "
    "      apk add --no-cache curl ca-certificates tar xz; "
    "    else "
    "      echo 'BenchFlow JS agent bootstrap requires curl, tar, and xz' >&2; "
    "      exit 127; "
    "    fi; "
    "  fi; "
    '  arch="$(uname -m)"; '
    '  case "$arch" in '
    "    x86_64|amd64) node_arch=x64 ;; "
    "    aarch64|arm64) node_arch=arm64 ;; "
    '    *) echo "Unsupported architecture for Node.js: $arch" >&2; exit 1 ;; '
    "  esac; "
    '  tmp="$(mktemp -d)"; '
    "  mkdir -p /opt/runtime; "
    '  curl -fsSLo "$tmp/node.tar.xz" '
    '"https://nodejs.org/dist/v${BF_NODE_VERSION}/node-v${BF_NODE_VERSION}-linux-${node_arch}.tar.xz"; '
    '  rm -rf "$BF_NODE_DIR"; '
    '  mkdir -p "$BF_NODE_DIR"; '
    '  tar -xJf "$tmp/node.tar.xz" -C "$BF_NODE_DIR" --strip-components=1 --no-same-owner; '
    '  rm -rf "$tmp"; '
    "fi; "
    f'export PATH="{_BENCHFLOW_NODE_PREFIX}/bin:$PATH"; '
    '"$BF_NODE_DIR/bin/node" --version; '
    '"$BF_NODE_DIR/bin/npm" --version'
)


def _npm_package_spec(package: str) -> str:
    """Return an npm install spec, defaulting unversioned packages to latest."""
    if "@" in package.lstrip("@"):
        return package
    return f"{package}@latest"


def pinned_npm_package(agent: str) -> tuple[str, str]:
    """Return the built-in ``(package, version)`` pin of a policy-gated client.

    ``agent`` names a policy-gated ACP agent, ``"claude-code"`` for the
    Claude Code CLI that ``claude-agent-acp`` runs, or ``"codex"`` for the
    Codex CLI the native harness runs. Reads the constants, not mutable
    ``AGENTS``: a manifest override must not redefine the version an
    admission gate verifies.
    """
    spec = {
        "claude-agent-acp": _CLAUDE_AGENT_ACP_PACKAGE,
        "claude-code": _CLAUDE_CODE_PACKAGE,
        "codex-acp": _CODEX_ACP_PACKAGE,
        "codex": _CODEX_CLI_PACKAGE,
    }[agent]
    package, _, version = spec.rpartition("@")
    return package, version


# npm links the Claude Code package's native binary here.
CLAUDE_CODE_EXECUTABLE_PATH = f"{_BENCHFLOW_JS_AGENT_PREFIX}/bin/claude"
# The Codex CLI's npm entry point is a Node script; this launcher runs it on
# BenchFlow's private Node, so the task image needs no Node of its own.
CODEX_CLI_EXECUTABLE_PATH = f"{_BENCHFLOW_BIN_PREFIX}/codex"


def js_agent_launcher(binary: str, env: tuple[tuple[str, str], ...] = ()) -> str:
    """The launcher script ``_js_agent_install`` writes to ``/opt/runtime/bin``.

    It exports ``env`` and runs the agent on BenchFlow's private Node. The
    native Claude admission gate compares the installed launcher against it.
    """
    lines = [
        "#!/bin/sh",
        *(f"export {name}={shlex.quote(value)}" for name, value in env),
        f"exec {_BENCHFLOW_NODE_PREFIX}/bin/node "
        f'{_BENCHFLOW_JS_AGENT_PREFIX}/bin/{binary} "$@"',
    ]
    return "".join(f"{line}\n" for line in lines)


_CLAUDE_AGENT_ACP_ENV = (("CLAUDE_CODE_EXECUTABLE", CLAUDE_CODE_EXECUTABLE_PATH),)
CLAUDE_AGENT_ACP_LAUNCHER = js_agent_launcher("claude-agent-acp", _CLAUDE_AGENT_ACP_ENV)


def _js_agent_install(
    binary: str,
    package: str,
    *,
    companion: str = "",
    launcher_env: tuple[tuple[str, str], ...] = (),
    verify: str = "",
) -> str:
    """Install an npm-distributed agent into BenchFlow's isolated prefix.

    ``companion`` is one more exact package spec installed into the same
    prefix. The launcher exports ``launcher_env`` before it starts the agent,
    and ``verify`` is a last shell check the install must pass.
    """
    agent_bin = f"{_BENCHFLOW_JS_AGENT_PREFIX}/bin/{binary}"
    wrapper = f"{_BENCHFLOW_BIN_PREFIX}/{binary}"
    package_spec = _npm_package_spec(package)
    install_guard = "" if package_spec == package else f"[ -x {agent_bin} ] || "
    npm = f"{_BENCHFLOW_NODE_PREFIX}/bin/npm install -g --prefix {_BENCHFLOW_JS_AGENT_PREFIX}"
    companion_install = f"( {npm} {companion} ) && " if companion else ""
    launcher_lines = js_agent_launcher(binary, launcher_env).splitlines()
    return (
        f"{_NODE_INSTALL} && "
        f"mkdir -p {_BENCHFLOW_JS_AGENT_PREFIX} {_BENCHFLOW_BIN_PREFIX} && "
        f'export PATH="{_JS_AGENT_PATH}" && '
        f"( {install_guard}{npm} {package_spec} ) && "
        f"{companion_install}"
        f"printf '%s\\n' {' '.join(shlex.quote(line) for line in launcher_lines)} "
        f"> {wrapper} && "
        f"chmod +x {wrapper} && "
        f"chmod -R a+rX /opt/runtime && "
        f"[ -x {agent_bin} ] && [ -x {wrapper} ]" + (f" && {verify}" if verify else "")
    )


def _claude_code_version_check() -> str:
    """Fail the install unless the pinned Claude Code CLI runs and reports its pin.

    The package's postinstall copies the native binary over a stub, so an
    install that skipped scripts or optional dependencies fails here rather
    than on the rollout's first model call.
    """
    _, version = pinned_npm_package("claude-code")
    cli = CLAUDE_CODE_EXECUTABLE_PATH
    expected = f"{version} (Claude Code)"
    return (
        f'( v="$({cli} --version 2>&1)"; [ "$v" = {shlex.quote(expected)} ] || '
        f'{{ echo "BenchFlow: {cli} is not Claude Code {version}: $v" >&2; exit 1; }} )'
    )


def _codex_cli_version_check() -> str:
    """Fail the install unless the pinned Codex CLI runs and reports its pin."""
    _, version = pinned_npm_package("codex")
    cli = CODEX_CLI_EXECUTABLE_PATH
    expected = f"codex-cli {version}"
    return (
        f'( v="$({cli} --version 2>&1)"; [ "$v" = {shlex.quote(expected)} ] || '
        f'{{ echo "BenchFlow: {cli} is not Codex {version}: $v" >&2; exit 1; }} )'
    )


def codex_cli_install_cmd() -> str:
    """Install the pinned Codex CLI next to codex-acp, for the native harness."""
    return _js_agent_install(
        "codex", _CODEX_CLI_PACKAGE, verify=_codex_cli_version_check()
    )


def claude_code_version_check() -> str:
    """The install-time check of the pinned Claude Code CLI (see below)."""
    return _claude_code_version_check()


def _js_agent_launch(binary: str, args: str = "") -> str:
    """Launch a JS agent through its isolated BenchFlow wrapper."""
    cmd = f"{_BENCHFLOW_BIN_PREFIX}/{binary}"
    return f"{cmd} {args}".rstrip()


def codex_app_server_launcher(
    node: str, cli: str, *, module_root: str | None = None
) -> str:
    """Pass session config to every Codex thread without replacing user TOML.

    codex-acp 2.0.1 ignores its own ``-c`` flags but starts CODEX_PATH with
    ``app-server``. Codex's global overrides apply to that server's implicit
    title thread too. Resolve the same bundled CLI as the adapter and its Apps
    admission probe, never PATH or a previous role's executable. No config
    parser or image Python is needed.
    """
    js = r"""
const {spawn} = require("node:child_process");
const config = JSON.parse(process.env.CODEX_CONFIG || "{}");
if (!config || Array.isArray(config) || typeof config !== "object") {
  throw new Error("CODEX_CONFIG must be an object");
}
function value(v) {
  if (typeof v === "string") return JSON.stringify(v).replace(/\x7f/g, "\\u007f");
  if (typeof v === "boolean" || (typeof v === "number" && Number.isFinite(v))) {
    return JSON.stringify(v);
  }
  if (Array.isArray(v)) return "[" + v.map(value).join(", ") + "]";
  if (v && typeof v === "object") {
    return "{" + Object.entries(v).map(([k, x]) => value(k) + "=" + value(x)).join(",") + "}";
  }
  throw new Error("CODEX_CONFIG contains a value without a TOML representation");
}
const provider = (config.model_providers || {})[config.model_provider];
if (provider && typeof provider.base_url === "string" && provider.base_url) {
  config.openai_base_url = provider.base_url;
}
const flags = Object.entries(config).flatMap(([k, v]) => {
  if (!/^[A-Za-z0-9_-]+$/.test(k)) throw new Error("CODEX_CONFIG top-level key is not a bare key");
  return ["-c", k + "=" + value(v)];
});
const cli = require.resolve(process.argv[1], {paths: MODULE_ROOTS});
const child = spawn(process.execPath, [cli, ...flags, ...process.argv.slice(2)], {stdio: "inherit"});
for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => child.kill(signal));
}
child.on("error", () => { console.error("BenchFlow: cannot start pinned Codex CLI"); process.exit(1); });
child.on("exit", (code, signal) => {
  if (signal) { process.removeAllListeners(signal); process.kill(process.pid, signal); }
  else process.exit(code === null ? 1 : code);
});
""".strip()
    js = js.replace("MODULE_ROOTS", json.dumps([module_root] if module_root else []))
    return f'#!/bin/sh\nexec {shlex.quote(node)} -e {shlex.quote(js)} {shlex.quote(cli)} "$@"\n'


def _codex_acp_install() -> str:
    wrapper = codex_app_server_launcher(
        f"{_BENCHFLOW_NODE_PREFIX}/bin/node",
        "@openai/codex/bin/codex.js",
        module_root=f"{_BENCHFLOW_JS_AGENT_PREFIX}/lib/node_modules/@agentclientprotocol/codex-acp",
    )
    encoded = base64.b64encode(visible_script(wrapper).encode()).decode()
    return (
        _js_agent_install("codex-acp", _CODEX_ACP_PACKAGE)
        + f" && {_BENCHFLOW_NODE_PREFIX}/bin/npm install --omit=dev"
        + f" --prefix {_BENCHFLOW_JS_AGENT_PREFIX}/lib/node_modules/@agentclientprotocol/codex-acp"
        + f" {_CODEX_CLI_PACKAGE}"
        + f' && ( v="$({_BENCHFLOW_BIN_PREFIX}/codex-acp cli -V 2>&1)"; '
        + f'[ "$v" = "codex-cli {pinned_npm_package("codex")[1]}" ] || '
        + '{ echo "Pinned Codex engine version check failed: $v" >&2; exit 1; } )'
        + f" && printf '%s' {shlex.quote(encoded)} | base64 -d > {CODEX_APP_SERVER_EXECUTABLE}"
        + f" && chmod 755 {CODEX_APP_SERVER_EXECUTABLE}"
    )


_MIMO_MANIFEST_INSTALL_CMD = (
    _NODE_INSTALL
    + " && "
    + "mkdir -p /opt/runtime/js-agents/mimo-acp && "
    + 'printf \'%s\' \'{"name":"bf-mimo-acp","private":true,"type":"module"}\' '
    + "> /opt/runtime/js-agents/mimo-acp/package.json && "
    + "cd /opt/runtime/js-agents/mimo-acp && "
    + "/opt/runtime/node/bin/npm install @mimo-ai/cli@0.1.15 "
    + "--no-audit --no-fund --loglevel=error && "
    + "chmod -R a+rX /opt/runtime && "
    + "[ -f /opt/runtime/js-agents/mimo-acp/node_modules/@mimo-ai/cli/bin/mimo ]"
)
_MIMO_MANIFEST_LAUNCHER = r"""
A="${BENCHFLOW_LITELLM_MODEL_ALIAS:-}"
B="${OPENAI_BASE_URL:-}"
K="${OPENAI_API_KEY:-}"
# Two wirings, both valid: PROXY mode (alias A present — write the custom
# "openai" provider at the proxy and strip the colliding OPENAI_* env) and
# DIRECT-provider mode (no alias — leave OPENAI_* intact and write nothing;
# benchflow delivers the model over ACP set_model as provider/model). The old
# launcher hard-failed (rc78) on "base URL without alias", which broke every
# direct-provider run.
if [ -n "$B" ] && [ -n "$A" ]; then
  (
  umask 077
  # Canonical config location MiMo Code (OpenCode fork) reads:
  # {home}/.config/mimocode/mimocode.json (matches benchflow-core's native mimo
  # agent). Also written to ./mimocode.json (cwd) as a belt-and-suspenders for
  # the proven-working run, so the routing is not cwd-dependent.
  CFG_HOME="${HOME:-/root}/.config/mimocode"
  mkdir -p "$CFG_HOME"
  CFG_JSON='{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "runtime": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Model endpoint",
      "options": { "baseURL": "'"$B"'", "apiKey": "'"${K:-runtime}"'" },
      "models": { "'"$A"'": { "name": "'"$A"'" } }
    }
  },
  "model": "runtime/'"$A"'",
  "small_model": "runtime/'"$A"'"
}'
  printf '%s\n' "$CFG_JSON" > "$CFG_HOME/mimocode.json"
  printf '%s\n' "$CFG_JSON" > ./mimocode.json
  ) || exit 1
  CFG_WRITTEN=yes
else
  CFG_WRITTEN=no
fi
if [ -n "${BF_PR8_DEBUG:-}" ]; then
  H=$(printf %s "$B" | sed -E "s#^[a-z]+://([^@]*@)?([^/:?]+).*#\2#")
  echo "BF_DIAG base_url_set=$([ -n \"$B\" ] && echo yes || echo no) host=${H:-NONE}"
  echo "BF_DIAG api_key_set=$([ -n \"$K\" ] && echo yes || echo no)"
  echo "BF_DIAG alias=${A:-NONE}"
  echo "BF_DIAG cfg_written=$CFG_WRITTEN cfg_cwd=$(pwd)/mimocode.json cfg_home=${HOME}/.config/mimocode/mimocode.json MIMOCODE_CONFIG=${MIMOCODE_CONFIG:-unset}"
  echo BF_DIAG mimocode_begin
  for p in ./mimocode.json "${HOME}/.config/mimocode/mimocode.json" "${MIMOCODE_CONFIG:-}"; do
    if [ -n "$p" ] && [ -f "$p" ]; then
      sed -E -e 's#("apiKey"[[:space:]]*:[[:space:]]*")[^"]*#\1<REDACTED>#g' \
             -e 's#://[^@/"]*@#://<REDACTED>@#g' "$p" \
        | awk -v pfx="BF_DIAG_CFG $p: " '{print pfx $0}'
    fi
  done
  echo BF_DIAG mimocode_end
fi
if [ -n "$A" ]; then
  # Proxy mode only: the alias is registered under the "benchflow" provider
  # (the id BenchFlow selects over ACP set_model, OPENCODE_PROXY_PROVIDER_ID),
  # and mimo's built-in openai provider auto-activates from
  # OPENAI_* env and would conflict with the custom provider written above.
  # In direct mode the env IS the provider config — leave it intact.
  unset OPENAI_BASE_URL OPENAI_API_KEY
fi
exec /opt/runtime/node/bin/node /opt/runtime/js-agents/mimo-acp/node_modules/@mimo-ai/cli/bin/mimo acp
"""
_MIMO_MANIFEST_LAUNCH_CMD = (
    "printf '%s' '"
    + base64.b64encode(visible_script(_MIMO_MANIFEST_LAUNCHER).encode()).decode()
    + "' | base64 -d > /tmp/mimo-acp-launch.sh && sh /tmp/mimo-acp-launch.sh"
)


# Path to the Pi launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
_PI_LAUNCHER = (Path(__file__).parent / "pi_acp_launcher.py").read_text()

# Path to the Harvey LAB ACP shim (runs Harvey LAB harness as an ACP agent)
_HARVEY_LAB_SHIM = (Path(__file__).parent / "harvey_lab_acp_shim.py").read_text()
# The two Python agents' hash locks (PT b179): uploaded by a live install
# (agents.install), written by the recipe in an image build; in the folder
# _agent_uv_setup checks root's alone before uv reads them either way.
_HARVEY_LAB_LOCK = RequirementsFile(
    _HARVEY_LAB_REQUIREMENTS_PATH,
    HARVEY_LAB_REQUIREMENTS,
    f"{_HARVEY_LAB_VENV}/bin/python",
)
_DEEPAGENTS_LOCK = RequirementsFile(
    _DEEPAGENTS_REQUIREMENTS_PATH,
    DEEPAGENTS_REQUIREMENTS,
    f"{_DEEPAGENTS_VENV}/bin/python",
)

# Path to the deepagents ACP shim (runs LangChain's create_deep_agent as an ACP agent)
_DEEPAGENTS_SHIM = (Path(__file__).parent / "deepagents_acp_shim.py").read_text()

# Path to the Antigravity ACP shim (drives `agy --input-format=stream-json` as an
# ACP agent; agy has no ACP mode of its own)
_ANTIGRAVITY_SHIM = (Path(__file__).parent / "antigravity_acp_shim.py").read_text()

# Path to the Muse Code ACP shim (runs one `muse exec --json` per ACP prompt;
# Muse Code has no ACP mode of its own)
_MUSE_SHIM = (Path(__file__).parent / "muse_acp_shim.py").read_text()


def _json_settings_merge_py(path: str, mutator: str) -> str:
    """The isolated Python command of an idempotent JSON-settings merge."""
    py = (
        "import json,os,pathlib;"
        f"p=pathlib.Path(os.path.expandvars(os.path.expanduser({path!r})));"
        "p.parent.mkdir(parents=True, exist_ok=True);"
        "d=json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {};"
        f"{mutator};"
        "p.write_text(json.dumps(d, indent=2) + '\\n')"
    )
    # Custom installs may bring only /usr/local/bin/python3. Keep that
    # supported image layout without consulting the previous role's PATH.
    return (
        "bf_settings_python=/usr/bin/python3; "
        '[ -x "$bf_settings_python" ] || bf_settings_python=/usr/local/bin/python3; '
        f'"$bf_settings_python" -I -S -c {shlex.quote(py)}'
    )


def _json_settings_merge(path: str, mutator: str) -> str:
    """Idempotent JSON-settings merge as a one-line bash snippet."""
    # Name the requirement: a bare "python3: not found" reads as an agent bug.
    return (
        "{ [ -x /usr/bin/python3 ] || [ -x /usr/local/bin/python3 ] || "
        "{ echo 'BenchFlow agent settings "
        "setup needs python3 in the task image' >&2; exit 127; }; "
        f"{_json_settings_merge_py(path, mutator)}; }}"
    )


def _json_settings_merge_node(path: str, mutator_js: str, mutator_py: str) -> str:
    """Idempotent JSON-settings merge with the Node runtime a JS agent installs.

    For agents whose install brings BenchFlow's private Node
    (``_js_agent_install``): their settings no longer need ``python3`` in the
    task image (REPORT B8: ``FROM ubuntu:24.04``, the base ``bench new``
    writes, has none). ``mutator_js`` edits ``d`` (the parsed file, ``{}`` when
    absent or empty) in plain ES2015 so a task's older ``node`` on PATH runs it
    too; ``mutator_py`` is the same edit for ``python3``, used only when no
    Node is there (a manifest or plugin replaced the install). ``path`` is
    expanded by the shell (``$BENCHFLOW_AGENT_HOME``).
    """
    js = (
        'var fs=require("fs"),path=require("path"),p=process.argv[1];'
        "fs.mkdirSync(path.dirname(p),{recursive:true});"
        'var t=fs.existsSync(p)?fs.readFileSync(p,"utf8").trim():"";'
        "var d=t?JSON.parse(t):{};"
        f"{mutator_js};"
        'fs.writeFileSync(p,JSON.stringify(d,null,2)+"\\n");'
    )
    node = f"{_BENCHFLOW_NODE_PREFIX}/bin/node"
    return (
        f'{{ bf_node={node}; [ -x "$bf_node" ] || '
        'bf_node="$(command -v node 2>/dev/null)" || bf_node=""; '
        f'if [ -n "$bf_node" ]; then "$bf_node" -e {shlex.quote(js)} "{path}"; '
        "elif [ -x /usr/bin/python3 ] || [ -x /usr/local/bin/python3 ]; then "
        f"{_json_settings_merge_py(path, mutator_py)}; "
        "else echo 'BenchFlow agent settings setup needs node (the agent install "
        "brings it) or python3 in the task image' >&2; exit 127; fi; }"
    )


def _js_deny_list(key: str, tools: tuple[str, ...]) -> str:
    """``mutator_js``: add ``tools`` to ``d[key].deny``, once each."""
    names = ",".join(f'"{tool}"' for tool in tools)
    return (
        f"d.{key}=d.{key}||{{}};d.{key}.deny=d.{key}.deny||[];"
        f"[{names}].forEach(function(t){{"
        f"if(d.{key}.deny.indexOf(t)<0)d.{key}.deny.push(t);}})"
    )


# OpenCode-family proxy fix: OpenCode and its MiMo fork validate provider/model
# ids against the models.dev catalog and reject the synthetic
# ``openai/runtime-<alias>`` that BenchFlow's LiteLLM proxy serves the model
# under (ProviderModelNotFoundError -> zero requests -> no
# ``trajectory/llm_trajectory.jsonl``, which ``benchflow-experiment-review``
# requires). Registering the alias under ``provider.openai.models`` (with the
# gateway baseURL) bypasses that catalog check so the agent accepts the id and
# routes through the proxy.
def _opencode_family_proxy_wrapper_install(binary: str, config_path: str) -> str:
    """Install ``/opt/runtime/bin/<binary>-proxy``: a thin wrapper that, in
    proxy mode, registers the LiteLLM gateway alias under a dedicated
    OpenCode provider, then execs the isolated agent binary. Idempotent
    (preserves existing config); no-op outside proxy mode.

    The gateway alias is registered under the ``{OPENCODE_PROXY_PROVIDER_ID}``
    provider using ``@ai-sdk/openai-compatible`` — NOT the built-in ``openai``
    provider. OpenCode-family agents hard-code the OpenAI **Responses API** for
    the ``openai`` provider id (``getModel`` calls ``provider.responses(id)``);
    BenchFlow's LiteLLM gateway — and the OpenAI-completions upstreams it fronts
    (e.g. DeepSeek) — only serve **chat completions**, so a Responses-API call
    404s/500s and the agent idles with zero tool calls. Overriding the ``openai``
    provider's ``npm`` does not help: OpenCode still calls ``.responses()`` and
    crashes with ``provider.responses is not a function``. A *separate* provider
    id routes through the chat-completions path, which is what the gateway
    expects. ``small_model`` is pinned to the same alias so OpenCode's
    title/summary helper stops falling back to its hard-coded ``gpt-5-nano``
    (which the gateway cannot serve either).

    A Gemini model has no alias: with ``{OPENCODE_GOOGLE_BASE_URL_ENV}`` set,
    the wrapper instead points OpenCode's built-in ``google`` provider at the
    gateway's Gemini pass-through and gives it the gateway key, so OpenCode's
    own Gemini client (thought signatures, thinking config) talks to LiteLLM.
    """
    real = f"{_BENCHFLOW_BIN_PREFIX}/{binary}"
    agent_bin = f"{_BENCHFLOW_JS_AGENT_PREFIX}/bin/{binary}"
    target = f"{_BENCHFLOW_BIN_PREFIX}/{binary}-proxy"
    provider_id = OPENCODE_PROXY_PROVIDER_ID
    # ``config_relpath`` is resolved against ``$BENCHFLOW_AGENT_HOME`` at launch
    # time — the same home the agent's ``disallow_web_tools_setup_cmd`` and
    # ``credential_files`` write to — so all writers target one config file even
    # when the sandbox home differs from ``$HOME``. Falls back to ``~``.
    config_relpath = config_path.lstrip("/")
    register_js = "\n".join(
        [
            'const fs = require("fs");',
            'const os = require("os");',
            'const path = require("path");',
            'const alias = (process.env.BENCHFLOW_LITELLM_MODEL_ALIAS || "").trim();',
            f'const googleBase = (process.env.{OPENCODE_GOOGLE_BASE_URL_ENV} || "").trim();',
            "if (alias || googleBase) {",
            '  const home = (process.env.BENCHFLOW_AGENT_HOME || "").trim() || os.homedir();',
            f"  const p = path.join(home, {config_relpath!r});",
            "  fs.mkdirSync(path.dirname(p), { recursive: true });",
            '  const text = fs.existsSync(p) ? fs.readFileSync(p, "utf8").trim() : "";',
            "  const d = text ? JSON.parse(text) : {};",
            "  const providers = d.provider ||= {};",
            "  if (alias) {",
            # Dedicated provider id (see docstring) → chat completions, not the
            # Responses API the built-in ``openai`` id is hard-coded to.
            f"    const prov = providers[{provider_id!r}] ||= {{}};",
            '    prov.npm = "@ai-sdk/openai-compatible";',
            '    prov.name ||= "BenchFlow Gateway";',
            "    const opts = prov.options ||= {};",
            '    const base = (process.env.OPENAI_BASE_URL || "").trim();',
            "    if (base) opts.baseURL = base;",
            '    const key = (process.env.OPENAI_API_KEY || "").trim();',
            "    if (key) opts.apiKey = key;",
            "    const models = prov.models ||= {};",
            "    models[alias] ||= { name: alias };",
            '    const catalog = JSON.parse(process.env.BENCHFLOW_PROVIDER_MODELS || "[]");',
            "    const metadata = Array.isArray(catalog) && catalog.find(m => m.id === alias);",
            "    if (metadata && Array.isArray(metadata.input)) {",
            '      const input = metadata.input.filter(x => ["text", "image", "audio", "video", "pdf"].includes(x));',
            '      models[alias].modalities = {input, output: ["text"]};',
            '      models[alias].attachment = input.includes("image");',
            "    }",
            f'    d.small_model = "{provider_id}/" + alias;',
            "  } else {",
            # Gemini: the built-in google provider, pointed at the gateway's
            # Gemini pass-through with the gateway key (see docstring).
            "    const opts = (providers.google ||= {}).options ||= {};",
            "    opts.baseURL = googleBase;",
            '    const key = (process.env.GOOGLE_GENERATIVE_AI_API_KEY || "").trim();',
            "    if (key) opts.apiKey = key;",
            "  }",
            '  fs.writeFileSync(p, JSON.stringify(d, null, 2) + "\\n", {mode: 0o600});',
            "}",
        ]
    )
    wrapper = (
        "#!/bin/sh\n"
        # Proxy mode only. Fail LOUD: if registration errors (malformed existing
        # config, unwritable path), do NOT launch — the agent would otherwise get
        # set_model "<provider>/<alias>" for a model now missing from its config
        # and hit ProviderModelNotFoundError with nothing explaining why. A hard
        # exit surfaces the cause instead of a silent broken proxy path.
        'if [ -n "$BENCHFLOW_LITELLM_MODEL_ALIAS" ] || '
        f'[ -n "${OPENCODE_GOOGLE_BASE_URL_ENV}" ]; then\n'
        f"  if ! {_BENCHFLOW_NODE_PREFIX}/bin/node - <<'JSEOF'\n"
        f"{register_js}\n"
        "JSEOF\n"
        "  then\n"
        f'    echo "benchflow {binary}-proxy: gateway alias registration failed; '
        'refusing to launch in proxy mode" >&2\n'
        "    exit 1\n"
        "  fi\n"
        "fi\n"
        # Exec the agent. A node-shim bin (shebang) goes through the isolated node
        # launcher; a native binary (e.g. opencode-ai 1.17.x ships its bin as a
        # native ELF, bin/opencode.exe) must run DIRECTLY — running it via `node`
        # parses the ELF as JS and crashes at startup with a SyntaxError.
        f'if [ "$(head -c2 {agent_bin} 2>/dev/null)" = "#!" ]; then\n'
        f'  exec {real} "$@"\n'
        "else\n"
        f'  PATH="{_BENCHFLOW_NODE_PREFIX}/bin:$PATH" exec {agent_bin} "$@"\n'
        "fi\n"
    )
    b64 = base64.b64encode(visible_script(wrapper).encode()).decode()
    return f"printf '%s' '{b64}' | base64 -d > {target} && chmod +x {target}"


@dataclass
class CredentialFile:
    """A file to write inside the container before agent launch."""

    path: str  # Target path in container (may use {home} placeholder)
    env_source: str  # Env var to read value from
    template: str = ""  # Template with {value} placeholder. Empty = raw value.
    mkdir: bool = True  # Create parent directory


@dataclass
class HostAuthFile:
    """A single file to copy from the host into the container."""

    host_path: str  # Path on host, e.g. "~/.claude/.credentials.json"
    container_path: str  # Destination in container (may use {home} placeholder)


@dataclass
class SubscriptionAuth:
    """Host CLI login credentials that can substitute for an API key.

    When the user has logged in via the agent CLI (e.g. ``claude login``),
    BenchFlow detects the auth files on the host, copies them into the
    container, and skips the API key requirement.

    ``detect_file`` is checked to determine if the user is logged in.
    All ``files`` are copied into the container when subscription auth is used.
    """

    replaces_env: str  # The env var this substitutes, e.g. "ANTHROPIC_API_KEY"
    detect_file: str  # Host path to check for login, e.g. "~/.claude/.credentials.json"
    files: list[HostAuthFile] = field(default_factory=list)  # All files to copy


# Who carries an agent's model calls (AgentConfig.model_route). "proxy": the
# LiteLLM proxy BenchFlow starts for the trial, which meters usage and records
# the model trajectory. "agent": the agent's own endpoint.
MODEL_ROUTE_PROXY = "proxy"
MODEL_ROUTE_AGENT = "agent"
MODEL_ROUTES = (MODEL_ROUTE_PROXY, MODEL_ROUTE_AGENT)


@dataclass
class AgentConfig:
    """Configuration for a supported agent."""

    name: str
    install_cmd: str
    launch_cmd: str
    requirements_files: tuple[RequirementsFile, ...] = field(default=(), kw_only=True)
    protocol: str = "acp"  # "acp" or "session-factory"
    session_factory: str = ""
    # Non-ACP only. When protocol == "session-factory", this is a
    # "module:callable" entrypoint that builds an Agent Protocol object.
    requires_env: list[str] = field(default_factory=list)
    description: str = ""
    skill_paths: list[str] = field(default_factory=list)
    install_timeout: int = 900  # seconds
    default_model: str = ""  # default model ID when --model is omitted
    model_route: str = MODEL_ROUTE_PROXY
    # "agent": the agent calls its own model endpoint with its own declared
    # credentials (manifest contract 1.1). BenchFlow then starts no proxy for
    # it, a model is optional, and a given model is only a label: recorded,
    # and sent over ACP where the agent takes one (agent_routes_own_model).
    api_protocol: str = ""
    # The LLM API protocol the agent natively speaks:
    # "anthropic-messages" | "openai-completions" | "openai-responses" |
    # "" (runtime/native).
    # Used to pick the correct provider endpoint when a provider exposes
    # multiple (e.g. zai has anthropic-messages, openai-responses, and
    # openai-completions).
    env_mapping: dict[str, str] = field(default_factory=dict)
    # Maps BENCHFLOW_PROVIDER_* → agent-native env var names.
    # Applied by SDK after provider resolution.
    credential_files: list[CredentialFile] = field(default_factory=list)
    # Credential files the launch command itself writes ({home}/{workspace}),
    # e.g. codex-acp's $HOME/.codex/auth.json. Recorded on the connection's
    # cleanup ledger before launch, so they are removed with the rest.
    launcher_credential_files: list[str] = field(default_factory=list)
    # Files to write into container before agent launch (e.g. auth.json).
    home_dirs: list[str] = field(default_factory=list)
    # Extra dot-dirs under $HOME to copy to sandbox user (for dirs not
    # derivable from skill_paths or credential_files).
    acp_model_format: str = "bare"
    # How the agent expects ACP model IDs in session/set_model or config options:
    # "bare"           — just the model name (e.g. "claude-sonnet-4-6").
    #                    Default; works for codex-acp and Claude config options.
    # "provider/model" — models.dev convention (e.g. "google/gemini-3.1-pro-preview").
    #                    Required by opencode, which uses Provider.parseModel()
    #                    to split on "/" and treats the first segment as provider ID.
    # "registered-provider/model" — BenchFlow provider prefix plus model ID
    #                    (e.g. "vllm/Qwen/Qwen3.5-35B"). Required by pi-acp,
    #                    whose launcher registers that provider key in models.json.
    subscription_auth: SubscriptionAuth | None = None
    # Host CLI login that can substitute for an API key (e.g. OAuth tokens
    # from `claude login`). Detected automatically; API keys take precedence.
    acp_subagent_transcript: bool = False
    # Only verified direct harnesses may request the legacy same-session child
    # transcript extension; this does not advertise native child sessions.
    acp_typed_failures: bool = False
    # Ask the agent for typed session failures (the AIR ``sessionFailure``
    # extension): a spent subscription then arrives as its own failure kind
    # instead of an opaque -32603. Only for harnesses whose failure records
    # were checked against the pinned version (benchflow.acp.client).
    supports_acp_set_model: bool = True
    # Some ACP agents configure the model through env/config at launch time and
    # do not implement session/set_model (e.g. OpenHands CLI ACP).
    # ACP session config option id used for model selection when an agent
    # exposes model as a session option instead of implementing set_model.
    acp_model_config_id: str = ""
    # ACP session config option id used for reasoning/thinking effort.
    acp_effort_config_id: str = ""
    # Shell snippet run after credentials/subscription auth are written when
    # BenchFlow's no-web policy is active. Uses BENCHFLOW_AGENT_HOME for the
    # target home so settings land in the same home the agent will run from.
    disallow_web_tools_setup_cmd: str = ""
    # Directories under $HOME that disallow_web_tools_setup_cmd may create and
    # that must remain writable by the sandbox user after the root-run setup.
    disallow_web_tools_owned_paths: list[str] = field(default_factory=list)
    # String appended to launch_cmd when BenchFlow's no-web policy is active.
    # Use for agents whose supported toggle is a launch/config override.
    disallow_web_tools_launch_suffix: str = ""
    # Shell snippet that switches off hosted (provider-side) search tools when
    # the denylist egress policy is active; local fetch tools stay on because
    # they go through the egress proxy. Reuses disallow_web_tools_owned_paths.
    disallow_hosted_search_setup_cmd: str = ""
    # String appended to launch_cmd when the denylist egress policy is active.
    disallow_hosted_search_launch_suffix: str = ""
    # How task-declared MCP servers are delivered to the agent:
    # "acp" sends them in session/new; "native-config" writes an agent-specific
    # config file before launch (for agents whose ACP server drops/reformats
    # MCP fields).
    task_mcp_transport: str = "acp"
    # Native-config target path, relative to $HOME unless absolute.
    task_mcp_config_path: str = ""
    # Folders the agent's tool keeps under the sandbox's temp folder
    # (``${TMPDIR:-/tmp}/<name>``), by plain name. The root-run install can
    # leave one root-owned (OpenCode's postinstall runs ``opencode --version``
    # as root, which creates /tmp/opencode), and the tool then fails as the
    # sandbox user (#1183). Before launch BenchFlow creates each one for the
    # sandbox user, or re-owns an existing real folder and its plain files.
    temp_dirs: list[str] = field(default_factory=list)


# Agents that call Google's GenerateContent API themselves (Gemini CLI and its
# successor, the Antigravity CLI) declare it by mapping the provider base URL
# onto this env var. The ACP runtime and the LiteLLM gateway key their
# Gemini-native handling (bare Google model ids, the byte-preserving Gemini
# pass-through route) off this mapping rather than off agent names.
GEMINI_NATIVE_BASE_URL_ENV = "GOOGLE_GEMINI_BASE_URL"


def routes_gemini_natively(agent_cfg: AgentConfig | None) -> bool:
    """True when ``agent_cfg`` talks to the Gemini API directly."""
    if agent_cfg is None:
        return False
    return (
        agent_cfg.env_mapping.get("BENCHFLOW_PROVIDER_BASE_URL")
        == GEMINI_NATIVE_BASE_URL_ENV
    )


_GEMINI_EXCLUDE_WEB_TOOLS_CMD = _json_settings_merge_node(
    "$BENCHFLOW_AGENT_HOME/.gemini/settings.json",
    "d.tools=d.tools||{};d.tools.exclude=d.tools.exclude||[];"
    '["google_web_search","web_fetch"].forEach(function(t){'
    "if(d.tools.exclude.indexOf(t)<0)d.tools.exclude.push(t);})",
    'd.setdefault("tools",{}).setdefault("exclude",[]);'
    '[d["tools"]["exclude"].append(t) for t in '
    '["google_web_search","web_fetch"] '
    'if t not in d["tools"]["exclude"]]',
)


_ANTIGRAVITY_INSTALL_CMD = (
    agy_install_cmd(_apt_install("curl", "ca-certificates"))
    + " && "
    + _install_python_script(
        f"{_BENCHFLOW_BIN_PREFIX}/antigravity-acp-shim", _ANTIGRAVITY_SHIM
    )
)
# agy has no tool-exclusion setting; its documented switch is a PreToolUse
# lifecycle hook answering {"decision": "deny"} (see antigravity_config).
_ANTIGRAVITY_NO_WEB_TOOLS_CMD = _json_settings_merge(
    f"$BENCHFLOW_AGENT_HOME/{ANTIGRAVITY_HOOKS_RELPATH}",
    hooks_deny_mutator(
        "benchflow-no-web",
        ANTIGRAVITY_WEB_TOOL_MATCHERS,
        "BenchFlow policy: web tools are disabled for this run.",
    ),
)
_ANTIGRAVITY_NO_HOSTED_SEARCH_CMD = _json_settings_merge(
    f"$BENCHFLOW_AGENT_HOME/{ANTIGRAVITY_HOOKS_RELPATH}",
    hooks_deny_mutator(
        "benchflow-no-hosted-search",
        ANTIGRAVITY_HOSTED_WEB_TOOL_MATCHERS,
        "BenchFlow policy: hosted web search and fetch are disabled for this run.",
    ),
)

_MUSE_INSTALL_CMD = (
    muse_install_cmd(_apt_install("curl", "ca-certificates"))
    + " && "
    + _install_python_script(f"{_BENCHFLOW_BIN_PREFIX}/muse-acp-shim", _MUSE_SHIM)
)

# Immutable built-in command, before manifest/plugin overrides. Policy checks
# must not confuse a replacement harness with the executable they probed.
# Global overrides on the pinned app-server cover threads codex-acp starts
# without session config (its title generator), preserving image config.toml.
# Remove only a provider file created by the former launcher, never user TOML.
CODEX_ACP_BUILTIN_LAUNCH = (
    "( umask 077; "
    'h="${BENCHFLOW_AGENT_HOME:-$HOME}"; '
    'if [ -n "$OPENAI_API_KEY" ]; then mkdir -p "$h/.codex" && '
    'printf \'{"OPENAI_API_KEY": "%s"}\' "$OPENAI_API_KEY" '
    '> "$h/.codex/auth.json" && chmod 600 "$h/.codex/auth.json"; '
    "fi; "
    'c="${CODEX_HOME:-$h/.codex}"; '
    'if [ "$(head -n 1 "$c/config.toml" 2>/dev/null)" = '
    "'# runtime-codex-home-config' ]; then "
    'rm -f "$c/config.toml" || exit 1; '
    "fi ) && "
    f"export CODEX_PATH={CODEX_APP_SERVER_EXECUTABLE} && exec "
    + _js_agent_launch(
        "codex-acp", "${OPENAI_BASE_URL:+-c openai_base_url=$OPENAI_BASE_URL}"
    )
)


def launcher_guard_env(agent: str, env: dict[str, str]) -> dict[str, str]:
    """Explicit values for every input that can enable a launcher writer.

    Missing overrides must be empty at the process boundary: an image or
    persistent container value must not activate a writer we did not prepare.
    Keep this after credential filtering, which can remove an input too.
    """
    names = {
        "codex-acp": ("OPENAI_API_KEY", "BENCHFLOW_CODEX_HOME_CONFIG"),
        "pi-acp": ("BENCHFLOW_PROVIDER_PROTOCOL", "BENCHFLOW_PROVIDER_BASE_URL"),
        "opencode": ("BENCHFLOW_LITELLM_MODEL_ALIAS", OPENCODE_GOOGLE_BASE_URL_ENV),
        "mimo": ("OPENAI_BASE_URL", "BENCHFLOW_LITELLM_MODEL_ALIAS"),
    }.get(agent.removeprefix("acpx:"), ())
    return {name: env.get(name, "") for name in names}


@dataclass(frozen=True)
class LauncherCredentialPlan:
    env_overrides: dict[str, str]
    paths: tuple[str, ...]


def launcher_credential_plan(
    agent: str,
    paths: list[str],
    env: dict[str, str],
    *,
    codex_acp: bool,
) -> LauncherCredentialPlan:
    """Bind the prepared/ledgered destinations to explicit launch guards.

    ``env`` is already credential-filtered. The caller carries these same
    overrides to launch and records each path before preparing it; cleanup
    consumes that ledger rather than recomputing guards in a later role.
    """
    overrides = launcher_guard_env(agent, env)
    if agent == "codex-acp" and codex_acp:
        from benchflow.agents.codex_config import codex_home_config

        try:
            config = json.loads(env.get("CODEX_CONFIG", "{}"))
        except (TypeError, ValueError):
            config = None
        # Legacy environment field is diagnostic only. The app-server wrapper
        # applies the same routing without writing/claiming the image config.
        overrides["BENCHFLOW_CODEX_HOME_CONFIG"] = codex_home_config(config) or ""
    return LauncherCredentialPlan(
        overrides,
        tuple(active_launcher_credential_files(agent, paths, {**env, **overrides})),
    )


def active_launcher_credential_files(
    agent: str, paths: list[str], env: dict[str, str]
) -> list[str]:
    """Select declared destinations using the launchers' actual write guards.

    A declaration is an inventory, not ownership of an inactive writer's
    file. In particular, MiMo's cwd config can be the task's answer. Native
    uploads are recorded separately, even if they share a launcher path.
    Unknown/custom destinations retain their explicit declaration contract.
    """
    agent = agent.removeprefix("acpx:")
    conditions = {
        "codex-acp": {
            "{home}/.codex/auth.json": bool(env.get("OPENAI_API_KEY")),
        },
        "pi-acp": {
            "{home}/.pi/agent/models.json": (
                env.get("BENCHFLOW_PROVIDER_PROTOCOL") == "openai-completions"
                and bool(env.get("BENCHFLOW_PROVIDER_BASE_URL"))
            ),
        },
        "opencode": {
            "{home}/.config/opencode/opencode.json": bool(
                env.get("BENCHFLOW_LITELLM_MODEL_ALIAS", "").strip()
                or env.get(OPENCODE_GOOGLE_BASE_URL_ENV, "").strip()
            ),
        },
        "mimo": dict.fromkeys(
            ("{home}/.config/mimocode/mimocode.json", "{workspace}/mimocode.json"),
            bool(
                env.get("OPENAI_BASE_URL") and env.get("BENCHFLOW_LITELLM_MODEL_ALIAS")
            ),
        ),
    }
    guards = conditions.get(agent, {})
    return [path for path in paths if guards.get(path, True)]


# Agent registry — all supported agents
AGENTS: dict[str, AgentConfig] = {
    "claude-agent-acp": AgentConfig(
        name="claude-agent-acp",
        acp_subagent_transcript=True,
        # Verified against claude-agent-acp 0.81.2's session-failure-extension.
        acp_typed_failures=True,
        description="Claude Code via ACP (Anthropic's Agent Client Protocol)",
        skill_paths=["$HOME/.claude/skills"],
        home_dirs=[".claude"],
        # The adapter and the Claude Code CLI are pinned separately (see
        # _CLAUDE_CODE_PACKAGE): a model that needs a newer CLI is a CLI bump.
        # Models reject a CLI below their minimum with
        # `claude_code_version_too_old` (HTTP 400) on the first API call
        # (claude-fable-5-1 needs 2.1.251, claude-opus-5-5 2.1.280). The
        # config-option wiring below (set_config_option + the "model"/"effort"
        # ids) was re-verified against 0.81.2 with
        # tests/test_acp_pinned_protocol_guard.py; the ids stay coupled to the
        # adapter pin, so re-run that guard when bumping either pin. runtime.py
        # uses capability-first dispatch for the rest of the family.
        install_cmd=_js_agent_install(
            "claude-agent-acp",
            _CLAUDE_AGENT_ACP_PACKAGE,
            companion=_CLAUDE_CODE_PACKAGE,
            launcher_env=_CLAUDE_AGENT_ACP_ENV,
            verify=_claude_code_version_check(),
        ),
        launch_cmd=_js_agent_launch("claude-agent-acp"),
        protocol="acp",
        requires_env=["ANTHROPIC_API_KEY"],
        api_protocol="anthropic-messages",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "ANTHROPIC_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "ANTHROPIC_AUTH_TOKEN",
            "BENCHFLOW_PROVIDER_MODEL": "ANTHROPIC_MODEL",
        },
        subscription_auth=SubscriptionAuth(
            replaces_env="ANTHROPIC_API_KEY",
            detect_file="~/.claude/.credentials.json",
            files=[
                HostAuthFile(
                    "~/.claude/.credentials.json", "{home}/.claude/.credentials.json"
                ),
            ],
        ),
        # Written with the agent's own Node, so an image without python3
        # runs the claude agent (REPORT B8).
        disallow_web_tools_setup_cmd=_json_settings_merge_node(
            "$BENCHFLOW_AGENT_HOME/.claude/settings.json",
            _js_deny_list("permissions", ("WebSearch", "WebFetch")),
            'd.setdefault("permissions",{}).setdefault("deny",[]);'
            '[d["permissions"]["deny"].append(t) for t in ["WebSearch","WebFetch"] '
            'if t not in d["permissions"]["deny"]]',
        ),
        disallow_web_tools_owned_paths=["$HOME/.claude"],
        disallow_hosted_search_setup_cmd=_json_settings_merge_node(
            "$BENCHFLOW_AGENT_HOME/.claude/settings.json",
            _js_deny_list("permissions", ("WebSearch",)),
            'd.setdefault("permissions",{}).setdefault("deny",[]);'
            '[d["permissions"]["deny"].append(t) for t in ["WebSearch"] '
            'if t not in d["permissions"]["deny"]]',
        ),
        supports_acp_set_model=False,
        acp_model_config_id="model",
        acp_effort_config_id="effort",
    ),
    "pi-acp": AgentConfig(
        name="pi-acp",
        description="Pi agent via ACP",
        skill_paths=["$HOME/.pi/agent/skills", "$HOME/.agents/skills"],
        install_cmd=(
            f"{_js_agent_install('pi', _PI_PACKAGE)} && "
            f"{_js_agent_install('pi-acp', _PI_ACP_PACKAGE)} && "
            # Deploy launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/pi-acp-launcher", _PI_LAUNCHER
            )
        ),
        launch_cmd=_python_shim_launch(f"{_BENCHFLOW_BIN_PREFIX}/pi-acp-launcher"),
        launcher_credential_files=["{home}/.pi/agent/models.json"],
        protocol="acp",
        acp_model_format="registered-provider/model",
        requires_env=[],  # inferred from --model at runtime
        # Pi is multi-protocol: speaks Anthropic natively and OpenAI via
        # models.json.  Empty lets the provider determine the protocol so
        # multi-endpoint providers (e.g. zai) route to the right URL.
        api_protocol="",
        # env_mapping intentionally empty — the launch wrapper handles
        # protocol-dependent translation (env vars for Anthropic,
        # models.json for OpenAI-compatible providers like vLLM).
    ),
    "codex-acp": AgentConfig(
        name="codex-acp",
        description="OpenAI Codex agent via ACP",
        skill_paths=["$HOME/.agents/skills"],
        # Pinned for reproducibility: an unpinned @agentclientprotocol install
        # floats to latest and can silently break agent activation when the ACP
        # protocol changes (claude-agent-acp above hit exactly this — sdk 0.24
        # dropped session/set_model). 1.6.0 still implements session/set_model,
        # but validates its modelId argument as ``model[effort]`` (0.0.45
        # accepted bare names); _codex_session_model_id maps bare ids onto the
        # session's advertised ``model[effort]`` variants, so model selection
        # keeps working across the bump. 1.6.0 also advertises a "model"
        # config option, but that option rejects ``model[effort]`` ids
        # (-32602), so runtime.py keeps codex on session/set_model — verified
        # live 2026-08-19 against gpt-5.6-sol via an Azure provider. 1.13.1
        # keeps both behaviours (session/set_model still parses
        # ``model[effort]``; the "model" option still takes bare ids only),
        # and 2.0.1 leaves session/set_model and its ``model[effort]`` ids
        # unchanged (#1147).
        # The pin decides which models Codex knows: codex resolves a model's
        # tool surface from the catalog bundled with the @openai/codex it
        # depends on, and a model missing from it gets the fallback surface
        # (plain function tools, no code mode / apply_patch / spawn_agent, #1145).
        # 1.6.0 (codex 0.148) lacked gpt-6-astra; 1.13.1 (codex 0.156.1) lacks
        # gpt-6.1-sol; 2.0.1 (@openai/codex ^0.159.1) lists it.
        install_cmd=_codex_acp_install(),
        # Self-write ~/.codex/auth.json from OPENAI_API_KEY in the launcher itself,
        # ONLY when the key is set (so subscription/host-auth mode is untouched),
        # instead of relying on core's credential_files writer. This makes the
        # decoupled manifest self-contained — like mimo/opencode — and is
        # byte-identical to the former credential_files template
        # ({"OPENAI_API_KEY": "<key>"}) and keeps the old 0600 secret mode.
        # `exec` so signals/PID reach codex.
        launch_cmd=CODEX_ACP_BUILTIN_LAUNCH,
        launcher_credential_files=["{home}/.codex/auth.json"],
        protocol="acp",
        requires_env=["OPENAI_API_KEY"],
        api_protocol="openai-responses",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "OPENAI_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "OPENAI_API_KEY",
        },
        subscription_auth=SubscriptionAuth(
            replaces_env="OPENAI_API_KEY",
            detect_file="~/.codex/auth.json",
            files=[
                HostAuthFile("~/.codex/auth.json", "{home}/.codex/auth.json"),
            ],
        ),
        # codex-acp ignores CLI -c flags. codex_config applies the web policy
        # through CODEX_CONFIG before launching the adapter.
    ),
    "gemini": AgentConfig(
        name="gemini",
        # Google folded Gemini CLI into the Antigravity CLI (developer blog,
        # 2026-05-19) and stopped serving AI Pro, Ultra and free users on it on
        # 2026-06-18; only paid API keys and enterprise licences remain.
        description=(
            "Google Gemini CLI via ACP (legacy, API key only: Google folded it "
            "into the Antigravity CLI, --agent antigravity)"
        ),
        skill_paths=["$HOME/.gemini/skills"],
        install_cmd=_js_agent_install("gemini", "@google/gemini-cli@0.42.0"),
        # BenchFlow already isolates the agent inside the task sandbox and
        # deliberately grants tool approval with --yolo. Gemini CLI 0.42 also
        # requires the workspace to be trusted before it will honor that mode;
        # headless ACP runs cannot answer the interactive trust prompt.
        launch_cmd=_js_agent_launch("gemini", "--acp --yolo --skip-trust"),
        protocol="acp",
        # The Gemini CLI reads GEMINI_API_KEY natively. GOOGLE_API_KEY is
        # accepted as an alias: auto_inherit_env mirrors it both ways so users
        # can set either one. Advertise GEMINI_API_KEY here so `agent show`
        # matches what the CLI actually reads (#342).
        requires_env=["GEMINI_API_KEY"],
        # Default to a sane Gemini model so `--agent gemini` works without
        # --model and never cross-wires to a Claude default (#343).
        default_model="gemini-3.5-flash-lite",
        # api_protocol intentionally empty: Gemini speaks Google's native
        # GenerateContent format, which no current PROVIDERS entry exposes as
        # a multi-endpoint option. Set this when a Gemini-compatible provider
        # with multiple endpoints (e.g. OpenRouter) is added.
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": GEMINI_NATIVE_BASE_URL_ENV,
            # Map to the CLI-native var; auto_inherit_env mirrors it to
            # GOOGLE_API_KEY for compatibility with users who set that alias.
            "BENCHFLOW_PROVIDER_API_KEY": "GEMINI_API_KEY",
        },
        # No subscription_auth: a Google sign-in no longer runs Gemini CLI for
        # AI Pro, Ultra or free accounts, and every BenchFlow Gemini run goes
        # through the LiteLLM proxy, which needs GEMINI_API_KEY. Copying the
        # sign-in's OAuth files into the sandbox gave the agent a refresh token
        # the run could not use.
        disallow_web_tools_setup_cmd=_GEMINI_EXCLUDE_WEB_TOOLS_CMD,
        disallow_web_tools_owned_paths=["$HOME/.gemini"],
        # web_fetch tries the hosted urlContext path before any local fetch, so
        # the egress proxy cannot filter it; the denylist excludes both tools.
        disallow_hosted_search_setup_cmd=_GEMINI_EXCLUDE_WEB_TOOLS_CMD,
    ),
    "antigravity": AgentConfig(
        name="antigravity",
        description=(
            "Google Antigravity CLI (agy, the Gemini CLI successor) via "
            "BenchFlow's ACP shim over its headless stream-json mode"
        ),
        # agy discovers skills globally under ~/.gemini/config/skills and per
        # workspace under <workspace>/.agents/skills (the shim launches agy
        # with --add-dir=<cwd>, which is what makes the workspace tree count).
        skill_paths=[
            ANTIGRAVITY_GLOBAL_SKILLS_PATH,
            ANTIGRAVITY_WORKSPACE_SKILLS_PATH,
        ],
        home_dirs=[ANTIGRAVITY_HOME_DIR],
        # Pinned native binary (see antigravity_config.AGY_RELEASES) plus the
        # Python shim; no Node.js is involved. The shim writes
        # ~/.gemini/antigravity-cli/settings.json at launch (Gemini API-key
        # mode when GEMINI_API_KEY is present, telemetry off, agy's own
        # terminal sandbox off) — Google sign-in lives in the OS keyring and
        # cannot be copied into a sandbox, so there is no subscription_auth.
        install_cmd=_ANTIGRAVITY_INSTALL_CMD,
        launch_cmd=_python_shim_launch(f"{_BENCHFLOW_BIN_PREFIX}/antigravity-acp-shim"),
        protocol="acp",
        requires_env=["GEMINI_API_KEY"],
        default_model="gemini-3.8-flash",
        # api_protocol intentionally empty: agy speaks Google's native
        # GenerateContent format, like the gemini agent.
        api_protocol="",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": GEMINI_NATIVE_BASE_URL_ENV,
            "BENCHFLOW_PROVIDER_API_KEY": "GEMINI_API_KEY",
        },
        # The shim implements session/set_model and a "thinking" config
        # option, so model and effort are driven over ACP like every other
        # ACP agent. agy's catalog ids carry the effort (gemini-3.8-flash-high);
        # BenchFlow keeps the bare id and passes --reasoning-effort through the
        # option, defaulting to high (agy requires one for every model).
        acp_effort_config_id="thinking",
        disallow_web_tools_setup_cmd=_ANTIGRAVITY_NO_WEB_TOOLS_CMD,
        disallow_web_tools_owned_paths=["$HOME/.gemini"],
        # search_web is hosted and read_url_content fetches server-side, so the
        # egress proxy never sees either; the denylist mode switches both off.
        disallow_hosted_search_setup_cmd=_ANTIGRAVITY_NO_HOSTED_SEARCH_CMD,
        # agy loads MCP servers from mcp_config.json, not from ACP session/new.
        task_mcp_transport="native-config",
        task_mcp_config_path=ANTIGRAVITY_MCP_CONFIG_RELPATH,
    ),
    "muse-code": AgentConfig(
        name="muse-code",
        description=(
            "Meta Muse Code CLI via BenchFlow's ACP shim over its headless "
            "`muse exec --json` mode"
        ),
        skill_paths=[MUSE_SKILLS_PATH],
        # Pinned native binary (see muse_config.MUSE_RELEASES) plus the Python
        # shim; no Node.js is involved.
        install_cmd=_MUSE_INSTALL_CMD,
        launch_cmd=_python_shim_launch(f"{_BENCHFLOW_BIN_PREFIX}/muse-acp-shim"),
        protocol="acp",
        # A subscription login, MUSE_AUTH_JSON (written to ~/.config/muse/
        # auth.json below), runs natively: Meta's subscription works only
        # through the CLI's own login. resolve_agent_env refuses a run without
        # one; a META_API_KEY run has no proxy route. The shim also withholds
        # META_API_KEY from Muse, which would otherwise prefer the key.
        requires_env=["MUSE_AUTH_JSON"],
        credential_files=[
            CredentialFile(
                path=f"{{home}}/{MUSE_AUTH_RELPATH}", env_source=MUSE_AUTH_JSON_ENV
            ),
        ],
        default_model="muse-spark-1.3",
        # Muse talks to Meta's API itself; no BenchFlow provider sits between.
        api_protocol="",
        # The shim implements session/set_model and an "effort" config option.
        acp_effort_config_id="effort",
        # Both switches become muse flags in the shim: --disable-web-tools, or
        # tools.web_search.mode="off" (Meta-side search, out of the egress
        # proxy's reach; web_fetch fetches from the sandbox through the proxy).
        disallow_web_tools_launch_suffix=" --disable-web-tools",
        disallow_hosted_search_launch_suffix=" --disable-hosted-search",
    ),
    "opencode": AgentConfig(
        name="opencode",
        description="OpenCode via ACP — open-source coding agent (TypeScript)",
        skill_paths=["$HOME/.opencode/skills"],
        # OpenCode reads its config from ~/.config/opencode; declaring that
        # subdir copies an image-provided config without all of ~/.config.
        home_dirs=[".opencode", ".config/opencode"],
        install_cmd=(
            _js_agent_install("opencode", "opencode-ai@1.19.5")
            + " && "
            + _opencode_family_proxy_wrapper_install(
                "opencode", ".config/opencode/opencode.json"
            )
        ),
        launch_cmd=f"{_BENCHFLOW_BIN_PREFIX}/opencode-proxy acp",
        launcher_credential_files=["{home}/.config/opencode/opencode.json"],
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime
        # The install's `opencode --version` (postinstall, as root) creates
        # $TMPDIR/opencode, which OpenCode writes as the sandbox user (#1183).
        temp_dirs=["opencode"],
        acp_model_format="provider/model",
        # OpenCode adds an "effort" config option once a reasoning model is
        # selected (its model variants; gemini-4-argon: low/medium/high) and
        # defaults it to the lowest, which then overrides any thinking level
        # in its config (1.18.11, checked 2026-10-14; 1.19.5 also offers a
        # "default" value and builds the variants itself, checked 2026-11-04).
        # The model is set first, so the option is there when the effort is set.
        acp_effort_config_id="effort",
        # OpenCode uses models.dev provider IDs — its parseModel() splits
        # modelId on "/" so set_model must send "google/gemini-3.1-pro-preview",
        # not just "gemini-3.1-pro-preview".
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "OPENAI_BASE_URL",
        },
        disallow_web_tools_setup_cmd=_json_settings_merge_node(
            "$BENCHFLOW_AGENT_HOME/.config/opencode/opencode.json",
            "d.tools=d.tools||{};d.tools.webfetch=false",
            'd.setdefault("tools",{})["webfetch"]=False',
        ),
        disallow_web_tools_owned_paths=["$HOME/.config/opencode"],
        disallow_hosted_search_setup_cmd=_json_settings_merge_node(
            "$BENCHFLOW_AGENT_HOME/.config/opencode/opencode.json",
            "d.tools=d.tools||{};d.tools.websearch=false",
            'd.setdefault("tools",{})["websearch"]=False',
        ),
    ),
    "mimo": AgentConfig(
        name="mimo",
        description=(
            "MiMo Code (Xiaomi, OpenCode fork) native `mimo acp` ACP server via "
            "the `mimo` CLI — no server.mjs (mimo is the ACP server)"
        ),
        install_cmd=_MIMO_MANIFEST_INSTALL_CMD,
        launch_cmd=_MIMO_MANIFEST_LAUNCH_CMD,
        launcher_credential_files=[
            "{home}/.config/mimocode/mimocode.json",
            "{workspace}/mimocode.json",
        ],
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime
        # MiMo Code ships a fixed endpoint for its native models.dev "xiaomi"
        # provider, so token-plan/regional keys (XIAOMI_BASE_URL) need a config
        # override. Written only when XIAOMI_API_KEY is present; the file holds
        # {env:...} references (resolved by the CLI at runtime), never the key
        # itself. The no-web-tools setup_cmd merges into this same file.
        credential_files=[
            CredentialFile(
                path="{home}/.config/mimocode/mimocode.json",
                env_source="XIAOMI_API_KEY",
                template=(
                    '{{"provider": {{"xiaomi": {{"options": '
                    '{{"baseURL": "{{env:XIAOMI_BASE_URL}}", '
                    '"apiKey": "{{env:XIAOMI_API_KEY}}"}}}}}}}}'
                ),
            ),
        ],
        install_timeout=1200,
        api_protocol="openai-completions",
        acp_model_format="provider/model",
        # MiMo Code is an OpenCode fork: `mimo acp` reports agentInfo.name="OpenCode"
        # and uses models.dev "provider/model" ids, so set_model must send e.g.
        # "runtime/runtime-<alias>" in proxy mode (the dedicated
        # OPENCODE_PROXY_PROVIDER_ID chat-completions provider the -proxy wrapper
        # registers — NOT the built-in "openai" id, whose Responses-API hard-coding
        # the gateway cannot serve), or "xiaomi/mimo-v2.5" in non-proxy mode via the
        # registered xiaomi provider (the ("mimo","xiaomi") models.dev heuristic in
        # acp/runtime.py keeps that prefix intact).
        env_mapping={
            # Map BOTH base_url and api_key (codex-acp precedent) so the non-proxy
            # path wires the key without an `if agent == "mimo"` core edit.
            "BENCHFLOW_PROVIDER_BASE_URL": "OPENAI_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "OPENAI_API_KEY",
        },
        disallow_web_tools_setup_cmd=_json_settings_merge_node(
            "$BENCHFLOW_AGENT_HOME/.config/mimocode/mimocode.json",
            "d.tools=d.tools||{};d.tools.webfetch=false",
            'd.setdefault("tools",{})["webfetch"]=False',
        ),
        disallow_web_tools_owned_paths=["$HOME/.config/mimocode"],
        disallow_hosted_search_setup_cmd=_json_settings_merge_node(
            "$BENCHFLOW_AGENT_HOME/.config/mimocode/mimocode.json",
            "d.tools=d.tools||{};d.tools.websearch=false",
            'd.setdefault("tools",{})["websearch"]=False',
        ),
    ),
    "harvey-lab-harness": AgentConfig(
        name="harvey-lab-harness",
        description="Harvey LAB harness — runs the original Harvey LAB agent loop "
        "(6 tools: bash, read, write, edit, glob, grep) via ACP shim",
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive && "
            "( command -v git >/dev/null 2>&1 && command -v curl >/dev/null 2>&1 || "
            f"  {_apt_install('git', 'curl', 'ca-certificates')} ) && "
            # The pinned uv, checked against its published sha256
            # (benchflow._uv_pin), as for deepagents and openhands.
            f"{_agent_uv_setup()} && "
            # Harvey LAB at a pinned commit (see _python_agent_pins), fetched
            # alone and checked out sparsely: the shim needs harness/ (about
            # 0.3 MB), while a full clone of the tree is about 3.2 GB of tasks.
            # An existing checkout at the pinned commit is reused; one at any
            # other commit is moved to it. A git without --filter or
            # sparse-checkout falls back to fetching the whole commit.
            f'( [ "$({_AGENT_GIT} -C {_HARVEY_LAB_ROOT} rev-parse HEAD 2>/dev/null)" = '
            f'"{HARVEY_LAB_COMMIT}" ] || ( '
            f"  ( [ -d {_HARVEY_LAB_ROOT}/.git ] || {_AGENT_GIT} init -q {_HARVEY_LAB_ROOT} ) && "
            f"  ( {_AGENT_GIT} -C {_HARVEY_LAB_ROOT} sparse-checkout set "
            f"{' '.join(HARVEY_LAB_SPARSE_PATHS)} >/dev/null 2>&1 || true ) && "
            f"  ( {_AGENT_GIT} -C {_HARVEY_LAB_ROOT} fetch -q --depth 1 --filter=blob:none "
            f"{HARVEY_LAB_REPO} {HARVEY_LAB_COMMIT} || "
            f"    {_AGENT_GIT} -C {_HARVEY_LAB_ROOT} fetch -q --depth 1 "
            f"{HARVEY_LAB_REPO} {HARVEY_LAB_COMMIT} ) && "
            f"  {_AGENT_GIT} -C {_HARVEY_LAB_ROOT} -c advice.detachedHead=false "
            f"checkout -q --force {HARVEY_LAB_COMMIT} ) ) && "
            # Recreate the dedicated venv on the exact managed Python patch:
            # installed metadata alone does not prove the lock's hashes.
            f"uv venv -q --clear --python {PYTHON_VERSION} {_HARVEY_LAB_VENV} && "
            f"{uv_lock_file_command(_HARVEY_LAB_VENV)} && "
            + _HARVEY_LAB_LOCK.command
            + " && "
            f"uv pip install -q --require-hashes --python {_HARVEY_LAB_VENV}/bin/python "
            f"-r {_HARVEY_LAB_REQUIREMENTS_PATH} && "
            # uv's lock files, writable by everyone, lose that (DET-R1).
            f"{uv_lock_seal_command(_HARVEY_LAB_VENV)} && "
            f"chmod -R a+rX {PYTHON_INSTALL_DIR} {_HARVEY_LAB_VENV} "
            f"{_HARVEY_LAB_ROOT} && "
            # Deploy ACP shim
            + _install_python_script(
                f"{_AGENT_UV_BIN}/harvey-lab-acp-shim",
                _HARVEY_LAB_SHIM,
                python=f"{_HARVEY_LAB_VENV}/bin/python",
            )
            # Fail the install, not the first prompt, when the pinned harness
            # or packages are missing: the shim imports these lazily.
            + f" && test -f {_HARVEY_LAB_ROOT}/harness/tools.py && "
            f"cd / && {_HARVEY_LAB_VENV}/bin/python -I -c "
            "'import anthropic, openai, google.genai, docx, pdfplumber, "
            "openpyxl, pptx, markitdown, pandas' >/dev/null"
        ),
        requirements_files=(_HARVEY_LAB_LOCK,),
        launch_cmd=f"HARVEY_LABS_ROOT={_HARVEY_LAB_ROOT} {_HARVEY_LAB_VENV}/bin/python -I {_AGENT_UV_BIN}/harvey-lab-acp-shim",
        protocol="acp",
        requires_env=[],  # inferred from model at runtime (ANTHROPIC_API_KEY, etc.)
        # Harvey LAB's Anthropic/Gemini adapters read ANTHROPIC_API_KEY /
        # GOOGLE_API_KEY directly (auto_inherit_env propagates a host key). But
        # the OpenAI-compatible adapter — the one that serves every proxied
        # provider (deepseek, the azure gpt-5.4-mini gateway, the benchflow-*
        # aliases) — reads OPENAI_BASE_URL/OPENAI_API_KEY, which only exist as
        # BENCHFLOW_PROVIDER_* on the proxy path. Without this mapping the
        # adapter had no base URL/key, hit api.openai.com unauthenticated, and
        # the harness loop ended on turn 0 with zero LLM activity
        # (suspected_api_error). Mapping them points the adapter at the gateway;
        # a direct ANTHROPIC/GEMINI run is unaffected (those adapters ignore
        # OPENAI_*).
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "OPENAI_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "OPENAI_API_KEY",
        },
    ),
    "deepagents": AgentConfig(
        name="deepagents",
        description="deepagents harness — runs LangChain's create_deep_agent loop "
        "(planning, sub-agents, filesystem + shell tools) via ACP shim, driving "
        "deepseek-v4-pro through the OpenAI-compatible provider",
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive && "
            # deepagents requires Python >=3.11, but task base images ship as low
            # as 3.6/3.8 (ubuntu:20.04, cached CI images), so a system-python venv
            # makes pip report "No matching distribution found for deepagents".
            # Provision a pinned interpreter with uv (same pattern as the OpenHands
            # install) so this works regardless of the base-image Python.
            "( command -v curl >/dev/null 2>&1 || "
            f"  {_apt_install('curl', 'ca-certificates')} ) && "
            # The pinned uv, its tarball checked against the published sha256
            # (benchflow._uv_pin), never the image's own uv or the latest one.
            f"{_agent_uv_setup()} && "
            # Recreate the dedicated venv before decoding and checking hashes;
            # a same-version package already present is not trusted provenance.
            f"uv venv -q --clear --python {PYTHON_VERSION} {_DEEPAGENTS_VENV} && "
            f"{uv_lock_file_command(_DEEPAGENTS_VENV)} && "
            # deepagents pulls in langchain/langchain-core/langchain-anthropic/
            # langchain-google-genai; langchain-openai is NOT a deepagents dep but
            # is required for the OpenAI-compatible deepseek-v4-pro chat model.
            # Both, and everything they pull in, are exact pins
            # (_python_agent_pins), so a new upstream release cannot change the
            # agent between two runs of the same BenchFlow.
            + _DEEPAGENTS_LOCK.command
            + " && "
            f"uv pip install -q --require-hashes --python {_DEEPAGENTS_VENV}/bin/python "
            f"-r {_DEEPAGENTS_REQUIREMENTS_PATH} && "
            # uv's lock files, writable by everyone, lose that (DET-R1).
            f"{uv_lock_seal_command(_DEEPAGENTS_VENV)} && "
            # Let the sandbox user traverse + execute the venv interpreter and the
            # uv-managed CPython it links to.
            f"chmod -R a+rX {_DEEPAGENTS_VENV} {PYTHON_INSTALL_DIR} && "
            # Deploy ACP shim
            + _install_python_script(
                f"{_AGENT_UV_BIN}/deepagents-acp-shim",
                _DEEPAGENTS_SHIM,
                python=f"{_DEEPAGENTS_VENV}/bin/python",
            )
            # Check the names the shim imports, only after the pinned install
            # succeeded. A pinned release that moves a name fails at setup.
            + f" && cd / && {_DEEPAGENTS_VENV}/bin/python -I -c "
            "'from deepagents import create_deep_agent; "
            "from deepagents.backends import LocalShellBackend; "
            "from langchain_openai import ChatOpenAI' >/dev/null"
        ),
        requirements_files=(_DEEPAGENTS_LOCK,),
        launch_cmd=(
            f"{_DEEPAGENTS_VENV}/bin/python -I {_AGENT_UV_BIN}/deepagents-acp-shim"
        ),
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime (DEEPSEEK_API_KEY, etc.)
        # api_protocol intentionally empty — the shim builds an OpenAI-compatible
        # ChatOpenAI from BENCHFLOW_PROVIDER_BASE_URL/API_KEY (with DEEPSEEK_*
        # fallback). Leaving it empty avoids pinning provider endpoint selection
        # so any OpenAI-compatible provider for the requested model works.
        # env_mapping intentionally empty — the shim reads BENCHFLOW_PROVIDER_*
        # directly (set unconditionally by resolve_provider_env when the model
        # carries a registered provider prefix), with DEEPSEEK_* as a fallback
        # (auto_inherit_env propagates those).
    ),
    "openhands": AgentConfig(
        name="openhands",
        launcher_credential_files=["{home}/.openhands/agent_settings.json"],
        description="OpenHands agent via ACP (multi-model, Python-based)",
        skill_paths=["$HOME/.agents/skills", "$WORKSPACE/.agents/skills"],
        home_dirs=[".openhands"],
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive && "
            # Root's install keeps the route's PATH: no $HOME/.local/bin, as
            # HOME can be the workspace and uv runs the first git on PATH as
            # root (FF-install-C15, PY8-PT-R8-3). _agent_uv_setup puts the
            # tool and its entry point in BenchFlow's protected tree.
            "( command -v curl >/dev/null 2>&1 && command -v git >/dev/null 2>&1 || "
            "  if command -v apt-get >/dev/null 2>&1; then "
            f"    {_apt_install('curl', 'ca-certificates', 'git')}; "
            "  elif command -v dnf >/dev/null 2>&1; then "
            "    dnf -y --allowerasing install curl ca-certificates git >/dev/null 2>&1; "
            "  elif command -v apk >/dev/null 2>&1; then "
            "    apk add --no-cache curl ca-certificates git >/dev/null 2>&1; "
            "  else "
            "    echo 'OpenHands GitHub install requires curl and git' >&2; "
            "    exit 127; "
            "  fi ) && "
            # The pinned uv, its tarball checked against the published sha256
            # (benchflow._uv_pin), never the image's own uv or the latest one.
            f"( {_agent_uv_setup()} && "
            # Pin the OpenHands CLI source and its matching SDK/tools release
            # together so the ACP runtime cannot drift or be forced onto an
            # older, API-incompatible SDK. The 1.28.1 line retains the
            # security_risk default fix and includes later long-run ACP,
            # terminal, tool-executor, and event-log stability fixes.
            f"printf 'openhands-sdk=={_OPENHANDS_SDK_VERSION}\\n"
            f"openhands-tools=={_OPENHANDS_TOOLS_VERSION}\\n' "
            f"> {_AGENT_UV_ROOT}/oh-sdk-overrides.txt && "
            f"{_AGENT_GIT_HOME} uv tool install --force "
            f"--overrides {_AGENT_UV_ROOT}/oh-sdk-overrides.txt "
            "--from "
            f"'git+https://github.com/OpenHands/OpenHands-CLI.git@{_OPENHANDS_CLI_GIT_REV}' "
            f"openhands --python {PYTHON_VERSION} && "
            "  uv tool list | grep -q '^openhands\\b' && "
            # uv's lock files (the tool folder's, each tool venv's, the
            # cache's), writable by everyone, lose that (DET-R1).
            f"  {uv_lock_seal_command(_OPENHANDS_TOOL_DIR)} ) && "
            # Let sandbox user traverse to the tool venv and read the
            # uv-managed Python it links to. In the && chain: a failed
            # pinned-uv fetch or check must not fall through to the checks
            # below on an image that already has openhands.
            f"chmod -R a+rX {PYTHON_INSTALL_DIR} {_OPENHANDS_TOOL_DIR} && "
            # Seed config so OpenHands ACP auth check passes before env override.
            "mkdir -p ~/.openhands && "
            'echo \'{"llm":{"model":"placeholder","api_key":"placeholder"}}\' '
            "> ~/.openhands/agent_settings.json && "
            f"test -x {_BENCHFLOW_BIN_PREFIX}/openhands"
        ),
        launch_cmd=(
            f'export PATH="{_BENCHFLOW_BIN_PREFIX}:$HOME/.local/bin:$PATH" && '
            "( umask 077; mkdir -p ~/.openhands && "
            # Write llm settings including base_url so the BenchFlow LiteLLM
            # gateway (LLM_BASE_URL) is honored. OpenHands' --override-with-envs
            # does not reliably apply base_url; it is omitted when unset.
            '{ printf \'{"llm":{"model":"%s","api_key":"%s"\' '
            '"$LLM_MODEL" "$LLM_API_KEY"; '
            'if [ -n "$LLM_BASE_URL" ]; then '
            'printf \',"base_url":"%s"\' "$LLM_BASE_URL"; fi; '
            'if [ -n "$LLM_API_VERSION" ]; then '
            'printf \',"api_version":"%s"\' "$LLM_API_VERSION"; fi; '
            'if [ -n "$LLM_TIMEOUT" ]; then '
            'case "$LLM_TIMEOUT" in *[!0-9]*) '
            'echo "LLM_TIMEOUT must be a non-negative integer" >&2; exit 2;; esac; '
            'printf \',"timeout":%s\' "$LLM_TIMEOUT"; fi; '
            'case "$LLM_REASONING_EFFORT" in '
            'max) printf \',"litellm_extra_body":{"reasoning":{"effort":"max"}}\' ;; '
            "none|low|medium|high|xhigh) "
            'printf \',"reasoning_effort":"%s",'
            '"litellm_extra_body":{"reasoning_effort":"%s"}\' '
            '"$LLM_REASONING_EFFORT" "$LLM_REASONING_EFFORT" ;; '
            '?*) printf \',"litellm_extra_body":{"reasoning_effort":"%s"}\' '
            '"$LLM_REASONING_EFFORT" ;; '
            "esac; "
            "printf '}}'; } > ~/.openhands/agent_settings.json ) && "
            'if [ "${BENCHFLOW_OPENHANDS_DISABLE_SUBAGENTS:-0}" = "1" ]; then '
            'OH_BIN="$(command -v openhands)"; '
            'OH_PY="$(dirname "$(readlink -f "$OH_BIN")")/python"; '
            '[ -x "$OH_PY" ] || { '
            'echo "Cannot locate OpenHands tool interpreter" >&2; exit 127; }; '
            '"$OH_PY" -I -c \'from pathlib import Path; '
            "import openhands_cli.utils as u; "
            "p=Path(u.__file__); s=p.read_text(); "
            'old="        Tool(name=task_tool_name),\\n"; '
            'new="        # BenchFlow: delegation disabled for this run.\\n"; '
            "assert old in s or new in s; "
            "p.write_text(s.replace(old,new,1))'; "
            "fi && "
            "openhands acp --always-approve --override-with-envs"
        ),
        protocol="acp",
        requires_env=["LLM_API_KEY"],
        api_protocol="",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "LLM_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "LLM_API_KEY",
        },
        supports_acp_set_model=False,
        task_mcp_transport="native-config",
        task_mcp_config_path=".openhands/mcp.json",
        disallow_web_tools_setup_cmd=(
            'mkdir -p "$BENCHFLOW_AGENT_HOME/.openhands" && '
            "printf '[agent]\\nenable_browsing = false\\n' "
            '> "$BENCHFLOW_AGENT_HOME/.openhands/config.toml"'
        ),
        disallow_web_tools_owned_paths=["$HOME/.openhands"],
    ),
}


# Derived lookup tables — install/launch commands by agent name.
# Updated by register_agent() when new agents are added at runtime.
AGENT_INSTALLERS: dict[str, str] = {name: a.install_cmd for name, a in AGENTS.items()}
AGENT_LAUNCH: dict[str, str] = {name: a.launch_cmd for name, a in AGENTS.items()}
_BUILTIN_INSTALL_CMDS = dict(AGENT_INSTALLERS)


def has_builtin_install(agent: str) -> bool:
    """Whether the install still uses an unchanged built-in recipe.

    Custom installers may repair the image before its final seal, so the
    early loader refusal cannot predict their effect.
    """
    name = agent.split()[0]
    original = _BUILTIN_INSTALL_CMDS.get(name)
    config = AGENTS.get(name)
    return (
        original is not None
        and config is not None
        and config.install_cmd == original
        and AGENT_INSTALLERS.get(name) == original
    )


# The install split. An agent's ``install_cmd`` is the cacheable
# part: system packages, BenchFlow's private Node and the agent's pinned
# package, all under /opt/runtime, written the same way every trial. What
# stays per trial is in the other fields and runs in Rollout.install_agent
# after it: the sandbox user, credential files, subscription auth, the web
# policy, skills and lockdown. The agents below install exactly the pinned
# version named here, so on Docker their install is built once into a
# cached image layer FROM the task image (sandbox.env_layers) and reused
# while its key matches; the version is part of that key and its record.
# Not here, so they install in every trial: pi-acp (installs `latest`), and
# harvey-lab-harness, deepagents and openhands, whose Python packages are not
# all pinned on this line. A manifest or plugin that replaces one of these
# agents' install command installs in every trial too (cacheable_install),
# unless the plugin registers its own pinned install with
# ``register_agent(cacheable_version=...)``.
CACHEABLE_INSTALLS: dict[str, str] = {
    "claude-agent-acp": (
        f"{pinned_npm_package('claude-agent-acp')[1]}"
        f"+claude-code@{pinned_npm_package('claude-code')[1]}"
    ),
    "codex-acp": pinned_npm_package("codex-acp")[1],
    "gemini": "0.42.0",
    "opencode": "1.19.5",
    "mimo": "0.1.15",
    "antigravity": AGY_VERSION,
}
# The commands those versions describe: the built-in ones before any
# override, and a plugin's own pinned install (register_agent).
_CACHEABLE_INSTALL_CMDS: dict[str, str] = {
    name: AGENTS[name].install_cmd for name in CACHEABLE_INSTALLS
}
_BUILTIN_CACHEABLE_INSTALLS = dict(CACHEABLE_INSTALLS)
_BUILTIN_CACHEABLE_INSTALL_CMDS = dict(_CACHEABLE_INSTALL_CMDS)


def cacheable_install(agent: str) -> tuple[str, str] | None:
    """``(pinned version, install command)`` of ``agent``'s cacheable install.

    None when the agent's install is not pinned, or when the command that
    would run is no longer the built-in one the version describes (a
    manifest, plugin or acpx wrapper replaced it): such an install runs in
    the trial, as before.
    """
    version = CACHEABLE_INSTALLS.get(agent)
    install_cmd = AGENT_INSTALLERS.get(agent)
    if version is None or install_cmd is None:
        return None
    if install_cmd != _CACHEABLE_INSTALL_CMDS[agent]:
        return None
    return version, install_cmd


def get_sandbox_home_dirs(agents: list[str] | None = None) -> set[str]:
    """Collect user home config/auth dirs BenchFlow may materialize for the sandbox user.

    Derives from three sources across the registered agents named in
    ``agents`` (all registered agents when ``agents`` is None; names that are
    not registered contribute nothing):
    - credential_files: {home}/.foo/... → ".foo"
    - subscription_auth.files: {home}/.foo/... → ".foo"
    - home_dirs: explicit extra home directories

    Skill paths are excluded: deploy_skills() now links those paths directly to a
    shared skills tree instead of relying on sandbox-home copies.
    """
    dirs: set[str] = set()
    if agents is None:
        cfgs = list(AGENTS.values())
    else:
        cfgs = [AGENTS[name] for name in agents if name in AGENTS]
    for cfg in cfgs:
        for cf in cfg.credential_files:
            # path uses {home}/.foo/... placeholder
            path = cf.path
            if path.startswith("{home}/."):
                dirname = path.removeprefix("{home}/").split("/")[0]
                dirs.add(dirname)
        if cfg.subscription_auth:
            for f in cfg.subscription_auth.files:
                if f.container_path.startswith("{home}/."):
                    dirname = f.container_path.removeprefix("{home}/").split("/")[0]
                    dirs.add(dirname)
        dirs.update(cfg.home_dirs)
    return dirs


def is_vertex_model(model: str) -> bool:
    """True if the model uses Vertex AI (GCP ADC auth, not API keys)."""
    from benchflow.agents.providers import find_provider

    result = find_provider(model)
    if result:
        _, cfg = result
        return cfg.auth_type == "adc"
    return False


def infer_env_key_for_model(model: str) -> str | None:
    """Infer the required API key environment variable from a model ID."""
    # Registered providers are authoritative about auth mode.
    from benchflow.agents.providers import find_provider, resolve_auth_env

    result = find_provider(model)
    if result is not None:
        _, cfg = result
        if cfg.auth_type != "api_key":
            return None
        return resolve_auth_env(model)
    # ADC-based providers and built-in Vertex prefixes
    if is_vertex_model(model):
        return None
    # Fallback heuristics for well-known model names
    m = model.lower()
    if "gemini" in m:
        return "GEMINI_API_KEY"
    if "gpt" in m or m.startswith("o1") or m.startswith("o3"):
        return "OPENAI_API_KEY"
    if "claude" in m or "haiku" in m or "sonnet" in m or "opus" in m:
        return "ANTHROPIC_API_KEY"
    return None


AGENT_ALIASES: dict[str, str] = {
    "claude": "claude-agent-acp",
    "codex": "codex-acp",
    "gemini": "gemini",
    "agy": "antigravity",
    "muse": "muse-code",
    "pi": "pi-acp",
    "openhands": "openhands",
    "oh": "openhands",
    "harvey-lab": "harvey-lab-harness",
    "deepagents": "deepagents",
}

VALID_PROTOCOLS = {"acp", "acpx", "session-factory"}


_CORE_AGENT_CONFIGS = dict(AGENTS)
_REGISTRY_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# The ``acpx:`` runtime-key namespace
# ---------------------------------------------------------------------------
#
# An ``acpx/<agent>`` spec resolves to an acpx-wrapped AgentConfig whose
# install/launch commands route through the acpx CLI. That wrapped config is
# registered into ``AGENTS`` (and the installer/launch maps) under a stable
# runtime key prefixed with ``ACPX_KEY_PREFIX`` so later *name-keyed* lookups
# in the Rollout/Evaluation path pick up the wrapped commands.
#
# This namespace is owned end to end by ``resolve_agent_key``: it is the only
# function that *mints* an ``acpx:`` key (by registering the wrapped config).
# Two other sites must agree with that convention and are documented here so
# the contract is explicit rather than implied:
#
#   - ``_acpx_wrap`` produces the wrapped config whose ``name`` carries the
#     ``acpx:`` prefix (via ``acpx_runtime_key``).
#   - ``resolve_agent`` round-trips an already-registered ``acpx:`` key:
#     ``parse_agent_spec`` leaves it whole under the default ``acp`` protocol,
#     and the ``protocol == "acp" and name in AGENTS`` branch returns it as-is.
#
# Changing the prefix or this round-trip behavior requires updating all three.
ACPX_KEY_PREFIX = "acpx:"


def acpx_runtime_key(canonical_name: str) -> str:
    """Return the stable ``acpx:`` runtime key for a canonical agent name.

    Single source of truth for the ``acpx:`` namespace — see the module-level
    comment above. ``resolve_agent_key`` registers the wrapped config under
    this key; ``resolve_agent`` round-trips it back to that config.
    """
    return f"{ACPX_KEY_PREFIX}{canonical_name}"


def parse_agent_spec(spec: str) -> tuple[str, str]:
    """Parse protocol/name and resolve aliases."""
    if is_operator_script(spec):
        return "acp", spec
    if "/" in spec:
        protocol, name = spec.split("/", 1)
    else:
        protocol, name = "acp", spec
    return protocol, AGENT_ALIASES.get(name, name)


def is_explicit_raw_agent_command(spec: str) -> bool:
    """Return whether *spec* explicitly uses supported raw-command syntax."""
    value = spec.strip()
    return bool(value) and (
        any(char.isspace() for char in value)
        or value.startswith(("/", "./", "../", "~/"))
    )


_ACPX_INSTALL = (
    f"{_NODE_INSTALL} && "
    f'export PATH="{_JS_AGENT_PATH}" && '
    f"( command -v acpx >/dev/null 2>&1 || "
    f"{_BENCHFLOW_NODE_PREFIX}/bin/npm install -g acpx@latest ) "
)


def _acpx_wrap(config: AgentConfig) -> AgentConfig:
    """Wrap an agent config to launch via acpx instead of direct ACP.

    acpx (https://acpx.sh/) is a headless CLI client for ACP that adds
    persistent sessions, crash recovery, and structured NDJSON output.
    The underlying agent's install, env, and credentials are preserved.
    """
    acpx_agent_name = config.name
    for alias, canonical in AGENT_ALIASES.items():
        if canonical == config.name:
            acpx_agent_name = alias
            break

    # The acpx wrapper only overrides name/install_cmd/launch_cmd. Every other
    # AgentConfig field must pass through from the underlying agent so that
    # routing-relevant attributes (api_protocol, default_model, env_mapping,
    # requires_env, credentials, …) survive when the wrapped config is cached
    # into AGENTS and later read by resolve_provider_env. ``protocol`` stays
    # "acp" because acpx itself speaks ACP regardless of the inner agent.
    return AgentConfig(
        # ``acpx:`` runtime key — see acpx_runtime_key / module-level contract.
        name=acpx_runtime_key(config.name),
        install_cmd=f"{config.install_cmd} && {_ACPX_INSTALL}",
        launch_cmd=(
            f'export PATH="{_JS_AGENT_PATH}" && acpx {acpx_agent_name} --approve-all'
        ),
        protocol="acp",
        session_factory=config.session_factory,
        requires_env=config.requires_env,
        description=f"{config.description} (via acpx)",
        skill_paths=config.skill_paths,
        install_timeout=config.install_timeout,
        default_model=config.default_model,
        model_route=config.model_route,
        api_protocol=config.api_protocol,
        env_mapping=config.env_mapping,
        credential_files=config.credential_files,
        launcher_credential_files=config.launcher_credential_files,
        home_dirs=config.home_dirs,
        acp_model_format=config.acp_model_format,
        subscription_auth=config.subscription_auth,
        # acpx forwarding of these extensions has not been verified.
        acp_subagent_transcript=False,
        acp_typed_failures=False,
        supports_acp_set_model=config.supports_acp_set_model,
        acp_model_config_id=config.acp_model_config_id,
        acp_effort_config_id=config.acp_effort_config_id,
        disallow_web_tools_setup_cmd=config.disallow_web_tools_setup_cmd,
        disallow_web_tools_owned_paths=config.disallow_web_tools_owned_paths,
        disallow_web_tools_launch_suffix=config.disallow_web_tools_launch_suffix,
        disallow_hosted_search_setup_cmd=config.disallow_hosted_search_setup_cmd,
        disallow_hosted_search_launch_suffix=config.disallow_hosted_search_launch_suffix,
        task_mcp_transport=config.task_mcp_transport,
        task_mcp_config_path=config.task_mcp_config_path,
        temp_dirs=config.temp_dirs,
    )


# Namespace shorthand — "<ns>:<id>" agent specs (design: #876).
# The namespace names the adaptation path; resolution is a deterministic
# name-candidate mapping onto the existing flat names (the naming law: the
# native/ACP path owns bare names, every other path prefixes "<path>-"). No new
# config fields, no ambiguity machinery — exact registered names and aliases
# always win (checked before this), so "acpx:" runtime keys are unaffected.
# Phase 2 (#876) keys the registry-backed agent auto-fetch on these same
# namespaces (`--agent acp:<id>[@version]`).
def _resolve_namespace_shorthand(name: str) -> AgentConfig | None:
    """Resolve "<ns>:<id>" (e.g. "omnigent:pi", "acp:mimo", "ai-sdk:codex").

    Candidates per namespace: ``acp:<id>`` tries the bare id then ``<id>-acp``
    (so ``acp:mimo`` → ``mimo``, ``acp:pi`` → ``pi-acp``, ``acp:claude`` →
    alias ``claude`` → ``claude-agent-acp``); ``ai-sdk:<id>`` /
    ``omnigent:<id>`` try the ``<ns>-<id>`` prefix form. Every candidate goes
    through AGENT_ALIASES, so alias-only names resolve too. Returns ``None``
    when nothing matches (callers fall through to the unknown-agent error).
    """
    ns, sep, ident = name.partition(":")
    if not sep or not ident:
        return None
    if ns == "acp":
        candidates = [ident, f"{ident}-acp"]
    elif ns in ("ai-sdk", "omnigent"):
        candidates = [f"{ns}-{ident}"]
    else:
        return None
    for candidate in candidates:
        candidate = AGENT_ALIASES.get(candidate, candidate)
        if candidate in AGENTS:
            return AGENTS[candidate]
    return None


# Agents BenchFlow runs itself instead of launching an ACP agent: ``oracle``
# runs the task's own solution (oracle/solve.sh); ``nop`` runs nothing, so the
# verifier scores the untouched workspace (the empty control run that proves a
# verifier does not pass on untouched state). Neither needs a model or
# credentials.
SCRIPTED_AGENTS = frozenset({"oracle", "nop"})


def is_scripted_agent(name: str | None) -> bool:
    """No ACP agent, model or install: controls and operator scripts."""
    return name in SCRIPTED_AGENTS or is_operator_script(name)


def is_operator_script(name: str | None) -> bool:
    """An operator's bash file, not the task's root oracle solution."""
    return isinstance(name, str) and name.startswith("script:")


def agent_routes_own_model(agent: str | None) -> bool:
    """True when ``agent`` declares ``model_route = "agent"``.

    Such an agent calls its own model endpoint from the sandbox: BenchFlow
    starts no proxy for it and cannot meter its usage. The agent is looked up
    by its registered name, an alias or an acpx spec; anything else (a raw
    command, a scripted agent) keeps the proxy route.
    """
    if not agent:
        return False
    config = AGENTS.get(agent)
    if config is None:
        _protocol, name = parse_agent_spec(agent)
        config = AGENTS.get(name) or AGENTS.get(name.removeprefix(ACPX_KEY_PREFIX))
    # Read like the other optional fields: a plugin's or a test's stand-in
    # config without the field keeps the proxy route.
    return getattr(config, "model_route", MODEL_ROUTE_PROXY) == MODEL_ROUTE_AGENT


def routed_model(agent: str | None, model: str | None) -> str | None:
    """The model BenchFlow routes for ``agent``: ``model``, or None for an
    agent that routes its own model calls, whatever label it was given.

    Route and credential decisions read this instead of the label, so such
    an agent takes the no-model DIRECT route: no proxy, no provider
    resolution, and only its own declared credentials.
    """
    return None if agent_routes_own_model(agent) else model


def resolve_agent(spec: str) -> AgentConfig:
    """Resolve an agent spec to an AgentConfig.

    Supports: bare name, alias, protocol/name, acpx/name, and the namespace
    shorthand "<ns>:<id>" for ns in {acp, ai-sdk, omnigent} — e.g.
    "omnigent:pi" → omnigent-pi, "acp:pi" → pi-acp (see #876).
    Raises KeyError with suggestions for unknown agents.
    """
    protocol, name = parse_agent_spec(spec)

    if protocol not in VALID_PROTOCOLS:
        raise KeyError(
            f"Unknown protocol: {protocol!r}. Valid: {', '.join(sorted(VALID_PROTOCOLS))}"
        )

    from benchflow.agents import remote_manifests

    remote_manifests.autoload_local_manifest_agents()
    protocol, name = parse_agent_spec(spec)

    # An already-resolved acpx runtime key (e.g. "acpx:claude-agent-acp")
    # round-trips: parse_agent_spec leaves it whole under the default "acp"
    # protocol and it lives in AGENTS. See the ACPX_KEY_PREFIX contract.
    direct = AGENTS.get(name) if protocol == "acp" else None
    if direct is not None:
        return direct

    if name not in AGENTS:
        shorthand = _resolve_namespace_shorthand(name)
        if shorthand is not None:
            return _acpx_wrap(shorthand) if protocol == "acpx" else shorthand

        # Miss-driven manifest auto-load (#876 Phase 2a): fetch DECLARATIVE
        # manifest agents from the pinned agents source (data only — their
        # install/launch strings run in the sandbox, same trust as task
        # sources) and retry. One-shot per process; local names always win.
        remote_manifests.autoload_remote_manifest_agents()
        registered_name = AGENT_ALIASES.get(name, name)
        config = AGENTS.get(registered_name)
        if config is not None:
            name = registered_name
            return _acpx_wrap(config) if protocol == "acpx" else config
        shorthand = _resolve_namespace_shorthand(name)
        if shorthand is not None:
            return _acpx_wrap(shorthand) if protocol == "acpx" else shorthand

        from difflib import get_close_matches

        # Breadcrumb: if agent plugin packages failed to load at import time
        # (see FAILED_AGENT_PLUGINS / _load_agent_plugin_packages), the missing
        # name is very likely one of theirs — connect the two events here, at
        # the point the user actually sees a failure.
        plugin_hint = ""
        if FAILED_AGENT_PLUGINS:
            failed = ", ".join(sorted(FAILED_AGENT_PLUGINS))
            plugin_hint = (
                f" Note: agent plugin(s) failed to load at startup: {failed} "
                "(see the startup warning for details)."
            )
        if remote_manifests.last_source_description:
            plugin_hint += f" ({remote_manifests.last_source_description} consulted)"
        available_agents = list(AGENTS)
        close = get_close_matches(name, available_agents, n=1, cutoff=0.6)
        if close:
            raise KeyError(
                f"Unknown agent: {name!r}. Did you mean: {close[0]!r}?{plugin_hint}"
            )
        raise KeyError(
            f"Unknown agent: {name!r}. Available: "
            f"{', '.join(sorted(available_agents))}{plugin_hint}"
        )

    config = AGENTS[name]
    if protocol == "acpx":
        return _acpx_wrap(config)
    return config


def session_factory_entrypoint(
    spec: str, *, resolver: Callable[[str], AgentConfig] | None = None
) -> str | None:
    """The connection route shared by admission and rollout dispatch.

    Only a complete session-factory configuration bypasses live transport.
    Unknown/raw commands retain the live route, as rollout has always done.
    A caller with its own resolution boundary supplies it here, so admission
    and dispatch interpret that lookup with the same rules.
    """
    try:
        config = (resolve_agent if resolver is None else resolver)(spec)
    except Exception:
        return None
    if config.protocol == "session-factory" and config.session_factory:
        return config.session_factory
    return None


def resolve_agent_key(spec: str) -> str:
    """Resolve an agent spec to a stable registry key.

    This function owns the ``acpx:`` runtime-key namespace (see the
    ``ACPX_KEY_PREFIX`` module-level contract). For plain ACP agents the key
    is the canonical agent name. For ``acpx/<agent>`` specs the acpx-wrapped
    config (acpx install/launch commands) is registered into
    ``AGENTS``/``AGENT_INSTALLERS``/``AGENT_LAUNCH`` under the stable runtime
    key ``acpx_runtime_key(<canonical>)`` so that name-keyed lookups in the
    Rollout/Evaluation path use the wrapped commands instead of the literal
    spec string. ``resolve_agent`` then round-trips that key back to the
    wrapped config.

    Unknown bare IDs fail closed. Explicit raw commands are returned unchanged.
    Built-in non-agent modes and unsupported protocols pass through for their
    existing downstream validators.
    """
    if spec in {"oracle", "acp/oracle"}:
        return "oracle"
    if is_scripted_agent(spec) or spec == "task-runtime":
        # A built-in scripted runner (oracle, nop) or the task runtime, not a
        # registry agent. Resolving it would miss, fail closed, and fetch the
        # remote agents source on every such run.
        return spec
    protocol, _ = parse_agent_spec(spec)
    if protocol not in VALID_PROTOCOLS:
        return spec
    try:
        config = resolve_agent(spec)
    except KeyError:
        if not is_explicit_raw_agent_command(spec):
            raise
        # The raw-command fallback bypasses resolve_agent's error entirely (the
        # spec is later exec'd verbatim in the sandbox), so surface the failed-
        # plugin breadcrumb HERE too — otherwise a plugin load failure manifests
        # only as a deep, minutes-later "command not found" in the rollout.
        if FAILED_AGENT_PLUGINS:
            import logging

            logging.getLogger(__name__).warning(
                "Agent spec %r is not a registered agent and will be treated as "
                "a raw command; note that agent plugin(s) failed to load at "
                "startup: %s",
                spec,
                ", ".join(sorted(FAILED_AGENT_PLUGINS)),
            )
        return spec
    if config.name not in AGENTS:
        AGENTS[config.name] = config
        AGENT_INSTALLERS[config.name] = config.install_cmd
        AGENT_LAUNCH[config.name] = config.launch_cmd
    return config.name


def get_agent(name: str) -> tuple[AgentConfig, str]:
    """Get agent config by name.

    Returns (config, default_model) where default_model comes from config.default_model.
    Raises KeyError if not found.
    """
    if name not in AGENTS:
        available = ", ".join(sorted(AGENTS.keys()))
        raise KeyError(f"Unknown agent: {name!r}. Available: {available}")
    config = AGENTS[name]
    return config, config.default_model


def list_agents() -> list[AgentConfig]:
    """List all registered agents."""
    return list(AGENTS.values())


def register_agent(
    name: str,
    install_cmd: str,
    launch_cmd: str,
    *,
    protocol: str = "acp",
    session_factory: str = "",
    requires_env: list[str] | None = None,
    description: str = "",
    skill_paths: list[str] | None = None,
    install_timeout: int = 900,
    default_model: str = "",
    model_route: str = MODEL_ROUTE_PROXY,
    api_protocol: str = "",
    env_mapping: dict[str, str] | None = None,
    credential_files: list[CredentialFile] | None = None,
    launcher_credential_files: list[str] | None = None,
    home_dirs: list[str] | None = None,
    subscription_auth: SubscriptionAuth | None = None,
    acp_model_format: str = "bare",
    supports_acp_set_model: bool = True,
    acp_model_config_id: str = "",
    acp_effort_config_id: str = "",
    disallow_web_tools_setup_cmd: str = "",
    disallow_web_tools_owned_paths: list[str] | None = None,
    disallow_web_tools_launch_suffix: str = "",
    disallow_hosted_search_setup_cmd: str = "",
    disallow_hosted_search_launch_suffix: str = "",
    cacheable_version: str = "",
) -> AgentConfig:
    """Register a custom agent at runtime.

    Usage:
        from benchflow.agents.registry import register_agent

        register_agent(
            name="my-agent",
            install_cmd="npm install -g my-agent",
            launch_cmd="my-agent --acp",
            requires_env=["MY_API_KEY"],
            skill_paths=["$HOME/.my-agent/skills"],
        )

    ``cacheable_version`` marks ``install_cmd`` as pinned, the way
    ``CACHEABLE_INSTALLS`` marks the built-in ones: it installs exactly what
    the version names, under /opt/runtime, the same way every trial, so on
    Docker it is built once into the agent's cached image layer and reused
    (the version is part of the layer's key and record). Only a plugin whose
    install pins every package should pass it. Without it the install runs in
    every trial, as before.

    ``model_route="agent"`` registers an agent that calls its own model
    endpoint, as the manifest field does (see AgentConfig.model_route).

    Returns the created AgentConfig.
    """
    if model_route not in MODEL_ROUTES:
        raise ValueError(
            f"model_route must be one of {', '.join(MODEL_ROUTES)}; got {model_route!r}"
        )
    config = AgentConfig(
        name=name,
        install_cmd=install_cmd,
        launch_cmd=launch_cmd,
        protocol=protocol,
        session_factory=session_factory,
        requires_env=requires_env or [],
        description=description,
        skill_paths=skill_paths or [],
        install_timeout=install_timeout,
        default_model=default_model,
        model_route=model_route,
        api_protocol=api_protocol,
        env_mapping=env_mapping or {},
        credential_files=credential_files or [],
        launcher_credential_files=launcher_credential_files or [],
        home_dirs=home_dirs or [],
        subscription_auth=subscription_auth,
        acp_model_format=acp_model_format,
        supports_acp_set_model=supports_acp_set_model,
        acp_model_config_id=acp_model_config_id,
        acp_effort_config_id=acp_effort_config_id,
        disallow_web_tools_setup_cmd=disallow_web_tools_setup_cmd,
        disallow_web_tools_owned_paths=disallow_web_tools_owned_paths or [],
        disallow_web_tools_launch_suffix=disallow_web_tools_launch_suffix,
        disallow_hosted_search_setup_cmd=disallow_hosted_search_setup_cmd,
        disallow_hosted_search_launch_suffix=disallow_hosted_search_launch_suffix,
    )
    with _REGISTRY_LOCK:
        AGENTS[name] = config
        AGENT_INSTALLERS[name] = install_cmd
        AGENT_LAUNCH[name] = launch_cmd
        if cacheable_version:
            CACHEABLE_INSTALLS[name] = cacheable_version
            _CACHEABLE_INSTALL_CMDS[name] = install_cmd
        elif name in _BUILTIN_CACHEABLE_INSTALLS:
            # A built-in's pin describes only its own command, so a replaced
            # command does not match it and cacheable_install says no.
            CACHEABLE_INSTALLS[name] = _BUILTIN_CACHEABLE_INSTALLS[name]
            _CACHEABLE_INSTALL_CMDS[name] = _BUILTIN_CACHEABLE_INSTALL_CMDS[name]
        else:
            # An earlier registration's pin does not describe this command.
            CACHEABLE_INSTALLS.pop(name, None)
            _CACHEABLE_INSTALL_CMDS.pop(name, None)
    return config


# --- Agent plugin packages (entry-point autoload) ------------------------------
# Out-of-core agent packages (e.g. the benchflow-ai/agents packages) register
# their agents either as an import side effect or via a zero-arg ``register()``.
# Without discovery, `bench eval run --agent omnigent-pi` only works if
# something in the process happened to import/call them first — in practice a
# hand-planted sitecustomize/.pth hack. Standard plugin pattern instead: any
# installed distribution may declare
#
#     [project.entry-points."benchflow.agents"]
#     my-agents = "my_agents"                          # module: import registers
#     # ...or:  my-agents = "my_agents.register:register"  (callable: invoked)
#
# and it is loaded here (after the core + manifest registries are built, so
# `register_agent` overwrite-by-name semantics apply — though a plugin may still
# choose not to overwrite, e.g. acp-registry skips names built-ins already own).
# A module-style value registers by import; a callable-style value is
# additionally invoked with no arguments. Installing the plugin distribution
# alongside benchflow (PyPI, git-subdirectory URL, or path) is then sufficient
# for `--agent <registered-name>` to resolve — note the DIST name, entry-point
# name, and registered AGENT name may all differ (installing dist `mimo-acp`
# registers agent `mimo`).
#
# Failure handling: each plugin is guarded (a broken plugin logs a warning with
# traceback and is skipped — siblings still load, the CLI never crashes), and a
# failed metadata scan disables discovery with a warning. Failures are recorded
# in FAILED_AGENT_PLUGINS so the eventual "Unknown agent" / raw-command-fallback
# paths can point back at the real cause.

# entry-point name -> "ExcType: message" for plugins that failed to load; the
# sentinel key "<entry-point-scan>" means discovery itself failed.
FAILED_AGENT_PLUGINS: dict[str, str] = {}


def _load_agent_plugin_packages() -> None:
    import logging
    from importlib.metadata import entry_points

    try:
        eps = entry_points(group="benchflow.agents")
    except Exception as exc:  # pragma: no cover - metadata backend quirks
        FAILED_AGENT_PLUGINS["<entry-point-scan>"] = describe_exception(exc)
        logging.getLogger(__name__).warning(
            "benchflow.agents entry-point scan failed; ALL agent plugins are "
            "disabled: %s",
            exc,
            exc_info=True,
        )
        return
    for ep in eps:
        try:
            loaded = ep.load()
            if callable(loaded):
                loaded()
        except Exception as exc:
            FAILED_AGENT_PLUGINS[ep.name] = describe_exception(exc)
            logging.getLogger(__name__).warning(
                "benchflow.agents plugin %r failed to load (some or all of its "
                "agents may be unavailable or partially registered): %s",
                ep.name,
                exc,
                exc_info=True,
            )


_load_agent_plugin_packages()
