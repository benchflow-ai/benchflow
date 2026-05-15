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
- ``protocol``           "acp" (default) or "cli". Almost always "acp".
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
                         ``credential_files``, e.g. ``.openclaw``).
- ``subscription_auth``  ``SubscriptionAuth`` describing host CLI login files
                         (e.g. ``claude login`` credentials) that can stand in
                         for an API key. API keys still take precedence.

Look at the existing entries below for worked examples:
``claude-agent-acp`` (subscription auth + env_mapping), ``codex-acp``
(credential_files), ``openclaw`` (home_dirs + custom shim), ``gemini``
(multi-file subscription auth).
"""

import base64
import shlex
from dataclasses import dataclass, field
from pathlib import Path


def _install_python_script(container_path: str, source: str) -> str:
    """Shell snippet that ensures python3 and writes `source` to container_path.

    Base64 transport — makes the install shell line content-agnostic so a line
    like `SHIMEOF` or `LAUNCHEREOF` inside the Python source can't collide with
    a heredoc terminator.

    Used by pi-acp, openclaw, and harvey-lab-harness — all three ship a Python
    launcher/shim baked into install_cmd. Semantics differ intentionally:
    pi and openclaw bridge BENCHFLOW_PROVIDER_* env vars to agent-native
    config; harvey-lab delegates to Harvey LAB's own model adapters which
    read provider env vars directly. A shared base is not yet justified —
    divergence is cheap, premature abstraction isn't.
    """
    encoded = base64.b64encode(source.encode()).decode()
    parent = shlex.quote(str(Path(container_path).parent))
    q_path = shlex.quote(container_path)
    return (
        "( command -v python3 >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1) ) && "
        f"mkdir -p {parent} && "
        f"echo {encoded} | base64 -d > {q_path} && "
        f"chmod +x {q_path}"
    )


# Isolated Node.js bootstrap for JavaScript-based ACP agents.
#
# Keep this out of system prefixes. Task images may need their own Node/npm
# versions, so BenchFlow installs agent runtime bits under /opt/benchflow.
# Install commands can put that prefix on PATH, but launch wrappers call the
# private Node binary explicitly so task subprocesses keep the task's PATH.
_BENCHFLOW_NODE_PREFIX = "/opt/benchflow/node"
_BENCHFLOW_JS_AGENT_PREFIX = "/opt/benchflow/js-agents"
_BENCHFLOW_BIN_PREFIX = "/opt/benchflow/bin"
_JS_AGENT_PATH = (
    f"{_BENCHFLOW_BIN_PREFIX}:{_BENCHFLOW_JS_AGENT_PREFIX}/bin:"
    f"{_BENCHFLOW_NODE_PREFIX}/bin:$PATH"
)
_NODE_INSTALL = (
    "set -o pipefail; "
    "export DEBIAN_FRONTEND=noninteractive; "
    f"BF_NODE_DIR={_BENCHFLOW_NODE_PREFIX}; "
    "BF_NODE_VERSION=22.14.0; "
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
    "  mkdir -p /opt/benchflow; "
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


def _js_agent_install(binary: str, package: str) -> str:
    """Install an npm-distributed agent into BenchFlow's isolated prefix."""
    agent_bin = f"{_BENCHFLOW_JS_AGENT_PREFIX}/bin/{binary}"
    wrapper = f"{_BENCHFLOW_BIN_PREFIX}/{binary}"
    return (
        f"{_NODE_INSTALL} && "
        f"mkdir -p {_BENCHFLOW_JS_AGENT_PREFIX} {_BENCHFLOW_BIN_PREFIX} && "
        f'export PATH="{_JS_AGENT_PATH}" && '
        f"( [ -x {agent_bin} ] || "
        f"{_BENCHFLOW_NODE_PREFIX}/bin/npm install -g "
        f"--prefix {_BENCHFLOW_JS_AGENT_PREFIX} {package}@latest ) && "
        f"printf '%s\\n' '#!/bin/sh' "
        f"'exec {_BENCHFLOW_NODE_PREFIX}/bin/node {agent_bin} \"$@\"' "
        f"> {wrapper} && "
        f"chmod +x {wrapper} && "
        f"chmod -R a+rX /opt/benchflow && "
        f"[ -x {agent_bin} ] && [ -x {wrapper} ]"
    )


def _js_agent_launch(binary: str, args: str = "") -> str:
    """Launch a JS agent through its isolated BenchFlow wrapper."""
    cmd = f"{_BENCHFLOW_BIN_PREFIX}/{binary}"
    return f"{cmd} {args}".rstrip()


# Path to the openclaw ACP shim script
_OPENCLAW_SHIM = (Path(__file__).parent / "openclaw_acp_shim.py").read_text()

# Path to the Pi launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
_PI_LAUNCHER = (Path(__file__).parent / "pi_acp_launcher.py").read_text()

# Path to the Harvey LAB ACP shim (runs Harvey LAB harness as an ACP agent)
_HARVEY_LAB_SHIM = (Path(__file__).parent / "harvey_lab_acp_shim.py").read_text()


def _json_settings_merge(path: str, mutator: str) -> str:
    """Idempotent JSON-settings merge as a one-line bash snippet."""
    py = (
        "import json,os,pathlib;"
        f"p=pathlib.Path(os.path.expandvars(os.path.expanduser({path!r})));"
        "p.parent.mkdir(parents=True, exist_ok=True);"
        "d=json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {};"
        f"{mutator};"
        "p.write_text(json.dumps(d, indent=2) + '\\n')"
    )
    return f"python3 -c {shlex.quote(py)}"


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


@dataclass
class AgentConfig:
    """Configuration for a supported agent."""

    name: str
    install_cmd: str
    launch_cmd: str
    protocol: str = "acp"  # "acp" or "cli"
    requires_env: list[str] = field(default_factory=list)
    description: str = ""
    skill_paths: list[str] = field(default_factory=list)
    install_timeout: int = 900  # seconds
    default_model: str = ""  # default model ID when --model is omitted
    api_protocol: str = ""
    # The LLM API protocol the agent natively speaks:
    # "anthropic-messages" | "openai-completions" | "openai-responses" |
    # "" (runtime/native).
    # Used to pick the correct provider endpoint when a provider exposes
    # multiple (e.g. zai has both anthropic-messages and openai-completions).
    env_mapping: dict[str, str] = field(default_factory=dict)
    # Maps BENCHFLOW_PROVIDER_* → agent-native env var names.
    # Applied by SDK after provider resolution.
    credential_files: list[CredentialFile] = field(default_factory=list)
    # Files to write into container before agent launch (e.g. auth.json).
    home_dirs: list[str] = field(default_factory=list)
    # Extra dot-dirs under $HOME to copy to sandbox user (for dirs not
    # derivable from skill_paths or credential_files, e.g. ".openclaw").
    acp_model_format: str = "bare"
    # How the agent expects the modelId in session/set_model:
    # "bare"           — just the model name (e.g. "claude-sonnet-4-6").
    #                    Default; works for claude-agent-acp, codex-acp.
    # "provider/model" — models.dev convention (e.g. "google/gemini-3.1-pro-preview").
    #                    Required by opencode, which uses Provider.parseModel()
    #                    to split on "/" and treats the first segment as provider ID.
    subscription_auth: SubscriptionAuth | None = None
    # Host CLI login that can substitute for an API key (e.g. OAuth tokens
    # from `claude login`). Detected automatically; API keys take precedence.
    supports_acp_set_model: bool = True
    # Some ACP agents configure the model through env/config at launch time and
    # do not implement session/set_model (e.g. OpenHands CLI ACP).
    disallow_web_tools_setup_cmd: str = ""
    # Shell snippet run after credentials/subscription auth are written when
    # BenchFlow's no-web policy is active. Uses BENCHFLOW_AGENT_HOME for the
    # target home so settings land in the same home the agent will run from.
    disallow_web_tools_launch_suffix: str = ""
    # String appended to launch_cmd when BenchFlow's no-web policy is active.
    # Use for agents whose supported toggle is a launch/config override.


# Agent registry — all supported agents
AGENTS: dict[str, AgentConfig] = {
    "claude-agent-acp": AgentConfig(
        name="claude-agent-acp",
        description="Claude Code via ACP (Anthropic's Agent Client Protocol)",
        skill_paths=["$HOME/.claude/skills"],
        install_cmd=_js_agent_install(
            "claude-agent-acp", "@zed-industries/claude-agent-acp"
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
        disallow_web_tools_setup_cmd=_json_settings_merge(
            "$BENCHFLOW_AGENT_HOME/.claude/settings.json",
            'd.setdefault("permissions",{}).setdefault("deny",[]);'
            '[d["permissions"]["deny"].append(t) for t in ["WebSearch","WebFetch"] '
            'if t not in d["permissions"]["deny"]]',
        ),
    ),
    "pi-acp": AgentConfig(
        name="pi-acp",
        description="Pi agent via ACP",
        skill_paths=["$HOME/.pi/agent/skills", "$HOME/.agents/skills"],
        install_cmd=(
            f"{_js_agent_install('pi', '@mariozechner/pi-coding-agent')} && "
            f"{_js_agent_install('pi-acp', 'pi-acp')} && "
            # Deploy launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/pi-acp-launcher", _PI_LAUNCHER
            )
        ),
        launch_cmd=f"{_BENCHFLOW_BIN_PREFIX}/pi-acp-launcher",
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime
        # Pi is multi-protocol: speaks Anthropic natively and OpenAI via
        # models.json.  Empty lets the provider determine the protocol so
        # multi-endpoint providers (e.g. zai) route to the right URL.
        api_protocol="",
        # env_mapping intentionally empty — the launch wrapper handles
        # protocol-dependent translation (env vars for Anthropic,
        # models.json for OpenAI-compatible providers like vLLM).
    ),
    "openclaw": AgentConfig(
        name="openclaw",
        description="OpenClaw agent via ACP shim — model set at runtime via --model",
        skill_paths=["$HOME/.claude/skills", "$WORKSPACE/skills"],
        install_cmd=(
            f"{_js_agent_install('openclaw', 'openclaw')} && "
            # Configure: auto-approve tools (no model — set at runtime via ACP set_model)
            "mkdir -p ~/.openclaw && "
            'echo \'{"version":1,"defaults":{"allow_all":true}}\''
            " > ~/.openclaw/exec-approvals.json && "
            # Deploy ACP shim
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/openclaw-acp-shim", _OPENCLAW_SHIM
            )
        ),
        launch_cmd=f"{_BENCHFLOW_BIN_PREFIX}/openclaw-acp-shim",
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime
        home_dirs=[".openclaw"],
    ),
    "codex-acp": AgentConfig(
        name="codex-acp",
        description="OpenAI Codex agent via ACP",
        skill_paths=["$HOME/.agents/skills"],
        install_cmd=_js_agent_install("codex-acp", "@zed-industries/codex-acp"),
        launch_cmd=_js_agent_launch(
            "codex-acp", "${OPENAI_BASE_URL:+-c openai_base_url=$OPENAI_BASE_URL}"
        ),
        protocol="acp",
        requires_env=["OPENAI_API_KEY"],
        api_protocol="openai-responses",
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "OPENAI_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "OPENAI_API_KEY",
        },
        credential_files=[
            CredentialFile(
                path="{home}/.codex/auth.json",
                env_source="OPENAI_API_KEY",
                template='{{"OPENAI_API_KEY": "{value}"}}',
            ),
        ],
        subscription_auth=SubscriptionAuth(
            replaces_env="OPENAI_API_KEY",
            detect_file="~/.codex/auth.json",
            files=[
                HostAuthFile("~/.codex/auth.json", "{home}/.codex/auth.json"),
            ],
        ),
        disallow_web_tools_launch_suffix=" -c tools.web_search=false",
    ),
    "gemini": AgentConfig(
        name="gemini",
        description="Google Gemini CLI via ACP",
        skill_paths=["$HOME/.gemini/skills"],
        install_cmd=_js_agent_install("gemini", "@google/gemini-cli"),
        launch_cmd=_js_agent_launch("gemini", "--acp --yolo"),
        protocol="acp",
        requires_env=["GOOGLE_API_KEY"],
        # api_protocol intentionally empty: Gemini speaks Google's native
        # GenerateContent format, which no current PROVIDERS entry exposes as
        # a multi-endpoint option. Set this when a Gemini-compatible provider
        # with multiple endpoints (e.g. OpenRouter) is added.
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "GEMINI_API_BASE_URL",
            "BENCHFLOW_PROVIDER_API_KEY": "GOOGLE_API_KEY",
        },
        subscription_auth=SubscriptionAuth(
            replaces_env="GEMINI_API_KEY",
            detect_file="~/.gemini/oauth_creds.json",
            files=[
                HostAuthFile(
                    "~/.gemini/oauth_creds.json", "{home}/.gemini/oauth_creds.json"
                ),
                HostAuthFile("~/.gemini/settings.json", "{home}/.gemini/settings.json"),
                HostAuthFile(
                    "~/.gemini/google_accounts.json",
                    "{home}/.gemini/google_accounts.json",
                ),
            ],
        ),
        disallow_web_tools_setup_cmd=_json_settings_merge(
            "$BENCHFLOW_AGENT_HOME/.gemini/settings.json",
            'd.setdefault("tools",{}).setdefault("exclude",[]);'
            '[d["tools"]["exclude"].append(t) for t in '
            '["google_web_search","web_fetch"] '
            'if t not in d["tools"]["exclude"]]',
        ),
    ),
    "opencode": AgentConfig(
        name="opencode",
        description="OpenCode via ACP — open-source coding agent (TypeScript)",
        skill_paths=["$HOME/.opencode/skills"],
        home_dirs=[".opencode"],
        install_cmd=_js_agent_install("opencode", "opencode-ai"),
        launch_cmd=_js_agent_launch("opencode", "acp"),
        protocol="acp",
        requires_env=[],  # inferred from --model at runtime
        acp_model_format="provider/model",
        # OpenCode uses models.dev provider IDs — its parseModel() splits
        # modelId on "/" so set_model must send "google/gemini-3.1-pro-preview",
        # not just "gemini-3.1-pro-preview".
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "OPENAI_BASE_URL",
        },
        disallow_web_tools_setup_cmd=_json_settings_merge(
            "$BENCHFLOW_AGENT_HOME/.config/opencode/opencode.json",
            'd.setdefault("tools",{})["webfetch"]=False',
        ),
    ),
    "harvey-lab-harness": AgentConfig(
        name="harvey-lab-harness",
        description="Harvey LAB harness — runs the original Harvey LAB agent loop "
        "(6 tools: bash, read, write, edit, glob, grep) via ACP shim",
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive && "
            # Ensure git is available
            "( command -v git >/dev/null 2>&1 || "
            "  (apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1) ) && "
            # Clone Harvey LAB repo
            "( [ -d /opt/harvey-labs/.git ] || "
            "  git clone --depth 1 https://github.com/harveyai/harvey-labs.git /opt/harvey-labs ) && "
            # Install Harvey LAB's Python dependencies
            "( command -v pip3 >/dev/null 2>&1 || "
            "  (apt-get update -qq && apt-get install -y -qq python3-pip >/dev/null 2>&1) ) && "
            "pip3 install -q anthropic openai google-genai "
            "python-docx pdfplumber openpyxl python-pptx markitdown pandas && "
            # Deploy ACP shim
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/harvey-lab-acp-shim", _HARVEY_LAB_SHIM
            )
        ),
        launch_cmd=f"HARVEY_LABS_ROOT=/opt/harvey-labs python3 {_BENCHFLOW_BIN_PREFIX}/harvey-lab-acp-shim",
        protocol="acp",
        requires_env=[],  # inferred from model at runtime (ANTHROPIC_API_KEY, etc.)
        # env_mapping intentionally empty — Harvey LAB adapters read
        # provider-specific env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY,
        # GOOGLE_API_KEY) directly; auto_inherit_env propagates these.
    ),
    "openhands": AgentConfig(
        name="openhands",
        description="OpenHands agent via ACP (multi-model, Python-based)",
        skill_paths=["$HOME/.agents/skills", "$WORKSPACE/.agents/skills"],
        home_dirs=[".openhands"],
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            "( command -v curl >/dev/null 2>&1 && command -v git >/dev/null 2>&1 || "
            "  if command -v apt-get >/dev/null 2>&1; then "
            "    apt-get update -qq && "
            "    apt-get install -y -qq curl ca-certificates git >/dev/null 2>&1; "
            "  elif command -v dnf >/dev/null 2>&1; then "
            "    dnf -y install curl ca-certificates git >/dev/null 2>&1; "
            "  elif command -v apk >/dev/null 2>&1; then "
            "    apk add --no-cache curl ca-certificates git >/dev/null 2>&1; "
            "  else "
            "    echo 'OpenHands GitHub install requires curl and git' >&2; "
            "    exit 127; "
            "  fi ) && "
            "( UV_OK=0; "
            "  if command -v uv >/dev/null 2>&1; then "
            "    UV_VER=$(uv --version 2>/dev/null | awk '{print $2}'); "
            '    if [ -n "$UV_VER" ] && '
            '       [ "$(printf \'%s\\n\' 0.11.6 "$UV_VER" | sort -V | head -n1)" = "0.11.6" ]; then '
            "      UV_OK=1; "
            "    fi; "
            "  fi; "
            '  if [ "$UV_OK" = 0 ]; then '
            "    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 && "
            '    export PATH="$HOME/.local/bin:$PATH"; '
            "  fi && "
            "uv tool install --force --refresh "
            "--from 'git+https://github.com/OpenHands/OpenHands-CLI.git@main' "
            "openhands --python 3.12 && "
            "  uv tool list | grep -q '^openhands\\b' ) && "
            # Let sandbox user traverse to uv-managed Python interpreter path.
            "chmod o+x /root /root/.local /root/.local/share "
            "/root/.local/share/uv /root/.local/share/uv/tools 2>/dev/null; "
            # Seed config so OpenHands ACP auth check passes before env override.
            "mkdir -p ~/.openhands && "
            'echo \'{"llm":{"model":"placeholder","api_key":"placeholder"}}\' '
            "> ~/.openhands/agent_settings.json && "
            "command -v openhands >/dev/null 2>&1"
        ),
        launch_cmd=(
            'export PATH="$HOME/.local/bin:$PATH" && '
            "mkdir -p ~/.openhands && "
            'printf \'{"llm":{"model":"%s","api_key":"%s"}}\' '
            '"$LLM_MODEL" "$LLM_API_KEY" > ~/.openhands/agent_settings.json && '
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
        disallow_web_tools_setup_cmd=(
            'mkdir -p "$BENCHFLOW_AGENT_HOME/.openhands" && '
            "printf '[agent]\\nenable_browsing = false\\n' "
            '> "$BENCHFLOW_AGENT_HOME/.openhands/config.toml"'
        ),
    ),
}


# Derived lookup tables — install/launch commands by agent name.
# Updated by register_agent() when new agents are added at runtime.
AGENT_INSTALLERS: dict[str, str] = {name: a.install_cmd for name, a in AGENTS.items()}
AGENT_LAUNCH: dict[str, str] = {name: a.launch_cmd for name, a in AGENTS.items()}


def get_sandbox_home_dirs() -> set[str]:
    """Collect user home config/auth dirs BenchFlow may materialize for the sandbox user.

    Derives from three sources across all registered agents:
    - credential_files: {home}/.foo/... → ".foo"
    - subscription_auth.files: {home}/.foo/... → ".foo"
    - home_dirs: explicit extras (e.g. ".openclaw")

    Skill paths are excluded: deploy_skills() now links those paths directly to a
    shared skills tree instead of relying on sandbox-home copies.
    """
    dirs: set[str] = set()
    for cfg in AGENTS.values():
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
    # Check custom providers first
    from benchflow.agents.providers import resolve_auth_env

    custom = resolve_auth_env(model)
    if custom is not None:
        return custom
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
    "pi": "pi-acp",
    "openclaw": "openclaw",
    "openhands": "openhands",
    "oh": "openhands",
    "harvey-lab": "harvey-lab-harness",
}

VALID_PROTOCOLS = {"acp", "harbor"}


def parse_agent_spec(spec: str) -> tuple[str, str]:
    """Parse an agent spec like 'acp/claude-agent-acp' or 'claude'.

    Returns (protocol, agent_name) with alias resolution.
    Bare names default to 'acp' protocol.
    """
    if "/" in spec:
        protocol, name = spec.split("/", 1)
    else:
        protocol, name = "acp", spec

    name = AGENT_ALIASES.get(name, name)
    return protocol, name


def resolve_agent(spec: str) -> AgentConfig:
    """Resolve an agent spec to an AgentConfig.

    Supports: bare name, alias, protocol/name.
    Raises KeyError with suggestions for unknown agents.
    """
    protocol, name = parse_agent_spec(spec)

    if protocol not in VALID_PROTOCOLS:
        raise KeyError(
            f"Unknown protocol: {protocol!r}. Valid: {', '.join(sorted(VALID_PROTOCOLS))}"
        )

    if protocol == "harbor":
        return AgentConfig(
            name=name,
            install_cmd="",
            launch_cmd="",
            protocol="harbor",
            requires_env=[],
            description=f"Harbor agent: {name}",
        )

    if name in AGENTS:
        return AGENTS[name]

    # Fuzzy suggestion
    from difflib import get_close_matches

    close = get_close_matches(name, list(AGENTS.keys()), n=1, cutoff=0.6)
    if close:
        raise KeyError(f"Unknown agent: {name!r}. Did you mean: {close[0]!r}?")
    raise KeyError(
        f"Unknown agent: {name!r}. Available: {', '.join(sorted(AGENTS.keys()))}"
    )


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
    requires_env: list[str] | None = None,
    description: str = "",
    skill_paths: list[str] | None = None,
    install_timeout: int = 900,
    env_mapping: dict[str, str] | None = None,
    credential_files: list[CredentialFile] | None = None,
    home_dirs: list[str] | None = None,
    subscription_auth: SubscriptionAuth | None = None,
    acp_model_format: str = "bare",
    supports_acp_set_model: bool = True,
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

    Returns the created AgentConfig.
    """
    config = AgentConfig(
        name=name,
        install_cmd=install_cmd,
        launch_cmd=launch_cmd,
        protocol=protocol,
        requires_env=requires_env or [],
        description=description,
        skill_paths=skill_paths or [],
        install_timeout=install_timeout,
        env_mapping=env_mapping or {},
        credential_files=credential_files or [],
        home_dirs=home_dirs or [],
        subscription_auth=subscription_auth,
        acp_model_format=acp_model_format,
        supports_acp_set_model=supports_acp_set_model,
    )
    AGENTS[name] = config
    AGENT_INSTALLERS[name] = install_cmd
    AGENT_LAUNCH[name] = launch_cmd
    return config
