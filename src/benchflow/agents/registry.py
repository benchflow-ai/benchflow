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


def _install_text_file(
    container_path: str,
    source: str,
    *,
    executable: bool = False,
) -> str:
    """Shell snippet that writes ``source`` to ``container_path``.

    Base64 transport — makes the install shell line content-agnostic so a line
    like `SHIMEOF` or `LAUNCHEREOF` inside the Python source can't collide with
    a heredoc terminator.
    """
    encoded = base64.b64encode(source.encode()).decode()
    parent = shlex.quote(str(Path(container_path).parent))
    q_path = shlex.quote(container_path)
    chmod = f"chmod +x {q_path}" if executable else f"chmod a+r {q_path}"
    return f"mkdir -p {parent} && echo {encoded} | base64 -d > {q_path} && {chmod}"


def _install_python_script(container_path: str, source: str) -> str:
    """Shell snippet that ensures python3 and writes `source` to container_path.

    Used by pi-acp, openclaw, and harvey-lab-harness — all three ship a Python
    launcher/shim baked into install_cmd. Semantics differ intentionally:
    pi and openclaw bridge BENCHFLOW_PROVIDER_* env vars to agent-native
    config; harvey-lab delegates to Harvey LAB's own model adapters which
    read provider env vars directly. A shared base is not yet justified —
    divergence is cheap, premature abstraction isn't.
    """
    return (
        "( command -v python3 >/dev/null 2>&1 || "
        "(apt-get update -qq && apt-get install -y -qq python3 >/dev/null 2>&1) ) && "
        + _install_text_file(container_path, source, executable=True)
    )


def _install_python_module(module_name: str, source: str) -> str:
    """Install a small BenchFlow helper module into the sandbox package path."""
    parts = module_name.split(".")
    if len(parts) < 2 or any(not part.isidentifier() for part in parts):
        raise ValueError(f"invalid module name: {module_name!r}")
    module_path = Path("/opt/benchflow").joinpath(*parts).with_suffix(".py")
    package_dirs: list[Path] = []
    package_dir = Path("/opt/benchflow")
    for part in parts[:-1]:
        package_dir /= part
        package_dirs.append(package_dir)
    init_files = " ".join(
        shlex.quote(str(path / "__init__.py")) for path in package_dirs
    )
    return _install_text_file(str(module_path), source) + f" && touch {init_files}"


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


# Isolated Node.js bootstrap for JavaScript-based ACP agents.
#
# Keep this out of system prefixes. Task images may need their own Node/npm
# versions, so BenchFlow installs agent runtime bits under /opt/benchflow.
# Install commands can put that prefix on PATH, but launch wrappers call the
# private Node binary explicitly so task subprocesses keep the task's PATH.
_BENCHFLOW_NODE_PREFIX = "/opt/benchflow/node"
_BENCHFLOW_JS_AGENT_PREFIX = "/opt/benchflow/js-agents"
_BENCHFLOW_BIN_PREFIX = "/opt/benchflow/bin"
_BENCHFLOW_PYTHONPATH = "/opt/benchflow"
_OPENHANDS_CLI_GIT_REV = "3ca17446c5d9c1e35e054803478a3501ec251ecf"
_OPENHANDS_SDK_VERSION = "1.22.1"
_OPENHANDS_TOOLS_VERSION = "1.22.1"
_BROWSER_USE_CLI_VERSION = "0.13.1"
_BROWSER_USE_CLI_VENV = "/opt/benchflow/browser-use-cli-venv"
_BROWSER_USE_CLI_BROWSERS = "/opt/benchflow/ms-playwright"

_STAGEHAND_AGENT_VERSION = "3.5.0"
_STAGEHAND_GOOGLE_VERSION = "2.0.74"
_STAGEHAND_PLAYWRIGHT_VERSION = "1.55.1"
_STAGEHAND_AGENT_DIR = "/opt/benchflow/stagehand-agent"
_STAGEHAND_BROWSERS = "/opt/benchflow/stagehand-ms-playwright"
_JS_AGENT_PATH = (
    f"{_BENCHFLOW_BIN_PREFIX}:{_BENCHFLOW_JS_AGENT_PREFIX}/bin:"
    f"{_BENCHFLOW_NODE_PREFIX}/bin:$PATH"
)
_BENCHFLOW_PYTHONPATH_PREFIX = (
    f"PYTHONPATH={_BENCHFLOW_PYTHONPATH}${{PYTHONPATH:+:$PYTHONPATH}}"
)
# Node >=22.19 is required by current openclaw (the JS agents install
# @latest); keep this pin at or above that floor or the openclaw ACP
# bootstrap aborts at its runtime version check (BF-10).
_NODE_INSTALL = (
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


def _npm_package_spec(package: str) -> str:
    """Return an npm install spec, defaulting unversioned packages to latest."""
    if "@" in package.lstrip("@"):
        return package
    return f"{package}@latest"


def _js_agent_install(binary: str, package: str) -> str:
    """Install an npm-distributed agent into BenchFlow's isolated prefix."""
    agent_bin = f"{_BENCHFLOW_JS_AGENT_PREFIX}/bin/{binary}"
    wrapper = f"{_BENCHFLOW_BIN_PREFIX}/{binary}"
    package_spec = _npm_package_spec(package)
    install_guard = "" if package_spec == package else f"[ -x {agent_bin} ] || "
    return (
        f"{_NODE_INSTALL} && "
        f"mkdir -p {_BENCHFLOW_JS_AGENT_PREFIX} {_BENCHFLOW_BIN_PREFIX} && "
        f'export PATH="{_JS_AGENT_PATH}" && '
        f"( {install_guard}{_BENCHFLOW_NODE_PREFIX}/bin/npm install -g "
        f"--prefix {_BENCHFLOW_JS_AGENT_PREFIX} {package_spec} ) && "
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


def _benchflow_python_launch(command: str) -> str:
    """Launch a sandbox Python shim with bundled BenchFlow helper modules."""
    return f"{_BENCHFLOW_PYTHONPATH_PREFIX} {command}"


# Path to the openclaw ACP shim script
_OPENCLAW_SHIM = (Path(__file__).parent / "openclaw_acp_shim.py").read_text()

# Path to the Pi launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
_PI_LAUNCHER = (Path(__file__).parent / "pi_acp_launcher.py").read_text()

# Path to the Harvey LAB ACP shim (runs Harvey LAB harness as an ACP agent)
_HARVEY_LAB_SHIM = (Path(__file__).parent / "harvey_lab_acp_shim.py").read_text()

# Path to the deepagents ACP shim (runs LangChain's create_deep_agent as an ACP agent)
_DEEPAGENTS_SHIM = (Path(__file__).parent / "deepagents_acp_shim.py").read_text()

# Path to the Browser Use smoke ACP shim (fixture agent adapter dogfood)
_BROWSER_USE_SMOKE_SHIM = (
    Path(__file__).parent / "browser_use_smoke_acp_shim.py"
).read_text()

# Path to the Computer Use smoke ACP shim (fixture desktop adapter dogfood)
_COMPUTER_USE_SMOKE_SHIM = (
    Path(__file__).parent / "computer_use_smoke_acp_shim.py"
).read_text()

# Path to the Browser Use CLI ACP shim (wraps the real browser-use CLI harness)
_BROWSER_USE_CLI_SHIM = (
    Path(__file__).parent / "browser_use_cli_acp_shim.py"
).read_text()

# Path to the Browser Use Agent ACP shim (wraps browser_use.Agent)
_BROWSER_USE_AGENT_SHIM = (
    Path(__file__).parent / "browser_use_agent_acp_shim.py"
).read_text()

# Path to the Stagehand Agent ACP shim (wraps Stagehand's DOM agent loop)
_STAGEHAND_AGENT_SHIM = (
    Path(__file__).parent / "stagehand_agent_acp_shim.py"
).read_text()

# Shared browser environment adapter runtime bundled for standalone shims.
_BROWSER_ENVIRONMENT_RUNTIME = (
    Path(__file__).parents[1] / "environment" / "browser_runtime.py"
).read_text()
_INSTALL_BROWSER_ENVIRONMENT_RUNTIME = _install_python_module(
    "benchflow.environment.browser_runtime",
    _BROWSER_ENVIRONMENT_RUNTIME,
)

# Shared desktop environment adapter runtime bundled for standalone shims.
_DESKTOP_ENVIRONMENT_RUNTIME = (
    Path(__file__).parents[1] / "environment" / "desktop_runtime.py"
).read_text()
_INSTALL_DESKTOP_ENVIRONMENT_RUNTIME = _install_python_module(
    "benchflow.environment.desktop_runtime",
    _DESKTOP_ENVIRONMENT_RUNTIME,
)


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
    protocol: str = "acp"  # "acp" | "cli" | "session-factory"
    session_factory: str = ""
    # Non-ACP only. When ``protocol == "session-factory"`` this is a dotted
    # ``module:callable`` entrypoint (e.g. ``omnigent.agent:build_omnigent_agent``)
    # that returns an object satisfying the Agent Protocol (connect/capabilities).
    # The kernel's non-ACP CONNECT branch (rollout) instantiates it INSTEAD of
    # connect_acp and wires the returned Session into the trajectory sink. Empty
    # for ACP agents (the default path), so this is fully additive.
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
    # multiple (e.g. zai has anthropic-messages, openai-responses, and
    # openai-completions).
    env_mapping: dict[str, str] = field(default_factory=dict)
    # Maps BENCHFLOW_PROVIDER_* → agent-native env var names.
    # Applied by SDK after provider resolution.
    credential_files: list[CredentialFile] = field(default_factory=list)
    # Files to write into container before agent launch (e.g. auth.json).
    home_dirs: list[str] = field(default_factory=list)
    # Extra dot-dirs under $HOME to copy to sandbox user (for dirs not
    # derivable from skill_paths or credential_files, e.g. ".openclaw").
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
    supports_acp_set_model: bool = True
    # Some ACP agents configure the model through env/config at launch time and
    # do not implement session/set_model (e.g. OpenHands CLI ACP).
    acp_model_config_id: str = ""
    # ACP session config option id used for model selection when an agent
    # exposes model as a session option instead of implementing set_model.
    acp_effort_config_id: str = ""
    # ACP session config option id used for reasoning/thinking effort.
    disallow_web_tools_setup_cmd: str = ""
    # Shell snippet run after credentials/subscription auth are written when
    # BenchFlow's no-web policy is active. Uses BENCHFLOW_AGENT_HOME for the
    # target home so settings land in the same home the agent will run from.
    disallow_web_tools_owned_paths: list[str] = field(default_factory=list)
    # Directories under $HOME that disallow_web_tools_setup_cmd may create and
    # that must remain writable by the sandbox user after the root-run setup.
    disallow_web_tools_launch_suffix: str = ""
    # String appended to launch_cmd when BenchFlow's no-web policy is active.
    # Use for agents whose supported toggle is a launch/config override.


# ── Pure agents ───────────────────────────────────────────────────────────────
# A *pure* agent exposes a single ``run`` callable and ships NO protocol code;
# the generic ``acp_serve.py`` wraps it on stdio ("the harness handles the
# shim"). This replaces the per-agent ACP shims for new agents.
_ACP_SERVE_SOURCE = (Path(__file__).parent / "acp_serve.py").read_text()
_ACP_SERVE_PATH = f"{_BENCHFLOW_BIN_PREFIX}/acp_serve.py"


def _pure_agent_config(
    *,
    name: str,
    agent_spec: str,
    pip: list[str],
    apt: tuple[str, ...] = (),
    requires_env: list[str] | None = None,
    default_model: str = "",
    description: str = "",
    install_timeout: int = 1200,
) -> AgentConfig:
    """AgentConfig for a pure agent (a ``module:callable`` run entry) served by
    the generic acp_serve.py. The agent ships zero benchflow/protocol code: its
    package is pip-installed (``pip``) and acp_serve.py wraps its ``run``.
    """
    venv = f"/opt/benchflow/{name}-venv"
    pip_specs = " ".join(shlex.quote(p) for p in pip)
    apt_step = (_apt_install(*apt) + " && ") if apt else ""
    install_cmd = (
        "export DEBIAN_FRONTEND=noninteractive && "
        "( command -v python3 >/dev/null 2>&1 || "
        f"  {_apt_install('python3', 'python3-venv', 'python3-pip')} ) && "
        + apt_step
        + f"( [ -x {venv}/bin/python ] || python3 -m venv {venv} || "
        f"  ( {_apt_install('python3-venv')} && python3 -m venv {venv} ) ) && "
        f"{venv}/bin/python -m pip install -q --upgrade pip && "
        f"{venv}/bin/python -m pip install -q {pip_specs} && "
        + _install_python_script(_ACP_SERVE_PATH, _ACP_SERVE_SOURCE)
        + " && chmod -R a+rX /opt/benchflow"
    )
    launch_cmd = _benchflow_python_launch(
        f"BENCHFLOW_AGENT_NAME={shlex.quote(name)} {venv}/bin/python "
        f"{_ACP_SERVE_PATH} {shlex.quote(agent_spec)}"
    )
    return AgentConfig(
        name=name,
        description=description or f"{name} — pure agent via generic ACP serve",
        install_cmd=install_cmd,
        launch_cmd=launch_cmd,
        protocol="acp",
        requires_env=requires_env or [],
        install_timeout=install_timeout,
        default_model=default_model,
    )


# Agent registry — all supported agents
AGENTS: dict[str, AgentConfig] = {
    "claude-agent-acp": AgentConfig(
        name="claude-agent-acp",
        description="Claude Code via ACP (Anthropic's Agent Client Protocol)",
        skill_paths=["$HOME/.claude/skills"],
        # Pinned to 0.40.0: the config-option wiring below (set_config_option +
        # the "model"/"effort" ids) targets this version's ACP protocol (sdk
        # 0.24, which dropped session/set_model). The option ids are coupled to
        # this pin — re-verify them when bumping. runtime.py uses
        # capability-first dispatch for the rest of the family.
        install_cmd=_js_agent_install(
            "claude-agent-acp", "@agentclientprotocol/claude-agent-acp@0.40.0"
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
        disallow_web_tools_owned_paths=["$HOME/.claude"],
        supports_acp_set_model=False,
        acp_model_config_id="model",
        acp_effort_config_id="effort",
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
        # Pinned for reproducibility: an unpinned @agentclientprotocol install
        # floats to latest and can silently break agent activation when the ACP
        # protocol changes (claude-agent-acp above hit exactly this — sdk 0.24
        # dropped session/set_model). 0.0.45 ships sdk 0.22.x, which still
        # implements session/set_model. If a future bump advertises a model
        # config option instead, runtime.py's capability-first dispatch routes
        # the model through that option — but re-verify model selection when
        # bumping this pin.
        install_cmd=_js_agent_install(
            "codex-acp", "@agentclientprotocol/codex-acp@0.0.45"
        ),
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
        install_cmd=_js_agent_install("gemini", "@google/gemini-cli@0.42.0"),
        launch_cmd=_js_agent_launch("gemini", "--acp --yolo"),
        protocol="acp",
        # The Gemini CLI reads GEMINI_API_KEY natively. GOOGLE_API_KEY is
        # accepted as an alias: auto_inherit_env mirrors it both ways so users
        # can set either one. Advertise GEMINI_API_KEY here so `agent show`
        # matches what the CLI actually reads (#342).
        requires_env=["GEMINI_API_KEY"],
        # Default to a sane Gemini model so `--agent gemini` works without
        # --model and never cross-wires to a Claude default (#343).
        default_model="gemini-2.5-flash",
        # api_protocol intentionally empty: Gemini speaks Google's native
        # GenerateContent format, which no current PROVIDERS entry exposes as
        # a multi-endpoint option. Set this when a Gemini-compatible provider
        # with multiple endpoints (e.g. OpenRouter) is added.
        env_mapping={
            "BENCHFLOW_PROVIDER_BASE_URL": "GOOGLE_GEMINI_BASE_URL",
            # Map to the CLI-native var; auto_inherit_env mirrors it to
            # GOOGLE_API_KEY for compatibility with users who set that alias.
            "BENCHFLOW_PROVIDER_API_KEY": "GEMINI_API_KEY",
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
        disallow_web_tools_owned_paths=["$HOME/.gemini"],
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
        disallow_web_tools_owned_paths=["$HOME/.config/opencode"],
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
            "( python3 -m venv /opt/benchflow/harvey-lab-venv 2>/dev/null || "
            "  (apt-get update -qq && apt-get install -y -qq python3-venv >/dev/null 2>&1 && "
            "   python3 -m venv /opt/benchflow/harvey-lab-venv) ) && "
            "/opt/benchflow/harvey-lab-venv/bin/python -m pip install -q "
            "anthropic openai google-genai "
            "python-docx pdfplumber openpyxl python-pptx markitdown pandas && "
            # Deploy ACP shim
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/harvey-lab-acp-shim", _HARVEY_LAB_SHIM
            )
        ),
        launch_cmd=f"HARVEY_LABS_ROOT=/opt/harvey-labs /opt/benchflow/harvey-lab-venv/bin/python {_BENCHFLOW_BIN_PREFIX}/harvey-lab-acp-shim",
        protocol="acp",
        requires_env=[],  # inferred from model at runtime (ANTHROPIC_API_KEY, etc.)
        # env_mapping intentionally empty — Harvey LAB adapters read
        # provider-specific env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY,
        # GOOGLE_API_KEY) directly; auto_inherit_env propagates these.
    ),
    "deepagents": AgentConfig(
        name="deepagents",
        description="deepagents harness — runs LangChain's create_deep_agent loop "
        "(planning, sub-agents, filesystem + shell tools) via ACP shim, driving "
        "deepseek-v4-pro through the OpenAI-compatible provider",
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive && "
            # deepagents is a Python package with heavy langchain deps; isolate
            # it in its own venv so it never collides with task-image Python.
            "( command -v pip3 >/dev/null 2>&1 || "
            "  (apt-get update -qq && apt-get install -y -qq python3-pip >/dev/null 2>&1) ) && "
            "( python3 -m venv /opt/benchflow/deepagents-venv 2>/dev/null || "
            "  (apt-get update -qq && apt-get install -y -qq python3-venv >/dev/null 2>&1 && "
            "   python3 -m venv /opt/benchflow/deepagents-venv) ) && "
            # deepagents pulls in langchain/langchain-core/langchain-anthropic/
            # langchain-google-genai; langchain-openai is NOT a deepagents dep but
            # is required for the OpenAI-compatible deepseek-v4-pro chat model.
            "/opt/benchflow/deepagents-venv/bin/python -m pip install -q "
            "deepagents langchain-openai && "
            # Let the sandbox user traverse + execute the venv interpreter.
            "chmod -R a+rX /opt/benchflow/deepagents-venv && "
            # Deploy ACP shim
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/deepagents-acp-shim", _DEEPAGENTS_SHIM
            )
        ),
        launch_cmd=(
            f"/opt/benchflow/deepagents-venv/bin/python {_BENCHFLOW_BIN_PREFIX}/deepagents-acp-shim"
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
    "browser-use-smoke": AgentConfig(
        name="browser-use-smoke",
        description=(
            "Browser Use smoke ACP shim — fixture agent adapter for local "
            "Browser Use-shaped parity checks"
        ),
        install_cmd=(
            _INSTALL_BROWSER_ENVIRONMENT_RUNTIME
            + " && "
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/browser-use-smoke-acp-shim",
                _BROWSER_USE_SMOKE_SHIM,
            )
        ),
        launch_cmd=_benchflow_python_launch(
            f"{_BENCHFLOW_BIN_PREFIX}/browser-use-smoke-acp-shim"
        ),
        protocol="acp",
        requires_env=[],
        default_model="browser-use-smoke",
    ),
    "computer-use-smoke": AgentConfig(
        name="computer-use-smoke",
        description=(
            "Computer Use smoke ACP shim — fixture desktop adapter for local "
            "Cua-backed parity checks"
        ),
        install_cmd=(
            _INSTALL_DESKTOP_ENVIRONMENT_RUNTIME
            + " && "
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/computer-use-smoke-acp-shim",
                _COMPUTER_USE_SMOKE_SHIM,
            )
        ),
        launch_cmd=_benchflow_python_launch(
            f"{_BENCHFLOW_BIN_PREFIX}/computer-use-smoke-acp-shim"
        ),
        protocol="acp",
        requires_env=[],
        default_model="computer-use-smoke",
    ),
    # mini-computer-agent: the pure computer-use agent (mini-swe-agent style) from
    # benchflow-ai/agents, served by the generic acp_serve.py. Coordinates are
    # [0,1000]-normalized -> pixels in the agent core (the earlier baked shim fed
    # them as raw pixels, ~3x off).
    "mini-computer-agent": _pure_agent_config(
        name="mini-computer-agent",
        agent_spec="mini_computer_agent.core:run",
        # Pip-install the published agent (it declares litellm). Branch ref so it
        # resolves pre-merge; switch to @main once the agents PR lands.
        pip=[
            "mini-computer-agent @ git+https://github.com/benchflow-ai/agents.git"
            "@feat/mini-computer-acp#subdirectory=mini-computer-agent",
        ],
        apt=("xdotool", "scrot"),
        requires_env=["GEMINI_API_KEY"],
        default_model="gemini-3.5-flash",
        description=(
            "mini-computer-agent — minimal computer-use agent (screenshot -> any "
            "vision model -> xdotool); pure agent served via the generic ACP serve"
        ),
    ),
    "browser-use-cli": AgentConfig(
        name="browser-use-cli",
        description=(
            "Browser Use CLI via ACP shim — direct-control browser harness "
            "smoke adapter"
        ),
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive; "
            "( command -v python3 >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq python3 python3-pip "
            "python3-venv >/dev/null 2>&1) ) && "
            f"( python3 -m venv {_BROWSER_USE_CLI_VENV} 2>/dev/null || "
            "(apt-get update -qq && apt-get install -y -qq python3-venv "
            f">/dev/null 2>&1 && python3 -m venv {_BROWSER_USE_CLI_VENV}) ) && "
            f"{_BROWSER_USE_CLI_VENV}/bin/python -m pip install -q uv "
            f"browser-use=={_BROWSER_USE_CLI_VERSION} && "
            f"PLAYWRIGHT_BROWSERS_PATH={_BROWSER_USE_CLI_BROWSERS} "
            f"PATH={_BROWSER_USE_CLI_VENV}/bin:$PATH "
            f"{_BROWSER_USE_CLI_VENV}/bin/browser-use install && "
            + _INSTALL_BROWSER_ENVIRONMENT_RUNTIME
            + " && "
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/browser-use-cli-acp-shim",
                _BROWSER_USE_CLI_SHIM,
            )
            + " && chmod -R a+rX /opt/benchflow"
        ),
        launch_cmd=_benchflow_python_launch(
            f"PLAYWRIGHT_BROWSERS_PATH={_BROWSER_USE_CLI_BROWSERS} "
            f"BROWSER_USE_BIN={_BROWSER_USE_CLI_VENV}/bin/browser-use "
            f"{_BROWSER_USE_CLI_VENV}/bin/python "
            f"{_BENCHFLOW_BIN_PREFIX}/browser-use-cli-acp-shim"
        ),
        protocol="acp",
        requires_env=[],
        install_timeout=1200,
        default_model="browser-use-cli",
    ),
    "browser-use-agent": AgentConfig(
        name="browser-use-agent",
        description=(
            "Browser Use Agent via ACP shim — LLM-driven Browser Use loop "
            "for browser eval smoke runs"
        ),
        install_cmd=(
            "export DEBIAN_FRONTEND=noninteractive; "
            "( command -v python3 >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq python3 python3-pip "
            "python3-venv >/dev/null 2>&1) ) && "
            f"( python3 -m venv {_BROWSER_USE_CLI_VENV} 2>/dev/null || "
            "(apt-get update -qq && apt-get install -y -qq python3-venv "
            f">/dev/null 2>&1 && python3 -m venv {_BROWSER_USE_CLI_VENV}) ) && "
            f"{_BROWSER_USE_CLI_VENV}/bin/python -m pip install -q uv "
            f"browser-use=={_BROWSER_USE_CLI_VERSION} && "
            f"PLAYWRIGHT_BROWSERS_PATH={_BROWSER_USE_CLI_BROWSERS} "
            f"PATH={_BROWSER_USE_CLI_VENV}/bin:$PATH "
            f"{_BROWSER_USE_CLI_VENV}/bin/browser-use install && "
            + _INSTALL_BROWSER_ENVIRONMENT_RUNTIME
            + " && "
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/browser-use-agent-acp-shim",
                _BROWSER_USE_AGENT_SHIM,
            )
            + " && chmod -R a+rX /opt/benchflow"
        ),
        launch_cmd=_benchflow_python_launch(
            f"PLAYWRIGHT_BROWSERS_PATH={_BROWSER_USE_CLI_BROWSERS} "
            f"{_BROWSER_USE_CLI_VENV}/bin/python "
            f"{_BENCHFLOW_BIN_PREFIX}/browser-use-agent-acp-shim"
        ),
        protocol="acp",
        requires_env=["GEMINI_API_KEY"],
        install_timeout=1200,
        default_model="gemini-2.5-flash",
    ),
    "stagehand-agent": AgentConfig(
        name="stagehand-agent",
        description=(
            "Stagehand Agent via ACP shim — DOM-mode Stagehand browser loop "
            "for browser eval smoke runs"
        ),
        install_cmd=(
            f"{_NODE_INSTALL} && "
            f"{_apt_install('ca-certificates', 'python3', 'make', 'g++')} && "
            f"mkdir -p {_STAGEHAND_AGENT_DIR} {_STAGEHAND_BROWSERS} && "
            f"cd {_STAGEHAND_AGENT_DIR} && "
            "( [ -d node_modules/@browserbasehq/stagehand ] && "
            "[ -d node_modules/@ai-sdk/google ] && "
            "[ -d node_modules/playwright ] || "
            f"( {_BENCHFLOW_NODE_PREFIX}/bin/npm init -y >/dev/null 2>&1 && "
            f"{_BENCHFLOW_NODE_PREFIX}/bin/npm install --omit=dev "
            "--no-audit --no-fund "
            f"@browserbasehq/stagehand@{_STAGEHAND_AGENT_VERSION} "
            f"@ai-sdk/google@{_STAGEHAND_GOOGLE_VERSION} "
            f"playwright@{_STAGEHAND_PLAYWRIGHT_VERSION} ) ) && "
            "( "
            f"PLAYWRIGHT_BROWSERS_PATH={_STAGEHAND_BROWSERS} "
            f"{_BENCHFLOW_NODE_PREFIX}/bin/node -e "
            '\'const fs=require("fs"); '
            'const { chromium }=require("playwright"); '
            "process.exit(fs.existsSync(chromium.executablePath()) ? 0 : 1)' || "
            f"PLAYWRIGHT_BROWSERS_PATH={_STAGEHAND_BROWSERS} "
            f"{_BENCHFLOW_NODE_PREFIX}/bin/npx playwright install chromium --with-deps "
            ") && "
            + _INSTALL_BROWSER_ENVIRONMENT_RUNTIME
            + " && "
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/stagehand-agent-acp-shim",
                _STAGEHAND_AGENT_SHIM,
            )
            + " && chmod -R a+rX /opt/benchflow"
        ),
        launch_cmd=_benchflow_python_launch(
            f"NODE_PATH={_STAGEHAND_AGENT_DIR}/node_modules "
            f"PLAYWRIGHT_BROWSERS_PATH={_STAGEHAND_BROWSERS} "
            f"STAGEHAND_AGENT_NODE={_BENCHFLOW_NODE_PREFIX}/bin/node "
            f"python3 {_BENCHFLOW_BIN_PREFIX}/stagehand-agent-acp-shim"
        ),
        protocol="acp",
        requires_env=["GEMINI_API_KEY"],
        install_timeout=1200,
        default_model="google/gemini-2.5-flash",
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
            f"    {_apt_install('curl', 'ca-certificates', 'git')}; "
            "  elif command -v dnf >/dev/null 2>&1; then "
            "    dnf -y --allowerasing install curl ca-certificates git >/dev/null 2>&1; "
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
            # Pin the OpenHands CLI source so the agent workflow cannot drift
            # with GitHub main; only override the buggy sdk/tools 1.21.0 pins.
            # SDK 1.22.x restores default-to-UNKNOWN for the synthetic
            # `security_risk` tool field without the API drift seen in 1.26.x.
            f"printf 'openhands-sdk=={_OPENHANDS_SDK_VERSION}\\n"
            f"openhands-tools=={_OPENHANDS_TOOLS_VERSION}\\n' "
            "> /tmp/oh-sdk-overrides.txt && "
            "uv tool install --force --refresh "
            "--overrides /tmp/oh-sdk-overrides.txt "
            "--from "
            f"'git+https://github.com/OpenHands/OpenHands-CLI.git@{_OPENHANDS_CLI_GIT_REV}' "
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
            # Write llm settings including base_url so the BenchFlow LiteLLM
            # gateway (LLM_BASE_URL) is honored. OpenHands' --override-with-envs
            # does not reliably apply base_url; it is omitted when unset.
            '{ printf \'{"llm":{"model":"%s","api_key":"%s"\' '
            '"$LLM_MODEL" "$LLM_API_KEY"; '
            'if [ -n "$LLM_BASE_URL" ]; then '
            'printf \',"base_url":"%s"\' "$LLM_BASE_URL"; fi; '
            'if [ -n "$LLM_API_VERSION" ]; then '
            'printf \',"api_version":"%s"\' "$LLM_API_VERSION"; fi; '
            "printf '}}'; } > ~/.openhands/agent_settings.json && "
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
        disallow_web_tools_owned_paths=["$HOME/.openhands"],
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
    "pi": "pi-acp",
    "openclaw": "openclaw",
    "openhands": "openhands",
    "oh": "openhands",
    "harvey-lab": "harvey-lab-harness",
    "deepagents": "deepagents",
    "mini-computer": "mini-computer-agent",
    "cua": "mini-computer-agent",
}

VALID_PROTOCOLS = {"acp", "acpx", "session-factory"}

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
    """Parse an agent spec like 'acp/claude-agent-acp', 'acpx/claude', or 'claude'.

    Returns (protocol, agent_name) with alias resolution.
    Bare names default to 'acp' protocol.
    The 'acpx' protocol routes through the acpx CLI (https://acpx.sh/).
    """
    if "/" in spec:
        protocol, name = spec.split("/", 1)
    else:
        protocol, name = "acp", spec

    name = AGENT_ALIASES.get(name, name)
    return protocol, name


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
        requires_env=config.requires_env,
        description=f"{config.description} (via acpx)",
        skill_paths=config.skill_paths,
        install_timeout=config.install_timeout,
        default_model=config.default_model,
        api_protocol=config.api_protocol,
        env_mapping=config.env_mapping,
        credential_files=config.credential_files,
        home_dirs=config.home_dirs,
        acp_model_format=config.acp_model_format,
        subscription_auth=config.subscription_auth,
        supports_acp_set_model=config.supports_acp_set_model,
        acp_model_config_id=config.acp_model_config_id,
        acp_effort_config_id=config.acp_effort_config_id,
        disallow_web_tools_setup_cmd=config.disallow_web_tools_setup_cmd,
        disallow_web_tools_owned_paths=config.disallow_web_tools_owned_paths,
        disallow_web_tools_launch_suffix=config.disallow_web_tools_launch_suffix,
    )


def resolve_agent(spec: str) -> AgentConfig:
    """Resolve an agent spec to an AgentConfig.

    Supports: bare name, alias, protocol/name, acpx/name.
    Raises KeyError with suggestions for unknown agents.
    """
    protocol, name = parse_agent_spec(spec)

    if protocol not in VALID_PROTOCOLS:
        raise KeyError(
            f"Unknown protocol: {protocol!r}. Valid: {', '.join(sorted(VALID_PROTOCOLS))}"
        )

    # An already-resolved acpx runtime key (e.g. "acpx:claude-agent-acp")
    # round-trips: parse_agent_spec leaves it whole under the default "acp"
    # protocol and it lives in AGENTS. See the ACPX_KEY_PREFIX contract.
    if protocol == "acp" and name in AGENTS:
        return AGENTS[name]

    if name not in AGENTS:
        from difflib import get_close_matches

        close = get_close_matches(name, list(AGENTS.keys()), n=1, cutoff=0.6)
        if close:
            raise KeyError(f"Unknown agent: {name!r}. Did you mean: {close[0]!r}?")
        raise KeyError(
            f"Unknown agent: {name!r}. Available: {', '.join(sorted(AGENTS.keys()))}"
        )

    config = AGENTS[name]
    if protocol == "acpx":
        return _acpx_wrap(config)
    return config


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

    Unknown agents are returned unchanged so callers can still surface their
    own diagnostics (raw-command fallback).
    """
    try:
        config = resolve_agent(spec)
    except KeyError:
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
    requires_env: list[str] | None = None,
    description: str = "",
    skill_paths: list[str] | None = None,
    install_timeout: int = 900,
    default_model: str = "",
    api_protocol: str = "",
    env_mapping: dict[str, str] | None = None,
    credential_files: list[CredentialFile] | None = None,
    home_dirs: list[str] | None = None,
    subscription_auth: SubscriptionAuth | None = None,
    acp_model_format: str = "bare",
    supports_acp_set_model: bool = True,
    acp_model_config_id: str = "",
    acp_effort_config_id: str = "",
    disallow_web_tools_setup_cmd: str = "",
    disallow_web_tools_owned_paths: list[str] | None = None,
    disallow_web_tools_launch_suffix: str = "",
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
        default_model=default_model,
        api_protocol=api_protocol,
        env_mapping=env_mapping or {},
        credential_files=credential_files or [],
        home_dirs=home_dirs or [],
        subscription_auth=subscription_auth,
        acp_model_format=acp_model_format,
        supports_acp_set_model=supports_acp_set_model,
        acp_model_config_id=acp_model_config_id,
        acp_effort_config_id=acp_effort_config_id,
        disallow_web_tools_setup_cmd=disallow_web_tools_setup_cmd,
        disallow_web_tools_owned_paths=disallow_web_tools_owned_paths or [],
        disallow_web_tools_launch_suffix=disallow_web_tools_launch_suffix,
    )
    AGENTS[name] = config
    AGENT_INSTALLERS[name] = install_cmd
    AGENT_LAUNCH[name] = launch_cmd
    return config
