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
- ``protocol``           "acp" (default), "cli", or "session-factory".
                         Almost always "acp".
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

# OpenCode-family proxy provider id. OpenCode hard-codes the OpenAI *Responses*
# API for the built-in ``openai`` provider id (its ``getModel`` calls
# ``provider.responses(id)``), which the LiteLLM gateway/DeepSeek cannot serve —
# so the gateway alias must be registered under a *separate* provider id that
# OpenCode routes through the chat-completions path. Shared with
# ``benchflow.acp.runtime._format_acp_model`` so set_model targets the same id.
OPENCODE_PROXY_PROVIDER_ID = "benchflow"
_OPENHANDS_CLI_GIT_REV = "3ca17446c5d9c1e35e054803478a3501ec251ecf"
_OPENHANDS_SDK_VERSION = "1.22.1"
_OPENHANDS_TOOLS_VERSION = "1.22.1"
_JS_AGENT_PATH = (
    f"{_BENCHFLOW_BIN_PREFIX}:{_BENCHFLOW_JS_AGENT_PREFIX}/bin:"
    f"{_BENCHFLOW_NODE_PREFIX}/bin:$PATH"
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


# Path to the openclaw ACP shim script
_OPENCLAW_SHIM = (Path(__file__).parent / "openclaw_acp_shim.py").read_text()

# Path to the Pi launch wrapper (bridges BENCHFLOW_PROVIDER_* → Pi config)
_PI_LAUNCHER = (Path(__file__).parent / "pi_acp_launcher.py").read_text()

# Path to the Harvey LAB ACP shim (runs Harvey LAB harness as an ACP agent)
_HARVEY_LAB_SHIM = (Path(__file__).parent / "harvey_lab_acp_shim.py").read_text()

# Path to the deepagents ACP shim (runs LangChain's create_deep_agent as an ACP agent)
_DEEPAGENTS_SHIM = (Path(__file__).parent / "deepagents_acp_shim.py").read_text()


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


# OpenCode-family proxy fix: OpenCode and its MiMo fork validate provider/model
# ids against the models.dev catalog and reject the synthetic
# ``openai/benchflow-<alias>`` that BenchFlow's LiteLLM proxy serves the model
# under (ProviderModelNotFoundError -> zero requests -> no
# ``trajectory/llm_trajectory.jsonl``, which ``benchflow-experiment-review``
# requires). Registering the alias under ``provider.openai.models`` (with the
# gateway baseURL) bypasses that catalog check so the agent accepts the id and
# routes through the proxy.
def _opencode_family_proxy_wrapper_install(binary: str, config_path: str) -> str:
    """Install ``/opt/benchflow/bin/<binary>-proxy``: a thin wrapper that, in
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
    register_py = "\n".join(
        [
            "import json, os, pathlib",
            'alias = os.environ.get("BENCHFLOW_LITELLM_MODEL_ALIAS", "").strip()',
            "if alias:",
            '    home = os.environ.get("BENCHFLOW_AGENT_HOME", "").strip() or os.path.expanduser("~")',
            f"    p = pathlib.Path(home) / {config_relpath!r}",
            "    p.parent.mkdir(parents=True, exist_ok=True)",
            "    d = json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}",
            # Dedicated provider id (see docstring) → chat completions, not the
            # Responses API the built-in ``openai`` id is hard-coded to.
            f'    prov = d.setdefault("provider", {{}}).setdefault({provider_id!r}, {{}})',
            '    prov["npm"] = "@ai-sdk/openai-compatible"',
            '    prov.setdefault("name", "BenchFlow Gateway")',
            '    opts = prov.setdefault("options", {})',
            '    base = os.environ.get("OPENAI_BASE_URL", "").strip()',
            "    if base:",
            '        opts["baseURL"] = base',
            '    key = os.environ.get("OPENAI_API_KEY", "").strip()',
            "    if key:",
            '        opts["apiKey"] = key',
            '    prov.setdefault("models", {}).setdefault(alias, {"name": alias})',
            f'    d["small_model"] = "{provider_id}/" + alias',
            '    p.write_text(json.dumps(d, indent=2) + "\\n")',
        ]
    )
    wrapper = (
        "#!/bin/sh\n"
        # Proxy mode only. Fail LOUD: if registration errors (malformed existing
        # config, unwritable path), do NOT launch — the agent would otherwise get
        # set_model "<provider>/<alias>" for a model now missing from its config
        # and hit ProviderModelNotFoundError with nothing explaining why. A hard
        # exit surfaces the cause instead of a silent broken proxy path.
        'if [ -n "$BENCHFLOW_LITELLM_MODEL_ALIAS" ]; then\n'
        "  if ! python3 - <<'PYEOF'\n"
        f"{register_py}\n"
        "PYEOF\n"
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
    b64 = base64.b64encode(wrapper.encode()).decode()
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


@dataclass
class AgentConfig:
    """Configuration for a supported agent."""

    name: str
    install_cmd: str
    launch_cmd: str
    protocol: str = "acp"  # "acp", "cli", or "session-factory"
    session_factory: str = ""
    # Non-ACP only. When protocol == "session-factory", this is a
    # "module:callable" entrypoint that builds an Agent Protocol object.
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


# Agent registry — all supported agents
AGENTS: dict[str, AgentConfig] = {
    "claude-agent-acp": AgentConfig(
        name="claude-agent-acp",
        description="Claude Code via ACP (Anthropic's Agent Client Protocol)",
        skill_paths=["$HOME/.claude/skills"],
        home_dirs=[".claude"],
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
        # Self-write ~/.codex/auth.json from OPENAI_API_KEY in the launcher itself,
        # ONLY when the key is set (so subscription/host-auth mode is untouched),
        # instead of relying on core's credential_files writer. This makes the
        # decoupled manifest self-contained — like mimo/opencode — and is
        # byte-identical to the former credential_files template
        # ({"OPENAI_API_KEY": "<key>"}) and keeps the old 0600 secret mode.
        # `exec` so signals/PID reach codex.
        launch_cmd=(
            'h="${BENCHFLOW_AGENT_HOME:-$HOME}"; '
            'if [ -n "$OPENAI_API_KEY" ]; then mkdir -p "$h/.codex" && '
            'printf \'{"OPENAI_API_KEY": "%s"}\' "$OPENAI_API_KEY" '
            '> "$h/.codex/auth.json" && chmod 600 "$h/.codex/auth.json"; '
            "fi; exec "
            + _js_agent_launch(
                "codex-acp", "${OPENAI_BASE_URL:+-c openai_base_url=$OPENAI_BASE_URL}"
            )
        ),
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
        install_cmd=(
            _js_agent_install("opencode", "opencode-ai")
            + " && "
            + _opencode_family_proxy_wrapper_install(
                "opencode", ".config/opencode/opencode.json"
            )
        ),
        launch_cmd=f"{_BENCHFLOW_BIN_PREFIX}/opencode-proxy acp",
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
    "mimo": AgentConfig(
        name="mimo",
        description="MiMo Code via ACP — Xiaomi's OpenCode fork (TypeScript)",
        skill_paths=["$HOME/.mimocode/skills"],
        home_dirs=[".mimocode", ".config/mimocode"],
        install_cmd=(
            _js_agent_install("mimo", "@mimo-ai/cli@0.1.4")
            + " && "
            + _opencode_family_proxy_wrapper_install(
                "mimo", ".config/mimocode/mimocode.json"
            )
        ),
        launch_cmd=f"{_BENCHFLOW_BIN_PREFIX}/mimo-proxy acp",
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
        acp_model_format="provider/model",
        # MiMo Code is an OpenCode fork: `mimo acp` reports agentInfo.name="OpenCode"
        # and uses models.dev "provider/model" ids, so set_model must send e.g.
        # "benchflow/benchflow-<alias>" in proxy mode (the dedicated
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
        disallow_web_tools_setup_cmd=_json_settings_merge(
            "$BENCHFLOW_AGENT_HOME/.config/mimocode/mimocode.json",
            'd.setdefault("tools",{})["webfetch"]=False',
        ),
        disallow_web_tools_owned_paths=["$HOME/.config/mimocode"],
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
            # deepagents requires Python >=3.11, but task base images ship as low
            # as 3.6/3.8 (ubuntu:20.04, cached CI images), so a system-python venv
            # makes pip report "No matching distribution found for deepagents".
            # Provision a pinned interpreter with uv (same pattern as the OpenHands
            # install) so this works regardless of the base-image Python.
            "( command -v curl >/dev/null 2>&1 || "
            f"  {_apt_install('curl', 'ca-certificates')} ) && "
            "( command -v uv >/dev/null 2>&1 || "
            "  ( curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 ) ) && "
            'export PATH="$HOME/.local/bin:$PATH" && '
            "uv venv --python 3.12 /opt/benchflow/deepagents-venv && "
            # deepagents pulls in langchain/langchain-core/langchain-anthropic/
            # langchain-google-genai; langchain-openai is NOT a deepagents dep but
            # is required for the OpenAI-compatible deepseek-v4-pro chat model.
            "uv pip install -q --python /opt/benchflow/deepagents-venv/bin/python "
            "deepagents langchain-openai && "
            # Let the sandbox user traverse + execute the venv interpreter and the
            # uv-managed CPython it links to.
            "chmod -R a+rX /opt/benchflow/deepagents-venv && "
            "chmod o+x /root /root/.local /root/.local/share "
            "/root/.local/share/uv /root/.local/share/uv/python 2>/dev/null; "
            # Deploy ACP shim
            + _install_python_script(
                f"{_BENCHFLOW_BIN_PREFIX}/deepagents-acp-shim", _DEEPAGENTS_SHIM
            )
            # Verify deepagents actually imports through the pinned venv. Without
            # this, install rc reflects only the shim-deploy's trailing `chmod +x`
            # (the `;` above is intentionally non-fatal), so a failed `uv pip
            # install deepagents` — the exact failure this block fixes — would
            # report success and fail opaquely later at launch. Mirrors OpenHands'
            # `command -v openhands` verification tail.
            + " && /opt/benchflow/deepagents-venv/bin/python -c 'import deepagents' "
            ">/dev/null 2>&1"
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
        session_factory=config.session_factory,
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
    session_factory: str = "",
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
        session_factory=session_factory,
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


# --- Opt-in dual-source registry (agent-decoupling decision #7) ---------------
# Merge agents declared as <dir>/manifest.toml files under $BENCHFLOW_AGENTS_DIR
# into the registry. A NO-OP when the env var is unset, so a default import of
# core is byte-for-byte unchanged; the manifest path only activates on explicit
# opt-in. The import is deferred to here (end of module) on purpose: manifest.py
# imports AgentConfig from this module, so a top-level import would be circular,
# and the merge must run after AGENTS / AGENT_ALIASES / AGENT_INSTALLERS /
# AGENT_LAUNCH are fully built above.
from benchflow.agents.manifest import (  # noqa: E402
    register_env_manifest_agents as _register_env_manifest_agents,
)

_register_env_manifest_agents()
