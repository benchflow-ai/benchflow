"""Schema invariants for the agent and provider registries.

These tests parametrize over every entry in ``AGENTS`` and ``PROVIDERS``,
so adding a new agent or provider automatically gets contract coverage —
no test changes required.

Read this file when adding a new registry entry: it documents the
required shape of ``AgentConfig`` and ``ProviderConfig`` and the
implicit conventions the SDK relies on (e.g. ``env_mapping`` keys must
start with ``BENCHFLOW_PROVIDER_``).
"""

import re
import shutil
import subprocess

import pytest

from benchflow.agents.providers import (
    PROVIDERS,
    find_provider,
    find_provider_for_bare_model,
    resolve_auth_env,
)
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
    _js_agent_install,
)

VALID_AGENT_PROTOCOLS = {"acp", "cli", "session-factory"}
# Empty api_protocol is valid for agents (they infer from the model name at
# runtime); providers must always declare an explicit protocol.
VALID_API_PROTOCOLS = {
    "",
    "anthropic-messages",
    "openai-responses",
    "openai-completions",
}
VALID_PROVIDER_API_PROTOCOLS = VALID_API_PROTOCOLS - {""}
VALID_AUTH_TYPES = {"api_key", "adc", "aws", "none"}
VALID_ACP_MODEL_FORMATS = {
    "bare",
    "provider/model",
    "registered-provider/model",
}
JS_ACP_AGENTS = {
    name
    for name, cfg in AGENTS.items()
    if cfg.protocol == "acp" and "npm install" in cfg.install_cmd
}


# ── AgentConfig invariants ──────────────────────────────────────────────────


@pytest.mark.parametrize("name,cfg", AGENTS.items(), ids=list(AGENTS.keys()))
def test_agent_field_shapes(name, cfg):
    """Per-field shape: name matches key, required strings non-empty, enums valid."""
    assert cfg.name == name, f"AGENTS[{name!r}].name is {cfg.name!r}"
    assert cfg.install_cmd, "install_cmd must be set"
    assert cfg.launch_cmd, "launch_cmd must be set"
    assert cfg.protocol in VALID_AGENT_PROTOCOLS, (
        f"protocol={cfg.protocol!r} not in {VALID_AGENT_PROTOCOLS}"
    )
    assert cfg.api_protocol in VALID_API_PROTOCOLS, (
        f"api_protocol={cfg.api_protocol!r} not in {VALID_API_PROTOCOLS}"
    )
    assert cfg.acp_model_format in VALID_ACP_MODEL_FORMATS, (
        f"acp_model_format={cfg.acp_model_format!r} not in {VALID_ACP_MODEL_FORMATS}"
    )
    assert isinstance(cfg.install_timeout, int) and cfg.install_timeout > 0
    assert isinstance(cfg.supports_acp_set_model, bool)
    assert isinstance(cfg.acp_model_config_id, str)
    assert isinstance(cfg.acp_effort_config_id, str)


@pytest.mark.parametrize("name,cfg", AGENTS.items(), ids=list(AGENTS.keys()))
def test_agent_collection_invariants(name, cfg):
    """Lists/dicts on AgentConfig must follow SDK conventions.

    - requires_env: list of non-empty strings (env var names)
    - env_mapping keys: must start with BENCHFLOW_PROVIDER_ (SDK propagates these)
    - env_mapping values: non-empty strings (agent-native env var names)
    - skill_paths: $HOME/ or $WORKSPACE/ relative
    - home_dirs: dot-prefixed (copied under sandbox $HOME)
    - disallow_web_tools_owned_paths: $HOME/ relative
    """
    assert all(isinstance(k, str) and k for k in cfg.requires_env)
    for key, val in cfg.env_mapping.items():
        assert key.startswith("BENCHFLOW_PROVIDER_"), (
            f"env_mapping key {key!r} must start with BENCHFLOW_PROVIDER_"
        )
        assert isinstance(val, str) and val
    for sp in cfg.skill_paths:
        assert sp.startswith(("$HOME/", "$WORKSPACE/")), (
            f"skill_path {sp!r} must start with $HOME/ or $WORKSPACE/"
        )
    for d in cfg.home_dirs:
        assert d.startswith("."), f"home_dirs entry {d!r} must start with '.'"
    for path in cfg.disallow_web_tools_owned_paths:
        assert path.startswith("$HOME/"), (
            f"disallow_web_tools_owned_paths entry {path!r} must start with $HOME/"
        )


@pytest.mark.parametrize("name,cfg", AGENTS.items(), ids=list(AGENTS.keys()))
def test_agent_install_cmd_targets_shared_paths(name, cfg):
    """Installed binaries must land in shared prefixes, not a root-only home.

    setup_sandbox_user() no longer recursively copies /root/.nvm or
    /root/.local/bin into the sandbox home. If an install_cmd placed its
    binary there, the sandbox user would silently lose access to the agent.
    """
    forbidden_binary_prefixes = ("/root/.nvm/", "/root/.local/bin/", "$HOME/.nvm/")
    for prefix in forbidden_binary_prefixes:
        assert prefix not in cfg.install_cmd, (
            f"{name!r} install_cmd writes under {prefix!r}; use /usr/local/bin "
            f"or another shared prefix so the sandbox user inherits the tool"
        )


@pytest.mark.parametrize("name", sorted(JS_ACP_AGENTS))
def test_js_acp_agents_use_isolated_node_runtime(name):
    """JS agents must not mutate task-owned Node/npm installations."""
    install_cmd = AGENTS[name].install_cmd
    launch_cmd = AGENTS[name].launch_cmd

    assert "/opt/benchflow/node" in install_cmd
    # Node >=22.19 is required by current openclaw (JS agents install @latest);
    # assert the floor, not a brittle exact pin (BF-10).
    pin = re.search(r"BF_NODE_VERSION=(\d+)\.(\d+)\.\d+", install_cmd)
    assert pin, "BF_NODE_VERSION pin missing from JS agent bootstrap"
    major, minor = int(pin.group(1)), int(pin.group(2))
    assert (major, minor) >= (22, 19), (
        f"pinned node {pin.group(0)} is below openclaw's >=22.19 floor"
    )
    assert "/opt/benchflow/js-agents" in install_cmd
    assert "/opt/benchflow/bin" in install_cmd
    assert "--prefix /opt/benchflow/js-agents" in install_cmd
    assert "/opt/benchflow/bin" in launch_cmd
    assert "/opt/benchflow/js-agents/bin:/opt/benchflow/node/bin:$PATH" in install_cmd
    assert (
        "exec /opt/benchflow/node/bin/node /opt/benchflow/js-agents/bin/" in install_cmd
    )
    # The launched program is the isolated bin — directly, or (codex-acp) after a
    # self-config-writing prefix ending in `; exec <bin>` (writes ~/.codex/auth.json).
    launched = launch_cmd.rsplit("; exec ", 1)[-1]
    assert launched.split()[0].startswith("/opt/benchflow/bin/")
    assert launched.split()[0] not in {"export", "env"}
    assert not launched.startswith("PATH=")

    forbidden_fragments = [
        'export PATH="/opt/benchflow/node/bin:/opt/benchflow/js-agents/bin:$PATH"',
        "exec /opt/benchflow/js-agents/bin/",
        "deb.nodesource.com",
        "apt-get install -y -qq nodejs",
        "apt-get install -y nodejs",
        "/usr/bin/node",
        "/usr/bin/npm",
        "/usr/bin/npx",
        "/usr/local/bin/node",
        "/usr/local/bin/npm",
        "/usr/local/bin/npx",
        "/usr/local/bin/pi-acp-launcher",
        "/usr/local/bin/openclaw-acp-shim",
        "command -v node",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in install_cmd, (
            f"{name!r} JS agent install must not mutate task Node/npm via {fragment!r}"
        )


# Bash-isms not supported by dash (Ubuntu/Debian's /bin/sh). The sandbox
# Docker/Daytona exec paths invoke ``sh -c install_cmd``; if /bin/sh is dash
# (ubuntu:24.04 base), any of these aborts the install on line 1. See #341.
_BASH_ISM_PATTERNS = {
    "set -o pipefail": re.compile(r"\bpipefail\b"),
    "[[ ... ]]": re.compile(r"\[\[\s"),
    "<<< (here-string)": re.compile(r"<<<"),
    "$(( ... )) (arith)": re.compile(r"\$\(\("),
    "function name() {}": re.compile(r"(?:^|[\s;&|])function\s+\w"),
    "declare/local -": re.compile(r"\b(?:declare|local)\s+-"),
    "<(...) (proc subst)": re.compile(r"[^<]<\("),
}


@pytest.mark.parametrize("name,cfg", AGENTS.items(), ids=list(AGENTS.keys()))
def test_agent_install_cmd_is_posix_sh_compatible(name, cfg):
    """Install runs under ``sh -c`` (dash on Ubuntu); reject bash-isms.

    Regression guard for #341: ``set -o pipefail`` at the top of the Node
    bootstrap aborted the install on line 1 when /bin/sh was dash.
    """
    for label, pattern in _BASH_ISM_PATTERNS.items():
        assert not pattern.search(cfg.install_cmd), (
            f"{name!r} install_cmd contains bash-ism {label!r}; the sandbox "
            "runs install_cmd under ``sh -c`` and /bin/sh is dash on Ubuntu "
            "(see #341). Rewrite using POSIX sh, or run the script with "
            "/bin/bash explicitly."
        )


@pytest.mark.skipif(
    shutil.which("dash") is None,
    reason="dash not installed on host; covered in CI",
)
@pytest.mark.parametrize("name,cfg", AGENTS.items(), ids=list(AGENTS.keys()))
def test_agent_install_cmd_parses_under_dash(name, cfg):
    """``dash -n`` syntax-checks every install_cmd against the real shell.

    Catches bash-isms the regex sweep misses (e.g. brace expansion edge cases,
    invalid redirections under dash). Complements the regex check above.
    """
    result = subprocess.run(
        ["dash", "-n"],
        input=cfg.install_cmd,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, (
        f"{name!r} install_cmd failed dash syntax check:\n{result.stderr}"
    )


def test_js_agent_install_respects_explicit_npm_package_specs():
    """Guards v0.5-integration@27752fa against moving npm latest installs."""
    pinned_cmd = _js_agent_install("gemini", "@google/gemini-cli@0.42.0")
    dist_tag_cmd = _js_agent_install("test-agent", "some-agent@next")
    scoped_default_cmd = _js_agent_install(
        "codex-acp", "@agentclientprotocol/codex-acp"
    )
    scoped_dist_tag_cmd = _js_agent_install("pi", "@mariozechner/pi-coding-agent@next")
    default_cmd = _js_agent_install("test-agent", "some-agent")

    assert "@google/gemini-cli@0.42.0" in pinned_cmd
    assert "@google/gemini-cli@latest" not in pinned_cmd
    assert "some-agent@next" in dist_tag_cmd
    assert "some-agent@next@latest" not in dist_tag_cmd
    assert "@agentclientprotocol/codex-acp@latest" in scoped_default_cmd
    assert "@mariozechner/pi-coding-agent@next" in scoped_dist_tag_cmd
    assert "@mariozechner/pi-coding-agent@next@latest" not in scoped_dist_tag_cmd
    assert "some-agent@latest" in default_cmd


def test_gemini_cli_install_is_pinned():
    """Guards v0.5-integration@27752fa against Daytona installing moving latest."""
    install_cmd = AGENTS["gemini"].install_cmd
    assert "@google/gemini-cli@0.42.0" in install_cmd
    assert "@google/gemini-cli@latest" not in install_cmd
    assert "[ -x /opt/benchflow/js-agents/bin/gemini ] ||" not in install_cmd


@pytest.mark.parametrize("name", sorted(JS_ACP_AGENTS))
def test_js_acp_agent_npm_failures_are_visible(name):
    """Npm stderr should reach agent/install-stdout.txt on install failure."""
    install_cmd = AGENTS[name].install_cmd
    assert "npm install" in install_cmd
    assert not re.search(r"npm install[^;&|)]*(?:2?>|&>)", install_cmd), (
        f"{name!r} install redirects npm output before BenchFlow can log it"
    )


@pytest.mark.parametrize("name", sorted(JS_ACP_AGENTS))
def test_js_acp_agent_install_commands_are_posix_sh_compatible(name):
    """Guards the fix from PR #423 against the regression where Docker
    JS-agent installs failed before launch because install_cmd used bash-only
    `set -o pipefail` under DockerSandbox.exec()'s POSIX `sh -c`.
    """
    install_cmd = AGENTS[name].install_cmd
    assert "set -o pipefail" not in install_cmd


@pytest.mark.parametrize("name,cfg", AGENTS.items(), ids=list(AGENTS.keys()))
def test_agent_credential_and_subscription_auth(name, cfg):
    """Optional credential_files and subscription_auth structures."""
    for cf in cfg.credential_files:
        assert cf.path, "CredentialFile.path must be non-empty"
        assert cf.env_source, "CredentialFile.env_source must be non-empty"
    if cfg.subscription_auth is not None:
        sa = cfg.subscription_auth
        assert sa.replaces_env, "SubscriptionAuth.replaces_env must be set"
        assert sa.detect_file, "SubscriptionAuth.detect_file must be set"
        for f in sa.files:
            assert f.host_path, "HostAuthFile.host_path must be set"
            assert f.container_path, "HostAuthFile.container_path must be set"


def test_agent_derived_dicts_in_sync():
    """AGENT_INSTALLERS and AGENT_LAUNCH are derived from AGENTS — must match."""
    assert set(AGENT_INSTALLERS) == set(AGENTS) == set(AGENT_LAUNCH)
    for name, cfg in AGENTS.items():
        assert AGENT_INSTALLERS[name] == cfg.install_cmd
        assert AGENT_LAUNCH[name] == cfg.launch_cmd


def test_agent_negative_config_invariants():
    """Specific agents must NOT have certain features configured.

    Tripwire for accidental config bleed (e.g. openclaw silently gaining
    credential_files because someone copy-pasted from codex). Positive
    per-agent assertions live in test_agent_registry.py /
    test_subscription_auth.py; this is the dedicated negative side.
    """
    no_credential_files = {"claude-agent-acp", "openclaw"}
    no_subscription_auth = {"openclaw", "pi-acp"}
    no_env_mapping = {"openclaw", "pi-acp"}

    for name in no_credential_files:
        assert AGENTS[name].credential_files == [], (
            f"{name!r} should not declare credential_files"
        )
    for name in no_subscription_auth:
        assert AGENTS[name].subscription_auth is None, (
            f"{name!r} should not declare subscription_auth"
        )
    for name in no_env_mapping:
        assert AGENTS[name].env_mapping == {}, (
            f"{name!r} should not declare env_mapping"
        )


def test_agent_api_protocol_has_provider_endpoint():
    """If an agent declares api_protocol, at least one provider must support it.

    The SDK propagates the agent's native api_protocol to provider endpoint
    selection (see sdk.py: resolve_base_url protocol arg). Without a provider
    that exposes a matching endpoint, multi-protocol routing silently uses the
    wrong URL.
    """
    available = set()
    for cfg in PROVIDERS.values():
        available.add(cfg.api_protocol)
        available.update(cfg.endpoints)
    for name, cfg in AGENTS.items():
        if cfg.api_protocol:
            assert cfg.api_protocol in available, (
                f"agent {name!r} api_protocol={cfg.api_protocol!r} "
                f"has no provider endpoint (available: {sorted(available)})"
            )


# ── ProviderConfig invariants ───────────────────────────────────────────────


@pytest.mark.parametrize("name,cfg", PROVIDERS.items(), ids=list(PROVIDERS.keys()))
def test_provider_field_shapes(name, cfg):
    """Per-field shape: name, api_protocol, auth_type, and auth_env consistency."""
    assert cfg.name == name
    assert cfg.api_protocol in VALID_PROVIDER_API_PROTOCOLS, (
        f"api_protocol={cfg.api_protocol!r} not in {VALID_PROVIDER_API_PROTOCOLS}"
    )
    assert cfg.auth_type in VALID_AUTH_TYPES, (
        f"auth_type={cfg.auth_type!r} not in {VALID_AUTH_TYPES}"
    )
    if cfg.auth_type == "api_key":
        assert cfg.auth_env, f"api_key provider {name!r} must set auth_env"
    else:
        assert cfg.auth_env is None, (
            f"{cfg.auth_type} provider {name!r} should not set auth_env"
        )


@pytest.mark.parametrize("name,cfg", PROVIDERS.items(), ids=list(PROVIDERS.keys()))
def test_provider_url_params_and_endpoints(name, cfg):
    """url_params <-> base_url consistency, plus endpoint protocol validity.

    - Every {placeholder} in base_url must have a url_params entry.
    - Every url_params key must actually appear in base_url or some endpoint
      (catches dead url_params from copy-paste mistakes).
    - Every endpoints key must be a valid api_protocol.
    """
    all_urls = " ".join([cfg.base_url, *cfg.endpoints.values()])
    placeholders_in_urls = set(re.findall(r"\{(\w+)\}", all_urls))
    for ph in placeholders_in_urls:
        assert ph in cfg.url_params, (
            f"{name!r}: url has {{{ph}}} but no url_params entry"
        )
    for key in cfg.url_params:
        assert key in placeholders_in_urls, (
            f"{name!r}: url_params[{key!r}] is not referenced in any URL"
        )
    for proto in cfg.endpoints:
        assert proto in VALID_PROVIDER_API_PROTOCOLS, (
            f"endpoint protocol {proto!r} not in {VALID_PROVIDER_API_PROTOCOLS}"
        )


@pytest.mark.parametrize("name,cfg", PROVIDERS.items(), ids=list(PROVIDERS.keys()))
def test_provider_models_and_credentials(name, cfg):
    """Provider models have required keys and unique ids; credential_files well-formed."""
    ids = [m.get("id") for m in cfg.models]
    for m in cfg.models:
        assert m.get("id"), f"model entry missing id: {m}"
    assert len(ids) == len(set(ids)), f"duplicate model ids in provider {name!r}: {ids}"
    for cf in cfg.credential_files:
        assert cf.get("path"), f"credential_files entry missing path: {cf}"
        assert cf.get("env_source"), f"credential_files entry missing env_source: {cf}"


@pytest.mark.parametrize("name,cfg", PROVIDERS.items(), ids=list(PROVIDERS.keys()))
def test_provider_model_prefixes_shape(name, cfg):
    """model_prefixes tokens are non-empty, lowercase, stripped, and prefix-free.

    ``find_provider_for_bare_model`` matches tokens against bare (already
    prefix-stripped) model ids, so a token containing ``/`` could never match.
    """
    for token in cfg.model_prefixes:
        assert token and isinstance(token, str), (
            f"{name!r}: model_prefixes entry must be a non-empty string: {token!r}"
        )
        assert token == token.strip().lower(), (
            f"{name!r}: model_prefixes token {token!r} must be lowercase/stripped"
        )
        assert "/" not in token, (
            f"{name!r}: model_prefixes token {token!r} must be a bare id family, "
            "not a provider/ prefix"
        )


def test_provider_model_prefixes_unique_and_resolvable():
    """Tokens are unique across providers and round-trip to their owner.

    Uniqueness keeps ``find_provider_for_bare_model``'s longest-token-wins
    resolution independent of registry declaration order; the round-trip
    check pins that every declared token actually routes to its provider.
    """
    owners: dict[str, str] = {}
    for name, cfg in PROVIDERS.items():
        for token in cfg.model_prefixes:
            assert token not in owners, (
                f"model_prefixes token {token!r} declared by both "
                f"{owners[token]!r} and {name!r}; resolution would depend on "
                "registry order"
            )
            owners[token] = name
            result = find_provider_for_bare_model(token)
            assert result is not None and result[0] == name, (
                f"token {token!r} does not resolve back to provider {name!r}"
            )


# ── Cross-cutting derived contracts ─────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("google-vertex/gemini-2.5-pro", "google-vertex"),
        ("anthropic-vertex/claude-sonnet-4-6", "anthropic-vertex"),
        ("azure-foundry-openai/gpt-5.5", "azure-foundry-openai"),
        ("azure-foundry-anthropic/claude-opus-4-5", "azure-foundry-anthropic"),
        ("aws-bedrock/openai.gpt-oss-20b-1:0", "aws-bedrock"),
        ("github-models/openai/gpt-4.1-mini", "github-models"),
        ("zai/glm-5", "zai"),
        ("vllm/local-model", "vllm"),
        ("kimi/kimi-k2.6", "kimi"),
        ("qwen-dashscope/qwen3.6-max-preview", "qwen-dashscope"),
        ("doubao-seed-2-pro/ep-test", "doubao-seed-2-pro"),
    ],
)
def test_find_provider_resolves_known_prefixes(model, expected):
    """Sanity check that prefix routing works for every registered provider."""
    result = find_provider(model)
    assert result is not None, f"find_provider({model!r}) returned None"
    assert result[0] == expected


def test_resolve_auth_env_matches_provider_auth_type():
    """API-key providers return their auth_env; ADC/none return None."""
    for name, cfg in PROVIDERS.items():
        result = resolve_auth_env(f"{name}/test-model")
        if cfg.auth_type == "api_key":
            assert result == cfg.auth_env, (
                f"{name!r}: resolve_auth_env returned {result!r}, expected {cfg.auth_env!r}"
            )
        else:
            assert result is None, (
                f"{name!r} ({cfg.auth_type}): resolve_auth_env should return None, got {result!r}"
            )


def test_harvey_maps_provider_env_to_openai_compatible_adapter():
    """Harvey LAB's OpenAI-compatible adapter reads OPENAI_BASE_URL/OPENAI_API_KEY;
    on the proxy path those only exist as BENCHFLOW_PROVIDER_*. Without this
    mapping the adapter had no endpoint/key and the harness loop ended on turn 0
    with zero LLM activity. Guard the mapping so deepseek / gpt-5.4-mini-gateway /
    every benchflow-* alias can route.
    """
    from benchflow.agents.registry import AGENTS

    em = AGENTS["harvey-lab-harness"].env_mapping
    assert em.get("BENCHFLOW_PROVIDER_BASE_URL") == "OPENAI_BASE_URL", em
    assert em.get("BENCHFLOW_PROVIDER_API_KEY") == "OPENAI_API_KEY", em
