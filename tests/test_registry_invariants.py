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

import pytest

from benchflow.agents.providers import (
    PROVIDERS,
    find_provider,
    resolve_auth_env,
)
from benchflow.agents.registry import (
    AGENT_INSTALLERS,
    AGENT_LAUNCH,
    AGENTS,
)

VALID_AGENT_PROTOCOLS = {"acp", "cli"}
# Empty api_protocol is valid for agents (they infer from the model name at
# runtime); providers must always declare an explicit protocol.
VALID_API_PROTOCOLS = {"", "anthropic-messages", "openai-completions"}
VALID_PROVIDER_API_PROTOCOLS = VALID_API_PROTOCOLS - {""}
VALID_AUTH_TYPES = {"api_key", "adc", "none"}


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
    assert isinstance(cfg.install_timeout, int) and cfg.install_timeout > 0


@pytest.mark.parametrize("name,cfg", AGENTS.items(), ids=list(AGENTS.keys()))
def test_agent_collection_invariants(name, cfg):
    """Lists/dicts on AgentConfig must follow SDK conventions.

    - requires_env: list of non-empty strings (env var names)
    - env_mapping keys: must start with BENCHFLOW_PROVIDER_ (SDK propagates these)
    - env_mapping values: non-empty strings (agent-native env var names)
    - skill_paths: $HOME/ or $WORKSPACE/ relative
    - home_dirs: dot-prefixed (copied under sandbox $HOME)
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


# ── Cross-cutting derived contracts ─────────────────────────────────────────


@pytest.mark.parametrize(
    "model,expected",
    [
        ("google-vertex/gemini-2.5-pro", "google-vertex"),
        ("anthropic-vertex/claude-sonnet-4-6", "anthropic-vertex"),
        ("zai/glm-5", "zai"),
        ("vllm/local-model", "vllm"),
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
