"""The credential policy: what an agent may receive in its sandbox.

Resolution inherits every credential the host exports (``auto_inherit_env``)
so one helper can pick the run's route; the agent must see only that route.
One table per run, :class:`CredentialPolicy`, computed once at the end of
``resolve_agent_env`` and recorded in the env by variable names only, maps
(agent, route) to the exact credentials the agent may receive:

    - its selected login, with that login's aliases;
    - its own jointly required parts (a custom agent's client id and secret);
    - its own declared credential files' sources (a custom agent's auth.json);
    - for a raw command with no model, the credential slots its caller passed;
    - on the Gemini CLI's native Vertex route, the host ADC as a
      file only;
    - on a proxy route, the proxy key and nothing else of a model or login
      credential: upstream keys and Vertex ADC stay in the proxy's storage.
      A slot already holding the proxy key may be placed as a variable or a
      file (connect_as wires the proxy before staging).

An agent that routes its own model calls (``model_route = "agent"``)
is on the direct route whatever model label it carries: the label never
selects a provider login, so it receives only its own declared credentials.

Every placement reads only this table: launch variables
(:func:`agent_launch_env`), credential files written into the agent's home
(:meth:`CredentialPolicy.admitted_value`, the only value source for ADC,
auth.json, Muse login and agent config files), the Gemini Vertex settings and host login files. A credential
variable is one in :func:`host_credential_env_names`, which includes every
agent's declared requirements, so another agent's credentials are never
admitted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from benchflow.agents.env import (
    _LOGIN_ALIASES,
    _LOGIN_KEYS_BY_FAMILY,
    _MUSE_AGENT,
    _MUSE_AUTH_JSON_ENV,
    _SUBSCRIPTION_AUTH_MARKER,
    CREDENTIAL_POLICY_MARKER,
    VERTEX_ADC_ENV,
    _login_family,
    _select_login,
    inheritable_env_names,
    uses_direct_anthropic_endpoint,
    uses_native_subscription_auth,
    uses_native_vertex_login,
)
from benchflow.agents.registry import AGENTS, launcher_guard_env, routed_model

# Inherited or derived names that never hold a credential. Every other name
# the resolver can inherit counts as a credential, so a name added to the
# inheritance list is withheld from the agent until it is named here.
_NON_CREDENTIAL_ENV = frozenset(
    {
        "AWS_DEFAULT_REGION",
        "AWS_REGION",
        "AWS_REGION_NAME",
        "AZURE_API_ENDPOINT",
        "AZURE_API_VERSION",
        "AZURE_RESOURCE",
        "BENCHFLOW_BEDROCK_THINKING_EFFORT",
        "BENCHFLOW_PROVIDER_BASE_URL",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
        "LLM_BASE_URL",
        "OPENAI_BASE_URL",
    }
)
_LITELLM_MASTER_KEY_ENV = "BENCHFLOW_LITELLM_MASTER_KEY"
_SCRIPTED_AGENTS = frozenset({"oracle", "nop"})

PROXY = "proxy"
NATIVE = "native"
DIRECT = "direct"
NONE = "none"


def model_credential_env_names() -> frozenset[str]:
    """Every variable that can hold a model-provider or login credential."""
    from benchflow.agents.providers import PROVIDERS

    url_params = {
        name for cfg in PROVIDERS.values() for name in cfg.url_params.values()
    }
    names = set(inheritable_env_names()) - _NON_CREDENTIAL_ENV - url_params
    for cfg in PROVIDERS.values():
        if cfg.auth_env:
            names.add(cfg.auth_env)
        names.update(cf["env_source"] for cf in cfg.credential_files)
    for cfg in AGENTS.values():
        mapped = _mapped_provider_key(cfg)
        if mapped:
            names.add(mapped)
    names.update(
        {
            _LITELLM_MASTER_KEY_ENV,
            _MUSE_AUTH_JSON_ENV,
            VERTEX_ADC_ENV,
            "BENCHFLOW_PROVIDER_API_KEY",
            "LLM_API_KEY",
            # Muse's pay-as-you-go key: Muse prefers it over the login.
            "META_API_KEY",
            # BenchFlow's own service and judge credentials.
            "BENCHFLOW_TASKMD_JUDGE_AUTH_TOKEN",
            "DAYTONA_API_KEY",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
        }
    )
    for keys in _LOGIN_KEYS_BY_FAMILY.values():
        names.update(keys)
    return frozenset(names)


def host_credential_env_names() -> frozenset[str]:
    """Every credential variable: model and login credentials, plus every
    agent's declared requirements and credential-file sources.

    A declared requirement is a credential of the agent that declares it; it
    reaches only that agent, as its own joint part (never a different agent
    that shares the run's env).
    """
    names = set(model_credential_env_names())
    for cfg in AGENTS.values():
        names.update(getattr(cfg, "requires_env", None) or ())
        names.update(
            cf.env_source for cf in getattr(cfg, "credential_files", None) or ()
        )
    return frozenset(names)


def _mapped_provider_key(cfg: object) -> str | None:
    return (getattr(cfg, "env_mapping", None) or {}).get("BENCHFLOW_PROVIDER_API_KEY")


@dataclass(frozen=True)
class CredentialPolicy:
    """The credentials one agent may receive on its route, by variable name."""

    route: str
    # The selected login of a native or no-model route, selected name first,
    # then its aliases. Recorded on a proxy route too (for login labels) but
    # never admitted there.
    login: tuple[str, ...] = ()
    # This agent's own jointly required parts that are present.
    joint: tuple[str, ...] = ()
    # Sources of this agent's own declared credential files (a custom
    # agent's auth.json) that are not competing login slots. On a proxy
    # route only those that are not model credentials are admitted.
    files: tuple[str, ...] = ()
    # A raw command's explicitly passed credential slots (no model).
    raw: tuple[str, ...] = ()
    # Credentials this agent receives only as a file (the Gemini CLI's
    # native Vertex ADC), never as a launch variable.
    file_only: tuple[str, ...] = ()
    # Whether the host's login files may be copied in.
    host_login: bool = False
    # Whether the Gemini CLI's Vertex settings.json is written.
    vertex_settings: bool = False
    _model_credentials: frozenset[str] = field(
        default_factory=model_credential_env_names, repr=False, compare=False
    )

    def _own(self) -> set[str]:
        if self.route == PROXY:
            # The agent's own joint parts that are not model credentials:
            # a proxy route holds no model or login credential but its key.
            own = {*self.joint, *self.files}
            return {n for n in own if n not in self._model_credentials}
        if self.route == NONE:
            return set()
        return {*self.login, *self.joint, *self.files, *self.raw, *self.file_only}

    def _proxy_slots(self, env: dict[str, str]) -> set[str]:
        """Model credential slots that hold the proxy key on a proxy route."""
        master_key = env.get(_LITELLM_MASTER_KEY_ENV)
        if self.route != PROXY or not master_key:
            return set()
        return {n for n in self._model_credentials if env.get(n) == master_key}

    def launch_names(self, env: dict[str, str]) -> frozenset[str]:
        """The credential variables the agent's process may hold."""
        return frozenset((self._own() - set(self.file_only)) | self._proxy_slots(env))

    def admitted_value(self, source: str, env: dict[str, str]) -> str | None:
        """``source``'s value if the table admits it into a file, else None.

        The only value source for credential files: a template renders
        what this returns, so no transform of an unadmitted value (escaped,
        encoded or reformatted) can reach the agent's home.
        """
        value = env.get(source)
        if not value:
            return None
        if source not in host_credential_env_names():
            return value
        if source in self._own() or source in self._proxy_slots(env):
            return value
        return None

    def to_json(self) -> str:
        return json.dumps(
            {
                "route": self.route,
                "login": list(self.login),
                "joint": list(self.joint),
                "files": list(self.files),
                "raw": list(self.raw),
                "file_only": list(self.file_only),
                "host_login": self.host_login,
                "vertex_settings": self.vertex_settings,
            },
            sort_keys=True,
        )


def _route(agent: str, model: str | None, env: dict[str, str]) -> str:
    if env.get(_LITELLM_MASTER_KEY_ENV):
        return PROXY
    if agent in _SCRIPTED_AGENTS:
        return NONE
    if uses_direct_anthropic_endpoint(agent, model, env):
        return DIRECT
    if routed_model(agent, model) is None:
        return DIRECT
    if uses_native_subscription_auth(agent, model, env):
        return NATIVE
    return PROXY


def _login_names(selected: str | None, env: dict[str, str], route: str) -> list[str]:
    if selected is None:
        return []
    names = [selected]
    for alias, canonical in _LOGIN_ALIASES.items():
        if selected == alias and route in (NATIVE, DIRECT):
            # Resolution points the CLI's own variable at the selection.
            names.append(canonical)
        elif selected == canonical and env.get(alias) == env.get(selected):
            names.append(alias)
    return names


def compute_credential_policy(
    agent: str,
    model: str | None,
    env: dict[str, str],
    explicit_keys: set[str] | frozenset[str],
) -> CredentialPolicy:
    """The table for one agent, from its resolved env and the caller's keys."""
    # A label of an agent that routes its own model calls selects nothing.
    model = routed_model(agent, model)
    route = _route(agent, model, env)
    if route == NONE:
        return CredentialPolicy(route=route)
    cfg = AGENTS.get(agent)
    if agent == _MUSE_AGENT:
        login = [_MUSE_AUTH_JSON_ENV] if env.get(_MUSE_AUTH_JSON_ENV) else []
        return CredentialPolicy(route=route, login=tuple(login))
    if uses_native_vertex_login(agent, model):
        adc = (VERTEX_ADC_ENV,) if env.get(VERTEX_ADC_ENV) else ()
        return CredentialPolicy(
            route=route, file_only=adc, vertex_settings=route == NATIVE
        )
    if cfg is None:
        # A raw command: no registry entry declares its credentials, so the
        # slots its caller passed are its own. With a model it is proxied.
        raw: tuple[str, ...] = ()
        if route == DIRECT:
            credentials = host_credential_env_names()
            raw = tuple(
                sorted(k for k in explicit_keys if k in credentials and env.get(k))
            )
        return CredentialPolicy(route=route, raw=raw)
    # Alternative login slots compete and one is selected; parts the agent
    # declares as jointly required are one credential, selected together.
    alternatives = list(
        _LOGIN_KEYS_BY_FAMILY.get(_login_family(agent, model) or "", ())
    )
    if model is None:
        mapped = _mapped_provider_key(cfg)
        if mapped:
            alternatives.append(mapped)
    joint = tuple(
        key
        for key in getattr(cfg, "requires_env", None) or ()
        if key not in alternatives and env.get(key)
    )
    # A declared file source is the agent's own login on a direct or native
    # route, even in a model slot (C25); _own() drops model slots on a proxy
    # route. Competing login slots are left to the selection below.
    files = tuple(
        cf.env_source
        for cf in getattr(cfg, "credential_files", None) or ()
        if cf.env_source not in alternatives
        and cf.env_source not in joint
        and env.get(cf.env_source)
    )
    selected = _select_login(alternatives, env, set(explicit_keys))
    login = tuple(_login_names(selected, env, route))
    host_login = (
        env.get(_SUBSCRIPTION_AUTH_MARKER) == "1"
        and route in (NATIVE, DIRECT)
        and not login
    )
    return CredentialPolicy(
        route=route, login=login, joint=joint, files=files, host_login=host_login
    )


def credential_policy(
    agent: str, model: str | None, env: dict[str, str]
) -> CredentialPolicy:
    """The run's recorded table; an env resolution did not produce gets the
    same rules with nothing counted as explicitly passed."""
    raw = env.get(CREDENTIAL_POLICY_MARKER)
    if raw is not None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            route = data.get("route")
            if env.get(_LITELLM_MASTER_KEY_ENV):
                route = PROXY
            if route in (PROXY, NATIVE, DIRECT, NONE):
                return CredentialPolicy(
                    route=route,
                    login=_names(data.get("login")),
                    joint=_names(data.get("joint")),
                    files=_names(data.get("files")),
                    raw=_names(data.get("raw")),
                    file_only=_names(data.get("file_only")),
                    host_login=data.get("host_login") is True and route != PROXY,
                    vertex_settings=data.get("vertex_settings") is True
                    and route == NATIVE,
                )
        return CredentialPolicy(route=NONE)
    return compute_credential_policy(agent, model, env, frozenset())


def _names(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(name for name in value if isinstance(name, str))


def agent_launch_env(
    agent: str, model: str | None, env: dict[str, str]
) -> dict[str, str]:
    """``env`` with only the credentials the run's table admits."""
    credentials = host_credential_env_names()
    allowed = credential_policy(agent, model, env).launch_names(env)
    launch = {
        name: value
        for name, value in env.items()
        if name not in credentials or name in allowed
    }
    # An omitted value inherits the container's value at exec. Preserve
    # explicit empty writer guards even when credential filtering removed
    # the corresponding slot, or Codex's launch producer removed stale TOML.
    launch.update(launcher_guard_env(agent, launch))
    return launch
