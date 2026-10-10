//! Which way a run reaches its model, and what the agent is told.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Value, json};

use super::key::{ProviderKey, find_provider_key};
use super::routing::{self, ALWAYS_DENIED, Pin, PinError, Upstreams};
use super::{Endpoint, Protocol, Provider, ProviderRegistry};
use crate::credentials::{CredentialError, HostEnv, Login, Origin, SourceKind, choose_login};
use crate::manifest::{AgentManifest, AgentProviderSpec, CredentialFamily};

/// The key the agent is given in place of the provider's: the forwarder
/// replaces it with the real one on every request.
pub const KEY_PLACEHOLDER: &str = "benchflow-forwarder-holds-the-key";

/// How a run reaches its model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// On the agent's own logins, with the model as the agent spells it.
    Agent,
    /// On a provider's endpoint, through the forwarder.
    Provider(Box<ProviderRoute>),
}

/// A run on a provider: which endpoint, which model, and how the agent is
/// pointed at it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderRoute {
    pub provider: Provider,
    pub endpoint: Endpoint,
    /// The model id the provider serves (`deepseek/deepseek-v4-flash-0731`),
    /// or `None` when the run names none and the agent picks its own.
    pub model: Option<String>,
    /// The agent's `[providers.<protocol>]` table for this endpoint.
    pub spec: AgentProviderSpec,
    /// Set when the agent's own API key brought the run here (Claude Code on
    /// `ANTHROPIC_API_KEY`) rather than a `<provider>/<model>` id. The agent
    /// then picks and names its model as it does on its own login: `{model}`
    /// stands for nothing, the effort keeps the agent's default, and the
    /// forwarder lets through any model the key can pay for (on a router that
    /// pins per model, only the run's).
    pub own_key: bool,
    /// On a router, what every request names so it reaches no denied
    /// upstream ([`pin_route`]); `None` until pinned, and for providers that
    /// are not routers.
    pub upstreams: Option<Upstreams>,
}

/// Why a model cannot run on its provider with this agent. The fix is the
/// user's (R215): another agent, model or provider; for
/// [`RouteError::Unpinnable`], the agent's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// The agent cannot talk to any endpoint the provider serves.
    Unsupported {
        agent: String,
        provider: String,
        /// What the agent speaks, for people (empty: no `[providers]`).
        agent_speaks: Vec<String>,
        /// What the provider serves, for people.
        provider_serves: String,
        /// Providers the agent could use instead.
        usable: Vec<String>,
    },
    /// The model id cannot be passed on safely.
    BadModel { model: String, reason: String },
    /// The effort level cannot be passed on safely.
    BadEffort { effort: String },
    /// The agent would send requests to a router itself, where BenchFlow
    /// cannot keep them off denied upstreams.
    Unpinnable {
        agent: String,
        /// What takes it there: a router's key (`JUNIE_OPENROUTER_API_KEY`),
        /// or a `--env` variable pointing at the router.
        via: String,
        router: String,
    },
}

impl RouteError {
    /// What the user can do about it.
    pub fn next_step(&self) -> String {
        match self {
            Self::Unsupported {
                agent,
                usable,
                agent_speaks,
                ..
            } => {
                if agent_speaks.is_empty() {
                    format!(
                        "pick an agent that can take a base URL (`bench agents` shows which); `{agent}` runs only on its own logins"
                    )
                } else if usable.is_empty() {
                    "pick another agent; `bench agents <name>` shows which endpoints an agent can use"
                        .to_string()
                } else {
                    format!(
                        "pick another agent, or a provider `{agent}` can use: {}",
                        usable.join(", ")
                    )
                }
            }
            Self::BadModel { .. } => {
                "name the model as `<provider>/<model id>` with letters, digits and `._:/@+-`"
                    .to_string()
            }
            Self::BadEffort { .. } => {
                "use an effort level of lowercase letters and digits, such as `high`".to_string()
            }
            Self::Unpinnable { router, via, .. } => format!(
                "run the model as `{router}/<model>` on an agent that can take a base URL (`bench agents` shows which), or drop {via}"
            ),
        }
    }
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported {
                agent,
                provider,
                agent_speaks,
                provider_serves,
                ..
            } => {
                if agent_speaks.is_empty() {
                    write!(
                        f,
                        "agent `{agent}` cannot use provider `{provider}`: its manifest has no [providers] table, so it runs only on its own logins"
                    )
                } else {
                    write!(
                        f,
                        "agent `{agent}` cannot use provider `{provider}`: the agent speaks {}, and {provider} serves {provider_serves}",
                        agent_speaks.join(" or ")
                    )
                }
            }
            Self::BadModel { model, reason } => write!(f, "model `{model}`: {reason}"),
            Self::BadEffort { effort } => {
                write!(f, "effort `{effort}` cannot be passed to the agent")
            }
            Self::Unpinnable { agent, via, router } => write!(
                f,
                "agent `{agent}` would send its requests to {router} itself ({via}), where BenchFlow cannot keep them off the upstreams it never uses ({})",
                ALWAYS_DENIED.join(", ")
            ),
        }
    }
}

impl std::error::Error for RouteError {}

/// The way `manifest`'s agent reaches `model` (0.8's provider prefixes,
/// with the agent's own spelling first):
///
/// 1. no model, or a model the manifest's `strip_prefixes` claim
///    (`openai/gpt-5` for Codex): the agent's own logins;
/// 2. `<provider>/<model>` with a registered provider: that provider, through
///    the first `[providers]` table of the agent's that the provider serves.
///    When there is none, an agent that names models this way itself (one
///    whose login is the model's provider's key, one that speaks only a
///    gateway's API and takes every id as the gateway's) keeps its own
///    logins, unless the provider is a router, whose requests BenchFlow must
///    pin; any other agent is refused;
/// 3. anything else (a bare id, an unregistered prefix): the agent's own
///    logins, as before providers existed.
///
/// On its own logins, a plain API key still goes through the forwarder when
/// the agent can take a base URL for it ([`resolve`]).
pub fn route(
    manifest: &AgentManifest,
    model: Option<&str>,
    providers: &ProviderRegistry,
) -> Result<Route, RouteError> {
    let Some(model) = model else {
        return Ok(Route::Agent);
    };
    if manifest
        .model
        .strip_prefixes
        .iter()
        .any(|prefix| model.starts_with(prefix.as_str()))
    {
        return Ok(Route::Agent);
    }
    let Some((prefix, id)) = model.split_once('/') else {
        return Ok(Route::Agent);
    };
    let Some(provider) = providers.find(prefix) else {
        return Ok(Route::Agent);
    };
    if let Some(reason) = bad_model_id(id) {
        return Err(RouteError::BadModel {
            model: model.to_string(),
            reason,
        });
    }
    for spec in &manifest.providers {
        if let Some(endpoint) = provider.usable_by(spec) {
            return Ok(Route::Provider(Box::new(ProviderRoute {
                provider: provider.clone(),
                endpoint: endpoint.clone(),
                model: Some(id.to_string()),
                spec: spec.clone(),
                own_key: false,
                upstreams: None,
            })));
        }
    }
    if provider.upstream_routing().is_none() && takes_provider_ids(manifest) {
        return Ok(Route::Agent);
    }
    Err(RouteError::Unsupported {
        agent: manifest.name.clone(),
        provider: provider.name.clone(),
        agent_speaks: manifest.providers.iter().map(|s| s.describe()).collect(),
        provider_serves: provider.serves(),
        usable: providers
            .providers()
            .iter()
            .filter(|p| manifest.providers.iter().any(|s| p.usable_by(s).is_some()))
            .map(|p| p.name.clone())
            .collect(),
    })
}

/// Whether the agent's own logins take `<provider>/<model>` ids: its login
/// is the key of the model's provider (OpenCode, aider), or it speaks only a
/// gateway's own API, whose ids look the same (fx's `anthropic/claude-...`).
fn takes_provider_ids(manifest: &AgentManifest) -> bool {
    manifest.credentials == CredentialFamily::Model
        || (!manifest.providers.is_empty()
            && manifest
                .providers
                .iter()
                .all(|s| s.protocol == Protocol::Gateway))
}

/// Why a provider's model id cannot be passed on, if it cannot: it goes into
/// environment variables and config files, so it is kept to a safe set.
fn bad_model_id(id: &str) -> Option<String> {
    if id.is_empty() {
        return Some("names a provider but no model after it".to_string());
    }
    if id.len() > 200 {
        return Some("is longer than 200 characters".to_string());
    }
    let bad = id.chars().find(|c| {
        !(c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '/' | '@' | '+' | '-'))
    });
    bad.map(|c| format!("`{}` is not allowed in a model id", c.escape_default()))
}

/// An effort level that can go into variables and config files.
pub(crate) fn is_effort(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 32
        && s.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_'))
}

/// The placeholders a template uses, or why it is invalid. Placeholders are
/// `{base_url}`, `{key}`, `{model}` and `{effort}`; other braces (JSON) are
/// left alone.
pub(crate) fn template_placeholders(template: &str) -> Result<Vec<&'static str>, String> {
    let mut found = Vec::new();
    for name in placeholder_names(template) {
        let known = PLACEHOLDERS.iter().find(|p| **p == name).ok_or_else(|| {
            format!(
                "`{{{name}}}` is not a placeholder; use {}",
                PLACEHOLDERS
                    .iter()
                    .map(|p| format!("`{{{p}}}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        if !found.contains(known) {
            found.push(*known);
        }
    }
    Ok(found)
}

const PLACEHOLDERS: [&str; 4] = ["base_url", "key", "model", "effort"];
/// Placeholders that can stand for nothing: a line using one then is left
/// out, so a config file never names an empty model.
const OPTIONAL: [&str; 2] = ["model", "effort"];

/// Every `{name}` with a lowercase identifier inside.
fn placeholder_names(template: &str) -> impl Iterator<Item = &str> {
    template.match_indices('{').filter_map(move |(start, _)| {
        let rest = &template[start + 1..];
        let end = rest.find('}')?;
        let name = &rest[..end];
        (!name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c == '_'))
            .then_some(name)
    })
}

/// Fill a template's placeholders, leaving out every line that uses an
/// optional placeholder with no value. `None` when nothing is left of a
/// template that had something in it: the variable is then not set at all.
fn render(template: &str, values: &[(&str, &str)]) -> Option<String> {
    let missing: Vec<String> = values
        .iter()
        .filter(|(name, value)| value.is_empty() && OPTIONAL.contains(name))
        .map(|(name, _)| format!("{{{name}}}"))
        .collect();
    let mut out: String = template
        .split_inclusive('\n')
        .filter(|line| !missing.iter().any(|m| line.contains(m.as_str())))
        .collect();
    for (name, value) in values {
        out = out.replace(&format!("{{{name}}}"), value);
    }
    (out.trim().is_empty() == template.trim().is_empty()).then_some(out)
}

impl ProviderRoute {
    /// The path the forwarder serves this endpoint under (`/anthropic`); a
    /// gateway's own API is served from the root, since its clients add
    /// their own paths (`/v4/ai/language-model`).
    pub fn prefix(&self) -> String {
        match self.endpoint.protocol {
            Protocol::Gateway => String::new(),
            protocol => format!("/{}", protocol.as_str()),
        }
    }

    /// The effort `{effort}` stands for: the run's, else the spec's default
    /// (on the agent's own key, nothing: the agent keeps its own default).
    fn effort<'a>(&'a self, effort: Option<&'a str>) -> Result<&'a str, RouteError> {
        match effort {
            Some(e) if is_effort(e) => Ok(e),
            Some(e) => Err(RouteError::BadEffort {
                effort: e.to_string(),
            }),
            None if self.own_key => Ok(""),
            None => Ok(self.spec.effort.as_deref().unwrap_or("")),
        }
    }

    /// What `{model}` stands for: the provider's model id, or nothing on the
    /// agent's own key.
    fn template_model(&self) -> &str {
        if self.own_key {
            ""
        } else {
            self.model.as_deref().unwrap_or("")
        }
    }

    /// Variables for the agent process when the forwarder serves this
    /// endpoint at `base_url` (`http://127.0.0.1:<port>/anthropic`): the
    /// endpoint and placeholder-key variables, the cleared ones (set empty),
    /// and the manifest's templates (a template left with nothing sets no
    /// variable). The provider's key is never among them.
    pub fn agent_env(
        &self,
        base_url: &str,
        effort: Option<&str>,
    ) -> Result<BTreeMap<String, String>, RouteError> {
        let effort = self.effort(effort)?;
        let values = [
            ("base_url", base_url),
            ("key", KEY_PLACEHOLDER),
            ("model", self.template_model()),
            ("effort", effort),
        ];
        let mut env = BTreeMap::new();
        for var in &self.spec.clear {
            env.insert(var.clone(), String::new());
        }
        if let Some(var) = &self.spec.base_url {
            env.insert(var.clone(), base_url.to_string());
        }
        if let Some(var) = &self.spec.key {
            env.insert(var.clone(), KEY_PLACEHOLDER.to_string());
        }
        for (var, template) in &self.spec.env {
            if let Some(value) = render(template, &values) {
                env.insert(var.clone(), value);
            }
        }
        Ok(env)
    }

    /// The model id to select over ACP, or `None` when the agent's
    /// variables carry it. On the agent's own key, `None` too: the agent
    /// selects the run's model as it does on its own login. Checks the
    /// effort either way.
    pub fn acp_model(&self, effort: Option<&str>) -> Result<Option<String>, RouteError> {
        let effort = self.effort(effort)?;
        if self.own_key || self.spec.model.is_empty() {
            return Ok(None);
        }
        Ok(render(
            &self.spec.model,
            &[
                ("base_url", ""),
                ("key", KEY_PLACEHOLDER),
                ("model", self.template_model()),
                ("effort", effort),
            ],
        )
        .filter(|m| !m.is_empty()))
    }

    /// The models the forwarder lets through: the run's (an agent's helper
    /// asking for another is refused before the key can pay for it), or on
    /// the agent's own key any model, as on its own login. On a router that
    /// pins per model, always the run's: no other model's requests can be
    /// pinned.
    pub fn allowed_models(&self) -> Vec<String> {
        let pinned_per_model = matches!(self.upstreams, Some(Upstreams::Only(_)));
        match &self.model {
            Some(model) if !self.own_key || pinned_per_model => vec![model.clone()],
            _ => Vec::new(),
        }
    }

    /// The forwarder's configuration: where it listens, the key file it
    /// reads, and the one route it serves, for [`Self::allowed_models`] and,
    /// on a router, naming the allowed (`only`) or denied (`ignore`)
    /// upstreams in every request. It holds no secret.
    pub fn forwarder_config(&self, key_file: &str) -> Value {
        let mut route = json!({
            "prefix": self.prefix(),
            "upstream": self.endpoint.base_url,
            "auth": self.endpoint.auth.as_str(),
            "models": self.allowed_models(),
        });
        match &self.upstreams {
            Some(Upstreams::Only(only)) => route["only"] = json!(only),
            Some(Upstreams::Ignore(denied)) => route["ignore"] = json!(denied),
            None => {}
        }
        json!({
            "listen": "127.0.0.1",
            "port": 0,
            "key_file": key_file,
            "provider": self.provider.name,
            "routes": [route],
        })
    }

    /// For people: `vercel's Anthropic-compatible endpoint`, `vercel's own
    /// API`.
    pub fn describe(&self) -> String {
        match self.endpoint.protocol {
            Protocol::Gateway => format!("{}'s own API", self.provider.name),
            protocol => format!("{}'s {} endpoint", self.provider.name, protocol.describe()),
        }
    }
}

/// Keep `route`'s requests off denied upstreams when its provider is a
/// router ([`routing`]): on the Vercel AI Gateway, lists the model's
/// upstreams (once per process) and keeps the allowed ones; on OpenRouter,
/// names the denied ones. Sets [`ProviderRoute::upstreams`]. Call it before
/// any sandbox starts; it blocks on one HTTP request the first time a model
/// is seen.
pub fn pin_route(route: &mut ProviderRoute, host: &HostEnv) -> Result<Option<Pin>, PinError> {
    let Some(routing) = route.provider.upstream_routing() else {
        return Ok(None);
    };
    let pin = routing::pin_blocking(&route.provider.name, &routing, route.model.as_deref(), host)?;
    route.upstreams = Some(pin.upstreams.clone());
    Ok(Some(pin))
}

/// How a run reaches its model, and the login it runs on.
#[derive(Debug, Clone)]
pub struct Access {
    pub route: Route,
    /// The agent's own login, or for a provider route a login that only
    /// labels and redacts the provider key (no variables, no files).
    pub login: Login,
    /// The provider key, for the forwarder only.
    pub key: Option<ProviderKey>,
}

/// Why a run cannot reach its model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AccessError {
    /// No usable login or key (R211).
    Login(CredentialError),
    /// The agent cannot use the provider, or the model id cannot be passed
    /// on (R215).
    Route(RouteError),
}

impl fmt::Display for AccessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Login(e) => e.fmt(f),
            Self::Route(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for AccessError {}

/// The route and login for `manifest`'s agent on `model` (already defaulted)
/// on this machine: [`route`], then the provider's key, or the agent's own
/// login ([`crate::credentials::find_login`]). When that login is a plain
/// API key a registered provider reads and the agent can take a base URL
/// for it (one of its `[providers]` tables), the key goes to the forwarder
/// too and the agent gets a placeholder; subscription logins reach the agent
/// as they are. A run that would take a router's key, or a `--env` base
/// URL, to a router without the forwarder is refused.
pub fn resolve(
    manifest: &AgentManifest,
    model: Option<&str>,
    host: &HostEnv,
    providers: &ProviderRegistry,
) -> Result<Access, AccessError> {
    match route(manifest, model, providers).map_err(AccessError::Route)? {
        Route::Provider(route) => {
            let key = find_provider_key(&route.provider, host).map_err(AccessError::Login)?;
            Ok(Access {
                login: key.login(&route.describe()),
                route: Route::Provider(route),
                key: Some(key),
            })
        }
        Route::Agent => {
            let chosen = choose_login(manifest, model, host).map_err(AccessError::Login)?;
            if let Some(plain) = &chosen.api_key
                && let Some(route) = own_key_route(manifest, model, &plain.var, host, providers)
                    .map_err(AccessError::Route)?
            {
                let key = ProviderKey::from_login(&route.provider.name, plain);
                return Ok(Access {
                    login: key.login(&route.describe()),
                    route: Route::Provider(route),
                    key: Some(key),
                });
            }
            if let Some((var, router)) = env_router(host) {
                return Err(AccessError::Route(RouteError::Unpinnable {
                    agent: manifest.name.clone(),
                    via: format!("--env {var}"),
                    router,
                }));
            }
            Ok(Access {
                route: Route::Agent,
                login: chosen.login,
                key: None,
            })
        }
    }
}

/// The provider endpoint a plain API key read under `var` goes to through
/// the forwarder: the first registered provider that reads `var` and serves
/// one of the agent's `[providers]` tables. `None` when the key stays with
/// the agent: no such provider, or the run's `--env` points the agent at an
/// endpoint of its own (then the key is meant for that one). A router's key
/// the agent would take to the router itself is refused.
fn own_key_route(
    manifest: &AgentManifest,
    model: Option<&str>,
    var: &str,
    host: &HostEnv,
    providers: &ProviderRegistry,
) -> Result<Option<Box<ProviderRoute>>, RouteError> {
    let readers = providers
        .providers()
        .iter()
        .filter(|p| p.key.env.iter().any(|v| v == var));
    for provider in readers {
        for spec in &manifest.providers {
            let Some(endpoint) = provider.usable_by(spec) else {
                continue;
            };
            let router = provider.upstream_routing().is_some();
            if !router && points_elsewhere(spec, host) {
                return Ok(None);
            }
            let model = model.map(|m| manifest.cli_model(m));
            // On a router the id goes into the listing's URL.
            if router
                && let Some(m) = &model
                && let Some(reason) = bad_model_id(m)
            {
                return Err(RouteError::BadModel {
                    model: m.clone(),
                    reason,
                });
            }
            return Ok(Some(Box::new(ProviderRoute {
                provider: provider.clone(),
                endpoint: endpoint.clone(),
                model,
                spec: spec.clone(),
                own_key: true,
                upstreams: None,
            })));
        }
    }
    match routing::router_of_key(var, providers) {
        Some(router) => Err(RouteError::Unpinnable {
            agent: manifest.name.clone(),
            via: var.to_string(),
            router,
        }),
        None => Ok(None),
    }
}

/// Whether the run's `--env` sets the table's base-URL variable, or a
/// `*_BASE_URL` it clears: the agent is pointed at an endpoint of the
/// user's choosing.
fn points_elsewhere(spec: &AgentProviderSpec, host: &HostEnv) -> bool {
    spec.base_url
        .iter()
        .chain(spec.clear.iter().filter(|v| v.ends_with("_BASE_URL")))
        .any(|var| matches!(host.get(var), Some((_, Origin::AgentEnv))))
}

/// A `--env` variable pointing the agent at a router's host, and the router.
fn env_router(host: &HostEnv) -> Option<(String, String)> {
    host.overrides().find_map(|(var, value)| {
        let upper = var.to_ascii_uppercase();
        if !["URL", "BASE", "ENDPOINT", "HOST"]
            .iter()
            .any(|part| upper.contains(part))
        {
            return None;
        }
        let router = routing::router_at(value.trim())?;
        Some((var.to_string(), router.to_string()))
    })
}

/// What BenchFlow does with one of an agent's credentials.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyHandling {
    /// The forwarder holds it for this endpoint (`vercel's own API`); the
    /// agent gets a placeholder.
    Forwarded { to: String },
    /// The agent gets it, for this reason.
    ToAgent { why: String },
    /// A run on it is refused: the agent would take it to this router
    /// itself.
    Refused { router: String },
}

impl fmt::Display for KeyHandling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Forwarded { to } => write!(
                f,
                "held by the forwarder for {to}; the agent gets a placeholder"
            ),
            Self::ToAgent { why } => write!(f, "given to the agent: {why}"),
            Self::Refused { router } => write!(
                f,
                "refused: the agent would take it to {router} itself, unpinned"
            ),
        }
    }
}

/// What happens to the credential `name` of kind `kind` when `manifest`'s
/// agent runs on it (with no `--env` pointing it elsewhere), for `bench
/// agents <name>`.
pub fn key_handling(
    manifest: &AgentManifest,
    name: &str,
    kind: SourceKind,
    providers: &ProviderRegistry,
) -> KeyHandling {
    let as_api_key = |var: &str| {
        let host = HostEnv::new(BTreeMap::new(), None);
        match own_key_route(manifest, None, var, &host, providers) {
            Ok(Some(route)) => KeyHandling::Forwarded {
                to: route.describe(),
            },
            Ok(None) => KeyHandling::ToAgent {
                why: if providers
                    .providers()
                    .iter()
                    .any(|p| p.key.env.iter().any(|v| v == var))
                {
                    "the agent has no [providers] table for an endpoint that takes it".to_string()
                } else {
                    "no built-in provider reads it".to_string()
                },
            },
            Err(RouteError::Unpinnable { router, .. }) => KeyHandling::Refused { router },
            Err(e) => KeyHandling::ToAgent { why: e.to_string() },
        }
    };
    match kind {
        SourceKind::ApiKey => as_api_key(name),
        SourceKind::AuthJson => KeyHandling::ToAgent {
            why: match as_api_key("OPENAI_API_KEY") {
                KeyHandling::Forwarded { .. } => {
                    "a ChatGPT login (an API key stored in it goes to the forwarder, as OPENAI_API_KEY does)"
                        .to_string()
                }
                _ => "a ChatGPT login".to_string(),
            },
        },
        SourceKind::AuthToken => KeyHandling::ToAgent {
            why: "a bearer token for an endpoint of the user's choosing".to_string(),
        },
        SourceKind::OauthToken | SourceKind::AccessToken | SourceKind::LoginFile => {
            KeyHandling::ToAgent {
                why: "a subscription login".to_string(),
            }
        }
    }
}
