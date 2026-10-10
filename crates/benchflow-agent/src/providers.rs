//! Model providers as data, and the forwarder that holds their keys.
//!
//! A provider is a TOML file that says where its endpoints are, where its key
//! comes from, and how to check the key. The built-in providers (`vercel`,
//! `openai`, `anthropic`, `openrouter`, `zai`, `deepseek`) are embedded in
//! the binary from `crates/benchflow-agent/providers/`; a project adds or
//! replaces providers with its own files ([`ProviderRegistry::with_files`]).
//!
//! A model named `<provider>/<model>` runs on that provider when `<provider>`
//! is registered: `vercel/deepseek/deepseek-v4-flash-0731` is the model
//! `deepseek/deepseek-v4-flash-0731` on the Vercel AI Gateway. The
//! provider's key is then the login, whatever the agent's own login family
//! is, and the agent must be able to talk to one of the provider's endpoints
//! (its manifest's [`[providers]`](crate::manifest) table). A prefix the
//! agent strips itself (`openai/` for Codex, `anthropic/` for Claude Code,
//! `google/` for Gemini CLI) keeps the agent on its own logins, as before.
//! See [`route`].
//!
//! The key never reaches the agent. Before the agent starts, BenchFlow starts
//! a small forwarder as root inside the sandbox, on loopback
//! ([`Forwarder`]): it reads the key from a root-only file, adds it to each
//! request in the header the provider wants, streams the answers back, and
//! serves only the provider's endpoints. The agent gets the forwarder's URL
//! and a placeholder key ([`KEY_PLACEHOLDER`]).
//!
//! ```toml
//! name = "vercel"
//! description = "Vercel AI Gateway"
//!
//! [openai]                                   # an OpenAI-compatible API
//! base_url = "https://ai-gateway.vercel.sh/v1"
//! apis = ["chat", "responses"]
//!
//! [anthropic]                                # an Anthropic-compatible API
//! base_url = "https://ai-gateway.vercel.sh/claude-code"
//! auth = "bearer"
//!
//! [key]
//! env = ["AI_GATEWAY_API_KEY"]
//! files = ["~/.config/benchflow/vercel-ai-gateway.env"]
//!
//! [check]
//! endpoint = "openai"
//! path = "/credits"
//! ```
//!
//! # Keys
//!
//! | Key | Type | Meaning |
//! |---|---|---|
//! | `name` | string, required | What goes before the first `/` of a model name: lowercase letters, digits and `-`. |
//! | `description` | string | One line for `bench agents` and `bench doctor`. |
//! | `[openai] base_url` | string | An OpenAI-compatible API, up to the version (`https://api.openai.com/v1`): requests go to `<base_url>/chat/completions` or `<base_url>/responses`. Must be `https`. |
//! | `[openai] apis` | strings | Which OpenAI APIs it serves: `"chat"` (Chat Completions) and `"responses"` (the Responses API, which Codex needs). Default `["chat"]`. |
//! | `[openai] auth` | string | How the key is sent: `"bearer"` (`Authorization: Bearer <key>`, the default) or `"x-api-key"`. |
//! | `[anthropic] base_url` | string | An Anthropic-compatible API, without the version (`https://api.anthropic.com`): requests go to `<base_url>/v1/messages`. Must be `https`. |
//! | `[anthropic] auth` | string | `"x-api-key"` (the default) or `"bearer"`. |
//! | `[key] env` | strings | The variables the key is read from, in order. |
//! | `[key] files` | strings | Files of `NAME=value` lines (`~/` is the home folder) read for the same variables after the environment. |
//! | `[check] endpoint` | string | `"openai"` or `"anthropic"`: which endpoint the check request goes to. |
//! | `[check] path` | string | A `GET` path under that endpoint that needs a valid key and spends nothing (`/credits`, `/models`). |
//! | `[check] headers` | table of strings | Extra headers for the check (`anthropic-version`). |
//! | `[gateway] base_url` | string | A gateway's own API at its root (the Vercel AI Gateway's AI SDK protocol, which fx speaks). `[gateway] auth` as for `[openai]`. |
//! | `[routing] endpoints` | string | For a router that picks among upstream providers and takes an allowlist (the Vercel AI Gateway): the `https` URL listing the upstreams serving a model, with `{model}` for the model id. Every request names the allowed ones ([`routing`]). |
//! | `[routing] ignore` | bool | For a router that takes a denylist (OpenRouter's `provider.ignore`): every request names the denied upstreams. Use either `endpoints` or this. |
//! | `[routing] deny` | strings | Upstreams never to use, on top of `baseten`, which BenchFlow always denies. |
//!
//! At least one of `[openai]`, `[anthropic]` and `[gateway]` is required,
//! and so is a key source. Unknown keys are errors, with a suggestion when one is close.

use std::fmt;
use std::path::{Path, PathBuf};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::manifest::keys::{self, Map, Schema, Table, Value};
use crate::manifest::{Checker, ManifestError, ManifestSource};

mod check;
mod forwarder;
mod key;
mod route;
pub mod routing;

pub use check::{CheckOutcome, check_key};
pub use forwarder::{
    FORWARDER_DIR, FORWARDER_JS, Forwarder, ForwarderError, ForwarderReport, NODE, ProviderSpend,
    node_layer,
};
pub use key::{KeyStatus, ProviderKey, find_provider_key, key_status};
pub use route::{
    Access, AccessError, KEY_PLACEHOLDER, KeyHandling, ProviderRoute, Route, RouteError,
    key_handling, pin_route, resolve, route,
};
pub(crate) use route::{is_effort, template_placeholders};
pub use routing::{ALWAYS_DENIED, DENY_ENV, Pin, PinError, Routing, Style, Upstreams};

/// A model API protocol: what an agent speaks and an endpoint serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// The Anthropic Messages API (`/v1/messages`).
    Anthropic,
    /// An OpenAI-compatible API (Chat Completions, the Responses API).
    #[serde(rename = "openai")]
    OpenAi,
    /// A gateway's own API (the Vercel AI Gateway's AI SDK protocol, which fx
    /// speaks), served from the gateway's root.
    Gateway,
}

impl Protocol {
    /// The name used in manifests (`anthropic`, `openai`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Gateway => "gateway",
        }
    }

    /// For people: `Anthropic-compatible`, `OpenAI-compatible`, `the
    /// gateway's own API`.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Anthropic => "Anthropic-compatible",
            Self::OpenAi => "OpenAI-compatible",
            Self::Gateway => "the gateway's own API",
        }
    }

    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s {
            "anthropic" => Some(Self::Anthropic),
            "openai" => Some(Self::OpenAi),
            "gateway" => Some(Self::Gateway),
            _ => None,
        }
    }
}

impl fmt::Display for Protocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An OpenAI API an endpoint serves or an agent calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiApi {
    /// Chat Completions (`/chat/completions`).
    Chat,
    /// The Responses API (`/responses`).
    Responses,
}

impl OpenAiApi {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Responses => "responses",
        }
    }

    /// For people: `Chat Completions`, `Responses API`.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Chat => "Chat Completions",
            Self::Responses => "Responses API",
        }
    }

    pub(crate) fn parse(s: &str) -> Option<Self> {
        match s {
            "chat" => Some(Self::Chat),
            "responses" => Some(Self::Responses),
            _ => None,
        }
    }
}

/// How the key is sent upstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Auth {
    /// `Authorization: Bearer <key>`.
    Bearer,
    /// `x-api-key: <key>`.
    XApiKey,
}

impl Auth {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bearer => "bearer",
            Self::XApiKey => "x-api-key",
        }
    }
}

/// One of a provider's endpoints.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Endpoint {
    pub protocol: Protocol,
    /// `https://...`, without a trailing slash.
    pub base_url: String,
    pub auth: Auth,
    /// With `openai`: the APIs it serves. Empty for `anthropic`.
    pub apis: Vec<OpenAiApi>,
}

/// Where a provider's key comes from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KeySource {
    /// Variables, in order.
    pub env: Vec<String>,
    /// `NAME=value` files read for the same variables, after the environment.
    pub files: Vec<String>,
}

/// A request that proves a key works without spending anything.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Check {
    pub endpoint: Protocol,
    /// Under the endpoint's base URL, starting with `/`.
    pub path: String,
    pub headers: IndexMap<String, String>,
}

/// A model provider, validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Provider {
    pub name: String,
    pub description: String,
    pub openai: Option<Endpoint>,
    pub anthropic: Option<Endpoint>,
    /// The gateway's own API, at its root (`[gateway]`).
    pub gateway: Option<Endpoint>,
    pub key: KeySource,
    pub check: Option<Check>,
    /// How its upstreams are listed and limited, as its file says.
    pub routing: Option<Routing>,
    pub source: ManifestSource,
}

impl Provider {
    /// How its requests are kept off denied upstreams: its `[routing]`, or
    /// the router's own for any provider on a known router's host (the
    /// Vercel AI Gateway, OpenRouter), so no file can turn it off. `None` for
    /// providers that are not routers.
    pub fn upstream_routing(&self) -> Option<Routing> {
        if let Some(routing) = &self.routing {
            return Some(routing.clone());
        }
        self.endpoints()
            .find_map(|e| Routing::for_host(&e.base_url))
    }

    /// Parse and validate a provider file's text.
    pub fn parse(text: &str, source: ManifestSource) -> Result<Self, ManifestError> {
        let doc = keys::Document::parse(text, &source)?;
        doc.check_against(SCHEMA)?;
        let raw: RawProvider = toml::from_str(text).map_err(|e| {
            let (line, column) = e
                .span()
                .map(|span| doc.position(span.start))
                .unwrap_or((0, 0));
            ManifestError::Syntax {
                origin: source.clone(),
                line,
                column,
                message: e.message().trim().to_string(),
            }
        })?;
        raw.validate(&doc, source)
    }

    /// Read and validate a provider file.
    pub fn load(path: &Path) -> Result<Self, ManifestError> {
        let text = std::fs::read_to_string(path).map_err(|error| ManifestError::Read {
            path: path.to_path_buf(),
            error,
        })?;
        Self::parse(&text, ManifestSource::File(path.to_path_buf()))
    }

    /// The built-in providers. They are checked by this crate's tests, so a
    /// failure here is a bug in BenchFlow.
    pub fn builtins() -> Vec<Self> {
        BUILTINS
            .iter()
            .map(|(file, text)| {
                Self::parse(text, ManifestSource::BuiltinProvider(file))
                    .unwrap_or_else(|e| panic!("BenchFlow bug: invalid built-in provider: {e}"))
            })
            .collect()
    }

    /// The endpoint for `protocol`, if the provider serves it.
    pub fn endpoint(&self, protocol: Protocol) -> Option<&Endpoint> {
        match protocol {
            Protocol::Anthropic => self.anthropic.as_ref(),
            Protocol::OpenAi => self.openai.as_ref(),
            Protocol::Gateway => self.gateway.as_ref(),
        }
    }

    /// The endpoint an agent's `[providers.<protocol>]` table can use, if
    /// the provider serves its protocol (and, for OpenAI, its API).
    pub fn usable_by(&self, spec: &crate::manifest::AgentProviderSpec) -> Option<&Endpoint> {
        let endpoint = self.endpoint(spec.protocol)?;
        match spec.api {
            Some(api) if !endpoint.apis.contains(&api) => None,
            _ => Some(endpoint),
        }
    }

    /// Its endpoints, OpenAI-compatible first.
    pub fn endpoints(&self) -> impl Iterator<Item = &Endpoint> {
        self.openai
            .iter()
            .chain(self.anthropic.iter())
            .chain(self.gateway.iter())
    }

    /// What it serves, for people: `OpenAI-compatible (Chat Completions,
    /// Responses API), Anthropic-compatible and the gateway's own API`.
    pub fn serves(&self) -> String {
        let parts: Vec<String> = self
            .endpoints()
            .map(|e| match e.protocol {
                Protocol::OpenAi => format!(
                    "{} ({})",
                    e.protocol.describe(),
                    e.apis
                        .iter()
                        .map(|a| a.describe())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                Protocol::Anthropic | Protocol::Gateway => e.protocol.describe().to_string(),
            })
            .collect();
        match parts.split_last() {
            Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
            _ => parts.join(""),
        }
    }
}

/// The built-in providers, in the order `bench doctor` lists them.
const BUILTINS: &[(&str, &str)] = &[
    ("vercel.toml", include_str!("../providers/vercel.toml")),
    ("openai.toml", include_str!("../providers/openai.toml")),
    (
        "anthropic.toml",
        include_str!("../providers/anthropic.toml"),
    ),
    (
        "openrouter.toml",
        include_str!("../providers/openrouter.toml"),
    ),
    ("zai.toml", include_str!("../providers/zai.toml")),
    ("deepseek.toml", include_str!("../providers/deepseek.toml")),
];

/// Every provider a run knows: the built-ins, then the project's.
#[derive(Debug, Clone)]
pub struct ProviderRegistry {
    providers: Vec<Provider>,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::builtin()
    }
}

impl ProviderRegistry {
    /// The built-in providers.
    pub fn builtin() -> Self {
        Self {
            providers: Provider::builtins(),
        }
    }

    /// The built-ins plus provider files. Each path is a file, or a folder
    /// whose `*.toml` files are all providers (read in name order). A
    /// provider named like one already registered replaces it.
    pub fn with_files<I, P>(paths: I) -> Result<Self, ManifestError>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut registry = Self::builtin();
        for path in paths {
            for file in toml_files(path.as_ref())? {
                registry.add(Provider::load(&file)?);
            }
        }
        Ok(registry)
    }

    /// Add a provider, replacing any of the same name.
    pub fn add(&mut self, provider: Provider) {
        match self.providers.iter().position(|p| p.name == provider.name) {
            Some(i) => self.providers[i] = provider,
            None => self.providers.push(provider),
        }
    }

    /// The provider named `name`.
    pub fn find(&self, name: &str) -> Option<&Provider> {
        self.providers.iter().find(|p| p.name == name)
    }

    /// Every provider, built-ins first.
    pub fn providers(&self) -> &[Provider] {
        &self.providers
    }
}

/// The `*.toml` files a path names: itself, or a folder's.
fn toml_files(path: &Path) -> Result<Vec<PathBuf>, ManifestError> {
    let read_error = |error| ManifestError::Read {
        path: path.to_path_buf(),
        error,
    };
    if !path.is_dir() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files = Vec::new();
    for entry in std::fs::read_dir(path).map_err(read_error)? {
        let file = entry.map_err(read_error)?.path();
        if file.extension().is_some_and(|ext| ext == "toml") && file.is_file() {
            files.push(file);
        }
    }
    files.sort();
    Ok(files)
}

// ---------------------------------------------------------------------------
// Raw TOML shape and validation
// ---------------------------------------------------------------------------

const SCHEMA: Schema = &[
    ("name", Value),
    ("description", Value),
    ("openai", Table(OPENAI)),
    ("anthropic", Table(ANTHROPIC)),
    ("gateway", Table(ANTHROPIC)),
    ("key", Table(KEY)),
    ("check", Table(CHECK)),
    ("routing", Table(ROUTING)),
];
const OPENAI: Schema = &[("base_url", Value), ("apis", Value), ("auth", Value)];
const ANTHROPIC: Schema = &[("base_url", Value), ("auth", Value)];
const KEY: Schema = &[("env", Value), ("files", Value)];
const CHECK: Schema = &[("endpoint", Value), ("path", Value), ("headers", Map)];
const ROUTING: Schema = &[("endpoints", Value), ("ignore", Value), ("deny", Value)];

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawProvider {
    name: String,
    #[serde(default)]
    description: String,
    openai: Option<RawEndpoint>,
    anthropic: Option<RawEndpoint>,
    gateway: Option<RawEndpoint>,
    #[serde(default)]
    key: RawKey,
    check: Option<RawCheck>,
    routing: Option<RawRouting>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRouting {
    endpoints: Option<String>,
    #[serde(default)]
    ignore: bool,
    #[serde(default)]
    deny: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawEndpoint {
    base_url: String,
    apis: Option<Vec<String>>,
    auth: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawKey {
    #[serde(default)]
    env: Vec<String>,
    #[serde(default)]
    files: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawCheck {
    endpoint: String,
    path: String,
    #[serde(default)]
    headers: IndexMap<String, String>,
}

/// A provider name: lowercase letters, digits and `-`, starting with a letter
/// or digit.
fn is_provider_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase() || c.is_ascii_digit())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// An HTTP header name.
fn is_header_name(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_'))
}

impl RawProvider {
    fn validate(
        self,
        doc: &keys::Document<'_>,
        source: ManifestSource,
    ) -> Result<Provider, ManifestError> {
        let check = Checker {
            doc,
            source: &source,
        };
        check.require(is_provider_name(&self.name), "name", || {
            format!(
                "`{}` is not a valid provider name; use lowercase letters, digits and `-`",
                self.name
            )
        })?;
        let openai = self
            .openai
            .map(|raw| raw.validate(&check, Protocol::OpenAi))
            .transpose()?;
        let anthropic = self
            .anthropic
            .map(|raw| raw.validate(&check, Protocol::Anthropic))
            .transpose()?;
        let gateway = self
            .gateway
            .map(|raw| raw.validate(&check, Protocol::Gateway))
            .transpose()?;
        if openai.is_none() && anthropic.is_none() && gateway.is_none() {
            return Err(check.fail(
                "name",
                "a provider needs an `[openai]`, `[anthropic]` or `[gateway]` endpoint",
            ));
        }
        for (i, name) in self.key.env.iter().enumerate() {
            check.require(
                crate::manifest::is_env_name(name),
                &format!("key.env[{i}]"),
                || format!("`{name}` is not an environment variable name"),
            )?;
        }
        for (i, file) in self.key.files.iter().enumerate() {
            check.require(
                file.starts_with("~/") || file.starts_with('/'),
                &format!("key.files[{i}]"),
                || format!("`{file}` must start with `~/` or `/`"),
            )?;
        }
        if self.key.env.is_empty() {
            return Err(check.fail(
                "key",
                "say which variables hold the key: `[key] env = [\"<NAME>\"]` (files are read for the same names)",
            ));
        }
        let check_request = match self.check {
            None => None,
            Some(raw) => {
                let Some(endpoint) = Protocol::parse(&raw.endpoint) else {
                    return Err(check.fail(
                        "check.endpoint",
                        format!("`{}` is not `openai` or `anthropic`", raw.endpoint),
                    ));
                };
                let served = match endpoint {
                    Protocol::OpenAi => openai.is_some(),
                    Protocol::Anthropic => anthropic.is_some(),
                    Protocol::Gateway => gateway.is_some(),
                };
                check.require(served, "check.endpoint", || {
                    format!("the provider has no `[{endpoint}]` endpoint to check")
                })?;
                check.require(
                    raw.path.starts_with('/') && !raw.path.contains("://"),
                    "check.path",
                    || format!("`{}` must be a path starting with `/`", raw.path),
                )?;
                for name in raw.headers.keys() {
                    check.require(
                        is_header_name(name),
                        &format!("check.headers.{name}"),
                        || format!("`{name}` is not a header name"),
                    )?;
                    check.require(
                        !matches!(
                            name.to_ascii_lowercase().as_str(),
                            "authorization" | "x-api-key"
                        ),
                        &format!("check.headers.{name}"),
                        || "the key's header is added by BenchFlow; leave it out".into(),
                    )?;
                }
                Some(Check {
                    endpoint,
                    path: raw.path,
                    headers: raw.headers,
                })
            }
        };
        let routing = match self.routing {
            None => None,
            Some(raw) => {
                let style = match (raw.endpoints, raw.ignore) {
                    (Some(_), true) => {
                        return Err(check.fail(
                            "routing.ignore",
                            "use either `endpoints` (list and allow) or `ignore = true` (name the denied), not both",
                        ));
                    }
                    (None, false) => {
                        return Err(check.fail(
                            "routing",
                            "say how the router is pinned: `endpoints = \"https://...{model}...\"` or `ignore = true`",
                        ));
                    }
                    (None, true) => routing::Style::Ignore,
                    (Some(endpoints), false) => {
                        check.require(
                            endpoints.starts_with("https://") && endpoints.contains("{model}"),
                            "routing.endpoints",
                            || {
                                format!(
                                    "`{endpoints}` must be an `https://` URL with `{{model}}` in it"
                                )
                            },
                        )?;
                        routing::Style::Only { endpoints }
                    }
                };
                let mut deny = Vec::new();
                for (i, name) in raw.deny.iter().enumerate() {
                    check.require(
                        !name.is_empty()
                            && name.chars().all(|c| {
                                c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_' | '.')
                            }),
                        &format!("routing.deny[{i}]"),
                        || format!("`{name}` is not an upstream name (lowercase letters, digits, `-`, `_`, `.`)"),
                    )?;
                    deny.push(name.clone());
                }
                Some(Routing { style, deny })
            }
        };
        Ok(Provider {
            name: self.name,
            description: self.description,
            openai,
            anthropic,
            gateway,
            key: KeySource {
                env: self.key.env,
                files: self.key.files,
            },
            check: check_request,
            routing,
            source,
        })
    }
}

impl RawEndpoint {
    fn validate(self, check: &Checker<'_>, protocol: Protocol) -> Result<Endpoint, ManifestError> {
        let table = protocol.as_str();
        let base_url = self.base_url.trim_end_matches('/').to_string();
        check.require(
            is_https_base(&base_url),
            &format!("{table}.base_url"),
            || {
                format!(
                    "`{}` must be an `https://` URL without a query or fragment",
                    self.base_url
                )
            },
        )?;
        let auth = match self.auth.as_deref() {
            None => match protocol {
                Protocol::OpenAi | Protocol::Gateway => Auth::Bearer,
                Protocol::Anthropic => Auth::XApiKey,
            },
            Some("bearer") => Auth::Bearer,
            Some("x-api-key") => Auth::XApiKey,
            Some(other) => {
                return Err(check.fail(
                    &format!("{table}.auth"),
                    format!("`{other}` is not `bearer` or `x-api-key`"),
                ));
            }
        };
        let apis = match (protocol, self.apis) {
            (Protocol::Anthropic | Protocol::Gateway, Some(_)) => {
                return Err(check.fail(
                    &format!("{table}.apis"),
                    "only an `[openai]` endpoint lists its APIs",
                ));
            }
            (Protocol::Anthropic | Protocol::Gateway, None) => Vec::new(),
            (Protocol::OpenAi, None) => vec![OpenAiApi::Chat],
            (Protocol::OpenAi, Some(names)) => {
                let mut apis = Vec::new();
                for (i, name) in names.iter().enumerate() {
                    let Some(api) = OpenAiApi::parse(name) else {
                        return Err(check.fail(
                            &format!("openai.apis[{i}]"),
                            format!("`{name}` is not `chat` or `responses`"),
                        ));
                    };
                    if !apis.contains(&api) {
                        apis.push(api);
                    }
                }
                check.require(!apis.is_empty(), "openai.apis", || {
                    "list at least one of `chat` and `responses`".into()
                })?;
                apis
            }
        };
        Ok(Endpoint {
            protocol,
            base_url,
            auth,
            apis,
        })
    }
}

/// An `https://host[/path]` URL without query, fragment, user info or
/// whitespace.
fn is_https_base(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else {
        return false;
    };
    let host = rest.split('/').next().unwrap_or("");
    !host.is_empty()
        && !host.contains('@')
        && !rest.contains(['?', '#', '\\'])
        && !rest.chars().any(|c| c.is_whitespace() || c.is_control())
        && !rest.split('/').any(|part| part == "..")
}

#[cfg(test)]
mod tests;
