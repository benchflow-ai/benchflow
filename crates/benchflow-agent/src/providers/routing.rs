//! Keeping requests off denied upstream providers.
//!
//! Some providers are routers: the Vercel AI Gateway and OpenRouter send each
//! request to one of the providers serving its model (sixteen serve
//! `deepseek/deepseek-v4-flash-0731` on Vercel), choosing among them
//! themselves. BenchFlow never lets a request reach a denied upstream:
//!
//! - [`ALWAYS_DENIED`] (`baseten`, BenchFlow's standing rule) cannot be
//!   removed by any file or setting;
//! - a provider file adds its own (`[routing] deny`);
//! - the user adds more with [`DENY_ENV`] (`BENCHFLOW_DENY_UPSTREAMS="a,b"`).
//!
//! Each router is pinned the way it understands ([`Style`]):
//!
//! - **the Vercel AI Gateway** has no "exclude" option, only an allowlist.
//!   For each model its list of upstreams is read once per process
//!   (`[routing] endpoints`, which needs no key), the denied ones are
//!   dropped, and the rest are named in every request
//!   (`providerOptions.gateway.only`, which it honors on chat completions,
//!   the Responses API, Anthropic messages and its own AI SDK protocol). When
//!   nothing is left, or the list cannot be read, the run stops before any
//!   sandbox starts ([`PinError`]);
//! - **OpenRouter** takes the denied upstreams in each request
//!   (`provider.ignore`, its documented provider-routing field).
//!
//! A provider whose endpoints are on one of these routers' hosts is pinned
//! even when its file has no `[routing]`, so replacing a built-in provider
//! cannot turn this off.

use std::collections::HashMap;
use std::fmt;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use serde::Serialize;
use serde_json::{Map, Value};

use super::ProviderRegistry;
use crate::credentials::HostEnv;

/// Upstreams BenchFlow never uses, whatever a provider file or the user says.
pub const ALWAYS_DENIED: &[&str] = &["baseten"];
/// The variable that denies more upstreams: names separated by commas or
/// spaces (`BENCHFLOW_DENY_UPSTREAMS="together,fireworks"`).
pub const DENY_ENV: &str = "BENCHFLOW_DENY_UPSTREAMS";
/// The Vercel AI Gateway's host.
pub const VERCEL_GATEWAY_HOST: &str = "ai-gateway.vercel.sh";
/// Where the Vercel AI Gateway lists a model's upstreams.
pub const VERCEL_ENDPOINTS: &str = "https://ai-gateway.vercel.sh/v1/models/{model}/endpoints";
/// OpenRouter's host.
pub const OPENROUTER_HOST: &str = "openrouter.ai";

/// How a router is kept off denied upstreams.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Style {
    /// List the model's upstreams at `endpoints` (`{model}` is the model id)
    /// and name the allowed ones in `providerOptions.gateway.only`.
    Only { endpoints: String },
    /// Name the denied upstreams in `provider.ignore`.
    Ignore,
}

/// How a provider's upstreams are limited (`[routing]`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Routing {
    pub style: Style,
    /// Upstreams the provider file denies, on top of [`ALWAYS_DENIED`].
    pub deny: Vec<String>,
}

impl Routing {
    /// The Vercel AI Gateway's, for a provider on its host without a
    /// `[routing]` of its own.
    pub fn vercel() -> Self {
        Self {
            style: Style::Only {
                endpoints: VERCEL_ENDPOINTS.to_string(),
            },
            deny: Vec::new(),
        }
    }

    /// OpenRouter's, for a provider on its host without a `[routing]` of its
    /// own.
    pub fn openrouter() -> Self {
        Self {
            style: Style::Ignore,
            deny: Vec::new(),
        }
    }

    /// The router's own, for a base URL on a known router's host.
    pub fn for_host(base_url: &str) -> Option<Self> {
        match router_at(base_url)? {
            "vercel" => Some(Self::vercel()),
            _ => Some(Self::openrouter()),
        }
    }
}

/// The router a URL is on (`vercel`, `openrouter`), if any.
pub(crate) fn router_at(url: &str) -> Option<&'static str> {
    match host_of(url)? {
        VERCEL_GATEWAY_HOST => Some("vercel"),
        OPENROUTER_HOST => Some("openrouter"),
        _ => None,
    }
}

/// Parts of a variable's name that mark a router's key, for agents that read
/// one under a name of their own (Junie's `JUNIE_OPENROUTER_API_KEY`).
const ROUTER_KEY_NAMES: [(&str, &str); 2] =
    [("OPENROUTER", "openrouter"), ("AI_GATEWAY", "vercel")];

/// The router a variable holds a key for: one a router provider reads, or
/// one named for a router.
pub(crate) fn router_of_key(var: &str, providers: &ProviderRegistry) -> Option<String> {
    if let Some(p) = providers
        .providers()
        .iter()
        .find(|p| p.upstream_routing().is_some() && p.key.env.iter().any(|v| v == var))
    {
        return Some(p.name.clone());
    }
    let upper = var.to_ascii_uppercase();
    ROUTER_KEY_NAMES
        .iter()
        .find(|(part, _)| upper.contains(part))
        .map(|(_, router)| router.to_string())
}

/// What every request names, so it reaches no denied upstream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Upstreams {
    /// `providerOptions.gateway.only`: the allowed upstreams. Replaces any
    /// list the agent sent.
    Only(Vec<String>),
    /// `provider.ignore`: the denied upstreams. Added to any list the agent
    /// sent.
    Ignore(Vec<String>),
}

impl Upstreams {
    /// Write the list into a request body. A body that is not a JSON object is
    /// left alone. Returns whether it changed.
    pub fn apply(&self, body: &mut Value) -> bool {
        match self {
            Self::Only(only) => {
                let Some(gateway) = object_at(body, &["providerOptions", "gateway"]) else {
                    return false;
                };
                gateway.insert(
                    "only".into(),
                    Value::Array(only.iter().cloned().map(Value::String).collect()),
                );
                true
            }
            Self::Ignore(denied) => {
                let Some(provider) = object_at(body, &["provider"]) else {
                    return false;
                };
                let mut ignore: Vec<Value> = match provider.get("ignore") {
                    Some(Value::Array(items)) => items.clone(),
                    _ => Vec::new(),
                };
                for name in denied {
                    let value = Value::String(name.clone());
                    if !ignore.contains(&value) {
                        ignore.push(value);
                    }
                }
                provider.insert("ignore".into(), Value::Array(ignore));
                true
            }
        }
    }

    /// For people: `only alibaba, novita`, `never baseten`.
    pub fn describe(&self) -> String {
        match self {
            Self::Only(only) => format!("only {}", only.join(", ")),
            Self::Ignore(denied) => format!("never {}", denied.join(", ")),
        }
    }
}

/// The object at `path` in `body`, made (replacing anything else in the
/// way) when missing.
fn object_at<'a>(body: &'a mut Value, path: &[&str]) -> Option<&'a mut Map<String, Value>> {
    let mut here = body.as_object_mut()?;
    for key in path {
        let next = here
            .entry((*key).to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if !next.is_object() {
            *next = Value::Object(Map::new());
        }
        here = next.as_object_mut()?;
    }
    Some(here)
}

/// How a model's requests are pinned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Pin {
    /// The model, when the router needed one to list its upstreams.
    pub model: Option<String>,
    pub upstreams: Upstreams,
    /// Upstreams that serve the model and are denied (listing routers only).
    pub excluded: Vec<String>,
}

/// Why a model's requests cannot be pinned. The run stops before any
/// sandbox starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinError {
    /// Every upstream serving the model is denied.
    AllDenied {
        provider: String,
        model: String,
        excluded: Vec<String>,
    },
    /// The router lists no upstream for the model (an unknown model id).
    NotServed {
        provider: String,
        model: String,
        /// The router's closest model id, when its model list could be read
        /// and one is close enough to be a likely typo.
        close: Option<String>,
        /// Where the router lists its models.
        list: Option<String>,
    },
    /// The list could not be read; BenchFlow sends nothing it cannot pin.
    Unlisted {
        provider: String,
        model: String,
        detail: String,
    },
    /// The router pins per model, and the run names none.
    NeedsModel { provider: String },
}

impl PinError {
    /// What to do about it.
    pub fn next_step(&self) -> String {
        match self {
            Self::AllDenied { .. } => format!(
                "pick a model another provider serves (BenchFlow never uses {}); `bench run` names every upstream it allows",
                ALWAYS_DENIED.join(", ")
            ),
            Self::NotServed {
                provider,
                close,
                list,
                ..
            } => match (close, list) {
                (Some(close), _) => format!("did you mean `{provider}/{close}`?"),
                (None, Some(list)) => {
                    format!("check the model id; {list} lists every model {provider} serves")
                }
                (None, None) => {
                    format!("check the model id; {provider}'s model list names the ones it serves")
                }
            },
            Self::Unlisted { .. } => {
                "check the network and run again; BenchFlow sends no gateway request it cannot pin to allowed upstreams"
                    .to_string()
            }
            Self::NeedsModel { provider } => {
                format!("name the model with --model, so BenchFlow can pin {provider}'s upstreams for it")
            }
        }
    }
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AllDenied {
                provider,
                model,
                excluded,
            } => write!(
                f,
                "no upstream {provider} could send `{model}` to is allowed: it is served only by {}, which BenchFlow does not use",
                excluded.join(", ")
            ),
            Self::NotServed {
                provider, model, ..
            } => {
                write!(f, "{provider} serves no model `{model}`")
            }
            Self::Unlisted {
                provider,
                model,
                detail,
            } => write!(
                f,
                "could not list the upstreams {provider} would send `{model}` to ({detail})"
            ),
            Self::NeedsModel { provider } => write!(
                f,
                "{provider} picks an upstream per model, and the run names no model to pin"
            ),
        }
    }
}

impl std::error::Error for PinError {}

/// Every upstream denied for `routing` here, lowercased and without
/// repeats: [`ALWAYS_DENIED`], the file's `deny`, then [`DENY_ENV`].
pub fn denylist(routing: &Routing, host: &HostEnv) -> Vec<String> {
    let from_env = host
        .get(DENY_ENV)
        .map(|(v, _)| v.to_string())
        .unwrap_or_default();
    let mut denied: Vec<String> = Vec::new();
    for name in ALWAYS_DENIED
        .iter()
        .map(|s| s.to_string())
        .chain(routing.deny.iter().cloned())
        .chain(
            from_env
                .split(|c: char| c == ',' || c.is_whitespace())
                .map(str::to_string),
        )
    {
        let name = name.trim().to_ascii_lowercase();
        if !name.is_empty() && !denied.contains(&name) {
            denied.push(name);
        }
    }
    denied
}

/// The upstream lists read so far in this process, by URL.
static LISTED: LazyLock<Mutex<HashMap<String, Vec<String>>>> = LazyLock::new(Mutex::default);

/// What a listing request may take.
const LIST_TIMEOUT: Duration = Duration::from_secs(20);

/// Why a listing failed.
enum ListError {
    NotFound,
    Failed(String),
}

/// The upstreams serving `model`, as the router names them: read once per
/// process and URL.
async fn upstreams(endpoints: &str, model: &str) -> Result<Vec<String>, ListError> {
    let url = endpoints.replace("{model}", model);
    if let Some(found) = LISTED.lock().unwrap_or_else(|p| p.into_inner()).get(&url) {
        return Ok(found.clone());
    }
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(LIST_TIMEOUT)
        .user_agent(concat!("benchflow/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| ListError::Failed(format!("no HTTP client: {e}")))?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| ListError::Failed(format!("{url}: {}", short_error(&e))))?;
    let status = response.status().as_u16();
    if status == 404 {
        return Err(ListError::NotFound);
    }
    if !(200..300).contains(&status) {
        return Err(ListError::Failed(format!("{url} answered HTTP {status}")));
    }
    let body: Value = response.json().await.map_err(|e| {
        ListError::Failed(format!(
            "{url} answered with something that is not JSON: {e}"
        ))
    })?;
    let names = parse_upstreams(&body)
        .ok_or_else(|| ListError::Failed(format!("{url} has no data.endpoints list")))?;
    LISTED
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .insert(url, names.clone());
    Ok(names)
}

/// Where a router lists its models: its per-model endpoints URL up to the
/// model (`https://ai-gateway.vercel.sh/v1/models/{model}/endpoints` gives
/// `https://ai-gateway.vercel.sh/v1/models`).
pub(crate) fn models_url(endpoints: &str) -> Option<String> {
    let (base, _) = endpoints.split_once("/{model}")?;
    Some(base.to_string())
}

/// The model ids a router lists (`data[].id`), or `None` when the list
/// cannot be read. Only for a suggestion, so it never stops a run.
async fn model_ids(url: &str) -> Option<Vec<String>> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(LIST_TIMEOUT)
        .user_agent(concat!("benchflow/", env!("CARGO_PKG_VERSION")))
        .build()
        .ok()?;
    let body: Value = client.get(url).send().await.ok()?.json().await.ok()?;
    Some(parse_model_ids(&body))
}

/// `data[].id` of a model list.
pub(crate) fn parse_model_ids(body: &Value) -> Vec<String> {
    body.get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|m| m.get("id").and_then(Value::as_str).map(str::to_string))
        .collect()
}

/// The listed model a misspelled one most likely meant.
pub(crate) fn closest_model(model: &str, ids: &[String]) -> Option<String> {
    crate::suggest::did_you_mean(model, ids.iter().map(String::as_str))
}

fn short_error(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        format!("no answer within {}s", LIST_TIMEOUT.as_secs())
    } else if e.is_connect() {
        "could not connect".to_string()
    } else {
        "the request failed".to_string()
    }
}

/// `data.endpoints[].provider_name`, lowercased, without repeats.
pub(crate) fn parse_upstreams(body: &Value) -> Option<Vec<String>> {
    let endpoints = body.get("data")?.get("endpoints")?.as_array()?;
    let mut names: Vec<String> = Vec::new();
    for endpoint in endpoints {
        if let Some(name) = endpoint.get("provider_name").and_then(Value::as_str) {
            let name = name.trim().to_ascii_lowercase();
            if !name.is_empty() && !names.contains(&name) {
                names.push(name);
            }
        }
    }
    Some(names)
}

/// How `provider`'s requests for `model` are pinned: the router's list
/// without the denied ones, or the denied ones to ignore.
pub async fn pin(
    provider: &str,
    routing: &Routing,
    model: Option<&str>,
    host: &HostEnv,
) -> Result<Pin, PinError> {
    let denied = denylist(routing, host);
    match &routing.style {
        Style::Ignore => Ok(Pin {
            model: model.map(str::to_string),
            upstreams: Upstreams::Ignore(denied),
            excluded: Vec::new(),
        }),
        Style::Only { endpoints } => {
            let Some(model) = model else {
                return Err(PinError::NeedsModel {
                    provider: provider.to_string(),
                });
            };
            let serving = upstreams(endpoints, model).await.map_err(|e| match e {
                ListError::NotFound => PinError::NotServed {
                    provider: provider.to_string(),
                    model: model.to_string(),
                    close: None,
                    list: models_url(endpoints),
                },
                ListError::Failed(detail) => PinError::Unlisted {
                    provider: provider.to_string(),
                    model: model.to_string(),
                    detail,
                },
            });
            let serving = match serving {
                Err(PinError::NotServed {
                    provider,
                    model,
                    list: Some(list),
                    ..
                }) => {
                    let close = match model_ids(&list).await {
                        Some(ids) => closest_model(&model, &ids),
                        None => None,
                    };
                    return Err(PinError::NotServed {
                        provider,
                        model,
                        close,
                        list: Some(list),
                    });
                }
                other => other?,
            };
            choose(provider, model, &serving, &denied)
        }
    }
}

/// The pin from the upstreams serving a model and the denied ones.
pub(crate) fn choose(
    provider: &str,
    model: &str,
    serving: &[String],
    denied: &[String],
) -> Result<Pin, PinError> {
    if serving.is_empty() {
        return Err(PinError::NotServed {
            provider: provider.to_string(),
            model: model.to_string(),
            close: None,
            list: None,
        });
    }
    let (excluded, only): (Vec<String>, Vec<String>) = serving
        .iter()
        .cloned()
        .partition(|name| denied.contains(name));
    if only.is_empty() {
        return Err(PinError::AllDenied {
            provider: provider.to_string(),
            model: model.to_string(),
            excluded,
        });
    }
    Ok(Pin {
        model: Some(model.to_string()),
        upstreams: Upstreams::Only(only),
        excluded,
    })
}

/// [`pin`] from synchronous code (agent construction, before any sandbox):
/// runs on a thread of its own with its own runtime, so it works inside and
/// outside an async context. It blocks the caller while the router's list is
/// read, once per model and process.
pub fn pin_blocking(
    provider: &str,
    routing: &Routing,
    model: Option<&str>,
    host: &HostEnv,
) -> Result<Pin, PinError> {
    let (provider, routing, model, host) = (
        provider.to_string(),
        routing.clone(),
        model.map(str::to_string),
        host.clone(),
    );
    let failed = |detail: String, provider: &str, model: &Option<String>| PinError::Unlisted {
        provider: provider.to_string(),
        model: model.clone().unwrap_or_default(),
        detail,
    };
    let (p, m) = (provider.clone(), model.clone());
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| failed(format!("no runtime: {e}"), &provider, &model))?;
        runtime.block_on(pin(&provider, &routing, model.as_deref(), &host))
    })
    .join()
    .unwrap_or_else(|_| Err(failed("the listing thread panicked".into(), &p, &m)))
}

/// The routing for a native endpoint on a router: `vercel/` and
/// `vercel-anthropic/` models (whatever their base URL: the gateway's own
/// listing at the gateway's host, else the same path under the base URL, a
/// proxy or a test's fake server), `openrouter/` models, and any base URL on
/// a router's host. `None` for other providers.
pub fn for_native_endpoint(provider: &str, base_url: &str) -> Option<Routing> {
    if let Some(routing) = Routing::for_host(base_url) {
        return Some(routing);
    }
    if provider == "vercel" || provider.starts_with("vercel-") {
        return Some(Routing {
            style: Style::Only {
                endpoints: format!(
                    "{}/models/{{model}}/endpoints",
                    base_url.trim_end_matches('/')
                ),
            },
            deny: Vec::new(),
        });
    }
    (provider == "openrouter").then(Routing::openrouter)
}

/// The host of an `http(s)://host[:port]/...` URL.
pub(crate) fn host_of(url: &str) -> Option<&str> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))?;
    let authority = rest.split('/').next()?;
    let host = authority.rsplit('@').next()?;
    Some(host.split(':').next().unwrap_or(host))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::json;

    use super::*;

    fn host(vars: &[(&str, &str)]) -> HostEnv {
        HostEnv::new(
            vars.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            None,
        )
    }

    fn names(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn baseten_is_always_denied_and_more_can_be_added() {
        let routing = Routing {
            deny: names(&["Wafer"]),
            ..Routing::vercel()
        };
        assert_eq!(denylist(&routing, &host(&[])), ["baseten", "wafer"]);
        assert_eq!(
            denylist(
                &routing,
                &host(&[(DENY_ENV, "Together, fireworks  baseten")])
            ),
            ["baseten", "wafer", "together", "fireworks"]
        );
        // An empty file list cannot remove it.
        assert_eq!(denylist(&Routing::vercel(), &host(&[])), ["baseten"]);
        assert_eq!(denylist(&Routing::openrouter(), &host(&[])), ["baseten"]);
    }

    #[test]
    fn the_pin_names_every_allowed_upstream_and_nothing_denied() {
        let serving = names(&["alibaba", "baseten", "novita"]);
        let pin = choose("vercel", "deepseek/x", &serving, &names(&["baseten"])).unwrap();
        assert_eq!(
            pin.upstreams,
            Upstreams::Only(names(&["alibaba", "novita"]))
        );
        assert_eq!(pin.excluded, ["baseten"]);
        // Served only by denied upstreams: refused, naming them.
        let error = choose(
            "vercel",
            "m/only-baseten",
            &names(&["baseten"]),
            &names(&["baseten"]),
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "no upstream vercel could send `m/only-baseten` to is allowed: it is served only by baseten, which BenchFlow does not use"
        );
        assert!(error.next_step().contains("never uses baseten"));
        assert!(matches!(
            choose("vercel", "m", &[], &names(&["baseten"])),
            Err(PinError::NotServed { .. })
        ));
    }

    #[tokio::test]
    async fn openrouter_ignores_the_denied_upstreams_and_needs_no_listing() {
        let pin = pin(
            "openrouter",
            &Routing::openrouter(),
            None,
            &host(&[(DENY_ENV, "together")]),
        )
        .await
        .unwrap();
        assert_eq!(
            pin.upstreams,
            Upstreams::Ignore(names(&["baseten", "together"]))
        );
        // A listing router needs the model.
        assert_eq!(
            pin_blocking("vercel", &Routing::vercel(), None, &host(&[])).unwrap_err(),
            PinError::NeedsModel {
                provider: "vercel".into()
            }
        );
    }

    #[test]
    fn the_gateway_listing_is_read_from_data_endpoints() {
        let body = json!({"data": {"id": "m", "endpoints": [
            {"provider_name": "Alibaba", "name": "alibaba"},
            {"provider_name": "baseten"},
            {"provider_name": "alibaba"},
            {"name": "no provider name"},
        ]}});
        assert_eq!(parse_upstreams(&body).unwrap(), ["alibaba", "baseten"]);
        assert_eq!(parse_upstreams(&json!({"data": {}})), None);
    }

    #[test]
    fn only_is_merged_into_the_body_and_replaces_the_agents() {
        let only = Upstreams::Only(names(&["alibaba", "novita"]));
        let mut body = json!({"model": "m", "providerOptions": {
            "gateway": {"only": ["baseten"], "order": ["novita"]},
            "anthropic": {"thinking": true},
        }});
        assert!(only.apply(&mut body));
        assert_eq!(
            body["providerOptions"],
            json!({"gateway": {"only": ["alibaba", "novita"], "order": ["novita"]}, "anthropic": {"thinking": true}})
        );
        let mut plain = json!({"model": "m"});
        only.apply(&mut plain);
        assert_eq!(
            plain["providerOptions"],
            json!({"gateway": {"only": ["alibaba", "novita"]}})
        );
        // Odd shapes are replaced, not kept.
        let mut odd = json!({"providerOptions": "x"});
        only.apply(&mut odd);
        assert_eq!(
            odd["providerOptions"]["gateway"]["only"],
            json!(["alibaba", "novita"])
        );
        let mut not_an_object = json!(["a"]);
        assert!(!only.apply(&mut not_an_object));
    }

    /// OpenRouter's documented provider-routing shape: `provider.ignore` is a
    /// list of provider slugs to skip; the agent's own preferences (`order`,
    /// `allow_fallbacks`, its own `ignore` entries) are kept.
    #[test]
    fn ignore_is_added_to_openrouters_provider_preferences() {
        let ignore = Upstreams::Ignore(names(&["baseten"]));
        let mut body = json!({
            "model": "deepseek/deepseek-chat",
            "messages": [],
            "provider": {"order": ["novita"], "allow_fallbacks": false, "ignore": ["together", "baseten"]},
        });
        assert!(ignore.apply(&mut body));
        assert_eq!(
            body["provider"],
            json!({"order": ["novita"], "allow_fallbacks": false, "ignore": ["together", "baseten"]})
        );
        let mut plain = json!({"model": "m", "messages": []});
        ignore.apply(&mut plain);
        assert_eq!(plain["provider"], json!({"ignore": ["baseten"]}));
        let mut odd = json!({"model": "m", "provider": "x"});
        ignore.apply(&mut odd);
        assert_eq!(odd["provider"], json!({"ignore": ["baseten"]}));
    }

    #[test]
    fn routers_are_known_by_host_and_native_name() {
        assert_eq!(
            for_native_endpoint("vercel", "https://ai-gateway.vercel.sh/v1"),
            Some(Routing::vercel())
        );
        assert_eq!(
            for_native_endpoint("vercel-anthropic", "http://127.0.0.1:9/v1/").map(|r| r.style),
            Some(Style::Only {
                endpoints: "http://127.0.0.1:9/v1/models/{model}/endpoints".into()
            })
        );
        assert_eq!(
            for_native_endpoint("custom", "https://ai-gateway.vercel.sh/v1"),
            Some(Routing::vercel())
        );
        assert_eq!(
            for_native_endpoint("openrouter", "https://openrouter.ai/api/v1"),
            Some(Routing::openrouter())
        );
        assert_eq!(
            for_native_endpoint("openrouter", "http://127.0.0.1:9/v1"),
            Some(Routing::openrouter())
        );
        assert_eq!(
            for_native_endpoint("openai", "https://api.openai.com/v1"),
            None
        );
        assert_eq!(
            host_of("https://user@ai-gateway.vercel.sh:443/v1"),
            Some("ai-gateway.vercel.sh")
        );
    }
}
