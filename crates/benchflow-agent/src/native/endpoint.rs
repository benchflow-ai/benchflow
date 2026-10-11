//! Where a native loop sends its model requests: the API shape, the base
//! URL, the key, and the model name the provider expects.
//!
//! A model is named `<provider>/<model>` (`vercel/deepseek/deepseek-v4-flash-0731`,
//! `anthropic/claude-haiku-4-5`). [`EndpointResolver`] turns that name into
//! an [`Endpoint`]. [`BuiltinProviders`] knows the providers BenchFlow ships
//! with, and falls back to the provider registry (`providers/`) for the
//! rest.

use std::fmt;

use crate::credentials::HostEnv;

/// The wire format of a model endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    /// `POST {base}/chat/completions` (OpenAI and every compatible server).
    OpenAiChat,
    /// `POST {base}/messages` (Anthropic and compatible servers).
    AnthropicMessages,
}

impl Api {
    pub fn as_str(self) -> &'static str {
        match self {
            Api::OpenAiChat => "openai-chat",
            Api::AnthropicMessages => "anthropic-messages",
        }
    }
}

/// A key that never shows in `Debug` output or logs.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    pub fn new(key: impl Into<String>) -> Self {
        Self(key.into().trim().to_string())
    }
    /// The value, for the one header that carries it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

/// A resolved model endpoint.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub api: Api,
    /// The base URL, without a trailing slash (`https://ai-gateway.vercel.sh/v1`).
    pub base_url: String,
    pub key: ApiKey,
    /// The model name sent to the provider (`deepseek/deepseek-v4-flash-0731`).
    pub model: String,
    /// The provider, as people see it (`vercel`).
    pub provider: String,
    /// Where the key came from, as people see it (`AI_GATEWAY_API_KEY (environment)`).
    pub key_label: String,
    /// On a router that picks among upstream providers: what every request
    /// names so it reaches no denied upstream (the allowed ones in
    /// `providerOptions.gateway.only` on the Vercel AI Gateway, the denied
    /// ones in `provider.ignore` on OpenRouter), set by
    /// [`crate::providers::routing`] before the run starts. Never Baseten.
    pub upstreams: Option<crate::providers::Upstreams>,
}

impl Endpoint {
    /// The full URL requests go to.
    pub fn url(&self) -> String {
        match self.api {
            Api::OpenAiChat => format!("{}/chat/completions", self.base_url),
            Api::AnthropicMessages => format!("{}/messages", self.base_url),
        }
    }
}

/// Why a model name could not become an endpoint.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum EndpointError {
    #[error("model `{model}` names no provider; native agents take `<provider>/<model>`")]
    NoProvider { model: String },
    #[error("no provider `{provider}` for model `{model}`")]
    UnknownProvider { provider: String, model: String },
    #[error("no key for provider `{provider}`: {wanted} is not set")]
    NoKey { provider: String, wanted: String },
    /// A base URL override that would send the key somewhere unsafe, or
    /// that came from somewhere other than this shell's environment.
    #[error("{message}")]
    BaseUrl { message: String, variable: String },
}

impl EndpointError {
    /// What to do about it.
    pub fn next_step(&self) -> String {
        match self {
            EndpointError::NoProvider { .. } | EndpointError::UnknownProvider { .. } => format!(
                "name the model as <provider>/<model> with one of: {}",
                known_providers().join(", ")
            ),
            // Not `--env`: those values reach the agent's sandbox, and a
            // native loop's key must stay on this machine.
            EndpointError::NoKey { wanted, .. } => {
                format!("export {wanted}=<key> in this shell, then run again")
            }
            EndpointError::BaseUrl { variable, .. } => format!(
                "export {variable}=https://... in this shell (http:// only for this machine: \
                 localhost, 127.0.0.1, ::1), or unset it"
            ),
        }
    }
}

/// Turns a model name into an endpoint. The provider registry implements
/// this when it lands; [`BuiltinProviders`] does until then.
pub trait EndpointResolver: Send + Sync {
    fn resolve(&self, model: &str, host: &HostEnv) -> Result<Endpoint, EndpointError>;
}

struct BuiltinProvider {
    name: &'static str,
    api: Api,
    base_url: &'static str,
    /// Variables holding the key, in the order they are tried.
    keys: &'static [&'static str],
    /// A variable that overrides the base URL.
    base_url_env: Option<&'static str>,
}

const BUILTIN: &[BuiltinProvider] = &[
    BuiltinProvider {
        name: "vercel",
        api: Api::OpenAiChat,
        base_url: "https://ai-gateway.vercel.sh/v1",
        keys: &["AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"],
        base_url_env: Some("AI_GATEWAY_BASE_URL"),
    },
    // The same gateway through its Anthropic-compatible messages API.
    BuiltinProvider {
        name: "vercel-anthropic",
        api: Api::AnthropicMessages,
        base_url: "https://ai-gateway.vercel.sh/v1",
        keys: &["AI_GATEWAY_API_KEY", "VERCEL_AI_GATEWAY_API_KEY"],
        base_url_env: Some("AI_GATEWAY_BASE_URL"),
    },
    BuiltinProvider {
        name: "openai",
        api: Api::OpenAiChat,
        base_url: "https://api.openai.com/v1",
        keys: &["OPENAI_API_KEY"],
        base_url_env: Some("OPENAI_BASE_URL"),
    },
    BuiltinProvider {
        name: "anthropic",
        api: Api::AnthropicMessages,
        base_url: "https://api.anthropic.com/v1",
        keys: &["ANTHROPIC_API_KEY"],
        base_url_env: None,
    },
    BuiltinProvider {
        name: "openrouter",
        api: Api::OpenAiChat,
        base_url: "https://openrouter.ai/api/v1",
        keys: &["OPENROUTER_API_KEY"],
        base_url_env: None,
    },
];

/// The providers BenchFlow knows without a registry: `vercel/` (the Vercel
/// AI Gateway, OpenAI-compatible), `vercel-anthropic/` (the same gateway's
/// Anthropic-compatible API), `openai/`, `anthropic/` and `openrouter/`. `BENCHFLOW_NATIVE_BASE_URL` points any of them at another
/// server (a local fake, a proxy).
#[derive(Debug, Clone, Copy, Default)]
pub struct BuiltinProviders;

/// Overrides every provider's base URL (tests, proxies).
pub const BASE_URL_OVERRIDE_ENV: &str = "BENCHFLOW_NATIVE_BASE_URL";

/// Variables that move where a provider key is sent. They are read only
/// from this shell's environment, never from `--env`, a job file,
/// bench.toml, or a project `.env` (which a cloned repository brings), and
/// only as https, or http to this machine.
pub const BASE_URL_VARIABLES: &[&str] = &[
    BASE_URL_OVERRIDE_ENV,
    "AI_GATEWAY_BASE_URL",
    "OPENAI_BASE_URL",
];

/// The base URL `variable` sets, checked: `None` when it is not set.
pub fn base_url_from(host: &HostEnv, variable: &str) -> Result<Option<String>, EndpointError> {
    let refuse = |message: String| EndpointError::BaseUrl {
        message,
        variable: variable.to_string(),
    };
    if let Some((_, crate::credentials::Origin::AgentEnv)) = host.get(variable) {
        return Err(refuse(format!(
            "{variable} was set for the run (--env, a job file, or bench.toml); it moves where the \
             provider key is sent, so it is read only from this shell's environment"
        )));
    }
    let Some(value) = host.environment(variable) else {
        return Ok(None);
    };
    let value = value.trim().trim_end_matches('/').to_string();
    let url = reqwest::Url::parse(&value)
        .map_err(|e| refuse(format!("{variable} is not a URL ({e})")))?;
    let name = url
        .host_str()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']');
    let loopback = name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    match url.scheme() {
        "https" => Ok(Some(value)),
        "http" if loopback => Ok(Some(value)),
        _ => Err(refuse(format!(
            "{variable} is {}://{}: the provider key would go there unencrypted; it must be \
             https, or http to this machine",
            url.scheme(),
            host_of(&value)
        ))),
    }
}

/// The host of a URL, for messages.
fn host_of(url: &str) -> String {
    reqwest::Url::parse(url)
        .ok()
        .and_then(|u| {
            let host = u.host_str()?.to_string();
            Some(match u.port() {
                Some(p) => format!("{host}:{p}"),
                None => host,
            })
        })
        .unwrap_or_else(|| url.to_string())
}

/// A note for the run's start when a variable sends `model`'s calls (and
/// its key) to a host other than its provider's own; `None` otherwise.
pub fn base_url_note(model: &str, host: &HostEnv) -> Option<String> {
    let endpoint = BuiltinProviders.resolve(model, host).ok()?;
    let own = default_base_url(&endpoint.provider)?;
    let (now, own) = (host_of(&endpoint.base_url), host_of(&own));
    (now != own).then(|| {
        let variable = BASE_URL_VARIABLES
            .iter()
            .find(|v| host.environment(v).is_some())
            .copied()
            .unwrap_or(BASE_URL_OVERRIDE_ENV);
        format!(
            "model calls for {} go to {now} ({variable}), not {own}; the {} key goes with them",
            endpoint.provider, endpoint.provider
        )
    })
}

/// A provider's own base URL.
fn default_base_url(provider: &str) -> Option<String> {
    if let Some(spec) = BUILTIN.iter().find(|p| p.name == provider) {
        return Some(spec.base_url.to_string());
    }
    let registry = crate::providers::ProviderRegistry::builtin();
    let p = registry.find(provider)?;
    p.openai
        .as_ref()
        .map(|e| e.base_url.clone())
        .or_else(|| p.anthropic.as_ref().map(|e| e.base_url.clone()))
}

impl EndpointResolver for BuiltinProviders {
    fn resolve(&self, model: &str, host: &HostEnv) -> Result<Endpoint, EndpointError> {
        let Some((provider, rest)) = model
            .split_once('/')
            .filter(|(p, r)| !p.is_empty() && !r.is_empty())
        else {
            return Err(EndpointError::NoProvider {
                model: model.to_string(),
            });
        };
        let Some(spec) = BUILTIN.iter().find(|p| p.name == provider) else {
            return resolve_from_registry(provider, rest, model, host);
        };
        let found = spec.keys.iter().find_map(|name| {
            host.get(name)
                .filter(|(v, _)| !v.trim().is_empty())
                .map(|(v, origin)| {
                    let origin = match origin {
                        crate::credentials::Origin::Environment => "environment",
                        crate::credentials::Origin::AgentEnv => "--env",
                        crate::credentials::Origin::File => "file",
                    };
                    (v.to_string(), format!("{name} ({origin})"))
                })
        });
        // Then the key files the provider registry knows
        // (`~/.config/benchflow/vercel-ai-gateway.env`).
        let found = found.or_else(|| {
            let registry_name = provider.strip_suffix("-anthropic").unwrap_or(provider);
            let registry = crate::providers::ProviderRegistry::builtin();
            let p = registry.find(registry_name)?;
            crate::providers::find_provider_key(p, host)
                .ok()
                .filter(|k| !k.value().trim().is_empty())
                .map(|k| (k.value().to_string(), k.source.clone()))
        });
        let Some((key, key_label)) = found else {
            return Err(EndpointError::NoKey {
                provider: provider.to_string(),
                wanted: spec.keys.join(" or "),
            });
        };
        let mut base_url = base_url_from(host, BASE_URL_OVERRIDE_ENV)?;
        if base_url.is_none()
            && let Some(variable) = spec.base_url_env
        {
            base_url = base_url_from(host, variable)?;
        }
        let base_url = base_url.unwrap_or_else(|| spec.base_url.to_string());
        Ok(Endpoint {
            api: spec.api,
            base_url: base_url.trim_end_matches('/').to_string(),
            key: ApiKey::new(key),
            model: rest.to_string(),
            provider: provider.to_string(),
            key_label,
            upstreams: None,
        })
    }
}

/// A provider the native loops have no entry for, from the provider
/// registry (`zai/`, `deepseek/`, `mimo/`): its OpenAI-compatible endpoint
/// when it takes a bearer key, else its Anthropic-compatible one when it takes
/// `x-api-key`, the way the native clients send keys.
fn resolve_from_registry(
    provider: &str,
    rest: &str,
    model: &str,
    host: &HostEnv,
) -> Result<Endpoint, EndpointError> {
    use crate::providers::{Auth, ProviderRegistry};
    let unknown = || EndpointError::UnknownProvider {
        provider: provider.to_string(),
        model: model.to_string(),
    };
    let registry = ProviderRegistry::builtin();
    let p = registry.find(provider).ok_or_else(unknown)?;
    let (api, base_url) = if let Some(e) = p.openai.as_ref().filter(|e| e.auth == Auth::Bearer) {
        (Api::OpenAiChat, e.base_url.clone())
    } else if let Some(e) = p.anthropic.as_ref().filter(|e| e.auth == Auth::XApiKey) {
        // The registry's Anthropic base URL stops before the version.
        (Api::AnthropicMessages, format!("{}/v1", e.base_url))
    } else {
        return Err(unknown());
    };
    let key = crate::providers::find_provider_key(p, host)
        .ok()
        .filter(|k| !k.value().trim().is_empty())
        .ok_or_else(|| EndpointError::NoKey {
            provider: provider.to_string(),
            wanted: p.key.env.join(" or "),
        })?;
    let base_url = base_url_from(host, BASE_URL_OVERRIDE_ENV)?.unwrap_or(base_url);
    Ok(Endpoint {
        api,
        base_url: base_url.trim_end_matches('/').to_string(),
        key: ApiKey::new(key.value()),
        model: rest.to_string(),
        provider: provider.to_string(),
        key_label: key.source.clone(),
        upstreams: None,
    })
}

/// Every provider a native loop can name: its own, then the registry's.
pub fn known_providers() -> Vec<String> {
    let mut names: Vec<String> = BUILTIN.iter().map(|p| p.name.to_string()).collect();
    for p in crate::providers::ProviderRegistry::builtin().providers() {
        if !names.contains(&p.name) {
            names.push(p.name.clone());
        }
    }
    names
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    fn host(vars: &[(&str, &str)]) -> HostEnv {
        HostEnv::new(
            vars.iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect::<BTreeMap<_, _>>(),
            None,
        )
    }

    /// The review's case: a cloned repository's bench.toml or job file sets
    /// `BENCHFLOW_NATIVE_BASE_URL=http://evil/v1`, and the MiMo key would go
    /// there. Base URLs come only from this shell's environment, as https
    /// or http to this machine.
    #[test]
    fn base_urls_come_only_from_this_shell_as_https_or_loopback() {
        let key = [("MIMO_API_KEY", "k")];
        // From the run's own variables (--env, a job file, bench.toml).
        let run_set = host(&key).with_overrides(BTreeMap::from([(
            BASE_URL_OVERRIDE_ENV.to_string(),
            "https://evil.example/v1".to_string(),
        )]));
        let err = BuiltinProviders
            .resolve("mimo/mimo-v2.6-flash", &run_set)
            .unwrap_err();
        assert!(matches!(err, EndpointError::BaseUrl { .. }), "{err}");
        assert!(err.to_string().contains("--env"), "{err}");
        // Plain http to another machine, from the shell.
        for model in ["mimo/m", "openai/m"] {
            let h = host(&[
                ("MIMO_API_KEY", "k"),
                ("OPENAI_API_KEY", "k"),
                (BASE_URL_OVERRIDE_ENV, "http://evil.example/v1"),
            ]);
            let err = BuiltinProviders.resolve(model, &h).unwrap_err();
            assert!(err.to_string().contains("unencrypted"), "{model}: {err}");
        }
        // The message never prints the URL's user and password.
        let h = host(&[
            ("MIMO_API_KEY", "k"),
            (BASE_URL_OVERRIDE_ENV, "http://user:tok3n@evil.example/v1"),
        ]);
        let err = BuiltinProviders
            .resolve("mimo/m", &h)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("http://evil.example") && !err.contains("tok3n"),
            "{err}"
        );
        let h = host(&[
            ("OPENAI_API_KEY", "k"),
            ("OPENAI_BASE_URL", "http://10.0.0.1/v1"),
        ]);
        assert!(BuiltinProviders.resolve("openai/m", &h).is_err());
        // https anywhere, http to this machine.
        for url in [
            "https://proxy.example/v1",
            "http://127.0.0.1:8317/v1",
            "http://localhost:9/v1",
            "http://[::1]:9/v1",
        ] {
            let h = host(&[("MIMO_API_KEY", "k"), (BASE_URL_OVERRIDE_ENV, url)]);
            let e = BuiltinProviders.resolve("mimo/m", &h).unwrap();
            assert_eq!(e.base_url, url);
        }
        // The note names the host when it is not the provider's own.
        let h = host(&[
            ("MIMO_API_KEY", "k"),
            (BASE_URL_OVERRIDE_ENV, "https://proxy.example/v1"),
        ]);
        let note = base_url_note("mimo/m", &h).unwrap();
        assert!(
            note.contains("go to proxy.example (BENCHFLOW_NATIVE_BASE_URL)"),
            "{note}"
        );
        assert!(base_url_note("mimo/m", &host(&key)).is_none());
    }

    #[test]
    fn vercel_models_go_to_the_gateway_with_its_key() {
        let e = BuiltinProviders
            .resolve(
                "vercel/deepseek/deepseek-v4-flash-0731",
                &host(&[("AI_GATEWAY_API_KEY", "vck_secret_value")]),
            )
            .unwrap();
        assert_eq!(e.api, Api::OpenAiChat);
        assert_eq!(e.model, "deepseek/deepseek-v4-flash-0731");
        assert_eq!(e.url(), "https://ai-gateway.vercel.sh/v1/chat/completions");
        assert_eq!(e.key_label, "AI_GATEWAY_API_KEY (environment)");
        assert!(!format!("{e:?}").contains("vck_secret_value"));
    }

    #[test]
    fn the_gateway_key_is_also_read_from_the_registrys_key_file() {
        let home = tempfile::tempdir().unwrap();
        let dir = home.path().join(".config/benchflow");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("vercel-ai-gateway.env"),
            "AI_GATEWAY_API_KEY=vck_from_the_file\n",
        )
        .unwrap();
        let h = HostEnv::new(BTreeMap::new(), Some(home.path().to_path_buf()));
        for model in ["vercel/x/y", "vercel-anthropic/x/y"] {
            let e = BuiltinProviders.resolve(model, &h).unwrap();
            assert_eq!(e.key.expose(), "vck_from_the_file");
            assert_eq!(
                e.key_label,
                "AI_GATEWAY_API_KEY in ~/.config/benchflow/vercel-ai-gateway.env"
            );
        }
    }

    #[test]
    fn the_gateway_also_speaks_anthropic_messages() {
        let e = BuiltinProviders
            .resolve(
                "vercel-anthropic/anthropic/claude-haiku-4.5",
                &host(&[("AI_GATEWAY_API_KEY", "vck_secret_value")]),
            )
            .unwrap();
        assert_eq!(e.api, Api::AnthropicMessages);
        assert_eq!(e.model, "anthropic/claude-haiku-4.5");
        assert_eq!(e.url(), "https://ai-gateway.vercel.sh/v1/messages");
    }

    #[test]
    fn anthropic_models_use_the_messages_api() {
        let e = BuiltinProviders
            .resolve(
                "anthropic/claude-haiku-4-5",
                &host(&[("ANTHROPIC_API_KEY", "sk-ant-xxxxxxxx")]),
            )
            .unwrap();
        assert_eq!(e.api, Api::AnthropicMessages);
        assert_eq!(e.url(), "https://api.anthropic.com/v1/messages");
    }

    #[test]
    fn a_missing_key_or_provider_says_what_to_set() {
        let err = BuiltinProviders
            .resolve("vercel/x/y", &host(&[]))
            .unwrap_err();
        assert!(
            err.next_step().contains("AI_GATEWAY_API_KEY"),
            "{}",
            err.next_step()
        );
        let err = BuiltinProviders.resolve("nope/x", &host(&[])).unwrap_err();
        assert!(matches!(err, EndpointError::UnknownProvider { .. }));
        let err = BuiltinProviders
            .resolve("bare-model", &host(&[]))
            .unwrap_err();
        assert!(matches!(err, EndpointError::NoProvider { .. }));
    }

    #[test]
    fn registry_providers_work_for_native_loops_too() {
        // MiMo's token plan: OpenAI-compatible with a bearer key.
        let e = BuiltinProviders
            .resolve(
                "mimo/mimo-v2.6-flash",
                &host(&[("MIMO_API_KEY", "k-123456789")]),
            )
            .unwrap();
        assert_eq!(e.api, Api::OpenAiChat);
        assert_eq!(
            e.url(),
            "https://token-plan-sgp.xiaomimimo.com/v1/chat/completions"
        );
        assert_eq!(e.model, "mimo-v2.6-flash");
        assert_eq!(e.key_label, "MIMO_API_KEY (environment)");
        // Z.ai and DeepSeek, which were refused before.
        for (model, var) in [
            ("zai/glm-5.3", "ZAI_API_KEY"),
            ("deepseek/deepseek-chat", "DEEPSEEK_API_KEY"),
        ] {
            let e = BuiltinProviders
                .resolve(model, &host(&[(var, "k-123456789")]))
                .unwrap();
            assert_eq!(e.api, Api::OpenAiChat, "{model}");
        }
        let err = BuiltinProviders
            .resolve("mimo/mimo-v2.6-flash", &host(&[]))
            .unwrap_err();
        assert!(
            err.next_step().contains("MIMO_API_KEY"),
            "{}",
            err.next_step()
        );
        let err = BuiltinProviders.resolve("nope/x", &host(&[])).unwrap_err();
        assert!(err.next_step().contains("mimo"), "{}", err.next_step());
    }

    #[test]
    fn the_override_points_any_provider_at_another_server() {
        let e = BuiltinProviders
            .resolve(
                "vercel/x/y",
                &host(&[
                    ("AI_GATEWAY_API_KEY", "k-123456789"),
                    (BASE_URL_OVERRIDE_ENV, "http://127.0.0.1:9/v1/"),
                ]),
            )
            .unwrap();
        assert_eq!(e.url(), "http://127.0.0.1:9/v1/chat/completions");
    }
}
