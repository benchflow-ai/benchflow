//! Where a native loop sends its model requests: the API shape, the base
//! URL, the key, and the model name the provider expects.
//!
//! A model is named `<provider>/<model>` (`vercel/deepseek/deepseek-v4-flash-0731`,
//! `anthropic/claude-haiku-4-5`). [`EndpointResolver`] turns that name into
//! an [`Endpoint`]. [`BuiltinProviders`] knows the providers BenchFlow ships
//! with until the provider registry (`providers/`) replaces it behind the
//! same trait.

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
}

impl EndpointError {
    /// What to do about it.
    pub fn next_step(&self) -> String {
        match self {
            EndpointError::NoProvider { .. } | EndpointError::UnknownProvider { .. } => format!(
                "name the model as <provider>/<model> with one of: {}",
                BUILTIN
                    .iter()
                    .map(|p| p.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            EndpointError::NoKey { wanted, .. } => {
                format!("export {wanted}=<key> (or pass --env {wanted}=<key>), then run again")
            }
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
            return Err(EndpointError::UnknownProvider {
                provider: provider.to_string(),
                model: model.to_string(),
            });
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
        let base_url = host
            .get(BASE_URL_OVERRIDE_ENV)
            .or_else(|| spec.base_url_env.and_then(|v| host.get(v)))
            .map(|(v, _)| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| spec.base_url.to_string());
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
