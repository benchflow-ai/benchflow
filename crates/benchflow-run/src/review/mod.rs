//! Model-assisted review: a model reads a task and judges its quality
//! ([`task`], `bench check --review`), or reads failed trials and says which
//! part failed ([`analyze`], `bench results --analyze`).
//!
//! What a model says here is an opinion. Everything it returns is parsed
//! into a fixed shape, the places it points at are checked against the files
//! (a line that does not exist is dropped, a quote that is not in the file is
//! marked as such), and every output says it is the model's opinion. The
//! checker's own results and BenchFlow's recorded problems are shown first
//! and never changed by the model.
//!
//! The model is called from the host with the native harness's client
//! ([`benchflow_agent::native::ModelClient`]), so on the Vercel AI Gateway
//! every request names the allowed upstreams (`providerOptions.gateway.only`,
//! never Baseten). The key comes from the provider registry: the provider's
//! variables, then its `~/.config/benchflow/<provider>.env` file. What is
//! sent is size-capped and redacted ([`evidence`]): the provider's key, the
//! secrets the caller names, and every value BenchFlow's redaction patterns
//! recognize never reach a request, and files that usually hold credentials
//! (`.env`, private keys) are not sent at all.

pub mod analyze;
pub mod evidence;
pub mod task;

use benchflow_agent::credentials::HostEnv;
use benchflow_agent::native::chat::{ChatRequest, FinishReason, Message};
use benchflow_agent::native::endpoint::{ApiKey, BASE_URL_OVERRIDE_ENV};
use benchflow_agent::native::{
    Api, BuiltinProviders, ClientOptions, Endpoint, EndpointResolver, ModelClient, TokenUsage,
};
use benchflow_agent::providers::routing::{self, Routing};
use benchflow_agent::providers::{Auth, OpenAiApi, PinError, ProviderRegistry, find_provider_key};
use benchflow_agent::trajectory::Redactor;
use serde::Serialize;
use serde_json::Value;

/// The model a review uses when none is named.
pub const DEFAULT_MODEL: &str = "vercel/deepseek/deepseek-v4-flash-0731";

/// What every saved review and analysis says about itself.
pub const OPINION: &str =
    "A model's opinion: BenchFlow checked where it points, not whether it is right.";

/// Why a review could not start or finish.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewError {
    pub kind: ReviewErrorKind,
    /// What happened, one line.
    pub message: String,
    /// What to do about it.
    pub next: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReviewErrorKind {
    /// The model name is not `<provider>/<model>`, or names an unknown
    /// provider or one the review client cannot call.
    Model,
    /// No key for the provider on this machine.
    NoKey,
    /// The provider failed, refused, or answered with something unusable.
    Provider,
}

impl std::fmt::Display for ReviewError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ReviewError {}

impl ReviewError {
    fn new(kind: ReviewErrorKind, message: impl Into<String>, next: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            next: next.into(),
        }
    }
}

/// Tokens and cost spent, for people and for the saved files.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Spend {
    /// Model calls that answered.
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// As the provider reported it; `None` when it reported none.
    pub cost_usd: Option<f64>,
}

impl Spend {
    pub fn add(&mut self, other: &Spend) {
        self.calls += other.calls;
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cost_usd = match (self.cost_usd, other.cost_usd) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
        };
    }

    fn add_usage(&mut self, usage: &TokenUsage) {
        self.add(&Spend {
            calls: 1,
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            cost_usd: usage.cost_usd,
        });
    }

    /// `12,345 tokens · $0.0021` (or `cost not reported`).
    pub fn describe(&self) -> String {
        let tokens = thousands(self.input_tokens + self.output_tokens);
        match self.cost_usd {
            Some(c) => format!("{tokens} tokens · ${c:.4}"),
            None if self.calls == 0 => format!("{tokens} tokens"),
            None => format!("{tokens} tokens · cost not reported by the provider"),
        }
    }
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// A model to review with: its name, the client, and what it may never see.
pub struct Reviewer {
    /// As the user named it (`vercel/deepseek/deepseek-v4-flash-0731`).
    pub model: String,
    client: ModelClient,
    redactor: Redactor,
}

impl std::fmt::Debug for Reviewer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reviewer")
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl Reviewer {
    /// Resolve `model` on this machine and, on a gateway, pin it to the
    /// allowed upstreams. `secrets` are values that must never be sent (the
    /// caller's credential variables); the provider's key is added.
    pub async fn connect(
        model: &str,
        host: &HostEnv,
        secrets: Vec<String>,
        options: ClientOptions,
    ) -> Result<Self, ReviewError> {
        let (endpoint, routing) = resolve(model, host)?;
        let endpoint = pin(endpoint, routing, host).await?;
        let redactor = Redactor::new(secrets).with_secrets([endpoint.key.expose().to_string()]);
        let client = ModelClient::new(endpoint, options).map_err(|e| {
            ReviewError::new(
                ReviewErrorKind::Provider,
                e.to_string(),
                "check this machine's network and TLS setup, then run again",
            )
        })?;
        Ok(Self {
            model: model.to_string(),
            client,
            redactor,
        })
    }

    /// Redacts the provider key, the caller's secrets, and BenchFlow's
    /// patterns.
    pub fn redactor(&self) -> &Redactor {
        &self.redactor
    }

    /// Where the key came from, for people (never the value).
    pub fn key_label(&self) -> &str {
        &self.client.endpoint().key_label
    }

    /// Ask for one JSON object with the `required` keys. A reply without
    /// one gets one more try with the reason; a reply cut off at the output
    /// limit gets one more try with twice the limit (reasoning models can
    /// spend it all before answering). A second bad reply is an error that
    /// quotes the start of the reply. `system` and `user` are redacted again
    /// here, so nothing a caller forgot reaches the request.
    async fn ask_json(
        &self,
        system: &str,
        user: &str,
        max_tokens: u32,
        required: &[&str],
    ) -> Result<(Value, Spend), (ReviewError, Spend)> {
        let system = self.redactor.redact(system).into_owned();
        let user = self.redactor.redact(user).into_owned();
        let mut spend = Spend::default();
        let mut messages = vec![Message::user(user)];
        let mut max_tokens = max_tokens;
        for attempt in 1..=2 {
            let req = ChatRequest {
                system: Some(system.clone()),
                messages: messages.clone(),
                max_tokens: Some(max_tokens),
                ..ChatRequest::default()
            };
            let response = match self.client.chat(&req, "review").await {
                Ok(r) => r,
                Err(e) => {
                    return Err((
                        ReviewError::new(
                            ReviewErrorKind::Provider,
                            format!("the model call failed: {e}"),
                            format!(
                                "the provider's problem; try again later or pass --model with another model ({})",
                                self.key_label()
                            ),
                        ),
                        spend,
                    ));
                }
            };
            spend.add_usage(&response.usage);
            let text = response.message.text.clone();
            let cut_off = response.finish == FinishReason::Length;
            let why = match extract_json_with(&text, required) {
                Some(v) if !cut_off => return Ok((v, spend)),
                _ if cut_off => format!(
                    "it was cut off at the output limit ({max_tokens} tokens) before the JSON was complete"
                ),
                _ if text.trim().is_empty() => "it was empty".to_string(),
                _ if extract_json(&text).is_some() => format!(
                    "its JSON object lacks {}",
                    required
                        .iter()
                        .map(|k| format!("`{k}`"))
                        .collect::<Vec<_>>()
                        .join(" and ")
                ),
                _ => "it holds no JSON object".to_string(),
            };
            if attempt == 2 {
                return Err((
                    ReviewError::new(
                        ReviewErrorKind::Provider,
                        format!(
                            "{} did not answer with the JSON asked for, twice ({why}); its answer began: {}",
                            self.model,
                            excerpt(&text, 160)
                        ),
                        "run again, or pass --model with another model; nothing it said is shown as a finding",
                    ),
                    spend,
                ));
            }
            if cut_off {
                // Ask the same again with room to finish.
                max_tokens = max_tokens.saturating_mul(2).min(MAX_OUTPUT_TOKENS);
                continue;
            }
            messages.push(Message::assistant(text));
            messages.push(Message::user(format!(
                "Your answer cannot be used: {why}. Answer again with only the JSON object described above, no other text."
            )));
        }
        unreachable!("the loop returns on its second attempt")
    }
}

/// The most output tokens a retry asks for.
const MAX_OUTPUT_TOKENS: u32 = 32_000;

/// The endpoint for `model` and, on a gateway, how its upstreams are listed.
fn resolve(model: &str, host: &HostEnv) -> Result<(Endpoint, Option<Routing>), ReviewError> {
    let Some((prefix, id)) = model
        .split_once('/')
        .filter(|(p, m)| !p.is_empty() && !m.is_empty())
    else {
        return Err(ReviewError::new(
            ReviewErrorKind::Model,
            format!("`{model}` does not name a provider"),
            format!("name the model as <provider>/<model>, such as {DEFAULT_MODEL}"),
        ));
    };
    let registry = ProviderRegistry::builtin();
    let override_url = host
        .get(BASE_URL_OVERRIDE_ENV)
        .map(|(v, _)| v.trim().trim_end_matches('/').to_string())
        .filter(|v| !v.is_empty());
    let Some(provider) = registry.find(prefix) else {
        // Providers only the native client knows (`vercel-anthropic`).
        let endpoint = BuiltinProviders.resolve(model, host).map_err(|e| {
            let kind = match e {
                benchflow_agent::native::EndpointError::NoKey { .. } => ReviewErrorKind::NoKey,
                _ => ReviewErrorKind::Model,
            };
            ReviewError::new(kind, e.to_string(), e.next_step())
        })?;
        let routing = routing::for_native_endpoint(&endpoint.provider, &endpoint.base_url);
        return Ok((endpoint, routing));
    };
    let (api, base_url) = if let Some(e) = provider
        .openai
        .as_ref()
        .filter(|e| e.auth == Auth::Bearer && e.apis.contains(&OpenAiApi::Chat))
    {
        (Api::OpenAiChat, e.base_url.clone())
    } else if let Some(e) = provider
        .anthropic
        .as_ref()
        .filter(|e| e.auth == Auth::XApiKey)
    {
        (Api::AnthropicMessages, format!("{}/v1", e.base_url))
    } else {
        return Err(ReviewError::new(
            ReviewErrorKind::Model,
            format!(
                "provider `{prefix}` serves no endpoint the review client can call (it needs OpenAI-compatible chat completions with a bearer key, or Anthropic messages with an x-api-key)"
            ),
            format!("pass --model with another provider, such as {DEFAULT_MODEL}"),
        ));
    };
    let key = find_provider_key(provider, host)
        .map_err(|e| ReviewError::new(ReviewErrorKind::NoKey, e.to_string(), e.next_step()))?;
    let base_url = override_url.clone().unwrap_or(base_url);
    let routing = match override_url {
        // A proxy or a test's server lists upstreams under its own URL.
        Some(_) => routing::for_native_endpoint(prefix, &base_url),
        None => provider.upstream_routing(),
    };
    Ok((
        Endpoint {
            api,
            base_url,
            key: ApiKey::new(key.value()),
            model: id.to_string(),
            provider: prefix.to_string(),
            key_label: key.source.clone(),
            upstreams: None,
        },
        routing,
    ))
}

/// On a router, keep every request off denied upstreams (never Baseten):
/// the allowed ones on the Vercel AI Gateway, the denied ones on OpenRouter;
/// a model no allowed upstream serves stops here.
async fn pin(
    mut endpoint: Endpoint,
    routing: Option<Routing>,
    host: &HostEnv,
) -> Result<Endpoint, ReviewError> {
    let Some(routing) = routing else {
        return Ok(endpoint);
    };
    match routing::pin(&endpoint.provider, &routing, Some(&endpoint.model), host).await {
        Ok(pin) => {
            endpoint.upstreams = Some(pin.upstreams);
            Ok(endpoint)
        }
        Err(e) => {
            let kind = match e {
                PinError::Unlisted { .. } => ReviewErrorKind::Provider,
                _ => ReviewErrorKind::Model,
            };
            Err(ReviewError::new(kind, e.to_string(), e.next_step()))
        }
    }
}

/// The first JSON object in a model's answer: the whole answer, a fenced
/// block, or an object with text around it. Reasoning in `<think>` tags is
/// skipped.
pub fn extract_json(text: &str) -> Option<Value> {
    extract_json_with(text, &[])
}

/// As [`extract_json`], but only an object holding every `required` key
/// counts, so an inner object of an answer that was cut off is not taken for
/// the answer.
pub fn extract_json_with(text: &str, required: &[&str]) -> Option<Value> {
    let fits = |v: &Value| required.iter().all(|k| v.get(k).is_some());
    let mut text = text.trim();
    if let Some(end) = text.find("</think>") {
        text = text[end + "</think>".len()..].trim();
    }
    if let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(text)
        && fits(&v)
    {
        return Some(v);
    }
    for (i, _) in text.match_indices('{').take(64) {
        let mut stream = serde_json::Deserializer::from_str(&text[i..]).into_iter::<Value>();
        if let Some(Ok(v @ Value::Object(_))) = stream.next()
            && fits(&v)
        {
            return Some(v);
        }
    }
    None
}

/// The first `max` characters of `text` on one line, for messages.
pub(crate) fn excerpt(text: &str, max: usize) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= max {
        format!("\"{one_line}\"")
    } else {
        let cut: String = one_line.chars().take(max).collect();
        format!("\"{cut}…\"")
    }
}

/// A string field of a model's JSON object, trimmed; empty when missing.
pub(crate) fn text_field(v: &Value, key: &str) -> String {
    match v.get(key) {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

/// The current time, for saved files.
pub(crate) fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_found_in_fences_prose_and_after_reasoning() {
        assert_eq!(extract_json(r#"{"a": 1}"#).unwrap()["a"], 1);
        assert_eq!(
            extract_json("Here you go:\n```json\n{\"a\": 2}\n```\nDone.").unwrap()["a"],
            2
        );
        assert_eq!(
            extract_json("<think>maybe {not json</think>{\"a\": 3}").unwrap()["a"],
            3
        );
        assert_eq!(
            extract_json("first {broken, then {\"a\": 4} trailing").unwrap()["a"],
            4
        );
        assert!(extract_json("no json here").is_none());
        assert!(extract_json("[1, 2]").is_none());
        assert!(extract_json("{\"a\": 1").is_none());
    }

    #[test]
    fn an_inner_object_of_a_cut_off_answer_is_not_the_answer() {
        let cut = r#"{"part": "agent", "evidence": [{"file": "x", "quote": "y"}], "explan"#;
        assert!(
            extract_json(cut).is_some(),
            "without required keys the inner object is found"
        );
        assert!(extract_json_with(cut, &["part", "explanation"]).is_none());
        let whole = r#"{"part": "agent", "explanation": "z", "evidence": []}"#;
        assert!(extract_json_with(whole, &["part", "explanation"]).is_some());
    }

    #[test]
    fn spend_reads_like_a_sentence() {
        let mut s = Spend::default();
        s.add_usage(&TokenUsage {
            input_tokens: 12_000,
            output_tokens: 345,
            cost_usd: Some(0.0021),
            ..TokenUsage::default()
        });
        assert_eq!(s.describe(), "12,345 tokens · $0.0021");
        let mut t = Spend::default();
        t.add_usage(&TokenUsage {
            input_tokens: 10,
            ..TokenUsage::default()
        });
        assert!(t.describe().contains("cost not reported"));
    }

    #[test]
    fn a_model_without_a_provider_says_how_to_name_one() {
        let host = HostEnv::new(Default::default(), None);
        let e = resolve("deepseek-v4", &host).unwrap_err();
        assert_eq!(e.kind, ReviewErrorKind::Model);
        assert!(e.next.contains(DEFAULT_MODEL));
    }

    #[test]
    fn a_missing_key_names_the_variable_and_the_file() {
        let host = HostEnv::new(Default::default(), None);
        let e = resolve(DEFAULT_MODEL, &host).unwrap_err();
        assert_eq!(e.kind, ReviewErrorKind::NoKey);
        assert!(e.next.contains("AI_GATEWAY_API_KEY"), "{}", e.next);
        assert!(e.next.contains("vercel-ai-gateway.env"), "{}", e.next);
    }
}
