//! The model client: one HTTP request per turn, retried with backoff when
//! the provider is rate limited or failing, every attempt recorded.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::chat::{ChatRequest, ChatResponse, TokenUsage};
use super::endpoint::{Api, Endpoint};
use super::error::{ModelError, ModelErrorKind, backoff, classify, plan_message, provider_message};
use super::{anthropic, openai};
use crate::trajectory::{Redactor, iso_from_ns, to_python_json};

/// How a client retries and waits.
#[derive(Debug, Clone)]
pub struct ClientOptions {
    /// The longest one request may take.
    pub timeout: Duration,
    /// Attempts per call, the first included.
    pub max_attempts: u32,
    pub backoff_base: Duration,
    pub backoff_cap: Duration,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(600),
            max_attempts: 6,
            backoff_base: Duration::from_secs(1),
            backoff_cap: Duration::from_secs(30),
        }
    }
}

/// `agent/model_calls.jsonl`: one line per attempt, secrets redacted.
struct CallLog {
    file: File,
    redactor: Redactor,
}

#[derive(Default)]
struct Totals {
    usage: TokenUsage,
    calls: u64,
    seq: u64,
}

/// Calls one model endpoint.
pub struct ModelClient {
    http: reqwest::Client,
    endpoint: Endpoint,
    options: ClientOptions,
    log: Option<Mutex<CallLog>>,
    totals: Mutex<Totals>,
}

impl std::fmt::Debug for ModelClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelClient")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

fn now_ns() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

impl ModelClient {
    pub fn new(endpoint: Endpoint, options: ClientOptions) -> Result<Self, ModelError> {
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .user_agent(format!("benchflow/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| {
                ModelError::new(
                    ModelErrorKind::Failed,
                    None,
                    format!("could not set up HTTPS for model calls: {e}"),
                )
            })?;
        Ok(Self {
            http,
            endpoint,
            options,
            log: None,
            totals: Mutex::new(Totals::default()),
        })
    }

    /// Record every attempt to `path` (JSON lines), redacting the key and
    /// `redactor`'s secrets.
    pub fn with_log(mut self, path: &Path, redactor: Redactor) -> Self {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let redactor = redactor.with_secrets([self.endpoint.key.expose().to_string()]);
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => self.log = Some(Mutex::new(CallLog { file, redactor })),
            Err(e) => tracing::warn!("cannot record model calls to {}: {e}", path.display()),
        }
        self
    }

    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Tokens and cost of every successful call so far.
    pub fn usage(&self) -> TokenUsage {
        self.totals.lock().map(|t| t.usage).unwrap_or_default()
    }

    /// Successful calls so far.
    pub fn calls(&self) -> u64 {
        self.totals.lock().map(|t| t.calls).unwrap_or(0)
    }

    /// Send `req` and read the answer. `purpose` labels the call in the log
    /// (`turn`, `summary`).
    pub async fn chat(&self, req: &ChatRequest, purpose: &str) -> Result<ChatResponse, ModelError> {
        let mut body = match self.endpoint.api {
            Api::OpenAiChat => openai::request_body(&self.endpoint.model, req),
            Api::AnthropicMessages => anthropic::request_body(&self.endpoint.model, req),
        };
        // On a router, never a denied upstream (never Baseten).
        if let Some(upstreams) = &self.endpoint.upstreams {
            upstreams.apply(&mut body);
        }
        let seq = self
            .totals
            .lock()
            .map(|mut t| {
                t.seq += 1;
                t.seq
            })
            .unwrap_or(0);
        let mut attempt = 0;
        loop {
            attempt += 1;
            let started_ns = now_ns();
            let started = Instant::now();
            let result = self.attempt(&body).await;
            let elapsed = started.elapsed();
            let (mut error, retry, retry_after) = match result {
                Ok((status, text, retry_after)) => {
                    let parsed: Option<Value> = serde_json::from_str(&text).ok();
                    let record = parsed.clone().unwrap_or(Value::String(text.clone()));
                    if (200..300).contains(&status) {
                        let parsed_response = parsed
                            .as_ref()
                            .ok_or_else(|| "the answer is not JSON".to_string())
                            .and_then(|v| match self.endpoint.api {
                                Api::OpenAiChat => openai::parse_response(v),
                                Api::AnthropicMessages => anthropic::parse_response(v),
                            });
                        match parsed_response {
                            Ok(response) => {
                                self.record(
                                    seq,
                                    attempt,
                                    purpose,
                                    started_ns,
                                    elapsed,
                                    &body,
                                    Some(status),
                                    &record,
                                    Some(&response.usage),
                                    None,
                                );
                                if let Ok(mut t) = self.totals.lock() {
                                    t.usage.add(&response.usage);
                                    t.calls += 1;
                                }
                                return Ok(response);
                            }
                            Err(why) => {
                                let message = format!(
                                    "{} answered with something BenchFlow cannot read ({why})",
                                    self.endpoint.provider
                                );
                                self.record(
                                    seq,
                                    attempt,
                                    purpose,
                                    started_ns,
                                    elapsed,
                                    &body,
                                    Some(status),
                                    &record,
                                    None,
                                    Some(&message),
                                );
                                // A cut-off or garbled body is worth another try.
                                (
                                    ModelError::new(ModelErrorKind::Failed, Some(status), message),
                                    true,
                                    None,
                                )
                            }
                        }
                    } else {
                        let (kind, retry) = classify(status, &text);
                        let message =
                            plan_message(&self.endpoint.provider, status, &provider_message(&text));
                        self.record(
                            seq,
                            attempt,
                            purpose,
                            started_ns,
                            elapsed,
                            &body,
                            Some(status),
                            &record,
                            None,
                            Some(&message),
                        );
                        (
                            ModelError::new(kind, Some(status), message),
                            retry,
                            retry_after,
                        )
                    }
                }
                Err(e) => {
                    let message = if e.is_timeout() {
                        format!(
                            "{} did not answer within {}s",
                            self.endpoint.provider,
                            self.options.timeout.as_secs()
                        )
                    } else {
                        format!(
                            "could not reach {} at {}: {}",
                            self.endpoint.provider,
                            self.endpoint.base_url,
                            error_chain(&e)
                        )
                    };
                    self.record(
                        seq,
                        attempt,
                        purpose,
                        started_ns,
                        elapsed,
                        &body,
                        None,
                        &Value::Null,
                        None,
                        Some(&message),
                    );
                    (
                        ModelError::new(ModelErrorKind::Unavailable, None, message),
                        true,
                        None,
                    )
                }
            };
            error.attempts = attempt;
            if !retry || attempt >= self.options.max_attempts {
                if retry && error.kind == ModelErrorKind::Failed {
                    error.kind = ModelErrorKind::Unavailable;
                }
                return Err(error);
            }
            let wait = backoff(
                self.options.backoff_base,
                self.options.backoff_cap,
                attempt,
                retry_after,
            );
            tracing::debug!("model call failed ({error}); retrying in {wait:?}");
            tokio::time::sleep(wait).await;
        }
    }

    async fn attempt(
        &self,
        body: &Value,
    ) -> Result<(u16, String, Option<Duration>), reqwest::Error> {
        let mut request = self
            .http
            .post(self.endpoint.url())
            .timeout(self.options.timeout)
            .json(body);
        request = match self.endpoint.api {
            Api::OpenAiChat => request.bearer_auth(self.endpoint.key.expose()),
            Api::AnthropicMessages => request
                .header("x-api-key", self.endpoint.key.expose())
                .header("anthropic-version", anthropic::VERSION),
        };
        let response = request.send().await?;
        let status = response.status().as_u16();
        let retry_after = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s >= 0.0)
            .map(Duration::from_secs_f64);
        let text = response.text().await?;
        Ok((status, text, retry_after))
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        seq: u64,
        attempt: u32,
        purpose: &str,
        started_ns: u64,
        elapsed: Duration,
        request: &Value,
        status: Option<u16>,
        response: &Value,
        usage: Option<&TokenUsage>,
        error: Option<&str>,
    ) {
        let Some(log) = &self.log else { return };
        let Ok(mut log) = log.lock() else { return };
        let mut request = request.clone();
        shrink_images(&mut request);
        let mut line = json!({
            "seq": seq,
            "attempt": attempt,
            "purpose": purpose,
            "started_at": iso_from_ns(started_ns),
            "duration_ms": elapsed.as_millis() as u64,
            "provider": self.endpoint.provider,
            "api": self.endpoint.api.as_str(),
            "url": self.endpoint.url(),
            "request": request,
            "status": status,
            "response": response,
            "usage": usage,
            "error": error,
        });
        log.redactor.redact_json(&mut line);
        let _ = writeln!(log.file, "{}", to_python_json(&line));
        let _ = log.file.flush();
    }
}

/// Replace the bytes of images in a request with a note of their size, so
/// the call log keeps the conversation without megabytes of base64
/// (computer-1 keeps its screenshots as files): OpenAI's `data:` URLs and
/// Anthropic's base64 image sources.
fn shrink_images(v: &mut Value) {
    match v {
        Value::String(s) if s.starts_with("data:image/") => {
            let kind = s.split(';').next().unwrap_or("data:image").to_string();
            *s = format!("{kind};base64,<{} characters not logged>", s.len());
        }
        Value::Object(map) => {
            let is_image_source = map.get("type").and_then(Value::as_str) == Some("base64")
                && map.contains_key("media_type");
            for (k, child) in map.iter_mut() {
                match child {
                    Value::String(s) if is_image_source && k == "data" => {
                        *s = format!("<{} characters not logged>", s.len());
                    }
                    _ => shrink_images(child),
                }
            }
        }
        Value::Array(items) => items.iter_mut().for_each(shrink_images),
        _ => {}
    }
}

/// The error and its causes, one line.
fn error_chain(e: &dyn std::error::Error) -> String {
    let mut parts = vec![e.to_string()];
    let mut source = e.source();
    while let Some(s) = source {
        let text = s.to_string();
        if !parts.iter().any(|p| p.contains(&text)) {
            parts.push(text);
        }
        source = s.source();
    }
    parts.join(": ")
}

/// Where a trial keeps the model calls.
pub fn calls_path(trial_dir: &Path) -> PathBuf {
    trial_dir.join("agent").join("model_calls.jsonl")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_bytes_are_not_logged() {
        let big = "A".repeat(1000);
        let mut v = json!({"messages": [{"content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{big}")}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": big}}
        ]}]});
        shrink_images(&mut v);
        let parts = &v["messages"][0]["content"];
        assert_eq!(parts[0]["text"], "look");
        assert_eq!(
            parts[1]["image_url"]["url"],
            "data:image/png;base64,<1022 characters not logged>"
        );
        assert_eq!(parts[2]["source"]["data"], "<1000 characters not logged>");
    }
}
