//! A client for a Karotte-compatible run backend (`/api/internal/*`).
//!
//! The routes, parameters, and payloads are Karotte's (Preference Model,
//! MIT License; `backend_client.py` and `docs/extending/backend.md`), so a
//! backend written for Karotte's external agent can drive BenchFlow:
//!
//! | Request | What BenchFlow sends or expects |
//! |---|---|
//! | `GET /api/internal/transcript_length?run_id=` | a JSON number, or 404 for a new run |
//! | `POST /api/internal/create_transcript?run_id=` | (no body) |
//! | `POST /api/internal/append_transcript?run_id=` | `{"event": {...}, "seq": n}`; a repeated `seq` overwrites |
//! | `POST /api/internal/update_run_state?run_id=` | `{"status", "score", "input_tokens", ...}` |
//! | `GET /api/internal/get_message?run_id=` | a `message_added` event, or 404 while there is none |
//! | `POST /api/internal/delete_message?run_id=` | consumes the message; any JSON body |
//!
//! As in Karotte, connection errors, timeouts, and 5xx answers are retried
//! with exponential backoff (capped at 10 s) within a time budget, each
//! attempt with its own deadline; other answers are not retried. When the
//! file named by `BENCHFLOW_BACKEND_TOKEN_PATH` (or the older
//! `KAROTTE_BACKEND_TOKEN_PATH`; default
//! `/var/run/secrets/service-account-token`) exists, every request carries
//! `Authorization: Bearer <its contents>`, read again for each request.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use crate::native::chat::ToolCallRequest;

/// The token file when no variable names one.
pub const DEFAULT_TOKEN_PATH: &str = "/var/run/secrets/service-account-token";
/// The variable naming the token file.
pub const TOKEN_PATH_ENV: &str = "BENCHFLOW_BACKEND_TOKEN_PATH";
/// The variable's older name, read when the new one is unset.
pub const TOKEN_PATH_ENV_OLD: &str = "KAROTTE_BACKEND_TOKEN_PATH";

/// How the client waits and retries.
#[derive(Debug, Clone)]
pub struct BackendOptions {
    /// The deadline of one attempt (Karotte: 30 s).
    pub attempt_timeout: Duration,
    /// How long one request keeps retrying (Karotte: 25 minutes).
    pub retry_budget: Duration,
    pub backoff_base: Duration,
    pub backoff_cap: Duration,
    /// The pause after a `get_message` 404 that came back at once, so a
    /// backend that does not long-poll is not asked in a tight loop.
    pub poll_gap: Duration,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            attempt_timeout: Duration::from_secs(30),
            retry_budget: Duration::from_secs(25 * 60),
            backoff_base: Duration::from_secs(1),
            backoff_cap: Duration::from_secs(10),
            poll_gap: Duration::from_millis(250),
        }
    }
}

/// A backend request that failed.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum BackendError {
    /// No answer (connection refused, timeouts, 5xx) for the whole retry
    /// budget.
    #[error("the backend at {url} did not answer {route} ({why})")]
    Unreachable {
        url: String,
        route: &'static str,
        why: String,
    },
    /// A 4xx answer, or another status the route does not allow.
    #[error("the backend at {url} answered {route} with {status}: {body}")]
    Rejected {
        url: String,
        route: &'static str,
        status: u16,
        body: String,
    },
    /// An answer BenchFlow cannot read.
    #[error("the backend at {url} sent a malformed answer to {route}: {why}")]
    Malformed {
        url: String,
        route: &'static str,
        why: String,
    },
}

impl BackendError {
    /// What the backend's owner should do.
    pub fn next_step(&self) -> String {
        match self {
            BackendError::Unreachable { url, .. } => format!(
                "check that the backend at {url} is running and reachable from this machine, then run again"
            ),
            BackendError::Rejected { route, .. } => format!(
                "fix the backend's {route} route (the backend protocol; see docs/rust/harnesses.md)"
            ),
            BackendError::Malformed { route, .. } if *route == GET_MESSAGE => format!(
                "fix the backend's {GET_MESSAGE}: it must answer 404 while no message is ready, or a message_added event such as \
                 {{\"type\": \"message_added\", \"message\": {{\"role\": \"assistant\", \"content\": \"...\", \"tool_calls\": [...]}}}}"
            ),
            BackendError::Malformed { route, .. } => format!(
                "fix the backend's {route} route (the backend protocol; see docs/rust/harnesses.md)"
            ),
        }
    }

    /// Worth running again as is (the backend was down).
    pub fn transient(&self) -> bool {
        matches!(self, BackendError::Unreachable { .. })
    }
}

pub const TRANSCRIPT_LENGTH: &str = "GET /api/internal/transcript_length";
pub const CREATE_TRANSCRIPT: &str = "POST /api/internal/create_transcript";
pub const APPEND_TRANSCRIPT: &str = "POST /api/internal/append_transcript";
pub const UPDATE_RUN_STATE: &str = "POST /api/internal/update_run_state";
pub const GET_MESSAGE: &str = "GET /api/internal/get_message";
pub const DELETE_MESSAGE: &str = "POST /api/internal/delete_message";

/// The model's next message, as the backend sent it.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendMessage {
    /// The `message_added` event, with a timestamp added when it had none:
    /// it goes back into the transcript as sent.
    pub event: Value,
    pub text: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCallRequest>,
    /// The turn's finish reason in OpenAI's terms (`stop`, `length`,
    /// `tool_calls`, `content_filter`).
    pub finish_reason: Option<String>,
}

/// Read a `get_message` answer the way Karotte validates a
/// `MessageAddedEvent`; `Err` says what is wrong. Only assistant messages
/// are accepted: the backend speaks for the model.
pub fn parse_message(body: &str) -> Result<BackendMessage, String> {
    let mut event: Value =
        serde_json::from_str(body).map_err(|e| format!("the body is not JSON ({e})"))?;
    let obj = event
        .as_object_mut()
        .ok_or("the body is not a JSON object")?;
    match obj.get("type") {
        None => {
            obj.insert("type".into(), "message_added".into());
        }
        Some(Value::String(t)) if t == "message_added" => {}
        Some(other) => return Err(format!("`type` is {other}, not \"message_added\"")),
    }
    if !obj.get("timestamp").is_some_and(Value::is_string) {
        obj.insert("timestamp".into(), now_iso().into());
    }
    let finish_reason = match obj.get("finish_reason") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(other) => return Err(format!("`finish_reason` is {other}, not a string")),
    };
    let message = obj
        .get("message")
        .and_then(Value::as_object)
        .ok_or("`message` is missing or not an object")?;
    match message.get("role") {
        None => {}
        Some(Value::String(r)) if r == "assistant" => {}
        Some(other) => {
            return Err(format!(
                "`message.role` is {other}; the backend speaks for the model, so it must be \"assistant\""
            ));
        }
    }
    let text = match message.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for (i, part) in parts.iter().enumerate() {
                let part = part
                    .as_object()
                    .ok_or_else(|| format!("`message.content[{i}]` is not an object"))?;
                if part.get("type").and_then(Value::as_str) == Some("text") {
                    text.push_str(part.get("text").and_then(Value::as_str).unwrap_or_default());
                }
            }
            text
        }
        Some(other) => {
            return Err(format!(
                "`message.content` is {other}, not a string, a list of parts, or null"
            ));
        }
    };
    let reasoning = match message.get("reasoning_content") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()).filter(|s| !s.is_empty()),
        Some(other) => {
            return Err(format!(
                "`message.reasoning_content` is {other}, not a string"
            ));
        }
    };
    let tool_calls = match message.get("tool_calls") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(calls)) => calls
            .iter()
            .enumerate()
            .map(|(i, c)| parse_tool_call(i, c))
            .collect::<Result<_, _>>()?,
        Some(other) => return Err(format!("`message.tool_calls` is {other}, not a list")),
    };
    Ok(BackendMessage {
        event,
        text,
        reasoning,
        tool_calls,
        finish_reason,
    })
}

fn parse_tool_call(i: usize, call: &Value) -> Result<ToolCallRequest, String> {
    let at = format!("`message.tool_calls[{i}]`");
    let call = call
        .as_object()
        .ok_or_else(|| format!("{at} is not an object"))?;
    let id = call
        .get("id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{at} has no `id`"))?;
    let function = call
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{at} has no `function` object"))?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("{at} has no `function.name`"))?;
    let arguments = match function.get("arguments") {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => {
            return Err(format!(
                "{at}'s `function.arguments` is {other}; it must be a JSON string, as OpenAI sends it"
            ));
        }
    };
    Ok(ToolCallRequest {
        id: id.to_string(),
        name: name.to_string(),
        arguments,
    })
}

/// Now, as Karotte's events carry it (`2026-10-10T12:00:00.123456Z`).
pub fn now_iso() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Micros, true)
}

/// Talks to one backend for one run.
#[derive(Debug, Clone)]
pub struct BackendClient {
    http: reqwest::Client,
    base_url: String,
    run_id: String,
    token_path: PathBuf,
    options: BackendOptions,
}

/// One attempt's result: the status and body, or why there was none.
type Attempt = Result<(u16, String), String>;

impl BackendClient {
    /// A client for `base_url` and `run_id`; `token_path` is the bearer
    /// token file (Karotte's default when `None`).
    pub fn new(
        base_url: &str,
        run_id: &str,
        token_path: Option<PathBuf>,
        options: BackendOptions,
    ) -> Result<Self, String> {
        let base_url = base_url.trim().trim_end_matches('/').to_string();
        let parsed = reqwest::Url::parse(&base_url)
            .map_err(|e| format!("`{base_url}` is not a URL ({e})"))?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(format!("`{base_url}` is not an http or https URL"));
        }
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .user_agent(format!("benchflow/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| format!("could not set up HTTP for the backend: {e}"))?;
        Ok(Self {
            http,
            base_url,
            run_id: run_id.to_string(),
            token_path: token_path.unwrap_or_else(|| PathBuf::from(DEFAULT_TOKEN_PATH)),
            options,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn run_id(&self) -> &str {
        &self.run_id
    }

    fn token(&self) -> Option<String> {
        std::fs::read_to_string(&self.token_path)
            .ok()
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
    }

    async fn attempt(&self, route: &'static str, body: Option<&Value>) -> Attempt {
        let (method, path) = route.split_once(' ').unwrap_or(("GET", route));
        let url = format!("{}{path}", self.base_url);
        let mut request = if method == "POST" {
            self.http.post(&url)
        } else {
            self.http.get(&url)
        }
        .query(&[("run_id", self.run_id.as_str())])
        .timeout(self.options.attempt_timeout);
        if let Some(token) = self.token() {
            request = request.bearer_auth(token);
        }
        if let Some(body) = body {
            request = request.json(body);
        }
        let response = request.send().await.map_err(|e| describe(&e))?;
        let status = response.status().as_u16();
        let text = response.text().await.map_err(|e| describe(&e))?;
        Ok((status, text))
    }

    /// Send one request, retrying connection errors, timeouts, and 5xx
    /// within the retry budget. `ok` lists the statuses the route may
    /// answer besides 2xx.
    async fn request(
        &self,
        route: &'static str,
        body: Option<&Value>,
        ok: &[u16],
    ) -> Result<(u16, String), BackendError> {
        let started = Instant::now();
        let mut wait = self.options.backoff_base;
        loop {
            let why = match self.attempt(route, body).await {
                Ok((status, text)) if (200..300).contains(&status) || ok.contains(&status) => {
                    return Ok((status, text));
                }
                Ok((status, text)) if status < 500 => {
                    return Err(BackendError::Rejected {
                        url: self.base_url.clone(),
                        route,
                        status,
                        body: snippet(&text),
                    });
                }
                Ok((status, text)) => format!("{status}: {}", snippet(&text)),
                Err(why) => why,
            };
            if started.elapsed() + wait > self.options.retry_budget {
                return Err(BackendError::Unreachable {
                    url: self.base_url.clone(),
                    route,
                    why: format!(
                        "{why}; still failing after {}s of retries",
                        started.elapsed().as_secs()
                    ),
                });
            }
            tracing::debug!("{route} failed ({why}); retrying in {wait:?}");
            tokio::time::sleep(wait).await;
            wait = (wait * 2).min(self.options.backoff_cap);
        }
    }

    fn malformed(&self, route: &'static str, why: impl Into<String>) -> BackendError {
        BackendError::Malformed {
            url: self.base_url.clone(),
            route,
            why: why.into(),
        }
    }

    /// Events already stored for the run, or `None` for a new run.
    pub async fn transcript_length(&self) -> Result<Option<u64>, BackendError> {
        let (status, text) = self.request(TRANSCRIPT_LENGTH, None, &[404]).await?;
        if status == 404 {
            return Ok(None);
        }
        serde_json::from_str::<Value>(&text)
            .ok()
            .and_then(|v| v.as_u64())
            .map(Some)
            .ok_or_else(|| {
                self.malformed(
                    TRANSCRIPT_LENGTH,
                    format!("expected a JSON number, got `{}`", snippet(&text)),
                )
            })
    }

    pub async fn create_transcript(&self) -> Result<(), BackendError> {
        self.request(CREATE_TRANSCRIPT, None, &[]).await.map(|_| ())
    }

    /// Store `event` at position `seq`.
    pub async fn append_transcript(&self, event: &Value, seq: u64) -> Result<(), BackendError> {
        let body = json!({"event": event, "seq": seq});
        self.request(APPEND_TRANSCRIPT, Some(&body), &[])
            .await
            .map(|_| ())
    }

    pub async fn update_run_state(&self, state: &Value) -> Result<(), BackendError> {
        self.request(UPDATE_RUN_STATE, Some(state), &[])
            .await
            .map(|_| ())
    }

    /// The next message, or `None` while the backend has none (404).
    pub async fn get_message(&self) -> Result<Option<BackendMessage>, BackendError> {
        let (status, text) = self.request(GET_MESSAGE, None, &[404]).await?;
        if status == 404 {
            return Ok(None);
        }
        parse_message(&text)
            .map(Some)
            .map_err(|why| self.malformed(GET_MESSAGE, why))
    }

    /// Consume the message just returned.
    pub async fn delete_message(&self) -> Result<(), BackendError> {
        let (_, text) = self.request(DELETE_MESSAGE, None, &[]).await?;
        serde_json::from_str::<Value>(&text)
            .map(|_| ())
            .map_err(|_| {
                self.malformed(
                    DELETE_MESSAGE,
                    format!("expected a JSON body, got `{}`", snippet(&text)),
                )
            })
    }

    /// Wait for the next message (Karotte's long poll: ask again after each
    /// 404), consume it, and return it. Unbounded; the caller sets the
    /// idle timeout.
    pub async fn next_message(&self) -> Result<BackendMessage, BackendError> {
        loop {
            let asked = Instant::now();
            if let Some(message) = self.get_message().await? {
                self.delete_message().await?;
                return Ok(message);
            }
            if asked.elapsed() < self.options.poll_gap {
                tokio::time::sleep(self.options.poll_gap).await;
            }
        }
    }
}

/// One line for a request error, with its cause.
fn describe(e: &reqwest::Error) -> String {
    if e.is_timeout() {
        return "timed out".into();
    }
    let mut parts = vec![e.to_string()];
    let mut source = std::error::Error::source(e);
    while let Some(s) = source {
        let text = s.to_string();
        if !parts.iter().any(|p| p.contains(&text)) {
            parts.push(text);
        }
        source = s.source();
    }
    parts.join(": ")
}

/// The first 300 characters of a body, on one line.
fn snippet(text: &str) -> String {
    let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match flat.char_indices().nth(300) {
        Some((i, _)) => format!("{}...", &flat[..i]),
        None => flat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_karotte_message_event_is_read_with_its_tool_calls() {
        let m = parse_message(
            r#"{"type": "message_added", "timestamp": "2026-10-10T00:00:00Z", "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Writing."}],
                            "reasoning_content": "think",
                            "tool_calls": [{"id": "c1", "type": "function",
                                            "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}}"#,
        )
        .unwrap();
        assert_eq!(m.text, "Writing.");
        assert_eq!(m.reasoning.as_deref(), Some("think"));
        assert_eq!(m.tool_calls[0].name, "bash");
        assert_eq!(m.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(m.event["timestamp"], "2026-10-10T00:00:00Z");
    }

    #[test]
    fn type_and_timestamp_are_filled_in_when_missing() {
        let m = parse_message(r#"{"message": {"content": "done"}}"#).unwrap();
        assert_eq!(m.event["type"], "message_added");
        assert!(m.event["timestamp"].as_str().unwrap().ends_with('Z'));
        assert!(m.tool_calls.is_empty());
    }

    #[test]
    fn malformed_messages_say_what_is_wrong() {
        for (body, says) in [
            ("not json", "not JSON"),
            ("[]", "not a JSON object"),
            (r#"{"type": "token_usage"}"#, "not \"message_added\""),
            (r#"{"type": "message_added"}"#, "`message` is missing"),
            (
                r#"{"message": {"role": "user", "content": "x"}}"#,
                "must be \"assistant\"",
            ),
            (r#"{"message": {"content": 7}}"#, "`message.content` is 7"),
            (r#"{"message": {"tool_calls": {"id": "c"}}}"#, "not a list"),
            (
                r#"{"message": {"tool_calls": [{"function": {"name": "bash"}}]}}"#,
                "has no `id`",
            ),
            (
                r#"{"message": {"tool_calls": [{"id": "c", "function": {"name": "bash", "arguments": {"a": 1}}}]}}"#,
                "must be a JSON string",
            ),
        ] {
            let why = parse_message(body).unwrap_err();
            assert!(why.contains(says), "{body}: {why}");
        }
    }

    #[test]
    fn only_http_urls_are_accepted() {
        assert!(BackendClient::new("ftp://x", "r", None, BackendOptions::default()).is_err());
        assert!(BackendClient::new("not a url", "r", None, BackendOptions::default()).is_err());
        let c = BackendClient::new("http://127.0.0.1:9/", "r", None, BackendOptions::default())
            .unwrap();
        assert_eq!(c.base_url(), "http://127.0.0.1:9");
    }
}
