//! The Daytona HTTP client: authentication, bounded retries with backoff,
//! and errors that say what failed, whose problem it is, and what to do.
//!
//! Requests carry the key as `Authorization: Bearer <key>` (marked sensitive
//! so it never shows in debug output) plus `X-Daytona-SDK-Version`, which
//! the sandbox daemon reads to decide its response formats (without it,
//! command logs come back as plain text with stdout and stderr mixed).
//! Every message built from a response or a transport error is redacted.

use std::fmt;
use std::time::Duration;

use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use reqwest::{RequestBuilder, Response, StatusCode};
use serde::de::DeserializeOwned;
use tokio_tungstenite::tungstenite;

use crate::SandboxError;
use crate::redact::Redactor;

/// Where to create a Daytona API key.
pub(crate) const KEY_URL: &str = "https://app.daytona.io/dashboard/keys";

/// The SDK version whose API behavior this client follows.
const SDK_VERSION: &str = "0.184.0";

/// The fix for a missing or rejected key.
pub(crate) fn key_fix() -> String {
    format!(
        "Create an API key at {KEY_URL}, then `export DAYTONA_API_KEY=<key>` (and DAYTONA_API_URL \
         if you use your own Daytona)."
    )
}

/// A secret string that never prints.
#[derive(Clone, serde::Deserialize)]
#[serde(transparent)]
pub(crate) struct Secret(String);

impl Secret {
    #[cfg(test)]
    pub(crate) fn new(value: String) -> Self {
        Self(value)
    }

    pub(crate) fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("[redacted]")
    }
}

/// Which failures a request may be repeated after.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Retry {
    /// Idempotent: repeat on 429, 5xx, and any transport error.
    Safe,
    /// Not idempotent: repeat only when the request surely had no effect
    /// (429, or a connection that never opened).
    Unsent,
}

/// A failed API call.
#[derive(Debug)]
pub(crate) enum ApiError {
    /// No key is configured.
    NoKey,
    /// The API answered with an error status (after any retries).
    Status {
        status: u16,
        message: String,
        what: String,
        attempts: u32,
    },
    /// The request did not complete (after any retries).
    Network {
        message: String,
        what: String,
        attempts: u32,
        /// The request may have reached Daytona before failing.
        maybe_sent: bool,
    },
    /// The answer could not be read.
    Decode { message: String, what: String },
}

impl ApiError {
    pub(crate) fn status(&self) -> Option<u16> {
        match self {
            Self::Status { status, .. } => Some(*status),
            _ => None,
        }
    }

    /// True when Daytona may have acted on the request although it failed.
    pub(crate) fn ambiguous(&self) -> bool {
        match self {
            Self::Status { status, .. } => *status >= 500,
            Self::Network { maybe_sent, .. } => *maybe_sent,
            _ => false,
        }
    }

    pub(crate) fn message(&self) -> String {
        match self {
            Self::NoKey => "DAYTONA_API_KEY is not set".to_string(),
            Self::Status {
                status, message, ..
            } => format!("HTTP {status}: {message}"),
            Self::Network { message, .. } | Self::Decode { message, .. } => message.clone(),
        }
    }
}

impl From<ApiError> for SandboxError {
    fn from(err: ApiError) -> Self {
        match err {
            ApiError::NoKey => SandboxError::Unavailable {
                message: "Daytona is not configured: DAYTONA_API_KEY is not set".to_string(),
                fix: key_fix(),
            },
            ApiError::Status {
                status: 401,
                message,
                what,
                ..
            } => SandboxError::Unavailable {
                message: format!("Daytona rejected the API key while {what} (HTTP 401: {message})"),
                fix: key_fix(),
            },
            ApiError::Status {
                status: 403,
                message,
                what,
                ..
            } => {
                let lower = message.to_lowercase().replace("insufficient_scope", "");
                let limit = ["quota", "limit", "exceed", "insufficient"]
                    .iter()
                    .any(|w| lower.contains(w));
                SandboxError::Unavailable {
                    message: format!("Daytona refused {what} (HTTP 403: {message})"),
                    fix: if limit {
                        "Your Daytona organization is at a resource limit. Lower the run's \
                         concurrency or the task's resources, delete idle sandboxes, or raise the \
                         limit in the Daytona dashboard."
                            .to_string()
                    } else {
                        format!(
                            "The API key lacks a permission this needs. Create a key with sandbox \
                             read, write and delete permissions at {KEY_URL} and export it as \
                             DAYTONA_API_KEY."
                        )
                    },
                }
            }
            ApiError::Status {
                status,
                message,
                what,
                attempts,
            } if status == 429 || status >= 500 => SandboxError::Transport(format!(
                "Daytona failed while {what} (HTTP {status}: {message}){}. This is on Daytona's \
                 side and usually passes; run again, with lower concurrency if it keeps happening.",
                tries(attempts)
            )),
            ApiError::Status {
                status,
                message,
                what,
                ..
            } => SandboxError::Unsupported(format!(
                "Daytona refused {what} (HTTP {status}: {message})"
            )),
            ApiError::Network {
                message,
                what,
                attempts,
                ..
            } => SandboxError::Transport(format!(
                "could not reach Daytona while {what} ({message}){}. Check the network (and \
                 DAYTONA_API_URL), then run again.",
                tries(attempts)
            )),
            ApiError::Decode { message, what } => SandboxError::Transport(format!(
                "Daytona's answer while {what} could not be read ({message}). The API may have \
                 changed; check for a newer BenchFlow."
            )),
        }
    }
}

fn tries(attempts: u32) -> String {
    if attempts > 1 {
        format!(", after {attempts} attempts")
    } else {
        String::new()
    }
}

/// The Daytona API.
pub(crate) struct Api {
    http: reqwest::Client,
    /// The API base, without a trailing `/` (`https://app.daytona.io/api`).
    base: String,
    key: Option<Secret>,
    pub(crate) target: Option<String>,
    redactor: Redactor,
}

impl fmt::Debug for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Api")
            .field("base", &self.base)
            .field("key", &self.key.as_ref().map(|_| "[redacted]"))
            .field("target", &self.target)
            .finish()
    }
}

/// How many times a request is tried at most.
const MAX_ATTEMPTS: u32 = 5;

impl Api {
    pub(crate) fn new(
        base: &str,
        key: Option<String>,
        target: Option<String>,
    ) -> Result<Self, SandboxError> {
        // Websockets go through rustls without an explicit config; make sure
        // a crypto provider is installed even if several are compiled in.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(15))
            .user_agent(format!("benchflow/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|e| SandboxError::Transport(format!("could not set up HTTPS: {e}")))?;
        let mut redactor = Redactor::new();
        let key = key.filter(|k| !k.trim().is_empty()).map(|k| {
            let k = k.trim().to_string();
            redactor.add(k.clone());
            Secret(k)
        });
        Ok(Self {
            http,
            base: base.trim_end_matches('/').to_string(),
            key,
            target: target.filter(|t| !t.is_empty()),
            redactor,
        })
    }

    pub(crate) fn has_key(&self) -> bool {
        self.key.is_some()
    }

    pub(crate) fn base(&self) -> &str {
        &self.base
    }

    pub(crate) fn redact(&self, text: &str) -> String {
        self.redactor.redact(text)
    }

    /// `base + path`.
    pub(crate) fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    fn headers(&self) -> Result<HeaderMap, ApiError> {
        let key = self.key.as_ref().ok_or(ApiError::NoKey)?;
        let mut headers = HeaderMap::new();
        let mut auth =
            HeaderValue::from_str(&format!("Bearer {}", key.expose())).map_err(|_| {
                ApiError::Decode {
                    message: "the API key contains characters an HTTP header cannot carry"
                        .to_string(),
                    what: "authenticating".to_string(),
                }
            })?;
        auth.set_sensitive(true);
        headers.insert(AUTHORIZATION, auth);
        headers.insert("X-Daytona-Source", HeaderValue::from_static("benchflow"));
        headers.insert(
            "X-Daytona-SDK-Version",
            HeaderValue::from_static(SDK_VERSION),
        );
        Ok(headers)
    }

    /// Send the request `build` makes, retrying per `retry`. `build` runs
    /// once per attempt, so a body can be made again (a file reopened).
    pub(crate) async fn send<F>(
        &self,
        what: &str,
        retry: Retry,
        mut build: F,
    ) -> Result<Response, ApiError>
    where
        F: FnMut(&reqwest::Client) -> Result<RequestBuilder, ApiError>,
    {
        let headers = self.headers()?;
        let mut attempt = 0;
        loop {
            attempt += 1;
            let request = build(&self.http)?.headers(headers.clone());
            match request.send().await {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status();
                    let wait = retry_after(&response);
                    let body = response.text().await.unwrap_or_default();
                    let again = status == StatusCode::TOO_MANY_REQUESTS
                        || (retry == Retry::Safe && status.is_server_error());
                    if again && attempt < MAX_ATTEMPTS {
                        tracing::debug!("Daytona {what}: HTTP {status}, retrying");
                        tokio::time::sleep(backoff(attempt, wait)).await;
                        continue;
                    }
                    return Err(ApiError::Status {
                        status: status.as_u16(),
                        message: self.redact(&error_message(&body)),
                        what: what.to_string(),
                        attempts: attempt,
                    });
                }
                Err(err) => {
                    let unsent = err.is_connect();
                    if (retry == Retry::Safe || unsent) && attempt < MAX_ATTEMPTS {
                        tracing::debug!("Daytona {what}: {err}, retrying");
                        tokio::time::sleep(backoff(attempt, None)).await;
                        continue;
                    }
                    return Err(ApiError::Network {
                        message: self.redact(&error_chain(&err)),
                        what: what.to_string(),
                        attempts: attempt,
                        maybe_sent: !unsent,
                    });
                }
            }
        }
    }

    /// Send and parse a JSON answer.
    pub(crate) async fn json<T, F>(&self, what: &str, retry: Retry, build: F) -> Result<T, ApiError>
    where
        T: DeserializeOwned,
        F: FnMut(&reqwest::Client) -> Result<RequestBuilder, ApiError>,
    {
        let response = self.send(what, retry, build).await?;
        let bytes = response.bytes().await.map_err(|e| ApiError::Network {
            message: self.redact(&error_chain(&e)),
            what: what.to_string(),
            attempts: 1,
            maybe_sent: true,
        })?;
        serde_json::from_slice(&bytes).map_err(|e| ApiError::Decode {
            message: format!(
                "{e}; the answer began {:?}",
                self.redact(&String::from_utf8_lossy(&bytes[..bytes.len().min(200)]))
            ),
            what: what.to_string(),
        })
    }

    /// Open a websocket to `url` (`https://` becomes `wss://`).
    pub(crate) async fn websocket(
        &self,
        url: &str,
        what: &str,
    ) -> Result<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
        ApiError,
    > {
        use tungstenite::client::IntoClientRequest;
        let ws_url = if let Some(rest) = url.strip_prefix("https://") {
            format!("wss://{rest}")
        } else if let Some(rest) = url.strip_prefix("http://") {
            format!("ws://{rest}")
        } else {
            url.to_string()
        };
        let headers = self.headers()?;
        let config = tungstenite::protocol::WebSocketConfig::default()
            .max_message_size(Some(512 << 20))
            .max_frame_size(Some(128 << 20));
        let mut attempt = 0;
        loop {
            attempt += 1;
            let mut request =
                ws_url
                    .as_str()
                    .into_client_request()
                    .map_err(|e| ApiError::Decode {
                        message: e.to_string(),
                        what: what.to_string(),
                    })?;
            for (name, value) in &headers {
                request.headers_mut().insert(name.clone(), value.clone());
            }
            match tokio::time::timeout(
                Duration::from_secs(30),
                tokio_tungstenite::connect_async_with_config(request, Some(config), true),
            )
            .await
            {
                Ok(Ok((stream, _))) => return Ok(stream),
                Ok(Err(tungstenite::Error::Http(response))) => {
                    let status = response.status().as_u16();
                    let body = response
                        .body()
                        .as_ref()
                        .map(|b| String::from_utf8_lossy(b).into_owned())
                        .unwrap_or_default();
                    if (status == 429 || status >= 500) && attempt < MAX_ATTEMPTS {
                        tokio::time::sleep(backoff(attempt, None)).await;
                        continue;
                    }
                    return Err(ApiError::Status {
                        status,
                        message: self.redact(&error_message(&body)),
                        what: what.to_string(),
                        attempts: attempt,
                    });
                }
                Ok(Err(err)) if attempt < MAX_ATTEMPTS => {
                    tracing::debug!("Daytona {what}: websocket {err}, retrying");
                    tokio::time::sleep(backoff(attempt, None)).await;
                }
                Ok(Err(err)) => {
                    return Err(ApiError::Network {
                        message: self.redact(&err.to_string()),
                        what: what.to_string(),
                        attempts: attempt,
                        maybe_sent: true,
                    });
                }
                Err(_) if attempt < MAX_ATTEMPTS => {
                    tokio::time::sleep(backoff(attempt, None)).await;
                }
                Err(_) => {
                    return Err(ApiError::Network {
                        message: "the websocket did not open within 30s".to_string(),
                        what: what.to_string(),
                        attempts: attempt,
                        maybe_sent: true,
                    });
                }
            }
        }
    }
}

/// Exponential backoff with jitter: 0.5 s, 1 s, 2 s, 4 s (capped at 8 s),
/// or the server's `Retry-After` when it is shorter than 30 s.
fn backoff(attempt: u32, retry_after: Option<Duration>) -> Duration {
    if let Some(wait) = retry_after.filter(|w| *w <= Duration::from_secs(30)) {
        return wait;
    }
    let base = Duration::from_millis(500) * 2u32.saturating_pow(attempt.saturating_sub(1));
    let jitter = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos() % 250))
        .unwrap_or(0);
    base.min(Duration::from_secs(8)) + Duration::from_millis(jitter)
}

fn retry_after(response: &Response) -> Option<Duration> {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// A transport error with its causes, which is where reqwest keeps the
/// useful part ("connection refused", "certificate...").
fn error_chain(err: &dyn std::error::Error) -> String {
    let mut text = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        let cause_text = cause.to_string();
        if !text.contains(&cause_text) {
            text.push_str(": ");
            text.push_str(&cause_text);
        }
        source = cause.source();
    }
    text
}

/// The human part of an error body: the JSON `message` (or `error`), an
/// HTML page's title, or the text itself, at most 500 characters.
pub(crate) fn error_message(body: &str) -> String {
    let body = body.trim();
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
        let pick = |key: &str| match json.get(key) {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Array(items)) => Some(
                items
                    .iter()
                    .map(|i| {
                        i.as_str()
                            .map(str::to_string)
                            .unwrap_or_else(|| i.to_string())
                    })
                    .collect::<Vec<_>>()
                    .join("; "),
            ),
            _ => None,
        };
        if let Some(message) = pick("message").or_else(|| pick("error")) {
            return shorten(&message);
        }
    }
    let lower = body.to_lowercase();
    if let Some(start) = lower.find("<title>")
        && let Some(end) = lower[start..].find("</title>")
    {
        return shorten(body[start + 7..start + end].trim());
    }
    if body.is_empty() {
        "no details".to_string()
    } else {
        shorten(body)
    }
}

fn shorten(text: &str) -> String {
    let one_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > 500 {
        format!("{}…", one_line.chars().take(500).collect::<String>())
    } else {
        one_line
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_come_from_json_html_or_text() {
        assert_eq!(
            error_message(r#"{"statusCode":400,"message":"bad request: no IP","code":"X"}"#),
            "bad request: no IP"
        );
        assert_eq!(
            error_message(r#"{"message":["cpu must be a number","disk too big"]}"#),
            "cpu must be a number; disk too big"
        );
        assert_eq!(
            error_message(
                "<html><head><title>502 Bad Gateway</title></head><body>..</body></html>"
            ),
            "502 Bad Gateway"
        );
        assert_eq!(error_message("  plain\n text "), "plain text");
        assert_eq!(error_message(""), "no details");
    }

    #[test]
    fn statuses_map_to_owned_errors() {
        let status = |status: u16, message: &str| ApiError::Status {
            status,
            message: message.to_string(),
            what: "creating a sandbox".to_string(),
            attempts: 3,
        };
        let SandboxError::Unavailable { fix, .. } = SandboxError::from(status(401, "Unauthorized"))
        else {
            panic!("401 is a setup problem")
        };
        assert!(fix.contains(KEY_URL));
        assert!(fix.contains("export DAYTONA_API_KEY"));
        let SandboxError::Unavailable { fix, .. } =
            SandboxError::from(status(403, "Total CPU quota exceeded"))
        else {
            panic!("403 is a setup problem")
        };
        assert!(fix.contains("resource limit"), "{fix}");
        let err = SandboxError::from(status(503, "Service Unavailable"));
        assert!(err.is_transient());
        assert!(err.to_string().contains("after 3 attempts"), "{err}");
        let err = SandboxError::from(status(429, "slow down"));
        assert!(err.is_transient());
        assert!(matches!(
            SandboxError::from(status(400, "bad")),
            SandboxError::Unsupported(_)
        ));
        assert!(matches!(
            SandboxError::from(ApiError::NoKey),
            SandboxError::Unavailable { .. }
        ));
    }

    #[test]
    fn backoff_grows_and_honors_retry_after() {
        assert!(backoff(1, None) < Duration::from_secs(1));
        assert!(backoff(4, None) >= Duration::from_secs(4));
        assert!(backoff(10, None) <= Duration::from_millis(8250));
        assert_eq!(
            backoff(1, Some(Duration::from_secs(3))),
            Duration::from_secs(3)
        );
        assert!(backoff(1, Some(Duration::from_secs(300))) < Duration::from_secs(1));
    }

    #[test]
    fn the_key_never_prints() {
        let api = Api::new(
            "https://example.invalid/api/",
            Some("dtn_secret_123".into()),
            None,
        )
        .unwrap();
        let debug = format!("{api:?}");
        assert!(!debug.contains("dtn_secret_123"), "{debug}");
        assert_eq!(api.redact("key dtn_secret_123 here"), "key [redacted] here");
        assert_eq!(api.url("/sandbox"), "https://example.invalid/api/sandbox");
        assert_eq!(format!("{:?}", Secret::new("x".into())), "[redacted]");
    }

    #[tokio::test]
    async fn no_key_fails_before_any_request() {
        let api = Api::new("https://example.invalid/api", None, None).unwrap();
        let err = api
            .send("listing sandboxes", Retry::Safe, |http| {
                Ok(http.get("https://example.invalid/"))
            })
            .await
            .unwrap_err();
        assert!(matches!(err, ApiError::NoKey));
    }
}
