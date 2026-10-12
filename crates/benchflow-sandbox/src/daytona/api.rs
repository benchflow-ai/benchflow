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
            // Daytona's rate limit: its own answer (convergence review
            // E9-C-A N2), kept apart from a 5xx, which a sandbox broken from
            // inside can surface through the toolbox's proxy.
            ApiError::Status {
                status: 429,
                message,
                what,
                attempts,
            } => SandboxError::Throttled(format!(
                "Daytona refused {what} for its rate limit (HTTP 429: {message}){}. This is on \
                 Daytona's side and usually passes; run again, with lower concurrency if it keeps \
                 happening.",
                tries(attempts)
            )),
            ApiError::Status {
                status,
                message,
                what,
                attempts,
            } if status >= 500 => SandboxError::Transport(format!(
                "Daytona failed while {what} (HTTP {status}: {message}){}. This is on Daytona's \
                 side and usually passes; run again, with lower concurrency if it keeps happening.",
                tries(attempts)
            )),
            // The sandbox is gone (deleted, or reclaimed by Daytona) or
            // not in a state to answer: the backend lost it, which no
            // setting of the run's could fix.
            ApiError::Status {
                status: status @ (404 | 409 | 410),
                message,
                what,
                ..
            } => SandboxError::Transport(format!(
                "Daytona lost the sandbox while {what} (HTTP {status}: {message}). It was deleted \
                 or stopped outside this run; run again, and check the Daytona dashboard if it \
                 keeps happening."
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
    /// Creates this run sent whose sandbox got no answer and has not been
    /// found yet (see `remove_if_it_lands`).
    pub(crate) unanswered: std::sync::atomic::AtomicUsize,
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

/// How many times a request is tried at most (but see [`THROTTLE_PATIENCE`]).
const MAX_ATTEMPTS: u32 = 5;

/// How long a request Daytona keeps refusing with 429 is tried again.
/// Five tries spanned 7.5 s, shorter than a rate limit's window: a 20 s
/// burst of 429s (as a shared account at high concurrency sees) failed a
/// trial's image (E5).
const THROTTLE_PATIENCE: Duration = Duration::from_secs(60);

/// The shortest wait before a try again, so `Retry-After: 0` is no busy loop.
const MIN_WAIT: Duration = Duration::from_millis(250);

/// The Daytona key goes to `DAYTONA_API_URL` as a bearer header: it must be
/// https, or http to this machine (a local fake), wherever it was set.
fn check_api_url(base: &str) -> Result<(), SandboxError> {
    let refuse = |why: &str| {
        Err(SandboxError::Unsupported(format!(
            "DAYTONA_API_URL {why}; the Daytona key is sent there, so it must be https, or http \
             to this machine (localhost, 127.0.0.1, ::1)"
        )))
    };
    let Ok(url) = reqwest::Url::parse(base.trim()) else {
        return refuse("is not a URL");
    };
    let host = url
        .host_str()
        .unwrap_or_default()
        .trim_start_matches('[')
        .trim_end_matches(']');
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback());
    match url.scheme() {
        "https" => Ok(()),
        "http" if loopback => Ok(()),
        _ => refuse(&format!("is {}://{host}", url.scheme())),
    }
}

impl Api {
    pub(crate) fn new(
        base: &str,
        key: Option<String>,
        target: Option<String>,
    ) -> Result<Self, SandboxError> {
        // Websockets go through rustls without an explicit config; make sure
        // a crypto provider is installed even if several are compiled in.
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        check_api_url(base)?;
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
            unanswered: std::sync::atomic::AtomicUsize::new(0),
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
        let first = std::time::Instant::now();
        loop {
            attempt += 1;
            let request = build(&self.http)?.headers(headers.clone());
            let sent = std::time::Instant::now();
            match request.send().await {
                Ok(response) if response.status().is_success() => {
                    tracing::debug!(
                        "Daytona {what}: HTTP {} in {} ms",
                        response.status().as_u16(),
                        sent.elapsed().as_millis()
                    );
                    return Ok(response);
                }
                Ok(response) => {
                    let status = response.status();
                    let wait = retry_after(&response);
                    let body = response.text().await.unwrap_or_default();
                    let throttled = status == StatusCode::TOO_MANY_REQUESTS;
                    let again = throttled || (retry == Retry::Safe && status.is_server_error());
                    let more = attempt < MAX_ATTEMPTS
                        || (throttled && first.elapsed() < THROTTLE_PATIENCE);
                    if again && more {
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
                        // The URL may carry a signed query (a build log's).
                        let err = err.without_url();
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
        return wait.max(MIN_WAIT);
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
    // ASCII lowercasing keeps every byte offset, so they index `body` too
    // (Unicode lowercasing changes lengths: `İ` grows, `K` shrinks).
    let lower = body.to_ascii_lowercase();
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

    /// Guards the D5 fix: the title was found in `body.to_lowercase()` and
    /// cut from `body`, so text whose lowercase is longer or shorter (`İ`,
    /// the Kelvin sign) before the title cut inside a character and panicked.
    #[test]
    fn an_error_page_with_non_ascii_text_keeps_its_title() {
        assert_eq!(error_message("<p>İstanbul</p><title>日本</title>"), "日本");
        assert_eq!(
            error_message("<p>\u{212a}\u{212a}\u{212a}</p><TITLE>Bad Gateway</TITLE>"),
            "Bad Gateway"
        );
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
        assert!(matches!(err, SandboxError::Transport(_)), "{err:?}");
        assert!(err.to_string().contains("after 3 attempts"), "{err}");
        // A 429 is Daytona's own answer, which no code inside a sandbox can
        // produce; a 5xx may come through the toolbox's proxy from a
        // sandbox broken from inside (convergence review E9-C-A N2).
        let err = SandboxError::from(status(429, "slow down"));
        assert!(err.is_transient());
        assert!(matches!(err, SandboxError::Throttled(_)), "{err:?}");
        assert!(matches!(
            SandboxError::from(status(400, "bad")),
            SandboxError::Unsupported(_)
        ));
        // Seen live: a sandbox deleted while its verifier ran was R401,
        // "your setup's problem".
        let err = SandboxError::from(status(
            404,
            "not found: sandbox a6e91758-44db-44bc-8ea0-ffd3c61d1768 not found",
        ));
        assert!(err.is_transient(), "{err}");
        assert!(err.to_string().contains("lost the sandbox"), "{err}");
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

    /// E5 (429s from a fault proxy before Daytona's API): a burst of 429s
    /// longer than five quick tries failed the request; `Retry-After: 0`
    /// made the tries back to back.
    #[tokio::test]
    async fn a_throttled_request_waits_out_the_burst() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::<std::time::Instant>::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = [0u8; 4096];
                let _ = sock.read(&mut buf).await;
                let n = {
                    let mut l = log.lock().unwrap();
                    l.push(std::time::Instant::now());
                    l.len()
                };
                let answer = if n <= 8 {
                    "HTTP/1.1 429 Too Many Requests\r\nretry-after: 0\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}"
                } else {
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 2\r\nconnection: close\r\n\r\n{}"
                };
                let _ = sock.write_all(answer.as_bytes()).await;
            }
        });
        let api = Api::new(&format!("http://{addr}/api"), Some("k".into()), None).unwrap();
        let response = api
            .send("listing sandboxes", Retry::Unsent, |http| {
                Ok(http.get(format!("http://{addr}/api/sandbox")))
            })
            .await;
        assert!(response.is_ok(), "{:?}", response.err());
        let times = seen.lock().unwrap().clone();
        assert_eq!(times.len(), 9);
        for pair in times.windows(2) {
            assert!(pair[1] - pair[0] >= Duration::from_millis(200), "{times:?}");
        }
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

    /// A cloned project's `.env` (or any other source) set
    /// `DAYTONA_API_URL=http://attacker.example`: the key would go there in
    /// cleartext. Only https, or http to this machine, is taken.
    #[test]
    fn the_api_url_must_be_https_or_this_machine() {
        for bad in [
            "http://attacker.example/api",
            "http://user:secret@10.0.0.5/api",
            "ftp://x/api",
            "not a url",
        ] {
            let err = Api::new(bad, Some("k".into()), None)
                .unwrap_err()
                .to_string();
            assert!(err.contains("must be https"), "{bad}: {err}");
            assert!(!err.contains("secret"), "{err}");
        }
        for good in [
            "https://app.daytona.io/api",
            "http://127.0.0.1:3986/api",
            "http://localhost:3986/api",
            "http://[::1]:3986/api",
        ] {
            assert!(Api::new(good, Some("k".into()), None).is_ok(), "{good}");
        }
    }
}
