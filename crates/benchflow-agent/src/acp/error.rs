//! Why an ACP exchange failed, whose problem it is, and what to do next.
//!
//! Display strings keep 0.8's wording where 0.8's classifier reads it (`ACP
//! error -32603: ...`, `Agent idle for ...`, `usage limit reached ...`), so a
//! `result.json` written from them is categorized as 0.8 categorized it.
//! [`AcpError::category`] gives that category directly.

use std::fmt;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;

use crate::headless::classify::{Signal, classify, classify_error};

/// Whose problem a failure is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Owner {
    /// The agent (its adapter or CLI) misbehaved, stalled, or ran out of time.
    Agent,
    /// The model provider failed or refused (rate limit, outage, bad request).
    ModelProvider,
    /// The user's login: rejected, or out of subscription usage.
    Login,
    /// The user asked for something the agent cannot do (a model it does not
    /// offer, an effort setting it lacks).
    User,
    /// The sandbox or the connection to it.
    Sandbox,
}

/// The idle watchdog's evidence when it stops a prompt (0.8's
/// `IdleTimeoutDiagnostic`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IdleReport {
    pub idle_timeout_sec: u64,
    pub idle_duration_sec: u64,
    pub wall_clock_elapsed_sec: u64,
    pub n_tool_calls: usize,
    pub n_message_chunks: usize,
    pub n_thought_chunks: usize,
    pub last_activity_at: DateTime<Utc>,
    pub pending_tool_call_ids: Vec<String>,
    pub expired_pending_tool_call_ids: Vec<String>,
    pub pending_grace_sec: u64,
    pub pending_set_last_changed_at: Option<DateTime<Utc>>,
    pub last_tool_update_at: Option<DateTime<Utc>>,
    pub n_tool_call_updates: u64,
    pub n_pending_tool_call_updates: u64,
    pub n_expired_pending_tool_call_updates: u64,
    /// Pending calls that got no update after later calls started: their
    /// terminal update was probably lost rather than still coming.
    pub lost_update_tool_call_ids: Vec<String>,
}

/// A failed ACP exchange.
#[derive(Debug)]
pub enum AcpError {
    /// The agent did not answer `initialize` or `session/new` in time.
    HandshakeTimeout {
        phase: &'static str,
        after: Duration,
    },
    /// No message, tool call or thought for too long while a prompt ran.
    Idle {
        message: String,
        report: Box<IdleReport>,
    },
    /// The prompt ran past its wall-clock budget. The trajectory ends with
    /// 0.8's `agent_timeout` marker, which says whether a tool call was
    /// still open (`terminal_trajectory_complete`).
    Timeout {
        after: Duration,
        /// The agent ended its turn within a few seconds of `session/cancel`.
        cancelled: bool,
    },
    /// The agent's process ended (or closed its stdout) while BenchFlow
    /// waited for an answer.
    Exited {
        code: Option<i32>,
        /// The last of what it wrote on stderr, secrets redacted.
        stderr_tail: String,
    },
    /// The login has used up its subscription.
    UsageLimit {
        /// When the window resets, when the agent said.
        reset_at: Option<DateTime<Utc>>,
        /// `5-hour`, `7-day`, ...
        window: Option<String>,
        /// The agent's own words.
        message: String,
        /// Which login ran out, as a label.
        login: Option<String>,
    },
    /// The agent answered with a JSON-RPC error.
    Agent {
        code: i64,
        message: String,
        /// The error's `data`, as sent (boxed: a JSON value is large).
        data: Option<Box<Value>>,
        /// The request that failed (`session/prompt`).
        method: String,
    },
    /// The agent's session does not offer the requested model.
    ModelNotOffered {
        agent: String,
        model: String,
        offered: Vec<String>,
    },
    /// The agent lacks something the run asked for.
    Unsupported { message: String },
    /// The agent broke the protocol (a malformed answer).
    Protocol { message: String },
    /// Writing to the agent failed.
    Io {
        context: String,
        source: std::io::Error,
    },
}

impl fmt::Display for AcpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::HandshakeTimeout { phase, after } => write!(
                f,
                "ACP {phase} timed out after {}s before the first prompt",
                after.as_secs_f64()
            ),
            Self::Idle { message, .. } => f.write_str(message),
            Self::Timeout { after, .. } => {
                write!(
                    f,
                    "Agent prompt exceeded wall-clock budget {}s",
                    after.as_secs()
                )
            }
            Self::Exited { code, stderr_tail } => {
                write!(f, "Agent process closed stdout")?;
                if let Some(code) = code {
                    write!(f, " (exit code {code})")?;
                }
                if !stderr_tail.trim().is_empty() {
                    write!(f, "; its stderr ends with: {}", stderr_tail.trim())?;
                }
                Ok(())
            }
            Self::UsageLimit {
                reset_at,
                window,
                message,
                login,
            } => {
                write!(f, "usage limit reached")?;
                if let Some(login) = login {
                    write!(f, " on login {login}")?;
                }
                let mut facts = Vec::new();
                if let Some(window) = window {
                    facts.push(format!("{window} window"));
                }
                if let Some(reset) = reset_at {
                    facts.push(format!("resets {}", reset.format("%Y-%m-%d %H:%M UTC")));
                }
                if !facts.is_empty() {
                    write!(f, ": {}", facts.join(", "))?;
                }
                write!(f, " ({message})")
            }
            Self::Agent { code, message, .. } => write!(f, "ACP error {code}: {message}"),
            Self::ModelNotOffered {
                agent,
                model,
                offered,
            } => write!(
                f,
                "agent integration failure [agent_model]: {agent} does not offer model '{model}' \
                 for this login; it offers: {}",
                offered.join(", ")
            ),
            Self::Unsupported { message } | Self::Protocol { message } => f.write_str(message),
            Self::Io { context, source } => write!(f, "{context}: {source}"),
        }
    }
}

impl std::error::Error for AcpError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

// 0.8's markers (`_utils/scoring.py`), matched in the lowercased error.
const PROVIDER_AUTH_MARKERS: [&str; 11] = [
    "permission_denied",
    "leaked",
    "failed to authenticate",
    "invalid bearer token",
    "invalid api key",
    "was rejected as invalid",
    "unauthorized",
    "provider auth failed",
    "http 401",
    "http 403",
    // MiniMax Code's custom providers (`BYOK provider ... upstream error: 401`).
    "upstream error: 401",
];
const PROVIDER_RATE_LIMIT_MARKERS: [&str; 4] = [
    "provider rate limited",
    "rate limit",
    "too many requests",
    "http 429",
];
const PROVIDER_UNAVAILABLE_MARKERS: [&str; 2] = ["provider unavailable", "http 503"];
const PROVIDER_REJECTED_MARKERS: [&str; 2] = ["provider rejected request", "http 400"];

/// What a JSON-RPC error from the agent says about the model provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProviderSignal {
    Auth,
    RateLimit,
    Unavailable,
    Rejected,
    /// The provider does not serve the model (retired: `[410] Gone`).
    Model,
}

/// The provider failure a piece of the agent's text shows: whose problem it
/// is and the line that says so.
fn provider_failure_in(text: &str) -> Option<(Owner, String)> {
    let (signal, line) = classify(text, Utc::now())?;
    let owner = match signal {
        Signal::Login => Owner::Login,
        // A spent subscription needs its reset time, which a log line
        // seldom has; leave it to the agent's own report.
        Signal::UsageLimit(_) => return None,
        Signal::Model => Owner::User,
        // A prompt over the model's context window is the agent's to
        // manage (0.8: `acp_error`), not a provider failure to retry.
        Signal::Overflow => return None,
        Signal::RateLimit | Signal::Unavailable | Signal::Rejected => Owner::ModelProvider,
    };
    Some((owner, line))
}

/// Longest reply read as a provider's error: a real answer that mentions a
/// rate limit is longer.
const REPLY_ERROR_MAX: usize = 600;

/// A turn that made no tool call and whose whole reply is a provider's
/// error (Hermes returns a 401 as an ordinary message and ends its turn):
/// whose problem it is and the line that says so. Pass the reply only when
/// the turn made no tool call.
pub fn provider_failure_in_reply(reply: &str) -> Option<(Owner, String)> {
    let reply = reply.trim();
    if reply.is_empty() || reply.chars().count() > REPLY_ERROR_MAX || !opens_like_an_error(reply) {
        return None;
    }
    provider_failure_in(reply)
}

/// Whether a provider error is a refused request (`HTTP 400`, `provider
/// rejected request`): sending it again gets the same answer, so it is
/// not retried (0.8's `provider_rejected` is permanent).
pub fn refused_request(line: &str) -> bool {
    matches!(classify(line, Utc::now()), Some((Signal::Rejected, _)))
}

/// Whether a provider error says the stream broke after the model began
/// answering ("API Error: Server error mid-response", "stream closed
/// before completion"): the turn may then count the partial answer's
/// output tokens, and is still the provider's failure.
pub fn broke_mid_stream(line: &str) -> bool {
    MID_STREAM.is_match(line)
}

/// Whether `text` ends with a broken stream's words ([`broke_mid_stream`]),
/// with nothing after them but space or a sentence's closing punctuation:
/// how a provider's error ends a reply. A quote ends in its closing mark
/// (`` ` ``, `"`, `)`), and an answer that goes on has words after it.
pub fn ends_mid_stream(text: &str) -> bool {
    MID_STREAM.find_iter(text).last().is_some_and(|m| {
        text[m.end()..]
            .chars()
            .all(|c| c.is_whitespace() || matches!(c, '.' | '!' | '?' | ',' | ';' | ':' | '…'))
    })
}

static MID_STREAM: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(
        r"(?i)mid-response|stream closed before completion|Connection to the API was lost",
    )
    .expect("a valid pattern")
});

/// Whether a reply opens the way a provider's error does ("API Error: 529
/// Overloaded", "Error: ...", "HTTP 401: ...", "[503] ...", "429 Too Many
/// Requests"), so a short answer that only mentions a rate limit or a
/// refused connection is graded as the answer it is.
fn opens_like_an_error(reply: &str) -> bool {
    // A 4xx or 5xx status code, then the end, a colon, a bracket, or a
    // capitalized word (not "500 requests a minute").
    let status = |s: &str| {
        let b = s.as_bytes();
        b.len() >= 3
            && matches!(b[0], b'4' | b'5')
            && b[1].is_ascii_digit()
            && b[2].is_ascii_digit()
            && match b.get(3) {
                None | Some(b':') | Some(b']') => true,
                Some(b' ') => b.get(4).is_none_or(|c| c.is_ascii_uppercase()),
                _ => false,
            }
    };
    let lower = reply.to_ascii_lowercase();
    // A provider's error body passed on as it came: an event of a stream
    // (codex-acp's reply when MiMo refused a request) or the JSON itself.
    let body = lower.strip_prefix("data:").unwrap_or(&lower).trim_start();
    if body.starts_with("{\"error\"") {
        return true;
    }
    lower.starts_with("api error")
        || lower.starts_with("error:")
        || (lower.starts_with("http ") && status(&reply[5..]))
        || reply.strip_prefix('[').is_some_and(status)
        || status(reply)
}

/// ACP's `auth_required` error code.
const AUTH_REQUIRED: i64 = -32000;

impl AcpError {
    /// The agent's ACP server refused to start a session until it is signed
    /// in with its vendor's own account (Cline's and Kimi's `Authentication
    /// required`), which a key BenchFlow passes cannot satisfy.
    pub fn needs_sign_in(&self) -> bool {
        match self {
            Self::Agent { code, message, .. } => {
                *code == AUTH_REQUIRED && message.to_lowercase().contains("authentication required")
            }
            _ => false,
        }
    }

    /// When the agent's error names no cause (OpenHands' bare `Internal
    /// error`), the provider failure its own log shows, if any: whose problem
    /// it is and the log line that says so.
    pub fn provider_failure_in_log(&self, log: &str) -> Option<(Owner, String)> {
        if !matches!(self, Self::Agent { .. })
            || self.needs_sign_in()
            || self.provider_signal().is_some()
            || self.names_its_own_cause()
        {
            return None;
        }
        provider_failure_in(log)
    }

    /// The error's own text says why (a prompt over the context window, a
    /// spent subscription), so its log is not read for another cause.
    fn names_its_own_cause(&self) -> bool {
        matches!(
            self.own_signal(),
            Some(Signal::Overflow | Signal::UsageLimit(_))
        )
    }

    /// What the error's own text (its message and `data.details`) says.
    fn own_signal(&self) -> Option<Signal> {
        let Self::Agent { message, data, .. } = self else {
            return None;
        };
        let details = match data.as_deref() {
            Some(Value::Object(map)) => map
                .get("details")
                .map(|d| d.to_string())
                .unwrap_or_default(),
            _ => String::new(),
        };
        classify_error(&format!("{message}\n{details}"), Utc::now()).map(|(s, _)| s)
    }

    /// The model-provider failure an agent error carries: 0.8's markers, read
    /// in the message and (beyond 0.8) in `data.details`, where the ACP
    /// TypeScript SDK puts the thrown error's text.
    fn provider_signal(&self) -> Option<ProviderSignal> {
        let Self::Agent { message, data, .. } = self else {
            return None;
        };
        let details = match data.as_deref() {
            Some(Value::Object(map)) => map
                .get("details")
                .map(|d| d.to_string())
                .unwrap_or_default(),
            _ => String::new(),
        };
        let text = format!("{message} {details}").to_lowercase();
        let any = |markers: &[&str]| markers.iter().any(|m| text.contains(m));
        let signal = classify_error(&format!("{message}\n{details}"), Utc::now()).map(|(s, _)| s);
        // A prompt over the context window is the agent's own (0.8's
        // canonical `provider_rejected` case, its fault "agent"), even when
        // the message also says `HTTP 400`; as on the reply and log paths.
        if signal == Some(Signal::Overflow) {
            return None;
        }
        if any(&PROVIDER_AUTH_MARKERS) {
            Some(ProviderSignal::Auth)
        } else if any(&PROVIDER_RATE_LIMIT_MARKERS) {
            Some(ProviderSignal::RateLimit)
        } else if any(&PROVIDER_UNAVAILABLE_MARKERS) {
            Some(ProviderSignal::Unavailable)
        } else if any(&PROVIDER_REJECTED_MARKERS) {
            Some(ProviderSignal::Rejected)
        } else {
            // Beyond 0.8's markers: the provider errors the headless agents
            // are read for (Claude's `API Error: 500 ...`, `rate_limit_error`,
            // `Request timed out`), so a provider's failure is not the agent's.
            match signal? {
                Signal::Login => Some(ProviderSignal::Auth),
                Signal::RateLimit => Some(ProviderSignal::RateLimit),
                Signal::Unavailable => Some(ProviderSignal::Unavailable),
                Signal::Rejected => Some(ProviderSignal::Rejected),
                // A prompt over the context window is the agent's own (0.8:
                // `acp_error`); 0.8's `http 400` marker is read above.
                Signal::Overflow => None,
                Signal::Model => Some(ProviderSignal::Model),
                Signal::UsageLimit(_) => None,
            }
        }
    }

    /// 0.8's `error_category` for `result.json`: `usage_limit`,
    /// `provider_auth`, `provider_rate_limit`, `provider_rejected`,
    /// `infra_failure`, `acp_error`, `idle_timeout`, `timeout`,
    /// `pipe_closed`, `agent_integration` or `other`.
    pub fn category(&self) -> &'static str {
        match self {
            Self::UsageLimit { .. } => "usage_limit",
            Self::Idle { .. } => "idle_timeout",
            Self::Timeout { .. } | Self::HandshakeTimeout { .. } => "timeout",
            Self::Exited { .. } | Self::Io { .. } => "pipe_closed",
            Self::ModelNotOffered { .. } => "agent_integration",
            Self::Unsupported { .. } => "other",
            Self::Protocol { .. } => "acp_error",
            Self::Agent { .. } if self.needs_sign_in() => "provider_auth",
            // A prompt over the context window: 0.8's category by 0.8's
            // markers (`provider_rejected` when the text says `http 400` or
            // `provider rejected request`, else `acp_error`), and 0.8's
            // fault for both: the agent's (see `owner`), not retried.
            Self::Agent { message, .. } if matches!(self.own_signal(), Some(Signal::Overflow)) => {
                let text = message.to_lowercase();
                if PROVIDER_REJECTED_MARKERS.iter().any(|m| text.contains(m)) {
                    "provider_rejected"
                } else {
                    "acp_error"
                }
            }
            Self::Agent { .. } => match self.provider_signal() {
                Some(ProviderSignal::Auth) => "provider_auth",
                Some(ProviderSignal::Model) => "agent_integration",
                Some(ProviderSignal::RateLimit) => "provider_rate_limit",
                Some(ProviderSignal::Unavailable) => "infra_failure",
                Some(ProviderSignal::Rejected) => "provider_rejected",
                None => "acp_error",
            },
        }
    }

    /// Whose problem this is.
    pub fn owner(&self) -> Owner {
        match self {
            Self::UsageLimit { .. } => Owner::Login,
            Self::ModelNotOffered { .. } | Self::Unsupported { .. } => Owner::User,
            Self::Io { .. } => Owner::Sandbox,
            Self::Agent { .. } if self.needs_sign_in() => Owner::User,
            Self::Agent { .. } => match self.provider_signal() {
                Some(ProviderSignal::Auth) => Owner::Login,
                Some(ProviderSignal::Model) => Owner::User,
                Some(_) => Owner::ModelProvider,
                None => Owner::Agent,
            },
            _ => Owner::Agent,
        }
    }

    /// The next step for the user.
    pub fn next_step(&self) -> String {
        match self {
            Self::UsageLimit { .. } => {
                "switch to another login or wait for the reset; `bench doctor` shows each login".into()
            }
            Self::HandshakeTimeout { .. } => {
                "the agent started but did not answer in time; read its log in the trial's agent/ folder"
                    .into()
            }
            Self::Idle { .. } => "read the agent log in the trial's agent/ folder; if the task runs long \
                 silent commands, raise --idle-timeout"
                .into(),
            Self::Timeout { .. } => "raise the agent timeout, or make the task smaller".into(),
            Self::Exited { .. } => "its stderr says why".into(),
            Self::ModelNotOffered { offered, .. } => {
                format!("pick one of the models it offers: {}", offered.join(", "))
            }
            Self::Unsupported { .. } => "drop the setting, or pick an agent that supports it".into(),
            Self::Io { .. } => "the connection to the agent broke; check the sandbox and retry".into(),
            Self::Protocol { .. } => "the agent broke the ACP protocol; check its version pin".into(),
            Self::Agent { .. } if self.needs_sign_in() => {
                "this agent's ACP server takes only its vendor's own sign-in, which BenchFlow \
                 cannot pass into a sandbox; run the agent headless, or pick another agent"
                    .into()
            }
            Self::Agent { .. } => match self.provider_signal() {
                Some(ProviderSignal::Model) => {
                    "the provider no longer serves this model; pick another with -m".into()
                }
                Some(ProviderSignal::Auth) => {
                    "the model provider rejected the login; `bench doctor` shows which credential the agent used"
                        .into()
                }
                Some(ProviderSignal::RateLimit) => "the provider is rate limiting; retry later or lower --concurrency".into(),
                Some(ProviderSignal::Unavailable) => "the provider is unavailable; retry later".into(),
                Some(ProviderSignal::Rejected) => {
                    "the provider refused the request; see the agent log".into()
                }
                None => "read the agent log in the trial's agent/ folder".into(),
            },
        }
    }

    /// True when running the same trial again may succeed. Usage limits,
    /// rejected logins and requests, and unsupported settings never are.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::UsageLimit { .. } | Self::ModelNotOffered { .. } | Self::Unsupported { .. } => {
                false
            }
            Self::Agent { .. } => {
                // A prompt over the context window overflows again.
                !self.needs_sign_in()
                    && !matches!(self.own_signal(), Some(Signal::Overflow))
                    && !matches!(
                        self.provider_signal(),
                        Some(
                            ProviderSignal::Auth
                                | ProviderSignal::RateLimit
                                | ProviderSignal::Rejected
                                | ProviderSignal::Model
                        )
                    )
            }
            _ => true,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn agent_error(message: &str, data: Option<Value>) -> AcpError {
        let data = data.map(Box::new);
        AcpError::Agent {
            code: -32603,
            message: message.into(),
            data,
            method: "session/prompt".into(),
        }
    }

    #[test]
    fn a_prompt_over_the_context_window_is_the_agents() {
        let e = AcpError::Agent {
            code: -32603,
            message:
                "Internal error: API Error: prompt is too long: 210000 tokens > 200000 maximum"
                    .into(),
            data: None,
            method: "session/prompt".into(),
        };
        assert_eq!(e.category(), "acp_error");
        let opaque = AcpError::Agent {
            code: -32603,
            message: "Internal error".into(),
            data: None,
            method: "session/prompt".into(),
        };
        assert_eq!(
            opaque.provider_failure_in_log("litellm: context_length_exceeded\nTraceback ..."),
            None
        );
    }

    /// A request the provider refused for another reason (an HTTP 400 for
    /// an unsupported parameter) is the provider's on every path, as the
    /// JSON-RPC message path has it, and is never retried (0.8's
    /// `provider_rejected` is permanent); a prompt over the context window
    /// is the agent's on every path, the JSON-RPC message's `HTTP 400`
    /// notwithstanding. Guards the D7b fixes of 61deae52 (FIX-REVIEW-A 5,
    /// FIX-REVIEW-B 7) and of REVIEW-CORRECT M1.
    #[test]
    fn a_refused_request_that_is_not_the_context_window_is_the_providers() {
        let line = "HTTP 400: Unsupported parameter 'temperature'";
        let opaque = AcpError::Agent {
            code: -32603,
            message: "Internal error".into(),
            data: None,
            method: "session/prompt".into(),
        };
        assert_eq!(
            opaque.provider_failure_in_log(&format!("{line}\nTraceback ...")),
            Some((Owner::ModelProvider, line.to_string()))
        );
        assert_eq!(
            provider_failure_in_reply(line),
            Some((Owner::ModelProvider, line.to_string()))
        );
        let message = AcpError::Agent {
            code: -32603,
            message: format!("Internal error: {line}"),
            data: None,
            method: "session/prompt".into(),
        };
        assert_eq!(message.category(), "provider_rejected");
        assert_eq!(message.owner(), Owner::ModelProvider);
        assert!(!message.is_retryable());
        assert!(refused_request(line));
        assert!(!refused_request("API Error: 529 Overloaded"));
        let overflow = AcpError::Agent {
            code: -32603,
            message: "Internal error: HTTP 400: context_length_exceeded".into(),
            data: None,
            method: "session/prompt".into(),
        };
        // 0.8's category for it, with 0.8's fault: the agent's.
        assert_eq!(overflow.category(), "provider_rejected");
        assert_eq!(overflow.owner(), Owner::Agent);
        assert!(!overflow.is_retryable());
        // Its log is not read for another cause: a tool's "HTTP 400" there
        // would make it a provider failure (REVIEW-OUTCOMES L1).
        let named = AcpError::Agent {
            code: -32603,
            message:
                "Internal error: API Error: prompt is too long: 210000 tokens > 200000 maximum"
                    .into(),
            data: None,
            method: "session/prompt".into(),
        };
        assert_eq!(
            named.provider_failure_in_log("curl: Error: HTTP 400 Bad Request\n"),
            None
        );
        assert_eq!(
            provider_failure_in_reply("Error: prompt is too long: 210000 tokens > 200000 maximum"),
            None
        );
    }

    #[test]
    fn only_a_stream_that_broke_mid_response_says_so() {
        for line in [
            "API Error: Server error mid-response",
            "API Error: Connection closed mid-response",
            "Error: stream closed before completion",
            "API Error: Connection to the API was lost",
        ] {
            assert!(broke_mid_stream(line), "{line}");
        }
        for line in ["Error: rate limit exceeded", "API Error: 529 Overloaded"] {
            assert!(!broke_mid_stream(line), "{line}");
        }
        // Ending a reply: only space or a sentence's end may follow, not a
        // quote's closing mark or more words (REVIEW-B (d7d) F1).
        for line in [
            "API Error: Server error mid-response",
            "API Error: Server error mid-response. ",
            "Error: stream closed before completion!",
        ] {
            assert!(ends_mid_stream(line), "{line}");
        }
        for line in [
            "API Error: Server error mid-response`.",
            "API Error: Server error mid-response\".",
            "API Error: Server error mid-response, retry with backoff.",
            "API Error: 529 Overloaded",
        ] {
            assert!(!ends_mid_stream(line), "{line}");
        }
    }

    #[test]
    fn provider_failures_are_recognized_as_0_8_does() {
        let cases = [
            (
                "Internal error: API Error: 401 Invalid API key",
                "provider_auth",
                Owner::Login,
            ),
            (
                "Internal error: Failed to authenticate",
                "provider_auth",
                Owner::Login,
            ),
            // Seen live: mcode acp with a placeholder MiniMax key.
            (
                "Internal error: MiniMax Code Runtime failed: BYOK provider custom_provider:benchflow-minimax upstream error: 401 login fail: Please carry the API secret key in the 'X-Api-Key' field of the request header",
                "provider_auth",
                Owner::Login,
            ),
            (
                "API Error: Request rejected (429) · rate limit exceeded",
                "provider_rate_limit",
                Owner::ModelProvider,
            ),
            (
                "Internal error: provider unavailable (HTTP 503)",
                "infra_failure",
                Owner::ModelProvider,
            ),
            (
                // 0.8's category for an overflow, and 0.8's fault for it:
                // the agent's (not the provider's, as this test had it).
                "provider rejected request (HTTP 400): prompt is too long",
                "provider_rejected",
                Owner::Agent,
            ),
            ("Internal error", "acp_error", Owner::Agent),
            // Claude Code's own words, which 0.8's markers miss.
            (
                "API Error: 500 {\"type\":\"error\",\"error\":{\"type\":\"api_error\",\"message\":\"Internal server error\"}}",
                "infra_failure",
                Owner::ModelProvider,
            ),
            (
                "API Error: 529 Overloaded",
                "infra_failure",
                Owner::ModelProvider,
            ),
            (
                "API Error: Request timed out.",
                "infra_failure",
                Owner::ModelProvider,
            ),
            (
                "API Error: 429 {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\"}}",
                "provider_rate_limit",
                Owner::ModelProvider,
            ),
            (
                "API Error: 401 {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"}}",
                "provider_auth",
                Owner::Login,
            ),
            (
                "Internal error: tool registry is empty",
                "acp_error",
                Owner::Agent,
            ),
        ];
        for (message, category, owner) in cases {
            let err = agent_error(message, None);
            assert_eq!(err.category(), category, "{message}");
            assert_eq!(err.owner(), owner, "{message}");
        }
        // The TypeScript SDK puts a thrown error's text in data.details.
        let err = agent_error(
            "Internal error",
            Some(json!({"details": "401 Unauthorized"})),
        );
        assert_eq!(err.category(), "provider_auth");
        assert!(!err.is_retryable());
        assert_eq!(err.to_string(), "ACP error -32603: Internal error");
    }

    #[test]
    fn todays_login_and_model_failures_are_not_the_agents() {
        // Seen live: cline-acp and kimi-acp take only their vendor's sign-in.
        for message in [
            "Authentication required: Call authenticate before starting a session",
            "Authentication required",
        ] {
            let err = AcpError::Agent {
                code: -32000,
                message: message.into(),
                data: None,
                method: "session/new".into(),
            };
            assert!(err.needs_sign_in(), "{message}");
            assert_eq!(err.owner(), Owner::User);
            assert!(!err.is_retryable());
            assert!(err.next_step().contains("vendor's own sign-in"));
        }
        // A retired model.
        let err = agent_error("Internal error: Exception: [410] Gone", None);
        assert_eq!(
            (err.category(), err.owner()),
            ("agent_integration", Owner::User)
        );
        assert!(!err.is_retryable());
        // OpenHands: a bare -32603 whose log names the cause (today's
        // stderr, trimmed; the rich panel wraps the provider's own words).
        let log = "                             litellm.AuthenticationError:                       \n\
                   \x20                            GeminiException - {                                \n\
                   \x20                              \"message\": \"API key not                     \n\
                   \x20                            valid. Please pass a valid API                     \n\
                   \x20Conversation logs are stored at:\n";
        let err = agent_error("Internal error", None);
        let (owner, line) = err.provider_failure_in_log(log).unwrap();
        assert_eq!(owner, Owner::Login);
        assert_eq!(line, "litellm.AuthenticationError:");
        assert_eq!(err.provider_failure_in_log("all quiet\n"), None);
        // An error that says why itself is read from its own words.
        let err = agent_error("API Error: 429 rate limit exceeded", None);
        assert_eq!(err.provider_failure_in_log(log), None);
    }

    #[test]
    fn a_providers_error_event_as_the_only_reply_is_the_providers() {
        // codex-acp on MiMo's Responses API (2026-10-12): the trial was
        // scored 0 as if the agent had answered.
        let reply = "data:{\"error\":{\"code\":\"responses_feature_not_supported\",\"message\":\"tool type 'web_search' is not supported by this gateway phase.\",\"param\":\"\",\"type\":\"unsupported_feature\"}}\n\n\n\n";
        let (owner, line) = provider_failure_in_reply(reply).unwrap();
        assert_eq!(owner, Owner::ModelProvider);
        assert!(line.contains("web_search"), "{line}");
        let body = "{\"error\":{\"message\":\"HTTP 400: bad request\"}}";
        assert!(provider_failure_in_reply(body).is_some());
        // An answer that quotes JSON is graded as the answer it is.
        assert_eq!(
            provider_failure_in_reply("I wrote {\"error\": null} to /app/out.json."),
            None
        );
    }

    #[test]
    fn a_provider_error_as_the_only_reply_is_the_providers() {
        // hermes-acp with a dead MiniMax key (routes evidence, today).
        let reply = "HTTP 401: login fail: Please carry the API secret key in the 'X-Api-Key' field of the request header";
        let (owner, line) = provider_failure_in_reply(reply).unwrap();
        assert_eq!((owner, line.as_str()), (Owner::Login, reply));
        assert_eq!(
            provider_failure_in_reply("API Error: 529 Overloaded").map(|f| f.0),
            Some(Owner::ModelProvider)
        );
        // An answer is not an error, even one that mentions a rate limit.
        assert_eq!(
            provider_failure_in_reply("Done: I created /app/hello.txt."),
            None
        );
        let long = format!(
            "{} The API's rate limit is 5 requests a minute.",
            "Here is the plan. ".repeat(40)
        );
        assert_eq!(provider_failure_in_reply(&long), None);
        assert_eq!(provider_failure_in_reply("  "), None);
        // Short answers that only mention a provider's words are answers.
        for answer in [
            "The API's rate limit is 100 requests a minute.",
            "500 requests a minute is the rate limit.",
            "Connection refused was the bug: the service listened on 8081. Fixed.",
            "It returns 429 Too Many Requests when you exceed the rate limit.",
        ] {
            assert_eq!(provider_failure_in_reply(answer), None, "{answer}");
        }
        assert!(provider_failure_in_reply("429 Too Many Requests: rate limit exceeded").is_some());
        assert!(provider_failure_in_reply("Error: rate limit exceeded, retry in 20s").is_some());
    }

    #[test]
    fn usage_limits_read_like_0_8() {
        let err = AcpError::UsageLimit {
            reset_at: Some(
                chrono::TimeZone::with_ymd_and_hms(&Utc, 2026, 10, 2, 16, 0, 0).unwrap(),
            ),
            window: Some("7-day".into()),
            message: "You've hit your weekly limit · resets Oct 2, 4pm (UTC)".into(),
            login: Some("CLAUDE_CODE_OAUTH_TOKEN (environment)".into()),
        };
        assert_eq!(
            err.to_string(),
            "usage limit reached on login CLAUDE_CODE_OAUTH_TOKEN (environment): 7-day window, \
             resets 2026-10-02 16:00 UTC (You've hit your weekly limit · resets Oct 2, 4pm (UTC))"
        );
        assert_eq!(err.category(), "usage_limit");
        assert_eq!(err.owner(), Owner::Login);
        assert!(!err.is_retryable());
    }

    #[test]
    fn process_and_time_failures() {
        let exited = AcpError::Exited {
            code: Some(137),
            stderr_tail: "Killed\n".into(),
        };
        assert_eq!(
            exited.to_string(),
            "Agent process closed stdout (exit code 137); its stderr ends with: Killed"
        );
        assert_eq!(exited.category(), "pipe_closed");
        let timeout = AcpError::Timeout {
            after: Duration::from_secs(600),
            cancelled: true,
        };
        assert_eq!(
            timeout.to_string(),
            "Agent prompt exceeded wall-clock budget 600s"
        );
        assert_eq!(timeout.category(), "timeout");
        let handshake = AcpError::HandshakeTimeout {
            phase: "initialize",
            after: Duration::from_secs(60),
        };
        assert_eq!(
            handshake.to_string(),
            "ACP initialize timed out after 60s before the first prompt"
        );
    }
}
