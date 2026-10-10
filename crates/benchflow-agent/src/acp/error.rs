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
}

impl AcpError {
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
        if any(&PROVIDER_AUTH_MARKERS) {
            Some(ProviderSignal::Auth)
        } else if any(&PROVIDER_RATE_LIMIT_MARKERS) {
            Some(ProviderSignal::RateLimit)
        } else if any(&PROVIDER_UNAVAILABLE_MARKERS) {
            Some(ProviderSignal::Unavailable)
        } else if any(&PROVIDER_REJECTED_MARKERS) {
            Some(ProviderSignal::Rejected)
        } else {
            None
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
            Self::Agent { .. } => match self.provider_signal() {
                Some(ProviderSignal::Auth) => "provider_auth",
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
            Self::Agent { .. } => match self.provider_signal() {
                Some(ProviderSignal::Auth) => Owner::Login,
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
            Self::Agent { .. } => match self.provider_signal() {
                Some(ProviderSignal::Auth) => {
                    "the model provider rejected the login; `bench doctor` shows which credential the agent used"
                        .into()
                }
                Some(ProviderSignal::RateLimit) => "the provider is rate limiting; retry later or lower --concurrency".into(),
                Some(ProviderSignal::Unavailable) => "the provider is unavailable; retry later".into(),
                Some(ProviderSignal::Rejected) => {
                    "the provider refused the request (often a context window overflow); see the agent log".into()
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
            Self::Agent { .. } => !matches!(
                self.provider_signal(),
                Some(ProviderSignal::Auth | ProviderSignal::RateLimit | ProviderSignal::Rejected)
            ),
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
                "provider rejected request (HTTP 400): prompt is too long",
                "provider_rejected",
                Owner::ModelProvider,
            ),
            ("Internal error", "acp_error", Owner::Agent),
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
