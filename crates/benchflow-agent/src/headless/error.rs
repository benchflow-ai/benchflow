//! Why a headless run failed, whose problem it is, and what to do next.

use std::fmt;
use std::time::Duration;

use chrono::{DateTime, Utc};

use crate::acp::Owner;

/// A failed headless run.
#[derive(Debug)]
pub enum HeadlessError {
    /// The run went past its wall-clock budget and was stopped.
    Timeout { after: Duration },
    /// Nothing on stdout for too long; the agent was stopped.
    Idle {
        after: Duration,
        /// Tool calls still open when it was stopped.
        pending_tool_calls: Vec<String>,
    },
    /// The login has used up its subscription or credits.
    UsageLimit {
        reset_at: Option<DateTime<Utc>>,
        /// `5-hour`, `7-day`, ...
        window: Option<String>,
        /// The agent's own words.
        message: String,
        /// Which login ran out, as a label.
        login: Option<String>,
    },
    /// The model provider rejected the login.
    LoginRejected { message: String },
    /// The login cannot use the requested model.
    ModelUnavailable { message: String },
    /// The agent lacks something only the user can give it: the code it
    /// runs (a LangGraph graph, a Strands factory, an Eve app) or an option.
    Setup {
        message: String,
        /// What to set, in the agent's words.
        fix: Option<String>,
    },
    /// The model provider limited, failed or refused the request.
    Provider {
        message: String,
        /// Running the trial again may succeed (a rate limit or an outage,
        /// not a refused request).
        retryable: bool,
        /// 0.8's category: `provider_rate_limit`, `infra_failure` or
        /// `provider_rejected`.
        category: &'static str,
    },
    /// The agent reported a failure BenchFlow does not recognize.
    AgentFailed { message: String, code: Option<i32> },
    /// The agent's process ended with a failure and said nothing BenchFlow
    /// could read about why.
    Exited {
        code: Option<i32>,
        /// The last of what it wrote on stderr, secrets redacted.
        stderr_tail: String,
        /// Whether its output said anything BenchFlow understood first.
        started: bool,
    },
    /// Reading the agent's output failed.
    Io {
        context: String,
        source: std::io::Error,
    },
}

impl fmt::Display for HeadlessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout { after } => write!(
                f,
                "Agent exceeded its wall-clock budget of {}s and was stopped",
                after.as_secs()
            ),
            Self::Idle {
                after,
                pending_tool_calls,
            } => {
                write!(
                    f,
                    "Agent idle for {}s (no output) and was stopped",
                    after.as_secs()
                )?;
                if !pending_tool_calls.is_empty() {
                    write!(
                        f,
                        "; tool calls still open: {}",
                        pending_tool_calls.join(", ")
                    )?;
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
            Self::LoginRejected { message } => {
                write!(f, "the model provider rejected the login: {message}")
            }
            Self::ModelUnavailable { message } => {
                write!(f, "the agent cannot use the requested model: {message}")
            }
            Self::Setup { message, .. } => write!(f, "the agent cannot start: {message}"),
            Self::Provider { message, .. } => write!(f, "the model provider failed: {message}"),
            Self::AgentFailed { message, code } => {
                write!(f, "the agent reported a failure")?;
                if let Some(code) = code {
                    write!(f, " and exited with code {code}")?;
                }
                write!(f, ": {message}")
            }
            Self::Exited {
                code, stderr_tail, ..
            } => {
                match code {
                    Some(code) => write!(f, "the agent exited with code {code}")?,
                    None => write!(f, "the agent's process ended")?,
                }
                if !stderr_tail.trim().is_empty() {
                    let last: Vec<&str> = stderr_tail.trim().lines().rev().take(3).collect();
                    let last: Vec<&str> = last.into_iter().rev().collect();
                    write!(f, "; its stderr ends with: {}", last.join(" | "))?;
                }
                Ok(())
            }
            Self::Io { context, source } => write!(f, "{context}: {source}"),
        }
    }
}

impl std::error::Error for HeadlessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}

impl HeadlessError {
    /// 0.8's `error_category` for `result.json`.
    pub fn category(&self) -> &'static str {
        match self {
            Self::Timeout { .. } => "timeout",
            Self::Idle { .. } => "idle_timeout",
            Self::UsageLimit { .. } => "usage_limit",
            Self::LoginRejected { .. } => "provider_auth",
            Self::ModelUnavailable { .. } | Self::Setup { .. } => "agent_integration",
            Self::Provider { category, .. } => category,
            Self::AgentFailed { .. } => "agent_error",
            Self::Exited { .. } | Self::Io { .. } => "pipe_closed",
        }
    }

    /// Whose problem this is.
    pub fn owner(&self) -> Owner {
        match self {
            Self::UsageLimit { .. } | Self::LoginRejected { .. } => Owner::Login,
            Self::ModelUnavailable { .. } | Self::Setup { .. } => Owner::User,
            Self::Provider { .. } => Owner::ModelProvider,
            Self::Io { .. } => Owner::Sandbox,
            _ => Owner::Agent,
        }
    }

    /// The next step for the user.
    pub fn next_step(&self, agent: &str) -> String {
        let log = format!("agent/{agent}.txt and agent/stderr.txt in the trial folder");
        match self {
            Self::Timeout { .. } => "raise the agent timeout, or make the task smaller".into(),
            Self::Idle { .. } => {
                format!("read {log}; if the task runs long silent commands, raise --idle-timeout")
            }
            Self::UsageLimit { .. } => {
                "switch to another login or wait for the reset; `bench doctor` shows each login"
                    .into()
            }
            Self::LoginRejected { .. } => format!(
                "log in again or set a valid key; `bench agents {agent}` shows which login the agent used"
            ),
            Self::ModelUnavailable { .. } => {
                "pick a model this login can use (`-m`), or use a login that has the model".into()
            }
            Self::Setup { fix: Some(fix), .. } => fix.clone(),
            Self::Setup { fix: None, .. } => format!(
                "give the agent what its message names (`bench agents {agent}` lists its options); see {log}"
            ),
            Self::Provider {
                retryable: true, ..
            } => "the provider is limiting or unavailable; retry later or lower concurrency".into(),
            Self::Provider { .. } => format!(
                "the provider refused the request (often a context window overflow); see {log}"
            ),
            Self::AgentFailed { .. } => format!("read {log}"),
            Self::Exited { started: false, .. } => format!(
                "the agent failed before doing anything; its stderr (above, and {log}) says why"
            ),
            Self::Exited { .. } => format!("the agent crashed; see {log}"),
            Self::Io { .. } => {
                "the connection to the sandbox broke while reading the agent; retry".into()
            }
        }
    }

    /// True when running the same trial again may succeed.
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::UsageLimit { .. }
            | Self::LoginRejected { .. }
            | Self::ModelUnavailable { .. }
            | Self::Setup { .. } => false,
            Self::Provider { retryable, .. } => *retryable,
            _ => true,
        }
    }
}
