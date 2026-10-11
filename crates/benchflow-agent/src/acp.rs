//! The Agent Client Protocol, client side (<https://agentclientprotocol.com>).
//!
//! ACP is JSON-RPC 2.0, one JSON object per line, over the agent process's
//! stdin and stdout. BenchFlow is the client: it starts the agent in the
//! sandbox, then
//!
//! 1. `initialize`s it (protocol version 1; no file system or terminal
//!    offered, since the agent has its own in the sandbox; ACP session
//!    notices taken; claude-agent-acp's typed failures asked for when the
//!    manifest says so);
//! 2. opens a session with `session/new` in the task's working folder, with
//!    the task's MCP servers;
//! 3. selects the model and reasoning effort the way the manifest says
//!    (a session config option, `session/set_model`, or Codex's
//!    `model[effort]` ids);
//! 4. sends the instruction with `session/prompt` and records every
//!    `session/update` into the [`Trajectory`](crate::trajectory::Trajectory)
//!    until the turn ends, answering permission prompts with the most
//!    permissive option and every other agent request with method-not-found.
//!
//! It is ported from BenchFlow 0.8's `acp/client.py`, `session.py`,
//! `runtime.py`, `watchdog.py` and `transport.py`, and robust to what agents
//! do in practice: lines of many megabytes (rewritten past 1 MiB as 0.8 does),
//! logs and terminal noise on stdout (passed on as agent output), a stall (the
//! idle watchdog, with extra grace while a tool call is pending), a crash
//! ([`AcpError::Exited`] with the exit code and stderr's tail), and a spent
//! subscription ([`AcpError::UsageLimit`], with the reset time when the agent
//! says).
//!
//! Live progress (tool calls, the last tool's title, tokens) reaches the
//! runner as [`AcpEvent`]s through [`ClientOptions::on_event`]. Whatever the
//! agent wrote that reaches an event, an error or a log line is redacted
//! first (the login's secrets and 0.8's patterns); the trajectory keeps the
//! agent's words, and the trajectory writer redacts them as it writes.

mod client;
mod conn;
mod error;
pub(crate) mod limits;
mod mcp;
mod model;
mod watchdog;
pub(crate) mod wire;

pub use client::{
    AcpClient, AgentInfo, AgentPipes, ClientOptions, DEFAULT_HANDSHAKE_TIMEOUT, ModelChoice,
    PROTOCOL_VERSION, PromptLimits, PromptOutcome, SessionInfo, StopReason, TrajectoryHandle,
};
pub use conn::EventSink;
pub use error::{AcpError, IdleReport, Owner, provider_failure_in_reply};
pub use mcp::{McpServer, McpTransport};
pub use watchdog::PENDING_GRACE_MULTIPLIER;
pub use wire::{DEFAULT_LINE_LIMIT, DEFAULT_MAX_LINE};

/// Counts for a live progress line.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Progress {
    pub tool_calls: usize,
    /// The latest tool call's title (its first line), or its kind.
    pub last_tool_title: Option<String>,
    pub message_chunks: usize,
    pub thought_chunks: usize,
    /// Tokens in the agent's context (its latest usage update), else the
    /// last prompt's reported total.
    pub tokens: Option<u64>,
}

/// Which of the agent's output streams a line came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputStream {
    /// Non-ACP text on stdout.
    Stdout,
    Stderr,
}

/// Something that happened on the connection, for live display and the
/// agent log. Delivered from the connection's tasks, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AcpEvent {
    /// The trajectory changed: new counts.
    Progress(Progress),
    /// The adapter sent a notice (a warning, a model fallback).
    Notice { severity: String, title: String },
    /// A permission prompt was answered.
    Permission {
        tool_call: Option<String>,
        option_id: String,
    },
    /// A line the agent wrote that is not ACP, secrets redacted: for the
    /// agent log (0.8's `agent/<agent>.txt`).
    AgentOutput { stream: OutputStream, line: String },
}

#[cfg(test)]
mod fuzz_tests;
#[cfg(test)]
mod tests;
