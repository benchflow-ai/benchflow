//! Headless agents: an agent's own non-interactive mode, its output parsed.
//!
//! Many coding agents have a mode that takes one prompt, works until done,
//! and prints what it did: `claude -p --output-format stream-json`, `codex
//! exec --json`, `opencode run --format json`, `goose run --output-format
//! stream-json`, `fx ask --json`, or plain text. SDK agents, a framework
//! rather than a CLI, get that mode from one of BenchFlow's runner scripts
//! ([`crate::runners`]), which all write the `benchflow-runner` format. The manifest's
//! `[headless]` table says how to start it and which [`OutputFormat`] its
//! stdout is in; the engine starts the process in the sandbox and hands it
//! to a [`Runner`], which
//!
//! - feeds every stdout line to the format's [`Parser`], which turns it into
//!   ACP-shaped updates, so the shared [`Trajectory`](crate::trajectory::Trajectory)
//!   merges them exactly as it merges ACP's (live progress, the trajectory
//!   writer and `summarize` work unchanged);
//! - keeps lines that are not part of the format, and stderr, for the agent
//!   log, secrets redacted;
//! - stops the process at the wall-clock budget or when it goes idle;
//! - and, when the run fails, says why as a [`HeadlessError`] with an owner
//!   and a next step: usage limits, rejected logins, unknown models and
//!   provider failures are recognized from the agent's own words (Harbor's
//!   and BenchFlow 0.8's error patterns), the rest is a crash or an agent
//!   error with stderr's tail as evidence.

pub(crate) mod classify;
mod error;
mod parser;
mod parsers;
mod runner;

use std::fmt;

use serde::Serialize;

pub use error::HeadlessError;
pub use parser::{Failure, Hint, Out, Parser, ToolDone, ToolStart};
pub use runner::{Finished, Recording, Runner, RunnerOptions};

/// The output formats BenchFlow parses, by the name a manifest uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum OutputFormat {
    /// `claude -p --verbose --output-format stream-json`.
    ClaudeStreamJson,
    /// `codex exec --json`.
    CodexJson,
    /// `opencode run --format json`.
    OpencodeJson,
    /// `gemini --prompt`: the answer as plain text.
    GeminiText,
    /// `goose run --output-format stream-json`.
    GooseStreamJson,
    /// `fx ask --json`.
    FxJson,
    /// aider's console.
    AiderText,
    /// BenchFlow's runner scripts for SDK agents ([`crate::runners`]).
    BenchflowRunner,
    /// Every line is the agent's message.
    Text,
    /// `kimi --prompt ... --output-format stream-json` (Kimi Code), or
    /// `kimi --print --output-format stream-json` (kimi-cli).
    KimiStreamJson,
    /// `mcode exec --output-format stream-json` (MiniMax Code).
    McodeStreamJson,
    /// `openclaw agent --local --json`, its session transcript followed.
    OpenclawJson,
    /// `muse exec --json` (Muse Code).
    MuseJson,
    /// mini-swe-agent's console, its trajectory file followed.
    MiniSweAgentTrajectory,
    /// SWE-agent's console, its `.traj` file followed.
    SweAgentTrajectory,
    /// trae-cli's console, its trajectory file followed.
    TraeTrajectory,
    /// `hermes chat -q`, then `hermes sessions export`.
    HermesSession,
    /// Code Puppy's console, then its usage file.
    CodePuppyText,
    /// `cursor-agent --print --output-format stream-json` (Cursor).
    CursorStreamJson,
    /// `agy --output-format stream-json` (Google Antigravity's CLI).
    AntigravityStreamJson,
    /// `grok --output-format streaming-json` (Grok Build), its chat history followed.
    GrokStreamingJson,
    /// `copilot --output-format json` (GitHub Copilot CLI).
    CopilotJson,
    /// `pi --print --mode json` (Pi).
    PiJson,
    /// `cline --json`, then its session's messages file.
    ClineJson,
    /// `openhands --headless --json` (the OpenHands CLI's SDK events).
    OpenhandsJson,
    /// `qwen --prompt` (Qwen Code), its chat recording followed.
    QwenSession,
    /// `vibe --output streaming` (Mistral Vibe 2.26), then its run export.
    VibeStreaming,
}

impl OutputFormat {
    pub const ALL: [OutputFormat; 27] = [
        Self::ClaudeStreamJson,
        Self::CodexJson,
        Self::OpencodeJson,
        Self::GeminiText,
        Self::GooseStreamJson,
        Self::FxJson,
        Self::AiderText,
        Self::BenchflowRunner,
        Self::Text,
        Self::KimiStreamJson,
        Self::McodeStreamJson,
        Self::OpenclawJson,
        Self::MuseJson,
        Self::MiniSweAgentTrajectory,
        Self::SweAgentTrajectory,
        Self::TraeTrajectory,
        Self::HermesSession,
        Self::CodePuppyText,
        Self::CursorStreamJson,
        Self::AntigravityStreamJson,
        Self::GrokStreamingJson,
        Self::CopilotJson,
        Self::PiJson,
        Self::ClineJson,
        Self::OpenhandsJson,
        Self::QwenSession,
        Self::VibeStreaming,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeStreamJson => "claude-stream-json",
            Self::CodexJson => "codex-json",
            Self::OpencodeJson => "opencode-json",
            Self::GeminiText => "gemini-text",
            Self::GooseStreamJson => "goose-stream-json",
            Self::FxJson => "fx-json",
            Self::AiderText => "aider-text",
            Self::BenchflowRunner => "benchflow-runner",
            Self::Text => "text",
            Self::KimiStreamJson => "kimi-stream-json",
            Self::McodeStreamJson => "mcode-stream-json",
            Self::OpenclawJson => "openclaw-json",
            Self::MuseJson => "muse-json",
            Self::MiniSweAgentTrajectory => "mini-swe-agent-trajectory",
            Self::SweAgentTrajectory => "swe-agent-trajectory",
            Self::TraeTrajectory => "trae-trajectory",
            Self::HermesSession => "hermes-session",
            Self::CodePuppyText => "code-puppy-text",
            Self::CursorStreamJson => "cursor-stream-json",
            Self::AntigravityStreamJson => "antigravity-stream-json",
            Self::GrokStreamingJson => "grok-streaming-json",
            Self::CopilotJson => "copilot-json",
            Self::PiJson => "pi-json",
            Self::ClineJson => "cline-json",
            Self::OpenhandsJson => "openhands-json",
            Self::QwenSession => "qwen-session",
            Self::VibeStreaming => "vibe-streaming",
        }
    }

    /// Every format's name, for error messages.
    pub fn names() -> Vec<&'static str> {
        Self::ALL.iter().map(|f| f.name()).collect()
    }

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    /// A fresh parser for one run.
    pub fn parser(self) -> Box<dyn Parser> {
        match self {
            Self::ClaudeStreamJson => Box::<parsers::claude::ClaudeStreamJson>::default(),
            Self::CodexJson => Box::<parsers::codex::CodexJson>::default(),
            Self::OpencodeJson => Box::<parsers::opencode::OpencodeJson>::default(),
            Self::GeminiText => Box::<parsers::text::GeminiText>::default(),
            Self::GooseStreamJson => Box::<parsers::goose::GooseStreamJson>::default(),
            Self::FxJson => Box::<parsers::fx::FxJson>::default(),
            Self::AiderText => Box::<parsers::text::AiderText>::default(),
            Self::BenchflowRunner => Box::<parsers::runner::BenchflowRunner>::default(),
            Self::Text => Box::<parsers::text::PlainText>::default(),
            Self::KimiStreamJson => Box::<parsers::kimi::KimiStreamJson>::default(),
            Self::McodeStreamJson => Box::<parsers::mcode::McodeStreamJson>::default(),
            Self::OpenclawJson => Box::<parsers::openclaw::OpenclawJson>::default(),
            Self::MuseJson => Box::<parsers::muse::MuseJson>::default(),
            Self::MiniSweAgentTrajectory => Box::<parsers::mini::MiniSweAgent>::default(),
            Self::SweAgentTrajectory => Box::<parsers::sweagent::SweAgent>::default(),
            Self::TraeTrajectory => Box::<parsers::trae::Trae>::default(),
            Self::HermesSession => Box::<parsers::hermes::HermesSession>::default(),
            Self::CodePuppyText => Box::<parsers::text::CodePuppyText>::default(),
            Self::CursorStreamJson => Box::<parsers::cursor::CursorStreamJson>::default(),
            Self::AntigravityStreamJson => {
                Box::<parsers::antigravity::AntigravityStreamJson>::default()
            }
            Self::GrokStreamingJson => Box::<parsers::grok::GrokStreamingJson>::default(),
            Self::CopilotJson => Box::<parsers::copilot::CopilotJson>::default(),
            Self::PiJson => Box::<parsers::pi::PiJson>::default(),
            Self::ClineJson => Box::<parsers::cline::ClineJson>::default(),
            Self::OpenhandsJson => Box::<parsers::openhands::OpenhandsJson>::default(),
            Self::QwenSession => Box::<parsers::qwen::QwenSession>::default(),
            Self::VibeStreaming => Box::<parsers::vibe::VibeStreaming>::default(),
        }
    }

    /// Helper scripts the format's commands run, as (path under the agent's
    /// home, contents); the harness writes them, executable, before the
    /// agent starts. The trajectory-file formats follow the agent's file
    /// while it works with `.benchflow/follow.py` or `.benchflow/follow.sh`
    /// (see `parsers/follow.rs`).
    pub fn helper_files(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::MiniSweAgentTrajectory | Self::SweAgentTrajectory | Self::TraeTrajectory => {
                &[parsers::follow::FOLLOW_PY]
            }
            Self::OpenclawJson => &[parsers::follow::FOLLOW_SQLITE],
            Self::GrokStreamingJson | Self::QwenSession => &[parsers::follow::FOLLOW_SH],
            _ => &[],
        }
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
pub(crate) mod tests;
