//! Native harnesses: agent loops that run inside `bench` on the host.
//!
//! The model is called over HTTP from the host ([`client`]), so the
//! provider key never enters the sandbox; the loop acts in the sandbox
//! through the sandbox API. Every request and response is recorded
//! (`agent/model_calls.jsonl`, secrets redacted), and every step goes into
//! the shared trajectory ([`recorder`]).
//!
//! - [`builtin`]: a tool-calling loop (`bash`, `view`, `replace`, `write`,
//!   `submit`) after Karotte's builtin agent.
//! - [`terminus`]: Terminus-2, the keystrokes-to-tmux loop, ported from
//!   Harbor with its prompt templates and response parsers.
//! - [`computer`]: computer-1, Harbor's screenshot, model, one action loop
//!   on a virtual display in the sandbox.
//!
//! All stop on the same budgets ([`budget`]) and report why they stopped.

pub mod anthropic;
pub mod budget;
pub mod builtin;
pub mod chat;
pub mod client;
pub mod computer;
pub mod endpoint;
pub mod error;
pub mod openai;
pub mod options;
pub mod recorder;
pub mod terminus;

use std::collections::BTreeMap;

use benchflow_sandbox::Sandbox;

pub use budget::{Budget, Limits, Meter, Stop};
pub use chat::TokenUsage;
pub use client::{ClientOptions, ModelClient};
pub use endpoint::{Api, BuiltinProviders, Endpoint, EndpointError, EndpointResolver};
pub use error::{ModelError, ModelErrorKind};
pub use recorder::Recorder;

/// The native agents, by name, with their aliases.
pub const AGENTS: &[(&str, &[&str], &str)] = &[
    (
        BUILTIN,
        &[],
        "BenchFlow's tool loop (bash, view, replace, write, submit); the model is called from the host",
    ),
    (
        TERMINUS_2,
        &["terminus2"],
        "Terminus-2: keystrokes to a tmux session, JSON replies, context summarization; the model is called from the host",
    ),
    (
        COMPUTER_1,
        &["computer1"],
        "computer-1: a screenshot of the sandbox's desktop, one action from a vision model, xdotool; needs a task image with a display",
    ),
];

pub const BUILTIN: &str = "builtin";
pub const TERMINUS_2: &str = "terminus-2";
pub const COMPUTER_1: &str = "computer-1";

/// The native agent a name or alias means.
pub fn find(name: &str) -> Option<&'static str> {
    AGENTS
        .iter()
        .find(|(n, aliases, _)| *n == name || aliases.contains(&name))
        .map(|(n, _, _)| *n)
}

/// Where a loop runs and what it may spend.
pub struct RunContext<'a> {
    pub sandbox: &'a dyn Sandbox,
    /// The user tools and the terminal run as.
    pub user: &'a str,
    pub workdir: &'a str,
    /// Environment for commands the agent runs (task and run variables,
    /// `HOME`; never a credential).
    pub env: BTreeMap<String, String>,
    /// The instruction, then each later stage.
    pub prompts: Vec<String>,
    pub limits: Limits,
    /// MCP servers the task declares (Terminus-2 lists them in its prompt).
    pub mcp: &'a [serde_json::Value],
    /// Where the loop may keep files on the host (the trial's `agent/`
    /// folder): computer-1's screenshots.
    pub agent_dir: Option<std::path::PathBuf>,
}

/// Which native loop to run, with its settings.
#[derive(Debug, Clone)]
pub enum NativeLoop {
    Builtin(builtin::BuiltinConfig),
    Terminus(terminus::TerminusConfig),
    Computer(computer::ComputerConfig),
}

impl NativeLoop {
    /// The loop for a native agent's name (from [`find`]).
    pub fn for_agent(name: &str) -> Option<Self> {
        match find(name)? {
            BUILTIN => Some(Self::Builtin(builtin::BuiltinConfig::default())),
            TERMINUS_2 => Some(Self::Terminus(terminus::TerminusConfig::default())),
            COMPUTER_1 => Some(Self::Computer(computer::ComputerConfig::default())),
            _ => None,
        }
    }

    /// Run the loop, record why it stopped, and close the trajectory.
    pub async fn run(
        &self,
        client: &ModelClient,
        recorder: &Recorder,
        ctx: &RunContext<'_>,
    ) -> Stop {
        let stop = match self {
            Self::Builtin(c) => builtin::run(client, recorder, ctx, c).await,
            Self::Terminus(c) => terminus::run(client, recorder, ctx, c).await,
            Self::Computer(c) => computer::run(client, recorder, ctx, c).await,
        };
        recorder.usage(&client.usage());
        stop.record(recorder);
        recorder.finish();
        stop
    }
}
