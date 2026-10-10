//! Agents: what BenchFlow runs inside a sandbox to attempt a task.
//!
//! - [`manifest`]: an agent is data, a TOML manifest saying how to install
//!   it, how to launch it, which logins it accepts, and how its model is
//!   selected. The built-in manifests (Claude Code, Codex, Gemini CLI,
//!   OpenCode, goose, aider, fx) are embedded in the binary; a project can
//!   add its own.
//! - [`credentials`]: finds the login an agent can use on this machine and
//!   turns it into the environment and files the sandbox needs.
//! - [`providers`]: model providers as data (`vercel/<model>`), and the
//!   forwarder that holds a provider's key inside the sandbox so the agent
//!   never sees it.
//! - [`acp`]: a client for the Agent Client Protocol, spoken over a
//!   process's stdin and stdout.
//! - [`headless`]: runs an agent's own non-interactive mode and parses its
//!   output stream into the same trajectory events.
//! - [`runners`]: the runner scripts that run SDK agents (a framework, not
//!   a CLI) headless.
//! - [`native`]: agent loops that run in `bench` on the host and call the
//!   model over HTTP (the builtin tool loop, Terminus-2).
//! - [`external`]: Karotte's external agent, where an outside backend (an
//!   RL trainer) decides each step over HTTP.
//! - [`acp_registry`]: the public ACP registry, whose entries run as
//!   `acp:<id>[@<version>]`.
//! - [`trajectory`]: the events of a run, written as 0.8's
//!   `acp_trajectory.jsonl` and exportable as ATIF.

pub mod acp;
pub mod acp_registry;
pub mod credentials;
pub mod external;
pub mod headless;
pub mod manifest;
pub mod native;
pub mod providers;
pub mod runners;
pub mod trajectory;

mod model_id;
mod shell;
mod suggest;

pub use manifest::{AgentManifest, ManifestError, Registry, UnknownAgent};

/// The scripted agents every run understands. They need no model and no key.
pub mod builtin {
    /// Runs the task's reference solution (`oracle/solve.sh`).
    pub const ORACLE: &str = "oracle";
    /// Does nothing; the verifier scores the untouched workspace.
    pub const NOP: &str = "nop";
    /// Prefix of BenchFlow's exploit probes (`probe:forge-reward`).
    pub const PROBE_PREFIX: &str = "probe:";

    pub fn is_scripted(name: &str) -> bool {
        name == ORACLE || name == NOP || name.starts_with(PROBE_PREFIX)
    }
}
