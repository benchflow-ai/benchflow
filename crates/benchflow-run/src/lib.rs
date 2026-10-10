//! The BenchFlow run engine.
//!
//! A [`job`] runs [`trial`]s: an [`agent::Attempt`] at a task in a sandbox,
//! handed to the task's verifier under [`hardening`], scored by
//! [`scoring`], and recorded on disk ([`record`]) in BenchFlow 0.8's layout.
//! [`testplan`] turns trials of the oracle, an empty run, negative controls,
//! and exploit [`probes`] into a verdict on whether a task proves itself.

pub mod acp_agent;
pub mod agent;
pub mod agent_check;
pub mod cache;
pub mod config;
pub mod confine;
pub mod confinement;
pub mod credentials;
pub mod custody;
pub mod diff;
pub mod dspy_agent;
pub mod envcheck;
pub mod events;
pub mod external_agent;
pub mod features;
mod forwarding;
pub mod hardening;
pub mod headless_agent;
pub mod images;
pub mod job;
pub mod jobfile;
pub mod lock;
pub mod metrics;
pub mod native_agent;
pub mod netpolicy;
pub mod outcome;
pub mod probes;
pub mod project;
pub mod read;
pub mod record;
pub mod registry;
pub mod regrade;
pub mod review;
pub mod scoring;
pub mod skills;
pub mod source;
pub mod testplan;
pub mod traces;
pub mod trainer;
pub mod trial;

pub use agent::{Attempt, ModelAgent};
pub use config::{Keep, RunOptions};
pub use events::{Event, Events};
pub use job::{JobOutcome, JobSpec};
pub use outcome::{Owner, Problem, Status};
pub use trial::Runtime;

/// Quote a word for a POSIX shell.
pub(crate) fn shell_quote(word: &str) -> String {
    benchflow_sandbox::shell::quote(word)
}
