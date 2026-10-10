//! Progress events, for whatever shows a run to a person.
//!
//! The engine sends events; the CLI renders them (a live display on a
//! terminal, plain lines in CI, JSON lines with `--json`). Sending never
//! blocks and never fails the run.

use std::path::PathBuf;

use serde::Serialize;
use tokio::sync::mpsc;

use crate::outcome::{Phase, Status};

/// One trial's result, as progress displays need it.
#[derive(Debug, Clone, Serialize)]
pub struct TrialSummary {
    pub trial: String,
    pub task: String,
    pub agent: String,
    pub status: Status,
    pub reward: Option<f64>,
    pub passed: Option<bool>,
    pub timed_out: bool,
    /// One line for the report: the failure, or a reward breakdown.
    pub reason: Option<String>,
    pub seconds: f64,
    pub dir: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "kebab-case")]
pub enum Event {
    JobStarted {
        dir: PathBuf,
        total: usize,
        /// Trials already finished in an earlier run of this job.
        reused: usize,
    },
    TrialStarted {
        trial: String,
        task: String,
        agent: String,
    },
    Phase {
        trial: String,
        phase: Phase,
    },
    /// Agent activity during its phase.
    Activity {
        trial: String,
        tool_calls: u64,
        last_tool: Option<String>,
        tokens: u64,
    },
    /// A line of output worth showing in verbose mode (image builds).
    Log {
        trial: String,
        line: String,
    },
    TrialFinished(TrialSummary),
    /// A trial is waiting for a retry after a transient failure.
    Retrying {
        trial: String,
        attempt: u32,
        reason: String,
    },
}

/// Where events go. Cloning is cheap; a closed receiver is ignored.
#[derive(Debug, Clone)]
pub struct Events {
    tx: Option<mpsc::UnboundedSender<Event>>,
}

impl Events {
    pub fn new(tx: mpsc::UnboundedSender<Event>) -> Self {
        Self { tx: Some(tx) }
    }
    /// A sink that drops everything.
    pub fn none() -> Self {
        Self { tx: None }
    }
    pub fn send(&self, event: Event) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(event);
        }
    }
    pub fn phase(&self, trial: &str, phase: Phase) {
        self.send(Event::Phase {
            trial: trial.to_string(),
            phase,
        });
    }
    pub fn log(&self, trial: &str, line: impl Into<String>) {
        self.send(Event::Log {
            trial: trial.to_string(),
            line: line.into(),
        });
    }
}
