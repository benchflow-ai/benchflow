//! Budgets for a native loop, and why a loop stopped.
//!
//! The wall clock covers the whole run; the idle timeout bounds each single
//! wait (one model call, one tool call), since a native loop is silent only
//! while it waits on one of them. `--max-tool-calls` is checked before each
//! tool call runs and `--max-tokens` before each model call, so a loop never
//! starts work its budget does not cover.

use std::future::Future;
use std::time::{Duration, Instant};

use benchflow_sandbox::SandboxError;

use super::error::ModelError;
use super::recorder::Recorder;

/// What a run may spend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Limits {
    /// The wall-clock budget for the whole run.
    pub wall: Option<Duration>,
    /// The longest one model call or tool call may stay silent.
    pub idle: Option<Duration>,
    /// Most tool calls.
    pub max_tool_calls: Option<u64>,
    /// Most tokens, input plus output, summed over every model call.
    pub max_tokens: Option<u64>,
}

/// A budget that stopped a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Budget {
    ToolCalls(u64),
    Tokens(u64),
}

impl Budget {
    /// The flag that set it.
    pub fn flag(&self) -> String {
        match self {
            Budget::ToolCalls(n) => format!("--max-tool-calls {n}"),
            Budget::Tokens(n) => format!("--max-tokens {n}"),
        }
    }
}

/// Why a loop stopped.
#[derive(Debug)]
pub enum Stop {
    /// The model ended its turn (no more tool calls, `submit`, or Terminus-2's
    /// confirmed `task_complete`).
    Finished,
    /// The loop stopped because the model kept failing in the same way (tool
    /// calls whose arguments are not JSON, turn after turn): the model's
    /// failure, but the agent had its turn, so the trial is still graded.
    ModelLoop { reason: String, turns: u32 },
    /// The model refused or a content filter ended the turn.
    Refused,
    /// A budget ran out.
    Budget(Budget),
    /// The wall clock ran out.
    Timeout(Duration),
    /// One wait outlasted the idle timeout; `what` says which.
    Idle { what: String, after: Duration },
    /// The model kept failing.
    Model(ModelError),
    /// The sandbox failed under a tool call.
    Sandbox(SandboxError),
    /// The loop's own setup failed (tmux missing and not installable); one
    /// line saying why.
    Setup(String),
    /// The loop could not go on; one line saying why.
    Harness(String),
    /// The task cannot host this loop (computer-1 on an image without a
    /// display): why, and what to do.
    Unsupported { why: String, next: String },
}

impl Stop {
    /// Record why the run stopped, so the trajectory says it.
    pub fn record(&self, recorder: &Recorder) {
        match self {
            Stop::Finished => {}
            Stop::ModelLoop { reason, .. } => recorder.notice(
                "error",
                &format!("stopped: {reason}"),
                Some("the model's failure; the trial is graded on what the agent left"),
            ),
            Stop::Refused => recorder.notice(
                "warning",
                "the model refused or a content filter ended its turn",
                None,
            ),
            Stop::Budget(b) => recorder.notice(
                "warning",
                &format!("stopped by {}", b.flag()),
                Some(&match b {
                    Budget::ToolCalls(n) => format!(
                        "the agent asked for tool call {} after using its budget of {n}; the call did not run",
                        n + 1
                    ),
                    Budget::Tokens(n) => format!(
                        "the agent had used its budget of {n} tokens (input plus output); no further model call was made"
                    ),
                }),
            ),
            Stop::Timeout(budget) => recorder.timeout(*budget),
            Stop::Idle { what, after } => recorder.notice(
                "error",
                &format!("stopped: {what} was silent for {}s (the idle timeout)", after.as_secs()),
                None,
            ),
            Stop::Model(e) => recorder.notice("error", &format!("model call failed: {e}"), None),
            Stop::Sandbox(e) => recorder.notice("error", &format!("sandbox failed: {e}"), None),
            Stop::Setup(why) | Stop::Harness(why) => recorder.notice("error", why, None),
            Stop::Unsupported { why, next } => recorder.notice("error", why, Some(next)),
        }
    }
}

/// A run's clock and counters.
#[derive(Debug, Clone)]
pub struct Meter {
    pub limits: Limits,
    started: Instant,
    pub tool_calls: u64,
}

impl Meter {
    pub fn new(limits: Limits) -> Self {
        Self {
            limits,
            started: Instant::now(),
            tool_calls: 0,
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }

    /// Time left on the wall clock; `None` without a budget.
    pub fn remaining(&self) -> Option<Duration> {
        self.limits
            .wall
            .map(|w| w.saturating_sub(self.started.elapsed()))
    }

    /// Stop before a model call when the clock or the tokens are spent.
    pub fn before_model_call(&self, tokens_used: u64) -> Result<(), Stop> {
        self.check_clock()?;
        match self.limits.max_tokens {
            Some(max) if tokens_used >= max => Err(Stop::Budget(Budget::Tokens(max))),
            _ => Ok(()),
        }
    }

    /// Stop before a tool call when the clock or the calls are spent;
    /// otherwise count the call.
    pub fn before_tool_call(&mut self) -> Result<(), Stop> {
        self.check_clock()?;
        if let Some(max) = self.limits.max_tool_calls
            && self.tool_calls >= max
        {
            return Err(Stop::Budget(Budget::ToolCalls(max)));
        }
        self.tool_calls += 1;
        Ok(())
    }

    fn check_clock(&self) -> Result<(), Stop> {
        match (self.limits.wall, self.remaining()) {
            (Some(w), Some(r)) if r.is_zero() => Err(Stop::Timeout(w)),
            _ => Ok(()),
        }
    }

    /// Wait for `fut` within the wall clock and the idle timeout.
    pub async fn bound<T>(&self, what: &str, fut: impl Future<Output = T>) -> Result<T, Stop> {
        let wall = self.remaining();
        let idle = self.limits.idle.filter(|d| !d.is_zero());
        let limit = match (wall, idle) {
            (Some(w), Some(i)) => Some(w.min(i)),
            (w, i) => w.or(i),
        };
        let Some(limit) = limit else {
            return Ok(fut.await);
        };
        match tokio::time::timeout(limit, fut).await {
            Ok(v) => Ok(v),
            Err(_) if wall.is_some_and(|w| w <= limit) => {
                Err(Stop::Timeout(self.limits.wall.unwrap_or_default()))
            }
            Err(_) => Err(Stop::Idle {
                what: what.to_string(),
                after: limit,
            }),
        }
    }

    /// Wait for `fut` within the wall clock only (for waits that carry
    /// their own timeout, such as a sandbox command).
    pub async fn bound_wall<T>(&self, fut: impl Future<Output = T>) -> Result<T, Stop> {
        match self.remaining() {
            None => Ok(fut.await),
            Some(r) => tokio::time::timeout(r, fut)
                .await
                .map_err(|_| Stop::Timeout(self.limits.wall.unwrap_or_default())),
        }
    }

    /// Sleep `d`, cut short by the wall clock.
    pub async fn sleep(&self, d: Duration) -> Result<(), Stop> {
        match self.remaining() {
            Some(r) if r <= d => {
                tokio::time::sleep(r).await;
                Err(Stop::Timeout(self.limits.wall.unwrap_or_default()))
            }
            _ => {
                tokio::time::sleep(d).await;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tool_call_budget_stops_the_call_after_the_last_allowed() {
        let mut m = Meter::new(Limits {
            max_tool_calls: Some(2),
            ..Limits::default()
        });
        assert!(m.before_tool_call().is_ok());
        assert!(m.before_tool_call().is_ok());
        assert!(matches!(
            m.before_tool_call(),
            Err(Stop::Budget(Budget::ToolCalls(2)))
        ));
        assert_eq!(m.tool_calls, 2);
    }

    #[test]
    fn the_token_budget_stops_the_next_model_call() {
        let m = Meter::new(Limits {
            max_tokens: Some(100),
            ..Limits::default()
        });
        assert!(m.before_model_call(99).is_ok());
        assert!(matches!(
            m.before_model_call(100),
            Err(Stop::Budget(Budget::Tokens(100)))
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn a_silent_wait_is_idle_unless_the_clock_runs_out_first() {
        let m = Meter::new(Limits {
            wall: Some(Duration::from_secs(100)),
            idle: Some(Duration::from_secs(10)),
            ..Limits::default()
        });
        let r = m
            .bound("the model", tokio::time::sleep(Duration::from_secs(20)))
            .await;
        assert!(matches!(r, Err(Stop::Idle { .. })));
        let m = Meter::new(Limits {
            wall: Some(Duration::from_secs(5)),
            idle: Some(Duration::from_secs(10)),
            ..Limits::default()
        });
        let r = m
            .bound("the model", tokio::time::sleep(Duration::from_secs(20)))
            .await;
        assert!(matches!(r, Err(Stop::Timeout(_))));
    }
}
