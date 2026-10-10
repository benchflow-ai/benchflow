//! The idle watchdog: stop a prompt when the agent goes quiet (0.8's
//! `acp/watchdog.py`).
//!
//! A prompt is idle when no tool call, message chunk or thought chunk has
//! arrived for the idle budget. A tool call still pending defers that: a
//! silent command may run long. Each pending call has its own grace (by
//! default three idle budgets), restarted by its own progress updates, and
//! also when a newer tool call starts (#1141): an agent that started
//! another call is not waiting on the older one, whose terminal update was
//! probably lost. A call whose grace runs out stops the prompt even while
//! other activity continues.

use std::collections::HashMap;
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::time::Instant;

use super::error::{AcpError, IdleReport};
use crate::trajectory::Activity;

/// A pending call defers the idle abort for at most this many idle budgets.
pub const PENDING_GRACE_MULTIPLIER: u32 = 3;

pub(crate) struct Watchdog {
    idle: Duration,
    grace: Duration,
    started: Instant,
    last_progress: Instant,
    last_activity_at: DateTime<Utc>,
    last_activity_count: usize,
    pending: Vec<(String, u64)>,
    pending_progress: HashMap<String, Instant>,
    pending_last_update_at: HashMap<String, DateTime<Utc>>,
    pending_set_changed_at: Option<DateTime<Utc>>,
    tool_calls: usize,
    /// Per pending call: tool calls that started after its last update.
    newer_calls: HashMap<String, usize>,
}

fn activity_count(activity: &Activity) -> usize {
    activity.tool_calls + activity.message_chunks + activity.thought_chunks
}

impl Watchdog {
    pub(crate) fn start(
        activity: &Activity,
        idle: Duration,
        grace: Option<Duration>,
        now: Instant,
        wall_now: DateTime<Utc>,
    ) -> Self {
        Self {
            idle,
            grace: grace.unwrap_or(idle * PENDING_GRACE_MULTIPLIER),
            started: now,
            last_progress: now,
            last_activity_at: wall_now,
            last_activity_count: activity_count(activity),
            pending: activity.pending.clone(),
            pending_progress: activity
                .pending
                .iter()
                .map(|(id, _)| (id.clone(), now))
                .collect(),
            pending_last_update_at: HashMap::new(),
            pending_set_changed_at: (!activity.pending.is_empty()).then_some(wall_now),
            tool_calls: activity.tool_calls,
            newer_calls: activity
                .pending
                .iter()
                .map(|(id, _)| (id.clone(), 0))
                .collect(),
        }
    }

    /// How often to look: a quarter of the idle and wall budgets, 1 to 30 s.
    pub(crate) fn poll_interval(idle: Duration, wall: Option<Duration>) -> Duration {
        let mut secs = (idle.as_secs() / 4).min(30);
        if let Some(wall) = wall {
            secs = secs.min((wall.as_secs() / 4).max(1));
        }
        Duration::from_secs(secs.max(1))
    }

    fn pending_ids(&self) -> impl Iterator<Item = &String> {
        self.pending.iter().map(|(id, _)| id)
    }

    /// Take in one look at the session's activity.
    pub(crate) fn observe(&mut self, activity: &Activity, now: Instant, wall_now: DateTime<Utc>) {
        let previous: HashMap<&str, u64> = self
            .pending
            .iter()
            .map(|(id, v)| (id.as_str(), *v))
            .collect();
        if !activity
            .pending
            .iter()
            .map(|(id, _)| id)
            .eq(self.pending_ids())
        {
            self.pending_set_changed_at = Some(wall_now);
        }
        let newer = activity.tool_calls.saturating_sub(self.tool_calls);
        self.tool_calls = activity.tool_calls;

        for (id, version) in &activity.pending {
            match previous.get(id.as_str()) {
                None => {
                    self.pending_progress.insert(id.clone(), now);
                    self.newer_calls.insert(id.clone(), 0);
                }
                // Every call recorded since the last look started after this
                // one: the agent moved on, so restart this call's grace.
                Some(_) if newer > 0 => {
                    self.pending_progress.insert(id.clone(), now);
                    *self.newer_calls.entry(id.clone()).or_insert(0) += newer;
                }
                Some(_) => {}
            }
            if *version > previous.get(id.as_str()).copied().unwrap_or(0) {
                self.pending_progress.insert(id.clone(), now);
                self.pending_last_update_at.insert(id.clone(), wall_now);
                self.newer_calls.insert(id.clone(), 0);
            }
        }
        let current: std::collections::HashSet<&str> =
            activity.pending.iter().map(|(id, _)| id.as_str()).collect();
        self.pending_progress
            .retain(|id, _| current.contains(id.as_str()));
        self.pending_last_update_at
            .retain(|id, _| current.contains(id.as_str()));
        self.newer_calls
            .retain(|id, _| current.contains(id.as_str()));
        self.pending = activity.pending.clone();

        let count = activity_count(activity);
        if count > self.last_activity_count {
            self.last_progress = now;
            self.last_activity_at = wall_now;
            self.last_activity_count = count;
        } else if !self.pending.is_empty() && self.expired_pending_ids(now).is_empty() {
            // Silent tools may run longer than the idle budget; the grace,
            // not unrelated noise, owns this deferral.
            self.last_progress = now;
            self.last_activity_at = wall_now;
        }
    }

    fn expired_pending_ids(&self, now: Instant) -> Vec<String> {
        self.pending_ids()
            .filter(|id| {
                self.pending_progress
                    .get(id.as_str())
                    .is_some_and(|at| now.duration_since(*at) >= self.grace)
            })
            .cloned()
            .collect()
    }

    fn lost_update_ids(&self) -> Vec<String> {
        self.pending_ids()
            .filter(|id| self.newer_calls.get(id.as_str()).copied().unwrap_or(0) > 0)
            .cloned()
            .collect()
    }

    pub(crate) fn idle_expired(&self, now: Instant) -> bool {
        !self.expired_pending_ids(now).is_empty()
            || now.duration_since(self.last_progress) >= self.idle
    }

    /// The error that stops the prompt, with 0.8's wording.
    pub(crate) fn timeout_error(
        &self,
        activity: &Activity,
        now: Instant,
        wall_now: DateTime<Utc>,
    ) -> AcpError {
        let pending_ids: Vec<String> = self.pending_ids().cloned().collect();
        let expired = self.expired_pending_ids(now);
        let versions: HashMap<&str, u64> = self
            .pending
            .iter()
            .map(|(id, v)| (id.as_str(), *v))
            .collect();
        let expired_updates: u64 = expired
            .iter()
            .map(|id| versions.get(id.as_str()).copied().unwrap_or(0))
            .sum();
        let idle_for = if expired.is_empty() {
            now.duration_since(self.last_progress)
        } else {
            expired
                .iter()
                .filter_map(|id| self.pending_progress.get(id.as_str()))
                .map(|at| now.duration_since(*at))
                .max()
                .unwrap_or_default()
        };
        let last_expired_update = expired
            .iter()
            .filter_map(|id| self.pending_last_update_at.get(id.as_str()))
            .max()
            .copied();
        let lost = self.lost_update_ids();
        let report = IdleReport {
            idle_timeout_sec: self.idle.as_secs(),
            idle_duration_sec: idle_for.as_secs(),
            wall_clock_elapsed_sec: now.duration_since(self.started).as_secs(),
            n_tool_calls: activity.tool_calls,
            n_message_chunks: activity.message_chunks,
            n_thought_chunks: activity.thought_chunks,
            last_activity_at: self.last_activity_at,
            pending_tool_call_ids: pending_ids,
            expired_pending_tool_call_ids: expired.clone(),
            pending_grace_sec: self.grace.as_secs(),
            pending_set_last_changed_at: self.pending_set_changed_at,
            last_tool_update_at: last_expired_update,
            n_tool_call_updates: activity.tool_call_updates,
            n_pending_tool_call_updates: self.pending.iter().map(|(_, v)| v).sum(),
            n_expired_pending_tool_call_updates: expired_updates,
            lost_update_tool_call_ids: lost.clone(),
        };
        let lost_note = if lost.is_empty() {
            String::new()
        } else {
            let calls: Vec<String> = lost
                .iter()
                .map(|id| {
                    format!(
                        "{id} ({} later)",
                        self.newer_calls.get(id.as_str()).copied().unwrap_or(0)
                    )
                })
                .collect();
            format!(
                "; lost update: {} pending tool call(s) got no update after later tool calls started: {}",
                lost.len(),
                calls.join(", ")
            )
        };
        let age = |then: Option<DateTime<Utc>>| match then {
            None => "never".to_string(),
            Some(then) => format!("{}s ago", (wall_now - then).num_seconds().max(0)),
        };
        let message = if expired.is_empty() {
            format!(
                "Agent idle for {}s with no new tool call, message, or thought \
                 (last activity {}s ago, {} tool calls so far){lost_note}",
                self.idle.as_secs(),
                now.duration_since(self.last_progress).as_secs(),
                activity.tool_calls,
            )
        } else {
            format!(
                "Agent idle for {}s: {} pending tool call(s) exceeded the {}s pending grace \
                 ({}; pending set last changed {}, last pending-call update {}, \
                 {expired_updates} relevant updates seen, {} tool calls so far){lost_note}",
                self.idle.as_secs(),
                expired.len(),
                self.grace.as_secs(),
                expired.join(", "),
                age(self.pending_set_changed_at),
                age(last_expired_update),
                activity.tool_calls,
            )
        };
        AcpError::Idle {
            message,
            report: Box::new(report),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn activity(tool_calls: usize, chunks: usize, pending: &[(&str, u64)]) -> Activity {
        Activity {
            tool_calls,
            message_chunks: chunks,
            thought_chunks: 0,
            tool_call_updates: pending.iter().map(|(_, v)| v).sum(),
            pending: pending.iter().map(|(id, v)| (id.to_string(), *v)).collect(),
        }
    }

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn poll_interval_follows_0_8() {
        assert_eq!(
            Watchdog::poll_interval(secs(600), Some(secs(3600))),
            secs(30)
        );
        assert_eq!(Watchdog::poll_interval(secs(20), Some(secs(3600))), secs(5));
        assert_eq!(Watchdog::poll_interval(secs(20), Some(secs(8))), secs(2));
        assert_eq!(Watchdog::poll_interval(secs(2), None), secs(1));
    }

    #[test]
    fn silence_without_pending_calls_is_idle() {
        let t0 = Instant::now();
        let wall = Utc::now();
        let mut dog = Watchdog::start(&activity(0, 0, &[]), secs(60), None, t0, wall);
        dog.observe(&activity(0, 3, &[]), t0 + secs(30), wall);
        assert!(!dog.idle_expired(t0 + secs(80)));
        assert!(dog.idle_expired(t0 + secs(90)));
        let err = dog.timeout_error(&activity(0, 3, &[]), t0 + secs(90), wall);
        assert_eq!(
            err.to_string(),
            "Agent idle for 60s with no new tool call, message, or thought (last activity 60s ago, 0 tool calls so far)"
        );
    }

    #[test]
    fn a_pending_call_defers_until_its_grace_runs_out() {
        let t0 = Instant::now();
        let wall = Utc::now();
        let mut dog = Watchdog::start(&activity(1, 0, &[("t1", 0)]), secs(60), None, t0, wall);
        // Silent for two idle budgets: the pending call defers the abort.
        dog.observe(&activity(1, 0, &[("t1", 0)]), t0 + secs(120), wall);
        assert!(!dog.idle_expired(t0 + secs(120)));
        // Its grace (3 x 60 s) runs out at 180 s.
        assert!(dog.idle_expired(t0 + secs(180)));
        let err = dog.timeout_error(&activity(1, 0, &[("t1", 0)]), t0 + secs(180), wall);
        assert!(
            err.to_string().starts_with(
                "Agent idle for 60s: 1 pending tool call(s) exceeded the 180s pending grace (t1; pending set last changed"
            ),
            "{err}"
        );
    }

    #[test]
    fn a_newer_call_restarts_the_grace_of_one_whose_update_was_lost() {
        // Guards 0.8's #1141: one lost terminal update must not kill a session
        // that keeps completing tool calls.
        let t0 = Instant::now();
        let wall = Utc::now();
        let mut dog = Watchdog::start(&activity(1, 0, &[("lost", 1)]), secs(60), None, t0, wall);
        for step in 1..=5u64 {
            // A new call starts and completes every 50 s; "lost" stays pending.
            dog.observe(
                &activity(1 + step as usize, 0, &[("lost", 1)]),
                t0 + secs(50 * step),
                wall,
            );
            assert!(!dog.idle_expired(t0 + secs(50 * step)), "step {step}");
        }
        let AcpError::Idle { report, .. } =
            dog.timeout_error(&activity(6, 0, &[("lost", 1)]), t0 + secs(250), wall)
        else {
            panic!()
        };
        assert_eq!(report.lost_update_tool_call_ids, ["lost"]);
        // Without new calls its grace runs out.
        assert!(dog.idle_expired(t0 + secs(250 + 180)));
    }

    #[test]
    fn progress_updates_restart_only_their_own_call() {
        let t0 = Instant::now();
        let wall = Utc::now();
        let mut dog = Watchdog::start(
            &activity(2, 0, &[("a", 0), ("b", 0)]),
            secs(60),
            Some(secs(100)),
            t0,
            wall,
        );
        dog.observe(&activity(2, 0, &[("a", 3), ("b", 0)]), t0 + secs(90), wall);
        // b has had no progress for 100 s.
        assert!(dog.idle_expired(t0 + secs(100)));
        let AcpError::Idle { report, .. } =
            dog.timeout_error(&activity(2, 0, &[("a", 3), ("b", 0)]), t0 + secs(100), wall)
        else {
            panic!()
        };
        assert_eq!(report.expired_pending_tool_call_ids, ["b"]);
        assert_eq!(report.pending_grace_sec, 100);
    }
}
