//! `bench test`: does a task prove itself?
//!
//! For each task, four kinds of trial run side by side on the same image:
//!
//! | Trial | Expectation |
//! |---|---|
//! | the oracle | passes: the verifier can say yes |
//! | an empty run (`nop`) | fails: the verifier can say no |
//! | each negative control in `controls/` | fails, or stays at or below its declared reward |
//! | each exploit probe | scores no higher than the empty run, and grading still finishes |
//!
//! A task whose every expectation holds proves itself.

use std::sync::Arc;

use benchflow_task::TaskPackage;
use serde::Serialize;

use crate::agent::Attempt;
use crate::outcome::Status;
use crate::probes::{self, Probe};
use crate::record::{TrialRecord, fmt_reward};

/// Which probes to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeSet {
    All,
    None,
    Only(Vec<String>),
}

/// The attempts `bench test` runs for one task.
pub fn attempts(task: &TaskPackage, probes: &ProbeSet) -> Vec<Attempt> {
    let mut v = Vec::new();
    if task.oracle.is_some() {
        v.push(Attempt::Oracle);
    }
    v.push(Attempt::Nop);
    for c in &task.controls {
        v.push(Attempt::Control(c.clone()));
    }
    let chosen: Vec<&'static Probe> = match probes {
        ProbeSet::All => probes::ALL.iter().collect(),
        ProbeSet::None => Vec::new(),
        ProbeSet::Only(ids) => ids.iter().filter_map(|id| probes::find(id)).collect(),
    };
    v.extend(chosen.into_iter().map(Attempt::Probe));
    v
}

/// The verdict on one expectation.
#[derive(Debug, Clone, Serialize)]
pub struct Check {
    /// `oracle`, `nop`, `control:<id>`, `probe:<id>`.
    pub attempt: String,
    pub ok: bool,
    /// The trial's reward, when scored.
    pub reward: Option<f64>,
    /// One line: what held or what went wrong.
    pub says: String,
    /// What to do, when it did not hold.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fix: Option<String>,
    pub trial: String,
}

/// The verdict on one task.
#[derive(Debug, Clone, Serialize)]
pub struct TaskVerdict {
    pub task: String,
    pub ok: bool,
    pub checks: Vec<Check>,
    /// Defenses that did not hold in any trial (not supported, off, failed).
    pub weak_defenses: Vec<String>,
    /// The same defenses with their state and what the trial noted, first
    /// trial first.
    pub weak_details: Vec<crate::outcome::Defense>,
}

/// Judge one task's trials.
pub fn judge(task: &Arc<TaskPackage>, records: &[&TrialRecord]) -> TaskVerdict {
    let by = |label: &str| records.iter().find(|r| r.agent == label).copied();
    let mut checks = Vec::new();

    // The empty run's reward is the bar probes must not clear.
    let nop = by("nop");
    let nop_reward = nop.and_then(|r| r.reward);

    if task.oracle.is_some() {
        if let Some(r) = by("oracle") {
            checks.push(match (r.status, r.passed) {
                (Status::Scored, Some(true)) => ok(r, "the reference solution passes"),
                (Status::Scored, _) => {
                    let mut says = format!(
                        "the reference solution scores {}, not a pass: the task is broken, \
                         not the agent, and an agent's 0 on it measures the task",
                        reward_text(r)
                    );
                    if let Some(w) = r.environment_warnings.first() {
                        says.push_str(&format!(" ({w})"));
                    }
                    bad(
                        r,
                        says,
                        "fix oracle/solve.sh, the verifier, or the image (a package gone \
                         upstream, a missing command): a correct answer must pass",
                    )
                }
                _ => broken(r, "the oracle"),
            });
        }
    } else {
        checks.push(Check {
            attempt: "oracle".into(),
            ok: false,
            reward: None,
            says: "the task has no reference solution, so nothing shows it can be solved".into(),
            fix: Some("add oracle/solve.sh (or solution/solve.sh in the split layout)".into()),
            trial: String::new(),
        });
    }

    if let Some(r) = nop {
        checks.push(match (r.status, r.passed) {
            (Status::Scored, Some(false)) => ok(r, &format!("doing nothing fails (reward {})", reward_text(r))),
            (Status::Scored, _) => bad(
                r,
                "doing nothing passes, so the verifier cannot tell a solution from the starting state".into(),
                "make the verifier check what the instruction asks for, not something already true",
            ),
            _ => broken(r, "the empty run"),
        });
    }

    for c in &task.controls {
        let label = format!("control:{}", c.id);
        if let Some(r) = by(&label) {
            let limit_ok = match (r.reward, c.max_reward) {
                (Some(rw), Some(max)) => rw <= max + 1e-9,
                (_, None) => r.passed == Some(false),
                (None, Some(_)) => false,
            };
            checks.push(match r.status {
                Status::Scored if limit_ok => ok(
                    r,
                    &format!("the control is refused (reward {})", reward_text(r)),
                ),
                Status::Scored => bad(
                    r,
                    format!(
                        "the control scores {}, above what it should earn",
                        reward_text(r)
                    ),
                    "tighten the verifier so this known-bad answer fails",
                ),
                _ => broken(r, &format!("control {}", c.id)),
            });
        }
    }

    for p in probes::ALL {
        let label = format!("probe:{}", p.id);
        let Some(r) = by(&label) else { continue };
        let check = match r.status {
            Status::Scored => {
                let above = match (r.reward, nop_reward) {
                    (Some(pr), Some(nr)) => pr > nr + 1e-9,
                    (Some(pr), None) => pr > 0.0,
                    _ => false,
                };
                if above || (r.passed == Some(true) && nop.and_then(|n| n.passed) != Some(true)) {
                    Check {
                        attempt: label,
                        ok: false,
                        reward: r.reward,
                        says: format!(
                            "VULNERABLE: {} raised the reward to {} ({})",
                            p.id,
                            reward_text(r),
                            p.tries
                        ),
                        fix: Some(format!("{}. Defense: {}.", p.fix, p.defense)),
                        trial: r.rollout_name.clone(),
                    }
                } else {
                    ok(r, &format!("{} is refused", p.id))
                }
            }
            Status::Unscored => {
                let hung = r.problem.as_ref().is_some_and(|x| x.code == "R303");
                Check {
                    attempt: label,
                    ok: false,
                    reward: None,
                    says: if hung {
                        format!("{} made grading hang until the verifier timed out", p.id)
                    } else {
                        format!(
                            "{} broke grading: {}",
                            p.id,
                            r.problem
                                .as_ref()
                                .map(|x| x.message.as_str())
                                .unwrap_or("unknown")
                        )
                    },
                    fix: Some(format!("{}. Defense: {}.", p.fix, p.defense)),
                    trial: r.rollout_name.clone(),
                }
            }
            _ => broken(r, &format!("probe {}", p.id)),
        };
        checks.push(check);
    }

    let mut weak: Vec<String> = Vec::new();
    let mut weak_details = Vec::new();
    for r in records {
        for d in r.defenses.values() {
            if d.state != crate::outcome::DefenseState::Enforced && !weak.contains(&d.name) {
                weak.push(d.name.clone());
                weak_details.push(d.clone());
            }
        }
    }
    TaskVerdict {
        task: task.name.clone(),
        ok: checks.iter().all(|c| c.ok),
        checks,
        weak_defenses: weak,
        weak_details,
    }
}

fn reward_text(r: &TrialRecord) -> String {
    r.reward.map(fmt_reward).unwrap_or_else(|| "nothing".into())
}

fn ok(r: &TrialRecord, says: &str) -> Check {
    Check {
        attempt: r.agent.clone(),
        ok: true,
        reward: r.reward,
        says: says.to_string(),
        fix: None,
        trial: r.rollout_name.clone(),
    }
}

fn bad(r: &TrialRecord, says: String, fix: &str) -> Check {
    Check {
        attempt: r.agent.clone(),
        ok: false,
        reward: r.reward,
        says,
        fix: Some(fix.to_string()),
        trial: r.rollout_name.clone(),
    }
}

fn broken(r: &TrialRecord, what: &str) -> Check {
    let (says, fix) = match &r.problem {
        Some(p) => (
            format!("{what} did not finish: {} [{}]", p.message, p.code),
            Some(format!("{} ({})", p.next, p.owner.problem())),
        ),
        None => (format!("{what} did not finish"), None),
    };
    Check {
        attempt: r.agent.clone(),
        ok: false,
        reward: r.reward,
        says,
        fix,
        trial: r.rollout_name.clone(),
    }
}
