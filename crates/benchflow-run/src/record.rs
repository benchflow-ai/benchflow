//! What a trial leaves on disk, in BenchFlow 0.8's layout, so `bf.load_job`,
//! the viewer, and existing scripts read Rust runs unchanged.
//!
//! ```text
//! <task>__<id>/
//!   result.json  config.json  timing.json  prompts.json  custody.json
//!   trajectory/acp_trajectory.jsonl
//!   agent/       oracle.txt, stdout and stderr of the agent
//!   verifier/    reward.txt, reward.json, ctrf.json, test-stdout.txt, review.json
//!   outputs/     saved outputs, when asked for
//!   agent/sessions/  the agent's own session files, laid out as in its home
//! ```

use std::collections::BTreeMap;
use std::path::Path;

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::Usage;
use crate::config::TrialDataset;
use crate::outcome::{DefenseState, Defenses, Phase, Problem, Status};

/// Seconds spent in each phase. The first four keys are 0.8's.
#[derive(Debug, Clone, Default, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Timing {
    pub environment_setup: f64,
    pub agent_execution: f64,
    pub verifier: f64,
    pub total: f64,
    #[serde(default)]
    pub image: f64,
    /// Inside `environment_setup`: starting the sandbox, then setting up the
    /// agent user and hidden folders.
    #[serde(default)]
    pub sandbox_start: f64,
    #[serde(default)]
    pub setup: f64,
    /// Inside `verifier`: the handoff (kill, clean, custody), then grading.
    #[serde(default)]
    pub handoff: f64,
    #[serde(default)]
    pub verify: f64,
}

/// What the runner removed or refused at the handoff.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Custody {
    /// FIFOs, sockets, and devices removed from the workdir.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub special_files: Vec<String>,
    /// Symlinks in the workdir that pointed outside it, or through `/proc`
    /// or `/dev/fd`, removed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub escaping_symlinks: Vec<String>,
    /// Interpreter links of the workdir's virtual environments, made again
    /// by root from the record taken before the agent's first turn.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repaired_links: Vec<String>,
    /// Test-runner files the agent planted, removed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub planted: Vec<String>,
    /// Test-runner files the agent edited, restored.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub restored: Vec<String>,
    /// Python files left in /tmp or /var/tmp, removed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tmp_modules: Vec<String>,
    /// Files over 1 GB in the workdir (reported, not removed).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub large_files: Vec<String>,
    /// Agent processes killed before grading.
    #[serde(default)]
    pub killed_processes: u64,
    /// Who owned the workdir while it was graded: root, or the task's
    /// verifier user, given it after the handoff (as BenchFlow 0.8 does).
    /// Absent when the agent ran as root or the change did not report back.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workdir_owner: Option<String>,
    /// Paths in the workdir that kept their owner when it changed hands
    /// (`chown` or `find` refused them: a path nested past what the image's
    /// `find` can walk, a file on a read-only bind mount of the workdir's
    /// own filesystem; at most 5 errors per change), or a change that ran
    /// past its time. The workdir folder itself always changed. Another
    /// filesystem mounted inside the workdir is not walked at all (`-xdev`)
    /// and is not listed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub workdir_unchanged: Vec<String>,
    /// Hidden folders the image shipped, locked before the agent ran.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locked: Vec<String>,
    /// Outputs not saved because they were links or special files.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped_special: Vec<String>,
    /// Outputs not saved because they were over the size cap.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skipped_large: Vec<String>,
    /// Files and bytes saved from the outputs.
    #[serde(default)]
    pub saved_files: u64,
    #[serde(default)]
    pub saved_bytes: u64,
    /// Credential files the task's image holds, found before the agent
    /// started (`bench test` scans; paths and kinds only, never content).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub credentials: Vec<crate::credentials::Finding>,
}

impl Custody {
    /// Anything the agent left that a careful grader would want to know.
    pub fn is_clean(&self) -> bool {
        self.special_files.is_empty()
            && self.escaping_symlinks.is_empty()
            && self.planted.is_empty()
            && self.restored.is_empty()
            && self.tmp_modules.is_empty()
    }
}

/// result.json: BenchFlow 0.8's fields, then the Rust engine's.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ResultFile {
    pub task_name: String,
    /// The trial's folder name, `<task>__<8 hex>`.
    pub rollout_name: String,
    /// The verifier's values; `reward` is the headline.
    pub rewards: Option<BTreeMap<String, f64>>,
    pub agent: String,
    /// The same as `agent` (0.8 wrote both).
    pub agent_name: String,
    pub model: Option<String>,
    pub skill_mode: String,
    pub n_tool_calls: u64,
    pub n_prompts: usize,
    pub agent_result: AgentResult,
    pub trajectory_summary: Value,
    /// Where the trajectory came from (`acp`, `scripted`, ...). 0.8 reads it
    /// at the top level: a trial without an ACP trajectory is then not
    /// judged an agent that did nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_source: Option<Value>,
    /// Set when the trial errored: what went wrong, with its code.
    pub error: Option<String>,
    pub error_category: Option<String>,
    /// Set when the verifier could not score the trial.
    pub verifier_error: Option<String>,
    pub verifier_error_category: Option<String>,
    /// Local time, `YYYY-MM-DD HH:MM:SS.ffffff`.
    pub started_at: String,
    pub finished_at: String,
    pub timing: Timing,
    /// `sha256:<hex>` over the task's files.
    pub task_digest: String,
    pub sandbox_id: Option<String>,
    pub task_id: String,
    pub status: Status,
    pub passed: Option<bool>,
    pub timed_out: bool,
    pub problem: Option<Problem>,
    /// Each defense and whether it held, by name.
    pub defenses: Defenses,
    pub login: Option<String>,
    /// The agent's version, as the agent reported it (ACP's `agentInfo`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_version: Option<String>,
    pub verifier_exit_code: Option<i32>,
    pub rubric: Option<Value>,
    pub engine: Engine,
    /// On dataset runs (`bench run -d`): the dataset and the digest its
    /// registry pins for the task.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry_digest: Option<String>,
    /// A task graded in stages: each stage's result (`step_name`,
    /// `verifier_result.rewards`), with its gate's outcome.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_results: Option<Value>,
    /// The agent's own session files, saved at the end of its turn for
    /// `bench handoff`; absent when the agent keeps no session BenchFlow
    /// can resume, or its session folders were empty.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_session: Option<AgentSession>,
    /// The agent's manifest raised the sandbox's size to the least the agent
    /// needs (its `[resources]`), above what the task asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources_raised: Option<ResourcesRaised>,
    /// Signs in the verifier's output or the agent's log that the task's
    /// environment failed (a package gone upstream, git refusing the
    /// repository), on a trial that scored without passing. The score
    /// stands; these say it may measure the task rather than the agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub environment_warnings: Vec<String>,
    /// The task's declared `[verifier] reward_range`, `[lo, hi]`, which
    /// widens [0, 1]; absent when it declares none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward_range: Option<(f64, f64)>,
}

/// What an agent's resource floor changed: for each resource it raised,
/// what the task asked for and what the sandbox got.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct ResourcesRaised {
    /// The agent whose manifest has the floor.
    pub agent: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_mb: Option<Raised>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpus: Option<Raised>,
}

/// One resource raised to an agent's floor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Raised {
    /// What the task asked for; absent when it left the size to the
    /// sandbox backend's default.
    pub task: Option<f64>,
    /// What the sandbox was asked for instead.
    pub sandbox: f64,
}

/// The agent's own session, saved into the trial folder (result.json's
/// `agent_session`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AgentSession {
    /// Where in the trial folder: `agent/sessions`, laid out as in the
    /// agent's home (`agent/sessions/.claude/projects/...`).
    pub path: String,
    /// The session folders saved, relative to the agent's home.
    pub dirs: Vec<String>,
    pub files: u64,
    pub bytes: u64,
    /// The session id the harness reported, when it reported one (ACP).
    pub session_id: Option<String>,
    /// Files not saved: login files, links and special files, and files
    /// over the size cap (paths relative to the agent's home).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub left_out: Vec<String>,
}

/// Where a trial keeps the agent's session files.
pub const SESSIONS_DIR: &str = "agent/sessions";

/// Usage counts, as 0.8's `agent_result`.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct AgentResult {
    pub n_tool_calls: u64,
    pub n_prompts: usize,
    pub n_input_tokens: u64,
    pub n_output_tokens: u64,
    pub n_cache_read_tokens: u64,
    pub n_cache_creation_tokens: u64,
    pub total_tokens: u64,
    pub cost_usd: Option<f64>,
    /// Where the numbers came from: `acp`, `headless`, `native`,
    /// `forwarder` (the provider's answers, read by BenchFlow's forwarder),
    /// or `unavailable`.
    pub usage_source: String,
    /// Who priced `cost_usd`: `provider` (what the provider billed, as its
    /// answers said), or `agent` (the agent's own figure, for a run that
    /// did not go through BenchFlow's forwarder). Absent when no cost is
    /// recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_source: Option<String>,
    /// Why no cost is recorded, when tokens were used.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_note: Option<String>,
}

/// The engine that wrote a file.
#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Engine {
    pub name: String,
    pub version: String,
}

impl Engine {
    pub fn current() -> Self {
        Self {
            name: "benchflow-rs".into(),
            version: env!("CARGO_PKG_VERSION").into(),
        }
    }
}

/// A finished (or failed) trial.
#[derive(Debug, Clone)]
pub struct TrialRecord {
    pub task_name: String,
    pub task_id: String,
    pub rollout_name: String,
    pub agent: String,
    pub model: Option<String>,
    pub status: Status,
    pub rewards: Option<BTreeMap<String, f64>>,
    pub reward: Option<f64>,
    pub passed: Option<bool>,
    pub timed_out: bool,
    /// The agent of this trial itself ran out of time. Differs from
    /// `timed_out` only in a branch child of a stage fork, which carries
    /// the parent's time-out (for `on_timeout = "fail"`) but may not have
    /// timed out in its own stages. Not in result.json.
    pub own_timed_out: bool,
    pub problem: Option<Problem>,
    pub usage: Usage,
    pub tool_calls: u64,
    pub prompts: Vec<String>,
    pub trajectory_summary: Value,
    pub timing: Timing,
    pub started_at: DateTime<Local>,
    pub finished_at: DateTime<Local>,
    pub task_digest: String,
    pub sandbox_id: Option<String>,
    pub sandbox_backend: String,
    pub defenses: Defenses,
    pub custody: Custody,
    pub login: Option<String>,
    /// The agent's version, when it reported one.
    pub agent_version: Option<String>,
    pub env_keys: Vec<String>,
    /// The verifier's exit code, when it ran.
    pub verifier_exit: Option<i32>,
    /// task.md rubric scores, when the task has a rubric.
    pub rubric: Option<Value>,
    /// The dataset the task came from (`bench run -d`).
    pub dataset: Option<TrialDataset>,
    /// Each stage's result, for a task graded in stages.
    pub step_results: Option<Value>,
    /// The agent's own session files, when the trial saved them.
    pub agent_session: Option<AgentSession>,
    /// The session the agent resumed (ACP `session/load` succeeded), when
    /// it was asked to resume one; kept even when its turn then failed.
    /// Not in result.json.
    pub resumed_session: Option<String>,
    /// What the agent's resource floor raised.
    pub resources_raised: Option<ResourcesRaised>,
    /// Signs that the task's environment failed (see [`crate::envcheck`]).
    pub environment_warnings: Vec<String>,
    /// The task's declared `[verifier] reward_range`, if any.
    pub reward_range: Option<(f64, f64)>,
}

fn ts(t: &DateTime<Local>) -> String {
    t.format("%Y-%m-%d %H:%M:%S%.6f").to_string()
}

/// How a trial the user stopped (Ctrl-C, R001) reads everywhere: the run's
/// summary and `bench results` say it in these words, and leave such trials
/// out of the pass rate.
pub const STOPPED_WORDS: &str =
    "stopped (Ctrl-C, a kill, or a closed terminal), not counted in the pass rate";

/// What a job's trials add up to, as every score shows it (the run's
/// summary, `bench results` in text and JSON, `--fail-under`): trials the
/// user stopped are counted apart, never in the pass rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Tally {
    /// Trials in the pass rate: every trial but the stopped ones.
    pub counted: usize,
    pub passed: usize,
    pub scored: usize,
    pub stopped: usize,
    /// Scored trials graded although `locked_system_dirs` failed: see
    /// [`TrialRecord::graded_unlocked`].
    pub graded_unlocked: usize,
}

impl Tally {
    pub fn of<'a>(records: impl IntoIterator<Item = &'a TrialRecord>) -> Self {
        let mut t = Tally::default();
        for r in records {
            if r.stopped() {
                t.stopped += 1;
                continue;
            }
            t.counted += 1;
            t.passed += usize::from(r.passed == Some(true));
            t.scored += usize::from(r.status == Status::Scored);
            t.graded_unlocked += usize::from(r.graded_unlocked());
        }
        t
    }
    /// The pass rate, `None` with nothing counted.
    pub fn rate(&self) -> Option<f64> {
        (self.counted > 0).then(|| self.passed as f64 / self.counted as f64)
    }
}

impl TrialRecord {
    /// Whether this trial was graded although `locked_system_dirs` failed,
    /// so the system folders the verifier runs from may have been the
    /// agent's to change. A repair before the agent ran (the image left
    /// them writable, and the lock removed that) does not count: the folders
    /// were locked.
    pub fn graded_unlocked(&self) -> bool {
        self.status == Status::Scored
            && self.defenses.get("locked_system_dirs").is_some_and(|d| {
                d.state == DefenseState::Failed
                    && !d
                        .note
                        .as_deref()
                        .is_some_and(|n| n.starts_with(crate::confine::SYSTEM_DIRS_REPAIRED))
            })
    }

    /// Whether a model agent ran in this trial with the sandbox daemon's
    /// ports open to it: on Daytona, the guard (`root_service`) could not be
    /// installed or proven, and the run went ahead as decided, the record
    /// saying why. Not when the agent runs as root (`off`, the task's
    /// choice) or never started (see [`Self::agent_started`]).
    pub fn model_ran_unguarded(&self) -> bool {
        // By the agent's kind, not its recorded model: an ACP agent on its
        // default model may report none.
        let scripted = matches!(self.agent.as_str(), "oracle" | "nop")
            || ["script:", "probe:", "control:"]
                .iter()
                .any(|p| self.agent.starts_with(p));
        !scripted
            && self.agent_started()
            && self.defenses.get("root_service").is_some_and(|d| {
                d.state == DefenseState::NotSupported
                    && d.note
                        .as_deref()
                        .is_some_and(|n| n.contains(crate::netpolicy::UNGUARDED))
            })
    }

    /// Whether the task's disk limit was not enforced on this trial's
    /// sandbox (decision 22: said once per job), with the reason.
    pub fn disk_not_enforced(&self) -> Option<&str> {
        self.defenses
            .get("disk_limit")
            .filter(|d| d.state == DefenseState::NotSupported)
            .map(|d| d.note.as_deref().unwrap_or("disk limit not enforced"))
    }

    /// Whether the disk limit's watch could not count the layer's entries
    /// on this trial (only its bytes were watched: said once per job).
    pub fn disk_entries_unwatched(&self) -> bool {
        self.defenses
            .get("disk_limit")
            .and_then(|d| d.note.as_deref())
            .is_some_and(|n| n.contains("so only its bytes were watched"))
    }

    /// Why a model agent in this trial could reach the cloud metadata
    /// endpoints, from its `cloud_metadata` record: the task opted out, the
    /// refusal could not be installed or shown to hold, or the agent ran as
    /// root. `None` when they were refused, for scripted attempts, for an
    /// agent that never started, and for records from before the defense
    /// existed.
    pub fn model_reached_metadata(&self) -> Option<&str> {
        let scripted = matches!(self.agent.as_str(), "oracle" | "nop")
            || ["script:", "probe:", "control:"]
                .iter()
                .any(|p| self.agent.starts_with(p));
        if scripted || !self.agent_started() {
            return None;
        }
        let d = self.defenses.get(crate::netpolicy::METADATA_DEFENSE)?;
        match d.state {
            DefenseState::Off | DefenseState::NotSupported => {
                Some(d.note.as_deref().unwrap_or("not refused"))
            }
            _ => None,
        }
    }

    /// Whether the agent's phase began: the agent had time, or no problem
    /// stopped the trial before its phase (the image, the sandbox's start,
    /// or its setup, the network's own install included). A later stage's
    /// setup problem in a multi-stage task comes after an agent that ran,
    /// which its time shows.
    pub fn agent_started(&self) -> bool {
        self.timing.agent_execution > 0.0
            || !self
                .problem
                .as_ref()
                .is_some_and(|p| matches!(p.phase, Phase::Image | Phase::Start | Phase::Setup))
    }

    /// The reward when it is outside the range its task allows: the
    /// declared `[verifier] reward_range` when the record names one, else
    /// [0, 1]. bench refuses such a reward when it grades, so this finds
    /// rewards in job folders it did not grade (BenchFlow 0.8's for a task
    /// whose range the record does not carry, an older bench's). They are
    /// counted as written and named, so they do not skew a mean silently.
    pub fn reward_out_of_range(&self) -> Option<f64> {
        let (lo, hi) = self.reward_range.unwrap_or((0.0, 1.0));
        self.reward.filter(|r| !(lo..=hi).contains(r))
    }

    /// Whether the user stopped this trial before it finished (R001): it
    /// says nothing about the agent, so pass rates leave it out.
    pub fn stopped(&self) -> bool {
        self.status == Status::Skipped && self.problem.as_ref().is_some_and(|p| p.code == "R001")
    }

    /// result.json, in 0.8's shape plus the Rust engine's fields.
    pub fn result_json(&self) -> Value {
        serde_json::to_value(self.result_file()).unwrap_or(Value::Null)
    }

    /// result.json as a typed value.
    pub fn result_file(&self) -> ResultFile {
        let (error, error_category, verifier_error, verifier_error_category) =
            match (&self.status, &self.problem) {
                (Status::Errored | Status::Skipped, Some(p)) => {
                    (Some(p.to_string()), Some(category(p)), None, None)
                }
                (Status::Unscored, Some(p)) => (None, None, Some(p.to_string()), Some(category(p))),
                _ => (None, None, None, None),
            };
        ResultFile {
            task_name: self.task_name.clone(),
            rollout_name: self.rollout_name.clone(),
            rewards: self.rewards.clone(),
            agent: self.agent.clone(),
            agent_name: self.agent.clone(),
            model: self.model.clone(),
            skill_mode: "no-skill".into(),
            n_tool_calls: self.tool_calls,
            n_prompts: self.prompts.len(),
            agent_result: AgentResult {
                n_tool_calls: self.tool_calls,
                n_prompts: self.prompts.len(),
                n_input_tokens: self.usage.input_tokens,
                n_output_tokens: self.usage.output_tokens,
                n_cache_read_tokens: self.usage.cache_read_tokens,
                n_cache_creation_tokens: self.usage.cache_write_tokens,
                total_tokens: self.usage.total(),
                cost_usd: self.usage.cost_usd,
                cost_source: self.usage.cost_source.clone(),
                cost_note: self.usage.cost_note.clone(),
                usage_source: if self.usage.source.is_empty() {
                    "unavailable".into()
                } else {
                    self.usage.source.clone()
                },
            },
            trajectory_summary: self.trajectory_summary.clone(),
            trajectory_source: self.trajectory_summary.get("trajectory_source").cloned(),
            error,
            error_category,
            verifier_error,
            verifier_error_category,
            started_at: ts(&self.started_at),
            finished_at: ts(&self.finished_at),
            timing: self.timing.clone(),
            task_digest: self.task_digest.clone(),
            sandbox_id: self.sandbox_id.clone(),
            task_id: self.task_id.clone(),
            status: self.status,
            passed: self.passed,
            timed_out: self.timed_out,
            problem: self.problem.clone(),
            defenses: self.defenses.clone(),
            login: self.login.clone(),
            agent_version: self.agent_version.clone(),
            verifier_exit_code: self.verifier_exit,
            rubric: self.rubric.clone(),
            engine: Engine::current(),
            dataset_name: self.dataset.as_ref().map(|d| d.name.clone()),
            dataset_version: self.dataset.as_ref().map(|d| d.version.clone()),
            registry_digest: self
                .dataset
                .as_ref()
                .and_then(|d| d.registry_digest.clone()),
            step_results: self.step_results.clone(),
            agent_session: self.agent_session.clone(),
            resources_raised: self.resources_raised.clone(),
            environment_warnings: self.environment_warnings.clone(),
            reward_range: self.reward_range,
        }
    }

    /// One results.jsonl row (0.8's Verifiers-style row, abbreviated).
    pub fn results_row(&self) -> Value {
        let mut metrics = serde_json::Map::new();
        if let Some(r) = &self.rewards {
            for (k, v) in r {
                metrics.insert(k.clone(), json!(v));
            }
        }
        json!({
            "task": self.task_name,
            "rollout_name": self.rollout_name,
            "agent": self.agent,
            "model": self.model,
            "reward": self.reward,
            "passed": self.passed,
            "metrics": metrics,
            "status": self.status,
            "error": self.problem.as_ref().map(|p| json!({
                "error": if self.status == Status::Unscored { "verifier_error" } else { "agent_error" },
                "error_chain_str": p.to_string(),
                "owner": p.owner,
            })),
            "token_usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
            "timing": self.timing,
            "info": { "schema_version": 2, "engine": "benchflow-rs" },
        })
    }

    /// One line for reports: the failure, or how the reward was earned.
    pub fn reason(&self) -> Option<String> {
        if let Some(p) = &self.problem {
            return Some(p.message.clone());
        }
        if self.passed == Some(false)
            && let Some(r) = self.reward
        {
            let mut s = format!("reward {}", fmt_reward(r));
            if self.timed_out {
                s.push_str(" (timed out)");
            }
            return Some(s);
        }
        None
    }

    /// Write the trial's record files, result.json last: its rename marks
    /// the trial finished, so a record is never read back with parts
    /// missing. Each write is tried three times (a full disk may free up).
    /// On failure the error names the file; nothing is deleted.
    pub fn write(&self, dir: &Path, config: &Value) -> Result<(), RecordWriteError> {
        let files: [(&'static str, Value); 5] = [
            ("config.json", config.clone()),
            (
                "timing.json",
                serde_json::to_value(&self.timing).unwrap_or_default(),
            ),
            ("prompts.json", json!(self.prompts)),
            (
                "custody.json",
                serde_json::to_value(&self.custody).unwrap_or_default(),
            ),
            ("result.json", self.result_json()),
        ];
        std::fs::create_dir_all(dir).map_err(|error| RecordWriteError {
            file: dir.display().to_string(),
            error,
        })?;
        for (name, value) in &files {
            let path = dir.join(name);
            let mut tries = 0;
            loop {
                match write_json(&path, value) {
                    Ok(()) => break,
                    Err(_) if tries < 2 => {
                        tries += 1;
                        std::thread::sleep(std::time::Duration::from_millis(250));
                    }
                    Err(error) => {
                        let _ = std::fs::remove_file(path.with_extension("json.tmp"));
                        return Err(RecordWriteError {
                            file: (*name).to_string(),
                            error,
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

/// A trial record file that could not be written.
#[derive(Debug)]
pub struct RecordWriteError {
    /// The file (or the trial folder) that failed.
    pub file: String,
    pub error: std::io::Error,
}

impl std::fmt::Display for RecordWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.file, self.error)
    }
}

/// 0.8's error categories, as near as the Rust failure model maps onto them.
fn category(p: &Problem) -> String {
    match p.code.as_str() {
        "R101" | "R102" | "R103" => "sandbox_startup".into(),
        "R201" => "agent_install".into(),
        "R202" | "R204" => "acp".into(),
        "R203" => "idle_timeout".into(),
        "R205" => "acp_error".into(),
        // BenchFlow 0.8's names: an agent that never acted before its time
        // ran out (3842ed22), or that cannot do what the run asked (a model
        // it does not offer), is an integration failure, not BenchFlow's.
        "R206" | "R215" => "agent_integration".into(),
        "R214" => "api_error".into(),
        "R211" | "R212" => "provider_auth".into(),
        "R213" => "usage_limit".into(),
        "R106" => "sandbox_startup".into(),
        "R301" => "reward_missing".into(),
        "R302" => "reward_parse".into(),
        "R303" => "verifier_timeout".into(),
        "R304" => "verifier_error".into(),
        "R305" | "R307" | "R314" | "R315" => "verifier_hardening".into(),
        "R311" => "verifier_error".into(),
        "R401" => "unsupported_task_feature".into(),
        _ => "internal".into(),
    }
}

/// A reward for people: `1`, `0`, `0.75`. Three places at most, but a
/// reward that is not exactly 0 or 1 never shows as `0` or `1` (0.9996
/// shows as 0.9996, not as a pass).
pub fn fmt_reward(r: f64) -> String {
    if r.fract() == 0.0 {
        format!("{}", r as i64)
    } else {
        let s = format!("{r:.3}");
        let s = s.trim_end_matches('0').trim_end_matches('.');
        if matches!(s, "0" | "1" | "-0" | "-1") {
            // Display never uses an exponent: 5e-324 took 326 characters.
            if r.abs() < 1e-4 {
                format!("{r:e}")
            } else {
                format!("{r}")
            }
        } else {
            s.to_string()
        }
    }
}

pub fn write_json(path: &Path, value: &Value) -> std::io::Result<()> {
    let mut text = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    text.push('\n');
    write_atomic(path, text.as_bytes())
}

/// Write `path` through a temporary file beside it, so a reader (or a run
/// resumed after a crash or a full disk) sees the old file or the new one,
/// never part of one. The temporary file reaches the disk before the
/// rename: a file system that reports a full disk only then (NFS) fails
/// the write rather than replace the old file with part of the new one.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    let tmp = path.with_file_name(name);
    let written = std::fs::File::create(&tmp).and_then(|mut f| {
        f.write_all(bytes)?;
        sync(&f)
    });
    if let Err(e) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path)?;
    // The rename reaches the disk too, so a power loss cannot bring the
    // old file back. A folder some file systems cannot sync is no failure.
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty())
        && let Ok(dir) = std::fs::File::open(parent)
        && let Err(e) = sync(&dir)
        && !unsyncable(&e)
    {
        return Err(e);
    }
    Ok(())
}

/// Flush `file` to the disk. On macOS `sync_all` is `F_FULLFSYNC` only,
/// which file systems other than APFS and HFS+ (SMB mounts, for one)
/// refuse; those get a plain `fsync`, as SQLite and Go do.
fn sync(file: &std::fs::File) -> std::io::Result<()> {
    sync_with(file.sync_all(), || plain_fsync(file))
}

/// `full`, or when the file system refuses it, `plain`.
fn sync_with(
    full: std::io::Result<()>,
    plain: impl FnOnce() -> std::io::Result<()>,
) -> std::io::Result<()> {
    match full {
        Err(e) if unsyncable(&e) => plain(),
        other => other,
    }
}

/// An error that says the file system cannot do this kind of sync.
fn unsyncable(e: &std::io::Error) -> bool {
    #[cfg(unix)]
    {
        // A list, not a pattern: on Linux ENOTSUP and EOPNOTSUPP are one
        // value, and a pattern naming both is unreachable there, an error
        // under `clippy -D warnings`.
        e.raw_os_error().is_some_and(|code| {
            [libc::ENOTSUP, libc::ENOTTY, libc::EINVAL, libc::EOPNOTSUPP].contains(&code)
        })
    }
    #[cfg(not(unix))]
    {
        let _ = e;
        false
    }
}

#[cfg(unix)]
fn plain_fsync(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: fsync on a descriptor this `File` owns and keeps open.
    #[allow(unsafe_code)]
    let rc = unsafe { libc::fsync(file.as_raw_fd()) };
    if rc == 0 {
        Ok(())
    } else {
        let e = std::io::Error::last_os_error();
        // A folder or a file system that cannot be synced at all.
        if unsyncable(&e) { Ok(()) } else { Err(e) }
    }
}

#[cfg(not(unix))]
fn plain_fsync(file: &std::fs::File) -> std::io::Result<()> {
    file.sync_all()
}

#[cfg(test)]
mod tests {
    /// The rate every score uses (`--fail-under` included) leaves stopped
    /// trials out: 3 passed and 2 stopped is 1.0, not 0.6.
    #[test]
    fn the_tally_rate_leaves_stopped_trials_out() {
        let rec = |v: serde_json::Value| {
            crate::read::record_from_json(&v, std::path::Path::new("t")).unwrap()
        };
        let mut recs: Vec<TrialRecord> = (0..3)
            .map(|_| rec(json!({"status": "scored", "rewards": {"reward": 1.0}, "passed": true})))
            .collect();
        for _ in 0..2 {
            recs.push(rec(json!({"status": "skipped", "problem": {
                "code": "R001", "owner": "you", "phase": "agent",
                "message": "stopped before it finished", "next": "rerun", "transient": false}})));
        }
        let t = Tally::of(&recs);
        assert_eq!((t.counted, t.passed, t.stopped), (3, 3, 2));
        assert_eq!(t.rate(), Some(1.0));
        assert_eq!(Tally::of(&recs[3..]).rate(), None);
    }

    /// F1 round 2 (F1-B 3): a model agent whose trial stopped before its
    /// phase (here at setup, after the network was enforced) is not counted
    /// as one that could reach the cloud metadata endpoints, nor as one that
    /// ran without the daemon guard; one that ran, or whose later stage's
    /// setup failed after it ran, is.
    #[test]
    fn an_agent_that_never_started_reached_nothing() {
        let rec = |phase: Option<&str>, agent_secs: f64| {
            let mut v = json!({"agent": "opencode", "status": "scored",
            "timing": {"environment_setup": 1.0, "agent_execution": agent_secs, "verifier": 0.0, "total": 1.0},
            "defenses": {
                "cloud_metadata": {"name": "cloud_metadata", "state": "off", "note": "the task opts out, so the agent can reach them"},
                "root_service": {"name": "root_service", "state": "not-supported", "note": format!("{}: no iptables", crate::netpolicy::UNGUARDED)},
            }});
            if let Some(phase) = phase {
                v["status"] = json!("errored");
                v["problem"] = json!({"code": "R104", "owner": "task", "phase": phase,
                    "message": "m", "next": "n", "transient": false});
            }
            crate::read::record_from_json(&v, std::path::Path::new("t")).unwrap()
        };
        // F2's R110 when the disk limit's watch cannot measure the layer
        // before the first turn: the agent never ran (review
        // F2-integrate-A A-1).
        let mut unwatched = rec(Some("setup"), 0.0);
        if let Some(p) = unwatched.problem.as_mut() {
            p.code = "R110".into();
            p.owner = crate::outcome::Owner::Sandbox;
        }
        for never in [rec(Some("setup"), 0.0), rec(Some("start"), 0.0), unwatched] {
            assert!(!never.agent_started());
            assert_eq!(never.model_reached_metadata(), None);
            assert!(!never.model_ran_unguarded());
        }
        for ran in [
            rec(None, 0.0),
            rec(Some("agent"), 0.0),
            rec(Some("setup"), 3.0),
        ] {
            assert!(ran.agent_started());
            assert!(ran.model_reached_metadata().is_some());
            assert!(ran.model_ran_unguarded());
        }
    }

    use super::*;

    /// D4 chaos runs: results.jsonl and the trainer files were written in
    /// place, so a crash or a full disk mid-write left part of a file.
    #[test]
    fn a_failed_write_leaves_the_old_file_whole() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("results.jsonl");
        write_atomic(&path, b"{\"a\":1}\n").unwrap();
        // The temporary file cannot be made: the write fails, the old
        // file stays as it was, and nothing is left behind.
        std::fs::create_dir(dir.path().join("results.jsonl.tmp")).unwrap();
        assert!(write_atomic(&path, b"{\"a\":2}\n").is_err());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"a\":1}\n");
        std::fs::remove_dir(dir.path().join("results.jsonl.tmp")).unwrap();
        write_atomic(&path, b"{\"a\":3}\n").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "{\"a\":3}\n");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    /// Review of rust/d4-chaos (Outcomes 5): on macOS `sync_all` is
    /// `F_FULLFSYNC` with no fallback, so every job file failed on an SMB
    /// mount. A refusal of the full sync falls back to a plain one; other
    /// errors stay errors.
    #[test]
    fn a_refused_full_sync_falls_back_to_a_plain_one() {
        let refused = || Err(std::io::Error::from_raw_os_error(libc::ENOTSUP));
        let mut plain_ran = false;
        sync_with(refused(), || {
            plain_ran = true;
            Ok(())
        })
        .unwrap();
        assert!(plain_ran);
        let full = || Err(std::io::Error::from_raw_os_error(libc::EIO));
        assert!(sync_with(full(), || Ok(())).is_err());
        // The atomic write syncs its folder too, and still works.
        let dir = tempfile::tempdir().unwrap();
        write_atomic(&dir.path().join("x.json"), b"{}").unwrap();
        assert_eq!(std::fs::read(dir.path().join("x.json")).unwrap(), b"{}");
    }

    /// Guards the D5 fix: rounding to three places showed a failed 0.9996
    /// as `1` (and 0.0004 as `0`) in the live view and in results.
    #[test]
    fn a_reward_near_0_or_1_does_not_show_as_0_or_1() {
        assert_eq!(fmt_reward(1.0), "1");
        assert_eq!(fmt_reward(0.0), "0");
        assert_eq!(fmt_reward(0.75), "0.75");
        assert_eq!(fmt_reward(2.0 / 3.0), "0.667");
        assert_eq!(fmt_reward(0.9996), "0.9996");
        assert_eq!(fmt_reward(0.0004), "0.0004");
        assert_eq!(fmt_reward(5e-324), "5e-324");
        assert_eq!(fmt_reward(0.00004), "4e-5");
        assert_eq!(fmt_reward(1.0 - 1e-12), "0.999999999999");
    }

    /// `bench run -a codex -m <a model Codex does not offer>` wrote
    /// `error_category: "internal"`, which reads as a BenchFlow bug.
    #[test]
    fn agent_failures_are_not_filed_as_internal() {
        for (code, want) in [
            ("R205", "acp_error"),
            ("R206", "agent_integration"),
            ("R214", "api_error"),
            ("R215", "agent_integration"),
        ] {
            let p = Problem::new(
                code,
                crate::outcome::Owner::You,
                crate::outcome::Phase::Agent,
                "m",
                "n",
            );
            assert_eq!(category(&p), want, "{code}");
        }
    }

    #[test]
    fn rewards_format_short() {
        assert_eq!(fmt_reward(1.0), "1");
        assert_eq!(fmt_reward(0.0), "0");
        assert_eq!(fmt_reward(0.75), "0.75");
        assert_eq!(fmt_reward(1.0 / 3.0), "0.333");
    }
}

#[cfg(test)]
mod category_tests {
    use crate::outcome::{Owner, Phase, Problem};

    /// The handoff's refusals read as hardening, not as a BenchFlow bug
    /// (REVIEW-OUTCOMES (d7c) L6).
    #[test]
    fn the_handoffs_refusals_are_hardening() {
        for code in ["R305", "R307", "R314", "R315"] {
            let p = Problem::new(code, Owner::Task, Phase::Handoff, "m", "n");
            assert_eq!(super::category(&p), "verifier_hardening", "{code}");
        }
    }
}
