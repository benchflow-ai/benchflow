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
use crate::outcome::{Defenses, Problem, Status};

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
    /// Symlinks in the workdir that pointed outside it, removed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub escaping_symlinks: Vec<String>,
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
    /// A task graded in stages: each stage's result, in Harbor's shape
    /// (`step_name`, `verifier_result.rewards`), with its gate's outcome.
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
    /// What the agent's resource floor raised.
    pub resources_raised: Option<ResourcesRaised>,
    /// Signs that the task's environment failed (see [`crate::envcheck`]).
    pub environment_warnings: Vec<String>,
}

fn ts(t: &DateTime<Local>) -> String {
    t.format("%Y-%m-%d %H:%M:%S%.6f").to_string()
}

impl TrialRecord {
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

    pub fn write(&self, dir: &Path, config: &Value) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        write_json(&dir.join("result.json"), &self.result_json())?;
        write_json(&dir.join("config.json"), config)?;
        write_json(
            &dir.join("timing.json"),
            &serde_json::to_value(&self.timing).unwrap_or_default(),
        )?;
        write_json(&dir.join("prompts.json"), &json!(self.prompts))?;
        write_json(
            &dir.join("custody.json"),
            &serde_json::to_value(&self.custody).unwrap_or_default(),
        )?;
        Ok(())
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
        "R305" | "R307" => "verifier_hardening".into(),
        "R311" => "verifier_error".into(),
        "R401" => "unsupported_task_feature".into(),
        _ => "internal".into(),
    }
}

/// A reward for people: `1`, `0`, `0.75`.
pub fn fmt_reward(r: f64) -> String {
    if r.fract() == 0.0 {
        format!("{}", r as i64)
    } else {
        let s = format!("{r:.3}");
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn write_json(path: &Path, value: &Value) -> std::io::Result<()> {
    let mut text = serde_json::to_string_pretty(value).map_err(std::io::Error::other)?;
    text.push('\n');
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

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
