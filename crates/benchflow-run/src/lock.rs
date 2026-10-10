//! `lock.json`: everything needed to rerun exactly what a job ran.
//!
//! Every job folder gets one, written before the first trial starts: the
//! `bench` build and its git commit, the resolved run options, the dataset
//! (when the tasks came from a registry), and per trial the task (id, path,
//! digest, format), the attempt (agent, model), the image, and the sandbox
//! backend, and the job files read (`-c`, paths or URLs) with their digests.
//! Environment variable values are never written, only their names.

use std::collections::HashMap;
use std::path::PathBuf;

use benchflow_sandbox::ImageSource;
use benchflow_task::TaskPackage;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::agent::Attempt;
use crate::config::RunOptions;
use crate::images;
use crate::trial::TrialSpec;

pub const FILE_NAME: &str = "lock.json";

/// The current shape of lock.json.
pub const SCHEMA_VERSION: u32 = 1;

/// A job's lock file.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Lock {
    pub schema_version: u32,
    pub job_name: String,
    /// When the lock was written (local time, RFC 3339).
    pub created_at: String,
    pub bench: BenchBuild,
    /// What decides whether a later run resumes this job.
    pub fingerprint: String,
    /// The dataset the tasks came from, for `bench run -d`.
    pub dataset: Option<LockDataset>,
    /// The targets and fetched sources, as evaluation.json records them.
    pub source: Value,
    /// The job files (`-c`) read, paths or URLs, each with its digest.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub job_files: Vec<crate::jobfile::JobFileRecord>,
    /// When this job regrades another: that job's folder.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regrade_of: Option<PathBuf>,
    pub options: LockOptions,
    pub trials: Vec<LockTrial>,
}

/// The `bench` build that wrote the lock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BenchBuild {
    /// The engine's crate version (`0.8.1-dev.0`).
    pub version: String,
    /// The binary's long version with its commit, when the build knew it
    /// (`0.8.1-dev.0 (f5260eaaa 2026-10-10)`).
    pub long_version: Option<String>,
    /// The short git commit the binary was built from.
    pub commit: Option<String>,
    /// That commit's date.
    pub commit_date: Option<String>,
}

impl BenchBuild {
    /// From the CLI's long version, `0.8.1-dev.0 (f5260eaaa 2026-10-10)`.
    pub fn from_long_version(long: Option<&str>) -> Self {
        let inner = long
            .and_then(|l| l.split_once('('))
            .and_then(|(_, rest)| rest.strip_suffix(')'));
        let mut words = inner.into_iter().flat_map(str::split_whitespace);
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            long_version: long.map(str::to_string),
            commit: words.next().map(str::to_string),
            commit_date: words.next().map(str::to_string),
        }
    }
}

/// A registry dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LockDataset {
    pub name: String,
    pub version: String,
    /// The registry it was resolved from: a URL or a file.
    pub registry: String,
}

/// The run options that decide what a trial does. Credentials are never
/// here; `agent_env_keys` names the variables passed with `--env`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LockOptions {
    pub sandbox: String,
    pub sandbox_user: String,
    pub trials_per_task: usize,
    pub concurrency: usize,
    pub agent_timeout_multiplier: f64,
    pub verifier_timeout_multiplier: f64,
    pub build_timeout_multiplier: f64,
    /// Seconds; null means the task's own limit, or none.
    pub default_agent_timeout_sec: Option<f64>,
    pub default_verifier_timeout_sec: f64,
    pub idle_timeout_sec: f64,
    pub pids_limit: u32,
    pub agent_env_keys: Vec<String>,
    pub extra_instruction: Option<String>,
    /// `never`, `on-failure`, or `always`.
    pub keep: String,
    pub save_outputs: bool,
    pub retries: u32,
    /// `--max-tool-calls`, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_calls: Option<u64>,
    /// `--max-tokens`, when set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// The container runtime asked for with `--runtime`, when one was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    /// The statuses `--rerun` asked to run again.
    pub rerun: Vec<String>,
    /// The skill folders `--skill` added, when any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<PathBuf>,
    /// False with `--no-task-skills`.
    #[serde(default = "yes", skip_serializing_if = "is_yes")]
    pub task_skills: bool,
    /// The session file `--load-trajectory` loads, when one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_trajectory: Option<PathBuf>,
    /// True with `--resume-sessions`.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub resume_sessions: bool,
}

fn yes() -> bool {
    true
}

fn is_yes(b: &bool) -> bool {
    *b
}

/// One planned trial.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LockTrial {
    /// The trial's folder name.
    pub trial: String,
    pub task: LockTask,
    pub attempt: LockAttempt,
    pub image: LockImage,
    /// The sandbox backend: docker, podman, or daytona.
    pub sandbox: String,
    /// The earlier trial this one was carried over from (`bench run
    /// --diff`), when it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub carried_from: Option<LockCarry>,
}

/// Where a carried-over trial came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LockCarry {
    /// `reuse` (copied as it was) or `regrade` (its saved outputs graded
    /// again with this task's verifier).
    pub action: String,
    /// The earlier trial's folder.
    pub trial_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LockTask {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    /// `sha256:<hex>` over the task's files.
    pub digest: String,
    /// The digest the dataset registry pins, when it gives one.
    pub registry_digest: Option<String>,
    /// `sha256:<hex>` over what the agent works from and what runs before
    /// its outputs are saved: the instruction, the image, the task's agent
    /// and sandbox settings, skills, mounts, services, and hooks (plus the
    /// reference solution for the oracle). Two trials with the same inputs
    /// digest and different `digest`s differ only in how they are graded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs_digest: Option<String>,
    /// The package format: task-md, task-md-v06, or split.
    pub format: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LockAttempt {
    /// The label in results: `claude`, `oracle`, `nop`, `control:<id>`,
    /// `probe:<id>`.
    pub agent: String,
    pub model: Option<String>,
    /// The reasoning effort asked for (`--effort`), when one was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// `model`, `oracle`, `nop`, `control`, or `probe`.
    pub kind: String,
    /// Changes whenever the agent's installed bits would change.
    pub agent_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LockImage {
    /// `pull` or `build`.
    pub source: String,
    /// The image pulled, for `pull`.
    pub reference: Option<String>,
    /// The Dockerfile built, for `build`.
    pub dockerfile: Option<PathBuf>,
    /// Content hash of everything that decides the image, agent layers
    /// included: the same hash means the same image.
    pub digest: String,
}

/// What the lock records about one job, besides its trials.
pub struct LockHeader<'a> {
    pub job_name: &'a str,
    pub fingerprint: &'a str,
    pub source: &'a Value,
    pub trials_per_task: usize,
    pub concurrency: usize,
    pub regrade_of: Option<PathBuf>,
}

/// Build the lock for `planned` trials under `options`.
pub fn build(header: LockHeader<'_>, planned: &[TrialSpec], options: &RunOptions) -> Lock {
    let mut images_seen: HashMap<(String, String), LockImage> = HashMap::new();
    let trials = planned
        .iter()
        .map(|t| {
            let (layers, fingerprint) = match &t.attempt {
                Attempt::Model(m) => (m.install_layers(), m.fingerprint()),
                Attempt::Script(s) => (Vec::new(), format!("sha256:{}", s.digest())),
                _ => (Vec::new(), String::new()),
            };
            let image = images_seen
                .entry((t.task.id.clone(), fingerprint.clone()))
                .or_insert_with(|| image(&t.task, layers))
                .clone();
            LockTrial {
                trial: t.name.clone(),
                task: LockTask {
                    id: t.task.id.clone(),
                    name: t.task.name.clone(),
                    path: std::fs::canonicalize(&t.task.root)
                        .unwrap_or_else(|_| t.task.root.clone()),
                    digest: t.task.digest.clone(),
                    registry_digest: options
                        .dataset
                        .as_ref()
                        .and_then(|d| d.stamp(&t.task.root).registry_digest),
                    format: serde_json::to_value(t.task.format)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_string))
                        .unwrap_or_default(),
                    inputs_digest: Some(crate::diff::inputs_digest(&t.task, &t.attempt)),
                },
                attempt: LockAttempt {
                    agent: t.attempt.label(),
                    model: t.attempt.model(),
                    effort: t.attempt.effort(),
                    kind: match &t.attempt {
                        Attempt::Oracle => "oracle",
                        Attempt::Nop => "nop",
                        Attempt::Control(_) => "control",
                        Attempt::Probe(_) => "probe",
                        Attempt::Model(_) => "model",
                        Attempt::Script(_) => "script",
                    }
                    .to_string(),
                    agent_fingerprint: (!fingerprint.is_empty()).then_some(fingerprint),
                },
                image,
                sandbox: options.sandbox.clone(),
                carried_from: None,
            }
        })
        .collect();
    Lock {
        schema_version: SCHEMA_VERSION,
        job_name: header.job_name.to_string(),
        created_at: chrono::Local::now().to_rfc3339(),
        bench: BenchBuild::from_long_version(options.bench_version.as_deref()),
        fingerprint: header.fingerprint.to_string(),
        dataset: options.dataset.as_ref().map(|d| LockDataset {
            name: d.name.clone(),
            version: d.version.clone(),
            registry: d.registry.clone(),
        }),
        source: header.source.clone(),
        job_files: options.job_files.clone(),
        regrade_of: header.regrade_of,
        options: lock_options(options, header.trials_per_task, header.concurrency),
        trials,
    }
}

pub(crate) fn lock_options(
    o: &RunOptions,
    trials_per_task: usize,
    concurrency: usize,
) -> LockOptions {
    LockOptions {
        sandbox: o.sandbox.clone(),
        sandbox_user: o.sandbox_user.clone(),
        trials_per_task,
        concurrency,
        agent_timeout_multiplier: o.timeout_multiplier,
        verifier_timeout_multiplier: o.verifier_timeout_multiplier,
        build_timeout_multiplier: o.build_timeout_multiplier,
        default_agent_timeout_sec: o.default_agent_timeout.map(|d| d.as_secs_f64()),
        default_verifier_timeout_sec: o.default_verifier_timeout.as_secs_f64(),
        idle_timeout_sec: o.idle_timeout.as_secs_f64(),
        pids_limit: o.pids_limit,
        runtime: o.runtime.clone(),
        agent_env_keys: o.agent_env.keys().cloned().collect(),
        extra_instruction: o.extra_instruction.clone(),
        keep: serde_json::to_value(o.keep)
            .ok()
            .and_then(|v| v.as_str().map(str::to_string))
            .unwrap_or_default(),
        save_outputs: o.save_outputs,
        retries: o.retries,
        max_tool_calls: o.max_tool_calls,
        max_tokens: o.max_tokens,
        rerun: o.rerun.names().iter().map(|s| s.to_string()).collect(),
        skills: o.skills.clone(),
        task_skills: o.task_skills,
        load_trajectory: o.load_trajectory.clone(),
        resume_sessions: o.resume_sessions,
    }
}

/// The task's image. Agent layers on top are named by the trial's
/// `agent_fingerprint`.
fn image(task: &TaskPackage, layers: Vec<String>) -> LockImage {
    let spec = images::spec_for(task, layers);
    match spec.source {
        ImageSource::Pull { reference } => LockImage {
            source: "pull".into(),
            reference: Some(reference),
            dockerfile: None,
            digest: spec.digest,
        },
        ImageSource::Build { dockerfile, .. } => LockImage {
            source: "build".into(),
            reference: None,
            dockerfile: Some(dockerfile),
            digest: spec.digest,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_splits_into_commit_and_date() {
        let b = BenchBuild::from_long_version(Some("0.8.1-dev.0 (f5260eaaa 2026-10-10)"));
        assert_eq!(b.commit.as_deref(), Some("f5260eaaa"));
        assert_eq!(b.commit_date.as_deref(), Some("2026-10-10"));
        let b = BenchBuild::from_long_version(Some("0.8.1-dev.0"));
        assert_eq!(b.commit, None);
        assert_eq!(b.long_version.as_deref(), Some("0.8.1-dev.0"));
    }
}
