//! Run options: everything a run needs besides the tasks and the agent.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// When to keep a sandbox after its trial, for inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Keep {
    /// Destroy every sandbox (the default).
    #[default]
    Never,
    /// Keep sandboxes whose trial did not pass.
    OnFailure,
    /// Keep every sandbox.
    Always,
}

/// Options that apply to every trial of a run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunOptions {
    /// The sandbox backend: `docker`, `podman`, `daytona`.
    pub sandbox: String,
    /// The user agents run as when the task names none.
    pub sandbox_user: String,
    /// Scale every task's agent timeout.
    pub timeout_multiplier: f64,
    /// Scale every task's verifier timeout.
    #[serde(default = "one")]
    pub verifier_timeout_multiplier: f64,
    /// Scale every task's image build timeout.
    #[serde(default = "one")]
    pub build_timeout_multiplier: f64,
    /// Agent timeout when the task sets none. `None`: no limit.
    pub default_agent_timeout: Option<Duration>,
    /// Abort a prompt when the agent is silent this long.
    pub idle_timeout: Duration,
    /// Stop the agent after this many tool calls (`--max-tool-calls`), where
    /// its harness can count them.
    pub max_tool_calls: Option<u64>,
    /// Stop the agent after this many tokens, input plus output
    /// (`--max-tokens`), where its harness can count them.
    pub max_tokens: Option<u64>,
    /// Verifier timeout when the task sets none.
    pub default_verifier_timeout: Duration,
    /// Most processes a sandbox may hold (fork-bomb guard).
    pub pids_limit: u32,
    /// Extra environment for the agent (`--env KEY=VALUE`).
    pub agent_env: BTreeMap<String, String>,
    /// Text appended to every instruction (`--extra-instruction`).
    pub extra_instruction: Option<String>,
    /// On each graded stage after the first, resume the agent's own session
    /// from the stage before (`--resume-sessions`), where the agent can.
    pub resume_sessions: bool,
    /// Skills the run adds for the agent (`--skill`): each a folder that
    /// holds a `SKILL.md`, one per name ([`crate::skills::resolve`]).
    pub skills: Vec<PathBuf>,
    /// Install the task's own skills; false with `--no-task-skills`
    /// (BenchFlow 0.8's `no-skill` mode).
    pub task_skills: bool,
    /// A session file the first prompt resumes (`--load-trajectory`).
    pub load_trajectory: Option<PathBuf>,
    pub keep: Keep,
    /// Save the agent's workdir (or declared outputs) into the trial folder.
    pub save_outputs: bool,
    /// Retry a trial whose failure is transient (infrastructure) this many
    /// times.
    pub retries: u32,
    /// Where image build logs and other caches live.
    pub cache_dir: PathBuf,
    /// On resume, run these finished trials again instead of reusing them.
    #[serde(default)]
    pub rerun: Rerun,
    /// The dataset the tasks came from, stamped into every trial.
    #[serde(default)]
    pub dataset: Option<DatasetStamp>,
    /// The job files (`-c`) the run read, with their digests; written to
    /// lock.json.
    #[serde(default)]
    pub job_files: Vec<crate::jobfile::JobFileRecord>,
    /// The `bench` build that runs the job, with its git commit when known
    /// (`0.8.1-dev.0 (f5260eaaa 2026-10-10)`); written to lock.json.
    #[serde(default)]
    pub bench_version: Option<String>,
    /// The container runtime sandboxes start under (`runsc` for gVisor);
    /// `None`: the backend's default.
    #[serde(default)]
    pub runtime: Option<String>,
    /// Scan each task's image for credential files before its empty run
    /// (`bench test` turns this on).
    #[serde(default)]
    pub scan_credentials: bool,
}

fn one() -> f64 {
    1.0
}

impl RunOptions {
    /// A task's verifier time limit (or the run's default), scaled.
    pub fn verifier_timeout(&self, task_timeout: Option<Duration>) -> Duration {
        task_timeout
            .unwrap_or(self.default_verifier_timeout)
            .mul_f64(self.verifier_timeout_multiplier.max(0.01))
    }
}

/// Which finished trials a resumed job runs again. Errored trials always run
/// again; scored and unscored ones are reused unless named here.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rerun {
    /// Trials that ran into an error (always rerun; accepted for clarity).
    pub errored: bool,
    /// Trials whose verifier could not score them.
    pub unscored: bool,
    /// Trials that were scored and did not pass.
    pub failed: bool,
}

impl Rerun {
    /// Parse `errored,unscored,failed` (any subset, comma-separated).
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut r = Rerun::default();
        for word in text.split(',').map(str::trim).filter(|w| !w.is_empty()) {
            match word {
                "errored" => r.errored = true,
                "unscored" => r.unscored = true,
                "failed" => r.failed = true,
                other => {
                    return Err(format!(
                        "`{other}` is not a trial status to rerun; use errored, unscored, or failed"
                    ));
                }
            }
        }
        Ok(r)
    }

    /// The names set, comma-separated.
    pub fn names(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        if self.errored {
            v.push("errored");
        }
        if self.unscored {
            v.push("unscored");
        }
        if self.failed {
            v.push("failed");
        }
        v
    }
}

/// A dataset's identity, and the digest its registry pins for each task.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DatasetStamp {
    pub name: String,
    pub version: String,
    /// The registry it was resolved from (a URL or a file).
    pub registry: String,
    /// The registry's digest for each task, keyed by the task's folder.
    #[serde(default)]
    pub digests: BTreeMap<PathBuf, String>,
}

impl DatasetStamp {
    /// What a trial of the task in `root` records about the dataset.
    pub fn stamp(&self, root: &std::path::Path) -> TrialDataset {
        TrialDataset {
            name: self.name.clone(),
            version: self.version.clone(),
            registry_digest: self.digests.get(root).cloned(),
        }
    }
}

/// The dataset a trial's task came from, as result.json records it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TrialDataset {
    pub name: String,
    pub version: String,
    /// The digest the registry pins for the task, when it gives one.
    pub registry_digest: Option<String>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            sandbox: "docker".into(),
            sandbox_user: crate::hardening::DEFAULT_USER.into(),
            timeout_multiplier: 1.0,
            verifier_timeout_multiplier: 1.0,
            build_timeout_multiplier: 1.0,
            default_agent_timeout: None,
            idle_timeout: Duration::from_secs(600),
            max_tool_calls: None,
            max_tokens: None,
            default_verifier_timeout: Duration::from_secs(600),
            pids_limit: 4096,
            agent_env: BTreeMap::new(),
            extra_instruction: None,
            resume_sessions: false,
            skills: Vec::new(),
            task_skills: true,
            load_trajectory: None,
            keep: Keep::Never,
            save_outputs: false,
            retries: 1,
            cache_dir: default_cache_dir(),
            rerun: Rerun::default(),
            dataset: None,
            job_files: Vec::new(),
            bench_version: None,
            runtime: None,
            scan_credentials: false,
        }
    }
}

/// `~/.cache/benchflow`, as BenchFlow 0.8 uses on every platform;
/// `BENCHFLOW_CACHE_DIR` or `XDG_CACHE_HOME` move it.
pub fn default_cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("BENCHFLOW_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("benchflow");
    }
    dirs::home_dir()
        .map(|h| h.join(".cache"))
        .unwrap_or_else(std::env::temp_dir)
        .join("benchflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rerun_kinds_parse() {
        let r = Rerun::parse("failed, unscored").unwrap();
        assert!(r.failed && r.unscored && !r.errored);
        assert_eq!(r.names(), ["unscored", "failed"]);
        assert!(
            Rerun::parse("passed")
                .unwrap_err()
                .contains("errored, unscored, or failed")
        );
    }
}
