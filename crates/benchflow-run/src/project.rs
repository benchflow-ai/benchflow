//! `bench.toml`: a project's defaults, found by walking up from the current
//! folder, as Cargo finds `Cargo.toml`.
//!
//! ```toml
//! [run]
//! agent = "claude"
//! model = "claude-sonnet-4-6"
//! sandbox = "docker"
//! concurrency = 8
//! jobs-dir = "jobs"
//!
//! [tasks]
//! paths = ["tasks"]
//!
//! [profile.ci.run]
//! sandbox = "daytona"
//! concurrency = 32
//! fail-under = 0.8
//! ```
//!
//! Every key is optional. Flags override the profile, which overrides `[run]`,
//! which overrides built-in defaults.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub const FILE_NAME: &str = "bench.toml";

/// Defaults for `bench run` and `bench test`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, schemars::JsonSchema)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct RunDefaults {
    pub agent: Option<String>,
    /// How the agent is driven: `acp` or `headless` (default: the agent's own).
    pub harness: Option<String>,
    pub model: Option<String>,
    pub reasoning_effort: Option<String>,
    pub sandbox: Option<String>,
    pub concurrency: Option<usize>,
    /// Trials per task: 1 or more (`bench run` refuses 0).
    #[schemars(range(min = 1))]
    pub trials: Option<usize>,
    pub jobs_dir: Option<PathBuf>,
    /// Scales the agent, verifier, and build timeouts.
    pub timeout_multiplier: Option<f64>,
    pub agent_timeout_multiplier: Option<f64>,
    pub verifier_timeout_multiplier: Option<f64>,
    pub build_timeout_multiplier: Option<f64>,
    pub idle_timeout: Option<String>,
    pub fail_under: Option<f64>,
    pub sandbox_user: Option<String>,
    pub retries: Option<u32>,
    /// The container runtime sandboxes start under, such as `runsc`
    /// (gVisor), for Docker daemons that have it.
    pub runtime: Option<String>,
    #[serde(default)]
    pub env: std::collections::BTreeMap<String, String>,
    /// Agent options (`--ak name=value`), checked against the agent's
    /// declared options before any sandbox starts.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub agent_options: std::collections::BTreeMap<String, serde_json::Value>,
    /// Resume the agent's own session on each graded stage after the first
    /// (`--resume-sessions`).
    pub resume_sessions: Option<bool>,
    /// Skill folders added for the agent (`--skill`): each a skill, or a
    /// folder of skills.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<PathBuf>,
    /// Install the task's own skills (`false`: `--no-task-skills`).
    pub task_skills: Option<bool>,
    /// A session file the agent's first prompt resumes
    /// (`--load-trajectory`).
    pub load_trajectory: Option<PathBuf>,
}

impl RunDefaults {
    /// `other`'s values where set, else ours.
    pub fn overlay(&self, other: &RunDefaults) -> RunDefaults {
        let mut env = self.env.clone();
        env.extend(other.env.clone());
        let mut agent_options = self.agent_options.clone();
        agent_options.extend(other.agent_options.clone());
        RunDefaults {
            agent: other.agent.clone().or(self.agent.clone()),
            harness: other.harness.clone().or(self.harness.clone()),
            model: other.model.clone().or(self.model.clone()),
            reasoning_effort: other
                .reasoning_effort
                .clone()
                .or(self.reasoning_effort.clone()),
            sandbox: other.sandbox.clone().or(self.sandbox.clone()),
            concurrency: other.concurrency.or(self.concurrency),
            trials: other.trials.or(self.trials),
            jobs_dir: other.jobs_dir.clone().or(self.jobs_dir.clone()),
            timeout_multiplier: other.timeout_multiplier.or(self.timeout_multiplier),
            agent_timeout_multiplier: other
                .agent_timeout_multiplier
                .or(self.agent_timeout_multiplier),
            verifier_timeout_multiplier: other
                .verifier_timeout_multiplier
                .or(self.verifier_timeout_multiplier),
            build_timeout_multiplier: other
                .build_timeout_multiplier
                .or(self.build_timeout_multiplier),
            idle_timeout: other.idle_timeout.clone().or(self.idle_timeout.clone()),
            fail_under: other.fail_under.or(self.fail_under),
            sandbox_user: other.sandbox_user.clone().or(self.sandbox_user.clone()),
            retries: other.retries.or(self.retries),
            runtime: other.runtime.clone().or(self.runtime.clone()),
            env,
            agent_options,
            resume_sessions: other.resume_sessions.or(self.resume_sessions),
            skills: if other.skills.is_empty() {
                self.skills.clone()
            } else {
                other.skills.clone()
            },
            task_skills: other.task_skills.or(self.task_skills),
            load_trajectory: other
                .load_trajectory
                .clone()
                .or(self.load_trajectory.clone()),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct TaskDefaults {
    /// Where `bench check`, `test`, and `run` look for tasks when given none.
    #[serde(default)]
    pub paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct Profile {
    #[serde(default)]
    pub run: RunDefaults,
}

/// A parsed `bench.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case", deny_unknown_fields)]
pub struct ProjectFile {
    #[serde(default)]
    pub run: RunDefaults,
    #[serde(default)]
    pub tasks: TaskDefaults,
    #[serde(default)]
    pub profile: std::collections::BTreeMap<String, Profile>,
    /// Project agents, `[agents.<name>]`, in the agent manifest schema.
    #[serde(default)]
    pub agents: std::collections::BTreeMap<String, toml::Value>,
}

/// A project: its file and the folder it lives in.
#[derive(Debug, Clone)]
pub struct Project {
    pub root: PathBuf,
    pub file: Option<PathBuf>,
    pub config: ProjectFile,
}

/// A bench.toml that could not be read.
#[derive(Debug, thiserror::Error)]
#[error("{path}: {message}")]
pub struct ProjectError {
    pub path: PathBuf,
    pub message: String,
}

impl Project {
    /// Find `bench.toml` from `start` upward. No file is fine: the project is
    /// `start` with defaults.
    pub fn discover(start: &Path) -> Result<Self, ProjectError> {
        let mut dir = Some(start);
        while let Some(d) = dir {
            let candidate = d.join(FILE_NAME);
            if candidate.is_file() {
                return Self::load(&candidate);
            }
            if d.join(".git").exists() {
                break;
            }
            dir = d.parent();
        }
        Ok(Self {
            root: start.to_path_buf(),
            file: None,
            config: ProjectFile::default(),
        })
    }

    pub fn load(path: &Path) -> Result<Self, ProjectError> {
        let text = std::fs::read_to_string(path).map_err(|e| ProjectError {
            path: path.to_path_buf(),
            message: e.to_string(),
        })?;
        let config: ProjectFile = toml::from_str(&text).map_err(|e| ProjectError {
            path: path.to_path_buf(),
            message: e.to_string().trim().to_string(),
        })?;
        Ok(Self {
            root: path.parent().unwrap_or(Path::new(".")).to_path_buf(),
            file: Some(path.to_path_buf()),
            config,
        })
    }

    /// `[run]` with a profile laid over it.
    pub fn run_defaults(&self, profile: Option<&str>) -> Result<RunDefaults, String> {
        match profile {
            None => Ok(self.config.run.clone()),
            Some(name) => match self.config.profile.get(name) {
                Some(p) => Ok(self.config.run.overlay(&p.run)),
                None => {
                    let known: Vec<&str> = self.config.profile.keys().map(String::as_str).collect();
                    Err(if known.is_empty() {
                        format!("no profile `{name}`: bench.toml defines no profiles")
                    } else {
                        format!(
                            "no profile `{name}`; bench.toml defines {}",
                            known.join(", ")
                        )
                    })
                }
            },
        }
    }

    /// The task folders to use when the command names none.
    pub fn task_paths(&self) -> Vec<PathBuf> {
        self.config
            .tasks
            .paths
            .iter()
            .map(|p| {
                if p.is_absolute() {
                    p.clone()
                } else {
                    self.root.join(p)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_overlay_run() {
        let text = r#"
[run]
agent = "claude"
concurrency = 4

[profile.ci.run]
concurrency = 32
fail-under = 0.8
"#;
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(FILE_NAME), text).unwrap();
        let p = Project::discover(d.path()).unwrap();
        let r = p.run_defaults(Some("ci")).unwrap();
        assert_eq!(r.agent.as_deref(), Some("claude"));
        assert_eq!(r.concurrency, Some(32));
        assert_eq!(r.fail_under, Some(0.8));
        assert!(p.run_defaults(Some("nope")).unwrap_err().contains("ci"));
    }

    #[test]
    fn unknown_keys_are_errors() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join(FILE_NAME), "[run]\nagnet = \"x\"\n").unwrap();
        let e = Project::discover(d.path()).unwrap_err();
        assert!(e.message.contains("agnet"), "{}", e.message);
    }
}
