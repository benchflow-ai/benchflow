//! The runtime view of a task package: one normalized shape, whatever the
//! format on disk.
//!
//! [`load`](crate::load) reads a folder in any supported format and returns a
//! [`TaskPackage`]. The run engine reads only this type; it never looks at
//! format-specific keys.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::rubric::Rubric;
use crate::values::Network;

/// The on-disk format a package was read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Format {
    /// task.md draft 2: the instruction first, then a ```toml task block.
    /// BenchFlow's native format.
    TaskMd,
    /// BenchFlow 0.8's task.md: YAML frontmatter, then the prompt.
    TaskMdV06,
    /// The split layout: task.toml + instruction.md + environment/ + tests/ +
    /// solution/ (Harbor's format, and BenchFlow's before 0.8).
    Split,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::TaskMd => "task.md",
            Format::TaskMdV06 => "task.md (0.8 frontmatter)",
            Format::Split => "task.toml (split layout)",
        }
    }
}

/// How the agent's image is obtained.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
pub enum ImageRef {
    /// A prebuilt image (`[sandbox] image`).
    Pull { reference: String },
    /// Build this Dockerfile with this context (`sandbox/Dockerfile`).
    Build {
        context: PathBuf,
        dockerfile: PathBuf,
    },
}

/// What happens when the agent runs out of time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum OnTimeout {
    /// Grade what the agent left (the default, as in Harbor).
    #[default]
    Grade,
    /// Grade it, and report passes at the time limit apart.
    GradeFlagged,
    /// Score 0.
    Fail,
}

/// Where the verifier runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "kebab-case")]
pub enum Isolation {
    /// In the agent's container, after the handoff.
    #[default]
    Shared,
    /// In a fresh container that receives only the saved outputs.
    Separate,
}

/// A readiness check (`[sandbox] ready`), Harbor's healthcheck renamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ready {
    pub run: String,
    pub interval: Duration,
    pub timeout: Duration,
    pub start_period: Duration,
    pub start_interval: Duration,
    pub retries: u32,
}

/// One output the runtime saves after the agent's run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Output {
    /// Absolute path in the sandbox.
    pub path: String,
    /// Name in the trial's record; defaults to the last component of `path`.
    pub save_as: String,
    pub max_bytes: Option<u64>,
    pub exclude: Vec<String>,
}

/// The agent's sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    pub image: ImageRef,
    pub cpus: Option<f64>,
    pub memory_mb: Option<u64>,
    pub disk_mb: Option<u64>,
    pub gpus: u32,
    pub build_timeout: Option<Duration>,
    /// The network for the sandbox's whole life, grading included.
    pub network: Network,
    pub workdir: Option<String>,
    pub env: BTreeMap<String, String>,
    pub ready: Option<Ready>,
    /// What is saved after the agent's run. Empty means `workdir`.
    pub outputs: Vec<Output>,
    /// A skills folder in the package to install for the agent.
    pub skills: Option<PathBuf>,
    /// A skills folder already in the sandbox (Harbor's `skills_dir`, which
    /// the image puts there): an absolute path, copied to the agent at run
    /// time.
    #[serde(default)]
    pub skills_dir: Option<String>,
    /// MCP servers, as the task wrote them (Harbor's server tables).
    pub mcp: Vec<serde_json::Value>,
}

/// The agent's phase.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentPolicy {
    /// `None`: no limit unless the run sets one.
    pub timeout: Option<Duration>,
    pub on_timeout: OnTimeout,
    /// The user the agent's commands run as; `None` lets the run choose.
    pub user: Option<String>,
    /// The network while the agent works; `None` is the sandbox's.
    pub network: Option<Network>,
}

/// The verifier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierPolicy {
    /// The verifier folder in the package (`verifier/` or `tests/`).
    pub dir: PathBuf,
    /// Where it appears while grading (`/verifier` or `/tests`).
    pub mount: String,
    /// The script to run, relative to `mount` (`test.sh`).
    pub entry: String,
    pub timeout: Option<Duration>,
    pub user: Option<String>,
    pub env: BTreeMap<String, String>,
    /// The network while grading; `None` is the sandbox's.
    pub network: Option<Network>,
    pub isolation: Isolation,
    /// task.md's rubric: when present, the reward is computed from it and the
    /// script's CTRF report, not read from a reward file.
    pub rubric: Option<Rubric>,
}

/// The reference solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OraclePolicy {
    /// The oracle folder in the package (`oracle/` or `solution/`).
    pub dir: PathBuf,
    /// Where it appears for the reference run (`/oracle` or `/solution`).
    pub mount: String,
    /// The script to run, relative to `mount` (`solve.sh`).
    pub entry: String,
    pub env: BTreeMap<String, String>,
}

/// A negative control: a script that runs in place of the agent and must not
/// pass. task.md keeps them in `controls/`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Control {
    /// The control's id (`known-bad-line-fit`), from its file name.
    pub id: String,
    /// The script, a path in the package.
    pub script: PathBuf,
    /// The highest reward the control may earn; above it the verifier is too
    /// lenient. `None`: below the pass threshold.
    pub max_reward: Option<f64>,
}

/// A later prompt in the same session (task.md stages that unlock in a chain).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    pub name: String,
    pub prompt: String,
}

/// A field the Rust runtime reads but cannot honor yet. Running a task with
/// one is refused before anything starts, naming the field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Unsupported {
    /// The field, as the task wrote it (`[sandbox] clock`).
    pub field: String,
    pub reason: String,
    /// Refused only when a model agent runs; the oracle and scripted runs are
    /// fine (an agent `budget`, for example).
    pub agents_only: bool,
}

/// A task package, normalized.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPackage {
    /// The package folder.
    pub root: PathBuf,
    pub format: Format,
    /// The task's id: `org/name` from its config, else its folder name.
    pub id: String,
    /// The folder name. Trial names start with it.
    pub name: String,
    pub title: Option<String>,
    pub version: Option<String>,
    /// Exactly what the agent receives as its first prompt.
    pub instruction: String,
    /// Later prompts in the same session, in order.
    pub stages: Vec<Stage>,
    pub agent: AgentPolicy,
    pub sandbox: SandboxPolicy,
    pub verifier: VerifierPolicy,
    pub oracle: Option<OraclePolicy>,
    pub controls: Vec<Control>,
    /// Fields this runtime cannot honor (see [`Unsupported`]).
    pub unsupported: Vec<Unsupported>,
    /// Graded stages, services, hooks, mounts, a verifier image, GPU types.
    #[serde(default)]
    pub extras: crate::extras::Extras,
    /// The config as written, as JSON, for the trial's record.
    pub config: serde_json::Value,
    /// `sha256:<hex>` over the package's files (see [`crate::digest`]).
    pub digest: String,
}

impl TaskPackage {
    /// The declared working folder, or `/app` when the task names none. A
    /// trial of a task that names none works in its image's WORKDIR instead
    /// (as BenchFlow 0.8 does), which only the running sandbox knows.
    pub fn workdir(&self) -> &str {
        self.sandbox.workdir.as_deref().unwrap_or("/app")
    }

    /// The agent's effective network.
    pub fn agent_network(&self) -> &Network {
        self.agent.network.as_ref().unwrap_or(&self.sandbox.network)
    }

    /// The verifier's effective network.
    pub fn verifier_network(&self) -> &Network {
        self.verifier
            .network
            .as_ref()
            .unwrap_or(&self.sandbox.network)
    }

    /// Hidden paths the agent must never see: the verifier and oracle mounts.
    pub fn hidden_mounts(&self) -> Vec<String> {
        let mut v = vec![self.verifier.mount.clone()];
        if let Some(o) = &self.oracle {
            v.push(o.mount.clone());
        }
        v
    }

    /// The package's file path for a package-relative path.
    pub fn path(&self, rel: impl AsRef<Path>) -> PathBuf {
        self.root.join(rel)
    }
}
