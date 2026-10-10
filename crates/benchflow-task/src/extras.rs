//! Runtime features beyond one prompt in one container: graded stages,
//! services beside the agent's container, a verifier image of its own,
//! hooks that save state before grading, mounted package files, and GPU
//! types.
//!
//! Every format fills the same [`Extras`] on its [`TaskPackage`]; the run
//! engine reads only this type.
//!
//! [`TaskPackage`]: crate::TaskPackage

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::package::{ImageRef, Output, Ready};

/// What a task asks of the runtime beyond its sandbox, agent, and verifier.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Extras {
    /// Stages graded one after another in the same sandbox (Harbor's
    /// `[[steps]]`, task.md stages graded on their own).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub steps: Option<Steps>,
    /// Containers beside the agent's, reached by name.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub services: Vec<Service>,
    /// Commands run after the agent's turn and before grading.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub hooks: Vec<Hook>,
    /// Package files placed read-only in the sandbox before the agent starts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mounts: Vec<Mount>,
    /// A separate verifier's own image, when it is not the task's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verifier_image: Option<VerifierImage>,
    /// The GPU models the task accepts; empty accepts any.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub gpu_types: Vec<String>,
    /// Literal environment values the Compose file sets on service main;
    /// the reader adds them to the sandbox's environment.
    #[serde(skip)]
    pub main_env: BTreeMap<String, String>,
    /// What reading the extras found for `bench check` and the run to show:
    /// Compose settings ignored, host values dropped.
    #[serde(skip)]
    pub diagnostics: Vec<crate::diag::Diagnostic>,
}

impl Extras {
    /// True when nothing beyond the base runtime is asked for.
    pub fn is_empty(&self) -> bool {
        self.steps.is_none()
            && self.services.is_empty()
            && self.hooks.is_empty()
            && self.mounts.is_empty()
            && self.verifier_image.is_none()
            && self.gpu_types.is_empty()
    }
}

/// How the trial's reward comes from its stages' rewards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Combine {
    /// Each reward key's mean over the stages that count.
    #[default]
    Mean,
    /// The last stage's rewards.
    Final,
}

/// What a stage the run never reached counts as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Unreached {
    /// Every reward key 0 (task.md's default).
    #[default]
    Zero,
    /// Left out (Harbor's rule: it averages only the steps that ran).
    Exclude,
}

/// The minimum a stage must score for the next one to unlock.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Gate {
    /// A minimum for `reward`.
    Reward(f64),
    /// A minimum for each named reward key; a missing key fails.
    Keys(BTreeMap<String, f64>),
}

impl Gate {
    /// The first threshold `rewards` misses, as `key=value below min`, or
    /// `None` when every threshold holds.
    pub fn failure(&self, rewards: Option<&BTreeMap<String, f64>>) -> Option<String> {
        let thresholds: Vec<(String, f64)> = match self {
            Gate::Reward(min) => vec![("reward".to_string(), *min)],
            Gate::Keys(keys) => keys.iter().map(|(k, v)| (k.clone(), *v)).collect(),
        };
        for (key, min) in thresholds {
            match rewards.and_then(|r| r.get(&key)) {
                Some(v) if *v >= min => {}
                Some(v) => return Some(format!("{key} = {v}, below the gate's {min}")),
                None => return Some(format!("no {key} reward, and the gate needs {min}")),
            }
        }
        None
    }
}

/// Stages graded one after another.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Steps {
    pub stages: Vec<StepSpec>,
    pub combine: Combine,
    pub unreached: Unreached,
}

/// One graded stage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StepSpec {
    /// The stage's name; its files go to `steps/<name>/` in the trial folder.
    pub name: String,
    /// What the agent receives when the stage unlocks.
    pub prompt: String,
    /// The agent's time for this stage; `None` is the task's.
    pub timeout: Option<Duration>,
    /// The verifier's time for this stage; `None` is the task's.
    pub verifier_timeout: Option<Duration>,
    /// Whether the stage is checked when it ends. An ungraded stage (task.md's
    /// opening turn before any stage unlocks) only runs the agent.
    pub graded: bool,
    /// Package folders whose files make up the stage's verifier, in order
    /// (a later folder's files replace an earlier one's). Empty: the task's
    /// verifier folder.
    pub verifier_dirs: Vec<PathBuf>,
    /// The stage's own reference solution folder; `None` is the task's.
    pub oracle_dir: Option<PathBuf>,
    pub gate: Option<Gate>,
    /// Saved when the stage ends, to `steps/<name>/artifacts/`.
    pub outputs: Vec<Output>,
    /// A package folder copied into the workdir when the stage unlocks; a
    /// `setup.sh` in it then runs as the agent's user (Harbor's
    /// `steps/<name>/workdir/`).
    pub workdir: Option<PathBuf>,
    /// Package files placed when the stage unlocks.
    pub mounts: Vec<Mount>,
    /// A readiness check run when the stage unlocks.
    pub ready: Option<Ready>,
}

/// A container beside the agent's, on a private network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Service {
    /// The host name the agent reaches it by.
    pub name: String,
    /// A prebuilt image, or a Dockerfile in the package (paths relative to
    /// the package).
    pub image: ImageRef,
    /// The command, as argv; empty runs the image's own.
    pub command: Vec<String>,
    pub env: BTreeMap<String, String>,
    /// Must pass, run inside the service, before the agent starts.
    pub ready: Option<Ready>,
}

/// A command run after the agent's turn and before grading, as root unless
/// it names a user, to save runtime state into files.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hook {
    pub run: String,
    /// The service it runs in; `None` is the agent's container.
    pub service: Option<String>,
    pub timeout: Duration,
    pub user: Option<String>,
}

/// A package file or folder placed read-only in the sandbox.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Mount {
    /// Relative to the package.
    pub source: PathBuf,
    /// Absolute path in the sandbox.
    pub target: String,
}

/// A separate verifier's image and where it came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierImage {
    /// Paths relative to the package.
    pub image: ImageRef,
    /// What chose it, for the record (`verifier/Dockerfile`,
    /// `[verifier.sandbox] image`).
    pub source: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gates_name_the_missing_threshold() {
        let mut r = BTreeMap::new();
        r.insert("reward".to_string(), 0.5);
        assert_eq!(Gate::Reward(0.5).failure(Some(&r)), None);
        assert!(
            Gate::Reward(1.0)
                .failure(Some(&r))
                .unwrap()
                .contains("below")
        );
        assert!(
            Gate::Reward(0.0)
                .failure(None)
                .unwrap()
                .contains("no reward")
        );
        let keys = Gate::Keys(BTreeMap::from([("tests".to_string(), 1.0)]));
        assert!(keys.failure(Some(&r)).unwrap().contains("no tests reward"));
    }
}
