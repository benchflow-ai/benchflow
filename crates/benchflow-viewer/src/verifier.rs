//! A trial's verifier output (`verifier/`) and its verifier-recovery
//! receipts (0.8's `_load_verifier` and `_load_recovery`).

use std::path::Path;
use std::time::UNIX_EPOCH;

use crate::metadata::ErrorBanner;
use crate::py;
use crate::pyjson::{Value, get, object};
use crate::steps::verifier_status;
use crate::values::{
    display_text, error_text, finite_float, load_json, optional_bool, optional_text, read_text,
};

/// The verifier files the viewer reads, all under `verifier/`.
pub const VERIFIER_SIDECARS: [&str; 4] = [
    "reward.txt",
    "test-stdout.txt",
    "test-stderr.txt",
    "ctrf.json",
];

/// At most this many entries of `verifier-recovery/` are considered.
const MAX_RECOVERY_ATTEMPTS: usize = 50;

#[derive(Debug, Clone, Default)]
pub struct VerifierArtifacts {
    pub reward: Option<String>,
    pub stdout: Option<String>,
    pub stderr: Option<String>,
    pub ctrf: Option<Vec<VerifierTest>>,
    pub recovery: Option<Recovery>,
}

#[derive(Debug, Clone)]
pub struct VerifierTest {
    pub name: String,
    pub status: &'static str,
    pub duration: Option<f64>,
}

/// Verifier-only recovery evidence: the admitted attempt named by
/// `verification.json`, then any other receipts, oldest first.
#[derive(Debug, Clone)]
pub struct Recovery {
    pub pointer: Option<String>,
    pub pointer_error: Option<String>,
    pub attempts: Vec<RecoveryAttempt>,
}

/// One `verifier-recovery/<id>/recovery.json` receipt.
#[derive(Debug, Clone, Default)]
pub struct RecoveryAttempt {
    pub attempt: String,
    pub admitted: bool,
    pub status: Option<String>,
    pub original_error: Option<String>,
    pub error: Option<String>,
    pub evidence: Option<String>,
    pub solver_replayed: Option<bool>,
    pub publication_error: Option<String>,
    pub cleanup_error: Option<String>,
    pub admission: Option<String>,
    pub reward: Option<f64>,
    pub verifier_sec: Option<f64>,
}

impl VerifierArtifacts {
    pub fn into_value(self) -> Value {
        object([
            ("reward", self.reward.into()),
            ("stdout", self.stdout.into()),
            ("stderr", self.stderr.into()),
            (
                "ctrf",
                self.ctrf
                    .map(|tests| {
                        Value::Array(
                            tests
                                .into_iter()
                                .map(|test| {
                                    object([
                                        ("name", test.name.into()),
                                        ("status", test.status.into()),
                                        ("duration", test.duration.into()),
                                    ])
                                })
                                .collect(),
                        )
                    })
                    .into(),
            ),
            ("recovery", self.recovery.map(Recovery::into_value).into()),
        ])
    }
}

impl Recovery {
    fn into_value(self) -> Value {
        object([
            ("pointer", self.pointer.into()),
            ("pointer_error", self.pointer_error.into()),
            (
                "attempts",
                Value::Array(
                    self.attempts
                        .into_iter()
                        .map(RecoveryAttempt::into_value)
                        .collect(),
                ),
            ),
        ])
    }

    /// Banners for an admitted attempt whose publication failed: `verifier/`
    /// then predates the admitted score.
    pub fn banners(&self) -> Vec<ErrorBanner> {
        self.attempts
            .iter()
            .filter(|attempt| attempt.admitted)
            .filter_map(|attempt| {
                let error = attempt.publication_error.as_deref().filter(|error| !error.is_empty())?;
                Some(ErrorBanner {
                    label: "verifier publication error".to_string(),
                    text: format!(
                        "{error}\nThe admitted score comes from {0}; verifier/ still holds the outputs from \
                         before recovery, and the recovered outputs are in {0}/verifier/.",
                        attempt.attempt
                    ),
                    level: Some("info"),
                })
            })
            .collect()
    }
}

impl RecoveryAttempt {
    fn into_value(self) -> Value {
        object([
            ("attempt", self.attempt.into()),
            ("admitted", self.admitted.into()),
            ("status", self.status.into()),
            ("original_error", self.original_error.into()),
            ("error", self.error.into()),
            ("evidence", self.evidence.into()),
            ("solver_replayed", self.solver_replayed.into()),
            ("publication_error", self.publication_error.into()),
            ("cleanup_error", self.cleanup_error.into()),
            ("admission", self.admission.into()),
            ("reward", self.reward.into()),
            ("verifier_sec", self.verifier_sec.into()),
        ])
    }

    fn from_record(attempt: String, record: Option<&Value>, admitted: bool) -> RecoveryAttempt {
        let Some(Value::Object(record)) = record else {
            return RecoveryAttempt {
                attempt,
                admitted,
                ..RecoveryAttempt::default()
            };
        };
        RecoveryAttempt {
            attempt,
            admitted,
            status: optional_text(get(record, "status")),
            original_error: error_text(get(record, "original_error")),
            error: error_text(get(record, "error")),
            evidence: optional_text(get(record, "evidence")),
            solver_replayed: optional_bool(get(record, "solver_replayed")),
            publication_error: error_text(get(record, "publication_error")),
            cleanup_error: error_text(get(record, "cleanup_error")),
            admission: optional_text(get(record, "admission")),
            reward: get(record, "rewards")
                .as_object()
                .and_then(|rewards| finite_float(get(rewards, "reward"))),
            verifier_sec: get(record, "timing")
                .as_object()
                .and_then(|timing| finite_float(get(timing, "verifier"))),
        }
    }
}

/// The verifier sidecars of a trial (or of a branch child's handoff folder).
pub fn load_verifier(dir: &Path) -> VerifierArtifacts {
    let [reward_file, stdout_file, stderr_file, ctrf_file] = VERIFIER_SIDECARS;
    let verifier = dir.join("verifier");
    let reward = read_text(&verifier.join(reward_file));
    let ctrf = load_json(&verifier.join(ctrf_file)).and_then(|ctrf| {
        let tests = ctrf.get("results").get("tests").as_array()?;
        Some(
            tests
                .iter()
                .filter_map(Value::as_object)
                .map(|test| VerifierTest {
                    name: display_text(get(test, "name")),
                    status: verifier_status(get(test, "status")),
                    duration: finite_float(get(test, "duration")),
                })
                .collect(),
        )
    });
    VerifierArtifacts {
        // `reward.strip() if reward else None`: empty text is no reward.
        reward: reward
            .filter(|reward| !reward.is_empty())
            .map(|reward| py::strip(&reward).to_string()),
        stdout: read_text(&verifier.join(stdout_file)),
        stderr: read_text(&verifier.join(stderr_file)),
        ctrf,
        recovery: None,
    }
}

/// `verifier-recovery/<id>` with `<id>` letters and digits only.
fn is_attempt_ref(text: &str) -> bool {
    let mut parts = text.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some("verifier-recovery"), Some(id), None) if py::isalnum(id))
}

/// The verification pointer and every recovery receipt, or `None` when the
/// trial has neither (`_load_recovery`).
pub fn load_recovery(rollout_dir: &Path) -> Option<Recovery> {
    let mut pointer = None;
    let mut pointer_error = None;
    let pointer_path = rollout_dir.join("verification.json");
    if pointer_path.is_file() {
        let attempt = load_json(&pointer_path)
            .map(|raw| raw.get("attempt").clone())
            .unwrap_or(Value::Null);
        match attempt.as_str() {
            Some(attempt) if is_attempt_ref(attempt) => pointer = Some(attempt.to_string()),
            _ => {
                pointer_error =
                    Some("does not reference verifier-recovery/<attempt id>".to_string())
            }
        }
    }

    let mut entries: Vec<std::path::PathBuf> =
        std::fs::read_dir(rollout_dir.join("verifier-recovery"))
            .map(|dir| {
                dir.filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .collect()
            })
            .unwrap_or_default();
    entries.sort();
    let mut receipts: Vec<(f64, String, std::path::PathBuf)> = Vec::new();
    for entry in entries.into_iter().take(MAX_RECOVERY_ATTEMPTS) {
        let name = py::os_text(entry.file_name().unwrap_or_default());
        if !py::isalnum(&name) {
            continue;
        }
        let receipt = entry.join("recovery.json");
        let Some(modified) = std::fs::metadata(&receipt)
            .ok()
            .and_then(|meta| meta.modified().ok())
        else {
            continue;
        };
        let modified = modified
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |since| since.as_secs_f64());
        receipts.push((modified, format!("verifier-recovery/{name}"), receipt));
    }
    if pointer.is_none() && pointer_error.is_none() && receipts.is_empty() {
        return None;
    }
    // Oldest first; equal times by name.
    receipts.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    let mut attempts: Vec<RecoveryAttempt> = receipts
        .into_iter()
        .map(|(_, attempt, path)| {
            let admitted = pointer.as_deref() == Some(attempt.as_str());
            RecoveryAttempt::from_record(attempt, load_json(&path).as_ref(), admitted)
        })
        .collect();
    if let Some(pointer) = &pointer {
        // A pointer whose receipt is absent (a dataset slice that fetched
        // only the pointer) still shows as the admitted attempt.
        if attempts.iter().all(|attempt| &attempt.attempt != pointer) {
            attempts.push(RecoveryAttempt {
                attempt: pointer.clone(),
                admitted: true,
                ..RecoveryAttempt::default()
            });
        }
    }
    // The admitted attempt first; the sort is stable.
    attempts.sort_by_key(|attempt| !attempt.admitted);
    Some(Recovery {
        pointer,
        pointer_error,
        attempts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attempt_refs_name_one_recovery_folder() {
        assert!(is_attempt_ref("verifier-recovery/abc123"));
        assert!(!is_attempt_ref("../../etc"));
        assert!(!is_attempt_ref("verifier-recovery/abc/def"));
        assert!(!is_attempt_ref("verifier-recovery/"));
        assert!(!is_attempt_ref("verifier-recovery/a-b"));
    }

    #[test]
    fn reward_text_is_stripped_and_empty_text_is_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("verifier")).unwrap();
        std::fs::write(dir.path().join("verifier/reward.txt"), " 1.0\r\n").unwrap();
        assert_eq!(load_verifier(dir.path()).reward.as_deref(), Some("1.0"));
        std::fs::write(dir.path().join("verifier/reward.txt"), "").unwrap();
        assert_eq!(load_verifier(dir.path()).reward, None);
        std::fs::write(dir.path().join("verifier/reward.txt"), "  ").unwrap();
        assert_eq!(load_verifier(dir.path()).reward.as_deref(), Some(""));
    }
}
