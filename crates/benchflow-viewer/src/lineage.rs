//! Rollout-branch lineage from a trial's `tree.json` (0.8's `_load_lineage`).
//!
//! A branched trial forks at a node and runs several children from there.
//! `tree.json` records each fork and child; a child whose evidence was
//! published under the trial folder can be opened as its own trajectory
//! (see [`crate::payload::build_branch_child_payload`]).

use std::path::{Path, PathBuf};

use crate::pyjson::{Int, Map, Value, get, object};
use crate::values::{
    display_text, error_text, exact_int, finite_float, load_json, optional_text, text_list,
};

/// The child archive file a branch child's trajectory is read from.
pub const BRANCH_OBSERVATION: &str = "observation.json";

#[derive(Debug, Clone)]
pub struct Lineage {
    pub nodes: usize,
    pub forks: Vec<BranchFork>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BranchFork {
    pub id: String,
    pub parent_node: Option<String>,
    pub status: Option<String>,
    pub value: Option<f64>,
    pub requested_children: Option<Int>,
    pub requested_layers: Vec<String>,
    pub captured_layers: Vec<String>,
    pub parent_restore: Option<String>,
    pub error: Option<String>,
    pub parent_restore_error: Option<String>,
    pub artifact_error: Option<String>,
    pub children: Vec<BranchChild>,
}

/// One child row of a fork.
#[derive(Debug, Clone)]
pub struct BranchChild {
    pub index: Option<Int>,
    pub node_id: Option<String>,
    pub status: Option<String>,
    pub reward: Option<f64>,
    pub reward_source: Option<String>,
    pub error: Option<String>,
    pub cleanup_error: Option<String>,
    /// `label`, `requested`, `execution` and `evidence`, in that order.
    pub intervention: [(&'static str, Option<String>); 4],
    pub artifacts_status: Option<String>,
    pub artifacts_path: Option<String>,
    /// `<fork id>/<node id>`, set only when the child's evidence was
    /// published inside the trial folder, so the viewer can open it.
    pub child_ref: Option<String>,
}

impl BranchChild {
    pub fn label(&self) -> Option<String> {
        self.intervention[0].1.clone()
    }
}

impl Lineage {
    fn unreadable(error: String) -> Lineage {
        Lineage {
            nodes: 0,
            forks: Vec::new(),
            error: Some(error),
        }
    }

    pub fn into_value(self) -> Value {
        object([
            ("source", "tree.json".into()),
            ("nodes", self.nodes.into()),
            (
                "forks",
                Value::Array(self.forks.into_iter().map(BranchFork::into_value).collect()),
            ),
            ("error", self.error.into()),
        ])
    }

    /// Every openable child ref, in tree order, each once.
    pub fn child_refs(&self) -> Vec<String> {
        let mut refs: Vec<String> = Vec::new();
        for child in self.forks.iter().flat_map(|fork| &fork.children) {
            if let Some(child_ref) = &child.child_ref
                && !refs.contains(child_ref)
            {
                refs.push(child_ref.clone());
            }
        }
        refs
    }
}

impl BranchFork {
    fn into_value(self) -> Value {
        object([
            ("id", self.id.into()),
            ("parent_node", self.parent_node.into()),
            ("status", self.status.into()),
            ("value", self.value.into()),
            ("requested_children", self.requested_children.into()),
            ("requested_layers", self.requested_layers.into()),
            ("captured_layers", self.captured_layers.into()),
            ("parent_restore", self.parent_restore.into()),
            ("error", self.error.into()),
            ("parent_restore_error", self.parent_restore_error.into()),
            ("artifact_error", self.artifact_error.into()),
            (
                "children",
                Value::Array(
                    self.children
                        .into_iter()
                        .map(BranchChild::into_value)
                        .collect(),
                ),
            ),
        ])
    }
}

impl BranchChild {
    fn into_value(self) -> Value {
        let intervention: Map = self
            .intervention
            .into_iter()
            .map(|(key, value)| (key.to_string(), value.into()))
            .collect();
        object([
            ("index", self.index.into()),
            ("node_id", self.node_id.into()),
            ("status", self.status.into()),
            ("reward", self.reward.into()),
            ("reward_source", self.reward_source.into()),
            ("error", self.error.into()),
            ("cleanup_error", self.cleanup_error.into()),
            ("intervention", Value::Object(intervention)),
            ("artifacts_status", self.artifacts_status.into()),
            ("artifacts_path", self.artifacts_path.into()),
            ("ref", self.child_ref.into()),
        ])
    }
}

/// `tree.json` as fork rows; `None` when the trial never branched.
pub fn load_lineage(rollout_dir: &Path) -> Option<Lineage> {
    let path = rollout_dir.join("tree.json");
    if !path.is_file() {
        return None;
    }
    let raw = match load_json(&path) {
        Some(Value::Object(raw)) if get(&raw, "kind").as_str() == Some("benchflow-branch-tree") => {
            raw
        }
        _ => {
            return Some(Lineage::unreadable(
                "not a benchflow-branch-tree document".to_string(),
            ));
        }
    };
    let version = get(&raw, "schema_version");
    // Python's `!= 1`: `1`, `1.0` and `true` all equal 1.
    let is_one = match version {
        Value::Int(int) => int.is(1),
        Value::Float(number) => *number == 1.0,
        Value::Bool(flag) => *flag,
        _ => false,
    };
    if !is_one {
        return Some(Lineage::unreadable(format!(
            "unsupported schema_version {}",
            display_text(version)
        )));
    }
    let forks = get(&raw, "forks")
        .as_array()
        .unwrap_or_default()
        .iter()
        .filter_map(Value::as_object)
        .map(|fork| parse_fork(rollout_dir, fork))
        .collect();
    let nodes = get(&raw, "nodes").as_array().map_or(0, <[Value]>::len);
    Some(Lineage {
        nodes,
        forks,
        error: None,
    })
}

fn parse_fork(rollout_dir: &Path, fork: &Map) -> BranchFork {
    let id = optional_text(get(fork, "id")).unwrap_or_default();
    let snapshot = get(fork, "snapshot");
    let children = get(fork, "children")
        .as_array()
        .unwrap_or_default()
        .iter()
        .filter_map(Value::as_object)
        .map(|child| parse_child(rollout_dir, &id, child))
        .collect();
    BranchFork {
        parent_node: optional_text(get(fork, "parent_node")),
        status: optional_text(get(fork, "status")),
        value: finite_float(get(fork, "value")),
        requested_children: exact_int(get(fork, "requested_children")),
        requested_layers: text_list(snapshot.get("requested_layers")),
        captured_layers: text_list(snapshot.get("captured_layers")),
        parent_restore: optional_text(get(fork, "parent_restore")),
        error: error_record_text(get(fork, "error")),
        parent_restore_error: error_record_text(get(fork, "parent_restore_error")),
        artifact_error: error_record_text(get(fork, "artifact_error")),
        children,
        id,
    }
}

fn parse_child(rollout_dir: &Path, fork_id: &str, raw: &Map) -> BranchChild {
    let intervention = get(raw, "intervention");
    let artifacts = get(raw, "artifacts");
    let node_id = optional_text(get(raw, "node_id"));
    let artifacts_status = optional_text(artifacts.get("status"));
    let artifacts_path = optional_text(artifacts.get("path"));
    let child_ref = match &node_id {
        Some(node_id)
            if is_safe_id(fork_id)
                && is_safe_id(node_id)
                && artifacts_status.as_deref() == Some("available")
                && branch_child_dir(rollout_dir, artifacts_path.as_deref()).is_some() =>
        {
            Some(format!("{fork_id}/{node_id}"))
        }
        _ => None,
    };
    let text =
        |key: &str| Some(display_text(intervention.get(key))).filter(|text| !text.is_empty());
    BranchChild {
        index: exact_int(get(raw, "index")),
        node_id,
        status: optional_text(get(raw, "status")),
        reward: finite_float(get(raw, "reward")),
        reward_source: optional_text(get(raw, "reward_source")),
        error: error_record_text(get(raw, "error")),
        cleanup_error: error_record_text(get(raw, "cleanup_error")),
        intervention: [
            ("label", text("label")),
            ("requested", text("requested")),
            ("execution", text("execution")),
            ("evidence", text("evidence")),
        ],
        artifacts_status,
        artifacts_path,
        child_ref,
    }
}

/// `{"type", "code"}` error records read `Type (code)`; messages never
/// enter `tree.json` (`_error_record_text`).
fn error_record_text(value: &Value) -> Option<String> {
    let Value::Object(record) = value else {
        return error_text(value);
    };
    let kind = optional_text(get(record, "type")).filter(|kind| !kind.is_empty());
    let code = optional_text(get(record, "code"))
        .filter(|code| !code.is_empty())
        .map(|code| format!("({code})"));
    let text = [kind, code]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
    (!text.is_empty()).then_some(text)
}

/// A fork or node id safe to put in a link: `^[A-Za-z0-9][A-Za-z0-9_.-]*$`,
/// where Python's `$` also matches before one final newline.
fn is_safe_id(text: &str) -> bool {
    let body = text.strip_suffix('\n').unwrap_or(text);
    let mut bytes = body.bytes();
    bytes
        .next()
        .is_some_and(|first| first.is_ascii_alphanumeric())
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'.' | b'-'))
}

/// A child's archive folder, only when it resolves inside the trial folder
/// (symlinks followed) and holds an `observation.json`.
pub fn branch_child_dir(rollout_dir: &Path, relative: Option<&str>) -> Option<PathBuf> {
    let relative = relative.filter(|relative| !relative.is_empty())?;
    let root = std::fs::canonicalize(rollout_dir).ok()?;
    let candidate = std::fs::canonicalize(rollout_dir.join(relative)).ok()?;
    if candidate == root || !candidate.starts_with(&root) {
        return None;
    }
    candidate
        .join(BRANCH_OBSERVATION)
        .is_file()
        .then_some(candidate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_ids_follow_python_dollar() {
        assert!(is_safe_id("n1"));
        assert!(is_safe_id("00000000000000000000000000000002"));
        assert!(is_safe_id("n1\n"));
        assert!(!is_safe_id("n1\n\n"));
        assert!(!is_safe_id("../x"));
        assert!(!is_safe_id(".hidden"));
        assert!(!is_safe_id(""));
        assert!(!is_safe_id("a/b"));
    }

    #[test]
    fn error_records_show_type_and_code_only() {
        let record = |text: &str| error_record_text(&crate::pyjson::parse(text).unwrap());
        assert_eq!(
            record(r#"{"type": "RuntimeError", "code": "fork_failed", "message": "secret"}"#)
                .as_deref(),
            Some("RuntimeError (fork_failed)")
        );
        assert_eq!(record(r#"{"code": "x"}"#).as_deref(), Some("(x)"));
        assert_eq!(record(r#"{"type": null}"#), None);
        assert_eq!(record(r#""plain""#).as_deref(), Some("plain"));
        assert_eq!(record("null"), None);
    }

    #[test]
    fn child_archives_must_stay_inside_the_trial() {
        let dir = tempfile::tempdir().unwrap();
        let trial = dir.path().join("trial");
        std::fs::create_dir_all(trial.join("branches/f/children/n1")).unwrap();
        std::fs::write(trial.join("branches/f/children/n1/observation.json"), "{}").unwrap();
        std::fs::create_dir_all(dir.path().join("outside")).unwrap();
        std::fs::write(dir.path().join("outside/observation.json"), "{}").unwrap();
        assert!(branch_child_dir(&trial, Some("branches/f/children/n1")).is_some());
        assert!(branch_child_dir(&trial, Some("../outside")).is_none());
        assert!(branch_child_dir(&trial, Some(".")).is_none());
        assert!(branch_child_dir(&trial, Some("")).is_none());
        assert!(branch_child_dir(&trial, Some("branches/f/children/missing")).is_none());
        assert!(branch_child_dir(&trial, None).is_none());
    }
}
