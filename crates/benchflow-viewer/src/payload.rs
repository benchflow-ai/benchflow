//! Everything the run page shows for one trial, built from its folder
//! (0.8's `ViewerPayload`, `_build_acp_payload`, `_build_branch_child_payload`).
//!
//! The JSON shape is the contract with the page's scripts, which are 0.8's
//! unchanged; field names and order match 0.8 exactly.

use std::path::Path;

use crate::lineage::{BRANCH_OBSERVATION, Lineage, branch_child_dir, load_lineage};
use crate::metadata::{
    ErrorBanner, RolloutMetadata, RunStatus, Timing, load_rollout_metadata, normalize_metadata,
};
use crate::py;
use crate::pyjson::{Int, Map, Value, object};
use crate::rubric::{RubricReview, load_rubric};
use crate::steps::{Step, StepCounts, load_prompts, normalize_steps, trajectory_steps};
use crate::values::load_json;
use crate::verifier::{VerifierArtifacts, load_recovery, load_verifier};

/// Where a trial's trajectory lives, relative to the trial folder.
pub const TRAJECTORY_FILE: &str = "trajectory/acp_trajectory.jsonl";

/// The run page's data for one trial or branch child.
#[derive(Debug, Clone)]
pub struct ViewerPayload {
    pub rollout_name: String,
    pub meta: Meta,
    pub steps: Vec<Step>,
    pub verifier: VerifierArtifacts,
    pub rubric: Option<RubricReview>,
    pub lineage: Option<Lineage>,
}

/// The header and Metrics data of a payload.
#[derive(Debug, Clone)]
pub struct Meta {
    pub metadata: RolloutMetadata,
    pub counts: StepCounts,
    pub branch: Option<BranchOrigin>,
}

/// Where a branch child sits in its parent's `tree.json`.
#[derive(Debug, Clone)]
pub struct BranchOrigin {
    pub parent_rollout: String,
    pub fork_id: String,
    pub node_id: String,
    pub index: Option<Int>,
    pub parent_node: Option<String>,
    pub status: Option<String>,
    pub intervention: Option<String>,
}

impl ViewerPayload {
    pub fn into_value(self) -> Value {
        object([
            ("schema_version", 1i64.into()),
            ("rollout_name", self.rollout_name.into()),
            ("meta", self.meta.into_value()),
            (
                "steps",
                Value::Array(self.steps.into_iter().map(Step::into_value).collect()),
            ),
            ("verifier", self.verifier.into_value()),
            ("rubric", self.rubric.map(RubricReview::into_value).into()),
            ("lineage", self.lineage.map(Lineage::into_value).into()),
        ])
    }
}

impl Meta {
    fn new(
        metadata: RolloutMetadata,
        steps: &[Step],
        extra_errors: Vec<ErrorBanner>,
        branch: Option<BranchOrigin>,
    ) -> Meta {
        let mut metadata = metadata;
        metadata.errors.extend(extra_errors);
        Meta {
            metadata,
            counts: StepCounts::of(steps),
            branch,
        }
    }

    fn into_value(self) -> Value {
        let RolloutMetadata {
            task_name,
            agent_name,
            model,
            skill_mode,
            reward,
            usage,
            timing,
            errors,
            trajectory_source,
            partial_trajectory,
            started_at,
            finished_at,
            status,
            ..
        } = self.metadata;
        let duration_sec = timing.as_ref().and_then(|timing| timing.total);
        object([
            ("task_name", task_name.into()),
            ("agent_name", agent_name.into()),
            ("model", model.into()),
            ("skill_mode", skill_mode.into()),
            ("reward", reward.into()),
            ("usage", Value::Object(usage.values)),
            ("counts", self.counts.into_value()),
            (
                "timing",
                timing
                    .map(|Timing { values, .. }| Value::Object(values))
                    .into(),
            ),
            ("duration_sec", duration_sec.into()),
            (
                "errors",
                Value::Array(errors.into_iter().map(ErrorBanner::into_value).collect()),
            ),
            ("trajectory_source", trajectory_source.into()),
            ("partial_trajectory", partial_trajectory.into()),
            ("started_at", started_at.into()),
            ("finished_at", finished_at.into()),
            ("status", status.map(RunStatus::into_value).into()),
            ("branch", self.branch.map(BranchOrigin::into_value).into()),
        ])
    }
}

impl BranchOrigin {
    fn into_value(self) -> Value {
        object([
            ("parent_rollout", self.parent_rollout.into()),
            ("fork_id", self.fork_id.into()),
            ("node_id", self.node_id.into()),
            ("index", self.index.into()),
            ("parent_node", self.parent_node.into()),
            ("status", self.status.into()),
            ("intervention", self.intervention.into()),
        ])
    }
}

/// The payload for a trial folder (one that holds [`TRAJECTORY_FILE`]).
///
/// Every artifact is optional and untrusted: a missing or malformed file
/// leaves its part of the page empty instead of failing the page.
pub fn build_payload(rollout_dir: &Path) -> ViewerPayload {
    let steps = trajectory_steps(
        &rollout_dir.join(TRAJECTORY_FILE),
        load_prompts(rollout_dir),
    );
    let recovery = load_recovery(rollout_dir);
    let banners = recovery
        .as_ref()
        .map(|recovery| recovery.banners())
        .unwrap_or_default();
    let meta = Meta::new(load_rollout_metadata(rollout_dir), &steps, banners, None);
    let mut verifier = load_verifier(rollout_dir);
    verifier.recovery = recovery;
    ViewerPayload {
        rollout_name: py::name_of(rollout_dir),
        meta,
        steps,
        verifier,
        rubric: load_rubric(rollout_dir),
        lineage: load_lineage(rollout_dir),
    }
}

/// The payload of one branch child, named `<fork id>/<node id>`.
///
/// The ref is resolved only through the trial's own `tree.json`, so a
/// crafted ref never reaches the filesystem; `None` when the tree does not
/// list it as an openable child.
pub fn build_branch_child_payload(rollout_dir: &Path, child_ref: &str) -> Option<ViewerPayload> {
    let lineage = load_lineage(rollout_dir)?;
    let (fork, child) = lineage
        .forks
        .iter()
        .flat_map(|fork| fork.children.iter().map(move |child| (fork, child)))
        .find(|(_, child)| child.child_ref.as_deref() == Some(child_ref))?;
    let child_dir = branch_child_dir(rollout_dir, child.artifacts_path.as_deref())?;
    let Some(Value::Object(observation)) = load_json(&child_dir.join(BRANCH_OBSERVATION)) else {
        return None;
    };
    let field = |key: &str| crate::pyjson::get(&observation, key).clone();
    let events: Vec<std::io::Result<Map>> = match field("trajectory") {
        Value::Array(items) => items
            .into_iter()
            .filter_map(|item| match item {
                Value::Object(event) => Some(Ok(event)),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let parent = load_rollout_metadata(rollout_dir);
    let mut error = field("error");
    let mut verifier_error = field("verifier_error");
    if child.status.as_deref() == Some("unscored") {
        // The recorded failure is the missing verdict, not the agent.
        if !py::truthy(&verifier_error) {
            verifier_error = error;
        }
        error = Value::Null;
    }
    let result = object([
        ("task_name", parent.task_name.into()),
        ("agent_name", parent.agent_name.into()),
        ("model", parent.model.into()),
        ("skill_mode", parent.skill_mode.into()),
        (
            "rewards",
            child
                .reward
                .map(|reward| object([("reward", reward.into())]))
                .into(),
        ),
        ("error", error),
        ("verifier_error", verifier_error),
        ("export_error", field("export_error")),
        ("agent_result", field("native_usage")),
        ("trajectory_source", "branch observation".into()),
    ]);
    let timing = field("timing");
    let metadata = normalize_metadata(
        result.as_object().expect("built as an object"),
        Some(&timing).filter(|timing| timing.as_object().is_some()),
    );
    let steps = normalize_steps(events, None);
    let origin = BranchOrigin {
        parent_rollout: py::name_of(rollout_dir),
        fork_id: fork.id.clone(),
        node_id: child.node_id.clone().unwrap_or_default(),
        index: child.index.clone(),
        parent_node: fork.parent_node.clone(),
        status: child.status.clone(),
        intervention: child.label(),
    };
    // In-place children hand off the mounted sidecars; isolated children
    // are full trial folders with their own verifier/.
    let mounted = child_dir.join("mounted");
    let verifier = load_verifier(if mounted.is_dir() {
        &mounted
    } else {
        &child_dir
    });
    Some(ViewerPayload {
        rollout_name: format!(
            "{} / branch {}",
            py::name_of(rollout_dir),
            child.node_id.as_deref().unwrap_or("None")
        ),
        meta: Meta::new(metadata, &steps, Vec::new(), Some(origin)),
        steps,
        verifier,
        rubric: None,
        lineage: None,
    })
}

/// Every openable branch child's payload, keyed by ref, for pages that
/// work without a server (`_branch_child_payloads`).
pub fn branch_child_payloads(rollout_dir: &Path, lineage: Option<&Lineage>) -> Map {
    let mut payloads = Map::new();
    for child_ref in lineage.map(Lineage::child_refs).unwrap_or_default() {
        if let Some(payload) = build_branch_child_payload(rollout_dir, &child_ref) {
            payloads.insert(child_ref, payload.into_value());
        }
    }
    payloads
}
