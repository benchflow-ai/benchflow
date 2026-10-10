//! The rubric review of a trial (0.8's `_load_rubric`): the automatic review
//! inside the trial when its task has a rubric, else the `bench review`
//! report for it in the nearest `review*` folder above it.

use std::path::{Path, PathBuf};

use crate::py;
use crate::pyjson::{Int, Map, Value, get, object};
use crate::values::{display_text, exact_int, load_json, optional_text};

/// How many folders above a trial are searched for `review*` folders.
const REVIEW_SEARCH_DEPTH: usize = 4;
const REVIEW_REPORT: &str = "review_report.json";

#[derive(Debug, Clone)]
pub struct RubricReview {
    pub reviewer_model: Option<String>,
    pub review_valid: bool,
    /// The report's scoring block as recorded.
    pub scoring: Map,
    pub summary: String,
    pub criteria: Vec<RubricCriterion>,
    /// The report file the review came from.
    pub source: String,
}

/// One criterion as the reviewer judged it.
#[derive(Debug, Clone)]
pub struct RubricCriterion {
    pub name: String,
    /// `true` for a blocker, `false` for a scored criterion, `None` for a
    /// legacy (v0.1) criterion that only has an outcome.
    pub blocker: Option<bool>,
    pub weight: Option<Int>,
    pub outcome: Option<String>,
    pub score: Option<Int>,
    pub explanation: String,
}

impl RubricReview {
    pub fn into_value(self) -> Value {
        object([
            ("reviewer_model", self.reviewer_model.into()),
            ("review_valid", self.review_valid.into()),
            ("scoring", Value::Object(self.scoring)),
            ("summary", self.summary.into()),
            (
                "criteria",
                Value::Array(
                    self.criteria
                        .into_iter()
                        .map(|criterion| {
                            object([
                                ("name", criterion.name.into()),
                                ("blocker", criterion.blocker.into()),
                                ("weight", criterion.weight.into()),
                                ("outcome", criterion.outcome.into()),
                                ("score", criterion.score.into()),
                                ("explanation", criterion.explanation.into()),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("source", self.source.into()),
        ])
    }
}

/// Criteria from a report's checks and the criteria metadata (`_criteria`).
fn criteria(checks: &Value, metadata: &Value) -> Vec<RubricCriterion> {
    let empty = Map::new();
    let checks = checks.as_object().unwrap_or(&empty);
    metadata
        .as_array()
        .unwrap_or_default()
        .iter()
        .filter_map(Value::as_object)
        .map(|meta| {
            let name = display_text(get(meta, "name"));
            let check = get(checks, &name);
            let blocker = get(meta, "blocker");
            RubricCriterion {
                blocker: (!blocker.is_null()).then(|| py::truthy(blocker)),
                weight: exact_int(get(meta, "weight")),
                outcome: optional_text(check.get("outcome")),
                score: exact_int(check.get("score")),
                explanation: display_text(check.get("explanation")),
                name,
            }
        })
        .collect()
}

fn reviewer_model(report: &Map) -> Option<String> {
    get(report, "reviewer")
        .as_object()
        .and_then(|reviewer| optional_text(get(reviewer, "model")))
}

fn scoring_map(value: &Value) -> Map {
    value.as_object().cloned().unwrap_or_default()
}

/// The last entry for `rollout_name` in one `bench review` report.
fn rubric_from_report(path: &Path, rollout_name: &str) -> Option<RubricReview> {
    let Value::Object(report) = load_json(path)? else {
        return None;
    };
    let trial = get(&report, "trials")
        .as_array()
        .unwrap_or_default()
        .iter()
        .filter_map(Value::as_object)
        .rfind(|trial| get(trial, "trial_name").as_str() == Some(rollout_name))?;
    Some(RubricReview {
        reviewer_model: reviewer_model(&report),
        review_valid: py::truthy(get(trial, "review_valid")),
        scoring: scoring_map(get(trial, "scoring")),
        summary: display_text(get(trial, "summary")),
        criteria: criteria(get(trial, "checks"), get(trial, "criterion_metadata")),
        source: py::path_text(path),
    })
}

/// The automatic review a task's rubric triggered: `result.json` names the
/// final report (`scoring.revision`), whose `rubric_snapshot` names the
/// criteria. Both must stay inside the trial folder.
fn automatic_rubric(rollout_dir: &Path) -> Option<RubricReview> {
    let result = load_json(&rollout_dir.join("result.json"))?;
    let revision = result
        .get("scoring")
        .get("revision")
        .as_str()
        .filter(|revision| !revision.is_empty())?;
    let root = std::fs::canonicalize(rollout_dir).ok()?;
    let inside = |relative: &str| {
        std::fs::canonicalize(rollout_dir.join(relative))
            .ok()
            .filter(|path| path.starts_with(&root))
    };
    let path = inside(revision)?;
    let Value::Object(report) = load_json(&path)? else {
        return None;
    };
    let rubric = get(&report, "rubric_snapshot")
        .as_str()
        .and_then(inside)
        .and_then(|snapshot| load_json(&snapshot));
    let metadata = rubric
        .as_ref()
        .map(|rubric| rubric.get("criteria"))
        .filter(|criteria| criteria.as_array().is_some());
    Some(RubricReview {
        reviewer_model: reviewer_model(&report),
        review_valid: py::truthy(get(&report, "review_valid")),
        scoring: scoring_map(get(&report, "scoring")),
        summary: display_text(get(&report, "summary")),
        criteria: criteria(get(&report, "checks"), metadata.unwrap_or(&Value::Null)),
        source: py::path_text(&path),
    })
}

/// `parent / name` as `pathlib` writes it (no `./` in front).
fn child(parent: &Path, name: &std::ffi::OsStr) -> PathBuf {
    if parent == Path::new(".") {
        PathBuf::from(name)
    } else {
        parent.join(name)
    }
}

/// Every `review_report.json` under a review folder, sorted (`rglob`,
/// which does not descend into symlinked folders).
fn review_reports(review_dir: &Path) -> Vec<PathBuf> {
    let mut reports: Vec<PathBuf> = walkdir::WalkDir::new(review_dir)
        .follow_links(false)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.depth() > 0 && entry.file_name() == REVIEW_REPORT)
        .map(|entry| entry.into_path())
        .collect();
    reports.sort();
    reports
}

/// The trial's rubric review, or `None` when it was never reviewed.
///
/// Above the trial, each level's `review*` folders are read in sorted
/// order; the last valid entry for this trial wins, and an invalid entry
/// (no scoring) never shadows a valid one.
pub fn load_rubric(rollout_dir: &Path) -> Option<RubricReview> {
    if let Some(automatic) = automatic_rubric(rollout_dir) {
        return Some(automatic);
    }
    let name = py::name_of(rollout_dir);
    let mut ancestor = rollout_dir.to_path_buf();
    for _ in 0..REVIEW_SEARCH_DEPTH {
        ancestor = py::parent_of(&ancestor);
        let mut review_dirs: Vec<PathBuf> = std::fs::read_dir(&ancestor)
            .map(|entries| {
                entries
                    .filter_map(Result::ok)
                    .filter(|entry| entry.file_name().as_encoded_bytes().starts_with(b"review"))
                    .map(|entry| child(&ancestor, &entry.file_name()))
                    .collect()
            })
            .unwrap_or_default();
        review_dirs.sort();
        let mut found: Option<RubricReview> = None;
        for review_dir in review_dirs.iter().filter(|dir| dir.is_dir()) {
            for report in review_reports(review_dir) {
                if let Some(rubric) = rubric_from_report(&report, &name)
                    && (!rubric.scoring.is_empty() || found.is_none())
                {
                    found = Some(rubric);
                }
            }
        }
        if found.is_some() {
            return found;
        }
        if ancestor == py::parent_of(&ancestor) {
            break;
        }
    }
    None
}
