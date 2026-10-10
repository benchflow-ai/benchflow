//! The integrated `scoring` block of `result.json` (0.8's
//! `benchflow.review.outcome.ScoringResult`, a strict pydantic model).
//!
//! A trial whose task has a rubric is scored by tests plus a rubric review;
//! the block records that verdict. The viewer only needs to know whether
//! the block is well formed and, if so, whether it is complete and passed.
//! A malformed block makes the trial unscored even beside a stale reward.

use crate::pyjson::{Int, Map, Value, get};

/// A valid scoring block.
#[derive(Debug, Clone, PartialEq)]
pub struct Scoring {
    /// `status == "complete"`; otherwise the block records a scoring error.
    pub complete: bool,
    pub passed: Option<bool>,
    pub error: Option<String>,
}

const FIELDS: [&str; 13] = [
    "schema_version",
    "policy",
    "status",
    "passed",
    "tests_pass",
    "all_blockers_pass",
    "failed_blockers",
    "verifier_reward",
    "rubric_reward",
    "reviewer_run",
    "revision",
    "error",
    "error_category",
];

/// `scoring_from_result` for a result whose `scoring` is not `null`:
/// `None` where 0.8 raises (a malformed block, or a complete verdict whose
/// reward envelope disagrees with it).
pub fn scoring_from_result(result: &Map) -> Option<Scoring> {
    let block = validate(get(result, "scoring"))?;
    if block.status_complete {
        // The trainer-facing results.jsonl carries a scalar `reward`.
        let envelope;
        let rewards = if !result.contains_key("rewards") && result.contains_key("reward") {
            envelope = Value::Object(Map::from_iter([(
                "reward".to_string(),
                get(result, "reward").clone(),
            )]));
            &envelope
        } else {
            get(result, "rewards")
        };
        let rewards = rewards.as_object()?;
        let verifier = block.verifier_reward?;
        let rubric = block.rubric_reward?;
        let reward = if block.passed == Some(true) {
            rubric
        } else {
            0.0
        };
        for (key, expected) in [
            ("reward", reward),
            ("verifier_reward", verifier),
            ("rubric_reward", rubric),
        ] {
            if key != "reward" && !rewards.contains_key(key) {
                continue;
            }
            if !reward_equals(get(rewards, key), expected) {
                return None;
            }
        }
    }
    Some(Scoring {
        complete: block.status_complete,
        passed: block.passed,
        error: block.error,
    })
}

/// Whether a stored reward is a finite number equal to `expected`
/// (`finite_reward(actual) is not None and actual == expected`, where
/// Python compares ints and floats exactly).
fn reward_equals(actual: &Value, expected: f64) -> bool {
    match actual {
        Value::Float(number) => number.is_finite() && *number == expected,
        Value::Int(int) => {
            int.to_f64().is_some()
                && expected.fract() == 0.0
                && expected.is_finite()
                && *int == Int::from_integral_f64(expected)
        }
        _ => false,
    }
}

struct Block {
    status_complete: bool,
    passed: Option<bool>,
    tests_pass: Option<bool>,
    all_blockers_pass: Option<bool>,
    failed_blockers: Vec<String>,
    verifier_reward: Option<f64>,
    rubric_reward: Option<f64>,
    reviewer_run: Option<String>,
    error: Option<String>,
    error_category: Option<String>,
}

/// Field types and the cross-field rules of `ScoringResult` (strict mode:
/// no coercion between booleans, numbers, and text).
fn validate(raw: &Value) -> Option<Block> {
    let map = raw.as_object()?;
    if map.keys().any(|key| !FIELDS.contains(&key.as_str())) {
        return None;
    }
    // `type(value) is int`, then `Literal[1]`.
    if let Some(version) = map.get("schema_version")
        && !matches!(version, Value::Int(int) if int.is(1))
    {
        return None;
    }
    if let Some(policy) = map.get("policy")
        && policy.as_str() != Some("tests-blockers-quality-v1")
    {
        return None;
    }
    let status_complete = match map.get("status")?.as_str()? {
        "complete" => true,
        "error" => false,
        _ => return None,
    };
    let failed_blockers = match get(map, "failed_blockers") {
        Value::Null if !map.contains_key("failed_blockers") => Vec::new(),
        Value::Array(items) => items
            .iter()
            .map(|item| item.as_str().map(str::to_string))
            .collect::<Option<Vec<_>>>()?,
        _ => return None,
    };
    let rubric_reward = optional_float(map, "rubric_reward")?;
    if rubric_reward.is_some_and(|reward| !(0.0..=1.0).contains(&reward)) {
        return None;
    }
    let revision = optional_text(map, "revision")?;
    if revision
        .as_deref()
        .is_some_and(|revision| !is_revision(revision))
    {
        return None;
    }
    let error_category = optional_text(map, "error_category")?;
    if error_category
        .as_deref()
        .is_some_and(|category| category != "reviewer_transport")
    {
        return None;
    }
    let block = Block {
        status_complete,
        passed: optional_bool(map, "passed")?,
        tests_pass: optional_bool(map, "tests_pass")?,
        all_blockers_pass: optional_bool(map, "all_blockers_pass")?,
        failed_blockers,
        verifier_reward: optional_float(map, "verifier_reward")?,
        rubric_reward,
        reviewer_run: optional_text(map, "reviewer_run")?,
        error: optional_text(map, "error")?,
        error_category,
    };
    consistent(&block).then_some(block)
}

/// `consistent_verdict`: an error carries an explanation and no verdict; a
/// complete verdict carries both gates and both rewards, and they agree.
fn consistent(block: &Block) -> bool {
    if !block.status_complete {
        return block
            .error
            .as_deref()
            .is_some_and(|error| !crate::py::strip(error).is_empty())
            && block.passed.is_none()
            && block.rubric_reward.is_none();
    }
    if block.error.is_some() || block.error_category.is_some() {
        return false;
    }
    let (Some(tests_pass), Some(all_blockers_pass)) = (block.tests_pass, block.all_blockers_pass)
    else {
        return false;
    };
    let (Some(verifier_reward), Some(_)) = (block.verifier_reward, block.rubric_reward) else {
        return false;
    };
    let mut unique = block.failed_blockers.clone();
    unique.sort();
    unique.dedup();
    tests_pass == (verifier_reward == 1.0)
        && block.passed == Some(tests_pass && all_blockers_pass)
        && all_blockers_pass == block.failed_blockers.is_empty()
        && unique.len() == block.failed_blockers.len()
        && block
            .reviewer_run
            .as_deref()
            .is_some_and(|run| !crate::py::strip(run).is_empty())
}

/// `^scoring/[A-Za-z0-9_-]+[.]json$` (pydantic anchors `$` at the very end).
fn is_revision(text: &str) -> bool {
    text.strip_prefix("scoring/")
        .and_then(|rest| rest.strip_suffix(".json"))
        .is_some_and(|stem| {
            !stem.is_empty()
                && stem
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
        })
}

/// An optional field: `Some(None)` when absent or `null`, `None` when the
/// field has the wrong type.
fn optional_bool(map: &Map, key: &str) -> Option<Option<bool>> {
    match get(map, key) {
        Value::Null => Some(None),
        Value::Bool(flag) => Some(Some(*flag)),
        _ => None,
    }
}

fn optional_text(map: &Map, key: &str) -> Option<Option<String>> {
    match get(map, key) {
        Value::Null => Some(None),
        Value::Str(text) => Some(Some(text.clone())),
        _ => None,
    }
}

/// A finite float field (strict mode takes ints too, never booleans).
fn optional_float(map: &Map, key: &str) -> Option<Option<f64>> {
    let number = match get(map, key) {
        Value::Null => return Some(None),
        Value::Int(int) => int.to_f64()?,
        Value::Float(number) => *number,
        _ => return None,
    };
    number.is_finite().then_some(Some(number))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::parse;

    const COMPLETE: &str = r#"{"schema_version": 1, "policy": "tests-blockers-quality-v1", "status": "complete",
        "passed": true, "tests_pass": true, "all_blockers_pass": true, "failed_blockers": [],
        "verifier_reward": 1.0, "rubric_reward": 0.75, "reviewer_run": "reviews/abc"}"#;

    fn result(rewards: &str, scoring: &str) -> Map {
        let text = format!(r#"{{"rewards": {rewards}, "scoring": {scoring}}}"#);
        parse(&text).unwrap().as_object().unwrap().clone()
    }

    #[test]
    fn a_complete_verdict_must_match_the_reward_envelope() {
        let scoring = scoring_from_result(&result(r#"{"reward": 0.75}"#, COMPLETE)).unwrap();
        assert!(scoring.complete && scoring.passed == Some(true));
        assert_eq!(
            scoring_from_result(&result(r#"{"reward": 0.5}"#, COMPLETE)),
            None
        );
        assert_eq!(scoring_from_result(&result("null", COMPLETE)), None);
    }

    #[test]
    fn malformed_blocks_are_refused() {
        assert_eq!(
            scoring_from_result(&result(r#"{"reward": 1}"#, r#"{"status": "complete"}"#)),
            None
        );
        assert_eq!(
            scoring_from_result(&result(
                r#"{"reward": 1}"#,
                r#"{"status": "error", "error": "  "}"#
            )),
            None
        );
        assert_eq!(scoring_from_result(&result(r#"{"reward": 1}"#, "[]")), None);
        let with_extra = COMPLETE.replace("\"status\"", "\"extra\": 1, \"status\"");
        assert_eq!(
            scoring_from_result(&result(r#"{"reward": 0.75}"#, &with_extra)),
            None
        );
    }

    #[test]
    fn a_scoring_error_is_valid_but_not_complete() {
        let scoring = scoring_from_result(&result(
            r#"{"reward": 1}"#,
            r#"{"status": "error", "error": "reviewer crashed"}"#,
        ))
        .unwrap();
        assert!(!scoring.complete);
        assert_eq!(scoring.error.as_deref(), Some("reviewer crashed"));
    }

    #[test]
    fn revisions_are_anchored() {
        assert!(is_revision("scoring/a1-b_2.json"));
        assert!(!is_revision("scoring/../x.json"));
        assert!(!is_revision("scoring/a.json\n"));
        assert!(!is_revision("scoring/.json"));
    }
}
