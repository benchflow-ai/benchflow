//! A trial's `result.json` and `timing.json`, normalized once for both the
//! run list and the run page (0.8's `_normalize_metadata`).

use std::path::Path;

use crate::classify::{
    assessment_status, assessment_withholds_score, classify_error, classify_verifier_error,
};
use crate::py;
use crate::pyjson::{self, Int, Map, Value, get, object};
use crate::scoring::scoring_from_result;
use crate::values::{
    error_text, finite_float, json_object, load_json, load_object, nonnegative_int, optional_bool,
    optional_text,
};

/// `result.json` fields holding a diagnostic about how a trial ended, in
/// 0.8's `DIAGNOSTIC_REGISTRY` order.
pub const DIAGNOSTIC_KEYS: [&str; 9] = [
    "usage_limit_info",
    "integration_failure_info",
    "idle_timeout_info",
    "agent_timeout_info",
    "sandbox_startup_info",
    "transport_error_info",
    "verifier_timeout_info",
    "api_error_info",
    "suspected_api_error_info",
];

/// The usage counters `agent_result` carries, normalized to whole numbers.
const USAGE_INT_FIELDS: [&str; 8] = [
    "n_tool_calls",
    "n_skill_invocations",
    "n_prompts",
    "n_input_tokens",
    "n_output_tokens",
    "n_cache_read_tokens",
    "n_cache_creation_tokens",
    "total_tokens",
];

/// The normalized projection of `result.json` and `timing.json`.
#[derive(Debug, Clone, Default)]
pub struct RolloutMetadata {
    pub task_name: Option<String>,
    pub agent_name: Option<String>,
    pub model: Option<String>,
    pub skill_mode: Option<String>,
    pub reward: Option<f64>,
    pub usage: Usage,
    pub timing: Option<Timing>,
    pub n_tool_calls: Option<Int>,
    pub errors: Vec<ErrorBanner>,
    pub has_error: bool,
    pub trajectory_source: Option<String>,
    pub partial_trajectory: Option<bool>,
    pub started_at: Option<String>,
    pub finished_at: Option<String>,
    pub status: Option<RunStatus>,
}

/// `agent_result`, with its counters, cost, and sources cleaned.
#[derive(Debug, Clone, Default)]
pub struct Usage {
    /// Every key `agent_result` had, in its order, values normalized.
    pub values: Map,
    pub n_tool_calls: Option<Int>,
    pub total_tokens: Option<Int>,
    pub cost_usd: Option<f64>,
}

/// Phase timings in seconds; anything that is not a finite number is `null`.
#[derive(Debug, Clone)]
pub struct Timing {
    pub values: Map,
    pub total: Option<f64>,
}

/// A line above the trace: an error, or a diagnostic shown as information.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorBanner {
    pub label: String,
    pub text: String,
    pub level: Option<&'static str>,
}

impl ErrorBanner {
    pub fn into_value(self) -> Value {
        let mut map = Map::new();
        map.insert("label".into(), self.label.into());
        map.insert("text".into(), self.text.into());
        if let Some(level) = self.level {
            map.insert("level".into(), level.into());
        }
        Value::Object(map)
    }
}

/// Execution and assessment, reported separately: whether the agent's run
/// finished, and whether a verdict was recorded. A verifier failure after a
/// clean run is "completed but unscored", never an agent failure.
#[derive(Debug, Clone)]
pub struct RunStatus {
    /// `completed`, `errored`, or `timed_out`.
    pub execution: &'static str,
    pub execution_detail: Option<String>,
    /// `scored` or `unscored`.
    pub assessment: &'static str,
    pub assessment_detail: Option<String>,
    pub summary: String,
    pub scoring: Option<Map>,
}

impl RunStatus {
    pub fn into_value(self) -> Value {
        object([
            ("execution", self.execution.into()),
            ("execution_detail", self.execution_detail.into()),
            ("assessment", self.assessment.into()),
            ("assessment_detail", self.assessment_detail.into()),
            ("summary", self.summary.into()),
            ("scoring", self.scoring.into()),
        ])
    }
}

impl Timing {
    fn from_value(raw: Option<&Value>) -> Option<Timing> {
        let raw = raw?.as_object()?;
        let values: Map = raw
            .iter()
            .map(|(key, value)| (key.clone(), finite_float(value).into()))
            .collect();
        let total = values.get("total").and_then(|total| match total {
            Value::Float(total) => Some(*total),
            _ => None,
        });
        Some(Timing { values, total })
    }
}

/// `_normalize_usage`: counters become whole numbers or `null`, the cost a
/// finite number or `null`, the sources text or `null`.
fn normalize_usage(raw: &Value) -> Usage {
    let mut values = json_object(raw);
    for key in USAGE_INT_FIELDS {
        let normalized = nonnegative_int(pyjson::get(&values, key));
        if let Some(slot) = values.get_mut(key) {
            *slot = normalized.into();
        }
    }
    let count = |key: &str| match pyjson::get(&values, key) {
        Value::Int(int) => Some(int.clone()),
        _ => None,
    };
    let n_tool_calls = count("n_tool_calls");
    let total_tokens = count("total_tokens");
    let cost_usd = finite_float(pyjson::get(&values, "cost_usd"));
    if let Some(slot) = values.get_mut("cost_usd") {
        *slot = cost_usd.into();
    }
    for key in ["usage_source", "price_source"] {
        let text = optional_text(pyjson::get(&values, key));
        if let Some(slot) = values.get_mut(key) {
            *slot = text.into();
        }
    }
    Usage {
        values,
        n_tool_calls,
        total_tokens,
        cost_usd,
    }
}

/// `_category_text`: `agent_timeout` reads `agent timeout`.
fn category_text(category: Option<&str>) -> Option<String> {
    category
        .filter(|category| !category.is_empty())
        .map(|category| category.replace('_', " "))
}

/// Text that is set and not empty (Python's `text or ...`).
fn nonempty(text: Option<String>) -> Option<String> {
    text.filter(|text| !text.is_empty())
}

/// Split one `result.json` into execution and assessment status (`_run_status`).
fn run_status(result: &Map, reward: Option<f64>) -> RunStatus {
    let error = error_text(get(result, "error"));
    let mut execution = "completed";
    let mut execution_detail = None;
    if let Some(error) = &error {
        let category = nonempty(optional_text(get(result, "error_category")))
            .or_else(|| classify_error(error).map(str::to_string));
        execution = if category
            .as_deref()
            .is_some_and(|category| category.contains("timeout"))
        {
            "timed_out"
        } else {
            "errored"
        };
        execution_detail = category_text(category.as_deref());
    }

    let mut assessment = "unscored";
    let mut scoring_block = None;
    let assessment_detail = if assessment_withholds_score(result) {
        let mut detail = format!(
            "assessment {}",
            assessment_status(result).unwrap_or_default()
        );
        if let Some(reason) = get(result, "assessment")
            .get("reason")
            .as_str()
            .filter(|reason| !reason.is_empty())
        {
            detail.push_str(&format!(" ({})", reason.replace('_', " ")));
        }
        Some(detail)
    } else if !get(result, "scoring").is_null() {
        scoring_block = Some(json_object(get(result, "scoring"))).filter(|block| !block.is_empty());
        match scoring_from_result(result) {
            None => Some("malformed scoring block".to_string()),
            Some(scoring) if scoring.complete => {
                assessment = "scored";
                Some(format!(
                    "tests + rubric, {}",
                    if scoring.passed == Some(true) {
                        "passed"
                    } else {
                        "not passed"
                    }
                ))
            }
            Some(scoring) => Some(format!(
                "scoring error: {}",
                scoring.error.as_deref().unwrap_or("None")
            )),
        }
    } else if reward.is_some() {
        assessment = "scored";
        None
    } else if let Some(verifier_error) =
        error_text(get(result, "verifier_error")).filter(|text| !text.is_empty())
    {
        let category = nonempty(optional_text(get(result, "verifier_error_category")))
            .or_else(|| classify_verifier_error(&verifier_error).map(str::to_string));
        Some(category_text(category.as_deref()).unwrap_or_else(|| "verifier error".to_string()))
    } else if error.is_some() {
        Some("no verdict after the execution error".to_string())
    } else {
        Some("no reward recorded".to_string())
    };

    let label = match execution {
        "timed_out" => "timed out",
        other => other,
    };
    let summary = match (assessment == "scored", error.is_none()) {
        (true, true) => "completed and scored".to_string(),
        (true, false) => format!("{label} but scored"),
        (false, true) => "completed but unscored".to_string(),
        (false, false) => format!("{label}, unscored"),
    };
    RunStatus {
        execution,
        execution_detail,
        assessment,
        assessment_detail,
        summary,
        scoring: scoring_block,
    }
}

/// Normalize a result document and the preferred timing (`_normalize_metadata`).
pub fn normalize_metadata(result: &Map, timing: Option<&Value>) -> RolloutMetadata {
    // Other harnesses keep the rewards in `verifier_result.rewards`.
    let mut reward = get(result, "rewards")
        .as_object()
        .or_else(|| get(result, "verifier_result").get("rewards").as_object())
        .and_then(|rewards| finite_float(get(rewards, "reward")));
    if assessment_withholds_score(result) {
        // A pending or unassessable assessment has no reward yet.
        reward = None;
    }
    let usage = normalize_usage(get(result, "agent_result"));
    let timing = Timing::from_value(timing);

    let agent_error = error_text(get(result, "error"));
    let verifier_error = error_text(get(result, "verifier_error"));
    let export_error = error_text(get(result, "export_error"));
    let has_error = [&agent_error, &verifier_error, &export_error]
        .iter()
        .any(|error| error.as_deref().is_some_and(|text| !text.is_empty()));
    let mut errors = Vec::new();
    if let Some(text) = agent_error {
        let label = nonempty(optional_text(get(result, "error_category")))
            .unwrap_or_else(|| "error".to_string());
        errors.push(ErrorBanner {
            label,
            text,
            level: None,
        });
    }
    if let Some(text) = verifier_error {
        let label = nonempty(optional_text(get(result, "verifier_error_category")))
            .unwrap_or_else(|| "verifier error".to_string());
        errors.push(ErrorBanner {
            label,
            text,
            level: None,
        });
    }
    if let Some(text) = export_error {
        errors.push(ErrorBanner {
            label: "export error".to_string(),
            text,
            level: None,
        });
    }
    for key in DIAGNOSTIC_KEYS {
        let value = get(result, key);
        if py::truthy(value) {
            errors.push(ErrorBanner {
                label: key.strip_suffix("_info").unwrap_or(key).replace('_', " "),
                text: pyjson::dumps(value),
                level: Some(if has_error { "error" } else { "info" }),
            });
        }
    }

    let top_level_tool_calls = nonnegative_int(get(result, "n_tool_calls"));
    RolloutMetadata {
        task_name: optional_text(get(result, "task_name")),
        agent_name: nonempty(optional_text(get(result, "agent_name")))
            .or_else(|| optional_text(get(result, "agent"))),
        model: optional_text(get(result, "model")),
        skill_mode: optional_text(get(result, "skill_mode")),
        reward,
        n_tool_calls: top_level_tool_calls.or_else(|| usage.n_tool_calls.clone()),
        usage,
        timing,
        errors,
        has_error,
        trajectory_source: optional_text(get(result, "trajectory_source")),
        partial_trajectory: optional_bool(get(result, "partial_trajectory")),
        started_at: optional_text(get(result, "started_at")),
        finished_at: optional_text(get(result, "finished_at")),
        status: (!result.is_empty()).then(|| run_status(result, reward)),
    }
}

/// Read and normalize a trial's metadata (`_load_rollout_metadata`):
/// `timing.json` when it is an object, else the timing inside `result.json`.
pub fn load_rollout_metadata(rollout_dir: &Path) -> RolloutMetadata {
    let result = load_object(&rollout_dir.join("result.json"));
    let preferred =
        load_json(&rollout_dir.join("timing.json")).filter(|timing| timing.as_object().is_some());
    let timing = preferred
        .as_ref()
        .or_else(|| Some(get(&result, "timing")).filter(|timing| timing.as_object().is_some()));
    normalize_metadata(&result, timing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pyjson::parse;

    fn status(text: &str) -> (String, Option<String>, String, Option<String>, String) {
        let result = parse(text).unwrap().as_object().unwrap().clone();
        let metadata = normalize_metadata(&result, None);
        let status = metadata.status.unwrap();
        (
            status.execution.to_string(),
            status.execution_detail,
            status.assessment.to_string(),
            status.assessment_detail,
            status.summary,
        )
    }

    #[test]
    fn execution_and_assessment_are_separate() {
        assert_eq!(
            status(r#"{"rewards": {"reward": 1.0}}"#),
            (
                "completed".into(),
                None,
                "scored".into(),
                None,
                "completed and scored".into()
            )
        );
        assert_eq!(
            status(r#"{"verifier_error": "verifier crashed: Failed to get session command: "}"#),
            (
                "completed".into(),
                None,
                "unscored".into(),
                Some("verifier infra".into()),
                "completed but unscored".into()
            )
        );
        assert_eq!(
            status(r#"{"error": "agent timed out", "error_category": "agent_timeout"}"#),
            (
                "timed_out".into(),
                Some("agent timeout".into()),
                "unscored".into(),
                Some("no verdict after the execution error".into()),
                "timed out, unscored".into()
            )
        );
        assert_eq!(
            status(r#"{"rewards": {"reward": 1.0}, "assessment": {"status": "pending", "reason": "awaiting_reviewer"}}"#).3,
            Some("assessment pending (awaiting reviewer)".into())
        );
    }

    #[test]
    fn diagnostics_are_info_unless_the_trial_failed() {
        let result = parse(r#"{"idle_timeout_info": {"idle_timeout_sec": 30}}"#).unwrap();
        let banners = normalize_metadata(result.as_object().unwrap(), None).errors;
        assert_eq!(banners[0].label, "idle timeout");
        assert_eq!(banners[0].level, Some("info"));
        assert_eq!(banners[0].text, r#"{"idle_timeout_sec": 30}"#);
    }

    #[test]
    fn usage_counters_are_cleaned_in_place() {
        let result = parse(r#"{"agent_result": {"zeta": 1, "total_tokens": "42", "n_prompts": -1, "cost_usd": "0.25"}}"#).unwrap();
        let usage = normalize_metadata(result.as_object().unwrap(), None).usage;
        assert_eq!(
            pyjson::dumps(&Value::Object(usage.values)),
            r#"{"zeta": 1, "total_tokens": 42, "n_prompts": null, "cost_usd": 0.25}"#
        );
        assert_eq!(usage.total_tokens, Some(Int::Small(42)));
    }
}
