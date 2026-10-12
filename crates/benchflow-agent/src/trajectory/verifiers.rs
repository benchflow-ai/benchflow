//! The Verifiers / ORS record: one scored rollout as the row prime-rl and
//! Verifiers train on (0.8's `trajectories/export.py`).
//!
//! The prompts and events become chat messages: prompts and
//! `user_message`s are `user` messages; `agent_message`s, `agent_thought`s,
//! tool calls (`[tool_call: <title>]` plus the text of their content) and
//! oracle runs (`[oracle: <command>]`) are `assistant` messages. The
//! messages up to the first `assistant` one are the record's `prompt`, the
//! rest its `completion`. A subagent's events are left out and counted in
//! `info.acp_projection`.
//!
//! The reward goes through 0.8's ORS conversion: a headline outside 0..1 is
//! recorded as `0.0` with `reward_valid` false; the other numeric rewards
//! and any `rubric` scores become `metrics`.

use serde_json::{Map, Value, json};

use super::export_common::{
    acp_projection_coverage, is_attributed_child, py_float_repr, py_str, str_or, str_or_empty,
    truthy,
};
use super::jsonl::to_python_json;
use super::redact::Redactor;

/// What a Verifiers record says about its rollout besides the events.
#[derive(Debug, Clone, Copy, Default)]
pub struct VerifiersMeta<'a> {
    /// The task's name; also the record's `environment`.
    pub task_id: &'a str,
    /// The prompts handed to the agent before any event was recorded.
    pub prompts: &'a [String],
    /// The verifier's reward map (`reward`, other values, an optional
    /// `rubric` list), or `None` when the trial was not scored.
    pub rewards: Option<&'a Map<String, Value>>,
    /// Why the verifier could not score the trial.
    pub error: Option<&'a str>,
    pub model: Option<&'a str>,
}

/// The record for `events` (each in its `acp_trajectory.jsonl` form).
pub fn to_verifiers_record(events: &[Value], meta: &VerifiersMeta<'_>) -> Value {
    let messages = to_messages(events, meta.prompts);
    let split = messages
        .iter()
        .position(|m| m["role"] == "assistant")
        .unwrap_or(messages.len());
    let (prompt, completion) = messages.split_at(split);
    let (reward, items, error) = verify_result(meta.rewards, meta.error);
    let (reward, valid, metadata) = to_ors(reward, &items, error);
    let mut record = json!({
        "example_id": 0,
        "prompt": prompt,
        "completion": completion,
        "reward": reward,
        "metrics": metadata["items"],
        "is_completed": true,
        "is_truncated": false,
        "info": {
            "task_id": meta.task_id,
            "environment": meta.task_id,
            "model": meta.model.unwrap_or_default(),
            "reward_valid": valid,
            "reward_metadata": metadata,
        },
    });
    if let Some(coverage) = acp_projection_coverage(events) {
        record["info"]["acp_projection"] = coverage;
    }
    record
}

/// The record as `trainer/verifiers.jsonl` holds it: redacted, one line.
pub fn verifiers_line(record: &Value, redactor: &Redactor) -> String {
    let mut clean = record.clone();
    // 0.8 redacts the record, reads it back, and redacts it again as it
    // writes; doing both keeps any rule that is not idempotent in step.
    redactor.redact_json(&mut clean);
    redactor.redact_json(&mut clean);
    super::export_common::cap_long_strings(
        &mut clean,
        super::export_common::DERIVED_TEXT_CAP,
        "trajectory/acp_trajectory.jsonl",
    );
    let mut line = to_python_json(&clean);
    line.push('\n');
    line
}

fn message(role: &str, content: String) -> Value {
    json!({"role": role, "content": content})
}

/// 0.8's `acp_events_to_messages`.
fn to_messages(events: &[Value], prompts: &[String]) -> Vec<Value> {
    let mut messages: Vec<Value> = prompts
        .iter()
        .filter(|p| !p.is_empty())
        .map(|p| message("user", p.clone()))
        .collect();
    for event in events.iter().filter_map(Value::as_object) {
        if is_attributed_child(event) {
            continue;
        }
        match event.get("type").and_then(Value::as_str) {
            Some("user_message") => {
                let text = str_or_empty(event, "text");
                if !text.is_empty() {
                    messages.push(message("user", text));
                }
            }
            Some("agent_message" | "agent_thought") => {
                let text = str_or_empty(event, "text");
                if !text.is_empty() {
                    messages.push(message("assistant", text));
                }
            }
            Some("tool_call") => messages.push(message("assistant", tool_call_text(event))),
            Some("oracle") => {
                let command = str_or(event, "command", "oracle");
                messages.push(message("assistant", format!("[oracle: {command}]")));
            }
            _ => {}
        }
    }
    messages
}

/// 0.8's `_tool_call_to_content`: the title, then each content block's
/// `text` (or its `content`, through `str()`), one per line.
fn tool_call_text(event: &Map<String, Value>) -> String {
    let title = match event.get("title").filter(|t| truthy(t)) {
        Some(title) => py_str(title),
        None => str_or(event, "kind", "tool_call"),
    };
    let mut parts = vec![format!("[tool_call: {title}]")];
    match event.get("content") {
        Some(Value::Array(items)) => {
            for item in items.iter().filter_map(Value::as_object) {
                let text = [item.get("text"), item.get("content")]
                    .into_iter()
                    .flatten()
                    .find(|t| truthy(t));
                if let Some(text) = text {
                    parts.push(py_str(text));
                }
            }
        }
        Some(Value::String(s)) if !s.is_empty() => parts.push(s.clone()),
        _ => {}
    }
    parts.join("\n")
}

/// A JSON number that is not a bool, as a float (Python's
/// `isinstance(v, (int, float)) and not isinstance(v, bool)`).
fn number(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

/// 0.8's `reward_map_to_verify_result`: the headline reward, the per-item
/// scores, and the error.
fn verify_result(
    rewards: Option<&Map<String, Value>>,
    error: Option<&str>,
) -> (f64, Vec<(String, f64)>, Option<String>) {
    let Some(rewards) = rewards else {
        return (
            0.0,
            Vec::new(),
            Some(error.unwrap_or("no rewards").to_string()),
        );
    };
    // `float(True)` is 1.0: the headline accepts a bool, the items do not.
    let headline = match rewards.get("reward") {
        Some(Value::Bool(b)) => f64::from(u8::from(*b)),
        other => number(other).unwrap_or(0.0),
    };
    let mut items: Vec<(String, f64)> = Vec::new();
    let mut set = |name: String, score: f64| match items.iter_mut().find(|(n, _)| *n == name) {
        Some(slot) => slot.1 = score,
        None => items.push((name, score)),
    };
    for (key, value) in rewards {
        if key == "reward" || key == "rubric" {
            continue;
        }
        if let Some(score) = number(Some(value)) {
            set(key.clone(), score);
        }
    }
    if let Some(Value::Array(rubric)) = rewards.get("rubric") {
        for (i, item) in rubric.iter().enumerate() {
            let Some(item) = item.as_object() else {
                continue;
            };
            let Some(score) = number(item.get("score")) else {
                continue;
            };
            set(str_or(item, "name", &format!("rubric_{i}")), score);
        }
    }
    (headline, items, error.map(str::to_string))
}

/// 0.8's `ORSAdapter.verify_result_to_ors`: the reward written, whether it
/// is valid, and the metadata.
fn to_ors(reward: f64, items: &[(String, f64)], error: Option<String>) -> (f64, bool, Value) {
    let reward_valid = reward.is_finite() && (0.0..=1.0).contains(&reward);
    let finite = |score: f64| if score.is_finite() { score } else { 0.0 };
    let item_map: Map<String, Value> = items
        .iter()
        .map(|(name, score)| (name.clone(), json!(finite(*score))))
        .collect();
    let is_valid = error.is_none() && reward_valid;
    let mut metadata = json!({
        "items": item_map,
        "events": [],
        "error": error,
        "space": "output",
        "granularity": "terminal",
    });
    if !reward_valid {
        let raw = py_float_repr(reward);
        metadata["raw_reward"] = json!(raw);
        if metadata["error"].is_null() {
            metadata["error"] = json!(format!("invalid reward: {raw}"));
        }
    }
    let raw_items: Map<String, Value> = items
        .iter()
        .filter(|(_, score)| !score.is_finite())
        .map(|(name, score)| (name.clone(), json!(py_float_repr(*score))))
        .collect();
    if !raw_items.is_empty() {
        metadata["raw_items"] = Value::Object(raw_items);
    }
    (if reward_valid { reward } else { 0.0 }, is_valid, metadata)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta<'a>(
        prompts: &'a [String],
        rewards: Option<&'a Map<String, Value>>,
    ) -> VerifiersMeta<'a> {
        VerifiersMeta {
            task_id: "t",
            prompts,
            rewards,
            error: None,
            model: Some("m"),
        }
    }

    #[test]
    fn a_record_splits_at_the_first_assistant_message() {
        let prompts = vec!["do it".to_string(), String::new()];
        let rewards = json!({"reward": 1, "rubric": [{"name": "style", "score": 0.5}, {"score": 1}, "x"], "tests": 3, "flag": true});
        let events = vec![
            json!({"type": "user_message", "text": "do it"}),
            json!({"type": "agent_thought", "text": "hmm"}),
            json!({"type": "tool_call", "tool_call_id": "a", "kind": "execute", "title": "", "content": [{"type": "content", "content": {"type": "text", "text": "out"}}]}),
            json!({"type": "agent_message", "text": "child", "parent_tool_call_id": "a"}),
            json!({"type": "oracle", "command": ""}),
        ];
        let record = to_verifiers_record(&events, &meta(&prompts, rewards.as_object()));
        // Python: export.write_rollout_verifiers_jsonl(...) on the same inputs.
        assert_eq!(
            verifiers_line(&record, &Redactor::default()),
            concat!(
                r#"{"example_id": 0, "prompt": [{"role": "user", "content": "do it"}, {"role": "user", "content": "do it"}], "#,
                r#""completion": [{"role": "assistant", "content": "hmm"}, {"role": "assistant", "content": "[tool_call: execute]\n{'type': 'text', 'text': 'out'}"}, {"role": "assistant", "content": "[oracle: oracle]"}], "#,
                r#""reward": 1.0, "metrics": {"tests": 3.0, "style": 0.5, "rubric_1": 1.0}, "is_completed": true, "is_truncated": false, "#,
                r#""info": {"task_id": "t", "environment": "t", "model": "m", "reward_valid": true, "reward_metadata": {"items": {"tests": 3.0, "style": 0.5, "rubric_1": 1.0}, "events": [], "error": null, "space": "output", "granularity": "terminal"}, "#,
                r#""acp_projection": {"scope": "root_agent_projection", "excluded_attributed_child_events": 1, "source_artifact": "trajectory/acp_trajectory.jsonl", "child_transcript_completeness": "unknown", "outcome_and_usage_scope": "rollout_not_partitioned_by_agent"}}}"#,
                "\n"
            )
        );
    }

    #[test]
    fn an_unscored_or_out_of_range_reward_is_invalid() {
        let record = to_verifiers_record(&[], &meta(&[], None));
        assert_eq!(record["reward"], json!(0.0));
        assert_eq!(record["info"]["reward_valid"], json!(false));
        assert_eq!(
            record["info"]["reward_metadata"]["error"],
            json!("no rewards")
        );
        assert_eq!(record["prompt"], json!([]));

        let rewards = json!({"reward": 1.5});
        let record = to_verifiers_record(&[], &meta(&[], rewards.as_object()));
        assert_eq!(record["reward"], json!(0.0));
        let metadata = &record["info"]["reward_metadata"];
        assert_eq!(metadata["raw_reward"], json!("1.5"));
        assert_eq!(metadata["error"], json!("invalid reward: 1.5"));
    }
}
