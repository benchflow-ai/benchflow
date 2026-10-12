//! The ADP record: one rollout as an Agent Data Protocol 1.3.1 `Trajectory`
//! (0.8's `trajectories/export_adp.py`), the format the OpenHands training
//! pipeline converts to SFT data.
//!
//! - prompts and `user_message`s become `text_observation`s from the user;
//! - an `agent_message` becomes a `message_action`, carrying the thoughts
//!   before it as `reasoning_content` (thoughts nothing follows become an
//!   empty `message_action` of their own);
//! - a tool call becomes an `api_action` (its kind as `function`, its title
//!   as `description`, `kwargs` empty) and a `text_observation` from the
//!   environment with its output; ids are unique, missing or repeated ones
//!   replaced by `call_000001`, `call_000002`, …;
//! - an oracle run becomes a `message_action` `[oracle: <command>]`.
//!
//! The terminal reward rides on the last action, or in
//! `details.terminal_reward` when there is none. A subagent's events are
//! left out and counted in `details.acp_projection`.

use std::collections::HashSet;

use serde_json::{Map, Value, json};

use super::export_common::{
    ThoughtBuffer, acp_projection_coverage, content_blocks_to_text, is_attributed_child, str_or,
    str_or_empty,
};
use super::jsonl::to_python_json;
use super::redact::Redactor;

/// The ADP schema version written.
pub const ADP_SCHEMA_VERSION: &str = "1.3.1";

const ACTION_CLASSES: [&str; 3] = ["api_action", "code_action", "message_action"];

/// What an ADP record says about its rollout besides the events.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdpMeta<'a> {
    /// The record's `id`.
    pub trajectory_id: &'a str,
    /// The task's name; also `details.environment`.
    pub task_id: &'a str,
    /// The prompts handed to the agent before any event was recorded.
    pub prompts: &'a [String],
    pub model: Option<&'a str>,
    /// The verifier's `reward`, as it is in the reward map.
    pub reward: Option<&'a Value>,
}

/// The record for `events` (each in its `acp_trajectory.jsonl` form), or
/// `None` when it has no content and no reward to carry: such a record
/// would say nothing.
pub fn to_adp_record(events: &[Value], meta: &AdpMeta<'_>) -> Option<Value> {
    let mut content = to_content(events, meta.prompts);
    let mut details = json!({
        "task_id": meta.task_id,
        "environment": meta.task_id,
        "model": meta.model.unwrap_or_default(),
    });
    if let Some(coverage) = acp_projection_coverage(events) {
        details["acp_projection"] = coverage;
    }
    if let Some(reward) = meta.reward.filter(|r| !r.is_null()) {
        let last_action = content
            .iter_mut()
            .rev()
            .find(|item| ACTION_CLASSES.iter().any(|c| item["class_"] == *c));
        match last_action {
            Some(action) => action["reward"] = reward.clone(),
            None => {
                details["terminal_reward"] = reward.clone();
                tracing::warn!(
                    trajectory = meta.trajectory_id,
                    "ADP trajectory has a terminal reward but no action to carry it; \
                     recorded under details.terminal_reward"
                );
            }
        }
    }
    if content.is_empty() && details.get("terminal_reward").is_none() {
        return None;
    }
    Some(json!({
        "schema_version": ADP_SCHEMA_VERSION,
        "id": meta.trajectory_id,
        "content": content,
        "details": details,
    }))
}

/// The record as `trainer/adp.jsonl` holds it: redacted, one line.
pub fn adp_line(record: &Value, redactor: &Redactor) -> String {
    let mut clean = record.clone();
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

fn user_observation(text: String) -> Value {
    json!({"class_": "text_observation", "content": text, "source": "user"})
}

/// 0.8's `acp_events_to_adp_content`.
fn to_content(events: &[Value], prompts: &[String]) -> Vec<Value> {
    let mut content: Vec<Value> = prompts
        .iter()
        .filter(|p| !p.is_empty())
        .map(|p| user_observation(p.clone()))
        .collect();
    let mut thoughts = ThoughtBuffer::default();
    let mut used_ids: HashSet<String> = HashSet::new();
    let mut synthesized = 0;
    let mut claim_call_id = |raw: String| {
        let mut candidate = raw;
        while candidate.is_empty() || used_ids.contains(&candidate) {
            synthesized += 1;
            candidate = format!("call_{synthesized:06}");
        }
        used_ids.insert(candidate.clone());
        candidate
    };
    let flush = |content: &mut Vec<Value>, thoughts: &mut ThoughtBuffer| {
        if let Some(reasoning) = thoughts.take() {
            content.push(json!({
                "class_": "message_action",
                "content": "",
                "reasoning_content": reasoning,
            }));
        }
    };
    let with_reasoning = |mut action: Map<String, Value>, thoughts: &mut ThoughtBuffer| {
        if let Some(reasoning) = thoughts.take() {
            action.insert("reasoning_content".into(), reasoning.into());
        }
        Value::Object(action)
    };

    for event in events.iter().filter_map(Value::as_object) {
        if is_attributed_child(event) {
            continue;
        }
        match event.get("type").and_then(Value::as_str) {
            Some("user_message") => {
                let text = str_or_empty(event, "text");
                if !text.is_empty() {
                    flush(&mut content, &mut thoughts);
                    content.push(user_observation(text));
                }
            }
            Some("agent_thought") => {
                let text = str_or_empty(event, "text");
                if !text.is_empty() {
                    thoughts.push(text);
                }
            }
            Some("agent_message") => {
                let text = str_or_empty(event, "text");
                if !text.is_empty() {
                    let mut action = Map::new();
                    action.insert("class_".into(), "message_action".into());
                    action.insert("content".into(), text.into());
                    content.push(with_reasoning(action, &mut thoughts));
                }
            }
            Some("tool_call") => {
                let call_id = claim_call_id(str_or_empty(event, "tool_call_id"));
                let mut action = Map::new();
                action.insert("class_".into(), "api_action".into());
                action.insert("tool_call_id".into(), call_id.clone().into());
                action.insert("function".into(), str_or(event, "kind", "tool").into());
                action.insert("kwargs".into(), json!({}));
                let title = str_or_empty(event, "title");
                if !title.is_empty() {
                    action.insert("description".into(), title.into());
                }
                content.push(with_reasoning(action, &mut thoughts));
                content.push(json!({
                    "class_": "text_observation",
                    "tool_call_id": call_id,
                    "content": content_blocks_to_text(event.get("content")),
                    "source": "environment",
                }));
            }
            Some("oracle") => {
                let command = str_or(event, "command", "oracle");
                content.push(
                    json!({"class_": "message_action", "content": format!("[oracle: {command}]")}),
                );
            }
            _ => {}
        }
    }
    flush(&mut content, &mut thoughts);
    content
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta<'a>(prompts: &'a [String], reward: Option<&'a Value>) -> AdpMeta<'a> {
        AdpMeta {
            trajectory_id: "t__1",
            task_id: "t",
            prompts,
            model: None,
            reward,
        }
    }

    #[test]
    fn a_record_pairs_actions_with_observations() {
        let prompts = vec!["do it".to_string()];
        let reward = json!(1);
        let events = vec![
            json!({"type": "agent_thought", "text": "plan"}),
            json!({"type": "tool_call", "tool_call_id": "a", "kind": "execute", "title": "ls", "content": [{"text": "x"}]}),
            json!({"type": "tool_call", "tool_call_id": "a", "kind": "", "title": "", "content": "raw"}),
            json!({"type": "tool_call", "kind": "read", "content": []}),
            json!({"type": "agent_thought", "text": "done?"}),
            json!({"type": "user_message", "text": "more"}),
            json!({"type": "agent_thought", "text": "last"}),
        ];
        let record = to_adp_record(&events, &meta(&prompts, Some(&reward))).unwrap();
        // Python: export_adp.write_rollout_adp_jsonl(...) on the same inputs.
        assert_eq!(
            adp_line(&record, &Redactor::default()),
            concat!(
                r#"{"schema_version": "1.3.1", "id": "t__1", "content": ["#,
                r#"{"class_": "text_observation", "content": "do it", "source": "user"}, "#,
                r#"{"class_": "api_action", "tool_call_id": "a", "function": "execute", "kwargs": {}, "description": "ls", "reasoning_content": "plan"}, "#,
                r#"{"class_": "text_observation", "tool_call_id": "a", "content": "x", "source": "environment"}, "#,
                r#"{"class_": "api_action", "tool_call_id": "call_000001", "function": "tool", "kwargs": {}}, "#,
                r#"{"class_": "text_observation", "tool_call_id": "call_000001", "content": "raw", "source": "environment"}, "#,
                r#"{"class_": "api_action", "tool_call_id": "call_000002", "function": "read", "kwargs": {}}, "#,
                r#"{"class_": "text_observation", "tool_call_id": "call_000002", "content": "", "source": "environment"}, "#,
                r#"{"class_": "message_action", "content": "", "reasoning_content": "done?"}, "#,
                r#"{"class_": "text_observation", "content": "more", "source": "user"}, "#,
                r#"{"class_": "message_action", "content": "", "reasoning_content": "last", "reward": 1}], "#,
                r#""details": {"task_id": "t", "environment": "t", "model": ""}}"#,
                "\n"
            )
        );
    }

    #[test]
    fn an_empty_record_is_written_only_to_carry_a_reward() {
        assert_eq!(to_adp_record(&[], &meta(&[], None)), None);
        let reward = json!(0.0);
        let record = to_adp_record(&[], &meta(&[], Some(&reward))).unwrap();
        assert_eq!(record["content"], json!([]));
        assert_eq!(record["details"]["terminal_reward"], json!(0.0));
        let prompts = vec!["p".to_string()];
        let record = to_adp_record(&[], &meta(&prompts, Some(&reward))).unwrap();
        assert_eq!(record["details"]["terminal_reward"], json!(0.0));
    }
}
