//! An ATIF trajectory as chat rows for a Hugging Face dataset, ported from
//! Harbor's `traces export` (`harbor/utils/traces_utils.py`,
//! `extract_conversations_from_trajectory`, `openai_to_sharegpt`) so the
//! same trajectory gives the same conversations.
//!
//! Each agent step ends an episode: row `episode-<n>` holds the
//! conversation up to and including that step. System and user steps are
//! `user` messages (the first system message carries the tool definitions
//! in `<tools>` tags, when the agent has any), an agent step is an
//! `assistant` message of its `<think>` reasoning, its text, and its
//! `<tool_call>` JSON, and the observations of every agent step but the
//! last are a `user` message.

use serde_json::{Map, Value, json};

use super::pyjson::{Dumps, dumps};

/// What a row says about its run besides the conversation.
#[derive(Debug, Clone, Default)]
pub struct RunMeta {
    pub agent: String,
    pub model: String,
    pub model_provider: String,
    pub date: String,
    pub run_id: String,
    pub task: String,
    pub trial_name: String,
    /// The reward as Python's `str(float)`, or what stopped the trial.
    pub result: Option<String>,
}

/// Why a trajectory gave no rows.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RowsError {
    #[error(
        "the trajectory has images or audio, and this export is text only; leave the trial out or use a multimodal pipeline"
    )]
    Multimodal,
}

/// `normalize_message_content`: text from a string, the first text part of
/// a list, or JSON of anything else.
fn normalize(content: Option<&Value>) -> String {
    match content {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) if parts.first().is_some_and(|p| p.get("text").is_some()) => {
            match &parts[0]["text"] {
                Value::String(s) => s.clone(),
                other => dumps(other, Dumps::UNICODE),
            }
        }
        Some(other) => dumps(other, Dumps::UNICODE),
    }
}

fn has_media(content: Option<&Value>) -> bool {
    content.and_then(Value::as_array).is_some_and(|parts| {
        parts.iter().any(|p| {
            matches!(
                p.get("type").and_then(Value::as_str),
                Some("image" | "audio")
            )
        })
    })
}

fn multimodal(traj: &Value) -> bool {
    steps(traj).iter().any(|s| {
        has_media(s.get("message"))
            || s.get("observation")
                .and_then(|o| o.get("results"))
                .and_then(Value::as_array)
                .is_some_and(|rs| rs.iter().any(|r| has_media(r.get("content"))))
    })
}

fn steps(traj: &Value) -> Vec<Value> {
    traj.get("steps")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
    }
}

fn source(step: &Value) -> &str {
    step.get("source").and_then(Value::as_str).unwrap_or("")
}

/// Text of a message for the assistant's content.
fn text(message: Option<&Value>) -> String {
    match message {
        Some(Value::String(s)) => s.clone(),
        other => normalize(other),
    }
}

/// One row per episode of the trajectory, in order.
pub fn rows(traj: &Value, meta: &RunMeta) -> Result<Vec<Map<String, Value>>, RowsError> {
    if multimodal(traj) {
        return Err(RowsError::Multimodal);
    }
    let steps = steps(traj);
    let agent = traj.get("agent").cloned().unwrap_or_else(|| json!({}));
    let agent_name = agent
        .get("name")
        .filter(|v| truthy(Some(v)))
        .and_then(Value::as_str)
        .map_or(meta.agent.clone(), str::to_string);
    let model = agent
        .get("model_name")
        .filter(|v| truthy(Some(v)))
        .and_then(Value::as_str)
        .map_or(meta.model.clone(), str::to_string);
    let tools = agent
        .get("tool_definitions")
        .filter(|v| truthy(Some(v)))
        .cloned();
    let ends: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| source(s) == "agent" && !truthy(s.get("is_copied_context")))
        .map(|(i, _)| i)
        .collect();
    let mut out = Vec::new();
    for (n, &end) in ends.iter().enumerate() {
        let conversation = episode(&steps[..=end], tools.as_ref());
        if conversation.is_empty() {
            continue;
        }
        let mut row = Map::new();
        row.insert("conversations".into(), Value::Array(conversation));
        row.insert("agent".into(), json!(agent_name));
        row.insert("model".into(), json!(model));
        row.insert("model_provider".into(), json!(meta.model_provider));
        row.insert("date".into(), json!(meta.date));
        row.insert("task".into(), json!(meta.task));
        row.insert("episode".into(), json!(format!("episode-{n}")));
        row.insert("run_id".into(), json!(meta.run_id));
        row.insert("trial_name".into(), json!(meta.trial_name));
        if let Some(t) = &tools {
            row.insert("tool_definitions".into(), t.clone());
        }
        row.insert("result".into(), json!(meta.result));
        out.push(row);
    }
    Ok(out)
}

/// The messages of one episode: `steps` ends with its agent step.
fn episode(steps: &[Value], tools: Option<&Value>) -> Vec<Value> {
    let last_agent = steps.iter().rposition(|s| source(s) == "agent");
    let mut tools_added = false;
    let mut out = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        let message = step.get("message").cloned().unwrap_or_else(|| json!(""));
        match source(step) {
            "system" => {
                let content = match tools {
                    Some(t) if !tools_added => {
                        tools_added = true;
                        let tools_json = dumps(
                            t,
                            Dumps {
                                ascii: false,
                                indent: Some(2),
                            },
                        );
                        json!(format!(
                            "<tools>\n{tools_json}\n</tools>\n\n{}",
                            super::pyjson::py_str(&message)
                        ))
                    }
                    _ => message,
                };
                out.push(json!({"role": "user", "content": content}));
            }
            "user" => out.push(json!({"role": "user", "content": message})),
            "agent" => {
                let mut parts: Vec<String> = Vec::new();
                if let Some(r) = step.get("reasoning_content").filter(|r| truthy(Some(r))) {
                    parts.push(format!("<think>{}</think>", super::pyjson::py_str(r)));
                }
                if truthy(Some(&message)) {
                    let mut m = text(Some(&message));
                    if m.contains("</think>") && !m.contains("<think>") {
                        m = format!("<think>{m}");
                    }
                    parts.push(m);
                }
                for call in step
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let obj = json!({
                        "name": call.get("function_name").cloned().unwrap_or(Value::Null),
                        "arguments": call.get("arguments").cloned().unwrap_or_else(|| json!({})),
                    });
                    parts.push(format!(
                        "<tool_call>\n{}\n</tool_call>",
                        dumps(&obj, Dumps::UNICODE)
                    ));
                }
                out.push(json!({"role": "assistant", "content": parts.join("\n")}));
                if Some(i) != last_agent {
                    let results = step
                        .get("observation")
                        .filter(|o| o.is_object())
                        .and_then(|o| o.get("results"))
                        .and_then(Value::as_array);
                    let texts: Vec<String> = results
                        .into_iter()
                        .flatten()
                        .filter_map(|r| r.as_object())
                        .filter(|r| r.contains_key("content"))
                        .map(|r| normalize(r.get("content")))
                        .collect();
                    if !texts.is_empty() {
                        out.push(json!({"role": "user", "content": texts.join("\n")}));
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// `openai_to_sharegpt`: `user` as `human`, `assistant` as `gpt`.
pub fn sharegpt(conversation: &[Value]) -> Vec<Value> {
    conversation
        .iter()
        .filter_map(|m| {
            let from = match m.get("role").and_then(Value::as_str)? {
                "user" => "human",
                "assistant" => "gpt",
                "system" => "system",
                _ => return None,
            };
            Some(json!({"from": from, "value": normalize(m.get("content"))}))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> RunMeta {
        RunMeta {
            agent: "claude".into(),
            model: "m".into(),
            ..RunMeta::default()
        }
    }

    #[test]
    fn one_row_per_agent_step_with_observations_between() {
        let traj = json!({"agent": {"name": "claude"}, "steps": [
            {"source": "user", "message": "do it"},
            {"source": "agent", "message": "", "reasoning_content": "plan",
             "tool_calls": [{"function_name": "Bash", "arguments": {"command": "ls"}}],
             "observation": {"results": [{"content": "a.txt"}]}},
            {"source": "agent", "message": "done"},
        ]});
        let rows = rows(&traj, &meta()).unwrap();
        assert_eq!(rows.len(), 2);
        let last = rows[1]["conversations"].as_array().unwrap();
        assert_eq!(last.len(), 4);
        assert_eq!(
            last[1]["content"],
            "<think>plan</think>\n<tool_call>\n{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>"
        );
        assert_eq!(last[2], json!({"role": "user", "content": "a.txt"}));
        assert_eq!(
            rows[0]["conversations"].as_array().unwrap().len(),
            2,
            "the last step's observation belongs to the next episode"
        );
        assert_eq!(rows[1]["episode"], "episode-1");
        let sg = sharegpt(last);
        assert_eq!(sg[0], json!({"from": "human", "value": "do it"}));
        assert_eq!(sg[1]["from"], "gpt");
    }

    #[test]
    fn media_is_refused() {
        let traj = json!({"steps": [{"source": "user", "message": [{"type": "image", "source": {"path": "a.png"}}]}, {"source": "agent", "message": "x"}]});
        assert_eq!(rows(&traj, &meta()), Err(RowsError::Multimodal));
    }
}
