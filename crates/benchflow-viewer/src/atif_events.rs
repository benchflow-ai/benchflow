//! Trial folders that keep their trajectory only as an ATIF document
//! (`agent/trajectory.json`, as other harnesses write it), read as the
//! events the rest of the viewer shows: each user step a prompt, a system
//! step a notice, and an agent step its reasoning, its text, and its tool
//! calls with their results.

use std::path::Path;

#[cfg(test)]
use crate::pyjson::get;
use crate::pyjson::{Map, Value, object};

/// Where such a trial keeps its ATIF document.
pub const ATIF_FILE: &str = "agent/trajectory.json";

/// The largest ATIF document read whole (the trajectory file is streamed;
/// this one is parsed in one piece).
const MAX_ATIF_BYTES: u64 = 100 * 1024 * 1024;

/// A trial with an ATIF document and a result, and no ACP trajectory.
pub fn is_atif_trial(path: &Path) -> bool {
    !path.join(crate::payload::TRAJECTORY_FILE).exists()
        && path.join(ATIF_FILE).is_file()
        && path.join("result.json").is_file()
}

fn text_of(value: &Value) -> String {
    match value {
        Value::Str(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        other => crate::pyjson::dumps(other),
    }
}

fn event(kind: &str, ts: &Value) -> Map {
    let mut map = Map::new();
    map.insert("type".into(), kind.into());
    if let Value::Str(_) = ts {
        map.insert("ts".into(), ts.clone());
    }
    map
}

/// The trial's ATIF document as events; `None` when it has none.
pub fn atif_trial_events(rollout_dir: &Path) -> Option<Vec<Map>> {
    events_within(rollout_dir, MAX_ATIF_BYTES)
}

/// The document as events, when it is at most `cap` bytes.
fn events_within(rollout_dir: &Path, cap: u64) -> Option<Vec<Map>> {
    let path = rollout_dir.join(ATIF_FILE);
    if std::fs::metadata(&path).ok()?.len() > cap {
        return None;
    }
    let doc = crate::values::load_json(&path)?;
    let steps = doc.get("steps").as_array()?;
    let mut events = Vec::new();
    for step in steps {
        let ts = step.get("timestamp");
        let message = text_of(step.get("message"));
        match step.get("source").as_str() {
            Some("user") => {
                let mut e = event("user_message", ts);
                e.insert("text".into(), message.into());
                events.push(e);
            }
            Some("system") => {
                let mut e = event("notice", ts);
                e.insert("severity".into(), "info".into());
                e.insert("title".into(), "System".into());
                e.insert("description".into(), message.into());
                events.push(e);
            }
            _ => {
                let reasoning = text_of(step.get("reasoning_content"));
                if !reasoning.is_empty() {
                    let mut e = event("agent_thought", ts);
                    e.insert("text".into(), reasoning.into());
                    events.push(e);
                }
                if !message.is_empty() {
                    let mut e = event("agent_message", ts);
                    e.insert("text".into(), message.into());
                    events.push(e);
                }
                let results: &[Value] = step
                    .get("observation")
                    .get("results")
                    .as_array()
                    .unwrap_or(&[]);
                let calls: &[Value] = step.get("tool_calls").as_array().unwrap_or(&[]);
                // Results that name no call belong to the step's last call.
                let unnamed: Vec<String> = results
                    .iter()
                    .filter(|r| r.get("source_call_id").as_str().is_none())
                    .map(|r| text_of(r.get("content")))
                    .collect();
                for (i, call) in calls.iter().enumerate() {
                    let id = call.get("tool_call_id").as_str().unwrap_or("").to_string();
                    let mut outputs: Vec<String> = results
                        .iter()
                        .filter(|r| !id.is_empty() && r.get("source_call_id").as_str() == Some(&id))
                        .map(|r| text_of(r.get("content")))
                        .collect();
                    if i + 1 == calls.len() {
                        outputs.extend(unnamed.iter().cloned());
                    }
                    let mut e = event("tool_call", ts);
                    e.insert("tool_call_id".into(), id.into());
                    e.insert("kind".into(), "other".into());
                    let name = call.get("function_name").as_str().unwrap_or("tool");
                    e.insert("title".into(), name.into());
                    e.insert("status".into(), "completed".into());
                    e.insert("raw_input".into(), call.get("arguments").clone());
                    let content: Vec<Value> = outputs
                        .into_iter()
                        .filter(|t| !t.is_empty())
                        .map(|t| {
                            object([
                                ("type", "content".into()),
                                (
                                    "content",
                                    object([("type", "text".into()), ("text", t.into())]),
                                ),
                            ])
                        })
                        .collect();
                    e.insert("content".into(), Value::Array(content));
                    events.push(e);
                }
                if calls.is_empty() && !unnamed.is_empty() {
                    let mut e = event("notice", ts);
                    e.insert("severity".into(), "info".into());
                    e.insert("title".into(), "Observation".into());
                    e.insert("description".into(), unnamed.join("\n").into());
                    events.push(e);
                }
            }
        }
    }
    Some(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E1 round 2: `bench view` refused a trial that keeps its trajectory
    /// only as `agent/trajectory.json`; its steps now read as events.
    #[test]
    fn an_atif_only_trial_reads_as_events() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("agent")).unwrap();
        std::fs::write(
            dir.path().join("result.json"),
            r#"{"verifier_result": {"rewards": {"reward": 1.0}}}"#,
        )
        .unwrap();
        std::fs::write(
            dir.path().join(ATIF_FILE),
            r#"{"schema_version": "ATIF-v1.8", "agent": {"name": "a", "version": "1"}, "steps": [
                {"step_id": 1, "source": "user", "message": "Fix it."},
                {"step_id": 2, "source": "agent", "message": "Plan: ls", "reasoning_content": "hmm",
                 "tool_calls": [{"tool_call_id": "c1", "function_name": "bash", "arguments": {"command": "ls"}}],
                 "observation": {"results": [{"source_call_id": "c1", "content": "a.txt"}]}},
                {"step_id": 3, "source": "agent", "message": "",
                 "tool_calls": [{"tool_call_id": "k1", "function_name": "bash_command", "arguments": {"keystrokes": "pwd\n"}}],
                 "observation": {"results": [{"content": "/app"}]}}]}"#,
        )
        .unwrap();
        assert!(is_atif_trial(dir.path()));
        assert!(crate::catalog::is_rollout_dir(dir.path()));
        let events = atif_trial_events(dir.path()).unwrap();
        let kinds: Vec<&str> = events
            .iter()
            .map(|e| get(e, "type").as_str().unwrap())
            .collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_thought",
                "agent_message",
                "tool_call",
                "tool_call"
            ]
        );
        assert_eq!(
            get(&events[3], "content").as_array().unwrap()[0]
                .get("content")
                .get("text")
                .as_str(),
            Some("a.txt")
        );
        assert_eq!(
            get(&events[4], "content").as_array().unwrap()[0]
                .get("content")
                .get("text")
                .as_str(),
            Some("/app")
        );
        let payload = crate::payload::build_payload(dir.path());
        // Review E1-B (low): a document over the cap is not read whole.
        let size = std::fs::metadata(dir.path().join(ATIF_FILE)).unwrap().len();
        assert!(events_within(dir.path(), size).is_some());
        assert!(events_within(dir.path(), size - 1).is_none());
        assert_eq!(payload.steps.len(), 5);
        assert_eq!(payload.meta.metadata.reward, Some(1.0));
    }
}
