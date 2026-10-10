//! Agents that record their steps in a file, followed while they work.
//!
//! mini-swe-agent, SWE-agent and trae-cli rewrite a JSON trajectory after
//! every step, and OpenClaw appends to a session transcript; their
//! console says little BenchFlow can read. Their commands run the agent
//! under a follower, written to the agent's home before it starts:
//!
//! - `.benchflow/follow.py FILE ARRAYS DROP -- COMMAND...` (standard-library
//!   Python, run with the agent's own interpreter) prints each new element
//!   of the named top-level arrays of the JSON file as it appears, and the
//!   file's other values once the agent exits;
//! - `.benchflow/follow.sh 'GLOB' COMMAND...` (POSIX sh) prints each new
//!   line of the newest JSON Lines file matching the glob;
//! - `.benchflow/follow-sqlite.js DATABASE SESSION_KEY -- COMMAND...`
//!   (Node.js 23.4+, run with OpenClaw's own) prints each new event of the
//!   transcript OpenClaw 2026.9 keeps in SQLite, as follow.sh prints a line.
//!
//! Both print the agent's own output as it comes, and each record as one
//! line, `{"@bf": "<array>", "item": <element>}` (`info` for the closing
//! values, `session` for JSON Lines), so a record is one JSON object the
//! runner can shrink like any other long line, and steps reach the
//! trajectory while the agent works (a run stopped at its budget keeps
//! them). Both exit with the agent's exit status.
//!
//! [`ChatMessages`] reads OpenAI chat-format messages (mini-swe-agent's and
//! Hermes's): assistant text, reasoning and tool calls, and tool results
//! matched to their calls.

use serde_json::{Value, json};

use crate::headless::parser::{
    Out, ToolDone, ToolStart, content_text, count, kind_for_tool, message, notice, one_line,
    str_of, thought, tool_title,
};
use crate::trajectory::Usage;

/// `.benchflow/follow.py`, for JSON files.
pub(crate) const FOLLOW_PY: (&str, &str) = (".benchflow/follow.py", include_str!("follow.py"));
/// `.benchflow/follow.sh`, for JSON Lines files.
pub(crate) const FOLLOW_SH: (&str, &str) = (".benchflow/follow.sh", include_str!("follow.sh"));
/// `.benchflow/follow-sqlite.js`, for OpenClaw's transcript in SQLite.
pub(crate) const FOLLOW_SQLITE: (&str, &str) = (
    ".benchflow/follow-sqlite.js",
    include_str!("follow-sqlite.js"),
);

/// A follower's record: its array's name and the element.
pub(crate) fn followed(line: &str) -> Option<(String, Value)> {
    let trimmed = line.trim();
    if !trimmed.starts_with("{\"@bf\"") {
        return None;
    }
    let Ok(Value::Object(mut record)) = serde_json::from_str::<Value>(trimmed) else {
        return None;
    };
    let name = record.get("@bf")?.as_str()?.to_string();
    Some((name, record.remove("item").unwrap_or(Value::Null)))
}

/// A tool call's arguments: an object as given, a JSON string parsed, and
/// anything else kept as `{"command": <text>}` (mini-swe-agent's fallback).
pub(crate) fn arguments(raw: Option<&Value>) -> Option<Value> {
    match raw? {
        Value::Null => None,
        Value::String(s) if s.trim().is_empty() => None,
        Value::String(s) => Some(
            serde_json::from_str::<Value>(s)
                .ok()
                .filter(Value::is_object)
                .unwrap_or_else(|| json!({"command": s})),
        ),
        Value::Object(_) => raw.cloned(),
        other => Some(json!({"command": other.to_string()})),
    }
}

/// Start a tool call named `name` with `input`.
pub(crate) fn start_call(out: &mut Vec<Out>, id: &str, name: &str, input: Option<Value>) {
    let name = if name.is_empty() { "tool" } else { name };
    out.push(
        ToolStart {
            id,
            title: tool_title(name, input.as_ref()),
            kind: kind_for_tool(name),
            status: "in_progress",
            input,
            tool_name: Some(name),
            parent: None,
        }
        .out(),
    );
}

/// Finish a tool call with its output text.
pub(crate) fn finish_call(out: &mut Vec<Out>, id: &str, failed: bool, text: Option<String>) {
    out.push(
        ToolDone {
            id,
            status: if failed { "failed" } else { "completed" },
            text,
            ..ToolDone::default()
        }
        .out(),
    );
}

/// Add OpenAI-style usage (`prompt_tokens` including the cache, or the
/// Responses API's `input_tokens`) to a running total.
pub(crate) fn add_openai_usage(total: &mut Usage, usage: &Value) -> bool {
    let input = count(usage.get("prompt_tokens")).or_else(|| count(usage.get("input_tokens")));
    let output =
        count(usage.get("completion_tokens")).or_else(|| count(usage.get("output_tokens")));
    if input.is_none() && output.is_none() {
        return false;
    }
    let details = |a: &str, b: &str| usage.get(a).or_else(|| usage.get(b));
    let cached = details("prompt_tokens_details", "input_tokens_details")
        .and_then(|d| count(d.get("cached_tokens")))
        .unwrap_or(0);
    let reasoning = details("completion_tokens_details", "output_tokens_details")
        .and_then(|d| count(d.get("reasoning_tokens")));
    let input = input.unwrap_or(0);
    total.input_tokens = Some(total.input_tokens.unwrap_or(0) + input.saturating_sub(cached));
    total.cached_read_tokens = Some(total.cached_read_tokens.unwrap_or(0) + cached.min(input));
    total.output_tokens = Some(total.output_tokens.unwrap_or(0) + output.unwrap_or(0));
    if let Some(r) = reasoning {
        total.thought_tokens = Some(total.thought_tokens.unwrap_or(0) + r);
    }
    true
}

/// OpenAI chat-format messages, read one at a time.
#[derive(Default)]
pub(crate) struct ChatMessages {
    /// Calls without a result yet, oldest first.
    pending: Vec<String>,
    /// User messages seen; the first is the task prompt, already recorded.
    users: usize,
    /// Calls given an id here (mini-swe-agent's text-mode actions).
    synthesized: usize,
    /// Token totals.
    pub usage: Usage,
    /// Whether any message carried usage.
    pub has_usage: bool,
}

impl ChatMessages {
    /// Read one message. `usage` is where its token counts are, if anywhere.
    pub fn message(&mut self, msg: &Value, usage: Option<&Value>, out: &mut Vec<Out>) {
        if let Some(usage) = usage
            && add_openai_usage(&mut self.usage, usage)
        {
            self.has_usage = true;
            out.push(Out::Usage(self.usage));
        }
        if msg.get("object").and_then(Value::as_str) == Some("response") {
            self.response(msg, out);
            return;
        }
        if str_of(msg, "type") == "function_call_output" {
            let id = str_of(msg, "call_id").to_string();
            let text = match msg.get("output") {
                Some(Value::String(s)) => Some(s.clone()),
                other => content_text(other),
            };
            self.result(Some(&id), text, out);
            return;
        }
        match str_of(msg, "role") {
            "system" | "developer" => {}
            "user" => {
                self.users += 1;
                if self.users == 1 {
                    return;
                }
                let text = content_text(msg.get("content")).unwrap_or_default();
                if self.pending.is_empty() {
                    // Feedback from the harness (a format error), not a tool's output.
                    if !text.trim().is_empty() {
                        out.push(notice("info", &one_line(&text)));
                    }
                } else {
                    self.result(None, Some(text), out);
                }
            }
            "tool" | "function" => {
                let id = msg
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                self.result(id.as_deref(), content_text(msg.get("content")), out);
            }
            "assistant" => self.assistant(msg, out),
            _ => {}
        }
    }

    fn assistant(&mut self, msg: &Value, out: &mut Vec<Out>) {
        for key in ["reasoning_content", "reasoning", "thinking"] {
            if let Some(text) = msg.get(key).and_then(Value::as_str)
                && !text.trim().is_empty()
            {
                out.push(thought(text));
                break;
            }
        }
        if let Some(text) = content_text(msg.get("content"))
            && !text.trim().is_empty()
        {
            out.push(message(&text));
        }
        let calls = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .filter(|c| !c.is_empty());
        if let Some(calls) = calls {
            for call in calls {
                let function = call.get("function").unwrap_or(call);
                let id = match str_of(call, "id") {
                    "" => self.next_id(),
                    id => id.to_string(),
                };
                let name = str_of(function, "name");
                let name = if name.is_empty() { "bash" } else { name };
                start_call(out, &id, name, arguments(function.get("arguments")));
                self.pending.push(id);
            }
            return;
        }
        // mini-swe-agent's text mode: the actions parsed from the reply.
        if let Some(actions) = msg
            .get("extra")
            .and_then(|e| e.get("actions"))
            .and_then(Value::as_array)
        {
            for action in actions {
                let command = match action.get("command") {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => other.to_string(),
                    None => continue,
                };
                let id = self.next_id();
                start_call(out, &id, "bash", Some(json!({"command": command})));
                self.pending.push(id);
            }
        }
    }

    /// A Responses API turn: `output` items of type message, reasoning and
    /// function_call.
    fn response(&mut self, msg: &Value, out: &mut Vec<Out>) {
        for item in msg
            .get("output")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match str_of(item, "type") {
                "message" => {
                    if let Some(text) = content_text(item.get("content"))
                        && !text.trim().is_empty()
                    {
                        out.push(message(&text));
                    }
                }
                "reasoning" => {
                    for key in ["summary", "content"] {
                        if let Some(text) = content_text(item.get(key))
                            && !text.trim().is_empty()
                        {
                            out.push(thought(&text));
                        }
                    }
                }
                "function_call" => {
                    let id = [str_of(item, "call_id"), str_of(item, "id")]
                        .into_iter()
                        .find(|s| !s.is_empty())
                        .map(str::to_string)
                        .unwrap_or_else(|| self.next_id());
                    let name = match str_of(item, "name") {
                        "" => "bash",
                        n => n,
                    };
                    start_call(out, &id, name, arguments(item.get("arguments")));
                    self.pending.push(id);
                }
                _ => {}
            }
        }
    }

    /// A tool's result: for the call `id`, else the oldest call still open.
    fn result(&mut self, id: Option<&str>, text: Option<String>, out: &mut Vec<Out>) {
        let id = match id.filter(|i| !i.is_empty()) {
            Some(id) => {
                self.pending.retain(|p| p != id);
                id.to_string()
            }
            None if self.pending.is_empty() => return,
            None => self.pending.remove(0),
        };
        finish_call(out, &id, false, text);
    }

    fn next_id(&mut self) -> String {
        self.synthesized += 1;
        format!("call-{}", self.synthesized)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_are_one_json_object_per_line() {
        let (name, item) =
            followed(r#"{"@bf":"messages","item":{"role":"assistant","content":"hi"}}"#).unwrap();
        assert_eq!(name, "messages");
        assert_eq!(item["content"], "hi");
        assert!(followed(r#"{"role":"assistant"}"#).is_none());
        assert!(followed("console output").is_none());
        // A record cut short (the agent was killed mid-write) is not one.
        assert!(followed(r#"{"@bf":"messages","item":{"role":"assi"#).is_none());
    }

    #[test]
    fn arguments_parse_strings_and_keep_the_rest() {
        assert_eq!(
            arguments(Some(&json!(r#"{"command": "ls"}"#))),
            Some(json!({"command": "ls"}))
        );
        assert_eq!(
            arguments(Some(&json!("not json"))),
            Some(json!({"command": "not json"}))
        );
        assert_eq!(arguments(Some(&json!({"a": 1}))), Some(json!({"a": 1})));
        assert_eq!(arguments(Some(&json!(""))), None);
        assert_eq!(arguments(None), None);
    }

    #[test]
    fn openai_usage_counts_the_cache_once() {
        let mut total = Usage::default();
        assert!(add_openai_usage(
            &mut total,
            &json!({"prompt_tokens": 600, "completion_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 100},
                    "completion_tokens_details": {"reasoning_tokens": 30}})
        ));
        assert!(add_openai_usage(
            &mut total,
            &json!({"input_tokens": 50, "output_tokens": 5})
        ));
        assert!(!add_openai_usage(&mut total, &json!({})));
        assert_eq!(
            (
                total.input_tokens,
                total.cached_read_tokens,
                total.output_tokens,
                total.thought_tokens
            ),
            (Some(550), Some(100), Some(125), Some(30))
        );
    }
}
