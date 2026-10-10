//! `antigravity-stream-json`: `agy --prompt ... --output-format stream-json`
//! (Google's Antigravity CLI).
//!
//! One JSON object per line, named by `event` (BenchFlow 0.8's ACP shim
//! drove agy over the same stream, and Harbor's `antigravity-cli` reads its
//! usage):
//!
//! - `init` (`conversation_id`): bookkeeping;
//! - `step_update` (`step_update`: `step_index`, `step_type`, `state`,
//!   `text_delta`, `tool_info` with `name`, `parameters`, `output`, `error`,
//!   and on a finished step `usage`): `agent_response` steps are the agent's
//!   words and `thinking`/`planning` steps its thoughts, streamed as deltas;
//!   `tool`/`tool_call` steps are tool calls (`ACTIVE` while running, then
//!   `DONE` or `ERROR`);
//! - `result` (`result`: `status`, `usage` with `input_tokens`,
//!   `output_tokens`, `cache_read_tokens` (apart from the input),
//!   `thinking_tokens`, and `error` when it failed): the run's end.
//!
//! agy writes `AGY_ERROR: {"short_error", "status", "retryable"}` to stderr
//! when it fails; the runner reads stderr for the cause.

use std::collections::HashSet;

use serde_json::Value;

use crate::headless::parser::{
    Failure, Hint, Out, Parser, ToolDone, ToolStart, count, json_object, message, one_line, str_of,
    thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct AntigravityStreamJson {
    /// Tool steps already opened, by call id.
    open: HashSet<String>,
    /// Each finished step's usage, by step index (a repeat restates it).
    steps: std::collections::BTreeMap<i64, Usage>,
    /// The `result` event's totals came.
    final_usage: bool,
}

impl Parser for AntigravityStreamJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "event") {
            "step_update" => self.step(event.get("step_update").unwrap_or(&Value::Null), out),
            "result" => self.result(event.get("result").unwrap_or(&Value::Null), out),
            "init" | "user" => {}
            _ => return false,
        }
        true
    }
}

impl AntigravityStreamJson {
    fn step(&mut self, step: &Value, out: &mut Vec<Out>) {
        let index = step.get("step_index").and_then(Value::as_i64).unwrap_or(-1);
        if let Some(usage) = step.get("usage").and_then(agy_usage)
            && !self.final_usage
        {
            self.steps.insert(index, usage);
            out.push(Out::Usage(sum(self.steps.values())));
        }
        let text = str_of(step, "text_delta");
        match str_of(step, "step_type") {
            "agent_response" => {
                if !text.is_empty() {
                    out.push(message(text));
                }
            }
            "thinking" | "thought" | "planning" | "reasoning" => {
                if !text.is_empty() {
                    out.push(thought(text));
                }
            }
            "tool" | "tool_call" => self.tool(step, index, out),
            "user_input" => {}
            other => {
                if !text.is_empty() {
                    out.push(thought(&format!("[{other}] {text}")));
                }
            }
        }
    }

    fn tool(&mut self, step: &Value, index: i64, out: &mut Vec<Out>) {
        let info = step.get("tool_info").unwrap_or(&Value::Null);
        let name = [str_of(info, "name"), str_of(step, "tool_name")]
            .into_iter()
            .find(|n| !n.is_empty())
            .unwrap_or("tool");
        let params = info.get("parameters").filter(|p| p.is_object());
        let id = format!("agy-{index}");
        if self.open.insert(id.clone()) {
            out.push(
                ToolStart {
                    id: &id,
                    title: title(name, params),
                    kind: tool_kind(name),
                    status: "in_progress",
                    input: params.cloned(),
                    tool_name: Some(name),
                    parent: None,
                }
                .out(),
            );
        }
        let state = str_of(step, "state");
        if state == "ACTIVE" || state.is_empty() {
            return;
        }
        let error = info.get("error").filter(|e| !e.is_null());
        let failed = state == "ERROR" || error.is_some();
        let text = if failed {
            Some(match error {
                Some(Value::Object(e)) => [str_of_map(e, "message"), str_of_map(e, "type")]
                    .into_iter()
                    .find(|s| !s.is_empty())
                    .unwrap_or("tool failed")
                    .to_string(),
                Some(Value::String(s)) => s.clone(),
                Some(other) => other.to_string(),
                None => "tool failed".to_string(),
            })
        } else {
            match info.get("output") {
                Some(Value::String(s)) => Some(s.clone()),
                Some(Value::Null) | None => None,
                Some(other) => Some(other.to_string()),
            }
        };
        self.open.remove(&id);
        out.push(
            ToolDone {
                id: &id,
                status: if failed { "failed" } else { "completed" },
                text,
                output: info.get("output").filter(|o| !o.is_null()).cloned(),
                parent: None,
            }
            .out(),
        );
    }

    fn result(&mut self, result: &Value, out: &mut Vec<Out>) {
        if let Some(usage) = result.get("usage").and_then(agy_usage) {
            self.final_usage = true;
            out.push(Out::Usage(usage));
        }
        // Calls the turn never finished.
        let status = if str_of(result, "status").eq_ignore_ascii_case("canceled") {
            "cancelled"
        } else {
            "failed"
        };
        let mut open: Vec<String> = self.open.drain().collect();
        open.sort();
        for id in open {
            out.push(
                ToolDone {
                    id: &id,
                    status,
                    ..ToolDone::default()
                }
                .out(),
            );
        }
        let state = str_of(result, "status").to_ascii_uppercase();
        let error = match result.get("error") {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(Value::Object(e)) => Some(
                [str_of_map(e, "message"), str_of_map(e, "short_error")]
                    .into_iter()
                    .find(|s| !s.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| Value::Object(e.clone()).to_string()),
            ),
            _ => None,
        };
        if error.is_some() || matches!(state.as_str(), "ERROR" | "FAILED" | "FAILURE") {
            let words = error.unwrap_or_else(|| format!("Antigravity ended with {state}"));
            let hint = if words.contains("RESOURCE_EXHAUSTED") {
                Some(Hint::Provider)
            } else {
                None
            };
            out.push(Out::Failure(Failure::fatal(words).hint(hint)));
        }
    }
}

fn str_of_map<'a>(map: &'a serde_json::Map<String, Value>, key: &str) -> &'a str {
    map.get(key).and_then(Value::as_str).unwrap_or("")
}

/// A call's title: a command line itself, else what [`tool_title`] makes
/// of the call.
fn title(name: &str, params: Option<&Value>) -> String {
    let command = params.and_then(|p| {
        ["CommandLine", "command"]
            .iter()
            .find_map(|k| p.get(*k).and_then(Value::as_str))
    });
    match command {
        Some(c) if tool_kind(name) == "execute" && !c.trim().is_empty() => one_line(c),
        _ => tool_title(name, params),
    }
}

/// agy's tool names, as 0.8's shim mapped them.
fn tool_kind(name: &str) -> &'static str {
    match name {
        "run_command" | "send_command_input" | "command_status" | "read_terminal"
        | "notebook_execution" => "execute",
        "view_file" | "view_code_item" | "view_content_chunk" | "list_dir" | "read_resource"
        | "list_resources" => "read",
        "grep_search" | "find_by_name" | "codebase_search" => "search",
        "write_to_file" | "replace_file_content" | "multi_replace_file_content" | "edit_file" => {
            "edit"
        }
        "search_web" | "read_url_content" => "fetch",
        other => crate::headless::parser::kind_for_tool(other),
    }
}

/// agy's usage: `cache_read_tokens` is apart from `input_tokens`.
fn agy_usage(value: &Value) -> Option<Usage> {
    if !value.is_object() {
        return None;
    }
    let usage = Usage {
        input_tokens: count(value.get("input_tokens")),
        output_tokens: count(value.get("output_tokens")),
        cached_read_tokens: count(value.get("cache_read_tokens")),
        thought_tokens: count(value.get("thinking_tokens")),
        ..Usage::default()
    };
    (usage != Usage::default()).then_some(usage)
}

fn sum<'a>(usages: impl Iterator<Item = &'a Usage>) -> Usage {
    let add = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    };
    usages.fold(Usage::default(), |acc, u| Usage {
        input_tokens: add(acc.input_tokens, u.input_tokens),
        output_tokens: add(acc.output_tokens, u.output_tokens),
        cached_read_tokens: add(acc.cached_read_tokens, u.cached_read_tokens),
        thought_tokens: add(acc.thought_tokens, u.thought_tokens),
        ..Usage::default()
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn step(index: i64, step_type: &str, state: &str, extra: Value) -> String {
        let mut s = json!({"step_index": index, "step_type": step_type, "state": state});
        if let Value::Object(e) = extra {
            s.as_object_mut().unwrap().extend(e);
        }
        json!({"event": "step_update", "step_update": s}).to_string()
    }

    #[tokio::test]
    async fn steps_become_thoughts_calls_and_the_answer() {
        let l = vec![
            json!({"event": "init", "conversation_id": "c0ffee00-1"}).to_string(),
            step(0, "thinking", "ACTIVE", json!({"text_delta": "I should list "})),
            step(0, "thinking", "DONE", json!({"text_delta": "the files."})),
            step(
                1,
                "tool",
                "ACTIVE",
                json!({"tool_info": {"name": "run_command", "parameters": {"CommandLine": "ls", "command": "ls"}}}),
            ),
            step(
                1,
                "tool",
                "DONE",
                json!({"tool_info": {"name": "run_command", "parameters": {"command": "ls"}, "output": "README.md"},
                       "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 40}}),
            ),
            step(
                2,
                "agent_response",
                "DONE",
                json!({"text_delta": "Done.", "usage": {"input_tokens": 20, "output_tokens": 5}}),
            ),
            json!({"event": "result", "result": {"status": "DONE", "usage": {"input_tokens": 120, "output_tokens": 15, "cache_read_tokens": 40, "thinking_tokens": 7}}}).to_string(),
        ];
        let run = run_lines("antigravity-stream-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_thought",
                "tool_call",
                "agent_message"
            ]
        );
        assert!(
            matches!(&events[1], Event::AgentThought(t) if t.text == "I should list the files.")
        );
        assert!(matches!(&events[2], Event::ToolCall(c)
            if c.title == "ls" && c.kind == "execute" && c.status.as_str() == "completed"));
        let u = run.usage.unwrap();
        assert_eq!(
            (
                u.input_tokens,
                u.output_tokens,
                u.cached_read_tokens,
                u.thought_tokens
            ),
            (Some(120), Some(15), Some(40), Some(7))
        );
    }

    #[tokio::test]
    async fn a_run_without_a_result_keeps_the_steps_usage() {
        let l = vec![
            step(
                0,
                "agent_response",
                "DONE",
                json!({"text_delta": "hi", "usage": {"input_tokens": 5, "output_tokens": 1}}),
            ),
            step(
                0,
                "agent_response",
                "DONE",
                json!({"usage": {"input_tokens": 5, "output_tokens": 1}}),
            ),
            step(
                1,
                "agent_response",
                "DONE",
                json!({"usage": {"input_tokens": 7, "output_tokens": 2}}),
            ),
        ];
        let run = run_lines("antigravity-stream-json", &l).await;
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (Some(12), Some(3)));
    }

    #[tokio::test]
    async fn a_failed_tool_and_a_failed_turn() {
        let l = vec![
            step(
                0,
                "tool",
                "ACTIVE",
                json!({"tool_info": {"name": "view_file", "parameters": {"AbsolutePath": "/x"}}}),
            ),
            step(
                1,
                "tool",
                "ERROR",
                json!({"tool_info": {"name": "write_to_file", "error": {"message": "denied"}}}),
            ),
            json!({"event": "result", "result": {"status": "ERROR", "error": "API key not valid. Please pass a valid API key."}}).to_string(),
        ];
        let run = run_lines_exit("antigravity-stream-json", &l, 1).await;
        let events = events_of(&run);
        assert!(
            matches!(&events[1], Event::ToolCall(c) if c.kind == "read" && c.status.as_str() == "failed")
        );
        assert!(
            matches!(&events[2], Event::ToolCall(c) if c.kind == "edit" && c.status.as_str() == "failed")
        );
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
