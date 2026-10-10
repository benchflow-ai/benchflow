//! `cursor-stream-json`: `cursor-agent --print --output-format stream-json`.
//!
//! One JSON object per line (Cursor's documented output format, as Harbor's
//! `cursor-cli` agent reads it):
//!
//! - `system` (`subtype: init`, `session_id`, `model`): bookkeeping;
//! - `user`: the prompt, which BenchFlow already recorded;
//! - `assistant` (`message.content`: text blocks): the agent's words, one
//!   message between tool calls;
//! - `thinking` (`subtype: delta | completed`, `text`): reasoning, streamed;
//! - `tool_call` (`subtype: started | completed`, `call_id`, `tool_call`: one
//!   key naming the tool, such as `shellToolCall`, holding `args` and, once
//!   completed, `result` with `success` or `error`);
//! - `interaction_query`: a permission prompt `--force` already answered;
//! - `result` (`is_error`, `result`, `usage` with `inputTokens`,
//!   `outputTokens`, `cacheReadTokens`, `cacheWriteTokens`, sometimes
//!   `totalCost` or `cost`): the run's end.

use std::collections::HashSet;

use serde_json::Value;

use crate::headless::parser::{
    Failure, Out, Parser, ToolDone, ToolStart, content_text, count, json_object, kind_for_tool,
    message, str_of, thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct CursorStreamJson {
    /// Calls already opened.
    open: HashSet<String>,
}

impl Parser for CursorStreamJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "type") {
            "assistant" => {
                let msg = event.get("message").unwrap_or(&Value::Null);
                if let Some(text) = content_text(msg.get("content"))
                    && !text.is_empty()
                {
                    out.push(message(&text));
                }
            }
            "thinking" => {
                let text = str_of(&event, "text");
                if !text.is_empty() {
                    out.push(thought(text));
                }
            }
            "tool_call" => self.tool_call(&event, out),
            "result" => result(&event, out),
            "system" | "user" | "interaction_query" => {}
            _ => return false,
        }
        true
    }
}

impl CursorStreamJson {
    fn tool_call(&mut self, event: &Value, out: &mut Vec<Out>) {
        let id = str_of(event, "call_id");
        // `{"shellToolCall": {"args": ..., "result": ...}}`.
        let Some((key, call)) = event
            .get("tool_call")
            .and_then(Value::as_object)
            .and_then(|m| m.iter().next())
        else {
            return;
        };
        let name = tool_name(key);
        let args = call.get("args").filter(|a| !a.is_null());
        if self.open.insert(id.to_string()) {
            out.push(
                ToolStart {
                    id,
                    title: tool_title(&name, args),
                    kind: kind_for_tool(&name),
                    status: "in_progress",
                    input: args.cloned(),
                    tool_name: Some(&name),
                    parent: None,
                }
                .out(),
            );
        }
        if str_of(event, "subtype") != "completed" {
            return;
        }
        let result = call.get("result").unwrap_or(&Value::Null);
        let failed = result.get("error").is_some()
            || result.get("failure").is_some()
            || result.get("rejected").is_some();
        let body = result
            .get("success")
            .or_else(|| result.get("error"))
            .or_else(|| result.get("failure"))
            .unwrap_or(result);
        out.push(
            ToolDone {
                id,
                status: if failed { "failed" } else { "completed" },
                text: result_text(body),
                output: (!result.is_null()).then(|| result.clone()),
                parent: None,
            }
            .out(),
        );
    }
}

/// `shellToolCall` is `shell`, `readToolCall` is `read`.
fn tool_name(key: &str) -> String {
    let name = key.strip_suffix("ToolCall").unwrap_or(key);
    if name.is_empty() {
        "tool".to_string()
    } else {
        name.to_string()
    }
}

/// A result's text: its output fields when it has them, else its JSON.
fn result_text(body: &Value) -> Option<String> {
    match body {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => {
            let parts: Vec<String> = ["stdout", "stderr", "content", "output", "message", "error"]
                .iter()
                .filter_map(|k| match map.get(*k)? {
                    Value::String(s) if !s.is_empty() => Some(s.clone()),
                    Value::Null => None,
                    v @ Value::Array(_) => content_text(Some(v)),
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                Some(body.to_string())
            } else {
                Some(parts.join("\n"))
            }
        }
        other => Some(other.to_string()),
    }
}

fn result(event: &Value, out: &mut Vec<Out>) {
    if let Some(usage) = event.get("usage").filter(|u| u.is_object()) {
        let u = Usage {
            input_tokens: count(usage.get("inputTokens")),
            output_tokens: count(usage.get("outputTokens")),
            cached_read_tokens: count(usage.get("cacheReadTokens")),
            cached_write_tokens: count(usage.get("cacheWriteTokens")),
            ..Usage::default()
        };
        if u != Usage::default() {
            out.push(Out::Usage(u));
        }
        if let Some(cost) = ["totalCost", "cost"]
            .iter()
            .find_map(|k| usage.get(*k).and_then(Value::as_f64))
        {
            out.push(Out::Cost(cost));
        }
    }
    if event.get("is_error").and_then(Value::as_bool) == Some(true) {
        let text = str_of(event, "result").trim();
        let words = if text.is_empty() {
            format!("Cursor ended with {}", str_of(event, "subtype"))
        } else {
            text.to_string()
        };
        out.push(Out::Failure(Failure::fatal(words)));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    // Harbor's test_cursor_cli_trajectory.py events.
    fn system() -> Value {
        json!({"type": "system", "subtype": "init", "apiKeySource": "env", "cwd": "/app",
               "session_id": "sess-1", "model": "claude-sonnet-4-5", "permissionMode": "force"})
    }

    fn assistant(text: &str, model_call_id: &str) -> Value {
        json!({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
               "session_id": "sess-1", "model_call_id": model_call_id, "timestamp_ms": 1_767_225_600_000_u64})
    }

    fn thinking(text: &str) -> Value {
        json!({"type": "thinking", "subtype": "delta", "text": text, "session_id": "sess-1",
               "timestamp_ms": 1_767_225_600_000_u64})
    }

    fn tool_call(call_id: &str, subtype: &str) -> Value {
        let mut call = json!({"args": {"command": "ls"}});
        if subtype == "completed" {
            call["result"] = json!({"success": {"stdout": "README.md"}});
        }
        json!({"type": "tool_call", "subtype": subtype, "call_id": call_id, "model_call_id": "mc-1",
               "session_id": "sess-1", "timestamp_ms": 1_767_225_601_000_u64,
               "tool_call": {"shellToolCall": call}})
    }

    fn lines(events: &[Value]) -> Vec<String> {
        events.iter().map(Value::to_string).collect()
    }

    #[tokio::test]
    async fn tool_calls_join_the_turn_and_the_result_reports_usage() {
        let l = lines(&[
            system(),
            thinking("Let me look around."),
            assistant("Checking files.", "mc-1"),
            tool_call("tc-1", "started"),
            tool_call("tc-1", "completed"),
            tool_call("tc-2", "completed"),
            assistant("Done.", "mc-2"),
            json!({"type": "result", "subtype": "success", "duration_ms": 1200, "duration_api_ms": 900,
                   "is_error": false, "result": "Done.", "session_id": "sess-1",
                   "usage": {"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 50,
                             "cacheWriteTokens": 5, "totalCost": 0.0123}}),
        ]);
        let run = run_lines("cursor-stream-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let kinds: Vec<&str> = events_of(&run).iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_thought",
                "agent_message",
                "tool_call",
                "tool_call",
                "agent_message"
            ]
        );
        assert!(matches!(&events_of(&run)[3], Event::ToolCall(c)
            if c.title == "ls" && c.status.as_str() == "completed" && c.kind == "execute"));
        let u = run.usage.unwrap();
        assert_eq!(
            (
                u.input_tokens,
                u.output_tokens,
                u.cached_read_tokens,
                u.cached_write_tokens
            ),
            (Some(100), Some(20), Some(50), Some(5))
        );
        assert_eq!(run.cost_usd, Some(0.0123));
    }

    #[tokio::test]
    async fn a_failed_tool_and_an_error_result() {
        let mut failed = tool_call("tc-1", "completed");
        failed["tool_call"]["shellToolCall"]["result"] =
            json!({"error": {"message": "permission denied"}});
        let l = lines(&[
            system(),
            failed,
            json!({"type": "result", "subtype": "error", "duration_ms": 1, "duration_api_ms": 1,
                   "is_error": true, "result": "Invalid API key", "session_id": "sess-1"}),
        ]);
        let run = run_lines_exit("cursor-stream-json", &l, 1).await;
        assert!(matches!(&events_of(&run)[1], Event::ToolCall(c) if c.status.as_str() == "failed"));
        let err = run.result.unwrap_err();
        assert!(err.to_string().contains("Invalid API key"), "{err}");
    }

    #[tokio::test]
    async fn unknown_events_and_noise_go_to_the_log() {
        let l = vec![
            "Cursor Agent v2026.10.01".to_string(),
            system().to_string(),
            json!({"type": "future_event"}).to_string(),
            assistant("hi", "mc-1").to_string(),
        ];
        let run = run_lines("cursor-stream-json", &l).await;
        assert!(run.result.is_ok());
        assert_eq!(run.log.len(), 2, "{:?}", run.log);
    }
}
