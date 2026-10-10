//! `copilot-json`: `copilot --prompt ... --output-format json` (GitHub
//! Copilot CLI).
//!
//! One JSON event per line, in one of two shapes depending on the model
//! (Harbor's `copilot-cli` reads both):
//!
//! - flat, Anthropic-style: `message` (`role`, `content`), `tool_use` (`id`,
//!   `name`, `input`), `tool_result` (`tool_use_id`, `content`,
//!   `is_error`), `usage` (`input_tokens`, `output_tokens`);
//! - Copilot's session events, each with a `data` object:
//!   `assistant.message` (`content`, `toolRequests` with `toolCallId`,
//!   `name`, `arguments`; `outputTokens`, `reasoningText`),
//!   `user.message`, `tool.execution_start`, `tool.execution_complete`
//!   (`toolCallId`, `success`, `result`, `error`), `session.error`
//!   (`message`), and a closing `result` (`exitCode`, `usage`).
//!
//! Streaming deltas (`*_delta`, `assistant.turn_*`, `session.*`) repeat
//! what the consolidated events say, and are skipped.

use std::collections::HashSet;

use serde_json::Value;

use super::follow::{arguments, start_call};
use crate::headless::parser::{
    Failure, Out, Parser, ToolDone, content_text, count, json_object, message, str_of, thought,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct CopilotJson {
    usage: Usage,
    /// Calls opened.
    open: HashSet<String>,
}

impl Parser for CopilotJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        let data = event.get("data").unwrap_or(&Value::Null);
        match str_of(&event, "type") {
            // Flat schema.
            "message" => {
                if str_of(&event, "role") == "assistant"
                    && let Some(text) = content_text(event.get("content"))
                    && !text.trim().is_empty()
                {
                    out.push(message(&text));
                }
            }
            "tool_use" => {
                let id = str_of(&event, "id").to_string();
                self.start(out, &id, str_of(&event, "name"), event.get("input"));
            }
            "tool_result" => {
                let failed = event.get("is_error").and_then(Value::as_bool) == Some(true);
                self.finish(
                    out,
                    str_of(&event, "tool_use_id"),
                    failed,
                    content_text(event.get("content")),
                );
            }
            "usage" => {
                self.add(
                    count(event.get("input_tokens")),
                    count(event.get("output_tokens")),
                    out,
                );
            }
            // Session-event schema.
            "assistant.message" => {
                for key in ["reasoningText", "reasoning"] {
                    if let Some(text) = data.get(key).and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        out.push(thought(text));
                        break;
                    }
                }
                if let Some(text) = content_text(data.get("content"))
                    && !text.trim().is_empty()
                {
                    out.push(message(&text));
                }
                for request in data
                    .get("toolRequests")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let id = str_of(request, "toolCallId").to_string();
                    self.start(out, &id, str_of(request, "name"), request.get("arguments"));
                }
                self.add(None, count(data.get("outputTokens")), out);
            }
            "tool.execution_start" => {
                let id = str_of(data, "toolCallId").to_string();
                let name = [str_of(data, "toolName"), str_of(data, "name")]
                    .into_iter()
                    .find(|n| !n.is_empty())
                    .unwrap_or("tool");
                self.start(out, &id, name, data.get("arguments"));
            }
            "tool.execution_complete" => {
                let error = data.get("error").filter(|e| !e.is_null());
                let failed =
                    error.is_some() || data.get("success").and_then(Value::as_bool) == Some(false);
                let mut text = result_text(data.get("result"));
                if let Some(error) = error {
                    let e = result_text(Some(error)).unwrap_or_default();
                    text = Some(match text {
                        Some(t) if !t.is_empty() => format!("{t}\n{e}"),
                        _ => e,
                    });
                }
                self.finish(out, str_of(data, "toolCallId"), failed, text);
            }
            "session.error" | "error" => {
                let words = [str_of(data, "message"), str_of(&event, "message")]
                    .into_iter()
                    .find(|m| !m.is_empty())
                    .map(str::to_string)
                    .unwrap_or_else(|| event.to_string());
                out.push(Out::Failure(Failure::fatal(words)));
            }
            "result" => {
                let usage = event.get("usage").or_else(|| data.get("usage"));
                if let Some(usage) = usage.filter(|u| u.is_object()) {
                    let input = count(usage.get("inputTokens"))
                        .or_else(|| count(usage.get("input_tokens")));
                    let output = count(usage.get("outputTokens"))
                        .or_else(|| count(usage.get("output_tokens")));
                    if input.is_some() || output.is_some() {
                        // The run's totals.
                        self.usage = Usage::default();
                        self.add(input, output, out);
                    }
                }
            }
            "user.message" => {}
            t if t.contains('.') || t.ends_with("_delta") => {}
            _ => return false,
        }
        true
    }
}

impl CopilotJson {
    fn start(&mut self, out: &mut Vec<Out>, id: &str, name: &str, input: Option<&Value>) {
        if self.open.insert(id.to_string()) {
            start_call(out, id, name, arguments(input));
        }
    }

    fn finish(&mut self, out: &mut Vec<Out>, id: &str, failed: bool, text: Option<String>) {
        if self.open.insert(id.to_string()) {
            // A result for a call this stream never opened.
            start_call(out, id, "tool", None);
        }
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

    fn add(&mut self, input: Option<u64>, output: Option<u64>, out: &mut Vec<Out>) {
        if input.is_none() && output.is_none() {
            return;
        }
        if let Some(i) = input {
            self.usage.input_tokens = Some(self.usage.input_tokens.unwrap_or(0) + i);
        }
        if let Some(o) = output {
            self.usage.output_tokens = Some(self.usage.output_tokens.unwrap_or(0) + o);
        }
        out.push(Out::Usage(self.usage));
    }
}

/// A tool result's text: its `content`, `output`, `stdout`, `text` or
/// `message`, else the whole value.
fn result_text(result: Option<&Value>) -> Option<String> {
    match result? {
        Value::Null => None,
        Value::Object(map) => ["content", "output", "stdout", "text", "message"]
            .iter()
            .find_map(|k| {
                map.get(*k)
                    .and_then(Value::as_str)
                    .filter(|s| !s.is_empty())
            })
            .map(str::to_string)
            .or_else(|| Some(Value::Object(map.clone()).to_string())),
        other => content_text(Some(other)),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn lines(events: &[Value]) -> Vec<String> {
        events.iter().map(Value::to_string).collect()
    }

    /// Harbor's test_copilot_cli_trajectory.py, flat (Anthropic-model) schema.
    #[tokio::test]
    async fn the_flat_schema() {
        let l = lines(&[
            json!({"type": "message", "role": "user", "content": "Create hello.txt"}),
            json!({"type": "message", "role": "assistant", "content": "I'll create it.", "model": "claude-sonnet-4.5"}),
            json!({"type": "tool_use", "id": "tu_1", "name": "bash", "input": {"command": "echo hi > hello.txt"}}),
            json!({"type": "tool_result", "tool_use_id": "tu_1", "content": "ok"}),
            json!({"type": "usage", "input_tokens": 100, "output_tokens": 20}),
            json!({"type": "usage", "input_tokens": 50, "output_tokens": 10}),
        ]);
        let run = run_lines("copilot-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let kinds: Vec<&str> = events_of(&run).iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, ["user_message", "agent_message", "tool_call"]);
        assert!(matches!(&events_of(&run)[2], Event::ToolCall(c)
            if c.title == "echo hi > hello.txt" && c.status.as_str() == "completed"));
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (Some(150), Some(30)));
    }

    /// The session-event (GPT-model) schema: parallel calls, results out of
    /// order, a failed call, and streaming events skipped.
    #[tokio::test]
    async fn the_session_event_schema() {
        let l = lines(&[
            json!({"type": "session.start", "data": {}}),
            json!({"type": "user.message", "data": {"content": "Create hello.txt"}}),
            json!({"type": "assistant.message_delta", "data": {"deltaContent": "I'll"}}),
            json!({"type": "assistant.message", "data": {"content": "I'll create it.", "outputTokens": 12,
                   "toolRequests": [
                       {"toolCallId": "c1", "name": "bash", "arguments": {"command": "echo hi > hello.txt"}},
                       {"toolCallId": "c2", "name": "apply_patch", "arguments": "*** Begin Patch"}]}}),
            json!({"type": "tool.execution_start", "data": {"toolCallId": "c1", "toolName": "bash"}}),
            json!({"type": "tool.execution_complete", "data": {"toolCallId": "c2", "success": false,
                   "error": {"message": "patch failed"}}}),
            json!({"type": "tool.execution_complete", "data": {"toolCallId": "c1", "success": true,
                   "result": {"content": "ok", "exitCode": 0}}}),
            json!({"type": "assistant.message", "data": {"content": "Done.", "outputTokens": 3}}),
            json!({"type": "result", "exitCode": 0}),
        ]);
        let run = run_lines("copilot-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_message",
                "tool_call",
                "tool_call",
                "agent_message"
            ]
        );
        assert!(matches!(&events[2], Event::ToolCall(c) if c.status.as_str() == "completed"));
        assert!(matches!(&events[3], Event::ToolCall(c) if c.status.as_str() == "failed"));
        // Copilot reports no input tokens in this schema.
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (None, Some(15)));
        assert!(run.log.is_empty(), "{:?}", run.log);
    }

    #[tokio::test]
    async fn a_session_error_fails_the_run() {
        let l = vec![
            "Error: No authentication information found.".to_string(),
            json!({"type": "session.error", "data": {"message": "Authentication failed: 401 Unauthorized"}}).to_string(),
        ];
        let run = run_lines_exit("copilot-json", &l, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
