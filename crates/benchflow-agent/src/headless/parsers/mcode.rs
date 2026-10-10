//! `mcode-stream-json`: `mcode exec --output-format stream-json` (MiniMax
//! Code, the npm `@minimax-ai/code` CLI).
//!
//! One event per line, each with `schemaVersion: 1`, `sequence`, `runId`,
//! `sessionId`, `turnId` and a `type` (read from mcode 0.6.7's exec event
//! projector; Harbor parses only usage):
//!
//! - `exec.started`, `session.started` / `session.resumed`, `turn.started`;
//! - `item.started` / `item.updated` / `item.completed` with an `item` whose
//!   `type` is `agent_message` or `reasoning` (`contentDelta` while
//!   streaming, the whole `content` when completed) or `tool_call`
//!   (`toolCall`, the model's request; the stream carries no tool output,
//!   so a call is marked completed when its request is);
//! - `turn.completed` with `usage` (`inputTokens`, the uncached input,
//!   `cacheReadTokens`, `cacheWriteTokens`, `outputTokens`,
//!   `reasoningTokens`), or `turn.failed` with `status` and `error`
//!   (`category`, `code`, `message`, `retryable`);
//! - `exec.completed` with the run's `result`.
//!
//! Harbor's older format, `{"type": "message", "message": {"role":
//! "assistant", "usage": {...}}}`, is read for its usage.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use super::follow::{arguments, finish_call, start_call};
use crate::headless::parser::{
    Failure, Hint, Out, Parser, count, json_object, message, str_of, thought,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct McodeStreamJson {
    /// Text items that streamed deltas, so their completed text is not repeated.
    streamed: HashSet<String>,
    /// Tool calls started, by item id, with the call id used.
    calls: HashMap<String, String>,
    usage: Usage,
}

impl Parser for McodeStreamJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "type") {
            "exec.started" | "session.started" | "session.resumed" | "turn.started"
            | "exec.completed" => {}
            kind @ ("item.started" | "item.updated" | "item.completed") => {
                if let Some(item) = event.get("item") {
                    self.item(item, kind == "item.completed", out);
                }
            }
            "turn.completed" => {
                if let Some(usage) = event.get("usage") {
                    self.add_usage(usage);
                    out.push(Out::Usage(self.usage));
                }
            }
            "turn.failed" => {
                let error = event.get("error").unwrap_or(&Value::Null);
                let text = match str_of(error, "message") {
                    "" => format!("mcode's turn {}", str_of(&event, "status")),
                    m => m.to_string(),
                };
                out.push(Out::Failure(Failure::fatal(text).hint(hint(error))));
            }
            // mcode before 0.6: whole messages, read for their usage.
            "message" => {
                let msg = event.get("message").unwrap_or(&Value::Null);
                if str_of(msg, "role") == "assistant"
                    && let Some(usage) = msg.get("usage")
                {
                    self.add_usage(usage);
                    out.push(Out::Usage(self.usage));
                }
            }
            _ => return false,
        }
        true
    }
}

impl McodeStreamJson {
    fn item(&mut self, item: &Value, done: bool, out: &mut Vec<Out>) {
        let id = str_of(item, "id").to_string();
        match str_of(item, "type") {
            kind @ ("agent_message" | "reasoning") => {
                let chunk = |text: &str| {
                    if kind == "reasoning" {
                        thought(text)
                    } else {
                        message(text)
                    }
                };
                if let Some(delta) = item.get("contentDelta").and_then(Value::as_str) {
                    if !delta.is_empty() {
                        self.streamed.insert(id.clone());
                        out.push(chunk(delta));
                    }
                } else if done
                    && !self.streamed.remove(&id)
                    && let Some(text) = item.get("content").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    out.push(chunk(text));
                }
                if done {
                    self.streamed.remove(&id);
                }
            }
            "tool_call" => {
                let call = item.get("toolCall").unwrap_or(&Value::Null);
                let function = call.get("function").unwrap_or(call);
                let name = str_of(function, "name");
                if !self.calls.contains_key(&id) && (done || !name.is_empty()) {
                    let call_id = match str_of(call, "id") {
                        "" => id.clone(),
                        c => c.to_string(),
                    };
                    let input = ["arguments", "input", "args"]
                        .iter()
                        .find_map(|k| arguments(function.get(*k)));
                    start_call(out, &call_id, name, input);
                    self.calls.insert(id.clone(), call_id);
                }
                if done && let Some(call_id) = self.calls.get(&id) {
                    finish_call(out, call_id, false, None);
                }
            }
            _ => {}
        }
    }

    fn add_usage(&mut self, usage: &Value) {
        let u = &mut self.usage;
        let add = |slot: &mut Option<u64>, key: &str| {
            if let Some(n) = count(usage.get(key)) {
                *slot = Some(slot.unwrap_or(0) + n);
            }
        };
        add(&mut u.input_tokens, "inputTokens");
        add(&mut u.output_tokens, "outputTokens");
        add(&mut u.cached_read_tokens, "cacheReadTokens");
        add(&mut u.cached_write_tokens, "cacheWriteTokens");
        add(&mut u.thought_tokens, "reasoningTokens");
    }
}

/// What mcode's error category or code says, when it says.
fn hint(error: &Value) -> Option<Hint> {
    let words =
        format!("{} {}", str_of(error, "category"), str_of(error, "code")).to_ascii_lowercase();
    if words.contains("auth") {
        Some(Hint::Login)
    } else if words.contains("quota") || words.contains("billing") || words.contains("insufficient")
    {
        Some(Hint::UsageLimit)
    } else if words.contains("rate") || words.contains("overload") || words.contains("provider") {
        Some(Hint::Provider)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn event(kind: &str, extra: Value) -> String {
        let mut e = json!({"schemaVersion": 1, "sequence": 1, "timestampMs": 0,
                           "runId": "r", "sessionId": "s", "turnId": "t", "type": kind});
        e.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        e.to_string()
    }

    #[tokio::test]
    async fn streamed_text_tool_calls_and_usage() {
        let lines = vec![
            event("exec.started", json!({})),
            event("session.started", json!({})),
            event("turn.started", json!({})),
            event(
                "item.started",
                json!({"item": {"id": "m1:reasoning", "type": "reasoning", "contentDelta": "Look first."}}),
            ),
            event(
                "item.started",
                json!({"item": {"id": "m1:message", "type": "agent_message", "contentDelta": "I'll "}}),
            ),
            event(
                "item.updated",
                json!({"item": {"id": "m1:message", "type": "agent_message", "contentDelta": "list files."}}),
            ),
            event(
                "item.completed",
                json!({"item": {"id": "m1:reasoning", "type": "reasoning", "content": "Look first."}}),
            ),
            event(
                "item.completed",
                json!({"item": {"id": "m1:message", "type": "agent_message", "content": "I'll list files."}}),
            ),
            event(
                "item.completed",
                json!({"item": {"id": "call_1", "type": "tool_call",
                "toolCall": {"id": "call_1", "name": "bash", "arguments": "{\"command\":\"ls\"}"}}}),
            ),
            event(
                "item.completed",
                json!({"item": {"id": "m2:message", "type": "agent_message", "content": "Done."}}),
            ),
            event(
                "turn.completed",
                json!({"model": {"id": "MiniMax-M3"}, "durationMs": 10,
                "usage": {"inputTokens": 900, "outputTokens": 40, "cacheReadTokens": 100, "reasoningTokens": 5}}),
            ),
            event("exec.completed", json!({"result": {"status": "succeeded"}})),
        ];
        let run = run_lines("mcode-stream-json", &lines).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_thought",
                "agent_message",
                "tool_call",
                "agent_message"
            ],
            "{events:?}"
        );
        assert!(matches!(&events[2], Event::AgentMessage(t) if t.text == "I'll list files."));
        assert!(matches!(&events[3], Event::ToolCall(c) if c.title == "ls"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(900), Some(100), Some(40))
        );
    }

    #[tokio::test]
    async fn a_failed_turn_fails_the_run_with_its_category() {
        let lines = vec![
            event("exec.started", json!({})),
            event(
                "turn.failed",
                json!({"status": "failed", "durationMs": 5,
                "error": {"category": "provider", "code": "AUTH_ERROR", "message": "Request failed", "retryable": false}}),
            ),
            event("exec.completed", json!({"result": {"status": "failed"}})),
        ];
        let run = run_lines_exit("mcode-stream-json", &lines, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    #[tokio::test]
    async fn harbors_older_message_events_give_usage() {
        // test_mcode.py's populate_context case.
        let lines = vec![
            json!({"type": "message", "message": {"role": "user", "usage": {"inputTokens": 999}}}).to_string(),
            json!({"type": "message", "message": {"role": "assistant", "usage": {"inputTokens": 10, "cacheReadTokens": 4, "outputTokens": 3}}}).to_string(),
            json!({"type": "message", "message": {"role": "assistant", "usage": {"inputTokens": 20, "cacheReadTokens": 6, "outputTokens": 5}}}).to_string(),
            json!({"type": "message", "message": {"role": "assistant", "content": "hi"}}).to_string(),
        ];
        let run = run_lines_exit("mcode-stream-json", &lines, 0).await;
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(30), Some(10), Some(8))
        );
    }

    #[tokio::test]
    async fn unknown_events_go_to_the_log() {
        let lines = vec![
            event("future.event", json!({})),
            event(
                "item.completed",
                json!({"item": {"id": "x", "type": "future_item"}}),
            ),
            event(
                "item.completed",
                json!({"item": {"id": "m", "type": "agent_message", "content": "ok"}}),
            ),
        ];
        let run = run_lines("mcode-stream-json", &lines).await;
        assert!(run.result.is_ok());
        assert_eq!(run.log.len(), 1);
    }
}
