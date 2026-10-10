//! `muse-json`: `muse exec --json` (Muse Code, Meta's CLI).
//!
//! Muse's run events, one per line, either bare (`{"kind": ...}`) or in
//! Muse's durable envelope (`{"schema_version": 1, "payload": {"kind":
//! "run", "event": {...}}}`, the shape of its `session.jsonl`, which
//! Harbor converts). Harbor's tests and Muse's replay format give the run
//! events:
//!
//! - `started` (`prompt`), `model_request_configured`,
//!   `model_response_created`;
//! - `text_delta` (`text`), then `assistant_message_committed` (`text`, the
//!   whole message: shown only when no delta came before it);
//! - `reasoning_delta`, `reasoning_committed`, `reasoning_summary_committed`
//!   (`text`; `encrypted_content` is opaque);
//! - `assistant_tool_calls_committed` (`tool_calls[{call_id, id, name,
//!   args}]`, args a JSON string);
//! - `tool_result` (`call_id` or `tool_call_id`, `text`) and
//!   `tool_result_batch_committed` (`results[{tool_call_id, text}]`, the
//!   committed results, which replace earlier ones);
//! - `model_completed` (`usage`: `input_tokens` including `cached_tokens`,
//!   `output_tokens`, `reasoning_tokens`, `cache_write_tokens`; `cost_usd`);
//! - `terminal` (`terminal`: `completed`, or how the run ended, and
//!   `reason`): anything but `completed` fails the run;
//! - an `error` event, or one whose kind ends in `_failed`, carries the
//!   failure's words (`message`, `error`).
//!
//! Unknown kinds are skipped, as Muse's replay does.

use std::collections::HashSet;

use serde_json::Value;

use super::follow::{arguments, finish_call, start_call};
use crate::headless::parser::{
    Failure, Out, Parser, content_text, count, json_object, message, notice, str_of, thought,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct MuseJson {
    /// Text streamed since the last committed message.
    text_streamed: bool,
    reasoning_streamed: bool,
    started: HashSet<String>,
    finished: HashSet<String>,
    usage: Usage,
    cost: f64,
    /// The last error's words, for a terminal event that gives none.
    last_error: Option<String>,
}

impl Parser for MuseJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(row) = json_object(line) else {
            return false;
        };
        let row = Value::Object(row);
        let event = match row.get("payload") {
            Some(payload) => match str_of(payload, "kind") {
                "run" => payload.get("event").cloned().unwrap_or(Value::Null),
                _ => return true,
            },
            None if row.get("kind").is_some() => row,
            None => return false,
        };
        self.event(&event, out);
        true
    }
}

impl MuseJson {
    fn event(&mut self, event: &Value, out: &mut Vec<Out>) {
        let kind = str_of(event, "kind");
        match kind {
            "model_response_created" => {
                self.text_streamed = false;
                self.reasoning_streamed = false;
            }
            "text_delta" => {
                let text = str_of(event, "text");
                if !text.is_empty() {
                    self.text_streamed = true;
                    out.push(message(text));
                }
            }
            "assistant_message_committed" => {
                let text = str_of(event, "text");
                if !std::mem::take(&mut self.text_streamed) && !text.is_empty() {
                    out.push(message(text));
                }
            }
            "reasoning_delta" | "reasoning_summary_delta" => {
                let text = str_of(event, "text");
                if !text.is_empty() {
                    self.reasoning_streamed = true;
                    out.push(thought(text));
                }
            }
            "reasoning_committed" | "reasoning_summary_committed" => {
                let text = str_of(event, "text");
                if !std::mem::take(&mut self.reasoning_streamed) && !text.trim().is_empty() {
                    out.push(thought(text));
                }
            }
            "assistant_tool_calls_committed" => {
                for call in event
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let id = call_id(call);
                    if id.is_empty() || !self.started.insert(id.clone()) {
                        continue;
                    }
                    let input = arguments(call.get("args").or_else(|| call.get("arguments")));
                    start_call(out, &id, str_of(call, "name"), input);
                }
            }
            "tool_result" => self.result(event, out),
            "tool_result_batch_committed" => {
                for result in event
                    .get("results")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    // The committed batch replaces a result seen before it.
                    self.finished.remove(&call_id(result));
                    self.result(result, out);
                }
            }
            "model_completed" => {
                if let Some(usage) = event.get("usage").filter(|u| u.is_object()) {
                    let input = count(usage.get("input_tokens")).unwrap_or(0);
                    let cached = count(usage.get("cached_tokens"))
                        .or_else(|| count(usage.get("cache_read_tokens")))
                        .unwrap_or(0);
                    let u = &mut self.usage;
                    u.input_tokens =
                        Some(u.input_tokens.unwrap_or(0) + input.saturating_sub(cached));
                    u.cached_read_tokens = Some(u.cached_read_tokens.unwrap_or(0) + cached);
                    u.output_tokens = Some(
                        u.output_tokens.unwrap_or(0)
                            + count(usage.get("output_tokens")).unwrap_or(0),
                    );
                    if let Some(w) = count(usage.get("cache_write_tokens")) {
                        u.cached_write_tokens = Some(u.cached_write_tokens.unwrap_or(0) + w);
                    }
                    if let Some(r) = count(usage.get("reasoning_tokens")) {
                        u.thought_tokens = Some(u.thought_tokens.unwrap_or(0) + r);
                    }
                    out.push(Out::Usage(self.usage));
                    if let Some(cost) = usage
                        .get("cost_usd")
                        .or_else(|| event.get("cost_usd"))
                        .and_then(Value::as_f64)
                    {
                        self.cost += cost;
                        out.push(Out::Cost(self.cost));
                    }
                }
            }
            "terminal" => {
                let terminal = str_of(event, "terminal");
                if terminal != "completed" {
                    let reason = match event.get("reason") {
                        Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
                        Some(Value::Object(o)) => {
                            Some(str_of(&Value::Object(o.clone()), "message").to_string())
                                .filter(|s| !s.is_empty())
                        }
                        _ => None,
                    };
                    let text = reason
                        .or_else(|| self.last_error.clone())
                        .unwrap_or_else(|| {
                            format!(
                                "Muse ended the run as {}",
                                if terminal.is_empty() {
                                    "unknown"
                                } else {
                                    terminal
                                }
                            )
                        });
                    out.push(Out::Failure(Failure::fatal(text)));
                }
            }
            _ if kind == "error" || kind.ends_with("_failed") => {
                let text = error_text(event).unwrap_or_else(|| format!("Muse reported {kind}"));
                self.last_error = Some(text.clone());
                out.push(Out::Failure(Failure::notice(text.clone())));
                out.push(notice("warning", &text));
            }
            _ => {}
        }
    }

    fn result(&mut self, result: &Value, out: &mut Vec<Out>) {
        let id = call_id(result);
        if id.is_empty() || !self.finished.insert(id.clone()) {
            return;
        }
        let text = match result.get("text") {
            Some(Value::String(s)) => Some(s.clone()),
            other => content_text(other.or_else(|| result.get("content"))),
        };
        let failed = result.get("is_error").and_then(Value::as_bool) == Some(true);
        finish_call(out, &id, failed, text);
    }
}

fn call_id(v: &Value) -> String {
    ["call_id", "tool_call_id", "id"]
        .iter()
        .map(|k| str_of(v, k))
        .find(|s| !s.is_empty())
        .unwrap_or("")
        .to_string()
}

fn error_text(event: &Value) -> Option<String> {
    let from = |v: &Value| match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Object(_) => Some(str_of(v, "message").to_string()).filter(|s| !s.is_empty()),
        _ => None,
    };
    ["message", "error", "reason"]
        .iter()
        .find_map(|k| event.get(*k).and_then(from))
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    /// Harbor's `_events()` (test_muse_code.py).
    fn events() -> Vec<Value> {
        vec![
            json!({"kind": "started", "prompt": "Read the answer"}),
            json!({"kind": "model_response_created", "response_id": "response-1"}),
            json!({"kind": "reasoning_committed", "message_id": "m0", "response_id": "response-1",
                   "text": "Read the file.", "encrypted_content": "opaque"}),
            json!({"kind": "text_delta", "text": "Reading"}),
            json!({"kind": "assistant_tool_calls_committed", "message_id": "m1", "response_id": "response-1",
                   "tool_calls": [
                     {"id": "fc-1", "call_id": "c1", "name": "bash", "args": "{\"command\":\"cat answer.txt\"}"},
                     {"id": "fc-2", "call_id": "c2", "name": "bash", "args": "{\"command\":\"pwd\"}"}]}),
            json!({"kind": "model_completed", "model": "muse-spark-1.2",
                   "usage": {"input_tokens": 20, "output_tokens": 5, "cached_tokens": 8, "reasoning_tokens": 2}}),
            json!({"kind": "tool_result", "call_id": "c2", "text": "/app"}),
            json!({"kind": "tool_result", "call_id": "c1", "text": "42"}),
            json!({"kind": "model_response_created", "response_id": "response-2"}),
            json!({"kind": "model_completed", "usage": {"input_tokens": 40, "output_tokens": 3, "cached_tokens": 20}}),
            json!({"kind": "assistant_message_committed", "message_id": "m2", "response_id": "response-2",
                   "text": "The answer is 42."}),
            json!({"kind": "terminal", "terminal": "completed", "reason": null}),
        ]
    }

    /// Wrapped as Muse's durable envelope.
    fn enveloped(events: &[Value]) -> Vec<String> {
        let mut rows = vec![
            json!({"schema_version": 1, "sequence": 1, "payload": {"kind": "metadata",
            "record": {"workspace_root": "/app", "provider_id": "meta"}}})
            .to_string(),
        ];
        rows.extend(events.iter().map(|e| {
            json!({"schema_version": 1, "stream": {"kind": "session", "id": "s"},
                   "payload": {"kind": "run", "run_id": "run-1", "event": e}})
            .to_string()
        }));
        rows
    }

    fn summary(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .map(|e| match e {
                Event::ToolCall(c) => format!("tool {} {}", c.title, c.status.as_str()),
                Event::AgentMessage(t) => format!("message {}", t.text),
                Event::AgentThought(t) => format!("thought {}", t.text),
                other => other.kind().to_string(),
            })
            .collect()
    }

    #[tokio::test]
    async fn run_events_bare_and_enveloped_read_the_same() {
        let bare: Vec<String> = events().iter().map(Value::to_string).collect();
        for lines in [bare, enveloped(&events())] {
            let run = run_lines("muse-json", &lines).await;
            assert!(run.result.is_ok(), "{:?}", run.result);
            assert_eq!(
                summary(events_of(&run)),
                [
                    "user_message",
                    "thought Read the file.",
                    "message Reading",
                    "tool cat answer.txt completed",
                    "tool pwd completed",
                    "message The answer is 42."
                ]
            );
            let u = run.usage.unwrap();
            assert_eq!(
                (u.input_tokens, u.cached_read_tokens, u.output_tokens),
                (Some(32), Some(28), Some(8))
            );
        }
    }

    #[tokio::test]
    async fn a_committed_batch_replaces_single_results() {
        let mut ev = events();
        ev.insert(
            8,
            json!({"kind": "tool_result_batch_committed", "batch_id": "b",
                   "results": [{"tool_call_index": 0, "tool_call_id": "c1", "text": "42 (committed)"},
                               {"tool_call_index": 1, "tool_call_id": "c2", "text": "/app"}]}),
        );
        let lines: Vec<String> = ev.iter().map(Value::to_string).collect();
        let run = run_lines("muse-json", &lines).await;
        let call = events_of(&run)
            .iter()
            .find_map(|e| match e {
                Event::ToolCall(c) if c.title == "cat answer.txt" => Some(format!("{c:?}")),
                _ => None,
            })
            .unwrap();
        assert!(call.contains("42 (committed)"), "{call}");
    }

    #[tokio::test]
    async fn a_failed_terminal_fails_the_run_with_the_errors_words() {
        let lines = vec![
            json!({"kind": "started", "prompt": "hi"}).to_string(),
            json!({"kind": "model_failed", "error": {"message": "401 Unauthorized: invalid API key"}}).to_string(),
            json!({"kind": "terminal", "terminal": "failed", "reason": null}).to_string(),
        ];
        let run = run_lines_exit("muse-json", &lines, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    #[tokio::test]
    async fn unknown_events_and_missing_usage() {
        // test_unknown_events_and_missing_usage.
        let lines = vec![
            json!({"kind": "started", "prompt": "hello"}).to_string(),
            json!({"kind": "future_auxiliary_fact", "value": 123}).to_string(),
            json!({"kind": "assistant_message_committed", "message_id": "m", "text": "hello"})
                .to_string(),
            "muse: plain text goes to the log".to_string(),
        ];
        let run = run_lines("muse-json", &lines).await;
        assert!(run.result.is_ok());
        assert_eq!(summary(events_of(&run)), ["user_message", "message hello"]);
        assert_eq!(run.usage, None);
        assert_eq!(run.log, ["muse: plain text goes to the log"]);
    }
}
