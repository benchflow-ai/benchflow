//! `qwen-session`: `qwen --prompt ... --chat-recording` (Qwen Code), its
//! chat recording followed.
//!
//! Qwen Code prints its answer as text and records the conversation in
//! `~/.qwen/projects/<project>/chats/<session>.jsonl`. The command runs it
//! under `follow.sh` on that file, so each record arrives as `{"@bf":
//! "session", "item": <record>}` while it works (Harbor's `qwen-coder`
//! converts the same file after the run). A record has a `type`, a
//! `message` with Gemini-style `parts`, and on model turns
//! `usageMetadata`:
//!
//! - `user`: the prompt, which BenchFlow already recorded;
//! - `assistant`: `text` parts (`thought: true` marks reasoning) and
//!   `functionCall` parts (`id`, `name`, `args`); `usageMetadata` with
//!   `promptTokenCount` (including `cachedContentTokenCount`),
//!   `candidatesTokenCount`, `thoughtsTokenCount`;
//! - `tool_result`: `functionResponse` parts (`id`, `response.output` or
//!   `response.error`).
//!
//! The console goes to the log; when no model turn is recorded (the run was
//! stopped, or failed first), it is the agent's message.

use serde_json::Value;

use super::follow::{finish_call, followed, start_call};
use crate::headless::parser::{Out, Parser, count, message, str_of, thought};
use crate::trajectory::Usage;

const KEEP_CHARS: usize = 256 * 1024;

#[derive(Default)]
pub struct QwenSession {
    console: String,
    turns: usize,
    usage: Usage,
}

impl Parser for QwenSession {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if let Some((name, item)) = followed(line) {
            if name == "session" {
                self.record(&item, out);
            }
            return true;
        }
        if self.console.len() < KEEP_CHARS {
            self.console.push_str(line);
            self.console.push('\n');
        }
        false
    }

    fn finish(&mut self, out: &mut Vec<Out>) {
        if self.turns == 0 && !self.console.trim().is_empty() {
            out.push(message(&self.console));
        }
    }
}

impl QwenSession {
    fn record(&mut self, record: &Value, out: &mut Vec<Out>) {
        let msg = record.get("message").unwrap_or(&Value::Null);
        let parts = msg.get("parts").and_then(Value::as_array);
        match str_of(record, "type") {
            "assistant" => {
                self.turns += 1;
                for part in parts.into_iter().flatten() {
                    if let Some(text) = part.get("text").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        if part.get("thought").and_then(Value::as_bool) == Some(true) {
                            out.push(thought(text));
                        } else {
                            out.push(message(text));
                        }
                    }
                    if let Some(call) = part.get("functionCall") {
                        let input = call.get("args").filter(|a| a.is_object()).cloned();
                        start_call(out, str_of(call, "id"), str_of(call, "name"), input);
                    }
                }
                if let Some(meta) = record.get("usageMetadata").filter(|m| m.is_object()) {
                    let prompt = count(meta.get("promptTokenCount")).unwrap_or(0);
                    let cached = count(meta.get("cachedContentTokenCount")).unwrap_or(0);
                    let add = |slot: &mut Option<u64>, n: u64| *slot = Some(slot.unwrap_or(0) + n);
                    let u = &mut self.usage;
                    add(&mut u.input_tokens, prompt.saturating_sub(cached));
                    add(&mut u.cached_read_tokens, cached.min(prompt));
                    add(
                        &mut u.output_tokens,
                        count(meta.get("candidatesTokenCount")).unwrap_or(0),
                    );
                    add(
                        &mut u.thought_tokens,
                        count(meta.get("thoughtsTokenCount")).unwrap_or(0),
                    );
                    out.push(Out::Usage(self.usage));
                }
            }
            "tool_result" => {
                for part in parts.into_iter().flatten() {
                    let Some(response) = part.get("functionResponse") else {
                        continue;
                    };
                    let body = response.get("response").unwrap_or(&Value::Null);
                    let (failed, text) = match (body.get("error"), body.get("output")) {
                        (Some(e), _) if !e.is_null() => (true, value_text(e)),
                        (_, Some(o)) => (false, value_text(o)),
                        _ => (false, value_text(body)),
                    };
                    finish_call(out, str_of(response, "id"), failed, text);
                }
            }
            _ => {}
        }
    }
}

fn value_text(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => Some(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn session(item: Value) -> String {
        json!({"@bf": "session", "item": item}).to_string()
    }

    /// Records shaped like Harbor's test_qwen_code_trajectory.py fixture.
    #[tokio::test]
    async fn the_recording_is_the_trajectory() {
        let l = vec![
            "Created hello.txt.".to_string(),
            session(
                json!({"type": "user", "sessionId": "s1", "timestamp": "t", "message": {"role": "user", "parts": [{"text": "Create hello.txt"}]}}),
            ),
            session(
                json!({"type": "assistant", "timestamp": "t", "model": "qwen3-coder-plus",
                "message": {"role": "model", "parts": [
                    {"text": "Planning the write.", "thought": true},
                    {"functionCall": {"id": "call_1", "name": "write_file", "args": {"file_path": "/app/hello.txt", "content": "hi"}}}]},
                "usageMetadata": {"promptTokenCount": 1000, "candidatesTokenCount": 50, "cachedContentTokenCount": 400, "thoughtsTokenCount": 10}}),
            ),
            session(json!({"type": "tool_result", "timestamp": "t",
                "message": {"role": "user", "parts": [{"functionResponse": {"id": "call_1", "name": "write_file", "response": {"output": "Wrote /app/hello.txt"}}}]}})),
            session(json!({"type": "assistant", "timestamp": "t",
                "message": {"role": "model", "parts": [{"text": "Created hello.txt."}]},
                "usageMetadata": {"promptTokenCount": 1100, "candidatesTokenCount": 8}})),
        ];
        let run = run_lines("qwen-session", &l).await;
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
        assert!(matches!(&events[2], Event::ToolCall(c)
            if c.title == "write_file /app/hello.txt" && c.kind == "edit" && c.status.as_str() == "completed"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(1700), Some(400), Some(58))
        );
        assert_eq!(run.log, ["Created hello.txt."]);
    }

    #[tokio::test]
    async fn without_a_recording_the_console_explains_the_failure() {
        let l = vec!["[API Error: 401 Incorrect API key provided.]".to_string()];
        let run = run_lines_exit("qwen-session", &l, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
