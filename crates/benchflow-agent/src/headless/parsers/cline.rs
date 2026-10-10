//! `cline-json`: `cline --json --yolo` (Cline's CLI), then its session's
//! messages.
//!
//! Cline prints its progress as JSON lines on stdout, each with a `type`:
//! `agent_event` (`event.type`: `iteration_start`, `content_start`,
//! `content_end`, `error`, ...), `error` (`message`), and a closing
//! `run_result` (`finishReason`, `text`, and `aggregateUsage` or `usage`
//! with `inputTokens`, `outputTokens`, `cacheReadTokens`,
//! `cacheWriteTokens`, `totalCost`). The run's result gives the usage, the
//! cost, and a failure (`finishReason: error`, its `text` the reason);
//! `error` lines go to the log and explain it. It keeps the conversation in
//! `~/.cline/data/sessions/<id>/<id>.messages.json`; after `cline` exits,
//! the command prints the newest session's file as `{"@bf": "messages",
//! "item": <file>}` (Harbor's `cline-cli` converts the same file). The file
//! is `{"sessionId", "messages": [...]}`, each message `{"role", "content",
//! "modelInfo", "metrics"}`:
//!
//! - content: a string, or Anthropic-style blocks: `text`, `thinking`
//!   (`text` or `thinking`), `tool_use` (`id`, `name`, `input`) in assistant
//!   messages, `tool_result` (`tool_use_id`, `content`, `is_error`) in user
//!   messages;
//! - `metrics` on assistant messages: `inputTokens`, `outputTokens`,
//!   `cacheReadTokens`, `cacheWriteTokens`, `cost`.
//!
//! When no session file arrives (the run was stopped or Cline failed before
//! saving one), the console's text lines are the agent's message.

use serde_json::Value;

use super::follow::{finish_call, followed, start_call};
use crate::headless::parser::{
    Failure, Out, Parser, content_text, count, json_object, message, str_of, thought,
};
use crate::trajectory::Usage;

/// Console characters kept for a run that saves no session.
const KEEP_CHARS: usize = 256 * 1024;

#[derive(Default)]
pub struct ClineJson {
    console: String,
    sessions: usize,
    /// `run_result` reported the run's usage.
    result_usage: bool,
}

impl Parser for ClineJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if let Some((name, item)) = followed(line) {
            if name == "messages" {
                self.session(&item, out);
            }
            return true;
        }
        let event = json_object(line).map(Value::Object);
        let kind = event.as_ref().map_or("", |e| str_of(e, "type"));
        let understood = match (kind, &event) {
            ("run_result", Some(e)) => {
                self.result(e, out);
                true
            }
            ("agent_event", _) => true,
            _ => false,
        };
        if self.console.len() < KEEP_CHARS {
            // The words of a JSON line are its `text` or `message`.
            let text = match &event {
                Some(e) => ["text", "message"]
                    .iter()
                    .map(|k| str_of(e, k))
                    .find(|t| !t.is_empty())
                    .unwrap_or("")
                    .to_string(),
                None => line.to_string(),
            };
            if !text.trim().is_empty() && kind != "error" {
                self.console.push_str(&text);
                self.console.push('\n');
            }
        }
        understood
    }

    fn finish(&mut self, out: &mut Vec<Out>) {
        if self.sessions == 0 && !self.console.trim().is_empty() {
            out.push(message(&self.console));
        }
    }
}

impl ClineJson {
    fn result(&mut self, event: &Value, out: &mut Vec<Out>) {
        let usage = event
            .get("aggregateUsage")
            .or_else(|| event.get("usage"))
            .filter(|u| u.is_object());
        if let Some(u) = usage {
            let usage = Usage {
                input_tokens: count(u.get("inputTokens")),
                output_tokens: count(u.get("outputTokens")),
                cached_read_tokens: count(u.get("cacheReadTokens")),
                cached_write_tokens: count(u.get("cacheWriteTokens")),
                ..Usage::default()
            };
            if usage != Usage::default() {
                self.result_usage = true;
                out.push(Out::Usage(usage));
            }
            if let Some(cost) = u
                .get("totalCost")
                .and_then(Value::as_f64)
                .filter(|c| *c > 0.0)
            {
                out.push(Out::Cost(cost));
            }
        }
        if str_of(event, "finishReason") == "error" {
            let text = str_of(event, "text").trim();
            let words = if text.is_empty() {
                "Cline's run ended with an error".to_string()
            } else {
                text.to_string()
            };
            out.push(Out::Failure(Failure::fatal(words)));
        }
    }

    fn session(&mut self, doc: &Value, out: &mut Vec<Out>) {
        let Some(messages) = doc.get("messages").and_then(Value::as_array) else {
            return;
        };
        self.sessions += 1;
        let mut usage = Usage::default();
        let mut has_usage = false;
        let mut cost = 0.0;
        for msg in messages {
            let content = msg.get("content");
            match str_of(msg, "role") {
                "assistant" => {
                    let blocks = match content {
                        Some(Value::Array(blocks)) => blocks.as_slice(),
                        Some(Value::String(text)) => {
                            if !text.trim().is_empty() {
                                out.push(message(text));
                            }
                            &[]
                        }
                        _ => &[],
                    };
                    for block in blocks {
                        match str_of(block, "type") {
                            "text" => {
                                let text = str_of(block, "text");
                                if !text.trim().is_empty() {
                                    out.push(message(text));
                                }
                            }
                            "thinking" => {
                                let text = [str_of(block, "text"), str_of(block, "thinking")]
                                    .into_iter()
                                    .find(|t| !t.is_empty())
                                    .unwrap_or("");
                                if !text.trim().is_empty() {
                                    out.push(thought(text));
                                }
                            }
                            "tool_use" => {
                                let input = block.get("input").filter(|i| i.is_object()).cloned();
                                start_call(out, str_of(block, "id"), str_of(block, "name"), input);
                            }
                            _ => {}
                        }
                    }
                    if let Some(m) = msg.get("metrics").filter(|m| m.is_object()) {
                        let add = |slot: &mut Option<u64>, key: &str| {
                            if let Some(n) = count(m.get(key)) {
                                *slot = Some(slot.unwrap_or(0) + n);
                            }
                        };
                        add(&mut usage.input_tokens, "inputTokens");
                        add(&mut usage.output_tokens, "outputTokens");
                        add(&mut usage.cached_read_tokens, "cacheReadTokens");
                        add(&mut usage.cached_write_tokens, "cacheWriteTokens");
                        has_usage |= usage != Usage::default();
                        cost += m.get("cost").and_then(Value::as_f64).unwrap_or(0.0);
                    }
                }
                "user" => {
                    for block in content.and_then(Value::as_array).into_iter().flatten() {
                        if str_of(block, "type") == "tool_result" {
                            let failed =
                                block.get("is_error").and_then(Value::as_bool) == Some(true);
                            finish_call(
                                out,
                                str_of(block, "tool_use_id"),
                                failed,
                                content_text(block.get("content")),
                            );
                        }
                    }
                }
                _ => {}
            }
        }
        // The run's result, when it came, already counted the whole run.
        if has_usage && !self.result_usage {
            out.push(Out::Usage(usage));
            if cost > 0.0 {
                out.push(Out::Cost(cost));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    /// Harbor's test_cline_trajectory.py conversation.
    fn doc() -> serde_json::Value {
        json!({"sessionId": "sess-1", "messages": [
            {"role": "user", "content": "List files.", "ts": 1},
            {"role": "assistant", "ts": 2, "modelInfo": {"id": "m"},
             "content": [{"type": "thinking", "thinking": "ls will do."},
                         {"type": "text", "text": "I'll list them."},
                         {"type": "tool_use", "id": "call_abc", "name": "run_commands", "input": {"commands": ["ls"]}}],
             "metrics": {"inputTokens": 100, "outputTokens": 10, "cacheReadTokens": 20, "cacheWriteTokens": 0, "cost": 0.001}},
            {"role": "user", "ts": 3, "content": [{"type": "tool_result", "tool_use_id": "call_abc", "content": "file1\nfile2"}]},
            {"role": "assistant", "ts": 4, "modelInfo": {"id": "m"}, "content": [{"type": "text", "text": "Done."}],
             "metrics": {"inputTokens": 50, "outputTokens": 5, "cost": 0.0005}}
        ]})
    }

    #[tokio::test]
    async fn the_session_file_is_the_trajectory() {
        let l = vec![
            json!({"type": "say", "say": "text", "text": "I'll list them."}).to_string(),
            json!({"@bf": "messages", "item": doc()}).to_string(),
        ];
        let run = run_lines("cline-json", &l).await;
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
            ]
        );
        assert!(matches!(&events[3], Event::ToolCall(c)
            if c.title == "run_commands" && c.status.as_str() == "completed"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.output_tokens, u.cached_read_tokens),
            (Some(150), Some(15), Some(20))
        );
        assert!((run.cost_usd.unwrap() - 0.0015).abs() < 1e-9);
    }

    #[tokio::test]
    async fn the_run_result_counts_the_run() {
        let l = vec![
            json!({"ts": "t", "type": "agent_event", "event": {"type": "iteration_start", "iteration": 1}}).to_string(),
            json!({"ts": "t", "type": "run_result", "finishReason": "completed", "iterations": 2, "text": "Done.",
                   "usage": {"inputTokens": 50, "outputTokens": 5, "cacheReadTokens": 0, "cacheWriteTokens": 0, "totalCost": 0.0005},
                   "aggregateUsage": {"inputTokens": 150, "outputTokens": 15, "cacheReadTokens": 20, "cacheWriteTokens": 0, "totalCost": 0.0015}}).to_string(),
            json!({"@bf": "messages", "item": doc()}).to_string(),
        ];
        let run = run_lines("cline-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.output_tokens, u.cached_read_tokens),
            (Some(150), Some(15), Some(20))
        );
        assert!((run.cost_usd.unwrap() - 0.0015).abs() < 1e-9);
        assert!(run.log.is_empty(), "{:?}", run.log);
    }

    /// Cline 3.0.70 with a rejected key, as seen live.
    #[tokio::test]
    async fn without_a_session_the_console_is_the_message() {
        let l = vec![
            r#"{"ts":"2026-10-10T14:01:58.410Z","type":"agent_event","event":{"type":"iteration_start","iteration":1}}"#.to_string(),
            r#"{"ts":"2026-10-10T14:01:58.620Z","type":"error","message":"invalid x-api-key"}"#.to_string(),
            r#"{"ts":"2026-10-10T14:01:58.612Z","type":"agent_event","event":{"type":"error","error":{"name":"Error","message":"invalid x-api-key"},"errorClass":"auth","recoverable":false,"iteration":1}}"#.to_string(),
            r#"{"ts":"2026-10-10T14:01:58.620Z","type":"run_result","finishReason":"error","iterations":1,"usage":{"inputTokens":0,"outputTokens":0,"cacheReadTokens":0,"cacheWriteTokens":0,"totalCost":0},"durationMs":269,"text":"invalid x-api-key"}"#.to_string(),
        ];
        let run = run_lines_exit("cline-json", &l, 1).await;
        assert!(
            events_of(&run).iter().any(
                |e| matches!(e, Event::AgentMessage(t) if t.text.contains("invalid x-api-key"))
            )
        );
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
