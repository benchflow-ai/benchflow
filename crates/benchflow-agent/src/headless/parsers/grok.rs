//! `grok-streaming-json`: `grok --single ... --output-format streaming-json`
//! (xAI's Grok Build), its session's `chat_history.jsonl` followed.
//!
//! grok's stdout carries response chunks, tool activity, per-call `usage`
//! events, and a closing `end` (or `error`) event with the run's totals and
//! cost (`total_cost_usd`, or `total_cost_usd_ticks` at 10^10 ticks to the
//! dollar). Harbor's `grok-build` reads only the usage from it and the steps
//! from the session's chat history; so does this parser. The command runs
//! grok under `follow.sh` on `~/.grok/sessions/*/<session-id>/chat_history.jsonl`,
//! whose lines arrive as `{"@bf": "session", "item": <message>}`, each with a
//! `type`:
//!
//! - `system`, `user`: the prompts (the first user message is the task);
//! - `reasoning` (`summary` and `content` parts with `text`): thoughts;
//! - `assistant` (`content`: text or text blocks; `tool_calls`: `id`,
//!   `name`, `arguments` as an object or JSON text);
//! - `tool_result` (`tool_call_id`, `content`).
//!
//! Usage counts `input_tokens` apart from the cache
//! (`cache_read_input_tokens`, `cache_creation_input_tokens`), with
//! `reasoning_tokens` inside `output_tokens`.

use serde_json::Value;

use super::follow::{arguments, finish_call, followed, start_call};
use crate::headless::parser::{
    Failure, Out, Parser, content_text, count, json_object, message, str_of, thought,
};
use crate::trajectory::Usage;

const TICKS_PER_DOLLAR: f64 = 10_000_000_000.0;

#[derive(Default)]
pub struct GrokStreamingJson {
    /// The per-call usage, summed until `end` gives the total.
    calls: Usage,
    /// `end` reported the totals.
    final_usage: bool,
    /// Calls opened from the chat history and not yet answered.
    pending: Vec<String>,
}

impl Parser for GrokStreamingJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if let Some((name, item)) = followed(line) {
            if name == "session" {
                self.history(&item, out);
            }
            return true;
        }
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "type") {
            "usage" => {
                if !self.final_usage
                    && let Some(u) = event.get("usage").and_then(grok_usage)
                {
                    add(&mut self.calls, &u);
                    out.push(Out::Usage(self.calls));
                }
            }
            "end" | "error" => self.end(&event, out),
            // Response chunks and tool activity: the chat history has them whole.
            _ => {}
        }
        true
    }
}

impl GrokStreamingJson {
    fn history(&mut self, msg: &Value, out: &mut Vec<Out>) {
        match str_of(msg, "type") {
            "reasoning" => {
                let mut parts: Vec<&str> = Vec::new();
                for key in ["summary", "content"] {
                    for part in msg.get(key).and_then(Value::as_array).into_iter().flatten() {
                        let text = str_of(part, "text");
                        if !text.is_empty() {
                            parts.push(text);
                        }
                    }
                }
                let text = parts.join("\n");
                if !text.trim().is_empty() {
                    out.push(thought(text.trim()));
                }
            }
            "assistant" => {
                if let Some(text) = content_text(msg.get("content"))
                    && !text.trim().is_empty()
                {
                    out.push(message(&text));
                }
                for call in msg
                    .get("tool_calls")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let id = str_of(call, "id").to_string();
                    start_call(
                        out,
                        &id,
                        str_of(call, "name"),
                        arguments(call.get("arguments")),
                    );
                    self.pending.push(id);
                }
            }
            "tool_result" => {
                let id = str_of(msg, "tool_call_id").to_string();
                self.pending.retain(|p| *p != id);
                finish_call(out, &id, false, content_text(msg.get("content")));
            }
            _ => {}
        }
    }

    fn end(&mut self, event: &Value, out: &mut Vec<Out>) {
        if let Some(u) = event.get("usage").and_then(grok_usage) {
            self.final_usage = true;
            out.push(Out::Usage(u));
        }
        if event.get("cost_is_partial").and_then(Value::as_bool) != Some(true) {
            let cost = match event.get("total_cost_usd_ticks").and_then(Value::as_u64) {
                Some(ticks) => Some(ticks as f64 / TICKS_PER_DOLLAR),
                None => event
                    .get("total_cost_usd")
                    .and_then(Value::as_f64)
                    .filter(|c| c.is_finite() && *c >= 0.0),
            };
            if let Some(cost) = cost {
                out.push(Out::Cost(cost));
            }
        }
        if str_of(event, "type") == "error" {
            let words = match event.get("error").or_else(|| event.get("message")) {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Object(e)) => e
                    .get("message")
                    .and_then(Value::as_str)
                    .map_or_else(|| Value::Object(e.clone()).to_string(), str::to_string),
                Some(other) => other.to_string(),
                None => "Grok reported an error".to_string(),
            };
            out.push(Out::Failure(Failure::fatal(words)));
        }
    }
}

fn grok_usage(value: &Value) -> Option<Usage> {
    if !value.is_object() {
        return None;
    }
    let u = Usage {
        input_tokens: count(value.get("input_tokens")),
        output_tokens: count(value.get("output_tokens")),
        cached_read_tokens: count(value.get("cache_read_input_tokens")),
        cached_write_tokens: count(value.get("cache_creation_input_tokens")),
        thought_tokens: count(value.get("reasoning_tokens")),
        total_tokens: None,
    };
    (u != Usage::default()).then_some(u)
}

fn add(total: &mut Usage, u: &Usage) {
    let sum = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    };
    total.input_tokens = sum(total.input_tokens, u.input_tokens);
    total.output_tokens = sum(total.output_tokens, u.output_tokens);
    total.cached_read_tokens = sum(total.cached_read_tokens, u.cached_read_tokens);
    total.cached_write_tokens = sum(total.cached_write_tokens, u.cached_write_tokens);
    total.thought_tokens = sum(total.thought_tokens, u.thought_tokens);
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn session(item: Value) -> String {
        json!({"@bf": "session", "item": item}).to_string()
    }

    /// Harbor's test_grok_build_trajectory.py messages: a reasoning item, a
    /// tool call and its result, and the answer.
    fn history() -> Vec<String> {
        vec![
            session(json!({"type": "system", "content": "You are Grok."})),
            session(
                json!({"type": "user", "content": [{"type": "text", "text": "Create hello.txt"}]}),
            ),
            session(json!({"type": "reasoning", "id": "r1",
                           "summary": [{"type": "summary_text", "text": "From summary."}],
                           "content": [{"type": "reasoning_text", "text": "From content."}]})),
            session(
                json!({"type": "assistant", "content": "", "model_id": "grok-4.5",
                           "tool_calls": [{"id": "call_1", "name": "run_terminal_cmd",
                                           "arguments": "{\"command\": \"echo hi > hello.txt\"}"}]}),
            ),
            session(json!({"type": "tool_result", "tool_call_id": "call_1", "content": "ok"})),
            session(
                json!({"type": "assistant", "content": "Created hello.txt.", "model_id": "grok-4.5"}),
            ),
        ]
    }

    #[tokio::test]
    async fn the_history_is_the_trajectory_and_end_is_authoritative() {
        let mut l = vec![json!({"type": "assistant", "content": "hi"}).to_string()];
        for r in [
            json!({"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 50}),
            json!({"input_tokens": 30, "output_tokens": 5}),
        ] {
            l.push(json!({"type": "usage", "usage": r}).to_string());
        }
        l.extend(history());
        // The terminal aggregate includes a subagent's usage the calls lack.
        l.push(json!({"type": "end", "stopReason": "end_turn",
                      "usage": {"input_tokens": 200, "output_tokens": 20, "cache_read_input_tokens": 50,
                                "reasoning_tokens": 4},
                      "total_cost_usd_ticks": 123_000_000_u64}).to_string());
        let run = run_lines("grok-streaming-json", &l).await;
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
            matches!(&events[1], Event::AgentThought(t) if t.text == "From summary.\nFrom content.")
        );
        assert!(matches!(&events[2], Event::ToolCall(c)
            if c.title.contains("echo hi > hello.txt") && c.status.as_str() == "completed"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.output_tokens, u.thought_tokens),
            (Some(200), Some(20), Some(4))
        );
        assert_eq!(run.cost_usd, Some(0.0123));
    }

    #[tokio::test]
    async fn without_end_the_calls_are_summed() {
        let l = vec![
            json!({"type": "usage", "usage": {"input_tokens": 100, "output_tokens": 10}})
                .to_string(),
            json!({"type": "usage", "usage": {"input_tokens": 30, "output_tokens": 5}}).to_string(),
            json!({"type": "usage", "usage": {"input_tokens": null}}).to_string(),
        ];
        let run = run_lines("grok-streaming-json", &l).await;
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (Some(130), Some(15)));
    }

    #[tokio::test]
    async fn an_error_event_fails_the_run() {
        let l = vec![
            json!({"type": "error", "error": {"message": "Incorrect API key provided: xai-****"}})
                .to_string(),
        ];
        let run = run_lines_exit("grok-streaming-json", &l, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
