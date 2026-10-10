//! `pi-json`: `pi --print --mode json` (the Pi coding agent).
//!
//! One JSON event per line: the session header (`type: session`), then the
//! agent loop's events: `agent_start`, `turn_start`, `message_start`,
//! `message_update` (a streaming delta, without the partial message since
//! pi 1.1), `message_end`, `tool_execution_start`, `tool_execution_update`,
//! `tool_execution_end`, `turn_end`, `agent_end`. Each `message_end` holds a
//! whole message, which is what this parser reads (Harbor's `pi` reads the
//! same messages from the session file):
//!
//! - `role: assistant`: `content` blocks `text`, `thinking` (`thinking`) and
//!   `toolCall` (`id`, `name`, `arguments`); `usage` (`input`, `output`,
//!   `cacheRead`, `cacheWrite`, `cost.total`); `stopReason`, and on a failed
//!   model call `stopReason: error` with `errorMessage`;
//! - `role: toolResult`: `toolCallId`, `content`, `isError`;
//! - `role: user`: the prompt, which BenchFlow already recorded.
//!
//! In JSON mode pi exits 0 even when the model call failed, so an
//! assistant message that stopped on an error fails the run.

use serde_json::Value;

use super::follow::{finish_call, start_call};
use crate::headless::parser::{
    Failure, Out, Parser, content_text, count, json_object, message, str_of, thought,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct PiJson {
    usage: Usage,
    cost: f64,
}

impl Parser for PiJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "type") {
            "message_end" => {
                let msg = event.get("message").unwrap_or(&Value::Null);
                match str_of(msg, "role") {
                    "assistant" => self.assistant(msg, out),
                    "toolResult" => {
                        let failed = msg.get("isError").and_then(Value::as_bool) == Some(true);
                        finish_call(
                            out,
                            str_of(msg, "toolCallId"),
                            failed,
                            content_text(msg.get("content")),
                        );
                        self.usage(msg.get("usage"), out);
                    }
                    _ => {}
                }
            }
            "session"
            | "agent_start"
            | "agent_end"
            | "turn_start"
            | "turn_end"
            | "message_start"
            | "message_update"
            | "tool_execution_start"
            | "tool_execution_update"
            | "tool_execution_end"
            | "auto_compaction_start"
            | "auto_compaction_end"
            | "auto_retry_start"
            | "auto_retry_end"
            | "model_change" => {}
            _ => return false,
        }
        true
    }
}

impl PiJson {
    fn assistant(&mut self, msg: &Value, out: &mut Vec<Out>) {
        for block in msg
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            match str_of(block, "type") {
                "text" => {
                    let text = str_of(block, "text");
                    if !text.trim().is_empty() {
                        out.push(message(text));
                    }
                }
                "thinking" => {
                    let text = str_of(block, "thinking");
                    if !text.trim().is_empty() {
                        out.push(thought(text));
                    }
                }
                "toolCall" => {
                    let input = block.get("arguments").filter(|a| a.is_object()).cloned();
                    start_call(out, str_of(block, "id"), str_of(block, "name"), input);
                }
                _ => {}
            }
        }
        self.usage(msg.get("usage"), out);
        let stop = str_of(msg, "stopReason");
        let error = str_of(msg, "errorMessage");
        if stop == "error" && error != "This operation was aborted" {
            let words = if error.is_empty() {
                "pi's model call failed".to_string()
            } else {
                error.to_string()
            };
            out.push(Out::Failure(Failure::fatal(words)));
        }
    }

    fn usage(&mut self, usage: Option<&Value>, out: &mut Vec<Out>) {
        let Some(usage) = usage.filter(|u| u.is_object()) else {
            return;
        };
        let fields = ["input", "output", "cacheRead", "cacheWrite"];
        if !fields.iter().any(|k| usage.get(*k).is_some()) {
            return;
        }
        let add = |slot: &mut Option<u64>, key: &str| {
            *slot = Some(slot.unwrap_or(0) + count(usage.get(key)).unwrap_or(0));
        };
        let u = &mut self.usage;
        add(&mut u.input_tokens, "input");
        add(&mut u.output_tokens, "output");
        add(&mut u.cached_read_tokens, "cacheRead");
        add(&mut u.cached_write_tokens, "cacheWrite");
        out.push(Out::Usage(self.usage));
        if let Some(cost) = usage
            .get("cost")
            .and_then(|c| c.get("total"))
            .and_then(Value::as_f64)
            .filter(|c| *c > 0.0)
        {
            self.cost += cost;
            out.push(Out::Cost(self.cost));
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines};
    use crate::trajectory::Event;

    fn end(message: Value) -> String {
        json!({"type": "message_end", "message": message}).to_string()
    }

    fn usage(input: u64, output: u64, cache_read: u64, cost: f64) -> Value {
        json!({"input": input, "output": output, "cacheRead": cache_read, "cacheWrite": 0,
               "totalTokens": input + output + cache_read, "cost": {"total": cost}})
    }

    /// The conversation of Harbor's test_pi.py, as `--mode json` events.
    #[tokio::test]
    async fn messages_become_thoughts_calls_and_the_answer() {
        let l = vec![
            json!({"type": "session", "version": 3, "id": "sess-1", "cwd": "/app"}).to_string(),
            json!({"type": "agent_start"}).to_string(),
            end(
                json!({"role": "user", "content": [{"type": "text", "text": "Create hello.txt"}]}),
            ),
            json!({"type": "message_update", "usage": usage(1, 1, 0, 0.0), "assistantMessageEvent": {"type": "text_delta", "delta": "I"}}).to_string(),
            end(json!({"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Write the file."},
                {"type": "text", "text": "I'll write it."},
                {"type": "toolCall", "id": "call_1", "name": "write", "arguments": {"path": "hello.txt", "content": "hi"}}],
                "usage": usage(100, 20, 50, 0.002), "stopReason": "toolUse"})),
            json!({"type": "tool_execution_end", "toolCallId": "call_1", "toolName": "write", "isError": false}).to_string(),
            end(
                json!({"role": "toolResult", "toolCallId": "call_1", "toolName": "write",
                   "content": [{"type": "text", "text": "Wrote 2 bytes"}], "isError": false}),
            ),
            end(
                json!({"role": "assistant", "content": [{"type": "text", "text": "Done."}],
                   "usage": usage(30, 5, 0, 0.001), "stopReason": "stop"}),
            ),
            json!({"type": "agent_end", "messages": []}).to_string(),
        ];
        let run = run_lines("pi-json", &l).await;
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
            if c.title == "write hello.txt" && c.status.as_str() == "completed"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.output_tokens, u.cached_read_tokens),
            (Some(130), Some(25), Some(50))
        );
        assert!((run.cost_usd.unwrap() - 0.003).abs() < 1e-9);
        assert!(run.log.is_empty(), "{:?}", run.log);
    }

    #[tokio::test]
    async fn a_model_error_fails_the_run_though_pi_exits_zero() {
        let l = vec![end(json!({"role": "assistant", "content": [],
            "stopReason": "error", "errorMessage": "401 {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"}}"}))];
        let run = run_lines("pi-json", &l).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
