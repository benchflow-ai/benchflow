//! `openhands-json`: `openhands --headless --json` (the OpenHands CLI).
//!
//! The CLI prints each event of its conversation as one JSON object, the
//! OpenHands SDK's own `model_dump()` with the event class in `kind`:
//!
//! - `ActionEvent`: a tool call (`tool_call_id`, `tool_name`, `tool_call`
//!   with `arguments` as JSON text, `action`), with the model's `thought`
//!   (text blocks) and `reasoning_content`;
//! - `ObservationEvent`: its result (`tool_call_id`, `observation` with
//!   `content` blocks and `is_error`); `UserRejectObservation` and
//!   `AgentErrorEvent` (`error`) end a call as failed;
//! - `MessageEvent` (`source`, `llm_message` with `role`, `content`,
//!   `reasoning_content`): the agent's words (the user's are the prompt);
//! - `ConversationErrorEvent` (`code`, `detail`): the run failed;
//! - `ConversationStateUpdateEvent`: state, including the LLM's
//!   `accumulated_token_usage` and `accumulated_cost` when it reports them.
//!
//! The CLI's console notices (a banner, `--json` warnings) go to the log.

use std::collections::HashSet;

use serde_json::Value;

use super::follow::{arguments, finish_call, start_call};
use crate::headless::parser::{
    Failure, Out, Parser, content_text, count, json_object, message, str_of, thought,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct OpenhandsJson {
    open: HashSet<String>,
}

impl Parser for OpenhandsJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "kind") {
            "ActionEvent" => self.action(&event, out),
            "ObservationEvent" => {
                let obs = event.get("observation").unwrap_or(&Value::Null);
                let failed = obs.get("is_error").and_then(Value::as_bool) == Some(true);
                let text = content_text(obs.get("content"));
                self.finish(&event, failed, text, out);
            }
            "UserRejectObservation" => {
                let text = Some(str_of(&event, "rejection_reason").to_string());
                self.finish(&event, true, text, out);
            }
            "AgentErrorEvent" => {
                let text = Some(str_of(&event, "error").to_string());
                if str_of(&event, "tool_call_id").is_empty() {
                    out.push(Out::Failure(Failure::notice(str_of(&event, "error"))));
                } else {
                    self.finish(&event, true, text, out);
                }
            }
            "MessageEvent" => {
                let msg = event.get("llm_message").unwrap_or(&Value::Null);
                if str_of(&event, "source") == "agent" || str_of(msg, "role") == "assistant" {
                    if let Some(text) = msg.get("reasoning_content").and_then(Value::as_str)
                        && !text.trim().is_empty()
                    {
                        out.push(thought(text));
                    }
                    if let Some(text) = content_text(msg.get("content"))
                        && !text.trim().is_empty()
                    {
                        out.push(message(&text));
                    }
                }
            }
            "ConversationErrorEvent" => {
                let detail = str_of(&event, "detail");
                let words = if detail.is_empty() {
                    str_of(&event, "code").to_string()
                } else {
                    detail.to_string()
                };
                out.push(Out::Failure(Failure::fatal(words)));
            }
            "ConversationStateUpdateEvent" => {
                if let Some(value) = event.get("value") {
                    usage(value, out);
                }
            }
            "" => return false,
            // Pauses, condensations, hook runs, streaming deltas.
            _ => {}
        }
        true
    }
}

impl OpenhandsJson {
    fn action(&mut self, event: &Value, out: &mut Vec<Out>) {
        if let Some(text) = event.get("reasoning_content").and_then(Value::as_str)
            && !text.trim().is_empty()
        {
            out.push(thought(text));
        }
        if let Some(text) = content_text(event.get("thought"))
            && !text.trim().is_empty()
        {
            out.push(thought(&text));
        }
        let call = event.get("tool_call").unwrap_or(&Value::Null);
        let id = [str_of(event, "tool_call_id"), str_of(call, "id")]
            .into_iter()
            .find(|i| !i.is_empty())
            .unwrap_or("")
            .to_string();
        let name = [str_of(event, "tool_name"), str_of(call, "name")]
            .into_iter()
            .find(|n| !n.is_empty())
            .unwrap_or("tool");
        let input = arguments(call.get("arguments")).or_else(|| {
            event.get("action").filter(|a| a.is_object()).map(|a| {
                let mut a = a.clone();
                a.as_object_mut().map(|m| m.remove("kind"));
                a
            })
        });
        if self.open.insert(id.clone()) {
            start_call(out, &id, name, input);
        }
    }

    fn finish(&mut self, event: &Value, failed: bool, text: Option<String>, out: &mut Vec<Out>) {
        let id = str_of(event, "tool_call_id").to_string();
        if self.open.insert(id.clone()) {
            start_call(out, &id, str_of(event, "tool_name"), None);
        }
        self.open.remove(&id);
        finish_call(out, &id, failed, text.filter(|t| !t.is_empty()));
    }
}

/// The token counts and cost in a state update: every
/// `accumulated_token_usage` (one per LLM: the agent's, a condenser's)
/// summed, and every `accumulated_cost`.
fn usage(value: &Value, out: &mut Vec<Out>) {
    let mut found: Vec<&Value> = Vec::new();
    let mut cost = 0.0;
    let mut has_cost = false;
    collect(value, &mut found, &mut cost, &mut has_cost, 0);
    if !found.is_empty() {
        let mut u = Usage::default();
        for t in found {
            let add = |slot: &mut Option<u64>, key: &str| {
                *slot = Some(slot.unwrap_or(0) + count(t.get(key)).unwrap_or(0));
            };
            add(&mut u.input_tokens, "prompt_tokens");
            add(&mut u.output_tokens, "completion_tokens");
            add(&mut u.cached_read_tokens, "cache_read_tokens");
            add(&mut u.cached_write_tokens, "cache_write_tokens");
            add(&mut u.thought_tokens, "reasoning_tokens");
        }
        // `prompt_tokens` includes the cache reads.
        if let (Some(i), Some(c)) = (u.input_tokens, u.cached_read_tokens) {
            u.input_tokens = Some(i.saturating_sub(c));
        }
        out.push(Out::Usage(u));
    }
    if has_cost {
        out.push(Out::Cost(cost));
    }
}

fn collect<'a>(
    value: &'a Value,
    found: &mut Vec<&'a Value>,
    cost: &mut f64,
    has_cost: &mut bool,
    depth: usize,
) {
    if depth > 8 {
        return;
    }
    match value {
        Value::Object(map) => {
            for (key, v) in map {
                match key.as_str() {
                    "accumulated_token_usage" if v.is_object() => found.push(v),
                    "accumulated_cost" => {
                        if let Some(c) = v.as_f64() {
                            *cost += c;
                            *has_cost = true;
                        }
                    }
                    _ => collect(v, found, cost, has_cost, depth + 1),
                }
            }
        }
        Value::Array(items) => {
            for v in items {
                collect(v, found, cost, has_cost, depth + 1);
            }
        }
        _ => {}
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

    /// SDK 1.21 events as the CLI's `json_callback` dumps them.
    #[tokio::test]
    async fn events_become_calls_results_and_the_answer() {
        let l = lines(&[
            json!({"kind": "MessageEvent", "id": "e1", "timestamp": "t", "source": "user",
                   "llm_message": {"role": "user", "content": [{"type": "text", "text": "Create hello.txt"}]}}),
            json!({"kind": "ActionEvent", "id": "e2", "timestamp": "t", "source": "agent",
                   "thought": [{"type": "text", "text": "I'll write the file."}],
                   "reasoning_content": null, "tool_name": "terminal", "tool_call_id": "call_1",
                   "tool_call": {"id": "call_1", "name": "terminal", "arguments": "{\"command\": \"echo hi > hello.txt\"}", "origin": "completion"},
                   "action": {"kind": "TerminalAction", "command": "echo hi > hello.txt"}, "llm_response_id": "r1"}),
            json!({"kind": "ObservationEvent", "id": "e3", "timestamp": "t", "source": "environment",
                   "tool_name": "terminal", "tool_call_id": "call_1", "action_id": "e2",
                   "observation": {"kind": "TerminalObservation", "content": [{"type": "text", "text": "ok"}], "is_error": false}}),
            json!({"kind": "ActionEvent", "id": "e4", "timestamp": "t", "source": "agent", "thought": [],
                   "tool_name": "str_replace_editor", "tool_call_id": "call_2",
                   "tool_call": {"id": "call_2", "name": "str_replace_editor", "arguments": "{\"command\": \"view\", \"path\": \"/nope\"}", "origin": "completion"}}),
            json!({"kind": "AgentErrorEvent", "id": "e5", "timestamp": "t", "source": "agent",
                   "tool_name": "str_replace_editor", "tool_call_id": "call_2", "error": "No such file"}),
            json!({"kind": "MessageEvent", "id": "e6", "timestamp": "t", "source": "agent",
                   "llm_message": {"role": "assistant", "content": [{"type": "text", "text": "Done."}]}}),
            json!({"kind": "ConversationStateUpdateEvent", "id": "e7", "timestamp": "t", "source": "environment",
                   "key": "stats", "value": {"usage_to_metrics": {"agent": {"accumulated_cost": 0.01,
                       "accumulated_token_usage": {"prompt_tokens": 300, "completion_tokens": 30, "cache_read_tokens": 100, "cache_write_tokens": 0, "reasoning_tokens": 5}}}}}),
        ]);
        let run = run_lines("openhands-json", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_thought",
                "tool_call",
                "tool_call",
                "agent_message"
            ]
        );
        assert!(matches!(&events[2], Event::ToolCall(c)
            if c.title == "echo hi > hello.txt" && c.status.as_str() == "completed"));
        assert!(matches!(&events[3], Event::ToolCall(c) if c.status.as_str() == "failed"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(200), Some(100), Some(30))
        );
        assert_eq!(run.cost_usd, Some(0.01));
    }

    #[tokio::test]
    async fn a_conversation_error_fails_the_run() {
        let l = vec![
            "Initializing agent...".to_string(),
            json!({"kind": "ConversationErrorEvent", "id": "e1", "timestamp": "t", "source": "environment",
                   "code": "AuthenticationError", "detail": "litellm.AuthenticationError: AnthropicException - invalid x-api-key"})
                .to_string(),
        ];
        let run = run_lines_exit("openhands-json", &l, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
