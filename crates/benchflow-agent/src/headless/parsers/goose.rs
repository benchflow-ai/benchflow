//! `goose-stream-json`: `goose run --output-format stream-json`.
//!
//! One JSON event per line:
//!
//! - `message` with `message: {id, role, content[]}`. Goose streams a
//!   message as many events with the same id, each a fragment. Content
//!   items: `text`, `thinking` (`thinking`, or `text`), `toolRequest` (`id`,
//!   `toolCall.value.{name, arguments}`, `toolCall.status`), and
//!   `toolResponse` (`id`, `toolResult.value.{content[], isError}`), which
//!   arrives in a `user` message (or, in older builds, the same one);
//! - `notification`: an extension's log or progress;
//! - `complete`: the run's usage (`input_tokens` includes the cache in goose
//!   1.37+; older builds report only `total_tokens`), `cost_usd` (1.43+);
//! - `error` with `error`: the run failed.

use serde_json::Value;

use crate::headless::parser::{
    Failure, Out, Parser, ToolDone, ToolStart, content_text, count, json_object, kind_for_tool,
    message, str_of, thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct GooseStreamJson;

impl Parser for GooseStreamJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "type") {
            "message" => {
                let msg = event.get("message").unwrap_or(&Value::Null);
                let assistant = str_of(msg, "role") == "assistant";
                for item in msg
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    content(item, assistant, out);
                }
            }
            "complete" => {
                let input = count(event.get("input_tokens"));
                let read = count(event.get("cache_read_input_tokens"));
                let write = count(event.get("cache_write_input_tokens"));
                let usage = Usage {
                    input_tokens: input.map(|i| {
                        i.saturating_sub(read.unwrap_or(0))
                            .saturating_sub(write.unwrap_or(0))
                    }),
                    output_tokens: count(event.get("output_tokens")),
                    total_tokens: count(event.get("total_tokens")),
                    cached_read_tokens: read,
                    cached_write_tokens: write,
                    thought_tokens: None,
                };
                if usage != Usage::default() {
                    out.push(Out::Usage(usage));
                }
                if let Some(cost) = event.get("cost_usd").and_then(Value::as_f64) {
                    out.push(Out::Cost(cost));
                }
            }
            "error" => {
                let text = match event.get("error") {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Null) | None => "goose reported an error".to_string(),
                    Some(other) => str_of(other, "message").to_string(),
                };
                out.push(Out::Failure(Failure::fatal(text)));
            }
            "notification" => {}
            _ => return false,
        }
        true
    }
}

fn content(item: &Value, assistant: bool, out: &mut Vec<Out>) {
    match str_of(item, "type") {
        "text" if assistant => {
            let text = str_of(item, "text");
            if !text.is_empty() {
                out.push(message(text));
            }
        }
        "thinking" => {
            let text = item
                .get("thinking")
                .or_else(|| item.get("text"))
                .and_then(Value::as_str)
                .unwrap_or("");
            if !text.is_empty() {
                out.push(thought(text));
            }
        }
        "toolRequest" => {
            let id = str_of(item, "id");
            let call = item.get("toolCall").unwrap_or(&Value::Null);
            let value = call.get("value").unwrap_or(&Value::Null);
            let name = match str_of(value, "name") {
                "" => "unknown",
                n => n,
            };
            let input = value.get("arguments");
            let failed = str_of(call, "status") == "error";
            out.push(
                ToolStart {
                    id,
                    title: tool_title(name, input),
                    kind: kind_for_tool(name),
                    status: if failed { "failed" } else { "in_progress" },
                    input: input.cloned(),
                    tool_name: Some(name),
                    parent: None,
                }
                .out(),
            );
        }
        "toolResponse" => {
            let result = item.get("toolResult").unwrap_or(&Value::Null);
            let value = result.get("value").unwrap_or(&Value::Null);
            let failed = str_of(result, "status") == "error"
                || value.get("isError").and_then(Value::as_bool) == Some(true);
            let text = content_text(value.get("content")).or_else(|| match result.get("error") {
                Some(Value::String(s)) => Some(s.clone()),
                _ => None,
            });
            out.push(
                ToolDone {
                    id: str_of(item, "id"),
                    status: if failed { "failed" } else { "completed" },
                    text,
                    output: Some(value.clone()).filter(|v| !v.is_null()),
                    parent: None,
                }
                .out(),
            );
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines};
    use crate::trajectory::{Event, ToolCallStatus};

    fn lines(events: &[Value]) -> Vec<String> {
        events.iter().map(Value::to_string).collect()
    }

    fn msg(id: &str, role: &str, content: Value) -> Value {
        json!({"type": "message", "message": {"id": id, "role": role, "created": 1708000001, "content": content}})
    }

    #[tokio::test]
    async fn harbors_sample_stream() {
        // test_goose_mcp.py's SAMPLE_JSONL.
        let run = run_lines(
            "goose-stream-json",
            &lines(&[
                msg("msg-user-1", "user", json!([{"type": "text", "text": "Complete the task."}])),
                msg("msg-asst-1", "assistant", json!([
                    {"type": "thinking", "text": "I need to list files first."},
                    {"type": "text", "text": "Let me check the directory contents."},
                    {"type": "toolRequest", "id": "tool-1", "toolCall": {"status": "success", "value": {"name": "shell", "arguments": {"command": "ls /app"}}}},
                    {"type": "toolResponse", "id": "tool-1", "toolResult": {"status": "success", "value": {"content": [{"type": "text", "text": "file1.txt\nfile2.txt"}], "isError": false}}},
                ])),
                msg("msg-asst-2", "assistant", json!([{"type": "text", "text": "Task complete."}])),
                json!({"type": "complete", "total_tokens": 1500}),
            ]),
        )
        .await;
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
        let Event::ToolCall(call) = &events[3] else {
            panic!()
        };
        assert_eq!(
            (call.tool_call_id.as_str(), call.title.as_str()),
            ("tool-1", "ls /app")
        );
        assert_eq!(call.kind, "execute");
        assert_eq!(call.status, ToolCallStatus::Completed);
        assert_eq!(
            call.raw_input.as_ref().unwrap(),
            &json!({"command": "ls /app"})
        );
        assert_eq!(run.usage.unwrap().total_tokens, Some(1500));
    }

    #[tokio::test]
    async fn streaming_chunks_join_and_responses_land_on_their_call() {
        let run = run_lines(
            "goose-stream-json",
            &lines(&[
                msg("msg-1", "assistant", json!([{"type": "text", "text": "Hello"}])),
                msg("msg-1", "assistant", json!([{"type": "text", "text": " world"}])),
                msg("msg-1", "assistant", json!([{"type": "text", "text": "!"}])),
                msg("msg-1", "assistant", json!([{"type": "toolRequest", "id": "tc-1", "toolCall": {"status": "success", "value": {"name": "developer__shell", "arguments": {"command": "ls"}}}}])),
                msg("msg-user-resp", "user", json!([{"type": "toolResponse", "id": "tc-1", "toolResult": {"status": "success", "value": {"content": [{"type": "text", "text": "file.txt"}], "isError": false}}}])),
                msg("msg-2", "assistant", json!([{"type": "text", "text": "Done"}])),
                msg("msg-2", "assistant", json!([{"type": "text", "text": "."}])),
                json!({"type": "complete", "input_tokens": 1000, "output_tokens": 200, "total_tokens": 1200,
                       "cache_read_input_tokens": 800, "cache_write_input_tokens": 150, "cost_usd": 0.42}),
            ]),
        )
        .await;
        let events = events_of(&run);
        let texts: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                Event::AgentMessage(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, ["Hello world!", "Done."]);
        let Event::ToolCall(call) = &events[2] else {
            panic!()
        };
        assert_eq!(call.status, ToolCallStatus::Completed);
        // goose's input includes the cache: 1000 - 800 - 150.
        let usage = run.usage.unwrap();
        assert_eq!(
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_read_tokens,
                usage.cached_write_tokens
            ),
            (Some(50), Some(200), Some(800), Some(150))
        );
        assert_eq!(run.cost_usd, Some(0.42));
    }

    #[tokio::test]
    async fn thinking_blocks_keep_their_reasoning_and_the_call_after_them() {
        // test_goose_thinking_blocks.py.
        let run = run_lines(
            "goose-stream-json",
            &lines(&[
                msg("m1", "assistant", json!([
                    {"type": "thinking", "thinking": "Need to read the file.", "signature": "s"},
                    {"type": "toolRequest", "id": "t1", "toolCall": {"value": {"name": "developer__shell", "arguments": {}}}},
                ])),
                json!({"type": "complete", "total_tokens": 10}),
            ]),
        )
        .await;
        let events = events_of(&run);
        assert!(matches!(&events[1], Event::AgentThought(t) if t.text == "Need to read the file."));
        assert!(
            matches!(&events[2], Event::ToolCall(c) if c.tool_name.as_deref() == Some("developer__shell"))
        );
    }

    #[tokio::test]
    async fn an_error_event_fails_the_run_and_noise_is_kept() {
        let run = run_lines(
            "goose-stream-json",
            &[
                "starting session | provider: anthropic".to_string(),
                json!({"type": "error", "error": "Something went wrong"}).to_string(),
            ],
        )
        .await;
        let err = run.result.unwrap_err();
        assert!(err.to_string().contains("Something went wrong"), "{err}");
        assert!(run.log.iter().any(|l| l.contains("starting session")));
    }
}
