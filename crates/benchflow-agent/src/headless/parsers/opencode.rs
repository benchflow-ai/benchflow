//! `opencode-json`: `opencode run --format json`.
//!
//! One JSON event per line, each with `type`, `sessionID`, a millisecond
//! `timestamp` and a `part`:
//!
//! - `step_start`, `step_finish` (`part.tokens`: `input`, `output`,
//!   `reasoning`, `cache.read`, `cache.write`; `part.cost`): one model call;
//! - `text`, `reasoning` (`part.text`): the agent's words and thoughts,
//!   each whole;
//! - `tool_use` (`part.callID`, `part.tool`, `part.state` with `status`,
//!   `input`, `output`, `title`, `error`): a finished tool call;
//! - `user` (newer OpenCode): the prompt, which BenchFlow already recorded;
//! - `error` (`error.name`, `error.data.message`): the run failed.

use serde_json::Value;

use crate::headless::parser::{
    Failure, Hint, Out, Parser, ToolDone, ToolStart, count, json_object, kind_for_tool, message,
    one_line, str_of, thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct OpencodeJson {
    usage: Usage,
    cost: f64,
}

impl Parser for OpencodeJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        let part = event.get("part").unwrap_or(&Value::Null);
        match str_of(&event, "type") {
            "step_start" | "user" => {}
            "text" => {
                let text = str_of(part, "text");
                if !text.is_empty() {
                    out.push(message(text));
                }
            }
            "reasoning" => {
                let text = str_of(part, "text");
                if !text.trim().is_empty() {
                    out.push(thought(text));
                }
            }
            "tool_use" => tool_use(part, out),
            "step_finish" => {
                if let Some(tokens) = part.get("tokens") {
                    let cache = tokens.get("cache").unwrap_or(&Value::Null);
                    let add =
                        |a: Option<u64>, b: Option<u64>| Some(a.unwrap_or(0) + b.unwrap_or(0));
                    let u = &mut self.usage;
                    u.input_tokens = add(u.input_tokens, count(tokens.get("input")));
                    u.output_tokens = add(u.output_tokens, count(tokens.get("output")));
                    u.thought_tokens = add(u.thought_tokens, count(tokens.get("reasoning")));
                    u.cached_read_tokens = add(u.cached_read_tokens, count(cache.get("read")));
                    u.cached_write_tokens = add(u.cached_write_tokens, count(cache.get("write")));
                    out.push(Out::Usage(self.usage));
                }
                if let Some(cost) = part.get("cost").and_then(Value::as_f64) {
                    self.cost += cost;
                    out.push(Out::Cost(self.cost));
                }
            }
            "error" => {
                let error = event.get("error").unwrap_or(&Value::Null);
                let data = error.get("data").unwrap_or(&Value::Null);
                let text = [
                    str_of(data, "message"),
                    str_of(error, "message"),
                    str_of(error, "name"),
                ]
                .into_iter()
                .find(|t| !t.is_empty())
                .unwrap_or("OpenCode reported an error");
                // The provider's HTTP status, when OpenCode passes it on.
                let hint = match count(data.get("statusCode")) {
                    Some(401 | 403) => Some(Hint::Login),
                    Some(429 | 500..=599) => Some(Hint::Provider),
                    _ => None,
                };
                out.push(Out::Failure(
                    Failure::fatal(format!("OpenCode emitted error event(s): {text}")).hint(hint),
                ));
            }
            _ => return false,
        }
        true
    }
}

fn tool_use(part: &Value, out: &mut Vec<Out>) {
    let id = match str_of(part, "callID") {
        "" => str_of(part, "id"),
        id => id,
    };
    let tool = str_of(part, "tool");
    let state = part.get("state").unwrap_or(&Value::Null);
    let input = state.get("input").filter(|v| !v.is_null());
    let title = match str_of(state, "title") {
        "" => tool_title(tool, input),
        t if kind_for_tool(tool) == "execute" => one_line(t),
        t => format!("{tool} {}", one_line(t)),
    };
    let status = match str_of(state, "status") {
        "error" => "failed",
        "completed" => "completed",
        "pending" => "pending",
        _ => "in_progress",
    };
    out.push(
        ToolStart {
            id,
            title,
            kind: kind_for_tool(tool),
            status: "in_progress",
            input: input.cloned(),
            tool_name: Some(tool),
            parent: None,
        }
        .out(),
    );
    let text = match status {
        "failed" => Some(str_of(state, "error").to_string()),
        _ => state.get("output").and_then(|o| match o {
            Value::String(s) => Some(s.clone()),
            Value::Null => None,
            other => Some(other.to_string()),
        }),
    };
    out.push(
        ToolDone {
            id,
            status,
            text,
            output: state.get("output").filter(|v| !v.is_null()).cloned(),
            parent: None,
        }
        .out(),
    );
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::{Event, ToolCallStatus};

    // Harbor's test_opencode.py event builders.
    fn step_start(m: &str) -> Value {
        json!({"type": "step_start", "timestamp": 1700000000000u64, "sessionID": "s1",
               "part": {"id": format!("prt_start_{m}"), "sessionID": "s1", "messageID": m, "type": "step-start", "snapshot": "abc"}})
    }
    fn text(m: &str, t: &str) -> Value {
        json!({"type": "text", "timestamp": 1700000001000u64, "sessionID": "s1",
               "part": {"id": format!("prt_text_{m}"), "sessionID": "s1", "messageID": m, "type": "text", "text": t}})
    }
    fn reasoning(m: &str, t: &str) -> Value {
        json!({"type": "reasoning", "timestamp": 1700000001500u64, "sessionID": "s1",
               "part": {"id": format!("prt_reasoning_{m}"), "sessionID": "s1", "messageID": m, "type": "reasoning", "text": t}})
    }
    fn tool_use(m: &str, tool: &str, input: Value, output: &str, call: &str) -> Value {
        json!({"type": "tool_use", "timestamp": 1700000002000u64, "sessionID": "s1",
               "part": {"id": format!("prt_tool_{m}"), "sessionID": "s1", "messageID": m, "type": "tool", "callID": call, "tool": tool,
                        "state": {"status": "completed", "input": input, "output": output}}})
    }
    fn step_finish(
        m: &str,
        cost: f64,
        input: u64,
        output: u64,
        cache_read: u64,
        cache_write: u64,
    ) -> Value {
        json!({"type": "step_finish", "timestamp": 1700000003000u64, "sessionID": "s1",
               "part": {"id": format!("prt_finish_{m}"), "sessionID": "s1", "messageID": m, "type": "step-finish", "reason": "stop", "cost": cost,
                        "tokens": {"total": input + output + cache_read + cache_write, "input": input, "output": output, "reasoning": 0,
                                   "cache": {"read": cache_read, "write": cache_write}}}})
    }
    fn lines(events: &[Value]) -> Vec<String> {
        events.iter().map(Value::to_string).collect()
    }

    #[tokio::test]
    async fn a_tool_call_turn() {
        let run = run_lines(
            "opencode-json",
            &lines(&[
                json!({"type": "user", "timestamp": 1699999999000u64, "sessionID": "s1", "parts": [{"type": "text", "text": "the real prompt"}]}),
                step_start("m1"),
                reasoning("m1", "I should inspect README first."),
                text("m1", "Let me create that file."),
                tool_use("m1", "write", json!({"filePath": "/app/hello.txt", "content": "Hello!"}), "Wrote file successfully.", "toolu_abc"),
                step_finish("m1", 0.02, 200, 80, 0, 0),
                step_start("m2"),
                text("m2", "File created successfully."),
                step_finish("m2", 0.005, 150, 30, 100, 0),
            ]),
        )
        .await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        // The stream's own `user` event is not recorded twice.
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
        assert_eq!(call.tool_call_id, "toolu_abc");
        assert_eq!(call.kind, "edit");
        assert_eq!(call.title, "write /app/hello.txt");
        assert_eq!(call.tool_name.as_deref(), Some("write"));
        assert_eq!(call.status, ToolCallStatus::Completed);
        assert_eq!(call.raw_input.as_ref().unwrap()["content"], "Hello!");
        // Harbor's final metrics: 80 + 30 completion tokens, 100 cached, $0.025.
        let usage = run.usage.unwrap();
        assert_eq!(usage.output_tokens, Some(110));
        assert_eq!(usage.input_tokens, Some(350));
        assert_eq!(usage.cached_read_tokens, Some(100));
        assert!((run.cost_usd.unwrap() - 0.025).abs() < 1e-9);
    }

    #[tokio::test]
    async fn empty_reasoning_is_ignored_and_failed_tools_fail() {
        let mut failing = tool_use("m1", "bash", json!({"command": "false"}), "", "c2");
        failing["part"]["state"] =
            json!({"status": "error", "input": {"command": "false"}, "error": "exit 1"});
        let run = run_lines(
            "opencode-json",
            &lines(&[step_start("m1"), reasoning("m1", ""), failing]),
        )
        .await;
        let events = events_of(&run);
        assert_eq!(events.len(), 2, "{events:?}");
        let Event::ToolCall(call) = &events[1] else {
            panic!()
        };
        assert_eq!(call.status, ToolCallStatus::Failed);
        assert_eq!(call.title, "false");
    }

    #[tokio::test]
    async fn an_error_event_fails_the_run_with_its_message() {
        // Harbor: "OpenCode emitted error event(s): provider unavailable".
        let run = run_lines(
            "opencode-json",
            &lines(&[json!({"type": "error", "error": {"name": "ProviderError", "data": {"message": "provider unavailable"}}})]),
        )
        .await;
        let err = run.result.unwrap_err();
        assert!(
            err.to_string()
                .contains("OpenCode emitted error event(s): provider unavailable"),
            "{err}"
        );
        assert_eq!(err.category(), "infra_failure");
        let run = run_lines_exit(
            "opencode-json",
            &lines(&[json!({"type": "error", "error": {"name": "APIError", "data": {"message": "Incorrect API key provided: sk-…"}}})]),
            1,
        )
        .await;
        assert_eq!(run.result.unwrap_err().category(), "provider_auth");
        // Seen live: Anthropic's words for a bad key, and the status code.
        for data in [
            json!({"message": "invalid x-api-key"}),
            json!({"message": "unrecognized", "statusCode": 401}),
        ] {
            let run = run_lines_exit(
                "opencode-json",
                &lines(&[json!({"type": "error", "error": {"name": "APIError", "data": data}})]),
                1,
            )
            .await;
            assert_eq!(run.result.unwrap_err().category(), "provider_auth");
            assert!(
                run.log
                    .iter()
                    .any(|l| l.contains("the agent reported an error"))
            );
        }
    }
}
