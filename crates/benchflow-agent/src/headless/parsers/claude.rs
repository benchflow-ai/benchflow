//! `claude-stream-json`: `claude -p --verbose --output-format stream-json`.
//!
//! One JSON object per line:
//!
//! - `system` (`init`, `api_retry`, ...): bookkeeping;
//! - `assistant`: one model message (Claude Code writes each content block
//!   on its own line, all with the same `message.id`): `text`, `thinking`
//!   and `tool_use` blocks, the message's `usage`, and on a failed call an
//!   `error` (`authentication_failed`, `billing_error`, `rate_limit`, ...)
//!   with the CLI's words as the text;
//! - `user`: tool results (`tool_result` blocks, `is_error` on failure);
//! - `result`: the run's end, with cumulative `usage`, `total_cost_usd`,
//!   and `is_error` with the final text when it failed;
//! - `rate_limit_event`: a subscription window's state; `rejected` means
//!   the window is spent.
//!
//! `parent_tool_use_id` marks a subagent's messages and calls; it becomes
//! the trajectory's subagent attribution.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::headless::parser::{
    Failure, Hint, Out, Parser, ToolDone, ToolStart, content_text, count, json_object,
    kind_for_tool, message, str_of, thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct ClaudeStreamJson {
    /// Each model message's usage, by message id (a message's blocks repeat it).
    per_message: BTreeMap<String, Usage>,
    /// Tool call ids already opened.
    open: HashMap<String, ()>,
    /// The `result` event reported the run's totals.
    final_usage: bool,
}

impl Parser for ClaudeStreamJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        let parent = event
            .get("parent_tool_use_id")
            .and_then(Value::as_str)
            .filter(|p| !p.is_empty());
        match str_of(&event, "type") {
            "assistant" => self.assistant(&event, parent, out),
            "user" => self.user(&event, parent, out),
            "result" => self.result(&event, out),
            "rate_limit_event" => {
                let info = event.get("rate_limit_info").unwrap_or(&Value::Null);
                if str_of(info, "status") == "rejected" {
                    let window = str_of(info, "rateLimitType");
                    out.push(Out::Failure(
                        Failure::notice(format!(
                            "Claude subscription {} limit rejected the request",
                            if window.is_empty() { "usage" } else { window }
                        ))
                        .hint(Some(Hint::UsageLimit)),
                    ));
                }
            }
            "system" | "stream_event" => {}
            _ => return false,
        }
        true
    }
}

impl ClaudeStreamJson {
    fn assistant(&mut self, event: &Value, parent: Option<&str>, out: &mut Vec<Out>) {
        let msg = event.get("message").unwrap_or(&Value::Null);
        let mut text = String::new();
        if let Some(blocks) = msg.get("content").and_then(Value::as_array) {
            for block in blocks {
                match str_of(block, "type") {
                    "text" => {
                        let t = str_of(block, "text");
                        text.push_str(t);
                        if !t.is_empty() {
                            out.push(with_parent(message(t), parent));
                        }
                    }
                    "thinking" => {
                        let t = block
                            .get("thinking")
                            .or_else(|| block.get("text"))
                            .and_then(Value::as_str)
                            .unwrap_or("");
                        if !t.is_empty() {
                            out.push(with_parent(thought(t), parent));
                        }
                    }
                    "tool_use" | "server_tool_use" | "mcp_tool_use" => {
                        let id = str_of(block, "id");
                        let name = str_of(block, "name");
                        let input = block.get("input");
                        self.open.insert(id.to_string(), ());
                        out.push(
                            ToolStart {
                                id,
                                title: tool_title(name, input),
                                kind: kind_for_tool(name),
                                status: "in_progress",
                                input: input.cloned(),
                                tool_name: Some(name),
                                parent,
                            }
                            .out(),
                        );
                    }
                    _ => {}
                }
            }
        }
        // A failed model call: the CLI's words are the message text.
        if let Some(error) = event
            .get("error")
            .or_else(|| msg.get("error"))
            .and_then(Value::as_str)
        {
            let hint = match error {
                "authentication_failed" => Some(Hint::Login),
                "billing_error" => Some(Hint::UsageLimit),
                "rate_limit" | "server_error" | "overloaded" => Some(Hint::Provider),
                _ => None,
            };
            let words = if text.trim().is_empty() {
                error.to_string()
            } else {
                text.trim().to_string()
            };
            out.push(Out::Failure(Failure::notice(words).hint(hint)));
        }
        if !self.final_usage
            && let Some(usage) = msg.get("usage").and_then(anthropic_usage)
        {
            let id = str_of(msg, "id");
            let key = if id.is_empty() {
                format!("#{}", self.per_message.len())
            } else {
                id.to_string()
            };
            self.per_message.insert(key, usage);
            out.push(Out::Usage(sum(self.per_message.values())));
        }
    }

    fn user(&mut self, event: &Value, parent: Option<&str>, out: &mut Vec<Out>) {
        let msg = event.get("message").unwrap_or(&Value::Null);
        let Some(blocks) = msg.get("content").and_then(Value::as_array) else {
            return;
        };
        for block in blocks {
            if str_of(block, "type") != "tool_result" {
                continue;
            }
            let id = str_of(block, "tool_use_id");
            let failed = block.get("is_error").and_then(Value::as_bool) == Some(true);
            let text = content_text(block.get("content"));
            if !self.open.contains_key(id) {
                // A result for a call this stream never opened.
                self.open.insert(id.to_string(), ());
                out.push(
                    ToolStart {
                        id,
                        title: "tool".to_string(),
                        kind: "other",
                        status: "in_progress",
                        parent,
                        ..ToolStart::default()
                    }
                    .out(),
                );
            }
            let output = event
                .get("tool_use_result")
                .filter(|v| !v.is_null())
                .cloned()
                .or_else(|| block.get("content").cloned());
            out.push(
                ToolDone {
                    id,
                    status: if failed { "failed" } else { "completed" },
                    text,
                    output,
                    parent,
                }
                .out(),
            );
        }
    }

    fn result(&mut self, event: &Value, out: &mut Vec<Out>) {
        if let Some(usage) = event.get("usage").and_then(anthropic_usage) {
            self.final_usage = true;
            out.push(Out::Usage(usage));
        }
        if let Some(cost) = event.get("total_cost_usd").and_then(Value::as_f64) {
            out.push(Out::Cost(cost));
        }
        if event.get("is_error").and_then(Value::as_bool) == Some(true) {
            let text = str_of(event, "result").trim();
            let subtype = str_of(event, "subtype");
            let words = if !text.is_empty() {
                text.to_string()
            } else if let Some(errors) = event.get("errors").and_then(Value::as_array) {
                errors
                    .iter()
                    .map(|e| e.as_str().map_or_else(|| e.to_string(), str::to_string))
                    .collect::<Vec<_>>()
                    .join("; ")
            } else {
                format!("Claude Code ended with {subtype}")
            };
            out.push(Out::Failure(Failure::fatal(words)));
        }
    }
}

fn with_parent(out: Out, parent: Option<&str>) -> Out {
    match (out, parent) {
        (Out::Update(mut update), Some(parent)) => {
            update["_meta"] = serde_json::json!({"claudeCode": {"parentToolUseId": parent}});
            Out::Update(update)
        }
        (out, _) => out,
    }
}

/// Anthropic's usage block: `input_tokens` excludes the cache.
fn anthropic_usage(value: &Value) -> Option<Usage> {
    let usage = Usage {
        input_tokens: count(value.get("input_tokens")),
        output_tokens: count(value.get("output_tokens")),
        total_tokens: None,
        cached_read_tokens: count(value.get("cache_read_input_tokens")),
        cached_write_tokens: count(value.get("cache_creation_input_tokens")),
        thought_tokens: None,
    };
    (usage != Usage::default()).then_some(usage)
}

fn sum<'a>(usages: impl Iterator<Item = &'a Usage>) -> Usage {
    let add = |a: Option<u64>, b: Option<u64>| match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    };
    usages.fold(Usage::default(), |acc, u| Usage {
        input_tokens: add(acc.input_tokens, u.input_tokens),
        output_tokens: add(acc.output_tokens, u.output_tokens),
        total_tokens: None,
        cached_read_tokens: add(acc.cached_read_tokens, u.cached_read_tokens),
        cached_write_tokens: add(acc.cached_write_tokens, u.cached_write_tokens),
        thought_tokens: None,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::headless::tests::{events_of, run_lines};
    use crate::trajectory::{Event, ToolCallStatus};

    fn line(v: Value) -> String {
        v.to_string()
    }

    /// The shapes of Harbor's Claude Code session fixtures
    /// (`test_claude_code_trajectory.py`), as stream-json lines.
    fn session() -> Vec<String> {
        vec![
            line(
                json!({"type": "system", "subtype": "init", "session_id": "s1", "model": "claude-sonnet-4-6", "tools": ["Bash"]}),
            ),
            line(
                json!({"type": "assistant", "session_id": "s1", "parent_tool_use_id": null,
                "message": {"id": "msg_1", "role": "assistant", "model": "claude-sonnet-4-6",
                    "content": [{"type": "thinking", "thinking": "Let me look at the files.", "signature": "sig"}],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}}}),
            ),
            line(
                json!({"type": "assistant", "session_id": "s1", "parent_tool_use_id": null,
                "message": {"id": "msg_1", "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls /app", "description": "List files"}}],
                    "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}}}),
            ),
            line(
                json!({"type": "user", "session_id": "s1", "parent_tool_use_id": null,
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "main.py\nREADME.md", "is_error": false}]},
                "tool_use_result": {"stdout": "main.py\nREADME.md", "stderr": "", "interrupted": false}}),
            ),
            line(
                json!({"type": "assistant", "session_id": "s1", "parent_tool_use_id": null,
                "message": {"id": "msg_2", "role": "assistant",
                    "content": [{"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"file_path": "/app/missing.py"}}],
                    "usage": {"input_tokens": 30, "output_tokens": 7}}}),
            ),
            line(
                json!({"type": "user", "session_id": "s1", "parent_tool_use_id": null,
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_2",
                    "content": [{"type": "text", "text": "File does not exist."}], "is_error": true}]}}),
            ),
            line(
                json!({"type": "assistant", "session_id": "s1", "parent_tool_use_id": null,
                "message": {"id": "msg_3", "role": "assistant", "content": [{"type": "text", "text": "Done: two files."}],
                    "usage": {"input_tokens": 40, "output_tokens": 9}}}),
            ),
            line(
                json!({"type": "result", "subtype": "success", "is_error": false, "result": "Done: two files.",
                "total_cost_usd": 0.0123, "num_turns": 3,
                "usage": {"input_tokens": 80, "output_tokens": 21, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}}),
            ),
        ]
    }

    #[tokio::test]
    async fn a_session_becomes_thoughts_tool_calls_and_messages() {
        let run = run_lines("claude-stream-json", &session()).await;
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
        let Event::ToolCall(bash) = &events[2] else {
            panic!()
        };
        assert_eq!(bash.tool_call_id, "toolu_1");
        assert_eq!(bash.kind, "execute");
        assert_eq!(bash.title, "ls /app");
        assert_eq!(bash.status, ToolCallStatus::Completed);
        assert_eq!(bash.tool_name.as_deref(), Some("Bash"));
        assert_eq!(bash.raw_input.as_ref().unwrap()["command"], "ls /app");
        assert_eq!(
            bash.raw_output.as_ref().unwrap()["stdout"],
            "main.py\nREADME.md"
        );
        let Event::ToolCall(read) = &events[3] else {
            panic!()
        };
        assert_eq!(read.status, ToolCallStatus::Failed);
        assert_eq!(read.kind, "read");
        // The result event's totals win over the per-message sums.
        let usage = run.usage.unwrap();
        assert_eq!(
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_read_tokens
            ),
            (Some(80), Some(21), Some(100))
        );
        assert_eq!(run.cost_usd, Some(0.0123));
        assert!(run.result.is_ok(), "{:?}", run.result);
    }

    #[tokio::test]
    async fn usage_is_summed_per_message_until_the_result_arrives() {
        let lines = &session()[..5];
        let run = run_lines("claude-stream-json", lines).await;
        let usage = run.usage.unwrap();
        // msg_1 counted once (two blocks), plus msg_2.
        assert_eq!(
            (
                usage.input_tokens,
                usage.output_tokens,
                usage.cached_read_tokens
            ),
            (Some(40), Some(12), Some(100))
        );
    }

    #[tokio::test]
    async fn a_subagents_calls_are_attributed_to_its_task() {
        let lines = vec![
            line(
                json!({"type": "assistant", "parent_tool_use_id": null, "message": {"id": "m1",
                "content": [{"type": "tool_use", "id": "task_1", "name": "Task", "input": {"description": "explore", "prompt": "look"}}]}}),
            ),
            line(
                json!({"type": "assistant", "parent_tool_use_id": "task_1", "message": {"id": "m2",
                "content": [{"type": "text", "text": "sub says hi"}, {"type": "tool_use", "id": "t2", "name": "Grep", "input": {"pattern": "fn main"}}]}}),
            ),
            line(
                json!({"type": "user", "parent_tool_use_id": "task_1", "message": {"content": [{"type": "tool_result", "tool_use_id": "t2", "content": "src/main.rs"}]}}),
            ),
            line(
                json!({"type": "user", "parent_tool_use_id": null, "message": {"content": [{"type": "tool_result", "tool_use_id": "task_1", "content": [{"type": "text", "text": "found it"}]}]}}),
            ),
        ];
        let run = run_lines("claude-stream-json", &lines).await;
        let events = events_of(&run);
        let grep = events
            .iter()
            .find_map(|e| match e {
                Event::ToolCall(c) if c.tool_call_id == "t2" => Some(c),
                _ => None,
            })
            .unwrap();
        assert_eq!(grep.parent_tool_call_id.as_deref(), Some("task_1"));
        assert_eq!(grep.title, "Grep fn main");
        let sub_text = events
            .iter()
            .find_map(|e| match e {
                Event::AgentMessage(t) => Some(t),
                _ => None,
            })
            .unwrap();
        assert_eq!(sub_text.parent_tool_call_id.as_deref(), Some("task_1"));
    }

    #[tokio::test]
    async fn a_spent_subscription_is_a_usage_limit() {
        let lines = vec![
            line(
                json!({"type": "assistant", "error": "rate_limit", "message": {"id": "m1",
                "content": [{"type": "text", "text": "You've hit your weekly limit · resets Oct 2, 4pm (UTC)"}]}}),
            ),
            line(
                json!({"type": "result", "subtype": "success", "is_error": true,
                "result": "You've hit your weekly limit · resets Oct 2, 4pm (UTC)"}),
            ),
        ];
        let run = crate::headless::tests::run_lines_exit("claude-stream-json", &lines, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "usage_limit", "{err}");
        assert!(err.to_string().contains("7-day window"), "{err}");
    }

    #[tokio::test]
    async fn a_rejected_login_is_a_login_error() {
        let lines = vec![
            line(
                json!({"type": "assistant", "error": "authentication_failed", "message": {"id": "m1",
                "content": [{"type": "text", "text": "Invalid API key · Please run /login"}]}}),
            ),
            line(
                json!({"type": "result", "subtype": "success", "is_error": true, "result": "Invalid API key · Please run /login"}),
            ),
        ];
        let run = crate::headless::tests::run_lines_exit("claude-stream-json", &lines, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
