//! `codex-json`: `codex exec --json`.
//!
//! One JSON event per line (codex-rs `exec_events.rs`):
//!
//! - `thread.started`, `turn.started`: bookkeeping;
//! - `item.started`, `item.updated`, `item.completed`, each with an `item`
//!   whose `type` is `agent_message` (`text`), `reasoning` (`text`),
//!   `command_execution` (`command`, `aggregated_output`, `exit_code`,
//!   `status`), `file_change` (`changes`, `status`), `mcp_tool_call`
//!   (`server`, `tool`, `arguments`, `result`, `error`, `status`),
//!   `web_search` (`query`), `todo_list` (`items`), or `error` (`message`);
//! - `turn.completed` with the turn's `usage` (`input_tokens`, which
//!   includes `cached_input_tokens`, and `output_tokens`);
//! - `turn.failed` with `error.message`: the run failed;
//! - `error` with `message`: a stream error (Codex also reports its
//!   reconnect attempts this way, so it is not fatal by itself).

use std::collections::HashSet;

use serde_json::{Value, json};

use crate::headless::parser::{
    Failure, Out, Parser, ToolDone, ToolStart, count, json_object, message, notice, one_line,
    str_of, thought,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct CodexJson {
    /// Tool-like items already opened, by id.
    open: HashSet<String>,
    /// Totals over the turns so far.
    usage: Usage,
}

impl Parser for CodexJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(event) = json_object(line) else {
            return false;
        };
        let event = Value::Object(event);
        match str_of(&event, "type") {
            "thread.started" | "turn.started" => {}
            "item.started" | "item.updated" | "item.completed" => {
                let done = str_of(&event, "type") == "item.completed";
                if let Some(item) = event.get("item") {
                    self.item(item, done, out);
                }
            }
            "turn.completed" => {
                if let Some(usage) = event.get("usage") {
                    self.add_usage(usage);
                    out.push(Out::Usage(self.usage));
                }
            }
            "turn.failed" => {
                let msg = event
                    .get("error")
                    .map(|e| str_of(e, "message").to_string())
                    .filter(|m| !m.is_empty())
                    .unwrap_or_else(|| "Codex reported turn.failed".to_string());
                out.push(Out::Failure(Failure::fatal(msg)));
            }
            "error" => {
                let msg = str_of(&event, "message");
                if !msg.is_empty() {
                    out.push(Out::Failure(Failure::notice(msg)));
                    out.push(notice("warning", &one_line(msg)));
                }
            }
            _ => return false,
        }
        true
    }
}

impl CodexJson {
    fn add_usage(&mut self, usage: &Value) {
        let input = count(usage.get("input_tokens")).unwrap_or(0);
        let cached = count(usage.get("cached_input_tokens")).unwrap_or(0);
        let output = count(usage.get("output_tokens")).unwrap_or(0);
        let reasoning = count(usage.get("reasoning_output_tokens"));
        let u = &mut self.usage;
        u.input_tokens = Some(u.input_tokens.unwrap_or(0) + input.saturating_sub(cached));
        u.cached_read_tokens = Some(u.cached_read_tokens.unwrap_or(0) + cached);
        u.output_tokens = Some(u.output_tokens.unwrap_or(0) + output);
        if let Some(r) = reasoning {
            u.thought_tokens = Some(u.thought_tokens.unwrap_or(0) + r);
        }
    }

    fn item(&mut self, item: &Value, done: bool, out: &mut Vec<Out>) {
        let id = str_of(item, "id");
        let status = str_of(item, "status");
        match str_of(item, "type") {
            // Text items arrive whole when they complete.
            "agent_message" => {
                if done && !str_of(item, "text").is_empty() {
                    out.push(message(str_of(item, "text")));
                }
            }
            "reasoning" => {
                if done && !str_of(item, "text").is_empty() {
                    out.push(thought(str_of(item, "text")));
                }
            }
            "command_execution" => {
                let command = str_of(item, "command");
                let input = json!({"command": command});
                let exit = item.get("exit_code").and_then(Value::as_i64);
                let failed = status == "failed"
                    || status == "declined"
                    || exit.is_some_and(|code| code != 0);
                self.tool(
                    out,
                    id,
                    one_line(command),
                    "execute",
                    Some(input),
                    done,
                    failed,
                    item.get("aggregated_output")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    Some(json!({
                        "exit_code": exit,
                        "aggregated_output": str_of(item, "aggregated_output"),
                    })),
                );
            }
            "file_change" => {
                let paths: Vec<String> = item
                    .get("changes")
                    .and_then(Value::as_array)
                    .map(|changes| {
                        changes
                            .iter()
                            .map(|c| format!("{} {}", str_of(c, "kind"), str_of(c, "path")))
                            .collect()
                    })
                    .unwrap_or_default();
                let title = if paths.is_empty() {
                    "edit files".to_string()
                } else {
                    format!("edit: {}", paths.join(", "))
                };
                self.tool(
                    out,
                    id,
                    one_line(&title),
                    "edit",
                    item.get("changes").map(|c| json!({"changes": c})),
                    // A file change is reported once, already applied.
                    done || status == "completed" || status == "failed",
                    status == "failed",
                    None,
                    None,
                );
            }
            "mcp_tool_call" => {
                let title = format!("{}.{}", str_of(item, "server"), str_of(item, "tool"));
                let error = item.get("error").filter(|e| !e.is_null()).map(|e| {
                    match e.get("message").and_then(Value::as_str) {
                        Some(m) => m.to_string(),
                        None => e.to_string(),
                    }
                });
                let result = item.get("result").filter(|r| !r.is_null());
                let text = error.clone().or_else(|| {
                    result.and_then(|r| crate::headless::parser::content_text(r.get("content")))
                });
                self.tool(
                    out,
                    id,
                    title,
                    "other",
                    item.get("arguments").cloned(),
                    done,
                    status == "failed" || error.is_some(),
                    text,
                    result.cloned(),
                );
            }
            "web_search" => {
                let query = str_of(item, "query");
                self.tool(
                    out,
                    id,
                    format!("web search {}", one_line(query)),
                    "fetch",
                    Some(json!({"query": query})),
                    done,
                    false,
                    None,
                    item.get("action").cloned(),
                );
            }
            "todo_list" => {
                let entries: Vec<Value> = item
                    .get("items")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .map(|i| {
                                let finished = i.get("completed").and_then(Value::as_bool);
                                json!({
                                    "content": str_of(i, "text"),
                                    "status": if finished == Some(true) { "completed" } else { "pending" },
                                    "priority": "medium",
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                out.push(Out::Update(
                    json!({"sessionUpdate": "plan", "entries": entries}),
                ));
            }
            "error" => {
                let msg = str_of(item, "message");
                if !msg.is_empty() {
                    out.push(notice("error", &one_line(msg)));
                    out.push(Out::Failure(Failure::notice(msg)));
                }
            }
            other => {
                // An item type this build does not know: a generic tool call,
                // so the work still shows.
                let title = if other.is_empty() { "item" } else { other };
                self.tool(
                    out,
                    id,
                    title.to_string(),
                    "other",
                    Some(item.clone()),
                    done,
                    status == "failed",
                    None,
                    None,
                );
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn tool(
        &mut self,
        out: &mut Vec<Out>,
        id: &str,
        title: String,
        kind: &str,
        input: Option<Value>,
        done: bool,
        failed: bool,
        text: Option<String>,
        output: Option<Value>,
    ) {
        if self.open.insert(id.to_string()) {
            out.push(
                ToolStart {
                    id,
                    title,
                    kind,
                    status: "in_progress",
                    input,
                    ..ToolStart::default()
                }
                .out(),
            );
        }
        if done {
            out.push(
                ToolDone {
                    id,
                    status: if failed { "failed" } else { "completed" },
                    text,
                    output,
                    parent: None,
                }
                .out(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::{Event, ToolCallStatus};

    fn lines(events: &[serde_json::Value]) -> Vec<String> {
        events.iter().map(|e| e.to_string()).collect()
    }

    #[tokio::test]
    async fn a_turn_becomes_commands_edits_and_a_message() {
        let run = run_lines(
            "codex-json",
            &lines(&[
                json!({"type": "thread.started", "thread_id": "0199"}),
                json!({"type": "turn.started"}),
                json!({"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "**Inspecting the repo**"}}),
                json!({"type": "item.started", "item": {"id": "item_1", "type": "command_execution", "command": "bash -lc ls", "aggregated_output": "", "exit_code": null, "status": "in_progress"}}),
                json!({"type": "item.completed", "item": {"id": "item_1", "type": "command_execution", "command": "bash -lc ls", "aggregated_output": "hello.txt\n", "exit_code": 0, "status": "completed"}}),
                json!({"type": "item.completed", "item": {"id": "item_2", "type": "file_change", "changes": [{"path": "/app/hello.txt", "kind": "update"}], "status": "completed"}}),
                json!({"type": "item.started", "item": {"id": "item_3", "type": "command_execution", "command": "pytest", "aggregated_output": "", "exit_code": null, "status": "in_progress"}}),
                json!({"type": "item.completed", "item": {"id": "item_3", "type": "command_execution", "command": "pytest", "aggregated_output": "1 failed", "exit_code": 1, "status": "failed"}}),
                json!({"type": "item.completed", "item": {"id": "item_4", "type": "todo_list", "items": [{"text": "fix", "completed": true}]}}),
                json!({"type": "item.completed", "item": {"id": "item_5", "type": "agent_message", "text": "Updated hello.txt."}}),
                json!({"type": "turn.completed", "usage": {"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 80}}),
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
                "tool_call",
                "tool_call",
                "tool_call",
                "agent_message"
            ]
        );
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls[0].title, "bash -lc ls");
        assert_eq!(calls[0].kind, "execute");
        assert_eq!(calls[0].status, ToolCallStatus::Completed);
        assert_eq!(calls[1].kind, "edit");
        assert_eq!(calls[1].title, "edit: update /app/hello.txt");
        assert_eq!(calls[2].status, ToolCallStatus::Failed);
        let usage = run.usage.unwrap();
        assert_eq!(
            (
                usage.input_tokens,
                usage.cached_read_tokens,
                usage.output_tokens
            ),
            (Some(200), Some(1000), Some(80))
        );
    }

    #[tokio::test]
    async fn reconnect_errors_alone_do_not_fail_the_run() {
        let run = run_lines(
            "codex-json",
            &lines(&[
                json!({"type": "error", "message": "Reconnecting... 1/5 (stream disconnected before completion)"}),
                json!({"type": "item.completed", "item": {"id": "i", "type": "agent_message", "text": "ok"}}),
                json!({"type": "turn.completed", "usage": {"input_tokens": 5, "cached_input_tokens": 0, "output_tokens": 2}}),
            ]),
        )
        .await;
        assert!(run.result.is_ok(), "{:?}", run.result);
    }

    #[tokio::test]
    async fn turn_failed_is_classified() {
        // Harbor's test_error_patterns cases.
        let cases = [
            (
                "unexpected status 401 Unauthorized: Missing bearer",
                "provider_auth",
            ),
            ("unexpected status 503 Service Unavailable", "infra_failure"),
            (
                "You\u{2019}ve hit your usage limit. Upgrade to Plus to continue using Codex, or try again at Oct 3rd, 2026 5:05 PM.",
                "usage_limit",
            ),
            ("Something odd", "agent_error"),
        ];
        for (message, category) in cases {
            let run = run_lines_exit(
                "codex-json",
                &lines(&[json!({"type": "turn.failed", "error": {"message": message}})]),
                1,
            )
            .await;
            let err = run.result.unwrap_err();
            assert_eq!(err.category(), category, "{message}: {err}");
        }
    }

    #[tokio::test]
    async fn unknown_items_still_show_as_tool_calls() {
        let run = run_lines(
            "codex-json",
            &lines(&[json!({"type": "item.completed", "item": {"id": "x", "type": "collab_tool_call", "status": "completed"}})]),
        )
        .await;
        let events = events_of(&run);
        assert!(matches!(&events[1], Event::ToolCall(c) if c.title == "collab_tool_call"));
    }
}
