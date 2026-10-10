//! `fx-json`: `fx ask --json`.
//!
//! fx documents no schema for `--json`, and Harbor parses none. This parser
//! reads the shapes the fx binary (0.0.13) carries, defensively, so a format
//! it does not know still ends up in the agent log rather than lost:
//!
//! - ACP-shaped updates (`{"sessionUpdate": ...}`, fx speaks ACP inside),
//!   alone or wrapped as `{"update": {...}}`, passed through;
//! - AI SDK stream parts: `text`/`text-delta` (`text` or `delta`),
//!   `reasoning`/`reasoning-delta`, `tool-call` (`toolCallId`, `toolName`,
//!   `input`), `tool-result` (`toolCallId`, `output`), `tool-error`,
//!   `finish` (`totalUsage`), `error`;
//! - the summary object `fx ask --json` prints at the end (`answer`,
//!   `final_output`, `text` or `result`; `usage` with `input_tokens` and
//!   `output_tokens`, where fx sums the main agent's tokens; `status`,
//!   `error`). In 0.0.13 it also lists the run's tool calls (`tool_calls`:
//!   `name`, `status` `success` or not, and for the shell `command_result`
//!   with `command`, `cwd`, `exit_code`, `duration_ms`), without their
//!   output;
//! - the session's detail, which the command prints after `fx ask` exits
//!   with `fx session <id> --json` (`kind: session_detail`; `history[]`
//!   with `assistant`, the final text, and `execution.tool_steps[]`, each
//!   with the `assistant` text before its calls, `tool_calls[]` (`id`,
//!   `name`, `arguments_json`) and `tool_results[]` (`tool_call_id`,
//!   `status`, `output`, for the shell a JSON string whose `output_delta`
//!   is the command's output)).
//!
//! A summary with `tool_calls` is held until stdout ends: the steps then
//! come from the session's detail when it followed, else from the summary's
//! list, and the answer comes after them.
//!
//! A whole document spread over several lines (pretty-printed JSON) is
//! collected and read at the end.

use serde_json::{Map, Value};

use crate::headless::parser::{
    Failure, Out, Parser, ToolDone, ToolStart, content_text, count, json_object, kind_for_tool,
    message, str_of, thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct FxJson {
    /// Lines of a multi-line JSON document, while one is open.
    pending: Vec<String>,
    /// Whether any message text was streamed (the summary's answer then
    /// repeats it).
    streamed_text: bool,
    /// Whether any tool call was streamed.
    streamed_tools: bool,
    /// fx 0.0.13's summary, held for the session detail that follows it.
    summary: Option<Value>,
    /// The session's detail (`fx session <id> --json`).
    detail: Option<Value>,
}

impl Parser for FxJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if !self.pending.is_empty() {
            self.pending.push(line.to_string());
            let joined = self.pending.join("\n");
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(&joined) {
                self.pending.clear();
                self.event(&map, out);
            }
            return true;
        }
        if let Some(map) = json_object(line) {
            return self.event(&map, out);
        }
        // The start of a pretty-printed document.
        if line.trim() == "{" {
            self.pending.push(line.to_string());
            return true;
        }
        false
    }

    fn finish(&mut self, out: &mut Vec<Out>) {
        if !self.pending.is_empty() {
            let text = std::mem::take(&mut self.pending).join("\n");
            out.push(Out::Failure(Failure::notice(format!(
                "fx's JSON output ended in the middle of a document: {}",
                text.chars().take(500).collect::<String>()
            ))));
        }
        let Some(summary) = self.summary.take() else {
            return;
        };
        let detail = self.detail.take();
        let answered = match &detail {
            Some(detail) => detail_steps(detail, out),
            None => {
                summary_calls(&summary, out);
                false
            }
        };
        if answered {
            self.streamed_text = true;
        }
        self.summary_out(&summary, out);
    }
}

impl FxJson {
    fn event(&mut self, map: &Map<String, Value>, out: &mut Vec<Out>) -> bool {
        let event = Value::Object(map.clone());
        if event.get("sessionUpdate").is_some() {
            match str_of(&event, "sessionUpdate") {
                "agent_message_chunk" => self.streamed_text = true,
                "tool_call" => self.streamed_tools = true,
                _ => {}
            }
            out.push(Out::Update(event));
            return true;
        }
        if let Some(update) = event
            .get("update")
            .filter(|u| u.get("sessionUpdate").is_some())
        {
            out.push(Out::Update(update.clone()));
            return true;
        }
        match str_of(&event, "type") {
            "text" | "text-delta" => {
                let text = event
                    .get("text")
                    .or_else(|| event.get("delta"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !text.is_empty() {
                    self.streamed_text = true;
                    out.push(message(text));
                }
                true
            }
            "reasoning" | "reasoning-delta" => {
                let text = event
                    .get("text")
                    .or_else(|| event.get("delta"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                if !text.is_empty() {
                    out.push(thought(text));
                }
                true
            }
            "tool-call" => {
                self.streamed_tools = true;
                let name = str_of(&event, "toolName");
                let input = event.get("input").or_else(|| event.get("args"));
                out.push(
                    ToolStart {
                        id: str_of(&event, "toolCallId"),
                        title: tool_title(name, input),
                        kind: kind_for_tool(name),
                        status: "in_progress",
                        input: input.cloned(),
                        tool_name: Some(name),
                        parent: None,
                    }
                    .out(),
                );
                true
            }
            "tool-result" | "tool-error" => {
                let failed = str_of(&event, "type") == "tool-error"
                    || event.get("isError").and_then(Value::as_bool) == Some(true);
                let output = event
                    .get("output")
                    .or_else(|| event.get("result"))
                    .or_else(|| event.get("error"));
                let text = output.and_then(|o| {
                    content_text(o.get("value").or(Some(o))).or_else(|| content_text(Some(o)))
                });
                out.push(
                    ToolDone {
                        id: str_of(&event, "toolCallId"),
                        status: if failed { "failed" } else { "completed" },
                        text,
                        output: output.cloned(),
                        parent: None,
                    }
                    .out(),
                );
                true
            }
            "finish" | "finish-step" | "step-finish" => {
                if let Some(usage) = event
                    .get("totalUsage")
                    .or_else(|| event.get("usage"))
                    .and_then(usage_of)
                {
                    out.push(Out::Usage(usage));
                }
                true
            }
            "error" => {
                let text = match event.get("error") {
                    Some(Value::String(s)) => s.clone(),
                    Some(other) => [str_of(other, "message"), str_of(&event, "message")]
                        .into_iter()
                        .find(|t| !t.is_empty())
                        .map_or_else(|| other.to_string(), str::to_string),
                    None => str_of(&event, "message").to_string(),
                };
                out.push(Out::Failure(Failure::fatal(text)));
                true
            }
            "start" | "start-step" | "text-start" | "text-end" | "reasoning-start"
            | "reasoning-end" | "tool-input-start" | "tool-input-delta" | "tool-input-end" => true,
            _ => self.summary(&event, out),
        }
    }

    /// The object `fx ask --json` ends with: the answer, usage and status;
    /// or the session's detail printed after it.
    fn summary(&mut self, event: &Value, out: &mut Vec<Out>) -> bool {
        if str_of(event, "kind") == "session_detail" {
            if self.summary.is_some() {
                self.detail = Some(event.clone());
            }
            return true;
        }
        if event.get("tool_calls").is_some_and(Value::is_array) && !self.streamed_tools {
            if !Self::is_summary(event) {
                return false;
            }
            self.summary = Some(event.clone());
            return true;
        }
        if !Self::is_summary(event) {
            return false;
        }
        self.summary_out(event, out);
        true
    }

    fn is_summary(event: &Value) -> bool {
        ["answer", "final_output", "text", "result", "output"]
            .iter()
            .any(|k| event.get(*k).is_some_and(Value::is_string))
            || event.get("usage").and_then(usage_of).is_some()
            || matches!(event.get("error"), Some(Value::String(s)) if !s.is_empty())
            || event.get("error").is_some_and(Value::is_object)
            || matches!(str_of(event, "status"), "failed" | "error")
    }

    /// The summary's answer (unless already recorded), usage, cost and
    /// error.
    fn summary_out(&mut self, event: &Value, out: &mut Vec<Out>) {
        let answer = ["answer", "final_output", "text", "result", "output"]
            .iter()
            .find_map(|k| event.get(*k).and_then(Value::as_str));
        let usage = event.get("usage").and_then(usage_of);
        let error = match event.get("error") {
            Some(Value::String(s)) if !s.is_empty() => Some(s.clone()),
            Some(Value::Object(e)) => e
                .get("message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| Some(Value::Object(e.clone()).to_string())),
            _ => None,
        };
        let failed_status = matches!(str_of(event, "status"), "failed" | "error");
        if let Some(answer) = answer
            && !answer.is_empty()
            && !self.streamed_text
        {
            out.push(message(answer));
        }
        if let Some(usage) = usage {
            out.push(Out::Usage(usage));
        }
        if let Some(cost) = ["cost_usd", "costUsd", "cost"]
            .iter()
            .find_map(|k| event.get(*k).and_then(Value::as_f64))
        {
            out.push(Out::Cost(cost));
        }
        if error.is_some() || failed_status {
            out.push(Out::Failure(Failure::fatal(error.unwrap_or_else(|| {
                format!("fx ended with status {}", str_of(event, "status"))
            }))));
        }
    }
}

/// The steps of a session's detail: each step's text, its tool calls and
/// their results, then the final answer. Whether it gave an answer.
fn detail_steps(detail: &Value, out: &mut Vec<Out>) -> bool {
    let mut answered = false;
    let history = detail.get("history").and_then(Value::as_array);
    for item in history.into_iter().flatten() {
        let steps = item
            .pointer("/execution/tool_steps")
            .and_then(Value::as_array);
        for step in steps.into_iter().flatten() {
            let text = str_of(step, "assistant");
            if !text.trim().is_empty() {
                out.push(message(text));
            }
            for call in step
                .get("tool_calls")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let name = str_of(call, "name");
                let input = call
                    .get("arguments_json")
                    .and_then(Value::as_str)
                    .and_then(|a| serde_json::from_str::<Value>(a).ok())
                    .or_else(|| call.get("arguments").cloned());
                out.push(
                    ToolStart {
                        id: str_of(call, "id"),
                        title: tool_title(name, input.as_ref()),
                        kind: kind_for_tool(name),
                        status: "in_progress",
                        input,
                        tool_name: Some(name),
                        parent: None,
                    }
                    .out(),
                );
            }
            for result in step
                .get("tool_results")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                // The shell's output is a JSON string; the command's own
                // output is its `output_delta`.
                let output = match result.get("output") {
                    Some(Value::String(s)) => {
                        serde_json::from_str::<Value>(s).unwrap_or(Value::String(s.clone()))
                    }
                    Some(other) => other.clone(),
                    None => Value::Null,
                };
                let text = match &output {
                    Value::String(s) => Some(s.clone()),
                    other => content_text(other.get("output_delta"))
                        .or_else(|| content_text(other.get("error")))
                        .or_else(|| content_text(Some(other))),
                };
                out.push(
                    ToolDone {
                        id: str_of(result, "tool_call_id"),
                        status: done_status(str_of(result, "status")),
                        text,
                        output: (!output.is_null()).then_some(output),
                        parent: None,
                    }
                    .out(),
                );
            }
        }
        let answer = str_of(item, "assistant");
        if !answer.trim().is_empty() {
            out.push(message(answer));
            answered = true;
        }
    }
    answered
}

/// The tool calls a 0.0.13 summary lists, without their output.
fn summary_calls(summary: &Value, out: &mut Vec<Out>) {
    let calls = summary.get("tool_calls").and_then(Value::as_array);
    for (i, call) in calls.into_iter().flatten().enumerate() {
        let id = format!("fx-tool-{}", i + 1);
        let name = str_of(call, "name");
        let result = call.get("command_result").filter(|r| r.is_object());
        let input = result
            .and_then(|r| r.get("command"))
            .map(|c| serde_json::json!({ "command": c }));
        out.push(
            ToolStart {
                id: &id,
                title: tool_title(name, input.as_ref()),
                kind: kind_for_tool(name),
                status: "in_progress",
                input,
                tool_name: Some(name),
                parent: None,
            }
            .out(),
        );
        let text = result
            .and_then(|r| r.get("exit_code"))
            .and_then(Value::as_i64)
            .map(|code| format!("exit code {code}"));
        out.push(
            ToolDone {
                id: &id,
                status: done_status(str_of(call, "status")),
                text,
                output: result.cloned(),
                parent: None,
            }
            .out(),
        );
    }
}

fn done_status(status: &str) -> &'static str {
    match status {
        "success" | "succeeded" | "completed" | "ok" => "completed",
        _ => "failed",
    }
}

/// `{input_tokens, output_tokens}` or the AI SDK's `{inputTokens, outputTokens}`.
fn usage_of(value: &Value) -> Option<Usage> {
    let field = |snake: &str, camel: &str| count(value.get(snake).or_else(|| value.get(camel)));
    let usage = Usage {
        input_tokens: field("input_tokens", "inputTokens"),
        output_tokens: field("output_tokens", "outputTokens"),
        total_tokens: field("total_tokens", "totalTokens"),
        cached_read_tokens: field("cached_input_tokens", "cachedInputTokens"),
        cached_write_tokens: None,
        thought_tokens: field("reasoning_tokens", "reasoningTokens"),
    };
    (usage != Usage::default()).then_some(usage)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines};
    use crate::trajectory::{Event, ToolCallStatus};

    #[tokio::test]
    async fn stream_parts_and_the_summary() {
        let lines: Vec<String> = [
            json!({"type": "start"}),
            json!({"type": "reasoning-delta", "delta": "plan it"}),
            json!({"type": "tool-call", "toolCallId": "c1", "toolName": "bash", "input": {"command": "make test"}}),
            json!({"type": "tool-result", "toolCallId": "c1", "output": {"type": "text", "value": "ok"}}),
            json!({"type": "text-delta", "delta": "All "}),
            json!({"type": "text-delta", "delta": "green."}),
            json!({"schema_version": 1, "session_id": "s", "answer": "All green.", "usage": {"input_tokens": 12, "output_tokens": 3}}),
        ]
        .iter()
        .map(|v| v.to_string())
        .collect();
        let run = run_lines("fx-json", &lines).await;
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
        let Event::ToolCall(call) = &events[2] else {
            panic!()
        };
        assert_eq!(
            (call.title.as_str(), call.status),
            ("make test", ToolCallStatus::Completed)
        );
        // The summary's answer repeats the streamed text; it is not added twice.
        assert!(matches!(&events[3], Event::AgentMessage(t) if t.text == "All green."));
        assert_eq!(run.usage.unwrap().output_tokens, Some(3));
    }

    #[tokio::test]
    async fn a_pretty_printed_summary_and_acp_updates() {
        let lines: Vec<String> = vec![
            json!({"sessionUpdate": "tool_call", "toolCallId": "t", "title": "read a.txt", "kind": "read", "status": "completed"}).to_string(),
            "{".into(),
            "  \"answer\": \"Done.\",".into(),
            "  \"usage\": {\"input_tokens\": 5, \"output_tokens\": null}".into(),
            "}".into(),
        ];
        let run = run_lines("fx-json", &lines).await;
        let events = events_of(&run);
        assert!(matches!(&events[1], Event::ToolCall(c) if c.title == "read a.txt"));
        assert!(matches!(&events[2], Event::AgentMessage(t) if t.text == "Done."));
        assert_eq!(run.usage.unwrap().input_tokens, Some(5));
    }

    #[tokio::test]
    async fn an_error_summary_fails_the_run() {
        let run = run_lines(
            "fx-json",
            &[
                json!({"status": "failed", "error": {"message": "AI Gateway: Invalid API key"}})
                    .to_string(),
            ],
        )
        .await;
        assert_eq!(run.result.unwrap_err().category(), "provider_auth");
    }

    /// `fx ask --json` and `fx session <id> --json` on a T7.1 `hello` run
    /// (fx 0.0.13, deepseek-v4-flash through the gateway).
    const ASK: &str = include_str!("testdata/fx-0.0.13-ask.json");
    const SESSION: &str = include_str!("testdata/fx-0.0.13-session.json");

    /// Guards T7.1's "fx records no tool calls": the summary's `tool_calls`
    /// and the session's detail were not read.
    #[tokio::test]
    async fn the_session_detail_gives_the_steps() {
        let lines = vec![
            "Full access enabled: fx permission checks disabled".to_string(),
            ASK.trim().to_string(),
            SESSION.trim().to_string(),
        ];
        let run = run_lines("fx-json", &lines).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_message",
                "tool_call",
                "agent_message"
            ]
        );
        let Event::ToolCall(call) = &events[2] else {
            panic!()
        };
        assert_eq!(call.status, ToolCallStatus::Completed);
        assert_eq!(call.kind, "execute");
        assert!(
            call.title.starts_with("printf '%s' 'Hello, world!'"),
            "{}",
            call.title
        );
        let output = serde_json::to_string(&call.content).unwrap();
        assert!(output.contains("13 /app/hello.txt"), "{output}");
        assert!(
            matches!(&events[3], Event::AgentMessage(t) if t.text.starts_with("Created `/app/hello.txt`"))
        );
        assert_eq!(run.usage.unwrap().input_tokens, Some(17354));
    }

    #[tokio::test]
    async fn without_the_detail_the_summary_lists_the_calls() {
        let run = run_lines("fx-json", &[ASK.trim().to_string()]).await;
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, ["user_message", "tool_call", "agent_message"]);
        let Event::ToolCall(call) = &events[1] else {
            panic!()
        };
        assert_eq!(call.status, ToolCallStatus::Completed);
        assert!(
            call.title.contains("od -c /app/hello.txt"),
            "{}",
            call.title
        );
        // A failed call is recorded as failed.
        let failed = json!({"final_output": "no", "tool_calls": [{"name": "shell", "status": "error",
            "command_result": {"command": "false", "exit_code": 1}}]});
        let run = run_lines("fx-json", &[failed.to_string()]).await;
        let Some(Event::ToolCall(call)) = events_of(&run).iter().find(|e| e.kind() == "tool_call")
        else {
            panic!()
        };
        assert_eq!(call.status, ToolCallStatus::Failed);
    }
}
