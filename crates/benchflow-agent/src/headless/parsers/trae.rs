//! `trae-trajectory`: trae-cli, its `--trajectory-file` followed.
//!
//! trae-agent rewrites its trajectory after every model call and step:
//! `{"task", "provider", "model", "llm_interactions": [...], "agent_steps":
//! [...], "success", "final_result", "execution_time"}`. The command runs it
//! under `follow.py` with `agent_steps` as the array and each step's
//! `llm_messages` (the whole conversation so far) dropped (see
//! [`super::follow`]); `llm_interactions` repeats the same calls with their
//! full input and is not followed. Each step arrives as `{"@bf":
//! "agent_steps", "item": {"step_number", "state", "llm_response":
//! {"content", "usage", "tool_calls"}, "tool_calls": [{"call_id", "name",
//! "arguments"}], "tool_results": [{"call_id", "success", "result",
//! "error"}], "reflection", "error"}}`.
//!
//! A step whose `state` is `error` ends the run (trae-agent stops at the
//! first model or agent error), and trae-cli still exits 0, so it fails the
//! run with the step's `error`. `{"@bf": "info", ...}` closes the run with
//! `success` and `final_result`.

use std::collections::HashSet;

use serde_json::Value;

use super::follow::{arguments, finish_call, followed, start_call};
use crate::headless::parser::{Failure, Out, Parser, count, message, notice, one_line, str_of};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct Trae {
    usage: Usage,
    /// Calls started, so a step recorded again does not start them twice.
    started: HashSet<String>,
    /// Calls finished.
    finished: HashSet<String>,
}

impl Parser for Trae {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some((name, item)) = followed(line) else {
            return false;
        };
        match name.as_str() {
            "agent_steps" => self.step(&item, out),
            "info" if item.get("success").and_then(Value::as_bool) == Some(false) => {
                let result = str_of(&item, "final_result");
                if !result.trim().is_empty() {
                    out.push(notice("warning", &one_line(result)));
                }
            }
            _ => {}
        }
        true
    }
}

impl Trae {
    fn step(&mut self, step: &Value, out: &mut Vec<Out>) {
        let response = step.get("llm_response").unwrap_or(&Value::Null);
        let content = str_of(response, "content");
        if !content.trim().is_empty() {
            out.push(message(content));
        }
        if let Some(usage) = response.get("usage").filter(|u| u.is_object()) {
            // trae-agent's input_tokens leaves the cache out, as Anthropic's does.
            let add = |slot: &mut Option<u64>, key: &str| {
                if let Some(n) = count(usage.get(key)) {
                    *slot = Some(slot.unwrap_or(0) + n);
                }
            };
            let u = &mut self.usage;
            add(&mut u.input_tokens, "input_tokens");
            add(&mut u.output_tokens, "output_tokens");
            add(&mut u.cached_read_tokens, "cache_read_input_tokens");
            add(&mut u.cached_write_tokens, "cache_creation_input_tokens");
            add(&mut u.thought_tokens, "reasoning_tokens");
            out.push(Out::Usage(self.usage));
        }
        let calls = step
            .get("tool_calls")
            .and_then(Value::as_array)
            .or_else(|| response.get("tool_calls").and_then(Value::as_array));
        for call in calls.into_iter().flatten() {
            let id = [str_of(call, "call_id"), str_of(call, "id")]
                .into_iter()
                .find(|s| !s.is_empty())
                .unwrap_or("")
                .to_string();
            if id.is_empty() || !self.started.insert(id.clone()) {
                continue;
            }
            start_call(
                out,
                &id,
                str_of(call, "name"),
                arguments(call.get("arguments")),
            );
        }
        for result in step
            .get("tool_results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let id = [str_of(result, "call_id"), str_of(result, "id")]
                .into_iter()
                .find(|s| !s.is_empty())
                .unwrap_or("")
                .to_string();
            if id.is_empty() || !self.finished.insert(id.clone()) {
                continue;
            }
            let failed = result.get("success").and_then(Value::as_bool) == Some(false);
            let error = str_of(result, "error");
            let text = if !error.is_empty() {
                Some(error.to_string())
            } else {
                match result.get("result") {
                    Some(Value::String(s)) => Some(s.clone()),
                    Some(Value::Null) | None => None,
                    Some(other) => Some(other.to_string()),
                }
            };
            finish_call(out, &id, failed, text);
        }
        let error = str_of(step, "error");
        if str_of(step, "state") == "error" {
            let text = if error.trim().is_empty() {
                "trae-agent stopped on an error".to_string()
            } else {
                error.to_string()
            };
            out.push(Out::Failure(Failure::fatal(text)));
        } else if !error.trim().is_empty() {
            out.push(Out::Failure(Failure::notice(error)));
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines};
    use crate::trajectory::Event;

    fn lines(steps: &[serde_json::Value], info: serde_json::Value) -> Vec<String> {
        let mut lines = vec!["Step 1: thinking...".to_string()];
        lines.extend(
            steps
                .iter()
                .map(|s| json!({"@bf": "agent_steps", "item": s}).to_string()),
        );
        lines.push(json!({"@bf": "info", "item": info}).to_string());
        lines
    }

    #[tokio::test]
    async fn steps_become_messages_calls_results_and_usage() {
        // The trajectory shape Harbor's converter reads (test_trae_agent.py).
        let steps = [
            json!({"step_number": 1, "state": "completed",
                   "llm_response": {"content": "I'll create the file.", "model": "claude-sonnet-4-6",
                     "usage": {"input_tokens": 1000, "output_tokens": 50, "cache_read_input_tokens": 200,
                               "cache_creation_input_tokens": 0, "reasoning_tokens": null},
                     "tool_calls": [{"call_id": "toolu_1", "name": "bash", "arguments": {"command": "echo hi > hello.txt"}}]},
                   "tool_calls": [{"call_id": "toolu_1", "name": "bash", "arguments": {"command": "echo hi > hello.txt"}}],
                   "tool_results": [{"call_id": "toolu_1", "success": true, "result": "", "error": null}],
                   "reflection": null, "error": null}),
            json!({"step_number": 2, "state": "completed",
                   "llm_response": {"content": "", "usage": {"input_tokens": 1200, "output_tokens": 20},
                     "tool_calls": [{"call_id": "toolu_2", "name": "str_replace_based_edit_tool",
                                     "arguments": "{\"command\": \"view\", \"path\": \"/app/missing.txt\"}"}]},
                   "tool_results": [{"call_id": "toolu_2", "success": false, "result": null,
                                     "error": "The path /app/missing.txt does not exist."}],
                   "reflection": "The tool execution failed", "error": null}),
            json!({"step_number": 3, "state": "completed",
                   "llm_response": {"content": "Done.", "usage": {"input_tokens": 1300, "output_tokens": 10},
                     "tool_calls": [{"call_id": "toolu_3", "name": "task_done", "arguments": {}}]},
                   "tool_results": [{"call_id": "toolu_3", "success": true, "result": "Task done."}]}),
        ];
        let run = run_lines(
            "trae-trajectory",
            &lines(&steps, json!({"success": true, "final_result": "Done."})),
        )
        .await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let calls: Vec<String> = events_of(&run)
            .iter()
            .filter_map(|e| match e {
                Event::ToolCall(c) => Some(format!("{}|{}|{}", c.title, c.kind, c.status.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            [
                "echo hi > hello.txt|execute|completed",
                "str_replace_based_edit_tool view|edit|failed",
                "task_done|other|completed"
            ]
        );
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.output_tokens, u.cached_read_tokens),
            (Some(3500), Some(80), Some(200))
        );
    }

    #[tokio::test]
    async fn an_error_step_fails_the_run_although_trae_exits_0() {
        let steps = [
            json!({"step_number": 1, "state": "error", "llm_response": null,
            "error": "Error code: 401 - {'type': 'error', 'error': {'type': 'authentication_error', 'message': 'invalid x-api-key'}}"}),
        ];
        let run = run_lines(
            "trae-trajectory",
            &lines(&steps, json!({"success": false, "final_result": null})),
        )
        .await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    #[tokio::test]
    async fn a_step_seen_twice_starts_its_calls_once() {
        let step = json!({"step_number": 1, "state": "completed",
            "tool_calls": [{"call_id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
            "tool_results": [{"call_id": "c1", "success": true, "result": "a"}]});
        let run = run_lines(
            "trae-trajectory",
            &lines(&[step.clone(), step], json!({"success": true})),
        )
        .await;
        let calls = events_of(&run)
            .iter()
            .filter(|e| matches!(e, Event::ToolCall(_)))
            .count();
        assert_eq!(calls, 1);
    }
}
