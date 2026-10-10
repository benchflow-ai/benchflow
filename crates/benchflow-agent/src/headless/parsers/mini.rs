//! `mini-swe-agent-trajectory`: mini-swe-agent, its `--output` file followed.
//!
//! mini-swe-agent (v2) rewrites its trajectory after every step:
//! `{"info": {...}, "messages": [...], "trajectory_format":
//! "mini-swe-agent-1.1"}`. The command runs it under `follow.py` with
//! `messages` as the array (see [`super::follow`]), so each message arrives
//! as `{"@bf": "messages", "item": <message>}`:
//!
//! - chat-format messages (`system`, `user`, `assistant` with `tool_calls`,
//!   `tool` with `tool_call_id`), usage under `extra.response.usage`, the
//!   query's cost under `extra.cost`; text-mode assistants carry their
//!   commands in `extra.actions`, and their observations come back as
//!   `user` messages;
//! - Responses API turns (`object: "response"`, `output` items, results as
//!   `function_call_output`), usage at the top level;
//! - `exit`, the last message: `extra.exit_status` is `Submitted`,
//!   `LimitsExceeded`, `TimeExceeded`, `RepeatedFormatError`, or, when an
//!   exception ended the run, the exception's class name, with
//!   `extra.exception_str` (mini-swe-agent then exits non-zero).
//!
//! `{"@bf": "info", ...}` closes the run with `info.model_stats.instance_cost`.
//! Everything else on stdout is mini-swe-agent's console, kept for the log
//! (its error lines explain a failure).

use serde_json::Value;

use super::follow::{ChatMessages, followed};
use crate::headless::parser::{Failure, Out, Parser, notice, str_of};

/// How a run ends without an error.
const CLEAN_EXITS: [&str; 5] = [
    "",
    "Submitted",
    "LimitsExceeded",
    "TimeExceeded",
    "RepeatedFormatError",
];

#[derive(Default)]
pub struct MiniSweAgent {
    chat: ChatMessages,
    cost: f64,
}

impl Parser for MiniSweAgent {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some((name, item)) = followed(line) else {
            return false;
        };
        match name.as_str() {
            "messages" => self.message(&item, out),
            "info" => {
                let info = item.get("info").unwrap_or(&item);
                if let Some(cost) = info
                    .get("model_stats")
                    .and_then(|s| s.get("instance_cost"))
                    .and_then(Value::as_f64)
                    .filter(|c| *c > 0.0)
                {
                    out.push(Out::Cost(cost));
                }
            }
            _ => {}
        }
        true
    }
}

impl MiniSweAgent {
    fn message(&mut self, msg: &Value, out: &mut Vec<Out>) {
        let extra = msg.get("extra");
        if str_of(msg, "role") == "exit" {
            let extra = extra.unwrap_or(&Value::Null);
            let status = str_of(extra, "exit_status");
            if CLEAN_EXITS.contains(&status) {
                if !status.is_empty() && status != "Submitted" {
                    out.push(notice(
                        "warning",
                        &format!("mini-swe-agent stopped: {status}"),
                    ));
                }
            } else {
                let detail = [str_of(extra, "exception_str"), str_of(msg, "content")]
                    .into_iter()
                    .find(|s| !s.trim().is_empty())
                    .unwrap_or("");
                let text = if detail.is_empty() {
                    status.to_string()
                } else {
                    format!("{status}: {detail}")
                };
                out.push(Out::Failure(Failure::fatal(text)));
            }
            return;
        }
        let usage = extra
            .and_then(|e| e.get("response"))
            .and_then(|r| r.get("usage"))
            .or_else(|| {
                (msg.get("object").and_then(Value::as_str) == Some("response"))
                    .then(|| msg.get("usage"))
                    .flatten()
            });
        self.chat.message(msg, usage, out);
        if let Some(cost) = extra
            .and_then(|e| e.get("cost"))
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

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn records(messages: &[Value], info: Option<Value>) -> Vec<String> {
        let mut lines = vec!["mini-swe-agent (step 1, $0.00):".to_string()];
        for m in messages {
            lines.push(json!({"@bf": "messages", "item": m}).to_string());
        }
        if let Some(info) = info {
            lines.push(json!({"@bf": "info", "item": {"info": info}}).to_string());
        }
        lines
    }

    fn usage(prompt: u64, completion: u64, cached: u64) -> Value {
        json!({"response": {"usage": {
            "prompt_tokens": prompt, "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": 0}}}})
    }

    /// Harbor's V2_TOOL_CALLING_TRAJECTORY (test_mini_swe_agent.py).
    fn tool_calling() -> Vec<Value> {
        let call = |id: &str, command: &str| {
            json!([{"id": id, "type": "function",
                    "function": {"name": "bash", "arguments": json!({"command": command}).to_string()}}])
        };
        vec![
            json!({"role": "system", "content": "You are a helpful assistant.", "extra": {}}),
            json!({"role": "user", "content": "Fix the import error in baz.py", "extra": {}}),
            json!({"role": "assistant", "content": "Let me look at the file to understand the import error.",
                   "tool_calls": call("call_abc123", "cat baz.py"), "extra": usage(600, 120, 100)}),
            json!({"role": "tool", "content": "import os\nimport sys\nfrom collections import OrderedDcit\n",
                   "tool_call_id": "call_abc123", "extra": {}}),
            json!({"role": "assistant", "content": "I see a typo in the import: OrderedDcit should be OrderedDict.",
                   "tool_calls": call("call_def456", "sed -i 's/OrderedDcit/OrderedDict/' baz.py"),
                   "extra": usage(900, 80, 300)}),
            json!({"role": "tool", "content": "[File edited successfully]", "tool_call_id": "call_def456", "extra": {}}),
            json!({"role": "assistant", "content": "Let me verify the fix works.",
                   "tool_calls": call("call_ghi789", "python -c \"import baz\""), "extra": usage(1100, 50, 500)}),
            json!({"role": "tool", "content": "", "tool_call_id": "call_ghi789", "extra": {}}),
            json!({"role": "exit", "content": "", "extra": {"exit_status": "Submitted", "submission": ""}}),
        ]
    }

    #[tokio::test]
    async fn tool_calling_trajectory_becomes_calls_results_and_usage() {
        let run = run_lines(
            "mini-swe-agent-trajectory",
            &records(
                &tool_calling(),
                Some(json!({"model_stats": {"instance_cost": 0.25}, "exit_status": "Submitted"})),
            ),
        )
        .await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        // The system prompt and the task are not repeated: BenchFlow
        // recorded the prompt already.
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_message",
                "tool_call",
                "agent_message",
                "tool_call",
                "agent_message",
                "tool_call"
            ],
            "{events:?}"
        );
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls[0].title, "cat baz.py");
        assert_eq!(calls[0].kind, "execute");
        assert_eq!(calls[0].status.as_str(), "completed");
        assert!(format!("{:?}", calls[0]).contains("OrderedDcit"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(1700), Some(900), Some(250))
        );
        assert_eq!(run.cost_usd, Some(0.25));
        assert_eq!(run.log, ["mini-swe-agent (step 1, $0.00):"]);
    }

    #[tokio::test]
    async fn parallel_calls_get_their_own_results() {
        // Harbor's V2_TOOL_CALLING_MULTI_TOOL.
        let messages = vec![
            json!({"role": "system", "content": "System prompt."}),
            json!({"role": "user", "content": "Do something."}),
            json!({"role": "assistant", "content": "I'll run two commands.", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}},
                {"id": "call_2", "type": "function", "function": {"name": "bash", "arguments": {"command": "pwd"}}}]}),
            json!({"role": "tool", "content": "/testbed", "tool_call_id": "call_2"}),
            json!({"role": "tool", "content": "file1.py\nfile2.py", "tool_call_id": "call_1"}),
        ];
        let run = run_lines("mini-swe-agent-trajectory", &records(&messages, None)).await;
        let calls: Vec<String> = events_of(&run)
            .iter()
            .filter_map(|e| match e {
                Event::ToolCall(c) => Some(format!("{} {} {:?}", c.title, c.status.as_str(), c)),
                _ => None,
            })
            .collect();
        assert!(calls[0].starts_with("ls completed") && calls[0].contains("file1.py"));
        assert!(calls[1].starts_with("pwd completed") && calls[1].contains("/testbed"));
    }

    #[tokio::test]
    async fn text_mode_actions_and_user_observations() {
        let messages = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "task"}),
            json!({"role": "assistant", "content": "THOUGHT: list\n```bash\nls\n```",
                   "extra": {"actions": [{"command": "ls"}]}}),
            json!({"role": "user", "content": "<returncode>0</returncode><output>a.py</output>"}),
            json!({"role": "user", "content": "Please always provide EXACTLY ONE action."}),
        ];
        let run = run_lines("mini-swe-agent-trajectory", &records(&messages, None)).await;
        let events = events_of(&run);
        assert!(
            matches!(&events[2], Event::ToolCall(c) if c.title == "ls" && c.status.as_str() == "completed" && format!("{c:?}").contains("a.py"))
        );
    }

    #[tokio::test]
    async fn responses_api_turns() {
        let messages = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "task"}),
            json!({"object": "response", "usage": {"input_tokens": 40, "output_tokens": 7,
                     "input_tokens_details": {"cached_tokens": 10}},
                   "output": [
                     {"type": "reasoning", "summary": []},
                     {"type": "message", "content": [{"type": "output_text", "text": "Checking."}]},
                     {"type": "function_call", "call_id": "fc_1", "name": "bash", "arguments": "{\"command\":\"ls\"}"},
                     "malformed item"]}),
            json!({"type": "function_call_output", "call_id": "fc_1", "output": "a.txt"}),
        ];
        let run = run_lines("mini-swe-agent-trajectory", &records(&messages, None)).await;
        let events = events_of(&run);
        assert!(matches!(&events[1], Event::AgentMessage(t) if t.text == "Checking."));
        assert!(
            matches!(&events[2], Event::ToolCall(c) if c.title == "ls" && c.status.as_str() == "completed")
        );
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.cached_read_tokens), (Some(30), Some(10)));
    }

    #[tokio::test]
    async fn an_exception_exit_fails_the_run_with_its_words() {
        // mini-swe-agent re-raises the exception and exits 1.
        let messages = vec![
            json!({"role": "system", "content": "s"}),
            json!({"role": "user", "content": "task"}),
            json!({"role": "exit", "content": "litellm.AuthenticationError: AnthropicException - invalid x-api-key",
                   "extra": {"exit_status": "AuthenticationError", "submission": "",
                             "exception_str": "litellm.AuthenticationError: AnthropicException - invalid x-api-key"}}),
        ];
        let run = run_lines_exit("mini-swe-agent-trajectory", &records(&messages, None), 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    #[tokio::test]
    async fn a_budget_exit_is_not_a_failure() {
        let mut messages = tool_calling();
        messages.pop();
        messages.push(json!({"role": "exit", "content": "LimitsExceeded",
                             "extra": {"exit_status": "LimitsExceeded", "submission": ""}}));
        let run = run_lines("mini-swe-agent-trajectory", &records(&messages, None)).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
    }

    #[tokio::test]
    async fn a_run_killed_mid_record_keeps_what_came_before() {
        let mut lines = records(&tool_calling()[..4], None);
        lines.push(r#"{"@bf":"messages","item":{"role":"assistant","content":"cut"#.to_string());
        let run = run_lines_exit("mini-swe-agent-trajectory", &lines, 137).await;
        assert!(run.result.is_err());
        let calls = events_of(&run)
            .iter()
            .filter(|e| matches!(e, Event::ToolCall(_)))
            .count();
        assert_eq!(calls, 1);
        assert!(run.log.iter().any(|l| l.contains("\"cut")));
    }

    #[tokio::test]
    async fn unknown_records_and_roles_are_ignored() {
        let lines = vec![
            json!({"@bf": "future", "item": {"x": 1}}).to_string(),
            json!({"@bf": "messages", "item": {"role": "observer", "content": "?"}}).to_string(),
            json!({"@bf": "messages", "item": {"role": "user", "content": "task"}}).to_string(),
            json!({"@bf": "messages", "item": {"role": "assistant", "content": "done"}})
                .to_string(),
        ];
        let run = run_lines("mini-swe-agent-trajectory", &lines).await;
        assert!(run.result.is_ok());
        assert_eq!(events_of(&run).len(), 2);
    }
}
