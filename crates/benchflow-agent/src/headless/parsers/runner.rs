//! `benchflow-runner`: the JSON lines BenchFlow's runner scripts write.
//!
//! SDK agents (`openhands-sdk`, `nemo-agent`, `strands`, `deerflow`,
//! `antigravity-sdk`, `langgraph`, `eve`) are a framework plus a runner
//! script from `crates/benchflow-agent/runners/` ([`crate::runners`]). Every
//! runner reports in this one format, one JSON object per line on stdout
//! (the runners send whatever else the framework prints to stderr):
//!
//! | `type` | Fields | Meaning |
//! |---|---|---|
//! | `start` | `agent`, `framework`, `version`, `model` | The runner built the agent. |
//! | `message` | `text` | A chunk of the agent's message. |
//! | `thought` | `text` | A chunk of its reasoning. |
//! | `tool_call` | `id`, `name`, `input` | A tool call started. |
//! | `tool_result` | `id`, `output`, `failed` | A tool call ended (`failed` true when the tool reported an error). |
//! | `usage` | `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`, `cost_usd` | Totals for the run so far. |
//! | `error` | `message`, `kind`, `fatal`, `fix` | The run failed (or, with `fatal: false`, something went wrong that it survived). `kind` is what the runner made of it: `login`, `usage_limit`, `provider`, `model`, `setup` (the user must give the agent something; `fix` says what), or `agent`. |
//! | `done` | `answer` | The run ended; `answer` is the final answer, recorded as the message when none was streamed. |
//!
//! A line with another `type`, or that is not JSON, goes to the agent log.

use serde_json::{Map, Value};

use crate::headless::parser::{
    Failure, Hint, Out, Parser, ToolDone, ToolStart, content_text, count, json_object,
    kind_for_tool, message, str_of, thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct BenchflowRunner {
    /// Whether any message text was streamed (a `done` answer then repeats it).
    streamed_text: bool,
}

impl Parser for BenchflowRunner {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(map) = json_object(line) else {
            return false;
        };
        self.event(&map, out)
    }
}

impl BenchflowRunner {
    fn event(&mut self, map: &Map<String, Value>, out: &mut Vec<Out>) -> bool {
        let event = Value::Object(map.clone());
        match str_of(&event, "type") {
            "start" => true,
            "message" => {
                let text = str_of(&event, "text");
                if !text.is_empty() {
                    self.streamed_text = true;
                    out.push(message(text));
                }
                true
            }
            "thought" => {
                let text = str_of(&event, "text");
                if !text.is_empty() {
                    out.push(thought(text));
                }
                true
            }
            "tool_call" => {
                let name = str_of(&event, "name");
                let input = event.get("input").filter(|v| !v.is_null());
                let title = match str_of(&event, "title") {
                    "" => tool_title(name, input),
                    title => title.to_string(),
                };
                out.push(
                    ToolStart {
                        id: str_of(&event, "id"),
                        title,
                        kind: kind_for_tool(name),
                        status: "in_progress",
                        input: input.cloned(),
                        tool_name: Some(name),
                        parent: event.get("parent").and_then(Value::as_str),
                    }
                    .out(),
                );
                true
            }
            "tool_result" => {
                let failed = event.get("failed").and_then(Value::as_bool) == Some(true);
                let output = event.get("output").filter(|v| !v.is_null());
                out.push(
                    ToolDone {
                        id: str_of(&event, "id"),
                        status: if failed { "failed" } else { "completed" },
                        text: content_text(output),
                        output: output.cloned(),
                        parent: event.get("parent").and_then(Value::as_str),
                    }
                    .out(),
                );
                true
            }
            "usage" => {
                let usage = Usage {
                    input_tokens: count(event.get("input_tokens")),
                    output_tokens: count(event.get("output_tokens")),
                    total_tokens: count(event.get("total_tokens")),
                    cached_read_tokens: count(event.get("cache_read_tokens")),
                    cached_write_tokens: count(event.get("cache_write_tokens")),
                    thought_tokens: count(event.get("reasoning_tokens")),
                };
                if usage != Usage::default() {
                    out.push(Out::Usage(usage));
                }
                if let Some(cost) = event.get("cost_usd").and_then(Value::as_f64) {
                    out.push(Out::Cost(cost));
                }
                true
            }
            "error" => {
                let text = match str_of(&event, "message") {
                    "" => "the runner reported an error without a message".to_string(),
                    text => text.to_string(),
                };
                let hint = match str_of(&event, "kind") {
                    "login" => Some(Hint::Login),
                    "usage_limit" => Some(Hint::UsageLimit),
                    "provider" => Some(Hint::Provider),
                    "model" => Some(Hint::Model),
                    "setup" => Some(Hint::Setup),
                    _ => None,
                };
                let fatal = event.get("fatal").and_then(Value::as_bool) != Some(false);
                let failure = if fatal {
                    Failure::fatal(text)
                } else {
                    Failure::notice(text)
                };
                let fix = Some(str_of(&event, "fix"))
                    .filter(|f| !f.is_empty())
                    .map(str::to_string);
                out.push(Out::Failure(failure.hint(hint).fix(fix)));
                true
            }
            "done" => {
                let answer = str_of(&event, "answer");
                if !answer.trim().is_empty() && !self.streamed_text {
                    out.push(message(answer));
                }
                true
            }
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::HeadlessError;
    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::{Event, ToolCallStatus};

    fn lines(values: &[serde_json::Value]) -> Vec<String> {
        values.iter().map(|v| v.to_string()).collect()
    }

    /// What run_openhands_sdk.py wrote for a run that edited a file (its
    /// output on the mock model in the runner tests).
    #[tokio::test]
    async fn a_run_with_tools_thoughts_and_usage() {
        let run = run_lines(
            "benchflow-runner",
            &lines(&[
                json!({"type": "start", "agent": "openhands-sdk", "framework": "openhands-sdk", "version": "1.54.0", "model": "openai/gpt-x"}),
                json!({"type": "thought", "text": "Write the file first."}),
                json!({"type": "tool_call", "id": "call_1", "name": "terminal", "input": {"command": "echo hi > out.txt"}}),
                json!({"type": "tool_result", "id": "call_1", "output": "", "failed": false}),
                json!({"type": "tool_call", "id": "call_2", "name": "file_editor", "input": {"command": "view", "path": "/app/out.txt"}}),
                json!({"type": "tool_result", "id": "call_2", "output": "hi", "failed": true}),
                json!({"type": "message", "text": "Done: out.txt says hi."}),
                json!({"type": "usage", "input_tokens": 1200, "output_tokens": 80, "cache_read_tokens": 1000, "cost_usd": 0.0042}),
                json!({"type": "done", "answer": "Done: out.txt says hi."}),
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
                "agent_message"
            ]
        );
        let Event::ToolCall(first) = &events[2] else {
            panic!()
        };
        assert_eq!(
            (first.title.as_str(), first.status),
            ("echo hi > out.txt", ToolCallStatus::Completed)
        );
        let Event::ToolCall(second) = &events[3] else {
            panic!()
        };
        assert_eq!(second.status, ToolCallStatus::Failed);
        // The done answer repeats the streamed message; it is not added twice.
        assert!(matches!(&events[4], Event::AgentMessage(t) if t.text == "Done: out.txt says hi."));
        let usage = run.usage.unwrap();
        assert_eq!(
            (usage.input_tokens, usage.cached_read_tokens),
            (Some(1200), Some(1000))
        );
        assert_eq!(run.cost_usd, Some(0.0042));
    }

    /// nemo-agent and eve report only their answer at the end.
    #[tokio::test]
    async fn an_answer_only_at_the_end_is_the_message() {
        let run = run_lines(
            "benchflow-runner",
            &lines(&[
                json!({"type": "start", "agent": "nemo-agent", "version": "1.9.0"}),
                json!({"type": "done", "answer": "42"}),
            ]),
        )
        .await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        assert!(matches!(&events_of(&run)[1], Event::AgentMessage(t) if t.text == "42"));
    }

    /// What run_strands.py wrote when the provider refused a placeholder
    /// key (OpenAI's 401, as the openai package raises it).
    #[tokio::test]
    async fn a_rejected_key_is_a_login_failure() {
        let run = run_lines_exit(
            "benchflow-runner",
            &lines(&[
                json!({"type": "start", "agent": "strands"}),
                json!({"type": "error", "kind": "login", "message": "AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key provided: sk-bench*******0000.'}}"}),
            ]),
            1,
        )
        .await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    /// The runner's kind decides when the words alone say nothing.
    #[tokio::test]
    async fn the_kind_is_the_fallback_for_unfamiliar_words() {
        let run = run_lines_exit(
            "benchflow-runner",
            &lines(&[
                json!({"type": "error", "kind": "login", "message": "ClientError: the gateway said no"}),
            ]),
            1,
        )
        .await;
        assert_eq!(run.result.unwrap_err().category(), "provider_auth");
    }

    /// A missing graph, factory or app is the user's to give, whatever the
    /// message resembles.
    #[tokio::test]
    async fn missing_user_code_says_what_to_set() {
        let run = run_lines_exit(
            "benchflow-runner",
            &lines(&[json!({
                "type": "error",
                "kind": "setup",
                "message": "langgraph needs a graph, and /app has no langgraph.json (missing API key is not the problem)",
                "fix": "put a LangGraph project in the task, or pass --env LANGGRAPH_PROJECT=<folder in the sandbox>",
            })]),
            2,
        )
        .await;
        let err = run.result.unwrap_err();
        let HeadlessError::Setup { message, fix } = &err else {
            panic!("{err:?}")
        };
        assert!(message.contains("langgraph.json"));
        assert!(
            err.next_step("langgraph").contains("LANGGRAPH_PROJECT"),
            "{fix:?}"
        );
        assert_eq!(err.category(), "agent_integration");
        assert!(!err.is_retryable());
    }

    /// Each Python runner's real output on the scripted model in
    /// runners/tests/mock_model.py: a shell tool call, its result, the
    /// answer, and usage (nemo-agent's default workflow only answers).
    #[tokio::test]
    async fn every_runners_recorded_output_parses() {
        let fixtures = [
            (
                "openhands-sdk",
                include_str!("../../../runners/tests/fixtures/openhands-sdk.jsonl"),
                true,
            ),
            (
                "strands",
                include_str!("../../../runners/tests/fixtures/strands.jsonl"),
                true,
            ),
            (
                "deerflow",
                include_str!("../../../runners/tests/fixtures/deerflow.jsonl"),
                true,
            ),
            (
                "langgraph",
                include_str!("../../../runners/tests/fixtures/langgraph.jsonl"),
                true,
            ),
            (
                "nemo-agent",
                include_str!("../../../runners/tests/fixtures/nemo-agent.jsonl"),
                false,
            ),
        ];
        for (agent, text, tools) in fixtures {
            let lines: Vec<String> = text.lines().map(str::to_string).collect();
            let run = run_lines("benchflow-runner", &lines).await;
            assert!(run.result.is_ok(), "{agent}: {:?}", run.result);
            assert!(
                run.log.is_empty(),
                "{agent}: lines the parser did not read: {:?}",
                run.log
            );
            let events = events_of(&run);
            let calls: Vec<_> = events
                .iter()
                .filter_map(|e| match e {
                    Event::ToolCall(c) => Some(c),
                    _ => None,
                })
                .collect();
            if tools {
                assert_eq!(calls.len(), 1, "{agent}");
                assert_eq!(calls[0].status, ToolCallStatus::Completed, "{agent}");
                assert!(
                    calls[0].title.contains("bf_mock_out.txt"),
                    "{agent}: {}",
                    calls[0].title
                );
            }
            assert!(
                matches!(events.last(), Some(Event::AgentMessage(t)) if t.text.starts_with("BF-MOCK-DONE")),
                "{agent}: {events:?}"
            );
            assert!(run.usage.unwrap().input_tokens.unwrap() >= 100, "{agent}");
        }
        let text = include_str!("../../../runners/tests/fixtures/strands-rejected-key.jsonl");
        let lines: Vec<String> = text.lines().map(str::to_string).collect();
        let run = run_lines_exit("benchflow-runner", &lines, 1).await;
        assert_eq!(run.result.unwrap_err().category(), "provider_auth");
    }

    /// What run_dspy_rlm.py wrote against a scripted model in DSPy's chat
    /// format: each REPL step is a `python` call, and the tools the code
    /// called are calls of their own.
    #[tokio::test]
    async fn the_dspy_rlm_runners_output_parses_with_nested_tool_calls() {
        let text = include_str!("../../../runners/tests/fixtures/dspy-rlm.jsonl");
        let lines: Vec<String> = text.lines().map(str::to_string).collect();
        let run = run_lines("benchflow-runner", &lines).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        assert!(run.log.is_empty(), "{:?}", run.log);
        let events = events_of(&run);
        let names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCall(c) => c.tool_name.as_deref().or(Some(c.title.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(names.len(), 5, "{names:?}");
        assert!(
            matches!(events.last(), Some(Event::AgentMessage(t)) if t.text == "wrote hello.txt"),
            "{events:?}"
        );
        let usage = run.usage.unwrap();
        assert_eq!(
            (usage.input_tokens, usage.output_tokens),
            (Some(200), Some(20))
        );
    }

    /// What run_nemo_agent.py wrote when NVIDIA had retired the model.
    #[tokio::test]
    async fn a_retired_model_is_the_users_to_change() {
        let run = run_lines_exit(
            "benchflow-runner",
            &lines(&[json!({
                "type": "error",
                "kind": "model",
                "message": "Exception: [410] Gone The model 'meta/llama-3.3-70b-instruct' has reached its end of life on 2026-08-26T09:00:00Z and is no longer available.",
            })]),
            1,
        )
        .await;
        let err = run.result.unwrap_err();
        assert!(
            matches!(err, HeadlessError::ModelUnavailable { .. }),
            "{err:?}"
        );
    }

    /// A non-fatal error (a retry) does not fail a run that finished.
    #[tokio::test]
    async fn a_survived_error_is_only_a_notice() {
        let run = run_lines(
            "benchflow-runner",
            &lines(&[
                json!({"type": "error", "kind": "provider", "fatal": false, "message": "rate limit, retrying"}),
                json!({"type": "message", "text": "ok"}),
            ]),
        )
        .await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        assert!(run.log.iter().any(|l| l.contains("(not fatal)")));
    }

    #[tokio::test]
    async fn other_lines_go_to_the_log() {
        let run = run_lines(
            "benchflow-runner",
            &[
                "Loaded 0 skills".to_string(),
                json!({"type": "heartbeat"}).to_string(),
                json!({"type": "message", "text": "hi"}).to_string(),
            ],
        )
        .await;
        assert!(run.result.is_ok());
        assert!(run.log.iter().any(|l| l == "Loaded 0 skills"));
        assert!(run.log.iter().any(|l| l.contains("heartbeat")));
    }
}
