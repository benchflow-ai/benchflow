//! `swe-agent-trajectory`: SWE-agent, its `.traj` file followed.
//!
//! SWE-agent (1.x) rewrites `<output_dir>/<instance>/<instance>.traj` after
//! every step: `{"trajectory": [...], "history": [...], "info": {...}}`.
//! The command runs it under `follow.py` with `trajectory` as the array and
//! each entry's `query` (the whole conversation so far) and `messages`
//! dropped (see [`super::follow`]), so each step arrives as `{"@bf":
//! "trajectory", "item": {"thought", "action", "observation", "response",
//! "execution_time", "state", "extra_info"}}`: the thought is the agent's
//! message, the action a command, the observation its output.
//!
//! `{"@bf": "info", "item": {"info": {...}}}` closes the run:
//! `model_stats` (`tokens_sent`, `tokens_received`, `instance_cost`) and
//! `exit_status` (`submitted`, `submitted (exit_cost)`, `exit_error`,
//! `exit_format`, ...). A budget or format exit is a notice. `exit_error`
//! and `exit_environment_error` mean the loop crashed (a rejected key, a
//! dead environment), which SWE-agent records as a last step, `Exit due to
//! unknown error: ...`, before exiting 0, so they fail the run with that
//! step's words. The console is kept for the log.

use serde_json::{Value, json};

use super::follow::{finish_call, followed, start_call};
use crate::headless::parser::{Failure, Out, Parser, count, message, notice, str_of};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct SweAgent {
    steps: usize,
    /// The words of an `Exit due to ...` step.
    exit: Option<String>,
}

/// Exit statuses that mean SWE-agent's loop crashed.
const CRASHES: [&str; 2] = ["exit_error", "exit_environment_error"];

impl Parser for SweAgent {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some((name, item)) = followed(line) else {
            return false;
        };
        match name.as_str() {
            "trajectory" => self.step(&item, out),
            "info" => self.info(item.get("info").unwrap_or(&item), out),
            _ => {}
        }
        true
    }
}

impl SweAgent {
    fn step(&mut self, step: &Value, out: &mut Vec<Out>) {
        self.steps += 1;
        let thought = str_of(step, "thought");
        let action = str_of(step, "action").trim();
        if action.is_empty() && thought.starts_with("Exit due to") {
            out.push(Out::Failure(Failure::notice(thought)));
            self.exit = Some(thought.to_string());
            return;
        }
        if !thought.trim().is_empty() {
            out.push(message(thought));
        }
        if action.is_empty() {
            return;
        }
        let id = format!("swe-agent-{}", self.steps);
        start_call(out, &id, "bash", Some(json!({"command": action})));
        let observation = match step.get("observation") {
            Some(Value::String(s)) => Some(s.clone()),
            Some(Value::Null) | None => None,
            Some(other) => Some(other.to_string()),
        };
        finish_call(out, &id, false, observation);
    }
}

impl SweAgent {
    fn info(&self, info: &Value, out: &mut Vec<Out>) {
        if let Some(stats) = info.get("model_stats") {
            let usage = Usage {
                input_tokens: count(stats.get("tokens_sent")),
                output_tokens: count(stats.get("tokens_received")),
                ..Usage::default()
            };
            if usage != Usage::default() {
                out.push(Out::Usage(usage));
            }
            if let Some(cost) = stats
                .get("instance_cost")
                .and_then(Value::as_f64)
                .filter(|c| *c > 0.0)
            {
                out.push(Out::Cost(cost));
            }
        }
        let status = str_of(info, "exit_status");
        if CRASHES.contains(&status) {
            let text = self
                .exit
                .clone()
                .unwrap_or_else(|| format!("SWE-agent stopped: {status}"));
            out.push(Out::Failure(Failure::fatal(text)));
        } else if !status.is_empty() && !status.starts_with("submitted") {
            out.push(notice("warning", &format!("SWE-agent stopped: {status}")));
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn lines(steps: &[serde_json::Value], info: serde_json::Value) -> Vec<String> {
        let mut lines = vec!["INFO     🤖 MODEL INPUT".to_string()];
        lines.extend(
            steps
                .iter()
                .map(|s| json!({"@bf": "trajectory", "item": s}).to_string()),
        );
        lines.push(json!({"@bf": "info", "item": {"info": info}}).to_string());
        lines
    }

    #[tokio::test]
    async fn steps_become_thoughts_commands_and_outputs() {
        // The .traj shape Harbor's converter reads (test_swe_agent.py).
        let steps = [
            json!({"response": "I'll look.\n```\nls\n```", "thought": "I'll look at the repository.",
                   "action": "ls\n", "observation": "README.md\nsrc", "execution_time": 0.1, "state": {}}),
            json!({"response": "", "thought": "", "action": "", "observation": ""}),
            json!({"thought": "Edit the file.", "action": "str_replace_editor create /app/hello.txt --file_text hi",
                   "observation": "File created successfully"}),
            json!({"thought": "Done.", "action": "submit", "observation": null}),
        ];
        let run = run_lines(
            "swe-agent-trajectory",
            &lines(
                &steps,
                json!({"exit_status": "submitted", "submission": "diff",
                       "model_stats": {"instance_cost": 0.12, "tokens_sent": 3000, "tokens_received": 400, "api_calls": 4}}),
            ),
        )
        .await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let calls: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCall(c) => Some(format!("{}|{}", c.title, c.status.as_str())),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls,
            [
                "ls|completed",
                "str_replace_editor create /app/hello.txt --file_text hi|completed",
                "submit|completed"
            ]
        );
        assert!(
            matches!(&events[1], Event::AgentMessage(t) if t.text == "I'll look at the repository.")
        );
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (Some(3000), Some(400)));
        assert_eq!(run.cost_usd, Some(0.12));
    }

    #[tokio::test]
    async fn an_error_exit_fails_the_run_although_swe_agent_exits_0() {
        // SWE-agent 1.1.0 on Daytona with a placeholder Anthropic key.
        let exit = "Exit due to unknown error: litellm.AuthenticationError: AnthropicException - {\"type\":\"error\",\"error\":{\"type\":\"authentication_error\",\"message\":\"invalid x-api-key\"}}";
        let l = lines(
            &[json!({"thought": exit, "action": "", "observation": "", "response": ""})],
            json!({"exit_status": "exit_error", "submission": "", "model_stats": {"instance_cost": 0, "tokens_sent": 0, "tokens_received": 0}}),
        );
        let run = run_lines_exit("swe-agent-trajectory", &l, 0).await;
        let err = run.result.as_ref().unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
        assert!(
            !events_of(&run)
                .iter()
                .any(|e| matches!(e, Event::AgentMessage(_))),
            "the crash is not the agent's message"
        );
    }

    #[tokio::test]
    async fn a_budget_exit_is_a_notice() {
        let l = lines(
            &[json!({"thought": "Look.", "action": "ls", "observation": "a"})],
            json!({"exit_status": "exit_cost", "model_stats": {}}),
        );
        let run = run_lines("swe-agent-trajectory", &l).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        assert!(
            events_of(&run)
                .iter()
                .any(|e| matches!(e, Event::Notice(n) if format!("{n:?}").contains("exit_cost")))
        );
    }
}
