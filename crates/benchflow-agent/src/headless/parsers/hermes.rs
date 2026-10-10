//! `hermes-session`: `hermes chat -q`, then `hermes sessions export`.
//!
//! Hermes (Nous Research) prints its progress and answer as text and keeps
//! the conversation in its session database. After `hermes chat` exits, the
//! command exports the home's sessions (`hermes sessions export`; the home
//! is this run's alone) and prints each as `{"@bf": "session", "item":
//! <session>}`; subagents' sessions (`source: tool`) are skipped. A session is
//! `{"id", "model", "input_tokens", "output_tokens", "cache_read_tokens",
//! "cache_write_tokens", "reasoning_tokens", "estimated_cost_usd",
//! "actual_cost_usd", "end_reason", "messages": [...]}`, its messages in
//! chat format (`tool_calls` as a list or a JSON string, `reasoning`).
//!
//! The console is kept for the log and for explaining a failure; when no
//! session arrives (the run was stopped, or the export failed), it is the
//! agent's message, so a stopped run keeps what Hermes said. Hermes exits 0
//! when it gives up on the model (`❌ Non-retryable client error (HTTP
//! 401). Aborting.`), so such a line fails the run.

use serde_json::Value;

use super::follow::{ChatMessages, followed};
use crate::headless::parser::{Failure, Out, Parser, count, message, str_of};
use crate::trajectory::Usage;

/// Console characters kept for a run that exports no session.
const KEEP_CHARS: usize = 256 * 1024;

#[derive(Default)]
pub struct HermesSession {
    console: String,
    sessions: usize,
}

impl Parser for HermesSession {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if let Some((name, item)) = followed(line) {
            if name == "session" {
                self.session(&item, out);
            }
            return true;
        }
        let trimmed = line.trim();
        if trimmed.starts_with('❌')
            && ["Non-retryable", "Aborting", "failed after"]
                .iter()
                .any(|w| trimmed.contains(w))
        {
            out.push(Out::Failure(Failure::fatal(trimmed)));
        }
        if self.console.len() < KEEP_CHARS {
            self.console.push_str(line);
            self.console.push('\n');
        }
        false
    }

    fn finish(&mut self, out: &mut Vec<Out>) {
        if self.sessions == 0 && !self.console.trim().is_empty() {
            out.push(message(&self.console));
        }
    }
}

impl HermesSession {
    fn session(&mut self, session: &Value, out: &mut Vec<Out>) {
        if str_of(session, "source") == "tool" {
            return;
        }
        self.sessions += 1;
        let mut chat = ChatMessages::default();
        for msg in session
            .get("messages")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            // Hermes stores tool calls as JSON text.
            let mut msg = msg.clone();
            if let Some(Value::String(calls)) = msg.get("tool_calls")
                && let Ok(parsed) = serde_json::from_str::<Value>(calls)
            {
                msg["tool_calls"] = parsed;
            }
            chat.message(&msg, None, out);
        }
        let usage = Usage {
            input_tokens: count(session.get("input_tokens")),
            output_tokens: count(session.get("output_tokens")),
            cached_read_tokens: count(session.get("cache_read_tokens")),
            cached_write_tokens: count(session.get("cache_write_tokens")),
            thought_tokens: count(session.get("reasoning_tokens")),
            total_tokens: None,
        };
        if usage.input_tokens.unwrap_or(0) + usage.output_tokens.unwrap_or(0) > 0 {
            out.push(Out::Usage(usage));
        }
        if let Some(cost) = ["actual_cost_usd", "estimated_cost_usd"]
            .iter()
            .find_map(|k| session.get(*k).and_then(Value::as_f64))
        {
            out.push(Out::Cost(cost));
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    /// A session as `hermes sessions export` writes it; the messages are
    /// Harbor's test_hermes_cli.py conversation.
    fn session() -> serde_json::Value {
        json!({"id": "20260710_101500_abc123", "source": "cli", "model": "claude-sonnet-4-6",
               "input_tokens": 1500, "output_tokens": 120, "cache_read_tokens": 300,
               "estimated_cost_usd": 0.0123, "end_reason": "completed",
               "messages": [
                 {"role": "user", "content": "Create hello.txt"},
                 {"role": "assistant", "content": "I'll create the file.", "reasoning": "Simple.",
                  "tool_calls": "[{\"id\": \"call_1\", \"type\": \"function\", \"function\": {\"name\": \"terminal\", \"arguments\": \"{\\\"command\\\": \\\"echo hi > hello.txt\\\"}\"}}]"},
                 {"role": "tool", "tool_call_id": "call_1", "content": "{\"output\": \"\", \"exit_code\": 0}"},
                 {"role": "assistant", "content": "Done: hello.txt says hi."}]})
    }

    #[tokio::test]
    async fn the_exported_session_is_the_trajectory() {
        let lines = vec![
            "╭─ Hermes ─╮".to_string(),
            "Done: hello.txt says hi.".to_string(),
            json!({"@bf": "session", "item": session()}).to_string(),
        ];
        let run = run_lines("hermes-session", &lines).await;
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
        assert!(
            matches!(&events[3], Event::ToolCall(c) if c.title == "echo hi > hello.txt" && c.status.as_str() == "completed")
        );
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.output_tokens, u.cached_read_tokens),
            (Some(1500), Some(120), Some(300))
        );
        assert_eq!(run.cost_usd, Some(0.0123));
        assert_eq!(run.log.len(), 2);
    }

    #[tokio::test]
    async fn without_a_session_the_console_is_the_message() {
        let lines = vec!["Working on it...".to_string(), "Partial answer".to_string()];
        let run = run_lines_exit("hermes-session", &lines, 0).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        assert!(
            matches!(&events_of(&run)[1], Event::AgentMessage(t) if t.text == "Working on it...\nPartial answer\n")
        );
    }

    #[tokio::test]
    async fn hermes_exits_0_after_a_rejected_key_and_still_fails() {
        // Hermes 0.19.0 on Daytona with a placeholder Anthropic key.
        let lines: Vec<String> = [
            "Query: Write the number 42 to `/app/answer.txt`.",
            "Initializing agent...",
            "⚠️  API call failed (attempt 1/3): AuthenticationError [HTTP 401]",
            "🔐 Anthropic 401 — authentication failed.",
            "   📝 Error: HTTP 401: invalid x-api-key",
            "❌ Non-retryable error (HTTP 401): HTTP 401: invalid x-api-key",
            "❌ Non-retryable client error (HTTP 401). Aborting.",
            "Messages:       1 (1 user, 0 tool calls)",
        ]
        .map(str::to_string)
        .to_vec();
        let mut lines = lines;
        lines.push(
            json!({"@bf": "session", "item": {"id": "s", "source": "cli",
            "messages": [{"role": "user", "content": "x"}]}})
            .to_string(),
        );
        lines.push(json!({"@bf": "session", "item": {"id": "t", "source": "tool",
            "messages": [{"role": "user", "content": "y"}, {"role": "assistant", "content": "sub"}]}}).to_string());
        let run = run_lines_exit("hermes-session", &lines, 0).await;
        let err = run.result.as_ref().unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
        assert_eq!(events_of(&run).len(), 1, "{:?}", events_of(&run));
    }

    #[tokio::test]
    async fn a_rejected_key_in_the_console_is_a_rejected_login() {
        let lines = vec![
            "API call failed: Error code: 401 - {'error': {'message': 'Invalid API key', 'type': 'authentication_error'}}".to_string(),
            json!({"@bf": "session", "item": {"id": "s", "messages": [{"role": "user", "content": "x"}]}}).to_string(),
        ];
        let run = run_lines_exit("hermes-session", &lines, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }
}
