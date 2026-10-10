//! `openclaw-json`: `openclaw agent --local --json`, its session transcript
//! followed.
//!
//! OpenClaw logs free-form lines while it works and prints one JSON envelope
//! when it ends (`{"payloads": [{"text", "isReasoning"}], "meta":
//! {"agentMeta": {"sessionId", "sessionFile", "usage": {"input", "output",
//! "cacheRead", "cacheWrite"}}, ...}}`, often over several lines). The steps
//! are in its session transcript: before 2026.9 a JSON Lines file under
//! `~/.openclaw/agents/<agent>/sessions/`, since then rows of the agent's
//! SQLite database (`~/.openclaw/agents/<agent>/agent/openclaw-agent.sqlite`,
//! table `transcript_events`, each row the same JSON a line held). The
//! command follows it with `follow-sqlite.js` (see [`super::follow`]), so
//! each event arrives as `{"@bf": "session", "item": <event>}`. Events of
//! `type: message` carry a `message`:
//!
//! - `role: user`: the first is the task, already recorded;
//! - `role: assistant`: `content` parts `text`, `thinking` and `toolCall`
//!   (`id`, `name`, `arguments` as an object or a JSON string), `usage`
//!   (`input`, the uncached input, `output`, `cacheRead`, `cacheWrite`),
//!   and `errorMessage` when the model call failed;
//! - `role: toolResult`: `toolCallId`, `content`, `details.aggregated` (the
//!   whole output), `isError`.
//!
//! Without a session file, the envelope's payloads are the answer and its
//! usage the run's. An assistant `errorMessage` or the envelope's `error`
//! (`{"ok": false, "error": {"type", "message"}}`, followed by more log
//! lines) explains a failure; OpenClaw exits non-zero when the run failed.

use std::collections::{HashSet, VecDeque};

use serde_json::Value;

use super::follow::{arguments, finish_call, followed, start_call};
use crate::headless::parser::{
    Failure, Out, Parser, content_text, count, message, notice, str_of, thought,
};
use crate::trajectory::Usage;

/// Lines of console output kept for finding the envelope.
const KEEP_LINES: usize = 4000;

#[derive(Default)]
pub struct OpenclawJson {
    /// Console lines, the last of which may hold the envelope.
    console: VecDeque<String>,
    session_seen: bool,
    users: usize,
    started: HashSet<String>,
    usage: Usage,
    has_usage: bool,
}

impl Parser for OpenclawJson {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        if let Some((name, item)) = followed(line) {
            if name == "session" {
                self.session(&item, out);
            }
            return true;
        }
        if self.console.len() == KEEP_LINES {
            self.console.pop_front();
        }
        self.console.push_back(line.to_string());
        false
    }

    fn finish(&mut self, out: &mut Vec<Out>) {
        let text: Vec<&str> = self.console.iter().map(String::as_str).collect();
        let Some(envelope) = last_json_object(&text) else {
            return;
        };
        if !self.session_seen {
            for payload in envelope
                .get("payloads")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let text = str_of(payload, "text");
                if text.trim().is_empty() {
                    continue;
                }
                if payload.get("isReasoning").and_then(Value::as_bool) == Some(true) {
                    out.push(thought(text));
                } else {
                    out.push(message(text));
                }
            }
        }
        let meta = envelope.get("meta").unwrap_or(&Value::Null);
        if !self.has_usage
            && let Some(usage) = meta.get("agentMeta").and_then(|a| a.get("usage"))
        {
            self.add_usage(usage, out);
        }
        if let Some(error) = meta.get("error").or_else(|| envelope.get("error")) {
            let text = match error {
                Value::String(s) => s.clone(),
                other => match str_of(other, "message") {
                    "" => other.to_string(),
                    m => m.to_string(),
                },
            };
            out.push(Out::Failure(Failure::notice(text)));
        }
    }
}

impl OpenclawJson {
    fn session(&mut self, line: &Value, out: &mut Vec<Out>) {
        if str_of(line, "type") != "message" {
            return;
        }
        let Some(msg) = line.get("message") else {
            return;
        };
        self.session_seen = true;
        match str_of(msg, "role") {
            "user" => {
                self.users += 1;
            }
            "assistant" => {
                for part in msg
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    match str_of(part, "type") {
                        "text" => {
                            let text = str_of(part, "text");
                            if !text.is_empty() {
                                out.push(message(text));
                            }
                        }
                        "thinking" | "reasoning" => {
                            let text = [str_of(part, "thinking"), str_of(part, "text")]
                                .into_iter()
                                .find(|s| !s.is_empty())
                                .unwrap_or("");
                            if !text.is_empty() {
                                out.push(thought(text));
                            }
                        }
                        "toolCall" | "tool_use" => {
                            let id = str_of(part, "id").to_string();
                            if id.is_empty() || !self.started.insert(id.clone()) {
                                continue;
                            }
                            let input =
                                arguments(part.get("arguments").or_else(|| part.get("input")));
                            start_call(out, &id, str_of(part, "name"), input);
                        }
                        _ => {}
                    }
                }
                if let Some(text) = msg.get("content").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    out.push(message(text));
                }
                if let Some(usage) = msg.get("usage") {
                    self.add_usage(usage, out);
                }
                let error = str_of(msg, "errorMessage");
                if !error.trim().is_empty() {
                    out.push(Out::Failure(Failure::notice(error)));
                    out.push(notice("warning", error));
                }
            }
            "toolResult" => {
                let id = str_of(msg, "toolCallId");
                if id.is_empty() {
                    return;
                }
                let text = msg
                    .get("details")
                    .and_then(|d| d.get("aggregated"))
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                    .map(str::to_string)
                    .or_else(|| content_text(msg.get("content")));
                let failed = msg.get("isError").and_then(Value::as_bool) == Some(true);
                finish_call(out, id, failed, text);
            }
            _ => {}
        }
    }

    fn add_usage(&mut self, usage: &Value, out: &mut Vec<Out>) {
        let input = count(usage.get("input"));
        let output = count(usage.get("output"));
        let read = count(usage.get("cacheRead"));
        let write = count(usage.get("cacheWrite"));
        if input.is_none() && output.is_none() && read.is_none() {
            return;
        }
        let u = &mut self.usage;
        let add = |slot: &mut Option<u64>, n: Option<u64>| {
            if let Some(n) = n {
                *slot = Some(slot.unwrap_or(0) + n);
            }
        };
        add(&mut u.input_tokens, input);
        add(&mut u.output_tokens, output);
        add(&mut u.cached_read_tokens, read);
        add(&mut u.cached_write_tokens, write);
        self.has_usage = true;
        out.push(Out::Usage(self.usage));
    }
}

/// The last JSON object in `lines` that starts a line (Harbor reads the
/// last one that runs to the end; OpenClaw 2026.9 prints its error envelope
/// and then more log lines). Log lines hold braces of their own, so only
/// lines that start with `{` start a candidate.
fn last_json_object(lines: &[&str]) -> Option<Value> {
    (0..lines.len()).rev().find_map(|start| {
        if !lines[start].starts_with('{') {
            return None;
        }
        let mut text = String::new();
        for line in &lines[start..] {
            text.push_str(line);
            text.push('\n');
            if line.trim_end().ends_with('}')
                && let Ok(v @ Value::Object(_)) = serde_json::from_str::<Value>(&text)
            {
                return Some(v);
            }
        }
        None
    })
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::last_json_object;
    use crate::headless::tests::{events_of, run_lines, run_lines_exit};
    use crate::trajectory::Event;

    fn session(lines: &[Value]) -> Vec<String> {
        lines
            .iter()
            .map(|l| json!({"@bf": "session", "item": l}).to_string())
            .collect()
    }

    /// Harbor's test_openclaw_session_jsonl_to_atif_steps_minimal.
    fn minimal() -> Vec<Value> {
        vec![
            json!({"type": "message", "timestamp": "2026-01-01T00:00:00Z",
                   "message": {"role": "user", "content": [{"type": "text", "text": "hi"}]}}),
            json!({"type": "message", "timestamp": "2026-01-01T00:00:01Z",
                   "message": {"role": "assistant", "content": [
                       {"type": "text", "text": "hello "},
                       {"type": "toolCall", "id": "c1", "name": "exec", "arguments": {"command": "x"}}],
                     "usage": {"input": 1, "output": 2, "cacheRead": 0}}}),
            json!({"type": "message", "timestamp": "2026-01-01T00:00:02Z",
                   "message": {"role": "toolResult", "toolCallId": "c1", "toolName": "exec",
                     "content": [{"type": "text", "text": "out"}], "details": {"aggregated": "out"}}}),
            json!({"type": "message", "timestamp": "2026-01-01T00:00:03Z",
                   "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}],
                     "usage": {"input": 3, "output": 4, "cacheRead": 0}}}),
        ]
    }

    #[tokio::test]
    async fn the_session_gives_the_steps_and_the_envelope_is_only_log() {
        let mut lines = vec!["[agent] starting".to_string()];
        lines.extend(session(&minimal()));
        // The envelope, pretty-printed as OpenClaw prints it.
        let envelope = json!({"payloads": [{"text": "done"}],
            "meta": {"agentMeta": {"sessionId": "s", "usage": {"input": 4, "output": 6}}}});
        lines.extend(
            serde_json::to_string_pretty(&envelope)
                .unwrap()
                .lines()
                .map(str::to_string),
        );
        let run = run_lines("openclaw-json", &lines).await;
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
        assert!(
            matches!(&events[2], Event::ToolCall(c) if c.title == "x" && c.kind == "execute" && c.status.as_str() == "completed")
        );
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (Some(4), Some(6)));
        assert!(run.log.iter().any(|l| l.contains("payloads")));
    }

    #[tokio::test]
    async fn without_a_session_the_envelope_is_the_answer() {
        // test_convert_envelope_basic.
        let envelope = json!({"payloads": [{"text": "hello", "isReasoning": false},
                                           {"text": "think", "isReasoning": true}],
            "meta": {"agentMeta": {"sessionId": "sess-abc", "usage": {"input": 10, "output": 5, "cacheRead": 2}}}});
        let lines = vec![
            r#"[tools] raw_params={"path": "/x"}"#.to_string(),
            envelope.to_string(),
        ];
        let run = run_lines("openclaw-json", &lines).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        assert!(matches!(&events[1], Event::AgentMessage(t) if t.text == "hello"));
        assert!(matches!(&events[2], Event::AgentThought(t) if t.text == "think"));
        let u = run.usage.unwrap();
        assert_eq!(
            (u.input_tokens, u.cached_read_tokens, u.output_tokens),
            (Some(10), Some(2), Some(5))
        );
    }

    #[tokio::test]
    async fn a_model_error_explains_the_failure() {
        let mut lines = session(&[
            json!({"type": "message", "message": {"role": "user", "content": "task"}}),
            json!({"type": "message", "message": {"role": "assistant", "content": [], "stopReason": "error",
                   "errorMessage": "401 Incorrect API key provided: sk-fake. You can find your API key at https://platform.openai.com/account/api-keys."}}),
        ]);
        lines.push("Error: agent run failed".to_string());
        let run = run_lines_exit("openclaw-json", &lines, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    #[tokio::test]
    async fn the_2026_9_error_envelope_explains_the_failure() {
        // OpenClaw 2026.9.9 on Daytona with a placeholder Anthropic key.
        let lines: Vec<String> = [
            "[agent/embedded] tool-search: cataloged 53 tools behind compact prompt surface",
            "{",
            "  \"ok\": false,",
            "  \"error\": {",
            "    \"type\": \"cli_error\",",
            "    \"message\": \"⚠️ anthropic/claude-haiku-4-5 request failed (authentication failed, HTTP 401). Re-authenticate the provider and try again.\"",
            "  }",
            "}",
            "[openclaw] The CLI command failed.",
            "[openclaw] Try: openclaw doctor",
        ]
        .map(str::to_string)
        .to_vec();
        let run = run_lines_exit("openclaw-json", &lines, 1).await;
        let err = run.result.unwrap_err();
        assert_eq!(err.category(), "provider_auth", "{err}");
    }

    #[test]
    fn the_last_object_starting_a_line_is_the_envelope() {
        assert_eq!(
            last_json_object(&["prefix noise", r#"{"payloads": [], "meta": {}}"#, ""]),
            Some(json!({"payloads": [], "meta": {}}))
        );
        assert_eq!(
            last_json_object(&["{\"a\": 1}", "trailing text"]),
            Some(json!({"a": 1}))
        );
        assert_eq!(last_json_object(&["  {\"a\": 1}", "{ not json"]), None);
        assert_eq!(
            last_json_object(&["{", "  \"a\": {", "    \"b\": 1", "  }", "}"]),
            Some(json!({"a": {"b": 1}}))
        );
        assert_eq!(last_json_object(&[]), None);
    }

    /// The transcript OpenClaw 2026.9.9 kept in SQLite on a T7.1 `hello`
    /// run (deepseek-v4-flash through the gateway), one event per line.
    const TRANSCRIPT: &str = include_str!("testdata/openclaw-2026.9.9-transcript.jsonl");

    /// A Node.js with `node:sqlite` and zstd (23.4+), as OpenClaw needs.
    fn node_with_sqlite() -> Option<std::path::PathBuf> {
        let paths = std::env::var_os("PATH")?;
        let node = std::env::split_paths(&paths)
            .map(|d| d.join("node"))
            .find(|p| p.is_file())?;
        let ok = std::process::Command::new(&node)
            .args([
                "-e",
                "require('node:sqlite'); require('node:zlib').zstdCompressSync",
            ])
            .output()
            .ok()?
            .status
            .success();
        ok.then_some(node)
    }

    /// Guards T7.1's "openclaw records no steps": 2026.9 moved its transcript
    /// from JSON Lines files to SQLite, so the old follower found nothing.
    #[tokio::test]
    async fn the_sqlite_transcript_is_followed_into_steps() {
        let Some(node) = node_with_sqlite() else {
            eprintln!("skipped: no Node.js with node:sqlite on PATH");
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("openclaw-agent.sqlite");
        let events = dir.path().join("events.jsonl");
        std::fs::write(&events, TRANSCRIPT).unwrap();
        // OpenClaw's two tables, as far as the follower reads them; the
        // assistant's tool call stored compressed, as OpenClaw stores large
        // events.
        let setup = r#"
const { DatabaseSync } = require('node:sqlite');
const zlib = require('node:zlib');
const fs = require('node:fs');
const [db, events] = process.argv.slice(1);
const d = new DatabaseSync(db);
d.exec(`CREATE TABLE session_nodes (session_key TEXT PRIMARY KEY, current_session_id TEXT NOT NULL);
        CREATE TABLE transcript_events (session_id TEXT NOT NULL, seq INTEGER NOT NULL, event_json TEXT,
          created_at INTEGER NOT NULL, event_zstd BLOB, PRIMARY KEY (session_id, seq));`);
d.prepare('INSERT INTO session_nodes VALUES (?, ?)').run('agent:main:main', 's1');
d.prepare('INSERT INTO session_nodes VALUES (?, ?)').run('agent:other:main', 's2');
const add = d.prepare('INSERT INTO transcript_events VALUES (?, ?, ?, 0, ?)');
fs.readFileSync(events, 'utf8').trim().split('\n').forEach((line, seq) => {
  if (line.includes('"toolCall"')) add.run('s1', seq, null, zlib.zstdCompressSync(Buffer.from(line)));
  else add.run('s1', seq, line, null);
});
add.run('s2', 0, '{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"another session"}]}}', null);
"#;
        let made = std::process::Command::new(&node)
            .args(["-e", setup])
            .arg(&db)
            .arg(&events)
            .output()
            .unwrap();
        assert!(
            made.status.success(),
            "{}",
            String::from_utf8_lossy(&made.stderr)
        );
        let follower = dir.path().join("follow-sqlite.js");
        std::fs::write(&follower, super::super::follow::FOLLOW_SQLITE.1).unwrap();
        let out = std::process::Command::new(&node)
            .arg(&follower)
            .arg(&db)
            .arg("agent:main:main")
            .args([
                "--",
                "sh",
                "-c",
                "echo '[agent] working'; printf 'no newline'; exit 3",
            ])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(3), "the agent's exit status");
        let stdout = String::from_utf8(out.stdout).unwrap();
        let lines: Vec<String> = stdout.lines().map(str::to_string).collect();
        assert!(lines.contains(&"[agent] working".to_string()), "{stdout}");
        assert!(lines.contains(&"no newline".to_string()), "{stdout}");
        assert_eq!(
            lines
                .iter()
                .filter(|l| l.starts_with(r#"{"@bf":"session""#))
                .count(),
            TRANSCRIPT.lines().count(),
            "{stdout}"
        );
        assert!(!stdout.contains("another session"));

        let run = run_lines("openclaw-json", &lines).await;
        let events = events_of(&run);
        let calls: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCall(c) => Some(c),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 1, "{events:?}");
        assert_eq!(calls[0].status.as_str(), "completed");
        assert!(
            calls[0].title.contains("printf 'Hello, world!'"),
            "{}",
            calls[0].title
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::AgentMessage(m) if m.text.contains("13 bytes")))
        );
        let usage = run.usage.unwrap();
        assert!(usage.output_tokens.unwrap_or(0) > 0, "{usage:?}");
    }
}
