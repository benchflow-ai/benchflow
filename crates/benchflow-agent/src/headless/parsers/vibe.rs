//! `vibe-streaming`: `vibe --output streaming` (Mistral Vibe 2.26), then
//! the run's export.
//!
//! Vibe 2.26 keeps its sessions in a journal (`~/.vibe/logs/session/unified/`)
//! instead of the `messages.jsonl` files older releases wrote, so the
//! command reads its streaming output instead: one JSON line per history
//! entry, printed once the entry is complete (`generationStatus:
//! "completed"`):
//!
//! - `type: message`: `role` `user` (the task, already recorded) or
//!   `assistant`, `content` a list of `{"type": "text", "text"}` blocks;
//! - `type: reasoning`: `text` (or `summary`);
//! - `type: effect`: one tool call with its result: `id`, `title`,
//!   `detail` (`toolName`, `kind`, `input`) and `state` (`status`:
//!   `completed`, `failed`, ...; `outputText`; `output`).
//!
//! After vibe exits, the command prints the run's export (`--output-dir`'s
//! `export.json`, trimmed) as `{"vibe_export": {"outcome", "error",
//! "session_id", "usage": {"input_tokens", "output_tokens",
//! "cached_input_tokens"}, "cost_usd", "steps"}}`. Its `input_tokens`
//! include the cached ones; `cost_usd` is vibe's own figure, from its price
//! table (0 for a model it does not price).

use serde_json::Value;

use crate::headless::parser::{
    Failure, Out, Parser, ToolDone, ToolStart, content_text, count, json_object, kind_for_tool,
    message, str_of, thought, tool_title,
};
use crate::trajectory::Usage;

#[derive(Default)]
pub struct VibeStreaming;

impl Parser for VibeStreaming {
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool {
        let Some(map) = json_object(line) else {
            return false;
        };
        let entry = Value::Object(map);
        if let Some(export) = entry.get("vibe_export") {
            export_of(export, out);
            return true;
        }
        match str_of(&entry, "type") {
            "message" => {
                if str_of(&entry, "role") == "assistant"
                    && let Some(text) = content_text(entry.get("content"))
                    && !text.trim().is_empty()
                {
                    out.push(message(&text));
                }
                true
            }
            "reasoning" => {
                let text = [str_of(&entry, "text"), str_of(&entry, "summary")]
                    .into_iter()
                    .find(|t| !t.trim().is_empty());
                if let Some(text) = text {
                    out.push(thought(text));
                }
                true
            }
            "effect" => {
                effect(&entry, out);
                true
            }
            _ => false,
        }
    }
}

/// A tool call and its result, from one `effect` entry.
fn effect(entry: &Value, out: &mut Vec<Out>) {
    let id = str_of(entry, "id");
    let detail = entry.get("detail").cloned().unwrap_or(Value::Null);
    let name = match str_of(&detail, "toolName") {
        "" => str_of(entry, "title"),
        n => n,
    };
    // `file_system.bash`: the kind is the last part's, else vibe's own kind.
    let short = name.rsplit('.').next().unwrap_or(name);
    let kind = match kind_for_tool(short) {
        "other" => kind_for_tool(str_of(&detail, "kind")),
        k => k,
    };
    let input = detail.get("input").filter(|i| !i.is_null());
    out.push(
        ToolStart {
            id,
            title: tool_title(short, input),
            kind,
            status: "in_progress",
            input: input.cloned(),
            tool_name: Some(name),
            parent: None,
        }
        .out(),
    );
    let state = entry.get("state").cloned().unwrap_or(Value::Null);
    let status = match str_of(&state, "status") {
        "completed" | "success" | "succeeded" => "completed",
        _ => "failed",
    };
    let text = match str_of(&state, "outputText") {
        "" => state.get("output").and_then(|o| {
            content_text(o.get("stdout"))
                .or_else(|| content_text(o.get("output")))
                .or_else(|| content_text(o.get("error")))
        }),
        t => Some(t.to_string()),
    };
    out.push(
        ToolDone {
            id,
            status,
            text,
            output: state.get("output").cloned(),
            parent: None,
        }
        .out(),
    );
}

/// The run's export: its usage, vibe's own cost, and its error.
fn export_of(export: &Value, out: &mut Vec<Out>) {
    if let Some(usage) = export.get("usage").filter(|u| u.is_object()) {
        let input = count(usage.get("input_tokens"));
        let cached = count(usage.get("cached_input_tokens")).unwrap_or(0);
        let output = count(usage.get("output_tokens"));
        if input.is_some() || output.is_some() {
            out.push(Out::Usage(Usage {
                input_tokens: input.map(|i| i.saturating_sub(cached)),
                output_tokens: output,
                cached_read_tokens: Some(cached),
                ..Usage::default()
            }));
        }
    }
    if let Some(cost) = export
        .get("cost_usd")
        .and_then(Value::as_f64)
        .filter(|c| *c > 0.0)
    {
        out.push(Out::Cost(cost));
    }
    let error = match export.get("error") {
        Some(Value::String(s)) if !s.trim().is_empty() => Some(s.clone()),
        Some(Value::Object(e)) => Some(
            e.get("message")
                .and_then(Value::as_str)
                .map_or_else(|| Value::Object(e.clone()).to_string(), str::to_string),
        ),
        _ => None,
    };
    if let Some(error) = error {
        out.push(Out::Failure(Failure::fatal(error)));
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use crate::headless::tests::{events_of, run_lines};
    use crate::trajectory::Event;

    /// `vibe --output streaming` on a T7.1 `hello` run (vibe 2.26.1,
    /// deepseek-v4-flash through the gateway), and its export's line.
    const STREAM: &str = include_str!("testdata/vibe-2.26.1-streaming.jsonl");

    /// Guards T7.1's "mistral-vibe headless loses its trajectory": vibe
    /// 2.26 writes no `messages.jsonl`, so the old export printed nothing.
    #[tokio::test]
    async fn the_stream_gives_the_steps_and_the_export_the_usage() {
        let lines: Vec<String> = STREAM.lines().map(str::to_string).collect();
        let run = run_lines("vibe-streaming", &lines).await;
        assert!(run.result.is_ok(), "{:?}", run.result);
        let events = events_of(&run);
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(kinds, ["user_message", "tool_call", "agent_message"]);
        let Event::ToolCall(call) = &events[1] else {
            panic!("{events:?}")
        };
        assert_eq!(call.kind, "execute");
        assert_eq!(call.status.as_str(), "completed");
        assert!(
            call.title.starts_with("printf 'Hello, world!'"),
            "{}",
            call.title
        );
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.output_tokens), (Some(15631), Some(252)));
        // Vibe prices nothing it does not know: no cost.
        assert_eq!(run.cost_usd, None);
    }

    #[tokio::test]
    async fn a_failed_tool_and_an_error_are_recorded() {
        let lines = [
            json!({"type": "effect", "id": "effect-1", "title": "file_system.read_file",
                   "detail": {"toolName": "file_system.read_file", "kind": "read", "input": {"path": "/nope"}},
                   "state": {"status": "failed", "output": {"error": "no such file"}}}),
            json!({"vibe_export": {"outcome": "failed", "error": "the model refused the request",
                   "usage": {"input_tokens": 10, "output_tokens": 0, "cached_input_tokens": 4}}}),
        ]
        .map(|l| l.to_string());
        let run = run_lines("vibe-streaming", &lines).await;
        let events = events_of(&run);
        let Some(Event::ToolCall(call)) = events.iter().find(|e| e.kind() == "tool_call") else {
            panic!("{events:?}")
        };
        assert_eq!(
            (call.kind.as_str(), call.status.as_str()),
            ("read", "failed")
        );
        assert!(run.result.is_err(), "the export's error fails the run");
        let u = run.usage.unwrap();
        assert_eq!((u.input_tokens, u.cached_read_tokens), (Some(6), Some(4)));
    }
}
