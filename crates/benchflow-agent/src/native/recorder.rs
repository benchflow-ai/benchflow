//! Recording a native loop into the shared trajectory.
//!
//! The loops speak in ACP-shaped updates (`tool_call`, `tool_call_update`,
//! `agent_message_chunk`, ...), so a native run merges into events exactly
//! as an ACP agent's would and reads the same in the viewer and in ATIF.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde_json::{Value, json};

use super::chat::TokenUsage;
use crate::trajectory::{Event, Receipt, ToolCallStatus, Trajectory, Usage};

/// A shared, cloneable trajectory a loop writes and a writer task reads.
#[derive(Clone, Default)]
pub struct Recorder {
    inner: Arc<Mutex<Trajectory>>,
}

impl std::fmt::Debug for Recorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Recorder").finish_non_exhaustive()
    }
}

/// An ACP text content block, as tool call content.
pub fn text_content(text: &str) -> Value {
    json!({"type": "content", "content": {"type": "text", "text": text}})
}

impl Recorder {
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> MutexGuard<'_, Trajectory> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn apply(&self, update: Value) {
        self.lock().apply_update(&update, Receipt::now());
    }

    /// A prompt sent to the model on the user's behalf.
    pub fn user(&self, text: &str) {
        self.lock().record_user_prompt(text, Receipt::now());
    }

    /// Text the model wrote.
    pub fn message(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.apply(json!({"sessionUpdate": "agent_message_chunk",
                          "content": {"type": "text", "text": text}}));
    }

    /// The model's reasoning.
    pub fn thought(&self, text: &str) {
        if text.is_empty() {
            return;
        }
        self.apply(json!({"sessionUpdate": "agent_thought_chunk",
                          "content": {"type": "text", "text": text}}));
    }

    /// A tool call starts: `kind` is ACP's (`execute`, `read`, `edit`,
    /// `other`), `tool_name` the loop's own name for the tool.
    pub fn tool_start(&self, id: &str, kind: &str, title: &str, tool_name: &str, raw_input: Value) {
        let mut t = self.lock();
        t.apply_update(
            &json!({"sessionUpdate": "tool_call", "toolCallId": id, "kind": kind,
                    "title": title, "status": "in_progress", "rawInput": raw_input}),
            Receipt::now(),
        );
        if let Some(call) = t.tool_call_mut(id) {
            call.tool_name = Some(tool_name.to_string());
        }
    }

    /// A tool call ends with `status`, its structured result, and the text
    /// the model reads.
    pub fn tool_finish(
        &self,
        id: &str,
        status: ToolCallStatus,
        raw_output: Value,
        text: Option<&str>,
    ) {
        let mut update = json!({"sessionUpdate": "tool_call_update", "toolCallId": id,
                                "status": status.as_str(), "rawOutput": raw_output});
        if let Some(text) = text {
            update["content"] = json!([text_content(text)]);
        }
        self.apply(update);
    }

    /// Something BenchFlow says about the run (a budget stop, a summary).
    pub fn notice(&self, severity: &str, title: &str, description: Option<&str>) {
        let mut update = json!({"sessionUpdate": "notice", "severity": severity, "title": title});
        if let Some(d) = description {
            update["description"] = Value::String(d.to_string());
        }
        self.apply(update);
    }

    /// The wall-clock budget ran out (0.8's `agent_timeout` event).
    pub fn timeout(&self, budget: Duration) {
        self.lock().record_timeout(budget, Receipt::now());
    }

    /// The session's usage so far.
    pub fn usage(&self, usage: &TokenUsage) {
        self.lock().record_usage(Usage {
            input_tokens: Some(usage.input_tokens),
            output_tokens: Some(usage.output_tokens),
            total_tokens: Some(usage.total()),
            cached_read_tokens: Some(usage.cache_read_tokens),
            cached_write_tokens: Some(usage.cache_write_tokens),
            thought_tokens: Some(usage.reasoning_tokens),
        });
    }

    /// Close the current run of text.
    pub fn finish(&self) {
        self.lock().finish_prompt();
    }

    pub fn snapshot(&self) -> Vec<Event> {
        self.lock().snapshot()
    }

    pub fn with<R>(&self, f: impl FnOnce(&Trajectory) -> R) -> R {
        f(&self.lock())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_call_merges_into_one_event_with_its_name() {
        let r = Recorder::new();
        r.user("do it");
        r.thought("thinking");
        r.message("Running ls.");
        r.tool_start("c1", "execute", "ls", "bash", json!({"command": "ls"}));
        r.tool_finish(
            "c1",
            ToolCallStatus::Completed,
            json!({"stdout": "a"}),
            Some("a"),
        );
        r.finish();
        let events = r.snapshot();
        let kinds: Vec<&str> = events.iter().map(|e| e.kind()).collect();
        assert_eq!(
            kinds,
            [
                "user_message",
                "agent_thought",
                "agent_message",
                "tool_call"
            ]
        );
        let Event::ToolCall(call) = &events[3] else {
            panic!()
        };
        assert_eq!(call.status, ToolCallStatus::Completed);
        assert_eq!(call.tool_name.as_deref(), Some("bash"));
        assert_eq!(call.raw_input, Some(json!({"command": "ls"})));
        assert_eq!(call.content.len(), 1);
    }
}
