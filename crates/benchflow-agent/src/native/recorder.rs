//! Recording a native loop into the shared trajectory.
//!
//! The loops speak in ACP-shaped updates (`tool_call`, `tool_call_update`,
//! `agent_message_chunk`, ...), so a native run merges into events exactly
//! as an ACP agent's would and reads the same in the viewer and in ATIF.

use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde_json::{Value, json};

use super::chat::TokenUsage;
use crate::trajectory::{
    Event, MODEL_CALL, Receipt, SYSTEM_PROMPT, ToolCallStatus, Trajectory, Usage,
};

/// A shared, cloneable trajectory a loop writes and a writer task reads.
#[derive(Clone, Default)]
pub struct Recorder {
    inner: Arc<Mutex<Trajectory>>,
    /// The client's totals at the last [`Recorder::model_call`].
    counted: Arc<Mutex<(TokenUsage, u64)>>,
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

    /// The system prompt the loop gives the model.
    pub fn system_prompt(&self, text: &str) {
        let mut fields = serde_json::Map::new();
        fields.insert("text".into(), text.into());
        self.lock()
            .record_marker(SYSTEM_PROMPT, fields, Receipt::now());
    }

    /// A turn's model call came back: what the calls since the last turn
    /// used (the client's `totals` less what was already counted, so
    /// retries and summaries count with the turn they served), how many
    /// calls that was, and why the last one stopped. Record it before the
    /// events the turn produces.
    pub fn model_call(
        &self,
        totals: &TokenUsage,
        calls: u64,
        finish: Option<&str>,
        model: Option<&str>,
    ) {
        let (usage, n) = {
            let mut counted = self.counted.lock().unwrap_or_else(|p| p.into_inner());
            let before = counted.0;
            let usage = TokenUsage {
                input_tokens: totals.input_tokens.saturating_sub(before.input_tokens),
                output_tokens: totals.output_tokens.saturating_sub(before.output_tokens),
                cache_read_tokens: totals
                    .cache_read_tokens
                    .saturating_sub(before.cache_read_tokens),
                cache_write_tokens: totals
                    .cache_write_tokens
                    .saturating_sub(before.cache_write_tokens),
                reasoning_tokens: totals
                    .reasoning_tokens
                    .saturating_sub(before.reasoning_tokens),
                cost_usd: match (totals.cost_usd, before.cost_usd) {
                    (Some(now), Some(then)) => Some(now - then),
                    (now, _) => now,
                },
            };
            let n = calls.saturating_sub(counted.1);
            *counted = (*totals, calls);
            (usage, n)
        };
        let mut fields = serde_json::Map::new();
        let mut u = json!({
            "prompt_tokens": usage.input_tokens,
            "completion_tokens": usage.output_tokens,
            "cached_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "reasoning_tokens": usage.reasoning_tokens,
        });
        if let Some(cost) = usage.cost_usd {
            u["cost_usd"] = json!(cost);
        }
        fields.insert("usage".into(), u);
        fields.insert("calls".into(), n.into());
        if let Some(finish) = finish {
            fields.insert("finish_reason".into(), finish.into());
        }
        if let Some(model) = model.filter(|m| !m.is_empty()) {
            fields.insert("model".into(), model.into());
        }
        self.lock()
            .record_marker(MODEL_CALL, fields, Receipt::now());
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

    /// E1 round 2: a turn's record holds what the calls since the last turn
    /// used, so a retry or a summary counts with the turn it served, and
    /// neither it nor the system prompt is a step in the summary.
    #[test]
    fn a_model_call_records_the_turns_share_of_the_totals() {
        let r = Recorder::new();
        r.system_prompt("Be brief.");
        let mut totals = TokenUsage {
            input_tokens: 100,
            output_tokens: 10,
            cache_read_tokens: 60,
            ..TokenUsage::default()
        };
        r.model_call(&totals, 1, Some("tool_calls"), Some("m"));
        totals.input_tokens += 250;
        totals.output_tokens += 30;
        totals.reasoning_tokens += 7;
        r.model_call(&totals, 3, Some("stop"), None);
        let events = r.snapshot();
        let calls: Vec<&serde_json::Map<String, Value>> = events
            .iter()
            .filter_map(|e| match e {
                Event::Other(map) if e.kind() == MODEL_CALL => Some(map),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0]["usage"]["prompt_tokens"], 100);
        assert_eq!(calls[0]["usage"]["cached_tokens"], 60);
        assert_eq!(calls[0]["calls"], 1);
        assert_eq!(calls[0]["model"], "m");
        assert_eq!(calls[1]["usage"]["prompt_tokens"], 250);
        assert_eq!(calls[1]["usage"]["completion_tokens"], 30);
        assert_eq!(calls[1]["usage"]["reasoning_tokens"], 7);
        assert_eq!(calls[1]["calls"], 2);
        assert_eq!(calls[1]["finish_reason"], "stop");
        assert!(events.iter().any(|e| e.kind() == SYSTEM_PROMPT));
        let summary = crate::trajectory::summarize(events.iter(), false, Some("native"));
        assert_eq!(summary.steps, 0);
    }

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
