//! Recording a session: ACP `session/update` notifications merged into
//! trajectory events the way 0.8's `acp/session.py` merges them.
//!
//! - Consecutive message (or thought) chunks of the same stream join into one
//!   event when the run of text ends: at a tool call, a notice, the next
//!   prompt, or the prompt's end. A subagent's chunks (Claude's
//!   `_meta.claudeCode.parentToolUseId`) form their own stream.
//! - A `tool_call_update` merges into its `tool_call` by id: the status,
//!   title, content, raw input and output, and the agent's tool name. Its
//!   content replaces the call's, as ACP specifies (OpenCode resends the
//!   whole output so far on every update; 0.8 appended it, repeating the
//!   output); an empty list leaves the content as it was. An update for an
//!   unknown id opens a call.
//! - A `notice` (what the adapter composed, not what the agent said) is its
//!   own event and never joins the agent's text.
//! - Plans, usage updates and every other update kind are kept as they
//!   arrived in [`Trajectory::other_updates`]; they are not trajectory events.

use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;

use super::event::{AgentTimeout, Event, Notice, Receipt, Text, ToolCall, ToolCallStatus};
use super::skill;

/// The kinds of `session/update` the recorder merges into events.
const RECOGNIZED: [&str; 7] = [
    "tool_call",
    "tool_call_update",
    "agent_message_chunk",
    "text_update",
    "agent_thought",
    "agent_thought_chunk",
    "notice",
];

/// An update kept as the agent sent it, with when it arrived.
#[derive(Debug, Clone, PartialEq)]
pub struct RawUpdate {
    pub update: Value,
    pub receipt: Receipt,
}

/// Token counts an agent reports for a session (cumulative), from a prompt's
/// `usage`. Every field is optional because agents report different ones.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    pub cached_read_tokens: Option<u64>,
    pub cached_write_tokens: Option<u64>,
    pub thought_tokens: Option<u64>,
}

impl Usage {
    /// Read ACP's `usage` object (camelCase, or snake_case as some agents
    /// write it). Numbers given as strings are accepted; `None` when it
    /// holds no count at all (0.8's `normalize_acp_usage`).
    pub fn from_acp(value: &Value) -> Option<Self> {
        let map = value.as_object()?;
        let field = |camel: &str, snake: &str| {
            let v = map.get(snake).or_else(|| map.get(camel))?;
            match v {
                Value::Number(n) => n
                    .as_u64()
                    .or_else(|| n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)),
                Value::String(s) => s.trim().parse().ok(),
                Value::Bool(b) => Some(u64::from(*b)),
                _ => None,
            }
        };
        let usage = Self {
            input_tokens: field("inputTokens", "input_tokens"),
            output_tokens: field("outputTokens", "output_tokens"),
            total_tokens: field("totalTokens", "total_tokens"),
            cached_read_tokens: field("cachedReadTokens", "cached_read_tokens"),
            cached_write_tokens: field("cachedWriteTokens", "cached_write_tokens"),
            thought_tokens: field("thoughtTokens", "thought_tokens"),
        };
        (usage != Self::default()).then_some(usage)
    }

    /// Total tokens: the reported total, else input plus output.
    pub fn tokens(&self) -> Option<u64> {
        self.total_tokens
            .or(match (self.input_tokens, self.output_tokens) {
                (None, None) => None,
                (i, o) => Some(i.unwrap_or(0) + o.unwrap_or(0)),
            })
    }
}

/// The context window an agent reports in `usage_update`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ContextUsage {
    /// Tokens in the context now.
    pub used: Option<u64>,
    /// The context window.
    pub size: Option<u64>,
    /// Cost so far, as the agent reports it (`{"amount", "currency"}`).
    pub cost: Option<Value>,
}

/// What one update changed, for live progress.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    /// A tool call started (its index into [`Trajectory::tool_calls`]).
    ToolCallStarted(usize),
    /// A known tool call changed.
    ToolCallUpdated(usize),
    MessageText,
    ThoughtText,
    Notice,
    Plan,
    ContextUsage,
    /// Kept in [`Trajectory::other_updates`], or nothing to record.
    Other,
}

/// What the idle watchdog reads: activity counts and the pending calls.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activity {
    pub tool_calls: usize,
    pub message_chunks: usize,
    pub thought_chunks: usize,
    pub tool_call_updates: u64,
    /// Calls not yet terminal, sorted by id, each with the number of
    /// non-terminal updates it has had (its progress version).
    pub pending: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stream {
    Message,
    Thought,
}

#[derive(Debug, Clone)]
struct PendingText {
    stream: Stream,
    text: Text,
}

/// A session's trajectory as it is recorded.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    events: Vec<Event>,
    pending_text: Vec<PendingText>,
    /// Event indices of tool calls, in the order they started.
    tool_events: Vec<usize>,
    /// Tool call id to its latest entry in `tool_events`.
    tool_by_id: HashMap<String, usize>,
    message_chunks: usize,
    thought_chunks: usize,
    tool_call_updates: u64,
    pending_versions: HashMap<String, u64>,
    other_updates: Vec<RawUpdate>,
    plan: Option<Value>,
    context: Option<ContextUsage>,
    usage: Option<Usage>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a prompt before it is sent.
    pub fn record_user_prompt(&mut self, text: &str, receipt: Receipt) {
        self.flush_text();
        self.events.push(Event::UserMessage(Text {
            text: text.to_string(),
            parent_tool_call_id: None,
            receipt: Some(receipt),
        }));
    }

    /// Close the prompt's last run of text.
    pub fn finish_prompt(&mut self) {
        self.flush_text();
    }

    /// Record BenchFlow's stop at the wall-clock budget, as 0.8's
    /// `agent_timeout` event.
    pub fn record_timeout(&mut self, budget: Duration, receipt: Receipt) {
        self.flush_text();
        let pending = self.pending_tool_call_ids();
        self.events.push(Event::AgentTimeout(AgentTimeout {
            reason: "wall_clock_timeout".to_string(),
            timeout_sec: budget.as_secs_f64(),
            terminal_trajectory_complete: pending.is_empty(),
            pending_tool_call_ids: pending,
            receipt: Some(receipt),
        }));
    }

    /// Record an event of a kind 0.8 never wrote (a native loop's
    /// [`MODEL_CALL`](super::event::MODEL_CALL) or
    /// [`SYSTEM_PROMPT`](super::event::SYSTEM_PROMPT)), with its receipt.
    pub fn record_marker(
        &mut self,
        kind: &str,
        mut fields: serde_json::Map<String, Value>,
        receipt: Receipt,
    ) {
        self.flush_text();
        let mut map = serde_json::Map::new();
        map.insert("type".into(), kind.into());
        map.append(&mut fields);
        map.insert("receipt".into(), receipt.to_json());
        map.insert(
            "ts".into(),
            super::event::iso_from_ns(receipt.first_observed_ns).into(),
        );
        self.events.push(Event::Other(map));
    }

    /// Record a prompt's reported usage (cumulative for the session).
    pub fn record_usage(&mut self, usage: Usage) {
        self.usage = Some(usage);
    }

    /// Merge one `session/update` (the `params.update` object).
    pub fn apply_update(&mut self, update: &Value, receipt: Receipt) -> Change {
        let kind = update
            .get("sessionUpdate")
            .and_then(Value::as_str)
            .unwrap_or("");
        if !RECOGNIZED.contains(&kind) {
            return self.keep_other(kind, update, receipt);
        }
        match kind {
            "tool_call" => self.start_tool_call(update, receipt),
            "tool_call_update" => self.update_tool_call(update, receipt),
            "agent_message_chunk" | "agent_thought_chunk" => {
                let stream = if kind == "agent_message_chunk" {
                    Stream::Message
                } else {
                    Stream::Thought
                };
                let content = update.get("content");
                if content.and_then(|c| c.get("type")).and_then(Value::as_str) != Some("text") {
                    return Change::Other;
                }
                let text = content
                    .and_then(|c| c.get("text"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                self.record_text(text, stream, update, receipt)
            }
            // Agents that send whole texts rather than chunks.
            "text_update" | "agent_thought" => {
                let stream = if kind == "text_update" {
                    Stream::Message
                } else {
                    Stream::Thought
                };
                match update.get("text").and_then(Value::as_str) {
                    Some(text) if !text.is_empty() => {
                        self.record_text(text, stream, update, receipt)
                    }
                    _ => Change::Other,
                }
            }
            _ => self.record_notice(update, receipt),
        }
    }

    fn keep_other(&mut self, kind: &str, update: &Value, receipt: Receipt) -> Change {
        self.other_updates.push(RawUpdate {
            update: update.clone(),
            receipt,
        });
        match kind {
            "plan" => {
                self.plan = update.get("entries").cloned();
                Change::Plan
            }
            "usage_update" => {
                self.context = Some(ContextUsage {
                    used: update.get("used").and_then(Value::as_u64),
                    size: update.get("size").and_then(Value::as_u64),
                    cost: update.get("cost").filter(|c| !c.is_null()).cloned(),
                });
                Change::ContextUsage
            }
            _ => Change::Other,
        }
    }

    fn record_text(
        &mut self,
        text: &str,
        stream: Stream,
        update: &Value,
        receipt: Receipt,
    ) -> Change {
        match stream {
            Stream::Message => self.message_chunks += 1,
            Stream::Thought => self.thought_chunks += 1,
        }
        self.pending_text.push(PendingText {
            stream,
            text: Text {
                text: text.to_string(),
                parent_tool_call_id: parent_tool_call_id(update),
                receipt: Some(receipt),
            },
        });
        match stream {
            Stream::Message => Change::MessageText,
            Stream::Thought => Change::ThoughtText,
        }
    }

    fn record_notice(&mut self, update: &Value, receipt: Receipt) -> Change {
        let title = text_field(update.get("title")).trim().to_string();
        if title.is_empty() {
            return Change::Other;
        }
        let severity = text_field(update.get("severity")).trim().to_string();
        let description = text_field(update.get("description")).trim().to_string();
        self.flush_text();
        self.events.push(Event::Notice(Notice {
            severity: if severity.is_empty() {
                "info".to_string()
            } else {
                severity
            },
            title,
            description: (!description.is_empty()).then_some(description),
            receipt: Some(receipt),
        }));
        Change::Notice
    }

    fn push_tool_call(&mut self, call: ToolCall) -> usize {
        let index = self.tool_events.len();
        self.tool_by_id.insert(call.tool_call_id.clone(), index);
        self.tool_events.push(self.events.len());
        self.events.push(Event::ToolCall(Box::new(call)));
        index
    }

    fn start_tool_call(&mut self, update: &Value, receipt: Receipt) -> Change {
        self.flush_text();
        let title = update
            .get("title")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let kind = skill::canonical_kind(
            Some(
                update
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("other"),
            ),
            &title,
        );
        let id = update
            .get("toolCallId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut call = ToolCall::new(id, kind, title, receipt);
        // The opening notification may already carry content (codex-acp's
        // file diffs) and raw I/O.
        if let Some(content) = update.get("content").and_then(Value::as_array) {
            call.content.extend(content.iter().cloned());
        }
        absorb_raw_io(&mut call, update);
        absorb_attribution(&mut call, update);
        // An opening call may already be terminal (codex-acp file edits).
        match update.get("status") {
            None | Some(Value::Null) => {}
            Some(status) => match status.as_str().and_then(ToolCallStatus::parse) {
                Some(status) => set_status(&mut call, status, receipt),
                None => tracing::warn!("unknown tool call status {status}"),
            },
        }
        Change::ToolCallStarted(self.push_tool_call(call))
    }

    fn update_tool_call(&mut self, update: &Value, receipt: Receipt) -> Change {
        self.tool_call_updates += 1;
        let id = update
            .get("toolCallId")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let (index, opened) = match self.tool_by_id.get(&id) {
            Some(index) => (*index, false),
            None => {
                self.flush_text();
                let title = update
                    .get("title")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let kind = skill::canonical_kind(
                    Some(update.get("kind").and_then(Value::as_str).unwrap_or("tool")),
                    &title,
                );
                (
                    self.push_tool_call(ToolCall::new(id.clone(), kind, title, receipt)),
                    true,
                )
            }
        };
        let event_index = self.tool_events[index];
        let Event::ToolCall(call) = &mut self.events[event_index] else {
            unreachable!("tool_events points at tool calls")
        };
        let before = Fingerprint::of(call);

        if let Some(title) = update.get("title").and_then(Value::as_str)
            && !title.trim().is_empty()
        {
            call.title = title.to_string();
        }
        let status = match update.get("status") {
            None => ToolCallStatus::InProgress,
            Some(status) => status
                .as_str()
                .and_then(ToolCallStatus::parse)
                .unwrap_or_else(|| {
                    tracing::warn!("unknown tool call status {status}");
                    ToolCallStatus::InProgress
                }),
        };
        if let Some(content) = update.get("content").and_then(Value::as_array)
            && !content.is_empty()
        {
            call.content = content.clone();
        }
        set_status(call, status, receipt);
        absorb_raw_io(call, update);
        absorb_attribution(call, update);
        if !status.is_terminal() {
            *self.pending_versions.entry(id).or_insert(0) += 1;
        }
        // Legacy harnesses only show a skill in the call's later content.
        if call.kind != "skill"
            && skill::is_skill_call(&call.kind, &call.title, None, &call.content)
        {
            call.kind = "skill".to_string();
        }
        if Fingerprint::of(call) != before
            && let Some(own) = call.receipt.as_mut()
        {
            own.last_observed_ns = receipt.last_observed_ns;
        }
        if opened {
            Change::ToolCallStarted(index)
        } else {
            Change::ToolCallUpdated(index)
        }
    }

    /// Join the pending run of text into events.
    fn flush_text(&mut self) {
        let pending = std::mem::take(&mut self.pending_text);
        self.events.extend(merge_text(pending));
    }

    /// Every event so far, the current run of text merged in, without
    /// closing it (safe while the prompt runs).
    pub fn events(&self) -> impl Iterator<Item = Cow<'_, Event>> {
        self.events
            .iter()
            .map(Cow::Borrowed)
            .chain(merge_text(self.pending_text.clone()).map(Cow::Owned))
    }

    /// A copy of every event so far.
    pub fn snapshot(&self) -> Vec<Event> {
        self.events().map(Cow::into_owned).collect()
    }

    /// The tool calls, in the order they started.
    pub fn tool_calls(&self) -> impl Iterator<Item = &ToolCall> {
        self.tool_events.iter().map(|i| match &self.events[*i] {
            Event::ToolCall(call) => &**call,
            _ => unreachable!("tool_events points at tool calls"),
        })
    }

    /// The latest tool call with this id, to change in place (the native
    /// loops name their tools here).
    pub(crate) fn tool_call_mut(&mut self, id: &str) -> Option<&mut ToolCall> {
        let event = *self.tool_events.get(*self.tool_by_id.get(id)?)?;
        match self.events.get_mut(event)? {
            Event::ToolCall(call) => Some(call),
            _ => None,
        }
    }

    /// How many tool calls the agent made (0.8's `n_tool_calls`).
    pub fn tool_call_count(&self) -> usize {
        self.tool_events.len()
    }

    /// The tool call at `index` (as [`Change`] reports it).
    pub fn tool_call(&self, index: usize) -> Option<&ToolCall> {
        match self.events.get(*self.tool_events.get(index)?)? {
            Event::ToolCall(call) => Some(call),
            _ => None,
        }
    }

    /// Ids of calls that are not terminal yet, in the order they started.
    pub fn pending_tool_call_ids(&self) -> Vec<String> {
        self.tool_calls()
            .filter(|c| !c.status.is_terminal())
            .map(|c| c.tool_call_id.clone())
            .collect()
    }

    /// Activity counts for the idle watchdog.
    pub fn activity(&self) -> Activity {
        let mut pending: Vec<(String, u64)> = self
            .tool_calls()
            .filter(|c| !c.status.is_terminal())
            .map(|c| {
                (
                    c.tool_call_id.clone(),
                    self.pending_versions
                        .get(&c.tool_call_id)
                        .copied()
                        .unwrap_or(0),
                )
            })
            .collect();
        pending.sort();
        Activity {
            tool_calls: self.tool_call_count(),
            message_chunks: self.message_chunks,
            thought_chunks: self.thought_chunks,
            tool_call_updates: self.tool_call_updates,
            pending,
        }
    }

    /// Updates that are not trajectory events (plans, usage updates, mode
    /// changes, kinds this build does not know), as the agent sent them.
    pub fn other_updates(&self) -> &[RawUpdate] {
        &self.other_updates
    }

    /// The agent's latest plan entries.
    pub fn plan(&self) -> Option<&Value> {
        self.plan.as_ref()
    }

    /// The latest context-window report.
    pub fn context_usage(&self) -> Option<&ContextUsage> {
        self.context.as_ref()
    }

    /// The latest prompt's reported usage.
    pub fn usage(&self) -> Option<Usage> {
        self.usage
    }

    /// The root agent's text so far: its messages, without subagents'.
    pub fn final_text(&self) -> String {
        self.events()
            .filter_map(|e| match e.as_ref() {
                Event::AgentMessage(t) if t.parent_tool_call_id.is_none() => Some(t.text.clone()),
                _ => None,
            })
            .collect()
    }
}

/// Merge consecutive chunks of the same stream into text events.
fn merge_text(pending: Vec<PendingText>) -> impl Iterator<Item = Event> {
    let mut merged: Vec<PendingText> = Vec::new();
    for chunk in pending {
        match merged.last_mut() {
            Some(last)
                if last.stream == chunk.stream
                    && last.text.parent_tool_call_id == chunk.text.parent_tool_call_id =>
            {
                last.text.text.push_str(&chunk.text.text);
                if let (Some(own), Some(new)) = (last.text.receipt.as_mut(), chunk.text.receipt) {
                    own.last_observed_ns = new.last_observed_ns;
                }
            }
            _ => merged.push(chunk),
        }
    }
    merged.into_iter().map(|p| match p.stream {
        Stream::Message => Event::AgentMessage(p.text),
        Stream::Thought => Event::AgentThought(p.text),
    })
}

/// The fields a written tool call shows, to tell whether an update changed
/// it (0.8's `recorded_state`). Raw values are compared by hash; terminal
/// chunks by their total size and the exit code, since chunks only ever
/// append (folding them on every chunk would be quadratic).
#[derive(PartialEq, Eq)]
struct Fingerprint {
    status: ToolCallStatus,
    title: String,
    kind: String,
    content_len: usize,
    raw_input: u64,
    raw_output: u64,
    terminal: Option<(usize, u64)>,
    parent: Option<String>,
}

impl Fingerprint {
    fn of(call: &ToolCall) -> Self {
        let has_terminal = call.terminal_output.is_some() || call.terminal_exit.is_some();
        Self {
            status: call.status,
            title: call.title.clone(),
            kind: call.kind.clone(),
            content_len: call.content.len(),
            raw_input: hash_json(call.raw_input.as_ref()),
            raw_output: hash_json(call.raw_output.as_ref()),
            terminal: has_terminal.then(|| {
                let bytes = call.terminal_output.as_ref().map_or(0, String::len);
                let exit = call.terminal_exit.as_ref().and_then(|e| e.get("exit_code"));
                (bytes, hash_json(exit))
            }),
            parent: call.parent_tool_call_id.clone(),
        }
    }
}

fn hash_json(value: Option<&Value>) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.map(Value::to_string).hash(&mut hasher);
    hasher.finish()
}

/// Move a call to `status`, noting when it first became terminal.
fn set_status(call: &mut ToolCall, status: ToolCallStatus, receipt: Receipt) {
    let was_terminal = call.status.is_terminal();
    call.status = status;
    if status.is_terminal() {
        if !was_terminal {
            call.finished_observed_ns = Some(receipt.last_observed_ns);
        }
    } else {
        call.finished_observed_ns = None;
    }
}

/// Keep the latest `rawInput`/`rawOutput` and codex-acp 2.x's terminal
/// chunks (`_meta.terminal_output_delta`, `_meta.terminal_exit`).
fn absorb_raw_io(call: &mut ToolCall, update: &Value) {
    if let Some(input) = update.get("rawInput").filter(|v| !v.is_null()) {
        call.raw_input = Some(input.clone());
    }
    if let Some(output) = update.get("rawOutput").filter(|v| !v.is_null()) {
        call.raw_output = Some(output.clone());
    }
    let Some(meta) = update.get("_meta").and_then(Value::as_object) else {
        return;
    };
    if let Some(chunk) = meta
        .get("terminal_output_delta")
        .and_then(|d| d.get("data"))
        .and_then(Value::as_str)
    {
        call.terminal_output
            .get_or_insert_with(String::new)
            .push_str(chunk);
    }
    if let Some(exit) = meta.get("terminal_exit").filter(|v| v.is_object()) {
        call.terminal_exit = Some(exit.clone());
    }
}

/// Claude ACP's tool name and subagent parent (`_meta.claudeCode`). A call
/// named as its own parent is not attributed (the adapter's progress
/// heartbeat for a running subagent does that), and a conflicting second
/// parent is ignored.
fn absorb_attribution(call: &mut ToolCall, update: &Value) {
    if let Some(name) = claude_meta(update, "toolName") {
        call.tool_name = Some(name.to_string());
    }
    let Some(parent) = claude_meta(update, "parentToolUseId") else {
        return;
    };
    if parent == call.tool_call_id {
        return;
    }
    match &call.parent_tool_call_id {
        None => call.parent_tool_call_id = Some(parent.to_string()),
        Some(existing) if existing != parent => {
            tracing::warn!("ignoring conflicting ACP child-tool parent attribution");
        }
        Some(_) => {}
    }
}

fn claude_meta<'a>(update: &'a Value, key: &str) -> Option<&'a str> {
    update
        .get("_meta")?
        .get("claudeCode")?
        .get(key)?
        .as_str()
        .filter(|s| !s.is_empty())
}

/// The subagent a text chunk belongs to (Claude's `parentToolUseId`).
fn parent_tool_call_id(update: &Value) -> Option<String> {
    claude_meta(update, "parentToolUseId").map(str::to_string)
}

/// Python's `str(value or "")` for the string fields of a notice.
fn text_field(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(false)) => String::new(),
        Some(other) => other.to_string(),
    }
}
