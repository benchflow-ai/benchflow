//! Trajectory events and their JSON form in 0.8's `acp_trajectory.jsonl`.
//!
//! [`Event::to_json`] writes exactly 0.8's keys in 0.8's order, so a file
//! BenchFlow writes reads like one 0.8 wrote. [`Event::from_json`] reads them
//! back; an event it cannot represent exactly (an unknown type, an unknown
//! key, a value of an unexpected shape) is kept as [`Event::Other`], as
//! written, so reading and writing a file never loses anything.

use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Utc};
use serde_json::{Map, Value, json};

/// When the host observed an event: Unix nanoseconds of its first and last
/// update. These are arrival times at BenchFlow, not model or tool
/// execution times.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Receipt {
    pub first_observed_ns: u64,
    pub last_observed_ns: u64,
}

impl Receipt {
    /// A receipt for something observed now.
    pub fn now() -> Self {
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        Self::at(ns)
    }

    /// A receipt for something observed at `ns` Unix nanoseconds.
    pub fn at(ns: u64) -> Self {
        Self {
            first_observed_ns: ns,
            last_observed_ns: ns,
        }
    }

    fn to_json(self) -> Value {
        json!({
            "source": "benchflow_host",
            "clock": "unix",
            "first_observed_ns": self.first_observed_ns.to_string(),
            "last_observed_ns": self.last_observed_ns.to_string(),
        })
    }

    fn from_json(value: &Value) -> Option<Self> {
        let map = value.as_object()?;
        if map.len() != 4 || map.get("source")? != "benchflow_host" || map.get("clock")? != "unix" {
            return None;
        }
        let ns = |key| map.get(key)?.as_str()?.parse::<u64>().ok();
        Some(Self {
            first_observed_ns: ns("first_observed_ns")?,
            last_observed_ns: ns("last_observed_ns")?,
        })
    }
}

/// A receipt time as 0.8 renders it: ISO 8601 UTC with microseconds, the
/// fraction left out when it is zero (Python's `isoformat`).
pub fn iso_from_ns(ns: u64) -> String {
    let micros = (ns / 1000) as i64;
    let Some(t) = DateTime::<Utc>::from_timestamp_micros(micros) else {
        return String::new();
    };
    let base = t.format("%Y-%m-%dT%H:%M:%S");
    match micros.rem_euclid(1_000_000) {
        0 => format!("{base}+00:00"),
        frac => format!("{base}.{frac:06}+00:00"),
    }
}

/// A tool call's lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ToolCallStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

impl ToolCallStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "pending" => Self::Pending,
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            _ => return None,
        })
    }

    /// Completed, failed or cancelled.
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

/// A run of agent or user text: consecutive chunks of one kind joined.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub text: String,
    /// The tool call (a subagent) whose stream this text belongs to.
    pub parent_tool_call_id: Option<String>,
    pub receipt: Option<Receipt>,
}

/// One tool call, with every update merged into it.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub tool_call_id: String,
    pub kind: String,
    pub title: String,
    pub status: ToolCallStatus,
    pub content: Vec<Value>,
    pub raw_input: Option<Value>,
    /// The agent's `rawOutput`, as sent.
    pub(crate) raw_output: Option<Value>,
    /// codex-acp 2.x's `_meta.terminal_output_delta` chunks, joined;
    /// `None` until the first chunk.
    pub(crate) terminal_output: Option<String>,
    /// codex-acp 2.x's `_meta.terminal_exit`.
    pub(crate) terminal_exit: Option<Value>,
    pub parent_tool_call_id: Option<String>,
    /// The agent's own name for the tool (Claude's `Bash`, where the kind
    /// says `execute`).
    pub tool_name: Option<String>,
    pub receipt: Option<Receipt>,
    /// When the update that made the call terminal arrived.
    pub finished_observed_ns: Option<u64>,
}

impl ToolCall {
    pub(crate) fn new(tool_call_id: String, kind: String, title: String, receipt: Receipt) -> Self {
        Self {
            tool_call_id,
            kind,
            title,
            status: ToolCallStatus::Pending,
            content: Vec::new(),
            raw_input: None,
            raw_output: None,
            terminal_output: None,
            terminal_exit: None,
            parent_tool_call_id: None,
            tool_name: None,
            receipt: Some(receipt),
            finished_observed_ns: None,
        }
    }

    /// The agent's `rawOutput`, completed from terminal chunks when it lacks
    /// the output (0.8's `ToolCallRecord.raw_output`): codex-acp 2.x sends a
    /// client without terminal support the output only as chunks, and a
    /// command can end with no `rawOutput` at all. The 1.x shape
    /// (`formatted_output`, `exit_code`) is rebuilt; an agent-sent
    /// `formatted_output` always wins.
    pub fn raw_output(&self) -> Option<Value> {
        if self.terminal_output.is_none() && self.terminal_exit.is_none() {
            return self.raw_output.clone();
        }
        let raw = self
            .raw_output
            .clone()
            .unwrap_or_else(|| Value::Object(Map::new()));
        let Value::Object(fields) = &raw else {
            return Some(raw);
        };
        if fields.contains_key("formatted_output") {
            return Some(raw);
        }
        let mut folded = Map::new();
        folded.insert(
            "formatted_output".into(),
            Value::String(self.terminal_output.clone().unwrap_or_default()),
        );
        folded.extend(fields.clone());
        if let Some(exit) = &self.terminal_exit {
            folded.insert(
                "exit_code".into(),
                exit.get("exit_code").cloned().unwrap_or(Value::Null),
            );
        }
        folded.entry("exit_code").or_insert(Value::Null);
        Some(Value::Object(folded))
    }

    /// The first line of the title, or the kind: what a progress line shows.
    pub fn display_title(&self) -> &str {
        let source = if self.title.trim().is_empty() {
            &self.kind
        } else {
            &self.title
        };
        source.trim().split('\n').next().unwrap_or("")
    }
}

/// A notice the adapter composed (a Codex warning, a model fallback): not
/// the agent's words.
#[derive(Debug, Clone, PartialEq)]
pub struct Notice {
    pub severity: String,
    pub title: String,
    pub description: Option<String>,
    pub receipt: Option<Receipt>,
}

/// BenchFlow stopped the agent at its wall-clock budget.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentTimeout {
    pub reason: String,
    pub timeout_sec: f64,
    pub pending_tool_call_ids: Vec<String>,
    /// True when no tool call was still pending.
    pub terminal_trajectory_complete: bool,
    pub receipt: Option<Receipt>,
}

/// The oracle ran the task's solution.
#[derive(Debug, Clone, PartialEq)]
pub struct Oracle {
    pub command: String,
    pub return_code: i64,
    /// The tail of the solution's output.
    pub stdout: String,
}

/// One line of `acp_trajectory.jsonl`.
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    UserMessage(Text),
    AgentMessage(Text),
    AgentThought(Text),
    ToolCall(Box<ToolCall>),
    Notice(Notice),
    AgentTimeout(AgentTimeout),
    Oracle(Oracle),
    /// The `nop` agent ran nothing.
    Nop {
        note: String,
    },
    /// An event this build does not model, kept as written.
    Other(Map<String, Value>),
}

/// The note 0.8 writes for the `nop` agent.
pub const NOP_NOTE: &str = "nop agent: nothing ran; the verifier scores the untouched workspace";

impl Event {
    /// The event's `type`.
    pub fn kind(&self) -> &str {
        match self {
            Self::UserMessage(_) => "user_message",
            Self::AgentMessage(_) => "agent_message",
            Self::AgentThought(_) => "agent_thought",
            Self::ToolCall(_) => "tool_call",
            Self::Notice(_) => "notice",
            Self::AgentTimeout(_) => "agent_timeout",
            Self::Oracle(_) => "oracle",
            Self::Nop { .. } => "nop",
            Self::Other(map) => map.get("type").and_then(Value::as_str).unwrap_or("unknown"),
        }
    }

    /// The `nop` agent's one event.
    pub fn nop() -> Self {
        Self::Nop {
            note: NOP_NOTE.to_string(),
        }
    }

    /// The event as 0.8 writes it, keys in 0.8's order.
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("type".into(), Value::String(self.kind().to_string()));
        let receipt = match self {
            Self::UserMessage(t) | Self::AgentMessage(t) | Self::AgentThought(t) => {
                out.insert("text".into(), Value::String(t.text.clone()));
                if let Some(parent) = &t.parent_tool_call_id {
                    out.insert("parent_tool_call_id".into(), Value::String(parent.clone()));
                }
                t.receipt
            }
            Self::ToolCall(tc) => {
                out.insert(
                    "tool_call_id".into(),
                    Value::String(tc.tool_call_id.clone()),
                );
                out.insert("kind".into(), Value::String(tc.kind.clone()));
                out.insert("title".into(), Value::String(tc.title.clone()));
                out.insert("status".into(), Value::String(tc.status.as_str().into()));
                out.insert("content".into(), Value::Array(tc.content.clone()));
                if let Some(input) = &tc.raw_input {
                    out.insert("raw_input".into(), input.clone());
                }
                if let Some(output) = tc.raw_output() {
                    out.insert("raw_output".into(), output);
                }
                if let Some(parent) = &tc.parent_tool_call_id {
                    out.insert("parent_tool_call_id".into(), Value::String(parent.clone()));
                }
                if let Some(name) = tc.tool_name.as_ref().filter(|n| !n.is_empty()) {
                    out.insert("tool_name".into(), Value::String(name.clone()));
                }
                if let Some(receipt) = tc.receipt {
                    out.insert("receipt".into(), receipt.to_json());
                    out.insert(
                        "started_at".into(),
                        iso_from_ns(receipt.first_observed_ns).into(),
                    );
                    if let Some(finished) = tc.finished_observed_ns {
                        out.insert("finished_at".into(), iso_from_ns(finished).into());
                    }
                }
                return Value::Object(out);
            }
            Self::Notice(n) => {
                out.insert("severity".into(), Value::String(n.severity.clone()));
                out.insert("title".into(), Value::String(n.title.clone()));
                if let Some(description) = &n.description {
                    out.insert("description".into(), Value::String(description.clone()));
                }
                n.receipt
            }
            Self::AgentTimeout(t) => {
                out.insert("reason".into(), Value::String(t.reason.clone()));
                out.insert("timeout_sec".into(), json!(t.timeout_sec));
                out.insert(
                    "pending_tool_call_ids".into(),
                    t.pending_tool_call_ids
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                );
                out.insert(
                    "terminal_trajectory_complete".into(),
                    Value::Bool(t.terminal_trajectory_complete),
                );
                t.receipt
            }
            Self::Oracle(o) => {
                out.insert("command".into(), Value::String(o.command.clone()));
                out.insert("return_code".into(), json!(o.return_code));
                out.insert("stdout".into(), Value::String(o.stdout.clone()));
                None
            }
            Self::Nop { note } => {
                out.insert("note".into(), Value::String(note.clone()));
                None
            }
            Self::Other(map) => return Value::Object(map.clone()),
        };
        if let Some(receipt) = receipt {
            out.insert("receipt".into(), receipt.to_json());
            out.insert("ts".into(), iso_from_ns(receipt.first_observed_ns).into());
        }
        Value::Object(out)
    }

    /// Read one event. Anything this build cannot represent exactly is kept
    /// as [`Event::Other`].
    pub fn from_json(map: Map<String, Value>) -> Self {
        match Self::parse_known(&map) {
            // Only accept the typed form when it writes back the same JSON,
            // key order included.
            Some(event) if same_json(&event.to_json(), &map) => event,
            _ => Self::Other(map),
        }
    }

    fn parse_known(map: &Map<String, Value>) -> Option<Self> {
        let s = |key: &str| map.get(key).and_then(Value::as_str).map(str::to_string);
        let receipt = || match map.get("receipt") {
            Some(r) => Receipt::from_json(r).map(Some),
            None => Some(None),
        };
        let text = || -> Option<Text> {
            Some(Text {
                text: s("text")?,
                parent_tool_call_id: s("parent_tool_call_id"),
                receipt: receipt()?,
            })
        };
        Some(match map.get("type")?.as_str()? {
            "user_message" => Self::UserMessage(text()?),
            "agent_message" => Self::AgentMessage(text()?),
            "agent_thought" => Self::AgentThought(text()?),
            "tool_call" => {
                let receipt = receipt()?;
                let finished_observed_ns = match map.get("finished_at") {
                    // Recover the receipt time the ISO form was made from;
                    // to_json checks it renders back identically.
                    Some(Value::String(iso)) => Some(ns_from_iso(iso)?),
                    Some(_) => return None,
                    None => None,
                };
                Self::ToolCall(Box::new(ToolCall {
                    tool_call_id: s("tool_call_id")?,
                    kind: s("kind")?,
                    title: s("title")?,
                    status: ToolCallStatus::parse(&s("status")?)?,
                    content: map.get("content")?.as_array()?.clone(),
                    raw_input: map.get("raw_input").cloned(),
                    raw_output: map.get("raw_output").cloned(),
                    terminal_output: None,
                    terminal_exit: None,
                    parent_tool_call_id: s("parent_tool_call_id"),
                    tool_name: s("tool_name"),
                    receipt,
                    finished_observed_ns,
                }))
            }
            "notice" => Self::Notice(Notice {
                severity: s("severity")?,
                title: s("title")?,
                description: s("description"),
                receipt: receipt()?,
            }),
            "agent_timeout" => Self::AgentTimeout(AgentTimeout {
                reason: s("reason")?,
                timeout_sec: map.get("timeout_sec")?.as_f64()?,
                pending_tool_call_ids: map
                    .get("pending_tool_call_ids")?
                    .as_array()?
                    .iter()
                    .map(|v| v.as_str().map(str::to_string))
                    .collect::<Option<_>>()?,
                terminal_trajectory_complete: map.get("terminal_trajectory_complete")?.as_bool()?,
                receipt: receipt()?,
            }),
            "oracle" => Self::Oracle(Oracle {
                command: s("command")?,
                return_code: map.get("return_code")?.as_i64()?,
                stdout: s("stdout")?,
            }),
            "nop" => Self::Nop { note: s("note")? },
            _ => return None,
        })
    }
}

/// JSON equality that also compares object key order (`Map`'s `==` does not).
fn same_json(value: &Value, map: &Map<String, Value>) -> bool {
    fn same_map(x: &Map<String, Value>, y: &Map<String, Value>) -> bool {
        x.len() == y.len()
            && x.iter()
                .zip(y)
                .all(|((ka, va), (kb, vb))| ka == kb && same(va, vb))
    }
    fn same(a: &Value, b: &Value) -> bool {
        match (a, b) {
            (Value::Object(x), Value::Object(y)) => same_map(x, y),
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same(a, b))
            }
            _ => a == b,
        }
    }
    matches!(value, Value::Object(own) if same_map(own, map))
}

/// The Unix nanoseconds an `iso_from_ns` string was made from (to the
/// microsecond).
fn ns_from_iso(iso: &str) -> Option<u64> {
    let t = DateTime::parse_from_rfc3339(iso).ok()?;
    let micros = u64::try_from(t.timestamp_micros()).ok()?;
    // Past the year 2554 nanoseconds overflow; such an event stays as written.
    micros.checked_mul(1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_times_match_python_isoformat() {
        assert_eq!(
            iso_from_ns(1_791_545_736_256_466_000),
            "2026-10-09T11:35:36.256466+00:00"
        );
        assert_eq!(
            iso_from_ns(1_791_545_736_000_000_999),
            "2026-10-09T11:35:36+00:00"
        );
        assert_eq!(
            iso_from_ns(1_791_545_736_000_001_000),
            "2026-10-09T11:35:36.000001+00:00"
        );
    }

    #[test]
    fn a_tool_call_writes_0_8_keys_in_0_8_order() {
        let mut tc = ToolCall::new(
            "t1".into(),
            "execute".into(),
            "ls".into(),
            Receipt::at(1_000_000_000),
        );
        tc.raw_input = Some(json!({"command": "ls"}));
        tc.tool_name = Some("Bash".into());
        tc.status = ToolCallStatus::Completed;
        tc.finished_observed_ns = Some(2_000_000_000);
        let keys: Vec<String> = Event::ToolCall(Box::new(tc))
            .to_json()
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect();
        assert_eq!(
            keys,
            [
                "type",
                "tool_call_id",
                "kind",
                "title",
                "status",
                "content",
                "raw_input",
                "tool_name",
                "receipt",
                "started_at",
                "finished_at"
            ]
        );
    }

    #[test]
    fn codex_terminal_chunks_rebuild_the_1x_raw_output() {
        // Guards 0.8's codex-acp 2.x fix (#1099 follow-up): output that
        // arrives only as terminal chunks survives with its exit code.
        let mut tc = ToolCall::new("c".into(), "execute".into(), "x".into(), Receipt::at(1));
        tc.terminal_output = Some("hello\nworld\n".into());
        tc.terminal_exit = Some(json!({"exit_code": 3, "signal": null}));
        assert_eq!(
            tc.raw_output(),
            Some(json!({"formatted_output": "hello\nworld\n", "exit_code": 3}))
        );
        tc.raw_output = Some(json!({"exit_code": 0}));
        tc.terminal_exit = None;
        assert_eq!(
            tc.raw_output(),
            Some(json!({"formatted_output": "hello\nworld\n", "exit_code": 0}))
        );
        tc.raw_output = Some(json!({"formatted_output": "agent's own", "exit_code": 0}));
        assert_eq!(
            tc.raw_output(),
            Some(json!({"formatted_output": "agent's own", "exit_code": 0}))
        );
    }

    #[test]
    fn a_far_future_time_is_kept_as_written_not_a_crash() {
        let far = json!({
            "type": "tool_call", "tool_call_id": "x", "kind": "read", "title": "t", "status": "completed",
            "content": [], "finished_at": "3000-01-01T00:00:00+00:00",
        });
        let Value::Object(map) = far.clone() else {
            unreachable!()
        };
        let event = Event::from_json(map);
        assert!(matches!(event, Event::Other(_)));
        assert_eq!(event.to_json(), far);
    }

    #[test]
    fn unknown_or_odd_events_are_kept_as_written() {
        let odd = json!({"type": "tool_call", "tool_call_id": "x", "kind": "read", "title": "t", "status": "weird", "content": []});
        let Value::Object(map) = odd.clone() else {
            unreachable!()
        };
        assert!(matches!(Event::from_json(map), Event::Other(_)));
        let extra = json!({"type": "agent_message", "text": "hi", "mystery": 1});
        let Value::Object(map) = extra.clone() else {
            unreachable!()
        };
        let event = Event::from_json(map);
        assert!(matches!(event, Event::Other(_)));
        assert_eq!(event.to_json(), extra);
        assert_eq!(event.kind(), "agent_message");
    }
}
