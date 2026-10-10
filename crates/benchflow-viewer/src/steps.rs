//! Trajectory events become the steps the page renders (0.8's
//! `_normalize_steps` and the step models of `models.py`).
//!
//! `trajectory/acp_trajectory.jsonl` holds one ACP event per line. Each
//! event becomes exactly one step, numbered in capture order; events a
//! subagent emitted are then nested under the tool call that spawned it
//! (see [`crate::subagents`]).

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use crate::py;
use crate::pyjson::{self, Map, Value, get, object};
use crate::subagents;
use crate::timestamp::parse_ts;
use crate::values::{display_text, finite_float, optional_bool, optional_text};

/// One entry of a payload's `steps` list.
#[derive(Debug, Clone)]
pub enum Step {
    Event(Event),
    /// Subagent events with no spawning tool card to sit under.
    Group(Group),
}

/// A step made from one captured event.
#[derive(Debug, Clone)]
pub struct Event {
    /// Capture order, from 1.
    pub i: usize,
    /// Epoch seconds, when the capture recorded a time.
    pub t: Option<f64>,
    pub body: Body,
}

#[derive(Debug, Clone)]
pub enum Body {
    Prompt {
        label: String,
        text: String,
    },
    Message {
        text: String,
    },
    Thought {
        text: String,
    },
    Tool(Box<ToolStep>),
    Timeout(TimeoutInfo),
    /// What the ACP adapter itself reported (a warning, a model fallback).
    Notice {
        severity: String,
        title: String,
        description: Option<String>,
    },
    /// An event type this viewer does not know, shown as its JSON.
    Unknown {
        event_type: String,
        text: String,
    },
}

#[derive(Debug, Clone)]
pub struct ToolStep {
    pub call: ToolCall,
    /// Seconds from start to finish, when both were recorded.
    pub dur: Option<f64>,
    /// The subagent this call spawned, with its events.
    pub subagent: Option<SubagentTrace>,
}

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub status: &'static str,
    pub content: Vec<String>,
    /// The color family the page uses for this call.
    pub hue: &'static str,
}

#[derive(Debug, Clone)]
pub struct TimeoutInfo {
    pub reason: String,
    pub timeout_sec: Option<f64>,
    pub pending: Vec<String>,
    pub complete: Option<bool>,
}

/// Events a subagent emitted, attributed through `parent_tool_call_id`.
#[derive(Debug, Clone)]
pub struct SubagentTrace {
    pub parent_tool_call_id: String,
    /// Levels below the main agent (1 = spawned by the main agent).
    pub depth: usize,
    pub steps: Vec<Step>,
    pub subagent_type: Option<String>,
    pub description: Option<String>,
}

/// A labelled group of subagent events whose spawning call is not shown.
#[derive(Debug, Clone)]
pub struct Group {
    pub gid: String,
    pub reason: GroupReason,
    pub subagent: SubagentTrace,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupReason {
    /// The spawning tool call was never captured.
    ParentNotCaptured,
    /// The spawning call is more than the depth cap below the main agent.
    TooDeep,
    /// The attribution loops back on itself.
    Cyclic,
}

impl GroupReason {
    fn as_str(self) -> &'static str {
        match self {
            GroupReason::ParentNotCaptured => "parent_not_captured",
            GroupReason::TooDeep => "too_deep",
            GroupReason::Cyclic => "cyclic",
        }
    }
}

// ── wire shape ──────────────────────────────────────────────────────────

impl Step {
    pub fn into_value(self) -> Value {
        match self {
            Step::Event(event) => event.into_value(),
            Step::Group(group) => object([
                ("kind", "subagent".into()),
                ("gid", group.gid.into()),
                ("reason", group.reason.as_str().into()),
                ("subagent", group.subagent.into_value()),
            ]),
        }
    }
}

impl Event {
    fn into_value(self) -> Value {
        let mut map = Map::new();
        map.insert("i".into(), self.i.into());
        map.insert("kind".into(), self.body.kind().into());
        if let Some(t) = self.t {
            map.insert("t".into(), t.into());
        }
        match self.body {
            Body::Prompt { label, text } => {
                map.insert("label".into(), label.into());
                map.insert("text".into(), text.into());
            }
            Body::Message { text } | Body::Thought { text } => {
                map.insert("text".into(), text.into());
            }
            Body::Tool(tool) => {
                let ToolStep {
                    call,
                    dur,
                    subagent,
                } = *tool;
                map.insert(
                    "tool".into(),
                    object([
                        ("id", call.id.into()),
                        ("kind", call.kind.into()),
                        ("title", call.title.into()),
                        ("status", call.status.into()),
                        ("content", call.content.into()),
                        ("hue", call.hue.into()),
                    ]),
                );
                if let Some(dur) = dur {
                    map.insert("dur".into(), dur.into());
                }
                if let Some(subagent) = subagent {
                    map.insert("subagent".into(), subagent.into_value());
                }
            }
            Body::Timeout(timeout) => {
                map.insert(
                    "timeout".into(),
                    object([
                        ("reason", timeout.reason.into()),
                        ("timeout_sec", timeout.timeout_sec.into()),
                        ("pending", timeout.pending.into()),
                        ("complete", timeout.complete.into()),
                    ]),
                );
            }
            Body::Notice {
                severity,
                title,
                description,
            } => {
                let text = match description.as_deref() {
                    Some(description) if !description.is_empty() => {
                        format!("{title}\n\n{description}")
                    }
                    _ => title.clone(),
                };
                map.insert("severity".into(), severity.into());
                map.insert("title".into(), title.into());
                map.insert("text".into(), text.into());
            }
            Body::Unknown { event_type, text } => {
                map.insert("type".into(), event_type.into());
                map.insert("text".into(), text.into());
            }
        }
        Value::Object(map)
    }
}

impl Body {
    pub fn kind(&self) -> &'static str {
        match self {
            Body::Prompt { .. } => "prompt",
            Body::Message { .. } => "message",
            Body::Thought { .. } => "thought",
            Body::Tool(_) => "tool",
            Body::Timeout(_) => "timeout",
            Body::Notice { .. } => "notice",
            Body::Unknown { .. } => "unknown",
        }
    }
}

impl SubagentTrace {
    fn into_value(self) -> Value {
        object([
            ("parent_tool_call_id", self.parent_tool_call_id.into()),
            ("depth", self.depth.into()),
            ("subagent_type", self.subagent_type.into()),
            ("description", self.description.into()),
            (
                "steps",
                Value::Array(self.steps.into_iter().map(Step::into_value).collect()),
            ),
        ])
    }
}

/// Step counts for the header and the Metrics tab (`StepCounts`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StepCounts {
    pub prompts: usize,
    pub messages: usize,
    pub thoughts: usize,
    pub tools: usize,
    /// Subagent traces, and the events inside them (also in the counts above).
    pub subagents: usize,
    pub subagent_events: usize,
}

impl StepCounts {
    /// Count every event, nested subagent events included.
    pub fn of(steps: &[Step]) -> StepCounts {
        let mut counts = StepCounts::default();
        let root_events = steps
            .iter()
            .filter(|step| matches!(step, Step::Event(_)))
            .count();
        let mut total_events = 0;
        let mut pending: Vec<&[Step]> = vec![steps];
        while let Some(list) = pending.pop() {
            for step in list {
                match step {
                    Step::Group(group) => {
                        counts.subagents += 1;
                        pending.push(&group.subagent.steps);
                    }
                    Step::Event(event) => {
                        total_events += 1;
                        match &event.body {
                            Body::Prompt { .. } => counts.prompts += 1,
                            Body::Message { .. } => counts.messages += 1,
                            Body::Thought { .. } => counts.thoughts += 1,
                            Body::Tool(tool) => {
                                counts.tools += 1;
                                if let Some(subagent) = &tool.subagent {
                                    counts.subagents += 1;
                                    pending.push(&subagent.steps);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        counts.subagent_events = total_events - root_events;
        counts
    }

    pub fn into_value(self) -> Value {
        let mut map = Map::new();
        map.insert("prompts".into(), self.prompts.into());
        map.insert("messages".into(), self.messages.into());
        map.insert("thoughts".into(), self.thoughts.into());
        map.insert("tools".into(), self.tools.into());
        // Omitted when there are none, keeping the common wire shape.
        if self.subagents > 0 {
            map.insert("subagents".into(), self.subagents.into());
            map.insert("subagent_events".into(), self.subagent_events.into());
        }
        Value::Object(map)
    }
}

// ── reading events ──────────────────────────────────────────────────────

/// Where Python's `str.splitlines()` breaks a line (after text mode has
/// turned `\r\n` and `\r` into `\n`).
fn is_line_break(ch: char) -> bool {
    matches!(
        ch,
        '\n' | '\r'
            | '\u{b}'
            | '\u{c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// The events of a trajectory file, read one line at a time so a large
/// file is never held in memory whole. Each non-blank line that parses to
/// a JSON object is an event; anything else is skipped (`_parse_jsonl`).
/// A read error ends the iteration with that error.
pub fn read_events(path: &Path) -> io::Result<impl Iterator<Item = io::Result<Map>>> {
    let mut reader = BufReader::with_capacity(256 * 1024, File::open(path)?);
    let mut line = Vec::new();
    let mut ready: std::collections::VecDeque<Map> = Default::default();
    Ok(std::iter::from_fn(move || {
        loop {
            if let Some(event) = ready.pop_front() {
                return Some(Ok(event));
            }
            line.clear();
            match reader.read_until(b'\n', &mut line) {
                Ok(0) => return None,
                Ok(_) => {
                    let text = String::from_utf8_lossy(&line);
                    for piece in text.split(is_line_break) {
                        if py::strip(piece).is_empty() {
                            continue;
                        }
                        if let Some(Value::Object(event)) = pyjson::parse(piece) {
                            ready.push_back(event);
                        }
                    }
                }
                Err(error) => return Some(Err(error)),
            }
        }
    }))
}

/// `prompts.json` as display texts (`_load_prompts`); `None` unless the file
/// holds a list.
pub fn load_prompts(rollout_dir: &Path) -> Option<Vec<String>> {
    match crate::values::load_json(&rollout_dir.join("prompts.json"))? {
        Value::Array(items) => Some(items.iter().map(display_text).collect()),
        _ => None,
    }
}

/// The steps of a trial's trajectory file, with `prompts` shown first when
/// the trajectory records no prompt of its own. A missing or unreadable
/// file has no steps (0.8 reads the whole file or nothing).
pub fn trajectory_steps(path: &Path, prompts: Option<Vec<String>>) -> Vec<Step> {
    match read_events(path) {
        Ok(events) => normalize_steps(events, prompts),
        Err(_) => normalize_steps(std::iter::empty(), prompts),
    }
}

/// What nesting needs to know about the event behind a step.
#[derive(Debug, Default, Clone)]
pub(crate) struct Source {
    /// The `parent_tool_call_id` of an event a subagent emitted.
    pub scope: Option<String>,
    /// A tool step's call id, as Python's `str(tool_call_id or "")`.
    pub call_id: Option<String>,
    /// From a spawning call's input: the subagent's type and task.
    pub subagent_type: Option<String>,
    pub description: Option<String>,
}

/// Events (or a read error) to steps (`_normalize_steps`).
pub fn normalize_steps<I>(events: I, prompts: Option<Vec<String>>) -> Vec<Step>
where
    I: IntoIterator<Item = io::Result<Map>>,
{
    let mut steps = Vec::new();
    let mut sources = Vec::new();
    let mut prompt_count = 0;
    let mut has_inline_prompts = false;
    for event in events {
        let Ok(event) = event else {
            // 0.8 reads the whole file first; a failed read means no events.
            steps.clear();
            sources.clear();
            has_inline_prompts = false;
            break;
        };
        if get_str(&event, "type") == Some("user_message") {
            has_inline_prompts = true;
        }
        let (body, t, source) = normalize_event(event, &mut prompt_count);
        steps.push(Step::Event(Event {
            i: steps.len() + 1,
            t,
            body,
        }));
        sources.push(source);
    }
    // Prompts from prompts.json lead only when the trajectory records none.
    if !has_inline_prompts && let Some(prompts) = prompts.filter(|prompts| !prompts.is_empty()) {
        let count = prompts.len();
        for step in &mut steps {
            if let Step::Event(event) = step {
                event.i += count;
            }
        }
        let prompt_steps = prompts.into_iter().enumerate().map(|(index, text)| {
            Step::Event(Event {
                i: index + 1,
                t: None,
                body: Body::Prompt {
                    label: format!("PROMPT {}", index + 1),
                    text,
                },
            })
        });
        steps.splice(0..0, prompt_steps);
        sources.splice(0..0, std::iter::repeat_n(Source::default(), count));
    }
    subagents::nest(steps, sources)
}

fn get_str<'a>(map: &'a Map, key: &str) -> Option<&'a str> {
    get(map, key).as_str()
}

/// One event to its step body, time, and nesting source.
fn normalize_event(event: Map, prompt_count: &mut usize) -> (Body, Option<f64>, Source) {
    let event_type = optional_text(get(&event, "type")).unwrap_or_default();
    let event_time = parse_ts(get(&event, "ts"));
    let mut source = Source {
        scope: get_str(&event, "parent_tool_call_id")
            .filter(|parent| !parent.is_empty())
            .map(str::to_string),
        ..Source::default()
    };
    let text = || display_text(get(&event, "text"));
    let (body, t) = match event_type.as_str() {
        "user_message" => {
            *prompt_count += 1;
            (
                Body::Prompt {
                    label: format!("PROMPT {prompt_count}"),
                    text: text(),
                },
                event_time,
            )
        }
        "agent_message" => (Body::Message { text: text() }, event_time),
        "agent_thought" => (Body::Thought { text: text() }, event_time),
        "tool_call" => {
            let call_id = get(&event, "tool_call_id");
            if py::truthy(call_id) {
                source.call_id = Some(py::str_of(call_id));
            }
            let raw_input = get(&event, "raw_input");
            source.subagent_type = spawn_text(raw_input, "subagent_type");
            source.description = spawn_text(raw_input, "description");
            let (tool, started) = tool_step(&event);
            (Body::Tool(Box::new(tool)), started)
        }
        "agent_timeout" => {
            let pending = match get(&event, "pending_tool_call_ids") {
                Value::Array(items) => items.iter().map(display_text).collect(),
                _ => Vec::new(),
            };
            let timeout = TimeoutInfo {
                reason: display_text(get(&event, "reason")),
                timeout_sec: finite_float(get(&event, "timeout_sec")),
                pending,
                complete: optional_bool(get(&event, "terminal_trajectory_complete")),
            };
            (Body::Timeout(timeout), event_time)
        }
        "notice" => (
            Body::Notice {
                severity: optional_text(get(&event, "severity"))
                    .filter(|severity| !severity.is_empty())
                    .unwrap_or_else(|| "info".to_string()),
                title: display_text(get(&event, "title")),
                description: optional_text(get(&event, "description")),
            },
            event_time,
        ),
        _ => {
            let text = pyjson::dumps(&Value::Object(event));
            (Body::Unknown { event_type, text }, event_time)
        }
    };
    (body, t, source)
}

/// A non-empty string from a spawning call's input (`_spawn_text`).
fn spawn_text(raw_input: &Value, key: &str) -> Option<String> {
    raw_input
        .get(key)
        .as_str()
        .filter(|text| !text.is_empty())
        .map(str::to_string)
}

/// A tool call event to its step and start time.
fn tool_step(event: &Map) -> (ToolStep, Option<f64>) {
    let raw_input = get(event, "raw_input");
    let title = display_text(get(event, "title"));
    let kind = display_text(get(event, "kind"));
    let kind = tool_kind(
        if kind.is_empty() { "other" } else { &kind },
        &title,
        raw_input,
    );
    let title = final_tool_title(&kind, title, raw_input, get(event, "content"));
    let started = parse_ts(get(event, "started_at")).or_else(|| parse_ts(get(event, "ts")));
    let finished = parse_ts(get(event, "finished_at"));
    let dur = match (started, finished) {
        (Some(started), Some(finished)) if finished >= started => Some(finished - started),
        _ => None,
    };
    let mut content = tool_content_texts(get(event, "content"));
    if content.is_empty() {
        content = raw_io_texts(&kind, raw_input, get(event, "raw_output"));
    }
    let hue = tool_hue(&kind, &title);
    let call = ToolCall {
        id: display_text(get(event, "tool_call_id")),
        status: tool_status(get(event, "status")),
        kind,
        title,
        content,
        hue,
    };
    (
        ToolStep {
            call,
            dur,
            subagent: None,
        },
        started,
    )
}

/// The kind a tool badge shows (`tool_kind`): `agent` for a call that spawns
/// a subagent, which harnesses report with the ACP kind `think`.
fn tool_kind(kind: &str, title: &str, raw_input: &Value) -> String {
    if py::strip(kind).to_lowercase() != "think" {
        return kind.to_string();
    }
    let named_spawn = py::first_word(title)
        .is_some_and(|word| matches!(word.to_lowercase().as_str(), "task" | "agent"));
    let task_input = raw_input.as_object().is_some_and(|input| {
        input.contains_key("prompt")
            && (input.contains_key("description") || input.contains_key("subagent_type"))
    });
    if named_spawn || task_input {
        "agent".to_string()
    } else {
        kind.to_string()
    }
}

/// The color family of a tool call (`tool_hue`).
pub(crate) fn tool_hue(kind: &str, title: &str) -> &'static str {
    const KNOWN: [&str; 7] = [
        "read", "edit", "execute", "fetch", "search", "think", "skill",
    ];
    const INFER: [(&str, &[&str]); 5] = [
        (
            "search",
            &["web", "search", "fetch", "grep", "glob", "browser"],
        ),
        ("execute", &["bash", "shell", "exec", "terminal", "command"]),
        (
            "edit",
            &["write", "edit", "patch", "delete", "move", "notebook"],
        ),
        ("read", &["read", "cat", "view", "ls", "list"]),
        ("skill", &["agent", "task", "skill", "oracle"]),
    ];
    let normalized = py::strip(kind).to_lowercase();
    if let Some(known) = KNOWN.into_iter().find(|known| *known == normalized) {
        return known;
    }
    if normalized == "agent" {
        return "skill";
    }
    let haystack = format!("{kind} {title}").to_lowercase();
    INFER
        .into_iter()
        .find(|(_, needles)| needles.iter().any(|needle| haystack.contains(needle)))
        .map_or("other", |(hue, _)| hue)
}

/// A tool status from the closed set the page styles (`normalize_tool_status`).
fn tool_status(value: &Value) -> &'static str {
    const STATUSES: [&str; 5] = ["pending", "in_progress", "completed", "failed", "cancelled"];
    let Some(text) = value.as_str() else {
        return "unknown";
    };
    let normalized = py::strip(text).to_lowercase().replace(['-', ' '], "_");
    STATUSES
        .into_iter()
        .find(|status| *status == normalized)
        .unwrap_or("unknown")
}

/// A verifier test status from the closed set the page styles.
pub(crate) fn verifier_status(value: &Value) -> &'static str {
    const STATUSES: [&str; 4] = ["passed", "failed", "skipped", "pending"];
    let Some(text) = value.as_str() else {
        return "unknown";
    };
    let normalized = py::strip(text).to_lowercase();
    STATUSES
        .into_iter()
        .find(|status| *status == normalized)
        .unwrap_or("unknown")
}

/// Texts of ACP tool content blocks (`_tool_content_texts`): text blocks,
/// nested text content, and diffs; terminal references carry no text.
fn tool_content_texts(content: &Value) -> Vec<String> {
    let Some(items) = content.as_array() else {
        return Vec::new();
    };
    let mut texts = Vec::new();
    for item in items {
        let text = match item {
            Value::Object(block) => {
                if get_str(block, "type") == Some("terminal") {
                    continue;
                }
                let inner = get(block, "content");
                if inner
                    .as_object()
                    .is_some_and(|inner| inner.contains_key("text"))
                {
                    display_text(inner.get("text"))
                } else if block.contains_key("text") {
                    display_text(get(block, "text"))
                } else if get_str(block, "type") == Some("diff") {
                    format!(
                        "diff {}\n--- old\n{}\n+++ new\n{}",
                        display_text(get(block, "path")),
                        display_text(get(block, "oldText")),
                        display_text(get(block, "newText"))
                    )
                } else {
                    pyjson::dumps(item)
                }
            }
            other => display_text(other),
        };
        if !text.is_empty() {
            texts.push(text);
        }
    }
    texts
}

/// `rawInput` / `rawOutput` for a call without content blocks
/// (`_raw_io_texts`): a command line and its output, else JSON.
fn raw_io_texts(kind: &str, raw_input: &Value, raw_output: &Value) -> Vec<String> {
    let mut texts = Vec::new();
    match raw_input {
        Value::Object(input) if kind == "execute" && input.contains_key("command") => {
            let command = get(input, "command");
            texts.push(match command {
                Value::Array(parts) => parts.iter().map(py::str_of).collect::<Vec<_>>().join(" "),
                other => display_text(other),
            });
        }
        Value::Null => {}
        other => texts.push(pyjson::dumps(other)),
    }
    match raw_output {
        Value::Object(output) if output.contains_key("formatted_output") => {
            let mut text = display_text(get(output, "formatted_output"));
            let exit_code = get(output, "exit_code");
            // `exit_code not in (None, 0)`: 0, 0.0 and false all equal 0.
            let is_zero = match exit_code {
                Value::Null => true,
                Value::Int(int) => int.is(0),
                Value::Float(number) => *number == 0.0,
                Value::Bool(flag) => !flag,
                _ => false,
            };
            if !is_zero {
                text = format!(
                    "{}\n[exit code {}]",
                    py::rstrip(&text),
                    py::str_of(exit_code)
                );
            }
            texts.push(text);
        }
        Value::Null => {}
        other => texts.push(pyjson::dumps(other)),
    }
    texts.retain(|text| !text.is_empty());
    texts
}

/// The adapter's final title for a provisional one (`_final_tool_title`).
///
/// claude-agent-acp titles a call before its input arrives and sends the
/// real title later; captures from before that was recorded kept the first
/// title, so it is rebuilt from the recorded path.
fn final_tool_title(kind: &str, title: String, raw_input: &Value, content: &Value) -> String {
    let verb = match (kind, py::strip(&title)) {
        ("edit", "Preparing file…" | "Write") => "Write",
        ("edit", "Edit") => "Edit",
        ("read", "Read File") => "Read",
        _ => return title,
    };
    let path = raw_input
        .get("file_path")
        .as_str()
        .filter(|path| !path.is_empty())
        .map(str::to_string)
        .or_else(|| single_diff_path(content));
    match path {
        Some(path) => format!("{verb} {path}"),
        None => title,
    }
}

/// The path every diff block names, when they all name the same non-empty one.
fn single_diff_path(content: &Value) -> Option<String> {
    let mut paths = content
        .as_array()?
        .iter()
        .filter(|item| {
            item.as_object()
                .is_some_and(|block| get_str(block, "type") == Some("diff"))
        })
        .map(|item| item.get("path"));
    let first = paths.next()?;
    let first = first.as_str().filter(|path| !path.is_empty())?;
    paths
        .all(|path| path.as_str() == Some(first))
        .then(|| first.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn events(lines: &[&str]) -> Vec<io::Result<Map>> {
        lines
            .iter()
            .map(|line| Ok(pyjson::parse(line).unwrap().as_object().unwrap().clone()))
            .collect()
    }

    fn wire(steps: Vec<Step>) -> Vec<String> {
        steps
            .into_iter()
            .map(|step| pyjson::dumps(&step.into_value()))
            .collect()
    }

    #[test]
    fn every_event_type_normalizes() {
        let steps = normalize_steps(
            events(&[
                r#"{"type": "user_message", "text": "do it"}"#,
                r#"{"type": "agent_thought", "text": "hmm"}"#,
                r#"{"type": "tool_call", "tool_call_id": "c1", "kind": "execute", "title": "ls", "status": "completed", "content": [{"type": "content", "content": {"type": "text", "text": "out"}}]}"#,
                r#"{"type": "agent_message", "text": "done"}"#,
                r#"{"type": "agent_timeout", "reason": "wall_clock_timeout", "timeout_sec": 5.0, "pending_tool_call_ids": ["c1"], "terminal_trajectory_complete": false}"#,
                r#"{"type": "notice", "title": "Heads up", "description": "detail"}"#,
                r#"{"type": "future_thing", "x": 1}"#,
            ]),
            None,
        );
        assert_eq!(
            wire(steps),
            [
                r#"{"i": 1, "kind": "prompt", "label": "PROMPT 1", "text": "do it"}"#,
                r#"{"i": 2, "kind": "thought", "text": "hmm"}"#,
                r#"{"i": 3, "kind": "tool", "tool": {"id": "c1", "kind": "execute", "title": "ls", "status": "completed", "content": ["out"], "hue": "execute"}}"#,
                r#"{"i": 4, "kind": "message", "text": "done"}"#,
                r#"{"i": 5, "kind": "timeout", "timeout": {"reason": "wall_clock_timeout", "timeout_sec": 5.0, "pending": ["c1"], "complete": false}}"#,
                r#"{"i": 6, "kind": "notice", "severity": "info", "title": "Heads up", "text": "Heads up\n\ndetail"}"#,
                r#"{"i": 7, "kind": "unknown", "type": "future_thing", "text": "{\"type\": \"future_thing\", \"x\": 1}"}"#,
            ]
        );
    }

    #[test]
    fn prompts_lead_only_when_the_trajectory_has_none() {
        let prompts = Some(vec!["from file".to_string()]);
        let steps = normalize_steps(
            events(&[r#"{"type": "agent_message", "text": "hi"}"#]),
            prompts.clone(),
        );
        assert_eq!(
            wire(steps),
            [
                r#"{"i": 1, "kind": "prompt", "label": "PROMPT 1", "text": "from file"}"#,
                r#"{"i": 2, "kind": "message", "text": "hi"}"#,
            ]
        );
        let steps = normalize_steps(
            events(&[r#"{"type": "user_message", "text": "inline"}"#]),
            prompts,
        );
        assert_eq!(
            wire(steps),
            [r#"{"i": 1, "kind": "prompt", "label": "PROMPT 1", "text": "inline"}"#]
        );
    }

    #[test]
    fn a_read_error_drops_every_event() {
        let mut items = events(&[r#"{"type": "agent_message", "text": "hi"}"#]);
        items.push(Err(io::Error::other("disk")));
        assert!(normalize_steps(items, None).is_empty());
    }

    #[test]
    fn spawning_calls_read_agent_and_hues_classify() {
        assert_eq!(tool_kind("think", "Task", &Value::Null), "agent");
        assert_eq!(
            tool_kind("think", "Plan the next step", &Value::Null),
            "think"
        );
        assert_eq!(tool_hue("other", "ToolSearch"), "search");
        assert_eq!(tool_hue("delete", "remove temp dir"), "edit");
        assert_eq!(tool_hue("agent", "x"), "skill");
        assert_eq!(tool_hue("mystery", "??"), "other");
        assert_eq!(
            tool_status(&Value::Str("IN-PROGRESS".into())),
            "in_progress"
        );
        assert_eq!(tool_status(&Value::Str("\"><script>".into())), "unknown");
    }

    #[test]
    fn line_breaks_follow_python_splitlines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.jsonl");
        std::fs::write(
            &path,
            "{\"a\": 1}\r\n{\"b\": 2}\u{b}{\"c\": 3}\u{85}{\"d\": \"x\u{2028}y\"}\n\u{feff}{\"e\": 4}\n[1]\n",
        )
        .unwrap();
        let keys: Vec<String> = read_events(&path)
            .unwrap()
            .map(|event| event.unwrap().keys().next().unwrap().clone())
            .collect();
        assert_eq!(keys, ["a", "b", "c"]);
    }
}
