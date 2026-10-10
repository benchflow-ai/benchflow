//! What a parser hands the runner, and the update shapes it builds.
//!
//! A parser turns one line of an agent's stdout into [`Out`]s. Messages,
//! thoughts and tool calls are written as ACP `session/update` objects, the
//! shape [`Trajectory::apply_update`](crate::trajectory::Trajectory::apply_update)
//! already merges, so chunks join, tool-call updates land on their call, and
//! the trajectory file, live progress and `summarize` work as they do for ACP.

use serde_json::{Map, Value, json};

use crate::trajectory::Usage;

/// Reads one agent's output format.
pub trait Parser: Send {
    /// Read one line of stdout (without its newline). Push what it means to
    /// `out` and return true, or return false for a line that is not part of
    /// the format (it goes to the agent log).
    fn line(&mut self, line: &str, out: &mut Vec<Out>) -> bool;

    /// Stdout ended (or the agent was stopped): flush anything held back.
    fn finish(&mut self, _out: &mut Vec<Out>) {}
}

/// Something a line said.
#[derive(Debug, Clone, PartialEq)]
pub enum Out {
    /// An ACP-shaped `session/update` (a message or thought chunk, a tool
    /// call or its update, a plan, a notice).
    Update(Value),
    /// Token counts for the whole run so far.
    Usage(Usage),
    /// Dollars spent so far, as the agent reports it.
    Cost(f64),
    /// The agent reported an error.
    Failure(Failure),
}

/// An error the agent reported in its output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    /// The agent's own words.
    pub message: String,
    /// True when the agent says the run failed (Codex's `turn.failed`, a
    /// Claude `result` with `is_error`). A non-fatal error (a retry notice)
    /// only explains a failure that shows up some other way.
    pub fatal: bool,
    /// What the agent's structured output says the error is, when it says.
    pub hint: Option<Hint>,
    /// What the user should do, when the agent says (a runner script naming
    /// the setting it lacks).
    pub fix: Option<String>,
}

/// A structured error category an agent reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hint {
    /// The login was rejected (Claude's `authentication_failed`).
    Login,
    /// The subscription or credits are spent (Claude's `billing_error`, a
    /// rejected rate-limit window).
    UsageLimit,
    /// The provider is limiting, overloaded or down.
    Provider,
    /// The login cannot use the model, or the agent cannot reach it.
    Model,
    /// The agent lacks something only the user can give it (a graph, a
    /// factory, an app, an option); the failure's `fix` says what.
    Setup,
}

impl Failure {
    pub fn fatal(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: true,
            hint: None,
            fix: None,
        }
    }

    pub fn notice(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            fatal: false,
            hint: None,
            fix: None,
        }
    }

    pub fn hint(mut self, hint: Option<Hint>) -> Self {
        self.hint = hint;
        self
    }

    pub fn fix(mut self, fix: Option<String>) -> Self {
        self.fix = fix;
        self
    }
}

/// A chunk of the agent's message.
pub fn message(text: &str) -> Out {
    Out::Update(json!({
        "sessionUpdate": "agent_message_chunk",
        "content": {"type": "text", "text": text},
    }))
}

/// A chunk of the agent's reasoning.
pub fn thought(text: &str) -> Out {
    Out::Update(json!({
        "sessionUpdate": "agent_thought_chunk",
        "content": {"type": "text", "text": text},
    }))
}

/// A notice from the agent (`severity`: info, warning, error).
pub fn notice(severity: &str, title: &str) -> Out {
    Out::Update(json!({
        "sessionUpdate": "notice",
        "severity": severity,
        "title": title,
    }))
}

/// A tool call as it starts (or, with a terminal `status`, as it is first
/// seen already finished).
#[derive(Debug, Clone, Default)]
pub struct ToolStart<'a> {
    pub id: &'a str,
    pub title: String,
    /// ACP's kind: read, edit, delete, move, search, execute, think, fetch, other.
    pub kind: &'a str,
    /// `pending`, `in_progress`, `completed` or `failed`.
    pub status: &'a str,
    pub input: Option<Value>,
    /// The agent's own name for the tool.
    pub tool_name: Option<&'a str>,
    /// The subagent call this one runs under.
    pub parent: Option<&'a str>,
}

impl ToolStart<'_> {
    pub fn out(&self) -> Out {
        let mut update = Map::new();
        update.insert("sessionUpdate".into(), json!("tool_call"));
        update.insert("toolCallId".into(), json!(self.id));
        update.insert("title".into(), json!(self.title));
        update.insert("kind".into(), json!(self.kind));
        update.insert(
            "status".into(),
            json!(if self.status.is_empty() {
                "in_progress"
            } else {
                self.status
            }),
        );
        if let Some(input) = &self.input {
            update.insert("rawInput".into(), input.clone());
        }
        attribution(&mut update, self.tool_name, self.parent);
        Out::Update(Value::Object(update))
    }
}

/// A tool call's result (or progress, with a non-terminal status).
#[derive(Debug, Clone, Default)]
pub struct ToolDone<'a> {
    pub id: &'a str,
    /// `completed` or `failed` (or `in_progress` for progress).
    pub status: &'a str,
    /// Text shown as the call's content.
    pub text: Option<String>,
    pub output: Option<Value>,
    pub parent: Option<&'a str>,
}

impl ToolDone<'_> {
    pub fn out(&self) -> Out {
        let mut update = Map::new();
        update.insert("sessionUpdate".into(), json!("tool_call_update"));
        update.insert("toolCallId".into(), json!(self.id));
        update.insert("status".into(), json!(self.status));
        if let Some(text) = self.text.as_deref().filter(|t| !t.is_empty()) {
            update.insert(
                "content".into(),
                json!([{"type": "content", "content": {"type": "text", "text": text}}]),
            );
        }
        if let Some(output) = &self.output {
            update.insert("rawOutput".into(), output.clone());
        }
        attribution(&mut update, None, self.parent);
        Out::Update(Value::Object(update))
    }
}

/// The `_meta.claudeCode` attribution the trajectory reads.
fn attribution(update: &mut Map<String, Value>, tool_name: Option<&str>, parent: Option<&str>) {
    let mut claude = Map::new();
    if let Some(name) = tool_name.filter(|n| !n.is_empty()) {
        claude.insert("toolName".into(), json!(name));
    }
    if let Some(parent) = parent.filter(|p| !p.is_empty()) {
        claude.insert("parentToolUseId".into(), json!(parent));
    }
    if !claude.is_empty() {
        update.insert("_meta".into(), json!({"claudeCode": claude}));
    }
}

/// A JSON object, when `line` is one (leading and trailing space allowed).
pub fn json_object(line: &str) -> Option<Map<String, Value>> {
    let trimmed = line.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    match serde_json::from_str(trimmed) {
        Ok(Value::Object(map)) => Some(map),
        _ => None,
    }
}

/// A string field, or `""`.
pub fn str_of<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

/// A count, from a number or a numeric string.
pub fn count(value: Option<&Value>) -> Option<u64> {
    match value? {
        Value::Number(n) => n
            .as_u64()
            .or_else(|| n.as_f64().filter(|f| *f >= 0.0).map(|f| f as u64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// The text of a tool result's content: a string, or the `text` of each
/// text block in an array, joined by newlines.
pub fn content_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(s) => Some(s.clone()),
        Value::Array(blocks) => {
            let texts: Vec<&str> = blocks
                .iter()
                .filter_map(|b| match b {
                    Value::String(s) => Some(s.as_str()),
                    Value::Object(_) => b.get("text").and_then(Value::as_str),
                    _ => None,
                })
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

/// A one-line title: the first line of `text`, at most 200 characters.
pub fn one_line(text: &str) -> String {
    let first = text.lines().next().unwrap_or("").trim();
    let mut title: String = first.chars().take(200).collect();
    if first.chars().count() > 200 || text.trim().lines().count() > 1 {
        title.push('…');
    }
    title
}

/// The ACP kind for a tool, from the words agents use for their tools.
pub fn kind_for_tool(name: &str) -> &'static str {
    let lower = name.to_ascii_lowercase();
    let n = lower.rsplit("__").next().unwrap_or(&lower);
    match n {
        "bash" | "shell" | "exec" | "exec_command" | "run_shell_command" | "command"
        | "terminal" | "killshell" | "bashoutput" => "execute",
        "read" | "view" | "read_file" | "read_many_files" | "cat" | "notebookread" => "read",
        "write"
        | "edit"
        | "multiedit"
        | "notebookedit"
        | "write_file"
        | "replace"
        | "patch"
        | "apply_patch"
        | "text_editor"
        | "str_replace"
        | "str_replace_based_edit_tool" => "edit",
        "grep"
        | "glob"
        | "ls"
        | "list"
        | "list_directory"
        | "search_file_content"
        | "find"
        | "codesearch" => "search",
        "webfetch" | "web_fetch" | "websearch" | "web_search" | "google_web_search" | "fetch" => {
            "fetch"
        }
        "task" | "agent" | "think" | "todowrite" | "todo_write" | "todoread" => "think",
        "skill" | "invoke_skill" | "activate_skill" => "skill",
        _ => "other",
    }
}

/// A short title for a tool call from its input: the command, path,
/// pattern or URL it names, else the tool's name.
pub fn tool_title(name: &str, input: Option<&Value>) -> String {
    const KEYS: [&str; 10] = [
        "command",
        "cmd",
        "file_path",
        "filePath",
        "path",
        "pattern",
        "url",
        "query",
        "description",
        "skill",
    ];
    let detail = input.and_then(|input| {
        KEYS.iter().find_map(|k| match input.get(*k)? {
            Value::String(s) if !s.trim().is_empty() => Some(s.clone()),
            Value::Array(parts) if !parts.is_empty() => Some(
                parts
                    .iter()
                    .map(|p| p.as_str().map_or_else(|| p.to_string(), str::to_string))
                    .collect::<Vec<_>>()
                    .join(" "),
            ),
            _ => None,
        })
    });
    match detail {
        Some(detail) if kind_for_tool(name) == "execute" => one_line(&detail),
        Some(detail) => format!("{name} {}", one_line(&detail)),
        None => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_titles_name_what_the_call_touches() {
        assert_eq!(
            tool_title("Bash", Some(&json!({"command": "ls -la\npwd"}))),
            "ls -la…"
        );
        assert_eq!(
            tool_title("Read", Some(&json!({"file_path": "/app/a.py"}))),
            "Read /app/a.py"
        );
        assert_eq!(
            tool_title("TodoWrite", Some(&json!({"todos": []}))),
            "TodoWrite"
        );
        assert_eq!(kind_for_tool("developer__shell"), "execute");
        assert_eq!(kind_for_tool("Glob"), "search");
        assert_eq!(kind_for_tool("mystery"), "other");
    }

    #[test]
    fn content_text_reads_strings_and_blocks() {
        assert_eq!(content_text(Some(&json!("x"))).as_deref(), Some("x"));
        assert_eq!(
            content_text(Some(&json!([{"type": "text", "text": "a"}, {"type": "image"}, {"type": "text", "text": "b"}])))
                .as_deref(),
            Some("a\nb")
        );
        assert_eq!(content_text(Some(&json!([]))), None);
        assert_eq!(content_text(None), None);
    }
}
