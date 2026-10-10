//! Which tool calls are skill invocations (0.8's `trajectories/metrics.py`).
//!
//! `kind == "skill"` is the canonical form. A tool kind, tool name or title
//! naming `invoke_skill`/`activate_skill` counts outright. Harnesses that do
//! not set the kind are matched on their own result shapes, and only for
//! unclassified kinds (`other`, `tool`, none), so a `read` or `execute` call
//! whose output quotes a skill is never reclassified:
//!
//! - OpenHands legacy: a `Tool: invoke_skill` header with a `[skill:` marker;
//! - opencode: a `## Skill: <name>` header (1.17) or `<skill_content name="...">` (1.18);
//! - claude-agent-acp: a `Launching skill: <name>` line.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use super::event::Event;

const SKILL_NAMES: [&str; 3] = ["invoke_skill", "activate_skill", "skill"];
/// A title is display text, so only the explicit spellings prove a skill.
const EXPLICIT_SKILL_NAMES: [&str; 2] = ["invoke_skill", "activate_skill"];
/// Only these kinds are eligible for title and content sniffing.
const SNIFFABLE_KINDS: [&str; 3] = ["", "other", "tool"];

static RESULT_HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)\A\s*Tool:\s*(?:invoke_skill|activate_skill)\b").expect("valid regex")
});
static CONTENT_ENVELOPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)\A\s*(?:<skill_content\s+name\s*=\s*"[^"]+"|#{1,6}\s*Skill:\s*\S|Launching\s+skill:\s*\S)"#)
        .expect("valid regex")
});

fn normalized(name: &str) -> String {
    name.trim().to_lowercase().replace('-', "_")
}

/// True when a tool call with these fields is a skill invocation.
pub(crate) fn is_skill_call(
    kind: &str,
    title: &str,
    tool_name: Option<&str>,
    content: &[Value],
) -> bool {
    let kind = normalized(kind);
    if SKILL_NAMES.contains(&kind.as_str()) {
        return true;
    }
    if tool_name.is_some_and(|n| SKILL_NAMES.contains(&normalized(n).as_str())) {
        return true;
    }
    let title = normalized(title);
    if EXPLICIT_SKILL_NAMES.contains(&title.as_str()) {
        return true;
    }
    if !SNIFFABLE_KINDS.contains(&kind.as_str()) {
        return false;
    }
    if SKILL_NAMES.contains(&title.as_str()) {
        return true;
    }
    content_is_skill_result(content)
}

/// The kind a new tool call is recorded with: `skill` for a skill
/// invocation, else the agent's kind (`other` when it sent none).
pub(crate) fn canonical_kind(kind: Option<&str>, title: &str) -> String {
    let raw = kind.filter(|k| !k.is_empty()).unwrap_or("other");
    if is_skill_call(kind.unwrap_or(""), title, None, &[]) {
        "skill".to_string()
    } else {
        raw.to_string()
    }
}

fn content_is_skill_result(content: &[Value]) -> bool {
    content.iter().filter_map(result_text).any(|text| {
        (RESULT_HEADER.is_match(text) && text.to_lowercase().contains("[skill:"))
            || CONTENT_ENVELOPE.is_match(text)
    })
}

/// The text of a tool-result block: `{"type": "content", "content": {"type":
/// "text", "text": ...}}`, or an inline `{"type": "text", "text": ...}`.
fn result_text(block: &Value) -> Option<&str> {
    let block = block.as_object()?;
    let is_text = |v: &Map<String, Value>| v.get("type").and_then(Value::as_str) == Some("text");
    if let Some(Value::Object(body)) = block.get("content")
        && is_text(body)
    {
        return body.get("text")?.as_str();
    }
    if is_text(block) {
        return block.get("text")?.as_str();
    }
    None
}

/// True when a trajectory event is a skill invocation.
pub(crate) fn is_skill_event(event: &Event) -> bool {
    match event {
        Event::ToolCall(call) => is_skill_call(
            &call.kind,
            &call.title,
            call.tool_name.as_deref(),
            &call.content,
        ),
        Event::Other(map) if map.get("type").and_then(Value::as_str) == Some("tool_call") => {
            is_skill_map(map)
        }
        _ => false,
    }
}

/// The same check on an event kept as written.
fn is_skill_map(map: &Map<String, Value>) -> bool {
    let s = |key: &str| map.get(key).and_then(Value::as_str).unwrap_or("");
    let nested = |key: &str| {
        map.get(key)
            .and_then(|v| v.get("name"))
            .and_then(Value::as_str)
    };
    let tool_name = [
        "tool_name",
        "toolName",
        "name",
        "function_name",
        "functionName",
    ]
    .iter()
    .map(|k| s(k))
    .find(|n| !normalized(n).is_empty())
    .or_else(|| nested("tool"))
    .or_else(|| nested("function"));
    let content = map
        .get("content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    is_skill_call(s("kind"), s("title"), tool_name, content)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn text_block(text: &str) -> Value {
        json!({"type": "content", "content": {"type": "text", "text": text}})
    }

    #[test]
    fn identity_signals_count_outright() {
        assert!(is_skill_call("skill", "x", None, &[]));
        assert!(is_skill_call("Invoke-Skill", "x", None, &[]));
        assert!(is_skill_call("execute", "x", Some("Skill"), &[]));
        assert!(is_skill_call("read", "activate_skill", None, &[]));
    }

    #[test]
    fn titles_and_content_count_only_for_unclassified_kinds() {
        assert!(is_skill_call("other", "Skill", None, &[]));
        assert!(!is_skill_call("read", "skill", None, &[]));
        let launching = [text_block("Launching skill: pdf")];
        assert!(is_skill_call("other", "Run", None, &launching));
        assert!(!is_skill_call("execute", "cat notes", None, &launching));
        assert!(is_skill_call(
            "tool",
            "t",
            None,
            &[text_block("  ## Skill: xlsx\nbody")]
        ));
        assert!(is_skill_call(
            "",
            "t",
            None,
            &[text_block(r#"<skill_content name="pdf">"#)]
        ));
        assert!(is_skill_call(
            "other",
            "t",
            None,
            &[text_block("Tool: invoke_skill\n[skill: pdf] loaded")]
        ));
        // A header without the marker, or a quote mid-text, is not a skill.
        assert!(!is_skill_call(
            "other",
            "t",
            None,
            &[text_block("Tool: invoke_skill\nnothing")]
        ));
        assert!(!is_skill_call(
            "other",
            "t",
            None,
            &[text_block("he said Launching skill: x")]
        ));
    }

    #[test]
    fn new_calls_get_the_canonical_kind() {
        assert_eq!(canonical_kind(Some("execute"), "Skill"), "execute");
        assert_eq!(canonical_kind(Some("other"), "Skill"), "skill");
        assert_eq!(canonical_kind(None, "x"), "other");
        assert_eq!(canonical_kind(Some(""), "invoke_skill"), "skill");
    }
}
