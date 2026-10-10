//! The parts of task.md's parser that conversion needs: where the
//! instruction ends and which typed blocks follow (task.md's
//! `tools/taskmd.py`, `parse_text`, `_scan`, `block_kind`). Checking a
//! task.md is [`crate::taskmd`]'s job; this module only splits the text, so
//! a converter can verify its own output and read a package to export.

use std::sync::LazyLock;

use regex::Regex;

use super::value::{Table, parse_toml, parse_yaml};

static FENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<indent>[ ]{0,3})(?P<marker>`{3,}|~{3,})[ \t]*(?P<info>[^`]*?)[ \t]*$")
        .unwrap()
});
static RESERVED_INFO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ ]{0,3}(`{3,}|~{3,})[ \t]*(notes|user|rubric|behaviors|(stage|role)[ \t]+\S+|(toml|yaml|yml)[ \t]+task)[ \t]*$")
        .unwrap()
});
static FENCE_RUN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^[ ]{0,3}(`{3,})").unwrap());

/// A typed block after the instruction.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Block {
    /// `config`, `stage`, `role`, `user`, or `notes`.
    pub kind: String,
    /// The block's argument: a stage's or role's name, or the config's
    /// language (`toml`, `yaml`).
    pub arg: Option<String>,
    /// The fence's info string as written.
    pub info: String,
    pub body: String,
}

/// A task.md split into its instruction and blocks.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Doc {
    /// The text before the first block, canary lines included, without the
    /// blank lines before the first block.
    pub instruction_source: String,
    pub blocks: Vec<Block>,
    /// The config: the first config block's, or legacy frontmatter's.
    pub config: Table,
}

/// Map a fence's info string to a block kind and argument.
fn block_kind(info: &str) -> (Option<String>, Option<String>) {
    let words: Vec<&str> = info.split_whitespace().collect();
    let Some(first) = words.first() else {
        return (None, None);
    };
    let head = first.to_lowercase();
    match (words.len(), head.as_str()) {
        (2, "toml" | "yaml" | "yml") if words[1] == "task" => (Some("config".into()), Some(head)),
        (1, "notes" | "user") => (Some(head), None),
        (2, "stage" | "role") => (Some(head), Some(words[1].to_string())),
        (1, "rubric" | "behaviors") => (Some("moved".into()), Some(head)),
        _ => (None, None),
    }
}

/// Legacy frontmatter: `(front, body)` when the text opens with `---`.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    if !text.starts_with("---\n") {
        return (None, text);
    }
    match text[4..].find("\n---\n") {
        Some(i) => (Some(&text[4..4 + i]), &text[4 + i + 5..]),
        None => (None, text),
    }
}

fn closes(line: &str, marker: &str) -> bool {
    let c = marker.chars().next().expect("a fence marker");
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return false;
    }
    let rest = &line[indent..];
    let run = rest.chars().take_while(|x| *x == c).count();
    run >= marker.chars().count()
        && rest[run * c.len_utf8()..]
            .chars()
            .all(|x| x == ' ' || x == '\t')
}

/// Split task.md text as task.md's parser does.
pub(crate) fn parse(text: &str) -> Doc {
    let text = text.replace("\r\n", "\n");
    let (front, body) = split_frontmatter(&text);
    let lines: Vec<&str> = body.split('\n').collect();
    let mut instruction: Vec<&str> = Vec::new();
    let mut blocks: Vec<Block> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let Some(m) = FENCE.captures(line) else {
            if blocks.is_empty() {
                instruction.push(line);
            }
            i += 1;
            continue;
        };
        let marker = m["marker"].to_string();
        let info = m["info"].to_string();
        let close = (i + 1..lines.len())
            .find(|&j| closes(lines[j], &marker))
            .unwrap_or(lines.len());
        match block_kind(&info) {
            (Some(kind), arg) if kind != "moved" => {
                let end = close.min(lines.len());
                blocks.push(Block {
                    kind,
                    arg,
                    info: info.clone(),
                    body: lines[i + 1..end].join("\n"),
                });
            }
            (Some(_), _) => {}
            (None, _) => {
                if blocks.is_empty() {
                    let end = (close + 1).min(lines.len());
                    instruction.extend_from_slice(&lines[i..end]);
                }
            }
        }
        i = close + 1;
    }
    let source = instruction.join("\n").trim_end_matches('\n').to_string();
    let mut sources: Vec<Table> = Vec::new();
    for b in blocks.iter().filter(|b| b.kind == "config") {
        let parsed = if b.arg.as_deref() == Some("toml") {
            parse_toml(&b.body).ok()
        } else {
            parse_yaml(&b.body).ok().and_then(|v| v.as_table().cloned())
        };
        sources.push(parsed.unwrap_or_default());
    }
    if let Some(front) = front {
        sources.push(
            parse_yaml(front)
                .ok()
                .and_then(|v| v.as_table().cloned())
                .unwrap_or_default(),
        );
    }
    Doc {
        instruction_source: source,
        blocks,
        config: sources.into_iter().next().unwrap_or_default(),
    }
}

/// A backtick fence longer than any backtick run at the start of a line in
/// `text`.
pub(crate) fn fence_for(text: &str) -> String {
    let longest = FENCE_RUN
        .captures_iter(text)
        .map(|c| c[1].len())
        .max()
        .unwrap_or(2);
    "`".repeat((longest + 1).max(3))
}

/// A fenced block: `fence info`, the body, the fence.
pub(crate) fn block(info: &str, body: &str) -> String {
    let fence = fence_for(body);
    format!("{fence}{info}\n{}\n{fence}", body.trim_end_matches('\n'))
}

/// Fences in an instruction that task.md's parser would read as blocks.
pub(crate) fn reserved_fences(instruction: &str) -> Vec<String> {
    RESERVED_INFO
        .find_iter(instruction)
        .map(|m| m.as_str().trim().to_string())
        .collect()
}

/// The 1-based line of a code fence nothing closes, if any.
pub(crate) fn open_fence(text: &str) -> Option<usize> {
    let normalized = text.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.split('\n').collect();
    let mut i = 0;
    while i < lines.len() {
        if let Some(m) = FENCE.captures(lines[i]) {
            let marker = m["marker"].to_string();
            match (i + 1..lines.len()).find(|&j| closes(lines[j], &marker)) {
                Some(end) => i = end,
                None => return Some(i + 1),
            }
        }
        i += 1;
    }
    None
}

/// `text` when task.md's parser gives back `instruction` from it (and the
/// config block, when `config`), else why not.
pub(crate) fn verified(
    text: String,
    instruction: &str,
    source: &str,
    config: bool,
) -> Result<String, String> {
    let doc = parse(&text);
    let norm = |s: &str| s.replace("\r\n", "\n").trim_end_matches('\n').to_string();
    if norm(&doc.instruction_source) == norm(instruction)
        && (!config || doc.blocks.iter().any(|b| b.kind == "config"))
    {
        return Ok(text);
    }
    let fences = reserved_fences(instruction);
    let why = if let Some(line) = open_fence(instruction) {
        format!(
            "the code fence at its line {line} is never closed, so the task.md blocks after the instruction, the config included, would become part of it; close the fence"
        )
    } else if !fences.is_empty() {
        format!(
            "it has fences a task.md parser reads as blocks ({}); give them a longer outer fence or another info string",
            fences.join(", ")
        )
    } else {
        "the task.md it becomes does not give it back".to_string()
    };
    Err(format!("{source}: {why}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::value::Val;

    #[test]
    fn splits_instruction_and_blocks() {
        let text = "Do the thing.\n\n```python\nprint(1)\n```\n\n```stage two\nNext.\n```\n\n```toml task\nname = \"a/b\"\n```\n";
        let doc = parse(text);
        assert_eq!(
            doc.instruction_source,
            "Do the thing.\n\n```python\nprint(1)\n```"
        );
        assert_eq!(doc.blocks.len(), 2);
        assert_eq!(doc.blocks[0].kind, "stage");
        assert_eq!(doc.blocks[0].arg.as_deref(), Some("two"));
        assert_eq!(doc.config["name"], Val::Str("a/b".into()));
    }

    #[test]
    fn fences_grow_past_the_body() {
        assert_eq!(fence_for("plain"), "```");
        assert_eq!(fence_for("````\nx\n````"), "`````");
        assert_eq!(block("toml task", "a = 1\n\n"), "```toml task\na = 1\n```");
    }

    #[test]
    fn verification_explains_failures() {
        let open = "Run this:\n\n```bash\necho hi\n";
        let text = format!("{}\n\n```toml task\nname = \"a/b\"\n```\n", open.trim_end());
        let err = verified(text, open, "instruction.md", true).unwrap_err();
        assert!(err.contains("line 3"), "{err}");
        let reserved = "Write:\n\n```toml task\nx = 1\n```";
        let text = format!("{reserved}\n\n```toml task\nname = \"a/b\"\n```\n");
        let err = verified(text, reserved, "instruction.md", true).unwrap_err();
        assert!(err.contains("```toml task"), "{err}");
        let fine = "Fine.";
        assert!(
            verified(
                format!("{fine}\n\n```toml task\nname = \"a/b\"\n```\n"),
                fine,
                "x",
                true
            )
            .is_ok()
        );
    }

    #[test]
    fn frontmatter_counts_as_config() {
        let doc = parse("---\nagent:\n  timeout_sec: 5\n---\nDo it.\n");
        assert_eq!(doc.instruction_source, "Do it.");
        assert!(doc.config.contains_key("agent"));
    }
}
