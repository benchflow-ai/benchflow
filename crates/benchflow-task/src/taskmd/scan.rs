//! Splitting task.md into the instruction and its blocks.
//!
//! The instruction is everything before the first task.md block, kept byte
//! for byte; the agent's view drops leading canary comments and blank lines.
//! A task.md block is a top-level fence whose info string is `notes`,
//! `user`, `stage <name>`, `role <name>`, or `<lang> task` with `<lang>` in
//! `toml`, `yaml`, `yml`. After the first block only blocks may follow.

use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

use super::code;
use super::report::{At, Sink};
use super::text::Text;

static FENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<indent> {0,3})(?P<marker>`{3,}|~{3,})[ \t]*(?P<info>[^`]*?)[ \t]*$").unwrap()
});
static CANARY_LINE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^<!--(?P<text>[^>]*canary[^>]*)-->[ \t]*$").unwrap());

/// Fences that used to be blocks and now live in `verifier/`: refused, so
/// judgment text never leaks into a prompt.
const MOVED_BLOCKS: &[(&str, &str)] = &[
    ("rubric", "verifier/rubric.json"),
    ("behaviors", "verifier/behaviors.json"),
];

/// What a block holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BlockKind {
    /// ` ```toml task `: the config.
    Config,
    /// ` ```stage <name> `: a prompt revealed later in the run.
    Stage,
    /// ` ```role <name> `: a role prompt in a multi-agent task.
    Role,
    /// ` ```user `: a simulated user's persona.
    User,
    /// ` ```notes `: author and reviewer notes.
    Notes,
}

impl BlockKind {
    /// The kind's name, as task.md writes it.
    pub fn as_str(self) -> &'static str {
        match self {
            BlockKind::Config => "config",
            BlockKind::Stage => "stage",
            BlockKind::Role => "role",
            BlockKind::User => "user",
            BlockKind::Notes => "notes",
        }
    }

    /// Whether the block can reach an agent (through the compiled prompt).
    pub fn is_prompt(self) -> bool {
        matches!(self, BlockKind::Stage | BlockKind::Role | BlockKind::User)
    }
}

/// One task.md block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Block {
    pub kind: BlockKind,
    /// The config's language (`toml`, `yaml`, `yml`), or a stage's or role's
    /// name.
    pub arg: Option<String>,
    /// The fence's info string, such as `toml task`.
    pub info: String,
    /// What the fence holds, without the fences.
    pub body: String,
    /// The 1-based line of the opening fence.
    pub line: usize,
    /// Where the body starts in the normalized text.
    #[serde(skip)]
    pub(crate) body_start: usize,
}

impl Block {
    /// What an agent sees when a stage, role, or user block is used: the
    /// body without leading canary comments and blank lines.
    pub fn prompt(&self) -> String {
        strip_canary(&self.body).1
    }
}

/// task.md split into its parts.
#[derive(Debug, Clone)]
pub(crate) struct Scanned {
    /// The text before the first block, canary lines included, without the
    /// blank lines before the first block.
    pub instruction_source: String,
    /// What the agent receives first.
    pub instruction: String,
    /// The leading canary comments, joined with "; ".
    pub canary: Option<String>,
    pub blocks: Vec<Block>,
    /// Legacy YAML frontmatter and its offset in the normalized text.
    pub frontmatter: Option<(String, usize)>,
}

/// The kind and argument of a fence's info string; `None` for an ordinary
/// fence that belongs to the instruction. `Err` names a reserved fence.
fn block_kind(info: &str) -> Option<Result<(BlockKind, Option<String>), &'static str>> {
    let words: Vec<&str> = info.split_whitespace().collect();
    let head = words.first()?.to_lowercase();
    match (words.len(), head.as_str()) {
        (2, "toml" | "yaml" | "yml") if words[1] == "task" => {
            Some(Ok((BlockKind::Config, Some(head))))
        }
        (1, "notes") => Some(Ok((BlockKind::Notes, None))),
        (1, "user") => Some(Ok((BlockKind::User, None))),
        (2, "stage") => Some(Ok((BlockKind::Stage, Some(words[1].to_string())))),
        (2, "role") => Some(Ok((BlockKind::Role, Some(words[1].to_string())))),
        (1, _) => MOVED_BLOCKS
            .iter()
            .find(|(k, _)| *k == head)
            .map(|(k, _)| Err(*k)),
        _ => None,
    }
}

/// Whether `info` (a fence's info string) opens a config block.
pub(crate) fn is_config_info(info: &str) -> bool {
    matches!(block_kind(info), Some(Ok((BlockKind::Config, _))))
}

/// The info string of a line that opens a fence, with the fence's marker.
pub(crate) fn fence(line: &str) -> Option<(char, usize, &str)> {
    let caps = FENCE.captures(line)?;
    let marker = caps.name("marker")?.as_str();
    let ch = marker.chars().next()?;
    Some((
        ch,
        marker.chars().count(),
        caps.name("info").map_or("", |m| m.as_str()),
    ))
}

/// Whether `line` closes a fence opened with `n` of `ch`.
fn closes(line: &str, ch: char, n: usize) -> bool {
    let indent = line.len() - line.trim_start_matches(' ').len();
    if indent > 3 {
        return false;
    }
    let rest = &line[indent..];
    let count = rest.chars().take_while(|c| *c == ch).count();
    count >= n
        && rest[count * ch.len_utf8()..]
            .chars()
            .all(|c| c == ' ' || c == '\t')
}

/// Leading canary comments and leading blank lines: (the comments' texts,
/// the rest).
pub(crate) fn strip_canary(text: &str) -> (Vec<String>, String) {
    let mut lines: Vec<&str> = text.split('\n').collect();
    let mut found = Vec::new();
    let mut start = 0;
    while start < lines.len() {
        let Some(caps) = CANARY_LINE.captures(lines[start].trim()) else {
            break;
        };
        found.push(caps["text"].trim().to_string());
        start += 1;
    }
    while start < lines.len() && lines[start].trim().is_empty() {
        start += 1;
    }
    lines.drain(..start);
    (found, lines.join("\n"))
}

/// Split normalized task.md text into its parts, reporting structure errors.
pub(crate) fn scan(text: &Text, sink: &mut Sink<'_>) -> Scanned {
    let all = text.as_str();
    let (frontmatter, body_offset, first_line) = split_frontmatter(all);
    let body = &all[body_offset..];
    // Each line of the body, with its offset in the normalized text.
    let mut lines: Vec<(usize, &str)> = Vec::new();
    let mut at = body_offset;
    for line in body.split('\n') {
        lines.push((at, line));
        at += line.len() + 1;
    }

    let mut instruction: Vec<&str> = Vec::new();
    let mut blocks: Vec<Block> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let (_, line) = lines[i];
        let line_no = first_line + i;
        let Some((ch, n, info)) = fence(line) else {
            if !blocks.is_empty() && !line.trim().is_empty() {
                sink.error(
                    code::TEXT_AFTER_BLOCKS,
                    At::Line(line_no),
                    "text after the first task.md block: move it into the instruction or a notes block",
                );
            } else if blocks.is_empty() {
                instruction.push(line);
            }
            i += 1;
            continue;
        };
        let close = match (i + 1..lines.len()).find(|&j| closes(lines[j].1, ch, n)) {
            Some(j) => j,
            None => {
                sink.error(
                    code::UNTERMINATED_FENCE,
                    At::Line(line_no),
                    "unterminated fenced block",
                );
                lines.len()
            }
        };
        match block_kind(info) {
            Some(Err(moved)) => {
                let to = MOVED_BLOCKS
                    .iter()
                    .find(|(k, _)| *k == moved)
                    .map_or("", |(_, to)| to);
                sink.error(
                    code::RESERVED_FENCE,
                    At::Line(line_no),
                    format!("```{moved} blocks are no longer part of task.md: move them to {to}"),
                );
            }
            None => {
                if blocks.is_empty() {
                    instruction.extend(
                        lines[i..(close + 1).min(lines.len())]
                            .iter()
                            .map(|(_, l)| *l),
                    );
                } else {
                    sink.error(
                        code::FENCE_AFTER_BLOCKS,
                        At::Line(line_no),
                        format!("fenced block ```{info} after the first task.md block: move it into the instruction"),
                    );
                }
            }
            Some(Ok((kind, arg))) => {
                let inner = &lines[(i + 1).min(lines.len())..close];
                let body_start = inner.first().map_or_else(
                    || lines.get(i + 1).map_or(all.len(), |(o, _)| *o),
                    |(o, _)| *o,
                );
                blocks.push(Block {
                    kind,
                    arg,
                    info: info.to_string(),
                    body: inner.iter().map(|(_, l)| *l).collect::<Vec<_>>().join("\n"),
                    line: line_no,
                    body_start,
                });
            }
        }
        i = close + 1;
    }

    let instruction_source = instruction.join("\n").trim_end_matches('\n').to_string();
    let (canaries, agent_view) = strip_canary(&instruction_source);
    let canary = (!canaries.is_empty()).then(|| canaries.join("; "));
    if agent_view.trim().is_empty() {
        sink.error(
            code::EMPTY_INSTRUCTION,
            At::Line(1),
            "the instruction is empty: task.md must start with the task itself",
        );
    }

    let mut seen: Vec<(BlockKind, Option<&str>)> = Vec::new();
    for b in &blocks {
        let named = matches!(b.kind, BlockKind::Stage | BlockKind::Role);
        let key = (b.kind, if named { b.arg.as_deref() } else { None });
        if seen.contains(&key) {
            sink.error(
                code::DUPLICATE_BLOCK,
                At::Line(b.line),
                format!("duplicate ```{} block", b.info),
            );
        }
        seen.push(key);
    }

    Scanned {
        instruction_source,
        instruction: agent_view,
        canary,
        blocks,
        frontmatter: frontmatter.map(|f| (f.to_string(), 4)),
    }
}

/// Legacy files open with YAML frontmatter: (the frontmatter, where the body
/// starts, the body's first line).
fn split_frontmatter(text: &str) -> (Option<&str>, usize, usize) {
    if !text.starts_with("---\n") {
        return (None, 0, 1);
    }
    match text[4..].find("\n---\n") {
        Some(rel) => {
            let end = rel + 4;
            let line = text[..end + 5].matches('\n').count() + 1;
            (Some(&text[4..end]), end + 5, line)
        }
        None => (None, 0, 1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(src: &str) -> (Scanned, Vec<String>) {
        let text = Text::new(src);
        let mut sink = Sink::new(None);
        let s = scan(&text, &mut sink);
        (
            s,
            sink.found
                .into_iter()
                .map(|f| format!("{} {}", f.diag.code, f.diag.message))
                .collect(),
        )
    }

    #[test]
    fn instruction_then_blocks() {
        let (s, d) = run("Do it.\n\n```notes\nn\n```\n\n```toml task\nname = \"a/b\"\n```\n");
        assert!(d.is_empty(), "{d:?}");
        assert_eq!(s.instruction, "Do it.");
        assert_eq!(s.blocks.len(), 2);
        assert_eq!(s.blocks[1].kind, BlockKind::Config);
        assert_eq!(s.blocks[1].arg.as_deref(), Some("toml"));
        assert_eq!(s.blocks[1].body, "name = \"a/b\"");
        assert_eq!(s.blocks[1].line, 7);
    }

    #[test]
    fn ordinary_fences_belong_to_the_instruction() {
        let (s, d) = run("Run:\n\n```bash\nls\n```\n\n```toml task\nx = 1\n```\n");
        assert!(d.is_empty());
        assert_eq!(s.instruction, "Run:\n\n```bash\nls\n```");
    }

    #[test]
    fn canary_comments_are_stripped() {
        let (s, _) = run("<!-- Canary abc -->\n<!-- task.md canary x -->\n\nDo it.\n");
        assert_eq!(s.canary.as_deref(), Some("Canary abc; task.md canary x"));
        assert_eq!(s.instruction, "Do it.");
        assert!(s.instruction_source.starts_with("<!--"));
    }

    #[test]
    fn structure_errors() {
        let (_, d) = run(
            "Do it.\n\n```notes\nn\n```\nstray\n```bash\nx\n```\n```rubric\n{}\n```\n```notes\nm\n```\n```stage a\n",
        );
        let codes: Vec<&str> = d.iter().map(|x| &x[..4]).collect();
        assert_eq!(codes, ["T101", "T104", "T103", "T102", "T106"]);
    }

    #[test]
    fn empty_instruction() {
        let (_, d) = run("<!-- canary -->\n\n```toml task\n```\n");
        assert_eq!(
            d,
            ["T105 the instruction is empty: task.md must start with the task itself"]
        );
    }

    #[test]
    fn longer_outer_fences_show_reserved_fences() {
        let (s, d) = run("Write:\n\n````\n```toml task\nx\n```\n````\n");
        assert!(d.is_empty());
        assert!(s.blocks.is_empty());
        assert!(s.instruction.contains("```toml task"));
    }

    #[test]
    fn frontmatter_moves_the_first_line() {
        let (s, _) = run("---\na: 1\n---\nDo it.\n```notes\nx\n```\n");
        assert_eq!(s.frontmatter.as_ref().map(|f| f.0.as_str()), Some("a: 1"));
        assert_eq!(s.blocks[0].line, 5);
    }
}
