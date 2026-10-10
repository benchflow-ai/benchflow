//! The parts of a 0.8 task.md: YAML frontmatter, then the prompt body with
//! its optional reserved sections, plus `prompts/` sidecar files.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

/// 0.8's reserved prompt headings: `## prompt`, `## role:<name>`,
/// `## scene:<name>`, `## user-persona` (in any case).
static SECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?mi)^##\s+(prompt|role:[A-Za-z0-9_.-]+|scene:[A-Za-z0-9_.-]+|user-persona)\s*$")
        .unwrap()
});
static ESCAPED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?mi)^\\(##\s+(?:prompt|role:[A-Za-z0-9_.-]+|scene:[A-Za-z0-9_.-]+|user-persona)\s*)$",
    )
    .unwrap()
});

/// A task.md split at its frontmatter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Split {
    /// The YAML between the `---` lines.
    pub front: String,
    /// The 1-based line of task.md where the YAML starts.
    pub front_line: usize,
    /// Everything after the closing `---`, leading newlines dropped.
    pub body: String,
}

/// Split task.md text the way 0.8 does: the first line is `---` (spaces
/// allowed), and the next line that is `---` closes the frontmatter.
pub(crate) fn split(text: &str) -> Result<Split, String> {
    let normalized = text.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.split_inclusive('\n').collect();
    if lines.first().is_none_or(|l| l.trim() != "---") {
        return Err("task.md must start with YAML frontmatter".into());
    }
    let Some(close) = lines
        .iter()
        .skip(1)
        .position(|l| l.trim() == "---")
        .map(|i| i + 1)
    else {
        return Err("task.md frontmatter is missing its closing ---".into());
    };
    Ok(Split {
        front: lines[1..close].concat(),
        front_line: 2,
        body: lines[close + 1..]
            .concat()
            .trim_start_matches('\n')
            .to_string(),
    })
}

/// The prompts a body carries.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Prompts {
    /// The base prompt: `## prompt`, else the text before the first
    /// section, else the whole body.
    pub instruction: String,
    pub roles: BTreeMap<String, String>,
    pub scenes: BTreeMap<String, String>,
    pub user_persona: Option<String>,
}

/// Split a body into its prompts. `Err` names a repeated section.
pub(crate) fn prompts(body: &str) -> Result<Prompts, String> {
    let matches: Vec<regex::Captures<'_>> = SECTION.captures_iter(body).collect();
    if matches.is_empty() {
        return Ok(Prompts {
            instruction: body.trim().to_string(),
            ..Prompts::default()
        });
    }
    let first = matches[0].get(0).expect("whole match").start();
    let preamble = body[..first].trim();
    let mut sections: Vec<(String, String)> = Vec::new();
    for (i, m) in matches.iter().enumerate() {
        let key = section_key(m[1].trim());
        if sections.iter().any(|(k, _)| *k == key) {
            return Err(format!("task.md has the section ## {key} twice"));
        }
        let start = m.get(0).expect("whole match").end();
        let end = matches
            .get(i + 1)
            .map_or(body.len(), |n| n.get(0).expect("whole match").start());
        sections.push((key, body[start..end].trim().to_string()));
    }
    let get = |k: &str| {
        sections
            .iter()
            .find(|(key, _)| key == k)
            .map(|(_, v)| v.clone())
    };
    let mut out = Prompts {
        instruction: unescape(get("prompt").as_deref().unwrap_or(preamble).trim()),
        user_persona: get("user-persona").map(|v| unescape(&v)),
        ..Prompts::default()
    };
    for (key, value) in &sections {
        if let Some(name) = key.strip_prefix("role:") {
            out.roles.insert(name.to_string(), unescape(value));
        } else if let Some(name) = key.strip_prefix("scene:") {
            out.scenes.insert(name.to_string(), unescape(value));
        }
    }
    Ok(out)
}

/// `Role:Planner` -> `role:Planner`: the kind is lowercased, the name kept.
fn section_key(raw: &str) -> String {
    match raw.split_once(':') {
        Some((kind, name)) => format!("{}:{name}", kind.to_lowercase()),
        None => raw.to_lowercase(),
    }
}

/// Undo the `\## prompt` escaping 0.8's importer applies to reserved
/// headings inside a prompt.
fn unescape(text: &str) -> String {
    ESCAPED.replace_all(text, "$1").into_owned()
}

/// Read `prompts/role.<name>.md`, `prompts/scene.<name>.md`, and
/// `prompts/user-persona.md` into `prompts`, where they win over headings
/// of the same name.
pub(crate) fn add_sidecars(dir: &Path, prompts: &mut Prompts) -> std::io::Result<()> {
    let folder = dir.join("prompts");
    if !folder.is_dir() {
        return Ok(());
    }
    let mut names: Vec<String> = std::fs::read_dir(&folder)?
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".md"))
        .collect();
    names.sort();
    for name in names {
        let path = folder.join(&name);
        if !path.is_file() {
            continue;
        }
        let stem = &name[..name.len() - 3];
        let text = crate::split::read_text(&path)?.trim().to_string();
        if stem == "user-persona" {
            prompts.user_persona = Some(text);
        } else if let Some(role) = stem.strip_prefix("role.") {
            prompts.roles.insert(role.to_string(), text);
        } else if let Some(scene) = stem.strip_prefix("scene.") {
            prompts.scenes.insert(scene.to_string(), text);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_frontmatter_like_08() {
        let s = split("---\na: 1\n--- \n\n\nBody\n").unwrap();
        assert_eq!(s.front, "a: 1\n");
        assert_eq!(s.body, "Body\n");
        assert!(split("Body").is_err());
        assert!(split("---\na: 1\n").is_err());
        let crlf = split("---\r\na: 1\r\n---\r\nBody\r\n").unwrap();
        assert_eq!(crlf.body, "Body\n");
    }

    #[test]
    fn a_plain_body_is_the_prompt() {
        let p = prompts("\n\nDo the thing.\n\n").unwrap();
        assert_eq!(p.instruction, "Do the thing.");
    }

    #[test]
    fn reserved_sections() {
        let body = "Preamble\n\n## Prompt\n\nDo it.\n\\## role:x\n\n## role:Solver\nSolve.\n## scene:a\nScene.\n## user-persona\nGrumpy.\n";
        let p = prompts(body).unwrap();
        assert_eq!(p.instruction, "Do it.\n## role:x");
        assert_eq!(p.roles.get("Solver").map(String::as_str), Some("Solve."));
        assert_eq!(p.scenes.get("a").map(String::as_str), Some("Scene."));
        assert_eq!(p.user_persona.as_deref(), Some("Grumpy."));
        let preamble = prompts("Base.\n## role:r\nRole.\n").unwrap();
        assert_eq!(preamble.instruction, "Base.");
        assert!(prompts("## prompt\na\n## prompt\nb\n").is_err());
    }
}
