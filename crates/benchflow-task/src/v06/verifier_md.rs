//! BenchFlow 0.8's verifier document, `verifier/verifier.md`: YAML
//! frontmatter whose `verifier` mapping names scoring strategies and picks
//! one (`default_strategy`, else the first). Only `script` strategies that
//! run one packaged script run in the Rust runtime for now; the others load
//! and are listed as unsupported.

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::package::Unsupported;
use crate::split::package::Layout;

static ROLE_SECTION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?mi)^##\s+role:([A-Za-z0-9_.-]+)\s*$").unwrap());

const KNOWN_TYPES: [&str; 6] = [
    "script",
    "reward-kit",
    "llm-judge",
    "agent-judge",
    "ors-episode",
    "taskmd",
];

/// The strategy a verifier document selects.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Selection {
    pub name: String,
    pub kind: String,
    pub command: Option<String>,
}

/// Read `<dir>/verifier.md` when it exists and return the strategy it
/// selects. `Err` says why the document is invalid, in 0.8's terms.
pub(crate) fn select(dir: &Path) -> Result<Option<Selection>, String> {
    let path = dir.join("verifier.md");
    if !path.exists() {
        return Ok(None);
    }
    let text = crate::split::read_text(&path).map_err(|e| format!("cannot read it: {e}"))?;
    parse(&text).map(Some)
}

/// Parse a verifier document's text.
pub(crate) fn parse(text: &str) -> Result<Selection, String> {
    let (front, body) = split(text)?;
    let verifier = mapping(front.get("verifier"), "verifier")?;
    let roles: Vec<String> = ROLE_SECTION
        .captures_iter(&body)
        .map(|c| c[1].to_string())
        .collect();
    let strategies = mapping(verifier.get("strategies"), "verifier.strategies")?;
    if strategies.is_empty() {
        return Err("verifier.strategies must contain at least one strategy".into());
    }
    let mut first: Option<String> = None;
    for (name, raw) in &strategies {
        if name.is_empty() {
            return Err("verifier.strategies keys must be non-empty strings".into());
        }
        let config = mapping(Some(raw), &format!("verifier.strategies.{name}"))?;
        let kind = required_str(
            config.get("type"),
            &format!("verifier.strategies.{name}.type"),
        )?;
        if !KNOWN_TYPES.contains(&kind.as_str()) {
            return Err(format!(
                "verifier.strategies.{name}.type is unsupported: {kind}"
            ));
        }
        validate(name, &kind, &config, &roles)?;
        first.get_or_insert_with(|| name.clone());
    }
    let selected = match optional_str(verifier.get("default_strategy"))? {
        Some(s) => s,
        None => first.unwrap_or_default(),
    };
    let Some(raw) = strategies.get(&selected) else {
        let mut names: Vec<&String> = strategies.keys().collect();
        names.sort();
        let names: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        return Err(format!(
            "verifier.default_strategy {selected:?} must name one of verifier.strategies: {}",
            names.join(", ")
        ));
    };
    optional_str(verifier.get("name"))?;
    if let Some(rubric) = verifier.get("rubric") {
        mapping(Some(rubric), "verifier.rubric")?;
    }
    if let Some(outputs) = verifier.get("outputs") {
        let outputs = mapping(Some(outputs), "verifier.outputs")?;
        match outputs.get("aggregate_policy") {
            None | Some(Value::Null) | Some(Value::Object(_)) => {}
            Some(_) => return Err("verifier.outputs.aggregate_policy must be a mapping".into()),
        }
    }
    let kind = raw["type"].as_str().unwrap_or_default().to_string();
    let command = raw
        .get("command")
        .and_then(Value::as_str)
        .map(str::to_string);
    Ok(Selection {
        name: selected,
        kind,
        command,
    })
}

impl Selection {
    /// Point the layout at the selected script, or record why the strategy
    /// cannot run. `folder` is the verifier folder's name in the package.
    pub(crate) fn apply(&self, folder: &str, layout: &mut Layout, refused: &mut Vec<Unsupported>) {
        let field = format!("{folder}/verifier.md");
        if self.kind != "script" {
            refused.push(Unsupported {
                field,
                reason: format!(
                    "the {} strategy {:?} of BenchFlow 0.8's verifier document is not supported yet; only script strategies run",
                    self.kind, self.name
                ),
                agents_only: false,
            });
            return;
        }
        match self.command.as_deref().and_then(script_entry) {
            Some(entry) => layout.verifier_entry = entry,
            None => refused.push(Unsupported {
                field,
                reason: format!(
                    "the script strategy {:?} runs {:?}; only a command that runs one script in the verifier's folder is supported yet",
                    self.name,
                    self.command.as_deref().unwrap_or_default()
                ),
                agents_only: false,
            }),
        }
    }
}

/// The packaged script a command runs: `test.sh`, `./test.sh`, or
/// `bash run.sh`; `None` for anything else.
pub(crate) fn script_entry(command: &str) -> Option<String> {
    let words = split_words(command)?;
    let script = match words.as_slice() {
        [script] => script,
        [shell, script] if shell == "bash" || shell == "sh" => script,
        _ => return None,
    };
    let rel = script.strip_prefix("./").unwrap_or(script);
    let safe = !rel.is_empty()
        && !rel.starts_with('/')
        && !rel.starts_with('-')
        && !rel.contains("://")
        && rel.split('/').all(|p| !p.is_empty() && p != "..");
    safe.then(|| rel.to_string())
}

/// Split a command into words the way a POSIX shell would for simple
/// commands: whitespace, quotes, and backslashes. `None` on an unclosed
/// quote.
fn split_words(text: &str) -> Option<Vec<String>> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if in_word {
                    words.push(std::mem::take(&mut cur));
                    in_word = false;
                }
            }
            '\'' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '\'' => break,
                        x => cur.push(x),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next()? {
                        '"' => break,
                        '\\' => cur.push(chars.next()?),
                        x => cur.push(x),
                    }
                }
            }
            '\\' => {
                in_word = true;
                cur.push(chars.next()?);
            }
            x => {
                in_word = true;
                cur.push(x);
            }
        }
    }
    if in_word {
        words.push(cur);
    }
    Some(words)
}

/// 0.8's frontmatter split: the first line is `---` (spaces allowed), and a
/// later `---` line closes it.
fn split(text: &str) -> Result<(Map<String, Value>, String), String> {
    let normalized = text.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.split_inclusive('\n').collect();
    if lines.first().is_none_or(|l| l.trim() != "---") {
        return Err("verifier.md must start with YAML frontmatter".into());
    }
    let Some(close) = lines
        .iter()
        .skip(1)
        .position(|l| l.trim() == "---")
        .map(|i| i + 1)
    else {
        return Err("verifier.md frontmatter is missing closing ---".into());
    };
    let front = lines[1..close].concat();
    let body = lines[close + 1..]
        .concat()
        .trim_start_matches('\n')
        .to_string();
    let value: Value = if front.trim().is_empty() {
        Value::Object(Map::new())
    } else {
        serde_yaml_ng::from_str(&front)
            .map_err(|e| format!("verifier.md frontmatter is not valid YAML: {e}"))?
    };
    match value {
        Value::Null => Ok((Map::new(), body)),
        Value::Object(m) => Ok((m, body)),
        _ => Err("verifier.md frontmatter must be a mapping".into()),
    }
}

fn mapping(value: Option<&Value>, path: &str) -> Result<Map<String, Value>, String> {
    match value {
        None | Some(Value::Null) => Err(format!("{path} is required")),
        Some(Value::Object(m)) => Ok(m.clone()),
        Some(_) => Err(format!("{path} must be a mapping")),
    }
}

fn optional_str(value: Option<&Value>) -> Result<Option<String>, String> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) if !s.is_empty() => Ok(Some(s.clone())),
        Some(_) => Err("expected a non-empty string".into()),
    }
}

fn required_str(value: Option<&Value>, path: &str) -> Result<String, String> {
    optional_str(value)?.ok_or_else(|| format!("{path} must be a non-empty string"))
}

fn safe_relative(value: &str, path: &str) -> Result<(), String> {
    if value.starts_with('/') || value.split('/').any(|p| p == "..") {
        return Err(format!("{path} must be a safe relative path"));
    }
    Ok(())
}

fn string_list(value: Option<&Value>, path: &str) -> Result<(), String> {
    match value {
        Some(Value::Array(items))
            if !items.is_empty()
                && items
                    .iter()
                    .all(|x| x.as_str().is_some_and(|s| !s.is_empty())) =>
        {
            Ok(())
        }
        _ => Err(format!("{path} must be a non-empty list of strings")),
    }
}

/// 0.8's per-type required fields.
fn validate(
    name: &str,
    kind: &str,
    config: &Map<String, Value>,
    roles: &[String],
) -> Result<(), String> {
    let p = format!("verifier.strategies.{name}");
    let has = |k: &str| config.contains_key(k);
    match kind {
        "script" => {
            required_str(config.get("command"), &format!("{p}.command"))?;
        }
        "reward-kit" => {
            safe_relative(
                &required_str(config.get("root"), &format!("{p}.root"))?,
                &format!("{p}.root"),
            )?;
            for key in ["criteria", "entrypoint"] {
                if has(key) {
                    let v = required_str(config.get(key), &format!("{p}.{key}"))?;
                    safe_relative(&v, &format!("{p}.{key}"))?;
                }
            }
        }
        "llm-judge" => {
            required_str(config.get("rubric"), &format!("{p}.rubric"))?;
            for key in ["model", "input_dir", "context"] {
                if has(key) {
                    required_str(config.get(key), &format!("{p}.{key}"))?;
                }
            }
            if has("context_file") {
                let v = required_str(config.get("context_file"), &format!("{p}.context_file"))?;
                safe_relative(&v, &format!("{p}.context_file"))?;
            }
            if has("context") && has("context_file") {
                return Err(format!(
                    "{p} must declare either context or context_file, not both"
                ));
            }
        }
        "agent-judge" => {
            let role = required_str(config.get("role"), &format!("{p}.role"))?;
            if !roles.contains(&role) {
                return Err(format!(
                    "{p}.role references missing ## role:{role} section"
                ));
            }
            if has("model") {
                required_str(config.get("model"), &format!("{p}.model"))?;
            }
            if required_str(config.get("isolation"), &format!("{p}.isolation"))? != "verifier-only"
            {
                return Err(format!("{p}.isolation must be 'verifier-only'"));
            }
            string_list(config.get("inputs"), &format!("{p}.inputs"))?;
        }
        "taskmd" => {
            let grading = config.get("grading").and_then(Value::as_str);
            if !matches!(grading, Some("rubric" | "script")) {
                return Err(format!("{p}.grading must be 'rubric' or 'script'"));
            }
            if has("command") {
                let v = required_str(config.get("command"), &format!("{p}.command"))?;
                safe_relative(&v, &format!("{p}.command"))?;
            } else if grading == Some("script") {
                return Err(format!("{p}.command is required when grading is 'script'"));
            }
            if let Some(w) = config.get("workdir")
                && !w.as_str().is_some_and(|s| s.starts_with('/'))
                && !w.is_null()
            {
                return Err(format!("{p}.workdir must be absolute"));
            }
            if !matches!(
                config
                    .get("isolation")
                    .and_then(Value::as_str)
                    .unwrap_or("shared"),
                "shared" | "separate"
            ) {
                return Err(format!("{p}.isolation must be 'shared' or 'separate'"));
            }
            if config.get("offline").is_some_and(|v| !v.is_boolean()) {
                return Err(format!("{p}.offline must be a boolean"));
            }
        }
        "ors-episode" => {
            string_list(config.get("inputs"), &format!("{p}.inputs"))?;
            if has("format") {
                let f = required_str(config.get("format"), &format!("{p}.format"))?;
                if !["json", "jsonl", "auto"].contains(&f.as_str()) {
                    return Err(format!("{p}.format must be json, jsonl, or auto"));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOC: &str = "---\ndocument_version: \"0.3\"\nverifier:\n  name: v\n  default_strategy: deterministic\n  strategies:\n    deterministic:\n      type: script\n      command: ./test.sh\n    judge:\n      type: agent-judge\n      role: verifier_judge\n      isolation: verifier-only\n      inputs: [a]\n---\n\n## role:verifier_judge\n\nGrade.\n";

    #[test]
    fn selects_the_default_strategy() {
        let s = parse(DOC).unwrap();
        assert_eq!(s.name, "deterministic");
        assert_eq!(s.kind, "script");
        assert_eq!(s.command.as_deref(), Some("./test.sh"));
    }

    #[test]
    fn refuses_what_08_refuses() {
        assert!(parse("no frontmatter").is_err());
        assert!(
            parse("---\nverifier: {}\n---\n")
                .unwrap_err()
                .contains("strategies")
        );
        let unknown = "---\nverifier:\n  strategies:\n    x:\n      type: magic\n---\n";
        assert!(parse(unknown).unwrap_err().contains("unsupported"));
        let missing_role = DOC.replace("## role:verifier_judge", "## nothing");
        assert!(parse(&missing_role).unwrap_err().contains("role"));
        let bad_default = DOC.replace("default_strategy: deterministic", "default_strategy: other");
        assert!(parse(&bad_default).is_err());
    }

    #[test]
    fn script_entries() {
        assert_eq!(script_entry("./test.sh").as_deref(), Some("test.sh"));
        assert_eq!(script_entry("bash run.sh").as_deref(), Some("run.sh"));
        assert_eq!(script_entry("sub/run.sh").as_deref(), Some("sub/run.sh"));
        assert_eq!(script_entry("python3 verify.py --fast"), None);
        assert_eq!(script_entry("/abs/test.sh"), None);
        assert_eq!(script_entry("../escape.sh"), None);
        assert_eq!(script_entry("'unclosed"), None);
    }
}
