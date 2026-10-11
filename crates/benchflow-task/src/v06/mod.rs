//! BenchFlow 0.8's task.md: YAML frontmatter (the task config, document keys
//! such as `agents` and `scenes`, and authoring shorthands), then the prompt.
//!
//! The package around it is `environment/` (the Dockerfile), `verifier/`
//! (or the legacy `tests/`) with `test.sh` and optionally `verifier.md`, and
//! `oracle/` (or the legacy `solution/`). 0.8 rejects unknown frontmatter
//! keys, and so does [`check`], naming the key and its line.
//!
//! The config model is Harbor's with 0.8's names (`sandbox`, `oracle`,
//! `sandbox_mode`), so it is read by [`crate::split::config`] in its
//! frontmatter dialect. Interaction features the Rust runtime cannot run yet
//! (scenes, simulated users, verifier strategies other than a script) load
//! and are listed as unsupported.

mod document;
mod normalize;
pub(crate) mod verifier_md;

use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::check::{CheckOptions, CheckReport};
use crate::diag::{Diagnostic, Explanation};
use crate::package::{Format, Unsupported};
use crate::split::config::{self, Codes, Locate, Reader};
use crate::split::keys::{self, Problem};
use crate::split::package::{self as build, Layout, Names, Parts};
use crate::split::schema::Dialect;
use crate::values::format_duration;

const CODES: Codes = Codes {
    invalid: "X202",
    ignored: "X211",
    deprecated: "X212",
    benchflow_only: "X211",
};

/// Keys of the frontmatter that belong to the document, not the config.
const DOCUMENT_KEYS: [&str; 4] = ["agents", "benchflow", "scenes", "user"];

/// The placeholder `bench tasks init` writes, which 0.8 refuses to run.
const PLACEHOLDER: &str = "[REPLACE:";

/// Frontmatter keys that mark a task 0.8 materializes with a task-format
/// plugin before it reads it.
const TASK_FORMATS: [&str; 1] = ["robouse"];

/// Diagnostic codes this module emits.
pub static EXPLANATIONS: &[Explanation] = &[
    Explanation {
        code: "X200",
        title: "task.md frontmatter cannot be read",
        body: "A BenchFlow 0.8 task.md starts with a line `---`, then YAML that is a mapping, then another \
`---` line, then the prompt. The message says which part is missing or invalid.",
    },
    Explanation {
        code: "X201",
        title: "unknown key in task.md frontmatter",
        body: "BenchFlow 0.8 rejects frontmatter keys its config does not define, so a typo fails instead \
of falling back to a default. The message names the closest key the table holds. Free-form fields belong \
under `metadata`; BenchFlow extensions under `benchflow`.",
    },
    Explanation {
        code: "X202",
        title: "invalid value in task.md frontmatter",
        body: "A value has the wrong type or breaks one of 0.8's rules, such as `allowed_hosts` without \
network_mode \"allowlist\", or a time limit that is not a positive number of seconds.",
    },
    Explanation {
        code: "X203",
        title: "renamed key in task.md frontmatter",
        body: "0.8 renamed the split layout's `environment` to `sandbox` and `verifier.environment_mode` to \
`verifier.sandbox_mode`. task.md accepts only the new names; task.toml imports convert the old ones.",
    },
    Explanation {
        code: "X204",
        title: "invalid prompt sections",
        body: "The body's reserved headings (`## prompt`, `## role:<name>`, `## scene:<name>`, `## \
user-persona`) may each appear once.",
    },
    Explanation {
        code: "X205",
        title: "invalid authoring shorthand or profile",
        body: "`name`, `image`, `verifier: <path>`, `oracle: <path>`, and `profile`/`profiles` expand into \
the config. A profile must be one 0.8 ships: code-change, reward-kit, acceptance-live, multi-agent, \
leaderboard-local, or 0.8's profile for split-layout compatibility, which bench still reads under 0.8's name \
but does not list. A path must stay inside the package.",
    },
    Explanation {
        code: "X206",
        title: "invalid verifier document",
        body: "verifier/verifier.md declares scoring strategies in YAML frontmatter under `verifier`. It \
needs at least one strategy, each with a known `type` and its required fields, and `default_strategy` must \
name one of them.",
    },
    Explanation {
        code: "X207",
        title: "files that do not take part",
        body: "task.toml and instruction.md beside a 0.8 task.md are ignored: task.md is the task. Remove \
them, or move what they say into task.md.",
    },
    Explanation {
        code: "X208",
        title: "missing package files",
        body: "A 0.8 package needs environment/ with a Dockerfile (unless the config names a prebuilt image), \
and a verifier folder (verifier/, or the legacy tests/) with test.sh or a verifier.md strategy.",
    },
    Explanation {
        code: "X209",
        title: "invalid agents, scenes, or user",
        body: "`agents.roles` maps role names to tables with an `agent`; `scenes` is a list whose turns name \
declared roles; `user` and `benchflow` are mappings.",
    },
    Explanation {
        code: "X210",
        title: "unfinished task",
        body: "The prompt is empty, or a file still holds a `[REPLACE: ...]` placeholder from `bench tasks \
init`. BenchFlow 0.8 refuses such a task.",
    },
    Explanation {
        code: "X211",
        title: "setting with no effect",
        body: "The setting is read but another one overrides it, such as `allow_internet` beside \
`network_mode`, or `memory` beside a different `memory_mb`. Remove the one that does not count.",
    },
    Explanation {
        code: "X212",
        title: "deprecated spelling",
        body: "`memory: 4G`, `storage: 10G`, and `allow_internet` still work but are deprecated in favor of \
`memory_mb`, `storage_mb`, and `network_mode`.",
    },
    Explanation {
        code: "X213",
        title: "task needs a task-format plugin",
        body: "Some 0.8 tasks are a task.md with a block such as `robouse:`, which a plugin installed beside \
BenchFlow 0.8 (a `benchflow.task_formats` entry point) turns into a full package, with its sandbox and \
verifier, before the task is read. This runtime does not run task-format plugins yet; run such tasks with \
BenchFlow 0.8.",
    },
];

static KEY_LINE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"^(\s*)(?:-\s+)?(?:"([^"]*)"|'([^']*)'|([^\s:#'"{}\[\],][^:#]*?))\s*:(?:\s|$)"#)
        .unwrap()
});

/// Finds a key path's line in YAML frontmatter, by indentation.
struct YamlLocator<'a> {
    lines: Vec<&'a str>,
    /// The file line of the YAML's first line.
    first: usize,
}

impl<'a> YamlLocator<'a> {
    fn new(text: &'a str, first: usize) -> Self {
        Self {
            lines: text.lines().collect(),
            first,
        }
    }

    fn line_of(&self, path: &str) -> Option<usize> {
        let keys = path_keys(path);
        let mut from = 0usize;
        let mut min_indent = 0usize;
        let mut found = None;
        for (depth, key) in keys.iter().enumerate() {
            let mut hit = None;
            for (i, line) in self.lines.iter().enumerate().skip(from) {
                let Some(c) = KEY_LINE.captures(line) else {
                    continue;
                };
                let indent = c[1].len();
                if depth > 0 && indent < min_indent && !line.trim().is_empty() && i > from {
                    break;
                }
                let name = c
                    .get(2)
                    .or(c.get(3))
                    .or(c.get(4))
                    .map(|m| m.as_str().trim());
                if name == Some(key.as_str()) && (depth == 0 || indent >= min_indent) {
                    hit = Some((i, indent));
                    break;
                }
            }
            let Some((i, indent)) = hit else { break };
            found = Some(i);
            from = i + 1;
            min_indent = indent + 1;
        }
        found.map(|i| self.first + i)
    }
}

impl Locate for YamlLocator<'_> {
    fn locate(&self, path: &str, _value: bool) -> (Option<usize>, Option<(usize, usize)>) {
        (self.line_of(path), None)
    }
}

/// The keys of a source path, without list indices or quotes.
fn path_keys(path: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    let mut depth = 0;
    for c in path.chars() {
        match c {
            '"' => quoted = !quoted,
            '[' if !quoted => depth += 1,
            ']' if !quoted => depth -= 1,
            '.' if !quoted && depth == 0 => {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
            }
            _ if depth == 0 => cur.push(c),
            _ => {}
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Check a 0.8 task.md package folder.
pub fn check(dir: &Path, _opts: &CheckOptions) -> CheckReport {
    let mut report = CheckReport::new(dir, Some(Format::TaskMdV06));
    let md_path = dir.join("task.md");
    let text = match crate::split::read_text(&md_path) {
        Ok(t) => t,
        Err(e) => {
            report.diagnostics.push(
                Diagnostic::error("X200", format!("cannot read task.md: {e}")).in_file(&md_path),
            );
            return report;
        }
    };
    let parts = match document::split(&text) {
        Ok(p) => p,
        Err(e) => {
            report
                .diagnostics
                .push(Diagnostic::error("X200", e).at(&md_path, Some(1), None));
            return report;
        }
    };
    let front: Map<String, Value> = match parse_yaml(&parts.front) {
        Ok(m) => m,
        Err(e) => {
            let (message, line) = in_file_lines(&e, parts.front_line);
            report.diagnostics.push(
                Diagnostic::error("X200", message)
                    .at(&md_path, Some(line), None)
                    .help("a value with `: ` in it needs quotes (`description: \"Fix it: now\"`)"),
            );
            return report;
        }
    };
    let loc = YamlLocator::new(&parts.front, parts.front_line);
    let normalized = match normalize::normalize(&front) {
        Ok((n, _)) => n,
        Err((path, message)) => {
            let line = loc.line_of(&path);
            report
                .diagnostics
                .push(Diagnostic::error("X205", message).at(&md_path, line, None));
            return report;
        }
    };
    let config_value = Value::Object(
        normalized
            .iter()
            .filter(|(k, _)| !DOCUMENT_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    );

    let plugin = TASK_FORMATS
        .iter()
        .find(|k| front.contains_key(**k))
        .copied();
    if let Some(name) = plugin {
        report.diagnostics.push(
            Diagnostic::error(
                "X213",
                format!("a {name} task: BenchFlow 0.8 builds its sandbox and verifier with the {name} task format before reading it"),
            )
            .at(&md_path, loc.line_of(name), None)
            .note_line("task formats (benchflow.task_formats plugins) are not supported by this runtime yet")
            .help(format!("run it with BenchFlow 0.8 and the {name} package installed")),
        );
    }
    check_renamed_keys(&config_value, &md_path, &loc, &mut report);
    for issue in keys::unknown_keys(&config_value, Dialect::Frontmatter) {
        if renamed(&issue.path).is_some() || Some(issue.path.as_str()) == plugin {
            continue;
        }
        let table = if issue.table.is_empty() {
            "the frontmatter".to_string()
        } else {
            format!("`{}`", issue.table)
        };
        let mut d = Diagnostic::error("X201", format!("unknown key `{}` in {table}", issue.key))
            .at(&md_path, loc.line_of(&issue.path), None);
        match (&issue.problem, &issue.suggestion) {
            (Problem::Misplaced { homes }, _) => {
                let homes: Vec<String> = homes
                    .iter()
                    .map(|h| {
                        if h.is_empty() {
                            "the top level".into()
                        } else {
                            format!("`{h}`")
                        }
                    })
                    .collect();
                d = d.help(format!("`{}` belongs in {}", issue.key, homes.join(", ")));
            }
            (Problem::Unknown, Some(s)) => d = d.help(format!("did you mean `{s}`?")),
            (Problem::Unknown, None) => {
                d = d
                    .help("free-form fields belong under `metadata`; extensions under `benchflow`");
            }
        }
        if !issue.holds.is_empty() {
            d = d.note_line(format!("{table} holds: {}", issue.holds.join(", ")));
        }
        report.diagnostics.push(d);
    }
    let mut reader = Reader {
        dialect: Dialect::Frontmatter,
        file: &md_path,
        loc: &loc,
        codes: CODES,
        diags: Vec::new(),
    };
    let cfg = config::read(&mut reader, &config_value);
    report.diagnostics.extend(reader.diags);

    let mut refused: Vec<Unsupported> = Vec::new();
    check_document(&normalized, &md_path, &loc, &mut report, &mut refused);
    let mut prompts_failed = false;
    let mut prompts = match document::prompts(&parts.body) {
        Ok(p) => p,
        Err(e) => {
            // Point at the section's second heading.
            let line = e.split("## ").nth(1).and_then(|rest| {
                let name = rest.split_whitespace().next()?.to_lowercase();
                text.lines()
                    .enumerate()
                    .filter(|(_, l)| {
                        l.trim()
                            .strip_prefix("## ")
                            .is_some_and(|h| h.trim().to_lowercase() == name)
                    })
                    .nth(1)
                    .map(|(i, _)| i + 1)
            });
            report.diagnostics.push(
                Diagnostic::error("X204", e)
                    .at(&md_path, line, None)
                    .help("merge the two sections into one"),
            );
            prompts_failed = true;
            document::Prompts::default()
        }
    };
    if let Err(e) = document::add_sidecars(dir, &mut prompts) {
        report.diagnostics.push(
            Diagnostic::error("X200", format!("cannot read prompts/: {e}"))
                .in_file(dir.join("prompts")),
        );
    }
    if prompts.instruction.trim().is_empty() && !prompts_failed {
        report.diagnostics.push(
            Diagnostic::error("X210", "the task.md prompt is empty")
                .in_file(&md_path)
                .help("write the instruction below the frontmatter"),
        );
    }
    if text.contains(PLACEHOLDER) {
        report.diagnostics.push(
            Diagnostic::error("X210", "task.md still holds a [REPLACE: ...] placeholder")
                .in_file(&md_path)
                .help("replace the placeholders with the task's own text"),
        );
    }
    if prompts.user_persona.is_some() {
        refused.push(Unsupported {
            field: "## user-persona".into(),
            reason: "simulated users are not supported yet".into(),
            agents_only: false,
        });
    }
    for name in ["task.toml", "instruction.md"] {
        if dir.join(name).exists() {
            report.diagnostics.push(
                Diagnostic::warning(
                    "X207",
                    format!("{name} beside task.md is ignored: task.md is the task"),
                )
                .in_file(dir.join(name))
                .help(format!(
                    "remove {name}, or move what it says into task.md, which is what runs"
                )),
            );
        }
    }
    let layout = if plugin.is_some() {
        // The plugin writes the package's folders; their absence says nothing.
        Layout::detect(dir, false)
    } else {
        check_files(dir, &cfg, &mut report, &mut refused)
    };
    report.facts = vec![
        format!(
            "grading needs a model: {}",
            if cfg.verifier.kind == "llm-judge" {
                "yes: BenchFlow 0.8's LLM judge"
            } else {
                "no"
            }
        ),
        format!(
            "worst-case grading time: {}",
            format_duration(crate::values::duration_from_secs(cfg.verifier.timeout_sec))
        ),
        match layout.oracle_dir {
            Some(d) => format!("reference solution: {d}/{}", layout.oracle_entry),
            None => "reference solution: none".into(),
        },
    ];
    report.package = Some(build::build(
        &cfg,
        Parts {
            root: dir,
            format: Format::TaskMdV06,
            instruction: prompts.instruction,
            layout,
            raw: Value::Object(front),
            names: Names {
                dialect: Dialect::Frontmatter,
                sandbox: "sandbox",
            },
            unsupported: refused,
        },
    ));
    report
}

fn parse_yaml(text: &str) -> Result<Map<String, Value>, String> {
    if text.trim().is_empty() {
        return Ok(Map::new());
    }
    let value: Value = serde_yaml_ng::from_str(text)
        .map_err(|e| format!("task.md frontmatter is not valid YAML: {e}"))?;
    match value {
        Value::Null => Ok(Map::new()),
        Value::Object(m) => Ok(m),
        _ => Err("task.md frontmatter must be a mapping".into()),
    }
}

/// A YAML error that counts lines from the frontmatter's first line, with
/// its line in the file's own numbering (the frontmatter starts at file line
/// `first`), and that line.
fn in_file_lines(message: &str, first: usize) -> (String, usize) {
    let Some(at) = message.rfind(" at line ") else {
        return (message.to_string(), first);
    };
    let rest = &message[at + " at line ".len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    let Ok(n) = digits.parse::<usize>() else {
        return (message.to_string(), first);
    };
    let line = first + n.saturating_sub(1);
    (
        format!(
            "{} at task.md line {line}{}",
            &message[..at],
            &rest[digits.len()..]
        ),
        line,
    )
}

/// Harbor spellings 0.8 renamed: `environment` at the top, and
/// `environment` and `environment_mode` in any verifier table.
fn renamed(path: &str) -> Option<(&'static str, &'static str)> {
    let keys = path_keys(path);
    match keys.as_slice() {
        [k] if k == "environment" => Some(("environment", "sandbox")),
        [.., parent, k] if parent == "verifier" && k == "environment" => {
            Some(("verifier.environment", "verifier.sandbox"))
        }
        [.., parent, k] if parent == "verifier" && k == "environment_mode" => {
            Some(("verifier.environment_mode", "verifier.sandbox_mode"))
        }
        _ => None,
    }
}

fn check_renamed_keys(
    config: &Value,
    file: &Path,
    loc: &YamlLocator<'_>,
    report: &mut CheckReport,
) {
    for issue in keys::unknown_keys(config, Dialect::Frontmatter) {
        if let Some((old, new)) = renamed(&issue.path) {
            report.diagnostics.push(
                Diagnostic::error("X203", format!("the `{old}` key was renamed to `{new}`"))
                    .at(file, loc.line_of(&issue.path), None)
                    .help(format!("rename it to `{new}`, which BenchFlow 0.8 reads")),
            );
        }
    }
}

/// `agents`, `scenes`, `user`, and `benchflow`: what 0.8's document parser
/// checks, and what the Rust runtime cannot run yet.
fn check_document(
    front: &Map<String, Value>,
    file: &Path,
    loc: &YamlLocator<'_>,
    report: &mut CheckReport,
    refused: &mut Vec<Unsupported>,
) {
    let mut err = |path: &str, msg: String| {
        report
            .diagnostics
            .push(Diagnostic::error("X209", msg).at(file, loc.line_of(path), None));
    };
    let mut roles: Vec<String> = Vec::new();
    match front.get("agents") {
        None | Some(Value::Null) => {}
        Some(Value::Object(agents)) => match agents.get("roles") {
            None | Some(Value::Null) => {}
            Some(Value::Object(rs)) => {
                for (name, role) in rs {
                    roles.push(name.clone());
                    let at = format!("agents.roles.{name}");
                    match role {
                        Value::Null => err(&at, format!("{at}.agent is required")),
                        Value::Object(r) => {
                            if !r
                                .get("agent")
                                .and_then(Value::as_str)
                                .is_some_and(|a| !a.is_empty())
                            {
                                err(&at, format!("{at}.agent is required"));
                            }
                            for key in ["model", "reasoning_effort", "skills_dir"] {
                                if r.get(key).is_some_and(|v| !v.is_null() && !v.is_string()) {
                                    err(
                                        &format!("{at}.{key}"),
                                        format!("{at}.{key} must be a string"),
                                    );
                                }
                            }
                            for key in ["timeout_sec", "idle_timeout_sec"] {
                                if r.get(key)
                                    .is_some_and(|v| !v.is_null() && !v.is_i64() && !v.is_u64())
                                {
                                    err(
                                        &format!("{at}.{key}"),
                                        format!("{at}.{key} must be a whole number"),
                                    );
                                }
                            }
                            if r.get("env").is_some_and(|v| !v.is_null() && !v.is_object()) {
                                err(&format!("{at}.env"), format!("{at}.env must be a mapping"));
                            }
                            if r.get("capabilities").is_some_and(|v| {
                                !v.is_null()
                                    && !v.as_array().is_some_and(|a| a.iter().all(Value::is_string))
                            }) {
                                err(
                                    &format!("{at}.capabilities"),
                                    format!("{at}.capabilities must be a list of strings"),
                                );
                            }
                        }
                        _ => err(&at, format!("{at} must be a mapping")),
                    }
                }
            }
            Some(_) => err("agents.roles", "agents.roles must be a mapping".into()),
        },
        Some(_) => err("agents", "agents must be a mapping".into()),
    }
    match front.get("scenes") {
        None | Some(Value::Null) => {}
        Some(Value::Array(scenes)) => {
            for (i, scene) in scenes.iter().enumerate() {
                let at = format!("scenes[{i}]");
                let Some(s) = scene.as_object() else {
                    if !scene.is_null() {
                        err("scenes", format!("{at} must be a mapping"));
                    }
                    continue;
                };
                let declared: Vec<String> = match s.get("roles") {
                    None | Some(Value::Null) => turn_roles(s).unwrap_or_else(|| roles.clone()),
                    Some(Value::Array(a)) if a.iter().all(Value::is_string) => a
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect(),
                    Some(_) => {
                        err("scenes", "scene roles must be a list of role names".into());
                        Vec::new()
                    }
                };
                for r in &declared {
                    if !roles.contains(r) {
                        err(
                            "scenes",
                            format!("{at}.roles references unknown role {r:?}"),
                        );
                    }
                }
                match s.get("turns") {
                    None | Some(Value::Null) => {}
                    Some(Value::Array(turns)) => {
                        for (j, turn) in turns.iter().enumerate() {
                            let role = match turn {
                                Value::String(r) => Some(r.clone()),
                                Value::Object(t) => match t.get("role") {
                                    Some(Value::String(r)) => Some(r.clone()),
                                    _ => {
                                        err("scenes", format!("turns[{j}].role is required"));
                                        None
                                    }
                                },
                                _ => {
                                    err("scenes", format!("turns[{j}] must be a mapping"));
                                    None
                                }
                            };
                            if let Some(r) = role {
                                if !roles.contains(&r) {
                                    err(
                                        "scenes",
                                        format!("turns[{j}].role references unknown role {r:?}"),
                                    );
                                } else if !declared.contains(&r) {
                                    err(
                                        "scenes",
                                        format!(
                                            "turns[{j}] references role {r:?} outside the scene"
                                        ),
                                    );
                                }
                            }
                        }
                    }
                    Some(_) => err("scenes", "scene turns must be a list".into()),
                }
            }
            if !scenes.is_empty() {
                refused.push(Unsupported {
                    field: "scenes".into(),
                    reason:
                        "scenes (several turns, roles, or agents in one run) are not supported yet"
                            .into(),
                    agents_only: false,
                });
            }
        }
        Some(_) => err("scenes", "scenes must be a list".into()),
    }
    match front.get("user") {
        None | Some(Value::Null) => {}
        Some(Value::Object(u)) => {
            if !u.is_empty() {
                refused.push(Unsupported {
                    field: "user".into(),
                    reason: "simulated users are not supported yet".into(),
                    agents_only: false,
                });
            }
        }
        Some(_) => err("user", "user must be a mapping".into()),
    }
    match front.get("benchflow") {
        None | Some(Value::Null) => {}
        Some(Value::Object(bf)) => {
            for (key, reason) in [
                ("agent_policy", "agent policy is not enforced yet"),
                ("runtime_policy", "runtime policy is not enforced yet"),
                ("teams", "agent-team handoffs are not supported yet"),
                ("nudges", "simulated-user nudges are not supported yet"),
            ] {
                if bf.contains_key(key) {
                    refused.push(Unsupported {
                        field: format!("benchflow.{key}"),
                        reason: reason.into(),
                        agents_only: false,
                    });
                }
            }
            let implementation = bf
                .get("verifier")
                .and_then(|v| v.get("implementation"))
                .and_then(|i| i.get("type"))
                .and_then(Value::as_str);
            if let Some(kind) = implementation
                && !["test-script", "script", "deterministic"].contains(&kind)
            {
                refused.push(Unsupported {
                    field: "benchflow.verifier.implementation.type".into(),
                    reason: format!(
                        "the {kind} verifier implementation is not supported yet; only scripts run"
                    ),
                    agents_only: false,
                });
            }
        }
        Some(_) => err("benchflow", "benchflow must be a mapping".into()),
    }
}

/// The roles a scene's turns name, in order, when the scene lists none.
fn turn_roles(scene: &Map<String, Value>) -> Option<Vec<String>> {
    let turns = scene.get("turns")?.as_array()?;
    let mut names: Vec<String> = Vec::new();
    for t in turns {
        let name = match t {
            Value::String(s) => Some(s.clone()),
            Value::Object(m) => m.get("role").and_then(Value::as_str).map(str::to_string),
            _ => None,
        };
        if let Some(n) = name
            && !names.contains(&n)
        {
            names.push(n);
        }
    }
    (!names.is_empty()).then_some(names)
}

/// The package's folders: the sandbox, the verifier, and the oracle.
fn check_files(
    dir: &Path,
    cfg: &config::Config,
    report: &mut CheckReport,
    refused: &mut Vec<Unsupported>,
) -> Layout {
    let mut layout = Layout::detect(dir, cfg.sandbox.os.as_deref() == Some("windows"));
    let env = dir.join("environment");
    // A prebuilt image needs no build folder: 0.8 runs it without one.
    let builds = cfg.sandbox.docker_image.is_none();
    if builds && !env.is_dir() {
        report.diagnostics.push(
            Diagnostic::error(
                "X208",
                "missing environment/: the folder the sandbox is built from",
            )
            .in_file(&env)
            .help("add environment/Dockerfile, or name a prebuilt image with `image:`"),
        );
    } else if builds && !env.join("Dockerfile").is_file() {
        report.diagnostics.push(
            Diagnostic::error("X208", "missing environment/Dockerfile")
                .in_file(env.join("Dockerfile"))
                .help("add a Dockerfile, or name a prebuilt image with `image:` or `sandbox.docker_image`"),
        );
    }
    let verifier_dir = dir.join(layout.verifier_dir);
    if !verifier_dir.is_dir() {
        report.diagnostics.push(
            Diagnostic::error(
                "X208",
                "missing verifier/: the folder with the script that scores the agent",
            )
            .in_file(dir.join("verifier"))
            .help("add verifier/test.sh, which writes the reward to /logs/verifier/reward.txt"),
        );
        return layout;
    }
    match verifier_md::select(&verifier_dir) {
        Ok(None) => {
            if cfg.verifier.kind == "test-script" && !verifier_dir.join("test.sh").is_file() {
                report.diagnostics.push(
                    Diagnostic::error(
                        "X208",
                        format!(
                            "{}/ has no test.sh and no verifier.md strategy",
                            layout.verifier_dir
                        ),
                    )
                    .in_file(verifier_dir.join("test.sh")),
                );
            }
        }
        Ok(Some(selection)) => {
            selection.apply(layout.verifier_dir, &mut layout, refused);
            if selection.kind == "script" && !verifier_dir.join(&layout.verifier_entry).is_file() {
                report.diagnostics.push(
                    Diagnostic::error(
                        "X208",
                        format!(
                            "the script strategy runs {}/{}, which does not exist",
                            layout.verifier_dir, layout.verifier_entry
                        ),
                    )
                    .in_file(verifier_dir.join("verifier.md")),
                );
            }
        }
        Err(e) => report.diagnostics.push(
            Diagnostic::error(
                "X206",
                format!("{}/verifier.md is invalid: {e}", layout.verifier_dir),
            )
            .in_file(verifier_dir.join("verifier.md")),
        ),
    }
    if let Some(oracle) = layout.oracle_dir {
        let solve = dir.join(oracle).join("solve.sh");
        if crate::split::read_text(&solve).is_ok_and(|t| t.contains(PLACEHOLDER)) {
            report.diagnostics.push(
                Diagnostic::error(
                    "X210",
                    format!("{oracle}/solve.sh still holds a [REPLACE: ...] placeholder"),
                )
                .in_file(solve),
            );
        }
    }
    layout
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locates_yaml_keys() {
        let text = "schema_version: '1.0'\nagent:\n  timeout_sec: 300\nverifier:\n  timeout_sec: 60\n  env:\n    A: b\nsandbox:\n  cpus: 1\n";
        let loc = YamlLocator::new(text, 2);
        assert_eq!(loc.line_of("verifier.timeout_sec"), Some(6));
        assert_eq!(loc.line_of("agent.timeout_sec"), Some(4));
        assert_eq!(loc.line_of("sandbox.cpus"), Some(10));
        assert_eq!(loc.line_of("verifier.env.A"), Some(8));
        assert_eq!(loc.line_of("sandbox.nothing"), Some(9));
    }

    #[test]
    fn path_keys_drop_indices_and_quotes() {
        assert_eq!(
            path_keys("steps[0].agent.\"a.b\""),
            vec!["steps", "agent", "a.b"]
        );
    }
}
