//! The split layout: `task.toml` + `instruction.md` + `environment/` +
//! `tests/` + `solution/` (and `steps/` for multi-step tasks).
//!
//! This is Harbor's task format (`task.toml` schema 1.0 to 1.4) and the
//! format of BenchFlow's own tasks before 0.8. It is read as is: everything
//! Harbor or BenchFlow 0.8 accepts loads. Unlike those readers, which drop
//! keys they do not know without a word, [`check`] warns about every key
//! that would be ignored, at its line and column, with the closest key the
//! table holds and a fix when the rename is certain.
//!
//! Fields the Rust runtime cannot honor yet (multi-step tasks, collect hooks,
//! a verifier image of its own, Windows, TPUs, ...) load, and are listed in
//! [`TaskPackage::unsupported`](crate::package::TaskPackage::unsupported), so
//! a run refuses them by name before anything starts.

pub(crate) mod config;
pub(crate) mod extras;
pub(crate) mod hosts;
pub(crate) mod keys;
pub(crate) mod lax;
pub(crate) mod package;
pub(crate) mod schema;

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;

use self::config::{Codes, Config, Locate, Reader};
use self::keys::{KeyIssue, Problem};
use self::package::{Layout, Names, Parts};
use self::schema::Dialect;
use crate::check::{CheckOptions, CheckReport};
use crate::diag::{Diagnostic, Explanation, Fix};
use crate::package::{Format, Unsupported};
use crate::source::{Source, line_col};
use crate::values::format_duration;

/// The codes a split-layout check emits.
pub(crate) const CODES: Codes = Codes {
    invalid: "X111",
    ignored: "X103",
    deprecated: "X102",
    benchflow_only: "X104",
};

/// Diagnostic codes this module emits.
pub static EXPLANATIONS: &[Explanation] = &[
    Explanation {
        code: "X100",
        title: "unknown key in task.toml",
        body: "Harbor and BenchFlow ignore a task.toml key their config does not define, so a misspelled \
setting such as `network_mod` falls back to its default without a word. The check names the closest key \
the table holds; when the spelling is clearly a typo, `bench check --fix` renames it.",
    },
    Explanation {
        code: "X101",
        title: "key in the wrong table",
        body: "The key exists, but in another table, and where it is no reader looks for it. For example \
`multi_step_reward_strategy` is a top-level key: under [task] it is ignored. Move the key to the table the \
message names.",
    },
    Explanation {
        code: "X102",
        title: "deprecated spelling",
        body: "Harbor still reads `memory = \"4G\"`, `storage = \"10G\"`, and `allow_internet`, but they are \
deprecated in favor of `memory_mb`, `storage_mb`, and `network_mode`. The check says how the old spelling is \
read; `bench check --fix` rewrites it.",
    },
    Explanation {
        code: "X103",
        title: "setting with no effect",
        body: "The setting is read but changes nothing, or another setting overrides it: for example \
`allow_internet` beside `network_mode`, or `multi_step_reward_strategy` in a task without [[steps]]. Where \
Harbor and BenchFlow 0.8 read the setting differently, the message says how.",
    },
    Explanation {
        code: "X104",
        title: "BenchFlow 0.8 spelling Harbor ignores",
        body: "BenchFlow 0.8 reads `[sandbox]`, `[oracle]`, `sandbox_mode`, `image`, and `[solution] NAME = \
\"value\"` in task.toml, but Harbor does not, so the task behaves differently in Harbor. Use Harbor's \
spelling: `[environment]`, `[solution]`, `environment_mode`, `docker_image`, and `[solution.env]`.",
    },
    Explanation {
        code: "X110",
        title: "task.toml is not valid TOML",
        body: "The file could not be parsed. The message gives the position of the first problem.",
    },
    Explanation {
        code: "X111",
        title: "invalid value in task.toml",
        body: "A value has the wrong type or breaks a rule Harbor or BenchFlow 0.8 enforces, such as a \
`name` that is not \"org/name\", `allowed_hosts` without network_mode = \"allowlist\", or a time limit that \
is not a positive number of seconds. Harbor refuses to load such a task.",
    },
    Explanation {
        code: "X112",
        title: "missing instruction.md",
        body: "A split-layout task keeps the agent's instruction in instruction.md beside task.toml (a \
multi-step task keeps one in each steps/<name>/ folder instead).",
    },
    Explanation {
        code: "X113",
        title: "missing verifier script",
        body: "The verifier runs tests/test.sh (verifier/test.sh in BenchFlow 0.8's layout) after the agent's \
run, and it writes the reward to /logs/verifier/reward.txt or reward.json. Without it the task cannot be \
scored.",
    },
    Explanation {
        code: "X114",
        title: "no image for the sandbox",
        body: "The sandbox is built from environment/Dockerfile, or pulled from `docker_image` in \
[environment]. The task has neither. Without an environment/ folder it is an error; with an empty one it is \
a warning, since Harbor runs such tasks on providers that supply their own machine (cloud desktops), which \
this runtime does not support yet.",
    },
];

static CANARY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^(<!--.*canary.*-->|#.*canary.*)$").unwrap());

/// Remove canary lines from the top of an instruction, the way Harbor does:
/// leading lines that are an HTML comment or a `#` comment containing the
/// word "canary" (in any case), and the blank lines after them.
pub fn strip_canary(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut i = 0;
    while i < lines.len() && CANARY.is_match(lines[i].trim()) {
        i += 1;
    }
    while i < lines.len() && lines[i].trim().is_empty() {
        i += 1;
    }
    lines[i..].join("\n")
}

/// Read a text file the way Python's `Path.read_text()` does: UTF-8, with
/// `\r\n` and `\r` turned into `\n`.
pub(crate) fn read_text(path: &Path) -> std::io::Result<String> {
    let bytes = std::fs::read(path)?;
    let text = String::from_utf8(bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    Ok(universal_newlines(text))
}

/// `\r\n` and `\r` to `\n`.
pub(crate) fn universal_newlines(text: String) -> String {
    if text.contains('\r') {
        text.replace("\r\n", "\n").replace('\r', "\n")
    } else {
        text
    }
}

/// Finds TOML keys and values in a file.
pub(crate) struct TomlLocator<'a> {
    pub src: &'a Source,
    pub text: &'a str,
}

impl Locate for TomlLocator<'_> {
    fn locate(&self, path: &str, value: bool) -> (Option<usize>, Option<(usize, usize)>) {
        let span = if value {
            self.src.value_span(path)
        } else {
            self.src.best_span(path)
        };
        match span {
            Some(r) => (Some(line_col(self.text, r.start).0), Some((r.start, r.end))),
            None => (None, None),
        }
    }
}

/// Check a split-layout package folder. The report's package is present
/// whenever `task.toml` parsed, even with errors elsewhere; its digest is
/// left for [`crate::load`] to compute.
pub fn check(dir: &Path, _opts: &CheckOptions) -> CheckReport {
    let mut report = CheckReport::new(dir, Some(Format::Split));
    let toml_path = dir.join("task.toml");
    let text = match std::fs::read(&toml_path).map(String::from_utf8) {
        Ok(Ok(t)) => t,
        Ok(Err(_)) => {
            report.diagnostics.push(
                Diagnostic::error("X110", "task.toml is not valid UTF-8").in_file(&toml_path),
            );
            return report;
        }
        Err(e) => {
            report.diagnostics.push(
                Diagnostic::error("X110", format!("cannot read task.toml: {e}"))
                    .in_file(&toml_path),
            );
            return report;
        }
    };
    let src = match Source::toml(&text, 0) {
        Ok(s) => s,
        Err(e) => {
            let span = e.span.as_ref().map(|r| (r.start, r.end));
            let line = e.span.as_ref().map(|r| line_col(&text, r.start).0);
            report.diagnostics.push(
                Diagnostic::error(
                    "X110",
                    format!("task.toml is not valid TOML: {}", e.message),
                )
                .at(&toml_path, line, span),
            );
            return report;
        }
    };
    let loc = TomlLocator {
        src: &src,
        text: &text,
    };
    for issue in keys::unknown_keys(&src.value, Dialect::TaskToml) {
        report
            .diagnostics
            .push(key_diagnostic(&issue, &toml_path, &loc, &src.value));
    }
    let mut reader = Reader {
        dialect: Dialect::TaskToml,
        file: &toml_path,
        loc: &loc,
        codes: CODES,
        diags: Vec::new(),
    };
    let cfg = config::read(&mut reader, &src.value);
    report.diagnostics.extend(reader.diags);

    let mut refused: Vec<Unsupported> = Vec::new();
    let instruction = read_instruction(dir, &cfg, &mut report);
    let layout = check_files(dir, &cfg, &mut report, &mut refused);
    report.facts = facts(&cfg, &layout);
    let names = Names {
        dialect: Dialect::TaskToml,
        sandbox: cfg.sandbox_key.unwrap_or("environment"),
    };
    report.package = Some(package::build(
        &cfg,
        Parts {
            root: dir,
            format: Format::Split,
            instruction,
            layout,
            raw: src.value.clone(),
            names,
            unsupported: refused,
        },
    ));
    report
}

/// The instruction the agent receives: instruction.md without its leading
/// canary lines. A multi-step task has none at the top.
fn read_instruction(dir: &Path, cfg: &Config, report: &mut CheckReport) -> String {
    let path = dir.join("instruction.md");
    match read_text(&path) {
        // A multi-step task's prompts are under steps/; one at the root is
        // still read, as 0.8 reads it (the task is refused before running).
        Ok(text) => strip_canary(&text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if cfg.steps == 0 {
                report.diagnostics.push(
                    Diagnostic::error("X112", "missing instruction.md: the agent's instruction")
                        .in_file(&path)
                        .help("write the task's instruction in instruction.md beside task.toml"),
                );
            }
            String::new()
        }
        Err(e) => {
            report.diagnostics.push(
                Diagnostic::error("X112", format!("cannot read instruction.md: {e}"))
                    .in_file(&path),
            );
            String::new()
        }
    }
}

/// Check the folders and pick the verifier and oracle.
fn check_files(
    dir: &Path,
    cfg: &Config,
    report: &mut CheckReport,
    refused: &mut Vec<Unsupported>,
) -> Layout {
    let mut layout = Layout::detect(dir, cfg.sandbox.os.as_deref() == Some("windows"));
    let verifier_dir = dir.join(layout.verifier_dir);
    match crate::v06::verifier_md::select(&verifier_dir) {
        Ok(None) => {}
        Ok(Some(selection)) => selection.apply(layout.verifier_dir, &mut layout, refused),
        Err(e) => report.diagnostics.push(
            Diagnostic::error(
                "X111",
                format!("{}/verifier.md is invalid: {e}", layout.verifier_dir),
            )
            .in_file(verifier_dir.join("verifier.md")),
        ),
    }
    let separate_image = cfg.verifier.separate_environment.is_some()
        || verifier_dir.join("Dockerfile").is_file()
        || verifier_dir.join("docker-compose.yaml").is_file();
    let needs_script = cfg.steps == 0 && cfg.verifier.kind == "test-script" && !separate_image;
    if needs_script && !verifier_dir.join(&layout.verifier_entry).is_file() {
        let rel = format!("{}/{}", layout.verifier_dir, layout.verifier_entry);
        report.diagnostics.push(
            Diagnostic::error(
                "X113",
                format!("missing {rel}: the script that scores the agent's work"),
            )
            .in_file(dir.join(&rel))
            .help(
                "write a script that checks the agent's work and writes /logs/verifier/reward.txt",
            ),
        );
    }
    let env = dir.join("environment");
    let no_image = cfg.sandbox.docker_image.is_none()
        && !env.join("Dockerfile").is_file()
        && !env.join("docker-compose.yaml").is_file();
    if no_image && !env.is_dir() {
        report.diagnostics.push(
            Diagnostic::error(
                "X114",
                "no environment/ folder and no docker_image: nothing to build or pull",
            )
            .in_file(&env)
            .help("add environment/Dockerfile, or set docker_image in [environment]"),
        );
    } else if no_image {
        // Harbor runs such a task on a provider that supplies the machine
        // itself, such as a cloud desktop.
        report.diagnostics.push(
            Diagnostic::warning("X114", "environment/ has no Dockerfile and [environment] no docker_image")
                .in_file(env.join("Dockerfile"))
                .note_line("Harbor runs such a task only on an environment provider that supplies its own machine")
                .help("add environment/Dockerfile, or set docker_image in [environment]"),
        );
        refused.push(Unsupported {
            field: "environment/Dockerfile".into(),
            reason: "the task names no image to build or pull; machines from other providers (desktops, cloud computers) are not supported yet".into(),
            agents_only: false,
        });
    }
    layout
}

/// Facts for people: what grading needs and its worst case.
fn facts(cfg: &Config, layout: &Layout) -> Vec<String> {
    let model = if cfg.verifier.kind == "llm-judge" {
        "yes: BenchFlow 0.8's LLM judge"
    } else {
        "no"
    };
    let mut out = vec![
        format!("grading needs a model: {model}"),
        format!(
            "worst-case grading time: {}",
            format_duration(std::time::Duration::from_secs_f64(cfg.verifier.timeout_sec))
        ),
    ];
    match layout.oracle_dir {
        Some(dir) => out.push(format!("reference solution: {dir}/{}", layout.oracle_entry)),
        None => out.push("reference solution: none".into()),
    }
    out
}

/// The diagnostic for a key no reader would honor.
fn key_diagnostic(
    issue: &KeyIssue,
    file: &Path,
    loc: &TomlLocator<'_>,
    value: &serde_json::Value,
) -> Diagnostic {
    let (line, span) = loc.locate(&issue.path, false);
    let table = keys::table_name(&issue.table);
    let mut d = match &issue.problem {
        Problem::Unknown => {
            let what = if issue.is_table {
                let full = if issue.table.is_empty() {
                    issue.key.clone()
                } else {
                    format!("{}.{}", issue.table, issue.key)
                };
                format!("unknown table [{full}] is ignored")
            } else {
                format!("unknown key `{}` in {table} is ignored", issue.key)
            };
            let mut d = Diagnostic::warning("X100", what).at(file, line, span);
            if let Some(s) = &issue.suggestion {
                d = d.label(format!("did you mean `{s}`?"));
            }
            d
        }
        Problem::Misplaced { homes } => {
            // Name the shallowest tables; deeper ones (a step's, the
            // verifier's own sandbox) go in a note.
            let top = homes.first().map_or(0, |h| keys::depth(h));
            let (near, far): (Vec<String>, Vec<String>) =
                homes.iter().cloned().partition(|h| keys::depth(h) == top);
            let mut d = Diagnostic::warning(
                "X101",
                format!(
                    "`{}` belongs in {}, not {table}; here it is ignored",
                    issue.key,
                    keys::list_tables(&near)
                ),
            )
            .at(file, line, span)
            .help(format!("move it to {}", keys::list_tables(&near)));
            if !far.is_empty() {
                d = d.note_line(format!("it is also read in {}", keys::list_tables(&far)));
            }
            d
        }
    };
    if !issue.holds.is_empty() {
        d = d.note_line(format!("{table} holds: {}", issue.holds.join(", ")));
    }
    if matches!(issue.problem, Problem::Unknown)
        && let Some(s) = &issue.suggestion
    {
        let fixable = span.filter(|&(start, end)| {
            issue.close
                && !issue.is_table
                && !sibling_exists(value, &issue.path, s)
                && span_is_key(loc.text, start, end, &issue.key)
        });
        match fixable {
            Some((start, end)) => {
                d = d.fix(Fix {
                    file: PathBuf::from(file),
                    span: (start, end),
                    replacement: s.clone(),
                    description: format!("rename `{}` to `{s}`", issue.key),
                });
            }
            // Without a span the label is not shown, so say it here.
            None if span.is_none() => d = d.help(format!("did you mean `{s}`?")),
            None => {}
        }
    }
    d
}

/// Whether the table holding `path` already has a key named `name` (a
/// rename would then make a duplicate key).
fn sibling_exists(root: &serde_json::Value, path: &str, name: &str) -> bool {
    let mut parts = split_path(path);
    parts.pop();
    let mut cur = root;
    for p in parts {
        cur = match (cur, p) {
            (serde_json::Value::Object(m), Seg::Key(k)) => match m.get(&k) {
                Some(v) => v,
                None => return false,
            },
            (serde_json::Value::Array(a), Seg::Index(i)) => match a.get(i) {
                Some(v) => v,
                None => return false,
            },
            _ => return false,
        };
    }
    cur.get(name).is_some()
}

enum Seg {
    Key(String),
    Index(usize),
}

/// Split a source path (`a.b[0]."c.d"`) into keys and indices.
fn split_path(path: &str) -> Vec<Seg> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = path.chars().peekable();
    let mut quoted = false;
    while let Some(c) = chars.next() {
        match c {
            '"' => quoted = !quoted,
            '\\' if quoted => {
                if let Some(n) = chars.next() {
                    cur.push(n);
                }
            }
            '.' if !quoted => {
                if !cur.is_empty() {
                    out.push(Seg::Key(std::mem::take(&mut cur)));
                }
            }
            '[' if !quoted => {
                if !cur.is_empty() {
                    out.push(Seg::Key(std::mem::take(&mut cur)));
                }
                let mut n = String::new();
                for d in chars.by_ref() {
                    if d == ']' {
                        break;
                    }
                    n.push(d);
                }
                if let Ok(i) = n.parse() {
                    out.push(Seg::Index(i));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        out.push(Seg::Key(cur));
    }
    out
}

/// Whether the text at the span is the bare key, so replacing it is a
/// rename and nothing else.
fn span_is_key(text: &str, start: usize, end: usize, key: &str) -> bool {
    text.get(start..end) == Some(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_canaries_like_harbor() {
        let text =
            "<!-- BENCHMARK DATA canary GUID 1234 -->\n# harbor-canary GUID x\n\n\nDo the thing.\n";
        assert_eq!(strip_canary(text), "Do the thing.\n");
        assert_eq!(strip_canary("  # Canary here  \nRest"), "Rest");
        assert_eq!(
            strip_canary("Do it.\n<!-- canary -->\n"),
            "Do it.\n<!-- canary -->\n"
        );
        assert_eq!(strip_canary("# Title\nBody"), "# Title\nBody");
    }

    #[test]
    fn universal_newlines_match_python() {
        assert_eq!(universal_newlines("a\r\nb\rc\n".into()), "a\nb\nc\n");
    }

    #[test]
    fn splits_source_paths() {
        let segs = split_path("steps[1].agent.\"a.b\"");
        assert_eq!(segs.len(), 4);
        assert!(matches!(&segs[3], Seg::Key(k) if k == "a.b"));
        assert!(matches!(&segs[1], Seg::Index(1)));
    }
}
