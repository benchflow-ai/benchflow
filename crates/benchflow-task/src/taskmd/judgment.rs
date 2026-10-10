//! `verifier/rubric.json` and `verifier/behaviors.json`: what counts as good,
//! and what to watch (task.md's docs/rubrics.md and docs/behaviors.md).
//!
//! Both files are recognized by their `$schema`: a verifier's own scripts
//! may keep a file of the same name in another format. The checks port the
//! reference's `check_rubric`, `evidence_problem`, `_stated_file`,
//! `check_stage_rubrics`, and `check_behaviors`.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::diag::{Diagnostic, Fix};
use crate::source::join;

use super::code;
use super::config::{dym_or, suggestion};
use super::pyfmt::{prefix, repr, repr_str, str_of};
use super::pyval::{num, table, truthy};
use super::report::{At, JsonLoc, Sink};
use super::text::read_text;
use super::vocab::{
    CHECKED_JUDGES, CONSEQUENCES, CRITERION_KEYS, JUDGES, MODEL_JUDGED, MONITOR_JUDGES,
    PRIVATE_DIRS, RUBRIC_KEYS, RUNTIME_EVIDENCE, SCHEMA_BASE, SCORING_KEYS, WATCH_KEYS,
    sorted_list, sorted_repr,
};

static CRITERION_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9][a-z0-9-]*\n?\z").unwrap());
static BEHAVIOR_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9][a-z0-9_-]*(?:\.[a-z0-9_-]+)+\n?\z").unwrap());
static SEVERITY_KEY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^-?\d+$").unwrap());
static WHITESPACE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s+").unwrap());

/// A rubric or behaviors file the task declares.
#[derive(Debug, Clone)]
pub(crate) struct Judgment {
    /// The file as JSON (an object).
    pub value: Value,
    /// The file's text as written, for positions.
    pub text: String,
    /// The file, relative to the package.
    pub file: PathBuf,
}

/// Read a JSON object: `None` when the file is missing, with a diagnostic
/// when it is not a JSON object (the reference's `_load_json`).
fn load_json(dir: &Path, rel: &str, diags: &mut Vec<Diagnostic>) -> Option<(Value, String)> {
    let path = dir.join(rel);
    if !path.exists() {
        return None;
    }
    let raw = std::fs::read(&path).ok();
    let text = raw
        .as_deref()
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(str::to_string);
    let Some(text) = text else {
        diags.push(
            Diagnostic::error(
                code::JUDGMENT_FILE,
                "not valid JSON: the file is not UTF-8 text",
            )
            .in_file(rel),
        );
        return None;
    };
    match serde_json::from_str::<Value>(&text) {
        Ok(Value::Object(m)) => Some((Value::Object(m), text)),
        Ok(_) => {
            diags.push(
                Diagnostic::error(code::JUDGMENT_FILE, "must be a JSON object").at(
                    rel,
                    Some(1),
                    None,
                ),
            );
            None
        }
        Err(e) => {
            let (line, col) = (e.line(), e.column());
            let offset =
                super::report::line_span(&text, line).map(|(s, _)| s + col.saturating_sub(1));
            diags.push(
                Diagnostic::error(code::JUDGMENT_FILE, format!("not valid JSON: {e}")).at(
                    rel,
                    Some(line.max(1)),
                    offset.map(|o| (o, (o + 1).min(text.len()))),
                ),
            );
            None
        }
    }
}

/// Whether a JSON object names task.md's schema of `kind`.
pub(crate) fn has_schema(value: &Value, kind: &str) -> bool {
    let schema = value.get("$schema").map(str_of).unwrap_or_default();
    schema.starts_with(&format!("{SCHEMA_BASE}{kind}-"))
}

/// `verifier/<kind>.json` when it names task.md's schema; a warning when it
/// does not (the reference's `_judgment`).
pub(crate) fn load(dir: &Path, kind: &str, diags: &mut Vec<Diagnostic>) -> Option<Judgment> {
    let rel = format!("verifier/{kind}.json");
    let (value, text) = load_json(dir, &rel, diags)?;
    if !has_schema(&value, kind) {
        let loc = JsonLoc::new(&rel, &text);
        let mut sink = Sink::new(Some(&loc));
        sink.warning(
            code::NO_SCHEMA,
            At::Key("\"$schema\""),
            format!(
                "no task.md $schema ({SCHEMA_BASE}{kind}-1.json), so this is read as a file for the verifier's scripts, not as the \
task's {kind}"
            ),
        );
        diags.extend(sink.into_diagnostics());
        return None;
    }
    Some(Judgment {
        value,
        text,
        file: PathBuf::from(rel),
    })
}

/// A task.md JSON file of `kind` read without diagnostics: `None` unless it
/// is a JSON object naming the schema (the reference's `_read_judgment`).
pub(crate) fn read(path: &Path, kind: &str) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    let value: Value = serde_json::from_str(&text).ok()?;
    (value.is_object() && has_schema(&value, kind)).then_some(value)
}

/// Text as a reader sees it: no markdown emphasis or code markers, spacing
/// collapsed, lower case (the reference's `_plain`).
pub(crate) fn plain(text: &str) -> String {
    let unmarked: String = text
        .chars()
        .filter(|c| !matches!(c, '*' | '_' | '`'))
        .collect();
    WHITESPACE.replace_all(&unmarked, " ").trim().to_lowercase()
}

/// Whether an evidence item can hold bytes the solver neither wrote nor ran.
fn independent_evidence(item: &Value) -> bool {
    let Value::String(s) = item else {
        return false;
    };
    let kind = s.split('#').next().unwrap_or("");
    if kind.starts_with('/')
        || kind.starts_with("diff:")
        || kind == "trajectory"
        || kind == "trajectory:reasoning"
    {
        return false;
    }
    matches!(kind, "tests" | "request-record" | "screenshots" | "video")
        || (!kind.contains('/') && !kind.contains('.'))
}

/// `posixpath.normpath`.
pub(crate) fn normpath(path: &str) -> String {
    if path.is_empty() {
        return ".".into();
    }
    let initial = if path.starts_with("//") && !path.starts_with("///") {
        2
    } else if path.starts_with('/') {
        1
    } else {
        0
    };
    let mut parts: Vec<&str> = Vec::new();
    for comp in path.split('/') {
        if comp.is_empty() || comp == "." {
            continue;
        }
        if comp != ".." || (initial == 0 && parts.is_empty()) || parts.last() == Some(&"..") {
            parts.push(comp);
        } else if !parts.is_empty() {
            parts.pop();
        }
    }
    let body = parts.join("/");
    let out = format!("{}{body}", "/".repeat(initial));
    if out.is_empty() { ".".into() } else { out }
}

/// Where an evidence item sits in the sandbox: absolute as written, or under
/// the working folder (the reference's `_sandbox_path`).
fn sandbox_path(item: &str, workdir: Option<&str>) -> String {
    if item.starts_with('/') {
        return normpath(item);
    }
    match workdir {
        Some(w) if !w.is_empty() => {
            let joined = if w.ends_with('/') {
                format!("{w}{item}")
            } else {
                format!("{w}/{item}")
            };
            normpath(&joined)
        }
        _ => normpath(item),
    }
}

/// Whether `path` is `folder` or inside it.
fn in_folder(path: &str, folder: &str) -> bool {
    let trimmed = folder.trim_end_matches('/');
    let folder = if trimmed.is_empty() { "/" } else { trimmed };
    path == folder
        || path.starts_with(&if folder.ends_with('/') {
            folder.to_string()
        } else {
            format!("{folder}/")
        })
}

/// The paths the runtime saves: (the declared outputs, or `None` when the
/// task declares none, and `[sandbox] workdir`).
pub(crate) fn saved_paths(config: &Value) -> (Option<Vec<String>>, Option<String>) {
    let sandbox = table(config, &["sandbox"]);
    let workdir = sandbox
        .get("workdir")
        .and_then(Value::as_str)
        .map(str::to_string);
    let Some(Value::Array(outputs)) = sandbox.get("outputs") else {
        return (None, workdir);
    };
    if outputs.is_empty() {
        return (None, workdir);
    }
    let declared = outputs
        .iter()
        .filter_map(|o| {
            let path = match o {
                Value::String(s) => Some(s.clone()),
                Value::Object(m) => m.get("path").and_then(Value::as_str).map(str::to_string),
                _ => None,
            }?;
            let service = o
                .get("service")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty());
            Some(match service {
                Some(svc) if path.starts_with('/') => {
                    format!("/taskmd/services/{svc}/{}", path.trim_start_matches('/'))
                }
                _ => path,
            })
        })
        .filter(|p| !p.trim().is_empty())
        .collect();
    (Some(declared), workdir)
}

/// What is wrong with an evidence item, if anything: `(is_error, message)`
/// (the reference's `evidence_problem`).
pub(crate) fn evidence_problem(item: &Value, config: &Value) -> Option<(bool, String)> {
    let Some(item) = item.as_str().filter(|s| !s.trim().is_empty()) else {
        return Some((true, "an evidence item is a non-empty string".into()));
    };
    let streams: Vec<&str> = match table(config, &["world", "record"]).get("streams") {
        Some(Value::Array(s)) => s.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    };
    let mut base = item.split('#').next().unwrap_or("");
    if RUNTIME_EVIDENCE.contains(&base) || streams.contains(&base) {
        return None;
    }
    if let Some(rest) = base.strip_prefix("diff:") {
        if rest.is_empty() {
            return Some((
                true,
                "diff: names a file item, such as diff:src/app.py".into(),
            ));
        }
        base = rest;
    }
    if let Some((head, _)) = base.split_once(':') {
        let runtime = [
            "trajectory",
            "tests",
            "screenshots",
            "video",
            "request-record",
        ];
        if runtime.contains(&head) || streams.contains(&head) {
            return Some((
                true,
                format!(
                    "evidence {item}: {head} takes no argument (trajectory:reasoning is the one exception)"
                ),
            ));
        }
    }
    let (outputs, workdir) = saved_paths(config);
    let target = sandbox_path(base, workdir.as_deref());
    if let Some(outputs) = outputs.filter(|o| !o.is_empty()) {
        if outputs
            .iter()
            .any(|o| in_folder(&target, &sandbox_path(o, workdir.as_deref())))
        {
            return None;
        }
        return Some((
            true,
            format!(
                "evidence {item} reads {target}, which no [sandbox] outputs entry saves: add {{ path = \"{target}\" }} to outputs, \
or a folder that holds it"
            ),
        ));
    }
    if base.starts_with('/') {
        match workdir.as_deref().filter(|w| !w.is_empty()) {
            Some(w) if !in_folder(&target, &sandbox_path(w, None)) => {
                return Some((
                    true,
                    format!(
                        "evidence {item} is outside [sandbox] workdir {w}, the folder the runtime saves when no outputs are declared: \
declare the path in [sandbox] outputs"
                    ),
                ));
            }
            Some(_) => {}
            None => {
                return Some((
                    false,
                    format!(
                        "evidence {item} is an absolute path, and with no [sandbox] workdir and no outputs the checker cannot tell \
whether the runtime saves it: set workdir, or declare the path in outputs"
                    ),
                ));
            }
        }
    }
    None
}

/// A path's parts as `PurePosixPath(p).parts` gives them.
fn path_parts(p: &str) -> Vec<&str> {
    let mut parts: Vec<&str> = Vec::new();
    if p.starts_with('/') {
        parts.push("/");
    }
    parts.extend(p.split('/').filter(|s| !s.is_empty() && *s != "."));
    parts
}

/// What a rubric check needs besides the rubric.
struct RubricCtx<'a> {
    told: Option<&'a [String]>,
    told_plain: Vec<String>,
    dir: &'a Path,
    config: Option<&'a Value>,
    deep: bool,
    /// Propose fixes (off while testing one).
    fixes: bool,
    loc: &'a JsonLoc,
}

/// Check a rubric (the reference's `check_rubric`). `told` is what the agent
/// is told; `config`, when given, resolves file evidence against what the
/// runtime saves.
pub(crate) fn check_rubric(
    rubric: &Judgment,
    told: Option<&[String]>,
    dir: &Path,
    config: Option<&Value>,
    deep: bool,
) -> Vec<Diagnostic> {
    let loc = JsonLoc::new(&rubric.file, &rubric.text);
    let ctx = RubricCtx {
        told,
        told_plain: told.unwrap_or_default().iter().map(|t| plain(t)).collect(),
        dir,
        config,
        deep,
        fixes: true,
        loc: &loc,
    };
    let mut sink = Sink::new(Some(&loc));
    rubric_rules(&rubric.value, &ctx, &mut sink);
    sink.into_diagnostics()
}

/// Whether editing a rubric's text leaves no error at `new_path` and no
/// longer reports `message`.
fn json_fix_holds(ctx: &RubricCtx<'_>, edited: &str, new_path: &str, message: &str) -> bool {
    let Ok(value) = serde_json::from_str::<Value>(edited) else {
        return false;
    };
    let loc = JsonLoc::new(&ctx.loc.file, edited);
    let again = RubricCtx {
        told: ctx.told,
        told_plain: ctx.told_plain.clone(),
        dir: ctx.dir,
        config: ctx.config,
        deep: false,
        fixes: false,
        loc: &loc,
    };
    let mut sink = Sink::new(None);
    rubric_rules(&value, &again, &mut sink);
    let under = |p: &str| {
        p == new_path
            || p.starts_with(&format!("{new_path}."))
            || p.starts_with(&format!("{new_path}["))
    };
    !sink.found.iter().any(|f| {
        f.diag.is_error() && (f.path.as_deref().is_some_and(under) || f.diag.message == message)
    })
}

/// Offer to replace the span at `span` with `replacement` in a rubric, when
/// that fixes the error just reported.
fn json_fix(
    ctx: &RubricCtx<'_>,
    sink: &mut Sink<'_>,
    span: Option<std::ops::Range<usize>>,
    replacement: String,
    new_path: &str,
    description: String,
) {
    let Some(span) = span else {
        return;
    };
    let Some(last) = sink.found.last() else {
        return;
    };
    let message = last.diag.message.clone();
    let text = &ctx.loc.text;
    let edited = format!("{}{replacement}{}", &text[..span.start], &text[span.end..]);
    if ctx.fixes
        && json_fix_holds(ctx, &edited, new_path, &message)
        && let Some(d) = sink.found.last_mut()
    {
        d.diag.fix = Some(Fix {
            file: ctx.loc.file.clone(),
            span: (span.start, span.end),
            replacement,
            description,
        });
    }
}

/// Label an unknown key with the closest one, and offer the rename.
fn rename_key(ctx: &RubricCtx<'_>, sink: &mut Sink<'_>, parent: &str, key: &str, allowed: &[&str]) {
    let Some(new) = suggestion(key, allowed) else {
        return;
    };
    if let Some(d) = sink.found.last_mut() {
        d.diag.label = Some(format!("did you mean `{new}`?"));
    }
    let path = join(parent, key);
    let replacement = serde_json::to_string(&new).unwrap_or_default();
    json_fix(
        ctx,
        sink,
        ctx.loc.spans.key(&path),
        replacement,
        &join(parent, &new),
        format!("rename `{key}` to `{new}`"),
    );
}

/// Label a misspelled value with the closest allowed one, and offer it.
fn fix_value(
    ctx: &RubricCtx<'_>,
    sink: &mut Sink<'_>,
    path: &str,
    value: &Value,
    allowed: &[&str],
) {
    let Value::String(s) = value else {
        return;
    };
    let Some(new) = suggestion(s, allowed) else {
        return;
    };
    if let Some(d) = sink.found.last_mut() {
        d.diag.label = Some(format!("did you mean \"{new}\"?"));
    }
    let replacement = serde_json::to_string(&new).unwrap_or_default();
    json_fix(
        ctx,
        sink,
        ctx.loc.spans.value(path),
        replacement,
        path,
        format!("write \"{new}\""),
    );
}

fn rubric_rules(rubric: &Value, ctx: &RubricCtx<'_>, sink: &mut Sink<'_>) {
    let Value::Object(top) = rubric else {
        return;
    };
    let mut unknown: Vec<&String> = top
        .keys()
        .filter(|k| !RUBRIC_KEYS.contains(&k.as_str()))
        .collect();
    unknown.sort();
    for k in unknown {
        if k.starts_with("x-") {
            continue;
        }
        sink.error(
            code::RUBRIC_KEY,
            At::Key(&join("", k)),
            format!(
                "unknown key {}{} a rubric holds {}, or x- extensions",
                repr_str(k),
                dym_or(k, RUBRIC_KEYS, ";"),
                sorted_list(RUBRIC_KEYS)
            ),
        );
        rename_key(ctx, sink, "", k, RUBRIC_KEYS);
    }
    scoring_rules(top, ctx, sink);
    let crits = top.get("criteria");
    let Some(Value::Array(crits)) = crits.filter(|c| truthy(Some(c))) else {
        if !truthy(top.get("extends")) {
            sink.error(
                code::NO_CRITERIA,
                At::Key("criteria"),
                "needs a non-empty criteria list (or extends)",
            );
        }
        return;
    };
    let mut ids: Vec<String> = Vec::new();
    for (n, c) in crits.iter().enumerate() {
        criterion_rules(top, n, c, &mut ids, ctx, sink);
    }
}

fn scoring_rules(top: &Map<String, Value>, ctx: &RubricCtx<'_>, sink: &mut Sink<'_>) {
    let Some(scoring) = top.get("scoring").filter(|v| !v.is_null()) else {
        return;
    };
    let Value::Object(scoring) = scoring else {
        sink.error(code::SCORING, At::Value("scoring"), "scoring is an object");
        return;
    };
    let mut extra: Vec<&String> = scoring
        .keys()
        .filter(|k| !SCORING_KEYS.contains(&k.as_str()))
        .collect();
    extra.sort();
    for k in extra {
        if k == "gates" {
            let any = if scoring.get("gates").and_then(Value::as_str) == Some("any") {
                "; \"any\", RewardKit's any-pass, has no rubric form, and a converter that meets it refuses"
            } else {
                ""
            };
            sink.error(
                code::SCORING,
                At::Key("scoring.gates"),
                format!("scoring.gates is gone: every gate must pass, always, so the key had one value, \"all\". Remove it{any}"),
            );
        } else if !k.starts_with("x-") {
            sink.error(
                code::RUBRIC_KEY,
                At::Key(&join("scoring", k)),
                format!(
                    "unknown key {} in scoring{} it holds {}, or x- extensions",
                    repr_str(k),
                    dym_or(k, SCORING_KEYS, ";"),
                    sorted_list(SCORING_KEYS)
                ),
            );
            rename_key(ctx, sink, "scoring", k, SCORING_KEYS);
        }
    }
    let method = scoring
        .get("method")
        .cloned()
        .unwrap_or(Value::String("points".into()));
    if !matches!(method.as_str(), Some("points" | "sum")) {
        sink.error(
            code::SCORING,
            At::Value("scoring.method"),
            "scoring.method is \"points\" (the default: earned points over the maximum positive points, clipped to [0, 1]) or \
\"sum\" (the raw, unclipped total)",
        );
        fix_value(ctx, sink, "scoring.method", &method, &["points", "sum"]);
    }
    let headline = scoring
        .get("headline")
        .cloned()
        .unwrap_or(Value::String("partial".into()));
    if !matches!(headline.as_str(), Some("partial" | "strict")) {
        sink.error(
            code::SCORING,
            At::Value("scoring.headline"),
            "scoring.headline is \"partial\" or \"strict\"",
        );
        fix_value(
            ctx,
            sink,
            "scoring.headline",
            &headline,
            &["partial", "strict"],
        );
    }
    if let Some(pt) = scoring.get("pass_threshold").filter(|v| !v.is_null()) {
        let bad = match num(pt) {
            None => true,
            Some(x) => x < 0.0 || (method.as_str() == Some("points") && x > 1.0),
        };
        if bad {
            sink.error(
                code::SCORING,
                At::Value("scoring.pass_threshold"),
                "scoring.pass_threshold is a fraction from 0 to 1 under method points, and a number of points, at least 0, under \
method sum",
            );
        }
    }
}

/// The path a criterion diagnostic points at: the key when the criterion
/// has it, else its id, else the criterion.
fn crit_path(c: &Map<String, Value>, base: &str, key: &str) -> String {
    if c.contains_key(key) {
        join(base, key)
    } else if c.contains_key("id") {
        join(base, "id")
    } else {
        base.to_string()
    }
}

fn criterion_rules(
    top: &Map<String, Value>,
    n: usize,
    c: &Value,
    ids: &mut Vec<String>,
    ctx: &RubricCtx<'_>,
    sink: &mut Sink<'_>,
) {
    let base = format!("criteria[{n}]");
    let Value::Object(c) = c else {
        sink.error(
            code::CRITERION,
            At::Value(&base),
            "criterion must be an object",
        );
        return;
    };
    let at = |key: &str| crit_path(c, &base, key);
    let cid = c.get("id");
    match cid {
        Some(Value::String(id)) if CRITERION_ID.is_match(id) => {
            if ids.contains(id) {
                sink.error(
                    code::CRITERION_ID,
                    At::Value(&at("id")),
                    format!("duplicate id {}", repr_str(id)),
                );
            } else {
                ids.push(id.clone());
            }
        }
        _ => {
            sink.error(
                code::CRITERION_ID,
                At::Value(&at("id")),
                "id must be lowercase letters, digits, and hyphens",
            );
        }
    }
    if let Some(remove) = c.get("remove") {
        let exact = remove == &Value::Bool(true) && c.len() == 2 && c.contains_key("id");
        if !exact {
            sink.error(
                code::REMOVAL,
                At::Value(&base),
                "a removal entry is exactly {\"id\": ..., \"remove\": true}",
            );
        }
        if !truthy(top.get("extends")) {
            sink.error(
                code::REMOVAL,
                At::Key(&at("remove")),
                "remove drops a criterion from a shared rubric, so it needs extends",
            );
        }
        return;
    }
    let mut unknown: Vec<&String> = c
        .keys()
        .filter(|k| !CRITERION_KEYS.contains(&k.as_str()))
        .collect();
    unknown.sort();
    for k in unknown {
        if k.starts_with("x-") {
            continue;
        }
        sink.error(
            code::RUBRIC_KEY,
            At::Key(&join(&base, k)),
            format!(
                "unknown key {}{} a criterion holds {}, or x- extensions",
                repr_str(k),
                dym_or(k, CRITERION_KEYS, ";"),
                sorted_list(CRITERION_KEYS)
            ),
        );
        rename_key(ctx, sink, &base, k, CRITERION_KEYS);
    }
    if !truthy(c.get("text")) {
        sink.error(code::CRITERION, At::Key(&at("text")), "text is required");
    }
    let gate = truthy(c.get("gate"));
    if !gate && !c.contains_key("points") {
        sink.error(
            code::CRITERION,
            At::Key(&at("points")),
            "set gate: true or give points",
        );
    } else if gate && c.contains_key("points") {
        sink.error(
            code::CRITERION,
            At::Key(&at("points")),
            "a criterion is a gate or carries points, not both",
        );
    }
    let judge = c.get("judge");
    let judge_str = judge.and_then(Value::as_str);
    if !judge_str.is_some_and(|j| JUDGES.contains(&j)) {
        sink.error(
            code::CRITERION,
            At::Value(&at("judge")),
            format!("judge is required: one of {}", sorted_repr(JUDGES)),
        );
        if let Some(j) = judge {
            fix_value(ctx, sink, &at("judge"), j, JUDGES);
        }
    }
    let checked = judge_str.is_some_and(|j| CHECKED_JUDGES.contains(&j));
    let model = judge_str.is_some_and(|j| MODEL_JUDGED.contains(&j));
    if checked && !truthy(c.get("check")) {
        sink.error(
            code::CRITERION,
            At::Key(&at("check")),
            format!(
                "judge '{}' needs a check (a test id, rule, world predicate, or replay recipe)",
                judge_str.unwrap_or_default()
            ),
        );
    }
    if model && !truthy(c.get("evidence")) {
        sink.error(
            code::CRITERION,
            At::Key(&at("evidence")),
            "a model-judged criterion needs an evidence list: what the judge reads, and all a verdict may cite",
        );
    }
    let levels = c.get("levels").filter(|v| !v.is_null());
    if let Some(l) = levels
        && !matches!(l, Value::Object(m) if !m.is_empty())
    {
        sink.error(
            code::CRITERION,
            At::Value(&at("levels")),
            "levels must map scores to anchor descriptions",
        );
    }
    let outcome = c.get("outcome").filter(|v| !v.is_null());
    if let Some(o) = outcome
        && !matches!(o.as_str(), Some("good" | "bad"))
    {
        sink.error(
            code::CRITERION,
            At::Value(&at("outcome")),
            "outcome is \"good\" (the default: the text states a good thing) or \"bad\" (the text states a bad thing); pass always \
means the task went as it should",
        );
        fix_value(ctx, sink, &at("outcome"), o, &["good", "bad"]);
    }
    if model
        && !c.contains_key("outcome")
        && let Some(p) = c.get("points").and_then(num)
        && p < 0.0
    {
        let shown = cid.map_or_else(|| "None".to_string(), repr);
        sink.warning(
            code::CRITERION,
            At::Value(&at("points")),
            format!(
                "{shown} is a draft-1 penalty, and its verdict word changed meaning: a model's pass used to charge the penalty, and now \
pass means the task went as it should, with negative points charged on fail. Its text names the bad outcome, so set \
\"outcome\": \"bad\""
            ),
        );
    }
    if let Some(cite) = c.get("cite") {
        let judge_shown = judge.map_or_else(|| "None".to_string(), str_of);
        if cite.as_str() != Some("independent") {
            sink.error(
                code::CITE,
                At::Value(&at("cite")),
                "cite is \"independent\": a pass then needs a citation the solver neither wrote nor ran",
            );
            fix_value(ctx, sink, &at("cite"), cite, &["independent"]);
        } else if !model {
            sink.error(
                code::CITE,
                At::Value(&at("cite")),
                format!("cite applies to criteria a model judges; the runtime decides a {judge_shown} criterion"),
            );
        } else if matches!(judge_str, Some("llm" | "vlm"))
            && !c
                .get("evidence")
                .and_then(Value::as_array)
                .is_some_and(|e| e.iter().any(independent_evidence))
        {
            sink.warning(
                code::CITE,
                At::Value(&at("cite")),
                format!(
                    "cite is \"independent\", but an {judge_shown} judge cites only its evidence, and every item here is the solver's: \
add tests or a runtime-recorded item (request-record, screenshots, video, a world stream), or judge it with an agent that can run \
the work"
                ),
            );
        }
    }
    if let Some(score) = c.get("score").filter(|v| !v.is_null()) {
        let ok = match score {
            Value::Object(m) if m.len() == 2 => {
                match (m.get("min").and_then(num), m.get("max").and_then(num)) {
                    (Some(lo), Some(hi)) => lo < hi,
                    _ => false,
                }
            }
            _ => false,
        };
        if !ok {
            sink.error(
                code::CONTINUOUS,
                At::Value(&at("score")),
                "score is a continuous range, {\"min\": <number>, \"max\": <number>}, with min below max",
            );
        }
        if levels.is_some() {
            sink.error(
                code::CONTINUOUS,
                At::Value(&at("score")),
                "a criterion has levels or a continuous score, not both",
            );
        }
        if gate {
            sink.error(
                code::CONTINUOUS,
                At::Value(&at("score")),
                "a gate passes or fails, so it has no continuous score; give the criterion points",
            );
        }
        if checked {
            sink.error(
                code::CONTINUOUS,
                At::Value(&at("score")),
                format!(
                    "a continuous score needs a model or human judge; the runtime decides a {} criterion as pass or fail, and float \
test results are deferred",
                    judge_str.unwrap_or_default()
                ),
            );
        }
    }
    if let Some(missing) = c.get("missing") {
        if missing.as_str() != Some("no-penalty") {
            sink.error(
                code::CONTINUOUS,
                At::Value(&at("missing")),
                "missing is \"no-penalty\"",
            );
            fix_value(ctx, sink, &at("missing"), missing, &["no-penalty"]);
        } else if outcome.and_then(Value::as_str) != Some("bad") {
            sink.warning(
                code::CONTINUOUS,
                At::Value(&at("missing")),
                "missing = \"no-penalty\" changes only outcome = \"bad\" criteria: a good-outcome criterion whose evidence is missing \
fails",
            );
        }
    }
    match c.get("evidence").filter(|v| !v.is_null()) {
        Some(Value::Array(items)) => {
            if let Some(config) = ctx.config {
                for (i, item) in items.iter().enumerate() {
                    if let Some((error, message)) = evidence_problem(item, config) {
                        let p = format!("{base}.evidence[{i}]");
                        if error {
                            sink.error(code::EVIDENCE, At::Value(&p), message);
                        } else {
                            sink.warning(code::EVIDENCE, At::Value(&p), message);
                        }
                    }
                }
            }
        }
        Some(_) => {
            sink.error(
                code::CRITERION,
                At::Value(&at("evidence")),
                "evidence is a list of items",
            );
        }
        None => {}
    }
    stated_rules(c, &base, ctx, sink);
}

fn stated_rules(c: &Map<String, Value>, base: &str, ctx: &RubricCtx<'_>, sink: &mut Sink<'_>) {
    let stated = c.get("stated").filter(|v| !v.is_null());
    let stated_path = join(base, "stated");
    let quotes: Vec<(String, &Value)> = match stated {
        Some(v @ (Value::String(_) | Value::Object(_))) => vec![(stated_path.clone(), v)],
        Some(Value::Array(items)) => items
            .iter()
            .enumerate()
            .map(|(i, v)| (format!("{stated_path}[{i}]"), v))
            .collect(),
        _ => Vec::new(),
    };
    let well_formed = !quotes.is_empty()
        && quotes
            .iter()
            .all(|(_, q)| matches!(q, Value::String(s) if !s.trim().is_empty()) || q.is_object());
    if stated.is_some() && !well_formed {
        sink.error(
            code::STATED,
            At::Value(&stated_path),
            "stated must be a quote, or a list of quotes, from what the agent is told",
        );
    } else if ctx.told.is_some() {
        for (path, q) in &quotes {
            match q {
                Value::Object(item) => stated_file(item, path, ctx, sink),
                Value::String(s) => {
                    let p = plain(s);
                    if !ctx.told_plain.iter().any(|t| t.contains(&p)) {
                        sink.error(
                            code::STATED,
                            At::Value(path),
                            format!(
                                "stated quote not found in what the agent is told: {}",
                                repr_str(prefix(s, 60))
                            ),
                        );
                    }
                }
                _ => {}
            }
        }
    }
    if stated.is_none() && !truthy(c.get("implicit")) {
        let shown = c.get("id").map_or_else(|| "None".to_string(), repr);
        let at = if c.contains_key("id") {
            join(base, "id")
        } else {
            base.to_string()
        };
        sink.warning(
            code::NOT_STATED,
            At::Value(&at),
            format!(
                "{shown} does not say where the agent is told about it: quote the instruction in stated, or explain in implicit why \
the agent should know"
            ),
        );
    }
}

/// A stated quote from a file the agent is told to read (the reference's
/// `_stated_file`).
fn stated_file(item: &Map<String, Value>, path: &str, ctx: &RubricCtx<'_>, sink: &mut Sink<'_>) {
    let (Some(Value::String(rel)), Some(Value::String(quote))) =
        (item.get("file"), item.get("quote"))
    else {
        sink.error(
            code::STATED,
            At::Value(path),
            "a stated file entry needs a file path and a quote",
        );
        return;
    };
    if quote.trim().is_empty() {
        sink.error(
            code::STATED,
            At::Value(path),
            "a stated file entry needs a file path and a quote",
        );
        return;
    }
    let parts = path_parts(rel);
    let file_path = join(path, "file");
    if parts.first().is_some_and(|p| PRIVATE_DIRS.contains(p)) {
        sink.error(
            code::STATED,
            At::Value(&file_path),
            format!("stated file {rel} is in a folder the agent never sees"),
        );
        return;
    }
    let name = parts.last().filter(|p| **p != "/").copied().unwrap_or("");
    if !ctx.told_plain.is_empty()
        && !ctx
            .told_plain
            .iter()
            .any(|t| t.contains(&name.to_lowercase()))
    {
        sink.warning(
            code::STATED,
            At::Value(&file_path),
            format!("the agent is never told about {name}; name the file in the instruction"),
        );
    }
    if !ctx.deep {
        return;
    }
    let f = ctx.dir.join(rel);
    if !f.is_file() {
        sink.error(
            code::STATED,
            At::Value(&file_path),
            format!("stated file {rel} not found in the package"),
        );
        return;
    }
    let Some(text) = read_text(&f) else {
        sink.warning(
            code::STATED,
            At::Value(&file_path),
            format!("cannot check a quote inside {rel}, which is not text; a reviewer must"),
        );
        return;
    };
    if !plain(&text).contains(&plain(quote)) {
        sink.error(
            code::STATED,
            At::Value(&join(path, "quote")),
            format!(
                "stated quote not found in {rel}: {}",
                repr_str(prefix(quote, 60))
            ),
        );
    }
}

/// `stages/<name>/verifier/rubric.json`, checked like the task's own, with
/// the stage's evidence resolved against what the task saves plus what the
/// stage's own outputs save (the reference's `check_stage_rubrics`).
pub(crate) fn check_stage_rubrics(
    config: &Value,
    dir: &Path,
    told: &[String],
    deep: bool,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for (name, stage) in table(config, &["stages"]) {
        let Value::Object(stage) = stage else {
            continue;
        };
        let rel = format!("stages/{name}/verifier/rubric.json");
        if !dir.join(&rel).is_file() {
            continue;
        }
        let Some((data, text)) = load_json(dir, &rel, &mut out) else {
            continue;
        };
        if !has_schema(&data, "rubric") {
            continue;
        }
        let mut sandbox = table(config, &["sandbox"]).clone();
        let (root_outputs, workdir) = saved_paths(config);
        let stage_outputs: Vec<Value> = match stage.get("outputs") {
            Some(Value::Array(items)) => items
                .iter()
                .filter(|o| o.is_string() || o.is_object())
                .cloned()
                .collect(),
            _ => Vec::new(),
        };
        if !stage_outputs.is_empty() {
            let mut base: Vec<Value> = match root_outputs.filter(|o| !o.is_empty()) {
                Some(o) => o.into_iter().map(Value::String).collect(),
                None => workdir
                    .filter(|w| !w.is_empty())
                    .into_iter()
                    .map(Value::String)
                    .collect(),
            };
            base.extend(stage_outputs);
            sandbox.insert("outputs".into(), Value::Array(base));
        }
        let mut stage_config = config.as_object().cloned().unwrap_or_default();
        stage_config.insert("sandbox".into(), Value::Object(sandbox));
        let judgment = Judgment {
            value: data,
            text,
            file: PathBuf::from(&rel),
        };
        out.extend(check_rubric(
            &judgment,
            Some(told),
            dir,
            Some(&Value::Object(stage_config)),
            deep,
        ));
    }
    out
}

/// The levels a severity map covers; 0 is "not detected".
const SCALE_0_3: &[&str] = &["1", "2", "3"];

fn consequence_ok(c: &Value) -> bool {
    match c {
        Value::String(s) => CONSEQUENCES.contains(&s.as_str()),
        Value::Object(m) => m.get("penalty").and_then(num).is_some_and(|p| p < 0.0),
        _ => false,
    }
}

fn severity_map(c: &Value) -> Option<&Map<String, Value>> {
    match c {
        Value::Object(m)
            if !m.is_empty()
                && !m.contains_key("penalty")
                && m.keys().all(|k| SEVERITY_KEY.is_match(k)) =>
        {
            Some(m)
        }
        _ => None,
    }
}

/// Check a behaviors file (the reference's `check_behaviors`).
pub(crate) fn check_behaviors(beh: &Judgment) -> Vec<Diagnostic> {
    let loc = JsonLoc::new(&beh.file, &beh.text);
    let mut sink = Sink::new(Some(&loc));
    behavior_rules(&beh.value, &mut sink);
    sink.into_diagnostics()
}

fn behavior_rules(beh: &Value, sink: &mut Sink<'_>) {
    let Value::Object(top) = beh else {
        return;
    };
    let Some(Value::Array(watch)) = top.get("watch").filter(|w| truthy(Some(w))) else {
        sink.error(
            code::BEHAVIOR,
            At::Key("watch"),
            "needs a non-empty watch list",
        );
        return;
    };
    for (n, b) in watch.iter().enumerate() {
        let base = format!("watch[{n}]");
        let Value::Object(b) = b else {
            sink.error(
                code::BEHAVIOR,
                At::Value(&base),
                "a watched behavior is an object",
            );
            continue;
        };
        let at = |key: &str| crit_path(b, &base, key);
        let bid = b.get("id").map(str_of).unwrap_or_default();
        if !BEHAVIOR_ID.is_match(&bid) {
            sink.error(
                code::BEHAVIOR,
                At::Value(&at("id")),
                "id must be a dotted taxonomy id such as process.overreach",
            );
        }
        if !truthy(b.get("definition")) {
            sink.error(
                code::BEHAVIOR,
                At::Key(&at("definition")),
                "definition is required: say what the behavior means in this task",
            );
        }
        let cons = b.get("consequence").cloned().unwrap_or(Value::Null);
        let scale = b
            .get("scale")
            .cloned()
            .unwrap_or(Value::String("binary".into()));
        let cons_at = at("consequence");
        if let Some(map) = severity_map(&cons) {
            match scale.as_str() {
                Some("binary") => {
                    sink.error(
                        code::CONSEQUENCE,
                        At::Value(&cons_at),
                        "a severity map needs scale = \"0-3\": a binary behavior is detected or not, so it carries one consequence",
                    );
                }
                Some("0-3") => {
                    if map.contains_key("0") {
                        sink.error(
                            code::CONSEQUENCE,
                            At::Key(&join(&cons_at, "0")),
                            "severity 0 means not detected, so it carries no consequence: map \"1\" to \"3\"",
                        );
                    }
                    let mut extra: Vec<&String> = map
                        .keys()
                        .filter(|k| !SCALE_0_3.contains(&k.as_str()) && *k != "0")
                        .collect();
                    extra.sort();
                    if !extra.is_empty() {
                        let shown: Vec<&str> = extra.iter().map(|s| s.as_str()).collect();
                        sink.error(
                            code::CONSEQUENCE,
                            At::Value(&cons_at),
                            format!("severity {} is not on the 0-3 scale", shown.join(", ")),
                        );
                    }
                    let missing: Vec<String> = SCALE_0_3
                        .iter()
                        .filter(|l| !map.contains_key(**l))
                        .map(|l| repr_str(l))
                        .collect();
                    if !missing.is_empty() {
                        sink.error(
                            code::CONSEQUENCE,
                            At::Value(&cons_at),
                            format!(
                                "the severity map must cover the whole 0-3 scale: add {}",
                                missing.join(", ")
                            ),
                        );
                    }
                    for (lvl, c) in map {
                        if !consequence_ok(c) {
                            sink.error(
                                code::CONSEQUENCE,
                                At::Value(&join(&cons_at, lvl)),
                                format!(
                                    "severity {lvl} maps to {}; a level's consequence is 'fail', 'invalid', 'report', or {{penalty: \
<negative number>}}",
                                    repr(c)
                                ),
                            );
                        }
                    }
                }
                _ => {}
            }
        } else if !consequence_ok(&cons) {
            sink.error(
                code::CONSEQUENCE,
                At::Value(&cons_at),
                "consequence must be 'fail', 'invalid', 'report', {penalty: <negative number>}, or, for a 0-3 behavior, a severity map \
such as {\"1\": \"report\", \"2\": {\"penalty\": -0.2}, \"3\": \"fail\"}",
            );
        }
        let detect = b.get("detect").filter(|v| !v.is_null());
        if detect
            .is_some_and(|d| !matches!(d.as_str(), Some("rule" | "world" | "monitor" | "human")))
        {
            sink.error(
                code::DETECT,
                At::Value(&at("detect")),
                "detect must be 'rule', 'world', 'monitor', or 'human'",
            );
        }
        if detect.and_then(Value::as_str) == Some("rule") && !bid.starts_with("integrity.") {
            let shown = if bid.is_empty() {
                "this behavior"
            } else {
                bid.as_str()
            };
            sink.error(
                code::DETECT,
                At::Value(&at("detect")),
                format!(
                    "detect = \"rule\" is for the integrity family, whose rules are built into the detector (docs/behaviors.md, \
Protected paths): a task cannot state a rule of its own yet. Detect {shown} with \"monitor\", from its definition, or with \"world\""
                ),
            );
        }
        let outcomes: Vec<&Value> = match severity_map(&cons) {
            Some(m) => m.values().collect(),
            None => vec![&cons],
        };
        let scope = b
            .get("scope")
            .cloned()
            .unwrap_or(Value::String("trial".into()));
        if outcomes.iter().any(|o| o.as_str() == Some("invalid"))
            && scope.as_str() == Some("trial")
            && !bid.starts_with("infra.")
        {
            sink.error(
                code::CONSEQUENCE,
                At::Value(&cons_at),
                "consequence \"invalid\" takes a trial out of the count, which only a failure outside the agent may do: it is for infra.* \
behaviors, and for scope = \"submission\", where the whole submission is invalid. The solver's own conduct is graded: use \"fail\"",
            );
        }
        if let Some(judge) = b.get("judge") {
            if detect.and_then(Value::as_str) != Some("monitor") {
                sink.error(
                    code::DETECT,
                    At::Key(&at("judge")),
                    "judge applies only to behaviors with detect = \"monitor\", which a model tags",
                );
            } else if !matches!(judge.as_str(), Some(j) if MONITOR_JUDGES.contains(&j)) {
                sink.error(
                    code::DETECT,
                    At::Value(&at("judge")),
                    "judge is \"llm\" (the default), \"vlm\", or \"agent\"",
                );
            }
        }
        if !matches!(scope.as_str(), Some("trial" | "submission")) {
            sink.error(
                code::BEHAVIOR,
                At::Value(&at("scope")),
                "scope must be 'trial' or 'submission'",
            );
        }
        if !matches!(scale.as_str(), Some("binary" | "0-3")) {
            sink.error(
                code::BEHAVIOR,
                At::Value(&at("scale")),
                "scale must be 'binary' or '0-3'",
            );
        }
        if let Some(paths) = b.get("paths").filter(|v| !v.is_null()) {
            let ok = matches!(paths, Value::Array(p) if !p.is_empty() && p.iter().all(|x| x.as_str().is_some_and(|s| s.starts_with('/'))));
            if !ok {
                sink.error(
                    code::BEHAVIOR,
                    At::Value(&at("paths")),
                    "paths must be a non-empty list of absolute paths, as the agent's sandbox names them",
                );
            }
        }
        let mut unknown: Vec<&String> = b
            .keys()
            .filter(|k| !WATCH_KEYS.contains(&k.as_str()))
            .collect();
        unknown.sort();
        for k in unknown {
            if k.starts_with("x-") {
                continue;
            }
            sink.error(
                code::RUBRIC_KEY,
                At::Key(&join(&base, k)),
                format!(
                    "unknown key {}{} a watched behavior has {}, or x- extensions",
                    repr_str(k),
                    dym_or(k, WATCH_KEYS, ";"),
                    sorted_list(WATCH_KEYS)
                ),
            )
            .label = suggestion(k, WATCH_KEYS).map(|s| format!("did you mean `{s}`?"));
        }
    }
    const TOP: &[&str] = &["$schema", "taxonomy", "watch", "paired"];
    let mut unknown: Vec<&String> = top.keys().filter(|k| !TOP.contains(&k.as_str())).collect();
    unknown.sort();
    for k in unknown {
        if k.starts_with("x-") {
            continue;
        }
        sink.error(
            code::RUBRIC_KEY,
            At::Key(&join("", k)),
            format!("unknown key {}{}", repr_str(k), dym_or(k, TOP, "")),
        );
    }
    if let Some(Value::Array(paired)) = top.get("paired") {
        for (n, b) in paired.iter().enumerate() {
            let base = format!("paired[{n}]");
            let id = b.get("id").map(str_of).unwrap_or_default();
            if !BEHAVIOR_ID.is_match(&id) || !truthy(b.get("definition")) {
                sink.error(
                    code::BEHAVIOR,
                    At::Value(&base),
                    "a paired behavior needs a taxonomy id and a definition",
                );
            }
            let cmp_ok = matches!(b.get("compare"), Some(Value::Array(a)) if a.len() == 2 && a.iter().all(Value::is_string));
            if !cmp_ok {
                sink.error(
                    code::BEHAVIOR,
                    At::Value(&join(&base, "compare")),
                    "compare must name two cells, such as [\"honeypot\", \"base\"]",
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn plain_text_ignores_markers_spacing_and_case() {
        assert_eq!(plain("Fix **the**\n  `bug`_now_"), "fix the bugnow");
    }

    #[test]
    fn normpath_like_posixpath() {
        assert_eq!(normpath("/app/./out/../x"), "/app/x");
        assert_eq!(normpath("//a//b"), "//a/b");
        assert_eq!(normpath("///a"), "/a");
        assert_eq!(normpath("a/../../b"), "../b");
        assert_eq!(normpath(""), ".");
        assert_eq!(normpath("/.."), "/");
    }

    #[test]
    fn evidence_resolves_against_saved_paths() {
        let cfg = json!({"sandbox": {"workdir": "/app", "outputs": [{"path": "/app/out"}, {"path": "/app/fit.json"}]}});
        assert!(evidence_problem(&json!("out/report.md"), &cfg).is_none());
        assert!(evidence_problem(&json!("diff:/app/out/fit.py"), &cfg).is_none());
        let (err, msg) = evidence_problem(&json!("report.md"), &cfg).unwrap();
        assert!(
            err && msg.contains("reads /app/report.md, which no [sandbox] outputs entry saves"),
            "{msg}"
        );
        let (err, msg) = evidence_problem(&json!("trajectory:steps"), &cfg).unwrap();
        assert!(err && msg.contains("trajectory takes no argument"));
        let wd = json!({"sandbox": {"workdir": "/app"}});
        assert!(
            evidence_problem(&json!("/etc/passwd"), &wd)
                .unwrap()
                .1
                .contains("outside [sandbox] workdir /app")
        );
        assert!(!evidence_problem(&json!("/x"), &json!({})).unwrap().0);
    }

    #[test]
    fn independent_items() {
        assert!(independent_evidence(&json!("tests")));
        assert!(independent_evidence(&json!("camera")));
        assert!(!independent_evidence(&json!("report.md")));
        assert!(!independent_evidence(&json!("trajectory")));
        assert!(!independent_evidence(&json!("diff:a.py")));
    }
}
