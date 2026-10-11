//! The builtin loop's tools, run in the sandbox as the agent user.
//!
//! The tool set and the result shapes follow Karotte's builtin agent
//! (Preference Model, MIT License): `bash` returns `stdout`, `stderr`,
//! `exit_code` and a `system` note, each stream cut at
//! [`MAX_OUTPUT_CHARS`] characters; `view` returns lines of a file;
//! `replace` is an exact string replacement. Two differences: each `bash`
//! call is its own shell started in the working folder (the sandbox API
//! runs commands to completion), and `replace` refuses an ambiguous match
//! unless `replace_all` is set, where Karotte replaced the first one.

use std::collections::BTreeMap;
use std::time::Duration;

use base64::Engine;
use benchflow_sandbox::{Exec, ExecOutput, Sandbox, SandboxError};
use serde_json::{Map, Value, json};

use crate::native::chat::ToolSpec;

/// Characters of stdout and of stderr kept in a `bash` result (Karotte's
/// `max_output_length`).
pub const MAX_OUTPUT_CHARS: usize = 16_000;
/// Most lines one `view` call returns (Karotte's `max_lines`).
pub const MAX_VIEW_LINES: u64 = 1000;
/// Most bytes of file content one `view` call returns.
const MAX_VIEW_BYTES: usize = 384 * 1024;
/// Largest file `replace` edits (Karotte's limit).
const MAX_EDIT_BYTES: usize = 10 * 1024 * 1024;
/// Base64 characters per command when a file is written in pieces.
const WRITE_CHUNK: usize = 65_536;
/// How long the file tools' own commands may take.
const FILE_OP_TIMEOUT: Duration = Duration::from_secs(60);

/// Where and as whom the tools run.
pub struct ToolContext<'a> {
    pub sandbox: &'a dyn Sandbox,
    pub user: &'a str,
    pub workdir: &'a str,
    pub env: &'a BTreeMap<String, String>,
    /// Default and longest per-call `bash` timeout.
    pub bash_timeout: Duration,
}

/// What a tool call produced.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolOutcome {
    /// The structured result (the trajectory's `raw_output`).
    pub result: Value,
    /// The text the model reads.
    pub text: String,
    /// The tool reported a failure (the model sees it and can recover).
    pub is_error: bool,
    /// `submit`: the turn is over.
    pub ends_turn: bool,
}

impl ToolOutcome {
    fn ok(result: Value) -> Self {
        let text = result_text(&result);
        Self {
            result,
            text,
            is_error: false,
            ends_turn: false,
        }
    }
    fn error(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            result: json!({"error": message}),
            text: format!("Error: {message}"),
            is_error: true,
            ends_turn: false,
        }
    }
}

/// The text the model reads for a structured result: the JSON object, as
/// Karotte hands MCP structured content to the model.
fn result_text(result: &Value) -> String {
    serde_json::to_string(result).unwrap_or_default()
}

/// The tools, as the model sees them.
pub fn specs(bash_timeout: Duration) -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "bash".into(),
            description: format!(
                "Execute a Bash `command` in the task's working folder. Each call runs in a fresh shell: `cd` and variables do not carry over to the next call, but files do. `timeout_s` is the maximum time in seconds to wait for the command to complete; it cannot be extended beyond the default ({}s). Output longer than {MAX_OUTPUT_CHARS} characters per stream is truncated.",
                bash_timeout.as_secs()
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The Bash command to run."},
                    "timeout_s": {"type": "number", "description": "Seconds to wait before the command is stopped."}
                },
                "required": ["command"]
            }),
        },
        ToolSpec {
            name: "view".into(),
            description: format!(
                "Read lines from a file. Lines are 1-indexed (first line is 1). Both from_line and to_line are inclusive. You can read up to {MAX_VIEW_LINES} lines at a time. For larger files, make multiple calls with different ranges. To read the beginning of a file, use from_line=1. To continue reading, use from_line=(previous to_line + 1). A relative file_path is taken from the working folder."
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "from_line": {"type": "integer", "minimum": 1},
                    "to_line": {"type": "integer", "minimum": 1}
                },
                "required": ["file_path", "from_line", "to_line"]
            }),
        },
        ToolSpec {
            name: "replace".into(),
            description: "Perform exact string replacement in a file. The replacement will FAIL if `old` is not found in the file, or if it occurs more than once and `replace_all` is not set; then include more surrounding text to make the match unique. Use `view` to see the exact file contents, including whitespace and indentation, before attempting the replacement. When replacing code, preserve the exact indentation (tabs/spaces) as it appears in the file.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old": {"type": "string", "description": "The exact text to replace."},
                    "new": {"type": "string", "description": "The text to put in its place."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence."}
                },
                "required": ["file_path", "old", "new"]
            }),
        },
        ToolSpec {
            name: "write".into(),
            description: "Write `content` to a file, replacing it if it exists and creating it and its folders if not. The content is written exactly as given.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["file_path", "content"]
            }),
        },
        ToolSpec {
            name: "submit".into(),
            description: "Call this when the task is done. It ends your turn; nothing you do after it runs.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "A short note on what you did."}
                }
            }),
        },
    ]
}

/// The trajectory kind and a title for a call.
pub fn describe(name: &str, args: &Map<String, Value>) -> (&'static str, String) {
    let s = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or_default();
    match name {
        "bash" => ("execute", s("command").to_string()),
        "view" => ("read", format!("view {}", s("file_path"))),
        "replace" => ("edit", format!("replace in {}", s("file_path"))),
        "write" => ("edit", format!("write {}", s("file_path"))),
        "submit" => ("other", "submit".to_string()),
        other => ("other", other.to_string()),
    }
}

/// Run one tool call. `Err` is a sandbox failure (the run cannot go on);
/// a tool's own failure is an `Ok` outcome the model reads.
pub async fn run(
    ctx: &ToolContext<'_>,
    name: &str,
    args: &Map<String, Value>,
) -> Result<ToolOutcome, SandboxError> {
    match name {
        "bash" => bash(ctx, args).await,
        "view" => view(ctx, args).await,
        "replace" => replace(ctx, args).await,
        "write" => write(ctx, args).await,
        "submit" => Ok(ToolOutcome {
            result: json!({"result": "submitted"}),
            text: "Submitted.".into(),
            is_error: false,
            ends_turn: true,
        }),
        other => Ok(ToolOutcome::error(format!(
            "there is no tool `{other}`; the tools are bash, view, replace, write, and submit"
        ))),
    }
}

fn string_arg<'a>(args: &'a Map<String, Value>, key: &str) -> Result<&'a str, ToolOutcome> {
    args.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| ToolOutcome::error(format!("missing the string argument `{key}`")))
}

fn int_arg(args: &Map<String, Value>, key: &str) -> Option<i64> {
    match args.get(key)? {
        Value::Number(n) => n.as_i64().or_else(|| n.as_f64().map(|f| f as i64)),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

/// An absolute path: as given, or under the working folder.
fn resolve(ctx: &ToolContext<'_>, path: &str) -> String {
    if path.starts_with('/') {
        path.to_string()
    } else {
        format!(
            "{}/{}",
            ctx.workdir.trim_end_matches('/'),
            path.trim_start_matches("./")
        )
    }
}

fn q(word: &str) -> String {
    benchflow_sandbox::shell::quote(word)
}

async fn exec_as_agent(
    ctx: &ToolContext<'_>,
    command: String,
    timeout: Duration,
) -> Result<ExecOutput, SandboxError> {
    ctx.sandbox
        .exec(
            Exec::new(command)
                .user(ctx.user.to_string())
                .cwd(ctx.workdir.to_string())
                .timeout(timeout),
        )
        .await
}

/// Cut `s` to `max` characters, as Karotte does (`...` marks the cut).
fn truncate_chars(s: &str, max: usize) -> (String, bool) {
    match s.char_indices().nth(max) {
        Some((i, _)) => (format!("{}...", &s[..i]), true),
        None => (s.to_string(), false),
    }
}

async fn bash(
    ctx: &ToolContext<'_>,
    args: &Map<String, Value>,
) -> Result<ToolOutcome, SandboxError> {
    let command = match string_arg(args, "command") {
        Ok(c) if !c.trim().is_empty() => c.trim_end().to_string(),
        Ok(_) => return Ok(ToolOutcome::error("No command provided.")),
        Err(e) => return Ok(e),
    };
    let max = ctx.bash_timeout;
    let timeout = capped_timeout(args.get("timeout_s").and_then(Value::as_f64), max);
    let exec = Exec::new(command)
        .user(ctx.user.to_string())
        .cwd(ctx.workdir.to_string())
        .envs(ctx.env.clone())
        .timeout(timeout);
    let out = ctx.sandbox.exec(exec).await?;
    let mut system = String::new();
    let (stdout, cut) = truncate_chars(&out.stdout, MAX_OUTPUT_CHARS);
    if cut {
        system.push_str(&format!(
            "stdout was truncated to {MAX_OUTPUT_CHARS} characters."
        ));
    }
    let (mut stderr, cut) = truncate_chars(&out.stderr, MAX_OUTPUT_CHARS);
    if cut {
        system.push_str(&format!(
            "stderr was truncated to {MAX_OUTPUT_CHARS} characters."
        ));
    }
    let mut result = Map::new();
    result.insert("stdout".into(), stdout.into());
    if out.timed_out {
        let note = format!(
            "[Interrupted due to timeout after {}s]",
            timeout.as_secs_f64()
        );
        stderr = if stderr.is_empty() {
            note
        } else {
            format!("{stderr}\n{note}")
        };
    }
    if !stderr.is_empty() {
        result.insert("stderr".into(), stderr.into());
    }
    if !system.is_empty() {
        result.insert("system".into(), system.into());
    }
    if !out.timed_out {
        result.insert("exit_code".into(), out.exit_code.into());
    }
    Ok(ToolOutcome::ok(Value::Object(result)))
}

async fn view(
    ctx: &ToolContext<'_>,
    args: &Map<String, Value>,
) -> Result<ToolOutcome, SandboxError> {
    let path = match string_arg(args, "file_path") {
        Ok(p) => resolve(ctx, p),
        Err(e) => return Ok(e),
    };
    let from = int_arg(args, "from_line").unwrap_or(1);
    let to = int_arg(args, "to_line").unwrap_or(from.saturating_add(MAX_VIEW_LINES as i64 - 1));
    if from < 1 {
        return Ok(ToolOutcome::error(
            "`from_line` must be greater than or equal to 1",
        ));
    }
    if to < from {
        return Ok(ToolOutcome::error(
            "`to_line` must be greater than or equal to `from_line`",
        ));
    }
    let requested = (to - from + 1) as u64;
    if requested > MAX_VIEW_LINES {
        return Ok(ToolOutcome::error(format!(
            "Requested {requested} lines, but the limit is {MAX_VIEW_LINES} lines per call"
        )));
    }
    let p = q(&path);
    let script = format!(
        "if [ ! -e {p} ]; then echo 'BF_NOT_FOUND' >&2; exit 3; fi; \
         if [ ! -f {p} ]; then echo 'BF_NOT_FILE' >&2; exit 3; fi; \
         if [ ! -r {p} ]; then echo 'BF_DENIED' >&2; exit 3; fi; \
         sed -n '{from},{to}p;{next}q' {p} | head -c {cap}; \
         echo; echo \"BF_LINES $(wc -l < {p})\" >&2",
        next = to.saturating_add(1),
        cap = MAX_VIEW_BYTES + 1
    );
    let out = exec_as_agent(ctx, script, FILE_OP_TIMEOUT).await?;
    if let Some(e) = file_error(&out, &path) {
        return Ok(e);
    }
    // `echo` added one newline so a last line without one is kept intact.
    let mut content = out
        .stdout
        .strip_suffix('\n')
        .unwrap_or(&out.stdout)
        .to_string();
    let mut truncated = false;
    if content.len() > MAX_VIEW_BYTES {
        let mut cut = MAX_VIEW_BYTES;
        while !content.is_char_boundary(cut) {
            cut -= 1;
        }
        content.truncate(cut);
        content.push_str(&format!(
            "\n\n[... output truncated: exceeded {MAX_VIEW_BYTES} bytes; request a narrower line range ...]"
        ));
        truncated = true;
    }
    if !truncated && content.is_empty() && from > 1 {
        let total = out
            .stderr
            .lines()
            .find_map(|l| l.strip_prefix("BF_LINES "))
            .map(|n| format!(" has {} lines", n.trim()))
            .unwrap_or_default();
        return Ok(ToolOutcome::error(format!(
            "File {path}{total}, but you requested lines {from}-{to}"
        )));
    }
    Ok(ToolOutcome::ok(json!({
        "file_path": path, "from_line": from, "to_line": to, "content": content
    })))
}

/// A file test's failure, as the model reads it.
fn file_error(out: &ExecOutput, path: &str) -> Option<ToolOutcome> {
    if out.success() {
        return None;
    }
    let msg = if out.stderr.contains("BF_NOT_FOUND") {
        format!("File not found: {path}")
    } else if out.stderr.contains("BF_NOT_FILE") {
        format!("Not a regular file: {path}")
    } else if out.stderr.contains("BF_DENIED") {
        format!("Permission denied: {path}")
    } else if out.timed_out {
        format!("Timed out reading {path}")
    } else {
        format!("Failed to read {path}: {}", out.stderr.trim())
    };
    Some(ToolOutcome::error(msg))
}

/// Read a whole file as the agent user, exactly (base64 in transit).
async fn read_file(
    ctx: &ToolContext<'_>,
    path: &str,
) -> Result<Result<Vec<u8>, ToolOutcome>, SandboxError> {
    let p = q(path);
    let script = format!(
        "if [ ! -e {p} ]; then echo 'BF_NOT_FOUND' >&2; exit 3; fi; \
         if [ ! -f {p} ]; then echo 'BF_NOT_FILE' >&2; exit 3; fi; \
         if [ ! -r {p} ]; then echo 'BF_DENIED' >&2; exit 3; fi; \
         if [ ! -w {p} ]; then echo 'BF_NO_WRITE' >&2; exit 3; fi; \
         size=$(wc -c < {p}); if [ \"$size\" -gt {MAX_EDIT_BYTES} ]; then echo \"BF_TOO_BIG $size\" >&2; exit 3; fi; \
         base64 < {p} | tr -d '\\n'"
    );
    let out = exec_as_agent(ctx, script, FILE_OP_TIMEOUT).await?;
    if out.stderr.contains("BF_NO_WRITE") {
        return Ok(Err(ToolOutcome::error(format!(
            "No write permission for file: {path}"
        ))));
    }
    if let Some(size) = out
        .stderr
        .lines()
        .find_map(|l| l.strip_prefix("BF_TOO_BIG "))
    {
        return Ok(Err(ToolOutcome::error(format!(
            "File too large to edit in place: {path} is {} bytes (limit {MAX_EDIT_BYTES} bytes)",
            size.trim()
        ))));
    }
    if let Some(e) = file_error(&out, path) {
        return Ok(Err(e));
    }
    match base64::engine::general_purpose::STANDARD.decode(out.stdout.trim()) {
        Ok(bytes) => Ok(Ok(bytes)),
        Err(e) => Ok(Err(ToolOutcome::error(format!(
            "Failed to read {path}: {e}"
        )))),
    }
}

/// Write bytes to a file as the agent user, keeping an existing file's
/// owner and mode. Large contents go through a staging file in pieces.
async fn write_bytes(
    ctx: &ToolContext<'_>,
    path: &str,
    bytes: &[u8],
) -> Result<Result<(), String>, SandboxError> {
    let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
    let p = q(path);
    let dir = match path.rsplit_once('/') {
        Some((d, _)) if !d.is_empty() => d.to_string(),
        _ => "/".to_string(),
    };
    let mkdir = format!("mkdir -p {}", q(&dir));
    let out = if b64.len() <= WRITE_CHUNK {
        exec_as_agent(
            ctx,
            format!("{mkdir} && printf %s {b} | base64 -d > {p}", b = q(&b64)),
            FILE_OP_TIMEOUT,
        )
        .await?
    } else {
        let staging = format!("/tmp/.benchflow-write-{}", uuid::Uuid::new_v4().simple());
        let s = q(&staging);
        for (i, chunk) in b64.as_bytes().chunks(WRITE_CHUNK).enumerate() {
            let chunk = std::str::from_utf8(chunk).unwrap_or_default();
            let redirect = if i == 0 { ">" } else { ">>" };
            let out = exec_as_agent(
                ctx,
                format!("printf %s {} {redirect} {s}", q(chunk)),
                FILE_OP_TIMEOUT,
            )
            .await?;
            if !out.success() {
                let _ = exec_as_agent(ctx, format!("rm -f {s}"), FILE_OP_TIMEOUT).await;
                return Ok(Err(out.combined()));
            }
        }
        exec_as_agent(
            ctx,
            format!("{mkdir} && base64 -d < {s} > {p}; rc=$?; rm -f {s}; exit $rc"),
            FILE_OP_TIMEOUT,
        )
        .await?
    };
    if out.success() {
        Ok(Ok(()))
    } else {
        Ok(Err(out.combined()))
    }
}

async fn replace(
    ctx: &ToolContext<'_>,
    args: &Map<String, Value>,
) -> Result<ToolOutcome, SandboxError> {
    let (path, old, new) = match (
        string_arg(args, "file_path"),
        string_arg(args, "old"),
        string_arg(args, "new"),
    ) {
        (Ok(p), Ok(o), Ok(n)) => (resolve(ctx, p), o, n),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return Ok(e),
    };
    let replace_all = args
        .get("replace_all")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if old.is_empty() {
        return Ok(ToolOutcome::error("String to replace must not be empty"));
    }
    let bytes = match read_file(ctx, &path).await? {
        Ok(b) => b,
        Err(e) => return Ok(e),
    };
    let Ok(content) = String::from_utf8(bytes) else {
        return Ok(ToolOutcome::error(format!(
            "{path} is not UTF-8 text; edit it with bash instead"
        )));
    };
    let occurrences = content.matches(old).count();
    if occurrences == 0 {
        return Ok(ToolOutcome::error(format!(
            "String not found in file: {path}"
        )));
    }
    if occurrences > 1 && !replace_all {
        return Ok(ToolOutcome::error(format!(
            "String occurs {occurrences} times in {path}; include more surrounding text to make it unique, or set replace_all to replace every occurrence"
        )));
    }
    let replaced = if replace_all {
        content.replace(old, new)
    } else {
        content.replacen(old, new, 1)
    };
    if replaced.len() > MAX_EDIT_BYTES {
        return Ok(ToolOutcome::error(format!(
            "Replacement would grow {path} to ~{} bytes, over the {MAX_EDIT_BYTES}-byte limit. Narrow the match or replace in smaller steps.",
            replaced.len()
        )));
    }
    if let Err(e) = write_bytes(ctx, &path, replaced.as_bytes()).await? {
        return Ok(ToolOutcome::error(format!(
            "Failed to write file {path}: {e}"
        )));
    }
    Ok(ToolOutcome::ok(json!({
        "result": "Replacement successful",
        "replacements": occurrences,
        "diff": snippet_diff(&content, old, new, &path),
    })))
}

/// A short unified-style view of the first replacement: the changed lines
/// with three lines of context, numbered as in the old file.
fn snippet_diff(content: &str, old: &str, new: &str, path: &str) -> String {
    let Some(at) = content.find(old) else {
        return String::new();
    };
    let first_line = content[..at].matches('\n').count();
    let old_lines = old.matches('\n').count() + 1;
    let lines: Vec<&str> = content.split('\n').collect();
    let start = first_line.saturating_sub(3);
    let end = (first_line + old_lines + 3).min(lines.len());
    let mut out = format!("--- {path}\n+++ {path}\n@@ -{} @@\n", start + 1);
    for l in &lines[start..first_line] {
        out.push_str(&format!(" {l}\n"));
    }
    let line_start = content[..at].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let old_end = at + old.len();
    let line_end = content[old_end..]
        .find('\n')
        .map(|i| old_end + i)
        .unwrap_or(content.len());
    for l in content[line_start..line_end].split('\n') {
        out.push_str(&format!("-{l}\n"));
    }
    let new_block = format!(
        "{}{}{}",
        &content[line_start..at],
        new,
        &content[old_end..line_end]
    );
    for l in new_block.split('\n') {
        out.push_str(&format!("+{l}\n"));
    }
    let after = (first_line + old_lines).min(lines.len());
    for l in &lines[after..end] {
        out.push_str(&format!(" {l}\n"));
    }
    if out.len() > 64 * 1024 {
        let mut cut = 64 * 1024;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
        out.push_str("\n(diff truncated)\n");
    }
    out
}

async fn write(
    ctx: &ToolContext<'_>,
    args: &Map<String, Value>,
) -> Result<ToolOutcome, SandboxError> {
    let (path, content) = match (string_arg(args, "file_path"), string_arg(args, "content")) {
        (Ok(p), Ok(c)) => (resolve(ctx, p), c),
        (Err(e), _) | (_, Err(e)) => return Ok(e),
    };
    if let Err(e) = write_bytes(ctx, &path, content.as_bytes()).await? {
        return Ok(ToolOutcome::error(format!(
            "Failed to write file {path}: {e}"
        )));
    }
    Ok(ToolOutcome::ok(json!({
        "result": "File written",
        "file_path": path,
        "bytes": content.len(),
    })))
}

/// The bash tool's time limit: the model's `timeout_s` up to `max`. A
/// number too big for a Duration (1e300) means `max`; zero, negative or
/// not a number means `max` too.
fn capped_timeout(asked: Option<f64>, max: Duration) -> Duration {
    match asked {
        Some(t) if t.is_finite() && t > 0.0 => {
            Duration::try_from_secs_f64(t).map_or(max, |d| d.min(max))
        }
        _ => max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_output_is_cut_with_a_marker() {
        let (s, cut) = truncate_chars(&"é".repeat(10), 4);
        assert_eq!(s, "éééé...");
        assert!(cut);
        assert_eq!(truncate_chars("abc", 4), ("abc".to_string(), false));
    }

    #[test]
    fn the_diff_shows_the_changed_line_in_context() {
        let d = snippet_diff("a\nb\nhello world\nc\n", "hello", "goodbye", "/f");
        assert!(d.contains("-hello world\n+goodbye world\n"), "{d}");
        assert!(d.contains(" b\n"), "{d}");
    }

    /// Guards the D5 fix: a model's `"timeout_s": 1e300` panicked in
    /// `Duration::from_secs_f64` before the cap applied.
    #[test]
    fn a_huge_bash_timeout_is_the_cap() {
        let max = Duration::from_secs(600);
        assert_eq!(capped_timeout(Some(1e300), max), max);
        assert_eq!(capped_timeout(Some(30.0), max), Duration::from_secs(30));
        assert_eq!(capped_timeout(Some(-1.0), max), max);
        assert_eq!(capped_timeout(None, max), max);
    }

    #[test]
    fn every_tool_has_a_schema_with_its_required_arguments() {
        let specs = specs(Duration::from_secs(600));
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["bash", "view", "replace", "write", "submit"]);
        assert!(specs[0].description.contains("600s"));
        assert_eq!(
            specs[2].parameters["required"],
            json!(["file_path", "old", "new"])
        );
    }
}
