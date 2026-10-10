//! What the package holds beyond task.md: something to run the agent in,
//! something that grades it, controls, scripts that call a model or name
//! their seat, a Compose file, a family's generator, and canaries.

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;
use walkdir::WalkDir;

use crate::diag::Diagnostic;
use crate::source::join;

use super::TaskDocument;
use super::code;
use super::judgment;
use super::pyfmt::{repr_str, str_of};
use super::pyval::{nonblank, py_eq, table, truthy};
use super::report::{At, ConfigLoc, FileLoc, Sink};
use super::text::read_text_lossy;
use super::vocab::{HOST_VAR, PROVIDER_VAR, SHARED_CANARY_GUIDS};

static HOST_VAR_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(HOST_VAR).unwrap());
static PROVIDER_VAR_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(PROVIDER_VAR).unwrap());
static RUNTIME_VAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)").unwrap());
static COMPOSE_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"privileged:\s*true|docker\.sock|network_mode:\s*["']?host|pid:\s*["']?host"#)
        .unwrap()
});
static PLACEHOLDER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{\{ *([A-Za-z_][A-Za-z0-9_]*) *\}\}").unwrap());

/// Variables a runtime sets in a Compose file's environment, naming its own
/// folders: a mount from one stays inside what the runtime owns.
const RUNTIME_COMPOSE_VARS: &[&str] = &[
    "HOST_VERIFIER_LOGS_PATH",
    "HOST_AGENT_LOGS_PATH",
    "HOST_ARTIFACTS_PATH",
    "CONTEXT_DIR",
    "T_BENCH_TASK_LOGS_PATH",
    "T_BENCH_TASK_AGENT_LOGS_PATH",
];

/// Compose settings that weaken a container's isolation from the host.
const COMPOSE_HOST_KEYS: &[(&str, &str)] = &[
    ("privileged", "privileged mode"),
    ("network_mode", "the host's network"),
    ("pid", "the host's process namespace"),
    ("ipc", "the host's IPC namespace"),
    ("userns_mode", "the host's user namespace"),
    ("cap_add", "added capabilities"),
    ("devices", "host devices"),
    ("security_opt", "security options"),
    ("cgroup_parent", "a host cgroup"),
];

/// Every env table in the config with its path: `[sandbox] env`, `[verifier]
/// env`, a stage's, a service's (the reference's `_env_tables`). Returns
/// (the dotted path the reference prints, the config path, the table).
fn env_tables<'a>(
    v: &'a Value,
    shown: &str,
    path: &str,
    out: &mut Vec<(String, String, &'a serde_json::Map<String, Value>)>,
) {
    match v {
        Value::Object(m) => {
            for (k, x) in m {
                let s = if shown.is_empty() {
                    k.clone()
                } else {
                    format!("{shown}.{k}")
                };
                let p = join(path, k);
                match x {
                    Value::Object(env) if k == "env" => out.push((s, p, env)),
                    _ => env_tables(x, &s, &p, out),
                }
            }
        }
        Value::Array(items) => {
            for (i, x) in items.iter().enumerate() {
                env_tables(x, &format!("{shown}.{i}"), &format!("{path}[{i}]"), out);
            }
        }
        _ => {}
    }
}

/// A package must not reach the operator's host: values resolved from the
/// host, and Compose settings that weaken isolation (the reference's
/// `check_host_boundary`). The Compose file is read only when `deep`.
pub(crate) fn host_boundary(
    config: &Value,
    dir: &Path,
    loc: &ConfigLoc<'_>,
    deep: bool,
) -> Vec<Diagnostic> {
    let mut sink = Sink::new(Some(loc));
    let mut tables = Vec::new();
    env_tables(config, "", "", &mut tables);
    for (shown, path, env) in tables {
        for (name, value) in env {
            for caps in HOST_VAR_RE.captures_iter(&str_of(value)) {
                let var = &caps[1];
                sink.warning(
                    code::HOST_VAR,
                    At::Value(&join(&path, name)),
                    format!(
                        "{shown}.{name} resolves ${{{var}}} from the host; a runtime resolves it only if the operator allows that variable, \
and never a provider key: a seat that calls a model gets the runtime's proxy and a scoped token instead"
                    ),
                );
            }
        }
    }
    let mut out = sink.into_diagnostics();
    if deep {
        out.extend(compose(config, dir));
    }
    out
}

/// The line, in a Compose file's text, of `needle` at or after the line
/// that names `service`.
fn compose_line(text: &str, service: &str, needle: &str) -> Option<usize> {
    let lines: Vec<&str> = text.split('\n').collect();
    let start = lines.iter().position(|l| {
        let t = l.trim_start().trim_start_matches(['"', '\'']);
        t.starts_with(service)
            && t[service.len()..]
                .trim_start_matches(['"', '\''])
                .starts_with(':')
    })?;
    lines[start..]
        .iter()
        .position(|l| l.contains(needle))
        .map(|i| start + i + 1)
}

fn compose(config: &Value, dir: &Path) -> Vec<Diagnostic> {
    let Some(compose) = table(config, &["sandbox"])
        .get("compose")
        .filter(|c| truthy(Some(c)))
    else {
        return Vec::new();
    };
    let rel = str_of(compose);
    let path = dir.join(&rel);
    if !path.is_file() {
        return Vec::new();
    }
    let Some(text) = read_text_lossy(&path) else {
        return Vec::new();
    };
    let loc = FileLoc {
        file: PathBuf::from(&rel),
        text: text.clone(),
    };
    let mut sink = Sink::new(Some(&loc));
    let doc: Option<Value> = serde_yaml_ng::from_str(&text).ok();
    let services = doc.as_ref().and_then(|d| d.get("services"));
    match services {
        Some(Value::Object(services)) => {
            for (name, svc) in services {
                let Value::Object(svc) = svc else {
                    continue;
                };
                for (key, what) in COMPOSE_HOST_KEYS {
                    let Some(value) = svc.get(*key) else {
                        continue;
                    };
                    let off = [
                        Value::Null,
                        Value::Bool(false),
                        Value::Array(vec![]),
                        Value::Object(Default::default()),
                    ]
                    .iter()
                    .any(|x| py_eq(value, x));
                    let namespaced = matches!(*key, "pid" | "ipc" | "network_mode" | "userns_mode");
                    if off || (namespaced && value.as_str() != Some("host")) {
                        continue;
                    }
                    if *key == "security_opt" {
                        let items = match value {
                            Value::Array(a) => a.iter().map(str_of).collect::<Vec<_>>(),
                            other => vec![str_of(other)],
                        };
                        if !items.iter().any(|v| v.contains("unconfined")) {
                            continue;
                        }
                    }
                    let at =
                        compose_line(&text, name, &format!("{key}:")).map_or(At::File, At::Line);
                    sink.error(
                        code::COMPOSE_HOST,
                        at,
                        format!(
                            "Compose service {name} asks for {what} ({key}); a runtime refuses settings that weaken isolation from the host \
unless its operator allows them for this task"
                        ),
                    );
                }
                let volumes: Vec<Value> = match svc.get("volumes") {
                    Some(Value::Array(v)) => v.clone(),
                    Some(Value::Object(m)) => m.keys().map(|k| Value::String(k.clone())).collect(),
                    _ => Vec::new(),
                };
                for vol in volumes {
                    let source = match &vol {
                        Value::Object(m) => {
                            m.get("source").and_then(Value::as_str).map(str::to_string)
                        }
                        other => {
                            let s = str_of(other);
                            s.contains(':')
                                .then(|| s.split(':').next().unwrap_or("").to_string())
                        }
                    };
                    let Some(source) = source.filter(|s| !s.is_empty()) else {
                        continue;
                    };
                    let climbs = source.split('/').any(|p| p == "..");
                    if let Some(caps) = RUNTIME_VAR.captures(&source)
                        && RUNTIME_COMPOSE_VARS.contains(&&caps[1])
                        && !climbs
                    {
                        continue;
                    }
                    if source.starts_with(['/', '~', '$'])
                        || climbs
                        || source.contains("docker.sock")
                    {
                        let at = compose_line(&text, name, &source).map_or(At::File, At::Line);
                        sink.error(
                            code::COMPOSE_HOST,
                            at,
                            format!(
                                "Compose service {name} mounts {source} from the host; mounts must stay inside the package or in folders the \
runtime provides"
                            ),
                        );
                    }
                }
            }
        }
        _ if COMPOSE_TEXT.is_match(&text) => {
            let line = text
                .split('\n')
                .position(|l| COMPOSE_TEXT.is_match(l))
                .map(|i| i + 1);
            sink.error(
                code::COMPOSE_HOST,
                line.map_or(At::File, At::Line),
                "the Compose file asks for host privileges (privileged mode, the host's network or processes, or the Docker socket); a \
runtime refuses them unless its operator allows them for this task",
            );
        }
        _ => {}
    }
    sink.into_diagnostics()
}

/// family@1 runs the generator by its `#!` line, so the file must exist in
/// the package and start with one (the reference's `check_family_files`).
pub(crate) fn family_files(config: &Value, dir: &Path, loc: &ConfigLoc<'_>) -> Vec<Diagnostic> {
    let Some(Value::String(generator)) = table(config, &["family"]).get("generator") else {
        return Vec::new();
    };
    let path = dir.join(generator);
    if !path.is_file() {
        let mut sink = Sink::new(Some(loc));
        sink.error(
            code::GENERATOR,
            At::Value("family.generator"),
            format!("[family] generator names {generator}, which is not a file in the package"),
        );
        return sink.into_diagnostics();
    }
    let head: Vec<u8> = std::fs::read(&path)
        .map(|b| b.into_iter().take(256).collect())
        .unwrap_or_default();
    if head.starts_with(b"#!/") {
        return Vec::new();
    }
    vec![
        Diagnostic::error(
            code::GENERATOR,
            format!(
                "{generator} needs a first line naming its interpreter, such as #!/usr/bin/env python3: family@1 runs it by that line"
            ),
        )
        .at(generator.as_str(), Some(1), None)
        .help("add a #! line as the generator's first line"),
    ]
}

/// Harbor features an import records under `[import.harbor]` because task.md
/// has no home for them (the reference's `check_imported`).
pub(crate) fn imported(config: &Value, dir: &Path, loc: &ConfigLoc<'_>) -> Vec<Diagnostic> {
    let harbor = table(config, &["import", "harbor"]);
    let mut sink = Sink::new(Some(loc));
    if let Some(Value::String(t)) = harbor.get("trajectory") {
        sink.warning(
            code::IMPORTED,
            At::Value("import.harbor.trajectory"),
            format!(
                "[import.harbor] trajectory: Harbor seeds the agent's session with {t}, an ATIF trajectory, before the agent starts; no \
task.md runtime loads it, so a task.md run starts without that context"
            ),
        );
    }
    if let Some(Value::Object(workdirs)) = harbor.get("workdirs") {
        for (name, path) in workdirs {
            let shown = str_of(path);
            let setup = if dir.join(&shown).join("setup.sh").is_file() {
                " and runs its setup.sh"
            } else {
                ""
            };
            sink.warning(
                code::IMPORTED,
                At::Value(&join("import.harbor.workdirs", name)),
                format!(
                    "[import.harbor] workdirs: before stage {name} starts, Harbor copies {shown}/ into the agent's working directory{setup}; \
no task.md runtime does, so the stage starts without them"
                ),
            );
        }
    }
    sink.into_diagnostics()
}

/// Canaries built on a public GUID detect nothing.
pub(crate) fn canaries(doc: &TaskDocument, loc: &ConfigLoc<'_>) -> Vec<Diagnostic> {
    let mut sink = Sink::new(Some(loc));
    let integrity_canary = table(doc.config_value(), &["integrity"])
        .get("canary")
        .filter(|v| truthy(Some(v)))
        .map(str_of)
        .unwrap_or_default();
    let comment = doc.canary.clone().unwrap_or_default();
    for (value, in_config) in [(comment, false), (integrity_canary, true)] {
        for (guid, what) in SHARED_CANARY_GUIDS {
            if !value.to_lowercase().contains(guid) {
                continue;
            }
            let at = if in_config {
                At::Value("integrity.canary")
            } else {
                let line = doc
                    .instruction_source
                    .split('\n')
                    .position(|l| l.to_lowercase().contains(guid))
                    .map_or(1, |i| i + 1);
                At::Line(line)
            };
            sink.warning(
                code::SHARED_CANARY,
                at,
                format!("the canary reuses {what}, which cannot detect contamination; give each task its own, for example from uuidgen"),
            );
        }
    }
    sink.into_diagnostics()
}

static SEAT_WORDS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?i)\b(oracle|(?:reference|official|correct|expected|gold(?:en)?|model) (?:solution|answer|output|implementation|fix)|known[- ]bad|near[- ]miss|no[- ]op (?:control|run|baseline)|negative control|(?:this|a|the) control\b|control (?:script|submission|run)|cheat(?:s|ing|er)?|should (?:score|fail|pass|get|earn)|expected to (?:score|fail|pass)|must (?:score|fail)|full marks)\b",
    )
    .unwrap()
});
static SLASH_COMMENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|\s)//\s*(?P<text>.*)$").unwrap());
static BLOCK_COMMENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"/\*\s*(?P<text>.*?)(?:\*/|$)").unwrap());
static HTML_COMMENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<!--\s*(?P<text>.*?)(?:-->|$)").unwrap());
static SQL_COMMENT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\s*--\s*(?P<text>.*)$").unwrap());

/// The text of a `#` comment: a `#` at the start or after a space, not
/// `#!`.
fn hash_comment(line: &str) -> Option<(usize, &str)> {
    let mut prev: Option<char> = None;
    for (i, c) in line.char_indices() {
        if c == '#' && prev.is_none_or(char::is_whitespace) && !line[i + 1..].starts_with('!') {
            let rest = &line[i + 1..];
            let skip = rest.len() - rest.trim_start().len();
            return Some((i + 1 + skip, &rest[skip..]));
        }
        prev = Some(c);
    }
    None
}

/// The comment texts on a line, with their offsets in the line, in the
/// order the reference tries them.
fn comment_texts(line: &str, sql: bool) -> Vec<(usize, &str)> {
    let mut out = Vec::new();
    if let Some(t) = hash_comment(line) {
        out.push(t);
    }
    for rx in [&*SLASH_COMMENT, &*BLOCK_COMMENT, &*HTML_COMMENT] {
        if let Some(m) = rx.captures(line).and_then(|c| c.name("text")) {
            out.push((m.start(), m.as_str()));
        }
    }
    if sql && let Some(m) = SQL_COMMENT.captures(line).and_then(|c| c.name("text")) {
        out.push((m.start(), m.as_str()));
    }
    out
}

/// Scripts a scripted seat runs: `oracle/`, `controls/`, each stage's
/// `oracle/`, and any path `[integrity.controls]` names (the reference's
/// `_seat_files`).
fn seat_files(config: &Value, dir: &Path) -> Vec<(PathBuf, &'static str)> {
    let mut roots: Vec<(PathBuf, &'static str)> = vec![
        (dir.join("oracle"), "the reference"),
        (dir.join("controls"), "a control"),
    ];
    let mut stages: Vec<PathBuf> = std::fs::read_dir(dir.join("stages"))
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_dir())
                .collect()
        })
        .unwrap_or_default();
    stages.sort();
    roots.extend(
        stages
            .into_iter()
            .map(|d| (d.join("oracle"), "the reference")),
    );
    for value in table(config, &["integrity", "controls"]).values() {
        let items: Vec<&Value> = match value {
            Value::Array(a) => a.iter().collect(),
            other => vec![other],
        };
        for v in items {
            if let Value::String(s) = v
                && !s.starts_with('/')
                && !s.split('/').any(|p| p == "..")
            {
                roots.push((dir.join(s), "a control"));
            }
        }
    }
    let mut out: Vec<(PathBuf, &'static str)> = Vec::new();
    for (root, seat) in roots {
        let files: Vec<PathBuf> = if root.is_file() {
            vec![root]
        } else if root.is_dir() {
            WalkDir::new(&root)
                .min_depth(1)
                .sort_by_file_name()
                .into_iter()
                .filter_map(Result::ok)
                .map(|e| e.into_path())
                .collect()
        } else {
            Vec::new()
        };
        for f in files {
            let small = std::fs::metadata(&f).is_ok_and(|m| m.is_file() && m.len() <= 1_000_000);
            if small && !out.iter().any(|(g, _)| g == &f) {
                out.push((f, seat));
            }
        }
    }
    out
}

/// Judging is blind to the seat, so a scripted seat's comments must not say
/// what it is (the reference's `check_seat_comments`).
pub(crate) fn seat_comments(config: &Value, dir: &Path) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for (file, seat) in seat_files(config, dir) {
        let Ok(bytes) = std::fs::read(&file) else {
            continue;
        };
        let Ok(raw) = String::from_utf8(bytes) else {
            continue;
        };
        let text = super::text::universal(&raw);
        let rel = file
            .strip_prefix(dir)
            .unwrap_or(&file)
            .to_string_lossy()
            .replace('\\', "/");
        let ext = file.extension().and_then(|e| e.to_str()).unwrap_or("");
        let patch = matches!(ext, "patch" | "diff");
        let sql = matches!(ext, "sql" | "lua");
        let loc = FileLoc {
            file: PathBuf::from(&rel),
            text: text.clone(),
        };
        let mut sink = Sink::new(Some(&loc));
        let mut line_start = 0;
        for full in text.split('\n') {
            let (line, shift) = if patch {
                if !full.starts_with('+') || full.starts_with("+++") {
                    line_start += full.len() + 1;
                    continue;
                }
                (&full[1..], 1)
            } else {
                (full, 0)
            };
            let hit = comment_texts(line, sql).into_iter().find_map(|(at, t)| {
                SEAT_WORDS
                    .find(t)
                    .map(|m| (at + m.start(), at + m.end(), m.as_str().to_string()))
            });
            if let Some((s, e, word)) = hit {
                let base = line_start + shift;
                sink.warning(
                    code::SEAT_COMMENT,
                    At::Span(base + s, base + e),
                    format!(
                        "a comment names the seat ({}), so a judge that reads this script can tell {seat} from a solver: keep such notes \
in [integrity.controls] or a notes block",
                        repr_str(&word)
                    ),
                );
            }
            line_start += full.len() + 1;
        }
        out.extend(sink.into_diagnostics());
    }
    out
}

const SHELL_PREFIX: &[&str] = &[
    "exec", "env", "time", "command", "nohup", "then", "do", "else", "!", "{", "sudo",
];
static LAUNCHERS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?:uvx|uv|pipx|python[\d.]*|.*/python[\d.]*)$").unwrap());
static TRAILING_COMMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\s#.*$").unwrap());
static SEGMENTS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"&&|\|\||[;&|()`]|\$\(").unwrap());
static ASSIGNMENT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Za-z_]\w*=").unwrap());

/// Whether a shell script runs RewardKit's command, directly or through
/// uvx, uv run, pipx run, or python -m (the reference's `runs_rewardkit`).
pub(crate) fn runs_rewardkit(script: &str) -> bool {
    for raw in script.split('\n') {
        let line = if raw.trim_start().starts_with('#') {
            String::new()
        } else {
            TRAILING_COMMENT.replace(raw, "").into_owned()
        };
        for segment in SEGMENTS.split(&line) {
            let mut words: Vec<&str> = segment
                .split_whitespace()
                .map(|w| w.trim_matches(['\'', '"']))
                .collect();
            while let Some(first) = words.first() {
                if SHELL_PREFIX.contains(first) || ASSIGNMENT.is_match(first) {
                    words.remove(0);
                } else {
                    break;
                }
            }
            let Some(first) = words.first() else {
                continue;
            };
            if first.rsplit('/').next() == Some("rewardkit") {
                return true;
            }
            if LAUNCHERS.is_match(first) && words[1..].contains(&"rewardkit") {
                return true;
            }
        }
    }
    false
}

/// Why the verifier's scripts call a model: `[verifier] models = true`, a
/// provider key its env names, or RewardKit judges that test.sh runs (the
/// reference's `script_models`). Empty when nothing says they do.
pub(crate) fn script_models(config: &Value, dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let mut tables = vec![(
        "[verifier]".to_string(),
        table(config, &["verifier"]),
        "verifier".to_string(),
    )];
    for (name, st) in table(config, &["stages"]) {
        tables.push((
            format!("[stages.{name}.verifier]"),
            table(st, &["verifier"]),
            format!("stages/{name}/verifier"),
        ));
    }
    let record = table(config, &["import", "harbor", "rewardkit"]);
    for (where_, t, folder) in tables {
        if t.get("models") == Some(&Value::Bool(true)) {
            out.push(format!("{where_} models = true"));
        }
        let env = match t.get("env") {
            Some(Value::Object(e)) => e.clone(),
            _ => Default::default(),
        };
        let mut named: Vec<String> = env
            .keys()
            .filter(|k| PROVIDER_VAR_RE.is_match(k))
            .cloned()
            .collect();
        for x in env.values() {
            for caps in HOST_VAR_RE.captures_iter(&str_of(x)) {
                if PROVIDER_VAR_RE.is_match(&caps[1]) {
                    named.push(caps[1].to_string());
                }
            }
        }
        named.sort();
        named.dedup();
        if !named.is_empty() {
            out.push(format!("{where_} env names {}", named.join(", ")));
        }
        let script = dir.join(&folder).join("test.sh");
        if script.is_file()
            && let Some(text) = read_text_lossy(&script)
            && runs_rewardkit(&text)
            && !(!record.is_empty() && !truthy(record.get("judges")))
        {
            out.push(format!(
                "{folder}/test.sh runs RewardKit, whose judges call a model"
            ));
        }
    }
    if let Some(judges) = record.get("judges").filter(|j| truthy(Some(j)))
        && !out.iter().any(|r| r.contains("RewardKit"))
    {
        let names: Vec<String> = match judges {
            Value::Array(a) => a.iter().map(str_of).collect(),
            Value::String(s) => s.chars().map(String::from).collect(),
            Value::Object(m) => m.keys().cloned().collect(),
            other => vec![str_of(other)],
        };
        out.push(format!(
            "[import.harbor] rewardkit records judge files that call a model ({})",
            names.join(", ")
        ));
    }
    out
}

/// A verifier script that calls a model needs `[verifier] models = true`
/// (the reference's `check_script_models`).
pub(crate) fn check_script_models(
    config: &Value,
    dir: &Path,
    loc: &ConfigLoc<'_>,
) -> Vec<Diagnostic> {
    let reasons: Vec<String> = script_models(config, dir)
        .into_iter()
        .filter(|r| !r.ends_with("models = true"))
        .collect();
    let stages = table(config, &["stages"]);
    let declared = std::iter::once(config)
        .chain(stages.values())
        .any(|t| table(t, &["verifier"]).get("models") == Some(&Value::Bool(true)));
    if reasons.is_empty() || declared {
        return Vec::new();
    }
    let mut sink = Sink::new(Some(loc));
    sink.warning(
        code::SCRIPT_MODELS,
        At::Key("verifier"),
        format!(
            "grading calls a model from a script ({}), but [verifier] models is not true: set [verifier] models = true, so the runtime \
gives the verifier a seat token and the proxy's URL. Without it the script cannot reach a model on a task.md runtime, and its judge \
calls fail",
            reasons.join("; ")
        ),
    );
    sink.into_diagnostics()
}

/// What the agent's container is made from, or `None` when nothing says
/// (the reference's `sandbox_source`).
pub(crate) fn sandbox_source(config: &Value, dir: &Path) -> Option<&'static str> {
    let sandbox = table(config, &["sandbox"]);
    if nonblank(sandbox.get("image")).is_some() {
        return Some("[sandbox] image");
    }
    if dir.join("sandbox/Dockerfile").is_file() {
        return Some("sandbox/Dockerfile");
    }
    if nonblank(sandbox.get("compose")).is_some() {
        return Some("[sandbox] compose");
    }
    let world = table(config, &["world"]);
    if nonblank(world.get("profile")).is_some() {
        return Some("[world] profile");
    }
    for (part, label) in [
        ("desktop", "[world.desktop] image"),
        ("mobile", "[world.mobile] image"),
    ] {
        if table(config, &["world", part])
            .get("image")
            .is_some_and(Value::is_string)
        {
            return Some(label);
        }
    }
    None
}

/// What grades the task: its rubric, its verifier script, a stage's, a
/// verifier image, or a preference protocol (the reference's `graders`).
pub(crate) fn graders(config: &Value, has_rubric: bool, dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if has_rubric {
        out.push("verifier/rubric.json".to_string());
    }
    if dir.join("verifier/test.sh").is_file() {
        out.push("verifier/test.sh".into());
    }
    for name in table(config, &["stages"]).keys() {
        let folder = dir.join("stages").join(name).join("verifier");
        if judgment::read(&folder.join("rubric.json"), "rubric").is_some() {
            out.push(format!("stages/{name}/verifier/rubric.json"));
        }
        if folder.join("test.sh").is_file() {
            out.push(format!("stages/{name}/verifier/test.sh"));
        }
    }
    if table(config, &["verifier", "sandbox"])
        .get("image")
        .is_some_and(Value::is_string)
    {
        out.push("[verifier.sandbox] image".into());
    }
    if !table(config, &["preference"]).is_empty() {
        out.push("[preference]".into());
    }
    out
}

/// `(key, path, config path)` for every control `[integrity.controls]`
/// names (the reference's `control_paths`).
pub(crate) fn control_paths(config: &Value) -> Vec<(String, Value, String)> {
    let mut out = Vec::new();
    for (key, value) in table(config, &["integrity", "controls"]) {
        if key == "injection" || key.starts_with("x-") {
            continue;
        }
        let base = join("integrity.controls", key);
        match value {
            Value::Array(items) => {
                for (i, v) in items.iter().enumerate() {
                    out.push((key.clone(), v.clone(), format!("{base}[{i}]")));
                }
            }
            other => out.push((key.clone(), other.clone(), base)),
        }
    }
    out
}

/// What a package holds beyond task.md: something the agent runs in,
/// something that grades it, and every control it names, as a file (the
/// reference's `check_package`).
pub(crate) fn check_package(
    config: &Value,
    has_rubric: bool,
    dir: &Path,
    loc: &ConfigLoc<'_>,
    deep: bool,
) -> Vec<Diagnostic> {
    let mut sink = Sink::new(Some(loc));
    if sandbox_source(config, dir).is_none() {
        sink.error(
            code::NO_SANDBOX,
            At::Key("sandbox"),
            "nothing says what the agent runs in: set [sandbox] image to a prebuilt image, or add sandbox/Dockerfile for the runtime \
to build",
        );
    }
    if graders(config, has_rubric, dir).is_empty() {
        sink.error(
            code::NO_GRADER,
            At::Key("verifier"),
            "nothing grades this task: add verifier/rubric.json, or verifier/test.sh, the script the runtime runs to check the solver's \
work (docs/package.md, Verifier scripts), or name a verifier image that holds its own scripts in [verifier.sandbox] image",
        );
    }
    for (key, value, path) in control_paths(config) {
        let where_ = format!("[integrity.controls] {key}");
        let Some(p) = nonblank(Some(&value)) else {
            sink.error(
                code::CONTROL_PATH,
                At::Value(&path),
                format!("{where_} names each control by its path in the package, such as \"controls/no-op.sh\""),
            );
            continue;
        };
        if p.starts_with(['/', '~']) || p.split('/').any(|s| s == "..") {
            sink.error(
                code::CONTROL_PATH,
                At::Value(&path),
                format!(
                    "{where_} names {p}, which is outside the package: a control is a file in it"
                ),
            );
        } else if !dir.join(p).is_file() {
            if deep {
                sink.error(
                    code::CONTROL_PATH,
                    At::Value(&path),
                    format!("{where_} names {p}, which is not a file in the package"),
                );
            }
        } else if p.ends_with(".patch") || p.ends_with(".diff") {
            sink.error(
                code::CONTROL_PATH,
                At::Value(&path),
                format!(
                    "{where_} names {p}, a patch: patch controls are cut, since no page says how a runtime applies one. Write a script that \
makes the change, which runs in place of the agent as oracle/solve.sh does"
                ),
            );
        }
    }
    sink.into_diagnostics()
}

/// The `{{name}}` placeholders the instruction and the stage prompts use, in
/// order of first use.
pub(crate) fn placeholders_used(doc: &TaskDocument) -> Vec<String> {
    let mut names: Vec<String> = Vec::new();
    let mut texts = vec![doc.instruction.clone()];
    texts.extend(
        doc.blocks
            .iter()
            .filter(|b| b.kind == super::BlockKind::Stage)
            .map(super::Block::prompt),
    );
    for text in texts {
        for caps in PLACEHOLDER.captures_iter(&text) {
            if !names.iter().any(|n| n == &caps[1]) {
                names.push(caps[1].to_string());
            }
        }
    }
    names
}

/// A note for a family whose prompts use placeholders: `bench check` runs
/// no package code, so it cannot verify that the generator fills them.
pub(crate) fn generator_note(doc: &TaskDocument, config: &Value) -> Option<Diagnostic> {
    let generator = table(config, &["family"])
        .get("generator")
        .and_then(Value::as_str)?;
    let used = placeholders_used(doc);
    if used.is_empty() {
        return None;
    }
    let shown: Vec<String> = used.iter().map(|n| format!("{{{{{n}}}}}")).collect();
    Some(
        Diagnostic::note(
            code::GENERATOR_NOT_RUN,
            format!(
                "the placeholders {} were not verified: checking runs no package code, so the family's generator, {generator}, was not run \
to see that it writes a value for each into instance.json",
                shown.join(", ")
            ),
        )
        .in_file("task.md")
        .help("run the generator once yourself (<generator> --seed 0 --out <empty folder>) and read instance.json's placeholders"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rewardkit_runs_only_as_a_command() {
        assert!(runs_rewardkit(
            "uvx --from 'harbor-rewardkit @ git+x' rewardkit /tests\n"
        ));
        assert!(runs_rewardkit(
            "set -e\nFOO=1 /usr/local/bin/rewardkit /tests"
        ));
        assert!(runs_rewardkit("python3 -m rewardkit /tests"));
        assert!(!runs_rewardkit("mkdir -p /logs/verifier/rewardkit"));
        assert!(!runs_rewardkit("# rewardkit /tests"));
        assert!(!runs_rewardkit("echo hi # rewardkit"));
    }

    #[test]
    fn comment_texts_like_the_reference() {
        assert_eq!(
            hash_comment("x = 1  # the reference fix"),
            Some((9, "the reference fix"))
        );
        assert_eq!(hash_comment("#!/bin/bash"), None);
        assert_eq!(hash_comment("a#b"), None);
        let t = comment_texts("int x; // should pass /* known bad */", false);
        assert_eq!(t[0].1, "should pass /* known bad */");
        assert_eq!(t[1].1, "known bad ");
    }

    #[test]
    fn seat_words() {
        for w in [
            "oracle",
            "Reference solution",
            "known-bad",
            "near miss",
            "this control",
            "cheating",
            "should score",
            "full marks",
        ] {
            assert!(SEAT_WORDS.is_match(w), "{w}");
        }
        assert!(!SEAT_WORDS.is_match("controller"));
    }
}
