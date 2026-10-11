//! A task's Compose file as the [`Service`]s BenchFlow starts beside the
//! agent's container.
//!
//! BenchFlow reads the part of Compose a task needs to declare services:
//! each service's `image` or `build`, `command`, `environment`,
//! `healthcheck`, `expose`, `networks`, and `depends_on`, and one plain
//! bridge network, for which the services' private network stands in.
//!
//! Service `main` is the agent's container, which BenchFlow configures from
//! the task. Harbor's template fills `main` with what Harbor's harness
//! manages (the task's own build, the image name, an idle command, the
//! network mode, the log folders, CPU and memory limits); BenchFlow manages
//! the same things itself, so they are ignored with a note (P010).
//!
//! No host value reaches a sandbox: an environment entry that reads a host
//! variable is dropped with a warning (P011), `${VAR:-default}` is its
//! default, and a bind mount of a host folder is dropped with a warning
//! (P012). Every other key is refused by name, never skipped: a named
//! volume, a published port, or a capability the runtime silently dropped
//! would change the task.
//!
//! A `command` written as one string is split into words as Compose splits
//! it ([`crate::shellwords`]): a string Compose refuses is an error (P014)
//! and refused, and one Compose cuts at a shell operator runs the words
//! before it, with a warning (P013).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde_json::Value;

use crate::diag::{Diagnostic, Explanation};
use crate::extras::Service;
use crate::package::{ImageRef, Ready, Unsupported};

static SERVICE_NAME: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$").unwrap());
static COMPOSE_DURATION: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(\d+(?:\.\d+)?)(us|ms|s|m|h)").unwrap());

/// Keys of a service other than `main` that BenchFlow honors.
const SERVICE_KEYS: &[&str] = &[
    "image",
    "build",
    "command",
    "environment",
    "healthcheck",
    "expose",
    "depends_on",
    "networks",
];

/// Diagnostic codes this module emits.
pub static EXPLANATIONS: &[Explanation] = &[
    Explanation {
        code: "P010",
        title: "Compose settings BenchFlow manages itself",
        body: "The split layout's Compose template fills service `main` with what a split-layout runner manages: the \
task's own build, the image name, an idle command (`sleep infinity`), `network_mode`, the runner's log folders \
(`${HOST_VERIFIER_LOGS_PATH}:${ENV_VERIFIER_LOGS_PATH}`), `deploy` CPU and memory limits, and `networks`. \
BenchFlow builds the task's Dockerfile, keeps the container running, sets the network and limits from the \
task's config, and collects /logs itself, so these settings are read and ignored. The note lists each one. \
A single plain bridge network is replaced by the services' private network, where every service is reached \
by its name and every port is open.",
    },
    Explanation {
        code: "P011",
        title: "Compose environment value taken from the host, dropped",
        body: "An `environment` entry in the Compose file reads a variable of the host that runs the task \
(`ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}`, or a bare `KEY`). BenchFlow never passes host values into a \
sandbox, so the entry is dropped and the variable is not set. A default (`${NAME:-value}`) is used as the \
value, never the host's; a literal value is set in the container's environment. Give the entry a literal \
value or a default if the task needs it. A sidecar's `command` or `healthcheck` test that reads a host \
variable is warned about too; it is passed on as written, never filled in from the host.",
    },
    Explanation {
        code: "P012",
        title: "Compose bind mount of a host folder, dropped",
        body: "A `volumes` entry in the Compose file mounts a folder of the host that runs the task (for \
example `${HOME}/.config/gcloud:/root/.config/gcloud:ro`). BenchFlow never mounts host folders into a \
sandbox, so the entry is dropped. A task that needs credentials or files must declare them another way: \
files in the package or the image, or a secret the run passes.",
    },
    Explanation {
        code: "P013",
        title: "Service command cut at a shell operator",
        body: "A service's `command` written as one string is split into words as Docker Compose splits it \
(BenchFlow 0.8 handed it to Compose), and no shell runs it. Compose stops reading at the first `;`, `&`, \
`|`, `<` or `>` outside quotes, runs the words before it, and drops the rest, along with a word of digits \
right before a `>` (the `2` of `2>&1`): `redis-server --save 60 1 > /dev/null` runs \
`redis-server --save 60 1`. BenchFlow runs the same words and warns. To redirect, pipe, or chain commands, run a shell: \
`sh -c 'redis-server --save 60 1 > /dev/null'`, or the list `[sh, -c, '...']`. The split is Docker Compose \
v5.4.0 to v5.6.0's (go-shellwords v1.0.14 and v1.0.15); older Compose releases treat a word of digits before \
`>` differently.",
    },
    Explanation {
        code: "P014",
        title: "Service command Docker Compose cannot split",
        body: "A service's `command` written as one string is split into words as Docker Compose splits it \
(BenchFlow 0.8 handed it to Compose). Compose refuses a string with an unclosed quote, backtick, or `$(`, a `(` \
outside quotes that does not open a `$(` (`python -c print(1)`), a `)` that closes none, or a lone backslash at \
the end, and with it the whole file, so the task did not start under 0.8 either. BenchFlow reports it when the \
task is checked and does not run the task. Write the command as a list (`[python, -c, 'print(1)']`), or quote \
the characters (`python -c 'print(1)'`). The split is Docker Compose v5.4.0 to v5.6.0's (go-shellwords v1.0.14 \
and v1.0.15). Other releases differ on a few strings around backslashes, backticks, and `$(`: Compose v5.3.1 \
and older run `echo a\\\\$(date)`, which these refuse. BenchFlow splits the string as written, while Compose \
first replaces `$$` with `$` and `${VAR:-default}` with its value, so a command that uses them next to a `(` can \
be judged differently: Compose refuses `'\\'$$(x)` and runs `echo ${X:-$}(1)`.",
    },
];

fn refuse(refused: &mut Vec<Unsupported>, field: String, reason: impl Into<String>) {
    refused.push(Unsupported {
        field,
        reason: reason.into(),
        agents_only: false,
    });
}

/// A Compose duration (`"2s"`, `"1m30s"`, `"500ms"`).
pub(crate) fn duration(v: &Value) -> Option<Duration> {
    let text = match v {
        Value::String(s) => s.trim().to_string(),
        Value::Number(n) => return n.as_f64().map(crate::values::duration_from_secs),
        _ => return None,
    };
    let mut secs = 0.0;
    let mut matched = 0;
    for c in COMPOSE_DURATION.captures_iter(&text) {
        let n: f64 = c[1].parse().ok()?;
        secs += n * match &c[2] {
            "us" => 1e-6,
            "ms" => 1e-3,
            "s" => 1.0,
            "m" => 60.0,
            _ => 3600.0,
        };
        matched += c[0].len();
    }
    (matched == text.len() && matched > 0).then(|| crate::values::duration_from_secs(secs))
}

/// A healthcheck `test` as a shell command: `["CMD", "curl", ...]` is
/// quoted word by word, `["CMD-SHELL", "..."]` and a plain string run as is.
fn test_command(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Array(items) => {
            let words: Vec<&str> = items.iter().filter_map(Value::as_str).collect();
            match words.split_first() {
                Some((&"CMD-SHELL", rest)) => Some(rest.join(" ")),
                Some((&"CMD", rest)) => Some(
                    rest.iter()
                        .map(|w| shell_word(w))
                        .collect::<Vec<_>>()
                        .join(" "),
                ),
                _ => None,
            }
        }
        _ => None,
    }
}

fn shell_word(w: &str) -> String {
    if !w.is_empty()
        && w.chars()
            .all(|c| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c))
    {
        w.to_string()
    } else {
        format!("'{}'", w.replace('\'', r"'\''"))
    }
}

/// Compose's healthcheck as a readiness check, with Compose's defaults.
fn healthcheck(v: &Value, field: &str, refused: &mut Vec<Unsupported>) -> Option<Ready> {
    let Value::Object(h) = v else {
        refuse(refused, field.to_string(), "must be a mapping");
        return None;
    };
    if h.get("disable").and_then(Value::as_bool) == Some(true) {
        return None;
    }
    let run = match h.get("test").and_then(test_command) {
        Some(r) => r,
        None => {
            refuse(
                refused,
                format!("{field}.test"),
                "must be a command string, or a list starting with CMD or CMD-SHELL",
            );
            return None;
        }
    };
    let mut ready = Ready {
        run,
        interval: Duration::from_secs(30),
        timeout: Duration::from_secs(30),
        start_period: Duration::ZERO,
        start_interval: Duration::from_secs(5),
        retries: 3,
    };
    for (key, value) in h {
        let slot = match key.as_str() {
            "interval" => &mut ready.interval,
            "timeout" => &mut ready.timeout,
            "start_period" => &mut ready.start_period,
            "start_interval" => &mut ready.start_interval,
            "retries" => {
                match value.as_u64().and_then(|n| u32::try_from(n).ok()) {
                    Some(n) => ready.retries = n,
                    None => refuse(
                        refused,
                        format!("{field}.retries"),
                        "must be a whole number",
                    ),
                }
                continue;
            }
            "test" | "disable" => continue,
            other => {
                refuse(
                    refused,
                    format!("{field}.{other}"),
                    "not a healthcheck key BenchFlow reads",
                );
                continue;
            }
        };
        match duration(value) {
            Some(d) => *slot = d,
            None => refuse(
                refused,
                format!("{field}.{key}"),
                "not a Compose duration such as \"2s\"",
            ),
        }
    }
    Some(ready)
}

/// How [`refused_command`] begins, so [`is_refused_command`] knows its
/// refusals.
const REFUSED_COMMAND: &str = "Docker Compose cannot split it into words";

/// Why a service's `command` string cannot run: Docker Compose refuses it.
pub(crate) fn refused_command(why: crate::shellwords::Refusal) -> String {
    format!(
        "{REFUSED_COMMAND} ({why}) and refuses the file, so the task failed under BenchFlow 0.8; write the command as a list, or quote those characters"
    )
}

/// Whether a refused field is a service command Docker Compose cannot split
/// ([`refused_command`], reported as P014).
pub(crate) fn is_refused_command(u: &Unsupported) -> bool {
    u.reason.starts_with(REFUSED_COMMAND)
}

/// P013: Compose stopped reading a service's `command` string at a shell
/// operator and dropped the rest; BenchFlow runs the same words.
pub(crate) fn cut_command(field: &str, argv: &[String], dropped: &str) -> Diagnostic {
    Diagnostic::warning(
        "P013",
        format!(
            "{field}: Docker Compose stops reading the command at a shell operator and drops {dropped:?}; BenchFlow runs the words before it, as 0.8 did: {}",
            serde_json::to_string(argv).unwrap_or_default()
        ),
    )
    .help("no shell runs a command string; to redirect, pipe, or chain commands, run one: `sh -c '...'`, or the list `[sh, -c, '...']`")
}

/// The Compose releases whose split [`crate::shellwords`] ports: P013 and
/// P014 are what they do. Compose v5.3.1 (go-shellwords v1.0.13) accepts
/// `echo a\\$(date)`, which these refuse, and older v2 releases differ on
/// more strings.
const COMPOSE_SPLIT: &str = "as Docker Compose v5.4.0 to v5.6.0 split it, with go-shellwords v1.0.14 and v1.0.15; other versions differ on a few strings";

/// P014: a service's `command` string Docker Compose refuses, for `bench
/// check`; the field is refused too ([`refused_command`]).
pub(crate) fn unsplittable_command(field: &str, why: crate::shellwords::Refusal) -> Diagnostic {
    Diagnostic::error(
        "P014",
        format!("{field}: Docker Compose cannot split the command into words: {why} ({COMPOSE_SPLIT})"),
    )
    .help("write the command as a list, or quote those characters; Compose refuses the whole file, so the task did not start under BenchFlow 0.8 either")
}

/// Why a service's `command` cannot run.
enum BadCommand {
    /// Docker Compose refuses the string.
    Compose(crate::shellwords::Refusal),
    /// Neither a string nor a list of strings.
    NotStrings,
}

/// A service's `command`: a list as written, or a string split into words
/// as Docker Compose splits it ([`crate::shellwords`]), with the text Compose
/// drops at a shell operator.
fn command(v: &Value) -> Result<(Vec<String>, Option<&str>), BadCommand> {
    match v {
        Value::String(s) => crate::shellwords::parse(s)
            .map(|split| (split.words, split.dropped))
            .map_err(BadCommand::Compose),
        Value::Array(items) if items.iter().all(Value::is_string) => Ok((
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect(),
            None,
        )),
        Value::Null => Ok((Vec::new(), None)),
        _ => Err(BadCommand::NotStrings),
    }
}

/// A Compose value with its `${VAR:-default}` defaults applied, or the host
/// variables it would read. A host's value is never used: `${VAR:-x}` and
/// `${VAR-x}` are `x`; `${VAR}`, `$VAR`, `${VAR:?...}`, and `${VAR:+...}`
/// read the host. `$$` is a literal `$`.
fn literal(text: &str) -> Result<String, Vec<String>> {
    let mut out = String::new();
    let mut host = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find('$') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        if let Some(r) = rest.strip_prefix("$$") {
            out.push('$');
            rest = r;
        } else if let Some(r) = rest.strip_prefix("${") {
            let Some(close) = closing_brace(r) else {
                host.push(r.to_string());
                break;
            };
            let inner = &r[..close];
            rest = &r[close + 1..];
            let name_end = inner
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(inner.len());
            let (name, op) = inner.split_at(name_end);
            match op.strip_prefix(":-").or_else(|| op.strip_prefix('-')) {
                Some(default) => match literal(default) {
                    Ok(v) => out.push_str(&v),
                    Err(names) => host.extend(names),
                },
                None => host.push(name.to_string()),
            }
        } else {
            let r = &rest[1..];
            let end = r
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(r.len());
            if end == 0 {
                out.push('$');
            } else {
                host.push(r[..end].to_string());
            }
            rest = &r[end..];
        }
    }
    out.push_str(rest);
    if host.is_empty() { Ok(out) } else { Err(host) }
}

/// The index of the `}` closing a `${` whose text starts `text`.
fn closing_brace(text: &str) -> Option<usize> {
    let mut depth = 0usize;
    let bytes = text.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        match b {
            b'{' if i > 0 && bytes[i - 1] == b'$' => depth += 1,
            b'}' if depth == 0 => return Some(i),
            b'}' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// Split `text` at `sep`, but not inside `${...}`.
fn split_outside_braces(text: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    let bytes = text.as_bytes();
    for (i, c) in text.char_indices() {
        match c {
            '{' if i > 0 && bytes[i - 1] == b'$' => depth += 1,
            '}' if depth > 0 => depth -= 1,
            c if c == sep && depth == 0 => {
                parts.push(&text[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&text[start..]);
    parts
}

/// The variable a value is made of alone (`${NAME}`, `${NAME:-x}`), if any.
fn variable_name(text: &str) -> Option<&str> {
    let inner = text.strip_prefix("${")?;
    let close = closing_brace(inner)?;
    if close + 1 != inner.len() {
        return None;
    }
    let end = inner
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(close);
    Some(&inner[..end])
}

/// What reading a Compose file found.
struct Reading<'a> {
    label: &'a str,
    refused: &'a mut Vec<Unsupported>,
    /// Settings BenchFlow manages itself, ignored: `services.main.build:
    /// BenchFlow builds environment/Dockerfile itself`.
    ignored: Vec<String>,
    /// Warnings (host values and host folders dropped, commands cut at a
    /// shell operator) and errors (command strings Compose refuses).
    diagnostics: Vec<Diagnostic>,
}

impl Reading<'_> {
    fn refuse(&mut self, field: String, reason: impl Into<String>) {
        refuse(self.refused, field, reason);
    }
    fn bad_command(&mut self, field: String, why: BadCommand) {
        match why {
            BadCommand::Compose(why) => {
                self.diagnostics
                    .push(unsplittable_command(&field, why).in_file(self.label));
                self.refuse(field, refused_command(why));
            }
            BadCommand::NotStrings => self.refuse(field, "must be a string or a list of strings"),
        }
    }
    fn ignore(&mut self, field: &str, why: &str) {
        self.ignored.push(format!("{field}: {why}"));
    }
    fn drop_host_env(&mut self, field: &str, key: &str, names: &[String]) {
        let names: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
        self.diagnostics.push(
            Diagnostic::warning(
                "P011",
                format!(
                    "{field}: `{key}` takes the host's {}; dropped, so no host value reaches the sandbox",
                    names.join(", ")
                ),
            )
            .in_file(self.label)
            .help("give it a literal value, or a default (`${NAME:-value}`), which BenchFlow uses; a model agent's own key is passed by the agent, not the task"),
        );
    }
    /// A sidecar's command or healthcheck reads host variables, which
    /// Compose would fill in and BenchFlow never does: it is passed on as
    /// written, with a warning.
    fn keep_host_text(&mut self, field: &str, text: &str, names: &[String]) {
        let names: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
        self.diagnostics.push(
            Diagnostic::warning(
                "P011",
                format!(
                    "{field}: `{text}` takes the host's {}; passed on as written, since no host value reaches the sandbox",
                    names.join(", ")
                ),
            )
            .in_file(self.label)
            .help("write the value out, or use a default (`${NAME:-value}`); a variable the service sets itself is `$$NAME`"),
        );
    }
    fn drop_host_mount(&mut self, field: &str, spec: &str) {
        self.diagnostics.push(
            Diagnostic::warning(
                "P012",
                format!("{field}: `{spec}` mounts a folder of the host; dropped, never mounted"),
            )
            .in_file(self.label)
            .help("a task cannot depend on the host's files; put what it needs in the package or the image"),
        );
    }
}

fn environment(v: &Value, field: &str, r: &mut Reading<'_>) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    // `None` is Compose's pass-through: the host's variable of that name.
    let pairs: Vec<(String, Option<String>)> = match v {
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| {
                let text = match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Null => None,
                    other => Some(other.to_string()),
                };
                (k.clone(), text)
            })
            .collect(),
        Value::Array(items) => {
            let mut pairs = Vec::new();
            for (i, item) in items.iter().enumerate() {
                match item.as_str() {
                    Some(s) => pairs.push(match s.split_once('=') {
                        Some((k, v)) => (k.to_string(), Some(v.to_string())),
                        None => (s.to_string(), None),
                    }),
                    None => r.refuse(format!("{field}[{i}]"), "must be a KEY=VALUE string"),
                }
            }
            pairs
        }
        _ => {
            r.refuse(
                field.to_string(),
                "must be a mapping or a list of KEY=VALUE",
            );
            return out;
        }
    };
    for (k, v) in pairs {
        match v.as_deref().map(literal) {
            Some(Ok(value)) => {
                out.insert(k, value);
            }
            Some(Err(names)) => r.drop_host_env(field, &k, &names),
            None => r.drop_host_env(field, &k, std::slice::from_ref(&k)),
        }
    }
    out
}

/// Harbor's log folders, which its harness mounts and BenchFlow collects
/// itself.
const HARBOR_LOG_SOURCES: &[&str] = &[
    "HOST_VERIFIER_LOGS_PATH",
    "HOST_AGENT_LOGS_PATH",
    "HOST_ARTIFACTS_PATH",
];

/// A service's `volumes`: Harbor's log folders are ignored, host folders
/// dropped with a warning, and named volumes and package folders refused.
fn volumes(v: &Value, field: &str, r: &mut Reading<'_>) {
    let Value::Array(items) = v else {
        r.refuse(field.to_string(), "must be a list");
        return;
    };
    for item in items {
        let (spec, source) = match item {
            Value::String(s) => {
                let parts = split_outside_braces(s, ':');
                if parts.len() < 2 {
                    r.refuse(
                        format!("{field} {s:?}"),
                        "an anonymous volume; volumes are not provided",
                    );
                    continue;
                }
                (s.clone(), parts[0].to_string())
            }
            Value::Object(m) => {
                let kind = m.get("type").and_then(Value::as_str).unwrap_or("volume");
                let source = m.get("source").and_then(Value::as_str).unwrap_or("");
                let target = m.get("target").and_then(Value::as_str).unwrap_or("");
                let spec = format!("{kind} {source}:{target}");
                if kind != "bind" {
                    r.refuse(
                        format!("{field} {spec:?}"),
                        format!("a {kind} mount; only bind mounts are read"),
                    );
                    continue;
                }
                (spec, source.to_string())
            }
            _ => {
                r.refuse(field.to_string(), "entries must be strings or mappings");
                continue;
            }
        };
        if variable_name(&source).is_some_and(|n| HARBOR_LOG_SOURCES.contains(&n)) {
            r.ignore(
                &format!("{field} {spec}"),
                "a split-layout runner's log folder; BenchFlow collects /logs itself",
            );
        } else if source.starts_with('/') || source.starts_with('~') || source.starts_with('$') {
            r.drop_host_mount(field, &spec);
        } else if source.starts_with('.') {
            r.refuse(
                format!("{field} {spec:?}"),
                "a bind mount of a package folder; place package files with the task's mounts instead",
            );
        } else {
            r.refuse(
                format!("{field} {spec:?}"),
                "a named volume; volumes are not provided",
            );
        }
    }
}

/// A service's `networks`: names of the one network the file declares, or
/// `default`. BenchFlow's private network for the services stands in for it.
fn networks(v: &Value, field: &str, declared: &[String], r: &mut Reading<'_>) {
    let names: Vec<String> = match v {
        Value::Array(items) if items.iter().all(Value::is_string) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        Value::Object(m) => {
            for (name, opts) in m {
                if !matches!(opts, Value::Null) && opts.as_object().is_none_or(|o| !o.is_empty()) {
                    r.refuse(
                        format!("{field}.{name}"),
                        "aliases and addresses are not provided; services are reached by service name",
                    );
                }
            }
            m.keys().cloned().collect()
        }
        _ => {
            r.refuse(field.to_string(), "must be a list of network names");
            return;
        }
    };
    for name in names {
        if name != "default" && !declared.contains(&name) {
            r.refuse(format!("{field} {name}"), "not a network the file declares");
        }
    }
}

/// The top-level `networks`: at most one, a plain bridge.
fn declared_networks(v: &Value, field: &str, r: &mut Reading<'_>) -> Vec<String> {
    let Value::Object(m) = v else {
        r.refuse(field.to_string(), "must be a mapping");
        return Vec::new();
    };
    if m.len() > 1 {
        r.refuse(
            field.to_string(),
            "BenchFlow puts every service on one private network; separate networks are not provided",
        );
        return Vec::new();
    }
    for (name, opts) in m {
        let plain = match opts {
            Value::Null => true,
            Value::Object(o) => o
                .iter()
                .all(|(k, v)| k == "driver" && v.as_str() == Some("bridge") || k.starts_with("x-")),
            _ => false,
        };
        if plain {
            r.ignore(
                &format!("{field}.{name}"),
                "BenchFlow's private network for the services stands in for it (services are reached by name; it has no route out)",
            );
        } else {
            r.refuse(
                format!("{field}.{name}"),
                "only a plain bridge network (`driver: bridge`) is read",
            );
        }
    }
    m.keys().cloned().collect()
}

/// True for a command that only keeps the container running.
fn is_idle(argv: &[String]) -> bool {
    let words: Vec<&str> = argv.iter().map(String::as_str).collect();
    let idle = |w: &[&str]| matches!(w, ["sleep", "infinity"] | ["tail", "-f", "/dev/null"]);
    match words.as_slice() {
        ["sh" | "bash" | "/bin/sh" | "/bin/bash", "-c", script] => {
            let inner: Vec<&str> = script.split_whitespace().collect();
            idle(&inner)
        }
        w => idle(w),
    }
}

/// Service `main`, the agent's container, which BenchFlow configures from
/// the task. Harbor's template fills keys its harness manages; BenchFlow
/// manages the same things itself, so they are ignored with a note. Its
/// literal environment values are returned for the agent's sandbox.
fn main_service(
    svc: &serde_json::Map<String, Value>,
    field: &str,
    base: &str,
    declared: &[String],
    r: &mut Reading<'_>,
) -> BTreeMap<String, String> {
    let mut env = BTreeMap::new();
    for (key, value) in svc {
        let at = format!("{field}.{key}");
        match key.as_str() {
            "depends_on" => {}
            k if k.starts_with("x-") => {}
            "build" => {
                let (context, dockerfile) = match value {
                    Value::String(c) => (c.as_str(), None),
                    Value::Object(b) => {
                        if let Some(other) =
                            b.keys().find(|k| *k != "context" && *k != "dockerfile")
                        {
                            r.refuse(format!("{at}.{other}"), "only context and dockerfile are read");
                            continue;
                        }
                        (
                            b.get("context").and_then(Value::as_str).unwrap_or("."),
                            b.get("dockerfile").and_then(Value::as_str),
                        )
                    }
                    _ => {
                        r.refuse(at, "must be a folder or a mapping");
                        continue;
                    }
                };
                let own_context = matches!(context, "." | "./")
                    || variable_name(context) == Some("CONTEXT_DIR");
                if own_context && matches!(dockerfile, None | Some("Dockerfile")) {
                    r.ignore(
                        &at,
                        &format!("BenchFlow builds the task's {base}/Dockerfile itself"),
                    );
                } else {
                    r.refuse(
                        at,
                        format!(
                            "main is built from {base}/Dockerfile; another build context or Dockerfile is not read"
                        ),
                    );
                }
            }
            "image" => match value.as_str() {
                Some(s)
                    if matches!(variable_name(s), Some("MAIN_IMAGE_NAME" | "PREBUILT_IMAGE_NAME"))
                        || (!s.contains('$') && svc.contains_key("build")) =>
                {
                    r.ignore(&at, "BenchFlow names the task's image itself");
                }
                _ => r.refuse(
                    at,
                    "main's image is the task's (its Dockerfile or docker_image); Compose cannot choose another",
                ),
            },
            "command" => match command(value) {
                Ok((argv, _)) if is_idle(&argv) => {
                    r.ignore(&at, "BenchFlow keeps the agent's container running itself");
                }
                Ok(_) => r.refuse(
                    at,
                    "main's command is not run; BenchFlow keeps the agent's container idle itself (only `sleep infinity` is accepted)",
                ),
                Err(why) => r.bad_command(at, why),
            },
            "network_mode" => match value.as_str() {
                Some(s)
                if s == "bridge"
                    || (variable_name(s) == Some("NETWORK_MODE")
                        && !matches!(literal(s), Ok(d) if !d.is_empty() && d != "bridge")) =>
            {
                    r.ignore(&at, "the task's network setting decides the agent's network");
                }
                _ => r.refuse(
                    at,
                    "the task's network setting decides the agent's network; only `bridge` is accepted here",
                ),
            },
            "environment" => env = environment(value, &at, r),
            "volumes" => volumes(value, &at, r),
            "deploy" => {
                let limits = value
                    .as_object()
                    .filter(|d| d.len() == 1)
                    .and_then(|d| d.get("resources"))
                    .and_then(Value::as_object)
                    .filter(|res| res.len() == 1)
                    .and_then(|res| res.get("limits"))
                    .and_then(Value::as_object);
                match limits {
                    Some(l) if l.keys().all(|k| k == "cpus" || k == "memory") => r.ignore(
                        &at,
                        "the task's cpus and memory set the agent's limits",
                    ),
                    _ => r.refuse(
                        at,
                        "only resources.limits.cpus and memory are read (the task's own limits apply)",
                    ),
                }
            }
            "networks" => {
                networks(value, &at, declared, r);
                r.ignore(&at, "the agent's container joins the services' private network");
            }
            "expose" => r.ignore(&at, "every port is reachable on the services' private network"),
            _ => r.refuse(
                at,
                "service main is the agent's container, which BenchFlow configures from the task; this setting is not read",
            ),
        }
    }
    env
}

/// What a Compose file adds to a task.
#[derive(Debug, Default)]
pub(crate) struct Compose {
    /// Containers beside the agent's.
    pub services: Vec<Service>,
    /// Service main's literal environment values, for the agent's sandbox.
    pub env: BTreeMap<String, String>,
    /// A note naming the settings ignored, and a warning for each host value
    /// or host folder dropped.
    pub diagnostics: Vec<Diagnostic>,
}

/// Read a Compose file's services. `base` is the Compose file's folder,
/// relative to the package (`environment` or `sandbox`); build contexts are
/// read relative to it. `label` names the file in refusals and diagnostics.
pub(crate) fn read(
    root: &Path,
    base: &str,
    label: &str,
    refused: &mut Vec<Unsupported>,
) -> Compose {
    let mut compose = Compose::default();
    let path = root.join(base).join("docker-compose.yaml");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            refuse(refused, label.to_string(), format!("cannot be read: {e}"));
            return compose;
        }
    };
    let doc: Value = match serde_yaml_ng::from_str(&text) {
        Ok(v) => v,
        Err(e) => {
            refuse(
                refused,
                label.to_string(),
                format!("is not valid YAML: {e}"),
            );
            return compose;
        }
    };
    let Value::Object(top) = doc else {
        refuse(refused, label.to_string(), "must be a mapping");
        return compose;
    };
    let first = refused.len();
    let mut r = Reading {
        label,
        refused,
        ignored: Vec::new(),
        diagnostics: Vec::new(),
    };
    let mut declared = Vec::new();
    for (key, value) in &top {
        match key.as_str() {
            "services" => {}
            k if k.starts_with("x-") => {}
            "version" | "name" => r.ignore(key, "Compose's file name and version are not used"),
            "networks" => declared = declared_networks(value, key, &mut r),
            _ => r.refuse(
                key.clone(),
                "only `services` and one `networks` entry are read; named volumes, configs, and secrets are not provided",
            ),
        }
    }
    let Some(Value::Object(services)) = top.get("services") else {
        r.refuse(format!("{label} services"), "must be a mapping");
        return compose;
    };
    for (name, svc) in services {
        let field = format!("services.{name}");
        let Value::Object(svc) = svc else {
            r.refuse(field, "must be a mapping");
            continue;
        };
        if name == "main" {
            compose.env = main_service(svc, &field, base, &declared, &mut r);
            continue;
        }
        if !SERVICE_NAME.is_match(name) {
            r.refuse(field, "not a Compose service name");
            continue;
        }
        if let Some(service) = sidecar(root, base, name, svc, &field, &declared, &mut r) {
            compose.services.push(service);
        }
    }
    // Refusals name the file; diagnostics carry it as their location.
    for u in &mut r.refused[first..] {
        if !u.field.starts_with(label) {
            u.field = format!("{label} {}", u.field);
        }
    }
    if !r.ignored.is_empty() {
        let mut note = Diagnostic::note(
            "P010",
            format!(
                "{} Compose {} BenchFlow manages itself {} ignored",
                r.ignored.len(),
                if r.ignored.len() == 1 {
                    "setting"
                } else {
                    "settings"
                },
                if r.ignored.len() == 1 { "is" } else { "are" },
            ),
        )
        .in_file(label);
        for line in r.ignored {
            note = note.note_line(line);
        }
        compose.diagnostics.push(note);
    }
    compose.diagnostics.extend(r.diagnostics);
    compose
}

/// A service beside the agent's container.
fn sidecar(
    root: &Path,
    base: &str,
    name: &str,
    svc: &serde_json::Map<String, Value>,
    field: &str,
    declared: &[String],
    r: &mut Reading<'_>,
) -> Option<Service> {
    for (key, value) in svc {
        match key.as_str() {
            "volumes" => volumes(value, &format!("{field}.volumes"), r),
            "networks" => networks(value, &format!("{field}.networks"), declared, r),
            k if SERVICE_KEYS.contains(&k) || k.starts_with("x-") => {}
            _ => r.refuse(
                format!("{field}.{key}"),
                "not provided for services: BenchFlow runs a service from its image or build, command, environment, and healthcheck only",
            ),
        }
    }
    let image = match (svc.get("image"), svc.get("build")) {
        (_, Some(build)) => {
            let (context, dockerfile) = match build {
                Value::String(c) => (c.clone(), "Dockerfile".to_string()),
                Value::Object(b) => {
                    for key in b.keys() {
                        if key != "context" && key != "dockerfile" {
                            r.refuse(
                                format!("{field}.build.{key}"),
                                "only context and dockerfile are read",
                            );
                        }
                    }
                    (
                        b.get("context")
                            .and_then(Value::as_str)
                            .unwrap_or(".")
                            .to_string(),
                        b.get("dockerfile")
                            .and_then(Value::as_str)
                            .unwrap_or("Dockerfile")
                            .to_string(),
                    )
                }
                _ => {
                    r.refuse(format!("{field}.build"), "must be a folder or a mapping");
                    return None;
                }
            };
            let rel = Path::new(context.trim_start_matches("./"));
            let outside = |p: &Path| {
                p.is_absolute() || p.components().any(|c| c == std::path::Component::ParentDir)
            };
            if dockerfile.contains("${") || outside(Path::new(&dockerfile)) {
                r.refuse(
                    format!("{field}.build.dockerfile"),
                    format!("{dockerfile:?} must be a file inside the build's folder"),
                );
                return None;
            }
            if context.contains("${") || outside(rel) {
                r.refuse(
                    format!("{field}.build"),
                    format!(
                        "{context:?} must be a folder inside {base}/, the Compose file's folder"
                    ),
                );
                return None;
            }
            let context = Path::new(base).join(rel);
            // Links count where they lead: the folder and its Dockerfile
            // must be the package's own.
            let package = std::fs::canonicalize(root).ok();
            let resolves_inside = |p: &Path| match (std::fs::canonicalize(p), &package) {
                (Ok(real), Some(pkg)) => real.starts_with(pkg),
                (Err(_), _) => true,
                _ => false,
            };
            if !resolves_inside(&root.join(&context))
                || !resolves_inside(&root.join(&context).join(&dockerfile))
            {
                r.refuse(
                    format!("{field}.build"),
                    format!(
                        "{context:?} or its {dockerfile} leads outside the task's folder (a link)",
                        context = context.to_string_lossy()
                    ),
                );
                return None;
            }
            if !root.join(&context).join(&dockerfile).is_file() {
                r.refuse(
                    format!("{field}.build"),
                    format!(
                        "{} has no {dockerfile}",
                        context.to_string_lossy().trim_end_matches("/.")
                    ),
                );
                return None;
            }
            ImageRef::Build {
                dockerfile: context.join(&dockerfile),
                context,
            }
        }
        (Some(Value::String(reference)), None) if !reference.contains("${") => ImageRef::Pull {
            reference: reference.clone(),
        },
        (Some(_), None) => {
            r.refuse(
                format!("{field}.image"),
                "must be a literal image reference",
            );
            return None;
        }
        (None, None) => {
            r.refuse(field.to_string(), "needs an image or a build");
            return None;
        }
    };
    let command = match svc.get("command").map(command) {
        None => Vec::new(),
        Some(Ok((argv, dropped))) => {
            if let Some(dropped) = dropped {
                let warning = cut_command(&format!("{field}.command"), &argv, dropped);
                r.diagnostics.push(warning.in_file(r.label));
            }
            argv
        }
        Some(Err(why)) => {
            r.bad_command(format!("{field}.command"), why);
            Vec::new()
        }
    };
    for word in &command {
        if let Err(names) = literal(word) {
            r.keep_host_text(&format!("{field}.command"), word, &names);
        }
    }
    let ready = svc
        .get("healthcheck")
        .and_then(|h| healthcheck(h, &format!("{field}.healthcheck"), r.refused));
    if let Some(ready) = &ready
        && let Err(names) = literal(&ready.run)
    {
        r.keep_host_text(&format!("{field}.healthcheck.test"), &ready.run, &names);
    }
    Some(Service {
        name: name.to_string(),
        image,
        command,
        env: svc
            .get("environment")
            .map(|e| environment(e, &format!("{field}.environment"), r))
            .unwrap_or_default(),
        ready,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A string command is split as Docker Compose splits it (0.8 handed it
    /// to Compose, which keeps `#hunter2`). One Compose refuses is an error
    /// and refused by name, with the cause, when the task is read: 0.8 failed
    /// when Compose read the file.
    #[test]
    fn command_strings_split_as_compose_splits_them() {
        let dir = package(
            "services:\n  main:\n    command: \"sleep 'infinity\"\n  cache:\n    image: redis:7\n    command: \"redis-server --save '' --requirepass #hunter2\"\n  web:\n    image: python:3\n    command: python -c print(1)\n",
            &[],
        );
        let (c, refused) = read_package(&dir);
        assert_eq!(
            c.services[0].command,
            ["redis-server", "--save", "", "--requirepass", "#hunter2"]
        );
        let refused: Vec<(&str, &str)> = refused
            .iter()
            .map(|u| (u.field.as_str(), u.reason.as_str()))
            .collect();
        assert_eq!(refused.len(), 2, "{refused:#?}");
        assert_eq!(refused[0].0, format!("{LABEL} services.main.command"));
        assert!(
            refused[0].1.contains("a single quote is not closed"),
            "{}",
            refused[0].1
        );
        assert_eq!(refused[1].0, format!("{LABEL} services.web.command"));
        assert!(
            refused[1]
                .1
                .starts_with("Docker Compose cannot split it into words (a `(` outside quotes does not open a `$(`)"),
            "{}",
            refused[1].1
        );
        // And `bench check` says so.
        assert_eq!(codes(&c), ["P014", "P014"]);
        assert!(c.diagnostics.iter().all(|d| d.is_error()));
        // It names the Compose releases it splits as (the D5b review of
        // cbf42071, finding 7: other releases split a few strings otherwise).
        assert_eq!(
            c.diagnostics[1].message,
            "services.web.command: Docker Compose cannot split the command into words: a `(` outside quotes does not open a `$(` (as Docker Compose v5.4.0 to v5.6.0 split it, with go-shellwords v1.0.14 and v1.0.15; other versions differ on a few strings)"
        );
    }

    /// Compose stops reading a command string at a shell operator and runs
    /// the words before it; BenchFlow runs the same words and says so.
    #[test]
    fn a_command_cut_at_a_shell_operator_runs_the_words_before_it() {
        let dir = package(
            "services:\n  cache:\n    image: redis:7\n    command: redis-server --save 60 1 > /dev/null\n  log:\n    image: busybox\n    command: \"syslogd -n 2>&1\"\n  web:\n    image: nginx\n    command: \"nginx -g 'daemon off;'\"\n",
            &[],
        );
        let (c, refused) = read_package(&dir);
        assert!(refused.is_empty(), "{refused:?}");
        let argv: Vec<&[String]> = c.services.iter().map(|s| s.command.as_slice()).collect();
        assert_eq!(argv[0], ["redis-server", "--save", "60", "1"]);
        assert_eq!(argv[1], ["syslogd", "-n"]);
        assert_eq!(argv[2], ["nginx", "-g", "daemon off;"]);
        assert_eq!(codes(&c), ["P013", "P013"]);
        let d = &c.diagnostics[0];
        assert_eq!(d.severity, crate::diag::Severity::Warning);
        assert_eq!(
            d.message,
            r#"services.cache.command: Docker Compose stops reading the command at a shell operator and drops "> /dev/null"; BenchFlow runs the words before it, as 0.8 did: ["redis-server","--save","60","1"]"#
        );
        assert!(c.diagnostics[1].message.contains(r#"drops "2>&1""#));
    }

    #[test]
    fn compose_durations() {
        assert_eq!(duration(&"2s".into()), Some(Duration::from_secs(2)));
        assert_eq!(duration(&"1m30s".into()), Some(Duration::from_secs(90)));
        assert_eq!(duration(&"500ms".into()), Some(Duration::from_millis(500)));
        assert_eq!(duration(&"soon".into()), None);
    }

    #[test]
    fn healthchecks_become_shell_commands() {
        let cmd = serde_json::json!(["CMD", "curl", "-sf", "http://localhost:8000/x y"]);
        assert_eq!(
            test_command(&cmd).unwrap(),
            "curl -sf 'http://localhost:8000/x y'"
        );
        let shell = serde_json::json!(["CMD-SHELL", "nc -z localhost 80"]);
        assert_eq!(test_command(&shell).unwrap(), "nc -z localhost 80");
    }

    fn package(compose: &str, files: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let env = dir.path().join("environment");
        std::fs::create_dir_all(&env).unwrap();
        for f in files {
            let p = env.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "FROM alpine\n").unwrap();
        }
        std::fs::write(env.join("docker-compose.yaml"), compose).unwrap();
        dir
    }

    const LABEL: &str = "environment/docker-compose.yaml";

    fn read_package(dir: &tempfile::TempDir) -> (Compose, Vec<Unsupported>) {
        let mut refused = Vec::new();
        let c = read(dir.path(), "environment", LABEL, &mut refused);
        (c, refused)
    }

    fn codes(c: &Compose) -> Vec<&str> {
        c.diagnostics.iter().map(|d| d.code).collect()
    }

    #[test]
    fn literals_never_take_the_hosts_value() {
        assert_eq!(literal("plain").unwrap(), "plain");
        assert_eq!(literal("${TEST_DIR:-/tests}").unwrap(), "/tests");
        assert_eq!(literal("${MODE-fast}").unwrap(), "fast");
        assert_eq!(literal("${X:-}").unwrap(), "");
        assert_eq!(literal("a$$b").unwrap(), "a$b");
        assert_eq!(literal("${A:-${B:-x}/y}").unwrap(), "x/y");
        assert_eq!(literal("${KEY}").unwrap_err(), ["KEY"]);
        assert_eq!(literal("$KEY/x").unwrap_err(), ["KEY"]);
        assert_eq!(literal("${KEY:?set it}").unwrap_err(), ["KEY"]);
        assert_eq!(literal("${KEY:+on}").unwrap_err(), ["KEY"]);
        assert_eq!(literal("${A:-${B}/y}").unwrap_err(), ["B"]);
    }

    #[test]
    fn harbor_template_main_is_accepted_and_the_host_key_dropped() {
        let dir = package(
            include_str!("../tests/fixtures/compose/drone-planning-control.yaml"),
            &["Dockerfile"],
        );
        let (c, refused) = read_package(&dir);
        assert!(refused.is_empty(), "{refused:?}");
        assert!(c.services.is_empty());
        assert!(c.env.is_empty());
        assert_eq!(codes(&c), ["P010", "P011"]);
        let note = &c.diagnostics[0];
        assert_eq!(note.severity, crate::diag::Severity::Note);
        let ignored: Vec<&str> = note
            .notes
            .iter()
            .map(|n| n.split(':').next().unwrap())
            .collect();
        assert_eq!(
            ignored,
            [
                "services.main.build",
                "services.main.image",
                "services.main.command",
                "services.main.network_mode",
                "services.main.volumes ${HOST_VERIFIER_LOGS_PATH}",
                "services.main.volumes ${HOST_AGENT_LOGS_PATH}",
                "services.main.deploy",
            ]
        );
        assert!(c.diagnostics[1].message.contains("`ANTHROPIC_API_KEY`"));
    }

    #[test]
    fn host_variables_in_a_sidecars_command_and_healthcheck_are_warned_about() {
        let dir = package(
            "services:\n  db:\n    image: postgres:16\n    command: [postgres, -c, 'port=${PGPORT}']\n    environment: [A=1, 7]\n    healthcheck:\n      test: [CMD-SHELL, 'pg_isready -U $PGUSER']\n",
            &[],
        );
        let (c, refused) = read_package(&dir);
        let fields: Vec<&str> = refused.iter().map(|u| u.field.as_str()).collect();
        assert_eq!(
            fields,
            ["environment/docker-compose.yaml services.db.environment[1]"]
        );
        let warned: Vec<&str> = c
            .diagnostics
            .iter()
            .filter(|d| d.code == "P011")
            .map(|d| d.message.as_str())
            .collect();
        assert_eq!(warned.len(), 2, "{warned:?}");
        assert!(
            warned[0].starts_with("services.db.command: `port=${PGPORT}`"),
            "{}",
            warned[0]
        );
        assert!(warned[1].contains("`PGUSER`"), "{}", warned[1]);
        // Passed on as written.
        assert_eq!(c.services[0].command[2], "port=${PGPORT}");
    }

    #[test]
    fn sidecar_task_runs_its_api_and_drops_host_values() {
        let dir = package(
            include_str!("../tests/fixtures/compose/fix-visual-stability.yaml"),
            &["Dockerfile", "Dockerfile.api"],
        );
        let (c, refused) = read_package(&dir);
        assert!(refused.is_empty(), "{refused:?}");
        assert_eq!(c.services.len(), 1);
        let api = &c.services[0];
        assert_eq!(api.name, "api");
        assert!(matches!(&api.image, ImageRef::Build { dockerfile, .. }
            if dockerfile == Path::new("environment/Dockerfile.api")));
        // Literal values and defaults reach the agent's environment.
        assert_eq!(c.env.get("TEST_DIR").map(String::as_str), Some("/tests"));
        assert_eq!(
            c.env
                .get("GOOGLE_APPLICATION_CREDENTIALS")
                .map(String::as_str),
            Some("/root/.config/gcloud/application_default_credentials.json")
        );
        assert_eq!(c.env.len(), 2);
        assert_eq!(codes(&c), ["P010", "P011", "P011", "P011", "P012"]);
        let dropped: Vec<&str> = c.diagnostics[1..]
            .iter()
            .map(|d| d.message.as_str())
            .collect();
        for name in [
            "CLAUDE_CODE_USE_VERTEX",
            "CLOUD_ML_REGION",
            "ANTHROPIC_VERTEX_PROJECT_ID",
        ] {
            assert!(
                dropped.iter().any(|m| m.contains(&format!("`{name}`"))),
                "{name}"
            );
        }
        assert!(dropped[3].contains("${HOME}/.config/gcloud"));
        assert!(
            c.diagnostics[0]
                .notes
                .iter()
                .any(|n| n.starts_with("networks.task-net"))
        );
    }

    #[test]
    fn what_benchflow_cannot_do_stays_refused_by_name() {
        let dir = package(
            "services:\n  main:\n    build:\n      context: .\n      dockerfile: Dockerfile.gpu\n    command: [python, serve.py]\n    network_mode: host\n    privileged: true\n    ports: ['8080:8080']\n    volumes:\n      - data:/data\n      - ./fixtures:/fixtures\n    deploy:\n      resources:\n        reservations:\n          devices: [{capabilities: [gpu]}]\n    networks:\n      lan:\n        aliases: [web]\n  db:\n    image: postgres:16\n    cap_add: [NET_ADMIN]\n    networks: [elsewhere]\nnetworks:\n  lan:\n    driver: overlay\nvolumes:\n  data: {}\n",
            &["Dockerfile", "Dockerfile.gpu"],
        );
        let (c, refused) = read_package(&dir);
        let fields: Vec<&str> = refused.iter().map(|u| u.field.as_str()).collect();
        for f in [
            "environment/docker-compose.yaml networks.lan",
            "environment/docker-compose.yaml volumes",
            "environment/docker-compose.yaml services.main.build",
            "environment/docker-compose.yaml services.main.command",
            "environment/docker-compose.yaml services.main.network_mode",
            "environment/docker-compose.yaml services.main.privileged",
            "environment/docker-compose.yaml services.main.ports",
            "environment/docker-compose.yaml services.main.volumes \"data:/data\"",
            "environment/docker-compose.yaml services.main.volumes \"./fixtures:/fixtures\"",
            "environment/docker-compose.yaml services.main.deploy",
            "environment/docker-compose.yaml services.main.networks.lan",
            "environment/docker-compose.yaml services.db.cap_add",
            "environment/docker-compose.yaml services.db.networks elsewhere",
        ] {
            assert!(fields.contains(&f), "{f} not in {fields:#?}");
        }
        assert_eq!(c.services.len(), 1);
        assert!(!codes(&c).contains(&"P012"));
    }

    #[test]
    fn two_networks_are_refused() {
        let dir = package(
            "services:\n  main:\n    networks: [front]\nnetworks:\n  front: {}\n  back: {}\n",
            &[],
        );
        let (_, refused) = read_package(&dir);
        assert_eq!(refused[0].field, "environment/docker-compose.yaml networks");
    }

    /// Guards 0f1eb7b5 (Compose on main): a sidecar's `dockerfile` was not
    /// checked, and a build folder that is a link was followed, so a build
    /// could read files outside the task.
    #[test]
    fn a_sidecar_builds_only_from_the_tasks_own_files() {
        for dockerfile in ["/etc/hosts", "../../Dockerfile", "${HOST_FILE}"] {
            let dir = package(
                &format!(
                    "services:\n  main: {{}}\n  api:\n    build:\n      context: ./api\n      dockerfile: {dockerfile:?}\n"
                ),
                &["api/Dockerfile", "Dockerfile"],
            );
            let (c, refused) = read_package(&dir);
            assert!(c.services.is_empty(), "{dockerfile}");
            assert!(
                refused
                    .iter()
                    .any(|u| u.field.ends_with("services.api.build.dockerfile")),
                "{dockerfile}: {refused:?}"
            );
        }
        #[cfg(unix)]
        {
            let outside = tempfile::tempdir().unwrap();
            std::fs::write(outside.path().join("Dockerfile"), "FROM alpine\n").unwrap();
            let dir = package("services:\n  main: {}\n  api:\n    build: ./api\n", &[]);
            std::os::unix::fs::symlink(outside.path(), dir.path().join("environment/api")).unwrap();
            let (c, refused) = read_package(&dir);
            assert!(c.services.is_empty());
            assert!(
                refused
                    .iter()
                    .any(|u| u.reason.contains("outside the task's folder")),
                "{refused:?}"
            );
        }
    }

    #[test]
    fn services_are_read_and_other_keys_refused_by_name() {
        let dir = package(
            "services:\n  main:\n    depends_on: [api]\n    volumes: [x:/x]\n  api:\n    build:\n      context: ./api\n    expose: ['8000']\n    environment:\n      - MODE=test\n      - TOKEN=${HOST_TOKEN}\n    healthcheck:\n      test: [CMD, wget, -q, http://localhost:8000]\n      interval: 2s\n      retries: 15\n  cache:\n    image: redis:7\n    ports: ['6379:6379']\n",
            &["api/Dockerfile"],
        );
        let (c, refused) = read_package(&dir);
        let services = c.services;
        assert_eq!(services.len(), 2);
        let api = &services[0];
        assert_eq!(api.name, "api");
        assert!(
            matches!(&api.image, ImageRef::Build { context, .. } if context == Path::new("environment/api"))
        );
        assert_eq!(api.env.get("MODE").map(String::as_str), Some("test"));
        assert!(!api.env.contains_key("TOKEN"));
        let ready = api.ready.as_ref().unwrap();
        assert_eq!(ready.run, "wget -q http://localhost:8000");
        assert_eq!(ready.interval, Duration::from_secs(2));
        assert_eq!(ready.retries, 15);
        let fields: Vec<&str> = refused.iter().map(|u| u.field.as_str()).collect();
        assert!(fields.contains(&"environment/docker-compose.yaml services.main.volumes \"x:/x\""));
        assert!(fields.contains(&"environment/docker-compose.yaml services.cache.ports"));
        // A host value in a service's environment is dropped with a warning,
        // as on main.
        assert!(
            c.diagnostics
                .iter()
                .any(|d| d.code == "P011" && d.message.contains("`TOKEN`"))
        );
    }
}
