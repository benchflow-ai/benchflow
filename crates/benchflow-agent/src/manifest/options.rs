//! Agent options (`[options.<name>]`): settings a run passes to one agent
//! with `bench run --ak <name>=<value>`, declared with a type so a wrong
//! name or value is refused before any sandbox starts.
//!
//! Each option says how it reaches the agent:
//!
//! - `env = "VAR"`: a variable for the agent process;
//! - `flag = "--max-turns"`: words added to the launch or headless command,
//!   which must say where they go with `"$@"` (the default ACP command, with
//!   `args`, takes them at its end). `{value}` in the flag is replaced by
//!   the value; without it the value follows as its own word, and a `bool`
//!   option adds the flag when true and nothing when false;
//! - `file` and `key`: a key set in a JSON or TOML file in the agent's home
//!   (`file = ".gemini/settings.json"`, `key = "model.maxSessionTurns"`),
//!   on top of the manifest's own `[[headless.files]]` copy when it has one;
//! - `install = "<package>"`: the version of a package the install pins
//!   (`@anthropic-ai/claude-code`), which changes the installed bits and so
//!   the agent's fingerprint. `default` must be the pinned version; `verify`
//!   lines that name it are updated too.
//!
//! The native agents (`terminus-2`, `builtin`, `computer-1`) declare their
//! options in code, with the same types ([`OptionRoute::Native`]).
//!
//! An option the run does not set is not passed at all: `default` says what
//! the agent does without it.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::{
    AgentManifest, Checker, Harness, HeadlessSpec, HomeFile, Install, InstallMethod, Launch,
    ManifestError, RESERVED_ENV, is_env_name,
};

/// The type of an option's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionKind {
    String,
    Int,
    Float,
    Bool,
    /// One of the option's `choices`.
    Enum,
}

impl OptionKind {
    const NAMES: [&'static str; 5] = ["string", "int", "float", "bool", "enum"];

    fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "string" => Self::String,
            "int" => Self::Int,
            "float" => Self::Float,
            "bool" => Self::Bool,
            "enum" => Self::Enum,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::String => "string",
            Self::Int => "int",
            Self::Float => "float",
            Self::Bool => "bool",
            Self::Enum => "enum",
        }
    }
}

/// An option's value, typed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OptionValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
}

// Floats are checked finite when parsed, so equality is reflexive.
impl Eq for OptionValue {}

impl fmt::Display for OptionValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(x) => write!(f, "{x}"),
            Self::String(s) => f.write_str(s),
        }
    }
}

impl OptionValue {
    fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }

    fn to_toml(&self) -> toml::Value {
        match self {
            Self::Bool(b) => toml::Value::Boolean(*b),
            Self::Int(i) => toml::Value::Integer(*i),
            Self::Float(x) => toml::Value::Float(*x),
            Self::String(s) => toml::Value::String(s.clone()),
        }
    }
}

/// How an option reaches the agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OptionRoute {
    /// A variable for the agent process.
    Env(String),
    /// Words for the command, `{value}` replaced by the value.
    Flag(String),
    /// A key (dotted) in a JSON or TOML file under the agent's home.
    File { path: String, key: String },
    /// The pinned version of this package in the install.
    Install(String),
    /// A setting of a native loop, applied in code.
    Native,
}

impl OptionRoute {
    /// For people: `env CLAUDE_CODE_MAX_TURNS`, `flag --max-turns`.
    pub fn describe(&self) -> String {
        match self {
            Self::Env(var) => format!("env {var}"),
            Self::Flag(flag) => format!("flag {flag}"),
            Self::File { path, key } => format!("~/{path} {key}"),
            Self::Install(package) => format!("installs {package}"),
            Self::Native => "native loop".into(),
        }
    }
}

/// One declared option.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AgentOption {
    /// Lowercase letters, digits and `_` (`max_turns`).
    pub name: String,
    pub kind: OptionKind,
    /// With `enum`: the values it takes.
    pub choices: Vec<String>,
    /// What the agent does when the run does not set it.
    pub default: Option<OptionValue>,
    pub description: String,
    /// With `int` and `float`: the smallest and largest values allowed.
    pub min: Option<OptionValue>,
    pub max: Option<OptionValue>,
    pub route: OptionRoute,
    /// The harnesses it reaches the agent on; empty: every one.
    pub harnesses: Vec<Harness>,
}

impl AgentOption {
    /// What its values look like, for people: `a whole number from 1`,
    /// `one of low, medium, high`.
    pub fn expects(&self) -> String {
        let range = match (&self.min, &self.max) {
            (Some(lo), Some(hi)) => format!(" from {lo} to {hi}"),
            (Some(lo), None) => format!(" from {lo}"),
            (None, Some(hi)) => format!(" up to {hi}"),
            (None, None) => String::new(),
        };
        match self.kind {
            OptionKind::String if matches!(self.route, OptionRoute::Install(_)) => {
                "a version".into()
            }
            OptionKind::String => "text".into(),
            OptionKind::Int => format!("a whole number{range}"),
            OptionKind::Float => format!("a number{range}"),
            OptionKind::Bool => "true or false".into(),
            OptionKind::Enum => format!("one of {}", self.choices.join(", ")),
        }
    }

    /// True when it reaches the agent on `harness`.
    pub fn reaches(&self, harness: Harness) -> bool {
        self.harnesses.is_empty() || self.harnesses.contains(&harness)
    }

    /// Parse a value from the command line; the error says what it takes.
    pub fn parse_value(&self, text: &str) -> Result<OptionValue, String> {
        let wrong = || format!("takes {}, not `{text}`", self.expects());
        let value = match self.kind {
            OptionKind::String => OptionValue::String(text.to_string()),
            OptionKind::Int => OptionValue::Int(text.trim().parse().map_err(|_| wrong())?),
            OptionKind::Float => {
                let x: f64 = text.trim().parse().map_err(|_| wrong())?;
                if !x.is_finite() {
                    return Err(wrong());
                }
                OptionValue::Float(x)
            }
            OptionKind::Bool => match text.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "on" => OptionValue::Bool(true),
                "false" | "0" | "no" | "off" => OptionValue::Bool(false),
                _ => return Err(wrong()),
            },
            OptionKind::Enum => {
                if !self.choices.iter().any(|c| c == text) {
                    let hint =
                        crate::suggest::did_you_mean(text, self.choices.iter().map(String::as_str))
                            .map(|s| format!(" (did you mean `{s}`?)"))
                            .unwrap_or_default();
                    return Err(format!("{}{hint}", wrong()));
                }
                OptionValue::String(text.to_string())
            }
        };
        self.check_range(&value).map_err(|_| wrong())?;
        // Text that is a word of its own after a flag must not read as
        // another flag of the agent's CLI.
        if let OptionRoute::Flag(template) = &self.route
            && self.kind == OptionKind::String
            && text.starts_with('-')
            && (!template.contains("{value}")
                || template.split_whitespace().any(|w| w == "{value}"))
        {
            return Err(format!(
                "takes text that does not start with `-`, not `{text}`: it would read as another flag"
            ));
        }
        if matches!(self.route, OptionRoute::Install(_)) && !is_version(text) {
            return Err(format!(
                "takes a version, such as `1.2.3`, not `{text}` (letters, digits, `.`, `+`, `-` and `_`)"
            ));
        }
        Ok(value)
    }

    fn check_range(&self, value: &OptionValue) -> Result<(), ()> {
        let x = match value {
            OptionValue::Int(i) => *i as f64,
            OptionValue::Float(x) => *x,
            _ => return Ok(()),
        };
        let as_f64 = |v: &Option<OptionValue>| match v {
            Some(OptionValue::Int(i)) => Some(*i as f64),
            Some(OptionValue::Float(x)) => Some(*x),
            _ => None,
        };
        if as_f64(&self.min).is_some_and(|lo| x < lo) || as_f64(&self.max).is_some_and(|hi| x > hi)
        {
            return Err(());
        }
        Ok(())
    }

    /// The words a `flag` option adds to the command for `value`.
    fn flag_words(template: &str, kind: OptionKind, value: &OptionValue) -> Vec<String> {
        let text = value.to_string();
        let has_value = template.contains("{value}");
        if kind == OptionKind::Bool && !has_value && value == &OptionValue::Bool(false) {
            return Vec::new();
        }
        let mut words: Vec<String> = template
            .split_whitespace()
            .map(|w| w.replace("{value}", &text))
            .collect();
        if !has_value && kind != OptionKind::Bool {
            words.push(text);
        }
        words
    }
}

/// Options a run sets, by name, in the order given.
pub type OptionValues = IndexMap<String, OptionValue>;

/// Why a run's options were refused: what is wrong and what to do; the fix
/// is the user's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionError {
    pub message: String,
    pub next: String,
}

impl fmt::Display for OptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for OptionError {}

/// Split `key=value` (`--ak max_turns=30`). Spaces around the key are
/// dropped and `-` in it reads as `_`, so `max-turns=30` works too.
pub fn parse_assignment(text: &str) -> Result<(String, String), OptionError> {
    let Some((key, value)) = text.split_once('=') else {
        return Err(OptionError {
            message: format!("agent option `{text}` has no value"),
            next: "write it as --ak <name>=<value>, such as --ak max_turns=30".into(),
        });
    };
    let key = key.trim().replace('-', "_");
    if key.is_empty() {
        return Err(OptionError {
            message: format!("agent option `{text}` has no name"),
            next: "write it as --ak <name>=<value>, such as --ak max_turns=30".into(),
        });
    }
    Ok((key, value.to_string()))
}

/// Check the options `given` (name and text, in order; a later one wins)
/// against those `agent` declares, for a run on `harness` (`None` for the
/// native loops, whose options all apply).
pub fn resolve(
    agent: &str,
    declared: &[AgentOption],
    given: &[(String, String)],
    harness: Option<Harness>,
) -> Result<OptionValues, OptionError> {
    let mut values = OptionValues::new();
    let list = format!("`bench agents {agent}` lists its options");
    for (name, text) in given {
        let Some(option) = declared.iter().find(|o| &o.name == name) else {
            if declared.is_empty() {
                return Err(OptionError {
                    message: format!("agent `{agent}` takes no options, and --ak {name} was given"),
                    next: format!(
                        "drop --ak {name}; pass a variable the agent reads with --env instead"
                    ),
                });
            }
            let hint = crate::suggest::did_you_mean(name, declared.iter().map(|o| o.name.as_str()))
                .map(|s| format!("did you mean `{s}`? "))
                .unwrap_or_default();
            return Err(OptionError {
                message: format!("agent `{agent}` has no option `{name}`"),
                next: format!(
                    "{hint}it takes {}; {list}",
                    declared
                        .iter()
                        .map(|o| o.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            });
        };
        let value = option.parse_value(text).map_err(|why| OptionError {
            message: format!("option `{name}` of {agent} {why}"),
            next: format!(
                "pass --ak {name}=<{}>; {list}",
                option.kind.as_str().replace("enum", "choice")
            ),
        })?;
        if let Some(h) = harness
            && !option.reaches(h)
        {
            let on: Vec<&str> = option.harnesses.iter().map(|h| h.as_str()).collect();
            return Err(OptionError {
                message: format!(
                    "option `{name}` of {agent} reaches the agent only on the {} harness, and this run uses {h}",
                    on.join(" and ")
                ),
                next: format!("add --harness {}, or drop --ak {name}", on[0]),
            });
        }
        values.insert(name.clone(), value);
    }
    Ok(values)
}

impl AgentManifest {
    /// Check the options `given` for a run on `harness`.
    pub fn resolve_options(
        &self,
        given: &[(String, String)],
        harness: Harness,
    ) -> Result<OptionValues, OptionError> {
        resolve(&self.name, &self.options, given, Some(harness))
    }

    /// This manifest with `values` (checked by [`Self::resolve_options`])
    /// applied: variables, command words, home files, and install pins.
    /// [`Self::chosen_options`] then lists them.
    pub fn with_options(&self, values: &OptionValues) -> AgentManifest {
        let mut m = self.clone();
        let mut acp_words: Vec<String> = Vec::new();
        let mut headless_words: Vec<String> = Vec::new();
        for (name, value) in values {
            let Some(option) = self.options.iter().find(|o| &o.name == name) else {
                continue;
            };
            let acp = option.reaches(Harness::Acp) && self.declares(Harness::Acp);
            let headless = option.reaches(Harness::Headless) && self.declares(Harness::Headless);
            match &option.route {
                OptionRoute::Env(var) => {
                    let text = value.to_string();
                    if acp {
                        m.launch.env.insert(var.clone(), text.clone());
                    }
                    if headless && let Some(h) = m.headless.as_mut() {
                        h.env.insert(var.clone(), text);
                    }
                }
                OptionRoute::Flag(template) => {
                    let words = AgentOption::flag_words(template, option.kind, value);
                    if acp {
                        acp_words.extend(words.iter().cloned());
                    }
                    if headless {
                        headless_words.extend(words);
                    }
                }
                OptionRoute::File { path, key } => {
                    if acp {
                        let base = m
                            .launch
                            .files
                            .iter()
                            .find(|f| &f.path == path)
                            .map(|f| f.contents.clone());
                        let contents = set_key(path, base.as_deref(), key, value)
                            .expect("checked when the manifest was read");
                        upsert(&mut m.launch.files, path, contents);
                    }
                    if headless && let Some(h) = m.headless.as_mut() {
                        let base = h
                            .files
                            .iter()
                            .find(|f| &f.path == path)
                            .map(|f| f.contents.clone());
                        let contents = set_key(path, base.as_deref(), key, value)
                            .expect("checked when the manifest was read");
                        upsert(&mut h.files, path, contents);
                    }
                }
                OptionRoute::Install(package) => {
                    let old = option.default.as_ref().map(|d| d.to_string());
                    repin(&mut m.install, package, old.as_deref(), &value.to_string());
                }
                OptionRoute::Native => {}
            }
        }
        if !acp_words.is_empty() {
            let quoted = quote_words(&acp_words);
            match &mut m.launch.command {
                Some(command) => *command = format!("set -- {quoted}\n{command}"),
                None => m.launch.args.extend(acp_words),
            }
        }
        if !headless_words.is_empty()
            && let Some(h) = m.headless.as_mut()
        {
            h.command = format!("set -- {}\n{}", quote_words(&headless_words), h.command);
        }
        m.chosen_options = values.clone();
        m
    }
}

fn quote_words(words: &[String]) -> String {
    words
        .iter()
        .map(|w| crate::shell::quote(w))
        .collect::<Vec<_>>()
        .join(" ")
}

fn upsert(files: &mut Vec<HomeFile>, path: &str, contents: String) {
    match files.iter_mut().find(|f| f.path == path) {
        Some(f) => f.contents = contents,
        None => files.push(HomeFile {
            path: path.to_string(),
            contents,
            mode: 0o644,
        }),
    }
}

/// A version a package pin takes: `1.2.3`, `2.1.280`, `0.86.2rc1`.
fn is_version(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphanumeric())
        && s.len() <= 64
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '+' | '-' | '_'))
}

/// `contents` (JSON or TOML, by the file's extension; empty for a new file)
/// with `key` (dotted) set to `value`.
fn set_key(
    path: &str,
    contents: Option<&str>,
    key: &str,
    value: &OptionValue,
) -> Result<String, String> {
    let parts: Vec<&str> = key.split('.').collect();
    if path.ends_with(".json") {
        let mut doc: serde_json::Value = match contents.map(str::trim) {
            None | Some("") => serde_json::json!({}),
            Some(text) => serde_json::from_str(text).map_err(|e| format!("is not JSON: {e}"))?,
        };
        let mut at = &mut doc;
        for part in &parts[..parts.len() - 1] {
            let map = at
                .as_object_mut()
                .ok_or_else(|| format!("`{key}` goes through a value that is not an object"))?;
            at = map
                .entry(part.to_string())
                .or_insert_with(|| serde_json::json!({}));
        }
        at.as_object_mut()
            .ok_or_else(|| format!("`{key}` goes through a value that is not an object"))?
            .insert(parts[parts.len() - 1].to_string(), value.to_json());
        serde_json::to_string_pretty(&doc).map_err(|e| e.to_string())
    } else {
        let mut doc: toml::Table = match contents {
            None => toml::Table::new(),
            Some(text) => toml::from_str(text).map_err(|e| format!("is not TOML: {e}"))?,
        };
        let mut at = &mut doc;
        for part in &parts[..parts.len() - 1] {
            at = at
                .entry(part.to_string())
                .or_insert_with(|| toml::Value::Table(toml::Table::new()))
                .as_table_mut()
                .ok_or_else(|| format!("`{key}` goes through a value that is not a table"))?;
        }
        at.insert(parts[parts.len() - 1].to_string(), value.to_toml());
        toml::to_string(&doc).map_err(|e| e.to_string())
    }
}

/// The name and version of an npm spec (`@scope/name@1.2.3`).
fn npm_parts(spec: &str) -> (&str, Option<&str>) {
    let at = if let Some(rest) = spec.strip_prefix('@') {
        rest.find('@').map(|i| i + 1)
    } else {
        spec.find('@')
    };
    match at {
        Some(i) => (&spec[..i], Some(&spec[i + 1..])),
        None => (spec, None),
    }
}

/// The name (without extras) and version of a pip pin (`name[x]==1.2`).
fn pip_parts(spec: &str) -> (&str, Option<&str>) {
    let (head, version) = match spec.split_once("==") {
        Some((h, v)) => (h, Some(v)),
        None => (spec, None),
    };
    (head.split('[').next().unwrap_or(head).trim(), version)
}

/// Every package spec the install pins, mutably, with how to read it.
fn specs_mut(install: &mut Install) -> Vec<(&mut String, bool)> {
    match &mut install.method {
        InstallMethod::Npm(npm) => std::iter::once((&mut npm.package, true))
            .chain(npm.companions.iter_mut().map(|c| (c, true)))
            .collect(),
        InstallMethod::Npx(n) => vec![(&mut n.package, true)],
        InstallMethod::UvTool(u) => vec![(&mut u.requirement, false)],
        InstallMethod::Uvx(u) => vec![(&mut u.package, false)],
        InstallMethod::UvVenv(v) => v.requirements.iter_mut().map(|r| (r, false)).collect(),
        InstallMethod::Script(_) | InstallMethod::Installer(_) | InstallMethod::Binary(_) => {
            Vec::new()
        }
    }
}

/// The version `package` is pinned to in `install`, when it is there.
fn pinned_version(install: &Install, package: &str) -> Option<Option<String>> {
    let mut copy = install.clone();
    specs_mut(&mut copy).into_iter().find_map(|(spec, npm)| {
        let (name, version) = if npm {
            npm_parts(spec)
        } else {
            pip_parts(spec)
        };
        (name == package).then(|| version.map(str::to_string))
    })
}

/// Pin `package` to `version`, and update `verify` lines that name `old`.
fn repin(install: &mut Install, package: &str, old: Option<&str>, version: &str) {
    for (spec, npm) in specs_mut(install) {
        let (name, _) = if npm {
            npm_parts(spec)
        } else {
            pip_parts(spec)
        };
        if name == package {
            let head = if npm {
                name.to_string()
            } else {
                spec.split("==").next().unwrap_or(name).to_string()
            };
            *spec = format!("{head}{}{version}", if npm { "@" } else { "==" });
        }
    }
    let Some(old) = old.filter(|o| !o.is_empty()) else {
        return;
    };
    let verify = match &mut install.method {
        InstallMethod::Npm(n) => n.verify.as_mut(),
        InstallMethod::Npx(n) => n.verify.as_mut(),
        InstallMethod::UvTool(u) => u.verify.as_mut(),
        InstallMethod::Uvx(u) => u.verify.as_mut(),
        InstallMethod::UvVenv(v) => v.verify.as_mut(),
        _ => None,
    };
    if let Some(v) = verify {
        *v = replace_version(v, old, version);
    }
}

/// `text` with each `old` that stands as a whole version (not part of a
/// longer one, such as `2.0.0` in `12.0.0` or `2.0.0-rc1`) replaced by
/// `new`.
pub(super) fn replace_version(text: &str, old: &str, new: &str) -> String {
    let part = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '+' | '_');
    let mut out = String::with_capacity(text.len());
    let mut rest = 0;
    for (at, _) in text.match_indices(old) {
        if at < rest {
            continue;
        }
        let end = at + old.len();
        let before = text[..at].chars().next_back();
        let after = text[end..].chars().next();
        if before.is_some_and(part) || after.is_some_and(part) {
            continue;
        }
        out.push_str(&text[rest..at]);
        out.push_str(new);
        rest = end;
    }
    out.push_str(&text[rest..]);
    out
}

// ---------------------------------------------------------------------------
// The `[options.<name>]` tables
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawOption {
    #[serde(rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    description: String,
    default: Option<toml::Value>,
    #[serde(default)]
    choices: Vec<String>,
    min: Option<toml::Value>,
    max: Option<toml::Value>,
    env: Option<String>,
    flag: Option<String>,
    file: Option<String>,
    key: Option<String>,
    install: Option<String>,
    harnesses: Option<Vec<String>>,
}

/// An option name: lowercase letters, digits and `_`, starting with a letter.
fn is_option_name(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

/// What the rest of the manifest says, for checking where options go.
pub(super) struct Parts<'a> {
    pub install: &'a Install,
    pub launch: &'a Launch,
    pub headless: Option<&'a HeadlessSpec>,
    pub harnesses: &'a [Harness],
}

pub(super) fn validate(
    raw: IndexMap<String, RawOption>,
    check: &Checker<'_>,
    parts: &Parts<'_>,
) -> Result<Vec<AgentOption>, ManifestError> {
    let mut options = Vec::new();
    for (name, raw) in raw {
        options.push(raw.validate(&name, check, parts)?);
    }
    Ok(options)
}

impl RawOption {
    fn validate(
        self,
        name: &str,
        check: &Checker<'_>,
        parts: &Parts<'_>,
    ) -> Result<AgentOption, ManifestError> {
        let at = |key: &str| format!("options.{name}.{key}");
        let here = format!("options.{name}");
        check.require(is_option_name(name), &here, || {
            format!("`{name}` is not an option name; use lowercase letters, digits and `_`, such as `max_turns`")
        })?;
        let kind = match self.kind.as_deref() {
            None => {
                return Err(check.fail(
                    &here,
                    format!(
                        "`type` is required: one of {}",
                        OptionKind::NAMES.join(", ")
                    ),
                ));
            }
            Some(k) => OptionKind::parse(k).ok_or_else(|| {
                let hint = crate::suggest::did_you_mean(k, OptionKind::NAMES)
                    .map(|s| format!("did you mean `{s}`? "))
                    .unwrap_or_default();
                check.fail(
                    &at("type"),
                    format!(
                        "unknown type `{k}`; {hint}use one of {}",
                        OptionKind::NAMES.join(", ")
                    ),
                )
            })?,
        };
        check.require(!self.description.trim().is_empty(), &here, || {
            "`description` is required: one short line for `bench agents`".into()
        })?;
        if kind == OptionKind::Enum {
            check.require(!self.choices.is_empty(), &here, || {
                "an `enum` option lists its `choices`".into()
            })?;
        } else {
            check.require(self.choices.is_empty(), &at("choices"), || {
                "only applies to `type = \"enum\"`".into()
            })?;
        }
        let numeric = matches!(kind, OptionKind::Int | OptionKind::Float);
        for (key, set) in [("min", self.min.is_some()), ("max", self.max.is_some())] {
            check.require(!set || numeric, &at(key), || {
                "only applies to `int` and `float` options".into()
            })?;
        }

        let routes: Vec<&str> = [
            ("env", self.env.is_some()),
            ("flag", self.flag.is_some()),
            ("file", self.file.is_some()),
            ("install", self.install.is_some()),
        ]
        .into_iter()
        .filter_map(|(k, set)| set.then_some(k))
        .collect();
        match routes.as_slice() {
            [] => {
                return Err(check.fail(
                    &here,
                    "say how it reaches the agent: `env = \"<VARIABLE>\"`, `flag = \"--<flag>\"`, \
                     `file` with `key`, or `install = \"<package>\"`",
                ));
            }
            [_] => {}
            [first, second, ..] => {
                return Err(check.fail(
                    &at(second),
                    format!("use either `{first}` or `{second}`, not both"),
                ));
            }
        }
        check.require(
            self.key.is_none() || self.file.is_some(),
            &at("key"),
            || "only applies with `file`".into(),
        )?;

        let mut harnesses = Vec::new();
        if let Some(names) = &self.harnesses {
            for (i, h) in names.iter().enumerate() {
                let key = format!("{}[{i}]", at("harnesses"));
                let harness = Harness::parse(h).ok_or_else(|| {
                    check.fail(&key, format!("unknown harness `{h}`; use acp or headless"))
                })?;
                check.require(parts.harnesses.contains(&harness), &key, || {
                    format!("this agent has no {harness} harness")
                })?;
                harnesses.push(harness);
            }
            check.require(!harnesses.is_empty(), &at("harnesses"), || {
                "lists no harness; leave it out to reach every one".into()
            })?;
        }
        let reaches = |h: Harness| {
            parts.harnesses.contains(&h) && (harnesses.is_empty() || harnesses.contains(&h))
        };

        let route = if let Some(var) = self.env {
            check.require(is_env_name(&var), &at("env"), || {
                format!("`{var}` is not an environment variable name")
            })?;
            check.require(
                !(reaches(Harness::Headless) && RESERVED_ENV.contains(&var.as_str())),
                &at("env"),
                || format!("`{var}` is set by BenchFlow; pick another variable"),
            )?;
            OptionRoute::Env(var)
        } else if let Some(flag) = self.flag {
            check.require(
                !flag.trim().is_empty()
                    && flag.chars().all(|c| {
                        c.is_ascii_alphanumeric()
                            || c == ' '
                            || matches!(c, '-' | '_' | '.' | '=' | '{' | '}' | ':' | ',' | '/')
                    }),
                &at("flag"),
                || {
                    format!(
                        "`{flag}` must be plain words (letters, digits, `-`, `_`, `.`, `=`, `:`, `,`, `/`), with `{{value}}` where the value goes"
                    )
                },
            )?;
            if reaches(Harness::Headless) {
                let command = parts.headless.map(|h| h.command.as_str()).unwrap_or("");
                check.require(command.contains("\"$@\""), &at("flag"), || {
                    "the [headless] command must say where flags go with \"$@\"; or set `harnesses` to those that take it".into()
                })?;
            }
            if reaches(Harness::Acp)
                && let Some(command) = &parts.launch.command
            {
                check.require(command.contains("\"$@\""), &at("flag"), || {
                    "the [launch] command must say where flags go with \"$@\"; or set `harnesses` to those that take it".into()
                })?;
            }
            OptionRoute::Flag(flag)
        } else if let Some(path) = self.file {
            let Some(key) = self.key else {
                return Err(check.fail(
                    &here,
                    "`key` is required with `file`: the setting, dotted (`model.maxSessionTurns`)",
                ));
            };
            check.require(
                (path.ends_with(".json") || path.ends_with(".toml"))
                    && !path.starts_with('/')
                    && !path.split('/').any(|p| p.is_empty() || p == ".."),
                &at("file"),
                || format!("`{path}` must be a .json or .toml file under the agent's home, written relative to it"),
            )?;
            check.require(
                !key.is_empty() && !key.split('.').any(str::is_empty),
                &at("key"),
                || format!("`{key}` is not a dotted key"),
            )?;
            if reaches(Harness::Headless)
                && let Some(base) = parts
                    .headless
                    .and_then(|h| h.files.iter().find(|f| f.path == path))
            {
                set_key(&path, Some(&base.contents), &key, &OptionValue::Bool(true)).map_err(
                    |why| check.fail(&at("file"), format!("the [[headless.files]] copy {why}")),
                )?;
            }
            OptionRoute::File { path, key }
        } else {
            let package = self.install.unwrap_or_default();
            check.require(kind == OptionKind::String, &at("type"), || {
                "an `install` option is a version: `type = \"string\"`".into()
            })?;
            check.require(self.harnesses.is_none(), &at("harnesses"), || {
                "an install serves every harness; remove `harnesses`".into()
            })?;
            match pinned_version(parts.install, &package) {
                None => {
                    return Err(check.fail(
                        &at("install"),
                        format!("the install pins no package `{package}` (install options take an npm, npx, uv-tool, uvx or uv-venv package)"),
                    ));
                }
                Some(None) => {
                    return Err(check.fail(
                        &at("install"),
                        format!("`{package}` is not pinned to a version in the install"),
                    ));
                }
                Some(Some(pinned)) => {
                    let default = self.default.as_ref().and_then(|d| d.as_str());
                    check.require(default == Some(pinned.as_str()), &at("default"), || {
                        format!("must be the version the install pins, \"{pinned}\"")
                    })?;
                }
            }
            OptionRoute::Install(package)
        };

        let probe = AgentOption {
            name: name.to_string(),
            kind,
            choices: self.choices,
            default: None,
            description: self.description.trim().to_string(),
            min: None,
            max: None,
            route,
            harnesses,
        };
        let typed = |key: &str, v: &toml::Value| -> Result<OptionValue, ManifestError> {
            let value = match (kind, v) {
                (OptionKind::Bool, toml::Value::Boolean(b)) => Some(OptionValue::Bool(*b)),
                (OptionKind::Int, toml::Value::Integer(i)) => Some(OptionValue::Int(*i)),
                (OptionKind::Float, toml::Value::Float(x)) if x.is_finite() => {
                    Some(OptionValue::Float(*x))
                }
                (OptionKind::Float, toml::Value::Integer(i)) => Some(OptionValue::Float(*i as f64)),
                (OptionKind::String, toml::Value::String(s)) => {
                    Some(OptionValue::String(s.clone()))
                }
                (OptionKind::Enum, toml::Value::String(s)) if probe.choices.contains(s) => {
                    Some(OptionValue::String(s.clone()))
                }
                _ => None,
            };
            value.ok_or_else(|| check.fail(&at(key), format!("must be {}", probe.expects())))
        };
        let min = self.min.as_ref().map(|v| typed("min", v)).transpose()?;
        let max = self.max.as_ref().map(|v| typed("max", v)).transpose()?;
        let default = self
            .default
            .as_ref()
            .map(|v| typed("default", v))
            .transpose()?;
        let mut option = AgentOption { min, max, ..probe };
        if let Some(d) = &default {
            option
                .check_range(d)
                .map_err(|_| check.fail(&at("default"), format!("must be {}", option.expects())))?;
        }
        option.default = default;
        Ok(option)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npm_and_pip_specs_split_into_name_and_version() {
        assert_eq!(
            npm_parts("@anthropic-ai/claude-code@2.1.280"),
            ("@anthropic-ai/claude-code", Some("2.1.280"))
        );
        assert_eq!(
            npm_parts("opencode-ai@1.18.11"),
            ("opencode-ai", Some("1.18.11"))
        );
        assert_eq!(npm_parts("@scope/x"), ("@scope/x", None));
        assert_eq!(
            pip_parts("strands-agents[openai]==1.59.0"),
            ("strands-agents", Some("1.59.0"))
        );
        assert_eq!(
            pip_parts("aider-chat==0.86.2"),
            ("aider-chat", Some("0.86.2"))
        );
    }

    #[test]
    fn a_key_is_set_in_json_and_toml_on_top_of_what_is_there() {
        let json = set_key(
            ".gemini/settings.json",
            Some(r#"{"model": {"name": "x"}, "a": 1}"#),
            "model.maxSessionTurns",
            &OptionValue::Int(7),
        )
        .unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["model"]["maxSessionTurns"], 7);
        assert_eq!(v["model"]["name"], "x");
        assert_eq!(v["a"], 1);
        let toml = set_key(
            "x/config.toml",
            None,
            "tools.web_search",
            &OptionValue::Bool(false),
        )
        .unwrap();
        assert!(
            toml.contains("[tools]") && toml.contains("web_search = false"),
            "{toml}"
        );
        assert!(set_key("a.json", Some("[1]"), "k", &OptionValue::Bool(true)).is_err());
    }

    #[test]
    fn flags_take_the_value_as_a_word_or_in_place() {
        let words = |t: &str, k: OptionKind, v: OptionValue| AgentOption::flag_words(t, k, &v);
        assert_eq!(
            words("--max-turns", OptionKind::Int, OptionValue::Int(3)),
            ["--max-turns", "3"]
        );
        assert_eq!(
            words(
                "-c web_search={value}",
                OptionKind::Enum,
                OptionValue::String("live".into())
            ),
            ["-c", "web_search=live"]
        );
        assert_eq!(
            words("--sandbox", OptionKind::Bool, OptionValue::Bool(true)),
            ["--sandbox"]
        );
        assert!(words("--sandbox", OptionKind::Bool, OptionValue::Bool(false)).is_empty());
    }
}
