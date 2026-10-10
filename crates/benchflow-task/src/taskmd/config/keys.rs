//! Unknown, misspelled, moved, and Harbor-spelled keys, with fixes.
//!
//! Every table task.md defines is closed: a key it does not define is an
//! error unless it starts with `x-`. The message names the closest key and
//! lists the table's keys, as the reference's does. When renaming the key
//! (and, for a Harbor spelling, rewriting its value) gives a config the
//! checks accept, the diagnostic carries that edit as a [`Fix`].
//!
//! [`Fix`]: crate::diag::Fix

use serde_json::{Map, Value};

use crate::source::join;
use crate::suggest::did_you_mean;

use super::super::code;
use super::super::pyfmt::{float_repr, repr_str};
use super::super::pyval::num;
use super::super::report::At;
use super::super::vocab::{
    ARRAYS, IDENTITY, IMPORT_HINT, KEYS, NESTED, TABLES, enum_values, harbor_name, keys, moved,
    nested as nested_keys, quoted_list, sorted_list,
};
use super::{Checker, toml_literal, values};

/// `"; did you mean X?"`, or `None`: the reference's `did_you_mean`.
pub(crate) fn suggestion(key: &str, allowed: &[&str]) -> Option<String> {
    did_you_mean(key, allowed.iter().copied())
}

/// The reference's `did_you_mean(...) or alt`.
pub(crate) fn dym_or(key: &str, allowed: &[&str], alt: &str) -> String {
    match suggestion(key, allowed) {
        Some(s) => format!("; did you mean {s}?"),
        None => alt.to_string(),
    }
}

/// Top-level keys: identity keys are not tables, tables are tables, and
/// nothing else is allowed.
pub(super) fn top_level(c: &mut Checker<'_>) {
    let Value::Object(top) = c.config else {
        return;
    };
    for (k, v) in top {
        let path = join("", k);
        if IDENTITY.contains(&k.as_str()) {
            if v.is_object() {
                c.err(
                    code::WRONG_SHAPE,
                    At::Key(&path),
                    format!("{k} is an identity key, not a table"),
                );
            }
        } else if !TABLES.contains(&k.as_str()) && !k.starts_with("x-") {
            unknown_top(c, k, &path);
        } else if ARRAYS.contains(&k.as_str()) {
            let ok = matches!(v, Value::Array(items) if items.iter().all(Value::is_object));
            if !ok {
                c.err(
                    code::WRONG_SHAPE,
                    At::Key(&path),
                    format!("{k} must be written as [[{k}]] tables"),
                );
            }
        } else if k != "import" && !k.starts_with("x-") && !v.is_object() {
            c.err(
                code::WRONG_SHAPE,
                At::Value(&path),
                format!("[{k}] must be a table"),
            );
        }
    }
}

/// The help for a Harbor spelling: the reference checker's message names
/// task-md's converter, which bench users may not have.
const HARBOR_HELP: &str = "`bench import <folder>` converts a whole Harbor task into task.md";

fn unknown_top(c: &mut Checker<'_>, key: &str, path: &str) {
    if let Some(hint) = harbor_name(key) {
        let message = format!("{key} is Harbor's name; task.md uses {hint} ({IMPORT_HINT})");
        c.err(code::HARBOR_KEY, At::Key(path), message.clone()).help = Some(HARBOR_HELP.into());
        harbor_fix(c, "", key, path, &message);
        return;
    }
    let all: Vec<&str> = IDENTITY.iter().chain(TABLES).copied().collect();
    let tables: Vec<String> = {
        let mut t: Vec<&str> = TABLES
            .iter()
            .copied()
            .filter(|t| !ARRAYS.contains(t))
            .collect();
        t.sort_unstable();
        t.iter().map(|t| format!("[{t}]")).collect()
    };
    let message = format!(
        "unknown key {key}{} The config block holds the identity keys {}, then the tables {} and [[credits]], or x- extensions",
        dym_or(key, &all, "."),
        sorted_list(IDENTITY),
        tables.join(", ")
    );
    c.err(code::UNKNOWN_KEY, At::Key(path), message.clone());
    rename_fix(c, key, path, "", &all, &message);
}

/// Unknown keys in the closed top-level tables (`[sandbox]`, `[agent]`,
/// `[verifier]`, `[oracle]`, `[runs]`), and their durations, sizes, and
/// networks.
pub(super) fn closed_tables(c: &mut Checker<'_>) {
    for (name, allowed) in KEYS {
        let Some(Value::Object(table)) = c.config.get(*name) else {
            continue;
        };
        let table_path = join("", name);
        for k in table.keys() {
            if allowed.contains(&k.as_str()) || k.starts_with("x-") {
                continue;
            }
            let path = join(&table_path, k);
            let where_ = format!("[{name}] ");
            if let Some(hint) = harbor_name(k) {
                let message =
                    format!("{where_}{k} is Harbor's name; task.md uses {hint} ({IMPORT_HINT})");
                c.err(code::HARBOR_KEY, At::Key(&path), message.clone())
                    .help = Some(HARBOR_HELP.into());
                harbor_fix(c, &table_path, k, &path, &message);
            } else {
                let message = format!(
                    "unknown key {where_}{k}{} [{name}] holds {}, or x- extensions",
                    dym_or(k, allowed, ";"),
                    sorted_list(allowed)
                );
                c.err(code::UNKNOWN_KEY, At::Key(&path), message.clone());
                rename_fix(c, k, &path, &table_path, allowed, &message);
            }
        }
        values::phase(c, table, &format!("[{name}]"), &table_path);
    }
}

/// Offer a rename to the closest allowed key when that fixes the config.
pub(super) fn rename_fix(
    c: &mut Checker<'_>,
    key: &str,
    path: &str,
    table_path: &str,
    allowed: &[&str],
    message: &str,
) {
    let Some(new) = suggestion(key, allowed) else {
        return;
    };
    if let Some(d) = c.sink.found.last_mut() {
        d.diag.label = Some(format!("did you mean `{new}`?"));
    }
    let Some(range) = c.key_range(path) else {
        return;
    };
    let new_path = join(table_path, &new);
    if c.fix_holds(range.clone(), &new, &new_path, message) {
        let fix = c.make_fix(range, new.clone(), format!("rename `{key}` to `{new}`"));
        if let (Some(fix), Some(d)) = (fix, c.sink.found.last_mut()) {
            d.diag.fix = Some(fix);
        }
    }
}

/// What a Harbor spelling becomes in task.md: the key, and the value when it
/// changes too. `None` when there is no direct equivalent.
fn harbor_equivalent(
    table_path: &str,
    key: &str,
    value: &Value,
) -> Option<(String, Option<Value>)> {
    let in_table = !table_path.is_empty();
    let renamed = |k: &str| Some((k.to_string(), None));
    match key {
        "environment" if !in_table => renamed("sandbox"),
        "environment" if table_path == "verifier" => renamed("sandbox"),
        "solution" if !in_table => renamed("oracle"),
        "metadata" if !in_table => renamed("about"),
        "docker_image" => renamed("image"),
        "skills_dir" => renamed("skills"),
        "mcp_servers" => renamed("mcp"),
        "environment_mode" => renamed("isolation"),
        "healthcheck" => renamed("ready"),
        "collect" => renamed("snapshot"),
        "memory_mb" => Some(("memory".into(), Some(Value::String(harbor_size(value)?)))),
        "storage_mb" => Some(("disk".into(), Some(Value::String(harbor_size(value)?)))),
        "timeout_sec" => Some((
            "timeout".into(),
            Some(Value::String(harbor_duration(value)?)),
        )),
        "build_timeout_sec" => Some((
            "build_timeout".into(),
            Some(Value::String(harbor_duration(value)?)),
        )),
        "network_mode" => {
            let net = match value.as_str()? {
                "no-network" => "none",
                "public" => "open",
                _ => return None,
            };
            Some(("network".into(), Some(Value::String(net.into()))))
        }
        "allow_internet" => {
            let net = if value.as_bool()? { "open" } else { "none" };
            Some(("network".into(), Some(Value::String(net.into()))))
        }
        "allowed_hosts"
            if value
                .as_array()
                .is_some_and(|a| a.iter().all(Value::is_string)) =>
        {
            Some(("network".into(), Some(value.clone())))
        }
        _ => None,
    }
}

/// Harbor's seconds as a task.md duration, as `tools/convert.py import`
/// writes them: 600 is `"10m"`, 5400 is `"1h30m"`, 0.5 is `"0.5s"`.
pub(crate) fn harbor_duration(value: &Value) -> Option<String> {
    let sec = num(value)?;
    if sec.fract() == 0.0 && sec >= 0.0 {
        let s = sec as u64;
        let (h, rest) = (s / 3600, s % 3600);
        let (m, s) = (rest / 60, rest % 60);
        let mut out = String::new();
        if h > 0 {
            out.push_str(&format!("{h}h"));
        }
        if m > 0 {
            out.push_str(&format!("{m}m"));
        }
        if s > 0 {
            out.push_str(&format!("{s}s"));
        }
        return Some(if out.is_empty() { "0s".into() } else { out });
    }
    Some(format!("{}s", float_repr(sec)))
}

/// Harbor's megabytes as a task.md size, as `tools/convert.py import` writes
/// them: 4096 is `"4 GB"`, 1536 is `"1536 MB"`.
pub(crate) fn harbor_size(value: &Value) -> Option<String> {
    let mb = num(value)?.trunc();
    if mb < 0.0 {
        return None;
    }
    let mb = mb as u64;
    for (unit, size) in [("TB", 1024 * 1024), ("GB", 1024)] {
        if mb > 0 && mb.is_multiple_of(size) {
            return Some(format!("{} {unit}", mb / size));
        }
    }
    Some(format!("{mb} MB"))
}

/// Offer the task.md spelling of a Harbor key, rewriting its value too when
/// the unit or form differs (`timeout_sec = 600` becomes `timeout = "10m"`).
fn harbor_fix(c: &mut Checker<'_>, table_path: &str, key: &str, path: &str, message: &str) {
    if key == "schema_version" && table_path.is_empty() {
        removal_fix(
            c,
            path,
            "remove `schema_version`, which task.md does not have",
            message,
        );
        return;
    }
    let value = match table_path {
        "" => c.config.get(key),
        t => super::super::pyval::table(c.config, &[t]).get(key),
    };
    let Some(value) = value else {
        return;
    };
    let Some((new, new_value)) = harbor_equivalent(table_path, key, value) else {
        return;
    };
    if let Some(d) = c.sink.found.last_mut() {
        d.diag.label = Some(format!("task.md spells this `{new}`"));
    }
    let new_path = join(table_path, &new);
    let (range, replacement, description) = match &new_value {
        None => {
            let Some(range) = c.key_range(path) else {
                return;
            };
            (range, new.clone(), format!("rename `{key}` to `{new}`"))
        }
        Some(v) => {
            let (Some(range), Some(lit), Some(old)) =
                (c.key_to_value(path), toml_literal(v), toml_literal(value))
            else {
                return;
            };
            (
                range,
                format!("{new} = {lit}"),
                format!("replace `{key} = {old}` with `{new} = {lit}`"),
            )
        }
    };
    if c.fix_holds(range.clone(), &replacement, &new_path, message)
        && let Some(fix) = c.make_fix(range, replacement, description)
        && let Some(d) = c.sink.found.last_mut()
    {
        d.diag.fix = Some(fix);
    }
}

/// Offer to delete a key and its value (the whole line when nothing else is
/// on it), when that leaves the config without this error.
pub(super) fn removal_fix(c: &mut Checker<'_>, path: &str, description: &str, message: &str) {
    let Some(range) = c.key_to_value(path) else {
        return;
    };
    let range = c.whole_lines(range);
    if c.fix_holds(range.clone(), "", path, message)
        && let Some(fix) = c.make_fix(range, String::new(), description.to_string())
        && let Some(d) = c.sink.found.last_mut()
    {
        d.diag.fix = Some(fix);
    }
}

/// The rule that governs the table at `rule_path` (the reference's
/// `_nested_rule`): a variant's patch and a stage's agent and verifier
/// tables follow the rules of the tables they stand for.
fn nested_rule(rule_path: &[String]) -> Option<String> {
    if rule_path.len() >= 3 && rule_path[0] == "variants" {
        return nested_rule(&rule_path[2..]);
    }
    if rule_path.len() >= 3
        && rule_path[0] == "stages"
        && (rule_path[2] == "agent" || rule_path[2] == "verifier")
    {
        return nested_rule(&rule_path[2..]);
    }
    if rule_path.len() == 1 && keys(&rule_path[0]).is_some() {
        return Some(rule_path[0].clone());
    }
    NESTED.iter().find_map(|(pattern, _)| {
        let parts: Vec<&str> = pattern.split('.').collect();
        (parts.len() == rule_path.len()
            && parts
                .iter()
                .zip(rule_path)
                .all(|(p, q)| *p == "*" || p == q))
        .then(|| pattern.to_string())
    })
}

/// Unknown keys below the top level, moved keys, and documented values: the
/// reference's `_check_nested`.
pub(super) fn nested(c: &mut Checker<'_>) {
    let Value::Object(top) = c.config else {
        return;
    };
    for (k, v) in top {
        match v {
            Value::Object(t) if k != "about" && k != "import" && !k.starts_with("x-") => {
                walk(c, t, std::slice::from_ref(k), &join("", k));
            }
            Value::Array(items) if ARRAYS.contains(&k.as_str()) => {
                for (i, x) in items.iter().enumerate() {
                    if let Value::Object(t) = x {
                        walk(c, t, &[format!("{k}[]")], &format!("{}[{i}]", join("", k)));
                    }
                }
            }
            _ => {}
        }
    }
    let stages = super::super::pyval::table(c.config, &["stages"]);
    for (name, stage) in stages {
        let Some(unlock) = stage.get("unlock") else {
            continue;
        };
        if !stage.is_object() || unlock.as_str().is_some_and(super::values::unlock_ok) {
            continue;
        }
        let path = join(&join("stages", name), "unlock");
        c.err(
            code::STAGES,
            At::Value(&path),
            format!("[stages.{name}] unlock must be on_submit, on_request, at_start, after:<stage>, or at_turn:<n>"),
        );
    }
}

fn walk(c: &mut Checker<'_>, table: &Map<String, Value>, rule_path: &[String], path: &str) {
    let rule = nested_rule(rule_path);
    let allowed: Option<&[&str]> = rule
        .as_deref()
        .and_then(|r| keys(r).or_else(|| nested_keys(r)));
    let checked_above = rule_path.len() == 1 && keys(&rule_path[0]).is_some();
    let where_ = format!(
        "[{}]",
        rule_path
            .iter()
            .map(|x| x.strip_suffix("[]").unwrap_or(x))
            .collect::<Vec<_>>()
            .join(".")
    );
    for (k, v) in table {
        if k.starts_with("x-") {
            continue;
        }
        let key_path = join(path, k);
        if let Some(allowed) = allowed
            && !allowed.contains(&k.as_str())
        {
            let rule_key = format!("{}.{k}", rule.as_deref().unwrap_or(""));
            if rule.as_deref() == Some("world.desktop") && k == "clock" {
                let message = super::clock::desktop_clock_moved(v, table);
                c.err(code::KEY_GONE, At::Key(&key_path), message);
            } else if let Some(m) = moved(&rule_key) {
                c.err(code::KEY_GONE, At::Key(&key_path), m);
                if m.ends_with("Remove the key") {
                    removal_fix(c, &key_path, &format!("remove `{k}`"), m);
                }
            } else if !checked_above {
                let message = format!(
                    "unknown key {} in {where_}{} it holds {}, or x- extensions",
                    repr_str(k),
                    dym_or(k, allowed, ";"),
                    sorted_list(allowed)
                );
                c.err(code::UNKNOWN_KEY, At::Key(&key_path), message.clone());
                rename_fix(c, k, &key_path, path, allowed, &message);
            }
            continue;
        }
        let rule_key = rule.as_ref().map(|r| format!("{r}.{k}"));
        let documented = rule_key.as_deref().and_then(enum_values);
        if rule.as_deref() == Some("training") && k == "reward" && v.as_str() == Some("binary") {
            let message = format!(
                "{where_} reward \"binary\" is spelled \"strict\", the rubric's name for the all-or-nothing score: write reward = \"strict\""
            );
            c.warn(code::TRAINING, At::Value(&key_path), message.clone());
            value_fix(
                c,
                &key_path,
                "\"strict\"",
                "write \"strict\" for \"binary\"",
                &message,
            );
        } else if let Some(values) = documented
            && !matches!(v, Value::String(s) if values.contains(&s.as_str()))
        {
            let message = format!("{where_} {k} must be one of {}", quoted_list(values));
            c.err(code::BAD_CHOICE, At::Value(&key_path), message.clone());
            if let Value::String(s) = v
                && let Some(close) = suggestion(s, values)
            {
                choice_fix(c, &key_path, &close, &message);
            }
        }
        match v {
            Value::Object(t) => {
                let mut next = rule_path.to_vec();
                next.push(k.clone());
                walk(c, t, &next, &key_path);
            }
            Value::Array(items) if !items.is_empty() && items.iter().all(Value::is_object) => {
                let mut next = rule_path.to_vec();
                next.push(format!("{k}[]"));
                for (i, x) in items.iter().enumerate() {
                    if let Value::Object(t) = x {
                        walk(c, t, &next, &format!("{key_path}[{i}]"));
                    }
                }
            }
            _ => {}
        }
    }
}

/// Offer to replace a value with `literal` (TOML text), when that fixes it.
pub(super) fn value_fix(
    c: &mut Checker<'_>,
    path: &str,
    literal: &str,
    description: &str,
    message: &str,
) {
    let Some(range) = c.value_range(path) else {
        return;
    };
    if c.fix_holds(range.clone(), literal, path, message)
        && let Some(fix) = c.make_fix(range, literal.to_string(), description.to_string())
        && let Some(d) = c.sink.found.last_mut()
    {
        d.diag.fix = Some(fix);
    }
}

/// Offer the closest documented value for a misspelled one.
pub(super) fn choice_fix(c: &mut Checker<'_>, path: &str, close: &str, message: &str) {
    if let Some(d) = c.sink.found.last_mut() {
        d.diag.label = Some(format!("did you mean \"{close}\"?"));
    }
    let literal = toml_literal(&Value::String(close.to_string())).unwrap_or_default();
    value_fix(c, path, &literal, &format!("write {literal}"), message);
}

/// Whether `v` is a whole number of seconds that a duration fix can rewrite.
pub(super) fn seconds_like(v: &Value) -> Option<f64> {
    match v {
        Value::Number(_) => num(v).filter(|s| *s >= 0.0),
        Value::String(s) if !s.is_empty() && s.trim().chars().all(|c| c.is_ascii_digit()) => {
            s.trim().parse().ok()
        }
        _ => None,
    }
}

/// Whether the config's `ENUMS` and `MOVED` tables cover every rule they
/// name (a consistency check for the vocabulary).
#[cfg(test)]
pub(crate) fn vocabulary_is_consistent() -> bool {
    use super::super::vocab::{ENUMS, MOVED};
    let rules: Vec<&str> = NESTED
        .iter()
        .map(|(r, _)| *r)
        .chain(KEYS.iter().map(|(t, _)| *t))
        .collect();
    let rule_of = |k: &str| {
        k.rsplit_once('.')
            .map(|(r, _)| r.to_string())
            .unwrap_or_default()
    };
    ENUMS
        .iter()
        .all(|(k, _)| rules.contains(&rule_of(k).as_str()))
        && MOVED
            .iter()
            .all(|(k, _)| rules.contains(&rule_of(k).as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn harbor_values_convert_as_the_importer_does() {
        assert_eq!(harbor_duration(&json!(600)).as_deref(), Some("10m"));
        assert_eq!(harbor_duration(&json!(5400)).as_deref(), Some("1h30m"));
        assert_eq!(harbor_duration(&json!(3661)).as_deref(), Some("1h1m1s"));
        assert_eq!(harbor_duration(&json!(0)).as_deref(), Some("0s"));
        assert_eq!(harbor_duration(&json!(0.5)).as_deref(), Some("0.5s"));
        assert_eq!(harbor_size(&json!(4096)).as_deref(), Some("4 GB"));
        assert_eq!(harbor_size(&json!(1536)).as_deref(), Some("1536 MB"));
        assert_eq!(harbor_size(&json!(1048576)).as_deref(), Some("1 TB"));
    }

    #[test]
    fn nested_rules_follow_what_a_table_stands_for() {
        let p = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            nested_rule(&p(&["variants", "v", "agent"])).as_deref(),
            Some("agent")
        );
        assert_eq!(
            nested_rule(&p(&["stages", "s", "verifier", "judges"])).as_deref(),
            Some("verifier.judges")
        );
        assert_eq!(
            nested_rule(&p(&["stages", "s", "ready"])).as_deref(),
            Some("stages.*.ready")
        );
        assert_eq!(
            nested_rule(&p(&["world", "safety", "limit[]"])).as_deref(),
            Some("world.safety.limit[]")
        );
        assert_eq!(nested_rule(&p(&["sandbox", "env"])), None);
        assert!(vocabulary_is_consistent());
    }
}
