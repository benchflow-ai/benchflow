//! A BenchFlow 0.8 ("v0.6") task.md package to task.md, ported from task.md's
//! `tools/convert.py import-v06` (`import_v06`, `v06_to_v1`,
//! `config_from_v06`).
//!
//! The prompt moves to the top; `## role:<name>`, `## scene:<name>`, and
//! `## user-persona` sections (or their `prompts/` files) become `role`,
//! `stage`, and `user` blocks; the frontmatter becomes a ` ```toml task `
//! block in task.md's names (0.8 took its config model from Harbor, so the
//! rewrite is a Harbor import). `benchflow:` and `robouse:` stay whole as
//! `[x-benchflow]` and `[x-robouse]`, and 0.8 settings task.md does not
//! define are kept under `[import.v06]`.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

use super::doc;
use super::files::{self, Rel};
use super::harbor::{self, ordered};
use super::import::declare_models;
use super::rewardkit;
use super::tomltext::{self, Comment, Comments};
use super::value::{Table, Val, parse_toml, parse_yaml};
use super::{ConvertError, ImportPlan, io_err};

static SECTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^## (prompt|role:[^\n]+|scene:[^\n]+|user-persona)[ \t]*$").unwrap()
});
static SIDECAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(role|scene)\.(.+)\.md$|^(user-persona)\.md$").unwrap());
static ROLE_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^## role:[ \t]*(\S+)").unwrap());
static QUOTED_HASH: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"["'][^"']*#"#).unwrap());

const TABLE_NAMES: [(&str, &str); 2] = [("sandbox", "environment"), ("oracle", "solution")];
const VERIFIER_NAMES: [(&str, &str); 2] = [
    ("sandbox_mode", "environment_mode"),
    ("sandbox", "environment"),
];
const NAMESPACES: [&str; 2] = ["benchflow", "robouse"];
const V1_ORDER: [&str; 17] = [
    "name",
    "title",
    "version",
    "description",
    "authors",
    "keywords",
    "about",
    "agent",
    "sandbox",
    "world",
    "stages",
    "roles",
    "user",
    "interaction",
    "verifier",
    "oracle",
    "provenance",
];
const V06_DIRS: [(&str, &str); 3] = [
    ("environment", "sandbox"),
    ("tests", "verifier"),
    ("solution", "oracle"),
];
/// The keys task.md's `[roles.<name>]`, `[user]`, and `[interaction]` hold.
const ROLE_KEYS: [&str; 5] = ["tools", "workspace", "model", "harness", "policy"];
const USER_KEYS: [&str; 5] = ["model", "max_turns", "stop", "isolation", "residual_risk"];
const INTERACTION_KEYS: [&str; 2] = ["mode", "handoff"];
/// Where 0.8 writes an acceptance-live report by default.
const LIVE_REPORT: &str = "evidence/acceptance/live-report.json";

/// The frontmatter, prompt, and other sections of a 0.8 task.md, with 0.8's
/// rule: `## prompt`, else the text before the first section, else the body.
fn split_v06(task_md: &str) -> (Option<String>, String, IndexMap<String, String>) {
    let (front, body) = if let Some(rest) = task_md.strip_prefix("---\n") {
        match rest.find("\n---\n") {
            Some(i) => (Some(rest[..i].to_string()), rest[i + 5..].to_string()),
            None => (None, task_md.to_string()),
        }
    } else {
        (None, task_md.to_string())
    };
    let heads: Vec<regex::Captures<'_>> = SECTION.captures_iter(&body).collect();
    if heads.is_empty() {
        return (front, body.trim_matches('\n').to_string(), IndexMap::new());
    }
    let mut sections: IndexMap<String, String> = IndexMap::new();
    for (n, h) in heads.iter().enumerate() {
        let end = heads
            .get(n + 1)
            .map_or(body.len(), |x| x.get(0).expect("match").start());
        let start = h.get(0).expect("match").end();
        sections.insert(
            h[1].to_string(),
            body[start..end].trim_matches('\n').to_string(),
        );
    }
    let first = heads[0].get(0).expect("match").start();
    let prompt = sections
        .shift_remove("prompt")
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| body[..first].trim_matches('\n').to_string());
    (front, prompt, sections)
}

/// YAML allows nulls and TOML does not: drop them, with a note.
fn toml_safe(v: Val, path: &str, notes: &mut Vec<String>) -> Val {
    match v {
        Val::Table(t) => {
            let mut out = Table::new();
            for (k, x) in t {
                if x == Val::Null {
                    notes.push(format!(
                        "dropped {path}{k}: null (TOML has no null; an absent key means the same to the runtime)"
                    ));
                    continue;
                }
                let child = format!("{path}{k}.");
                out.insert(k, toml_safe(x, &child, notes));
            }
            Val::Table(out)
        }
        Val::List(items) => Val::List(
            items
                .into_iter()
                .filter(|x| *x != Val::Null)
                .map(|x| toml_safe(x, path, notes))
                .collect(),
        ),
        other => other,
    }
}

fn table_mut<'t>(t: &'t mut Table, key: &str) -> &'t mut Table {
    let entry = t
        .entry(key.to_string())
        .or_insert_with(|| Val::Table(Table::new()));
    if !matches!(entry, Val::Table(_)) {
        *entry = Val::Table(Table::new());
    }
    entry.as_table_mut().expect("a table")
}

fn merge(dst: &mut Table, src: &Table) {
    for (k, v) in src {
        match (dst.get_mut(k), v) {
            (Some(Val::Table(d)), Val::Table(s)) => merge(d, s),
            _ => {
                dst.insert(k.clone(), v.clone());
            }
        }
    }
}

/// Switch a Harbor-shaped table to 0.8's names (sandbox, oracle,
/// sandbox_mode), in steps too.
fn rename_to_v06(d: &mut Table) {
    for (v06, harbor) in TABLE_NAMES {
        if let Some(v) = d.shift_remove(harbor) {
            d.insert(v06.to_string(), v);
        }
    }
    let rename = |ver: &mut Val| {
        if let Val::Table(ver) = ver {
            for (v06, harbor) in VERIFIER_NAMES {
                if let Some(v) = ver.shift_remove(harbor) {
                    ver.insert(v06.to_string(), v);
                }
            }
        }
    };
    if let Some(v) = d.get_mut("verifier") {
        rename(v);
    }
    match d.get_mut("steps") {
        Some(Val::List(steps)) => {
            for st in steps {
                if let Some(v) = st.as_table_mut().and_then(|t| t.get_mut("verifier")) {
                    rename(v);
                }
            }
        }
        Some(Val::Table(steps)) => {
            for st in steps.values_mut() {
                if let Some(v) = st.as_table_mut().and_then(|t| t.get_mut("verifier")) {
                    rename(v);
                }
            }
        }
        _ => {}
    }
}

/// Move keys task.md does not define out of `[roles.<name>]`, `[user]`,
/// and `[interaction]`, and return them.
fn split_unmodeled(cfg: &mut Table) -> Table {
    let mut out = Table::new();
    for table in ["roles", "user", "interaction"] {
        let Some(Val::Table(t)) = cfg.get_mut(table) else {
            continue;
        };
        if table == "roles" {
            for (name, role) in t.iter_mut() {
                let Val::Table(role) = role else { continue };
                let extra_keys: Vec<String> = role
                    .keys()
                    .filter(|k| !ROLE_KEYS.contains(&k.as_str()) && !k.starts_with("x-"))
                    .cloned()
                    .collect();
                let mut extra = Table::new();
                for k in extra_keys {
                    if let Some(v) = role.shift_remove(&k) {
                        extra.insert(k, v);
                    }
                }
                if !extra.is_empty() {
                    table_mut(&mut out, "roles").insert(name.clone(), Val::Table(extra));
                }
            }
        } else {
            let allowed: &[&str] = if table == "user" {
                &USER_KEYS
            } else {
                &INTERACTION_KEYS
            };
            let extra_keys: Vec<String> = t
                .keys()
                .filter(|k| !allowed.contains(&k.as_str()) && !k.starts_with("x-"))
                .cloned()
                .collect();
            let mut extra = Table::new();
            for k in extra_keys {
                if let Some(v) = t.shift_remove(&k) {
                    extra.insert(k, v);
                }
            }
            let now_empty = t.is_empty();
            if !extra.is_empty() {
                out.insert(table.to_string(), Val::Table(extra));
            }
            if now_empty {
                cfg.shift_remove(table);
            }
        }
    }
    out
}

/// Dotted paths of a table, two levels deep.
fn paths(d: &Table, pre: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (k, v) in d {
        let p = if pre.is_empty() {
            k.clone()
        } else {
            format!("{pre}.{k}")
        };
        match v {
            Val::Table(t) if pre.matches('.').count() < 1 => out.extend(paths(t, &p)),
            _ => out.push(p),
        }
    }
    out
}

/// 0.8 frontmatter data to task.md config, with notes.
fn config_from_v06(front: Val) -> Result<(Table, Vec<String>), String> {
    let mut notes = Vec::new();
    let Val::Table(mut d) = toml_safe(front, "", &mut notes) else {
        return Ok((Table::new(), notes));
    };
    for (v06, split) in TABLE_NAMES {
        if d.contains_key(v06) && d.contains_key(split) {
            return Err(format!(
                "frontmatter has both {v06} and {split}; v0.6 treats them as one table"
            ));
        }
        if d.contains_key(split) {
            notes.push(format!("{split} is v0.6's legacy spelling of {v06}"));
        } else if let Some(v) = d.shift_remove(v06) {
            d.insert(split.to_string(), v);
        }
    }
    if let Some(Val::Table(ver)) = d.get_mut("verifier") {
        for (v06, split) in VERIFIER_NAMES {
            if let Some(v) = ver.shift_remove(v06) {
                ver.insert(split.to_string(), v);
            }
        }
    }
    let mut extra = Table::new();
    match d.shift_remove("agents") {
        Some(Val::Table(agents)) => {
            if let Some(roles @ Val::Table(_)) = agents.get("roles") {
                extra.insert("roles".into(), roles.clone());
            }
            let rest: Table = agents.into_iter().filter(|(k, _)| k != "roles").collect();
            if !rest.is_empty() {
                table_mut(&mut extra, "interaction").insert("agents".into(), Val::Table(rest));
            }
        }
        Some(other) => {
            table_mut(&mut extra, "interaction").insert("agents".into(), other);
        }
        None => {}
    }
    if let Some(scenes) = d.shift_remove("scenes") {
        table_mut(&mut extra, "interaction").insert("scenes".into(), scenes);
    }
    if let Some(user) = d.shift_remove("user") {
        extra.insert("user".into(), user);
    }
    for ns in NAMESPACES {
        if let Some(v) = d.shift_remove(ns) {
            extra.insert(format!("x-{ns}"), v);
        }
    }
    let imported = harbor::config_from_harbor(&d, "v06")?;
    let mut native = imported.config;
    notes.extend(imported.notes.into_iter().map(|n| {
        n.replace(
            "a deprecated split-layout spelling",
            "v0.6's deprecated spelling",
        )
    }));
    if let Some(Val::Table(kept)) = native
        .get_mut("import")
        .and_then(|i| i.as_table_mut())
        .and_then(|i| i.get_mut("v06"))
    {
        rename_to_v06(kept);
    }
    for (k, v) in extra {
        match (native.get_mut(&k), &v) {
            (Some(Val::Table(n)), Val::Table(e)) => merge(n, e),
            _ => {
                native.insert(k, v);
            }
        }
    }
    let unmodeled = split_unmodeled(&mut native);
    if !unmodeled.is_empty() {
        merge(
            table_mut(table_mut(&mut native, "import"), "v06"),
            &unmodeled,
        );
        let mut p = paths(&unmodeled, "");
        p.sort();
        notes.push(format!("kept under [import.v06]: {}", p.join(", ")));
    }
    Ok((ordered(&native, &order(&native)), notes))
}

fn order(native: &Table) -> Vec<&str> {
    let mut xs: Vec<&str> = native
        .keys()
        .filter(|k| k.starts_with("x-"))
        .map(String::as_str)
        .collect();
    xs.sort();
    V1_ORDER
        .iter()
        .copied()
        .chain(xs)
        .chain(["import"])
        .collect()
}

fn yaml_comments(front: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in front.split('\n') {
        let s = line.trim();
        if s.starts_with('#') {
            out.push(s.to_string());
        } else if let Some(i) = line.find(" #")
            && !QUOTED_HASH.is_match(line)
        {
            out.push(line[i + 1..].trim().to_string());
        }
    }
    out
}

/// What a 0.8 package's folders add to its config.
#[derive(Default)]
struct Package {
    /// The bundled skills folder, where import puts it.
    skills: Option<String>,
    /// A record of verifier/verifier.md.
    verifier_md: Option<Table>,
    /// RewardKit judge files in the verifier's folder.
    rk_judges: Vec<String>,
}

fn verifier_document(path: &Path, rel: &str) -> Table {
    let mut record = Table::new();
    record.insert("path".into(), Val::Str(rel.to_string()));
    let text = std::fs::read(path)
        .map(|b| String::from_utf8_lossy(&b).replace("\r\n", "\n"))
        .unwrap_or_default();
    let (front, body) = match text
        .strip_prefix("---\n")
        .and_then(|r| r.find("\n---\n").map(|i| (r, i)))
    {
        Some((r, i)) => (Some(&r[..i]), &r[i + 5..]),
        None => (None, text.as_str()),
    };
    let data = front.and_then(|f| parse_yaml(f).ok());
    if let Some(Val::Table(ver)) = data
        .as_ref()
        .and_then(Val::as_table)
        .and_then(|d| d.get("verifier"))
    {
        if let Some(Val::Str(d)) = ver.get("default_strategy") {
            record.insert("default_strategy".into(), Val::Str(d.clone()));
        }
        if let Some(Val::Table(strategies)) = ver.get("strategies") {
            let s: Table = strategies
                .iter()
                .map(|(n, s)| {
                    let kind = match s {
                        Val::Table(t) => match t.get("type") {
                            Some(Val::Str(k)) => k.clone(),
                            Some(other) => harbor::py_repr(other),
                            None => "unknown".into(),
                        },
                        _ => "unknown".into(),
                    };
                    (n.clone(), Val::Str(kind))
                })
                .collect();
            record.insert("strategies".into(), Val::Table(s));
        }
    }
    let roles: Vec<Val> = ROLE_HEADING
        .captures_iter(body)
        .map(|c| Val::Str(c[1].to_string()))
        .collect();
    if !roles.is_empty() {
        record.insert("roles".into(), Val::List(roles));
    }
    record
}

fn package_of(src: &Path, placed: &IndexMap<Rel, Rel>) -> Package {
    let moved: Vec<(String, String)> = placed
        .iter()
        .map(|(new, old)| (files::posix(old), files::posix(new)))
        .collect();
    let mut out = Package {
        skills: moved
            .iter()
            .find(|(old, _)| old.starts_with("environment/skills/"))
            .map(|(_, new)| format!("{}/skills", new.split('/').next().unwrap_or_default())),
        ..Package::default()
    };
    for old in ["verifier/verifier.md", "tests/verifier.md"] {
        if let Some((_, new)) = moved.iter().find(|(o, _)| o == old) {
            out.verifier_md = Some(verifier_document(&src.join(old), new));
            break;
        }
    }
    let mut sorted = moved.clone();
    sorted.sort();
    for (old, new) in sorted {
        if !(new.starts_with("verifier/") && new.ends_with(".toml")) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(src.join(&old)) else {
            continue;
        };
        let Ok(data) = parse_toml(&text) else {
            continue;
        };
        if (data.contains_key("judge") || data.contains_key("criterion"))
            && rewardkit::is_judge_toml(&data)
        {
            out.rk_judges.push(new);
        }
    }
    out
}

fn add_package(native: &mut Table, package: &Package, notes: &mut Vec<String>) {
    if let Some(skills) = &package.skills {
        let current = native
            .get("sandbox")
            .and_then(Val::as_table)
            .and_then(|s| s.get("skills"))
            .cloned();
        if let Some(cur) = current.filter(|c| c.as_str() != Some(skills.as_str())) {
            let shown = harbor::py_repr(&cur);
            table_mut(table_mut(table_mut(native, "import"), "v06"), "sandbox")
                .insert("skills_dir".into(), cur);
            notes.push(format!(
                "sandbox.skills_dir = {shown} is where the v0.6 runtime mounts the bundled skills; kept under [import.v06]"
            ));
        }
        table_mut(native, "sandbox").insert("skills".into(), Val::Str(skills.clone()));
        notes.push(format!(
            "environment/skills/ holds the task's bundled skills, which the v0.6 runtime installs by convention in its with-skill mode; declared as [sandbox] skills = '{skills}', so a task.md run installs them"
        ));
    }
    if let Some(record) = &package.verifier_md {
        let kept = table_mut(table_mut(native, "import"), "v06");
        let path = record
            .get("path")
            .and_then(Val::as_str)
            .unwrap_or_default()
            .to_string();
        if kept.contains_key("verifier_md") {
            notes.push(format!(
                "{path} is not recorded: the frontmatter has a top-level verifier_md key"
            ));
            return;
        }
        kept.insert("verifier_md".into(), Val::Table(record.clone()));
        let (listed, others) = match record.get("strategies") {
            Some(Val::Table(s)) => {
                let listed = s
                    .iter()
                    .map(|(n, t)| format!("{n} ({})", t.as_str().unwrap_or_default()))
                    .collect::<Vec<_>>()
                    .join(", ");
                let others: BTreeSet<String> = s
                    .values()
                    .filter_map(Val::as_str)
                    .filter(|t| *t != "script")
                    .map(str::to_string)
                    .collect();
                (listed, others.into_iter().collect::<Vec<_>>())
            }
            _ => ("unreadable".to_string(), Vec::new()),
        };
        let tail = if others.is_empty() {
            String::new()
        } else {
            format!(", not the {} strategies", others.join(", "))
        };
        notes.push(format!(
            "{path} is v0.6's verifier document, which draft 1 does not map: strategies {listed}. It is copied as a file and recorded under [import.v06] verifier_md; a task.md runtime runs the verifier's test.sh{tail}"
        ));
    }
}

/// A 0.8 task.md and its sidecars to task.md text, with notes.
fn v06_to_v1(
    task_md: &str,
    sidecars: &IndexMap<String, String>,
    package: &Package,
) -> Result<(String, Vec<String>), ConvertError> {
    let (front, prompt, mut sections) = split_v06(task_md);
    for (k, v) in sidecars {
        sections.insert(k.clone(), v.clone());
    }
    let mut notes = Vec::new();
    let mut parts = vec![prompt.trim_end_matches('\n').to_string()];
    for (name, text) in &sections {
        let info = if let Some(r) = name.strip_prefix("role:") {
            format!("role {}", r.trim())
        } else if let Some(s) = name.strip_prefix("scene:") {
            format!("stage {}", s.trim())
        } else {
            "user".to_string()
        };
        parts.push(doc::block(&info, text));
    }
    let has_package =
        package.skills.is_some() || package.verifier_md.is_some() || !package.rk_judges.is_empty();
    if front.is_some() || has_package {
        let invalid = |message: String| ConvertError::Invalid {
            file: "task.md".into(),
            message,
        };
        let mut native = Table::new();
        if let Some(f) = &front {
            let data = parse_yaml(f)
                .map_err(|e| invalid(format!("the frontmatter is not valid YAML: {e}")))?;
            let data = if data == Val::Null {
                Val::Table(Table::new())
            } else {
                data
            };
            let (n, more) = config_from_v06(data).map_err(invalid)?;
            native = n;
            notes = more;
        }
        add_package(&mut native, package, &mut notes);
        declare_models(&mut native, &package.rk_judges, &mut notes);
        let native = ordered(&native, &order(&native));
        let comments = yaml_comments(front.as_deref().unwrap_or(""));
        let mut map = Comments::new();
        if !comments.is_empty() {
            map.insert(
                Vec::new(),
                Comment {
                    lead: comments,
                    inline: None,
                },
            );
        }
        let toml = tomltext::dumps(&native, &map)
            .ok_or_else(|| invalid("a value TOML cannot hold".into()))?;
        parts.push(doc::block("toml task", &toml));
    }
    let text = format!("{}\n", parts.join("\n\n"));
    if parts.len() > 1 {
        let config = front.is_some() || has_package;
        let text = doc::verified(text, &prompt, "the v0.6 prompt", config)
            .map_err(ConvertError::Instruction)?;
        return Ok((text, notes));
    }
    Ok((text, notes))
}

fn sidecars(dir: &Path) -> Result<IndexMap<String, String>, ConvertError> {
    let mut out = IndexMap::new();
    let folder = dir.join("prompts");
    if !folder.is_dir() {
        return Ok(out);
    }
    let mut names: Vec<String> = std::fs::read_dir(&folder)
        .map_err(io_err(&folder))?
        .filter_map(Result::ok)
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    for name in names {
        let Some(c) = SIDECAR.captures(&name) else {
            continue;
        };
        let key = match c.get(1) {
            Some(kind) => format!("{}:{}", kind.as_str(), &c[2]),
            None => "user-persona".to_string(),
        };
        let path = folder.join(&name);
        let text = crate::split::read_text(&path).map_err(io_err(&path))?;
        out.insert(key, text.trim_matches('\n').to_string());
    }
    Ok(out)
}

/// Notes about 0.8 oracle cases that pass only most of the time, whose
/// reruns task.md's converter records as evidence.
fn oracle_rerun_notes(front: Option<&str>, src: &Path) -> Vec<String> {
    let Some(front) = front.filter(|f| f.contains("acceptance_live")) else {
        return Vec::new();
    };
    let Ok(Val::Table(data)) = parse_yaml(front) else {
        return Vec::new();
    };
    let live = data
        .get("benchflow")
        .and_then(Val::as_table)
        .and_then(|b| b.get("evidence"))
        .and_then(Val::as_table)
        .and_then(|e| e.get("acceptance_live"))
        .and_then(Val::as_table);
    let Some(live) = live else { return Vec::new() };
    let report = live
        .get("report")
        .and_then(Val::as_str)
        .unwrap_or(LIVE_REPORT)
        .to_string();
    let present = src.join(&report).is_file();
    let mut notes = Vec::new();
    for case in live
        .get("cases")
        .and_then(Val::as_list)
        .map(Vec::as_slice)
        .unwrap_or_default()
    {
        let Some(c) = case.as_table() else { continue };
        let (Some(Val::Str(kind)), Some(Val::Str(name))) = (c.get("type"), c.get("name")) else {
            continue;
        };
        let reruns = c
            .get("reruns")
            .filter(|r| r.truthy())
            .cloned()
            .unwrap_or(Val::Int(1));
        let limit = c
            .get("expect")
            .and_then(Val::as_table)
            .and_then(|e| e.get("flake_rate_max"))
            .cloned();
        let many = reruns.py_float().is_some_and(|r| r > 1.0);
        let flaky = limit.as_ref().is_some_and(|l| *l != Val::Null);
        if kind != "oracle" || !(many || flaky) {
            continue;
        }
        let shown_limit = limit.map_or("None".to_string(), |l| harbor::py_repr(&l));
        notes.push(if present {
            format!(
                "warning: the v0.6 oracle case '{name}' passes only most of the time (reruns {}, flake_rate_max {shown_limit}); task.md does not lower the oracle's bar, and this importer does not yet record its reruns from {report} in evidence/runs.json as task.md's tools/convert.py does",
                harbor::py_repr(&reruns)
            )
        } else {
            format!(
                "the v0.6 oracle case '{name}' passes only most of the time (reruns {}, flake_rate_max {shown_limit}), but its live report {report} is not in the package, so no runs were recorded; task.md's oracle must earn full marks every time",
                harbor::py_repr(&reruns)
            )
        });
    }
    notes
}

/// Plan the import of the 0.8 task.md package at `src`.
pub(crate) fn plan(src: &Path) -> Result<ImportPlan, ConvertError> {
    let present: BTreeSet<String> = std::fs::read_dir(src)
        .map_err(io_err(src))?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    let mut placed: IndexMap<Rel, Rel> = IndexMap::new();
    let mut skipped: Vec<String> = Vec::new();
    for rel in files::list(src).map_err(io_err(src))? {
        let top = rel[0].as_str();
        if rel == ["task.md"] || (top == "prompts" && rel.len() == 2 && SIDECAR.is_match(&rel[1])) {
            continue;
        }
        if rel == ["task.toml"] || rel == ["instruction.md"] {
            skipped.push(format!(
                "{} is a split-layout copy of this task; the v0.6 task.md is the source, so it was not carried",
                rel[0]
            ));
            continue;
        }
        let new = V06_DIRS
            .iter()
            .find(|(old, _)| *old == top)
            .map(|(_, n)| *n);
        let to: Rel = match new {
            Some(n) if !present.contains(n) => std::iter::once(n.to_string())
                .chain(rel[1..].iter().cloned())
                .collect(),
            _ => rel.clone(),
        };
        if let Some(other) = placed.get(&to) {
            return Err(ConvertError::Collision(format!(
                "{} and {} would both become {}",
                files::posix(&rel),
                files::posix(other),
                files::posix(&to)
            )));
        }
        placed.insert(to, rel);
    }
    let md_path = src.join("task.md");
    let task_md = crate::split::read_text(&md_path).map_err(io_err(&md_path))?;
    let package = package_of(src, &placed);
    let (text, mut notes) = v06_to_v1(&task_md, &sidecars(src)?, &package)?;
    notes.extend(skipped);
    let front = task_md
        .strip_prefix("---\n")
        .and_then(|r| r.find("\n---\n").map(|i| r[..i].to_string()));
    notes.extend(oracle_rerun_notes(front.as_deref(), src));
    Ok(ImportPlan {
        task_md: text,
        copies: placed.into_iter().collect(),
        writes: Vec::new(),
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_v06_sections() {
        let (front, prompt, sections) =
            split_v06("---\na: 1\n---\n\n## prompt\n\nDo it.\n\n## role:solver  \nSolve.\n");
        assert_eq!(front.as_deref(), Some("a: 1"));
        assert_eq!(prompt, "Do it.");
        assert_eq!(
            sections.get("role:solver  ").map(String::as_str),
            Some("Solve.")
        );
        let (_, prompt, _) = split_v06("Just a body.\n");
        assert_eq!(prompt, "Just a body.");
    }

    #[test]
    fn converts_frontmatter() {
        let front = parse_yaml(
            "schema_version: '1.3'\nagent:\n  timeout_sec: 300\nsandbox:\n  cpus: 2\n  memory_mb: 4096\nverifier:\n  timeout_sec: 60\n  sandbox_mode: separate\nagents:\n  roles:\n    solver:\n      agent: claude-agent-acp\nbenchflow:\n  document_version: '0.3'\nmetadata:\n  tags: [a]\n  note: ~\n",
        )
        .unwrap();
        let (cfg, notes) = config_from_v06(front).unwrap();
        let keys: Vec<&String> = cfg.keys().collect();
        assert_eq!(
            keys,
            vec![
                "about",
                "agent",
                "sandbox",
                "roles",
                "verifier",
                "x-benchflow",
                "import"
            ]
        );
        assert_eq!(
            cfg["sandbox"].as_table().unwrap()["memory"],
            Val::Str("4 GB".into())
        );
        assert_eq!(
            cfg["verifier"].as_table().unwrap()["isolation"],
            Val::Str("separate".into())
        );
        let kept = cfg["import"].as_table().unwrap()["v06"].as_table().unwrap();
        assert_eq!(kept["schema_version"], Val::Str("1.3".into()));
        assert_eq!(
            kept["roles"].as_table().unwrap()["solver"]
                .as_table()
                .unwrap()["agent"],
            Val::Str("claude-agent-acp".into())
        );
        assert!(notes.iter().any(|n| n.starts_with("dropped metadata.note")));
        assert!(
            notes
                .iter()
                .any(|n| n.starts_with("kept under [import.v06]: roles.solver"))
        );
    }

    #[test]
    fn comments_in_yaml() {
        assert_eq!(
            yaml_comments("# top\na: 1 # why\nb: \"x # not a comment\"\n"),
            vec!["# top".to_string(), "# why".to_string()]
        );
    }
}
