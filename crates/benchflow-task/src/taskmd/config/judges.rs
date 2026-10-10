//! `[verifier.judges]` and its role tables, the task's and each stage's
//! (the config half of the reference's `check_judges`).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::source::join;

use super::super::code;
use super::super::pyfmt::{json_dumps, repr_str};
use super::super::pyval::{count_value, duration_s, int, is_count, nonblank, table};
use super::super::report::{At, Found};
use super::super::vocab::{
    AGENT_JUDGE_KEYS, AGGREGATES, COUNT_FORM, JUDGE_KEYS, JUDGE_MOVED, JUDGE_RESOURCES,
    JUDGE_ROLES, VIEW_KEYS, quoted_list, sorted_list,
};
use super::keys::{choice_fix, dym_or, removal_fix, rename_fix, suggestion};
use super::{Checker, values};

static HARNESS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[^@\s]+@[^@\s]+$").unwrap());

/// Each `[verifier.judges]` table: the task's, and each stage's that has one.
fn scopes(config: &Value) -> Vec<(String, String, &Map<String, Value>, String)> {
    let mut out = vec![(
        "[verifier.judges]".to_string(),
        "verifier.judges".to_string(),
        table(config, &["verifier", "judges"]),
        "verifier".to_string(),
    )];
    for (name, st) in table(config, &["stages"]) {
        let judges = table(st, &["verifier", "judges"]);
        if !judges.is_empty() {
            out.push((
                format!("[stages.{name}.verifier.judges]"),
                join(&join(&join("stages", name), "verifier"), "judges"),
                judges,
                format!("stages/{name}/verifier"),
            ));
        }
    }
    out
}

/// The role tables of every `[verifier.judges]`.
pub(super) fn check(c: &mut Checker<'_>) {
    for (where_, path, judges, vdir) in scopes(c.config) {
        for k in ["samples", "min_samples"] {
            if let Some(v) = judges.get(k)
                && !is_count(Some(v))
            {
                c.err(
                    code::BAD_NUMBER,
                    At::Value(&join(&path, k)),
                    format!("{where_} {k} is a whole number, at least 1"),
                );
            }
        }
        min_samples(c, judges, &where_, &path);
        if let Some(v) = judges.get("lazy")
            && !v.is_boolean()
        {
            c.err(
                code::BAD_CHOICE,
                At::Value(&join(&path, "lazy")),
                format!("{where_} lazy is true (the default) or false"),
            );
        }
        let open = where_.strip_suffix(']').unwrap_or(&where_).to_string();
        for role in JUDGE_ROLES {
            let Some(value) = judges.get(*role).filter(|v| !v.is_null()) else {
                continue;
            };
            if nonblank(Some(value)).is_some() {
                continue;
            }
            if *role == "panel"
                && matches!(value, Value::Array(m) if !m.is_empty() && m.iter().all(Value::is_string))
            {
                continue;
            }
            let role_path = join(&path, role);
            let Value::Object(t) = value else {
                c.err(
                    code::WRONG_SHAPE,
                    At::Value(&role_path),
                    format!("{where_} {role} is a model name, or a table {open}.{role}]"),
                );
                continue;
            };
            role_rules(c, t, role, &format!("{open}.{role}]"), &role_path, &vdir);
        }
    }
}

fn min_samples(c: &mut Checker<'_>, t: &Map<String, Value>, where_: &str, path: &str) {
    if let (Some(s), Some(m)) = (
        t.get("samples").and_then(int),
        t.get("min_samples").and_then(int),
    ) && s >= 1
        && m >= 1
        && m > s
    {
        c.err(
            code::JUDGES,
            At::Value(&join(path, "min_samples")),
            format!("{where_} min_samples is at most samples"),
        );
    }
}

/// One `[verifier.judges.<role>]` table: closed keys, with the agent role's
/// own keys refused on the others (the reference's `_check_judge_role`).
fn role_rules(
    c: &mut Checker<'_>,
    t: &Map<String, Value>,
    role: &str,
    where_: &str,
    path: &str,
    vdir: &str,
) {
    let agent = role == "agent";
    let allowed: Vec<&str> = JUDGE_KEYS
        .iter()
        .chain(if agent { AGENT_JUDGE_KEYS } else { &[] })
        .copied()
        .collect();
    for k in t.keys() {
        if k.starts_with("x-") || allowed.contains(&k.as_str()) {
            continue;
        }
        let kp = join(path, k);
        if let Some((_, gone)) = JUDGE_MOVED.iter().find(|(m, _)| m == k) {
            let message = format!("{where_} {gone}");
            c.err(code::KEY_GONE, At::Key(&kp), message.clone());
            if gone.ends_with("Remove the key") {
                removal_fix(c, &kp, &format!("remove `{k}`"), &message);
            }
        } else if AGENT_JUDGE_KEYS.contains(&k.as_str()) {
            let article = if role == "llm" { "an" } else { "a" };
            c.err(
                code::JUDGES,
                At::Key(&kp),
                format!("{where_} {k} is for the agent role only: {article} {role} judge is one model call with the evidence inline"),
            );
        } else {
            let mut sorted_allowed = allowed.clone();
            sorted_allowed.sort_unstable();
            let message = format!(
                "unknown key {} in {where_}{} it holds {}, or x- extensions",
                repr_str(k),
                dym_or(k, &allowed, ";"),
                sorted_allowed.join(", ")
            );
            c.err(code::UNKNOWN_KEY, At::Key(&kp), message.clone());
            rename_fix(c, k, &kp, path, &allowed, &message);
        }
    }
    if let Some(model) = t.get("model").filter(|v| !v.is_null()) {
        let ok = nonblank(Some(model)).is_some()
            || (role == "panel"
                && matches!(model, Value::Array(m) if !m.is_empty() && m.iter().all(Value::is_string)));
        if !ok {
            let panel = if role == "panel" {
                ", or a list of them for a panel"
            } else {
                ""
            };
            c.err(
                code::JUDGES,
                At::Value(&join(path, "model")),
                format!("{where_} model names a model{panel}"),
            );
        }
    }
    if t.contains_key("effort") && nonblank(t.get("effort")).is_none() {
        c.err(
            code::JUDGES,
            At::Value(&join(path, "effort")),
            format!("{where_} effort is the model's reasoning effort, such as \"high\""),
        );
    }
    for k in ["samples", "min_samples"] {
        if let Some(v) = t.get(k)
            && !is_count(Some(v))
        {
            c.err(
                code::BAD_NUMBER,
                At::Value(&join(path, k)),
                format!("{where_} {k} is a whole number, at least 1"),
            );
        }
    }
    min_samples(c, t, where_, path);
    if let Some(a) = t.get("aggregate")
        && !matches!(a, Value::String(s) if AGGREGATES.contains(&s.as_str()))
    {
        let p = join(path, "aggregate");
        let message = format!(
            "{where_} aggregate must be one of {}",
            quoted_list(AGGREGATES)
        );
        c.err(code::BAD_CHOICE, At::Value(&p), message.clone());
        if let Value::String(s) = a
            && let Some(close) = suggestion(s, AGGREGATES)
        {
            choice_fix(c, &p, &close, &message);
        }
    }
    if let Some(per) = t.get("per")
        && !matches!(per.as_str(), Some("rubric" | "criterion"))
    {
        let p = join(path, "per");
        let message = format!(
            "{where_} per is \"rubric\" (one session grades all of the role's criteria) or \"criterion\" (one session each)"
        );
        c.err(code::BAD_CHOICE, At::Value(&p), message.clone());
        if let Value::String(s) = per
            && let Some(close) = suggestion(s, &["rubric", "criterion"])
        {
            choice_fix(c, &p, &close, &message);
        }
    }
    if let Some(v) = t.get("timeout")
        && duration_s(Some(v)).is_none()
    {
        c.err(
            code::BAD_DURATION,
            At::Value(&join(path, "timeout")),
            format!("{where_} timeout must be a duration such as \"90s\", \"15m\", or \"1h30m\""),
        );
    }
    if let Some(brief) = t.get("brief") {
        check_brief(c, brief, vdir, where_, &join(path, "brief"));
    }
    if let Some(budget) = t.get("budget").filter(|v| !v.is_null()) {
        budget_rules(c, budget, role, where_, &join(path, "budget"));
    }
    if !agent {
        return;
    }
    if let Some(h) = t.get("harness").filter(|v| !v.is_null())
        && !h.as_str().is_some_and(|s| HARNESS.is_match(s))
    {
        c.err(
            code::JUDGES,
            At::Value(&join(path, "harness")),
            format!(
                "{where_} harness names a harness and its version, such as \"judge-loop@1\": conformance is per harness version"
            ),
        );
    }
    if let Some(res) = t.get("resources").filter(|v| !v.is_null()) {
        let rp = join(path, "resources");
        match res {
            Value::Object(r) => {
                let mut extra: Vec<&String> = r
                    .keys()
                    .filter(|k| !JUDGE_RESOURCES.contains(&k.as_str()))
                    .collect();
                extra.sort();
                for k in extra {
                    let kp = join(&rp, k);
                    let message = format!(
                        "unknown key {} in {where_} resources{} it holds {}",
                        repr_str(k),
                        dym_or(k, JUDGE_RESOURCES, ";"),
                        sorted_list(JUDGE_RESOURCES)
                    );
                    c.err(code::UNKNOWN_KEY, At::Key(&kp), message.clone());
                    rename_fix(c, k, &kp, &rp, JUDGE_RESOURCES, &message);
                }
                let sizes: Map<String, Value> = r
                    .iter()
                    .filter(|(k, _)| *k == "memory" || *k == "disk")
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                values::phase(c, &sizes, &format!("{where_} resources"), &rp);
            }
            _ => {
                c.err(
                    code::WRONG_SHAPE,
                    At::Value(&rp),
                    format!(
                        "{where_} resources is a table of {}",
                        sorted_list(JUDGE_RESOURCES)
                    ),
                );
            }
        }
    }
    if let Some(s) = t.get("services")
        && !s.is_boolean()
    {
        c.err(
            code::BAD_CHOICE,
            At::Value(&join(path, "services")),
            format!(
                "{where_} services is true (the task's services run for the judge too) or false"
            ),
        );
    }
    if let Some(views) = t.get("views").filter(|v| !v.is_null()) {
        views_rules(c, views, where_, &join(path, "views"));
    }
}

fn budget_rules(c: &mut Checker<'_>, budget: &Value, role: &str, where_: &str, path: &str) {
    const KEYS: &[&str] = &["tokens", "tool_calls"];
    let extra = budget
        .as_object()
        .and_then(|b| b.keys().find(|k| !KEYS.contains(&k.as_str())).cloned());
    let Some(b) = budget.as_object().filter(|_| extra.is_none()) else {
        let message = format!("{where_} budget is a table of tokens and tool_calls");
        match extra {
            Some(k) => {
                let kp = join(path, &k);
                c.err(code::WRONG_SHAPE, At::Key(&kp), message.clone());
                rename_fix(c, &k, &kp, path, KEYS, &message);
            }
            None => {
                c.err(code::WRONG_SHAPE, At::Value(path), message);
            }
        }
        return;
    };
    if let Some(tokens) = b.get("tokens").filter(|v| !v.is_null())
        && count_value(Some(tokens)).is_none()
    {
        c.err(
            code::BAD_NUMBER,
            At::Value(&join(path, "tokens")),
            format!("{where_} budget tokens is a count: {COUNT_FORM}"),
        );
    }
    if let Some(calls) = b.get("tool_calls") {
        let p = join(path, "tool_calls");
        if role != "agent" {
            c.err(
                code::JUDGES,
                At::Key(&p),
                format!("{where_} budget tool_calls is for the agent role only: an {role} judge makes one call and uses no tools"),
            );
        } else if !is_count(Some(calls)) {
            c.err(
                code::BAD_NUMBER,
                At::Value(&p),
                format!("{where_} budget tool_calls is a whole number, at least 1"),
            );
        }
    }
}

fn views_rules(c: &mut Checker<'_>, views: &Value, where_: &str, path: &str) {
    let Value::Array(items) = views else {
        c.err(
            code::WRONG_SHAPE,
            At::Value(path),
            format!(
                "{where_} views is a list of {{ path, format = \"trajectory-1\" }} tables, the files the runtime writes for the judge from \
its own records"
            ),
        );
        return;
    };
    for (n, view) in items.iter().enumerate() {
        let at = format!("{where_} views[{n}]");
        let vp = format!("{path}[{n}]");
        let Value::Object(v) = view else {
            if let Value::String(s) = view {
                c.err(
                    code::KEY_GONE,
                    At::Value(&vp),
                    format!(
                        "{at} is a bare path, {}, which meant format \"atif\": ATIF views are cut, since no page defines their bytes. Write \
{{ path = {}, format = \"trajectory-1\" }}",
                        repr_str(s),
                        json_dumps(view)
                    ),
                );
            } else {
                c.err(
                    code::WRONG_SHAPE,
                    At::Value(&vp),
                    format!("{at} is a table {{ path, format = \"trajectory-1\" }}"),
                );
            }
            continue;
        };
        let mut extra: Vec<&String> = v
            .keys()
            .filter(|k| !VIEW_KEYS.contains(&k.as_str()))
            .collect();
        extra.sort();
        for k in extra {
            if k.starts_with("x-") {
                continue;
            }
            let kp = join(&vp, k);
            let message = format!(
                "unknown key {} in {at}{} a view holds path and format",
                repr_str(k),
                dym_or(k, VIEW_KEYS, ";")
            );
            c.err(code::UNKNOWN_KEY, At::Key(&kp), message.clone());
            rename_fix(c, k, &kp, &vp, VIEW_KEYS, &message);
        }
        if !v
            .get("path")
            .and_then(Value::as_str)
            .is_some_and(|p| p.starts_with('/'))
        {
            c.err(
                code::MISSING_KEY,
                At::Value(&join(&vp, "path")),
                format!("{at} needs a path: the absolute path the judge reads it at, such as \"/logs/agent/trajectory.json\""),
            );
        }
        let format = v.get("format").and_then(Value::as_str);
        if format == Some("atif") {
            let p = join(&vp, "format");
            let message = format!(
                "{at} format \"atif\" is cut, since no page defines an ATIF view's bytes: write format = \"trajectory-1\", the solver's \
trajectory in the runtime's own record"
            );
            c.err(code::KEY_GONE, At::Value(&p), message.clone());
            super::keys::value_fix(
                c,
                &p,
                "\"trajectory-1\"",
                "write \"trajectory-1\"",
                &message,
            );
        } else if format != Some("trajectory-1") {
            c.err(
                code::JUDGES,
                At::Value(&join(&vp, "format")),
                format!("{at} format is \"trajectory-1\", the solver's trajectory in the runtime's own record"),
            );
        }
    }
}

/// A role's brief: a markdown file under the verifier folder, which only
/// grading sees (the reference's `_check_brief`).
fn check_brief(c: &mut Checker<'_>, value: &Value, vdir: &str, where_: &str, path: &str) {
    let Some(brief) = nonblank(Some(value)) else {
        c.err(
            code::JUDGE_BRIEF,
            At::Value(path),
            format!("{where_} brief names a markdown file under {vdir}/"),
        );
        return;
    };
    if brief.contains("://") {
        c.err(
            code::JUDGE_BRIEF,
            At::Value(path),
            format!(
                "{where_} brief {brief} is a URL, and briefs are files in the package: copy the shared brief into {vdir}/, and apply a \
change across a dataset's tasks with a tool"
            ),
        );
        return;
    }
    let rel = super::super::judgment::normpath(brief);
    if !rel.starts_with(&format!("{vdir}/")) {
        c.err(
            code::JUDGE_BRIEF,
            At::Value(path),
            format!(
                "{where_} brief {brief} must be a file under {vdir}/, which only grading sees; anywhere else the agent could read it"
            ),
        );
        return;
    }
    let Some(dir) = c.dir else {
        return;
    };
    if !dir.join(&rel).is_file() {
        c.err(
            code::JUDGE_BRIEF,
            At::Value(path),
            format!("{where_} brief {brief} is not in the package"),
        );
    } else if c.deep {
        for diag in super::super::judges::brief_file(dir, &rel) {
            c.sink.found.push(Found { path: None, diag });
        }
    }
}
