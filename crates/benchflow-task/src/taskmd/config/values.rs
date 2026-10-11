//! Config values: durations, sizes, networks, documented choices, readiness
//! checks, services, budgets, credits, snapshots, stages, outputs, training,
//! controls, and roles.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::source::join;

use super::super::code;
use super::super::pyfmt::repr_str;
use super::super::pyval::{
    count_value, duration_s, int, is_count, is_size, network_ok, nonblank, num, table,
};
use super::super::report::At;
use super::super::scan::BlockKind;
use super::super::vocab::{
    COUNT_FORM, OUTPUT_KEYS, SERVICE_KEYS, WORLD_KINDS, quoted_list, sorted_repr,
};
use super::Checker;
use super::keys::{
    choice_fix, dym_or, harbor_duration, rename_fix, seconds_like, suggestion, value_fix,
};

static UNLOCK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:on_submit|on_request|at_start|after:[\w.-]+|at_turn:\d+)\n?\z").unwrap()
});
static LOOSE_SIZE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)^\s*(\d+(?:\.\d+)?)\s*(m|g|t)b?\s*$").unwrap());

/// Whether a stage's unlock is one task.md defines.
pub(crate) fn unlock_ok(s: &str) -> bool {
    UNLOCK.is_match(s)
}

/// Durations, sizes, and the network of a phase table (`[agent]`,
/// `[sandbox]`, `[verifier]`, a stage's tables): the reference's
/// `_check_phase`.
pub(crate) fn phase(c: &mut Checker<'_>, table: &Map<String, Value>, where_: &str, path: &str) {
    for k in ["timeout", "build_timeout"] {
        let Some(v) = table.get(k) else {
            continue;
        };
        if duration_s(Some(v)).is_none() {
            let p = join(path, k);
            let message =
                format!("{where_} {k} must be a duration such as \"90s\", \"15m\", or \"1h30m\"");
            c.err(code::BAD_DURATION, At::Value(&p), message.clone());
            if let Some(secs) = seconds_like(v)
                && let Some(d) = harbor_duration(&Value::from(secs))
            {
                let shown = super::super::pyfmt::str_of(v);
                if let Some(diag) = c.sink.found.last_mut() {
                    diag.diag.label = Some(format!("if this is {shown} seconds, write \"{d}\""));
                }
                value_fix(
                    c,
                    &p,
                    &format!("\"{d}\""),
                    &format!("write {shown} seconds as \"{d}\""),
                    &message,
                );
            }
        }
    }
    for k in ["memory", "disk"] {
        let Some(v) = table.get(k) else {
            continue;
        };
        if !is_size(v) {
            let p = join(path, k);
            let message = format!("{where_} {k} must be a size such as \"512 MB\" or \"4 GB\"");
            c.err(code::BAD_SIZE, At::Value(&p), message.clone());
            if let Some(Value::String(s)) = Some(v)
                && let Some(caps) = LOOSE_SIZE.captures(s)
            {
                let unit = match caps[2].to_ascii_uppercase().as_str() {
                    "M" => "MB",
                    "G" => "GB",
                    _ => "TB",
                };
                let fixed = format!("{} {unit}", &caps[1]);
                value_fix(
                    c,
                    &p,
                    &format!("\"{fixed}\""),
                    &format!("write \"{fixed}\""),
                    &message,
                );
            }
        }
    }
    if let Some(v) = table.get("network")
        && !network_ok(v)
    {
        let p = join(path, "network");
        let message = format!(
            "{where_} network must be \"none\", \"open\", a list of hosts, or {{ block = [hosts] }}"
        );
        c.err(code::BAD_NETWORK, At::Value(&p), message.clone());
        let fixed = match v.as_str() {
            Some("no-network") => Some("none"),
            Some("public") => Some("open"),
            _ => None,
        };
        if let Some(fixed) = fixed {
            if let Some(d) = c.sink.found.last_mut() {
                d.diag.label = Some(format!(
                    "the split layout's \"{}\" is \"{fixed}\" in task.md",
                    v.as_str().unwrap_or("")
                ));
            }
            value_fix(
                c,
                &p,
                &format!("\"{fixed}\""),
                &format!("write \"{fixed}\""),
                &message,
            );
        }
    }
}

/// The documented values of single keys in the closed tables.
pub(super) fn enums(c: &mut Checker<'_>) {
    const CHOICES: &[(&str, &str, &[&str])] = &[
        ("verifier", "isolation", &["shared", "separate"]),
        ("verifier", "combine_stages", &["mean", "final"]),
        ("verifier", "unreached_stages", &["zero", "exclude"]),
        ("agent", "on_timeout", &["grade", "grade-flagged", "fail"]),
        (
            "sandbox",
            "boundary",
            &["container", "gvisor", "microvm", "vm"],
        ),
        ("runs", "errored", &["zero", "retry"]),
        ("runs", "trajectory", &["required", "optional"]),
        ("agent", "timeout_basis", &["wall", "environment"]),
    ];
    for (t, key, allowed) in CHOICES {
        let Some(v) = table(c.config, &[t]).get(*key) else {
            continue;
        };
        if v.is_null() || matches!(v, Value::String(s) if allowed.contains(&s.as_str())) {
            continue;
        }
        let path = join(&join("", t), key);
        let message = format!("[{t}] {key} must be one of {}", quoted_list(allowed));
        c.err(code::BAD_CHOICE, At::Value(&path), message.clone());
        if let Value::String(s) = v
            && let Some(close) = suggestion(s, allowed)
        {
            choice_fix(c, &path, &close, &message);
        }
    }
}

/// `[verifier] feedback` and `models`, for the task's verifier and each
/// stage's.
pub(super) fn verifier_keys(
    c: &mut Checker<'_>,
    table: &Map<String, Value>,
    where_: &str,
    path: &str,
) {
    match table.get("feedback") {
        Some(Value::String(f)) if f == "rationales" => {
            c.err(
                code::KEY_GONE,
                At::Value(&join(path, "feedback")),
                format!(
                    "{where_} feedback = \"rationales\" is cut: a trainable seat may be shown its verdicts, levels, and scores, never a \
judge's rationales. Write feedback = \"verdicts\" or \"none\""
                ),
            );
        }
        Some(v)
            if !v.is_null() && !matches!(v, Value::String(s) if s == "none" || s == "verdicts") =>
        {
            let p = join(path, "feedback");
            let message = format!("{where_} feedback must be one of 'none', 'verdicts'");
            c.err(code::BAD_CHOICE, At::Value(&p), message.clone());
            if let Value::String(s) = v
                && let Some(close) = suggestion(s, &["none", "verdicts"])
            {
                choice_fix(c, &p, &close, &message);
            }
        }
        _ => {}
    }
    if let Some(v) = table.get("models")
        && !v.is_boolean()
    {
        c.err(
            code::BAD_CHOICE,
            At::Value(&join(path, "models")),
            format!(
                "{where_} models is true when the verifier's scripts call a model themselves, such as a RewardKit judge in test.sh, \
or false (the default)"
            ),
        );
    }
}

/// A readiness check: a table with `run` and Harbor's healthcheck timing
/// keys (the reference's `_check_ready`).
pub(crate) fn ready(c: &mut Checker<'_>, ready: Option<&Value>, where_: &str, path: &str) {
    let Some(ready) = ready else {
        return;
    };
    let Value::Object(r) = ready else {
        c.err(
            code::WRONG_SHAPE,
            At::Value(path),
            format!(
                "{where_} is a table: {{ run = \"<command>\" }}, with interval, timeout, start_period, start_interval, and retries"
            ),
        );
        return;
    };
    if nonblank(r.get("run")).is_none() {
        c.err(
            code::MISSING_KEY,
            At::Key(&join(path, "run")),
            format!("{where_} needs run, the command that must pass"),
        );
    }
    for k in ["interval", "timeout", "start_period", "start_interval"] {
        if let Some(v) = r.get(k)
            && duration_s(Some(v)).is_none()
        {
            let p = join(path, k);
            let message = format!("{where_} {k} must be a duration such as \"5s\" or \"1m\"");
            c.err(code::BAD_DURATION, At::Value(&p), message.clone());
            if let Some(secs) = seconds_like(v)
                && let Some(d) = harbor_duration(&Value::from(secs))
            {
                value_fix(
                    c,
                    &p,
                    &format!("\"{d}\""),
                    &format!("write \"{d}\""),
                    &message,
                );
            }
        }
    }
    if let Some(v) = r.get("retries")
        && !int(v).is_some_and(|n| n >= 0)
    {
        c.err(
            code::BAD_NUMBER,
            At::Value(&join(path, "retries")),
            format!("{where_} retries is a whole number, at least 0"),
        );
    }
}

/// `[[sandbox.services]]`: each has a name (not `main`, and not twice) and
/// an image or a build folder, and only the service keys.
pub(super) fn services(c: &mut Checker<'_>) {
    let Some(services) = table(c.config, &["sandbox"]).get("services") else {
        return;
    };
    if services.is_null() {
        return;
    }
    let entries: Vec<(String, Option<&Value>)> = match services {
        Value::Array(items) => items
            .iter()
            .enumerate()
            .map(|(i, s)| (format!("sandbox.services[{i}]"), Some(s)))
            .collect(),
        _ => vec![("sandbox.services".to_string(), None)],
    };
    let mut names: Vec<(String, String)> = Vec::new();
    for (path, svc) in entries {
        let Some(Value::Object(s)) = svc else {
            c.err(
                code::MISSING_KEY,
                At::Value(&path),
                "each [[sandbox.services]] entry has a name and an image or a build folder",
            );
            continue;
        };
        let Some(Value::String(name)) = s.get("name") else {
            c.err(
                code::MISSING_KEY,
                At::Key(&path),
                "each [[sandbox.services]] entry has a name and an image or a build folder",
            );
            continue;
        };
        if !s.contains_key("image") && !s.contains_key("build") {
            c.err(
                code::MISSING_KEY,
                At::Key(&path),
                "each [[sandbox.services]] entry has a name and an image or a build folder",
            );
            continue;
        }
        if name == "main" {
            c.err(
                code::SERVICES,
                At::Value(&join(&path, "name")),
                "the service name main is the agent's own container; name the service something else",
            );
        }
        names.push((name.clone(), join(&path, "name")));
        for k in s.keys() {
            if !SERVICE_KEYS.contains(&k.as_str()) && !k.starts_with("x-") {
                let kp = join(&path, k);
                let message = format!("unknown key {k} in service {name}");
                c.err(code::UNKNOWN_KEY, At::Key(&kp), message.clone());
                rename_fix(c, k, &kp, &path, SERVICE_KEYS, &message);
            }
        }
        ready(
            c,
            s.get("ready"),
            &format!("service {name} ready"),
            &join(&path, "ready"),
        );
    }
    let mut dups: Vec<&String> = names
        .iter()
        .map(|(n, _)| n)
        .filter(|n| names.iter().filter(|(m, _)| m == *n).count() > 1)
        .collect();
    dups.sort();
    dups.dedup();
    for dup in dups {
        let second = names
            .iter()
            .filter(|(n, _)| n == dup)
            .nth(1)
            .map(|(_, p)| p.clone())
            .unwrap_or_default();
        c.err(
            code::SERVICES,
            At::Value(&second),
            format!("service {dup} is declared twice"),
        );
    }
}

/// `[agent] budget`: a table of `tool_calls` and `tokens`.
pub(super) fn budget(c: &mut Checker<'_>) {
    let Some(budget) = table(c.config, &["agent"]).get("budget") else {
        return;
    };
    if budget.is_null() {
        return;
    }
    const BUDGET_KEYS: &[&str] = &["tool_calls", "tokens"];
    let extra = budget.as_object().and_then(|b| {
        b.keys()
            .find(|k| !BUDGET_KEYS.contains(&k.as_str()))
            .cloned()
    });
    if !budget.is_object() || extra.is_some() {
        let message = "[agent] budget is a table of tool_calls and tokens";
        match extra {
            Some(k) => {
                let kp = join("agent.budget", &k);
                c.err(code::WRONG_SHAPE, At::Key(&kp), message);
                rename_fix(c, &k, &kp, "agent.budget", BUDGET_KEYS, message);
            }
            None => {
                c.err(code::WRONG_SHAPE, At::Value("agent.budget"), message);
            }
        }
        return;
    }
    let b = budget.as_object().expect("checked above");
    if let Some(t) = b.get("tokens")
        && count_value(Some(t)).is_none()
    {
        c.err(
            code::BAD_NUMBER,
            At::Value("agent.budget.tokens"),
            format!("[agent] budget tokens is a count: {COUNT_FORM}"),
        );
    }
    if let Some(t) = b.get("tool_calls")
        && !is_count(Some(t))
    {
        c.err(
            code::BAD_NUMBER,
            At::Value("agent.budget.tool_calls"),
            "[agent] budget tool_calls is a whole number, at least 1",
        );
    }
}

/// `[[credits]]` entries and `authors`.
pub(super) fn credits_and_authors(c: &mut Checker<'_>) {
    const ROLES: &[&str] = &["author", "advisor", "domain-reviewer", "technical-reviewer"];
    if let Some(Value::Array(credits)) = c.config.get("credits") {
        for (i, entry) in credits.iter().enumerate() {
            let Value::Object(e) = entry else {
                continue;
            };
            let path = format!("credits[{i}]");
            let name_ok = e.get("name").is_some_and(Value::is_string);
            let role_ok = match e.get("role") {
                None | Some(Value::Null) => true,
                Some(Value::String(r)) => ROLES.contains(&r.as_str()),
                _ => false,
            };
            if !name_ok || !role_ok {
                let at = if name_ok {
                    join(&path, "role")
                } else {
                    path.clone()
                };
                let message = "each [[credits]] entry has a name and a role: author, advisor, domain-reviewer, or technical-reviewer";
                if name_ok {
                    c.err(code::BAD_CHOICE, At::Value(&at), message);
                    if let Some(Value::String(r)) = e.get("role")
                        && let Some(close) = suggestion(r, ROLES)
                    {
                        choice_fix(c, &at, &close, message);
                    }
                } else {
                    c.err(code::MISSING_KEY, At::Key(&at), message);
                }
            }
        }
    }
    if let Some(Value::Array(authors)) = c.config.get("authors") {
        for (i, a) in authors.iter().enumerate() {
            let ok = a.is_string() || a.get("name").is_some_and(Value::is_string);
            if !ok {
                c.err(
                    code::WRONG_SHAPE,
                    At::Value(&format!("authors[{i}]")),
                    "authors entries are \"Name <email>\" strings or { name, email } tables",
                );
            }
        }
    }
}

/// `[[verifier.snapshot]]` entries: `run`, and `reads`, the absolute paths
/// whose state the command reads.
pub(super) fn snapshots(
    c: &mut Checker<'_>,
    verifier: &Map<String, Value>,
    where_: &str,
    path: &str,
) {
    let Some(snaps) = verifier.get("snapshot") else {
        return;
    };
    if snaps.is_null() {
        return;
    }
    let sp = join(path, "snapshot");
    let Some(items) = snaps.as_array().filter(|a| a.iter().all(Value::is_object)) else {
        c.err(
            code::WRONG_SHAPE,
            At::Value(&sp),
            format!(
                "{where_} snapshot is a list of tables: [[verifier.snapshot]] with run and reads"
            ),
        );
        return;
    };
    for (n, s) in items.iter().enumerate() {
        let at = format!("{where_} snapshot[{n}]");
        let p = format!("{sp}[{n}]");
        if nonblank(s.get("run")).is_none() {
            c.err(
                code::MISSING_KEY,
                At::Key(&join(&p, "run")),
                format!("{at} needs run, the command"),
            );
        }
        match s.get("reads") {
            None | Some(Value::Null) => {
                c.err(
                    code::MISSING_KEY,
                    At::Key(&join(&p, "reads")),
                    format!(
                        "{at} needs reads, the absolute paths whose state the command reads: the runtime puts each on a volume when \
the sandbox starts, and the snapshot runs in a fresh container that sees only those paths. The split layout's collect has no such key, so \
an imported entry needs them added"
                    ),
                );
            }
            Some(reads) => {
                let ok = matches!(reads, Value::Array(r) if !r.is_empty() && r.iter().all(|x| {
                    x.as_str().is_some_and(|s| s.starts_with('/') && !s.split('/').any(|seg| seg == ".."))
                }));
                if !ok {
                    c.err(
                        code::BAD_PATH,
                        At::Value(&join(&p, "reads")),
                        format!("{at} reads is a list of absolute paths, without .."),
                    );
                }
            }
        }
        if let Some(t) = s.get("timeout")
            && duration_s(Some(t)).is_none()
        {
            c.err(
                code::BAD_DURATION,
                At::Value(&join(&p, "timeout")),
                format!("{at} timeout is a duration, such as \"2m\""),
            );
        }
    }
}

/// `[world] kind`.
pub(super) fn world_kind(c: &mut Checker<'_>) {
    let Some(Value::Object(world)) = c.config.get("world") else {
        return;
    };
    if matches!(world.get("kind"), Some(Value::String(k)) if WORLD_KINDS.contains(&k.as_str())) {
        return;
    }
    let message = format!("[world] kind must be one of {}", sorted_repr(WORLD_KINDS));
    c.err(code::BAD_CHOICE, At::Value("world.kind"), message.clone());
    if let Some(Value::String(k)) = world.get("kind")
        && let Some(close) = suggestion(k, WORLD_KINDS)
    {
        choice_fix(c, "world.kind", &close, &message);
    }
}

/// Stages: one may unlock at start, each needs an unlock and a block, and
/// each stage's tables, readiness check, and outputs.
pub(super) fn stages(c: &mut Checker<'_>) {
    let stages = table(c.config, &["stages"]);
    let staged = c.stage_blocks();
    let first: Vec<&String> = stages
        .iter()
        .filter(|(_, st)| {
            st.get("unlock").and_then(Value::as_str) == Some("at_start") && st.is_object()
        })
        .map(|(n, _)| n)
        .collect();
    if first.len() > 1 {
        let p = join(&join("stages", first[1]), "unlock");
        c.err(
            code::STAGES,
            At::Value(&p),
            "only one stage can unlock at_start: its prompt is the instruction",
        );
    }
    for n in &first {
        if staged.contains(&n.as_str()) {
            let line = c.block_line(BlockKind::Stage, n).unwrap_or(1);
            c.err(
                code::STAGES,
                At::Line(line),
                format!("stage {n} unlocks at_start, so its prompt is the instruction; remove its ```stage {n} block"),
            );
        }
    }
    for (n, st) in stages {
        let Value::Object(st) = st else {
            continue;
        };
        let sp = join("stages", n);
        for part in ["agent", "verifier"] {
            if let Some(Value::Object(t)) = st.get(part) {
                phase(c, t, &format!("[stages.{n}.{part}]"), &join(&sp, part));
            }
        }
        ready(
            c,
            st.get("ready"),
            &format!("[stages.{n}] ready"),
            &join(&sp, "ready"),
        );
        if !st.contains_key("unlock") {
            c.err(
                code::MISSING_KEY,
                At::Key(&sp),
                format!(
                    "[stages.{n}] needs unlock, when its prompt is revealed: on_submit, on_request, at_start, after:<stage>, or at_turn:<n>"
                ),
            );
        }
        if !first.contains(&n) && !staged.contains(&n.as_str()) {
            c.err(
                code::STAGES,
                At::Key(&sp),
                format!("[stages.{n}] has no ```stage {n} block"),
            );
        }
        outputs(c, st.get("outputs"), &format!("[stages.{n}]"), &sp);
    }
}

fn is_output_size(v: &Value) -> bool {
    int(v).is_some_and(|n| n > 0) || is_size(v)
}

/// `[sandbox] outputs` and a stage's outputs: absolute paths, or closed
/// `{ path, save_as, max_bytes }` tables (the reference's `check_outputs`).
pub(crate) fn outputs(
    c: &mut Checker<'_>,
    outputs: Option<&Value>,
    where_: &str,
    table_path: &str,
) {
    let Some(outputs) = outputs else {
        return;
    };
    if outputs.is_null() {
        return;
    }
    let op = join(table_path, "outputs");
    let Value::Array(items) = outputs else {
        c.err(
            code::WRONG_SHAPE,
            At::Value(&op),
            format!("{where_} outputs is a list of paths or {{ path, save_as, max_bytes }} tables"),
        );
        return;
    };
    for (n, o) in items.iter().enumerate() {
        let at = format!("{where_} outputs[{n}]");
        let p = format!("{op}[{n}]");
        let path = match o {
            Value::Object(entry) => {
                let mut unknown: Vec<&String> = entry
                    .keys()
                    .filter(|k| !OUTPUT_KEYS.contains(&k.as_str()))
                    .collect();
                unknown.sort();
                for k in unknown {
                    if k.starts_with("x-") {
                        continue;
                    }
                    let kp = join(&p, k);
                    let message = format!(
                        "unknown key {} in {at}{} an output holds path, save_as, and max_bytes (exclude and service come from split-layout \
imports), or x- extensions",
                        repr_str(k),
                        dym_or(k, OUTPUT_KEYS, ";")
                    );
                    c.err(code::UNKNOWN_KEY, At::Key(&kp), message.clone());
                    rename_fix(c, k, &kp, &p, OUTPUT_KEYS, &message);
                }
                if let Some(save_as) = entry.get("save_as")
                    && !save_as.is_null()
                {
                    let ok = save_as.as_str().is_some_and(|s| {
                        !s.trim().is_empty()
                            && !s.starts_with('/')
                            && !s.split('/').any(|seg| seg == "..")
                    });
                    if !ok {
                        c.err(
                            code::BAD_PATH,
                            At::Value(&join(&p, "save_as")),
                            format!(
                                "{at} save_as is a relative name for the saved copy, without .."
                            ),
                        );
                    }
                }
                if let Some(mb) = entry.get("max_bytes")
                    && !is_output_size(mb)
                {
                    c.err(
                        code::BAD_SIZE,
                        At::Value(&join(&p, "max_bytes")),
                        format!("{at} max_bytes is a positive number of bytes or a size such as \"100 MB\""),
                    );
                }
                if let Some(ex) = entry.get("exclude")
                    && !matches!(ex, Value::Array(a) if a.iter().all(Value::is_string))
                {
                    c.err(
                        code::WRONG_SHAPE,
                        At::Value(&join(&p, "exclude")),
                        format!("{at} exclude is a list of patterns"),
                    );
                }
                entry.get("path")
            }
            Value::String(_) => Some(o),
            _ => {
                c.err(
                    code::WRONG_SHAPE,
                    At::Value(&p),
                    format!("{at} is a path or a {{ path, save_as, max_bytes }} table"),
                );
                continue;
            }
        };
        let at_path = if o.is_object() {
            join(&p, "path")
        } else {
            p.clone()
        };
        match path.and_then(|v| nonblank(Some(v))) {
            None => {
                c.err(
                    code::MISSING_KEY,
                    At::Value(&at_path),
                    format!(
                        "{at} needs a path: an absolute path to a file or a folder in the sandbox"
                    ),
                );
            }
            Some(s) if !s.starts_with('/') => {
                c.err(
                    code::BAD_PATH,
                    At::Value(&at_path),
                    format!(
                        "{at} path {s} is relative; an output's path is absolute, such as /work/{s}"
                    ),
                );
            }
            Some(s) if s.split('/').any(|seg| seg == "..") => {
                c.err(
                    code::BAD_PATH,
                    At::Value(&at_path),
                    format!("{at} path {s} climbs out with ..; write the path it means"),
                );
            }
            Some(_) => {}
        }
    }
}

/// `[training]`.
pub(super) fn training(c: &mut Checker<'_>) {
    let t = table(c.config, &["training"]);
    if let Some(band) = t.get("learnable_band")
        && !band.is_null()
    {
        let ok = match band {
            Value::Array(b) if b.len() == 2 => match (num(&b[0]), num(&b[1])) {
                (Some(lo), Some(hi)) => 0.0 <= lo && lo < hi && hi <= 1.0,
                _ => false,
            },
            _ => false,
        };
        if !ok {
            c.err(
                code::TRAINING,
                At::Value("training.learnable_band"),
                "[training] learnable_band is [low, high], two pass rates with 0 <= low < high <= 1",
            );
        }
    }
    if let Some(f) = t.get("fork")
        && !f.is_boolean()
    {
        c.err(
            code::TRAINING,
            At::Value("training.fork"),
            "[training] fork is true (the default: a runtime may start episodes from a snapshot) or false (every episode is set up \
from scratch)",
        );
    }
    if let Some(mc) = t.get("max_concurrent")
        && !mc.is_null()
        && !int(mc).is_some_and(|n| n >= 1)
    {
        c.err(
            code::TRAINING,
            At::Value("training.max_concurrent"),
            "[training] max_concurrent is a whole number of episodes, at least 1",
        );
    }
    if let Some(a) = t.get("admit")
        && !a.is_object()
    {
        c.err(
            code::TRAINING,
            At::Value("training.admit"),
            "[training] admit is a table: { max_pass_rate, models, reasoning_effort }",
        );
    }
    if let Some(d) = t.get("difficulty")
        && !d.is_string()
    {
        c.err(
            code::TRAINING,
            At::Value("training.difficulty"),
            "[training] difficulty names an evidence file, such as \"evidence/difficulty.json\"",
        );
    }
}

/// `[integrity.controls] injection`: the controls, by key, to pair with
/// injected twins.
pub(super) fn injection(c: &mut Checker<'_>) {
    let controls = table(c.config, &["integrity", "controls"]);
    let Some(injection) = controls.get("injection") else {
        return;
    };
    if injection.is_null() {
        return;
    }
    let path = "integrity.controls.injection";
    let Some(names) = injection
        .as_array()
        .filter(|a| a.iter().all(Value::is_string))
    else {
        c.err(
            code::INJECTION,
            At::Value(path),
            "[integrity.controls] injection is a list of the controls to pair with injected twins, such as [\"known_bad\"]",
        );
        return;
    };
    const CONTROLS: &[&str] = &["no_op", "known_bad", "near_miss", "cheats"];
    for (i, name) in names.iter().enumerate() {
        let name = name.as_str().unwrap_or_default();
        let p = format!("{path}[{i}]");
        if !CONTROLS.contains(&name) {
            let message = format!(
                "[integrity.controls] injection names {}; it names controls by key: known_bad, near_miss, no_op, or cheats",
                repr_str(name)
            );
            c.err(code::INJECTION, At::Value(&p), message.clone());
            if let Some(close) = suggestion(name, CONTROLS) {
                choice_fix(c, &p, &close, &message);
            }
        } else if !controls.contains_key(name) {
            c.err(
                code::INJECTION,
                At::Value(&p),
                format!(
                    "[integrity.controls] injection pairs {name}, which the table does not declare"
                ),
            );
        }
    }
}

/// `[roles.<name>]`: tables, with `policy` a boolean.
pub(super) fn roles(c: &mut Checker<'_>) {
    let roles = table(c.config, &["roles"]);
    for (name, role) in roles {
        let rp = join("roles", name);
        if !role.is_object() && !name.starts_with("x-") {
            c.err(
                code::WRONG_SHAPE,
                At::Value(&rp),
                format!("[roles.{name}] is a table of the role's settings"),
            );
        }
        if let Value::Object(r) = role
            && let Some(p) = r.get("policy")
            && !p.is_boolean()
        {
            c.err(
                code::BAD_CHOICE,
                At::Value(&join(&rp, "policy")),
                format!(
                    "[roles.{name}] policy is true for a trainable role, or false (the default)"
                ),
            );
        }
    }
}
