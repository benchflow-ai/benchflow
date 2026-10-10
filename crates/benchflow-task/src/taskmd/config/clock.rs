//! `[sandbox] clock` and `timezone` (task.md's docs/clocks.md).
//!
//! A clock sets the time the agent sees: a start in RFC 3339 with the zone's
//! offset at that instant, how it advances, and how the runtime enforces it.
//! These rules check the start's offset against the zone, that a set clock
//! runs with no network, and whether the run can cross midnight, for the
//! task and for each variant that changes its sandbox or agent.

use std::collections::HashSet;
use std::sync::LazyLock;

use jiff::civil::DateTime;
use jiff::tz::{AmbiguousOffset, Offset, TimeZone};
use regex::Regex;
use serde_json::{Map, Value};

use crate::source::join;

use super::super::code;
use super::super::pyfmt::{fmt_duration, json_dumps, repr_str, str_of};
use super::super::pyval::{duration_s, py_eq, table};
use super::super::report::At;
use super::Checker;

static RFC3339: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{4}-\d{2}-\d{2}[Tt]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:[Zz]|[+-]\d{2}:\d{2})\n?\z")
        .unwrap()
});
static LOCAL_TIME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^\d{4}-\d{2}-\d{2}[Tt ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?\n?\z").unwrap()
});
static ISO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2})(?::(\d{2})(?::(\d{2})(?:\.(\d+))?)?)?)?(?:(Z)|([+-])(\d{2}):(\d{2}))?$",
    )
    .unwrap()
});

/// A zone by name, or why not: `"unknown"`, or that this machine has no
/// time zone data (the reference's `_zone`).
fn zone(name: &str) -> Result<TimeZone, &'static str> {
    match jiff::tz::db().get(name) {
        Ok(tz) => Ok(tz),
        Err(_) if jiff::tz::db().get("America/New_York").is_ok() => Err("unknown"),
        Err(_) => Err("this machine has no time zone data to check it against"),
    }
}

/// A date and time as Python's `datetime` holds one: civil time, and an
/// offset when it is aware.
#[derive(Debug, Clone, Copy)]
struct PyDateTime {
    dt: DateTime,
    offset: Option<i32>,
}

impl PyDateTime {
    /// `datetime.fromisoformat` for the forms a task writes: a date, then
    /// optionally a time (to the hour, minute, or second, with a fraction),
    /// then optionally `Z` or an offset.
    fn parse(text: &str) -> Option<Self> {
        let caps = ISO.captures(text)?;
        let n = |i: usize| {
            caps.get(i)
                .map_or(Some(0), |m| m.as_str().parse::<i32>().ok())
        };
        let micros = caps.get(7).map_or(0, |m| {
            let digits: String = m.as_str().chars().take(6).collect();
            format!("{digits:0<6}").parse::<i32>().unwrap_or(0)
        });
        let dt = DateTime::new(
            n(1)? as i16,
            n(2)? as i8,
            n(3)? as i8,
            n(4)? as i8,
            n(5)? as i8,
            n(6)? as i8,
            micros * 1000,
        )
        .ok()?;
        let offset = if caps.get(8).is_some() {
            Some(0)
        } else if let Some(sign) = caps.get(9) {
            let secs = n(10)? * 3600 + n(11)? * 60;
            Some(if sign.as_str() == "-" { -secs } else { secs })
        } else {
            None
        };
        Some(Self { dt, offset })
    }

    fn micros(&self) -> i32 {
        self.dt.subsec_nanosecond() / 1000
    }

    /// `isoformat()`: `2026-03-14T09:00:00-04:00`.
    fn isoformat(&self) -> String {
        let d = self.dt;
        let mut out = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            d.year(),
            d.month(),
            d.day(),
            d.hour(),
            d.minute(),
            d.second()
        );
        if self.micros() != 0 {
            out.push_str(&format!(".{:06}", self.micros()));
        }
        if let Some(off) = self.offset {
            out.push_str(&iso_offset(off));
        }
        out
    }

    /// `strftime('%H:%M')`.
    fn hh_mm(&self) -> String {
        format!("{:02}:{:02}", self.dt.hour(), self.dt.minute())
    }

    /// The same instant in another offset.
    fn at_offset(&self, offset: i32) -> Option<Self> {
        let ts = Offset::from_seconds(self.offset?)
            .ok()?
            .to_timestamp(self.dt)
            .ok()?;
        let dt = Offset::from_seconds(offset).ok()?.to_datetime(ts);
        Some(Self {
            dt,
            offset: Some(offset),
        })
    }

    /// `astimezone(zone)`.
    fn in_zone(&self, tz: &TimeZone) -> Option<Self> {
        let ts = Offset::from_seconds(self.offset?)
            .ok()?
            .to_timestamp(self.dt)
            .ok()?;
        self.at_offset(tz.to_offset(ts).seconds())
    }
}

/// An offset as `isoformat()` writes it: `-04:00`, or `+05:21:10`.
fn iso_offset(secs: i32) -> String {
    let sign = if secs < 0 { '-' } else { '+' };
    let a = secs.abs();
    let (h, m, s) = (a / 3600, a % 3600 / 60, a % 60);
    if s != 0 {
        format!("{sign}{h:02}:{m:02}:{s:02}")
    } else {
        format!("{sign}{h:02}:{m:02}")
    }
}

/// An offset as the reference's `_offset` writes it: whole minutes.
fn short_offset(secs: i32) -> String {
    let sign = if secs < 0 { '-' } else { '+' };
    let minutes = secs.abs() / 60;
    format!("{sign}{:02}:{:02}", minutes / 60, minutes % 60)
}

/// Python's `str()` of a TOML date-time: `2026-03-14 09:00:00`,
/// `1979-05-27 07:32:00+00:00`, `2026-03-14`, or `09:00:00`.
pub(crate) fn toml_datetime_str(d: &toml_edit::Datetime) -> String {
    let date = d
        .date
        .map(|x| format!("{:04}-{:02}-{:02}", x.year, x.month, x.day));
    let time = d.time.map(|t| {
        let micros = t.nanosecond.unwrap_or(0) / 1000;
        let base = format!("{:02}:{:02}:{:02}", t.hour, t.minute, t.second.unwrap_or(0));
        if micros != 0 {
            format!("{base}.{micros:06}")
        } else {
            base
        }
    });
    let offset = d.offset.map(|o| match o {
        toml_edit::Offset::Z => "+00:00".to_string(),
        toml_edit::Offset::Custom { minutes } => iso_offset(i32::from(minutes) * 60),
    });
    match (date, time) {
        (Some(date), Some(time)) => format!("{date} {time}{}", offset.unwrap_or_default()),
        (Some(date), None) => date,
        (None, Some(time)) => time,
        (None, None) => String::new(),
    }
}

/// A clock start as a value: `Ok` with an aware time, or `Err` with why not
/// (the reference's `clock_start`).
fn clock_start(
    value: &Value,
    datetime: Option<&toml_edit::Datetime>,
) -> Result<PyDateTime, &'static str> {
    if let Some(d) = datetime {
        return match (d.date.is_some(), d.time.is_some(), d.offset.is_some()) {
            (true, true, true) => {
                PyDateTime::parse(&toml_datetime_str(d)).ok_or("not an RFC 3339 time")
            }
            (true, true, false) => Err("no offset"),
            (true, false, _) => Err("a date, not a time"),
            _ => Err("not an RFC 3339 time"),
        };
    }
    match value {
        Value::String(s) if RFC3339.is_match(s) => PyDateTime::parse(
            s.trim_end_matches('\n')
                .replace('z', "Z")
                .replace('t', "T")
                .as_str(),
        )
        .ok_or("not an RFC 3339 time"),
        Value::String(s) if LOCAL_TIME.is_match(s) => Err("no offset"),
        _ => Err("not an RFC 3339 time"),
    }
}

/// A local time written with its zone's offset at that time, or with a
/// placeholder offset (the reference's `_with_offset`).
fn with_offset(local: &str, zone_name: Option<&str>) -> String {
    let tz = zone_name.and_then(|n| zone(n).ok());
    let Some(naive) = PyDateTime::parse(&local.replace(' ', "T")) else {
        return format!("{local}<offset>");
    };
    let Some(tz) = tz else {
        return format!("{}<offset>", naive.isoformat());
    };
    // Python's fold=0: in a gap or a fold, the offset before the transition.
    let offset = match tz.to_ambiguous_timestamp(naive.dt).offset() {
        AmbiguousOffset::Unambiguous { offset } => offset,
        AmbiguousOffset::Gap { before, .. } | AmbiguousOffset::Fold { before, .. } => before,
    };
    PyDateTime {
        dt: naive.dt,
        offset: Some(offset.seconds()),
    }
    .isoformat()
}

/// The message for `[world.desktop] clock`, which moved to `[sandbox]
/// clock`.
pub(crate) fn desktop_clock_moved(value: &Value, desktop: &Map<String, Value>) -> String {
    if let Value::String(s) = value
        && let Some(rest) = s.strip_prefix("frozen:")
    {
        let tz = desktop
            .get("timezone")
            .filter(|v| super::super::pyval::truthy(Some(v)));
        let start = with_offset(rest.trim(), tz.map(str_of).as_deref());
        return format!(
            "[world.desktop] clock moved to [sandbox] clock: write [sandbox] clock = {{ start = \"{start}\", advance = \"none\", \
enforce = \"world\" }}, which is what \"{s}\" meant, and set network = \"none\" for the agent's phase, which a clock requires"
        );
    }
    "[world.desktop] clock moved to [sandbox] clock = { start, advance, enforce }: one clock for the whole task (docs/document.md, \
Clocks)"
        .to_string()
}

/// A variant's own `[sandbox]` and `[agent]` patches, and its path.
struct Patches<'m> {
    sandbox: &'m Map<String, Value>,
    agent: &'m Map<String, Value>,
    base: String,
}

/// The tables a run of the rules reads, and where their keys sit.
struct Scope<'m> {
    sandbox: Map<String, Value>,
    agent: Map<String, Value>,
    patches: Option<Patches<'m>>,
    prefix: String,
}

impl Scope<'_> {
    fn sandbox_path(&self, key: &str) -> String {
        match &self.patches {
            Some(p) if p.sandbox.contains_key(key) => join(&join(&p.base, "sandbox"), key),
            _ => join("sandbox", key),
        }
    }

    fn agent_path(&self, key: &str) -> String {
        match &self.patches {
            Some(p) if p.agent.contains_key(key) => join(&join(&p.base, "agent"), key),
            _ => join("agent", key),
        }
    }
}

/// The clock rules for the task, and for each variant only what it changes.
pub(super) fn check(c: &mut Checker<'_>) {
    let config = c.config;
    let sandbox = table(config, &["sandbox"]);
    let agent = table(config, &["agent"]);
    rules(
        c,
        &Scope {
            sandbox: sandbox.clone(),
            agent: agent.clone(),
            patches: None,
            prefix: String::new(),
        },
    );
    for (name, variant) in table(config, &["variants"]) {
        let Value::Object(v) = variant else {
            continue;
        };
        if !v.contains_key("sandbox") && !v.contains_key("agent") {
            continue;
        }
        let vs_patch = table(variant, &["sandbox"]);
        let va_patch = table(variant, &["agent"]);
        let mut vs = sandbox.clone();
        vs.extend(vs_patch.iter().map(|(k, x)| (k.clone(), x.clone())));
        let mut va = agent.clone();
        va.extend(va_patch.iter().map(|(k, x)| (k.clone(), x.clone())));
        let clock_unset = vs.get("clock").is_none_or(Value::is_null);
        if clock_unset && py_eq(&Value::Object(va.clone()), &Value::Object(agent.clone())) {
            continue;
        }
        let prefix = format!("variant {name}: ");
        let before = c.sink.found.len();
        let base = join("variants", name);
        rules(
            c,
            &Scope {
                sandbox: vs,
                agent: va,
                patches: Some(Patches {
                    sandbox: vs_patch,
                    agent: va_patch,
                    base,
                }),
                prefix: prefix.clone(),
            },
        );
        // Report a variant only for what it changes.
        let seen: HashSet<String> = c.sink.found[..before]
            .iter()
            .map(|f| {
                f.diag
                    .message
                    .strip_prefix(&prefix)
                    .unwrap_or(&f.diag.message)
                    .to_string()
            })
            .collect();
        let mut i = before;
        while i < c.sink.found.len() {
            let m = &c.sink.found[i].diag.message;
            if seen.contains(m.strip_prefix(&prefix).unwrap_or(m)) {
                c.sink.found.remove(i);
            } else {
                i += 1;
            }
        }
    }
}

/// The reference's `_clock_rules`.
fn rules(c: &mut Checker<'_>, s: &Scope<'_>) {
    let config = c.config;
    let w = s.prefix.as_str();
    let mut tz_name: Option<String> = None;
    let mut tz: Option<TimeZone> = None;
    let world_zones: Vec<(String, String)> = ["desktop", "browser"]
        .iter()
        .filter_map(|wk| match table(config, &["world", wk]).get("timezone") {
            Some(Value::String(z)) => Some((format!("[world.{wk}] timezone"), z.clone())),
            _ => None,
        })
        .collect();
    let tz_path = s.sandbox_path("timezone");
    if let Some(tz_value) = s.sandbox.get("timezone") {
        match tz_value {
            Value::String(name) if !name.trim().is_empty() => {
                tz_name = Some(name.clone());
                match zone(name) {
                    Ok(z) => tz = Some(z),
                    Err("unknown") => {
                        c.err(
                            code::TIMEZONE,
                            At::Value(&tz_path),
                            format!(
                                "{w}[sandbox] timezone {} is not a zone in the tz database; write a name such as \"America/New_York\" or \
\"UTC\"",
                                repr_str(name)
                            ),
                        );
                    }
                    Err(why) => {
                        c.warn(
                            code::TIMEZONE,
                            At::Value(&tz_path),
                            format!(
                                "{w}cannot check [sandbox] timezone {}: {why}",
                                repr_str(name)
                            ),
                        );
                    }
                }
            }
            _ => {
                c.err(
                    code::TIMEZONE,
                    At::Value(&tz_path),
                    format!("{w}[sandbox] timezone is a zone name such as \"America/New_York\""),
                );
            }
        }
        let env_tz = table(&Value::Object(s.sandbox.clone()), &["env"])
            .get("TZ")
            .cloned();
        if let (Some(name), Some(env_tz)) = (&tz_name, env_tz) {
            let same = if env_tz.as_str() == Some(name.as_str()) {
                "the same zone".to_string()
            } else {
                format!("{}, which differs", super::super::pyfmt::repr(&env_tz))
            };
            let env_path = join(&s.sandbox_path("env"), "TZ");
            c.warn(
                code::TIMEZONE,
                At::Key(&env_path),
                format!("{w}[sandbox] env sets TZ to {same}; [sandbox] timezone wins over it, so remove TZ from env"),
            );
        }
        let shown = tz_name
            .as_deref()
            .map_or_else(|| "None".to_string(), repr_str);
        for (key, other) in &world_zones {
            if tz_name.as_deref() != Some(other.as_str()) {
                c.warn(
                    code::TIMEZONE,
                    At::Value(&tz_path),
                    format!("{w}[sandbox] timezone {shown} and {key} {} differ; the agent should see one zone", repr_str(other)),
                );
            }
        }
    } else if let Some((_, first)) = world_zones.first() {
        tz_name = Some(first.clone());
        tz = zone(first).ok();
    }

    let clock_path = s.sandbox_path("clock");
    let Some(clock) = s.sandbox.get("clock").filter(|v| !v.is_null()) else {
        return;
    };
    let clock = match clock {
        Value::Object(m) => m,
        Value::String(text) => {
            let hint = if text.starts_with("frozen:") {
                let mut desk = Map::new();
                if let Some(n) = &tz_name {
                    desk.insert("timezone".into(), Value::String(n.clone()));
                }
                let full = desktop_clock_moved(clock, &desk);
                full.split_once(": ")
                    .map(|(_, rest)| format!(": {rest}"))
                    .unwrap_or_default()
            } else {
                String::new()
            };
            c.err(
                code::CLOCK,
                At::Value(&clock_path),
                format!("{w}[sandbox] clock is a table {{ start, advance, enforce }}{hint}"),
            );
            return;
        }
        _ => {
            c.err(
                code::CLOCK,
                At::Value(&clock_path),
                format!("{w}[sandbox] clock is a table {{ start, advance, enforce }}"),
            );
            return;
        }
    };
    let advance = clock
        .get("advance")
        .cloned()
        .unwrap_or(Value::String("real".into()));
    let enforce = clock.get("enforce").filter(|v| !v.is_null());
    let start_path = join(&clock_path, "start");
    let mut start: Option<PyDateTime> = None;
    if let Some(start_value) = clock.get("start") {
        let datetime = c.datetime(&start_path);
        match clock_start(start_value, datetime.as_ref()) {
            Ok(t) => start = Some(t),
            Err("no offset") => {
                let shown = datetime
                    .as_ref()
                    .map_or_else(|| str_of(start_value), toml_datetime_str);
                c.err(
                    code::CLOCK,
                    At::Value(&start_path),
                    format!(
                        "{w}[sandbox] clock start {shown} has no offset: write it in RFC 3339 with the zone's offset, such as {}, so the \
instant is not left to the machine that runs it",
                        with_offset(&shown, tz_name.as_deref())
                    ),
                );
            }
            Err(why) => {
                c.err(
                    code::CLOCK,
                    At::Value(&start_path),
                    format!(
                        "{w}[sandbox] clock start is an RFC 3339 time with an offset, such as \"2026-03-14T09:00:00-04:00\" ({why})"
                    ),
                );
            }
        }
    } else if advance.as_str() == Some("real") {
        c.err(
            code::CLOCK,
            At::Value(&clock_path),
            format!(
                "{w}[sandbox] clock sets nothing: with no start and advance = \"real\" it is the real clock. Give a start, or remove the \
clock"
            ),
        );
    }
    if enforce.is_none() && (clock.contains_key("start") || advance.as_str() != Some("real")) {
        c.err(
            code::CLOCK,
            At::Value(&clock_path),
            format!(
                "{w}[sandbox] clock needs enforce: \"preload\" (a library in the sandbox; fidelity, not protection), \"vm\" (a guest \
with its own clock), or \"world\" (the task's own services hold the time)"
            ),
        );
    }
    if advance.as_str() == Some("none") && enforce.is_some_and(|e| e.as_str() != Some("world")) {
        c.err(
            code::CLOCK,
            At::Value(&join(&clock_path, "enforce")),
            format!(
                "{w}[sandbox] clock advance = \"none\" needs enforce = \"world\": a clock that stands still under preload or vm breaks \
sleeps and timers"
            ),
        );
    }

    // The network while the clock runs: the agent's phase, and each stage's.
    let (net_key, net_value, net_path) = match s.agent.get("network") {
        Some(v) => (
            "[agent] network".to_string(),
            v.clone(),
            s.agent_path("network"),
        ),
        None => (
            "[sandbox] network".to_string(),
            s.sandbox
                .get("network")
                .cloned()
                .unwrap_or(Value::String("open".into())),
            if s.sandbox.contains_key("network") {
                s.sandbox_path("network")
            } else {
                clock_path.clone()
            },
        ),
    };
    let mut phases = vec![(net_key, net_value, net_path)];
    for (name, st) in table(config, &["stages"]) {
        if let Some(net) = table(st, &["agent"])
            .get("network")
            .filter(|v| !v.is_null())
        {
            let p = join(&join(&join("stages", name), "agent"), "network");
            phases.push((format!("[stages.{name}.agent] network"), net.clone(), p));
        }
    }
    for (key, net, path) in phases {
        if net.as_str() == Some("none") {
            continue;
        }
        let shown = match &net {
            Value::String(t) => format!("\"{t}\""),
            other => json_dumps(other),
        };
        let default = if key == "[sandbox] network" && !s.sandbox.contains_key("network") {
            " (the default)"
        } else {
            ""
        };
        c.err(
            code::CLOCK_NETWORK,
            At::Value(&path),
            format!(
                "{w}a set clock needs network = \"none\" while it runs, and {key} is {shown}{default}: the network tells the real date \
(TLS, HTTP Date headers, package mirrors). Set [agent] network = \"none\""
            ),
        );
    }

    let Some(start) = start else {
        return;
    };
    let start_offset = start.offset.unwrap_or(0);
    let local = if tz.is_none() && tz_name.is_none() {
        if start_offset != 0 {
            let utc = start.at_offset(0).map(|u| u.hh_mm()).unwrap_or_default();
            c.warn(
                code::TIMEZONE,
                At::Value(&start_path),
                format!(
                    "{w}[sandbox] clock start is at {}, but no timezone is set, so the sandbox keeps the image's zone (usually UTC), \
where it reads {utc}; set [sandbox] timezone",
                    short_offset(start_offset)
                ),
            );
        }
        start
    } else if let Some(zone) = &tz {
        let Some(local) = start.in_zone(zone) else {
            return;
        };
        if local.offset != start.offset {
            let shown = c.datetime(&start_path).as_ref().map_or_else(
                || clock.get("start").map(str_of).unwrap_or_default(),
                toml_datetime_str,
            );
            let lo = short_offset(local.offset.unwrap_or(0));
            c.err(
                code::TIMEZONE,
                At::Value(&start_path),
                format!(
                    "{w}[sandbox] clock start {shown} is at {}, but {} is at {lo} at that instant, so the agent would see {}; write \
start = \"{}\", or the time you meant with {lo}",
                    short_offset(start_offset),
                    tz_name.as_deref().unwrap_or_default(),
                    local.hh_mm(),
                    local.isoformat()
                ),
            );
        }
        local
    } else {
        start
    };
    if advance.as_str() == Some("none") {
        return;
    }
    let timeout = duration_s(s.agent.get("timeout"));
    let zone_name = tz_name.clone().unwrap_or_else(|| "the image's zone".into());
    let timeout_path = s.agent_path("timeout");
    match timeout {
        None => {
            c.warn(
                code::CLOCK_MIDNIGHT,
                At::Value(&start_path),
                format!(
                    "{w}cannot tell whether the run crosses midnight in {zone_name}: [agent] timeout is not set, so the run's length is \
the runtime's"
                ),
            );
        }
        Some(secs) => {
            let end = jiff::SignedDuration::try_from_secs_f64(secs)
                .ok()
                .and_then(|d| local.dt.checked_add(d).ok());
            if end.is_some_and(|e| e.date() != local.dt.date()) {
                c.warn(
                    code::CLOCK_MIDNIGHT,
                    At::Value(&timeout_path),
                    format!(
                        "{w}the run can cross midnight in {zone_name}: it starts at {} and may run {}, so the agent's date can change \
during the run while TASKMD_CLOCK_START keeps the start's date. Start earlier in the day, or have the verifier read \
TASKMD_CLOCK_HANDOFF",
                        local.hh_mm(),
                        fmt_duration(secs)
                    ),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isoformat_like_python() {
        let t = PyDateTime::parse("2026-03-14T09:00:00.5-04:00").unwrap();
        assert_eq!(t.isoformat(), "2026-03-14T09:00:00.500000-04:00");
        let t = PyDateTime::parse("2026-03-14T09:00:00.123456789Z").unwrap();
        assert_eq!(t.isoformat(), "2026-03-14T09:00:00.123456+00:00");
        assert_eq!(
            PyDateTime::parse("2026-03-14T09:00").unwrap().isoformat(),
            "2026-03-14T09:00:00"
        );
        assert!(PyDateTime::parse("2026-02-30T09:00:00Z").is_none());
    }

    #[test]
    fn offsets_follow_fold_zero() {
        if zone("America/New_York").is_err() {
            return; // no time zone data on this machine
        }
        assert_eq!(
            with_offset("2026-03-08 02:30:00", Some("America/New_York")),
            "2026-03-08T02:30:00-05:00"
        );
        assert_eq!(
            with_offset("2026-11-01T01:30:00", Some("America/New_York")),
            "2026-11-01T01:30:00-04:00"
        );
        assert_eq!(
            with_offset("2026-03-14T09:00:00", Some("America/New_York")),
            "2026-03-14T09:00:00-04:00"
        );
        assert_eq!(
            with_offset("2026-03-14T09:00:00", None),
            "2026-03-14T09:00:00<offset>"
        );
        assert_eq!(with_offset("garbage", None), "garbage<offset>");
        assert_eq!(iso_offset(5 * 3600 + 21 * 60 + 10), "+05:21:10");
        assert_eq!(short_offset(-(4 * 3600)), "-04:00");
    }

    #[test]
    fn toml_datetimes_print_as_python_does() {
        let d: toml_edit::Datetime = "1979-05-27T07:32:00Z".parse().unwrap();
        assert_eq!(toml_datetime_str(&d), "1979-05-27 07:32:00+00:00");
        let d: toml_edit::Datetime = "2026-03-14T09:00:00.5-04:00".parse().unwrap();
        assert_eq!(toml_datetime_str(&d), "2026-03-14 09:00:00.500000-04:00");
        let d: toml_edit::Datetime = "2026-03-14".parse().unwrap();
        assert_eq!(toml_datetime_str(&d), "2026-03-14");
    }
}
