//! Harbor's `task.toml` config to task.md's config and back, ported from
//! task.md's `tools/convert.py` (`config_from_harbor`, `config_to_harbor`).
//!
//! Import keeps everything: Harbor settings become task.md settings
//! (`[environment]` is `[sandbox]`, `memory_mb = 4096` is `memory = "4
//! GB"`, steps are stages), and keys task.md does not model are kept as
//! written under `[import.harbor]` (or `[import.v06]`). Every import
//! records where each Harbor path went, so comments follow their settings.
//! Export does the reverse and lists what Harbor cannot hold.

use std::collections::HashMap;
use std::sync::LazyLock;

use regex::Regex;

use super::tomltext::{self, Comments};
use super::value::{KPath, Seg, Table, Val, dotted, kidx, kjoin, kkey, kp, py_float_repr};

/// How a value changes on its way between the formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conv {
    Same,
    /// Seconds to a task.md duration.
    Duration,
    /// A task.md duration to seconds.
    Seconds,
    /// Megabytes to a task.md size.
    Size,
    /// A task.md size to megabytes.
    Megabytes,
    Ready,
    Sandbox,
    Snapshot,
    Outputs,
    Authors,
}

/// One key: Harbor's name, task.md's name, the conversion each way.
struct Spec {
    harbor: &'static str,
    native: &'static str,
    to_native: Conv,
    to_harbor: Conv,
}

const fn same(harbor: &'static str, native: &'static str) -> Spec {
    Spec {
        harbor,
        native,
        to_native: Conv::Same,
        to_harbor: Conv::Same,
    }
}

const fn conv(
    harbor: &'static str,
    native: &'static str,
    to_native: Conv,
    to_harbor: Conv,
) -> Spec {
    Spec {
        harbor,
        native,
        to_native,
        to_harbor,
    }
}

static SANDBOX: &[Spec] = &[
    same("docker_image", "image"),
    same("os", "os"),
    same("cpus", "cpus"),
    conv("memory_mb", "memory", Conv::Size, Conv::Megabytes),
    conv("storage_mb", "disk", Conv::Size, Conv::Megabytes),
    same("gpus", "gpus"),
    same("gpu_types", "gpu_types"),
    same("tpu", "tpu"),
    same("workdir", "workdir"),
    same("env", "env"),
    same("skills_dir", "skills"),
    same("mcp_servers", "mcp"),
    conv("healthcheck", "ready", Conv::Ready, Conv::Ready),
    conv(
        "build_timeout_sec",
        "build_timeout",
        Conv::Duration,
        Conv::Seconds,
    ),
];
/// bench's opt-out from refusing the cloud metadata endpoints to the agent.
/// task.md keeps it in `[sandbox]`; the split layout and BenchFlow 0.8's
/// task.md keep it in `[metadata]`, the one table whose keys BenchFlow 0.8
/// never refuses (its `[environment]` refuses keys it does not know).
pub(crate) const METADATA_OPT_OUT: &str = "x-allow-cloud-metadata";

static READY: &[Spec] = &[
    same("command", "run"),
    conv("interval_sec", "interval", Conv::Duration, Conv::Seconds),
    conv("timeout_sec", "timeout", Conv::Duration, Conv::Seconds),
    conv(
        "start_period_sec",
        "start_period",
        Conv::Duration,
        Conv::Seconds,
    ),
    conv(
        "start_interval_sec",
        "start_interval",
        Conv::Duration,
        Conv::Seconds,
    ),
    same("retries", "retries"),
];
static AGENT: &[Spec] = &[
    conv("timeout_sec", "timeout", Conv::Duration, Conv::Seconds),
    same("user", "user"),
];
static VERIFIER: &[Spec] = &[
    conv("timeout_sec", "timeout", Conv::Duration, Conv::Seconds),
    same("user", "user"),
    same("env", "env"),
    same("environment_mode", "isolation"),
    conv("environment", "sandbox", Conv::Sandbox, Conv::Sandbox),
    conv("collect", "snapshot", Conv::Snapshot, Conv::Snapshot),
];
static SNAPSHOT: &[Spec] = &[
    same("command", "run"),
    same("service", "service"),
    conv("timeout_sec", "timeout", Conv::Duration, Conv::Seconds),
    same("user", "user"),
];
static OUTPUT: &[Spec] = &[
    same("source", "path"),
    same("destination", "save_as"),
    same("exclude", "exclude"),
    same("service", "service"),
];
static PACKAGE: &[Spec] = &[
    same("name", "name"),
    same("version", "version"),
    same("description", "description"),
    conv("authors", "authors", Conv::Authors, Conv::Authors),
    same("keywords", "keywords"),
];
static STEP: &[Spec] = &[
    same("min_reward", "gate"),
    conv("healthcheck", "ready", Conv::Ready, Conv::Ready),
    conv("artifacts", "outputs", Conv::Outputs, Conv::Outputs),
];
static ENV_ONLY: &[Spec] = &[same("env", "env")];
const PHASE_NETWORK: [&str; 2] = ["network_mode", "allowed_hosts"];

static DURATION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?:(?P<h>\d+(?:\.\d+)?)h)?(?:(?P<m>\d+(?:\.\d+)?)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?$")
        .unwrap()
});
static SIZE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(?P<n>\d+(?:\.\d+)?) ?(?P<u>MB|GB|TB)$").unwrap());
static AUTHOR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^(?P<name>[^<>]*?)(?: <(?P<email>[^<>@\s]+@[^<>\s]+)>)?$").unwrap()
});

/// Python's `repr()` of a value, for notes.
pub(crate) fn py_repr(v: &Val) -> String {
    match v {
        Val::Str(s) => {
            if s.contains('\'') && !s.contains('"') {
                format!("\"{s}\"")
            } else {
                format!("'{}'", s.replace('\\', "\\\\").replace('\'', "\\'"))
            }
        }
        Val::Int(i) => i.to_string(),
        Val::Float(f) => py_float_repr(*f),
        Val::Bool(b) => {
            if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Val::Date(d) => d.clone(),
        Val::Null => "None".into(),
        Val::List(items) => format!(
            "[{}]",
            items.iter().map(py_repr).collect::<Vec<_>>().join(", ")
        ),
        Val::Table(t) => format!(
            "{{{}}}",
            t.iter()
                .map(|(k, v)| format!("{}: {}", py_repr(&Val::Str(k.clone())), py_repr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// 600 -> "10m", 5400 -> "1h30m", 0.5 -> "0.5s".
pub(crate) fn duration(v: &Val) -> Option<String> {
    let sec = v.py_float()?;
    if sec.fract() == 0.0 && sec.is_finite() && sec >= 0.0 {
        let total = sec as u64;
        let (h, rest) = (total / 3600, total % 3600);
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
        if out.is_empty() {
            out.push_str("0s");
        }
        Some(out)
    } else {
        Some(format!("{}s", py_float_repr(sec)))
    }
}

/// A task.md duration to seconds.
pub(crate) fn seconds(v: &Val) -> Option<f64> {
    let text = v.as_str()?;
    let c = DURATION.captures(text)?;
    if c.name("h").is_none() && c.name("m").is_none() && c.name("s").is_none() {
        return None;
    }
    let part = |n: &str| {
        c.name(n)
            .map_or(0.0, |m| m.as_str().parse::<f64>().unwrap_or(0.0))
    };
    Some(part("h") * 3600.0 + part("m") * 60.0 + part("s"))
}

/// 4096 -> "4 GB", 1536 -> "1536 MB" (1 GB is 1024 MB, as in Docker).
pub(crate) fn size(v: &Val) -> Option<String> {
    let mb = v.py_int()?;
    for (unit, n) in [("TB", 1024 * 1024), ("GB", 1024)] {
        if mb != 0 && mb.rem_euclid(n) == 0 {
            return Some(format!("{} {unit}", mb.div_euclid(n)));
        }
    }
    Some(format!("{mb} MB"))
}

/// A task.md size to megabytes, rounding as Python's `round` does.
pub(crate) fn megabytes(v: &Val) -> Option<i64> {
    let text = v.as_str()?;
    let c = SIZE.captures(text)?;
    let n: f64 = c["n"].parse().ok()?;
    let unit = match &c["u"] {
        "MB" => 1.0,
        "GB" => 1024.0,
        _ => 1024.0 * 1024.0,
    };
    Some((n * unit).round_ties_even() as i64)
}

/// Harbor's parser for its deprecated `memory = "4G"`.
fn harbor_size_mb(text: &str) -> Option<i64> {
    crate::split::config::harbor_size_mb(text)
}

/// Harbor `network_mode` + `allowed_hosts` to task.md's `network`.
fn network(mode: &Val, hosts: Option<&Val>) -> Option<Val> {
    match mode.as_str()? {
        "no-network" => Some(Val::Str("none".into())),
        "public" => Some(Val::Str("open".into())),
        "allowlist" => Some(Val::List(match hosts {
            None => Vec::new(),
            Some(h) if !h.truthy() => Vec::new(),
            Some(h) => iterate(h)?,
        })),
        _ => None,
    }
}

/// Python's `list(v)`: a list's items, a string's characters, a table's
/// keys; `None` for anything else.
fn iterate(v: &Val) -> Option<Vec<Val>> {
    match v {
        Val::List(items) => Some(items.clone()),
        Val::Str(s) => Some(s.chars().map(|c| Val::Str(c.to_string())).collect()),
        Val::Table(t) => Some(t.keys().map(|k| Val::Str(k.clone())).collect()),
        _ => None,
    }
}

fn author(a: &Val) -> Val {
    let Val::Table(t) = a else { return a.clone() };
    let keys_ok = t.keys().all(|k| k == "name" || k == "email");
    let Some(name) = t.get("name").and_then(Val::as_str) else {
        return a.clone();
    };
    if !keys_ok || name.contains('<') {
        return a.clone();
    }
    match t.get("email") {
        None => Val::Str(name.to_string()),
        Some(Val::Str(email)) => {
            let joined = format!("{name} <{email}>");
            match AUTHOR.captures(&joined) {
                Some(c)
                    if c.name("email").map(|m| m.as_str()) == Some(email.as_str())
                        && &c["name"] == name =>
                {
                    Val::Str(joined)
                }
                _ => a.clone(),
            }
        }
        Some(_) => a.clone(),
    }
}

fn author_harbor(a: &Val) -> Val {
    let Val::Str(s) = a else { return a.clone() };
    let mut t = Table::new();
    match AUTHOR.captures(s) {
        Some(c) if c.name("email").is_some() => {
            t.insert("name".into(), Val::Str(c["name"].to_string()));
            t.insert("email".into(), Val::Str(c["email"].to_string()));
        }
        _ => {
            t.insert("name".into(), Val::Str(s.clone()));
        }
    }
    Val::Table(t)
}

/// `{k: d[k] for k in first if k in d} | {the rest, in order}`.
pub(crate) fn ordered(d: &Table, first: &[&str]) -> Table {
    let mut out = Table::new();
    for k in first {
        if let Some(v) = d.get(*k) {
            out.insert((*k).to_string(), v.clone());
        }
    }
    for (k, v) in d {
        if !first.contains(&k.as_str()) {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

/// A key path for keeping: list indices become string keys.
fn keep_key(s: &Seg) -> String {
    match s {
        Seg::Key(k) => k.clone(),
        Seg::Index(i) => i.to_string(),
    }
}

/// Harbor `task.toml` data to task.md config, recording where each Harbor
/// path went. Keys task.md does not model go to `[import.<stash>]`.
pub(crate) struct Import {
    stash_name: String,
    pub moves: HashMap<KPath, KPath>,
    keep: Table,
    pub notes: Vec<String>,
}

/// A conversion Python would refuse with a TypeError or ValueError.
pub(crate) struct Refused;

impl Import {
    fn new(stash: &str) -> Self {
        Self {
            stash_name: stash.to_string(),
            moves: HashMap::new(),
            keep: Table::new(),
            notes: Vec::new(),
        }
    }

    fn stash(&mut self, hpath: &[Seg], value: Val, kpath: Option<&[Seg]>) {
        let kpath = kpath.unwrap_or(hpath);
        let mut d = &mut self.keep;
        for p in &kpath[..kpath.len() - 1] {
            let entry = d
                .entry(keep_key(p))
                .or_insert_with(|| Val::Table(Table::new()));
            if !matches!(entry, Val::Table(_)) {
                *entry = Val::Table(Table::new());
            }
            d = entry.as_table_mut().expect("a table");
        }
        d.insert(keep_key(&kpath[kpath.len() - 1]), value);
        let mut to = kp(&["import", &self.stash_name]);
        to.extend(kpath.iter().cloned());
        self.moves.insert(hpath.to_vec(), to);
    }

    fn convert(
        &mut self,
        conv: Conv,
        v: &Val,
        hpath: &[Seg],
        npath: &[Seg],
        kpath: &[Seg],
    ) -> Result<Val, Refused> {
        Ok(match conv {
            Conv::Same | Conv::Seconds | Conv::Megabytes => v.clone(),
            Conv::Duration => Val::Str(duration(v).ok_or(Refused)?),
            Conv::Size => Val::Str(size(v).ok_or(Refused)?),
            Conv::Ready => Val::Table(self.table(v, READY, hpath, npath, Some(kpath), &[])?),
            Conv::Sandbox => Val::Table(self.sandbox(v, hpath, npath, Some(kpath))?),
            Conv::Snapshot | Conv::Outputs => {
                let spec = if conv == Conv::Snapshot {
                    SNAPSHOT
                } else {
                    OUTPUT
                };
                let mut items = Vec::new();
                for (i, item) in iterate(v).ok_or(Refused)?.into_iter().enumerate() {
                    if item.as_table().is_some() {
                        let t = self.table(
                            &item,
                            spec,
                            &kidx(hpath, i),
                            &kidx(npath, i),
                            Some(&kidx(kpath, i)),
                            &[],
                        )?;
                        items.push(Val::Table(t));
                    } else {
                        items.push(item);
                    }
                }
                Val::List(items)
            }
            Conv::Authors => Val::List(iterate(v).ok_or(Refused)?.iter().map(author).collect()),
        })
    }

    fn table(
        &mut self,
        src: &Val,
        spec: &[Spec],
        hpath: &[Seg],
        npath: &[Seg],
        kpath: Option<&[Seg]>,
        skip: &[&str],
    ) -> Result<Table, Refused> {
        let Val::Table(src) = src else {
            return Err(Refused);
        };
        let base = kpath.unwrap_or(hpath).to_vec();
        let mut out = Table::new();
        for s in spec {
            let Some(v) = src.get(s.harbor) else { continue };
            let hp = kkey(hpath, s.harbor);
            let np = kkey(npath, s.native);
            let kk = kkey(&base, s.harbor);
            match self.convert(s.to_native, v, &hp, &np, &kk) {
                Ok(x) => {
                    out.insert(s.native.to_string(), x);
                    self.moves.insert(hp, np);
                }
                Err(Refused) => {
                    self.stash(&hp, v.clone(), Some(&kk));
                    self.notes.push(format!(
                        "kept {} as written: its value is not in the form the split layout documents",
                        dotted(&hp)
                    ));
                }
            }
        }
        for (h, v) in src {
            if !spec.iter().any(|s| s.harbor == h) && !skip.contains(&h.as_str()) {
                self.stash(&kkey(hpath, h), v.clone(), Some(&kkey(&base, h)));
            }
        }
        Ok(out)
    }

    fn network(
        &mut self,
        src: &Table,
        hpath: &[Seg],
        npath: &[Seg],
        out: &mut Table,
        kpath: Option<&[Seg]>,
        default: Option<&str>,
    ) {
        let base = kpath.unwrap_or(hpath).to_vec();
        let mode = src
            .get("network_mode")
            .cloned()
            .or_else(|| default.map(|d| Val::Str(d.into())));
        let Some(mode) = mode else {
            if let Some(h) = src.get("allowed_hosts") {
                self.stash(
                    &kkey(hpath, "allowed_hosts"),
                    h.clone(),
                    Some(&kkey(&base, "allowed_hosts")),
                );
            }
            return;
        };
        match network(&mode, src.get("allowed_hosts")) {
            Some(n) => {
                out.insert("network".into(), n);
                for h in PHASE_NETWORK {
                    self.moves.insert(kkey(hpath, h), kkey(npath, "network"));
                }
            }
            None => {
                for h in PHASE_NETWORK {
                    if let Some(v) = src.get(h) {
                        self.stash(&kkey(hpath, h), v.clone(), Some(&kkey(&base, h)));
                    }
                }
                self.notes.push(format!(
                    "kept {}.network_mode = {} as written: the split layout's values are no-network, public, and allowlist",
                    dotted(hpath),
                    py_repr(&mode)
                ));
            }
        }
    }

    fn sandbox(
        &mut self,
        src: &Val,
        hpath: &[Seg],
        npath: &[Seg],
        kpath: Option<&[Seg]>,
    ) -> Result<Table, Refused> {
        let Val::Table(src) = src else {
            return Err(Refused);
        };
        let mut src = src.clone();
        let base = kpath.unwrap_or(hpath).to_vec();
        for (old, new, native) in [
            ("memory", "memory_mb", "memory"),
            ("storage", "storage_mb", "disk"),
        ] {
            let Some(v) = src.shift_remove(old) else {
                continue;
            };
            match &v {
                Val::Str(text) => {
                    let mb = harbor_size_mb(text).ok_or(Refused)?;
                    src.entry(new.to_string()).or_insert(Val::Int(mb));
                    self.moves.insert(kkey(hpath, old), kkey(npath, native));
                    let shown = size(&src[new]).ok_or(Refused)?;
                    self.notes.push(format!(
                        "{} is a deprecated split-layout spelling; imported as {shown}",
                        dotted(&kkey(hpath, old))
                    ));
                }
                _ => self.stash(&kkey(hpath, old), v.clone(), Some(&kkey(&base, old))),
            }
        }
        if let Some(allow) = src.shift_remove("allow_internet") {
            let has_hosts = src.get("allowed_hosts").is_some();
            if src.contains_key("network_mode") || has_hosts {
                self.stash(
                    &kkey(hpath, "allow_internet"),
                    allow,
                    Some(&kkey(&base, "allow_internet")),
                );
            } else {
                let mode = if allow.truthy() {
                    "public"
                } else {
                    "no-network"
                };
                src.insert("network_mode".into(), Val::Str(mode.into()));
                self.moves
                    .insert(kkey(hpath, "allow_internet"), kkey(npath, "network"));
                let shown = if allow.truthy() { "'open'" } else { "'none'" };
                self.notes.push(format!(
                    "{}.allow_internet is a deprecated split-layout spelling; imported as network = {shown}",
                    dotted(hpath)
                ));
            }
        }
        let mut out = self.table(
            &Val::Table(src.clone()),
            SANDBOX,
            hpath,
            npath,
            kpath,
            &PHASE_NETWORK,
        )?;
        if src.contains_key("network_mode") || src.contains_key("allowed_hosts") {
            self.network(&src, hpath, npath, &mut out, kpath, Some("public"));
        }
        let first: Vec<&str> = SANDBOX[..8]
            .iter()
            .map(|s| s.native)
            .chain(["network"])
            .collect();
        Ok(ordered(&out, &first))
    }

    fn phase(
        &mut self,
        src: &Val,
        spec: &[Spec],
        hpath: &[Seg],
        npath: &[Seg],
        kpath: Option<&[Seg]>,
    ) -> Result<Table, Refused> {
        let mut out = self.table(src, spec, hpath, npath, kpath, &PHASE_NETWORK)?;
        if let Val::Table(t) = src {
            self.network(t, hpath, npath, &mut out, kpath, None);
        }
        Ok(out)
    }
}

/// The cloud metadata opt-out as the split reader reads it (`lax::boolean`:
/// `"yes"`, `1`, `"off"`, ...), as the boolean task.md requires (F1-B-r2
/// L3). A value the split reader refuses is kept, so task.md refuses it
/// too, by name.
fn opt_out_value(v: &Val) -> Val {
    let json = match v {
        Val::Bool(b) => serde_json::Value::Bool(*b),
        Val::Int(i) => serde_json::Value::from(*i),
        Val::Float(f) => serde_json::Value::from(*f),
        Val::Str(s) => serde_json::Value::String(s.clone()),
        _ => return v.clone(),
    };
    crate::split::lax::boolean(&json).map_or_else(|_| v.clone(), Val::Bool)
}

/// The result of mapping a Harbor config: the task.md config, where each
/// Harbor path went, and notes.
pub(crate) struct Imported {
    pub config: Table,
    pub moves: HashMap<KPath, KPath>,
    pub notes: Vec<String>,
}

/// Harbor `task.toml` data to task.md config. `Err` names a value Harbor's
/// own parser would refuse in a way convert.py does not survive.
pub(crate) fn config_from_harbor(cfg: &Table, stash: &str) -> Result<Imported, String> {
    let mut im = Import::new(stash);
    let mut out = Table::new();
    let is_table = |k: &str| matches!(cfg.get(k), Some(Val::Table(_)));
    let tables: Vec<&str> = [
        "task",
        "metadata",
        "environment",
        "agent",
        "verifier",
        "solution",
    ]
    .into_iter()
    .filter(|k| is_table(k))
    .collect();
    let has = |k: &str| tables.contains(&k);
    let refused = |what: &str| {
        format!("cannot import [{what}]: a split-layout runner would refuse its values")
    };
    if has("task") {
        let t = im
            .table(&cfg["task"], PACKAGE, &kp(&["task"]), &[], None, &[])
            .map_err(|_| refused("task"))?;
        out.extend(t);
        im.moves.insert(kp(&["task"]), Vec::new());
    }
    let mut opt_out = None;
    if has("metadata") {
        let mut about = cfg["metadata"].clone();
        if let Val::Table(t) = &mut about {
            opt_out = t.shift_remove(METADATA_OPT_OUT);
        }
        if !matches!(&about, Val::Table(t) if t.is_empty()) {
            out.insert("about".into(), about);
        }
        im.moves.insert(kp(&["metadata"]), kp(&["about"]));
        if opt_out.is_some() {
            im.moves.insert(
                kp(&["metadata", METADATA_OPT_OUT]),
                kp(&["sandbox", METADATA_OPT_OUT]),
            );
        }
    }
    let mut sandbox = if has("environment") {
        im.sandbox(
            &cfg["environment"],
            &kp(&["environment"]),
            &kp(&["sandbox"]),
            None,
        )
        .map_err(|_| refused("environment"))?
    } else {
        Table::new()
    };
    if let Some(artifacts) = cfg.get("artifacts") {
        let mut wrapper = Table::new();
        wrapper.insert("artifacts".into(), artifacts.clone());
        let outputs = im
            .table(
                &Val::Table(wrapper),
                &STEP[2..3],
                &[],
                &kp(&["sandbox"]),
                None,
                &[],
            )
            .map_err(|_| refused("artifacts"))?;
        if let Some(o) = outputs.get("outputs") {
            sandbox.insert("outputs".into(), o.clone());
        }
    }
    if let Some(v) = opt_out {
        let read = opt_out_value(&v);
        if read != v {
            im.notes.push(format!(
                "metadata.{METADATA_OPT_OUT} = {} imported as [sandbox] {METADATA_OPT_OUT} = {}",
                py_repr(&v),
                read.truthy()
            ));
        }
        sandbox.insert(METADATA_OPT_OUT.into(), read);
    }
    if has("environment") || !sandbox.is_empty() {
        out.insert("sandbox".into(), Val::Table(sandbox));
        im.moves.insert(kp(&["environment"]), kp(&["sandbox"]));
    }
    if has("agent") {
        let a = im
            .phase(&cfg["agent"], AGENT, &kp(&["agent"]), &kp(&["agent"]), None)
            .map_err(|_| refused("agent"))?;
        out.insert("agent".into(), Val::Table(a));
        im.moves.insert(kp(&["agent"]), kp(&["agent"]));
    }
    let mut verifier = if has("verifier") {
        im.phase(
            &cfg["verifier"],
            VERIFIER,
            &kp(&["verifier"]),
            &kp(&["verifier"]),
            None,
        )
        .map_err(|_| refused("verifier"))?
    } else {
        Table::new()
    };
    if let Some(m) = cfg.get("multi_step_reward_strategy") {
        verifier.insert("combine_stages".into(), m.clone());
        im.moves.insert(
            kp(&["multi_step_reward_strategy"]),
            kp(&["verifier", "combine_stages"]),
        );
    }
    if has("verifier") || !verifier.is_empty() {
        out.insert("verifier".into(), Val::Table(verifier));
        im.moves.insert(kp(&["verifier"]), kp(&["verifier"]));
    }
    if has("solution") {
        let o = im
            .table(
                &cfg["solution"],
                ENV_ONLY,
                &kp(&["solution"]),
                &kp(&["oracle"]),
                None,
                &[],
            )
            .map_err(|_| refused("solution"))?;
        out.insert("oracle".into(), Val::Table(o));
        im.moves.insert(kp(&["solution"]), kp(&["oracle"]));
    }
    let mut handled: Vec<&str> = tables.clone();
    handled.extend(["artifacts", "multi_step_reward_strategy"]);
    let steps: Option<Vec<(String, &Table)>> = match cfg.get("steps") {
        Some(Val::List(items)) if !items.is_empty() => items
            .iter()
            .map(|st| match st {
                Val::Table(t) => match t.get("name") {
                    Some(Val::Str(n)) => Some((n.clone(), t)),
                    _ => None,
                },
                _ => None,
            })
            .collect(),
        _ => None,
    };
    if let Some(steps) = steps {
        handled.push("steps");
        let mut stages = Table::new();
        for (i, (name, step)) in steps.iter().enumerate() {
            let hp: KPath = vec![Seg::Key("steps".into()), Seg::Index(i)];
            let np = kp(&["stages", name]);
            let kpth = kp(&["steps", name]);
            let mut st = Table::new();
            let unlock = if i == 0 {
                "at_start".to_string()
            } else {
                format!("after:{}", steps[i - 1].0)
            };
            st.insert("unlock".into(), Val::Str(unlock));
            for (part, spec) in [("agent", AGENT), ("verifier", VERIFIER)] {
                if let Some(v @ Val::Table(_)) = step.get(part) {
                    let t = im
                        .phase(
                            v,
                            spec,
                            &kkey(&hp, part),
                            &kkey(&np, part),
                            Some(&kkey(&kpth, part)),
                        )
                        .map_err(|_| refused("steps"))?;
                    st.insert(part.into(), Val::Table(t));
                    im.moves.insert(kkey(&hp, part), kkey(&np, part));
                }
            }
            let rest: Table = step
                .iter()
                .filter(|(k, v)| {
                    *k != "name"
                        && !((*k == "agent" || *k == "verifier") && matches!(v, Val::Table(_)))
                })
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            let t = im
                .table(&Val::Table(rest), STEP, &hp, &np, Some(&kpth), &[])
                .map_err(|_| refused("steps"))?;
            st.extend(t);
            im.moves.insert(kkey(&hp, "name"), np.clone());
            im.moves.insert(hp.clone(), np.clone());
            stages.insert(name.clone(), Val::Table(st));
        }
        out.insert("stages".into(), Val::Table(stages));
        let strategy = cfg.get("multi_step_reward_strategy");
        if strategy.is_none() || strategy == Some(&Val::Str("mean".into())) {
            let v = out
                .entry("verifier".to_string())
                .or_insert_with(|| Val::Table(Table::new()));
            if let Val::Table(t) = v {
                t.insert("unreached_stages".into(), Val::Str("exclude".into()));
            }
        }
    }
    if let Some(source) = cfg.get("source") {
        handled.push("source");
        let mut prov = Table::new();
        prov.insert("source".into(), source.clone());
        out.insert("provenance".into(), Val::Table(prov));
        im.moves
            .insert(kp(&["source"]), kp(&["provenance", "source"]));
    }
    for (k, v) in cfg {
        if !handled.contains(&k.as_str()) {
            im.stash(&kp(&[k]), v.clone(), None);
        }
    }
    if !im.keep.is_empty() {
        let imp = out
            .entry("import".to_string())
            .or_insert_with(|| Val::Table(Table::new()));
        if let Val::Table(t) = imp {
            t.insert(stash.to_string(), Val::Table(im.keep.clone()));
        }
    }
    let order = [
        "name",
        "title",
        "version",
        "description",
        "authors",
        "keywords",
        "about",
        "agent",
        "sandbox",
        "verifier",
        "oracle",
        "stages",
        "provenance",
        "import",
    ];
    Ok(Imported {
        config: ordered(&out, &order),
        moves: im.moves,
        notes: im.notes,
    })
}

/// How much a loss matters (task.md's three classes of keys).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum LossSeverity {
    /// The task would run differently: export refuses unless told to degrade.
    RunAffecting,
    /// A rule checked only when a result claims to be scored.
    Scoring,
    /// Information a runtime may ignore.
    Metadata,
}

/// Something an export cannot carry.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Loss {
    /// The setting or file, as the report names it.
    pub path: String,
    pub reason: String,
    pub severity: LossSeverity,
    /// The loss as task.md's converter returns it: `path`, often with a
    /// note on what happens instead. Commands print `path: reason`.
    #[serde(skip)]
    pub text: String,
}

impl Loss {
    fn new(
        text: impl Into<String>,
        severity: LossSeverity,
        reason: impl Into<String>,
        path: Option<String>,
    ) -> Self {
        let text = text.into();
        Self {
            path: path.unwrap_or_else(|| text.clone()),
            reason: reason.into(),
            severity,
            text,
        }
    }
}

const NO_EQUIVALENT: &str = "the split layout has no equivalent";

fn table_loss(table: &str) -> Option<(LossSeverity, &'static str)> {
    use LossSeverity::*;
    Some(match table {
        "about" => (Metadata, "kept only as [metadata]"),
        "training" => (Metadata, "the split layout has no training metadata"),
        "credits" => (Metadata, "the split layout has no credits"),
        "provenance" => (
            Metadata,
            "the split layout keeps only [provenance] source, as its top-level source",
        ),
        "runs" => (
            Scoring,
            "the split layout has no run policy, so a scored run's rules go unchecked",
        ),
        "integrity" => (
            Scoring,
            "the split layout checks no controls or integrity thresholds",
        ),
        "world" => (
            RunAffecting,
            "the split layout has no worlds: the agent would get a container, not the world the task declares",
        ),
        "roles" => (RunAffecting, "a split-layout runner runs one agent"),
        "user" => (RunAffecting, "the split layout has no simulated user"),
        "interaction" => (
            RunAffecting,
            "a split-layout runner runs one agent with no simulated user",
        ),
        "family" => (
            RunAffecting,
            "the split layout has no generated instances: one fixed folder is all it runs",
        ),
        "variants" => (RunAffecting, "the split layout has no variants"),
        "matrix" => (RunAffecting, "the split layout has no run matrix"),
        "tiers" => (RunAffecting, "the split layout has no tiers"),
        "conventions" => (RunAffecting, "the split layout has no conventions table"),
        "trajectory" => (
            RunAffecting,
            "the split layout records no trajectory requirements",
        ),
        "preference" => (RunAffecting, "the split layout has no preference judging"),
        _ => return None,
    })
}

fn key_loss(where_: &str, key: &str) -> Loss {
    use LossSeverity::*;
    let table = where_.rsplit('.').next().unwrap_or(where_);
    if key.starts_with("x-") {
        return Loss::new(
            format!("[{where_}] {key}"),
            Metadata,
            "an extension key, which a split-layout runner ignores",
            None,
        );
    }
    let (severity, reason) = match (table, key) {
        ("sandbox", "clock") => (
            RunAffecting,
            "the split layout has no clock, so it would run the task on real time",
        ),
        ("sandbox", "timezone") => (
            RunAffecting,
            "the split layout has no time zone setting, so the task would run in the image's zone",
        ),
        ("sandbox", "mounts") => (
            RunAffecting,
            "the split layout mounts no package folders into the sandbox",
        ),
        ("sandbox", "boundary") => (
            RunAffecting,
            "a split-layout runner does not take an isolation boundary from the task",
        ),
        ("agent", "network_reason") => (
            Metadata,
            "the split layout keeps no reason for a network setting",
        ),
        ("verifier", "human") => (RunAffecting, "the split layout has no human assessors"),
        _ => (RunAffecting, NO_EQUIVALENT),
    };
    Loss::new(format!("[{where_}] {key}"), severity, reason, None)
}

/// The keys task.md defines in the tables Harbor also has.
fn native_keys(table: &str) -> Vec<&'static str> {
    match table {
        "sandbox" => SANDBOX
            .iter()
            .map(|s| s.native)
            .chain(["network", "outputs", "compose", "services"])
            .collect(),
        "agent" => AGENT.iter().map(|s| s.native).chain(["network"]).collect(),
        "verifier" => VERIFIER
            .iter()
            .map(|s| s.native)
            .chain(["network", "combine_stages", "unreached_stages", "mount"])
            .collect(),
        "oracle" => vec!["env", "mount"],
        _ => Vec::new(),
    }
}

const TOP: [&str; 14] = [
    "name",
    "title",
    "version",
    "description",
    "authors",
    "keywords",
    "about",
    "sandbox",
    "agent",
    "verifier",
    "oracle",
    "stages",
    "provenance",
    "import",
];
/// Where Harbor's scripts expect the verifier and the oracle.
pub(crate) const HARBOR_MOUNTS: [(&str, &str); 2] =
    [("verifier", "/tests"), ("oracle", "/solution")];
/// Where `environment/docker-compose.yaml` lands.
pub(crate) const HARBOR_COMPOSE: &str = "sandbox/docker-compose.yaml";

fn harbor_mount(part: &str) -> &'static str {
    HARBOR_MOUNTS
        .iter()
        .find(|(p, _)| *p == part)
        .map_or("", |(_, m)| m)
}

/// task.md config to Harbor `task.toml` data, recording where each task.md
/// path went and what Harbor cannot hold.
struct Export {
    moves: HashMap<KPath, KPath>,
    losses: Vec<Loss>,
}

/// A task.md value export cannot convert (a malformed config the checker
/// refuses too).
fn bad(path: &[Seg]) -> String {
    format!(
        "cannot export [{}]: its value is not in the form task.md documents",
        dotted(path)
    )
}

impl Export {
    fn convert(
        &mut self,
        conv: Conv,
        v: &Val,
        npath: &[Seg],
        hpath: &[Seg],
    ) -> Result<Val, String> {
        Ok(match conv {
            Conv::Same | Conv::Duration | Conv::Size => v.clone(),
            Conv::Seconds => Val::Float(seconds(v).ok_or_else(|| bad(npath))?),
            Conv::Megabytes => Val::Int(megabytes(v).ok_or_else(|| bad(npath))?),
            Conv::Ready => Val::Table(self.table(v, READY, npath, hpath, true)?),
            Conv::Sandbox => {
                let t = self.sandbox(v, npath, hpath)?;
                if let Val::Table(src) = v {
                    self.unknown(src, &native_keys("sandbox"), &dotted(npath));
                }
                Val::Table(t)
            }
            Conv::Snapshot | Conv::Outputs => {
                let spec = if conv == Conv::Snapshot {
                    SNAPSHOT
                } else {
                    OUTPUT
                };
                let items = v.as_list().ok_or_else(|| bad(npath))?;
                let mut out = Vec::new();
                for (i, x) in items.iter().enumerate() {
                    if x.as_table().is_some() {
                        out.push(Val::Table(self.table(
                            x,
                            spec,
                            &kidx(npath, i),
                            &kidx(hpath, i),
                            true,
                        )?));
                    } else {
                        out.push(x.clone());
                    }
                }
                Val::List(out)
            }
            Conv::Authors => Val::List(
                v.as_list()
                    .ok_or_else(|| bad(npath))?
                    .iter()
                    .map(author_harbor)
                    .collect(),
            ),
        })
    }

    fn table(
        &mut self,
        src: &Val,
        spec: &[Spec],
        npath: &[Seg],
        hpath: &[Seg],
        strict: bool,
    ) -> Result<Table, String> {
        let Val::Table(src) = src else {
            return Err(bad(npath));
        };
        if strict {
            let known: Vec<&str> = spec.iter().map(|s| s.native).collect();
            self.unknown(src, &known, &dotted(npath));
        }
        let mut out = Table::new();
        for s in spec {
            let Some(v) = src.get(s.native) else { continue };
            let np = kkey(npath, s.native);
            let hp = kkey(hpath, s.harbor);
            let x = self.convert(s.to_harbor, v, &np, &hp)?;
            out.insert(s.harbor.to_string(), x);
            self.moves.insert(np, hp);
        }
        Ok(out)
    }

    fn network(
        &mut self,
        src: &Table,
        npath: &[Seg],
        hpath: &[Seg],
        out: &mut Table,
    ) -> Result<(), String> {
        let Some(v) = src.get("network") else {
            return Ok(());
        };
        let (mode, hosts) = match v {
            Val::Str(s) if s == "none" => ("no-network", None),
            Val::Str(s) if s == "open" => ("public", None),
            Val::List(_) => ("allowlist", Some(v.clone())),
            Val::Table(t) if t.contains_key("block") => {
                let at = dotted(npath);
                self.losses.push(Loss::new(
                    format!("the network block list in [{at}]"),
                    LossSeverity::RunAffecting,
                    "the split layout has no network that is open except for listed hosts",
                    Some(format!("[{at}] network")),
                ));
                return Ok(());
            }
            _ => {
                return Err(format!(
                    "cannot export [{}] network: not a network setting",
                    dotted(npath)
                ));
            }
        };
        out.insert("network_mode".into(), Val::Str(mode.into()));
        if let Some(h) = hosts {
            out.insert("allowed_hosts".into(), h);
        }
        self.moves
            .insert(kkey(npath, "network"), kkey(hpath, "network_mode"));
        Ok(())
    }

    fn sandbox(&mut self, src: &Val, npath: &[Seg], hpath: &[Seg]) -> Result<Table, String> {
        let mut out = self.table(src, SANDBOX, npath, hpath, false)?;
        if let Val::Table(t) = src {
            self.network(t, npath, hpath, &mut out)?;
        }
        Ok(out)
    }

    fn phase(
        &mut self,
        src: &Val,
        spec: &[Spec],
        npath: &[Seg],
        hpath: &[Seg],
    ) -> Result<Table, String> {
        let mut out = self.table(src, spec, npath, hpath, false)?;
        if let Val::Table(t) = src {
            self.network(t, npath, hpath, &mut out)?;
        }
        Ok(out)
    }

    fn unknown(&mut self, src: &Table, known: &[&str], where_: &str) {
        for (k, v) in src {
            if known.contains(&k.as_str()) {
                continue;
            }
            if k == "judges"
                && where_.rsplit('.').next() == Some("verifier")
                && let Val::Table(judges) = v
            {
                for (role, rv) in judges {
                    let names: Vec<String> = match rv {
                        Val::Table(t) => t
                            .keys()
                            .map(|key| format!("[{where_}.judges.{role}] {key}"))
                            .collect(),
                        _ => vec![format!("[{where_}.judges] {role}")],
                    };
                    for name in names {
                        self.losses.push(Loss::new(
                            name,
                            LossSeverity::RunAffecting,
                            "a split-layout runner runs no task.md judges: its reward comes from test.sh alone",
                            None,
                        ));
                    }
                }
                continue;
            }
            self.losses.push(key_loss(where_, k));
        }
    }
}

/// The result of mapping a task.md config to Harbor's.
pub(crate) struct Exported {
    pub config: Table,
    pub moves: HashMap<KPath, KPath>,
    pub losses: Vec<Loss>,
}

/// A `[sandbox] skills` value that names a folder in the package, such as
/// `sandbox/skills`, rather than a path in the sandbox.
pub(crate) fn package_folder(skills: Option<&Val>) -> bool {
    let Some(Val::Str(s)) = skills else {
        return false;
    };
    let first = s.split('/').next().unwrap_or("");
    (first == "sandbox" || first == "world") && s.trim_matches('/').contains('/')
}

/// `[integrity]` that holds only cheat oracles in `cheat/`, which
/// Terminal-Bench 3's Harbor layout already declares.
fn harbor_cheats(integrity: &Val) -> bool {
    let Val::Table(i) = integrity else {
        return false;
    };
    if i.len() != 1 {
        return false;
    }
    let Some(Val::Table(c)) = i.get("controls") else {
        return false;
    };
    if c.len() != 1 {
        return false;
    }
    let Some(Val::List(cheats)) = c.get("cheats") else {
        return false;
    };
    !cheats.is_empty()
        && cheats
            .iter()
            .all(|x| x.as_str().is_some_and(|s| s.starts_with("cheat/")))
}

/// Whether stages unlock one after another, as Harbor's steps do.
fn is_step_chain(stages: &Table) -> bool {
    let names: Vec<&String> = stages.keys().collect();
    stages.values().enumerate().all(|(i, st)| {
        let want = if i == 0 {
            "at_start".to_string()
        } else {
            format!("after:{}", names[i - 1])
        };
        matches!(st, Val::Table(t) if t.get("unlock") == Some(&Val::Str(want)))
    })
}

/// An `[import.harbor]` entry import wrote for a Harbor file, rather than a
/// task.toml key it kept.
pub(crate) fn harbor_record(key: &str, value: Option<&Val>) -> bool {
    let Some(value) = value else { return false };
    match key {
        "trajectory" => value.as_str().is_some_and(|s| s.rsplit('/').next() == Some("trajectory.json")),
        "workdirs" => match value {
            Val::Table(t) => {
                !t.is_empty() && t.iter().all(|(n, v)| v.as_str() == Some(format!("stages/{n}/workdir").as_str()))
            }
            _ => false,
        },
        "originals" => match value {
            Val::Table(t) => {
                !t.is_empty()
                    && t.values().all(|v| matches!(v, Val::Table(r) if r.contains_key("original") && r.contains_key("converted")))
            }
            _ => false,
        },
        "rewardkit" => match value {
            Val::Table(t) => matches!(t.get("scripts"), Some(Val::List(_))) && matches!(t.get("judges"), Some(Val::List(_))),
            _ => false,
        },
        _ => false,
    }
}

static PROVIDER_KEY: LazyLock<Regex> = LazyLock::new(|| {
    let providers = "OPENAI|AZURE_OPENAI|AZURE_AI|AZURE|ANTHROPIC|CLAUDE|OPENROUTER|GEMINI|GOOGLE|VERTEXAI|VERTEX|MISTRAL|CODESTRAL|\
DEEPSEEK|GROQ|TOGETHERAI|TOGETHER_AI|TOGETHER|FIREWORKS_AI|FIREWORKS|XAI|COHERE|CO|HUGGINGFACE|HF|REPLICATE|\
PERPLEXITYAI|PERPLEXITY|CEREBRAS|SAMBANOVA|NVIDIA_NIM|NVIDIA|DASHSCOPE|MOONSHOT|ZAI|ZHIPUAI|MINIMAX|BASETEN|\
DEEPINFRA|NOVITA|NEBIUS|HYPERBOLIC|AI21|VOYAGE|LITELLM|PORTKEY|BEDROCK|WATSONX|DATABRICKS|GOOGLE_GENERATIVE_AI|\
LLM|MODEL|JUDGE";
    Regex::new(&format!(
        r"^(?:[A-Z0-9]+_)*(?:{providers})_(?:API_KEY|APIKEY|API_TOKEN|AUTH_TOKEN|KEY|TOKEN|BASE_URL|API_BASE)$|^(?:AWS_BEARER_TOKEN_BEDROCK|HF_TOKEN|HUGGING_FACE_HUB_TOKEN)$"
    ))
    .unwrap()
});
static PROVIDER_VAR: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?:ANTHROPIC|OPENAI|AZURE_OPENAI|AZURE|OPENROUTER|GEMINI|GOOGLE|GOOGLE_GENERATIVE_AI|VERTEXAI|MISTRAL|COHERE|CO|GROQ|TOGETHER|TOGETHERAI|DEEPSEEK|XAI|FIREWORKS|FIREWORKS_AI|PERPLEXITY|PERPLEXITYAI|BASETEN|DEEPINFRA|CEREBRAS|HUGGINGFACE|HF|LITELLM|MOONSHOT|DASHSCOPE|ZHIPUAI|NVIDIA|NVIDIA_NIM|SAMBANOVA|NEBIUS|NOVITA|HYPERBOLIC|REPLICATE|AI21|VOYAGE)_(?:API_KEY|AUTH_TOKEN|BASE_URL|API_BASE)$|^HF_TOKEN$|^AWS_BEARER_TOKEN_BEDROCK$",
    )
    .unwrap()
});
static ENV_REF: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\$\{?([A-Za-z_]\w*)").unwrap());

/// A model provider's key or endpoint variable, such as `OPENAI_API_KEY`.
pub(crate) fn is_provider_var(name: &str) -> bool {
    PROVIDER_KEY.is_match(name) || PROVIDER_VAR.is_match(name)
}

/// The variables of an env table that carry a model provider's key or
/// endpoint, by their own name or by the host variable their value names.
pub(crate) fn provider_vars(env: &Val) -> Vec<String> {
    let Val::Table(env) = env else {
        return Vec::new();
    };
    env.iter()
        .filter(|(k, v)| {
            let text = match v {
                Val::Str(s) => s.clone(),
                other => py_str(other),
            };
            is_provider_var(k) || ENV_REF.captures_iter(&text).any(|c| is_provider_var(&c[1]))
        })
        .map(|(k, _)| k.clone())
        .collect()
}

/// Python's `str()` of a value.
fn py_str(v: &Val) -> String {
    match v {
        Val::Str(s) | Val::Date(s) => s.clone(),
        other => py_repr(other),
    }
}

/// The env tables a task.md config gives its verifier scripts: `[verifier]
/// env`, `[verifier.sandbox] env`, and each stage's, as (path, table).
pub(crate) fn verifier_envs(cfg: &Table) -> Vec<(String, Val)> {
    let mut places: Vec<(String, Option<&Val>)> = vec![("verifier".into(), cfg.get("verifier"))];
    if let Some(Val::Table(stages)) = cfg.get("stages") {
        for (n, st) in stages {
            let v = match st {
                Val::Table(t) => t.get("verifier"),
                _ => None,
            };
            places.push((format!("stages.{n}.verifier"), v));
        }
    }
    let mut out = Vec::new();
    for (where_, ver) in places {
        let Some(Val::Table(ver)) = ver else { continue };
        if let Some(env @ Val::Table(_)) = ver.get("env") {
            out.push((format!("{where_}.env"), env.clone()));
        }
        if let Some(Val::Table(sb)) = ver.get("sandbox")
            && let Some(env @ Val::Table(_)) = sb.get("env")
        {
            out.push((format!("{where_}.sandbox.env"), env.clone()));
        }
    }
    out
}

/// Whether import recorded RewardKit in the task.
pub(crate) fn rk_recorded(cfg: &Table) -> bool {
    let kept = match cfg.get("import") {
        Some(Val::Table(i)) => i.get(crate::taskmd::split_record(|k| i.contains_key(k))),
        _ => None,
    };
    match kept {
        Some(Val::Table(k)) => harbor_record("rewardkit", k.get("rewardkit")),
        _ => false,
    }
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

/// task.md config to Harbor `task.toml` data. `[import.<stash>]` is merged
/// back; tables in `extra_top` are the caller's to handle and are no loss.
pub(crate) fn config_to_harbor(
    cfg: &Table,
    stash: &str,
    extra_top: &[&str],
) -> Result<Exported, String> {
    let mut ex = Export {
        moves: HashMap::new(),
        losses: Vec::new(),
    };
    ex.moves.insert(Vec::new(), Vec::new());
    let mut out = Table::new();
    let package = ex.table(
        &Val::Table(cfg.clone()),
        PACKAGE,
        &[],
        &kp(&["task"]),
        false,
    )?;
    if !package.is_empty() && !package.contains_key("name") {
        ex.losses.push(Loss::new(
            "version, description, authors, and keywords (the split layout's [task] table needs a name)",
            LossSeverity::Metadata,
            "the split layout's [task] table needs a name",
            Some("version, description, authors, keywords".into()),
        ));
    } else if !package.is_empty() {
        out.insert("task".into(), Val::Table(package));
    }
    if cfg.contains_key("title") {
        ex.losses.push(Loss::new(
            "title",
            LossSeverity::Metadata,
            "the split layout's [task] table has no title",
            None,
        ));
    }
    if let Some(Val::Table(about)) = cfg.get("about") {
        // task.md reads the cloud metadata opt-out only in [sandbox], so
        // under [about] it opts nothing out; copied into [metadata], where
        // the split reader honours it, it would (F1-A-r2).
        let mut about = about.clone();
        if let Some(v) = about.shift_remove(METADATA_OPT_OUT) {
            ex.losses.push(Loss::new(
                format!(
                    "[about] {METADATA_OPT_OUT} = {} (not copied: task.md reads it only in [sandbox], and in [metadata] it would opt the task out)",
                    py_repr(&v)
                ),
                LossSeverity::Metadata,
                "task.md reads the opt-out only in [sandbox], so it opted nothing out; in the split layout's [metadata] it would opt the task out",
                Some(format!("[about] {METADATA_OPT_OUT}")),
            ));
        }
        if !about.is_empty() || cfg["about"].as_table().is_some_and(Table::is_empty) {
            out.insert("metadata".into(), Val::Table(about));
            ex.moves.insert(kp(&["about"]), kp(&["metadata"]));
        }
    }
    let mut sb: Option<Table> = cfg.get("sandbox").and_then(Val::as_table).cloned();
    if let Some(v) = sb.as_mut().and_then(|s| s.shift_remove(METADATA_OPT_OUT)) {
        let metadata = out
            .entry("metadata".to_string())
            .or_insert_with(|| Val::Table(Table::new()));
        if let Val::Table(m) = metadata {
            m.insert(METADATA_OPT_OUT.into(), v);
        }
        ex.moves.insert(
            kp(&["sandbox", METADATA_OPT_OUT]),
            kp(&["metadata", METADATA_OPT_OUT]),
        );
    }
    if let Some(s) = sb.as_mut() {
        if package_folder(s.get("skills")) {
            let shown = py_repr(&s["skills"]);
            ex.losses.push(Loss::new(
                format!("[sandbox] skills = {shown} (a folder in the package; the split layout's skills_dir is a path in the environment, so the image must copy the folder there)"),
                LossSeverity::RunAffecting,
                "the split layout's skills_dir is a path in the environment, so the agent gets no skills unless the image copies the folder there",
                Some("[sandbox] skills".into()),
            ));
            s.shift_remove("skills");
        }
        if s.contains_key("compose") || s.contains_key("services") {
            if let Some(c) = s.get("compose")
                && c != &Val::Str(HARBOR_COMPOSE.into())
            {
                ex.losses.push(Loss::new(
                        format!("[sandbox] compose = {} (a split-layout runner reads services only from environment/docker-compose.yaml)", py_repr(c)),
                        LossSeverity::RunAffecting,
                        "a split-layout runner reads services only from environment/docker-compose.yaml",
                        Some("[sandbox] compose".into()),
                    ));
            }
            if s.get("services").is_some_and(Val::truthy) {
                ex.losses.push(Loss::new(
                    "[[sandbox.services]] (the split layout declares services only in environment/docker-compose.yaml)",
                    LossSeverity::RunAffecting,
                    "the split layout declares services only in environment/docker-compose.yaml",
                    Some("[[sandbox.services]]".into()),
                ));
            }
            s.shift_remove("compose");
            s.shift_remove("services");
        }
    }
    let original_sandbox = cfg.get("sandbox").and_then(Val::as_table);
    if let Some(s) = &sb {
        let only_services = original_sandbox.is_none_or(|o| {
            o.keys()
                .all(|k| k == "compose" || k == "services" || k == METADATA_OPT_OUT)
        });
        if !s.is_empty() || (original_sandbox.is_some_and(|o| !o.is_empty()) && !only_services) {
            let env = ex.sandbox(
                &Val::Table(s.clone()),
                &kp(&["sandbox"]),
                &kp(&["environment"]),
            )?;
            out.insert("environment".into(), Val::Table(env));
            ex.moves.insert(kp(&["sandbox"]), kp(&["environment"]));
            if s.contains_key("outputs") {
                let t = ex.table(
                    &Val::Table(s.clone()),
                    &STEP[2..3],
                    &kp(&["sandbox"]),
                    &[],
                    false,
                )?;
                if let Some(a) = t.get("artifacts") {
                    out.insert("artifacts".into(), a.clone());
                }
            }
            ex.unknown(s, &native_keys("sandbox"), "sandbox");
        }
    }
    let ver: Table = cfg
        .get("verifier")
        .and_then(Val::as_table)
        .cloned()
        .unwrap_or_default();
    let oracle_mount = cfg
        .get("oracle")
        .and_then(Val::as_table)
        .and_then(|o| o.get("mount"));
    for (part, mount) in [("verifier", ver.get("mount")), ("oracle", oracle_mount)] {
        if let Some(m) = mount {
            let want = harbor_mount(part);
            if m != &Val::Str(want.into()) {
                ex.losses.push(Loss::new(
                    format!(
                        "[{part}] mount = {} (a split-layout runner always uses {want})",
                        py_repr(m)
                    ),
                    LossSeverity::RunAffecting,
                    format!("a split-layout runner always uses {want}, so the scripts would not find their files"),
                    Some(format!("[{part}] mount")),
                ));
            }
        }
    }
    if ver.get("models") == Some(&Val::Bool(true))
        && !verifier_envs(cfg)
            .iter()
            .any(|(_, env)| !provider_vars(env).is_empty())
        && !rk_recorded(cfg)
    {
        ex.losses.push(Loss::new(
            "[verifier] models = true (a split-layout runner gives a verifier script no model of its own)",
            LossSeverity::RunAffecting,
            "a split-layout verifier reaches a model only with a provider key that [verifier.env] passes from the host, and this task passes none",
            Some("[verifier] models".into()),
        ));
    }
    for (part, spec) in [("agent", AGENT), ("verifier", VERIFIER)] {
        let Some(Val::Table(src)) = cfg.get(part) else {
            continue;
        };
        let mut src = src.clone();
        if part == "verifier"
            && ["unreached_stages", "mount", "models"]
                .iter()
                .any(|k| src.contains_key(*k))
        {
            for k in ["unreached_stages", "mount", "models"] {
                src.shift_remove(k);
            }
            if src.is_empty() {
                continue;
            }
        }
        let t = ex.phase(&Val::Table(src.clone()), spec, &kp(&[part]), &kp(&[part]))?;
        out.insert(part.into(), Val::Table(t));
        ex.moves.insert(kp(&[part]), kp(&[part]));
        ex.unknown(&src, &native_keys(part), part);
    }
    if let Some(cs) = ver.get("combine_stages") {
        out.insert("multi_step_reward_strategy".into(), cs.clone());
        ex.moves.insert(
            kp(&["verifier", "combine_stages"]),
            kp(&["multi_step_reward_strategy"]),
        );
    }
    let oracle: Option<Table> = cfg.get("oracle").and_then(Val::as_table).map(|o| {
        o.iter()
            .filter(|(k, _)| *k != "mount")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    });
    if let Some(o) = oracle.filter(|o| !o.is_empty()) {
        let t = ex.table(
            &Val::Table(o.clone()),
            ENV_ONLY,
            &kp(&["oracle"]),
            &kp(&["solution"]),
            false,
        )?;
        out.insert("solution".into(), Val::Table(t));
        ex.moves.insert(kp(&["oracle"]), kp(&["solution"]));
        ex.unknown(&o, &native_keys("oracle"), "oracle");
    }
    let stages: Table = cfg
        .get("stages")
        .and_then(Val::as_table)
        .cloned()
        .unwrap_or_default();
    let unreached_zero = ver
        .get("unreached_stages")
        .is_none_or(|v| v == &Val::Str("zero".into()));
    let combine_mean = ver
        .get("combine_stages")
        .is_none_or(|v| v == &Val::Str("mean".into()));
    let gated = stages
        .values()
        .any(|st| matches!(st, Val::Table(t) if t.contains_key("gate")));
    if !stages.is_empty() && unreached_zero && combine_mean && gated {
        ex.losses.push(Loss::new(
            "unreached stages scoring 0 (a split-layout runner averages only the steps that ran)",
            LossSeverity::RunAffecting,
            "a split-layout runner averages only the steps that ran, so a run that stops early scores higher",
            Some("[verifier] unreached_stages".into()),
        ));
    }
    if !stages.is_empty() && is_step_chain(&stages) {
        let mut steps = Vec::new();
        for (i, (name, st)) in stages.iter().enumerate() {
            let Val::Table(st) = st else { continue };
            let mut step = Table::new();
            step.insert("name".into(), Val::Str(name.clone()));
            for (part, spec) in [("agent", AGENT), ("verifier", VERIFIER)] {
                if let Some(v @ Val::Table(pt)) = st.get(part) {
                    let np = kp(&["stages", name, part]);
                    let hp = vec![
                        Seg::Key("steps".into()),
                        Seg::Index(i),
                        Seg::Key(part.into()),
                    ];
                    let t = ex.phase(v, spec, &np, &hp)?;
                    step.insert(part.into(), Val::Table(t));
                    ex.moves.insert(np.clone(), hp);
                    ex.unknown(pt, &native_keys(part), &format!("stages.{name}.{part}"));
                }
            }
            let sp = kp(&["stages", name]);
            let hp = vec![Seg::Key("steps".into()), Seg::Index(i)];
            let t = ex.table(&Val::Table(st.clone()), STEP, &sp, &hp, false)?;
            step.extend(t);
            ex.moves.insert(sp.clone(), hp);
            let mut known = vec!["unlock", "agent", "verifier"];
            known.extend(STEP.iter().map(|s| s.native));
            ex.unknown(st, &known, &format!("stages.{name}"));
            steps.push(Val::Table(step));
        }
        out.insert("steps".into(), Val::List(steps));
    } else if !stages.is_empty() {
        ex.losses.push(Loss::new(
            "stages other than a simple sequence",
            LossSeverity::RunAffecting,
            "split-layout steps unlock one after another",
            Some("[stages]".into()),
        ));
    }
    let prov: Table = cfg
        .get("provenance")
        .and_then(Val::as_table)
        .cloned()
        .unwrap_or_default();
    if let Some(source) = prov.get("source") {
        out.insert("source".into(), source.clone());
        ex.moves
            .insert(kp(&["provenance", "source"]), kp(&["source"]));
    }
    if prov.keys().any(|k| k != "source") {
        let (sev, reason) = table_loss("provenance").expect("a known table");
        ex.losses.push(Loss::new("[provenance]", sev, reason, None));
    }
    let kept: Table = cfg
        .get("import")
        .and_then(Val::as_table)
        .and_then(|i| i.get(stash))
        .and_then(Val::as_table)
        .cloned()
        .unwrap_or_default();
    if !kept.is_empty() {
        let mut kept = kept;
        kept.shift_remove("instruction");
        let records: Vec<String> = kept
            .iter()
            .filter(|(k, v)| harbor_record(k, Some(v)))
            .map(|(k, _)| k.clone())
            .collect();
        for k in records {
            kept.shift_remove(&k);
        }
        let step_extra = kept.shift_remove("steps");
        match (step_extra, out.get_mut("steps")) {
            (Some(Val::Table(extra)), Some(Val::List(steps))) => {
                let index: HashMap<String, usize> = steps
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| {
                        s.as_table()
                            .and_then(|t| t.get("name"))
                            .and_then(Val::as_str)
                            .map(|n| (n.to_string(), i))
                    })
                    .collect();
                for (name, e) in &extra {
                    if let (Some(&i), Val::Table(e)) = (index.get(name), e) {
                        if let Val::Table(step) = &mut steps[i] {
                            merge(step, e);
                        }
                        ex.moves.insert(
                            kp(&["import", stash, "steps", name]),
                            vec![Seg::Key("steps".into()), Seg::Index(i)],
                        );
                    }
                }
            }
            (Some(other), _) => {
                kept.insert("steps".into(), other);
            }
            (None, _) => {}
        }
        merge(&mut out, &kept);
        ex.moves.insert(kp(&["import", stash]), Vec::new());
    }
    for (k, v) in cfg {
        if k == "integrity" && harbor_cheats(v) {
            continue;
        }
        if !TOP.contains(&k.as_str()) && !extra_top.contains(&k.as_str()) {
            let (sev, reason) = if k.starts_with("x-") {
                (
                    LossSeverity::Metadata,
                    "an extension key, which a split-layout runner ignores",
                )
            } else {
                table_loss(k).unwrap_or((LossSeverity::RunAffecting, NO_EQUIVALENT))
            };
            let text = if matches!(v, Val::Table(_)) {
                format!("[{k}]")
            } else {
                k.clone()
            };
            ex.losses.push(Loss::new(text, sev, reason, None));
        }
    }
    let first = [
        "schema_version",
        "version",
        "source",
        "multi_step_reward_strategy",
        "task",
        "steps",
        "metadata",
        "verifier",
        "agent",
        "environment",
        "solution",
    ];
    Ok(Exported {
        config: ordered(&out, &first),
        moves: ex.moves,
        losses: ex.losses,
    })
}

/// Move comments to the setting their key became, or the nearest table
/// that exists.
pub(crate) fn remap_comments(
    comments: &Comments,
    moves: &HashMap<KPath, KPath>,
    data: &Table,
) -> Comments {
    let mut out = Comments::new();
    let end = tomltext::end();
    for (path, c) in comments {
        if *path == end {
            tomltext::merge(&mut out, path.clone(), c);
            continue;
        }
        let mut n = path.len();
        while n > 0 && !moves.contains_key(&path[..n]) {
            n -= 1;
        }
        let mut new = if n > 0 {
            kjoin(&moves[&path[..n]], &path[n..])
        } else {
            path.clone()
        };
        while !new.is_empty() && !exists(data, &new) {
            new.pop();
        }
        tomltext::merge(&mut out, new, c);
    }
    out
}

fn exists(d: &Table, path: &[Seg]) -> bool {
    let mut cur: Option<&Val> = None;
    for (i, p) in path.iter().enumerate() {
        let next = match (i, cur, p) {
            (0, _, Seg::Key(k)) => d.get(k),
            (_, Some(Val::Table(t)), Seg::Key(k)) => t.get(k),
            (_, Some(Val::List(l)), Seg::Index(j)) => l.get(*j),
            _ => None,
        };
        match next {
            Some(v) => cur = Some(v),
            None => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::value::parse_toml;

    #[test]
    fn values_convert_like_convert_py() {
        assert_eq!(duration(&Val::Float(600.0)).unwrap(), "10m");
        assert_eq!(duration(&Val::Int(5400)).unwrap(), "1h30m");
        assert_eq!(duration(&Val::Float(0.5)).unwrap(), "0.5s");
        assert_eq!(duration(&Val::Int(0)).unwrap(), "0s");
        assert_eq!(duration(&Val::Int(3661)).unwrap(), "1h1m1s");
        assert!(duration(&Val::Str("x".into())).is_none());
        assert_eq!(size(&Val::Int(4096)).unwrap(), "4 GB");
        assert_eq!(size(&Val::Int(1536)).unwrap(), "1536 MB");
        assert_eq!(size(&Val::Int(1048576)).unwrap(), "1 TB");
        assert_eq!(size(&Val::Int(0)).unwrap(), "0 MB");
        assert_eq!(seconds(&Val::Str("1h30m".into())), Some(5400.0));
        assert_eq!(seconds(&Val::Str("".into())), None);
        assert_eq!(megabytes(&Val::Str("4 GB".into())), Some(4096));
        assert_eq!(megabytes(&Val::Str("0.5 MB".into())), Some(0));
        assert_eq!(megabytes(&Val::Str("1.5 MB".into())), Some(2));
    }

    #[test]
    fn authors_round_trip() {
        let mut t = Table::new();
        t.insert("name".into(), Val::Str("Ada".into()));
        t.insert("email".into(), Val::Str("ada@x.org".into()));
        let a = author(&Val::Table(t.clone()));
        assert_eq!(a, Val::Str("Ada <ada@x.org>".into()));
        assert_eq!(author_harbor(&a), Val::Table(t));
    }

    #[test]
    fn imports_a_config_and_keeps_unknown_keys() {
        let cfg = parse_toml(
            "version = \"1.0\"\nsource = \"tb\"\n[task]\nname = \"a/b\"\n[metadata]\ndifficulty = \"easy\"\n[environment]\nmemory = \"4G\"\ncpus = 2\nallow_internet = false\n[agent]\ntimeout_sec = 600.0\n[verifier]\ntimeout_sec = 120\n",
        )
        .unwrap();
        let im = config_from_harbor(&cfg, "harbor").unwrap();
        let c = &im.config;
        assert_eq!(c["name"], Val::Str("a/b".into()));
        let sb = c["sandbox"].as_table().unwrap();
        assert_eq!(
            sb.keys().collect::<Vec<_>>(),
            vec!["cpus", "memory", "network"]
        );
        assert_eq!(sb["memory"], Val::Str("4 GB".into()));
        assert_eq!(sb["network"], Val::Str("none".into()));
        assert_eq!(
            c["agent"].as_table().unwrap()["timeout"],
            Val::Str("10m".into())
        );
        assert_eq!(
            c["provenance"].as_table().unwrap()["source"],
            Val::Str("tb".into())
        );
        let kept = c["import"].as_table().unwrap()["harbor"]
            .as_table()
            .unwrap();
        assert_eq!(kept["version"], Val::Str("1.0".into()));
        assert_eq!(im.notes.len(), 2);
    }

    /// F1 round 2 (F1-A 1): the cloud metadata opt-out is `[sandbox]
    /// x-allow-cloud-metadata` in task.md and `[metadata]
    /// x-allow-cloud-metadata` in the split layout, where BenchFlow 0.8
    /// loads it (its `[environment]` refuses a key it does not know); the
    /// converter moves it between the two, each way, and loses nothing.
    #[test]
    fn the_cloud_metadata_opt_out_moves_between_sandbox_and_metadata() {
        let cfg =
            parse_toml("name = \"a/b\"\n[sandbox]\nx-allow-cloud-metadata = true\ncpus = 2\n")
                .unwrap();
        let ex = config_to_harbor(&cfg, "harbor", &[]).unwrap();
        let h = &ex.config;
        assert_eq!(
            h["metadata"].as_table().unwrap()[METADATA_OPT_OUT],
            Val::Bool(true)
        );
        let env = h["environment"].as_table().unwrap();
        assert!(!env.contains_key(METADATA_OPT_OUT), "{env:?}");
        assert!(!env.contains_key("allow_cloud_metadata"), "{env:?}");
        assert!(ex.losses.is_empty(), "{:?}", ex.losses.len());
        // Alone in [sandbox], it leaves no empty [environment] behind.
        let alone =
            parse_toml("name = \"a/b\"\n[sandbox]\nx-allow-cloud-metadata = true\n").unwrap();
        let ex = config_to_harbor(&alone, "harbor", &[]).unwrap();
        assert!(!ex.config.contains_key("environment"));
        // And back: [metadata] keeps the task's own keys as [about].
        let split = parse_toml(
            "[task]\nname = \"a/b\"\n[metadata]\ndifficulty = \"easy\"\nx-allow-cloud-metadata = true\n",
        )
        .unwrap();
        let im = config_from_harbor(&split, "harbor").unwrap();
        let c = &im.config;
        assert_eq!(
            c["sandbox"].as_table().unwrap()[METADATA_OPT_OUT],
            Val::Bool(true)
        );
        let about = c["about"].as_table().unwrap();
        assert_eq!(about.keys().collect::<Vec<_>>(), vec!["difficulty"]);
    }

    /// F1-B-r2 L3: a split value the split reader takes as a boolean
    /// (`"yes"`, `1`, `"off"`) is imported as that boolean, which task.md
    /// requires; one it refuses is kept, so task.md refuses it too.
    #[test]
    fn a_loosely_boolean_opt_out_imports_as_a_boolean() {
        for (raw, read) in [
            ("\"yes\"", Val::Bool(true)),
            ("\"Off\"", Val::Bool(false)),
            ("1", Val::Bool(true)),
            ("true", Val::Bool(true)),
            ("\"maybe\"", Val::Str("maybe".into())),
        ] {
            let split = parse_toml(&format!(
                "[task]\nname = \"a/b\"\n[metadata]\nx-allow-cloud-metadata = {raw}\n"
            ))
            .unwrap();
            let im = config_from_harbor(&split, "harbor").unwrap();
            assert_eq!(
                im.config["sandbox"].as_table().unwrap()[METADATA_OPT_OUT],
                read,
                "{raw}"
            );
            let noted = im.notes.iter().any(|n| n.contains(METADATA_OPT_OUT));
            assert_eq!(
                noted,
                matches!(raw, "\"yes\"" | "\"Off\"" | "1"),
                "{raw}: {:?}",
                im.notes
            );
        }
    }

    /// F1-A-r2: task.md ignores the opt-out under [about]; export must not
    /// move it into [metadata], where the split reader would honour it.
    #[test]
    fn an_opt_out_under_about_is_not_exported() {
        let cfg = parse_toml(
            "name = \"a/b\"\n[about]\ndifficulty = \"easy\"\nx-allow-cloud-metadata = true\n",
        )
        .unwrap();
        let ex = config_to_harbor(&cfg, "harbor", &[]).unwrap();
        let m = ex.config["metadata"].as_table().unwrap();
        assert_eq!(m.keys().collect::<Vec<_>>(), vec!["difficulty"]);
        assert!(
            ex.losses
                .iter()
                .any(|l| l.path == "[about] x-allow-cloud-metadata"),
            "{:?}",
            ex.losses
        );
        // Alone in [about], it leaves no [metadata]; [sandbox]'s own
        // opt-out still lands there.
        let cfg = parse_toml("name = \"a/b\"\n[about]\nx-allow-cloud-metadata = true\n").unwrap();
        let ex = config_to_harbor(&cfg, "harbor", &[]).unwrap();
        assert!(!ex.config.contains_key("metadata"), "{:?}", ex.config);
        let cfg = parse_toml(
            "name = \"a/b\"\n[about]\nx-allow-cloud-metadata = true\n[sandbox]\nx-allow-cloud-metadata = false\n",
        )
        .unwrap();
        let ex = config_to_harbor(&cfg, "harbor", &[]).unwrap();
        assert_eq!(
            ex.config["metadata"].as_table().unwrap()[METADATA_OPT_OUT],
            Val::Bool(false)
        );
    }

    #[test]
    fn exports_back_with_losses() {
        let cfg = parse_toml(
            "name = \"a/b\"\ntitle = \"T\"\n[sandbox]\nmemory = \"4 GB\"\nnetwork = { block = [\"x.com\"] }\nclock = { start = \"2026-01-01T00:00:00Z\" }\n[verifier]\ntimeout = \"2m\"\nmount = \"/tests\"\n[import.harbor]\nversion = \"1.0\"\n",
        )
        .unwrap();
        let ex = config_to_harbor(&cfg, "harbor", &[]).unwrap();
        let c = &ex.config;
        assert_eq!(
            c.keys().collect::<Vec<_>>(),
            vec!["version", "task", "verifier", "environment"]
        );
        assert_eq!(
            c["environment"].as_table().unwrap()["memory_mb"],
            Val::Int(4096)
        );
        assert_eq!(
            c["verifier"].as_table().unwrap()["timeout_sec"],
            Val::Float(120.0)
        );
        let paths: Vec<&str> = ex.losses.iter().map(|l| l.path.as_str()).collect();
        assert!(paths.contains(&"title"));
        assert!(paths.contains(&"[sandbox] network"));
        assert!(paths.contains(&"[sandbox] clock"));
    }

    #[test]
    fn provider_keys() {
        assert!(is_provider_var("OPENAI_API_KEY"));
        assert!(is_provider_var("VERIFIER_ANTHROPIC_API_KEY"));
        assert!(!is_provider_var("HOME"));
        let mut env = Table::new();
        env.insert("KEY".into(), Val::Str("${JUDGE_API_KEY}".into()));
        env.insert("OTHER".into(), Val::Str("x".into()));
        assert_eq!(provider_vars(&Val::Table(env)), vec!["KEY".to_string()]);
    }
}
