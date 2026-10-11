//! Shared pieces of the parser property tests (`tests/fuzz_*.rs` in each
//! crate): inputs made by mutating real files, random JSON shaped like the
//! file it stands in for, and a time limit per input.
//!
//! The case count is proptest's: `PROPTEST_CASES=200000 cargo test --test
//! fuzz_parsers` runs a long campaign; the default keeps `cargo test` quick.
//! `BENCH_FUZZ_SLOW_MS` sets the per-input time limit (default 5000).

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use proptest::prelude::*;
use proptest::test_runner::Config;
use serde_json::{Map, Value};

/// Cases per property unless `PROPTEST_CASES` says otherwise.
pub const DEFAULT_CASES: u32 = 96;

pub fn config() -> Config {
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CASES);
    Config {
        cases,
        failure_persistence: None,
        max_shrink_iters: 2048,
        ..Config::default()
    }
}

/// [`config`] for properties that cost a second or so per input (whole job
/// folders through a server): an eighth of the cases.
pub fn config_slow() -> Config {
    let mut c = config();
    c.cases = (c.cases / 8).max(1);
    c
}

static RUNS: std::sync::Mutex<Vec<(String, u64)>> = std::sync::Mutex::new(Vec::new());

/// Counts inputs per target and, with `BENCH_FUZZ_PROGRESS=<n>`, prints a
/// line to stderr every n of them (long campaigns run with `--nocapture`).
fn count(what: &str) {
    let Some(every) = std::env::var("BENCH_FUZZ_PROGRESS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
    else {
        return;
    };
    let mut runs = RUNS.lock().unwrap_or_else(|e| e.into_inner());
    let n = match runs.iter_mut().find(|(w, _)| w == what) {
        Some((_, n)) => {
            *n += 1;
            *n
        }
        None => {
            runs.push((what.to_string(), 1));
            1
        }
    };
    if n.is_multiple_of(every.max(1)) {
        eprintln!("fuzz-progress {what} {n}");
    }
}

/// Runs `f`, failing when one input takes longer than the limit.
pub fn timed<T>(what: &str, f: impl FnOnce() -> T) -> T {
    count(what);
    let limit = std::env::var("BENCH_FUZZ_SLOW_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);
    let start = Instant::now();
    let out = f();
    let took = start.elapsed();
    assert!(
        took < Duration::from_millis(limit),
        "{what} took {took:?} on one input"
    );
    out
}

/// Text fragments that tend to break parsers: delimiters of the formats we
/// read, odd numbers, odd whitespace, non-ASCII, paths and patterns.
pub const TOKENS: &[&str] = &[
    "\"",
    "'",
    "\\",
    "[",
    "]",
    "{",
    "}",
    "[[",
    "]]",
    "=",
    ":",
    ",",
    "\n",
    "\r\n",
    "\r",
    "\t",
    " ",
    "---\n",
    "+++\n",
    "```",
    "~~~",
    "\"\"\"",
    "'''",
    "#",
    "- ",
    "* ",
    "&a",
    "*a",
    "!!str",
    "? ",
    "|\n",
    ">\n",
    "<",
    ">",
    "</",
    "/>",
    "\0",
    "\u{7f}",
    "\u{85}",
    "\u{a0}",
    "\u{2028}",
    "\u{feff}",
    "\u{1c}",
    "ß",
    "İ",
    "Σ",
    "é",
    "e\u{301}",
    "🦀",
    "١",
    "0",
    "-0",
    "1",
    "-1",
    "1.0",
    "1e999",
    "-1e999",
    "1e-999",
    "NaN",
    "nan",
    "inf",
    "-inf",
    "Infinity",
    "0x10",
    "0o7",
    "1_000",
    "18446744073709551616",
    "-9223372036854775809",
    "9007199254740993",
    "0.1",
    "1.5",
    "2.5",
    "true",
    "false",
    "null",
    "~",
    "yes",
    "no",
    "on",
    "off",
    "2026-10-10",
    "2026-10-10T17:00:00Z",
    "2026-10-10T17:00:00",
    "12:00",
    "30s",
    "5m",
    "1h",
    "1.5h",
    "2G",
    "512Mi",
    "-5",
    "../",
    "../../etc/passwd",
    "/",
    "/abs/path",
    "~/x",
    "C:\\",
    "*",
    "**",
    "**/",
    "?",
    "[!a]",
    "{a,b}",
    "$1",
    "${X}",
    "$(x)",
    "`x`",
    "%s",
    "{}",
    "{0}",
    "\\u0000",
    "\\ud800",
    "\\uDFFF",
    "\\x00",
];

/// One edit of a text.
#[derive(Debug, Clone)]
pub enum Edit {
    Insert(f64, String),
    Delete(f64, usize),
    Replace(f64, usize, String),
    Repeat(f64, usize, usize),
    Splice(f64, usize, f64, usize),
    Truncate(f64),
}

fn token() -> impl Strategy<Value = String> {
    prop_oneof![
        6 => proptest::sample::select(TOKENS).prop_map(str::to_string),
        2 => "\\PC{0,12}",
        1 => any::<char>().prop_map(String::from),
        1 => (proptest::sample::select(&["a", "9", " ", "[", "{", "\"", "é"][..]), 64usize..20_000)
            .prop_map(|(s, n)| s.repeat(n)),
    ]
}

fn edit() -> impl Strategy<Value = Edit> {
    let at = 0.0f64..=1.0;
    prop_oneof![
        5 => (at.clone(), token()).prop_map(|(a, t)| Edit::Insert(a, t)),
        3 => (at.clone(), 1usize..40).prop_map(|(a, n)| Edit::Delete(a, n)),
        4 => (at.clone(), 1usize..12, token()).prop_map(|(a, n, t)| Edit::Replace(a, n, t)),
        1 => (at.clone(), 1usize..80, 2usize..64).prop_map(|(a, n, k)| Edit::Repeat(a, n, k)),
        2 => (at.clone(), 1usize..200, at.clone(), 0usize..4)
            .prop_map(|(a, n, b, s)| Edit::Splice(a, n, b, s)),
        1 => at.prop_map(Edit::Truncate),
    ]
}

fn pos(at: f64, len: usize) -> usize {
    ((at * len as f64) as usize).min(len)
}

/// Applies `edits` to `seeds[pick]`; a splice copies a run of another seed.
pub fn apply(seeds: &[String], pick: usize, edits: &[Edit]) -> String {
    let mut text: Vec<char> = seeds[pick % seeds.len()].chars().collect();
    for e in edits {
        let len = text.len();
        match e {
            Edit::Insert(a, t) => {
                let p = pos(*a, len);
                text.splice(p..p, t.chars());
            }
            Edit::Delete(a, n) => {
                let p = pos(*a, len);
                let end = (p + n).min(len);
                text.drain(p..end);
            }
            Edit::Replace(a, n, t) => {
                let p = pos(*a, len);
                let end = (p + n).min(len);
                text.splice(p..end, t.chars());
            }
            Edit::Repeat(a, n, k) => {
                let p = pos(*a, len);
                let end = (p + n).min(len);
                let run: Vec<char> = text[p..end].to_vec();
                for _ in 1..*k {
                    text.splice(end..end, run.iter().copied());
                }
            }
            Edit::Splice(a, n, b, s) => {
                let other: Vec<char> = seeds[(pick + s) % seeds.len()].chars().collect();
                let from = pos(*b, other.len());
                let end = (from + n).min(other.len());
                let p = pos(*a, len);
                text.splice(p..p, other[from..end].iter().copied());
            }
            Edit::Truncate(a) => text.truncate(pos(*a, len)),
        }
        if text.len() > 400_000 {
            text.truncate(400_000);
        }
    }
    text.into_iter().collect()
}

/// Texts made by editing one of `seeds` a few times (or a seed untouched).
pub fn mutated(seeds: Vec<String>) -> impl Strategy<Value = String> {
    assert!(!seeds.is_empty(), "no seed files");
    let n = seeds.len();
    (0..n, proptest::collection::vec(edit(), 0..8))
        .prop_map(move |(pick, edits)| apply(&seeds, pick, &edits))
}

/// A JSON scalar that stresses number and string handling.
pub fn json_leaf() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::from),
        any::<u64>().prop_map(Value::from),
        proptest::sample::select(
            &[
                0.0,
                -0.0,
                0.5,
                1.0,
                1.5,
                2.5,
                -1.0,
                1e-300,
                1e300,
                f64::MAX,
                f64::MIN_POSITIVE
            ][..]
        )
        .prop_map(Value::from),
        any::<f64>().prop_map(Value::from),
        proptest::sample::select(TOKENS).prop_map(Value::from),
        "\\PC{0,24}".prop_map(Value::from),
    ]
}

/// Random JSON whose object keys and many strings come from `keys` (the
/// names and tags the reader looks for), so it gets past its first checks.
pub fn json_with(keys: &'static [&'static str]) -> impl Strategy<Value = Value> {
    let key = prop_oneof![
        8 => proptest::sample::select(keys).prop_map(str::to_string),
        1 => "\\PC{0,8}",
    ];
    let leaf = prop_oneof![
        3 => json_leaf(),
        2 => proptest::sample::select(keys).prop_map(Value::from),
    ];
    leaf.prop_recursive(5, 96, 8, move |inner| {
        prop_oneof![
            proptest::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
            proptest::collection::vec((key.clone(), inner), 0..10)
                .prop_map(|kv| Value::Object(kv.into_iter().collect::<Map<_, _>>())),
        ]
    })
}

/// A seed JSON document with a few of its values (at any depth) replaced by
/// random JSON, or a key renamed or dropped.
pub fn json_mutated(
    seeds: Vec<Value>,
    keys: &'static [&'static str],
) -> impl Strategy<Value = Value> {
    assert!(!seeds.is_empty(), "no seed files");
    let n = seeds.len();
    let change = (any::<u32>(), 0u8..4, json_with(keys));
    (0..n, proptest::collection::vec(change, 0..6)).prop_map(move |(pick, changes)| {
        let mut doc = seeds[pick].clone();
        for (walk, how, new) in changes {
            mutate_at(&mut doc, walk, how, new);
        }
        doc
    })
}

fn mutate_at(doc: &mut Value, mut walk: u32, how: u8, new: Value) {
    let mut node = doc;
    loop {
        let len = match node {
            Value::Object(m) => m.len(),
            Value::Array(a) => a.len(),
            _ => 0,
        };
        if len == 0 || walk.is_multiple_of(3) {
            break;
        }
        let i = (walk as usize / 3) % len;
        walk /= 7;
        node = match node {
            Value::Object(m) => {
                if how == 3 && walk.is_multiple_of(2) {
                    let key = m.keys().nth(i).cloned().unwrap_or_default();
                    m.shift_remove(&key);
                    return;
                }
                m.values_mut().nth(i).unwrap()
            }
            Value::Array(a) => &mut a[i],
            _ => unreachable!(),
        };
    }
    match (how, node) {
        (2, Value::Object(m)) => {
            if let Value::Object(extra) = new {
                m.extend(extra);
            }
        }
        (_, node) => *node = new,
    }
}

/// The text of every file under `dir` (recursively) whose name passes `want`.
pub fn files(dir: &Path, want: &dyn Fn(&Path) -> bool) -> Vec<(PathBuf, String)> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        let mut entries: Vec<_> = entries.flatten().map(|e| e.path()).collect();
        entries.sort();
        for p in entries {
            if p.is_dir() {
                stack.push(p);
            } else if want(&p)
                && let Ok(text) = std::fs::read_to_string(&p)
            {
                out.push((p, text));
            }
        }
    }
    out
}

/// Texts of the files named `names` anywhere under `dirs` (relative to the
/// workspace's `crates/`), capped at `max` and 256 KB each.
pub fn seeds(dirs: &[&str], names: &dyn Fn(&str) -> bool, max: usize) -> Vec<String> {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut out = Vec::new();
    for d in dirs {
        for (_, text) in files(&crates.join(d), &|p: &Path| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(names)
        }) {
            if text.len() <= 256 * 1024 && out.len() < max {
                out.push(text);
            }
        }
    }
    out
}

/// The lines of a file of words (one per line), kept for the whole run.
pub fn words(path: &Path) -> &'static [&'static str] {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let list: Vec<&'static str> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| &*Box::leak(l.to_string().into_boxed_str()))
        .collect();
    Box::leak(list.into_boxed_slice())
}

/// Parsed JSON seeds (files that do not parse are skipped).
pub fn json_seeds(dirs: &[&str], names: &dyn Fn(&str) -> bool, max: usize) -> Vec<Value> {
    seeds(dirs, names, max * 4)
        .iter()
        .filter_map(|t| serde_json::from_str(t).ok())
        .take(max)
        .collect()
}
