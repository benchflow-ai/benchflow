//! Shared pieces of the parser property tests (`tests/fuzz_*.rs` in each
//! crate): inputs made by mutating real files, random JSON shaped like the
//! file it stands in for, and a time limit per input.
//!
//! The case count is proptest's: `PROPTEST_CASES=200000 cargo test --test
//! fuzz_parsers` runs a long campaign; the default keeps `cargo test` quick.
//! Each input runs under [`timed`]: past `BENCH_FUZZ_SLOW_MS` (default 5000)
//! it is logged as slow and waited for; past `BENCH_FUZZ_HANG_MS` (default
//! ten times that, and at least 120000) it fails the property as hung.
//! `BENCH_FUZZ_SEED` sets the seed every property starts from (see
//! [`rng_seed`]). A failing case's own seed is saved under the target folder,
//! never beside the source, and tried first on the next run; not in CI, and
//! never for a hang (see [`persistence_dir`]). `PROPTEST_MAX_SHRINK_ITERS`
//! and `PROPTEST_MAX_SHRINK_TIME` override the shrink budgets below.

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::io::Write as _;
use std::path::{Component, Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use proptest::prelude::*;
use proptest::test_runner::{
    Config, FailurePersistence, FileFailurePersistence, PersistedSeed, RngSeed,
};
use serde_json::{Map, Value};

/// Cases per property unless `PROPTEST_CASES` says otherwise.
pub const DEFAULT_CASES: u32 = 96;

/// The seed properties start from in CI, so a failure there can be run
/// again as it was.
pub const CI_SEED: u64 = 20_261_010;

/// The seed proptest starts each property from: `BENCH_FUZZ_SEED` (or
/// proptest's own `PROPTEST_RNG_SEED`) when set; [`CI_SEED`] when `CI` is
/// set (GitHub Actions sets it), so CI runs the same inputs every time;
/// otherwise random, so each local run tries new inputs.
pub fn rng_seed() -> RngSeed {
    if let Some(seed) = std::env::var("BENCH_FUZZ_SEED")
        .ok()
        .and_then(|s| s.trim().parse().ok())
    {
        return RngSeed::Fixed(seed);
    }
    if std::env::var_os("PROPTEST_RNG_SEED").is_some() {
        return Config::default().rng_seed;
    }
    if in_ci() {
        return RngSeed::Fixed(CI_SEED);
    }
    RngSeed::Random
}

/// Whether `CI` is set (to anything but empty, `0` or `false`).
fn in_ci() -> bool {
    std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0" && v != "false")
}

/// The folder failing seeds are saved in: `proptest-regressions/` in the
/// target folder the test binary was built in (`target/`, or wherever
/// `CARGO_TARGET_DIR` points), never beside the test's source. `None`, so
/// nothing is saved or replayed, when `CI` is set (CI's fixed seed already
/// makes a failure there reproducible), when `BENCH_FUZZ_PERSIST=0` or
/// `PROPTEST_DISABLE_FAILURE_PERSISTENCE` is set, or when the target folder
/// cannot be found.
pub fn persistence_dir() -> Option<PathBuf> {
    let off = in_ci()
        || std::env::var_os("PROPTEST_DISABLE_FAILURE_PERSISTENCE").is_some()
        || std::env::var("BENCH_FUZZ_PERSIST").is_ok_and(|v| v.trim() == "0");
    if off {
        return None;
    }
    target_dir().map(|t| t.join("proptest-regressions"))
}

/// The target folder of the running test binary, which Cargo puts at
/// `<target>/[<triple>/]<profile>/deps/<name>-<hash>`: the nearest folder
/// above it holding Cargo's `CACHEDIR.TAG`, else the `<profile>` folder
/// above `deps`.
pub fn target_dir() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let mut above = exe.ancestors().skip(1);
    if let Some(dir) = above.clone().find(|d| d.join("CACHEDIR.TAG").is_file()) {
        return Some(dir.to_path_buf());
    }
    above
        .find(|d| d.file_name().is_some_and(|n| n == "deps"))
        .and_then(Path::parent)
        .map(Path::to_path_buf)
}

/// The seed file in `dir` for the test source `source` (proptest's
/// `file!()`): the source's path with its folders joined by `__` and `.txt`
/// for `.rs`, so `crates/benchflow-task/tests/fuzz_parsers.rs` saves to
/// `crates__benchflow-task__tests__fuzz_parsers.txt`.
pub fn persistence_file(dir: &Path, source: &str) -> PathBuf {
    let parts: Vec<String> = Path::new(source)
        .components()
        .filter_map(|c| match c {
            Component::Normal(part) => Some(part.to_string_lossy().into_owned()),
            _ => None,
        })
        .collect();
    let flat = parts.join("__");
    let flat = flat.strip_suffix(".rs").unwrap_or(&flat);
    dir.join(format!("{flat}.txt"))
}

/// Proptest's own file format and replay, at [`persistence_file`] under the
/// folder it holds, except that a property in which an input hung (see
/// [`timed`]) saves nothing: whether an input finishes in time depends on
/// the machine's load as much as on the input, and a saved seed would be
/// replayed first on every later run.
#[derive(Debug, Clone, PartialEq)]
struct UnderTarget(PathBuf);

impl UnderTarget {
    fn file(&self, source: Option<&'static str>) -> Option<FileFailurePersistence> {
        let path = persistence_file(&self.0, source?);
        let path: &'static str = Box::leak(path.to_string_lossy().into_owned().into_boxed_str());
        Some(FileFailurePersistence::Direct(path))
    }
}

impl FailurePersistence for UnderTarget {
    fn load_persisted_failures2(&self, source_file: Option<&'static str>) -> Vec<PersistedSeed> {
        self.file(source_file)
            .map(|f| f.load_persisted_failures2(None))
            .unwrap_or_default()
    }

    fn save_persisted_failure2(
        &mut self,
        source_file: Option<&'static str>,
        seed: PersistedSeed,
        shrunken_value: &dyn std::fmt::Debug,
    ) {
        if let Some(what) = hung_here()
            && !FAILED_FIRST.with(|f| f.get())
        {
            eprintln!("fuzz: not saving the seed of this failure: an input of {what} hung");
            return;
        }
        if let Some(mut f) = self.file(source_file) {
            f.save_persisted_failure2(None, seed, shrunken_value);
        }
    }

    fn box_clone(&self) -> Box<dyn FailurePersistence> {
        Box::new(self.clone())
    }

    fn eq(&self, other: &dyn FailurePersistence) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|s| s.trim().parse().ok())
}

/// A config with this many shrink steps and milliseconds of shrinking at
/// most, unless the `PROPTEST_MAX_SHRINK_*` variables say otherwise.
fn config_with(shrink_iters: u32, shrink_ms: u32) -> Config {
    let cases = env_u32("PROPTEST_CASES").unwrap_or(DEFAULT_CASES);
    let persistence =
        persistence_dir().map(|dir| Box::new(UnderTarget(dir)) as Box<dyn FailurePersistence>);
    Config {
        cases,
        failure_persistence: persistence,
        max_shrink_iters: env_u32("PROPTEST_MAX_SHRINK_ITERS").unwrap_or(shrink_iters),
        max_shrink_time: env_u32("PROPTEST_MAX_SHRINK_TIME").unwrap_or(shrink_ms),
        rng_seed: rng_seed(),
        ..Config::default()
    }
}

/// For in-memory parsers (milliseconds per input): a deep shrink, so a
/// failure comes back as a small input, but at most two minutes of it.
pub fn config() -> Config {
    config_with(2048, 120_000)
}

/// For properties that write and read whole folders (task folders, imports,
/// job folders: 5 to 50 ms per input on an idle machine, much more on a
/// loaded one, and up to the time limit per try when an input hangs): a
/// short shrink, so a failure is reported within a minute or two.
pub fn config_files() -> Config {
    config_with(64, 60_000)
}

/// For properties that cost a second or so per input (whole job folders
/// through a server): an eighth of the cases and a shorter shrink.
pub fn config_slow() -> Config {
    let mut c = config_with(16, 60_000);
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

fn env_ms(name: &str) -> Option<Duration> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .map(Duration::from_millis)
}

/// The time after which an input counts as slow: `BENCH_FUZZ_SLOW_MS`,
/// default 5000. A slow input is logged and waited for, not failed.
pub fn slow_limit() -> Duration {
    env_ms("BENCH_FUZZ_SLOW_MS").unwrap_or(Duration::from_millis(5000))
}

/// The time after which an input counts as hung and fails its property:
/// `BENCH_FUZZ_HANG_MS`, default ten times [`slow_limit`] and at least two
/// minutes, so a loaded CI machine does not fail an input that is only slow.
pub fn hang_cap() -> Duration {
    env_ms("BENCH_FUZZ_HANG_MS").unwrap_or_else(|| {
        slow_limit()
            .saturating_mul(10)
            .max(Duration::from_secs(120))
    })
}

thread_local! {
    /// The target whose input hung on this thread, if one did. libtest runs
    /// each test on a thread of its own, so this is per property.
    static HUNG: RefCell<Option<String>> = const { RefCell::new(None) };
}

thread_local! {
    /// Whether an input failed on this thread before any hung: then a hang
    /// came while shrinking that failure, and its seed, which regenerates
    /// the first failing input, is still saved.
    static FAILED_FIRST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Notes a failing input on this thread, unless one already hung.
fn note_failure() {
    if hung_here().is_none() {
        FAILED_FIRST.with(|f| f.set(true));
    }
}

/// The target (the `what` given to [`timed`]) whose input hung on this
/// thread, if one did.
pub fn hung_here() -> Option<String> {
    HUNG.with(|h| h.borrow().clone())
}

/// What [`timed`] returns for an input it does not run: a pass.
pub trait Pass {
    fn pass() -> Self;
    /// Whether this answer fails the case.
    fn failed(&self) -> bool {
        false
    }
}

impl Pass for () {
    fn pass() {}
}

impl<T: Pass, E> Pass for Result<T, E> {
    fn pass() -> Self {
        Ok(T::pass())
    }
    fn failed(&self) -> bool {
        self.as_ref().is_ok_and(Pass::failed) || self.is_err()
    }
}

/// Writes one line to the process's stderr, past libtest's capture, so CI
/// logs show it even when the test passes.
fn log(line: &str) {
    let _ = writeln!(std::io::stderr(), "{line}");
}

/// Runs `f` on a thread of its own and waits for it, so an input that hangs
/// fails the test instead of stalling it. A panic in `f` fails the case with
/// its own message.
///
/// - Past [`slow_limit`] the input is slow: a `fuzz-slow` line goes to
///   stderr (with the target and, once it ends, the time it took), and it is
///   waited for. A slow input does not fail.
/// - Past [`hang_cap`] it is hung: the case fails, and its thread is left
///   running. Proptest then shrinks the failure, which would start another
///   thread for each shrink step that hangs too, each spinning or stuck. So
///   after a hang, `timed` on the same thread runs nothing and returns
///   [`Pass::pass`]: proptest finds no smaller failing input and reports
///   the one that hung, as it was. No seed is saved for it (see
///   [`persistence_dir`]).
pub fn timed<T: Pass + Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    timed_within(slow_limit(), hang_cap(), what, f)
}

/// [`timed`] with its own slow limit and hang cap.
pub fn timed_within<T: Pass + Send + 'static>(
    limit: Duration,
    cap: Duration,
    what: &str,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    if hung_here().is_some() {
        return T::pass();
    }
    count(what);
    let start = Instant::now();
    let limit = limit.min(cap);
    let (done, finished) = mpsc::channel();
    let worker = std::thread::Builder::new()
        .name(format!("fuzz {what}"))
        .spawn(move || {
            let _ = done.send(f());
        })
        .expect("start a thread for one input");
    let mut slow = false;
    let out = loop {
        let until = if slow { cap } else { limit };
        match finished.recv_timeout(until.saturating_sub(start.elapsed())) {
            Ok(out) => break out,
            Err(RecvTimeoutError::Timeout) if !slow && limit < cap => {
                slow = true;
                log(&format!(
                    "fuzz-slow: {what}: one input is still running after {limit:?} (BENCH_FUZZ_SLOW_MS); waiting up to {cap:?} (BENCH_FUZZ_HANG_MS)"
                ));
            }
            Err(RecvTimeoutError::Timeout) => {
                HUNG.with(|h| *h.borrow_mut() = Some(what.to_string()));
                panic!(
                    "{what} did not finish one input within {cap:?} (BENCH_FUZZ_HANG_MS), so it counts as hung. Its thread is left running; this property's later inputs (proptest's shrink steps) are not run, so the input shown is the one that hung, unshrunk, and its seed is not saved"
                )
            }
            Err(RecvTimeoutError::Disconnected) => match worker.join() {
                Err(panic) => {
                    note_failure();
                    std::panic::resume_unwind(panic)
                }
                Ok(()) => unreachable!("the input's thread ended without an answer or a panic"),
            },
        }
    };
    let _ = worker.join();
    if out.failed() {
        note_failure();
    }
    if slow {
        log(&format!(
            "fuzz-slow: {what}: one input took {:?} (over {limit:?}; slow, not failed)",
            start.elapsed()
        ));
    }
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
