//! Property tests over the task crate's readers of outside input: task.md,
//! task.toml and the split layout, BenchFlow 0.6 tasks, rubric.json,
//! ctrf.json, reward.txt and reward.json, the value grammars (durations,
//! sizes, counts), and the Terminal-Bench import (task.yaml and
//! docker-compose). Each must return or report an error on any input, never
//! panic, and finish one input within the time limit.
//!
//! Inputs are real fixture files with random edits, or random JSON shaped
//! like the file. See `fuzz_support` for the knobs (`PROPTEST_CASES`).

mod fuzz_support;

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering::SeqCst;
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use benchflow_task::check::CheckOptions;
use benchflow_task::ctrf::Ctrf;
use benchflow_task::rubric::Rubric;
use benchflow_task::{reward, taskmd, values};
use fuzz_support::*;
use proptest::prelude::*;
use proptest::test_runner::{RngSeed, TestError, TestRunner};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Every task folder among the fixtures (holding a task.md or task.toml).
static TASK_DIRS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut dirs: Vec<PathBuf> = files(&fixtures(), &|p: &Path| {
        matches!(
            p.file_name().and_then(|n| n.to_str()),
            Some("task.md" | "task.toml")
        )
    })
    .into_iter()
    .filter_map(|(p, _)| p.parent().map(Path::to_path_buf))
    .collect();
    dirs.sort();
    dirs.dedup();
    dirs
});

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let p = e.path();
        let t = to.join(e.file_name());
        if p.is_dir() {
            copy_dir(&p, &t);
        } else {
            std::fs::copy(&p, &t).unwrap();
        }
    }
}

const RUBRIC_KEYS: &[&str] = &[
    "$schema",
    "version",
    "extends",
    "criteria",
    "id",
    "name",
    "description",
    "points",
    "gate",
    "threshold",
    "kind",
    "type",
    "check",
    "test",
    "tests",
    "judge",
    "model",
    "stated",
    "files",
    "prompts",
    "scoring",
    "weight",
    "pass",
    "fail",
    "partial",
    "agreement",
    "behaviors",
    "rubric",
    "items",
    "aggregate",
    "min",
    "max",
];

const CTRF_KEYS: &[&str] = &[
    "results",
    "tool",
    "name",
    "summary",
    "tests",
    "status",
    "filePath",
    "file_path",
    "duration",
    "message",
    "trace",
    "passed",
    "failed",
    "skipped",
    "pending",
    "other",
    "extra",
    "suite",
];

const REWARD_KEYS: &[&str] = &[
    "reward", "score", "accuracy", "passed", "total", "details", "metrics", "raw", "errors",
];

proptest! {
    #![proptest_config(config())]

    /// task.md text straight into the parser.
    #[test]
    fn taskmd_parse(text in mutated(seeds(&["benchflow-task/tests/fixtures"], &|n| n == "task.md", 400))) {
        timed("taskmd::parse", move || {
            let doc = taskmd::parse(&text, Path::new("task.md"));
            let _ = doc.agent_view();
            let _ = doc.title();
            let _ = doc.told();
        });
    }

    /// rubric.json: real rubrics with values swapped for random JSON.
    #[test]
    fn rubric_from_value(doc in json_mutated(
        json_seeds(&["benchflow-task/tests/fixtures"], &|n| n.starts_with("rubric") && n.ends_with(".json"), 200),
        RUBRIC_KEYS,
    )) {
        timed("Rubric::from_value", move || {
            if let Ok(r) = Rubric::from_value(&doc) {
                let _ = r.needs_model();
                let ctrf = Ctrf::from_slice(br#"{"results":{"tests":[{"name":"a","status":"passed"}]}}"#);
                let verdicts = r.test_verdicts(&ctrf);
                let _ = r.try_score(&verdicts);
                let score = r.grade(&ctrf, &verdicts);
                let _ = score.review_json();
            }
        });
    }

    /// rubric.json as text, edited at the byte level.
    #[test]
    fn rubric_from_path(text in mutated(seeds(&["benchflow-task/tests/fixtures"], &|n| n.starts_with("rubric") && n.ends_with(".json"), 200))) {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("rubric.json");
        std::fs::write(&p, &text).unwrap();
        timed("Rubric::from_path", move || { let _ = Rubric::from_path(&p); });
    }

    /// ctrf.json, as random JSON of its shape and as edited text; then the
    /// rubric's test matching over the tests it holds.
    #[test]
    fn ctrf_parse(
        doc in json_mutated(
            {
                let mut s = json_seeds(&["benchflow-task", "benchflow-run", "benchflow-viewer"], &|n| n.starts_with("ctrf"), 50);
                s.push(serde_json::json!({"results": {"tool": {"name": "pytest"}, "summary": {"tests": 1, "passed": 1}, "tests": [{"name": "test_a", "status": "passed", "duration": 0.1}]}}));
                s
            },
            CTRF_KEYS,
        ),
        check in prop_oneof![Just("test_a".to_string()), "\\PC{0,16}", proptest::sample::select(TOKENS).prop_map(str::to_string)],
    ) {
        let bytes = serde_json::to_vec(&doc).unwrap();
        timed("Ctrf::from_slice", move || {
            let c = Ctrf::from_slice(&bytes);
            let _ = c.matching(&check);
            let _ = c.verdict(&check);
        });
    }

    #[test]
    fn ctrf_bytes(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
        timed("Ctrf::from_slice", move || { let _ = Ctrf::from_slice(&bytes); });
    }

    /// reward.txt and reward.json, alone and together.
    #[test]
    fn rewards(
        txt in prop::option::of(prop_oneof![
            mutated(vec!["1".into(), "0.5\n".into(), " 1.0 ".into(), "0".into()]),
            proptest::sample::select(TOKENS).prop_map(str::to_string),
        ]),
        json in prop::option::of(prop_oneof![
            json_with(REWARD_KEYS).prop_map(|v| v.to_string()),
            mutated(vec![r#"{"reward": 1.0}"#.into(), r#"{"reward": 0.5, "accuracy": 0.25, "details": {"a": 1}}"#.into(), r#"{"score": 1}"#.into(), r#"{"reward": 0.5, "tests": 12, "metrics": {"acc": 0.3}}"#.into(), r#"{"reward": 0.42, "raw": 42, "errors": 3}"#.into()]),
        ]),
    ) {
        let tmp = tempfile::tempdir().unwrap();
        if let Some(t) = &txt { std::fs::write(tmp.path().join("reward.txt"), t).unwrap(); }
        if let Some(j) = &json { std::fs::write(tmp.path().join("reward.json"), j).unwrap(); }
        timed("read_rewards", move || {
            if let Ok(r) = reward::read_rewards(tmp.path()) {
                prop_assert!((0.0..=1.0).contains(&r.reward), "a reward must be in [0, 1]: {r:?}");
                // Every number 0.8's validate_reward_map range-checks: not
                // those under its reserved keys (raw, errors, ...).
                prop_assert!(r.values.iter().filter(|(k, _)| !reward::UNCHECKED_KEYS.contains(&k.as_str())).all(|(_, v)| (0.0..=1.0).contains(v)), "a scored number outside [0, 1]: {r:?}");
                let metrics = r.extra.get("metrics").and_then(|m| m.as_object());
                prop_assert!(metrics.into_iter().flatten().filter_map(|(_, v)| v.as_f64()).all(|v| (0.0..=1.0).contains(&v)), "a metric outside [0, 1]: {r:?}");
            }
            Ok(())
        })?;
    }

    /// The value grammars of task.md.
    #[test]
    fn value_grammars(text in prop_oneof![
        "[0-9]{1,30}(\\.[0-9]{0,30})?(h|m|s| MB| GB| TB|MB|GB|TB|K|M)?",
        "([0-9]{1,24}h)?([0-9]{1,24}m)?([0-9]{1,24}s)?",
        mutated(vec!["1h30m".into(), "90s".into(), "4 GB".into(), "512MB".into(), "400K".into(), "2M".into()]),
    ]) {
        timed("values", move || {
            if let Some(d) = values::parse_duration(&text) {
                let _ = values::format_duration(d);
            }
            let _ = values::parse_size_mb(&text);
            let _ = values::parse_bytes(&Value::String(text.clone()));
            let _ = values::parse_count(&Value::String(text.clone()));
        });
    }
}

proptest! {
    // Whole folders written and read: a short shrink.
    #![proptest_config(config_files())]

    /// A real task folder with its task.md or task.toml edited, loaded and
    /// checked the way `bench check` does.
    #[test]
    fn task_folder_load_and_check(
        pick in any::<prop::sample::Index>(),
        text in mutated(seeds(&["benchflow-task/tests/fixtures"], &|n| n == "task.md" || n == "task.toml", 400)),
    ) {
        let src = pick.get(&TASK_DIRS);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        copy_dir(src, &dir);
        // Write the edited text over whichever config the folder has.
        let name = if dir.join("task.md").exists() { "task.md" } else { "task.toml" };
        std::fs::write(dir.join(name), &text).unwrap();
        timed("load + check", move || {
            let _ = benchflow_task::load(&dir);
            let _ = benchflow_task::check(&dir, &CheckOptions::default());
            let _ = benchflow_task::detect(&dir);
        });
    }

    /// A Terminal-Bench task with its task.yaml or docker-compose.yaml
    /// edited, imported to task.md.
    #[test]
    fn tbench_import(
        which in 0usize..2,
        pick in any::<prop::sample::Index>(),
        text in mutated(seeds(&["benchflow-task/tests/fixtures/tbench/input"], &|n| n == "task.yaml" || n.starts_with("docker-compose"), 50)),
    ) {
        let mut inputs: Vec<PathBuf> = std::fs::read_dir(fixtures().join("tbench/input")).unwrap()
            .flatten().map(|e| e.path()).filter(|p| p.join("task.yaml").exists()).collect();
        // Directory order differs between file systems; the CI seed must
        // pick the same folder everywhere.
        inputs.sort();
        let src = pick.get(&inputs);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("in");
        copy_dir(src, &dir);
        let name = if which == 0 { "task.yaml" } else { "docker-compose.yaml" };
        std::fs::write(dir.join(name), &text).unwrap();
        let out = tmp.path().join("out");
        timed("import tbench", move || {
            let _ = benchflow_task::convert::import_to_taskmd(&dir, &out, false);
        });
    }

    /// A split-layout or 0.6 task with its config edited, imported to
    /// task.md.
    #[test]
    fn split_import(
        pick in any::<prop::sample::Index>(),
        text in mutated(seeds(&["benchflow-task/tests/fixtures"], &|n| n == "task.toml" || n == "task.md", 400)),
    ) {
        let candidates: Vec<&PathBuf> = TASK_DIRS.iter()
            .filter(|d| !matches!(benchflow_task::detect(d), Some(benchflow_task::Format::TaskMd) | None))
            .collect();
        let src = pick.get(&candidates);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("in");
        copy_dir(src, &dir);
        let name = if dir.join("task.toml").exists() { "task.toml" } else { "task.md" };
        std::fs::write(dir.join(name), &text).unwrap();
        let out = tmp.path().join("out");
        timed("import split", move || {
            let _ = benchflow_task::convert::import_to_taskmd(&dir, &out, false);
        });
    }
}

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// `timed` fails an input that hangs once its hang cap is up instead of
/// waiting for it (the hung thread is left running), and after that runs
/// nothing more on the same thread. It used to time an input only after it
/// finished.
#[test]
fn timed_stops_waiting_for_a_hung_input() {
    // On a thread of its own, so the hang's mark stays off libtest's.
    std::thread::spawn(timed_stops_waiting_for_a_hung_input_here)
        .join()
        .unwrap();
}

fn timed_stops_waiting_for_a_hung_input_here() {
    let start = Instant::now();
    let hung = std::panic::catch_unwind(|| {
        timed_within(ms(50), ms(200), "self-test hang", || {
            std::thread::sleep(Duration::from_secs(30))
        })
    });
    let message = hung.expect_err("a hung input fails");
    let message = message
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(message.contains("did not finish"), "{message}");
    assert!(
        start.elapsed() < Duration::from_secs(10),
        "waited {:?}",
        start.elapsed()
    );
    assert_eq!(hung_here().as_deref(), Some("self-test hang"));
    let ran = Arc::new(AtomicBool::new(false));
    let flag = ran.clone();
    timed_within(ms(50), ms(200), "self-test after", move || {
        flag.store(true, SeqCst)
    });
    assert!(
        !ran.load(SeqCst),
        "an input ran after a hang on the same thread"
    );
}

/// `timed` passes a panic and a value through.
#[test]
fn timed_passes_a_panic_and_a_value_through() {
    let panicked = std::panic::catch_unwind(|| {
        timed_within::<()>(ms(5000), ms(30_000), "self-test panic", || {
            panic!("its own message")
        })
    });
    let panicked = panicked.expect_err("a panic fails the input");
    assert_eq!(panicked.downcast_ref::<&str>(), Some(&"its own message"));
    assert_eq!(
        timed_within(ms(5000), ms(30_000), "self-test value", || Err::<(), _>(
            "its value"
        )),
        Err("its value")
    );
}

/// An input past the slow limit but within the hang cap is waited for and
/// does not fail (it is only logged), so a loaded machine does not fail a
/// slow input. It used to fail at the slow limit.
#[test]
fn timed_waits_for_a_slow_input() {
    let start = Instant::now();
    let out = timed_within(ms(20), ms(30_000), "self-test slow", || {
        std::thread::sleep(ms(300));
        Err::<(), _>("finished")
    });
    assert_eq!(out, Err("finished"));
    assert!(start.elapsed() >= ms(300));
    assert_eq!(hung_here(), None);
}

/// Once an input of a property hangs, its shrink steps are not run: one
/// stuck thread per hung property, not one per shrink step, and the failure
/// shows the input that hung. It used to start a thread for each shrink step
/// that hung too.
#[test]
fn a_hung_input_is_not_shrunk() {
    // On a thread of its own, so the hang's mark stays off libtest's.
    std::thread::spawn(a_hung_input_is_not_shrunk_here)
        .join()
        .unwrap();
}

fn a_hung_input_is_not_shrunk_here() {
    let started = Arc::new(AtomicUsize::new(0));
    let first = Mutex::new(None);
    let mut cfg = config();
    // Its own count: PROPTEST_CASES=0 (skipping properties) runs none.
    cfg.cases = 16;
    cfg.failure_persistence = None;
    cfg.rng_seed = RngSeed::Fixed(7);
    cfg.max_shrink_iters = 64;
    let mut runner = TestRunner::new(cfg);
    let start = Instant::now();
    let result = runner.run(&(100u32..1_000_000), |x| {
        first.lock().unwrap().get_or_insert(x);
        let started = started.clone();
        timed_within(ms(20), ms(100), "self-test hangs", move || {
            started.fetch_add(1, SeqCst);
            std::thread::sleep(Duration::from_secs(10));
        });
        Ok(())
    });
    let first = first.lock().unwrap().expect("one input ran");
    assert_ne!(
        first, 100,
        "the first input cannot shrink; pick another seed"
    );
    match result {
        Err(TestError::Fail(why, value)) => {
            assert!(why.message().contains("did not finish"), "{why}");
            assert_eq!(value, first, "the failure shows a shrunk input");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(started.load(SeqCst), 1, "inputs started");
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "took {:?}",
        start.elapsed()
    );
}

/// A failing property saves its seed under the target folder, never beside
/// its source, and saves nothing when an input hung. Seeds used to be saved
/// as `<test file>.proptest-regressions` beside the test's source, hangs
/// included. The source here is a stand-in in a temporary folder.
#[test]
fn a_failing_case_writes_nothing_beside_its_source() {
    let tmp = tempfile::tempdir().unwrap();
    let src = tmp.path().join("src");
    std::fs::create_dir(&src).unwrap();
    let file = src.join("fuzz_stand_in.rs");
    std::fs::write(&file, "// stands in for a test file\n").unwrap();
    let source: &'static str = Box::leak(file.to_string_lossy().into_owned().into_boxed_str());
    let saved = persistence_dir().map(|dir| persistence_file(&dir, source));
    if let Some(saved) = &saved {
        let target = target_dir().expect("a target folder");
        assert!(
            saved.starts_with(&target),
            "{saved:?} is outside {target:?}"
        );
        assert!(!saved.starts_with(env!("CARGO_MANIFEST_DIR")), "{saved:?}");
        let _ = std::fs::remove_file(saved);
    }
    // An ordinary failure, a hang, and an ordinary failure whose first
    // shrink step hangs: only the plain hang saves nothing.
    for (hang, hang_shrinking) in [(false, false), (true, false), (false, true)] {
        let mut cfg = config();
        // Its own count: PROPTEST_CASES=0 (skipping properties) runs none.
        cfg.cases = 16;
        cfg.source_file = Some(source);
        cfg.rng_seed = RngSeed::Fixed(7);
        cfg.max_shrink_iters = 8;
        let first = Mutex::new(None::<u32>);
        // A thread of its own, so the hang's mark stays with this property.
        let result = std::thread::spawn(move || {
            TestRunner::new(cfg).run(&(100u32..1000), |x| {
                let shrinking = *first.lock().unwrap().get_or_insert(x) != x;
                let hangs = hang || (hang_shrinking && shrinking);
                // Only an input meant to hang gets the short cap.
                let cap = if hangs {
                    ms(100)
                } else {
                    Duration::from_secs(60)
                };
                timed_within(ms(20), cap, "self-test stand-in", move || {
                    if hangs {
                        std::thread::sleep(Duration::from_secs(10));
                    }
                    assert!(x < 100, "fails");
                });
                Ok(())
            })
        })
        .join()
        .unwrap();
        assert!(result.is_err(), "the stand-in property fails");
        let mut beside: Vec<_> = std::fs::read_dir(&src)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        beside.sort();
        assert_eq!(beside, ["fuzz_stand_in.rs"], "hang: {hang}");
        if let Some(saved) = &saved {
            assert_eq!(saved.is_file(), !hang, "hang: {hang}, {saved:?}");
            let _ = std::fs::remove_file(saved);
        }
    }
}

/// The seed file's name follows the test source's path, and the folder it
/// goes in is the test binary's target folder.
#[test]
fn seed_files_go_under_the_target_folder() {
    let dir = Path::new("/t/proptest-regressions");
    assert_eq!(
        persistence_file(dir, "crates/benchflow-task/tests/fuzz_parsers.rs"),
        dir.join("crates__benchflow-task__tests__fuzz_parsers.txt")
    );
    assert_eq!(
        persistence_file(dir, "/abs/src/acp/fuzz_tests.rs"),
        dir.join("abs__src__acp__fuzz_tests.txt")
    );
    let target = target_dir().expect("a target folder");
    let exe = std::env::current_exe().unwrap();
    assert!(exe.starts_with(&target), "{exe:?} is outside {target:?}");
    assert!(
        !target.starts_with(env!("CARGO_MANIFEST_DIR")),
        "{target:?}"
    );
}

/// Guards the D5 fix: a split task with a time limit too big for `Duration`
/// (`timeout_sec = 1e20`, which 0.8 loads) panicked while loading.
#[test]
fn a_huge_time_limit_loads() {
    for (key, value) in [
        ("timeout_sec", "1e20"),
        ("timeout_sec", "\"1e20\""),
        ("build_timeout_sec", "1e300"),
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("t");
        copy_dir(&fixtures().join("split/full"), &dir);
        let text = std::fs::read_to_string(dir.join("task.toml")).unwrap();
        let section = if key == "build_timeout_sec" {
            "[environment]"
        } else {
            "[verifier]"
        };
        let text = text.replacen(section, &format!("{section}\n{key} = {value}"), 1);
        let text = dedupe_key(&text, section, key);
        std::fs::write(dir.join("task.toml"), &text).unwrap();
        let _ = benchflow_task::load(&dir);
        let _ = benchflow_task::check(&dir, &CheckOptions::default());
    }
}

/// `text` with only the first `key = ...` line under `section` kept.
fn dedupe_key(text: &str, section: &str, key: &str) -> String {
    let mut out = Vec::new();
    let mut in_section = false;
    let mut seen = false;
    for line in text.lines() {
        if line.starts_with('[') {
            in_section = line.trim() == section;
        } else if in_section && line.split('=').next().is_some_and(|k| k.trim() == key) {
            if seen {
                continue;
            }
            seen = true;
        }
        out.push(line);
    }
    out.join("\n")
}
