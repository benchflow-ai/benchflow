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
use std::sync::LazyLock;

use benchflow_task::check::CheckOptions;
use benchflow_task::ctrf::Ctrf;
use benchflow_task::rubric::Rubric;
use benchflow_task::{reward, taskmd, values};
use fuzz_support::*;
use proptest::prelude::*;
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
    "reward", "score", "accuracy", "passed", "total", "details", "metrics",
];

proptest! {
    #![proptest_config(config())]

    /// task.md text straight into the parser.
    #[test]
    fn taskmd_parse(text in mutated(seeds(&["benchflow-task/tests/fixtures"], &|n| n == "task.md", 400))) {
        timed("taskmd::parse", || {
            let doc = taskmd::parse(&text, Path::new("task.md"));
            let _ = doc.agent_view();
            let _ = doc.title();
            let _ = doc.told();
        });
    }

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
        timed("load + check", || {
            let _ = benchflow_task::load(&dir);
            let _ = benchflow_task::check(&dir, &CheckOptions::default());
            let _ = benchflow_task::detect(&dir);
        });
    }

    /// rubric.json: real rubrics with values swapped for random JSON.
    #[test]
    fn rubric_from_value(doc in json_mutated(
        json_seeds(&["benchflow-task/tests/fixtures"], &|n| n.starts_with("rubric") && n.ends_with(".json"), 200),
        RUBRIC_KEYS,
    )) {
        timed("Rubric::from_value", || {
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
        timed("Rubric::from_path", || { let _ = Rubric::from_path(&p); });
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
        timed("Ctrf::from_slice", || {
            let c = Ctrf::from_slice(&bytes);
            let _ = c.matching(&check);
            let _ = c.verdict(&check);
        });
    }

    #[test]
    fn ctrf_bytes(bytes in prop::collection::vec(any::<u8>(), 0..2048)) {
        timed("Ctrf::from_slice", || { let _ = Ctrf::from_slice(&bytes); });
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
            mutated(vec![r#"{"reward": 1.0}"#.into(), r#"{"reward": 0.5, "accuracy": 0.25, "details": {"a": 1}}"#.into(), r#"{"score": 1}"#.into()]),
        ]),
    ) {
        let tmp = tempfile::tempdir().unwrap();
        if let Some(t) = &txt { std::fs::write(tmp.path().join("reward.txt"), t).unwrap(); }
        if let Some(j) = &json { std::fs::write(tmp.path().join("reward.json"), j).unwrap(); }
        timed("read_rewards", || {
            if let Ok(r) = reward::read_rewards(tmp.path()) {
                prop_assert!((0.0..=1.0).contains(&r.reward), "a reward must be in [0, 1]: {r:?}");
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
        timed("values", || {
            if let Some(d) = values::parse_duration(&text) {
                let _ = values::format_duration(d);
            }
            let _ = values::parse_size_mb(&text);
            let _ = values::parse_bytes(&Value::String(text.clone()));
            let _ = values::parse_count(&Value::String(text.clone()));
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
        let inputs: Vec<PathBuf> = std::fs::read_dir(fixtures().join("tbench/input")).unwrap()
            .flatten().map(|e| e.path()).filter(|p| p.join("task.yaml").exists()).collect();
        let src = pick.get(&inputs);
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("in");
        copy_dir(src, &dir);
        let name = if which == 0 { "task.yaml" } else { "docker-compose.yaml" };
        std::fs::write(dir.join(name), &text).unwrap();
        let out = tmp.path().join("out");
        timed("import tbench", || {
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
        timed("import split", || {
            let _ = benchflow_task::convert::import_to_taskmd(&dir, &out, false);
        });
    }
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
