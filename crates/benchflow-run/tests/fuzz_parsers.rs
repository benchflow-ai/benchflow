//! Property tests over the run crate's readers of outside input: job files
//! (YAML, TOML, JSON), finished job folders (result.json, lock.json,
//! summary.json, config.json and the trajectories in them) as read by
//! `bench results`, regrade, diff and the trace exports, dataset registries
//! and refs, judge replies, and what sandbox programs print. None may panic
//! or take too long on any input.

#[path = "../../benchflow-task/tests/fuzz_support/mod.rs"]
mod fuzz_support;

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use benchflow_run::{
    config, confine, credentials, diff, hardening, jobfile, judge, read, registry,
};
use benchflow_task::rubric::Outcome;
use fuzz_support::*;
use proptest::prelude::*;
use serde_json::Value;

fn crates() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// Job folders among the fixtures: folders holding trial folders.
static JOB_DIRS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut out = Vec::new();
    for root in [
        "benchflow-viewer/tests/fixtures/jobs",
        "benchflow-run/tests/fixtures/export/jobs",
    ] {
        for (p, _) in files(&crates().join(root), &|p: &Path| p.ends_with("result.json")) {
            if let Some(job) = p.parent().and_then(Path::parent) {
                out.push(job.to_path_buf());
            }
        }
    }
    out.sort();
    out.dedup();
    out
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

const RESULT_KEYS: &[&str] = &[
    "task_name",
    "rollout_name",
    "trial_name",
    "rewards",
    "reward",
    "agent",
    "agent_name",
    "model",
    "n_tool_calls",
    "n_prompts",
    "agent_result",
    "final_metrics",
    "trajectory_summary",
    "error",
    "error_category",
    "verifier_error",
    "verifier_error_category",
    "started_at",
    "finished_at",
    "timing",
    "task_digest",
    "sandbox_id",
    "status",
    "owner",
    "problem",
    "attempt",
    "task_path",
    "task",
    "path",
    "config",
    "trials",
    "summary",
    "mean_reward",
    "pass_at_k",
    "n_trials",
    "agent_session",
    "usage_tracking",
    "partial_trajectory",
    "trajectory_source",
    "scenes",
    "loop",
    "version",
    "job_name",
    "tasks",
    "digest",
    "inputs_digest",
    "seed",
    "trial",
    "agent_timeout_info",
    "sessionUpdate",
    "type",
    "text",
    "content",
    "tool_call_id",
    "ts",
    "event",
    "duration_sec",
];

const JOB_KEYS: &[&str] = &[
    "targets",
    "dataset",
    "registry",
    "include",
    "exclude",
    "run",
    "agent",
    "harness",
    "model",
    "reasoning-effort",
    "sandbox",
    "concurrency",
    "trials",
    "jobs-dir",
    "timeout-multiplier",
    "agent-timeout-multiplier",
    "verifier-timeout-multiplier",
    "build-timeout-multiplier",
    "idle-timeout",
    "fail-under",
    "sandbox-user",
    "retries",
    "runtime",
    "env",
    "agent-options",
    "resume-sessions",
    "skills",
    "task-skills",
    "load-trajectory",
    "tasks",
    "agents",
];

/// A job file's text in one of its three formats, from random JSON.
fn job_text(v: &Value, fmt: u8) -> Option<(String, &'static str)> {
    match fmt {
        0 => Some((v.to_string(), "job.json")),
        1 => serde_yaml_ng::to_string(v).ok().map(|t| (t, "job.yaml")),
        _ => toml::to_string(v).ok().map(|t| (t, "job.toml")),
    }
}

proptest! {
    #![proptest_config(config())]

    /// Job files: random JSON of a job file's shape written as JSON, YAML or
    /// TOML, then edited as text; loaded alone and merged after a base.
    #[test]
    fn job_files(
        doc in json_with(JOB_KEYS),
        fmt in 0u8..3,
        edits in prop::collection::vec(any::<(u8, prop::sample::Index)>(), 0..3),
        noise in mutated(vec!["agent: nop\nconcurrency: 2\ninclude: [\"a*\"]\n".into(), "agent = \"nop\"\n[run]\ntrials = 3\n".into(), "{\"agent\": \"codex\", \"trials\": 2}".into(), "env:\n  X: on\n  Y: 012\ntrials: 1e3\n".into()]),
    ) {
        let tmp = tempfile::tempdir().unwrap();
        let mut paths = Vec::new();
        if let Some((text, name)) = job_text(&doc, fmt) {
            let mut text = text;
            for (k, at) in &edits {
                let i = at.index(text.len() + 1);
                if text.is_char_boundary(i) {
                    text.insert_str(i, TOKENS[*k as usize % TOKENS.len()]);
                }
            }
            let p = tmp.path().join(name);
            std::fs::write(&p, text).unwrap();
            paths.push(p);
        }
        let ext = ["job.yaml", "job.toml", "job.json"][edits.len() % 3];
        let p = tmp.path().join(format!("noise-{ext}"));
        std::fs::write(&p, &noise).unwrap();
        paths.push(p);
        timed("jobfile::load", || {
            for one in &paths {
                if let Ok(j) = jobfile::load(std::slice::from_ref(one)) {
                    let _ = j.to_toml();
                }
            }
            let _ = jobfile::load(&paths);
        });
    }

    /// A real job folder with one of its records swapped for edited text or
    /// random JSON of its shape, read by every reader of finished jobs.
    #[test]
    fn job_folder_readers(
        pick in any::<prop::sample::Index>(),
        target in any::<prop::sample::Index>(),
        as_text in any::<bool>(),
        doc in json_with(RESULT_KEYS),
        text in mutated(seeds(&["benchflow-viewer/tests/fixtures/jobs", "benchflow-run/tests/fixtures/export/jobs"], &|n| n.ends_with(".json") || n.ends_with(".jsonl"), 300)),
    ) {
        let src = pick.get(&JOB_DIRS);
        let tmp = tempfile::tempdir().unwrap();
        let job = tmp.path().join("job");
        copy_dir(src, &job);
        let records: Vec<PathBuf> = files(&job, &|p: &Path| {
            p.extension().is_some_and(|e| e == "json" || e == "jsonl" || e == "txt")
        }).into_iter().map(|(p, _)| p).collect();
        if !records.is_empty() {
            let file = target.get(&records);
            let body = if as_text { text.clone() } else { doc.to_string() };
            std::fs::write(file, body).unwrap();
        }
        timed("job readers", || {
            let view = read::read_job(&job);
            let _ = view.agent();
            let _ = view.model();
            let _ = read::read_lock(&job);
            let _ = read::find_jobs(tmp.path());
            for t in read::trial_dirs(&job) {
                if let Some(rec) = read::read_trial(&t) {
                    let _ = benchflow_run::traces::trial_atif(&t, &rec);
                    let _ = benchflow_run::trainer::read_events(&t);
                }
            }
            let _ = diff::load_sources(std::slice::from_ref(&job));
            let _ = benchflow_run::regrade::plan_with(&job, &|_| Err("not loaded in the fuzz test".into()));
            let _ = benchflow_run::regrade::saved_outputs(&job);
            let opts = benchflow_run::traces::ExportOptions::default();
            let _ = benchflow_run::traces::export_hf(&job, &tmp.path().join("hf"), &opts);
            let _ = benchflow_run::traces::export_otel(&job, &tmp.path().join("otel.jsonl"), &opts);
            let _ = benchflow_run::trainer::write_job(&job);
        });
    }

    /// result.json as random JSON straight into the record reader.
    #[test]
    fn result_json(doc in json_mutated(
        json_seeds(&["benchflow-viewer/tests/fixtures/jobs", "benchflow-run/tests/fixtures/export/jobs"], &|n| n == "result.json", 60),
        RESULT_KEYS,
    )) {
        timed("record_from_json", || { let _ = read::record_from_json(&doc, Path::new("/nonexistent/trial")); });
    }

    /// Dataset registries, dataset files and dataset refs.
    #[test]
    fn registries(
        doc in json_with(&["datasets", "name", "version", "description", "tasks", "path", "git_url", "git_commit_id", "url", "ref", "commit", "subdir", "digest", "registry", "id", "source"]),
        text in "\\PC{0,40}",
        spec in prop_oneof!["\\PC{0,30}", mutated(vec!["terminal-bench@2.0".into(), "skillsbench".into(), "org/name@1.2.3".into(), "https://github.com/o/r@main#sub".into()])],
    ) {
        let tmp = tempfile::tempdir().unwrap();
        let p = tmp.path().join("dataset.json");
        std::fs::write(&p, doc.to_string()).unwrap();
        let p2 = tmp.path().join("dataset.toml");
        std::fs::write(&p2, &text).unwrap();
        timed("registry", || {
            let _ = registry::parse_registry(&doc.to_string(), "fuzz");
            let _ = registry::parse_registry(&text, "fuzz");
            let _ = registry::read_dataset_file(&p);
            let _ = registry::read_dataset_file(&p2);
            let _ = registry::parse_ref(&spec);
            let _ = benchflow_run::source::parse_remote(&spec);
            let _ = config::Rerun::parse(&spec);
        });
    }

    /// A judge's reply, for each kind of criterion.
    #[test]
    fn judge_replies(
        scale in 0u8..3,
        reply in prop_oneof![
            json_with(&["verdict", "reason", "reasoning", "score", "level", "evidence", "quotes", "pass", "fail", "holds", "value", "criterion", "id"]).prop_map(|v| v.to_string()),
            mutated(vec![
                "{\"verdict\": \"pass\", \"reason\": \"ok\"}".into(),
                "```json\n{\"score\": 3, \"reason\": \"r\", \"evidence\": [\"a\"]}\n```".into(),
                "{\"level\": \"good\", \"reason\": \"x\"}".into(),
            ]),
        ],
    ) {
        let scale = match scale {
            0 => judge::Scale::Holds,
            1 => judge::Scale::Levels(vec![("good".into(), "fine".into()), ("bad".into(), "not".into())]),
            _ => judge::Scale::Range(0.0, 5.0),
        };
        let item = judge::Item {
            id: "c1".into(),
            text: "the answer is right".into(),
            outcome: Outcome::Good,
            scale,
            evidence: vec![],
            reference: None,
            guidance: None,
        };
        timed("parse_verdict", || { let _ = judge::parse_verdict(&item, &reply); });
    }

    /// What the sandbox-side programs print (`BF key value` lines).
    #[test]
    fn sandbox_reports(text in prop_oneof![
        "\\PC{0,200}",
        prop::collection::vec(("(BF )?", proptest::sample::select(&["memtotal", "mem", "credential", "env", "user", "done", "oom", "cgroup", "pids", "x"][..]), " ", "\\PC{0,30}"), 0..12)
            .prop_map(|v| v.into_iter().map(|(a, b, c, d)| format!("{a}{b}{c}{d}\n")).collect::<String>()),
    ]) {
        timed("sandbox reports", || {
            let _ = hardening::Report::parse(&text);
            let _ = confine::parse_memory(&text);
            let _ = confine::VerifierEnv::parse(&text);
            let _ = credentials::parse_scan(&text);
        });
    }
}
