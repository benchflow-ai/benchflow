//! `bench export <job> --format hf|otel` against Harbor's own exporters.
//!
//! `tests/fixtures/export/expected/` was written by `make_expected.py` beside
//! it, which runs Harbor's `extract_conversations_from_trajectory`,
//! `openai_to_sharegpt`, and `harbor-atif2otel`'s `convert_trajectory` on the
//! same trials. Rows must hold the same values, and trace lines must match
//! byte for byte. The jobs: a synthetic one (two turns, a subagent,
//! reasoning, tool definitions, context management, an image, a trial with
//! no ATIF file), a job this engine wrote, and two jobs BenchFlow 0.8 wrote.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use benchflow_run::traces::{self, AtifSource, Episodes, ExportOptions};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn job(name: &str) -> PathBuf {
    match name {
        "synthetic" | "rust-claude" => fixtures().join("export/jobs").join(name),
        other => fixtures().join("trainer-0.8").join(other),
    }
}

fn expected(name: &str, kind: &str) -> String {
    std::fs::read_to_string(fixtures().join(format!("export/expected/{name}.{kind}.jsonl")))
        .unwrap()
        .replace("\"VERSION\"", &format!("\"{}\"", env!("CARGO_PKG_VERSION")))
}

fn lines(text: &str) -> Vec<Value> {
    text.lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

const JOBS: [(&str, Episodes); 4] = [
    ("synthetic", Episodes::All),
    ("rust-claude", Episodes::All),
    ("rb-claude-opus5-3", Episodes::Last),
    ("rb-codex-astra3", Episodes::All),
];

#[test]
fn hf_rows_match_harbors() {
    let work = tempfile::tempdir().unwrap();
    for (name, episodes) in JOBS {
        let out = work.path().join(name);
        let opts = ExportOptions {
            episodes,
            sharegpt: true,
            ..ExportOptions::default()
        };
        let rep = traces::export_hf(&job(name), &out, &opts).unwrap();
        let ours = lines(&std::fs::read_to_string(out.join("data/train.jsonl")).unwrap());
        let want = lines(&expected(name, "hf"));
        let theirs: BTreeSet<String> = want
            .iter()
            .map(|r| r["trial_name"].as_str().unwrap().to_string())
            .collect();
        let compared: Vec<&Value> = ours
            .iter()
            .filter(|r| theirs.contains(r["trial_name"].as_str().unwrap()))
            .collect();
        assert_eq!(compared.len(), want.len(), "{name}: as many rows");
        for (got, want) in compared.iter().zip(&want) {
            assert_eq!(
                *got, want,
                "{name}: {} {}",
                want["trial_name"], want["episode"]
            );
        }
        assert_eq!(rep.written, ours.len());
        let card = std::fs::read_to_string(out.join("README.md")).unwrap();
        assert!(
            card.starts_with("---\nconfigs:\n") && card.contains("path: data/train.jsonl"),
            "{card}"
        );
    }
}

#[test]
fn otel_lines_match_harbors_byte_for_byte() {
    let work = tempfile::tempdir().unwrap();
    for (name, _) in JOBS {
        let out = work.path().join(format!("{name}.jsonl"));
        let rep = traces::export_otel(&job(name), &out, &ExportOptions::default()).unwrap();
        let ours = std::fs::read_to_string(&out).unwrap();
        let want = expected(name, "otel");
        if name == "synthetic" {
            // Harbor's script reads only ATIF files; ours also rebuilds one.
            for line in want.lines() {
                assert!(
                    ours.lines().any(|l| l == line),
                    "{name}: a line of Harbor's is ours"
                );
            }
            assert_eq!(ours.lines().count(), want.lines().count() + 1);
        } else {
            assert!(
                ours == want,
                "{name}: the same bytes\n--- ours\n{}\n--- Harbor's\n{}",
                &ours[..ours.len().min(3000)],
                &want[..want.len().min(3000)]
            );
        }
        assert_eq!(rep.written, ours.lines().count());
    }
}

#[test]
fn synthetic_job_reports_every_trial() {
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("hf");
    let opts = ExportOptions {
        sharegpt: true,
        instruction: true,
        verifier_output: true,
        ..ExportOptions::default()
    };
    let rep = traces::export_hf(&job("synthetic"), &out, &opts).unwrap();
    let by_trial = |t: &str| rep.trials.iter().find(|x| x.trial == t).unwrap().clone();
    assert_eq!(by_trial("turns__00000001").atif, Some(AtifSource::Trainer));
    assert_eq!(by_trial("turns__00000001").written, 3);
    assert!(
        by_trial("picture__00000002")
            .skipped
            .as_deref()
            .unwrap()
            .contains("images")
    );
    assert!(
        by_trial("failed__00000003")
            .skipped
            .as_deref()
            .unwrap()
            .contains("no trajectory")
    );
    let rebuilt = by_trial("rebuilt__00000004");
    assert_eq!(rebuilt.atif, Some(AtifSource::Rebuilt));
    assert!(rebuilt.written > 0);
    let rows = lines(&std::fs::read_to_string(out.join("data/train.jsonl")).unwrap());
    let first = rows
        .iter()
        .find(|r| r["trial_name"] == "turns__00000001")
        .unwrap();
    assert_eq!(first["instruction"], "You are a careful agent.");
    assert_eq!(first["verifier_output"], "2 passed\n");
    assert_eq!(first["result"], "1.0");
    assert_eq!(first["model_provider"], "anthropic");
    let card = std::fs::read_to_string(out.join("README.md")).unwrap();
    assert!(
        card.contains("picture__00000002") && card.contains("conversations_sharegpt"),
        "{card}"
    );

    // Filters, last episodes, and an existing output.
    let opts = ExportOptions {
        filter: traces::Filter::Success,
        episodes: Episodes::Last,
        overwrite: true,
        ..ExportOptions::default()
    };
    let rep = traces::export_hf(&job("synthetic"), &out, &opts).unwrap();
    assert_eq!(rep.written, 2, "turns and rebuilt passed; one row each");
    let e = traces::export_hf(&job("synthetic"), &out, &ExportOptions::default()).unwrap_err();
    assert!(e.to_string().contains("already exists"), "{e}");
    let e = traces::export_otel(
        &fixtures().join("export"),
        &work.path().join("x.jsonl"),
        &ExportOptions::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("not a job folder"), "{e}");
}

fn copy_dir(from: &Path, to: &Path) {
    for e in walkdir::WalkDir::new(from) {
        let e = e.unwrap();
        let t = to.join(e.path().strip_prefix(from).unwrap());
        if e.file_type().is_dir() {
            std::fs::create_dir_all(&t).unwrap();
        } else {
            std::fs::copy(e.path(), &t).unwrap();
        }
    }
}

/// Guards 97e4cf1c (bench export): a shared job's links were followed, so
/// an export (a dataset meant to be published) could hold files of the
/// machine that exported it.
#[cfg(unix)]
#[test]
fn links_in_a_shared_job_are_not_exported() {
    use std::os::unix::fs::symlink;
    let work = tempfile::tempdir().unwrap();
    let secret = work.path().join("secret.txt");
    std::fs::write(&secret, "NOT-FOR-EXPORT\n").unwrap();
    let outside = work.path().join("outside");
    copy_dir(&job("rust-claude"), &outside);
    let atif = outside.join("hello__5af89adc/trainer/atif.json");
    let text = std::fs::read_to_string(&atif).unwrap();
    std::fs::write(&atif, text.replacen("\"hello", "\"NOT-FOR-EXPORT hello", 1)).unwrap();

    let shared = work.path().join("shared");
    copy_dir(&job("rust-claude"), &shared);
    let trial = shared.join("hello__5af89adc");
    std::fs::remove_file(trial.join("verifier/test-stdout.txt")).unwrap();
    symlink(&secret, trial.join("verifier/test-stdout.txt")).unwrap();
    // A trial folder that is a link to another job's trial.
    symlink(
        outside.join("hello__5af89adc"),
        shared.join("linked__00000000"),
    )
    .unwrap();
    let opts = ExportOptions {
        verifier_output: true,
        ..ExportOptions::default()
    };
    let out = work.path().join("out");
    let rep = traces::export_hf(&shared, &out, &opts).unwrap();
    let rows = std::fs::read_to_string(out.join("data/train.jsonl")).unwrap();
    assert!(rep.written > 0);
    assert!(!rows.contains("NOT-FOR-EXPORT"), "{rows}");

    // A trajectory file that is a link: the trial is skipped and says why.
    std::fs::remove_file(trial.join("trainer/atif.json")).unwrap();
    symlink(&atif, trial.join("trainer/atif.json")).unwrap();
    let out = work.path().join("out2");
    let e = traces::export_hf(&shared, &out, &opts);
    let text = std::fs::read_to_string(out.join("data/train.jsonl")).unwrap_or_default();
    assert!(!text.contains("NOT-FOR-EXPORT"), "{text}");
    match e {
        Ok(rep) => assert!(
            rep.trials.iter().any(|t| t
                .skipped
                .as_deref()
                .is_some_and(|s| s.contains("is a link"))),
            "{rep:?}"
        ),
        Err(e) => assert!(e.to_string().contains("skipped"), "{e}"),
    }
}

/// A job of `n` copies of rust-claude's trial, trial `i` run by agent
/// `agents[i % len]` on model `models[i % len]` ("" for none).
fn many_trials(at: &Path, n: usize, agents: &[&str], models: &[&str]) {
    let from = job("rust-claude");
    for f in ["lock.json", "summary.json"] {
        std::fs::copy(from.join(f), at.join(f)).unwrap();
    }
    for i in 0..n {
        let name = format!("hello__{i:08x}");
        let dir = at.join(&name);
        copy_dir(&from.join("hello__5af89adc"), &dir);
        let path = dir.join("result.json");
        let mut r: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        r["rollout_name"] = name.into();
        r["agent"] = agents[i % agents.len()].into();
        let model = models[i % models.len()];
        r["model"] = if model.is_empty() {
            Value::Null
        } else {
            model.into()
        };
        std::fs::write(&path, serde_json::to_string(&r).unwrap()).unwrap();
    }
}

/// The export reads a job one trial at a time (an export of 10,000 trials
/// held every result.json and peaked at 434 MB); the dataset card still
/// counts every trial and names each agent and model once, in order.
#[test]
fn card_counts_every_trial_agent_and_model() {
    let work = tempfile::tempdir().unwrap();
    let at = work.path().join("many");
    std::fs::create_dir(&at).unwrap();
    many_trials(
        &at,
        12,
        &["codex", "claude", "claude"],
        &["", "z/m2", "a/m1", "z/m2"],
    );
    let out = work.path().join("hf");
    let rep = traces::export_hf(&at, &out, &ExportOptions::default()).unwrap();
    assert_eq!(rep.trials.len(), 12);
    assert!(rep.trials.iter().all(|t| t.written > 0), "{:?}", rep.trials);
    let rows = lines(&std::fs::read_to_string(out.join("data/train.jsonl")).unwrap());
    assert_eq!(rows.len(), rep.written);
    let card = std::fs::read_to_string(out.join("README.md")).unwrap();
    assert!(
        card.contains("of 12 trials in the BenchFlow job `many`")
            && card.contains("Agents: `claude`, `codex`. Models: `a/m1`, `z/m2`."),
        "{card}"
    );
}

/// `cargo test --release -p benchflow-run --test trace_export -- --ignored --nocapture`:
/// time an export of 10,000 trials (`/usr/bin/time -l` on the binary gives
/// its peak memory; see the soak report).
#[test]
#[ignore = "benchmark: writes 10,000 trial folders"]
fn export_of_10000_trials() {
    let work = tempfile::tempdir().unwrap();
    let at = work.path().join("big");
    std::fs::create_dir(&at).unwrap();
    many_trials(&at, 10_000, &["claude"], &["anthropic/m"]);
    let started = std::time::Instant::now();
    let rep = traces::export_hf(&at, &work.path().join("hf"), &ExportOptions::default()).unwrap();
    eprintln!(
        "exported {} rows from 10,000 trials in {:?}",
        rep.written,
        started.elapsed()
    );
    assert_eq!(rep.trials.len(), 10_000);
}
