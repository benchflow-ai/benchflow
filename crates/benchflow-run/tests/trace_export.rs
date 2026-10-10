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
