//! Global flags against subcommands' own: `-C` and `bench new --dir` share a
//! field name, and `--json` must give every listing a JSON document.

use assert_cmd::Command;

fn bench(cwd: &std::path::Path) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(cwd)
        .env("BENCHFLOW_CACHE_DIR", cwd.join("cache"))
        .env("NO_COLOR", "1");
    c
}

#[test]
fn new_dir_is_not_taken_for_dash_c() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join("out")).unwrap();
    bench(tmp.path())
        .args(["new", "foo", "--dir", "out"])
        .assert()
        .success();
    assert!(tmp.path().join("out/foo/task.md").is_file());
    assert!(!tmp.path().join("out/out").exists());
}

#[test]
fn dash_c_still_changes_folder() {
    let tmp = tempfile::tempdir().unwrap();
    std::fs::create_dir(tmp.path().join("proj")).unwrap();
    bench(tmp.path())
        .args(["-C", "proj", "new", "bar"])
        .assert()
        .success();
    assert!(tmp.path().join("proj/bar/task.md").is_file());
}

#[test]
fn agents_json_lists_every_agent() {
    let tmp = tempfile::tempdir().unwrap();
    let out = bench(tmp.path())
        .args(["agents", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let doc: serde_json::Value = serde_json::from_slice(&out).expect("one JSON document");
    // One JSON array, an entry per agent (the shape the name guards read).
    let names: Vec<&str> = doc
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["name"].as_str().unwrap())
        .collect();
    assert!(
        names.contains(&"oracle") && names.contains(&"claude"),
        "{names:?}"
    );
}

#[test]
fn agents_json_shows_one_agent() {
    let tmp = tempfile::tempdir().unwrap();
    let out = bench(tmp.path())
        .args(["--json", "agents", "codex"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let doc: serde_json::Value = serde_json::from_slice(&out).expect("one JSON document");
    assert_eq!(doc["name"], "codex");
}

#[test]
fn a_judge_model_without_a_provider_is_refused_before_anything_runs() {
    let tmp = tempfile::tempdir().unwrap();
    bench(tmp.path()).args(["new", "t"]).assert().success();
    let out = bench(tmp.path())
        .args(["run", "t", "--judge-model", "nomodel", "--dry-run"])
        .assert()
        .code(2)
        .get_output()
        .stderr
        .clone();
    assert!(String::from_utf8_lossy(&out).contains("<provider>/<model>"));
    let out = bench(tmp.path())
        .args([
            "run",
            "t",
            "--judge-model",
            "vercel/deepseek/x",
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(!String::from_utf8_lossy(&out.stderr).contains("--judge-model"));
}

#[test]
fn zero_trials_is_refused_not_run_as_one() {
    let tmp = tempfile::tempdir().unwrap();
    bench(tmp.path()).args(["new", "t"]).assert().success();
    let out = bench(tmp.path())
        .args(["run", "t", "-k", "0", "--dry-run"])
        .assert()
        .code(2)
        .get_output()
        .stderr
        .clone();
    let out = String::from_utf8_lossy(&out);
    assert!(out.contains("-k 0 asks for no trials"), "{out}");
    assert!(out.contains("drop -k"), "{out}");
    std::fs::write(
        tmp.path().join("job.toml"),
        "targets = [\"t\"]\ntrials = 0\n",
    )
    .unwrap();
    let out = bench(tmp.path())
        .args(["run", "-c", "job.toml", "--dry-run"])
        .assert()
        .code(2)
        .get_output()
        .stderr
        .clone();
    let out = String::from_utf8_lossy(&out);
    assert!(
        out.contains("trials = 0 in the job file (job.toml)"),
        "{out}"
    );
    assert!(!out.contains("drop -k"), "{out}");
}

#[test]
fn agents_json_shows_native_external_and_dspy_agents() {
    let tmp = tempfile::tempdir().unwrap();
    for (name, harness) in [
        ("builtin", Some("native")),
        ("terminus-2", Some("native")),
        ("external", Some("backend")),
        ("dspy-rlm", None),
    ] {
        let out = bench(tmp.path())
            .args(["--json", "agents", name])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let doc: serde_json::Value = serde_json::from_slice(&out)
            .unwrap_or_else(|e| panic!("{name}: not one JSON document ({e}): {out:?}"));
        assert_eq!(doc["name"], name, "{doc}");
        if let Some(h) = harness {
            assert_eq!(doc["kind"], h, "{doc}");
        }
    }
}

#[test]
fn a_judge_model_with_an_unknown_provider_is_refused_before_anything_runs() {
    let tmp = tempfile::tempdir().unwrap();
    bench(tmp.path()).args(["new", "t"]).assert().success();
    let out = bench(tmp.path())
        .args(["run", "t", "--judge-model", "nosuchprovider/x", "--dry-run"])
        .assert()
        .code(2)
        .get_output()
        .stderr
        .clone();
    let out = String::from_utf8_lossy(&out);
    assert!(
        out.contains("no provider BenchFlow knows: `nosuchprovider`"),
        "{out}"
    );
    // A known provider passes the check whatever its key.
    bench(tmp.path())
        .env_remove("DEEPSEEK_API_KEY")
        .args([
            "run",
            "t",
            "--judge-model",
            "deepseek/deepseek-v4-flash",
            "--dry-run",
        ])
        .assert()
        .success();
}
