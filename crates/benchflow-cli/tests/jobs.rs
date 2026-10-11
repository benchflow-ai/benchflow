//! Job-level features through the binary: job files, schemas, datasets, and
//! regrade's planning. Nothing here starts a sandbox.

use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;

use assert_cmd::Command;
use predicates::prelude::*;

fn fixture_task() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchflow-run/tests/fixtures/tasks/hello")
}

fn copy_dir(from: &Path, to: &Path) {
    for e in walkdir::WalkDir::new(from) {
        let e = e.unwrap();
        let dest = to.join(e.path().strip_prefix(from).unwrap());
        if e.file_type().is_dir() {
            std::fs::create_dir_all(&dest).unwrap();
        } else {
            std::fs::copy(e.path(), &dest).unwrap();
        }
    }
}

fn bench(dir: &Path, cache: &Path) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(dir)
        .env("BENCHFLOW_CACHE_DIR", cache)
        .env("NO_COLOR", "1")
        .env_remove("DAYTONA_API_KEY");
    c
}

#[test]
fn job_files_merge_and_print_back() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("tasks")).unwrap();
    std::fs::write(
        d.path().join("base.toml"),
        "targets = [\"tasks\"]\nagent = \"nop\"\ninclude = [\"a*\"]\nconcurrency = 2\n",
    )
    .unwrap();
    std::fs::write(
        d.path().join("more.yml"),
        "include: [\"b*\"]\nconcurrency: 6\nverifier-timeout-multiplier: 3\n",
    )
    .unwrap();
    let out = bench(d.path(), &d.path().join("cache"))
        .args([
            "run",
            "-c",
            "base.toml",
            "-c",
            "more.yml",
            "-k",
            "4",
            "--print-config",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(out).unwrap();
    let v: toml::Table = toml::from_str(&text).unwrap();
    assert_eq!(v["agent"].as_str(), Some("nop"));
    assert_eq!(v["concurrency"].as_integer(), Some(6));
    assert_eq!(v["trials"].as_integer(), Some(4), "flags win");
    assert_eq!(v["include"].as_array().unwrap().len(), 2, "lists append");
    assert_eq!(v["verifier-timeout-multiplier"].as_float(), Some(3.0));
}

#[test]
fn bad_job_files_are_usage_errors() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("job.toml"), "agnet = \"nop\"\n").unwrap();
    bench(d.path(), &d.path().join("cache"))
        .args(["run", "-c", "job.toml", "--print-config"])
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("agnet").and(predicate::str::contains("bench schema job")),
        );
    bench(d.path(), &d.path().join("cache"))
        .args(["run", "--rerun", "passed", "--print-config"])
        .assert()
        .success();
    bench(d.path(), &d.path().join("cache"))
        .args(["run", "--rerun", "passed", "--dry-run"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("errored, unscored, or failed"));
}

/// A server on this machine that answers `/job.yaml` with a job file and
/// anything else with 404, for as many requests as come.
fn serve_job_file(body: &'static str) -> String {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    std::thread::spawn(move || {
        for mut sock in listener.incoming().flatten() {
            let mut buf = [0u8; 4096];
            let n = sock.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let (status, text) = if req.starts_with("GET /job.yaml ") {
                ("200 OK", body)
            } else {
                ("404 Not Found", "404: Not Found")
            };
            let _ = write!(
                sock,
                "HTTP/1.1 {status}\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            );
        }
    });
    format!("http://127.0.0.1:{port}")
}

#[test]
fn a_job_file_from_a_url_merges_and_a_missing_one_is_refused() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(
        d.path().join("base.toml"),
        "agent = \"nop\"\nconcurrency = 2\n",
    )
    .unwrap();
    let base = serve_job_file("concurrency: 5\ninclude: [\"b*\"]\n");
    let out = bench(d.path(), &d.path().join("cache"))
        .args([
            "run",
            "-c",
            "base.toml",
            "-c",
            &format!("{base}/job.yaml"),
            "--print-config",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: toml::Table = toml::from_str(&String::from_utf8(out).unwrap()).unwrap();
    assert_eq!(v["agent"].as_str(), Some("nop"));
    assert_eq!(v["concurrency"].as_integer(), Some(5));
    bench(d.path(), &d.path().join("cache"))
        .args(["run", "-c", &format!("{base}/nope.yaml"), "--print-config"])
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("could not be read").and(predicate::str::contains("HTTP 404")),
        );
}

#[test]
fn every_schema_is_json() {
    let d = tempfile::tempdir().unwrap();
    for kind in ["task", "job", "result", "lock"] {
        let out = bench(d.path(), &d.path().join("cache"))
            .args(["schema", kind])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert!(v["properties"].is_object(), "{kind}");
    }
}

/// A git repository holding the fixture task under `tasks/hello`, and a
/// registry that pins it.
fn registry_fixture(root: &Path) -> (PathBuf, String) {
    registry_fixture_pinned(root, None)
}

/// A git repository with one task, and a registry that pins it to the
/// repository's commit and to `digest` (default: the task's real digest).
fn registry_fixture_pinned(root: &Path, digest: Option<&str>) -> (PathBuf, String) {
    let repo = root.join("repo");
    copy_dir(&fixture_task(), &repo.join("tasks/hello"));
    let digest = match digest {
        Some(d) => d.to_string(),
        None => benchflow_task::digest::benchflow_digest(&repo.join("tasks/hello")).unwrap(),
    };
    let git = |args: &[&str]| {
        let out = StdCommand::new("git")
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .current_dir(&repo)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    };
    git(&["init", "-q"]);
    git(&["config", "uploadpack.allowFilter", "true"]);
    git(&["config", "uploadpack.allowAnySHA1InWant", "true"]);
    git(&["add", "."]);
    git(&["commit", "-q", "-m", "tasks"]);
    let sha = git(&["rev-parse", "HEAD"]);
    let registry = root.join("registry.json");
    std::fs::write(
        &registry,
        serde_json::json!([
            {"name": "demo", "version": "1.0", "description": "old", "tasks": []},
            {"name": "demo", "version": "1.1", "description": "the fixture", "tasks": [
                {"name": "hello", "git_url": format!("file://{}", repo.display()), "git_commit_id": sha, "path": "tasks/hello", "digest": digest}
            ]}
        ])
        .to_string(),
    )
    .unwrap();
    (registry, sha)
}

#[test]
fn datasets_resolve_from_a_registry_file_and_fetch_pinned() {
    if which("git").is_none() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (registry, sha) = registry_fixture(d.path());
    let cache = d.path().join("cache");
    // A dry run lists the tasks without fetching them.
    let out = bench(d.path(), &cache)
        .args(["--json", "run", "-d", "demo", "--registry"])
        .arg(&registry)
        .arg("--dry-run")
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let plan: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(plan["dataset"]["version"], "1.1", "the highest version");
    assert_eq!(plan["tasks"][0]["commit"], sha.as_str());
    assert!(
        !cache.join("sources").exists(),
        "nothing fetched on a dry run"
    );
    // A real run fetches the pinned commit, loads the task, and stops at the
    // sandbox (none is ready here).
    bench(d.path(), &cache)
        .args([
            "run",
            "-d",
            "demo@1.1",
            "-a",
            "nop",
            "-s",
            "daytona",
            "--registry",
        ])
        .arg(&registry)
        .assert()
        .code(3)
        .stderr(
            predicate::str::contains("fetching").and(predicate::str::contains(
                "1 of 1 pinned task match the registry",
            )),
        );
    let checkout = cache.join("sources/git").join(
        d.path()
            .join("repo")
            .strip_prefix("/")
            .unwrap_or(&d.path().join("repo")),
    );
    assert!(checkout.join(&sha).join("tasks/hello/task.md").is_file());
    // Unknown versions name the ones there are.
    bench(d.path(), &cache)
        .args(["run", "-d", "demo@9", "--dry-run", "--registry"])
        .arg(&registry)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("its versions there: 1.0, 1.1"));
    // A dataset and targets together are refused.
    bench(d.path(), &cache)
        .args(["run", "tasks", "-d", "demo@1.1", "--dry-run"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("one or the other"));
}

#[test]
fn a_dataset_plan_counts_the_selected_tasks_and_names_their_source() {
    let d = tempfile::tempdir().unwrap();
    let registry = d.path().join("registry.json");
    let commit = "0123456789abcdef0123456789abcdef01234567";
    std::fs::write(
        &registry,
        serde_json::json!([{"name": "demo", "version": "1.0", "tasks": [
            {"name": "a", "git_url": "https://example.com/r.git", "git_commit_id": commit, "path": "a", "digest": "sha256:aa"},
            {"name": "b", "git_url": "https://example.com/r.git", "git_commit_id": commit, "path": "b"},
            {"name": "c", "git_url": "https://example.com/r.git", "git_commit_id": commit, "path": "c"}
        ]}])
        .to_string(),
    )
    .unwrap();
    let cache = d.path().join("cache");
    // Selected tasks are counted as selected, not as "before --include".
    bench(d.path(), &cache)
        .args(["run", "-d", "demo", "-n", "2", "--dry-run", "--registry"])
        .arg(&registry)
        .assert()
        .success()
        .stdout(
            predicate::str::contains("2 tasks of 3 (after --include, --exclude, and --limit)")
                .and(predicate::str::contains("before --include").not())
                .and(predicate::str::contains(
                    "from     https://example.com/r.git at 0123456789ab",
                ))
                .and(predicate::str::contains("1 of 2 pinned by the registry")),
        );
    bench(d.path(), &cache)
        .args(["run", "-d", "demo", "-x", "a", "--dry-run", "--registry"])
        .arg(&registry)
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "none in the registry; lock.json records each task's digest",
        ));
}

#[test]
fn a_task_that_does_not_match_its_pinned_digest_is_refused() {
    if which("git").is_none() {
        return;
    }
    let d = tempfile::tempdir().unwrap();
    let (registry, _sha) = registry_fixture_pinned(d.path(), Some("sha256:00"));
    let cache = d.path().join("cache");
    bench(d.path(), &cache)
        .args([
            "run",
            "-d",
            "demo@1.1",
            "-a",
            "nop",
            "-s",
            "daytona",
            "--registry",
        ])
        .arg(&registry)
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("do not match the digests the registry pins")
                .and(predicate::str::contains("pinned sha256:00")),
        );
}

#[test]
fn a_local_dataset_toml_runs_its_tasks() {
    let d = tempfile::tempdir().unwrap();
    let ds = d.path().join("ds");
    copy_dir(&fixture_task(), &ds.join("hello"));
    std::fs::write(
        ds.join("dataset.toml"),
        "[dataset]\nname = \"mine\"\nversion = \"0.1\"\n\n[[tasks]]\nname = \"hello\"\n",
    )
    .unwrap();
    bench(d.path(), &d.path().join("cache"))
        .args(["run", "-d", "ds", "--dry-run"])
        .assert()
        .success()
        .stdout(predicate::str::contains("mine@0.1").and(predicate::str::contains("hello")));
}

/// A dataset dry run with no earlier job fetches nothing, yet still runs
/// the agent's preflight: with no login it names what a run would stop on
/// and exits 3 (not ready), as a real run would, with or without --json.
#[test]
fn a_dataset_dry_run_without_a_login_exits_not_ready() {
    let d = tempfile::tempdir().unwrap();
    let ds = d.path().join("ds");
    copy_dir(&fixture_task(), &ds.join("hello"));
    std::fs::write(
        ds.join("dataset.toml"),
        "[dataset]\nname = \"mine\"\nversion = \"0.1\"\n\n[[tasks]]\nname = \"hello\"\n",
    )
    .unwrap();
    let plan = |json: bool| {
        let mut c = bench(d.path(), &d.path().join("cache"));
        c.env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("HOME", d.path())
            .env("NO_COLOR", "1")
            .env("BENCHFLOW_CACHE_DIR", d.path().join("cache"));
        if json {
            c.arg("--json");
        }
        let out = c
            .args(["run", "-d", "ds", "-a", "claude", "--dry-run"])
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(3), "{out:?}");
        String::from_utf8(out.stdout).unwrap()
    };
    let out = plan(false);
    assert!(out.contains("mine@0.1"), "{out}");
    assert!(
        out.contains(
            "a run would stop here, before any sandbox starts: [R211] no login for agent `claude`"
        ),
        "{out}"
    );
    let out = plan(true);
    let doc: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(doc["dataset"]["name"], "mine", "{out}");
    // A script reading the plan learns why, as from a task's plan.
    assert_eq!(doc["not_ready"]["code"], "R211", "{out}");
    assert!(
        doc["not_ready"]["next"]
            .as_str()
            .is_some_and(|n| !n.is_empty()),
        "{out}"
    );
}

#[test]
fn regrade_lists_trials_without_outputs_and_how_to_fix_it() {
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("tasks/hello");
    copy_dir(&fixture_task(), &task);
    let trial = d.path().join("jobs/j1/hello__00000001");
    std::fs::create_dir_all(&trial).unwrap();
    std::fs::write(
        d.path().join("jobs/j1/evaluation.json"),
        r#"{"job_name": "j1"}"#,
    )
    .unwrap();
    std::fs::write(
        trial.join("result.json"),
        r#"{"task_name": "hello", "rollout_name": "hello__00000001", "agent": "nop", "status": "scored", "rewards": {"reward": 0.0}}"#,
    )
    .unwrap();
    std::fs::write(
        trial.join("config.json"),
        serde_json::json!({"task_path": task}).to_string(),
    )
    .unwrap();
    bench(d.path(), &d.path().join("cache"))
        .args(["regrade", "jobs/j1"])
        .assert()
        .code(1)
        .stderr(predicate::str::contains("nothing in"))
        .stdout(
            predicate::str::contains("saved no outputs")
                .and(predicate::str::contains("--save-outputs")),
        );
    bench(d.path(), &d.path().join("cache"))
        .args(["regrade", "tasks"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("is not a job folder"));
}

/// An earlier job of the oracle on `task`, as `bench run` writes it, with
/// one scored trial: its lock.json, result.json, and config.json.
fn earlier_oracle_job(job: &Path, task: &Path) {
    use benchflow_run::{Attempt, RunOptions, diff, lock};
    let t = std::sync::Arc::new(benchflow_task::load(task).unwrap());
    let planned = diff::planned_trials(&[(t.clone(), vec![Attempt::Oracle])], 1, "j1");
    let options = RunOptions {
        sandbox: "daytona".into(),
        ..RunOptions::default()
    };
    let l = lock::build(
        lock::LockHeader {
            job_name: "j1",
            fingerprint: "f",
            source: &serde_json::Value::Null,
            trials_per_task: 1,
            concurrency: 1,
            regrade_of: None,
        },
        &planned,
        &options,
    );
    let trial = job.join(&planned[0].name);
    std::fs::create_dir_all(&trial).unwrap();
    std::fs::write(job.join("lock.json"), serde_json::to_string(&l).unwrap()).unwrap();
    std::fs::write(
        trial.join("result.json"),
        serde_json::json!({"task_name": "hello", "rollout_name": planned[0].name, "agent": "oracle", "status": "scored", "rewards": {"reward": 1.0}, "task_digest": t.digest}).to_string(),
    )
    .unwrap();
    std::fs::write(
        trial.join("config.json"),
        serde_json::json!({"task_path": task}).to_string(),
    )
    .unwrap();
}

#[test]
fn diff_previews_reuse_and_rerun_against_an_earlier_job() {
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("tasks/hello");
    copy_dir(&fixture_task(), &task);
    earlier_oracle_job(&d.path().join("jobs/j1"), &task);
    let run = |extra: &[&str]| {
        let mut args = vec!["run", "tasks/hello", "-a", "oracle", "-s", "daytona"];
        args.extend_from_slice(extra);
        bench(d.path(), &d.path().join("cache"))
            .args(&args)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone()
    };
    // By name in the jobs folder.
    let out = String::from_utf8(run(&["--diff", "j1", "--dry-run"])).unwrap();
    assert!(out.contains("Reuse plan against 1 earlier job"), "{out}");
    assert!(out.contains("reuse    hello"), "{out}");
    assert!(
        out.contains("1 trial: 1 reuse · 0 regrade · 0 rerun"),
        "{out}"
    );
    // The tests change: the earlier trial saved no outputs, so it reruns.
    std::fs::write(task.join("verifier/test.sh"), "#!/bin/bash\nexit 1\n").unwrap();
    let out = run(&["--diff", "jobs/j1", "--dry-run", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["trials"][0]["action"], "rerun");
    assert_eq!(
        v["trials"][0]["reason"],
        "the task's tests or verifier changed, and the earlier trial saved no outputs to grade again"
    );
    // Settings that change the agent's run are part of the match.
    let out = String::from_utf8(run(&[
        "--diff",
        "j1",
        "--dry-run",
        "--agent-timeout-multiplier",
        "2",
    ]))
    .unwrap();
    assert!(
        out.contains("the agent timeout multiplier differs: 1 then, 2 now"),
        "{out}"
    );
}

#[test]
fn diff_refuses_jobs_it_cannot_read() {
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("tasks/hello");
    copy_dir(&fixture_task(), &task);
    bench(d.path(), &d.path().join("cache"))
        .args([
            "run",
            "tasks/hello",
            "-a",
            "oracle",
            "--diff",
            "nope",
            "--dry-run",
        ])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--diff names no job"));
    std::fs::create_dir_all(d.path().join("jobs/old/hello__1")).unwrap();
    bench(d.path(), &d.path().join("cache"))
        .args([
            "run",
            "tasks/hello",
            "-a",
            "oracle",
            "--diff",
            "old",
            "--dry-run",
        ])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("no lock.json"));
}

fn which(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(program))
        .find(|p| p.is_file())
}

/// Write a finished job of `rewards` (one trial each, task `t<i>`) with a
/// long model name.
fn write_job(root: &Path, name: &str, rewards: &[f64]) {
    std::fs::create_dir_all(root.join(format!("jobs/{name}"))).unwrap();
    std::fs::write(
        root.join(format!("jobs/{name}/evaluation.json")),
        serde_json::json!({ "job_name": name }).to_string(),
    )
    .unwrap();
    for (i, r) in rewards.iter().enumerate() {
        let trial = root.join(format!("jobs/{name}/t{i}__0000000{i}"));
        std::fs::create_dir_all(&trial).unwrap();
        std::fs::write(
            trial.join("result.json"),
            serde_json::json!({
                "task_name": format!("t{i}"), "rollout_name": format!("t{i}__0000000{i}"),
                "agent": "builtin", "model": "vercel/deepseek/deepseek-v4-flash-0731",
                "status": "scored", "passed": *r >= 1.0, "rewards": {"reward": r},
            })
            .to_string(),
        )
        .unwrap();
    }
}

/// With a long model name the job list and a comparison still fit in 100
/// columns: the list gives each job two lines instead of a table row.
#[test]
fn results_fit_in_100_columns_with_a_long_model_name() {
    let d = tempfile::tempdir().unwrap();
    write_job(d.path(), "2026-10-10__09-24-44", &[1.0, 1.0, 1.0]);
    write_job(d.path(), "2026-10-10__09-24-44-regrade-1", &[1.0, 0.0, 1.0]);
    let list = bench(d.path(), &d.path().join("cache"))
        .arg("results")
        .output()
        .unwrap();
    let compare = bench(d.path(), &d.path().join("cache"))
        .args([
            "results",
            "jobs/2026-10-10__09-24-44",
            "jobs/2026-10-10__09-24-44-regrade-1",
        ])
        .output()
        .unwrap();
    let list_out = String::from_utf8_lossy(&list.stdout);
    assert!(
        list_out.contains("builtin · vercel/deepseek/deepseek-v4-flash-0731"),
        "{list_out}"
    );
    assert!(
        list_out.contains("3 trials · 2 passed · 66.7%"),
        "{list_out}"
    );
    let compare_out = String::from_utf8_lossy(&compare.stdout);
    assert!(compare_out.contains("A better on 1"), "{compare_out}");
    for out in [list_out, compare_out] {
        for line in out.lines() {
            assert!(line.chars().count() <= 100, "wider than 100: {line}");
        }
    }
}

/// `bench results A B` gives a paired interval only when it means
/// something: none with too few shared tasks or no spread (the cases it
/// printed as `+1.000 to +1.000` and `-5.853 to +6.853` on 2026-10-10),
/// never one outside what rewards from 0 to 1 allow, and null bounds in
/// `--json` when there is none.
#[test]
fn results_give_a_paired_interval_only_when_it_means_something() {
    let d = tempfile::tempdir().unwrap();
    let compare = |a: &[f64], b: &[f64], json: bool| {
        write_job(d.path(), "a", a);
        write_job(d.path(), "b", b);
        let mut cmd = bench(d.path(), &d.path().join("cache"));
        cmd.args(["results", "jobs/a", "jobs/b"]);
        if json {
            cmd.arg("--json");
        }
        let out = String::from_utf8(cmd.output().unwrap().stdout).unwrap();
        std::fs::remove_dir_all(d.path().join("jobs")).unwrap();
        out
    };
    let out = compare(&[1.0, 1.0], &[0.0, 0.0], false);
    assert!(
        out.contains("+1.000 mean reward over 2 shared tasks (no interval: too few tasks)"),
        "{out}"
    );
    let out = compare(&[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], false);
    assert!(
        out.contains("(no interval: every task differs by the same amount)"),
        "{out}"
    );
    // Differences 1, 1, 0, 1, 1: the t-interval (+0.245 to +1.355) reaches
    // past the most a 0-to-1 reward can differ by.
    let out = compare(&[1.0; 5], &[0.0, 0.0, 1.0, 0.0, 0.0], false);
    assert!(out.contains("(95% CI +0.245 to +1.000)"), "{out}");
    let v: serde_json::Value = serde_json::from_str(&compare(&[1.0], &[0.0], true)).unwrap();
    let result = &v["paired"][0]["result"];
    assert!(
        result["low"].is_null() && result["high"].is_null(),
        "{result}"
    );
    assert_eq!(result["no_interval"], "too few tasks");
    assert_eq!(result["diff"], 1.0);
}

/// Trials the user stopped (Ctrl-C, R001) are left out of the pass rate
/// and listed apart, in the words the run's own summary uses; on
/// 2026-10-10 they read as `0/2` and "your setup's problem".
#[test]
fn results_leave_stopped_trials_out_of_the_pass_rate() {
    let d = tempfile::tempdir().unwrap();
    write_job(d.path(), "j", &[1.0, 1.0, 1.0]);
    for i in 3..5 {
        let trial = d.path().join(format!("jobs/j/t{i}__0000000{i}"));
        std::fs::create_dir_all(&trial).unwrap();
        std::fs::write(
            trial.join("result.json"),
            serde_json::json!({
                "task_name": "t9", "rollout_name": format!("t{i}__0000000{i}"),
                "agent": "builtin", "model": "vercel/deepseek/deepseek-v4-flash-0731",
                "status": "skipped",
                "problem": {"code": "R001", "owner": "you", "phase": "agent",
                    "message": "stopped before it finished",
                    "next": "rerun the same command to resume the job", "transient": false},
            })
            .to_string(),
        )
        .unwrap();
    }
    let out = bench(d.path(), &d.path().join("cache"))
        .args(["results", "jobs/j"])
        .output()
        .unwrap();
    let out = String::from_utf8(out.stdout).unwrap();
    assert!(
        out.contains("Score 3/3 passed · 100.0% · partial: 2 stopped"),
        "{out}"
    );
    assert!(
        out.contains("2 trials stopped (Ctrl-C, a kill, or a closed terminal), not counted in the pass rate [R001]"),
        "{out}"
    );
    assert!(
        !out.contains("t9"),
        "a task with only stopped trials has no row: {out}"
    );
    assert!(!out.contains("problem"), "{out}");
    // lock.json planned two more that never started (a drain).
    std::fs::write(
        d.path().join("jobs/j/lock.json"),
        serde_json::json!({"trials": (0..7).map(|i| serde_json::json!({"trial": format!("t{i}__0000000{i}")})).collect::<Vec<_>>()}).to_string(),
    )
    .unwrap();
    let again = bench(d.path(), &d.path().join("cache"))
        .args(["results", "jobs/j"])
        .output()
        .unwrap();
    let again = String::from_utf8(again.stdout).unwrap();
    assert!(again.contains("2 trials planned never ran"), "{again}");
    // --json counts the same trials as the text.
    let json = bench(d.path(), &d.path().join("cache"))
        .args(["results", "jobs/j", "--json"])
        .output()
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(v["stopped"].as_u64(), Some(2), "{v}");
    assert_eq!(v["never_ran"].as_u64(), Some(2), "{v}");
    assert_eq!(v["partial"], true, "{v}");
    assert_eq!(v["pass_rate"]["n"].as_u64(), Some(3), "{v}");
    assert_eq!(v["pass_rate"]["value"].as_f64(), Some(1.0), "{v}");
    // The list of jobs (two, so it is a list) counts them apart too.
    write_job(d.path(), "k", &[1.0]);
    let list = bench(d.path(), &d.path().join("cache"))
        .arg("results")
        .output()
        .unwrap();
    let list = String::from_utf8(list.stdout).unwrap();
    // The list says what never ran, and the rate is partial (no interval).
    assert!(
        list.contains("3 trials + 2 stopped + 2 never ran · 3 passed · 100.0% partial"),
        "{list}"
    );
}

/// `bench view --port N` with N taken says the port is taken, not that the
/// folder given is wrong.
#[test]
fn view_on_a_taken_port_says_so() {
    let d = tempfile::tempdir().unwrap();
    write_job(d.path(), "j", &[1.0]);
    let trajectory = d.path().join("jobs/j/t0__00000000/trajectory");
    std::fs::create_dir_all(&trajectory).unwrap();
    std::fs::write(trajectory.join("acp_trajectory.jsonl"), "").unwrap();
    let held = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = held.local_addr().unwrap().port().to_string();
    let out = bench(d.path(), &d.path().join("cache"))
        .args(["view", "jobs", "--no-open", "--port", &port])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8(out.stderr).unwrap();
    assert!(err.contains(&format!("port {port} is taken")), "{err}");
    assert!(!err.contains("pass a job folder"), "{err}");
}

/// A job a live run holds, with no trial finished yet, reads as running
/// with its planned agent, not as `? · – · 0 trials`.
#[test]
fn results_show_a_running_job_as_running() {
    let d = tempfile::tempdir().unwrap();
    write_job(d.path(), "done", &[1.0]);
    let job = d.path().join("jobs/live");
    std::fs::create_dir_all(&job).unwrap();
    std::fs::write(job.join("evaluation.json"), "{}").unwrap();
    std::fs::write(
        job.join("lock.json"),
        serde_json::json!({"trials": [{"trial": "t__1", "task": {}, "image": {}, "sandbox": "daytona",
            "attempt": {"agent": "codex", "model": "gpt-6.1-sol", "kind": "model"}}]})
        .to_string(),
    )
    .unwrap();
    let lock = std::fs::File::create(job.join(".bench-run.lock")).unwrap();
    lock.lock().unwrap();
    let out = bench(d.path(), &d.path().join("cache"))
        .arg("results")
        .output()
        .unwrap();
    let out = String::from_utf8(out.stdout).unwrap();
    let line = out
        .lines()
        .find(|l| l.contains("jobs/live"))
        .unwrap_or_default()
        .to_string();
    assert!(line.contains("codex"), "{out}");
    assert!(out.contains("running"), "{out}");
    assert!(!line.contains('?'), "{out}");
}

#[test]
fn export_writes_a_jobs_traces_and_refuses_mixed_flags() {
    let d = tempfile::tempdir().unwrap();
    let job = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../benchflow-run/tests/fixtures/export/jobs/synthetic");
    let out = d.path().join("hf");
    bench(d.path(), &d.path().join("cache"))
        .arg("export")
        .arg(&job)
        .arg(&out)
        .arg("--sharegpt")
        .assert()
        .success()
        .stdout(predicate::str::contains("rows from 2 of 4 trials"))
        .stdout(predicate::str::contains(
            "picture__00000002: the trajectory has images",
        ))
        // The output's path as seen from here, not an absolute one.
        .stdout(predicate::str::contains("datasets.load_dataset(\"hf\")"));
    let rows = std::fs::read_to_string(out.join("data/train.jsonl")).unwrap();
    assert!(rows.contains("conversations_sharegpt"));
    // One trial and one row read as such, here and in the dataset card.
    let one = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../benchflow-run/tests/fixtures/export/jobs/rust-claude");
    bench(d.path(), &d.path().join("cache"))
        .arg("export")
        .arg(&one)
        .arg(d.path().join("one"))
        .args(["--episodes", "last"])
        .assert()
        .success()
        .stdout(predicate::str::contains("1 row from 1 of 1 trial\n"));
    let card = std::fs::read_to_string(d.path().join("one/README.md")).unwrap();
    assert!(
        card.contains("1 row from 1 of 1 trial in the BenchFlow job"),
        "{card}"
    );
    assert!(out.join("README.md").is_file());
    bench(d.path(), &d.path().join("cache"))
        .arg("export")
        .arg(&job)
        .args(["--format", "otel", "traces"])
        .assert()
        .success()
        .stdout(predicate::str::contains("3 traces"));
    assert!(
        d.path().join("traces.jsonl").is_file(),
        "otel output gets .jsonl"
    );
    bench(d.path(), &d.path().join("cache"))
        .arg("export")
        .arg(&job)
        .arg(&out)
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("already exists").and(predicate::str::contains("--overwrite")),
        );
    bench(d.path(), &d.path().join("cache"))
        .arg("export")
        .arg(&job)
        .args(["--format", "otel", "--sharegpt"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--sharegpt shapes dataset rows"));
    bench(d.path(), &d.path().join("cache"))
        .arg("export")
        .arg(fixture_task())
        .args(["--format", "hf"])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("is not a job folder"));
}

/// `bench results` shows each trial's and the job's model cost when the
/// trials recorded one, and "not recorded" (never $0) when they did not.
#[test]
fn results_show_the_cost_the_trials_recorded() {
    let d = tempfile::tempdir().unwrap();
    let job = d.path().join("jobs/j");
    for (task, cost) in [("a", Some(0.1)), ("b", Some(0.04)), ("c", None)] {
        let trial = job.join(format!("{task}__00000001"));
        std::fs::create_dir_all(&trial).unwrap();
        std::fs::write(
            trial.join("result.json"),
            serde_json::json!({
                "task_name": task, "agent": "claude", "model": "vercel/x",
                "status": "scored", "rewards": {"reward": 1.0},
                "agent_result": {"n_input_tokens": 1000, "n_output_tokens": 50, "cost_usd": cost},
            })
            .to_string(),
        )
        .unwrap();
    }
    std::fs::write(job.join("evaluation.json"), "{}").unwrap();
    let results = |extra: &[&str]| {
        let out = bench(d.path(), &d.path().join("cache"))
            .arg("results")
            .arg(&job)
            .args(extra)
            .output()
            .unwrap();
        String::from_utf8(out.stdout).unwrap()
    };
    let out = results(&[]);
    assert!(
        out.contains("Usage  3.0K in · 150 out · $0.14 (2 of 3 trials recorded a cost)"),
        "{out}"
    );
    assert!(
        out.contains("$0.10") && out.contains("not recorded"),
        "{out}"
    );
    assert!(!out.contains("$0.00"), "{out}");
    let out = results(&["--trials"]);
    assert!(
        out.contains("$0.04") && out.contains("not recorded"),
        "{out}"
    );
    let v: serde_json::Value = serde_json::from_str(&results(&["--json"])).unwrap();
    assert_eq!(v["cost_recorded"], 2);
    assert!((v["cost_usd"].as_f64().unwrap() - 0.14).abs() < 1e-9);
}

/// `--dry-run` says whether the run would resume a job, by the rule the
/// run uses, and how many of its trials it would keep.
#[test]
fn a_dry_run_says_whether_it_would_resume_a_job() {
    use benchflow_run::Attempt;
    use benchflow_run::job::{JobSpec, fingerprint, trial_name};
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("tasks/hello");
    copy_dir(&fixture_task(), &task);
    let dry = |extra: &[&str]| {
        let mut args = vec![
            "run",
            "tasks/hello",
            "-a",
            "oracle",
            "-s",
            "daytona",
            "--dry-run",
        ];
        args.extend_from_slice(extra);
        let out = bench(d.path(), &d.path().join("cache"))
            .args(&args)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        String::from_utf8(out).unwrap()
    };
    assert!(dry(&[]).contains("job      would start a new job"));
    // An earlier job of this command, with its one trial finished.
    let t = std::sync::Arc::new(benchflow_task::load(&task).unwrap());
    let spec = JobSpec {
        carry: Vec::new(),
        units: vec![(t.clone(), vec![Attempt::Oracle])],
        trials: 1,
        jobs_dir: d.path().join("jobs"),
        job_name: None,
        fresh: false,
        concurrency: 1,
        source: serde_json::Value::Null,
    };
    let job = d.path().join("jobs/2026-10-10__09-00-00");
    std::fs::create_dir_all(&job).unwrap();
    std::fs::write(
        job.join("evaluation.json"),
        serde_json::json!({"engine": {"fingerprint": fingerprint(&spec)}}).to_string(),
    )
    .unwrap();
    let out = dry(&[]);
    assert!(
        out.contains("job      would resume jobs/2026-10-10__09-00-00 (0 of 1 trial kept)"),
        "{out}"
    );
    let name = trial_name("2026-10-10__09-00-00", &t, &Attempt::Oracle, 0);
    std::fs::create_dir_all(job.join(&name)).unwrap();
    std::fs::write(
        job.join(&name).join("result.json"),
        serde_json::json!({"task_name": "hello", "rollout_name": name, "agent": "oracle", "status": "scored", "rewards": {"reward": 1.0}, "task_digest": t.digest}).to_string(),
    )
    .unwrap();
    let out = dry(&[]);
    assert!(out.contains("(1 of 1 trial kept)"), "{out}");
    let v: serde_json::Value = serde_json::from_str(&dry(&["--json"])).unwrap();
    assert_eq!(v["job"]["resume"], true);
    assert_eq!(v["job"]["kept"], 1);
    assert!(dry(&["--fresh"]).contains("would start a new job"));
    assert!(dry(&["--job", "other"]).contains("would start a new job"));
}

/// Guards the D5 fix: 0.8 writes `started_at` as a naive local time, and in
/// the hour repeated when clocks go back the time has two readings; Rust
/// took that as no time at all and showed the trial as started in 1970.
#[test]
fn a_08_start_time_in_the_repeated_hour_reads() {
    let d = tempfile::tempdir().unwrap();
    let job = d.path().join("jobs/old");
    let trial = job.join("a__00000001");
    std::fs::create_dir_all(&trial).unwrap();
    std::fs::write(
        trial.join("result.json"),
        serde_json::json!({
            "task_name": "a", "agent": "claude", "rewards": {"reward": 1.0},
            "started_at": "2026-11-01 01:30:00.250000",
            "finished_at": "2026-11-01 01:59:00.000000",
        })
        .to_string(),
    )
    .unwrap();
    std::fs::write(job.join("summary.json"), "{}").unwrap();
    let out = bench(d.path(), &d.path().join("cache"))
        .env("TZ", "America/Los_Angeles")
        .args(["results", "--trials", "--json"])
        .arg(&job)
        .output()
        .unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("2026-11-01"), "{text}");
    assert!(!text.contains("1969") && !text.contains("1970"), "{text}");
}
