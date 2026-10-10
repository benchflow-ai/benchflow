//! `bench handoff` through the binary, with a stand-in `claude` on `PATH`
//! that prints how it was started. Nothing here starts a sandbox.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::json;

fn write(p: &Path, text: &str) {
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, text).unwrap();
}

/// A trial folder a Claude Code run left, with its saved session.
fn trial(root: &Path, agent: &str, session: bool) -> std::path::PathBuf {
    let dir = root.join("jobs/j/hello__84c1190c");
    let mut result = json!({
        "task_name": "hello",
        "rollout_name": "hello__84c1190c",
        "agent": agent,
        "status": "scored",
        "rewards": {"reward": 1.0},
    });
    if session {
        result["agent_session"] = json!({
            "path": "agent/sessions", "dirs": [".claude/projects"],
            "files": 1, "bytes": 3, "session_id": "aaa",
        });
        write(
            &dir.join("agent/sessions/.claude/projects/-app/aaa.jsonl"),
            "{}\n",
        );
    }
    write(&dir.join("result.json"), &result.to_string());
    dir
}

fn bench(cwd: &Path, home: &Path, bin: &Path) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(cwd)
        .env("HOME", home)
        .env_remove("CLAUDE_CONFIG_DIR")
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("NO_COLOR", "1");
    c
}

#[test]
fn a_claude_trial_is_copied_under_its_id_and_the_cli_resumes_it_here() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    let trial = trial(&root, "claude", true);
    let bin = root.join("bin");
    write(
        &bin.join("claude"),
        "#!/bin/sh\necho \"claude started: $*\"\n",
    );
    std::fs::set_permissions(bin.join("claude"), std::fs::Permissions::from_mode(0o755)).unwrap();
    let home = root.join("home");
    let work = root.join("work");
    std::fs::create_dir_all(&work).unwrap();
    // From another folder the trial is not this project's, so it needs --trust.
    bench(&work, &home, &bin)
        .arg("handoff")
        .arg(&trial)
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "outside this project's jobs folder",
        ))
        .stderr(predicate::str::contains("runs with your permissions"))
        .stderr(predicate::str::contains("pass --trust"));
    assert!(!home.exists(), "nothing copied before --trust");
    bench(&work, &home, &bin)
        .arg("handoff")
        .arg(&trial)
        .arg("--trust")
        .args(["--", "--version"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "claude started: --resume aaa --version",
        ))
        .stderr(predicate::str::contains("resuming Claude Code session aaa"));
    let slug: String = work
        .to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect();
    let copied = home.join(".claude/projects").join(slug).join("aaa.jsonl");
    assert_eq!(std::fs::read_to_string(copied).unwrap(), "{}\n");
}

#[test]
fn dry_run_copies_nothing_and_prints_the_command() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    let trial = trial(&root, "claude:headless", true);
    let home = root.join("home");
    bench(&root, &home, &root.join("no-bin"))
        .arg("handoff")
        .arg(&trial)
        .arg("--dry-run")
        .assert()
        .success()
        .stdout("claude --resume aaa\n");
    assert!(!home.exists(), "nothing copied");
}

#[test]
fn a_missing_cli_and_an_unsaved_session_are_refused_with_a_next_step() {
    let d = tempfile::tempdir().unwrap();
    let root = d.path().canonicalize().unwrap();
    let trial = trial(&root, "claude", true);
    let home = root.join("home");
    bench(&root, &home, &root.join("no-bin"))
        .arg("handoff")
        .arg(&trial)
        .assert()
        .code(3)
        .stderr(predicate::str::contains("is not on PATH"))
        .stderr(predicate::str::contains(
            "npm install -g @anthropic-ai/claude-code",
        ));
    let d2 = tempfile::tempdir().unwrap();
    let old = trial_without_session(d2.path());
    bench(d2.path(), &home, &root.join("no-bin"))
        .arg("handoff")
        .arg(&old)
        .assert()
        .code(2)
        .stderr(predicate::str::contains("saved no Claude Code session"));
    bench(d2.path(), &home, &root.join("no-bin"))
        .arg("handoff")
        .arg(d2.path().join("jobs/j"))
        .assert()
        .code(2)
        .stderr(predicate::str::contains("not a trial folder"));
}

fn trial_without_session(root: &Path) -> std::path::PathBuf {
    trial(root, "claude", false)
}
