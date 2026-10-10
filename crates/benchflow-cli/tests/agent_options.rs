//! `bench run --ak`: agent options are checked before any sandbox starts,
//! and job files carry them. Nothing here starts a sandbox.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;

fn bench(dir: &Path) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(dir)
        .env("BENCHFLOW_CACHE_DIR", dir.join("cache"))
        .env("NO_COLOR", "1")
        .env_remove("DAYTONA_API_KEY");
    c
}

fn hello() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchflow-run/tests/fixtures/tasks/hello")
}

#[test]
fn an_unknown_option_is_refused_with_a_suggestion_before_anything_runs() {
    let d = tempfile::tempdir().unwrap();
    bench(d.path())
        .args([
            "run",
            "-a",
            "claude",
            "--ak",
            "max_turn=3",
            "-s",
            "daytona",
            "--dry-run",
        ])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("agent `claude` has no option `max_turn`")
                .and(predicate::str::contains("did you mean `max_turns`?")),
        );
}

#[test]
fn a_wrong_value_or_harness_says_what_to_do() {
    let d = tempfile::tempdir().unwrap();
    bench(d.path())
        .args(["run", "-a", "claude", "--harness", "headless"])
        .args(["--agent-option", "max_turns=lots", "--dry-run"])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "option `max_turns` of claude takes a whole number from 1, not `lots`",
        ));
    bench(d.path())
        .args(["run", "-a", "claude", "--ak", "max_turns=3", "--dry-run"])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains("add --harness headless"));
    bench(d.path())
        .args([
            "run",
            "-a",
            "terminus-2",
            "--ak",
            "temperature=9",
            "--dry-run",
        ])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains("a number from 0 to 2, not `9`"));
    bench(d.path())
        .args(["run", "-a", "oracle", "--ak", "x=1", "--dry-run"])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains("apply to model agents"));
}

#[test]
fn job_files_carry_agent_options_and_flags_win() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(
        d.path().join("job.toml"),
        "agent = \"claude\"\nresume-sessions = true\n[agent-options]\nmax_turns = 30\nversion = \"2.1.280\"\n",
    )
    .unwrap();
    let out = bench(d.path())
        .args([
            "run",
            "-c",
            "job.toml",
            "--ak",
            "max_turns=40",
            "--print-config",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: toml::Table = toml::from_str(&String::from_utf8(out).unwrap()).unwrap();
    let options = v["agent-options"].as_table().unwrap();
    assert_eq!(options["max_turns"].as_str(), Some("40"), "the flag wins");
    assert_eq!(options["version"].as_str(), Some("2.1.280"));
    assert_eq!(v["resume-sessions"].as_bool(), Some(true));
}

#[test]
fn bench_agents_lists_an_agents_options_and_resume() {
    let d = tempfile::tempdir().unwrap();
    bench(d.path())
        .args(["agents", "claude"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("max_turns")
                .and(predicate::str::contains("flag --max-turns (headless only)"))
                .and(predicate::str::contains(
                    "installs @anthropic-ai/claude-code",
                ))
                .and(predicate::str::contains(
                    "on acp and headless with --resume-sessions",
                )),
        );
    bench(d.path())
        .args(["agents", "terminus-2"])
        .assert()
        .success()
        .stdout(predicate::str::contains("parser_name").and(predicate::str::contains("json, xml")));
    bench(d.path())
        .args(["agents", "kimi"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Options none"));
}
