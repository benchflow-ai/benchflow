//! `bench run --skill`, `--no-task-skills`, and `--load-trajectory`: what is
//! checked before any sandbox starts. Nothing here starts a sandbox.

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
fn a_skill_folder_must_hold_skills() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("pack/licenses")).unwrap();
    bench(d.path())
        .args(["run", "-a", "claude", "--skill", "pack", "--dry-run"])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("--skill: pack holds folders without a SKILL.md: licenses")
                .and(predicate::str::contains(
                    "pass a folder that holds a SKILL.md",
                )),
        );
    bench(d.path())
        .args(["run", "-a", "claude", "--skill", "nope", "--dry-run"])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--skill: nope does not exist"));
}

#[test]
fn an_agent_with_no_skills_folder_cannot_take_skills() {
    let d = tempfile::tempdir().unwrap();
    let skill = d.path().join("csv");
    std::fs::create_dir_all(&skill).unwrap();
    std::fs::write(skill.join("SKILL.md"), "---\nname: csv\n---\n").unwrap();
    bench(d.path())
        .args(["run", "-a", "aider", "--skill", "csv", "--dry-run"])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("aider has no skills folder")
                .and(predicate::str::contains("cannot take --skill")),
        );
}

#[test]
fn a_session_file_is_checked_against_the_agents_format() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(d.path().join("session.jsonl"), "{\"type\":\"user\"}\n").unwrap();
    std::fs::write(d.path().join("trajectory.json"), "{}").unwrap();
    bench(d.path())
        .args([
            "run",
            "-a",
            "claude",
            "--load-trajectory",
            "session.jsonl",
            "--dry-run",
        ])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "`session.jsonl` is not a Claude Code session file named <session-id>.jsonl",
        ));
    bench(d.path())
        .args([
            "run",
            "-a",
            "claude",
            "--load-trajectory",
            "trajectory.json",
            "--dry-run",
        ])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "an ATIF trajectory (.json) cannot be loaded yet",
        ));
}

#[test]
fn agents_that_cannot_load_a_session_are_refused_with_a_next_step() {
    let d = tempfile::tempdir().unwrap();
    let file = d.path().join("0b9c8a52-6f3e-4f0e-9d7a-2a5c1e7b4d10.jsonl");
    std::fs::write(&file, "{\"type\":\"user\"}\n").unwrap();
    bench(d.path())
        .args(["run", "-a", "gemini", "--load-trajectory"])
        .arg(&file)
        .arg("--dry-run")
        .arg(hello())
        .assert()
        .code(2)
        .stderr(
            predicate::str::contains("gemini cannot load a session from a file on this harness")
                .and(predicate::str::contains("run -a claude or -a codex")),
        );
    bench(d.path())
        .args(["run", "-a", "oracle", "--load-trajectory"])
        .arg(&file)
        .arg("--dry-run")
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains("oracle is not one"));
}

#[test]
fn job_files_carry_skills_and_a_session() {
    let d = tempfile::tempdir().unwrap();
    std::fs::write(
        d.path().join("job.toml"),
        "agent = \"claude\"\nskills = [\"skills/a\"]\ntask-skills = false\nload-trajectory = \"s.jsonl\"\n",
    )
    .unwrap();
    let out = bench(d.path())
        .args(["run", "--skill", "skills/b", "--print-config", "-c"])
        .arg(d.path().join("job.toml"))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: toml::Table = toml::from_str(&String::from_utf8(out).unwrap()).unwrap();
    let skills: Vec<&str> = v["skills"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|s| s.as_str())
        .collect();
    assert_eq!(skills, ["skills/b"], "the flag's list replaces the file's");
    assert_eq!(v["task-skills"].as_bool(), Some(false));
    let loaded = PathBuf::from(v["load-trajectory"].as_str().unwrap());
    assert!(loaded.is_absolute(), "relative to the job file");
    assert!(loaded.ends_with("s.jsonl"));
    // Without the flag, the file's skills are the file's folder's.
    let out = bench(d.path())
        .args(["run", "--print-config", "-c"])
        .arg(d.path().join("job.toml"))
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: toml::Table = toml::from_str(&String::from_utf8(out).unwrap()).unwrap();
    let skill = PathBuf::from(v["skills"][0].as_str().unwrap());
    assert!(
        skill.is_absolute() && skill.ends_with("skills/a"),
        "{skill:?}"
    );
}

#[test]
fn bench_agents_says_where_skills_go_and_what_sessions_load() {
    let d = tempfile::tempdir().unwrap();
    bench(d.path())
        .args(["agents", "claude"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("copied into $HOME/.claude/skills").and(
                predicate::str::contains(
                    "on acp and headless with --load-trajectory: a Claude Code session file",
                ),
            ),
        );
    bench(d.path())
        .args(["agents", "aider"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("none: the agent takes no skills")
                .and(predicate::str::contains("no: --load-trajectory is refused")),
        );
    // Agents whose CLI needs more than the folder say how they find it.
    for (agent, says) in [
        (
            "kimi-code",
            "copied into $HOME/.kimi-code/skills; named to kimi with --skills-dir",
        ),
        (
            "mistral-vibe",
            "copied into $HOME/.vibe/skills; Vibe's global skills folder, also listed as `skill_paths`",
        ),
        (
            "eve",
            "copied into $HOME/.benchflow/eve-skills; then the runner copies each into the app's agent/skills/",
        ),
        ("muse-code", "copied into $HOME/.config/muse/skills"),
    ] {
        bench(d.path())
            .args(["agents", agent])
            .assert()
            .success()
            .stdout(predicate::str::contains(says));
    }
}

/// A scripted attempt runs to its task's timeout, so `--idle-timeout`,
/// `--resume-sessions`, `--max-tool-calls`, and `--max-tokens` reach only
/// model agents. Given with the oracle or a `script:`, they are named in a
/// note instead of being ignored without a word. Guards the E7 examples fix
/// on `rust/e7-examples`: a silent 60-second script under `--idle-timeout
/// 10s` ran to the end and passed, with nothing said.
#[test]
fn flags_for_model_agents_are_named_when_a_script_runs() {
    let d = tempfile::tempdir().unwrap();
    bench(d.path())
        .args(["run", "-s", "docker", "--dry-run", "--idle-timeout", "10s"])
        .arg(hello())
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "oracle runs as a script, so --idle-timeout has no effect: it reaches only model agents",
        ));
    std::fs::write(d.path().join("s.sh"), "#!/bin/bash\ntrue\n").unwrap();
    bench(d.path())
        .args(["run", "-a", "script:s.sh", "-s", "docker", "--dry-run"])
        .args(["--resume-sessions", "--max-tool-calls", "5", "--max-tokens", "100"])
        .arg(hello())
        .assert()
        .success()
        .stderr(predicate::str::contains(
            "script:s.sh runs as a script, so --resume-sessions, --max-tool-calls, --max-tokens have no effect: they reach only model agents",
        ));
    // Without those flags, nothing is said.
    bench(d.path())
        .args(["run", "-s", "docker", "--dry-run"])
        .arg(hello())
        .assert()
        .success()
        .stderr(predicate::str::contains("runs as a script").not());
}

/// `bench run --backend docker` (the word the docs use for a sandbox
/// backend) is refused, as before, and now says the sandbox goes in `-s`.
/// Guards the E7 examples fix on `rust/e7-examples`: the refusal only said
/// "drop --backend, or pass -a external".
#[test]
fn backend_naming_a_sandbox_points_at_dash_s() {
    let d = tempfile::tempdir().unwrap();
    for sandbox in ["docker", "daytona"] {
        bench(d.path())
            .args(["run", "--dry-run", "--backend", sandbox])
            .arg(hello())
            .assert()
            .code(2)
            .stderr(predicate::str::contains(
                "--backend is for the external agent, not `oracle`",
            ))
            .stderr(predicate::str::contains(format!(
                "to run in {sandbox}, pass -s {sandbox} (--sandbox)"
            )));
    }
    bench(d.path())
        .args(["run", "--dry-run", "--backend", "http://127.0.0.1:9"])
        .arg(hello())
        .assert()
        .code(2)
        .stderr(predicate::str::contains(
            "drop --backend, or pass -a external",
        ));
}
