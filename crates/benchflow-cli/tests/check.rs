//! `bench check` through the binary, on a task named by a relative path.
//! Nothing here starts a sandbox.

use std::path::Path;

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

/// Diagnostics name files relative to their task's folder. Both the source
/// snippet and `--fix` once resolved them against the current folder, so a
/// task under `tasks/` showed no snippet and `--fix` failed as a bug.
#[test]
fn check_shows_the_line_and_fix_repairs_a_task_under_a_subfolder() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("tasks")).unwrap();
    bench(d.path())
        .args(["new", "tasks/fizzbuzz"])
        .assert()
        .success();
    let file = d.path().join("tasks/fizzbuzz/task.md");
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(text.contains("\ntimeout = "), "{text}");
    std::fs::write(&file, text.replacen("\ntimeout = ", "\ntimout = ", 1)).unwrap();

    bench(d.path())
        .arg("check")
        .assert()
        .failure()
        .stderr(predicate::str::contains("T201"))
        .stderr(predicate::str::contains("timout = "))
        .stderr(predicate::str::contains("tasks/fizzbuzz/task.md"));

    bench(d.path())
        .args(["check", "--fix"])
        .assert()
        .success()
        .stderr(predicate::str::contains("fixed tasks/fizzbuzz/task.md"));
    let fixed = std::fs::read_to_string(&file).unwrap();
    assert!(
        fixed.contains("\ntimeout = ") && !fixed.contains("timout"),
        "{fixed}"
    );

    bench(d.path()).arg("check").assert().success();
}

/// `bench new`'s next steps fit in 100 columns, and so does `bench check`'s
/// headline for a typo: its list of allowed keys moves to a note (which
/// wraps at a terminal's width, and not at all in a pipe).
#[test]
fn new_and_check_fit_in_100_columns() {
    let d = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(d.path().join("tasks")).unwrap();
    std::fs::create_dir_all(d.path().join("home")).unwrap();
    let out = bench(d.path())
        .env("HOME", d.path().join("home"))
        .args(["new", "fizzbuzz"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let file = d.path().join("tasks/fizzbuzz/task.md");
    let text = std::fs::read_to_string(&file).unwrap();
    std::fs::write(&file, text.replacen("\ntimeout = ", "\ntimout = ", 1)).unwrap();
    let check = bench(d.path()).arg("check").output().unwrap();
    let err = String::from_utf8_lossy(&check.stderr);
    assert!(
        err.contains("error[T201]: unknown key [agent] timout\n"),
        "{err}"
    );
    assert!(err.contains("note: [agent] holds budget,"), "{err}");
    for printed in [&out.stdout, &out.stderr] {
        for line in String::from_utf8_lossy(printed).lines() {
            assert!(line.chars().count() <= 100, "wider than 100: {line}");
        }
    }
}
