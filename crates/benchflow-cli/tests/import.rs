//! `bench import` through the binary on Terminal-Bench v1 tasks, then
//! `bench check` on what it wrote. Nothing here starts a sandbox.

use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;

fn bench(dir: &Path) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(dir)
        .env("BENCHFLOW_CACHE_DIR", dir.join("cache"))
        .env("NO_COLOR", "1");
    c
}

fn tbench_inputs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../benchflow-task/tests/fixtures/tbench/input")
}

#[test]
fn imports_a_folder_of_terminal_bench_tasks_and_they_check_clean() {
    let d = tempfile::tempdir().unwrap();
    let out = d.path().join("out");
    bench(d.path())
        .arg("import")
        .arg(tbench_inputs())
        .arg(&out)
        .args(["--cpus", "2"])
        .assert()
        // yaml-solution cannot be converted; the other three are.
        .code(1)
        .stdout(predicate::str::contains("converted from Terminal-Bench v1"))
        .stdout(predicate::str::contains("rewrite it as solution.sh"))
        .stdout(predicate::str::contains(
            "moved into the Dockerfile: ENV MODE=test",
        ));
    for name in ["hello", "services", "settings"] {
        let md = std::fs::read_to_string(out.join(name).join("task.md")).unwrap();
        assert!(md.contains("cpus = 2"), "{name}: {md}");
    }
    assert!(!out.join("yaml-solution").exists());
    bench(d.path())
        .arg("check")
        .arg(&out)
        .assert()
        .success()
        .stdout(predicate::str::contains("3 ok"));
}
