//! `bench run` shows what the task reader ignored, as `bench check` does: a
//! setting in the wrong table must not be dropped in silence. Nothing here
//! starts a sandbox.

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

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dest = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dest);
        } else {
            std::fs::copy(e.path(), dest).unwrap();
        }
    }
}

/// The split-layout fixture, with `extra` added under `[agent]`.
fn split_task(root: &Path, extra: &str) -> PathBuf {
    let src = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../benchflow-run/tests/fixtures/tasks/hello-split");
    let dst = root.join("hello-split");
    copy_dir(&src, &dst);
    let toml = dst.join("task.toml");
    let body = std::fs::read_to_string(&toml).unwrap();
    std::fs::write(
        &toml,
        body.replace("[agent]\n", &format!("[agent]\n{extra}")),
    )
    .unwrap();
    dst
}

#[test]
fn a_network_setting_in_the_wrong_table_is_named_before_the_run() {
    let d = tempfile::tempdir().unwrap();
    let task = split_task(d.path(), "allow_internet = false\n");
    bench(d.path())
        .args(["run", "-a", "oracle", "--dry-run"])
        .arg(&task)
        .assert()
        .stderr(
            predicate::str::contains(
                "hello-split: `allow_internet` belongs in [environment], not [agent]",
            )
            .and(predicate::str::contains("[X101]")),
        );
}

#[test]
fn a_task_without_warnings_prints_none() {
    let d = tempfile::tempdir().unwrap();
    let task = split_task(d.path(), "");
    bench(d.path())
        .args(["run", "-a", "oracle", "--dry-run"])
        .arg(&task)
        .assert()
        .stderr(predicate::str::contains("X101").not());
}
