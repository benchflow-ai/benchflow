//! `bench cache` through the binary, against a fake `docker` CLI on PATH
//! and a temporary cache folder. No container engine is touched.

use std::os::unix::fs::PermissionsExt;
use std::path::Path;

use assert_cmd::Command;

/// A fake `docker` that lists two BenchFlow images (one used long ago, one
/// just now, by the last-use record) and records every call.
fn fake_docker(bin: &Path) {
    std::fs::create_dir_all(bin).unwrap();
    let script = r#"#!/bin/sh
dir=$(cd "$(dirname "$0")" && pwd)
echo "$*" >> "$dir/calls"
case "$1 $2" in
"image ls") printf 'sha256:old\nsha256:new\n';;
"image inspect") printf 'sha256:old\t2000000000\t2026-01-01T00:00:00Z\tbenchflow/hello:abc\nsha256:new\t1000\t2026-01-01T00:00:00Z\tbenchflow/fresh:def\n';;
esac
exit 0
"#;
    let p = bin.join("docker");
    std::fs::write(&p, script).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
}

fn bench(home: &Path, cache: &Path, bin: &Path) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(home)
        .env("BENCHFLOW_CACHE_DIR", cache)
        .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
        .env("NO_COLOR", "1")
        .env("COLUMNS", "160");
    c
}

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, body).unwrap();
}

#[test]
fn cache_lists_cleans_only_with_remove_and_never_touches_what_is_not_ours() {
    let d = tempfile::tempdir().unwrap();
    let cache = d.path().join("cache");
    let bin = d.path().join("bin");
    fake_docker(&bin);
    write(
        &cache,
        "sources/github.com/org/repo/main/tasks/a/task.md",
        "aaaa",
    );
    write(&cache, "sources/github.com/org/repo/main/.git/HEAD", "");
    write(&cache, "logs/crash-1.log", "boom");
    write(&cache, "task-formats/robouse/x", "not ours");
    let now = chrono::Utc::now().to_rfc3339();
    write(
        &cache,
        "docker-image-use.json",
        &format!(r#"{{"sha256:new": "{now}"}}"#),
    );

    let out = bench(d.path(), &cache, &bin)
        .args(["cache"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("benchflow/hello:abc"), "{text}");
    assert!(text.contains("2.0 GB"), "{text}");
    assert!(text.contains("github.com/org/repo@main"), "{text}");
    assert!(text.contains("left alone: task-formats"), "{text}");
    assert!(text.contains("benchflow/fresh:def"), "{text}");
    assert!(
        !text.contains("kept:") && !text.contains('!'),
        "a plain list keeps nothing back, so it marks nothing: {text}"
    );

    // A dry run removes nothing.
    let out = bench(d.path(), &cache, &bin)
        .args(["cache", "clean"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("Would remove 3 items"), "{text}");
    assert!(text.contains("kept: used in the last 30 minutes"), "{text}");
    assert!(cache.join("sources/github.com/org/repo/main").exists());
    let calls = std::fs::read_to_string(bin.join("calls")).unwrap();
    assert!(!calls.contains("rmi"), "{calls}");

    // --remove removes what was listed, by tag, never forced; the image
    // used minutes ago and the folder that is not ours stay.
    bench(d.path(), &cache, &bin)
        .args(["cache", "clean", "--remove"])
        .assert()
        .success();
    let calls = std::fs::read_to_string(bin.join("calls")).unwrap();
    let rmi: Vec<&str> = calls.lines().filter(|l| l.starts_with("rmi")).collect();
    assert_eq!(rmi, ["rmi benchflow/hello:abc"]);
    assert!(!cache.join("sources/github.com").exists());
    assert!(!cache.join("logs").exists());
    assert!(cache.join("task-formats/robouse/x").exists());
    assert!(cache.join("docker-image-use.json").exists());
}

#[test]
fn cache_filters_by_kind_and_age() {
    let d = tempfile::tempdir().unwrap();
    let cache = d.path().join("cache");
    let bin = d.path().join("bin");
    fake_docker(&bin);
    write(&cache, "logs/crash-1.log", "boom");
    write(&cache, "registries/abc.json", "{}");
    let out = bench(d.path(), &cache, &bin)
        .args(["cache", "clean", "--only", "logs", "--remove", "--json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(v["removed"].as_array().unwrap().len(), 1);
    assert!(!cache.join("logs").exists());
    assert!(cache.join("registries/abc.json").exists());
    assert!(!bin.join("calls").exists(), "--only logs asks no daemon");
    // Nothing was used 30 days ago.
    let out = bench(d.path(), &cache, &bin)
        .args([
            "cache",
            "clean",
            "--only",
            "registries",
            "--older-than",
            "30d",
            "--json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let v: serde_json::Value = serde_json::from_slice(&out).unwrap();
    assert!(v["items"].as_array().unwrap().is_empty());
    bench(d.path(), &cache, &bin)
        .args(["cache", "--only", "nope"])
        .assert()
        .code(2);
}
