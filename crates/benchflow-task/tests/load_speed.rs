//! Discovering and checking a thousand packages is fast: the design asks
//! for well under a second. Checking includes each package's digest, which
//! reads every file, and runs on every core as `bench check` runs it. Debug
//! builds are several times slower than release, so the bound here is
//! loose; `--release` prints the real time.

use std::fs;
use std::path::Path;
use std::time::Instant;

use benchflow_task::CheckOptions;

fn write(root: &Path, rel: &str, body: &str) {
    let p = root.join(rel);
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(p, body).unwrap();
}

const TOML: &str = "version = \"1.0\"\n\n[metadata]\ndifficulty = \"medium\"\ntags = [\"a\", \"b\"]\n\n[verifier]\ntimeout_sec = 300.0\n\n[agent]\ntimeout_sec = 1800.0\n\n[environment]\nbuild_timeout_sec = 600.0\ncpus = 2\nmemory_mb = 4096\nstorage_mb = 10240\n";
const MD: &str = "---\nschema_version: \"1.3\"\nmetadata:\n  difficulty: medium\n  tags: [a, b]\nagent:\n  timeout_sec: 900\nverifier:\n  timeout_sec: 120\nsandbox:\n  cpus: 1\n  memory_mb: 2048\n---\n\n## prompt\n\nDo the thing, carefully.\n";

#[test]
fn a_thousand_packages_check_quickly() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..500 {
        let t = format!("split/group-{}/task-{i}", i % 10);
        write(dir.path(), &format!("{t}/task.toml"), TOML);
        write(
            dir.path(),
            &format!("{t}/instruction.md"),
            "Do the thing.\n",
        );
        write(
            dir.path(),
            &format!("{t}/environment/Dockerfile"),
            "FROM ubuntu:24.04\n",
        );
        write(dir.path(), &format!("{t}/tests/test.sh"), "#!/bin/bash\n");
        write(
            dir.path(),
            &format!("{t}/solution/solve.sh"),
            "#!/bin/bash\n",
        );
        let m = format!("v06/group-{}/task-{i}", i % 10);
        write(dir.path(), &format!("{m}/task.md"), MD);
        write(
            dir.path(),
            &format!("{m}/environment/Dockerfile"),
            "FROM ubuntu:24.04\n",
        );
        write(
            dir.path(),
            &format!("{m}/verifier/test.sh"),
            "#!/bin/bash\n",
        );
        write(dir.path(), &format!("{m}/oracle/solve.sh"), "#!/bin/bash\n");
    }
    let start = Instant::now();
    let roots = benchflow_task::discover(dir.path());
    let found = start.elapsed();
    assert_eq!(roots.len(), 1000);
    // As `bench check` does: one thread per core, each checking a slice.
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let ok: usize = std::thread::scope(|s| {
        roots
            .chunks(roots.len().div_ceil(threads))
            .map(|chunk| {
                s.spawn(move || {
                    chunk
                        .iter()
                        .filter(|r| benchflow_task::check(r, &CheckOptions::default()).ok())
                        .count()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .sum()
    });
    let total = start.elapsed();
    assert_eq!(ok, 1000);
    println!("discover {found:?}, discover and check (with digests, {threads} threads) {total:?}");
    let bound = if cfg!(debug_assertions) { 10.0 } else { 1.0 };
    assert!(total.as_secs_f64() < bound, "took {total:?}");
}
