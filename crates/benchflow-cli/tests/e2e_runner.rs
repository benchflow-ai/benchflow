//! The e2e runner's own unit tests (`e2e/test_sweep.py`), run here so
//! `cargo test` covers the runner's startup sweep: it may remove only what
//! earlier runs of the suite on this machine left, never another user's or
//! another machine's sandboxes.

use std::path::Path;
use std::process::Command;

#[test]
fn the_e2e_runners_startup_sweep_removes_only_its_own_dead_runs() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    // run.py needs Python 3.11 (tomllib); macOS's own python3 is older, so
    // a newer one on PATH is looked for first.
    let Some(python) = ["python3.13", "python3.12", "python3.11", "python3"]
        .into_iter()
        .find(|p| {
            Command::new(p)
                .args(["-c", "import sys; sys.exit(sys.version_info < (3, 11))"])
                .status()
                .is_ok_and(|s| s.success())
        })
    else {
        eprintln!("skipped: no Python 3.11 or newer on PATH");
        return;
    };
    let out = Command::new(python)
        .arg("-I")
        .arg(repo.join("e2e/test_sweep.py"))
        .env_remove("BENCHFLOW_CACHE_DIR")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
