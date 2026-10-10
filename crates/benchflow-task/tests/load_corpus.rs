//! Check a whole local corpus and print what the checks say, for spotting
//! false alarms on real tasks. Off by default; run it with
//!
//! ```sh
//! BF_CORPUS=path/to/tasks cargo test -p benchflow-task --release \
//!     --test load_corpus -- --ignored --nocapture
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use benchflow_task::{CheckOptions, Severity};

#[test]
#[ignore = "a development tool: needs BF_CORPUS"]
fn check_a_local_corpus() {
    let Ok(corpus) = std::env::var("BF_CORPUS") else {
        eprintln!("set BF_CORPUS");
        return;
    };
    let start = Instant::now();
    let roots = benchflow_task::discover(&PathBuf::from(&corpus));
    let found = start.elapsed();
    let mut by_code: BTreeMap<String, usize> = BTreeMap::new();
    let mut failing = 0;
    for root in &roots {
        let report = benchflow_task::check(root, &CheckOptions::default());
        if !report.ok() {
            failing += 1;
        }
        for d in &report.diagnostics {
            *by_code
                .entry(format!("{} {:?}", d.code, d.severity))
                .or_default() += 1;
            if d.severity != Severity::Note {
                println!("{}: {} {}", root.display(), d.code, d.message);
            }
        }
        if let Some(p) = &report.package {
            for u in &p.unsupported {
                *by_code
                    .entry(format!("unsupported {}", u.field))
                    .or_default() += 1;
            }
        }
    }
    println!(
        "{} packages, {failing} with errors; discover {:?}, discover and check {:?}",
        roots.len(),
        found,
        start.elapsed()
    );
    for (k, n) in by_code {
        println!("{n:6} {k}");
    }
}
