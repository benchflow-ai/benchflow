//! Convert a whole local corpus, for comparing with task.md's
//! `tools/convert.py` at scale. Off by default; run it with
//!
//! ```sh
//! BF_CORPUS=path/to/tasks BF_CORPUS_OUT=/tmp/rust-out \
//!     cargo test -p benchflow-task --test convert_corpus -- --ignored
//! ```
//!
//! Every task folder under `BF_CORPUS` is imported into
//! `$BF_CORPUS_OUT/import/<name>` and exported back into
//! `$BF_CORPUS_OUT/roundtrip/<name>`; failures go to `<name>.error` beside
//! them. Run convert.py the same way and diff the two trees.

use std::fs;
use std::path::PathBuf;

use benchflow_task::convert::{self, ConvertError, ExportOptions};

#[test]
#[ignore = "a development tool: needs BF_CORPUS and BF_CORPUS_OUT"]
fn convert_a_local_corpus() {
    let (Ok(corpus), Ok(out)) = (std::env::var("BF_CORPUS"), std::env::var("BF_CORPUS_OUT")) else {
        eprintln!("set BF_CORPUS and BF_CORPUS_OUT");
        return;
    };
    let out = PathBuf::from(out);
    let corpus = PathBuf::from(corpus);
    for task in benchflow_task::discover(&corpus) {
        let rel = task
            .strip_prefix(&corpus)
            .unwrap_or(&task)
            .to_string_lossy()
            .into_owned();
        let name = if rel.is_empty() {
            "task".to_string()
        } else {
            rel.replace('/', "__")
        };
        let imported = out.join("import").join(&name);
        match convert::import_to_taskmd(&task, &imported, false) {
            Ok(_) => {
                let rt = out.join("roundtrip").join(&name);
                let opts = ExportOptions {
                    allow_degraded: true,
                    ..Default::default()
                };
                if let Err(e) = convert::export_split_with(&imported, &rt, &opts) {
                    fs::create_dir_all(out.join("roundtrip")).unwrap();
                    fs::write(
                        out.join("roundtrip").join(format!("{name}.error")),
                        e.to_string(),
                    )
                    .unwrap();
                }
            }
            Err(ConvertError::AlreadyTaskMd(_)) => {}
            Err(e) => {
                fs::create_dir_all(out.join("import")).unwrap();
                fs::write(
                    out.join("import").join(format!("{name}.error")),
                    e.to_string(),
                )
                .unwrap();
            }
        }
    }
}
