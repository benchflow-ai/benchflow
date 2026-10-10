//! `bench import` of Terminal-Bench v1 tasks against Harbor's
//! `task migrate`. See `tests/fixtures/tbench/README.md`: importing each
//! input and exporting it back must give exactly the folder Harbor's
//! converter wrote, and the imported task must check clean.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use benchflow_task::convert::{self, ConvertError, ExportOptions, ImportOptions};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tbench")
}

/// Every file under `root`: relative path -> (bytes, executable).
fn tree(root: &Path) -> BTreeMap<String, (Vec<u8>, bool)> {
    let mut out = BTreeMap::new();
    for entry in walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let rel = entry
            .path()
            .strip_prefix(root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        out.insert(
            rel,
            (fs::read(entry.path()).unwrap(), executable(entry.path())),
        );
    }
    out
}

#[cfg(unix)]
fn executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(p).unwrap().permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable(_: &Path) -> bool {
    false
}

#[test]
fn imports_match_harbors_migrate_and_check_clean() {
    let work = tempfile::tempdir().unwrap();
    for name in ["hello", "settings", "services"] {
        let src = fixtures().join("input").join(name);
        assert!(
            convert::is_tbench(&src),
            "{name} is a Terminal-Bench v1 task"
        );
        let imported = work.path().join("import").join(name);
        let rep =
            convert::import_to_taskmd_with(&src, &imported, &ImportOptions::default()).unwrap();
        assert_eq!(rep.converted_from, Some("terminal-bench-v1"));
        assert_eq!(
            benchflow_task::detect(&imported),
            Some(benchflow_task::Format::TaskMd)
        );
        let report = benchflow_task::check(&imported, &benchflow_task::CheckOptions { deep: true });
        assert!(
            report.ok(),
            "{name} checks clean: {:#?}",
            report.diagnostics
        );

        let back = work.path().join("export").join(name);
        convert::export_split_with(&imported, &back, &ExportOptions::default()).unwrap();
        let mut got = tree(&back);
        got.remove("export-report.json");
        let want = tree(&fixtures().join("expected").join(name));
        let names = |t: &BTreeMap<String, (Vec<u8>, bool)>| t.keys().cloned().collect::<Vec<_>>();
        assert_eq!(names(&got), names(&want), "{name}: the same files");
        for (path, (bytes, exec)) in &want {
            let (g, e) = &got[path];
            assert!(
                g == bytes,
                "{name}/{path}: same bytes\n--- got\n{}\n--- want\n{}",
                String::from_utf8_lossy(g),
                String::from_utf8_lossy(bytes)
            );
            assert_eq!(e, exec, "{name}/{path}: same executable bit");
        }
    }
}

#[test]
fn solution_yaml_is_refused_and_writes_nothing() {
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("o");
    let err = convert::import_to_taskmd(&fixtures().join("input/yaml-solution"), &out, false)
        .unwrap_err();
    assert!(matches!(err, ConvertError::Invalid { .. }), "{err}");
    assert!(
        err.to_string().contains("rewrite it as solution.sh"),
        "{err}"
    );
    assert!(!out.exists());
}

#[test]
fn resources_are_set_on_request() {
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("o");
    let opts = ImportOptions {
        tbench_resources: convert::Resources {
            cpus: Some(4),
            memory_mb: Some(8192),
            ..Default::default()
        },
        ..Default::default()
    };
    convert::import_to_taskmd_with(&fixtures().join("input/hello"), &out, &opts).unwrap();
    let md = fs::read_to_string(out.join("task.md")).unwrap();
    assert!(
        md.contains("cpus = 4") && md.contains("memory = \"8 GB\""),
        "{md}"
    );
}

#[test]
fn discovery_finds_tbench_tasks() {
    let found = convert::discover_tbench(&fixtures().join("input"));
    let names: Vec<_> = found
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(names, ["hello", "services", "settings", "yaml-solution"]);
    assert!(convert::discover_tbench(&fixtures().join("expected")).is_empty());
}
