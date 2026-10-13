//! `bench import` and `bench export` against task.md's reference converter.
//!
//! `tests/fixtures/convert/expected/` was written by task.md's
//! `tools/convert.py` (`import`, `import-v06`, `export-harbor`) from the
//! inputs beside it; see its `README.md`. Every converted package and
//! every exported folder must match it file for file, byte for byte, with
//! the same executable bits, and every export must report the same losses.
//! The one deliberate difference: where convert.py's messages name another
//! framework, BenchFlow's say what is meant (see `common/mod.rs`), and the
//! expected messages are read through that mapping; BenchFlow's own are
//! checked to name none. The import record and its folder (`[import.harbor]`,
//! `import/harbor/`) are identifiers other tools read, and keep convert.py's
//! names, compared byte for byte.

mod common;

use common::{naming, neutral_json};

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use benchflow_task::convert::{self, ConvertError, ExportOptions, ExportStatus};
use serde_json::Value;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/convert")
}

fn results() -> serde_json::Map<String, Value> {
    let text = fs::read_to_string(fixtures().join("expected/results.json")).unwrap();
    match neutral_json(&serde_json::from_str(&text).unwrap()) {
        Value::Object(m) => m,
        _ => panic!("results.json is an object"),
    }
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
        let bytes = fs::read(entry.path()).unwrap();
        out.insert(rel, (bytes, executable(entry.path())));
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

fn assert_same_tree(got: &Path, want: &Path, skip: &[&str]) {
    let mut g = tree(got);
    for s in skip {
        g.remove(*s);
    }
    let w = tree(want);
    let gk: Vec<&String> = g.keys().collect();
    let wk: Vec<&String> = w.keys().collect();
    assert_eq!(
        gk,
        wk,
        "files differ between {} and {}",
        got.display(),
        want.display()
    );
    for (rel, (bytes, exe)) in &w {
        let (gb, ge) = &g[rel];
        if gb != bytes {
            panic!(
                "{rel} differs\n--- got ({})\n{}\n--- want ({})\n{}",
                got.display(),
                String::from_utf8_lossy(gb),
                want.display(),
                String::from_utf8_lossy(bytes)
            );
        }
        assert_eq!(ge, exe, "{rel}: the executable bit differs");
    }
}

fn status_name(s: ExportStatus) -> &'static str {
    match s {
        ExportStatus::Lossless => "lossless",
        ExportStatus::Degraded => "degraded",
        ExportStatus::Refused => "refused",
    }
}

/// Export `src` and compare the outcome with the golden `key`.
fn check_export(
    src: &Path,
    key: &str,
    golden_tree: &Path,
    results: &serde_json::Map<String, Value>,
) {
    let want = &results[key];
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("out");
    let report = match convert::export_split_with(src, &out, &ExportOptions::default()) {
        Ok(r) => r,
        Err(ConvertError::Refused(r)) => {
            assert!(!out.exists(), "{key}: a refused export writes nothing");
            *r
        }
        Err(e) => panic!("{key}: {e}"),
    };
    assert_eq!(status_name(report.status), want["status"], "{key}: status");
    let losses: Vec<Value> = report
        .losses
        .iter()
        .map(|l| serde_json::to_value(l).unwrap())
        .collect();
    assert_eq!(Value::Array(losses), want["losses"], "{key}: losses");
    let said = report
        .losses
        .iter()
        .flat_map(|l| [l.reason.as_str(), l.text.as_str()])
        .chain(report.rewritten.iter().map(|r| r.reason.as_str()))
        .chain([report.summary.as_str()]);
    let found = naming(said);
    assert!(found.is_empty(), "{key}: {found:#?}");
    assert_eq!(
        serde_json::to_value(&report.restored_originals).unwrap(),
        want["restored_originals"],
        "{key}"
    );
    assert_eq!(
        serde_json::to_value(&report.rewritten).unwrap(),
        want["rewritten"],
        "{key}"
    );
    assert_eq!(
        serde_json::to_value(&report.restored_extension_paths).unwrap(),
        want["restored_extension_paths"],
        "{key}"
    );
    if report.status != ExportStatus::Refused {
        assert_same_tree(&out, golden_tree, &["export-report.json"]);
        let written: Value =
            serde_json::from_str(&fs::read_to_string(out.join("export-report.json")).unwrap())
                .unwrap();
        assert_eq!(written["status"], want["status"]);
        // The report's format names, as convert.py writes them
        // (`"from": "task.md", "to": "harbor"`): identifiers other tools
        // read.
        assert_eq!(
            (&written["report"], &written["from"], &written["to"]),
            (&"export".into(), &"task.md".into(), &"harbor".into()),
            "{key}"
        );
    }
}

fn check_kind(kind: &str) -> usize {
    let results = results();
    let mut checked = 0;
    let mut inputs: Vec<PathBuf> = fs::read_dir(fixtures().join(kind))
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    inputs.sort();
    for src in inputs {
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        let key = format!("{kind}/{name}");
        let want = &results[&key];
        let work = tempfile::tempdir().unwrap();
        let out = work.path().join(&name);
        match convert::import_to_taskmd(&src, &out, false) {
            Ok(report) => {
                assert!(
                    want.get("error").is_none(),
                    "{key}: convert.py refused it: {}",
                    want["error"]
                );
                assert_same_tree(
                    &out,
                    &fixtures().join("expected").join(kind).join(&name),
                    &[],
                );
                let found = naming(report.notes.iter().map(|n| n.text.as_str()));
                assert!(found.is_empty(), "{key}: {found:#?}");
                let notes = want["notes"].as_array().unwrap().len();
                assert!(
                    report.notes.len() >= notes.saturating_sub(1),
                    "{key}: notes {:?}",
                    report.notes
                );
                check_export(
                    &out,
                    &format!("roundtrip/{key}"),
                    &fixtures().join("expected/roundtrip").join(kind).join(&name),
                    &results,
                );
                // The package convert.py wrote exports to the same files.
                check_export(
                    &fixtures().join("expected").join(kind).join(&name),
                    &format!("roundtrip/{key}"),
                    &fixtures().join("expected/roundtrip").join(kind).join(&name),
                    &results,
                );
            }
            Err(e) => {
                let expected = want
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| panic!("{key}: {e}"));
                assert_eq!(e.to_string(), expected, "{key}");
                assert!(!out.exists(), "{key}: a failed import writes nothing");
            }
        }
        checked += 1;
    }
    checked
}

#[test]
fn harbor_imports_match_convert_py() {
    assert!(check_kind("harbor") >= 9);
}

#[test]
fn v06_imports_match_convert_py() {
    assert!(check_kind("v06") >= 11);
}

#[test]
fn exports_match_convert_py() {
    let results = results();
    let mut n = 0;
    for entry in fs::read_dir(fixtures().join("export"))
        .unwrap()
        .filter_map(Result::ok)
    {
        let src = entry.path();
        if !src.is_dir() {
            continue;
        }
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        check_export(
            &src,
            &format!("export/{name}"),
            &fixtures().join("expected/export").join(&name),
            &results,
        );
        n += 1;
    }
    assert!(n >= 5);
}

#[test]
fn import_never_overwrites_unless_asked() {
    let src = fixtures().join("harbor/odd-values");
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("out");
    fs::create_dir_all(&out).unwrap();
    fs::write(out.join("keep.txt"), "mine").unwrap();
    let err = convert::import_to_taskmd(&src, &out, false).unwrap_err();
    assert!(matches!(err, ConvertError::OutputExists(_)), "{err}");
    assert_eq!(fs::read_to_string(out.join("keep.txt")).unwrap(), "mine");
    let opts = convert::ImportOptions {
        overwrite: true,
        ..Default::default()
    };
    convert::import_to_taskmd_with(&src, &out, &opts).unwrap();
    assert!(!out.join("keep.txt").exists());
    assert!(out.join("task.md").is_file());
}

#[test]
fn import_refuses_task_md_and_overlapping_folders() {
    let work = tempfile::tempdir().unwrap();
    let draft2 = work.path().join("t");
    fs::create_dir_all(&draft2).unwrap();
    fs::write(draft2.join("task.md"), "Do it.\n").unwrap();
    assert!(matches!(
        convert::import_to_taskmd(&draft2, &work.path().join("o"), false),
        Err(ConvertError::AlreadyTaskMd(_))
    ));
    let src = fixtures().join("harbor/odd-values");
    assert!(matches!(
        convert::import_to_taskmd(&src, &src.join("inside"), false),
        Err(ConvertError::OutputOverlaps { .. })
    ));
    assert!(matches!(
        convert::import_to_taskmd(&work.path().join("missing"), &work.path().join("o2"), false),
        Err(ConvertError::NotATask(_))
    ));
}

#[test]
fn degraded_exports_write_when_allowed() {
    let src = fixtures().join("export/clock-refused");
    let work = tempfile::tempdir().unwrap();
    let out = work.path().join("out");
    let opts = ExportOptions {
        allow_degraded: true,
        ..Default::default()
    };
    let report = convert::export_split_with(&src, &out, &opts).unwrap();
    assert_eq!(report.status, ExportStatus::Degraded);
    assert!(out.join("task.toml").is_file());
    assert!(out.join("export-report.json").is_file());
}

/// The task.md checker finds in each imported package exactly the errors
/// task.md's reference checker finds in convert.py's import of it
/// (`expected/checked.json`, written by `check_goldens.py`). The errors
/// there come from incomplete inputs, such as a task with no Dockerfile,
/// never from the conversion.
#[test]
fn imports_check_as_the_reference_imports_do() {
    let text = fs::read_to_string(fixtures().join("expected/checked.json")).unwrap();
    let checked: BTreeMap<String, Vec<String>> =
        serde_json::from_value(neutral_json(&serde_json::from_str(&text).unwrap())).unwrap();
    let results = results();
    let mut seen = 0;
    for kind in ["harbor", "v06"] {
        for entry in fs::read_dir(fixtures().join(kind))
            .unwrap()
            .filter_map(Result::ok)
        {
            let src = entry.path();
            let key = format!("{kind}/{}", src.file_name().unwrap().to_string_lossy());
            if results[&key].get("error").is_some() {
                continue;
            }
            let work = tempfile::tempdir().unwrap();
            let out = work.path().join("pkg");
            convert::import_to_taskmd(&src, &out, false).unwrap();
            let report = benchflow_task::check(&out, &benchflow_task::CheckOptions { deep: true });
            let mut errors: Vec<String> = report
                .diagnostics
                .iter()
                .filter(|d| d.is_error())
                .map(|d| d.message.clone())
                .collect();
            errors.sort();
            assert_eq!(errors, checked[&key], "{key}");
            seen += 1;
        }
    }
    assert_eq!(seen, checked.len(), "every recorded package was checked");
}

/// A package with both import records (from hand edits or merges): export
/// restores originals from the one it reads, `[import.harbor]`, copies
/// neither folder into the output, and says the other was left out instead
/// of reporting a lossless export.
#[test]
fn a_package_with_both_import_records_exports_neither_folder() {
    let src = fixtures().join("expected/harbor/build-steps");
    let work = tempfile::tempdir().unwrap();
    let pkg = work.path().join("pkg");
    for (rel, (bytes, _)) in tree(&src) {
        let to = pkg.join(&rel);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        fs::write(to, bytes).unwrap();
    }
    let md = fs::read_to_string(pkg.join("task.md")).unwrap();
    let md = md.replacen(
        "[import.harbor]\n",
        "[import.split]\n# from a hand edit\nversion = \"1.0\"  # an old record\n\n[import.harbor]\n",
        1,
    );
    fs::write(pkg.join("task.md"), md).unwrap();
    fs::create_dir_all(pkg.join("import/split")).unwrap();
    fs::write(pkg.join("import/split/task.toml"), "stray = true\n").unwrap();
    fs::write(pkg.join("import/harbor/notes.txt"), "not an original\n").unwrap();
    let out = work.path().join("out");
    let report = convert::export_split_with(&pkg, &out, &ExportOptions::default()).unwrap();
    assert_eq!(report.status, ExportStatus::Degraded);
    assert!(!out.join("import").exists(), "{:?}", tree(&out).keys());
    let left_out: Vec<&str> = report.losses.iter().map(|l| l.path.as_str()).collect();
    assert_eq!(left_out, ["[import.split]"], "{:?}", report.losses);
    // Both folders are named, the one read included.
    let reason = &report.losses[0].reason;
    assert!(
        reason.contains("import/split/") && reason.contains("the other files in import/harbor/"),
        "{reason}"
    );
    // The ignored record's comments go with it: task.toml is restored.
    assert!(
        report.restored_originals.iter().any(|r| r == "task.toml"),
        "{:?}",
        report.rewritten
    );
    // The originals still come back byte for byte.
    let want = fixtures().join("expected/roundtrip/harbor/build-steps");
    assert_eq!(
        fs::read(out.join("task.toml")).unwrap(),
        fs::read(want.join("task.toml")).unwrap()
    );
}

/// A package whose record is `[import.split]` (an unreleased build wrote
/// it) exports to the same files as the same package under convert.py's
/// `[import.harbor]`.
#[test]
fn a_package_with_only_the_other_record_name_exports_the_same() {
    let src = fixtures().join("expected/harbor/build-steps");
    let work = tempfile::tempdir().unwrap();
    let pkg = work.path().join("pkg");
    for (rel, (bytes, exe)) in tree(&src) {
        let rel = rel.replacen("import/harbor/", "import/split/", 1);
        let to = pkg.join(&rel);
        fs::create_dir_all(to.parent().unwrap()).unwrap();
        let bytes = if rel == "task.md" {
            String::from_utf8(bytes)
                .unwrap()
                .replace("[import.harbor", "[import.split")
                .into_bytes()
        } else {
            bytes
        };
        fs::write(&to, bytes).unwrap();
        #[cfg(unix)]
        if exe {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&to, fs::Permissions::from_mode(0o755)).unwrap();
        }
    }
    assert!(pkg.join("import/split").is_dir());
    let out = work.path().join("out");
    let report = convert::export_split_with(&pkg, &out, &ExportOptions::default()).unwrap();
    let want = &results()["roundtrip/harbor/build-steps"];
    assert_eq!(status_name(report.status), want["status"]);
    assert_eq!(
        serde_json::to_value(&report.restored_originals).unwrap(),
        want["restored_originals"]
    );
    assert_same_tree(
        &out,
        &fixtures().join("expected/roundtrip/harbor/build-steps"),
        &["export-report.json"],
    );
}

fn write_files(root: &Path, files: &[(&str, &str)]) {
    for (rel, text) in files {
        let path = root.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, text).unwrap();
    }
}

/// F1-B-r2 L3: the split reader takes `x-allow-cloud-metadata = "yes"` as
/// true; `bench import` writes `true`, which task.md reads, so the imported
/// package checks clean and still opts out. The same for a BenchFlow 0.8
/// task.md, whose import goes the same way.
#[test]
fn a_loosely_boolean_opt_out_imports_as_a_boolean() {
    let work = tempfile::tempdir().unwrap();
    let split = work.path().join("split");
    write_files(
        &split,
        &[
            (
                "task.toml",
                "schema_version = \"1.4\"\n\n[task]\nname = \"acme/meta\"\n\n[metadata]\nx-allow-cloud-metadata = \"yes\"\n",
            ),
            ("instruction.md", "Do it.\n"),
            ("environment/Dockerfile", "FROM ubuntu:24.04\n"),
            ("tests/test.sh", "echo 1 > /logs/verifier/reward.txt\n"),
        ],
    );
    let v06 = work.path().join("v06");
    write_files(
        &v06,
        &[
            (
                "task.md",
                "---\nschema_version: \"1.3\"\ntask:\n  name: acme/meta\nmetadata:\n  x-allow-cloud-metadata: \"yes\"\n---\nDo it.\n",
            ),
            ("environment/Dockerfile", "FROM ubuntu:24.04\n"),
            ("tests/test.sh", "echo 1 > /logs/verifier/reward.txt\n"),
        ],
    );
    for src in [&split, &v06] {
        let before = benchflow_task::check(src, &benchflow_task::CheckOptions { deep: true });
        assert!(
            before
                .package
                .expect("a package")
                .sandbox
                .allow_cloud_metadata,
            "{}",
            src.display()
        );
        let out = work.path().join(format!(
            "out-{}",
            src.file_name().unwrap().to_string_lossy()
        ));
        convert::import_to_taskmd(src, &out, false).unwrap();
        let md = fs::read_to_string(out.join("task.md")).unwrap();
        assert!(md.contains("x-allow-cloud-metadata = true\n"), "{md}");
        let after = benchflow_task::check(&out, &benchflow_task::CheckOptions { deep: true });
        assert!(after.ok(), "{}: {:#?}", src.display(), after.diagnostics);
        assert!(
            after
                .package
                .expect("a package")
                .sandbox
                .allow_cloud_metadata
        );
    }
}

/// F1-A-r2: task.md ignores the opt-out under `[about]`; its export must
/// not opt the split task out, so `[metadata]` keeps the task's own keys
/// and not the opt-out, and the export names what it left out.
#[test]
fn an_opt_out_under_about_is_not_exported() {
    let work = tempfile::tempdir().unwrap();
    let src = work.path().join("t");
    write_files(
        &src,
        &[
            (
                "task.md",
                "Do it.\n\n```toml task\nname = \"acme/meta\"\nversion = \"1.0.0\"\n\n[about]\ndifficulty = \"easy\"\nx-allow-cloud-metadata = true\n```\n",
            ),
            ("sandbox/Dockerfile", "FROM ubuntu:24.04\n"),
            ("verifier/test.sh", "echo 1 > /logs/verifier/reward.txt\n"),
        ],
    );
    let before = benchflow_task::check(&src, &benchflow_task::CheckOptions { deep: true });
    assert!(
        !before
            .package
            .expect("a package")
            .sandbox
            .allow_cloud_metadata
    );
    let out = work.path().join("out");
    let report = convert::export_split(&src, &out, false).unwrap();
    let toml = fs::read_to_string(out.join("task.toml")).unwrap();
    assert!(!toml.contains("x-allow-cloud-metadata"), "{toml}");
    assert!(toml.contains("difficulty = \"easy\""), "{toml}");
    assert!(
        report
            .losses
            .iter()
            .any(|l| l.path == "[about] x-allow-cloud-metadata"),
        "{:?}",
        report.losses
    );
    let after = benchflow_task::check(&out, &benchflow_task::CheckOptions { deep: true });
    assert!(
        !after
            .package
            .expect("a package")
            .sandbox
            .allow_cloud_metadata
    );
}
