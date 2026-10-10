//! Codes `bench check` emits itself, after the task crate's checks,
//! triggered through the binary as a user meets them.

use std::path::Path;

use assert_cmd::Command;

fn write(root: &Path, rel: &str, text: &str) {
    let path = root.join(rel);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

/// P003: a credential file in the folder the image is built from. The
/// warning names the file and how to keep it out of the build, and never
/// prints what is in it.
#[test]
fn a_credential_file_in_the_build_folder_is_named_not_shown() {
    const SECRET: &str = "ghp_fakefakefakefakefakefake0123456789";
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("t");
    write(
        &task,
        "task.md",
        "Fix the bug.\n\n```toml task\nname = \"acme/t\"\nversion = \"1.0.0\"\n```\n",
    );
    write(
        &task,
        "sandbox/Dockerfile",
        "FROM ubuntu:24.04\nCOPY . /app\n",
    );
    write(
        &task,
        "sandbox/.netrc",
        &format!("machine github.com login bot password {SECRET}\n"),
    );
    write(
        &task,
        "verifier/test.sh",
        "echo 1 > /logs/verifier/reward.txt\n",
    );
    write(&task, "oracle/solve.sh", "true\n");
    let out = Command::cargo_bin("bench")
        .unwrap()
        .current_dir(d.path())
        .env("HOME", d.path())
        .env("BENCHFLOW_CACHE_DIR", d.path().join(".cache"))
        .env("NO_COLOR", "1")
        .args(["check", "t"])
        .output()
        .unwrap();
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(all.contains("warning[P003]"), "{all}");
    assert!(all.contains("sandbox/.netrc"), "{all}");
    assert!(all.contains("help:"), "{all}");
    assert!(!all.contains(SECRET), "the file's content is never shown");
}

/// P004: a rubric's model judge reads a reference the package lacks. The
/// error names the criterion and the file; with the file in place, the
/// check passes and says which model judges how many criteria.
#[test]
fn a_judge_reference_the_package_lacks_is_an_error() {
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("t");
    write(
        &task,
        "task.md",
        "Write `/app/memo.md`.\n\n```toml task\nname = \"acme/t\"\nversion = \"1.0.0\"\n\n[sandbox]\nworkdir = \"/app\"\n```\n",
    );
    write(&task, "sandbox/Dockerfile", "FROM ubuntu:24.04\n");
    write(&task, "verifier/test.sh", "true\n");
    write(&task, "oracle/solve.sh", "true\n");
    write(
        &task,
        "verifier/rubric.json",
        r#"{"$schema": "https://task.md/schema/rubric-1.json", "x-judge": {"model": "vercel/judge/model"},
            "criteria": [{"id": "memo", "text": "The memo names the total.", "judge": "llm", "points": 1,
            "stated": "Write `/app/memo.md`", "evidence": ["/app/memo.md"], "reference": "verifier/answers.md"}]}"#,
    );
    let check = || {
        let out = Command::cargo_bin("bench")
            .unwrap()
            .current_dir(d.path())
            .env("HOME", d.path())
            .env("BENCHFLOW_CACHE_DIR", d.path().join(".cache"))
            .env("NO_COLOR", "1")
            .args(["check", "t"])
            .output()
            .unwrap();
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    };
    let all = check();
    assert!(all.contains("error[P004]"), "{all}");
    assert!(
        all.contains("memo: reference: verifier/answers.md cannot be read"),
        "{all}"
    );
    write(&task, "verifier/answers.md", "The total is 42.\n");
    let all = check();
    assert!(!all.contains("P004"), "{all}");
}

/// A split-layout task named by a relative path: the warning shows its line
/// of task.toml (once `t/`, not `t/t/`), and `--fix` applies the rename
/// rather than failing as a BenchFlow bug.
#[test]
fn a_split_task_by_relative_path_shows_its_source_and_fixes() {
    let d = tempfile::tempdir().unwrap();
    let task = d.path().join("t");
    write(
        &task,
        "task.toml",
        "schema_version = \"1.4\"\n\n[task]\nname = \"acme/t\"\n\n[environment]\nnetwork_mod = \"none\"\n",
    );
    write(&task, "instruction.md", "Fix the bug.\n");
    write(&task, "environment/Dockerfile", "FROM ubuntu:24.04\n");
    write(
        &task,
        "tests/test.sh",
        "echo 1 > /logs/verifier/reward.txt\n",
    );
    let bench = |args: &[&str]| {
        let out = Command::cargo_bin("bench")
            .unwrap()
            .current_dir(d.path())
            .env("HOME", d.path())
            .env("BENCHFLOW_CACHE_DIR", d.path().join(".cache"))
            .env("NO_COLOR", "1")
            .args(args)
            .output()
            .unwrap();
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    };
    let all = bench(&["check", "t"]);
    assert!(all.contains("warning[X100]"), "{all}");
    assert!(all.contains("t/task.toml:7"), "{all}");
    assert!(!all.contains("t/t/"), "{all}");
    assert!(
        all.contains("network_mod = \"none\""),
        "the source line is shown: {all}"
    );
    let all = bench(&["check", "--fix", "t"]);
    assert!(!all.contains("bug"), "{all}");
    let toml = std::fs::read_to_string(task.join("task.toml")).unwrap();
    assert!(toml.contains("network_mode = \"none\""), "{toml}");
}
