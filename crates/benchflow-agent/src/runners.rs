//! Runner scripts for SDK agents, embedded in the binary.
//!
//! Some agents are a Python or Node framework rather than a command-line
//! program: OpenHands' software agent SDK, NVIDIA's NeMo Agent Toolkit,
//! Strands, DeerFlow, Google's Antigravity SDK, a LangGraph graph, an Eve
//! app. For each, BenchFlow ships a small runner script that builds the agent
//! from the framework, runs it once on the instruction, and reports what it
//! does as JSON lines on stdout in the `benchflow-runner` format (see
//! [`crate::headless`]'s parser for it). The scripts live in
//! `crates/benchflow-agent/runners/`; a manifest writes one into the agent's
//! home with `[[headless.files]] runner = "run_strands.py"`, and its
//! headless command runs it with the framework's interpreter.
//!
//! A runner reads the instruction from `$BENCHFLOW_INSTRUCTION_FILE` and the
//! model from `$BENCHFLOW_MODEL`. The Python runners share `bf_events.py`
//! (the event writer and error classification), so a manifest writes it
//! next to its runner.

/// Every runner, by file name.
const RUNNERS: &[(&str, &str)] = &[
    ("bf_events.py", include_str!("../runners/bf_events.py")),
    (
        "run_openhands_sdk.py",
        include_str!("../runners/run_openhands_sdk.py"),
    ),
    (
        "run_nemo_agent.py",
        include_str!("../runners/run_nemo_agent.py"),
    ),
    ("run_strands.py", include_str!("../runners/run_strands.py")),
    (
        "run_deerflow.py",
        include_str!("../runners/run_deerflow.py"),
    ),
    (
        "run_antigravity_sdk.py",
        include_str!("../runners/run_antigravity_sdk.py"),
    ),
    (
        "run_langgraph.py",
        include_str!("../runners/run_langgraph.py"),
    ),
    ("run_eve.mjs", include_str!("../runners/run_eve.mjs")),
    (
        "run_dspy_rlm.py",
        include_str!("../runners/run_dspy_rlm.py"),
    ),
];

/// The runner named `name`, if BenchFlow has it.
pub fn get(name: &str) -> Option<&'static str> {
    RUNNERS
        .iter()
        .find(|(file, _)| *file == name)
        .map(|(_, text)| *text)
}

/// Every runner's file name, for error messages.
pub fn names() -> Vec<&'static str> {
    RUNNERS.iter().map(|(file, _)| *file).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_runner_is_embedded_and_found_by_name() {
        for name in names() {
            let text = get(name).unwrap();
            assert!(text.len() > 200, "{name} is nearly empty");
        }
        assert!(get("run_nothing.py").is_none());
    }

    #[test]
    fn python_runners_import_the_shared_event_writer() {
        for name in names()
            .into_iter()
            .filter(|n| n.starts_with("run_") && n.ends_with(".py"))
        {
            let text = get(name).unwrap();
            assert!(text.contains("import bf_events"), "{name}");
            assert!(
                text.contains("BENCHFLOW_INSTRUCTION_FILE")
                    || text.contains("bf_events.instruction()"),
                "{name} must read the instruction file"
            );
        }
    }

    /// Run run_eve.mjs with `node` on a fake Eve app whose `eve` fails to
    /// start; the app folder afterwards. `None` without node.
    fn eve_app_after_run(
        flat: bool,
        skills: Option<&std::path::Path>,
    ) -> Option<tempfile::TempDir> {
        use std::os::unix::fs::PermissionsExt;
        let node = std::process::Command::new("node").arg("--version").output();
        if !node.is_ok_and(|o| o.status.success()) {
            eprintln!("skipped: no node");
            return None;
        }
        let app = tempfile::tempdir().unwrap();
        let dir = app.path();
        std::fs::write(dir.join("package.json"), r#"{"dependencies":{"eve":"*"}}"#).unwrap();
        if flat {
            std::fs::write(dir.join("agent.ts"), "").unwrap();
            std::fs::write(dir.join("instructions.md"), "").unwrap();
        } else {
            std::fs::create_dir_all(dir.join("agent")).unwrap();
        }
        let bin = dir.join("node_modules/.bin/eve");
        std::fs::create_dir_all(bin.parent().unwrap()).unwrap();
        std::fs::write(&bin, "#!/bin/sh\nexit 3\n").unwrap();
        std::fs::set_permissions(&bin, std::fs::Permissions::from_mode(0o755)).unwrap();
        let work = tempfile::tempdir().unwrap();
        let runner = work.path().join("run_eve.mjs");
        std::fs::write(&runner, get("run_eve.mjs").unwrap()).unwrap();
        let instruction = work.path().join("instruction.md");
        std::fs::write(&instruction, "Do the task.").unwrap();
        let mut cmd = std::process::Command::new("node");
        cmd.arg(&runner)
            .arg(&instruction)
            .env("EVE_PROJECT", dir)
            .env("EVE_SKIP_BUILD", "1")
            .env("EVE_STARTUP_TIMEOUT_MS", "2000")
            .env("HOME", work.path())
            .env_remove("BENCHFLOW_SKILLS_DIR");
        if let Some(s) = skills {
            cmd.env("BENCHFLOW_SKILLS_DIR", s);
        }
        let out = cmd.output().unwrap();
        assert!(!out.status.success(), "the fake eve cannot start");
        Some(app)
    }

    /// Delivered skills: two skills, a folder with no SKILL.md, and a loose
    /// file.
    fn delivered() -> tempfile::TempDir {
        let s = tempfile::tempdir().unwrap();
        for name in ["csv-tools", "pdf-reading"] {
            std::fs::create_dir_all(s.path().join(name).join("scripts")).unwrap();
            std::fs::write(s.path().join(name).join("SKILL.md"), name).unwrap();
            std::fs::write(s.path().join(name).join("scripts/run.sh"), "true").unwrap();
        }
        std::fs::create_dir_all(s.path().join("licenses")).unwrap();
        std::fs::write(s.path().join("README.md"), "loose").unwrap();
        s
    }

    fn listing(dir: &std::path::Path) -> Vec<String> {
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map(|d| {
                d.flatten()
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        names
    }

    #[test]
    fn the_eve_runner_copies_delivered_skills_into_the_app_with_a_prefix() {
        let skills = delivered();
        let Some(app) = eve_app_after_run(false, Some(skills.path())) else {
            return;
        };
        let into = app.path().join("agent/skills");
        assert_eq!(
            listing(&into),
            ["benchflow_csv-tools", "benchflow_pdf-reading"]
        );
        assert!(into.join("benchflow_csv-tools/scripts/run.sh").is_file());
        // A flat app keeps its skills beside agent.ts.
        let Some(app) = eve_app_after_run(true, Some(skills.path())) else {
            return;
        };
        assert_eq!(
            listing(&app.path().join("skills")),
            ["benchflow_csv-tools", "benchflow_pdf-reading"]
        );
    }

    #[test]
    fn the_eve_runner_adds_no_skills_folder_without_skills() {
        let Some(app) = eve_app_after_run(false, None) else {
            return;
        };
        assert!(!app.path().join("agent/skills").exists());
    }
}
