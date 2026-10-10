//! Signs that a trial's environment, not the agent, failed.
//!
//! A verifier that cannot run its tests often still writes a reward of 0,
//! so a broken task (an apt package gone upstream, a command missing from
//! the image, git refusing a repository it does not own, a BenchFlow folder
//! it cannot write) reads as the agent's failure. The score stays as the
//! verifier wrote it; the trial and the job's summary carry a warning that
//! names what was seen, so a reader can tell the two apart.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// How much of the end of each log is read.
const READ_TAIL: u64 = 256 * 1024;

/// Folders BenchFlow owns inside the sandbox.
const BENCHFLOW_PATHS: &[&str] = &["/logs/", "/opt/benchflow", "/tmp/benchflow"];

/// One sign: its name, and whether it counts in the agent's own log (an
/// agent that tries a command it lacks is exploring, not broken).
struct Sign {
    what: &'static str,
    in_agent_log: bool,
    matches: fn(&str) -> bool,
}

const SIGNS: &[Sign] = &[
    Sign {
        what: "git refused a repository owned by another user (\"dubious ownership\")",
        in_agent_log: true,
        matches: |l| l.contains("dubious ownership"),
    },
    Sign {
        what: "apt could not find a package (\"Unable to locate package\")",
        in_agent_log: true,
        matches: |l| l.contains("Unable to locate package"),
    },
    Sign {
        what: "a command was missing (\"command not found\")",
        in_agent_log: false,
        matches: |l| l.contains("command not found"),
    },
    Sign {
        what: "a BenchFlow folder could not be used (\"Permission denied\")",
        in_agent_log: true,
        matches: |l| {
            l.contains("Permission denied") && BENCHFLOW_PATHS.iter().any(|p| l.contains(p))
        },
    },
];

/// The warnings for a trial folder: one per sign per log, each with the
/// first line that showed it.
pub fn warnings(trial_dir: &Path) -> Vec<String> {
    let mut logs: Vec<(String, bool)> = ["test-stdout.txt", "test-stderr.txt"]
        .iter()
        .map(|f| (format!("verifier/{f}"), false))
        .collect();
    if let Ok(entries) = std::fs::read_dir(trial_dir.join("agent")) {
        let mut names: Vec<String> = entries
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".txt"))
            .collect();
        names.sort();
        logs.extend(names.into_iter().map(|n| (format!("agent/{n}"), true)));
    }
    let mut out = Vec::new();
    for (rel, is_agent) in logs {
        let Some(text) = tail(&trial_dir.join(&rel)) else {
            continue;
        };
        out.extend(warnings_in(&text, &rel, is_agent));
    }
    out
}

/// The warnings for one log's text.
fn warnings_in(text: &str, source: &str, is_agent: bool) -> Vec<String> {
    SIGNS
        .iter()
        .filter(|s| !is_agent || s.in_agent_log)
        .filter_map(|s| {
            let line = text.lines().map(str::trim).find(|l| (s.matches)(l))?;
            Some(format!("{source}: {}: {}", s.what, clip(line)))
        })
        .collect()
}

fn clip(line: &str) -> String {
    if line.chars().count() > 160 {
        format!("{}…", line.chars().take(160).collect::<String>())
    } else {
        line.to_string()
    }
}

fn tail(path: &Path) -> Option<String> {
    let mut f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    f.seek(SeekFrom::Start(len.saturating_sub(READ_TAIL)))
        .ok()?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todays_broken_tasks_are_named() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("verifier")).unwrap();
        std::fs::create_dir_all(dir.path().join("agent")).unwrap();
        // fix-git's oracle (scale-tb30) and chess-best-move's (T4.7-final).
        std::fs::write(
            dir.path().join("verifier/test-stdout.txt"),
            "fatal: detected dubious ownership in repository at '/app/personal-site'\n\
             E: Unable to locate package stockfish\n\
             /tests/test.sh: line 4: uv: command not found\n",
        )
        .unwrap();
        std::fs::write(
            dir.path().join("agent/claude-code.txt"),
            "bash: rg: command not found\n\
             sh: 1: cannot create /logs/agent/out.txt: Permission denied\n\
             cat: /root/secret: Permission denied\n",
        )
        .unwrap();
        let w = warnings(dir.path());
        assert_eq!(w.len(), 4, "{w:#?}");
        assert!(
            w[0].starts_with("verifier/test-stdout.txt: git refused"),
            "{}",
            w[0]
        );
        assert!(w[0].ends_with("'/app/personal-site'"), "{}", w[0]);
        assert!(w[1].contains("Unable to locate package stockfish"));
        assert!(w[2].contains("uv: command not found"));
        // The agent's own missing command and a non-BenchFlow path are not signs.
        assert_eq!(
            w[3],
            "agent/claude-code.txt: a BenchFlow folder could not be used (\"Permission denied\"): \
             sh: 1: cannot create /logs/agent/out.txt: Permission denied"
        );
    }

    #[test]
    fn a_clean_trial_has_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("verifier")).unwrap();
        std::fs::write(dir.path().join("verifier/test-stdout.txt"), "1 failed\n").unwrap();
        assert!(warnings(dir.path()).is_empty());
    }
}
