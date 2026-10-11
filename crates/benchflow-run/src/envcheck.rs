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

/// One sign: its name, whether it counts in the agent's own log (an
/// agent that tries a command it lacks is exploring, not broken), and
/// whether a line shows it (given whether the line is the agent's).
struct Sign {
    what: &'static str,
    in_agent_log: bool,
    matches: fn(&str, bool) -> bool,
}

const SIGNS: &[Sign] = &[
    Sign {
        what: "git refused a repository owned by another user (\"dubious ownership\")",
        in_agent_log: true,
        matches: |l, _| l.contains("dubious ownership"),
    },
    Sign {
        what: "apt could not find a package (\"Unable to locate package\")",
        in_agent_log: true,
        matches: |l, _| l.contains("Unable to locate package"),
    },
    Sign {
        what: "a command was missing (\"command not found\")",
        in_agent_log: false,
        matches: |l, _| l.contains("command not found"),
    },
    Sign {
        what: "a BenchFlow folder could not be used (\"Permission denied\")",
        in_agent_log: true,
        matches: |l, agent| {
            // The verifier's folder is closed to the agent on purpose: the
            // agent refused there is the defense holding, not a fault. The
            // verifier refused there is.
            l.contains("Permission denied")
                && !(agent && l.contains("/logs/verifier"))
                && BENCHFLOW_PATHS.iter().any(|p| names_path(l, p))
        },
    },
];

/// Whether `line` names a path that starts with `prefix`, not a longer
/// path that merely contains it (`/var/logs/` is not `/logs/`).
fn names_path(line: &str, prefix: &str) -> bool {
    line.match_indices(prefix).any(|(i, _)| {
        line[..i]
            .chars()
            .next_back()
            .is_none_or(|c| !(c.is_alphanumeric() || matches!(c, '/' | '.' | '_' | '-')))
    })
}

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
            let line = text
                .lines()
                .map(str::trim)
                .find(|l| (s.matches)(l, is_agent))?;
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
             cat: /root/secret: Permission denied\n\
             cat: /logs/verifier/reward.txt: Permission denied\n\
             cat: /var/logs/app.log: Permission denied\n",
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
        // The agent's own missing command, a non-BenchFlow path, and the
        // closed verifier folder (the defense holding) are not signs.
        assert_eq!(
            w[3],
            "agent/claude-code.txt: a BenchFlow folder could not be used (\"Permission denied\"): \
             sh: 1: cannot create /logs/agent/out.txt: Permission denied"
        );
    }

    /// The verifier refused at its own folder is the environment failing,
    /// the case this module is for; only the agent refused there is the
    /// defense holding. Guards the D7b fix of 431d4f35, which dropped both.
    #[test]
    fn a_verifier_refused_at_its_own_folder_is_a_sign() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("verifier")).unwrap();
        std::fs::write(
            dir.path().join("verifier/test-stdout.txt"),
            "1 passed\ntest.sh: line 9: /logs/verifier/reward.txt: Permission denied\n",
        )
        .unwrap();
        let w = warnings(dir.path());
        assert_eq!(w.len(), 1, "{w:#?}");
        assert!(w[0].starts_with("verifier/test-stdout.txt: a BenchFlow folder"));
    }

    #[test]
    fn a_clean_trial_has_none() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("verifier")).unwrap();
        std::fs::write(dir.path().join("verifier/test-stdout.txt"), "1 failed\n").unwrap();
        assert!(warnings(dir.path()).is_empty());
    }
}
