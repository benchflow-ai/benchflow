//! `[resume]`: whether the agent's own session can carry on into the next
//! graded stage (`bench run --resume-sessions`).
//!
//! The handoff before each stage is graded kills every agent process; the
//! agent's home survives it. An agent that can resume keeps its session
//! there (`session_dirs`), and the next stage starts it on that session:
//! over ACP with `session/load` and the previous stage's session id (when
//! the agent offers `loadSession`), headless with `$BENCHFLOW_RESUME` set,
//! which the command turns into the CLI's own resume flag
//! (`${BENCHFLOW_RESUME:+--continue}`).
//!
//! `load` names the agent's session file format, for `bench run
//! --load-trajectory` (Harbor's flag of the same name): the file is placed
//! where the CLI finds it, and the first prompt resumes it the same way.

use serde::{Deserialize, Serialize};

use super::options::Parts;
use super::{Checker, Harness, ManifestError};

/// How an agent resumes its session (`[resume]`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ResumeSpec {
    /// Over ACP: `session/load` with the previous stage's session id.
    pub acp: bool,
    /// Headless: the command resumes when `$BENCHFLOW_RESUME` is set.
    pub headless: bool,
    /// Where under its home the agent keeps sessions (`.claude/projects`):
    /// checked before a stage resumes.
    pub session_dirs: Vec<String>,
    /// The agent's session file format, when a session from a file can be
    /// loaded before the first prompt (`--load-trajectory`).
    pub load: Option<SessionFile>,
}

/// A session file format BenchFlow knows how to place for an agent's CLI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SessionFile {
    /// Claude Code's session: `<session-id>.jsonl` in
    /// `~/.claude/projects/<working folder, every character but letters,
    /// digits and `-` written as `-`>/`.
    ClaudeJsonl,
    /// Codex's rollout: `rollout-<YYYY-MM-DDThh-mm-ss>-<session-id>.jsonl`
    /// in `~/.codex/sessions/<YYYY>/<MM>/<DD>/`.
    CodexRollout,
}

impl SessionFile {
    pub const ALL: [SessionFile; 2] = [SessionFile::ClaudeJsonl, SessionFile::CodexRollout];

    pub fn as_str(&self) -> &'static str {
        match self {
            SessionFile::ClaudeJsonl => "claude-jsonl",
            SessionFile::CodexRollout => "codex-rollout",
        }
    }

    /// The file name the format takes, for people.
    pub fn expects(&self) -> &'static str {
        match self {
            SessionFile::ClaudeJsonl => {
                "a Claude Code session file named <session-id>.jsonl (from ~/.claude/projects/<folder>/)"
            }
            SessionFile::CodexRollout => {
                "a Codex rollout file named rollout-<YYYY-MM-DDThh-mm-ss>-<session-id>.jsonl (from ~/.codex/sessions/)"
            }
        }
    }

    /// The session id in a file of this format named `name`, after checking
    /// that `bytes` are JSON lines; why not, otherwise.
    pub fn check(&self, name: &str, bytes: &[u8]) -> Result<String, String> {
        let id = match self {
            SessionFile::ClaudeJsonl => name
                .strip_suffix(".jsonl")
                .filter(|stem| is_uuid(stem))
                .map(str::to_string),
            SessionFile::CodexRollout => codex_rollout(name).map(|(_, id)| id.to_string()),
        }
        .ok_or_else(|| format!("`{name}` is not {}", self.expects()))?;
        let text = std::str::from_utf8(bytes).map_err(|_| format!("`{name}` is not UTF-8 text"))?;
        let mut lines = 0;
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<serde_json::Value>(line) {
                Ok(serde_json::Value::Object(_)) => lines += 1,
                _ => return Err(format!("line {} of `{name}` is not a JSON object", i + 1)),
            }
        }
        if lines == 0 {
            return Err(format!("`{name}` holds no session: it has no JSON lines"));
        }
        Ok(id)
    }

    /// Where a file named `name` goes, relative to the agent's home, for an
    /// agent working in `workdir`. `name` passed [`Self::check`].
    pub fn place(&self, name: &str, workdir: &str) -> String {
        match self {
            SessionFile::ClaudeJsonl => {
                let folder: String = workdir
                    .chars()
                    .map(|c| {
                        if c.is_ascii_alphanumeric() || c == '-' {
                            c
                        } else {
                            '-'
                        }
                    })
                    .collect();
                format!(".claude/projects/{folder}/{name}")
            }
            SessionFile::CodexRollout => {
                let (day, _) = codex_rollout(name).unwrap_or(("1970-01-01", ""));
                format!(".codex/sessions/{}/{name}", day.replace('-', "/"))
            }
        }
    }
}

impl std::str::FromStr for SessionFile {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, ()> {
        SessionFile::ALL
            .into_iter()
            .find(|f| f.as_str() == s)
            .ok_or(())
    }
}

fn is_uuid(s: &str) -> bool {
    uuid::Uuid::try_parse(s).is_ok_and(|_| s.len() == 36)
}

/// A Codex rollout file name's day (`YYYY-MM-DD`) and session id.
fn codex_rollout(name: &str) -> Option<(&str, &str)> {
    let rest = name.strip_prefix("rollout-")?.strip_suffix(".jsonl")?;
    let split = rest.len().checked_sub(37)?;
    let (stamp, id) = (rest.get(..split)?, rest.get(split + 1..)?);
    let shape = "dddd-dd-ddTdd-dd-dd";
    let stamp_ok = stamp.len() == shape.len()
        && stamp.chars().zip(shape.chars()).all(|(c, s)| match s {
            'd' => c.is_ascii_digit(),
            s => c == s,
        });
    (rest.as_bytes()[split] == b'-' && stamp_ok && is_uuid(id)).then(|| (&stamp[..10], id))
}

impl ResumeSpec {
    /// True when the agent can resume a session on `harness`.
    pub fn on(&self, harness: Harness) -> bool {
        match harness {
            Harness::Acp => self.acp,
            Harness::Headless => self.headless,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawResume {
    #[serde(default)]
    acp: bool,
    #[serde(default)]
    headless: bool,
    #[serde(default)]
    session_dirs: Vec<String>,
    load: Option<String>,
}

pub(super) fn validate(
    raw: Option<RawResume>,
    check: &Checker<'_>,
    parts: &Parts<'_>,
) -> Result<ResumeSpec, ManifestError> {
    let Some(raw) = raw else {
        return Ok(ResumeSpec::default());
    };
    check.require(
        !raw.acp || parts.harnesses.contains(&Harness::Acp),
        "resume.acp",
        || "this agent has no acp harness".into(),
    )?;
    if raw.headless {
        let command = parts.headless.map(|h| h.command.as_str());
        check.require(command.is_some(), "resume.headless", || {
            "this agent has no headless harness".into()
        })?;
        check.require(
            command.is_some_and(|c| c.contains("BENCHFLOW_RESUME")),
            "resume.headless",
            || {
                "the [headless] command must resume when $BENCHFLOW_RESUME is set, such as `${BENCHFLOW_RESUME:+--continue}`"
                    .into()
            },
        )?;
    }
    check.require(
        !(raw.acp || raw.headless) || !raw.session_dirs.is_empty(),
        "resume",
        || {
            "`session_dirs` is required: the folders under the home where the agent keeps its sessions, such as `[\".claude/projects\"]`"
                .into()
        },
    )?;
    for (i, dir) in raw.session_dirs.iter().enumerate() {
        check.require(
            !dir.is_empty()
                && !dir.starts_with('/')
                && !dir.split('/').any(|p| p.is_empty() || p == ".."),
            &format!("resume.session_dirs[{i}]"),
            || {
                format!(
                    "`{dir}` must be a folder under the home directory, such as `.claude/projects`"
                )
            },
        )?;
    }
    let load = match &raw.load {
        None => None,
        Some(name) => {
            let format = name.parse::<SessionFile>().ok();
            check.require(format.is_some(), "resume.load", || {
                let known: Vec<&str> = SessionFile::ALL.iter().map(|f| f.as_str()).collect();
                format!(
                    "`{name}` is not a session format BenchFlow can place; known: {}",
                    known.join(", ")
                )
            })?;
            check.require(raw.acp || raw.headless, "resume.load", || {
                "a loaded session is resumed like a stage's: set `acp` or `headless` too".into()
            })?;
            format
        }
    };
    Ok(ResumeSpec {
        acp: raw.acp,
        headless: raw.headless,
        session_dirs: raw.session_dirs,
        load,
    })
}
