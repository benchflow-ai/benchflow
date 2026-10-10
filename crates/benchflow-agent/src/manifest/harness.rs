//! Which harnesses an agent declares, and the `[headless]` table.
//!
//! An agent declares `acp` when its manifest has a `[launch]` table, or has
//! no `[headless]` table (every manifest written before headless harnesses
//! existed is an ACP agent). It declares `headless` when it has a
//! `[headless]` table. The top-level `harness` key picks the default; it
//! must name one the manifest declares.

use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use super::{Checker, ManifestError, is_env_name};
use crate::headless::OutputFormat;

/// How BenchFlow drives an agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Harness {
    /// The Agent Client Protocol over the agent's stdin and stdout.
    Acp,
    /// The agent's own non-interactive mode, its output stream parsed.
    Headless,
}

impl Harness {
    /// Every harness a manifest can declare, in the order `bench agents` lists them.
    pub const ALL: [Harness; 2] = [Harness::Acp, Harness::Headless];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Acp => "acp",
            Self::Headless => "headless",
        }
    }

    /// `acp` or `headless`.
    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|h| h.as_str() == name)
    }
}

impl fmt::Display for Harness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Variables BenchFlow sets for a headless command; a manifest cannot.
pub const RESERVED_ENV: [&str; 6] = [
    "BENCHFLOW_RESUME",
    "BENCHFLOW_INSTRUCTION",
    "BENCHFLOW_INSTRUCTION_FILE",
    "BENCHFLOW_MODEL",
    "BENCHFLOW_EFFORT",
    "HOME",
];

/// The `[headless]` table: how to run the agent without ACP.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HeadlessSpec {
    /// POSIX `sh` source that runs the agent on the instruction and exits.
    pub command: String,
    /// The parser that reads its stdout.
    pub output: OutputFormat,
    /// Non-secret variables for the command.
    pub env: IndexMap<String, String>,
    /// Files written in the agent's home before it starts.
    pub files: Vec<HomeFile>,
}

/// A file the headless command needs in the agent's home (a settings file).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HomeFile {
    /// Relative to the agent user's home (`.gemini/settings.json`).
    pub path: String,
    pub contents: String,
    /// Unix permissions. Default `0o644`.
    pub mode: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RawHeadless {
    command: Option<String>,
    output: Option<String>,
    #[serde(default)]
    env: IndexMap<String, String>,
    #[serde(default)]
    files: Vec<RawHomeFile>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawHomeFile {
    path: String,
    contents: Option<String>,
    /// One of BenchFlow's runner scripts, instead of `contents`.
    runner: Option<String>,
    mode: Option<u32>,
}

impl RawHeadless {
    pub(super) fn validate(self, check: &Checker<'_>) -> Result<HeadlessSpec, ManifestError> {
        let Some(command) = self.command else {
            return Err(check.fail(
                "headless",
                "`command` is required: the shell command that runs the agent on \
                 \"$BENCHFLOW_INSTRUCTION_FILE\" and exits",
            ));
        };
        check.require(!command.trim().is_empty(), "headless.command", || {
            "is empty".into()
        })?;
        let Some(output) = self.output else {
            return Err(check.fail(
                "headless",
                format!(
                    "`output` is required: the format of the command's stdout, one of {}",
                    OutputFormat::names().join(", ")
                ),
            ));
        };
        let output = OutputFormat::parse(&output).ok_or_else(|| {
            let hint = crate::suggest::did_you_mean(&output, OutputFormat::names())
                .map(|s| format!("did you mean `{s}`? "))
                .unwrap_or_default();
            check.fail(
                "headless.output",
                format!(
                    "unknown output format `{output}`; {hint}use one of {}",
                    OutputFormat::names().join(", ")
                ),
            )
        })?;
        for name in self.env.keys() {
            let key = format!("headless.env.{name}");
            check.require(is_env_name(name), &key, || {
                format!("`{name}` is not an environment variable name")
            })?;
            check.require(!RESERVED_ENV.contains(&name.as_str()), &key, || {
                format!("`{name}` is set by BenchFlow for every headless run; remove it")
            })?;
        }
        let mut files: Vec<HomeFile> = Vec::new();
        for (i, file) in self.files.into_iter().enumerate() {
            let key = format!("headless.files[{i}]");
            let path = file.path.trim_start_matches("./").to_string();
            check.require(
                !path.is_empty()
                    && !path.starts_with('/')
                    && !path.starts_with('$')
                    && !path.ends_with('/')
                    && !path.split('/').any(|part| part.is_empty() || part == ".."),
                &key,
                || {
                    format!(
                        "`path = \"{}\"` must be a file under the agent's home, written relative to it, such as `.gemini/settings.json`",
                        file.path
                    )
                },
            )?;
            check.require(!files.iter().any(|f| f.path == path), &key, || {
                format!("`{path}` is listed twice")
            })?;
            let contents = match (file.contents, &file.runner) {
                (Some(_), Some(_)) => {
                    return Err(check.fail(
                        &key,
                        "use either `contents` or `runner`, not both",
                    ));
                }
                (None, Some(name)) => crate::runners::get(name)
                    .ok_or_else(|| {
                        let hint = crate::suggest::did_you_mean(name, crate::runners::names())
                            .map(|s| format!("did you mean `{s}`? "))
                            .unwrap_or_default();
                        check.fail(
                            &key,
                            format!(
                                "`runner = \"{name}\"` is not one of BenchFlow's runner scripts; {hint}use one of {}",
                                crate::runners::names().join(", ")
                            ),
                        )
                    })?
                    .to_string(),
                (contents, None) => contents.unwrap_or_default(),
            };
            let mode = file.mode.unwrap_or(0o644);
            check.require(mode <= 0o777 && mode & 0o400 != 0, &key, || {
                format!(
                    "`mode = {mode:#o}` must let the agent read the file and set no special bits; write it in octal, such as `0o600`"
                )
            })?;
            files.push(HomeFile {
                path,
                contents,
                mode,
            });
        }
        Ok(HeadlessSpec {
            command,
            output,
            env: self.env,
            files,
        })
    }
}

/// The harnesses a manifest declares, and its default (the `harness` key).
pub(super) fn resolve(
    check: &Checker<'_>,
    harness: Option<&str>,
    acp: bool,
    headless: bool,
) -> Result<(Harness, Vec<Harness>), ManifestError> {
    let declared: Vec<Harness> = Harness::ALL
        .into_iter()
        .filter(|h| match h {
            Harness::Acp => acp,
            Harness::Headless => headless,
        })
        .collect();
    let chosen = match harness {
        None => declared[0],
        Some(name) => {
            let names: Vec<&str> = Harness::ALL.iter().map(|h| h.as_str()).collect();
            let Some(h) = Harness::parse(name) else {
                let hint = crate::suggest::did_you_mean(name, names.iter().copied())
                    .map(|s| format!("did you mean `{s}`? "))
                    .unwrap_or_default();
                return Err(check.fail(
                    "harness",
                    format!(
                        "unknown harness `{name}`; {hint}use one of {}",
                        names.join(", ")
                    ),
                ));
            };
            check.require(declared.contains(&h), "harness", || match h {
                Harness::Acp => {
                    "`acp` needs a `[launch]` table saying how to start the agent's ACP server"
                        .into()
                }
                Harness::Headless => {
                    "`headless` needs a `[headless]` table with the command and its output format"
                        .into()
                }
            })?;
            h
        }
    };
    Ok((chosen, declared))
}
