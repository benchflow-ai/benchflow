//! `bench handoff`: resume a finished trial's agent session in the agent's
//! own CLI, here.
//!
//! A trial saves the agent's session folders into `agent/sessions/` at the
//! end of its turn (result.json's `agent_session`). The handoff copies the
//! session into the folder the local CLI reads, under the session's own id,
//! and starts the CLI resuming it in the current folder:
//!
//! ```text
//! Claude Code  agent/sessions/.claude/projects/<slug>/<id>.jsonl
//!              -> $CLAUDE_CONFIG_DIR (~/.claude)/projects/<this folder's slug>/<id>.jsonl
//!              claude --resume <id>
//! Codex        agent/sessions/.codex/sessions/YYYY/MM/DD/rollout-...-<id>.jsonl
//!              -> $CODEX_HOME (~/.codex)/sessions/YYYY/MM/DD/rollout-...-<id>.jsonl
//!              codex resume <id>
//! ```
//!
//! A local session with the same id is never overwritten unless asked
//! (`--replace`), and the handoff says what it copied. A trial from outside
//! this project's jobs folder is someone else's: its session becomes the
//! local agent's context, run with the user's permissions, so it is handed
//! off only with `--trust`.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use benchflow_run::record::AgentSession;
use serde_json::json;

use crate::cli::HandoffArgs;
use crate::error::{CliError, Exit, Result};
use crate::ui::{self, dim, ok_mark};
use crate::{note, say};

/// An agent CLI that can resume a saved session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cli {
    Claude,
    Codex,
}

impl Cli {
    /// The CLI for a trial's agent label (`claude`, `codex:headless`).
    fn for_agent(label: &str) -> Option<Self> {
        match label.split(':').next().unwrap_or(label) {
            "claude" => Some(Cli::Claude),
            "codex" => Some(Cli::Codex),
            _ => None,
        }
    }
    fn program(self) -> &'static str {
        match self {
            Cli::Claude => "claude",
            Cli::Codex => "codex",
        }
    }
    fn product(self) -> &'static str {
        match self {
            Cli::Claude => "Claude Code",
            Cli::Codex => "Codex",
        }
    }
    fn install(self) -> &'static str {
        match self {
            Cli::Claude => "npm install -g @anthropic-ai/claude-code",
            Cli::Codex => "npm install -g @openai/codex",
        }
    }
    /// The session folder under the agent's home, as its manifest declares.
    fn session_dir(self) -> &'static str {
        match self {
            Cli::Claude => ".claude/projects",
            Cli::Codex => ".codex/sessions",
        }
    }
    fn command(self, id: &str, extra: &[String]) -> Vec<String> {
        let mut c: Vec<String> = match self {
            Cli::Claude => vec!["claude".into(), "--resume".into(), id.into()],
            Cli::Codex => vec!["codex".into(), "resume".into(), id.into()],
        };
        c.extend(extra.iter().cloned());
        c
    }
}

/// Where the local CLIs keep their state (from the environment, else the
/// home folder), and the folder the session is resumed in.
struct Local {
    cwd: PathBuf,
    home: PathBuf,
    claude_config_dir: Option<PathBuf>,
    codex_home: Option<PathBuf>,
}

impl Local {
    fn from_env(cwd: PathBuf) -> Option<Self> {
        let var = |k: &str| {
            std::env::var_os(k)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        Some(Self {
            cwd,
            home: var("HOME")?,
            claude_config_dir: var("CLAUDE_CONFIG_DIR"),
            codex_home: var("CODEX_HOME"),
        })
    }
    /// The folder the CLI reads sessions from.
    fn sessions(&self, cli: Cli) -> PathBuf {
        match cli {
            Cli::Claude => self
                .claude_config_dir
                .clone()
                .unwrap_or_else(|| self.home.join(".claude"))
                .join("projects"),
            Cli::Codex => self
                .codex_home
                .clone()
                .unwrap_or_else(|| self.home.join(".codex"))
                .join("sessions"),
        }
    }
}

/// One saved session in the trial.
#[derive(Debug, Clone)]
struct Saved {
    id: String,
    /// The session's files, relative to the CLI's sessions folder in the
    /// trial (the transcript first).
    files: Vec<PathBuf>,
    modified: SystemTime,
}

/// A file the handoff places.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Placed {
    /// Copied; nothing was there.
    New,
    /// The same content is already there.
    Same,
    /// A different file was there and is replaced (`--replace`).
    Replaced,
}

#[derive(Debug)]
struct Plan {
    cli: Cli,
    id: String,
    /// (from, to, what happens).
    files: Vec<(PathBuf, PathBuf, Placed)>,
    /// The other sessions the trial saved, by id.
    others: Vec<String>,
    command: Vec<String>,
}

/// Claude Code's folder name for a working folder: every character but
/// ASCII letters, digits, and `-` becomes `-`.
fn claude_slug(cwd: &Path) -> String {
    cwd.to_string_lossy()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' {
                c
            } else {
                '-'
            }
        })
        .collect()
}

/// The largest saved file the handoff copies (the cap the trial saved
/// session files under).
const MAX_FILE_BYTES: u64 = benchflow_run::hardening::MAX_OUTPUT_FILE_BYTES;

/// Whether a session id from the trial is safe to name a file and to pass
/// to the CLI: letters, digits, `-` and `_`, starting with a letter or
/// digit. A trial folder can come from someone else, and the id becomes an
/// argument of the command the handoff starts.
fn safe_id(id: &str) -> bool {
    id.len() <= 128
        && id.chars().next().is_some_and(|c| c.is_ascii_alphanumeric())
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

/// A regular file (not a link) of at most [`MAX_FILE_BYTES`].
fn plain_file(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_file() && m.len() <= MAX_FILE_BYTES)
}

/// A real folder (not a link).
fn plain_dir(p: &Path) -> bool {
    std::fs::symlink_metadata(p).is_ok_and(|m| m.file_type().is_dir())
}

fn files_under(dir: &Path) -> Vec<PathBuf> {
    if !plain_dir(dir) {
        return Vec::new();
    }
    let mut out: Vec<PathBuf> = walkdir::WalkDir::new(dir)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file() && plain_file(e.path()))
        .map(|e| e.into_path())
        .collect();
    out.sort();
    out
}

/// The trial's saved sessions folder for `cli`, when it is a real folder
/// inside the trial: result.json's `path` must be relative, without `..`,
/// and no folder on the way may be a link.
fn sessions_root(trial: &Path, path: &str, cli: Cli) -> Option<PathBuf> {
    let rel = Path::new(path).join(cli.session_dir());
    let mut at = trial.to_path_buf();
    for c in rel.components() {
        let std::path::Component::Normal(part) = c else {
            return None;
        };
        at.push(part);
        if !plain_dir(&at) {
            return None;
        }
    }
    Some(at)
}

fn modified(p: &Path) -> SystemTime {
    std::fs::metadata(p)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH)
}

/// The id a Codex rollout carries: its first line's `session_meta` id, else
/// the UUID its file name ends with.
fn codex_id(file: &Path) -> Option<String> {
    let first = std::fs::read_to_string(file)
        .ok()
        .and_then(|t| t.lines().next().map(str::to_string));
    if let Some(v) = first.and_then(|l| serde_json::from_str::<serde_json::Value>(&l).ok())
        && let Some(id) = v["payload"]["id"].as_str()
    {
        return Some(id.to_string());
    }
    let stem = file.file_stem()?.to_string_lossy().to_string();
    let tail = stem.get(stem.len().checked_sub(36)?..)?;
    (tail.len() == 36 && tail.chars().filter(|c| *c == '-').count() == 4).then(|| tail.into())
}

/// The sessions saved under `root` (the CLI's sessions folder in the
/// trial), newest first.
fn saved_sessions(cli: Cli, root: &Path) -> Vec<Saved> {
    let mut out: Vec<Saved> = Vec::new();
    match cli {
        Cli::Claude => {
            // <slug>/<id>.jsonl, and <slug>/<id>/ (its subagents and tool
            // results) beside it.
            let Ok(projects) = std::fs::read_dir(root) else {
                return out;
            };
            for project in projects.filter_map(|e| e.ok()).map(|e| e.path()) {
                if !plain_dir(&project) {
                    continue;
                }
                let Ok(entries) = std::fs::read_dir(&project) else {
                    continue;
                };
                for f in entries.filter_map(|e| e.ok()).map(|e| e.path()) {
                    if !(plain_file(&f) && f.extension().is_some_and(|x| x == "jsonl")) {
                        continue;
                    }
                    let Some(id) = f
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .filter(|id| safe_id(id))
                    else {
                        continue;
                    };
                    let mut files = vec![f.clone()];
                    files.extend(files_under(&project.join(&id)));
                    out.push(Saved {
                        modified: modified(&f),
                        id,
                        files: files
                            .iter()
                            .filter_map(|p| p.strip_prefix(root).ok().map(Path::to_path_buf))
                            .collect(),
                    });
                }
            }
        }
        Cli::Codex => {
            for f in files_under(root) {
                let name = f.file_name().unwrap_or_default().to_string_lossy();
                if !(name.starts_with("rollout-") && name.ends_with(".jsonl")) {
                    continue;
                }
                let Some(id) = codex_id(&f).filter(|id| safe_id(id)) else {
                    continue;
                };
                out.push(Saved {
                    modified: modified(&f),
                    id,
                    files: f
                        .strip_prefix(root)
                        .map(|p| vec![p.to_path_buf()])
                        .unwrap_or_default(),
                });
            }
        }
    }
    out.sort_by(|a, b| b.modified.cmp(&a.modified).then(a.id.cmp(&b.id)));
    out
}

/// Where a saved session's file goes for the local CLI.
fn destination(cli: Cli, local: &Local, rel: &Path) -> PathBuf {
    let sessions = local.sessions(cli);
    match cli {
        // The project folder is this folder's; the rest keeps its path.
        Cli::Claude => {
            let mut parts = rel.components();
            parts.next();
            sessions.join(claude_slug(&local.cwd)).join(parts.as_path())
        }
        Cli::Codex => sessions.join(rel),
    }
}

fn plan(
    trial: &Path,
    agent: &str,
    session: Option<&AgentSession>,
    wanted: Option<&str>,
    replace: bool,
    extra: &[String],
    local: &Local,
) -> Result<Plan> {
    let Some(cli) = Cli::for_agent(agent) else {
        let saved = session.map(|s| trial.join(&s.path));
        return Err(match (agent.split(':').next().unwrap_or(agent), saved) {
            (a @ ("oracle" | "nop"), _) => {
                CliError::usage(format!("this trial's attempt is the {a}, not an agent: there is no session to hand off"))
                    .next("pass a trial a model agent ran (-a claude or -a codex)")
            }
            (_, Some(dir)) => CliError::usage(format!(
                "bench handoff resumes Claude Code and Codex sessions; this trial's agent is {agent}"
            ))
            .next(format!(
                "its saved session files are in {}: start {agent}'s own CLI on them",
                ui::path(&dir)
            )),
            _ => CliError::usage(format!(
                "bench handoff resumes Claude Code and Codex sessions; this trial's agent is {agent}, which keeps no session BenchFlow can resume"
            ))
            .next("run the task with -a claude or -a codex to hand the trial off"),
        });
    };
    let Some(session) = session else {
        return Err(CliError::usage(format!(
            "this trial saved no {} session",
            cli.product()
        ))
        .next(format!(
            "trials from an older bench did not save it, and an agent that stopped early may not have written one; run the task again: bench run <task> -a {}",
            cli.program()
        )));
    };
    let root = trial.join(&session.path).join(cli.session_dir());
    let Some(root) = sessions_root(trial, &session.path, cli) else {
        return Err(CliError::usage(format!(
            "no {} session in {}",
            cli.product(),
            ui::path(&root)
        ))
        .next("the trial's saved session folder must be a folder inside the trial, not a link; run the task again"));
    };
    let saved = saved_sessions(cli, &root);
    if saved.is_empty() {
        return Err(CliError::usage(format!(
            "no {} session in {}",
            cli.product(),
            ui::path(&root)
        ))
        .next("the trial's saved session folder holds no transcript the handoff can use (links, files over 100 MB, and session ids other than letters, digits, `-` and `_` are left alone); run the task again"));
    }
    let ids = || {
        saved
            .iter()
            .map(|s| s.id.clone())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let chosen = match wanted {
        Some(w) => saved.iter().find(|s| s.id == w).ok_or_else(|| {
            CliError::usage(format!("this trial saved no session {w}"))
                .next(format!("pass one of: {}", ids()))
        })?,
        None => session
            .session_id
            .as_deref()
            .and_then(|id| saved.iter().find(|s| s.id == id))
            .unwrap_or(&saved[0]),
    };
    let mut files = Vec::new();
    for rel in &chosen.files {
        let from = root.join(rel);
        let to = destination(cli, local, rel);
        let placed = match std::fs::read(&to) {
            Err(_) => Placed::New,
            Ok(there) if std::fs::read(&from).is_ok_and(|b| b == there) => Placed::Same,
            Ok(_) if replace => Placed::Replaced,
            Ok(_) => {
                return Err(CliError::usage(format!(
                    "{} already has a different session {} at {}",
                    cli.product(),
                    chosen.id,
                    ui::path(&to)
                ))
                .next(format!(
                    "pass --replace to overwrite it with the trial's, or resume yours with `{}`",
                    cli.command(&chosen.id, &[]).join(" ")
                )));
            }
        };
        files.push((from, to, placed));
    }
    Ok(Plan {
        cli,
        id: chosen.id.clone(),
        files,
        others: saved
            .iter()
            .filter(|s| s.id != chosen.id)
            .map(|s| s.id.clone())
            .collect(),
        command: cli.command(&chosen.id, extra),
    })
}

/// Whether the trial is inside the jobs folder, after links are resolved.
/// A jobs folder that does not exist holds no trial.
fn inside_jobs(trial: &Path, jobs_dir: &Path) -> bool {
    match (trial.canonicalize(), jobs_dir.canonicalize()) {
        (Ok(t), Ok(j)) => t.starts_with(&j) && t != j,
        _ => false,
    }
}

/// The program's path, when it is on `PATH`.
fn on_path(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|d| d.join(program))
        .find(|p| p.is_file())
}

pub fn run(a: HandoffArgs) -> Result<Exit> {
    let trial = a.trial.clone();
    if !trial.is_dir() {
        return Err(
            CliError::usage(format!("{} is not a folder", trial.display()))
                .next("pass a trial folder: bench handoff jobs/<job>/<task>__<id>"),
        );
    }
    if !trial.join("result.json").is_file() {
        let inside = benchflow_run::read::trial_dirs(&trial)
            .next()
            .map(|t| format!(": bench handoff {}", t.display()))
            .unwrap_or_default();
        return Err(CliError::usage(format!(
            "{} is not a trial folder (no result.json)",
            trial.display()
        ))
        .next(format!("pass a trial folder inside a job{inside}")));
    }
    let rec = benchflow_run::read::read_trial(&trial).ok_or_else(|| {
        CliError::usage(format!(
            "{}'s result.json could not be read",
            trial.display()
        ))
    })?;
    let project = crate::context::project()?;
    let jobs_dir = project
        .config
        .run
        .jobs_dir
        .clone()
        .map(|d| project.root.join(d))
        .unwrap_or_else(|| PathBuf::from("jobs"));
    if !a.trust && !inside_jobs(&trial, &jobs_dir) {
        return Err(CliError::usage(format!(
            "{} is outside this project's jobs folder ({}), so it may be someone else's trial",
            trial.display(),
            jobs_dir.display()
        ))
        .next(format!(
            "the session's history becomes your local agent's context, and the agent runs with your permissions on this machine; read the transcript in {} first (or see it with --dry-run --trust), then pass --trust to hand it off",
            ui::path(&trial.join("agent"))
        )));
    }
    let cwd = std::env::current_dir()
        .map_err(|e| CliError::not_ready(format!("the current folder is unreadable: {e}")))?;
    let local = Local::from_env(cwd).ok_or_else(|| {
        CliError::not_ready("HOME is not set, so the agent CLI's folder is unknown")
    })?;
    let p = plan(
        &trial,
        &rec.agent,
        rec.agent_session.as_ref(),
        a.session.as_deref(),
        a.replace,
        &a.args,
        &local,
    )?;
    if !a.dry_run && on_path(p.cli.program()).is_none() {
        return Err(CliError::not_ready(format!(
            "the {} CLI ({}) is not on PATH",
            p.cli.product(),
            p.cli.program()
        ))
        .next(format!(
            "install it ({}), then run this again",
            p.cli.install()
        )));
    }
    if ui::ui().json {
        let files: Vec<_> = p
            .files
            .iter()
            .map(|(from, to, placed)| {
                json!({
                    "from": from, "to": to,
                    "placed": match placed { Placed::New => "new", Placed::Same => "same", Placed::Replaced => "replaced" },
                })
            })
            .collect();
        println!(
            "{}",
            json!({
                "agent": rec.agent,
                "session_id": p.id,
                "other_sessions": p.others,
                "files": files,
                "command": p.command,
                "dry_run": a.dry_run,
            })
        );
    }
    let verb = if a.dry_run { "would copy" } else { "copied" };
    for (from, to, placed) in &p.files {
        match placed {
            Placed::Same => note!("  {} {}", dim("already there"), ui::path(to)),
            Placed::New | Placed::Replaced => {
                if !a.dry_run {
                    if let Some(parent) = to.parent() {
                        std::fs::create_dir_all(parent).map_err(|e| {
                            CliError::failed(format!("could not create {}: {e}", parent.display()))
                        })?;
                    }
                    std::fs::copy(from, to).map_err(|e| {
                        CliError::failed(format!("could not copy to {}: {e}", to.display()))
                    })?;
                }
                let what = if *placed == Placed::Replaced {
                    format!("{verb}, replacing a different session with the same id,")
                } else {
                    verb.to_string()
                };
                note!("  {} {} to {}", dim(what), ui::path(from), ui::path(to));
            }
        }
    }
    if !p.others.is_empty() {
        note!(
            "  {}",
            dim(format!(
                "the trial saved other sessions too ({}); pick one with --session",
                p.others.join(", ")
            ))
        );
    }
    if a.dry_run {
        say!("{}", p.command.join(" "));
        return Ok(Exit::Ok);
    }
    note!(
        "{} resuming {} session {} from {} here: {}",
        ok_mark(),
        p.cli.product(),
        p.id,
        trial.file_name().unwrap_or_default().to_string_lossy(),
        p.command.join(" ")
    );
    exec(&p.command)
}

#[cfg(unix)]
fn exec(command: &[String]) -> Result<Exit> {
    use std::os::unix::process::CommandExt;
    let e = std::process::Command::new(&command[0])
        .args(&command[1..])
        .exec();
    Err(CliError::failed(format!(
        "{} did not start: {e}",
        command[0]
    )))
}

#[cfg(not(unix))]
fn exec(command: &[String]) -> Result<Exit> {
    let status = std::process::Command::new(&command[0])
        .args(&command[1..])
        .status()
        .map_err(|e| CliError::failed(format!("{} did not start: {e}", command[0])))?;
    Ok(if status.success() {
        Exit::Ok
    } else {
        Exit::Failed
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(id: Option<&str>) -> AgentSession {
        AgentSession {
            path: "agent/sessions".into(),
            dirs: vec![".claude/projects".into()],
            files: 1,
            bytes: 1,
            session_id: id.map(str::to_string),
            left_out: Vec::new(),
        }
    }

    fn write(p: &Path, text: &str) {
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    fn local(home: &Path) -> Local {
        Local {
            cwd: PathBuf::from("/w"),
            home: home.to_path_buf(),
            claude_config_dir: None,
            codex_home: None,
        }
    }

    #[test]
    fn only_trials_inside_the_jobs_folder_count_as_this_projects() {
        let tmp = tempfile::tempdir().unwrap();
        let jobs = tmp.path().join("jobs");
        let mine = jobs.join("j/t__1");
        std::fs::create_dir_all(&mine).unwrap();
        let theirs = tmp.path().join("Downloads/j/t__1");
        std::fs::create_dir_all(&theirs).unwrap();
        assert!(inside_jobs(&mine, &jobs));
        assert!(inside_jobs(&jobs.join("j/../j/t__1"), &jobs));
        assert!(!inside_jobs(&theirs, &jobs));
        assert!(!inside_jobs(&jobs, &jobs));
        assert!(!inside_jobs(&jobs.join("../Downloads/j/t__1"), &jobs));
        assert!(!inside_jobs(&mine, &tmp.path().join("no-such-jobs")));
        // A link in the jobs folder to a trial elsewhere is not this project's.
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&theirs, jobs.join("linked")).unwrap();
            assert!(!inside_jobs(&jobs.join("linked"), &jobs));
        }
    }

    #[test]
    fn claude_sessions_go_to_this_folders_project_under_their_own_id() {
        let tmp = tempfile::tempdir().unwrap();
        let trial = tmp.path().join("trial");
        let saved = trial.join("agent/sessions/.claude/projects/-app");
        write(&saved.join("aaa.jsonl"), "{}\n");
        write(&saved.join("aaa/subagents/agent-1.jsonl"), "{}\n");
        write(&saved.join("bbb.jsonl"), "{}\n");
        let home = tmp.path().join("home");
        let p = plan(
            &trial,
            "claude",
            Some(&session(Some("aaa"))),
            None,
            false,
            &["--version".into()],
            &Local {
                cwd: PathBuf::from("/Users/me/my.repo"),
                ..local(&home)
            },
        )
        .unwrap();
        assert_eq!(p.id, "aaa", "the session the harness recorded");
        assert_eq!(p.others, vec!["bbb".to_string()]);
        assert_eq!(p.command, ["claude", "--resume", "aaa", "--version"]);
        let project = home.join(".claude/projects/-Users-me-my-repo");
        let to: Vec<_> = p.files.iter().map(|(_, to, _)| to.clone()).collect();
        assert_eq!(
            to,
            [
                project.join("aaa.jsonl"),
                project.join("aaa/subagents/agent-1.jsonl")
            ]
        );
        assert!(p.files.iter().all(|(_, _, placed)| *placed == Placed::New));
    }

    #[test]
    fn a_different_local_session_with_the_same_id_is_kept_unless_replace_is_given() {
        let tmp = tempfile::tempdir().unwrap();
        let trial = tmp.path().join("trial");
        write(
            &trial.join("agent/sessions/.claude/projects/-app/aaa.jsonl"),
            "trial\n",
        );
        let home = tmp.path().join("home");
        let mine = home.join(".claude/projects/-w/aaa.jsonl");
        write(&mine, "mine\n");
        let s = session(None);
        let e = plan(&trial, "claude", Some(&s), None, false, &[], &local(&home)).unwrap_err();
        assert!(
            e.message.contains("already has a different session aaa"),
            "{}",
            e.message
        );
        assert!(e.next.unwrap().contains("--replace"));
        let p = plan(&trial, "claude", Some(&s), None, true, &[], &local(&home)).unwrap();
        assert_eq!(p.files[0].2, Placed::Replaced);
        write(&mine, "trial\n");
        let p = plan(&trial, "claude", Some(&s), None, false, &[], &local(&home)).unwrap();
        assert_eq!(p.files[0].2, Placed::Same, "the same file is left alone");
    }

    #[test]
    fn codex_rollouts_keep_their_path_and_resume_by_their_id() {
        let tmp = tempfile::tempdir().unwrap();
        let trial = tmp.path().join("trial");
        let id = "0199a213-81c0-7800-8aa1-bbab2a035a53";
        let rel = format!("2026/10/10/rollout-2026-10-10T09-00-00-{id}.jsonl");
        write(
            &trial.join("agent/sessions/.codex/sessions").join(&rel),
            &format!("{{\"type\":\"session_meta\",\"payload\":{{\"id\":\"{id}\"}}}}\n"),
        );
        let home = tmp.path().join("home");
        let mut l = local(&home);
        l.codex_home = Some(tmp.path().join("codex-home"));
        let p = plan(
            &trial,
            "codex:headless",
            Some(&session(None)),
            None,
            false,
            &[],
            &l,
        )
        .unwrap();
        assert_eq!(p.command, ["codex", "resume", id]);
        assert_eq!(
            p.files[0].1,
            tmp.path().join("codex-home/sessions").join(&rel)
        );
    }

    #[test]
    fn the_newest_session_is_resumed_and_an_unknown_one_is_refused() {
        let tmp = tempfile::tempdir().unwrap();
        let trial = tmp.path().join("trial");
        let dir = trial.join("agent/sessions/.claude/projects/-app");
        write(&dir.join("old.jsonl"), "{}\n");
        write(&dir.join("new.jsonl"), "{}\n");
        let old = std::fs::File::options()
            .write(true)
            .open(dir.join("old.jsonl"))
            .unwrap();
        old.set_modified(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(10))
            .unwrap();
        let home = tmp.path().join("home");
        let s = session(None);
        let p = plan(&trial, "claude", Some(&s), None, false, &[], &local(&home)).unwrap();
        assert_eq!(p.id, "new");
        let p = plan(
            &trial,
            "claude",
            Some(&s),
            Some("old"),
            false,
            &[],
            &local(&home),
        )
        .unwrap();
        assert_eq!(p.id, "old");
        let e = plan(
            &trial,
            "claude",
            Some(&s),
            Some("x"),
            false,
            &[],
            &local(&home),
        )
        .unwrap_err();
        assert!(e.next.unwrap().contains("new, old"));
    }

    #[test]
    fn session_ids_that_are_not_plain_words_are_never_handed_off() {
        let tmp = tempfile::tempdir().unwrap();
        let trial = tmp.path().join("trial");
        let dir = trial.join("agent/sessions/.claude/projects/-app");
        write(&dir.join("-flag.jsonl"), "{}\n");
        write(&dir.join("..jsonl"), "{}\n");
        write(&dir.join("ok-1.jsonl"), "{}\n");
        let home = tmp.path().join("home");
        let s = session(Some("-flag"));
        let p = plan(&trial, "claude", Some(&s), None, false, &[], &local(&home)).unwrap();
        assert_eq!(p.id, "ok-1");
        assert!(p.others.is_empty(), "{:?}", p.others);
        let e = plan(
            &trial,
            "claude",
            Some(&s),
            Some("-flag"),
            false,
            &[],
            &local(&home),
        )
        .unwrap_err();
        assert!(e.message.contains("saved no session"), "{}", e.message);

        let codex = tmp.path().join("codex-trial");
        write(
            &codex.join("agent/sessions/.codex/sessions/2026/10/10/rollout-x.jsonl"),
            "{\"type\":\"session_meta\",\"payload\":{\"id\":\"--some-flag\"}}\n",
        );
        let e = plan(
            &codex,
            "codex",
            Some(&session(None)),
            None,
            false,
            &[],
            &local(&home),
        )
        .unwrap_err();
        assert!(e.message.contains("no Codex session"), "{}", e.message);
    }

    #[cfg(unix)]
    #[test]
    fn links_and_paths_out_of_the_trial_are_not_followed() {
        use std::os::unix::fs::symlink;
        let tmp = tempfile::tempdir().unwrap();
        let elsewhere = tmp.path().join("elsewhere/.claude/projects/-x");
        write(&elsewhere.join("aaa.jsonl"), "{}\n");
        let home = tmp.path().join("home");

        // A path in result.json that leaves the trial.
        let trial = tmp.path().join("t1");
        std::fs::create_dir_all(&trial).unwrap();
        for path in ["../elsewhere", "/"] {
            let mut s = session(None);
            s.path = path.into();
            let e = plan(&trial, "claude", Some(&s), None, false, &[], &local(&home)).unwrap_err();
            assert!(
                e.message.contains("no Claude Code session"),
                "{path}: {}",
                e.message
            );
        }

        // A sessions folder, a project folder, or a transcript that is a link.
        let trial = tmp.path().join("t2");
        std::fs::create_dir_all(trial.join("agent")).unwrap();
        symlink(tmp.path().join("elsewhere"), trial.join("agent/sessions")).unwrap();
        assert!(
            plan(
                &trial,
                "claude",
                Some(&session(None)),
                None,
                false,
                &[],
                &local(&home)
            )
            .is_err()
        );

        let trial = tmp.path().join("t3");
        std::fs::create_dir_all(trial.join("agent/sessions/.claude/projects")).unwrap();
        symlink(
            &elsewhere,
            trial.join("agent/sessions/.claude/projects/-app"),
        )
        .unwrap();
        assert!(
            plan(
                &trial,
                "claude",
                Some(&session(None)),
                None,
                false,
                &[],
                &local(&home)
            )
            .is_err()
        );

        let trial = tmp.path().join("t4");
        let dir = trial.join("agent/sessions/.claude/projects/-app");
        std::fs::create_dir_all(dir.join("bbb")).unwrap();
        symlink(elsewhere.join("aaa.jsonl"), dir.join("aaa.jsonl")).unwrap();
        write(&dir.join("bbb.jsonl"), "{}\n");
        symlink(elsewhere.join("aaa.jsonl"), dir.join("bbb/linked.jsonl")).unwrap();
        let p = plan(
            &trial,
            "claude",
            Some(&session(None)),
            None,
            false,
            &[],
            &local(&home),
        )
        .unwrap();
        assert_eq!(p.id, "bbb");
        assert_eq!(p.files.len(), 1, "{:?}", p.files);
    }

    #[test]
    fn agents_without_a_resumable_session_are_refused_with_a_next_step() {
        let tmp = tempfile::tempdir().unwrap();
        let l = local(tmp.path());
        let e = plan(tmp.path(), "oracle", None, None, false, &[], &l).unwrap_err();
        assert!(
            e.message.contains("the oracle, not an agent"),
            "{}",
            e.message
        );
        let e = plan(tmp.path(), "aider", None, None, false, &[], &l).unwrap_err();
        assert!(e.message.contains("keeps no session"), "{}", e.message);
        assert!(e.next.unwrap().contains("-a claude"));
        let mut s = session(None);
        s.dirs = vec![".gemini/tmp".into()];
        let e = plan(tmp.path(), "gemini", Some(&s), None, false, &[], &l).unwrap_err();
        assert!(e.next.unwrap().contains("agent/sessions"));
        let e = plan(tmp.path(), "claude", None, None, false, &[], &l).unwrap_err();
        assert!(
            e.message.contains("saved no Claude Code session"),
            "{}",
            e.message
        );
        assert!(e.next.unwrap().contains("bench run <task> -a claude"));
    }
}
