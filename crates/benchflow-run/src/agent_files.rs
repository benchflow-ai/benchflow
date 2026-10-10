//! Files the run puts in place for a model agent before it starts: the
//! trial's skills ([`crate::skills`]) and a session loaded from a file
//! (`bench run --load-trajectory`, Harbor's flag of the same name).
//!
//! A loaded session goes where the agent's CLI keeps its sessions, as its
//! manifest's `[resume] load` format says (Claude Code's
//! `~/.claude/projects/<folder>/<id>.jsonl`, Codex's
//! `~/.codex/sessions/<day>/rollout-*.jsonl`), owned by the agent user, and
//! the first prompt resumes it as a later stage resumes the one before:
//! ACP `session/load` with the file's session id, or headless with
//! `$BENCHFLOW_RESUME` set. Later stages start fresh unless the run also
//! asks for `--resume-sessions`. config.json records both under `skills`
//! and `load_trajectory`.

use std::collections::BTreeSet;
use std::path::Path;

use benchflow_sandbox::shell;
use serde_json::json;
use sha2::Digest;

use super::{SETUP_TIMEOUT, Trial, in_package, sandbox_problem};
use crate::agent::{Attempt, ModelAgent, Resume};
use crate::hardening::Report;
use crate::outcome::{Owner, Phase, Problem};
use crate::skills;

impl Trial<'_> {
    /// Copy the trial's skills into the agent's skill folders, and record
    /// what happened (or why nothing did) for config.json.
    pub(super) async fn place_skills(&mut self) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        let opts = &self.rt.options;
        let declared = task
            .sandbox
            .skills
            .as_ref()
            .map(|p| {
                p.strip_prefix(&task.root)
                    .unwrap_or(p)
                    .display()
                    .to_string()
            })
            .or(task.sandbox.skills_dir.clone());
        if declared.is_none() && opts.skills.is_empty() {
            return Ok(());
        }
        let task_folder = task
            .sandbox
            .skills
            .as_ref()
            .filter(|_| opts.task_skills)
            .map(|p| in_package(&task.root, p));
        let task_sandbox = task.sandbox.skills_dir.clone().filter(|_| opts.task_skills);
        let run_names: Vec<String> = opts.skills.iter().map(|p| skills::name(p)).collect();
        let mut record = json!({
            "task": declared,
            "task_skills": opts.task_skills,
            "run": opts.skills,
            "names": [],
            "placed_in": [],
        });
        let why = match &self.spec.attempt {
            Attempt::Model(agent) => {
                let paths = agent.skill_paths();
                if task_folder.is_none() && task_sandbox.is_none() && run_names.is_empty() {
                    Some(self.left_out())
                } else if paths.is_empty() {
                    Some(format!(
                        "{} has no skills folder (its manifest names no skill_paths), so it got none of the skills",
                        agent.name()
                    ))
                } else {
                    let targets: Vec<(String, String)> = paths
                        .iter()
                        .filter_map(|p| skills::expand(p, &self.home, &self.workdir))
                        .collect();
                    let (names, missing) = self
                        .copy_skills(&targets, task_folder.as_deref(), task_sandbox.as_deref())
                        .await?;
                    record["names"] = json!(names);
                    record["placed_in"] = json!(targets.iter().map(|t| &t.1).collect::<Vec<_>>());
                    missing.map(|d| {
                        format!("the task's skills folder {d} is not in the image, so only the run's skills were installed")
                    })
                }
            }
            _ => Some(format!(
                "{} runs without skills; only model agents get them",
                self.spec.attempt.label()
            )),
        };
        if let Some(why) = why {
            if matches!(self.spec.attempt, Attempt::Model(_)) {
                self.rt.events.log(self.name(), format!("skills: {why}"));
            }
            record["why"] = json!(why);
        }
        self.skills = Some(record);
        Ok(())
    }

    /// Name the agent's first skills folder in `env` as
    /// `BENCHFLOW_SKILLS_DIR` when skills were placed there, for CLIs that
    /// take the folder as a flag or a config key; drop the variable
    /// otherwise, so a command can test it.
    pub(super) fn skills_env(&self, env: &mut std::collections::BTreeMap<String, String>) {
        let placed = self.skills.as_ref().and_then(|r| {
            let has_names = r["names"].as_array().is_some_and(|n| !n.is_empty());
            r["placed_in"][0].as_str().filter(|_| has_names)
        });
        match placed {
            Some(dir) => {
                env.insert(skills::DIR_ENV.into(), dir.to_string());
            }
            None => {
                env.remove(skills::DIR_ENV);
            }
        }
    }

    /// Why a `--no-task-skills` trial has none, and whether its image has
    /// them anyway.
    fn left_out(&self) -> String {
        let task = self.task();
        let baked = match &task.sandbox.image {
            benchflow_task::package::ImageRef::Build { dockerfile, .. } => {
                std::fs::read_to_string(in_package(&task.root, dockerfile))
                    .unwrap_or_default()
                    .lines()
                    .any(|l| {
                        let words: Vec<&str> = l.split_whitespace().collect();
                        matches!(
                            words.first().map(|w| w.to_ascii_uppercase()).as_deref(),
                            Some("COPY" | "ADD")
                        ) && words
                            .iter()
                            .skip(1)
                            .any(|w| w.trim_end_matches('/').ends_with("skills"))
                    })
            }
            _ => false,
        };
        if baked {
            "--no-task-skills left out the task's skills, but its Dockerfile copies a skills folder into the image, so the agent may still find them".into()
        } else {
            "--no-task-skills left out the task's skills".into()
        }
    }

    /// Upload the skills and copy them into each target; the names in
    /// place, and the image's skills folder when it was missing.
    async fn copy_skills(
        &mut self,
        targets: &[(String, String)],
        task_folder: Option<&Path>,
        task_sandbox: Option<&str>,
    ) -> Result<(Vec<String>, Option<String>), Problem> {
        let unreadable = |what: &Path| {
            Problem::new(
                "R107",
                Owner::You,
                Phase::Setup,
                format!("the skills folder {} cannot be read", what.display()),
                "check that the folder exists and you can read it",
            )
        };
        if let Some(folder) = task_folder {
            // The folder must be the package's own: a path with `..`, an
            // absolute path, or a link would upload a folder of this machine
            // to the agent.
            let root = std::fs::canonicalize(&self.spec.task.root).ok();
            let inside = std::fs::canonicalize(folder)
                .ok()
                .zip(root)
                .is_some_and(|(f, r)| f.starts_with(&r) && f != r);
            if folder.exists() && !inside {
                return Err(Problem::new(
                    "R107",
                    Owner::Task,
                    Phase::Setup,
                    format!(
                        "the task's skills folder {} is outside the task's folder",
                        folder.display()
                    ),
                    "keep the skills in a folder of the task (no `..`, absolute paths, or links), or run with --no-task-skills",
                ));
            }
            if !folder.is_dir() {
                return Err(Problem::new(
                    "R107",
                    Owner::Task,
                    Phase::Setup,
                    format!("the task's skills folder {} is missing", folder.display()),
                    "fix the task's skills folder, or run with --no-task-skills",
                ));
            }
            if let Some(bytes) = skills::too_large(folder, skills::MAX_SKILLS_BYTES) {
                return Err(too_large(folder, bytes, Owner::Task));
            }
            self.sb()
                .upload(folder, skills::STAGING)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        }
        let run = self.rt.options.skills.clone();
        for dir in &run {
            if !dir.is_dir() {
                return Err(unreadable(dir));
            }
            if let Some(bytes) = skills::too_large(dir, skills::MAX_SKILLS_BYTES) {
                return Err(too_large(dir, bytes, Owner::You));
            }
            self.sb()
                .upload(
                    dir,
                    &format!("{}.run/{}", skills::STAGING, skills::name(dir)),
                )
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        }
        let run_names: Vec<String> = run.iter().map(|p| skills::name(p)).collect();
        let script = skills::place_script(&self.user, targets, task_sandbox, &run_names);
        let out = self
            .root(script, SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        if !out.success() {
            return Err(Problem::new(
                "R104",
                Owner::Task,
                Phase::Setup,
                format!("could not give the agent user `{}` its skills", self.user),
                "the image needs mkdir, cp, and chown; or run without skills (--no-task-skills, no --skill)",
            )
            .detail(out.tail(2000)));
        }
        let report = Report::parse(&out.stdout);
        let first = targets.first().map(|t| t.1.as_str()).unwrap_or_default();
        let names: BTreeSet<String> = report
            .all("skill")
            .iter()
            .filter_map(|l| l.strip_prefix(first)?.strip_prefix(' '))
            .map(str::to_string)
            .collect();
        let missing = report.first("skills-missing").map(str::to_string);
        Ok((names.into_iter().collect(), missing))
    }

    /// Put the `--load-trajectory` session where the agent's CLI resumes
    /// it, owned by the agent user; the session the first prompt resumes.
    pub(super) async fn place_session(
        &mut self,
        agent: &dyn ModelAgent,
    ) -> Result<Option<Resume>, Problem> {
        let Some(path) = self.rt.options.load_trajectory.clone() else {
            return Ok(None);
        };
        let Some(format) = agent.session_file() else {
            return Err(Problem::new(
                "R215",
                Owner::You,
                Phase::Setup,
                format!(
                    "{} cannot load a session from a file on this harness",
                    agent.name()
                ),
                "drop --load-trajectory, or run an agent whose manifest has [resume] load (claude, codex)",
            ));
        };
        let unreadable = |why: String| {
            Problem::new(
                "R107",
                Owner::You,
                Phase::Setup,
                format!("the --load-trajectory file {} {why}", path.display()),
                format!("pass {}", format.expects()),
            )
        };
        // As large as a session file the trial saves may be.
        let bytes = crate::agent::read_capped(&path, crate::hardening::MAX_OUTPUT_FILE_BYTES)
            .map_err(|e| unreadable(format!("cannot be read: {e}")))?;
        let name = skills::name(&path);
        let id = format
            .check(&name, &bytes)
            .map_err(|why| unreadable(format!("is refused: {why}")))?;
        let home = self.home.trim_end_matches('/').to_string();
        let rel = format.place(&name, &self.workdir);
        let target = format!("{home}/{rel}");
        self.sb()
            .write_file(&target, &bytes, 0o600)
            .await
            .map_err(|e| sandbox_problem(Phase::Setup, e))?;
        if self.user != "root" {
            let owner = shell::quote(&format!("{}:", self.user));
            let mut paths = Vec::new();
            let mut at = home.clone();
            for part in rel.split('/') {
                at = format!("{at}/{part}");
                paths.push(shell::quote(&at));
            }
            let out = self
                .root(format!("chown {owner} {}", paths.join(" ")), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Setup, e))?;
            if !out.success() {
                return Err(Problem::new(
                    "R104",
                    Owner::Task,
                    Phase::Setup,
                    format!(
                        "could not give the agent user `{}` the loaded session",
                        self.user
                    ),
                    "the image needs chown",
                )
                .detail(out.tail(2000)));
            }
        }
        // The trial keeps what it started from.
        let kept = self.spec.dir.join("loaded-session");
        let _ = std::fs::create_dir_all(&kept);
        let _ = std::fs::write(kept.join(&name), &bytes);
        self.loaded = Some(json!({
            "path": path,
            "format": format.as_str(),
            "session_id": id,
            "sha256": hex::encode(sha2::Sha256::digest(&bytes)),
            "placed_at": target,
        }));
        Ok(Some(Resume {
            session_id: Some(id),
            loaded: true,
        }))
    }
}

/// A skills folder over [`skills::MAX_SKILLS_BYTES`] (R107).
fn too_large(folder: &Path, bytes: u64, owner: Owner) -> Problem {
    Problem::new(
        "R107",
        owner,
        Phase::Setup,
        format!(
            "the skills folder {} holds more than {} MB ({} MB)",
            folder.display(),
            skills::MAX_SKILLS_BYTES / (1024 * 1024),
            bytes / (1024 * 1024)
        ),
        "skills are instructions and small scripts; move large data out of the folder",
    )
}
