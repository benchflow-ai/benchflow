//! What a trial collects before grading: the agent's own session files,
//! the task's hooks (commands that save runtime state into files), its
//! declared outputs with an artifacts manifest, and the separate verifier's
//! image.
//!
//! ```text
//! <trial>/
//!   agent/sessions/<dir>/...  the agent's session folders, as in its home
//!   hooks/<n>.txt             each hook's command, exit code, and output
//!   artifacts/<save_as>       each declared output
//!   artifacts/manifest.json   one entry per output, in Harbor's shape
//! ```

use std::path::{Path, PathBuf};

use benchflow_sandbox::{Exec, ImageSpec};
use benchflow_task::TaskPackage;
use benchflow_task::package::Output;
use serde_json::{Value, json};

use super::{SETUP_TIMEOUT, Trial, sandbox_problem};
use crate::agent::Attempt;
use crate::copy_out;
use crate::hardening::{self, Report};
use crate::images;
use crate::outcome::{DefenseState, Owner, Phase, Problem, defense};
use crate::record::{AgentSession, SESSIONS_DIR};

/// The separate verifier's own image, when the task names one.
pub(super) fn verifier_image_spec(task: &TaskPackage) -> Option<ImageSpec> {
    task.extras
        .verifier_image
        .as_ref()
        .map(|v| images::spec_for_image(task, &v.image, &format!("{}-verifier", task.name)))
}

fn hook_problem(n: usize, message: String, detail: String) -> Problem {
    Problem::new(
        "R311",
        Owner::Task,
        Phase::Handoff,
        format!("hook {n} before grading {message}"),
        format!("hooks/{n}.txt in the trial folder has its output; fix the command in the task"),
    )
    .detail(detail)
}

impl Trial<'_> {
    /// Save the agent's session folders (its manifest's `[resume]
    /// session_dirs`) into `agent/sessions/`, at the end of its turn (each
    /// stage's, so the last one's wins), for `bench handoff`. They are packed
    /// under the output caps; login files are never saved. A session that
    /// could not be saved is logged, never a failure of the trial.
    ///
    /// Root packs the folders, so the agent user's processes are killed
    /// first (`kill`): one still running could otherwise swap a folder for
    /// a link while root reads it. If the kill fails, nothing is saved.
    pub(super) async fn save_session(&mut self, kill: bool) {
        let Attempt::Model(agent) = &self.spec.attempt else {
            return;
        };
        let Some(dirs) = agent.session_dirs() else {
            return;
        };
        if self.sandbox.is_none() || self.home.is_empty() {
            return;
        }
        if kill && self.user != "root" {
            let killed = match self
                .root(hardening::kill_user(&self.user), SETUP_TIMEOUT)
                .await
            {
                Ok(out) if out.success() => Report::parse(&out.stdout)
                    .first("killed")
                    .and_then(|n| n.parse::<u64>().ok()),
                _ => None,
            };
            let Some(n) = killed else {
                self.rt.events.log(
                    self.name(),
                    "the agent's session was not saved: its processes could not be stopped first",
                );
                return;
            };
            self.killed_before_save += n;
        }
        self.hold_watch();
        let saved = self.collect_session(&dirs).await;
        self.release_watch(hardening::SESSIONS_TAR).await;
        match saved {
            Ok(saved) => self.agent_session = saved,
            Err(e) => self.rt.events.log(
                self.name(),
                format!("the agent's session was not saved: {e}"),
            ),
        }
    }

    async fn collect_session(&mut self, dirs: &[String]) -> Result<Option<AgentSession>, String> {
        let home = self.home.trim_end_matches('/').to_string();
        let paths: Vec<String> = dirs.iter().map(|d| format!("{home}/{d}")).collect();
        let out = self
            .root(
                hardening::pack_into(&paths, hardening::SESSIONS_TAR),
                SETUP_TIMEOUT,
            )
            .await
            .map_err(|e| e.to_string())?;
        if !out.success() {
            return Err(format!("packing failed: {}", out.tail(300)));
        }
        let report = Report::parse(&out.stdout);
        let in_home = |p: &str| p.strip_prefix(&format!("{home}/")).unwrap_or(p).to_string();
        let mut left_out: Vec<String> = report
            .all("skipped-special")
            .into_iter()
            .chain(report.all("skipped-large"))
            .map(in_home)
            .collect();
        let agent_dir = self.spec.dir.join("agent");
        let _ = std::fs::create_dir_all(&agent_dir);
        let archive = agent_dir.join("sessions.tar");
        // Under a byte cap and a deadline (reviews F2-A-r67 P1, F2-A-r8 M1):
        // a root agent's leftover process can swap the archive for a huge
        // sparse file or a folder, or make the copy never end. Either way
        // the session is not saved, and no partial archive stays in the
        // trial folder.
        copy_out::copy_out(
            self.sb(),
            hardening::SESSIONS_TAR,
            &archive,
            copy_out::ARCHIVE_CAP,
            copy_out::COPY_OUT_TIMEOUT,
            copy_out::Shape::File,
        )
        .await
        .map_err(|e| format!("copying it out failed: {e}"))?;
        let dest = self.spec.dir.join(SESSIONS_DIR);
        let _ = std::fs::remove_dir_all(&dest);
        let home_rel = PathBuf::from(home.trim_start_matches('/'));
        let in_session = |rel: &Path| {
            let rest = rel.strip_prefix(&home_rel).ok()?;
            let text = rest.to_string_lossy().replace('\\', "/");
            (!crate::credentials::is_login_file(&text)).then(|| rest.to_path_buf())
        };
        let unpacked =
            crate::custody::unpack_within(&archive, &dest, &in_session, self.unpack_budget());
        copy_out::remove(&archive);
        let unpacked = unpacked.map_err(|e| {
            // A partial copy is not kept (review F2-RA-1).
            let _ = std::fs::remove_dir_all(&dest);
            e.to_string()
        })?;
        if unpacked.over_budget > 0 {
            self.rt.events.log(
                self.name(),
                format!(
                    "{} files of the agent's session were not saved: past the byte budget, each name at its file's full size",
                    unpacked.over_budget
                ),
            );
        }
        left_out.extend(unpacked.left_out.iter().map(|p| {
            Path::new(p)
                .strip_prefix(&home_rel)
                .map(|r| r.display().to_string())
                .unwrap_or_else(|_| p.clone())
        }));
        if unpacked.files == 0 {
            let _ = std::fs::remove_dir_all(&dest);
            return Ok(None);
        }
        Ok(Some(AgentSession {
            path: SESSIONS_DIR.into(),
            dirs: dirs.to_vec(),
            files: unpacked.files as u64,
            bytes: unpacked.bytes,
            session_id: self.agent.session_id.clone(),
            left_out,
        }))
    }

    /// Note which image graded the trial, when it was the verifier's own.
    pub(super) fn note_verifier_image(&mut self) {
        if let Some(v) = &self.spec.task.extras.verifier_image {
            let source = v.source.clone();
            defense(
                &mut self.defenses,
                "verifier_image",
                DefenseState::Enforced,
                Some(format!("graded in an image of its own, from {source}")),
            );
        }
    }

    /// Run the task's hooks, after every agent process is gone and before
    /// grading: as root unless a hook names its user, in the agent's
    /// container or in a service. In the agent's container a hook works in
    /// the agent's files as the verifier does, so it runs in the verifier's
    /// environment: the trusted `PATH`, loader variables unset, and
    /// `PYTHONSAFEPATH` as the verifier gets it.
    pub(super) async fn run_hooks(&mut self) -> Result<(), Problem> {
        let task = self.spec.task.clone();
        if task.extras.hooks.is_empty() {
            return Ok(());
        }
        let dir = self.spec.dir.join("hooks");
        let _ = std::fs::create_dir_all(&dir);
        let mut env = std::collections::BTreeMap::new();
        if task.extras.hooks.iter().any(|h| h.service.is_none()) {
            let out = self
                .root(hardening::trusted_path(&self.workdir), SETUP_TIMEOUT)
                .await
                .map_err(|e| sandbox_problem(Phase::Handoff, e))?;
            let Some(path) = Report::parse(&out.stdout).first("path").map(str::to_string) else {
                return Err(super::unreported("the trusted PATH check for hooks", &out));
            };
            env.insert("PATH".to_string(), path);
            env.insert("PYTHONDONTWRITEBYTECODE".to_string(), "1".to_string());
            crate::confine::add_safepath(&mut env, &task.verifier.env, task.format);
        }
        let unset = hardening::unset_vars(&task.verifier.env);
        for (i, hook) in task.extras.hooks.iter().enumerate() {
            let n = i + 1;
            let user = hook.user.clone().unwrap_or_else(|| "root".into());
            let mut exec = Exec::new(hook.run.clone())
                .user(user.clone())
                .timeout(hook.timeout);
            let target = match &hook.service {
                None => {
                    exec.command =
                        format!("{}{}", hardening::verifier_env_prefix(&unset), hook.run);
                    exec = exec
                        .cwd(self.workdir.clone())
                        .envs(env.clone())
                        .unset(unset.iter().copied());
                    self.sb()
                }
                Some(name) => match self.x.services.iter().find(|(s, _)| s == name) {
                    Some((_, sb)) => sb.as_ref(),
                    None => {
                        return Err(hook_problem(
                            n,
                            format!("runs in service {name}, which the task does not declare"),
                            String::new(),
                        ));
                    }
                },
            };
            let out = target
                .exec(exec)
                .await
                .map_err(|e| sandbox_problem(Phase::Handoff, e))?;
            let place = hook.service.as_deref().unwrap_or("the agent's container");
            let _ = std::fs::write(
                dir.join(format!("{n}.txt")),
                format!(
                    "$ {}\n# in {place}, as {user}; exit code {}{}\n{}\n",
                    hook.run,
                    out.exit_code,
                    if out.timed_out { ", timed out" } else { "" },
                    out.combined()
                ),
            );
            self.rt.events.log(
                self.name(),
                format!("hook {n} in {place}: exit code {}", out.exit_code),
            );
            if out.timed_out {
                return Err(hook_problem(
                    n,
                    format!("ran past its {}s timeout", hook.timeout.as_secs()),
                    out.tail(1500),
                ));
            }
            if !out.success() {
                return Err(hook_problem(
                    n,
                    format!("failed with exit code {}", out.exit_code),
                    out.tail(1500),
                ));
            }
        }
        Ok(())
    }

    /// Save `outputs` under the custody caps into `dest/outputs/` (at their
    /// own paths) and `dest/artifacts/` with a manifest.
    pub(super) async fn collect_into(
        &mut self,
        outputs: &[Output],
        dest: &Path,
    ) -> Result<(), Problem> {
        self.hold_watch();
        let r = self.collect_into_held(outputs, dest).await;
        self.release_watch(hardening::OUTPUTS_TAR).await;
        r
    }

    async fn collect_into_held(&mut self, outputs: &[Output], dest: &Path) -> Result<(), Problem> {
        let paths: Vec<String> = outputs.iter().map(|o| o.path.clone()).collect();
        let out = self
            .root(hardening::pack_outputs(&paths), SETUP_TIMEOUT)
            .await
            .map_err(|e| sandbox_problem(Phase::Save, e))?;
        let r = Report::parse(&out.stdout);
        self.custody
            .skipped_special
            .extend(r.all("skipped-special").into_iter().map(str::to_string));
        self.custody
            .skipped_large
            .extend(r.all("skipped-large").into_iter().map(str::to_string));
        if !out.success() {
            return Err(Problem::new(
                "R103",
                Owner::Sandbox,
                Phase::Save,
                "the stage's outputs could not be packed",
                "retry; if it repeats, report it with the trial folder",
            )
            .detail(out.tail(1500)));
        }
        let _ = std::fs::create_dir_all(dest);
        let archive = dest.join("outputs.tar");
        copy_out::copy_out(
            self.sb(),
            hardening::OUTPUTS_TAR,
            &archive,
            copy_out::ARCHIVE_CAP,
            copy_out::COPY_OUT_TIMEOUT,
            copy_out::Shape::File,
        )
        .await
        .map_err(|e| super::outputs_copy_problem(e, "the stage's outputs"))?;
        let unpacked = dest.join("outputs");
        let all = |rel: &Path| Some(rel.to_path_buf());
        match crate::custody::unpack_within(&archive, &unpacked, &all, self.unpack_budget()) {
            Ok(u) if u.over_budget > 0 => self.rt.events.log(
                self.name(),
                format!(
                    "{} files of the stage's outputs were not saved: past the byte budget, each name at its file's full size",
                    u.over_budget
                ),
            ),
            Ok(_) => {}
            Err(e) => {
                // A partial copy is not kept (review F2-RA-1).
                let _ = std::fs::remove_dir_all(&unpacked);
                self.rt
                    .events
                    .log(self.name(), format!("outputs were not unpacked: {e}"));
            }
        }
        copy_out::remove(&archive);
        write_artifacts(&unpacked, outputs, &dest.join("artifacts"));
        Ok(())
    }
}

/// Copy each declared output from `unpacked` (the outputs at their own
/// paths) to `artifacts/<save_as>`, leaving out what its `exclude` patterns
/// match, and write `artifacts/manifest.json`.
pub(super) fn write_artifacts(unpacked: &Path, outputs: &[Output], artifacts: &Path) {
    if outputs.is_empty() {
        return;
    }
    let _ = std::fs::create_dir_all(artifacts);
    let mut manifest: Vec<Value> = Vec::new();
    for o in outputs {
        let from = unpacked.join(o.path.trim_start_matches('/'));
        let to = artifacts.join(&o.save_as);
        let exclude = &o.exclude;
        let (kind, status) = if (from.is_dir() || from.is_file()) && source_excluded(&from, exclude)
        {
            let kind = if from.is_dir() { "directory" } else { "file" };
            (kind, "excluded")
        } else if from.is_dir() {
            let copied = copy_tree(&from, &to, exclude);
            ("directory", if copied == 0 { "empty" } else { "ok" })
        } else if from.is_file() {
            if let Some(parent) = to.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match std::fs::copy(&from, &to) {
                Ok(_) => ("file", "ok"),
                Err(_) => ("file", "failed"),
            }
        } else {
            // Missing, or not saved (a link, a special file, over the cap).
            ("file", "failed")
        };
        manifest.push(json!({
            "source": o.path,
            "destination": format!("artifacts/{}", o.save_as),
            "type": kind,
            "status": status,
            "service": null,
            "exclude": o.exclude,
        }));
    }
    let _ = std::fs::write(
        artifacts.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap_or_default(),
    );
}

/// Whether an output's own `exclude` patterns match the output itself, as
/// 0.8's capture checks its source before walking it: a folder by its name
/// or as `.` (its path relative to itself), a file by its name. Such an
/// output is not copied at all (`{path = "/app/run.log", exclude =
/// ["*.log"]}` captures nothing in 0.8).
fn source_excluded(from: &Path, exclude: &[String]) -> bool {
    use benchflow_task::fnmatch::fnmatchcase;
    let name = from
        .file_name()
        .map(|f| f.to_string_lossy())
        .unwrap_or_default();
    let relative = if from.is_dir() { "." } else { &*name };
    exclude
        .iter()
        .any(|p| fnmatchcase(relative, p) || fnmatchcase(&name, p))
}

/// Copy regular files under `from` to `to`, skipping what an `exclude`
/// pattern matches (by `/`-separated relative path or by name, with 0.8's
/// `fnmatch.fnmatchcase`, so no pattern is invalid); an excluded folder is
/// skipped whole, as 0.8's capture does. Returns the files copied.
fn copy_tree(from: &Path, to: &Path, exclude: &[String]) -> usize {
    use benchflow_task::fnmatch::fnmatchcase;
    let excluded = |rel: &Path| {
        let path = rel
            .to_string_lossy()
            .replace(std::path::MAIN_SEPARATOR, "/");
        let name = rel
            .file_name()
            .map(|f| f.to_string_lossy())
            .unwrap_or_default();
        exclude
            .iter()
            .any(|p| fnmatchcase(&path, p) || fnmatchcase(&name, p))
    };
    let mut n = 0;
    for entry in walkdir::WalkDir::new(from)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e| {
            e.depth() == 0 || e.path().strip_prefix(from).is_ok_and(|rel| !excluded(rel))
        })
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let Ok(rel) = entry.path().strip_prefix(from) else {
            continue;
        };
        let dest: PathBuf = to.join(rel);
        if let Some(parent) = dest.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if std::fs::copy(entry.path(), &dest).is_ok() {
            n += 1;
        }
    }
    let _ = std::fs::create_dir_all(to);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards the D5 fix: `exclude` was matched against files only, so
    /// `exclude = ["node_modules"]` still copied node_modules/x/index.js;
    /// 0.8 drops an excluded folder with everything under it.
    #[test]
    fn an_excluded_folder_is_skipped_whole() {
        let d = tempfile::tempdir().unwrap();
        let from = d.path().join("from");
        for f in [
            "report.txt",
            "node_modules/x/index.js",
            "cache/a.bin",
            "src/__pycache__/m.pyc",
            "src/main.py",
            "data[1/x.csv",
        ] {
            let p = from.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, "x").unwrap();
        }
        let to = d.path().join("to");
        let exclude: Vec<String> = ["node_modules", "cache", "__pycache__", "data[1"]
            .map(String::from)
            .to_vec();
        assert_eq!(copy_tree(&from, &to, &exclude), 2);
        assert!(to.join("report.txt").is_file() && to.join("src/main.py").is_file());
        assert!(!to.join("node_modules").exists() && !to.join("cache").exists());
        assert!(!to.join("src/__pycache__").exists());
        // An unclosed `[` is a literal, as in fnmatch; globset refused the
        // pattern and the folder was copied.
        assert!(!to.join("data[1").exists());
    }

    #[test]
    fn the_manifest_lists_each_output_in_harbors_shape() {
        let d = tempfile::tempdir().unwrap();
        let unpacked = d.path().join("outputs");
        std::fs::create_dir_all(unpacked.join("app/logs")).unwrap();
        std::fs::write(unpacked.join("app/report.txt"), "r").unwrap();
        std::fs::write(unpacked.join("app/logs/a.log"), "a").unwrap();
        std::fs::write(unpacked.join("app/logs/b.tmp"), "b").unwrap();
        let out = |path: &str, save_as: &str, exclude: &[&str]| Output {
            path: path.into(),
            save_as: save_as.into(),
            max_bytes: None,
            exclude: exclude.iter().map(|s| s.to_string()).collect(),
        };
        let artifacts = d.path().join("artifacts");
        write_artifacts(
            &unpacked,
            &[
                out("/app/report.txt", "report.txt", &[]),
                out("/app/logs", "logs", &["*.tmp"]),
                out("/app/missing", "missing", &[]),
            ],
            &artifacts,
        );
        assert!(artifacts.join("report.txt").is_file());
        assert!(artifacts.join("logs/a.log").is_file());
        assert!(!artifacts.join("logs/b.tmp").exists(), "excluded");
        let m: Vec<Value> = serde_json::from_str(
            &std::fs::read_to_string(artifacts.join("manifest.json")).unwrap(),
        )
        .unwrap();
        let rows: Vec<(&str, &str, &str)> = m
            .iter()
            .map(|e| {
                (
                    e["destination"].as_str().unwrap(),
                    e["type"].as_str().unwrap(),
                    e["status"].as_str().unwrap(),
                )
            })
            .collect();
        assert_eq!(
            rows,
            [
                ("artifacts/report.txt", "file", "ok"),
                ("artifacts/logs", "directory", "ok"),
                ("artifacts/missing", "file", "failed")
            ]
        );
    }

    /// 0.8 matches `exclude` against the output itself too: a file output
    /// whose name matches, or a folder whose name (or `.`) does, captures
    /// nothing.
    #[test]
    fn an_output_its_own_exclude_matches_is_not_copied() {
        let d = tempfile::tempdir().unwrap();
        let unpacked = d.path().join("outputs");
        for f in [
            "app/run.log",
            "app/node_modules/x.js",
            "app/logs/a.txt",
            "app/keep/b.txt",
        ] {
            let p = unpacked.join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, "x").unwrap();
        }
        let out = |path: &str, save_as: &str, exclude: &[&str]| Output {
            path: path.into(),
            save_as: save_as.into(),
            max_bytes: None,
            exclude: exclude.iter().map(|s| s.to_string()).collect(),
        };
        let artifacts = d.path().join("artifacts");
        write_artifacts(
            &unpacked,
            &[
                out("/app/run.log", "run.log", &["*.log"]),
                out("/app/node_modules/", "nm", &["node_modules"]),
                out("/app/logs", "logs", &["*"]),
                out("/app/keep", "keep", &["*.log"]),
            ],
            &artifacts,
        );
        assert!(!artifacts.join("run.log").exists());
        assert!(!artifacts.join("nm").exists());
        assert!(!artifacts.join("logs/a.txt").exists());
        assert!(artifacts.join("keep/b.txt").is_file());
        let m: Vec<Value> = serde_json::from_str(
            &std::fs::read_to_string(artifacts.join("manifest.json")).unwrap(),
        )
        .unwrap();
        let statuses: Vec<(&str, &str)> = m
            .iter()
            .map(|e| (e["type"].as_str().unwrap(), e["status"].as_str().unwrap()))
            .collect();
        assert_eq!(
            statuses,
            [
                ("file", "excluded"),
                ("directory", "excluded"),
                ("directory", "excluded"),
                ("directory", "ok")
            ]
        );
    }
}
