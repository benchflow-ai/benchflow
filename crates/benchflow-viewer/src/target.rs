//! What a `bench view` argument points at, decided as 0.8 decides it.

use std::path::{Path, PathBuf};

use crate::catalog::{discover, discover_jobs, is_rollout_dir, runs_cap};
use crate::py;

/// What the viewer serves.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// One trial folder (it holds `trajectory/acp_trajectory.jsonl`): the
    /// page shows that trajectory.
    Trial { dir: PathBuf },
    /// A job folder or a folder of jobs: the page lists the trials found up
    /// to four folders down and loads each one when it is opened.
    Runs {
        root: PathBuf,
        /// Trials found when the server started (the list is rescanned on
        /// every page load).
        runs: usize,
        /// Whether more trials exist than the list shows
        /// (`BENCHFLOW_VIEWER_MAX_RUNS`, 500 by default).
        capped: bool,
    },
    /// Several job folders served side by side (`bench view <jobA> <jobB>`):
    /// the run list shows the trials of all of them, each id starting with
    /// its job's name, and the page opens on the compare view.
    Jobs {
        /// Each folder with the name its ids start with: the folder's own
        /// name; when two share one, the parent folder's name and a dash
        /// before it (`rust-codex-r1`, `py-codex-r1`), then `-2`, `-3`, ...
        jobs: Vec<(String, PathBuf)>,
        runs: usize,
        capped: bool,
    },
}

impl Target {
    /// The folder this target reads (the first one for [`Target::Jobs`]).
    pub fn path(&self) -> &Path {
        match self {
            Target::Trial { dir } => dir,
            Target::Runs { root, .. } => root,
            Target::Jobs { jobs, .. } => &jobs[0].1,
        }
    }

    /// The page to open first, after the server's address: `/compare` for
    /// several jobs, else the run list (`""`).
    pub fn start_path(&self) -> &'static str {
        match self {
            Target::Jobs { .. } => "/compare",
            _ => "",
        }
    }
}

/// Why a path cannot be viewed.
#[derive(Debug, thiserror::Error)]
pub enum ViewError {
    #[error("{0} does not exist")]
    NotFound(PathBuf),
    #[error(
        "{0} is a file; bench view opens a trial folder, a job folder, or a folder of jobs \
         (session JSONL files are not supported yet)"
    )]
    IsAFile(PathBuf),
    #[error(
        "{0} holds turn*.txt files from a BenchFlow older than ACP trajectories; this viewer reads \
         trajectory/acp_trajectory.jsonl only (view it with BenchFlow 0.8's `bench eval view`)"
    )]
    LegacyTrial(PathBuf),
    #[error(
        "no trajectories found in {0}: expected a trial folder (with trajectory/acp_trajectory.jsonl) \
         or a folder with trials at most four levels down"
    )]
    NoTrajectories(PathBuf),
    #[error("{0} is not a trial folder: it has no trajectory/acp_trajectory.jsonl")]
    NotATrial(PathBuf),
}

/// Decide what `root` is: a trial folder, or a folder with trials below it
/// (a job folder or a folder of jobs).
pub fn detect(root: &Path) -> Result<Target, ViewError> {
    let root = py::normalize(root);
    if root.is_file() {
        return Err(ViewError::IsAFile(root));
    }
    if !root.is_dir() {
        return Err(ViewError::NotFound(root));
    }
    if is_rollout_dir(&root) {
        return Ok(Target::Trial { dir: root });
    }
    if has_turn_files(&root) {
        return Err(ViewError::LegacyTrial(root));
    }
    let cap = runs_cap();
    let found = discover(&root, cap.saturating_add(1)).len();
    if found == 0 {
        return Err(ViewError::NoTrajectories(root));
    }
    Ok(Target::Runs {
        root,
        runs: found.min(cap),
        capped: found > cap,
    })
}

/// Serve several folders at once: one path is [`detect`]ed as usual; two
/// or more must each be a folder with trials in it (a job, a folder of
/// jobs, or a trial) and become [`Target::Jobs`].
pub fn detect_jobs(paths: &[PathBuf]) -> Result<Target, ViewError> {
    match paths {
        [] => Err(ViewError::NoTrajectories(PathBuf::from("."))),
        [one] => detect(one),
        many => {
            let mut folders: Vec<(String, PathBuf)> = Vec::new();
            for path in many {
                let path = py::normalize(path);
                if path.is_file() {
                    return Err(ViewError::IsAFile(path));
                }
                if !path.is_dir() {
                    return Err(ViewError::NotFound(path));
                }
                if discover_jobs(&[(String::new(), path.clone())], 1).is_empty() {
                    return Err(ViewError::NoTrajectories(path));
                }
                let base = match py::name_of(&path) {
                    name if name.is_empty() || name.starts_with('.') => "job".to_string(),
                    name => name,
                };
                folders.push((base, path));
            }
            // Jobs that share a folder name (rust/codex-r1 and py/codex-r1)
            // are told apart by their parent folder's name; `-2`, `-3`, ...
            // only when that is not enough.
            let mut jobs: Vec<(String, PathBuf)> = Vec::new();
            for (base, path) in &folders {
                let shared = folders.iter().filter(|(b, _)| b == base).count() > 1;
                let parent = path
                    .parent()
                    .map(py::name_of)
                    .filter(|p| !p.is_empty() && !p.starts_with('.'));
                let base = match parent {
                    Some(p) if shared => format!("{p}-{base}"),
                    _ => base.clone(),
                };
                let mut name = base.clone();
                let mut suffix = 2;
                while jobs.iter().any(|(taken, _)| *taken == name) {
                    name = format!("{base}-{suffix}");
                    suffix += 1;
                }
                jobs.push((name, path.clone()));
            }
            let cap = runs_cap();
            let found = discover_jobs(&jobs, cap.saturating_add(1)).len();
            Ok(Target::Jobs {
                jobs,
                runs: found.min(cap),
                capped: found > cap,
            })
        }
    }
}

/// A pre-ACP trial: Claude Code stream-json in `turn*.txt` files.
fn has_turn_files(dir: &Path) -> bool {
    std::fs::read_dir(dir).is_ok_and(|entries| {
        entries.filter_map(Result::ok).any(|entry| {
            let name = entry.file_name();
            let name = name.as_encoded_bytes();
            name.starts_with(b"turn") && name.ends_with(b".txt")
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(dir: &Path) {
        std::fs::create_dir_all(dir.join("trajectory")).unwrap();
        std::fs::write(dir.join("trajectory/acp_trajectory.jsonl"), "").unwrap();
    }

    #[test]
    fn a_trial_a_job_and_a_folder_of_jobs() {
        let dir = tempfile::tempdir().unwrap();
        let jobs = dir.path().join("jobs");
        trial(&jobs.join("job-1/task__0001"));
        trial(&jobs.join("job-1/task__0002"));
        trial(&jobs.join("job-2/task__0003"));
        assert_eq!(
            detect(&jobs.join("job-1/task__0001")).unwrap(),
            Target::Trial {
                dir: jobs.join("job-1/task__0001")
            }
        );
        assert_eq!(
            detect(&jobs.join("job-1")).unwrap(),
            Target::Runs {
                root: jobs.join("job-1"),
                runs: 2,
                capped: false
            }
        );
        assert_eq!(
            detect(&jobs).unwrap(),
            Target::Runs {
                root: jobs.clone(),
                runs: 3,
                capped: false
            }
        );
    }

    #[test]
    fn several_jobs_get_unique_names() {
        let dir = tempfile::tempdir().unwrap();
        trial(&dir.path().join("a/smoke/task__0001"));
        trial(&dir.path().join("b/smoke/task__0002"));
        std::fs::create_dir(dir.path().join("empty")).unwrap();
        let paths = [dir.path().join("a/smoke"), dir.path().join("b/smoke")];
        let target = detect_jobs(&paths).unwrap();
        assert_eq!(
            target,
            Target::Jobs {
                jobs: vec![
                    ("a-smoke".into(), paths[0].clone()),
                    ("b-smoke".into(), paths[1].clone())
                ],
                runs: 2,
                capped: false
            }
        );
        assert_eq!(target.start_path(), "/compare");
        assert!(matches!(
            detect_jobs(&[paths[0].clone(), dir.path().join("empty")]),
            Err(ViewError::NoTrajectories(_))
        ));
        assert_eq!(
            detect_jobs(&paths[..1]).unwrap(),
            Target::Runs {
                root: paths[0].clone(),
                runs: 1,
                capped: false
            }
        );
    }

    #[test]
    fn unusable_paths_say_why() {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(
            detect(&dir.path().join("missing")),
            Err(ViewError::NotFound(_))
        ));
        std::fs::write(dir.path().join("session.jsonl"), "{}").unwrap();
        assert!(matches!(
            detect(&dir.path().join("session.jsonl")),
            Err(ViewError::IsAFile(_))
        ));
        std::fs::create_dir(dir.path().join("empty")).unwrap();
        assert!(matches!(
            detect(&dir.path().join("empty")),
            Err(ViewError::NoTrajectories(_))
        ));
        std::fs::create_dir(dir.path().join("legacy")).unwrap();
        std::fs::write(dir.path().join("legacy/turn1.txt"), "{}").unwrap();
        assert!(matches!(
            detect(&dir.path().join("legacy")),
            Err(ViewError::LegacyTrial(_))
        ));
    }
}
