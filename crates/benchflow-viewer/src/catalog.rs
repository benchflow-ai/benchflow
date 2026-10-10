//! Finding trials under a folder and summarizing them for the run list
//! (0.8's `viewer/catalog.py`).

use std::path::{Path, PathBuf};

use crate::lineage::load_lineage;
use crate::metadata::{RunStatus, load_rollout_metadata};
use crate::payload::TRAJECTORY_FILE;
use crate::py;
use crate::pyjson::{Value, object};

/// How many folders below the root trials are looked for. HF trajectory
/// datasets nest one level deeper than local job folders:
/// `<root>/jobs/<run>/<timestamp>/<trial>`.
const MAX_DEPTH: usize = 4;

/// The run list shows at most this many runs unless
/// `BENCHFLOW_VIEWER_MAX_RUNS` says otherwise.
const DEFAULT_RUNS_CAP: usize = 500;

/// A trial found under the root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Run {
    /// The trial's path below the root, `/`-separated: the run list's id.
    pub id: String,
    pub dir: PathBuf,
}

/// Whether a folder is a trial: it holds an ACP trajectory.
pub fn is_rollout_dir(path: &Path) -> bool {
    path.join(TRAJECTORY_FILE).exists()
}

/// The run cap: `BENCHFLOW_VIEWER_MAX_RUNS` when it is an integer (at least
/// 1), else 500.
pub fn runs_cap() -> usize {
    std::env::var("BENCHFLOW_VIEWER_MAX_RUNS")
        .ok()
        .map_or(Some(DEFAULT_RUNS_CAP as i64), |text| py::parse_int(&text))
        .map_or(DEFAULT_RUNS_CAP, |cap| {
            usize::try_from(cap.max(1)).unwrap_or(usize::MAX)
        })
}

/// Trials under `base`, sorted by path, at most `cap` of them.
///
/// The ids double as the `/api/rollout?id=` whitelist: an id is only ever
/// resolved by exact membership here, so a crafted id (`../` and the like)
/// never reaches the filesystem. Hidden folders are skipped; folder
/// symlinks are followed (serving a folder trusts what it links to).
pub fn discover(base: &Path, cap: usize) -> Vec<Run> {
    let mut found = Vec::new();
    for (name, child) in child_dirs(base) {
        walk(&child, name, 1, cap, &mut found);
    }
    found
}

/// Trials under several named folders (`bench view <jobA> <jobB>`), each
/// searched as [`discover`] searches a root, with ids that start with the
/// folder's name: `<name>/<trial>`. The names keep their given order.
pub fn discover_jobs(jobs: &[(String, PathBuf)], cap: usize) -> Vec<Run> {
    let mut found = Vec::new();
    for (name, dir) in jobs {
        if is_rollout_dir(dir) {
            walk(dir, name.clone(), 0, cap, &mut found);
            continue;
        }
        for (child_name, child) in child_dirs(dir) {
            walk(&child, format!("{name}/{child_name}"), 1, cap, &mut found);
        }
    }
    found
}

fn walk(dir: &Path, id: String, depth: usize, cap: usize, found: &mut Vec<Run>) {
    if found.len() >= cap {
        return;
    }
    if is_rollout_dir(dir) {
        // Trials do not nest.
        found.push(Run {
            id,
            dir: dir.to_path_buf(),
        });
        return;
    }
    if depth >= MAX_DEPTH {
        return;
    }
    for (name, child) in child_dirs(dir) {
        walk(&child, format!("{id}/{name}"), depth + 1, cap, found);
    }
}

/// The non-hidden folders in `dir` (symlinks followed), sorted by name.
fn child_dirs(dir: &Path) -> Vec<(String, PathBuf)> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut children: Vec<(std::ffi::OsString, PathBuf)> = entries
        .filter_map(Result::ok)
        .map(|entry| (entry.file_name(), entry.path()))
        .filter(|(_, path)| path.is_dir())
        .collect();
    children.sort();
    children
        .into_iter()
        .map(|(name, path)| (py::os_text(&name), path))
        .filter(|(name, _)| !name.starts_with('.'))
        .collect()
}

/// The trial a run-list id names, found by a fresh scan; `None` for any id
/// the scan does not list, even when the path it spells exists.
pub fn resolve(base: &Path, id: &str) -> Option<Run> {
    discover(base, runs_cap())
        .into_iter()
        .find(|run| run.id == id)
}

/// [`resolve`] for several named folders ([`discover_jobs`]).
pub fn resolve_jobs(jobs: &[(String, PathBuf)], id: &str) -> Option<Run> {
    discover_jobs(jobs, runs_cap())
        .into_iter()
        .find(|run| run.id == id)
}

/// The run-list row for a trial: identity, verdict, and row-level stats.
pub fn rollout_summary(run: &Run) -> Value {
    let metadata = load_rollout_metadata(&run.dir);
    let lineage = load_lineage(&run.dir);
    let name = py::name_of(&run.dir);
    let status = metadata.status.as_ref();
    let text = |pick: fn(&RunStatus) -> Option<String>| status.and_then(pick);
    object([
        ("id", run.id.clone().into()),
        ("name", name.clone().into()),
        (
            "task_name",
            metadata
                .task_name
                .clone()
                .filter(|task| !task.is_empty())
                .unwrap_or(name)
                .into(),
        ),
        ("agent_name", metadata.agent_name.clone().into()),
        ("model", metadata.model.clone().into()),
        ("reward", metadata.reward.into()),
        ("has_error", metadata.has_error.into()),
        ("skill_mode", metadata.skill_mode.clone().into()),
        (
            "duration_sec",
            metadata
                .timing
                .as_ref()
                .and_then(|timing| timing.total)
                .into(),
        ),
        ("cost_usd", metadata.usage.cost_usd.into()),
        ("total_tokens", metadata.usage.total_tokens.clone().into()),
        ("n_tool_calls", metadata.n_tool_calls.clone().into()),
        (
            "execution",
            text(|status| Some(status.execution.to_string())).into(),
        ),
        (
            "execution_detail",
            text(|status| status.execution_detail.clone()).into(),
        ),
        (
            "assessment",
            text(|status| Some(status.assessment.to_string())).into(),
        ),
        (
            "assessment_detail",
            text(|status| status.assessment_detail.clone()).into(),
        ),
        (
            "status_summary",
            text(|status| Some(status.summary.clone())).into(),
        ),
        (
            "branch_forks",
            lineage.as_ref().map(|lineage| lineage.forks.len()).into(),
        ),
        (
            "branch_children",
            lineage
                .as_ref()
                .map(|lineage| {
                    lineage
                        .forks
                        .iter()
                        .map(|fork| fork.children.len())
                        .sum::<usize>()
                })
                .into(),
        ),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trial(root: &Path, rel: &str) {
        let dir = root.join(rel).join("trajectory");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("acp_trajectory.jsonl"),
            "{\"type\": \"agent_message\", \"text\": \"x\"}",
        )
        .unwrap();
    }

    fn ids(runs: Vec<Run>) -> Vec<String> {
        runs.into_iter().map(|run| run.id).collect()
    }

    #[test]
    fn discovery_finds_nested_trials_and_skips_hidden_folders() {
        let dir = tempfile::tempdir().unwrap();
        trial(dir.path(), "job-a/task-1__aaaa0000");
        trial(dir.path(), "job-b/nested/task-2__bbbb0000");
        trial(dir.path(), ".hidden/task-3__cccc0000");
        trial(dir.path(), "jobs/run-a/2026-04-22__01-27-25/t__ffff0000");
        trial(dir.path(), "too/deep/for/the/scan__0000");
        std::fs::create_dir(dir.path().join("not-a-rollout")).unwrap();
        assert_eq!(
            ids(discover(dir.path(), 100)),
            [
                "job-a/task-1__aaaa0000",
                "job-b/nested/task-2__bbbb0000",
                "jobs/run-a/2026-04-22__01-27-25/t__ffff0000"
            ]
        );
    }

    #[test]
    fn the_cap_keeps_a_sorted_prefix() {
        let dir = tempfile::tempdir().unwrap();
        for index in 0..4 {
            trial(dir.path(), &format!("job-{index}/t__{index:04}0000"));
        }
        assert_eq!(
            ids(discover(dir.path(), 2)),
            ["job-0/t__00000000", "job-1/t__00010000"]
        );
    }

    #[test]
    fn named_jobs_prefix_their_ids() {
        let dir = tempfile::tempdir().unwrap();
        trial(dir.path(), "a/job/t-1__aaaa0000");
        trial(dir.path(), "b/job/2026-10-10__00-00-00/t-1__bbbb0000");
        trial(dir.path(), "c/lone__cccc0000");
        let jobs = vec![
            ("job".to_string(), dir.path().join("a/job")),
            ("job-2".to_string(), dir.path().join("b/job")),
            ("lone".to_string(), dir.path().join("c/lone__cccc0000")),
        ];
        assert_eq!(
            ids(discover_jobs(&jobs, 100)),
            [
                "job/t-1__aaaa0000",
                "job-2/2026-10-10__00-00-00/t-1__bbbb0000",
                "lone"
            ]
        );
        assert_eq!(
            resolve_jobs(&jobs, "job-2/2026-10-10__00-00-00/t-1__bbbb0000")
                .unwrap()
                .dir,
            dir.path().join("b/job/2026-10-10__00-00-00/t-1__bbbb0000")
        );
        assert_eq!(resolve_jobs(&jobs, "job/../job-2"), None);
    }

    #[test]
    fn ids_resolve_only_by_membership() {
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("base");
        trial(&base, "inside/run__aaaa0000");
        trial(dir.path(), "outside/secret__bbbb0000");
        assert_eq!(
            resolve(&base, "inside/run__aaaa0000").unwrap().dir,
            base.join("inside/run__aaaa0000")
        );
        assert_eq!(resolve(&base, "../outside/secret__bbbb0000"), None);
        assert_eq!(resolve(&base, "inside/../inside/run__aaaa0000"), None);
        assert_eq!(resolve(&base, ""), None);
    }
}
