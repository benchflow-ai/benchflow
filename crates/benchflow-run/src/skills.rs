//! Skills for the agent: folders that each hold a `SKILL.md` and whatever it
//! bundles, which an agent's CLI discovers in its own skills folder.
//!
//! A trial's skills come from two places:
//!
//! - the task's: a folder in the package (`[sandbox] skills`, or a split
//!   task's `environment/skills/`), copied whole, or a folder the image
//!   fills (Harbor's `skills_dir`, an absolute path), copied from inside the
//!   sandbox; `bench run --no-task-skills` leaves them out (BenchFlow 0.8's
//!   `no-skill` mode);
//! - the run's: each `bench run --skill <folder>`, a skill or a folder of
//!   skills ([`find`]), which replaces a task skill of the same name.
//!
//! Before a model agent starts, every skill is copied into each of the
//! agent's skill folders (its manifest's `skill_paths`, such as
//! `~/.claude/skills`), owned by the agent user. An agent whose manifest names
//! no skill folders gets none, and the trial's config.json says so. The
//! agent's command finds the first folder in `$BENCHFLOW_SKILLS_DIR`, for a
//! CLI that takes it as a flag or a config key.

use std::path::{Path, PathBuf};

use benchflow_sandbox::shell;

/// The file that makes a folder a skill.
pub const SKILL_FILE: &str = "SKILL.md";

/// The variable naming, to the agent's command, the skills folder the skills
/// were copied into (its manifest's first `skill_paths` entry, expanded); set
/// only when the trial placed skills there.
pub const DIR_ENV: &str = "BENCHFLOW_SKILLS_DIR";

/// Where skills are gathered in the sandbox before they are copied to the
/// agent: under BenchFlow's root-only state folder.
pub const STAGING: &str = "/var/lib/benchflow/skills";

/// The most a skills folder may hold, in bytes: as much as a trial's saved
/// outputs.
pub const MAX_SKILLS_BYTES: u64 = crate::hardening::MAX_OUTPUT_TOTAL_BYTES;

/// Whether `folder` holds more than `max` bytes of files (links not
/// followed): the size it holds when it does.
pub fn too_large(folder: &Path, max: u64) -> Option<u64> {
    let mut total = 0u64;
    for entry in walkdir::WalkDir::new(folder).into_iter().flatten() {
        if entry.file_type().is_file() {
            total += entry.metadata().map(|m| m.len()).unwrap_or(0);
            if total > max {
                return Some(total);
            }
        }
    }
    None
}

/// The skill folders `path` names: itself when it holds a `SKILL.md`, else
/// each folder directly in it (all of which must hold one), as Harbor's
/// `--skill` reads them. Hidden folders are skipped.
pub fn find(path: &Path) -> Result<Vec<PathBuf>, String> {
    if !path.is_dir() {
        return Err(if path.exists() {
            format!("{} is not a folder", path.display())
        } else {
            format!("{} does not exist", path.display())
        });
    }
    if path.join(SKILL_FILE).is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| format!("{} cannot be read: {e}", path.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir() && !hidden(p))
        .collect();
    dirs.sort();
    let without: Vec<String> = dirs
        .iter()
        .filter(|d| !d.join(SKILL_FILE).is_file())
        .map(|d| name(d))
        .collect();
    if !without.is_empty() {
        return Err(format!(
            "{} holds folders without a {SKILL_FILE}: {}",
            path.display(),
            without.join(", ")
        ));
    }
    if dirs.is_empty() {
        return Err(format!(
            "{} is neither a skill (a folder with a {SKILL_FILE}) nor a folder of skills",
            path.display()
        ));
    }
    Ok(dirs)
}

/// Every `--skill` folder's skills, the later of two with the same name
/// winning, in name order.
pub fn resolve(paths: &[PathBuf]) -> Result<Vec<PathBuf>, String> {
    let mut by_name = std::collections::BTreeMap::new();
    for path in paths {
        for dir in find(path)? {
            let dir = std::fs::canonicalize(&dir).unwrap_or(dir);
            by_name.insert(name(&dir), dir);
        }
    }
    Ok(by_name.into_values().collect())
}

/// The skills directly in a task's skills folder: its folders that hold a
/// `SKILL.md`. (The folder is copied whole; this is what the record lists.)
pub fn names_in(folder: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(folder)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join(SKILL_FILE).is_file() && !hidden(p))
        .map(|p| name(&p))
        .collect();
    names.sort();
    names
}

/// A folder's last component.
pub fn name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default()
}

fn hidden(path: &Path) -> bool {
    name(path).starts_with('.')
}

/// A manifest skill path (`$HOME/.claude/skills`) as `(base, folder)` for
/// [`place_script`]: the agent's home or working folder, and the path
/// under it. `None` for a path that starts with neither.
pub fn expand(path: &str, home: &str, workdir: &str) -> Option<(String, String)> {
    let (base, rest) = if let Some(rest) = path.strip_prefix("$HOME") {
        (home, rest)
    } else {
        (workdir, path.strip_prefix("$WORKSPACE")?)
    };
    let base = base.trim_end_matches('/');
    Some((base.to_string(), format!("{base}{rest}")))
}

/// The root script that copies [`STAGING`] into each agent skill folder.
///
/// Each target is `(base, folder)`: `base` is the agent's home or working
/// folder, which the agent user already owns, and `folder` lies under it.
/// Folders between them are created and handed to `user` (unless the agent
/// runs as root), and the copy is the user's. `from_sandbox` is a folder
/// the image filled (Harbor's `skills_dir`), copied into the staging folder
/// first; `run_names` are the run's skills, already uploaded to
/// `<STAGING>.run/<name>`, which replace task skills of the same name. The
/// script prints `BF skills-missing <folder>` when the image's folder is
/// absent, `BF skill <folder> <name>` for each skill in place, and removes
/// the staging folders.
pub fn place_script(
    user: &str,
    targets: &[(String, String)],
    from_sandbox: Option<&str>,
    run_names: &[String],
) -> String {
    let q = shell::quote;
    let stage = q(STAGING);
    let run_stage = q(&format!("{STAGING}.run"));
    let mut s = String::from("set -e\n");
    s.push_str(&format!("mkdir -p {stage}\n"));
    if let Some(dir) = from_sandbox {
        let d = q(dir);
        s.push_str(&format!(
            "if [ -d {d} ]; then cp -R {d}/. {stage}/; else echo \"BF skills-missing \"{d}; fi\n"
        ));
    }
    for n in run_names {
        s.push_str(&format!(
            "rm -rf {stage}/{n}; cp -R {run_stage}/{n} {stage}/{n}\n",
            n = q(n)
        ));
    }
    let owner = (user != "root" && user != "0").then(|| q(&format!("{user}:")));
    for (base, folder) in targets {
        let base = base.trim_end_matches('/');
        let rest = folder
            .strip_prefix(base)
            .unwrap_or(folder)
            .trim_start_matches('/');
        let mut path = base.to_string();
        for part in rest.split('/').filter(|p| !p.is_empty()) {
            path = format!("{path}/{part}");
            s.push_str(&format!("mkdir -p {p}\n", p = q(&path)));
            if let Some(o) = &owner {
                s.push_str(&format!("chown -h {o} {p}\n", p = q(&path)));
            }
        }
        let f = q(folder);
        s.push_str(&format!("cp -R {stage}/. {f}/\n"));
        if let Some(o) = &owner {
            s.push_str(&format!("chown -R -h {o} {f}\n"));
        }
        s.push_str(&format!(
            "for d in {f}/*/; do [ -f \"$d\"{SKILL_FILE} ] && echo \"BF skill \"{f}\" $(basename \"$d\")\"; done; true\n"
        ));
    }
    s.push_str(&format!("rm -rf {stage} {run_stage}\n"));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skill(root: &Path, name: &str) -> PathBuf {
        let d = root.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(SKILL_FILE), format!("---\nname: {name}\n---\n")).unwrap();
        d
    }

    #[test]
    fn a_skills_folder_over_the_cap_is_too_large() {
        let tmp = tempfile::tempdir().unwrap();
        let s = skill(tmp.path(), "csv");
        std::fs::write(s.join("data.bin"), vec![0u8; 100]).unwrap();
        assert_eq!(too_large(tmp.path(), 1024), None);
        assert!(too_large(tmp.path(), 50).is_some_and(|n| n > 50));
    }

    #[test]
    fn a_skill_or_a_folder_of_skills_as_harbor_reads_them() {
        let tmp = tempfile::tempdir().unwrap();
        let one = skill(tmp.path(), "one");
        assert_eq!(find(&one).unwrap(), [one]);
        let root = tmp.path().join("pack");
        let (a, b) = (skill(&root, "a"), skill(&root, "b"));
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join("README.md"), "loose files are fine").unwrap();
        assert_eq!(find(&root).unwrap(), [a, b]);
        std::fs::create_dir_all(root.join("licenses")).unwrap();
        assert!(
            find(&root)
                .unwrap_err()
                .contains("without a SKILL.md: licenses")
        );
        assert!(
            find(&tmp.path().join("nope"))
                .unwrap_err()
                .contains("does not exist")
        );
        let empty = tmp.path().join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(find(&empty).unwrap_err().contains("neither a skill"));
    }

    #[test]
    fn the_later_of_two_skills_with_one_name_wins() {
        let tmp = tempfile::tempdir().unwrap();
        let first = skill(&tmp.path().join("x"), "csv");
        let second = skill(&tmp.path().join("y"), "csv");
        let other = skill(&tmp.path().join("y"), "pdf");
        let got = resolve(&[first, tmp.path().join("y")]).unwrap();
        let canon = |p: PathBuf| std::fs::canonicalize(p).unwrap();
        assert_eq!(got, [canon(second), canon(other)]);
    }

    #[test]
    fn skill_paths_expand_against_the_home_or_the_workdir() {
        assert_eq!(
            expand("$HOME/.claude/skills", "/home/agent/", "/app"),
            Some(("/home/agent".into(), "/home/agent/.claude/skills".into()))
        );
        assert_eq!(
            expand("$WORKSPACE/.agents/skills", "/root", "/app"),
            Some(("/app".into(), "/app/.agents/skills".into()))
        );
        assert_eq!(expand("/etc/skills", "/root", "/app"), None);
    }

    #[test]
    fn the_script_hands_each_new_folder_to_the_agent_user() {
        let s = place_script(
            "agent",
            &[(
                "/home/agent".into(),
                "/home/agent/.config/opencode/skills".into(),
            )],
            Some("/skills"),
            &["csv".into()],
        );
        for line in [
            "if [ -d /skills ]; then cp -R /skills/. /var/lib/benchflow/skills/;",
            "rm -rf /var/lib/benchflow/skills/csv; cp -R /var/lib/benchflow/skills.run/csv /var/lib/benchflow/skills/csv",
            "mkdir -p /home/agent/.config\nchown -h agent: /home/agent/.config\n",
            "chown -h agent: /home/agent/.config/opencode/skills\n",
            "cp -R /var/lib/benchflow/skills/. /home/agent/.config/opencode/skills/\n",
            "chown -R -h agent: /home/agent/.config/opencode/skills\n",
            "rm -rf /var/lib/benchflow/skills /var/lib/benchflow/skills.run\n",
        ] {
            assert!(s.contains(line), "{line}\n---\n{s}");
        }
        assert!(
            !s.contains("chown -h agent: /home/agent\n"),
            "the home is the user's already"
        );
        let root = place_script(
            "root",
            &[("/root".into(), "/root/.claude/skills".into())],
            None,
            &[],
        );
        assert!(!root.contains("chown"), "{root}");
    }

    /// The script itself, run by `sh` on this machine with its folders moved
    /// under a temporary one and the current user as the agent user.
    #[cfg(unix)]
    #[test]
    fn the_script_copies_and_reports_in_a_real_shell() {
        let tmp = tempfile::tempdir().unwrap();
        let t = tmp.path().to_str().unwrap();
        let stage = format!("{t}{STAGING}");
        skill(Path::new(&stage), "csv");
        std::fs::write(format!("{stage}/notes.md"), "not a skill").unwrap();
        skill(Path::new(&format!("{stage}.run")), "sql");
        skill(&tmp.path().join("image"), "docx");
        let home = format!("{t}/home/agent");
        std::fs::create_dir_all(&home).unwrap();
        let user = String::from_utf8(
            std::process::Command::new("id")
                .arg("-un")
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap();
        let script = place_script(
            user.trim(),
            &[(home.clone(), format!("{home}/.claude/skills"))],
            Some(&format!("{t}/image")),
            &["sql".into()],
        )
        .replace(STAGING, &stage);
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success(),
            "{script}\n{stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let target = format!("{home}/.claude/skills");
        for n in ["csv", "docx", "sql"] {
            assert!(
                stdout.contains(&format!("BF skill {target} {n}\n")),
                "{stdout}"
            );
            assert!(Path::new(&format!("{target}/{n}/SKILL.md")).is_file());
        }
        assert!(
            Path::new(&format!("{target}/notes.md")).is_file(),
            "loose files come along"
        );
        assert!(!Path::new(&stage).exists() && !Path::new(&format!("{stage}.run")).exists());
        // A missing image folder is reported, not fatal.
        let script = place_script(user.trim(), &[], Some(&format!("{t}/nope")), &[])
            .replace(STAGING, &stage);
        let out = std::process::Command::new("sh")
            .arg("-c")
            .arg(&script)
            .output()
            .unwrap();
        assert!(out.status.success());
        assert!(
            String::from_utf8_lossy(&out.stdout).contains(&format!("BF skills-missing {t}/nope"))
        );
    }
}
