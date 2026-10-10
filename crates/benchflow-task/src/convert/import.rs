//! A split-layout (Harbor) task folder to a task.md package, ported from
//! task.md's `tools/convert.py import` (`import_harbor`,
//! `harbor_to_taskmd`).
//!
//! | Harbor | task.md |
//! |---|---|
//! | `instruction.md` | The top of `task.md` |
//! | `task.toml` | The ` ```toml task ` block, in task.md's names |
//! | `environment/` | `sandbox/` (a `docker-compose.yaml` becomes `[sandbox] compose`) |
//! | `tests/`, `solution/` | `verifier/`, `oracle/`, mounted at `/tests` and `/solution` |
//! | `steps/<name>/` | The first step's instruction on top, later steps as stage blocks, files under `stages/<name>/` |
//! | `cheat/solve.sh` | Stays, declared as `[integrity.controls] cheats` |
//! | `trajectory.json`, `steps/<name>/workdir/` | Stay, recorded under `[import.harbor]` |
//!
//! Where export would write `task.toml` or an instruction differently from
//! Harbor's bytes, the original bytes are kept in `import/harbor/` with
//! their hashes under `[import.harbor] originals`, so exporting the
//! unchanged task gives back Harbor's content digest.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

use super::doc;
use super::export;
use super::files::{self, Rel};
use super::harbor::{self, HARBOR_COMPOSE, HARBOR_MOUNTS, ordered, provider_vars, verifier_envs};
use super::rewardkit;
use super::tomltext;
use super::value::{Table, Val, parse_toml};
use super::{ConvertError, ImportPlan, io_err};

/// Where a package keeps Harbor files import rewrites, byte for byte.
pub(crate) const ORIGINALS: &str = "import/harbor";

static SKILLS_COPY: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:COPY|ADD)\s+(?:--\S+\s+)*(?:\./?|(?:\./)?skills(?:/\S*)?)\s").unwrap()
});

/// Harbor folders and their task.md names.
const HARBOR_DIRS: [(&str, &str); 3] = [
    ("environment", "sandbox"),
    ("tests", "verifier"),
    ("solution", "oracle"),
];

fn harbor_dir(name: &str) -> Option<&'static str> {
    HARBOR_DIRS
        .iter()
        .find(|(h, _)| *h == name)
        .map(|(_, n)| *n)
}

/// Where a file of a Harbor task goes in the task.md package (`None`:
/// folded into task.md).
pub(crate) fn harbor_path(rel: &[String], steps: &BTreeSet<String>) -> Option<Rel> {
    let parts: Vec<&str> = rel.iter().map(String::as_str).collect();
    if parts == ["instruction.md"] || parts == ["task.toml"] {
        return None;
    }
    if let Some(n) = harbor_dir(parts[0]) {
        let mut out = vec![n.to_string()];
        out.extend(rel[1..].iter().cloned());
        return Some(out);
    }
    if parts[0] == "steps" && parts.len() > 2 && steps.contains(parts[1]) {
        if parts[2..] == ["instruction.md"] {
            return None;
        }
        let mut out = vec![
            "stages".to_string(),
            rel[1].clone(),
            harbor_dir(parts[2]).unwrap_or(parts[2]).to_string(),
        ];
        out.extend(rel[3..].iter().cloned());
        return Some(out);
    }
    Some(rel.to_vec())
}

/// Which of Harbor's folders and files beyond task.toml a task has.
fn harbor_folders(src: &Path, steps: &BTreeSet<String>) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for folder in ["tests", "solution"] {
        if src.join(folder).is_dir()
            || steps
                .iter()
                .any(|n| src.join("steps").join(n).join(folder).is_dir())
        {
            out.insert(folder.to_string());
        }
    }
    if src.join("environment/docker-compose.yaml").is_file() {
        out.insert("environment/docker-compose.yaml".into());
    }
    if src.join("trajectory.json").is_file() {
        out.insert("trajectory.json".into());
    }
    for n in steps {
        if src.join("steps").join(n).join("trajectory.json").is_file() {
            out.insert(format!("steps/{n}/trajectory.json"));
        }
        if src.join("steps").join(n).join("workdir").is_dir() {
            out.insert(format!("steps/{n}/workdir"));
        }
    }
    if let Some(s) = ["solve.sh", "solve.bat"]
        .into_iter()
        .find(|s| src.join("cheat").join(s).is_file())
    {
        out.insert(format!("cheat/{s}"));
    }
    out
}

/// What a bundled `environment/skills/` folder needs said about it.
pub(crate) struct Skills {
    /// The task comes from a BenchFlow registry, which installs the folder.
    pub benchflow: bool,
    /// The Dockerfile line that copies the folder into the image.
    pub copy: Option<String>,
}

fn skills_info(src: &Path, benchflow: bool) -> Option<Skills> {
    if !files::has_files(&src.join("environment/skills")) {
        return None;
    }
    let dockerfile = src.join("environment/Dockerfile");
    let copy = std::fs::read(&dockerfile).ok().and_then(|bytes| {
        let text = String::from_utf8_lossy(&bytes);
        text.lines()
            .enumerate()
            .find(|(_, line)| SKILLS_COPY.is_match(line))
            .map(|(n, line)| format!("environment/Dockerfile line {}: {}", n + 1, line.trim()))
    });
    Some(Skills { benchflow, copy })
}

fn table_mut<'t>(t: &'t mut Table, key: &str) -> &'t mut Table {
    let entry = t
        .entry(key.to_string())
        .or_insert_with(|| Val::Table(Table::new()));
    if !matches!(entry, Val::Table(_)) {
        *entry = Val::Table(Table::new());
    }
    entry.as_table_mut().expect("a table")
}

/// Record the Harbor conventions a task's files follow beyond task.toml.
fn harbor_files(native: &mut Table, folders: &BTreeSet<String>, notes: &mut Vec<String>) {
    let stages: Vec<String> = native
        .get("stages")
        .and_then(Val::as_table)
        .map(|s| s.keys().cloned().collect())
        .unwrap_or_default();
    let mut record = Table::new();
    let loaded = match stages.first() {
        Some(first) => format!("steps/{first}/trajectory.json"),
        None => "trajectory.json".into(),
    };
    if folders.contains(&loaded) {
        let at = match stages.first() {
            Some(first) => format!("stages/{first}/trajectory.json"),
            None => "trajectory.json".into(),
        };
        record.insert("trajectory".into(), Val::Str(at));
        notes.push(format!(
            "{loaded} is ATIF context Harbor seeds the agent's session with before it starts; no task.md runtime loads it, so it is recorded under [import.harbor] trajectory, and the checker warns"
        ));
    }
    let workdirs: Table = stages
        .iter()
        .filter(|n| folders.contains(&format!("steps/{n}/workdir")))
        .map(|n| (n.clone(), Val::Str(format!("stages/{n}/workdir"))))
        .collect();
    if !workdirs.is_empty() {
        let mut shown: Vec<String> = workdirs
            .values()
            .filter_map(Val::as_str)
            .map(str::to_string)
            .collect();
        shown.sort();
        notes.push(format!(
            "Harbor copies steps/<name>/workdir/ into the agent's working directory before the step starts and runs its setup.sh; no task.md runtime does, so {} are recorded under [import.harbor] workdirs, and the checker warns",
            shown.join(", ")
        ));
        record.insert("workdirs".into(), Val::Table(workdirs));
    }
    if !record.is_empty() {
        let kept = table_mut(table_mut(native, "import"), "harbor");
        let mut clash: Vec<String> = record
            .keys()
            .filter(|k| kept.contains_key(*k))
            .cloned()
            .collect();
        clash.sort();
        for key in clash {
            notes.push(format!(
                "task.toml has a top-level {key} key, so the Harbor feature it would record stays unrecorded"
            ));
            record.shift_remove(&key);
        }
        kept.extend(record);
    }
    let cheats: Vec<&String> = folders.iter().filter(|f| f.starts_with("cheat/")).collect();
    if !cheats.is_empty() {
        let list: Vec<Val> = cheats.iter().map(|c| Val::Str((*c).clone())).collect();
        let mut controls = Table::new();
        controls.insert("cheats".into(), Val::List(list));
        let mut integrity = Table::new();
        integrity.insert("controls".into(), Val::Table(controls));
        native.insert("integrity".into(), Val::Table(integrity));
        let shown: Vec<String> = cheats.iter().map(|c| format!("'{c}'")).collect();
        notes.push(format!(
            "cheat/ holds a cheat oracle (Terminal-Bench 3's convention): declared as [integrity.controls] cheats = [{}]; its files stay in place",
            shown.join(", ")
        ));
    }
}

/// A bundled `environment/skills/` folder, which lands in `sandbox/skills/`.
fn bundled_skills(native: &mut Table, skills: &Skills, notes: &mut Vec<String>) {
    if native
        .get("sandbox")
        .and_then(Val::as_table)
        .is_some_and(|s| s.contains_key("skills"))
    {
        return;
    }
    if skills.benchflow {
        table_mut(native, "sandbox").insert("skills".into(), Val::Str("sandbox/skills".into()));
        notes.push(
            "environment/skills/ holds the task's bundled skills, which BenchFlow's runtime installs by convention (the task comes from a BenchFlow registry); declared as [sandbox] skills = \"sandbox/skills\", so a task.md run installs them".into(),
        );
        return;
    }
    let copy = skills
        .copy
        .as_ref()
        .map(|c| {
            format!(
                " The image copies the folder ({c}), so only an agent that looks there finds them."
            )
        })
        .unwrap_or_default();
    notes.push(format!(
        "warning: environment/skills/ holds skills that nothing installs: Harbor installs skills only from the skills_dir that task.toml names, and it names none.{copy} If the task means them to be installed, as BenchFlow's runtime does by convention, declare [sandbox] skills = \"sandbox/skills\" (import-dataset does so for a BenchFlow registry)"
    ));
}

/// `[verifier] models = true` when the verifier's scripts call a model.
pub(crate) fn declare_models(native: &mut Table, rk_judges: &[String], notes: &mut Vec<String>) {
    let mut why: Vec<String> = Vec::new();
    if !rk_judges.is_empty() {
        why.push(format!(
            "RewardKit judges that stay scripts ({})",
            rk_judges.join(", ")
        ));
    }
    for (where_, env) in verifier_envs(native) {
        let keys = provider_vars(&env);
        if !keys.is_empty() {
            why.push(format!("[{where_}] passes {}", keys.join(", ")));
        }
    }
    let has_models = native
        .get("verifier")
        .and_then(Val::as_table)
        .is_some_and(|v| v.contains_key("models"));
    if why.is_empty() || has_models {
        return;
    }
    table_mut(native, "verifier").insert("models".into(), Val::Bool(true));
    notes.push(format!(
        "the verifier's scripts call a model: {}. Declared as [verifier] models = true, so a task.md runtime gives them its model proxy and a scoped token; on Harbor they read the keys [verifier.env] passes from the host",
        why.join("; ")
    ));
}

/// The inputs of [`harbor_to_taskmd`] besides the text.
pub(crate) struct Sources<'a> {
    pub instruction: Option<&'a str>,
    pub task_toml: &'a str,
    pub steps: &'a BTreeMap<String, String>,
    pub folders: &'a BTreeSet<String>,
    pub originals: Option<&'a Table>,
    pub skills: Option<&'a Skills>,
    pub rewardkit: Option<&'a rewardkit::Scan>,
}

/// A Harbor task's text to task.md text, with notes.
pub(crate) fn harbor_to_taskmd(s: &Sources<'_>) -> Result<(String, Vec<String>), ConvertError> {
    let invalid = |message: String| ConvertError::Invalid {
        file: "task.toml".into(),
        message,
    };
    let cfg = parse_toml(s.task_toml).map_err(|e| invalid(format!("not valid TOML: {e}")))?;
    let imported = harbor::config_from_harbor(&cfg, "harbor").map_err(invalid)?;
    let (mut native, moves, mut notes) = (imported.config, imported.moves, imported.notes);
    for (folder, part) in [("tests", "verifier"), ("solution", "oracle")] {
        if s.folders.contains(folder) {
            let mount = HARBOR_MOUNTS
                .iter()
                .find(|(p, _)| *p == part)
                .map(|(_, m)| *m)
                .unwrap_or_default();
            table_mut(&mut native, part).insert("mount".into(), Val::Str(mount.into()));
        }
    }
    if s.folders.contains("environment/docker-compose.yaml") {
        table_mut(&mut native, "sandbox").insert("compose".into(), Val::Str(HARBOR_COMPOSE.into()));
    }
    harbor_files(&mut native, s.folders, &mut notes);
    if let Some(skills) = s.skills {
        bundled_skills(&mut native, skills, &mut notes);
    }
    let mut rk_judges: Vec<String> = Vec::new();
    if let Some(scan) = s.rewardkit {
        if scan.found {
            let kept = table_mut(table_mut(&mut native, "import"), "harbor");
            if kept.contains_key("rewardkit") {
                notes.push("task.toml has a top-level rewardkit key, so the RewardKit record stays unwritten".into());
            } else {
                kept.insert("rewardkit".into(), Val::Table(scan.record()));
            }
            rk_judges = scan.judges.clone();
        }
        notes.extend(scan.notes.iter().cloned());
    }
    declare_models(&mut native, &rk_judges, &mut notes);
    if let Some(originals) = s.originals {
        let kept = table_mut(table_mut(&mut native, "import"), "harbor");
        if kept.contains_key("originals") {
            notes.push("task.toml has a top-level originals key, so the record of Harbor's original files stays unwritten".into());
        } else {
            kept.insert("originals".into(), Val::Table(originals.clone()));
        }
    }
    let mut native = ordered(
        &native,
        &[
            "name",
            "title",
            "version",
            "description",
            "authors",
            "keywords",
            "about",
            "agent",
            "sandbox",
            "verifier",
            "oracle",
            "stages",
            "integrity",
            "provenance",
            "import",
        ],
    );
    let stage_names: Vec<String> = native
        .get("stages")
        .and_then(Val::as_table)
        .map(|t| t.keys().cloned().collect())
        .unwrap_or_default();
    let mut blocks: Vec<String> = Vec::new();
    let instruction: String = if let Some(first) = stage_names.first() {
        for name in &stage_names[1..] {
            blocks.push(doc::block(
                &format!("stage {name}"),
                s.steps.get(name).map_or("", String::as_str),
            ));
        }
        if let Some(md) = s.instruction.filter(|t| !t.trim().is_empty()) {
            table_mut(table_mut(&mut native, "import"), "harbor")
                .insert("instruction".into(), Val::Str(md.to_string()));
            notes.push("instruction.md is not shown to agents in a multi-step Harbor task; kept under [import.harbor]".into());
        }
        s.steps.get(first).cloned().unwrap_or_default()
    } else {
        s.instruction.unwrap_or_default().to_string()
    };
    let comments = tomltext::scan_comments(s.task_toml).map_err(invalid)?;
    let comments = harbor::remap_comments(&comments, &moves, &native);
    let toml = tomltext::dumps(&native, &comments)
        .ok_or_else(|| invalid("a value TOML cannot hold".into()))?;
    blocks.push(doc::block("toml task", &toml));
    let text = format!(
        "{}\n\n{}\n",
        instruction.trim_end_matches('\n'),
        blocks.join("\n\n")
    );
    let source = match stage_names.first() {
        Some(first) => format!("steps/{first}/instruction.md"),
        None => "instruction.md".into(),
    };
    let text =
        doc::verified(text, &instruction, &source, true).map_err(ConvertError::Instruction)?;
    Ok((text, notes))
}

/// The Harbor files export writes from task.md text: task.toml and the
/// instructions, as bytes.
pub(crate) fn harbor_generated(task_md: &str) -> Result<IndexMap<String, Vec<u8>>, ConvertError> {
    let out = export::taskmd_to_harbor(task_md, None)?;
    let mut files = IndexMap::new();
    files.insert("task.toml".to_string(), out.toml.into_bytes());
    if let Some(i) = out.instruction {
        files.insert("instruction.md".to_string(), i.into_bytes());
    }
    for (name, text) in out.steps {
        files.insert(format!("steps/{name}/instruction.md"), text.into_bytes());
    }
    Ok(files)
}

/// Plan the import of the Harbor task at `src`.
pub(crate) fn plan(src: &Path, benchflow_skills: bool) -> Result<ImportPlan, ConvertError> {
    let toml_path = src.join("task.toml");
    let task_toml = crate::split::read_text(&toml_path).map_err(io_err(&toml_path))?;
    let cfg = parse_toml(&task_toml).map_err(|e| ConvertError::Invalid {
        file: toml_path.clone(),
        message: format!("not valid TOML: {e}"),
    })?;
    let names: BTreeSet<String> = cfg
        .get("steps")
        .and_then(Val::as_list)
        .map(|steps| {
            steps
                .iter()
                .filter_map(|s| s.as_table()?.get("name")?.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let mut steps: BTreeMap<String, String> = BTreeMap::new();
    for n in &names {
        let p = src.join("steps").join(n).join("instruction.md");
        if p.exists() {
            steps.insert(n.clone(), crate::split::read_text(&p).map_err(io_err(&p))?);
        }
    }
    let mut placed: IndexMap<Rel, Rel> = IndexMap::new();
    for rel in files::list(src).map_err(io_err(src))? {
        let Some(to) = harbor_path(&rel, &names) else {
            continue;
        };
        if placed.contains_key(&to) || to == ["task.md"] {
            let other = placed
                .get(&to)
                .map_or("task.md".to_string(), |r| files::posix(r));
            return Err(ConvertError::Collision(format!(
                "{} and {other} would both become {}",
                files::posix(&rel),
                files::posix(&to)
            )));
        }
        placed.insert(to, rel);
    }
    let instr_path = src.join("instruction.md");
    let instruction = if instr_path.exists() {
        Some(crate::split::read_text(&instr_path).map_err(io_err(&instr_path))?)
    } else {
        None
    };
    let rewardkit = rewardkit::scan(src, &cfg, &names);
    let folders = harbor_folders(src, &names);
    let skills = skills_info(src, benchflow_skills);
    let mut sources = Sources {
        instruction: instruction.as_deref(),
        task_toml: &task_toml,
        steps: &steps,
        folders: &folders,
        originals: None,
        skills: skills.as_ref(),
        rewardkit: rewardkit.as_ref(),
    };
    let (mut text, notes) = harbor_to_taskmd(&sources)?;
    let mut notes = notes;
    let generated = harbor_generated(&text)?;
    let mut record = Table::new();
    let mut differ: Vec<String> = Vec::new();
    for (rel, data) in &generated {
        let path = files::under(src, &files::rel(rel));
        if !path.is_file() {
            continue;
        }
        let original = files::sha256_file(&path).map_err(io_err(&path))?;
        let converted = files::sha256_bytes(data);
        if original != converted {
            differ.push(rel.clone());
        }
        let mut r = Table::new();
        r.insert("original".into(), Val::Str(original));
        r.insert("converted".into(), Val::Str(converted));
        record.insert(rel.clone(), Val::Table(r));
    }
    let mut writes: Vec<(String, Vec<u8>)> = Vec::new();
    let taken = differ
        .iter()
        .any(|rel| placed.contains_key(&files::rel(&format!("{ORIGINALS}/{rel}"))));
    if !record.is_empty() && !taken {
        sources.originals = Some(&record);
        let (kept_text, _) = harbor_to_taskmd(&sources)?;
        if harbor_generated(&kept_text)? == generated {
            text = kept_text;
            for rel in &differ {
                let path = files::under(src, &files::rel(rel));
                let data = std::fs::read(&path).map_err(io_err(&path))?;
                writes.push((format!("{ORIGINALS}/{rel}"), data));
            }
            if !differ.is_empty() {
                notes.push(format!(
                    "{} kept byte for byte in {ORIGINALS}/, so exporting the unchanged task gives back Harbor's exact files and content digest",
                    differ.join(", ")
                ));
            }
        }
    }
    Ok(ImportPlan {
        task_md: text,
        copies: placed.into_iter().collect(),
        writes,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: &[&str]) -> BTreeSet<String> {
        n.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn harbor_paths() {
        let steps = names(&["one"]);
        let p = |s: &str| harbor_path(&files::rel(s), &steps).map(|r| files::posix(&r));
        assert_eq!(p("task.toml"), None);
        assert_eq!(p("instruction.md"), None);
        assert_eq!(
            p("environment/Dockerfile").as_deref(),
            Some("sandbox/Dockerfile")
        );
        assert_eq!(p("tests/test.sh").as_deref(), Some("verifier/test.sh"));
        assert_eq!(p("solution/solve.sh").as_deref(), Some("oracle/solve.sh"));
        assert_eq!(p("steps/one/instruction.md"), None);
        assert_eq!(
            p("steps/one/tests/test.sh").as_deref(),
            Some("stages/one/verifier/test.sh")
        );
        assert_eq!(
            p("steps/one/workdir/a").as_deref(),
            Some("stages/one/workdir/a")
        );
        assert_eq!(p("steps/other/x").as_deref(), Some("steps/other/x"));
        assert_eq!(p("README.md").as_deref(), Some("README.md"));
    }

    #[test]
    fn converts_a_small_task() {
        let folders = names(&["tests", "solution"]);
        let steps = BTreeMap::new();
        let s = Sources {
            instruction: Some("Do the thing.\n"),
            task_toml: "version = \"1.0\"\n\n[agent]\ntimeout_sec = 300.0 # five minutes\n\n[verifier]\ntimeout_sec = 60\n\n[verifier.env]\nOPENAI_API_KEY = \"${OPENAI_API_KEY}\"\n",
            steps: &steps,
            folders: &folders,
            originals: None,
            skills: None,
            rewardkit: None,
        };
        let (text, notes) = harbor_to_taskmd(&s).unwrap();
        assert_eq!(
            text,
            "Do the thing.\n\n```toml task\n[agent]\ntimeout = \"5m\"  # five minutes\n\n[verifier]\ntimeout = \"1m\"\nenv = { OPENAI_API_KEY = \"${OPENAI_API_KEY}\" }\nmount = \"/tests\"\nmodels = true\n\n[oracle]\nmount = \"/solution\"\n\n[import.harbor]\nversion = \"1.0\"\n```\n"
        );
        assert_eq!(notes.len(), 1);
    }
}
