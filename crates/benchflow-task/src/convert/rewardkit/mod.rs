//! RewardKit, Harbor's grading library, in a Harbor task: where a `test.sh`
//! runs its `rewardkit` command, the judge TOMLs it grades with, and the
//! release the task installs. Import records them under `[import.harbor]
//! rewardkit` and declares `[verifier] models = true`, so on a task.md
//! runtime the scripts get a model; `test.sh` keeps running RewardKit, so
//! scores on Harbor do not change. Ported from task.md's `tools/convert.py`
//! (`rewardkit_scan`, `rk_calls`, `rk_install`, `rk_record`); converting
//! judges into a rubric (`--convert-judges`) is not ported.

mod judge;
mod shell;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::LazyLock;

use indexmap::IndexMap;
use regex::Regex;

pub(crate) use judge::is_judge_toml;
use shell::{Env, EnvVal};

use super::files;
use super::import::harbor_path;
use super::value::{Table, Val, parse_toml};

/// `[\w.${}/-]`: the characters of a path to the rewardkit command.
fn path_char(c: char) -> bool {
    c.is_alphanumeric() || "_.${}/-".contains(c)
}

static LAUNCH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r#"^(?:"#,
        r#"(?:uvx|uv\s+tool\s+run|pipx\s+run)(?:\s+(?:--from|--with|--python|-p|--with-requirements|--with-editable)(?:=|\s+)"#,
        r#"(?:'[^']*'|"[^"]*"|[^\s'"]+)|\s+-[-\w]+)*"#,
        r#"|uv\s+run(?:\s+(?:--with|--python|-p|--project|--directory|--from|--env-file)(?:=|\s+)(?:'[^']*'|"[^"]*"|[^\s'"]+)"#,
        r#"|\s+-[-\w]+)*(?:\s+python[\d.]*\s+-m)?"#,
        r#"|(?:'[^']*'|"[^"]*"|[^\s'";&|]*python[\d.]*)\s+(?:-\w+\s+)*-m"#,
        r#")\s+$"#
    ))
    .unwrap()
});
static PREFIX: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"^(?:(?:exec|time|command|nohup|env)\s+|[A-Za-z_]\w*=(?:'[^']*'|"[^"]*"|[^\s'"]*)\s+)*$"#,
    )
    .unwrap()
});
static LEAD_OK: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:;|&&|\|\||\||\(|\$\(|`|\{)$").unwrap());
static LEAD_WORD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|\s)(?:then|do|else|!|if|while|until)$").unwrap());
static OPERATOR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[;&|()<>`]").unwrap());
static TEMPLATE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\$\{([^}:]+)(?::-(.*))?\}$").unwrap());
static PIN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"harbor-rewardkit(?:\[[\w,\s-]*\])?(?P<spec>\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[\w.*+!-]+(?:\s*,\s*(?:===|==|~=|!=|<=|>=|<|>)\s*[\w.*+!-]+)*)?",
    )
    .unwrap()
});
static GIT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"harbor-rewardkit\s*@\s*git\+https://github\.com/harbor-framework/harbor(?:\.git)?@(?P<rev>[\w.-]+)").unwrap()
});

/// The boundary RewardKit's words need before them.
fn boundary(before: Option<char>) -> bool {
    match before {
        None => true,
        Some(c) => c.is_whitespace() || ";&|(`'\"".contains(c),
    }
}

/// Where a script runs the rewardkit CLI: the line's index among the
/// logical lines, the launcher before it, the prefix (`env`, `VAR=...`),
/// and its arguments.
#[derive(Debug, Clone, PartialEq)]
struct Call {
    index: usize,
    launcher: String,
    prefix: String,
    args: Vec<String>,
}

/// The leftmost match of `re` that ends `before` and starts at a word
/// boundary, as Python's lookbehind-guarded search finds it.
fn match_to_end(re: &Regex, before: &str) -> Option<usize> {
    let starts: Vec<usize> = before
        .char_indices()
        .map(|(i, _)| i)
        .chain([before.len()])
        .collect();
    starts
        .into_iter()
        .find(|&p| boundary(before[..p].chars().last()) && re.is_match(&before[p..]))
}

/// The calls of `rewardkit` in a script.
fn calls(text: &str) -> (Vec<shell::Line>, Vec<usize>, Vec<Call>) {
    let lines = shell::lines(text);
    let depths = shell::depths(&lines);
    let mut out = Vec::new();
    for (n, ln) in lines.iter().enumerate() {
        let code = shell::code(&ln.text);
        for (start, end) in rewardkit_words(code) {
            let mut before = code[..start].to_string();
            let mut launcher = String::new();
            if let Some(p) = match_to_end(&LAUNCH, &before) {
                launcher = before[p..].to_string();
                before.truncate(p);
            }
            let p = (0..=before.len())
                .filter(|i| before.is_char_boundary(*i))
                .find(|&i| PREFIX.is_match(&before[i..]))
                .unwrap_or(before.len());
            let prefix = before[p..].to_string();
            let lead = before[..p].trim_end();
            if !lead.is_empty() && !(LEAD_OK.is_match(lead) || LEAD_WORD.is_match(lead)) {
                continue;
            }
            let rest = &code[end..];
            let args = match shell::words(rest) {
                Some(words) => split_args(&words),
                None => rest
                    .split_whitespace()
                    .take_while(|tok| !OPERATOR.is_match(tok))
                    .map(str::to_string)
                    .collect(),
            };
            out.push(Call {
                index: n,
                launcher,
                prefix,
                args,
            });
            break;
        }
    }
    (lines, depths, out)
}

/// The spans of `(?:[\w.${}-]*/)*rewardkit` at a word boundary and followed
/// by the end or a separator, left to right, as Python's `finditer` gives
/// them.
fn rewardkit_words(code: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut from = 0;
    while from <= code.len() {
        let mut found = None;
        for p in (from..=code.len()).filter(|i| code.is_char_boundary(*i)) {
            if !boundary(code[..p].chars().last()) {
                continue;
            }
            // The longest path prefix first, as the greedy pattern tries it.
            let mut ends: Vec<usize> = Vec::new();
            for (j, _) in code[p..].match_indices("rewardkit") {
                let j = p + j;
                let path = &code[p..j];
                let chars_ok = path.chars().all(path_char);
                let path_ok = chars_ok && (path.is_empty() || path.ends_with('/'));
                let after = code[j + 9..].chars().next();
                let after_ok = after.is_none_or(|c| c.is_whitespace() || ";&|)`'\"".contains(c));
                if path_ok && after_ok {
                    ends.push(j + 9);
                }
                if !chars_ok {
                    break;
                }
            }
            if let Some(&end) = ends.last() {
                found = Some((p, end));
                break;
            }
        }
        match found {
            Some((s, e)) => {
                out.push((s, e));
                from = if e > s { e } else { e + 1 };
            }
            None => break,
        }
    }
    out
}

fn split_args(words: &[String]) -> Vec<String> {
    let mut args: Vec<String> = Vec::new();
    for w in words {
        if !w.is_empty() && w.chars().all(|c| ";&|()<>".contains(c)) {
            if args
                .last()
                .is_some_and(|a| !a.is_empty() && a.chars().all(|c| c.is_ascii_digit()))
                && w.starts_with(['<', '>'])
            {
                args.pop();
            }
            return args;
        }
        args.push(w.clone());
    }
    args
}

/// The rewardkit CLI's arguments.
#[derive(Debug, Default)]
struct Argv {
    dirs: Vec<String>,
    judge: Option<String>,
    model: Option<String>,
    env: Vec<String>,
}

fn argv(args: &[String]) -> Argv {
    let mut out = Argv::default();
    let flags: &[(&str, &str)] = &[
        ("--workspace", "workspace"),
        ("--output", "output"),
        ("--judge", "judge"),
        ("-j", "judge"),
        ("--model", "model"),
        ("-m", "model"),
        ("--judge-env", "env"),
        ("--je", "env"),
    ];
    let counts = [
        "--max-concurrent-programmatic",
        "--mcprog",
        "--mcp",
        "--max-concurrent-llm",
        "--mcllm",
        "--mcl",
        "--max-concurrent-agent",
        "--mcagent",
        "--mca",
    ];
    let mut i = 0;
    while i < args.len() {
        let (mut a, mut val) = (args[i].as_str(), None::<String>);
        if a.starts_with("--")
            && let Some((k, v)) = a.split_once('=')
        {
            a = k;
            val = Some(v.to_string());
        }
        let flag = flags.iter().find(|(f, _)| *f == a).map(|(_, n)| *n);
        if flag.is_some() || counts.contains(&a) {
            if val.is_none() {
                i += 1;
                val = args.get(i).cloned();
            }
            match flag {
                Some("env") => out.env.push(val.unwrap_or_default()),
                Some("judge") => out.judge = val,
                Some("model") => out.model = val,
                _ => {}
            }
        } else if a.starts_with('-') && a != "-" {
        } else {
            out.dirs.push(a.to_string());
        }
        i += 1;
    }
    out
}

/// The verifier's environment as Harbor starts it: `[verifier.env]`, with
/// a step's own over the task's.
fn harbor_env(cfg: &Table, step: Option<&str>) -> Env {
    let mut env: IndexMap<String, Val> = cfg
        .get("verifier")
        .and_then(Val::as_table)
        .and_then(|v| v.get("env"))
        .and_then(Val::as_table)
        .map(|t| t.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    if let Some(step) = step {
        for st in cfg
            .get("steps")
            .and_then(Val::as_list)
            .map(Vec::as_slice)
            .unwrap_or_default()
        {
            let Some(t) = st.as_table() else { continue };
            if t.get("name").and_then(Val::as_str) != Some(step) {
                continue;
            }
            if let Some(e) = t
                .get("verifier")
                .and_then(Val::as_table)
                .and_then(|v| v.get("env"))
                .and_then(Val::as_table)
            {
                env.extend(e.iter().map(|(k, v)| (k.clone(), v.clone())));
            }
        }
    }
    env.into_iter()
        .map(|(k, v)| {
            let value = match &v {
                Val::Str(s) => match TEMPLATE.captures(s) {
                    Some(c) => EnvVal::Host {
                        var: c[1].to_string(),
                        default: c.get(2).map(|d| d.as_str().to_string()),
                    },
                    None => EnvVal::Str(s.clone()),
                },
                other => EnvVal::Str(super::harbor::py_repr(other).replace('\'', "")),
            };
            (k, value)
        })
        .collect()
}

/// Which RewardKit a task installs, from the texts that decide it in order;
/// `None` when none names it.
fn install_spec(sources: &[String]) -> Option<String> {
    for text in sources {
        if let Some(g) = GIT.captures(text) {
            return Some(format!("harbor@{}", &g["rev"]));
        }
        if let Some(m) = PIN.captures(text) {
            let spec: String = m
                .name("spec")
                .map_or("", |s| s.as_str())
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            return Some(spec);
        }
    }
    None
}

/// `/tests` paths to the task's files, as Harbor uploads each root to
/// `/tests` in turn.
fn view(src: &Path, roots: &[String]) -> IndexMap<String, String> {
    let mut v: IndexMap<String, String> = IndexMap::new();
    for root in roots {
        let base = files::under(src, &files::rel(root));
        if !base.is_dir() {
            continue;
        }
        for rel in files::list_all(&base) {
            let r = files::posix(&rel);
            v.insert(format!("/tests/{r}"), format!("{root}/{r}"));
        }
    }
    v
}

/// The judge TOMLs RewardKit discovers under one `/tests` folder: valid
/// ones, and ones that fail its model.
fn discover(
    src: &Path,
    view: &IndexMap<String, String>,
    folder: &str,
) -> (Vec<String>, Vec<String>) {
    let d = format!("{}/", folder.trim_end_matches('/'));
    let mut judges = Vec::new();
    let mut invalid = Vec::new();
    let mut entries: Vec<(&String, &String)> = view.iter().collect();
    entries.sort();
    for (vpath, hpath) in entries {
        let Some(rest) = vpath.strip_prefix(&d) else {
            continue;
        };
        let parts: Vec<&str> = rest.split('/').collect();
        if parts[..parts.len() - 1]
            .iter()
            .any(|p| p.starts_with('.') || p.starts_with("__"))
        {
            continue;
        }
        if !parts[parts.len() - 1].ends_with(".toml") {
            continue;
        }
        let path = files::under(src, &files::rel(hpath));
        let data = std::fs::read(&path)
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .and_then(|t| parse_toml(&t).ok());
        let Some(data) = data else {
            invalid.push(hpath.clone());
            continue;
        };
        if data.contains_key("judge") || data.contains_key("criterion") {
            if is_judge_toml(&data) {
                judges.push(hpath.clone());
            } else {
                invalid.push(hpath.clone());
            }
        }
    }
    (judges, invalid)
}

/// One rewardkit call: the script, the judge TOMLs it grades with, and the
/// folders it was given.
struct Run {
    script: String,
    judges: Vec<String>,
    dirs: Vec<String>,
}

/// What RewardKit does in a task, as import records it.
#[derive(Debug, Default)]
pub(crate) struct Scan {
    /// Package paths of the scripts that run RewardKit, sorted.
    pub scripts: Vec<String>,
    /// Package paths of its judge TOMLs, sorted.
    pub judges: Vec<String>,
    /// The `installs` entry: the package spec the task pins, if any.
    pub installs: Option<String>,
    /// Whether anything at all was found (calls or judge files).
    pub found: bool,
    /// Plain notes for people.
    pub notes: Vec<String>,
}

impl Scan {
    /// The `[import.harbor] rewardkit` table.
    pub(crate) fn record(&self) -> Table {
        let mut out = Table::new();
        if let Some(i) = &self.installs {
            out.insert("installs".into(), Val::Str(i.clone()));
        }
        out.insert(
            "scripts".into(),
            Val::List(self.scripts.iter().cloned().map(Val::Str).collect()),
        );
        out.insert(
            "judges".into(),
            Val::List(self.judges.iter().cloned().map(Val::Str).collect()),
        );
        out
    }
}

/// Scan the Harbor task at `src`. `None` when it has no RewardKit.
pub(crate) fn scan(src: &Path, cfg: &Table, steps: &BTreeSet<String>) -> Option<Scan> {
    let mut scripts: Vec<(String, Option<String>)> = vec![("tests/test.sh".into(), None)];
    scripts.extend(
        steps
            .iter()
            .map(|n| (format!("steps/{n}/tests/test.sh"), Some(n.clone()))),
    );
    let workdir = cfg
        .get("environment")
        .and_then(Val::as_table)
        .and_then(|e| e.get("workdir"))
        .and_then(Val::as_str)
        .map(str::to_string);
    let separate = cfg
        .get("verifier")
        .and_then(Val::as_table)
        .and_then(|v| v.get("environment_mode"))
        .and_then(Val::as_str)
        == Some("separate");
    let mut images: Vec<String> = if separate {
        vec!["tests/Dockerfile".into(), "environment/Dockerfile".into()]
    } else {
        vec!["environment/Dockerfile".into(), "tests/Dockerfile".into()]
    };
    for n in steps {
        images.push(format!("steps/{n}/environment/Dockerfile"));
        images.push(format!("steps/{n}/tests/Dockerfile"));
    }
    let recipes: Vec<String> = images
        .iter()
        .filter_map(|p| std::fs::read(files::under(src, &files::rel(p))).ok())
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .collect();
    let mut runs: Vec<Run> = Vec::new();
    let mut run_steps: BTreeSet<String> = BTreeSet::new();
    let mut graded: BTreeSet<String> = BTreeSet::new();
    let mut install: Option<Option<String>> = None;
    for (spath, step) in &scripts {
        let path = files::under(src, &files::rel(spath));
        let Ok(bytes) = std::fs::read(&path) else {
            continue;
        };
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let (lines, depths, found_calls) = calls(&text);
        for call in found_calls {
            let mut env0 = harbor_env(cfg, step.as_deref());
            if let Some(w) = &workdir {
                env0.entry("PWD".into()).or_insert(EnvVal::Str(w.clone()));
            }
            let (mut env, before) = shell::state(&lines, &depths, call.index, &env0);
            let a = argv(&call.args);
            let prefix_words: Vec<&str> = call.prefix.split_whitespace().collect();
            for w in prefix_words {
                if let Some((k, v)) = w.split_once('=').filter(|_| !w.starts_with('=')) {
                    let value = shell::expand(v, &env)
                        .filter(|s| !s.is_empty())
                        .map_or(EnvVal::Unknown, EnvVal::Str);
                    env.insert(k.to_string(), value);
                }
            }
            for entry in &a.env {
                let (k, v) = entry.split_once('=').unwrap_or((entry.as_str(), ""));
                let value = shell::expand(v, &env)
                    .filter(|s| !s.is_empty())
                    .map_or(EnvVal::Unknown, EnvVal::Str);
                env.insert(k.to_string(), value);
            }
            for (flag, var) in [(&a.judge, "REWARDKIT_JUDGE"), (&a.model, "REWARDKIT_MODEL")] {
                if let Some(f) = flag {
                    let value = shell::expand(f, &env)
                        .filter(|s| !s.is_empty())
                        .map_or(EnvVal::Unknown, EnvVal::Str);
                    env.insert(var.to_string(), value);
                }
            }
            let dirs: Vec<Option<String>> = a.dirs.iter().map(|d| shell::expand(d, &env)).collect();
            let mut roots = vec!["tests".to_string()];
            if let Some(s) = step {
                roots.push(format!("steps/{s}/tests"));
            }
            let v = view(src, &roots);
            let mut judges = Vec::new();
            for d in dirs.iter().flatten() {
                if !d.is_empty() && (d == "/tests" || d.starts_with("/tests/")) {
                    let (j, inv) = discover(src, &v, d);
                    graded.extend(j.iter().cloned());
                    graded.extend(inv);
                    judges.extend(j);
                }
            }
            let mut sources = vec![call.launcher.clone()];
            sources.extend(before.iter().rev().map(|b| b.command.clone()));
            sources.extend(recipes.iter().cloned());
            if install.is_none() {
                install = Some(install_spec(&sources));
            }
            if let Some(s) = step {
                run_steps.insert(s.clone());
            }
            runs.push(Run {
                script: spath.clone(),
                judges,
                dirs: a.dirs.clone(),
            });
        }
    }
    let mut loose: Vec<String> = Vec::new();
    let mut plain = 0usize;
    let mut roots = vec!["tests".to_string()];
    roots.extend(steps.iter().map(|n| format!("steps/{n}/tests")));
    let mut tomls: Vec<String> = Vec::new();
    for root in &roots {
        let base = files::under(src, &files::rel(root));
        if !base.is_dir() {
            continue;
        }
        for rel in files::list_all(&base) {
            let r = files::posix(&rel);
            if r.ends_with(".toml") {
                tomls.push(format!("{root}/{r}"));
            }
        }
    }
    tomls.sort();
    for p in tomls {
        if graded.contains(&p) {
            continue;
        }
        let data = std::fs::read(files::under(src, &files::rel(&p)))
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .and_then(|t| parse_toml(&t).ok());
        let Some(data) = data else { continue };
        if data.contains_key("judge") || data.contains_key("criterion") {
            if is_judge_toml(&data) {
                loose.push(p);
            } else if data.contains_key("criterion") {
                plain += 1;
            }
        }
    }
    if runs.is_empty() && loose.is_empty() && plain == 0 {
        return None;
    }
    let installs = match install {
        Some(spec) => spec,
        None => install_spec(&recipes),
    };
    let mut steps_of = run_steps.clone();
    steps_of.extend(loose.iter().filter_map(|p| {
        p.strip_prefix("steps/")
            .and_then(|r| r.split('/').next())
            .map(str::to_string)
    }));
    let pkg = |h: &str, s: &BTreeSet<String>| {
        harbor_path(&files::rel(h), s).map_or_else(|| h.to_string(), |r| files::posix(&r))
    };
    let scripts_found: BTreeSet<String> = runs.iter().map(|r| pkg(&r.script, steps)).collect();
    let mut judge_set: BTreeSet<String> = runs
        .iter()
        .flat_map(|r| r.judges.iter().map(|j| pkg(j, &steps_of)))
        .collect();
    judge_set.extend(loose.iter().map(|j| pkg(j, &steps_of)));
    let mut notes = Vec::new();
    for Run {
        script: spath,
        judges,
        dirs,
    } in &runs
    {
        let over = if dirs.is_empty() {
            "/tests".to_string()
        } else {
            dirs.join(", ")
        };
        notes.push(format!(
            "{} runs RewardKit over {over}{}; recorded under [import.harbor] rewardkit",
            pkg(spath, steps),
            if judges.is_empty() {
                "; its programmatic checks stay in test.sh".to_string()
            } else {
                format!(
                    "; its judges ({}) stay a script, so on Harbor scores do not change",
                    judges.len()
                )
            }
        ));
    }
    for j in &loose {
        notes.push(format!(
            "{} is a RewardKit judge TOML that no rewardkit call in test.sh grades; it stays a file, recorded under [import.harbor] rewardkit",
            pkg(j, &steps_of)
        ));
    }
    Some(Scan {
        scripts: scripts_found.into_iter().collect(),
        judges: judge_set.into_iter().collect(),
        installs: installs.map(|spec| {
            if spec.starts_with("harbor@") {
                spec
            } else {
                format!("harbor-rewardkit{spec}")
            }
        }),
        found: !runs.is_empty() || !loose.is_empty(),
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_calls_and_launchers() {
        let (_, _, c) = calls(
            "#!/bin/sh\nmkdir -p /logs/verifier/rewardkit\nuvx --from 'harbor-rewardkit~=0.1.0' rewardkit \\\n  --workspace /app /tests > /logs/out.txt\n",
        );
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].launcher, "uvx --from 'harbor-rewardkit~=0.1.0' ");
        assert_eq!(c[0].args, vec!["--workspace", "/app", "/tests"]);
        let (_, _, c) = calls("cd /app && env A=1 rewardkit /tests\n");
        assert_eq!(c[0].prefix, "env A=1 ");
        let (_, _, c) = calls("echo rewardkit\n");
        assert!(c.is_empty(), "an argument is not a call");
        let (_, _, c) = calls("python3 -m rewardkit /tests/judges\n");
        assert_eq!(c[0].launcher, "python3 -m ");
    }

    #[test]
    fn install_specs() {
        assert_eq!(
            install_spec(&["pip install harbor-rewardkit==0.2.1".into()]).as_deref(),
            Some("==0.2.1")
        );
        assert_eq!(
            install_spec(&["uv tool install harbor-rewardkit".into()]).as_deref(),
            Some("")
        );
        assert_eq!(
            install_spec(&["pip install harbor-rewardkit @ git+https://github.com/harbor-framework/harbor@58789aad".into()]).as_deref(),
            Some("harbor@58789aad")
        );
        assert_eq!(install_spec(&["nothing".into()]), None);
    }

    #[test]
    fn argv_reads_flags() {
        let a = argv(&[
            "--workspace".into(),
            "/app".into(),
            "--judge=x".into(),
            "/tests".into(),
            "--je".into(),
            "K=V".into(),
        ]);
        assert_eq!(a.dirs, vec!["/tests"]);
        assert_eq!(a.judge.as_deref(), Some("x"));
        assert_eq!(a.env, vec!["K=V"]);
    }
}
