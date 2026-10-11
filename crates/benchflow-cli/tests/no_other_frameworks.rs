//! What the `bench` binary prints names no other evaluation framework:
//! `bench --help`, `bench agents` and every agent's page, `bench explain`
//! and every code's page, and every `bench schema`; and what `bench check`,
//! `bench import`, `bench export`, `bench run --dry-run`, `bench new
//! --list` and failing commands print over the test fixtures. Names a user
//! may still type keep working without being shown. Nothing here starts a
//! sandbox.

#[path = "../src/other_frameworks.rs"]
mod other_frameworks;

use std::path::{Path, PathBuf};
use std::process::Command;

use other_frameworks::{named, named_except};

/// `bench <args>` in an empty folder: what it printed, both streams.
fn bench(dir: &Path, args: &[&str]) -> String {
    bench_env(dir, args, &[])
}

/// [`bench`] with more environment variables. The cache is a folder of its
/// own, so nothing from an earlier run is read.
fn bench_env(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> String {
    bench_exit(dir, args, env).1
}

/// [`bench_env`], and the exit code. Proxy exceptions from this machine
/// are removed, so a test's proxy is the only way out.
fn bench_exit(dir: &Path, args: &[&str], env: &[(&str, &str)]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_bench"))
        .args(args)
        .current_dir(dir)
        .env("NO_COLOR", "1")
        .env("COLUMNS", "400")
        .env("BENCHFLOW_CACHE_DIR", dir.join(".cache"))
        // The ACP registry from the snapshot this binary ships, never the
        // network: a run must not wait on a fetch, and outside entries
        // (which the naming policy exempts) must not fail the test.
        .env("BENCHFLOW_ACP_REGISTRY", acp_snapshot())
        .env_remove("NO_PROXY")
        .env_remove("no_proxy")
        .env_remove("ALL_PROXY")
        .env_remove("all_proxy")
        .envs(env.iter().copied())
        .output()
        .unwrap();
    (
        out.status.code().unwrap_or(-1),
        format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ),
    )
}

/// The ACP registry snapshot built into bench, as a file the runs read
/// instead of the network; whichever snapshot the build carries.
fn acp_snapshot() -> &'static Path {
    static FILE: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    FILE.get_or_init(|| {
        let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("acp-registry-snapshot.json");
        std::fs::write(&path, benchflow_agent::acp_registry::SNAPSHOT).unwrap();
        path
    })
}

/// The proxy settings of a machine that cannot reach the network.
const OFFLINE: [(&str, &str); 4] = [
    ("HTTPS_PROXY", "http://127.0.0.1:1"),
    ("HTTP_PROXY", "http://127.0.0.1:1"),
    ("https_proxy", "http://127.0.0.1:1"),
    ("http_proxy", "http://127.0.0.1:1"),
];

/// A dataset that cannot be looked up because a default registry cannot be
/// read is not a usage error: it exits 1 ("could not be looked up"), not 2
/// ("no dataset"), whether both registries are out of reach or only one is
/// and the other lacks it.
#[test]
fn an_unreadable_registry_is_not_a_missing_dataset() {
    let dir = tempfile::tempdir().unwrap();
    let args = ["run", "-d", "nosuch-bench@1.0", "--dry-run"];
    let (code, said) = bench_exit(dir.path(), &args, &OFFLINE);
    assert_eq!(code, 1, "{said}");
    assert!(said.contains("could not be looked up"), "{said}");
    // BenchFlow's registry read from a fresh cached copy, which lacks the
    // dataset; the public one out of reach.
    let copy = benchflow_run::registry::cached_copy(
        &dir.path().join(".cache"),
        benchflow_run::registry::BENCHFLOW_REGISTRY,
    );
    std::fs::create_dir_all(copy.parent().unwrap()).unwrap();
    std::fs::write(
        &copy,
        r#"[{"name": "nosuch-bencg", "version": "1.0", "tasks": []}]"#,
    )
    .unwrap();
    let (code, said) = bench_exit(dir.path(), &args, &OFFLINE);
    assert_eq!(code, 1, "{said}");
    assert!(said.contains("could not be looked up"), "{said}");
    assert!(
        said.contains("could not read the public dataset registry"),
        "{said}"
    );
    assert!(said.contains("it is not in BenchFlow's registry"), "{said}");
    assert!(said.contains("similar names: nosuch-bencg"), "{said}");
    assert!(named(&said).is_empty(), "{said}");
}

/// The first word of each line that starts with two spaces and then a name
/// matching `pick`.
fn first_words(text: &str, pick: impl Fn(&str) -> bool) -> Vec<String> {
    text.lines()
        .filter_map(|l| l.strip_prefix("  "))
        .filter_map(|l| l.split_whitespace().next())
        .filter(|w| pick(w))
        .map(str::to_string)
        .collect()
}

#[test]
fn what_bench_prints_names_no_other_framework() {
    let dir = tempfile::tempdir().unwrap();
    let mut pages: Vec<(String, String)> = Vec::new();
    let mut run = |args: &[&str]| {
        pages.push((format!("bench {}", args.join(" ")), bench(dir.path(), args)));
    };
    run(&["--help"]);
    run(&["agents"]);
    run(&["explain"]);
    run(&["--json", "explain"]);
    for schema in ["task", "job", "result", "lock"] {
        run(&["schema", schema]);
    }
    let agents = first_words(&bench(dir.path(), &["agents"]), |w| {
        w.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
            && w != "name"
    });
    assert!(
        agents.len() > 30,
        "only {} agents: {agents:?}",
        agents.len()
    );
    for a in &agents {
        run(&["agents", a]);
    }
    let codes = first_words(&bench(dir.path(), &["explain"]), |w| {
        w.len() == 4
            && w.starts_with(|c: char| c.is_ascii_uppercase())
            && w[1..].chars().all(|c| c.is_ascii_digit())
    });
    assert!(codes.len() > 100, "only {} codes", codes.len());
    for c in &codes {
        run(&["explain", c]);
    }
    let found: Vec<String> = pages
        .iter()
        .flat_map(|(cmd, text)| named(text).into_iter().map(move |n| format!("{cmd}: {n}")))
        .collect();
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// `karotte-external`, the external agent's older name, still names it,
/// but no list shows it.
#[test]
fn an_old_agent_name_still_works_unlisted() {
    let dir = tempfile::tempdir().unwrap();
    let old = bench(dir.path(), &["agents", "karotte-external"]);
    let new = bench(dir.path(), &["agents", "external"]);
    assert_eq!(old, new);
    assert!(new.contains("backend"), "{new}");
    assert!(!bench(dir.path(), &["agents"]).contains("karotte"));
}

/// `bench --json agents` lists the agents, as `bench agents` does.
#[test]
fn the_agent_list_has_a_json_form() {
    let dir = tempfile::tempdir().unwrap();
    let text = bench(dir.path(), &["--json", "agents"]);
    let v: serde_json::Value =
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("{e}: {text:?}"));
    let agents = v.as_array().unwrap();
    assert!(agents.len() > 30, "only {} agents", agents.len());
    let names: Vec<&str> = agents.iter().filter_map(|a| a["name"].as_str()).collect();
    for n in ["oracle", "nop", "claude", "external"] {
        assert!(names.contains(&n), "{n} not in {names:?}");
    }
    assert!(
        agents
            .iter()
            .all(|a| a["description"].is_string() && a["harnesses"].is_array())
    );
    // Harnesses are what --harness takes; the rest is the agent's kind.
    let by = |n: &str| agents.iter().find(|a| a["name"] == n).unwrap().clone();
    assert_eq!(
        by("claude")["harnesses"],
        serde_json::json!(["acp", "headless"])
    );
    assert_eq!(by("claude")["kind"], "manifest");
    assert_eq!(by("oracle")["kind"], "script");
    assert_eq!(by("terminus-2")["kind"], "native");
    assert_eq!(by("external")["kind"], "backend");
    let found = named(&text);
    assert!(found.is_empty(), "{found:?}");
    // One agent's page is its entry; the ACP registry is a list too.
    for (name, want) in [
        ("claude", "claude"),
        ("terminus-2", "terminus-2"),
        ("terminus2", "terminus-2"),
        ("external", "external"),
        ("karotte-external", "external"),
        ("dspy-rlm", "dspy-rlm"),
    ] {
        let page = bench(dir.path(), &["--json", "agents", name]);
        let v: serde_json::Value =
            serde_json::from_str(&page).unwrap_or_else(|e| panic!("{name}: {e}: {page:?}"));
        assert_eq!(v["name"], want, "{page}");
        assert!(named(&page).is_empty(), "{page}");
    }
    let reg = bench(dir.path(), &["--json", "agents", "--registry"]);
    let v: serde_json::Value =
        serde_json::from_str(&reg).unwrap_or_else(|e| panic!("{e}: {reg:?}"));
    assert!(v.as_array().is_some_and(|a| a.len() > 10), "{reg}");
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .unwrap()
}

/// `typed`, each path in it as bench shows paths under the home folder
/// (`~/...`), and each path's last name (a task is named after its folder).
fn with_home(typed: &[&str]) -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let mut out: Vec<String> = typed.iter().map(|t| t.to_string()).collect();
    out.extend(
        typed
            .iter()
            .filter(|t| t.starts_with('/'))
            .filter_map(|t| Path::new(t).file_name())
            .map(|n| n.to_string_lossy().into_owned()),
    );
    if !home.is_empty() {
        out.extend(
            typed
                .iter()
                .filter_map(|t| t.strip_prefix(home.as_str()))
                .map(|rest| format!("~{rest}")),
        );
    }
    out
}

/// Every folder under `root` holding a task.md or task.toml.
fn task_folders(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter(|e| e.file_name() == "task.md" || e.file_name() == "task.toml")
        .filter_map(|e| e.path().parent().map(Path::to_path_buf))
        // Originals an import kept are not a task anyone checks.
        .filter(|p| !p.components().any(|c| c.as_os_str() == "import"))
        .collect();
    out.sort();
    out.dedup();
    out
}

/// The folders directly under `root`.
fn folders(root: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(root)
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    out.sort();
    out
}

/// Every text file's content under `dirs`, lower-cased: the package's own
/// words, which bench may repeat (a canary, a comment it points at, a
/// task's name).
fn own_text(dirs: &[&Path]) -> String {
    let mut out = String::new();
    for d in dirs {
        for e in walkdir::WalkDir::new(d).into_iter().filter_map(Result::ok) {
            // Its files' paths, then their text.
            if let Ok(rel) = e.path().strip_prefix(d) {
                out.push_str(&rel.to_string_lossy().to_lowercase());
                out.push('\n');
            }
            if e.file_type().is_file()
                && let Ok(t) = std::fs::read_to_string(e.path())
            {
                out.push_str(&t.to_lowercase());
                out.push('\n');
            }
        }
    }
    out
}

/// The texts of `out`: when it is JSON (one document, or one per line), each
/// string in it, so a value's end is its own; otherwise its lines.
fn texts(out: &str) -> Vec<String> {
    let mut found = Vec::new();
    let mut collect = |v: &serde_json::Value| collect_strings(v, &mut found);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(out.trim()) {
        collect(&v);
        return found;
    }
    let mut plain = Vec::new();
    for line in out.lines() {
        match serde_json::from_str::<serde_json::Value>(line.trim()) {
            Ok(v) if v.is_object() || v.is_array() => collect(&v),
            _ => plain.push(line.to_string()),
        }
    }
    found.extend(plain);
    found
        .iter()
        .flat_map(|t| t.lines().map(str::to_string).collect::<Vec<_>>())
        .collect()
}

fn collect_strings(v: &serde_json::Value, out: &mut Vec<String>) {
    match v {
        serde_json::Value::String(s) => out.push(s.clone()),
        serde_json::Value::Array(a) => a.iter().for_each(|x| collect_strings(x, out)),
        serde_json::Value::Object(o) => o.iter().for_each(|(k, x)| {
            out.push(k.clone());
            collect_strings(x, out)
        }),
        _ => {}
    }
}

/// The lines of `out` that name a framework in bench's own words: a name
/// that is not in a value typed (`typed`, such as the paths given; flags do
/// not count), and not where bench quotes the package (`own`: its files'
/// text and their paths). The line and `own` are compared as plain words
/// (CamelCase split, lower case, punctuation as spaces). A quote is a long
/// span of the package's: the whole source line after a code frame's `N |`
/// gutter (at least 20 characters), or each mention with four words around
/// it. Failing that, a mention is the package's only when every token
/// holding it (a task name such as `acme/harbor-keys`) is longer than the
/// name and appears verbatim in `own`. Anything else is reported, including
/// a name `named()` finds but no word of the line holds.
fn leaks(cmd: &str, out: &str, own: &str, typed: &[&str]) -> Vec<String> {
    use other_frameworks::plain_words;
    let typed: Vec<&str> = typed
        .iter()
        .copied()
        .filter(|t| !t.starts_with('-'))
        .collect();
    let typed = with_home(&typed);
    let typed: Vec<&str> = typed.iter().map(String::as_str).collect();
    let own_raw = own.to_lowercase();
    let own = plain_words(own);
    let mut found = Vec::new();
    for line in texts(out) {
        let names = named_except(&line, &typed);
        if names.is_empty() {
            continue;
        }
        let source = line
            .trim_start()
            .trim_start_matches(|c: char| c.is_ascii_digit())
            .trim_start();
        let source = source
            .strip_prefix('|')
            .or_else(|| source.strip_prefix('│'))
            .map(plain_words);
        if source.is_some_and(|s| s.len() >= 20 && own.contains(&s)) {
            continue;
        }
        let plain = plain_words(&line);
        let words: Vec<&str> = plain.split(' ').collect();
        let tokens: Vec<String> = line
            .split_whitespace()
            .map(|t| {
                t.trim_matches(|c: char| !c.is_alphanumeric() && !"/-_.@".contains(c))
                    .to_lowercase()
            })
            .collect();
        let quoted = |name: &str| {
            let plain_name = plain_words(name);
            let n: Vec<&str> = plain_name.split(' ').collect();
            let at: Vec<usize> = (0..(words.len() + 1).saturating_sub(n.len()))
                .filter(|&i| words[i..i + n.len()] == n[..])
                .collect();
            let in_context = !at.is_empty()
                && at.iter().all(|&i| {
                    // Four words of context, split any way around the name.
                    (0..=4).any(|before| {
                        let end = i + n.len() + 4 - before;
                        i >= before
                            && end <= words.len()
                            && own.contains(&words[i - before..end].join(" "))
                    })
                });
            let holding: Vec<&String> =
                tokens.iter().filter(|t| named(t).contains(&name)).collect();
            let package_tokens = !holding.is_empty()
                && holding
                    .iter()
                    .all(|t| t.len() > name.len() + 2 && own_raw.contains(t.as_str()));
            in_context || package_tokens
        };
        if names.iter().any(|n| !quoted(n)) {
            found.push(format!("{cmd}: {line}"));
        }
    }
    found
}

#[test]
fn the_leak_finder_tells_bench_from_the_package() {
    let own = "# misplaced: harbor ignores this here.\nname = \"acme/harbor-keys\"\n";
    assert!(
        leaks(
            "c",
            "  7 | # Misplaced: Harbor ignores this here.",
            own,
            &[]
        )
        .is_empty()
    );
    assert!(leaks("c", "task acme/harbor-keys is ok", own, &[]).is_empty());
    // ...but only because the package has that name.
    assert_eq!(leaks("c", "task acme/harbor-keys is ok", "", &[]).len(), 1);
    // A quote with its punctuation, and a frame line of a multi-line span.
    assert!(
        leaks(
            "c",
            "note: bench repeats \"# Misplaced: Harbor ignores this here.\"",
            own,
            &[]
        )
        .is_empty()
    );
    assert!(
        leaks(
            "c",
            "  7 | | # Misplaced: Harbor ignores this here.",
            own,
            &[]
        )
        .is_empty()
    );
    // Names glued to punctuation or inside CamelCase are names.
    for line in [
        "note: written in *Harbor* layout",
        "note: kept as \u{201c}Harbor\u{201d} wrote it",
        "note: the split layout\u{2014}Harbor's own",
        "hint: pass --to=harbor to keep it",
        "export to=harbor done",
        "set runner=Harbor",
        "error: <Harbor> section is not read",
        "error: convert::harbor::Error",
        "note: see docs/harbor/layout.md",
        "warning: task #harbor tag",
        "@harbor",
        "error: SplitHarborConfig { name: \"x\" }",
        "graded by InspectAI",
    ] {
        assert_eq!(leaks("c", line, own, &[]).len(), 1, "{line}");
    }
    assert_eq!(leaks("c", "Harbor seeds the session", own, &[]).len(), 1);
    assert_eq!(leaks("c", "warning: Harbor ignores it", "", &[]).len(), 1);
    assert!(leaks("c", "cannot read /x/harbor/odd", "", &["/x/harbor/odd"]).is_empty());
    // Two-word names are names too.
    assert_eq!(
        leaks("c", "converted from Inspect AI logs", "", &[]).len(),
        1
    );
    // Two words of the package's are not a quote of it.
    let judge = "# harbor rewardkit llm-as-judge\n";
    assert_eq!(
        leaks(
            "c",
            "warning[T222]: Harbor RewardKit grades these judge files",
            judge,
            &[]
        )
        .len(),
        1
    );
    // A flag is not a typed value.
    assert_eq!(leaks("c", "uses inspect-ai", "", &["-a"]).len(), 1);
    // In JSON, an address ends at its quote.
    assert_eq!(
        leaks(
            "c",
            r#"{"url":"https://example.com/x","next":"Harbor runs"}"#,
            "",
            &[]
        )
        .len(),
        1
    );
}

/// `bench check` (text and JSON) of every fixture task; `bench import` of
/// every split-layout, 0.8 and Terminal-Bench v1 fixture, `bench check` of
/// what it wrote, and `bench export` of that; `bench export` (also with
/// `--allow-degraded`) of every export fixture.
#[test]
fn what_check_import_and_export_print_names_no_other_framework() {
    let dir = tempfile::tempdir().unwrap();
    let root = fixtures();
    let mut found = Vec::new();
    let mut runs = 0;
    let mut scan = |args: &[&str], own: &[&Path], typed: &[&str]| {
        let out = bench(dir.path(), args);
        runs += 1;
        found.extend(leaks(
            &format!("bench {}", args.join(" ")),
            &out,
            &own_text(own),
            typed,
        ));
    };
    for task in task_folders(&root.join("benchflow-task/tests/fixtures"))
        .into_iter()
        .chain(task_folders(
            &root.join("benchflow-run/tests/fixtures/tasks"),
        ))
    {
        let t = task.to_str().unwrap();
        scan(&["check", t], &[&task], &[t]);
        scan(&["--json", "check", t], &[&task], &[t]);
    }
    let convert = root.join("benchflow-task/tests/fixtures/convert");
    let sources: Vec<PathBuf> = ["harbor", "v06"]
        .iter()
        .flat_map(|k| folders(&convert.join(k)))
        .chain(folders(
            &root.join("benchflow-task/tests/fixtures/tbench/input"),
        ))
        .collect();
    for (i, src) in sources.iter().enumerate() {
        let s = src.to_str().unwrap();
        let out = dir.path().join(format!("import-{i}"));
        let o = out.to_str().unwrap();
        scan(
            &["--json", "import", s, &format!("{o}-json")],
            &[src],
            &[s, o],
        );
        scan(&["import", s, o], &[src], &[s, o]);
        if !out.join("task.md").is_file() {
            continue;
        }
        scan(&["check", o], &[src], &[s, o]);
        let back = format!("{o}-split");
        scan(&["export", o, &back], &[src], &[s, o, &back]);
        scan(
            &["--json", "export", o, &format!("{back}-json")],
            &[src],
            &[s, o, &back],
        );
    }
    for (i, src) in folders(&convert.join("export")).iter().enumerate() {
        let s = src.to_str().unwrap();
        let o = dir.path().join(format!("export-{i}"));
        let o = o.to_str().unwrap();
        scan(&["export", s, o], &[src], &[s, o]);
        scan(
            &["export", "--allow-degraded", s, &format!("{o}-d")],
            &[src],
            &[s, o],
        );
        scan(&["--json", "export", s, &format!("{o}-j")], &[src], &[s, o]);
    }
    assert!(runs > 600, "only {runs} runs");
    assert!(found.is_empty(), "{}", found.join("\n"));
}

/// `bench run --dry-run` on the run fixtures (the external agent under its
/// old name too), `bench new --list`, a dataset no registry can be reached
/// for, and commands that fail on what they were given.
#[test]
fn what_runs_and_errors_print_names_no_other_framework() {
    let dir = tempfile::tempdir().unwrap();
    let tasks = fixtures().join("benchflow-run/tests/fixtures/tasks");
    let hello = tasks.join("hello");
    let h = hello.to_str().unwrap();
    let mut found = Vec::new();
    let mut scan = |args: &[&str], env: &[(&str, &str)], typed: &[&str]| -> String {
        let out = bench_env(dir.path(), args, env);
        found.extend(leaks(&format!("bench {}", args.join(" ")), &out, "", typed));
        out
    };
    for task in folders(&tasks) {
        let t = task.to_str().unwrap();
        for agent in ["oracle", "nop"] {
            scan(&["run", t, "-a", agent, "--dry-run"], &[], &[t]);
        }
    }
    for agent in ["external", "karotte-external"] {
        scan(&["run", h, "-a", agent, "--dry-run"], &[], &[h, agent]);
        scan(
            &[
                "run",
                h,
                "-a",
                agent,
                "--backend",
                "http://127.0.0.1:1",
                "--dry-run",
            ],
            &[],
            &[h, agent],
        );
    }
    scan(&["new", "--list"], &[], &[]);
    for template in ["basic", "python", "split"] {
        let name = format!("new-{template}");
        scan(&["new", &name, "--template", template], &[], &[&name]);
    }
    scan(&["agents", "--registry"], &[], &[]);
    scan(&["test", "--list-probes"], &[], &[]);
    scan(&["--json", "test", "--list-probes"], &[], &[]);
    let jobs = fixtures().join("benchflow-run/tests/fixtures/export/jobs");
    for job in folders(&jobs) {
        let j = job.to_str().unwrap();
        scan(&["results", j], &[], &[j]);
        scan(&["--json", "results", j], &[], &[j]);
    }
    // Offline: the HTTP library's errors repeat the registry's address.
    let offline = OFFLINE;
    let said = scan(
        &["run", "-d", "nosuch-bench@1.0", "--dry-run"],
        &offline,
        &["nosuch-bench@1.0"],
    );
    assert!(said.contains("the public dataset registry"), "{said}");
    assert!(said.contains("registries/ in its cache folder"), "{said}");
    assert!(!said.contains("https://"), "{said}");
    scan(
        &["--json", "run", "-d", "nosuch-bench@1.0", "--dry-run"],
        &offline,
        &["nosuch-bench@1.0"],
    );
    for args in [
        &["run", "/nonexistent/task", "--dry-run"][..],
        &["run", h, "-a", "nosuch-agent", "--dry-run"],
        &["run", h, "-a", "harbor-agent", "--dry-run"],
        &["agents", "nosuch-agent"],
        &["check", "/nonexistent/task"],
        &["explain", "Z999"],
        &["import", "/nonexistent/task"],
        &["export", "/nonexistent/task"],
        &["new", "x", "--template", "nosuch"],
    ] {
        let typed: Vec<&str> = args.to_vec();
        scan(args, &[], &typed);
    }
    assert!(found.is_empty(), "{}", found.join("\n"));
}
