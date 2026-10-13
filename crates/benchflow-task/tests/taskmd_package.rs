//! A checked draft 2 package as the run engine's `TaskPackage`: honored
//! fields map onto it, and the fields BenchFlow 0.8 refuses become
//! `Unsupported` entries that name them.

use std::path::Path;
use std::time::Duration;

use benchflow_task::check::CheckOptions;
use benchflow_task::package::{ImageRef, Isolation, OnTimeout, TaskPackage};
use benchflow_task::taskmd;
use benchflow_task::values::Network;
use tempfile::TempDir;

fn package(config: &str, files: &[(&str, &str)], instruction: &str) -> (TempDir, TaskPackage) {
    let dir = tempfile::tempdir().unwrap();
    let mut all: Vec<(&str, String)> = vec![
        ("sandbox/Dockerfile", "FROM scratch\n".into()),
        ("verifier/test.sh", "pytest\n".into()),
        ("oracle/solve.sh", "echo fix\n".into()),
        (
            "task.md",
            format!(
                "{instruction}\n\n```toml task\nname = \"acme/pkg\"\nversion = \"1.2.0\"\ntitle = \"A task\"\n{config}```\n"
            ),
        ),
    ];
    all.extend(files.iter().map(|(p, t)| (*p, t.to_string())));
    for (rel, text) in all {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    let report = taskmd::check(dir.path(), &CheckOptions { deep: true });
    let errors: Vec<&str> = report
        .diagnostics
        .iter()
        .filter(|d| d.is_error())
        .map(|d| d.message.as_str())
        .collect();
    assert!(errors.is_empty(), "{errors:#?}");
    let pkg = report.package.expect("a package");
    (dir, pkg)
}

fn refused(pkg: &TaskPackage) -> Vec<(&str, bool)> {
    pkg.unsupported
        .iter()
        .map(|u| (u.field.as_str(), u.agents_only))
        .collect()
}

#[test]
fn hello_world_maps_onto_the_package() {
    let dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/taskmd/examples/hello-world");
    let report = taskmd::check(&dir, &CheckOptions { deep: true });
    let pkg = report.package.expect("hello-world checks clean");
    assert_eq!(pkg.id, "examples/hello-world");
    assert_eq!(pkg.name, "hello-world");
    assert_eq!(pkg.version.as_deref(), Some("1.3.0"));
    assert_eq!(pkg.title.as_deref(), Some("Hello, world"));
    assert_eq!(pkg.instruction, taskmd::agent_view(&dir).unwrap());
    assert_eq!(pkg.agent.timeout, Some(Duration::from_secs(120)));
    assert_eq!(
        pkg.sandbox.image,
        ImageRef::Build {
            context: dir.join("sandbox"),
            dockerfile: dir.join("sandbox/Dockerfile")
        }
    );
    assert_eq!(pkg.verifier.mount, "/verifier");
    assert_eq!(pkg.verifier.entry, "test.sh");
    let rubric = pkg.verifier.rubric.as_ref().expect("a rubric");
    assert!(!rubric.needs_model());
    assert_eq!(rubric.criteria[0].id, "hello");
    let oracle = pkg.oracle.as_ref().expect("an oracle");
    assert_eq!(
        (oracle.mount.as_str(), oracle.entry.as_str()),
        ("/oracle", "solve.sh")
    );
    assert!(pkg.unsupported.is_empty(), "{:#?}", pkg.unsupported);
    assert!(pkg.digest.starts_with("sha256:") && pkg.digest.len() == 71);
    assert_eq!(
        report.facts[0],
        "grading needs a model: no (1 criterion decided by tests, rules, the world, or replay)"
    );
}

#[test]
fn honored_fields_fill_the_package() {
    let (dir, pkg) = package(
        "[agent]\ntimeout = \"1h30m\"\non_timeout = \"grade\"\nuser = \"agent\"\nnetwork = \"none\"\ntimeout_basis = \"wall\"\n\n\
[sandbox]\nimage = \"python:3.12-slim\"\ncpus = 2\nmemory = \"4 GB\"\ndisk = \"10 GB\"\ngpus = 1\nbuild_timeout = \"10m\"\n\
network = [\"pypi.org\"]\nworkdir = \"/app\"\nenv = { MODE = \"fast\" }\nmounts = []\n\
ready = { run = \"curl -f localhost\", interval = \"2s\" }\n\
outputs = [\"/app/report.md\", { path = \"/app/out/\", save_as = \"results\", exclude = [\"*.pyc\"] }]\n\n\
[verifier]\ntimeout = \"5m\"\nuser = \"root\"\nenv = { STRICT = \"1\" }\nnetwork = \"none\"\nisolation = \"separate\"\nmount = \"/tests\"\n\n\
[oracle]\nenv = { SEED = \"3\" }\nmount = \"/solution\"\n",
        &[],
        "Write the report.",
    );
    assert!(pkg.unsupported.is_empty(), "{:#?}", pkg.unsupported);
    assert_eq!(pkg.agent.timeout, Some(Duration::from_secs(5400)));
    assert_eq!(pkg.agent.on_timeout, OnTimeout::Grade);
    assert_eq!(pkg.agent.user.as_deref(), Some("agent"));
    assert_eq!(pkg.agent.network, Some(Network::None));
    assert_eq!(
        pkg.sandbox.image,
        ImageRef::Pull {
            reference: "python:3.12-slim".into()
        }
    );
    assert_eq!(
        (
            pkg.sandbox.cpus,
            pkg.sandbox.memory_mb,
            pkg.sandbox.disk_mb,
            pkg.sandbox.gpus
        ),
        (Some(2.0), Some(4096), Some(10240), 1)
    );
    assert_eq!(pkg.sandbox.build_timeout, Some(Duration::from_secs(600)));
    assert_eq!(pkg.sandbox.network, Network::Allow(vec!["pypi.org".into()]));
    assert_eq!(pkg.workdir(), "/app");
    assert_eq!(pkg.sandbox.env["MODE"], "fast");
    let ready = pkg.sandbox.ready.as_ref().unwrap();
    assert_eq!(ready.run, "curl -f localhost");
    assert_eq!(
        (ready.interval, ready.timeout, ready.retries),
        (Duration::from_secs(2), Duration::from_secs(30), 3)
    );
    assert_eq!(pkg.sandbox.outputs.len(), 2);
    assert_eq!(pkg.sandbox.outputs[0].save_as, "report.md");
    assert_eq!(pkg.sandbox.outputs[1].save_as, "results");
    assert_eq!(pkg.sandbox.outputs[1].exclude, ["*.pyc"]);
    assert_eq!(pkg.verifier.timeout, Some(Duration::from_secs(300)));
    assert_eq!(pkg.verifier.user.as_deref(), Some("root"));
    assert_eq!(pkg.verifier.isolation, Isolation::Separate);
    assert_eq!(pkg.verifier.network, Some(Network::None));
    assert_eq!(pkg.verifier.mount, "/tests");
    assert_eq!(pkg.verifier.dir, dir.path().join("verifier"));
    assert!(pkg.verifier.rubric.is_none());
    let oracle = pkg.oracle.as_ref().unwrap();
    assert_eq!(
        (oracle.mount.as_str(), oracle.env["SEED"].as_str()),
        ("/solution", "3")
    );
    assert_eq!(pkg.hidden_mounts(), ["/tests", "/solution"]);
    assert_eq!(pkg.config["agent"]["user"], "agent");
}

#[test]
fn refused_fields_are_named() {
    let (_dir, pkg) = package(
        "[agent]\non_timeout = \"fail\"\nbudget = { tokens = \"2M\" }\nsystem_prompt_append = \"x.md\"\ntimeout_basis = \"environment\"\n\n\
[sandbox]\nos = \"windows\"\ngpu_types = [\"a100\"]\ntpu = \"v4\"\nnetwork = { block = [\"x.com\"] }\nskills = \"sandbox/skills\"\n\
mounts = [\"data:/data\"]\nboundary = \"vm\"\nenv = { KEY = \"${HOST_KEY}\" }\noutputs = [{ path = \"/app/x\", max_bytes = \"10 MB\" }]\n\n\
[verifier]\nmodels = true\nsnapshot = [{ run = \"x\", reads = [\"/data\"] }]\n\n[verifier.human]\nassessors = 2\n\n\
[training]\nmax_concurrent = 2\n\n[trajectory]\nrequire = [\"tool-calls\", \"screenshots\"]\n\n[integrity]\nforbidden_sources = [\"x\"]\n\
profile = \"runner-separated\"\n\n[matrix]\naxes = []\n\n[x-benchflow]\nlegacy = true\n",
        &[],
        "Fix it.",
    );
    let got = refused(&pkg);
    for want in [
        ("[agent] on_timeout", false),
        ("[agent] budget", true),
        ("[agent] system_prompt_append", true),
        ("[agent] timeout_basis", false),
        ("[sandbox] os", false),
        ("[sandbox] tpu", false),
        ("[sandbox] mounts data:/data", false),
        ("[sandbox] boundary", false),
        ("[sandbox] env.KEY", false),
        ("[sandbox] outputs[0] max_bytes", false),
        ("[verifier] models", false),
        ("[verifier] human", false),
        ("[training] max_concurrent", false),
        ("[trajectory] require screenshots", false),
        ("[integrity] forbidden_sources", false),
        ("[integrity] profile", false),
        ("[matrix]", false),
        ("[x-benchflow]", false),
    ] {
        assert!(got.contains(&want), "{want:?} not refused: {got:#?}");
    }
    assert!(!got.iter().any(|(f, _)| f.contains("tool-calls")));
    // Skills are installed for the agent now.
    assert!(!got.iter().any(|(f, _)| f.starts_with("[sandbox] skills")));
    assert!(
        pkg.sandbox
            .skills
            .as_ref()
            .unwrap()
            .ends_with("sandbox/skills")
    );
    // A blocklist is enforced now (a uid firewall), so it is kept, not refused.
    assert!(!got.iter().any(|(f, _)| *f == "[sandbox] network"));
    assert_eq!(pkg.sandbox.network, Network::Block(vec!["x.com".into()]));
    assert_eq!(pkg.agent.on_timeout, OnTimeout::Fail);
    // Read now: GPU types are checked by the backend, snapshots run as hooks.
    assert_eq!(pkg.extras.gpu_types, ["a100"]);
    assert_eq!(pkg.extras.hooks.len(), 1);
    assert!(
        pkg.unsupported
            .iter()
            .all(|u| !u.reason.is_empty() && !u.reason.ends_with('.'))
    );
}

#[test]
fn networks_narrow_but_never_widen() {
    let (_d, pkg) = package(
        "[sandbox]\nnetwork = \"none\"\n\n[agent]\nnetwork = \"open\"\n",
        &[],
        "Fix it.",
    );
    assert_eq!(refused(&pkg), [("[agent] network", false)]);
    let (_d, pkg) = package("[agent]\nnetwork = [\"pypi.org\"]\n", &[], "Fix it.");
    assert!(pkg.unsupported.is_empty());
    assert_eq!(
        pkg.agent_network(),
        &Network::Allow(vec!["pypi.org".into()])
    );
    let (_d, pkg) = package(
        "[sandbox]\nnetwork = \"none\"\n\n[verifier]\nnetwork = \"open\"\n",
        &[],
        "Fix it.",
    );
    assert_eq!(refused(&pkg), [("[verifier] network", false)]);
    // A shared verifier's host list narrows an open sandbox: enforced.
    let (_d, pkg) = package("[verifier]\nnetwork = [\"pypi.org\"]\n", &[], "Fix it.");
    assert!(pkg.unsupported.is_empty());
    assert_eq!(
        pkg.verifier_network(),
        &Network::Allow(vec!["pypi.org".into()])
    );
    // But never widens an offline one.
    let (_d, pkg) = package(
        "[sandbox]\nnetwork = \"none\"\n\n[verifier]\nnetwork = [\"pypi.org\"]\n",
        &[],
        "Fix it.",
    );
    assert_eq!(refused(&pkg), [("[verifier] network", false)]);
    let (_d, pkg) = package(
        "[verifier]\nisolation = \"separate\"\nnetwork = \"open\"\n",
        &[],
        "Fix it.",
    );
    assert!(pkg.unsupported.is_empty());
    assert_eq!(pkg.verifier_network(), &Network::Open);
}

#[test]
fn stages_that_unlock_in_a_chain_become_later_prompts() {
    let (_d, pkg) = package(
        "[stages.intro]\nunlock = \"at_start\"\n\n[stages.review]\nunlock = \"on_submit\"\n\n[stages.ship]\nunlock = \"after:review\"\n",
        &[],
        "Fix the bug.\n\n```stage review\n<!-- canary 1 -->\nNow review it.\n```\n\n```stage ship\nShip it.\n```",
    );
    assert!(pkg.unsupported.is_empty(), "{:#?}", pkg.unsupported);
    let stages: Vec<(&str, &str)> = pkg
        .stages
        .iter()
        .map(|s| (s.name.as_str(), s.prompt.as_str()))
        .collect();
    assert_eq!(stages, [("review", "Now review it."), ("ship", "Ship it.")]);
    assert_eq!(pkg.instruction, "Fix the bug.\n");

    let (_d, pkg) = package(
        "[stages.later]\nunlock = \"on_request\"\n\n[stages.timed]\nunlock = \"on_submit\"\n\n[stages.timed.agent]\ntimeout = \"5m\"\n",
        &[],
        "Fix it.\n\n```stage later\nL.\n```\n\n```stage timed\nT.\n```",
    );
    let mut got = refused(&pkg);
    got.sort();
    assert_eq!(got, [("[stages.later] unlock", false)]);
    assert_eq!(pkg.stages.len(), 1);
    // A stage with settings of its own is graded on its own: the instruction
    // is an ungraded opening turn, then the stage.
    let steps = pkg.extras.steps.as_ref().unwrap();
    let turns: Vec<(&str, bool, Option<u64>)> = steps
        .stages
        .iter()
        .map(|s| (s.name.as_str(), s.graded, s.timeout.map(|t| t.as_secs())))
        .collect();
    assert_eq!(
        turns,
        [("instruction", false, None), ("timed", true, Some(300))]
    );
}

/// A stage's name becomes a folder in the task and in the trial, so a name
/// that is not one plain folder name is refused before anything runs.
#[test]
fn stage_names_must_be_plain_folder_names() {
    for name in ["../escape", "/abs", "a/b", "..", "."] {
        let (_d, pkg) = package(
            &format!("[stages.\"{name}\"]\nunlock = \"at_start\"\ngate = 1.0\n"),
            &[],
            "Fix it.",
        );
        assert!(
            pkg.unsupported
                .iter()
                .any(|u| u.field.starts_with("[stages.") && u.reason.contains("folder name")),
            "{name}: {:#?}",
            pkg.unsupported
        );
        assert!(pkg.extras.steps.is_none(), "{name}");
    }
}

#[test]
fn controls_come_from_the_table_and_the_folder() {
    let (dir, pkg) = package(
        "[integrity.controls]\nno_op = \"controls/no-op.sh\"\nknown_bad = [\"controls/fixed-delay.sh\"]\ncheats = [\"cheat/edit-tests.sh\"]\n",
        &[
            ("controls/no-op.sh", "#!/bin/bash\nexit 0\n"),
            ("controls/fixed-delay.sh", "#!/bin/bash\n"),
            ("controls/extra.sh", "#!/bin/bash\n"),
            ("controls/README.md", "Notes.\n"),
            ("cheat/edit-tests.sh", "#!/bin/bash\n"),
        ],
        "Fix it.",
    );
    let controls: Vec<(&str, Option<f64>)> = pkg
        .controls
        .iter()
        .map(|c| (c.id.as_str(), c.max_reward))
        .collect();
    assert_eq!(
        controls,
        [
            ("no-op", Some(0.0)),
            ("fixed-delay", None),
            ("edit-tests", Some(0.0)),
            ("extra", None)
        ]
    );
    assert!(
        pkg.controls
            .iter()
            .all(|c| dir.path().join(&c.script).is_file())
    );
}

#[test]
fn judged_rubrics_are_kept_and_unsupported_judges_refused() {
    let rubric = r#"{"$schema": "https://task.md/schema/rubric-1.json", "criteria": [
      {"id": "works", "gate": true, "judge": "test", "check": "test_works", "text": "It works.", "stated": "Fix it"},
      {"id": "method", "points": 2, "judge": "agent", "evidence": ["report.md", "diff:app.py"], "text": "Sound.", "stated": "Fix it"},
      {"id": "fig", "points": 1, "judge": "vlm", "evidence": ["fig.png"], "text": "F.", "stated": "Fix it"},
      {"id": "look", "points": 1, "judge": "human", "text": "L.", "stated": "Fix it"}
    ]}"#;
    let (_d, pkg) = package(
        "[verifier.judges.agent]\neffort = \"high\"\nbudget = { tokens = \"1M\" }\n",
        &[
            ("verifier/rubric.json", rubric),
            ("verifier/behaviors.json", BEHAVIORS),
        ],
        "Fix it.",
    );
    let r = pkg.verifier.rubric.as_ref().unwrap();
    assert!(r.needs_model());
    let got = refused(&pkg);
    for want in [
        "verifier/behaviors.json",
        "verifier/rubric.json fig",
        "verifier/rubric.json look",
        "verifier/rubric.json method evidence diff:app.py",
        "[verifier.judges.agent] effort",
        "[verifier.judges.agent]",
    ] {
        assert!(
            got.iter().any(|(f, _)| *f == want),
            "{want} not refused: {got:#?}"
        );
    }
}

const BEHAVIORS: &str = r#"{"$schema": "https://task.md/schema/behaviors-1.json", "watch": [
  {"id": "process.loop", "definition": "Loops.", "consequence": "report"}
]}"#;

#[test]
fn the_digest_changes_with_any_file() {
    let (dir, pkg) = package("", &[], "Fix it.");
    std::fs::write(dir.path().join("sandbox/data.txt"), "x").unwrap();
    let again = taskmd::check(dir.path(), &CheckOptions { deep: true })
        .package
        .unwrap();
    assert_ne!(pkg.digest, again.digest);
}

#[test]
fn as_0_8_refuses_them() {
    // A separate verifier inherits the sandbox's host list, which its own
    // sandbox's firewall enforces (0.8 refused this).
    let (_d, pkg) = package(
        "[sandbox]\nnetwork = [\"pypi.org\"]\n\n[verifier]\nisolation = \"separate\"\n",
        &[],
        "Fix it.",
    );
    assert!(pkg.unsupported.is_empty());
    assert_eq!(
        pkg.verifier_network(),
        &Network::Allow(vec!["pypi.org".into()])
    );
    // A Compose file nothing declares.
    let (_d, pkg) = package(
        "",
        &[("sandbox/docker-compose.yaml", "services: {}\n")],
        "Fix it.",
    );
    assert_eq!(refused(&pkg), [("sandbox/docker-compose.yaml", false)]);
    // Only ${VAR} is a host variable.
    let (_d, pkg) = package(
        "[sandbox]\nenv = { PRICE = \"${\", KEY = \"${KEY:-x}\" }\n",
        &[],
        "Fix it.",
    );
    assert_eq!(refused(&pkg), [("[sandbox] env.KEY", false)]);
    assert_eq!(pkg.sandbox.env["PRICE"], "${");
}

/// A `[[sandbox.services]]` command string is split as Docker Compose split
/// it under 0.8 (`#hunter2` stays); one Compose refuses is a check error
/// (P014) and refused by name, where 0.8 failed when Compose read its file.
#[test]
fn service_command_strings_split_as_compose_splits_them() {
    let (d, pkg) = package(
        "[[sandbox.services]]\nname = \"cache\"\nimage = \"redis:7\"\ncommand = \"redis-server --save '' --requirepass #hunter2\"\n\n[[sandbox.services]]\nname = \"web\"\nimage = \"python:3\"\ncommand = \"python -c print(1)\"\n",
        &[],
        "Fix it.",
    );
    assert_eq!(pkg.extras.services.len(), 1);
    assert_eq!(
        pkg.extras.services[0].command,
        ["redis-server", "--save", "", "--requirepass", "#hunter2"]
    );
    assert_eq!(refused(&pkg), [("[sandbox.services.web] command", false)]);
    assert!(
        pkg.unsupported[0]
            .reason
            .contains("(a `(` outside quotes does not open a `$(`)"),
        "{}",
        pkg.unsupported[0].reason
    );
    let report = benchflow_task::load::check(d.path(), &CheckOptions { deep: true });
    assert!(!report.ok());
    let errors: Vec<(&str, &str)> = report
        .diagnostics
        .iter()
        .filter(|d| d.is_error())
        .map(|d| (d.code, d.message.as_str()))
        .collect();
    assert_eq!(
        errors,
        [(
            "P014",
            "[sandbox.services.web] command: Docker Compose cannot split the command into words: a `(` outside quotes does not open a `$(` (as Docker Compose v5.4.0 to v5.6.0 split it, with go-shellwords v1.0.14 and v1.0.15; other versions differ on a few strings)"
        )]
    );
}

/// Compose stops reading a command string at a shell operator; BenchFlow
/// runs the words before it, as 0.8 did through Compose, and warns (P013).
#[test]
fn a_service_command_cut_at_a_shell_operator_is_warned_about() {
    let (_d, pkg) = package(
        "[[sandbox.services]]\nname = \"cache\"\nimage = \"redis:7\"\ncommand = \"redis-server --save 60 1 > /dev/null\"\n",
        &[],
        "Fix it.",
    );
    assert!(pkg.unsupported.is_empty(), "{:#?}", pkg.unsupported);
    assert_eq!(
        pkg.extras.services[0].command,
        ["redis-server", "--save", "60", "1"]
    );
    let warned: Vec<(&str, &str)> = pkg
        .extras
        .diagnostics
        .iter()
        .map(|d| (d.code, d.message.as_str()))
        .collect();
    assert_eq!(warned.len(), 1, "{warned:#?}");
    assert_eq!(warned[0].0, "P013");
    assert!(
        warned[0]
            .1
            .starts_with("[sandbox.services.cache] command: Docker Compose stops reading the command at a shell operator and drops \"> /dev/null\""),
        "{}",
        warned[0].1
    );
}

#[test]
fn a_yaml_config_block_is_read() {
    let dir = tempfile::tempdir().unwrap();
    for (rel, text) in [
        ("sandbox/Dockerfile", "FROM scratch\n"),
        ("verifier/test.sh", "pytest\n"),
        (
            "task.md",
            "Fix it.\n\n```yaml task\nname: acme/yaml\nagent:\n  timeout: 10m\n  timout: 1m\n```\n",
        ),
    ] {
        let path = dir.path().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    let report = taskmd::check(dir.path(), &CheckOptions { deep: true });
    let d = report
        .diagnostics
        .iter()
        .find(|d| d.code == "T201")
        .expect("the typo");
    // YAML has no spans: the diagnostic points at the block.
    assert_eq!(d.location.as_ref().unwrap().line, Some(3));
    assert!(d.fix.is_none());
    std::fs::write(
        dir.path().join("task.md"),
        "Fix it.\n\n```yaml task\nname: acme/yaml\nagent:\n  timeout: 10m\n```\n",
    )
    .unwrap();
    let pkg = taskmd::check(dir.path(), &CheckOptions { deep: true })
        .package
        .unwrap();
    assert_eq!(pkg.id, "acme/yaml");
    assert_eq!(pkg.agent.timeout, Some(Duration::from_secs(600)));
}

#[test]
fn an_absolute_skills_path_is_a_folder_in_the_sandbox() {
    let (_d, pkg) = package("[sandbox]\nskills = \"/skills\"\n", &[], "Fix it.");
    assert!(pkg.unsupported.is_empty(), "{:?}", pkg.unsupported);
    assert_eq!(pkg.sandbox.skills, None);
    assert_eq!(pkg.sandbox.skills_dir.as_deref(), Some("/skills"));
}

/// F1 (rc.2): `[sandbox] x-allow-cloud-metadata = true` (an extension key,
/// so the format's own checker accepts it) opts the task out of the refusal
/// of the cloud metadata endpoints; anything but a boolean is refused by
/// name.
#[test]
fn the_cloud_metadata_opt_out_is_read() {
    let (_d, pkg) = package("[sandbox]\nx-allow-cloud-metadata = true\n", &[], "Fix it.");
    assert!(pkg.unsupported.is_empty(), "{:?}", pkg.unsupported);
    assert!(pkg.sandbox.allow_cloud_metadata);
    let (_d, pkg) = package("[sandbox]\nnetwork = \"open\"\n", &[], "Fix it.");
    assert!(!pkg.sandbox.allow_cloud_metadata);
    let (_d, pkg) = package(
        "[sandbox]\nx-allow-cloud-metadata = \"yes\"\n",
        &[],
        "Fix it.",
    );
    assert_eq!(refused(&pkg), [("[sandbox] x-allow-cloud-metadata", false)]);
    assert!(!pkg.sandbox.allow_cloud_metadata);
}
