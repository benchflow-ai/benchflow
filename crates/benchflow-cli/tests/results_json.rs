//! `bench results --json` carries what the text's warning lines say about
//! each trial's defenses, so a script can read it (G-17): the trials whose
//! model agent could reach the cloud metadata endpoints, and those whose
//! model agent ran without the guard on the sandbox daemon's ports, each
//! with its record's state and note. The JSON is read as untyped values.
//! Nothing here starts a sandbox.

use std::path::Path;

use assert_cmd::Command;
use serde_json::{Value, json};

fn bench(dir: &Path) -> Command {
    let mut c = Command::cargo_bin("bench").unwrap();
    c.current_dir(dir)
        .env("BENCHFLOW_CACHE_DIR", dir.join("cache"))
        .env("NO_COLOR", "1")
        .env_remove("DAYTONA_API_KEY");
    c
}

fn results_json(dir: &Path, args: &[&str]) -> Value {
    let out = bench(dir)
        .args(["results", "--json"])
        .args(args)
        .output()
        .unwrap();
    assert!(out.status.success(), "{out:?}");
    serde_json::from_slice(&out.stdout).unwrap()
}

const OPTED: &str = "the task opts out ([metadata] x-allow-cloud-metadata = true), so the agent can reach the cloud metadata endpoints";
const UNGUARDED: &str = "the task's image has no iptables, and BenchFlow could not install it, so the agent can reach the sandbox daemon's ports (2280, 22220, 22222, 33333)";

/// A defense record's `(state, note)`.
type Record<'a> = (&'a str, &'a str);

/// A job whose trials are `(agent, status, cloud_metadata, root_service)`.
fn write_job(root: &Path, name: &str, trials: &[(&str, &str, Record<'_>, Record<'_>)]) {
    let job = root.join(format!("jobs/{name}"));
    std::fs::create_dir_all(&job).unwrap();
    std::fs::write(
        job.join("evaluation.json"),
        json!({ "job_name": name }).to_string(),
    )
    .unwrap();
    for (i, (agent, status, meta, root_service)) in trials.iter().enumerate() {
        let dir = job.join(format!("t{i}__0000000{i}"));
        std::fs::create_dir_all(&dir).unwrap();
        let mut result = json!({
            "task_name": format!("t{i}"), "rollout_name": format!("t{i}__0000000{i}"),
            "agent": agent, "model": "mimo/mimo-v2.6-flash", "status": status,
            "passed": *status == "scored", "rewards": {"reward": 1.0},
            "timing": {"agent_execution": 12.5},
            "defenses": {
                "cloud_metadata": {"name": "cloud_metadata", "state": meta.0, "note": meta.1},
                "root_service": {"name": "root_service", "state": root_service.0, "note": root_service.1},
            },
        });
        if *status == "errored" {
            result["rewards"] = Value::Null;
            result["error"] = json!("R214 the model provider refused every request");
        }
        std::fs::write(dir.join("result.json"), result.to_string()).unwrap();
    }
}

#[test]
fn results_json_names_the_trials_whose_model_agent_was_exposed() {
    let d = tempfile::tempdir().unwrap();
    let enforced = ("enforced", "refused");
    write_job(
        d.path(),
        "j",
        &[
            // Opted out, on Daytona without the guard.
            (
                "opencode",
                "scored",
                ("off", OPTED),
                ("not-supported", UNGUARDED),
            ),
            // Both held.
            ("opencode", "scored", enforced, enforced),
            // The oracle is no model agent.
            (
                "oracle",
                "scored",
                ("off", OPTED),
                ("not-supported", UNGUARDED),
            ),
            // A model agent that ran and then errored is named too.
            ("opencode", "errored", ("off", OPTED), enforced),
        ],
    );
    write_job(d.path(), "k", &[("opencode", "scored", enforced, enforced)]);

    let one = results_json(d.path(), &["jobs/j"]);
    // Every trial's own records, as result.json has them.
    for t in one["trials"].as_array().unwrap() {
        assert!(t["defenses"]["cloud_metadata"]["state"].is_string(), "{t}");
        assert!(t["defenses"]["root_service"]["state"].is_string(), "{t}");
    }
    assert_eq!(
        one["cloud_metadata_reachable"],
        json!([
            {"trial": "t0__00000000", "state": "off", "note": OPTED},
            {"trial": "t3__00000003", "state": "off", "note": OPTED},
        ]),
        "{one:#}"
    );
    assert_eq!(
        one["root_service_unguarded"],
        json!([{"trial": "t0__00000000", "state": "not-supported", "note": UNGUARDED}]),
        "{one:#}"
    );

    // The job list and a comparison carry the same, per job.
    let many = results_json(d.path(), &[]);
    let jobs = many.as_array().unwrap();
    assert_eq!(jobs.len(), 2, "{many:#}");
    for job in jobs {
        let exposed = job["dir"].as_str().unwrap().ends_with('j');
        let n = job["cloud_metadata_reachable"].as_array().unwrap().len();
        assert_eq!(n, if exposed { 2 } else { 0 }, "{job:#}");
        let n = job["root_service_unguarded"].as_array().unwrap().len();
        assert_eq!(n, usize::from(exposed), "{job:#}");
    }
    let compared = results_json(d.path(), &["jobs/j", "jobs/k"]);
    assert_eq!(
        compared["jobs"][0]["cloud_metadata_reachable"],
        one["cloud_metadata_reachable"]
    );
    assert_eq!(compared["jobs"][1]["root_service_unguarded"], json!([]));
}
