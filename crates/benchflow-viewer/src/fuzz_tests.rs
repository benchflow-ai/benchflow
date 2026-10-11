//! Property tests over the viewer's readers of outside input: a trial or
//! job folder with one file swapped for edited text or random JSON, read
//! into payloads, pages and every API route; and the `json.loads` port on
//! any text. None may panic or take too long.

#[path = "../../benchflow-task/tests/fuzz_support/mod.rs"]
mod fuzz_support;

use std::path::{Path, PathBuf};
use std::sync::LazyLock;

use axum::body::{Body, to_bytes};
use axum::http::{Request, header};
use fuzz_support::*;
use proptest::prelude::*;
use tower::ServiceExt;

use crate::{catalog, page, payload, py, pyjson, timestamp};

fn fixture_jobs() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jobs")
}

/// Trial folders among the fixtures.
static TRIALS: LazyLock<Vec<PathBuf>> = LazyLock::new(|| {
    let mut out: Vec<PathBuf> = files(&fixture_jobs(), &|p: &Path| {
        p.ends_with("trajectory/acp_trajectory.jsonl")
    })
    .into_iter()
    .filter_map(|(p, _)| p.parent().and_then(Path::parent).map(Path::to_path_buf))
    .collect();
    out.sort();
    out
});

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap().flatten() {
        let p = e.path();
        let t = to.join(e.file_name());
        if p.is_dir() {
            copy_dir(&p, &t);
        } else {
            std::fs::copy(&p, &t).unwrap();
        }
    }
}

const KEYS: &[&str] = &[
    "task_name",
    "rollout_name",
    "rewards",
    "reward",
    "agent",
    "agent_name",
    "model",
    "error",
    "started_at",
    "finished_at",
    "timing",
    "trajectory_summary",
    "verifier",
    "tests",
    "status",
    "name",
    "results",
    "summary",
    "sessionUpdate",
    "update",
    "type",
    "text",
    "content",
    "toolCallId",
    "tool_call_id",
    "title",
    "kind",
    "rawInput",
    "rawOutput",
    "ts",
    "event",
    "first_observed_ns",
    "last_observed_ns",
    "message",
    "role",
    "usage",
    "criteria",
    "verdict",
    "score",
    "points",
    "lineage",
    "parent",
    "children",
    "branch",
    "ref",
    "prompts",
    "scenes",
    "loop",
    "n_tool_calls",
    "duration_sec",
    "agent_execution",
    "verifier_execution",
    "environment_setup",
    "plan",
    "entries",
];

fn get(router: &axum::Router, uri: &str) {
    let Ok(request) = Request::get(uri)
        .header(header::HOST, "127.0.0.1:8765")
        .body(Body::empty())
    else {
        return;
    };
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let Ok(response) = router.clone().oneshot(request).await;
        let _ = to_bytes(response.into_body(), 64 << 20).await;
    });
}

proptest! {
    #![proptest_config(config_slow())]

    /// A fixture job with one file swapped, viewed through payloads, pages
    /// and the HTTP routes with random query values.
    #[test]
    fn job_folder_views(
        pick in any::<prop::sample::Index>(),
        target in any::<prop::sample::Index>(),
        as_text in any::<bool>(),
        doc in json_with(KEYS),
        text in mutated(seeds(&["benchflow-viewer/tests/fixtures/jobs"], &|n| n.ends_with(".json") || n.ends_with(".jsonl") || n.ends_with(".txt"), 300)),
        query in prop_oneof!["\\PC{0,30}", proptest::sample::select(TOKENS).prop_map(str::to_string)],
    ) {
        let trial = pick.get(&TRIALS);
        let tmp = tempfile::tempdir().unwrap();
        let job = tmp.path().join("job");
        copy_dir(trial.parent().unwrap(), &job);
        let records: Vec<PathBuf> = files(&job, &|p: &Path| {
            p.extension().is_some_and(|e| e == "json" || e == "jsonl" || e == "txt")
        }).into_iter().map(|(p, _)| p).collect();
        if !records.is_empty() {
            let file = target.get(&records);
            std::fs::write(file, if as_text { text.clone() } else { doc.to_string() }).unwrap();
        }
        let q: String = query.bytes().map(|b| format!("%{b:02X}")).collect();
        timed("viewer", move || {
            let runs = catalog::discover(tmp.path(), 1000);
            let _ = page::browse_boot(&runs, false);
            for run in &runs {
                let _ = payload::build_payload(&run.dir);
                let _ = page::single_boot(&run.dir);
                let _ = page::render_static(&run.dir);
                let _ = payload::build_branch_child_payload(&run.dir, &query);
            }
            if let Ok(target) = crate::detect(tmp.path()) {
                let router = crate::app(target, 8765);
                let id = runs.first().map(|r| r.id.clone()).unwrap_or_default();
                for uri in [
                    "/".to_string(),
                    format!("/api/rollout?id={id}"),
                    format!("/api/rollout?id={q}"),
                    format!("/api/compare?a={id}&b={q}"),
                    format!("/api/outcomes?filter={q}"),
                    "/compare".to_string(),
                    "/outcomes".to_string(),
                ] {
                    get(&router, &uri);
                }
            }
        });
    }
}

proptest! {
    #![proptest_config(config())]

    /// The `json.loads` port reads any text, and what it reads writes back
    /// to text that reads the same.
    #[test]
    fn pyjson_round_trip(text in prop_oneof![
        json_with(KEYS).prop_map(|v| v.to_string()),
        mutated(seeds(&["benchflow-viewer/tests/fixtures/jobs"], &|n| n.ends_with(".json"), 100)),
        "\\PC{0,64}",
    ]) {
        timed("pyjson", move || {
            if let Some(v) = pyjson::parse(&text) {
                let once = pyjson::dumps(&v);
                let again = pyjson::parse(&once).expect("dumps output parses");
                prop_assert_eq!(pyjson::dumps(&again), once);
                let _ = timestamp::parse_ts(&v);
            }
            let _ = py::parse_float(&text);
            let _ = py::parse_int(&text);
            Ok(())
        })?;
    }
}

/// The viewer's float formatting against Python's `repr(float)` on the
/// 9,111 floats (4,000 of them ties) of the agent crate's fixture (0.8's
/// Python 3.12).
#[test]
fn float_repr_matches_python() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../benchflow-agent/tests/fixtures/python-float-repr.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    for line in text.lines() {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let value = f64::from_bits(case["bits"].as_u64().unwrap());
        assert_eq!(
            pyjson::float_repr(value),
            case["repr"].as_str().unwrap(),
            "{value:e}"
        );
    }
}
