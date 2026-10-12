//! The compare, outcomes, and Pareto views over HTTP: a job the Rust engine wrote
//! beside one BenchFlow 0.8 wrote, straight to the router (no network)
//! except where `serve_jobs` starts a real server on the loopback interface.

use std::path::Path;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, StatusCode, header};
use benchflow_viewer::{ServeOptions, Target, app, detect, serve_jobs};
use serde_json::Value;
use tower::ServiceExt;

const PORT: u16 = 8765;
const OLD: &str = "old-job/2026-01-01__00-00-00";

fn write_trial(dir: &Path, result: &str, timing: Option<&str>) {
    std::fs::create_dir_all(dir.join("trajectory")).unwrap();
    std::fs::write(
        dir.join("trajectory/acp_trajectory.jsonl"),
        r#"{"type": "agent_message", "text": "done"}"#,
    )
    .unwrap();
    std::fs::write(dir.join("result.json"), result).unwrap();
    if let Some(timing) = timing {
        std::fs::write(dir.join("timing.json"), timing).unwrap();
    }
}

fn rust_result(
    task: &str,
    status: &str,
    passed: Option<bool>,
    reward: Option<f64>,
    problem: &str,
) -> String {
    format!(
        r#"{{"task_name": "{task}", "agent": "claude", "agent_name": "claude", "model": "opus",
            "rewards": {}, "status": "{status}", "passed": {}, "problem": {problem},
            "error": null, "verifier_error": null}}"#,
        reward.map_or("null".into(), |reward| format!(
            "{{\"reward\": {reward:?}}}"
        )),
        passed.map_or("null".into(), |passed| passed.to_string()),
    )
}

/// `jobs/rust-job` (written by the Rust engine) and `jobs/old-job/<time>`
/// (written by 0.8), over the same four tasks.
fn corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let rust = dir.path().join("jobs/rust-job");
    let timing = |setup: f64, agent: f64| {
        format!(
            r#"{{"environment_setup": {setup:?}, "agent_execution": {agent:?}, "verifier": 1.0, "total": 99.0}}"#
        )
    };
    write_trial(
        &rust.join("alpha__0001"),
        &rust_result("alpha", "scored", Some(true), Some(1.0), "null"),
        Some(&timing(2.0, 10.0)),
    );
    write_trial(
        &rust.join("beta__0001"),
        &rust_result("beta", "scored", Some(false), Some(0.0), "null"),
        Some(&timing(4.0, 20.0)),
    );
    write_trial(
        &rust.join("gamma__0001"),
        &rust_result(
            "gamma",
            "unscored",
            None,
            None,
            r#"{"code": "R304", "owner": "task", "phase": "verify", "message": "the verifier wrote no reward", "next": "fix the task's test.sh", "transient": false}"#,
        ),
        Some(&timing(6.0, 30.0)),
    );
    write_trial(
        &rust.join("delta__0001"),
        &rust_result(
            "delta",
            "errored",
            None,
            None,
            r#"{"code": "R201", "owner": "sandbox", "phase": "start", "message": "the sandbox did not start", "next": "check the backend", "transient": true}"#,
        ),
        None,
    );
    let old = dir.path().join("jobs").join(OLD);
    for (name, result) in [
        (
            "alpha__0002",
            r#"{"task_name": "alpha", "agent_name": "codex", "rewards": {"reward": 1.0}, "error": null}"#,
        ),
        (
            "beta__0002",
            r#"{"task_name": "beta", "agent_name": "codex", "rewards": {"reward": 1.0}, "error": null}"#,
        ),
        (
            "gamma__0002",
            r#"{"task_name": "gamma", "agent_name": "codex", "rewards": {"reward": 1.0}, "error": null}"#,
        ),
        (
            "delta__0002",
            r#"{"task_name": "delta", "agent_name": "codex", "rewards": null, "error": "RuntimeError: ACP error -32603: Internal error", "error_category": "acp_error",
                "timing": {"environment_setup": 5.0, "total": 6.0}}"#,
        ),
    ] {
        write_trial(&old.join(name), result, None);
    }
    dir
}

fn router(dir: &Path) -> Router {
    app(detect(&dir.join("jobs")).unwrap(), PORT)
}

async fn send(
    router: &Router,
    method: Method,
    uri: &str,
    host: &str,
) -> (StatusCode, String, String) {
    let request = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::HOST, host)
        .body(Body::empty())
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|value| value.to_str().unwrap().to_string())
        .unwrap_or_default();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (
        status,
        content_type,
        String::from_utf8(body.to_vec()).unwrap(),
    )
}

async fn api(router: &Router, uri: &str) -> Value {
    let (status, content_type, body) = send(router, Method::GET, uri, "localhost:8765").await;
    assert_eq!(status, StatusCode::OK, "{uri}");
    assert_eq!(content_type, "application/json; charset=utf-8");
    serde_json::from_str(&body).unwrap()
}

fn boot(page: &str) -> Value {
    let data = page
        .split("type=\"application/json\">")
        .nth(1)
        .unwrap()
        .split("</script>")
        .next()
        .unwrap();
    serde_json::from_str(data).unwrap()
}

fn tasks(view: &Value) -> Vec<&str> {
    view["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["task"].as_str().unwrap())
        .collect()
}

fn close(value: &Value, expected: f64) {
    let actual = value.as_f64().unwrap();
    assert!(
        (actual - expected).abs() < 1e-9,
        "{actual} is not {expected}"
    );
}

#[tokio::test]
async fn compare_puts_tasks_by_job_with_pass_rates_and_a_paired_difference() {
    let dir = corpus();
    let router = router(dir.path());
    let view = api(&router, &format!("/api/compare?runs=rust-job,{OLD}")).await;
    assert_eq!(view["mode"], "compare");

    let columns = view["columns"].as_array().unwrap();
    assert_eq!(columns.len(), 2);
    assert_eq!(
        (columns[0]["job"].as_str(), columns[0]["arm"].as_str()),
        (Some("rust-job"), Some("claude \u{b7} opus"))
    );
    assert_eq!(columns[1]["arm"], "codex");
    assert_eq!(
        columns[0]["counts"],
        serde_json::json!({"passed": 1, "failed": 1, "unscored": 1, "errored": 1})
    );
    assert_eq!(
        columns[1]["counts"],
        serde_json::json!({"passed": 3, "failed": 0, "unscored": 0, "errored": 1})
    );
    close(&columns[0]["pass_rate"], 0.25);
    // Wilson 95% for 1 of 4 and 3 of 4.
    close(&columns[0]["pass_interval"][0], 0.045_587_260_809_700_55);
    close(&columns[0]["pass_interval"][1], 0.699_358_157_417_598_1);
    close(&columns[1]["pass_interval"][0], 0.300_641_842_582_401_84);

    // beta disagrees on reward, gamma on outcome (unscored vs passed).
    assert_eq!(tasks(&view), ["beta", "gamma", "alpha", "delta"]);
    let beta = &view["rows"][0]["cells"];
    assert_eq!(beta[0][0]["id"], "rust-job/beta__0001");
    assert_eq!(beta[0][0]["outcome"], "failed");
    assert_eq!(beta[1][0]["id"], format!("{OLD}/beta__0002"));

    // Shared scored tasks: alpha (1 - 1) and beta (0 - 1).
    let pair = &view["pairs"][0];
    assert_eq!((pair["a"].as_u64(), pair["b"].as_u64()), (Some(0), Some(1)));
    assert_eq!(pair["shared"], 2);
    close(&pair["mean_difference"], -0.5);
    // Two shared tasks give no interval (with one, t(0.975, 1) = 12.7
    // made it -6.853 to +5.853 for rewards that differ by at most 1).
    assert!(pair["interval"].is_null(), "{pair}");
    assert_eq!(pair["no_interval"], "too few tasks");
    assert_eq!(
        [
            &pair["better"],
            &pair["worse"],
            &pair["tied"],
            &pair["excluded"],
            &pair["one_side"]
        ],
        [0, 1, 1, 2, 0].map(Value::from).each_ref()
    );
    assert_eq!(view["query"]["runs"], serde_json::json!(["rust-job", OLD]));
    assert_eq!(view["unknown"], serde_json::json!([]));
}

/// `runs=` takes a job as typed in the shell (`jobs/rust-job`, the form the
/// docs show) as well as its id under the folder served.
#[tokio::test]
async fn compare_takes_jobs_as_paths_from_the_shell() {
    let dir = corpus();
    let router = router(dir.path());
    let view = api(
        &router,
        &format!("/api/compare?runs=jobs/rust-job,./jobs/{OLD}/"),
    )
    .await;
    assert_eq!(view["query"]["runs"], serde_json::json!(["rust-job", OLD]));
    assert_eq!(view["unknown"], serde_json::json!([]));
    let view = api(&router, "/api/compare?runs=jobs/nope").await;
    assert_eq!(view["unknown"], serde_json::json!(["jobs/nope"]));
}

/// Delta review B3: a drained job is not shown as complete. Its lock.json
/// planned 3 trials and 1 ran, so /compare marks the column with 2 never
/// ran and partial, and /outcomes says so per job.
#[tokio::test]
async fn compare_and_outcomes_say_how_many_planned_trials_never_ran() {
    let dir = tempfile::tempdir().unwrap();
    let job = dir.path().join("jobs/drained");
    let trial = job.join("a__1");
    std::fs::create_dir_all(trial.join("trajectory")).unwrap();
    std::fs::write(trial.join("trajectory/acp_trajectory.jsonl"), "").unwrap();
    std::fs::write(
        trial.join("result.json"),
        r#"{"task_name": "a", "agent": "x", "status": "scored", "passed": true, "rewards": {"reward": 1.0}}"#,
    )
    .unwrap();
    std::fs::write(
        job.join("lock.json"),
        r#"{"trials": [{"trial": "a__1"}, {"trial": "b__1"}, {"trial": "c__1"}]}"#,
    )
    .unwrap();
    let router = router(dir.path());
    let view = api(&router, "/api/compare").await;
    let column = &view["columns"][0];
    assert_eq!(column["never_ran"], 2, "{view}");
    assert_eq!(column["partial"], true);
    assert_eq!(view["never_ran"]["drained"], 2);
    let outcomes = api(&router, "/api/outcomes").await;
    assert_eq!(outcomes["never_ran"]["drained"], 2, "{outcomes}");
}

/// Final review D2D-1: a drained job with two agents. Its lock.json planned
/// two trials of each, and only `codex`'s second never ran, so on /compare
/// and /pareto only codex's column and point count it and are partial (the
/// rate then has no interval on the page); claude's are whole. On /outcomes the job's
/// row carries the never-ran count and `partial`, which the page shows by
/// the job's name.
#[tokio::test]
async fn never_ran_trials_count_toward_their_own_arm_and_mark_outcomes_rows() {
    let dir = tempfile::tempdir().unwrap();
    let job = dir.path().join("jobs/drained");
    let ran = |name: &str, agent: &str| {
        let trial = job.join(name);
        std::fs::create_dir_all(trial.join("trajectory")).unwrap();
        std::fs::write(trial.join("trajectory/acp_trajectory.jsonl"), "").unwrap();
        std::fs::write(
            trial.join("result.json"),
            format!(
                r#"{{"task_name": "{}", "agent": "{agent}", "model": "m1", "status": "scored", "passed": true, "rewards": {{"reward": 1.0}}}}"#,
                &name[..1]
            ),
        )
        .unwrap();
    };
    ran("a__1", "claude");
    ran("b__1", "claude");
    ran("a__2", "codex");
    std::fs::write(
        job.join("lock.json"),
        r#"{"trials": [
            {"trial": "a__1", "attempt": {"agent": "claude", "model": "m1"}},
            {"trial": "b__1", "attempt": {"agent": "claude", "model": "m1"}},
            {"trial": "a__2", "attempt": {"agent": "codex", "model": "m1"}},
            {"trial": "b__2", "attempt": {"agent": "codex", "model": "m1"}}]}"#,
    )
    .unwrap();
    let router = router(dir.path());
    let view = api(&router, "/api/compare").await;
    let columns = view["columns"].as_array().unwrap();
    let column = |arm: &str| {
        columns
            .iter()
            .find(|c| c["arm"] == arm)
            .unwrap_or_else(|| panic!("no column {arm}: {view}"))
    };
    assert_eq!(column("claude \u{b7} m1")["never_ran"], 0, "{view}");
    assert_eq!(column("claude \u{b7} m1")["partial"], false, "{view}");
    assert_eq!(column("codex \u{b7} m1")["never_ran"], 1, "{view}");
    assert_eq!(column("codex \u{b7} m1")["partial"], true, "{view}");
    assert_eq!(view["never_ran"]["drained"], 1);

    let pareto = api(&router, "/api/pareto").await;
    let point = |arm: &str| {
        pareto["points"]
            .as_array()
            .unwrap()
            .iter()
            .find(|p| p["arm"] == arm)
            .unwrap_or_else(|| panic!("no point {arm}: {pareto}"))
            .clone()
    };
    assert_eq!(point("claude \u{b7} m1")["partial"], false, "{pareto}");
    assert_eq!(point("codex \u{b7} m1")["never_ran"], 1, "{pareto}");
    assert_eq!(point("codex \u{b7} m1")["partial"], true, "{pareto}");

    let outcomes = api(&router, "/api/outcomes").await;
    let row = &outcomes["summary"][0];
    assert_eq!(row["job"], "drained", "{outcomes}");
    assert_eq!(row["never_ran"], 1, "{outcomes}");
    assert_eq!(row["partial"], true, "{outcomes}");

    // The pages carry the same data, and their script reads it.
    let (_, _, page) = send(&router, Method::GET, "/outcomes", "localhost:8765").await;
    assert_eq!(boot(&page)["summary"][0]["never_ran"], 1);
    assert!(page.contains("job.never_ran"));
    let (_, _, page) = send(&router, Method::GET, "/compare", "localhost:8765").await;
    assert!(page.contains("column.partial"));
}

#[tokio::test]
async fn filters_in_the_query_choose_jobs_trials_and_rows() {
    let dir = corpus();
    let router = router(dir.path());
    let both = format!("runs=rust-job,{OLD}");

    let failed = api(&router, &format!("/api/compare?{both}&status=failed")).await;
    assert_eq!(tasks(&failed), ["beta"]);
    assert_eq!(failed["query"]["status"], serde_json::json!(["failed"]));

    let named = api(&router, &format!("/api/compare?{both}&task=ALP")).await;
    assert_eq!(tasks(&named), ["alpha"]);

    let one_agent = api(&router, &format!("/api/compare?{both}&agent=codex")).await;
    let columns = one_agent["columns"].as_array().unwrap();
    assert_eq!(columns[0]["trials"], 0);
    assert_eq!(columns[1]["trials"], 4);
    assert_eq!(one_agent["agents"], serde_json::json!(["claude", "codex"]));

    // No runs means every job; an unknown one is reported, not an error.
    let all = api(&router, "/api/compare?runs=rust-job,nope").await;
    assert_eq!(all["query"]["runs"], serde_json::json!(["rust-job"]));
    assert_eq!(all["unknown"], serde_json::json!(["nope"]));
    let every = api(&router, "/api/compare").await;
    assert_eq!(
        every["query"]["runs"],
        serde_json::json!(["old-job/2026-01-01__00-00-00", "rust-job"])
    );
    assert_eq!(
        every["jobs"][1],
        serde_json::json!({"id": "rust-job", "name": "rust-job", "trials": 4})
    );
    // An id outside the scan is never read, even when the folder exists.
    let outside = api(&router, "/api/compare?runs=../jobs/rust-job").await;
    assert_eq!(outside["unknown"], serde_json::json!(["../jobs/rust-job"]));
    assert_eq!(outside["rows"], serde_json::json!([]));
}

#[tokio::test]
async fn outcomes_count_trials_group_problems_and_time_phases() {
    let dir = corpus();
    let router = router(dir.path());
    let view = api(&router, &format!("/api/outcomes?run=rust-job,{OLD}")).await;
    assert_eq!(view["mode"], "outcomes");
    let summary = view["summary"].as_array().unwrap();
    assert_eq!(summary[0]["trials"], 4);
    assert_eq!(
        summary[1]["counts"],
        serde_json::json!({"passed": 3, "failed": 0, "unscored": 0, "errored": 1})
    );

    let problems: Vec<(String, Value, String, Value)> = view["problems"]
        .as_array()
        .unwrap()
        .iter()
        .map(|problem| {
            (
                problem["code"].as_str().unwrap().to_string(),
                problem["owner"].clone(),
                problem["source"].as_str().unwrap().to_string(),
                problem["by_job"].clone(),
            )
        })
        .collect();
    assert_eq!(
        problems,
        [
            (
                "R201".into(),
                "sandbox".into(),
                "problem".into(),
                serde_json::json!({"rust-job": 1, OLD: 0})
            ),
            (
                "R304".into(),
                "task".into(),
                "problem".into(),
                serde_json::json!({"rust-job": 1, OLD: 0})
            ),
            (
                "acp_error".into(),
                Value::Null,
                "error".into(),
                serde_json::json!({"rust-job": 0, OLD: 1})
            ),
        ]
    );
    let r304 = &view["problems"][1];
    assert_eq!(r304["message"], "the verifier wrote no reward");
    assert_eq!(r304["next"], "fix the task's test.sh");
    assert_eq!(r304["trials"][0]["id"], "rust-job/gamma__0001");
    assert_eq!(view["owners"].as_array().unwrap().len(), 3);

    // timing.json first, else the timing in result.json; missing phases skipped.
    let rust = &view["timing"][0]["phases"];
    assert_eq!(
        rust["environment_setup"],
        serde_json::json!({"n": 3, "median": 4.0, "p90": 5.6})
    );
    close(&rust["agent_execution"]["p90"], 28.0);
    let old = &view["timing"][1]["phases"];
    assert_eq!(old["environment_setup"]["n"], 1);
    assert_eq!(
        old["agent_execution"],
        serde_json::json!({"n": 0, "median": null, "p90": null})
    );

    let errored = api(&router, "/api/outcomes?status=errored").await;
    let counts: Vec<&Value> = errored["summary"]
        .as_array()
        .unwrap()
        .iter()
        .map(|job| &job["trials"])
        .collect();
    assert_eq!(counts, [1, 1].map(Value::from).each_ref());
}

#[tokio::test]
async fn the_pages_carry_their_data_and_link_to_each_other() {
    let dir = corpus();
    let router = router(dir.path());
    let query = format!("?runs=rust-job,{OLD}&status=failed");
    let (status, content_type, page) = send(
        &router,
        Method::GET,
        &format!("/compare{query}"),
        "localhost:8765",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "text/html; charset=utf-8");
    // The page's data is what the API answers for the same query.
    assert_eq!(
        boot(&page),
        api(&router, &format!("/api/compare{query}")).await
    );
    assert!(page.contains("<title>compare - jobs</title>"));
    assert!(page.contains("<a href=\"/compare\" aria-current=\"page\""));
    assert!(page.contains("<a href=\"/outcomes\""));
    assert!(page.contains("BF.views = "));

    let (_, _, page) = send(
        &router,
        Method::GET,
        "/outcomes?run=rust-job",
        "localhost:8765",
    )
    .await;
    assert_eq!(boot(&page)["mode"], "outcomes");
    assert!(page.contains("<a href=\"/outcomes\" aria-current=\"page\""));

    // The run list links to both views and is otherwise 0.8's page.
    let (_, _, list) = send(&router, Method::GET, "/", "localhost:8765").await;
    assert!(list.contains("<a href=\"/\" aria-current=\"page\""));
    assert!(list.contains("<a href=\"/compare\""));
    assert_eq!(boot(&list)["mode"], "browse");
}

/// The corpus plus two jobs that record cost: `priced-job` passes both
/// trials at $0.40 a trial, `cheap-job` one of two at $0.10.
fn priced_corpus() -> tempfile::TempDir {
    let dir = corpus();
    let trial = |job: &str, name: &str, passed: bool, cost: f64, tokens: &str, total: f64| {
        let reward = if passed { 1.0 } else { 0.0 };
        write_trial(
            &dir.path().join("jobs").join(job).join(name),
            &format!(
                r#"{{"task_name": "{}", "agent": "codex", "model": "deepseek", "status": "scored",
                    "passed": {passed}, "rewards": {{"reward": {reward:?}}}, "problem": null,
                    "agent_result": {{"cost_usd": {cost:?}, {tokens}}}}}"#,
                name.split("__").next().unwrap()
            ),
            Some(&format!(
                r#"{{"environment_setup": 1.0, "agent_execution": 1.0, "verifier": 1.0, "total": {total:?}}}"#
            )),
        );
    };
    let acp = |n: u64| format!(r#""total_tokens": {n}, "usage_source": "acp""#);
    let none = r#""total_tokens": 0, "usage_source": "unavailable""#;
    trial("priced-job", "alpha__0003", true, 0.5, &acp(1000), 50.0);
    trial("priced-job", "beta__0003", true, 0.3, &acp(3000), 70.0);
    trial("cheap-job", "alpha__0004", true, 0.1, none, 10.0);
    trial("cheap-job", "beta__0004", false, 0.1, none, 30.0);
    dir
}

fn point<'a>(view: &'a Value, job: &str) -> &'a Value {
    view["points"]
        .as_array()
        .unwrap()
        .iter()
        .find(|point| point["job_name"] == job)
        .unwrap_or_else(|| panic!("no point for {job}: {view}"))
}

#[tokio::test]
async fn pareto_puts_pass_rate_against_cost_time_and_tokens_and_marks_the_frontier() {
    let dir = priced_corpus();
    let router = router(dir.path());
    let view = api(&router, "/api/pareto").await;
    assert_eq!(view["mode"], "pareto");
    assert_eq!(view["points"].as_array().unwrap().len(), 4);
    assert_eq!(
        view["same_tasks"], false,
        "the priced jobs ran two of the four tasks"
    );
    let same = api(&router, &format!("/api/pareto?runs=rust-job,{OLD}")).await;
    assert_eq!(same["same_tasks"], true);

    let priced = point(&view, "priced-job");
    assert_eq!(
        (priced["trials"].as_u64(), priced["passed"].as_u64()),
        (Some(2), Some(2))
    );
    close(&priced["pass_rate"], 1.0);
    close(&priced["cost_per_trial"], 0.4);
    close(&priced["median_seconds"], 60.0);
    close(&priced["tokens_per_trial"], 2000.0);
    assert_eq!(priced["arm"], "codex \u{b7} deepseek");
    assert_eq!(
        priced["frontier"],
        serde_json::json!({"cost_per_trial": true, "median_seconds": true, "tokens_per_trial": true})
    );

    // Cheaper but passing less often: on the cost frontier, not the time one
    // (the 0.8 job passes more often in less time).
    let cheap = point(&view, "cheap-job");
    close(&cheap["pass_rate"], 0.5);
    close(&cheap["cost_per_trial"], 0.1);
    close(&cheap["median_seconds"], 20.0);
    assert_eq!(
        cheap["tokens_per_trial"],
        Value::Null,
        "usage was unavailable"
    );
    assert_eq!(cheap["tokens_per_trial_n"], 0);
    assert_eq!(
        cheap["frontier"],
        serde_json::json!({"cost_per_trial": true, "median_seconds": false, "tokens_per_trial": null})
    );

    // No cost recorded: off every cost frontier, still on the time one.
    let old = point(&view, OLD);
    close(&old["pass_rate"], 0.75);
    assert_eq!(old["cost_per_trial"], Value::Null);
    close(&old["median_seconds"], 6.0);
    assert_eq!(old["median_seconds_n"], 1);
    assert_eq!(old["frontier"]["median_seconds"], true);
    let rust = point(&view, "rust-job");
    close(&rust["pass_rate"], 0.25);
    let interval = rust["pass_interval"].as_array().unwrap();
    assert!(interval[0].as_f64().unwrap() < 0.25 && interval[1].as_f64().unwrap() > 0.25);
    assert_eq!(
        rust["median_seconds_n"], 3,
        "the errored trial recorded no time"
    );
    assert_eq!(rust["frontier"]["median_seconds"], false);

    // Pass rate needs every outcome, so a status filter changes nothing;
    // the job and agent filters choose points.
    assert_eq!(
        api(&router, "/api/pareto?status=failed").await["points"],
        view["points"]
    );
    let two = api(&router, "/api/pareto?runs=cheap-job,rust-job").await;
    assert_eq!(two["points"].as_array().unwrap().len(), 2);
    assert_eq!(point(&two, "cheap-job")["frontier"]["median_seconds"], true);
    let codex = api(&router, "/api/pareto?agent=codex").await;
    assert!(
        codex["points"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p["job_name"] != "rust-job")
    );
}

#[tokio::test]
async fn the_pareto_page_carries_its_data_and_every_view_links_to_it() {
    let dir = priced_corpus();
    let router = router(dir.path());
    let (status, _, page) = send(
        &router,
        Method::GET,
        "/pareto?runs=priced-job,cheap-job",
        "localhost:8765",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        boot(&page),
        api(&router, "/api/pareto?runs=priced-job,cheap-job").await
    );
    assert!(page.contains("<a href=\"/pareto\" aria-current=\"page\""));
    assert!(page.contains("<span class=\"app\">pareto</span>"));
    for uri in ["/", "/compare", "/outcomes"] {
        let (_, _, other) = send(&router, Method::GET, uri, "localhost:8765").await;
        assert!(other.contains("<a href=\"/pareto\""), "{uri}");
    }
}

#[tokio::test]
async fn hostile_task_names_stay_data() {
    let dir = corpus();
    write_trial(
        &dir.path().join("jobs/rust-job/evil__0001"),
        r#"{"task_name": "</script><script>alert(1)</script>", "rewards": {"reward": 0.0}}"#,
        None,
    );
    let router = router(dir.path());
    let (_, _, page) = send(&router, Method::GET, "/compare", "localhost:8765").await;
    assert!(!page.contains("</script><script>alert(1)"));
    assert!(tasks(&boot(&page)).contains(&"</script><script>alert(1)</script>"));
}

#[tokio::test]
async fn the_views_refuse_what_the_viewer_refuses() {
    let dir = corpus();
    let router = router(dir.path());
    for uri in ["/compare", "/outcomes", "/pareto"] {
        let (status, _, body) = send(&router, Method::HEAD, uri, "localhost:8765").await;
        assert_eq!((status, body.as_str()), (StatusCode::OK, ""), "{uri}");
        assert_eq!(
            send(&router, Method::POST, uri, "localhost:8765").await.0,
            StatusCode::METHOD_NOT_ALLOWED
        );
        assert_eq!(
            send(&router, Method::GET, uri, "evil.example:8765").await.0,
            StatusCode::FORBIDDEN
        );
    }
    for uri in ["/api/compare", "/api/outcomes", "/api/pareto"] {
        assert_eq!(
            send(&router, Method::HEAD, uri, "localhost:8765").await.0,
            StatusCode::METHOD_NOT_ALLOWED
        );
    }
    // A single trial has no jobs to compare.
    let trial = app(
        detect(&dir.path().join("jobs/rust-job/alpha__0001")).unwrap(),
        PORT,
    );
    for uri in [
        "/compare",
        "/outcomes",
        "/pareto",
        "/api/compare",
        "/api/pareto",
    ] {
        assert_eq!(
            send(&trial, Method::GET, uri, "localhost:8765").await.0,
            StatusCode::NOT_FOUND,
            "{uri}"
        );
    }
}

#[tokio::test]
async fn serve_jobs_names_each_job_and_opens_on_compare() {
    let dir = corpus();
    let jobs = dir.path().join("jobs");
    let handle = serve_jobs(
        &[jobs.join("rust-job"), jobs.join("old-job")],
        ServeOptions {
            port: Some(0),
            ..ServeOptions::default()
        },
    )
    .await
    .unwrap();
    let Target::Jobs {
        jobs: named,
        runs,
        capped,
    } = &handle.target
    else {
        panic!("{:?}", handle.target);
    };
    assert_eq!(
        named
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        ["rust-job", "old-job"]
    );
    assert_eq!((*runs, *capped), (8, false));
    assert_eq!(handle.target.start_path(), "/compare");

    let router = app(handle.target.clone(), PORT);
    let view = api(&router, "/api/compare").await;
    assert_eq!(view["root"], "rust-job vs old-job");
    assert_eq!(view["query"]["runs"], serde_json::json!(["rust-job", OLD]));
    // Run ids carry the job's name, and open in the run page.
    let id = view["rows"][0]["cells"][1][0]["id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(id, format!("{OLD}/beta__0002"));
    let (status, _, payload) = send(
        &router,
        Method::GET,
        &format!("/api/rollout?id={id}"),
        "localhost:8765",
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(payload.contains("\"steps\""));
    handle.stop().await.unwrap();
}
