//! The viewer's HTTP contract: routes, refusals, and headers. Requests go
//! straight to the router (no network) except in the one test that starts
//! a real server on the loopback interface.

use std::path::Path;

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Method, Request, StatusCode, header};
use benchflow_viewer::{ServeOptions, Target, app, detect, serve};
use tower::ServiceExt;

const PORT: u16 = 8765;

fn write_trial(dir: &Path, events: &[&str], result: &str) {
    std::fs::create_dir_all(dir.join("trajectory")).unwrap();
    std::fs::write(
        dir.join("trajectory/acp_trajectory.jsonl"),
        events.join("\n"),
    )
    .unwrap();
    std::fs::write(dir.join("result.json"), result).unwrap();
}

/// A folder of jobs: two trials, and a real trial outside the served root.
fn corpus() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let jobs = dir.path().join("jobs");
    write_trial(
        &jobs.join("job-1/alpha__aaaa0001"),
        &[
            r#"{"type": "user_message", "text": "solve alpha"}"#,
            r#"{"type": "agent_message", "text": "alpha solved"}"#,
        ],
        r#"{"task_name": "alpha-task", "rewards": {"reward": 1.0}}"#,
    );
    write_trial(
        &jobs.join("job-1/beta__bbbb0001"),
        &[r#"{"type": "agent_message", "text": "</script><script>alert(1)</script>"}"#],
        r#"{"task_name": "beta-task", "rewards": {"reward": 0.0}}"#,
    );
    write_trial(
        &dir.path().join("outside/secret__cccc0001"),
        &[r#"{"type": "agent_message", "text": "secret"}"#],
        "{}",
    );
    dir
}

fn runs_app(dir: &Path) -> Router {
    app(detect(&dir.join("jobs")).unwrap(), PORT)
}

async fn send(
    router: &Router,
    method: Method,
    uri: &str,
    host: Option<&str>,
) -> (StatusCode, axum::http::HeaderMap, String) {
    let mut request = Request::builder().method(method).uri(uri);
    if let Some(host) = host {
        request = request.header(header::HOST, host);
    }
    let response = router
        .clone()
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, headers, String::from_utf8_lossy(&body).into_owned())
}

async fn get(router: &Router, uri: &str) -> (StatusCode, axum::http::HeaderMap, String) {
    send(router, Method::GET, uri, Some("localhost:8765")).await
}

fn boot(page: &str) -> serde_json::Value {
    let data = page
        .split("type=\"application/json\">")
        .nth(1)
        .unwrap()
        .split("</script>")
        .next()
        .unwrap();
    serde_json::from_str(data).unwrap()
}

#[tokio::test]
async fn the_run_list_page_lists_every_trial() {
    let dir = corpus();
    let (status, headers, page) = get(&runs_app(dir.path()), "/").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(headers[header::CONTENT_TYPE], "text/html; charset=utf-8");
    let boot = boot(&page);
    assert_eq!(boot["mode"], "browse");
    assert_eq!(boot["capped"], false);
    let ids: Vec<&str> = boot["rollouts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|run| run["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["job-1/alpha__aaaa0001", "job-1/beta__bbbb0001"]);
    assert_eq!(boot["rollouts"][0]["task_name"], "alpha-task");
    assert!(page.contains("<title>jobs</title>"));
}

#[tokio::test]
async fn a_run_loads_its_payload_and_hostile_text_stays_data() {
    let dir = corpus();
    let router = runs_app(dir.path());
    let (status, headers, body) = get(&router, "/api/rollout?id=job-1%2Fbeta__bbbb0001").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        headers[header::CONTENT_TYPE],
        "application/json; charset=utf-8"
    );
    let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(
        payload["steps"][0]["text"],
        "</script><script>alert(1)</script>"
    );
    assert_eq!(payload["meta"]["reward"], 0.0);
}

/// G10's bug bash, finding 1: a trial's page showed none of its defense
/// record. The payload now carries every row, in the record's order, with
/// its state and why, read untyped so a row this viewer does not know (or
/// one that is not an object) still shows; a record without one gets no key.
#[tokio::test]
async fn a_trial_payload_carries_every_row_of_its_defense_record() {
    let dir = corpus();
    write_trial(
        &dir.path().join("jobs/job-2/gamma__dddd0001"),
        &[r#"{"type": "agent_message", "text": "done"}"#],
        r#"{"task_name": "gamma-task", "rewards": {"reward": 1.0}, "defenses": {
            "agent_user": {"name": "agent_user", "state": "enforced", "note": "agent, uid 1001"},
            "cloud_metadata": {"name": "cloud_metadata", "state": "off",
              "note": "the task opts out ([metadata] x-allow-cloud-metadata = true)"},
            "network_policy": {"name": "network_policy", "state": "enforced", "note": "agent: open"},
            "a_future_defense": {"name": "a_future_defense", "state": "quarantined", "extra": [1]},
            "process_limit": {"name": "process_limit", "state": "not-supported"},
            "not_an_object": "enforced"}}"#,
    );
    let router = runs_app(dir.path());
    let (status, _, body) = get(&router, "/api/rollout?id=job-2%2Fgamma__dddd0001").await;
    assert_eq!(status, StatusCode::OK);
    let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        payload["meta"]["defenses"],
        serde_json::json!([
            {"name": "agent_user", "state": "enforced", "note": "agent, uid 1001"},
            {"name": "cloud_metadata", "state": "off",
             "note": "the task opts out ([metadata] x-allow-cloud-metadata = true)"},
            {"name": "network_policy", "state": "enforced", "note": "agent: open"},
            {"name": "a_future_defense", "state": "quarantined", "note": null},
            {"name": "process_limit", "state": "not-supported", "note": null},
            {"name": "not_an_object", "state": null, "note": null},
        ])
    );
    // A 0.8 trial has no record, and its payload no key (0.8's, byte for byte).
    let (_, _, body) = get(&router, "/api/rollout?id=job-1%2Falpha__aaaa0001").await;
    let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(payload["meta"].get("defenses").is_none(), "{body}");
    // So is a record whose `defenses` is not an object.
    write_trial(
        &dir.path().join("jobs/job-2/delta__eeee0001"),
        &[],
        r#"{"task_name": "delta-task", "defenses": ["agent_user"]}"#,
    );
    let (status, _, body) = get(&router, "/api/rollout?id=job-2%2Fdelta__eeee0001").await;
    assert_eq!(status, StatusCode::OK);
    let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(payload["meta"].get("defenses").is_none(), "{body}");
}

/// G11 review A, M1: a `bench branch` child's page shows the defense record
/// of the sandbox that graded it, from the child's own result.json (the
/// e2e fixture `fork-nop`: the parent records 9 rows, child n1 16), on the
/// served page and in a trial saved as one file alike.
#[tokio::test]
async fn a_branch_child_payload_carries_its_own_defense_record() {
    let job = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../e2e/viewer/fixtures/fork-nop");
    let trial = job.join("hello__ec981dd5");
    let child = "8b9d7724a324483c9b5f767f1b2e7155/n1";
    let names = |payload: &serde_json::Value| -> Vec<String> {
        payload["meta"]["defenses"]
            .as_array()
            .map(|rows| {
                rows.iter()
                    .map(|row| row["name"].as_str().unwrap().to_string())
                    .collect()
            })
            .unwrap_or_default()
    };
    let router = app(detect(&job).unwrap(), PORT);
    let (status, _, body) = get(
        &router,
        &format!(
            "/api/rollout?id=hello__ec981dd5&branch={}",
            child.replace('/', "%2F")
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let served: serde_json::Value = serde_json::from_str(&body).unwrap();
    let rows = names(&served);
    assert_eq!(rows.len(), 16, "{rows:?}");
    // Rows only the child's record has (the grading sandbox's checks).
    for name in ["clean_verifier_env", "kill_before_grading", "trusted_path"] {
        assert!(rows.iter().any(|row| row == name), "{name} in {rows:?}");
    }
    assert_eq!(served["meta"]["defenses"][0]["state"], "enforced");
    // The static page embeds the same payload for its `?branch=` link.
    let page = benchflow_viewer::render_static(&trial).unwrap();
    assert_eq!(
        boot(&page)["branches"][child]["meta"]["defenses"],
        served["meta"]["defenses"]
    );
}

/// G10's bug bash, findings 1 and 4: the run page draws the defense record
/// and keeps the run list's filter on the views "runs" link with a script
/// of its own, loaded after 0.8's detail view and before its boot, on the
/// served run list and on a trial saved as one file alike.
#[tokio::test]
async fn the_run_page_carries_the_additions_script_before_boot() {
    let dir = corpus();
    let router = runs_app(dir.path());
    let (_, _, list) = get(&router, "/").await;
    let page =
        benchflow_viewer::render_static(&dir.path().join("jobs/job-1/alpha__aaaa0001")).unwrap();
    for page in [&list, &page] {
        let additions = page.find("BF.additions = ").expect("the additions script");
        assert!(page.find("BF.detail = ").unwrap() < additions);
        assert!(additions < page.find("BF.navigation = ").unwrap());
        assert!(page.contains("BF.detail.loadPayload = "));
        assert!(page.contains("#bf-views a[href='/']"));
    }
}

#[tokio::test]
async fn unknown_and_crafted_ids_are_refused() {
    let dir = corpus();
    let router = runs_app(dir.path());
    for uri in [
        "/api/rollout",
        "/api/rollout?id=",
        "/api/rollout?id=job-1",
        "/api/rollout?id=../outside/secret__cccc0001",
        "/api/rollout?id=..%2Foutside%2Fsecret__cccc0001",
        "/api/rollout?id=job-1/../../outside/secret__cccc0001",
        "/api/rollout?id=%2Fetc%2Fpasswd",
        "/api/rollout?id=job-1%2Falpha__aaaa0001%2F..%2Fbeta__bbbb0001",
    ] {
        let (status, headers, body) = get(&router, uri).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
        assert_eq!(
            headers[header::CONTENT_TYPE],
            "application/json; charset=utf-8"
        );
        assert_eq!(body, r#"{"error": "unknown rollout id"}"#, "{uri}");
    }
}

#[tokio::test]
async fn branch_refs_resolve_only_through_the_tree() {
    let dir = corpus();
    let router = runs_app(dir.path());
    let (status, _, body) = get(
        &router,
        "/api/rollout?id=job-1/alpha__aaaa0001&branch=f1/n1",
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(body, r#"{"error": "unknown branch child"}"#);
    // An empty branch is no branch, as Python's parse_qs reads it.
    let (status, _, body) = get(&router, "/api/rollout?id=job-1/alpha__aaaa0001&branch=").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("alpha solved"));
}

#[tokio::test]
async fn foreign_host_names_are_refused() {
    let dir = corpus();
    let router = runs_app(dir.path());
    for host in [
        "attacker.example",
        "attacker.example:8765",
        "localhost:9999",
        "localhost",
    ] {
        let (status, _, body) = send(&router, Method::GET, "/", Some(host)).await;
        assert_eq!(status, StatusCode::FORBIDDEN, "{host}");
        assert!(body.contains("http://localhost:8765/"), "{body}");
    }
    let (status, _, _) = send(
        &router,
        Method::GET,
        "/api/rollout?id=job-1/alpha__aaaa0001",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    for host in ["localhost:8765", "127.0.0.1:8765", "[::1]:8765"] {
        assert_eq!(
            send(&router, Method::GET, "/", Some(host)).await.0,
            StatusCode::OK,
            "{host}"
        );
    }
}

/// The run list leaves out a trial a live run has not finished (its stats,
/// 0.8's page, would count it as unscored); once no process holds the job,
/// the same trial is listed (a run that stopped without a result).
#[tokio::test]
async fn the_run_list_leaves_out_trials_a_live_run_has_not_finished() {
    let dir = corpus();
    let pending = dir.path().join("jobs/job-1/gamma__dddd0001");
    std::fs::create_dir_all(pending.join("trajectory")).unwrap();
    std::fs::write(pending.join("trajectory/acp_trajectory.jsonl"), "").unwrap();
    let lock = std::fs::File::create(dir.path().join("jobs/job-1/.bench-run.lock")).unwrap();
    lock.lock().unwrap();
    let ids = |page: &str| -> Vec<String> {
        boot(page)["rollouts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|run| run["id"].as_str().unwrap().to_string())
            .collect()
    };
    let (_, _, page) = get(&runs_app(dir.path()), "/").await;
    assert_eq!(
        ids(&page),
        ["job-1/alpha__aaaa0001", "job-1/beta__bbbb0001"]
    );
    drop(lock);
    let (_, _, page) = get(&runs_app(dir.path()), "/").await;
    assert_eq!(
        ids(&page),
        [
            "job-1/alpha__aaaa0001",
            "job-1/beta__bbbb0001",
            "job-1/gamma__dddd0001"
        ]
    );
}

#[tokio::test]
async fn other_routes_and_methods_are_refused_without_reading_files() {
    let dir = corpus();
    let router = runs_app(dir.path());
    let (status, _, body) = get(&router, "/api/rollouts").await;
    assert_eq!(
        (status, body.as_str()),
        (StatusCode::NOT_FOUND, "not found")
    );
    assert_eq!(
        get(&router, "/jobs/job-1/alpha__aaaa0001/result.json")
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    let (status, _, body) = send(&router, Method::HEAD, "/", Some("localhost:8765")).await;
    assert_eq!((status, body.as_str()), (StatusCode::OK, ""));
    for uri in ["/api/rollout?id=job-1/alpha__aaaa0001", "/pyproject.toml"] {
        let (status, headers, _) = send(&router, Method::HEAD, uri, Some("localhost:8765")).await;
        assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED, "{uri}");
        assert_eq!(headers[header::ALLOW], "GET");
    }
    let (status, headers, _) = send(&router, Method::POST, "/", Some("localhost:8765")).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    assert!(headers[header::ALLOW].to_str().unwrap().contains("GET"));
}

#[tokio::test]
async fn every_answer_forbids_framing_caching_and_sniffing() {
    let dir = corpus();
    let router = runs_app(dir.path());
    for (uri, host) in [
        ("/", "localhost:8765"),
        ("/api/rollout?id=nope", "localhost:8765"),
        ("/", "evil.example"),
    ] {
        let (_, headers, _) = send(&router, Method::GET, uri, Some(host)).await;
        assert_eq!(headers[header::CACHE_CONTROL], "no-store", "{uri}");
        assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY");
        assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        assert_eq!(headers[header::REFERRER_POLICY], "no-referrer");
        assert_eq!(
            headers[header::CONTENT_SECURITY_POLICY],
            "frame-ancestors 'none'"
        );
    }
}

#[tokio::test]
async fn pages_reflect_the_files_as_they_are_now() {
    let dir = corpus();
    let router = runs_app(dir.path());
    write_trial(
        &dir.path().join("jobs/job-2/gamma__dddd0001"),
        &[r#"{"type": "agent_message", "text": "late"}"#],
        r#"{"task_name": "gamma-task"}"#,
    );
    let (_, _, page) = get(&router, "/").await;
    assert_eq!(boot(&page)["rollouts"].as_array().unwrap().len(), 3);
    let (status, _, _) = get(&router, "/api/rollout?id=job-2/gamma__dddd0001").await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_large_trajectory_streams_in_full() {
    let dir = tempfile::tempdir().unwrap();
    let trial = dir.path().join("big__00000001");
    let text = "x".repeat(400);
    let line = format!(r#"{{"type": "agent_message", "text": "{text}"}}"#);
    let events = vec![line.as_str(); 25_000];
    write_trial(&trial, &events, "{}");
    let router = app(detect(dir.path()).unwrap(), PORT);
    let (status, headers, body) = get(&router, "/api/rollout?id=big__00000001").await;
    assert_eq!(status, StatusCode::OK);
    // Streamed, so no length is known up front.
    assert!(!headers.contains_key(header::CONTENT_LENGTH));
    let payload: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(payload["steps"].as_array().unwrap().len(), 25_000);
    assert_eq!(payload["meta"]["counts"]["messages"], 25_000);
}

#[tokio::test]
async fn serve_listens_answers_and_stops() {
    let dir = corpus();
    let handle = serve(
        &dir.path().join("jobs"),
        ServeOptions {
            port: Some(0),
            ..ServeOptions::default()
        },
    )
    .await
    .unwrap();
    assert!(handle.addr.ip().is_loopback());
    assert_eq!(
        handle.url,
        format!("http://localhost:{}", handle.addr.port())
    );
    assert!(matches!(
        handle.target,
        Target::Runs {
            runs: 2,
            capped: false,
            ..
        }
    ));

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(handle.addr).await.unwrap();
    let request = format!(
        "GET / HTTP/1.1\r\nHost: localhost:{}\r\nConnection: close\r\n\r\n",
        handle.addr.port()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut response = String::new();
    stream.read_to_string(&mut response).await.unwrap();
    assert!(
        response.starts_with("HTTP/1.1 200 OK"),
        "{}",
        &response[..80.min(response.len())]
    );
    assert!(response.contains("\"mode\": \"browse\""));

    let addr = handle.addr;
    handle.stop().await.unwrap();
    assert!(
        tokio::net::TcpStream::connect(addr).await.is_err(),
        "still listening after stop"
    );
}

#[tokio::test]
async fn serve_refuses_what_it_cannot_show() {
    let dir = tempfile::tempdir().unwrap();
    let error = serve(
        dir.path(),
        ServeOptions {
            port: Some(0),
            ..ServeOptions::default()
        },
    )
    .await
    .unwrap_err();
    assert!(
        error.to_string().contains("no trajectories found"),
        "{error}"
    );
}
