//! The Rust viewer serves exactly what BenchFlow 0.8 serves.
//!
//! `tests/fixtures/generate.py` recorded the 0.8 Python viewer's output for
//! every trial under `tests/fixtures/jobs` into `tests/fixtures/golden`.
//! These tests ask the Rust viewer for the same things over its HTTP routes
//! (no network: requests go straight to the router) and compare bytes.

use std::path::{Path, PathBuf};

use axum::Router;
use axum::body::{Body, to_bytes};
use axum::http::{Request, StatusCode};
use benchflow_viewer::{Target, app, detect, render_static};
use sha2::{Digest, Sha256};
use tower::ServiceExt;

const PORT: u16 = 8765;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn root() -> PathBuf {
    fixtures().join("jobs")
}

fn golden_text(relative: &str) -> String {
    let path = fixtures().join("golden").join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()))
}

fn golden_json(relative: &str) -> serde_json::Value {
    serde_json::from_str(&golden_text(relative)).unwrap()
}

/// Absolute fixture paths written as `<root>`, as the goldens have them.
fn normalize(text: &str) -> String {
    let root = root();
    let canonical = std::fs::canonicalize(&root).unwrap();
    text.replace(canonical.to_str().unwrap(), "<root>")
        .replace(root.to_str().unwrap(), "<root>")
}

fn sha256(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Percent-encode a query value.
fn encode(text: &str) -> String {
    text.bytes()
        .map(|byte| match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => {
                (byte as char).to_string()
            }
            other => format!("%{other:02X}"),
        })
        .collect()
}

async fn get(router: &Router, uri: &str) -> (StatusCode, String) {
    let request = Request::get(uri)
        .header("host", format!("localhost:{PORT}"))
        .body(Body::empty())
        .unwrap();
    let response = router.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(body.to_vec()).unwrap())
}

/// The boot data a page embeds.
fn boot_text(page: &str) -> &str {
    page.split("type=\"application/json\">")
        .nth(1)
        .unwrap()
        .split("</script>")
        .next()
        .unwrap()
}

/// Compare long texts with a readable failure: the first differing line.
fn assert_same(actual: &str, expected: &str, what: &str) {
    if actual == expected {
        return;
    }
    let position = actual
        .bytes()
        .zip(expected.bytes())
        .position(|(a, b)| a != b)
        .unwrap_or(actual.len().min(expected.len()));
    let from = position.saturating_sub(120);
    let context = |text: &str| {
        text.get(from..(position + 120).min(text.len()))
            .unwrap_or("")
            .to_string()
    };
    panic!(
        "{what} differs from 0.8 at byte {position}\n  rust: …{}…\n  0.8:  …{}…",
        context(actual),
        context(expected)
    );
}

fn runs_app() -> Router {
    let target = detect(&root()).unwrap();
    assert!(
        matches!(target, Target::Runs { capped: false, .. }),
        "{target:?}"
    );
    app(target, PORT)
}

#[tokio::test]
async fn every_trial_payload_matches_0_8() {
    let router = runs_app();
    let index = golden_json("index.json");
    let entries = index.as_array().unwrap();
    assert!(entries.len() > 150, "only {} fixture trials", entries.len());
    for entry in entries {
        let id = entry["id"].as_str().unwrap();
        let (status, body) = get(&router, &format!("/api/rollout?id={}", encode(id))).await;
        assert_eq!(status, StatusCode::OK, "{id}");
        assert_same(
            &normalize(&body),
            &golden_text(entry["payload"].as_str().unwrap()),
            &format!("payload of {id}"),
        );
    }
}

#[tokio::test]
async fn every_branch_child_payload_matches_0_8() {
    let router = runs_app();
    let mut checked = 0;
    for entry in golden_json("index.json").as_array().unwrap() {
        let id = entry["id"].as_str().unwrap();
        for (child_ref, file) in entry["branches"].as_object().unwrap() {
            let uri = format!(
                "/api/rollout?id={}&branch={}",
                encode(id),
                encode(child_ref)
            );
            let (status, body) = get(&router, &uri).await;
            let expected = golden_text(file.as_str().unwrap());
            if expected == "null" {
                assert_eq!(status, StatusCode::NOT_FOUND, "{id} {child_ref:?}");
            } else {
                assert_eq!(status, StatusCode::OK, "{id} {child_ref:?}");
                assert_same(
                    &normalize(&body),
                    &expected,
                    &format!("branch {child_ref:?} of {id}"),
                );
            }
            checked += 1;
        }
    }
    assert!(checked >= 8, "only {checked} branch children");
}

/// The run list page with its one addition removed: the links to the
/// compare and outcomes views, one `<nav id="bf-views">` element placed
/// just before the theme button. Fails unless there is exactly one.
fn without_view_links(page: &str) -> String {
    const START: &str = "<nav id=\"bf-views\"";
    const END: &str = "</nav>\n      ";
    assert_eq!(page.matches(START).count(), 1, "one nav element");
    let start = page.find(START).unwrap();
    let end = start + page[start..].find(END).unwrap() + END.len();
    assert!(
        page[end..].starts_with("<button id=\"theme-toggle\""),
        "the nav sits before the theme button"
    );
    format!("{}{}", &page[..start], &page[end..])
}

/// A page with the Rust viewer's script addition taken out: `additions.js`,
/// joined after `detail.js` as 0.8 joins its scripts. Fails unless it is
/// there exactly once.
fn without_additions(page: &str) -> String {
    let added = format!("\n\n{}", include_str!("../assets/additions.js"));
    assert_eq!(page.matches(&added).count(), 1, "one additions script");
    page.replacen(&added, "", 1)
}

#[tokio::test]
async fn the_run_list_page_matches_0_8() {
    let router = runs_app();
    let (status, page) = get(&router, "/").await;
    assert_eq!(status, StatusCode::OK);
    // 0.8's page, byte for byte, once the view links and the additions
    // script are taken out.
    let page = normalize(&without_additions(&without_view_links(&page)));
    assert_same(
        boot_text(&page),
        &golden_text("pages/browse.boot.txt"),
        "run list data",
    );
    assert_eq!(
        sha256(&page),
        golden_text("pages/browse.sha256").trim(),
        "run list page"
    );
}

#[test]
fn static_trial_pages_match_0_8() {
    for entry in golden_json("pages/index.json").as_array().unwrap() {
        let id = entry["id"].as_str().unwrap();
        let name = entry["page"].as_str().unwrap();
        let page = normalize(&without_additions(
            &render_static(&root().join(id)).unwrap(),
        ));
        assert_same(
            boot_text(&page),
            &golden_text(&format!("{name}.boot.txt")),
            &format!("page data of {id}"),
        );
        assert_eq!(
            sha256(&page),
            golden_text(&format!("{name}.sha256")).trim(),
            "page of {id}"
        );
    }
}

#[tokio::test]
async fn a_trial_folder_serves_its_own_page() {
    let trial = root().join("branch-tree/claude-continue/hello-world-task__0000abcd");
    let target = detect(&trial).unwrap();
    assert_eq!(target, Target::Trial { dir: trial.clone() });
    let router = app(target, PORT);
    let (status, page) = get(&router, "/").await;
    assert_eq!(status, StatusCode::OK);
    // The same page `render_static` writes, which matches 0.8 above.
    assert_eq!(page, render_static(&trial).unwrap());
    assert_eq!(
        get(&router, "/api/rollout?id=x").await.0,
        StatusCode::NOT_FOUND
    );
}

#[tokio::test]
async fn a_job_folder_lists_only_its_trials() {
    let job = root().join("oracle-hello/2026-09-29__12-46-15");
    let router = app(detect(&job).unwrap(), PORT);
    let (_, page) = get(&router, "/").await;
    let boot: serde_json::Value = serde_json::from_str(boot_text(&page)).unwrap();
    let ids: Vec<&str> = boot["rollouts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|run| run["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        [
            "hello-2__f2da4092",
            "hello-3__f061ba46",
            "hello-4__0e8d5b9a"
        ]
    );
    // A job-relative id serves the same payload the jobs root serves.
    let (status, body) = get(&router, "/api/rollout?id=hello-3__f061ba46").await;
    assert_eq!(status, StatusCode::OK);
    let full = runs_app();
    let (_, from_root) = get(
        &full,
        "/api/rollout?id=oracle-hello/2026-09-29__12-46-15/hello-3__f061ba46",
    )
    .await;
    assert_eq!(body, from_root);
}

#[test]
fn fixture_paths_exist() {
    assert!(Path::new(&root()).is_dir());
}
