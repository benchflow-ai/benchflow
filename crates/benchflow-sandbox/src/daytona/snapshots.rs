//! Checkpoints as Daytona snapshots.
//!
//! - take: `POST /sandbox/{id}/snapshot` with `{"name"}` (it answers with the
//!   sandbox, not the snapshot), then `GET /snapshots/{name}` until the
//!   snapshot is `active` and the sandbox has left `snapshotting`;
//! - delete: `GET /snapshots/{name}` for its id, `DELETE /snapshots/{id}`;
//! - list: `GET /snapshots?page=&limit=`, page by page, keeping our names.
//!
//! Snapshots carry no labels: whose a snapshot is comes from its name
//! ([`crate::checkpoint`]), and a name that is not ours is never deleted.

use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};

use super::api::{Api, ApiError, Retry};
use super::lifecycle;
use super::types::{CreateSnapshot, SnapshotDto, SnapshotPage};
use crate::SandboxError;
use crate::checkpoint::{self, Checkpoint, Listed, Sweep};

/// How long a snapshot may take to become active.
pub(super) const TIMEOUT: Duration = Duration::from_secs(20 * 60);
/// How often its state is checked meanwhile.
pub(super) const POLL_EVERY: Duration = Duration::from_secs(4);
/// A snapshot not found this long after its create was accepted is taken
/// as never made.
const NOT_FOUND_GRACE: Duration = Duration::from_secs(60);
/// Snapshot states that end a take as failed.
const FAILED: &[&str] = &["error", "build_failed", "inactive"];
const GB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Snapshot sandbox `sandbox_id` as `name`, and wait until it is usable.
pub(super) async fn take(
    api: &Api,
    sandbox_id: &str,
    name: &str,
    limit: Duration,
    every: Duration,
) -> Result<Checkpoint, SandboxError> {
    let started = Instant::now();
    if checkpoint::parse(name).is_none() {
        return Err(SandboxError::Unsupported(format!(
            "`{name}` is not a checkpoint name (a bug in the caller)"
        )));
    }
    let body = CreateSnapshot { name };
    let sent = api
        .send("snapshotting a sandbox", Retry::Unsent, |http| {
            Ok(http
                .post(api.url(&format!("/sandbox/{sandbox_id}/snapshot")))
                .timeout(Duration::from_secs(120))
                .json(&body))
        })
        .await;
    match sent {
        Ok(_) => {}
        // It may have gone through: the poll finds out.
        Err(err) if err.ambiguous() => {
            tracing::debug!(
                "snapshot of {sandbox_id} failed ambiguously: {}",
                err.message()
            )
        }
        Err(err) => return Err(err.into()),
    }
    let failed = |message: String| async move {
        if let Err(err) = delete(api, name).await {
            tracing::warn!("could not delete the failed snapshot {name}: {err}");
        }
        Err(SandboxError::Transport(message))
    };
    let snapshot = loop {
        match get(api, name).await {
            Ok(dto) if dto.state() == "active" => break dto,
            Ok(dto) if FAILED.contains(&dto.state()) => {
                return failed(format!(
                    "Daytona's snapshot {name} of sandbox {sandbox_id} ended `{}` ({}); the \
                     sandbox's filesystem may be too large, or Daytona failed; run again",
                    dto.state(),
                    dto.error_reason.as_deref().unwrap_or("no reason given")
                ))
                .await;
            }
            Ok(_) => {}
            Err(err) if err.status() == Some(404) && started.elapsed() < NOT_FOUND_GRACE => {}
            Err(err) if err.status() == Some(404) => {
                // Delete it in case it appears after all; if it appears
                // only later, the sweep takes it (its name has our pid).
                return failed(format!(
                    "Daytona accepted the snapshot of sandbox {sandbox_id} but never made {name}; \
                     run again"
                ))
                .await;
            }
            Err(err) => return failed(SandboxError::from(err).to_string()).await,
        }
        if started.elapsed() > limit {
            return failed(format!(
                "Daytona's snapshot {name} of sandbox {sandbox_id} was not active after {}s; run \
                 again, or with a smaller sandbox filesystem",
                limit.as_secs()
            ))
            .await;
        }
        tokio::time::sleep(every).await;
    };
    // The sandbox takes commands again once it has left `snapshotting`.
    loop {
        match lifecycle::get(api, sandbox_id).await {
            Ok(dto) if dto.state() == "snapshotting" && started.elapsed() <= limit => {
                tokio::time::sleep(every).await
            }
            Ok(dto) if matches!(dto.state(), "error" | "build_failed") => {
                // The snapshot is active and would outlive the failed take.
                return failed(format!(
                    "sandbox {sandbox_id} ended `{}` while Daytona snapshotted it ({})",
                    dto.state(),
                    dto.error_reason.as_deref().unwrap_or("no reason given")
                ))
                .await;
            }
            _ => break,
        }
    }
    Ok(Checkpoint {
        provider: "daytona".to_string(),
        reference: name.to_string(),
        seconds: started.elapsed().as_secs_f64(),
        stored_bytes: bytes(&snapshot),
        default_user: None,
    })
}

fn bytes(dto: &SnapshotDto) -> Option<u64> {
    dto.size
        .filter(|s| s.is_finite() && *s > 0.0)
        .map(|s| (s * GB) as u64)
}

async fn get(api: &Api, name: &str) -> Result<SnapshotDto, ApiError> {
    api.json("checking a snapshot", Retry::Safe, |http| {
        Ok(http
            .get(api.url(&format!("/snapshots/{name}")))
            .timeout(Duration::from_secs(60)))
    })
    .await
}

/// How many times a delete Daytona refuses for now (409) is tried.
const DELETE_ATTEMPTS: u32 = 4;
/// How long a delete Daytona accepted is watched until the snapshot is gone.
const DELETE_SETTLE: Duration = Duration::from_secs(30);

/// Delete snapshot `name`. True once it is gone (or was never there); false
/// when Daytona accepted the delete and still lists the snapshot.
pub(super) async fn delete(api: &Api, name: &str) -> Result<bool, SandboxError> {
    if checkpoint::parse(name).is_none() {
        return Err(SandboxError::Unsupported(format!(
            "`{name}` is not a checkpoint; only checkpoint snapshots are deleted this way"
        )));
    }
    let mut attempt = 0;
    loop {
        attempt += 1;
        let dto = match get(api, name).await {
            Ok(dto) => dto,
            Err(err) if err.status() == Some(404) => return Ok(true),
            Err(err) => return Err(err.into()),
        };
        if dto.state() == "removing" {
            return Ok(false);
        }
        // The API retries transport failures, 429 and 5xx itself.
        let result = api
            .send("deleting a snapshot", Retry::Safe, |http| {
                Ok(http
                    .delete(api.url(&format!("/snapshots/{}", dto.id)))
                    .timeout(Duration::from_secs(60)))
            })
            .await;
        match result {
            Ok(_) => {}
            Err(err) if err.status() == Some(404) => return Ok(true),
            // In use, or changing state: wait and try again.
            Err(err) if err.status() == Some(409) && attempt < DELETE_ATTEMPTS => {
                tokio::time::sleep(Duration::from_secs(2u64 << attempt)).await;
                continue;
            }
            Err(err) => {
                return Err(SandboxError::Transport(format!(
                    "could not delete Daytona snapshot {name}: {}",
                    err.message()
                )));
            }
        }
        // Daytona removes a snapshot a few seconds after accepting the
        // delete; wait that long before calling it deferred.
        let accepted = Instant::now();
        loop {
            match get(api, name).await {
                Err(err) if err.status() == Some(404) => return Ok(true),
                _ if accepted.elapsed() >= DELETE_SETTLE => return Ok(false),
                _ => tokio::time::sleep(Duration::from_secs(2)).await,
            }
        }
    }
}

/// Every checkpoint snapshot of this owner scope.
pub(super) async fn list(api: &Api) -> Result<Vec<SnapshotDto>, ApiError> {
    let mut out = Vec::new();
    for page in 1..=100u32 {
        let answer: SnapshotPage = api
            .json("listing snapshots", Retry::Safe, |http| {
                Ok(http
                    .get(api.url("/snapshots"))
                    .query(&[("page", page.to_string()), ("limit", "100".to_string())])
                    .timeout(Duration::from_secs(60)))
            })
            .await?;
        let empty = answer.items.is_empty();
        out.extend(
            answer
                .items
                .into_iter()
                .filter(|s| checkpoint::ours(&s.name)),
        );
        let last = answer.total_pages.map_or(empty, |t| f64::from(page) >= t);
        if last || empty {
            break;
        }
    }
    Ok(out)
}

fn created(dto: &SnapshotDto) -> Option<DateTime<Utc>> {
    dto.created_at
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|t| t.with_timezone(&Utc))
}

pub(super) fn listed(dto: SnapshotDto) -> Listed {
    Listed {
        created: created(&dto),
        state: dto.state.clone(),
        checkpoint: Checkpoint {
            provider: "daytona".to_string(),
            stored_bytes: bytes(&dto),
            default_user: None,
            reference: dto.name,
            seconds: 0.0,
        },
    }
}

/// Delete the checkpoint snapshots [`checkpoint::verdict`] condemns. Best
/// effort: a failure is logged, not returned.
pub(super) async fn sweep(api: &Api) -> usize {
    let snapshots = match list(api).await {
        Ok(snapshots) => snapshots,
        Err(err) => {
            tracing::warn!(
                "could not list Daytona snapshots to sweep: {}",
                err.message()
            );
            return 0;
        }
    };
    let now = Utc::now();
    let mut removed = 0;
    for snapshot in snapshots {
        let age = created(&snapshot).and_then(|c| now.signed_duration_since(c).to_std().ok());
        if checkpoint::verdict(&snapshot.name, age).await != Sweep::Delete {
            continue;
        }
        match delete(api, &snapshot.name).await {
            Ok(_) => {
                removed += 1;
                tracing::info!("deleted leftover Daytona snapshot {}", snapshot.name);
            }
            Err(err) => tracing::warn!(
                "could not delete leftover Daytona snapshot {}: {err}",
                snapshot.name
            ),
        }
    }
    removed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// A fake Daytona API on localhost: each request is answered with
    /// `handler(method, path)` and recorded as `METHOD path body`.
    async fn serve(
        handler: impl Fn(&str, &str) -> (u16, String) + Send + Sync + 'static,
    ) -> (Api, Arc<Mutex<Vec<String>>>) {
        let handler = Arc::new(handler);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                let handler = handler.clone();
                let log = log.clone();
                tokio::spawn(async move {
                    let mut buf = Vec::new();
                    let mut chunk = [0u8; 4096];
                    let head_end = loop {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            return;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                            break p + 4;
                        }
                    };
                    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                    let length = head
                        .lines()
                        .find_map(|l| {
                            let (k, v) = l.split_once(':')?;
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse::<usize>().ok())
                                .flatten()
                        })
                        .unwrap_or(0);
                    while buf.len() < head_end + length {
                        let n = sock.read(&mut chunk).await.unwrap_or(0);
                        if n == 0 {
                            break;
                        }
                        buf.extend_from_slice(&chunk[..n]);
                    }
                    let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
                    let mut first = head.split_whitespace();
                    let method = first.next().unwrap_or_default().to_string();
                    let path = first.next().unwrap_or_default().to_string();
                    log.lock()
                        .unwrap()
                        .push(format!("{method} {path} {body}").trim_end().to_string());
                    let (status, answer) = handler(&method, &path);
                    let reply = format!(
                        "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{answer}",
                        answer.len()
                    );
                    let _ = sock.write_all(reply.as_bytes()).await;
                    let _ = sock.shutdown().await;
                });
            }
        });
        let api = Api::new(&format!("http://{addr}"), Some("k".into()), None).unwrap();
        (api, seen)
    }

    fn snapshot(id: &str, name: &str, state: &str) -> String {
        serde_json::json!({
            "id": id, "name": name, "state": state, "size": 2.0,
            "errorReason": "boom", "createdAt": "2026-10-01T00:00:00Z",
        })
        .to_string()
    }

    fn mine() -> String {
        checkpoint::name_for(
            &crate::owner::scope(),
            crate::owner::machine_id(),
            std::process::id(),
        )
    }

    #[tokio::test]
    async fn a_snapshot_is_polled_until_active() {
        let name = mine();
        let polls = Arc::new(AtomicUsize::new(0));
        let (n, p) = (name.clone(), polls.clone());
        let (api, seen) = serve(move |method, path| match (method, path) {
            ("POST", "/sandbox/s1/snapshot") => (200, r#"{"id":"s1"}"#.into()),
            ("GET", "/sandbox/s1") => (200, r#"{"id":"s1","state":"started"}"#.into()),
            ("GET", _) => {
                let state = if p.fetch_add(1, Ordering::SeqCst) < 2 {
                    "pending"
                } else {
                    "active"
                };
                (200, snapshot("sn1", &n, state))
            }
            _ => (500, "{}".into()),
        })
        .await;
        let made = take(
            &api,
            "s1",
            &name,
            Duration::from_secs(10),
            Duration::from_millis(5),
        )
        .await
        .unwrap();
        assert_eq!(made.provider, "daytona");
        assert_eq!(made.reference, name);
        assert_eq!(made.stored_bytes, Some(2 * 1024 * 1024 * 1024));
        let seen = seen.lock().unwrap().clone();
        assert_eq!(
            seen[0],
            format!("POST /sandbox/s1/snapshot {{\"name\":\"{name}\"}}")
        );
        assert_eq!(polls.load(Ordering::SeqCst), 3);
    }

    /// Before a snapshot, every command session the daemon keeps is
    /// deleted, and the listing is read again until it is empty.
    #[tokio::test]
    async fn the_daemons_sessions_are_cleared_before_a_snapshot() {
        let deleted = Arc::new(AtomicUsize::new(0));
        let d = deleted.clone();
        let (api, seen) = serve(move |method, path| match (method, path) {
            ("GET", "/s1/process/session") => {
                if d.load(Ordering::SeqCst) >= 2 {
                    (200, "[]".into())
                } else {
                    (
                        200,
                        r#"[{"sessionId":"bf-a","commands":[]},{"sessionId":"agent-x","commands":[]}]"#
                            .into(),
                    )
                }
            }
            ("DELETE", _) => {
                d.fetch_add(1, Ordering::SeqCst);
                (204, String::new())
            }
            _ => (500, "{}".into()),
        })
        .await;
        let base = api.base().to_string();
        let toolbox = super::super::toolbox::Toolbox::new(Arc::new(api), &base, "s1");
        toolbox
            .clear_sessions(Duration::from_secs(10))
            .await
            .unwrap();
        let seen = seen.lock().unwrap().clone();
        assert!(
            seen.contains(&"DELETE /s1/process/session/bf-a".to_string()),
            "{seen:?}"
        );
        assert!(
            seen.contains(&"DELETE /s1/process/session/agent-x".to_string()),
            "{seen:?}"
        );
        assert_eq!(seen.last().unwrap(), "GET /s1/process/session");
    }

    #[tokio::test]
    async fn a_failed_or_slow_snapshot_fails_and_is_deleted() {
        for (state, limit) in [
            ("error", Duration::from_secs(10)),
            ("pending", Duration::from_millis(30)),
        ] {
            let name = mine();
            let deleted = Arc::new(AtomicBool::new(false));
            let (n, d) = (name.clone(), deleted.clone());
            let (api, seen) = serve(move |method, path| match (method, path) {
                ("POST", _) => (200, r#"{"id":"s1"}"#.into()),
                ("DELETE", "/snapshots/sn1") => {
                    d.store(true, Ordering::SeqCst);
                    (200, "{}".into())
                }
                ("GET", _) if d.load(Ordering::SeqCst) => (404, r#"{"message":"gone"}"#.into()),
                ("GET", _) => (200, snapshot("sn1", &n, state)),
                _ => (500, "{}".into()),
            })
            .await;
            let err = take(&api, "s1", &name, limit, Duration::from_millis(5))
                .await
                .unwrap_err();
            let text = err.to_string();
            if state == "error" {
                assert!(text.contains("`error`") && text.contains("boom"), "{text}");
            } else {
                assert!(text.contains("not active"), "{text}");
            }
            assert!(deleted.load(Ordering::SeqCst), "{:?}", seen.lock().unwrap());
        }
    }

    #[tokio::test]
    async fn delete_refuses_other_names_and_reads_gone_and_removing() {
        let (api, seen) = serve(|_, path| {
            if path.ends_with("-000000000000") {
                (404, r#"{"message":"not found"}"#.into())
            } else {
                (200, snapshot("sn1", "x", "removing"))
            }
        })
        .await;
        assert!(matches!(
            delete(&api, "ubuntu-24-04").await,
            Err(SandboxError::Unsupported(_))
        ));
        assert!(seen.lock().unwrap().is_empty());
        let name = mine();
        let gone = format!("{}000000000000", &name[..name.len() - 12]);
        assert!(delete(&api, &gone).await.unwrap());
        assert!(!delete(&api, &name).await.unwrap(), "still removing");
        assert!(!seen.lock().unwrap().iter().any(|r| r.starts_with("DELETE")));
    }

    #[tokio::test]
    async fn the_sweep_deletes_dead_runs_snapshots_and_never_foreign_ones() {
        let mut child = tokio::process::Command::new("true").spawn().unwrap();
        let dead = child.id().unwrap();
        child.wait().await.unwrap();
        let here = crate::owner::machine_id();
        let gone = checkpoint::name_for(&crate::owner::scope(), here, dead);
        let live = mine();
        let foreign = checkpoint::name_for("someone-else", here, dead);
        let page = serde_json::json!({
            "items": [
                serde_json::from_str::<serde_json::Value>(&snapshot("g", &gone, "active")).unwrap(),
                serde_json::from_str::<serde_json::Value>(&snapshot("l", &live, "active")).unwrap(),
                serde_json::from_str::<serde_json::Value>(&snapshot("f", &foreign, "active")).unwrap(),
                serde_json::from_str::<serde_json::Value>(&snapshot("o", "ubuntu-base", "active")).unwrap(),
            ],
            "total": 4, "page": 1, "totalPages": 1,
        })
        .to_string();
        let deleted = Arc::new(AtomicBool::new(false));
        let (g, d) = (gone.clone(), deleted.clone());
        let (api, seen) = serve(move |method, path| match (method, path) {
            ("GET", p) if p.starts_with("/snapshots?") => (200, page.clone()),
            ("DELETE", "/snapshots/g") => {
                d.store(true, Ordering::SeqCst);
                (200, "{}".into())
            }
            ("GET", p) if p == format!("/snapshots/{g}") && !d.load(Ordering::SeqCst) => {
                (200, snapshot("g", &g, "active"))
            }
            _ => (404, r#"{"message":"not found"}"#.into()),
        })
        .await;
        let listed = list(&api).await.unwrap();
        assert_eq!(listed.len(), 2, "only ours: {listed:?}");
        assert_eq!(sweep(&api).await, 1);
        let deletes: Vec<String> = seen
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.starts_with("DELETE"))
            .cloned()
            .collect();
        assert_eq!(deletes, ["DELETE /snapshots/g"]);
    }
}
