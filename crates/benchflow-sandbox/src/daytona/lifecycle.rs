//! A Daytona sandbox's life: create it, wait until it runs (streaming its
//! image build log meanwhile), delete it.
//!
//! Lessons from BenchFlow 0.8: a create that fails ambiguously (a timeout, a
//! gateway error) may still have made a sandbox, so every create carries a
//! one-off attempt label that finds it again; a sandbox that fails to start
//! (`build_failed`, `error`) is not deleted by Daytona, so the caller deletes
//! it; a delete may answer 404 when an earlier attempt already went through.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use futures::StreamExt;

use super::api::{Api, ApiError, Retry};
use super::types::{CreateSandbox, NetworkSettings, SandboxDto, SandboxPage, UrlDto};
use crate::SandboxError;

/// The label that tells one create call's sandbox apart.
pub(crate) const ATTEMPT_LABEL: &str = "ai.benchflow.attempt";

/// How many lines of a build log a failure keeps.
const LOG_TAIL_LINES: usize = 40;

/// Create a sandbox. `body.labels` must include [`ATTEMPT_LABEL`].
pub(crate) async fn create(
    api: &Api,
    body: &CreateSandbox<'_>,
) -> Result<SandboxDto, SandboxError> {
    let attempt = body.labels.get(ATTEMPT_LABEL).cloned().unwrap_or_default();
    let result: Result<SandboxDto, ApiError> = api
        .json("creating a sandbox", Retry::Unsent, |http| {
            Ok(http
                .post(api.url("/sandbox"))
                .timeout(Duration::from_secs(120))
                .json(body))
        })
        .await;
    match result {
        Ok(dto) => Ok(dto),
        Err(err) if err.ambiguous() && !attempt.is_empty() => {
            // The create may have gone through; look for its sandbox.
            tracing::debug!(
                "create failed ambiguously ({}); looking for its sandbox",
                err.message()
            );
            match find_by_label(api, ATTEMPT_LABEL, &attempt).await {
                Ok(Some(dto)) => Ok(dto),
                _ => Err(err.into()),
            }
        }
        Err(err) => Err(err.into()),
    }
}

/// The sandbox carrying `label=value`, if exactly that one exists.
pub(crate) async fn find_by_label(
    api: &Api,
    label: &str,
    value: &str,
) -> Result<Option<SandboxDto>, ApiError> {
    let filter = serde_json::json!({ label: value }).to_string();
    let page: SandboxPage = api
        .json("looking up a sandbox by label", Retry::Safe, |http| {
            Ok(http
                .get(api.url("/sandbox"))
                .query(&[("labels", filter.as_str()), ("limit", "10")])
                .timeout(Duration::from_secs(60)))
        })
        .await?;
    Ok(page
        .items
        .into_iter()
        .find(|s| s.labels.get(label).map(String::as_str) == Some(value)))
}

/// Every sandbox whose labels include all of `labels`, page by page.
pub(crate) async fn list(
    api: &Api,
    labels: &serde_json::Value,
) -> Result<Vec<SandboxDto>, ApiError> {
    let filter = labels.to_string();
    let mut out = Vec::new();
    let mut cursor: Option<String> = None;
    for _ in 0..100 {
        let page: SandboxPage = api
            .json("listing sandboxes", Retry::Safe, |http| {
                let mut request = http
                    .get(api.url("/sandbox"))
                    .query(&[("labels", filter.as_str()), ("limit", "200")])
                    .timeout(Duration::from_secs(60));
                if let Some(c) = &cursor {
                    request = request.query(&[("cursor", c.as_str())]);
                }
                Ok(request)
            })
            .await?;
        out.extend(page.items);
        match page.next_cursor.filter(|c| !c.is_empty()) {
            Some(next) => cursor = Some(next),
            None => break,
        }
    }
    Ok(out)
}

pub(crate) async fn get(api: &Api, id: &str) -> Result<SandboxDto, ApiError> {
    api.json("checking a sandbox", Retry::Safe, |http| {
        Ok(http
            .get(api.url(&format!("/sandbox/{id}")))
            .timeout(Duration::from_secs(60)))
    })
    .await
}

/// Change a running sandbox's network: block it all, or limit it to an
/// allow list (empty lifts the list). Daytona applies it outside the
/// sandbox, to every process in it.
pub(crate) async fn set_network(
    api: &Api,
    id: &str,
    settings: &NetworkSettings,
) -> Result<(), SandboxError> {
    api.send("changing a sandbox's network", Retry::Safe, |http| {
        Ok(http
            .post(api.url(&format!("/sandbox/{id}/network-settings")))
            .timeout(Duration::from_secs(60))
            .json(settings))
    })
    .await?;
    Ok(())
}

/// Delete a sandbox; one that is already gone, or already being deleted,
/// counts as deleted. While another state change is in progress (Daytona
/// answers 409, for example while the sandbox is still starting), wait for
/// it to finish and try again, for up to 90 seconds.
pub(crate) async fn delete(api: &Api, id: &str) -> Result<(), SandboxError> {
    let deadline = Instant::now() + Duration::from_secs(90);
    let result = loop {
        let result = api
            .send("deleting a sandbox", Retry::Safe, |http| {
                Ok(http
                    .delete(api.url(&format!("/sandbox/{id}")))
                    .timeout(Duration::from_secs(120)))
            })
            .await;
        match result {
            Err(err) if err.status() == Some(409) => {
                match get(api, id).await {
                    Err(e) if e.status() == Some(404) => break Ok(()),
                    Ok(dto) if matches!(dto.state(), "destroying" | "destroyed") => break Ok(()),
                    _ => {}
                }
                if Instant::now() > deadline {
                    break Err(err);
                }
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
            other => break other.map(|_| ()),
        }
    };
    match result {
        Ok(()) => Ok(()),
        Err(err) if err.status() == Some(404) => Ok(()),
        Err(err) => Err(match SandboxError::from(err) {
            SandboxError::Transport(m) | SandboxError::Unsupported(m) => {
                SandboxError::Transport(format!(
                    "{}. Sandbox {id} may still exist; Daytona deletes it on its own once it has been \
                 idle for its auto-stop interval, and `bench` sweeps it at the next run.",
                    m.trim_end_matches('.')
                ))
            }
            other => other,
        }),
    }
}

/// A sandbox that is running.
#[derive(Debug)]
pub(crate) struct Started {
    pub dto: SandboxDto,
    /// Daytona built (or staged) the image for it.
    pub built: bool,
}

/// Why a sandbox did not start.
#[derive(Debug)]
pub(crate) struct Failure {
    pub state: String,
    pub reason: String,
    pub log_tail: Vec<String>,
    /// It did not fail; it ran out of time.
    pub timed_out: bool,
    /// The API failed while waiting.
    pub api: Option<SandboxError>,
}

/// Wait until sandbox `id` runs, for at most `limit`, sending its build
/// log's lines to `log` while Daytona builds the image.
pub(crate) async fn wait_started(
    api: &Api,
    id: &str,
    limit: Duration,
    log: &(dyn Fn(&str) + Send + Sync),
) -> Result<Started, Box<Failure>> {
    let started = Instant::now();
    let mut built = false;
    let mut logs: Option<futures::stream::BoxStream<'static, reqwest::Result<bytes::Bytes>>> = None;
    let mut logs_tried = false;
    let mut partial: Vec<u8> = Vec::new();
    let mut tail: VecDeque<String> = VecDeque::new();
    let mut interval = Duration::from_millis(300);

    let emit = |raw: &[u8], tail: &mut VecDeque<String>| {
        let line = String::from_utf8_lossy(raw);
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            return;
        }
        log(line);
        if tail.len() == LOG_TAIL_LINES {
            tail.pop_front();
        }
        tail.push_back(line.to_string());
    };

    loop {
        let dto = match get(api, id).await {
            Ok(dto) => dto,
            Err(err) => {
                let missing = err.status() == Some(404);
                return Err(Box::new(Failure {
                    state: if missing { "gone" } else { "unknown" }.to_string(),
                    reason: err.message(),
                    log_tail: tail.into(),
                    timed_out: false,
                    api: (!missing).then(|| err.into()),
                }));
            }
        };
        let state = dto.state().to_string();
        match state.as_str() {
            "started" => return Ok(Started { dto, built }),
            "error" | "build_failed" | "destroyed" | "destroying" => {
                // Give the log a moment to deliver its last lines.
                if let Some(stream) = logs.as_mut() {
                    let deadline = tokio::time::sleep(Duration::from_secs(3));
                    tokio::pin!(deadline);
                    loop {
                        tokio::select! {
                            _ = &mut deadline => break,
                            chunk = stream.next() => match chunk {
                                Some(Ok(bytes)) => split_lines(&mut partial, &bytes, |l| emit(l, &mut tail)),
                                _ => break,
                            },
                        }
                    }
                }
                if !partial.is_empty() {
                    let rest = std::mem::take(&mut partial);
                    emit(&rest, &mut tail);
                }
                return Err(Box::new(Failure {
                    state,
                    reason: dto.error_reason.clone().unwrap_or_default(),
                    log_tail: tail.into(),
                    timed_out: false,
                    api: None,
                }));
            }
            "pending_build" | "building_snapshot" => {
                built = true;
                if logs.is_none() && !logs_tried {
                    logs_tried = true;
                    logs = open_build_log(api, id).await;
                }
            }
            _ => {}
        }
        if started.elapsed() > limit {
            return Err(Box::new(Failure {
                state,
                reason: format!("still not running after {}s", limit.as_secs()),
                log_tail: tail.into(),
                timed_out: true,
                api: None,
            }));
        }
        // Wait out the poll interval, reading the build log meanwhile.
        let tick = tokio::time::sleep(interval);
        tokio::pin!(tick);
        loop {
            tokio::select! {
                _ = &mut tick => break,
                chunk = async { logs.as_mut().expect("guarded").next().await }, if logs.is_some() => match chunk {
                    Some(Ok(bytes)) => split_lines(&mut partial, &bytes, |l| emit(l, &mut tail)),
                    _ => logs = None,
                },
            }
        }
        interval = (interval * 2).min(Duration::from_secs(2));
    }
}

/// Open the follow stream of a sandbox's build log. `None` if it cannot be
/// opened: the build is still followed by polling, just without its log.
async fn open_build_log(
    api: &Api,
    id: &str,
) -> Option<futures::stream::BoxStream<'static, reqwest::Result<bytes::Bytes>>> {
    let url: UrlDto = api
        .json("getting the build log address", Retry::Safe, |http| {
            Ok(http
                .get(api.url(&format!("/sandbox/{id}/build-logs-url")))
                .timeout(Duration::from_secs(30)))
        })
        .await
        .map_err(|e| tracing::debug!("no build log for {id}: {}", e.message()))
        .ok()?;
    let response = api
        .send("following the build log", Retry::Safe, |http| {
            Ok(http.get(&url.url).query(&[("follow", "true")]))
        })
        .await
        .map_err(|e| tracing::debug!("cannot follow the build log of {id}: {}", e.message()))
        .ok()?;
    Some(response.bytes_stream().boxed())
}

/// Append `bytes` to `partial` and hand every complete line to `line`.
fn split_lines(partial: &mut Vec<u8>, bytes: &[u8], mut line: impl FnMut(&[u8])) {
    partial.extend_from_slice(bytes);
    let mut start = 0;
    while let Some(pos) = partial[start..].iter().position(|b| *b == b'\n') {
        line(&partial[start..start + pos]);
        start += pos + 1;
    }
    partial.drain(..start);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_are_split_across_chunks() {
        let mut partial = Vec::new();
        let mut lines = Vec::new();
        for chunk in ["#1 load\n#2 co", "py /context /\n", "#3 RUN x\n#4 part"] {
            split_lines(&mut partial, chunk.as_bytes(), |l| {
                lines.push(String::from_utf8_lossy(l).into_owned())
            });
        }
        assert_eq!(lines, ["#1 load", "#2 copy /context /", "#3 RUN x"]);
        assert_eq!(partial, b"#4 part");
    }
}
