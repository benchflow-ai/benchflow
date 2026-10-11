//! The local web server behind `bench view` (0.8's `viewer/server.py`).
//!
//! | Request | Trial folder | Job folder or folder of jobs |
//! |---|---|---|
//! | `GET /` | the trial's page, its data inline | the run list page |
//! | `GET /api/rollout?id=<run>[&branch=<fork>/<node>]` | 404 | the run's (or branch child's) payload; 404 JSON for an unknown id |
//! | `GET /compare?runs=<job>,<job>[&status=&task=&agent=]` | 404 | tasks by job with pass rates and paired differences |
//! | `GET /outcomes?run=<job>[,<job>][&status=&task=&agent=]` | 404 | outcome counts, problems by code and owner, time per phase |
//! | `GET /pareto?runs=<job>,<job>[&task=&agent=]` | 404 | pass rate against cost, time, and tokens per trial, with each axis's frontier |
//! | `GET /api/compare?...`, `GET /api/outcomes?...`, `GET /api/pareto?...` | 404 | the same views' data as JSON |
//! | `HEAD /` | 200, empty | 200, empty |
//! | any other `GET` | 404 | 404 |
//! | any other method | 405 | 405 |
//!
//! Every request must name this server in its `Host` header (`localhost`
//! or an IP address, with this port), which defeats DNS rebinding: a web
//! page cannot get a browser to send its own host name here and read the
//! answer. Run ids are honored only by membership in a fresh scan, so no
//! request can read outside the root. Pages are built on request, never at
//! startup, and a large payload is sent while it is being serialized.
//!
//! The run list page is 0.8's, with one addition: links to the compare,
//! outcomes, and Pareto views (new in the Rust viewer; see `views.rs`). Job ids in
//! their queries are matched against the jobs of a fresh scan, like run ids.

use std::convert::Infallible;
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{RawQuery, Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinError, JoinHandle};

use crate::catalog::{Run, discover, discover_jobs, resolve, resolve_jobs, runs_cap};
use crate::page::{browse_boot, single_boot, write_browse_page, write_page, write_view_page};
use crate::payload::{build_branch_child_payload, build_payload};
use crate::py;
use crate::pyjson::{Value, Writer};
use crate::target::{Target, detect, detect_jobs};
use crate::views::{self, Outcome, ViewQuery};

/// With no port given, the first free one from here up is used.
const FIRST_PORT: u16 = 8765;
const PORT_ATTEMPTS: u16 = 100;
/// How long requests in flight may finish after a stop.
const GRACE: Duration = Duration::from_secs(2);

const HTML: &str = "text/html; charset=utf-8";
const JSON: &str = "application/json; charset=utf-8";
const TEXT: &str = "text/plain; charset=utf-8";

/// How to serve.
#[derive(Debug, Clone)]
pub struct ServeOptions {
    /// The address to listen on; `127.0.0.1` keeps the viewer on this machine.
    pub host: IpAddr,
    /// The port; `None` takes the first free port from 8765 up.
    pub port: Option<u16>,
    /// Open the page in the default browser once listening.
    pub open_browser: bool,
}

impl Default for ServeOptions {
    fn default() -> Self {
        ServeOptions {
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: None,
            open_browser: false,
        }
    }
}

/// A running viewer. Dropping it stops the server.
#[derive(Debug)]
pub struct ServerHandle {
    /// Where the page is, e.g. `http://localhost:8765`.
    pub url: String,
    /// The address the server listens on.
    pub addr: SocketAddr,
    /// What is being served.
    pub target: Target,
    shutdown: Option<oneshot::Sender<()>>,
    task: JoinHandle<io::Result<()>>,
}

impl ServerHandle {
    /// Serve until Ctrl-C, then stop (see [`ServerHandle::stop`]).
    pub async fn wait(mut self) -> anyhow::Result<()> {
        tokio::select! {
            signal = tokio::signal::ctrl_c() => signal.context("listening for Ctrl-C")?,
            ended = &mut self.task => return finished(ended),
        }
        self.stop().await
    }

    /// Stop listening, give requests in flight two seconds to finish, then
    /// close what is left.
    pub async fn stop(mut self) -> anyhow::Result<()> {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        match tokio::time::timeout(GRACE, &mut self.task).await {
            Ok(ended) => finished(ended),
            Err(_) => {
                self.task.abort();
                Ok(())
            }
        }
    }
}

impl Drop for ServerHandle {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
    }
}

fn finished(ended: Result<io::Result<()>, JoinError>) -> anyhow::Result<()> {
    ended
        .context("the viewer server failed")?
        .context("the viewer server stopped with an error")
}

/// Serve `root` (a trial folder, a job folder, or a folder of jobs) on a
/// local web server and return once it is listening.
///
/// Nothing is read at startup beyond finding out what `root` is; each page
/// and payload is built from the files when it is requested. The server
/// never writes to the job folders.
pub async fn serve(root: &Path, options: ServeOptions) -> anyhow::Result<ServerHandle> {
    let root = root.to_path_buf();
    let target = tokio::task::spawn_blocking(move || detect(&root))
        .await
        .context("reading the view target")??;
    serve_target(target, options).await
}

/// Serve several job folders side by side ([`Target::Jobs`]); the browser,
/// when opened, opens on the compare view. One path is the same as [`serve`].
pub async fn serve_jobs(paths: &[PathBuf], options: ServeOptions) -> anyhow::Result<ServerHandle> {
    let paths = paths.to_vec();
    let target = tokio::task::spawn_blocking(move || detect_jobs(&paths))
        .await
        .context("reading the view targets")??;
    serve_target(target, options).await
}

async fn serve_target(target: Target, options: ServeOptions) -> anyhow::Result<ServerHandle> {
    let listener = listen(options.host, options.port).await?;
    let addr = listener
        .local_addr()
        .context("reading the listening address")?;
    let url = display_url(options.host, addr.port());
    let router = app(target.clone(), addr.port());
    let page = format!("{url}{}", target.start_path());
    let (shutdown, stopped) = oneshot::channel::<()>();
    let task = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = stopped.await;
            })
            .await
    });
    if options.open_browser
        && let Err(error) = open::that_detached(&page)
    {
        tracing::warn!(%error, %page, "could not open a browser");
    }
    Ok(ServerHandle {
        url,
        addr,
        target,
        shutdown: Some(shutdown),
        task,
    })
}

/// The viewer's routes for `target`, answering only requests addressed to
/// `port` (see the module docs). [`serve`] runs this on a socket; tests and
/// embedders can call it directly.
pub fn app(target: Target, port: u16) -> Router {
    let lists_runs = !matches!(target, Target::Trial { .. });
    let state = Arc::new(AppState { target, port });
    let mut router = Router::new().route("/", get(index).head(head_index));
    if lists_runs {
        router = router
            .route("/api/rollout", get(api_rollout).head(head_not_allowed))
            .route("/compare", get(compare_page).head(head_index))
            .route("/outcomes", get(outcomes_page).head(head_index))
            .route("/api/compare", get(compare_api).head(head_not_allowed))
            .route("/api/outcomes", get(outcomes_api).head(head_not_allowed))
            .route("/pareto", get(pareto_page).head(head_index))
            .route("/api/pareto", get(pareto_api).head(head_not_allowed));
    }
    router
        .fallback(fallback)
        .layer(middleware::from_fn_with_state(state.clone(), check_host))
        .layer(middleware::map_response(protect))
        .with_state(state)
}

#[derive(Debug)]
struct AppState {
    target: Target,
    port: u16,
}

type AppStateRef = State<Arc<AppState>>;

impl AppState {
    /// The trials on offer, from a fresh scan (none for a trial folder).
    fn scan(&self, cap: usize) -> Vec<Run> {
        match &self.target {
            Target::Trial { .. } => Vec::new(),
            Target::Runs { root, .. } => discover(root, cap),
            Target::Jobs { jobs, .. } => discover_jobs(jobs, cap),
        }
    }

    /// The run list's capped scan: the runs and whether there were more.
    fn listed(&self) -> (Vec<Run>, bool) {
        let cap = runs_cap();
        let mut runs = self.scan(cap.saturating_add(1));
        let capped = runs.len() > cap;
        runs.truncate(cap);
        (runs, capped)
    }

    fn resolve(&self, id: &str) -> Option<Run> {
        match &self.target {
            Target::Trial { .. } => None,
            Target::Runs { root, .. } => resolve(root, id),
            Target::Jobs { jobs, .. } => resolve_jobs(jobs, id),
        }
    }

    /// The page title, and the name of trials sitting directly in the root.
    fn title(&self) -> String {
        match &self.target {
            Target::Jobs { jobs, .. } => jobs
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>()
                .join(" vs "),
            target => py::name_of(target.path()),
        }
    }
}

async fn index(State(state): AppStateRef) -> Response {
    match &state.target {
        Target::Trial { dir } => {
            let dir = dir.clone();
            stream(HTML, move |writer| {
                let boot = single_boot(&dir);
                write_page(writer, &py::name_of(&dir), &boot);
            })
        }
        Target::Runs { .. } | Target::Jobs { .. } => stream(HTML, move |writer| {
            let (runs, capped) = state.listed();
            write_browse_page(writer, &state.title(), &browse_boot(&runs, capped));
        }),
    }
}

/// `HEAD /` never touches the files.
async fn head_index() -> Response {
    [(header::CONTENT_TYPE, HTML)].into_response()
}

async fn head_not_allowed() -> Response {
    method_not_allowed()
}

/// What `/api/rollout` found.
enum Lookup {
    Payload(Value),
    UnknownRun,
    UnknownBranch,
}

async fn api_rollout(State(state): AppStateRef, RawQuery(query): RawQuery) -> Response {
    if matches!(state.target, Target::Trial { .. }) {
        return not_found();
    }
    let id = query_value(query.as_deref(), "id");
    let branch = query_value(query.as_deref(), "branch");
    let lookup = tokio::task::spawn_blocking(move || {
        let Some(run) = id.and_then(|id| state.resolve(&id)) else {
            return Lookup::UnknownRun;
        };
        match branch {
            None => Lookup::Payload(build_payload(&run.dir).into_value()),
            // Resolved only through the run's own tree.json.
            Some(child_ref) => build_branch_child_payload(&run.dir, &child_ref)
                .map_or(Lookup::UnknownBranch, |payload| {
                    Lookup::Payload(payload.into_value())
                }),
        }
    })
    .await;
    match lookup {
        Ok(Lookup::Payload(payload)) => stream(JSON, move |writer| writer.value(&payload)),
        Ok(Lookup::UnknownRun) => json_not_found("unknown rollout id"),
        Ok(Lookup::UnknownBranch) => json_not_found("unknown branch child"),
        Err(error) => {
            tracing::error!(%error, "building a payload failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                [(header::CONTENT_TYPE, TEXT)],
                "internal error",
            )
                .into_response()
        }
    }
}

/// Which new view a request is for.
#[derive(Clone, Copy)]
enum View {
    Compare,
    Outcomes,
    Pareto,
}

impl View {
    fn name(self) -> &'static str {
        match self {
            View::Compare => "compare",
            View::Outcomes => "outcomes",
            View::Pareto => "pareto",
        }
    }
}

async fn compare_page(state: AppStateRef, RawQuery(query): RawQuery) -> Response {
    view_response(state, query, View::Compare, true)
}

async fn outcomes_page(state: AppStateRef, RawQuery(query): RawQuery) -> Response {
    view_response(state, query, View::Outcomes, true)
}

async fn pareto_page(state: AppStateRef, RawQuery(query): RawQuery) -> Response {
    view_response(state, query, View::Pareto, true)
}

async fn pareto_api(state: AppStateRef, RawQuery(query): RawQuery) -> Response {
    view_response(state, query, View::Pareto, false)
}

async fn compare_api(state: AppStateRef, RawQuery(query): RawQuery) -> Response {
    view_response(state, query, View::Compare, false)
}

async fn outcomes_api(state: AppStateRef, RawQuery(query): RawQuery) -> Response {
    view_response(state, query, View::Outcomes, false)
}

/// Read every listed trial's result and timing, compute the view, and send
/// it as a page (its data inline) or as JSON.
fn view_response(
    State(state): AppStateRef,
    query: Option<String>,
    view: View,
    page: bool,
) -> Response {
    let query = view_query(query.as_deref());
    stream(if page { HTML } else { JSON }, move |writer| {
        let (runs, capped) = state.listed();
        let trials: Vec<views::Trial> = runs
            .iter()
            .filter(|run| !views::in_progress(run))
            .map(views::read_trial)
            .collect();
        let title = state.title();
        let data = match view {
            View::Compare => views::compare(&trials, &title, capped, &query),
            View::Outcomes => views::outcomes(&trials, &title, capped, &query),
            View::Pareto => views::pareto(&trials, &title, capped, &query),
        };
        if page {
            write_view_page(
                writer,
                view.name(),
                &format!("{} - {title}", view.name()),
                &data,
            );
        } else {
            writer.value(&data);
        }
    })
}

/// The jobs and filters of a compare, outcomes, or Pareto URL. `runs` and `run`
/// both name jobs, comma-separated, each one percent-encoded on its own.
fn view_query(query: Option<&str>) -> ViewQuery {
    let mut jobs = query_list(query, "runs");
    jobs.extend(query_list(query, "run"));
    ViewQuery {
        jobs,
        status: query_list(query, "status")
            .iter()
            .filter_map(|status| Outcome::parse(status))
            .collect(),
        task: query_value(query, "task").unwrap_or_default(),
        agent: query_value(query, "agent").unwrap_or_default(),
    }
}

/// The first non-empty value of `key`, split at its literal commas before
/// decoding (so an encoded comma stays inside one item); empty items dropped.
fn query_list(query: Option<&str>, key: &str) -> Vec<String> {
    let Some(raw) = query.and_then(|query| {
        query.split('&').find_map(|pair| {
            let (name, value) = pair.split_once('=')?;
            (!value.is_empty() && percent_decode(name) == key).then_some(value)
        })
    }) else {
        return Vec::new();
    };
    raw.split(',')
        .map(percent_decode)
        .filter(|item| !item.is_empty())
        .collect()
}

async fn fallback(method: Method) -> Response {
    if method == Method::GET {
        not_found()
    } else {
        method_not_allowed()
    }
}

fn not_found() -> Response {
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, TEXT)],
        "not found",
    )
        .into_response()
}

fn method_not_allowed() -> Response {
    (StatusCode::METHOD_NOT_ALLOWED, [(header::ALLOW, "GET")]).into_response()
}

fn json_not_found(message: &str) -> Response {
    (
        StatusCode::NOT_FOUND,
        [(header::CONTENT_TYPE, JSON)],
        format!("{{\"error\": \"{message}\"}}"),
    )
        .into_response()
}

/// Refuse a request that does not name this server in exactly one `Host` header.
async fn check_host(State(state): AppStateRef, request: Request, next: Next) -> Response {
    let mut hosts = request.headers().get_all(header::HOST).iter();
    let allowed = match (hosts.next(), hosts.next()) {
        (Some(host), None) => host
            .to_str()
            .is_ok_and(|host| host_allowed(host, state.port)),
        _ => false,
    };
    if allowed {
        return next.run(request).await;
    }
    let body = format!(
        "This viewer only answers at http://localhost:{}/ (or an IP address of this machine); \
         other host names are refused.\n",
        state.port
    );
    (StatusCode::FORBIDDEN, [(header::CONTENT_TYPE, TEXT)], body).into_response()
}

/// Whether a `Host` value names this server: `localhost` or an IP address,
/// with this port (80 when none is given). A DNS-rebinding attack always
/// arrives with the attacker's host name, never with one of these.
fn host_allowed(value: &str, port: u16) -> bool {
    if value.is_empty() || value.contains(['/', '?', '#', '@', '\\', ' ']) {
        return false;
    }
    let port_text = if let Some(rest) = value.strip_prefix('[') {
        let Some((address, after)) = rest.split_once(']') else {
            return false;
        };
        if address.parse::<Ipv6Addr>().is_err() {
            return false;
        }
        match after {
            "" => None,
            after => match after.strip_prefix(':') {
                Some(port_text) => Some(port_text),
                None => return false,
            },
        }
    } else {
        let (name, port_text) = match value.rsplit_once(':') {
            Some((name, port_text)) => (name, Some(port_text)),
            None => (value, None),
        };
        if !name.eq_ignore_ascii_case("localhost") && name.parse::<Ipv4Addr>().is_err() {
            return false;
        }
        port_text
    };
    let given = match port_text {
        None => 80,
        Some(text) if !text.is_empty() && text.bytes().all(|byte| byte.is_ascii_digit()) => {
            match text.parse::<u32>() {
                Ok(given) => given,
                Err(_) => return false,
            }
        }
        Some(_) => return false,
    };
    given == u32::from(port)
}

/// Headers every answer carries: never cached, never framed, no referrer.
async fn protect(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("frame-ancestors 'none'"),
    );
    response
}

/// Run `write` on a blocking thread and send its output as it is written,
/// so a large page never sits in memory whole. Writing stops early if the
/// client goes away.
fn stream<F>(content_type: &'static str, write: F) -> Response
where
    F: FnOnce(&mut Writer<'_>) + Send + 'static,
{
    let (sender, mut receiver) = mpsc::channel::<Bytes>(4);
    tokio::task::spawn_blocking(move || {
        let mut flush = |chunk: String| sender.blocking_send(Bytes::from(chunk)).is_ok();
        let mut writer = Writer::streaming(&mut flush);
        write(&mut writer);
        writer.finish();
    });
    let chunks = futures::stream::poll_fn(move |context| {
        receiver
            .poll_recv(context)
            .map(|chunk| chunk.map(Ok::<_, Infallible>))
    });
    (
        [(header::CONTENT_TYPE, content_type)],
        Body::from_stream(chunks),
    )
        .into_response()
}

/// The first non-empty value of `key` in a query string, decoded like
/// Python's `parse_qs`: `+` is a space and bad UTF-8 becomes U+FFFD.
fn query_value(query: Option<&str>, key: &str) -> Option<String> {
    query?.split('&').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        (!value.is_empty() && percent_decode(name) == key).then(|| percent_decode(value))
    })
}

fn percent_decode(text: &str) -> String {
    let hex = |byte: Option<&u8>| {
        byte.and_then(|byte| (*byte as char).to_digit(16))
            .map(|digit| digit as u8)
    };
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                out.push(b' ');
                index += 1;
            }
            b'%' => match (hex(bytes.get(index + 1)), hex(bytes.get(index + 2))) {
                (Some(high), Some(low)) => {
                    out.push(high * 16 + low);
                    index += 3;
                }
                _ => {
                    out.push(b'%');
                    index += 1;
                }
            },
            byte => {
                out.push(byte);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A listening socket: `port`, or the first free port from 8765 up.
async fn listen(host: IpAddr, port: Option<u16>) -> anyhow::Result<TcpListener> {
    if let Some(port) = port {
        let addr = SocketAddr::new(host, port);
        return TcpListener::bind(addr)
            .await
            .with_context(|| format!("could not listen on {addr}"));
    }
    let last = FIRST_PORT + (PORT_ATTEMPTS - 1);
    for port in FIRST_PORT..=last {
        if answers(host, port).await {
            continue;
        }
        if let Ok(listener) = TcpListener::bind(SocketAddr::new(host, port)).await {
            return Ok(listener);
        }
    }
    anyhow::bail!("no free port on {host} from {FIRST_PORT} to {last}; pass a port")
}

/// Whether something already listens on `port`. macOS lets a server bind
/// 127.0.0.1 beside another one bound to 0.0.0.0, which would hide it.
async fn answers(host: IpAddr, port: u16) -> bool {
    let probe = match host {
        IpAddr::V4(address) if address.is_unspecified() => IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(address) if address.is_unspecified() => IpAddr::V6(Ipv6Addr::LOCALHOST),
        address => address,
    };
    let connect = TcpStream::connect(SocketAddr::new(probe, port));
    matches!(
        tokio::time::timeout(Duration::from_millis(250), connect).await,
        Ok(Ok(_))
    )
}

/// The address to print: `localhost` for a local or all-interfaces server.
fn display_url(host: IpAddr, port: u16) -> String {
    match host {
        host if host.is_loopback() || host.is_unspecified() => format!("http://localhost:{port}"),
        IpAddr::V4(address) => format!("http://{address}:{port}"),
        IpAddr::V6(address) => format!("http://[{address}]:{port}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_header_names_this_server() {
        for (value, ok) in [
            ("localhost:8765", true),
            ("LOCALHOST:8765", true),
            ("127.0.0.1:8765", true),
            ("[::1]:8765", true),
            ("192.168.1.9:8765", true),
            ("localhost:8766", false),
            ("localhost", false),
            ("localhost:", false),
            ("attacker.example:8765", false),
            ("localhost.:8765", false),
            ("user@localhost:8765", false),
            ("localhost:8765/x", false),
            ("[::1]8765", false),
            ("[nope]:8765", false),
            ("", false),
        ] {
            assert_eq!(host_allowed(value, 8765), ok, "{value}");
        }
        assert!(host_allowed("localhost", 80));
    }

    #[test]
    fn query_values_decode_like_parse_qs() {
        let query = Some("id=job%2Fa__1&branch=&x=1&id=second&flag");
        assert_eq!(query_value(query, "id").as_deref(), Some("job/a__1"));
        assert_eq!(query_value(query, "branch"), None);
        assert_eq!(
            query_value(Some("id=a+b%2Bc%zz"), "id").as_deref(),
            Some("a b+c%zz")
        );
        assert_eq!(
            query_value(Some("id=%E2%82"), "id").as_deref(),
            Some("\u{fffd}")
        );
        assert_eq!(query_value(None, "id"), None);
    }

    #[test]
    fn view_queries_split_jobs_before_decoding() {
        let query = view_query(Some(
            "runs=job%2Ca,jobs/b%2F2026&run=c&status=failed,nope,errored&task=Alpha+x&agent=oracle",
        ));
        assert_eq!(query.jobs, ["job,a", "jobs/b/2026", "c"]);
        assert_eq!(
            query.status.into_iter().collect::<Vec<_>>(),
            [Outcome::Failed, Outcome::Errored]
        );
        assert_eq!(query.task, "Alpha x");
        assert_eq!(query.agent, "oracle");
        assert_eq!(view_query(None), ViewQuery::default());
        assert_eq!(query_list(Some("runs=,,a,"), "runs"), ["a"]);
    }

    #[test]
    fn urls_name_localhost_for_local_servers() {
        assert_eq!(
            display_url(IpAddr::V4(Ipv4Addr::LOCALHOST), 8765),
            "http://localhost:8765"
        );
        assert_eq!(
            display_url(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 8765),
            "http://localhost:8765"
        );
        assert_eq!(
            display_url("10.0.0.2".parse().unwrap(), 9000),
            "http://10.0.0.2:9000"
        );
        assert_eq!(
            display_url("fe80::1".parse().unwrap(), 9000),
            "http://[fe80::1]:9000"
        );
    }
}
