//! The forwarder's JavaScript, run with Node.js against a fake provider on
//! loopback. Skipped, with a note, when no `node` is on `PATH`.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::process::{Child, Command};

use crate::providers::FORWARDER_JS;

const KEY: &str = "sk-fake-provider-key-0123456789abcdefXYZ";

fn node() -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join("node"))
        .find(|path| path.is_file())
}

/// One request the fake provider got.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_str())
    }
}

type SeenLog = Arc<Mutex<Vec<Seen>>>;

/// A fake provider: `/…/echo` answers with the request's headers, `/…/sse`
/// sends two events half a second apart, `/…/status/401` refuses,
/// `/…/reset-once` drops the connection the first time without answering,
/// anything else answers `{"ok": true}`.
async fn fake_provider() -> (u16, SeenLog) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let seen: SeenLog = Arc::default();
    let log = seen.clone();
    let dropped: Arc<Mutex<std::collections::HashSet<String>>> = Arc::default();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            tokio::spawn(serve(stream, log.clone(), dropped.clone()));
        }
    });
    (port, seen)
}

async fn serve(
    stream: TcpStream,
    log: SeenLog,
    dropped: Arc<Mutex<std::collections::HashSet<String>>>,
) {
    let mut reader = BufReader::new(stream);
    loop {
        let mut request_line = String::new();
        if reader.read_line(&mut request_line).await.unwrap_or(0) == 0 {
            return;
        }
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or_default().to_string();
        let path = parts.next().unwrap_or_default().to_string();
        let mut headers = Vec::new();
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).await.unwrap_or(0) == 0 {
                return;
            }
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            if let Some((name, value)) = line.split_once(':') {
                let name = name.trim().to_ascii_lowercase();
                let value = value.trim().to_string();
                if name == "content-length" {
                    length = value.parse().unwrap_or(0);
                }
                headers.push((name, value));
            }
        }
        let mut body = vec![0; length];
        if reader.read_exact(&mut body).await.is_err() {
            return;
        }
        let seen = Seen {
            method,
            path: path.clone(),
            headers: headers.clone(),
            body,
        };
        log.lock().unwrap().push(seen);
        if path.ends_with("/reset-once") && dropped.lock().unwrap().insert(path.clone()) {
            // Close without an answer.
            return;
        }
        let stream = reader.get_mut();
        if let Some((_, body)) = RECORDED
            .iter()
            .find(|(name, _)| path.ends_with(&format!("/recorded/{name}")))
        {
            let kind = if path.ends_with(".sse") {
                "text/event-stream"
            } else {
                "application/json"
            };
            let head = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: {kind}\r\ncontent-length: {}\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(head.as_bytes()).await;
            let _ = stream.write_all(body.as_bytes()).await;
            continue;
        }
        if path.ends_with("/sse") {
            let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\nconnection: close\r\n\r\n";
            let _ = stream.write_all(head.as_bytes()).await;
            let _ = stream.write_all(b"data: one\n\n").await;
            let _ = stream.flush().await;
            tokio::time::sleep(Duration::from_millis(500)).await;
            let _ = stream.write_all(b"data: two\n\n").await;
            let _ = stream.shutdown().await;
            return;
        }
        // `/…/header-echo` sends the request's authorization back in a
        // response header.
        let extra = if path.ends_with("/header-echo") {
            let auth = headers
                .iter()
                .find(|(k, _)| k == "authorization")
                .map(|(_, v)| v.clone())
                .unwrap_or_default();
            format!("x-echo-auth: {auth}\r\n")
        } else {
            String::new()
        };
        let (status, body) = if path.ends_with("/echo") {
            let map: serde_json::Map<String, Value> = headers
                .into_iter()
                .map(|(k, v)| (k, Value::String(v)))
                .collect();
            ("200 OK", Value::Object(map).to_string())
        } else if path.ends_with("/status/401") {
            ("401 Unauthorized", r#"{"error": "bad key"}"#.to_string())
        } else {
            ("200 OK", r#"{"ok": true}"#.to_string())
        };
        let answer = format!(
            "HTTP/1.1 {status}\r\ncontent-type: application/json\r\n{extra}content-length: {}\r\n\r\n{body}",
            body.len()
        );
        if stream.write_all(answer.as_bytes()).await.is_err() {
            return;
        }
    }
}

/// Answers as providers sent them: the Vercel AI Gateway's (recorded
/// 2026-10-10 on deepseek/deepseek-v4-flash-0731, ids and routing details
/// cut), and, written from the providers' documented shapes, Anthropic's API
/// with prompt caching, OpenRouter's, and the AI SDK's own protocol (fx's).
const RECORDED: &[(&str, &str)] = &[
    (
        "gateway-chat.json",
        include_str!("recorded/gateway-chat.json"),
    ),
    (
        "gateway-chat.sse",
        include_str!("recorded/gateway-chat.sse"),
    ),
    (
        "gateway-messages.json",
        include_str!("recorded/gateway-messages.json"),
    ),
    (
        "gateway-messages.sse",
        include_str!("recorded/gateway-messages.sse"),
    ),
    (
        "gateway-responses.sse",
        include_str!("recorded/gateway-responses.sse"),
    ),
    (
        "anthropic-cached.sse",
        include_str!("recorded/anthropic-cached.sse"),
    ),
    (
        "openrouter-chat.json",
        include_str!("recorded/openrouter-chat.json"),
    ),
    (
        "ai-sdk-language-model.sse",
        include_str!("recorded/ai-sdk-language-model.sse"),
    ),
];

/// A forwarder process with its port and its output so far.
struct Running {
    child: Child,
    port: u16,
    stderr: Arc<Mutex<String>>,
}

impl Running {
    fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    async fn stop(mut self) -> String {
        let _ = self.child.kill().await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        self.stderr.lock().unwrap().clone()
    }
}

/// Write the forwarder, its config and a key file (with `key_mode`) into
/// `dir`, and start it with an empty environment.
async fn start(dir: &Path, routes: Value, key_mode: u32) -> Result<Running, (i32, String)> {
    let node = node().expect("node");
    std::fs::write(dir.join("forwarder.js"), FORWARDER_JS).unwrap();
    let key_file = dir.join("key");
    std::fs::write(&key_file, KEY).unwrap();
    std::fs::set_permissions(&key_file, std::fs::Permissions::from_mode(key_mode)).unwrap();
    let config = json!({"listen": "127.0.0.1", "port": 0, "key_file": key_file, "routes": routes});
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let mut child = Command::new(node)
        .arg(dir.join("forwarder.js"))
        .arg(dir.join("config.json"))
        .env_clear()
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let stderr = Arc::new(Mutex::new(String::new()));
    let mut err_lines = BufReader::new(child.stderr.take().unwrap()).lines();
    let keep = stderr.clone();
    tokio::spawn(async move {
        while let Ok(Some(line)) = err_lines.next_line().await {
            let mut s = keep.lock().unwrap();
            s.push_str(&line);
            s.push('\n');
        }
    });
    let mut out_lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let first = tokio::time::timeout(Duration::from_secs(20), out_lines.next_line())
        .await
        .expect("the forwarder answered in time");
    match first {
        Ok(Some(line)) => {
            assert!(!line.contains(KEY));
            let ready: Value = serde_json::from_str(&line).unwrap();
            assert_eq!(ready["benchflow_forwarder"], "ready");
            let port = ready["port"].as_u64().unwrap() as u16;
            Ok(Running {
                child,
                port,
                stderr,
            })
        }
        _ => {
            let status = child.wait().await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
            let err = stderr.lock().unwrap().clone();
            Err((status.code().unwrap_or(-1), err))
        }
    }
}

/// A raw HTTP/1.1 request, for targets an HTTP client would rewrite.
async fn raw(port: u16, request_line: &str) -> String {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let request = format!("{request_line}\r\nhost: 127.0.0.1\r\nconnection: close\r\n\r\n");
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut answer = String::new();
    let _ = tokio::time::timeout(Duration::from_secs(10), stream.read_to_string(&mut answer)).await;
    answer
}

fn routes(upstream_port: u16) -> Value {
    json!([
        {"prefix": "/anthropic", "upstream": format!("http://127.0.0.1:{upstream_port}/claude-code"), "auth": "bearer"},
        {"prefix": "/openai", "upstream": format!("http://127.0.0.1:{upstream_port}/v1"), "auth": "x-api-key"},
    ])
}

#[tokio::test]
async fn the_forwarder_adds_the_key_streams_and_serves_nothing_else() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let fwd = start(dir.path(), routes(upstream), 0o600).await.unwrap();
    // The key file is gone once read.
    assert!(!dir.path().join("key").exists());
    let client = reqwest::Client::new();

    // Bearer route: the placeholder headers are replaced by the key.
    let answer = client
        .post(fwd.url("/anthropic/v1/messages?beta=true"))
        .header("authorization", "Bearer benchflow-forwarder-holds-the-key")
        .header("x-api-key", "benchflow-forwarder-holds-the-key")
        .header("anthropic-version", "2023-06-01")
        .body(r#"{"model": "m", "max_tokens": 1}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    assert_eq!(answer.text().await.unwrap(), r#"{"ok": true}"#);
    {
        let seen = seen.lock().unwrap();
        let request = seen.last().unwrap();
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/claude-code/v1/messages?beta=true");
        assert_eq!(
            request.header("authorization"),
            Some(format!("Bearer {KEY}").as_str())
        );
        assert_eq!(request.header("x-api-key"), None);
        assert_eq!(request.header("anthropic-version"), Some("2023-06-01"));
        assert_eq!(request.header("accept-encoding"), Some("identity"));
        assert_eq!(
            request.header("host"),
            Some(format!("127.0.0.1:{upstream}").as_str())
        );
        assert_eq!(request.body, br#"{"model": "m", "max_tokens": 1}"#);
    }

    // x-api-key route: the key goes in x-api-key, no bearer token.
    let answer = client
        .get(fwd.url("/openai/models"))
        .bearer_auth("benchflow-forwarder-holds-the-key")
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    {
        let seen = seen.lock().unwrap();
        let request = seen.last().unwrap();
        assert_eq!(request.path, "/v1/models");
        assert_eq!(request.header("x-api-key"), Some(KEY));
        assert_eq!(request.header("authorization"), None);
    }

    // An answer that echoes the key gets it masked, at the same length.
    let answer = client.get(fwd.url("/anthropic/echo")).send().await.unwrap();
    let length = answer.content_length();
    let text = answer.text().await.unwrap();
    assert!(!text.contains(KEY), "{text}");
    assert!(text.contains(&"*".repeat(KEY.len())), "{text}");
    assert_eq!(length, Some(text.len() as u64));

    // Server-sent events arrive as they are sent, not when the answer ends.
    let started = Instant::now();
    let mut answer = client.get(fwd.url("/anthropic/sse")).send().await.unwrap();
    assert_eq!(answer.headers()["content-type"], "text/event-stream");
    let first = answer.chunk().await.unwrap().unwrap();
    assert_eq!(&first[..], b"data: one\n\n");
    assert!(
        started.elapsed() < Duration::from_millis(400),
        "the first event took {:?}",
        started.elapsed()
    );
    let mut rest = Vec::new();
    while let Some(chunk) = answer.chunk().await.unwrap() {
        rest.extend_from_slice(&chunk);
    }
    assert_eq!(rest, b"data: two\n\n");

    // The provider's refusal is passed on as it is.
    let answer = client
        .get(fwd.url("/anthropic/status/401"))
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 401);

    // Nothing outside the routes is served, and none of it reaches the provider.
    let before = seen.lock().unwrap().len();
    for (request_line, status) in [
        ("GET /other/v1/models HTTP/1.1", "404"),
        ("GET / HTTP/1.1", "404"),
        ("GET /anthropicx/v1 HTTP/1.1", "404"),
        ("GET /anthropic/../v1/credits HTTP/1.1", "400"),
        ("GET /anthropic/%2e%2e/v1/credits HTTP/1.1", "400"),
        ("GET /anthropic/v1/%2E%2E/%2e%2e/x HTTP/1.1", "400"),
        ("GET /anthropic/v1\\..\\x HTTP/1.1", "400"),
        ("GET http://example.com/anthropic/v1 HTTP/1.1", "400"),
    ] {
        let answer = raw(fwd.port, request_line).await;
        assert!(
            answer.starts_with(&format!("HTTP/1.1 {status}")),
            "{request_line}: {answer}"
        );
        assert!(!answer.contains(KEY));
    }
    assert_eq!(seen.lock().unwrap().len(), before);

    // A WebSocket upgrade is refused.
    let mut stream = TcpStream::connect(("127.0.0.1", fwd.port)).await.unwrap();
    stream
        .write_all(b"GET /openai/responses HTTP/1.1\r\nhost: x\r\nconnection: upgrade\r\nupgrade: websocket\r\n\r\n")
        .await
        .unwrap();
    let mut answer = String::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), stream.read_to_string(&mut answer)).await;
    assert!(answer.starts_with("HTTP/1.1 501"), "{answer}");

    let log = fwd.stop().await;
    assert!(!log.contains(KEY), "{log}");
    // One line per request, with its path and status.
    assert!(
        log.contains(r#""path":"/anthropic/v1/messages","status":200"#),
        "{log}"
    );
}

#[tokio::test]
async fn an_unreachable_provider_is_a_502_without_the_key() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    // A port nothing listens on.
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let dir = tempfile::tempdir().unwrap();
    let fwd = start(dir.path(), routes(closed), 0o600).await.unwrap();
    let answer = reqwest::get(fwd.url("/anthropic/v1/messages"))
        .await
        .unwrap();
    assert_eq!(answer.status(), 502);
    let body = answer.text().await.unwrap();
    assert!(body.contains("the provider could not be reached"), "{body}");
    assert!(!body.contains(KEY));
    let log = fwd.stop().await;
    assert!(!log.contains(KEY));
}

#[tokio::test]
async fn a_key_file_others_can_read_is_refused() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let Err((code, stderr)) = start(dir.path(), routes(1), 0o644).await else {
        panic!("the forwarder started with a key file others can read")
    };
    assert_eq!(code, 2);
    assert!(stderr.contains("mode 600"), "{stderr}");
    assert!(!stderr.contains(KEY));
    // Refused before it was read: the file is left for whoever wrote it.
    assert!(dir.path().join("key").exists());
}

/// The key check sends the key to the provider's own address, in the
/// provider's header, and reads only the status.
#[tokio::test]
async fn the_key_check_reads_only_the_status() {
    use crate::providers::{
        Auth, Check, CheckOutcome, Endpoint, KeySource, OpenAiApi, Protocol, Provider, ProviderKey,
        check_key,
    };
    let (port, seen) = fake_provider().await;
    let provider = |path: &str, auth: Auth| Provider {
        name: "local".into(),
        description: String::new(),
        openai: Some(Endpoint {
            protocol: Protocol::OpenAi,
            base_url: format!("http://127.0.0.1:{port}/v1"),
            auth,
            apis: vec![OpenAiApi::Chat],
        }),
        anthropic: None,
        gateway: None,
        key: KeySource {
            env: vec!["LOCAL_KEY".into()],
            files: Vec::new(),
        },
        check: Some(Check {
            endpoint: Protocol::OpenAi,
            path: path.into(),
            headers: [("x-check".to_string(), "1".to_string())]
                .into_iter()
                .collect(),
        }),
        routing: None,
        source: crate::manifest::ManifestSource::BuiltinProvider("local.toml"),
    };
    let key = ProviderKey::for_tests("local", KEY);
    let timeout = Duration::from_secs(10);
    assert_eq!(
        check_key(&provider("/credits", Auth::Bearer), &key, timeout).await,
        CheckOutcome::Accepted
    );
    {
        let seen = seen.lock().unwrap();
        let request = seen.last().unwrap();
        assert_eq!(
            (request.method.as_str(), request.path.as_str()),
            ("GET", "/v1/credits")
        );
        assert_eq!(
            request.header("authorization"),
            Some(format!("Bearer {KEY}").as_str())
        );
        assert_eq!(request.header("x-check"), Some("1"));
    }
    assert_eq!(
        check_key(&provider("/status/401", Auth::XApiKey), &key, timeout).await,
        CheckOutcome::Rejected { status: 401 }
    );
    assert_eq!(
        seen.lock().unwrap().last().unwrap().header("x-api-key"),
        Some(KEY)
    );
    let mut none = provider("/credits", Auth::Bearer);
    none.check = None;
    assert_eq!(check_key(&none, &key, timeout).await, CheckOutcome::NoCheck);
    let closed = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap().port()
    };
    let mut unreachable = provider("/credits", Auth::Bearer);
    unreachable.openai.as_mut().unwrap().base_url = format!("http://127.0.0.1:{closed}/v1");
    let outcome = check_key(&unreachable, &key, timeout).await;
    assert!(
        matches!(outcome, CheckOutcome::Inconclusive { .. }),
        "{outcome:?}"
    );
    assert!(!outcome.describe().contains(KEY));
}

#[tokio::test]
async fn a_dropped_connection_is_retried_only_when_that_is_safe() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let count = |path: &str| {
        seen.lock()
            .unwrap()
            .iter()
            .filter(|s| s.path.ends_with(path))
            .count()
    };

    // A POST on a fresh connection may have reached the provider: it is not
    // sent again, and the agent's own client decides (a 502 it retries).
    let dir = tempfile::tempdir().unwrap();
    let fwd = start(dir.path(), routes(upstream), 0o600).await.unwrap();
    let client = reqwest::Client::new();
    let answer = client
        .post(fwd.url("/anthropic/v1/post/reset-once"))
        .body(r#"{"model": "m"}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 502);
    assert_eq!(count("/v1/post/reset-once"), 1);

    // A GET is sent again, with the same headers.
    let answer = client
        .get(fwd.url("/anthropic/v1/get/reset-once"))
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    assert_eq!(count("/v1/get/reset-once"), 2);
    let log = fwd.stop().await;
    assert!(log.contains("retrying after ECONNRESET"), "{log}");
    assert!(!log.contains(KEY));
}

#[tokio::test]
async fn only_the_runs_model_is_served() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let routes = json!([{
        "prefix": "/openai",
        "upstream": format!("http://127.0.0.1:{upstream}/v1"),
        "auth": "bearer",
        "models": ["deepseek/deepseek-v4-flash-0731"],
    }]);
    let fwd = start(dir.path(), routes, 0o600).await.unwrap();
    let client = reqwest::Client::new();
    let post = |model: &str| {
        client
            .post(fwd.url("/openai/responses"))
            .header("content-type", "application/json")
            .body(json!({"model": model, "input": "hi"}).to_string())
            .send()
    };
    assert_eq!(
        post("deepseek/deepseek-v4-flash-0731")
            .await
            .unwrap()
            .status(),
        200
    );
    // A variant suffix is the same model.
    assert_eq!(
        post("deepseek/deepseek-v4-flash-0731[1m]")
            .await
            .unwrap()
            .status(),
        200
    );
    // Another model never reaches the provider.
    let before = seen.lock().unwrap().len();
    let refused = post("gpt-5.6-luna").await.unwrap();
    assert_eq!(refused.status(), 403);
    let text = refused.text().await.unwrap();
    assert!(
        text.contains("this run may use only deepseek/deepseek-v4-flash-0731; the request asked for gpt-5.6-luna"),
        "{text}"
    );
    assert_eq!(seen.lock().unwrap().len(), before);
    // A request that names no model (a model list) passes; a body that is
    // not JSON cannot be checked, so it is refused.
    assert_eq!(
        client
            .get(fwd.url("/openai/models"))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let seen_before = seen.lock().unwrap().len();
    assert_eq!(
        client
            .post(fwd.url("/openai/files"))
            .body("plain text")
            .send()
            .await
            .unwrap()
            .status(),
        415
    );
    assert_eq!(seen.lock().unwrap().len(), seen_before);
    // Other models named elsewhere in the body (a router's fallback list, a
    // message batch, a model that is not a string), and batches whose
    // models sit in a file, are refused too.
    let send = |path: &str, body: String| {
        client
            .post(fwd.url(path))
            .header("content-type", "application/json")
            .body(body)
            .send()
    };
    let pinned = "deepseek/deepseek-v4-flash-0731";
    for body in [
        json!({"model": pinned, "models": [pinned, "gpt-5.6-luna"]}),
        json!({"model": pinned, "providerOptions": {"gateway": {"models": ["gpt-5.6-luna"]}}}),
        json!({"requests": [{"custom_id": "a", "params": {"model": "gpt-5.6-luna"}}]}),
        json!({"model": [pinned]}),
    ] {
        let answer = send("/openai/responses", body.to_string()).await.unwrap();
        assert_eq!(answer.status(), 403, "{body}");
    }
    let answer = send(
        "/openai/batches",
        json!({"input_file_id": "file-1"}).to_string(),
    )
    .await
    .unwrap();
    assert_eq!(answer.status(), 403);
    assert_eq!(seen.lock().unwrap().len(), seen_before);
    // A key given twice is sent as the forwarder read it.
    let answer = send(
        "/openai/responses",
        format!(r#"{{"model":"gpt-5.6-luna","input":"hi","model":"{pinned}"}}"#),
    )
    .await
    .unwrap();
    assert_eq!(answer.status(), 200);
    let sent = String::from_utf8_lossy(&seen.lock().unwrap().last().unwrap().body).into_owned();
    assert!(!sent.contains("gpt-5.6-luna"), "{sent}");
    let log = fwd.stop().await;
    assert!(log.contains("the request asked for gpt-5.6-luna"), "{log}");
}

/// On a gateway, every JSON body is pinned to the allowed upstreams: merged
/// into the agent's `providerOptions`, replacing any `only` it set.
#[tokio::test]
async fn gateway_requests_are_pinned_to_the_allowed_upstreams() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let routes = json!([{
        "prefix": "/anthropic",
        "upstream": format!("http://127.0.0.1:{upstream}/claude-code"),
        "auth": "bearer",
        "models": ["deepseek/deepseek-v4-flash-0731"],
        "only": ["alibaba", "novita"],
    }]);
    let fwd = start(dir.path(), routes, 0o600).await.unwrap();
    let client = reqwest::Client::new();
    let last_body =
        || -> Value { serde_json::from_slice(&seen.lock().unwrap().last().unwrap().body).unwrap() };

    // The agent's own routing wish loses its `only`, keeps the rest.
    let answer = client
        .post(fwd.url("/anthropic/v1/messages"))
        .header("content-type", "application/json")
        .body(
            json!({
                "model": "deepseek/deepseek-v4-flash-0731",
                "max_tokens": 8,
                "providerOptions": {
                    "gateway": {"only": ["baseten"], "order": ["novita"]},
                    "anthropic": {"cacheControl": true},
                },
            })
            .to_string(),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    let body = last_body();
    assert_eq!(
        body["providerOptions"],
        json!({"gateway": {"only": ["alibaba", "novita"], "order": ["novita"]}, "anthropic": {"cacheControl": true}})
    );
    assert_eq!(body["max_tokens"], 8);
    let content_length: usize = seen
        .lock()
        .unwrap()
        .last()
        .unwrap()
        .header("content-length")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(content_length, serde_json::to_vec(&body).unwrap().len());

    // A body without providerOptions gets them; so does one with no model
    // (count_tokens, say).
    client
        .post(fwd.url("/anthropic/v1/messages/count_tokens"))
        .body(json!({"messages": []}).to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(
        last_body()["providerOptions"],
        json!({"gateway": {"only": ["alibaba", "novita"]}})
    );

    // A body that is not a JSON object cannot be pinned, so it is refused.
    let refused = client
        .post(fwd.url("/anthropic/v1/files"))
        .body("plain text")
        .send()
        .await
        .unwrap();
    assert_eq!(refused.status(), 415);
    assert!(last_body().get("providerOptions").is_some());
    let log = fwd.stop().await;
    assert!(!log.contains(KEY));
}

#[tokio::test]
async fn a_route_with_an_empty_upstream_list_is_refused() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let routes = json!([{"prefix": "/openai", "upstream": "https://x.example/v1", "auth": "bearer", "only": []}]);
    let Err((code, stderr)) = start(dir.path(), routes, 0o600).await else {
        panic!("the forwarder started with nowhere allowed to send requests")
    };
    assert_eq!(code, 2);
    assert!(stderr.contains("must be a non-empty list"), "{stderr}");
}

/// On OpenRouter, every JSON body names the denied upstreams in
/// `provider.ignore` (its documented provider-routing field), keeping the
/// agent's own preferences and the entries it already ignores.
#[tokio::test]
async fn openrouter_requests_ignore_the_denied_upstreams() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let routes = json!([{
        "prefix": "/openai",
        "upstream": format!("http://127.0.0.1:{upstream}/api/v1"),
        "auth": "bearer",
        "models": ["deepseek/deepseek-chat"],
        "ignore": ["baseten", "together"],
    }]);
    let fwd = start(dir.path(), routes, 0o600).await.unwrap();
    let client = reqwest::Client::new();
    let last = || seen.lock().unwrap().last().unwrap().clone();
    let post = |body: Value| {
        client
            .post(fwd.url("/openai/chat/completions"))
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
    };

    let answer = post(json!({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "hi"}],
        "provider": {"order": ["novita"], "allow_fallbacks": false, "ignore": ["fireworks", "baseten"]},
    }))
    .await
    .unwrap();
    assert_eq!(answer.status(), 200);
    let seen_now = last();
    assert_eq!(seen_now.path, "/api/v1/chat/completions");
    let body: Value = serde_json::from_slice(&seen_now.body).unwrap();
    assert_eq!(
        body["provider"],
        json!({"order": ["novita"], "allow_fallbacks": false, "ignore": ["fireworks", "baseten", "together"]})
    );
    assert!(body.get("providerOptions").is_none());

    post(json!({"model": "deepseek/deepseek-chat", "messages": []}))
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&last().body).unwrap();
    assert_eq!(body["provider"], json!({"ignore": ["baseten", "together"]}));
    let log = fwd.stop().await;
    assert!(log.contains(r#""pinned":"ignore""#), "{log}");
    assert!(!log.contains(KEY));
}

/// A gateway's own API (fx's) is served from the root, its paths passed on
/// as they are, and the model it names in headers is held to the run's.
#[tokio::test]
async fn a_gateway_api_is_served_from_the_root_with_its_model_headers_checked() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let routes = json!([{
        "prefix": "",
        "upstream": format!("http://127.0.0.1:{upstream}"),
        "auth": "bearer",
        "models": ["deepseek/deepseek-v4-flash-0731"],
        "only": ["alibaba", "novita"],
    }]);
    let fwd = start(dir.path(), routes, 0o600).await.unwrap();
    let client = reqwest::Client::new();
    let call = |header: &'static str, model: &'static str| {
        client
            .post(fwd.url("/v4/ai/language-model"))
            .header("content-type", "application/json")
            .header("authorization", "Bearer benchflow-forwarder-holds-the-key")
            .header(header, model)
            .body(
                json!({"prompt": [], "providerOptions": {"gateway": {"order": ["novita"]}}})
                    .to_string(),
            )
            .send()
    };
    let answer = call("ai-language-model-id", "deepseek/deepseek-v4-flash-0731")
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    let got = seen.lock().unwrap().last().unwrap().clone();
    assert_eq!(got.path, "/v4/ai/language-model");
    assert_eq!(
        got.header("authorization"),
        Some(format!("Bearer {KEY}").as_str())
    );
    assert_eq!(
        got.header("ai-language-model-id"),
        Some("deepseek/deepseek-v4-flash-0731")
    );
    let body: Value = serde_json::from_slice(&got.body).unwrap();
    assert_eq!(
        body["providerOptions"],
        json!({"gateway": {"order": ["novita"], "only": ["alibaba", "novita"]}})
    );

    // Another model, named in any of the gateway's model headers, never
    // reaches it.
    let before = seen.lock().unwrap().len();
    for header in [
        "ai-language-model-id",
        "x-vercel-ai-gateway-model",
        "x-vercel-ai-gateway-model-id",
    ] {
        let refused = call(header, "google/gemini-2.5-flash").await.unwrap();
        assert_eq!(refused.status(), 403, "{header}");
    }
    assert_eq!(seen.lock().unwrap().len(), before);
    let log = fwd.stop().await;
    assert!(log.contains(r#""pinned":"only""#), "{log}");
    assert!(
        log.contains("the request asked for google/gemini-2.5-flash"),
        "{log}"
    );
    assert!(!log.contains(KEY));
}

#[tokio::test]
async fn a_route_cannot_both_allow_and_deny_upstreams() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let routes = json!([{"prefix": "", "upstream": "https://x.example", "auth": "bearer", "only": ["a"], "ignore": ["baseten"]}]);
    let Err((code, stderr)) = start(dir.path(), routes, 0o600).await else {
        panic!("the forwarder started with both only and ignore")
    };
    assert_eq!(code, 2);
    assert!(stderr.contains("both only and ignore"), "{stderr}");
}

/// A body the forwarder cannot read (compressed, or not a JSON object) would
/// pass the model check and the pin unchecked: on a route that has either,
/// it is refused with 415, says why, and never reaches the provider.
#[tokio::test]
async fn a_body_the_forwarder_cannot_read_is_refused() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let routes = json!([
        {
            "prefix": "/openai",
            "upstream": format!("http://127.0.0.1:{upstream}/v1"),
            "auth": "bearer",
            "models": ["deepseek/deepseek-v4-flash-0731"],
            "only": ["alibaba", "novita"],
        },
        // The agent's own key off a router: nothing to check or pin.
        {
            "prefix": "/anthropic",
            "upstream": format!("http://127.0.0.1:{upstream}/own"),
            "auth": "x-api-key",
        },
    ]);
    let fwd = start(dir.path(), routes, 0o600).await.unwrap();
    let client = reqwest::Client::new();

    // A gzip-compressed body that asks for Baseten.
    let json_body = json!({
        "model": "deepseek/deepseek-v4-flash-0731",
        "input": "hi",
        "providerOptions": {"gateway": {"only": ["baseten"]}},
    })
    .to_string();
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    std::io::Write::write_all(&mut gz, json_body.as_bytes()).unwrap();
    let compressed = gz.finish().unwrap();
    let post = |path: &str, encoding: Option<&str>, body: Vec<u8>| {
        let mut request = client
            .post(fwd.url(path))
            .header("content-type", "application/json")
            .body(body);
        if let Some(encoding) = encoding {
            request = request.header("content-encoding", encoding);
        }
        request.send()
    };

    let answer = post("/openai/responses", Some("gzip"), compressed.clone())
        .await
        .unwrap();
    assert_eq!(answer.status(), 415);
    let error: Value = answer.json().await.unwrap();
    assert_eq!(error["error"]["type"], "benchflow_forwarder");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains("content-encoding gzip"), "{message}");
    assert!(message.contains("send the body uncompressed"), "{message}");
    // Not JSON at all, or JSON that is not an object: refused too.
    for body in [b"plain text".to_vec(), b"[1, 2]".to_vec()] {
        let answer = post("/openai/responses", None, body).await.unwrap();
        assert_eq!(answer.status(), 415);
    }
    assert!(
        seen.lock().unwrap().is_empty(),
        "nothing reached the provider"
    );
    // An empty body has nothing to check, and passes.
    let answer = post("/openai/responses", None, Vec::new()).await.unwrap();
    assert_eq!(answer.status(), 200);
    // The same JSON uncompressed is checked and pinned.
    post(
        "/openai/responses",
        Some("identity"),
        json_body.clone().into_bytes(),
    )
    .await
    .unwrap();
    let pinned: Value = serde_json::from_slice(&seen.lock().unwrap().last().unwrap().body).unwrap();
    assert_eq!(
        pinned["providerOptions"]["gateway"]["only"],
        json!(["alibaba", "novita"])
    );

    // A route with nothing to check passes a compressed body as it is.
    let answer = post("/anthropic/v1/messages", Some("gzip"), compressed.clone())
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    let got = seen.lock().unwrap().last().unwrap().clone();
    assert_eq!(got.path, "/own/v1/messages");
    assert_eq!(got.body, compressed);

    let log = fwd.stop().await;
    let refusals = log
        .lines()
        .filter(|line| line.contains(r#""status":415"#))
        .count();
    assert_eq!(refusals, 3, "{log}");
    assert!(log.contains("content-encoding gzip"), "{log}");
    assert!(!log.contains(KEY));
}

#[tokio::test]
async fn each_answer_logs_the_usage_and_cost_the_provider_reported() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, _) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let fwd = start(dir.path(), routes(upstream), 0o600).await.unwrap();
    let client = reqwest::Client::new();
    for (name, _) in RECORDED {
        let answer = client
            .post(fwd.url(&format!("/openai/recorded/{name}")))
            .header("content-type", "application/json")
            .body(r#"{"model": "m", "stream": true}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(answer.status(), 200, "{name}");
        let _ = answer.bytes().await.unwrap();
    }
    // A token count and a model list bill nothing.
    for path in ["/anthropic/v1/messages/count_tokens", "/openai/models"] {
        let _ = client.get(fwd.url(path)).send().await.unwrap();
    }
    let _ = client
        .post(fwd.url("/anthropic/v1/messages/count_tokens"))
        .body("{}")
        .send()
        .await
        .unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let log = fwd.stop().await;
    assert!(!log.contains(KEY));

    // name: input (not cached), cache read, cache write, output, cost.
    let expect: &[(&str, [u64; 4], Option<f64>)] = &[
        ("gateway-chat.json", [84, 0, 0, 5], Some(0.00004356)),
        ("gateway-chat.sse", [84, 0, 0, 5], Some(0.00004356)),
        ("gateway-messages.json", [84, 0, 0, 5], Some(0.00004356)),
        ("gateway-messages.sse", [84, 0, 0, 5], Some(0.00004356)),
        ("gateway-responses.sse", [84, 0, 0, 16], Some(0.00005808)),
        ("anthropic-cached.sse", [12, 4000, 300, 42], None),
        ("openrouter-chat.json", [20, 100, 0, 8], Some(0.0000452)),
        (
            "ai-sdk-language-model.sse",
            [50, 200, 0, 30],
            Some(0.000021),
        ),
    ];
    let lines: Vec<Value> = log
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    for (name, counts, cost) in expect {
        let line = lines
            .iter()
            .find(|l| l["path"].as_str().is_some_and(|p| p.ends_with(name)))
            .unwrap_or_else(|| panic!("no log line for {name}: {log}"));
        let u = &line["usage"];
        let got = ["input", "cache_read", "cache_write", "output"].map(|k| u[k].as_u64().unwrap());
        assert_eq!(&got, counts, "{name}: {line}");
        match cost {
            Some(c) => assert!(
                (line["cost"].as_f64().unwrap() - c).abs() < 1e-12,
                "{name}: {line}"
            ),
            None => assert!(line.get("cost").is_none(), "{name}: {line}"),
        }
    }
    for line in lines.iter().filter(|l| {
        l["path"]
            .as_str()
            .is_some_and(|p| p.contains("count_tokens") || p.ends_with("/models"))
    }) {
        assert!(
            line.get("usage").is_none() && line.get("cost").is_none(),
            "{line}"
        );
    }

    // Summed: one call (Anthropic's) said no cost, so the run's is unknown.
    let report = crate::providers::ForwarderReport {
        log,
        ..Default::default()
    };
    let spend = report.spend();
    assert_eq!((spend.calls, spend.with_usage, spend.priced), (8, 8, 7));
    assert_eq!(spend.input_tokens, 84 * 4 + 84 + 12 + 20 + 50);
    assert_eq!(spend.cost_usd(), None);
    let note = spend.unknown_cost().unwrap();
    assert!(note.contains("7 of 8"), "{note}");
}

/// A body over the cap gets 413 and a reason, not a reset connection, and
/// never reaches the provider.
#[tokio::test]
async fn an_oversized_body_is_a_413() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let fwd = start(dir.path(), routes(upstream), 0o600).await.unwrap();
    let answer = reqwest::Client::new()
        .post(fwd.url("/openai/responses"))
        .header("content-type", "application/json")
        .body(vec![b' '; 64 * 1024 * 1024 + 1])
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 413);
    let error: Value = answer.json().await.unwrap();
    assert!(
        error["error"]["message"]
            .as_str()
            .unwrap()
            .contains("64 MiB")
    );
    assert!(seen.lock().unwrap().is_empty());
    fwd.stop().await;
}

/// A provider that sends the key back in a response header does not show
/// it to the agent.
#[tokio::test]
async fn the_key_is_masked_in_response_headers() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let (upstream, _seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let fwd = start(dir.path(), routes(upstream), 0o600).await.unwrap();
    let answer = reqwest::Client::new()
        .get(fwd.url("/anthropic/header-echo"))
        .send()
        .await
        .unwrap();
    assert_eq!(answer.status(), 200);
    let echoed = answer.headers()["x-echo-auth"]
        .to_str()
        .unwrap()
        .to_string();
    assert!(echoed.starts_with("Bearer **"), "{echoed}");
    assert!(!echoed.contains(KEY), "{echoed}");
    fwd.stop().await;
}

/// A small PRNG for the forwarder fuzz test (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }
}

/// One random, often malformed, HTTP request.
fn fuzz_request(rng: &mut Rng) -> (String, Vec<u8>) {
    const METHODS: &[&str] = &[
        "GET", "POST", "PUT", "DELETE", "OPTIONS", "CONNECT", "TRACE", "get", "P OST", "", "\u{0}",
    ];
    const PATHS: &[&str] = &[
        "/anthropic/v1/messages",
        "/openai/chat/completions",
        "/anthropic/sse",
        "/anthropic/status/401",
        "/anthropic/reset-once",
        "/",
        "*",
        "http://evil.example/anthropic/x",
        "//anthropic/x",
        "/anthropic/../../etc/passwd",
        "/anthropic/%2e%2e/x",
        "/anthropic%00/x",
        "/openai?model=x",
        "/anthropic/v1/messages?x=%ZZ",
        "/ANTHROPIC/x",
        "/anthropicx/y",
        "",
    ];
    const VERSIONS: &[&str] = &["HTTP/1.1", "HTTP/1.0", "HTTP/2", "HTTP/9.9", "", "http/1.1"];
    const HEADERS: &[&str] = &[
        "host: 127.0.0.1",
        "host: evil.example",
        "connection: close",
        "connection: keep-alive",
        "content-length: 5",
        "content-length: -1",
        "content-length: 99999999999999999999",
        "content-length: 3\r\ncontent-length: 7",
        "transfer-encoding: chunked",
        "transfer-encoding: gzip, chunked",
        "content-encoding: gzip",
        "content-encoding: br",
        "expect: 100-continue",
        "authorization: Bearer benchflow-forwarder-holds-the-key",
        "x-api-key: sk-other",
        "authorization: Basic Zm9vOmJhcg==",
        "x-forwarded-host: evil.example",
        "content-type: application/json",
        "x: \u{1}\u{7f}",
        "bad header line",
        ": empty-name",
        "x-very-long: VERYLONG",
    ];
    const BODIES: &[&str] = &[
        "",
        "{}",
        r#"{"model": "m", "max_tokens": 1}"#,
        "{\"model\": ",
        "5\r\nhello\r\n0\r\n\r\n",
        "zz\r\nhello\r\n",
        "ffffffffffffffff\r\nx",
        "\u{0}\u{0}\u{0}",
        "not json",
    ];
    let path = format!(
        "{}{}",
        rng.pick(PATHS),
        if rng.below(4) == 0 {
            rng.pick(PATHS)
        } else {
            ""
        }
    );
    // Mostly well-formed, so most requests get past the HTTP parser to the
    // forwarder's own handling; one part in a few is broken.
    let method = if rng.below(8) == 0 {
        rng.pick(METHODS)
    } else {
        rng.pick(&["GET", "POST"])
    };
    let version = if rng.below(8) == 0 {
        rng.pick(VERSIONS)
    } else {
        "HTTP/1.1"
    };
    let line = format!("{method} {path} {version}");
    let mut head = format!("{line}\r\nhost: 127.0.0.1");
    for _ in 0..rng.below(4) {
        let h = rng
            .pick(HEADERS)
            .replace("VERYLONG", &"a".repeat(rng.below(70_000)));
        head.push_str("\r\n");
        head.push_str(&h);
    }
    let body = rng.pick(BODIES);
    let framed = head.contains("content-length") || head.contains("transfer-encoding");
    if !framed && rng.below(8) != 0 {
        head.push_str(&format!("\r\ncontent-length: {}", body.len()));
    }
    head.push_str(if rng.below(16) == 0 {
        "\n\n"
    } else {
        "\r\n\r\n"
    });
    let mut bytes = head.into_bytes();
    bytes.extend_from_slice(body.as_bytes());
    if rng.below(12) == 0 {
        let n = rng.below(4096);
        bytes.extend((0..n).map(|_| rng.next() as u8));
    }
    if rng.below(12) == 0 {
        let keep = rng.below(bytes.len() + 1);
        bytes.truncate(keep);
    }
    (path, bytes)
}

/// Random and malformed requests (bad request lines, conflicting lengths,
/// broken chunking, huge headers, truncated bodies, raw bytes) never stop
/// the forwarder and never show the key, and it still serves a good request
/// afterwards. `BENCH_FUZZ_FORWARDER_CASES` sets the count (default 150),
/// `BENCH_FUZZ_SEED` the seed (printed on failure).
#[tokio::test]
async fn random_requests_never_stop_the_forwarder_or_show_the_key() {
    if node().is_none() {
        eprintln!("skipped: no `node` on PATH to run the forwarder with");
        return;
    }
    let cases: usize = std::env::var("BENCH_FUZZ_FORWARDER_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(150);
    let seed: u64 = std::env::var("BENCH_FUZZ_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                | 1
        });
    let mut rng = Rng(seed);
    let (upstream, _seen) = fake_provider().await;
    let dir = tempfile::tempdir().unwrap();
    let mut fwd = start(dir.path(), routes(upstream), 0o600).await.unwrap();
    let client = reqwest::Client::new();
    let mut statuses: std::collections::BTreeMap<String, usize> = Default::default();
    for i in 0..cases {
        let (path, request) = fuzz_request(&mut rng);
        if let Ok(Ok(mut stream)) = tokio::time::timeout(
            Duration::from_secs(5),
            TcpStream::connect(("127.0.0.1", fwd.port)),
        )
        .await
        {
            let _ = stream.write_all(&request).await;
            // Half-closing ends a truncated request at once instead of
            // after a timeout; Node then drops a request it would answer
            // asynchronously, so some requests get no answer (status none).
            let _ = stream.shutdown().await;
            let mut answer = Vec::new();
            let _ =
                tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut answer)).await;
            let answer = String::from_utf8_lossy(&answer);
            let status = answer.get(9..12).unwrap_or("none").to_string();
            if std::env::var_os("BENCH_FUZZ_SHOW").is_some() && i < 40 {
                eprintln!(
                    "{:?} -> {:?}",
                    String::from_utf8_lossy(&request)
                        .chars()
                        .take(160)
                        .collect::<String>(),
                    answer.chars().take(200).collect::<String>()
                );
            }
            *statuses.entry(status).or_default() += 1;
            if !path.contains("echo") {
                assert!(
                    !answer.contains(KEY),
                    "seed {seed} case {i}: the key in {answer:.400}"
                );
            }
        }
        assert!(
            fwd.child.try_wait().unwrap().is_none(),
            "seed {seed} case {i}: the forwarder exited on {:?}\n{}",
            String::from_utf8_lossy(&request)
                .chars()
                .take(400)
                .collect::<String>(),
            fwd.stderr.lock().unwrap()
        );
        if i % 25 == 24 || i + 1 == cases {
            let ok = client
                .post(fwd.url("/anthropic/v1/messages"))
                .body(r#"{"model": "m", "max_tokens": 1}"#)
                .timeout(Duration::from_secs(10))
                .send()
                .await;
            assert!(
                ok.as_ref().is_ok_and(|a| a.status() == 200),
                "seed {seed} case {i}: a good request failed after fuzzing: {ok:?}"
            );
        }
    }
    eprintln!("forwarder fuzz: seed {seed}, {cases} requests, answers by status {statuses:?}");
    let log = fwd.stop().await;
    assert!(
        !log.contains(KEY),
        "seed {seed}: the key in the forwarder's log"
    );
}
