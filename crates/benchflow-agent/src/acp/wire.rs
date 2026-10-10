//! The wire: JSON-RPC 2.0 messages, one per line, on the agent's stdout.
//!
//! Agents write other things there too (logs, a shell prompt glued to the
//! first message on a PTY, lines of many megabytes when a tool returns
//! images). [`LineReader`] reads whole lines up to a hard cap, [`shrink`]
//! rewrites a line over the size limit the way 0.8 does (#1138) so its
//! message survives without the bulk, and [`decode`] accepts only JSON-RPC
//! 2.0 envelopes; everything else is noise for the agent log.

use std::pin::Pin;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};

/// Lines longer than this are rewritten to fit, as 0.8 does (its default):
/// images and long base64 strings become notes, long text is cut.
pub const DEFAULT_LINE_LIMIT: usize = 1024 * 1024;
/// Lines longer than this are skipped whole (0.8's oversized-line cap).
pub const DEFAULT_MAX_LINE: usize = 256 * 1024 * 1024;

/// One line read from the agent.
#[derive(Debug)]
pub(crate) enum Line {
    /// A line without its newline.
    Text(Vec<u8>),
    /// A line over the cap, consumed and dropped.
    TooLong(usize),
}

/// Reads newline-terminated lines of any length up to a cap.
pub(crate) struct LineReader {
    inner: BufReader<Pin<Box<dyn AsyncRead + Send>>>,
    max: usize,
}

impl LineReader {
    pub(crate) fn new(reader: Pin<Box<dyn AsyncRead + Send>>, max: usize) -> Self {
        Self {
            inner: BufReader::with_capacity(64 * 1024, reader),
            max,
        }
    }

    /// The next line; `None` at the end of the stream. A last line without a
    /// newline is still returned.
    pub(crate) async fn next_line(&mut self) -> std::io::Result<Option<Line>> {
        let mut line = Vec::new();
        let mut dropped = 0usize;
        loop {
            let buf = self.inner.fill_buf().await?;
            if buf.is_empty() {
                return Ok(match (line.is_empty(), dropped) {
                    (true, 0) => None,
                    (_, 0) => Some(Line::Text(line)),
                    _ => Some(Line::TooLong(dropped + line.len())),
                });
            }
            let (chunk, done) = match buf.iter().position(|b| *b == b'\n') {
                Some(i) => (&buf[..i], Some(i + 1)),
                None => (buf, None),
            };
            if dropped > 0 || line.len() + chunk.len() > self.max {
                dropped += line.len() + chunk.len();
                line.clear();
            } else {
                line.extend_from_slice(chunk);
            }
            let consumed = done.unwrap_or(buf.len());
            self.inner.consume(consumed);
            if done.is_some() {
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                return Ok(Some(if dropped > 0 {
                    Line::TooLong(dropped)
                } else {
                    Line::Text(line)
                }));
            }
        }
    }
}

/// A JSON-RPC 2.0 message, when `text` is one: a request or notification
/// (`method`, and neither `result` nor `error`), or a response (`id` and
/// exactly one of `result` and `error`). JSON that is not an envelope (an
/// agent's JSON-formatted log line) is not a message (0.8's
/// `decode_json_rpc_message`).
pub(crate) fn decode(text: &str) -> Option<Map<String, Value>> {
    let Ok(Value::Object(message)) = serde_json::from_str::<Value>(text) else {
        return None;
    };
    is_envelope(&message).then_some(message)
}

pub(crate) fn is_envelope(message: &Map<String, Value>) -> bool {
    if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return false;
    }
    let has_result = message.contains_key("result");
    let has_error = message.contains_key("error");
    if let Some(method) = message.get("method") {
        return method.is_string() && !has_result && !has_error;
    }
    message.contains_key("id") && has_result != has_error
}

static ANSI_CSI: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\[[0-?]*[ -/]*[@-~]").expect("valid regex"));
static ANSI_OSC: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)").expect("valid regex"));

/// [`decode`], tolerating terminal noise before the first message: a PTY can
/// glue the shell prompt (with escape sequences) to the agent's first line.
/// Used only until the first message decodes, so a log line that quotes a
/// valid envelope later can never pass as protocol traffic.
pub(crate) fn decode_first(text: &str) -> Option<Map<String, Value>> {
    if let Some(message) = decode(text) {
        return Some(message);
    }
    let brace = text.find('{')?;
    if brace > 0
        && let Some(message) = decode(&text[brace..])
    {
        return Some(message);
    }
    if !text.contains('\x1b') {
        return None;
    }
    let cleaned = ANSI_CSI
        .replace_all(&ANSI_OSC.replace_all(text, ""), "")
        .into_owned();
    decode(&cleaned[cleaned.find('{')?..])
}

/// Strings shorter than this are never replaced or cut.
const LONG: usize = 4096;
/// Keys that identify a message; never rewritten.
const KEEP_KEYS: [&str; 12] = [
    "jsonrpc",
    "id",
    "method",
    "sessionId",
    "sessionUpdate",
    "toolCallId",
    "status",
    "kind",
    "title",
    "stopReason",
    "type",
    "mimeType",
];

fn note(what: &str, size: usize, limit: usize) -> String {
    format!(
        "[BenchFlow removed {what} ({size} bytes): the ACP line was over the {limit}-byte limit]"
    )
}

/// A JSON line over `limit` bytes, rewritten to fit as 0.8 does (#1138):
/// image and audio blocks become a note saying what was removed, other long
/// base64 strings too, and then the longest remaining strings are cut until
/// the line fits. Keys that identify the message are never touched, so a
/// `tool_call_update` still closes its call. `None` when the line is not
/// JSON (it stays noise).
pub(crate) fn shrink(line: &[u8], limit: usize) -> Option<Value> {
    let node: Value = serde_json::from_slice(line).ok()?;
    if !node.is_object() && !node.is_array() {
        return None;
    }
    let mut node = strip_media(node, limit);
    fit(&mut node, limit);
    Some(node)
}

fn is_base64(s: &str) -> bool {
    s.bytes()
        .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'+' | b'/' | b'=' | b'\r' | b'\n'))
}

fn strip_media(node: Value, limit: usize) -> Value {
    match node {
        Value::Object(map) => {
            let kind = map.get("type").and_then(Value::as_str);
            if let (Some(kind @ ("image" | "audio")), Some(Value::String(data))) =
                (kind, map.get("data"))
                && data.chars().count() >= LONG
            {
                let mime = map
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .filter(|m| !m.is_empty());
                let what = format!("{kind} data ({})", mime.unwrap_or("unknown type"));
                let mut text = Map::new();
                text.insert("type".into(), "text".into());
                text.insert(
                    "text".into(),
                    note(&what, data.chars().count(), limit).into(),
                );
                return Value::Object(text);
            }
            Value::Object(
                map.into_iter()
                    .map(|(k, v)| {
                        let v = if KEEP_KEYS.contains(&k.as_str()) {
                            v
                        } else {
                            strip_media(v, limit)
                        };
                        (k, v)
                    })
                    .collect(),
            )
        }
        Value::Array(items) => {
            Value::Array(items.into_iter().map(|v| strip_media(v, limit)).collect())
        }
        Value::String(s) if s.len() >= LONG && is_base64(&s) => {
            Value::String(note("base64 data", s.chars().count(), limit))
        }
        other => other,
    }
}

/// A path to a value: object keys and array indices.
#[derive(Clone)]
enum Step {
    Key(String),
    Index(usize),
}

fn long_strings(node: &Value, path: &mut Vec<Step>, found: &mut Vec<(usize, Vec<Step>)>) {
    match node {
        Value::Object(map) => {
            for (key, value) in map {
                if !KEEP_KEYS.contains(&key.as_str()) {
                    path.push(Step::Key(key.clone()));
                    long_strings(value, path, found);
                    path.pop();
                }
            }
        }
        Value::Array(items) => {
            for (i, value) in items.iter().enumerate() {
                path.push(Step::Index(i));
                long_strings(value, path, found);
                path.pop();
            }
        }
        Value::String(s) => {
            let size = s.chars().count();
            if size >= LONG {
                found.push((size, path.clone()));
            }
        }
        _ => {}
    }
}

fn at_path<'a>(node: &'a mut Value, path: &[Step]) -> Option<&'a mut Value> {
    path.iter().try_fold(node, |node, step| match step {
        Step::Key(key) => node.get_mut(key.as_str()),
        Step::Index(i) => node.get_mut(*i),
    })
}

fn fit(node: &mut Value, limit: usize) {
    for _ in 0..64 {
        let size = serde_json::to_vec(node).map_or(0, |v| v.len());
        let excess = size.saturating_sub(limit);
        if excess == 0 {
            return;
        }
        let mut found = Vec::new();
        long_strings(node, &mut Vec::new(), &mut found);
        // The first of the longest, as Python's max() picks.
        let Some((size, path)) = found
            .into_iter()
            .reduce(|best, next| if next.0 > best.0 { next } else { best })
        else {
            return;
        };
        let keep = (LONG / 2).max(size.saturating_sub(excess).saturating_sub(256));
        if let Some(Value::String(s)) = at_path(node, &path) {
            let mut cut: String = s.chars().take(keep).collect();
            cut.push(' ');
            cut.push_str(&note("the rest of this text", size - keep, limit));
            *s = cut;
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    async fn lines(input: &[u8], max: usize) -> Vec<Line> {
        let mut reader = LineReader::new(Box::pin(std::io::Cursor::new(input.to_vec())), max);
        let mut out = Vec::new();
        while let Some(line) = reader.next_line().await.unwrap() {
            out.push(line);
        }
        out
    }

    #[tokio::test]
    async fn reads_lines_of_any_size_up_to_the_cap() {
        let big = "x".repeat(200_000);
        let input = format!("a\r\n{big}\n\nlast");
        let got = lines(input.as_bytes(), 1_000_000).await;
        let texts: Vec<String> = got
            .iter()
            .map(|l| match l {
                Line::Text(t) => String::from_utf8(t.clone()).unwrap(),
                Line::TooLong(n) => format!("too long {n}"),
            })
            .collect();
        assert_eq!(
            texts,
            ["a".to_string(), big, String::new(), "last".to_string()]
        );

        let got = lines(format!("{}\nok\n", "y".repeat(5000)).as_bytes(), 4096).await;
        assert!(matches!(got[0], Line::TooLong(5000)));
        assert!(matches!(&got[1], Line::Text(t) if t == b"ok"));
    }

    #[test]
    fn only_json_rpc_envelopes_are_messages() {
        assert!(decode(r#"{"jsonrpc":"2.0","id":1,"result":{}}"#).is_some());
        assert!(decode(r#"{"jsonrpc":"2.0","method":"session/update","params":{}}"#).is_some());
        assert!(decode(r#"{"jsonrpc":"2.0","id":1,"result":{},"error":{}}"#).is_none());
        assert!(decode(r#"{"jsonrpc":"2.0","method":5}"#).is_none());
        assert!(decode(r#"{"level":"info","msg":"starting"}"#).is_none());
        assert!(decode("[1,2]").is_none());
        assert!(decode("Starting agent...").is_none());
    }

    #[test]
    fn a_shell_prompt_glued_to_the_first_message_is_tolerated() {
        let glued = "\x1b[?2004h\x1b]0;root@box: /app\x07root@box:/app# \x1b[?2004l{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}";
        assert!(decode(glued).is_none());
        assert!(decode_first(glued).is_some());
        assert!(decode_first("INFO {\"jsonrpc\": \"1.0\"}").is_none());
    }

    #[test]
    fn an_oversized_line_keeps_its_message_and_drops_the_bulk() {
        // Guards 0.8's #1138: a Read of a PDF returns pages as base64 images in
        // one multi-megabyte tool_call_update; the update must still close the call.
        let image = "A".repeat(3_000_000);
        let line = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "s", "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "t1",
                "status": "completed",
                "content": [{"type": "content", "content": {"type": "image", "data": image, "mimeType": "image/png"}}],
            }},
        })
        .to_string();
        let shrunk = shrink(line.as_bytes(), DEFAULT_LINE_LIMIT).unwrap();
        let update = &shrunk["params"]["update"];
        assert_eq!(update["status"], "completed");
        assert_eq!(update["toolCallId"], "t1");
        assert_eq!(
            update["content"][0]["content"],
            json!({"type": "text", "text": "[BenchFlow removed image data (image/png) (3000000 bytes): the ACP line was over the 1048576-byte limit]"})
        );
        assert!(serde_json::to_vec(&shrunk).unwrap().len() < DEFAULT_LINE_LIMIT);

        // Long text is cut, keeping its start.
        let text = format!("start {}", "word ".repeat(400_000));
        let line = json!({"jsonrpc": "2.0", "method": "x", "params": {"text": text}}).to_string();
        let shrunk = shrink(line.as_bytes(), DEFAULT_LINE_LIMIT).unwrap();
        let cut = shrunk["params"]["text"].as_str().unwrap();
        assert!(cut.starts_with("start word"));
        assert!(cut.ends_with("the ACP line was over the 1048576-byte limit]"));
        assert!(serde_json::to_vec(&shrunk).unwrap().len() <= DEFAULT_LINE_LIMIT);
        assert_eq!(shrink(b"not json", 10), None);
    }
}
