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
                    (_, 0) => {
                        // A last line without its newline loses its `\r`
                        // too, like every other line.
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }
                        Some(Line::Text(line))
                    }
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

/// A line with more JSON tokens than this is not parsed as it is: as values
/// it would take about 70 bytes each (a 256 MiB line of `0,` took 9 GB).
/// Its largest parts that do not identify the message are first replaced,
/// as bytes, until it has at most half as many.
const MAX_TOKENS: usize = 1 << 20;
/// Keys whose presence marks a part that identifies the message; such a part
/// is never replaced whole (its other values may be). Only where an ACP
/// envelope has them: in the message, its `params` or `result`, and the
/// `update` or `toolCall` in those (objects at depth [`IDENTITY_DEPTH`] or
/// less). An `id` deep inside a tool's output identifies nothing.
const IDENTITY_KEYS: [&str; 7] = [
    "jsonrpc",
    "id",
    "method",
    "sessionId",
    "sessionUpdate",
    "toolCallId",
    "stopReason",
];
/// The deepest objects whose [`IDENTITY_KEYS`] count (the top is 0).
const IDENTITY_DEPTH: usize = 2;
/// What a replaced part's note calls it.
const PART: &str = "part of this message";

/// An over-long line, rewritten.
#[derive(Debug)]
pub(crate) struct Shrunk {
    pub(crate) value: Value,
    /// Whether the rewrite fits the limit. When it does not (what identifies
    /// the message is itself over the limit, a 2 MB `title` say), it is
    /// passed on over the limit, as 0.8 passes it: a line kept whole beats a
    /// stopped agent.
    pub(crate) fits: bool,
}

/// A JSON line over `limit` bytes, rewritten to fit as 0.8 does (#1138):
/// image and audio blocks become a note saying what was removed, other long
/// base64 strings too, and then the longest remaining strings are cut until
/// the line fits. Where 0.8 would still pass the line on over the limit
/// (many medium strings, or bulk that is not text), parts that do not
/// identify the message are replaced by a note until it fits, keeping their
/// type (an array keeps its first items and ends with a note item) and the
/// deepest part that holds the bulk (so `_meta`'s small fields stay). Keys
/// that identify the message are never touched, so a `tool_call_update`
/// still closes its call. A line of more than [`MAX_TOKENS`] tokens has its
/// bulk replaced as bytes before it is parsed. `None` when the line is not
/// JSON (it stays noise), or has more tokens than any replacement can bring
/// under [`MAX_TOKENS`] (a top-level array of a million numbers, say).
pub(crate) fn shrink(line: &[u8], limit: usize) -> Option<Shrunk> {
    let reduced;
    let line = if tokens(line) > MAX_TOKENS {
        reduced = reduce_bulk(line, limit)?;
        &reduced[..]
    } else {
        line
    };
    let node: Value = serde_json::from_slice(line).ok()?;
    if !node.is_object() && !node.is_array() {
        return None;
    }
    let mut node = strip_media(node, limit);
    let mut size = fit(&mut node, limit);
    if size > limit {
        size = replace_parts(&mut node, size, limit);
    }
    Some(Shrunk {
        value: node,
        fits: size <= limit,
    })
}

/// About how many values and keys a JSON text holds, counted without
/// parsing: brackets, commas and colons outside strings.
fn tokens(line: &[u8]) -> usize {
    let (mut count, mut in_string, mut escaped) = (1usize, false, false);
    for &b in line {
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_string = false,
                _ => {}
            }
        } else {
            match b {
                b'"' => in_string = true,
                b'{' | b'[' | b',' | b':' => count += 1,
                _ => {}
            }
        }
    }
    count
}

/// A container in a line, found without parsing it.
struct Span {
    start: usize,
    /// One past its closing bracket.
    end: usize,
    tokens: usize,
    /// Its nearest enclosing span in the list, if any.
    parent: Option<usize>,
    /// Whether it may be replaced: below the top, not under a key that is
    /// never rewritten, holding no key that identifies the message.
    free: bool,
    /// Whether one free span inside it holds nearly all its tokens; then
    /// that one is replaced instead, so its small siblings stay.
    bulk_inside: bool,
}

/// `line`, too big to parse, with its largest free containers replaced by a
/// note (an array by `["<note>"]`, an object by `{"note": "<note>"}`) until
/// it holds at most `MAX_TOKENS / 2` tokens. `None` when the containers that
/// may be replaced do not hold that many, or the nesting is deeper than JSON
/// parsing allows anyway.
fn reduce_bulk(line: &[u8], limit: usize) -> Option<Vec<u8>> {
    struct Open {
        start: usize,
        tokens_at: usize,
        object: bool,
        free: bool,
        identity: bool,
        expect_key: bool,
        key: Vec<u8>,
    }
    const MIN_TOKENS: usize = 1024;
    let mut spans: Vec<Span> = Vec::new();
    let mut stack: Vec<Open> = Vec::new();
    let (mut count, mut in_string, mut escaped) = (1usize, false, false);
    let mut key: Option<Vec<u8>> = None;
    for (i, &b) in line.iter().enumerate() {
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => {
                    in_string = false;
                    let depth = stack.len().saturating_sub(1);
                    if let (Some(k), Some(top)) = (key.take(), stack.last_mut()) {
                        top.identity |= depth <= IDENTITY_DEPTH
                            && IDENTITY_KEYS.iter().any(|w| w.as_bytes() == k.as_slice());
                        top.key = k;
                    }
                    continue;
                }
                _ => {}
            }
            if let Some(k) = key.as_mut().filter(|k| k.len() < 32) {
                k.push(b);
            }
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                if let Some(top) = stack.last_mut()
                    && top.object
                    && top.expect_key
                {
                    top.expect_key = false;
                    key = Some(Vec::new());
                }
            }
            b'{' | b'[' => {
                count += 1;
                if stack.len() >= 128 {
                    return None;
                }
                let free = stack.last().is_some_and(|top| {
                    !top.object || !KEEP_KEYS.iter().any(|w| w.as_bytes() == top.key.as_slice())
                });
                stack.push(Open {
                    start: i,
                    tokens_at: count,
                    object: b == b'{',
                    free,
                    identity: false,
                    expect_key: b == b'{',
                    key: Vec::new(),
                });
            }
            b'}' | b']' => {
                let open = stack.pop()?;
                if let Some(top) = stack.last_mut() {
                    top.identity |= open.identity;
                }
                let tokens = count - open.tokens_at + 1;
                if tokens >= MIN_TOKENS {
                    spans.push(Span {
                        start: open.start,
                        end: i + 1,
                        tokens,
                        parent: None,
                        free: open.free && !open.identity,
                        bulk_inside: false,
                    });
                }
            }
            b',' => {
                count += 1;
                if let Some(top) = stack.last_mut() {
                    top.expect_key = top.object;
                }
            }
            b':' => count += 1,
            _ => {}
        }
    }
    // Spans close inner-first; in start order, a sweep finds each one's
    // nearest enclosing span.
    spans.sort_by_key(|s| s.start);
    let mut open: Vec<usize> = Vec::new();
    for i in 0..spans.len() {
        while open.last().is_some_and(|&o| spans[o].end <= spans[i].start) {
            open.pop();
        }
        spans[i].parent = open.last().copied();
        open.push(i);
    }
    for i in 0..spans.len() {
        if let Some(p) = spans[i].parent
            && spans[i].free
            && spans[i].tokens * 10 >= spans[p].tokens * 9
        {
            spans[p].bulk_inside = true;
        }
    }
    let need = count.saturating_sub(MAX_TOKENS / 2);
    let mut order: Vec<usize> = (0..spans.len())
        .filter(|&i| spans[i].free && !spans[i].bulk_inside)
        .collect();
    order.sort_by_key(|&i| std::cmp::Reverse(spans[i].tokens));
    let mut chosen: std::collections::BTreeMap<usize, usize> = Default::default();
    let mut removed = 0;
    for i in order {
        if removed >= need {
            break;
        }
        let s = &spans[i];
        if chosen
            .range(..=s.start)
            .next_back()
            .is_some_and(|(_, &end)| end >= s.end)
        {
            continue;
        }
        chosen.insert(s.start, s.end);
        removed += s.tokens - 1;
    }
    if removed < need {
        return None;
    }
    let mut out = Vec::with_capacity(line.len() / 2);
    let mut at = 0;
    for (&start, &end) in &chosen {
        out.extend_from_slice(&line[at..start]);
        let text = Value::String(note(PART, end - start, limit)).to_string();
        if line[start] == b'{' {
            out.extend_from_slice(b"{\"note\":");
            out.extend_from_slice(text.as_bytes());
            out.push(b'}');
        } else {
            out.push(b'[');
            out.extend_from_slice(text.as_bytes());
            out.push(b']');
        }
        at = end;
    }
    out.extend_from_slice(&line[at..]);
    Some(out)
}

/// Bytes of `value` written as compact JSON, without writing it.
fn json_len(value: &Value) -> usize {
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0 += buf.len();
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Count(0);
    let _ = serde_json::to_writer(&mut count, value);
    count.0
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

/// Calls `f` on every value `shrink` may rewrite, in pre-order, with its
/// index in that order: everything but the values under a
/// key that is never rewritten. The same numbering as [`at_index_mut`].
fn walk<'a>(node: &'a Value, next: &mut usize, f: &mut impl FnMut(&'a Value, usize)) {
    let me = *next;
    *next += 1;
    f(node, me);
    match node {
        Value::Object(map) => {
            for (key, value) in map {
                if !KEEP_KEYS.contains(&key.as_str()) {
                    walk(value, next, f);
                }
            }
        }
        Value::Array(items) => {
            for value in items {
                walk(value, next, f);
            }
        }
        _ => {}
    }
}

/// The value at pre-order `target` in [`walk`]'s numbering.
fn at_index_mut(node: &mut Value, target: usize) -> Option<&mut Value> {
    fn go<'a>(node: &'a mut Value, target: usize, next: &mut usize) -> Option<&'a mut Value> {
        let me = *next;
        *next += 1;
        if me == target {
            return Some(node);
        }
        match node {
            Value::Object(map) => {
                for (key, value) in map.iter_mut() {
                    if KEEP_KEYS.contains(&key.as_str()) {
                        continue;
                    }
                    if let Some(found) = go(value, target, next) {
                        return Some(found);
                    }
                }
                None
            }
            Value::Array(items) => {
                for value in items.iter_mut() {
                    if let Some(found) = go(value, target, next) {
                        return Some(found);
                    }
                }
                None
            }
            _ => None,
        }
    }
    go(node, target, &mut 0)
}

/// 0.8's `_fit`: up to 64 times, cut the longest string to about what
/// makes the line fit. Returns the line's size after.
fn fit(node: &mut Value, limit: usize) -> usize {
    let mut size = json_len(node);
    for _ in 0..64 {
        let excess = size.saturating_sub(limit);
        if excess == 0 {
            break;
        }
        // The first of the longest, as Python's max() picks.
        let mut longest: Option<(usize, usize)> = None;
        walk(node, &mut 0, &mut |value, index| {
            if let Value::String(s) = value {
                let chars = s.chars().count();
                if chars >= LONG && longest.is_none_or(|(best, _)| chars > best) {
                    longest = Some((chars, index));
                }
            }
        });
        let Some((chars, index)) = longest else {
            break;
        };
        let keep = (LONG / 2).max(chars.saturating_sub(excess).saturating_sub(256));
        if let Some(value) = at_index_mut(node, index) {
            let before = json_len(value);
            if let Value::String(s) = value {
                let mut cut: String = s.chars().take(keep).collect();
                cut.push(' ');
                cut.push_str(&note("the rest of this text", chars - keep, limit));
                *s = cut;
            }
            size = size - before + json_len(value);
        }
    }
    size
}

/// A part of a parsed line that may be replaced whole, by its index in
/// [`walk`]'s order: its subtree is `index..end`.
struct Part {
    index: usize,
    end: usize,
    size: usize,
    /// Bytes its replacement keeps: an object's never-rewritten keys.
    kept: usize,
    /// Whether one part inside it holds nearly all its bytes; then that one
    /// is replaced instead, so its small siblings (`_meta`'s `toolName`)
    /// stay.
    bulk_inside: bool,
}

/// Bytes a replaced part may take: its note, in a note item.
fn least(limit: usize) -> usize {
    note(PART, usize::MAX, limit).len() + 160
}

/// Replaces parts of `node` that do not identify the message with notes so
/// that it fits: the smallest part whose replacement alone is enough, else
/// the largest parts, largest first. A part holding nearly all of a bigger
/// one is taken instead of it. Returns the size after (still over `limit`
/// when the parts that identify the message alone are too big).
fn replace_parts(node: &mut Value, mut size: usize, limit: usize) -> usize {
    // A round can fall short (an estimate, or parts that hold little but
    // what they keep); plan again on what is left while that helps.
    for _ in 0..8 {
        let after = replace_round(node, size, limit);
        if after <= limit || after >= size {
            return after;
        }
        size = after;
    }
    size
}

/// One plan over `node`'s parts, applied. Returns the size after.
fn replace_round(node: &mut Value, size: usize, limit: usize) -> usize {
    let mut found = Vec::new();
    parts(node, 0, &mut 0, &mut found);
    let excess = size.saturating_sub(limit);
    let least = least(limit);
    // What replacing a part saves: what it takes, less what its
    // replacement keeps and the note.
    let saving = |p: &Part| p.size.saturating_sub(p.kept + least);
    // index -> how many bytes the part may keep
    let mut chosen: std::collections::BTreeMap<usize, (usize, usize)> = Default::default();
    let one = found
        .iter()
        .filter(|p| saving(p) >= excess)
        .min_by_key(|p| p.size);
    if let Some(p) = one {
        chosen.insert(p.index, (p.end, p.size - excess));
    } else {
        found.retain(|p| !p.bulk_inside && saving(p) > 0);
        found.sort_by_key(|p| std::cmp::Reverse(saving(p)));
        let mut saved = 0;
        for p in &found {
            if saved >= excess {
                break;
            }
            if chosen
                .range(..=p.index)
                .next_back()
                .is_some_and(|(_, &(end, _))| end >= p.end)
            {
                continue;
            }
            saved += saving(p);
            chosen.insert(p.index, (p.end, 0));
        }
    }
    replace_at(node, &mut 0, &chosen, limit);
    json_len(node)
}

/// Replaces each chosen part, keeping its type: a string becomes a note;
/// an array keeps the first items that fit what it may keep and ends with
/// a note item shaped like its items; an object keeps its never-rewritten
/// keys and gets a `note`.
fn replace_at(
    node: &mut Value,
    next: &mut usize,
    chosen: &std::collections::BTreeMap<usize, (usize, usize)>,
    limit: usize,
) {
    let me = *next;
    *next += 1;
    if let Some(&(end, keep)) = chosen.get(&me) {
        *next = end;
        let size = json_len(node);
        match node {
            Value::Array(items) => {
                let mut kept = 2;
                let total = items.len();
                let shaped = note_item(items.first());
                let n = items
                    .iter()
                    .take_while(|v| {
                        kept += json_len(v) + 1;
                        kept + least(limit) <= keep
                    })
                    .count();
                let removed: usize = items[n..].iter().map(|v| json_len(v) + 1).sum();
                items.truncate(n);
                let what = format!("{} of {total} items of {PART}", total - n);
                items.push(shaped(note(&what, removed, limit)));
            }
            Value::Object(map) => {
                map.retain(|k, _| KEEP_KEYS.contains(&k.as_str()));
                map.insert("note".into(), note(PART, size, limit).into());
            }
            other => *other = Value::String(note(PART, size, limit)),
        }
        return;
    }
    match node {
        Value::Object(map) => {
            for (key, value) in map.iter_mut() {
                if !KEEP_KEYS.contains(&key.as_str()) {
                    replace_at(value, next, chosen, limit);
                }
            }
        }
        Value::Array(items) => {
            for value in items.iter_mut() {
                replace_at(value, next, chosen, limit);
            }
        }
        _ => {}
    }
}

/// A note item shaped like `first`, the array's first item: an ACP tool
/// call content block, a content block, or a plain string.
fn note_item(first: Option<&Value>) -> fn(String) -> Value {
    match first.and_then(|v| v.get("type")).and_then(Value::as_str) {
        Some("content") => {
            |text| serde_json::json!({"type": "content", "content": {"type": "text", "text": text}})
        }
        Some(_) => |text| serde_json::json!({"type": "text", "text": text}),
        None => Value::String,
    }
}

/// The parts of `node` that may be replaced whole (below the top, not under
/// a key that is never rewritten, holding no key that identifies the
/// message, and bigger than a note), with their place in [`walk`]'s order.
/// Returns the size of `node` and whether it holds an identifying key.
fn parts(node: &Value, depth: usize, next: &mut usize, found: &mut Vec<Part>) -> (usize, bool) {
    let index = *next;
    *next += 1;
    let mut biggest = 0;
    let mut kept = 0;
    let (size, identity) = match node {
        Value::Object(map) => {
            let mut size = 1 + map.len().max(1);
            let mut identity = false;
            for (key, value) in map {
                let key_len = json_len(&Value::String(key.clone())) + 1;
                size += key_len;
                if KEEP_KEYS.contains(&key.as_str()) {
                    identity |= depth <= IDENTITY_DEPTH && IDENTITY_KEYS.contains(&key.as_str());
                    let len = json_len(value);
                    kept += key_len + len + 1;
                    size += len;
                } else {
                    let child = *next;
                    let (len, holds) = parts(value, depth + 1, next, found);
                    if let Some(p) = found.last().filter(|p| p.index == child) {
                        biggest = biggest.max(p.size);
                    }
                    size += len;
                    identity |= holds;
                }
            }
            (size, identity)
        }
        Value::Array(items) => {
            let mut size = 1 + items.len().max(1);
            let mut identity = false;
            for value in items {
                let child = *next;
                let (len, holds) = parts(value, depth + 1, next, found);
                if let Some(p) = found.last().filter(|p| p.index == child) {
                    biggest = biggest.max(p.size);
                }
                size += len;
                identity |= holds;
            }
            (size, identity)
        }
        other => (json_len(other), false),
    };
    if depth > 0 && !identity && size > 256 {
        found.push(Part {
            index,
            end: *next,
            size,
            kept,
            bulk_inside: biggest * 10 >= size * 9,
        });
    }
    (size, identity)
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

    /// Guards the D5 fix in ce57f923: a last line without a newline kept
    /// its `\r`, unlike every other line.
    #[tokio::test]
    async fn a_last_line_without_a_newline_loses_its_cr_too() {
        let got = lines(b"a\r", 100).await;
        assert!(matches!(&got[..], [Line::Text(t)] if t == b"a"), "{got:?}");
        let got = lines(b"a\r\nb\r", 100).await;
        assert!(
            matches!(&got[..], [Line::Text(a), Line::Text(b)] if a == b"a" && b == b"b"),
            "{got:?}"
        );
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

    /// The rewrite of `line`, which must fit `limit`.
    fn fitted(line: &str, limit: usize) -> Value {
        let shrunk = shrink(line.as_bytes(), limit).expect("a rewrite");
        let size = serde_json::to_vec(&shrunk.value).unwrap().len();
        assert!(shrunk.fits && size <= limit, "{size} bytes, limit {limit}");
        shrunk.value
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
        let shrunk = fitted(&line, DEFAULT_LINE_LIMIT);
        let update = &shrunk["params"]["update"];
        assert_eq!(update["status"], "completed");
        assert_eq!(update["toolCallId"], "t1");
        assert_eq!(
            update["content"][0]["content"],
            json!({"type": "text", "text": "[BenchFlow removed image data (image/png) (3000000 bytes): the ACP line was over the 1048576-byte limit]"})
        );

        // Long text is cut, keeping its start.
        let text = format!("start {}", "word ".repeat(400_000));
        let line = json!({"jsonrpc": "2.0", "method": "x", "params": {"text": text}}).to_string();
        let shrunk = fitted(&line, DEFAULT_LINE_LIMIT);
        let cut = shrunk["params"]["text"].as_str().unwrap();
        assert!(cut.starts_with("start word"));
        assert!(cut.ends_with("the ACP line was over the 1048576-byte limit]"));
        assert!(shrink(b"not json", 10).is_none());
    }

    #[test]
    fn a_line_of_too_many_values_loses_its_bulk_before_it_is_parsed() {
        // As values, a 256 MiB line of `0,` took about 9 GB. A 2.2 MB
        // tool_call_update whose rawOutput holds 1.1M values was then
        // dropped whole; now the rawOutput is replaced, as bytes, and the
        // call still closes.
        let values = format!("[{}0]", "0,".repeat(1_100_000));
        let line = format!(
            r#"{{"jsonrpc":"2.0","method":"session/update","params":{{"sessionId":"s","update":{{"sessionUpdate":"tool_call_update","toolCallId":"t1","status":"completed","rawOutput":{values}}}}}}}"#
        );
        assert!(tokens(line.as_bytes()) > MAX_TOKENS);
        let shrunk = fitted(&line, DEFAULT_LINE_LIMIT);
        let update = &shrunk["params"]["update"];
        assert_eq!(
            (&update["toolCallId"], &update["status"]),
            (&json!("t1"), &json!("completed"))
        );
        let raw = update["rawOutput"].as_array().expect("still an array");
        assert!(
            raw[0]
                .as_str()
                .unwrap()
                .starts_with("[BenchFlow removed part of this message (2200003 bytes)")
        );

        // A top-level array of a million numbers has nothing that may go:
        // it is noise, unread.
        let line = format!("[{}0]", "0,".repeat(MAX_TOKENS));
        assert_eq!(tokens(line.as_bytes()), MAX_TOKENS + 2);
        assert!(shrink(line.as_bytes(), DEFAULT_LINE_LIMIT).is_none());
        // Brackets and commas inside strings are text, not tokens.
        assert_eq!(tokens(br#"{"a":"[,:{\"\\",b:[1]}"#), 6);
    }

    #[test]
    fn a_replaced_part_keeps_its_type() {
        // 600 texts of 5,000 characters: cutting the 64 longest, as 0.8
        // does, leaves 2.6 MB. The content stays an array (the trajectory
        // reads only arrays): its first blocks, then one note block.
        let blocks: Vec<Value> = (0..600)
            .map(|i| json!({"type": "content", "content": {"type": "text", "text": format!("{i:05}{}", "é".repeat(4995))}}))
            .collect();
        let line = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "s", "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "t1",
                "status": "completed",
                "content": blocks,
            }},
        })
        .to_string();
        let shrunk = fitted(&line, DEFAULT_LINE_LIMIT);
        let update = &shrunk["params"]["update"];
        assert_eq!(update["toolCallId"], "t1");
        let content = update["content"].as_array().expect("still an array");
        assert!(content.len() > 50, "{} blocks kept", content.len() - 1);
        assert!(
            content[0]["content"]["text"]
                .as_str()
                .unwrap()
                .starts_with("00000")
        );
        let last = &content[content.len() - 1];
        assert_eq!(last["type"], "content");
        assert!(
            last["content"]["text"]
                .as_str()
                .unwrap()
                .contains(" of 600 items of part of this message")
        );

        // A top-level array of many parts loses the largest ones, largest first.
        let parts: Vec<Value> = (0..40).map(|i| json!("x ".repeat(50 * (i + 1)))).collect();
        let shrunk = fitted(&Value::Array(parts).to_string(), 20_000);
        let kept = shrunk.as_array().unwrap();
        assert_eq!(kept[0], json!("x ".repeat(50)));
        assert!(
            kept[39]
                .as_str()
                .unwrap()
                .starts_with("[BenchFlow removed part")
        );
    }

    #[test]
    fn the_part_that_holds_the_bulk_goes_not_its_small_siblings() {
        // Two bulky parts, neither enough alone: `_meta` was replaced whole
        // (it was the largest), losing the toolName the trajectory names
        // the call by. Now the array inside it that holds the bulk goes.
        let patch: Vec<Value> = (0..300)
            .map(|i| json!(format!("{i} {}", "-".repeat(5000))))
            .collect();
        let blocks: Vec<Value> = (0..300)
            .map(|_| json!({"type": "content", "content": {"type": "text", "text": "+".repeat(3000)}}))
            .collect();
        let line = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "s", "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "t1",
                "_meta": {"claudeCode": {"toolName": "Write", "toolResponse": {"structuredPatch": patch}}},
                "content": blocks,
            }},
        })
        .to_string();
        let shrunk = fitted(&line, 200_000);
        let update = &shrunk["params"]["update"];
        assert_eq!(update["_meta"]["claudeCode"]["toolName"], "Write");
        assert!(update["_meta"]["claudeCode"]["toolResponse"]["structuredPatch"].is_array());
        assert!(update["content"].is_array());
    }

    #[test]
    fn an_id_deep_in_a_tools_output_does_not_pin_it() {
        // `id` marks the message only in its envelope; tool output full of
        // objects with an `id` was never replaced and went on over the limit.
        let leaves: Vec<Value> = (0..100)
            .map(|i| json!({"id": i, "t": "x".repeat(300)}))
            .collect();
        let line = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "s", "update": {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "t1",
                "rawOutput": {"id": 1, "c": {"id": 2, "c": leaves}},
            }},
        })
        .to_string();
        let shrunk = fitted(&line, 10_000);
        let update = &shrunk["params"]["update"];
        assert_eq!(update["toolCallId"], "t1");
        assert_eq!(update["rawOutput"]["c"]["id"], 2);
        assert!(update["rawOutput"]["c"]["c"].is_array());
    }

    #[test]
    fn an_object_that_keeps_most_of_itself_is_not_counted_as_saved() {
        // The harness review's input: an object deep enough to be replaced
        // whose never-rewritten keys hold most of it. Its saving was planned
        // as its whole size, so it was the one part replaced, it saved 59
        // bytes, and the line stayed over the limit.
        let cjk = "漢".repeat(24);
        let x: Map<String, Value> = [
            "type",
            "mimeType",
            "jsonrpc",
            "method",
            "toolCallId",
            "sessionUpdate",
            "text",
            "data",
        ]
        .iter()
        .map(|k| (k.to_string(), json!(cjk)))
        .collect();
        let many: Map<String, Value> = (0..42)
            .map(|i| (format!("t{i}"), json!("abcd ".repeat(20))))
            .collect();
        let line = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"update": {"id": 1, "content": x}, "many": many},
        })
        .to_string();
        for limit in (4900..5100).step_by(7) {
            let shrunk = fitted(&line, limit);
            assert_eq!(shrunk["params"]["update"]["id"], 1);
        }
    }

    #[test]
    fn a_line_that_cannot_fit_is_passed_on_over_the_limit() {
        // What identifies the message is never cut: a title over the limit
        // (Claude's ACP adapter titles a call with its shell command) goes
        // on whole, as in 0.8, rather than being lost.
        let line = json!({
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": "s", "update": {
                "sessionUpdate": "tool_call",
                "toolCallId": "t1",
                "title": "cat <<EOF\n".to_string() + &"x".repeat(5000),
                "rawInput": {"command": "y".repeat(3000)},
            }},
        })
        .to_string();
        let shrunk = shrink(line.as_bytes(), 4096).expect("passed on");
        assert!(!shrunk.fits);
        let update = &shrunk.value["params"]["update"];
        assert_eq!(update["toolCallId"], "t1");
        assert!(update["title"].as_str().unwrap().len() > 5000);
        // What could go, went: the string that held rawInput's bulk.
        assert!(
            update["rawInput"]["command"]
                .as_str()
                .unwrap()
                .starts_with("[BenchFlow removed part")
        );
    }
}
