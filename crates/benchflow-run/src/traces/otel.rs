//! An ATIF trajectory as OpenTelemetry spans in OTLP JSON, ported from
//! Harbor's `harbor-atif2otel` (`convert.py`, `ids.py`) so the same
//! trajectory gives the same line, byte for byte.
//!
//! One trace per trajectory: an `AGENT` root span, a `turn-<n>` span per
//! user turn when there are several, an `LLM` span per agent step, a `TOOL`
//! span per tool call, and a `CHAIN` span per context-management step.
//! Embedded subagent trajectories become spans under the tool call that
//! started them. Span attributes follow OpenInference's names. Trace and
//! span ids are SHA-256 of seeds, so exporting twice gives the same ids.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use super::pyjson::{Dumps, dumps, py_str};

/// What the resource and scope name as the producer.
const SDK_NAME: &str = "benchflow";
const SDK_VERSION: &str = env!("CARGO_PKG_VERSION");
const TOOL_ARG_MAX_BYTES: usize = 2048;
/// The longest attribute value, in bytes, before it is truncated.
pub const MAX_ATTRIBUTE_BYTES: usize = 10240;

fn sha(seed: &str) -> [u8; 32] {
    Sha256::digest(seed.as_bytes()).into()
}

fn trace_id(seed: &str) -> String {
    hex::encode(&sha(seed)[..16])
}

fn span_id(seed: &str) -> String {
    hex::encode(&sha(seed)[..8])
}

/// Cut `s` to `max` bytes of UTF-8, ending in `...[truncated]`.
fn truncate(s: &str, max: usize) -> String {
    const SUFFIX: &str = "...[truncated]";
    if max == 0 {
        return String::new();
    }
    if s.len() <= max {
        return s.to_string();
    }
    if max <= SUFFIX.len() {
        return SUFFIX[..max].to_string();
    }
    let mut limit = max - SUFFIX.len();
    while limit > 0 && !s.is_char_boundary(limit) {
        limit -= 1;
    }
    format!("{}{SUFFIX}", &s[..limit])
}

/// An attribute value; `None` leaves the attribute out.
enum Attr {
    Str(String),
    Int(i64),
    Double(f64),
    Bool(bool),
}

/// An ATIF JSON value as `_make_kv` turns it into an attribute.
fn attr_of(v: Option<&Value>) -> Option<Attr> {
    match v? {
        Value::Null => None,
        Value::Bool(b) => Some(Attr::Bool(*b)),
        Value::Number(n) if !n.is_f64() => n
            .as_i64()
            .map(Attr::Int)
            .or_else(|| n.as_f64().map(Attr::Double)),
        Value::Number(n) => n.as_f64().map(Attr::Double),
        Value::String(s) => Some(Attr::Str(s.clone())),
        other => Some(Attr::Str(dumps(other, Dumps::ASCII))),
    }
}

fn s(v: impl Into<String>) -> Option<Attr> {
    Some(Attr::Str(v.into()))
}

fn attrs(pairs: Vec<(&str, Option<Attr>)>) -> Vec<Value> {
    pairs
        .into_iter()
        .filter_map(|(k, v)| {
            let value = match v? {
                Attr::Str(s) => json!({"stringValue": s}),
                Attr::Int(i) => json!({"intValue": i.to_string()}),
                Attr::Double(d) => json!({"doubleValue": d}),
                Attr::Bool(b) => json!({"boolValue": b}),
            };
            Some(json!({"key": k, "value": value}))
        })
        .collect()
}

/// An ISO 8601 time as Unix nanoseconds, computed as Python's
/// `int(datetime.fromisoformat(ts).timestamp() * 1e9)` (microseconds,
/// then a float); 0 when absent or unreadable.
fn nanos(ts: Option<&Value>) -> i64 {
    let Some(ts) = ts.and_then(Value::as_str).filter(|t| !t.is_empty()) else {
        return 0;
    };
    let aware = ["%Y-%m-%dT%H:%M:%S%.f%:z", "%Y-%m-%d %H:%M:%S%.f%:z"]
        .iter()
        .find_map(|f| chrono::DateTime::parse_from_str(&ts.replace('Z', "+00:00"), f).ok())
        .map(|d| d.timestamp_micros());
    let naive = || {
        ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%d %H:%M:%S%.f"]
            .iter()
            .find_map(|f| chrono::NaiveDateTime::parse_from_str(ts, f).ok())
            .map(|d| d.and_utc().timestamp_micros())
    };
    let parsed = aware.or_else(naive).ok_or(());
    match parsed {
        Ok(micros) => ((micros as f64 / 1e6) * 1e9) as i64,
        Err(_) => 0,
    }
}

fn get<'a>(v: &'a Value, k: &str) -> Option<&'a Value> {
    v.get(k).filter(|x| !x.is_null())
}

fn str_of<'a>(v: &'a Value, k: &str) -> Option<&'a str> {
    get(v, k).and_then(Value::as_str)
}

/// `_stringify_message`: text, with media parts as `[image: <path>]`.
pub(crate) fn stringify(message: Option<&Value>) -> String {
    match message {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .map(|p| match p {
                Value::Object(_) => {
                    let kind = str_of(p, "type").unwrap_or("");
                    if kind == "image" || kind == "audio" {
                        let source = get(p, "source").cloned().unwrap_or_else(|| json!({}));
                        let path = source
                            .get("path")
                            .or_else(|| source.get("url"))
                            .map(py_str)
                            .unwrap_or_else(|| "unknown".into());
                        format!("[{kind}: {path}]")
                    } else {
                        p.get("text").map(py_str).unwrap_or_default()
                    }
                }
                Value::String(s) => s.clone(),
                _ => String::new(),
            })
            .collect(),
        Some(other) => py_str(other),
    }
}

fn source(step: &Value) -> &str {
    str_of(step, "source").unwrap_or("")
}

fn add(a: Option<&Value>, b: Option<&Value>) -> Option<Attr> {
    let (a, b) = (a.and_then(Value::as_i64), b.and_then(Value::as_i64));
    if a.is_none() && b.is_none() {
        return None;
    }
    Some(Attr::Int(a.unwrap_or(0) + b.unwrap_or(0)))
}

struct Ctx<'a> {
    trace_id: String,
    seed_base: String,
    trace_seed: &'a str,
    service: &'a str,
    max_bytes: usize,
}

fn span(
    ctx: &Ctx<'_>,
    id: &str,
    parent: Option<&str>,
    name: &str,
    start: i64,
    end: i64,
    attributes: Vec<Value>,
) -> Value {
    let mut m = Map::new();
    m.insert("traceId".into(), json!(ctx.trace_id));
    m.insert("spanId".into(), json!(id));
    if let Some(p) = parent.filter(|p| !p.is_empty()) {
        m.insert("parentSpanId".into(), json!(p));
    }
    if !name.is_empty() {
        m.insert("name".into(), json!(name));
    }
    m.insert("kind".into(), json!("SPAN_KIND_INTERNAL"));
    if start != 0 {
        m.insert("startTimeUnixNano".into(), json!(start.to_string()));
    }
    if end != 0 {
        m.insert("endTimeUnixNano".into(), json!(end.to_string()));
    }
    if !attributes.is_empty() {
        m.insert("attributes".into(), Value::Array(attributes));
    }
    m.insert("status".into(), json!({"code": "STATUS_CODE_OK"}));
    Value::Object(m)
}

/// The spans of one trajectory, its root first.
fn spans(traj: &Value, trace_seed: &str, service: &str, max_bytes: usize) -> Vec<Value> {
    let tid = trace_id(trace_seed);
    let seed_base = match str_of(traj, "trajectory_id") {
        Some(t) => format!("{tid}:{t}"),
        None => tid.clone(),
    };
    let ctx = Ctx {
        trace_id: tid,
        seed_base,
        trace_seed,
        service,
        max_bytes,
    };
    let root_id = span_id(&format!("{}:root", ctx.seed_base));
    let empty = json!({});
    let agent = get(traj, "agent").unwrap_or(&empty);
    let all: Vec<Value> = get(traj, "steps")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut steps: Vec<Value> = all
        .iter()
        .filter(|s| !get(s, "is_copied_context").is_some_and(truthy))
        .cloned()
        .collect();
    if steps.is_empty() {
        steps = all;
    }
    let mut observations: Map<String, Value> = Map::new();
    for step in &steps {
        let Some(results) = get(step, "observation")
            .and_then(|o| o.get("results"))
            .and_then(Value::as_array)
        else {
            continue;
        };
        for r in results.iter().filter(|r| r.is_object()) {
            if let Some(id) = str_of(r, "source_call_id").filter(|i| !i.is_empty()) {
                observations.insert(id.to_string(), r.clone());
            }
        }
    }
    let mut subagents: Map<String, Value> = Map::new();
    for sub in get(traj, "subagent_trajectories")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        if let Some(id) = str_of(sub, "trajectory_id").filter(|i| !i.is_empty()) {
            subagents.insert(id.to_string(), sub.clone());
        }
    }
    let first_ts = steps.first().map_or(0, |s| nanos(s.get("timestamp")));
    let last_ts = steps.last().map_or(first_ts, |s| nanos(s.get("timestamp")));
    let metrics = get(traj, "final_metrics")
        .cloned()
        .unwrap_or_else(|| json!({}));
    let input = steps
        .iter()
        .find(|s| source(s) == "user")
        .map(|s| stringify(s.get("message")))
        .unwrap_or_default();
    let output = steps
        .iter()
        .rev()
        .find(|s| source(s) == "agent")
        .map(|s| stringify(s.get("message")))
        .unwrap_or_default();
    let mut root_attrs = attrs(vec![
        ("openinference.span.kind", s("AGENT")),
        ("agent.name", attr_of(agent.get("name"))),
        ("agent.version", attr_of(agent.get("version"))),
        ("session.id", attr_of(traj.get("session_id"))),
        ("trajectory.id", attr_of(traj.get("trajectory_id"))),
        ("atif.schema_version", attr_of(traj.get("schema_version"))),
        ("llm.model_name", attr_of(agent.get("model_name"))),
        ("input.value", s(truncate(&input, max_bytes))),
        ("output.value", s(truncate(&output, max_bytes))),
        (
            "llm.token_count.prompt",
            attr_of(metrics.get("total_prompt_tokens")),
        ),
        (
            "llm.token_count.completion",
            attr_of(metrics.get("total_completion_tokens")),
        ),
        (
            "llm.token_count.total",
            add(
                metrics.get("total_prompt_tokens"),
                metrics.get("total_completion_tokens"),
            ),
        ),
        ("llm.cost.total", attr_of(metrics.get("total_cost_usd"))),
    ]);
    for (i, td) in get(agent, "tool_definitions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let schema = match td {
            Value::Object(_) => dumps(td, Dumps::ASCII),
            other => py_str(other),
        };
        root_attrs.extend(attrs(vec![(
            &format!("llm.tools.{i}.tool.json_schema"),
            s(truncate(&schema, TOOL_ARG_MAX_BYTES)),
        )]));
    }
    let root_name = match agent.get("name") {
        None => "agent".into(),
        Some(Value::Null) => String::new(),
        Some(v) => py_str(v),
    };
    let mut out = vec![span(
        &ctx, &root_id, None, &root_name, first_ts, last_ts, root_attrs,
    )];

    // Turns: a new one at each user step after the first.
    let mut turns: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut seen_user = false;
    for (i, step) in steps.iter().enumerate() {
        if source(step) == "user" {
            if seen_user && !current.is_empty() {
                turns.push(std::mem::take(&mut current));
            }
            seen_user = true;
        }
        current.push(i);
    }
    if !current.is_empty() {
        turns.push(current);
    }
    let multi = turns.len() > 1;
    for (t, indices) in turns.iter().enumerate() {
        let parent = if multi {
            let id = span_id(&format!("{}:turn:{t}", ctx.seed_base));
            let start = nanos(steps[indices[0]].get("timestamp"));
            let end = turns
                .get(t + 1)
                .map_or(last_ts, |next| nanos(steps[next[0]].get("timestamp")));
            out.push(span(
                &ctx,
                &id,
                Some(&root_id),
                &format!("turn-{t}"),
                start,
                end,
                attrs(vec![("openinference.span.kind", s("AGENT"))]),
            ));
            id
        } else {
            root_id.clone()
        };
        for &i in indices {
            let step = &steps[i];
            let ctx_mgmt = get(step, "extra")
                .and_then(|e| get(e, "context_management"))
                .filter(|v| truthy(v));
            if source(step) == "system"
                && let Some(action) = ctx_mgmt
            {
                let id = span_id(&format!("{}:chain:{i}", ctx.seed_base));
                let ts = nanos(step.get("timestamp"));
                out.push(span(
                    &ctx,
                    &id,
                    Some(&parent),
                    "context-management",
                    ts,
                    ts,
                    attrs(vec![
                        ("openinference.span.kind", s("CHAIN")),
                        ("context_management.action", s(py_str(action))),
                    ]),
                ));
                continue;
            }
            if source(step) != "agent" {
                continue;
            }
            let calls: Vec<Value> = get(step, "tool_calls")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let ts = nanos(step.get("timestamp"));
            let next_ts = steps
                .get(i + 1)
                .map_or(last_ts, |n| nanos(n.get("timestamp")));
            let llm_calls = get(step, "llm_call_count").and_then(Value::as_i64);
            if llm_calls != Some(0) {
                let inputs: Vec<Value> = indices
                    .iter()
                    .take_while(|&&j| j < i)
                    .map(|&j| &steps[j])
                    .filter(|p| matches!(source(p), "user" | "system"))
                    .map(|p| json!({"role": source(p), "content": stringify(p.get("message"))}))
                    .collect();
                let step_metrics = get(step, "metrics").cloned().unwrap_or_else(|| json!({}));
                // Python's `step.get("model_name") or agent.get("model_name")`.
                let model_value = match step.get("model_name") {
                    Some(v) if truthy(v) => Some(v),
                    _ => agent.get("model_name"),
                };
                let mut pairs = vec![
                    ("openinference.span.kind", s("LLM")),
                    ("llm.model_name", attr_of(model_value)),
                    (
                        "llm.token_count.prompt",
                        attr_of(step_metrics.get("prompt_tokens")),
                    ),
                    (
                        "llm.token_count.completion",
                        attr_of(step_metrics.get("completion_tokens")),
                    ),
                    (
                        "llm.token_count.prompt_details.cache_read",
                        attr_of(step_metrics.get("cached_tokens")),
                    ),
                    ("llm.cost.total", attr_of(step_metrics.get("cost_usd"))),
                    (
                        "input.value",
                        s(truncate(
                            &dumps(&Value::Array(inputs), Dumps::ASCII),
                            max_bytes,
                        )),
                    ),
                    (
                        "output.value",
                        s(truncate(&stringify(step.get("message")), max_bytes)),
                    ),
                ];
                if let Some(r) = get(step, "reasoning_content").filter(|v| truthy(v)) {
                    pairs.push((
                        "metadata.reasoning_content",
                        s(truncate(&py_str(r), max_bytes)),
                    ));
                }
                if let Some(n) = llm_calls.filter(|n| *n > 1) {
                    pairs.push(("metadata.aggregated_llm_calls", Some(Attr::Int(n))));
                }
                let id = span_id(&format!("{}:llm:{i}", ctx.seed_base));
                let name = model_value
                    .filter(|v| truthy(v))
                    .map(py_str)
                    .unwrap_or_else(|| "llm".into());
                out.push(span(
                    &ctx,
                    &id,
                    Some(&parent),
                    &name,
                    ts,
                    next_ts,
                    attrs(pairs),
                ));
            }
            tool_spans(
                &ctx,
                &mut out,
                &parent,
                i,
                &calls,
                &observations,
                &subagents,
                ts,
            );
        }
    }
    out
}

fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

#[allow(clippy::too_many_arguments)]
fn tool_spans(
    ctx: &Ctx<'_>,
    out: &mut Vec<Value>,
    parent: &str,
    step_idx: usize,
    calls: &[Value],
    observations: &Map<String, Value>,
    subagents: &Map<String, Value>,
    base_ts: i64,
) {
    for (n, call) in calls.iter().enumerate() {
        let call_id = call.get("tool_call_id").map(py_str).unwrap_or_default();
        let name_value = match call.get("function_name") {
            None => Some("unknown_tool".to_string()),
            Some(Value::Null) => None,
            Some(v) => Some(py_str(v)),
        };
        let name = name_value.clone().unwrap_or_default();
        let args = match call.get("arguments") {
            None => "{}".to_string(),
            Some(a @ Value::Object(_)) => dumps(a, Dumps::ASCII),
            Some(a) if !truthy(a) => String::new(),
            Some(a) => py_str(a),
        };
        let id = span_id(&format!("{}:tool:{step_idx}:{n}", ctx.seed_base));
        let ts = base_ts + (n as i64 + 1) * 1_000_000;
        let empty = json!({});
        let obs = observations.get(&call_id).unwrap_or(&empty);
        let result = match obs.get("content") {
            Some(c) if truthy(c) => stringify(Some(c)),
            _ => String::new(),
        };
        out.push(span(
            ctx,
            &id,
            Some(parent),
            &name,
            ts,
            ts,
            attrs(vec![
                ("openinference.span.kind", s("TOOL")),
                ("tool.name", name_value.map(Attr::Str)),
                ("input.value", s(truncate(&args, TOOL_ARG_MAX_BYTES))),
                ("output.value", s(truncate(&result, TOOL_ARG_MAX_BYTES))),
            ]),
        ));
        let refs: Vec<Value> = match obs.get("subagent_trajectory_ref") {
            None => Vec::new(),
            Some(Value::Array(a)) => a.clone(),
            Some(other) => vec![other.clone()],
        };
        for (r, reference) in refs.iter().enumerate() {
            let Some(ref_id) = str_of(reference, "trajectory_id").filter(|i| !i.is_empty()) else {
                continue;
            };
            let Some(sub) = subagents.get(ref_id) else {
                continue;
            };
            let seed = format!("{}:subagent:{id}:{r}:{ref_id}", ctx.trace_seed);
            for mut sub_span in spans(sub, &seed, ctx.service, ctx.max_bytes) {
                if let Some(m) = sub_span.as_object_mut() {
                    m.insert("traceId".into(), json!(ctx.trace_id));
                    if !m.contains_key("parentSpanId") {
                        // Keep the field order: parentSpanId follows spanId.
                        let mut ordered = Map::new();
                        for (k, v) in std::mem::take(m) {
                            let after_span = k == "spanId";
                            ordered.insert(k, v);
                            if after_span {
                                ordered.insert("parentSpanId".into(), json!(id));
                            }
                        }
                        *m = ordered;
                    }
                }
                out.push(sub_span);
            }
        }
    }
}

/// One OTLP JSON line (`TracesData`) for a trajectory, as Harbor's
/// `export_trials` writes it with `trace_seed` set to the trial's name,
/// except that the resource and scope name BenchFlow as the producer.
pub fn trace_line(traj: &Value, trace_seed: &str, service: &str) -> String {
    let spans = spans(traj, trace_seed, service, MAX_ATTRIBUTE_BYTES);
    let data = json!({
        "resourceSpans": [{
            "resource": {"attributes": attrs(vec![
                ("service.name", s(service)),
                ("telemetry.sdk.language", s("rust")),
                ("telemetry.sdk.name", s(SDK_NAME)),
                ("telemetry.sdk.version", s(SDK_VERSION)),
            ])},
            "scopeSpans": [{
                "scope": {"name": SDK_NAME, "version": SDK_VERSION},
                "spans": spans,
            }],
        }]
    });
    dumps(&data, Dumps::ASCII)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncates_on_a_character_boundary() {
        assert_eq!(truncate("short", 100), "short");
        let long = "é".repeat(20);
        let t = truncate(&long, 20);
        assert!(t.ends_with("...[truncated]") && t.len() <= 20, "{t}");
        assert_eq!(truncate("abcdefghijklmnopqrstuvwxyz", 5), "...[t");
    }

    #[test]
    fn times_are_python_nanoseconds() {
        assert_eq!(nanos(Some(&json!("1970-01-01T00:00:01Z"))), 1_000_000_000);
        assert_eq!(
            nanos(Some(&json!("2026-10-10T12:37:47.560588+00:00"))),
            1_791_635_867_560_587_776
        );
        assert_eq!(nanos(Some(&json!("not a time"))), 0);
        assert_eq!(nanos(None), 0);
    }

    #[test]
    fn ids_are_deterministic() {
        let t = json!({"agent": {"name": "a"}, "steps": [{"source": "user", "message": "hi"}, {"source": "agent", "message": "ok"}]});
        assert_eq!(
            trace_line(&t, "trial-1", "benchflow"),
            trace_line(&t, "trial-1", "benchflow")
        );
        assert_ne!(
            trace_line(&t, "trial-1", "benchflow"),
            trace_line(&t, "trial-2", "benchflow")
        );
    }
}
