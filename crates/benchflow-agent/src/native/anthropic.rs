//! Anthropic messages: the request body and the answer.

use serde_json::{Map, Value, json};

use super::chat::{
    AssistantMessage, ChatRequest, ChatResponse, DEFAULT_ANTHROPIC_MAX_TOKENS, FinishReason,
    Message, TokenUsage, ToolCallRequest,
};
use super::openai::{MALFORMED_ARGUMENTS, number};

/// The API version header value.
pub const VERSION: &str = "2023-06-01";

fn tool_input(arguments: &str) -> Value {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(v @ Value::Object(_)) => v,
        _ => serde_json::from_str(MALFORMED_ARGUMENTS).unwrap_or_else(|_| json!({})),
    }
}

/// The JSON body for `POST /messages`. Consecutive messages of one role
/// are merged (tool results and the next user text share a user turn), as
/// the API requires roles to alternate.
pub fn request_body(model: &str, req: &ChatRequest) -> Value {
    let mut turns: Vec<(String, Vec<Value>)> = Vec::new();
    let mut push = |role: &str, blocks: Vec<Value>| match turns.last_mut() {
        Some((r, existing)) if r == role => existing.extend(blocks),
        _ => turns.push((role.to_string(), blocks)),
    };
    for m in &req.messages {
        match m {
            Message::User(text) => push("user", vec![json!({"type": "text", "text": text})]),
            Message::UserImages { text, images } => {
                let mut blocks: Vec<Value> = Vec::new();
                if !text.trim().is_empty() {
                    blocks.push(json!({"type": "text", "text": text}));
                }
                for image in images {
                    blocks.push(json!({"type": "image", "source": {
                        "type": "base64", "media_type": image.media_type, "data": image.base64}}));
                }
                push("user", blocks);
            }
            Message::Assistant(a) => {
                let mut blocks: Vec<Value> = a.thinking_blocks.clone();
                if !a.text.is_empty() {
                    blocks.push(json!({"type": "text", "text": a.text}));
                }
                for c in &a.tool_calls {
                    blocks.push(json!({"type": "tool_use", "id": c.id, "name": c.name,
                                       "input": tool_input(&c.arguments)}));
                }
                if blocks.is_empty() {
                    blocks.push(json!({"type": "text", "text": "(no answer)"}));
                }
                push("assistant", blocks);
            }
            Message::Tool { call_id, content } => push(
                "user",
                vec![json!({"type": "tool_result", "tool_use_id": call_id, "content": content})],
            ),
        }
    }
    let messages: Vec<Value> = turns
        .into_iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    let mut body = Map::new();
    body.insert("model".into(), model.into());
    body.insert(
        "max_tokens".into(),
        req.max_tokens
            .unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS)
            .into(),
    );
    if let Some(system) = req.system.as_ref().filter(|s| !s.is_empty()) {
        body.insert("system".into(), system.clone().into());
    }
    body.insert("messages".into(), Value::Array(messages));
    if !req.tools.is_empty() {
        body.insert(
            "tools".into(),
            req.tools
                .iter()
                .map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.parameters}))
                .collect(),
        );
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".into(), t.into());
    }
    if !req.stop.is_empty() {
        body.insert("stop_sequences".into(), req.stop.clone().into());
    }
    Value::Object(body)
}

/// Read a messages answer.
pub fn parse_response(body: &Value) -> Result<ChatResponse, String> {
    let blocks = body
        .get("content")
        .and_then(Value::as_array)
        .ok_or("the answer has no content")?;
    let mut message = AssistantMessage::default();
    let mut reasoning = String::new();
    for b in blocks {
        match b.get("type").and_then(Value::as_str) {
            Some("text") => message
                .text
                .push_str(b.get("text").and_then(Value::as_str).unwrap_or_default()),
            Some("tool_use") => message.tool_calls.push(ToolCallRequest {
                id: b
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: b
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                arguments: b.get("input").cloned().unwrap_or(json!({})).to_string(),
            }),
            Some("thinking") => {
                reasoning.push_str(
                    b.get("thinking")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                );
                message.thinking_blocks.push(b.clone());
            }
            Some("redacted_thinking") => message.thinking_blocks.push(b.clone()),
            _ => {}
        }
    }
    message.reasoning = (!reasoning.is_empty()).then_some(reasoning);
    let finish = body
        .get("stop_reason")
        .and_then(Value::as_str)
        .map(FinishReason::from_anthropic)
        .unwrap_or(FinishReason::Stop);
    let usage = body.get("usage").cloned().unwrap_or(Value::Null);
    let n = |k: &str| {
        number(usage.get(k))
            .filter(|n| *n >= 0.0)
            .map(|n| n as u64)
            .unwrap_or(0)
    };
    let (input, read, write) = (
        n("input_tokens"),
        n("cache_read_input_tokens"),
        n("cache_creation_input_tokens"),
    );
    Ok(ChatResponse {
        message,
        finish,
        usage: TokenUsage {
            // Anthropic counts cached tokens apart from `input_tokens`; the
            // total prompt is their sum.
            input_tokens: input + read + write,
            output_tokens: n("output_tokens"),
            cache_read_tokens: read,
            cache_write_tokens: write,
            reasoning_tokens: 0,
            // The Vercel AI Gateway reports cost in its provider metadata.
            cost_usd: number(usage.get("cost"))
                .or_else(|| number(body.pointer("/provider_metadata/gateway/cost"))),
        },
        model: body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

#[cfg(test)]
mod tests {
    use super::super::chat::ToolSpec;
    use super::*;

    #[test]
    fn tool_results_and_user_text_share_a_turn() {
        let req = ChatRequest {
            system: Some("sys".into()),
            messages: vec![
                Message::user("go"),
                Message::Assistant(AssistantMessage {
                    text: "ok".into(),
                    tool_calls: vec![ToolCallRequest {
                        id: "t1".into(),
                        name: "bash".into(),
                        arguments: r#"{"command":"ls"}"#.into(),
                    }],
                    ..AssistantMessage::default()
                }),
                Message::Tool {
                    call_id: "t1".into(),
                    content: "a.txt".into(),
                },
                Message::user("continue"),
            ],
            tools: vec![ToolSpec {
                name: "bash".into(),
                description: "d".into(),
                parameters: json!({"type": "object"}),
            }],
            ..ChatRequest::default()
        };
        let body = request_body("claude-haiku-4-5", &req);
        assert_eq!(body["system"], "sys");
        assert_eq!(body["max_tokens"], DEFAULT_ANTHROPIC_MAX_TOKENS);
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[1]["content"][1]["input"]["command"], "ls");
        assert_eq!(msgs[2]["content"][0]["type"], "tool_result");
        assert_eq!(msgs[2]["content"][1]["text"], "continue");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn answers_give_text_calls_thinking_and_usage() {
        let body = json!({"model": "claude-haiku-4-5", "stop_reason": "tool_use",
            "content": [{"type": "thinking", "thinking": "plan", "signature": "s"},
                        {"type": "text", "text": "Running."},
                        {"type": "tool_use", "id": "toolu_1", "name": "bash", "input": {"command": "ls"}}],
            "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100}});
        let r = parse_response(&body).unwrap();
        assert_eq!(r.message.text, "Running.");
        assert_eq!(r.message.reasoning.as_deref(), Some("plan"));
        assert_eq!(r.message.thinking_blocks.len(), 1);
        assert_eq!(r.message.tool_calls[0].arguments, r#"{"command":"ls"}"#);
        assert_eq!(r.finish, FinishReason::ToolCalls);
        assert_eq!(r.usage.input_tokens, 110);
        assert_eq!(r.usage.cache_read_tokens, 100);
        assert_eq!(r.usage.cost_usd, None);
    }

    #[test]
    fn gateway_cost_is_read_from_provider_metadata() {
        let body = json!({"content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "provider_metadata": {"gateway": {"cost": "0.000221264"}}});
        assert_eq!(
            parse_response(&body).unwrap().usage.cost_usd,
            Some(0.000221264)
        );
    }
}
