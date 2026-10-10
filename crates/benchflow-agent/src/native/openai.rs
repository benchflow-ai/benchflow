//! OpenAI-compatible chat completions: the request body and the answer.

use serde_json::{Map, Value, json};

use super::chat::{
    AssistantMessage, ChatRequest, ChatResponse, FinishReason, Message, TokenUsage, ToolCallRequest,
};

/// What a malformed tool call's arguments are replaced with before the
/// conversation goes back to the provider (Karotte's sanitization: some
/// providers reject a request whose history holds invalid JSON).
pub const MALFORMED_ARGUMENTS: &str = r#"{"_error": "malformed JSON in original arguments"}"#;

/// The arguments as they go back to the provider: valid JSON, or the
/// placeholder.
pub fn sanitized_arguments(arguments: &str) -> String {
    let trimmed = arguments.trim();
    if trimmed.is_empty() {
        return "{}".into();
    }
    match serde_json::from_str::<Value>(trimmed) {
        Ok(Value::Object(_)) => trimmed.to_string(),
        _ => MALFORMED_ARGUMENTS.into(),
    }
}

/// The JSON body for `POST /chat/completions`.
pub fn request_body(model: &str, req: &ChatRequest) -> Value {
    let mut messages = Vec::new();
    if let Some(system) = req.system.as_ref().filter(|s| !s.is_empty()) {
        messages.push(json!({"role": "system", "content": system}));
    }
    for m in &req.messages {
        messages.push(match m {
            Message::User(text) => json!({"role": "user", "content": text}),
            Message::UserImages { text, images } => {
                // An empty text part is refused by some servers.
                let mut parts: Vec<Value> = Vec::new();
                if !text.trim().is_empty() {
                    parts.push(json!({"type": "text", "text": text}));
                }
                for image in images {
                    parts.push(json!({"type": "image_url", "image_url": {"url": image.data_url()}}));
                }
                json!({"role": "user", "content": parts})
            }
            Message::Assistant(a) => {
                let mut msg = Map::new();
                msg.insert("role".into(), "assistant".into());
                msg.insert(
                    "content".into(),
                    if a.text.is_empty() && !a.tool_calls.is_empty() {
                        Value::Null
                    } else {
                        Value::String(a.text.clone())
                    },
                );
                if !a.tool_calls.is_empty() {
                    msg.insert(
                        "tool_calls".into(),
                        a.tool_calls
                            .iter()
                            .map(|c| {
                                json!({"id": c.id, "type": "function", "function": {
                                    "name": c.name, "arguments": sanitized_arguments(&c.arguments)}})
                            })
                            .collect(),
                    );
                    // Thinking models (DeepSeek's among them) want their
                    // reasoning back alongside the calls it led to.
                    if let Some(r) = a.reasoning.as_ref().filter(|r| !r.is_empty()) {
                        msg.insert("reasoning_content".into(), Value::String(r.clone()));
                    }
                }
                Value::Object(msg)
            }
            Message::Tool { call_id, content } => {
                json!({"role": "tool", "tool_call_id": call_id, "content": content})
            }
        });
    }
    let mut body = Map::new();
    body.insert("model".into(), model.into());
    body.insert("messages".into(), Value::Array(messages));
    if !req.tools.is_empty() {
        body.insert(
            "tools".into(),
            req.tools
                .iter()
                .map(|t| {
                    json!({"type": "function", "function": {
                        "name": t.name, "description": t.description, "parameters": t.parameters}})
                })
                .collect(),
        );
        body.insert("tool_choice".into(), "auto".into());
    }
    if let Some(n) = req.max_tokens {
        body.insert("max_tokens".into(), n.into());
    }
    if let Some(t) = req.temperature {
        body.insert("temperature".into(), t.into());
    }
    if !req.stop.is_empty() {
        body.insert("stop".into(), req.stop.clone().into());
    }
    if let Some(e) = &req.reasoning_effort {
        body.insert("reasoning_effort".into(), e.clone().into());
    }
    Value::Object(body)
}

/// Read a chat completion. `Err` says what was missing.
pub fn parse_response(body: &Value) -> Result<ChatResponse, String> {
    let choice = body
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|c| c.first())
        .ok_or("the answer has no choices")?;
    let message = choice.get("message").ok_or("the answer has no message")?;
    let text = match message.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    };
    let reasoning = ["reasoning_content", "reasoning"]
        .iter()
        .find_map(|k| message.get(*k).and_then(Value::as_str))
        .filter(|r| !r.is_empty())
        .map(str::to_string);
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .enumerate()
                .map(|(i, c)| {
                    let f = c.get("function").cloned().unwrap_or(Value::Null);
                    let arguments = match f.get("arguments") {
                        Some(Value::String(s)) => s.clone(),
                        Some(Value::Null) | None => String::new(),
                        Some(other) => other.to_string(),
                    };
                    ToolCallRequest {
                        id: c
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|s| !s.is_empty())
                            .map(str::to_string)
                            .unwrap_or_else(|| format!("call_{i}")),
                        name: f
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        arguments,
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .map(FinishReason::from_openai)
        .unwrap_or(FinishReason::Stop);
    Ok(ChatResponse {
        message: AssistantMessage {
            text,
            tool_calls,
            reasoning,
            thinking_blocks: Vec::new(),
        },
        finish,
        usage: parse_usage(body),
        model: body
            .get("model")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// A number given as a number or a string.
pub(crate) fn number(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

fn count(v: Option<&Value>) -> u64 {
    number(v)
        .filter(|n| *n >= 0.0)
        .map(|n| n as u64)
        .unwrap_or(0)
}

/// Usage and cost. Cost comes from `usage.cost` (Vercel AI Gateway,
/// OpenRouter) or the gateway's provider metadata.
fn parse_usage(body: &Value) -> TokenUsage {
    let usage = body.get("usage").cloned().unwrap_or(Value::Null);
    let cost = number(usage.get("cost"))
        .or_else(|| number(usage.get("cost_usd")))
        .or_else(|| {
            let meta = body
                .pointer("/choices/0/message/provider_metadata/gateway")
                .or_else(|| body.pointer("/provider_metadata/gateway"))?;
            number(meta.get("cost"))
        });
    TokenUsage {
        input_tokens: count(usage.get("prompt_tokens")),
        output_tokens: count(usage.get("completion_tokens")),
        cache_read_tokens: count(usage.pointer("/prompt_tokens_details/cached_tokens")),
        cache_write_tokens: count(usage.get("cache_creation_input_tokens")),
        reasoning_tokens: count(usage.pointer("/completion_tokens_details/reasoning_tokens")),
        cost_usd: cost,
    }
}

#[cfg(test)]
mod tests {
    use super::super::chat::ToolSpec;
    use super::*;

    #[test]
    fn requests_carry_tools_history_and_stop() {
        let req = ChatRequest {
            system: Some("be brief".into()),
            messages: vec![
                Message::user("hi"),
                Message::Assistant(AssistantMessage {
                    text: String::new(),
                    tool_calls: vec![ToolCallRequest {
                        id: "c1".into(),
                        name: "bash".into(),
                        arguments: "{not json".into(),
                    }],
                    reasoning: Some("think".into()),
                    thinking_blocks: Vec::new(),
                }),
                Message::Tool {
                    call_id: "c1".into(),
                    content: "ok".into(),
                },
            ],
            tools: vec![ToolSpec {
                name: "bash".into(),
                description: "run".into(),
                parameters: json!({"type": "object"}),
            }],
            max_tokens: Some(100),
            temperature: None,
            stop: vec!["</done>".into()],
            reasoning_effort: Some("high".into()),
        };
        let body = request_body("m", &req);
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][2]["content"], Value::Null);
        assert_eq!(
            body["messages"][2]["tool_calls"][0]["function"]["arguments"],
            MALFORMED_ARGUMENTS
        );
        assert_eq!(body["messages"][2]["reasoning_content"], "think");
        assert_eq!(body["messages"][3]["tool_call_id"], "c1");
        assert_eq!(body["tools"][0]["function"]["name"], "bash");
        assert_eq!(body["stop"][0], "</done>");
        assert_eq!(body["reasoning_effort"], "high");
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn answers_give_text_calls_usage_and_cost() {
        let body = json!({
            "model": "deepseek/deepseek-v4-flash-0731",
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": "Let me look.", "reasoning": "hmm",
                "tool_calls": [{"id": "call_a", "type": "function",
                    "function": {"name": "bash", "arguments": "{\"command\": \"ls\"}"}}]}}],
            "usage": {"prompt_tokens": 32, "completion_tokens": 20, "cost": 4.048e-05,
                      "prompt_tokens_details": {"cached_tokens": 8},
                      "completion_tokens_details": {"reasoning_tokens": 17}}
        });
        let r = parse_response(&body).unwrap();
        assert_eq!(r.message.text, "Let me look.");
        assert_eq!(r.message.reasoning.as_deref(), Some("hmm"));
        assert_eq!(r.message.tool_calls[0].name, "bash");
        assert_eq!(r.finish, FinishReason::ToolCalls);
        assert_eq!(r.usage.input_tokens, 32);
        assert_eq!(r.usage.cache_read_tokens, 8);
        assert_eq!(r.usage.reasoning_tokens, 17);
        assert_eq!(r.usage.cost_usd, Some(4.048e-05));
    }

    #[test]
    fn gateway_metadata_cost_is_read_when_usage_has_none() {
        let body = json!({"choices": [{"message": {"content": "x",
            "provider_metadata": {"gateway": {"cost": "0.5"}}}}], "usage": {}});
        assert_eq!(parse_response(&body).unwrap().usage.cost_usd, Some(0.5));
    }

    #[test]
    fn an_answer_without_choices_is_an_error() {
        assert!(parse_response(&json!({"error": "x"})).is_err());
    }
}
