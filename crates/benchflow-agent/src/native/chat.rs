//! A conversation, independent of the provider's wire format.
//!
//! The loops build [`ChatRequest`]s from these types; [`super::openai`] and
//! [`super::anthropic`] turn them into each API's JSON and read the answer
//! back into a [`ChatResponse`].

use serde::Serialize;
use serde_json::Value;

/// One message in a conversation.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    User(String),
    /// A user turn with images (computer-1's screenshots): the text, then
    /// each image.
    UserImages {
        text: String,
        images: Vec<Image>,
    },
    Assistant(AssistantMessage),
    /// The result of one tool call, answering the call with this id.
    Tool {
        call_id: String,
        content: String,
    },
}

impl Message {
    pub fn user(text: impl Into<String>) -> Self {
        Message::User(text.into())
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Message::Assistant(AssistantMessage {
            text: text.into(),
            ..AssistantMessage::default()
        })
    }
    /// The text of the message (tool results included), for size estimates.
    pub fn text_len(&self) -> usize {
        match self {
            Message::User(t) => t.len(),
            Message::UserImages { text, .. } => text.len(),
            Message::Assistant(a) => {
                a.text.len()
                    + a.tool_calls
                        .iter()
                        .map(|c| c.name.len() + c.arguments.len())
                        .sum::<usize>()
            }
            Message::Tool { content, .. } => content.len(),
        }
    }
}

/// An image in a user turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    /// `image/png`, `image/jpeg`.
    pub media_type: String,
    /// The bytes, base64-encoded.
    pub base64: String,
}

impl Image {
    /// A `data:` URL, as OpenAI-compatible servers take images.
    pub fn data_url(&self) -> String {
        format!("data:{};base64,{}", self.media_type, self.base64)
    }
}

/// What the model said in one turn.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssistantMessage {
    pub text: String,
    pub tool_calls: Vec<ToolCallRequest>,
    /// The model's reasoning, when the provider returns it. Sent back with
    /// tool calls, which some providers require.
    pub reasoning: Option<String>,
    /// Anthropic's thinking blocks, kept as sent so they can be returned.
    pub thinking_blocks: Vec<Value>,
}

/// A tool call the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    /// The arguments as the model wrote them (JSON text, maybe invalid).
    pub arguments: String,
}

/// A tool the model may call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    /// JSON Schema of the arguments.
    pub parameters: Value,
}

/// One request to the model.
#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    /// Most output tokens. Anthropic requires one; it defaults to
    /// [`DEFAULT_ANTHROPIC_MAX_TOKENS`] there and is left out elsewhere.
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub stop: Vec<String>,
    /// `reasoning_effort` (OpenAI-compatible) or a thinking budget level.
    pub reasoning_effort: Option<String>,
}

/// Anthropic's `max_tokens` when the request sets none.
pub const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 16_384;

/// Why the model stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    ToolCalls,
    /// It hit the output limit; the answer is cut off.
    Length,
    ContentFilter,
    Other(String),
}

impl FinishReason {
    pub fn from_openai(s: &str) -> Self {
        match s {
            "stop" => Self::Stop,
            "tool_calls" | "function_call" => Self::ToolCalls,
            "length" => Self::Length,
            "content_filter" => Self::ContentFilter,
            other => Self::Other(other.to_string()),
        }
    }
    /// The reason as OpenAI names it (`stop`, `tool_calls`, `length`,
    /// `content_filter`), or the provider's own word.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Stop => "stop",
            Self::ToolCalls => "tool_calls",
            Self::Length => "length",
            Self::ContentFilter => "content_filter",
            Self::Other(s) => s,
        }
    }

    pub fn from_anthropic(s: &str) -> Self {
        match s {
            "end_turn" | "stop_sequence" => Self::Stop,
            "tool_use" => Self::ToolCalls,
            "max_tokens" => Self::Length,
            "refusal" => Self::ContentFilter,
            other => Self::Other(other.to_string()),
        }
    }
}

/// Tokens and cost of one call, or summed over many.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct TokenUsage {
    /// Prompt tokens, cached ones included (OpenAI's `prompt_tokens`).
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Prompt tokens read from the provider's cache (part of `input_tokens`
    /// for OpenAI-compatible servers; separate for Anthropic).
    pub cache_read_tokens: u64,
    pub cache_write_tokens: u64,
    /// Reasoning tokens (part of `output_tokens`).
    pub reasoning_tokens: u64,
    /// Cost in US dollars, when the provider reports it.
    pub cost_usd: Option<f64>,
}

impl TokenUsage {
    pub fn add(&mut self, other: &TokenUsage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_read_tokens += other.cache_read_tokens;
        self.cache_write_tokens += other.cache_write_tokens;
        self.reasoning_tokens += other.reasoning_tokens;
        self.cost_usd = match (self.cost_usd, other.cost_usd) {
            (None, None) => None,
            (a, b) => Some(a.unwrap_or(0.0) + b.unwrap_or(0.0)),
        };
    }

    /// Input plus output tokens: what `--max-tokens` counts.
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }
}

/// The model's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    pub message: AssistantMessage,
    pub finish: FinishReason,
    pub usage: TokenUsage,
    /// The model that answered, as the provider names it.
    pub model: Option<String>,
}
