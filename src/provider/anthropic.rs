use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::types::*;

use super::traits::{ProbeResult, Provider};

pub struct AnthropicProvider {
    client: Client,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.anthropic.com".into(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    fn build_messages_body(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
    ) -> serde_json::Value {
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| self.message_to_api(m))
            .collect();

        let mut body = json!({
            "model": model.id,
            "max_tokens": model.max_output_tokens.max(4096),
            "system": system,
            "messages": api_messages,
            "stream": true,
        });

        if !tools.is_empty() {
            let api_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = json!(api_tools);
        }

        body
    }

    fn message_to_api(&self, msg: &Message) -> serde_json::Value {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "user", // Anthropic wraps tool results in user messages
            Role::System => "user",
        };

        let content: Vec<serde_json::Value> = msg
            .content
            .iter()
            .map(|block| match block {
                ContentBlock::Text { text } => json!({"type": "text", "text": text}),
                ContentBlock::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    json!({
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": arguments,
                    })
                }
                ContentBlock::ToolResult {
                    tool_call_id,
                    content,
                    is_error,
                } => {
                    json!({
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                        "is_error": is_error,
                    })
                }
                ContentBlock::Thinking { text } => {
                    json!({"type": "thinking", "thinking": text})
                }
                ContentBlock::Image { media_type, data } => {
                    json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    })
                }
            })
            .collect();

        json!({
            "role": role,
            "content": content,
        })
    }

    fn parse_sse_event(&self, event_type: &str, data: &serde_json::Value) -> Option<StreamDelta> {
        match event_type {
            "content_block_delta" => {
                let delta = data.get("delta")?;
                let delta_type = delta.get("type")?.as_str()?;
                match delta_type {
                    "text_delta" => {
                        let text = delta.get("text")?.as_str()?.to_string();
                        Some(StreamDelta::TextDelta { text })
                    }
                    "thinking_delta" => {
                        let text = delta.get("thinking")?.as_str()?.to_string();
                        Some(StreamDelta::ThinkingDelta { text })
                    }
                    "input_json_delta" => {
                        let index = data.get("index")?.as_u64()? as usize;
                        let partial = delta.get("partial_json")?.as_str()?.to_string();
                        Some(StreamDelta::ToolCallDelta {
                            id: format!("block_{index}"),
                            name: String::new(),
                            arguments_delta: partial,
                        })
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

impl Default for AnthropicProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
        auth: &AuthProfile,
        event_tx: mpsc::UnboundedSender<StreamDelta>,
    ) -> SoulResult<Message> {
        let body = self.build_messages_body(messages, system, tools, model);
        let url = format!("{}/v1/messages", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &auth.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();

            if status.as_u16() == 429 {
                return Err(SoulError::RateLimited {
                    provider: "anthropic".into(),
                    retry_after_ms: 5000,
                });
            }
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(SoulError::Auth(format!("Anthropic auth failed: {body}")));
            }
            return Err(SoulError::Provider(format!(
                "Anthropic API error {status}: {body}"
            )));
        }

        // Parse SSE stream
        let bytes = response.bytes().await?;
        let text = String::from_utf8_lossy(&bytes);

        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut usage = TokenUsage::new(0, 0);

        for line in text.lines() {
            if let Some(data_str) = line.strip_prefix("data: ") {
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(data_str) {
                    let event_type = data.get("type").and_then(|v| v.as_str()).unwrap_or("");

                    match event_type {
                        "content_block_start" => {
                            let block = data.get("content_block").unwrap_or(&json!(null));
                            let block_type =
                                block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            match block_type {
                                "text" => {
                                    content_blocks.push(ContentBlock::Text {
                                        text: String::new(),
                                    });
                                }
                                "tool_use" => {
                                    let id = block
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let name = block
                                        .get("name")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    content_blocks.push(ContentBlock::ToolCall {
                                        id,
                                        name,
                                        arguments: json!({}),
                                    });
                                }
                                "thinking" => {
                                    content_blocks.push(ContentBlock::Thinking {
                                        text: String::new(),
                                    });
                                }
                                _ => {}
                            }
                        }
                        "content_block_delta" => {
                            if let Some(delta) = self.parse_sse_event(event_type, &data) {
                                let _ = event_tx.send(delta.clone());

                                if let Some(last) = content_blocks.last_mut() {
                                    match (&delta, last) {
                                        (
                                            StreamDelta::TextDelta { text },
                                            ContentBlock::Text { text: ref mut t },
                                        ) => {
                                            t.push_str(text);
                                        }
                                        (
                                            StreamDelta::ThinkingDelta { text },
                                            ContentBlock::Thinking { text: ref mut t },
                                        ) => {
                                            t.push_str(text);
                                        }
                                        (
                                            StreamDelta::ToolCallDelta {
                                                arguments_delta, ..
                                            },
                                            ContentBlock::ToolCall {
                                                arguments: ref mut args,
                                                ..
                                            },
                                        ) => {
                                            // Accumulate JSON fragments
                                            let current = args.as_str().unwrap_or("").to_string();
                                            let new = format!("{current}{arguments_delta}");
                                            *args = serde_json::Value::String(new);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        "message_delta" => {
                            if let Some(u) = data.get("usage") {
                                if let Some(out) = u.get("output_tokens").and_then(|v| v.as_u64()) {
                                    usage.output_tokens = out as usize;
                                }
                            }
                        }
                        "message_start" => {
                            if let Some(msg) = data.get("message") {
                                if let Some(u) = msg.get("usage") {
                                    if let Some(inp) =
                                        u.get("input_tokens").and_then(|v| v.as_u64())
                                    {
                                        usage.input_tokens = inp as usize;
                                    }
                                    if let Some(cr) =
                                        u.get("cache_read_input_tokens").and_then(|v| v.as_u64())
                                    {
                                        usage.cache_read_tokens = cr as usize;
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        // Fix accumulated tool call JSON strings → proper JSON values
        for block in &mut content_blocks {
            if let ContentBlock::ToolCall { arguments, .. } = block {
                if let Some(s) = arguments.as_str() {
                    if let Ok(parsed) = serde_json::from_str(s) {
                        *arguments = parsed;
                    }
                }
            }
        }

        let mut msg = Message::new(Role::Assistant, content_blocks);
        msg.model = Some(model.id.clone());
        msg.usage = Some(usage);

        Ok(msg)
    }

    async fn count_tokens(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
        auth: &AuthProfile,
    ) -> SoulResult<usize> {
        let mut body = self.build_messages_body(messages, system, tools, model);
        body.as_object_mut().unwrap().remove("stream");

        let url = format!("{}/v1/messages/count_tokens", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &auth.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(SoulError::Provider(format!("Token count failed: {body}")));
        }

        let data: serde_json::Value = response.json().await?;
        let count = data
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        Ok(count)
    }

    async fn probe(&self, model: &ModelInfo, auth: &AuthProfile) -> SoulResult<ProbeResult> {
        let url = format!("{}/v1/messages", self.base_url);
        let body = json!({
            "model": model.id,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "quota"}],
        });

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &auth.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let utilization = response
            .headers()
            .get("anthropic-ratelimit-unified-5h-utilization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok());

        let healthy = response.status().is_success();

        Ok(ProbeResult {
            healthy,
            rate_limit_remaining: utilization.map(|u| 1.0 - u),
            rate_limit_utilization: utilization,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_kind_is_anthropic() {
        let provider = AnthropicProvider::new();
        assert_eq!(provider.kind(), ProviderKind::Anthropic);
    }

    #[test]
    fn custom_base_url() {
        let provider = AnthropicProvider::with_base_url("http://localhost:8081");
        assert_eq!(provider.base_url, "http://localhost:8081");
    }

    #[test]
    fn message_to_api_user() {
        let provider = AnthropicProvider::new();
        let msg = Message::user("hello");
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "user");
        assert_eq!(api["content"][0]["type"], "text");
        assert_eq!(api["content"][0]["text"], "hello");
    }

    #[test]
    fn message_to_api_assistant_with_tool_call() {
        let provider = AnthropicProvider::new();
        let msg = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("Let me check"),
                ContentBlock::tool_call("tc1", "read", json!({"path": "/tmp/test.txt"})),
            ],
        );
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "assistant");
        assert_eq!(api["content"][0]["type"], "text");
        assert_eq!(api["content"][1]["type"], "tool_use");
        assert_eq!(api["content"][1]["name"], "read");
    }

    #[test]
    fn message_to_api_tool_result() {
        let provider = AnthropicProvider::new();
        let msg = Message::tool_result("tc1", "file contents", false);
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "user"); // Anthropic uses user role for tool results
        assert_eq!(api["content"][0]["type"], "tool_result");
        assert_eq!(api["content"][0]["tool_use_id"], "tc1");
    }

    #[test]
    fn builds_messages_body_with_tools() {
        let provider = AnthropicProvider::new();
        let model = ModelInfo {
            id: "claude-sonnet-4-5-20250929".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let messages = vec![Message::user("hello")];
        let tools = vec![ToolDefinition {
            name: "read".into(),
            description: "Read a file".into(),
            input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        }];

        let body = provider.build_messages_body(&messages, "system prompt", &tools, &model);
        assert_eq!(body["model"], "claude-sonnet-4-5-20250929");
        assert_eq!(body["system"], "system prompt");
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"][0]["name"], "read");
    }

    #[test]
    fn builds_messages_body_without_tools() {
        let provider = AnthropicProvider::new();
        let model = ModelInfo {
            id: "claude-haiku-4-5-20251001".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: false,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let messages = vec![Message::user("hello")];

        let body = provider.build_messages_body(&messages, "system", &[], &model);
        assert!(body.get("tools").is_none());
    }

    #[test]
    fn parse_sse_text_delta() {
        let provider = AnthropicProvider::new();
        let data = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}
        });
        let delta = provider.parse_sse_event("content_block_delta", &data);
        assert!(matches!(delta, Some(StreamDelta::TextDelta { text }) if text == "Hello"));
    }

    #[test]
    fn parse_sse_thinking_delta() {
        let provider = AnthropicProvider::new();
        let data = json!({
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Let me think..."}
        });
        let delta = provider.parse_sse_event("content_block_delta", &data);
        assert!(
            matches!(delta, Some(StreamDelta::ThinkingDelta { text }) if text == "Let me think...")
        );
    }

    #[test]
    fn parse_sse_unknown_event() {
        let provider = AnthropicProvider::new();
        let data = json!({"type": "ping"});
        let delta = provider.parse_sse_event("ping", &data);
        assert!(delta.is_none());
    }
}
