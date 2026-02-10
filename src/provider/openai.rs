use async_trait::async_trait;
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::types::*;

use super::traits::{ProbeResult, Provider};

pub struct OpenAIProvider {
    client: Client,
    base_url: String,
}

impl OpenAIProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            base_url: "https://api.openai.com".into(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into(),
        }
    }

    fn build_body(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
    ) -> serde_json::Value {
        let mut api_messages = vec![json!({"role": "system", "content": system})];

        for msg in messages {
            api_messages.push(self.message_to_api(msg));
        }

        let mut body = json!({
            "model": model.id,
            "messages": api_messages,
            "stream": true,
        });

        if model.max_output_tokens > 0 {
            body["max_tokens"] = json!(model.max_output_tokens);
        }

        if !tools.is_empty() {
            let api_tools: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        }
                    })
                })
                .collect();
            body["tools"] = json!(api_tools);
        }

        body
    }

    fn message_to_api(&self, msg: &Message) -> serde_json::Value {
        match msg.role {
            Role::Assistant => {
                let mut result = json!({"role": "assistant"});
                let mut content_text = String::new();
                let mut tool_calls: Vec<serde_json::Value> = Vec::new();

                for block in &msg.content {
                    match block {
                        ContentBlock::Text { text } => content_text.push_str(text),
                        ContentBlock::ToolCall { id, name, arguments } => {
                            tool_calls.push(json!({
                                "id": id,
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments.to_string(),
                                }
                            }));
                        }
                        _ => {}
                    }
                }

                if !content_text.is_empty() {
                    result["content"] = json!(content_text);
                }
                if !tool_calls.is_empty() {
                    result["tool_calls"] = json!(tool_calls);
                }
                result
            }
            Role::Tool => {
                let block = msg.content.first();
                if let Some(ContentBlock::ToolResult {
                    tool_call_id,
                    content,
                    ..
                }) = block
                {
                    json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,
                    })
                } else {
                    json!({"role": "user", "content": msg.text_content()})
                }
            }
            Role::User => json!({"role": "user", "content": msg.text_content()}),
            Role::System => json!({"role": "system", "content": msg.text_content()}),
        }
    }
}

impl Default for OpenAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAI
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
        let body = self.build_body(messages, system, tools, model);
        let url = format!("{}/v1/chat/completions", self.base_url);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", auth.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();

            if status.as_u16() == 429 {
                return Err(SoulError::RateLimited {
                    provider: "openai".into(),
                    retry_after_ms: 5000,
                });
            }
            return Err(SoulError::Provider(format!(
                "OpenAI API error {status}: {body}"
            )));
        }

        let bytes = response.bytes().await?;
        let text = String::from_utf8_lossy(&bytes);

        let mut content_text = String::new();
        let mut tool_calls: Vec<(String, String, String)> = Vec::new(); // (id, name, args)
        let mut usage = TokenUsage::new(0, 0);

        for line in text.lines() {
            if let Some(data_str) = line.strip_prefix("data: ") {
                if data_str.trim() == "[DONE]" {
                    break;
                }
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(data_str) {
                    if let Some(choices) = data.get("choices").and_then(|v| v.as_array()) {
                        if let Some(choice) = choices.first() {
                            if let Some(delta) = choice.get("delta") {
                                // Text content
                                if let Some(content) =
                                    delta.get("content").and_then(|v| v.as_str())
                                {
                                    content_text.push_str(content);
                                    let _ = event_tx.send(StreamDelta::TextDelta {
                                        text: content.to_string(),
                                    });
                                }

                                // Tool calls
                                if let Some(tcs) =
                                    delta.get("tool_calls").and_then(|v| v.as_array())
                                {
                                    for tc in tcs {
                                        let idx =
                                            tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0)
                                                as usize;

                                        // Ensure vector has enough entries
                                        while tool_calls.len() <= idx {
                                            tool_calls.push((String::new(), String::new(), String::new()));
                                        }

                                        if let Some(id) =
                                            tc.get("id").and_then(|v| v.as_str())
                                        {
                                            tool_calls[idx].0 = id.to_string();
                                        }
                                        if let Some(func) = tc.get("function") {
                                            if let Some(name) =
                                                func.get("name").and_then(|v| v.as_str())
                                            {
                                                tool_calls[idx].1 = name.to_string();
                                            }
                                            if let Some(args) =
                                                func.get("arguments").and_then(|v| v.as_str())
                                            {
                                                tool_calls[idx].2.push_str(args);
                                                let _ =
                                                    event_tx.send(StreamDelta::ToolCallDelta {
                                                        id: tool_calls[idx].0.clone(),
                                                        name: tool_calls[idx].1.clone(),
                                                        arguments_delta: args.to_string(),
                                                    });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Usage
                    if let Some(u) = data.get("usage") {
                        if let Some(inp) = u.get("prompt_tokens").and_then(|v| v.as_u64()) {
                            usage.input_tokens = inp as usize;
                        }
                        if let Some(out) = u.get("completion_tokens").and_then(|v| v.as_u64()) {
                            usage.output_tokens = out as usize;
                        }
                    }
                }
            }
        }

        // Build content blocks
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        if !content_text.is_empty() {
            content_blocks.push(ContentBlock::text(content_text));
        }
        for (id, name, args_str) in tool_calls {
            if !name.is_empty() {
                let args = serde_json::from_str(&args_str).unwrap_or(json!({}));
                content_blocks.push(ContentBlock::tool_call(id, name, args));
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
        _system: &str,
        _tools: &[ToolDefinition],
        _model: &ModelInfo,
        _auth: &AuthProfile,
    ) -> SoulResult<usize> {
        // OpenAI doesn't have a native token counting endpoint
        // Use estimation: ~4 chars per token
        let total: usize = messages.iter().map(|m| m.estimate_tokens()).sum();
        Ok(total)
    }

    async fn probe(
        &self,
        model: &ModelInfo,
        auth: &AuthProfile,
    ) -> SoulResult<ProbeResult> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = json!({
            "model": model.id,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}],
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", auth.api_key))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let healthy = response.status().is_success();
        Ok(ProbeResult {
            healthy,
            rate_limit_remaining: None,
            rate_limit_utilization: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_kind_is_openai() {
        let provider = OpenAIProvider::new();
        assert_eq!(provider.kind(), ProviderKind::OpenAI);
    }

    #[test]
    fn custom_base_url() {
        let provider = OpenAIProvider::with_base_url("http://localhost:8081");
        assert_eq!(provider.base_url, "http://localhost:8081");
    }

    #[test]
    fn message_to_api_user() {
        let provider = OpenAIProvider::new();
        let msg = Message::user("hello");
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "user");
        assert_eq!(api["content"], "hello");
    }

    #[test]
    fn message_to_api_assistant_with_tool_calls() {
        let provider = OpenAIProvider::new();
        let msg = Message::new(
            Role::Assistant,
            vec![
                ContentBlock::text("I'll check"),
                ContentBlock::tool_call("tc1", "read", json!({"path": "/tmp/a.txt"})),
            ],
        );
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "assistant");
        assert_eq!(api["content"], "I'll check");
        assert!(api["tool_calls"].is_array());
        assert_eq!(api["tool_calls"][0]["function"]["name"], "read");
    }

    #[test]
    fn message_to_api_tool_result() {
        let provider = OpenAIProvider::new();
        let msg = Message::tool_result("tc1", "file contents here", false);
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "tool");
        assert_eq!(api["tool_call_id"], "tc1");
        assert_eq!(api["content"], "file contents here");
    }

    #[test]
    fn builds_body_with_tools() {
        let provider = OpenAIProvider::new();
        let model = ModelInfo {
            id: "gpt-4o".into(),
            provider: ProviderKind::OpenAI,
            context_window: 128_000,
            max_output_tokens: 4096,
            supports_thinking: false,
            supports_tools: true,
            supports_images: true,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let messages = vec![Message::user("test")];
        let tools = vec![ToolDefinition {
            name: "bash".into(),
            description: "Run command".into(),
            input_schema: json!({"type": "object"}),
        }];

        let body = provider.build_body(&messages, "system", &tools, &model);
        assert_eq!(body["model"], "gpt-4o");
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "bash");
        // System message is first
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "system");
    }

    #[test]
    fn builds_body_without_tools() {
        let provider = OpenAIProvider::new();
        let model = ModelInfo {
            id: "gpt-4o-mini".into(),
            provider: ProviderKind::OpenAI,
            context_window: 128_000,
            max_output_tokens: 0,
            supports_thinking: false,
            supports_tools: false,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let messages = vec![Message::user("hi")];
        let body = provider.build_body(&messages, "sys", &[], &model);
        assert!(body.get("tools").is_none());
        assert!(body.get("max_tokens").is_none());
    }
}
