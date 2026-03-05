use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::types::*;

use super::traits::{ProbeResult, Provider};

pub struct OllamaProvider {
    client: Client,
    base_url: String,
}

impl OllamaProvider {
    pub fn new() -> Self {
        Self {
            client: super::llm_client(),
            base_url: "http://localhost:11434".into(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: super::llm_client(),
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
            body["options"] = json!({"num_predict": model.max_output_tokens});
        }

        // Enable extended thinking for qwen models that support it (qwen3 base).
        // think=true must be at the TOP LEVEL of the request body (not inside options).
        // Note: qwen3-coder variants do NOT support thinking — ollama returns an error.
        // Only enable for non-coder qwen3 models.
        let model_id_lower = model.id.to_lowercase();
        if model_id_lower.contains("qwen3") && !model_id_lower.contains("coder") {
            body["think"] = json!(true);
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
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            tool_calls.push(json!({
                                "id": id,
                                "function": {
                                    "name": name,
                                    "arguments": arguments,
                                }
                            }));
                        }
                        _ => {}
                    }
                }

                result["content"] = json!(content_text);
                if !tool_calls.is_empty() {
                    result["tool_calls"] = json!(tool_calls);
                }
                result
            }
            Role::Tool => {
                let block = msg.content.first();
                if let Some(ContentBlock::ToolResult {
                    content,
                    ..
                }) = block
                {
                    json!({
                        "role": "tool",
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

impl Default for OllamaProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl Provider for OllamaProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Ollama
    }

    async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
        _auth: &AuthProfile,
        event_tx: mpsc::UnboundedSender<StreamDelta>,
    ) -> SoulResult<Message> {
        let body = self.build_body(messages, system, tools, model);
        let url = format!("{}/api/chat", self.base_url);

        tracing::debug!(
            url = %url,
            model = %model.id,
            messages = messages.len(),
            tools = tools.len(),
            "Ollama: sending request"
        );

        let start = std::time::Instant::now();

        let response = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                tracing::error!(
                    url = %url,
                    model = %model.id,
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    error = %e,
                    "Ollama: request failed"
                );
                e
            })?;

        let status = response.status();
        tracing::debug!(
            status = %status,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Ollama: response received"
        );

        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            tracing::warn!(
                status = %status,
                body_preview = %if body.len() > 500 { &body[..500] } else { &body },
                "Ollama: API error"
            );

            if status.as_u16() == 429 {
                return Err(SoulError::RateLimited {
                    provider: "ollama".into(),
                    retry_after_ms: 5000,
                });
            }
            return Err(SoulError::Provider(format!(
                "Ollama API error {status}: {body}"
            )));
        }

        // Native /api/chat returns NDJSON (one JSON object per line)
        let bytes = response.bytes().await?;
        let text = String::from_utf8_lossy(&bytes);

        tracing::debug!(
            body_len = bytes.len(),
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Ollama: response body received"
        );

        let mut content_text = String::new();
        let mut tool_calls: Vec<(String, String, serde_json::Value)> = Vec::new(); // (id, name, args)
        let mut usage = TokenUsage::new(0, 0);

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if let Ok(data) = serde_json::from_str::<serde_json::Value>(line) {
                // Extract content from message
                if let Some(message) = data.get("message") {
                    // `thinking` field — model's internal CoT, keep separate, don't emit to stream
                    // (ollama puts extended thinking here when think=true)

                    // Text content — strip any inline <think>...</think> blocks
                    if let Some(content) = message.get("content").and_then(|v| v.as_str()) {
                        if !content.is_empty() {
                            let visible = strip_think_tags(content);
                            if !visible.is_empty() {
                                content_text.push_str(&visible);
                                let _ = event_tx.send(StreamDelta::TextDelta {
                                    text: visible,
                                });
                            }
                        }
                    }

                    // Tool calls — native API sends them as parsed JSON
                    if let Some(tcs) = message.get("tool_calls").and_then(|v| v.as_array()) {
                        for tc in tcs {
                            let id = tc
                                .get("id")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .to_string();

                            if let Some(func) = tc.get("function") {
                                let name = func
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                // Native API: arguments is already parsed JSON (not a string)
                                let args = func
                                    .get("arguments")
                                    .cloned()
                                    .unwrap_or(json!({}));

                                if !name.is_empty() {
                                    let _ = event_tx.send(StreamDelta::ToolCallDelta {
                                        id: id.clone(),
                                        name: name.clone(),
                                        arguments_delta: args.to_string(),
                                    });
                                    tool_calls.push((id, name, args));
                                }
                            }
                        }
                    }
                }

                // Usage from the final chunk (done=true)
                if data.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                    if let Some(prompt_tokens) =
                        data.get("prompt_eval_count").and_then(|v| v.as_u64())
                    {
                        usage.input_tokens = prompt_tokens as usize;
                    }
                    if let Some(completion_tokens) =
                        data.get("eval_count").and_then(|v| v.as_u64())
                    {
                        usage.output_tokens = completion_tokens as usize;
                    }
                }
            }
        }

        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        if !content_text.is_empty() {
            content_blocks.push(ContentBlock::text(content_text));
        }
        for (id, name, args) in tool_calls {
            content_blocks.push(ContentBlock::tool_call(id, name, args));
        }

        let mut msg = Message::new(Role::Assistant, content_blocks);
        msg.model = Some(model.id.clone());
        msg.usage = Some(usage.clone());

        tracing::debug!(
            model = %model.id,
            text_len = msg.text_content().len(),
            tool_calls = msg.tool_calls().len(),
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "Ollama: response parsed"
        );

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
        // Ollama doesn't have a native token counting endpoint
        // Use estimation: ~4 chars per token
        let total: usize = messages.iter().map(|m| m.estimate_tokens()).sum();
        Ok(total)
    }

    async fn probe(&self, model: &ModelInfo, _auth: &AuthProfile) -> SoulResult<ProbeResult> {
        // Ollama: check if the model is available via /api/tags
        let url = format!("{}/api/tags", self.base_url);

        let response = self.client.get(&url).send().await;

        match response {
            Ok(resp) if resp.status().is_success() => {
                // Check if the specific model is in the list
                let body = resp.text().await.unwrap_or_default();
                let has_model = body.contains(&model.id);
                Ok(ProbeResult {
                    healthy: has_model,
                    rate_limit_remaining: None,
                    rate_limit_utilization: None,
                })
            }
            Ok(_) => Ok(ProbeResult {
                healthy: false,
                rate_limit_remaining: None,
                rate_limit_utilization: None,
            }),
            Err(_) => Ok(ProbeResult {
                healthy: false,
                rate_limit_remaining: None,
                rate_limit_utilization: None,
            }),
        }
    }
}

/// Strip `<think>...</think>` blocks from model output.
/// Some models (qwen3 base) emit thinking inline in the content field.
/// We keep the thinking internal — strip it before exposing as assistant text.
fn strip_think_tags(s: &str) -> String {
    let mut result = String::new();
    let mut rest = s;
    loop {
        if let Some(start) = rest.find("<think>") {
            result.push_str(&rest[..start]);
            if let Some(end) = rest[start..].find("</think>") {
                rest = &rest[start + end + "</think>".len()..];
            } else {
                // Unclosed tag — drop the rest (still thinking)
                break;
            }
        } else {
            result.push_str(rest);
            break;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_kind_is_ollama() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.kind(), ProviderKind::Ollama);
    }

    #[test]
    fn default_base_url() {
        let provider = OllamaProvider::new();
        assert_eq!(provider.base_url, "http://localhost:11434");
    }

    #[test]
    fn custom_base_url() {
        let provider = OllamaProvider::with_base_url("http://gpu-server:11434");
        assert_eq!(provider.base_url, "http://gpu-server:11434");
    }

    #[test]
    fn message_to_api_user() {
        let provider = OllamaProvider::new();
        let msg = Message::user("hello");
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "user");
        assert_eq!(api["content"], "hello");
    }

    #[test]
    fn message_to_api_assistant_with_tool_calls() {
        let provider = OllamaProvider::new();
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
        // Native API: arguments should be JSON, not string
        assert_eq!(api["tool_calls"][0]["function"]["arguments"]["path"], "/tmp/a.txt");
    }

    #[test]
    fn message_to_api_tool_result() {
        let provider = OllamaProvider::new();
        let msg = Message::tool_result("tc1", "file contents here", false);
        let api = provider.message_to_api(&msg);
        assert_eq!(api["role"], "tool");
        assert_eq!(api["content"], "file contents here");
    }

    #[test]
    fn builds_body_with_tools() {
        let provider = OllamaProvider::new();
        let model = ModelInfo {
            id: "minimax-m2.5:cloud".into(),
            provider: ProviderKind::Ollama,
            context_window: 128_000,
            max_output_tokens: 4096,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
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
        assert_eq!(body["model"], "minimax-m2.5:cloud");
        assert!(body["tools"].is_array());
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "bash");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "system");
        // Native API uses options.num_predict instead of max_tokens
        assert_eq!(body["options"]["num_predict"], 4096);
    }

    #[test]
    fn qwen3_base_enables_thinking() {
        let provider = OllamaProvider::new();
        let model = ModelInfo {
            id: "qwen3:30b".into(),
            provider: ProviderKind::Ollama,
            context_window: 32768,
            max_output_tokens: 2048,
            supports_thinking: true,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let messages = vec![Message::user("hello")];
        let body = provider.build_body(&messages, "sys", &[], &model);
        // qwen3 base supports thinking — think=true at top level
        assert_eq!(body["think"], true, "qwen3 base must have think=true at top level");
    }

    #[test]
    fn qwen3_coder_no_thinking() {
        let provider = OllamaProvider::new();
        let model = ModelInfo {
            id: "qwen3-coder:30b".into(),
            provider: ProviderKind::Ollama,
            context_window: 32768,
            max_output_tokens: 2048,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let messages = vec![Message::user("hello")];
        let body = provider.build_body(&messages, "sys", &[], &model);
        // qwen3-coder does not support thinking
        assert!(body.get("think").is_none(), "qwen3-coder must not set think flag");
    }

    #[test]
    fn strip_think_tags_basic() {
        assert_eq!(strip_think_tags("<think>reasoning here</think>answer"), "answer");
        assert_eq!(strip_think_tags("prefix<think>thinking</think>suffix"), "prefixsuffix");
        assert_eq!(strip_think_tags("no think tags"), "no think tags");
        assert_eq!(strip_think_tags("<think>unclosed"), "");
        // Streaming chunks must preserve whitespace
        assert_eq!(strip_think_tags(" sorry"), " sorry");
        assert_eq!(strip_think_tags("I"), "I");
    }

    #[test]
    fn builds_body_without_tools() {
        let provider = OllamaProvider::new();
        let model = ModelInfo {
            id: "llama3.2".into(),
            provider: ProviderKind::Ollama,
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
        assert!(body.get("options").is_none());
    }
}
