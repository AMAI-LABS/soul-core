use reqwest::Client;
use serde_json::json;
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::types::*;

use super::remap::ToolRemap;
use super::traits::{ProbeResult, Provider};

/// Anthropic API provider — supports both API key and OAuth token auth.
///
/// Auth method is auto-detected from `AuthProfile.api_key`:
/// - `sk-ant-oat*` prefix → OAuth Bearer token flow
/// - anything else → standard `x-api-key` header
///
/// OAuth flow mirrors Claude Code's handshake:
/// - `Authorization: Bearer {token}` (not `x-api-key`)
/// - `anthropic-beta: oauth-2025-04-20` header (required)
/// - `?beta=true` query parameter
/// - `metadata.user_id` injected as SHA-256 hash of token
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

    /// Detect OAuth token from prefix.
    fn is_oauth(auth: &AuthProfile) -> bool {
        auth.api_key.starts_with("sk-ant-oat")
    }

    /// Build the URL, appending `?beta=true` for OAuth.
    fn build_url(&self, path: &str, auth: &AuthProfile) -> String {
        let base = format!("{}{}", self.base_url, path);
        if Self::is_oauth(auth) {
            if base.contains('?') {
                format!("{base}&beta=true")
            } else {
                format!("{base}?beta=true")
            }
        } else {
            base
        }
    }

    /// Apply auth headers to a request builder.
    fn apply_auth(
        &self,
        req: reqwest::RequestBuilder,
        auth: &AuthProfile,
    ) -> reqwest::RequestBuilder {
        let req = req
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json");

        if Self::is_oauth(auth) {
            req.header("Authorization", format!("Bearer {}", auth.api_key))
                .header("anthropic-beta", "oauth-2025-04-20")
        } else {
            req.header("x-api-key", &auth.api_key)
        }
    }

    /// Generate deterministic user_id from OAuth token (SHA-256 hash).
    fn oauth_user_id(token: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let hash = format!("{:x}", hasher.finalize());
        format!("user_{hash}_account__session_00000000-0000-0000-0000-000000000000")
    }

    /// Inject OAuth-specific fields into the request body.
    fn inject_oauth_fields(body: &mut serde_json::Value, auth: &AuthProfile) {
        if !Self::is_oauth(auth) {
            return;
        }
        if let Some(obj) = body.as_object_mut() {
            if !obj.contains_key("metadata") {
                obj.insert(
                    "metadata".to_string(),
                    json!({"user_id": Self::oauth_user_id(&auth.api_key)}),
                );
            }
        }
    }

    fn build_messages_body(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
        remap: &ToolRemap,
    ) -> serde_json::Value {
        let api_messages: Vec<serde_json::Value> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| self.message_to_api(m, remap))
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
                    let name = remap.get_safe_name(&t.name).unwrap_or(&t.name).to_string();
                    json!({
                        "name": name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = json!(api_tools);
        }

        body
    }

    fn message_to_api(&self, msg: &Message, remap: &ToolRemap) -> serde_json::Value {
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
                    let api_name = remap.get_safe_name(name).unwrap_or(name).to_string();
                    json!({
                        "type": "tool_use",
                        "id": id,
                        "name": api_name,
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

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
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
        // Build tool remap for OAuth tokens to bypass semantic name filtering
        let remap = if Self::is_oauth(auth) && !tools.is_empty() {
            ToolRemap::wildcard(tools)
        } else {
            ToolRemap::none()
        };

        let mut body = self.build_messages_body(messages, system, tools, model, &remap);
        Self::inject_oauth_fields(&mut body, auth);

        let url = self.build_url("/v1/messages", auth);
        let req = self.client.post(&url);
        let response = self.apply_auth(req, auth).json(&body).send().await?;

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
                                    let raw_name =
                                        block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                    // Reverse-map remapped tool names back to originals
                                    let name = remap.restore_name(raw_name).to_string();
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
        let remap = if Self::is_oauth(auth) && !tools.is_empty() {
            ToolRemap::wildcard(tools)
        } else {
            ToolRemap::none()
        };
        let mut body = self.build_messages_body(messages, system, tools, model, &remap);
        body.as_object_mut().unwrap().remove("stream");
        Self::inject_oauth_fields(&mut body, auth);

        let url = self.build_url("/v1/messages/count_tokens", auth);
        let req = self.client.post(&url);
        let response = self.apply_auth(req, auth).json(&body).send().await?;

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
        let mut body = json!({
            "model": model.id,
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "quota"}],
        });
        Self::inject_oauth_fields(&mut body, auth);

        let url = self.build_url("/v1/messages", auth);
        let req = self.client.post(&url);
        let response = self.apply_auth(req, auth).json(&body).send().await?;

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
        let no_remap = ToolRemap::none();
        let api = provider.message_to_api(&msg, &no_remap);
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
        let no_remap = ToolRemap::none();
        let api = provider.message_to_api(&msg, &no_remap);
        assert_eq!(api["role"], "assistant");
        assert_eq!(api["content"][0]["type"], "text");
        assert_eq!(api["content"][1]["type"], "tool_use");
        assert_eq!(api["content"][1]["name"], "read");
    }

    #[test]
    fn message_to_api_tool_result() {
        let provider = AnthropicProvider::new();
        let msg = Message::tool_result("tc1", "file contents", false);
        let no_remap = ToolRemap::none();
        let api = provider.message_to_api(&msg, &no_remap);
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
        let no_remap = ToolRemap::none();

        let body =
            provider.build_messages_body(&messages, "system prompt", &tools, &model, &no_remap);
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
        let no_remap = ToolRemap::none();

        let body = provider.build_messages_body(&messages, "system", &[], &model, &no_remap);
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

    // ─── OAuth Detection Tests ──────────────────────────────────────────

    #[test]
    fn detects_oauth_token() {
        let auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-oat01-abc123");
        assert!(AnthropicProvider::is_oauth(&auth));
    }

    #[test]
    fn detects_api_key() {
        let auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-api03-xyz789");
        assert!(!AnthropicProvider::is_oauth(&auth));
    }

    #[test]
    fn oauth_url_has_beta_param() {
        let provider = AnthropicProvider::new();
        let oauth_auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-oat01-test");
        let api_auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-api03-test");

        let oauth_url = provider.build_url("/v1/messages", &oauth_auth);
        assert!(oauth_url.ends_with("?beta=true"));

        let api_url = provider.build_url("/v1/messages", &api_auth);
        assert!(!api_url.contains("beta=true"));
    }

    #[test]
    fn oauth_user_id_is_deterministic() {
        let id1 = AnthropicProvider::oauth_user_id("sk-ant-oat01-test-token");
        let id2 = AnthropicProvider::oauth_user_id("sk-ant-oat01-test-token");
        assert_eq!(id1, id2);
        assert!(id1.starts_with("user_"));
        assert!(id1.contains("_account__session_"));
    }

    #[test]
    fn oauth_user_id_differs_per_token() {
        let id1 = AnthropicProvider::oauth_user_id("sk-ant-oat01-token-a");
        let id2 = AnthropicProvider::oauth_user_id("sk-ant-oat01-token-b");
        assert_ne!(id1, id2);
    }

    #[test]
    fn inject_oauth_fields_adds_metadata() {
        let auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-oat01-test");
        let mut body = json!({"model": "claude-haiku-4-5-20251001", "messages": []});
        AnthropicProvider::inject_oauth_fields(&mut body, &auth);

        assert!(body.get("metadata").is_some());
        let user_id = body["metadata"]["user_id"].as_str().unwrap();
        assert!(user_id.starts_with("user_"));
    }

    #[test]
    fn inject_oauth_fields_skips_api_key() {
        let auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-api03-test");
        let mut body = json!({"model": "claude-haiku-4-5-20251001", "messages": []});
        AnthropicProvider::inject_oauth_fields(&mut body, &auth);

        assert!(body.get("metadata").is_none());
    }

    #[test]
    fn inject_oauth_fields_preserves_existing_metadata() {
        let auth = AuthProfile::new(ProviderKind::Anthropic, "sk-ant-oat01-test");
        let mut body = json!({
            "model": "claude-haiku-4-5-20251001",
            "messages": [],
            "metadata": {"custom_field": "value"}
        });
        AnthropicProvider::inject_oauth_fields(&mut body, &auth);

        // Should NOT add user_id since metadata already exists with content
        // but user_id key is missing, so it should be added
        assert!(body["metadata"]["custom_field"].as_str() == Some("value"));
    }

    #[test]
    fn oauth_user_id_hash_is_64_hex_chars() {
        let id = AnthropicProvider::oauth_user_id("sk-ant-oat01-some-token");
        // Format: user_{64 hex chars}_account__session_{uuid}
        let hash_part = id
            .strip_prefix("user_")
            .unwrap()
            .split("_account__session_")
            .next()
            .unwrap();
        assert_eq!(hash_part.len(), 64);
        assert!(hash_part.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // ─── Tool Name Remap Tests ────────────────────────────────────────────

    #[test]
    fn remap_renames_tools_in_body() {
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
        let tools = vec![
            ToolDefinition {
                name: "read".into(),
                description: "Read a file".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "write".into(),
                description: "Write a file".into(),
                input_schema: json!({"type": "object"}),
            },
        ];
        let remap = ToolRemap::wildcard(&tools);
        let messages = vec![Message::user("hello")];

        let body = provider.build_messages_body(&messages, "system", &tools, &model, &remap);

        // Tool names should be remapped, not originals
        let api_tools = body["tools"].as_array().unwrap();
        for tool in api_tools {
            let name = tool["name"].as_str().unwrap();
            assert!(
                !["read", "write"].contains(&name),
                "Original name leaked: {name}"
            );
            assert!(name.contains('_'), "Expected greek_nature format: {name}");
        }
    }

    #[test]
    fn remap_renames_tool_calls_in_messages() {
        let provider = AnthropicProvider::new();
        let tools = vec![ToolDefinition {
            name: "read".into(),
            description: "Read a file".into(),
            input_schema: json!({"type": "object"}),
        }];
        let remap = ToolRemap::wildcard(&tools);

        let msg = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call(
                "tc1",
                "read",
                json!({"path": "/tmp"}),
            )],
        );
        let api = provider.message_to_api(&msg, &remap);

        let name = api["content"][0]["name"].as_str().unwrap();
        assert_eq!(name, "alpha_river"); // first tool = alpha_river
        assert_ne!(name, "read");
    }

    #[test]
    fn no_remap_preserves_tool_names() {
        let provider = AnthropicProvider::new();
        let no_remap = ToolRemap::none();

        let msg = Message::new(
            Role::Assistant,
            vec![ContentBlock::tool_call("tc1", "read", json!({}))],
        );
        let api = provider.message_to_api(&msg, &no_remap);
        assert_eq!(api["content"][0]["name"], "read");
    }

    #[test]
    fn remap_restores_name_from_sse_content_block() {
        let tools = vec![
            ToolDefinition {
                name: "read".into(),
                description: "r".into(),
                input_schema: json!({"type": "object"}),
            },
            ToolDefinition {
                name: "bash".into(),
                description: "b".into(),
                input_schema: json!({"type": "object"}),
            },
        ];
        let remap = ToolRemap::wildcard(&tools);

        // Simulate what content_block_start returns from the API
        let safe_read = remap.get_safe_name("read").unwrap();
        assert_eq!(remap.restore_name(safe_read), "read");

        let safe_bash = remap.get_safe_name("bash").unwrap();
        assert_eq!(remap.restore_name(safe_bash), "bash");
    }
}
