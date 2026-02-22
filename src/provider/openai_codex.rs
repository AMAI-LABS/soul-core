/// OpenAI Codex provider — ChatGPT/Codex OAuth subscription auth.
///
/// Wire protocol is identical to OpenAI (`/v1/chat/completions`, SSE streaming),
/// but authenticates via a short-lived OAuth access token instead of an API key.
///
/// # Auth flow (mirrors OpenClaw's openai-codex scheme)
///
/// 1. User completes ChatGPT OAuth → receives `access_token` + `refresh_token`
/// 2. Build an `AuthProfile` with `AuthProfile::new_oauth(ProviderKind::OpenAICodex, ...)`
/// 3. On each request, check `auth_mode.is_token_expired()` and refresh if needed
///    (the caller or a wrapping `BalancedProvider` slot manages refresh)
/// 4. Send `Authorization: Bearer <access_token>` — identical to OpenAI API key auth
///
/// # Token refresh
///
/// Call `OpenAICodexProvider::refresh_token` with the expired `AuthProfile` to obtain
/// a new access token. This hits the OpenAI token endpoint and returns an updated profile.
/// In production, use `AuthProfileStore` (application layer) under a lock to prevent
/// concurrent refresh races.
///
/// # Model strings
///
/// Use the `openai-codex/<model>` format, e.g. `openai-codex/gpt-5.3-codex`.
/// The provider strips the prefix and sends only the model ID to the API.
use reqwest::Client;
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::types::*;

use super::traits::{ProbeResult, Provider};

/// OpenAI Codex refresh endpoint.
const OPENAI_TOKEN_ENDPOINT: &str = "https://auth.openai.com/token";

/// OpenAI Codex client ID (matches the OAuth app used by ChatGPT clients).
const OPENAI_CODEX_CLIENT_ID: &str = "app_EMXIVkSdDFUDqBr0klkJzlXX";

pub struct OpenAICodexProvider {
    client: Client,
    base_url: String,
}

impl OpenAICodexProvider {
    pub fn new() -> Self {
        Self {
            client: super::llm_client(),
            base_url: "https://api.openai.com".into(),
        }
    }

    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self {
            client: super::llm_client(),
            base_url: base_url.into(),
        }
    }

    /// Refresh an expired OAuth access token using the stored refresh token.
    ///
    /// Returns an updated `AuthProfile` with the new access token and expiry.
    /// The caller is responsible for persisting the updated profile.
    ///
    /// # Errors
    /// - `SoulError::Auth` if the refresh token is invalid or revoked
    /// - `SoulError::Provider` on network / server errors
    pub async fn refresh_token(client: &Client, profile: &AuthProfile) -> SoulResult<AuthProfile> {
        let refresh_token = match &profile.auth_mode {
            AuthMode::OAuth { refresh_token, .. } => refresh_token.clone(),
            AuthMode::ApiKey => {
                return Err(SoulError::Auth(
                    "refresh_token called on non-OAuth AuthProfile".into(),
                ))
            }
        };

        let body = json!({
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        });

        let response = client
            .post(OPENAI_TOKEN_ENDPOINT)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| SoulError::Provider(format!("Codex token refresh network error: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            if status.as_u16() == 400 || status.as_u16() == 401 {
                return Err(SoulError::Auth(format!(
                    "Codex OAuth refresh rejected ({status}): {text}"
                )));
            }
            return Err(SoulError::Provider(format!(
                "Codex token refresh error {status}: {text}"
            )));
        }

        let data: serde_json::Value = response
            .json()
            .await
            .map_err(|e| SoulError::Provider(format!("Codex token refresh parse error: {e}")))?;

        let access_token = data["access_token"]
            .as_str()
            .ok_or_else(|| SoulError::Auth("Codex token refresh: missing access_token".into()))?
            .to_string();

        // `expires_in` is seconds from now; convert to absolute Unix ms
        let expires_in_secs = data["expires_in"].as_u64().unwrap_or(3600);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let expires_at_ms = now_ms + expires_in_secs * 1000;

        // Prefer a new refresh_token if provided; fall back to existing one
        let new_refresh_token = data["refresh_token"]
            .as_str()
            .unwrap_or(&refresh_token)
            .to_string();

        let mut updated = profile.clone();
        updated.api_key = access_token;
        updated.auth_mode = AuthMode::OAuth {
            refresh_token: new_refresh_token,
            expires_at_ms,
        };

        tracing::debug!(
            profile_id = %profile.id,
            expires_at_ms,
            "OpenAICodex: access token refreshed"
        );

        Ok(updated)
    }

    fn build_body(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
    ) -> serde_json::Value {
        // Strip any "openai-codex/" prefix from the model ID before sending
        let model_id = model
            .id
            .strip_prefix("openai-codex/")
            .unwrap_or(&model.id);

        let mut api_messages = vec![json!({"role": "system", "content": system})];
        for msg in messages {
            api_messages.push(self.message_to_api(msg));
        }

        let mut body = json!({
            "model": model_id,
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
                if let Some(ContentBlock::ToolResult { tool_call_id, content, .. }) = block {
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

impl Default for OpenAICodexProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl Provider for OpenAICodexProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAICodex
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

        tracing::debug!(
            url = %url,
            model = %model.id,
            messages = messages.len(),
            tools = tools.len(),
            is_oauth = auth.auth_mode.is_oauth(),
            token_expired = auth.auth_mode.is_token_expired(),
            "OpenAICodex: sending request"
        );

        if auth.auth_mode.is_token_expired() {
            return Err(SoulError::Auth(
                "OpenAI Codex access token has expired — call OpenAICodexProvider::refresh_token first".into(),
            ));
        }

        let start = std::time::Instant::now();

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", auth.api_key))
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
                    "OpenAICodex: request failed"
                );
                e
            })?;

        let status = response.status();
        tracing::debug!(
            status = %status,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "OpenAICodex: response received"
        );

        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            tracing::warn!(
                status = %status,
                body_preview = %if body_text.len() > 500 { &body_text[..500] } else { &body_text },
                "OpenAICodex: API error"
            );

            if status.as_u16() == 401 {
                return Err(SoulError::Auth(format!(
                    "OpenAI Codex auth rejected — token may be expired or revoked ({status}): {body_text}"
                )));
            }
            if status.as_u16() == 429 {
                return Err(SoulError::RateLimited {
                    provider: "openai-codex".into(),
                    retry_after_ms: 5000,
                });
            }
            return Err(SoulError::Provider(format!(
                "OpenAI Codex API error {status}: {body_text}"
            )));
        }

        let bytes = response.bytes().await?;
        let text = String::from_utf8_lossy(&bytes);

        let mut content_text = String::new();
        let mut tool_calls: Vec<(String, String, String)> = Vec::new();
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
                                        let idx = tc
                                            .get("index")
                                            .and_then(|v| v.as_u64())
                                            .unwrap_or(0)
                                            as usize;
                                        while tool_calls.len() <= idx {
                                            tool_calls.push((
                                                String::new(),
                                                String::new(),
                                                String::new(),
                                            ));
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

                    if let Some(u) = data.get("usage") {
                        if let Some(inp) =
                            u.get("prompt_tokens").and_then(|v| v.as_u64())
                        {
                            usage.input_tokens = inp as usize;
                        }
                        if let Some(out) =
                            u.get("completion_tokens").and_then(|v| v.as_u64())
                        {
                            usage.output_tokens = out as usize;
                        }
                    }
                }
            }
        }

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
        msg.usage = Some(usage.clone());

        tracing::debug!(
            model = %model.id,
            text_len = msg.text_content().len(),
            tool_calls = msg.tool_calls().len(),
            input_tokens = usage.input_tokens,
            output_tokens = usage.output_tokens,
            elapsed_ms = start.elapsed().as_millis() as u64,
            "OpenAICodex: response parsed"
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
        let total: usize = messages.iter().map(|m| m.estimate_tokens()).sum();
        Ok(total)
    }

    async fn probe(&self, model: &ModelInfo, auth: &AuthProfile) -> SoulResult<ProbeResult> {
        if auth.auth_mode.is_token_expired() {
            return Ok(ProbeResult {
                healthy: false,
                rate_limit_remaining: None,
                rate_limit_utilization: None,
            });
        }

        let model_id = model
            .id
            .strip_prefix("openai-codex/")
            .unwrap_or(&model.id);
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = json!({
            "model": model_id,
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

        let healthy = response.status().is_success()
            || response.status().as_u16() == 400; // 400 = model exists but bad request

        Ok(ProbeResult {
            healthy,
            rate_limit_remaining: None,
            rate_limit_utilization: None,
        })
    }
}

// ─── Model helpers ────────────────────────────────────────────────────────────

/// Pre-configured `ModelInfo` for `gpt-5.3-codex` (OpenAI Codex subscription).
///
/// Use this when building `AgentConfig` with an `openai-codex` auth profile.
pub fn gpt_5_3_codex() -> ModelInfo {
    ModelInfo {
        id: "openai-codex/gpt-5.3-codex".into(),
        provider: ProviderKind::OpenAICodex,
        context_window: 200_000,
        max_output_tokens: 65_536,
        supports_thinking: false,
        supports_tools: true,
        supports_images: true,
        cost_per_input_token: 0.0,  // subscription — no per-token cost
        cost_per_output_token: 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn provider_kind_is_openai_codex() {
        let provider = OpenAICodexProvider::new();
        assert_eq!(provider.kind(), ProviderKind::OpenAICodex);
    }

    #[test]
    fn provider_kind_display() {
        assert_eq!(ProviderKind::OpenAICodex.to_string(), "openai-codex");
    }

    #[test]
    fn custom_base_url() {
        let provider = OpenAICodexProvider::with_base_url("http://localhost:9999");
        assert_eq!(provider.base_url, "http://localhost:9999");
    }

    #[test]
    fn gpt_5_3_codex_model_info() {
        let m = gpt_5_3_codex();
        assert_eq!(m.id, "openai-codex/gpt-5.3-codex");
        assert_eq!(m.provider, ProviderKind::OpenAICodex);
        assert!(m.supports_tools);
        assert_eq!(m.context_window, 200_000);
    }

    #[test]
    fn auth_profile_oauth_new() {
        let profile = AuthProfile::new_oauth(
            ProviderKind::OpenAICodex,
            "access-tok",
            "refresh-tok",
            9_999_999_999_000,
        );
        assert_eq!(profile.provider, ProviderKind::OpenAICodex);
        assert_eq!(profile.api_key, "access-tok");
        assert!(profile.auth_mode.is_oauth());
        assert!(!profile.auth_mode.is_token_expired());
    }

    #[test]
    fn auth_mode_expired_token() {
        let profile = AuthProfile::new_oauth(
            ProviderKind::OpenAICodex,
            "access-tok",
            "refresh-tok",
            1_000, // expired (ms in the past)
        );
        assert!(profile.auth_mode.is_token_expired());
    }

    #[test]
    fn auth_mode_api_key_never_expires() {
        let profile = AuthProfile::new(ProviderKind::OpenAI, "sk-test");
        assert!(!profile.auth_mode.is_token_expired());
        assert!(!profile.auth_mode.is_oauth());
    }

    #[test]
    fn auth_mode_is_api_key_skip_serialization() {
        let profile = AuthProfile::new(ProviderKind::OpenAI, "sk-test");
        let json = serde_json::to_string(&profile).unwrap();
        // auth_mode field should be absent (ApiKey is the default, skip_serializing_if)
        assert!(!json.contains("auth_mode"), "auth_mode should be omitted for ApiKey: {json}");
    }

    #[test]
    fn auth_mode_oauth_serializes() {
        let profile = AuthProfile::new_oauth(
            ProviderKind::OpenAICodex,
            "access",
            "refresh",
            1_700_000_000_000,
        );
        let json = serde_json::to_string(&profile).unwrap();
        assert!(json.contains("oauth"), "auth_mode should be present: {json}");
        assert!(json.contains("refresh_token"), "{json}");
        assert!(json.contains("expires_at_ms"), "{json}");
    }

    #[test]
    fn model_id_prefix_stripped_in_body() {
        let provider = OpenAICodexProvider::new();
        let model = gpt_5_3_codex();
        let body = provider.build_body(&[], "sys", &[], &model);
        // The API receives "gpt-5.3-codex", not "openai-codex/gpt-5.3-codex"
        assert_eq!(body["model"], "gpt-5.3-codex");
    }

    #[test]
    fn model_id_without_prefix_passes_through() {
        let provider = OpenAICodexProvider::new();
        let model = ModelInfo {
            id: "gpt-5.3-codex".into(),
            provider: ProviderKind::OpenAICodex,
            context_window: 200_000,
            max_output_tokens: 65_536,
            supports_thinking: false,
            supports_tools: true,
            supports_images: true,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let body = provider.build_body(&[], "sys", &[], &model);
        assert_eq!(body["model"], "gpt-5.3-codex");
    }

    #[test]
    fn provider_kind_serializes_as_kebab_case() {
        let json = serde_json::to_string(&ProviderKind::OpenAICodex).unwrap();
        assert_eq!(json, r#""openai-codex""#);
    }

    #[test]
    fn provider_kind_deserializes_from_kebab_case() {
        let kind: ProviderKind = serde_json::from_str(r#""openai-codex""#).unwrap();
        assert_eq!(kind, ProviderKind::OpenAICodex);
    }
}
