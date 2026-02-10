use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::types::*;

/// Core provider trait — abstracts LLM API communication
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait Provider: Send + Sync {
    /// Get provider kind
    fn kind(&self) -> ProviderKind;

    /// Stream a completion response, sending deltas through the channel
    async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
        auth: &AuthProfile,
        event_tx: mpsc::UnboundedSender<StreamDelta>,
    ) -> SoulResult<Message>;

    /// Non-streaming completion (convenience, default impl collects stream)
    async fn complete(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
        auth: &AuthProfile,
    ) -> SoulResult<Message> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let result = self
            .stream(messages, system, tools, model, auth, tx)
            .await?;
        // Drain any remaining deltas
        while rx.try_recv().is_ok() {}
        Ok(result)
    }

    /// Count tokens for a set of messages (provider-specific)
    async fn count_tokens(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        model: &ModelInfo,
        auth: &AuthProfile,
    ) -> SoulResult<usize>;

    /// Health check / quota probe (cheap call)
    async fn probe(&self, model: &ModelInfo, auth: &AuthProfile) -> SoulResult<ProbeResult>;
}

/// Result of a provider health probe
#[derive(Debug, Clone)]
pub struct ProbeResult {
    pub healthy: bool,
    pub rate_limit_remaining: Option<f64>,
    pub rate_limit_utilization: Option<f64>,
}

/// Configuration for routing provider requests through a transparent proxy.
///
/// In WASM browser environments, direct API calls face CORS restrictions.
/// A transparent proxy deployment forwards requests to the real API endpoint
/// while being served from the same origin as the WASM app.
///
/// # Example
///
/// ```rust
/// use soul_core::provider::{ProxyConfig, AnthropicProvider, OpenAIProvider};
///
/// // Deploy a transparent proxy at your app's origin
/// let proxy = ProxyConfig::new("https://your-app.example.com/api");
///
/// // Create providers that route through the proxy
/// let anthropic = AnthropicProvider::with_base_url(proxy.anthropic_url());
/// let openai = OpenAIProvider::with_base_url(proxy.openai_url());
/// ```
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Base URL of the transparent proxy (e.g. "https://proxy.example.com")
    pub base_url: String,
    /// Path prefix for Anthropic routes (default: "/anthropic")
    pub anthropic_prefix: String,
    /// Path prefix for OpenAI routes (default: "/openai")
    pub openai_prefix: String,
}

impl ProxyConfig {
    /// Create a proxy config with default path prefixes.
    ///
    /// The proxy is expected to forward:
    /// - `{base_url}/anthropic/v1/messages` → `https://api.anthropic.com/v1/messages`
    /// - `{base_url}/openai/v1/chat/completions` → `https://api.openai.com/v1/chat/completions`
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            anthropic_prefix: "/anthropic".into(),
            openai_prefix: "/openai".into(),
        }
    }

    /// Create a passthrough proxy config where the proxy handles all routes directly.
    ///
    /// Use this when the proxy is a direct stand-in for a single API
    /// (e.g. mock-llm-service's transparent proxy mode).
    pub fn passthrough(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            anthropic_prefix: String::new(),
            openai_prefix: String::new(),
        }
    }

    /// Get the base URL for Anthropic API calls through the proxy.
    pub fn anthropic_url(&self) -> String {
        format!("{}{}", self.base_url, self.anthropic_prefix)
    }

    /// Get the base URL for OpenAI API calls through the proxy.
    pub fn openai_url(&self) -> String {
        format!("{}{}", self.base_url, self.openai_prefix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_result_creates() {
        let probe = ProbeResult {
            healthy: true,
            rate_limit_remaining: Some(0.82),
            rate_limit_utilization: Some(0.18),
        };
        assert!(probe.healthy);
        assert_eq!(probe.rate_limit_remaining, Some(0.82));
    }

    // Trait object safety check
    #[test]
    fn provider_is_object_safe() {
        fn _assert_object_safe(_: &dyn Provider) {}
    }

    #[test]
    fn proxy_config_default_prefixes() {
        let proxy = ProxyConfig::new("https://proxy.example.com");
        assert_eq!(proxy.anthropic_url(), "https://proxy.example.com/anthropic");
        assert_eq!(proxy.openai_url(), "https://proxy.example.com/openai");
    }

    #[test]
    fn proxy_config_passthrough() {
        let proxy = ProxyConfig::passthrough("http://localhost:8081");
        assert_eq!(proxy.anthropic_url(), "http://localhost:8081");
        assert_eq!(proxy.openai_url(), "http://localhost:8081");
    }

    #[test]
    fn proxy_config_trailing_slash_preserved() {
        let proxy = ProxyConfig::new("https://api.example.com/");
        // Caller is responsible for base_url format
        assert_eq!(proxy.anthropic_url(), "https://api.example.com//anthropic");
    }
}
