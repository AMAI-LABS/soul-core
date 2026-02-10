use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::types::*;

/// Core provider trait — abstracts LLM API communication
#[async_trait]
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
}
