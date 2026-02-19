use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::SoulResult;

/// Events received from a messaging gateway
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GatewayEvent {
    MessageReceived {
        channel_id: String,
        sender: String,
        text: String,
        #[serde(default)]
        metadata: serde_json::Value,
    },
    Error {
        source: String,
        message: String,
    },
}

/// Outbound message from agent to gateway
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GatewayMessage {
    Text { text: String },
}

impl GatewayMessage {
    pub fn text(s: impl Into<String>) -> Self {
        GatewayMessage::Text { text: s.into() }
    }
}

/// Core gateway trait — abstracts messaging platform communication
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait Gateway: Send + Sync {
    /// Gateway name (e.g. "telegram", "imessage", "whatsapp")
    fn name(&self) -> &str;

    /// Start receiving messages, sending events through the channel
    async fn start(&self, event_tx: mpsc::UnboundedSender<GatewayEvent>) -> SoulResult<()>;

    /// Send a message to a specific channel/chat
    async fn send(&self, channel_id: &str, message: GatewayMessage) -> SoulResult<()>;

    /// Stop the gateway
    async fn stop(&self) -> SoulResult<()>;
}

/// Registry of available gateways
pub struct GatewayRegistry {
    gateways: Vec<Box<dyn Gateway>>,
}

impl GatewayRegistry {
    pub fn new() -> Self {
        Self {
            gateways: Vec::new(),
        }
    }

    pub fn register(&mut self, gateway: Box<dyn Gateway>) {
        self.gateways.push(gateway);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Gateway> {
        self.gateways
            .iter()
            .find(|g| g.name() == name)
            .map(|g| g.as_ref())
    }

    pub fn names(&self) -> Vec<&str> {
        self.gateways.iter().map(|g| g.name()).collect()
    }

    pub fn len(&self) -> usize {
        self.gateways.len()
    }

    pub fn is_empty(&self) -> bool {
        self.gateways.is_empty()
    }
}

impl Default for GatewayRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_event_serializes() {
        let event = GatewayEvent::MessageReceived {
            channel_id: "chat_123".into(),
            sender: "user_456".into(),
            text: "hello".into(),
            metadata: serde_json::json!({}),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains(r#""type":"message_received""#));
        assert!(json.contains(r#""channel_id":"chat_123""#));
    }

    #[test]
    fn gateway_message_text() {
        let msg = GatewayMessage::text("hello world");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains(r#""type":"text""#));
        assert!(json.contains(r#""text":"hello world""#));
    }

    #[test]
    fn gateway_registry_empty() {
        let registry = GatewayRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.names().is_empty());
        assert!(registry.get("telegram").is_none());
    }

    #[test]
    fn gateway_trait_is_object_safe() {
        fn _assert_object_safe(_: &dyn Gateway) {}
    }
}
