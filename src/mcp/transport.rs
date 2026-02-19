//! MCP transport layer — abstracts communication with MCP servers.

#[cfg(test)]
use async_trait::async_trait;

use super::protocol::{JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
use crate::error::SoulResult;

/// Transport trait for sending JSON-RPC messages to an MCP server.
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait McpTransport: Send + Sync {
    /// Send a request and wait for a response.
    async fn send(&self, request: JsonRpcRequest) -> SoulResult<JsonRpcResponse>;

    /// Send a notification (no response expected).
    async fn send_notification(&self, notification: JsonRpcNotification) -> SoulResult<()>;

    /// Close the transport.
    async fn close(&self) -> SoulResult<()>;
}

/// Mock transport for testing — returns pre-configured responses.
#[cfg(test)]
pub mod mock {
    use super::super::protocol::{JsonRpcId, JsonRpcResponse};
    use super::*;
    use std::sync::Mutex;

    pub struct MockTransport {
        responses: Mutex<Vec<JsonRpcResponse>>,
        sent_requests: Mutex<Vec<JsonRpcRequest>>,
        sent_notifications: Mutex<Vec<JsonRpcNotification>>,
    }

    impl MockTransport {
        pub fn new(responses: Vec<JsonRpcResponse>) -> Self {
            Self {
                responses: Mutex::new(responses),
                sent_requests: Mutex::new(Vec::new()),
                sent_notifications: Mutex::new(Vec::new()),
            }
        }

        pub fn sent_requests(&self) -> Vec<JsonRpcRequest> {
            self.sent_requests.lock().unwrap().clone()
        }

        pub fn sent_notifications(&self) -> Vec<JsonRpcNotification> {
            self.sent_notifications.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl McpTransport for MockTransport {
        async fn send(&self, request: JsonRpcRequest) -> SoulResult<JsonRpcResponse> {
            self.sent_requests.lock().unwrap().push(request.clone());
            let mut responses = self.responses.lock().unwrap();
            if responses.is_empty() {
                Ok(JsonRpcResponse::error(
                    request.id,
                    super::super::protocol::JsonRpcError {
                        code: -32603,
                        message: "No more mock responses".into(),
                        data: None,
                    },
                ))
            } else {
                let mut resp = responses.remove(0);
                resp.id = request.id;
                Ok(resp)
            }
        }

        async fn send_notification(&self, notification: JsonRpcNotification) -> SoulResult<()> {
            self.sent_notifications.lock().unwrap().push(notification);
            Ok(())
        }

        async fn close(&self) -> SoulResult<()> {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::protocol::{JsonRpcId, JsonRpcNotification, JsonRpcRequest, JsonRpcResponse};
    use super::mock::MockTransport;
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn mock_transport_returns_response() {
        let transport = MockTransport::new(vec![JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({"tools": []}),
        )]);

        let req = JsonRpcRequest::new(1i64, "tools/list");
        let resp = transport.send(req).await.unwrap();
        assert!(!resp.is_error());
        assert_eq!(resp.id, JsonRpcId::Number(1));
    }

    #[tokio::test]
    async fn mock_transport_tracks_requests() {
        let transport = MockTransport::new(vec![JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({}),
        )]);

        let req = JsonRpcRequest::new(1i64, "tools/list");
        transport.send(req).await.unwrap();

        let sent = transport.sent_requests();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].method, "tools/list");
    }

    #[tokio::test]
    async fn mock_transport_notification() {
        let transport = MockTransport::new(vec![]);
        let notif = JsonRpcNotification::new("notifications/initialized");
        transport.send_notification(notif).await.unwrap();

        let sent = transport.sent_notifications();
        assert_eq!(sent.len(), 1);
    }

    #[tokio::test]
    async fn mock_transport_empty_returns_error() {
        let transport = MockTransport::new(vec![]);
        let req = JsonRpcRequest::new(1i64, "tools/list");
        let resp = transport.send(req).await.unwrap();
        assert!(resp.is_error());
    }
}
