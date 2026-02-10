//! MCP executor — delegates tool calls to MCP servers.

use std::sync::Arc;

use tokio::sync::{mpsc, Mutex};

use crate::error::{SoulError, SoulResult};
use crate::mcp::McpClient;
use crate::tool::ToolOutput;
use crate::types::ToolDefinition;

use super::ToolExecutor;

/// Executes tools by forwarding to an MCP server.
pub struct McpExecutor {
    client: Arc<Mutex<McpClient>>,
}

impl McpExecutor {
    pub fn new(client: Arc<Mutex<McpClient>>) -> Self {
        Self { client }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl ToolExecutor for McpExecutor {
    async fn execute(
        &self,
        definition: &ToolDefinition,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let client = self.client.lock().await;
        let result = client.call_tool(&definition.name, arguments).await?;

        let text: String = result
            .content
            .iter()
            .filter_map(|c| match c {
                crate::mcp::McpContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if result.is_error {
            Ok(ToolOutput::error(text))
        } else {
            Ok(ToolOutput::success(text))
        }
    }

    fn executor_name(&self) -> &str {
        "mcp"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::protocol::{JsonRpcId, JsonRpcResponse};
    use crate::mcp::transport::mock::MockTransport;
    use serde_json::json;

    fn test_def() -> ToolDefinition {
        ToolDefinition {
            name: "test_tool".into(),
            description: "Test".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    #[tokio::test]
    async fn delegates_to_mcp_client() {
        let init_resp = JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test"},
                "capabilities": {}
            }),
        );
        let call_resp = JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({
                "content": [{"type": "text", "text": "mcp result"}],
                "isError": false
            }),
        );

        let transport = Box::new(MockTransport::new(vec![init_resp, call_resp]));
        let client = McpClient::new(transport);
        let client_arc = Arc::new(Mutex::new(client));

        {
            let mut c = client_arc.lock().await;
            c.initialize().await.unwrap();
        }

        let executor = McpExecutor::new(client_arc);
        let result = executor
            .execute(&test_def(), "c1", json!({}), None)
            .await
            .unwrap();
        assert_eq!(result.content, "mcp result");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn propagates_mcp_errors() {
        let init_resp = JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test"},
                "capabilities": {}
            }),
        );
        let call_resp = JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({
                "content": [{"type": "text", "text": "error occurred"}],
                "isError": true
            }),
        );

        let transport = Box::new(MockTransport::new(vec![init_resp, call_resp]));
        let client = McpClient::new(transport);
        let client_arc = Arc::new(Mutex::new(client));

        {
            let mut c = client_arc.lock().await;
            c.initialize().await.unwrap();
        }

        let executor = McpExecutor::new(client_arc);
        let result = executor
            .execute(&test_def(), "c1", json!({}), None)
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[test]
    fn executor_name() {
        let transport = Box::new(MockTransport::new(vec![]));
        let client = McpClient::new(transport);
        let executor = McpExecutor::new(Arc::new(Mutex::new(client)));
        assert_eq!(executor.executor_name(), "mcp");
    }
}
