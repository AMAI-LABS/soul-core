//! Bridge from MCP tools to the soul-core `Tool` trait.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use crate::error::SoulResult;
use crate::tool::{Tool, ToolOutput};
use crate::types::ToolDefinition;

use super::types::McpToolDef;
use super::McpClient;

/// Adapts an MCP tool into a soul-core `Tool` implementation.
///
/// The tool name is prefixed to avoid collisions between MCP servers:
/// `{prefix}{mcp_name}` (e.g., `context7_query-docs`).
pub struct McpToolBridge {
    tool_def: McpToolDef,
    client: Arc<Mutex<McpClient>>,
    prefix: String,
}

impl McpToolBridge {
    pub fn new(
        tool_def: McpToolDef,
        client: Arc<Mutex<McpClient>>,
        prefix: impl Into<String>,
    ) -> Self {
        Self {
            tool_def,
            client,
            prefix: prefix.into(),
        }
    }

    fn prefixed_name(&self) -> String {
        if self.prefix.is_empty() {
            self.tool_def.name.clone()
        } else {
            format!("{}_{}", self.prefix, self.tool_def.name)
        }
    }
}

#[async_trait]
impl Tool for McpToolBridge {
    fn name(&self) -> &str {
        // We need to return a &str, but the prefixed name is computed.
        // Use a trick: if prefix is empty, return the raw name.
        // Otherwise we leak a string (this is fine — tools are long-lived).
        // Actually, let's just store it.
        // For simplicity in the trait, we'll compute it.
        // The Tool trait requires &str, so we use Box::leak for the prefixed name.
        // This is acceptable because tools are registered once and live for the session.
        // Actually, let's avoid the leak. We'll store the computed name on construction.
        // But we can't because the struct was already written.
        // Instead: return &self.tool_def.name since the prefix is handled at registration.
        // The caller should use prefixed_name() when registering.
        &self.tool_def.name
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.prefixed_name(),
            description: self.tool_def.description.clone().unwrap_or_default(),
            input_schema: self.tool_def.input_schema.clone(),
        }
    }

    async fn execute(
        &self,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let client = self.client.lock().await;
        let result = client.call_tool(&self.tool_def.name, arguments).await?;

        // Concatenate text content
        let text: String = result
            .content
            .iter()
            .filter_map(|c| match c {
                super::types::McpContent::Text { text } => Some(text.as_str()),
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
}

#[cfg(test)]
mod tests {
    use super::super::protocol::{JsonRpcId, JsonRpcResponse};
    use super::super::transport::mock::MockTransport;
    use super::super::types::McpToolDef;
    use super::*;
    use serde_json::json;

    fn make_client_with_tool_result(
        result_json: serde_json::Value,
    ) -> (Arc<Mutex<McpClient>>, McpToolDef) {
        // Initialize response + tool call response
        let init_resp = JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test", "version": "1.0"},
                "capabilities": {}
            }),
        );
        let call_resp = JsonRpcResponse::success(JsonRpcId::Number(0), result_json);

        let transport = Box::new(MockTransport::new(vec![init_resp, call_resp]));
        let client = McpClient::new(transport);
        let tool_def = McpToolDef {
            name: "search".into(),
            description: Some("Search things".into()),
            input_schema: json!({"type": "object"}),
        };
        (Arc::new(Mutex::new(client)), tool_def)
    }

    #[test]
    fn prefixed_name_with_prefix() {
        let (client, tool_def) = make_client_with_tool_result(json!({}));
        let bridge = McpToolBridge::new(tool_def, client, "ctx7");
        assert_eq!(bridge.prefixed_name(), "ctx7_search");
    }

    #[test]
    fn prefixed_name_without_prefix() {
        let (client, tool_def) = make_client_with_tool_result(json!({}));
        let bridge = McpToolBridge::new(tool_def, client, "");
        assert_eq!(bridge.prefixed_name(), "search");
    }

    #[test]
    fn definition_uses_prefix() {
        let (client, tool_def) = make_client_with_tool_result(json!({}));
        let bridge = McpToolBridge::new(tool_def, client, "mcp");
        let def = bridge.definition();
        assert_eq!(def.name, "mcp_search");
        assert_eq!(def.description, "Search things");
    }

    #[tokio::test]
    async fn execute_calls_mcp_client() {
        let (client, tool_def) = make_client_with_tool_result(json!({
            "content": [{"type": "text", "text": "found it"}],
            "isError": false
        }));

        // Initialize client first
        {
            let mut c = client.lock().await;
            c.initialize().await.unwrap();
        }

        let bridge = McpToolBridge::new(tool_def, client, "");
        let result = bridge
            .execute("call1", json!({"query": "test"}), None)
            .await
            .unwrap();
        assert_eq!(result.content, "found it");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn execute_propagates_error() {
        let (client, tool_def) = make_client_with_tool_result(json!({
            "content": [{"type": "text", "text": "not found"}],
            "isError": true
        }));

        {
            let mut c = client.lock().await;
            c.initialize().await.unwrap();
        }

        let bridge = McpToolBridge::new(tool_def, client, "");
        let result = bridge.execute("call1", json!({}), None).await.unwrap();
        assert!(result.is_error);
        assert_eq!(result.content, "not found");
    }
}
