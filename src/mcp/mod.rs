//! MCP (Model Context Protocol) client for integrating external tool servers.
//!
//! Provides JSON-RPC transport, MCP protocol types, and a bridge to soul-core's `Tool` trait.

pub mod bridge;
pub mod protocol;
pub mod transport;
pub mod types;

pub use bridge::McpToolBridge;
pub use types::{McpContent, McpResource, McpServerInfo, McpToolDef, McpToolResult};

use crate::error::{SoulError, SoulResult};
use crate::tool::ToolRegistry;

use protocol::{JsonRpcId, JsonRpcNotification, JsonRpcRequest};
use transport::McpTransport;

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Client for communicating with an MCP server.
pub struct McpClient {
    transport: Box<dyn McpTransport>,
    next_id: AtomicI64,
    server_info: Option<McpServerInfo>,
    tools: Vec<McpToolDef>,
    resources: Vec<McpResource>,
}

impl McpClient {
    pub fn new(transport: Box<dyn McpTransport>) -> Self {
        Self {
            transport,
            next_id: AtomicI64::new(1),
            server_info: None,
            tools: Vec::new(),
            resources: Vec::new(),
        }
    }

    fn next_id(&self) -> JsonRpcId {
        JsonRpcId::Number(self.next_id.fetch_add(1, Ordering::SeqCst))
    }

    /// Initialize the MCP connection (handshake).
    pub async fn initialize(&mut self) -> SoulResult<&McpServerInfo> {
        let req =
            JsonRpcRequest::new(self.next_id(), "initialize").with_params(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "soul-core",
                    "version": env!("CARGO_PKG_VERSION")
                },
                "capabilities": {}
            }));

        let resp = self.transport.send(req).await?;
        if let Some(err) = resp.error {
            return Err(SoulError::Mcp {
                server: "unknown".into(),
                message: err.message,
            });
        }

        let result = resp.result.unwrap_or_default();

        let server_name = result
            .pointer("/serverInfo/name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();
        let server_version = result
            .pointer("/serverInfo/version")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let capabilities: types::ServerCapabilities =
            serde_json::from_value(result.get("capabilities").cloned().unwrap_or_default())
                .unwrap_or_default();
        let instructions = result
            .get("instructions")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        self.server_info = Some(McpServerInfo {
            name: server_name,
            version: server_version,
            capabilities,
            instructions,
        });

        // Send initialized notification
        let notif = JsonRpcNotification::new("notifications/initialized");
        self.transport.send_notification(notif).await?;

        Ok(self.server_info.as_ref().unwrap())
    }

    /// List available tools from the server.
    pub async fn list_tools(&mut self) -> SoulResult<&[McpToolDef]> {
        let req = JsonRpcRequest::new(self.next_id(), "tools/list");
        let resp = self.transport.send(req).await?;

        if let Some(err) = resp.error {
            return Err(SoulError::JsonRpc {
                code: err.code,
                message: err.message,
            });
        }

        let result = resp.result.unwrap_or_default();
        let tools: Vec<McpToolDef> =
            serde_json::from_value(result.get("tools").cloned().unwrap_or_default())
                .unwrap_or_default();

        self.tools = tools;
        Ok(&self.tools)
    }

    /// Call a tool on the server.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> SoulResult<McpToolResult> {
        let req =
            JsonRpcRequest::new(self.next_id(), "tools/call").with_params(serde_json::json!({
                "name": name,
                "arguments": arguments,
            }));

        let resp = self.transport.send(req).await?;
        if let Some(err) = resp.error {
            return Err(SoulError::JsonRpc {
                code: err.code,
                message: err.message,
            });
        }

        let result = resp.result.unwrap_or_default();
        let tool_result: McpToolResult =
            serde_json::from_value(result).map_err(|e| SoulError::Mcp {
                server: self
                    .server_info
                    .as_ref()
                    .map(|s| s.name.clone())
                    .unwrap_or_default(),
                message: format!("Failed to parse tool result: {e}"),
            })?;

        Ok(tool_result)
    }

    /// List available resources from the server.
    pub async fn list_resources(&mut self) -> SoulResult<&[McpResource]> {
        let req = JsonRpcRequest::new(self.next_id(), "resources/list");
        let resp = self.transport.send(req).await?;

        if let Some(err) = resp.error {
            return Err(SoulError::JsonRpc {
                code: err.code,
                message: err.message,
            });
        }

        let result = resp.result.unwrap_or_default();
        let resources: Vec<McpResource> =
            serde_json::from_value(result.get("resources").cloned().unwrap_or_default())
                .unwrap_or_default();

        self.resources = resources;
        Ok(&self.resources)
    }

    /// Read a resource by URI.
    pub async fn read_resource(&self, uri: &str) -> SoulResult<String> {
        let req =
            JsonRpcRequest::new(self.next_id(), "resources/read").with_params(serde_json::json!({
                "uri": uri,
            }));

        let resp = self.transport.send(req).await?;
        if let Some(err) = resp.error {
            return Err(SoulError::JsonRpc {
                code: err.code,
                message: err.message,
            });
        }

        let result = resp.result.unwrap_or_default();
        let contents = result
            .pointer("/contents/0/text")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(contents)
    }

    /// Register all MCP tools into a ToolRegistry.
    pub async fn register_tools(
        client_arc: Arc<Mutex<McpClient>>,
        registry: &mut ToolRegistry,
        prefix: &str,
    ) -> SoulResult<usize> {
        let tools = {
            let mut client = client_arc.lock().await;
            client.list_tools().await?.to_vec()
        };
        let count = tools.len();

        for tool_def in tools {
            let bridge = McpToolBridge::new(tool_def, client_arc.clone(), prefix);
            registry.register(Box::new(bridge));
        }

        Ok(count)
    }

    /// Close the connection.
    pub async fn close(&self) -> SoulResult<()> {
        self.transport.close().await
    }

    /// Get server info (available after initialize).
    pub fn server_info(&self) -> Option<&McpServerInfo> {
        self.server_info.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use protocol::{JsonRpcId, JsonRpcResponse};
    use serde_json::json;
    use transport::mock::MockTransport;

    fn init_response() -> JsonRpcResponse {
        JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "test-server", "version": "1.0.0"},
                "capabilities": {"tools": {}},
                "instructions": "Use these tools wisely"
            }),
        )
    }

    fn tools_list_response() -> JsonRpcResponse {
        JsonRpcResponse::success(
            JsonRpcId::Number(0),
            json!({
                "tools": [
                    {
                        "name": "search",
                        "description": "Search things",
                        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}}
                    },
                    {
                        "name": "read",
                        "description": "Read a file",
                        "inputSchema": {"type": "object"}
                    }
                ]
            }),
        )
    }

    #[tokio::test]
    async fn initialize_handshake() {
        let transport = Box::new(MockTransport::new(vec![init_response()]));
        let mut client = McpClient::new(transport);

        let info = client.initialize().await.unwrap();
        assert_eq!(info.name, "test-server");
        assert_eq!(info.version.as_deref(), Some("1.0.0"));
        assert_eq!(info.instructions.as_deref(), Some("Use these tools wisely"));
    }

    #[tokio::test]
    async fn list_tools_returns_definitions() {
        let transport = Box::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(),
        ]));
        let mut client = McpClient::new(transport);
        client.initialize().await.unwrap();

        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[1].name, "read");
    }

    #[tokio::test]
    async fn call_tool_returns_result() {
        let transport = Box::new(MockTransport::new(vec![
            init_response(),
            JsonRpcResponse::success(
                JsonRpcId::Number(0),
                json!({
                    "content": [{"type": "text", "text": "42 results found"}],
                    "isError": false
                }),
            ),
        ]));
        let mut client = McpClient::new(transport);
        client.initialize().await.unwrap();

        let result = client
            .call_tool("search", json!({"query": "test"}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
    }

    #[tokio::test]
    async fn list_resources() {
        let transport = Box::new(MockTransport::new(vec![
            init_response(),
            JsonRpcResponse::success(
                JsonRpcId::Number(0),
                json!({
                    "resources": [
                        {"uri": "file:///tmp/a.txt", "name": "a.txt"}
                    ]
                }),
            ),
        ]));
        let mut client = McpClient::new(transport);
        client.initialize().await.unwrap();

        let resources = client.list_resources().await.unwrap();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].uri, "file:///tmp/a.txt");
    }

    #[tokio::test]
    async fn read_resource() {
        let transport = Box::new(MockTransport::new(vec![
            init_response(),
            JsonRpcResponse::success(
                JsonRpcId::Number(0),
                json!({
                    "contents": [{"uri": "file:///tmp/a.txt", "text": "hello world"}]
                }),
            ),
        ]));
        let mut client = McpClient::new(transport);
        client.initialize().await.unwrap();

        let content = client.read_resource("file:///tmp/a.txt").await.unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn register_tools_into_registry() {
        let transport = Box::new(MockTransport::new(vec![
            init_response(),
            tools_list_response(),
        ]));
        let client = McpClient::new(transport);
        let client_arc = Arc::new(Mutex::new(client));

        // Initialize
        {
            let mut c = client_arc.lock().await;
            c.initialize().await.unwrap();
        }

        let mut registry = ToolRegistry::new();
        let count = McpClient::register_tools(client_arc, &mut registry, "mcp")
            .await
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(registry.len(), 2);
    }

    #[tokio::test]
    async fn server_info_before_init() {
        let transport = Box::new(MockTransport::new(vec![]));
        let client = McpClient::new(transport);
        assert!(client.server_info().is_none());
    }

    #[tokio::test]
    async fn initialize_error_returns_err() {
        let transport = Box::new(MockTransport::new(vec![JsonRpcResponse::error(
            JsonRpcId::Number(1),
            protocol::JsonRpcError {
                code: -32600,
                message: "Bad request".into(),
                data: None,
            },
        )]));
        let mut client = McpClient::new(transport);
        let result = client.initialize().await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn close_succeeds() {
        let transport = Box::new(MockTransport::new(vec![]));
        let client = McpClient::new(transport);
        client.close().await.unwrap();
    }
}
