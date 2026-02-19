//! MCP protocol types for tool definitions, resources, and capabilities.

use serde::{Deserialize, Serialize};

/// MCP tool definition as returned by the server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpToolDef {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

/// MCP resource definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResource {
    pub uri: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Content returned from MCP tool execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum McpContent {
    Text { text: String },
    Image { data: String, mime_type: String },
    Resource { resource: McpResourceContent },
}

/// Resource content within an MCP tool result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResourceContent {
    pub uri: String,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Result from an MCP tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct McpToolResult {
    pub content: Vec<McpContent>,
    #[serde(default)]
    pub is_error: bool,
}

/// Server capabilities returned during initialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ServerCapabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resources: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logging: Option<serde_json::Value>,
}

/// Information about an MCP server.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerInfo {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(default)]
    pub capabilities: ServerCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_def_serializes() {
        let tool = McpToolDef {
            name: "read_file".into(),
            description: Some("Read a file".into()),
            input_schema: json!({
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }),
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("read_file"));
        assert!(json.contains("inputSchema"));
    }

    #[test]
    fn tool_def_roundtrip() {
        let tool = McpToolDef {
            name: "search".into(),
            description: None,
            input_schema: json!({"type": "object"}),
        };
        let json = serde_json::to_string(&tool).unwrap();
        let deserialized: McpToolDef = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "search");
        assert!(deserialized.description.is_none());
    }

    #[test]
    fn resource_serializes() {
        let resource = McpResource {
            uri: "file:///tmp/test.txt".into(),
            name: "test.txt".into(),
            description: Some("A test file".into()),
            mime_type: Some("text/plain".into()),
        };
        let json = serde_json::to_string(&resource).unwrap();
        assert!(json.contains("file:///tmp/test.txt"));
    }

    #[test]
    fn content_text_serializes() {
        let content = McpContent::Text {
            text: "hello world".into(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("hello world"));
    }

    #[test]
    fn content_image_serializes() {
        let content = McpContent::Image {
            data: "base64data".into(),
            mime_type: "image/png".into(),
        };
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.contains("\"type\":\"image\""));
    }

    #[test]
    fn tool_result_serializes() {
        let result = McpToolResult {
            content: vec![McpContent::Text {
                text: "result".into(),
            }],
            is_error: false,
        };
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: McpToolResult = serde_json::from_str(&json).unwrap();
        assert!(!deserialized.is_error);
        assert_eq!(deserialized.content.len(), 1);
    }

    #[test]
    fn server_capabilities_default() {
        let caps = ServerCapabilities::default();
        assert!(caps.tools.is_none());
        assert!(caps.resources.is_none());
    }

    #[test]
    fn server_info_serializes() {
        let info = McpServerInfo {
            name: "test-server".into(),
            version: Some("1.0.0".into()),
            capabilities: ServerCapabilities {
                tools: Some(json!({})),
                ..Default::default()
            },
            instructions: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("test-server"));
    }
}
