use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::types::ToolDefinition;

/// A tool that can be executed by the agent
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name (must match the definition name)
    fn name(&self) -> &str;

    /// Tool definition for sending to the LLM
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given arguments
    async fn execute(
        &self,
        call_id: &str,
        arguments: serde_json::Value,
        partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput>;
}

/// Output from a tool execution
#[derive(Debug, Clone)]
pub struct ToolOutput {
    pub content: String,
    pub is_error: bool,
    pub metadata: serde_json::Value,
}

impl ToolOutput {
    pub fn success(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: false,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn error(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            is_error: true,
            metadata: serde_json::Value::Null,
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Registry of tools available to the agent
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools.iter().find(|t| t.name() == name).map(|t| t.as_ref())
    }

    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.iter().map(|t| t.definition()).collect()
    }

    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoTool;

    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "echo".into(),
                description: "Echo back the input".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"]
                }),
            }
        }

        async fn execute(
            &self,
            _call_id: &str,
            arguments: serde_json::Value,
            _partial_tx: Option<mpsc::UnboundedSender<String>>,
        ) -> SoulResult<ToolOutput> {
            let message = arguments
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("no message");
            Ok(ToolOutput::success(message))
        }
    }

    #[test]
    fn tool_output_success() {
        let output = ToolOutput::success("result");
        assert_eq!(output.content, "result");
        assert!(!output.is_error);
    }

    #[test]
    fn tool_output_error() {
        let output = ToolOutput::error("failed");
        assert_eq!(output.content, "failed");
        assert!(output.is_error);
    }

    #[test]
    fn tool_output_with_metadata() {
        let output = ToolOutput::success("ok").with_metadata(json!({"duration_ms": 42}));
        assert_eq!(output.metadata["duration_ms"], 42);
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut registry = ToolRegistry::new();
        assert!(registry.is_empty());

        registry.register(Box::new(EchoTool));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());

        let tool = registry.get("echo");
        assert!(tool.is_some());
        assert_eq!(tool.unwrap().name(), "echo");

        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn registry_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let defs = registry.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "echo");
    }

    #[test]
    fn registry_names() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(EchoTool));

        let names = registry.names();
        assert_eq!(names, vec!["echo"]);
    }

    #[tokio::test]
    async fn tool_execute() {
        let tool = EchoTool;
        let result = tool
            .execute("call_1", json!({"message": "hello world"}), None)
            .await
            .unwrap();
        assert_eq!(result.content, "hello world");
        assert!(!result.is_error);
    }

    // Trait object safety
    #[test]
    fn tool_is_object_safe() {
        fn _assert_object_safe(_: &dyn Tool) {}
    }
}
