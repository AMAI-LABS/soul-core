//! Direct executor — wraps an existing `ToolRegistry` for backward compatibility.

use std::sync::Arc;

#[cfg(test)]
use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::tool::{ToolOutput, ToolRegistry};
use crate::types::ToolDefinition;

use super::ToolExecutor;

/// Wraps a `ToolRegistry` as a `ToolExecutor`.
///
/// This bridges the existing tool system into the executor registry,
/// enabling zero-breaking-change migration.
pub struct DirectExecutor {
    tools: Arc<ToolRegistry>,
}

impl DirectExecutor {
    pub fn new(tools: Arc<ToolRegistry>) -> Self {
        Self { tools }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl ToolExecutor for DirectExecutor {
    async fn execute(
        &self,
        definition: &ToolDefinition,
        call_id: &str,
        arguments: serde_json::Value,
        partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let tool = self
            .tools
            .get(&definition.name)
            .ok_or_else(|| SoulError::ToolExecution {
                tool_name: definition.name.clone(),
                message: format!("Unknown tool: {}", definition.name),
            })?;

        tool.execute(call_id, arguments, partial_tx).await
    }

    fn executor_name(&self) -> &str {
        "direct"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;
    use serde_json::json;

    struct MockTool;

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            "mock"
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: "mock".into(),
                description: "Mock tool".into(),
                input_schema: json!({"type": "object"}),
            }
        }

        async fn execute(
            &self,
            _call_id: &str,
            _arguments: serde_json::Value,
            _partial_tx: Option<mpsc::UnboundedSender<String>>,
        ) -> SoulResult<ToolOutput> {
            Ok(ToolOutput::success("mock result"))
        }
    }

    #[tokio::test]
    async fn delegates_to_tool_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(MockTool));
        let executor = DirectExecutor::new(Arc::new(registry));

        let def = ToolDefinition {
            name: "mock".into(),
            description: "".into(),
            input_schema: json!({}),
        };

        let result = executor.execute(&def, "c1", json!({}), None).await.unwrap();
        assert_eq!(result.content, "mock result");
    }

    #[tokio::test]
    async fn unknown_tool_errors() {
        let registry = ToolRegistry::new();
        let executor = DirectExecutor::new(Arc::new(registry));

        let def = ToolDefinition {
            name: "nonexistent".into(),
            description: "".into(),
            input_schema: json!({}),
        };

        let result = executor.execute(&def, "c1", json!({}), None).await;
        assert!(result.is_err());
    }

    #[test]
    fn executor_name_is_direct() {
        let registry = ToolRegistry::new();
        let executor = DirectExecutor::new(Arc::new(registry));
        assert_eq!(executor.executor_name(), "direct");
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<DirectExecutor>();
    }
}
