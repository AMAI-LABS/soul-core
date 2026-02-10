//! Executor registry — config-driven tool execution with pluggable backends.
//!
//! The `ToolExecutor` trait abstracts tool execution. Implementations include:
//! - `DirectExecutor` — wraps existing `ToolRegistry` (backward compat)
//! - `ShellExecutor` — runs shell commands
//! - `HttpExecutor` — makes HTTP calls
//! - `McpExecutor` — delegates to MCP servers
//! - `LlmExecutor` — delegates to an LLM

pub mod direct;
pub mod http;
pub mod llm;
pub mod mcp;
pub mod shell;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::tool::ToolOutput;
use crate::types::ToolDefinition;

/// Trait for executing tools via different backends.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool call.
    async fn execute(
        &self,
        definition: &ToolDefinition,
        call_id: &str,
        arguments: serde_json::Value,
        partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput>;

    /// Name of this executor type.
    fn executor_name(&self) -> &str;
}

/// A tool defined by configuration (executor + config).
#[derive(Debug, Clone)]
pub struct ConfigTool {
    pub definition: ToolDefinition,
    pub executor_name: String,
    pub executor_config: serde_json::Value,
}

/// Registry of executors and config-defined tools.
///
/// Routes tool calls to the appropriate executor based on configuration.
pub struct ExecutorRegistry {
    executors: HashMap<String, Arc<dyn ToolExecutor>>,
    config_tools: HashMap<String, ConfigTool>,
    fallback: Option<Arc<dyn ToolExecutor>>,
}

impl ExecutorRegistry {
    pub fn new() -> Self {
        Self {
            executors: HashMap::new(),
            config_tools: HashMap::new(),
            fallback: None,
        }
    }

    /// Register an executor by name.
    pub fn register_executor(&mut self, executor: Arc<dyn ToolExecutor>) {
        self.executors
            .insert(executor.executor_name().to_string(), executor);
    }

    /// Register a config-defined tool.
    pub fn register_config_tool(&mut self, tool: ConfigTool) {
        self.config_tools.insert(tool.definition.name.clone(), tool);
    }

    /// Set a fallback executor for tools not found in config.
    pub fn set_fallback(&mut self, executor: Arc<dyn ToolExecutor>) {
        self.fallback = Some(executor);
    }

    /// Execute a tool by name, routing to the appropriate executor.
    pub async fn execute(
        &self,
        tool_name: &str,
        call_id: &str,
        arguments: serde_json::Value,
        partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        // Check config tools first
        if let Some(config_tool) = self.config_tools.get(tool_name) {
            if let Some(executor) = self.executors.get(&config_tool.executor_name) {
                return executor
                    .execute(&config_tool.definition, call_id, arguments, partial_tx)
                    .await;
            }
            return Err(SoulError::ExecutorNotFound {
                name: config_tool.executor_name.clone(),
            });
        }

        // Try fallback
        if let Some(fallback) = &self.fallback {
            let def = ToolDefinition {
                name: tool_name.to_string(),
                description: String::new(),
                input_schema: serde_json::json!({"type": "object"}),
            };
            return fallback.execute(&def, call_id, arguments, partial_tx).await;
        }

        Err(SoulError::ToolExecution {
            tool_name: tool_name.to_string(),
            message: format!("No executor found for tool '{tool_name}'"),
        })
    }

    /// Get all tool definitions from config tools.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.config_tools
            .values()
            .map(|ct| ct.definition.clone())
            .collect()
    }

    /// Check if a tool exists in the registry.
    pub fn has_tool(&self, name: &str) -> bool {
        self.config_tools.contains_key(name) || self.fallback.is_some()
    }
}

impl Default for ExecutorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct EchoExecutor;

    #[async_trait]
    impl ToolExecutor for EchoExecutor {
        async fn execute(
            &self,
            definition: &ToolDefinition,
            _call_id: &str,
            arguments: serde_json::Value,
            _partial_tx: Option<mpsc::UnboundedSender<String>>,
        ) -> SoulResult<ToolOutput> {
            Ok(ToolOutput::success(format!(
                "{}({})",
                definition.name, arguments
            )))
        }

        fn executor_name(&self) -> &str {
            "echo"
        }
    }

    struct FailExecutor;

    #[async_trait]
    impl ToolExecutor for FailExecutor {
        async fn execute(
            &self,
            _definition: &ToolDefinition,
            _call_id: &str,
            _arguments: serde_json::Value,
            _partial_tx: Option<mpsc::UnboundedSender<String>>,
        ) -> SoulResult<ToolOutput> {
            Ok(ToolOutput::error("always fails"))
        }

        fn executor_name(&self) -> &str {
            "fail"
        }
    }

    fn config_tool(name: &str, executor: &str) -> ConfigTool {
        ConfigTool {
            definition: ToolDefinition {
                name: name.into(),
                description: format!("Tool {name}"),
                input_schema: json!({"type": "object"}),
            },
            executor_name: executor.into(),
            executor_config: json!({}),
        }
    }

    #[tokio::test]
    async fn routes_to_correct_executor() {
        let mut registry = ExecutorRegistry::new();
        registry.register_executor(Arc::new(EchoExecutor));
        registry.register_config_tool(config_tool("my_tool", "echo"));

        let result = registry
            .execute("my_tool", "c1", json!({"a": 1}), None)
            .await
            .unwrap();
        assert!(result.content.contains("my_tool"));
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn missing_executor_errors() {
        let mut registry = ExecutorRegistry::new();
        registry.register_config_tool(config_tool("my_tool", "nonexistent"));

        let result = registry.execute("my_tool", "c1", json!({}), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unknown_tool_errors() {
        let registry = ExecutorRegistry::new();
        let result = registry.execute("unknown", "c1", json!({}), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn fallback_executor() {
        let mut registry = ExecutorRegistry::new();
        registry.set_fallback(Arc::new(EchoExecutor));

        let result = registry
            .execute("anything", "c1", json!({}), None)
            .await
            .unwrap();
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn config_tool_takes_priority_over_fallback() {
        let mut registry = ExecutorRegistry::new();
        registry.register_executor(Arc::new(FailExecutor));
        registry.set_fallback(Arc::new(EchoExecutor));
        registry.register_config_tool(config_tool("my_tool", "fail"));

        let result = registry
            .execute("my_tool", "c1", json!({}), None)
            .await
            .unwrap();
        // Should use FailExecutor, not EchoExecutor fallback
        assert!(result.is_error);
    }

    #[test]
    fn definitions_returns_config_tools() {
        let mut registry = ExecutorRegistry::new();
        registry.register_config_tool(config_tool("tool_a", "echo"));
        registry.register_config_tool(config_tool("tool_b", "echo"));

        let defs = registry.definitions();
        assert_eq!(defs.len(), 2);
    }

    #[test]
    fn has_tool_checks_config_and_fallback() {
        let mut registry = ExecutorRegistry::new();
        assert!(!registry.has_tool("anything"));

        registry.register_config_tool(config_tool("my_tool", "echo"));
        assert!(registry.has_tool("my_tool"));
        assert!(!registry.has_tool("other"));

        registry.set_fallback(Arc::new(EchoExecutor));
        assert!(registry.has_tool("other")); // fallback covers all
    }

    #[tokio::test]
    async fn multiple_executors() {
        let mut registry = ExecutorRegistry::new();
        registry.register_executor(Arc::new(EchoExecutor));
        registry.register_executor(Arc::new(FailExecutor));
        registry.register_config_tool(config_tool("echo_tool", "echo"));
        registry.register_config_tool(config_tool("fail_tool", "fail"));

        let r1 = registry
            .execute("echo_tool", "c1", json!({}), None)
            .await
            .unwrap();
        assert!(!r1.is_error);

        let r2 = registry
            .execute("fail_tool", "c2", json!({}), None)
            .await
            .unwrap();
        assert!(r2.is_error);
    }
}
