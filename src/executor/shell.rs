//! Shell executor — runs tools as shell commands via VirtualExecutor.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::tool::ToolOutput;
use crate::types::ToolDefinition;
use crate::vexec::VirtualExecutor;

use super::ToolExecutor;

/// Executes tools by running shell commands through a [`VirtualExecutor`].
///
/// The tool arguments should contain a `command` field with the shell command to run.
/// On native platforms, use [`NativeExecutor`](crate::vexec::NativeExecutor).
/// In WASM / tests, inject a [`MockExecutor`](crate::vexec::MockExecutor) or
/// [`NoopExecutor`](crate::vexec::NoopExecutor).
pub struct ShellExecutor {
    exec: Arc<dyn VirtualExecutor>,
    default_timeout_secs: u64,
    cwd: Option<String>,
}

impl ShellExecutor {
    pub fn new(exec: Arc<dyn VirtualExecutor>) -> Self {
        Self {
            exec,
            default_timeout_secs: 120,
            cwd: None,
        }
    }

    /// Create a ShellExecutor backed by the native OS executor.
    #[cfg(feature = "native")]
    pub fn native() -> Self {
        Self::new(Arc::new(crate::vexec::NativeExecutor::new()))
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.default_timeout_secs = secs;
        self
    }

    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }
}

#[async_trait]
impl ToolExecutor for ShellExecutor {
    async fn execute(
        &self,
        definition: &ToolDefinition,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let command = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SoulError::ToolExecution {
                tool_name: definition.name.clone(),
                message: "Missing 'command' argument".into(),
            })?;

        let output = self
            .exec
            .exec_shell(command, self.default_timeout_secs, self.cwd.as_deref())
            .await
            .map_err(|e| SoulError::ToolExecution {
                tool_name: definition.name.clone(),
                message: format!("Failed to execute: {e}"),
            })?;

        if output.success() {
            Ok(ToolOutput::success(output.stdout))
        } else {
            let content = if output.stderr.is_empty() {
                format!("Exit code: {}\n{}", output.exit_code, output.stdout)
            } else {
                format!(
                    "Exit code: {}\nstderr: {}\nstdout: {}",
                    output.exit_code, output.stderr, output.stdout
                )
            };
            Ok(ToolOutput::error(content))
        }
    }

    fn executor_name(&self) -> &str {
        "shell"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vexec::{ExecOutput, MockExecutor};
    use serde_json::json;

    fn test_def() -> ToolDefinition {
        ToolDefinition {
            name: "shell_test".into(),
            description: "Test".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn mock_ok(stdout: &str) -> Arc<dyn VirtualExecutor> {
        Arc::new(MockExecutor::always_ok(stdout))
    }

    fn mock_fail(exit_code: i32) -> Arc<dyn VirtualExecutor> {
        Arc::new(MockExecutor::new(vec![ExecOutput {
            stdout: String::new(),
            stderr: "error output".into(),
            exit_code,
        }]))
    }

    #[tokio::test]
    async fn echo_command() {
        let executor = ShellExecutor::new(mock_ok("hello\n"));
        let result = executor
            .execute(&test_def(), "c1", json!({"command": "echo hello"}), None)
            .await
            .unwrap();
        assert_eq!(result.content.trim(), "hello");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn missing_command_errors() {
        let executor = ShellExecutor::new(mock_ok(""));
        let result = executor
            .execute(&test_def(), "c1", json!({"other": "value"}), None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn failing_command() {
        let executor = ShellExecutor::new(mock_fail(42));
        let result = executor
            .execute(&test_def(), "c1", json!({"command": "exit 42"}), None)
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("42"));
    }

    #[test]
    fn executor_name() {
        let executor = ShellExecutor::new(mock_ok(""));
        assert_eq!(executor.executor_name(), "shell");
    }

    #[tokio::test]
    async fn custom_timeout() {
        let executor = ShellExecutor::new(mock_ok("fast")).with_timeout(1);
        let result = executor
            .execute(&test_def(), "c1", json!({"command": "echo fast"}), None)
            .await
            .unwrap();
        assert!(!result.is_error);
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_echo() {
        let executor = ShellExecutor::native();
        let result = executor
            .execute(&test_def(), "c1", json!({"command": "echo hello"}), None)
            .await
            .unwrap();
        assert_eq!(result.content.trim(), "hello");
        assert!(!result.is_error);
    }
}
