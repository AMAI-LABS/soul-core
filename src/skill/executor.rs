//! Skill executors — run skill definitions via VirtualExecutor.

use std::sync::Arc;

use super::SkillDefinition;
use crate::error::{SoulError, SoulResult};
use crate::tool::ToolOutput;
use crate::vexec::VirtualExecutor;

/// Trait for executing skills.
#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
pub trait SkillExecutor: Send + Sync {
    async fn execute(
        &self,
        skill: &SkillDefinition,
        arguments: &serde_json::Value,
    ) -> SoulResult<ToolOutput>;
}

/// Executes Shell-type skills by rendering templates and running commands
/// through a [`VirtualExecutor`].
pub struct ShellSkillExecutor {
    exec: Arc<dyn VirtualExecutor>,
    default_timeout_secs: u64,
}

impl ShellSkillExecutor {
    pub fn new(exec: Arc<dyn VirtualExecutor>) -> Self {
        Self {
            exec,
            default_timeout_secs: 120,
        }
    }

    /// Create a ShellSkillExecutor backed by the native OS executor.
    #[cfg(feature = "native")]
    pub fn native() -> Self {
        Self::new(Arc::new(crate::vexec::NativeExecutor::new()))
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.default_timeout_secs = secs;
        self
    }

    /// Render a template by replacing `{{key}}` placeholders with argument values.
    pub fn render_template(template: &str, arguments: &serde_json::Value) -> SoulResult<String> {
        let mut result = template.to_string();

        if let Some(obj) = arguments.as_object() {
            for (key, value) in obj {
                let placeholder = format!("{{{{{key}}}}}");
                let replacement = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                result = result.replace(&placeholder, &replacement);
            }
        }

        // Check for unresolved placeholders
        if result.contains("{{") && result.contains("}}") {
            return Err(SoulError::ToolExecution {
                tool_name: "shell_skill".into(),
                message: format!("Unresolved placeholders in template: {result}"),
            });
        }

        Ok(result)
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl SkillExecutor for ShellSkillExecutor {
    async fn execute(
        &self,
        skill: &SkillDefinition,
        arguments: &serde_json::Value,
    ) -> SoulResult<ToolOutput> {
        match &skill.execution {
            super::SkillExecution::Shell {
                command_template,
                timeout_secs,
            } => {
                let command = Self::render_template(command_template, arguments)?;
                let timeout = timeout_secs.unwrap_or(self.default_timeout_secs);

                let output = self
                    .exec
                    .exec_shell(&command, timeout, None)
                    .await
                    .map_err(|e| SoulError::ToolExecution {
                        tool_name: skill.name.clone(),
                        message: format!("Failed to execute command: {e}"),
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
            other => Err(SoulError::ToolExecution {
                tool_name: skill.name.clone(),
                message: format!("ShellSkillExecutor cannot handle execution type: {other:?}"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vexec::{ExecOutput, MockExecutor};
    use serde_json::json;

    fn mock_ok(stdout: &str) -> Arc<dyn VirtualExecutor> {
        Arc::new(MockExecutor::always_ok(stdout))
    }

    fn mock_fail(exit_code: i32) -> Arc<dyn VirtualExecutor> {
        Arc::new(MockExecutor::new(vec![ExecOutput {
            stdout: String::new(),
            stderr: "error".into(),
            exit_code,
        }]))
    }

    #[test]
    fn render_simple_template() {
        let result =
            ShellSkillExecutor::render_template("echo '{{message}}'", &json!({"message": "hello"}))
                .unwrap();
        assert_eq!(result, "echo 'hello'");
    }

    #[test]
    fn render_multiple_placeholders() {
        let result = ShellSkillExecutor::render_template(
            "grep '{{pattern}}' {{file}}",
            &json!({"pattern": "TODO", "file": "main.rs"}),
        )
        .unwrap();
        assert_eq!(result, "grep 'TODO' main.rs");
    }

    #[test]
    fn render_number_value() {
        let result =
            ShellSkillExecutor::render_template("sleep {{seconds}}", &json!({"seconds": 5}))
                .unwrap();
        assert_eq!(result, "sleep 5");
    }

    #[test]
    fn render_unresolved_placeholder_errors() {
        let result =
            ShellSkillExecutor::render_template("echo '{{missing}}'", &json!({"other": "value"}));
        assert!(result.is_err());
    }

    #[test]
    fn render_no_placeholders() {
        let result = ShellSkillExecutor::render_template("echo hello", &json!({})).unwrap();
        assert_eq!(result, "echo hello");
    }

    #[tokio::test]
    async fn execute_echo_command() {
        let executor = ShellSkillExecutor::new(mock_ok("hello\n"));
        let skill = SkillDefinition {
            name: "echo_test".into(),
            description: "Test echo".into(),
            input_schema: json!({"type": "object"}),
            execution: super::super::SkillExecution::Shell {
                command_template: "echo '{{message}}'".into(),
                timeout_secs: Some(5),
            },
            body: String::new(),
            source_path: None,
        };

        let result = executor
            .execute(&skill, &json!({"message": "hello"}))
            .await
            .unwrap();
        assert!(result.content.trim() == "hello");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn execute_failing_command() {
        let executor = ShellSkillExecutor::new(mock_fail(1));
        let skill = SkillDefinition {
            name: "fail_test".into(),
            description: "Test failure".into(),
            input_schema: json!({"type": "object"}),
            execution: super::super::SkillExecution::Shell {
                command_template: "exit 1".into(),
                timeout_secs: Some(5),
            },
            body: String::new(),
            source_path: None,
        };

        let result = executor.execute(&skill, &json!({})).await.unwrap();
        assert!(result.is_error);
    }

    #[cfg(feature = "native")]
    #[tokio::test]
    async fn native_execute_echo() {
        let executor = ShellSkillExecutor::native();
        let skill = SkillDefinition {
            name: "echo_test".into(),
            description: "Test".into(),
            input_schema: json!({"type": "object"}),
            execution: super::super::SkillExecution::Shell {
                command_template: "echo '{{msg}}'".into(),
                timeout_secs: Some(5),
            },
            body: String::new(),
            source_path: None,
        };

        let result = executor
            .execute(&skill, &json!({"msg": "world"}))
            .await
            .unwrap();
        assert_eq!(result.content.trim(), "world");
        assert!(!result.is_error);
    }
}
