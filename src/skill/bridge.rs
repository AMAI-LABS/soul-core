//! Bridge from skills to the soul-core `Tool` trait.

use std::sync::Arc;

#[cfg(test)]
use async_trait::async_trait;

use crate::error::SoulResult;
use crate::tool::{Tool, ToolOutput};
use crate::types::ToolDefinition;

use super::executor::SkillExecutor;
use super::SkillDefinition;

/// Adapts a skill definition + executor into a soul-core `Tool`.
pub struct SkillToolBridge {
    skill: SkillDefinition,
    executor: Arc<dyn SkillExecutor>,
}

impl SkillToolBridge {
    pub fn new(skill: SkillDefinition, executor: Arc<dyn SkillExecutor>) -> Self {
        Self { skill, executor }
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl Tool for SkillToolBridge {
    fn name(&self) -> &str {
        &self.skill.name
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.skill.name.clone(),
            description: self.skill.description.clone(),
            input_schema: self.skill.input_schema.clone(),
        }
    }

    async fn execute(
        &self,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<tokio::sync::mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        self.executor.execute(&self.skill, &arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct MockExecutor;

    #[async_trait]
    impl SkillExecutor for MockExecutor {
        async fn execute(
            &self,
            skill: &SkillDefinition,
            arguments: &serde_json::Value,
        ) -> SoulResult<ToolOutput> {
            let msg = arguments
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("no input");
            Ok(ToolOutput::success(format!("{}: {}", skill.name, msg)))
        }
    }

    fn test_skill() -> SkillDefinition {
        SkillDefinition {
            name: "test_skill".into(),
            description: "A test skill".into(),
            input_schema: json!({"type": "object", "properties": {"input": {"type": "string"}}}),
            execution: crate::skill::SkillExecution::Shell {
                command_template: "echo test".into(),
                timeout_secs: None,
            },
            body: "Test body".into(),
            source_path: None,
        }
    }

    #[test]
    fn name_returns_skill_name() {
        let bridge = SkillToolBridge::new(test_skill(), Arc::new(MockExecutor));
        assert_eq!(bridge.name(), "test_skill");
    }

    #[test]
    fn definition_maps_correctly() {
        let bridge = SkillToolBridge::new(test_skill(), Arc::new(MockExecutor));
        let def = bridge.definition();
        assert_eq!(def.name, "test_skill");
        assert_eq!(def.description, "A test skill");
    }

    #[tokio::test]
    async fn execute_delegates_to_executor() {
        let bridge = SkillToolBridge::new(test_skill(), Arc::new(MockExecutor));
        let result = bridge
            .execute("call1", json!({"input": "hello"}), None)
            .await
            .unwrap();
        assert_eq!(result.content, "test_skill: hello");
        assert!(!result.is_error);
    }

    #[tokio::test]
    async fn execute_without_input() {
        let bridge = SkillToolBridge::new(test_skill(), Arc::new(MockExecutor));
        let result = bridge.execute("call1", json!({}), None).await.unwrap();
        assert_eq!(result.content, "test_skill: no input");
    }

    #[test]
    fn skill_tool_bridge_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SkillToolBridge>();
    }
}
