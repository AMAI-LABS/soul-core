//! Skills system — config-driven tool definitions loaded from files.
//!
//! Skills are defined in `.skill` or `.md` files with YAML frontmatter
//! specifying the execution strategy (shell, MCP tool, LLM delegate, custom).

pub mod bridge;
pub mod executor;
pub mod loader;
pub mod parser;

pub use bridge::SkillToolBridge;
pub use executor::{ShellSkillExecutor, SkillExecutor};
pub use loader::SkillLoader;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// A parsed skill definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
    pub execution: SkillExecution,
    pub body: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<PathBuf>,
}

/// How a skill is executed.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SkillExecution {
    Shell {
        command_template: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_secs: Option<u64>,
    },
    McpTool {
        server: String,
        tool_name: String,
    },
    LlmDelegate {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        system_prompt: Option<String>,
    },
    Custom {
        executor_name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn skill_definition_serializes() {
        let skill = SkillDefinition {
            name: "test".into(),
            description: "A test".into(),
            input_schema: json!({"type": "object"}),
            execution: SkillExecution::Shell {
                command_template: "echo test".into(),
                timeout_secs: Some(30),
            },
            body: "body".into(),
            source_path: None,
        };

        let json = serde_json::to_string(&skill).unwrap();
        let deserialized: SkillDefinition = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "test");
    }

    #[test]
    fn skill_execution_variants_serialize() {
        let shell = SkillExecution::Shell {
            command_template: "echo".into(),
            timeout_secs: None,
        };
        let json = serde_json::to_string(&shell).unwrap();
        assert!(json.contains("\"type\":\"shell\""));

        let mcp = SkillExecution::McpTool {
            server: "ctx7".into(),
            tool_name: "query".into(),
        };
        let json = serde_json::to_string(&mcp).unwrap();
        assert!(json.contains("\"type\":\"mcp_tool\""));

        let llm = SkillExecution::LlmDelegate {
            model: None,
            system_prompt: None,
        };
        let json = serde_json::to_string(&llm).unwrap();
        assert!(json.contains("\"type\":\"llm_delegate\""));

        let custom = SkillExecution::Custom {
            executor_name: "my_exec".into(),
        };
        let json = serde_json::to_string(&custom).unwrap();
        assert!(json.contains("\"type\":\"custom\""));
    }
}
