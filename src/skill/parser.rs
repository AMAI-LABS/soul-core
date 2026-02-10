//! Parser for skill files with YAML frontmatter.
//!
//! Skill format:
//! ```text
//! ---
//! name: my_skill
//! description: Does something
//! input_schema: { type: object, ... }
//! execution:
//!   type: shell
//!   command_template: "echo {{message}}"
//! ---
//! Body text (description, documentation, etc.)
//! ```

use super::{SkillDefinition, SkillExecution};
use crate::error::{SoulError, SoulResult};

/// Parse a skill file content into a SkillDefinition.
pub fn parse_skill(content: &str) -> SoulResult<SkillDefinition> {
    let (frontmatter, body) =
        extract_frontmatter(content).ok_or_else(|| SoulError::SkillParse {
            message: "Missing YAML frontmatter (expected --- delimiters)".into(),
        })?;

    let raw: RawSkillFrontmatter =
        serde_yaml::from_str(frontmatter).map_err(|e| SoulError::SkillParse {
            message: format!("Invalid YAML frontmatter: {e}"),
        })?;

    let execution = parse_execution(&raw.execution)?;

    Ok(SkillDefinition {
        name: raw.name,
        description: raw.description,
        input_schema: raw
            .input_schema
            .unwrap_or(serde_json::json!({"type": "object"})),
        execution,
        body: body.to_string(),
        source_path: None,
    })
}

/// Extract YAML frontmatter between --- delimiters.
/// Returns (frontmatter, body) or None if delimiters aren't found.
fn extract_frontmatter(content: &str) -> Option<(&str, &str)> {
    let trimmed = content.trim_start();
    if !trimmed.starts_with("---") {
        return None;
    }

    // Find opening delimiter end
    let after_first = &trimmed[3..];
    let after_first = after_first.trim_start_matches(['\r', '\n']);

    // Find closing delimiter
    let end_pos = after_first.find("\n---")?;
    let frontmatter = &after_first[..end_pos];
    let body = &after_first[end_pos + 4..];
    let body = body.trim_start_matches(['\r', '\n']);

    Some((frontmatter, body))
}

#[derive(serde::Deserialize)]
struct RawSkillFrontmatter {
    name: String,
    description: String,
    #[serde(default)]
    input_schema: Option<serde_json::Value>,
    execution: RawExecution,
}

#[derive(serde::Deserialize)]
struct RawExecution {
    #[serde(rename = "type")]
    exec_type: String,
    #[serde(default)]
    command_template: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    server: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    executor_name: Option<String>,
}

fn parse_execution(raw: &RawExecution) -> SoulResult<SkillExecution> {
    match raw.exec_type.as_str() {
        "shell" => {
            let command_template =
                raw.command_template
                    .clone()
                    .ok_or_else(|| SoulError::SkillParse {
                        message: "Shell execution requires 'command_template'".into(),
                    })?;
            Ok(SkillExecution::Shell {
                command_template,
                timeout_secs: raw.timeout_secs,
            })
        }
        "mcp_tool" => {
            let server = raw.server.clone().ok_or_else(|| SoulError::SkillParse {
                message: "MCP tool execution requires 'server'".into(),
            })?;
            let tool_name = raw.tool_name.clone().ok_or_else(|| SoulError::SkillParse {
                message: "MCP tool execution requires 'tool_name'".into(),
            })?;
            Ok(SkillExecution::McpTool { server, tool_name })
        }
        "llm_delegate" => Ok(SkillExecution::LlmDelegate {
            model: raw.model.clone(),
            system_prompt: raw.system_prompt.clone(),
        }),
        "custom" => {
            let executor_name = raw
                .executor_name
                .clone()
                .ok_or_else(|| SoulError::SkillParse {
                    message: "Custom execution requires 'executor_name'".into(),
                })?;
            Ok(SkillExecution::Custom { executor_name })
        }
        other => Err(SoulError::SkillParse {
            message: format!("Unknown execution type: '{other}'"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHELL_SKILL: &str = r#"---
name: search_codebase
description: Search the codebase for a pattern
input_schema:
  type: object
  properties:
    pattern:
      type: string
  required:
    - pattern
execution:
  type: shell
  command_template: "rg '{{pattern}}' --json"
  timeout_secs: 30
---
Search the codebase using ripgrep. Returns JSON output.
"#;

    const MCP_SKILL: &str = r#"---
name: query_docs
description: Query documentation
execution:
  type: mcp_tool
  server: context7
  tool_name: query-docs
---
Queries documentation via MCP.
"#;

    const LLM_SKILL: &str = r#"---
name: summarize
description: Summarize text
execution:
  type: llm_delegate
  model: claude-haiku-4-5-20251001
  system_prompt: You are a summarizer.
---
Summarize the input text.
"#;

    const CUSTOM_SKILL: &str = r#"---
name: deploy
description: Deploy to production
execution:
  type: custom
  executor_name: deploy_executor
---
Deploys the current build.
"#;

    #[test]
    fn parse_shell_skill() {
        let skill = parse_skill(SHELL_SKILL).unwrap();
        assert_eq!(skill.name, "search_codebase");
        assert_eq!(skill.description, "Search the codebase for a pattern");
        assert!(skill.body.contains("ripgrep"));
        match &skill.execution {
            SkillExecution::Shell {
                command_template,
                timeout_secs,
            } => {
                assert!(command_template.contains("{{pattern}}"));
                assert_eq!(*timeout_secs, Some(30));
            }
            _ => panic!("Expected Shell execution"),
        }
    }

    #[test]
    fn parse_mcp_skill() {
        let skill = parse_skill(MCP_SKILL).unwrap();
        assert_eq!(skill.name, "query_docs");
        match &skill.execution {
            SkillExecution::McpTool { server, tool_name } => {
                assert_eq!(server, "context7");
                assert_eq!(tool_name, "query-docs");
            }
            _ => panic!("Expected McpTool execution"),
        }
    }

    #[test]
    fn parse_llm_skill() {
        let skill = parse_skill(LLM_SKILL).unwrap();
        assert_eq!(skill.name, "summarize");
        match &skill.execution {
            SkillExecution::LlmDelegate {
                model,
                system_prompt,
            } => {
                assert_eq!(model.as_deref(), Some("claude-haiku-4-5-20251001"));
                assert!(system_prompt.as_ref().unwrap().contains("summarizer"));
            }
            _ => panic!("Expected LlmDelegate execution"),
        }
    }

    #[test]
    fn parse_custom_skill() {
        let skill = parse_skill(CUSTOM_SKILL).unwrap();
        match &skill.execution {
            SkillExecution::Custom { executor_name } => {
                assert_eq!(executor_name, "deploy_executor");
            }
            _ => panic!("Expected Custom execution"),
        }
    }

    #[test]
    fn parse_default_input_schema() {
        let skill = parse_skill(MCP_SKILL).unwrap();
        assert_eq!(skill.input_schema, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn parse_with_input_schema() {
        let skill = parse_skill(SHELL_SKILL).unwrap();
        assert!(skill.input_schema.get("properties").is_some());
    }

    #[test]
    fn missing_frontmatter_errors() {
        let result = parse_skill("no frontmatter here");
        assert!(result.is_err());
    }

    #[test]
    fn malformed_yaml_errors() {
        let content = "---\n[invalid yaml\n---\nbody";
        let result = parse_skill(content);
        assert!(result.is_err());
    }

    #[test]
    fn missing_required_field_errors() {
        let content = "---\nname: test\n---\nbody";
        let result = parse_skill(content);
        assert!(result.is_err());
    }

    #[test]
    fn shell_missing_template_errors() {
        let content = r#"---
name: bad
description: Missing template
execution:
  type: shell
---
body"#;
        let result = parse_skill(content);
        assert!(result.is_err());
    }

    #[test]
    fn unknown_execution_type_errors() {
        let content = r#"---
name: bad
description: Unknown type
execution:
  type: quantum_compute
---
body"#;
        let result = parse_skill(content);
        assert!(result.is_err());
    }

    #[test]
    fn extract_frontmatter_works() {
        let (fm, body) = extract_frontmatter("---\nkey: value\n---\nbody text").unwrap();
        assert_eq!(fm, "key: value");
        assert_eq!(body, "body text");
    }

    #[test]
    fn extract_frontmatter_no_delimiters() {
        assert!(extract_frontmatter("no delimiters").is_none());
    }

    #[test]
    fn empty_body() {
        let content = "---\nname: test\ndescription: test\nexecution:\n  type: llm_delegate\n---\n";
        let skill = parse_skill(content).unwrap();
        assert!(skill.body.is_empty());
    }
}
