/// Policy registry — stores distilled policies and exposes them as Tool implementations.
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use serde_json::{json, Value};
use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::tool::{Tool, ToolOutput};
use crate::types::ToolDefinition;

/// A distilled policy — a shell script that replaces LLM inference for known subtasks.
///
/// The script receives arguments as env vars: AMAI_ARG_<NAME>=<value>.
/// Stdout → output. This is the "code-as-policy" variant from the AutoHarness paper.
#[derive(Debug, Clone)]
pub struct DistilledPolicy {
    /// The tool this policy replaces.
    pub tool_name: String,
    /// Argument keys this policy was distilled from.
    pub arg_keys: Vec<String>,
    /// Shell script that produces the output.
    pub script: String,
    /// Human-readable description of what was distilled.
    pub description: String,
    /// How many observations led to this distillation.
    pub observation_count: usize,
}

impl DistilledPolicy {
    pub fn new(
        tool_name: impl Into<String>,
        arg_keys: Vec<String>,
        script: impl Into<String>,
        description: impl Into<String>,
        observation_count: usize,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            arg_keys,
            script: script.into(),
            description: description.into(),
            observation_count,
        }
    }

    /// Execute the policy with the given arguments.
    /// Returns (output, is_error).
    pub async fn execute_script(&self, arguments: &Value) -> (String, bool) {
        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c").arg(&self.script);

        if let Some(obj) = arguments.as_object() {
            for (k, v) in obj {
                let env_key = format!("AMAI_ARG_{}", k.to_uppercase().replace('-', "_"));
                let env_val = match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                cmd.env(env_key, env_val);
            }
        }

        match cmd.output().await {
            Ok(out) => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                let stderr = String::from_utf8_lossy(&out.stderr).to_string();
                if out.status.success() {
                    let content = if stdout.is_empty() { stderr } else { stdout };
                    (content, false)
                } else {
                    (stderr, true)
                }
            }
            Err(e) => (format!("Policy execution error: {e}"), true),
        }
    }
}

/// A `Tool` implementation backed by a `DistilledPolicy`.
///
/// When registered in a `ToolRegistry` (via `DynamicToolHandle`), this tool
/// executes the policy script instead of calling the LLM — zero inference cost.
pub struct PolicyTool {
    policy: DistilledPolicy,
    definition: ToolDefinition,
}

impl PolicyTool {
    pub fn new(policy: DistilledPolicy) -> Self {
        // Build schema from arg_keys: each key is a required string
        let properties: serde_json::Map<String, Value> = policy
            .arg_keys
            .iter()
            .map(|k| (k.clone(), json!({"type": "string"})))
            .collect();

        let definition = ToolDefinition {
            name: format!("policy_{}", policy.tool_name),
            description: format!(
                "[DISTILLED POLICY] {} — Runs without LLM inference. Observation count: {}.",
                policy.description, policy.observation_count
            ),
            input_schema: json!({
                "type": "object",
                "properties": properties,
                "required": policy.arg_keys
            }),
        };

        Self { policy, definition }
    }
}

#[async_trait]
impl Tool for PolicyTool {
    fn name(&self) -> &str {
        &self.definition.name
    }

    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    async fn execute(
        &self,
        _call_id: &str,
        arguments: Value,
        partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let (content, is_error) = self.policy.execute_script(&arguments).await;

        if let Some(tx) = &partial_tx {
            if !content.is_empty() && !is_error {
                let _ = tx.send(content.clone());
            }
        }

        Ok(if is_error {
            ToolOutput::error(content)
        } else {
            ToolOutput::success(content)
        })
    }
}

/// Thread-safe registry of distilled policies.
#[derive(Clone)]
pub struct PolicyRegistry {
    /// Keyed by fingerprint: `tool_name:sorted_arg_keys`.
    inner: Arc<RwLock<HashMap<String, DistilledPolicy>>>,
}

impl PolicyRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a policy under a fingerprint.
    pub fn register(&self, fingerprint: impl Into<String>, policy: DistilledPolicy) {
        self.inner.write().unwrap().insert(fingerprint.into(), policy);
    }

    /// Look up a policy by fingerprint.
    pub fn get_by_fingerprint(&self, fingerprint: &str) -> Option<DistilledPolicy> {
        self.inner.read().unwrap().get(fingerprint).cloned()
    }

    /// Get all policies as `PolicyTool` instances (for bulk registration).
    pub fn all_as_tools(&self) -> Vec<PolicyTool> {
        self.inner
            .read()
            .unwrap()
            .values()
            .cloned()
            .map(PolicyTool::new)
            .collect()
    }

    /// Number of registered policies.
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().is_empty()
    }
}

impl Default for PolicyRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn distilled_policy_new() {
        let p = DistilledPolicy::new(
            "ls",
            vec!["path".into()],
            "printf '%s' 'file.txt'",
            "Static: always returns file.txt",
            5,
        );
        assert_eq!(p.tool_name, "ls");
        assert_eq!(p.arg_keys, vec!["path"]);
        assert_eq!(p.observation_count, 5);
    }

    #[tokio::test]
    async fn distilled_policy_execute_static() {
        let p = DistilledPolicy::new(
            "ls",
            vec![],
            "printf '%s' 'README.md'",
            "Static output",
            3,
        );
        let (content, is_error) = p.execute_script(&json!({})).await;
        assert!(!is_error, "Should succeed");
        assert!(content.contains("README.md"), "Got: {content}");
    }

    #[tokio::test]
    async fn distilled_policy_execute_with_arg() {
        let p = DistilledPolicy::new(
            "echo",
            vec!["text".into()],
            "printf '%s' \"$AMAI_ARG_TEXT\"",
            "Identity: output = input",
            5,
        );
        let (content, is_error) = p.execute_script(&json!({"text": "hello world"})).await;
        assert!(!is_error);
        assert_eq!(content.trim(), "hello world");
    }

    #[tokio::test]
    async fn distilled_policy_execute_failure() {
        let p = DistilledPolicy::new("bad", vec![], "exit 1", "Always fails", 5);
        let (_, is_error) = p.execute_script(&json!({})).await;
        assert!(is_error);
    }

    #[test]
    fn policy_tool_name_prefixed() {
        let p = DistilledPolicy::new("ls", vec![], "printf ''", "desc", 3);
        let tool = PolicyTool::new(p);
        assert_eq!(tool.name(), "policy_ls");
    }

    #[test]
    fn policy_tool_definition_has_schema() {
        let p = DistilledPolicy::new("read", vec!["path".into()], "cat", "read file", 5);
        let tool = PolicyTool::new(p);
        let def = tool.definition();
        assert!(def.input_schema["properties"]["path"].is_object());
        assert!(def.description.contains("DISTILLED POLICY"));
    }

    #[tokio::test]
    async fn policy_tool_execute_returns_output() {
        let p = DistilledPolicy::new(
            "echo",
            vec!["text".into()],
            "printf '%s' \"$AMAI_ARG_TEXT\"",
            "Identity",
            5,
        );
        let tool = PolicyTool::new(p);
        let output = tool.execute("c1", json!({"text": "test_value"}), None).await.unwrap();
        assert!(!output.is_error);
        assert!(output.content.contains("test_value"), "Got: {}", output.content);
    }

    #[test]
    fn policy_registry_register_and_retrieve() {
        let reg = PolicyRegistry::new();
        let policy = DistilledPolicy::new("ls", vec![], "printf ''", "desc", 3);
        reg.register("ls:", policy);

        assert_eq!(reg.len(), 1);
        let retrieved = reg.get_by_fingerprint("ls:");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().tool_name, "ls");
    }

    #[test]
    fn policy_registry_all_as_tools() {
        let reg = PolicyRegistry::new();
        reg.register("ls:", DistilledPolicy::new("ls", vec![], "printf ''", "d", 3));
        reg.register("read:path", DistilledPolicy::new("read", vec!["path".into()], "cat", "d", 5));

        let tools = reg.all_as_tools();
        assert_eq!(tools.len(), 2);
    }

    #[test]
    fn policy_registry_missing_returns_none() {
        let reg = PolicyRegistry::new();
        assert!(reg.get_by_fingerprint("nonexistent").is_none());
    }

    #[tokio::test]
    async fn policy_tool_is_a_tool_trait_object() {
        let p = DistilledPolicy::new("ls", vec![], "printf 'ok'", "desc", 3);
        let tool: Box<dyn Tool> = Box::new(PolicyTool::new(p));
        assert_eq!(tool.name(), "policy_ls");
        let output = tool.execute("c1", json!({}), None).await.unwrap();
        assert!(!output.is_error);
    }
}
