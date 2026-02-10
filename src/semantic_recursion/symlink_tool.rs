//! Symlink Resolution Tool — gives LLMs the ability to expand symlink references.
//!
//! When context is symlinked, the LLM sees compact references like `[A3F2B1]: summary`.
//! This tool lets the LLM call `resolve_symlink` to get the full original content
//! for any hash it needs to examine in detail.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::{mpsc, RwLock};

use crate::error::SoulResult;
use crate::tool::{Tool, ToolOutput};
use crate::types::ToolDefinition;

use super::symlink::SymlinkStore;

/// Tool that resolves symlink hashes back to full content.
/// Shared ownership via `Arc<RwLock<SymlinkStore>>` so the agent loop
/// can symlink messages while the tool resolves them concurrently.
pub struct ResolveSymlinkTool {
    store: Arc<RwLock<SymlinkStore>>,
}

impl ResolveSymlinkTool {
    pub fn new(store: Arc<RwLock<SymlinkStore>>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for ResolveSymlinkTool {
    fn name(&self) -> &str {
        "resolve_symlink"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "resolve_symlink".into(),
            description: "Expand a symlink hash reference (e.g. A3F2B1) back to its full original content. Use this when you see [XXXXXX] references in the conversation and need the complete text. You can resolve multiple hashes at once.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "hashes": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "One or more 6-character hex hashes to resolve (e.g. [\"A3F2B1\", \"00CAFE\"])"
                    }
                },
                "required": ["hashes"]
            }),
        }
    }

    async fn execute(
        &self,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let hashes: Vec<String> = match arguments.get("hashes") {
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect(),
            _ => {
                return Ok(ToolOutput::error(
                    "Invalid arguments: expected {\"hashes\": [\"HASH1\", \"HASH2\"]}",
                ));
            }
        };

        if hashes.is_empty() {
            return Ok(ToolOutput::error("No hashes provided"));
        }

        let store = self.store.read().await;
        let mut results = Vec::new();
        let mut not_found = Vec::new();

        for hash in &hashes {
            match store.resolve(hash) {
                Some(result) => {
                    results.push(format!(
                        "--- [{}] ({}) ---\n{}",
                        result.hash, result.summary, result.original
                    ));
                }
                None => {
                    not_found.push(hash.clone());
                }
            }
        }

        let mut output = String::new();

        if !results.is_empty() {
            output.push_str(&results.join("\n\n"));
        }

        if !not_found.is_empty() {
            if !output.is_empty() {
                output.push_str("\n\n");
            }
            output.push_str(&format!("Not found: {}", not_found.join(", ")));
        }

        let metadata = json!({
            "resolved": results.len(),
            "not_found": not_found.len(),
        });

        Ok(ToolOutput::success(output).with_metadata(metadata))
    }
}

/// Tool that searches symlinks by keyword.
pub struct SearchSymlinksTool {
    store: Arc<RwLock<SymlinkStore>>,
}

impl SearchSymlinksTool {
    pub fn new(store: Arc<RwLock<SymlinkStore>>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for SearchSymlinksTool {
    fn name(&self) -> &str {
        "search_symlinks"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "search_symlinks".into(),
            description: "Search through symlinked context by keyword. Returns matching symlink hashes and their summaries. Use this when you need to find relevant context but don't have the exact hash.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword or phrase to search for in symlinked content"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn execute(
        &self,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let query = match arguments.get("query").and_then(|v| v.as_str()) {
            Some(q) => q,
            None => {
                return Ok(ToolOutput::error(
                    "Invalid arguments: expected {\"query\": \"search term\"}",
                ));
            }
        };

        let store = self.store.read().await;
        let results = store.search(query);

        if results.is_empty() {
            return Ok(ToolOutput::success(format!(
                "No symlinks matching \"{query}\""
            )));
        }

        let output: Vec<String> = results
            .iter()
            .map(|link| format!("[{}]: {}", link.hash, link.summary))
            .collect();

        let metadata = json!({ "count": results.len() });

        Ok(ToolOutput::success(format!(
            "Found {} matching symlinks:\n{}",
            results.len(),
            output.join("\n")
        ))
        .with_metadata(metadata))
    }
}

/// Tool that lists all active symlinks (for LLM orientation).
pub struct ListSymlinksTool {
    store: Arc<RwLock<SymlinkStore>>,
}

impl ListSymlinksTool {
    pub fn new(store: Arc<RwLock<SymlinkStore>>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl Tool for ListSymlinksTool {
    fn name(&self) -> &str {
        "list_symlinks"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_symlinks".into(),
            description: "List all active symlink references with their summaries. Use this to see what context has been symlinked and what hashes are available.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    async fn execute(
        &self,
        _call_id: &str,
        _arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let store = self.store.read().await;
        let symlinks = store.all_symlinks();

        if symlinks.is_empty() {
            return Ok(ToolOutput::success("No symlinks active"));
        }

        let mut lines: Vec<String> = symlinks
            .iter()
            .map(|link| {
                format!(
                    "[{}]: {} ({} tokens → {} tokens, saved {})",
                    link.hash,
                    link.summary,
                    link.original_tokens,
                    link.symlink_tokens,
                    link.original_tokens.saturating_sub(link.symlink_tokens),
                )
            })
            .collect();
        lines.sort(); // deterministic order

        let total_saved = store.total_tokens_saved();
        lines.push(format!(
            "\nTotal: {} symlinks, {} tokens saved",
            store.len(),
            total_saved
        ));

        Ok(ToolOutput::success(lines.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_store() -> Arc<RwLock<SymlinkStore>> {
        let mut store = SymlinkStore::new();
        store.create(
            0,
            "Full content about Rust async programming with tokio runtime and futures",
            "Rust async discussion",
        );
        store.create(
            1,
            "Python web development with Django and Flask frameworks",
            "Python web dev",
        );
        store.create(
            2,
            "Database optimization techniques for PostgreSQL",
            "DB optimization",
        );
        Arc::new(RwLock::new(store))
    }

    #[tokio::test]
    async fn resolve_single_hash() {
        let store = make_store();
        let tool = ResolveSymlinkTool::new(store.clone());

        let s = store.read().await;
        let hash = s.hash_for_node(0).unwrap().to_string();
        drop(s);

        let result = tool
            .execute("c1", json!({"hashes": [hash]}), None)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Rust async programming"));
        assert!(result.content.contains("tokio"));
    }

    #[tokio::test]
    async fn resolve_multiple_hashes() {
        let store = make_store();
        let tool = ResolveSymlinkTool::new(store.clone());

        let s = store.read().await;
        let h0 = s.hash_for_node(0).unwrap().to_string();
        let h1 = s.hash_for_node(1).unwrap().to_string();
        drop(s);

        let result = tool
            .execute("c1", json!({"hashes": [h0, h1]}), None)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Rust async"));
        assert!(result.content.contains("Django"));
        assert_eq!(result.metadata["resolved"], 2);
    }

    #[tokio::test]
    async fn resolve_not_found() {
        let store = make_store();
        let tool = ResolveSymlinkTool::new(store);

        let result = tool
            .execute("c1", json!({"hashes": ["ZZZZZZ"]}), None)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Not found"));
    }

    #[tokio::test]
    async fn resolve_mixed_found_and_not() {
        let store = make_store();
        let tool = ResolveSymlinkTool::new(store.clone());

        let s = store.read().await;
        let h0 = s.hash_for_node(0).unwrap().to_string();
        drop(s);

        let result = tool
            .execute("c1", json!({"hashes": [h0, "BADONE"]}), None)
            .await
            .unwrap();
        assert!(result.content.contains("Rust async"));
        assert!(result.content.contains("Not found: BADONE"));
    }

    #[tokio::test]
    async fn resolve_bad_arguments() {
        let store = make_store();
        let tool = ResolveSymlinkTool::new(store);

        let result = tool
            .execute("c1", json!({"wrong": "field"}), None)
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn resolve_empty_hashes() {
        let store = make_store();
        let tool = ResolveSymlinkTool::new(store);

        let result = tool
            .execute("c1", json!({"hashes": []}), None)
            .await
            .unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn search_finds_matches() {
        let store = make_store();
        let tool = SearchSymlinksTool::new(store);

        let result = tool
            .execute("c1", json!({"query": "rust"}), None)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("1 matching"));
        assert!(result.content.contains("Rust async"));
    }

    #[tokio::test]
    async fn search_no_matches() {
        let store = make_store();
        let tool = SearchSymlinksTool::new(store);

        let result = tool
            .execute("c1", json!({"query": "nonexistent"}), None)
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No symlinks matching"));
    }

    #[tokio::test]
    async fn search_bad_arguments() {
        let store = make_store();
        let tool = SearchSymlinksTool::new(store);

        let result = tool.execute("c1", json!({}), None).await.unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn list_all_symlinks() {
        let store = make_store();
        let tool = ListSymlinksTool::new(store);

        let result = tool.execute("c1", json!({}), None).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("3 symlinks"));
        assert!(result.content.contains("Rust async"));
        assert!(result.content.contains("Python web"));
        assert!(result.content.contains("DB optimization"));
        assert!(result.content.contains("tokens saved"));
    }

    #[tokio::test]
    async fn list_empty_store() {
        let store = Arc::new(RwLock::new(SymlinkStore::new()));
        let tool = ListSymlinksTool::new(store);

        let result = tool.execute("c1", json!({}), None).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No symlinks active"));
    }

    #[test]
    fn tool_definitions_valid() {
        let store = Arc::new(RwLock::new(SymlinkStore::new()));

        let resolve = ResolveSymlinkTool::new(store.clone());
        assert_eq!(resolve.name(), "resolve_symlink");
        let def = resolve.definition();
        assert!(def.description.contains("symlink"));
        assert!(def.input_schema["properties"]["hashes"].is_object());

        let search = SearchSymlinksTool::new(store.clone());
        assert_eq!(search.name(), "search_symlinks");

        let list = ListSymlinksTool::new(store);
        assert_eq!(list.name(), "list_symlinks");
    }
}
