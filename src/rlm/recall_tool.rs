//! RecallTool — lets the agent search its own conversation history via RLM REPL
//!
//! When the agent's context window doesn't contain what it needs, it can invoke
//! this tool to run an RLM query over its full (externalized) conversation history.
//! The tool receives a cached snapshot of the serialized document and runs the
//! RLM REPL loop to extract the answer.

use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::provider::Provider;
use crate::tool::{Tool, ToolOutput};
use crate::types::*;

use super::engine::{RlmConfig, RlmEngine};

/// Shared handle to the cached serialized context document.
///
/// The agent loop updates this incrementally each turn. The recall tool reads
/// a snapshot when invoked.
pub type SharedContextDoc = Arc<std::sync::RwLock<String>>;

/// Tool that lets the agent search its own conversation history.
///
/// Backed by the RLM engine — runs a REPL loop over the serialized context
/// document to answer the agent's query about its past work.
pub struct RecallTool {
    provider: Arc<dyn Provider>,
    model: ModelInfo,
    /// Optional cheaper model for sub-queries within the RLM REPL
    sub_model: Option<ModelInfo>,
    auth: AuthProfile,
    context_doc: SharedContextDoc,
}

impl RecallTool {
    pub fn new(
        provider: Arc<dyn Provider>,
        model: ModelInfo,
        auth: AuthProfile,
        context_doc: SharedContextDoc,
    ) -> Self {
        Self {
            provider,
            model,
            sub_model: None,
            auth,
            context_doc,
        }
    }

    /// Set a cheaper model for sub-queries within recall's RLM REPL.
    pub fn with_sub_model(mut self, sub_model: ModelInfo) -> Self {
        self.sub_model = Some(sub_model);
        self
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl Tool for RecallTool {
    fn name(&self) -> &str {
        "recall"
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall".into(),
            description: "Search your conversation history for past work, decisions, tool results, or any information from earlier turns that is no longer in your context window. Uses an RLM REPL to navigate the full externalized history.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in your conversation history. Be specific: 'What was the test output for the auth module?' is better than 'test results'."
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
        let query = arguments
            .get("query")
            .and_then(|v| v.as_str())
            .unwrap_or("Summarize the conversation history");

        // Read snapshot of the cached context document
        let doc = {
            let guard = self.context_doc.read().unwrap();
            guard.clone()
        };

        if doc.trim().is_empty() {
            return Ok(ToolOutput::success(
                "No conversation history available yet.".to_string(),
            ));
        }

        let mut config = RlmConfig::new(self.model.clone());
        if let Some(ref sub) = self.sub_model {
            config.sub_model = Some(sub.clone());
        }
        let engine = RlmEngine::new(self.provider.clone(), config, self.auth.clone());

        match engine.completion(doc, Some(query)).await {
            Ok(result) => {
                let summary = format!(
                    "{}\n\n[Recall: {} iterations, {} LLM calls]",
                    result.answer, result.iterations.len(), result.total_llm_calls
                );
                Ok(ToolOutput::success(summary))
            }
            Err(e) => Ok(ToolOutput::error(format!(
                "Recall failed: {e}"
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_tool_definition() {
        let doc = Arc::new(std::sync::RwLock::new(String::new()));
        let model = ModelInfo {
            id: "test".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: false,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        };
        let auth = AuthProfile::new(ProviderKind::Anthropic, "test-key");

        // Use a mock provider — just testing definition shape
        struct MockProvider;
        #[async_trait::async_trait]
        impl Provider for MockProvider {
            fn kind(&self) -> ProviderKind {
                ProviderKind::Custom("mock".into())
            }
            async fn stream(
                &self,
                _: &[Message],
                _: &str,
                _: &[ToolDefinition],
                _: &ModelInfo,
                _: &AuthProfile,
                _: mpsc::UnboundedSender<StreamDelta>,
            ) -> SoulResult<Message> {
                Ok(Message::assistant("mock"))
            }
            async fn count_tokens(
                &self,
                _: &[Message],
                _: &str,
                _: &[ToolDefinition],
                _: &ModelInfo,
                _: &AuthProfile,
            ) -> SoulResult<usize> {
                Ok(0)
            }
            async fn probe(
                &self,
                _: &ModelInfo,
                _: &AuthProfile,
            ) -> SoulResult<crate::provider::ProbeResult> {
                Ok(crate::provider::ProbeResult {
                    healthy: true,
                    rate_limit_remaining: Some(1.0),
                    rate_limit_utilization: Some(0.0),
                })
            }
        }

        let tool = RecallTool::new(Arc::new(MockProvider), model, auth, doc);
        let def = tool.definition();
        assert_eq!(def.name, "recall");
        assert!(def.description.contains("conversation history"));
        assert!(def.input_schema.get("properties").is_some());
    }
}
