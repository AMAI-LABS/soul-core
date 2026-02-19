//! LLM executor — delegates tool calls to an LLM.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::error::SoulResult;
use crate::tool::ToolOutput;
use crate::types::ToolDefinition;

use super::ToolExecutor;

/// Request to an LLM for tool execution.
#[derive(Debug, Clone)]
pub struct LlmExecutorRequest {
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub user_message: String,
}

/// Function type for LLM delegation.
pub type LlmFn = Arc<
    dyn Fn(LlmExecutorRequest) -> Pin<Box<dyn Future<Output = SoulResult<String>> + Send>>
        + Send
        + Sync,
>;

/// Executes tools by delegating to an LLM.
pub struct LlmExecutor {
    llm_fn: LlmFn,
    default_model: Option<String>,
    default_system_prompt: Option<String>,
}

impl LlmExecutor {
    pub fn new(llm_fn: LlmFn) -> Self {
        Self {
            llm_fn,
            default_model: None,
            default_system_prompt: None,
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.default_system_prompt = Some(prompt.into());
        self
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl ToolExecutor for LlmExecutor {
    async fn execute(
        &self,
        definition: &ToolDefinition,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let user_message = if let Some(text) = arguments.get("text").and_then(|v| v.as_str()) {
            text.to_string()
        } else {
            format!(
                "Execute tool '{}' with arguments: {}",
                definition.name,
                serde_json::to_string_pretty(&arguments).unwrap_or_default()
            )
        };

        let request = LlmExecutorRequest {
            model: self.default_model.clone(),
            system_prompt: self.default_system_prompt.clone(),
            user_message,
        };

        let result = (self.llm_fn)(request).await?;
        Ok(ToolOutput::success(result))
    }

    fn executor_name(&self) -> &str {
        "llm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_def() -> ToolDefinition {
        ToolDefinition {
            name: "summarize".into(),
            description: "Summarize text".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn mock_llm_fn() -> LlmFn {
        Arc::new(|req: LlmExecutorRequest| {
            Box::pin(async move { Ok(format!("LLM response to: {}", req.user_message)) })
        })
    }

    #[tokio::test]
    async fn executes_with_text_argument() {
        let executor = LlmExecutor::new(mock_llm_fn());
        let result = executor
            .execute(
                &test_def(),
                "c1",
                json!({"text": "Please summarize this"}),
                None,
            )
            .await
            .unwrap();
        assert!(result.content.contains("Please summarize this"));
    }

    #[tokio::test]
    async fn executes_without_text_argument() {
        let executor = LlmExecutor::new(mock_llm_fn());
        let result = executor
            .execute(&test_def(), "c1", json!({"data": [1, 2, 3]}), None)
            .await
            .unwrap();
        assert!(result.content.contains("summarize"));
    }

    #[tokio::test]
    async fn passes_model_and_system_prompt() {
        let llm_fn: LlmFn = Arc::new(|req: LlmExecutorRequest| {
            Box::pin(async move {
                Ok(format!(
                    "model={}, sys={}",
                    req.model.unwrap_or_default(),
                    req.system_prompt.unwrap_or_default()
                ))
            })
        });

        let executor = LlmExecutor::new(llm_fn)
            .with_model("haiku")
            .with_system_prompt("Be concise");
        let result = executor
            .execute(&test_def(), "c1", json!({"text": "test"}), None)
            .await
            .unwrap();
        assert!(result.content.contains("model=haiku"));
        assert!(result.content.contains("sys=Be concise"));
    }

    #[test]
    fn executor_name() {
        let executor = LlmExecutor::new(mock_llm_fn());
        assert_eq!(executor.executor_name(), "llm");
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LlmExecutor>();
    }
}
