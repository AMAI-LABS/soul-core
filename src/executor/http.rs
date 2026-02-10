//! HTTP executor — executes tools as HTTP requests.

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::tool::ToolOutput;
use crate::types::ToolDefinition;

use super::ToolExecutor;

/// Executes tools by making HTTP requests.
///
/// Arguments should contain `url` and optionally `method`, `headers`, `body`.
pub struct HttpExecutor {
    client: reqwest::Client,
}

impl HttpExecutor {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }

    pub fn with_client(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl Default for HttpExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolExecutor for HttpExecutor {
    async fn execute(
        &self,
        definition: &ToolDefinition,
        _call_id: &str,
        arguments: serde_json::Value,
        _partial_tx: Option<mpsc::UnboundedSender<String>>,
    ) -> SoulResult<ToolOutput> {
        let url = arguments
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| SoulError::ToolExecution {
                tool_name: definition.name.clone(),
                message: "Missing 'url' argument".into(),
            })?;

        let method = arguments
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET");

        let builder = match method.to_uppercase().as_str() {
            "GET" => self.client.get(url),
            "POST" => self.client.post(url),
            "PUT" => self.client.put(url),
            "DELETE" => self.client.delete(url),
            "PATCH" => self.client.patch(url),
            other => {
                return Err(SoulError::ToolExecution {
                    tool_name: definition.name.clone(),
                    message: format!("Unsupported HTTP method: {other}"),
                });
            }
        };

        let builder = if let Some(body) = arguments.get("body") {
            builder.json(body)
        } else {
            builder
        };

        let response = builder.send().await.map_err(|e| SoulError::ToolExecution {
            tool_name: definition.name.clone(),
            message: format!("HTTP request failed: {e}"),
        })?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| SoulError::ToolExecution {
                tool_name: definition.name.clone(),
                message: format!("Failed to read response body: {e}"),
            })?;

        if status.is_success() {
            Ok(ToolOutput::success(body))
        } else {
            Ok(ToolOutput::error(format!(
                "HTTP {}: {}",
                status.as_u16(),
                body
            )))
        }
    }

    fn executor_name(&self) -> &str {
        "http"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_def() -> ToolDefinition {
        ToolDefinition {
            name: "http_test".into(),
            description: "Test".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    #[tokio::test]
    async fn missing_url_errors() {
        let executor = HttpExecutor::new();
        let result = executor
            .execute(&test_def(), "c1", json!({"method": "GET"}), None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn unsupported_method_errors() {
        let executor = HttpExecutor::new();
        let result = executor
            .execute(
                &test_def(),
                "c1",
                json!({"url": "http://localhost", "method": "FOOBAR"}),
                None,
            )
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn executor_name() {
        let executor = HttpExecutor::new();
        assert_eq!(executor.executor_name(), "http");
    }

    // Note: actual HTTP tests would require wiremock,
    // which is in dev-dependencies. Integration tests handle this.

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HttpExecutor>();
    }

    #[test]
    fn with_custom_client() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let executor = HttpExecutor::with_client(client);
        assert_eq!(executor.executor_name(), "http");
    }
}
