use thiserror::Error;

#[derive(Error, Debug)]
pub enum SoulError {
    #[error("Provider error: {0}")]
    Provider(String),

    #[error("Provider rate limited: {provider}, retry after {retry_after_ms}ms")]
    RateLimited {
        provider: String,
        retry_after_ms: u64,
    },

    #[error("Auth error: {0}")]
    Auth(String),

    #[error("Tool execution error: tool={tool_name}, {message}")]
    ToolExecution { tool_name: String, message: String },

    #[error("Context overflow: {used_tokens} tokens used, {max_tokens} max")]
    ContextOverflow {
        used_tokens: usize,
        max_tokens: usize,
    },

    #[error("Compaction failed: {0}")]
    CompactionFailed(String),

    #[error("Session error: {0}")]
    Session(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Agent interrupted")]
    Interrupted,

    #[error("Failover exhausted: tried {attempts} providers")]
    FailoverExhausted { attempts: usize },

    #[error("Permission denied: tool={tool_name}, {reason}")]
    PermissionDenied { tool_name: String, reason: String },

    #[error("Budget exceeded: {message}")]
    BudgetExceeded { message: String },

    #[error("MCP error: server={server}, {message}")]
    Mcp { server: String, message: String },

    #[error("JSON-RPC error: code={code}, {message}")]
    JsonRpc { code: i32, message: String },

    #[error("Skill parse error: {message}")]
    SkillParse { message: String },

    #[error("Executor not found: {name}")]
    ExecutorNotFound { name: String },

    #[error("{0}")]
    Other(#[from] anyhow::Error),
}

pub type SoulResult<T> = Result<T, SoulError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_formats() {
        let err = SoulError::Provider("connection refused".into());
        assert_eq!(err.to_string(), "Provider error: connection refused");

        let err = SoulError::RateLimited {
            provider: "anthropic".into(),
            retry_after_ms: 5000,
        };
        assert!(err.to_string().contains("5000ms"));

        let err = SoulError::ContextOverflow {
            used_tokens: 200_000,
            max_tokens: 180_000,
        };
        assert!(err.to_string().contains("200000"));

        let err = SoulError::ToolExecution {
            tool_name: "bash".into(),
            message: "command not found".into(),
        };
        assert!(err.to_string().contains("bash"));
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SoulError>();
    }

    #[test]
    fn io_error_converts() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let soul_err: SoulError = io_err.into();
        assert!(matches!(soul_err, SoulError::Io(_)));
    }

    #[test]
    fn json_error_converts() {
        let json_err = serde_json::from_str::<String>("invalid").unwrap_err();
        let soul_err: SoulError = json_err.into();
        assert!(matches!(soul_err, SoulError::Serialization(_)));
    }
}
