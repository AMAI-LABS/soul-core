use std::fmt;
use serde::{Deserialize, Serialize};

/// Intent of an LLM call — determines which provider/model gets selected.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    /// Complex reasoning, planning, architecture decisions
    Reasoning,
    /// Writing, editing, reviewing code
    CodeGeneration,
    /// Fast conversational turns, simple Q&A
    QuickChat,
    /// Subagent work: search, summarize, classify
    Subagent,
    /// Embedding generation
    Embedding,
    /// Custom intent
    Custom(String),
}

impl fmt::Display for Intent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Intent::Reasoning => write!(f, "reasoning"),
            Intent::CodeGeneration => write!(f, "code_generation"),
            Intent::QuickChat => write!(f, "quick_chat"),
            Intent::Subagent => write!(f, "subagent"),
            Intent::Embedding => write!(f, "embedding"),
            Intent::Custom(s) => write!(f, "{s}"),
        }
    }
}

impl Intent {
    pub fn from_str(s: &str) -> Self {
        match s {
            "reasoning" => Intent::Reasoning,
            "code_generation" => Intent::CodeGeneration,
            "quick_chat" => Intent::QuickChat,
            "subagent" => Intent::Subagent,
            "embedding" => Intent::Embedding,
            other => Intent::Custom(other.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_display() {
        assert_eq!(Intent::Reasoning.to_string(), "reasoning");
        assert_eq!(Intent::CodeGeneration.to_string(), "code_generation");
        assert_eq!(Intent::Custom("x".into()).to_string(), "x");
    }

    #[test]
    fn intent_roundtrip() {
        assert_eq!(Intent::from_str("reasoning"), Intent::Reasoning);
        assert_eq!(Intent::from_str("unknown"), Intent::Custom("unknown".into()));
    }

    #[test]
    fn intent_serializes() {
        let json = serde_json::to_string(&Intent::Reasoning).unwrap();
        assert_eq!(json, r#""reasoning""#);
    }
}
