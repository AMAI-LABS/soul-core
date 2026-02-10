//! Permission enforcement system for tool execution.
//!
//! Provides a [`PermissionGate`] trait for checking tool permissions,
//! a [`PermissionManager`](manager::PermissionManager) rule engine, and a
//! [`PermissionHook`](hook::PermissionHook) adapter that plugs into the existing
//! [`HookPipeline`](crate::hook::HookPipeline).

pub mod hook;
pub mod manager;
pub mod pattern;

use std::future::Future;
use std::pin::Pin;

use serde::{Deserialize, Serialize};

/// Risk classification for tool operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    ReadOnly,
    Configuration,
    Execution,
    Destructive,
}

impl std::fmt::Display for RiskLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RiskLevel::ReadOnly => write!(f, "read_only"),
            RiskLevel::Configuration => write!(f, "configuration"),
            RiskLevel::Execution => write!(f, "execution"),
            RiskLevel::Destructive => write!(f, "destructive"),
        }
    }
}

/// Decision from a permission check
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow,
    Deny {
        reason: String,
    },
    Ask {
        prompt: String,
        tool_name: String,
        risk_level: RiskLevel,
    },
}

/// Trait for permission enforcement gates.
///
/// Implementors check whether a tool call with given arguments should be allowed.
pub trait PermissionGate: Send + Sync {
    fn check<'a>(
        &'a self,
        tool_name: &'a str,
        arguments: &'a serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = PermissionDecision> + Send + 'a>>;
}

/// A rule that matches tool calls and determines permission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionRule {
    pub tool_pattern: String,
    pub argument_patterns: Vec<ArgumentMatcher>,
    pub action: RuleAction,
    pub priority: i32,
}

/// Matches a specific argument path against a glob pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArgumentMatcher {
    pub path: String,
    pub pattern: String,
}

/// Action to take when a rule matches
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuleAction {
    Allow,
    Deny { reason: String },
    Ask { prompt: String },
}

/// Default permission policy when no rules match
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DefaultPolicy {
    Allow,
    Deny,
    Ask,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_level_ordering() {
        assert!(RiskLevel::ReadOnly < RiskLevel::Configuration);
        assert!(RiskLevel::Configuration < RiskLevel::Execution);
        assert!(RiskLevel::Execution < RiskLevel::Destructive);
    }

    #[test]
    fn risk_level_display() {
        assert_eq!(RiskLevel::ReadOnly.to_string(), "read_only");
        assert_eq!(RiskLevel::Destructive.to_string(), "destructive");
    }

    #[test]
    fn permission_decision_serializes() {
        let allow = PermissionDecision::Allow;
        let json = serde_json::to_string(&allow).unwrap();
        assert!(json.contains("allow"));

        let deny = PermissionDecision::Deny {
            reason: "blocked".into(),
        };
        let json = serde_json::to_string(&deny).unwrap();
        assert!(json.contains("deny"));
        assert!(json.contains("blocked"));
    }

    #[test]
    fn rule_action_serializes() {
        let allow = RuleAction::Allow;
        let json = serde_json::to_string(&allow).unwrap();
        let deserialized: RuleAction = serde_json::from_str(&json).unwrap();
        assert!(matches!(deserialized, RuleAction::Allow));
    }
}
