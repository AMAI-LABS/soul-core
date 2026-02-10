use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use super::pattern::{glob_match, match_argument};
use super::{
    ArgumentMatcher, DefaultPolicy, PermissionDecision, PermissionGate, PermissionRule, RiskLevel,
    RuleAction,
};

/// Rule-based permission manager.
///
/// Evaluates rules in priority order (highest first). Within the same priority:
/// - Deny rules are checked first (deny wins)
/// - Then Allow rules
/// - Then Ask rules
///
/// If no rule matches, the default policy applies.
pub struct PermissionManager {
    rules: Vec<PermissionRule>,
    default_policy: DefaultPolicy,
    risk_classifications: HashMap<String, RiskLevel>,
}

impl PermissionManager {
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            default_policy: DefaultPolicy::Ask,
            risk_classifications: HashMap::new(),
        }
    }

    /// Set the default policy for when no rules match.
    pub fn with_default_policy(mut self, policy: DefaultPolicy) -> Self {
        self.default_policy = policy;
        self
    }

    /// Add a permission rule.
    pub fn add_rule(&mut self, rule: PermissionRule) -> &mut Self {
        self.rules.push(rule);
        // Sort by priority descending, deny-first within same priority
        self.rules.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                let a_deny = matches!(a.action, RuleAction::Deny { .. });
                let b_deny = matches!(b.action, RuleAction::Deny { .. });
                b_deny.cmp(&a_deny)
            })
        });
        self
    }

    /// Classify a tool pattern with a risk level.
    pub fn classify_risk(
        &mut self,
        tool_pattern: impl Into<String>,
        level: RiskLevel,
    ) -> &mut Self {
        self.risk_classifications.insert(tool_pattern.into(), level);
        self
    }

    /// Get the risk level for a tool name.
    pub fn risk_level_for(&self, tool_name: &str) -> RiskLevel {
        for (pattern, level) in &self.risk_classifications {
            if glob_match(pattern, tool_name) {
                return *level;
            }
        }
        RiskLevel::Execution // default
    }

    /// Evaluate permission for a tool call.
    pub fn evaluate(&self, tool_name: &str, arguments: &serde_json::Value) -> PermissionDecision {
        for rule in &self.rules {
            if !glob_match(&rule.tool_pattern, tool_name) {
                continue;
            }

            // Check argument matchers — all must match
            let args_match = rule
                .argument_patterns
                .iter()
                .all(|am| match_argument(arguments, &am.path, &am.pattern));

            if !args_match {
                continue;
            }

            // Rule matches
            return match &rule.action {
                RuleAction::Allow => PermissionDecision::Allow,
                RuleAction::Deny { reason } => PermissionDecision::Deny {
                    reason: reason.clone(),
                },
                RuleAction::Ask { prompt } => PermissionDecision::Ask {
                    prompt: prompt.clone(),
                    tool_name: tool_name.to_string(),
                    risk_level: self.risk_level_for(tool_name),
                },
            };
        }

        // No rule matched → default policy
        match self.default_policy {
            DefaultPolicy::Allow => PermissionDecision::Allow,
            DefaultPolicy::Deny => PermissionDecision::Deny {
                reason: format!("No rule allows tool '{tool_name}'"),
            },
            DefaultPolicy::Ask => PermissionDecision::Ask {
                prompt: format!("Allow tool '{tool_name}'?"),
                tool_name: tool_name.to_string(),
                risk_level: self.risk_level_for(tool_name),
            },
        }
    }
}

impl Default for PermissionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PermissionGate for PermissionManager {
    fn check<'a>(
        &'a self,
        tool_name: &'a str,
        arguments: &'a serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = PermissionDecision> + Send + 'a>> {
        Box::pin(async move { self.evaluate(tool_name, arguments) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn deny_rule(pattern: &str, reason: &str, priority: i32) -> PermissionRule {
        PermissionRule {
            tool_pattern: pattern.into(),
            argument_patterns: vec![],
            action: RuleAction::Deny {
                reason: reason.into(),
            },
            priority,
        }
    }

    fn allow_rule(pattern: &str, priority: i32) -> PermissionRule {
        PermissionRule {
            tool_pattern: pattern.into(),
            argument_patterns: vec![],
            action: RuleAction::Allow,
            priority,
        }
    }

    fn ask_rule(pattern: &str, prompt: &str, priority: i32) -> PermissionRule {
        PermissionRule {
            tool_pattern: pattern.into(),
            argument_patterns: vec![],
            action: RuleAction::Ask {
                prompt: prompt.into(),
            },
            priority,
        }
    }

    #[test]
    fn deny_rule_blocks() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(deny_rule("bash", "bash is dangerous", 10));

        let result = mgr.evaluate("bash", &json!({}));
        assert!(
            matches!(result, PermissionDecision::Deny { reason } if reason.contains("dangerous"))
        );
    }

    #[test]
    fn allow_rule_passes() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(allow_rule("read", 10));

        let result = mgr.evaluate("read", &json!({}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    #[test]
    fn ask_rule_prompts() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(ask_rule("write", "Allow file write?", 10));

        let result = mgr.evaluate("write", &json!({}));
        assert!(matches!(result, PermissionDecision::Ask { .. }));
    }

    #[test]
    fn deny_wins_over_allow_same_priority() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(allow_rule("bash", 10));
        mgr.add_rule(deny_rule("bash", "blocked", 10));

        let result = mgr.evaluate("bash", &json!({}));
        assert!(matches!(result, PermissionDecision::Deny { .. }));
    }

    #[test]
    fn higher_priority_wins() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(allow_rule("bash", 20));
        mgr.add_rule(deny_rule("bash", "blocked", 10));

        let result = mgr.evaluate("bash", &json!({}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    #[test]
    fn glob_pattern_matches() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(allow_rule("read_*", 10));

        assert!(matches!(
            mgr.evaluate("read_file", &json!({})),
            PermissionDecision::Allow
        ));
        assert!(!matches!(
            mgr.evaluate("write_file", &json!({})),
            PermissionDecision::Allow
        ));
    }

    #[test]
    fn wildcard_allows_all() {
        let mut mgr = PermissionManager::new().with_default_policy(DefaultPolicy::Deny);
        mgr.add_rule(allow_rule("*", 1));

        assert!(matches!(
            mgr.evaluate("anything", &json!({})),
            PermissionDecision::Allow
        ));
    }

    #[test]
    fn default_policy_ask() {
        let mgr = PermissionManager::new(); // default = Ask
        let result = mgr.evaluate("unknown_tool", &json!({}));
        assert!(matches!(result, PermissionDecision::Ask { .. }));
    }

    #[test]
    fn default_policy_deny() {
        let mgr = PermissionManager::new().with_default_policy(DefaultPolicy::Deny);
        let result = mgr.evaluate("unknown_tool", &json!({}));
        assert!(matches!(result, PermissionDecision::Deny { .. }));
    }

    #[test]
    fn default_policy_allow() {
        let mgr = PermissionManager::new().with_default_policy(DefaultPolicy::Allow);
        let result = mgr.evaluate("unknown_tool", &json!({}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    #[test]
    fn argument_pattern_matching() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(PermissionRule {
            tool_pattern: "bash".into(),
            argument_patterns: vec![ArgumentMatcher {
                path: "/command".into(),
                pattern: "rm *".into(),
            }],
            action: RuleAction::Deny {
                reason: "rm commands blocked".into(),
            },
            priority: 20,
        });
        mgr.add_rule(allow_rule("bash", 10));

        // rm command → deny
        let result = mgr.evaluate("bash", &json!({"command": "rm -rf /"}));
        assert!(matches!(result, PermissionDecision::Deny { .. }));

        // ls command → allow (arg pattern doesn't match deny rule)
        let result = mgr.evaluate("bash", &json!({"command": "ls -la"}));
        assert!(matches!(result, PermissionDecision::Allow));
    }

    #[test]
    fn multiple_argument_patterns_all_must_match() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(PermissionRule {
            tool_pattern: "write".into(),
            argument_patterns: vec![
                ArgumentMatcher {
                    path: "/path".into(),
                    pattern: "/etc/*".into(),
                },
                ArgumentMatcher {
                    path: "/force".into(),
                    pattern: "true".into(),
                },
            ],
            action: RuleAction::Deny {
                reason: "forced writes to /etc blocked".into(),
            },
            priority: 10,
        });

        // Both match → deny
        let result = mgr.evaluate("write", &json!({"path": "/etc/passwd", "force": true}));
        assert!(matches!(result, PermissionDecision::Deny { .. }));

        // Only path matches → falls through (default Ask)
        let result = mgr.evaluate("write", &json!({"path": "/etc/passwd", "force": false}));
        assert!(matches!(result, PermissionDecision::Ask { .. }));
    }

    #[test]
    fn risk_classification() {
        let mut mgr = PermissionManager::new();
        mgr.classify_risk("read*", RiskLevel::ReadOnly);
        mgr.classify_risk("bash", RiskLevel::Destructive);

        assert_eq!(mgr.risk_level_for("read_file"), RiskLevel::ReadOnly);
        assert_eq!(mgr.risk_level_for("bash"), RiskLevel::Destructive);
        assert_eq!(mgr.risk_level_for("unknown"), RiskLevel::Execution); // default
    }

    #[test]
    fn ask_includes_risk_level() {
        let mut mgr = PermissionManager::new();
        mgr.classify_risk("bash", RiskLevel::Destructive);
        mgr.add_rule(ask_rule("bash", "Allow bash?", 10));

        let result = mgr.evaluate("bash", &json!({}));
        match result {
            PermissionDecision::Ask { risk_level, .. } => {
                assert_eq!(risk_level, RiskLevel::Destructive);
            }
            _ => panic!("Expected Ask"),
        }
    }

    #[tokio::test]
    async fn permission_gate_trait() {
        let mut mgr = PermissionManager::new();
        mgr.add_rule(allow_rule("read", 10));

        let gate: &dyn PermissionGate = &mgr;
        let result = gate.check("read", &json!({})).await;
        assert!(matches!(result, PermissionDecision::Allow));
    }
}
