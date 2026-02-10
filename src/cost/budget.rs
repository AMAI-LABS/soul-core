use serde::{Deserialize, Serialize};

/// Policy that defines spending and resource limits for an agent session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetPolicy {
    /// USD threshold that triggers a warning (but doesn't stop execution)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warn_threshold_usd: Option<f64>,

    /// USD threshold that stops execution
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hard_limit_usd: Option<f64>,

    /// Maximum total tokens (input + output) before stopping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_total_tokens: Option<u64>,

    /// Maximum number of agent turns before stopping
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<u64>,
}

impl BudgetPolicy {
    pub fn new() -> Self {
        Self {
            warn_threshold_usd: None,
            hard_limit_usd: None,
            max_total_tokens: None,
            max_turns: None,
        }
    }

    pub fn with_hard_limit(mut self, usd: f64) -> Self {
        self.hard_limit_usd = Some(usd);
        self
    }

    pub fn with_warn_threshold(mut self, usd: f64) -> Self {
        self.warn_threshold_usd = Some(usd);
        self
    }

    pub fn with_max_tokens(mut self, tokens: u64) -> Self {
        self.max_total_tokens = Some(tokens);
        self
    }

    pub fn with_max_turns(mut self, turns: u64) -> Self {
        self.max_turns = Some(turns);
        self
    }
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Current status relative to budget limits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetStatus {
    Ok,
    Warning,
    Exceeded,
}

/// Enforces budget policies against current cost state.
pub struct BudgetEnforcer {
    policy: BudgetPolicy,
}

impl BudgetEnforcer {
    pub fn new(policy: BudgetPolicy) -> Self {
        Self { policy }
    }

    /// Check the current budget status.
    pub fn check(&self, total_cost_usd: f64, total_tokens: u64, total_turns: u64) -> BudgetStatus {
        // Check hard limits first
        if let Some(limit) = self.policy.hard_limit_usd {
            if total_cost_usd >= limit {
                return BudgetStatus::Exceeded;
            }
        }
        if let Some(limit) = self.policy.max_total_tokens {
            if total_tokens >= limit {
                return BudgetStatus::Exceeded;
            }
        }
        if let Some(limit) = self.policy.max_turns {
            if total_turns >= limit {
                return BudgetStatus::Exceeded;
            }
        }

        // Check warning thresholds
        if let Some(threshold) = self.policy.warn_threshold_usd {
            if total_cost_usd >= threshold {
                return BudgetStatus::Warning;
            }
        }

        BudgetStatus::Ok
    }

    /// Returns true if execution should stop.
    pub fn should_stop(&self, total_cost_usd: f64, total_tokens: u64, total_turns: u64) -> bool {
        self.check(total_cost_usd, total_tokens, total_turns) == BudgetStatus::Exceeded
    }

    /// Get a reference to the underlying policy.
    pub fn policy(&self) -> &BudgetPolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_when_under_all_limits() {
        let enforcer = BudgetEnforcer::new(
            BudgetPolicy::new()
                .with_hard_limit(10.0)
                .with_max_tokens(1_000_000)
                .with_max_turns(100),
        );
        assert_eq!(enforcer.check(1.0, 1000, 5), BudgetStatus::Ok);
        assert!(!enforcer.should_stop(1.0, 1000, 5));
    }

    #[test]
    fn warning_on_threshold() {
        let enforcer = BudgetEnforcer::new(
            BudgetPolicy::new()
                .with_warn_threshold(5.0)
                .with_hard_limit(10.0),
        );
        assert_eq!(enforcer.check(5.0, 0, 0), BudgetStatus::Warning);
        assert!(!enforcer.should_stop(5.0, 0, 0));
    }

    #[test]
    fn exceeded_on_cost() {
        let enforcer = BudgetEnforcer::new(BudgetPolicy::new().with_hard_limit(10.0));
        assert_eq!(enforcer.check(10.0, 0, 0), BudgetStatus::Exceeded);
        assert!(enforcer.should_stop(10.0, 0, 0));
    }

    #[test]
    fn exceeded_on_tokens() {
        let enforcer = BudgetEnforcer::new(BudgetPolicy::new().with_max_tokens(1000));
        assert_eq!(enforcer.check(0.0, 1000, 0), BudgetStatus::Exceeded);
        assert!(enforcer.should_stop(0.0, 1000, 0));
    }

    #[test]
    fn exceeded_on_turns() {
        let enforcer = BudgetEnforcer::new(BudgetPolicy::new().with_max_turns(10));
        assert_eq!(enforcer.check(0.0, 0, 10), BudgetStatus::Exceeded);
        assert!(enforcer.should_stop(0.0, 0, 10));
    }

    #[test]
    fn no_limits_always_ok() {
        let enforcer = BudgetEnforcer::new(BudgetPolicy::new());
        assert_eq!(
            enforcer.check(999_999.0, 999_999_999, 999_999),
            BudgetStatus::Ok
        );
        assert!(!enforcer.should_stop(999_999.0, 999_999_999, 999_999));
    }

    #[test]
    fn cost_exceeded_beats_warning() {
        let enforcer = BudgetEnforcer::new(
            BudgetPolicy::new()
                .with_warn_threshold(5.0)
                .with_hard_limit(10.0),
        );
        // At the hard limit, should be exceeded (not just warning)
        assert_eq!(enforcer.check(10.0, 0, 0), BudgetStatus::Exceeded);
    }

    #[test]
    fn budget_policy_serializes() {
        let policy = BudgetPolicy::new().with_hard_limit(25.0).with_max_turns(50);
        let json = serde_json::to_string(&policy).unwrap();
        let deserialized: BudgetPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.hard_limit_usd, Some(25.0));
        assert_eq!(deserialized.max_turns, Some(50));
    }

    #[test]
    fn budget_status_serializes() {
        let json = serde_json::to_string(&BudgetStatus::Exceeded).unwrap();
        assert_eq!(json, "\"exceeded\"");
    }

    #[test]
    fn policy_accessor() {
        let policy = BudgetPolicy::new().with_hard_limit(5.0);
        let enforcer = BudgetEnforcer::new(policy);
        assert_eq!(enforcer.policy().hard_limit_usd, Some(5.0));
    }
}
