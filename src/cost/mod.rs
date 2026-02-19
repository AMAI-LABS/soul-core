//! Cost tracking and budget enforcement for agent sessions.
//!
//! Tracks token usage, computes costs per model, and enforces budget policies.
//!
//! # Example
//!
//! ```rust
//! use soul_core::cost::{CostTracker, BudgetEnforcer, BudgetPolicy};
//! use soul_core::types::{TokenUsage, ModelInfo, ProviderKind};
//!
//! let tracker = CostTracker::new("session-1".into());
//! let enforcer = BudgetEnforcer::new(BudgetPolicy::new().with_hard_limit(1.0));
//! ```

pub mod budget;
pub mod pricing;

pub use budget::{BudgetEnforcer, BudgetPolicy, BudgetStatus};
pub use pricing::{compute_cost, cost_per_token};

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use serde::{Deserialize, Serialize};

use crate::types::{ModelInfo, TokenUsage};

/// Tracks cumulative costs for an agent session.
pub struct CostTracker {
    inner: Arc<RwLock<CostState>>,
    session_id: String,
}

/// Internal mutable state for cost tracking.
#[derive(Debug, Clone, Default)]
struct CostState {
    total_input_tokens: u64,
    total_output_tokens: u64,
    total_cache_read_tokens: u64,
    total_cache_write_tokens: u64,
    total_cost_usd: f64,
    total_turns: u64,
    by_model: HashMap<String, ModelCostBreakdown>,
    by_tool: HashMap<String, ToolCostBreakdown>,
    turns: Vec<TurnCost>,
}

/// Cost breakdown for a specific model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCostBreakdown {
    pub model_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub request_count: u64,
}

/// Cost association for a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCostBreakdown {
    pub tool_name: String,
    pub associated_cost_usd: f64,
    pub call_count: u64,
}

/// Cost for a single turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnCost {
    pub turn_index: u64,
    pub model_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd: f64,
    pub tool_calls: Vec<String>,
    pub timestamp_ms: u64,
}

/// Event emitted after each turn with cost information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEvent {
    pub session_id: String,
    pub turn_index: u64,
    pub model_id: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub turn_cost_usd: f64,
    pub cumulative_cost_usd: f64,
    pub budget_remaining_usd: Option<f64>,
    pub timestamp_ms: u64,
}

/// Summary of all costs for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostSummary {
    pub session_id: String,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cost_usd: f64,
    pub total_turns: u64,
    pub by_model: Vec<ModelCostBreakdown>,
    pub by_tool: Vec<ToolCostBreakdown>,
}

impl CostTracker {
    pub fn new(session_id: String) -> Self {
        Self {
            inner: Arc::new(RwLock::new(CostState::default())),
            session_id,
        }
    }

    /// Record a turn's token usage and compute cost.
    pub async fn record_turn(
        &self,
        model_id: &str,
        usage: &TokenUsage,
        model_info: &ModelInfo,
        tool_calls: &[String],
    ) -> CostEvent {
        let turn_cost = compute_cost(usage, model_info);

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let mut state = self.inner.write().await;

        state.total_input_tokens += usage.input_tokens as u64;
        state.total_output_tokens += usage.output_tokens as u64;
        state.total_cache_read_tokens += usage.cache_read_tokens as u64;
        state.total_cache_write_tokens += usage.cache_write_tokens as u64;
        state.total_cost_usd += turn_cost;

        let turn_index = state.total_turns;
        state.total_turns += 1;

        // Update by-model breakdown
        let model_entry = state
            .by_model
            .entry(model_id.to_string())
            .or_insert_with(|| ModelCostBreakdown {
                model_id: model_id.to_string(),
                input_tokens: 0,
                output_tokens: 0,
                cost_usd: 0.0,
                request_count: 0,
            });
        model_entry.input_tokens += usage.input_tokens as u64;
        model_entry.output_tokens += usage.output_tokens as u64;
        model_entry.cost_usd += turn_cost;
        model_entry.request_count += 1;

        // Update by-tool breakdown
        for tool_name in tool_calls {
            let tool_entry =
                state
                    .by_tool
                    .entry(tool_name.clone())
                    .or_insert_with(|| ToolCostBreakdown {
                        tool_name: tool_name.clone(),
                        associated_cost_usd: 0.0,
                        call_count: 0,
                    });
            tool_entry.call_count += 1;
            // Associate the turn cost evenly across tool calls
            if !tool_calls.is_empty() {
                tool_entry.associated_cost_usd += turn_cost / tool_calls.len() as f64;
            }
        }

        state.turns.push(TurnCost {
            turn_index,
            model_id: model_id.to_string(),
            input_tokens: usage.input_tokens as u64,
            output_tokens: usage.output_tokens as u64,
            cost_usd: turn_cost,
            tool_calls: tool_calls.to_vec(),
            timestamp_ms: now_ms,
        });

        CostEvent {
            session_id: self.session_id.clone(),
            turn_index,
            model_id: model_id.to_string(),
            input_tokens: usage.input_tokens as u64,
            output_tokens: usage.output_tokens as u64,
            turn_cost_usd: turn_cost,
            cumulative_cost_usd: state.total_cost_usd,
            budget_remaining_usd: None,
            timestamp_ms: now_ms,
        }
    }

    /// Get a full cost summary.
    pub async fn summary(&self) -> CostSummary {
        let state = self.inner.read().await;
        CostSummary {
            session_id: self.session_id.clone(),
            total_input_tokens: state.total_input_tokens,
            total_output_tokens: state.total_output_tokens,
            total_cost_usd: state.total_cost_usd,
            total_turns: state.total_turns,
            by_model: state.by_model.values().cloned().collect(),
            by_tool: state.by_tool.values().cloned().collect(),
        }
    }

    /// Get total cost in USD.
    pub async fn total_cost_usd(&self) -> f64 {
        self.inner.read().await.total_cost_usd
    }

    /// Get total tokens used (input + output).
    pub async fn total_tokens(&self) -> u64 {
        let state = self.inner.read().await;
        state.total_input_tokens + state.total_output_tokens
    }

    /// Get total turn count.
    pub async fn total_turns(&self) -> u64 {
        self.inner.read().await.total_turns
    }

    /// Get the session ID.
    pub fn session_id(&self) -> &str {
        &self.session_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderKind;

    fn test_model() -> ModelInfo {
        ModelInfo {
            id: "claude-sonnet-4-5-20250929".into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.000003,
            cost_per_output_token: 0.000015,
        }
    }

    #[tokio::test]
    async fn record_single_turn() {
        let tracker = CostTracker::new("s1".into());
        let usage = TokenUsage::new(1000, 500);
        let event = tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &test_model(), &[])
            .await;

        assert_eq!(event.session_id, "s1");
        assert_eq!(event.turn_index, 0);
        assert_eq!(event.input_tokens, 1000);
        assert_eq!(event.output_tokens, 500);
        assert!(event.turn_cost_usd > 0.0);
        assert_eq!(event.cumulative_cost_usd, event.turn_cost_usd);
    }

    #[tokio::test]
    async fn multi_turn_accumulation() {
        let tracker = CostTracker::new("s1".into());
        let usage = TokenUsage::new(1000, 500);
        let model = test_model();

        let e1 = tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &model, &[])
            .await;
        let e2 = tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &model, &[])
            .await;

        assert_eq!(e2.turn_index, 1);
        assert!((e2.cumulative_cost_usd - e1.turn_cost_usd * 2.0).abs() < 0.0001);
        assert_eq!(tracker.total_turns().await, 2);
    }

    #[tokio::test]
    async fn by_model_breakdown() {
        let tracker = CostTracker::new("s1".into());
        let model1 = test_model();
        let mut model2 = test_model();
        model2.id = "gpt-4o".into();
        model2.cost_per_input_token = 0.0000025;
        model2.cost_per_output_token = 0.00001;

        let usage = TokenUsage::new(1000, 500);
        tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &model1, &[])
            .await;
        tracker.record_turn("gpt-4o", &usage, &model2, &[]).await;

        let summary = tracker.summary().await;
        assert_eq!(summary.by_model.len(), 2);
        assert_eq!(summary.total_turns, 2);
    }

    #[tokio::test]
    async fn by_tool_breakdown() {
        let tracker = CostTracker::new("s1".into());
        let usage = TokenUsage::new(1000, 500);
        let model = test_model();

        tracker
            .record_turn(
                "claude-sonnet-4-5-20250929",
                &usage,
                &model,
                &["read".into(), "write".into()],
            )
            .await;

        let summary = tracker.summary().await;
        assert_eq!(summary.by_tool.len(), 2);
        for tool in &summary.by_tool {
            assert_eq!(tool.call_count, 1);
        }
    }

    #[tokio::test]
    async fn summary_matches_totals() {
        let tracker = CostTracker::new("s1".into());
        let usage = TokenUsage::new(1000, 500);
        let model = test_model();

        tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &model, &[])
            .await;

        let summary = tracker.summary().await;
        assert_eq!(summary.total_input_tokens, 1000);
        assert_eq!(summary.total_output_tokens, 500);
        assert!(summary.total_cost_usd > 0.0);
        assert!((summary.total_cost_usd - tracker.total_cost_usd().await).abs() < 0.0001);
    }

    #[tokio::test]
    async fn cost_event_serializes() {
        let tracker = CostTracker::new("s1".into());
        let usage = TokenUsage::new(100, 50);
        let event = tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &test_model(), &[])
            .await;

        let json = serde_json::to_string(&event).unwrap();
        let deserialized: CostEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.session_id, "s1");
        assert_eq!(deserialized.input_tokens, 100);
    }

    #[tokio::test]
    async fn cost_summary_serializes() {
        let tracker = CostTracker::new("s1".into());
        let usage = TokenUsage::new(100, 50);
        tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &test_model(), &[])
            .await;

        let summary = tracker.summary().await;
        let json = serde_json::to_string(&summary).unwrap();
        let deserialized: CostSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.session_id, "s1");
    }

    #[tokio::test]
    async fn session_id_accessor() {
        let tracker = CostTracker::new("my-session".into());
        assert_eq!(tracker.session_id(), "my-session");
    }

    #[tokio::test]
    async fn total_tokens_accessor() {
        let tracker = CostTracker::new("s1".into());
        let usage = TokenUsage::new(1000, 500);
        tracker
            .record_turn("claude-sonnet-4-5-20250929", &usage, &test_model(), &[])
            .await;
        assert_eq!(tracker.total_tokens().await, 1500);
    }
}
