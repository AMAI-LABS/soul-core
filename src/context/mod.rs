use crate::error::{SoulError, SoulResult};
use crate::types::*;

/// Context window manager — tracks token usage, triggers compaction
pub struct ContextManager {
    config: ContextConfig,
    compaction_attempted: bool,
}

#[derive(Debug, Clone)]
pub struct ContextConfig {
    pub max_tokens: usize,
    pub compaction_threshold: f64,
    pub safety_margin: f64,
    pub min_preserved_messages: usize,
    pub reserve_tokens_floor: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_tokens: 200_000,
            compaction_threshold: 0.75,
            safety_margin: 1.2,
            min_preserved_messages: 4,
            reserve_tokens_floor: 8_000,
        }
    }
}

/// Result of a compaction operation
#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub messages_before: usize,
    pub messages_after: usize,
    pub tokens_before: usize,
    pub tokens_after: usize,
    pub strategy: CompactionStrategy,
}

/// Strategy used for compaction
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Remove tool result contents, keep paths/references (reversible)
    ReversibleOffload,
    /// Summarize old messages into a compact form
    Summarize,
    /// Preserve structured state, drop old messages
    StructuredPreserve,
    /// Combined: offload + summarize
    Hybrid,
}

impl ContextManager {
    pub fn new(config: ContextConfig) -> Self {
        Self {
            config,
            compaction_attempted: false,
        }
    }

    /// Estimate total tokens in a message list
    pub fn estimate_tokens(messages: &[Message]) -> usize {
        messages.iter().map(|m| m.estimate_tokens()).sum()
    }

    /// Check if compaction should be triggered
    pub fn needs_compaction(&self, messages: &[Message], system_tokens: usize) -> bool {
        let total = Self::estimate_tokens(messages) + system_tokens;
        let threshold = (self.config.max_tokens as f64 * self.config.compaction_threshold) as usize;
        total > threshold
    }

    /// Check if context would overflow
    pub fn would_overflow(&self, messages: &[Message], system_tokens: usize) -> bool {
        let total = Self::estimate_tokens(messages) + system_tokens;
        let effective_max =
            (self.config.max_tokens as f64 / self.config.safety_margin) as usize;
        total > effective_max
    }

    /// Perform reversible offloading — replace large tool results with references
    pub fn offload_tool_results(
        messages: &mut [Message],
        preserve_last_n: usize,
    ) -> usize {
        let len = messages.len();
        let offload_end = len.saturating_sub(preserve_last_n);
        let mut offloaded = 0;

        for msg in messages.iter_mut().take(offload_end) {
            if msg.role == Role::Tool {
                for block in &mut msg.content {
                    if let ContentBlock::ToolResult {
                        content, is_error, ..
                    } = block
                    {
                        if !*is_error && content.len() > 200 {
                            let preview = &content[..100.min(content.len())];
                            *content = format!(
                                "[Offloaded: {} chars] {preview}...",
                                content.len()
                            );
                            offloaded += 1;
                        }
                    }
                }
            }
        }
        offloaded
    }

    /// Drop old messages while preserving minimum required context
    pub fn prune_old_messages(
        messages: &[Message],
        target_tokens: usize,
        min_preserve: usize,
    ) -> Vec<Message> {
        let len = messages.len();
        if len <= min_preserve {
            return messages.to_vec();
        }

        // Always preserve at least min_preserve from the end
        let mut result: Vec<Message> = Vec::new();
        let mut tokens = 0;

        // Add messages from the end until we hit target
        let preserved: Vec<&Message> = messages.iter().rev().take(min_preserve).collect();
        for msg in preserved.iter().rev() {
            tokens += msg.estimate_tokens();
            result.push((*msg).clone());
        }

        // If we're still under target, add more from the end
        if tokens < target_tokens && len > min_preserve {
            let remaining: Vec<&Message> = messages
                .iter()
                .rev()
                .skip(min_preserve)
                .collect();

            for msg in remaining {
                let msg_tokens = msg.estimate_tokens();
                if tokens + msg_tokens > target_tokens {
                    break;
                }
                tokens += msg_tokens;
                result.insert(0, msg.clone());
            }
        }

        result
    }

    /// Compact messages with circuit breaker
    pub fn compact(
        &mut self,
        messages: &mut Vec<Message>,
        structured_state: Option<&StructuredState>,
        system_tokens: usize,
    ) -> SoulResult<CompactionResult> {
        if self.compaction_attempted {
            return Err(SoulError::CompactionFailed(
                "Circuit breaker: compaction already attempted this cycle".into(),
            ));
        }
        self.compaction_attempted = true;

        let messages_before = messages.len();
        let tokens_before = Self::estimate_tokens(messages) + system_tokens;

        // Strategy 1: Reversible offloading first
        let offloaded = Self::offload_tool_results(messages, self.config.min_preserved_messages);

        let tokens_after_offload = Self::estimate_tokens(messages) + system_tokens;
        let threshold =
            (self.config.max_tokens as f64 * self.config.compaction_threshold) as usize;

        let strategy = if tokens_after_offload <= threshold {
            CompactionStrategy::ReversibleOffload
        } else {
            // Strategy 2: Prune old messages
            let target = threshold.saturating_sub(self.config.reserve_tokens_floor);
            *messages = Self::prune_old_messages(
                messages,
                target,
                self.config.min_preserved_messages,
            );

            // If structured state exists, inject it as a system-like message
            if let Some(state) = structured_state {
                if !state.items.is_empty() {
                    let state_json = serde_json::to_string_pretty(state)
                        .unwrap_or_default();
                    let state_msg = Message::user(format!(
                        "[Compaction recovery — structured state preserved]\n{state_json}"
                    ));
                    messages.insert(0, state_msg);
                }
            }

            if offloaded > 0 {
                CompactionStrategy::Hybrid
            } else {
                CompactionStrategy::StructuredPreserve
            }
        };

        let messages_after = messages.len();
        let tokens_after = Self::estimate_tokens(messages) + system_tokens;

        Ok(CompactionResult {
            messages_before,
            messages_after,
            tokens_before,
            tokens_after,
            strategy,
        })
    }

    /// Reset the circuit breaker (call after successful LLM response)
    pub fn reset_circuit_breaker(&mut self) {
        self.compaction_attempted = false;
    }

    /// Get token utilization as a fraction (0.0 to 1.0+)
    pub fn utilization(&self, messages: &[Message], system_tokens: usize) -> f64 {
        let total = Self::estimate_tokens(messages) + system_tokens;
        total as f64 / self.config.max_tokens as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_messages(count: usize, text_len: usize) -> Vec<Message> {
        (0..count)
            .map(|i| {
                if i % 2 == 0 {
                    Message::user("x".repeat(text_len))
                } else {
                    Message::assistant("y".repeat(text_len))
                }
            })
            .collect()
    }

    fn make_tool_messages(count: usize, content_len: usize) -> Vec<Message> {
        (0..count)
            .map(|i| Message::tool_result(format!("tc_{i}"), "z".repeat(content_len), false))
            .collect()
    }

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(ContextManager::estimate_tokens(&[]), 0);
    }

    #[test]
    fn estimate_tokens_basic() {
        let messages = vec![Message::user("hello world")]; // ~11 chars = ~4 tokens + 4 overhead
        let tokens = ContextManager::estimate_tokens(&messages);
        assert!(tokens > 0);
        assert!(tokens < 20);
    }

    #[test]
    fn needs_compaction_below_threshold() {
        let config = ContextConfig {
            max_tokens: 10_000,
            compaction_threshold: 0.75,
            ..Default::default()
        };
        let mgr = ContextManager::new(config);
        let messages = make_messages(2, 10); // very small
        assert!(!mgr.needs_compaction(&messages, 100));
    }

    #[test]
    fn needs_compaction_above_threshold() {
        let config = ContextConfig {
            max_tokens: 100,
            compaction_threshold: 0.75,
            ..Default::default()
        };
        let mgr = ContextManager::new(config);
        let messages = make_messages(20, 100); // large
        assert!(mgr.needs_compaction(&messages, 100));
    }

    #[test]
    fn would_overflow() {
        let config = ContextConfig {
            max_tokens: 100,
            safety_margin: 1.2,
            ..Default::default()
        };
        let mgr = ContextManager::new(config);
        let messages = make_messages(20, 100);
        assert!(mgr.would_overflow(&messages, 100));
    }

    #[test]
    fn offload_tool_results_large() {
        let mut messages = make_tool_messages(5, 500);
        let offloaded = ContextManager::offload_tool_results(&mut messages, 2);

        // Should offload first 3, preserve last 2
        assert_eq!(offloaded, 3);

        // First 3 should be offloaded
        for msg in &messages[..3] {
            let text = msg.text_content();
            // Tool results don't have text content, check the content blocks
            if let ContentBlock::ToolResult { content, .. } = &msg.content[0] {
                assert!(content.contains("[Offloaded:"));
            }
        }

        // Last 2 should be preserved
        for msg in &messages[3..] {
            if let ContentBlock::ToolResult { content, .. } = &msg.content[0] {
                assert!(!content.contains("[Offloaded:"));
            }
        }
    }

    #[test]
    fn offload_skips_small_results() {
        let mut messages = make_tool_messages(3, 50); // under 200 char threshold
        let offloaded = ContextManager::offload_tool_results(&mut messages, 0);
        assert_eq!(offloaded, 0);
    }

    #[test]
    fn offload_skips_errors() {
        let mut messages = vec![Message::tool_result("tc1", "x".repeat(500), true)];
        let offloaded = ContextManager::offload_tool_results(&mut messages, 0);
        assert_eq!(offloaded, 0);
    }

    #[test]
    fn prune_preserves_minimum() {
        let messages = make_messages(10, 10);
        let pruned = ContextManager::prune_old_messages(&messages, 1, 4);
        assert!(pruned.len() >= 4);
    }

    #[test]
    fn prune_small_list_unchanged() {
        let messages = make_messages(3, 10);
        let pruned = ContextManager::prune_old_messages(&messages, 1000, 4);
        assert_eq!(pruned.len(), 3); // less than min_preserve, keep all
    }

    #[test]
    fn compact_circuit_breaker() {
        let config = ContextConfig {
            max_tokens: 100,
            ..Default::default()
        };
        let mut mgr = ContextManager::new(config);
        let mut messages = make_messages(20, 100);

        // First compaction succeeds
        let result = mgr.compact(&mut messages, None, 0);
        assert!(result.is_ok());

        // Second compaction fails (circuit breaker)
        let mut messages2 = make_messages(20, 100);
        let result = mgr.compact(&mut messages2, None, 0);
        assert!(matches!(result, Err(SoulError::CompactionFailed(_))));
    }

    #[test]
    fn compact_circuit_breaker_resets() {
        let config = ContextConfig {
            max_tokens: 100,
            ..Default::default()
        };
        let mut mgr = ContextManager::new(config);
        let mut messages = make_messages(20, 100);

        let _ = mgr.compact(&mut messages, None, 0);
        mgr.reset_circuit_breaker();

        let mut messages2 = make_messages(20, 100);
        let result = mgr.compact(&mut messages2, None, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn compact_preserves_structured_state() {
        let config = ContextConfig {
            max_tokens: 200,
            compaction_threshold: 0.5,
            min_preserved_messages: 2,
            ..Default::default()
        };
        let mut mgr = ContextManager::new(config);
        let mut messages = make_messages(50, 100);

        let mut state = StructuredState::new();
        state.add("Task 1", None);
        state.set_status(0, ItemStatus::Completed);
        state.add("Task 2", None);
        state.set_status(1, ItemStatus::InProgress);

        let result = mgr.compact(&mut messages, Some(&state), 0).unwrap();
        assert!(result.messages_after < result.messages_before);

        // Check that structured state was injected
        let first_text = messages[0].text_content();
        assert!(first_text.contains("structured state preserved"));
        assert!(first_text.contains("Task 1"));
        assert!(first_text.contains("Task 2"));
    }

    #[test]
    fn utilization_calculation() {
        let config = ContextConfig {
            max_tokens: 1000,
            ..Default::default()
        };
        let mgr = ContextManager::new(config);

        let messages = vec![Message::user("hello")];
        let util = mgr.utilization(&messages, 0);
        assert!(util > 0.0);
        assert!(util < 1.0);
    }

    #[test]
    fn compact_result_strategy() {
        let config = ContextConfig {
            max_tokens: 10_000,
            compaction_threshold: 0.75,
            min_preserved_messages: 2,
            ..Default::default()
        };
        let mut mgr = ContextManager::new(config);

        // Small messages with large tool results → ReversibleOffload
        let mut messages = vec![
            Message::user("hello"),
            Message::assistant("checking"),
            Message::tool_result("tc1", "x".repeat(2000), false),
            Message::user("ok"),
            Message::assistant("done"),
        ];

        let result = mgr.compact(&mut messages, None, 0).unwrap();
        // Should have used some form of offloading or pruning
        assert!(result.tokens_after <= result.tokens_before);
    }
}
