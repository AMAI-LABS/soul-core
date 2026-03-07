/// Harness synthesis — AutoHarness-style constraint layer for agent tool calls.
///
/// Implements the core mechanism from "AutoHarness: Improving LLM Agents by
/// Automatically Synthesizing a Code Harness" (Google DeepMind, 2025).
///
/// The harness learns from tool failures and generates shell-based validators
/// that prevent recurrence. This is P1 of the autopoietic agent architecture.
///
/// # Three Harness Variants (matching the paper)
///
/// - **ActionVerifier**: Validates tool args before execution. Rejects + retries on failure.
///   Registered as a `before_tool_call` ModifyingHook.
/// - **ActionFilter**: Generates allowed-arg sets for the LLM to choose from.
///   (Future: requires LLM call, not implemented here yet.)
/// - **PolicyCode**: Pure shell policy — bypasses LLM entirely for known patterns.
///   (Future: registered as standalone tool via DynamicToolHandle.)
///
/// # Thompson Sampling
///
/// Each candidate harness gets a Beta(α, β) prior. On success: α += 1. On failure: β += 1.
/// Selection: sample from Beta distribution, pick highest sample.
/// This balances exploration (try new candidates) vs exploitation (use best known).
///
/// # Integration
///
/// ```rust
/// use soul_core::harness::{HarnessStore, ActionVerifierHook};
/// use soul_core::hook::HookPipeline;
/// use std::sync::Arc;
///
/// let store = Arc::new(HarnessStore::new());
/// let hook = ActionVerifierHook::new(store, 3); // activate after 3 failures per tool
///
/// let mut pipeline = HookPipeline::new();
/// pipeline.add_modifying(Arc::new(hook));
/// ```
pub mod store;
pub mod thompson;
pub mod verifier;

pub use store::{FailureRecord, HarnessCandidate, HarnessStore};
pub use thompson::ThompsonSampler;
pub use verifier::ActionVerifierHook;
