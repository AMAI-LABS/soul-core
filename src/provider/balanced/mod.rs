//! Balanced provider — wraps multiple LLM providers with intent-based routing,
//! rate limit tracking, and automatic failover.
//!
//! Implements `Provider` trait so it drops in anywhere a single provider is used.

pub mod intent;
pub mod rate_limit;

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::{SoulError, SoulResult};
use crate::types::*;

use super::traits::{ProbeResult, Provider};
use intent::Intent;
use rate_limit::RateLimitTracker;

/// A provider slot in the balancer
pub struct ProviderSlot {
    pub name: String,
    pub provider: Arc<dyn Provider>,
    pub model: ModelInfo,
    pub auth: AuthProfile,
    pub weight: u32,
    pub rate_limit: RateLimitTracker,
}

/// Intent-to-provider mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentMapping {
    /// Ordered list of preferred provider names for this intent
    pub preferred: Vec<String>,
    /// Override model per provider for this intent
    #[serde(default)]
    pub models: HashMap<String, String>,
}

/// Routing strategy
#[derive(Debug, Clone, PartialEq)]
pub enum Strategy {
    /// Cycle through available providers
    RoundRobin,
    /// Distribute by weight
    Weighted,
    /// Pick least-loaded provider
    LeastLoaded,
    /// Use first available (ordered by config)
    Failover,
}

/// The balanced provider — wraps N providers, routes by intent
pub struct BalancedProvider {
    slots: Vec<ProviderSlot>,
    intents: HashMap<Intent, IntentMapping>,
    strategy: Strategy,
    counter: AtomicUsize,
}

/// Status of the balanced provider
#[derive(Debug, Clone, Serialize)]
pub struct BalancedStatus {
    pub total_slots: usize,
    pub available_slots: usize,
    pub slots: Vec<SlotStatus>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SlotStatus {
    pub name: String,
    pub provider_kind: String,
    pub model: String,
    pub weight: u32,
    pub available: bool,
    pub rate_limit: rate_limit::RateLimitStatus,
}

impl BalancedProvider {
    pub fn new(strategy: Strategy) -> Self {
        Self {
            slots: Vec::new(),
            intents: HashMap::new(),
            strategy,
            counter: AtomicUsize::new(0),
        }
    }

    /// Add a provider slot
    pub fn add_slot(
        &mut self,
        name: impl Into<String>,
        provider: Arc<dyn Provider>,
        model: ModelInfo,
        auth: AuthProfile,
        weight: u32,
        rate_limit: RateLimitTracker,
    ) {
        self.slots.push(ProviderSlot {
            name: name.into(),
            provider,
            model,
            auth,
            weight,
            rate_limit,
        });
    }

    /// Map an intent to preferred providers
    pub fn map_intent(&mut self, intent: Intent, mapping: IntentMapping) {
        self.intents.insert(intent, mapping);
    }

    /// Get status of all slots
    pub fn status(&self) -> BalancedStatus {
        let slots: Vec<SlotStatus> = self
            .slots
            .iter()
            .map(|s| SlotStatus {
                name: s.name.clone(),
                provider_kind: s.provider.kind().to_string(),
                model: s.model.id.clone(),
                weight: s.weight,
                available: s.rate_limit.can_accept(),
                rate_limit: s.rate_limit.status(),
            })
            .collect();
        let available = slots.iter().filter(|s| s.available).count();
        BalancedStatus {
            total_slots: slots.len(),
            available_slots: available,
            slots,
        }
    }

    /// Select a slot index for the given intent
    fn select(&self, intent: Option<&Intent>) -> SoulResult<usize> {
        // Build candidate list based on intent preferences
        let candidates = if let Some(intent) = intent {
            if let Some(mapping) = self.intents.get(intent) {
                // Try preferred providers first (in order)
                let mut ordered: Vec<usize> = Vec::new();
                for pref_name in &mapping.preferred {
                    if let Some(idx) = self
                        .slots
                        .iter()
                        .position(|s| &s.name == pref_name && s.rate_limit.can_accept())
                    {
                        ordered.push(idx);
                    }
                }
                // Add remaining available slots as fallback
                for (idx, slot) in self.slots.iter().enumerate() {
                    if slot.rate_limit.can_accept() && !ordered.contains(&idx) {
                        ordered.push(idx);
                    }
                }
                ordered
            } else {
                self.available_indices()
            }
        } else {
            self.available_indices()
        };

        if candidates.is_empty() {
            return Err(SoulError::FailoverExhausted {
                attempts: self.slots.len(),
            });
        }

        match self.strategy {
            Strategy::Failover => Ok(candidates[0]),
            Strategy::RoundRobin => {
                let idx = self.counter.fetch_add(1, Ordering::Relaxed);
                Ok(candidates[idx % candidates.len()])
            }
            Strategy::Weighted => {
                let total: u32 = candidates.iter().map(|&i| self.slots[i].weight).sum();
                if total == 0 {
                    return Ok(candidates[0]);
                }
                let target = (self.counter.fetch_add(1, Ordering::Relaxed) as u32) % total;
                let mut cum = 0u32;
                for &idx in &candidates {
                    cum += self.slots[idx].weight;
                    if target < cum {
                        return Ok(idx);
                    }
                }
                Ok(*candidates.last().unwrap())
            }
            Strategy::LeastLoaded => {
                let best = candidates
                    .iter()
                    .min_by_key(|&&i| self.slots[i].rate_limit.status().rpm_used)
                    .unwrap();
                Ok(*best)
            }
        }
    }

    fn available_indices(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.rate_limit.can_accept())
            .map(|(i, _)| i)
            .collect()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl Provider for BalancedProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Custom("balanced".into())
    }

    async fn stream(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        _model: &ModelInfo,
        _auth: &AuthProfile,
        event_tx: mpsc::UnboundedSender<StreamDelta>,
    ) -> SoulResult<Message> {
        // TODO: extract intent from system prompt or messages
        // For now, use no intent (pure load balancing)
        //
        // Rate limit retry: on 429/overload, sleep and retry up to 3 times before
        // exhausting. This converts rate-limit errors into pauses rather than crashes,
        // which is essential when using a single-provider config.
        let slot_attempts = self.slots.len().min(5);
        // Additional retries for rate-limited scenarios (sleep-and-retry)
        let mut rate_limit_retries = 0;
        const MAX_RATE_LIMIT_RETRIES: usize = 3;
        const RATE_LIMIT_SLEEP_SECS: u64 = 65; // Slightly over 1 minute to let TPM window reset

        for _attempt in 0..slot_attempts {
            let idx = match self.select(None) {
                Ok(idx) => idx,
                Err(_) => {
                    // All slots in cooldown — if we have rate limit retries left, sleep and try again
                    if rate_limit_retries < MAX_RATE_LIMIT_RETRIES {
                        rate_limit_retries += 1;
                        tracing::warn!(
                            retry = rate_limit_retries,
                            sleep_secs = RATE_LIMIT_SLEEP_SECS,
                            "Balanced: all slots in cooldown, sleeping for rate limit reset"
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(RATE_LIMIT_SLEEP_SECS)).await;
                        // Clear cooldowns after sleep so slots become available again
                        for slot in &self.slots {
                            slot.rate_limit.clear_cooldown();
                        }
                        continue;
                    }
                    return Err(SoulError::FailoverExhausted { attempts: slot_attempts });
                }
            };
            let slot = &self.slots[idx];

            tracing::info!(
                provider = %slot.name,
                model = %slot.model.id,
                "Balanced: routing request"
            );

            let stream_future = slot
                .provider
                .stream(messages, system, tools, &slot.model, &slot.auth, event_tx.clone());

            // Timeout guard: prevent infinite hangs from stalled LLM connections.
            // Individual providers have reqwest timeouts, but this is a defense-in-depth
            // layer at the routing level.
            match tokio::time::timeout(super::LLM_REQUEST_TIMEOUT, stream_future).await {
                Ok(Ok(msg)) => {
                    let tokens = msg
                        .usage
                        .as_ref()
                        .map(|u| u.total() as u64)
                        .unwrap_or(0);
                    slot.rate_limit.record_success(tokens);
                    return Ok(msg);
                }
                Ok(Err(SoulError::RateLimited { retry_after_ms, .. })) => {
                    let sleep_ms = retry_after_ms.max(RATE_LIMIT_SLEEP_SECS * 1000);
                    slot.rate_limit.set_cooldown(sleep_ms / 1000);
                    slot.rate_limit.record_failure("rate_limited");
                    if rate_limit_retries < MAX_RATE_LIMIT_RETRIES {
                        rate_limit_retries += 1;
                        tracing::warn!(
                            provider = %slot.name,
                            retry = rate_limit_retries,
                            sleep_ms,
                            "Balanced: rate limited, sleeping before retry"
                        );
                        tokio::time::sleep(std::time::Duration::from_millis(sleep_ms)).await;
                        slot.rate_limit.clear_cooldown();
                    }
                    continue;
                }
                Ok(Err(e)) => {
                    slot.rate_limit.record_failure(&e.to_string());
                    slot.rate_limit.set_cooldown(30);
                    continue;
                }
                Err(_elapsed) => {
                    tracing::error!(
                        provider = %slot.name,
                        model = %slot.model.id,
                        timeout_secs = super::LLM_REQUEST_TIMEOUT.as_secs(),
                        "LLM request timed out"
                    );
                    slot.rate_limit.record_failure("timeout");
                    slot.rate_limit.set_cooldown(60);
                    continue;
                }
            }
        }

        Err(SoulError::FailoverExhausted {
            attempts: slot_attempts,
        })
    }

    /// Intent-aware completion — pass intent via model.id field as "intent:reasoning" etc.
    async fn count_tokens(
        &self,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        _model: &ModelInfo,
        _auth: &AuthProfile,
    ) -> SoulResult<usize> {
        // Use first available provider for token counting
        if let Ok(idx) = self.select(None) {
            let slot = &self.slots[idx];
            return slot
                .provider
                .count_tokens(messages, system, tools, &slot.model, &slot.auth)
                .await;
        }
        // Fallback: estimate
        Ok(messages.iter().map(|m| m.estimate_tokens()).sum())
    }

    async fn probe(&self, _model: &ModelInfo, _auth: &AuthProfile) -> SoulResult<ProbeResult> {
        let available = self.available_indices().len();
        Ok(ProbeResult {
            healthy: available > 0,
            rate_limit_remaining: None,
            rate_limit_utilization: None,
        })
    }
}

/// Convenience: create a BalancedProvider with intent-aware stream method
impl BalancedProvider {
    /// Stream with explicit intent — picks the best provider for this intent
    pub async fn stream_with_intent(
        &self,
        intent: &Intent,
        messages: &[Message],
        system: &str,
        tools: &[ToolDefinition],
        event_tx: mpsc::UnboundedSender<StreamDelta>,
    ) -> SoulResult<Message> {
        let max_attempts = self.slots.len().min(5);

        // Resolve model override from intent mapping
        for _attempt in 0..max_attempts {
            let idx = self.select(Some(intent))?;
            let slot = &self.slots[idx];

            // Check if intent mapping overrides the model
            let model = if let Some(mapping) = self.intents.get(intent) {
                if let Some(override_model_id) = mapping.models.get(&slot.name) {
                    let mut m = slot.model.clone();
                    m.id = override_model_id.clone();
                    m
                } else {
                    slot.model.clone()
                }
            } else {
                slot.model.clone()
            };

            tracing::info!(
                provider = %slot.name,
                model = %model.id,
                intent = %intent,
                "Balanced: routing intent-aware request"
            );

            match slot
                .provider
                .stream(messages, system, tools, &model, &slot.auth, event_tx.clone())
                .await
            {
                Ok(msg) => {
                    let tokens = msg
                        .usage
                        .as_ref()
                        .map(|u| u.total() as u64)
                        .unwrap_or(0);
                    slot.rate_limit.record_success(tokens);
                    return Ok(msg);
                }
                Err(SoulError::RateLimited { .. }) => {
                    slot.rate_limit.set_cooldown(60);
                    slot.rate_limit.record_failure("rate_limited");
                    continue;
                }
                Err(e) => {
                    slot.rate_limit.record_failure(&e.to_string());
                    slot.rate_limit.set_cooldown(30);
                    continue;
                }
            }
        }

        Err(SoulError::FailoverExhausted {
            attempts: max_attempts,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    struct FakeProvider {
        name: String,
        response: StdMutex<Option<Message>>,
    }

    impl FakeProvider {
        fn new(name: &str, text: &str) -> Self {
            Self {
                name: name.into(),
                response: StdMutex::new(Some(Message::assistant(text))),
            }
        }
        fn failing(name: &str) -> Self {
            Self {
                name: name.into(),
                response: StdMutex::new(None),
            }
        }
    }

    #[async_trait::async_trait]
    impl Provider for FakeProvider {
        fn kind(&self) -> ProviderKind {
            ProviderKind::Custom(self.name.clone())
        }

        async fn stream(
            &self,
            _messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _model: &ModelInfo,
            _auth: &AuthProfile,
            _event_tx: mpsc::UnboundedSender<StreamDelta>,
        ) -> SoulResult<Message> {
            let resp = self.response.lock().unwrap();
            match resp.as_ref() {
                Some(msg) => Ok(msg.clone()),
                None => Err(SoulError::Provider("fake failure".into())),
            }
        }

        async fn count_tokens(
            &self,
            messages: &[Message],
            _system: &str,
            _tools: &[ToolDefinition],
            _model: &ModelInfo,
            _auth: &AuthProfile,
        ) -> SoulResult<usize> {
            Ok(messages.iter().map(|m| m.estimate_tokens()).sum())
        }

        async fn probe(
            &self,
            _model: &ModelInfo,
            _auth: &AuthProfile,
        ) -> SoulResult<ProbeResult> {
            Ok(ProbeResult {
                healthy: true,
                rate_limit_remaining: None,
                rate_limit_utilization: None,
            })
        }
    }

    fn test_model(name: &str) -> ModelInfo {
        ModelInfo {
            id: name.into(),
            provider: ProviderKind::Custom("test".into()),
            context_window: 128_000,
            max_output_tokens: 4096,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        }
    }

    fn test_auth() -> AuthProfile {
        AuthProfile::new(ProviderKind::Custom("test".into()), "")
    }

    #[test]
    fn balanced_status() {
        let mut bp = BalancedProvider::new(Strategy::RoundRobin);
        bp.add_slot(
            "groq",
            Arc::new(FakeProvider::new("groq", "hi")),
            test_model("llama-70b"),
            test_auth(),
            30,
            RateLimitTracker::new(Some(30), None, None),
        );
        bp.add_slot(
            "gemini",
            Arc::new(FakeProvider::new("gemini", "hello")),
            test_model("gemini-flash"),
            test_auth(),
            40,
            RateLimitTracker::new(Some(15), Some(1500), None),
        );

        let status = bp.status();
        assert_eq!(status.total_slots, 2);
        assert_eq!(status.available_slots, 2);
        assert_eq!(status.slots[0].name, "groq");
        assert_eq!(status.slots[1].name, "gemini");
    }

    #[test]
    fn round_robin_cycles() {
        let mut bp = BalancedProvider::new(Strategy::RoundRobin);
        bp.add_slot("a", Arc::new(FakeProvider::new("a", "")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());
        bp.add_slot("b", Arc::new(FakeProvider::new("b", "")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());

        assert_eq!(bp.select(None).unwrap(), 0);
        assert_eq!(bp.select(None).unwrap(), 1);
        assert_eq!(bp.select(None).unwrap(), 0);
    }

    #[test]
    fn failover_uses_first() {
        let mut bp = BalancedProvider::new(Strategy::Failover);
        bp.add_slot("primary", Arc::new(FakeProvider::new("p", "")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());
        bp.add_slot("backup", Arc::new(FakeProvider::new("b", "")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());

        assert_eq!(bp.select(None).unwrap(), 0);
        assert_eq!(bp.select(None).unwrap(), 0);
    }

    #[test]
    fn skips_rate_limited() {
        let mut bp = BalancedProvider::new(Strategy::Failover);
        let rl = RateLimitTracker::new(Some(1), None, None);
        rl.record_success(0); // exhaust
        bp.add_slot("limited", Arc::new(FakeProvider::new("l", "")), test_model("m"), test_auth(), 50, rl);
        bp.add_slot("open", Arc::new(FakeProvider::new("o", "")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());

        assert_eq!(bp.select(None).unwrap(), 1);
    }

    #[test]
    fn intent_prefers_mapped_providers() {
        let mut bp = BalancedProvider::new(Strategy::Failover);
        bp.add_slot("groq", Arc::new(FakeProvider::new("g", "")), test_model("llama"), test_auth(), 10, RateLimitTracker::unlimited());
        bp.add_slot("gemini", Arc::new(FakeProvider::new("gem", "")), test_model("gemini"), test_auth(), 10, RateLimitTracker::unlimited());

        bp.map_intent(
            Intent::Reasoning,
            IntentMapping {
                preferred: vec!["gemini".into()],
                models: HashMap::new(),
            },
        );

        // Without intent → first available
        assert_eq!(bp.select(None).unwrap(), 0);
        // With reasoning intent → gemini preferred
        assert_eq!(bp.select(Some(&Intent::Reasoning)).unwrap(), 1);
    }

    #[test]
    fn weighted_distributes() {
        let mut bp = BalancedProvider::new(Strategy::Weighted);
        bp.add_slot("heavy", Arc::new(FakeProvider::new("h", "")), test_model("m"), test_auth(), 90, RateLimitTracker::unlimited());
        bp.add_slot("light", Arc::new(FakeProvider::new("l", "")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());

        let mut counts = [0u32; 2];
        for _ in 0..100 {
            counts[bp.select(None).unwrap()] += 1;
        }
        assert!(counts[0] > 80);
        assert!(counts[1] < 20);
    }

    #[test]
    fn no_providers_returns_error() {
        let bp = BalancedProvider::new(Strategy::RoundRobin);
        assert!(bp.select(None).is_err());
    }

    #[tokio::test]
    async fn stream_failover_on_error() {
        let mut bp = BalancedProvider::new(Strategy::Failover);
        bp.add_slot("bad", Arc::new(FakeProvider::failing("bad")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());
        bp.add_slot("good", Arc::new(FakeProvider::new("good", "success")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());

        let (tx, _rx) = mpsc::unbounded_channel();
        let msg = bp
            .stream(
                &[Message::user("hi")],
                "system",
                &[],
                &test_model("m"),
                &test_auth(),
                tx,
            )
            .await
            .unwrap();

        assert_eq!(msg.text_content(), "success");
    }

    #[tokio::test]
    async fn stream_with_intent_selects_correctly() {
        let mut bp = BalancedProvider::new(Strategy::Failover);
        bp.add_slot("groq", Arc::new(FakeProvider::new("groq", "fast")), test_model("llama"), test_auth(), 10, RateLimitTracker::unlimited());
        bp.add_slot("gemini", Arc::new(FakeProvider::new("gemini", "smart")), test_model("gemini"), test_auth(), 10, RateLimitTracker::unlimited());

        bp.map_intent(
            Intent::Reasoning,
            IntentMapping {
                preferred: vec!["gemini".into()],
                models: HashMap::new(),
            },
        );

        let (tx, _rx) = mpsc::unbounded_channel();
        let msg = bp
            .stream_with_intent(
                &Intent::Reasoning,
                &[Message::user("think hard")],
                "system",
                &[],
                tx,
            )
            .await
            .unwrap();

        assert_eq!(msg.text_content(), "smart");
    }

    #[tokio::test]
    async fn probe_healthy_when_available() {
        let mut bp = BalancedProvider::new(Strategy::RoundRobin);
        bp.add_slot("a", Arc::new(FakeProvider::new("a", "")), test_model("m"), test_auth(), 10, RateLimitTracker::unlimited());

        let result = bp.probe(&test_model("m"), &test_auth()).await.unwrap();
        assert!(result.healthy);
    }
}
