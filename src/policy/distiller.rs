/// Policy distiller — observes tool calls and distills repeated patterns into policies.
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::registry::{DistilledPolicy, PolicyRegistry};

/// A single tool call observation (input + output pair).
#[derive(Debug, Clone)]
pub struct CallObservation {
    /// Tool name.
    pub tool_name: String,
    /// Arguments passed.
    pub arguments: serde_json::Value,
    /// Output content.
    pub output: String,
    /// Whether the call succeeded (no error).
    pub success: bool,
}

impl CallObservation {
    pub fn new(
        tool_name: impl Into<String>,
        arguments: serde_json::Value,
        output: impl Into<String>,
        success: bool,
    ) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments,
            output: output.into(),
            success,
        }
    }

    /// Fingerprint: `tool_name:sorted_arg_keys`.
    ///
    /// Captures the structural shape of a call (which args) without the values.
    /// This allows matching across calls with different values but the same pattern.
    pub fn fingerprint(&self) -> String {
        let mut keys: Vec<&str> = self.arguments
            .as_object()
            .map(|o| o.keys().map(|k| k.as_str()).collect())
            .unwrap_or_default();
        keys.sort_unstable();
        format!("{}:{}", self.tool_name, keys.join(","))
    }
}

/// Per-fingerprint distillation state.
struct FingerprintState {
    /// All successful observations for this fingerprint.
    successes: Vec<CallObservation>,
    /// Whether this fingerprint has been distilled.
    distilled: bool,
}

impl FingerprintState {
    fn new() -> Self {
        Self {
            successes: Vec::new(),
            distilled: false,
        }
    }
}

/// Generates a shell policy from a set of successful observations.
///
/// Strategy:
/// - If all observations have identical outputs → generate a static echo policy
/// - Otherwise → generate a dynamic policy that runs a bash template
fn generate_policy_script(observations: &[CallObservation]) -> Option<(String, String)> {
    if observations.is_empty() {
        return None;
    }

    let tool_name = &observations[0].tool_name;

    // Case 1: All outputs identical → pure static policy
    let first_output = &observations[0].output;
    let all_identical = observations.iter().all(|o| &o.output == first_output);

    if all_identical {
        // Static: always returns the same output (e.g. deterministic read)
        // Escape the output for shell
        let escaped = first_output.replace('\'', "'\\''");
        let script = format!("printf '%s' '{escaped}'");
        let description = format!(
            "Static policy for '{}': always returns identical output (seen {} times)",
            tool_name,
            observations.len()
        );
        return Some((script, description));
    }

    // Case 2: Output varies → template-based policy using arg env vars
    // Find the arg key that correlates most with output variation
    let example = &observations[0];
    if let Some(obj) = example.arguments.as_object() {
        if obj.len() == 1 {
            let key = obj.keys().next().unwrap();
            let env_key = format!("AMAI_ARG_{}", key.to_uppercase().replace('-', "_"));

            // Check if any observation shows output == arg value (identity transform)
            let is_identity = observations.iter().all(|o| {
                let arg_val = o.arguments.get(key).and_then(|v| v.as_str()).unwrap_or("");
                o.output.trim() == arg_val
            });

            if is_identity {
                let script = format!("printf '%s' \"${{{env_key}}}\"");
                let description = format!(
                    "Identity policy for '{}': output = input '{}' (seen {} times)",
                    tool_name, key, observations.len()
                );
                return Some((script, description));
            }
        }
    }

    // Case 3: Complex pattern — not distillable into a simple policy yet
    None
}

/// Distills repeated successful tool call patterns into shell policies.
///
/// Thread-safe. Call `observe()` after each tool execution.
/// When a fingerprint reaches `distillation_threshold` successes,
/// the distiller attempts to generate a policy and registers it.
#[derive(Clone)]
pub struct PolicyDistiller {
    inner: Arc<RwLock<HashMap<String, FingerprintState>>>,
    registry: PolicyRegistry,
    distillation_threshold: usize,
}

impl PolicyDistiller {
    /// Create a new distiller.
    ///
    /// `distillation_threshold`: how many identical-shaped successes before distillation.
    /// Recommended: 5. Lower = more aggressive (may distill non-deterministic patterns).
    pub fn new(registry: PolicyRegistry, distillation_threshold: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            registry,
            distillation_threshold,
        }
    }

    /// Observe a tool call result. Returns true if a new policy was distilled.
    pub fn observe(&self, observation: CallObservation) -> bool {
        if !observation.success {
            // Only track successes
            return false;
        }

        let fingerprint = observation.fingerprint();
        let mut store = self.inner.write().unwrap();
        let state = store.entry(fingerprint.clone()).or_insert_with(FingerprintState::new);

        if state.distilled {
            return false; // Already distilled, skip
        }

        state.successes.push(observation);

        if state.successes.len() >= self.distillation_threshold {
            // Attempt distillation
            if let Some((script, description)) = generate_policy_script(&state.successes) {
                let tool_name = state.successes[0].tool_name.clone();
                // Extract arg keys from fingerprint
                let arg_keys: Vec<String> = fingerprint
                    .split(':')
                    .nth(1)
                    .unwrap_or("")
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .collect();

                let policy = DistilledPolicy::new(
                    &tool_name,
                    arg_keys,
                    script,
                    description,
                    state.successes.len(),
                );

                tracing::info!(
                    tool = %tool_name,
                    fingerprint = %fingerprint,
                    observations = state.successes.len(),
                    "PolicyDistiller: distilled new policy"
                );

                self.registry.register(fingerprint, policy);
                state.distilled = true;
                return true;
            }
        }

        false
    }

    /// Number of fingerprints being tracked (including distilled).
    pub fn tracked_fingerprints(&self) -> usize {
        self.inner.read().unwrap().len()
    }

    /// Number of distilled policies.
    pub fn distilled_count(&self) -> usize {
        self.inner.read().unwrap().values().filter(|s| s.distilled).count()
    }

    /// Success count for a specific fingerprint.
    pub fn success_count(&self, fingerprint: &str) -> usize {
        self.inner.read().unwrap()
            .get(fingerprint)
            .map(|s| s.successes.len())
            .unwrap_or(0)
    }

    /// Access the underlying policy registry.
    pub fn registry(&self) -> &PolicyRegistry {
        &self.registry
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_distiller(threshold: usize) -> PolicyDistiller {
        PolicyDistiller::new(PolicyRegistry::new(), threshold)
    }

    #[test]
    fn fingerprint_includes_tool_and_sorted_keys() {
        let obs = CallObservation::new("read", json!({"b": 2, "a": 1}), "ok", true);
        // Keys should be sorted: a,b
        assert_eq!(obs.fingerprint(), "read:a,b");
    }

    #[test]
    fn fingerprint_empty_args() {
        let obs = CallObservation::new("ls", json!({}), ".", true);
        assert_eq!(obs.fingerprint(), "ls:");
    }

    #[test]
    fn observe_tracks_successes() {
        let distiller = make_distiller(5);
        let obs = CallObservation::new("bash", json!({"command": "ls"}), "file.txt", true);
        distiller.observe(obs);
        assert_eq!(distiller.success_count("bash:command"), 1);
    }

    #[test]
    fn observe_ignores_failures() {
        let distiller = make_distiller(5);
        let obs = CallObservation::new("bash", json!({"command": "ls"}), "error", false);
        distiller.observe(obs);
        assert_eq!(distiller.success_count("bash:command"), 0);
    }

    #[test]
    fn static_policy_distilled_after_threshold() {
        let distiller = make_distiller(3);

        // Same tool, same args, same output (static/deterministic)
        for _ in 0..3 {
            distiller.observe(CallObservation::new(
                "ls",
                json!({}),
                "README.md\nCargo.toml",
                true,
            ));
        }

        assert_eq!(distiller.distilled_count(), 1);
        let policy = distiller.registry().get_by_fingerprint("ls:");
        assert!(policy.is_some(), "Policy should be registered");
        let policy = policy.unwrap();
        assert!(policy.script.contains("README.md") || policy.script.contains("printf"));
    }

    #[test]
    fn no_distillation_below_threshold() {
        let distiller = make_distiller(5);
        for _ in 0..4 {
            distiller.observe(CallObservation::new("read", json!({"path": "/f"}), "content", true));
        }
        assert_eq!(distiller.distilled_count(), 0);
    }

    #[test]
    fn distillation_not_repeated() {
        let distiller = make_distiller(3);
        // First 3: distill
        for _ in 0..3 {
            distiller.observe(CallObservation::new("ls", json!({}), "file", true));
        }
        assert_eq!(distiller.distilled_count(), 1);

        // More observations shouldn't re-distill or panic
        for _ in 0..10 {
            distiller.observe(CallObservation::new("ls", json!({}), "file", true));
        }
        assert_eq!(distiller.distilled_count(), 1); // Still 1
    }

    #[test]
    fn varying_output_non_distillable() {
        let distiller = make_distiller(3);
        // Same shape but varying, complex output (not identity)
        distiller.observe(CallObservation::new("bash", json!({"command": "date"}), "Mon Jan 1", true));
        distiller.observe(CallObservation::new("bash", json!({"command": "date"}), "Tue Jan 2", true));
        distiller.observe(CallObservation::new("bash", json!({"command": "date"}), "Wed Jan 3", true));

        // Non-identity, multi-value → should NOT distill (generate_policy_script returns None)
        // (date output varies, not identity to arg)
        assert_eq!(distiller.distilled_count(), 0);
    }

    #[test]
    fn generate_policy_static() {
        let obs = vec![
            CallObservation::new("ls", json!({}), "file.txt", true),
            CallObservation::new("ls", json!({}), "file.txt", true),
            CallObservation::new("ls", json!({}), "file.txt", true),
        ];
        let result = generate_policy_script(&obs);
        assert!(result.is_some());
        let (script, desc) = result.unwrap();
        assert!(script.contains("file.txt"));
        assert!(desc.contains("static") || desc.contains("identical"));
    }

    #[test]
    fn generate_policy_identity() {
        let obs = vec![
            CallObservation::new("echo", json!({"text": "hello"}), "hello", true),
            CallObservation::new("echo", json!({"text": "world"}), "world", true),
            CallObservation::new("echo", json!({"text": "foo"}), "foo", true),
        ];
        let result = generate_policy_script(&obs);
        assert!(result.is_some());
        let (script, desc) = result.unwrap();
        assert!(script.contains("AMAI_ARG_TEXT"), "Script should use arg: {script}");
        assert!(desc.contains("identity") || desc.contains("Identity"));
    }

    #[test]
    fn generate_policy_empty_returns_none() {
        let result = generate_policy_script(&[]);
        assert!(result.is_none());
    }
}
