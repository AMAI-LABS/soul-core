/// Harness store — tracks tool failures and synthesized harness candidates.
///
/// Thread-safe store for failure records and harness candidates per tool.
/// Used by ActionVerifierHook to decide when to synthesize a new harness
/// and which candidate to use for validation.
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::thompson::ThompsonSampler;

/// A recorded tool failure.
#[derive(Debug, Clone)]
pub struct FailureRecord {
    /// Tool name that failed.
    pub tool_name: String,
    /// Arguments passed to the tool at failure.
    pub arguments: serde_json::Value,
    /// Error message from the tool.
    pub error: String,
    /// Timestamp (seconds since epoch).
    pub timestamp: u64,
}

impl FailureRecord {
    pub fn new(tool_name: impl Into<String>, arguments: serde_json::Value, error: impl Into<String>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Self {
            tool_name: tool_name.into(),
            arguments,
            error: error.into(),
            timestamp,
        }
    }
}

/// A synthesized harness candidate for a specific tool.
///
/// The harness is a shell script that validates tool arguments.
/// Exit 0 = valid (allow the call). Non-zero = invalid (block + report error).
/// Args are passed as env vars: AMAI_ARG_<NAME>=<value>.
#[derive(Debug, Clone)]
pub struct HarnessCandidate {
    /// Shell script that validates args. Exit 0 = pass, non-zero = reject.
    pub script: String,
    /// Human-readable description of what this harness checks.
    pub description: String,
    /// Number of times this candidate allowed a call that succeeded.
    pub success_count: usize,
    /// Number of times this candidate's verdict was wrong.
    pub failure_count: usize,
}

impl HarnessCandidate {
    pub fn new(script: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            script: script.into(),
            description: description.into(),
            success_count: 0,
            failure_count: 0,
        }
    }

    /// Run this harness against given arguments. Returns Ok(()) if valid, Err(msg) if invalid.
    pub async fn validate(
        &self,
        arguments: &serde_json::Value,
    ) -> Result<(), String> {
        let mut cmd = tokio::process::Command::new("bash");
        cmd.arg("-c").arg(&self.script);

        if let Some(obj) = arguments.as_object() {
            for (k, v) in obj {
                let env_key = format!("AMAI_ARG_{}", k.to_uppercase().replace('-', "_"));
                let env_val = match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                cmd.env(env_key, env_val);
            }
        }

        let output = cmd.output().await.map_err(|e| format!("Harness exec error: {e}"))?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            Err(if stderr.is_empty() {
                format!("Harness rejected call (exit {})", output.status.code().unwrap_or(-1))
            } else {
                stderr.trim().to_string()
            })
        }
    }
}

/// Per-tool harness state.
struct ToolHarnessState {
    failures: Vec<FailureRecord>,
    candidates: Vec<HarnessCandidate>,
    sampler: ThompsonSampler,
}

impl ToolHarnessState {
    fn new() -> Self {
        Self {
            failures: Vec::new(),
            candidates: Vec::new(),
            sampler: ThompsonSampler::new(0),
        }
    }
}

/// Thread-safe store for all harness state across tools.
#[derive(Clone)]
pub struct HarnessStore {
    inner: Arc<RwLock<HashMap<String, ToolHarnessState>>>,
}

impl HarnessStore {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Record a tool failure.
    pub fn record_failure(&self, record: FailureRecord) {
        let mut store = self.inner.write().unwrap();
        store.entry(record.tool_name.clone()).or_insert_with(ToolHarnessState::new).failures.push(record);
    }

    /// Get failure count for a tool.
    pub fn failure_count(&self, tool_name: &str) -> usize {
        self.inner.read().unwrap()
            .get(tool_name)
            .map(|s| s.failures.len())
            .unwrap_or(0)
    }

    /// Get recent failures for a tool (for harness synthesis prompt).
    pub fn recent_failures(&self, tool_name: &str, limit: usize) -> Vec<FailureRecord> {
        self.inner.read().unwrap()
            .get(tool_name)
            .map(|s| s.failures.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default()
    }

    /// Add a new harness candidate for a tool.
    pub fn add_candidate(&self, tool_name: &str, candidate: HarnessCandidate) {
        let mut store = self.inner.write().unwrap();
        let state = store.entry(tool_name.to_string()).or_insert_with(ToolHarnessState::new);
        state.candidates.push(candidate);
        state.sampler.add_arm();
    }

    /// Get the number of harness candidates for a tool.
    pub fn candidate_count(&self, tool_name: &str) -> usize {
        self.inner.read().unwrap()
            .get(tool_name)
            .map(|s| s.candidates.len())
            .unwrap_or(0)
    }

    /// Select a candidate via Thompson sampling. Returns (index, cloned candidate) or None.
    pub fn select_candidate(&self, tool_name: &str) -> Option<(usize, HarnessCandidate)> {
        let mut store = self.inner.write().unwrap();
        let state = store.get_mut(tool_name)?;
        if state.candidates.is_empty() {
            return None;
        }
        let idx = state.sampler.select();
        Some((idx, state.candidates[idx].clone()))
    }

    /// Record that the selected candidate (by index) allowed a call that succeeded.
    pub fn record_candidate_success(&self, tool_name: &str, candidate_idx: usize) {
        let mut store = self.inner.write().unwrap();
        if let Some(state) = store.get_mut(tool_name) {
            state.sampler.record_success(candidate_idx);
            if let Some(c) = state.candidates.get_mut(candidate_idx) {
                c.success_count += 1;
            }
        }
    }

    /// Record that the selected candidate (by index) made a wrong verdict.
    pub fn record_candidate_failure(&self, tool_name: &str, candidate_idx: usize) {
        let mut store = self.inner.write().unwrap();
        if let Some(state) = store.get_mut(tool_name) {
            state.sampler.record_failure(candidate_idx);
            if let Some(c) = state.candidates.get_mut(candidate_idx) {
                c.failure_count += 1;
            }
        }
    }

    /// Get the best candidate (by mean) for a tool, for reporting.
    pub fn best_candidate(&self, tool_name: &str) -> Option<HarnessCandidate> {
        let store = self.inner.read().unwrap();
        let state = store.get(tool_name)?;
        let best_idx = state.sampler.best_arm()?;
        state.candidates.get(best_idx).cloned()
    }

    /// Get all failure records across all tools (for cross-session persistence).
    pub fn all_failures(&self) -> Vec<FailureRecord> {
        self.inner.read().unwrap()
            .values()
            .flat_map(|s| s.failures.iter().cloned())
            .collect()
    }

    /// Clear failures for a tool (after harness synthesis converges).
    pub fn clear_failures(&self, tool_name: &str) {
        let mut store = self.inner.write().unwrap();
        if let Some(state) = store.get_mut(tool_name) {
            state.failures.clear();
        }
    }
}

impl Default for HarnessStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn store_records_failures() {
        let store = HarnessStore::new();
        assert_eq!(store.failure_count("bash"), 0);

        store.record_failure(FailureRecord::new("bash", json!({"command": "rm -rf /"}), "permission denied"));
        store.record_failure(FailureRecord::new("bash", json!({"command": "cat /etc/passwd"}), "blocked"));
        assert_eq!(store.failure_count("bash"), 2);
        assert_eq!(store.failure_count("read"), 0);
    }

    #[test]
    fn store_recent_failures_returns_last_n() {
        let store = HarnessStore::new();
        for i in 0..10 {
            store.record_failure(FailureRecord::new("bash", json!({}), format!("error {i}")));
        }
        let recent = store.recent_failures("bash", 3);
        assert_eq!(recent.len(), 3);
        // Should be the last 3 (most recent first due to rev())
        assert!(recent[0].error.contains("9") || recent[0].error.contains("8"));
    }

    #[test]
    fn store_adds_and_selects_candidates() {
        let store = HarnessStore::new();
        store.add_candidate("bash", HarnessCandidate::new("exit 0", "Always allow"));
        store.add_candidate("bash", HarnessCandidate::new("exit 1", "Always block"));

        assert_eq!(store.candidate_count("bash"), 2);
        let selected = store.select_candidate("bash");
        assert!(selected.is_some());
    }

    #[test]
    fn store_thompson_favors_better_candidate() {
        let store = HarnessStore::new();
        store.add_candidate("bash", HarnessCandidate::new("exit 0", "arm 0"));
        store.add_candidate("bash", HarnessCandidate::new("exit 0", "arm 1"));

        // Give arm 1 many successes
        for _ in 0..20 {
            store.record_candidate_success("bash", 1);
        }
        for _ in 0..5 {
            store.record_candidate_failure("bash", 0);
        }

        let mut arm1_wins = 0;
        for _ in 0..50 {
            if let Some((idx, _)) = store.select_candidate("bash") {
                if idx == 1 {
                    arm1_wins += 1;
                }
            }
        }
        assert!(arm1_wins > 25, "Arm 1 should dominate, got {arm1_wins}/50");
    }

    #[tokio::test]
    async fn harness_candidate_validate_pass() {
        let candidate = HarnessCandidate::new("exit 0", "Always pass");
        assert!(candidate.validate(&json!({"x": "1"})).await.is_ok());
    }

    #[tokio::test]
    async fn harness_candidate_validate_fail() {
        let candidate = HarnessCandidate::new("echo 'bad args' >&2; exit 1", "Always fail");
        let result = candidate.validate(&json!({})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("bad args"));
    }

    #[tokio::test]
    async fn harness_candidate_validate_with_args() {
        // Only allow if AMAI_ARG_VALUE is a number
        let candidate = HarnessCandidate::new(
            r#"if ! echo "$AMAI_ARG_VALUE" | grep -qE '^[0-9]+$'; then echo "not a number" >&2; exit 1; fi"#,
            "Require numeric value",
        );

        assert!(candidate.validate(&json!({"value": "42"})).await.is_ok());
        let err = candidate.validate(&json!({"value": "hello"})).await;
        assert!(err.is_err());
        assert!(err.unwrap_err().contains("not a number"));
    }

    #[test]
    fn store_clear_failures() {
        let store = HarnessStore::new();
        store.record_failure(FailureRecord::new("bash", json!({}), "err"));
        assert_eq!(store.failure_count("bash"), 1);
        store.clear_failures("bash");
        assert_eq!(store.failure_count("bash"), 0);
    }

    #[test]
    fn store_all_failures_across_tools() {
        let store = HarnessStore::new();
        store.record_failure(FailureRecord::new("bash", json!({}), "err1"));
        store.record_failure(FailureRecord::new("read", json!({}), "err2"));
        let all = store.all_failures();
        assert_eq!(all.len(), 2);
    }
}
