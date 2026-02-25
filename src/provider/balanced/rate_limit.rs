use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use chrono::{DateTime, Utc};
use serde::Serialize;

/// Runtime rate limit tracking for a single provider slot
pub struct RateLimitTracker {
    /// Config limits
    pub rpm_limit: Option<u32>,
    pub rpd_limit: Option<u32>,
    pub tpm_limit: Option<u64>,
    /// Counters
    requests_this_minute: AtomicU64,
    requests_today: AtomicU64,
    tokens_this_minute: AtomicU64,
    total_requests: AtomicU64,
    total_failures: AtomicU64,
    /// Windows
    minute_start: Mutex<DateTime<Utc>>,
    day_start: Mutex<DateTime<Utc>>,
    /// Cooldown
    cooldown_until: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RateLimitStatus {
    pub rpm_limit: Option<u32>,
    pub rpm_used: u64,
    pub rpd_limit: Option<u32>,
    pub rpd_used: u64,
    pub tpm_limit: Option<u64>,
    pub tpm_used: u64,
    pub total_requests: u64,
    pub total_failures: u64,
    pub in_cooldown: bool,
    pub last_error: Option<String>,
}

impl RateLimitTracker {
    pub fn new(rpm: Option<u32>, rpd: Option<u32>, tpm: Option<u64>) -> Self {
        let now = Utc::now();
        Self {
            rpm_limit: rpm,
            rpd_limit: rpd,
            tpm_limit: tpm,
            requests_this_minute: AtomicU64::new(0),
            requests_today: AtomicU64::new(0),
            tokens_this_minute: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            minute_start: Mutex::new(now),
            day_start: Mutex::new(now),
            cooldown_until: Mutex::new(None),
            last_error: Mutex::new(None),
        }
    }

    /// No limits — for local providers like Ollama
    pub fn unlimited() -> Self {
        Self::new(None, None, None)
    }

    /// Check if this provider can accept a request right now
    pub fn can_accept(&self) -> bool {
        // Check cooldown
        if let Ok(guard) = self.cooldown_until.lock() {
            if let Some(until) = *guard {
                if Utc::now() < until {
                    return false;
                }
            }
        }

        self.reset_windows_if_needed();

        if let Some(rpm) = self.rpm_limit {
            if self.requests_this_minute.load(Ordering::Relaxed) >= rpm as u64 {
                return false;
            }
        }
        if let Some(rpd) = self.rpd_limit {
            if self.requests_today.load(Ordering::Relaxed) >= rpd as u64 {
                return false;
            }
        }
        if let Some(tpm) = self.tpm_limit {
            if self.tokens_this_minute.load(Ordering::Relaxed) >= tpm {
                return false;
            }
        }
        true
    }

    pub fn record_success(&self, tokens: u64) {
        self.reset_windows_if_needed();
        self.requests_this_minute.fetch_add(1, Ordering::Relaxed);
        self.requests_today.fetch_add(1, Ordering::Relaxed);
        self.tokens_this_minute.fetch_add(tokens, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_failure(&self, error: &str) {
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        if let Ok(mut guard) = self.last_error.lock() {
            *guard = Some(error.to_string());
        }
    }

    pub fn set_cooldown(&self, secs: u64) {
        if let Ok(mut guard) = self.cooldown_until.lock() {
            *guard = Some(Utc::now() + chrono::Duration::seconds(secs as i64));
        }
    }

    /// Clear any active cooldown, allowing the slot to accept requests again.
    /// Used after sleeping for rate limit reset.
    pub fn clear_cooldown(&self) {
        if let Ok(mut guard) = self.cooldown_until.lock() {
            *guard = None;
        }
    }

    pub fn status(&self) -> RateLimitStatus {
        self.reset_windows_if_needed();
        RateLimitStatus {
            rpm_limit: self.rpm_limit,
            rpm_used: self.requests_this_minute.load(Ordering::Relaxed),
            rpd_limit: self.rpd_limit,
            rpd_used: self.requests_today.load(Ordering::Relaxed),
            tpm_limit: self.tpm_limit,
            tpm_used: self.tokens_this_minute.load(Ordering::Relaxed),
            total_requests: self.total_requests.load(Ordering::Relaxed),
            total_failures: self.total_failures.load(Ordering::Relaxed),
            in_cooldown: self
                .cooldown_until
                .lock()
                .ok()
                .and_then(|g| *g)
                .map(|u| Utc::now() < u)
                .unwrap_or(false),
            last_error: self.last_error.lock().ok().and_then(|g| g.clone()),
        }
    }

    fn reset_windows_if_needed(&self) {
        let now = Utc::now();
        if let Ok(mut start) = self.minute_start.lock() {
            if (now - *start).num_seconds() >= 60 {
                *start = now;
                self.requests_this_minute.store(0, Ordering::Relaxed);
                self.tokens_this_minute.store(0, Ordering::Relaxed);
            }
        }
        if let Ok(mut start) = self.day_start.lock() {
            if (now - *start).num_hours() >= 24 {
                *start = now;
                self.requests_today.store(0, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unlimited_always_accepts() {
        let rl = RateLimitTracker::unlimited();
        assert!(rl.can_accept());
        rl.record_success(1_000_000);
        assert!(rl.can_accept());
    }

    #[test]
    fn rpm_blocks_when_exhausted() {
        let rl = RateLimitTracker::new(Some(2), None, None);
        assert!(rl.can_accept());
        rl.record_success(0);
        assert!(rl.can_accept());
        rl.record_success(0);
        assert!(!rl.can_accept());
    }

    #[test]
    fn cooldown_blocks() {
        let rl = RateLimitTracker::unlimited();
        assert!(rl.can_accept());
        rl.set_cooldown(60);
        assert!(!rl.can_accept());
    }

    #[test]
    fn failure_tracking() {
        let rl = RateLimitTracker::unlimited();
        rl.record_failure("connection refused");
        let status = rl.status();
        assert_eq!(status.total_failures, 1);
        assert_eq!(status.last_error, Some("connection refused".into()));
    }

    #[test]
    fn status_snapshot() {
        let rl = RateLimitTracker::new(Some(30), Some(1500), Some(1_000_000));
        rl.record_success(100);
        rl.record_success(200);
        let s = rl.status();
        assert_eq!(s.rpm_used, 2);
        assert_eq!(s.rpd_used, 2);
        assert_eq!(s.tpm_used, 300);
        assert_eq!(s.total_requests, 2);
    }
}
