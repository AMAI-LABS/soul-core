/// Thompson sampling for harness candidate selection.
///
/// Each candidate maintains a Beta(α, β) distribution:
/// - α (successes): incremented when candidate correctly allows a valid call
/// - β (failures): incremented when candidate blocks a valid call or misses an error
///
/// Selection: sample a value from each candidate's Beta distribution,
/// pick the candidate with the highest sample. This naturally balances
/// exploration (try untested candidates) vs exploitation (use best known).
///
/// Reference: Thompson (1933), "On the Likelihood that One Unknown Probability
/// Exceeds Another in View of the Evidence of Two Samples"

/// A Beta distribution sampler using the Johnk method (exact, no external deps).
///
/// Johnk's method: sample U ~ Uniform, V ~ Uniform repeatedly until U^(1/α) + V^(1/β) ≤ 1,
/// then return U^(1/α) / (U^(1/α) + V^(1/β)).
///
/// For α=1, β=1 (uniform prior), this reduces to a simple Uniform sample.
fn sample_beta(alpha: f64, beta: f64, rng_state: &mut u64) -> f64 {
    if alpha <= 0.0 || beta <= 0.0 {
        return 0.5;
    }

    // Simple LCG for reproducible-ish sampling without external deps
    let mut next = || -> f64 {
        *rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = ((*rng_state >> 33) as u32) as f64;
        bits / (u32::MAX as f64)
    };

    // For alpha=1 and beta=1, just return uniform (Beta(1,1) = Uniform(0,1))
    if (alpha - 1.0).abs() < 1e-9 && (beta - 1.0).abs() < 1e-9 {
        return next();
    }

    // Johnk's method (works well for alpha, beta ≥ 1)
    let inv_a = 1.0 / alpha;
    let inv_b = 1.0 / beta;
    for _ in 0..1000 {
        let u = next();
        let v = next();
        let ua = u.powf(inv_a);
        let vb = v.powf(inv_b);
        let sum = ua + vb;
        if sum <= 1.0 {
            return ua / sum;
        }
    }
    // Fallback: return mean
    alpha / (alpha + beta)
}

/// Per-candidate Beta distribution parameters.
#[derive(Debug, Clone)]
pub struct BetaArms {
    /// Number of observed successes (correctly allowed valid calls).
    pub alpha: f64,
    /// Number of observed failures (missed errors or false blocks).
    pub beta: f64,
}

impl BetaArms {
    /// Uninformative prior: Beta(1,1) = Uniform(0,1).
    pub fn new() -> Self {
        Self { alpha: 1.0, beta: 1.0 }
    }

    pub fn record_success(&mut self) {
        self.alpha += 1.0;
    }

    pub fn record_failure(&mut self) {
        self.beta += 1.0;
    }

    /// Expected value (mean of Beta distribution): α / (α + β).
    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Sample from this Beta distribution.
    pub fn sample(&self, rng_state: &mut u64) -> f64 {
        sample_beta(self.alpha, self.beta, rng_state)
    }
}

impl Default for BetaArms {
    fn default() -> Self {
        Self::new()
    }
}

/// Thompson sampler over N harness candidates.
///
/// Maintains one BetaArms per candidate index.
/// `select()` returns the index of the candidate with highest sample.
pub struct ThompsonSampler {
    arms: Vec<BetaArms>,
    rng_state: u64,
}

impl ThompsonSampler {
    /// Create a sampler with `n` arms, each starting at Beta(1,1).
    pub fn new(n: usize) -> Self {
        // Seed from current time (good enough for this use case)
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(12345);
        Self {
            arms: (0..n).map(|_| BetaArms::new()).collect(),
            rng_state: seed,
        }
    }

    /// Add a new arm (new candidate harness).
    pub fn add_arm(&mut self) {
        self.arms.push(BetaArms::new());
    }

    /// Select the arm with the highest Thompson sample. Returns the arm index.
    pub fn select(&mut self) -> usize {
        if self.arms.is_empty() {
            return 0;
        }
        self.arms
            .iter()
            .enumerate()
            .map(|(i, arm)| (i, arm.sample(&mut self.rng_state)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Record success for arm at `index`.
    pub fn record_success(&mut self, index: usize) {
        if let Some(arm) = self.arms.get_mut(index) {
            arm.record_success();
        }
    }

    /// Record failure for arm at `index`.
    pub fn record_failure(&mut self, index: usize) {
        if let Some(arm) = self.arms.get_mut(index) {
            arm.record_failure();
        }
    }

    /// Number of arms.
    pub fn len(&self) -> usize {
        self.arms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.arms.is_empty()
    }

    /// Expected success rate for arm at `index`.
    pub fn mean(&self, index: usize) -> f64 {
        self.arms.get(index).map(|a| a.mean()).unwrap_or(0.5)
    }

    /// Best arm by mean (for reporting/convergence check).
    pub fn best_arm(&self) -> Option<usize> {
        self.arms
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.mean().partial_cmp(&b.1.mean()).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beta_arms_start_at_uniform() {
        let arm = BetaArms::new();
        assert_eq!(arm.alpha, 1.0);
        assert_eq!(arm.beta, 1.0);
        assert!((arm.mean() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn beta_arms_update_correctly() {
        let mut arm = BetaArms::new();
        arm.record_success();
        arm.record_success();
        arm.record_failure();
        // alpha=3, beta=2 → mean=3/5=0.6
        assert!((arm.mean() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn thompson_sampler_selects_best_arm() {
        let mut sampler = ThompsonSampler::new(3);
        // Arm 1 has many successes → should be selected most often
        for _ in 0..20 {
            sampler.record_success(1);
        }
        for _ in 0..5 {
            sampler.record_failure(0);
            sampler.record_failure(2);
        }

        // Run 100 selections — arm 1 should win most
        let mut counts = [0usize; 3];
        for _ in 0..100 {
            let selected = sampler.select();
            counts[selected] += 1;
        }
        assert!(
            counts[1] > counts[0],
            "Arm 1 ({}) should dominate arm 0 ({})",
            counts[1],
            counts[0]
        );
        assert!(
            counts[1] > counts[2],
            "Arm 1 ({}) should dominate arm 2 ({})",
            counts[1],
            counts[2]
        );
    }

    #[test]
    fn thompson_sampler_add_arm() {
        let mut sampler = ThompsonSampler::new(2);
        assert_eq!(sampler.len(), 2);
        sampler.add_arm();
        assert_eq!(sampler.len(), 3);
    }

    #[test]
    fn thompson_sampler_best_arm() {
        let mut sampler = ThompsonSampler::new(3);
        for _ in 0..10 {
            sampler.record_success(2);
        }
        assert_eq!(sampler.best_arm(), Some(2));
    }

    #[test]
    fn thompson_empty_sampler() {
        let mut sampler = ThompsonSampler::new(0);
        assert_eq!(sampler.select(), 0); // graceful
        assert!(sampler.best_arm().is_none());
    }

    #[test]
    fn sample_beta_uniform_prior() {
        let mut rng = 42u64;
        // Beta(1,1) should give values in [0,1]
        for _ in 0..100 {
            let v = sample_beta(1.0, 1.0, &mut rng);
            assert!(v >= 0.0 && v <= 1.0, "out of range: {v}");
        }
    }

    #[test]
    fn sample_beta_skewed_high() {
        let mut rng = 42u64;
        // Beta(20, 1) → heavily skewed toward 1, mean ≈ 0.95
        let samples: Vec<f64> = (0..200).map(|_| sample_beta(20.0, 1.0, &mut rng)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean > 0.8, "Mean {mean} should be >0.8 for Beta(20,1)");
    }

    #[test]
    fn sample_beta_skewed_low() {
        let mut rng = 999u64;
        // Beta(1, 20) → heavily skewed toward 0, mean ≈ 0.05
        let samples: Vec<f64> = (0..200).map(|_| sample_beta(1.0, 20.0, &mut rng)).collect();
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(mean < 0.2, "Mean {mean} should be <0.2 for Beta(1,20)");
    }
}
