use crate::types::{ModelInfo, TokenUsage};

/// Returns (cost_per_input_token, cost_per_output_token) in USD for a given model.
///
/// If the model has non-zero pricing in its `ModelInfo`, those values are used.
/// Otherwise, falls back to a built-in pricing table.
pub fn cost_per_token(model_id: &str) -> (f64, f64) {
    match model_id {
        // Anthropic
        "claude-opus-4-6" | "claude-opus-4-20250514" => (0.000015, 0.000075),
        "claude-sonnet-4-5-20250929" | "claude-sonnet-4-20250514" => (0.000003, 0.000015),
        "claude-haiku-4-5-20251001" | "claude-3-5-haiku-20241022" => (0.0000008, 0.000004),
        "claude-3-5-sonnet-20241022" => (0.000003, 0.000015),

        // OpenAI
        "gpt-4o" | "gpt-4o-2024-11-20" => (0.0000025, 0.00001),
        "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => (0.00000015, 0.0000006),
        "gpt-4-turbo" => (0.00001, 0.00003),
        "o1" | "o1-2024-12-17" => (0.000015, 0.00006),
        "o1-mini" | "o1-mini-2024-09-12" => (0.000003, 0.000012),
        "o3-mini" => (0.0000011, 0.0000044),

        // Google
        "gemini-2.0-flash" => (0.0000001, 0.0000004),
        "gemini-1.5-pro" => (0.00000125, 0.000005),

        // Fallback — conservative estimate
        _ => (0.000003, 0.000015),
    }
}

/// Compute the cost of a single turn using the model's own pricing if available,
/// otherwise falling back to the built-in pricing table.
pub fn compute_cost(usage: &TokenUsage, model_info: &ModelInfo) -> f64 {
    let (input_rate, output_rate) =
        if model_info.cost_per_input_token > 0.0 || model_info.cost_per_output_token > 0.0 {
            (
                model_info.cost_per_input_token,
                model_info.cost_per_output_token,
            )
        } else {
            cost_per_token(&model_info.id)
        };

    let input_cost = usage.input_tokens as f64 * input_rate;
    let output_cost = usage.output_tokens as f64 * output_rate;
    // Cache reads are typically 10% of input cost, writes same as input
    let cache_read_cost = usage.cache_read_tokens as f64 * input_rate * 0.1;
    let cache_write_cost = usage.cache_write_tokens as f64 * input_rate * 1.25;

    input_cost + output_cost + cache_read_cost + cache_write_cost
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ProviderKind;

    fn test_model(id: &str) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            provider: ProviderKind::Anthropic,
            context_window: 200_000,
            max_output_tokens: 8192,
            supports_thinking: false,
            supports_tools: true,
            supports_images: false,
            cost_per_input_token: 0.0,
            cost_per_output_token: 0.0,
        }
    }

    #[test]
    fn known_model_pricing() {
        let (input, output) = cost_per_token("claude-opus-4-6");
        assert_eq!(input, 0.000015);
        assert_eq!(output, 0.000075);
    }

    #[test]
    fn sonnet_pricing() {
        let (input, output) = cost_per_token("claude-sonnet-4-5-20250929");
        assert_eq!(input, 0.000003);
        assert_eq!(output, 0.000015);
    }

    #[test]
    fn haiku_pricing() {
        let (input, output) = cost_per_token("claude-haiku-4-5-20251001");
        assert_eq!(input, 0.0000008);
        assert_eq!(output, 0.000004);
    }

    #[test]
    fn openai_pricing() {
        let (input, output) = cost_per_token("gpt-4o");
        assert_eq!(input, 0.0000025);
        assert_eq!(output, 0.00001);
    }

    #[test]
    fn unknown_model_fallback() {
        let (input, output) = cost_per_token("some-unknown-model-v99");
        assert!(input > 0.0);
        assert!(output > 0.0);
    }

    #[test]
    fn compute_cost_basic() {
        let usage = TokenUsage::new(1000, 500);
        let model = test_model("claude-sonnet-4-5-20250929");
        let cost = compute_cost(&usage, &model);
        // 1000 * 0.000003 + 500 * 0.000015 = 0.003 + 0.0075 = 0.0105
        assert!((cost - 0.0105).abs() < 0.0001);
    }

    #[test]
    fn compute_cost_zero_tokens() {
        let usage = TokenUsage::new(0, 0);
        let model = test_model("claude-sonnet-4-5-20250929");
        let cost = compute_cost(&usage, &model);
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn compute_cost_with_cache() {
        let usage = TokenUsage {
            input_tokens: 1000,
            output_tokens: 500,
            cache_read_tokens: 2000,
            cache_write_tokens: 500,
        };
        let model = test_model("claude-sonnet-4-5-20250929");
        let cost = compute_cost(&usage, &model);
        // input: 1000 * 0.000003 = 0.003
        // output: 500 * 0.000015 = 0.0075
        // cache_read: 2000 * 0.000003 * 0.1 = 0.0006
        // cache_write: 500 * 0.000003 * 1.25 = 0.001875
        assert!(cost > 0.01);
    }

    #[test]
    fn model_info_pricing_overrides_table() {
        let usage = TokenUsage::new(1000, 500);
        let mut model = test_model("claude-sonnet-4-5-20250929");
        model.cost_per_input_token = 0.00001;
        model.cost_per_output_token = 0.00005;
        let cost = compute_cost(&usage, &model);
        // 1000 * 0.00001 + 500 * 0.00005 = 0.01 + 0.025 = 0.035
        assert!((cost - 0.035).abs() < 0.0001);
    }

    #[test]
    fn gemini_pricing() {
        let (input, output) = cost_per_token("gemini-2.0-flash");
        assert!(input < 0.000001); // Gemini flash is very cheap
        assert!(output < 0.000001);
    }
}
