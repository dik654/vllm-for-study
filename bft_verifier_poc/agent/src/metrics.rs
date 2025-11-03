use bft_common::RequestMetrics;
use rand::Rng;

/// Metrics generator for testing and simulation
///
/// Generates realistic LLM request metrics with proper distributions
pub struct MetricsGenerator {
    rng: rand::rngs::ThreadRng,
}

impl MetricsGenerator {
    /// Create new metrics generator
    pub fn new() -> Self {
        Self {
            rng: rand::thread_rng(),
        }
    }

    /// Generate random metrics with realistic values
    pub fn generate_random_metrics(&mut self) -> RequestMetrics {
        // Prompt tokens: 10-2000 (typical range for most requests)
        let prompt_tokens = self.rng.gen_range(10..2000);

        // Completion tokens: 10-1000 (usually shorter than prompt)
        let completion_tokens = self.rng.gen_range(10..1000);

        // Latency: 100ms-10s (typical for LLM inference)
        let e2e_latency_ms = self.rng.gen_range(100..10000);

        // Cost: $0.0001-$0.1 per request (realistic pricing)
        let estimated_cost = self.rng.gen_range(0.0001..0.1);

        RequestMetrics::new(
            prompt_tokens,
            completion_tokens,
            e2e_latency_ms,
            estimated_cost,
        )
    }

    /// Generate metrics with specific prompt size
    pub fn generate_with_prompt_size(&mut self, prompt_tokens: u32) -> RequestMetrics {
        let completion_tokens = self.rng.gen_range(10..1000);
        let e2e_latency_ms = self.rng.gen_range(100..10000);
        let estimated_cost = (prompt_tokens as f64 * 0.00001) + (completion_tokens as f64 * 0.00002);

        RequestMetrics::new(
            prompt_tokens,
            completion_tokens,
            e2e_latency_ms,
            estimated_cost,
        )
    }

    /// Generate short request metrics (< 100 tokens)
    pub fn generate_short(&mut self) -> RequestMetrics {
        let prompt_tokens = self.rng.gen_range(10..100);
        let completion_tokens = self.rng.gen_range(10..100);
        let e2e_latency_ms = self.rng.gen_range(100..1000);
        let estimated_cost = self.rng.gen_range(0.0001..0.001);

        RequestMetrics::new(
            prompt_tokens,
            completion_tokens,
            e2e_latency_ms,
            estimated_cost,
        )
    }

    /// Generate long request metrics (> 1000 tokens)
    pub fn generate_long(&mut self) -> RequestMetrics {
        let prompt_tokens = self.rng.gen_range(1000..5000);
        let completion_tokens = self.rng.gen_range(500..2000);
        let e2e_latency_ms = self.rng.gen_range(2000..30000);
        let estimated_cost = self.rng.gen_range(0.01..0.1);

        RequestMetrics::new(
            prompt_tokens,
            completion_tokens,
            e2e_latency_ms,
            estimated_cost,
        )
    }

    /// Generate metrics with realistic latency based on token count
    pub fn generate_realistic(&mut self) -> RequestMetrics {
        let prompt_tokens = self.rng.gen_range(10..2000);
        let completion_tokens = self.rng.gen_range(10..1000);

        // Realistic latency: ~10ms per output token + base overhead
        let base_latency = 200; // Base overhead
        let per_token_latency = 15; // ms per token
        let e2e_latency_ms = base_latency + (completion_tokens as u64 * per_token_latency);

        // Add some jitter (±20%)
        let jitter = self.rng.gen_range(0.8..1.2);
        let e2e_latency_ms = (e2e_latency_ms as f64 * jitter) as u64;

        // Pricing: $0.50 per 1M input tokens, $1.50 per 1M output tokens
        let input_cost = (prompt_tokens as f64 / 1_000_000.0) * 0.50;
        let output_cost = (completion_tokens as f64 / 1_000_000.0) * 1.50;
        let estimated_cost = input_cost + output_cost;

        RequestMetrics::new(
            prompt_tokens,
            completion_tokens,
            e2e_latency_ms,
            estimated_cost,
        )
    }
}

impl Default for MetricsGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_metrics() {
        let mut gen = MetricsGenerator::new();
        let metrics = gen.generate_random_metrics();

        // Check ranges
        assert!(metrics.prompt_tokens >= 10);
        assert!(metrics.prompt_tokens <= 2000);
        assert!(metrics.completion_tokens >= 10);
        assert!(metrics.completion_tokens <= 1000);
        assert!(metrics.e2e_latency_ms >= 100);
        assert!(metrics.e2e_latency_ms <= 10000);
        assert!(metrics.estimated_cost >= 0.0001);
        assert!(metrics.estimated_cost <= 0.1);
    }

    #[test]
    fn test_generate_with_prompt_size() {
        let mut gen = MetricsGenerator::new();
        let metrics = gen.generate_with_prompt_size(500);

        assert_eq!(metrics.prompt_tokens, 500);
        assert!(metrics.completion_tokens > 0);
    }

    #[test]
    fn test_generate_short() {
        let mut gen = MetricsGenerator::new();
        let metrics = gen.generate_short();

        assert!(metrics.prompt_tokens < 100);
        assert!(metrics.completion_tokens < 100);
        assert!(metrics.e2e_latency_ms < 1000);
    }

    #[test]
    fn test_generate_long() {
        let mut gen = MetricsGenerator::new();
        let metrics = gen.generate_long();

        assert!(metrics.prompt_tokens >= 1000);
        assert!(metrics.completion_tokens >= 500);
        assert!(metrics.e2e_latency_ms >= 2000);
    }

    #[test]
    fn test_generate_realistic() {
        let mut gen = MetricsGenerator::new();
        let metrics = gen.generate_realistic();

        // Check that latency is reasonable for token count
        let expected_latency_ms = 200 + (metrics.completion_tokens as u64 * 15);
        let min_latency = (expected_latency_ms as f64 * 0.8) as u64;
        let max_latency = (expected_latency_ms as f64 * 1.2) as u64;

        assert!(metrics.e2e_latency_ms >= min_latency);
        assert!(metrics.e2e_latency_ms <= max_latency);

        // Check cost is positive
        assert!(metrics.estimated_cost > 0.0);
    }

    #[test]
    fn test_multiple_generations_different() {
        let mut gen = MetricsGenerator::new();

        let metrics1 = gen.generate_random_metrics();
        let metrics2 = gen.generate_random_metrics();

        // Should generate different values (with very high probability)
        assert!(
            metrics1.prompt_tokens != metrics2.prompt_tokens
                || metrics1.completion_tokens != metrics2.completion_tokens
        );
    }

    #[test]
    fn test_default() {
        let _gen = MetricsGenerator::default();
        // Just test that default works
    }
}
