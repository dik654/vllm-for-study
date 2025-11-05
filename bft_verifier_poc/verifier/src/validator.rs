use bft_common::{current_timestamp, RequestMetrics, ValidationError};

/// Validate metrics for sanity checks
///
/// Checks:
/// - Token counts are in reasonable range (1-1,000,000)
/// - Latency is reasonable (1-600,000ms = 10 minutes)
/// - Cost is non-negative
pub fn validate_metrics(metrics: &RequestMetrics) -> Result<(), ValidationError> {
    // Validate prompt tokens
    if metrics.prompt_tokens == 0 {
        return Err(ValidationError::InvalidTokenCount(metrics.prompt_tokens));
    }
    if metrics.prompt_tokens > 1_000_000 {
        return Err(ValidationError::InvalidTokenCount(metrics.prompt_tokens));
    }

    // Validate completion tokens (can be 0 for aborted requests)
    if metrics.completion_tokens > 1_000_000 {
        return Err(ValidationError::InvalidTokenCount(
            metrics.completion_tokens,
        ));
    }

    // Validate latency
    if metrics.e2e_latency_ms == 0 {
        return Err(ValidationError::InvalidLatency(metrics.e2e_latency_ms));
    }
    if metrics.e2e_latency_ms > 600_000 {
        // 10 minutes
        return Err(ValidationError::InvalidLatency(metrics.e2e_latency_ms));
    }

    // Validate cost
    if metrics.estimated_cost < 0.0 {
        return Err(ValidationError::InvalidCost(metrics.estimated_cost));
    }
    if metrics.estimated_cost > 1000.0 {
        // Sanity check: $1000 per request is too high
        return Err(ValidationError::InvalidCost(metrics.estimated_cost));
    }

    Ok(())
}

/// Validate timestamp is not too far in the future or past
///
/// Security hardening:
/// - Past tolerance: 60 seconds (was 1 hour)
/// - Future tolerance: 10 seconds (was 5 minutes)
///
/// This prevents replay attacks with old metrics
pub fn validate_timestamp(timestamp: u64) -> Result<(), ValidationError> {
    const TIMESTAMP_TOLERANCE_PAST: u64 = 60;    // 60 seconds in the past
    const TIMESTAMP_TOLERANCE_FUTURE: u64 = 10;  // 10 seconds in the future

    let now = current_timestamp();

    // Reject timestamps too far in the future
    if timestamp > now + TIMESTAMP_TOLERANCE_FUTURE {
        return Err(ValidationError::InvalidTimestamp(timestamp));
    }

    // Reject timestamps too far in the past
    if timestamp < now.saturating_sub(TIMESTAMP_TOLERANCE_PAST) {
        return Err(ValidationError::InvalidTimestamp(timestamp));
    }

    Ok(())
}

/// Validate request ID format (basic check)
pub fn validate_request_id(request_id: &str) -> Result<(), ValidationError> {
    if request_id.is_empty() {
        return Err(ValidationError::MissingField("request_id".to_string()));
    }

    if request_id.len() > 256 {
        return Err(ValidationError::OutOfRange(
            "request_id too long".to_string(),
        ));
    }

    Ok(())
}

/// Validate agent ID format
pub fn validate_agent_id(agent_id: &str) -> Result<(), ValidationError> {
    if agent_id.is_empty() {
        return Err(ValidationError::MissingField("agent_id".to_string()));
    }

    if agent_id.len() > 256 {
        return Err(ValidationError::OutOfRange("agent_id too long".to_string()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_metrics() {
        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        assert!(validate_metrics(&metrics).is_ok());
    }

    #[test]
    fn test_zero_prompt_tokens() {
        let metrics = RequestMetrics::new(0, 50, 1000, 0.001);
        let result = validate_metrics(&metrics);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidationError::InvalidTokenCount(0)
        ));
    }

    #[test]
    fn test_excessive_prompt_tokens() {
        let metrics = RequestMetrics::new(2_000_000, 50, 1000, 0.001);
        let result = validate_metrics(&metrics);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidationError::InvalidTokenCount(2_000_000)
        ));
    }

    #[test]
    fn test_zero_completion_tokens_allowed() {
        // Zero completion tokens is OK (aborted request)
        let metrics = RequestMetrics::new(100, 0, 1000, 0.001);
        assert!(validate_metrics(&metrics).is_ok());
    }

    #[test]
    fn test_excessive_completion_tokens() {
        let metrics = RequestMetrics::new(100, 2_000_000, 1000, 0.001);
        let result = validate_metrics(&metrics);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_latency() {
        let metrics = RequestMetrics::new(100, 50, 0, 0.001);
        let result = validate_metrics(&metrics);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidationError::InvalidLatency(0)
        ));
    }

    #[test]
    fn test_excessive_latency() {
        let metrics = RequestMetrics::new(100, 50, 700_000, 0.001);
        let result = validate_metrics(&metrics);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidationError::InvalidLatency(700_000)
        ));
    }

    #[test]
    fn test_negative_cost() {
        let metrics = RequestMetrics::new(100, 50, 1000, -0.001);
        let result = validate_metrics(&metrics);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidationError::InvalidCost(c) if c < 0.0
        ));
    }

    #[test]
    fn test_excessive_cost() {
        let metrics = RequestMetrics::new(100, 50, 1000, 1001.0);
        let result = validate_metrics(&metrics);
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_timestamp() {
        let now = current_timestamp();
        assert!(validate_timestamp(now).is_ok());
        assert!(validate_timestamp(now - 60).is_ok()); // 1 minute ago
        assert!(validate_timestamp(now + 60).is_ok()); // 1 minute future
    }

    #[test]
    fn test_timestamp_too_far_future() {
        let now = current_timestamp();
        let result = validate_timestamp(now + 400); // 6+ minutes future
        assert!(result.is_err());
    }

    #[test]
    fn test_timestamp_too_far_past() {
        let now = current_timestamp();
        let result = validate_timestamp(now.saturating_sub(4000)); // 1+ hour past
        assert!(result.is_err());
    }

    #[test]
    fn test_valid_request_id() {
        assert!(validate_request_id("req-123").is_ok());
        assert!(validate_request_id("uuid-123-456-789").is_ok());
    }

    #[test]
    fn test_empty_request_id() {
        let result = validate_request_id("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidationError::MissingField(_)
        ));
    }

    #[test]
    fn test_too_long_request_id() {
        let long_id = "a".repeat(300);
        let result = validate_request_id(&long_id);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ValidationError::OutOfRange(_)
        ));
    }

    #[test]
    fn test_valid_agent_id() {
        assert!(validate_agent_id("agent-1").is_ok());
        assert!(validate_agent_id("server-prod-01").is_ok());
    }

    #[test]
    fn test_empty_agent_id() {
        let result = validate_agent_id("");
        assert!(result.is_err());
    }

    #[test]
    fn test_too_long_agent_id() {
        let long_id = "a".repeat(300);
        let result = validate_agent_id(&long_id);
        assert!(result.is_err());
    }

    #[test]
    fn test_edge_case_max_valid_tokens() {
        let metrics = RequestMetrics::new(1_000_000, 1_000_000, 1000, 0.001);
        assert!(validate_metrics(&metrics).is_ok());
    }

    #[test]
    fn test_edge_case_max_valid_latency() {
        let metrics = RequestMetrics::new(100, 50, 600_000, 0.001);
        assert!(validate_metrics(&metrics).is_ok());
    }

    #[test]
    fn test_edge_case_max_valid_cost() {
        let metrics = RequestMetrics::new(100, 50, 1000, 1000.0);
        assert!(validate_metrics(&metrics).is_ok());
    }
}
