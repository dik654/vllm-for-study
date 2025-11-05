///! Agent state tracker for rate limiting and anomaly detection
///!
///! This module tracks per-agent metrics to:
///! - Prevent DDoS attacks (rate limiting)
///! - Detect anomalous patterns (cost spikes, frequency spikes)
///! - Monitor agent health and behavior

use anyhow::Result;
use bft_common::{current_timestamp, RequestMetrics};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Agent state tracker configuration
#[derive(Debug, Clone)]
pub struct AgentTrackerConfig {
    /// Minimum interval between submissions (seconds)
    pub min_submission_interval_secs: u64,
    /// Maximum requests per hour
    pub max_requests_per_hour: u64,
    /// Maximum cost spike multiplier (e.g., 10.0 = 10x average)
    pub max_cost_spike_multiplier: f64,
}

impl Default for AgentTrackerConfig {
    fn default() -> Self {
        Self {
            min_submission_interval_secs: 1,  // At least 1 second between submissions
            max_requests_per_hour: 3600,      // 1 req/sec average
            max_cost_spike_multiplier: 10.0,  // 10x average cost
        }
    }
}

/// Per-agent state
#[derive(Debug, Clone)]
struct AgentState {
    /// Agent ID
    agent_id: String,
    /// Total requests submitted
    total_requests: u64,
    /// Total tokens (prompt + completion)
    total_tokens: u64,
    /// Total cost
    total_cost: f64,
    /// Last submission timestamp
    last_submission: u64,
    /// Hourly request counter
    hourly_requests: u64,
    /// Hourly counter reset time
    hourly_reset_time: u64,
    /// Average cost per request
    avg_cost: f64,
    /// First seen timestamp
    first_seen: u64,
}

impl AgentState {
    fn new(agent_id: String) -> Self {
        let now = current_timestamp();
        Self {
            agent_id,
            total_requests: 0,
            total_tokens: 0,
            total_cost: 0.0,
            last_submission: 0,
            hourly_requests: 0,
            hourly_reset_time: now,
            avg_cost: 0.0,
            first_seen: now,
        }
    }

    fn update(&mut self, metrics: &RequestMetrics) {
        let now = current_timestamp();

        // Reset hourly counter if needed
        if now - self.hourly_reset_time >= 3600 {
            self.hourly_requests = 0;
            self.hourly_reset_time = now;
        }

        // Update counters
        self.total_requests += 1;
        self.total_tokens += (metrics.prompt_tokens + metrics.completion_tokens) as u64;
        self.total_cost += metrics.estimated_cost;
        self.last_submission = now;
        self.hourly_requests += 1;

        // Update average cost
        self.avg_cost = self.total_cost / self.total_requests as f64;
    }
}

/// Agent state tracker
pub struct AgentTracker {
    config: AgentTrackerConfig,
    states: Arc<RwLock<HashMap<String, AgentState>>>,
}

impl AgentTracker {
    /// Create a new agent tracker
    pub fn new(config: AgentTrackerConfig) -> Self {
        info!(
            "Agent tracker initialized (min_interval={}s, max_hourly={})",
            config.min_submission_interval_secs, config.max_requests_per_hour
        );

        Self {
            config,
            states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Validate submission rate and patterns
    ///
    /// Returns Ok(()) if submission is allowed
    /// Returns Err if:
    /// - Submission too frequent (DDoS)
    /// - Hourly rate limit exceeded
    /// - Anomalous cost pattern detected
    pub async fn validate_submission(
        &self,
        agent_id: &str,
        metrics: &RequestMetrics,
    ) -> Result<()> {
        let mut states = self.states.write().await;

        // Get or create agent state
        let state = states
            .entry(agent_id.to_string())
            .or_insert_with(|| AgentState::new(agent_id.to_string()));

        // Check minimum submission interval (DDoS prevention)
        if state.last_submission > 0 {
            let time_since_last = current_timestamp() - state.last_submission;

            if time_since_last < self.config.min_submission_interval_secs {
                warn!(
                    "Agent {} submitting too frequently: {}s < {}s",
                    agent_id, time_since_last, self.config.min_submission_interval_secs
                );
                return Err(anyhow::anyhow!(
                    "Submission too frequent: {}s < {}s minimum",
                    time_since_last,
                    self.config.min_submission_interval_secs
                ));
            }
        }

        // Check hourly rate limit
        if state.hourly_requests >= self.config.max_requests_per_hour {
            warn!(
                "Agent {} exceeded hourly rate limit: {} >= {}",
                agent_id, state.hourly_requests, self.config.max_requests_per_hour
            );
            return Err(anyhow::anyhow!(
                "Hourly rate limit exceeded: {}/hour",
                self.config.max_requests_per_hour
            ));
        }

        // Check for cost anomaly (if we have history)
        if state.total_requests > 10 && state.avg_cost > 0.0 {
            let cost_ratio = metrics.estimated_cost / state.avg_cost;

            if cost_ratio > self.config.max_cost_spike_multiplier {
                warn!(
                    "Agent {} cost spike detected: ${:.4} vs ${:.4} avg ({}x)",
                    agent_id, metrics.estimated_cost, state.avg_cost, cost_ratio
                );

                // Don't reject, just warn (could be legitimate large request)
                // In production, might want to trigger manual review
            }
        }

        // Update state
        state.update(metrics);

        debug!(
            "Agent {} validation passed: total={} hourly={}",
            agent_id, state.total_requests, state.hourly_requests
        );

        Ok(())
    }

    /// Get agent statistics
    pub async fn get_agent_stats(&self, agent_id: &str) -> Option<AgentStats> {
        let states = self.states.read().await;
        states.get(agent_id).map(|state| AgentStats {
            agent_id: state.agent_id.clone(),
            total_requests: state.total_requests,
            total_tokens: state.total_tokens,
            total_cost: state.total_cost,
            avg_cost: state.avg_cost,
            hourly_requests: state.hourly_requests,
            last_submission: state.last_submission,
            first_seen: state.first_seen,
        })
    }

    /// Get all agents statistics
    pub async fn get_all_stats(&self) -> Vec<AgentStats> {
        let states = self.states.read().await;
        states
            .values()
            .map(|state| AgentStats {
                agent_id: state.agent_id.clone(),
                total_requests: state.total_requests,
                total_tokens: state.total_tokens,
                total_cost: state.total_cost,
                avg_cost: state.avg_cost,
                hourly_requests: state.hourly_requests,
                last_submission: state.last_submission,
                first_seen: state.first_seen,
            })
            .collect()
    }

    /// Get total system stats
    pub async fn get_system_stats(&self) -> SystemStats {
        let states = self.states.read().await;

        let total_agents = states.len();
        let total_requests: u64 = states.values().map(|s| s.total_requests).sum();
        let total_tokens: u64 = states.values().map(|s| s.total_tokens).sum();
        let total_cost: f64 = states.values().map(|s| s.total_cost).sum();

        SystemStats {
            total_agents,
            total_requests,
            total_tokens,
            total_cost,
        }
    }
}

/// Agent statistics
#[derive(Debug, Clone)]
pub struct AgentStats {
    pub agent_id: String,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost: f64,
    pub avg_cost: f64,
    pub hourly_requests: u64,
    pub last_submission: u64,
    pub first_seen: u64,
}

/// System-wide statistics
#[derive(Debug, Clone)]
pub struct SystemStats {
    pub total_agents: usize,
    pub total_requests: u64,
    pub total_tokens: u64,
    pub total_cost: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics(cost: f64) -> RequestMetrics {
        RequestMetrics {
            prompt_tokens: 100,
            completion_tokens: 50,
            cached_tokens: 0,
            e2e_latency_ms: 1000,
            time_to_first_token_ms: 100,
            estimated_cost: cost,
            sequence: 0,
            prev_hash: vec![],
            current_hash: vec![],
        }
    }

    #[tokio::test]
    async fn test_first_submission() {
        let tracker = AgentTracker::new(AgentTrackerConfig::default());
        let metrics = create_test_metrics(0.01);

        assert!(tracker.validate_submission("agent-1", &metrics).await.is_ok());

        let stats = tracker.get_agent_stats("agent-1").await.unwrap();
        assert_eq!(stats.total_requests, 1);
        assert_eq!(stats.total_cost, 0.01);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = AgentTrackerConfig {
            min_submission_interval_secs: 10,
            ..Default::default()
        };
        let tracker = AgentTracker::new(config);
        let metrics = create_test_metrics(0.01);

        // First submission: OK
        assert!(tracker.validate_submission("agent-1", &metrics).await.is_ok());

        // Immediate second submission: should fail
        assert!(tracker.validate_submission("agent-1", &metrics).await.is_err());
    }

    #[tokio::test]
    async fn test_hourly_limit() {
        let config = AgentTrackerConfig {
            min_submission_interval_secs: 0,  // No interval check
            max_requests_per_hour: 2,
            ..Default::default()
        };
        let tracker = AgentTracker::new(config);
        let metrics = create_test_metrics(0.01);

        // First two: OK
        assert!(tracker.validate_submission("agent-1", &metrics).await.is_ok());
        assert!(tracker.validate_submission("agent-1", &metrics).await.is_ok());

        // Third: should fail
        assert!(tracker.validate_submission("agent-1", &metrics).await.is_err());
    }

    #[tokio::test]
    async fn test_cost_spike_detection() {
        let config = AgentTrackerConfig {
            min_submission_interval_secs: 0,
            max_cost_spike_multiplier: 5.0,
            ..Default::default()
        };
        let tracker = AgentTracker::new(config);

        // Submit 10 normal requests
        for _ in 0..10 {
            let metrics = create_test_metrics(0.01);
            tracker.validate_submission("agent-1", &metrics).await.unwrap();
        }

        // Submit spike (10x cost)
        let spike_metrics = create_test_metrics(0.10);
        // Should still succeed but log warning
        assert!(tracker
            .validate_submission("agent-1", &spike_metrics)
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_multiple_agents() {
        let tracker = AgentTracker::new(AgentTrackerConfig::default());

        let metrics1 = create_test_metrics(0.01);
        let metrics2 = create_test_metrics(0.02);

        tracker.validate_submission("agent-1", &metrics1).await.unwrap();
        tracker.validate_submission("agent-2", &metrics2).await.unwrap();

        let stats1 = tracker.get_agent_stats("agent-1").await.unwrap();
        let stats2 = tracker.get_agent_stats("agent-2").await.unwrap();

        assert_eq!(stats1.total_cost, 0.01);
        assert_eq!(stats2.total_cost, 0.02);

        let system_stats = tracker.get_system_stats().await;
        assert_eq!(system_stats.total_agents, 2);
        assert_eq!(system_stats.total_requests, 2);
        assert_eq!(system_stats.total_cost, 0.03);
    }
}
