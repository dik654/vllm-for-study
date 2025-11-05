///! Metrics chain validator for detecting tampering and omissions
///!
///! This module validates the blockchain-style chain of metrics submissions.
///! Each metrics includes:
///! - sequence: Sequential number (1, 2, 3, ...)
///! - prev_hash: Hash of previous metrics
///! - current_hash: Hash of current metrics
///!
///! This prevents:
///! - Skipping metrics (sequence gap detected)
///! - Reordering metrics (prev_hash mismatch)
///! - Tampering with past metrics (chain break)

use anyhow::{Context, Result};
use bft_common::RequestMetrics;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Agent chain state
#[derive(Debug, Clone)]
struct AgentChainState {
    /// Last sequence number received
    last_sequence: u64,
    /// Last metrics hash
    last_hash: Vec<u8>,
    /// Total metrics received
    total_received: u64,
}

impl AgentChainState {
    /// Create genesis state (for first metrics)
    fn genesis() -> Self {
        Self {
            last_sequence: 0,
            last_hash: vec![0u8; 32],  // Empty hash
            total_received: 0,
        }
    }
}

/// Chain validator
pub struct ChainValidator {
    /// Per-agent chain states
    agent_chains: Arc<RwLock<HashMap<String, AgentChainState>>>,
}

impl ChainValidator {
    /// Create a new chain validator
    pub fn new() -> Self {
        Self {
            agent_chains: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Validate metrics chain
    ///
    /// Returns Ok(()) if chain is valid
    /// Returns Err if:
    /// - Sequence number skipped
    /// - Previous hash doesn't match
    /// - Current hash is incorrect
    pub async fn validate_chain(
        &self,
        agent_id: &str,
        metrics: &RequestMetrics,
    ) -> Result<()> {
        let mut chains = self.agent_chains.write().await;

        // Get or create agent state
        let state = chains
            .entry(agent_id.to_string())
            .or_insert_with(AgentChainState::genesis);

        // Validate sequence number
        let expected_sequence = state.last_sequence + 1;

        if metrics.sequence == 0 {
            // Sequence not set - skip chain validation
            // (backward compatibility)
            debug!("Chain validation skipped: sequence not set");
            return Ok(());
        }

        if metrics.sequence != expected_sequence {
            warn!(
                "Sequence mismatch: expected {}, got {}",
                expected_sequence, metrics.sequence
            );
            return Err(anyhow::anyhow!(
                "Sequence number skip detected: expected {}, got {}",
                expected_sequence,
                metrics.sequence
            ));
        }

        // Validate previous hash
        if metrics.prev_hash != state.last_hash {
            warn!(
                "Previous hash mismatch: expected {:?}, got {:?}",
                hex::encode(&state.last_hash),
                hex::encode(&metrics.prev_hash)
            );
            return Err(anyhow::anyhow!("Chain broken: prev_hash mismatch"));
        }

        // Validate current hash
        let calculated_hash = calculate_metrics_hash(metrics);
        if metrics.current_hash != calculated_hash {
            warn!(
                "Current hash mismatch: expected {:?}, got {:?}",
                hex::encode(&calculated_hash),
                hex::encode(&metrics.current_hash)
            );
            return Err(anyhow::anyhow!("Current hash mismatch"));
        }

        // Update state
        state.last_sequence = metrics.sequence;
        state.last_hash = metrics.current_hash.clone();
        state.total_received += 1;

        debug!(
            "Chain validated: agent={} seq={} total={}",
            agent_id, metrics.sequence, state.total_received
        );

        Ok(())
    }

    /// Get agent stats
    pub async fn get_agent_stats(&self, agent_id: &str) -> Option<AgentChainStats> {
        let chains = self.agent_chains.read().await;
        chains.get(agent_id).map(|state| AgentChainStats {
            last_sequence: state.last_sequence,
            total_received: state.total_received,
        })
    }

    /// Get all agents stats
    pub async fn get_all_stats(&self) -> HashMap<String, AgentChainStats> {
        let chains = self.agent_chains.read().await;
        chains
            .iter()
            .map(|(agent_id, state)| {
                (
                    agent_id.clone(),
                    AgentChainStats {
                        last_sequence: state.last_sequence,
                        total_received: state.total_received,
                    },
                )
            })
            .collect()
    }
}

/// Agent chain statistics
#[derive(Debug, Clone)]
pub struct AgentChainStats {
    pub last_sequence: u64,
    pub total_received: u64,
}

/// Calculate hash of metrics for chain validation
pub fn calculate_metrics_hash(metrics: &RequestMetrics) -> Vec<u8> {
    let mut hasher = Sha256::new();

    // Hash all fields in deterministic order
    hasher.update(metrics.sequence.to_le_bytes());
    hasher.update(&metrics.prev_hash);
    hasher.update(metrics.prompt_tokens.to_le_bytes());
    hasher.update(metrics.completion_tokens.to_le_bytes());
    hasher.update(metrics.cached_tokens.to_le_bytes());
    hasher.update(metrics.e2e_latency_ms.to_le_bytes());
    hasher.update(metrics.time_to_first_token_ms.to_le_bytes());
    hasher.update(metrics.estimated_cost.to_le_bytes());

    hasher.finalize().to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_metrics(sequence: u64, prev_hash: Vec<u8>) -> RequestMetrics {
        let mut metrics = RequestMetrics {
            prompt_tokens: 100,
            completion_tokens: 50,
            cached_tokens: 10,
            e2e_latency_ms: 1000,
            time_to_first_token_ms: 100,
            estimated_cost: 0.01,
            sequence,
            prev_hash,
            current_hash: vec![],
        };

        // Calculate current hash
        metrics.current_hash = calculate_metrics_hash(&metrics);

        metrics
    }

    #[tokio::test]
    async fn test_first_metrics() {
        let validator = ChainValidator::new();

        let metrics = create_metrics(1, vec![0u8; 32]);

        assert!(validator.validate_chain("agent-1", &metrics).await.is_ok());
    }

    #[tokio::test]
    async fn test_sequence_validation() {
        let validator = ChainValidator::new();

        // Metrics 1
        let metrics1 = create_metrics(1, vec![0u8; 32]);
        assert!(validator.validate_chain("agent-1", &metrics1).await.is_ok());

        // Metrics 2 (correct sequence)
        let metrics2 = create_metrics(2, metrics1.current_hash.clone());
        assert!(validator.validate_chain("agent-1", &metrics2).await.is_ok());

        // Metrics 4 (skip 3 - should fail)
        let metrics4 = create_metrics(4, metrics2.current_hash.clone());
        assert!(validator.validate_chain("agent-1", &metrics4).await.is_err());
    }

    #[tokio::test]
    async fn test_hash_chain_validation() {
        let validator = ChainValidator::new();

        // Metrics 1
        let metrics1 = create_metrics(1, vec![0u8; 32]);
        assert!(validator.validate_chain("agent-1", &metrics1).await.is_ok());

        // Metrics 2 with wrong prev_hash (should fail)
        let metrics2 = create_metrics(2, vec![0xAB; 32]);
        assert!(validator.validate_chain("agent-1", &metrics2).await.is_err());
    }

    #[tokio::test]
    async fn test_current_hash_validation() {
        let validator = ChainValidator::new();

        let mut metrics = create_metrics(1, vec![0u8; 32]);

        // Tamper with current_hash
        metrics.current_hash = vec![0xFF; 32];

        assert!(validator.validate_chain("agent-1", &metrics).await.is_err());
    }

    #[tokio::test]
    async fn test_multiple_agents() {
        let validator = ChainValidator::new();

        // Agent 1
        let metrics1_a = create_metrics(1, vec![0u8; 32]);
        assert!(validator
            .validate_chain("agent-1", &metrics1_a)
            .await
            .is_ok());

        // Agent 2 (different chain)
        let metrics1_b = create_metrics(1, vec![0u8; 32]);
        assert!(validator
            .validate_chain("agent-2", &metrics1_b)
            .await
            .is_ok());

        // Agent 1 sequence 2
        let metrics2_a = create_metrics(2, metrics1_a.current_hash.clone());
        assert!(validator
            .validate_chain("agent-1", &metrics2_a)
            .await
            .is_ok());

        // Stats
        let stats1 = validator.get_agent_stats("agent-1").await.unwrap();
        assert_eq!(stats1.last_sequence, 2);
        assert_eq!(stats1.total_received, 2);

        let stats2 = validator.get_agent_stats("agent-2").await.unwrap();
        assert_eq!(stats2.last_sequence, 1);
        assert_eq!(stats2.total_received, 1);
    }

    #[tokio::test]
    async fn test_backward_compatibility() {
        let validator = ChainValidator::new();

        // Metrics without sequence (backward compatible)
        let metrics = RequestMetrics {
            prompt_tokens: 100,
            completion_tokens: 50,
            cached_tokens: 10,
            e2e_latency_ms: 1000,
            time_to_first_token_ms: 100,
            estimated_cost: 0.01,
            sequence: 0,  // Not set
            prev_hash: vec![],
            current_hash: vec![],
        };

        // Should succeed (skip validation)
        assert!(validator.validate_chain("agent-1", &metrics).await.is_ok());
    }
}
