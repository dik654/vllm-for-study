use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core metrics collected from LLM request
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RequestMetrics {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub e2e_latency_ms: u64,
    pub estimated_cost: f64,
}

impl RequestMetrics {
    /// Create new RequestMetrics with validation
    pub fn new(
        prompt_tokens: u32,
        completion_tokens: u32,
        e2e_latency_ms: u64,
        estimated_cost: f64,
    ) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            e2e_latency_ms,
            estimated_cost,
        }
    }

    /// Total tokens
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens + self.completion_tokens
    }
}

/// TPM Quote structure (simulated for PoC)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpmQuote {
    /// PCR register values (index -> hash)
    pub pcr_values: HashMap<u32, Vec<u8>>,
    /// Quote signature (Ed25519 for PoC)
    pub quote_signature: Vec<u8>,
    /// Nonce for replay protection
    pub nonce: Vec<u8>,
}

impl TpmQuote {
    pub fn new(pcr_values: HashMap<u32, Vec<u8>>, quote_signature: Vec<u8>, nonce: Vec<u8>) -> Self {
        Self {
            pcr_values,
            quote_signature,
            nonce,
        }
    }

    /// Get PCR value by index
    pub fn get_pcr(&self, index: u32) -> Option<&Vec<u8>> {
        self.pcr_values.get(&index)
    }
}

/// Complete metric submission from agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSubmission {
    pub agent_id: String,
    pub request_id: String,
    pub timestamp: u64,
    pub metrics: RequestMetrics,
    pub quote: TpmQuote,
}

impl MetricSubmission {
    pub fn new(
        agent_id: String,
        request_id: String,
        timestamp: u64,
        metrics: RequestMetrics,
        quote: TpmQuote,
    ) -> Self {
        Self {
            agent_id,
            request_id,
            timestamp,
            metrics,
            quote,
        }
    }
}

/// Verification request sent to verifiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRequest {
    pub verification_id: String,
    pub epoch: u64,
    pub submission: MetricSubmission,
}

impl VerificationRequest {
    pub fn new(verification_id: String, epoch: u64, submission: MetricSubmission) -> Self {
        Self {
            verification_id,
            epoch,
            submission,
        }
    }
}

/// Verification result enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationResult {
    Pass,
    Fail,
}

impl std::fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationResult::Pass => write!(f, "PASS"),
            VerificationResult::Fail => write!(f, "FAIL"),
        }
    }
}

/// Vote from a verifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierVote {
    pub verifier_id: String,
    pub verification_id: String,
    pub result: VerificationResult,
    pub reason: String,
    pub timestamp: u64,
    pub signature: Vec<u8>,
}

impl VerifierVote {
    pub fn new(
        verifier_id: String,
        verification_id: String,
        result: VerificationResult,
        reason: String,
        timestamp: u64,
        signature: Vec<u8>,
    ) -> Self {
        Self {
            verifier_id,
            verification_id,
            result,
            reason,
            timestamp,
            signature,
        }
    }
}

/// Consensus outcome after vote collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusOutcome {
    pub verification_id: String,
    pub epoch: u64,
    pub committee: Vec<String>,
    pub votes: HashMap<String, VerifierVote>,
    pub result: Option<VerificationResult>,
    pub quorum_reached: bool,
    pub timestamp: u64,
}

impl ConsensusOutcome {
    pub fn new(
        verification_id: String,
        epoch: u64,
        committee: Vec<String>,
        votes: HashMap<String, VerifierVote>,
        result: Option<VerificationResult>,
        quorum_reached: bool,
        timestamp: u64,
    ) -> Self {
        Self {
            verification_id,
            epoch,
            committee,
            votes,
            result,
            quorum_reached,
            timestamp,
        }
    }

    /// Get vote count by result
    pub fn vote_count(&self, result: VerificationResult) -> usize {
        self.votes
            .values()
            .filter(|v| v.result == result)
            .count()
    }

    /// Get pass vote count
    pub fn pass_count(&self) -> usize {
        self.vote_count(VerificationResult::Pass)
    }

    /// Get fail vote count
    pub fn fail_count(&self) -> usize {
        self.vote_count(VerificationResult::Fail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_request_metrics_total_tokens() {
        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        assert_eq!(metrics.total_tokens(), 150);
    }

    #[test]
    fn test_verification_result_display() {
        assert_eq!(VerificationResult::Pass.to_string(), "PASS");
        assert_eq!(VerificationResult::Fail.to_string(), "FAIL");
    }

    #[test]
    fn test_consensus_outcome_vote_counts() {
        let mut votes = HashMap::new();
        votes.insert(
            "v1".to_string(),
            VerifierVote::new(
                "v1".to_string(),
                "vid1".to_string(),
                VerificationResult::Pass,
                "valid".to_string(),
                1000,
                vec![],
            ),
        );
        votes.insert(
            "v2".to_string(),
            VerifierVote::new(
                "v2".to_string(),
                "vid1".to_string(),
                VerificationResult::Pass,
                "valid".to_string(),
                1001,
                vec![],
            ),
        );
        votes.insert(
            "v3".to_string(),
            VerifierVote::new(
                "v3".to_string(),
                "vid1".to_string(),
                VerificationResult::Fail,
                "invalid".to_string(),
                1002,
                vec![],
            ),
        );

        let outcome = ConsensusOutcome::new(
            "vid1".to_string(),
            1,
            vec!["v1".to_string(), "v2".to_string(), "v3".to_string()],
            votes,
            Some(VerificationResult::Pass),
            true,
            1003,
        );

        assert_eq!(outcome.pass_count(), 2);
        assert_eq!(outcome.fail_count(), 1);
    }

    #[test]
    fn test_serialization() {
        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let json = serde_json::to_string(&metrics).unwrap();
        let deserialized: RequestMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(metrics, deserialized);
    }
}
