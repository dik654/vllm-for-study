use bft_common::{ConsensusError, VerificationResult, VerifierVote};
use std::collections::HashMap;
use std::time::Duration;
use tokio::time::timeout;
use tracing::{info, warn};

/// Consensus manager for collecting votes and reaching consensus
pub struct ConsensusManager {
    /// Quorum size (2f+1 for Byzantine fault tolerance)
    quorum: usize,

    /// Timeout for vote collection
    vote_timeout: Duration,
}

impl ConsensusManager {
    /// Create new consensus manager
    ///
    /// # Arguments
    /// * `quorum` - Minimum votes needed for consensus (typically 2f+1)
    /// * `vote_timeout` - Maximum time to wait for votes
    pub fn new(quorum: usize, vote_timeout: Duration) -> Self {
        Self {
            quorum,
            vote_timeout,
        }
    }

    /// Reach consensus from collected votes
    ///
    /// Returns (result, reason) where:
    /// - result: Some(Pass/Fail) if quorum reached, None otherwise
    /// - reason: Human-readable explanation
    pub fn reach_consensus(
        &self,
        votes: &HashMap<String, VerifierVote>,
    ) -> (Option<VerificationResult>, String) {
        if votes.is_empty() {
            return (None, "No votes received".to_string());
        }

        // Count votes by result
        let mut pass_count = 0;
        let mut fail_count = 0;

        for vote in votes.values() {
            match vote.result {
                VerificationResult::Pass => pass_count += 1,
                VerificationResult::Fail => fail_count += 1,
            }
        }

        info!(
            total_votes = votes.len(),
            pass_count,
            fail_count,
            quorum = self.quorum,
            "Tallying votes"
        );

        // Check if we have quorum for PASS
        if pass_count >= self.quorum {
            let reason = format!(
                "{}/{} verifiers agree: PASS (quorum: {})",
                pass_count,
                votes.len(),
                self.quorum
            );
            info!("{}", reason);
            return (Some(VerificationResult::Pass), reason);
        }

        // Check if we have quorum for FAIL
        if fail_count >= self.quorum {
            let reason = format!(
                "{}/{} verifiers agree: FAIL (quorum: {})",
                fail_count,
                votes.len(),
                self.quorum
            );
            warn!("{}", reason);
            return (Some(VerificationResult::Fail), reason);
        }

        // No quorum reached
        let reason = format!(
            "No quorum reached (PASS: {}, FAIL: {}, Quorum: {}, Total: {})",
            pass_count, fail_count, self.quorum, votes.len()
        );
        warn!("{}", reason);
        (None, reason)
    }

    /// Calculate quorum size for given fault tolerance
    ///
    /// For f Byzantine faults: N ≥ 3f+1, quorum = 2f+1
    pub fn calculate_quorum(byzantine_faults: usize) -> usize {
        2 * byzantine_faults + 1
    }

    /// Calculate minimum nodes for given fault tolerance
    pub fn calculate_min_nodes(byzantine_faults: usize) -> usize {
        3 * byzantine_faults + 1
    }

    /// Get quorum size
    pub fn quorum(&self) -> usize {
        self.quorum
    }

    /// Get vote timeout
    pub fn vote_timeout(&self) -> Duration {
        self.vote_timeout
    }

    /// Check if we have enough votes to possibly reach consensus
    pub fn can_reach_consensus(&self, votes: &HashMap<String, VerifierVote>, committee_size: usize) -> bool {
        let received = votes.len();
        let remaining = committee_size.saturating_sub(received);

        // Count current votes
        let mut pass_count = 0;
        let mut fail_count = 0;
        for vote in votes.values() {
            match vote.result {
                VerificationResult::Pass => pass_count += 1,
                VerificationResult::Fail => fail_count += 1,
            }
        }

        // Check if PASS can still reach quorum
        let max_pass = pass_count + remaining;
        if max_pass >= self.quorum {
            return true;
        }

        // Check if FAIL can still reach quorum
        let max_fail = fail_count + remaining;
        if max_fail >= self.quorum {
            return true;
        }

        false
    }

    /// Validate vote signature (placeholder for PoC)
    pub fn validate_vote(&self, _vote: &VerifierVote) -> Result<(), ConsensusError> {
        // In production, verify vote signature with verifier's public key
        // For PoC, we trust the votes
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_common::current_timestamp;

    fn create_vote(
        verifier_id: &str,
        result: VerificationResult,
    ) -> VerifierVote {
        VerifierVote::new(
            verifier_id.to_string(),
            "verification-1".to_string(),
            result,
            "test".to_string(),
            current_timestamp(),
            vec![],
        )
    }

    #[test]
    fn test_consensus_pass_with_quorum() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));

        let mut votes = HashMap::new();
        votes.insert("v1".to_string(), create_vote("v1", VerificationResult::Pass));
        votes.insert("v2".to_string(), create_vote("v2", VerificationResult::Pass));
        votes.insert("v3".to_string(), create_vote("v3", VerificationResult::Pass));
        votes.insert("v4".to_string(), create_vote("v4", VerificationResult::Fail));

        let (result, reason) = manager.reach_consensus(&votes);

        assert_eq!(result, Some(VerificationResult::Pass));
        assert!(reason.contains("PASS"));
        assert!(reason.contains("3/4"));
    }

    #[test]
    fn test_consensus_fail_with_quorum() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));

        let mut votes = HashMap::new();
        votes.insert("v1".to_string(), create_vote("v1", VerificationResult::Fail));
        votes.insert("v2".to_string(), create_vote("v2", VerificationResult::Fail));
        votes.insert("v3".to_string(), create_vote("v3", VerificationResult::Fail));
        votes.insert("v4".to_string(), create_vote("v4", VerificationResult::Pass));

        let (result, reason) = manager.reach_consensus(&votes);

        assert_eq!(result, Some(VerificationResult::Fail));
        assert!(reason.contains("FAIL"));
    }

    #[test]
    fn test_no_quorum() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));

        let mut votes = HashMap::new();
        votes.insert("v1".to_string(), create_vote("v1", VerificationResult::Pass));
        votes.insert("v2".to_string(), create_vote("v2", VerificationResult::Pass));
        votes.insert("v3".to_string(), create_vote("v3", VerificationResult::Fail));
        votes.insert("v4".to_string(), create_vote("v4", VerificationResult::Fail));

        let (result, reason) = manager.reach_consensus(&votes);

        assert_eq!(result, None);
        assert!(reason.contains("No quorum"));
        assert!(reason.contains("PASS: 2"));
        assert!(reason.contains("FAIL: 2"));
    }

    #[test]
    fn test_empty_votes() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));
        let votes = HashMap::new();

        let (result, reason) = manager.reach_consensus(&votes);

        assert_eq!(result, None);
        assert!(reason.contains("No votes"));
    }

    #[test]
    fn test_exact_quorum() {
        let manager = ConsensusManager::new(2, Duration::from_secs(5));

        let mut votes = HashMap::new();
        votes.insert("v1".to_string(), create_vote("v1", VerificationResult::Pass));
        votes.insert("v2".to_string(), create_vote("v2", VerificationResult::Pass));
        votes.insert("v3".to_string(), create_vote("v3", VerificationResult::Fail));

        let (result, _) = manager.reach_consensus(&votes);

        assert_eq!(result, Some(VerificationResult::Pass));
    }

    #[test]
    fn test_calculate_quorum() {
        assert_eq!(ConsensusManager::calculate_quorum(0), 1); // f=0: quorum=1
        assert_eq!(ConsensusManager::calculate_quorum(1), 3); // f=1: quorum=3
        assert_eq!(ConsensusManager::calculate_quorum(2), 5); // f=2: quorum=5
    }

    #[test]
    fn test_calculate_min_nodes() {
        assert_eq!(ConsensusManager::calculate_min_nodes(0), 1); // f=0: N=1
        assert_eq!(ConsensusManager::calculate_min_nodes(1), 4); // f=1: N=4
        assert_eq!(ConsensusManager::calculate_min_nodes(2), 7); // f=2: N=7
    }

    #[test]
    fn test_can_reach_consensus_yes() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));

        let mut votes = HashMap::new();
        votes.insert("v1".to_string(), create_vote("v1", VerificationResult::Pass));
        votes.insert("v2".to_string(), create_vote("v2", VerificationResult::Pass));

        // 2 PASS votes, 2 remaining (committee size 4)
        // Max PASS = 2 + 2 = 4 >= 3 (quorum)
        assert!(manager.can_reach_consensus(&votes, 4));
    }

    #[test]
    fn test_can_reach_consensus_no() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));

        let mut votes = HashMap::new();
        votes.insert("v1".to_string(), create_vote("v1", VerificationResult::Pass));
        votes.insert("v2".to_string(), create_vote("v2", VerificationResult::Fail));
        votes.insert("v3".to_string(), create_vote("v3", VerificationResult::Fail));

        // 1 PASS, 2 FAIL, 0 remaining (committee size 3)
        // Max PASS = 1 + 0 = 1 < 3 (quorum)
        // Max FAIL = 2 + 0 = 2 < 3 (quorum)
        assert!(!manager.can_reach_consensus(&votes, 3));
    }

    #[test]
    fn test_quorum_getter() {
        let manager = ConsensusManager::new(5, Duration::from_secs(10));
        assert_eq!(manager.quorum(), 5);
    }

    #[test]
    fn test_vote_timeout_getter() {
        let manager = ConsensusManager::new(3, Duration::from_secs(7));
        assert_eq!(manager.vote_timeout(), Duration::from_secs(7));
    }

    #[test]
    fn test_validate_vote() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));
        let vote = create_vote("v1", VerificationResult::Pass);

        // For PoC, always returns Ok
        assert!(manager.validate_vote(&vote).is_ok());
    }

    #[test]
    fn test_unanimous_pass() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));

        let mut votes = HashMap::new();
        votes.insert("v1".to_string(), create_vote("v1", VerificationResult::Pass));
        votes.insert("v2".to_string(), create_vote("v2", VerificationResult::Pass));
        votes.insert("v3".to_string(), create_vote("v3", VerificationResult::Pass));

        let (result, reason) = manager.reach_consensus(&votes);

        assert_eq!(result, Some(VerificationResult::Pass));
        assert!(reason.contains("3/3"));
    }

    #[test]
    fn test_single_dissenter() {
        let manager = ConsensusManager::new(3, Duration::from_secs(5));

        let mut votes = HashMap::new();
        for i in 1..=4 {
            let result = if i == 4 {
                VerificationResult::Fail
            } else {
                VerificationResult::Pass
            };
            votes.insert(format!("v{}", i), create_vote(&format!("v{}", i), result));
        }

        let (result, _) = manager.reach_consensus(&votes);

        // 3 PASS, 1 FAIL - should reach consensus on PASS
        assert_eq!(result, Some(VerificationResult::Pass));
    }
}
