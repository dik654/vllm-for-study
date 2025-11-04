use crate::consensus::ConsensusManager;
use crate::epoch::EpochManager;
use crate::vrf::VrfSelector;
use bft_common::{current_timestamp, ConsensusOutcome, VerificationResult, VerifierVote};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};

pub mod bft_verifier {
    tonic::include_proto!("bft_verifier");
}

use bft_verifier::coordinator_server::Coordinator;
use bft_verifier::{ConsensusOutcome as ProtoConsensusOutcome, MetricSubmission, SubmissionAck, VerificationQuery};

/// Coordinator service implementation
pub struct CoordinatorService {
    /// VRF selector for committee selection
    vrf_selector: Arc<RwLock<VrfSelector>>,

    /// Consensus manager
    consensus_manager: Arc<ConsensusManager>,

    /// Epoch manager
    epoch_manager: Arc<EpochManager>,

    /// Verifier pool (list of all available verifiers)
    verifier_pool: Arc<RwLock<Vec<String>>>,

    /// Verifier gRPC clients (verifier_id -> endpoint)
    verifier_endpoints: Arc<HashMap<String, String>>,

    /// Results cache (verification_id -> consensus outcome)
    results_cache: Arc<RwLock<HashMap<String, ConsensusOutcome>>>,
}

impl CoordinatorService {
    pub fn new(
        vrf_selector: VrfSelector,
        consensus_manager: ConsensusManager,
        epoch_manager: EpochManager,
        verifier_pool: Vec<String>,
        verifier_endpoints: HashMap<String, String>,
    ) -> Self {
        Self {
            vrf_selector: Arc::new(RwLock::new(vrf_selector)),
            consensus_manager: Arc::new(consensus_manager),
            epoch_manager: Arc::new(epoch_manager),
            verifier_pool: Arc::new(RwLock::new(verifier_pool)),
            verifier_endpoints: Arc::new(verifier_endpoints),
            results_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Convert common types to proto types
    fn to_proto_outcome(&self, outcome: ConsensusOutcome) -> ProtoConsensusOutcome {
        // Convert votes
        let mut proto_votes = HashMap::new();
        for (k, v) in outcome.votes {
            let proto_vote = bft_verifier::VerifierVote {
                verifier_id: v.verifier_id,
                verification_id: v.verification_id,
                result: match v.result {
                    VerificationResult::Pass => 0,
                    VerificationResult::Fail => 1,
                },
                reason: v.reason,
                timestamp: v.timestamp,
                signature: v.signature,
            };
            proto_votes.insert(k, proto_vote);
        }

        ProtoConsensusOutcome {
            verification_id: outcome.verification_id,
            epoch: outcome.epoch,
            committee: outcome.committee,
            votes: proto_votes,
            result: outcome.result.map(|r| match r {
                VerificationResult::Pass => 0,
                VerificationResult::Fail => 1,
            }),
            quorum_reached: outcome.quorum_reached,
            timestamp: outcome.timestamp,
        }
    }
}

#[tonic::async_trait]
impl Coordinator for CoordinatorService {
    async fn submit_metrics(
        &self,
        request: Request<MetricSubmission>,
    ) -> Result<Response<SubmissionAck>, Status> {
        let submission = request.into_inner();
        let agent_id = submission.agent_id.clone();
        let request_id = submission.request_id.clone();

        info!(
            agent_id = %agent_id,
            request_id = %request_id,
            "Received metric submission"
        );

        // Generate verification ID
        let verification_id = uuid::Uuid::new_v4().to_string();

        // Get current epoch
        let current_epoch = self.epoch_manager.current_epoch();
        info!(epoch = current_epoch, "Current epoch");

        // Select committee for this epoch
        let pool = self.verifier_pool.read().await;
        let selector = self.vrf_selector.read().await;
        let committee = selector.select_committee(current_epoch, &pool);
        drop(pool);
        drop(selector);

        if committee.is_empty() {
            error!("No verifiers available in pool");
            return Ok(Response::new(SubmissionAck {
                verification_id,
                accepted: false,
            }));
        }

        info!(
            committee_size = committee.len(),
            committee = ?committee,
            "Selected committee"
        );

        // Create verification request
        let verification_request = bft_verifier::VerificationRequest {
            verification_id: verification_id.clone(),
            epoch: current_epoch,
            submission: Some(submission),
        };

        // Collect votes from committee (in parallel with early termination)
        let (vote_tx, mut vote_rx) = tokio::sync::mpsc::channel(committee.len());

        for verifier_id in &committee {
            if let Some(endpoint) = self.verifier_endpoints.get(verifier_id) {
                let endpoint = endpoint.clone();
                let verifier_id = verifier_id.clone();
                let req = verification_request.clone();
                let timeout_duration = self.consensus_manager.vote_timeout();
                let tx = vote_tx.clone();

                tokio::spawn(async move {
                    match tokio::time::timeout(
                        timeout_duration,
                        Self::request_vote(endpoint, req),
                    )
                    .await
                    {
                        Ok(Ok(vote)) => {
                            let _ = tx.send(Some((verifier_id, vote))).await;
                        }
                        Ok(Err(e)) => {
                            warn!("Failed to get vote from {}: {}", verifier_id, e);
                            let _ = tx.send(None).await;
                        }
                        Err(_) => {
                            warn!("Timeout getting vote from {}", verifier_id);
                            let _ = tx.send(None).await;
                        }
                    }
                });
            } else {
                warn!("No endpoint found for verifier: {}", verifier_id);
            }
        }

        // Drop sender so receiver knows when all votes are sent
        drop(vote_tx);

        // Collect votes with early termination
        let mut votes = HashMap::new();
        let mut votes_received = 0;
        let total_verifiers = committee.len();
        let global_timeout = tokio::time::sleep(self.consensus_manager.vote_timeout());
        tokio::pin!(global_timeout);

        let early_terminated = loop {
            tokio::select! {
                Some(vote_result) = vote_rx.recv() => {
                    votes_received += 1;

                    if let Some((verifier_id, proto_vote)) = vote_result {
                        // Convert proto vote to common type
                        let vote = VerifierVote::new(
                            proto_vote.verifier_id,
                            proto_vote.verification_id,
                            if proto_vote.result == 0 {
                                VerificationResult::Pass
                            } else {
                                VerificationResult::Fail
                            },
                            proto_vote.reason,
                            proto_vote.timestamp,
                            proto_vote.signature,
                        );
                        votes.insert(verifier_id, vote);

                        // Check for early consensus
                        if let Some(early_result) = self.consensus_manager.has_early_consensus(&votes) {
                            info!(
                                votes_collected = votes.len(),
                                total_verifiers,
                                result = ?early_result,
                                "Early consensus reached!"
                            );
                            break true;
                        }
                    }

                    // All votes received
                    if votes_received >= total_verifiers {
                        info!(votes_collected = votes.len(), "All votes received");
                        break false;
                    }
                }
                _ = &mut global_timeout => {
                    warn!(
                        votes_collected = votes.len(),
                        total_verifiers,
                        "Global vote collection timeout"
                    );
                    break false;
                }
                else => {
                    // Channel closed
                    break false;
                }
            }
        };

        info!(
            votes_received = votes.len(),
            early_terminated,
            "Collected votes"
        );

        // Reach consensus
        let (result, reason) = self.consensus_manager.reach_consensus(&votes);

        let quorum_reached = result.is_some();

        // Store outcome
        let outcome = ConsensusOutcome::new(
            verification_id.clone(),
            current_epoch,
            committee,
            votes,
            result,
            quorum_reached,
            current_timestamp(),
        );

        self.results_cache
            .write()
            .await
            .insert(verification_id.clone(), outcome);

        info!(
            verification_id = %verification_id,
            quorum_reached,
            early_terminated,
            result = ?result,
            "Consensus complete: {}",
            reason
        );

        Ok(Response::new(SubmissionAck {
            verification_id,
            accepted: quorum_reached,
        }))
    }

    async fn query_consensus(
        &self,
        request: Request<VerificationQuery>,
    ) -> Result<Response<ProtoConsensusOutcome>, Status> {
        let query = request.into_inner();
        let cache = self.results_cache.read().await;

        match cache.get(&query.verification_id) {
            Some(outcome) => {
                info!(verification_id = %query.verification_id, "Query result found");
                Ok(Response::new(self.to_proto_outcome(outcome.clone())))
            }
            None => {
                warn!(verification_id = %query.verification_id, "Query result not found");
                Err(Status::not_found("Verification ID not found"))
            }
        }
    }
}

impl CoordinatorService {
    /// Request vote from a verifier (helper function)
    async fn request_vote(
        endpoint: String,
        req: bft_verifier::VerificationRequest,
    ) -> Result<bft_verifier::VerifierVote, Box<dyn std::error::Error + Send + Sync>> {
        use bft_verifier::verifier_client::VerifierClient;

        let mut client = VerifierClient::connect(endpoint).await?;
        let response = client.verify(req).await?;
        Ok(response.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_coordinator_service_creation() {
        let vrf_selector = VrfSelector::new([0u8; 32], 3);
        let consensus_manager = ConsensusManager::new(2, Duration::from_secs(5));
        let epoch_manager = EpochManager::new(Duration::from_secs(30));
        let verifier_pool = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];
        let verifier_endpoints = HashMap::new();

        let _service = CoordinatorService::new(
            vrf_selector,
            consensus_manager,
            epoch_manager,
            verifier_pool,
            verifier_endpoints,
        );

        // Just test that it constructs successfully
    }
}
