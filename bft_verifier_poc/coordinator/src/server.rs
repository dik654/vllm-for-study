use crate::agent_tracker::{AgentTracker, AgentTrackerConfig};
use crate::connection_pool::VerifierConnectionPool;
use crate::consensus::ConsensusManager;
use crate::epoch::EpochManager;
use crate::nonce_manager::NonceManager;
use crate::vrf::VrfSelector;
use bft_common::{current_timestamp, ConsensusOutcome, VerificationResult, VerifierVote};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

pub mod bft_verifier {
    tonic::include_proto!("bft_verifier");
}

use bft_verifier::coordinator_server::Coordinator;
use bft_verifier::{ConsensusOutcome as ProtoConsensusOutcome, MetricSubmission, SubmissionAck, VerificationQuery};

/// Handle for background verification tasks
///
/// This is a lightweight clone of CoordinatorService that can be moved
/// into background tasks for async verification.
#[derive(Clone)]
struct CoordinatorServiceHandle {
    vrf_selector: Arc<RwLock<VrfSelector>>,
    consensus_manager: Arc<ConsensusManager>,
    epoch_manager: Arc<EpochManager>,
    verifier_pool: Arc<RwLock<Vec<String>>>,
    connection_pool: Arc<VerifierConnectionPool>,
    results_cache: Arc<RwLock<HashMap<String, ConsensusOutcome>>>,
    speculative_execution: bool,
    nonce_manager: Arc<NonceManager>,
    agent_tracker: Arc<AgentTracker>,
}

impl CoordinatorServiceHandle {
    /// Verify metrics in background (async mode)
    ///
    /// This performs the full verification process asynchronously
    /// without blocking the caller.
    async fn verify_in_background(
        &self,
        verification_id: String,
        submission: bft_verifier::MetricSubmission,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Get current epoch
        let current_epoch = self.epoch_manager.current_epoch();

        // Select committee
        let pool = self.verifier_pool.read().await;
        let selector = self.vrf_selector.read().await;
        let committee = selector.select_committee(current_epoch, &pool);
        drop(pool);
        drop(selector);

        if committee.is_empty() {
            return Err("No verifiers available in pool".into());
        }

        debug!(
            verification_id = %verification_id,
            committee_size = committee.len(),
            "Background verification: committee selected"
        );

        // Create verification request
        let verification_request = bft_verifier::VerificationRequest {
            verification_id: verification_id.clone(),
            epoch: current_epoch,
            submission: Some(submission),
        };

        // Collect votes (reuse existing logic)
        let (vote_tx, mut vote_rx) = tokio::sync::mpsc::channel(committee.len());

        for verifier_id in &committee {
            let verifier_id = verifier_id.clone();
            let req = verification_request.clone();
            let timeout_duration = self.consensus_manager.vote_timeout();
            let tx = vote_tx.clone();
            let connection_pool = Arc::clone(&self.connection_pool);

            tokio::spawn(async move {
                match tokio::time::timeout(
                    timeout_duration,
                    CoordinatorService::request_vote_with_pool(connection_pool, verifier_id.clone(), req),
                )
                .await
                {
                    Ok(Ok(vote)) => {
                        let _ = tx.send(Some((verifier_id, vote))).await;
                    }
                    Ok(Err(e)) => {
                        warn!("Background: Failed to get vote: {}", e);
                        let _ = tx.send(None).await;
                    }
                    Err(_) => {
                        warn!("Background: Timeout getting vote");
                        let _ = tx.send(None).await;
                    }
                }
            });
        }

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
                        if let Some(_early_result) = self.consensus_manager.has_early_consensus(&votes) {
                            debug!(
                                verification_id = %verification_id,
                                "Background: Early consensus reached"
                            );
                            break true;
                        }
                    }

                    if votes_received >= total_verifiers {
                        break false;
                    }
                }
                _ = &mut global_timeout => {
                    warn!(
                        verification_id = %verification_id,
                        "Background: Vote collection timeout"
                    );
                    break false;
                }
                else => {
                    break false;
                }
            }
        };

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
            "Background verification complete: {}",
            reason
        );

        Ok(())
    }
}

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

    /// Connection pool for verifier gRPC clients
    connection_pool: Arc<VerifierConnectionPool>,

    /// Results cache (verification_id -> consensus outcome)
    results_cache: Arc<RwLock<HashMap<String, ConsensusOutcome>>>,

    /// Enable speculative execution (pre-warm next epoch's committee)
    speculative_execution: bool,

    /// Nonce manager for replay attack prevention
    nonce_manager: Arc<NonceManager>,

    /// Agent tracker for rate limiting and anomaly detection
    agent_tracker: Arc<AgentTracker>,
}

impl CoordinatorService {
    pub fn new(
        vrf_selector: VrfSelector,
        consensus_manager: ConsensusManager,
        epoch_manager: EpochManager,
        verifier_pool: Vec<String>,
        verifier_endpoints: HashMap<String, String>,
    ) -> Self {
        Self::with_options(
            vrf_selector,
            consensus_manager,
            epoch_manager,
            verifier_pool,
            verifier_endpoints,
            true, // Enable speculative execution by default
        )
    }

    pub fn with_options(
        vrf_selector: VrfSelector,
        consensus_manager: ConsensusManager,
        epoch_manager: EpochManager,
        verifier_pool: Vec<String>,
        verifier_endpoints: HashMap<String, String>,
        speculative_execution: bool,
    ) -> Self {
        // Create connection pool
        let max_connections = verifier_pool.len().max(10);
        let connection_pool = VerifierConnectionPool::new(verifier_endpoints, max_connections);

        // Initialize security modules
        let nonce_manager = NonceManager::new();
        let agent_tracker = AgentTracker::new(AgentTrackerConfig::default());

        Self {
            vrf_selector: Arc::new(RwLock::new(vrf_selector)),
            consensus_manager: Arc::new(consensus_manager),
            epoch_manager: Arc::new(epoch_manager),
            verifier_pool: Arc::new(RwLock::new(verifier_pool)),
            connection_pool: Arc::new(connection_pool),
            results_cache: Arc::new(RwLock::new(HashMap::new())),
            speculative_execution,
            nonce_manager: Arc::new(nonce_manager),
            agent_tracker: Arc::new(agent_tracker),
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
        let async_mode = submission.async_mode;

        info!(
            agent_id = %agent_id,
            request_id = %request_id,
            async_mode = async_mode,
            "Received metric submission"
        );

        // Security validation: Nonce (if provided)
        if !submission.coordinator_nonce.is_empty() {
            if let Err(e) = self.nonce_manager.validate_and_consume(&submission.coordinator_nonce).await {
                warn!(
                    agent_id = %agent_id,
                    error = %e,
                    "Nonce validation failed"
                );
                return Ok(Response::new(SubmissionAck {
                    verification_id: String::new(),
                    accepted: false,
                    status: 2, // FAILED
                    message: format!("Nonce validation failed: {}", e),
                }));
            }
        }

        // Security validation: Agent tracker (rate limiting & anomaly detection)
        if let Some(ref metrics) = submission.metrics {
            let request_metrics = bft_common::RequestMetrics::new(
                metrics.prompt_tokens,
                metrics.completion_tokens,
                metrics.e2e_latency_ms,
                metrics.estimated_cost,
            );

            if let Err(e) = self.agent_tracker.validate_submission(&agent_id, &request_metrics).await {
                warn!(
                    agent_id = %agent_id,
                    error = %e,
                    "Agent tracker validation failed"
                );
                return Ok(Response::new(SubmissionAck {
                    verification_id: String::new(),
                    accepted: false,
                    status: 2, // FAILED
                    message: format!("Rate limit or anomaly detected: {}", e),
                }));
            }
        }

        // Generate verification ID
        let verification_id = uuid::Uuid::new_v4().to_string();

        // Async mode: return immediately and verify in background
        if async_mode {
            info!(
                verification_id = %verification_id,
                "Async mode: returning immediately, verification in background"
            );

            // Clone necessary data for background task
            let verification_id_bg = verification_id.clone();
            let submission_bg = bft_verifier::MetricSubmission {
                agent_id: submission.agent_id,
                request_id: submission.request_id,
                timestamp: submission.timestamp,
                metrics: submission.metrics,
                quote: submission.quote,
                async_mode: false, // Process synchronously in background
            };

            // Spawn background verification task
            let coordinator_service = CoordinatorServiceHandle {
                vrf_selector: Arc::clone(&self.vrf_selector),
                consensus_manager: Arc::clone(&self.consensus_manager),
                epoch_manager: Arc::clone(&self.epoch_manager),
                verifier_pool: Arc::clone(&self.verifier_pool),
                connection_pool: Arc::clone(&self.connection_pool),
                results_cache: Arc::clone(&self.results_cache),
                speculative_execution: self.speculative_execution,
                nonce_manager: Arc::clone(&self.nonce_manager),
                agent_tracker: Arc::clone(&self.agent_tracker),
            };

            tokio::spawn(async move {
                match coordinator_service
                    .verify_in_background(verification_id_bg.clone(), submission_bg)
                    .await
                {
                    Ok(_) => {
                        info!(
                            verification_id = %verification_id_bg,
                            "Background verification completed successfully"
                        );
                    }
                    Err(e) => {
                        error!(
                            verification_id = %verification_id_bg,
                            error = %e,
                            "Background verification failed"
                        );
                    }
                }
            });

            // Return immediately with PENDING status
            return Ok(Response::new(SubmissionAck {
                verification_id,
                accepted: false, // Deprecated field
                status: 0,       // PENDING
                message: "Verification in progress (async mode)".to_string(),
            }));
        }

        // Sync mode: existing synchronous logic below
        info!(
            verification_id = %verification_id,
            "Sync mode: waiting for verification to complete"
        );

        // Generate verification ID (already done above)
        // let verification_id = uuid::Uuid::new_v4().to_string();

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

        // Speculative execution: pre-warm next epoch's committee
        if self.speculative_execution {
            let next_epoch = current_epoch + 1;
            let next_committee = selector.select_committee(next_epoch, &pool);

            if !next_committee.is_empty() {
                debug!(
                    next_epoch,
                    next_committee_size = next_committee.len(),
                    "Pre-warming next epoch's committee"
                );

                let connection_pool = Arc::clone(&self.connection_pool);
                tokio::spawn(async move {
                    connection_pool.warm_up(&next_committee).await;
                });
            }
        }

        // Create verification request
        let verification_request = bft_verifier::VerificationRequest {
            verification_id: verification_id.clone(),
            epoch: current_epoch,
            submission: Some(submission),
        };

        // Collect votes from committee (in parallel with early termination)
        let (vote_tx, mut vote_rx) = tokio::sync::mpsc::channel(committee.len());

        for verifier_id in &committee {
            let verifier_id = verifier_id.clone();
            let req = verification_request.clone();
            let timeout_duration = self.consensus_manager.vote_timeout();
            let tx = vote_tx.clone();
            let connection_pool = Arc::clone(&self.connection_pool);

            tokio::spawn(async move {
                match tokio::time::timeout(
                    timeout_duration,
                    Self::request_vote_with_pool(connection_pool, verifier_id.clone(), req),
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

        // Return with appropriate status
        let (status, message) = if quorum_reached {
            match result {
                Some(VerificationResult::Pass) => (1, format!("Verification passed: {}", reason)),
                Some(VerificationResult::Fail) => (2, format!("Verification failed: {}", reason)),
                None => (2, "No consensus reached".to_string()),
            }
        } else {
            (2, format!("Quorum not reached: {}", reason))
        };

        Ok(Response::new(SubmissionAck {
            verification_id,
            accepted: quorum_reached,  // Deprecated but kept for compatibility
            status,
            message,
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
    /// Request vote from a verifier using connection pool
    ///
    /// Uses the connection pool to reuse existing connections,
    /// avoiding connection overhead on each request.
    async fn request_vote_with_pool(
        connection_pool: Arc<VerifierConnectionPool>,
        verifier_id: String,
        req: bft_verifier::VerificationRequest,
    ) -> Result<bft_verifier::VerifierVote, Box<dyn std::error::Error + Send + Sync>> {
        // Get client from pool (reuses existing connection or creates new one)
        let mut client = connection_pool.get_client(&verifier_id).await?;

        // Send verification request
        let response = client.verify(req).await?;
        Ok(response.into_inner())
    }

    /// Request vote from a verifier (legacy - direct connection)
    #[allow(dead_code)]
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
