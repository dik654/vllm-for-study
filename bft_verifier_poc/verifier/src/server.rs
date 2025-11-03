use crate::tpm::SimulatedTpmVerifier;
use crate::validator::{validate_agent_id, validate_metrics, validate_request_id, validate_timestamp};
use bft_common::{current_timestamp, sign_message, signature_to_bytes, VerificationResult};
use ed25519_dalek::SigningKey;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{error, info, warn};

// Generated proto types will be available after compilation
// For now, we'll define the structure assuming tonic-build will generate:
// - bft_verifier::verifier_server::{Verifier, VerifierServer}
// - bft_verifier::{VerificationRequest, VerifierVote, Empty, VerifierStatus, VerificationResult as ProtoResult}

pub mod bft_verifier {
    tonic::include_proto!("bft_verifier");
}

use bft_verifier::verifier_server::Verifier;
use bft_verifier::{Empty, VerificationRequest, VerifierStatus, VerifierVote};

/// Verifier service implementation
pub struct VerifierService {
    /// This verifier's ID
    verifier_id: String,

    /// TPM verifier for quote validation
    tpm_verifier: Arc<RwLock<SimulatedTpmVerifier>>,

    /// Signing key for vote signatures
    signing_key: SigningKey,

    /// Statistics
    verified_count: Arc<RwLock<u64>>,
}

impl VerifierService {
    pub fn new(
        verifier_id: String,
        tpm_verifier: SimulatedTpmVerifier,
        signing_key: SigningKey,
    ) -> Self {
        Self {
            verifier_id,
            tpm_verifier: Arc::new(RwLock::new(tpm_verifier)),
            signing_key,
            verified_count: Arc::new(RwLock::new(0)),
        }
    }

    /// Convert common VerificationResult to proto enum
    fn to_proto_result(result: VerificationResult) -> i32 {
        match result {
            VerificationResult::Pass => 0, // PASS = 0 in proto
            VerificationResult::Fail => 1, // FAIL = 1 in proto
        }
    }
}

#[tonic::async_trait]
impl Verifier for VerifierService {
    async fn verify(
        &self,
        request: Request<VerificationRequest>,
    ) -> Result<Response<VerifierVote>, Status> {
        let req = request.into_inner();
        let verification_id = req.verification_id.clone();

        info!(
            verifier_id = %self.verifier_id,
            verification_id = %verification_id,
            "Received verification request"
        );

        // Extract submission
        let submission = req
            .submission
            .ok_or_else(|| Status::invalid_argument("Missing submission"))?;

        let agent_id = submission.agent_id.clone();
        let request_id = submission.request_id.clone();

        // Step 1: Validate agent ID and request ID
        if let Err(e) = validate_agent_id(&agent_id) {
            warn!("Invalid agent_id: {}", e);
            return self.create_vote_response(
                verification_id,
                VerificationResult::Fail,
                format!("Invalid agent_id: {}", e),
            );
        }

        if let Err(e) = validate_request_id(&request_id) {
            warn!("Invalid request_id: {}", e);
            return self.create_vote_response(
                verification_id,
                VerificationResult::Fail,
                format!("Invalid request_id: {}", e),
            );
        }

        // Step 2: Validate timestamp
        if let Err(e) = validate_timestamp(submission.timestamp) {
            warn!("Invalid timestamp: {}", e);
            return self.create_vote_response(
                verification_id,
                VerificationResult::Fail,
                format!("Invalid timestamp: {}", e),
            );
        }

        // Step 3: Extract and validate metrics
        let proto_metrics = submission
            .metrics
            .ok_or_else(|| Status::invalid_argument("Missing metrics"))?;

        let metrics = bft_common::RequestMetrics::new(
            proto_metrics.prompt_tokens,
            proto_metrics.completion_tokens,
            proto_metrics.e2e_latency_ms,
            proto_metrics.estimated_cost,
        );

        if let Err(e) = validate_metrics(&metrics) {
            warn!("Invalid metrics: {}", e);
            return self.create_vote_response(
                verification_id,
                VerificationResult::Fail,
                format!("Invalid metrics: {}", e),
            );
        }

        // Step 4: Extract TPM quote
        let proto_quote = submission
            .quote
            .ok_or_else(|| Status::invalid_argument("Missing quote"))?;

        let quote = bft_common::TpmQuote::new(
            proto_quote.pcr_values,
            proto_quote.quote_signature,
            proto_quote.nonce,
        );

        // Step 5: Verify TPM quote
        let tpm_verifier = self.tpm_verifier.read().await;
        match tpm_verifier.verify_quote(&agent_id, &metrics, &quote) {
            Ok(true) => {
                info!(
                    verifier_id = %self.verifier_id,
                    verification_id = %verification_id,
                    agent_id = %agent_id,
                    "Verification PASSED"
                );

                // Increment verified count
                let mut count = self.verified_count.write().await;
                *count += 1;

                self.create_vote_response(
                    verification_id,
                    VerificationResult::Pass,
                    "All checks passed".to_string(),
                )
            }
            Ok(false) => {
                warn!("Verification returned false (should not happen)");
                self.create_vote_response(
                    verification_id,
                    VerificationResult::Fail,
                    "Verification failed unexpectedly".to_string(),
                )
            }
            Err(e) => {
                warn!(
                    verifier_id = %self.verifier_id,
                    verification_id = %verification_id,
                    error = %e,
                    "Verification FAILED"
                );

                self.create_vote_response(
                    verification_id,
                    VerificationResult::Fail,
                    format!("Verification error: {}", e),
                )
            }
        }
    }

    async fn heartbeat(&self, _request: Request<Empty>) -> Result<Response<VerifierStatus>, Status> {
        let count = *self.verified_count.read().await;

        let status = VerifierStatus {
            verifier_id: self.verifier_id.clone(),
            healthy: true,
            verified_count: count,
        };

        Ok(Response::new(status))
    }
}

impl VerifierService {
    /// Create a signed vote response
    fn create_vote_response(
        &self,
        verification_id: String,
        result: VerificationResult,
        reason: String,
    ) -> Result<Response<VerifierVote>, Status> {
        let timestamp = current_timestamp();

        // Create message to sign: verification_id + result + timestamp
        let mut message = verification_id.as_bytes().to_vec();
        message.push(result as u8);
        message.extend_from_slice(&timestamp.to_le_bytes());

        // Sign the vote
        let signature = sign_message(&self.signing_key, &message);
        let signature_bytes = signature_to_bytes(&signature).to_vec();

        let vote = VerifierVote {
            verifier_id: self.verifier_id.clone(),
            verification_id,
            result: Self::to_proto_result(result),
            reason,
            timestamp,
            signature: signature_bytes,
        };

        Ok(Response::new(vote))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_common::{generate_keypair, RequestMetrics};
    use std::collections::HashMap;

    #[test]
    fn test_verifier_service_creation() {
        let (signing_key, _) = generate_keypair();
        let tpm_verifier = SimulatedTpmVerifier::new(HashMap::new());

        let service = VerifierService::new(
            "verifier-1".to_string(),
            tpm_verifier,
            signing_key,
        );

        assert_eq!(service.verifier_id, "verifier-1");
    }

    #[test]
    fn test_to_proto_result() {
        assert_eq!(VerifierService::to_proto_result(VerificationResult::Pass), 0);
        assert_eq!(VerifierService::to_proto_result(VerificationResult::Fail), 1);
    }

    // Note: Full integration tests would require running gRPC server
    // Those will be in integration tests (Phase 5)
}
