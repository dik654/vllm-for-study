use anyhow::Result;
use bft_common::{current_timestamp, MetricSubmission, RequestMetrics, TpmQuote};
use tonic::transport::Channel;
use tracing::{info, warn};

pub mod bft_verifier {
    tonic::include_proto!("bft_verifier");
}

use bft_verifier::coordinator_client::CoordinatorClient as GrpcClient;
use bft_verifier::{MetricSubmission as ProtoSubmission, SubmissionAck, VerificationQuery};

/// Client for submitting metrics to coordinator
pub struct CoordinatorClient {
    client: GrpcClient<Channel>,
    agent_id: String,
}

impl CoordinatorClient {
    /// Connect to coordinator
    pub async fn connect(endpoint: String, agent_id: String) -> Result<Self> {
        info!("Connecting to coordinator at {}", endpoint);
        let client = GrpcClient::connect(endpoint).await?;
        info!("Connected to coordinator");

        Ok(Self { client, agent_id })
    }

    /// Submit metrics with TPM quote (synchronous mode)
    ///
    /// Waits for verification to complete before returning.
    /// Returns (verification_id, accepted).
    pub async fn submit_metrics(
        &mut self,
        request_id: String,
        metrics: RequestMetrics,
        quote: TpmQuote,
    ) -> Result<(String, bool)> {
        self.submit_metrics_with_mode(request_id, metrics, quote, false)
            .await
    }

    /// Submit metrics with TPM quote (asynchronous mode)
    ///
    /// Returns immediately with verification_id.
    /// Verification happens in background.
    /// Use query_result() to check status later.
    pub async fn submit_metrics_async(
        &mut self,
        request_id: String,
        metrics: RequestMetrics,
        quote: TpmQuote,
    ) -> Result<String> {
        let (verification_id, _) = self
            .submit_metrics_with_mode(request_id, metrics, quote, true)
            .await?;
        Ok(verification_id)
    }

    /// Internal method to submit with configurable async mode
    async fn submit_metrics_with_mode(
        &mut self,
        request_id: String,
        metrics: RequestMetrics,
        quote: TpmQuote,
        async_mode: bool,
    ) -> Result<(String, bool)> {
        // Convert to proto types
        let proto_metrics = bft_verifier::RequestMetrics {
            prompt_tokens: metrics.prompt_tokens,
            completion_tokens: metrics.completion_tokens,
            e2e_latency_ms: metrics.e2e_latency_ms,
            estimated_cost: metrics.estimated_cost,
        };

        let proto_quote = bft_verifier::TpmQuote {
            pcr_values: quote.pcr_values,
            quote_signature: quote.quote_signature,
            nonce: quote.nonce,
        };

        let submission = ProtoSubmission {
            agent_id: self.agent_id.clone(),
            request_id: request_id.clone(),
            timestamp: current_timestamp(),
            metrics: Some(proto_metrics),
            quote: Some(proto_quote),
            async_mode,
        };

        info!(
            agent_id = %self.agent_id,
            request_id = %request_id,
            async_mode = async_mode,
            "Submitting metrics"
        );

        let response = self.client.submit_metrics(submission).await?;
        let ack = response.into_inner();

        let status_str = match ack.status {
            0 => "PENDING",
            1 => "ACCEPTED",
            2 => "REJECTED",
            _ => "UNKNOWN",
        };

        info!(
            verification_id = %ack.verification_id,
            accepted = ack.accepted,
            status = status_str,
            message = %ack.message,
            "Received acknowledgment"
        );

        Ok((ack.verification_id, ack.accepted))
    }

    /// Query verification result
    pub async fn query_result(
        &mut self,
        verification_id: String,
    ) -> Result<Option<VerificationResult>> {
        let query = VerificationQuery {
            verification_id: verification_id.clone(),
        };

        info!(verification_id = %verification_id, "Querying result");

        match self.client.query_consensus(query).await {
            Ok(response) => {
                let outcome = response.into_inner();

                if outcome.quorum_reached {
                    let result = match outcome.result {
                        Some(0) => VerificationResult::Pass,
                        Some(1) => VerificationResult::Fail,
                        _ => return Ok(None),
                    };

                    info!(
                        verification_id = %verification_id,
                        result = ?result,
                        votes = outcome.votes.len(),
                        "Query successful"
                    );

                    Ok(Some(result))
                } else {
                    warn!(verification_id = %verification_id, "No quorum reached");
                    Ok(None)
                }
            }
            Err(e) => {
                warn!(verification_id = %verification_id, error = %e, "Query failed");
                Err(e.into())
            }
        }
    }

    /// Get agent ID
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationResult {
    Pass,
    Fail,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_result() {
        let result1 = VerificationResult::Pass;
        let result2 = VerificationResult::Fail;

        assert_ne!(result1, result2);
        assert_eq!(result1, VerificationResult::Pass);
    }

    // Note: Full integration tests with real gRPC server will be in Phase 5
}
