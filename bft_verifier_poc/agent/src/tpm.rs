use bft_common::{
    compute_pcr_from_metrics, generate_nonce, sign_message, signature_to_bytes, RequestMetrics,
    TpmQuote,
};
use ed25519_dalek::SigningKey;
use std::collections::HashMap;

/// Simulated TPM agent for quote generation
///
/// In production, this would use real TPM 2.0 APIs
pub struct SimulatedTpmAgent {
    /// Agent's signing key (simulates TPM Attestation Identity Key)
    signing_key: SigningKey,
}

impl SimulatedTpmAgent {
    /// Create new TPM agent with signing key
    pub fn new(signing_key: SigningKey) -> Self {
        Self { signing_key }
    }

    /// Generate TPM quote for metrics
    ///
    /// Process:
    /// 1. Compute PCR value from metrics
    /// 2. Generate nonce for freshness
    /// 3. Sign PCR + nonce with TPM key
    /// 4. Return TPM quote
    pub fn generate_quote(&self, metrics: &RequestMetrics) -> TpmQuote {
        // 1. Compute PCR from metrics (PCR extension)
        let pcr_value = compute_pcr_from_metrics(metrics);

        // 2. Generate fresh nonce (prevents replay attacks)
        let nonce = generate_nonce();

        // 3. Create message to sign: PCR + nonce
        let mut message = pcr_value.clone();
        message.extend_from_slice(&nonce);

        // 4. Sign with TPM key
        let signature = sign_message(&self.signing_key, &message);
        let signature_bytes = signature_to_bytes(&signature).to_vec();

        // 5. Create TPM quote (we use PCR index 10 for metrics)
        let mut pcr_values = HashMap::new();
        pcr_values.insert(10, pcr_value);

        TpmQuote::new(pcr_values, signature_bytes, nonce)
    }

    /// Get signing key reference (for testing)
    #[cfg(test)]
    pub fn signing_key(&self) -> &SigningKey {
        &self.signing_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_common::{generate_keypair, signature_from_bytes, verify_signature};

    #[test]
    fn test_generate_quote() {
        let (signing_key, verifying_key) = generate_keypair();
        let agent = SimulatedTpmAgent::new(signing_key);

        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let quote = agent.generate_quote(&metrics);

        // Check quote structure
        assert!(quote.get_pcr(10).is_some());
        assert!(!quote.quote_signature.is_empty());
        assert_eq!(quote.nonce.len(), 32);

        // Verify signature
        let pcr = quote.get_pcr(10).unwrap();
        let mut message = pcr.clone();
        message.extend_from_slice(&quote.nonce);

        let signature = signature_from_bytes(&quote.quote_signature).unwrap();
        assert!(verify_signature(&verifying_key, &message, &signature).is_ok());
    }

    #[test]
    fn test_quote_unique_for_different_metrics() {
        let (signing_key, _) = generate_keypair();
        let agent = SimulatedTpmAgent::new(signing_key);

        let metrics1 = RequestMetrics::new(100, 50, 1000, 0.001);
        let metrics2 = RequestMetrics::new(200, 50, 1000, 0.001);

        let quote1 = agent.generate_quote(&metrics1);
        let quote2 = agent.generate_quote(&metrics2);

        // Different metrics should produce different PCR values
        assert_ne!(quote1.get_pcr(10), quote2.get_pcr(10));
    }

    #[test]
    fn test_quote_unique_nonces() {
        let (signing_key, _) = generate_keypair();
        let agent = SimulatedTpmAgent::new(signing_key);

        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);

        let quote1 = agent.generate_quote(&metrics);
        let quote2 = agent.generate_quote(&metrics);

        // Same metrics but different quotes (due to nonce)
        assert_ne!(quote1.nonce, quote2.nonce);
    }

    #[test]
    fn test_quote_verifiable() {
        let (signing_key, verifying_key) = generate_keypair();
        let agent = SimulatedTpmAgent::new(signing_key);

        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let quote = agent.generate_quote(&metrics);

        // Extract PCR and nonce
        let pcr = quote.get_pcr(10).unwrap();
        let mut message = pcr.clone();
        message.extend_from_slice(&quote.nonce);

        // Verify signature
        let signature = signature_from_bytes(&quote.quote_signature).unwrap();
        let result = verify_signature(&verifying_key, &message, &signature);

        assert!(result.is_ok());
    }

    #[test]
    fn test_tampered_metrics_not_verifiable() {
        let (signing_key, verifying_key) = generate_keypair();
        let agent = SimulatedTpmAgent::new(signing_key);

        // Original metrics
        let original_metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let quote = agent.generate_quote(&original_metrics);

        // Tampered metrics
        let tampered_metrics = RequestMetrics::new(200, 50, 1000, 0.001);
        let tampered_pcr = compute_pcr_from_metrics(&tampered_metrics);

        // Try to verify with tampered PCR
        let mut message = tampered_pcr;
        message.extend_from_slice(&quote.nonce);

        let signature = signature_from_bytes(&quote.quote_signature).unwrap();
        let result = verify_signature(&verifying_key, &message, &signature);

        // Should fail verification
        assert!(result.is_err());
    }

    #[test]
    fn test_pcr_index_10() {
        let (signing_key, _) = generate_keypair();
        let agent = SimulatedTpmAgent::new(signing_key);

        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let quote = agent.generate_quote(&metrics);

        // Should have PCR at index 10
        assert!(quote.pcr_values.contains_key(&10));
        assert_eq!(quote.pcr_values.len(), 1);
    }

    #[test]
    fn test_consistent_pcr_for_same_metrics() {
        let (signing_key, _) = generate_keypair();
        let agent = SimulatedTpmAgent::new(signing_key);

        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);

        let quote1 = agent.generate_quote(&metrics);
        let quote2 = agent.generate_quote(&metrics);

        // PCR should be same for same metrics
        assert_eq!(quote1.get_pcr(10), quote2.get_pcr(10));
    }
}
