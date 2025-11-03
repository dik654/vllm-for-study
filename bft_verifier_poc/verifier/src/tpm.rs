use bft_common::{
    compute_pcr_from_metrics, signature_from_bytes, verify_signature, verifying_key_from_bytes,
    RequestMetrics, TpmQuote, VerificationError,
};
use ed25519_dalek::VerifyingKey;
use std::collections::HashMap;

/// Simulated TPM verifier for PoC
/// In production, this would use real TPM 2.0 APIs (tpm2-tss)
pub struct SimulatedTpmVerifier {
    /// Map of agent_id -> public key for verification
    agent_public_keys: HashMap<String, VerifyingKey>,
}

impl SimulatedTpmVerifier {
    /// Create new TPM verifier with agent public keys
    pub fn new(agent_keys: HashMap<String, VerifyingKey>) -> Self {
        Self {
            agent_public_keys: agent_keys,
        }
    }

    /// Add an agent's public key
    pub fn add_agent_key(&mut self, agent_id: String, public_key: VerifyingKey) {
        self.agent_public_keys.insert(agent_id, public_key);
    }

    /// Verify TPM quote for given metrics
    ///
    /// Process:
    /// 1. Check agent is known
    /// 2. Recompute PCR from metrics
    /// 3. Verify quote signature
    ///
    /// Returns Ok(true) if verification succeeds
    pub fn verify_quote(
        &self,
        agent_id: &str,
        metrics: &RequestMetrics,
        quote: &TpmQuote,
    ) -> Result<bool, VerificationError> {
        // 1. Get agent's public key
        let public_key = self
            .agent_public_keys
            .get(agent_id)
            .ok_or_else(|| VerificationError::UnknownAgent(agent_id.to_string()))?;

        // 2. Compute expected PCR from metrics
        let expected_pcr = compute_pcr_from_metrics(metrics);

        // 3. Get PCR value from quote (we use PCR index 10 for metrics)
        let quote_pcr = quote
            .get_pcr(10)
            .ok_or(VerificationError::MissingPcr(10))?;

        // 4. Verify PCR matches
        if quote_pcr != &expected_pcr {
            return Err(VerificationError::InvalidPcr);
        }

        // 5. Reconstruct message that was signed (PCR + nonce)
        let mut message = expected_pcr.clone();
        message.extend_from_slice(&quote.nonce);

        // 6. Parse signature from quote
        let signature = signature_from_bytes(&quote.quote_signature)?;

        // 7. Verify signature
        verify_signature(public_key, &message, &signature)?;

        Ok(true)
    }

    /// Get number of registered agents
    pub fn agent_count(&self) -> usize {
        self.agent_public_keys.len()
    }

    /// Check if agent is registered
    pub fn has_agent(&self, agent_id: &str) -> bool {
        self.agent_public_keys.contains_key(agent_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bft_common::{generate_keypair, generate_nonce, sign_message, signature_to_bytes};

    #[test]
    fn test_verify_valid_quote() {
        // Setup: Create agent keypair
        let (agent_signing_key, agent_verifying_key) = generate_keypair();
        let agent_id = "agent-1";

        // Create verifier with agent's public key
        let mut keys = HashMap::new();
        keys.insert(agent_id.to_string(), agent_verifying_key);
        let verifier = SimulatedTpmVerifier::new(keys);

        // Create metrics
        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);

        // Agent generates quote
        let pcr = compute_pcr_from_metrics(&metrics);
        let nonce = generate_nonce();
        let mut message = pcr.clone();
        message.extend_from_slice(&nonce);
        let signature = sign_message(&agent_signing_key, &message);

        let mut pcr_values = HashMap::new();
        pcr_values.insert(10, pcr);
        let quote = TpmQuote::new(pcr_values, signature_to_bytes(&signature).to_vec(), nonce);

        // Verify
        let result = verifier.verify_quote(agent_id, &metrics, &quote);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_verify_unknown_agent() {
        let verifier = SimulatedTpmVerifier::new(HashMap::new());
        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let quote = TpmQuote::new(HashMap::new(), vec![], vec![]);

        let result = verifier.verify_quote("unknown-agent", &metrics, &quote);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VerificationError::UnknownAgent(_)
        ));
    }

    #[test]
    fn test_verify_invalid_signature() {
        // Setup: Two different keypairs
        let (agent_signing_key, _) = generate_keypair();
        let (_, wrong_verifying_key) = generate_keypair(); // Different key

        let agent_id = "agent-1";
        let mut keys = HashMap::new();
        keys.insert(agent_id.to_string(), wrong_verifying_key); // Wrong key
        let verifier = SimulatedTpmVerifier::new(keys);

        // Create metrics and quote with agent_signing_key
        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let pcr = compute_pcr_from_metrics(&metrics);
        let nonce = generate_nonce();
        let mut message = pcr.clone();
        message.extend_from_slice(&nonce);
        let signature = sign_message(&agent_signing_key, &message);

        let mut pcr_values = HashMap::new();
        pcr_values.insert(10, pcr);
        let quote = TpmQuote::new(pcr_values, signature_to_bytes(&signature).to_vec(), nonce);

        // Verify should fail (signature won't verify with wrong key)
        let result = verifier.verify_quote(agent_id, &metrics, &quote);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_tampered_metrics() {
        // Setup
        let (agent_signing_key, agent_verifying_key) = generate_keypair();
        let agent_id = "agent-1";
        let mut keys = HashMap::new();
        keys.insert(agent_id.to_string(), agent_verifying_key);
        let verifier = SimulatedTpmVerifier::new(keys);

        // Original metrics
        let original_metrics = RequestMetrics::new(100, 50, 1000, 0.001);

        // Agent generates quote for original metrics
        let pcr = compute_pcr_from_metrics(&original_metrics);
        let nonce = generate_nonce();
        let mut message = pcr.clone();
        message.extend_from_slice(&nonce);
        let signature = sign_message(&agent_signing_key, &message);

        let mut pcr_values = HashMap::new();
        pcr_values.insert(10, pcr);
        let quote = TpmQuote::new(pcr_values, signature_to_bytes(&signature).to_vec(), nonce);

        // Attacker tampers with metrics
        let tampered_metrics = RequestMetrics::new(200, 50, 1000, 0.001); // Changed!

        // Verify should fail (PCR won't match)
        let result = verifier.verify_quote(agent_id, &tampered_metrics, &quote);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), VerificationError::InvalidPcr));
    }

    #[test]
    fn test_missing_pcr() {
        let (_, agent_verifying_key) = generate_keypair();
        let agent_id = "agent-1";
        let mut keys = HashMap::new();
        keys.insert(agent_id.to_string(), agent_verifying_key);
        let verifier = SimulatedTpmVerifier::new(keys);

        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);
        let quote = TpmQuote::new(HashMap::new(), vec![], vec![]); // No PCR values

        let result = verifier.verify_quote(agent_id, &metrics, &quote);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            VerificationError::MissingPcr(10)
        ));
    }

    #[test]
    fn test_agent_management() {
        let mut verifier = SimulatedTpmVerifier::new(HashMap::new());
        assert_eq!(verifier.agent_count(), 0);

        let (_, key1) = generate_keypair();
        verifier.add_agent_key("agent-1".to_string(), key1);
        assert_eq!(verifier.agent_count(), 1);
        assert!(verifier.has_agent("agent-1"));
        assert!(!verifier.has_agent("agent-2"));

        let (_, key2) = generate_keypair();
        verifier.add_agent_key("agent-2".to_string(), key2);
        assert_eq!(verifier.agent_count(), 2);
        assert!(verifier.has_agent("agent-2"));
    }
}
