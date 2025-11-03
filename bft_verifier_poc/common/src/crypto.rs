use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::OsRng;
use sha2::{Digest, Sha256};

use crate::error::VerificationError;
use crate::types::RequestMetrics;

/// Generate a new Ed25519 keypair
pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

/// Sign a message with Ed25519
pub fn sign_message(signing_key: &SigningKey, message: &[u8]) -> Signature {
    signing_key.sign(message)
}

/// Verify an Ed25519 signature
pub fn verify_signature(
    verifying_key: &VerifyingKey,
    message: &[u8],
    signature: &Signature,
) -> Result<(), VerificationError> {
    verifying_key
        .verify(message, signature)
        .map_err(|_| VerificationError::InvalidSignature)
}

/// Compute SHA-256 hash
pub fn compute_sha256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

/// Compute PCR value from metrics (simulated TPM PCR extension)
/// In real TPM, this would be: PCR_new = SHA256(PCR_old || data)
/// For PoC, we just hash the metrics data
pub fn compute_pcr_from_metrics(metrics: &RequestMetrics) -> Vec<u8> {
    let mut hasher = Sha256::new();

    // Hash token counts
    hasher.update(metrics.prompt_tokens.to_le_bytes());
    hasher.update(metrics.completion_tokens.to_le_bytes());

    // Hash latency
    hasher.update(metrics.e2e_latency_ms.to_le_bytes());

    // Hash cost (convert to bits for deterministic hashing)
    hasher.update(metrics.estimated_cost.to_le_bytes());

    hasher.finalize().to_vec()
}

/// Serialize signing key to bytes (32 bytes)
pub fn signing_key_to_bytes(key: &SigningKey) -> [u8; 32] {
    key.to_bytes()
}

/// Deserialize signing key from bytes
pub fn signing_key_from_bytes(bytes: &[u8; 32]) -> SigningKey {
    SigningKey::from_bytes(bytes)
}

/// Serialize verifying key to bytes (32 bytes)
pub fn verifying_key_to_bytes(key: &VerifyingKey) -> [u8; 32] {
    key.to_bytes()
}

/// Deserialize verifying key from bytes
pub fn verifying_key_from_bytes(bytes: &[u8; 32]) -> Result<VerifyingKey, VerificationError> {
    VerifyingKey::from_bytes(bytes)
        .map_err(|e| VerificationError::CryptoError(e.to_string()))
}

/// Convert signature to bytes (64 bytes)
pub fn signature_to_bytes(sig: &Signature) -> [u8; 64] {
    sig.to_bytes()
}

/// Convert bytes to signature
pub fn signature_from_bytes(bytes: &[u8]) -> Result<Signature, VerificationError> {
    if bytes.len() != 64 {
        return Err(VerificationError::CryptoError(format!(
            "Invalid signature length: expected 64, got {}",
            bytes.len()
        )));
    }

    let mut sig_bytes = [0u8; 64];
    sig_bytes.copy_from_slice(bytes);

    Signature::from_bytes(&sig_bytes)
}

/// Generate random nonce
pub fn generate_nonce() -> Vec<u8> {
    use rand::RngCore;
    let mut nonce = vec![0u8; 32];
    OsRng.fill_bytes(&mut nonce);
    nonce
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let (signing_key1, verifying_key1) = generate_keypair();
        let (signing_key2, verifying_key2) = generate_keypair();

        // Different keypairs should be different
        assert_ne!(
            signing_key_to_bytes(&signing_key1),
            signing_key_to_bytes(&signing_key2)
        );
        assert_ne!(
            verifying_key_to_bytes(&verifying_key1),
            verifying_key_to_bytes(&verifying_key2)
        );
    }

    #[test]
    fn test_sign_and_verify() {
        let (signing_key, verifying_key) = generate_keypair();
        let message = b"test message";

        let signature = sign_message(&signing_key, message);

        // Verification should succeed
        assert!(verify_signature(&verifying_key, message, &signature).is_ok());

        // Verification with wrong message should fail
        assert!(verify_signature(&verifying_key, b"wrong message", &signature).is_err());
    }

    #[test]
    fn test_verify_with_wrong_key() {
        let (signing_key1, _) = generate_keypair();
        let (_, verifying_key2) = generate_keypair();

        let message = b"test message";
        let signature = sign_message(&signing_key1, message);

        // Verification with wrong public key should fail
        assert!(verify_signature(&verifying_key2, message, &signature).is_err());
    }

    #[test]
    fn test_sha256() {
        let data = b"hello world";
        let hash1 = compute_sha256(data);
        let hash2 = compute_sha256(data);

        // Same input should produce same hash
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 32);

        // Different input should produce different hash
        let hash3 = compute_sha256(b"different");
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_compute_pcr_from_metrics() {
        let metrics1 = RequestMetrics::new(100, 50, 1000, 0.001);
        let metrics2 = RequestMetrics::new(100, 50, 1000, 0.001);
        let metrics3 = RequestMetrics::new(200, 50, 1000, 0.001);

        let pcr1 = compute_pcr_from_metrics(&metrics1);
        let pcr2 = compute_pcr_from_metrics(&metrics2);
        let pcr3 = compute_pcr_from_metrics(&metrics3);

        // Same metrics should produce same PCR
        assert_eq!(pcr1, pcr2);
        assert_eq!(pcr1.len(), 32);

        // Different metrics should produce different PCR
        assert_ne!(pcr1, pcr3);
    }

    #[test]
    fn test_key_serialization() {
        let (signing_key, verifying_key) = generate_keypair();

        // Serialize
        let signing_bytes = signing_key_to_bytes(&signing_key);
        let verifying_bytes = verifying_key_to_bytes(&verifying_key);

        // Deserialize
        let signing_key2 = signing_key_from_bytes(&signing_bytes);
        let verifying_key2 = verifying_key_from_bytes(&verifying_bytes).unwrap();

        // Should be able to sign with deserialized key
        let message = b"test";
        let sig = sign_message(&signing_key2, message);
        assert!(verify_signature(&verifying_key2, message, &sig).is_ok());
    }

    #[test]
    fn test_signature_serialization() {
        let (signing_key, verifying_key) = generate_keypair();
        let message = b"test message";

        let signature = sign_message(&signing_key, message);
        let sig_bytes = signature_to_bytes(&signature);

        assert_eq!(sig_bytes.len(), 64);

        let signature2 = signature_from_bytes(&sig_bytes).unwrap();
        assert!(verify_signature(&verifying_key, message, &signature2).is_ok());
    }

    #[test]
    fn test_signature_from_invalid_bytes() {
        // Too short
        let result = signature_from_bytes(&[0u8; 32]);
        assert!(result.is_err());

        // Too long
        let result = signature_from_bytes(&[0u8; 100]);
        assert!(result.is_err());
    }

    #[test]
    fn test_generate_nonce() {
        let nonce1 = generate_nonce();
        let nonce2 = generate_nonce();

        assert_eq!(nonce1.len(), 32);
        assert_eq!(nonce2.len(), 32);

        // Should generate different nonces
        assert_ne!(nonce1, nonce2);
    }

    #[test]
    fn test_end_to_end_crypto_flow() {
        // Agent generates keypair
        let (agent_signing_key, agent_verifying_key) = generate_keypair();

        // Agent creates metrics
        let metrics = RequestMetrics::new(100, 50, 1000, 0.001);

        // Agent computes PCR
        let pcr = compute_pcr_from_metrics(&metrics);

        // Agent generates nonce
        let nonce = generate_nonce();

        // Agent signs PCR + nonce
        let mut message = pcr.clone();
        message.extend_from_slice(&nonce);
        let signature = sign_message(&agent_signing_key, &message);

        // Verifier verifies the signature
        assert!(verify_signature(&agent_verifying_key, &message, &signature).is_ok());
    }
}
