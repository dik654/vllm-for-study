pub mod crypto;
pub mod error;
pub mod types;

// Re-export commonly used items
pub use crypto::{
    compute_pcr_from_metrics, compute_sha256, generate_keypair, generate_nonce, sign_message,
    signature_from_bytes, signature_to_bytes, signing_key_from_bytes, signing_key_to_bytes,
    verify_signature, verifying_key_from_bytes, verifying_key_to_bytes,
};

pub use error::{
    ConfigError, ConsensusError, NetworkError, ValidationError, VerificationError,
};

pub use types::{
    ConsensusOutcome, MetricSubmission, RequestMetrics, TpmQuote, VerificationRequest,
    VerificationResult, VerifierVote,
};

/// Get current Unix timestamp in seconds
pub fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

/// Get current Unix timestamp in milliseconds
pub fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_timestamp() {
        let ts1 = current_timestamp();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = current_timestamp();

        // Should be monotonically increasing
        assert!(ts2 >= ts1);
    }

    #[test]
    fn test_current_timestamp_ms() {
        let ts1 = current_timestamp_ms();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ts2 = current_timestamp_ms();

        // Should be monotonically increasing and at least 10ms apart
        assert!(ts2 > ts1);
        assert!(ts2 - ts1 >= 10);
    }
}
