use thiserror::Error;

/// Errors related to TPM quote verification
#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("Unknown agent: {0}")]
    UnknownAgent(String),

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Invalid PCR value")]
    InvalidPcr,

    #[error("Validation failed: {0}")]
    ValidationFailed(String),

    #[error("Missing PCR register: {0}")]
    MissingPcr(u32),

    #[error("Cryptography error: {0}")]
    CryptoError(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Errors related to metric validation
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Invalid token count: {0} (expected: 1-1000000)")]
    InvalidTokenCount(u32),

    #[error("Invalid latency: {0}ms (expected: 1-600000)")]
    InvalidLatency(u64),

    #[error("Invalid cost: {0} (expected: >= 0.0)")]
    InvalidCost(f64),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(u64),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Out of range: {0}")]
    OutOfRange(String),
}

/// Errors related to network operations
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Request timeout")]
    Timeout,

    #[error("gRPC error: {0}")]
    GrpcError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid endpoint: {0}")]
    InvalidEndpoint(String),
}

/// Errors related to consensus protocol
#[derive(Error, Debug)]
pub enum ConsensusError {
    #[error("Quorum not reached (got {0} votes, need {1})")]
    QuorumNotReached(usize, usize),

    #[error("No votes received")]
    NoVotes,

    #[error("Invalid vote: {0}")]
    InvalidVote(String),

    #[error("Epoch mismatch: expected {expected}, got {actual}")]
    EpochMismatch { expected: u64, actual: u64 },

    #[error("Committee selection failed: {0}")]
    CommitteeSelectionFailed(String),

    #[error("Vote collection timeout")]
    VoteCollectionTimeout,
}

/// Errors related to configuration
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Invalid configuration: {0}")]
    Invalid(String),

    #[error("Missing configuration: {0}")]
    Missing(String),

    #[error("Configuration parse error: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Conversion from ed25519_dalek::SignatureError
impl From<ed25519_dalek::SignatureError> for VerificationError {
    fn from(err: ed25519_dalek::SignatureError) -> Self {
        VerificationError::CryptoError(err.to_string())
    }
}

/// Conversion from tonic::Status to NetworkError
#[cfg(feature = "tonic")]
impl From<tonic::Status> for NetworkError {
    fn from(status: tonic::Status) -> Self {
        NetworkError::GrpcError(status.to_string())
    }
}

/// Conversion from serde_json::Error to NetworkError
impl From<serde_json::Error> for NetworkError {
    fn from(err: serde_json::Error) -> Self {
        NetworkError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verification_error_display() {
        let err = VerificationError::UnknownAgent("agent-1".to_string());
        assert_eq!(err.to_string(), "Unknown agent: agent-1");
    }

    #[test]
    fn test_validation_error_display() {
        let err = ValidationError::InvalidTokenCount(0);
        assert!(err.to_string().contains("Invalid token count"));
    }

    #[test]
    fn test_consensus_error_quorum() {
        let err = ConsensusError::QuorumNotReached(2, 3);
        assert!(err.to_string().contains("2 votes"));
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn test_epoch_mismatch() {
        let err = ConsensusError::EpochMismatch {
            expected: 10,
            actual: 11,
        };
        assert!(err.to_string().contains("expected 10"));
        assert!(err.to_string().contains("got 11"));
    }
}
