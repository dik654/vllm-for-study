# BFT Verifier PoC - Implementation TODO

## Project Structure
```
bft_verifier_poc/
├── Cargo.toml              # Workspace configuration
├── SPECIFICATION.md        # ✅ Done
├── TODO.md                 # This file
├── README.md               # Usage guide
├── docker-compose.yml      # Local deployment
├── proto/                  # Protocol Buffer definitions
│   └── bft_verifier.proto
├── common/                 # Shared types and utilities
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── types.rs        # Shared data structures
│       ├── crypto.rs       # Cryptography utilities
│       └── error.rs        # Error types
├── coordinator/            # Consensus coordinator
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── vrf.rs          # VRF verifier selection
│       ├── consensus.rs    # Consensus manager
│       ├── epoch.rs        # Epoch management
│       └── server.rs       # gRPC server
├── verifier/               # Verifier node
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── tpm.rs          # Simulated TPM verification
│       ├── validator.rs    # Metric validation
│       └── server.rs       # gRPC server
├── agent/                  # Agent (metrics submitter)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── metrics.rs      # Metrics generation
│       ├── tpm.rs          # Simulated TPM quote
│       └── client.rs       # gRPC client
├── tests/                  # Integration tests
│   ├── test_consensus.rs
│   ├── test_vrf.rs
│   ├── test_byzantine.rs
│   └── test_performance.rs
└── examples/               # Usage examples
    ├── simple_verification.rs
    └── benchmark.rs
```

## Implementation Phases

### ✅ Phase 0: Project Setup (COMPLETED)
- [x] Create directory structure
- [x] Write SPECIFICATION.md
- [x] Write TODO.md

---

### 📋 Phase 1: Foundation & Protocol Definition

**Goal**: Set up Rust workspace and define Protocol Buffers schema

#### Task 1.1: Cargo Workspace Setup
**File**: `Cargo.toml`

```toml
[workspace]
members = [
    "common",
    "coordinator",
    "verifier",
    "agent",
]
resolver = "2"

[workspace.dependencies]
tokio = { version = "1.35", features = ["full"] }
tonic = "0.11"
prost = "0.12"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
sha2 = "0.10"
ed25519-dalek = "2.0"
rand = "0.8"
uuid = { version = "1.6", features = ["v4", "serde"] }
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }
anyhow = "1.0"
thiserror = "1.0"
```

**Checklist**:
- [ ] Create `Cargo.toml` with workspace definition
- [ ] Add all required dependencies
- [ ] Verify `cargo check` passes
- [ ] Test workspace build with `cargo build --workspace`

**Estimated Time**: 30 minutes

---

#### Task 1.2: Protocol Buffer Definition
**File**: `proto/bft_verifier.proto`

**Checklist**:
- [ ] Create `proto/` directory
- [ ] Copy protobuf definition from SPECIFICATION.md
- [ ] Add build.rs for proto compilation
- [ ] Test proto generation with `cargo build`
- [ ] Verify generated Rust types

**Estimated Time**: 45 minutes

---

#### Task 1.3: Common Types Module
**File**: `common/src/types.rs`

**Checklist**:
- [ ] Create `common/` crate
- [ ] Define `RequestMetrics` struct
- [ ] Define `TpmQuote` struct
- [ ] Define `MetricSubmission` struct
- [ ] Define `VerificationRequest` struct
- [ ] Define `VerifierVote` struct
- [ ] Define `ConsensusOutcome` struct
- [ ] Define `VerificationResult` enum
- [ ] Add Serde derives for all types
- [ ] Add unit tests for serialization
- [ ] Document all public types

**Code Template**:
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub e2e_latency_ms: u64,
    pub estimated_cost: f64,
}

// ... (see SPECIFICATION.md for complete definitions)
```

**Estimated Time**: 1.5 hours

---

#### Task 1.4: Error Types Module
**File**: `common/src/error.rs`

**Checklist**:
- [ ] Define `VerificationError` enum with thiserror
- [ ] Define `ValidationError` enum
- [ ] Define `NetworkError` enum
- [ ] Implement Display and Error traits
- [ ] Add conversion from common error types
- [ ] Add unit tests

**Code Template**:
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("Unknown agent: {0}")]
    UnknownAgent(String),

    #[error("Invalid signature")]
    InvalidSignature,

    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

// ... more error types
```

**Estimated Time**: 1 hour

---

#### Task 1.5: Cryptography Utilities Module
**File**: `common/src/crypto.rs`

**Checklist**:
- [ ] Implement SHA-256 hashing utilities
- [ ] Implement Ed25519 key generation
- [ ] Implement Ed25519 signing
- [ ] Implement Ed25519 verification
- [ ] Add PCR computation helper
- [ ] Add unit tests for all functions
- [ ] Add integration test with real signatures

**Code Template**:
```rust
use ed25519_dalek::{Keypair, PublicKey, Signature, Signer, Verifier};
use sha2::{Digest, Sha256};
use rand::rngs::OsRng;

pub fn generate_keypair() -> Keypair {
    let mut csprng = OsRng;
    Keypair::generate(&mut csprng)
}

pub fn sign_message(keypair: &Keypair, message: &[u8]) -> Signature {
    keypair.sign(message)
}

pub fn verify_signature(
    public_key: &PublicKey,
    message: &[u8],
    signature: &Signature,
) -> Result<(), ed25519_dalek::SignatureError> {
    public_key.verify(message, signature)
}

pub fn compute_sha256(data: &[u8]) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().to_vec()
}

// ... more crypto utilities
```

**Estimated Time**: 2 hours

---

**Phase 1 Total Estimated Time**: 5.5 hours

---

### 📋 Phase 2: Verifier Node Implementation

**Goal**: Implement the verifier service that validates metrics and casts votes

#### Task 2.1: Simulated TPM Verification
**File**: `verifier/src/tpm.rs`

**Checklist**:
- [ ] Create `SimulatedTpmVerifier` struct
- [ ] Implement `verify_quote()` method
- [ ] Implement `compute_pcr_from_metrics()` helper
- [ ] Load agent public keys from config
- [ ] Add signature verification using Ed25519
- [ ] Add unit tests with valid/invalid quotes
- [ ] Add integration test with agent-generated quotes

**Code Template**:
```rust
use common::types::{RequestMetrics, TpmQuote};
use common::error::VerificationError;
use ed25519_dalek::PublicKey;
use std::collections::HashMap;

pub struct SimulatedTpmVerifier {
    agent_public_keys: HashMap<String, PublicKey>,
}

impl SimulatedTpmVerifier {
    pub fn new(agent_keys: HashMap<String, PublicKey>) -> Self {
        Self { agent_public_keys: agent_keys }
    }

    pub fn verify_quote(
        &self,
        agent_id: &str,
        metrics: &RequestMetrics,
        quote: &TpmQuote,
    ) -> Result<bool, VerificationError> {
        // Implementation from SPECIFICATION.md
        todo!()
    }

    fn compute_pcr_from_metrics(&self, metrics: &RequestMetrics) -> Vec<u8> {
        // Implementation from SPECIFICATION.md
        todo!()
    }
}
```

**Estimated Time**: 2 hours

---

#### Task 2.2: Metric Validation
**File**: `verifier/src/validator.rs`

**Checklist**:
- [ ] Implement `validate_metrics()` function
- [ ] Check token count ranges
- [ ] Check latency ranges
- [ ] Check cost validity
- [ ] Add business rule validation
- [ ] Add unit tests for each validation rule
- [ ] Add edge case tests

**Code Template**:
```rust
use common::types::RequestMetrics;
use common::error::ValidationError;

pub fn validate_metrics(metrics: &RequestMetrics) -> Result<(), ValidationError> {
    // Token validation
    if metrics.prompt_tokens == 0 || metrics.prompt_tokens > 1_000_000 {
        return Err(ValidationError::InvalidTokenCount(metrics.prompt_tokens));
    }

    // Latency validation
    if metrics.e2e_latency_ms == 0 || metrics.e2e_latency_ms > 600_000 {
        return Err(ValidationError::InvalidLatency(metrics.e2e_latency_ms));
    }

    // Cost validation
    if metrics.estimated_cost < 0.0 {
        return Err(ValidationError::InvalidCost(metrics.estimated_cost));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_metrics() {
        let metrics = RequestMetrics {
            prompt_tokens: 100,
            completion_tokens: 50,
            e2e_latency_ms: 1000,
            estimated_cost: 0.001,
        };
        assert!(validate_metrics(&metrics).is_ok());
    }

    // ... more tests
}
```

**Estimated Time**: 1.5 hours

---

#### Task 2.3: Verifier gRPC Server
**File**: `verifier/src/server.rs`

**Checklist**:
- [ ] Create `VerifierService` struct
- [ ] Implement `Verify` gRPC method
- [ ] Integrate TPM verification
- [ ] Integrate metric validation
- [ ] Generate `VerifierVote` response
- [ ] Sign vote with verifier's private key
- [ ] Add logging for all operations
- [ ] Add error handling
- [ ] Add unit tests with mock TPM verifier

**Code Template**:
```rust
use tonic::{Request, Response, Status};
use bft_verifier_proto::verifier_server::Verifier;
use bft_verifier_proto::{VerificationRequest, VerifierVote};

pub struct VerifierService {
    verifier_id: String,
    tpm_verifier: SimulatedTpmVerifier,
    signing_key: Keypair,
}

#[tonic::async_trait]
impl Verifier for VerifierService {
    async fn verify(
        &self,
        request: Request<VerificationRequest>,
    ) -> Result<Response<VerifierVote>, Status> {
        let req = request.into_inner();

        // 1. Validate metrics
        // 2. Verify TPM quote
        // 3. Generate vote
        // 4. Sign vote
        // 5. Return response

        todo!()
    }

    async fn heartbeat(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<VerifierStatus>, Status> {
        // Return health status
        todo!()
    }
}
```

**Estimated Time**: 2.5 hours

---

#### Task 2.4: Verifier Main Entry Point
**File**: `verifier/src/main.rs`

**Checklist**:
- [ ] Load configuration from file/env
- [ ] Initialize logging with tracing
- [ ] Parse verifier config
- [ ] Load agent public keys
- [ ] Create `SimulatedTpmVerifier`
- [ ] Start gRPC server
- [ ] Add graceful shutdown handling
- [ ] Add CLI argument parsing
- [ ] Test end-to-end startup

**Code Template**:
```rust
use tonic::transport::Server;
use tracing_subscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // Load config
    let config = load_config()?;

    // Create service
    let service = VerifierService::new(config)?;

    // Start server
    Server::builder()
        .add_service(VerifierServer::new(service))
        .serve(config.listen_addr.parse()?)
        .await?;

    Ok(())
}
```

**Estimated Time**: 1.5 hours

---

**Phase 2 Total Estimated Time**: 7.5 hours

---

### 📋 Phase 3: Coordinator Implementation

**Goal**: Implement consensus coordinator with VRF selection and vote collection

#### Task 3.1: VRF Verifier Selection
**File**: `coordinator/src/vrf.rs`

**Checklist**:
- [ ] Create `VrfSelector` struct
- [ ] Implement `select_committee()` method
- [ ] Use SHA-256 for VRF hash
- [ ] Implement Fisher-Yates shuffle with deterministic seed
- [ ] Add configuration for committee size
- [ ] Add unit tests for determinism
- [ ] Add unit tests for fairness (chi-square test)
- [ ] Add benchmark for selection speed

**Code Template**:
```rust
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

pub struct VrfSelector {
    vrf_seed: [u8; 32],
    committee_size: usize,
}

impl VrfSelector {
    pub fn new(vrf_seed: [u8; 32], committee_size: usize) -> Self {
        Self { vrf_seed, committee_size }
    }

    pub fn select_committee(&self, epoch: u64, verifier_pool: &[String]) -> Vec<String> {
        // Implementation from SPECIFICATION.md
        let mut input = self.vrf_seed.to_vec();
        input.extend_from_slice(&epoch.to_le_bytes());

        let hash = Sha256::digest(&input);
        let seed = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        let mut rng = StdRng::seed_from_u64(seed);
        let mut pool = verifier_pool.to_vec();
        pool.shuffle(&mut rng);

        pool.into_iter().take(self.committee_size).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_selection() {
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec!["v1".into(), "v2".into(), "v3".into(), "v4".into()];

        let committee1 = selector.select_committee(1, &verifiers);
        let committee2 = selector.select_committee(1, &verifiers);

        assert_eq!(committee1, committee2);
    }

    #[test]
    fn test_different_epochs_different_committees() {
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec!["v1".into(), "v2".into(), "v3".into(), "v4".into()];

        let committee1 = selector.select_committee(1, &verifiers);
        let committee2 = selector.select_committee(2, &verifiers);

        assert_ne!(committee1, committee2);
    }
}
```

**Estimated Time**: 2 hours

---

#### Task 3.2: Epoch Management
**File**: `coordinator/src/epoch.rs`

**Checklist**:
- [ ] Create `EpochManager` struct
- [ ] Implement `current_epoch()` method
- [ ] Implement `time_until_next_epoch()` method
- [ ] Add epoch change notification system
- [ ] Add unit tests with time mocking
- [ ] Add integration test with real time

**Code Template**:
```rust
use std::time::{Duration, Instant};

pub struct EpochManager {
    epoch_duration: Duration,
    start_time: Instant,
}

impl EpochManager {
    pub fn new(epoch_duration: Duration) -> Self {
        Self {
            epoch_duration,
            start_time: Instant::now(),
        }
    }

    pub fn current_epoch(&self) -> u64 {
        self.start_time.elapsed().as_secs() / self.epoch_duration.as_secs()
    }

    pub fn time_until_next_epoch(&self) -> Duration {
        let elapsed = self.start_time.elapsed();
        let current_epoch_secs = self.current_epoch() * self.epoch_duration.as_secs();
        let next_epoch_secs = current_epoch_secs + self.epoch_duration.as_secs();
        Duration::from_secs(next_epoch_secs.saturating_sub(elapsed.as_secs()))
    }
}
```

**Estimated Time**: 1 hour

---

#### Task 3.3: Consensus Manager
**File**: `coordinator/src/consensus.rs`

**Checklist**:
- [ ] Create `ConsensusManager` struct
- [ ] Implement `collect_votes()` method with parallel requests
- [ ] Implement `reach_consensus()` method
- [ ] Add timeout handling for vote collection
- [ ] Add vote validation (signature check)
- [ ] Implement quorum logic (2f+1)
- [ ] Add metrics tracking (Prometheus)
- [ ] Add unit tests for consensus logic
- [ ] Add integration tests with mock verifiers

**Code Template**:
```rust
use tokio::time::{timeout, Duration};
use futures::future::join_all;
use std::collections::HashMap;
use common::types::{VerifierVote, VerificationResult};

pub struct ConsensusManager {
    quorum: usize,
    vote_timeout: Duration,
    verifier_clients: HashMap<String, VerifierClient>,
}

impl ConsensusManager {
    pub async fn collect_votes(
        &self,
        committee: &[String],
        request: VerificationRequest,
    ) -> HashMap<String, VerifierVote> {
        let futures: Vec<_> = committee
            .iter()
            .filter_map(|v_id| {
                self.verifier_clients.get(v_id).map(|client| {
                    let req = request.clone();
                    async move {
                        client.verify(req).await
                    }
                })
            })
            .collect();

        let results = timeout(self.vote_timeout, join_all(futures)).await;

        // Process results and build vote map
        // ...

        HashMap::new() // placeholder
    }

    pub fn reach_consensus(
        &self,
        votes: &HashMap<String, VerifierVote>,
    ) -> (Option<VerificationResult>, String) {
        let mut pass_count = 0;
        let mut fail_count = 0;

        for vote in votes.values() {
            match vote.result {
                VerificationResult::Pass => pass_count += 1,
                VerificationResult::Fail => fail_count += 1,
            }
        }

        if pass_count >= self.quorum {
            (
                Some(VerificationResult::Pass),
                format!("{}/{} verifiers agree: PASS", pass_count, votes.len())
            )
        } else if fail_count >= self.quorum {
            (
                Some(VerificationResult::Fail),
                format!("{}/{} verifiers agree: FAIL", fail_count, votes.len())
            )
        } else {
            (
                None,
                format!(
                    "No quorum reached (PASS: {}, FAIL: {}, Quorum: {})",
                    pass_count, fail_count, self.quorum
                )
            )
        }
    }
}
```

**Estimated Time**: 3 hours

---

#### Task 3.4: Coordinator gRPC Server
**File**: `coordinator/src/server.rs`

**Checklist**:
- [ ] Create `CoordinatorService` struct
- [ ] Implement `SubmitMetrics` method
- [ ] Implement `QueryConsensus` method
- [ ] Integrate VRF selection
- [ ] Integrate consensus manager
- [ ] Store verification results (in-memory for PoC)
- [ ] Add request ID generation (UUID)
- [ ] Add logging for all operations
- [ ] Add unit tests

**Code Template**:
```rust
use tonic::{Request, Response, Status};
use bft_verifier_proto::coordinator_server::Coordinator;
use bft_verifier_proto::{MetricSubmission, SubmissionAck, ConsensusOutcome};

pub struct CoordinatorService {
    vrf_selector: VrfSelector,
    consensus_manager: ConsensusManager,
    epoch_manager: EpochManager,
    verifier_pool: Vec<String>,
    results_cache: Arc<RwLock<HashMap<String, ConsensusOutcome>>>,
}

#[tonic::async_trait]
impl Coordinator for CoordinatorService {
    async fn submit_metrics(
        &self,
        request: Request<MetricSubmission>,
    ) -> Result<Response<SubmissionAck>, Status> {
        let submission = request.into_inner();
        let verification_id = uuid::Uuid::new_v4().to_string();

        // 1. Get current epoch
        let epoch = self.epoch_manager.current_epoch();

        // 2. Select committee
        let committee = self.vrf_selector.select_committee(epoch, &self.verifier_pool);

        // 3. Collect votes
        let votes = self.consensus_manager.collect_votes(&committee, submission).await;

        // 4. Reach consensus
        let (result, reason) = self.consensus_manager.reach_consensus(&votes);

        // 5. Store outcome
        let outcome = ConsensusOutcome {
            verification_id: verification_id.clone(),
            epoch,
            committee,
            votes,
            result,
            quorum_reached: result.is_some(),
            timestamp: current_timestamp(),
        };

        self.results_cache.write().await.insert(verification_id.clone(), outcome);

        Ok(Response::new(SubmissionAck {
            verification_id,
            accepted: result.is_some(),
        }))
    }

    async fn query_consensus(
        &self,
        request: Request<VerificationQuery>,
    ) -> Result<Response<ConsensusOutcome>, Status> {
        let query = request.into_inner();
        let cache = self.results_cache.read().await;

        cache.get(&query.verification_id)
            .cloned()
            .map(Response::new)
            .ok_or_else(|| Status::not_found("Verification ID not found"))
    }
}
```

**Estimated Time**: 3 hours

---

#### Task 3.5: Coordinator Main Entry Point
**File**: `coordinator/src/main.rs`

**Checklist**:
- [ ] Load configuration
- [ ] Initialize logging
- [ ] Create gRPC clients for all verifiers
- [ ] Initialize VRF selector
- [ ] Initialize epoch manager
- [ ] Initialize consensus manager
- [ ] Start gRPC server
- [ ] Add graceful shutdown
- [ ] Test end-to-end startup

**Estimated Time**: 1.5 hours

---

**Phase 3 Total Estimated Time**: 10.5 hours

---

### 📋 Phase 4: Agent Implementation

**Goal**: Implement agent that generates metrics and submits to coordinator

#### Task 4.1: Simulated Metrics Generation
**File**: `agent/src/metrics.rs`

**Checklist**:
- [ ] Create `MetricsGenerator` struct
- [ ] Implement `generate_random_metrics()` for testing
- [ ] Add realistic distributions (normal for latency, etc.)
- [ ] Add configuration for metric ranges
- [ ] Add unit tests

**Code Template**:
```rust
use common::types::RequestMetrics;
use rand::Rng;

pub struct MetricsGenerator {
    rng: rand::rngs::ThreadRng,
}

impl MetricsGenerator {
    pub fn new() -> Self {
        Self { rng: rand::thread_rng() }
    }

    pub fn generate_random_metrics(&mut self) -> RequestMetrics {
        RequestMetrics {
            prompt_tokens: self.rng.gen_range(10..1000),
            completion_tokens: self.rng.gen_range(10..500),
            e2e_latency_ms: self.rng.gen_range(100..5000),
            estimated_cost: self.rng.gen_range(0.0001..0.01),
        }
    }
}
```

**Estimated Time**: 1 hour

---

#### Task 4.2: Simulated TPM Quote Generation
**File**: `agent/src/tpm.rs`

**Checklist**:
- [ ] Create `SimulatedTpmAgent` struct
- [ ] Implement `generate_quote()` method
- [ ] Compute PCR from metrics (same as verifier)
- [ ] Sign with agent's private key
- [ ] Add nonce generation
- [ ] Add unit tests

**Code Template**:
```rust
use common::types::{RequestMetrics, TpmQuote};
use ed25519_dalek::Keypair;
use sha2::{Digest, Sha256};
use rand::RngCore;

pub struct SimulatedTpmAgent {
    signing_key: Keypair,
}

impl SimulatedTpmAgent {
    pub fn new(signing_key: Keypair) -> Self {
        Self { signing_key }
    }

    pub fn generate_quote(&self, metrics: &RequestMetrics) -> TpmQuote {
        // 1. Compute PCR from metrics
        let pcr_value = self.compute_pcr_from_metrics(metrics);

        // 2. Generate nonce
        let mut nonce = vec![0u8; 32];
        rand::thread_rng().fill_bytes(&mut nonce);

        // 3. Sign PCR + nonce
        let mut message = pcr_value.clone();
        message.extend_from_slice(&nonce);
        let signature = self.signing_key.sign(&message);

        TpmQuote {
            pcr_values: std::collections::HashMap::from([(10, pcr_value)]),
            quote_signature: signature.to_bytes().to_vec(),
            nonce,
        }
    }

    fn compute_pcr_from_metrics(&self, metrics: &RequestMetrics) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(metrics.prompt_tokens.to_le_bytes());
        hasher.update(metrics.completion_tokens.to_le_bytes());
        hasher.update(metrics.e2e_latency_ms.to_le_bytes());
        hasher.finalize().to_vec()
    }
}
```

**Estimated Time**: 1.5 hours

---

#### Task 4.3: Agent gRPC Client
**File**: `agent/src/client.rs`

**Checklist**:
- [ ] Create `CoordinatorClient` wrapper
- [ ] Implement `submit_metrics()` method
- [ ] Implement `query_result()` method
- [ ] Add retry logic with exponential backoff
- [ ] Add error handling
- [ ] Add unit tests with mock server

**Code Template**:
```rust
use tonic::transport::Channel;
use bft_verifier_proto::coordinator_client::CoordinatorClient as GrpcClient;
use common::types::{MetricSubmission, ConsensusOutcome};

pub struct CoordinatorClient {
    client: GrpcClient<Channel>,
}

impl CoordinatorClient {
    pub async fn connect(endpoint: String) -> anyhow::Result<Self> {
        let client = GrpcClient::connect(endpoint).await?;
        Ok(Self { client })
    }

    pub async fn submit_metrics(
        &mut self,
        submission: MetricSubmission,
    ) -> anyhow::Result<String> {
        let response = self.client.submit_metrics(submission).await?;
        Ok(response.into_inner().verification_id)
    }

    pub async fn query_result(
        &mut self,
        verification_id: String,
    ) -> anyhow::Result<ConsensusOutcome> {
        let response = self.client.query_consensus(VerificationQuery {
            verification_id,
        }).await?;
        Ok(response.into_inner())
    }
}
```

**Estimated Time**: 1.5 hours

---

#### Task 4.4: Agent Main Entry Point
**File**: `agent/src/main.rs`

**Checklist**:
- [ ] Load configuration
- [ ] Initialize logging
- [ ] Load agent's signing key
- [ ] Connect to coordinator
- [ ] Implement metrics submission loop
- [ ] Add CLI mode (single submission vs continuous)
- [ ] Add graceful shutdown
- [ ] Test end-to-end

**Code Template**:
```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = load_config()?;
    let signing_key = load_signing_key(&config.signing_key_path)?;
    let tpm_agent = SimulatedTpmAgent::new(signing_key);
    let mut metrics_gen = MetricsGenerator::new();
    let mut client = CoordinatorClient::connect(config.coordinator_endpoint).await?;

    loop {
        // Generate metrics
        let metrics = metrics_gen.generate_random_metrics();
        let quote = tpm_agent.generate_quote(&metrics);

        // Submit
        let submission = MetricSubmission {
            agent_id: config.agent_id.clone(),
            request_id: uuid::Uuid::new_v4().to_string(),
            timestamp: current_timestamp(),
            metrics,
            quote,
        };

        match client.submit_metrics(submission).await {
            Ok(verification_id) => {
                tracing::info!("Submitted metrics, verification_id: {}", verification_id);
            }
            Err(e) => {
                tracing::error!("Failed to submit metrics: {}", e);
            }
        }

        sleep(Duration::from_secs(5)).await;
    }
}
```

**Estimated Time**: 1.5 hours

---

**Phase 4 Total Estimated Time**: 5.5 hours

---

### 📋 Phase 5: Integration Testing

**Goal**: Test the complete system with multiple components

#### Task 5.1: Consensus Test (TC-001, TC-002)
**File**: `tests/test_consensus.rs`

**Checklist**:
- [ ] Set up test environment (3 verifiers, 1 coordinator, 1 agent)
- [ ] Test honest consensus (all verifiers vote PASS)
- [ ] Test Byzantine tolerance (1 malicious verifier)
- [ ] Verify quorum logic
- [ ] Add assertions on consensus outcome
- [ ] Clean up test resources

**Estimated Time**: 2 hours

---

#### Task 5.2: VRF Fairness Test (TC-007)
**File**: `tests/test_vrf.rs`

**Checklist**:
- [ ] Run 1000 epochs
- [ ] Track selection count per verifier
- [ ] Compute chi-square statistic
- [ ] Assert fair distribution (p < 0.05)
- [ ] Test determinism across multiple runs

**Estimated Time**: 1.5 hours

---

#### Task 5.3: Byzantine Fault Test (TC-010, TC-011)
**File**: `tests/test_byzantine.rs`

**Checklist**:
- [ ] Test verifier crash during vote collection
- [ ] Test verifier returning invalid vote
- [ ] Test network timeout scenario
- [ ] Verify consensus still reached with f faults
- [ ] Test edge case: exactly f+1 votes

**Estimated Time**: 2 hours

---

#### Task 5.4: Performance Benchmark (TC-008, TC-009)
**File**: `tests/test_performance.rs`

**Checklist**:
- [ ] Benchmark latency (100 sequential requests)
- [ ] Compute p50, p95, p99
- [ ] Benchmark throughput (1000 parallel requests)
- [ ] Measure CPU and memory usage
- [ ] Generate performance report
- [ ] Compare against targets in SPECIFICATION.md

**Estimated Time**: 2.5 hours

---

**Phase 5 Total Estimated Time**: 8 hours

---

### 📋 Phase 6: Docker & Deployment

**Goal**: Containerize and enable easy local deployment

#### Task 6.1: Dockerfile
**File**: `Dockerfile`

**Checklist**:
- [ ] Multi-stage build (builder + runtime)
- [ ] Build all workspace members
- [ ] Copy binaries to runtime image
- [ ] Minimize image size
- [ ] Test docker build

**Code Template**:
```dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --workspace

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/coordinator /usr/local/bin/
COPY --from=builder /app/target/release/verifier /usr/local/bin/
COPY --from=builder /app/target/release/agent /usr/local/bin/
```

**Estimated Time**: 1 hour

---

#### Task 6.2: Docker Compose
**File**: `docker-compose.yml`

**Checklist**:
- [ ] Define coordinator service
- [ ] Define 4 verifier services
- [ ] Define agent service
- [ ] Set up networking
- [ ] Mount configuration files
- [ ] Add health checks
- [ ] Test `docker-compose up`

**Code Template**:
```yaml
version: '3.8'

services:
  coordinator:
    build: .
    command: coordinator
    ports:
      - "50051:50051"
    environment:
      - RUST_LOG=info
      - VRF_SEED=0000000000000000000000000000000000000000000000000000000000000000
      - EPOCH_DURATION_SECS=30
      - COMMITTEE_SIZE=3
      - QUORUM=2

  verifier-1:
    build: .
    command: verifier
    environment:
      - VERIFIER_ID=verifier-1
      - LISTEN_ADDR=0.0.0.0:50052
      - RUST_LOG=info

  verifier-2:
    build: .
    command: verifier
    environment:
      - VERIFIER_ID=verifier-2
      - LISTEN_ADDR=0.0.0.0:50053

  verifier-3:
    build: .
    command: verifier
    environment:
      - VERIFIER_ID=verifier-3
      - LISTEN_ADDR=0.0.0.0:50054

  verifier-4:
    build: .
    command: verifier
    environment:
      - VERIFIER_ID=verifier-4
      - LISTEN_ADDR=0.0.0.0:50055

  agent:
    build: .
    command: agent
    environment:
      - AGENT_ID=agent-1
      - COORDINATOR_ENDPOINT=http://coordinator:50051
    depends_on:
      - coordinator
```

**Estimated Time**: 1.5 hours

---

#### Task 6.3: Configuration Management
**File**: `config/` directory

**Checklist**:
- [ ] Create config templates
- [ ] Generate test keys for all nodes
- [ ] Create key distribution script
- [ ] Document configuration options
- [ ] Test with different configs

**Estimated Time**: 1 hour

---

**Phase 6 Total Estimated Time**: 3.5 hours

---

### 📋 Phase 7: Documentation & Examples

**Goal**: Create comprehensive documentation and usage examples

#### Task 7.1: README.md
**File**: `README.md`

**Checklist**:
- [ ] Project overview
- [ ] Architecture diagram
- [ ] Quick start guide
- [ ] Installation instructions
- [ ] Configuration guide
- [ ] API documentation
- [ ] Testing guide
- [ ] Troubleshooting section

**Estimated Time**: 2 hours

---

#### Task 7.2: Usage Examples
**File**: `examples/simple_verification.rs`

**Checklist**:
- [ ] Simple single verification example
- [ ] Batch verification example
- [ ] Byzantine fault demonstration
- [ ] Performance benchmark example
- [ ] Add comments and documentation

**Estimated Time**: 1.5 hours

---

#### Task 7.3: API Documentation
**Files**: All source files

**Checklist**:
- [ ] Add rustdoc comments to all public APIs
- [ ] Add examples in doc comments
- [ ] Generate `cargo doc`
- [ ] Review and fix warnings
- [ ] Host docs locally

**Estimated Time**: 2 hours

---

**Phase 7 Total Estimated Time**: 5.5 hours

---

### 📋 Phase 8: Final Testing & Validation

**Goal**: Comprehensive end-to-end testing and validation

#### Task 8.1: Manual Testing
**Checklist**:
- [ ] Start full system with docker-compose
- [ ] Submit 100 metrics from agent
- [ ] Verify all reach consensus
- [ ] Kill one verifier, verify system continues
- [ ] Check logs for errors
- [ ] Verify metrics (Prometheus)

**Estimated Time**: 1.5 hours

---

#### Task 8.2: Success Criteria Validation
**Checklist**:
- [ ] ✅ 3+ verifiers reaching consensus (TC-001)
- [ ] ✅ Byzantine fault tolerance (TC-002)
- [ ] ✅ VRF-based selection working (TC-005, TC-006)
- [ ] ✅ <100ms consensus latency (TC-008)
- [ ] ✅ Detect tampered metrics (TC-003)
- [ ] Document all test results

**Estimated Time**: 1 hour

---

#### Task 8.3: Performance Report
**Checklist**:
- [ ] Run all benchmarks
- [ ] Generate graphs (latency distribution, throughput)
- [ ] Compare against targets
- [ ] Document bottlenecks
- [ ] Create performance report document

**Estimated Time**: 1.5 hours

---

**Phase 8 Total Estimated Time**: 4 hours

---

## Summary

### Total Estimated Time: 50 hours

### Phase Breakdown
- **Phase 1**: Foundation & Protocol Definition - 5.5 hours
- **Phase 2**: Verifier Node - 7.5 hours
- **Phase 3**: Coordinator - 10.5 hours
- **Phase 4**: Agent - 5.5 hours
- **Phase 5**: Integration Testing - 8 hours
- **Phase 6**: Docker & Deployment - 3.5 hours
- **Phase 7**: Documentation - 5.5 hours
- **Phase 8**: Final Testing - 4 hours

### Critical Path
1. Phase 1 (Foundation) → Phase 2 (Verifier) → Phase 3 (Coordinator) → Phase 4 (Agent)
2. Then parallel: Phase 5 (Testing) + Phase 6 (Docker) + Phase 7 (Docs)
3. Finally: Phase 8 (Validation)

### Key Milestones
- **Milestone 1**: All components compile and run individually (End of Phase 4)
- **Milestone 2**: Full system integration working (End of Phase 5)
- **Milestone 3**: Production-ready PoC (End of Phase 8)

### Next Steps
1. Review this TODO with stakeholders
2. Set up development environment
3. Begin Phase 1 implementation
4. Iterate based on testing feedback

---

**Document Version**: 1.0
**Last Updated**: 2025-11-03
**Estimated Completion**: 6-7 working days (assuming 8 hours/day)
