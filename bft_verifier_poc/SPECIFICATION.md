# BFT Verifier PoC - Technical Specification

## 1. Project Overview

### 1.1 Purpose
Proof of Concept implementation of a Byzantine Fault Tolerant (BFT) consensus system for distributed metric verification in LLM serving environments. This system ensures tamper-proof metrics collection across distributed vLLM servers using TPM-based attestation and BFT consensus.

### 1.2 Goals
- **Primary**: Demonstrate BFT consensus for metric verification with 2f+1 quorum
- **Secondary**: Validate VRF-based verifier selection mechanism
- **Tertiary**: Measure performance overhead and identify optimization opportunities

### 1.3 Non-Goals (for PoC)
- Production-ready deployment scripts
- Advanced blockchain integration (optional recording only)
- Full TPM hardware integration (use simulated quotes)
- Multi-datacenter consensus

### 1.4 Success Criteria
- ✅ 3+ verifiers reaching consensus on metric verification
- ✅ Byzantine fault tolerance (tolerating 1 malicious verifier in 4-verifier setup)
- ✅ VRF-based fair verifier selection
- ✅ <100ms consensus latency for 3-verifier setup
- ✅ Detect and reject tampered metrics

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Distributed System                        │
│                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                │
│  │ Agent 1  │   │ Agent 2  │   │ Agent N  │                │
│  │ (vLLM +  │   │ (vLLM +  │   │ (vLLM +  │                │
│  │ Metrics) │   │ Metrics) │   │ Metrics) │                │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                │
│       │              │              │                        │
│       │ Submit       │ Submit       │ Submit                 │
│       │ Metrics+     │ Metrics+     │ Metrics+               │
│       │ Quote        │ Quote        │ Quote                  │
│       ▼              ▼              ▼                        │
│  ┌─────────────────────────────────────────┐                │
│  │      Consensus Coordinator              │                │
│  │  - VRF Verifier Selection               │                │
│  │  - Epoch Management                     │                │
│  │  - Vote Collection                      │                │
│  │  - Consensus Decision                   │                │
│  └──────────────┬──────────────────────────┘                │
│                 │                                            │
│                 │ Select Committee (2f+1)                    │
│                 ▼                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Verifier 1│ │Verifier 2│ │Verifier 3│ │Verifier 4│       │
│  │  - TPM   │ │  - TPM   │ │  - TPM   │ │  - TPM   │       │
│  │  Quote   │ │  Quote   │ │  Quote   │ │  Quote   │       │
│  │  Check   │ │  Check   │ │  Check   │ │  Check   │       │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │
│       │            │            │            │              │
│       └────────────┴────────────┴────────────┘              │
│                    │                                         │
│                    │ Votes (PASS/FAIL)                       │
│                    ▼                                         │
│       ┌────────────────────────┐                            │
│       │   Consensus Result     │                            │
│       │   ✓ PASS (2f+1 agree)  │                            │
│       │   ✗ FAIL (no quorum)   │                            │
│       └────────────────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Roles

#### Agent
- Collects LLM metrics (tokens, latency, cost)
- Generates simulated TPM quote
- Submits metrics + quote to Coordinator

#### Consensus Coordinator
- Manages verifier pool
- Selects committee using VRF for each epoch
- Broadcasts verification requests to selected verifiers
- Collects votes and determines consensus

#### Verifier
- Verifies TPM quote signature
- Validates metric integrity
- Casts vote (PASS/FAIL)
- Reports result to Coordinator

## 3. Core Components

### 3.1 Data Structures

#### MetricSubmission
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSubmission {
    pub agent_id: String,
    pub request_id: String,
    pub timestamp: u64,
    pub metrics: RequestMetrics,
    pub quote: TpmQuote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestMetrics {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub e2e_latency_ms: u64,
    pub estimated_cost: f64,
    // Simplified for PoC
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TpmQuote {
    pub pcr_values: HashMap<u32, Vec<u8>>, // PCR index -> hash
    pub quote_signature: Vec<u8>,
    pub nonce: Vec<u8>,
}
```

#### Verification Request/Response
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationRequest {
    pub verification_id: String,
    pub epoch: u64,
    pub submission: MetricSubmission,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierVote {
    pub verifier_id: String,
    pub verification_id: String,
    pub result: VerificationResult,
    pub reason: String,
    pub timestamp: u64,
    pub signature: Vec<u8>, // Verifier's signature on vote
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VerificationResult {
    Pass,
    Fail,
}
```

#### Consensus Outcome
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusOutcome {
    pub verification_id: String,
    pub epoch: u64,
    pub committee: Vec<String>,
    pub votes: HashMap<String, VerifierVote>,
    pub result: Option<VerificationResult>,
    pub quorum_reached: bool,
    pub timestamp: u64,
}
```

### 3.2 VRF Verifier Selection

#### Algorithm
```rust
pub struct VrfSelector {
    vrf_seed: [u8; 32],
    committee_size: usize,
}

impl VrfSelector {
    pub fn select_committee(&self, epoch: u64, verifier_pool: &[String]) -> Vec<String> {
        // VRF: seed + epoch -> deterministic but unpredictable random
        let mut input = self.vrf_seed.to_vec();
        input.extend_from_slice(&epoch.to_le_bytes());

        let hash = sha256(&input);
        let seed = u64::from_le_bytes(hash[0..8].try_into().unwrap());

        // Fisher-Yates shuffle with deterministic seed
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pool = verifier_pool.to_vec();
        pool.shuffle(&mut rng);

        pool.into_iter().take(self.committee_size).collect()
    }
}
```

**Properties**:
- **Deterministic**: Same epoch → same committee (all nodes agree)
- **Unpredictable**: Cannot predict future committees
- **Fair**: All verifiers have equal selection probability
- **Verifiable**: Anyone can verify the selection was correct

### 3.3 Consensus Protocol (Simple Voting)

#### State Machine
```
┌─────────┐
│ CREATED │ (verification request created)
└────┬────┘
     │
     │ broadcast to committee
     ▼
┌─────────────┐
│ COLLECTING  │ (collecting votes from verifiers)
└────┬────────┘
     │
     │ timeout or all votes received
     ▼
┌──────────┐
│ TALLYING │ (counting votes)
└────┬─────┘
     │
     ├─────► (2f+1 PASS votes) ──► CONSENSUS_PASS
     │
     ├─────► (2f+1 FAIL votes) ──► CONSENSUS_FAIL
     │
     └─────► (no quorum) ────────► CONSENSUS_FAILED
```

#### Consensus Manager
```rust
pub struct ConsensusManager {
    quorum: usize, // 2f+1
    vote_timeout: Duration,
}

impl ConsensusManager {
    pub async fn collect_votes(
        &self,
        committee: &[String],
        request: VerificationRequest,
    ) -> HashMap<String, VerifierVote> {
        let mut votes = HashMap::new();
        let futures: Vec<_> = committee
            .iter()
            .map(|v_id| self.request_vote(v_id.clone(), request.clone()))
            .collect();

        // Wait for all votes or timeout
        let results = timeout(self.vote_timeout, join_all(futures)).await;

        match results {
            Ok(vote_results) => {
                for (verifier_id, vote_result) in committee.iter().zip(vote_results) {
                    if let Ok(vote) = vote_result {
                        votes.insert(verifier_id.clone(), vote);
                    }
                }
            }
            Err(_) => {
                // Timeout - use partial votes
            }
        }

        votes
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
            (Some(VerificationResult::Pass), format!("{}/{} verifiers agree: PASS", pass_count, votes.len()))
        } else if fail_count >= self.quorum {
            (Some(VerificationResult::Fail), format!("{}/{} verifiers agree: FAIL", fail_count, votes.len()))
        } else {
            (None, format!("No quorum reached (PASS: {}, FAIL: {}, Quorum: {})", pass_count, fail_count, self.quorum))
        }
    }
}
```

### 3.4 Epoch Management

```rust
pub struct EpochManager {
    epoch_duration: Duration,
    start_time: Instant,
}

impl EpochManager {
    pub fn current_epoch(&self) -> u64 {
        self.start_time.elapsed().as_secs() / self.epoch_duration.as_secs()
    }

    pub fn time_until_next_epoch(&self) -> Duration {
        let elapsed = self.start_time.elapsed();
        let current_epoch_end = (self.current_epoch() + 1) * self.epoch_duration.as_secs();
        Duration::from_secs(current_epoch_end) - elapsed
    }
}
```

**Configuration**:
- PoC epoch duration: 30 seconds
- Production recommendation: 5-15 minutes

## 4. Network Protocol

### 4.1 Communication Pattern

**Choice**: gRPC (for PoC simplicity and performance)

**Alternative considered**: HTTP/JSON (easier debugging, but higher overhead)

### 4.2 gRPC Service Definition

```protobuf
syntax = "proto3";

package bft_verifier;

service Verifier {
  rpc Verify(VerificationRequest) returns (VerifierVote);
  rpc Heartbeat(Empty) returns (VerifierStatus);
}

service Coordinator {
  rpc SubmitMetrics(MetricSubmission) returns (SubmissionAck);
  rpc QueryConsensus(VerificationQuery) returns (ConsensusOutcome);
}

message MetricSubmission {
  string agent_id = 1;
  string request_id = 2;
  uint64 timestamp = 3;
  RequestMetrics metrics = 4;
  TpmQuote quote = 5;
}

message RequestMetrics {
  uint32 prompt_tokens = 1;
  uint32 completion_tokens = 2;
  uint64 e2e_latency_ms = 3;
  double estimated_cost = 4;
}

message TpmQuote {
  map<uint32, bytes> pcr_values = 1;
  bytes quote_signature = 2;
  bytes nonce = 3;
}

message VerificationRequest {
  string verification_id = 1;
  uint64 epoch = 2;
  MetricSubmission submission = 3;
}

message VerifierVote {
  string verifier_id = 1;
  string verification_id = 2;
  VerificationResult result = 3;
  string reason = 4;
  uint64 timestamp = 5;
  bytes signature = 6;
}

enum VerificationResult {
  PASS = 0;
  FAIL = 1;
}

message ConsensusOutcome {
  string verification_id = 1;
  uint64 epoch = 2;
  repeated string committee = 3;
  map<string, VerifierVote> votes = 4;
  optional VerificationResult result = 5;
  bool quorum_reached = 6;
  uint64 timestamp = 7;
}

message SubmissionAck {
  string verification_id = 1;
  bool accepted = 2;
}

message VerificationQuery {
  string verification_id = 1;
}

message Empty {}

message VerifierStatus {
  string verifier_id = 1;
  bool healthy = 2;
  uint64 verified_count = 3;
}
```

## 5. Technology Stack

### 5.1 Rust Crates

#### Core
- `tokio` (1.35+): Async runtime
- `tonic` (0.11+): gRPC framework
- `prost` (0.12+): Protocol Buffers

#### Serialization
- `serde` (1.0+): Serialization framework
- `serde_json` (1.0+): JSON support

#### Cryptography
- `sha2` (0.10+): SHA-256 hashing
- `rand` (0.8+): RNG for VRF
- `ed25519-dalek` (2.0+): Signature verification (simplified TPM)

#### Utilities
- `uuid` (1.6+): ID generation
- `tracing` (0.1+): Logging and instrumentation
- `anyhow` (1.0+): Error handling

#### Testing
- `tokio-test`: Async testing utilities
- `mockall` (0.12+): Mocking framework

### 5.2 Development Tools
- `cargo-watch`: Auto-rebuild on changes
- `cargo-nextest`: Fast test runner
- `cargo-flamegraph`: Performance profiling

## 6. Verification Logic (Simplified for PoC)

### 6.1 TPM Quote Verification (Simulated)

For PoC, we simulate TPM quotes using Ed25519 signatures:

```rust
pub struct SimulatedTpmVerifier {
    agent_public_keys: HashMap<String, PublicKey>,
}

impl SimulatedTpmVerifier {
    pub fn verify_quote(
        &self,
        agent_id: &str,
        metrics: &RequestMetrics,
        quote: &TpmQuote,
    ) -> Result<bool, VerificationError> {
        // 1. Get agent's public key
        let public_key = self.agent_public_keys.get(agent_id)
            .ok_or(VerificationError::UnknownAgent)?;

        // 2. Reconstruct message that was signed
        let message = self.compute_pcr_from_metrics(metrics);

        // 3. Verify signature
        let signature = Signature::from_bytes(&quote.quote_signature)?;
        public_key.verify(&message, &signature)
            .map(|_| true)
            .map_err(|_| VerificationError::InvalidSignature.into())
    }

    fn compute_pcr_from_metrics(&self, metrics: &RequestMetrics) -> Vec<u8> {
        // Simulate PCR extension
        let mut hasher = Sha256::new();
        hasher.update(metrics.prompt_tokens.to_le_bytes());
        hasher.update(metrics.completion_tokens.to_le_bytes());
        hasher.update(metrics.e2e_latency_ms.to_le_bytes());
        hasher.finalize().to_vec()
    }
}
```

**Note**: In production, this would use actual TPM 2.0 APIs via `tpm2-tss` crate.

### 6.2 Metric Validation

Basic sanity checks:

```rust
pub fn validate_metrics(metrics: &RequestMetrics) -> Result<(), ValidationError> {
    // Check token counts are reasonable
    if metrics.prompt_tokens == 0 || metrics.prompt_tokens > 1_000_000 {
        return Err(ValidationError::InvalidTokenCount);
    }

    // Check latency is reasonable
    if metrics.e2e_latency_ms == 0 || metrics.e2e_latency_ms > 600_000 {
        return Err(ValidationError::InvalidLatency);
    }

    // Check cost is non-negative
    if metrics.estimated_cost < 0.0 {
        return Err(ValidationError::InvalidCost);
    }

    Ok(())
}
```

## 7. Performance Requirements

### 7.1 Latency Targets

| Metric | Target (3 verifiers) | Target (7 verifiers) |
|--------|---------------------|---------------------|
| Vote collection | < 50ms | < 80ms |
| Consensus decision | < 10ms | < 20ms |
| Total (agent → result) | < 100ms | < 150ms |

### 7.2 Throughput Targets

| Setup | Target |
|-------|--------|
| 3 verifiers | 100 verifications/sec |
| 7 verifiers | 50 verifications/sec |

### 7.3 Resource Limits

- Memory per verifier: < 50MB
- CPU per verifier: < 5% (idle), < 30% (active)
- Network bandwidth: < 1 Mbps per verifier

## 8. Test Scenarios

### 8.1 Functional Tests

#### TC-001: Honest Consensus
- **Setup**: 3 verifiers, all honest, valid metrics
- **Expected**: PASS with 3/3 votes

#### TC-002: Single Byzantine Verifier
- **Setup**: 4 verifiers (f=1), 1 malicious (always votes FAIL)
- **Expected**: PASS with 3/4 votes (quorum = 3)

#### TC-003: Invalid Metrics
- **Setup**: 3 verifiers, metrics with invalid signature
- **Expected**: FAIL with 3/3 votes

#### TC-004: No Quorum
- **Setup**: 4 verifiers, 2 vote PASS, 2 vote FAIL
- **Expected**: Consensus failed (quorum = 3 not reached)

### 8.2 VRF Tests

#### TC-005: Deterministic Selection
- **Setup**: Same epoch, same seed
- **Expected**: Same committee selected

#### TC-006: Epoch Rotation
- **Setup**: Advance epoch
- **Expected**: Different committee selected

#### TC-007: Fair Distribution
- **Setup**: 1000 epochs, 10 verifiers, committee_size=3
- **Expected**: Each verifier selected ~300 times (±10%)

### 8.3 Performance Tests

#### TC-008: Latency Benchmark
- **Setup**: 3 verifiers, 100 sequential verifications
- **Expected**: p50 < 50ms, p99 < 100ms

#### TC-009: Throughput Benchmark
- **Setup**: 3 verifiers, 1000 parallel verifications
- **Expected**: > 100 verifications/sec

### 8.4 Fault Tolerance Tests

#### TC-010: Verifier Crash
- **Setup**: 4 verifiers, 1 crashes during vote collection
- **Expected**: Consensus reached with 3 votes

#### TC-011: Network Partition
- **Setup**: 5 verifiers, 2 unreachable (network timeout)
- **Expected**: Consensus reached with 3 votes if quorum satisfied

## 9. Configuration

### 9.1 Coordinator Config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatorConfig {
    pub listen_addr: String,              // "0.0.0.0:50051"
    pub verifier_endpoints: Vec<String>,  // ["http://v1:50052", "http://v2:50053", ...]
    pub committee_size: usize,            // 3
    pub quorum: usize,                    // 2 (for f=1)
    pub vote_timeout_ms: u64,             // 5000
    pub epoch_duration_secs: u64,         // 30
    pub vrf_seed: String,                 // "hex-encoded-32-bytes"
}
```

### 9.2 Verifier Config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierConfig {
    pub verifier_id: String,              // "verifier-1"
    pub listen_addr: String,              // "0.0.0.0:50052"
    pub signing_key: String,              // "hex-encoded-private-key"
    pub agent_public_keys: HashMap<String, String>, // agent_id -> public_key
}
```

### 9.3 Agent Config

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_id: String,                 // "agent-1"
    pub coordinator_endpoint: String,     // "http://coordinator:50051"
    pub signing_key: String,              // "hex-encoded-private-key"
}
```

## 10. Deployment Architecture (PoC)

### 10.1 Local Development (Docker Compose)

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

  verifier-1:
    build: .
    command: verifier
    environment:
      - VERIFIER_ID=verifier-1
      - RUST_LOG=info

  verifier-2:
    build: .
    command: verifier
    environment:
      - VERIFIER_ID=verifier-2

  verifier-3:
    build: .
    command: verifier
    environment:
      - VERIFIER_ID=verifier-3

  agent-1:
    build: .
    command: agent
    environment:
      - AGENT_ID=agent-1
```

### 10.2 Monitoring

- Prometheus metrics endpoint on each component
- Grafana dashboard for visualization

**Key metrics**:
- `bft_verifications_total` (counter)
- `bft_consensus_latency_seconds` (histogram)
- `bft_vote_collection_duration_seconds` (histogram)
- `bft_votes_received` (gauge)
- `bft_consensus_failures_total` (counter)

## 11. Success Metrics

### 11.1 Correctness
- [ ] All honest verifiers reach consensus (100% in TC-001)
- [ ] Byzantine tolerance works (TC-002 passes)
- [ ] Invalid metrics rejected (100% in TC-003)

### 11.2 Performance
- [ ] Latency < 100ms for 3-verifier setup
- [ ] Throughput > 100 verifications/sec
- [ ] VRF selection fair (TC-007 passes)

### 11.3 Reliability
- [ ] Verifier crash tolerated (TC-010 passes)
- [ ] Network partition tolerated (TC-011 passes)

## 12. Future Work (Post-PoC)

### Phase 2: Production Hardening
- Real TPM 2.0 integration via `tpm2-tss`
- Persistent storage (PostgreSQL/SQLite)
- TLS for all gRPC connections
- Authentication & authorization

### Phase 3: Advanced Features
- PBFT or Tendermint consensus
- Blockchain recording (Ethereum, Hyperledger)
- Multi-datacenter deployment
- Dynamic verifier pool management

### Phase 4: Optimization
- Parallel verification batching
- Optimistic fast path (pre-vote)
- Adaptive timeout based on network conditions
- Verifier reputation system

## 13. References

- [Byzantine Fault Tolerance](https://en.wikipedia.org/wiki/Byzantine_fault)
- [VRF Specification](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vrf-15)
- [TPM 2.0 Specification](https://trustedcomputinggroup.org/resource/tpm-library-specification/)
- [gRPC Best Practices](https://grpc.io/docs/guides/performance/)
- [Rust Async Programming](https://rust-lang.github.io/async-book/)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-03
**Author**: Claude (AI Assistant)
**Status**: Ready for Implementation
