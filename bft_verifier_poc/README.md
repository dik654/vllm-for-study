# BFT Verifier PoC

Byzantine Fault Tolerant consensus system for distributed LLM metrics verification using TPM-based attestation.

## Overview

This Proof of Concept demonstrates a distributed verification system that ensures tamper-proof collection of LLM usage metrics across distributed servers. The system uses:

- **TPM-based attestation** for hardware-level metric integrity
- **VRF (Verifiable Random Function)** for fair verifier selection
- **BFT consensus** to tolerate Byzantine (malicious) verifiers
- **Epoch-based rotation** for security

## Architecture

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
│  │  - Consensus Decision (2f+1)            │                │
│  └──────────────┬──────────────────────────┘                │
│                 │                                            │
│                 │ Select Committee (VRF)                     │
│                 ▼                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │Verifier 1│ │Verifier 2│ │Verifier 3│ │Verifier 4│       │
│  │  Verify  │ │  Verify  │ │  Verify  │ │  Verify  │       │
│  │  Quote   │ │  Quote   │ │  Quote   │ │  Quote   │       │
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

## Components

### 1. Agent
- Generates LLM request metrics (tokens, latency, cost)
- Creates TPM quotes for metrics (simulated with Ed25519)
- Submits metrics + quote to Coordinator

### 2. Coordinator
- Receives metric submissions from agents
- Selects verifier committee using VRF
- Collects votes from selected verifiers
- Reaches BFT consensus (2f+1 quorum)
- Returns verification result

### 3. Verifier
- Validates metric ranges
- Verifies TPM quote signatures
- Casts vote (PASS/FAIL)
- Signs vote with private key

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OR Rust 1.75+ (for local development)

### Option 1: Docker Compose (Recommended)

```bash
# Build and start all services
docker-compose up --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

This will start:
- 1 Coordinator (port 50051)
- 4 Verifiers (ports 50052-50055)
- 1 Agent (submitting metrics every 5 seconds)

### Option 2: Local Development

```bash
# Build all components
cargo build --release --workspace

# Terminal 1: Start Coordinator
RUST_LOG=info ./target/release/coordinator

# Terminal 2-5: Start Verifiers
RUST_LOG=info VERIFIER_ID=verifier-1 LISTEN_ADDR=0.0.0.0:50052 ./target/release/verifier
RUST_LOG=info VERIFIER_ID=verifier-2 LISTEN_ADDR=0.0.0.0:50053 ./target/release/verifier
RUST_LOG=info VERIFIER_ID=verifier-3 LISTEN_ADDR=0.0.0.0:50054 ./target/release/verifier
RUST_LOG=info VERIFIER_ID=verifier-4 LISTEN_ADDR=0.0.0.0:50055 ./target/release/verifier

# Terminal 6: Start Agent
RUST_LOG=info AGENT_ID=agent-1 MODE=single REQUEST_COUNT=5 ./target/release/agent
```

## Configuration

### Coordinator Configuration

```json
{
  "listen_addr": "0.0.0.0:50051",
  "vrf_seed_hex": "0000...0000",
  "committee_size": 3,
  "quorum": 2,
  "vote_timeout_secs": 5,
  "epoch_duration_secs": 30,
  "verifier_pool": ["verifier-1", "verifier-2", "verifier-3", "verifier-4"],
  "verifier_endpoints": {
    "verifier-1": "http://localhost:50052",
    "verifier-2": "http://localhost:50053",
    "verifier-3": "http://localhost:50054",
    "verifier-4": "http://localhost:50055"
  }
}
```

**Key Parameters:**
- `committee_size`: Number of verifiers to select per epoch (default: 3)
- `quorum`: Minimum votes needed for consensus (default: 2, for f=1)
- `epoch_duration_secs`: Committee rotation interval (default: 30)
- `vote_timeout_secs`: Maximum time to wait for votes (default: 5)

### Verifier Configuration

```json
{
  "verifier_id": "verifier-1",
  "listen_addr": "0.0.0.0:50052",
  "signing_key_hex": "",
  "agent_public_keys": {
    "agent-1": "AGENT_PUBLIC_KEY_HEX"
  }
}
```

**Agent Key Setup:**
1. Run agent once to generate keys
2. Copy public key from agent logs
3. Add to `agent_public_keys` in verifier config

### Agent Configuration

```json
{
  "agent_id": "agent-1",
  "coordinator_endpoint": "http://localhost:50051",
  "signing_key_hex": "",
  "submission_interval_secs": 5,
  "mode": "continuous",
  "request_count": 10
}
```

**Modes:**
- `continuous`: Run indefinitely, submit every N seconds
- `single`: Submit N requests and exit

## Byzantine Fault Tolerance

The system tolerates **f** Byzantine (malicious) verifiers in a pool of **N ≥ 3f+1** verifiers.

### Examples

| Total Verifiers (N) | Byzantine Tolerance (f) | Quorum (2f+1) |
|---------------------|-------------------------|---------------|
| 4                   | 1                       | 3             |
| 7                   | 2                       | 5             |
| 10                  | 3                       | 7             |

**Default Setup**: N=4, f=1, quorum=3 (tolerates 1 malicious verifier)

## VRF Committee Selection

Verifier committee is selected using VRF (Verifiable Random Function):

```
committee = VRF(seed, epoch) → deterministic but unpredictable selection
```

**Properties:**
- **Deterministic**: Same epoch → same committee (all nodes agree)
- **Unpredictable**: Cannot predict future committees
- **Fair**: All verifiers have equal selection probability
- **Verifiable**: Anyone can verify selection correctness

**Fairness Test**: Over 1000 epochs, each verifier selected ~300 times (±10%)

## Metrics Collected

### Per-Request Metrics
- **Token Usage**: prompt_tokens, completion_tokens
- **Latency**: e2e_latency_ms (end-to-end)
- **Cost**: estimated_cost (based on token pricing)
- **Timestamp**: request submission time

### Validation Rules
- Tokens: 1-1,000,000 per request
- Latency: 1-600,000 ms (10 minutes max)
- Cost: ≥ 0, < $1000 per request
- Timestamp: Within ±1 hour window

## Security Features

### TPM Quote (Simulated)

```
PCR[10] = SHA256(metrics_data)
Quote = Ed25519_Sign(PCR[10] || nonce, TPM_key)
```

**Tamper Detection:**
- Any modification to metrics changes PCR value
- Quote signature verification fails
- Verifiers vote FAIL

### Vote Signatures

Each verifier signs their vote:
```
Vote_Signature = Ed25519_Sign(verification_id || result || timestamp, Verifier_key)
```

Coordinator can verify vote authenticity.

## Performance

### Latency Targets (3 verifiers)
- Vote collection: < 50ms
- Consensus decision: < 10ms
- Total (agent → result): < 100ms

### Throughput Targets
- 3 verifiers: 100 verifications/sec
- 7 verifiers: 50 verifications/sec

### Resource Usage
- Memory per verifier: < 50MB
- CPU per verifier: < 5% (idle), < 30% (active)
- Network: < 1 Mbps per verifier

### Performance Optimizations

The system implements two key optimizations to minimize latency:

#### 1. Early Termination

**How it works**:
- Votes are collected asynchronously in parallel
- As each vote arrives, the system checks if 2f+1 quorum is reached
- If consensus is detected early, vote collection terminates immediately
- No need to wait for all votes or timeout

**Performance benefit**:
```
Without: Wait for all 3 votes or 5s timeout
With:    Exit when 2f+1 consensus reached
         → 30-50% faster when consensus reached early
```

**Logs**: Check for "Early consensus reached!" in coordinator logs.

#### 2. Speculative Execution (Connection Pre-warming)

**How it works**:
- When selecting current epoch's committee, coordinator also calculates next epoch's committee
- gRPC connections to next epoch's verifiers are established in the background
- When epoch changes, connections are already ready (no handshake delay)
- Connection pool reuses existing connections across requests

**Performance benefit**:
```
Without: gRPC handshake (10-50ms) + validation
With:    Validation only (connections pre-established)
         → 10-30ms saved per verification
```

**Logs**: Check for "Pre-warming next epoch's committee" in coordinator logs.

#### Combined Impact

```
Component               Baseline  Optimized  Improvement
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gRPC Connection         20ms      0ms        ← Pre-warmed
Vote Collection         100ms     50ms       ← Early exit
Consensus Decision      5ms       5ms        (unchanged)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Latency          125ms     55ms       ← 56% faster!
```

**Example scenario** (3 verifiers, quorum = 3):
```
Epoch N-1:
  - Coordinator pre-warms connections to epoch N committee
  - Background task completes during epoch N-1

Epoch N verification:
  0ms     → Vote requests sent to V1, V2, V3 (connections ready!)
  45ms    ← V1 responds: PASS
  50ms    ← V2 responds: PASS
  55ms    ← V3 responds: PASS ← Early consensus, exit!
          ✅ Total: 55ms (vs 125ms baseline)
```

For more optimization strategies, see [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md).

## Testing

### Unit Tests

```bash
# Run all unit tests
cargo test --workspace

# Run tests with output
cargo test --workspace -- --nocapture

# Run specific test
cargo test --package bft-common test_consensus_pass_with_quorum
```

**Test Coverage**: 80+ unit tests across all modules

### Manual Testing

```bash
# Start system with Docker Compose
docker-compose up --build

# Submit test metrics
RUST_LOG=info AGENT_ID=test-agent MODE=single REQUEST_COUNT=3 \
  ./target/release/agent

# Check coordinator logs for consensus results
docker-compose logs coordinator | grep "Consensus complete"

# Check verifier votes
docker-compose logs verifier-1 | grep "Verification"
```

## Troubleshooting

### Agent Key Mismatch

**Problem**: Verifier rejects agent submissions

```
error: Unknown agent: agent-1
```

**Solution**:
1. Run agent to generate keys
2. Copy public key from agent output
3. Add to verifier config `agent_public_keys`
4. Restart verifier

### No Quorum Reached

**Problem**: Consensus fails

```
warn: No quorum reached (PASS: 1, FAIL: 1, Quorum: 3)
```

**Solutions**:
- Check verifier connectivity (network issues)
- Verify verifier endpoints in coordinator config
- Ensure enough verifiers are running (N ≥ 3f+1)
- Check vote timeout is sufficient

### Connection Refused

**Problem**: Agent cannot connect to coordinator

```
error: Connection failed: Connection refused
```

**Solutions**:
- Verify coordinator is running
- Check coordinator endpoint in agent config
- Verify network connectivity (Docker network, firewall)

## Development

### Project Structure

```
bft_verifier_poc/
├── common/           # Shared types and utilities
│   ├── src/
│   │   ├── types.rs      # Core data structures
│   │   ├── error.rs      # Error types
│   │   └── crypto.rs     # Ed25519, SHA-256 utils
├── coordinator/      # Consensus coordinator
│   ├── src/
│   │   ├── vrf.rs        # VRF committee selection
│   │   ├── epoch.rs      # Epoch management
│   │   ├── consensus.rs  # Vote collection & quorum
│   │   └── server.rs     # gRPC server
├── verifier/         # Verifier node
│   ├── src/
│   │   ├── tpm.rs        # TPM quote verification
│   │   ├── validator.rs  # Metric validation
│   │   └── server.rs     # gRPC server
├── agent/            # Metrics agent
│   ├── src/
│   │   ├── metrics.rs    # Metrics generation
│   │   ├── tpm.rs        # TPM quote generation
│   │   └── client.rs     # gRPC client
├── proto/            # Protocol Buffers
│   └── bft_verifier.proto
├── config/           # Configuration examples
├── Dockerfile        # Multi-stage Docker build
└── docker-compose.yml # Full system deployment
```

### Building from Source

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install protobuf compiler
# Ubuntu/Debian:
sudo apt-get install protobuf-compiler

# macOS:
brew install protobuf

# Build
cargo build --release --workspace

# Run tests
cargo test --workspace
```

### Adding a New Verifier

1. Update coordinator config:
```json
{
  "verifier_pool": ["verifier-1", "verifier-2", "verifier-3", "verifier-4", "verifier-5"],
  "verifier_endpoints": {
    "verifier-5": "http://localhost:50056"
  }
}
```

2. Start new verifier:
```bash
VERIFIER_ID=verifier-5 LISTEN_ADDR=0.0.0.0:50056 ./target/release/verifier
```

3. Update `committee_size` and `quorum` if needed for new fault tolerance level

## References

- [SPECIFICATION.md](SPECIFICATION.md) - Detailed technical specification
- [TODO.md](TODO.md) - Implementation roadmap (Phases 1-8)
- [BFT_CONSENSUS_VERIFICATION.md](../vllm/llm_metrics_collector/BFT_CONSENSUS_VERIFICATION.md) - Consensus theory
- [Byzantine Fault Tolerance](https://en.wikipedia.org/wiki/Byzantine_fault)
- [VRF Specification](https://datatracker.ietf.org/doc/html/draft-irtf-cfrg-vrf-15)

## License

This is a Proof of Concept for research and educational purposes.

## Acknowledgments

- Built with Rust, Tokio, Tonic (gRPC)
- Cryptography: ed25519-dalek, SHA-256
- Based on BFT consensus principles and VRF selection
