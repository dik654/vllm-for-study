# BFT Verifier PoC - Complete Deployment Guide

Complete guide for deploying the BFT verification system with all Phase 1, Phase 2, and Phase 3 security features.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Level 1: Software-based Security (Phase 1)](#level-1-software-based-security-phase-1)
4. [Level 2: TPM Hardware Security (Phase 2)](#level-2-tpm-hardware-security-phase-2)
5. [Level 3: Full System Integrity (Phase 3)](#level-3-full-system-integrity-phase-3)
6. [Configuration Reference](#configuration-reference)
7. [Monitoring and Troubleshooting](#monitoring-and-troubleshooting)

---

## Quick Start

### Minimal Setup (Development)

```bash
# 1. Clone repository
git clone https://github.com/your-org/vllm-for-study.git
cd vllm-for-study

# 2. Build BFT Verifier PoC
cd bft_verifier_poc
cargo build --release

# 3. Start Coordinator
./target/release/coordinator --config config/coordinator.toml

# 4. Start Verifiers (in separate terminals)
./target/release/verifier --config config/verifier1.toml
./target/release/verifier --config config/verifier2.toml
./target/release/verifier --config config/verifier3.toml

# 5. Start BFT Agent
./target/release/agent --config config/agent.toml

# 6. Start vLLM with metrics collector
cd ../vllm
python -m vllm.llm_metrics_collector.server \
  --pending-queue-dir /tmp/vllm_metrics_queue
```

### Production Setup (with all security features)

See [Level 3](#level-3-full-system-integrity-phase-3) for complete production deployment.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Process (Python)                   │
│  ┌────────────────┐          ┌──────────────────┐          │
│  │ LLM Engine     │────────▶ │ Metrics Collector│          │
│  │ (GPU Inference)│          │  + File Queue    │          │
│  └────────────────┘          └──────────────────┘          │
│                                      │                       │
│                                      ▼ write files           │
│                              pending/*.jsonl                 │
└──────────────────────────────────────────────────────────────┘
                                       │
                                       │ file monitor
                                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    BFT Agent (Rust)                         │
│  ┌─────────────────┐      ┌────────────────────┐           │
│  │ File Monitor    │─────▶│ TPM Agent          │           │
│  │ (polling)       │      │ (Ed25519/TPM/KL)   │           │
│  └─────────────────┘      └────────────────────┘           │
│                                      │                       │
│                                      │ gRPC                  │
└──────────────────────────────────────┼───────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   Coordinator (Rust)                        │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐       │
│  │ NonceManager │  │ AgentTracker│  │ VRF Selector │       │
│  └──────────────┘  └─────────────┘  └──────────────┘       │
│                           │                                  │
│                           │ select committee + distribute    │
└───────────────────────────┼──────────────────────────────────┘
                            ▼
        ┌───────────────────────────────────────┐
        │                                       │
        ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Verifier 1   │    │ Verifier 2   │    │ Verifier 3   │
│              │    │              │    │              │
│ Chain        │    │ Chain        │    │ Chain        │
│ Validator    │    │ Validator    │    │ Validator    │
│              │    │              │    │              │
│ TPM Verifier │    │ TPM Verifier │    │ TPM Verifier │
└──────────────┘    └──────────────┘    └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    BFT Consensus (2f+1 quorum)
                            │
                            ▼
                    ✓ PASS / ✗ FAIL
```

### Component Responsibilities

| Component | Purpose | Security Features |
|-----------|---------|-------------------|
| **vLLM Metrics Collector** | Collect inference metrics | Chain hash generation |
| **BFT Agent** | Generate TPM quotes | Ed25519/TPM/Keylime attestation |
| **Coordinator** | Orchestrate verification | Nonce validation, Rate limiting |
| **Verifiers** | Independent validation | Chain validation, TPM verification |

---

## Level 1: Software-based Security (Phase 1)

**Use Case:** Development, testing, and production systems without TPM hardware

**Security Features:**
- ✅ Ed25519 digital signatures
- ✅ BFT consensus (2f+1 quorum)
- ✅ Replay attack prevention (nonces)
- ✅ Blockchain-style metrics chain
- ✅ Rate limiting and DDoS protection
- ✅ Cost spike detection

### 1.1 Build Level 1 System

```bash
cd bft_verifier_poc
cargo build --release --workspace
```

### 1.2 Configure Coordinator

**config/coordinator.toml:**
```toml
[coordinator]
listen_address = "0.0.0.0:50051"

[vrf]
# VRF seed for committee selection (hex string)
seed = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
committee_size = 3

[consensus]
# N ≥ 3f+1, quorum = 2f+1
# With N=7, f=2, quorum=5
total_verifiers = 7
fault_tolerance = 2
vote_timeout_secs = 5

[epoch]
epoch_duration_secs = 300  # 5 minutes

[verifiers]
pool = [
  "verifier-1",
  "verifier-2",
  "verifier-3",
  "verifier-4",
  "verifier-5",
  "verifier-6",
  "verifier-7"
]

[verifiers.endpoints]
"verifier-1" = "http://192.168.1.101:50052"
"verifier-2" = "http://192.168.1.102:50052"
"verifier-3" = "http://192.168.1.103:50052"
"verifier-4" = "http://192.168.1.104:50052"
"verifier-5" = "http://192.168.1.105:50052"
"verifier-6" = "http://192.168.1.106:50052"
"verifier-7" = "http://192.168.1.107:50052"

# Security: Nonce Manager (Phase 1-A)
[security.nonce_manager]
enabled = true
max_issued_nonces = 100000
max_used_nonces = 100000
cleanup_interval_secs = 3600  # Clean up old nonces every hour

# Security: Agent Tracker (Phase 1-B)
[security.agent_tracker]
enabled = true
min_submission_interval_secs = 1      # Min 1s between submissions
max_requests_per_hour = 3600          # Max 3600 req/hour per agent
cost_history_size = 100               # Track last 100 submissions
max_cost_spike_multiplier = 10.0      # Warn on >10x cost spike
```

### 1.3 Configure Verifiers

**config/verifier1.toml:**
```toml
[verifier]
verifier_id = "verifier-1"
listen_address = "0.0.0.0:50052"

[signing]
# Ed25519 private key (hex string - generate unique per verifier)
private_key = "your-private-key-here"

# Security: Chain Validator (Phase 1-A)
[security.chain_validator]
enabled = true
max_agents = 10000
cleanup_interval_secs = 3600
max_sequence_gap = 10  # Allow up to 10 missed sequences before rejecting
```

**Generate Ed25519 Keys:**
```bash
# Use the provided key generation utility
./target/release/generate-keys

# Output:
# Verifier 1:
#   Private: a1b2c3d4...
#   Public:  e5f6g7h8...
#
# Verifier 2:
#   Private: i9j0k1l2...
#   Public:  m3n4o5p6...
# ...
```

### 1.4 Configure BFT Agent

**config/agent.toml:**
```toml
[agent]
agent_id = "agent-vllm-gpu-001"
coordinator_endpoint = "http://coordinator.example.com:50051"

[file_monitor]
pending_dir = "/tmp/vllm_metrics_queue/pending"
sent_dir = "/tmp/vllm_metrics_queue/sent"
failed_dir = "/tmp/vllm_metrics_queue/failed"
poll_interval_secs = 1
batch_size = 10
retry_failed_interval_secs = 300

[tpm]
# Use simulated TPM (Ed25519)
mode = "simulated"

# Ed25519 signing key for simulated TPM
private_key = "your-agent-private-key-here"
```

### 1.5 Configure vLLM Metrics Collector

**Python configuration:**
```python
from vllm.llm_metrics_collector import PendingQueueStorage

# Initialize queue storage
queue = PendingQueueStorage(
    base_dir="/tmp/vllm_metrics_queue",
    max_queue_size=10000
)

# Enable chain generation
queue.enable_chain(
    agent_id="agent-vllm-gpu-001",
    sequence_start=1
)

# Usage in your vLLM integration
metrics = RequestMetrics(
    prompt_tokens=100,
    completion_tokens=50,
    e2e_latency_ms=1500,
    estimated_cost=0.0025,
    cached_tokens=10,
    time_to_first_token_ms=200
)

# Enqueue metrics (chain fields added automatically)
queue.enqueue(metrics)
```

### 1.6 Start Level 1 System

```bash
# Terminal 1: Coordinator
./target/release/coordinator --config config/coordinator.toml

# Terminal 2-8: Verifiers
./target/release/verifier --config config/verifier1.toml
./target/release/verifier --config config/verifier2.toml
# ... (start all 7 verifiers)

# Terminal 9: BFT Agent
./target/release/agent --config config/agent.toml

# Terminal 10: vLLM
cd ../vllm
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-7b-hf \
  --enable-metrics-collector \
  --metrics-queue-dir /tmp/vllm_metrics_queue
```

### 1.7 Verify Level 1 Security

**Test Replay Attack Prevention:**
```bash
# Submit metrics twice with same nonce
curl -X POST http://coordinator:50051/submit \
  -d '{"agent_id": "test", "nonce": "12345", "metrics": {...}}'

# First: SUCCESS
# Second: FAILED - "Nonce already used (replay attack)"
```

**Test Rate Limiting:**
```bash
# Submit 100 requests in 1 second
for i in {1..100}; do
  curl -X POST http://coordinator:50051/submit \
    -d '{"agent_id": "test", "metrics": {...}}' &
done

# Some requests will be rejected:
# "Submission too frequent: 0.01s < 1s minimum interval"
```

**Test Chain Validation:**
```bash
# Submit metrics with broken chain
curl -X POST http://verifier:50052/verify \
  -d '{"metrics": {"sequence": 5, "prev_hash": "wrong-hash", ...}}'

# Response: FAIL - "Chain broken: prev_hash mismatch"
```

---

## Level 2: TPM Hardware Security (Phase 2)

**Use Case:** Production systems with TPM 2.0 hardware, high-security environments

**Additional Security Features:**
- ✅ Hardware-based attestation (TPM 2.0)
- ✅ PCR extend operations
- ✅ Hardware-signed quotes
- ✅ Tamper-evident metrics

### 2.1 Prerequisites

**Hardware:**
- TPM 2.0 chip (`/dev/tpm0` or `/dev/tpmrm0`)
- Check: `ls -la /dev/tpm*`

**Software:**
```bash
# Install TPM tools
sudo apt-get install tpm2-tools libtss2-dev

# Verify TPM
tpm2_pcrread
```

### 2.2 Build with TPM Support

```bash
cd bft_verifier_poc
cargo build --release --features tpm-hardware
```

### 2.3 Generate TPM Keys

```bash
# Create TPM primary key
tpm2_createprimary -C e -c primary.ctx

# Create TPM signing key
tpm2_create -C primary.ctx -G rsa2048:rsassa:null -u tpm_key.pub -r tpm_key.priv

# Load key
tpm2_load -C primary.ctx -u tpm_key.pub -r tpm_key.priv -c tpm_key.ctx

# Make persistent
tpm2_evictcontrol -C o -c tpm_key.ctx 0x81010001
```

### 2.4 Configure TPM Agent

**config/agent-tpm.toml:**
```toml
[agent]
agent_id = "agent-vllm-tpm-001"
coordinator_endpoint = "http://coordinator.example.com:50051"

[file_monitor]
pending_dir = "/tmp/vllm_metrics_queue/pending"
sent_dir = "/tmp/vllm_metrics_queue/sent"
failed_dir = "/tmp/vllm_metrics_queue/failed"
poll_interval_secs = 1
batch_size = 10

[tpm]
# Use real TPM hardware
mode = "hardware"
device_path = "/dev/tpmrm0"
pcr_index = 16  # Use PCR 16 for metrics
key_handle = "0x81010001"  # Persistent key handle

# Optionally use TPM simulator for testing
# mode = "simulator"
# simulator_endpoint = "127.0.0.1:2321"
```

### 2.5 Configure TPM Verifiers

**config/verifier1-tpm.toml:**
```toml
[verifier]
verifier_id = "verifier-1"
listen_address = "0.0.0.0:50052"

[signing]
private_key = "your-private-key-here"

[tpm]
# Enable TPM quote verification
mode = "hardware"
# Public keys of authorized agents (extract from TPM)
allowed_agents = [
  { agent_id = "agent-vllm-tpm-001", public_key_file = "/etc/bft/agent001.pub" }
]
```

### 2.6 Start Level 2 System

```bash
# Same as Level 1, but using TPM-enabled binaries
./target/release/agent --config config/agent-tpm.toml
./target/release/verifier --config config/verifier1-tpm.toml
# ...
```

### 2.7 Verify TPM Integration

```bash
# Check PCR values
tpm2_pcrread sha256:16

# Monitor PCR extends (should change with each metric submission)
watch -n 1 'tpm2_pcrread sha256:16'

# Verify quote signatures
./target/release/verify-tpm-quote \
  --quote-file /tmp/quote.bin \
  --pcr-index 16 \
  --public-key /etc/bft/agent001.pub
```

---

## Level 3: Full System Integrity (Phase 3)

**Use Case:** Maximum security production environments, compliance requirements

**Additional Security Features:**
- ✅ Keylime remote attestation
- ✅ IMA kernel-level measurements
- ✅ Full binary integrity verification
- ✅ Runtime attestation monitoring

See [PHASE3_KEYLIME_INTEGRATION.md](PHASE3_KEYLIME_INTEGRATION.md) for complete setup.

**Quick Summary:**

1. Install Keylime:
```bash
git clone https://github.com/keylime/keylime.git
cd keylime
sudo python3 setup.py install
```

2. Configure IMA:
```bash
# Add to /etc/default/grub
GRUB_CMDLINE_LINUX="ima_policy=tcb ima_hash=sha256"
sudo update-grub && sudo reboot
```

3. Start Keylime Services:
```bash
sudo systemctl start keylime-registrar
sudo systemctl start keylime-verifier
sudo systemctl start keylime-agent
```

4. Register vLLM Agent:
```bash
keylime_tenant -c add \
  -t agent-vllm-001 \
  -v http://keylime-verifier:8881 \
  --uuid $(uuidgen) \
  --allowlist /etc/bft/allowlist.txt \
  --exclude /etc/bft/excludelist.txt
```

---

## Configuration Reference

### Security Configuration Defaults

| Module | Parameter | Default | Description |
|--------|-----------|---------|-------------|
| **NonceManager** | ||||
| | max_issued_nonces | 100,000 | Max nonces in issued cache |
| | max_used_nonces | 100,000 | Max nonces in used cache |
| | cleanup_interval_secs | 3600 | Cleanup old nonces every 1h |
| **AgentTracker** | ||||
| | min_submission_interval_secs | 1 | Min time between submissions |
| | max_requests_per_hour | 3600 | Hourly rate limit |
| | cost_history_size | 100 | Track last N submissions |
| | max_cost_spike_multiplier | 10.0 | Warn threshold for cost spikes |
| **ChainValidator** | ||||
| | max_agents | 10,000 | Max tracked agents |
| | cleanup_interval_secs | 3600 | Cleanup inactive agents every 1h |
| | max_sequence_gap | 10 | Max allowed sequence gap |

### Performance Tuning

**High-Throughput Deployment (>10K req/s):**
```toml
[security.agent_tracker]
min_submission_interval_secs = 0.01  # 100 req/s per agent
max_requests_per_hour = 360000       # 100 req/s sustained

[consensus]
vote_timeout_secs = 1  # Reduce latency

[epoch]
epoch_duration_secs = 60  # Faster rotation
```

**Low-Latency Deployment (<100ms p99):**
```toml
[coordinator]
speculative_execution = true  # Pre-warm next epoch committee

[verifiers]
connection_pool_size = 100  # More concurrent connections

[agent.file_monitor]
poll_interval_secs = 0.1  # 100ms polling
batch_size = 50           # Batch processing
```

---

## Monitoring and Troubleshooting

### Prometheus Metrics

All components expose Prometheus metrics on `/metrics`:

```bash
# Coordinator
curl http://coordinator:9090/metrics

# Key metrics:
# - bft_submissions_total
# - bft_nonce_validation_failures_total
# - bft_rate_limit_rejections_total
# - bft_consensus_latency_seconds

# Verifier
curl http://verifier:9091/metrics

# Key metrics:
# - bft_verifications_total
# - bft_chain_validation_failures_total
# - bft_tpm_verification_failures_total
```

### Log Analysis

**Check for replay attacks:**
```bash
grep "Nonce already used" coordinator.log
```

**Check for rate limiting:**
```bash
grep "Rate limit exceeded" coordinator.log | wc -l
```

**Check for chain breaks:**
```bash
grep "Chain broken" verifier*.log
```

**Check for cost spikes:**
```bash
grep "Cost spike detected" coordinator.log
```

### Health Checks

```bash
# Coordinator health
curl http://coordinator:50051/health

# Verifier health
curl http://verifier:50052/heartbeat

# Agent status
./target/release/agent --status
```

### Troubleshooting Common Issues

**Issue: "Nonce validation failed"**
```bash
# Solution: Ensure agent requests nonces before submission
# Agent should call /get-nonce endpoint first
```

**Issue: "Chain validation failed: prev_hash mismatch"**
```bash
# Solution: Agent missed metrics or sequence out of order
# Check agent.log for file processing errors
# Verify pending queue is not corrupted
```

**Issue: "TPM quote verification failed"**
```bash
# Solution: Check TPM public key registration
tpm2_readpublic -c 0x81010001

# Verify PCR index matches configuration
tpm2_pcrread sha256:16
```

**Issue: "Rate limit exceeded"**
```bash
# Solution: Adjust rate limits or add more agents
# Edit coordinator.toml:
[security.agent_tracker]
max_requests_per_hour = 7200  # Increase limit
```

---

## Next Steps

1. **Deploy Level 1** for development/testing
2. **Deploy Level 2** for production with TPM hardware
3. **Deploy Level 3** for maximum security compliance
4. **Monitor** using Prometheus + Grafana dashboards
5. **Scale** by adding more verifiers (maintain N ≥ 3f+1)

For advanced topics, see:
- [SECURITY_INTEGRATION_GUIDE.md](SECURITY_INTEGRATION_GUIDE.md) - Code-level integration
- [PHASE2_TPM_INTEGRATION.md](PHASE2_TPM_INTEGRATION.md) - TPM deep dive
- [PHASE3_KEYLIME_INTEGRATION.md](PHASE3_KEYLIME_INTEGRATION.md) - Keylime setup
- [PERFORMANCE_OPTIMIZATIONS.md](PERFORMANCE_OPTIMIZATIONS.md) - Performance tuning
