# TPM Integration - Next Steps

## Status: Level 2 (TPM Hardware Security) - Ready for Integration

### Completed ✅

1. **Phase 1-A: Core Security Modules**
   - [x] NonceManager (replay attack prevention)
   - [x] ChainValidator (blockchain-style metrics chain)
   - [x] Timestamp hardening (±60s validation)
   - [x] 15 comprehensive unit tests

2. **Phase 1-B: DDoS Protection**
   - [x] AgentTracker (rate limiting)
   - [x] Cost spike detection
   - [x] Multi-agent isolation

3. **Phase 2: TPM Infrastructure**
   - [x] Feature flag support (tpm-hardware)
   - [x] TpmConfig struct in agent
   - [x] Configuration files (simulated + hardware)
   - [x] TPM key generation script (generate-tpm-keys.sh)
   - [x] TPM simulator setup script (setup-tpm-simulator.sh)
   - [x] RealTpmAgent implementation (agent/src/tpm_real.rs)

4. **Phase 3: Keylime Documentation**
   - [x] Complete Keylime integration guide
   - [x] IMA configuration
   - [x] Systemd service files
   - [x] Deployment scripts

5. **Integration**
   - [x] NonceManager integrated into Coordinator
   - [x] AgentTracker integrated into Coordinator
   - [x] ChainValidator integrated into Verifier
   - [x] Extended RequestMetrics with chain fields

6. **Documentation**
   - [x] DEPLOYMENT_GUIDE.md (670 lines)
   - [x] SECURITY_SUMMARY.md (607 lines)
   - [x] SECURITY_INTEGRATION_GUIDE.md
   - [x] PHASE2_TPM_INTEGRATION.md
   - [x] PHASE3_KEYLIME_INTEGRATION.md

---

## Next Steps 🔄

### Task 1: Complete RealTpmAgent Integration in main.rs

**Priority**: HIGH
**Estimated Time**: 3-4 hours

**Objective**: Fully integrate RealTpmAgent into agent/src/main.rs

**Current Status**:
- RealTpmAgent implemented in agent/src/tpm_real.rs ✅
- Placeholder error in main.rs ⚠️

**Implementation Steps**:

1. **Import RealTpmAgent** (agent/src/main.rs):
```rust
#[cfg(feature = "tpm-hardware")]
use bft_agent::tpm_real::RealTpmAgent;
```

2. **Create TPM Agent Trait**:
```rust
// agent/src/lib.rs
pub trait TpmAgent {
    fn generate_quote(&self, metrics: &RequestMetrics) -> TpmQuote;
}

impl TpmAgent for SimulatedTpmAgent { ... }
#[cfg(feature = "tpm-hardware")]
impl TpmAgent for RealTpmAgent { ... }
```

3. **Update main.rs TPM initialization**:
```rust
#[cfg(feature = "tpm-hardware")]
if use_real_tpm {
    let tpm_agent = RealTpmAgent::new(
        &config.tpm.device_path,
        config.tpm.pcr_index,
        &config.tpm.key_handle
    )?;
    run_with_tpm_agent(&mut client, &tpm_agent, ...).await?;
} else {
    let tpm_agent = SimulatedTpmAgent::new(signing_key);
    run_with_tpm_agent(&mut client, &tpm_agent, ...).await?;
}

#[cfg(not(feature = "tpm-hardware"))]
{
    let tpm_agent = SimulatedTpmAgent::new(signing_key);
    run_with_tpm_agent(&mut client, &tpm_agent, ...).await?;
}
```

4. **Refactor run_* functions to use trait**:
```rust
async fn run_single_mode<T: TpmAgent>(
    client: &mut CoordinatorClient,
    tpm_agent: &T,
    ...
) -> Result<()>
```

**Testing**:
```bash
# Test with simulator
sudo ./scripts/setup-tpm-simulator.sh
source /etc/bft-tpm-env
./scripts/generate-tpm-keys-simulator.sh agent-test
cargo build --release --features tpm-hardware
AGENT_CONFIG=./tpm_keys/agent-test/agent_config.json ./target/release/agent
```

**Files to Modify**:
- agent/src/lib.rs (add TpmAgent trait)
- agent/src/main.rs (integrate RealTpmAgent)
- agent/src/tpm.rs (impl TpmAgent for SimulatedTpmAgent)
- agent/src/tpm_real.rs (impl TpmAgent for RealTpmAgent)

---

### Task 2: Verifier TPM Quote Verification

**Priority**: HIGH
**Estimated Time**: 2-3 hours

**Objective**: Enhance verifier to validate real TPM quotes

**Current Status**:
- SimulatedTpmVerifier validates Ed25519 signatures ✅
- No TPM quote structure validation ⚠️

**Implementation Steps**:

1. **Create TpmQuoteVerifier** (verifier/src/tpm_verifier.rs):
```rust
pub struct TpmQuoteVerifier {
    allowed_agents: HashMap<String, AgentTpmInfo>,
}

pub struct AgentTpmInfo {
    pub public_key_pem: String,
    pub expected_pcr_policy: Option<Vec<u8>>,
}

impl TpmQuoteVerifier {
    pub fn verify_tpm_quote(
        &self,
        agent_id: &str,
        metrics: &RequestMetrics,
        quote: &TpmQuote,
    ) -> Result<bool> {
        // 1. Verify agent is registered
        // 2. Verify quote signature with TPM public key
        // 3. Verify PCR values (if policy exists)
        // 4. Verify nonce matches metrics hash
    }
}
```

2. **Update verifier config**:
```toml
# verifier/config/verifier.toml
[tpm]
mode = "hardware"  # or "simulated"

[[tpm.allowed_agents]]
agent_id = "agent-vllm-001"
public_key_file = "/etc/bft/agent001.pub"
pcr_policy_file = "/etc/bft/agent001_pcr_policy.json"
```

3. **Integrate into VerifierService**:
```rust
// verifier/src/server.rs
pub struct VerifierService {
    tpm_verifier: Arc<RwLock<TpmQuoteVerifier>>,
    // ...
}
```

**Testing**:
```bash
# Generate agent key
sudo ./scripts/generate-tpm-keys.sh agent-001

# Copy public key to verifier
sudo cp tpm_keys/agent-001/public_key.pem /etc/bft/agent001.pub

# Update verifier config
# Run verifier
./target/release/verifier --config config/verifier_tpm.toml
```

**Files to Create**:
- verifier/src/tpm_verifier.rs (new)
- verifier/config/verifier_tpm.toml (new)

**Files to Modify**:
- verifier/src/lib.rs (add tpm_verifier module)
- verifier/src/server.rs (integrate TpmQuoteVerifier)

---

### Task 3: End-to-End TPM Testing

**Priority**: MEDIUM
**Estimated Time**: 4-6 hours

**Objective**: Create comprehensive integration tests

**Test Scenarios**:

1. **test_tpm_simulator_e2e.rs**:
```rust
#[tokio::test]
async fn test_tpm_simulator_full_flow() {
    // 1. Setup TPM simulator
    // 2. Generate keys
    // 3. Start coordinator + verifiers
    // 4. Start agent with TPM
    // 5. Submit metrics
    // 6. Verify consensus
    // 7. Verify PCR extended
}
```

2. **test_tpm_quote_tampering.rs**:
```rust
#[tokio::test]
async fn test_tampered_quote_rejected() {
    // 1. Generate valid quote
    // 2. Modify quote signature
    // 3. Submit to verifier
    // 4. Verify rejection
}
```

3. **test_tpm_pcr_verification.rs**:
```rust
#[tokio::test]
async fn test_pcr_chain_validation() {
    // 1. Submit 10 metrics
    // 2. Verify PCR extended 10 times
    // 3. Verify PCR value matches expected chain
}
```

4. **test_tpm_key_rotation.rs**:
```rust
#[tokio::test]
async fn test_agent_key_rotation() {
    // 1. Start with key A
    // 2. Submit metrics
    // 3. Rotate to key B
    // 4. Submit metrics
    // 5. Verify both keys work
}
```

**Testing Infrastructure**:
```bash
# tests/common/tpm_test_utils.rs
pub async fn setup_tpm_simulator() -> Result<TpmSimulator>
pub async fn generate_test_tpm_key() -> Result<TpmTestKey>
pub async fn start_test_services() -> Result<TestCluster>
```

**Files to Create**:
- tests/tpm/test_tpm_simulator_e2e.rs
- tests/tpm/test_tpm_quote_tampering.rs
- tests/tpm/test_tpm_pcr_verification.rs
- tests/tpm/test_tpm_key_rotation.rs
- tests/common/tpm_test_utils.rs

---

### Task 4: Performance Benchmarking

**Priority**: MEDIUM
**Estimated Time**: 2-3 hours

**Objective**: Measure TPM performance impact

**Benchmarks**:

1. **TPM Quote Generation**:
```rust
// benches/tpm_quote_generation.rs
fn bench_simulated_quote(b: &mut Bencher) { ... }
fn bench_tpm_hardware_quote(b: &mut Bencher) { ... }
```

2. **PCR Extend Operations**:
```rust
fn bench_pcr_extend(b: &mut Bencher) { ... }
```

3. **Quote Verification**:
```rust
fn bench_quote_verification_ed25519(b: &mut Bencher) { ... }
fn bench_quote_verification_tpm(b: &mut Bencher) { ... }
```

4. **End-to-End Latency**:
```rust
fn bench_e2e_simulated(b: &mut Bencher) { ... }
fn bench_e2e_tpm_hardware(b: &mut Bencher) { ... }
```

**Expected Results**:
- Simulated TPM: <1ms per quote
- Hardware TPM: 5-10ms per quote
- PCR extend: 2-5ms
- Quote verification: 1-3ms

**Files to Create**:
- benches/tpm_quote_generation.rs
- benches/tpm_pcr_operations.rs
- benches/tpm_verification.rs
- benches/tpm_e2e_latency.rs

---

### Task 5: Production Deployment Guide

**Priority**: MEDIUM
**Estimated Time**: 2-3 hours

**Objective**: Complete production deployment documentation

**Updates to DEPLOYMENT_GUIDE.md**:

1. **Hardware Requirements Section**:
   - TPM 2.0 chip requirements
   - Motherboard compatibility
   - BIOS settings
   - Linux kernel requirements

2. **Installation Steps**:
   - Package installation
   - TPM initialization
   - Key generation automation
   - Service configuration

3. **Monitoring and Alerting**:
   - TPM health checks
   - PCR monitoring
   - Quote failure alerts
   - Key rotation procedures

4. **Troubleshooting**:
   - TPM device not found
   - Quote generation failures
   - PCR extend failures
   - Resource Manager issues

**New Files**:
- docs/TPM_PRODUCTION_GUIDE.md
- docs/TPM_TROUBLESHOOTING.md
- docs/TPM_MONITORING.md

---

### Task 6: Security Audit and Documentation

**Priority**: LOW
**Estimated Time**: 4-6 hours

**Objective**: Comprehensive security review

**Audit Areas**:

1. **TPM Key Management**:
   - Key generation security
   - Key storage protection
   - Key rotation procedures
   - Key backup/recovery

2. **Quote Validation**:
   - Signature verification
   - Nonce freshness
   - PCR policy enforcement
   - Replay attack prevention

3. **Attack Scenarios**:
   - TPM quote forgery attempts
   - PCR manipulation
   - Key extraction attacks
   - Side-channel attacks

4. **Compliance**:
   - FIPS 140-2/3 compliance
   - Common Criteria evaluation
   - Industry best practices

**Deliverables**:
- SECURITY_AUDIT_TPM.md
- ATTACK_SCENARIOS_TPM.md
- COMPLIANCE_TPM.md

---

## Timeline

### Sprint 1 (Week 1)
- Task 1: RealTpmAgent integration (3-4 days)
- Task 2: Verifier TPM verification (2-3 days)

### Sprint 2 (Week 2)
- Task 3: End-to-end testing (4-5 days)
- Task 4: Performance benchmarking (2-3 days)

### Sprint 3 (Week 3)
- Task 5: Production deployment guide (2-3 days)
- Task 6: Security audit (4-6 days)

**Total Estimated Time**: 3 weeks (15-20 work days)

---

## Success Criteria

### Functionality
- [ ] Agent can generate TPM quotes with real hardware
- [ ] Verifier validates TPM quotes correctly
- [ ] All integration tests pass
- [ ] Performance benchmarks meet targets (<15ms overhead)

### Quality
- [ ] Code coverage >90% for TPM modules
- [ ] All security tests pass
- [ ] Documentation complete and reviewed
- [ ] Production deployment tested

### Production Readiness
- [ ] Hardware TPM deployment successful
- [ ] Monitoring and alerting configured
- [ ] Runbooks and troubleshooting guides complete
- [ ] Security audit completed and issues resolved

---

## Dependencies

### External Dependencies
- tss-esapi crate (v7.4+)
- TPM 2.0 hardware or simulator
- tpm2-tools (v5.0+)
- libtss2-dev

### Internal Dependencies
- Phase 1 security modules (completed ✅)
- BFT consensus system (completed ✅)
- gRPC infrastructure (completed ✅)

---

## Risk Mitigation

### Technical Risks
1. **TPM hardware availability**: Use simulator for development
2. **Performance issues**: Async TPM operations, batching
3. **Compatibility issues**: Test multiple TPM vendors
4. **Key management complexity**: Automated scripts and tooling

### Schedule Risks
1. **Integration complexity**: Start with simulator, hardware later
2. **Testing delays**: Parallel testing and development
3. **Documentation time**: Document as you go

---

## Notes

- All TPM code is behind feature flag for gradual rollout
- Backward compatibility maintained (simulated mode still works)
- Can deploy Level 1 (simulated) immediately
- Level 2 (TPM) optional upgrade path
- Level 3 (Keylime) documented for future

---

**Last Updated**: 2025-11-06
**Status**: Level 2 Infrastructure Complete, Integration In Progress
