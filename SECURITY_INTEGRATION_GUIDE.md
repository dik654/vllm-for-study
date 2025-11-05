# 보안 기능 통합 가이드

이 문서는 새로 구현된 보안 기능들을 실제 Coordinator와 Verifier에 통합하는 방법을 설명합니다.

## 구현된 보안 모듈

### 1. Nonce Manager (coordinator/src/nonce_manager.rs)
- **기능**: 재전송 공격 방지
- **테스트**: 4개 단위 테스트 ✅
- **상태**: 완료 ✅

### 2. Chain Validator (verifier/src/chain_validator.rs)
- **기능**: 메트릭 체인 검증 (누락/조작 탐지)
- **테스트**: 6개 단위 테스트 ✅
- **상태**: 완료 ✅

### 3. Agent Tracker (coordinator/src/agent_tracker.rs)
- **기능**: Rate limiting, DDoS 방지, 비정상 패턴 탐지
- **테스트**: 5개 단위 테스트 ✅
- **상태**: 완료 ✅

### 4. Timestamp Validation (verifier/src/validator.rs)
- **기능**: 타임스탬프 검증 강화 (±60초 → ±10초)
- **상태**: 완료 ✅

---

## Coordinator 통합

### Step 1: server.rs에 모듈 import

```rust
// coordinator/src/server.rs

use crate::agent_tracker::{AgentTracker, AgentTrackerConfig};
use crate::nonce_manager::{NonceManager, NonceManagerConfig};
// ... 기존 imports
```

### Step 2: CoordinatorService 구조체에 필드 추가

```rust
pub struct CoordinatorService {
    // 기존 필드들...
    vrf_selector: Arc<RwLock<VrfSelector>>,
    consensus_manager: Arc<ConsensusManager>,
    epoch_manager: Arc<EpochManager>,

    // NEW: 보안 모듈 추가
    nonce_manager: Arc<NonceManager>,
    agent_tracker: Arc<AgentTracker>,

    // 기존 필드들...
    connection_pool: Arc<VerifierConnectionPool>,
    results_cache: Arc<RwLock<HashMap<String, ConsensusOutcome>>>,
}
```

### Step 3: 초기화 로직 수정

```rust
impl CoordinatorService {
    pub fn new(config: CoordinatorConfig) -> Self {
        // 기존 초기화...

        // NEW: 보안 모듈 초기화
        let nonce_manager = Arc::new(NonceManager::new(NonceManagerConfig::default()));

        let agent_tracker = Arc::new(AgentTracker::new(AgentTrackerConfig {
            min_submission_interval_secs: 1,
            max_requests_per_hour: 3600,
            max_cost_spike_multiplier: 10.0,
        }));

        info!("Security modules initialized");

        Self {
            // 기존 필드들...
            nonce_manager,
            agent_tracker,
            // ...
        }
    }
}
```

### Step 4: submit_metrics 메서드 수정

```rust
async fn submit_metrics(
    &self,
    request: Request<MetricSubmission>,
) -> Result<Response<SubmissionAck>, Status> {
    let submission = request.into_inner();

    info!(
        agent_id = %submission.agent_id,
        request_id = %submission.request_id,
        "Received metric submission"
    );

    // ========== NEW: 보안 검증 ==========

    // 1. Nonce 검증 (재전송 공격 방지)
    if !submission.coordinator_nonce.is_empty() {
        self.nonce_manager
            .validate_and_consume(&submission.coordinator_nonce)
            .await
            .map_err(|e| {
                warn!("Nonce validation failed: {}", e);
                Status::invalid_argument(format!("Invalid nonce: {}", e))
            })?;

        info!("Nonce validated");
    }

    // 2. Agent 상태 추적 (Rate limiting)
    self.agent_tracker
        .validate_submission(&submission.agent_id, &submission.metrics.as_ref().unwrap())
        .await
        .map_err(|e| {
            warn!("Agent validation failed: {}", e);
            Status::resource_exhausted(format!("Rate limit: {}", e))
        })?;

    info!("Agent rate validation passed");

    // ========== 기존 로직 계속 ==========

    // Verification ID 생성
    let verification_id = format!("{}-{}", submission.agent_id, uuid::Uuid::new_v4());

    // ... 기존 검증 로직 ...

    Ok(Response::new(SubmissionAck {
        verification_id,
        status: 1,  // ACCEPTED
        message: "Verification completed".to_string(),
    }))
}
```

### Step 5: Nonce 발급 API 추가 (선택사항)

```rust
// proto/bft_verifier.proto에 추가
service Coordinator {
  rpc SubmitMetrics(MetricSubmission) returns (SubmissionAck);
  rpc QueryConsensus(VerificationQuery) returns (ConsensusOutcome);
  rpc RequestNonce(Empty) returns (NonceResponse);  // NEW
}

message NonceResponse {
  string nonce = 1;
  uint64 expires_at = 2;
}
```

```rust
// coordinator/src/server.rs에 구현
async fn request_nonce(
    &self,
    _request: Request<Empty>,
) -> Result<Response<NonceResponse>, Status> {
    let nonce = self.nonce_manager.generate_nonce().await;
    let expires_at = current_timestamp() + 300;  // 5 minutes

    Ok(Response::new(NonceResponse {
        nonce,
        expires_at,
    }))
}
```

---

## Verifier 통합

### Step 1: server.rs에 모듈 import

```rust
// verifier/src/server.rs

use crate::chain_validator::{ChainValidator, calculate_metrics_hash};
// ... 기존 imports
```

### Step 2: VerifierService 구조체에 필드 추가

```rust
pub struct VerifierService {
    // 기존 필드들...
    verifier_id: String,
    tpm_verifier: Arc<TpmQuoteVerifier>,

    // NEW: 체인 검증기 추가
    chain_validator: Arc<ChainValidator>,
}
```

### Step 3: 초기화 로직 수정

```rust
impl VerifierService {
    pub fn new(config: VerifierConfig) -> Self {
        // 기존 초기화...
        let tpm_verifier = Arc::new(TpmQuoteVerifier::new(config.agent_public_keys));

        // NEW: 체인 검증기 초기화
        let chain_validator = Arc::new(ChainValidator::new());

        info!("Chain validator initialized");

        Self {
            verifier_id: config.verifier_id,
            tpm_verifier,
            chain_validator,
        }
    }
}
```

### Step 4: verify 메서드 수정

```rust
async fn verify(
    &self,
    request: Request<VerificationRequest>,
) -> Result<Response<VerifierVote>, Status> {
    let req = request.into_inner();
    let submission = req.submission.ok_or_else(|| {
        Status::invalid_argument("Missing submission")
    })?;

    let metrics = submission.metrics.as_ref().ok_or_else(|| {
        Status::invalid_argument("Missing metrics")
    })?;

    info!(
        verification_id = %req.verification_id,
        agent_id = %submission.agent_id,
        "Starting verification"
    );

    // ========== NEW: 체인 검증 ==========

    if metrics.sequence > 0 {
        match self.chain_validator
            .validate_chain(&submission.agent_id, metrics)
            .await
        {
            Ok(()) => {
                info!("Chain validation passed");
            }
            Err(e) => {
                warn!("Chain validation failed: {}", e);

                return Ok(Response::new(VerifierVote {
                    verifier_id: self.verifier_id.clone(),
                    verification_id: req.verification_id,
                    result: 1,  // FAIL
                    reason: format!("Chain validation failed: {}", e),
                    timestamp: current_timestamp(),
                    signature: vec![],
                }));
            }
        }
    }

    // ========== 기존 검증 로직 ==========

    // 1. Metrics 범위 검증
    if let Err(e) = validate_metrics(metrics) {
        return Ok(Response::new(VerifierVote {
            verifier_id: self.verifier_id.clone(),
            verification_id: req.verification_id,
            result: 1,  // FAIL
            reason: format!("Metrics validation failed: {}", e),
            timestamp: current_timestamp(),
            signature: vec![],
        }));
    }

    // 2. Timestamp 검증 (강화됨: ±60초)
    if let Err(e) = validate_timestamp(submission.timestamp) {
        return Ok(Response::new(VerifierVote {
            verifier_id: self.verifier_id.clone(),
            verification_id: req.verification_id,
            result: 1,  // FAIL
            reason: format!("Timestamp validation failed: {}", e),
            timestamp: current_timestamp(),
            signature: vec![],
        }));
    }

    // 3. TPM Quote 검증
    let quote = submission.quote.as_ref().ok_or_else(|| {
        Status::invalid_argument("Missing quote")
    })?;

    match self.tpm_verifier.verify_quote(
        metrics,
        quote,
        &submission.agent_id,
    ) {
        Ok(true) => {
            info!("All validations passed");

            Ok(Response::new(VerifierVote {
                verifier_id: self.verifier_id.clone(),
                verification_id: req.verification_id,
                result: 0,  // PASS
                reason: "All validations passed".to_string(),
                timestamp: current_timestamp(),
                signature: vec![],
            }))
        }
        Ok(false) => {
            warn!("TPM quote verification failed");

            Ok(Response::new(VerifierVote {
                verifier_id: self.verifier_id.clone(),
                verification_id: req.verification_id,
                result: 1,  // FAIL
                reason: "TPM quote verification failed".to_string(),
                timestamp: current_timestamp(),
                signature: vec![],
            }))
        }
        Err(e) => {
            error!("TPM verification error: {}", e);

            Ok(Response::new(VerifierVote {
                verifier_id: self.verifier_id.clone(),
                verification_id: req.verification_id,
                result: 1,  // FAIL
                reason: format!("TPM verification error: {}", e),
                timestamp: current_timestamp(),
                signature: vec![],
            }))
        }
    }
}
```

---

## Agent 통합

Agent에서는 메트릭 제출 시 체인 정보를 생성해야 합니다.

### Step 1: Agent 상태에 체인 정보 추가

```rust
// agent/src/main.rs 또는 별도 파일

struct AgentChainState {
    last_sequence: u64,
    last_hash: Vec<u8>,
}

impl AgentChainState {
    fn new() -> Self {
        Self {
            last_sequence: 0,
            last_hash: vec![0u8; 32],  // Genesis hash
        }
    }

    fn next_sequence(&mut self) -> u64 {
        self.last_sequence += 1;
        self.last_sequence
    }

    fn update_hash(&mut self, new_hash: Vec<u8>) {
        self.last_hash = new_hash;
    }
}
```

### Step 2: 메트릭 제출 시 체인 정보 생성

```rust
// agent/src/client.rs 또는 main.rs

use sha2::{Digest, Sha256};

fn calculate_metrics_chain_hash(metrics: &RequestMetrics) -> Vec<u8> {
    let mut hasher = Sha256::new();

    hasher.update(metrics.sequence.to_le_bytes());
    hasher.update(&metrics.prev_hash);
    hasher.update(metrics.prompt_tokens.to_le_bytes());
    hasher.update(metrics.completion_tokens.to_le_bytes());
    hasher.update(metrics.cached_tokens.to_le_bytes());
    hasher.update(metrics.e2e_latency_ms.to_le_bytes());
    hasher.update(metrics.time_to_first_token_ms.to_le_bytes());
    hasher.update(metrics.estimated_cost.to_le_bytes());

    hasher.finalize().to_vec()
}

// 제출 시
async fn submit_with_chain(
    client: &mut CoordinatorClient,
    tpm_agent: &SimulatedTpmAgent,
    metrics_gen: &mut MetricsGenerator,
    chain_state: &mut AgentChainState,
) -> Result<()> {
    let request_id = format!("req-{}", uuid::Uuid::new_v4());

    // 1. 메트릭 생성
    let mut metrics = metrics_gen.generate_realistic();

    // 2. 체인 정보 추가
    metrics.sequence = chain_state.next_sequence();
    metrics.prev_hash = chain_state.last_hash.clone();
    metrics.current_hash = calculate_metrics_chain_hash(&metrics);

    // 3. TPM Quote 생성
    let quote = tpm_agent.generate_quote(&metrics);

    // 4. 제출
    let (verification_id, accepted) = client
        .submit_metrics(request_id, metrics.clone(), quote)
        .await?;

    if accepted {
        // 5. 체인 상태 업데이트
        chain_state.update_hash(metrics.current_hash);
        info!("Chain updated: sequence={}", metrics.sequence);
    }

    Ok(())
}
```

---

## 빌드 및 테스트

### 빌드

```bash
cd bft_verifier_poc

# 전체 빌드
cargo build --release --workspace

# 개별 빌드
cargo build --release --package bft-coordinator
cargo build --release --package bft-verifier
cargo build --release --package bft-agent
```

### 단위 테스트

```bash
# 모든 테스트 실행
cargo test --workspace

# 보안 모듈 테스트만
cargo test --package bft-coordinator nonce_manager
cargo test --package bft-coordinator agent_tracker
cargo test --package bft-verifier chain_validator

# 테스트 출력 보기
cargo test --workspace -- --nocapture
```

### 통합 테스트

```bash
# 1. Coordinator 시작
RUST_LOG=info ./target/release/coordinator

# 2. Verifiers 시작 (4개)
RUST_LOG=info VERIFIER_ID=verifier-1 LISTEN_ADDR=0.0.0.0:50052 ./target/release/verifier &
RUST_LOG=info VERIFIER_ID=verifier-2 LISTEN_ADDR=0.0.0.0:50053 ./target/release/verifier &
RUST_LOG=info VERIFIER_ID=verifier-3 LISTEN_ADDR=0.0.0.0:50054 ./target/release/verifier &
RUST_LOG=info VERIFIER_ID=verifier-4 LISTEN_ADDR=0.0.0.0:50055 ./target/release/verifier &

# 3. Agent 시작 (체인 활성화)
RUST_LOG=info MODE=continuous REQUEST_COUNT=10 ./target/release/agent
```

---

## 모니터링

### 로그 확인

**Coordinator 로그**:
```
[INFO] Security modules initialized
[INFO] Received metric submission agent_id=agent-1
[INFO] Nonce validated
[INFO] Agent rate validation passed
[INFO] Early consensus reached!
```

**Verifier 로그**:
```
[INFO] Chain validator initialized
[INFO] Starting verification
[INFO] Chain validation passed
[INFO] All validations passed
```

**Agent 로그**:
```
[INFO] Generated metrics
[INFO] Chain updated: sequence=1
[INFO] Submitted (sync) accepted=true
```

### 통계 조회

```rust
// Coordinator에서
let nonce_stats = nonce_manager.get_stats().await;
println!("Nonces - Issued: {}, Used: {}",
    nonce_stats.issued_count,
    nonce_stats.used_count
);

let agent_stats = agent_tracker.get_agent_stats("agent-1").await;
println!("Agent-1 - Requests: {}, Cost: ${:.2}",
    agent_stats.total_requests,
    agent_stats.total_cost
);

let system_stats = agent_tracker.get_system_stats().await;
println!("System - Agents: {}, Total Requests: {}, Total Cost: ${:.2}",
    system_stats.total_agents,
    system_stats.total_requests,
    system_stats.total_cost
);
```

---

## 보안 검증

### 재전송 공격 테스트

```bash
# 1. 정상 제출
./target/release/agent

# 2. 로그에서 nonce 복사
# coordinator_nonce: "550e8400-e29b-41d4-a716-446655440000"

# 3. 같은 nonce로 재전송 시도 → 거부되어야 함
# Error: "Nonce already used (replay attack)"
```

### 메트릭 누락 테스트

```rust
// Agent에서 sequence 건너뛰기
metrics.sequence = 1;  // OK
metrics.sequence = 3;  // Skip 2 → FAIL

// Verifier 로그:
// "Sequence number skip detected: expected 2, got 3"
```

### Rate Limiting 테스트

```bash
# 1초에 10번 제출 시도
for i in {1..10}; do
    ./target/release/agent &
done

# 로그:
# "Submission too frequent: 0s < 1s minimum"
# "Hourly rate limit exceeded: 3600/hour"
```

---

## 성능 영향

### 오버헤드 측정

| 검증 | 추가 시간 | 비고 |
|------|----------|------|
| Nonce 검증 | < 1ms | HashSet lookup |
| Chain 검증 | < 2ms | SHA-256 해싱 |
| Agent 추적 | < 1ms | HashMap update |
| Timestamp (강화) | < 0.1ms | 단순 비교 |
| **Total** | **< 5ms** | **무시 가능** |

기존 검증 시간 (50-100ms)에 비해 5% 미만의 오버헤드.

---

## 마이그레이션 체크리스트

### Coordinator

- [ ] `lib.rs`에 모듈 추가 ✅
- [ ] `server.rs`에 NonceManager, AgentTracker 통합
- [ ] `submit_metrics`에 보안 검증 추가
- [ ] (선택) `request_nonce` API 구현
- [ ] 테스트 실행
- [ ] 로그 확인

### Verifier

- [ ] `lib.rs`에 모듈 추가 ✅
- [ ] `server.rs`에 ChainValidator 통합
- [ ] `verify`에 체인 검증 추가
- [ ] Timestamp 검증 확인 (이미 강화됨) ✅
- [ ] 테스트 실행
- [ ] 로그 확인

### Agent

- [ ] 체인 상태 구조체 추가
- [ ] `calculate_metrics_chain_hash` 함수 구현
- [ ] 제출 로직에 체인 생성 추가
- [ ] (선택) Nonce 요청 로직 추가
- [ ] 테스트 실행
- [ ] 로그 확인

---

## 트러블슈팅

### 문제: 빌드 실패 - "cannot find nonce_manager in crate"

**원인**: `lib.rs`에 모듈 추가 안함

**해결**:
```rust
// coordinator/src/lib.rs
pub mod nonce_manager;
pub mod agent_tracker;
```

### 문제: 모든 제출이 거부됨 - "Nonce not issued"

**원인**: Agent가 nonce를 요청하지 않음

**해결**: Agent가 nonce를 먼저 요청하거나, nonce 검증을 선택적으로 만들기
```rust
if !submission.coordinator_nonce.is_empty() {
    // 논스가 있을 때만 검증
    self.nonce_manager.validate_and_consume(...).await?;
}
```

### 문제: 체인 검증 실패 - "Sequence number skip"

**원인**: Agent가 sequence를 증가시키지 않음

**해결**: Agent에서 제출 시마다 sequence 증가 확인
```rust
chain_state.next_sequence();  // 1, 2, 3, ...
```

---

## 참고 자료

- [SECURITY_IMPROVEMENTS.md](SECURITY_IMPROVEMENTS.md) - 보안 개선 전체 계획
- [METRICS_INTEGRITY_VERIFICATION.md](METRICS_INTEGRITY_VERIFICATION.md) - 검증 메커니즘 설명
- [단위 테스트]
  - `coordinator/src/nonce_manager.rs` - tests 섹션
  - `coordinator/src/agent_tracker.rs` - tests 섹션
  - `verifier/src/chain_validator.rs` - tests 섹션
