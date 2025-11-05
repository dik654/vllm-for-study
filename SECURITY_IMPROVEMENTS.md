# 보안 약점 해결 방안 및 구현

## 현재 약점 분석

### 약점 1: Agent 손상 시 서명키 탈취 가능

**문제**:
```
공격자가 LLM 서버에 침입 → Agent 프로세스 손상 → 서명키 파일 읽기
→ 조작된 메트릭으로 유효한 서명 생성 가능
```

**영향도**: 🔴 높음 (단일 서버 완전 손상)

### 약점 2: vLLM 메트릭 조작 탐지 불가

**문제**:
```
vLLM이 조작된 메트릭 생성 → Agent는 "그대로" 서명
→ Verifier는 조작 여부를 알 수 없음
```

**영향도**: 🔴 높음 (메트릭 신뢰성 핵심)

### 약점 3: 재전송 공격 (Replay Attack)

**문제**:
```
공격자가 이전 정상 메트릭 + Quote 저장
→ 나중에 동일한 메트릭 재전송
→ Verifier는 정상으로 인식
```

**영향도**: 🟡 중간 (동일 메트릭 중복 제출)

### 약점 4: 메트릭 순서 조작

**문제**:
```
공격자가 메트릭 제출 순서 변경
→ 특정 메트릭 누락 또는 순서 뒤바꿈
→ 전체 사용량 추적 불가
```

**영향도**: 🟡 중간 (감사 추적 어려움)

---

## 해결 방안

### 🛡️ 방안 1: 논스 기반 재전송 공격 방지

**아이디어**: Coordinator가 각 메트릭 제출 요청마다 고유한 논스(nonce) 발급

**구현**:

```
┌─────────────────────────────────────────────────────────┐
│ 1. Agent → Coordinator: "메트릭 제출하고 싶어요"        │
│ 2. Coordinator → Agent: nonce=0x1234abcd (일회용)      │
│ 3. Agent: Quote = Sign(metrics || nonce)               │
│ 4. Agent → Coordinator: (metrics, quote, nonce)        │
│ 5. Coordinator: nonce 사용 여부 확인                    │
│    - 이미 사용됨 → 재전송 공격! 거부                   │
│    - 미사용 → 검증 진행                                │
│ 6. Coordinator: nonce를 "사용됨" 목록에 추가           │
└─────────────────────────────────────────────────────────┘
```

**효과**:
- ✅ 재전송 공격 완벽 차단
- ✅ Coordinator가 중앙에서 논스 관리
- ✅ 논스 재사용 불가

**구현 난이도**: ⭐⭐☆☆☆

### 🛡️ 방안 2: 메트릭 시퀀스 체인

**아이디어**: 각 메트릭에 시퀀스 번호와 이전 메트릭 해시 포함 (블록체인 방식)

**구현**:

```
Request 1:
  sequence: 1
  prev_hash: 0x0000...0000 (genesis)
  metrics: {tokens: 100, cost: 0.01}
  current_hash: SHA256(sequence || prev_hash || metrics)
             = 0xabcd...

Request 2:
  sequence: 2
  prev_hash: 0xabcd... (Request 1의 해시)
  metrics: {tokens: 200, cost: 0.02}
  current_hash: SHA256(2 || 0xabcd || metrics)
             = 0x1234...

Request 3:
  sequence: 3
  prev_hash: 0x1234... (Request 2의 해시)
  ...
```

**Verifier 검증**:
```rust
// 1. 시퀀스 번호 확인
if metrics.sequence != expected_sequence {
    return Fail("순서 어긋남");
}

// 2. 이전 해시 확인
if metrics.prev_hash != stored_previous_hash {
    return Fail("체인 끊김");
}

// 3. 현재 해시 검증
let calculated_hash = SHA256(metrics.sequence || metrics.prev_hash || metrics.data);
if calculated_hash != metrics.current_hash {
    return Fail("해시 불일치");
}
```

**공격 시나리오**:
```
공격자가 Request 2를 건너뛰고 Request 3 제출 시도:
  sequence: 3
  prev_hash: 0xabcd... (Request 1의 해시) ❌

Verifier:
  expected prev_hash: 0x1234... (Request 2)
  received prev_hash: 0xabcd...
  → 불일치! Request 2 누락 탐지 ✅
```

**효과**:
- ✅ 메트릭 누락 탐지
- ✅ 순서 조작 탐지
- ✅ 감사 추적 완전성 보장

**구현 난이도**: ⭐⭐⭐☆☆

### 🛡️ 방안 3: 타임스탬프 윈도우 검증 강화

**아이디어**: 타임스탬프를 현재 시간과 비교하여 너무 오래되거나 미래인 메트릭 거부

**구현**:

```rust
// 현재 구현 (약함)
if (quote.timestamp - now).abs() > 3600 {  // ±1시간
    return Fail("타임스탬프 범위 벗어남");
}

// 개선된 구현 (강함)
const TIMESTAMP_TOLERANCE_PAST: u64 = 60;    // 과거 60초 허용
const TIMESTAMP_TOLERANCE_FUTURE: u64 = 10;  // 미래 10초 허용

let now = current_timestamp();
let age = now - quote.timestamp;

if age > TIMESTAMP_TOLERANCE_PAST {
    return Fail("타임스탬프 너무 오래됨");
}

if age < -(TIMESTAMP_TOLERANCE_FUTURE as i64) {
    return Fail("타임스탬프 미래 시간");
}
```

**효과**:
- ✅ 오래된 메트릭 재전송 차단
- ✅ 시간 동기화 문제 탐지
- ✅ 미래 타임스탬프 공격 방지

**구현 난이도**: ⭐☆☆☆☆

### 🛡️ 방안 4: Agent별 메트릭 카운터 추적

**아이디어**: Coordinator가 각 Agent의 총 메트릭 수, 토큰 수 등 누적 추적

**구현**:

```rust
// Coordinator가 Agent별 상태 저장
struct AgentState {
    agent_id: String,
    total_requests: u64,       // 총 요청 수
    total_tokens: u64,         // 총 토큰 수
    total_cost: f64,           // 총 비용
    last_submission: u64,      // 마지막 제출 시간
    hourly_requests: u64,      // 시간당 요청 수
}

// 새 메트릭 제출 시 검증
fn validate_agent_metrics(&self, agent_id: &str, metrics: &RequestMetrics)
    -> Result<()> {

    let state = self.get_agent_state(agent_id)?;

    // 1. 요청 빈도 확인 (DDoS 방지)
    let time_since_last = current_timestamp() - state.last_submission;
    if time_since_last < 10 {  // 10초 이내 재제출
        return Err("제출 너무 빈번");
    }

    // 2. 시간당 요청 수 확인
    if state.hourly_requests > 1000 {
        return Err("시간당 요청 한도 초과");
    }

    // 3. 비정상 패턴 탐지
    let avg_cost = state.total_cost / state.total_requests as f64;
    if metrics.estimated_cost > avg_cost * 10.0 {
        warn!("비정상적으로 높은 비용 탐지");
        // 추가 검증 요청 또는 경고
    }

    Ok(())
}
```

**효과**:
- ✅ DDoS 공격 방지
- ✅ 비정상 패턴 탐지
- ✅ Agent별 사용량 모니터링

**구현 난이도**: ⭐⭐⭐☆☆

### 🛡️ 방안 5: 서명키 암호화 저장

**아이디어**: Agent의 서명키를 평문이 아닌 암호화하여 저장

**현재 방식 (취약)**:
```rust
// agent_config.json
{
    "signing_key_hex": "0123456789abcdef..."  // 평문 저장 ❌
}
```

**개선 방식**:
```rust
// 1. 서명키를 OS keyring에 저장 (Linux: libsecret, macOS: Keychain)
use keyring::Entry;

let entry = Entry::new("bft-agent", &agent_id)?;
entry.set_password(&signing_key_hex)?;

// 2. 또는 환경 변수로만 전달 (파일에 저장 안함)
let signing_key = env::var("BFT_AGENT_SIGNING_KEY")?;

// 3. 또는 LUKS 암호화 파티션에 저장
// /etc/bft-agent/keys/agent-1.key (LUKS 볼륨)
```

**효과**:
- ✅ 파일 탈취만으로 키 획득 불가
- ✅ OS 레벨 보호
- ⚠️ Agent 프로세스 메모리 덤프는 여전히 가능

**구현 난이도**: ⭐⭐☆☆☆

### 🛡️ 방안 6: 메트릭 스냅샷 비교 (향후)

**아이디어**: vLLM 내부와 Agent 양쪽에서 독립적으로 메트릭 수집, 비교

**구현**:

```
vLLM (Python):
┌────────────────────────────────┐
│ LLM 추론 완료                  │
│   ↓                            │
│ Metrics Collector 1 (vLLM)    │
│   • prompt_tokens: 1000       │
│   • completion_tokens: 500    │
│   ↓ 파일 저장                  │
│ /var/log/vllm/metrics.jsonl   │
└────────────────────────────────┘

Agent (Rust):
┌────────────────────────────────┐
│ Metrics Collector 2 (독립)    │
│   • vLLM API 호출 모니터링    │
│   • 토큰 수 재계산             │
│   ↓                            │
│ 비교: vLLM vs Agent           │
│   abs(1000 - 1005) < 10? ✅   │
│   ↓                            │
│ 일치 → 서명 및 제출            │
│ 불일치 → 경고 및 거부          │
└────────────────────────────────┘
```

**효과**:
- ✅ vLLM 메트릭 조작 탐지
- ✅ 이중 검증
- ⚠️ 구현 복잡도 매우 높음

**구현 난이도**: ⭐⭐⭐⭐⭐

---

## 구현 우선순위

### 즉시 구현 (Phase 1-A)

| 방안 | 난이도 | 효과 | 우선순위 |
|------|--------|------|----------|
| 타임스탬프 윈도우 강화 | ⭐☆☆☆☆ | 🟢 중 | ⬆️ 높음 |
| Agent별 카운터 추적 | ⭐⭐⭐☆☆ | 🟢 중 | ⬆️ 높음 |
| 논스 기반 재전송 방지 | ⭐⭐☆☆☆ | 🟢 높음 | ⬆️ 높음 |

### 단기 구현 (Phase 1-B)

| 방안 | 난이도 | 효과 | 우선순위 |
|------|--------|------|----------|
| 메트릭 시퀀스 체인 | ⭐⭐⭐☆☆ | 🟢 높음 | ⬆️ 중간 |
| 서명키 암호화 저장 | ⭐⭐☆☆☆ | 🟡 중 | ➡️ 중간 |

### 중장기 구현 (Phase 2)

| 방안 | 난이도 | 효과 | 우선순위 |
|------|--------|------|----------|
| 실제 TPM 통합 | ⭐⭐⭐⭐☆ | 🟢 매우 높음 | ⬆️ 높음 |
| 메트릭 스냅샷 비교 | ⭐⭐⭐⭐⭐ | 🟢 높음 | ⬇️ 낮음 |

---

## 구현 계획

### Step 1: 타임스탬프 검증 강화 ✅

**파일**: `bft_verifier_poc/verifier/src/validator.rs`

```rust
// Before
const TIMESTAMP_TOLERANCE: u64 = 3600;  // ±1시간

// After
const TIMESTAMP_TOLERANCE_PAST: u64 = 60;    // 60초 전까지
const TIMESTAMP_TOLERANCE_FUTURE: u64 = 10;  // 10초 후까지
```

### Step 2: 논스 시스템 구현 ✅

**새 파일**: `bft_verifier_poc/coordinator/src/nonce_manager.rs`

```rust
pub struct NonceManager {
    used_nonces: Arc<RwLock<HashSet<String>>>,
    expiry_time: u64,  // 논스 유효 기간 (초)
}

impl NonceManager {
    pub fn generate_nonce(&self) -> String {
        // UUID v4 생성
        uuid::Uuid::new_v4().to_string()
    }

    pub fn validate_and_consume(&self, nonce: &str) -> Result<()> {
        let mut used = self.used_nonces.write().await;

        if used.contains(nonce) {
            return Err("Nonce already used");
        }

        used.insert(nonce.to_string());
        Ok(())
    }
}
```

**프로토콜 수정**: `bft_verifier_poc/proto/bft_verifier.proto`

```protobuf
message MetricSubmission {
  string agent_id = 1;
  string request_id = 2;
  uint64 timestamp = 3;
  RequestMetrics metrics = 4;
  TpmQuote quote = 5;
  bool async_mode = 6;
  string nonce = 7;  // NEW: 논스 추가
}
```

### Step 3: 메트릭 시퀀스 체인 구현 ✅

**프로토콜 수정**:

```protobuf
message RequestMetrics {
  int32 prompt_tokens = 1;
  int32 completion_tokens = 2;
  int32 cached_tokens = 3;
  uint64 e2e_latency_ms = 4;
  uint64 time_to_first_token_ms = 5;
  double estimated_cost = 6;

  // NEW: 체인 필드
  uint64 sequence = 7;        // 시퀀스 번호
  bytes prev_hash = 8;        // 이전 메트릭 해시
  bytes current_hash = 9;     // 현재 메트릭 해시
}
```

**새 파일**: `bft_verifier_poc/verifier/src/chain_validator.rs`

```rust
pub struct ChainValidator {
    agent_chains: Arc<RwLock<HashMap<String, AgentChainState>>>,
}

struct AgentChainState {
    last_sequence: u64,
    last_hash: Vec<u8>,
}

impl ChainValidator {
    pub fn validate_chain(&self, agent_id: &str, metrics: &RequestMetrics)
        -> Result<()> {

        let chains = self.agent_chains.read().await;
        let state = chains.get(agent_id);

        match state {
            None => {
                // 첫 메트릭 (genesis)
                if metrics.sequence != 1 {
                    return Err("First sequence must be 1");
                }
                if !metrics.prev_hash.is_empty() {
                    return Err("First prev_hash must be empty");
                }
            }
            Some(state) => {
                // 시퀀스 확인
                if metrics.sequence != state.last_sequence + 1 {
                    return Err("Sequence number skip detected");
                }

                // 이전 해시 확인
                if metrics.prev_hash != state.last_hash {
                    return Err("Chain broken: prev_hash mismatch");
                }
            }
        }

        // 현재 해시 검증
        let calculated_hash = calculate_metrics_hash(metrics);
        if metrics.current_hash != calculated_hash {
            return Err("Current hash mismatch");
        }

        Ok(())
    }
}
```

### Step 4: Agent별 상태 추적 ✅

**새 파일**: `bft_verifier_poc/coordinator/src/agent_tracker.rs`

```rust
pub struct AgentTracker {
    states: Arc<RwLock<HashMap<String, AgentState>>>,
}

struct AgentState {
    total_requests: u64,
    total_tokens: u64,
    total_cost: f64,
    last_submission: u64,
    hourly_counter: HourlyCounter,
}

impl AgentTracker {
    pub fn validate_submission_rate(&self, agent_id: &str) -> Result<()> {
        let states = self.states.read().await;
        let state = states.get(agent_id)?;

        // 최소 제출 간격 (DDoS 방지)
        let time_since_last = current_timestamp() - state.last_submission;
        if time_since_last < 10 {
            return Err("Submission too frequent");
        }

        // 시간당 요청 수 제한
        if state.hourly_counter.count() > 1000 {
            return Err("Hourly request limit exceeded");
        }

        Ok(())
    }
}
```

---

## 보안 강화 후 시스템 구조

```
┌──────────────────────────────────────────────────────────┐
│              개선된 보안 메커니즘                         │
├──────────────────────────────────────────────────────────┤
│                                                            │
│ 1️⃣ 논스 시스템                                           │
│    Coordinator → Agent: nonce=xyz                        │
│    Agent → Coordinator: Sign(metrics || nonce)           │
│    ✅ 재전송 공격 차단                                    │
│                                                            │
│ 2️⃣ 메트릭 체인                                           │
│    Req1 → hash1                                          │
│    Req2 → hash2 = SHA256(Req2 || hash1)                 │
│    Req3 → hash3 = SHA256(Req3 || hash2)                 │
│    ✅ 누락/순서 조작 탐지                                │
│                                                            │
│ 3️⃣ 타임스탬프 강화                                       │
│    과거: 60초, 미래: 10초 허용                           │
│    ✅ 오래된 메트릭 차단                                 │
│                                                            │
│ 4️⃣ Agent 추적                                            │
│    요청 빈도, 토큰 수, 비용 모니터링                     │
│    ✅ 비정상 패턴 탐지                                   │
│                                                            │
│ 5️⃣ BFT 합의 (기존)                                      │
│    2f+1 정족수                                           │
│    ✅ 악의적 검증자 배제                                 │
└──────────────────────────────────────────────────────────┘
```

---

## 예상 보안 개선 효과

| 공격 유형 | 개선 전 | 개선 후 |
|-----------|---------|---------|
| **재전송 공격** | ❌ 취약 | ✅ 차단 (논스) |
| **메트릭 누락** | ❌ 탐지 불가 | ✅ 탐지 (체인) |
| **순서 조작** | ❌ 탐지 불가 | ✅ 탐지 (체인) |
| **DDoS** | ⚠️ 부분 차단 | ✅ 차단 (Rate Limit) |
| **오래된 메트릭** | ⚠️ 1시간 허용 | ✅ 60초만 허용 |
| **서명키 탈취** | ❌ 취약 | ⚠️ 여전히 취약* |
| **vLLM 조작** | ❌ 탐지 불가 | ⚠️ 여전히 불가* |

*향후 TPM 통합으로 해결 예정

---

## 구현 체크리스트

### Phase 1-A (즉시 구현) ✅

- [ ] 타임스탬프 검증 강화
  - [ ] `TIMESTAMP_TOLERANCE_PAST` 추가
  - [ ] `TIMESTAMP_TOLERANCE_FUTURE` 추가
  - [ ] Verifier 검증 로직 수정

- [ ] 논스 시스템
  - [ ] `NonceManager` 구현
  - [ ] 프로토콜에 nonce 필드 추가
  - [ ] Coordinator에 논스 발급 로직
  - [ ] Verifier에 논스 검증 로직
  - [ ] 만료된 논스 자동 정리

- [ ] Agent 상태 추적
  - [ ] `AgentTracker` 구현
  - [ ] 요청 빈도 제한
  - [ ] 시간당 요청 수 제한
  - [ ] 비정상 패턴 탐지

### Phase 1-B (단기 구현)

- [ ] 메트릭 시퀀스 체인
  - [ ] 프로토콜에 sequence, prev_hash, current_hash 추가
  - [ ] `ChainValidator` 구현
  - [ ] Agent에 체인 생성 로직
  - [ ] Verifier에 체인 검증 로직
  - [ ] Coordinator에 체인 상태 저장

- [ ] 서명키 보호
  - [ ] OS keyring 통합
  - [ ] 환경 변수 전용 모드
  - [ ] 암호화 저장 옵션

### Phase 2 (중장기)

- [ ] 실제 TPM 통합
  - [ ] tpm2-tss 크레이트 통합
  - [ ] TPM 2.0 Quote 생성
  - [ ] Verifier TPM 검증

- [ ] Keylime 통합 (선택)
  - [ ] Keylime Agent 설치
  - [ ] IMA 측정 활성화
  - [ ] Keylime Verifier 설정

---

## 테스트 계획

### 단위 테스트

```rust
#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_nonce_reuse_prevention() {
        let manager = NonceManager::new();
        let nonce = manager.generate_nonce();

        // 첫 사용: 성공
        assert!(manager.validate_and_consume(&nonce).await.is_ok());

        // 재사용: 실패
        assert!(manager.validate_and_consume(&nonce).await.is_err());
    }

    #[tokio::test]
    async fn test_chain_sequence_validation() {
        let validator = ChainValidator::new();

        // Sequence 1: 성공
        let metrics1 = create_metrics(1, vec![], hash1);
        assert!(validator.validate_chain("agent-1", &metrics1).await.is_ok());

        // Sequence 3 (2 건너뜀): 실패
        let metrics3 = create_metrics(3, hash1, hash3);
        assert!(validator.validate_chain("agent-1", &metrics3).await.is_err());
    }
}
```

### 통합 테스트

```bash
# 재전송 공격 테스트
./test_replay_attack.sh

# 메트릭 누락 테스트
./test_sequence_skip.sh

# Rate limit 테스트
./test_rate_limit.sh
```

---

## 마이그레이션 가이드

### 기존 시스템에서 업그레이드

1. **프로토콜 업데이트**
   ```bash
   cd bft_verifier_poc
   # proto 파일 수정 후
   cargo build
   ```

2. **Coordinator 업그레이드**
   ```bash
   # 기존 Coordinator 중지
   systemctl stop bft-coordinator

   # 새 버전 배포
   cp target/release/coordinator /usr/local/bin/

   # 재시작
   systemctl start bft-coordinator
   ```

3. **Verifier 순차 업그레이드**
   ```bash
   # 한 번에 한 개씩 업그레이드 (무중단 배포)
   for verifier in verifier-{1..4}; do
       systemctl stop $verifier
       cp target/release/verifier /usr/local/bin/
       systemctl start $verifier
       sleep 10  # 안정화 대기
   done
   ```

4. **Agent 업그레이드**
   ```bash
   # 각 LLM 서버에서
   systemctl stop bft-agent
   cp target/release/agent /usr/local/bin/
   systemctl start bft-agent
   ```

### 하위 호환성

- ✅ nonce 필드는 선택적 (기본값: 빈 문자열)
- ✅ sequence 필드는 선택적 (기본값: 0)
- ✅ 기존 메트릭도 검증 가능 (체인 미사용 모드)

---

## 모니터링 지표

### 보안 관련 메트릭

```
bft_nonce_reuse_attempts_total       # 논스 재사용 시도 횟수
bft_chain_break_detected_total       # 체인 끊김 탐지 횟수
bft_timestamp_violations_total       # 타임스탬프 위반 횟수
bft_rate_limit_exceeded_total        # Rate limit 초과 횟수
bft_agent_total_requests_total       # Agent별 총 요청 수
bft_agent_total_tokens_total         # Agent별 총 토큰 수
```

### Grafana 대시보드

```
┌────────────────────────────────────────┐
│ 보안 이벤트 대시보드                   │
├────────────────────────────────────────┤
│ 📊 논스 재사용 시도 (최근 1시간)      │
│    ▁▂▁▁▃▁▁▁▁▁ → 정상                 │
│                                         │
│ 📊 체인 끊김 탐지 (최근 1시간)        │
│    ▁▁▁▁▁▁▁▁▁▁ → 정상                 │
│                                         │
│ 📊 Agent 요청 빈도 (최근 5분)         │
│    agent-1: 45 req/min                 │
│    agent-2: 52 req/min                 │
│    agent-3: 38 req/min                 │
└────────────────────────────────────────┘
```

---

## 참고 자료

- [Replay Attack Prevention](https://en.wikipedia.org/wiki/Replay_attack)
- [Blockchain Basics](https://en.wikipedia.org/wiki/Blockchain)
- [Nonce (Cryptography)](https://en.wikipedia.org/wiki/Cryptographic_nonce)
- [Rate Limiting Strategies](https://cloud.google.com/architecture/rate-limiting-strategies-techniques)
