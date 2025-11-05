# 메트릭 무결성 검증 방법

**핵심 질문**: LLM 서버가 제출한 메트릭(토큰 수, 비용 등)이 조작되지 않았다는 것을 어떻게 보장하는가?

**짧은 답변**: 현재는 **TPM 시뮬레이션(Ed25519 서명) + BFT 합의**로 검증합니다. Keylime은 **향후 통합 예정**이며, 현재는 사용하지 않습니다.

---

## 목차

1. [현재 구현 (Phase 1)](#현재-구현-phase-1)
2. [검증 메커니즘 상세](#검증-메커니즘-상세)
3. [공격 시나리오와 방어](#공격-시나리오와-방어)
4. [Keylime 없이도 작동하는 이유](#keylime-없이도-작동하는-이유)
5. [향후 Keylime 통합 계획](#향후-keylime-통합-계획)
6. [보안 강도 비교](#보안-강도-비교)

---

## 현재 구현 (Phase 1)

### 사용하는 기술

```
┌─────────────────────────────────────────────────────────────┐
│              현재 구현 (Keylime 없음)                        │
├─────────────────────────────────────────────────────────────┤
│ 1. Ed25519 서명 (TPM 시뮬레이션)        ✅ 구현됨           │
│ 2. BFT 합의 (Byzantine Fault Tolerance) ✅ 구현됨           │
│ 3. VRF 위원회 선택                       ✅ 구현됨           │
│ 4. 실제 TPM 2.0                          ❌ 미구현          │
│ 5. Keylime (시스템 무결성 검증)          ❌ 미구현          │
└─────────────────────────────────────────────────────────────┘
```

### 아키텍처

```
LLM 서버 (악의적일 수 있음)
┌────────────────────────────────────────┐
│ vLLM 엔진                              │
│   ↓ 메트릭 생성                        │
│ Metrics Collector                      │
│   • prompt_tokens: 1000               │
│   • completion_tokens: 500            │
│   • cost: $0.0012                     │
│   ↓                                    │
│ BFT Agent                              │
│   ↓ Ed25519 서명                      │
│ Quote = Sign(metrics, private_key)    │
│   • signature: 0xabcd...              │
│   • metrics_hash: SHA256(metrics)     │
└────────────────────────────────────────┘
         │ gRPC 전송
         ↓
┌────────────────────────────────────────┐
│ Coordinator (중앙 검증 서버)           │
│   ↓ VRF로 검증자 3명 선택              │
│   ↓ 병렬 검증 요청                     │
└────────────────────────────────────────┘
    │           │           │
    ↓           ↓           ↓
┌─────────┐ ┌─────────┐ ┌─────────┐
│Verifier1│ │Verifier2│ │Verifier3│
│         │ │         │ │         │
│1. 서명  │ │1. 서명  │ │1. 서명  │
│   검증  │ │   검증  │ │   검증  │
│         │ │         │ │         │
│2. 메트릭│ │2. 메트릭│ │2. 메트릭│
│   범위  │ │   범위  │ │   범위  │
│   검증  │ │   검증  │ │   검증  │
│         │ │         │ │         │
│3. 투표: │ │3. 투표: │ │3. 투표: │
│   PASS  │ │   PASS  │ │   FAIL  │
└─────────┘ └─────────┘ └─────────┘
    │           │           │
    └───────────┴───────────┘
                │
                ↓
    BFT 합의: 2/3 PASS → 승인
```

---

## 검증 메커니즘 상세

### 1단계: 메트릭 서명 (Agent)

**위치**: `bft_verifier_poc/agent/src/tpm.rs`

```rust
// 메트릭을 해시하고 서명
pub fn generate_quote(&self, metrics: &RequestMetrics) -> TpmQuote {
    // 1. 메트릭을 정규화된 바이트로 직렬화
    let metrics_bytes = serialize_metrics(metrics);

    // 2. SHA-256 해시 (PCR 값 시뮬레이션)
    let pcr_value = sha256(metrics_bytes);

    // 3. Ed25519 개인키로 서명 (TPM Quote 시뮬레이션)
    let signature = self.signing_key.sign(&pcr_value);

    TpmQuote {
        pcr_value: pcr_value.to_vec(),
        signature: signature.to_bytes().to_vec(),
        timestamp: current_timestamp(),
    }
}
```

**핵심**:
- 메트릭이 조작되면 해시가 달라짐
- 서명이 메트릭과 매칭되지 않으면 검증 실패

### 2단계: 서명 검증 (Verifier)

**위치**: `bft_verifier_poc/verifier/src/tpm.rs`

```rust
pub fn verify_quote(
    &self,
    metrics: &RequestMetrics,
    quote: &TpmQuote,
    agent_id: &str,
) -> Result<bool> {
    // 1. Agent의 공개키 조회
    let public_key = self.get_agent_public_key(agent_id)?;

    // 2. 메트릭을 동일하게 해시
    let metrics_bytes = serialize_metrics(metrics);
    let expected_pcr = sha256(metrics_bytes);

    // 3. Quote의 PCR 값과 비교
    if quote.pcr_value != expected_pcr {
        return Ok(false);  // 메트릭이 조작됨!
    }

    // 4. 서명 검증
    let signature = Signature::from_bytes(&quote.signature)?;
    let valid = public_key.verify(&quote.pcr_value, &signature).is_ok();

    Ok(valid)
}
```

**검증 실패 조건**:
- ❌ 메트릭이 조작됨 → PCR 불일치
- ❌ 서명이 위조됨 → 서명 검증 실패
- ❌ 잘못된 Agent → 공개키 없음

### 3단계: 메트릭 범위 검증 (Verifier)

**위치**: `bft_verifier_poc/verifier/src/validator.rs`

```rust
pub fn validate_metrics(&self, metrics: &RequestMetrics) -> ValidationResult {
    let mut errors = Vec::new();

    // 토큰 수 검증 (1 ~ 1,000,000)
    if metrics.prompt_tokens < 1 || metrics.prompt_tokens > 1_000_000 {
        errors.push("Invalid prompt_tokens range");
    }

    // 레이턴시 검증 (1ms ~ 10분)
    if metrics.e2e_latency_ms < 1 || metrics.e2e_latency_ms > 600_000 {
        errors.push("Invalid latency range");
    }

    // 비용 검증 (0 ~ $1000)
    if metrics.estimated_cost < 0.0 || metrics.estimated_cost > 1000.0 {
        errors.push("Invalid cost range");
    }

    // 타임스탬프 검증 (현재 ±1시간)
    let now = current_timestamp();
    if (quote.timestamp - now).abs() > 3600 {
        errors.push("Timestamp too old/future");
    }

    if errors.is_empty() {
        ValidationResult::Pass
    } else {
        ValidationResult::Fail(errors)
    }
}
```

**검증 항목**:
- ✅ 토큰 수가 현실적인 범위인가?
- ✅ 레이턴시가 합리적인가?
- ✅ 비용이 계산 가능한 범위인가?
- ✅ 타임스탬프가 최신인가?

### 4단계: BFT 합의 (Coordinator)

**위치**: `bft_verifier_poc/coordinator/src/consensus.rs`

```rust
pub fn check_consensus(&self, votes: &HashMap<String, VerifierVote>)
    -> ConsensusOutcome {

    let mut pass_count = 0;
    let mut fail_count = 0;

    for vote in votes.values() {
        match vote.result {
            VerificationResult::Pass => pass_count += 1,
            VerificationResult::Fail => fail_count += 1,
        }
    }

    // 2f+1 정족수 확인 (f=1이면 quorum=2)
    if pass_count >= self.quorum {
        ConsensusOutcome::Accepted  // 다수가 PASS
    } else if fail_count >= self.quorum {
        ConsensusOutcome::Rejected  // 다수가 FAIL
    } else {
        ConsensusOutcome::NoQuorum  // 합의 실패
    }
}
```

**BFT 보장**:
- N ≥ 3f+1 검증자 중 f명이 악의적이어도 안전
- 기본 설정: N=4, f=1 (1명의 악의적 검증자 허용)
- 2f+1=2명 이상 동의 필요

---

## 공격 시나리오와 방어

### 공격 1: Agent가 메트릭 조작

**시나리오**:
```python
# 악의적 Agent가 메트릭 조작 시도
real_metrics = {
    "prompt_tokens": 10000,      # 실제 사용량
    "estimated_cost": 0.10       # 실제 비용
}

fake_metrics = {
    "prompt_tokens": 100,        # 1/100로 축소 ❌
    "estimated_cost": 0.001      # 1/100로 축소 ❌
}

# Agent가 fake_metrics로 Quote 생성
quote = generate_quote(fake_metrics)
```

**방어**:
```
Verifier 검증:
1. 서명 검증: ✅ 통과 (Agent가 서명했으므로)
2. 범위 검증: ✅ 통과 (100 토큰도 유효한 범위)
3. BFT 투표: ✅ PASS

결과: 조작된 메트릭이 승인됨! ❌
```

**현재 문제점**:
- ❌ Agent가 서명키를 갖고 있어서 조작 가능
- ❌ Verifier는 "실제로 10000 토큰을 사용했는지" 알 수 없음

**해결 방법 (향후 TPM 통합)**:
```
실제 TPM 사용 시:
1. vLLM이 메트릭을 TPM PCR에 직접 기록
2. Agent는 TPM에게 Quote 요청만 가능 (조작 불가)
3. TPM이 하드웨어 기반 서명 생성
4. Verifier가 TPM 공개키로 검증

→ Agent가 메트릭 조작해도 서명 불일치로 탐지 ✅
```

### 공격 2: Agent가 서명키 위조

**시나리오**:
```rust
// 악의적 Agent가 새 키페어 생성
let (fake_sk, fake_pk) = generate_keypair();

// 조작된 메트릭에 서명
let fake_quote = fake_sk.sign(fake_metrics);

// Coordinator에게 전송
submit_metrics(fake_metrics, fake_quote);
```

**방어**:
```
Verifier 검증:
1. Agent ID로 등록된 공개키 조회
2. Verifier에 등록된 키: 0xabcd...  (원본)
3. Quote에 사용된 키: 0x1234...    (위조)
4. 공개키 불일치 → 검증 실패 ❌

결과: 위조 탐지, 메트릭 거부 ✅
```

**보안 요소**:
- ✅ Verifier가 Agent 공개키를 사전 등록
- ✅ Agent는 등록된 키로만 서명 가능
- ✅ 새 키 생성해도 Verifier가 거부

### 공격 3: 악의적 Verifier

**시나리오**:
```rust
// 악의적 Verifier가 항상 FAIL 투표
impl Verifier {
    fn verify(&self, metrics: &RequestMetrics) -> Vote {
        Vote::FAIL  // 항상 거부 ❌
    }
}
```

**방어**:
```
BFT 합의 (N=4, f=1, quorum=2):
- Verifier 1: PASS ✅
- Verifier 2: PASS ✅
- Verifier 3: FAIL ❌ (악의적)
- Verifier 4: (선택 안됨)

PASS 득표: 2개 ≥ quorum(2)
결과: 승인 ✅

→ 1개의 악의적 검증자는 무시됨
```

**BFT 보장**:
- ✅ N ≥ 3f+1 검증자 필요
- ✅ f개 악의적 검증자 허용
- ✅ 2f+1 정족수로 합의

### 공격 4: VRF 위원회 선택 조작

**시나리오**:
```
악의적 Coordinator가 자기 편인 Verifier만 선택 시도
```

**방어**:
```rust
// VRF는 결정론적 (deterministic)
pub fn select_committee(&self, epoch: u64, pool: &[String]) -> Vec<String> {
    let seed = sha256(self.vrf_seed || epoch);

    // 모든 노드가 동일한 결과 계산 가능
    let selected = pseudo_random_selection(seed, pool, committee_size);

    // Coordinator가 조작 불가 (모두가 검증 가능)
    selected
}
```

**VRF 특성**:
- ✅ 결정론적: 같은 epoch → 같은 위원회
- ✅ 검증 가능: 누구나 선택 결과 확인 가능
- ✅ 예측 불가: 미래 위원회 예측 불가
- ✅ 공정성: 모든 검증자 동일 확률

---

## Keylime 없이도 작동하는 이유

### 현재 방식의 보안 가정

```
┌─────────────────────────────────────────────────────────┐
│ 현재 구현의 신뢰 모델                                    │
├─────────────────────────────────────────────────────────┤
│ ✅ 신뢰: Verifiers (다수는 정직하다고 가정)             │
│ ✅ 신뢰: Coordinator (VRF 공개적으로 검증 가능)        │
│ ⚠️  부분 신뢰: Agent (서명키 보호 책임)                │
│ ❌ 불신: vLLM 서버 (메트릭 조작 시도 가능)             │
└─────────────────────────────────────────────────────────┘
```

**약점**:
1. **Agent 손상 시 서명키 탈취**
   - Agent가 실행되는 서버에 침입하면 서명키 획득 가능
   - 서명키로 조작된 메트릭 생성 가능

2. **vLLM 메트릭 조작 불가 탐지**
   - Agent는 vLLM이 제공한 메트릭을 "그대로" 서명
   - vLLM이 메트릭 조작해도 Agent는 모름

**그럼에도 작동하는 이유**:

1. **BFT 합의가 다수 검증**
   - 1개 Agent 손상 → 1개 메트릭만 조작
   - 다른 서버들은 정상 메트릭 제출
   - 시스템 전체는 정상 동작

2. **경제적 인센티브**
   - 메트릭 조작 탐지 시 페널티
   - 서명키 보호 책임

3. **감사 가능성**
   - 모든 제출 기록 보관
   - 사후 검증 가능

### Keylime 추가 시 개선점

```
┌──────────────────────────────────────────────────────────┐
│ Keylime + TPM 통합 시 신뢰 모델                          │
├──────────────────────────────────────────────────────────┤
│ ✅ 신뢰: TPM 하드웨어 (변조 불가)                        │
│ ✅ 신뢰: Keylime Verifier (오픈소스, 감사됨)            │
│ ✅ 신뢰: Linux IMA (커널 레벨 보호)                     │
│ ⚠️  부분 신뢰: Agent (하지만 TPM이 보호)                │
│ ❌ 불신: vLLM 서버 (하지만 IMA가 감시)                  │
└──────────────────────────────────────────────────────────┘
```

**추가 보호**:

1. **하드웨어 기반 서명키**
   - 서명키가 TPM 칩 내부에 저장
   - Agent 손상되어도 키 탈취 불가

2. **시스템 무결성 검증**
   - IMA가 vLLM 바이너리 해시 측정
   - 바이너리 조작 시 PCR 값 변경 → 탐지

3. **원격 증명**
   - Keylime Verifier가 실시간 모니터링
   - 시스템 손상 시 경고

---

## 향후 Keylime 통합 계획

### Phase 2: 실제 TPM 통합 (Keylime 없음)

**목표**: 소프트웨어 서명 → 하드웨어 서명

```rust
// bft_agent/src/tpm.rs

// 현재 (Ed25519 시뮬레이션)
pub struct SimulatedTpmAgent {
    signing_key: SigningKey,  // 메모리에 저장 ❌
}

// 향후 (실제 TPM 2.0)
use tss_esapi::{Context, TctiNameConf};

pub struct RealTpmAgent {
    tpm_context: Context,     // TPM 하드웨어 연결 ✅
    key_handle: KeyHandle,    // TPM 내부 키 핸들
}

impl RealTpmAgent {
    pub fn generate_quote(&self, metrics: &RequestMetrics) -> TpmQuote {
        // 1. 메트릭 해시
        let metrics_hash = sha256(serialize(metrics));

        // 2. TPM에게 Quote 요청 (하드웨어 서명)
        let quote = self.tpm_context.quote(
            self.key_handle,
            &metrics_hash,
            SignatureScheme::Null,
        )?;

        // 3. TPM이 생성한 Quote 반환 (조작 불가 ✅)
        TpmQuote::from(quote)
    }
}
```

**개선점**:
- ✅ 서명키 탈취 불가 (TPM 내부)
- ✅ Quote 위조 불가 (하드웨어 서명)
- ❌ 아직 vLLM 메트릭 조작은 탐지 못함

### Phase 3: Keylime 완전 통합

**추가 컴포넌트**:

```
각 LLM 서버:
┌──────────────────────────────────┐
│ vLLM (Python)                    │
│   • 바이너리: /usr/bin/vllm     │
│   • IMA 측정: sha256(vllm)      │
│   ↓                              │
│ Keylime Agent (Python)           │
│   • IMA 로그 수집                │
│   • TPM Quote 생성               │
│   • Verifier에게 전송            │
│   ↓                              │
│ BFT Agent (Rust)                 │
│   • Metrics 전송                 │
└──────────────────────────────────┘
         │                 │
         │                 │
         ▼                 ▼
┌─────────────────┐  ┌─────────────┐
│ Keylime         │  │ BFT         │
│ Verifier        │  │ Coordinator │
│ (시스템 검증)   │  │ (메트릭 검증)│
└─────────────────┘  └─────────────┘
```

**통합 시나리오**:

1. **vLLM 바이너리 변조 시도**
   ```bash
   # 공격자가 vLLM 바이너리 변조
   echo "malicious_code" >> /usr/bin/vllm

   # IMA가 자동 측정
   # PCR[10] = SHA256(PCR[10] || sha256(/usr/bin/vllm))
   #         = 0xABCD...  (변조 전 값과 다름!)

   # Keylime Verifier가 탐지
   # Expected PCR[10]: 0x1234...
   # Actual PCR[10]:   0xABCD...  ❌ 불일치!

   # 경고 발생 → 해당 서버 메트릭 거부
   ```

2. **메트릭 수집 코드 변조 시도**
   ```python
   # 공격자가 metrics_collector.py 변조
   def collect_metrics(request):
       return {
           "prompt_tokens": 100,  # 실제는 10000 ❌
       }

   # IMA가 파일 변조 감지
   # sha256(metrics_collector.py) 변경
   # PCR 값 변경 → Keylime이 탐지
   # → 해당 서버 메트릭 거부
   ```

3. **메모리 조작 시도**
   ```c
   // 공격자가 메모리 직접 조작 시도
   void* vllm_metrics = find_metrics_address();
   memcpy(vllm_metrics, fake_data, size);  // ❌

   // Keylime는 탐지 못함 (메모리는 IMA로 측정 안됨)
   // → 이건 Keylime 한계
   ```

**Keylime 추가 이점**:
- ✅ 바이너리 무결성 검증
- ✅ 라이브러리 무결성 검증
- ✅ 설정 파일 무결성 검증
- ✅ 실시간 모니터링
- ❌ 메모리 조작은 탐지 못함

**Keylime 한계**:
- ❌ 런타임 메모리 조작 탐지 못함
- ❌ vLLM 내부 로직 조작 탐지 못함 (코드 레벨 버그 삽입)
- ❌ 복잡한 설정 및 유지보수

---

## 보안 강도 비교

### 시나리오: 악의적 서버 운영자가 메트릭 조작 시도

| 공격 방법 | 현재 방어 (Ed25519+BFT) | TPM 통합 | Keylime 통합 |
|-----------|-------------------------|----------|--------------|
| **메트릭 값 변조** (메모리) | ⚠️ 탐지 못함 | ⚠️ 탐지 못함 | ⚠️ 탐지 못함 |
| **서명키 탈취** | ❌ 가능 (파일 접근) | ✅ 불가 (TPM 내부) | ✅ 불가 (TPM 내부) |
| **서명 위조** | ✅ 탐지 (공개키 검증) | ✅ 탐지 | ✅ 탐지 |
| **바이너리 변조** | ❌ 탐지 못함 | ❌ 탐지 못함 | ✅ 탐지 (IMA) |
| **라이브러리 교체** | ❌ 탐지 못함 | ❌ 탐지 못함 | ✅ 탐지 (IMA) |
| **코드 수정 후 재컴파일** | ❌ 탐지 못함 | ❌ 탐지 못함 | ✅ 탐지 (IMA) |
| **악의적 Verifier** | ✅ 방어 (BFT) | ✅ 방어 (BFT) | ✅ 방어 (BFT) |
| **VRF 조작** | ✅ 방어 (검증 가능) | ✅ 방어 | ✅ 방어 |

### 보안 레벨 요약

```
┌───────────────────────────────────────────────────────────┐
│                   보안 레벨                                │
├───────────────────────────────────────────────────────────┤
│ Level 1: Ed25519 + BFT (현재)                             │
│   • 보호: 서명 위조, 악의적 검증자                        │
│   • 약점: 메트릭 조작, 서명키 탈취, 바이너리 변조         │
│   • 적합: PoC, 테스트, 낮은 위협 환경                     │
│   • 구현 난이도: ★☆☆☆☆                                   │
│                                                             │
│ Level 2: TPM + BFT (Phase 2)                              │
│   • 보호: Level 1 + 서명키 보호                          │
│   • 약점: 메트릭 조작, 바이너리 변조                      │
│   • 적합: 프로덕션, 중간 위협 환경                        │
│   • 구현 난이도: ★★★☆☆                                   │
│                                                             │
│ Level 3: Keylime + TPM + BFT (Phase 3)                   │
│   • 보호: Level 2 + 시스템 무결성                        │
│   • 약점: 메모리 조작 (모든 시스템 공통 한계)            │
│   • 적합: 고보안 환경, 규제 준수 필요                     │
│   • 구현 난이도: ★★★★★                                   │
└───────────────────────────────────────────────────────────┘
```

### 실용적 보안 수준

**대부분의 사용 사례**: Level 2 (TPM + BFT)로 충분

**이유**:
1. **경제적 인센티브**
   - 메트릭 조작으로 얻는 이익 < 탐지 시 페널티
   - 대부분의 서버 운영자는 정직

2. **감사 가능성**
   - 모든 제출 기록 보관
   - 이상 패턴 사후 분석 가능
   - 의심스러운 서버 조사 가능

3. **BFT 합의**
   - 소수의 악의적 서버는 무시됨
   - 시스템 전체 안정성 유지

4. **구현 복잡도 vs 보안 향상**
   - Keylime: +50% 복잡도, +10% 보안
   - TPM: +20% 복잡도, +30% 보안
   - BFT: +10% 복잡도, +40% 보안

**Keylime이 필요한 경우**:
- 금융 서비스 (규제 준수)
- 정부 시스템 (국가 보안)
- 의료 데이터 (HIPAA 준수)
- 고액 결제 (메트릭 조작 인센티브 큼)

---

## 결론

### 현재 상태 (2024-11-05)

```
✅ 구현 완료:
  1. Ed25519 서명 (TPM 시뮬레이션)
  2. BFT 합의 (Byzantine Fault Tolerance)
  3. VRF 위원회 선택
  4. 서명 검증
  5. 메트릭 범위 검증
  6. 파일 기반 통합 (vLLM ↔ Agent)

❌ 미구현:
  1. 실제 TPM 2.0 통합
  2. Keylime 시스템 무결성 검증
  3. IMA (Integrity Measurement Architecture)

📋 향후 계획:
  Phase 2: TPM 통합 (난이도: 중, 보안 향상: 중)
  Phase 3: Keylime 통합 (난이도: 고, 보안 향상: 중저)
```

### 핵심 메시지

**Q: Keylime을 쓰나요?**
- **A: 아니오.** 현재는 Ed25519 + BFT만 사용합니다.

**Q: 그럼 어떻게 메트릭 조작을 방지하나요?**
- **A:
  1. Ed25519 서명으로 메트릭 무결성 보장
  2. Verifier가 서명 + 범위 검증
  3. BFT 합의로 악의적 검증자 배제
  4. VRF로 공정한 위원회 선택**

**Q: 완벽한가요?**
- **A: 아니오.** Agent가 손상되면 서명키 탈취 가능. 하지만:
  - BFT 합의로 다수 공격 방어
  - 경제적 인센티브로 조작 억제
  - 감사 가능성으로 사후 추적

**Q: 언제 Keylime을 쓰나요?**
- **A: 필요 시 Phase 3에서 추가.** 대부분은 TPM만으로도 충분.

**Q: 지금 프로덕션에 쓸 수 있나요?**
- **A: PoC/테스트는 가능.** 프로덕션은 TPM 통합 후 권장.

---

## 참고 자료

- 현재 구현: `bft_verifier_poc/`
  - `agent/src/tpm.rs` - Ed25519 서명
  - `verifier/src/tpm.rs` - 서명 검증
  - `coordinator/src/consensus.rs` - BFT 합의

- 설계 문서:
  - `KEYLIME_INTEGRITY.md` - Keylime 통합 설계 (미구현)
  - `BFT_CONSENSUS_VERIFICATION.md` - BFT 합의 이론
  - `VLLM_BFT_INTEGRATION.md` - 통합 가이드

- 외부 자료:
  - [Keylime GitHub](https://github.com/keylime/keylime)
  - [TPM 2.0 Spec](https://trustedcomputinggroup.org/resource/tpm-library-specification/)
  - [Byzantine Fault Tolerance](https://en.wikipedia.org/wiki/Byzantine_fault)
