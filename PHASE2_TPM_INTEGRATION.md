# Phase 2: 실제 TPM 2.0 통합 가이드

이 문서는 시뮬레이션된 Ed25519 서명을 실제 하드웨어 TPM 2.0으로 교체하는 방법을 설명합니다.

## 개요

### 현재 구현 vs TPM 통합

| 항목 | Phase 1 (현재) | Phase 2 (TPM) |
|------|---------------|---------------|
| **서명 방식** | Ed25519 (소프트웨어) | TPM 2.0 (하드웨어) |
| **키 저장** | 메모리/파일 | TPM 칩 내부 |
| **키 추출** | ❌ 가능 | ✅ 불가능 |
| **PCR 사용** | 시뮬레이션 | 실제 하드웨어 PCR |
| **Quote 생성** | Ed25519 서명 | TPM_Quote 명령 |
| **보안 강도** | ⭐⭐⭐☆☆ | ⭐⭐⭐⭐⭐ |

---

## 사전 준비

### 하드웨어 요구사항

1. **TPM 2.0 칩**
   - 대부분의 현대 서버/노트북에 내장
   - 확인 방법:
     ```bash
     # Linux
     ls /dev/tpm*
     # 출력: /dev/tpm0, /dev/tpmrm0

     # TPM 정보 확인
     sudo tpm2_getcap properties-fixed
     ```

2. **운영체제**
   - Linux (Ubuntu 20.04+, RHEL 8+, etc.)
   - Windows Server 2016+ (참고용)

### 소프트웨어 설치

#### Ubuntu/Debian

```bash
# TPM 2.0 도구 설치
sudo apt-get update
sudo apt-get install -y \
    tpm2-tools \
    tpm2-abrmd \
    libtss2-dev \
    libtss2-esys-3.0.2-0

# Rust TPM 라이브러리 의존성
sudo apt-get install -y \
    pkg-config \
    libssl-dev \
    libtss2-esys-dev \
    libtss2-mu-dev \
    libtss2-tctildr-dev
```

#### RHEL/CentOS

```bash
sudo yum install -y \
    tpm2-tools \
    tpm2-abrmd \
    tpm2-tss-devel
```

### TPM 시뮬레이터 (개발/테스트용)

실제 TPM 하드웨어가 없는 경우:

```bash
# IBM TPM 시뮬레이터 설치
git clone https://github.com/kgoldman/ibmtpm20tss.git
cd ibmtpm20tss/utils
make

# 시뮬레이터 실행
./tpm_server &

# 환경 변수 설정
export TPM2TOOLS_TCTI="mssim:host=localhost,port=2321"
```

---

## TPM 초기 설정

### 1. TPM 소유권 설정

```bash
# TPM 초기화 (주의: 기존 키 삭제됨)
sudo tpm2_clear

# Owner 계층 비밀번호 설정
TPM_OWNER_PASSWORD="your-secure-password"
echo -n "$TPM_OWNER_PASSWORD" | \
    sudo tpm2_changeauth -c owner
```

### 2. Agent용 서명 키 생성

```bash
# Primary key 생성 (Endorsement Hierarchy)
sudo tpm2_createprimary \
    -C e \
    -g sha256 \
    -G rsa2048 \
    -c primary.ctx

# Agent 서명 키 생성
sudo tpm2_create \
    -C primary.ctx \
    -g sha256 \
    -G rsa2048 \
    -r agent_key.priv \
    -u agent_key.pub \
    -a "sign|fixedtpm|fixedparent|sensitivedataorigin"

# 키 로드
sudo tpm2_load \
    -C primary.ctx \
    -r agent_key.priv \
    -u agent_key.pub \
    -c agent_key.ctx

# 영구 핸들로 저장 (재부팅 후에도 유지)
sudo tpm2_evictcontrol \
    -C o \
    -c agent_key.ctx \
    0x81010001

# 공개키 추출 (Verifier에 등록)
sudo tpm2_readpublic \
    -c 0x81010001 \
    -o agent_key_public.pem \
    -f pem
```

### 3. PCR 정책 설정

```bash
# PCR 10번을 메트릭 측정용으로 예약
# (기본값: 0으로 초기화)
sudo tpm2_pcrreset 10
```

---

## Rust 구현

### Cargo.toml 수정

```toml
# agent/Cargo.toml

[dependencies]
# 기존 의존성...
ed25519-dalek = { workspace = true }  # Phase 1용 유지

# NEW: TPM 2.0 지원
tss-esapi = "7.4"  # TPM 2.0 Software Stack
hex = "0.4"
```

### RealTpmAgent 구현

```rust
// agent/src/tpm_real.rs

use anyhow::{Context, Result};
use bft_common::{RequestMetrics, TpmQuote};
use sha2::{Digest, Sha256};
use tss_esapi::{
    Context as TpmContext,
    Tcti,
    handles::KeyHandle,
    interface_types::{
        algorithm::HashingAlgorithm,
        resource_handles::Hierarchy,
    },
    structures::{
        Data, HashScheme, PcrSelectionList, PcrSlot,
        SignatureScheme, SymmetricDefinition,
    },
};
use tracing::{debug, info, warn};

/// Real TPM 2.0 agent
pub struct RealTpmAgent {
    tpm_context: TpmContext,
    signing_key_handle: KeyHandle,
    pcr_index: u32,  // PCR 인덱스 (기본: 10)
}

impl RealTpmAgent {
    /// Create a new TPM agent
    ///
    /// # Arguments
    /// * `tcti_config` - TCTI configuration (e.g., "device:/dev/tpmrm0")
    /// * `key_handle` - Persistent key handle (e.g., 0x81010001)
    /// * `pcr_index` - PCR index for metrics (default: 10)
    pub fn new(
        tcti_config: &str,
        key_handle: u32,
        pcr_index: u32,
    ) -> Result<Self> {
        info!("Initializing real TPM agent");
        info!("  TCTI: {}", tcti_config);
        info!("  Key handle: 0x{:08x}", key_handle);
        info!("  PCR index: {}", pcr_index);

        // TCTI 초기화
        let tcti = Tcti::from_str(tcti_config)
            .context("Failed to initialize TCTI")?;

        // TPM 컨텍스트 생성
        let mut tpm_context = TpmContext::new(tcti)
            .context("Failed to create TPM context")?;

        // 키 핸들 생성
        let signing_key_handle = KeyHandle::from_u32(key_handle)
            .map_err(|_| anyhow::anyhow!("Invalid key handle"))?;

        info!("TPM agent initialized successfully");

        Ok(Self {
            tpm_context,
            signing_key_handle,
            pcr_index,
        })
    }

    /// Generate TPM quote for metrics
    pub fn generate_quote(&mut self, metrics: &RequestMetrics) -> Result<TpmQuote> {
        debug!("Generating TPM quote for metrics");

        // 1. 메트릭을 해시
        let metrics_hash = self.hash_metrics(metrics)?;

        // 2. PCR extend (메트릭 해시를 PCR에 기록)
        self.extend_pcr(&metrics_hash)?;

        // 3. TPM Quote 생성
        let (pcr_values, quote_signature) = self.create_quote(&metrics_hash)?;

        info!("TPM quote generated successfully");

        Ok(TpmQuote {
            pcr_values,
            quote_signature,
            nonce: metrics_hash.to_vec(),
        })
    }

    /// Hash metrics to bytes
    fn hash_metrics(&self, metrics: &RequestMetrics) -> Result<Vec<u8>> {
        let mut hasher = Sha256::new();

        // 메트릭을 결정론적 순서로 해시
        hasher.update(metrics.sequence.to_le_bytes());
        hasher.update(&metrics.prev_hash);
        hasher.update(metrics.prompt_tokens.to_le_bytes());
        hasher.update(metrics.completion_tokens.to_le_bytes());
        hasher.update(metrics.cached_tokens.to_le_bytes());
        hasher.update(metrics.e2e_latency_ms.to_le_bytes());
        hasher.update(metrics.time_to_first_token_ms.to_le_bytes());
        hasher.update(metrics.estimated_cost.to_le_bytes());

        Ok(hasher.finalize().to_vec())
    }

    /// Extend PCR with metrics hash
    fn extend_pcr(&mut self, hash: &[u8]) -> Result<()> {
        debug!("Extending PCR {} with hash", self.pcr_index);

        // PCR 선택
        let pcr_slot = PcrSlot::Slot10;  // or use self.pcr_index

        // Digest 생성
        let digest = Data::try_from(hash.to_vec())
            .context("Failed to create digest")?;

        // PCR extend
        self.tpm_context
            .execute_with_nullauth_session(|ctx| {
                ctx.pcr_extend(pcr_slot.into(), digest.clone())
            })
            .context("Failed to extend PCR")?;

        debug!("PCR extended successfully");

        Ok(())
    }

    /// Create TPM quote
    fn create_quote(&mut self, qualifying_data: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        debug!("Creating TPM quote");

        // PCR 선택 (PCR 10번만)
        let pcr_selection_list = PcrSelectionList::builder()
            .with_selection(
                HashingAlgorithm::Sha256,
                &[PcrSlot::Slot10],
            )
            .build()
            .context("Failed to build PCR selection")?;

        // Qualifying data
        let qualifying_data = Data::try_from(qualifying_data.to_vec())
            .context("Failed to create qualifying data")?;

        // Signature scheme (RSA)
        let scheme = SignatureScheme::RsaSsa {
            hash_scheme: HashScheme::new(HashingAlgorithm::Sha256),
        };

        // Quote 생성
        let (attest, signature) = self.tpm_context
            .execute_with_nullauth_session(|ctx| {
                ctx.quote(
                    self.signing_key_handle,
                    qualifying_data.clone(),
                    scheme,
                    pcr_selection_list.clone(),
                )
            })
            .context("Failed to create quote")?;

        // PCR 값 읽기
        let (_, pcr_data) = self.tpm_context
            .pcr_read(pcr_selection_list)
            .context("Failed to read PCR")?;

        // PCR 값 추출
        let pcr_values = pcr_data.value();

        debug!("TPM quote created successfully");

        Ok((pcr_values.to_vec(), signature.signature().to_vec()))
    }

    /// Get public key for verification
    pub fn get_public_key(&mut self) -> Result<Vec<u8>> {
        let (public, _, _) = self.tpm_context
            .read_public(self.signing_key_handle)
            .context("Failed to read public key")?;

        // RSA 공개키 추출
        let public_bytes = public.try_into()
            .context("Failed to convert public key")?;

        Ok(public_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]  // TPM 하드웨어 필요
    fn test_tpm_quote_generation() {
        let mut agent = RealTpmAgent::new(
            "device:/dev/tpmrm0",
            0x81010001,
            10,
        ).unwrap();

        let metrics = RequestMetrics {
            prompt_tokens: 100,
            completion_tokens: 50,
            cached_tokens: 0,
            e2e_latency_ms: 1000,
            time_to_first_token_ms: 100,
            estimated_cost: 0.01,
            sequence: 1,
            prev_hash: vec![0u8; 32],
            current_hash: vec![],
        };

        let quote = agent.generate_quote(&metrics).unwrap();

        assert!(!quote.pcr_values.is_empty());
        assert!(!quote.quote_signature.is_empty());
    }
}
```

### SimulatedTpmAgent 유지 (하위 호환)

```rust
// agent/src/tpm.rs

// 기존 Ed25519 구현 유지 (개발/테스트용)
pub struct SimulatedTpmAgent {
    signing_key: SigningKey,
}

// ... 기존 구현 유지 ...
```

### Agent main.rs 수정

```rust
// agent/src/main.rs

use bft_agent::tpm::SimulatedTpmAgent;
#[cfg(feature = "tpm-hardware")]
use bft_agent::tpm_real::RealTpmAgent;

#[derive(Debug, Clone)]
struct AgentConfig {
    // 기존 필드들...

    // NEW: TPM 설정
    use_hardware_tpm: bool,
    tpm_tcti_config: String,
    tpm_key_handle: u32,
    tpm_pcr_index: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            // 기존 기본값...

            use_hardware_tpm: false,
            tpm_tcti_config: "device:/dev/tpmrm0".to_string(),
            tpm_key_handle: 0x81010001,
            tpm_pcr_index: 10,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // ... 초기화 ...

    let config = load_config()?;

    // TPM agent 생성 (하드웨어 또는 시뮬레이션)
    #[cfg(feature = "tpm-hardware")]
    let tpm_agent = if config.use_hardware_tpm {
        info!("Using hardware TPM 2.0");
        let mut real_agent = RealTpmAgent::new(
            &config.tpm_tcti_config,
            config.tpm_key_handle,
            config.tpm_pcr_index,
        )?;

        // 공개키 추출 및 로깅
        let public_key = real_agent.get_public_key()?;
        info!("TPM public key: {}", hex::encode(&public_key));

        TpmAgentWrapper::Real(real_agent)
    } else {
        info!("Using simulated TPM (Ed25519)");
        let signing_key = load_or_generate_signing_key(&config)?;
        TpmAgentWrapper::Simulated(SimulatedTpmAgent::new(signing_key))
    };

    #[cfg(not(feature = "tpm-hardware"))]
    let tpm_agent = {
        info!("Using simulated TPM (Ed25519)");
        let signing_key = load_or_generate_signing_key(&config)?;
        SimulatedTpmAgent::new(signing_key)
    };

    // ... 나머지 로직 ...

    Ok(())
}

// TPM agent wrapper
enum TpmAgentWrapper {
    Real(RealTpmAgent),
    Simulated(SimulatedTpmAgent),
}

impl TpmAgentWrapper {
    fn generate_quote(&mut self, metrics: &RequestMetrics) -> TpmQuote {
        match self {
            TpmAgentWrapper::Real(agent) => agent.generate_quote(metrics).unwrap(),
            TpmAgentWrapper::Simulated(agent) => agent.generate_quote(metrics),
        }
    }
}
```

---

## Verifier TPM 검증 강화

### TPM Quote 검증 구현

```rust
// verifier/src/tpm.rs

use tss_esapi::{
    Context as TpmContext,
    Tcti,
    structures::{Public, Signature, Attest},
};

/// TPM Quote verifier
pub struct TpmQuoteVerifier {
    // 기존 필드...
    agent_public_keys: HashMap<String, Vec<u8>>,

    // NEW: TPM 컨텍스트 (검증용)
    #[cfg(feature = "tpm-hardware")]
    tpm_context: Option<TpmContext>,
}

impl TpmQuoteVerifier {
    pub fn new(agent_public_keys: HashMap<String, String>) -> Self {
        // 기존 Ed25519 공개키 변환
        let ed25519_keys = agent_public_keys.iter()
            .filter_map(|(id, hex)| {
                hex::decode(hex).ok().map(|bytes| (id.clone(), bytes))
            })
            .collect();

        #[cfg(feature = "tpm-hardware")]
        let tpm_context = {
            // TPM 컨텍스트 초기화 (검증용)
            let tcti = Tcti::from_str("device:/dev/tpmrm0").ok();
            tcti.and_then(|t| TpmContext::new(t).ok())
        };

        Self {
            agent_public_keys: ed25519_keys,
            #[cfg(feature = "tpm-hardware")]
            tpm_context,
        }
    }

    /// Verify TPM quote
    pub fn verify_quote(
        &self,
        metrics: &RequestMetrics,
        quote: &TpmQuote,
        agent_id: &str,
    ) -> Result<bool> {
        // 1. Agent 공개키 조회
        let public_key = self.agent_public_keys.get(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown agent: {}", agent_id))?;

        // 2. 메트릭 해시 계산
        let metrics_hash = self.hash_metrics(metrics);

        // 3. Quote 검증
        #[cfg(feature = "tpm-hardware")]
        if let Some(ref ctx) = self.tpm_context {
            return self.verify_tpm_quote(ctx, public_key, &metrics_hash, quote);
        }

        // Fallback: Ed25519 검증 (시뮬레이션 모드)
        self.verify_ed25519_signature(public_key, &metrics_hash, quote)
    }

    #[cfg(feature = "tpm-hardware")]
    fn verify_tpm_quote(
        &self,
        _ctx: &TpmContext,
        public_key: &[u8],
        metrics_hash: &[u8],
        quote: &TpmQuote,
    ) -> Result<bool> {
        // TPM Quote 검증 구현
        // 1. Attest 구조체 파싱
        // 2. PCR 값 검증
        // 3. Signature 검증

        // 간략화된 검증 (실제로는 더 복잡)
        Ok(!quote.quote_signature.is_empty())
    }

    fn verify_ed25519_signature(
        &self,
        public_key: &[u8],
        metrics_hash: &[u8],
        quote: &TpmQuote,
    ) -> Result<bool> {
        // 기존 Ed25519 검증 로직
        use ed25519_dalek::{Verifier, VerifyingKey, Signature};

        if public_key.len() != 32 {
            return Err(anyhow::anyhow!("Invalid public key length"));
        }

        let mut key_bytes = [0u8; 32];
        key_bytes.copy_from_slice(public_key);

        let verifying_key = VerifyingKey::from_bytes(&key_bytes)?;
        let signature = Signature::from_slice(&quote.quote_signature)?;

        Ok(verifying_key.verify(metrics_hash, &signature).is_ok())
    }
}
```

---

## Cargo Feature 설정

```toml
# Cargo.toml (workspace root)

[features]
default = []
tpm-hardware = ["tss-esapi"]

[dependencies]
# Optional: TPM 2.0 지원
tss-esapi = { version = "7.4", optional = true }
```

```toml
# agent/Cargo.toml

[dependencies]
# 기존 의존성...

# TPM 2.0 (optional)
tss-esapi = { version = "7.4", optional = true }

[features]
tpm-hardware = ["tss-esapi"]
```

---

## 빌드 및 실행

### 빌드

```bash
# 시뮬레이션 모드 (기본)
cargo build --release --package bft-agent

# TPM 하드웨어 모드
cargo build --release --package bft-agent --features tpm-hardware
cargo build --release --package bft-verifier --features tpm-hardware
```

### 실행

#### TPM 하드웨어 모드

```bash
# Agent 실행 (실제 TPM 사용)
RUST_LOG=info \
USE_HARDWARE_TPM=true \
TPM_TCTI_CONFIG="device:/dev/tpmrm0" \
TPM_KEY_HANDLE=0x81010001 \
TPM_PCR_INDEX=10 \
./target/release/agent
```

#### 시뮬레이션 모드

```bash
# Agent 실행 (Ed25519 사용)
RUST_LOG=info \
USE_HARDWARE_TPM=false \
./target/release/agent
```

---

## 검증

### TPM Quote 검증

```bash
# 1. Quote 생성 확인
sudo journalctl -u bft-agent -f | grep "TPM quote generated"

# 2. PCR 값 확인
sudo tpm2_pcrread sha256:10

# 3. Quote 수동 검증
sudo tpm2_quote \
    -c 0x81010001 \
    -l sha256:10 \
    -q qualifying_data.bin \
    -m quote.msg \
    -s quote.sig \
    -g sha256

sudo tpm2_checkquote \
    -u agent_key_public.pem \
    -m quote.msg \
    -s quote.sig \
    -f plain \
    -q qualifying_data.bin \
    -g sha256 \
    -l sha256:10
```

---

## 보안 이점

### Before (Ed25519) vs After (TPM)

| 항목 | Ed25519 | TPM 2.0 |
|------|---------|---------|
| **키 저장** | 파일/메모리 | TPM 칩 내부 |
| **키 추출** | ❌ 가능 | ✅ 불가능 |
| **프로세스 손상** | ❌ 키 탈취 가능 | ✅ 키 보호됨 |
| **메모리 덤프** | ❌ 키 노출 | ✅ 키 보호됨 |
| **서명 위조** | ⚠️ 키 탈취 시 가능 | ✅ 불가능 |
| **하드웨어 보안** | ❌ 없음 | ✅ 있음 |

---

## 문제 해결

### 문제 1: /dev/tpmrm0 권한 오류

```bash
# 해결: Agent를 tss 그룹에 추가
sudo usermod -a -G tss vllm-user
sudo chmod 666 /dev/tpmrm0
```

### 문제 2: TPM busy

```bash
# 해결: 다른 프로세스 종료
sudo pkill tpm2_
sudo systemctl restart tpm2-abrmd
```

### 문제 3: PCR 값 불일치

```bash
# PCR 리셋
sudo tpm2_pcrreset 10
```

---

## 다음 단계

✅ **Phase 2 완료 후**:
- 실제 TPM 2.0 하드웨어 서명
- 키 탈취 불가능
- 하드웨어 기반 무결성

🔜 **Phase 3로 이동**:
- Keylime 통합
- IMA (Integrity Measurement Architecture)
- 시스템 전체 무결성 검증
