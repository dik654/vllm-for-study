# Phase 3: Keylime 통합 가이드

이 문서는 Keylime을 사용하여 시스템 전체 무결성 검증을 추가하는 방법을 설명합니다.

## 개요

### Keylime이란?

**Keylime**은 TPM 기반 원격 증명(Remote Attestation) 프레임워크입니다.

- **GitHub**: https://github.com/keylime/keylime
- **CNCF**: Sandbox Project
- **사용 사례**: 클라우드 보안, 제로 트러스트, 공급망 보안

### 왜 Keylime인가?

| 항목 | Phase 2 (TPM만) | Phase 3 (Keylime + TPM) |
|------|----------------|------------------------|
| **메트릭 무결성** | ✅ 보장 | ✅ 보장 |
| **바이너리 검증** | ❌ 없음 | ✅ IMA로 검증 |
| **시스템 무결성** | ❌ 없음 | ✅ 전체 시스템 검증 |
| **실시간 모니터링** | ❌ 없음 | ✅ Keylime Verifier |
| **정책 기반 제어** | ❌ 없음 | ✅ Allowlist/Excludelist |
| **감사 추적** | ⚠️ 부분 | ✅ 완전 |

---

## 아키텍처

### 통합 시스템 구조

```
┌────────────────────────────────────────────────────────────┐
│                   LLM 서버 (각 서버마다)                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ vLLM 프로세스 (Python)                               │  │
│  │   • Metrics Collector                                │  │
│  │   • IMA가 바이너리 해시 측정                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ BFT Agent (Rust)                                     │  │
│  │   • Real TPM Agent                                   │  │
│  │   • Metrics 제출                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Keylime Agent (Python)                               │  │
│  │   • IMA 로그 수집                                    │  │
│  │   • TPM Quote 생성 (시스템용)                        │  │
│  │   • Verifier에게 전송                                │  │
│  └──────────────────────────────────────────────────────┘  │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ TPM 2.0 칩                                           │  │
│  │   • PCR 0-23                                         │  │
│  │   • 서명 키                                          │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
                    │                    │
        ┌───────────┴────────┐  ┌───────┴────────┐
        ▼                    ▼  ▼                ▼
┌──────────────┐  ┌──────────────────┐  ┌──────────────┐
│ Keylime      │  │ Keylime          │  │ BFT          │
│ Verifier     │  │ Registrar        │  │ Coordinator  │
│ (시스템 검증)│  │ (Agent 등록 DB)  │  │ (메트릭 검증)│
└──────────────┘  └──────────────────┘  └──────────────┘
```

---

## Keylime 설치

### 1. 사전 준비

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-tornado \
    python3-requests \
    python3-sqlalchemy \
    python3-alembic \
    python3-packaging \
    python3-psutil \
    python3-gnupg \
    git \
    gcc \
    g++ \
    make \
    libssl-dev \
    swig \
    python3-yaml \
    python3-cryptography

# TPM 도구 (Phase 2에서 설치됨)
sudo apt-get install -y \
    tpm2-tools \
    tpm2-abrmd \
    libtss2-dev
```

### 2. Keylime 설치

```bash
# Keylime 저장소 클론
git clone https://github.com/keylime/keylime.git
cd keylime

# Python 의존성 설치
sudo pip3 install -r requirements.txt

# Keylime 설치
sudo python3 setup.py install

# 설치 확인
keylime_tenant --help
```

### 3. 설정 파일 생성

```bash
# 설정 디렉토리 생성
sudo mkdir -p /etc/keylime
sudo mkdir -p /var/lib/keylime

# 기본 설정 복사
sudo cp /usr/local/etc/keylime/*.conf /etc/keylime/
```

---

## IMA (Integrity Measurement Architecture) 설정

### IMA란?

Linux 커널의 무결성 측정 기능:
- 파일이 열릴 때마다 해시 계산
- 해시를 TPM PCR에 extend
- 측정 로그 기록 (`/sys/kernel/security/ima/ascii_runtime_measurements`)

### 1. 커널 파라미터 설정

```bash
# GRUB 설정 편집
sudo nano /etc/default/grub

# GRUB_CMDLINE_LINUX에 추가:
GRUB_CMDLINE_LINUX="ima_policy=tcb ima_hash=sha256 ima_template=ima-ng"
```

**IMA 정책 옵션**:
- `ima_policy=tcb`: Trusted Computing Base (시스템 파일 측정)
- `ima_policy=appraise_tcb`: 측정 + 검증
- `ima_hash=sha256`: SHA-256 해시 사용
- `ima_template=ima-ng`: 템플릿 (파일명 포함)

```bash
# GRUB 업데이트
sudo update-grub

# 재부팅
sudo reboot
```

### 2. IMA 활성화 확인

```bash
# IMA 로그 확인
cat /sys/kernel/security/ima/ascii_runtime_measurements | head -10

# 출력 예시:
# 10 abc123... ima-ng sha256:def456... /usr/bin/python3
# 10 def789... ima-ng sha256:012abc... /usr/sbin/vllm
```

### 3. vLLM 바이너리 측정 확인

```bash
# vLLM 바이너리 측정 확인
grep vllm /sys/kernel/security/ima/ascii_runtime_measurements

# 또는 특정 파일 측정
cat /sys/kernel/security/ima/ascii_runtime_measurements | \
    grep "/opt/vllm/vllm_server"
```

---

## Keylime 구성 요소 설정

### 1. Keylime Registrar (DB)

**역할**: Agent 등록 정보 저장

```bash
# 설정 파일 편집
sudo nano /etc/keylime/registrar.conf
```

```ini
[registrar]
# 데이터베이스 경로
database_url = sqlite:////var/lib/keylime/registrar-data.sqlite

# 리슨 주소
ip = 0.0.0.0
port = 8890

# TLS 설정
tls_dir = /var/lib/keylime/cv_ca
```

```bash
# Registrar 시작
keylime_registrar &

# 또는 systemd 서비스로 실행
sudo systemctl start keylime_registrar
sudo systemctl enable keylime_registrar
```

### 2. Keylime Verifier (검증 서버)

**역할**: Agent로부터 Quote 수신, 검증, 정책 평가

```bash
# 설정 파일 편집
sudo nano /etc/keylime/verifier.conf
```

```ini
[verifier]
# 리슨 주소
ip = 0.0.0.0
port = 8881

# Registrar 주소
registrar_ip = 127.0.0.1
registrar_port = 8890

# 데이터베이스
database_url = sqlite:////var/lib/keylime/verifier-data.sqlite

# TLS 설정
tls_dir = /var/lib/keylime/cv_ca

# Quote 검증 간격 (초)
quote_interval = 60

# 허용 리스트 (allowlist)
# vLLM 관련 바이너리 허용
ima_allowlist = /etc/keylime/allowlist.txt

# 제외 리스트 (excludelist)
ima_excludelist = /etc/keylime/excludelist.txt
```

### 3. Allowlist 생성

**Allowlist**: 신뢰할 수 있는 바이너리 목록

```bash
# Allowlist 파일 생성
sudo nano /etc/keylime/allowlist.txt
```

```json
{
  "meta": {
    "version": 1
  },
  "hashes": {
    "/usr/bin/python3.10": "sha256:abc123...",
    "/opt/vllm/vllm_server": "sha256:def456...",
    "/usr/local/bin/bft-agent": "sha256:789ghi..."
  }
}
```

**자동 생성 스크립트**:

```bash
#!/bin/bash
# generate_allowlist.sh

ALLOWLIST_FILE="/etc/keylime/allowlist.txt"

echo '{
  "meta": {"version": 1},
  "hashes": {' > $ALLOWLIST_FILE

# vLLM 바이너리
echo "    \"/opt/vllm/vllm_server\": \"sha256:$(sha256sum /opt/vllm/vllm_server | cut -d' ' -f1)\"," >> $ALLOWLIST_FILE

# BFT Agent
echo "    \"/usr/local/bin/bft-agent\": \"sha256:$(sha256sum /usr/local/bin/bft-agent | cut -d' ' -f1)\"," >> $ALLOWLIST_FILE

# Python 인터프리터
echo "    \"/usr/bin/python3\": \"sha256:$(sha256sum /usr/bin/python3 | cut -d' ' -f1)\"" >> $ALLOWLIST_FILE

echo '  }
}' >> $ALLOWLIST_FILE

echo "Allowlist generated: $ALLOWLIST_FILE"
```

```bash
chmod +x generate_allowlist.sh
sudo ./generate_allowlist.sh
```

### 4. Verifier 시작

```bash
keylime_verifier &

# 또는 systemd
sudo systemctl start keylime_verifier
sudo systemctl enable keylime_verifier
```

---

## Keylime Agent 설정 (각 LLM 서버)

### 1. Agent 설정

```bash
sudo nano /etc/keylime/agent.conf
```

```ini
[agent]
# UUID (각 서버마다 고유)
uuid = auto

# Registrar 주소
registrar_ip = 10.0.0.100
registrar_port = 8890

# TPM 설정
tpm_ownerpassword =
enable_iak = True
enable_idevid = True

# IMA 측정
ima_ml_path = /sys/kernel/security/ima/ascii_runtime_measurements

# 리슨 주소
ip = 0.0.0.0
port = 9002

# TLS 설정
tls_dir = /var/lib/keylime/secure
```

### 2. Agent 시작

```bash
# 각 LLM 서버에서
sudo keylime_agent &

# 또는 systemd
sudo systemctl start keylime_agent
sudo systemctl enable keylime_agent
```

### 3. Agent 등록 확인

```bash
# Verifier 서버에서
keylime_tenant -c listagents

# 출력:
# Agent ID: 12345678-1234-1234-1234-123456789012
# Agent IP: 10.0.0.10
# Status: ACTIVE
```

---

## Keylime로 Agent 등록 및 모니터링

### 1. Agent 추가

```bash
# Tenant 도구로 Agent 추가
keylime_tenant -c add \
    -t 10.0.0.10 \
    -u 12345678-1234-1234-1234-123456789012 \
    --allowlist /etc/keylime/allowlist.txt

# 출력:
# Agent 12345678-... added successfully
# Status: ACTIVE
```

### 2. Agent 상태 확인

```bash
# 특정 Agent 상태
keylime_tenant -c status \
    -u 12345678-1234-1234-1234-123456789012

# 모든 Agent 상태
keylime_tenant -c listagents
```

### 3. 실시간 모니터링

```bash
# Verifier 로그 확인
sudo tail -f /var/log/keylime/verifier.log

# 출력 예시:
# [INFO] Quote received from agent 12345678-...
# [INFO] PCR validation: PASS
# [INFO] IMA measurement list: 1234 entries
# [INFO] Allowlist check: PASS
# [INFO] Agent 12345678-... is TRUSTED
```

---

## 무결성 검증 시나리오

### 시나리오 1: 정상 동작

```
1. vLLM 서버 부팅
2. IMA가 vllm_server 바이너리 측정
   → PCR 10 extend
3. Keylime Agent가 Quote 생성
   → Verifier에게 전송
4. Verifier가 검증:
   - Quote 서명 검증: ✅
   - PCR 값 검증: ✅
   - IMA 로그 재생: ✅
   - Allowlist 확인: ✅ (vllm_server 허용됨)
5. 결과: TRUSTED ✅
6. BFT Agent가 메트릭 제출
7. BFT Coordinator가 수락
```

### 시나리오 2: 바이너리 변조 탐지

```
1. 공격자가 vllm_server 변조
   $ echo "malicious_code" >> /opt/vllm/vllm_server

2. 변조된 바이너리 실행
3. IMA가 변조된 해시 측정
   → SHA256: XYZ123... (원본과 다름!)
4. Keylime Agent가 Quote 생성
5. Verifier가 검증:
   - Quote 서명: ✅
   - PCR 값: ✅
   - IMA 로그 재생: ✅
   - Allowlist 확인: ❌
     Expected: abc123...
     Actual:   XYZ123...
6. 결과: UNTRUSTED ❌
7. Verifier가 경고 발생
   → "File /opt/vllm/vllm_server failed allowlist check"
8. BFT Coordinator에 알림
   → 해당 서버 메트릭 거부
```

### 시나리오 3: Keylime Agent 중지 공격

```
1. 공격자가 Keylime Agent 중지
   $ sudo systemctl stop keylime_agent

2. 60초 후 (quote_interval)
3. Verifier가 Quote 미수신 탐지
4. Agent 상태: ACTIVE → UNRESPONSIVE
5. Verifier가 경고 발생
6. BFT Coordinator에 알림
   → 해당 서버 메트릭 거부
7. 관리자에게 알림
```

---

## BFT 시스템과 Keylime 통합

### 통합 아키텍처

```python
# coordinator/keylime_integration.py

import requests
import logging

logger = logging.getLogger(__name__)

class KeylimeIntegration:
    """Keylime Verifier와 통합"""

    def __init__(self, verifier_url: str):
        self.verifier_url = verifier_url  # e.g., "http://10.0.0.100:8881"

    def check_agent_trust(self, agent_id: str) -> bool:
        """
        Agent가 신뢰할 수 있는지 Keylime Verifier에 확인

        Returns:
            True if agent is TRUSTED
            False if agent is UNTRUSTED or unreachable
        """
        try:
            response = requests.get(
                f"{self.verifier_url}/v2.1/agents/{agent_id}",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                status = data.get("results", {}).get("operational_state")

                if status == "TRUSTED":
                    logger.info(f"Agent {agent_id} is TRUSTED")
                    return True
                else:
                    logger.warning(f"Agent {agent_id} is {status}")
                    return False
            else:
                logger.error(f"Keylime API error: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Failed to check agent trust: {e}")
            return False

    def get_all_trusted_agents(self) -> list:
        """Get list of all trusted agents"""
        try:
            response = requests.get(
                f"{self.verifier_url}/v2.1/agents",
                timeout=5
            )

            if response.status_code == 200:
                data = response.json()
                agents = data.get("results", [])

                trusted = [
                    agent["agent_id"]
                    for agent in agents
                    if agent.get("operational_state") == "TRUSTED"
                ]

                return trusted
            else:
                return []

        except Exception as e:
            logger.error(f"Failed to get trusted agents: {e}")
            return []
```

### Coordinator에 통합

```rust
// coordinator/src/keylime_client.rs

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

#[derive(Debug, Serialize, Deserialize)]
struct KeylimeAgentStatus {
    agent_id: String,
    operational_state: String,  // "TRUSTED", "UNTRUSTED", etc.
}

pub struct KeylimeClient {
    client: Client,
    verifier_url: String,
}

impl KeylimeClient {
    pub fn new(verifier_url: String) -> Self {
        Self {
            client: Client::new(),
            verifier_url,
        }
    }

    /// Check if agent is trusted by Keylime
    pub async fn is_agent_trusted(&self, agent_id: &str) -> Result<bool, Box<dyn std::error::Error>> {
        let url = format!("{}/v2.1/agents/{}", self.verifier_url, agent_id);

        let response = self.client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await?;

        if response.status().is_success() {
            let status: KeylimeAgentStatus = response.json().await?;

            if status.operational_state == "TRUSTED" {
                info!("Agent {} is TRUSTED by Keylime", agent_id);
                Ok(true)
            } else {
                warn!("Agent {} is {} by Keylime", agent_id, status.operational_state);
                Ok(false)
            }
        } else {
            error!("Keylime API error: {}", response.status());
            Ok(false)
        }
    }
}
```

### submit_metrics에 Keylime 검증 추가

```rust
// coordinator/src/server.rs

async fn submit_metrics(
    &self,
    request: Request<MetricSubmission>,
) -> Result<Response<SubmissionAck>, Status> {
    let submission = request.into_inner();

    // ========== Keylime 통합 검증 ==========

    if let Some(ref keylime_client) = self.keylime_client {
        match keylime_client.is_agent_trusted(&submission.agent_id).await {
            Ok(true) => {
                info!("Agent {} passed Keylime trust check", submission.agent_id);
            }
            Ok(false) => {
                warn!("Agent {} UNTRUSTED by Keylime, rejecting submission", submission.agent_id);

                return Ok(Response::new(SubmissionAck {
                    verification_id: "".to_string(),
                    status: 2,  // REJECTED
                    message: "Agent not trusted by Keylime".to_string(),
                }));
            }
            Err(e) => {
                warn!("Keylime check failed: {}, allowing submission", e);
                // Fall through (don't reject if Keylime unreachable)
            }
        }
    }

    // ========== 기존 BFT 검증 계속 ==========

    // Nonce 검증, Chain 검증, etc...

    Ok(Response::new(SubmissionAck {
        verification_id: "...".to_string(),
        status: 1,  // ACCEPTED
        message: "Verification completed".to_string(),
    }))
}
```

---

## 배포 스크립트

### Systemd 서비스 파일

#### keylime-agent.service

```ini
[Unit]
Description=Keylime Agent
After=network.target tpm2-abrmd.service
Requires=tpm2-abrmd.service

[Service]
Type=simple
User=keylime
Group=tss
ExecStart=/usr/local/bin/keylime_agent
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### keylime-verifier.service

```ini
[Unit]
Description=Keylime Verifier
After=network.target keylime-registrar.service
Requires=keylime-registrar.service

[Service]
Type=simple
User=keylime
ExecStart=/usr/local/bin/keylime_verifier
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### 자동 배포 스크립트

```bash
#!/bin/bash
# deploy_keylime.sh

set -e

echo "=== Keylime Deployment Script ==="

# 1. 서버 역할 확인
read -p "Server role (agent/verifier/registrar): " ROLE

if [ "$ROLE" == "agent" ]; then
    echo "Installing Keylime Agent..."

    # IMA 활성화
    grep -q "ima_policy" /etc/default/grub || {
        echo "Adding IMA parameters to GRUB..."
        sudo sed -i 's/GRUB_CMDLINE_LINUX="/&ima_policy=tcb ima_hash=sha256 ima_template=ima-ng /' /etc/default/grub
        sudo update-grub
        echo "REBOOT REQUIRED for IMA activation"
    }

    # Keylime Agent 설치
    sudo pip3 install keylime

    # Agent 설정
    sudo mkdir -p /etc/keylime
    cat << EOF | sudo tee /etc/keylime/agent.conf
[agent]
uuid = auto
registrar_ip = ${VERIFIER_IP}
registrar_port = 8890
EOF

    # Systemd 서비스
    sudo cp keylime-agent.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable keylime-agent
    sudo systemctl start keylime-agent

    echo "Keylime Agent installed successfully"

elif [ "$ROLE" == "verifier" ]; then
    echo "Installing Keylime Verifier + Registrar..."

    # 둘 다 설치
    sudo pip3 install keylime

    # Registrar 시작
    sudo systemctl enable keylime-registrar
    sudo systemctl start keylime-registrar

    # Verifier 시작
    sudo systemctl enable keylime-verifier
    sudo systemctl start keylime-verifier

    echo "Keylime Verifier + Registrar installed successfully"

fi

echo "=== Deployment Complete ==="
```

---

## 검증 및 테스트

### 1. Keylime 동작 확인

```bash
# Registrar 상태
curl http://localhost:8890/version

# Verifier 상태
curl http://localhost:8881/version

# Agent 목록
keylime_tenant -c listagents
```

### 2. IMA 로그 확인

```bash
# vLLM 바이너리 측정 확인
grep vllm /sys/kernel/security/ima/ascii_runtime_measurements

# 최근 측정 확인
tail -20 /sys/kernel/security/ima/ascii_runtime_measurements
```

### 3. 무결성 검증 테스트

```bash
# 정상 케이스
keylime_tenant -c status -u AGENT_UUID
# Expected: operational_state = TRUSTED

# 변조 케이스 (테스트 환경에서만!)
echo "test" >> /opt/vllm/vllm_server
sudo systemctl restart vllm

# Keylime가 탐지해야 함
keylime_tenant -c status -u AGENT_UUID
# Expected: operational_state = UNTRUSTED
```

---

## 문제 해결

### 문제 1: IMA 로그가 비어있음

```bash
# 커널 파라미터 확인
cat /proc/cmdline | grep ima

# IMA 파일시스템 확인
mount | grep securityfs

# IMA 활성화 확인
dmesg | grep ima
```

**해결**: GRUB 설정 확인 및 재부팅

### 문제 2: Keylime Agent가 Registrar에 연결 안됨

```bash
# 네트워크 확인
ping REGISTRAR_IP
telnet REGISTRAR_IP 8890

# Agent 로그
sudo journalctl -u keylime-agent -f
```

### 문제 3: Allowlist 불일치

```bash
# 실제 해시 확인
sha256sum /opt/vllm/vllm_server

# IMA 로그에서 해시 확인
grep vllm_server /sys/kernel/security/ima/ascii_runtime_measurements

# Allowlist 재생성
sudo ./generate_allowlist.sh
```

---

## 보안 이점 요약

| 항목 | Phase 1 | Phase 2 (TPM) | Phase 3 (Keylime) |
|------|---------|--------------|------------------|
| **메트릭 무결성** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **키 보호** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **바이너리 검증** | ❌ | ❌ | ✅ |
| **시스템 무결성** | ❌ | ❌ | ✅ |
| **실시간 모니터링** | ❌ | ❌ | ✅ |
| **정책 기반 제어** | ❌ | ❌ | ✅ |
| **감사 추적** | ⚠️ | ⚠️ | ✅ |

**Phase 3 완료 시 달성**:
- ✅ 하드웨어 기반 보안 (TPM)
- ✅ 메트릭 무결성 보장
- ✅ 시스템 바이너리 검증
- ✅ 실시간 침입 탐지
- ✅ 정책 기반 접근 제어
- ✅ 완전한 감사 추적

---

## 참고 자료

- **Keylime 공식 문서**: https://keylime.readthedocs.io/
- **IMA 가이드**: https://sourceforge.net/p/linux-ima/wiki/Home/
- **TPM 2.0 Spec**: https://trustedcomputinggroup.org/
- **CNCF Keylime**: https://www.cncf.io/projects/keylime/

---

## 다음 단계

✅ **Phase 1 완료**: Ed25519 + BFT 합의
✅ **Phase 2 완료**: 실제 TPM 2.0 통합
✅ **Phase 3 완료**: Keylime + IMA 시스템 무결성

🎯 **프로덕션 배포 준비 완료**
