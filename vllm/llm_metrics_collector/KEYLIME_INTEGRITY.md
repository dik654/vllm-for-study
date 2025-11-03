# Keylime 기반 분산 LLM 메트릭 무결성 보장 시스템

TPM(Trusted Platform Module)과 Keylime을 사용하여 분산된 LLM 서버의 메트릭 조작을 방지하는 종단간(end-to-end) 솔루션입니다.

---

## 📋 목차

1. [개요](#1-개요)
2. [Keylime 및 TPM 기초](#2-keylime-및-tpm-기초)
3. [위협 모델](#3-위협-모델)
4. [아키텍처 설계](#4-아키텍처-설계)
5. [구현 가이드](#5-구현-가이드)
6. [보안 검증](#6-보안-검증)
7. [대안 및 비교](#7-대안-및-비교)

---

## 1. 개요

### 1.1 문제 정의

**시나리오**:
- 여러 대의 분산된 LLM 서버가 메트릭(토큰 사용량, 레이턴시, 비용 등)을 수집
- 악의적인 서버 운영자가 메트릭을 조작하여 과소/과다 청구 가능
- 중앙 집계 시스템이 조작된 메트릭을 검증할 방법 필요

**요구사항**:
1. ✅ **무결성(Integrity)**: 메트릭이 조작되지 않았음을 보장
2. ✅ **증명(Attestation)**: 메트릭을 생성한 시스템이 신뢰할 수 있음을 증명
3. ✅ **감사(Auditability)**: 메트릭 생성 과정을 추적 가능
4. ✅ **부인 방지(Non-repudiation)**: 서버가 메트릭 생성을 부인할 수 없음

### 1.2 Keylime + TPM 솔루션

**핵심 아이디어**:
```
메트릭 생성 → TPM에 측정 기록 → Keylime 원격 증명 → 검증자가 무결성 확인
```

**장점**:
- ✅ 하드웨어 기반 보안 (TPM 칩)
- ✅ 조작 불가능한 측정 (PCR Extend 연산)
- ✅ 실시간 모니터링
- ✅ CNCF 프로젝트 (오픈소스)

---

## 2. Keylime 및 TPM 기초

### 2.1 TPM (Trusted Platform Module)

**TPM이란?**
- 하드웨어 보안 칩 (마더보드에 내장 또는 별도 모듈)
- 암호화 키 저장, 무결성 측정, 원격 증명

**핵심 구성요소**:

#### PCR (Platform Configuration Register)
- TPM 내부의 특수 레지스터 (24개, 각 32바이트)
- **Extend 연산만 가능** (덮어쓰기 불가)
- 공식: `PCR_new = Hash(PCR_old || data_to_extend)`

```python
# 예시
PCR[10] = 0x0000...0000  # 초기값
PCR[10] = SHA256(PCR[10] || "metric_data_1")  # 첫 번째 extend
PCR[10] = SHA256(PCR[10] || "metric_data_2")  # 두 번째 extend
# → 순서가 바뀌거나 데이터가 변경되면 최종 PCR 값이 달라짐
```

**특징**:
- ✅ **누적성(Cumulative)**: 모든 측정이 순차적으로 누적
- ✅ **비가역성(Irreversible)**: 이전 값으로 되돌릴 수 없음
- ✅ **순서 보장**: 측정 순서가 중요

#### Quote (TPM 서명)
- TPM이 현재 PCR 값을 **개인키로 서명**
- 원격 검증자가 **공개키로 검증** 가능
- 조작 불가능한 증명 제공

```python
# TPM Quote 예시
quote = {
    'pcr_values': {10: '0xabcd...', 16: '0x1234...'},
    'signature': 'TPM_private_key_signed',
    'timestamp': 1699123456,
}
```

---

### 2.2 Keylime 아키텍처

**GitHub**: https://github.com/keylime/keylime
⭐ 500+ stars, CNCF Sandbox Project

#### 3-Tier 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│                  Keylime Verifier                       │
│  (중앙 검증 서버 - 메트릭 무결성 검증)                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ├──────────┬──────────┬──────────┐
                        ▼          ▼          ▼          ▼
              ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
              │ Keylime Agent│ │ Keylime Agent│ │ Keylime Agent│
              │ + vLLM Server│ │ + vLLM Server│ │ + vLLM Server│
              │   (Node 1)   │ │   (Node 2)   │ │   (Node 3)   │
              └──────────────┘ └──────────────┘ └──────────────┘
                        │
                        ▼
              ┌──────────────┐
              │   Registrar  │
              │ (에이전트 등록)│
              └──────────────┘
```

**구성요소**:

1. **Keylime Agent** (각 LLM 서버에 설치)
   - TPM과 통신
   - 메트릭 측정을 TPM에 기록
   - Quote를 Verifier에 전송

2. **Keylime Verifier** (중앙 서버)
   - Agent로부터 Quote 수신
   - 측정 로그(Event Log) 재생(Replay)
   - 무결성 검증 및 정책 평가

3. **Keylime Registrar** (중앙 DB)
   - Agent 등록 정보 저장
   - TPM 공개키 관리

---

### 2.3 IMA (Integrity Measurement Architecture)

**Linux 커널 기능**으로 파일 무결성 측정:

```bash
# IMA 측정 로그 확인
cat /sys/kernel/security/ima/ascii_runtime_measurements

# 예시 출력
10 abc123... ima-ng sha256:def456... /usr/bin/python3
10 def789... ima-ng sha256:012abc... /opt/vllm/metrics_collector.py
```

**동작 방식**:
1. 파일이 실행/읽기될 때마다 해시 계산
2. 해시를 TPM PCR에 Extend
3. 측정 로그에 기록
4. Verifier가 로그를 재생하여 PCR 값 검증

---

## 3. 위협 모델

### 3.1 공격 시나리오

#### 시나리오 1: 메트릭 데이터 조작
**공격**: LLM 서버 운영자가 메트릭 파일을 직접 수정
```python
# 공격자가 시도
metrics_file = "/var/log/vllm/metrics.jsonl"
with open(metrics_file, 'r+') as f:
    data = json.load(f)
    data['prompt_tokens'] = 100  # 실제는 1000이었지만 100으로 변조
    json.dump(data, f)
```

**방어**:
- IMA가 파일 변조를 감지 → PCR 값 변경
- Verifier가 예상 PCR과 비교 → 변조 탐지

#### 시나리오 2: 메트릭 수집 코드 변조
**공격**: 메트릭을 적게 기록하도록 코드 수정

**방어**:
- 메트릭 수집 바이너리의 해시가 IMA에 기록됨
- 코드 변조 시 해시 변경 → PCR 값 변경
- 허용 리스트(allowlist)에 없는 해시 거부

#### 시나리오 3: TPM/Keylime 비활성화
**공격**: Keylime Agent 중지

**방어**:
- Verifier가 Agent의 heartbeat 모니터링
- 응답 없으면 경고 → 해당 서버 메트릭 거부
- Systemd watchdog로 Agent 자동 재시작

#### 시나리오 4: 롤백 공격
**공격**: 이전 시점의 정상 메트릭으로 롤백

**방어**:
- TPM Quote에 타임스탬프 포함
- Nonce (일회용 난수) 사용
- 순차 카운터로 재생 공격 방지

---

### 3.2 신뢰 경계

```
┌─────────────────────────────────────────────────────┐
│              Trusted Computing Base (TCB)           │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ TPM Chip   │  │ BIOS/UEFI  │  │ Bootloader │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│         Measured Components (검증 대상)             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │ OS Kernel  │  │ Keylime    │  │ vLLM       │    │
│  │            │  │ Agent      │  │ Metrics    │    │
│  └────────────┘  └────────────┘  └────────────┘    │
└─────────────────────────────────────────────────────┘
```

**가정**:
- ✅ TPM 칩은 신뢰 가능 (하드웨어 보안)
- ✅ BIOS/Bootloader는 Secure Boot로 보호
- ✅ Keylime Verifier는 신뢰할 수 있는 환경에서 실행
- ⚠️ OS와 애플리케이션은 검증 필요

---

## 4. 아키텍처 설계

### 4.1 전체 시스템 아키텍처

```
┌───────────────────────────────────────────────────────────────┐
│                    Central Control Plane                      │
│  ┌────────────────────┐         ┌────────────────────┐        │
│  │ Keylime Verifier   │◄────────│ Metrics Aggregator │        │
│  │ (무결성 검증)      │         │ (메트릭 수집)      │        │
│  └────────────────────┘         └────────────────────┘        │
│           │                              ▲                     │
│           │ Policy Check                 │ Verified Metrics   │
│           ▼                              │                     │
│  ┌────────────────────┐         ┌────────────────────┐        │
│  │ Keylime Registrar  │         │ Billing System     │        │
│  │ (에이전트 등록)    │         │ (청구 시스템)      │        │
│  └────────────────────┘         └────────────────────┘        │
└───────────────────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┬──────────────┐
        │               │               │              │
        ▼               ▼               ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ LLM Server 1 │ │ LLM Server 2 │ │ LLM Server 3 │ │ LLM Server N │
│              │ │              │ │              │ │              │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │ vLLM     │ │ │ │ vLLM     │ │ │ │ vLLM     │ │ │ │ vLLM     │ │
│ │ Engine   │ │ │ │ Engine   │ │ │ │ Engine   │ │ │ │ Engine   │ │
│ └─────┬────┘ │ │ └─────┬────┘ │ │ └─────┬────┘ │ │ └─────┬────┘ │
│       │      │ │       │      │ │       │      │ │       │      │
│       ▼      │ │       ▼      │ │       ▼      │ │       ▼      │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │ Metrics  │ │ │ │ Metrics  │ │ │ │ Metrics  │ │ │ │ Metrics  │ │
│ │Collector │ │ │ │Collector │ │ │ │Collector │ │ │ │Collector │ │
│ │+ TPM Log │ │ │ │+ TPM Log │ │ │ │+ TPM Log │ │ │ │+ TPM Log │ │
│ └─────┬────┘ │ │ └─────┬────┘ │ │ └─────┬────┘ │ │ └─────┬────┘ │
│       │      │ │       │      │ │       │      │ │       │      │
│       ▼      │ │       ▼      │ │       ▼      │ │       ▼      │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │ Keylime  │ │ │ │ Keylime  │ │ │ │ Keylime  │ │ │ │ Keylime  │ │
│ │ Agent    │ │ │ │ Agent    │ │ │ │ Agent    │ │ │ │ Agent    │ │
│ └─────┬────┘ │ │ └─────┬────┘ │ │ └─────┬────┘ │ │ └─────┬────┘ │
│       │      │ │       │      │ │       │      │ │       │      │
│       ▼      │ │       ▼      │ │       ▼      │ │       ▼      │
│ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │ │ ┌──────────┐ │
│ │   TPM    │ │ │ │   TPM    │ │ │ │   TPM    │ │ │ │   TPM    │ │
│ │PCR 10,16 │ │ │ │PCR 10,16 │ │ │ │PCR 10,16 │ │ │ │PCR 10,16 │ │
│ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │ │ └──────────┘ │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

---

### 4.2 메트릭 수집 플로우

#### Step 1: 메트릭 생성 및 측정

```python
# vllm/llm_metrics_collector/integrity/tpm_logger.py

import hashlib
import json
from typing import Dict, Any
from vllm.llm_metrics_collector.models.request_metrics import RequestMetrics

class TPMMetricsLogger:
    """TPM 기반 메트릭 무결성 로거"""

    def __init__(self, ima_log_path: str = "/sys/kernel/security/ima/ascii_runtime_measurements"):
        self.ima_log_path = ima_log_path
        self.pcr_index = 10  # 사용자 정의 측정용 PCR

    def log_metric_to_ima(self, metrics: RequestMetrics) -> str:
        """메트릭을 IMA에 기록하여 TPM에 측정.

        Args:
            metrics: 기록할 메트릭

        Returns:
            측정된 메트릭의 해시
        """
        # 1. 메트릭을 정규화된 JSON으로 변환 (순서 보장)
        metric_json = json.dumps(
            metrics.to_dict(),
            sort_keys=True,  # 키 순서 고정
            separators=(',', ':')  # 공백 제거
        )

        # 2. SHA-256 해시 계산
        metric_hash = hashlib.sha256(metric_json.encode()).hexdigest()

        # 3. IMA-MEASURE 시스템 콜 (커널이 자동으로 TPM에 extend)
        # 실제로는 커널 API를 통해 수행되지만, 여기서는 파일 쓰기로 트리거
        ima_entry = f"metric_{metrics.request_id}"

        # IMA에 측정 트리거 (실제 구현은 커널 모듈 필요)
        # 여기서는 개념적으로 표현
        self._trigger_ima_measurement(ima_entry, metric_json)

        return metric_hash

    def _trigger_ima_measurement(self, name: str, data: str):
        """IMA 측정 트리거 (실제로는 커널 API 사용)."""
        # 옵션 1: 파일로 저장하여 IMA가 자동 측정하도록
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode='w',
            prefix=f'metrics_{name}_',
            suffix='.json',
            delete=False
        ) as f:
            f.write(data)
            # IMA가 파일 생성을 감지하고 측정

        # 옵션 2: evmctl을 사용하여 직접 측정 (고급)
        # subprocess.run(['evmctl', 'ima_measurement', f.name])
```

#### Step 2: TPM Quote 생성 및 전송

```python
# vllm/llm_metrics_collector/integrity/attestation.py

import subprocess
import json
from typing import Dict, Any

class KeylimeAttestor:
    """Keylime을 통한 원격 증명"""

    def __init__(self, verifier_url: str = "http://verifier:8881"):
        self.verifier_url = verifier_url

    def create_quote(self, pcr_indices: list[int] = [10, 16]) -> Dict[str, Any]:
        """TPM Quote 생성.

        Args:
            pcr_indices: Quote에 포함할 PCR 인덱스

        Returns:
            Quote 데이터 (PCR 값 + TPM 서명)
        """
        # tpm2-tools를 사용하여 Quote 생성
        cmd = [
            'tpm2_quote',
            '-c', '0x81010001',  # AK (Attestation Key) 핸들
            '-l', ','.join(f'sha256:{i}' for i in pcr_indices),
            '-q', 'test_nonce',  # Nonce (재생 공격 방지)
            '-m', '/tmp/quote.msg',
            '-s', '/tmp/quote.sig',
            '-o', '/tmp/quote.pcr',
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"TPM Quote 생성 실패: {result.stderr}")

        # Quote 데이터 읽기
        with open('/tmp/quote.pcr', 'rb') as f:
            pcr_data = f.read()

        with open('/tmp/quote.sig', 'rb') as f:
            signature = f.read()

        return {
            'pcr_values': self._parse_pcr_values(pcr_data),
            'signature': signature.hex(),
            'nonce': 'test_nonce',
            'timestamp': time.time(),
        }

    def send_to_verifier(self, quote: Dict[str, Any], metrics: RequestMetrics):
        """Quote와 메트릭을 Verifier에 전송.

        Args:
            quote: TPM Quote
            metrics: 메트릭 데이터
        """
        payload = {
            'agent_id': self._get_agent_id(),
            'quote': quote,
            'metrics': metrics.to_dict(),
            'ima_log': self._get_ima_log_excerpt(),
        }

        response = requests.post(
            f"{self.verifier_url}/v1/agents/attestation",
            json=payload
        )

        if response.status_code != 200:
            raise RuntimeError(f"Verifier 전송 실패: {response.text}")

    def _get_ima_log_excerpt(self) -> list[str]:
        """최근 IMA 로그 추출."""
        with open('/sys/kernel/security/ima/ascii_runtime_measurements', 'r') as f:
            lines = f.readlines()
            # 최근 100개 항목만
            return lines[-100:]
```

#### Step 3: Verifier에서 검증

```python
# vllm/llm_metrics_collector/integrity/verifier.py

import hashlib
import json
from typing import Dict, Any, List

class MetricsVerifier:
    """Keylime Verifier에서 메트릭 무결성 검증"""

    def __init__(self, allowlist: Dict[str, Any]):
        """
        Args:
            allowlist: 허용된 측정 리스트
                {
                    "hashes": ["sha256:abc123...", "sha256:def456..."],
                    "binaries": ["/usr/bin/vllm", "/opt/metrics_collector.py"]
                }
        """
        self.allowlist = allowlist

    def verify_attestation(
        self,
        agent_id: str,
        quote: Dict[str, Any],
        metrics: RequestMetrics,
        ima_log: List[str]
    ) -> bool:
        """원격 증명 검증.

        Args:
            agent_id: Agent ID
            quote: TPM Quote
            metrics: 메트릭 데이터
            ima_log: IMA 측정 로그

        Returns:
            검증 성공 여부
        """
        # 1. TPM Quote 서명 검증
        if not self._verify_quote_signature(agent_id, quote):
            print(f"❌ Quote 서명 검증 실패: {agent_id}")
            return False

        # 2. IMA 로그 재생 (Replay)
        computed_pcr = self._replay_ima_log(ima_log)

        # 3. 계산된 PCR과 Quote의 PCR 비교
        expected_pcr = quote['pcr_values'].get(10)
        if computed_pcr != expected_pcr:
            print(f"❌ PCR 불일치: computed={computed_pcr}, expected={expected_pcr}")
            return False

        # 4. Allowlist 검증 (허용된 바이너리만 실행되었는가?)
        if not self._verify_allowlist(ima_log):
            print(f"❌ Allowlist 검증 실패: 허용되지 않은 바이너리 실행됨")
            return False

        # 5. 메트릭 데이터 검증 (IMA 로그에 포함되어 있는가?)
        if not self._verify_metric_in_log(metrics, ima_log):
            print(f"❌ 메트릭이 IMA 로그에 없음")
            return False

        print(f"✅ 검증 성공: {agent_id}")
        return True

    def _verify_quote_signature(self, agent_id: str, quote: Dict[str, Any]) -> bool:
        """TPM Quote의 서명 검증.

        Returns:
            서명이 유효하면 True
        """
        # 1. Registrar에서 Agent의 TPM 공개키 가져오기
        tpm_public_key = self._get_tpm_public_key(agent_id)

        # 2. Quote 메시지와 서명 추출
        quote_message = json.dumps(quote['pcr_values'], sort_keys=True).encode()
        signature = bytes.fromhex(quote['signature'])

        # 3. RSA 서명 검증 (실제로는 tpm2-tools 사용)
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        try:
            tpm_public_key.verify(
                signature,
                quote_message,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            return True
        except Exception as e:
            print(f"서명 검증 실패: {e}")
            return False

    def _replay_ima_log(self, ima_log: List[str]) -> str:
        """IMA 로그를 재생하여 PCR 값 계산.

        Args:
            ima_log: IMA 로그 라인들

        Returns:
            계산된 PCR 값 (hex)
        """
        # PCR 초기값 (모두 0)
        pcr_value = b'\x00' * 32  # SHA-256 = 32 bytes

        for line in ima_log:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            pcr_index = int(parts[0])
            template_hash = parts[1]
            file_hash = parts[3]

            # PCR 10만 처리 (우리가 사용하는 인덱스)
            if pcr_index != 10:
                continue

            # PCR Extend 연산: PCR_new = SHA256(PCR_old || measurement)
            measurement = bytes.fromhex(file_hash.split(':')[1])
            pcr_value = hashlib.sha256(pcr_value + measurement).digest()

        return pcr_value.hex()

    def _verify_allowlist(self, ima_log: List[str]) -> bool:
        """Allowlist에 있는 바이너리만 실행되었는지 검증."""
        for line in ima_log:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            file_hash = parts[3]
            file_path = parts[4] if len(parts) > 4 else ""

            # 파일 해시가 allowlist에 있는지 확인
            if file_hash not in self.allowlist['hashes']:
                # 또는 파일 경로가 허용 목록에 있는지 확인
                if not any(allowed in file_path for allowed in self.allowlist['binaries']):
                    print(f"⚠️ 허용되지 않은 파일 실행: {file_path} ({file_hash})")
                    return False

        return True

    def _verify_metric_in_log(self, metrics: RequestMetrics, ima_log: List[str]) -> bool:
        """메트릭이 IMA 로그에 기록되었는지 확인."""
        # 메트릭의 해시 계산
        metric_json = json.dumps(metrics.to_dict(), sort_keys=True, separators=(',', ':'))
        metric_hash = hashlib.sha256(metric_json.encode()).hexdigest()

        # IMA 로그에서 해당 해시 검색
        for line in ima_log:
            if f"sha256:{metric_hash}" in line:
                return True

        return False
```

---

### 4.3 통합된 Collector

```python
# vllm/llm_metrics_collector/collectors/attested_collector.py

from vllm.llm_metrics_collector.collectors.vllm_collector import VLLMMetricsCollector
from vllm.llm_metrics_collector.integrity.tpm_logger import TPMMetricsLogger
from vllm.llm_metrics_collector.integrity.attestation import KeylimeAttestor

class AttestedMetricsCollector(VLLMMetricsCollector):
    """무결성이 보장된 메트릭 수집기."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.tpm_logger = TPMMetricsLogger()
        self.attestor = KeylimeAttestor()

    def collect_with_attestation(
        self,
        finished_stats,
        **kwargs
    ) -> RequestMetrics:
        """메트릭 수집 + TPM 측정 + 원격 증명.

        Returns:
            무결성이 보장된 RequestMetrics
        """
        # 1. 기본 메트릭 수집
        metrics = self.collect(finished_stats=finished_stats, **kwargs)

        # 2. TPM에 측정 기록
        metric_hash = self.tpm_logger.log_metric_to_ima(metrics)
        metrics.measurement_hash = metric_hash

        # 3. TPM Quote 생성
        quote = self.attestor.create_quote(pcr_indices=[10, 16])

        # 4. Verifier에 전송
        self.attestor.send_to_verifier(quote, metrics)

        return metrics
```

---

## 5. 구현 가이드

### 5.1 환경 설정

#### Step 1: TPM 확인

```bash
# TPM 2.0 존재 확인
ls /dev/tpm0

# TPM 정보 확인
tpm2_getcap properties-fixed

# PCR 값 읽기
tpm2_pcrread sha256:10,16
```

#### Step 2: IMA 활성화

```bash
# 1. GRUB 설정 수정
sudo vim /etc/default/grub

# 다음 추가:
GRUB_CMDLINE_LINUX="ima_policy=tcb ima_appraise=fix ima_template=ima-ng"

# 2. GRUB 업데이트
sudo update-grub
sudo reboot

# 3. IMA 활성화 확인
cat /sys/kernel/security/ima/ascii_runtime_measurements | head
```

#### Step 3: Keylime 설치

```bash
# Python 환경
pip install keylime

# 또는 시스템 패키지 (Ubuntu/Debian)
sudo apt install keylime-agent keylime-verifier keylime-registrar

# 서비스 시작
sudo systemctl start keylime_agent
sudo systemctl start keylime_verifier
sudo systemctl start keylime_registrar
```

**공식 문서**: https://keylime.dev/

---

### 5.2 Agent 등록

```bash
# 1. Agent를 Registrar에 등록
keylime_tenant -c add -t <AGENT_IP> -v <VERIFIER_IP>

# 2. Allowlist 생성 (허용할 바이너리 해시)
keylime_create_policy \
  -m /sys/kernel/security/ima/ascii_runtime_measurements \
  -o /etc/keylime/allowlist.json

# 3. Policy 적용
keylime_tenant -c update -t <AGENT_IP> --allowlist /etc/keylime/allowlist.json
```

---

### 5.3 프로덕션 배포

#### Docker Compose 예시

```yaml
# docker-compose.yml

version: '3.8'

services:
  # Keylime Verifier (중앙 서버)
  verifier:
    image: keylime/keylime_verifier:latest
    ports:
      - "8881:8881"
    environment:
      - KEYLIME_DATABASE_URL=postgresql://keylime:password@db:5432/verifier
    volumes:
      - ./policies:/etc/keylime/policies
    depends_on:
      - db

  # Keylime Registrar
  registrar:
    image: keylime/keylime_registrar:latest
    ports:
      - "8890:8890"
    environment:
      - KEYLIME_DATABASE_URL=postgresql://keylime:password@db:5432/registrar
    depends_on:
      - db

  # PostgreSQL
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=keylime
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=verifier
    volumes:
      - keylime_db:/var/lib/postgresql/data

  # Metrics Aggregator
  aggregator:
    build: ./aggregator
    environment:
      - VERIFIER_URL=http://verifier:8881
    depends_on:
      - verifier

volumes:
  keylime_db:
```

#### LLM 서버 (Agent 포함)

```dockerfile
# Dockerfile

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# vLLM 설치
RUN pip install vllm

# Keylime Agent 설치
RUN apt-get update && apt-get install -y \
    tpm2-tools \
    keylime-agent

# 메트릭 수집기 복사
COPY vllm/llm_metrics_collector /opt/vllm/llm_metrics_collector

# Keylime Agent 설정
COPY keylime-agent.conf /etc/keylime/agent.conf

# 시작 스크립트
COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

CMD ["/opt/start.sh"]
```

```bash
# start.sh

#!/bin/bash

# Keylime Agent 시작
keylime_agent &

# vLLM 서버 시작 (메트릭 수집 활성화)
python -m vllm.entrypoints.api_server \
  --model gpt2 \
  --enable-metrics \
  --metrics-collector attested

wait
```

---

## 6. 보안 검증

### 6.1 시험 시나리오

#### 테스트 1: 정상 케이스
```bash
# 정상 메트릭 수집
curl -X POST http://llm-server:8000/v1/completions \
  -d '{"prompt": "Hello", "max_tokens": 10}'

# Verifier 확인
curl http://verifier:8881/v1/agents | jq '.agents[] | select(.agent_id=="server1")'

# 결과: operational_state = "Get Quote"
```

#### 테스트 2: 메트릭 조작 시도
```bash
# Agent 서버에서 메트릭 파일 변조
echo '{"prompt_tokens": 1}' > /var/log/vllm/metrics.jsonl

# Verifier 확인
curl http://verifier:8881/v1/agents/server1/status

# 결과: operational_state = "Failed"
# 이유: PCR 값이 예상과 다름 (IMA가 변조 감지)
```

#### 테스트 3: 코드 변조 시도
```bash
# 메트릭 수집 코드 수정
vim /opt/vllm/llm_metrics_collector/collectors/vllm_collector.py
# (토큰 카운트를 절반으로 변경)

# 재시작
systemctl restart vllm

# Verifier 확인
# 결과: operational_state = "Failed"
# 이유: 바이너리 해시가 allowlist와 다름
```

#### 테스트 4: Keylime Agent 비활성화
```bash
# Agent 중지
systemctl stop keylime_agent

# Verifier 확인 (30초 후)
# 결과: operational_state = "Failed" (Timeout)
# 액션: 해당 서버의 메트릭 거부
```

---

### 6.2 감사 로그

```python
# vllm/llm_metrics_collector/integrity/audit.py

class AuditLogger:
    """무결성 검증 감사 로그"""

    def log_verification_result(
        self,
        agent_id: str,
        metrics: RequestMetrics,
        verification_result: bool,
        reason: str = ""
    ):
        """검증 결과 기록.

        Args:
            agent_id: Agent ID
            metrics: 메트릭
            verification_result: 검증 성공 여부
            reason: 실패 시 이유
        """
        audit_entry = {
            'timestamp': time.time(),
            'agent_id': agent_id,
            'request_id': metrics.request_id,
            'verification_result': verification_result,
            'reason': reason,
            'metrics_hash': hashlib.sha256(
                json.dumps(metrics.to_dict(), sort_keys=True).encode()
            ).hexdigest(),
        }

        # 변조 불가능한 로그 저장 (예: 블록체인, immutable DB)
        self._write_to_audit_log(audit_entry)

    def _write_to_audit_log(self, entry: dict):
        """감사 로그 저장 (append-only)."""
        with open('/var/log/keylime/audit.jsonl', 'a') as f:
            f.write(json.dumps(entry) + '\n')
```

---

## 7. 대안 및 비교

### 7.1 다른 무결성 보장 방법

| 방법 | 장점 | 단점 | 비용 |
|------|------|------|------|
| **Keylime + TPM** | ✅ 하드웨어 기반<br>✅ 변조 불가능<br>✅ 실시간 모니터링 | ⚠️ TPM 칩 필요<br>⚠️ 설정 복잡 | 중간 |
| **Blockchain** | ✅ 분산 원장<br>✅ 변조 불가능 | ❌ 높은 레이턴시<br>❌ 확장성 한계 | 높음 |
| **Digital Signatures** | ✅ 간단한 구현<br>✅ 표준 암호 | ⚠️ 키 관리 필요<br>⚠️ 코드 무결성 미보장 | 낮음 |
| **Secure Enclaves** (SGX, TDX) | ✅ 메모리 암호화<br>✅ 코드 무결성 | ❌ Intel CPU만<br>❌ 복잡도 높음 | 중간 |
| **Trusted Execution** (AWS Nitro) | ✅ 클라우드 통합<br>✅ 관리 불필요 | ⚠️ 클라우드 종속<br>❌ 비용 | 높음 |

**추천**:
- 온프레미스 → **Keylime + TPM**
- 클라우드 → **AWS Nitro Enclaves** 또는 **GCP Confidential VMs**
- 간단한 케이스 → **Digital Signatures**

---

### 7.2 Keylime의 제약사항

**제약 1: TPM 필요**
- 해결: 가상 TPM (vTPM) 사용 가능
- 클라우드: AWS, GCP, Azure 모두 vTPM 지원

**제약 2: 성능 오버헤드**
- PCR Extend: ~1ms
- Quote 생성: ~10ms
- 영향: 요청당 < 20ms (무시 가능)

**제약 3: Allowlist 관리**
- 소프트웨어 업데이트 시마다 allowlist 갱신 필요
- 해결: CI/CD에 자동화 통합

---

### 7.3 하이브리드 접근

**최소 보장 (Tier 1)**:
- Digital signature로 메트릭 서명
- 간단하지만 코드 변조 불가능

**중간 보장 (Tier 2)**:
- Keylime + TPM
- 코드 및 데이터 무결성 보장

**최대 보장 (Tier 3)**:
- Keylime + Confidential Computing (SGX/TDX)
- 메모리 내 데이터도 암호화

---

## 8. 참고 자료

### 공식 문서
- **Keylime**: https://keylime.dev/
- **GitHub**: https://github.com/keylime/keylime
- **User Guide**: https://keylime.readthedocs.io/

### 관련 기술
- **TPM 2.0**: https://trustedcomputinggroup.org/resource/tpm-library-specification/
- **IMA**: https://sourceforge.net/p/linux-ima/wiki/Home/
- **tpm2-tools**: https://github.com/tpm2-software/tpm2-tools

### 논문
- "Remote Attestation: A Survey" (2018)
- "Keylime: Practical Challenge-Response based Attestation for the Cloud" (2020)

---

## 9. 결론

### 9.1 Keylime으로 가능한 것

✅ **메트릭 조작 방지**: TPM 기반 측정으로 변조 불가능
✅ **코드 무결성**: 메트릭 수집 바이너리 검증
✅ **실시간 모니터링**: 지속적인 증명으로 실시간 검증
✅ **감사 추적**: 완전한 측정 로그
✅ **분산 확장**: 수천 대 서버 지원

### 9.2 구현 우선순위

**Phase 1** (필수):
1. TPM/IMA 설정
2. 기본 Keylime Agent 배포
3. 메트릭 측정 통합

**Phase 2** (권장):
1. Allowlist 관리 자동화
2. 감사 로그 시스템
3. 모니터링 대시보드

**Phase 3** (고급):
1. Confidential Computing 통합
2. 블록체인 감사 로그
3. 자동 인시던트 대응

### 9.3 최종 판단

**Keylime + TPM은 분산 LLM 메트릭 무결성 보장에 매우 적합합니다.**

- ✅ 하드웨어 기반 보안으로 강력한 보장
- ✅ CNCF 프로젝트로 커뮤니티 지원
- ✅ 실제 프로덕션 사용 사례 다수 (Red Hat, IBM 등)
- ⚠️ 초기 설정은 복잡하지만 자동화 가능
- ⚠️ TPM 필요 (하지만 대부분의 현대 서버 지원)

**권장**: 중요한 청구 시스템이라면 반드시 구현할 가치가 있습니다!
