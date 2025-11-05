# vLLM + BFT Verifier 통합 가이드

vLLM Metrics Collector와 BFT Verifier PoC를 파일 기반으로 통합하는 방법입니다.

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                      LLM 서버 (예: GPU 서버)                     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ vLLM 프로세스 (Python)                                     │ │
│  │   • LLM 추론 수행                                          │ │
│  │   • Metrics Collector 내장                                │ │
│  │   • 요청 완료 시 metrics 수집                             │ │
│  └──────────────┬─────────────────────────────────────────────┘ │
│                 │ metrics.to_json()                              │
│                 ▼                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Pending Queue (/var/log/vllm/metrics/)                     │ │
│  │   pending/                                                  │ │
│  │     req-123.jsonl  ← 새 파일 생성                         │ │
│  │     req-456.jsonl                                           │ │
│  └──────────────┬─────────────────────────────────────────────┘ │
│                 │ 파일 모니터링 (1초마다 폴링)                   │
│                 ▼                                                 │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ BFT Agent 프로세스 (Rust)                                 │ │
│  │   • pending/ 디렉토리 모니터링                            │ │
│  │   • .jsonl 파일 읽기                                      │ │
│  │   • TPM Quote 생성 (현재: Ed25519 시뮬레이션)            │ │
│  │   • Coordinator에게 gRPC 전송                             │ │
│  └──────────────┬─────────────────────────────────────────────┘ │
│                 │                                                 │
│                 ├─ 성공 → sent/req-123_20251105_123456.jsonl    │
│                 └─ 실패 → failed/req-123_20251105_123500.jsonl  │
└─────────────────┼─────────────────────────────────────────────────┘
                  │ gRPC (http://coordinator:50051)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   중앙 Coordinator 서버                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Coordinator (Rust)                                         │ │
│  │   • VRF 기반 위원회 선택 (3/10 verifiers)                │ │
│  │   • 병렬 투표 수집                                        │ │
│  │   • BFT 합의 (2f+1 quorum)                               │ │
│  │   • 결과 반환                                             │ │
│  └──────────────┬─────────────────────────────────────────────┘ │
└─────────────────┼─────────────────────────────────────────────────┘
                  │
        ┌─────────┼─────────┬─────────┐
        ▼         ▼         ▼         ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │Verifier1││Verifier2││Verifier3││Verifier4│
   │  (Rust) ││  (Rust) ││  (Rust) ││  (Rust) │
   │  검증   ││  검증   ││  검증   ││  검증   │
   │  투표   ││  투표   ││  투표   ││  투표   │
   └─────────┘└─────────┘└─────────┘└─────────┘
```

## 데이터 흐름

### 1. vLLM에서 Metrics 수집 (Python)

```python
from vllm.llm_metrics_collector import (
    VLLMMetricsCollector,
    PendingQueueStorage,
    TokenBasedPricing,
)

# 초기화 (vLLM 서버 시작 시 1회)
collector = VLLMMetricsCollector(
    model_name="llama-2-7b",
    engine_id="vllm-gpu-1"
)

pricing = TokenBasedPricing(
    input_price_per_1k=0.0005,
    output_price_per_1k=0.0015,
)

# Pending queue storage (BFT Agent가 모니터링할 디렉토리)
queue = PendingQueueStorage("/var/log/vllm/metrics")

# 각 요청 완료 시 호출
def on_request_finished(finished_stats, request_id, user_id):
    # 1. Metrics 수집
    metrics = collector.collect(
        finished_stats=finished_stats,
        request_id=request_id,
        user_id=user_id,
        arrival_time=time.time()
    )

    # 2. Cost 계산
    metrics = collector.enrich(metrics, pricing_calculator=pricing)

    # 3. Pending queue에 저장 (BFT Agent가 읽어갈 파일)
    file_path = queue.enqueue(metrics)
    # → /var/log/vllm/metrics/pending/req-123.jsonl 생성

    # 4. 사용자에게 즉시 응답 반환 (검증은 백그라운드에서 진행)
    return response_to_user
```

**생성되는 파일 예시** (`/var/log/vllm/metrics/pending/req-123.jsonl`):
```json
{"request_id": "req-123", "user_id": "user-456", "model_name": "llama-2-7b", "prompt_tokens": 150, "completion_tokens": 50, "cached_tokens": 20, "e2e_latency": 2.345, "time_to_first_token": 0.123, "estimated_cost": 0.00012}
```

### 2. BFT Agent가 파일 모니터링 (Rust)

```bash
# BFT Agent 실행 (file_monitor 모드)
MODE=file_monitor \
FILE_MONITOR_DIR=/var/log/vllm/metrics \
ASYNC_MODE=true \
AGENT_ID=agent-gpu-1 \
COORDINATOR_ENDPOINT=http://coordinator:50051 \
./target/release/agent
```

**동작:**
1. **1초마다** `pending/` 디렉토리 스캔
2. `.jsonl` 파일 발견 시 읽기
3. JSON 파싱 → `RequestMetrics` 구조체로 변환
4. TPM Quote 생성 (Ed25519 서명)
5. Coordinator에게 gRPC로 전송
6. **성공 시**: `sent/` 디렉토리로 이동
7. **실패 시**: `failed/` 디렉토리로 이동 (에러 메시지 포함)

### 3. Coordinator가 BFT 합의 수행

1. VRF로 검증자 3명 선택 (예: V1, V2, V4)
2. 병렬로 3개 검증자에게 검증 요청
3. 투표 수집 (2f+1 = 2개 동의 필요)
4. **Early termination**: 2개 투표 도착 즉시 합의 완료
5. Agent에게 결과 반환 (ACCEPTED / REJECTED)

### 4. 파일 상태 변화

```
초기:
  pending/req-123.jsonl  ← Agent가 읽음

전송 성공:
  sent/req-123_20251105_143022.jsonl  ← 타임스탬프 추가

전송 실패:
  failed/req-123_20251105_143025.jsonl
  내용:
    {"request_id": "req-123", ...}
    {"error": "Coordinator connection failed", "failed_at": "20251105_143025"}
```

## 설치 및 배포

### 사전 준비

1. **vLLM 서버** (Python 환경)
2. **BFT 컴포넌트** (Rust 바이너리)
   - Coordinator (중앙 서버 1대)
   - Verifiers (분산 서버 4+대)
   - Agent (각 vLLM 서버마다 1개)

### 1단계: vLLM 서버에 Metrics Collector 통합

```python
# vllm_server.py
import time
from vllm import LLM, SamplingParams
from vllm.llm_metrics_collector import (
    VLLMMetricsCollector,
    PendingQueueStorage,
    TokenBasedPricing,
)

# 초기화
llm = LLM(model="llama-2-7b")
collector = VLLMMetricsCollector(model_name="llama-2-7b", engine_id="vllm-1")
pricing = TokenBasedPricing(input_price_per_1k=0.0005, output_price_per_1k=0.0015)
queue = PendingQueueStorage("/var/log/vllm/metrics")

# 요청 처리
def generate(prompt, request_id, user_id):
    arrival_time = time.time()

    # LLM 추론
    outputs = llm.generate(prompt, SamplingParams(temperature=0.7, max_tokens=100))

    # Metrics 수집 및 저장 (실제로는 finished_stats 사용)
    # metrics = collector.collect(finished_stats=..., request_id=request_id, ...)
    # metrics = collector.enrich(metrics, pricing_calculator=pricing)
    # queue.enqueue(metrics)  # → pending/req-123.jsonl 생성

    return outputs[0].text
```

### 2단계: BFT Agent 설치 (각 vLLM 서버)

```bash
# Agent 바이너리를 vLLM 서버에 복사
scp target/release/agent vllm-server:/usr/local/bin/bft-agent

# Systemd 서비스 등록
sudo tee /etc/systemd/system/bft-agent.service <<EOF
[Unit]
Description=BFT Agent for vLLM Metrics Verification
After=network.target

[Service]
Type=simple
User=vllm
Environment="RUST_LOG=info"
Environment="MODE=file_monitor"
Environment="AGENT_ID=agent-vllm-1"
Environment="COORDINATOR_ENDPOINT=http://coordinator.example.com:50051"
Environment="FILE_MONITOR_DIR=/var/log/vllm/metrics"
Environment="ASYNC_MODE=true"
ExecStart=/usr/local/bin/bft-agent
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 시작
sudo systemctl daemon-reload
sudo systemctl enable bft-agent
sudo systemctl start bft-agent

# 로그 확인
sudo journalctl -u bft-agent -f
```

### 3단계: Coordinator 및 Verifiers 설치

```bash
# Coordinator (중앙 서버)
cd bft_verifier_poc
cargo build --release --package bft-coordinator

# 실행
LISTEN_ADDR=0.0.0.0:50051 \
COMMITTEE_SIZE=3 \
QUORUM=2 \
./target/release/coordinator

# Verifiers (각 분산 서버)
VERIFIER_ID=verifier-1 \
LISTEN_ADDR=0.0.0.0:50052 \
./target/release/verifier
```

### 4단계: Docker Compose로 테스트 (권장)

```bash
# 전체 시스템 테스트
cd bft_verifier_poc
docker-compose up --build

# 포함 컴포넌트:
# - Coordinator (1개)
# - Verifiers (4개)
# - Agent (1개, 테스트용)
```

## 모니터링 및 관리

### 큐 상태 확인

```python
# Python에서 큐 통계 확인
queue = PendingQueueStorage("/var/log/vllm/metrics")
stats = queue.get_stats()

print(f"Pending: {stats['pending']}")   # 아직 전송 안된 파일
print(f"Sent: {stats['sent']}")         # 전송 완료 파일
print(f"Failed: {stats['failed']}")     # 전송 실패 파일
```

### 오래된 파일 정리

```python
# Sent 파일 정리 (7일 이상)
deleted = queue.clear_sent(older_than_days=7)
print(f"Deleted {deleted} sent files")

# Failed 파일 정리 (30일 이상)
deleted = queue.clear_failed(older_than_days=30)
print(f"Deleted {deleted} failed files")
```

### Agent 로그 확인

```bash
# Systemd 로그
sudo journalctl -u bft-agent -f

# 출력 예시:
# [INFO] File monitor initialized
# [INFO] Processing file: pending/req-123.jsonl (request_id: req-123)
# [INFO] Read metrics from file: prompt_tokens=150, cost=0.00012
# [INFO] Submitted (async - verification in background)
# [INFO] File moved to sent
```

## 성능 특성

### 지연 시간 (Latency)

| 단계 | 지연 시간 | 비고 |
|------|----------|------|
| vLLM 추론 | 1-10초 | 모델 크기에 따라 |
| Metrics 수집 | < 1ms | 메모리 복사 |
| 파일 저장 | < 5ms | SSD 기준 |
| **사용자 응답** | **추론 + 6ms** | **검증 대기 없음** |
| Agent 파일 감지 | 0-1초 | 폴링 간격 |
| BFT 검증 | 50-100ms | 3개 검증자, early termination |
| 파일 이동 | < 1ms | |

**핵심**: 사용자는 검증 완료를 기다리지 않음 (0ms 추가 대기)

### 처리량 (Throughput)

- **vLLM**: 모델 성능에 따라 (예: 10 req/sec)
- **Agent**: 최대 10 파일/초 처리 (batch_size=10, poll=1초)
- **Coordinator**: 100+ req/sec (병렬 처리)

### 저장 공간

- **Pending**: 일시적 (보통 < 100 파일)
- **Sent**: 1 req = ~200 bytes → 1M req = 200 MB
- **Failed**: 예외 상황만 (보통 < 10 파일)

**권장**: 주기적으로 sent 정리 (7일마다)

## 장애 처리

### Scenario 1: Agent 다운

**상황**: BFT Agent 프로세스 종료

**영향**:
- ✅ vLLM 정상 동작 (파일은 계속 생성됨)
- ⚠️ Metrics가 pending에 누적
- ❌ 검증 중단

**복구**:
```bash
# Agent 재시작
sudo systemctl restart bft-agent

# → pending 파일들을 자동으로 처리 시작
```

### Scenario 2: Coordinator 다운

**상황**: 중앙 Coordinator 서버 다운

**영향**:
- ✅ vLLM 정상 동작
- ⚠️ Agent는 전송 실패 → failed로 이동
- ❌ 검증 불가

**복구**:
```bash
# Coordinator 재시작
# Failed 파일을 pending으로 복구
mv /var/log/vllm/metrics/failed/*.jsonl /var/log/vllm/metrics/pending/

# → Agent가 재전송
```

### Scenario 3: 디스크 풀

**상황**: `/var/log/vllm/metrics/` 디스크 풀

**영향**:
- ❌ vLLM 새 요청 실패 (파일 저장 불가)

**예방**:
```bash
# Cron으로 자동 정리 (매일 실행)
0 0 * * * /usr/local/bin/cleanup_metrics.py

# cleanup_metrics.py
queue = PendingQueueStorage("/var/log/vllm/metrics")
queue.clear_sent(older_than_days=7)
queue.clear_failed(older_than_days=30)
```

## 보안 고려사항

### 현재 구현 (Ed25519 시뮬레이션)

- ✅ Metrics 무결성 보장 (서명 검증)
- ⚠️ 소프트웨어 기반 (TPM 없음)
- ⚠️ Agent 손상 시 서명키 탈취 가능

### 향후 개선 (실제 TPM)

- ✅ 하드웨어 기반 보안
- ✅ 서명키 탈취 불가
- ✅ 시스템 무결성 검증 (IMA)

**구현 방법**: `bft_agent`에서 `tpm2-tss` Rust 크레이트 사용

## 환경변수 레퍼런스

### Agent 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `MODE` | `continuous` | 동작 모드: `single`, `continuous`, `file_monitor` |
| `AGENT_ID` | `agent-1` | Agent 고유 ID |
| `COORDINATOR_ENDPOINT` | `http://localhost:50051` | Coordinator 주소 |
| `ASYNC_MODE` | `false` | 비동기 모드 (true 권장) |
| `FILE_MONITOR_DIR` | `/var/log/vllm/metrics` | 큐 디렉토리 |
| `FILE_MONITOR_POLL_SECS` | `1` | 폴링 간격 (초) |
| `FILE_MONITOR_BATCH_SIZE` | `10` | 한 번에 처리할 파일 수 |

### Coordinator 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `LISTEN_ADDR` | `0.0.0.0:50051` | 리슨 주소 |
| `COMMITTEE_SIZE` | `3` | 위원회 크기 |
| `QUORUM` | `2` | 합의 정족수 (2f+1) |
| `VOTE_TIMEOUT_SECS` | `5` | 투표 타임아웃 |
| `EPOCH_DURATION_SECS` | `30` | Epoch 주기 |

## 예제: 전체 통합 테스트

```python
# test_integration.py
import time
from vllm.llm_metrics_collector import (
    VLLMMetricsCollector,
    PendingQueueStorage,
    TokenBasedPricing,
)
from vllm.llm_metrics_collector.models import RequestMetrics
from vllm.llm_metrics_collector.models.enums import FinishReason

# 1. Setup
collector = VLLMMetricsCollector(model_name="test-model", engine_id="test-engine")
pricing = TokenBasedPricing(input_price_per_1k=0.0005, output_price_per_1k=0.0015)
queue = PendingQueueStorage("/tmp/test_metrics")

# 2. Generate test metrics
metrics = RequestMetrics(
    request_id="test-req-1",
    user_id="test-user",
    model_name="test-model",
    prompt_tokens=100,
    completion_tokens=50,
    cached_tokens=10,
    e2e_latency=1.234,
    time_to_first_token=0.123,
    estimated_cost=0.0,
    arrival_time=time.time(),
    finish_time=time.time() + 1.234,
    finish_reason=FinishReason.COMPLETED,
)

# 3. Enrich with cost
metrics = collector.enrich(metrics, pricing_calculator=pricing)
print(f"Cost: ${metrics.estimated_cost:.6f}")

# 4. Enqueue (BFT Agent will pick this up)
file_path = queue.enqueue(metrics)
print(f"Created: {file_path}")

# 5. Check stats
stats = queue.get_stats()
print(f"Pending: {stats['pending']}, Sent: {stats['sent']}, Failed: {stats['failed']}")

# 6. Wait for Agent to process
time.sleep(5)

# 7. Check again
stats = queue.get_stats()
print(f"After processing - Pending: {stats['pending']}, Sent: {stats['sent']}")
```

```bash
# BFT Agent 실행 (별도 터미널)
MODE=file_monitor \
FILE_MONITOR_DIR=/tmp/test_metrics \
ASYNC_MODE=true \
COORDINATOR_ENDPOINT=http://localhost:50051 \
./target/release/agent

# 테스트 실행
python test_integration.py

# 출력:
# Cost: $0.000105
# Created: /tmp/test_metrics/pending/test-req-1.jsonl
# Pending: 1, Sent: 0, Failed: 0
# (5초 대기...)
# After processing - Pending: 0, Sent: 1
```

## 트러블슈팅

### 문제: Pending 파일이 처리되지 않음

**확인사항**:
1. Agent가 실행 중인지: `systemctl status bft-agent`
2. 디렉토리 권한: `ls -la /var/log/vllm/metrics/pending/`
3. Agent 로그: `journalctl -u bft-agent -n 50`

**해결**:
```bash
# 권한 수정
sudo chown -R vllm:vllm /var/log/vllm/metrics/
sudo chmod 755 /var/log/vllm/metrics/pending/
```

### 문제: Failed 파일이 계속 생성됨

**원인**: Coordinator 연결 실패

**확인**:
```bash
# Coordinator 연결 테스트
curl -v http://coordinator:50051
# 또는
grpcurl -plaintext coordinator:50051 list
```

**해결**:
```bash
# Coordinator 재시작
# 네트워크 방화벽 확인
sudo ufw allow 50051/tcp
```

### 문제: 성능 저하

**확인**:
```python
queue = PendingQueueStorage("/var/log/vllm/metrics")
stats = queue.get_stats()

if stats['pending'] > 100:
    print("⚠️ Pending queue backlog!")
    # Agent 성능 부족 or Coordinator 과부하
```

**해결**:
```bash
# Agent 병렬 처리 증가
FILE_MONITOR_BATCH_SIZE=20 \
FILE_MONITOR_POLL_SECS=0.5 \
./target/release/agent

# 또는 Agent 수평 확장 (여러 개 실행)
```

## 다음 단계

1. ✅ **현재**: 파일 기반 통합 완료
2. 🔜 **단기**: 실제 TPM 통합 (`tpm2-tss` 사용)
3. 🔜 **중기**: Keylime 통합 (시스템 무결성 검증)
4. 🔜 **장기**: 프로덕션 배포 (Kubernetes, 모니터링)

## 참고 자료

- [vLLM Metrics Collector README](vllm/llm_metrics_collector/README.md)
- [BFT Verifier PoC README](bft_verifier_poc/README.md)
- [BFT Architecture](bft_verifier_poc/ARCHITECTURE.md)
- [Performance Optimizations](bft_verifier_poc/PERFORMANCE_OPTIMIZATIONS.md)
- [Keylime Integration Design](vllm/llm_metrics_collector/KEYLIME_INTEGRITY.md)
