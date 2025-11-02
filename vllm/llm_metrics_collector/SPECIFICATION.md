# LLM Metrics Collection System - Specification

## 1. 목적 및 개요 (Purpose & Overview)

### 1.1 목적
LLM 서비스의 사용량을 정확히 측정하고 객관적인 가격 책정(pricing)을 위한 메트릭 수집 시스템을 구축합니다.

### 1.2 핵심 목표
- **요청당(per-request) 상세 메트릭 수집**: 모든 요청의 자원 사용량을 추적
- **가격 책정 가능한 객관적 지표**: 토큰 사용량, 처리 시간, 자원 사용량 등
- **사용량 분석 및 리포팅**: 사용자별, 모델별, 기간별 집계 기능
- **확장성**: vLLM의 기존 메트릭 시스템과 통합 가능한 구조
- **표준화**: OpenAI API 호환 사용량 포맷 지원

---

## 2. 수집할 메트릭 카테고리

### 2.1 토큰 사용량 메트릭 (Token Usage Metrics)
**가격 책정의 핵심 지표**

#### 2.1.1 입력 토큰 (Input Tokens)
- `prompt_tokens`: 실제 처리된 프롬프트 토큰 수
- `cached_tokens`: 캐시에서 재사용된 토큰 수 (할인 가능)
- `uncached_tokens`: 실제 계산이 필요했던 토큰 수
- `system_tokens`: 시스템 프롬프트 토큰 수
- `user_tokens`: 사용자 입력 토큰 수

#### 2.1.2 출력 토큰 (Output Tokens)
- `completion_tokens`: 생성된 토큰 수
- `reasoning_tokens`: 추론 단계 토큰 (특정 모델용)
- `truncated_tokens`: max_tokens로 인한 잘린 토큰 수

#### 2.1.3 총 토큰 (Total Tokens)
- `total_tokens`: prompt_tokens + completion_tokens
- `billable_tokens`: 실제 과금 대상 토큰 (캐시 할인 적용)

**가격 책정 공식 예시**:
```
cost = (uncached_tokens × input_price_per_1k) + (completion_tokens × output_price_per_1k)
```

---

### 2.2 시간 메트릭 (Timing Metrics)
**성능 기반 가격 책정 및 SLA 측정**

#### 2.2.1 요청 레이턴시 (Request Latency)
- `arrival_time`: 요청 도착 시각 (wall-clock timestamp)
- `queued_time`: 큐 대기 시간 (초)
- `prefill_time`: Prefill 단계 처리 시간 (초)
- `decode_time`: Decode 단계 처리 시간 (초)
- `inference_time`: 총 추론 시간 = prefill + decode (초)
- `e2e_latency`: 종단간 레이턴시 (초)

#### 2.2.2 토큰 생성 속도
- `time_to_first_token` (TTFT): 첫 토큰까지 시간 (초)
- `time_per_output_token` (TPOT): 평균 토큰 생성 시간 (초)
- `inter_token_latency`: 토큰 간 지연 (초)
- `tokens_per_second`: 처리 속도 (토큰/초)

**프리미엄 가격 책정**:
- TTFT < 1초: 프리미엄 요금 (실시간 응답)
- TTFT > 5초: 표준 요금 (배치 처리)

---

### 2.3 자원 사용량 메트릭 (Resource Usage Metrics)
**자원 기반 가격 책정**

#### 2.3.1 메모리 사용량
- `kv_cache_blocks_used`: 사용된 KV 캐시 블록 수
- `kv_cache_memory_bytes`: KV 캐시 메모리 사용량 (바이트)
- `peak_gpu_memory_bytes`: 피크 GPU 메모리 사용량 (바이트)
- `gpu_memory_seconds`: GPU 메모리 사용량 × 시간 (GB·초)

#### 2.3.2 계산 자원
- `gpu_compute_time`: GPU 계산 시간 (초)
- `batch_size`: 배치 처리 시 크기 (효율성 지표)
- `num_preemptions`: 선점(preemption) 횟수 (우선순위 반영)

#### 2.3.3 특수 자원
- `lora_adapter_name`: 사용된 LoRA 어댑터 (추가 과금 가능)
- `spec_decode_tokens`: Speculative decoding으로 절약된 토큰
- `prefix_cache_hit_rate`: 프리픽스 캐시 히트율 (효율성 지표)

---

### 2.4 요청 메타데이터 (Request Metadata)
**분석 및 필터링용**

#### 2.4.1 식별 정보
- `request_id`: 고유 요청 ID
- `user_id`: 사용자 식별자 (청구서 발행용)
- `api_key_hash`: API 키 해시 (사용량 추적용)
- `organization_id`: 조직 ID (그룹 과금용)

#### 2.4.2 모델 정보
- `model_name`: 사용된 모델 이름
- `model_version`: 모델 버전
- `engine_id`: vLLM 엔진 인스턴스 ID

#### 2.4.3 요청 파라미터
- `max_tokens`: 요청한 최대 토큰 수
- `temperature`: 샘플링 온도
- `top_p`, `top_k`: 샘플링 파라미터
- `n`: 병렬 생성 개수 (multiple choices)
- `stream`: 스트리밍 여부

#### 2.4.4 종료 정보
- `finish_reason`: 종료 사유 (STOP, LENGTH, ABORT)
- `error_code`: 에러 발생 시 코드
- `error_message`: 에러 메시지

---

### 2.5 비즈니스 메트릭 (Business Metrics)
**수익 및 사용량 분석**

#### 2.5.1 비용 계산
- `estimated_cost`: 예상 비용 (USD)
- `input_cost`: 입력 토큰 비용
- `output_cost`: 출력 토큰 비용
- `cache_discount`: 캐시 할인 금액
- `priority_surcharge`: 우선순위 추가 요금

#### 2.5.2 할당량 관리
- `quota_consumed`: 소비된 할당량
- `quota_remaining`: 남은 할당량
- `rate_limit_tier`: 속도 제한 티어

---

## 3. 데이터 구조 설계

### 3.1 RequestMetrics 클래스
```python
@dataclass
class RequestMetrics:
    """단일 요청의 모든 메트릭을 포함하는 데이터 클래스"""

    # 식별 정보
    request_id: str
    user_id: Optional[str]
    organization_id: Optional[str]
    api_key_hash: Optional[str]

    # 모델 정보
    model_name: str
    model_version: str
    engine_id: str

    # 토큰 사용량
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    uncached_tokens: int

    # 시간 메트릭 (초)
    arrival_time: float  # wall-clock timestamp
    queued_time: float
    prefill_time: float
    decode_time: float
    inference_time: float
    e2e_latency: float
    time_to_first_token: float
    time_per_output_token: float

    # 자원 사용량
    kv_cache_blocks_used: int
    kv_cache_memory_bytes: int
    peak_gpu_memory_bytes: Optional[int]
    gpu_compute_time: float
    num_preemptions: int

    # 요청 파라미터
    max_tokens: int
    temperature: float
    top_p: float
    n: int  # number of choices
    stream: bool

    # 종료 정보
    finish_reason: str  # STOP, LENGTH, ABORT
    error_code: Optional[str]
    error_message: Optional[str]

    # 비용 계산
    estimated_cost: float
    input_cost: float
    output_cost: float
    cache_discount: float

    # 효율성 지표
    prefix_cache_hit_rate: float
    batch_size: int
    tokens_per_second: float
```

### 3.2 AggregatedMetrics 클래스
```python
@dataclass
class AggregatedMetrics:
    """집계된 메트릭 (사용자별, 기간별, 모델별)"""

    # 집계 기준
    aggregation_key: str  # user_id, organization_id, model_name 등
    time_window_start: datetime
    time_window_end: datetime

    # 요청 통계
    total_requests: int
    successful_requests: int
    failed_requests: int
    aborted_requests: int

    # 토큰 통계
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_cached_tokens: int

    # 비용 통계
    total_cost: float
    total_input_cost: float
    total_output_cost: float
    total_cache_discount: float

    # 성능 통계
    avg_e2e_latency: float
    p50_e2e_latency: float
    p95_e2e_latency: float
    p99_e2e_latency: float
    avg_time_to_first_token: float
    avg_tokens_per_second: float

    # 자원 사용 통계
    total_gpu_compute_seconds: float
    avg_batch_size: float
    total_preemptions: int
```

---

## 4. 가격 책정 모델 (Pricing Models)

### 4.1 토큰 기반 가격 책정 (Token-based Pricing)
**가장 표준적인 방식 - OpenAI, Anthropic 등이 사용**

```python
class TokenBasedPricing:
    """토큰 단위 가격 책정"""

    def calculate_cost(self, metrics: RequestMetrics) -> float:
        input_cost = (metrics.uncached_tokens / 1000) * self.input_price_per_1k
        output_cost = (metrics.completion_tokens / 1000) * self.output_price_per_1k
        cache_discount = (metrics.cached_tokens / 1000) * self.cache_discount_per_1k
        return input_cost + output_cost - cache_discount
```

**가격 예시**:
- GPT-4: $30/1M input tokens, $60/1M output tokens
- GPT-3.5: $0.50/1M input, $1.50/1M output
- 캐시 할인: 입력 가격의 50% 할인

### 4.2 시간 기반 가격 책정 (Time-based Pricing)
**컴퓨팅 자원 중심 - 긴 생성 작업에 유리**

```python
class TimeBasedPricing:
    """GPU 사용 시간 기반 가격 책정"""

    def calculate_cost(self, metrics: RequestMetrics) -> float:
        gpu_seconds = metrics.inference_time
        return gpu_seconds * self.price_per_gpu_second
```

### 4.3 하이브리드 가격 책정 (Hybrid Pricing)
**토큰 + 시간 조합 - 공정한 과금**

```python
class HybridPricing:
    """토큰 + 시간 조합 가격 책정"""

    def calculate_cost(self, metrics: RequestMetrics) -> float:
        token_cost = self.token_pricing.calculate_cost(metrics)
        time_cost = metrics.inference_time * self.price_per_second

        # 둘 중 높은 값 선택 (또는 가중 평균)
        return max(token_cost, time_cost)
        # 또는: return 0.7 * token_cost + 0.3 * time_cost
```

### 4.4 티어 기반 가격 책정 (Tiered Pricing)
**사용량에 따른 할인**

```python
class TieredPricing:
    """사용량 구간별 차등 가격"""

    tiers = [
        (0, 1_000_000, 0.50),      # 0-1M tokens: $0.50/1k
        (1_000_000, 10_000_000, 0.40),  # 1M-10M: $0.40/1k
        (10_000_000, float('inf'), 0.30)  # 10M+: $0.30/1k
    ]
```

### 4.5 우선순위 기반 가격 책정 (Priority-based Pricing)
**응답 속도 보장에 따른 차등 과금**

```python
class PriorityPricing:
    """응답 속도 보장 레벨별 가격"""

    REALTIME = 2.0   # 2배 요금 (TTFT < 1초 보장)
    STANDARD = 1.0   # 표준 요금
    BATCH = 0.5      # 50% 할인 (TTFT 무보장)
```

---

## 5. 구현 아키텍처

### 5.1 시스템 구성도
```
┌─────────────────────────────────────────────────────────┐
│                   vLLM Engine Core                      │
│  (기존 메트릭 수집: FinishedRequestStats)               │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          LLM Metrics Collector (신규)                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │  MetricsCollector                                │   │
│  │  - collect_request_metrics()                     │   │
│  │  - enrich_with_cost_info()                       │   │
│  │  - enrich_with_resource_info()                   │   │
│  └──────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────┘
                        │
           ┌────────────┼────────────┐
           ▼            ▼            ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐
    │  Storage │ │ Exporter │ │ Analyzer │
    │ Backend  │ │ (JSON/   │ │ (Aggre-  │
    │ (File/   │ │ CSV/     │ │ gation/  │
    │ DB/S3)   │ │ Parquet) │ │ Report)  │
    └──────────┘ └──────────┘ └──────────┘
```

### 5.2 핵심 컴포넌트

#### 5.2.1 MetricsCollector
- vLLM의 `FinishedRequestStats`를 `RequestMetrics`로 변환
- 추가 정보 수집 (자원 사용량, 비용 계산)
- 메트릭 유효성 검증

#### 5.2.2 StorageBackend (추상 인터페이스)
- **FileStorageBackend**: JSON/JSONL 파일 저장
- **DatabaseStorageBackend**: PostgreSQL/MySQL 저장
- **CloudStorageBackend**: S3/GCS 저장
- **BufferedStorageBackend**: 배치 쓰기 최적화

#### 5.2.3 MetricsExporter
- JSON 내보내기
- CSV 내보내기
- Parquet 내보내기 (분석용)
- OpenAI 호환 포맷

#### 5.2.4 MetricsAnalyzer
- 사용자별 집계
- 기간별 집계 (시간, 일, 월)
- 모델별 집계
- 비용 리포트 생성
- 통계 분석 (평균, 분위수, 표준편차)

#### 5.2.5 PricingCalculator
- 다양한 가격 책정 모델 지원
- 사용자 정의 가격표 로드
- 실시간 비용 계산

---

## 6. 통합 지점 (Integration Points)

### 6.1 vLLM V1 API Server 통합
```python
# vllm/entrypoints/openai/api_server.py
from vllm.llm_metrics_collector import MetricsCollector

collector = MetricsCollector(config)

@app.post("/v1/completions")
async def create_completion(request):
    # ... 기존 처리 ...

    # 메트릭 수집
    metrics = collector.collect_from_response(
        request=request,
        response=response,
        finished_stats=engine_output.finished_requests
    )

    # 저장
    await collector.save_async(metrics)

    return response
```

### 6.2 Stat Logger Plugin으로 통합
```python
# vLLM의 플러그인 시스템 활용
from vllm.v1.metrics.loggers import StatLoggerBase

class LLMMetricsStatLogger(StatLoggerBase):
    def log(self, stats: IterationStats):
        for finished_req in stats.finished_requests:
            metrics = self.collector.collect(finished_req)
            self.storage.save(metrics)
```

---

## 7. 사용 예시 (Usage Examples)

### 7.1 기본 사용법
```python
from vllm.llm_metrics_collector import (
    MetricsCollector,
    FileStorageBackend,
    TokenBasedPricing
)

# 초기화
storage = FileStorageBackend("/var/log/vllm/metrics")
pricing = TokenBasedPricing(
    input_price_per_1k=0.50,
    output_price_per_1k=1.50,
    cache_discount_per_1k=0.25
)

collector = MetricsCollector(
    storage=storage,
    pricing=pricing
)

# 요청 완료 시 메트릭 수집
metrics = collector.collect_request_metrics(
    finished_stats=finished_request_stats,
    user_id="user_123",
    organization_id="org_456"
)

# 자동 저장됨
```

### 7.2 집계 및 분석
```python
from vllm.llm_metrics_collector import MetricsAnalyzer

analyzer = MetricsAnalyzer(storage)

# 사용자별 월간 리포트
report = analyzer.generate_monthly_report(
    user_id="user_123",
    year=2025,
    month=11
)

print(f"Total requests: {report.total_requests}")
print(f"Total cost: ${report.total_cost:.2f}")
print(f"Total tokens: {report.total_tokens:,}")
print(f"Avg latency: {report.avg_e2e_latency:.3f}s")
```

### 7.3 데이터 내보내기
```python
from vllm.llm_metrics_collector import MetricsExporter

exporter = MetricsExporter(storage)

# CSV로 내보내기
exporter.export_to_csv(
    output_path="/tmp/metrics.csv",
    start_date="2025-11-01",
    end_date="2025-11-30",
    user_id="user_123"
)

# Parquet로 내보내기 (데이터 분석용)
exporter.export_to_parquet(
    output_path="/tmp/metrics.parquet",
    partition_by=["date", "user_id"]
)
```

---

## 8. 성능 고려사항

### 8.1 오버헤드 최소화
- **비동기 저장**: 메트릭 수집이 응답 레이턴시에 영향 없도록
- **배치 쓰기**: 개별 요청마다 I/O 하지 않고 배치로 처리
- **샘플링**: 높은 QPS 환경에서 샘플링 옵션 제공

### 8.2 확장성
- **파티셔닝**: 날짜/사용자별 파티션으로 쿼리 성능 향상
- **압축**: Parquet, JSONL.gz 등 압축 포맷 지원
- **보관 정책**: 오래된 메트릭 아카이빙 또는 집계 후 삭제

---

## 9. 보안 및 프라이버시

### 9.1 개인정보 보호
- **PII 제외**: 실제 프롬프트 내용은 저장하지 않음
- **해시 처리**: user_id, api_key는 해시 저장
- **암호화**: 저장 시 암호화 옵션 (encryption-at-rest)

### 9.2 접근 제어
- **사용자 격리**: 사용자는 자신의 메트릭만 조회 가능
- **관리자 대시보드**: 관리자는 전체 메트릭 조회 가능
- **API 키 기반 인증**: 메트릭 API 접근 시 인증 필요

---

## 10. 표준 준수

### 10.1 OpenAI API 호환성
```json
{
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150,
    "prompt_tokens_details": {
      "cached_tokens": 20
    }
  }
}
```

### 10.2 OpenTelemetry 호환성
- Span attributes로 메트릭 첨부
- Trace context 연결
- 표준 semantic conventions 준수

---

## 11. 향후 확장 가능성

### 11.1 고급 기능
- **예측 분석**: 비용 예측, 사용량 트렌드 분석
- **알림**: 할당량 초과, 비정상 사용 패턴 감지
- **자동 최적화**: 비용 절감을 위한 자동 파라미터 조정 제안
- **A/B 테스팅**: 모델/파라미터 변경 시 비용/성능 비교

### 11.2 통합 가능성
- **Grafana 대시보드**: 실시간 메트릭 시각화
- **Stripe/Payment Gateway**: 자동 청구서 발행
- **DataDog/New Relic**: 모니터링 플랫폼 통합
- **BigQuery/Snowflake**: 데이터 웨어하우스 통합

---

## 12. 성공 지표 (Success Metrics)

### 12.1 기술적 성공 지표
- ✅ 모든 요청의 메트릭 수집 (100% coverage)
- ✅ 메트릭 수집 오버헤드 < 1ms (p99)
- ✅ 메트릭 정확도 99.9% 이상
- ✅ 데이터 손실률 < 0.01%

### 12.2 비즈니스 성공 지표
- ✅ 정확한 월간 청구서 발행 가능
- ✅ 사용자별 비용 분석 가능
- ✅ 비용 최적화 인사이트 제공
- ✅ 투명한 가격 책정 달성

---

## 부록: 참고 자료

### A. vLLM 기존 메트릭 시스템
- 위치: `vllm/v1/metrics/`
- 핵심 클래스: `FinishedRequestStats`, `PrometheusStatLogger`
- 문서: `docs/design/metrics.md`, `docs/usage/metrics.md`

### B. 유사 시스템
- OpenAI Usage API
- Anthropic Billing API
- AWS Bedrock Pricing
- Google Vertex AI Pricing

### C. 관련 표준
- OpenTelemetry Gen AI Semantic Conventions
- OpenAPI Specification for Usage Data
- Cloud FinOps Best Practices
