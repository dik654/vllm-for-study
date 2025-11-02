# LLM Metrics Collector - Implementation TODO

## 프로젝트 개요
vLLM에서 LLM 사용량 및 가격 책정을 위한 메트릭 수집 시스템 구축

---

## Phase 1: 프로젝트 구조 및 기본 설정

### 1.1 디렉터리 구조 생성
- [x] `/vllm/llm_metrics_collector/` 메인 디렉터리 생성
- [ ] `/vllm/llm_metrics_collector/__init__.py` 생성
- [ ] `/vllm/llm_metrics_collector/models/` 데이터 모델 디렉터리
- [ ] `/vllm/llm_metrics_collector/collectors/` 수집기 디렉터리
- [ ] `/vllm/llm_metrics_collector/storage/` 저장 백엔드 디렉터리
- [ ] `/vllm/llm_metrics_collector/pricing/` 가격 계산 디렉터리
- [ ] `/vllm/llm_metrics_collector/exporters/` 내보내기 디렉터리
- [ ] `/vllm/llm_metrics_collector/analyzers/` 분석 디렉터리
- [ ] `/vllm/llm_metrics_collector/utils/` 유틸리티 디렉터리
- [ ] `/tests/llm_metrics_collector/` 테스트 디렉터리

**예상 구조**:
```
vllm/llm_metrics_collector/
├── __init__.py
├── SPECIFICATION.md          # 완료
├── TODO.md                   # 현재 문서
├── README.md                 # 사용자 가이드
├── models/
│   ├── __init__.py
│   ├── request_metrics.py    # RequestMetrics 데이터 클래스
│   ├── aggregated_metrics.py # AggregatedMetrics 데이터 클래스
│   └── enums.py              # FinishReason, PricingTier 등
├── collectors/
│   ├── __init__.py
│   ├── base.py               # MetricsCollectorBase 추상 클래스
│   ├── vllm_collector.py     # vLLM FinishedRequestStats → RequestMetrics
│   └── resource_collector.py # 자원 사용량 추가 수집
├── storage/
│   ├── __init__.py
│   ├── base.py               # StorageBackend 추상 인터페이스
│   ├── file_storage.py       # FileStorageBackend (JSON/JSONL)
│   ├── database_storage.py   # DatabaseStorageBackend (PostgreSQL)
│   └── memory_storage.py     # MemoryStorageBackend (테스트/개발용)
├── pricing/
│   ├── __init__.py
│   ├── base.py               # PricingCalculator 추상 클래스
│   ├── token_based.py        # TokenBasedPricing
│   ├── time_based.py         # TimeBasedPricing
│   ├── hybrid.py             # HybridPricing
│   ├── tiered.py             # TieredPricing
│   └── config.py             # PricingConfig (가격표 로드)
├── exporters/
│   ├── __init__.py
│   ├── base.py               # ExporterBase 추상 클래스
│   ├── json_exporter.py      # JSON/JSONL 내보내기
│   ├── csv_exporter.py       # CSV 내보내기
│   └── parquet_exporter.py   # Parquet 내보내기
├── analyzers/
│   ├── __init__.py
│   ├── aggregator.py         # 메트릭 집계 로직
│   ├── reporter.py           # 리포트 생성기
│   └── statistics.py         # 통계 계산 (평균, 분위수 등)
└── utils/
    ├── __init__.py
    ├── time_utils.py         # 시간 관련 유틸리티
    ├── hash_utils.py         # 해싱 유틸리티
    └── validation.py         # 데이터 유효성 검증
```

---

## Phase 2: 데이터 모델 구현

### 2.1 Enums 정의
**파일**: `models/enums.py`

- [ ] `FinishReason` enum 정의
  - [ ] `STOP`: 정상 종료
  - [ ] `LENGTH`: max_tokens 도달
  - [ ] `ABORT`: 클라이언트 취소 또는 에러
  - [ ] `PREEMPTED`: 선점됨 (재시도 필요)

- [ ] `PricingTier` enum 정의
  - [ ] `REALTIME`: 실시간 우선순위
  - [ ] `STANDARD`: 표준
  - [ ] `BATCH`: 배치 (저가)

- [ ] `StorageFormat` enum 정의
  - [ ] `JSON`, `JSONL`, `CSV`, `PARQUET`

**검증**:
- [ ] 각 enum이 문자열 값을 가지는지 확인
- [ ] vLLM의 기존 `FinishReason`과 매핑 가능한지 확인

---

### 2.2 RequestMetrics 데이터 클래스
**파일**: `models/request_metrics.py`

- [ ] `RequestMetrics` dataclass 정의
  - [ ] 식별 정보 필드 (request_id, user_id, organization_id, api_key_hash)
  - [ ] 모델 정보 필드 (model_name, model_version, engine_id)
  - [ ] 토큰 사용량 필드 (prompt_tokens, completion_tokens, cached_tokens 등)
  - [ ] 시간 메트릭 필드 (arrival_time, queued_time, prefill_time 등)
  - [ ] 자원 사용량 필드 (kv_cache_blocks_used, gpu_compute_time 등)
  - [ ] 요청 파라미터 필드 (max_tokens, temperature, top_p 등)
  - [ ] 종료 정보 필드 (finish_reason, error_code, error_message)
  - [ ] 비용 정보 필드 (estimated_cost, input_cost, output_cost 등)

- [ ] 유효성 검증 메서드
  - [ ] `validate()`: 필수 필드 체크, 값 범위 검증
  - [ ] `is_successful()`: 요청 성공 여부 반환

- [ ] 직렬화/역직렬화 메서드
  - [ ] `to_dict()`: dict로 변환
  - [ ] `from_dict(data: dict)`: dict에서 생성
  - [ ] `to_json()`: JSON 문자열로 변환
  - [ ] `from_json(json_str: str)`: JSON에서 생성

- [ ] OpenAI 호환 포맷 변환
  - [ ] `to_openai_usage()`: OpenAI UsageInfo 포맷으로 변환

**검증**:
- [ ] 모든 필드의 타입 힌트가 정확한지 확인
- [ ] Optional 필드는 default=None 설정
- [ ] datetime 필드는 timezone-aware인지 확인

---

### 2.3 AggregatedMetrics 데이터 클래스
**파일**: `models/aggregated_metrics.py`

- [ ] `AggregatedMetrics` dataclass 정의
  - [ ] 집계 기준 필드 (aggregation_key, time_window_start, time_window_end)
  - [ ] 요청 통계 필드 (total_requests, successful_requests, failed_requests)
  - [ ] 토큰 통계 필드 (total_prompt_tokens, total_completion_tokens)
  - [ ] 비용 통계 필드 (total_cost, total_input_cost, total_output_cost)
  - [ ] 성능 통계 필드 (avg_e2e_latency, p50/p95/p99_e2e_latency)
  - [ ] 자원 사용 통계 필드 (total_gpu_compute_seconds, avg_batch_size)

- [ ] 직렬화 메서드
  - [ ] `to_dict()`, `from_dict()`
  - [ ] `to_json()`, `from_json()`

- [ ] 리포트 생성 메서드
  - [ ] `to_report_string()`: 읽기 쉬운 텍스트 리포트 생성

**검증**:
- [ ] 집계 계산 로직이 올바른지 단위 테스트

---

## Phase 3: 메트릭 수집기 구현

### 3.1 Base Collector
**파일**: `collectors/base.py`

- [ ] `MetricsCollectorBase` 추상 클래스 정의
  - [ ] `collect()` 추상 메서드: 메트릭 수집
  - [ ] `enrich()` 추상 메서드: 추가 정보 보강
  - [ ] `validate()` 메서드: 수집된 메트릭 검증

**설계**:
```python
from abc import ABC, abstractmethod
from typing import Optional
from ..models.request_metrics import RequestMetrics

class MetricsCollectorBase(ABC):
    @abstractmethod
    def collect(self, **kwargs) -> RequestMetrics:
        """메트릭 수집"""
        pass

    @abstractmethod
    def enrich(self, metrics: RequestMetrics, **kwargs) -> RequestMetrics:
        """추가 정보로 보강"""
        pass

    def validate(self, metrics: RequestMetrics) -> bool:
        """메트릭 유효성 검증"""
        return metrics.validate()
```

---

### 3.2 vLLM Collector
**파일**: `collectors/vllm_collector.py`

- [ ] `VLLMMetricsCollector` 클래스 구현
  - [ ] vLLM의 `FinishedRequestStats`를 입력으로 받음
  - [ ] `RequestMetrics`로 변환

- [ ] 필드 매핑 구현
  - [ ] `num_prompt_tokens` → `prompt_tokens`
  - [ ] `num_generation_tokens` → `completion_tokens`
  - [ ] `e2e_latency` → `e2e_latency`
  - [ ] `queued_time` → `queued_time`
  - [ ] `prefill_time` → `prefill_time`
  - [ ] `decode_time` → `decode_time`
  - [ ] `inference_time` → `inference_time`
  - [ ] `first_token_latency` → `time_to_first_token`
  - [ ] `mean_time_per_output_token` → `time_per_output_token`
  - [ ] `finish_reason` → `finish_reason` (enum 변환)

- [ ] 추가 필드 계산
  - [ ] `total_tokens` = prompt_tokens + completion_tokens
  - [ ] `uncached_tokens` = prompt_tokens - cached_tokens
  - [ ] `tokens_per_second` = completion_tokens / decode_time

- [ ] 누락 필드 처리
  - [ ] cached_tokens는 별도 수집 필요 (v1 metrics에서 가져오기)
  - [ ] user_id, organization_id는 요청 컨텍스트에서 가져오기
  - [ ] 자원 사용량은 ResourceCollector로 위임

**검증**:
- [ ] 샘플 `FinishedRequestStats` 데이터로 변환 테스트
- [ ] 모든 필수 필드가 채워지는지 확인
- [ ] None 필드는 Optional로 처리되는지 확인

---

### 3.3 Resource Collector
**파일**: `collectors/resource_collector.py`

- [ ] `ResourceMetricsCollector` 클래스 구현
  - [ ] GPU 메모리 사용량 수집
  - [ ] KV 캐시 사용량 수집
  - [ ] 배치 사이즈 수집

- [ ] vLLM V1 metrics 연동
  - [ ] `SchedulerStats`에서 `kv_cache_usage` 가져오기
  - [ ] `IterationStats`에서 `num_generation_tokens` 가져오기

- [ ] 메모리 프로파일링 연동
  - [ ] `vllm.utils.mem_utils.MemorySnapshot` 활용
  - [ ] 요청별 피크 메모리 추정

**주의사항**:
- 자원 메트릭은 요청 단위가 아닌 전역/배치 단위일 수 있음
- 요청별로 할당하려면 추정 로직 필요 (예: 토큰 비율로 안분)

**검증**:
- [ ] 실제 vLLM 요청 처리 시 메모리 사용량 측정
- [ ] KV 캐시 블록 계산이 정확한지 확인

---

## Phase 4: 저장 백엔드 구현

### 4.1 Base Storage
**파일**: `storage/base.py`

- [ ] `StorageBackend` 추상 클래스 정의
  - [ ] `save(metrics: RequestMetrics)`: 단일 메트릭 저장
  - [ ] `save_batch(metrics: List[RequestMetrics])`: 배치 저장
  - [ ] `load(filters: dict)`: 필터 조건으로 메트릭 조회
  - [ ] `count(filters: dict)`: 메트릭 개수 조회
  - [ ] `close()`: 리소스 정리

- [ ] 비동기 메서드 추가
  - [ ] `async save_async()`
  - [ ] `async save_batch_async()`
  - [ ] `async load_async()`

**설계**:
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.request_metrics import RequestMetrics

class StorageBackend(ABC):
    @abstractmethod
    def save(self, metrics: RequestMetrics) -> None:
        pass

    @abstractmethod
    def save_batch(self, metrics: List[RequestMetrics]) -> None:
        pass

    @abstractmethod
    def load(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: int = 0
    ) -> List[RequestMetrics]:
        pass

    @abstractmethod
    def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
```

---

### 4.2 Memory Storage (테스트용)
**파일**: `storage/memory_storage.py`

- [ ] `MemoryStorageBackend` 구현
  - [ ] 메모리 내 리스트로 메트릭 저장
  - [ ] 필터링 로직 구현 (user_id, date range 등)
  - [ ] 정렬 및 페이지네이션 지원

- [ ] 테스트 유틸리티 메서드
  - [ ] `clear()`: 모든 데이터 삭제
  - [ ] `get_all()`: 모든 메트릭 반환

**검증**:
- [ ] 수천 개 메트릭 저장 후 조회 성능 측정
- [ ] 필터링이 정확하게 동작하는지 확인

---

### 4.3 File Storage
**파일**: `storage/file_storage.py`

- [ ] `FileStorageBackend` 구현
  - [ ] JSONL (JSON Lines) 포맷으로 저장
  - [ ] 날짜별 파티셔닝 (예: `2025-11-02.jsonl`)
  - [ ] Append-only 쓰기 (기존 파일에 추가)

- [ ] 파일 로테이션 구현
  - [ ] 일별, 주별, 월별 로테이션 옵션
  - [ ] 압축 옵션 (gzip)

- [ ] 조회 최적화
  - [ ] 날짜 범위 필터는 해당 파일만 읽기
  - [ ] 인덱스 파일 생성 (선택사항)

**설계**:
```python
class FileStorageBackend(StorageBackend):
    def __init__(
        self,
        base_dir: str,
        partition_by: str = "day",  # hour, day, week, month
        compress: bool = False
    ):
        self.base_dir = Path(base_dir)
        self.partition_by = partition_by
        self.compress = compress
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, timestamp: float) -> Path:
        dt = datetime.fromtimestamp(timestamp)
        if self.partition_by == "day":
            filename = dt.strftime("%Y-%m-%d.jsonl")
        # ...
        return self.base_dir / filename
```

**검증**:
- [ ] 파티셔닝이 정확하게 동작하는지 확인
- [ ] 압축 파일 읽기/쓰기 테스트
- [ ] 대용량 파일 (수십만 줄) 처리 성능 측정

---

### 4.4 Database Storage
**파일**: `storage/database_storage.py`

- [ ] `DatabaseStorageBackend` 구현
  - [ ] PostgreSQL 연동 (psycopg2 또는 asyncpg)
  - [ ] 테이블 스키마 자동 생성

- [ ] 테이블 스키마 정의
```sql
CREATE TABLE llm_request_metrics (
    id BIGSERIAL PRIMARY KEY,
    request_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    organization_id VARCHAR(255),
    model_name VARCHAR(255) NOT NULL,
    arrival_time TIMESTAMP WITH TIME ZONE NOT NULL,

    -- 토큰
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    cached_tokens INTEGER DEFAULT 0,

    -- 시간 (초)
    queued_time REAL,
    prefill_time REAL,
    decode_time REAL,
    inference_time REAL,
    e2e_latency REAL,
    time_to_first_token REAL,

    -- 자원
    kv_cache_blocks_used INTEGER,
    gpu_compute_time REAL,

    -- 비용
    estimated_cost REAL,
    input_cost REAL,
    output_cost REAL,

    -- 종료 정보
    finish_reason VARCHAR(50),
    error_code VARCHAR(50),

    -- 인덱스
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    INDEX idx_user_id (user_id),
    INDEX idx_arrival_time (arrival_time),
    INDEX idx_model_name (model_name)
);
```

- [ ] 배치 삽입 최적화
  - [ ] executemany() 사용
  - [ ] COPY 명령어 활용 (PostgreSQL)

- [ ] 쿼리 최적화
  - [ ] 적절한 인덱스 생성
  - [ ] 파티셔닝 (시간 범위별)

**검증**:
- [ ] 트랜잭션 처리 테스트
- [ ] 동시성 테스트 (여러 프로세스가 동시 쓰기)
- [ ] 쿼리 성능 측정 (EXPLAIN ANALYZE)

---

## Phase 5: 가격 계산 구현

### 5.1 Base Pricing
**파일**: `pricing/base.py`

- [ ] `PricingCalculator` 추상 클래스 정의
  - [ ] `calculate_cost(metrics: RequestMetrics) -> float`
  - [ ] `calculate_input_cost(metrics: RequestMetrics) -> float`
  - [ ] `calculate_output_cost(metrics: RequestMetrics) -> float`

---

### 5.2 Token-based Pricing
**파일**: `pricing/token_based.py`

- [ ] `TokenBasedPricing` 구현
  - [ ] 입력 토큰당 가격 (per 1K tokens)
  - [ ] 출력 토큰당 가격 (per 1K tokens)
  - [ ] 캐시 할인율

- [ ] 비용 계산 로직
```python
def calculate_cost(self, metrics: RequestMetrics) -> float:
    input_cost = (metrics.uncached_tokens / 1000) * self.input_price_per_1k
    output_cost = (metrics.completion_tokens / 1000) * self.output_price_per_1k
    cache_discount = (metrics.cached_tokens / 1000) * self.cache_discount_per_1k
    return input_cost + output_cost - cache_discount
```

**검증**:
- [ ] OpenAI 가격과 동일하게 계산되는지 확인
- [ ] 캐시 할인이 정확한지 검증

---

### 5.3 Hybrid Pricing
**파일**: `pricing/hybrid.py`

- [ ] `HybridPricing` 구현
  - [ ] 토큰 기반 + 시간 기반 조합
  - [ ] 가중치 조정 가능 (alpha × token_cost + beta × time_cost)

---

### 5.4 Pricing Config
**파일**: `pricing/config.py`

- [ ] `PricingConfig` 클래스 구현
  - [ ] YAML/JSON 파일에서 가격표 로드
  - [ ] 모델별 가격 설정

- [ ] 설정 파일 예시
```yaml
models:
  gpt-4:
    input_price_per_1k: 30.0
    output_price_per_1k: 60.0
    cache_discount_per_1k: 15.0
  gpt-3.5-turbo:
    input_price_per_1k: 0.5
    output_price_per_1k: 1.5
    cache_discount_per_1k: 0.25
```

**검증**:
- [ ] YAML 파싱 테스트
- [ ] 누락된 모델 처리 (기본값 사용)

---

## Phase 6: 데이터 내보내기 구현

### 6.1 Base Exporter
**파일**: `exporters/base.py`

- [ ] `ExporterBase` 추상 클래스 정의
  - [ ] `export(metrics: List[RequestMetrics], output_path: str)`

---

### 6.2 JSON Exporter
**파일**: `exporters/json_exporter.py`

- [ ] `JSONExporter` 구현
  - [ ] JSON 배열 포맷
  - [ ] JSONL (JSON Lines) 포맷

---

### 6.3 CSV Exporter
**파일**: `exporters/csv_exporter.py`

- [ ] `CSVExporter` 구현
  - [ ] 헤더 포함
  - [ ] 플랫 구조 (중첩 없음)

---

### 6.4 Parquet Exporter
**파일**: `exporters/parquet_exporter.py`

- [ ] `ParquetExporter` 구현
  - [ ] PyArrow 또는 FastParquet 사용
  - [ ] 파티셔닝 지원 (date, user_id)

**검증**:
- [ ] Pandas로 읽어서 데이터 무결성 확인
- [ ] 압축률 및 읽기 성능 측정

---

## Phase 7: 분석 및 집계 구현

### 7.1 Aggregator
**파일**: `analyzers/aggregator.py`

- [ ] `MetricsAggregator` 클래스 구현
  - [ ] 사용자별 집계
  - [ ] 기간별 집계 (시간, 일, 주, 월)
  - [ ] 모델별 집계

- [ ] 집계 함수
  - [ ] `aggregate_by_user(user_id, start_date, end_date)`
  - [ ] `aggregate_by_period(period, start_date, end_date)`
  - [ ] `aggregate_by_model(model_name, start_date, end_date)`

**검증**:
- [ ] 샘플 데이터로 집계 정확성 확인
- [ ] 대용량 데이터 집계 성능 측정

---

### 7.2 Statistics Calculator
**파일**: `analyzers/statistics.py`

- [ ] 통계 함수 구현
  - [ ] `calculate_percentile(values, percentile)`
  - [ ] `calculate_mean(values)`
  - [ ] `calculate_median(values)`
  - [ ] `calculate_stddev(values)`

- [ ] 분포 분석
  - [ ] 히스토그램 생성
  - [ ] 이상치 탐지 (outlier detection)

---

### 7.3 Reporter
**파일**: `analyzers/reporter.py`

- [ ] `MetricsReporter` 클래스 구현
  - [ ] 월간 리포트 생성
  - [ ] 사용자별 리포트 생성
  - [ ] 모델별 리포트 생성

- [ ] 리포트 포맷
  - [ ] 텍스트 리포트
  - [ ] HTML 리포트
  - [ ] PDF 리포트 (선택사항)

**예시 리포트**:
```
=== Monthly Usage Report ===
Period: 2025-11-01 to 2025-11-30
User: user_123

Requests:
  Total: 1,234
  Successful: 1,200 (97.2%)
  Failed: 34 (2.8%)

Tokens:
  Prompt Tokens: 500,000
  Completion Tokens: 250,000
  Total Tokens: 750,000
  Cached Tokens: 100,000 (20% cache hit rate)

Performance:
  Avg E2E Latency: 2.5s
  P50 Latency: 1.8s
  P95 Latency: 5.2s
  P99 Latency: 8.1s
  Avg TTFT: 0.8s

Cost:
  Input Cost: $250.00
  Output Cost: $375.00
  Cache Discount: -$50.00
  Total Cost: $575.00
```

---

## Phase 8: vLLM 통합

### 8.1 Stat Logger Plugin
**파일**: `collectors/stat_logger_plugin.py`

- [ ] `LLMMetricsStatLogger` 구현
  - [ ] `vllm.v1.metrics.loggers.StatLoggerBase` 상속
  - [ ] `log(stats: IterationStats)` 메서드 구현

- [ ] vLLM 플러그인 등록
  - [ ] `setup.py` 또는 `pyproject.toml`에 entry point 추가
```python
[project.entry-points."vllm.stat_logger_plugins"]
llm_metrics_collector = "vllm.llm_metrics_collector.collectors.stat_logger_plugin:LLMMetricsStatLoggerFactory"
```

**검증**:
- [ ] vLLM 서버 실행 시 플러그인이 로드되는지 확인
- [ ] 요청 처리 시 메트릭이 수집되는지 확인

---

### 8.2 OpenAI API Server 통합
**파일**: `entrypoints/openai_integration.py`

- [ ] OpenAI API 응답에 usage 정보 추가
  - [ ] `UsageInfo` 객체 생성
  - [ ] `prompt_tokens_details`에 cached_tokens 포함

- [ ] 메트릭 수집 미들웨어 (선택사항)
  - [ ] FastAPI middleware로 모든 요청/응답 추적

**검증**:
- [ ] cURL로 API 호출 후 usage 정보 확인
- [ ] 스트리밍 요청에서도 메트릭 수집되는지 확인

---

## Phase 9: 유틸리티 구현

### 9.1 Time Utils
**파일**: `utils/time_utils.py`

- [ ] 시간 관련 유틸리티
  - [ ] `get_current_timestamp()`: 현재 타임스탬프
  - [ ] `timestamp_to_datetime(ts: float)`: 변환
  - [ ] `get_date_range(start, end)`: 날짜 범위 생성

---

### 9.2 Hash Utils
**파일**: `utils/hash_utils.py`

- [ ] 해싱 유틸리티
  - [ ] `hash_api_key(api_key: str)`: API 키 해시 (SHA256)
  - [ ] `hash_user_id(user_id: str)`: 사용자 ID 해시 (선택적)

**검증**:
- [ ] 해시 충돌 가능성 검토
- [ ] 동일 입력에 동일 출력 확인

---

### 9.3 Validation
**파일**: `utils/validation.py`

- [ ] 유효성 검증 함수
  - [ ] `validate_request_id(request_id: str)`
  - [ ] `validate_token_count(tokens: int)`
  - [ ] `validate_timestamp(ts: float)`

---

## Phase 10: 테스트 작성

### 10.1 Unit Tests

**파일**: `tests/llm_metrics_collector/test_models.py`
- [ ] RequestMetrics 직렬화/역직렬화 테스트
- [ ] AggregatedMetrics 계산 테스트
- [ ] Enum 변환 테스트

**파일**: `tests/llm_metrics_collector/test_collectors.py`
- [ ] VLLMMetricsCollector 변환 테스트
- [ ] ResourceMetricsCollector 수집 테스트

**파일**: `tests/llm_metrics_collector/test_storage.py`
- [ ] MemoryStorageBackend CRUD 테스트
- [ ] FileStorageBackend 파티셔닝 테스트
- [ ] DatabaseStorageBackend 쿼리 테스트

**파일**: `tests/llm_metrics_collector/test_pricing.py`
- [ ] TokenBasedPricing 계산 테스트
- [ ] HybridPricing 가중치 테스트
- [ ] PricingConfig 로드 테스트

**파일**: `tests/llm_metrics_collector/test_exporters.py`
- [ ] JSONExporter 내보내기 테스트
- [ ] CSVExporter 포맷 테스트
- [ ] ParquetExporter 파티셔닝 테스트

**파일**: `tests/llm_metrics_collector/test_analyzers.py`
- [ ] MetricsAggregator 집계 테스트
- [ ] MetricsReporter 리포트 생성 테스트
- [ ] Statistics 계산 테스트

---

### 10.2 Integration Tests

**파일**: `tests/llm_metrics_collector/test_integration.py`
- [ ] 전체 플로우 테스트 (수집 → 저장 → 조회 → 집계 → 내보내기)
- [ ] vLLM 서버와 통합 테스트 (실제 요청 처리)
- [ ] 멀티프로세스 환경 테스트

---

### 10.3 Performance Tests

**파일**: `tests/llm_metrics_collector/test_performance.py`
- [ ] 대용량 메트릭 저장 성능 (10만+ 메트릭)
- [ ] 복잡한 쿼리 성능 (날짜 범위 + 사용자 필터)
- [ ] 집계 성능 (수만 건 데이터 집계)
- [ ] 내보내기 성능 (CSV, Parquet)

**목표**:
- 메트릭 수집 오버헤드 < 1ms (p99)
- 10만 메트릭 배치 저장 < 1초
- 월간 리포트 생성 < 5초

---

## Phase 11: 문서화

### 11.1 README 작성
**파일**: `vllm/llm_metrics_collector/README.md`

- [ ] 프로젝트 소개
- [ ] 주요 기능
- [ ] 설치 방법
- [ ] 빠른 시작 가이드
- [ ] 설정 옵션
- [ ] 사용 예시
- [ ] FAQ

---

### 11.2 API 문서
**파일**: `vllm/llm_metrics_collector/docs/API.md`

- [ ] 모든 public 클래스 및 메서드 문서화
- [ ] 파라미터 설명
- [ ] 반환값 설명
- [ ] 예외 처리
- [ ] 코드 예시

---

### 11.3 설정 가이드
**파일**: `vllm/llm_metrics_collector/docs/CONFIGURATION.md`

- [ ] Storage Backend 설정
- [ ] Pricing 설정
- [ ] Export 설정
- [ ] 환경 변수 목록

---

### 11.4 통합 가이드
**파일**: `vllm/llm_metrics_collector/docs/INTEGRATION.md`

- [ ] vLLM V1 API Server 통합 방법
- [ ] Stat Logger Plugin 사용법
- [ ] Prometheus 메트릭 활용
- [ ] 커스텀 Storage Backend 구현 가이드

---

## Phase 12: 예제 및 데모

### 12.1 기본 예제
**파일**: `examples/llm_metrics_collector/basic_usage.py`

- [ ] 간단한 메트릭 수집 예제
- [ ] 파일 저장 예제
- [ ] 조회 및 분석 예제

---

### 12.2 고급 예제
**파일**: `examples/llm_metrics_collector/advanced_usage.py`

- [ ] 데이터베이스 통합 예제
- [ ] 커스텀 Pricing 구현 예제
- [ ] 비동기 저장 예제

---

### 12.3 대시보드 예제
**파일**: `examples/llm_metrics_collector/dashboard.py`

- [ ] Streamlit 또는 Gradio 대시보드
- [ ] 실시간 메트릭 시각화
- [ ] 비용 분석 차트

---

## Phase 13: 배포 준비

### 13.1 패키징
- [ ] `__init__.py`에 public API export
- [ ] 버전 관리 (`__version__`)
- [ ] 의존성 명시 (`requirements.txt`)

---

### 13.2 Docker 지원
**파일**: `examples/llm_metrics_collector/Dockerfile`

- [ ] vLLM + Metrics Collector 통합 Docker 이미지
- [ ] PostgreSQL 연동 Docker Compose

---

### 13.3 CI/CD
- [ ] GitHub Actions 워크플로우
  - [ ] 단위 테스트 실행
  - [ ] 코드 커버리지 측정
  - [ ] 린팅 (flake8, black, mypy)

---

## Phase 14: 최적화 및 튜닝

### 14.1 성능 최적화
- [ ] 배치 쓰기 최적화 (버퍼 크기 조정)
- [ ] 인덱스 최적화 (데이터베이스)
- [ ] 캐싱 전략 (자주 조회되는 집계 결과)

---

### 14.2 메모리 최적화
- [ ] 대용량 데이터 스트리밍 처리
- [ ] Generator 패턴 활용 (메모리에 전체 로드 방지)

---

### 14.3 동시성 처리
- [ ] 멀티프로세스 안전성 확보 (파일 락, DB 트랜잭션)
- [ ] 비동기 I/O 최적화 (asyncio, aiofiles)

---

## Phase 15: 보안 및 프라이버시

### 15.1 데이터 보호
- [ ] API 키 해싱 (저장 시 평문 금지)
- [ ] 프롬프트 내용 저장 금지 (메타데이터만)
- [ ] 민감 정보 필터링 (정규식 기반)

---

### 15.2 접근 제어
- [ ] 사용자별 메트릭 격리
- [ ] 관리자 권한 확인
- [ ] API 인증 (토큰 기반)

---

### 15.3 암호화
- [ ] 저장 시 암호화 옵션 (AES-256)
- [ ] 전송 시 암호화 (HTTPS)

---

## Phase 16: 모니터링 및 알림

### 16.1 헬스 체크
- [ ] Storage Backend 연결 상태 확인
- [ ] 메트릭 수집 실패율 모니터링

---

### 16.2 알림 시스템
- [ ] 사용량 할당량 초과 알림
- [ ] 비정상 사용 패턴 감지
- [ ] 비용 임계값 알림

---

## 체크리스트 요약

### 필수 (MVP)
- [ ] Phase 1: 프로젝트 구조 생성
- [ ] Phase 2: 데이터 모델 구현
- [ ] Phase 3: 메트릭 수집기 구현
- [ ] Phase 4.1-4.2: Memory/File Storage 구현
- [ ] Phase 5.1-5.2: Token-based Pricing 구현
- [ ] Phase 7.1: 기본 집계 구현
- [ ] Phase 10.1: 단위 테스트 작성
- [ ] Phase 11.1: README 작성

### 중요
- [ ] Phase 4.4: Database Storage 구현
- [ ] Phase 6: 모든 Exporter 구현
- [ ] Phase 7.2-7.3: 통계 및 리포트 구현
- [ ] Phase 8: vLLM 통합
- [ ] Phase 10.2: 통합 테스트
- [ ] Phase 11.2-11.4: 상세 문서화

### 선택사항
- [ ] Phase 5.3-5.4: 고급 Pricing 모델
- [ ] Phase 10.3: 성능 테스트
- [ ] Phase 12: 예제 및 데모
- [ ] Phase 13: 배포 준비
- [ ] Phase 14: 최적화
- [ ] Phase 15: 보안 강화
- [ ] Phase 16: 모니터링

---

## 개발 일정 (예상)

### Week 1: 기초 구조
- Phase 1-2 완료
- Phase 3 시작

### Week 2: 핵심 기능
- Phase 3-4 완료
- Phase 5 시작

### Week 3: 분석 및 통합
- Phase 5-7 완료
- Phase 8 시작

### Week 4: 테스트 및 문서화
- Phase 8 완료
- Phase 10-11 완료
- Phase 12-13 (선택사항)

---

## 다음 단계

1. **Phase 1 시작**: 디렉터리 구조 및 `__init__.py` 파일 생성
2. **Phase 2 시작**: `RequestMetrics` 데이터 클래스 구현
3. **Phase 3 시작**: `VLLMMetricsCollector` 구현

**시작 명령**:
```bash
# 1. 구조 생성
cd vllm/llm_metrics_collector
touch __init__.py
mkdir -p models collectors storage pricing exporters analyzers utils
touch models/__init__.py collectors/__init__.py ...

# 2. 첫 번째 파일 작성
# models/enums.py 부터 시작
```

---

## 참고 자료

- vLLM V1 Metrics: `/vllm/v1/metrics/`
- OpenAI Usage API: https://platform.openai.com/docs/api-reference/usage
- Prometheus Best Practices: https://prometheus.io/docs/practices/naming/
- PostgreSQL Partitioning: https://www.postgresql.org/docs/current/ddl-partitioning.html
