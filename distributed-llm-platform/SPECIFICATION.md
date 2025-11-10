# 분산 LLM 인프라 플랫폼 Specification v2.0

## 1. 시스템 개요

### 1.1 목적
여러 서버(Provider)가 LLM API, 메모리, 스토리지 리소스를 제공하고, 중앙 Router가 이를 통합하여 사용자에게 서비스를 제공하는 분산 인프라 플랫폼.

### 1.2 주요 기능
- **Provider**: LLM inference, Context/General Purpose Memory/Storage 제공
- **Router**: 요청 라우팅, metric 수집, 정산 처리
- **Billing**: OpenRouter 스타일 commission + 부가 서비스 마진

### 1.3 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        Central Router                        │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Request Router │  │ Metric Collector│  │ Billing Engine │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│  │ Provider Mgmt  │  │ User Mgmt      │  │ Payment Proc   │ │
│  └────────────────┘  └────────────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼────┐          ┌────▼────┐          ┌────▼────┐
   │Provider1│          │Provider2│          │Provider3│
   │ ┌──────┐│          │ ┌──────┐│          │ ┌──────┐│
   │ │ vLLM ││          │ │ vLLM ││          │ │ vLLM ││
   │ └──────┘│          │ └──────┘│          │ └──────┘│
   │ ┌──────┐│          │ ┌──────┐│          │ ┌──────┐│
   │ │Ctx Mem│          │ │Ctx Mem│          │ │Ctx Mem│
   │ │GP Mem││          │ │GP Mem││          │ │GP Mem││
   │ └──────┘│          │ └──────┘│          │ └──────┘│
   │ ┌──────┐│          │ ┌──────┐│          │ ┌──────┐│
   │ │Ctx Stor│         │ │Ctx Stor│         │ │Ctx Stor│
   │ │GP Stor││         │ │GP Stor││         │ │GP Stor││
   │ └──────┘│          │ └──────┘│          │ └──────┘│
   └─────────┘          └─────────┘          └─────────┘
```

### 1.4 서비스 타입 정의

#### LLM API Service
- vLLM 기반 inference
- OpenAI-compatible API
- Commission 기반 과금 (5%)

#### Context Memory Service (컨텍스트 메모리)
- **용도**: LLM 개인화 컨텍스트 저장
- **기능**: KV cache, conversation history, RAG embeddings
- **특징**: 빠른 액세스, 휘발성
- **가격**: 독립적 가격 책정

#### Context Storage Service (컨텍스트 스토리지)
- **용도**: 사용자별 LLM 데이터 영구 저장
- **기능**: Chat history, fine-tuning data, user preferences
- **특징**: 영구 저장, 백업
- **가격**: 독립적 가격 책정

#### General Purpose Memory Service (범용 메모리)
- **용도**: 서버 실행 환경
- **기능**: Application memory, cache, database buffer
- **특징**: On-demand allocation/release
- **가격**: 시간당 과금

#### General Purpose Storage Service (범용 스토리지)
- **용도**: 파일 저장, 데이터베이스
- **기능**: 파일 업로드/다운로드, object storage
- **특징**: 티어별 성능 (Hot/Cold)
- **가격**: 티어별 차등 과금

---

## 2. Provider Node 상세 설계

### 2.1 Provider 역할

1. **LLM API Endpoint** - vLLM 기반 inference 서비스
2. **Context Memory Service** - LLM 컨텍스트용 메모리
3. **Context Storage Service** - LLM 데이터 영구 저장
4. **General Purpose Memory** - 범용 메모리 할당
5. **General Purpose Storage** - 범용 파일 저장

### 2.2 Provider Agent 구성

```python
# provider_agent/config.py
from decimal import Decimal
from typing import List

class ProviderConfig:
    provider_id: str              # 고유 Provider ID
    provider_name: str            # Provider 이름
    router_url: str               # Central Router URL

    # LLM Service
    vllm_config: VllmConfig       # vLLM 설정
    supported_models: List[str]   # 지원 모델 목록
    max_concurrent_requests: int  # 최대 동시 요청

    # LLM Pricing (OpenRouter 스타일 - 직접 노출)
    token_input_rate: Decimal     # Input token 단가
    token_output_rate: Decimal    # Output token 단가 (보통 8배)
    cached_input_discount: float = 0.9  # Cached input 90% 할인
    request_base_rate: Decimal    # Request 기본 단가
    enable_batch_processing: bool = True
    batch_discount: float = 0.5   # Batch processing 50% 할인

    # Context Memory Service (LLM 컨텍스트용)
    context_memory_capacity_gb: int
    context_memory_hourly_rate: Decimal

    # Context Storage Service (LLM 데이터 영구 저장용)
    context_storage_capacity_tb: int
    context_storage_hourly_rate: Decimal

    # General Purpose Memory Service (범용 메모리)
    gp_memory_capacity_gb: int
    gp_memory_hourly_rate: Decimal

    # General Purpose Storage Service (범용 스토리지)
    gp_storage_capacity_tb: int
    gp_storage_hot_hourly_rate: Decimal   # Hot tier (빠른 액세스)
    gp_storage_cold_hourly_rate: Decimal  # Cold tier (아카이브)

    # Health & Metrics
    heartbeat_interval_sec: int = 10
    metric_report_interval_sec: int = 60

class ServiceType(Enum):
    """서비스 타입 정의"""
    LLM_API = "llm_api"
    CONTEXT_MEMORY = "context_memory"
    CONTEXT_STORAGE = "context_storage"
    GP_MEMORY = "gp_memory"
    GP_STORAGE = "gp_storage"
```

### 2.3 Provider API Endpoints

#### 2.3.1 LLM Inference API
```
POST /v1/completions
POST /v1/chat/completions
POST /v1/embeddings
POST /v1/batch/completions        # Batch processing (50% 할인)
```

**Request Headers**:
```
Authorization: Bearer <provider_token>
X-Request-ID: <unique_request_id>
X-User-ID: <end_user_id>
X-Enable-Cache: true              # Prompt caching 활성화
```

**Response Headers**:
```
X-Provider-ID: <provider_id>
X-Token-Usage-Input: <input_tokens>
X-Token-Usage-Input-Cached: <cached_input_tokens>
X-Token-Usage-Output: <output_tokens>
X-Latency-Ms: <latency>
X-Cache-Hit: <true|false>
```

#### 2.3.2 Context Memory Service API (LLM 컨텍스트용)
```
# KV Cache 관리
POST /v1/context/memory/allocate
{
  "user_id": "user_123",
  "session_id": "session_456",
  "size_gb": 2.0,
  "ttl_hours": 24
}

DELETE /v1/context/memory/release/{session_id}

GET /v1/context/memory/status/{user_id}

# Conversation History 저장/조회
POST /v1/context/memory/conversations
{
  "user_id": "user_123",
  "conversation_id": "conv_789",
  "messages": [...]
}

GET /v1/context/memory/conversations/{conversation_id}
```

#### 2.3.3 Context Storage Service API (LLM 데이터 저장용)
```
# Chat History 영구 저장
POST /v1/context/storage/history
{
  "user_id": "user_123",
  "data": {...}
}

# Fine-tuning 데이터 저장
POST /v1/context/storage/training-data
{
  "user_id": "user_123",
  "dataset": [...]
}

# User Preferences 저장
PUT /v1/context/storage/preferences/{user_id}
{
  "preferences": {...}
}

GET /v1/context/storage/{user_id}/{data_type}
DELETE /v1/context/storage/{user_id}/{data_type}
```

#### 2.3.4 General Purpose Memory Service API (범용 메모리)
```
# 메모리 할당 (서버 실행용)
POST /v1/gp/memory/allocate
{
  "user_id": "user_123",
  "resource_id": "app_instance_456",
  "size_gb": 16.0,
  "purpose": "application"  # application, cache, database
}

PUT /v1/gp/memory/resize/{resource_id}
{
  "new_size_gb": 32.0
}

DELETE /v1/gp/memory/release/{resource_id}

GET /v1/gp/memory/status/{user_id}
```

#### 2.3.5 General Purpose Storage Service API (범용 스토리지)
```
# 파일 업로드 (Hot/Cold tier 선택)
POST /v1/gp/storage/upload
Content-Type: multipart/form-data
X-Storage-Tier: hot  # hot or cold

{
  "file": <binary>,
  "user_id": "user_123",
  "path": "/data/myfile.db"
}

# 파일 다운로드
GET /v1/gp/storage/download/{user_id}/{path}

# Tier 변경 (Hot ↔ Cold)
PUT /v1/gp/storage/tier/{user_id}/{path}
{
  "new_tier": "cold"
}

DELETE /v1/gp/storage/{user_id}/{path}

GET /v1/gp/storage/status/{user_id}
```

### 2.4 Metric Collection

Provider는 다음 metric을 수집하여 Router에 주기적으로 전송:

```python
class ProviderMetrics:
    timestamp: datetime
    provider_id: str

    # LLM Metrics
    request_count: int
    total_input_tokens: int
    total_cached_input_tokens: int  # Cached input tokens
    total_output_tokens: int
    batch_request_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_count: int
    cache_hit_rate: float

    # Context Memory Metrics
    context_memory_allocated_gb: float
    context_memory_sessions: int

    # Context Storage Metrics
    context_storage_used_tb: float
    context_storage_objects: int

    # General Purpose Memory Metrics
    gp_memory_allocated_gb: float
    gp_memory_available_gb: float
    gp_memory_instances: int

    # General Purpose Storage Metrics
    gp_storage_hot_used_tb: float
    gp_storage_cold_used_tb: float
    gp_storage_total_objects: int

    # System Health
    cpu_usage_pct: float
    gpu_usage_pct: float
    gpu_memory_usage_pct: float
    uptime_hours: float
```

---

## 3. Central Router 상세 설계

### 3.1 Router 역할
1. **요청 라우팅**: 최적의 Provider 선택 및 트래픽 분산
2. **Metric 수집**: 모든 Provider의 metric 수집 및 집계
3. **정산 처리**: Commission 기반 Provider 수익 계산
4. **인증/인가**: API key 관리 및 접근 제어

### 3.2 Router 구성 요소

#### 3.2.1 Request Router

**라우팅 전략**:
```python
class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"           # 순차 분배
    LEAST_LATENCY = "least_latency"       # 최저 지연시간
    LEAST_LOADED = "least_loaded"         # 최저 부하
    COST_OPTIMIZED = "cost_optimized"     # 비용 최적화
    WEIGHTED = "weighted"                  # 가중치 기반

class RouterConfig:
    strategy: RoutingStrategy = RoutingStrategy.LEAST_LATENCY
    health_check_interval_sec: int = 30
    provider_timeout_sec: int = 300
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    fallback_on_failure: bool = True  # 실패 시 다른 Provider로 fallback
```

**Provider 선택 알고리즘**:
```python
def select_provider(
    request: Request,
    providers: List[Provider],
    strategy: RoutingStrategy
) -> Provider:
    """
    요청에 대한 최적의 Provider 선택

    고려 요소:
    1. Provider 가용성 (health check)
    2. 지원 모델 (requested model)
    3. 현재 부하 (concurrent requests)
    4. 과거 성능 (latency, success rate)
    5. 비용 (pricing)
    """
    # Filter: 가용하고 요청 모델을 지원하는 Provider
    available = [p for p in providers
                 if p.is_healthy and request.model in p.supported_models]

    if not available:
        raise NoProviderAvailable()

    if strategy == RoutingStrategy.LEAST_LATENCY:
        return min(available, key=lambda p: p.avg_latency_ms)

    elif strategy == RoutingStrategy.LEAST_LOADED:
        return min(available,
                   key=lambda p: p.current_requests / p.max_requests)

    elif strategy == RoutingStrategy.COST_OPTIMIZED:
        return min(available, key=lambda p: p.calculate_cost(request))

    elif strategy == RoutingStrategy.WEIGHTED:
        # Weight = (1 - normalized_latency) * 0.4
        #        + (1 - normalized_load) * 0.3
        #        + (1 - normalized_cost) * 0.3
        weights = [calculate_weight(p, request) for p in available]
        return random.choices(available, weights=weights)[0]
```

#### 3.2.2 Metric Collector

**수집 구조**:
```python
class MetricCollector:
    """Provider로부터 metric을 수집하고 집계"""

    def __init__(self, db: Database, redis: Redis):
        self.db = db                    # 장기 저장용 DB
        self.redis = redis              # 실시간 집계용 cache
        self.aggregation_window = 60    # 60초 단위 집계

    async def collect_metrics(self, provider_id: str, metrics: ProviderMetrics):
        """Provider로부터 metric 수신 및 저장"""
        # Redis에 실시간 metric 저장
        await self.redis.zadd(
            f"metrics:{provider_id}",
            {metrics.timestamp.timestamp(): json.dumps(metrics.dict())}
        )

        # DB에 집계된 metric 저장 (시간별)
        await self.db.insert_metric(metrics)

    async def get_provider_stats(
        self,
        provider_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> ProviderStats:
        """Provider의 통계 조회"""
        metrics = await self.db.query_metrics(
            provider_id, start_time, end_time
        )

        return ProviderStats(
            total_requests=sum(m.request_count for m in metrics),
            total_input_tokens=sum(m.total_input_tokens for m in metrics),
            total_cached_input_tokens=sum(m.total_cached_input_tokens for m in metrics),
            total_output_tokens=sum(m.total_output_tokens for m in metrics),
            avg_latency_ms=statistics.mean(m.avg_latency_ms for m in metrics),
            cache_hit_rate=statistics.mean(m.cache_hit_rate for m in metrics),
            uptime_pct=calculate_uptime(metrics),
            revenue=calculate_revenue(metrics)
        )
```

#### 3.2.3 Billing Engine (OpenRouter 스타일 Commission)

**정산 로직**:
```python
from decimal import Decimal

class BillingEngine:
    """Provider 수익 정산 및 사용자 결제 처리 (OpenRouter 스타일)"""

    # Platform commission rates
    LLM_API_COMMISSION = Decimal("0.05")      # 5% for LLM API
    MEMORY_STORAGE_MARGIN = Decimal("0.20")   # 20% for Memory/Storage

    def calculate_llm_cost(
        self,
        input_tokens: int,
        cached_input_tokens: int,
        output_tokens: int,
        provider: Provider,
        is_batch: bool = False
    ) -> Decimal:
        """LLM API 비용 계산"""
        # Fresh input cost
        fresh_input_cost = (
            (input_tokens - cached_input_tokens) *
            provider.token_input_rate
        )

        # Cached input cost (90% 할인)
        cached_input_cost = (
            cached_input_tokens *
            provider.token_input_rate *
            Decimal("0.1")  # 10%만 부과
        )

        # Output cost (보통 input의 8배)
        output_cost = output_tokens * provider.token_output_rate

        # Base request cost
        base_cost = provider.request_base_rate

        total = fresh_input_cost + cached_input_cost + output_cost + base_cost

        # Batch processing 50% 할인
        if is_batch:
            total *= Decimal("0.5")

        return total

    def calculate_provider_revenue(
        self,
        provider_id: str,
        start_date: date,
        end_date: date
    ) -> ProviderRevenue:
        """Provider 수익 계산 (OpenRouter 스타일)"""
        stats = self.metric_collector.get_provider_stats(
            provider_id, start_date, end_date
        )

        # LLM API 수익 (provider가 받을 금액 - commission 제외 전)
        llm_gross_revenue = Decimal("0")

        # 실제 요청 내역에서 계산
        requests = self.get_provider_requests(provider_id, start_date, end_date)
        for req in requests:
            llm_gross_revenue += self.calculate_llm_cost(
                req.input_tokens,
                req.cached_input_tokens,
                req.output_tokens,
                provider,
                req.is_batch
            )

        # Context Memory 수익 (독립 가격)
        context_memory_hours = self.calculate_context_memory_hours(
            provider_id, start_date, end_date
        )
        context_memory_revenue = (
            context_memory_hours * provider.context_memory_hourly_rate
        )

        # Context Storage 수익 (독립 가격)
        context_storage_hours = self.calculate_context_storage_hours(
            provider_id, start_date, end_date
        )
        context_storage_revenue = (
            context_storage_hours * provider.context_storage_hourly_rate
        )

        # GP Memory 수익 (독립 가격)
        gp_memory_hours = self.calculate_gp_memory_hours(
            provider_id, start_date, end_date
        )
        gp_memory_revenue = gp_memory_hours * provider.gp_memory_hourly_rate

        # GP Storage 수익 (Hot/Cold 분리)
        gp_storage_usage = self.calculate_gp_storage_hours(
            provider_id, start_date, end_date
        )
        gp_storage_revenue = (
            gp_storage_usage['hot'] * provider.gp_storage_hot_hourly_rate +
            gp_storage_usage['cold'] * provider.gp_storage_cold_hourly_rate
        )

        # 총 수익
        total_revenue = (
            llm_gross_revenue +
            context_memory_revenue +
            context_storage_revenue +
            gp_memory_revenue +
            gp_storage_revenue
        )

        # Platform 수수료 (서비스별 차등)
        llm_commission = llm_gross_revenue * self.LLM_API_COMMISSION
        memory_storage_fee = (
            (context_memory_revenue + context_storage_revenue +
             gp_memory_revenue + gp_storage_revenue) *
            self.MEMORY_STORAGE_MARGIN
        )
        platform_fee = llm_commission + memory_storage_fee

        # Provider 순수익
        net_revenue = total_revenue - platform_fee

        return ProviderRevenue(
            provider_id=provider_id,
            period_start=start_date,
            period_end=end_date,
            llm_gross_revenue=llm_gross_revenue,
            llm_commission=llm_commission,
            context_memory_revenue=context_memory_revenue,
            context_storage_revenue=context_storage_revenue,
            gp_memory_revenue=gp_memory_revenue,
            gp_storage_revenue=gp_storage_revenue,
            total_revenue=total_revenue,
            platform_fee=platform_fee,
            net_revenue=net_revenue
        )

    def calculate_user_charge(
        self,
        user_id: str,
        start_date: date,
        end_date: date
    ) -> UserCharge:
        """사용자 결제 금액 계산"""
        usage = self.get_user_usage(user_id, start_date, end_date)

        # LLM API: Provider 원가 + 5% commission
        llm_charge = Decimal("0")
        for req in usage.requests:
            provider_cost = self.calculate_llm_cost(
                req.input_tokens,
                req.cached_input_tokens,
                req.output_tokens,
                req.provider,
                req.is_batch
            )
            llm_charge += provider_cost * (Decimal("1") + self.LLM_API_COMMISSION)

        # Memory/Storage: Provider 원가 + 20% margin
        context_memory_charge = (
            usage.context_memory_hours * usage.provider.context_memory_hourly_rate *
            (Decimal("1") + self.MEMORY_STORAGE_MARGIN)
        )

        context_storage_charge = (
            usage.context_storage_hours * usage.provider.context_storage_hourly_rate *
            (Decimal("1") + self.MEMORY_STORAGE_MARGIN)
        )

        gp_memory_charge = (
            usage.gp_memory_hours * usage.provider.gp_memory_hourly_rate *
            (Decimal("1") + self.MEMORY_STORAGE_MARGIN)
        )

        gp_storage_charge = (
            (usage.gp_storage_hot_hours * usage.provider.gp_storage_hot_hourly_rate +
             usage.gp_storage_cold_hours * usage.provider.gp_storage_cold_hourly_rate) *
            (Decimal("1") + self.MEMORY_STORAGE_MARGIN)
        )

        total_charge = (
            llm_charge +
            context_memory_charge +
            context_storage_charge +
            gp_memory_charge +
            gp_storage_charge
        )

        return UserCharge(
            user_id=user_id,
            period_start=start_date,
            period_end=end_date,
            llm_charge=llm_charge,
            context_memory_charge=context_memory_charge,
            context_storage_charge=context_storage_charge,
            gp_memory_charge=gp_memory_charge,
            gp_storage_charge=gp_storage_charge,
            total_charge=total_charge
        )
```

---

## 4. API Specification

### 4.1 User-facing API (Router)

#### 4.1.1 LLM Inference
```
POST /v1/chat/completions
Authorization: Bearer <user_api_key>
Content-Type: application/json

{
  "model": "llama-3-70b",
  "messages": [...],
  "temperature": 0.7,
  "max_tokens": 1000,
  "enable_cache": true  # Prompt caching 활성화
}

Response:
{
  "id": "req_xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "llama-3-70b",
  "choices": [...],
  "usage": {
    "prompt_tokens": 100,
    "prompt_tokens_cached": 50,  # Cached tokens
    "completion_tokens": 200,
    "total_tokens": 300
  },
  "x_provider_id": "provider_123",
  "x_latency_ms": 1234,
  "x_cache_hit": true
}
```

#### 4.1.2 Batch Processing (50% 할인)
```
POST /v1/batch/completions
Authorization: Bearer <user_api_key>
Content-Type: application/json

{
  "requests": [
    {
      "model": "llama-3-70b",
      "messages": [...],
      "custom_id": "req_1"
    },
    ...
  ],
  "max_delay_hours": 24  # 최대 24시간 이내 처리
}

Response:
{
  "batch_id": "batch_xxx",
  "status": "queued",
  "estimated_completion": "2025-11-11T12:00:00Z",
  "discount_applied": 0.5  # 50% 할인
}
```

#### 4.1.3 Context Memory (LLM 컨텍스트)
```
POST /v1/context/memory/sessions
Authorization: Bearer <user_api_key>
Content-Type: application/json

{
  "size_gb": 2.0,
  "ttl_hours": 24
}

Response:
{
  "session_id": "session_xxx",
  "allocated_gb": 2.0,
  "expires_at": "2025-11-11T12:00:00Z",
  "hourly_rate_usd": 0.036  # Provider 단가 + 20%
}
```

#### 4.1.4 GP Storage (범용 스토리지)
```
POST /v1/gp/storage/files
Authorization: Bearer <user_api_key>
Content-Type: multipart/form-data
X-Storage-Tier: hot

file: <binary>

Response:
{
  "file_id": "file_xxx",
  "path": "/user_123/mydata.db",
  "size_gb": 5.0,
  "tier": "hot",
  "hourly_rate_usd": 0.0012
}
```

#### 4.1.5 사용량 조회
```
GET /v1/usage?start_date=2025-11-01&end_date=2025-11-30
Authorization: Bearer <user_api_key>

Response:
{
  "user_id": "user_xxx",
  "period": {
    "start": "2025-11-01",
    "end": "2025-11-30"
  },
  "usage": {
    "llm_api": {
      "total_requests": 1000,
      "total_input_tokens": 50000,
      "total_cached_input_tokens": 10000,
      "total_output_tokens": 100000,
      "batch_requests": 100
    },
    "context_memory_hours": 720.0,
    "context_storage_hours": 1440.0,
    "gp_memory_hours": 360.0,
    "gp_storage": {
      "hot_hours": 500.0,
      "cold_hours": 940.0
    }
  },
  "charges": {
    "llm_api": 45.00,
    "context_memory": 25.92,
    "context_storage": 17.28,
    "gp_memory": 10.80,
    "gp_storage": 1.13,
    "total": 100.13,
    "currency": "USD"
  }
}
```

### 4.2 Provider-facing API (Router)

#### 4.2.1 Provider 등록
```
POST /v1/providers/register
Content-Type: application/json

{
  "provider_name": "My LLM Provider",
  "endpoint_url": "https://provider.example.com",
  "supported_models": ["llama-3-70b", "llama-3-8b"],
  "pricing": {
    "token_input_rate": 0.000001,    # $1 per 1M tokens
    "token_output_rate": 0.000008,   # $8 per 1M tokens (8배)
    "cached_input_discount": 0.9,
    "request_base_rate": 0.01,
    "batch_discount": 0.5,

    "context_memory_hourly_rate": 0.03,
    "context_storage_hourly_rate": 0.012,
    "gp_memory_hourly_rate": 0.009,
    "gp_storage_hot_hourly_rate": 0.001,
    "gp_storage_cold_hourly_rate": 0.0002
  },
  "capacity": {
    "max_concurrent_requests": 100,
    "context_memory_capacity_gb": 50,
    "context_storage_capacity_tb": 5,
    "gp_memory_capacity_gb": 200,
    "gp_storage_capacity_tb": 20
  }
}

Response:
{
  "provider_id": "provider_xxx",
  "provider_token": "prov_tok_xxx",
  "status": "registered"
}
```

#### 4.2.2 수익 조회
```
GET /v1/providers/{provider_id}/revenue?start_date=2025-11-01&end_date=2025-11-30
Authorization: Bearer <provider_token>

Response:
{
  "provider_id": "provider_xxx",
  "period": {
    "start": "2025-11-01",
    "end": "2025-11-30"
  },
  "revenue": {
    "llm_gross_revenue": 1000.00,
    "llm_commission": 50.00,  # 5%
    "context_memory_revenue": 200.00,
    "context_storage_revenue": 150.00,
    "gp_memory_revenue": 80.00,
    "gp_storage_revenue": 70.00,
    "total_gross_revenue": 1500.00,
    "platform_fee": 150.00,  # 5% LLM + 20% Memory/Storage
    "net_revenue": 1350.00,
    "currency": "USD"
  },
  "stats": {
    "total_requests": 10000,
    "total_input_tokens": 500000,
    "total_cached_input_tokens": 100000,
    "total_output_tokens": 1000000,
    "avg_latency_ms": 234.5,
    "cache_hit_rate": 0.20,
    "uptime_pct": 99.9
  }
}
```

---

## 5. Database Schema

### 5.1 Providers Table
```sql
CREATE TABLE providers (
    provider_id VARCHAR(64) PRIMARY KEY,
    provider_name VARCHAR(255) NOT NULL,
    endpoint_url VARCHAR(512) NOT NULL,
    provider_token VARCHAR(128) NOT NULL UNIQUE,
    status VARCHAR(32) NOT NULL,  -- active, suspended, inactive

    -- LLM Pricing
    token_input_rate DECIMAL(12, 10) NOT NULL,
    token_output_rate DECIMAL(12, 10) NOT NULL,
    cached_input_discount DECIMAL(3, 2) DEFAULT 0.9,
    request_base_rate DECIMAL(10, 6) NOT NULL,
    batch_discount DECIMAL(3, 2) DEFAULT 0.5,

    -- Context Memory/Storage Pricing
    context_memory_hourly_rate DECIMAL(10, 6) NOT NULL,
    context_storage_hourly_rate DECIMAL(10, 6) NOT NULL,

    -- GP Memory/Storage Pricing
    gp_memory_hourly_rate DECIMAL(10, 6) NOT NULL,
    gp_storage_hot_hourly_rate DECIMAL(10, 6) NOT NULL,
    gp_storage_cold_hourly_rate DECIMAL(10, 6) NOT NULL,

    -- Capacity
    max_concurrent_requests INT NOT NULL,
    context_memory_capacity_gb INT NOT NULL,
    context_storage_capacity_tb INT NOT NULL,
    gp_memory_capacity_gb INT NOT NULL,
    gp_storage_capacity_tb INT NOT NULL,

    -- Metadata
    supported_models JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    INDEX idx_status (status),
    INDEX idx_created_at (created_at)
);
```

### 5.2 Users Table
```sql
CREATE TABLE users (
    user_id VARCHAR(64) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    api_key VARCHAR(128) NOT NULL UNIQUE,
    status VARCHAR(32) NOT NULL,  -- active, suspended, inactive
    tier VARCHAR(32) NOT NULL DEFAULT 'free',  -- free, starter, pro, enterprise

    -- Billing
    billing_email VARCHAR(255),
    payment_method_id VARCHAR(128),
    credit_balance DECIMAL(12, 2) DEFAULT 0.00,  # Prepaid credits

    -- Limits
    rate_limit_per_minute INT DEFAULT 60,
    monthly_budget_usd DECIMAL(10, 2),

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    INDEX idx_api_key (api_key),
    INDEX idx_status (status),
    INDEX idx_tier (tier)
);
```

### 5.3 Requests Table (Log)
```sql
CREATE TABLE requests (
    request_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    provider_id VARCHAR(64) NOT NULL,

    -- Request Info
    model VARCHAR(128) NOT NULL,
    endpoint VARCHAR(128) NOT NULL,
    is_batch BOOLEAN DEFAULT FALSE,

    -- Token Usage
    input_tokens INT NOT NULL,
    cached_input_tokens INT DEFAULT 0,
    output_tokens INT NOT NULL,
    latency_ms INT NOT NULL,

    -- Cost Breakdown
    provider_cost_usd DECIMAL(12, 8) NOT NULL,
    commission_usd DECIMAL(12, 8) NOT NULL,
    user_charge_usd DECIMAL(12, 8) NOT NULL,

    -- Status
    status VARCHAR(32) NOT NULL,  -- success, error, timeout
    error_message TEXT,

    -- Timestamp
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id),

    INDEX idx_user_created (user_id, created_at),
    INDEX idx_provider_created (provider_id, created_at),
    INDEX idx_created_at (created_at),
    INDEX idx_is_batch (is_batch)
);
```

### 5.4 Resource Usage Table (Memory/Storage)
```sql
CREATE TABLE resource_usage (
    usage_id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    provider_id VARCHAR(64) NOT NULL,

    -- Resource Info
    resource_type VARCHAR(32) NOT NULL,  -- context_memory, context_storage, gp_memory, gp_storage
    resource_id VARCHAR(128) NOT NULL,

    -- Usage Metrics
    allocated_size_gb FLOAT,
    storage_tier VARCHAR(16),  -- hot, cold (for gp_storage)

    -- Time Tracking
    started_at TIMESTAMP NOT NULL,
    ended_at TIMESTAMP,
    hours_used FLOAT,

    -- Cost
    hourly_rate_usd DECIMAL(10, 6) NOT NULL,
    total_cost_usd DECIMAL(12, 8) NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id),

    INDEX idx_user_resource (user_id, resource_type),
    INDEX idx_provider_resource (provider_id, resource_type),
    INDEX idx_resource_id (resource_id),
    INDEX idx_time_range (started_at, ended_at)
);
```

### 5.5 Provider Revenue Table
```sql
CREATE TABLE provider_revenue (
    revenue_id BIGSERIAL PRIMARY KEY,
    provider_id VARCHAR(64) NOT NULL,

    -- Period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Revenue Breakdown
    llm_gross_revenue DECIMAL(12, 2) NOT NULL,
    llm_commission DECIMAL(12, 2) NOT NULL,
    context_memory_revenue DECIMAL(12, 2) NOT NULL,
    context_storage_revenue DECIMAL(12, 2) NOT NULL,
    gp_memory_revenue DECIMAL(12, 2) NOT NULL,
    gp_storage_revenue DECIMAL(12, 2) NOT NULL,
    total_gross_revenue DECIMAL(12, 2) NOT NULL,

    -- Fees
    platform_fee DECIMAL(12, 2) NOT NULL,
    net_revenue DECIMAL(12, 2) NOT NULL,

    -- Payment Status
    payment_status VARCHAR(32) NOT NULL,  -- pending, paid, failed
    paid_at TIMESTAMP,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY (provider_id) REFERENCES providers(provider_id),

    UNIQUE (provider_id, period_start, period_end),
    INDEX idx_payment_status (payment_status)
);
```

### 5.6 User Charges Table
```sql
CREATE TABLE user_charges (
    charge_id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,

    -- Period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Charge Breakdown
    llm_charge DECIMAL(12, 2) NOT NULL,
    context_memory_charge DECIMAL(12, 2) NOT NULL,
    context_storage_charge DECIMAL(12, 2) NOT NULL,
    gp_memory_charge DECIMAL(12, 2) NOT NULL,
    gp_storage_charge DECIMAL(12, 2) NOT NULL,
    total_charge DECIMAL(12, 2) NOT NULL,

    -- Payment Status
    payment_status VARCHAR(32) NOT NULL,  -- pending, paid, failed, refunded
    payment_method VARCHAR(64),
    transaction_id VARCHAR(128),
    paid_at TIMESTAMP,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY (user_id) REFERENCES users(user_id),

    UNIQUE (user_id, period_start, period_end),
    INDEX idx_user_period (user_id, period_start),
    INDEX idx_payment_status (payment_status)
);
```

---

## 6. 보안 및 인증

### 6.1 API Key 관리
- **User API Key**: `user_` prefix, 사용자 인증용
- **Provider Token**: `prov_tok_` prefix, Provider 인증용
- 모든 key는 SHA-256 해시 후 DB 저장
- Rate limiting: Redis 기반 token bucket 알고리즘

### 6.2 요청 검증
```python
async def authenticate_user(api_key: str) -> User:
    """사용자 API key 검증"""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    user = await db.get_user_by_api_key_hash(key_hash)

    if not user or user.status != "active":
        raise Unauthorized("Invalid or inactive API key")

    # Rate limiting check (tier별 다름)
    tier_limits = {
        "free": 60,
        "starter": 300,
        "pro": 1000,
        "enterprise": 10000
    }
    rate_limit = tier_limits[user.tier]

    if not await rate_limiter.check(user.user_id, rate_limit):
        raise RateLimitExceeded()

    return user

async def authenticate_provider(provider_token: str) -> Provider:
    """Provider token 검증"""
    token_hash = hashlib.sha256(provider_token.encode()).hexdigest()
    provider = await db.get_provider_by_token_hash(token_hash)

    if not provider or provider.status != "active":
        raise Unauthorized("Invalid or inactive provider token")

    return provider
```

### 6.3 데이터 암호화
- **전송 중**: TLS 1.3 강제 사용
- **저장 중**: 민감 정보(API key, payment info) AES-256 암호화
- **로그**: PII 정보 마스킹 또는 제외

---

## 7. 모니터링 및 알림

### 7.1 Health Check
```python
class HealthChecker:
    """Provider health 상태 모니터링"""

    async def check_provider_health(self, provider: Provider) -> HealthStatus:
        """Provider health check"""
        try:
            start = time.time()
            response = await http_client.get(
                f"{provider.endpoint_url}/health",
                timeout=5.0
            )
            latency_ms = (time.time() - start) * 1000

            is_healthy = (
                response.status_code == 200 and
                latency_ms < provider.max_latency_ms and
                response.json().get("status") == "healthy"
            )

            return HealthStatus(
                provider_id=provider.provider_id,
                is_healthy=is_healthy,
                latency_ms=latency_ms,
                checked_at=datetime.now()
            )
        except Exception as e:
            return HealthStatus(
                provider_id=provider.provider_id,
                is_healthy=False,
                error=str(e),
                checked_at=datetime.now()
            )
```

### 7.2 알림 조건
- Provider down (3회 연속 health check 실패)
- Latency spike (P95 > 5초)
- Error rate > 5%
- Provider 수익 이상 (예상 대비 30% 이상 차이)
- 사용자 결제 실패
- Cache hit rate 급락 (< 10%)

---

## 8. 확장성 고려사항

### 8.1 수평적 확장
- **Router**: Stateless 설계, 여러 인스턴스 배포 가능
- **Load Balancer**: Nginx/HAProxy로 Router 인스턴스 분산
- **Database**: Read replica 구성, 쓰기는 master, 읽기는 replica

### 8.2 캐싱 전략
- **Provider 정보**: Redis에 5분 캐시
- **사용자 정보**: Redis에 10분 캐시
- **Metric 집계**: Redis에 실시간 저장, 1시간마다 DB에 영구 저장
- **Prompt Cache**: Provider별 독립 캐시 (5분/1시간 TTL)

### 8.3 비동기 처리
- **Metric 수집**: Kafka/RabbitMQ로 비동기 처리
- **정산 계산**: Celery/RQ로 배치 처리 (일별/월별)
- **알림 전송**: 비동기 worker로 처리
- **Batch Processing**: 24시간 이내 비동기 처리 큐

---

## 9. 배포 전략

### 9.1 Infrastructure
```yaml
# docker-compose.yml 예시
version: '3.8'

services:
  router:
    image: distributed-llm-router:latest
    deploy:
      replicas: 3
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://...
      - KAFKA_BROKERS=kafka:9092
    ports:
      - "8000:8000"

  provider-agent:
    image: distributed-llm-provider:latest
    environment:
      - ROUTER_URL=http://router:8000
      - PROVIDER_TOKEN=${PROVIDER_TOKEN}
    volumes:
      - ./models:/models
      - ./context-storage:/context-storage
      - ./gp-storage:/gp-storage

  postgres:
    image: postgres:15
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7
    volumes:
      - redisdata:/data

  kafka:
    image: confluentinc/cp-kafka:latest

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
```

### 9.2 CI/CD
- **테스트**: Unit tests, Integration tests, E2E tests
- **배포**: Blue-Green 배포 또는 Canary 배포
- **롤백**: 자동 health check 실패 시 롤백

---

## 10. 가격 정책 (OpenRouter 스타일 + 티어별)

### 10.1 Provider 단가 (투명하게 공개)

```
LLM API:
  - Input token: $1.00 per 1M tokens ($0.000001/token)
  - Cached input: $0.10 per 1M tokens (90% 할인)
  - Output token: $8.00 per 1M tokens ($0.000008/token, 8배)
  - Request base: $0.01/request

Batch Processing (24시간 이내):
  - 모든 LLM 요금 50% 할인

Context Memory (LLM 컨텍스트용):
  - $0.03/GB/hour

Context Storage (LLM 데이터 영구 저장):
  - $0.012/GB/hour

General Purpose Memory (범용 메모리):
  - $0.009/GB/hour

General Purpose Storage (범용 스토리지):
  - Hot tier: $0.001/GB/hour
  - Cold tier: $0.0002/GB/hour (80% 할인)
```

### 10.2 사용자 단가 (Provider 단가 + Commission/Margin)

```
LLM API (Provider 단가 + 5% commission):
  - Input token: $1.05 per 1M tokens
  - Cached input: $0.105 per 1M tokens
  - Output token: $8.40 per 1M tokens
  - Request base: $0.0105/request

Batch Processing:
  - 모든 LLM 요금 50% 할인 (Provider 할인 그대로 전달)

Context Memory (Provider 단가 + 20% margin):
  - $0.036/GB/hour

Context Storage (Provider 단가 + 20% margin):
  - $0.0144/GB/hour

General Purpose Memory (Provider 단가 + 20% margin):
  - $0.0108/GB/hour

General Purpose Storage (Provider 단가 + 20% margin):
  - Hot tier: $0.0012/GB/hour
  - Cold tier: $0.00024/GB/hour
```

### 10.3 플랫폼 수수료

```
LLM API:
  - 5% commission (OpenRouter와 동일)
  - 성공한 요청만 과금 (fallback 실패는 무료)

Memory & Storage:
  - 20% margin
  - OpenRouter에는 없는 독점 서비스
```

### 10.4 Tier별 혜택

```yaml
Free Tier:
  monthly_quota: 1M tokens
  rate_limit: 60 RPM
  features:
    - Basic routing
    - Standard support
    - 5% commission on LLM
  price: $0/month

Starter Tier:
  monthly_quota: 10M tokens
  rate_limit: 300 RPM
  features:
    - Advanced routing
    - Prompt caching
    - Priority support
    - 5% commission on LLM
  price: $20/month

Pro Tier:
  monthly_quota: 100M tokens
  rate_limit: 1000 RPM
  features:
    - All routing strategies
    - Prompt caching
    - Batch processing
    - Priority support
    - 5% commission on LLM
    - 20% off Memory/Storage
  price: $100/month

Enterprise Tier:
  monthly_quota: Unlimited
  rate_limit: Custom
  features:
    - All features
    - SLA 99.99%
    - Dedicated support
    - Custom commission (협상 가능 3-10%)
    - Volume discounts
    - White-label option
  price: Custom
```

### 10.5 비용 계산 예시

#### 시나리오: 한 달 사용량
```
LLM API:
- 10M input tokens
- 20M output tokens
- 1000 requests
- Batch: 100 requests (2M input, 4M output)

Context Memory:
- 평균 2GB, 24시간 사용 = 720 GB-hours

Context Storage:
- 평균 5GB, 30일 사용 = 3,600 GB-hours

GP Storage:
- Hot: 10GB, 30일 = 7,200 GB-hours
- Cold: 100GB, 30일 = 72,000 GB-hours
```

#### Provider 수익
```python
# LLM API (Provider 원가)
llm_revenue = (
    10_000_000 * 0.000001 +          # Input: $10.00
    20_000_000 * 0.000008 +          # Output: $160.00
    1000 * 0.01 +                    # Requests: $10.00
    (2_000_000 * 0.000001 +
     4_000_000 * 0.000008) * 0.5     # Batch (50% 할인): $18.00
) = $198.00

# Memory & Storage
context_memory = 720 * 0.03 = $21.60
context_storage = 3600 * 0.012 = $43.20
gp_storage = 7200 * 0.001 + 72000 * 0.0002 = $21.60

total_revenue = 198 + 21.60 + 43.20 + 21.60 = $284.40

# Platform fee
llm_commission = 198 * 0.05 = $9.90
memory_storage_fee = (21.60 + 43.20 + 21.60) * 0.20 = $17.28
platform_fee = 9.90 + 17.28 = $27.18

# Provider 순수익
net_revenue = 284.40 - 27.18 = $257.22
```

#### 사용자 결제
```python
# LLM API (Provider 원가 + 5%)
llm_charge = 198 * 1.05 = $207.90

# Memory & Storage (Provider 원가 + 20%)
memory_storage_charge = (21.60 + 43.20 + 21.60) * 1.20 = $103.68

# 총 결제 금액
total_charge = 207.90 + 103.68 = $311.58
```

#### 플랫폼 수익
```python
platform_revenue = total_charge - provider_net_revenue
                 = 311.58 - 257.22
                 = $54.36

profit_margin = 54.36 / 311.58 = 17.4%
```

---

## 11. 구현 우선순위

### Phase 1: MVP (4-6주)
1. ✅ Router 기본 구조 (FastAPI)
2. ✅ Provider 등록 및 인증
3. ✅ 단순 라운드로빈 라우팅
4. ✅ 기본 LLM API proxy (commission 5%)
5. ✅ Metric 수집 (기본)
6. ✅ PostgreSQL + Redis 연동

### Phase 2: Core Features (6-8주)
1. ✅ 고급 라우팅 전략 (least latency, cost optimized)
2. ✅ Prompt caching 지원
3. ✅ Batch processing (50% 할인)
4. ✅ Context Memory/Storage 서비스
5. ✅ GP Memory/Storage 서비스 (Hot/Cold tier)
6. ✅ 상세 metric 수집 및 집계
7. ✅ Provider 수익 정산 로직 (commission 기반)
8. ✅ 사용자 결제 로직
9. ✅ Health check & monitoring

### Phase 3: Production Ready (8-10주)
1. ✅ Prepaid credit system
2. ✅ 결제 시스템 연동 (Stripe/PayPal)
3. ✅ Dashboard (Provider/User)
4. ✅ 알림 시스템
5. ✅ 로깅 및 감사
6. ✅ 보안 강화 (rate limiting, DDoS 방어)
7. ✅ 부하 테스트 및 최적화
8. ✅ Tier별 기능 제한

### Phase 4: Advanced Features (추가)
1. ✅ Provider SLA 모니터링
2. ✅ 자동 스케일링
3. ✅ Multi-region 지원
4. ✅ ML 기반 라우팅 최적화
5. ✅ Marketplace (Provider 랭킹, 리뷰)
6. ✅ White-label API (Enterprise)

---

## 12. OpenRouter와의 차별점

| 기능 | OpenRouter | 우리 플랫폼 |
|------|-----------|-----------|
| **LLM API** | ✅ 5% commission | ✅ 5% commission (동일) |
| **Prompt Caching** | ✅ 지원 | ✅ 지원 (90% 할인) |
| **Batch Processing** | ❌ 없음 | ✅ 50% 할인 |
| **Context Memory** | ❌ 없음 | ✅ 독립 서비스 |
| **Context Storage** | ❌ 없음 | ✅ 독립 서비스 |
| **GP Memory** | ❌ 없음 | ✅ 독립 서비스 |
| **GP Storage** | ❌ 없음 | ✅ Hot/Cold tier |
| **Provider 생태계** | ❌ 고정 Provider | ✅ 누구나 Provider 가능 |
| **수익 모델** | Commission만 | Commission + Margin |
| **Fallback** | ✅ 지원 | ✅ 지원 (실패만 무료) |

---

## 13. 참고 자료

### 13.1 유사 플랫폼
- **OpenRouter**: LLM API aggregator (5% commission)
- **Helicone**: LLM observability platform
- **Portkey**: LLM gateway
- **Replicate**: AI model hosting (commission 기반)

### 13.2 기술 스택 추천
- **Backend**: FastAPI (Python) or Go
- **Database**: PostgreSQL (transactional), TimescaleDB (metrics)
- **Cache**: Redis (+ Redis Stack for vector DB)
- **Message Queue**: Kafka or RabbitMQ
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK stack (Elasticsearch, Logstash, Kibana)
- **Deployment**: Kubernetes or Docker Swarm
- **Object Storage**: MinIO (S3-compatible)

### 13.3 참고 문서
- OpenAI API Pricing: https://openai.com/api/pricing/
- Anthropic Claude Pricing: https://www.anthropic.com/pricing
- OpenRouter Pricing: https://openrouter.ai/pricing
- vLLM Documentation: https://docs.vllm.ai/
