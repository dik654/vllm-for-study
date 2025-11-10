# 분산 LLM 인프라 플랫폼 Specification

## 1. 시스템 개요

### 1.1 목적
여러 서버(Provider)가 LLM API, 메모리, 스토리지 리소스를 제공하고, 중앙 Router가 이를 통합하여 사용자에게 서비스를 제공하는 분산 인프라 플랫폼.

### 1.2 주요 기능
- **Provider**: LLM inference, KV cache, storage 리소스 제공
- **Router**: 요청 라우팅, metric 수집, 정산 처리
- **Billing**: 사용자 결제, Provider 수익 정산

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
   │ │Memory││          │ │Memory││          │ │Memory││
   │ └──────┘│          │ └──────┘│          │ └──────┘│
   │ ┌──────┐│          │ ┌──────┐│          │ ┌──────┐│
   │ │Storage│          │ │Storage│          │ │Storage│
   │ └──────┘│          │ └──────┘│          │ └──────┘│
   └─────────┘          └─────────┘          └─────────┘
```

---

## 2. Provider Node 상세 설계

### 2.1 Provider 역할
1. **LLM API Endpoint** - vLLM 기반 inference 서비스
2. **Memory Service** - Distributed KV cache 제공
3. **Storage Service** - Model weights, datasets 저장

### 2.2 Provider Agent 구성

```python
# provider_agent/config.py
class ProviderConfig:
    provider_id: str              # 고유 Provider ID
    provider_name: str            # Provider 이름
    router_url: str               # Central Router URL

    # LLM Service
    vllm_config: VllmConfig       # vLLM 설정
    supported_models: list[str]   # 지원 모델 목록
    max_concurrent_requests: int  # 최대 동시 요청

    # Memory Service
    memory_capacity_gb: int       # 제공 가능 메모리 (GB)
    memory_hourly_rate: Decimal   # 시간당 메모리 단가

    # Storage Service
    storage_capacity_tb: int      # 제공 가능 스토리지 (TB)
    storage_hourly_rate: Decimal  # 시간당 스토리지 단가

    # Pricing
    token_input_rate: Decimal     # Input token 단가
    token_output_rate: Decimal    # Output token 단가
    request_base_rate: Decimal    # Request 기본 단가

    # Health & Metrics
    heartbeat_interval_sec: int = 10
    metric_report_interval_sec: int = 60
```

### 2.3 Provider API Endpoints

#### 2.3.1 LLM Inference API
```
POST /v1/completions
POST /v1/chat/completions
POST /v1/embeddings
```

**Request Headers**:
```
Authorization: Bearer <provider_token>
X-Request-ID: <unique_request_id>
X-User-ID: <end_user_id>
```

**Response Headers**:
```
X-Provider-ID: <provider_id>
X-Token-Usage-Input: <input_tokens>
X-Token-Usage-Output: <output_tokens>
X-Latency-Ms: <latency>
```

#### 2.3.2 Memory Service API
```
POST /v1/memory/allocate    # 메모리 할당
DELETE /v1/memory/release   # 메모리 해제
GET /v1/memory/status       # 메모리 상태
```

#### 2.3.3 Storage Service API
```
POST /v1/storage/upload     # 파일 업로드
GET /v1/storage/download    # 파일 다운로드
DELETE /v1/storage/delete   # 파일 삭제
GET /v1/storage/status      # 스토리지 상태
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
    total_output_tokens: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_count: int

    # Memory Metrics
    memory_allocated_gb: float
    memory_available_gb: float
    memory_utilization_pct: float

    # Storage Metrics
    storage_used_tb: float
    storage_available_tb: float
    storage_utilization_pct: float

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
3. **정산 처리**: Provider 수익 계산 및 사용자 결제 처리
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
```

**Provider 선택 알고리즘**:
```python
def select_provider(
    request: Request,
    providers: list[Provider],
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
            total_output_tokens=sum(m.total_output_tokens for m in metrics),
            avg_latency_ms=statistics.mean(m.avg_latency_ms for m in metrics),
            uptime_pct=calculate_uptime(metrics),
            revenue=calculate_revenue(metrics)
        )
```

#### 3.2.3 Billing Engine

**정산 로직**:
```python
class BillingEngine:
    """Provider 수익 정산 및 사용자 결제 처리"""

    def calculate_provider_revenue(
        self,
        provider_id: str,
        start_date: date,
        end_date: date
    ) -> ProviderRevenue:
        """Provider 수익 계산"""
        stats = self.metric_collector.get_provider_stats(
            provider_id, start_date, end_date
        )

        # LLM API 수익
        llm_revenue = (
            stats.total_input_tokens * provider.token_input_rate +
            stats.total_output_tokens * provider.token_output_rate +
            stats.total_requests * provider.request_base_rate
        )

        # Memory 수익 (시간당 할당된 메모리)
        memory_hours = self.calculate_memory_usage_hours(provider_id, start_date, end_date)
        memory_revenue = memory_hours * provider.memory_hourly_rate

        # Storage 수익 (시간당 사용된 스토리지)
        storage_hours = self.calculate_storage_usage_hours(provider_id, start_date, end_date)
        storage_revenue = storage_hours * provider.storage_hourly_rate

        total_revenue = llm_revenue + memory_revenue + storage_revenue

        # 플랫폼 수수료 차감 (예: 20%)
        platform_fee = total_revenue * self.platform_fee_rate
        net_revenue = total_revenue - platform_fee

        return ProviderRevenue(
            provider_id=provider_id,
            period_start=start_date,
            period_end=end_date,
            llm_revenue=llm_revenue,
            memory_revenue=memory_revenue,
            storage_revenue=storage_revenue,
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

        # 회사 정책에 따른 요금 계산
        # Provider 단가에 마진을 추가
        margin_multiplier = 1.5  # 50% 마진

        llm_charge = (
            usage.total_input_tokens * self.company_token_input_rate +
            usage.total_output_tokens * self.company_token_output_rate +
            usage.total_requests * self.company_request_base_rate
        )

        memory_charge = usage.memory_hours * self.company_memory_hourly_rate
        storage_charge = usage.storage_hours * self.company_storage_hourly_rate

        total_charge = llm_charge + memory_charge + storage_charge

        return UserCharge(
            user_id=user_id,
            period_start=start_date,
            period_end=end_date,
            llm_charge=llm_charge,
            memory_charge=memory_charge,
            storage_charge=storage_charge,
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
  "max_tokens": 1000
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
    "completion_tokens": 200,
    "total_tokens": 300
  },
  "x_provider_id": "provider_123",
  "x_latency_ms": 1234
}
```

#### 4.1.2 사용량 조회
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
    "total_requests": 1000,
    "total_input_tokens": 50000,
    "total_output_tokens": 100000,
    "memory_hours": 720.0,
    "storage_hours": 1440.0
  },
  "charges": {
    "llm": 45.00,
    "memory": 15.00,
    "storage": 10.00,
    "total": 70.00,
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
    "token_input_rate": 0.0001,
    "token_output_rate": 0.0002,
    "request_base_rate": 0.01,
    "memory_hourly_rate": 0.02,
    "storage_hourly_rate": 0.001
  },
  "capacity": {
    "max_concurrent_requests": 100,
    "memory_capacity_gb": 100,
    "storage_capacity_tb": 10
  }
}

Response:
{
  "provider_id": "provider_xxx",
  "provider_token": "prov_tok_xxx",
  "status": "registered"
}
```

#### 4.2.2 Metric 전송
```
POST /v1/providers/{provider_id}/metrics
Authorization: Bearer <provider_token>
Content-Type: application/json

{
  "timestamp": "2025-11-10T12:00:00Z",
  "metrics": {
    "request_count": 100,
    "total_input_tokens": 5000,
    "total_output_tokens": 10000,
    "avg_latency_ms": 234.5,
    "memory_allocated_gb": 50.0,
    "storage_used_tb": 5.0,
    ...
  }
}
```

#### 4.2.3 수익 조회
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
    "llm_revenue": 1000.00,
    "memory_revenue": 300.00,
    "storage_revenue": 100.00,
    "total_revenue": 1400.00,
    "platform_fee": 280.00,
    "net_revenue": 1120.00,
    "currency": "USD"
  },
  "stats": {
    "total_requests": 10000,
    "total_input_tokens": 500000,
    "total_output_tokens": 1000000,
    "avg_latency_ms": 234.5,
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

    -- Pricing
    token_input_rate DECIMAL(10, 6) NOT NULL,
    token_output_rate DECIMAL(10, 6) NOT NULL,
    request_base_rate DECIMAL(10, 6) NOT NULL,
    memory_hourly_rate DECIMAL(10, 6) NOT NULL,
    storage_hourly_rate DECIMAL(10, 6) NOT NULL,

    -- Capacity
    max_concurrent_requests INT NOT NULL,
    memory_capacity_gb INT NOT NULL,
    storage_capacity_tb INT NOT NULL,

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

    -- Billing
    billing_email VARCHAR(255),
    payment_method_id VARCHAR(128),

    -- Limits
    rate_limit_per_minute INT DEFAULT 60,
    monthly_budget_usd DECIMAL(10, 2),

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),

    INDEX idx_api_key (api_key),
    INDEX idx_status (status)
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
    endpoint VARCHAR(128) NOT NULL,  -- /v1/chat/completions, etc.

    -- Usage
    input_tokens INT NOT NULL,
    output_tokens INT NOT NULL,
    latency_ms INT NOT NULL,

    -- Cost
    cost_usd DECIMAL(10, 6) NOT NULL,

    -- Status
    status VARCHAR(32) NOT NULL,  -- success, error, timeout
    error_message TEXT,

    -- Timestamp
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (provider_id) REFERENCES providers(provider_id),

    INDEX idx_user_created (user_id, created_at),
    INDEX idx_provider_created (provider_id, created_at),
    INDEX idx_created_at (created_at)
);
```

### 5.4 Metrics Table (Aggregated)
```sql
CREATE TABLE provider_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    provider_id VARCHAR(64) NOT NULL,

    -- Time bucket (hourly aggregation)
    bucket_time TIMESTAMP NOT NULL,

    -- LLM Metrics
    request_count INT NOT NULL,
    total_input_tokens BIGINT NOT NULL,
    total_output_tokens BIGINT NOT NULL,
    avg_latency_ms FLOAT NOT NULL,
    p95_latency_ms FLOAT NOT NULL,
    p99_latency_ms FLOAT NOT NULL,
    error_count INT NOT NULL,

    -- Memory Metrics
    avg_memory_allocated_gb FLOAT NOT NULL,
    avg_memory_utilization_pct FLOAT NOT NULL,

    -- Storage Metrics
    avg_storage_used_tb FLOAT NOT NULL,
    avg_storage_utilization_pct FLOAT NOT NULL,

    -- System Health
    avg_cpu_usage_pct FLOAT NOT NULL,
    avg_gpu_usage_pct FLOAT NOT NULL,
    uptime_minutes INT NOT NULL,

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),

    FOREIGN KEY (provider_id) REFERENCES providers(provider_id),

    UNIQUE (provider_id, bucket_time),
    INDEX idx_provider_bucket (provider_id, bucket_time)
);
```

### 5.5 Revenue Table
```sql
CREATE TABLE provider_revenue (
    revenue_id BIGSERIAL PRIMARY KEY,
    provider_id VARCHAR(64) NOT NULL,

    -- Period
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,

    -- Revenue Breakdown
    llm_revenue DECIMAL(12, 2) NOT NULL,
    memory_revenue DECIMAL(12, 2) NOT NULL,
    storage_revenue DECIMAL(12, 2) NOT NULL,
    total_revenue DECIMAL(12, 2) NOT NULL,

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
    memory_charge DECIMAL(12, 2) NOT NULL,
    storage_charge DECIMAL(12, 2) NOT NULL,
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

    # Rate limiting check
    if not await rate_limiter.check(user.user_id, user.rate_limit_per_minute):
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

### 8.3 비동기 처리
- **Metric 수집**: Kafka/RabbitMQ로 비동기 처리
- **정산 계산**: Celery/RQ로 배치 처리 (일별/월별)
- **알림 전송**: 비동기 worker로 처리

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
      - ./storage:/storage

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

## 10. 가격 정책 예시

### 10.1 Provider 단가 (예시)
```
LLM API:
  - Input token: $0.0001/token
  - Output token: $0.0002/token
  - Request base: $0.01/request

Memory:
  - $0.02/GB/hour

Storage:
  - $0.001/GB/hour
```

### 10.2 사용자 단가 (예시, Provider 단가 + 50% 마진)
```
LLM API:
  - Input token: $0.00015/token
  - Output token: $0.0003/token
  - Request base: $0.015/request

Memory:
  - $0.03/GB/hour

Storage:
  - $0.0015/GB/hour
```

### 10.3 플랫폼 수수료
- Provider 수익의 20% (또는 고정 금액)

---

## 11. 구현 우선순위

### Phase 1: MVP (4-6주)
1. ✅ Router 기본 구조 (FastAPI)
2. ✅ Provider 등록 및 인증
3. ✅ 단순 라운드로빈 라우팅
4. ✅ 기본 LLM API proxy
5. ✅ Metric 수집 (기본)
6. ✅ PostgreSQL + Redis 연동

### Phase 2: Core Features (6-8주)
1. ✅ 고급 라우팅 전략 (least latency, cost optimized)
2. ✅ Memory & Storage 서비스
3. ✅ 상세 metric 수집 및 집계
4. ✅ Provider 수익 정산 로직
5. ✅ 사용자 결제 로직
6. ✅ Health check & monitoring

### Phase 3: Production Ready (8-10주)
1. ✅ 결제 시스템 연동 (Stripe/PayPal)
2. ✅ Dashboard (Provider/User)
3. ✅ 알림 시스템
4. ✅ 로깅 및 감사
5. ✅ 보안 강화 (rate limiting, DDoS 방어)
6. ✅ 부하 테스트 및 최적화

### Phase 4: Advanced Features (추가)
1. ✅ Provider SLA 모니터링
2. ✅ 자동 스케일링
3. ✅ Multi-region 지원
4. ✅ ML 기반 라우팅 최적화
5. ✅ Marketplace (Provider 랭킹, 리뷰)

---

## 12. 참고 자료

### 12.1 유사 플랫폼
- **OpenRouter**: LLM API aggregator
- **Helicone**: LLM observability platform
- **Portkey**: LLM gateway

### 12.2 기술 스택 추천
- **Backend**: FastAPI (Python) or Go
- **Database**: PostgreSQL (transactional), TimescaleDB (metrics)
- **Cache**: Redis
- **Message Queue**: Kafka or RabbitMQ
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK stack (Elasticsearch, Logstash, Kibana)
- **Deployment**: Kubernetes or Docker Swarm
