# LLM Metrics Collector for vLLM

A comprehensive metrics collection system for tracking LLM usage, resource consumption, and cost calculation in vLLM.

## 🎯 Purpose

This system enables:
- **Per-request metrics collection**: Track every LLM request in detail
- **Objective pricing**: Calculate costs based on token usage, time, and resources
- **Usage analysis**: Aggregate and analyze metrics by user, model, time period
- **Billing**: Generate accurate billing reports for LLM services

## 📊 What Metrics Are Collected?

### Token Usage
- **Prompt tokens**: Input tokens processed
- **Completion tokens**: Output tokens generated
- **Cached tokens**: Tokens served from cache
- **Total tokens**: Sum of all tokens

### Timing Metrics
- **Time to First Token (TTFT)**: Latency until first token
- **Time Per Output Token (TPOT)**: Average token generation time
- **End-to-end Latency**: Total request duration
- **Queue time**: Time waiting in queue
- **Prefill time**: Prompt processing time
- **Decode time**: Token generation time

### Resource Usage
- **GPU memory**: Peak GPU memory usage
- **KV cache**: KV cache blocks and memory used
- **Batch size**: Processing batch size
- **Preemptions**: Number of times preempted

### Cost Information
- **Estimated cost**: Total cost for the request
- **Input cost**: Cost for input tokens
- **Output cost**: Cost for output tokens
- **Cache discount**: Discount for cached tokens

## 🚀 Quick Start

### Basic Usage

```python
from vllm.llm_metrics_collector import (
    VLLMMetricsCollector,
    MemoryStorageBackend,
    TokenBasedPricing,
)

# 1. Set up storage
storage = MemoryStorageBackend()

# 2. Set up pricing
pricing = TokenBasedPricing(
    input_price_per_1k=0.0005,   # $0.50 per 1M input tokens
    output_price_per_1k=0.0015,  # $1.50 per 1M output tokens
    cache_discount_per_1k=0.00025
)

# 3. Create collector
collector = VLLMMetricsCollector(
    model_name="gpt-3.5-turbo",
    engine_id="engine-1"
)

# 4. Collect metrics from vLLM request
metrics = collector.collect(
    finished_stats=finished_request_stats,  # From vLLM
    request_id="req-123",
    user_id="user-456",
    arrival_time=time.time()
)

# 5. Enrich with costs
metrics = collector.enrich(metrics, pricing_calculator=pricing)

# 6. Save
storage.save(metrics)

# 7. Query later
user_metrics = storage.load(filters={"user_id": "user-456"})
print(f"User has {len(user_metrics)} requests")
```

### File Storage

For production, use file storage with automatic partitioning:

```python
from vllm.llm_metrics_collector.storage import FileStorageBackend

storage = FileStorageBackend(
    base_dir="/var/log/vllm/metrics",
    partition_by="day",  # Creates one file per day
    compress=False  # Optional gzip compression
)

storage.save(metrics)

# Query by time range
from datetime import datetime, timedelta

today = datetime.now()
yesterday = today - timedelta(days=1)

metrics = storage.load(filters={
    "start_time": yesterday.timestamp(),
    "end_time": today.timestamp(),
    "model_name": "gpt-3.5-turbo"
})
```

## 📦 Components

### Data Models

#### RequestMetrics
Comprehensive metrics for a single request:
```python
@dataclass
class RequestMetrics:
    request_id: str
    user_id: Optional[str]
    model_name: str

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int

    e2e_latency: float
    time_to_first_token: float

    estimated_cost: float
    # ... and many more fields
```

#### AggregatedMetrics
Aggregated metrics over a time window:
```python
@dataclass
class AggregatedMetrics:
    aggregation_key: str  # e.g., user_id, model_name
    time_window_start: datetime
    time_window_end: datetime

    total_requests: int
    total_tokens: int
    total_cost: float

    avg_e2e_latency: float
    p95_e2e_latency: float
    # ... and many more fields
```

### Collectors

#### VLLMMetricsCollector
Converts vLLM's `FinishedRequestStats` to `RequestMetrics`:

```python
collector = VLLMMetricsCollector(
    model_name="gpt-3.5-turbo",
    engine_id="engine-1"
)

metrics = collector.collect(
    finished_stats=vllm_stats,
    request_id="req-123",
    user_id="user-456"
)
```

### Storage Backends

#### MemoryStorageBackend
In-memory storage for testing:
```python
storage = MemoryStorageBackend()
storage.save(metrics)
storage.clear()  # Clear all data
```

#### FileStorageBackend
File-based storage with partitioning:
```python
storage = FileStorageBackend(
    base_dir="/var/log/vllm/metrics",
    partition_by="day",  # hour, day, week, month
    compress=False
)
```

File structure:
```
/var/log/vllm/metrics/
  2025-11-02.jsonl
  2025-11-03.jsonl
  ...
```

### Pricing Calculators

#### TokenBasedPricing
Standard token-based pricing:

```python
# GPT-4 style pricing
pricing = TokenBasedPricing(
    input_price_per_1k=0.03,    # $30/1M tokens
    output_price_per_1k=0.06,   # $60/1M tokens
    cache_discount_per_1k=0.015 # 50% discount on cached
)

cost = pricing.calculate_cost(metrics)
print(f"Request cost: ${cost:.4f}")

# Estimate before request
estimated = pricing.estimate_cost_for_tokens(
    prompt_tokens=1000,
    completion_tokens=500,
    cached_tokens=200
)
```

## 🔍 Querying Metrics

### Filtering

All storage backends support filtering:

```python
# Filter by user
user_metrics = storage.load(filters={"user_id": "user-123"})

# Filter by time range
metrics = storage.load(filters={
    "start_time": 1698969600.0,
    "end_time": 1699056000.0
})

# Filter by model
model_metrics = storage.load(filters={"model_name": "gpt-3.5-turbo"})

# Filter by cost
expensive = storage.load(filters={"min_cost": 0.01})

# Combine filters
metrics = storage.load(filters={
    "user_id": "user-123",
    "model_name": "gpt-3.5-turbo",
    "start_time": yesterday.timestamp(),
    "end_time": today.timestamp(),
})

# Pagination
page1 = storage.load(limit=100, offset=0)
page2 = storage.load(limit=100, offset=100)
```

### Counting

```python
total = storage.count()
user_count = storage.count(filters={"user_id": "user-123"})
```

## 💰 Cost Calculation

### Token-Based Pricing

```python
from vllm.llm_metrics_collector.pricing import TokenBasedPricing

pricing = TokenBasedPricing(
    input_price_per_1k=0.0005,
    output_price_per_1k=0.0015,
    cache_discount_per_1k=0.00025
)

# Calculate cost
cost = pricing.calculate_cost(metrics)

# Breakdown
input_cost = pricing.calculate_input_cost(metrics)
output_cost = pricing.calculate_output_cost(metrics)
cache_discount = pricing.calculate_cache_discount(metrics)

print(f"Input: ${input_cost:.4f}")
print(f"Output: ${output_cost:.4f}")
print(f"Cache Discount: -${cache_discount:.4f}")
print(f"Total: ${cost:.4f}")
```

### Pricing Tiers

Support for different pricing tiers:

```python
pricing = TokenBasedPricing(
    input_price_per_1k=0.0005,
    output_price_per_1k=0.0015,
    apply_tier_multiplier=True  # Enable tier pricing
)

# Tier multipliers:
# - REALTIME: 2.0x (TTFT < 1s)
# - STANDARD: 1.0x (normal)
# - BATCH: 0.5x (TTFT > 5s)
```

## 📈 Example Use Cases

### 1. User Billing Report

```python
# Get all requests for a user this month
import time
from datetime import datetime, timedelta

now = datetime.now()
month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

user_metrics = storage.load(filters={
    "user_id": "user-123",
    "start_time": month_start.timestamp(),
    "end_time": now.timestamp()
})

# Calculate total cost
total_cost = sum(m.estimated_cost for m in user_metrics)
total_tokens = sum(m.total_tokens for m in user_metrics)

print(f"User: user-123")
print(f"Requests: {len(user_metrics)}")
print(f"Total Tokens: {total_tokens:,}")
print(f"Total Cost: ${total_cost:.2f}")
```

### 2. Model Performance Analysis

```python
# Analyze a specific model
model_metrics = storage.load(filters={"model_name": "gpt-3.5-turbo"})

latencies = [m.e2e_latency for m in model_metrics]
ttfts = [m.time_to_first_token for m in model_metrics]

avg_latency = sum(latencies) / len(latencies)
avg_ttft = sum(ttfts) / len(ttfts)

print(f"Model: gpt-3.5-turbo")
print(f"Total Requests: {len(model_metrics)}")
print(f"Avg E2E Latency: {avg_latency:.3f}s")
print(f"Avg TTFT: {avg_ttft:.3f}s")
```

### 3. Cache Efficiency Analysis

```python
# Analyze cache hit rate
all_metrics = storage.load()

total_prompt_tokens = sum(m.prompt_tokens for m in all_metrics)
total_cached_tokens = sum(m.cached_tokens for m in all_metrics)

cache_hit_rate = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0

print(f"Total Prompt Tokens: {total_prompt_tokens:,}")
print(f"Cached Tokens: {total_cached_tokens:,}")
print(f"Cache Hit Rate: {cache_hit_rate*100:.2f}%")
```

## 🔧 Advanced Usage

### Custom Pricing

Implement your own pricing calculator:

```python
from vllm.llm_metrics_collector.pricing import PricingCalculator

class CustomPricing(PricingCalculator):
    def calculate_cost(self, metrics: RequestMetrics) -> float:
        # Your custom pricing logic
        base_cost = metrics.total_tokens * 0.00001

        # Add latency surcharge
        if metrics.e2e_latency > 10.0:
            base_cost *= 1.5

        return base_cost
```

### Integration with vLLM API Server

```python
# In your vLLM API server code
from vllm.llm_metrics_collector import (
    VLLMMetricsCollector,
    FileStorageBackend,
    TokenBasedPricing,
)

# Initialize once
storage = FileStorageBackend("/var/log/vllm/metrics")
pricing = TokenBasedPricing(...)
collector = VLLMMetricsCollector(model_name="...")

# On each request completion
def on_request_finished(request, response, finished_stats):
    metrics = collector.collect(
        finished_stats=finished_stats,
        request_id=request.request_id,
        user_id=request.user_id,
        arrival_time=request.arrival_time,
        temperature=request.temperature,
        # ... other parameters
    )

    metrics = collector.enrich(metrics, pricing_calculator=pricing)
    storage.save(metrics)
```

## 📁 File Structure

```
vllm/llm_metrics_collector/
├── __init__.py              # Main package exports
├── README.md                # This file
├── SPECIFICATION.md         # Detailed specification
├── TODO.md                  # Implementation plan
├── models/
│   ├── enums.py            # FinishReason, PricingTier, etc.
│   ├── request_metrics.py  # RequestMetrics dataclass
│   └── aggregated_metrics.py
├── collectors/
│   ├── base.py             # MetricsCollectorBase
│   └── vllm_collector.py   # VLLMMetricsCollector
├── storage/
│   ├── base.py             # StorageBackend interface
│   ├── memory_storage.py   # MemoryStorageBackend
│   └── file_storage.py     # FileStorageBackend
└── pricing/
    ├── base.py             # PricingCalculator
    └── token_based.py      # TokenBasedPricing
```

## 🛠️ Development Status

Current implementation (Phase 1-5):
- ✅ Data models (RequestMetrics, AggregatedMetrics)
- ✅ Enums (FinishReason, PricingTier, StorageFormat)
- ✅ Collectors (VLLMMetricsCollector)
- ✅ Storage backends (Memory, File)
- ✅ Pricing (TokenBasedPricing)

Coming soon (Phase 6-16):
- 🔜 Data exporters (CSV, Parquet)
- 🔜 Metrics aggregators
- 🔜 Report generators
- 🔜 Database storage backend
- 🔜 Advanced pricing models
- 🔜 vLLM Stat Logger plugin integration

## 📖 Documentation

- [SPECIFICATION.md](SPECIFICATION.md): Comprehensive system specification
- [TODO.md](TODO.md): Detailed implementation plan with 16 phases

## 🤝 Contributing

See [TODO.md](TODO.md) for the implementation roadmap. Contributions are welcome!

## 📄 License

Same as vLLM (Apache 2.0)
