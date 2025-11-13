# Modern Attention Variants

## 🎯 목표

**프로덕션 환경에서 사용되는 효율적인 Attention 메커니즘 이해하기**

2023-2024년 현재, 대부분의 최신 LLM은 표준 Multi-Head Attention (MHA)을 사용하지 않습니다!
- **PaLM, Falcon, StarCoder** → Multi-Query Attention (MQA)
- **Llama 2, Mistral, Gemma** → Grouped-Query Attention (GQA)

왜? **KV Cache 메모리를 50-90% 절감**하면서 성능은 유지하기 때문입니다.

---

## 📊 문제: Multi-Head Attention의 메모리 병목

### Standard Multi-Head Attention (MHA)

```python
# Standard MHA
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 각 head마다 독립적인 Q, K, V projection
        self.W_q = nn.Linear(d_model, d_model)  # (512, 512)
        self.W_k = nn.Linear(d_model, d_model)  # (512, 512)
        self.W_v = nn.Linear(d_model, d_model)  # (512, 512)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]

        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        Q = self.split_heads(self.W_q(x))  # (batch, 8, seq_len, 64)
        K = self.split_heads(self.W_k(x))  # (batch, 8, seq_len, 64)
        V = self.split_heads(self.W_v(x))  # (batch, 8, seq_len, 64)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # (batch, 8, seq_len, 64) → (batch, seq_len, 512)
        out = self.combine_heads(out)
        return self.W_o(out)
```

### KV Cache 메모리 문제

**생성 시 KV Cache 필요**:
```python
# Autoregressive generation
for step in range(max_new_tokens):
    # 이전 token들의 K, V를 cache에 저장
    # cache shape: (batch, num_heads, seq_len, d_k)
    past_k = cache['k']  # (1, 8, 1000, 64)
    past_v = cache['v']  # (1, 8, 1000, 64)

    # 새 token만 계산
    q_new = W_q(x_new)  # (1, 1, 512)
    k_new = W_k(x_new)
    v_new = W_v(x_new)

    # Cache 업데이트
    cache['k'] = torch.cat([past_k, k_new], dim=2)
    cache['v'] = torch.cat([past_v, v_new], dim=2)
```

**메모리 계산**:
```
For Llama 7B (d_model=4096, num_heads=32, num_layers=32):

Single token KV cache per layer:
  K: (num_heads, d_k) = (32, 128) = 4096 floats
  V: (num_heads, d_k) = (32, 128) = 4096 floats
  Total: 8192 floats = 32 KB (FP32) or 16 KB (FP16)

For 2048 context length, 32 layers:
  2048 × 32 KB × 32 layers = 2 GB per sequence!

Batch size 16 → 32 GB just for KV cache!
```

**문제**:
- Large batch inference 불가능
- Long context 처리 어려움
- GPU 메모리 대부분을 KV cache가 차지

---

## 🚀 Solution 1: Multi-Query Attention (MQA)

### 핵심 아이디어

**"모든 head가 같은 K, V를 공유하면 어떨까?"**

```
Standard MHA:
  num_heads개의 (Q, K, V) 쌍

MQA:
  num_heads개의 Q
  + 단 1개의 (K, V) 쌍 (모든 Q head가 공유)
```

### 구현

```python
class MultiQueryAttention(nn.Module):
    """MQA: All Q heads share single K, V"""
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64

        # Q: num_heads개
        self.W_q = nn.Linear(d_model, d_model)  # (512, 512)

        # K, V: 단 1개! (d_k 차원)
        self.W_k = nn.Linear(d_model, self.d_k)  # (512, 64) ← 작아짐!
        self.W_v = nn.Linear(d_model, self.d_k)  # (512, 64)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]

        # Q: (batch, num_heads, seq_len, d_k)
        Q = self.split_heads(self.W_q(x))  # (batch, 8, seq_len, 64)

        # K, V: (batch, 1, seq_len, d_k) - single head
        K = self.W_k(x).unsqueeze(1)  # (batch, 1, seq_len, 64)
        V = self.W_v(x).unsqueeze(1)  # (batch, 1, seq_len, 64)

        # K, V broadcast across all Q heads
        # (batch, 8, seq_len, 64) @ (batch, 1, 64, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)

        # (batch, 8, seq_len, seq_len) @ (batch, 1, seq_len, 64)
        # → (batch, 8, seq_len, 64)
        out = torch.matmul(attn, V)

        out = self.combine_heads(out)
        return self.W_o(out)

    def split_heads(self, x):
        # (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        # (batch, num_heads, seq_len, d_k) → (batch, seq_len, d_model)
        batch_size, _, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.num_heads * d_k)
```

### 메모리 절감

```python
# MHA KV Cache
# K: (batch, num_heads, seq_len, d_k) = (1, 8, 2048, 64)
# V: (batch, num_heads, seq_len, d_k) = (1, 8, 2048, 64)
mha_cache_size = 2 * 8 * 2048 * 64 = 2,097,152 floats

# MQA KV Cache
# K: (batch, 1, seq_len, d_k) = (1, 1, 2048, 64)
# V: (batch, 1, seq_len, d_k) = (1, 1, 2048, 64)
mqa_cache_size = 2 * 1 * 2048 * 64 = 262,144 floats

# Reduction
reduction = 1 - (mqa_cache_size / mha_cache_size)
print(f"MQA saves {reduction*100:.1f}% KV cache memory!")  # 87.5%
```

**결과**:
- **KV Cache: 87.5% 감소** (8배 작아짐)
- Llama 7B 기준: 2GB → 256MB per sequence
- **Batch size를 8배 증가 가능!**

### 성능 Trade-off

**질문**: 성능 저하는?

**답변**:
- Quality 측면: **1-2% 하락** (대부분 무시 가능)
- Speed 측면: **30-50% 빠름** (작은 KV로 더 빠른 attention)

**Why it works:**
- 대부분의 정보는 Q 변환에서 추출됨
- K, V는 주로 "어디를 볼지" 가이드 역할
- 여러 view (multiple K, V)가 critical하지 않음

### 사용 사례

**PaLM (Google, 2022)**:
- 540B 파라미터 모델에 MQA 사용
- Inference throughput 2배 향상

**Falcon (TII, 2023)**:
- Falcon-40B, Falcon-180B
- MQA로 효율적인 추론 가능

**StarCoder (BigCode, 2023)**:
- 15B 코드 생성 모델
- MQA로 긴 코드 컨텍스트 처리

---

## ⚖️ Solution 2: Grouped-Query Attention (GQA)

### 핵심 아이디어

**"MHA vs MQA의 중간: 여러 Q head가 하나의 K, V를 공유"**

```
MHA (Multi-Head Attention):
  8 Q heads → 8 K heads, 8 V heads

MQA (Multi-Query Attention):
  8 Q heads → 1 K head, 1 V head

GQA (Grouped-Query Attention):
  8 Q heads → 2 K heads, 2 V heads (4 Q heads per KV group)

  Group 1: Q₀, Q₁, Q₂, Q₃ share K₀, V₀
  Group 2: Q₄, Q₅, Q₆, Q₇ share K₁, V₁
```

### 구현

```python
class GroupedQueryAttention(nn.Module):
    """GQA: Groups of Q heads share K, V"""
    def __init__(self, d_model=512, num_heads=8, num_kv_heads=2):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"

        self.num_heads = num_heads        # 8 Q heads
        self.num_kv_heads = num_kv_heads  # 2 KV heads
        self.num_groups = num_heads // num_kv_heads  # 4 Q heads per group

        self.d_k = d_model // num_heads  # 64

        # Q: num_heads개
        self.W_q = nn.Linear(d_model, d_model)  # (512, 512)

        # K, V: num_kv_heads개 (num_heads보다 작음)
        kv_dim = self.num_kv_heads * self.d_k  # 2 * 64 = 128
        self.W_k = nn.Linear(d_model, kv_dim)  # (512, 128)
        self.W_v = nn.Linear(d_model, kv_dim)  # (512, 128)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]

        # Q: (batch, num_heads, seq_len, d_k)
        Q = self.split_heads(self.W_q(x), self.num_heads)
        # (batch, 8, seq_len, 64)

        # K, V: (batch, num_kv_heads, seq_len, d_k)
        K = self.split_heads(self.W_k(x), self.num_kv_heads)
        V = self.split_heads(self.W_v(x), self.num_kv_heads)
        # (batch, 2, seq_len, 64)

        # Expand K, V to match Q: repeat each KV head for its group
        # (batch, 2, seq_len, 64) → (batch, 8, seq_len, 64)
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        # Standard attention with expanded K, V
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = self.combine_heads(out)
        return self.W_o(out)

    def split_heads(self, x, num_heads):
        batch_size, seq_len, dim = x.shape
        d_k = dim // num_heads
        x = x.view(batch_size, seq_len, num_heads, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.num_heads * d_k)
```

### 더 효율적인 구현 (repeat 없이)

```python
def forward_efficient(self, x):
    """GQA without explicit repeat_interleave"""
    batch_size, seq_len = x.shape[:2]

    # Q: (batch, num_heads, seq_len, d_k)
    Q = self.split_heads(self.W_q(x), self.num_heads)

    # K, V: (batch, num_kv_heads, seq_len, d_k)
    K = self.split_heads(self.W_k(x), self.num_kv_heads)
    V = self.split_heads(self.W_v(x), self.num_kv_heads)

    # Reshape Q to group structure
    # (batch, 8, seq_len, 64) → (batch, 2, 4, seq_len, 64)
    Q = Q.view(batch_size, self.num_kv_heads, self.num_groups, seq_len, self.d_k)

    # Add group dimension to K, V
    # (batch, 2, seq_len, 64) → (batch, 2, 1, seq_len, 64)
    K = K.unsqueeze(2)
    V = V.unsqueeze(2)

    # Attention within groups
    # Q: (batch, 2, 4, seq_len, 64)
    # K: (batch, 2, 1, seq_len, 64)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)

    # (batch, 2, 4, seq_len, 64) → (batch, 8, seq_len, 64)
    out = out.view(batch_size, self.num_heads, seq_len, self.d_k)

    out = self.combine_heads(out)
    return self.W_o(out)
```

### 메모리 절감

```python
# Llama 2 7B: 32 Q heads, 8 KV heads (4 groups)

# MHA KV Cache
mha_cache = 2 * 32 * seq_len * 128  # K + V, 32 heads

# GQA KV Cache
gqa_cache = 2 * 8 * seq_len * 128   # K + V, 8 heads

# Reduction
reduction = 1 - (8 / 32)
print(f"GQA saves {reduction*100:.1f}% KV cache!")  # 75%
```

**Trade-off 비교**:

| Method | KV Cache Size | Quality | Speed |
|--------|---------------|---------|-------|
| MHA | 100% | Best | Baseline |
| GQA (num_groups=4) | 25% | 99% | 1.5x faster |
| GQA (num_groups=8) | 12.5% | 98% | 2x faster |
| MQA | 12.5% | 96-97% | 2.5x faster |

**결론**: GQA는 **성능과 효율의 최적 균형**

---

## 🏆 실전 사용 사례

### Llama 2 (Meta, 2023)

```python
# Llama 2 7B configuration
{
    "d_model": 4096,
    "num_heads": 32,       # Q heads
    "num_kv_heads": 8,     # KV heads (GQA with 4 groups)
    "num_layers": 32,
}

# KV Cache savings:
# 32 layers × 2048 seq_len × 128 d_k × 2 (K+V)
# MHA: 32 heads → 512 MB per sequence
# GQA: 8 heads → 128 MB per sequence (75% reduction!)
```

### Mistral 7B (Mistral AI, 2023)

```python
# Mistral 7B configuration
{
    "d_model": 4096,
    "num_heads": 32,
    "num_kv_heads": 8,     # GQA with 4 groups
    "sliding_window": 4096,  # Local attention window
}

# GQA + Sliding Window = extremely efficient long context!
```

### Gemma (Google, 2024)

```python
# Gemma 7B configuration
{
    "d_model": 3072,
    "num_heads": 16,
    "num_kv_heads": 16,    # Note: num_kv_heads = num_heads
}

# Gemma uses MHA, NOT GQA!
# Why? 7B is small enough, prioritizes quality
```

---

## 💻 Complete Implementation with KV Cache

### Efficient GQA with Caching

```python
class GQAWithCache(nn.Module):
    """Production-ready GQA with KV caching"""
    def __init__(self, d_model=512, num_heads=8, num_kv_heads=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        kv_dim = self.num_kv_heads * self.d_k
        self.W_k = nn.Linear(d_model, kv_dim)
        self.W_v = nn.Linear(d_model, kv_dim)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None, use_cache=False):
        """
        Args:
            x: (batch, seq_len, d_model)
            past_kv: tuple of (past_k, past_v) or None
            use_cache: whether to return updated cache

        Returns:
            out: (batch, seq_len, d_model)
            new_kv: tuple of (k, v) if use_cache else None
        """
        batch_size, seq_len = x.shape[:2]

        # Q for current tokens
        Q = self.split_heads(self.W_q(x), self.num_heads)
        # (batch, num_heads, seq_len, d_k)

        # K, V for current tokens
        K_new = self.split_heads(self.W_k(x), self.num_kv_heads)
        V_new = self.split_heads(self.W_v(x), self.num_kv_heads)
        # (batch, num_kv_heads, seq_len, d_k)

        # Concatenate with past KV if available
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K_new], dim=2)
            V = torch.cat([past_v, V_new], dim=2)
        else:
            K, V = K_new, V_new

        # Expand K, V for groups
        K_expanded = K.repeat_interleave(self.num_groups, dim=1)
        V_expanded = V.repeat_interleave(self.num_groups, dim=1)
        # (batch, num_heads, total_seq_len, d_k)

        # Attention
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_expanded)

        out = self.combine_heads(out)
        out = self.W_o(out)

        if use_cache:
            return out, (K, V)
        return out, None

    def split_heads(self, x, num_heads):
        batch_size, seq_len, dim = x.shape
        d_k = dim // num_heads
        x = x.view(batch_size, seq_len, num_heads, d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.num_heads * d_k)


# Usage: Autoregressive generation
def generate_with_gqa(model, prompt_ids, max_new_tokens=50):
    """Generation with KV caching"""
    model.eval()
    generated = prompt_ids
    past_kv = None

    with torch.no_grad():
        # First forward pass: process entire prompt
        logits, past_kv = model(prompt_ids, past_kv=None, use_cache=True)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

        # Subsequent tokens: only process new token
        for _ in range(max_new_tokens - 1):
            logits, past_kv = model(next_token, past_kv=past_kv, use_cache=True)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == eos_token_id:
                break

    return generated


# Memory profiling
def profile_kv_cache():
    """Compare memory usage"""
    import torch.cuda as cuda

    # Configuration
    batch_size = 1
    seq_len = 2048
    d_model = 4096
    num_layers = 32

    # MHA (32 heads)
    mha_kv_size = batch_size * num_layers * 2 * 32 * seq_len * 128 * 2  # bytes (FP16)
    print(f"MHA KV cache: {mha_kv_size / 1e9:.2f} GB")

    # GQA (8 KV heads)
    gqa_kv_size = batch_size * num_layers * 2 * 8 * seq_len * 128 * 2
    print(f"GQA KV cache: {gqa_kv_size / 1e9:.2f} GB")

    print(f"Savings: {(1 - gqa_kv_size/mha_kv_size)*100:.1f}%")


profile_kv_cache()
# Output:
# MHA KV cache: 2.15 GB
# GQA KV cache: 0.54 GB
# Savings: 75.0%
```

---

## 📊 Performance Comparison

### Benchmark Setup

```python
import time
import torch

def benchmark_attention(AttentionClass, batch_size=8, seq_len=2048, **kwargs):
    """Benchmark attention variant"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AttentionClass(**kwargs).to(device)
    x = torch.randn(batch_size, seq_len, kwargs['d_model']).to(device)

    # Warmup
    for _ in range(10):
        _ = model(x)

    # Timing
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        out = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Memory
    memory_mb = torch.cuda.max_memory_allocated() / 1e6

    return elapsed / 100, memory_mb


# Run benchmarks
config = {'d_model': 512, 'num_heads': 8}

print("Multi-Head Attention (MHA):")
mha_time, mha_mem = benchmark_attention(MultiHeadAttention, **config)
print(f"  Time: {mha_time*1000:.2f} ms")
print(f"  Memory: {mha_mem:.2f} MB")

print("\nMulti-Query Attention (MQA):")
mqa_time, mqa_mem = benchmark_attention(MultiQueryAttention, **config)
print(f"  Time: {mqa_time*1000:.2f} ms")
print(f"  Memory: {mqa_mem:.2f} MB")
print(f"  Speedup: {mha_time/mqa_time:.2f}x")
print(f"  Memory savings: {(1-mqa_mem/mha_mem)*100:.1f}%")

print("\nGrouped-Query Attention (GQA, 2 groups):")
gqa_time, gqa_mem = benchmark_attention(
    GroupedQueryAttention, num_kv_heads=2, **config
)
print(f"  Time: {gqa_time*1000:.2f} ms")
print(f"  Memory: {gqa_mem:.2f} MB")
print(f"  Speedup: {mha_time/gqa_time:.2f}x")
print(f"  Memory savings: {(1-gqa_mem/mha_mem)*100:.1f}%")
```

**Expected Results (A100 GPU)**:
```
Multi-Head Attention (MHA):
  Time: 12.5 ms
  Memory: 450 MB

Multi-Query Attention (MQA):
  Time: 5.2 ms
  Memory: 120 MB
  Speedup: 2.4x
  Memory savings: 73.3%

Grouped-Query Attention (GQA, 2 groups):
  Time: 7.8 ms
  Memory: 180 MB
  Speedup: 1.6x
  Memory savings: 60.0%
```

---

## 🎯 When to Use Which?

### Decision Tree

```
Start
  │
  ├─ Small model (<3B params)?
  │    └─ Use MHA (quality > efficiency)
  │
  ├─ Medium model (3B-13B)?
  │    └─ Use GQA (best balance)
  │         • num_kv_heads = num_heads / 4
  │         • Example: 32 Q heads, 8 KV heads
  │
  └─ Large model (>13B)?
       ├─ Quality critical? → GQA (num_heads / 2)
       └─ Throughput critical? → MQA
```

### Recommendations by Use Case

**1. Research / Small Models (<3B)**
- **Use**: MHA
- **Reason**: Quality first, memory not bottleneck
- **Example**: BERT-base, GPT-2

**2. Production LLMs (7B-13B)**
- **Use**: GQA with num_groups = 4
- **Reason**: Best quality/efficiency trade-off
- **Example**: Llama 2, Mistral

**3. Massive Batch Inference (>13B)**
- **Use**: MQA or GQA with num_groups = 8
- **Reason**: Maximize throughput
- **Example**: PaLM, serving workloads

**4. Long Context Applications**
- **Use**: GQA + Sliding Window
- **Reason**: KV cache grows with seq_len
- **Example**: Mistral (32K context)

**5. Edge Deployment**
- **Use**: MQA
- **Reason**: Minimize memory footprint
- **Example**: Mobile LLMs

---

## 🔬 Uptraining: MHA → GQA Conversion

### Problem

**기존 MHA 모델을 GQA로 변환하려면?**

**Naive approach**: Train from scratch
- **Cost**: 수백만 달러
- **Time**: 수개월

**Better approach**: Uptraining (continued training)

### Uptraining Procedure (Llama 2 방식)

```python
def convert_mha_to_gqa(mha_checkpoint, num_groups):
    """
    Convert trained MHA to GQA initialization

    Strategy: Mean pooling of K, V heads
    """
    num_heads = mha_checkpoint['num_heads']
    num_kv_heads = num_heads // num_groups

    # MHA weights: (d_model, num_heads * d_k)
    W_k_mha = mha_checkpoint['W_k']  # (512, 512) for 8 heads
    W_v_mha = mha_checkpoint['W_v']

    # Reshape to (num_heads, d_k, d_model)
    W_k_mha = W_k_mha.T.view(num_heads, d_k, d_model)
    W_v_mha = W_v_mha.T.view(num_heads, d_k, d_model)

    # Group and mean pool
    # (8, d_k, d_model) → (2, d_k, d_model)
    W_k_gqa = W_k_mha.view(num_kv_heads, num_groups, d_k, d_model).mean(dim=1)
    W_v_gqa = W_v_mha.view(num_kv_heads, num_groups, d_k, d_model).mean(dim=1)

    # Reshape back
    W_k_gqa = W_k_gqa.view(num_kv_heads * d_k, d_model).T
    W_v_gqa = W_v_gqa.view(num_kv_heads * d_k, d_model).T

    return {
        'W_q': mha_checkpoint['W_q'],  # Keep Q unchanged
        'W_k': W_k_gqa,
        'W_v': W_v_gqa,
        'W_o': mha_checkpoint['W_o'],
    }


# Uptraining recipe
def uptrain_to_gqa(mha_model, train_loader, num_steps=5000):
    """
    Continue training MHA model as GQA

    Llama 2 approach: 5% of original pre-training compute
    """
    # Convert to GQA initialization
    gqa_model = convert_mha_to_gqa(mha_model.state_dict(), num_groups=4)

    # Small learning rate (10x smaller than pre-training)
    optimizer = torch.optim.AdamW(gqa_model.parameters(), lr=1e-5)

    # Continue training
    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break

        loss = gqa_model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    return gqa_model


# Meta's findings:
# - Uptraining takes ~5% of original pre-training compute
# - Recovers 99%+ of original quality
# - Massive savings compared to training from scratch
```

---

## 📝 Summary

### Key Takeaways

1. **MHA → MQA → GQA**: Progressive efficiency improvements

2. **Memory Savings**:
   - MQA: **87.5% KV cache reduction**
   - GQA (4 groups): **75% reduction**

3. **Quality Trade-offs**:
   - MHA: 100% quality (baseline)
   - GQA: 99% quality
   - MQA: 96-97% quality

4. **Production Standard (2024)**:
   - 7B-13B models: **GQA with 4 groups**
   - 70B+ models: **GQA or MQA**

5. **Uptraining Works**:
   - Convert MHA → GQA with 5% compute
   - Mean pooling initialization

### Architecture Comparison

```python
# Modern LLM Architectures

Llama 2 7B:
    num_heads: 32
    num_kv_heads: 8  # GQA (4:1 ratio)

Llama 2 13B:
    num_heads: 40
    num_kv_heads: 40  # MHA (kept for quality)

Llama 2 70B:
    num_heads: 64
    num_kv_heads: 8   # GQA (8:1 ratio, aggressive)

Mistral 7B:
    num_heads: 32
    num_kv_heads: 8   # GQA (4:1)

Falcon 40B:
    num_heads: 128
    num_kv_heads: 1   # MQA (extreme)
```

---

## 🎓 Exercises

### Exercise 1: Implement MQA

Implement Multi-Query Attention from scratch and verify memory savings.

```python
# Test your implementation
mqa = MultiQueryAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)
out = mqa(x)
assert out.shape == (2, 100, 512)
print("✓ MQA implementation correct!")
```

### Exercise 2: GQA Parameter Count

Calculate the parameter reduction of GQA vs MHA.

```python
def count_parameters(d_model, num_heads, num_kv_heads=None):
    """
    MHA: W_q, W_k, W_v, W_o all (d_model, d_model)
    GQA: W_q (d_model, d_model), W_k, W_v (d_model, num_kv_heads * d_k)
    """
    # Your code here
    pass

mha_params = count_parameters(4096, 32)
gqa_params = count_parameters(4096, 32, num_kv_heads=8)
print(f"Parameter reduction: {(1-gqa_params/mha_params)*100:.1f}%")
```

### Exercise 3: Convert Your Model

Take a trained Transformer and convert it to GQA using uptraining.

```python
# Load your pre-trained model
model = torch.load('my_transformer.pt')

# Convert to GQA
gqa_model = convert_mha_to_gqa(model, num_groups=4)

# Uptrain
gqa_model = uptrain_to_gqa(gqa_model, train_loader, num_steps=5000)

# Evaluate
evaluate(gqa_model, test_loader)
```

---

## 📚 References

**Papers**:
1. **Fast Transformer Decoding** (Shazeer, 2019) - Introduced MQA
2. **GQA: Training Generalized Multi-Query Transformer** (Ainslie et al., 2023)
3. **Llama 2** (Touvron et al., 2023) - GQA in practice
4. **Mistral 7B** (Jiang et al., 2023) - GQA + Sliding Window

**Code**:
- HuggingFace Transformers: `modeling_llama.py` (GQA implementation)
- Mistral reference implementation
- vLLM: Optimized GQA kernels

---

## ⏭️ Next Steps

이제 현대적인 Attention 변형을 이해했으니:

1. **Flash Attention** 학습 → Phase 5.7에서 CUDA 구현 확인
2. **Sliding Window Attention** → Mistral의 local attention
3. **Paged Attention** → vLLM의 메모리 관리

👉 Continue to [Phase 2: Modern LLM Architectures](../phase2-bert-gpt/04-modern-llm-architectures.md)

**현대 LLM의 필수 지식을 마스터했습니다!** 🚀
