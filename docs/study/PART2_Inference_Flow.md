# PART 2: Inference Flow (LLM 추론의 동작 메커니즘)

> LLM이 실제로 어떻게 토큰을 생성하는지 완벽히 이해하기

---

## 목차
1. [Prefill vs Decode 단계](#1-prefill-vs-decode-단계)
2. [KV Cache의 정확한 구조](#2-kv-cache의-정확한-구조)
3. [KV Cache 재사용과 GPU 비용](#3-kv-cache-재사용과-gpu-비용)
4. [KV Cache의 GPU 메모리 점유](#4-kv-cache의-gpu-메모리-점유)
5. [Attention 패턴 변화](#5-attention-패턴-변화)
6. [RoPE와 Decoding에서의 주파수 문제](#6-rope와-decoding에서의-주파수-문제)
7. [Sliding Window Attention](#7-sliding-window-attention)
8. [면접 예상 질문 및 답변](#8-면접-예상-질문-및-답변)

---

## 1. Prefill vs Decode 단계

### 1.1 LLM 추론의 두 단계

LLM 추론은 근본적으로 다른 두 단계로 나뉩니다:

```
┌─────────────────────────────────────────────────────────────┐
│  사용자 입력: "What is the capital of France?"              │
│                        ↓                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ PREFILL (Prompt Processing)                          │   │
│  │ - 전체 입력을 한 번에 처리                            │   │
│  │ - 모든 토큰의 KV 캐시 생성                            │   │
│  │ - Compute-bound (GPU 연산 위주)                       │   │
│  │ - 병렬 처리 가능                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                        ↓                                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ DECODE (Token Generation)                            │   │
│  │ - 한 번에 하나의 토큰 생성                            │   │
│  │ - 기존 KV 캐시 재사용 + 새 토큰 KV 추가              │   │
│  │ - Memory-bound (메모리 대역폭 위주)                   │   │
│  │ - 순차적 (자기회귀)                                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                        ↓                                    │
│  출력: "The capital of France is Paris."                   │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Prefill 단계 상세

**목적:** 입력 프롬프트의 모든 토큰에 대해 KV 캐시를 한 번에 계산

**연산 흐름:**
```python
def prefill(input_tokens):
    """
    input_tokens: [batch_size, prompt_length]
    """
    # 모든 토큰의 임베딩 계산
    x = embed(input_tokens)  # [B, T_prompt, d]

    for layer in transformer_layers:
        # Self-Attention: 모든 토큰이 서로 참조
        q = layer.W_q(x)  # [B, T_prompt, d]
        k = layer.W_k(x)  # [B, T_prompt, d]
        v = layer.W_v(x)  # [B, T_prompt, d]

        # KV 캐시 저장
        kv_cache[layer] = (k, v)

        # Attention 계산 (Causal Mask 적용)
        attn_out = attention(q, k, v, causal=True)
        x = layer.forward(attn_out)

    # 마지막 토큰만 logits 계산
    logits = lm_head(x[:, -1, :])  # [B, vocab_size]
    return logits, kv_cache
```

**특성:**
- **Compute-bound:** 행렬 곱이 대부분
- **높은 GPU 활용률:** 큰 행렬 연산으로 병렬화 효율적
- **FLOPS:**
  ```
  Attention: O(T² × d)
  FFN: O(T × d × d_ff)
  총: O(T² × d + T × d²)
  ```

### 1.3 Decode 단계 상세

**목적:** 새 토큰 하나를 생성하고 KV 캐시에 추가

**연산 흐름:**
```python
def decode_step(new_token, kv_cache):
    """
    new_token: [batch_size, 1]
    kv_cache: 이전까지 생성된 모든 KV
    """
    x = embed(new_token)  # [B, 1, d]

    for layer in transformer_layers:
        # 새 토큰의 Q, K, V 계산
        q = layer.W_q(x)     # [B, 1, d]
        k_new = layer.W_k(x) # [B, 1, d]
        v_new = layer.W_v(x) # [B, 1, d]

        # KV 캐시 업데이트
        k_cached, v_cached = kv_cache[layer]
        k = concat(k_cached, k_new, dim=1)  # [B, T+1, d]
        v = concat(v_cached, v_new, dim=1)  # [B, T+1, d]
        kv_cache[layer] = (k, v)

        # Attention: 새 토큰이 모든 이전 토큰 참조
        # Q: [B, 1, d], K/V: [B, T+1, d]
        attn_out = attention(q, k, v)  # [B, 1, d]
        x = layer.forward(attn_out)

    logits = lm_head(x[:, -1, :])
    return logits, kv_cache
```

**특성:**
- **Memory-bound:** KV 캐시 읽기가 병목
- **낮은 GPU 활용률:** 작은 연산량 대비 큰 데이터 이동
- **시간 분석:**
  ```
  연산: O(1 × T × d) ≈ O(Td)  - 매우 작음
  메모리: O(T × d) 읽기       - 병목!
  ```

### 1.4 Prefill vs Decode 비교

| 특성 | Prefill | Decode |
|------|---------|--------|
| 입력 크기 | T 토큰 | 1 토큰 |
| 병렬화 | 높음 (모든 토큰 동시) | 낮음 (순차 생성) |
| 계산 패턴 | Compute-bound | Memory-bound |
| GPU 활용률 | 50-80% | 5-20% |
| 주요 비용 | FLOPS | 메모리 대역폭 |
| 최적화 방향 | Tensor Core 활용 | 메모리 접근 최소화 |

### 1.5 실제 레이턴시 분석

**예시 (LLaMA-7B, A100):**
```
프롬프트 100 토큰 Prefill: ~50ms
각 Decode 단계: ~10ms

100 토큰 생성:
총 시간 = 50ms + 100 × 10ms = 1050ms
→ Decode가 95%의 시간 점유!
```

**Throughput 관점:**
```
Prefill: 100 tokens / 50ms = 2000 tokens/sec
Decode: 1 token / 10ms = 100 tokens/sec
→ 20배 차이!
```

---

## 2. KV Cache의 정확한 구조

### 2.1 KV Cache가 저장하는 것

**정의:** 이전에 계산된 Key와 Value 벡터를 저장하여 재계산 방지

**구조:**
```
KV Cache[layer_idx][batch_idx] = {
    "key":   [seq_len, num_kv_heads, head_dim],
    "value": [seq_len, num_kv_heads, head_dim]
}
```

**전체 구조:**
```
KV Cache Shape: [num_layers, 2, batch_size, num_kv_heads, seq_len, head_dim]
                     │      │      │           │            │        │
                     │      │      │           │            │        └─ 64 (d/h)
                     │      │      │           │            └─ 가변 (생성된 토큰 수)
                     │      │      │           └─ 8 (GQA) 또는 32 (MHA)
                     │      │      └─ 배치 크기
                     │      └─ Key와 Value
                     └─ 레이어 수 (32)
```

### 2.2 vLLM의 KV Cache 명세

```python
# vllm/v1/kv_cache_interface.py 참조

@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int   # KV head 수 (GQA 시 줄어듦)
    head_size: int      # 각 head의 차원
    dtype: torch.dtype  # 데이터 타입

    @property
    def page_size_bytes(self) -> int:
        return (
            2                           # Key + Value
            * self.block_size           # 블록 내 토큰 수
            * self.num_kv_heads         # KV head 수
            * self.head_size            # head 차원
            * get_dtype_size(self.dtype) # 바이트/원소
        )
```

### 2.3 왜 Q는 캐싱하지 않는가?

**Self-Attention에서:**
```
Attention = softmax(QK^T / √d) V
```

- **K와 V:** 이전 토큰들의 정보 → 새 토큰 생성 시 재사용
- **Q:** 현재 토큰의 쿼리 → 각 스텝마다 새로 계산

**Decode 단계에서:**
```
새 토큰 q_new (1개)가 모든 이전 K, V 참조:

scores = q_new × [k_1, k_2, ..., k_T]^T  → [1, T]
output = scores × [v_1, v_2, ..., v_T]   → [1, d]
```

### 2.4 메모리 레이아웃

**연속 메모리 (Naive):**
```
Request 1: [████████████████]  seq_len=16
Request 2: [████████]          seq_len=8
Request 3: [████████████████████████]  seq_len=24
           ↑ 메모리 시작

문제: 가변 길이로 인한 파편화
```

**vLLM의 Paged Memory:**
```
Block 0: [████] Block 1: [████] Block 2: [████] ...
         ↑Request 1     ↑Request 2      ↑Request 1

블록 테이블:
Request 1 → [0, 2, 5, ...]
Request 2 → [1, 3, ...]
```

---

## 3. KV Cache 재사용과 GPU 비용

### 3.1 KV Cache가 없다면?

**Decode 단계마다 전체 재계산:**
```python
# KV Cache 없이
def decode_without_cache(all_tokens):
    """매 단계마다 처음부터 재계산"""
    for layer in transformer_layers:
        q = layer.W_q(all_tokens)  # [B, T, d]
        k = layer.W_k(all_tokens)  # [B, T, d]
        v = layer.W_v(all_tokens)  # [B, T, d]
        # ... attention 계산
```

**N 토큰 생성 시 총 연산량:**
```
KV Cache 없음:
1단계: O(1²)
2단계: O(2²)
...
N단계: O(N²)
총: O(1² + 2² + ... + N²) = O(N³)

KV Cache 있음:
1단계: O(1)
2단계: O(2)  (캐시된 KV 읽기)
...
N단계: O(N)
총: O(N²)  ← N배 절약!
```

### 3.2 구체적 비용 분석

**예시: 1000 토큰 생성**
```
KV Cache 없음:
총 연산 ≈ 1000³ / 3 = 333M 단위 연산
시간 ≈ 333초 (가정: 1M 연산/초)

KV Cache 있음:
총 연산 ≈ 1000² / 2 = 500K 단위 연산
시간 ≈ 0.5초

속도 향상: 666배!
```

### 3.3 Trade-off: 연산 vs 메모리

**KV Cache의 비용:**
```
메모리 = 2 × num_layers × num_heads × seq_len × head_dim × dtype_bytes

LLaMA-7B, seq_len=2048:
= 2 × 32 × 32 × 2048 × 128 × 2 (fp16)
= 1.07 GB per request!
```

**Batch에서의 스케일링:**
```
Batch=1:  1.07 GB
Batch=8:  8.56 GB
Batch=16: 17.1 GB ← A100 80GB에서도 제한됨
```

### 3.4 메모리 대역폭 병목

**Decode가 느린 이유:**
```
A100 스펙:
- 메모리 대역폭: 2TB/s
- FP16 연산: 312 TFLOPS

Decode 단계:
- KV 캐시 읽기: 2 × 32 × 32 × T × 128 × 2 = 0.5MB × T
- T=2048일 때: 1GB 읽기
- 시간: 1GB / 2TB/s = 0.5ms (이론적)

실제로는 비효율적 메모리 접근으로 2-5배 느림
```

**Arithmetic Intensity:**
```
AI = FLOPS / Bytes accessed

Decode Attention:
FLOPS ≈ 2 × T × d (Q·K + softmax·V)
Bytes ≈ 2 × T × d (KV 읽기)
AI ≈ 1  (매우 낮음!)

Prefill Attention:
FLOPS ≈ T² × d
Bytes ≈ T × d
AI ≈ T  (시퀀스가 길수록 높음)
```

---

## 4. KV Cache의 GPU 메모리 점유

### 4.1 메모리 구성 요소

```
GPU 메모리 구성:
┌────────────────────────────────────────────┐
│  모델 가중치 (Model Weights)                │  ← 고정
│  - 14GB (LLaMA-7B, fp16)                   │
├────────────────────────────────────────────┤
│  KV Cache                                  │  ← 동적, 주요 변수
│  - 요청 수 × 시퀀스 길이에 비례              │
├────────────────────────────────────────────┤
│  Activation Memory                         │  ← 작음
│  - 중간 계산 결과                           │
├────────────────────────────────────────────┤
│  CUDA Context & 기타                       │  ← 고정
│  - ~1GB                                    │
└────────────────────────────────────────────┘
```

### 4.2 KV Cache 크기 공식

**일반 공식:**
```
KV_Cache_Size = 2 × L × H_kv × T × D_h × dtype_size × B

where:
L = num_layers
H_kv = num_kv_heads (GQA면 작아짐)
T = sequence_length
D_h = head_dim
B = batch_size
```

**주요 모델별 KV Cache (per token per batch):**

| 모델 | Layers | KV Heads | Head Dim | Per Token |
|------|--------|----------|----------|-----------|
| LLaMA-7B | 32 | 32 | 128 | 0.5 MB |
| LLaMA-13B | 40 | 40 | 128 | 0.8 MB |
| LLaMA-70B | 80 | 8 (GQA) | 128 | 0.32 MB |
| Mistral-7B | 32 | 8 (GQA) | 128 | 0.13 MB |

### 4.3 메모리 제한이 Throughput에 미치는 영향

**시나리오 분석:**
```
A100 80GB GPU, LLaMA-7B:

가용 메모리: 80GB - 14GB(모델) - 2GB(기타) = 64GB for KV Cache

max_tokens = 64GB / 0.5MB = 128K tokens

Case 1: 긴 문맥 (4K tokens/request)
max_batch = 128K / 4K = 32 requests

Case 2: 짧은 문맥 (512 tokens/request)
max_batch = 128K / 512 = 256 requests

→ 8배 더 많은 동시 요청 처리!
```

### 4.4 메모리 최적화 기법

**1. GQA (Grouped Query Attention):**
```
MHA (32 heads): 0.5 MB/token
GQA (8 heads): 0.13 MB/token → 4배 절약
```

**2. KV Cache Quantization:**
```
FP16: 2 bytes/element
INT8: 1 byte/element → 2배 절약
FP8: 1 byte/element
INT4: 0.5 bytes/element → 4배 절약 (품질 저하 주의)
```

**3. Sliding Window:**
```
Full context (4K): 4K tokens cached
Sliding Window (1K): 1K tokens cached → 4배 절약
```

---

## 5. Attention 패턴 변화

### 5.1 Full Attention

**모든 토큰이 모든 토큰 참조:**
```
      t0  t1  t2  t3  t4
t0 [  ●   ●   ●   ●   ●  ]
t1 [  ●   ●   ●   ●   ●  ]
t2 [  ●   ●   ●   ●   ●  ]
t3 [  ●   ●   ●   ●   ●  ]
t4 [  ●   ●   ●   ●   ●  ]

● = attention 가능
```

**용도:** Encoder (BERT), Bidirectional 문맥 필요 시
**복잡도:** O(T²)

### 5.2 Causal (Autoregressive) Attention

**미래 토큰 참조 불가:**
```
      t0  t1  t2  t3  t4
t0 [  ●   ○   ○   ○   ○  ]
t1 [  ●   ●   ○   ○   ○  ]
t2 [  ●   ●   ●   ○   ○  ]
t3 [  ●   ●   ●   ●   ○  ]
t4 [  ●   ●   ●   ●   ●  ]

● = attention 가능, ○ = 마스킹
```

**용도:** GPT, LLaMA 등 모든 LLM
**복잡도:** O(T²) but 절반만 계산

### 5.3 Sliding Window Attention

**고정 크기 윈도우 내에서만 참조:**
```
Window = 3

      t0  t1  t2  t3  t4  t5
t0 [  ●   ○   ○   ○   ○   ○  ]
t1 [  ●   ●   ○   ○   ○   ○  ]
t2 [  ●   ●   ●   ○   ○   ○  ]
t3 [  ○   ●   ●   ●   ○   ○  ]
t4 [  ○   ○   ●   ●   ●   ○  ]
t5 [  ○   ○   ○   ●   ●   ●  ]
```

**용도:** Mistral, Longformer
**복잡도:** O(T × W) where W = window size

### 5.4 Grouped Query Attention (GQA)

**Query heads를 KV head 그룹에 매핑:**
```
Q heads: [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7]
                    ↓
KV heads: [KV0,    KV0,    KV1,    KV1   ]
          [0-1그룹] [2-3그룹] [4-5그룹] [6-7그룹]

여러 Q head가 같은 KV 공유
```

**메모리 절약:**
```
MHA: num_kv_heads = num_q_heads = 32
GQA: num_kv_heads = 8, num_q_heads = 32
MQA: num_kv_heads = 1

KV Cache 비율:
MHA : GQA : MQA = 32 : 8 : 1 = 4 : 1 : 0.125
```

### 5.5 Attention 패턴 비교

```
┌─────────────────────────────────────────────────────────────────┐
│  Full Attention    │  Causal Attention  │  Sliding Window      │
│  ●●●●●●●●●●       │  ●○○○○○○○○○       │  ●○○○○○○○○○        │
│  ●●●●●●●●●●       │  ●●○○○○○○○○       │  ●●○○○○○○○○        │
│  ●●●●●●●●●●       │  ●●●○○○○○○○       │  ●●●○○○○○○○        │
│  ●●●●●●●●●●       │  ●●●●○○○○○○       │  ○●●●○○○○○○        │
│  ●●●●●●●●●●       │  ●●●●●○○○○○       │  ○○●●●○○○○○        │
│                    │                    │                      │
│  O(T²)            │  O(T²/2)          │  O(T×W)             │
│  Encoder          │  Decoder/LLM       │  Long context LLM   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. RoPE와 Decoding에서의 주파수 문제

### 6.1 RoPE 복습

**핵심 수식:**
```
f(x, m) = R_m × x

R_m = [cos(m·θ_0)  -sin(m·θ_0)   0            0          ]
      [sin(m·θ_0)   cos(m·θ_0)   0            0          ]
      [0            0            cos(m·θ_1)  -sin(m·θ_1) ]
      [0            0            sin(m·θ_1)   cos(m·θ_1) ]

θ_i = 10000^{-2i/d}
```

### 6.2 위치 외삽 문제

**문제 상황:**
```
학습: position 0 ~ 2048
추론: position 0 ~ 8192 (4배 길이)

position 4096에서:
θ_0 = 10000^0 = 1
각도 = 4096 × 1 = 4096 rad ≈ 652 × 2π

→ 학습 시 본 적 없는 주파수 패턴!
```

**증상:**
- 긴 문맥에서 성능 급락
- Perplexity 폭증
- 일관성 없는 출력

### 6.3 해결책 1: Position Interpolation (PI)

**아이디어:** 위치를 스케일 다운하여 학습된 범위에 맞춤

```python
# 원래
position_id = actual_position

# Position Interpolation
scale = original_context_len / extended_context_len
position_id = actual_position * scale

# 예: 2048 → 8192 확장 시
position_id = actual_position * (2048/8192) = actual_position * 0.25
```

**장점:** Fine-tuning 없이 적용 가능
**단점:** 해상도 손실

### 6.4 해결책 2: NTK-aware Scaling

**아이디어:** base frequency를 조정

```python
# 원래
base = 10000

# NTK-aware
scale_factor = extended_len / original_len
base_scaled = base * (scale_factor ** (dim / (dim - 2)))

# 예: 4배 확장, dim=128
base_scaled = 10000 * (4 ** (128/126)) ≈ 41,000
```

**효과:** 저주파(전체 위치)와 고주파(세밀한 위치) 균형 유지

### 6.5 해결책 3: YaRN (Yet another RoPE extensioN)

**결합 접근법:**
```
1. 저주파 차원: NTK-aware scaling
2. 고주파 차원: Position Interpolation
3. 중간 차원: 두 방법의 혼합
```

**수식:**
```
θ'_i = θ_i / s_i

s_i = {
    1                     if θ_i < π/λ_max      (고주파, 변경 없음)
    s                     if θ_i > π/λ_min      (저주파, 풀 스케일링)
    interpolate(1, s)     otherwise             (혼합)
}
```

### 6.6 실제 적용 (vLLM)

```python
# vllm/model_executor/layers/rotary_embedding.py

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position: int,
        base: int = 10000,
        scaling_factor: float = 1.0,
        rope_type: str = "default"  # default, linear, yarn
    ):
        # Base frequency 계산
        inv_freq = 1.0 / (base ** (
            torch.arange(0, rotary_dim, 2) / rotary_dim
        ))

        # Scaling 적용
        if rope_type == "linear":
            inv_freq = inv_freq / scaling_factor
        elif rope_type == "yarn":
            inv_freq = self._yarn_scaling(inv_freq, scaling_factor)

        self.register_buffer("inv_freq", inv_freq)
```

---

## 7. Sliding Window Attention

### 7.1 기본 개념

**목적:** 긴 시퀀스에서 메모리와 계산 효율성 확보

**핵심:**
```
각 토큰은 최근 W개의 토큰만 참조
→ KV Cache 크기 고정: O(W) instead of O(T)
→ Attention 복잡도: O(T×W) instead of O(T²)
```

### 7.2 수학적 정의

```
SlidingWindowAttention(Q, K, V, W):
    for i in range(T):
        start = max(0, i - W + 1)
        end = i + 1
        scores_i = Q[i] @ K[start:end].T / sqrt(d)
        attn_i = softmax(scores_i)
        output_i = attn_i @ V[start:end]
    return output
```

### 7.3 Effective Context Length

**레이어 스택 효과:**
```
Layer 1: 각 토큰이 W 토큰 참조
Layer 2: 각 토큰이 2W 토큰 정보 (간접적)
...
Layer L: 각 토큰이 L×W 토큰 정보

Mistral (W=4096, L=32):
Effective context = 32 × 4096 = 131K tokens!
```

**시각화:**
```
Layer 1:  [...][A][B][C][D]  ← D는 A,B,C,D 직접 참조
                   ↓
Layer 2:  [...][.][.][E][F]  ← F는 E 통해 A,B,C,D 간접 참조
                   ↓
Layer 3:  더 넓은 범위의 정보 전파
```

### 7.4 vLLM에서의 구현

```python
# vllm/v1/kv_cache_interface.py

@dataclass(frozen=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        max_model_len = vllm_config.model_config.max_model_len
        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens

        # Sliding window + 새로 스케줄된 토큰
        num_tokens = min(
            self.sliding_window - 1 + max_num_batched_tokens,
            max_model_len
        )

        # +1: 윈도우가 블록 경계에서 시작하지 않을 수 있음
        return (cdiv(num_tokens, self.block_size) + 1) * self.page_size_bytes
```

### 7.5 Sliding Window의 한계

**1. 정보 병목:**
```
매우 긴 문서에서 초반 정보가 중반에서 손실될 수 있음
해결: 일부 레이어에서 Full Attention 사용 (Longformer)
```

**2. 청크 경계 문제:**
```
Chunked Prefill 시 윈도우 크기보다 작은 청크:
Chunk 1: [A, B, C, D]
Chunk 2: [E, F, G, H]  ← E가 A,B,C,D를 참조해야 하는데...
```

---

## 8. 면접 예상 질문 및 답변

### Q1: Prefill과 Decode의 병목이 다른 이유는?

**답변:**
Prefill은 대량의 행렬 곱셈이 주요 연산이라 **compute-bound**입니다.
GPU의 Tensor Core를 효율적으로 활용할 수 있어 연산량 대비 빠릅니다.

Decode는 1개 토큰에 대해 전체 KV Cache를 읽어야 해서 **memory-bound**입니다.
```
연산량: O(d) 작음
메모리 접근: O(T×d) 큼
```
Arithmetic Intensity가 ~1로 매우 낮아 GPU 연산 능력을 활용하지 못합니다.

### Q2: KV Cache 없이 LLM을 추론하면 어떻게 되나요?

**답변:**
N 토큰 생성 시:
- KV Cache 있음: O(N²) 연산
- KV Cache 없음: O(N³) 연산

100 토큰 생성 기준 약 **50배** 느려집니다.
각 decode 단계마다 전체 시퀀스를 처음부터 재계산해야 하기 때문입니다.

### Q3: GQA가 MHA 대비 메모리를 얼마나 절약하나요?

**답변:**
KV Cache 크기는 `num_kv_heads`에 비례합니다.

```
MHA (LLaMA-7B): 32 KV heads
GQA (LLaMA-70B): 8 KV heads
MQA: 1 KV head

메모리 비율 = 32 : 8 : 1 = 4 : 1 : 0.125
```

LLaMA-70B는 GQA로 모델 크기는 10배지만 KV Cache는 LLaMA-7B의 **64%**만 사용합니다.

### Q4: Sliding Window Attention이 긴 컨텍스트를 처리할 수 있는 원리는?

**답변:**
각 레이어에서 W 토큰만 직접 참조하지만, L개 레이어를 거치면 정보가 전파됩니다.

```
Effective context ≈ L × W
```

Mistral의 경우 W=4096, L=32로 이론적 128K 컨텍스트입니다.
다만 간접적인 정보 전달이므로 직접 참조보다 정보 손실이 있을 수 있습니다.

### Q5: RoPE의 위치 외삽 문제와 해결책은?

**답변:**
**문제:** RoPE의 주파수 패턴이 학습 범위를 벗어나면 성능 급락

**해결책들:**
1. **Position Interpolation:** 위치를 압축하여 학습 범위 내로 매핑
   - 장점: 간단
   - 단점: 해상도 손실

2. **NTK-aware Scaling:** Base frequency 조정
   - 주파수별 차별적 스케일링

3. **YaRN:** 주파수 대역별 최적 전략 혼합
   - 현재 가장 효과적인 방법

### Q6: Decode 단계의 GPU 활용률이 낮은 이유는?

**답변:**
GPU는 높은 **Arithmetic Intensity**(연산/메모리 비율)에서 효율적입니다.

Decode 단계:
```
연산: Q(1×d) × K^T(d×T) = O(T×d)
메모리: KV 읽기 O(T×d)
AI = 연산/메모리 ≈ 1
```

반면 GPU는 AI > 100 에서 최적화되어 있어, Decode의 AI=1은 메모리 대역폭에 완전히 제한됩니다.
그래서 **Continuous Batching**으로 여러 요청을 묶어 AI를 높이는 것이 핵심 최적화입니다.

---

## 참고 자료

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) (vLLM 논문)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
- [Mistral 7B Technical Report](https://arxiv.org/abs/2310.06825)
- vLLM 소스코드: `vllm/v1/kv_cache_interface.py`, `vllm/v1/core/`
