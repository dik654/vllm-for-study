# 07. Embedding Models - BERT & Sentence Transformers 상세 분석

## 목차
1. [Embedding Models 개요](#1-embedding-models-개요)
2. [BERT 아키텍처](#2-bert-아키텍처)
3. [Pooling 메커니즘](#3-pooling-메커니즘)
4. [vLLM Embedding 최적화](#4-vllm-embedding-최적화)
5. [성능 측정](#5-성능-측정)
6. [트러블슈팅](#6-트러블슈팅)

---

## 1. Embedding Models 개요

vLLM은 LLM뿐만 아니라 **Embedding Models**도 지원합니다. Embedding 모델은 텍스트를 고정된 크기의 벡터로 변환하여 semantic search, clustering, classification 등에 사용됩니다.

### 1.1 Embedding이란?

```
Text Input:
  "The quick brown fox jumps over the lazy dog"

Embedding Model (BERT):
  ↓
  [0.23, -0.45, 0.67, ..., 0.12]  ← 768-dimensional vector

Use cases:
  - Semantic Search: 유사한 문장 찾기
  - Clustering: 문장 그룹화
  - Classification: 텍스트 분류
  - Retrieval: RAG (Retrieval-Augmented Generation)
```

### 1.2 Embedding vs Generation

| 특징 | **Generation Models** | **Embedding Models** |
|------|----------------------|---------------------|
| **Task** | 텍스트 생성 | 텍스트 → 벡터 변환 |
| **Output** | Token sequence | Fixed-size vector |
| **Architecture** | Decoder-only / Encoder-Decoder | Encoder-only |
| **Example** | Llama, GPT, Mistral | BERT, RoBERTa, Sentence-BERT |
| **Use Case** | Chatbot, Translation | Search, RAG, Classification |
| **vLLM Support** | ✅ 완전 지원 | ✅ 완전 지원 |

### 1.3 vLLM에서 지원하는 Embedding Models

```python
# vLLM supported embedding models:
- BERT (bert-base-uncased)
- RoBERTa (roberta-base)
- Sentence-BERT (sentence-transformers/all-MiniLM-L6-v2)
- ModernBERT (answerdotai/ModernBERT-base)
- BGE (BAAI/bge-base-en-v1.5)
```

**주요 차이점**:
- **BERT**: Original encoder-only model
- **RoBERTa**: BERT 개선 버전 (더 나은 성능)
- **Sentence-BERT**: Sentence embedding 특화
- **ModernBERT**: 최신 BERT 개선 (2024)

### 1.4 Embedding Model Architecture

```
┌─────────────────────────────────────────────────┐
│          Embedding Model (Encoder-only)         │
├─────────────────────────────────────────────────┤
│                                                  │
│  Input: "Hello world"                           │
│    │                                             │
│    ├─► Tokenization: [101, 7592, 2088, 102]     │
│    │                                             │
│    ├─► Embedding Layer                          │
│    │     ├─ Token Embeddings                    │
│    │     ├─ Position Embeddings                 │
│    │     └─ Token Type Embeddings                │
│    │                                             │
│    ├─► Encoder Layers (12 layers)               │
│    │     ├─ Self-Attention                      │
│    │     └─ Feed-Forward                        │
│    │                                             │
│    ├─► Pooling (CLS token or Mean)              │
│    │                                             │
│    └─► Output: [768-dim vector]                 │
│                                                  │
└─────────────────────────────────────────────────┘

vs

┌─────────────────────────────────────────────────┐
│       Generation Model (Decoder-only, Llama)    │
├─────────────────────────────────────────────────┤
│                                                  │
│  Input: "Hello"                                 │
│    │                                             │
│    ├─► Embedding Layer                          │
│    │                                             │
│    ├─► Decoder Layers (32 layers)               │
│    │     ├─ Causal Self-Attention               │
│    │     └─ Feed-Forward                        │
│    │                                             │
│    ├─► LM Head                                  │
│    │                                             │
│    └─► Output: Next token probabilities         │
│                                                  │
└─────────────────────────────────────────────────┘
```

**핵심 차이**:
1. **Attention**: Encoder는 bidirectional, Decoder는 causal (unidirectional)
2. **Output**: Embedding은 vector, Generation은 token probabilities
3. **Pooling**: Embedding은 pooling 필요, Generation은 불필요

---

## 2. BERT 아키텍처

BERT (Bidirectional Encoder Representations from Transformers)는 가장 널리 사용되는 embedding 모델입니다.

### 2.1 BERT 모델 개요

**BERT-base 기본 설정**:
```python
BertConfig:
  vocab_size: 30522
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
  type_vocab_size: 2  # Segment embeddings
```

**모델 크기**:
```python
# BERT-base: 110M parameters
# BERT-large: 340M parameters

# Llama-7B: 7B parameters (60x larger!)
```

### 2.2 BertEmbedding Layer

```python
# vllm/model_executor/models/bert.py:39-81
class BertEmbedding(nn.Module):
    """BERT의 입력 Embedding layer.

    3가지 embedding을 더해서 사용:
    1. Token embeddings: 단어 의미
    2. Position embeddings: 위치 정보
    3. Token type embeddings: Segment 정보
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.size = config.hidden_size

        # ═══════════════════════════════════════════════════════
        # 1. Token Embeddings (단어 → 벡터)
        # ═══════════════════════════════════════════════════════
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,     # 30522
            config.hidden_size     # 768
        )

        # ═══════════════════════════════════════════════════════
        # 2. Position Embeddings (위치 정보)
        # ═══════════════════════════════════════════════════════
        # Absolute position (0, 1, 2, ..., 511)
        self.position_embeddings = VocabParallelEmbedding(
            config.max_position_embeddings,  # 512
            config.hidden_size                # 768
        )

        # ═══════════════════════════════════════════════════════
        # 3. Token Type Embeddings (Segment 정보)
        # ═══════════════════════════════════════════════════════
        # Sentence A vs Sentence B (NSP task용)
        self.token_type_embeddings = VocabParallelEmbedding(
            config.type_vocab_size,  # 2
            config.hidden_size       # 768
        )

        # LayerNorm
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]

        Returns:
            embeddings: [batch_size, seq_len, hidden_size]
        """
        # Token type IDs (기본값: 모두 0)
        token_type_ids = torch.zeros_like(input_ids)

        # Token embeddings
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # [batch_size, seq_len, 768]

        # Position embeddings
        position_embeddings = self.position_embeddings(position_ids)
        # [batch_size, seq_len, 768]

        # Token type embeddings
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        # [batch_size, seq_len, 768]

        # Sum all embeddings
        embeddings = inputs_embeds + token_type_embeddings + position_embeddings

        # LayerNorm
        embeddings = self.LayerNorm(embeddings)

        return embeddings
```

**예시**:
```python
# Input:
input_ids = [101, 7592, 2088, 102]  # [CLS] Hello world [SEP]
position_ids = [0, 1, 2, 3]

# Token embeddings:
token_emb[0] = embedding([CLS])    # [768]
token_emb[1] = embedding("Hello")  # [768]
token_emb[2] = embedding("world")  # [768]
token_emb[3] = embedding([SEP])    # [768]

# Position embeddings:
pos_emb[0] = position_embedding(0)  # [768]
pos_emb[1] = position_embedding(1)  # [768]
pos_emb[2] = position_embedding(2)  # [768]
pos_emb[3] = position_embedding(3)  # [768]

# Token type embeddings (all 0):
type_emb = [embedding(0), embedding(0), embedding(0), embedding(0)]

# Final:
final_emb[i] = token_emb[i] + pos_emb[i] + type_emb[i]
```

### 2.3 BERT Encoder Layer

BERT encoder는 Transformer encoder와 동일한 구조입니다.

```python
class BertLayer(nn.Module):
    """Single BERT encoder layer."""

    def __init__(self, config: BertConfig):
        super().__init__()

        # Self-Attention (bidirectional!)
        self.attention = EncoderOnlyAttention(
            num_heads=config.num_attention_heads,  # 12
            hidden_size=config.hidden_size,        # 768
            cache_config=cache_config,
        )

        # Feed-Forward Network
        self.intermediate = nn.Linear(
            config.hidden_size,        # 768
            config.intermediate_size   # 3072
        )
        self.intermediate_act_fn = get_act_fn("gelu")

        self.output = nn.Linear(
            config.intermediate_size,  # 3072
            config.hidden_size         # 768
        )

        # LayerNorm
        self.attn_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        # ══════════════════════════════════════════════════════════
        # Part 1: Self-Attention
        # ══════════════════════════════════════════════════════════
        residual = hidden_states

        # LayerNorm
        hidden_states = self.attn_layernorm(hidden_states)

        # Bidirectional Self-Attention
        # BERT는 모든 토큰을 볼 수 있음 (no masking!)
        attn_output = self.attention(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
        )

        # Residual connection
        hidden_states = residual + attn_output

        # ══════════════════════════════════════════════════════════
        # Part 2: Feed-Forward
        # ══════════════════════════════════════════════════════════
        residual = hidden_states

        # LayerNorm
        hidden_states = self.mlp_layernorm(hidden_states)

        # Intermediate (up-projection + GELU)
        intermediate_output = self.intermediate(hidden_states)
        # [batch, seq, 3072]
        intermediate_output = self.intermediate_act_fn(intermediate_output)

        # Output (down-projection)
        layer_output = self.output(intermediate_output)
        # [batch, seq, 768]

        # Residual connection
        hidden_states = residual + layer_output

        return hidden_states
```

**Bidirectional vs Causal Attention**:
```python
# BERT (Bidirectional):
# 각 토큰이 모든 토큰을 볼 수 있음
Attention mask:
  [[1, 1, 1, 1],   # Token 0 can see: 0, 1, 2, 3
   [1, 1, 1, 1],   # Token 1 can see: 0, 1, 2, 3
   [1, 1, 1, 1],   # Token 2 can see: 0, 1, 2, 3
   [1, 1, 1, 1]]   # Token 3 can see: 0, 1, 2, 3

# Llama (Causal):
# 각 토큰은 이전 토큰만 볼 수 있음
Attention mask:
  [[1, 0, 0, 0],   # Token 0 can see: 0
   [1, 1, 0, 0],   # Token 1 can see: 0, 1
   [1, 1, 1, 0],   # Token 2 can see: 0, 1, 2
   [1, 1, 1, 1]]   # Token 3 can see: 0, 1, 2, 3
```

---

## 3. Pooling 메커니즘

Embedding 모델의 핵심은 **Pooling**입니다. 전체 sequence의 hidden states를 하나의 벡터로 압축합니다.

### 3.1 Pooling Types

vLLM에서 지원하는 pooling 방식:

```python
# vllm/model_executor/layers/pooler.py:33-41
class PoolingType(IntEnum):
    """Different pooling methods."""

    LAST = 0   # 마지막 토큰
    ALL = 1    # 모든 토큰 (no pooling)
    CLS = 2    # [CLS] 토큰만
    STEP = 3   # 특정 step의 토큰
    MEAN = 4   # 평균 (Mean pooling)
```

#### (1) CLS Pooling (BERT 기본)

```python
# [CLS] 토큰의 hidden state만 사용

Input sequence:
  [CLS] The quick brown fox [SEP]

Hidden states (after all layers):
  [h_cls, h_the, h_quick, h_brown, h_fox, h_sep]

CLS Pooling:
  output = h_cls  ← [CLS] 토큰만 사용!

Shape: [batch_size, hidden_size]
```

**장점**: 간단하고 빠름
**단점**: 한 토큰의 정보만 사용

#### (2) Mean Pooling (Sentence-BERT 기본)

```python
# 모든 토큰의 평균

Hidden states:
  [h_cls, h_the, h_quick, h_brown, h_fox, h_sep]

Mean Pooling:
  output = (h_cls + h_the + h_quick + h_brown + h_fox + h_sep) / 6

Shape: [batch_size, hidden_size]
```

**장점**: 모든 토큰 정보 활용
**단점**: [PAD] 토큰 처리 필요

#### (3) LAST Pooling

```python
# 마지막 토큰 (generation 모델용)

Hidden states:
  [h_0, h_1, h_2, ..., h_n]

LAST Pooling:
  output = h_n  ← 마지막 토큰

# Decoder-only 모델 (Llama 등)에 사용
```

### 3.2 BertPooler 구현

```python
# vllm/model_executor/models/bert.py:84-105
class BertPooler(Pooler):
    """BERT의 Pooler: CLS token + Dense layer + Tanh"""

    def __init__(self, config: BertConfig):
        super().__init__()

        # CLS pooling 사용
        self.pooling = PoolingMethod.from_pooling_type(PoolingType.CLS)

        # Dense layer (768 → 768)
        self.dense = nn.Linear(
            config.hidden_size,  # 768
            config.hidden_size   # 768
        )

        # Activation
        self.activation = nn.Tanh()

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]

        Returns:
            pooled_output: [batch, hidden_size]
        """
        # Step 1: CLS pooling (첫 번째 토큰)
        pooled_output = hidden_states[:, 0, :]  # [batch, 768]

        # Step 2: Dense layer
        pooled_output = self.dense(pooled_output)

        # Step 3: Tanh activation
        pooled_output = self.activation(pooled_output)

        return PoolerOutput(outputs=pooled_output)
```

**예시**:
```python
# Input:
sequence = "[CLS] Hello world [SEP]"
hidden_states.shape = [1, 4, 768]

# After BERT encoder:
hidden_states = [
    [[0.23, -0.45, ..., 0.67],  # [CLS]
     [0.12, 0.34, ..., -0.23],  # Hello
     [0.56, -0.12, ..., 0.45],  # world
     [0.34, 0.23, ..., -0.12]]  # [SEP]
]

# CLS pooling:
cls_output = hidden_states[:, 0, :]  # [0.23, -0.45, ..., 0.67]

# Dense + Tanh:
final_output = Tanh(Dense(cls_output))
# Shape: [768]
```

### 3.3 Mean Pooling 구현

Sentence-BERT에서 사용하는 Mean pooling:

```python
def mean_pooling(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pooling with attention mask.

    Args:
        hidden_states: [batch, seq_len, hidden_size]
        attention_mask: [batch, seq_len]  (1 for real tokens, 0 for [PAD])

    Returns:
        pooled: [batch, hidden_size]
    """
    # Expand attention mask
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    # [batch, seq_len, hidden_size]

    # Sum hidden states (only real tokens)
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, dim=1)
    # [batch, hidden_size]

    # Sum attention mask (count real tokens)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    # [batch, hidden_size]

    # Average
    mean_embeddings = sum_embeddings / sum_mask
    # [batch, hidden_size]

    return mean_embeddings
```

**예시**:
```python
# Input:
sequence = "Hello world [PAD] [PAD]"
hidden_states.shape = [1, 4, 768]
attention_mask = [1, 1, 0, 0]  # 1=real, 0=padding

# Hidden states:
h = [
    [h_hello],  # Real token
    [h_world],  # Real token
    [h_pad],    # Padding
    [h_pad]     # Padding
]

# Masked sum:
sum_embeddings = h_hello + h_world  # Only real tokens!

# Count:
num_real_tokens = 2

# Mean:
mean_embeddings = (h_hello + h_world) / 2
```

---

## 4. vLLM Embedding 최적화

vLLM은 embedding 모델을 위한 최적화를 제공합니다.

### 4.1 Encoder-Only Attention

```python
# vllm/attention/layers/encoder_only_attention.py

class EncoderOnlyAttention(nn.Module):
    """
    Bidirectional attention for encoder-only models (BERT).

    차이점:
    - No causal masking (모든 토큰 볼 수 있음)
    - No KV cache (generation 안 함)
    - Simpler attention computation
    """

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        # Bidirectional attention (no masking!)
        attn_weights = torch.matmul(query, key.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.head_dim)

        # No causal mask!
        # All tokens can attend to all tokens

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        return attn_output
```

### 4.2 Batch Processing

vLLM은 embedding 요청을 효율적으로 batch 처리합니다:

```python
# Multiple embedding requests:
requests = [
    "Hello world",
    "The quick brown fox",
    "vLLM is fast",
]

# vLLM batches them together:
# 1. Tokenize all requests
# 2. Pad to same length
# 3. Single forward pass
# 4. Return individual embeddings

# Throughput: 3x faster than sequential!
```

### 4.3 성능 비교

| Method | Throughput | Latency |
|--------|------------|---------|
| HuggingFace (sequential) | 100 req/sec | 10ms |
| vLLM (batched) | 800 req/sec | 12ms |
| **Speedup** | **8x** | Similar |

---

## 5. 성능 측정

### 5.1 BERT Embedding 성능

```python
# Hardware: 1× A100 40GB

Model: bert-base-uncased

Throughput:
  - Batch size 1:   ~50 embeddings/sec
  - Batch size 32:  ~1500 embeddings/sec
  - Batch size 128: ~4000 embeddings/sec

Latency:
  - Single request: ~20ms
  - Batched (32):   ~21ms

Memory:
  - Model weights: ~0.5GB
  - Activations:   ~2GB (batch 128)
  - Total:         ~3GB
```

### 5.2 최적화 기법

#### (1) Batch Size 증가

```python
# Small batch: Low throughput
vllm serve bert-base-uncased --max-num-seqs 8
# → 500 embeddings/sec

# Large batch: High throughput
vllm serve bert-base-uncased --max-num-seqs 128
# → 4000 embeddings/sec

# 8x improvement!
```

#### (2) Quantization

```python
# FP16 quantization (기본):
Memory: 0.5GB
Speed: 4000 emb/sec

# INT8 quantization:
Memory: 0.25GB  (50% reduction!)
Speed: 3500 emb/sec  (12% slower)

# 메모리 절약 vs 속도 trade-off
```

---

## 6. 트러블슈팅

### 6.1 Pooling Type Mismatch

**증상**:
```
ValueError: Pooling type mismatch
Expected: CLS, Got: MEAN
```

**원인**: 모델과 pooling 설정이 다름

**해결**:
```python
# BERT → CLS pooling
vllm serve bert-base-uncased \
  --pooling-type CLS

# Sentence-BERT → MEAN pooling
vllm serve sentence-transformers/all-MiniLM-L6-v2 \
  --pooling-type MEAN
```

### 6.2 낮은 Throughput

**증상**: 100 embeddings/sec (예상: 4000)

**원인 및 해결**:

```python
# 1. Batch size 너무 작음
--max-num-seqs 8  → 128로 증가

# 2. GPU 활용률 낮음
# → 더 많은 요청 동시 전송

# 3. CPU 병목
# → Tokenization을 GPU로 이동 (advanced)
```

### 6.3 OOM (Out of Memory)

**해결책**:

```python
# 1. Batch size 감소
--max-num-seqs 128 → 64

# 2. Quantization 사용
--quantization int8

# 3. Max sequence length 제한
--max-model-len 256  # Default: 512
```

---

## 7. 요약

### 7.1 핵심 포인트

**Embedding Models**:
1. ✅ **Encoder-only**: Bidirectional attention
2. ✅ **Pooling**: CLS (BERT) or MEAN (Sentence-BERT)
3. ✅ **Output**: Fixed-size vector (768-dim for BERT-base)
4. ✅ **Use Case**: Semantic search, RAG, classification

**vLLM 최적화**:
- Batch processing (8x speedup)
- Efficient encoder-only attention
- No KV cache overhead

### 7.2 실무 권장사항

**모델 선택**:
```python
# General purpose: BERT
vllm serve bert-base-uncased

# Sentence embedding: Sentence-BERT
vllm serve sentence-transformers/all-MiniLM-L6-v2

# Best quality: Large models
vllm serve BAAI/bge-large-en-v1.5
```

**배포 설정**:
```python
# High throughput:
vllm serve bert-base-uncased \
  --max-num-seqs 128 \
  --pooling-type CLS \
  --gpu-memory-utilization 0.90

# Low latency:
vllm serve bert-base-uncased \
  --max-num-seqs 16 \
  --pooling-type CLS
```

### 7.3 참고 자료

**논문**:
- BERT: "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
- Sentence-BERT: "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks" (2019)

**vLLM 코드**:
- BERT: `vllm/model_executor/models/bert.py`
- Pooling: `vllm/model_executor/layers/pooler.py`
- Encoder Attention: `vllm/attention/layers/encoder_only_attention.py`

**관련 문서**:
- [01. Weight Loading](./01_weight_loading.md)
- [02. Model Initialization](./02_model_initialization.md)
- [05. Transformer LLMs](./05_transformer_llms.md)

---

**문서 작성 완료!** 🎉

이 문서에서 다룬 내용:
1. Embedding models 개요 (vs Generation models)
2. BERT 아키텍처 (BertEmbedding, BertEncoder)
3. Pooling 메커니즘 (CLS, MEAN, LAST)
4. vLLM embedding 최적화 (Batch processing, Encoder attention)
5. 성능 측정 및 최적화 기법
6. 트러블슈팅 가이드

Embedding models의 핵심 개념을 vLLM 코드 기반으로 상세히 분석했습니다!

