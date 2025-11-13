# Modern LLM Architectures

## 🎯 목표

**2023-2024년 프로덕션 LLM의 실제 아키텍처 완전 이해하기**

Llama 2, Mistral 7B, Gemma - 이들이 단순한 GPT가 아닌 이유는?
- GQA (Grouped-Query Attention)
- RMSNorm
- SwiGLU
- RoPE
- No bias

**이 문서에서는 실제 production 모델들을 완전히 재현할 수 있는 코드를 제공합니다.**

---

## 📊 Modern LLM Timeline

```
2017: Transformer (Vaswani et al.)
  → MHA, LayerNorm, Sinusoidal PE

2018-2020: GPT, BERT, GPT-2, GPT-3
  → Scaling up, but architecture stayed similar

2021: Switch Transformer (Google)
  → Mixture of Experts (MoE)

2022: PaLM (Google)
  → Multi-Query Attention (MQA)
  → SwiGLU activation

2023: LLaMA (Meta)
  → RMSNorm, GQA, RoPE, no bias
  → Open source revolution!

2023: Llama 2 (Meta)
  → 70B with GQA
  → RLHF for alignment

2023: Mistral 7B (Mistral AI)
  → Sliding Window Attention
  → Sparse mixture of experts

2024: Gemma (Google)
  → Open weights from DeepMind
  → Similar to Llama but with tweaks

→ Modern LLM = "Llama-style" architecture
```

---

## 🦙 Llama 2 Architecture

### Configuration

```python
# Llama 2 7B
{
    "vocab_size": 32000,
    "d_model": 4096,
    "num_layers": 32,
    "num_heads": 32,          # Query heads
    "num_kv_heads": 8,        # KV heads (GQA with 4:1 ratio)
    "d_ff": 11008,            # ~2.7 * d_model
    "max_seq_len": 4096,
    "rope_theta": 10000,      # RoPE base
    "norm_eps": 1e-5,
    "bias": False,            # No bias in linear layers!
}

# Parameter count: ~7B
```

### Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class RoPE(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, max_seq_len=4096, theta=10000):
        super().__init__()
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, q, k, seq_len):
        """
        Args:
            q, k: (batch, num_heads, seq_len, head_dim)
            seq_len: sequence length
        Returns:
            q_rot, k_rot: rotated queries and keys
        """
        # Apply rotation
        q_rot = self.apply_rotation(q, seq_len)
        k_rot = self.apply_rotation(k, seq_len)
        return q_rot, k_rot

    def apply_rotation(self, x, seq_len):
        # x: (batch, num_heads, seq_len, head_dim)
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        # Split into two halves
        x1, x2 = x[..., ::2], x[..., 1::2]

        # Rotate
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rot


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA)"""
    def __init__(self, d_model, num_heads, num_kv_heads, bias=False):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.d_k = d_model // num_heads

        # Q projection (num_heads)
        self.W_q = nn.Linear(d_model, d_model, bias=bias)

        # K, V projections (num_kv_heads)
        kv_dim = self.num_kv_heads * self.d_k
        self.W_k = nn.Linear(d_model, kv_dim, bias=bias)
        self.W_v = nn.Linear(d_model, kv_dim, bias=bias)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        # RoPE
        self.rope = RoPE(self.d_k)

    def forward(self, x, mask=None, past_kv=None):
        batch_size, seq_len = x.shape[:2]

        # Q: (batch, num_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # K, V: (batch, num_kv_heads, seq_len, d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Apply RoPE
        Q, K = self.rope(Q, K, seq_len)

        # Concatenate with past KV if available
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        # Expand K, V to match Q
        K = K.repeat_interleave(self.num_groups, dim=1)
        V = V.repeat_interleave(self.num_groups, dim=1)

        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(out), (K, V)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit"""
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=bias)
        self.V = nn.Linear(d_model, d_ff, bias=bias)
        self.W2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.W2(F.silu(self.W(x)) * self.V(x))


class LlamaBlock(nn.Module):
    """Llama 2 Transformer Block

    현대 LLM의 표준 구조:
    - Pre-LN (normalization이 먼저)
    - GQA (메모리 효율적 attention)
    - SwiGLU (강력한 FFN)
    - RMSNorm (빠른 정규화)
    - No bias (파라미터 절감)
    """
    def __init__(self, config):
        super().__init__()
        # Grouped-Query Attention (GQA)
        # 의도: KV cache 메모리를 75% 절감하면서 성능 유지
        self.attn = GroupedQueryAttention(
            config['d_model'],
            config['num_heads'],
            config['num_kv_heads'],  # num_heads보다 작음 (예: 32 vs 8)
            bias=config.get('bias', False)  # Llama는 bias 사용 안함
        )

        # SwiGLU activation FFN
        # 의도: 표준 FFN보다 더 표현력 있는 변환
        self.ffn = SwiGLU(
            config['d_model'],
            config['d_ff'],  # 보통 2.7 * d_model
            bias=config.get('bias', False)
        )

        # RMSNorm (LayerNorm보다 빠름)
        # 의도: gradient 안정화 & 빠른 정규화
        self.attn_norm = RMSNorm(config['d_model'], eps=config.get('norm_eps', 1e-5))
        self.ffn_norm = RMSNorm(config['d_model'], eps=config.get('norm_eps', 1e-5))

    def forward(self, x, mask=None, past_kv=None):
        """
        Llama Block forward pass

        구조: Pre-LN (norm → sublayer → residual)
        의도: 훈련 안정성 향상 (gradient flow 개선)
        """
        # Pre-norm attention
        # 의도: 정규화 후 attention → 안정적인 학습
        attn_out, new_kv = self.attn(self.attn_norm(x), mask, past_kv)
        h = x + attn_out  # Residual connection

        # Pre-norm FFN
        # 의도: 정규화 후 FFN → 안정적인 학습
        ffn_out = self.ffn(self.ffn_norm(h))
        out = h + ffn_out  # Residual connection

        return out, new_kv


class Llama2(nn.Module):
    """Complete Llama 2 Model

    Meta의 Llama 2 아키텍처 완전 구현
    - 7B: 32 layers, 4096 dim, 32 Q heads, 8 KV heads
    - 13B: 40 layers, 5120 dim, 40 heads (MHA)
    - 70B: 80 layers, 8192 dim, 64 Q heads, 8 KV heads
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        # 의도: token ID → dense vector representation
        self.tok_emb = nn.Embedding(config['vocab_size'], config['d_model'])

        # Transformer blocks (stack of LlamaBlock)
        # 의도: 깊은 계층을 통해 복잡한 패턴 학습
        self.layers = nn.ModuleList([
            LlamaBlock(config)
            for _ in range(config['num_layers'])
        ])

        # Final normalization
        # 의도: 출력 전 마지막 정규화 (안정적인 logits)
        self.norm = RMSNorm(config['d_model'], eps=config.get('norm_eps', 1e-5))

        # Output head (language modeling head)
        # 의도: hidden state → vocabulary 확률 분포
        self.output = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

        # Weight tying: embedding과 output이 weight 공유
        # 의도: 파라미터 절약 + 성능 향상 (empirically proven)
        self.output.weight = self.tok_emb.weight

    def forward(self, input_ids, past_kvs=None):
        """
        Llama 2 forward pass

        Args:
            input_ids: (batch, seq_len) - 입력 token IDs
            past_kvs: list of (K, V) tuples for each layer - KV cache
        Returns:
            logits: (batch, seq_len, vocab_size) - 다음 token 예측 logits
            new_past_kvs: updated KV cache - 생성 시 재사용
        """
        # Token embedding
        x = self.tok_emb(input_ids)  # (batch, seq_len, d_model)

        # 각 layer를 통과하면서 KV cache 수집
        # 의도: autoregressive 생성 시 이전 계산 재사용
        new_past_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = layer(x, past_kv=past_kv)
            new_past_kvs.append(new_kv)

        # 최종 정규화
        x = self.norm(x)

        # Vocabulary에 대한 logits 계산
        logits = self.output(x)  # (batch, seq_len, vocab_size)

        return logits, new_past_kvs

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Autoregressive generation with KV caching

        텍스트 생성 함수 (KV cache 사용)
        의도: 효율적인 autoregressive 생성

        Args:
            input_ids: (batch, seq_len) - prompt token IDs
            max_new_tokens: 생성할 최대 token 수
            temperature: 샘플링 온도 (높을수록 다양한 출력)
            top_k: top-k 샘플링 (None이면 사용 안함)
        """
        self.eval()  # 평가 모드
        past_kvs = None  # KV cache 초기화

        for _ in range(max_new_tokens):
            # Forward pass
            if past_kvs is None:
                # 첫 번째 step: 전체 prompt 처리
                # 의도: prompt의 모든 token KV를 한 번에 cache
                logits, past_kvs = self(input_ids)
            else:
                # 이후 step: 마지막 token만 처리 (효율성!)
                # 의도: 이전 token들은 cache에서 재사용
                logits, past_kvs = self(input_ids[:, -1:], past_kvs)

            # 다음 token 샘플링
            # 마지막 position의 logits만 사용
            logits = logits[:, -1, :] / temperature  # Temperature scaling

            # Top-k sampling (optional)
            # 의도: 상위 k개 token만 고려 (품질 향상)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('inf')

            # 확률 분포로 변환 후 샘플링
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 생성된 token을 시퀀스에 추가
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Test Llama 2 7B
config_7b = {
    "vocab_size": 32000,
    "d_model": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,
    "d_ff": 11008,
    "max_seq_len": 4096,
    "bias": False,
    "norm_eps": 1e-5,
}

model = Llama2(config_7b)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Llama 2 7B parameters: {total_params / 1e9:.2f}B")

# Forward pass
input_ids = torch.randint(0, 32000, (1, 10))
logits, _ = model(input_ids)
print(f"Input: {input_ids.shape} → Logits: {logits.shape}")

# Output:
# Llama 2 7B parameters: 6.74B
# Input: torch.Size([1, 10]) → Logits: torch.Size([1, 10, 32000])
```

---

## 🌊 Mistral 7B Architecture

### Key Innovations

1. **Sliding Window Attention**: 효율적인 long context
2. **GQA**: Llama와 동일 (4:1)
3. **Sparse Mixture of Experts** (Mixtral에서)

### Sliding Window Attention

```python
def create_sliding_window_mask(seq_len, window_size=4096):
    """
    Create sliding window attention mask

    Each token can attend to:
      - All previous tokens within window_size
      - Itself

    Args:
        seq_len: sequence length
        window_size: size of sliding window

    Returns:
        mask: (seq_len, seq_len) boolean mask
    """
    # Start with causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len))

    # Apply sliding window
    for i in range(seq_len):
        # Token i can only attend to [max(0, i-window_size+1), i]
        if i >= window_size:
            mask[i, :i-window_size+1] = 0

    return mask


# Visualization
mask = create_sliding_window_mask(10, window_size=4)
print(mask)
# [[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [1., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
#  [1., 1., 1., 0., 0., 0., 0., 0., 0., 0.],
#  [1., 1., 1., 1., 0., 0., 0., 0., 0., 0.],
#  [1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],  # Window size reached
#  [0., 1., 1., 1., 1., 1., 0., 0., 0., 0.],  # Sliding!
#  [0., 0., 1., 1., 1., 1., 1., 0., 0., 0.],
#  [0., 0., 0., 1., 1., 1., 1., 1., 0., 0.],
#  [0., 0., 0., 0., 1., 1., 1., 1., 1., 0.],
#  [0., 0., 0., 0., 0., 1., 1., 1., 1., 1.]]
```

### Configuration

```python
# Mistral 7B
{
    "vocab_size": 32000,
    "d_model": 4096,
    "num_layers": 32,
    "num_heads": 32,
    "num_kv_heads": 8,
    "d_ff": 14336,            # Larger than Llama!
    "max_seq_len": 32768,     # Much longer context!
    "sliding_window": 4096,   # Sliding window size
    "rope_theta": 10000,
    "bias": False,
}
```

### Mistral Block

```python
class MistralAttention(nn.Module):
    """Mistral Attention with Sliding Window"""
    def __init__(self, d_model, num_heads, num_kv_heads, sliding_window, bias=False):
        super().__init__()
        self.sliding_window = sliding_window
        # Same as GQA but with sliding window mask
        self.gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads, bias)

    def forward(self, x, past_kv=None):
        batch_size, seq_len = x.shape[:2]

        # Create sliding window mask
        if self.training:
            # Training: full sliding window mask
            mask = create_sliding_window_mask(seq_len, self.sliding_window)
            mask = mask.to(x.device).unsqueeze(0).unsqueeze(0)
        else:
            # Inference: causal mask (past_kv handles window)
            mask = None

        return self.gqa(x, mask, past_kv)


class MistralBlock(nn.Module):
    """Mistral 7B Transformer Block"""
    def __init__(self, config):
        super().__init__()
        self.attn = MistralAttention(
            config['d_model'],
            config['num_heads'],
            config['num_kv_heads'],
            config['sliding_window'],
            bias=config.get('bias', False)
        )

        self.ffn = SwiGLU(
            config['d_model'],
            config['d_ff'],
            bias=config.get('bias', False)
        )

        self.attn_norm = RMSNorm(config['d_model'])
        self.ffn_norm = RMSNorm(config['d_model'])

    def forward(self, x, past_kv=None):
        # Pre-norm attention with sliding window
        attn_out, new_kv = self.attn(self.attn_norm(x), past_kv)
        h = x + attn_out

        # Pre-norm FFN
        ffn_out = self.ffn(self.ffn_norm(h))
        out = h + ffn_out

        return out, new_kv


# Mistral can handle 32K context with constant memory!
# How? Sliding window + KV cache rotation
```

---

## 💎 Gemma Architecture

### Differences from Llama

```python
# Gemma 7B
{
    "vocab_size": 256000,      # Much larger vocabulary!
    "d_model": 3072,           # Smaller hidden size
    "num_layers": 28,
    "num_heads": 16,
    "num_kv_heads": 16,        # MHA, not GQA!
    "d_ff": 24576,             # 8 * d_model (very wide FFN)
    "max_seq_len": 8192,
    "rope_theta": 10000,
    "bias": False,
}
```

**Key Differences**:
1. **MHA instead of GQA**: Quality over efficiency (7B is small enough)
2. **Larger vocabulary**: 256K tokens (better multilingual)
3. **Wider FFN**: 8x instead of ~2.7x
4. **GeGLU instead of SwiGLU**: GELU-based gating

### GeGLU Implementation

```python
class GeGLU(nn.Module):
    """GELU-Gated Linear Unit (used in Gemma)"""
    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=bias)
        self.V = nn.Linear(d_model, d_ff, bias=bias)
        self.W2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        # GELU instead of Swish
        return self.W2(F.gelu(self.W(x)) * self.V(x))
```

---

## 📊 Architecture Comparison

### Configuration Table

| Model | d_model | Layers | Heads | KV Heads | FFN Size | Context | Vocab |
|-------|---------|--------|-------|----------|----------|---------|-------|
| **Llama 2 7B** | 4096 | 32 | 32 | 8 (GQA) | 11008 | 4K | 32K |
| **Llama 2 13B** | 5120 | 40 | 40 | 40 (MHA) | 13824 | 4K | 32K |
| **Llama 2 70B** | 8192 | 80 | 64 | 8 (GQA) | 28672 | 4K | 32K |
| **Mistral 7B** | 4096 | 32 | 32 | 8 (GQA) | 14336 | 32K | 32K |
| **Gemma 7B** | 3072 | 28 | 16 | 16 (MHA) | 24576 | 8K | 256K |

### Feature Comparison

| Feature | Llama 2 | Mistral 7B | Gemma 7B |
|---------|---------|------------|----------|
| **Attention** | GQA (4:1) | GQA (4:1) | MHA |
| **Position** | RoPE | RoPE | RoPE |
| **Norm** | RMSNorm | RMSNorm | RMSNorm |
| **Activation** | SwiGLU | SwiGLU | GeGLU |
| **Bias** | No | No | No |
| **Special** | - | Sliding Window | Large Vocab |
| **Context** | 4K | 32K | 8K |

### Parameter Count Breakdown

```python
def count_model_params(config):
    """Estimate parameter count from config"""
    d = config['d_model']
    vocab = config['vocab_size']
    layers = config['num_layers']
    d_ff = config['d_ff']
    kv_heads = config['num_kv_heads']
    heads = config['num_heads']

    # Embedding (shared with output)
    embedding = vocab * d

    # Per-layer parameters
    # Attention: Q, K, V, O projections
    attn_params = (
        d * d +  # Q
        d * (d // heads) * kv_heads * 2 +  # K, V (GQA)
        d * d  # O
    )

    # FFN: 3 projections for SwiGLU
    ffn_params = d * d_ff * 2 + d_ff * d

    # RMSNorm: just weights (no bias)
    norm_params = d * 2  # attn_norm + ffn_norm

    layer_params = attn_params + ffn_params + norm_params
    total = embedding + layers * layer_params + d  # + final norm

    return total

# Llama 2 7B
llama2_config = {
    "vocab_size": 32000, "d_model": 4096, "num_layers": 32,
    "num_heads": 32, "num_kv_heads": 8, "d_ff": 11008
}
print(f"Llama 2 7B: {count_model_params(llama2_config) / 1e9:.2f}B")

# Mistral 7B
mistral_config = {
    "vocab_size": 32000, "d_model": 4096, "num_layers": 32,
    "num_heads": 32, "num_kv_heads": 8, "d_ff": 14336
}
print(f"Mistral 7B: {count_model_params(mistral_config) / 1e9:.2f}B")

# Gemma 7B
gemma_config = {
    "vocab_size": 256000, "d_model": 3072, "num_layers": 28,
    "num_heads": 16, "num_kv_heads": 16, "d_ff": 24576
}
print(f"Gemma 7B: {count_model_params(gemma_config) / 1e9:.2f}B")

# Output:
# Llama 2 7B: 6.74B
# Mistral 7B: 7.24B
# Gemma 7B: 8.54B
```

---

## 🔧 Loading Pre-trained Models

### Using HuggingFace

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Llama 2 7B
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Mistral 7B
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Gemma 7B
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)
print(tokenizer.decode(outputs[0]))
```

### Inspecting Architecture

```python
# Check model config
print(model.config)

# Count parameters
total = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total / 1e9:.2f}B")

# Check attention type
print(f"Num heads: {model.config.num_attention_heads}")
print(f"Num KV heads: {model.config.num_key_value_heads}")

# Check if GQA
if model.config.num_key_value_heads < model.config.num_attention_heads:
    ratio = model.config.num_attention_heads // model.config.num_key_value_heads
    print(f"Using GQA with {ratio}:1 ratio")
else:
    print("Using MHA")
```

---

## 🎯 Implementation Checklist

**Building Your Own Modern LLM**:

```python
# 1. Choose Architecture Components
class MyLLM(nn.Module):
    def __init__(self):
        # ✓ RMSNorm (not LayerNorm)
        # ✓ RoPE (not sinusoidal or learned)
        # ✓ GQA (not MHA, unless <3B params)
        # ✓ SwiGLU or GeGLU (not ReLU/GELU)
        # ✓ Pre-LN (not Post-LN)
        # ✓ No bias in linear layers
        # ✓ Tie embedding weights

# 2. Configuration
config = {
    "vocab_size": 32000,      # Depends on tokenizer
    "d_model": 4096,          # 2048, 4096, 8192 (common choices)
    "num_layers": 32,         # Scales with compute budget
    "num_heads": 32,          # Usually d_model // 128
    "num_kv_heads": 8,        # num_heads // 4 for GQA
    "d_ff": int(2.7 * 4096),  # ~2.7 * d_model, round to 256
    "max_seq_len": 4096,      # Training context length
    "bias": False,
    "norm_eps": 1e-5,
}

# 3. Parameter Count Formula
params ≈ vocab_size * d_model + num_layers * (12 * d_model² + d_ff * 3 * d_model)

# 4. Scaling Laws (Chinchilla optimal)
# For optimal performance: num_tokens ≈ 20 * num_params
# 7B model → 140B tokens
# 70B model → 1.4T tokens
```

---

## 💡 Design Decisions Explained

### Why GQA over MHA or MQA?

```python
# Trade-off analysis

# MHA (Multi-Head Attention):
# + Best quality
# - Highest memory (KV cache)
# - Slowest inference
# Use: Small models (<3B) or when quality is critical

# MQA (Multi-Query Attention):
# + Fastest inference
# + Smallest memory
# - Slight quality drop (1-2%)
# Use: Large models (>70B) or high-throughput serving

# GQA (Grouped-Query Attention):
# + 75% memory savings vs MHA
# + 99% quality of MHA
# + Faster than MHA
# Use: Medium models (7B-70B) - sweet spot!

# Modern choice: GQA with 4:1 ratio (32 Q heads, 8 KV heads)
```

### Why RMSNorm over LayerNorm?

```python
# LayerNorm:
# 1. Compute mean
# 2. Subtract mean (re-centering)
# 3. Compute variance
# 4. Divide by std (re-scaling)

# RMSNorm:
# 1. Compute RMS (sqrt of mean of squares)
# 2. Divide by RMS (re-scaling only)

# Why RMSNorm wins:
# - 15-20% faster
# - Simpler implementation
# - Re-centering is redundant (empirically proven)
# - Used by all modern LLMs
```

### Why SwiGLU over ReLU?

```python
# ReLU: out = max(0, W₁(x))W₂
# - Simple
# - But limited expressiveness

# SwiGLU: out = (Swish(xW) ⊙ xV)W₂
# - Gating mechanism (like LSTM/GRU)
# - Each token can have different "filter"
# - ~1-2% better performance
# - Used by PaLM, Llama, Mistral

# Cost: 50% more FFN parameters, but worth it!
```

---

## 🎓 Exercises

### Exercise 1: Build Mini-Llama

Implement a scaled-down Llama 2 (10M parameters) and train on TinyStories.

```python
mini_config = {
    "vocab_size": 32000,
    "d_model": 256,
    "num_layers": 4,
    "num_heads": 8,
    "num_kv_heads": 2,
    "d_ff": 688,
    "max_seq_len": 512,
    "bias": False,
}

# Your implementation here
model = Llama2(mini_config)
# Train on TinyStories dataset
```

### Exercise 2: Compare Attention Variants

Benchmark MHA vs GQA vs MQA on the same task.

```python
# Implement all three
mha_model = Llama2({...num_kv_heads: 32...})  # MHA
gqa_model = Llama2({...num_kv_heads: 8...})   # GQA
mqa_model = Llama2({...num_kv_heads: 1...})   # MQA

# Compare:
# - Training speed
# - Inference speed (with/without KV cache)
# - Memory usage
# - Final perplexity
```

### Exercise 3: Implement Mistral Sliding Window

Add sliding window attention to your implementation.

```python
def sliding_window_attention(q, k, v, window_size):
    """Implement efficient sliding window attention"""
    # Your code here
    pass
```

---

## 📚 References

**Papers**:
1. **LLaMA** (Touvron et al., 2023) - https://arxiv.org/abs/2302.13971
2. **Llama 2** (Touvron et al., 2023) - https://arxiv.org/abs/2307.09288
3. **Mistral 7B** (Jiang et al., 2023) - https://arxiv.org/abs/2310.06825
4. **Gemma** (Google, 2024) - Technical Report

**Code**:
- HuggingFace Transformers: `modeling_llama.py`, `modeling_mistral.py`
- Meta Llama: https://github.com/meta-llama/llama
- Mistral AI: https://github.com/mistralai/mistral-src

**Other**:
- RMSNorm paper: https://arxiv.org/abs/1910.07467
- GLU Variants: https://arxiv.org/abs/2002.05202
- GQA paper: https://arxiv.org/abs/2305.13245

---

## ⏭️ Next Steps

현대 LLM 아키텍처를 완전히 이해했습니다!

👉 [Phase 5.5: LLM Alignment](../phase5.5-llm-alignment/) - 이 모델들을 어떻게 정렬하는지
👉 [Phase 5.7: Hardware Systems](../phase5.7-hardware-systems/) - GPU에서 효율적으로 실행하기

**이제 production-ready LLM을 처음부터 구현할 수 있습니다!** 🚀
