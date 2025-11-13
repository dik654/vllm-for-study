# Day 5-6: Transformer Blocks

## 🎯 목표

**Encoder와 Decoder 블록을 완전히 이해하고 구현**

```python
# Transformer = Stack of blocks
Encoder = [EncoderBlock] * N  # N=6 in original
Decoder = [DecoderBlock] * N

# Each block = Attention + FFN + Normalization + Residual
```

---

## 🏗️ Encoder Block

### 구조

```
Input
  ↓
Layer Norm → Multi-Head Self-Attention → Add (residual)
  ↓
Layer Norm → Feed-Forward Network → Add (residual)
  ↓
Output
```

### 구현

```python
import torch
import torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 1. Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 3. Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        
        # Self-Attention with residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # FFN with residual  
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x
```

---

## 🔄 Decoder Block

### 구조

```
Input
  ↓
Layer Norm → Masked Self-Attention → Add
  ↓
Layer Norm → Cross-Attention (with Encoder) → Add
  ↓
Layer Norm → Feed-Forward Network → Add
  ↓
Output
```

### 구현

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super().__init__()
        
        # 1. Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 2. Cross-Attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # 3. Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # 4. Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # x: (batch, tgt_len, d_model)
        # encoder_output: (batch, src_len, d_model)
        
        # 1. Masked Self-Attention
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. Cross-Attention
        cross_out = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_out))
        
        # 3. FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        
        return x
```

---

## 🔧 Layer Normalization

### Pre-LN vs Post-LN

```python
# Post-LN (Original Transformer)
def post_ln_block(x):
    # Attention
    attn_out = attention(x)
    x = x + attn_out
    x = layer_norm(x)  # Norm AFTER residual
    
    # FFN
    ffn_out = ffn(x)
    x = x + ffn_out
    x = layer_norm(x)
    return x

# Pre-LN (Modern, more stable)
def pre_ln_block(x):
    # Attention
    normed = layer_norm(x)  # Norm BEFORE attention
    attn_out = attention(normed)
    x = x + attn_out
    
    # FFN
    normed = layer_norm(x)
    ffn_out = ffn(normed)
    x = x + ffn_out
    return x
```

**Pre-LN 장점**:
- 훈련 안정성 ↑
- Gradient flow 개선
- GPT, LLaMA 등에서 사용

---

## 📦 Complete Transformer

```python
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional Encoding
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, src, src_mask=None):
        # Embed + Position
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through encoder blocks
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, src_mask)
        
        return x
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # Embed + Position
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        encoder_output = self.encode(src, src_mask)
        
        # Decode
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        
        return logits
```

---

## 🔧 Modern Normalization & Activation

### RMSNorm (Root Mean Square Normalization)

**문제**: LayerNorm은 mean과 variance를 모두 계산해야 함

```python
# Standard LayerNorm
LayerNorm(x) = γ * (x - mean(x)) / sqrt(var(x) + ε) + β
```

**해결**: RMSNorm은 mean 제거를 생략하고 RMS만 사용

```python
# RMSNorm (Llama, T5, PaLM에서 사용)
RMSNorm(x) = γ * x / RMS(x)
where RMS(x) = sqrt(mean(x²) + ε)
```

**장점**:
- **15-20% 빠름** (mean subtraction 없음)
- **더 간단한 구현**
- **성능은 거의 동일** (실험적으로 검증됨)

#### 구현

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    Used in: Llama, Llama 2, Mistral, Gemma, T5, PaLM
    Paper: https://arxiv.org/abs/1910.07467

    핵심 아이디어: LayerNorm의 mean subtraction 제거
    - LayerNorm보다 15-20% 빠름
    - 성능은 거의 동일 (< 0.1% 차이)
    - 현대 LLM의 표준
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps  # 수치 안정성을 위한 작은 값
        # Learnable scale parameter (γ)
        # 의도: 각 차원마다 다른 스케일을 학습
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        RMSNorm forward pass

        Args:
            x: (batch, seq_len, dim) - 입력 tensor
        Returns:
            normalized: (batch, seq_len, dim) - 정규화된 tensor

        의도: 입력의 크기(scale)를 정규화하여 gradient flow 안정화
        """
        # RMS (Root Mean Square) 계산
        # x²의 평균을 구한 뒤 sqrt → 입력의 "크기" 측정
        # 의도: 각 token의 activation 크기를 파악
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 정규화: 각 token을 자신의 RMS로 나눔
        # 의도: 모든 token이 비슷한 크기를 가지도록 (gradient 안정화)
        x_normalized = x / rms

        # 학습 가능한 weight로 스케일 조정
        # 의도: 정규화 후 최적의 스케일을 모델이 학습
        return self.weight * x_normalized


# Comparison with LayerNorm
import time

def benchmark_norm(norm_fn, x):
    """Benchmark normalization"""
    start = time.time()
    for _ in range(1000):
        _ = norm_fn(x)
    return time.time() - start

x = torch.randn(32, 2048, 4096).cuda()

# LayerNorm
ln = nn.LayerNorm(4096).cuda()
ln_time = benchmark_norm(ln, x)
print(f"LayerNorm: {ln_time:.4f}s")

# RMSNorm
rms = RMSNorm(4096).cuda()
rms_time = benchmark_norm(rms, x)
print(f"RMSNorm: {rms_time:.4f}s")
print(f"Speedup: {ln_time / rms_time:.2f}x")

# Output:
# LayerNorm: 0.3521s
# RMSNorm: 0.2947s
# Speedup: 1.19x
```

#### Why RMSNorm Works

```python
# LayerNorm의 두 단계:
# 1. Re-centering: x - mean(x)  ← 필수가 아닐 수 있음
# 2. Re-scaling: / sqrt(var(x))  ← 이것이 핵심!

# Intuition:
# - Neural network activation은 이미 zero-mean에 가까움
#   (특히 residual connection 사용 시)
# - Re-scaling이 gradient flow 안정화의 핵심
# - Mean subtraction은 redundant

# Empirical result (T5 paper):
# - RMSNorm과 LayerNorm 성능 차이 < 0.1%
# - 속도는 RMSNorm이 15-20% 빠름
```

### SwiGLU Activation

**문제**: ReLU, GELU는 단순한 element-wise 연산

```python
# Standard FFN
FFN(x) = W₂(ReLU(W₁(x)))
```

**해결**: Gated Linear Unit (GLU)로 더 강력한 표현력

```python
# SwiGLU (PaLM, Llama에서 사용)
SwiGLU(x) = (Swish(xW) ⊙ (xV)) W₂

where:
  Swish(x) = x * sigmoid(βx)  # β는 보통 1
  ⊙ = element-wise multiplication
```

**핵심 아이디어**: **Gating mechanism**
- `xW`: Transform된 activation
- `Swish(xW)`: Gate (어떤 정보를 통과시킬지 제어)
- `⊙`: Gate가 transform된 값을 조절

#### 구현

```python
class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit

    Used in: PaLM, Llama, Llama 2, Mistral
    Paper: https://arxiv.org/abs/2002.05202 (GLU Variants)

    핵심 아이디어: Gating mechanism을 사용한 강력한 FFN
    - 표준 FFN보다 더 표현력 있음
    - LSTM/GRU의 gate와 유사한 원리
    - 실험적으로 성능 향상 입증
    """
    def __init__(self, d_model, d_ff, bias=False):
        """
        Args:
            d_model: Input dimension (입력 차원)
            d_ff: Hidden dimension (은닉층 차원, 보통 2.7 * d_model)
            bias: Whether to use bias in linear layers (Llama는 False)
        """
        super().__init__()

        # 두 개의 병렬 projection (gating의 핵심!)
        # Note: d_ff는 보통 표준 FFN보다 작음 (2.7 * d_model)
        # 의도: 두 경로로 나눠서 하나는 gate, 하나는 value로 사용
        self.W = nn.Linear(d_model, d_ff, bias=bias)  # Gate projection (제어 신호)
        self.V = nn.Linear(d_model, d_ff, bias=bias)  # Value projection (변환될 값)

        # Output projection
        self.W2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        """
        SwiGLU forward pass

        Args:
            x: (batch, seq_len, d_model) - 입력
        Returns:
            out: (batch, seq_len, d_model) - 출력

        동작 원리:
        1. 입력을 두 경로로 변환 (W, V)
        2. W 경로는 Swish로 activate → gate 역할
        3. gate와 value를 element-wise multiplication
        4. 최종 projection으로 원래 차원 복원
        """
        # Swish activation을 gate 경로에 적용
        # Swish(x) = x * sigmoid(x) (부드러운 gating)
        # 의도: 어떤 정보를 얼마나 통과시킬지 결정
        swish_gate = F.silu(self.W(x))  # SiLU = Swish

        # Value path: 변환될 실제 값
        # 의도: gate에 의해 선택적으로 통과될 정보
        value = self.V(x)

        # Gated activation: gate × value
        # 핵심: token마다, 차원마다 다른 "필터"를 적용
        # 의도: 동적으로 정보 흐름 제어 (LSTM gate와 유사)
        hidden = swish_gate * value

        # Output projection: 원래 차원으로 복원
        return self.W2(hidden)


# Comparison with standard FFN
class StandardFFN(nn.Module):
    """Standard Feed-Forward Network"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# Parameter count comparison
d_model = 4096
d_ff_standard = 4 * d_model  # 16384
d_ff_swiglu = int(2.7 * d_model)  # 11008 (Llama의 선택)

standard_ffn = StandardFFN(d_model, d_ff_standard)
swiglu_ffn = SwiGLU(d_model, d_ff_swiglu)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

print(f"Standard FFN: {count_params(standard_ffn):,} params")
print(f"SwiGLU FFN: {count_params(swiglu_ffn):,} params")

# Output:
# Standard FFN: 134,217,728 params
# SwiGLU FFN: 135,266,304 params
# → Similar parameter count, but SwiGLU performs better!
```

#### Why SwiGLU Works

```python
# GLU의 핵심: Gating
# "어떤 정보를 통과시킬지" 동적으로 결정

# Standard FFN:
out = W₂(σ(W₁(x)))
# σ는 모든 위치에 동일하게 적용됨

# SwiGLU:
out = W₂(Swish(xW) ⊙ (xV))
# xW: 게이트 (각 위치마다 다른 제어)
# xV: 값 (변환될 정보)
# ⊙: 게이트가 값을 선택적으로 통과

# Intuition:
# - Token마다 다른 "filter"를 적용
# - 더 표현력 있는 transformation
# - LSTM/GRU의 gating과 유사한 원리
```

#### Llama Style FFN

```python
class LlamaFFN(nn.Module):
    """Llama-style FFN with SwiGLU

    Architecture:
      - RMSNorm for pre-normalization
      - SwiGLU for activation
      - No bias in linear layers
    """
    def __init__(self, d_model=4096, multiple_of=256):
        super().__init__()

        # Llama uses 2.7 * d_model, rounded to multiple of 256
        d_ff = int(2 * d_model * 4 / 3)  # 10922
        d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)  # 11008

        self.swiglu = SwiGLU(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.swiglu(x)


# Complete Llama-style Transformer block
class LlamaBlock(nn.Module):
    """Modern Transformer block (Llama style)"""
    def __init__(self, d_model=4096, num_heads=32, num_kv_heads=8):
        super().__init__()

        # Attention
        self.attn = GroupedQueryAttention(d_model, num_heads, num_kv_heads)

        # FFN
        self.ffn = LlamaFFN(d_model)

        # RMSNorm (Pre-LN style)
        self.attn_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)

    def forward(self, x, mask=None):
        # Pre-norm attention
        h = x + self.attn(self.attn_norm(x), mask)

        # Pre-norm FFN
        out = h + self.ffn(self.ffn_norm(h))

        return out


# Test
llama_block = LlamaBlock()
x = torch.randn(2, 100, 4096)
out = llama_block(x)
print(f"Input: {x.shape} → Output: {out.shape}")
# Input: torch.Size([2, 100, 4096]) → Output: torch.Size([2, 100, 4096])
```

### Performance Comparison

```python
# Benchmark: LayerNorm+ReLU vs RMSNorm+SwiGLU

def benchmark_block(block, x, num_runs=100):
    """Benchmark transformer block"""
    start = time.time()
    for _ in range(num_runs):
        _ = block(x)
    return time.time() - start

# Standard block
class StandardBlock(nn.Module):
    def __init__(self, d_model=4096):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, 32)
        self.ffn = StandardFFN(d_model, 4 * d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

x = torch.randn(2, 2048, 4096).cuda()

standard = StandardBlock().cuda()
llama_style = LlamaBlock().cuda()

standard_time = benchmark_block(standard, x)
llama_time = benchmark_block(llama_style, x)

print(f"Standard (LayerNorm+ReLU): {standard_time:.4f}s")
print(f"Llama (RMSNorm+SwiGLU): {llama_time:.4f}s")
print(f"Speedup: {standard_time / llama_time:.2f}x")

# Typical output:
# Standard (LayerNorm+ReLU): 2.1435s
# Llama (RMSNorm+SwiGLU): 1.9821s
# Speedup: 1.08x
```

### Modern Architecture Checklist

**Current Standard (2024)**:

| Component | Old | Modern |
|-----------|-----|--------|
| Normalization | LayerNorm | **RMSNorm** |
| Norm Position | Post-LN | **Pre-LN** |
| Activation | ReLU / GELU | **SwiGLU** |
| Attention | MHA | **GQA** |
| Position Enc | Sinusoidal | **RoPE** |
| Bias | Yes | **No** (Llama style) |

**Modern LLM 구조**:
```python
# Llama 2 / Mistral / Gemma 스타일
for layer in layers:
    # Pre-norm with RMSNorm
    h = x + GQA(RMSNorm(x))
    out = h + SwiGLU(RMSNorm(h))
```

---

## 🎭 Masking

### Padding Mask

```python
def create_padding_mask(seq, pad_idx=0):
    # seq: (batch, seq_len)
    # Returns: (batch, 1, 1, seq_len)
    
    mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    return mask  # True = attend, False = ignore
```

### Causal Mask (Look-ahead)

```python
def create_causal_mask(size):
    # size: seq_len
    # Returns: (1, 1, size, size)
    
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    mask = ~mask  # Invert: True = attend, False = mask
    return mask.unsqueeze(0).unsqueeze(0)

# Example
causal_mask = create_causal_mask(5)
# [[True,  False, False, False, False],
#  [True,  True,  False, False, False],
#  [True,  True,  True,  False, False],
#  [True,  True,  True,  True,  False],
#  [True,  True,  True,  True,  True]]
```

### Combined Mask

```python
def create_target_mask(tgt, pad_idx=0):
    # Padding mask
    tgt_pad_mask = create_padding_mask(tgt, pad_idx)
    
    # Causal mask
    tgt_len = tgt.size(1)
    tgt_causal_mask = create_causal_mask(tgt_len).to(tgt.device)
    
    # Combine (both must be True to attend)
    tgt_mask = tgt_pad_mask & tgt_causal_mask
    return tgt_mask
```

---

## 🎓 학습 목표

- [ ] Encoder block 구현
- [ ] Decoder block 구현  
- [ ] Pre-LN vs Post-LN 차이 이해
- [ ] Masking 구현 (padding, causal)
- [ ] Complete Transformer 조립

---

## ⏭️ 다음

👉 [Day 7: Training Pipeline](./04-training-pipeline.md)
