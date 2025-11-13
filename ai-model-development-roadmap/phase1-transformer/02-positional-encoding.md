# Day 3-4: Positional Encoding

## 🎯 목표

**Transformer에 위치 정보를 주입하는 다양한 방법 이해 및 구현**

```python
# Transformer는 순서를 모름!
"I love you" == "you love I" (Transformer 입장에서)

# Positional Encoding 추가
"I love you" + positions → Transformer → correct understanding!
```

---

## 📖 왜 Positional Encoding이 필요한가?

### RNN vs Transformer

```python
# RNN: 순서가 내장됨
for t in range(seq_len):
    hidden[t] = rnn(input[t], hidden[t-1])
# → 자연스럽게 순서 정보 포함

# Transformer: Attention은 순서 무관!
attention_weights = softmax(Q @ K.T / sqrt(d_k))
output = attention_weights @ V
# → 단어 순서를 바꿔도 결과 동일!
```

**해결**: 입력에 위치 정보 추가!

---

## 🔢 1. Sinusoidal Positional Encoding (Original)

**"Attention is All You Need" 논문의 방법**

### 수식

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

pos: 위치 (0, 1, 2, ...)
i: dimension index (0, 1, ..., d_model/2)
```

### 구현

```python
import numpy as np
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # Position과 dimension을 위한 matrix 생성
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # sin for even indices, cos for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (max_len, d_model) → (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        return x

# 사용
d_model = 512
pos_encoder = SinusoidalPositionalEncoding(d_model)

# Input embeddings
x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)

# Add positional encoding
x_with_pos = pos_encoder(x)
```

### 시각화

```python
import matplotlib.pyplot as plt

# Generate PE
pe = SinusoidalPositionalEncoding(128, max_len=100)
pe_matrix = pe.pe.squeeze(0).numpy()  # (100, 128)

plt.figure(figsize=(15, 5))
plt.imshow(pe_matrix, aspect='auto', cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position')
plt.colorbar()
plt.title('Sinusoidal Positional Encoding')
plt.show()

# 패턴: 저주파 (왼쪽) → 고주파 (오른쪽)
```

### 장점

1. **Extrapolation**: 훈련보다 긴 sequence도 처리 가능
   ```python
   # Train: max_len=512
   # Inference: seq_len=1000 → works!
   ```

2. **상대적 위치 학습 가능**:
   ```
   sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
   → PE(pos+k)를 PE(pos)의 선형 결합으로 표현 가능!
   ```

---

## 📚 2. Learned Positional Embedding

**BERT, GPT 방식: Embedding layer로 학습**

### 구현

```python
class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        
        # Learned embedding table
        self.pos_embedding = nn.Embedding(max_len, d_model)
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        
        # Position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # (1, seq_len)
        
        # Get position embeddings
        pos_emb = self.pos_embedding(positions)  # (1, seq_len, d_model)
        
        # Add
        x = x + pos_emb
        return x

# 사용
pos_emb = LearnedPositionalEmbedding(d_model=512, max_len=512)
x_with_pos = pos_emb(x)
```

### 장단점

**장점**:
- 유연함 (데이터에서 최적의 encoding 학습)
- 구현 간단

**단점**:
- Extrapolation 불가 (max_len 초과 불가)
- 더 많은 파라미터 필요

---

## 🔄 3. RoPE (Rotary Position Embedding)

**LLaMA, PaLM 등 최신 모델에서 사용**

### 핵심 아이디어

Positional information을 **rotation**으로 표현!

```python
# 2D rotation matrix
R(θ) = [[cos(θ), -sin(θ)],
        [sin(θ),  cos(θ)]]

# Position m의 vector를 m*θ만큼 회전
q_m = R(m*θ) @ q
k_n = R(n*θ) @ k

# Attention: q_m^T k_n
q_m^T k_n = q^T R(m*θ)^T R(n*θ) k
          = q^T R((n-m)*θ) k

# 결과: 상대적 위치 (n-m)만 의존!
```

### 구현

```python
class RoPEPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        self.d_model = d_model
        
        # Compute theta
        # θ_i = 10000^(-2i/d)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, x, seq_len):
        # x: (batch, seq_len, num_heads, head_dim)
        
        # Position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        
        # Compute frequencies
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)  # (seq_len, d_model/2)
        
        # Concatenate sin and cos
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_model)
        
        # Compute sin and cos
        cos = emb.cos()  # (seq_len, d_model)
        sin = emb.sin()  # (seq_len, d_model)
        
        return cos, sin
    
def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding"""
    # x: (batch, seq_len, num_heads, head_dim)
    # cos, sin: (seq_len, head_dim)
    
    # Rotate
    x1 = x[..., ::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # Apply rotation
    rotated = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    rotated = rotated.flatten(-2)  # Merge last two dims
    return rotated

# 사용 in Attention
class RoPEAttention(nn.Module):
    def forward(self, q, k, v):
        # q, k, v: (batch, seq_len, num_heads, head_dim)
        
        seq_len = q.size(1)
        
        # Get RoPE
        rope = RoPEPositionalEncoding(head_dim)
        cos, sin = rope(q, seq_len)
        
        # Apply to Q and K
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Standard attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        return output
```

### 장점

1. **상대적 위치만 의존** (이론적으로 증명됨)
2. **Extrapolation 가능** (Sinusoidal처럼)
3. **효율적** (추가 파라미터 없음)

---

## 📏 4. ALiBi (Attention with Linear Biases)

**가장 간단! Attention score에 bias만 추가**

### 핵심 아이디어

```python
# Standard attention
scores = Q @ K.T / sqrt(d_k)

# ALiBi: Add linear bias based on distance
bias = -m * abs(i - j)  # m: head-specific slope
scores = Q @ K.T / sqrt(d_k) + bias

# 예: m=1일 때
# Position:  0   1   2   3
#   0:      [0  -1  -2  -3]
#   1:      [-1  0  -1  -2]
#   2:      [-2 -1   0  -1]
#   3:      [-3 -2  -1   0]
```

### 구현

```python
class ALiBiPositionalBias(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        
        # Head-specific slopes
        # Geometric sequence: 2^(-8/n), 2^(-16/n), ...
        slopes = torch.Tensor(self._get_slopes(num_heads))
        self.register_buffer('slopes', slopes)
    
    def _get_slopes(self, num_heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]
        
        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            # Closest power of 2
            closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
            return (get_slopes_power_of_2(closest_power_of_2) +
                    self._get_slopes(2 * closest_power_of_2)[:num_heads - closest_power_of_2])
    
    def forward(self, seq_len):
        # Create distance matrix
        # (seq_len, seq_len)
        arange = torch.arange(seq_len, device=self.slopes.device)
        distance = arange[None, :] - arange[:, None]  # Broadcasting
        distance = distance.abs()
        
        # Apply slopes
        # (num_heads, seq_len, seq_len)
        bias = -distance[None, :, :] * self.slopes[:, None, None]
        
        return bias

# 사용 in Attention
class ALiBiAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.alibi = ALiBiPositionalBias(num_heads)
        # ... Q, K, V projections
    
    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head
        # (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Add ALiBi bias
        alibi_bias = self.alibi(seq_len)  # (num_heads, seq_len, seq_len)
        scores = scores + alibi_bias[None, :, :, :]  # Broadcast batch dim
        
        # Softmax and output
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        
        return output
```

### 장점

1. **No position embedding at all!** (가장 효율적)
2. **Excellent extrapolation** (훈련 길이의 10배도 가능)
3. **간단함**

---

## 📊 비교표

| Method | Params | Extrapolation | Relative Position | Speed |
|--------|--------|---------------|-------------------|-------|
| Sinusoidal | 0 | ✅ Good | ✅ Yes (implicit) | Fast |
| Learned | O(L×D) | ❌ No | ❌ No | Fast |
| RoPE | 0 | ✅ Excellent | ✅ Yes (explicit) | Medium |
| ALiBi | 0 | ✅ Best | ✅ Yes (explicit) | Fastest |

**추천**:
- **General use**: RoPE (LLaMA, PaLM)
- **Long sequences**: ALiBi (MPT, BLOOM)
- **Classic**: Sinusoidal (Original Transformer)

---

## 🎓 학습 목표 체크리스트

- [ ] Sinusoidal PE의 수식 이해 및 구현
- [ ] Learned PE 구현
- [ ] RoPE의 rotation 개념 이해
- [ ] ALiBi의 linear bias 구현
- [ ] 각 방법의 장단점 비교
- [ ] Extrapolation 실험

---

## 💻 실습 과제

1. **Sinusoidal PE 시각화**
   - 다양한 d_model에서 패턴 관찰
   - High frequency vs Low frequency

2. **Extrapolation 실험**
   - Train: 512 tokens
   - Test: 1024, 2048 tokens
   - 각 방법별 성능 비교

3. **Custom PE 설계**
   - 자신만의 positional encoding 방법 제안

---

## 📚 참고 자료

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Sinusoidal
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE
- [Train Short, Test Long](https://arxiv.org/abs/2108.12409) - ALiBi

---

## ⏭️ 다음

Positional encoding을 마스터했습니다!

👉 [Day 5-6: Transformer Blocks](./03-transformer-blocks.md)에서 **전체 Transformer**를 조립합니다!
