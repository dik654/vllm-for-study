# Day 1-2: Attention 메커니즘 완전 구현

## 📋 목차
1. [Attention의 수학적 기초](#1-mathematical-foundations)
2. [Scaled Dot-Product Attention](#2-scaled-dot-product-attention)
3. [Multi-Head Attention](#3-multi-head-attention)
4. [Masking과 Attention Patterns](#4-masking-and-patterns)
5. [최적화 기법](#5-optimization)

---

## 1. Attention의 수학적 기초

### 📖 이론

**Attention**은 입력 시퀀스의 어떤 부분에 "주의"를 기울일지 결정하는 메커니즘입니다.

#### 핵심 개념

**Query (Q), Key (K), Value (V):**
- **Query**: "무엇을 찾고 있는가?"
- **Key**: "나는 무엇인가?"
- **Value**: "실제 정보"

**Attention 수식:**

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

여기서:
- $Q \in \mathbb{R}^{n \times d_k}$: Query 행렬
- $K \in \mathbb{R}^{m \times d_k}$: Key 행렬
- $V \in \mathbb{R}^{m \times d_v}$: Value 행렬
- $d_k$: Key/Query의 차원
- $n$: Query 시퀀스 길이
- $m$: Key/Value 시퀀스 길이

### 💻 실습 1: NumPy로 Attention 구현 (수식 그대로)

```python
# examples/attention_from_scratch.py
import numpy as np
import matplotlib.pyplot as plt

def softmax(x, axis=-1):
    """
    Numerically stable softmax
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention (NumPy 구현)

    Args:
        Q: Query matrix (batch_size, n_queries, d_k)
        K: Key matrix (batch_size, n_keys, d_k)
        V: Value matrix (batch_size, n_keys, d_v)
        mask: Optional mask (batch_size, n_queries, n_keys)

    Returns:
        output: (batch_size, n_queries, d_v)
        attention_weights: (batch_size, n_queries, n_keys)
    """
    # Step 1: Q @ K^T
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, n_q, n_k)

    # Step 2: Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)

    # Step 3: Apply mask (optional)
    if mask is not None:
        scores = scores + (mask * -1e9)  # Large negative value

    # Step 4: Softmax
    attention_weights = softmax(scores, axis=-1)

    # Step 5: weighted sum of values
    output = np.matmul(attention_weights, V)  # (batch, n_q, d_v)

    return output, attention_weights

# 테스트
if __name__ == "__main__":
    # 간단한 예제
    batch_size = 2
    seq_len = 4
    d_model = 8

    # 랜덤 Q, K, V 생성
    np.random.seed(42)
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    # Attention 계산
    output, attention_weights = scaled_dot_product_attention(Q, K, V)

    print("=== Scaled Dot-Product Attention 테스트 ===")
    print(f"Q shape: {Q.shape}")
    print(f"K shape: {K.shape}")
    print(f"V shape: {V.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")

    # Attention weights 시각화
    plt.figure(figsize=(10, 4))
    for i in range(min(2, batch_size)):
        plt.subplot(1, 2, i+1)
        plt.imshow(attention_weights[i], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title(f'Attention Weights (Batch {i})')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig('attention_weights.png')
    print("\n✓ Attention weights 시각화 저장: attention_weights.png")
```

### 💻 실습 2: Attention의 직관적 이해

```python
# examples/attention_intuition.py
import numpy as np
import matplotlib.pyplot as plt

def demonstrate_attention():
    """
    Attention 메커니즘의 직관적 데모
    """
    # 간단한 예제: "The cat sat on the mat"
    words = ["The", "cat", "sat", "on", "the", "mat"]
    n = len(words)

    # 간단한 단어 임베딩 (랜덤하지만 의미있게)
    np.random.seed(42)
    embeddings = {
        "The": np.array([1.0, 0.0, 0.0, 0.5]),
        "cat": np.array([0.0, 1.0, 0.0, 0.8]),
        "sat": np.array([0.0, 0.0, 1.0, 0.3]),
        "on": np.array([0.5, 0.0, 0.5, 0.2]),
        "the": np.array([1.0, 0.0, 0.0, 0.5]),
        "mat": np.array([0.2, 0.1, 0.0, 0.9]),
    }

    # Query, Key, Value 행렬 (단순화: identity mapping)
    Q = np.array([embeddings[w] for w in words])
    K = np.array([embeddings[w] for w in words])
    V = np.array([embeddings[w] for w in words])

    # Attention 계산
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    output = attention_weights @ V

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Attention scores (before softmax)
    im1 = axes[0].imshow(scores, cmap='RdBu_r', aspect='auto')
    axes[0].set_xticks(range(n))
    axes[0].set_yticks(range(n))
    axes[0].set_xticklabels(words)
    axes[0].set_yticklabels(words)
    axes[0].set_title('Attention Scores\n(Q·K^T / √d_k)')
    axes[0].set_xlabel('Key')
    axes[0].set_ylabel('Query')
    plt.colorbar(im1, ax=axes[0])

    # 2. Attention weights (after softmax)
    im2 = axes[1].imshow(attention_weights, cmap='viridis', aspect='auto')
    axes[1].set_xticks(range(n))
    axes[1].set_yticks(range(n))
    axes[1].set_xticklabels(words)
    axes[1].set_yticklabels(words)
    axes[1].set_title('Attention Weights\n(after softmax)')
    axes[1].set_xlabel('Key')
    axes[1].set_ylabel('Query')
    plt.colorbar(im2, ax=axes[1])

    # 3. 특정 단어에 대한 attention 분포
    query_word = "cat"
    query_idx = words.index(query_word)
    axes[2].bar(range(n), attention_weights[query_idx])
    axes[2].set_xticks(range(n))
    axes[2].set_xticklabels(words, rotation=45)
    axes[2].set_title(f'Attention weights for "{query_word}"')
    axes[2].set_ylabel('Weight')

    plt.tight_layout()
    plt.savefig('attention_intuition.png', dpi=150)
    print("✓ Attention 직관 시각화 저장: attention_intuition.png")

    # "cat"이 어디에 주목하는지 출력
    print(f"\n'{query_word}'이 주목하는 단어:")
    for i, (word, weight) in enumerate(zip(words, attention_weights[query_idx])):
        print(f"  {word}: {weight:.3f}")

if __name__ == "__main__":
    demonstrate_attention()
```

---

## 2. Scaled Dot-Product Attention (PyTorch 구현)

### 📖 왜 Scaling?

$\sqrt{d_k}$로 나누는 이유:
- $d_k$가 크면 dot product의 magnitude가 커짐
- Softmax의 gradient가 매우 작아짐 (vanishing gradient)
- Scaling으로 안정적인 학습 가능

### 💻 실습 3: PyTorch로 효율적 구현

```python
# examples/attention_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention_torch(Q, K, V, mask=None, dropout=None):
    """
    PyTorch로 구현한 Scaled Dot-Product Attention

    Args:
        Q: (batch, n_heads, seq_len_q, d_k)
        K: (batch, n_heads, seq_len_k, d_k)
        V: (batch, n_heads, seq_len_v, d_v)
        mask: (batch, 1, seq_len_q, seq_len_k) or broadcastable
        dropout: Dropout layer

    Returns:
        output: (batch, n_heads, seq_len_q, d_v)
        attention_weights: (batch, n_heads, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Dropout (optional)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Weighted sum
    output = torch.matmul(attention_weights, V)

    return output, attention_weights

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention as a Module
    """
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch, n_heads, seq_len, d_k)
            mask: (batch, 1, seq_len_q, seq_len_k)
        """
        d_k = Q.size(-1)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights

# 테스트
if __name__ == "__main__":
    batch_size = 2
    n_heads = 8
    seq_len = 10
    d_k = 64

    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)

    # 함수형
    output, attn = scaled_dot_product_attention_torch(Q, K, V)
    print(f"Output shape: {output.shape}")
    print(f"Attention shape: {attn.shape}")

    # 모듈형
    attention = ScaledDotProductAttention(dropout=0.1)
    output, attn = attention(Q, K, V)
    print(f"\nModule output shape: {output.shape}")

    # PyTorch 2.0+ built-in (더 빠름!)
    if hasattr(F, 'scaled_dot_product_attention'):
        output_builtin = F.scaled_dot_product_attention(Q, K, V)
        print(f"\nBuilt-in output shape: {output_builtin.shape}")
        print("✓ PyTorch 2.0+ built-in 사용 가능!")
```

### 💻 실습 4: Attention의 메모리 및 계산 복잡도 분석

```python
# examples/attention_complexity.py
import torch
import time
import matplotlib.pyplot as plt
from torch.profiler import profile, record_function, ProfilerActivity

def benchmark_attention_complexity():
    """
    Attention의 O(n^2) 복잡도 실험적 검증
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    seq_lengths = [128, 256, 512, 1024, 2048]
    d_model = 512
    n_heads = 8
    d_k = d_model // n_heads

    times = []
    memory = []

    for seq_len in seq_lengths:
        print(f"Testing seq_len={seq_len}...")

        Q = torch.randn(1, n_heads, seq_len, d_k, device=device)
        K = torch.randn(1, n_heads, seq_len, d_k, device=device)
        V = torch.randn(1, n_heads, seq_len, d_k, device=device)

        # Warmup
        for _ in range(3):
            _ = F.scaled_dot_product_attention(Q, K, V)

        # Timing
        if device == 'cuda':
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            output = F.scaled_dot_product_attention(Q, K, V)
        if device == 'cuda':
            torch.cuda.synchronize()

        elapsed = (time.time() - start) / 100
        times.append(elapsed * 1000)  # ms

        # Memory
        if device == 'cuda':
            mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            memory.append(mem)
            torch.cuda.reset_peak_memory_stats()

    # 결과 출력
    print("\n=== 벤치마크 결과 ===")
    print(f"{'Seq Length':<12} {'Time (ms)':<12} {'Memory (MB)':<15}")
    print("-" * 40)
    for seq, t, m in zip(seq_lengths, times, memory if memory else [0]*len(times)):
        print(f"{seq:<12} {t:<12.3f} {m:<15.1f}")

    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time complexity
    ax1.plot(seq_lengths, times, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Time Complexity of Attention')
    ax1.grid(True, alpha=0.3)

    # Expected O(n^2)
    theoretical = [(s/seq_lengths[0])**2 * times[0] for s in seq_lengths]
    ax1.plot(seq_lengths, theoretical, '--', alpha=0.5, label='O(n²) reference')
    ax1.legend()

    # Memory complexity
    if memory:
        ax2.plot(seq_lengths, memory, 'o-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('attention_complexity.png', dpi=150)
    print("\n✓ 복잡도 분석 시각화 저장: attention_complexity.png")

if __name__ == "__main__":
    benchmark_attention_complexity()
```

---

## 3. Multi-Head Attention

### 📖 이론

**Multi-Head Attention**은 여러 개의 attention을 병렬로 수행합니다:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

**장점:**
- 다양한 representation subspaces에 주목
- 다른 위치의 정보를 다양하게 집계

### 💻 실습 5: Multi-Head Attention 완전 구현

```python
# examples/multi_head_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 완전 구현
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Args:
            d_model: 모델의 차원
            n_heads: Head 개수
            dropout: Dropout 비율
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 각 head의 차원

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """
        (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: (batch, 1, seq_len_q, seq_len_k)

        Returns:
            output: (batch, seq_len_q, d_model)
            attention_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # (batch, seq_len_q, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # Split into heads
        Q = self.split_heads(Q, batch_size)  # (batch, n_heads, seq_len_q, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum
        context = torch.matmul(attention_weights, V)  # (batch, n_heads, seq_len_q, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous()  # (batch, seq_len_q, n_heads, d_k)
        context = context.view(batch_size, -1, self.d_model)  # (batch, seq_len_q, d_model)

        # Output projection
        output = self.W_o(context)

        return output, attention_weights

# 테스트
if __name__ == "__main__":
    batch_size = 2
    seq_len = 10
    d_model = 512
    n_heads = 8

    # 모델 생성
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.1)

    # 입력
    x = torch.randn(batch_size, seq_len, d_model)

    # Self-attention
    output, attn_weights = mha(x, x, x)

    print("=== Multi-Head Attention 테스트 ===")
    print(f"입력 shape: {x.shape}")
    print(f"출력 shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    # 파라미터 수 계산
    total_params = sum(p.numel() for p in mha.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")
    print(f"  W_q, W_k, W_v: {3 * d_model * d_model:,}")
    print(f"  W_o: {d_model * d_model:,}")
```

---

## 4. Masking과 Attention Patterns

### 📖 이론

**Masking**은 특정 위치에 attention이 가지 못하도록 합니다:

1. **Padding Mask**: 패딩 토큰 무시
2. **Causal Mask**: 미래 토큰 참조 방지 (디코더)

### 💻 실습 6: Masking 구현

```python
# examples/attention_masking.py
import torch
import matplotlib.pyplot as plt

def create_padding_mask(seq, pad_token=0):
    """
    Padding mask 생성

    Args:
        seq: (batch, seq_len)
        pad_token: 패딩 토큰 ID

    Returns:
        mask: (batch, 1, 1, seq_len)
    """
    # 패딩이 아닌 곳은 1, 패딩은 0
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask

def create_causal_mask(seq_len):
    """
    Causal (lookahead) mask 생성

    Returns:
        mask: (1, 1, seq_len, seq_len)
    """
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)

def visualize_masks():
    """
    다양한 mask 시각화
    """
    seq_len = 8

    # 1. Causal mask
    causal_mask = create_causal_mask(seq_len)[0, 0]

    # 2. Padding mask 예시
    seq = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])  # 마지막 3개가 패딩
    padding_mask = create_padding_mask(seq)[0, 0, 0]

    # 3. Combined mask (causal + padding)
    combined_mask = causal_mask * padding_mask

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    masks = [
        (causal_mask, "Causal Mask\n(prevents looking ahead)"),
        (padding_mask, "Padding Mask\n(ignores padding tokens)"),
        (combined_mask, "Combined Mask")
    ]

    for ax, (mask, title) in zip(axes, masks):
        im = ax.imshow(mask, cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('attention_masks.png', dpi=150)
    print("✓ Mask 시각화 저장: attention_masks.png")

if __name__ == "__main__":
    visualize_masks()

    # 실제 attention 계산에 적용
    batch_size = 2
    n_heads = 4
    seq_len = 8
    d_k = 64

    Q = torch.randn(batch_size, n_heads, seq_len, d_k)
    K = torch.randn(batch_size, n_heads, seq_len, d_k)
    V = torch.randn(batch_size, n_heads, seq_len, d_k)

    # Causal mask 적용
    mask = create_causal_mask(seq_len)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    scores_masked = scores.masked_fill(mask == 0, -1e9)

    print("\n=== Masking 효과 ===")
    print("Masked scores (첫 번째 배치, 첫 번째 head):")
    print(scores_masked[0, 0])

    # Softmax 적용
    attn_weights = F.softmax(scores_masked, dim=-1)
    print("\nAttention weights (causal mask 적용 후):")
    print(attn_weights[0, 0])
```

---

## ✅ Day 1-2 완료 체크리스트

### 이론 이해
- [ ] Attention의 Q, K, V 개념 명확히 이해
- [ ] Scaling factor √d_k의 필요성 이해
- [ ] Multi-Head Attention의 장점 이해
- [ ] Masking의 종류와 용도 이해

### 실습 완료
- [ ] NumPy로 attention 밑바닥 구현
- [ ] PyTorch로 효율적 구현
- [ ] Multi-Head Attention 완전 구현
- [ ] Masking 구현 및 시각화
- [ ] 계산 복잡도 실험적 검증

### 다음 단계
[Day 3-4: Positional Encoding](./02-positional-encoding.md)로 진행!
