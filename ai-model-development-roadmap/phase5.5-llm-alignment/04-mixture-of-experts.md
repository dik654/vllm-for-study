# Day 8-10: Mixture of Experts (MoE)

## 🎯 학습 목표

**"DeepSeek-V3는 왜 671B 파라미터인데 빠른가?"**

MoE (Mixture of Experts)의 핵심:
- ✅ **Sparse Activation**: 전체 파라미터 중 일부만 사용
- ✅ **Conditional Computation**: 입력에 따라 다른 expert 활성화
- ✅ **효율성**: 큰 모델의 성능 + 작은 모델의 속도

---

## 1. MoE의 핵심 아이디어

### 📖 직관

**일반 모델**: 모든 파라미터가 항상 활성화
```python
def ffn(x):
    # 모든 가중치 사용
    h = W1 @ x  # (d_ff, d_model) @ (d_model,)
    h = relu(h)
    output = W2 @ h  # (d_model, d_ff) @ (d_ff,)
    return output

# 계산량: 2 × d_model × d_ff
```

**MoE**: 필요한 expert만 선택적으로 활성화
```python
def moe(x, num_experts=8, top_k=2):
    # 1. Router: 어떤 expert를 사용할지 결정
    router_logits = router_network(x)  # (num_experts,)
    top_k_indices, top_k_weights = topk(router_logits, k=top_k)

    # 2. 선택된 expert만 실행
    output = 0
    for idx, weight in zip(top_k_indices, top_k_weights):
        output += weight * experts[idx](x)

    return output

# 계산량: (top_k / num_experts) × 기존 FFN
# 예: top_k=2, num_experts=8 → 25%만 사용!
```

### 💡 왜 효율적인가?

**DeepSeek-V3 예시:**
```
전체 파라미터: 671B
활성화 파라미터: 37B (5.5%)

→ 671B 모델의 성능
→ 37B 모델의 속도와 메모리
```

---

## 2. MoE 아키텍처 구성 요소

### 2.1 Router (Gating Network)

**역할**: 각 입력에 대해 어떤 expert를 사용할지 결정

```python
# router.py
import torch
import torch.nn as nn

class Router(nn.Module):
    """
    Router: input → expert selection
    """
    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        # Simple linear layer
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x, top_k=2):
        """
        Args:
            x: (batch, seq_len, d_model)
            top_k: 선택할 expert 개수

        Returns:
            indices: (batch, seq_len, top_k)
            weights: (batch, seq_len, top_k)
        """
        # Gate logits
        gate_logits = self.gate(x)  # (batch, seq_len, num_experts)

        # Softmax (각 토큰마다)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-K selection
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=top_k, dim=-1)

        # Normalize (top-k의 합이 1이 되도록)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        return top_k_indices, top_k_probs
```

### 2.2 Expert Networks

**일반적으로 FFN (Feed-Forward Network)**

```python
# expert.py
class Expert(nn.Module):
    """
    단일 Expert: FFN
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        h = self.activation(self.w1(x))
        output = self.w2(h)
        return output
```

### 2.3 MoE Layer

**전체 MoE 레이어**

```python
# moe_layer.py
class MoELayer(nn.Module):
    """
    Complete MoE Layer
    """
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = Router(d_model, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # 1. Routing
        expert_indices, expert_weights = self.router(x, top_k=self.top_k)
        # expert_indices: (batch, seq_len, top_k)
        # expert_weights: (batch, seq_len, top_k)

        # 2. Expert computation
        # Reshape for efficient batching
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (expert_indices == expert_idx).any(dim=-1)  # (batch, seq_len)
            mask_flat = mask.view(-1)  # (batch * seq_len,)

            if mask_flat.any():
                # Extract tokens for this expert
                expert_input = x_flat[mask_flat]  # (num_tokens, d_model)

                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)

                # Get weights for this expert
                # (복잡하지만 개념은 간단: weight를 곱해서 더함)
                expert_weight = expert_weights[..., expert_idx][mask]

                # Accumulate to output
                output[mask_flat] += expert_weight.unsqueeze(-1) * expert_output

        # Reshape back
        output = output.view(batch_size, seq_len, d_model)

        return output
```

---

## 3. Load Balancing

### 📖 문제: Expert Collapse

**문제**: 일부 expert만 계속 사용되고 나머지는 놀게 됨

```python
# 나쁜 예
Expert 0: 80%의 토큰 처리  ← 과부하
Expert 1: 15%
Expert 2-7: 5% (나머지) ← 거의 안 씀
```

**결과**:
- ❌ 병렬화 이점 사라짐 (1개만 일함)
- ❌ 용량 낭비 (대부분 expert 안 씀)
- ❌ 성능 저하

### 💡 해결책: Load Balancing Loss

**목표**: 모든 expert가 균등하게 사용되도록

```python
# load_balancing.py
def load_balancing_loss(router_probs, num_experts):
    """
    Load balancing auxiliary loss

    Args:
        router_probs: (batch, seq_len, num_experts)
    Returns:
        loss: scalar
    """
    # 각 expert가 처리한 토큰 비율
    fraction_per_expert = router_probs.mean(dim=[0, 1])  # (num_experts,)

    # 이상적으로는 모두 1/num_experts
    ideal_fraction = 1.0 / num_experts

    # L2 loss (균등 분포에서 벗어난 정도)
    loss = ((fraction_per_expert - ideal_fraction) ** 2).sum()

    return loss

# 전체 loss에 추가
total_loss = task_loss + alpha * load_balancing_loss(router_probs)
# alpha는 보통 0.01 정도
```

### 💻 실습 1: Load Balancing 시각화

```python
# visualize_load_balancing.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_expert_usage(router_probs, num_experts=8):
    """
    Expert 사용 분포 시각화

    Args:
        router_probs: (batch, seq_len, num_experts)
    """
    # 각 expert의 평균 사용률
    usage = router_probs.mean(dim=[0, 1]).cpu().numpy()

    # 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_experts), usage)
    plt.axhline(y=1.0/num_experts, color='r', linestyle='--',
                label=f'Ideal ({1.0/num_experts:.3f})')
    plt.xlabel('Expert Index')
    plt.ylabel('Usage Fraction')
    plt.title('Expert Load Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 불균형 정도 (coefficient of variation)
    cv = np.std(usage) / np.mean(usage)
    plt.text(0.7, 0.9, f'CV: {cv:.3f}', transform=plt.gca().transAxes)

    plt.savefig('expert_load_balance.png', dpi=150)
    print(f"Expert usage: {usage}")
    print(f"Std dev: {np.std(usage):.4f}")
    print(f"✓ 시각화 저장: expert_load_balance.png")

# 테스트
num_experts = 8
batch, seq_len = 4, 100

# Bad case (불균형)
bad_probs = torch.zeros(batch, seq_len, num_experts)
bad_probs[..., 0] = 0.8  # Expert 0에 집중
bad_probs[..., 1:] = 0.2 / (num_experts - 1)

visualize_expert_usage(bad_probs, num_experts)
```

---

## 4. 실전 MoE 모델들

### 4.1 Switch Transformer (Google, 2021)

**특징**:
- **Switch Routing**: Top-1 (1개 expert만 선택)
- 1.6T 파라미터
- 간단하고 효과적

```python
# switch_transformer.py
class SwitchMoELayer(nn.Module):
    """
    Switch Transformer: Top-1 routing
    """
    def __init__(self, d_model, d_ff, num_experts):
        super().__init__()
        self.router = Router(d_model, num_experts)
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])
        self.num_experts = num_experts

    def forward(self, x):
        # Top-1 routing
        expert_idx, expert_weight = self.router(x, top_k=1)

        # Dispatch to experts
        output = torch.zeros_like(x)
        for i in range(self.num_experts):
            mask = (expert_idx.squeeze(-1) == i)
            if mask.any():
                expert_input = x[mask]
                expert_output = self.experts[i](expert_input)
                output[mask] = expert_output

        return output
```

### 4.2 Mixtral 8x7B (Mistral AI, 2023)

**특징**:
- **8개 expert**, 각 7B
- **Top-2 routing**
- 46.7B 총 파라미터, 12.9B 활성화

```python
# mixtral_style.py
class MixtralMoE(nn.Module):
    """
    Mixtral: Top-2 routing, 8 experts
    """
    def __init__(self, d_model=4096, d_ff=14336, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = 2  # Mixtral은 Top-2

        # Router (single linear layer)
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # 8 Experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff) for _ in range(num_experts)
        ])

    def forward(self, x):
        # Gate logits
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-2 selection
        top_k_probs, top_k_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute output
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            for expert_idx in range(self.num_experts):
                mask = (top_k_indices[..., k] == expert_idx)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    weight = top_k_probs[..., k][mask].unsqueeze(-1)
                    output[mask] += weight * expert_output

        return output
```

### 4.3 DeepSeek-V3 (2024)

**특징**:
- **671B 파라미터**, 37B 활성화
- **Multi-head Latent Attention** (MLA)
- **Fine-grained Expert Segmentation**
- **Auxiliary-loss-free Load Balancing**

**핵심 개선사항**:

1. **Expert Segmentation**: Expert를 더 작은 단위로 분할
```python
# DeepSeek-V3: 256 experts (very fine-grained!)
num_experts = 256
expert_size = smaller  # 각 expert가 작음
```

2. **Load Balancing without Auxiliary Loss**
```python
# 기존: auxiliary loss 추가
loss = task_loss + alpha * load_balance_loss

# DeepSeek-V3: Router에 bias term 추가
# 사용 적은 expert에게 bonus
expert_usage_count = count_usage(router_history)
bias = -log(expert_usage_count + epsilon)  # 적게 쓰인 expert일수록 큰 bias
gate_logits = gate_logits + bias
```

---

## 5. MoE 훈련 시 주의사항

### ⚠️ 도전과제

1. **통신 오버헤드**
   - Expert들이 다른 GPU에 분산
   - Token routing으로 GPU 간 통신 필요
   - All-to-all communication 병목

2. **불균형 Batch**
   - 일부 expert에 토큰 몰림
   - GPU 활용률 불균형

3. **Training Instability**
   - Router 학습이 불안정할 수 있음
   - Expert collapse (일부만 사용)

### ✅ 해결 방법

```python
# 1. Capacity Factor: expert당 최대 토큰 수 제한
capacity = (seq_len * batch_size / num_experts) * capacity_factor
# capacity_factor > 1.0 (여유 공간)

# 2. Dropout on Router
router_logits = F.dropout(router_logits, p=0.1, training=True)

# 3. Expert Dropout
# 랜덤하게 일부 expert 제거 (정규화)

# 4. Soft Router (deterministic + stochastic)
if training:
    # Stochastic: 다양한 expert 탐색
    expert_idx = torch.multinomial(gate_probs, num_samples=top_k)
else:
    # Deterministic: Top-K
    expert_idx = torch.topk(gate_probs, k=top_k)[1]
```

---

## 6. 실습: Mini MoE 구현

### 💻 실습 2: 완전한 MoE Transformer Block

```python
# moe_transformer_block.py
class MoETransformerBlock(nn.Module):
    """
    Transformer block with MoE FFN
    """
    def __init__(self, d_model, num_heads, d_ff, num_experts=8, top_k=2):
        super().__init__()

        # Multi-Head Self-Attention (일반)
        self.attention = nn.MultiheadAttention(d_model, num_heads)

        # MoE FFN (여기가 다름!)
        self.moe = MoELayer(d_model, d_ff, num_experts, top_k)

        # Layer Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        # Self-Attention
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)

        # MoE FFN
        moe_out = self.moe(x)
        x = self.norm2(x + moe_out)

        return x

# 테스트
if __name__ == "__main__":
    batch, seq_len, d_model = 2, 10, 512
    num_heads = 8
    d_ff = 2048
    num_experts = 8
    top_k = 2

    block = MoETransformerBlock(d_model, num_heads, d_ff, num_experts, top_k)

    x = torch.randn(batch, seq_len, d_model)
    output = block(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ MoE Transformer Block 동작 확인!")

    # 파라미터 수 비교
    dense_params = 2 * d_model * d_ff  # W1, W2
    moe_params = num_experts * dense_params

    print(f"\n파라미터 비교:")
    print(f"Dense FFN: {dense_params / 1e6:.1f}M")
    print(f"MoE ({num_experts} experts): {moe_params / 1e6:.1f}M")
    print(f"활성화 (Top-{top_k}): {top_k/num_experts * 100:.1f}% = {top_k * dense_params / 1e6:.1f}M")
```

### 💻 실습 3: MoE vs Dense 성능 비교

```python
# compare_moe_dense.py
import time
import torch

def benchmark_ffn(model, x, num_runs=100):
    """FFN inference 속도 측정"""
    model.eval()

    # Warmup
    for _ in range(10):
        _ = model(x)

    # Timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_runs):
        _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start
    return elapsed / num_runs

# Setup
batch, seq_len, d_model = 4, 128, 512
d_ff = 2048
num_experts = 8
top_k = 2

x = torch.randn(batch, seq_len, d_model)
if torch.cuda.is_available():
    x = x.cuda()

# Dense FFN
dense_ffn = Expert(d_model, d_ff)
if torch.cuda.is_available():
    dense_ffn = dense_ffn.cuda()

# MoE
moe_ffn = MoELayer(d_model, d_ff, num_experts, top_k)
if torch.cuda.is_available():
    moe_ffn = moe_ffn.cuda()

# Benchmark
dense_time = benchmark_ffn(dense_ffn, x)
moe_time = benchmark_ffn(moe_ffn, x)

print("=== 성능 비교 ===")
print(f"Dense FFN: {dense_time*1000:.2f} ms")
print(f"MoE FFN: {moe_time*1000:.2f} ms")
print(f"Speedup: {dense_time/moe_time:.2f}x")

# 파라미터 효율
dense_params = sum(p.numel() for p in dense_ffn.parameters())
moe_params = sum(p.numel() for p in moe_ffn.parameters())
moe_active = moe_params * (top_k / num_experts)

print(f"\nDense 파라미터: {dense_params/1e6:.1f}M")
print(f"MoE 파라미터: {moe_params/1e6:.1f}M")
print(f"MoE 활성화: {moe_active/1e6:.1f}M ({top_k/num_experts*100:.1f}%)")
```

---

## ✅ Day 8-10 완료 체크리스트

### 이론 이해
- [ ] MoE의 sparse activation 원리 이해
- [ ] Router의 역할 이해
- [ ] Load balancing의 필요성 이해
- [ ] Switch, Mixtral, DeepSeek-V3 차이 이해

### 실습 완료
- [ ] Router 구현
- [ ] MoE Layer 구현
- [ ] Load balancing loss 구현
- [ ] MoE Transformer block 구현
- [ ] MoE vs Dense 성능 비교

### AI 연결
- [ ] DeepSeek-V3의 효율성 비밀 설명 가능
- [ ] Mixtral 8x7B 아키텍처 이해
- [ ] MoE의 장단점 분석 가능

---

## 🎯 최종 챌린지

**Mini Mixtral 구현!**

- 4개 expert
- Top-2 routing
- Load balancing
- 간단한 언어 모델링 태스크 훈련
- Dense baseline과 성능/속도 비교

성공하면, **최신 SOTA 모델의 핵심을 마스터**했습니다! 🎉

---

## ⏭️ 다음 단계

👉 [Day 11-12: Constitutional AI](./05-constitutional-ai.md)

**"이제 DeepSeek-V3를 완전히 이해합니다!"** 🚀
