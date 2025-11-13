# Context Length Extension Techniques

## 🎯 목표

**훈련된 모델을 더 긴 컨텍스트로 확장하기**

문제: Llama 2는 4K context로 훈련되었는데, 32K context에서 사용하고 싶다면?
- ❌ 처음부터 재훈련: 수백만 달러
- ✅ Context extension: 1% 미만 비용으로 가능!

---

## 📊 문제 정의

### RoPE의 한계

```python
# RoPE는 훈련 시 본 길이에 최적화됨
# Training: seq_len = 2048
rope = RoPE(d_k=128, max_seq_len=2048, theta=10000)

# Inference: seq_len = 4096 (out of distribution!)
# Position 2048+ 는 훈련 때 본 적 없음
# → 성능 급격히 저하 (perplexity 폭발)
```

**직관적 이해**:
```python
# RoPE frequency
freq_k = 1 / (theta ** (2k / d))

# Position p의 rotation angle
angle = p * freq_k

# Training: p ∈ [0, 2048]
# Inference: p ∈ [0, 4096]
# → 각도가 훈련 때 본 범위를 벗어남!
```

### Extrapolation Problem

```python
# Perplexity at different sequence lengths

Training length: 2048

Test at 2048: Perplexity = 5.2 ✓ Good
Test at 3072: Perplexity = 8.4 ⚠️ Degrading
Test at 4096: Perplexity = 15.7 ❌ Poor
Test at 8192: Perplexity = 127.3 ❌❌ Useless

# Model "forgets" how to process long sequences
```

---

## 🔧 Solution 1: Position Interpolation (PI)

### 핵심 아이디어

**"긴 sequence를 훈련 길이로 압축하자!"**

```python
# Original RoPE
position = 4096  # Out of distribution

# Position Interpolation
position_scaled = 4096 * (2048 / 4096)  # = 2048
# → 항상 훈련 범위 내!
```

### 수학적 원리

```python
# Original RoPE
θ_k = 1 / (10000 ^ (2k / d))
RoPE(x, p) = x * exp(i * p * θ_k)

# Position Interpolation
L_train = 2048  # Training context length
L_target = 4096  # Target context length
scale = L_train / L_target  # 0.5

RoPE_PI(x, p) = x * exp(i * (p * scale) * θ_k)
                = RoPE(x, p * scale)

# Position 4096 → scaled to 2048 (within training range!)
```

### 구현

```python
class RoPE_PI(nn.Module):
    """RoPE with Position Interpolation"""
    def __init__(self, dim, max_seq_len=2048, theta=10000, scale=1.0):
        super().__init__()
        self.scale = scale  # L_train / L_target

        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute scaled positions
        t = torch.arange(max_seq_len).type_as(self.inv_freq) * self.scale
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, seq_len):
        """
        Args:
            x: (batch, num_heads, seq_len, head_dim)
        Returns:
            x_rotated: rotated positions
        """
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        # Rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rot


# Usage: Extend 2K → 8K
rope_extended = RoPE_PI(
    dim=128,
    max_seq_len=8192,
    scale=2048 / 8192  # 0.25
)

# Now can handle 8K context!
x = torch.randn(1, 32, 8192, 128)
x_rot = rope_extended(x, seq_len=8192)
print(f"Extended to {x_rot.shape[2]} tokens")
```

### Fine-tuning

```python
def extend_context_with_pi(model, target_length, train_length=2048):
    """
    Extend model context with Position Interpolation

    Steps:
    1. Scale RoPE: scale = train_length / target_length
    2. Fine-tune on long sequences (1000-5000 steps)
    3. Use small learning rate (1e-5)
    """
    # Update RoPE scale in all layers
    scale = train_length / target_length
    for layer in model.layers:
        layer.attn.rope.scale = scale

    # Fine-tune
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for step, batch in enumerate(long_context_loader):
        # batch has sequences of length target_length
        loss = model(batch['input_ids'], labels=batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step >= 1000:  # Short fine-tuning!
            break

    return model


# Extend Llama 2 from 4K to 32K
model_extended = extend_context_with_pi(
    model=llama2_4k,
    target_length=32768,
    train_length=4096
)

# Result: 32K context with only 1000 steps!
```

### Results

```python
# Meta's findings (2023)

Llama 2 7B: 4K → 32K extension

Fine-tuning steps: 1,000
Training cost: < 0.1% of original pre-training

Performance:
  4K: Perplexity = 5.12 (baseline)
  8K: Perplexity = 5.18 (+1.2%)
  16K: Perplexity = 5.29 (+3.3%)
  32K: Perplexity = 5.47 (+6.8%)

→ Successfully extended with minimal degradation!
```

---

## 🌀 Solution 2: YaRN (Yet another RoPE extensioN)

### 문제: PI의 한계

```python
# Position Interpolation compresses ALL positions

Position 0 → 0 * 0.25 = 0 ✓
Position 100 → 100 * 0.25 = 25 (too compressed!)
Position 4000 → 4000 * 0.25 = 1000 ✓

# Low positions are over-compressed
# → Information loss for nearby tokens
# → Attention 패턴 왜곡
```

### YaRN's Idea

**"저주파수만 interpolate하고, 고주파수는 extrapolate"**

```python
# RoPE frequencies
θ_k = 1 / (10000 ^ (2k / d))

# Low k (high frequency): 빠르게 회전, 가까운 토큰 구별
# High k (low frequency): 느리게 회전, 먼 토큰 구별

# YaRN strategy:
# - High frequency (low k): No interpolation (local attention)
# - Low frequency (high k): Interpolation (long-range)
```

### 구현

```python
class RoPE_YaRN(nn.Module):
    """YaRN: Frequency-dependent interpolation"""
    def __init__(
        self,
        dim,
        max_seq_len=2048,
        theta=10000,
        scale=1.0,
        alpha=1.0,  # Extrapolation factor
        beta=32.0,  # Frequency boundary
    ):
        super().__init__()
        self.scale = scale
        self.alpha = alpha
        self.beta = beta

        # Compute frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

        # Frequency-dependent scaling
        # Low freq (high k): interpolate
        # High freq (low k): extrapolate
        freq_scale = torch.ones_like(inv_freq)

        # Compute scaling factors for each frequency
        for i, freq in enumerate(inv_freq):
            # Boundary frequency
            f_boundary = theta ** (beta / dim)

            if freq > f_boundary:
                # High frequency: no interpolation (or slight extrapolation)
                freq_scale[i] = self.alpha
            else:
                # Low frequency: interpolation
                # Smooth transition using sigmoid
                t = (torch.log(freq) - torch.log(torch.tensor(f_boundary))) / torch.log(torch.tensor(theta))
                freq_scale[i] = self.scale + (self.alpha - self.scale) * torch.sigmoid(10 * t)

        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("freq_scale", freq_scale)

        # Precompute embeddings
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        # Apply frequency-dependent scaling
        freqs = torch.outer(t, self.inv_freq * freq_scale)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, x, seq_len):
        cos = self.cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rot


# Usage
rope_yarn = RoPE_YaRN(
    dim=128,
    max_seq_len=32768,
    scale=2048 / 32768,  # 16x extension
    alpha=1.0,
    beta=32.0
)
```

### YaRN vs PI Comparison

```python
# Benchmark: Llama 2 7B, 4K → 32K

Method: Position Interpolation (PI)
  Perplexity at 32K: 5.47
  Pass@1 (code): 23.1%
  MMLU: 44.2%

Method: YaRN
  Perplexity at 32K: 5.31 (-3%)
  Pass@1 (code): 25.8% (+11.7%)
  MMLU: 45.9% (+3.8%)

→ YaRN consistently better!
```

---

## 🔬 Solution 3: NTK-Aware Scaling

### Neural Tangent Kernel (NTK) Theory

**핵심 통찰**: RoPE의 theta를 조정하면 frequency를 변경할 수 있음

```python
# Original RoPE
θ = 10000
freq_k = 1 / (θ ^ (2k / d))

# NTK-Aware
θ_new = θ * scale_factor
freq_k_new = 1 / (θ_new ^ (2k / d))
# → 모든 frequency가 함께 조정됨
```

### 구현

```python
def ntk_aware_rope(dim, max_seq_len, theta_base=10000, train_len=2048, target_len=8192):
    """
    NTK-Aware RoPE Scaling

    Dynamically adjust theta based on target length
    """
    # Compute scaling factor
    scale = target_len / train_len  # 4.0

    # NTK-Aware: adjust theta
    # theta_new = theta_base * scale ^ (d / (d - 2))
    alpha = dim / (dim - 2)  # For d=128: 128/126 ≈ 1.016
    theta_new = theta_base * (scale ** alpha)

    print(f"Original theta: {theta_base}")
    print(f"NTK theta: {theta_new:.2f}")
    print(f"Scale factor: {scale}")

    # Use new theta for RoPE
    inv_freq = 1.0 / (theta_new ** (torch.arange(0, dim, 2).float() / dim))

    return inv_freq


# Example: 2K → 8K
inv_freq_ntk = ntk_aware_rope(
    dim=128,
    max_seq_len=8192,
    train_len=2048,
    target_len=8192
)

# Output:
# Original theta: 10000
# NTK theta: 40634.92
# Scale factor: 4.0
```

### Dynamic NTK

```python
class DynamicNTKRoPE(nn.Module):
    """Dynamically adjust theta based on sequence length"""
    def __init__(self, dim, base_seq_len=2048, theta_base=10000):
        super().__init__()
        self.dim = dim
        self.base_seq_len = base_seq_len
        self.theta_base = theta_base
        self.alpha = dim / (dim - 2)

    def forward(self, x, seq_len):
        # Dynamically compute theta for this sequence length
        if seq_len > self.base_seq_len:
            scale = seq_len / self.base_seq_len
            theta = self.theta_base * (scale ** self.alpha)
        else:
            theta = self.theta_base

        # Compute frequencies with new theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))

        # Compute embeddings
        t = torch.arange(seq_len, device=x.device).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)

        # Rotate
        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rot = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rot


# Usage: Automatic adjustment
rope_dynamic = DynamicNTKRoPE(dim=128, base_seq_len=2048)

# Automatically handles any sequence length!
x_short = torch.randn(1, 32, 1000, 128)
x_long = torch.randn(1, 32, 16000, 128)

out_short = rope_dynamic(x_short, seq_len=1000)   # theta = 10000
out_long = rope_dynamic(x_long, seq_len=16000)    # theta = auto-adjusted

print(f"Short: {out_short.shape}")  # Handles both!
print(f"Long: {out_long.shape}")
```

---

## 📊 Method Comparison

### Feature Matrix

| Method | Interpolate | Extrapolate | Frequency-Aware | Dynamic | Fine-tuning |
|--------|-------------|-------------|-----------------|---------|-------------|
| **PI** | ✓ All | ✗ | ✗ | ✗ | 1K steps |
| **YaRN** | ✓ Low freq | ✓ High freq | ✓ | ✗ | 400 steps |
| **NTK** | ✗ | ✓ All | ✓ | ✓ | 0 steps! |

### Performance Comparison

```python
# Llama 2 7B: 4K → 32K extension

Method: Naive (no extension)
  Perplexity: 127.3 ❌
  Works: No

Method: Position Interpolation
  Perplexity: 5.47
  Training: 1000 steps
  Works: Yes ✓

Method: YaRN
  Perplexity: 5.31 (best!)
  Training: 400 steps
  Works: Yes ✓

Method: NTK-Aware
  Perplexity: 5.89
  Training: 0 steps (zero-shot!)
  Works: Yes ✓

Method: Dynamic NTK
  Perplexity: 5.72
  Training: 0 steps
  Works: Yes ✓
  Bonus: Any length!
```

### Use Case Recommendations

```python
# Decision tree

if training_budget > 1000_steps:
    use YaRN  # Best quality
elif need_zero_shot:
    use Dynamic_NTK  # No fine-tuning
elif need_variable_length:
    use Dynamic_NTK  # Automatic adjustment
else:
    use PI  # Simple and effective
```

---

## 💻 Complete Example: Extending Llama 2

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Llama 2 7B (4K context)
model_id = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Original max length: {model.config.max_position_embeddings}")

# Method 1: Position Interpolation
def extend_with_pi(model, target_length):
    """Extend with Position Interpolation"""
    scale = model.config.max_position_embeddings / target_length

    # Update config
    model.config.max_position_embeddings = target_length

    # Update RoPE in all layers
    for layer in model.model.layers:
        # Modify rope scaling
        layer.self_attn.rotary_emb.scaling_factor = scale

    return model

model_extended = extend_with_pi(model, target_length=16384)
print(f"Extended max length: {model_extended.config.max_position_embeddings}")

# Method 2: YaRN (requires rope_scaling config)
model.config.rope_scaling = {
    "type": "yarn",
    "factor": 4.0,  # 4K → 16K
    "original_max_position_embeddings": 4096,
}

# Method 3: Dynamic NTK (automatic)
model.config.rope_scaling = {
    "type": "dynamic",
    "factor": 4.0,
}

# Test with long context
long_text = "Your very long document here..." * 1000
inputs = tokenizer(long_text, return_tensors="pt", truncation=False).to(model.device)

print(f"Input length: {inputs['input_ids'].shape[1]} tokens")

# Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
    )

print(tokenizer.decode(outputs[0]))
```

---

## 🎯 Practical Guidelines

### When to Use Each Method

**Position Interpolation (PI)**:
```python
# Use when:
- Fixed target length known in advance
- Have budget for 1000+ fine-tuning steps
- Need simple implementation
- Extending by 2-8x

# Example:
# Llama 2 4K → 32K for specific application
```

**YaRN**:
```python
# Use when:
- Need best quality
- Have budget for 400+ steps
- Extending by 4-16x
- Research/production hybrid

# Example:
# Building a long-context chatbot
```

**NTK-Aware**:
```python
# Use when:
- Zero-shot extension needed
- No fine-tuning budget
- Quick prototyping
- Moderate extension (2-4x)

# Example:
# Rapid deployment, A/B testing
```

**Dynamic NTK**:
```python
# Use when:
- Variable input lengths
- Need flexibility
- Zero-shot
- Real-time adjustment

# Example:
# API service with varying context lengths
```

### Implementation Checklist

```python
# Step 1: Choose method
method = "yarn"  # or "pi", "ntk", "dynamic_ntk"

# Step 2: Determine scale
train_length = 4096
target_length = 32768
scale = target_length / train_length  # 8x

# Step 3: Update model
if method == "pi":
    # Modify RoPE frequency scaling
    for layer in model.layers:
        layer.attn.rope.scale = 1.0 / scale

elif method == "yarn":
    # Frequency-dependent scaling
    model.config.rope_scaling = {
        "type": "yarn",
        "factor": scale,
        "original_max_position_embeddings": train_length,
    }

elif method in ["ntk", "dynamic_ntk"]:
    # Adjust theta
    model.config.rope_scaling = {
        "type": "dynamic",
        "factor": scale,
    }

# Step 4: Fine-tune (if needed)
if method in ["pi", "yarn"]:
    fine_tune_on_long_sequences(model, steps=1000)

# Step 5: Evaluate
evaluate_on_long_context(model, test_set)
```

---

## 📚 References

**Papers**:
1. **Position Interpolation** (Chen et al., 2023) - https://arxiv.org/abs/2306.15595
2. **YaRN** (Peng et al., 2023) - https://arxiv.org/abs/2309.00071
3. **NTK-Aware Scaling** (bloc97, 2023) - Reddit/GitHub discussion
4. **Long Context Llama** (Meta, 2023) - Technical report

**Code**:
- HuggingFace Transformers: `rope_scaling` parameter
- `scaled-rope` library
- LongChat implementation

**Benchmarks**:
- LongBench: Long-context understanding benchmark
- ZeroScrolls: Multi-task long-context evaluation

---

## 🎓 Exercises

### Exercise 1: Implement PI

Extend a small GPT model from 512 to 2048 context.

```python
# Your implementation
def extend_gpt(model, target_len=2048):
    # Modify RoPE
    pass

# Test
extended_model = extend_gpt(gpt_512)
test_long_sequence(extended_model, length=2048)
```

### Exercise 2: Compare Methods

Benchmark PI vs YaRN vs NTK on the same task.

```python
# Metrics to compare:
# - Perplexity
# - Fine-tuning cost
# - Inference speed
# - Memory usage
```

### Exercise 3: Dynamic Length Handling

Build a system that automatically adjusts to input length.

```python
class AdaptiveModel:
    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        # Automatically choose best method
        if seq_len <= train_len:
            return self.model(input_ids)
        elif seq_len <= train_len * 4:
            return self.model_ntk(input_ids)
        else:
            return self.model_yarn(input_ids)
```

---

## ⏭️ Next Steps

Long context 처리를 마스터했습니다!

👉 [Modern LLM Architectures](../phase2-bert-gpt/04-modern-llm-architectures.md) - 이제 long context를 활용하는 모델들
👉 [Phase 5: Efficient Methods](../phase5-modern-techniques/01-efficient-methods.md) - Long context의 메모리 최적화

**이제 4K 모델을 128K+로 확장할 수 있습니다!** 🚀
