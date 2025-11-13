# State Space Models (SSMs): Mamba and Beyond

## 🎯 목표

**Transformer의 대안: State Space Models 이해하기**

문제: Attention은 O(n²) 복잡도 → Long sequences에서 비효율적
해결: State Space Models (S4, Mamba) - O(n) 복잡도!

---

## 📊 Transformer의 한계

### Quadratic Complexity

```python
# Multi-Head Attention
Q, K, V = x @ W_q, x @ W_k, x @ W_v  # (seq_len, d_model)

# Attention scores
scores = Q @ K.T  # (seq_len, seq_len) ← O(n²) memory!

# Memory usage:
# seq_len = 1K: 1M elements
# seq_len = 10K: 100M elements (100x!)
# seq_len = 100K: 10B elements (10,000x!)

# Impractical for long sequences!
```

### Alternatives

```python
# Attempted solutions:

# 1. Sparse Attention (Longformer, BigBird)
# → Still O(n²) asymptotically

# 2. Linear Attention (Performers, FNet)
# → O(n), but quality loss

# 3. Sliding Window (Mistral)
# → Fixed context size, not true long-range

# 4. State Space Models (S4, Mamba)
# → O(n) AND good quality! ✓
```

---

## 🌊 State Space Models: Basics

### Classical State Space

```python
# Continuous-time system

# State equation:
h'(t) = A h(t) + B u(t)

# Output equation:
y(t) = C h(t) + D u(t)

where:
  h(t): hidden state (n-dimensional)
  u(t): input
  y(t): output
  A, B, C, D: learned parameters
```

### Discretization

```python
# Neural networks work with discrete time steps

# Discretize using step size Δ:
h_t = A_bar h_{t-1} + B_bar u_t
y_t = C h_t

where:
  A_bar = exp(Δ A)  # Matrix exponential
  B_bar = (Δ A)^{-1} (exp(Δ A) - I) Δ B

# Now we can process sequences!
```

### Recurrent vs. Convolutional View

```python
# Recurrent view (inference):
for t in range(seq_len):
    h_t = A_bar @ h_{t-1} + B_bar @ u_t
    y_t = C @ h_t

# Convolutional view (training):
# Can compute entire sequence in parallel!
K = compute_ssm_kernel(A_bar, B_bar, C, seq_len)
y = convolve(u, K)

# Training: O(n log n) with FFT
# Inference: O(n) recurrent
```

---

## 🔬 S4 (Structured State Space)

### Key Innovation: HiPPO Matrix

```python
# Problem: Naively learned A doesn't work
# Solution: HiPPO (High-order Polynomial Projection Operators)

# HiPPO provides structured initialization for A
# → Provably good for memorizing long sequences

def hippo_matrix(N):
    """
    HiPPO-LegS matrix

    Designed to remember past optimally
    """
    A = np.zeros((N, N))

    for n in range(N):
        for k in range(N):
            if n > k:
                A[n, k] = -(2*n + 1) ** 0.5 * (2*k + 1) ** 0.5
            elif n == k:
                A[n, k] = -n - 1
            else:
                A[n, k] = 0

    return A


# Key property: Compresses history efficiently
# → Can remember information from 1000s of steps ago!
```

### S4 Architecture

```python
import torch
import torch.nn as nn

class S4Layer(nn.Module):
    """
    S4 (Structured State Space) Layer

    Paper: https://arxiv.org/abs/2111.00396
    """
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # Initialize A with HiPPO
        A = self.hippo_matrix(d_state)
        self.A = nn.Parameter(torch.tensor(A, dtype=torch.float32))

        # B, C are learned
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))

        # Step size
        self.log_dt = nn.Parameter(torch.randn(d_model))

    def hippo_matrix(self, N):
        # Simplified HiPPO initialization
        A = torch.zeros(N, N)
        for n in range(N):
            A[n, :n+1] = -(2*torch.arange(n+1) + 1).sqrt() * (2*n + 1).sqrt()
            A[n, n] = -n - 1
        return A

    def forward(self, u):
        """
        Args:
            u: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = u.shape

        # Discretize
        dt = torch.exp(self.log_dt)  # (d_model,)
        A_bar = torch.matrix_exp(dt.unsqueeze(-1) * self.A)  # (d_model, d_state, d_state)
        B_bar = torch.bmm(
            torch.inverse(dt.unsqueeze(-1) * self.A) @
            (torch.matrix_exp(dt.unsqueeze(-1) * self.A) - torch.eye(self.d_state)),
            (dt * self.B).unsqueeze(-1)
        )  # (d_model, d_state, 1)

        # Recurrent computation (inference)
        h = torch.zeros(batch, self.d_state, d_model, device=u.device)
        outputs = []

        for t in range(seq_len):
            # h_t = A_bar h_{t-1} + B_bar u_t
            h = torch.einsum('bsd,dse->bse', h, A_bar.transpose(-2, -1)) + \
                torch.einsum('bsd,dse->bse', u[:, t:t+1, :], B_bar.transpose(0, 1))

            # y_t = C h_t
            y_t = torch.einsum('bsd,ds->bd', h, self.C)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y


# Use in model
class S4Model(nn.Module):
    def __init__(self, d_model=256, d_state=64, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            S4Layer(d_model, d_state)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(self.norm(x))  # Residual
        return x
```

### S4 Performance

```python
# Long Range Arena Benchmark (sequences up to 16K tokens)

Model: Transformer (full attention)
  Accuracy: 59.1%
  Speed: Baseline (1x)
  Max length: 4K

Model: Longformer (sparse attention)
  Accuracy: 56.3%
  Speed: 2x
  Max length: 16K

Model: S4
  Accuracy: 76.3% ⭐ (much better!)
  Speed: 60x ⭐⭐
  Max length: 1M+ ⭐⭐⭐

→ S4 is faster, more accurate, handles longer sequences!
```

---

## 🐍 Mamba: The State Space Language Model

### S4 → Mamba Evolution

```python
# S4 limitations:
# 1. Fixed A, B, C parameters for all inputs
# → Can't dynamically attend to relevant parts

# Mamba solution:
# 1. Input-dependent parameters (selective SSM)
# 2. Hardware-efficient implementation
# 3. Simplified architecture
```

### Selective SSM

```python
class SelectiveSSM(nn.Module):
    """
    Mamba's Selective State Space Model

    Key idea: Make B, C, Δ input-dependent!

    Paper: https://arxiv.org/abs/2312.00752
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A is fixed (HiPPO-inspired)
        A = -torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A))

        # Projections for input-dependent parameters
        self.x_proj = nn.Linear(d_model, d_state + d_state + d_model, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape

        # Compute input-dependent B, C, Δ
        x_proj = self.x_proj(x)  # (batch, seq_len, d_state*2 + d_model)

        B, C, delta = torch.split(
            x_proj,
            [self.d_state, self.d_state, d_model],
            dim=-1
        )

        # B, C: (batch, seq_len, d_state)
        # delta: (batch, seq_len, d_model)

        # Softplus for positivity
        delta = F.softplus(delta)

        # A is fixed
        A = -torch.exp(self.A_log.float())  # (d_state,)

        # Discretize with input-dependent delta
        # A_bar = exp(delta * A)
        A_bar = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_model, d_state)

        # B_bar = delta * B
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(2)  # (batch, seq_len, d_model, d_state)

        # Recurrent scan (parallel associative scan)
        y = self.selective_scan(x, A_bar, B_bar, C)

        return self.out_proj(y)

    def selective_scan(self, x, A_bar, B_bar, C):
        """
        Parallel scan algorithm (hardware-efficient)

        Instead of sequential:
          for t in range(T):
              h_t = A_bar_t * h_{t-1} + B_bar_t * x_t

        Use parallel prefix sum!
        """
        # Simplified version (actual implementation uses custom CUDA kernel)
        batch, seq_len, d_model, d_state = A_bar.shape

        h = torch.zeros(batch, d_state, d_model, device=x.device)
        outputs = []

        for t in range(seq_len):
            # h = A * h + B * x
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t:t+1].unsqueeze(-1)

            # y = C * h
            y_t = torch.einsum('bds,bs->bd', h, C[:, t])
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


# Mamba block (combines SSM with gating)
class MambaBlock(nn.Module):
    """Complete Mamba block"""
    def __init__(self, d_model, d_state=16, expand_factor=2):
        super().__init__()
        self.d_inner = d_model * expand_factor

        # Projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # SSM
        self.ssm = SelectiveSSM(self.d_inner, d_state)

        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)

        # Split into two branches (like GLU)
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)

        # SSM branch
        x_ssm = self.ssm(x_ssm)

        # Gate branch (SiLU activation)
        x_gate = F.silu(x_gate)

        # Combine
        x = x_ssm * x_gate

        # Output projection
        x = self.out_proj(x)

        return x + residual
```

### Mamba Architecture

```python
class Mamba(nn.Module):
    """
    Mamba Language Model

    Alternative to Transformer with linear complexity
    """
    def __init__(
        self,
        vocab_size=50000,
        d_model=768,
        d_state=16,
        num_layers=24,
    ):
        super().__init__()

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state)
            for _ in range(num_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        x = self.embedding(input_ids)

        # Through Mamba blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


# Test
model = Mamba(vocab_size=50000, d_model=768, num_layers=24)
input_ids = torch.randint(0, 50000, (2, 1024))
logits = model(input_ids)

print(f"Input: {input_ids.shape}")
print(f"Output: {logits.shape}")

# Input: torch.Size([2, 1024])
# Output: torch.Size([2, 1024, 50000])
```

---

## 📊 Mamba vs Transformer

### Complexity Comparison

```python
# Sequence length: n
# Model dimension: d
# State dimension: s (typically 16-64)

# Transformer:
# - Memory: O(n²)
# - Compute: O(n² d)
# - Can't handle n > 10K efficiently

# Mamba:
# - Memory: O(n)
# - Compute: O(n d s)
# - Can handle n = 1M+!

# Example: n=100K, d=1024, s=16
transformer_memory = 100_000 ** 2  # 10B
mamba_memory = 100_000 * 1024 * 16  # 1.6M
ratio = transformer_memory / mamba_memory
print(f"Memory reduction: {ratio:.0f}x")  # ~6250x!
```

### Performance Benchmarks

```python
# Mamba 2.8B vs Transformer 2.7B

Task: Language Modeling (wikitext)
  Transformer: PPL = 12.7
  Mamba: PPL = 11.3 ✓ (better!)

Task: Long-context QA (16K tokens)
  Transformer: 48.2% accuracy (with tricks)
  Mamba: 67.1% accuracy ✓✓

Task: Inference speed (RTX 4090)
  Transformer (seq_len=2K): 140 tokens/sec
  Mamba (seq_len=2K): 280 tokens/sec (2x faster)
  Mamba (seq_len=100K): 180 tokens/sec (still fast!)
  Transformer (seq_len=100K): OOM ❌

# Mamba is faster, better on long sequences!
```

---

## 🔀 Hybrid Architectures

### Best of Both Worlds

```python
# Idea: Combine Attention (global) + SSM (efficient local)

class HybridBlock(nn.Module):
    """
    Hybrid: Attention for some layers, SSM for others

    Example: Jamba (AI21 Labs)
    """
    def __init__(self, d_model, use_attention=True):
        super().__init__()
        if use_attention:
            self.layer = TransformerBlock(d_model)
        else:
            self.layer = MambaBlock(d_model)

    def forward(self, x):
        return self.layer(x)


class HybridModel(nn.Module):
    """
    Hybrid model: Attention every N layers, SSM otherwise

    Jamba pattern: A-A-M-M-M-M-A-A-M-M-M-M...
    """
    def __init__(self, d_model=1024, num_layers=32, attention_every=8):
        super().__init__()
        self.layers = nn.ModuleList([
            HybridBlock(
                d_model,
                use_attention=(i % attention_every == 0)
            )
            for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# Result: Most layers are efficient SSM, some attention for global info!
```

---

## 🎯 When to Use SSMs

### Decision Matrix

```python
# Use Transformer when:
# - Sequence length < 4K
# - Need proven architecture
# - Rich ecosystem (HuggingFace)
# - Instruction following critical

# Use Mamba when:
# - Sequence length > 10K
# - Memory constrained
# - Long-context understanding
# - Inference speed critical

# Use Hybrid (Jamba) when:
# - Need both global and local
# - Medium length (4K-32K)
# - Want best of both
```

### Current Limitations

```python
# Mamba challenges:

# 1. Ecosystem
# → Fewer pre-trained models
# → Less tooling support

# 2. Training
# → Need custom CUDA kernels for speed
# → Harder to optimize

# 3. Understanding
# → Attention patterns are interpretable
# → SSM hidden states are opaque

# 4. Transfer learning
# → Transformer checkpoints everywhere
# → Mamba just starting

# Future: Likely to improve rapidly!
```

---

## 💻 Try Mamba

### Using Pre-trained Models

```python
# HuggingFace has Mamba support

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Mamba model
model = AutoModelForCausalLM.from_pretrained(
    "state-spaces/mamba-2.8b",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

# Generate
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7
)

print(tokenizer.decode(outputs[0]))
```

### Training from Scratch

```python
# Train small Mamba on TinyStories

from datasets import load_dataset

# Load data
dataset = load_dataset("roneneldan/TinyStories")

# Create model
model = Mamba(
    vocab_size=50257,
    d_model=512,
    d_state=16,
    num_layers=12
)

# Train (similar to transformer training)
# ... training loop ...

# Mamba trains ~2x faster than equivalent Transformer!
```

---

## 📚 References

**Papers**:
1. **S4** (Gu et al., 2021) - https://arxiv.org/abs/2111.00396
2. **Mamba** (Gu & Dao, 2023) - https://arxiv.org/abs/2312.00752
3. **Jamba** (AI21 Labs, 2024) - Hybrid architecture
4. **HiPPO** (Gu et al., 2020) - Foundations

**Code**:
- Official Mamba: https://github.com/state-spaces/mamba
- HuggingFace Transformers: Mamba support
- S4 implementation: https://github.com/state-spaces/s4

**Resources**:
- Annotated S4: https://srush.github.io/annotated-s4/
- Mamba explained: https://thegradient.pub/mamba/

---

## 🎓 Exercises

### Exercise 1: S4 Layer

Implement a simple S4 layer and test on a sequence task.

```python
# Task: Copy sequence
input: [1, 2, 3, 4, 5, ..., 0, 0, 0, ...]
output: [0, 0, 0, ..., 1, 2, 3, 4, 5, ...]

# S4 should learn to remember and recall!
```

### Exercise 2: Compare S4 vs Transformer

Benchmark on Long Range Arena tasks.

```python
def benchmark(model, task, seq_lengths=[1K, 4K, 16K, 64K]):
    for seq_len in seq_lengths:
        # Measure:
        # - Accuracy
        # - Speed
        # - Memory
        pass
```

### Exercise 3: Build Hybrid Model

Create a model mixing Attention and Mamba layers.

```python
# Find optimal mixing ratio
# Test on various sequence lengths
```

---

## ⏭️ Next Steps

State Space Models를 이해했습니다!

👉 [Speculative Decoding](./03-speculative-decoding.md) - SSM으로 더 빠른 inference
👉 [Phase 5.7: Hardware Systems](../phase5.7-hardware-systems/) - Custom CUDA kernels

**Transformer의 대안을 마스터했습니다!** 🌊
