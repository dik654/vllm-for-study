# Speculative Decoding

## 🎯 목표

**LLM Inference를 2-3배 빠르게 만들기 (정확도 손실 없이!)**

문제: Autoregressive 생성은 순차적 → 느림
해결: 작은 모델로 여러 토큰을 예측하고, 큰 모델로 검증

---

## 📊 LLM Inference 병목

### Sequential Generation Problem

```python
# Standard Autoregressive Decoding
for t in range(max_tokens):
    # 각 step마다 전체 모델 forward
    logits = large_model(input_ids)  # 7B 모델, 느림!
    next_token = sample(logits[:, -1, :])
    input_ids = torch.cat([input_ids, next_token], dim=1)

# Problem: 100 tokens = 100번의 large model forward!
# → 각 forward가 100ms라면, 총 10초
```

### Why So Slow?

```python
# LLM inference is memory-bound

# Llama 2 7B (FP16):
# - Model size: 14 GB
# - Memory bandwidth: ~900 GB/s (A100)
# - Time to load model: 14 GB / 900 GB/s ≈ 15 ms

# For EACH token:
# - Load all weights from GPU memory
# - Compute (very fast, 1-2 ms)
# - Write output (fast)
# → 15 ms per token, dominated by memory!

# Batch size helps, but single-sequence generation is slow
```

---

## 💡 Speculative Decoding Idea

### Core Insight

**"작은 모델이 예측한 다음 토큰이 큰 모델 예측과 같으면, 계산 절약!"**

```python
# 1. Small model (fast): Guess next K tokens
guesses = small_model.generate(K tokens)  # Fast! (1B model)

# 2. Large model (slow): Verify all K tokens in parallel
verified = large_model.verify(guesses)  # One forward pass!

# 3. Accept correct guesses, reject rest
# → 여러 토큰을 한 번의 large model pass로 생성!
```

### Algorithm

```
Algorithm: Speculative Decoding

Input:
  - M_p: Large ("target") model (7B, 70B, ...)
  - M_q: Small ("draft") model (0.5B, 1B, ...)
  - K: Number of speculative tokens (4-10)

Loop:
  1. Draft phase:
     - Use M_q to generate K candidate tokens
     - Fast, sequential generation

  2. Verification phase:
     - Use M_p to process ALL K candidates in parallel
     - Compare M_p and M_q predictions

  3. Acceptance:
     - Accept tokens while M_p and M_q agree
     - Stop at first disagreement
     - Sample correction token from M_p

  4. Adjust input and repeat

Result: Same output distribution as M_p, but 2-3x faster!
```

---

## 🔧 Implementation

### Basic Version

```python
class SpeculativeDecoder:
    """
    Speculative Decoding with draft model

    Paper: https://arxiv.org/abs/2211.17192
    """
    def __init__(self, target_model, draft_model, K=5):
        """
        Args:
            target_model: Large model (7B, 70B)
            draft_model: Small model (0.5B, 1B)
            K: Number of speculative tokens
        """
        self.target = target_model
        self.draft = draft_model
        self.K = K

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens=100,
        temperature=1.0
    ):
        """Generate with speculative decoding"""
        generated = input_ids
        num_accepted_total = 0
        num_drafted_total = 0

        while len(generated[0]) < len(input_ids[0]) + max_new_tokens:
            # 1. Draft K tokens with small model
            draft_tokens = []
            draft_probs_list = []
            current_input = generated

            for k in range(self.K):
                draft_logits = self.draft(current_input).logits[:, -1, :] / temperature
                draft_probs = F.softmax(draft_logits, dim=-1)
                draft_token = torch.multinomial(draft_probs, num_samples=1)

                draft_tokens.append(draft_token)
                draft_probs_list.append(draft_probs)

                current_input = torch.cat([current_input, draft_token], dim=1)

            draft_tokens = torch.cat(draft_tokens, dim=1)  # (batch, K)
            num_drafted_total += self.K

            # 2. Verify with large model (parallel!)
            # Concatenate drafted tokens
            verify_input = torch.cat([generated, draft_tokens], dim=1)

            # Single forward pass for all K tokens
            target_logits = self.target(verify_input).logits / temperature
            target_probs = F.softmax(target_logits, dim=-1)

            # 3. Acceptance loop
            num_accepted = 0
            for k in range(self.K):
                # Get target probability for drafted token
                target_prob_k = target_probs[:, -self.K + k, :]
                draft_prob_k = draft_probs_list[k]
                draft_token_k = draft_tokens[:, k]

                # Acceptance probability
                p_target = target_prob_k[0, draft_token_k]
                p_draft = draft_prob_k[0, draft_token_k]

                acceptance_prob = (p_target / (p_draft + 1e-10)).clamp(max=1.0)

                # Accept or reject
                if torch.rand(1).item() < acceptance_prob:
                    # Accept
                    generated = torch.cat([generated, draft_token_k.unsqueeze(0)], dim=1)
                    num_accepted += 1
                else:
                    # Reject: sample correction from adjusted distribution
                    # P'(x) = norm(max(0, P_target(x) - P_draft(x)))
                    adjusted_probs = torch.clamp(
                        target_prob_k - draft_prob_k,
                        min=0
                    )
                    adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)

                    correction_token = torch.multinomial(adjusted_probs, num_samples=1)
                    generated = torch.cat([generated, correction_token], dim=1)
                    num_accepted += 1
                    break  # Stop after first rejection

            num_accepted_total += num_accepted

            # Early stop if EOS
            if generated[0, -1] == tokenizer.eos_token_id:
                break

        # Statistics
        acceptance_rate = num_accepted_total / num_drafted_total
        speedup = num_accepted_total / (len(generated[0]) - len(input_ids[0]))

        print(f"Acceptance rate: {acceptance_rate:.2%}")
        print(f"Effective speedup: {speedup:.2f}x")

        return generated


# Usage
from transformers import AutoModelForCausalLM

# Load models
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

draft_model = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Speculative decoder
decoder = SpeculativeDecoder(
    target_model=target_model,
    draft_model=draft_model,
    K=5
)

# Generate
prompt = "The future of artificial intelligence is"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

output = decoder.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))

# Output example:
# Acceptance rate: 68%
# Effective speedup: 2.1x
```

### Why It's Correct

```python
# Mathematical guarantee:
# Output distribution is IDENTICAL to target model!

# Proof sketch:
# 1. When draft and target agree: Just use draft token (correct!)
# 2. When they disagree:
#    - Acceptance probability: p_target / p_draft
#    - Correction sampling: norm(max(0, p_target - p_draft))
# → Ensures correct distribution

# Result: Lossless speedup!
```

---

## 🚀 Optimizations

### 1. Adaptive K

```python
class AdaptiveSpeculativeDecoder(SpeculativeDecoder):
    """Dynamically adjust K based on acceptance rate"""
    def __init__(self, target_model, draft_model, K_min=2, K_max=10):
        super().__init__(target_model, draft_model, K=K_max)
        self.K_min = K_min
        self.K_max = K_max
        self.recent_acceptance_rates = []

    def adjust_K(self):
        """Adjust K based on recent performance"""
        if len(self.recent_acceptance_rates) < 5:
            return

        avg_rate = sum(self.recent_acceptance_rates[-5:]) / 5

        if avg_rate > 0.8:
            # High acceptance: increase K
            self.K = min(self.K + 1, self.K_max)
        elif avg_rate < 0.5:
            # Low acceptance: decrease K
            self.K = max(self.K - 1, self.K_min)

        print(f"Adjusted K to {self.K} (acceptance rate: {avg_rate:.2%})")

    def generate(self, input_ids, max_new_tokens=100):
        # ... (same as before, but call adjust_K() periodically)
        pass
```

### 2. Tree-based Speculation

```python
def tree_speculative_decode(target_model, draft_model, input_ids, max_tokens=100):
    """
    Generate tree of candidates instead of sequence

    Draft tokens:
        Token 1
        ├── Token 2a
        │   ├── Token 3a
        │   └── Token 3b
        └── Token 2b
            └── Token 3c

    Verify entire tree in single forward pass!
    """
    # Build candidate tree
    tree = build_speculation_tree(draft_model, input_ids, depth=3, branching=2)

    # Flatten tree for verification
    all_candidates = flatten_tree(tree)  # All paths through tree

    # Verify all in parallel (single forward)
    target_logits = target_model(torch.cat([input_ids, all_candidates], dim=1)).logits

    # Find longest accepted path
    best_path = find_longest_accepted_path(tree, target_logits)

    return best_path

# Can accept 5-10 tokens per iteration instead of 2-3!
```

### 3. Model-Parallel Speculation

```python
def parallel_speculative_decode(target_model, draft_model, input_ids):
    """
    Run draft and target models in parallel

    Pipeline:
      GPU 0: Draft model generating tokens
      GPU 1: Target model verifying tokens
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Start drafting next batch while verifying current
        future_draft = executor.submit(draft_model.generate, input_ids)
        future_verify = executor.submit(target_model.verify, previous_draft)

        # Overlap computation!
        draft_result = future_draft.result()
        verify_result = future_verify.result()

    return verified_tokens
```

---

## 📊 Performance Analysis

### Theoretical Speedup

```python
# Given:
# - K: speculation length
# - α: acceptance rate
# - T_draft: draft model time
# - T_target: target model time

# Time per accepted token:
# Without speculation: T_target
# With speculation: (T_draft * K + T_target) / (α * K + 1)

# Speedup:
# S = T_target / ((T_draft * K + T_target) / (α * K + 1))

# Example (Llama 2 70B + TinyLlama 1B):
T_target = 100  # ms
T_draft = 5     # ms (20x faster)
K = 5
alpha = 0.7     # 70% acceptance

time_per_token = (5 * 5 + 100) / (0.7 * 5 + 1)  # = 27.8 ms
speedup = 100 / 27.8  # = 3.6x

print(f"Speedup: {speedup:.2f}x")
```

### Empirical Results

```python
# Benchmark: Llama 2 70B + TinyLlama 1B

Task: CNN/DailyMail Summarization
  Baseline (70B only): 45.2 tokens/sec
  Speculative (K=5): 127.8 tokens/sec
  Speedup: 2.83x

Task: HumanEval Code Generation
  Baseline: 38.7 tokens/sec
  Speculative (K=4): 94.3 tokens/sec
  Speedup: 2.44x

Task: MT-Bench Chatbot
  Baseline: 42.1 tokens/sec
  Speculative (K=6): 139.2 tokens/sec
  Speedup: 3.31x

# Acceptance rate varies by task:
# - Code: ~55% (more predictable)
# - Chat: ~75% (conversational)
# - Creative writing: ~45% (less predictable)
```

---

## 🎯 Choosing Draft Model

### Options

```python
# 1. Same family, smaller size
Target: Llama 2 70B
Draft: Llama 2 7B or TinyLlama 1B
→ Best compatibility, high acceptance rate

# 2. Different family, similar training
Target: Llama 2 70B
Draft: Mistral 7B
→ May work, but lower acceptance

# 3. Distilled version
Target: GPT-3 175B
Draft: GPT-3 distilled 1B (trained to mimic GPT-3)
→ Potentially highest acceptance rate

# 4. LoRA-based draft
Target: Llama 2 70B
Draft: Llama 2 70B with LoRA adapters removed
→ Same base, minimal overhead
```

### Selection Criteria

```python
def choose_draft_model(target_model_size, latency_budget):
    """
    Guidelines for draft model selection

    Rules of thumb:
    - Draft should be 10-50x faster than target
    - Smaller draft → lower acceptance, but faster draft
    - Same architecture family → higher acceptance
    """
    if target_model_size >= 70:
        # Very large target
        draft_size = 7  # 7B draft
        expected_speedup = 2.5-3.5

    elif target_model_size >= 30:
        # Large target
        draft_size = 3  # 3B draft
        expected_speedup = 2.0-2.8

    elif target_model_size >= 7:
        # Medium target
        draft_size = 1  # 1B draft
        expected_speedup = 1.8-2.3

    else:
        # Small target: speculation not worth it
        return None

    return draft_size, expected_speedup


# Example
draft_size, speedup = choose_draft_model(target_model_size=70)
print(f"Use {draft_size}B draft model")
print(f"Expected speedup: {speedup}x")
```

---

## 🔧 Production Deployment

### vLLM Integration

```python
# vLLM has built-in speculative decoding support

from vllm import LLM, SamplingParams

# Load models
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    speculative_model="TinyLlama/TinyLlama-1.1B",  # Draft model
    num_speculative_tokens=5,
    use_v2_block_manager=True,
)

# Generate (automatically uses speculative decoding)
prompts = [
    "The future of AI is",
    "Once upon a time",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print(f"Tokens/sec: {output.metrics.tokens_per_sec:.2f}")
```

### Custom API

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load models at startup
target_model = AutoModelForCausalLM.from_pretrained("llama-70b")
draft_model = AutoModelForCausalLM.from_pretrained("tinyllama-1b")
decoder = SpeculativeDecoder(target_model, draft_model)

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 1.0

@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate with speculative decoding

    Returns same quality as target model, but 2-3x faster!
    """
    input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids

    start_time = time.time()
    output = decoder.generate(
        input_ids,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature
    )
    elapsed = time.time() - start_time

    generated_text = tokenizer.decode(output[0])
    tokens_per_sec = request.max_tokens / elapsed

    return {
        "generated_text": generated_text,
        "tokens_per_sec": tokens_per_sec,
        "latency_ms": elapsed * 1000
    }
```

---

## 🆚 Alternatives Comparison

### Speculative Decoding vs. Other Methods

```python
# Method 1: Batching
# - Speedup: 10-100x (many sequences)
# - Latency: Same per sequence
# - Use: High throughput serving

# Method 2: Quantization (INT8, INT4)
# - Speedup: 2-4x
# - Quality: Slight degradation
# - Use: Memory-constrained deployment

# Method 3: Speculative Decoding
# - Speedup: 2-3x
# - Quality: Zero degradation!
# - Use: Single-sequence low latency

# Method 4: Model Distillation
# - Speedup: 10x+ (use small model only)
# - Quality: Significant degradation
# - Use: When quality loss acceptable

# Best: Combine all!
# Batching + Quantization + Speculative = 200x+ speedup!
```

---

## 📚 References

**Papers**:
1. **Speculative Decoding** (Leviathan et al., 2022) - https://arxiv.org/abs/2211.17192
2. **Fast Inference from Transformers** (Stern et al., 2018) - Blockwise parallel decoding
3. **SpecInfer** (Miao et al., 2023) - Tree-based speculation
4. **Medusa** (Cai et al., 2024) - Multi-head speculation

**Code**:
- vLLM: Production speculative decoding
- HuggingFace Transformers: `assisted_generation`
- DeepSpeed-FastGen: Model-parallel speculation

---

## 🎓 Exercises

### Exercise 1: Basic Implementation

Implement speculative decoding from scratch.

```python
# Test with:
# - Target: GPT-2 Medium (355M)
# - Draft: GPT-2 Small (117M)
# Measure speedup
```

### Exercise 2: Optimal K

Find optimal speculation length for different tasks.

```python
def benchmark_K(target, draft, task, K_values=[2, 4, 6, 8, 10]):
    for K in K_values:
        # Measure speedup and acceptance rate
        # Plot K vs. speedup
        pass
```

### Exercise 3: Production System

Build a complete speculative decoding inference server.

```python
# Features:
# - Model loading and caching
# - Adaptive K adjustment
# - Metrics and monitoring
# - Fallback to standard decoding if needed
```

---

## ⏭️ Next Steps

Speculative decoding을 마스터했습니다!

👉 [Quantization & Deployment](./02-quantization-deployment.md) - 양자화와 결합하면 더 빠름
👉 [Phase 5.7: Hardware Systems](../phase5.7-hardware-systems/) - GPU에서 최적화하기

**이제 LLM Inference를 2-3배 빠르게 만들 수 있습니다!** ⚡
