# Day 10: 실전 최적화 사례

## 🎯 목표

**실제 AI 시스템의 최적화 전략 이해**

---

## 🔍 Case 1: Attention 메모리 최적화

### 문제: Quadratic Memory

```python
# Naive Attention
def attention_naive(Q, K, V):  # (batch, seq_len, d_model)
    scores = Q @ K.transpose(-2, -1)  # (batch, seq_len, seq_len)
    # 문제: seq_len=8192 → 64M floats → 256MB per batch!
    
    attn_weights = F.softmax(scores / sqrt(d_k), dim=-1)
    output = attn_weights @ V
    return output

# seq_len이 길어지면 OOM!
```

### 해결 1: Flash Attention

**핵심**: Tiling + Recomputation

```python
# Pseudo-code
def flash_attention(Q, K, V, block_size=128):
    seq_len, d_model = Q.shape
    output = torch.zeros_like(Q)
    
    # Q를 block으로 나눔
    for q_start in range(0, seq_len, block_size):
        Q_block = Q[q_start:q_start+block_size]
        
        # Online softmax accumulator
        max_scores = torch.full((block_size,), -inf)
        sum_exp = torch.zeros(block_size)
        acc = torch.zeros(block_size, d_model)
        
        # K, V를 block으로 순회
        for kv_start in range(0, seq_len, block_size):
            K_block = K[kv_start:kv_start+block_size]
            V_block = V[kv_start:kv_start+block_size]
            
            # Scores for this block
            scores = Q_block @ K_block.T / sqrt(d_k)
            
            # Update online softmax
            new_max = torch.maximum(max_scores, scores.max(dim=1))
            rescale = torch.exp(max_scores - new_max)
            
            acc = acc * rescale.unsqueeze(-1)
            sum_exp = sum_exp * rescale
            
            exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
            acc += exp_scores @ V_block
            sum_exp += exp_scores.sum(dim=1)
            
            max_scores = new_max
        
        # Normalize
        output[q_start:q_start+block_size] = acc / sum_exp.unsqueeze(-1)
    
    return output

# 메모리: O(seq_len * block_size) vs O(seq_len^2)
# seq_len=8192, block_size=128 → 32x memory reduction!
```

**결과**:
- Memory: 256MB → 8MB (32x reduction)
- Speed: 2-4x faster (IO bound → compute bound)
- GPT-3 크기 모델을 single GPU에서 훈련 가능!

---

## 📦 Case 2: KV Cache 최적화 (vLLM)

### 문제: Memory Fragmentation

```python
# Generation 중 KV cache 저장
class NaiveKVCache:
    def __init__(self, max_seq_len):
        # 각 sequence마다 독립적인 cache
        self.cache = {}
    
    def allocate(self, seq_id, seq_len):
        # 고정 크기 할당 (최대 길이 기준)
        self.cache[seq_id] = torch.zeros(
            max_seq_len, d_model  # Wasteful!
        )
    
# 문제:
# - 대부분 sequence는 max_seq_len보다 짧음
# - 메모리 낭비 50-90%!
# - Batch size 작아짐
```

### 해결: PagedAttention (vLLM)

**핵심**: OS의 virtual memory처럼 paging!

```python
class PagedKVCache:
    def __init__(self, block_size=16):
        self.block_size = block_size
        # Physical blocks (실제 메모리)
        self.physical_blocks = []
        # Logical → Physical mapping
        self.block_tables = {}
    
    def allocate(self, seq_id):
        # 필요한 만큼만 block 할당
        self.block_tables[seq_id] = []
    
    def append(self, seq_id, kv):
        # Block이 가득 차면 새 block 할당
        if len(self.block_tables[seq_id]) == 0 or \
           self.block_tables[seq_id][-1].is_full():
            new_block = self.allocate_physical_block()
            self.block_tables[seq_id].append(new_block)
        
        # 마지막 block에 추가
        self.block_tables[seq_id][-1].append(kv)

# Attention with paging
@triton.jit
def paged_attention_kernel(
    Q, block_tables, key_cache, value_cache, ...
):
    # block_tables[seq_id]로 physical blocks 찾기
    # → scattered memory access를 efficient하게!
    ...

# 결과:
# - Memory 활용률: 50% → 90%
# - Batch size: 2x-3x 증가
# - Throughput: 2x-3x 증가
```

---

## 🔢 Case 3: Quantization (INT8/INT4)

### 문제: FP16도 메모리 많이 사용

```python
# LLaMA 70B (FP16)
70B parameters × 2 bytes = 140GB
→ 2x A100 (80GB) 필요!
```

### 해결: Quantization

**INT8 (8-bit)**
```python
# Per-tensor quantization
def quantize_int8(tensor):
    # 1. Scale 계산
    scale = tensor.abs().max() / 127.0
    
    # 2. Quantize
    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    
    return quantized, scale

def dequantize_int8(quantized, scale):
    return quantized.float() * scale

# Matrix multiplication with INT8
C = (A_int8 @ B_int8).float() * scale_A * scale_B

# LLaMA 70B (INT8)
70B × 1 byte = 70GB → Single A100!
```

**INT4 (4-bit) with GPTQ**
```python
# Group-wise quantization
def quantize_gptq(W, group_size=128):
    # W: (out_features, in_features)
    
    # In_features를 group으로 나눔
    W_reshaped = W.reshape(-1, group_size)
    
    # 각 group마다 scale
    scales = W_reshaped.abs().max(dim=-1, keepdim=True) / 7.0
    
    # Quantize to 4-bit
    W_quant = (W_reshaped / scales).round().clamp(-8, 7)
    
    # Pack: 2 values per byte
    W_packed = pack_4bit(W_quant)
    
    return W_packed, scales

# LLaMA 70B (INT4)
70B × 0.5 byte = 35GB → 절반!
```

**Trade-offs**
```
         | Memory | Speed | Accuracy
---------|--------|-------|----------
FP16     | 1x     | 1x    | 100%
INT8     | 0.5x   | 2x    | 99.5%
INT4     | 0.25x  | 3x    | 98-99%

→ INT4 with GPTQ: minimal accuracy loss!
```

---

## ⚙️ Case 4: Kernel Fusion

### 문제: 여러 작은 kernels

```python
# Transformer layer
class TransformerLayer(nn.Module):
    def forward(self, x):
        # 1. LayerNorm
        x = self.ln1(x)  # Kernel 1
        
        # 2. Attention
        q = self.q_proj(x)  # Kernel 2
        k = self.k_proj(x)  # Kernel 3
        v = self.v_proj(x)  # Kernel 4
        attn = F.softmax(q @ k.T / sqrt(d_k), dim=-1)  # Kernel 5, 6
        x = attn @ v  # Kernel 7
        
        # 3. Add & Norm
        x = x + residual  # Kernel 8
        x = self.ln2(x)  # Kernel 9
        
        # 4. FFN
        x = self.ffn(x)  # Kernel 10, 11
        
        # Total: 11 kernel launches!
        # → Memory bandwidth bound
```

### 해결: torch.compile

```python
# PyTorch 2.0+
compiled_layer = torch.compile(layer)

# 내부적으로 Triton으로 kernel fusion
# 11 kernels → 2-3 fused kernels
# 결과: 1.5-2x speedup!
```

**수동 fusion (Triton)**
```python
@triton.jit
def fused_attention_forward(
    Q_ptr, K_ptr, V_ptr, output_ptr,
    ln_weight_ptr, ln_bias_ptr,  # LayerNorm params
    ...
):
    # 1. LayerNorm
    # 2. Attention
    # 3. All in one kernel!
    ...

# Single kernel launch
# → 메모리 접근 최소화
```

---

## 🌐 Case 5: Distributed Training

### 문제: 모델이 single GPU에 안 들어감

```python
# GPT-3 175B (FP16)
175B × 2 bytes = 350GB
→ 1x A100 (80GB) 불가능!
```

### 해결 1: Data Parallelism

```python
# 각 GPU가 full model + different data
# DDP (DistributedDataParallel)
model = nn.parallel.DistributedDataParallel(model)

# 문제: 모델이 GPU 메모리보다 크면?
```

### 해결 2: Model Parallelism

**Pipeline Parallelism**
```python
# 모델을 layer별로 나눔
# GPU 0: Layers 0-11
# GPU 1: Layers 12-23
# GPU 2: Layers 24-35
# GPU 3: Layers 36-47

from torch.distributed.pipeline.sync import Pipe
model = Pipe(model, chunks=8)
```

**Tensor Parallelism**
```python
# 각 layer를 tensor별로 나눔
# GPU 0: W[:, :d/2]
# GPU 1: W[:, d/2:]

# Megatron-LM style
class ColumnParallelLinear:
    def forward(self, x):
        # All-gather input
        x_parallel = all_gather(x)
        # Compute on local shard
        output = F.linear(x_parallel, self.weight_shard)
        return output
```

### 해결 3: ZeRO (DeepSpeed)

**ZeRO-3: 모든 것을 shard**
```python
from deepspeed import zero

# Optimizer states, gradients, parameters 모두 분산
# → 메모리 1/N (N = GPU 수)

config = {
    "zero_optimization": {
        "stage": 3,  # Shard everything
        "offload_optimizer": True,  # CPU offload
    }
}

model_engine = deepspeed.initialize(model=model, config=config)

# 1T parameters on 64 GPUs!
```

---

## 📊 성능 측정 & Profiling

### nsys (Nsight Systems)

```bash
# Profile GPU execution
nsys profile -o output python train.py

# View timeline
nsys-ui output.qdrep

# Key metrics:
# - GPU utilization
# - Memory bandwidth
# - Kernel duration
# - CPU-GPU transfer
```

### torch.profiler

```python
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True
) as prof:
    model(input)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Output:
# -------  ------------  -------  -------
#  Name         CPU Time  CUDA Time  Count
# -------  ------------  -------  -------
#  matmul      0.5ms     5.2ms     24
#  softmax     0.1ms     1.3ms     12
#  ...
```

---

## 🎓 최적화 체크리스트

**메모리 최적화**
- [ ] Flash Attention for long sequences
- [ ] Paged KV cache for inference
- [ ] Quantization (INT8/INT4)
- [ ] Gradient checkpointing

**속도 최적화**
- [ ] Kernel fusion (torch.compile)
- [ ] Mixed precision (FP16/BF16)
- [ ] Efficient attention (Flash, xFormers)
- [ ] Batch size tuning

**확장성**
- [ ] Data parallelism (DDP)
- [ ] Pipeline parallelism
- [ ] Tensor parallelism
- [ ] ZeRO optimizer

---

## 📈 Before & After

### Training GPT-3 175B

```
Before optimizations:
- Hardware: 1024x A100 GPUs
- Throughput: 300 tokens/sec
- Cost: $10M

After optimizations:
- Flash Attention: 2x speedup
- Mixed precision: 2x speedup  
- ZeRO-3: 4x memory efficiency
- Total: 512x A100, 1200 tokens/sec, $5M

→ 50% cost reduction!
```

---

## 🎯 완료 기준

이 Phase를 완료하면:
- ✅ Flash Attention의 tiling 전략 이해
- ✅ vLLM의 PagedAttention 원리 이해
- ✅ Quantization (INT8/INT4) 적용 가능
- ✅ Kernel fusion으로 2x speedup 가능
- ✅ Distributed training 전략 선택 가능

---

## ⏭️ 다음 단계

축하합니다! 🎉

Phase 5.7을 완료하여 **하드웨어부터 최적화까지** 모든 것을 이해했습니다!

이제:
- GPU에서 Q @ K.T가 어떻게 계산되는지 설명 가능 ✅
- CUDA로 custom kernel 작성 가능 ✅
- Triton으로 고수준 최적화 가능 ✅
- 실전 시스템 (vLLM, DeepSpeed) 이해 ✅

👉 [Phase 6: 실전 프로젝트](../phase6-project/)에서 모든 지식을 통합하세요!

**"이제 당신은 AI 시스템의 모든 레이어를 이해합니다!"** 🚀
