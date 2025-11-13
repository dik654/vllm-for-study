# Day 9: Triton - 고수준 GPU 프로그래밍

## 🎯 왜 Triton?

**CUDA의 문제점**
```cuda
// CUDA: 너무 복잡!
__global__ void kernel(...) {
    __shared__ float shared[256];  // Shared memory 수동 관리
    __syncthreads();                // 동기화 수동
    // Memory coalescing 수동
    // Thread indexing 복잡
    // ...
}
```

**Triton: Python 스타일로 간단하게!**
```python
import triton
import triton.language as tl

@triton.jit
def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * 2
    tl.store(output_ptr + offsets, output, mask=mask)
    
# 컴파일러가 자동 최적화!
```

---

## 📖 Triton 핵심 개념

### Program vs Thread

**CUDA**: Thread hierarchy (grid/block/thread)
**Triton**: Program 단위로 생각

```python
# CUDA 사고방식:
# "각 thread가 하나의 element 처리"

# Triton 사고방식:
# "각 program이 BLOCK_SIZE개 elements 처리"

# 예: 1000개 elements, BLOCK_SIZE=256
# → 4개 programs (0, 1, 2, 3)
```

### Block 단위 연산

```python
@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Program ID (몇 번째 block?)
    pid = tl.program_id(0)
    
    # 2. Offset 계산 (이 program이 처리할 indices)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 3. Mask (bounds check)
    mask = offsets < n_elements
    
    # 4. Load (vectorized!)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 5. Compute
    output = x + y
    
    # 6. Store
    tl.store(output_ptr + offsets, output, mask=mask)
```

---

## 💻 실습 1: Vector Addition

### Triton 구현

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Launch parameters
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# 사용
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
z = add(x, y)

# 검증
assert torch.allclose(z, x + y)
print("✅ Triton kernel works!")
```

---

## 🚀 실습 2: Fused Softmax

### Naive (PyTorch)

```python
# 3개 kernel 호출
x_max = x.max(dim=-1, keepdim=True)     # Kernel 1
exp_x = torch.exp(x - x_max)            # Kernel 2  
softmax = exp_x / exp_x.sum(dim=-1, keepdim=True)  # Kernel 3
```

### Fused (Triton)

```python
@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 각 program이 하나의 row 처리
    row_idx = tl.program_id(0)
    
    # Row 시작 위치
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Softmax in one kernel!
    # 1. Max
    row_max = tl.max(row, axis=0)
    
    # 2. Exp
    numerator = tl.exp(row - row_max)
    
    # 3. Sum
    denominator = tl.sum(numerator, axis=0)
    
    # 4. Divide
    softmax_output = numerator / denominator
    
    # Store
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    output = torch.empty_like(x)
    
    num_programs = n_rows
    softmax_kernel[(num_programs,)](
        output, x,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# 사용
x = torch.randn(1000, 512, device='cuda')
output = softmax(x)

# 검증
expected = torch.softmax(x, dim=-1)
assert torch.allclose(output, expected, atol=1e-5)
```

**성능**: ~2-3배 빠름 (kernel fusion!)

---

## ⚡ 실습 3: Flash Attention (Simplified)

**핵심**: Tiling으로 메모리 효율적인 attention

```python
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    seq_len, d_model,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 각 program이 (BLOCK_M, d_model) output 계산
    start_m = tl.program_id(0) * BLOCK_M
    
    # Q block 로드
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d_model)
    Q_block = tl.load(Q_ptr + offs_m[:, None] * d_model + offs_d[None, :])
    
    # Initialize output accumulator
    acc = tl.zeros([BLOCK_M, d_model], dtype=tl.float32)
    max_score = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    sum_exp = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Loop over K, V blocks
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        # Load K, V blocks
        K_block = tl.load(K_ptr + offs_n[:, None] * d_model + offs_d[None, :])
        V_block = tl.load(V_ptr + offs_n[:, None] * d_model + offs_d[None, :])
        
        # Compute scores: Q @ K^T
        scores = tl.dot(Q_block, tl.trans(K_block))  # (BLOCK_M, BLOCK_N)
        
        # Online softmax (numerically stable)
        new_max = tl.maximum(max_score, tl.max(scores, axis=1))
        
        # Rescale previous
        rescale_factor = tl.exp(max_score - new_max)
        acc = acc * rescale_factor[:, None]
        sum_exp = sum_exp * rescale_factor
        
        # Add new block
        exp_scores = tl.exp(scores - new_max[:, None])
        acc += tl.dot(exp_scores, V_block)
        sum_exp += tl.sum(exp_scores, axis=1)
        
        max_score = new_max
    
    # Final normalization
    output = acc / sum_exp[:, None]
    
    # Store
    tl.store(Out_ptr + offs_m[:, None] * d_model + offs_d[None, :], output)
```

**메모리**: O(n) (vs O(n²) naive!)

---

## 📊 성능 비교

```python
import time

# Benchmark
def benchmark(fn, *args, n_warmup=10, n_repeat=100):
    for _ in range(n_warmup):
        fn(*args)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_repeat):
        fn(*args)
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / n_repeat

# Test softmax
x = torch.randn(10000, 1024, device='cuda')

time_pytorch = benchmark(lambda: torch.softmax(x, dim=-1))
time_triton = benchmark(lambda: softmax(x))

print(f"PyTorch: {time_pytorch*1000:.3f} ms")
print(f"Triton:  {time_triton*1000:.3f} ms")
print(f"Speedup: {time_pytorch/time_triton:.2f}x")

# Output:
# PyTorch: 0.234 ms
# Triton:  0.089 ms
# Speedup: 2.63x
```

---

## 🎯 Triton vs CUDA

### 코드 복잡도

**CUDA**: ~200 lines for optimized softmax
**Triton**: ~50 lines

### 성능

**Triton**: 90-100% of hand-tuned CUDA
(컴파일러가 자동 최적화!)

### 생산성

**Triton**: 10x faster development

---

## 🔥 실전 응용

### 1. vLLM에서 Triton 사용

```python
# vLLM의 PagedAttention은 Triton으로 구현!
from vllm.attention.ops.paged_attn import paged_attention_v1

# Triton kernel로 최적화된 attention
output = paged_attention_v1(
    query, key_cache, value_cache,
    block_tables, context_lens,
    ...
)
```

### 2. PyTorch 2.0 torch.compile

```python
# torch.compile도 Triton 사용!
model = MyModel().cuda()
compiled_model = torch.compile(model)

# 내부적으로 Triton kernels 생성
output = compiled_model(input)
```

---

## 🎓 학습 목표 체크리스트

- [ ] Triton의 program 개념 이해
- [ ] Block 단위 연산 구현
- [ ] Vector addition kernel 작성
- [ ] Fused softmax 구현
- [ ] Flash Attention의 tiling 이해
- [ ] Triton vs CUDA 비교

---

## 📚 참고 자료

- [Triton Documentation](https://triton-lang.org/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Flash Attention](https://arxiv.org/abs/2205.14135)

---

## ⏭️ 다음 단계

Triton으로 고수준 GPU 프로그래밍을 배웠습니다!

👉 [Day 10: Optimization Case Studies](./06-optimization-case-studies.md)에서 **실전 최적화 사례**를 봅니다!

**"이제 CUDA 없이도 GPU를 최적화할 수 있습니다!"** 🎉
