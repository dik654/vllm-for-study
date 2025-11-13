# Phase 5.7: Hardware & Systems - GPU/NPU/TPU 완전 이해

## 🎯 왜 하드웨어를 알아야 하는가?

**"소프트웨어는 하드웨어 위에서 돌아갑니다."**

### ❓ 현실 체크

코드만 알고 하드웨어를 모르면:
```python
# 느린 코드
for i in range(1000):
    output[i] = model(input[i])  # 하나씩 처리
# → 왜 느린지 모름

# 빠른 코드
output = model(input)  # 배치 처리
# → 왜 빠른지 모름
```

→ **"그냥 배치가 빠르다더라"** (이해 없는 암기)

### 🔥 하드웨어를 알면

```python
# GPU 아키텍처 이해
GPU = 수천 개의 작은 코어
→ 병렬 처리에 최적화
→ Batch processing이 빠른 이유!

# Memory hierarchy 이해
Global Memory: 느림 (수백 사이클)
Shared Memory: 빠름 (수 사이클)
→ Memory coalescing이 중요한 이유!

# Tensor Core 이해
Tensor Core = 4×4 행렬 곱셈 하드웨어
→ Mixed precision (FP16) 엄청 빠른 이유!
```

### 💡 실전 예시

**DeepSeek-V3가 왜 효율적인가?**
```
1. MoE (소프트웨어): Sparse activation
2. GPU 최적화 (하드웨어): Kernel fusion, Memory layout
→ 둘 다 알아야 진짜 이해!
```

---

## 📚 커리큘럼 (1.5주, 60시간)

### Week 1: GPU Architecture & CUDA

#### [Day 1-2: GPU 아키텍처 기초](./01-gpu-architecture.md)

**GPU vs CPU**
```
CPU: 4-16 코어, 고성능, 순차 처리
GPU: 수천 코어, 저성능, 병렬 처리

AI는 왜 GPU?
→ 행렬 곱셈 = 수백만 개 독립 연산
→ GPU의 수천 코어로 병렬 처리!
```

**NVIDIA GPU 구조**
- **SM (Streaming Multiprocessor)**
  - 실제 연산 유닛
  - A100: 108개 SM

- **CUDA Core**
  - FP32 연산
  - SM당 64개 (A100)

- **Tensor Core**
  - 행렬 곱셈 특화 (4×4)
  - FP16/BF16에서 초고속
  - Transformer에 최적!

**Memory Hierarchy**
```
Register: 가장 빠름 (1 cycle)
Shared Memory: 매우 빠름 (수 cycle)
L1 Cache: 빠름
L2 Cache: 중간
Global Memory (VRAM): 느림 (수백 cycle)

→ 핵심: 빠른 메모리 최대한 활용!
```

💻 **실습**: GPU 정보 확인
```python
import torch
print(torch.cuda.get_device_name(0))
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Multiprocessors: {torch.cuda.get_device_properties(0).multi_processor_count}")
```

#### [Day 3-4: CUDA 프로그래밍 기초](./02-cuda-programming.md)

**CUDA 개념**

**Kernel**: GPU에서 실행되는 함수
```cuda
__global__ void add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

**실행 모델**
```
Grid: 전체 작업
├── Block: 작업 그룹 (Shared Memory 공유)
│   └── Thread: 개별 작업
```

**PyTorch에서 CUDA 사용**
```python
# 1. 데이터를 GPU로
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()

# 2. 연산 (자동으로 GPU에서)
z = x @ y  # CUDA kernel 자동 호출!

# 3. CPU로 가져오기
z_cpu = z.cpu()
```

💻 **실습**: Custom CUDA Extension
```python
# PyTorch C++/CUDA extension
# vector_add_cuda.cu
__global__ void vector_add_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// PyTorch wrapper
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::zeros_like(a);
    int threads = 256;
    int blocks = (a.size(0) + threads - 1) / threads;

    vector_add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        a.size(0)
    );
    return c;
}
```

#### [Day 5-6: CUDA 최적화](./03-cuda-optimization.md)

**1. Memory Coalescing**
```cuda
// Bad: Strided access
for (int i = threadIdx.x; i < n; i += 32) {
    sum += arr[i];  // 32 스레드가 멀리 떨어진 메모리 접근
}

// Good: Coalesced access
int idx = threadIdx.x;
sum = arr[idx];  // 연속된 32 스레드가 연속된 메모리 접근
```

**2. Shared Memory 활용**
```cuda
__shared__ float shared_data[256];

// Global → Shared (느림 → 빠름)
shared_data[threadIdx.x] = global_data[idx];
__syncthreads();  // 모든 스레드 동기화

// Shared에서 여러 번 사용 (빠름!)
result = shared_data[threadIdx.x] + shared_data[threadIdx.x + 1];
```

**3. Kernel Fusion**
```python
# Bad: 여러 kernel 호출
x = torch.relu(x)  # Kernel 1
x = x + bias       # Kernel 2
x = torch.dropout(x)  # Kernel 3

# Good: 하나로 fusion
x = fused_relu_add_dropout(x, bias)  # 1 kernel
```

💻 **실습**: Matrix Multiplication 최적화
```cuda
// Naive (느림)
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    for (int k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Optimized: Shared memory + tiling
__global__ void matmul_shared(float *A, float *B, float *C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Tile 단위로 처리...
    // (상세 구현은 문서 참조)
}
```

---

### Week 2: NPU/TPU & Advanced Topics

#### [Day 7-8: NPU/TPU 아키텍처](./04-npu-tpu-architecture.md)

**NPU (Neural Processing Unit)**

**설계 철학**: AI 연산에 특화
```
GPU: 범용 병렬 처리
NPU: AI 전용 최적화
  - Matrix multiplication
  - Convolution
  - Activation functions
```

**Google TPU (Tensor Processing Unit)**

**핵심: Systolic Array**
```
Systolic Array = 데이터가 흐르는 배열

[PE] → [PE] → [PE] → [PE]
 ↓      ↓      ↓      ↓
[PE] → [PE] → [PE] → [PE]
 ↓      ↓      ↓      ↓
[PE] → [PE] → [PE] → [PE]

PE (Processing Element) = MAC (Multiply-Accumulate)

장점:
- 고효율 (데이터 재사용)
- 저전력
- 고성능 행렬 곱셈
```

**TPU v4 스펙**
- 275 TFLOPS (BF16)
- 128GB HBM
- Systolic array: 128×128

**Apple Neural Engine**
```
설계: 저전력 추론
- On-device AI (iPhone, Mac)
- CoreML 최적화
- 16 core (A17 Pro)
```

**비교**
```
           | GPU (A100)    | TPU v4      | Apple ANE
-----------|---------------|-------------|------------
설계 목적  | 범용          | 훈련+추론   | 추론
성능       | 312 TFLOPS    | 275 TFLOPS  | ~15 TOPS
메모리     | 80GB HBM      | 128GB HBM   | Shared
소비전력   | 400W          | ~200W       | ~5W
```

💻 **실습**:
- CoreML로 모델 변환 (Apple)
- TensorFlow로 TPU 사용 (Google Cloud)

#### [Day 9: Triton - 고수준 GPU 프로그래밍](./05-triton-programming.md)

**Triton (OpenAI)**

**문제**: CUDA는 어렵다
```cuda
// CUDA: 복잡함
__global__ void kernel(...) {
    // Shared memory 관리
    // Thread 동기화
    // Memory coalescing
    // ...
}
```

**해결**: Triton으로 간단하게
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Block 단위로 처리
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    # Load
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Compute
    output = x + y

    # Store
    tl.store(output_ptr + offsets, output, mask=mask)
```

**장점**
- Python 스타일 (쉬움)
- 자동 최적화 (메모리, 병렬화)
- PyTorch와 통합

💻 **실습**: Flash Attention을 Triton으로
```python
@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # ...
):
    # Tiling으로 메모리 효율적인 attention
    # (상세 구현)
```

#### [Day 10: 실전 최적화 사례](./06-optimization-case-studies.md)

**Case 1: Attention 최적화**
```python
# Naive: O(n²) memory
scores = Q @ K.T  # (seq_len, seq_len) 저장
attn = softmax(scores)
output = attn @ V

# Flash Attention: O(n) memory
# Tiling + recomputation
output = flash_attention(Q, K, V)  # 메모리 효율적!
```

**Case 2: Large Batch Training**
```python
# Naive: OOM (Out of Memory)
loss = model(huge_batch)

# Gradient Accumulation
for micro_batch in split(huge_batch):
    loss = model(micro_batch)
    loss.backward()  # gradient 누적
optimizer.step()
```

**Case 3: Mixed Precision**
```python
# FP32: 느림
model.float()

# FP16 + Tensor Core: 빠름!
model.half()
# A100 Tensor Core: 312 TFLOPS (FP16) vs 156 TFLOPS (FP32)
```

---

## 🎓 학습 목표

### 이론 이해
- [ ] GPU vs CPU 차이 (병렬 처리)
- [ ] Memory hierarchy (Register → Global)
- [ ] CUDA execution model (Grid/Block/Thread)
- [ ] Systolic array (TPU의 핵심)
- [ ] Tensor Core의 동작 원리

### 실습 완료
- [ ] 간단한 CUDA kernel 작성
- [ ] Memory coalescing 실험
- [ ] Shared memory 활용
- [ ] Triton으로 커스텀 kernel
- [ ] Flash Attention 이해

### 최적화 능력
- [ ] Memory bottleneck 식별
- [ ] Kernel fusion 적용
- [ ] Mixed precision 활용
- [ ] Batch size 최적화
- [ ] Profiling (nsys, nvprof)

---

## 🔧 필수 도구

```bash
# CUDA Toolkit
nvcc --version

# Profiling
nvidia-smi  # GPU 상태
nvprof      # Kernel profiling
nsys        # Nsight Systems

# Triton
pip install triton

# PyTorch CUDA extensions
pip install ninja  # Fast C++ compilation
```

---

## 📊 Before & After

### Before (하드웨어 모름)
```python
# 느린 코드
for x in data:
    y = model(x)
# "왜 느리지? 더 빠른 GPU 사면 되나?"
```

### After (하드웨어 이해)
```python
# GPU 병렬성 활용
y = model(data)  # Batch로!

# Memory 최적화
with torch.cuda.amp.autocast():  # FP16
    y = model(data)

# Kernel fusion
y = torch.compile(model)(data)  # PyTorch 2.0

# "메모리 대역폭이 bottleneck이군. Batch size 줄이고
#  gradient accumulation으로 해결!"
```

---

## 🎯 완료 기준

이 Phase를 완료하면:
- ✅ GPU에서 `Q @ K.T`가 어떻게 계산되는지 설명 가능
- ✅ NPU/TPU가 GPU와 왜 다른지 이해
- ✅ CUDA kernel 작성 및 디버깅 가능
- ✅ Triton으로 커스텀 연산 구현
- ✅ 모델 훈련 속도 2-3배 최적화 가능
- ✅ Memory/Compute bottleneck 식별 및 해결

---

## 🔥 실전 응용

### 1. DeepSpeed/Megatron 이해
```python
# 왜 이렇게 하는지 이해 가능
model = torch.nn.parallel.DistributedDataParallel(model)
# → GPU 간 통신 최적화

deepspeed.init_distributed()
# → ZeRO optimizer가 메모리를 어떻게 나누는지
```

### 2. 커스텀 연산 구현
```python
# Flash Attention 스타일 커스텀 연산
@triton.jit
def my_fused_kernel(...):
    # 여러 연산을 하나로 fusion
    # Memory 접근 최소화
```

### 3. 효율적인 추론
```python
# TensorRT로 최적화
import tensorrt as trt
# Kernel fusion, precision calibration
```

---

## 📚 필수 리소스

### 문서
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton Documentation](https://triton-lang.org/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

### 강의
- [Intro to Parallel Programming (Udacity)](https://www.udacity.com/course/intro-to-parallel-programming--cs344)
- [PMPP Book](http://www.elsevierdirect.com/morgan_kaufmann/kirk/) - Programming Massively Parallel Processors

### 블로그
- [OpenAI Triton](https://openai.com/research/triton)
- [Horace He - Making Deep Learning Go Brrrr](https://horace.io/brrr_intro.html)
- [Lil'Log - The Transformer Family](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)

---

## ⏭️ 다음 단계

Phase 5.7을 완료했다면, 하드웨어와 소프트웨어를 모두 이해하는 **완전한 AI 엔지니어**입니다!

👉 [Phase 6: 실전 프로젝트](../phase6-project/)에서 모든 지식을 통합하세요!

**"이제 GPU에서 일어나는 모든 것을 이해합니다!"** 🚀
