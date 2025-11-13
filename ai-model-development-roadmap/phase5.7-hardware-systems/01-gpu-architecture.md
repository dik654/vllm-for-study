# Day 1-2: GPU 아키텍처 - 벡터 내적이 어떻게 계산되는가

## 🎯 학습 목표

**"Q @ K.T가 GPU에서 실제로 어떻게 계산되는지 완전히 이해합니다."**

---

## 1. CPU vs GPU: 왜 AI는 GPU인가?

### 📖 근본적인 차이

**CPU (Central Processing Unit)**
```
설계: 순차 처리 최적화
코어: 4-16개 (고성능)
클럭: ~5GHz
캐시: 큼 (수십 MB)

작업: if/else, loop, 복잡한 제어 흐름
```

**GPU (Graphics Processing Unit)**
```
설계: 병렬 처리 최적화
코어: 수천 개 (저성능)
클럭: ~1-2GHz
캐시: 작음

작업: 같은 연산을 수백만 번 (픽셀, 행렬)
```

### 💡 AI는 왜 GPU?

```python
# Attention의 핵심: 행렬 곱셈
Q = (batch, seq_len, d_k)  # (4, 512, 64)
K = (batch, seq_len, d_k)

scores = Q @ K.T  # (4, 512, 512)
# → 4 × 512 × 512 × 64 = 67,108,864 번의 곱셈!

# CPU: 16 코어로 순차 처리 → 느림
# GPU: 10,000+ 코어로 병렬 처리 → 빠름!
```

---

## 2. NVIDIA GPU 아키텍처 (A100 기준)

### 📊 계층 구조

```
GPU (전체 칩)
├── GPC (Graphics Processing Cluster) × 8
│   ├── TPC (Texture Processing Cluster) × 7-9
│   │   └── SM (Streaming Multiprocessor) × 2
│   │       ├── CUDA Core × 64
│   │       ├── Tensor Core × 4
│   │       ├── Shared Memory: 164KB
│   │       ├── L1 Cache: 192KB
│   │       └── Register File: 256KB
│   └── ...
├── L2 Cache: 40MB
└── HBM2 (Global Memory): 80GB
```

**A100 총계**:
- SM: 108개
- CUDA Core: 6,912개
- Tensor Core: 432개
- Memory Bandwidth: 1,935 GB/s

### 🔬 SM (Streaming Multiprocessor) - 실제 연산 유닛

**SM 하나의 구조**
```
SM
├── CUDA Core × 64
│   ├── FP32 unit (단정밀도)
│   └── INT32 unit (정수)
│
├── Tensor Core × 4
│   └── 4×4 행렬 곱셈 (FP16/BF16)
│
├── Shared Memory: 164KB
│   └── 블록 내 스레드 간 공유
│
├── Register File: 256KB
│   └── 각 스레드 전용
│
├── L1 Cache + Texture Cache
│
└── Warp Scheduler × 4
    └── 32 스레드 묶음 관리
```

---

## 3. 벡터 내적이 실제로 계산되는 과정

### 💻 예제: 간단한 Dot Product

```python
# Python
a = [1, 2, 3, 4]  # 4차원 벡터
b = [5, 6, 7, 8]
dot = sum(a[i] * b[i] for i in range(4))  # 70
```

### 🔄 GPU에서의 실행 (단계별)

#### **Step 1: CPU → GPU 메모리 복사**
```python
a_gpu = torch.tensor(a).cuda()  # CPU → GPU
b_gpu = torch.tensor(b).cuda()
```

```
CPU Memory          →    GPU Global Memory (HBM)
[1,2,3,4]           →    Address 0x1000: [1,2,3,4]
[5,6,7,8]           →    Address 0x2000: [5,6,7,8]

복사 속도: PCIe 4.0 기준 ~16 GB/s
```

#### **Step 2: Kernel 실행 준비**

```python
# PyTorch가 내부적으로 하는 일
torch.dot(a_gpu, b_gpu)

# → CUDA kernel 호출
dot_product_kernel<<<1, 4>>>(a_ptr, b_ptr, result_ptr, n=4);
```

**Kernel 설정**:
```
Grid: 1 블록
Block: 4 스레드 (각 원소 하나씩)
```

#### **Step 3: Global Memory → Shared Memory**

```
SM 내부:

Global Memory (느림, ~200 cycles)
    ↓ Load
Shared Memory (빠름, ~20 cycles)
    ↓
Register (가장 빠름, 1 cycle)
```

**CUDA 코드**:
```cuda
__global__ void dot_product_kernel(float *a, float *b, float *result, int n) {
    // Shared memory 선언
    __shared__ float partial_sum[256];

    int tid = threadIdx.x;

    // Step 1: Load to register (각 스레드)
    float my_a = a[tid];  // Global → Register
    float my_b = b[tid];

    // Step 2: Multiply
    float my_product = my_a * my_b;  // Register 내에서

    // Step 3: Store to shared memory
    partial_sum[tid] = my_product;
    __syncthreads();  // 모든 스레드 동기화

    // Step 4: Reduction (합산)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        __syncthreads();
    }

    // Step 5: Write result
    if (tid == 0) {
        result[0] = partial_sum[0];
    }
}
```

#### **Step 4: SM에서 실행**

```
SM 내 4개 스레드:

Thread 0: 1 × 5 = 5
Thread 1: 2 × 6 = 12
Thread 2: 3 × 7 = 21
Thread 3: 4 × 8 = 32

→ 병렬 실행! (동시에 4개 곱셈)

Reduction (합산):
[5, 12, 21, 32]
→ [17, 53]    (0+1, 2+3)
→ [70]        (0+1)
```

#### **Step 5: GPU → CPU 결과 복사**

```python
result = dot_gpu.cpu()  # GPU → CPU
print(result)  # 70
```

---

## 4. 행렬 곱셈 (Matrix Multiplication)

### 📐 수식

$$
C = A \times B
$$

$$
C[i,j] = \sum_{k=0}^{K-1} A[i,k] \times B[k,j]
$$

### 💻 Naive GPU 구현

```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    // C[i,j] 계산
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // i
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // j

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
            // Global memory 접근: 매우 느림!
        }
        C[row * N + col] = sum;
    }
}
```

**문제점**:
```
각 원소 계산마다 Global Memory 접근 2K번
→ Memory bandwidth가 병목!
```

### 🚀 최적화: Tiling + Shared Memory

```cuda
#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Tile 단위로 처리
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Global → Shared (블록당 1번)
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Shared memory에서 계산 (빠름!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

**개선 효과**:
```
Naive: Global memory 접근 2MNK번
Tiled: Global memory 접근 MN + NK번 (약 K배 감소!)

예: 1024×1024 행렬
Naive: 2GB memory traffic
Tiled: ~2MB memory traffic
→ 1000배 감소!
```

---

## 5. Tensor Core - 하드웨어 가속

### 🔥 Tensor Core란?

**CUDA Core**: 하나의 FP32 곱셈+덧셈
**Tensor Core**: **4×4 행렬** 곱셈을 **한 사이클**에!

```
Tensor Core (1 cycle):
    [4×4] @ [4×4] = [4×4]

CUDA Core (16 cycles):
    16번의 곱셈+덧셈
```

### 💡 FP16/BF16에서 폭발적 속도

```
A100 성능:
- FP32 (CUDA Core): 19.5 TFLOPS
- FP16 (Tensor Core): 312 TFLOPS  ← 16배!
- BF16 (Tensor Core): 312 TFLOPS
```

### 💻 PyTorch에서 Tensor Core 활용

```python
# Automatic Mixed Precision (AMP)
from torch.cuda.amp import autocast, GradScaler

model = MyModel().cuda()
optimizer = Adam(model.parameters())
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    # Forward: FP16 (Tensor Core 사용!)
    with autocast():
        output = model(data)
        loss = criterion(output, target)

    # Backward: FP32로 gradient 누적 (안정성)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**효과**:
```
속도: 2-3배 향상
메모리: 50% 절약
정확도: 거의 동일 (BF16 사용 시)
```

---

## 6. Memory Hierarchy 완전 이해

### 📊 메모리 속도 비교 (A100 기준)

| 메모리 | 크기 | Latency | Bandwidth |
|--------|------|---------|-----------|
| Register | ~256KB/SM | 1 cycle | 극고속 |
| Shared Memory | 164KB/SM | ~20 cycles | ~19 TB/s |
| L1 Cache | 192KB/SM | ~30 cycles | ~15 TB/s |
| L2 Cache | 40MB | ~200 cycles | ~3 TB/s |
| HBM (Global) | 80GB | ~300 cycles | 1.9 TB/s |

**핵심**: Register > Shared > L1 > L2 > Global

### 💻 실습: Memory Bandwidth 측정

```python
import torch
import time

def measure_bandwidth(size_mb, device='cuda'):
    # 데이터 준비
    n = size_mb * 1024 * 1024 // 4  # float32
    a = torch.randn(n, device=device)
    b = torch.randn(n, device=device)

    # Warmup
    for _ in range(10):
        c = a + b

    # 측정
    torch.cuda.synchronize()
    start = time.time()

    repeats = 100
    for _ in range(repeats):
        c = a + b  # Read 2×size, Write 1×size = 3×size

    torch.cuda.synchronize()
    elapsed = time.time() - start

    # Bandwidth 계산
    bytes_transferred = 3 * size_mb * repeats  # MB
    bandwidth_gbps = (bytes_transferred / elapsed) / 1024  # GB/s

    print(f"Size: {size_mb}MB")
    print(f"Bandwidth: {bandwidth_gbps:.1f} GB/s")
    print(f"Peak: {torch.cuda.get_device_properties(0).memory_bandwidth / 1e9:.1f} GB/s")
    print(f"Efficiency: {bandwidth_gbps / (torch.cuda.get_device_properties(0).memory_bandwidth / 1e9) * 100:.1f}%")

# 테스트
for size in [1, 10, 100, 1000]:
    measure_bandwidth(size)
    print()
```

---

## 7. 실전 예제: Attention은 어떻게 계산되는가?

### 💻 PyTorch Attention

```python
import torch
import torch.nn.functional as F

batch, seq_len, d_k = 4, 512, 64

Q = torch.randn(batch, seq_len, d_k).cuda()
K = torch.randn(batch, seq_len, d_k).cuda()
V = torch.randn(batch, seq_len, d_k).cuda()

# Attention 계산
scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (4, 512, 512)
attn = F.softmax(scores, dim=-1)
output = attn @ V
```

### 🔍 GPU에서 실제로 일어나는 일

#### **1. Q @ K.T (Matrix Multiplication)**
```
Shape: (4, 512, 64) @ (4, 64, 512) = (4, 512, 512)

GPU 실행:
- cuBLAS 라이브러리 호출 (NVIDIA 최적화 GEMM)
- Tensor Core 사용 (FP16/BF16)
- Tiling + Shared Memory
- 수천 개 스레드 병렬 실행

시간: ~0.5ms (A100 기준)
```

#### **2. Softmax**
```
각 행마다 독립적:
row_i = softmax(scores[i])

병렬화:
- 512개 행을 동시에 처리
- 각 SM에서 여러 행 담당

시간: ~0.1ms
```

#### **3. Attn @ V**
```
다시 행렬 곱셈:
(4, 512, 512) @ (4, 512, 64) = (4, 512, 64)

cuBLAS + Tensor Core

시간: ~0.3ms
```

**총 시간**: ~1ms (Flash Attention으로 더 빠르게 가능!)

---

## ✅ Day 1-2 완료 체크리스트

### 이론 이해
- [ ] CPU vs GPU 차이 (병렬 처리)
- [ ] SM, CUDA Core, Tensor Core 구조
- [ ] Memory hierarchy (Register → Global)
- [ ] Memory bandwidth가 왜 중요한지

### 실습 완료
- [ ] GPU 정보 확인 (torch.cuda)
- [ ] Dot product의 병렬 계산 이해
- [ ] Matrix multiplication 최적화 (tiling)
- [ ] Memory bandwidth 측정

### 하드웨어 연결
- [ ] Attention 계산이 GPU에서 어떻게 되는지 설명 가능
- [ ] Tensor Core의 역할 이해
- [ ] Mixed precision이 왜 빠른지 설명 가능

---

## 🎯 최종 챌린지

**"Q @ K.T가 GPU에서 어떻게 계산되는가?" 상세히 설명하세요!**

정답에 포함되어야 할 것:
1. Memory 복사 (CPU → GPU)
2. cuBLAS kernel 호출
3. Tiling + Shared Memory
4. Tensor Core 활용 (FP16)
5. 병렬 실행 (수천 스레드)

이것을 설명할 수 있으면 **GPU 아키텍처 마스터!** 🎉

---

## ⏭️ 다음 단계

👉 [Day 3-4: CUDA 프로그래밍](./02-cuda-programming.md)

**"이제 GPU의 모든 것을 이해합니다!"** 🚀
