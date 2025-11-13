# Day 5-6: CUDA 최적화

## 🎯 목표

**Naive CUDA → 10-100배 최적화!**

```
Naive matmul: 50 GFLOPS
↓ Memory coalescing
100 GFLOPS (2x)
↓ Shared memory
500 GFLOPS (10x)
↓ Tensor cores
5000 GFLOPS (100x!)
```

---

## 🔧 1. Memory Coalescing

### 문제: Strided Access

```cuda
// Bad: 32 threads가 멀리 떨어진 메모리 접근
__global__ void bad_access(float *data) {
    int idx = threadIdx.x * 32;  // 0, 32, 64, ...
    float val = data[idx];  // 메모리 낭비!
}
```

### 해결: Coalesced Access

```cuda
// Good: 연속된 32 threads가 연속된 메모리 접근
__global__ void good_access(float *data) {
    int idx = threadIdx.x;  // 0, 1, 2, ...
    float val = data[idx];  // 효율적!
}
```

**원리**: GPU는 32 threads (warp)를 묶어서 실행
- 연속 메모리 접근 → 1번에 로드 ✅
- 띄엄띄엄 접근 → 32번 로드 ❌

---

## 🚀 2. Shared Memory

### Matrix Multiplication 최적화

**Naive (느림)**
```cuda
// Global memory에서 반복 접근
for (int k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
    // A, B를 매번 global에서 읽음!
}
```

**Optimized with Shared Memory (빠름!)**
```cuda
#define TILE_SIZE 16

__global__ void matmul_shared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Tile 단위로 처리
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 1. Global → Shared (협업)
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();  // 모든 thread 동기화
        
        // 2. Shared에서 계산 (빠름!)
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

**성능 향상**: Global memory 접근 ~100배 감소!

---

## ⚡ 3. Kernel Fusion

### 문제: 여러 Kernel 호출

```python
# Bad: 3번 kernel 호출 + 2번 메모리 왕복
x = torch.relu(x)         # Kernel 1
x = x + bias              # Kernel 2
x = torch.dropout(x, p)   # Kernel 3
```

### 해결: Fused Kernel

```cuda
__global__ void fused_relu_add_dropout(
    float *x, float *bias, float *out, float *mask,
    int n, float p
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // 1. ReLU
        float val = max(0.0f, x[i]);
        
        // 2. Add bias
        val += bias[i];
        
        // 3. Dropout
        if (mask[i] < p) val = 0.0f;
        else val /= (1.0f - p);
        
        out[i] = val;
    }
}
```

**이득**:
- Memory bandwidth 절약
- Kernel launch overhead 감소
- 2-3배 빠름!

---

## 🎯 4. Occupancy 최적화

### Occupancy란?

```
Occupancy = 실제 활성 threads / 최대 가능 threads
```

**높은 occupancy = GPU 활용률 높음**

### Block/Thread Size 조정

```cuda
// Bad: 낮은 occupancy
dim3 threads(32, 1);   // 32 threads/block만 사용
// → GPU cores 대부분 놀음!

// Good: 높은 occupancy  
dim3 threads(16, 16);  // 256 threads/block
// → GPU cores 활발히 사용!
```

**CUDA Occupancy Calculator** 사용:
```bash
nvcc --ptxas-options=-v kernel.cu
# Output: registers per thread, shared memory usage
# → Occupancy 계산
```

---

## 🔥 5. Warp Divergence 최소화

### 문제: Branch Divergence

```cuda
// Bad: warp 내에서 다른 경로
__global__ void divergent(int *data, int n) {
    int i = threadIdx.x;
    if (i % 2 == 0) {
        // 절반 threads는 이것 실행
        data[i] *= 2;
    } else {
        // 나머지 절반은 이것 실행
        data[i] += 1;
    }
    // → 두 경로를 순차적으로 실행! (느림)
}
```

### 해결

```cuda
// Good: Predication 사용
__global__ void no_divergence(int *data, int n) {
    int i = threadIdx.x;
    int is_even = (i % 2 == 0);
    
    // 모든 threads가 같은 코드 실행 (predicated)
    data[i] = is_even ? data[i] * 2 : data[i] + 1;
}
```

---

## 💻 실습: Flash Attention (Simplified)

**Naive Attention**: O(n²) memory
```python
scores = Q @ K.T  # (seq_len, seq_len) 저장
attn = softmax(scores)
output = attn @ V
```

**Flash Attention**: O(n) memory with tiling

```cuda
#define BLOCK_SIZE 128

__global__ void flash_attention_kernel(
    float *Q, float *K, float *V, float *O,
    int seq_len, int d_model
) {
    __shared__ float Q_shared[BLOCK_SIZE][D_MODEL];
    __shared__ float K_shared[BLOCK_SIZE][D_MODEL];
    __shared__ float V_shared[BLOCK_SIZE][D_MODEL];
    
    // Tiling: seq_len을 BLOCK_SIZE로 분할
    for (int block_q = 0; block_q < seq_len; block_q += BLOCK_SIZE) {
        // Load Q block to shared
        // ...
        
        float max_score = -INFINITY;
        float sum_exp = 0.0f;
        float acc[D_MODEL] = {0};
        
        for (int block_k = 0; block_k < seq_len; block_k += BLOCK_SIZE) {
            // Load K, V blocks to shared
            // ...
            
            // Compute attention for this block
            // Update running statistics (online softmax)
            // ...
        }
        
        // Write output
        // ...
    }
}
```

**핵심**: 전체 attention matrix를 저장하지 않고, block 단위로 계산!

---

## 📊 성능 측정

### CUDA Events

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// Kernel execution
kernel<<<blocks, threads>>>();
cudaEventRecord(stop);

cudaEventSynchronize(stop);
float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);

printf("Time: %.3f ms\n", milliseconds);
```

### Nsight Profiler

```bash
# Profile with Nsight Systems
nsys profile -o output ./your_program

# View results
nsys-ui output.qdrep

# Key metrics:
# - Memory bandwidth utilization
# - SM occupancy
# - Warp stall reasons
```

---

## 🎓 학습 목표 체크리스트

- [ ] Memory coalescing 이해 및 적용
- [ ] Shared memory로 matrix multiplication 최적화
- [ ] Kernel fusion의 이점 이해
- [ ] Occupancy 최적화
- [ ] Warp divergence 최소화
- [ ] Flash Attention의 tiling 전략 이해

---

## 📚 Optimization Checklist

코드 최적화 순서:
1. ✅ **Profile first!** (어디가 느린지 측정)
2. ✅ **Memory coalescing** (가장 큰 영향)
3. ✅ **Shared memory** (자주 재사용되는 데이터)
4. ✅ **Kernel fusion** (여러 작은 kernel 합치기)
5. ✅ **Occupancy** (thread/block size 조정)
6. ✅ **Algorithm** (Tiling, online algorithms)

---

## ⏭️ 다음 단계

CUDA 최적화를 배웠습니다! 하지만 NPU/TPU는 어떻게 다를까요?

👉 [Day 7-8: NPU/TPU Architecture](./04-npu-tpu-architecture.md)에서 **AI 전용 칩**을 배웁니다!

**"이제 GPU 성능을 최대로 끌어낼 수 있습니다!"** ⚡
