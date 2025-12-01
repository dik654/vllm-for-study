# PART 5: GPU/CUDA Level

> Senior/High-Level 엔지니어가 알아야 하는 GPU 최적화 지식

---

## 목차
1. [MatMul과 FlashAttention 커널 구조](#1-matmul과-flashattention-커널-구조)
2. [Block-Sparse Attention GPU 구현](#2-block-sparse-attention-gpu-구현)
3. [Warp-Level Parallelism과 Attention 최적화](#3-warp-level-parallelism과-attention-최적화)
4. [L2 Cache Hit-Rate과 KV Cache Locality](#4-l2-cache-hit-rate과-kv-cache-locality)
5. [CUDA Stream과 Pipeline 병렬화](#5-cuda-stream과-pipeline-병렬화)
6. [Pinned Memory와 Host-Device 전송](#6-pinned-memory와-host-device-전송)
7. [GPU 메모리 Layout과 성능](#7-gpu-메모리-layout과-성능)
8. [FlashAttention과 IO-Bound 문제](#8-flashattention과-io-bound-문제)
9. [GPU Occupancy와 Attention](#9-gpu-occupancy와-attention)
10. [면접 예상 질문 및 답변](#10-면접-예상-질문-및-답변)

---

## 1. MatMul과 FlashAttention 커널 구조

### 1.1 GPU 아키텍처 기초

```
NVIDIA GPU 계층 구조:
┌─────────────────────────────────────────────────────────────────┐
│  GPU Device                                                     │
│  ├── SM (Streaming Multiprocessor) × 108 (A100)                │
│  │   ├── CUDA Cores × 64                                       │
│  │   ├── Tensor Cores × 4                                      │
│  │   ├── Shared Memory: 164KB                                  │
│  │   ├── L1 Cache: 192KB                                       │
│  │   └── Register File: 256KB                                  │
│  │                                                              │
│  ├── L2 Cache: 40MB (전체 공유)                                │
│  │                                                              │
│  └── HBM (High Bandwidth Memory): 80GB                         │
│      └── Bandwidth: 2TB/s                                       │
└─────────────────────────────────────────────────────────────────┘

실행 단위:
Thread → Warp (32 threads) → Block → Grid
```

### 1.2 MatMul 커널 최적화

**Naive MatMul:**
```cuda
// C = A × B, A: [M, K], B: [K, N], C: [M, N]
__global__ void matmul_naive(float* A, float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];  // 글로벌 메모리 접근!
        }
        C[row * N + col] = sum;
    }
}

// 문제: K번의 글로벌 메모리 읽기 (매우 느림)
```

**Tiled MatMul (Shared Memory 활용):**
```cuda
#define TILE_SIZE 32

__global__ void matmul_tiled(float* A, float* B, float* C, int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // 타일을 shared memory로 로드
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_SIZE + threadIdx.y < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Shared memory에서 계산 (빠름!)
        for (int k = 0; k < TILE_SIZE; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// 개선: 글로벌 메모리 접근 TILE_SIZE배 감소
```

### 1.3 Tensor Core 활용

```cuda
// Tensor Core: 4x4 행렬 연산을 한 사이클에 수행
// WMMA (Warp Matrix Multiply Accumulate) API

#include <mma.h>
using namespace nvcuda::wmma;

__global__ void matmul_tensor_core(half* A, half* B, float* C, int M, int K, int N) {
    // Fragment 선언
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;

    fill_fragment(c_frag, 0.0f);

    for (int k = 0; k < K; k += 16) {
        // Fragment 로드
        load_matrix_sync(a_frag, A + row * K + k, K);
        load_matrix_sync(b_frag, B + k * N + col, N);

        // Tensor Core MMA 연산
        mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // 결과 저장
    store_matrix_sync(C + row * N + col, c_frag, N, mem_row_major);
}

// 성능: FP16에서 312 TFLOPS (A100)
```

### 1.4 FlashAttention 커널 구조

**기존 Attention의 문제:**
```python
# 표준 Attention
def standard_attention(Q, K, V):
    S = Q @ K.T           # [N, N] - HBM에 저장
    P = softmax(S)        # [N, N] - HBM에서 읽고 저장
    O = P @ V             # [N, d] - HBM에서 읽기
    return O

# 메모리 접근: O(N² × d) - N이 크면 HBM 대역폭 병목
```

**FlashAttention의 해결:**
```
핵심 아이디어: Tiling + Online Softmax

┌─────────────────────────────────────────────────────────────────┐
│  1. Q, K, V를 블록으로 분할                                     │
│  2. 블록별로 attention 계산                                     │
│  3. Online softmax로 블록 결과 병합                             │
│  4. 중간 결과를 SRAM에만 유지 (HBM 미사용)                      │
└─────────────────────────────────────────────────────────────────┘

메모리 계층:
HBM (느림): Q, K, V 읽기 한 번, O 쓰기 한 번
SRAM (빠름): 모든 중간 계산

→ 메모리 접근: O(N × d) - N²에서 N으로 감소!
```

```cuda
// FlashAttention 의사 코드
__global__ void flash_attention(
    float* Q, float* K, float* V, float* O,
    int N, int d, int Br, int Bc  // Br, Bc: 블록 크기
) {
    // SRAM 할당
    __shared__ float Qi[Br][d];    // Query 블록
    __shared__ float Kj[Bc][d];    // Key 블록
    __shared__ float Vj[Bc][d];    // Value 블록
    __shared__ float Sij[Br][Bc];  // Attention scores
    __shared__ float Oi[Br][d];    // Output 블록

    // Online softmax 상태
    float mi[Br] = {-INFINITY};    // 행별 최댓값
    float li[Br] = {0};            // 행별 합

    // Q 블록 로드
    load_to_sram(Q, Qi, block_row, Br, d);

    // K, V 블록을 순회하며 계산
    for (int j = 0; j < num_kv_blocks; j++) {
        load_to_sram(K, Kj, j, Bc, d);
        load_to_sram(V, Vj, j, Bc, d);

        // S_ij = Q_i × K_j^T (SRAM 내 계산)
        matmul_sram(Qi, Kj, Sij, Br, d, Bc);

        // Online softmax update
        float mi_new = max(mi, rowmax(Sij));
        float li_new = exp(mi - mi_new) * li + rowsum(exp(Sij - mi_new));

        // Output update
        Oi = (li * exp(mi - mi_new) * Oi + exp(Sij - mi_new) @ Vj) / li_new;

        mi = mi_new;
        li = li_new;
    }

    // 최종 출력을 HBM에 저장
    store_to_hbm(Oi, O, block_row, Br, d);
}
```

---

## 2. Block-Sparse Attention GPU 구현

### 2.1 Block-Sparse 패턴

```
Dense Attention:        Block-Sparse Attention:
[████████████████]     [████░░░░░░░░████]
[████████████████]     [████████░░░░░░░░]
[████████████████]     [░░░░████████░░░░]
[████████████████]     [░░░░░░░░████████]

● = 계산하는 블록
░ = 스킵하는 블록

→ 희소성 50%면 연산량 50% 감소
```

### 2.2 Block-Sparse 커널 구조

```cuda
// Block-sparse attention 커널
__global__ void block_sparse_attention(
    float* Q, float* K, float* V, float* O,
    int* block_indices,     // 활성 블록 인덱스
    int num_active_blocks,  // 활성 블록 수
    int block_size
) {
    int block_idx = blockIdx.x;
    if (block_idx >= num_active_blocks) return;

    // 이 스레드 블록이 처리할 Q, K 블록 위치
    int q_block = block_indices[block_idx * 2];
    int k_block = block_indices[block_idx * 2 + 1];

    // 해당 블록만 계산
    __shared__ float Qi[BLOCK_SIZE][HEAD_DIM];
    __shared__ float Ki[BLOCK_SIZE][HEAD_DIM];
    __shared__ float Vi[BLOCK_SIZE][HEAD_DIM];

    // 블록 로드
    load_block(Q, Qi, q_block, block_size);
    load_block(K, Ki, k_block, block_size);
    load_block(V, Vi, k_block, block_size);

    // Attention 계산 (dense와 동일)
    float scores[BLOCK_SIZE];
    compute_attention(Qi, Ki, Vi, scores);

    // 출력에 atomic 누적
    atomic_add_output(O, scores, q_block);
}
```

### 2.3 희소 패턴 종류

```
1. Fixed Pattern (Longformer):
   [████████░░░░░░░░████]  ← Local + Global tokens
   [████████████░░░░░░░░]
   [░░░░████████████░░░░]
   [████░░░░░░░░████████]

2. Strided Pattern (Sparse Transformer):
   [████░░░░████░░░░████]  ← 일정 간격으로 attend
   [░░░░████░░░░████░░░░]
   [████░░░░████░░░░████]

3. Dynamic Pattern (BigBird):
   [████░░██░░░░░░██████]  ← Random + Local + Global
   다양한 희소 패턴 조합

4. Learned Pattern:
   - 데이터 기반으로 중요한 블록만 선택
   - Routing 네트워크 필요
```

### 2.4 성능 분석

```
Dense Attention:
FLOPS = 2 × N² × d
Memory = O(N²)

Block-Sparse (희소성 s):
FLOPS = 2 × N² × d × (1 - s)
Memory = O(N² × (1 - s))

예시 (N=4096, d=128, s=75%):
Dense:  2 × 16M × 128 = 4B FLOPS
Sparse: 4B × 0.25 = 1B FLOPS

→ 4배 연산 감소!

실제 속도:
- 오버헤드(인덱싱 등)로 이론치보다 낮음
- 보통 2-3배 속도 향상 달성
```

---

## 3. Warp-Level Parallelism과 Attention 최적화

### 3.1 Warp 기초

```
Warp = 32개 스레드의 SIMT 그룹
- 모든 스레드가 같은 명령어 실행
- 분기 시 순차 실행 (Divergence)

┌─────────────────────────────────────────────────────────────────┐
│  Warp 0: [T0][T1][T2]...[T31]  → 동시 실행                     │
│  Warp 1: [T32][T33]...[T63]   → 동시 실행                      │
│  ...                                                            │
│                                                                 │
│  Warp Scheduler가 warp 단위로 명령어 발행                      │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Warp-Level Primitives

```cuda
// Warp shuffle: 레지스터 간 데이터 교환 (shared memory 불필요)

// __shfl_sync: 특정 레인에서 값 가져오기
float val = __shfl_sync(0xffffffff, my_val, src_lane);

// __shfl_xor_sync: XOR 패턴으로 교환 (reduction에 유용)
float val = __shfl_xor_sync(0xffffffff, my_val, mask);

// __shfl_down_sync: 아래 레인에서 값 가져오기
float val = __shfl_down_sync(0xffffffff, my_val, delta);

// Warp reduction 예시
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
```

### 3.3 Attention에서의 Warp 최적화

```cuda
// FlashAttention에서 warp-level softmax
__device__ void warp_softmax(float* scores, int len) {
    int lane = threadIdx.x % 32;

    // 1. Warp 내 최댓값 찾기
    float max_val = scores[lane];
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));

    // 2. exp(x - max) 계산
    float exp_val = expf(scores[lane] - max_val);

    // 3. Warp 내 합계
    float sum = exp_val;
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_xor_sync(0xffffffff, sum, offset);

    // 4. 정규화
    scores[lane] = exp_val / sum;
}

// 장점: shared memory 동기화 없이 빠른 softmax
```

### 3.4 Warp Divergence 최소화

```cuda
// 나쁜 예: Warp Divergence
__global__ void attention_bad(float* scores, int seq_len) {
    int tid = threadIdx.x;

    if (tid < seq_len) {  // Warp 내 스레드마다 다른 경로!
        scores[tid] = process(scores[tid]);
    }
}

// 좋은 예: Predication 활용
__global__ void attention_good(float* scores, int seq_len) {
    int tid = threadIdx.x;

    // 모든 스레드가 같은 명령 실행, 결과만 마스킹
    float result = process(scores[tid]);
    if (tid < seq_len) {
        scores[tid] = result;
    }
}
```

---

## 4. L2 Cache Hit-Rate과 KV Cache Locality

### 4.1 GPU 캐시 계층

```
┌─────────────────────────────────────────────────────────────────┐
│  레지스터: 256KB/SM, ~1 cycle latency                          │
│      ↓                                                          │
│  Shared Memory: 164KB/SM, ~30 cycles                           │
│      ↓                                                          │
│  L1 Cache: 192KB/SM, ~30 cycles (Shared와 공유)                │
│      ↓                                                          │
│  L2 Cache: 40MB (전체 GPU 공유), ~200 cycles                   │
│      ↓                                                          │
│  HBM: 80GB, ~400-600 cycles                                    │
└─────────────────────────────────────────────────────────────────┘

L2 Cache 중요성:
- HBM 접근 대비 2-3배 빠름
- 자주 접근하는 데이터 캐싱
- KV Cache가 L2에 맞으면 성능 크게 향상
```

### 4.2 KV Cache Locality

```
KV Cache 접근 패턴:

Decode 단계:
Query: 1개 토큰의 Q
Key/Value: 모든 이전 토큰의 KV

┌─────────────────────────────────────────────────────────────────┐
│  배치 처리 시 KV Cache 접근:                                    │
│                                                                 │
│  Req1: [K0, K1, K2, ..., K100]  ← 연속 읽기 (좋음)             │
│  Req2: [K0, K1, K2, ..., K50]   ← 연속 읽기 (좋음)             │
│  Req3: [K0, K1, K2, ..., K200]  ← 연속 읽기 (좋음)             │
│                                                                 │
│  하지만 요청 간에는 비연속적 접근 발생                         │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 L2 Hit Rate 최적화

```python
# 배치 크기와 L2 캐시 관계

def analyze_l2_hit_rate(batch_size, seq_len, kv_per_token, l2_size):
    """
    L2 캐시 히트율 분석
    """
    # 한 배치의 KV 캐시 크기
    kv_cache_size = batch_size * seq_len * kv_per_token

    # L2에 얼마나 들어가는지
    if kv_cache_size <= l2_size:
        # 전체가 L2에 캐싱됨
        hit_rate = 0.9  # 높은 히트율
    else:
        # 일부만 캐싱, LRU에 따라 교체
        hit_rate = l2_size / kv_cache_size

    return hit_rate

# 예시 (A100)
l2_size = 40 * 1024 * 1024  # 40MB
kv_per_token = 0.5 * 1024 * 1024  # 0.5MB (LLaMA-7B)

# 작은 배치
hit_rate_small = analyze_l2_hit_rate(4, 2048, kv_per_token, l2_size)
# = 4 * 2048 * 0.5MB = 4GB >> 40MB → 낮은 히트율

# 실제로는 시간적 지역성 덕분에 더 나음
```

### 4.4 메모리 접근 패턴 최적화

```cuda
// Coalesced Memory Access (병합 메모리 접근)

// 나쁜 예: Strided access
__global__ void bad_access(float* data, int stride) {
    int tid = threadIdx.x;
    float val = data[tid * stride];  // 스레드마다 다른 캐시 라인
}

// 좋은 예: Coalesced access
__global__ void good_access(float* data) {
    int tid = threadIdx.x;
    float val = data[tid];  // 연속된 32개 float = 1개 트랜잭션
}

// KV Cache에서:
// [batch, seq, heads, dim] → heads 차원 연속이 좋음
// [batch, heads, seq, dim] → seq 차원 연속이 더 좋음 (attention 패턴)
```

---

## 5. CUDA Stream과 Pipeline 병렬화

### 5.1 CUDA Stream 기초

```cuda
// Stream: GPU 연산의 순차 큐

cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// 같은 stream 내: 순차 실행
kernel1<<<grid, block, 0, stream1>>>();
kernel2<<<grid, block, 0, stream1>>>();  // kernel1 후에 실행

// 다른 stream: 병렬 가능
kernel1<<<grid, block, 0, stream1>>>();
kernel2<<<grid, block, 0, stream2>>>();  // 동시 실행 가능

cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

### 5.2 연산-통신 오버랩

```
┌─────────────────────────────────────────────────────────────────┐
│  Without Overlap:                                               │
│  [Copy H→D][Compute][Copy D→H][Copy H→D][Compute][Copy D→H]    │
│                                                                 │
│  With Overlap:                                                  │
│  [Copy H→D 1][Compute 1  ][Copy D→H 1]                        │
│             [Copy H→D 2][Compute 2  ][Copy D→H 2]              │
│                        [Copy H→D 3][Compute 3  ][Copy D→H 3]   │
│                                                                 │
│  → 파이프라인으로 총 시간 단축                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 LLM 추론에서의 Stream 활용

```python
# vLLM에서의 multi-stream 활용 (개념적)

class MultiStreamExecutor:
    def __init__(self, num_streams=2):
        self.streams = [torch.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0

    def execute_batch(self, batch):
        stream = self.streams[self.current_stream]

        with torch.cuda.stream(stream):
            # 입력 전송
            inputs = batch.to('cuda', non_blocking=True)

            # 모델 실행
            outputs = self.model(inputs)

            # 출력 전송
            outputs = outputs.to('cpu', non_blocking=True)

        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return outputs

# 이점:
# - 배치 N의 계산 중에 배치 N+1의 입력 전송
# - GPU 유휴 시간 최소화
```

### 5.4 Event 동기화

```cuda
cudaEvent_t event;
cudaEventCreate(&event);

// Stream 1에서 연산 후 이벤트 기록
kernel1<<<..., stream1>>>();
cudaEventRecord(event, stream1);

// Stream 2에서 해당 이벤트 대기
cudaStreamWaitEvent(stream2, event, 0);
kernel2<<<..., stream2>>>();  // kernel1 완료 후 실행

cudaEventDestroy(event);
```

---

## 6. Pinned Memory와 Host-Device 전송

### 6.1 Pinned Memory란?

```
일반 메모리 (Pageable):
┌─────────────────────────────────────────────────────────────────┐
│  CPU Memory ──(page fault)──→ 임시 버퍼 ──(DMA)──→ GPU        │
│                               (pinned)                          │
│  • OS가 페이지를 스왑할 수 있음                                │
│  • DMA 전송 전 pinned 메모리로 복사 필요                       │
│  • 느림                                                        │
└─────────────────────────────────────────────────────────────────┘

Pinned Memory (Page-locked):
┌─────────────────────────────────────────────────────────────────┐
│  CPU Memory (pinned) ────────(DMA)────────→ GPU                │
│                                                                 │
│  • OS가 절대 스왑하지 않음                                     │
│  • GPU가 직접 DMA 접근 가능                                    │
│  • 빠름 (2-3배)                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 PyTorch에서 Pinned Memory

```python
# Pinned memory 할당
tensor_pinned = torch.empty(1000, 1000, pin_memory=True)

# 일반 텐서를 pinned로
tensor = torch.randn(1000, 1000)
tensor_pinned = tensor.pin_memory()

# non_blocking 전송 (비동기)
gpu_tensor = tensor_pinned.to('cuda', non_blocking=True)

# DataLoader에서 활용
dataloader = DataLoader(
    dataset,
    batch_size=32,
    pin_memory=True,  # 자동으로 pinned memory 사용
    num_workers=4
)
```

### 6.3 전송 대역폭 비교

```
PCIe 4.0 x16:
- 이론적 대역폭: 32 GB/s (양방향)
- 실제 H→D: ~25 GB/s
- 실제 D→H: ~25 GB/s

Pinned vs Pageable:
┌─────────────────────────────────────────────────────────────────┐
│  크기      │  Pageable   │  Pinned     │  속도 향상           │
├─────────────────────────────────────────────────────────────────┤
│  1 MB     │  ~100 μs    │  ~40 μs     │  2.5x                │
│  10 MB    │  ~1 ms      │  ~0.4 ms    │  2.5x                │
│  100 MB   │  ~10 ms     │  ~4 ms      │  2.5x                │
│  1 GB     │  ~100 ms    │  ~40 ms     │  2.5x                │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 KV Cache Swap에서의 활용

```python
# KV Cache swap-out 최적화
class OptimizedSwap:
    def __init__(self):
        # Pinned memory 버퍼 사전 할당
        self.cpu_buffer = torch.empty(
            MAX_SWAP_SIZE,
            dtype=torch.float16,
            pin_memory=True
        )

    def swap_out(self, gpu_blocks, block_ids):
        # 비동기 전송
        for i, block_id in enumerate(block_ids):
            gpu_block = self.gpu_cache[block_id]
            cpu_block = self.cpu_buffer[i * BLOCK_SIZE:(i+1) * BLOCK_SIZE]

            # non_blocking=True로 비동기 복사
            cpu_block.copy_(gpu_block, non_blocking=True)

        # 필요시 동기화
        torch.cuda.synchronize()

    def swap_in(self, block_ids):
        for i, block_id in enumerate(block_ids):
            cpu_block = self.cpu_buffer[i * BLOCK_SIZE:(i+1) * BLOCK_SIZE]
            gpu_block = self.gpu_cache[block_id]

            gpu_block.copy_(cpu_block, non_blocking=True)
```

---

## 7. GPU 메모리 Layout과 성능

### 7.1 메모리 Layout 기초

```
Row-Major (C-style):         Column-Major (Fortran-style):
A[0,0] A[0,1] A[0,2]         A[0,0] A[1,0] A[2,0]
A[1,0] A[1,1] A[1,2]         A[0,1] A[1,1] A[2,1]
A[2,0] A[2,1] A[2,2]         A[0,2] A[1,2] A[2,2]

메모리: [A00,A01,A02,A10...]  메모리: [A00,A10,A20,A01...]
```

### 7.2 Attention에서의 최적 Layout

```
KV Cache Layout 옵션:

옵션 1: [batch, layer, head, seq, dim]
- 장점: 배치/레이어 병렬화에 좋음
- 단점: attention 시 seq 차원 접근 비연속

옵션 2: [layer, batch, head, seq, dim]
- 장점: 레이어별 처리에 좋음
- 단점: 배치 간 비연속

옵션 3: [layer, batch, seq, head, dim]  ← vLLM 선택
- 장점: PagedAttention에 최적화
- seq 차원 연속으로 블록 읽기 효율적
```

### 7.3 메모리 정렬

```cuda
// 메모리 정렬의 중요성

// 비정렬 접근 (느림)
struct BadStruct {
    char a;      // 1 byte
    float b;     // 4 bytes, but misaligned!
    char c;      // 1 byte
};  // 총 6 bytes, but 정렬 문제

// 정렬된 접근 (빠름)
struct GoodStruct {
    float b;     // 4 bytes, aligned
    char a;      // 1 byte
    char c;      // 1 byte
    char pad[2]; // padding
};  // 총 8 bytes, 정렬됨

// GPU에서 128-byte 정렬이 최적
// cuBLAS, cuDNN 모두 정렬된 텐서 가정
```

### 7.4 텐서 Contiguity

```python
# Contiguous tensor의 중요성

# 연속적
a = torch.randn(4, 4)
print(a.is_contiguous())  # True

# 비연속적 (전치 후)
b = a.t()
print(b.is_contiguous())  # False!

# CUDA 커널은 대부분 contiguous 가정
# 비연속 텐서 → 자동 복사 또는 에러

# 해결: contiguous() 호출
b_contig = b.contiguous()  # 메모리 복사 발생
```

---

## 8. FlashAttention과 IO-Bound 문제

### 8.1 IO-Bound vs Compute-Bound

```
┌─────────────────────────────────────────────────────────────────┐
│  Compute-Bound:                                                 │
│  - GPU 연산 능력이 병목                                        │
│  - 해결: 더 빠른 GPU, 연산 최적화                              │
│  - 예: 대규모 행렬 곱, 큰 배치 학습                            │
│                                                                 │
│  IO-Bound (Memory-Bound):                                       │
│  - 메모리 대역폭이 병목                                        │
│  - 해결: 메모리 접근 최소화, 캐시 활용                         │
│  - 예: Attention (중간 결과가 큼), 작은 배치 추론              │
└─────────────────────────────────────────────────────────────────┘

판단 기준: Arithmetic Intensity (AI)
AI = FLOPS / Bytes accessed

A100 균형점: ~200 FLOPS/byte
AI < 200 → IO-bound
AI > 200 → Compute-bound
```

### 8.2 표준 Attention의 IO 문제

```
표준 Attention:
S = Q × K^T     (N×d × d×N = N×N)
P = softmax(S)  (N×N → N×N)
O = P × V       (N×N × N×d = N×d)

메모리 접근:
1. Q 읽기: N × d
2. K 읽기: N × d
3. S 쓰기: N × N  ← 큰 중간 결과!
4. S 읽기 (softmax): N × N
5. P 쓰기: N × N
6. P 읽기: N × N
7. V 읽기: N × d
8. O 쓰기: N × d

총 IO: O(N² + Nd)

N=2048, d=128:
IO = 2048² × 2 (읽기+쓰기) × 4 bytes = 32MB
단일 head, 단일 layer만으로!
```

### 8.3 FlashAttention의 IO 최적화

```
FlashAttention:
- S, P를 HBM에 저장하지 않음!
- SRAM(shared memory)에서 블록 단위 계산
- Online softmax로 블록 결과 병합

메모리 접근:
1. Q 읽기: N × d (한 번)
2. K 읽기: N × d (한 번)
3. V 읽기: N × d (한 번)
4. O 쓰기: N × d (한 번)

총 IO: O(Nd)  ← N² 제거!

N=2048, d=128:
IO = 4 × 2048 × 128 × 2 bytes = 2MB
→ 16배 IO 감소!
```

### 8.4 FlashAttention 성능

```
┌─────────────────────────────────────────────────────────────────┐
│  seq_len  │ Standard  │ FlashAttn │ Speedup │ Memory     │
├─────────────────────────────────────────────────────────────────┤
│  512      │  1.0 ms   │  0.4 ms   │  2.5x   │ 8x less    │
│  1024     │  4.0 ms   │  0.9 ms   │  4.4x   │ 16x less   │
│  2048     │  16 ms    │  2.0 ms   │  8x     │ 32x less   │
│  4096     │  64 ms    │  4.5 ms   │  14x    │ 64x less   │
│  8192     │  OOM      │  10 ms    │  ∞      │ N/A        │
└─────────────────────────────────────────────────────────────────┘

긴 시퀀스에서 더 큰 효과!
```

---

## 9. GPU Occupancy와 Attention

### 9.1 Occupancy란?

```
Occupancy = Active Warps / Maximum Warps per SM

높은 Occupancy:
- 메모리 지연 시 다른 warp 실행
- 지연 숨기기 (latency hiding)
- 일반적으로 성능에 좋음

낮은 Occupancy 원인:
1. 레지스터 과다 사용
2. Shared memory 과다 사용
3. 블록 크기 부적절
```

### 9.2 Occupancy 계산

```cuda
// CUDA Occupancy Calculator 활용

#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    int blockSize = 256;
    int minGridSize, optimalBlockSize;

    // 최적 블록 크기 자동 계산
    cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &optimalBlockSize,
        my_kernel,
        0,  // dynamic shared memory
        0   // block size limit
    );

    // Occupancy 계산
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks,
        my_kernel,
        blockSize,
        0  // dynamic shared memory
    );

    float occupancy = (maxActiveBlocks * blockSize)
                    / (float)prop.maxThreadsPerMultiProcessor;

    printf("Occupancy: %.2f%%\n", occupancy * 100);
}
```

### 9.3 Attention 커널의 Occupancy 최적화

```cuda
// FlashAttention에서의 trade-off

// 높은 occupancy 설정
__global__ void attention_high_occ() {
    // 작은 shared memory
    __shared__ float smem[1024];  // 4KB

    // → 많은 블록 동시 실행 가능
    // → 하지만 타일 크기 제한
}

// 낮은 occupancy, 높은 효율
__global__ void attention_low_occ() {
    // 큰 shared memory
    __shared__ float smem[16384];  // 64KB

    // → 적은 블록 동시 실행
    // → 하지만 큰 타일로 HBM 접근 감소
    // → FlashAttention이 이 전략 선택
}

// 최적점: Occupancy보다 메모리 효율이 더 중요
// FlashAttention은 ~50% occupancy에서 최적 성능
```

### 9.4 Occupancy vs Performance

```
┌─────────────────────────────────────────────────────────────────┐
│  Occupancy %  │  SRAM Size  │  Tile Size  │  Performance      │
├─────────────────────────────────────────────────────────────────┤
│  100%         │  16 KB      │  32×32      │  baseline         │
│  75%          │  32 KB      │  64×64      │  1.2x             │
│  50%          │  64 KB      │  128×64     │  1.5x  ← 최적     │
│  25%          │  128 KB     │  256×64     │  1.3x             │
└─────────────────────────────────────────────────────────────────┘

결론: Attention은 Memory-bound이므로
높은 occupancy보다 효율적인 메모리 사용이 더 중요
```

---

## 10. 면접 예상 질문 및 답변

### Q1: FlashAttention이 기존 attention보다 빠른 이유는?

**답변:**
기존 attention은 **IO-bound**입니다. 중간 결과(N×N attention 행렬)를 HBM에 저장하고 다시 읽어야 해서 메모리 대역폭이 병목입니다.

FlashAttention은:
1. **Tiling**: Q, K, V를 블록으로 나눠 SRAM에서 처리
2. **Online Softmax**: 중간 attention 행렬을 HBM에 저장하지 않음
3. **IO 복잡도**: O(N²) → O(N)으로 감소

긴 시퀀스(4K+)에서 10배 이상 빠르고, 메모리도 훨씬 적게 사용합니다.

### Q2: Tensor Core와 CUDA Core의 차이는?

**답변:**
**CUDA Core:** 범용 연산, 스칼라 FP32/FP64 연산
- A100: 6912개, ~19 TFLOPS (FP32)

**Tensor Core:** 행렬 연산 전용, 4×4 행렬 연산을 한 사이클에
- A100: 432개, ~312 TFLOPS (FP16)
- WMMA API로 16×16×16 행렬 곱 가속

LLM의 대부분 연산(MatMul)이 Tensor Core에서 실행되어 높은 throughput 달성.

### Q3: Pinned Memory를 사용하는 이유는?

**답변:**
일반 메모리는 DMA 전송 전에 pinned 버퍼로 복사가 필요합니다.

```
Pageable: CPU Memory → Pinned Buffer → GPU (2단계)
Pinned: CPU Memory ─────────────────→ GPU (1단계)
```

Pinned memory는:
1. OS가 스왑하지 않아 GPU가 직접 DMA 접근
2. **2-3배** 빠른 전송
3. **Non-blocking** 전송 가능 (연산과 전송 오버랩)

KV Cache swap 시 특히 중요합니다.

### Q4: GPU Occupancy가 항상 높을수록 좋은가요?

**답변:**
아닙니다. **Attention은 Memory-bound**이므로 Occupancy보다 **메모리 효율**이 더 중요합니다.

```
높은 Occupancy (100%):
- 작은 shared memory → 작은 타일
- HBM 접근 많음

낮은 Occupancy (50%):
- 큰 shared memory → 큰 타일
- HBM 접근 적음 → 더 빠름!
```

FlashAttention은 약 50% occupancy에서 최적 성능을 보입니다.

### Q5: Warp Divergence란 무엇이고 어떻게 피하나요?

**답변:**
Warp(32 스레드)는 SIMT로 같은 명령을 실행합니다. if문으로 분기하면 두 경로를 **순차 실행**합니다.

```cuda
// Divergence 발생
if (threadIdx.x < 16) {
    path_A();  // 먼저 실행
} else {
    path_B();  // 나중에 실행
}
// → 2배 느림!
```

**해결책:**
1. Warp 단위로 조건 통일
2. Predication 활용 (조건부 결과만 마스킹)
3. 데이터 재배치로 분기 최소화

### Q6: KV Cache의 최적 메모리 layout은?

**답변:**
vLLM은 `[layer, batch, seq, head, dim]` 레이아웃을 사용합니다.

이유:
1. **PagedAttention**: 블록 단위(seq 차원) 접근에 최적화
2. **Coalesced access**: seq 차원이 연속이면 warp의 메모리 접근이 병합됨
3. **Cache locality**: 한 요청의 KV가 연속 메모리에 위치

```
접근 패턴: attention(Q[i], K[:seq_len], V[:seq_len])
→ seq 차원 연속이 유리
```

---

## 참고 자료

- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS: CUDA Templates for Linear Algebra](https://github.com/NVIDIA/cutlass)
- vLLM 소스코드:
  - `vllm/attention/ops/` - Attention 커널 구현
  - `csrc/` - CUDA 커널 소스
