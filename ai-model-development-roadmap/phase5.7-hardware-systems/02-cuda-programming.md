# Day 3-4: CUDA 프로그래밍 기초

## 🎯 목표

**GPU에서 직접 코드 실행하기!**

```cuda
// CPU 코드 (느림)
for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
}

// GPU 코드 (빠름!)
__global__ void add(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// 수천 개 thread가 병렬로 실행!
```

---

## 📖 CUDA 실행 모델

### Thread Hierarchy

```
Grid (전체 작업)
  ├── Block 0
  │     ├── Thread 0
  │     ├── Thread 1
  │     └── ...
  ├── Block 1
  │     ├── Thread 0
  │     └── ...
  └── ...

한 번에 수만 개 thread 실행!
```

### 내장 변수

```cuda
// Block index within grid
blockIdx.x, blockIdx.y, blockIdx.z

// Thread index within block  
threadIdx.x, threadIdx.y, threadIdx.z

// Block dimension (block 안의 thread 개수)
blockDim.x, blockDim.y, blockDim.z

// Grid dimension (grid 안의 block 개수)
gridDim.x, gridDim.y, gridDim.z

// Global thread ID 계산
int tid = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 💻 실습 1: Vector Addition

### CUDA C++ 코드

```cuda
// vector_add.cu
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // 각 thread의 고유 ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // 1. CPU 메모리 할당
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // 2. GPU 메모리 할당
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // 3. CPU → GPU 복사
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // 4. Kernel 실행
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    // 5. GPU → CPU 복사
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // 6. 검증
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %.1f\n", i, h_c[i]);
    }
    
    // 7. 메모리 해제
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    free(h_a); free(h_b); free(h_c);
    
    return 0;
}
```

### 컴파일 & 실행

```bash
# CUDA 코드 컴파일
nvcc vector_add.cu -o vector_add

# 실행
./vector_add

# 출력:
# c[0] = 0.0
# c[1] = 3.0
# c[2] = 6.0
# ...
```

---

## 🐍 PyTorch에서 CUDA 사용

### 기본 사용법

```python
import torch

# 1. Tensor를 GPU로
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()

# 2. GPU에서 연산 (자동!)
z = x @ y  # Matrix multiplication on GPU

# 3. CPU로 가져오기
z_cpu = z.cpu()

# 또는 device 명시
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1000, 1000, device=device)
```

### Custom CUDA Extension

**PyTorch에서 C++/CUDA 코드 사용**

```cpp
// cuda_extension.cu
#include <torch/extension.h>

__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    auto c = torch::zeros_like(a);
    
    int threads = 256;
    int blocks = (a.size(0) + threads - 1) / threads;
    
    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        a.size(0)
    );
    
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add_cuda, "Vector add (CUDA)");
}
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_extension',
    ext_modules=[
        CUDAExtension('cuda_extension', [
            'cuda_extension.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

```bash
# 빌드
python setup.py install

# 사용
import torch
import cuda_extension

a = torch.randn(1000).cuda()
b = torch.randn(1000).cuda()
c = cuda_extension.add(a, b)
```

---

## 🧮 Matrix Multiplication (Naive)

```cuda
__global__ void matmul_naive(float *A, float *B, float *C, int M, int N, int K) {
    // C[M, N] = A[M, K] @ B[K, N]
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 사용
dim3 threads(16, 16);
dim3 blocks((N + 15) / 16, (M + 15) / 16);
matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
```

**문제점**: Global memory 접근이 너무 많음 → 느림!

(다음 Day 5-6에서 최적화!)

---

## 📊 메모리 관리

### 메모리 계층

```
Fastest ↑
    Register (thread 전용, ~KB)
    Shared Memory (block 공유, ~100KB)
    L1 Cache
    L2 Cache
    Global Memory (모든 thread, ~GB)
Slowest ↓
```

### Unified Memory (간편!)

```cuda
// Old way: 복잡
cudaMalloc(&d_a, bytes);
cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

// Unified Memory: 간단!
float *a;
cudaMallocManaged(&a, bytes);  // CPU & GPU 모두 접근 가능!

// GPU kernel
kernel<<<blocks, threads>>>(a);
cudaDeviceSynchronize();

// CPU에서도 바로 사용
printf("%f\n", a[0]);

cudaFree(a);
```

---

## 🎓 학습 목표 체크리스트

- [ ] CUDA thread hierarchy 이해 (Grid/Block/Thread)
- [ ] `blockIdx`, `threadIdx` 계산 이해
- [ ] Vector addition kernel 작성
- [ ] CPU ↔ GPU 메모리 복사 이해
- [ ] PyTorch CUDA extension 빌드
- [ ] Naive matrix multiplication 구현

---

## 📚 참고 자료

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch Custom C++/CUDA Extensions](https://pytorch.org/tutorials/advanced/cpp_extension.html)

---

## ⏭️ 다음 단계

CUDA 기초를 배웠지만, naive 구현은 느립니다!

👉 [Day 5-6: CUDA Optimization](./03-cuda-optimization.md)에서 **10-100배 빠르게** 만듭니다!

**"이제 GPU에서 코드를 실행할 수 있습니다!"** 🚀
