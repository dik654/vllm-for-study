# Day 7-8: NPU/TPU 아키텍처

## 🎯 왜 GPU만으로는 부족한가?

```
GPU: 범용 병렬 처리 (게임, 과학, AI)
NPU/TPU: AI 전용 최적화 (행렬곱, 컨볼루션만)

→ 전용 칩이 더 효율적!
```

---

## 🧠 NPU (Neural Processing Unit)

### 설계 철학

**AI 워크로드에 특화**
- Matrix multiplication
- Convolution
- Activation functions
- Pooling

**vs GPU**
```
GPU: 
  - 범용성 ↑
  - 프로그래밍 자유도 ↑
  - 전력 효율 ↓

NPU:
  - AI 전용 ↑
  - 저전력 ↑
  - 범용성 ↓
```

### Apple Neural Engine

**M3 Pro (16-core ANE)**
- 추론 전용 (not training)
- 35 TOPS (INT8)
- CoreML과 통합
- 매우 저전력 (~5W)

**사용 예시: CoreML**
```python
import coremltools as ct
import torch

# PyTorch 모델
model = MyModel()

# CoreML로 변환
traced_model = torch.jit.trace(model, example_input)
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape)]
)

# ANE에서 실행!
mlmodel.save("model.mlpackage")

# iOS/macOS에서:
# prediction = model.prediction(input)
# → Neural Engine가 자동으로 실행
```

---

## 🔷 Google TPU (Tensor Processing Unit)

### Systolic Array의 핵심

**CPU/GPU: Von Neumann 아키텍처**
```
1. 메모리에서 데이터 읽기
2. ALU로 계산
3. 메모리에 쓰기
→ Memory bandwidth가 병목!
```

**TPU: Systolic Array**
```
데이터가 PE(Processing Element) 배열을 "흐른다"

Input → [PE] → [PE] → [PE] → Output
          ↓      ↓      ↓
        [PE] → [PE] → [PE]
          ↓      ↓      ↓  
        [PE] → [PE] → [PE]
        
각 PE = MAC (Multiply-Accumulate)
```

### Matrix Multiplication on Systolic Array

```
A × B = C

[a00 a01]   [b00 b01]   [c00 c01]
[a10 a11] × [b10 b11] = [c10 c11]

Systolic array:
  - A의 row가 왼쪽에서 오른쪽으로 흐름
  - B의 column이 위에서 아래로 흐름
  - PE는 누적합 계산

장점:
- 데이터 재사용 극대화
- Memory access 최소화
- 고효율, 저전력
```

### TPU v4 스펙

```
Compute: 275 TFLOPS (BF16)
Memory: 128GB HBM2
Interconnect: 3D torus
Power: ~200W

vs A100 GPU:
  A100: 312 TFLOPS, 80GB, 400W
  TPU v4: 275 TFLOPS, 128GB, 200W
  
→ TPU가 메모리 크고 전력 효율 좋음!
```

### TPU Programming (JAX)

```python
import jax
import jax.numpy as jnp

# JAX는 TPU에 최적화된 NumPy
@jax.jit  # JIT compile for TPU
def matmul(A, B):
    return jnp.dot(A, B)

# TPU에서 실행
A = jnp.ones((1000, 1000))
B = jnp.ones((1000, 1000))
C = matmul(A, B)  # Runs on TPU!

# Multi-TPU (pmap)
@jax.pmap  # Parallel map across TPUs
def parallel_matmul(A, B):
    return jnp.dot(A, B)

# 8 TPU cores
A_sharded = jnp.ones((8, 1000, 1000))
B_sharded = jnp.ones((8, 1000, 1000))
C_sharded = parallel_matmul(A_sharded, B_sharded)
```

---

## 📱 Edge NPUs

### Qualcomm Hexagon NPU

**스마트폰용 AI 칩**
```
Snapdragon 8 Gen 3:
  - Hexagon NPU
  - 14 TOPS (INT8)
  - 매우 저전력 (<2W)
  - On-device AI (no cloud)
```

### Google Edge TPU

**IoT/Embedded용**
```
Coral Dev Board:
  - Edge TPU
  - 4 TOPS (INT8)
  - 2W
  - TensorFlow Lite 실행
```

**사용 예시**
```python
from pycoral.utils import edgetpu
from pycoral.adapters import common

# TPU로 모델 로드
interpreter = edgetpu.make_interpreter('model_edgetpu.tflite')
interpreter.allocate_tensors()

# 추론
common.set_input(interpreter, image)
interpreter.invoke()
output = common.output_tensor(interpreter, 0)
```

---

## 🆚 GPU vs TPU vs NPU 비교

### 훈련 (Training)

```
GPU (A100):
  ✅ 가장 범용적
  ✅ 다양한 프레임워크 지원
  ✅ 디버깅 쉬움
  
TPU (v4):
  ✅ 대규모 훈련 효율적
  ✅ JAX/TensorFlow 최적화
  ❌ PyTorch 지원 제한적
  
NPU:
  ❌ 대부분 추론 전용
```

### 추론 (Inference)

```
GPU:
  ✅ 범용적
  ✅ Batch 처리
  ❌ 전력 소모 큼
  
TPU:
  ✅ 대규모 추론 효율적
  ✅ 전력 효율
  ❌ Edge 배포 어려움
  
NPU:
  ✅ 저전력 (스마트폰, IoT)
  ✅ Realtime 추론
  ❌ 처리량 작음
```

### 비용

```
클라우드 (시간당):
  A100 GPU: ~$3
  TPU v4:   ~$2.5
  
Edge:
  Jetson Nano (GPU): ~$100
  Coral TPU:         ~$60
  
→ 용도에 따라 선택!
```

---

## 🔬 실습: 다양한 하드웨어에서 실행

### 1. GPU (PyTorch)

```python
import torch

model = MyModel().cuda()
input = torch.randn(1, 3, 224, 224).cuda()
output = model(input)
```

### 2. TPU (JAX)

```python
import jax

model = MyModel()
input = jax.random.normal(jax.random.PRNGKey(0), (1, 224, 224, 3))
output = model.apply(params, input)  # Runs on TPU
```

### 3. Apple ANE (CoreML)

```python
import coremltools as ct

mlmodel = ct.models.MLModel("model.mlpackage")
prediction = mlmodel.predict({"input": image})
# → Neural Engine에서 자동 실행
```

### 4. Edge TPU (TFLite)

```python
from pycoral.utils import edgetpu

interpreter = edgetpu.make_interpreter('model_edgetpu.tflite')
# ... (위 코드 참조)
```

---

## 🎓 학습 목표 체크리스트

- [ ] Systolic array 동작 원리 이해
- [ ] GPU vs TPU vs NPU 차이 설명 가능
- [ ] TPU의 전력 효율성 이유 이해
- [ ] Edge NPU의 사용 사례 이해
- [ ] 각 하드웨어에 모델 배포 경험

---

## 📊 언제 어떤 하드웨어?

**대규모 훈련**
→ GPU cluster (A100, H100) 또는 TPU pod

**대규모 추론 (클라우드)**
→ TPU (전력 효율) 또는 GPU (범용성)

**Edge 추론 (스마트폰, IoT)**
→ NPU (Qualcomm, Apple, Edge TPU)

**연구/개발**
→ GPU (디버깅 쉽고 범용적)

---

## ⏭️ 다음 단계

하드웨어 아키텍처를 이해했지만, CUDA는 너무 어렵습니다!

👉 [Day 9: Triton Programming](./05-triton-programming.md)에서 **고수준 GPU 프로그래밍**을 배웁니다!

**"이제 AI 칩들의 차이를 이해합니다!"** 🧠
