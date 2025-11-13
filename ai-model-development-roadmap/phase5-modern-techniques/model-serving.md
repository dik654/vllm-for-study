# Model Serving: 프로덕션 배포 & 추론 최적화

## 🎯 훈련 vs 추론

```
훈련 (Training):
- Batch 처리 (1000s of examples)
- Backpropagation 필요
- 시간 제약 느슨함 (hours/days)
- GPU 활용률 높음

추론 (Inference):
- 실시간 (ms latency)
- Forward pass만
- 처리량 (throughput) & 지연시간 (latency) 중요
- 비용 효율성 중요
```

---

## 🚀 추론 최적화 기법

### 1. Model Optimization

#### 1.1 Quantization

**INT8 Quantization**
```python
import torch
from torch.quantization import quantize_dynamic

# Dynamic quantization (가장 쉬움)
model_int8 = quantize_dynamic(
    model,
    {nn.Linear, nn.LSTM},  # 어떤 layer를 quantize할지
    dtype=torch.qint8
)

# 결과:
# - 모델 크기: 4x 감소
# - 속도: 2-3x 빠름
# - 정확도: ~1% 감소
```

**Static Quantization (더 빠름)**
```python
from torch.quantization import quantize_static, prepare

# 1. Calibration (representative data로 range 측정)
model.eval()
model_prepared = prepare(model, inplace=False)

with torch.no_grad():
    for batch in calibration_dataloader:
        model_prepared(batch)

# 2. Quantize
model_quantized = quantize_static(model_prepared)

# 결과: Dynamic보다 더 빠름!
```

#### 1.2 Pruning

**Unstructured Pruning**
```python
import torch.nn.utils.prune as prune

# 가중치의 30% 제거 (magnitude 기준)
prune.l1_unstructured(model.linear1, name='weight', amount=0.3)

# Forward pass는 동일하게 작동 (sparse tensor 사용)
output = model(input)

# 영구 적용
prune.remove(model.linear1, 'weight')
```

**Structured Pruning (더 실용적)**
```python
# 전체 channel/filter 제거
prune.ln_structured(
    model.conv1,
    name='weight',
    amount=0.3,
    n=2,  # L2 norm
    dim=0  # Output channels
)
```

#### 1.3 Knowledge Distillation

**Teacher → Student**
```python
# Large teacher model
teacher = LargeModel()
teacher.eval()

# Small student model
student = SmallModel()

# Distillation loss
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.5, temperature=2.0):
    # Soft targets (teacher knowledge)
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    soft_prob = F.log_softmax(student_logits / temperature, dim=-1)
    soft_loss = F.kl_div(soft_prob, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets (ground truth)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Combined
    return alpha * soft_loss + (1 - alpha) * hard_loss

# Training loop
for batch in dataloader:
    inputs, labels = batch
    
    # Teacher prediction (no grad!)
    with torch.no_grad():
        teacher_logits = teacher(inputs)
    
    # Student prediction
    student_logits = student(inputs)
    
    # Loss
    loss = distillation_loss(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()

# Student가 teacher의 50-90% 성능을 10x 작은 크기로!
```

---

### 2. Runtime Optimization

#### 2.1 ONNX Runtime

**PyTorch → ONNX**
```python
import torch.onnx

# Export
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# ONNX Runtime으로 추론
import onnxruntime as ort

session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])

# Inference
outputs = session.run(
    None,
    {'input': input_numpy}
)

# 1.5-2x faster than PyTorch!
```

#### 2.2 TensorRT

**NVIDIA GPU 최적화**
```python
import tensorrt as trt

# ONNX → TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

with trt.Builder(TRT_LOGGER) as builder, \
     builder.create_network() as network, \
     trt.OnnxParser(network, TRT_LOGGER) as parser:
    
    # Config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 precision
    
    # Parse ONNX
    with open("model.onnx", 'rb') as f:
        parser.parse(f.read())
    
    # Build engine
    engine = builder.build_engine(network, config)
    
    # Save
    with open("model.trt", "wb") as f:
        f.write(engine.serialize())

# Inference
# ... (TensorRT inference code)

# 2-5x faster than PyTorch!
# GPU utilization: 90%+
```

---

### 3. LLM Serving (vLLM)

#### 3.1 vLLM 기본 사용

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(model="meta-llama/Llama-2-7b-hf", tensor_parallel_size=2)

# Sampling params
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")

# 처리량: HuggingFace의 10-20x!
```

**왜 빠른가?**
```
1. PagedAttention (KV cache 효율화)
2. Continuous batching (dynamic batching)
3. CUDA/Triton kernels 최적화
4. Tensor parallelism 내장
```

#### 3.2 OpenAI-compatible API Server

```python
# vLLM server 시작
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --tensor-parallel-size 2 \
    --port 8000

# Client
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)

# Streaming
stream = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
)

for chunk in stream:
    print(chunk.choices[0].delta.content, end='', flush=True)
```

---

### 4. Batching Strategies

#### 4.1 Static Batching

```python
# 간단하지만 비효율적
batch_size = 32
batch = []

for request in requests:
    batch.append(request)
    
    if len(batch) == batch_size:
        # Process batch
        outputs = model(batch)
        batch = []

# 문제:
# - 마지막 batch는 작을 수 있음
# - 긴 sequence가 하나라도 있으면 모든 sequence가 기다림
```

#### 4.2 Dynamic Batching

```python
import asyncio
from collections import deque

class DynamicBatcher:
    def __init__(self, model, max_batch_size=32, max_wait_ms=10):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
    
    async def add_request(self, request):
        future = asyncio.Future()
        self.queue.append((request, future))
        return await future
    
    async def process_batches(self):
        while True:
            # Wait for requests
            await asyncio.sleep(self.max_wait_ms / 1000)
            
            if not self.queue:
                continue
            
            # Collect batch
            batch = []
            futures = []
            
            while len(batch) < self.max_batch_size and self.queue:
                request, future = self.queue.popleft()
                batch.append(request)
                futures.append(future)
            
            # Process
            outputs = self.model(batch)
            
            # Return results
            for future, output in zip(futures, outputs):
                future.set_result(output)
```

#### 4.3 Continuous Batching (vLLM)

```
기존 batching:
Batch 1: [Seq A (50 tokens), Seq B (50 tokens)]
→ A가 끝나도 B 때문에 기다림

Continuous batching:
Batch 1: [Seq A, Seq B]
... Seq A finished ...
Batch 2: [Seq B, Seq C]  # A 자리에 C 추가!
... Seq B finished ...
Batch 3: [Seq C, Seq D]

→ GPU utilization 최대화!
```

---

### 5. Caching

#### 5.1 KV Cache

```python
# Naive: 매번 전체 sequence 다시 계산
def generate_naive(model, input_ids):
    for _ in range(max_new_tokens):
        outputs = model(input_ids)  # 전체 다시 계산!
        next_token = outputs[:, -1:].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids

# Optimized: KV cache 사용
def generate_with_cache(model, input_ids):
    past_key_values = None
    
    for _ in range(max_new_tokens):
        if past_key_values is None:
            # First iteration: full forward
            outputs = model(input_ids, use_cache=True)
        else:
            # Subsequent: only last token
            outputs = model(input_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
        
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    
    return input_ids

# 50x faster!
```

#### 5.2 Prompt Caching

```python
# 같은 system prompt 재사용
system_prompt = "You are a helpful assistant."

# Compute once, cache KV
system_kv_cache = compute_kv_cache(system_prompt)

# Requests
for user_message in user_messages:
    # Reuse cached system prompt KV!
    full_kv = concat(system_kv_cache, compute_kv_cache(user_message))
    response = generate_from_kv(full_kv)

# 30-50% latency reduction!
```

---

## 📊 Monitoring & Metrics

### Key Metrics

```python
# 1. Latency (지연시간)
# - P50, P95, P99 latency
# - Time to first token (TTFT)
# - Time per output token (TPOT)

# 2. Throughput (처리량)
# - Requests per second (RPS)
# - Tokens per second (TPS)

# 3. Resource Utilization
# - GPU utilization (aim for >80%)
# - GPU memory usage
# - CPU usage

# 4. Quality
# - Response quality metrics
# - Error rate
```

### Prometheus + Grafana

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUEST_COUNT = Counter('inference_requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('inference_latency_seconds', 'Request latency')
GPU_UTIL = Gauge('gpu_utilization_percent', 'GPU utilization')

# In serving code
@REQUEST_LATENCY.time()
def serve_request(request):
    REQUEST_COUNT.inc()
    result = model(request)
    return result

# Start metrics server
start_http_server(9090)

# Grafana dashboard에서 시각화!
```

---

## 🏗️ 프로덕션 아키텍처

### Load Balancer + Multiple Replicas

```
                        ┌─────────────┐
Internet ──────────────>│ Load Balancer│
                        └──────┬──────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
            ┌───▼────┐    ┌───▼────┐    ┌───▼────┐
            │Model   │    │Model   │    │Model   │
            │Replica1│    │Replica2│    │Replica3│
            │(2xGPU) │    │(2xGPU) │    │(2xGPU) │
            └────────┘    └────────┘    └────────┘
```

**Kubernetes Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-serving
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: vllm
        image: vllm/vllm-openai:latest
        args:
          - --model
          - meta-llama/Llama-2-7b-hf
          - --tensor-parallel-size
          - "2"
        resources:
          limits:
            nvidia.com/gpu: 2
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  type: LoadBalancer
  selector:
    app: llm-serving
  ports:
  - port: 80
    targetPort: 8000
```

---

## 🎓 Best Practices

### 1. 모델 선택
```
- 작은 모델로 시작 (7B → 13B → 70B)
- Distilled 모델 고려 (Llama-7B-chat-distilled)
- Task-specific fine-tuning
```

### 2. 하드웨어 선택
```
Latency-critical: A100 GPU (expensive but fast)
Throughput-focused: L4 GPU (cheap, good throughput)
Budget: T4 GPU (lowest cost)
```

### 3. 최적화 순서
```
1. KV cache (필수)
2. Dynamic batching
3. FP16/BF16
4. Quantization (INT8)
5. TensorRT/ONNX (추가 속도)
6. Tensor parallelism (큰 모델)
```

---

## 📚 참고 자료

- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT](https://developer.nvidia.com/tensorrt)
- [ONNX Runtime](https://onnxruntime.ai/)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)

---

## 🎯 완료 기준

이제:
- ✅ Quantization, Pruning, Distillation 적용 가능
- ✅ vLLM으로 LLM serving
- ✅ TensorRT/ONNX로 최적화
- ✅ Production-ready deployment 이해

**"이제 모델을 프로덕션에 배포할 수 있습니다!"** 🚀
