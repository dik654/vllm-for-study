# Week 12: Quantization & Production Deployment

## 🎯 목표

**모델을 압축하고 프로덕션 환경에 최적화하여 배포하기!**

```python
# 목표:
- 모델 크기: 13GB → 3GB (4x 압축)
- 추론 속도: 2x 향상
- 품질 손실: < 1%

# 방법:
- Quantization (FP16 → INT8 → INT4)
- 모델 변환 (ONNX, TensorRT)
- Serving infrastructure
```

---

## 📊 Part 1: Quantization 기초

### 1.1 왜 Quantization?

```python
# FP32: 32 bits per parameter
# FP16: 16 bits (2x compression)
# INT8: 8 bits (4x compression)
# INT4: 4 bits (8x compression)

# LLaMA-7B example:
# FP32: 7B × 4 bytes = 28GB
# FP16: 7B × 2 bytes = 14GB
# INT8: 7B × 1 byte = 7GB
# INT4: 7B × 0.5 bytes = 3.5GB
```

### 1.2 Quantization 수식

**Affine Quantization**:

```python
# Quantize
Q = round((X - zero_point) / scale)

# Dequantize
X ≈ scale × Q + zero_point

# Example:
# FP32 range: [-1.0, 1.0]
# INT8 range: [-128, 127]

scale = (max_val - min_val) / 255
zero_point = -128 - min_val / scale
```

### 1.3 구현

```python
import torch

def quantize_tensor(tensor, num_bits=8):
    """
    Symmetric quantization (zero_point = 0)

    Args:
        tensor: FP32 tensor
        num_bits: Target bit-width

    Returns:
        quantized: INT tensor
        scale: Scaling factor
    """
    # Compute range
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1

    # Compute scale (symmetric)
    max_val = tensor.abs().max()
    scale = max_val / qmax

    # Quantize
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax).to(torch.int8)

    return quantized, scale

def dequantize_tensor(quantized, scale):
    """
    Dequantize back to FP32
    """
    return quantized.to(torch.float32) * scale

# Example
x = torch.randn(100, 100)
print(f"Original: {x.element_size() * x.numel() / 1024:.2f} KB")

q, scale = quantize_tensor(x, num_bits=8)
print(f"Quantized: {q.element_size() * q.numel() / 1024:.2f} KB")

x_deq = dequantize_tensor(q, scale)
error = (x - x_deq).abs().mean()
print(f"Quantization error: {error:.6f}")
```

---

## 🔧 Part 2: Post-Training Quantization (PTQ)

### 2.1 Dynamic Quantization

**가장 간단! Weights만 quantize, activations는 runtime에**

```python
import torch
from transformers import AutoModelForCausalLM

# Load model
model = AutoModelForCausalLM.from_pretrained('gpt2')

# Dynamic quantization
model_quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8
)

# Save
torch.save(model_quantized.state_dict(), 'model_int8.pt')

# Compare size
import os
original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
quantized_size = os.path.getsize('model_int8.pt') / 1024**2

print(f"Original: {original_size:.2f} MB")
print(f"Quantized: {quantized_size:.2f} MB")
print(f"Compression: {original_size / quantized_size:.2f}x")
```

### 2.2 Static Quantization

**Weights + Activations 모두 quantize (더 빠름!)**

```python
# Calibration 필요!

# 1. Prepare model
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # x86
# model.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # ARM

# 2. Fuse modules
model_fused = torch.quantization.fuse_modules(
    model,
    [['conv', 'bn', 'relu']]  # Fuse Conv-BN-ReLU
)

# 3. Prepare for quantization
model_prepared = torch.quantization.prepare(model_fused)

# 4. Calibrate (run through representative data)
with torch.no_grad():
    for batch in calibration_loader:
        model_prepared(batch)

# 5. Convert to quantized model
model_quantized = torch.quantization.convert(model_prepared)

# Now inference is INT8!
output = model_quantized(input)
```

---

## 🎯 Part 3: Advanced Quantization

### 3.1 GPTQ (Layer-wise Quantization)

**핵심 아이디어**: Layer-by-layer optimal quantization

```python
# Auto-GPTQ library
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# Quantization config
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4-bit
    group_size=128,  # Group size for quantization
    desc_act=False,  # Describe activation order
)

# Load model
model = AutoGPTQForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf',
    quantize_config=quantize_config
)

# Quantize (needs calibration data)
model.quantize(calibration_dataset)

# Save
model.save_quantized('./gptq_model')

# Load quantized model
model = AutoGPTQForCausalLM.from_quantized('./gptq_model', device='cuda:0')

# Inference
output = model.generate(**inputs, max_new_tokens=100)
```

### 3.2 AWQ (Activation-aware Weight Quantization)

**핵심 아이디어**: Protect salient weights based on activation magnitudes

```python
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Load model
model = AutoAWQForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

# Quantization config
quant_config = {
    'zero_point': True,
    'q_group_size': 128,
    'w_bit': 4,
    'version': 'GEMM'
}

# Quantize
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_text  # Text samples
)

# Save
model.save_quantized('./awq_model')

# Load
model = AutoAWQForCausalLM.from_quantized('./awq_model', fuse_layers=True)

# AWQ typically gives better quality than GPTQ at same bit-width!
```

### 3.3 GGUF (llama.cpp format)

**CPU 최적화 모델 format**

```bash
# Convert to GGUF
python convert-hf-to-gguf.py \
    --model meta-llama/Llama-2-7b-hf \
    --outfile llama-2-7b.gguf \
    --outtype q4_0  # 4-bit quantization

# Quantization types:
# - q4_0: 4-bit, fast
# - q4_1: 4-bit, better quality
# - q5_0: 5-bit
# - q5_1: 5-bit, better quality
# - q8_0: 8-bit
```

```python
# Use with llama-cpp-python
from llama_cpp import Llama

# Load model
llm = Llama(
    model_path='./llama-2-7b.gguf',
    n_ctx=2048,  # Context length
    n_threads=8,  # CPU threads
    n_gpu_layers=32  # Layers to offload to GPU
)

# Generate
output = llm(
    "What is machine learning?",
    max_tokens=100,
    temperature=0.7
)

print(output['choices'][0]['text'])

# Benefits:
# - Runs on CPU!
# - Low memory usage
# - Fast inference
```

---

## 🏗️ Part 4: 모델 변환

### 4.1 ONNX (Open Neural Network Exchange)

**프레임워크 독립적 format**

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# Dummy input
dummy_input = tokenizer("This is a test", return_tensors='pt')

# Export to ONNX
torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    'model.onnx',
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch', 1: 'sequence'},
        'attention_mask': {0: 'batch', 1: 'sequence'},
        'output': {0: 'batch', 1: 'sequence'}
    },
    opset_version=14
)

# ONNX Runtime inference
import onnxruntime as ort

session = ort.InferenceSession('model.onnx', providers=['CUDAExecutionProvider'])

# Prepare input
ort_inputs = {
    'input_ids': dummy_input['input_ids'].numpy(),
    'attention_mask': dummy_input['attention_mask'].numpy()
}

# Run
ort_outputs = session.run(None, ort_inputs)

print(f"ONNX output shape: {ort_outputs[0].shape}")
```

### 4.2 TensorRT (NVIDIA GPU)

**NVIDIA GPU 최적화**

```python
# Using ONNX → TensorRT

# 1. Export to ONNX (as above)

# 2. Convert to TensorRT
import tensorrt as trt

# Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Builder
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# Parser
parser = trt.OnnxParser(network, TRT_LOGGER)

# Parse ONNX
with open('model.onnx', 'rb') as f:
    parser.parse(f.read())

# Config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

# FP16
if builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)

# INT8 (needs calibration)
# config.set_flag(trt.BuilderFlag.INT8)

# Build engine
engine = builder.build_serialized_network(network, config)

# Save
with open('model.trt', 'wb') as f:
    f.write(engine)

# Inference
# (use TensorRT runtime)
```

---

## 🚀 Part 5: Model Serving

### 5.1 FastAPI REST API

```python
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI()

# Load model once at startup
@app.on_event("startup")
async def load_model():
    global model
    model = pipeline(
        'text-generation',
        model='gpt2',
        device=0 if torch.cuda.is_available() else -1
    )

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerationRequest):
    """Generate text"""
    output = model(
        request.prompt,
        max_length=request.max_length,
        temperature=request.temperature,
        do_sample=True
    )

    return {
        "generated_text": output[0]['generated_text'],
        "prompt": request.prompt
    }

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}

# Run:
# uvicorn api:app --host 0.0.0.0 --port 8000
```

### 5.2 vLLM (High-throughput Inference)

**PagedAttention으로 처리량 10-20x 향상!**

```python
from vllm import LLM, SamplingParams

# Load model
llm = LLM(
    model='meta-llama/Llama-2-7b-hf',
    tensor_parallel_size=2,  # Multi-GPU
    dtype='float16'
)

# Sampling params
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=100
)

# Batch inference
prompts = [
    "What is AI?",
    "Explain quantum computing",
    "How does photosynthesis work?"
]

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")
    print()

# Benefits:
# - 10-20x higher throughput vs HuggingFace
# - Continuous batching
# - Efficient KV cache management
# - PagedAttention (no memory waste)
```

### 5.3 TorchServe

**PyTorch 공식 serving framework**

```bash
# 1. Create model archive
torch-model-archiver \
    --model-name my_model \
    --version 1.0 \
    --model-file model.py \
    --serialized-file model.pt \
    --handler text_classifier \
    --export-path model_store

# 2. Start server
torchserve \
    --start \
    --ncs \
    --model-store model_store \
    --models my_model.mar

# 3. Inference
curl -X POST http://localhost:8080/predictions/my_model \
    -H "Content-Type: application/json" \
    -d '{"text": "This is a test"}'

# 4. Management API
curl http://localhost:8081/models
```

---

## 📊 Part 6: 성능 최적화

### 6.1 Batching Strategies

```python
# Static batching (inefficient)
def static_batch_inference(requests, batch_size=8):
    """Wait until batch is full"""
    batch = []

    for req in requests:
        batch.append(req)

        if len(batch) == batch_size:
            # Process batch
            process_batch(batch)
            batch = []

    # Process remaining
    if batch:
        process_batch(batch)

# Continuous batching (vLLM style)
def continuous_batching(requests):
    """Add/remove requests dynamically"""
    active_requests = []

    while requests or active_requests:
        # Add new requests
        while requests and len(active_requests) < max_batch:
            active_requests.append(requests.pop(0))

        # Generate one token for all
        outputs = model.generate_next_token(active_requests)

        # Remove completed requests
        active_requests = [
            req for req, out in zip(active_requests, outputs)
            if not is_complete(out)
        ]
```

### 6.2 KV Cache Management

```python
class KVCache:
    """
    Key-Value cache for transformer inference
    """

    def __init__(self, max_batch_size, max_seq_len, num_layers, num_heads, head_dim):
        self.cache_k = torch.zeros(
            num_layers, max_batch_size, num_heads, max_seq_len, head_dim
        )
        self.cache_v = torch.zeros(
            num_layers, max_batch_size, num_heads, max_seq_len, head_dim
        )
        self.seq_lens = torch.zeros(max_batch_size, dtype=torch.long)

    def update(self, layer_idx, batch_idx, key, value):
        """Update cache for a specific position"""
        seq_len = self.seq_lens[batch_idx]

        self.cache_k[layer_idx, batch_idx, :, seq_len] = key
        self.cache_v[layer_idx, batch_idx, :, seq_len] = value

        self.seq_lens[batch_idx] += 1

    def get(self, layer_idx, batch_idx):
        """Get cached KV for a batch element"""
        seq_len = self.seq_lens[batch_idx]

        return (
            self.cache_k[layer_idx, batch_idx, :, :seq_len],
            self.cache_v[layer_idx, batch_idx, :, :seq_len]
        )
```

### 6.3 Speculative Decoding

**병렬로 여러 토큰 생성 시도!**

```python
def speculative_decoding(target_model, draft_model, prompt, k=5):
    """
    Generate k tokens with draft model,
    verify with target model
    """
    tokens = tokenize(prompt)

    while len(tokens) < max_length:
        # 1. Draft: Generate k tokens quickly
        draft_tokens = draft_model.generate(tokens, k)

        # 2. Target: Verify in parallel
        for i in range(k):
            target_prob = target_model.get_prob(tokens + draft_tokens[:i+1])
            draft_prob = draft_model.get_prob(tokens + draft_tokens[:i+1])

            # Accept if target probability is high enough
            if target_prob[-1] > draft_prob[-1] * threshold:
                tokens.append(draft_tokens[i])
            else:
                # Reject remaining
                tokens.append(target_model.sample(tokens))
                break

    return tokens

# Benefits:
# - 2-3x speedup
# - Same quality as target model
# - Requires small draft model
```

---

## 🎓 Complete Deployment Example

```python
# production_inference.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

class InferenceServer:
    def __init__(self, model_path, quantization='int8'):
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if quantization == 'fp16' else torch.float32,
            device_map='auto'
        )

        # Quantize if needed
        if quantization == 'int8':
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt, max_length=100, temperature=0.7):
        """Generate text"""
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# FastAPI app
app = FastAPI(title="Model Inference API")

# Global server instance
server = None

@app.on_event("startup")
async def startup():
    global server
    server = InferenceServer('gpt2', quantization='int8')

class Request(BaseModel):
    prompt: str
    max_length: int = 100
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: Request):
    try:
        output = server.generate(
            request.prompt,
            request.max_length,
            request.temperature
        )
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

---

## 🎓 학습 목표

- [ ] Quantization 원리 이해 및 구현
- [ ] GPTQ, AWQ 사용
- [ ] ONNX, TensorRT 변환
- [ ] FastAPI로 REST API 구축
- [ ] vLLM으로 고처리량 서빙
- [ ] 최적화 기법 (batching, KV cache) 이해

---

## 📊 성능 비교

| Method | Size | Latency | Throughput | Quality |
|--------|------|---------|------------|---------|
| FP32 | 1x | 1x | 1x | 100% |
| FP16 | 0.5x | 0.6x | 1.5x | 99.9% |
| INT8 | 0.25x | 0.4x | 2.5x | 99% |
| INT4 (GPTQ) | 0.125x | 0.3x | 3x | 98% |
| ONNX | 1x | 0.7x | 1.3x | 100% |
| TensorRT | 1x | 0.5x | 2x | 100% |

---

## 🎉 축하합니다!

**Phase 5 완료! 최신 효율화 기법을 모두 마스터했습니다!**

이제 다음을 할 수 있습니다:
- ✅ PEFT로 대규모 모델 효율적 훈련
- ✅ Quantization으로 4-8x 압축
- ✅ Production-ready API 배포
- ✅ 고처리량 inference 시스템 구축

---

## ⏭️ 다음 단계

👉 [Phase 6: Final Project](../phase6-project/)

**이제 모든 지식을 통합하여 자신만의 프로젝트를 만듭니다!** 🚀
