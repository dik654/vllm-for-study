# Day 1-2: HuggingFace 핵심 용어 마스터

## 📋 목차
1. [Model Card - 모델 문서화 표준](#1-model-card)
2. [모델 저장 포맷](#2-model-formats)
3. [모델 변환](#3-model-conversion)
4. [Quantization 기법](#4-quantization)
5. [Parameter-Efficient Fine-Tuning](#5-peft)

---

## 1. Model Card - 모델 문서화 표준

### 📖 이론

**Model Card**는 머신러닝 모델을 문서화하는 표준 방법입니다. 다음 정보를 포함합니다:

- **Model Details**: 아키텍처, 파라미터 수, 훈련 데이터
- **Intended Use**: 모델의 용도와 한계
- **Performance**: 벤치마크 결과
- **Limitations**: 알려진 편향과 제약사항
- **Training Details**: 하이퍼파라미터, 하드웨어

### 💻 실습 1: Model Card 작성

```python
# examples/create_model_card.py
from huggingface_hub import ModelCard, ModelCardData

# Model Card 데이터 정의
card_data = ModelCardData(
    language='ko',
    license='apache-2.0',
    library_name='transformers',
    tags=['text-generation', 'korean', 'gpt'],
    datasets=['your-dataset'],
    metrics=['perplexity'],
)

# Model Card 내용 작성
content = """
# My Korean GPT Model

## Model Description

이 모델은 한국어 텍스트 생성을 위한 GPT-2 기반 모델입니다.

## Training Data

- **Dataset**: 한국어 Wikipedia + 뉴스 기사
- **Size**: 10GB의 텍스트 데이터
- **Preprocessing**: 특수문자 제거, 소문자 변환

## Training Procedure

### Hyperparameters

- Learning rate: 5e-5
- Batch size: 32
- Epochs: 3
- Optimizer: AdamW
- Warmup steps: 500

### Hardware

- GPU: 4x NVIDIA A100 (40GB)
- Training time: 48 hours

## Evaluation

### Perplexity

| Dataset | Perplexity |
|---------|------------|
| Test    | 15.3       |
| Valid   | 14.8       |

## Limitations

- 2048 토큰 이상의 긴 컨텍스트 처리 어려움
- 전문 용어에 대한 이해 부족
- 편향된 데이터로 인한 잠재적 편향 존재

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-username/model-name")
tokenizer = AutoTokenizer.from_pretrained("your-username/model-name")

text = "안녕하세요"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Citation

```bibtex
@misc{yourmodel2024,
  author = {Your Name},
  title = {My Korean GPT Model},
  year = {2024},
  publisher = {HuggingFace},
  howpublished = {\\url{https://huggingface.co/your-username/model-name}}
}
```
"""

# Model Card 생성
card = ModelCard(content)
card.data = card_data

# 로컬에 저장
card.save("examples/model_card.md")
print("Model Card 작성 완료!")
```

### ✅ TODO 1
- [ ] 위 코드를 실행하여 Model Card 생성
- [ ] HuggingFace Hub에서 인기 모델 3개의 Model Card 읽고 분석
- [ ] 자신만의 가상 모델에 대한 Model Card 작성

---

## 2. 모델 저장 포맷

### 📖 Safetensors vs Pickle

#### **Pickle (.bin, .pt)**

**장점:**
- PyTorch 기본 포맷
- 파이썬 객체 직렬화 가능

**단점:**
- 보안 위험 (임의 코드 실행 가능)
- 느린 로딩 속도
- 메모리 비효율

```python
import torch

# Pickle 포맷 저장
model_state = model.state_dict()
torch.save(model_state, "model.bin")

# Pickle 포맷 로딩
state_dict = torch.load("model.bin")
model.load_state_dict(state_dict)
```

#### **Safetensors**

**장점:**
- 보안 (데이터만 포함, 코드 실행 불가)
- 빠른 로딩 (zero-copy, lazy loading)
- 언어 독립적 (Rust, Python, JS 등)

**단점:**
- 텐서만 저장 가능 (메타데이터는 별도)

### 💻 실습 2: Safetensors 변환

```python
# examples/convert_to_safetensors.py
import torch
from safetensors.torch import save_file, load_file
import time

# 테스트용 큰 모델 생성
print("큰 텐서 생성...")
large_tensor = {
    "layer1.weight": torch.randn(10000, 10000),
    "layer2.weight": torch.randn(10000, 10000),
    "layer3.weight": torch.randn(10000, 10000),
}

# Pickle 저장 및 로딩 시간 측정
print("\n=== Pickle 포맷 ===")
start = time.time()
torch.save(large_tensor, "model_pickle.bin")
save_time_pickle = time.time() - start
print(f"저장 시간: {save_time_pickle:.3f}초")

start = time.time()
loaded_pickle = torch.load("model_pickle.bin")
load_time_pickle = time.time() - start
print(f"로딩 시간: {load_time_pickle:.3f}초")

# Safetensors 저장 및 로딩 시간 측정
print("\n=== Safetensors 포맷 ===")
start = time.time()
save_file(large_tensor, "model_safetensors.safetensors")
save_time_safe = time.time() - start
print(f"저장 시간: {save_time_safe:.3f}초")

start = time.time()
loaded_safe = load_file("model_safetensors.safetensors")
load_time_safe = time.time() - start
print(f"로딩 시간: {load_time_safe:.3f}초")

# 속도 비교
print("\n=== 성능 비교 ===")
print(f"저장 속도 향상: {save_time_pickle/save_time_safe:.2f}x")
print(f"로딩 속도 향상: {load_time_pickle/load_time_safe:.2f}x")

# 파일 크기 비교
import os
pickle_size = os.path.getsize("model_pickle.bin") / (1024**3)
safe_size = os.path.getsize("model_safetensors.safetensors") / (1024**3)
print(f"\nPickle 크기: {pickle_size:.3f} GB")
print(f"Safetensors 크기: {safe_size:.3f} GB")
```

### 💻 실습 3: HuggingFace 모델 변환

```python
# examples/convert_hf_model.py
from transformers import AutoModel
from safetensors.torch import save_file
import torch

def convert_model_to_safetensors(model_name, output_path):
    """
    HuggingFace 모델을 safetensors 포맷으로 변환
    """
    print(f"모델 로딩: {model_name}")
    model = AutoModel.from_pretrained(model_name)

    # State dict 추출
    state_dict = model.state_dict()

    # Safetensors로 저장
    print(f"Safetensors로 변환 중...")
    save_file(state_dict, output_path)

    print(f"완료! 저장 위치: {output_path}")

    # 검증
    from safetensors.torch import load_file
    loaded = load_file(output_path)

    # 텐서 비교
    for key in state_dict.keys():
        assert torch.allclose(state_dict[key], loaded[key])

    print("검증 완료: 모든 텐서가 정확히 일치합니다.")

# 예제 실행
if __name__ == "__main__":
    convert_model_to_safetensors(
        "bert-base-uncased",
        "bert-base-uncased-safetensors.safetensors"
    )
```

### ✅ TODO 2
- [ ] Pickle과 Safetensors 속도 비교 실험 실행
- [ ] 작은 모델을 Safetensors로 변환
- [ ] 변환된 모델 로딩 및 inference 테스트

---

## 3. 모델 변환 (GGUF, ONNX, TorchScript)

### 📖 이론

#### **GGUF (GPT-Generated Unified Format)**
- llama.cpp에서 사용하는 포맷
- CPU에서 빠른 inference
- Quantization 내장

#### **ONNX (Open Neural Network Exchange)**
- 프레임워크 독립적
- 다양한 하드웨어 지원 (CPU, GPU, NPU)
- 최적화된 런타임

#### **TorchScript**
- PyTorch 모델의 중간 표현
- C++에서 실행 가능
- 모바일 배포

### 💻 실습 4: ONNX 변환

```python
# examples/convert_to_onnx.py
import torch
from transformers import AutoModel, AutoTokenizer
import onnx
import onnxruntime as ort
import numpy as np

def convert_to_onnx(model_name, output_path, opset_version=14):
    """
    HuggingFace 모델을 ONNX로 변환
    """
    # 모델과 토크나이저 로딩
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 모델을 evaluation 모드로
    model.eval()

    # 더미 입력 생성
    dummy_input = tokenizer("Hello world", return_tensors="pt")

    # ONNX로 변환
    print(f"ONNX로 변환 중 (opset {opset_version})...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "last_hidden_state": {0: "batch", 1: "sequence"},
        },
        opset_version=opset_version,
    )

    # 변환된 모델 검증
    print("ONNX 모델 검증 중...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX 모델이 유효합니다.")

    # ONNX Runtime으로 inference 테스트
    print("\nONNX Runtime inference 테스트...")
    ort_session = ort.InferenceSession(output_path)

    # PyTorch 모델 출력
    with torch.no_grad():
        pytorch_output = model(**dummy_input).last_hidden_state

    # ONNX 모델 출력
    ort_inputs = {
        "input_ids": dummy_input["input_ids"].numpy(),
        "attention_mask": dummy_input["attention_mask"].numpy(),
    }
    onnx_output = ort_session.run(None, ort_inputs)[0]

    # 출력 비교
    diff = np.abs(pytorch_output.numpy() - onnx_output).max()
    print(f"최대 차이: {diff:.6f}")

    if diff < 1e-4:
        print("✓ PyTorch와 ONNX 출력이 일치합니다!")
    else:
        print("⚠ 출력 차이가 큽니다. 검토가 필요합니다.")

    return output_path

# 예제 실행
if __name__ == "__main__":
    convert_to_onnx(
        "distilbert-base-uncased",
        "distilbert-base-uncased.onnx"
    )
```

### 💻 실습 5: TorchScript 변환

```python
# examples/convert_to_torchscript.py
import torch
from transformers import AutoModel, AutoTokenizer

def convert_to_torchscript(model_name, output_path, use_trace=True):
    """
    HuggingFace 모델을 TorchScript로 변환

    Args:
        use_trace: True면 tracing, False면 scripting 사용
    """
    # 모델 로딩
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # 더미 입력
    dummy_input = tokenizer("Hello world", return_tensors="pt")

    if use_trace:
        print("Tracing 방식으로 변환 중...")
        # Tracing: 실제 실행을 기록
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model,
                (dummy_input["input_ids"], dummy_input["attention_mask"])
            )
        traced_model.save(output_path)
        print(f"✓ Traced 모델 저장: {output_path}")

    else:
        print("Scripting 방식으로 변환 중...")
        # Scripting: 코드를 직접 분석
        scripted_model = torch.jit.script(model)
        scripted_model.save(output_path)
        print(f"✓ Scripted 모델 저장: {output_path}")

    # 변환된 모델 테스트
    print("\nTorchScript 모델 테스트...")
    loaded_model = torch.jit.load(output_path)

    with torch.no_grad():
        original_output = model(**dummy_input).last_hidden_state
        jit_output = loaded_model(
            dummy_input["input_ids"],
            dummy_input["attention_mask"]
        ).last_hidden_state

    # 출력 비교
    diff = torch.abs(original_output - jit_output).max().item()
    print(f"최대 차이: {diff:.6f}")

    if diff < 1e-5:
        print("✓ 원본과 TorchScript 출력이 일치합니다!")

    return output_path

# 예제 실행
if __name__ == "__main__":
    # Tracing 방식
    convert_to_torchscript(
        "distilbert-base-uncased",
        "distilbert_traced.pt",
        use_trace=True
    )
```

### ✅ TODO 3
- [ ] 작은 모델을 ONNX로 변환하고 inference 속도 비교
- [ ] TorchScript의 tracing vs scripting 차이 이해
- [ ] 변환된 모델의 크기와 성능 비교 문서 작성

---

## 4. Quantization 기법

### 📖 이론

**Quantization**은 모델의 가중치와 활성화를 낮은 정밀도로 표현하여:
- 모델 크기 감소
- 추론 속도 향상
- 메모리 사용량 감소

#### **포맷 비교**

| 포맷 | 비트 수 | 범위 | 정밀도 |
|------|---------|------|--------|
| FP32 | 32 | ±3.4×10³⁸ | 높음 |
| FP16 | 16 | ±65,504 | 중간 |
| BF16 | 16 | ±3.4×10³⁸ | 중간 (FP32와 같은 범위) |
| INT8 | 8 | -128~127 | 낮음 |
| INT4 | 4 | -8~7 | 매우 낮음 |

#### **BF16 (Brain Float 16)**
- Google이 개발
- FP32와 같은 지수 범위 (8비트)
- 더 작은 가수 (7비트)
- 훈련에 적합 (overflow 적음)

### 💻 실습 6: 다양한 Precision 비교

```python
# examples/precision_comparison.py
import torch
from transformers import AutoModel, AutoTokenizer
import time

def benchmark_precision(model_name, text, precisions=["fp32", "fp16", "bf16"]):
    """
    다양한 precision에서 모델 성능 비교
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors="pt")

    results = {}

    for precision in precisions:
        print(f"\n=== {precision.upper()} ===")

        # 모델 로딩
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        # Precision 변환
        if precision == "fp16":
            model = model.half()
            inputs_converted = {k: v.half() if v.dtype == torch.float32 else v
                              for k, v in inputs.items()}
        elif precision == "bf16":
            model = model.to(torch.bfloat16)
            inputs_converted = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                              for k, v in inputs.items()}
        else:  # fp32
            inputs_converted = inputs

        # GPU로 이동 (있는 경우)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        inputs_converted = {k: v.to(device) for k, v in inputs_converted.items()}

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(**inputs_converted)

        # 속도 측정
        times = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                output = model(**inputs_converted)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        # 메모리 사용량
        if device == "cuda":
            memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            memory_mb = 0

        # 모델 크기
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        results[precision] = {
            "latency_ms": avg_time * 1000,
            "memory_mb": memory_mb,
            "model_size_mb": param_size_mb,
        }

        print(f"평균 지연시간: {avg_time*1000:.2f} ms")
        print(f"모델 크기: {param_size_mb:.2f} MB")
        if device == "cuda":
            print(f"GPU 메모리: {memory_mb:.2f} MB")

    # 결과 요약
    print("\n=== 결과 요약 ===")
    print(f"{'Precision':<10} {'속도 (ms)':<12} {'모델 크기 (MB)':<18} {'메모리 (MB)':<15}")
    print("-" * 60)

    for precision, metrics in results.items():
        print(f"{precision.upper():<10} {metrics['latency_ms']:<12.2f} "
              f"{metrics['model_size_mb']:<18.2f} {metrics['memory_mb']:<15.2f}")

    return results

# 예제 실행
if __name__ == "__main__":
    results = benchmark_precision(
        "distilbert-base-uncased",
        "The quick brown fox jumps over the lazy dog.",
        precisions=["fp32", "fp16", "bf16"]
    )
```

### 💻 실습 7: INT8 Quantization

```python
# examples/int8_quantization.py
import torch
from transformers import AutoModel, AutoTokenizer
from torch.quantization import quantize_dynamic

def quantize_model_int8(model_name):
    """
    모델을 INT8로 dynamic quantization
    """
    # 모델 로딩
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    # 원본 모델 크기
    original_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"원본 모델 크기: {original_size:.2f} MB")

    # Dynamic Quantization
    print("\nINT8 quantization 적용 중...")
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},  # Linear 레이어만 quantize
        dtype=torch.qint8
    )

    # Quantized 모델 크기
    quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters()) / 1024**2
    print(f"Quantized 모델 크기: {quantized_size:.2f} MB")
    print(f"압축률: {original_size/quantized_size:.2f}x")

    # 출력 비교
    test_text = "This is a test sentence."
    inputs = tokenizer(test_text, return_tensors="pt")

    with torch.no_grad():
        original_output = model(**inputs).last_hidden_state
        quantized_output = quantized_model(**inputs).last_hidden_state

    # 정확도 손실 측정
    diff = torch.abs(original_output - quantized_output).mean().item()
    print(f"\n평균 절대 차이: {diff:.6f}")

    # 상대 오차
    relative_error = (diff / torch.abs(original_output).mean().item()) * 100
    print(f"상대 오차: {relative_error:.2f}%")

    return quantized_model

# 예제 실행
if __name__ == "__main__":
    quantized_model = quantize_model_int8("distilbert-base-uncased")
```

### 💻 실습 8: BitsAndBytes 4-bit Quantization

```python
# examples/4bit_quantization.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_4bit_model(model_name):
    """
    BitsAndBytes를 사용한 4-bit quantization
    """
    # 4-bit 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_use_double_quant=True,  # Nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16,  # Computation dtype
    )

    print("4-bit 모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # 자동으로 GPU 할당
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 메모리 사용량 출력
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"GPU 메모리 사용량: {memory_mb:.2f} MB")

    # Text generation 테스트
    print("\n=== Text Generation 테스트 ===")
    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"생성된 텍스트:\n{generated_text}")

    return model, tokenizer

# 예제 실행
if __name__ == "__main__":
    # 작은 모델로 테스트 (GPT-2)
    model, tokenizer = load_4bit_model("gpt2")
```

### ✅ TODO 4
- [ ] FP32, FP16, BF16 벤치마크 실행 및 결과 비교
- [ ] INT8 quantization 적용하고 정확도 손실 측정
- [ ] 4-bit quantization으로 큰 모델 로딩 (예: GPT-2-large)
- [ ] Quantization 기법별 trade-off 분석 문서 작성

---

## 5. Parameter-Efficient Fine-Tuning (PEFT)

### 📖 이론

**PEFT**는 전체 모델을 fine-tuning하는 대신 일부 파라미터만 조정하여:
- 메모리 효율적
- 빠른 훈련
- 여러 태스크에 대한 adapter 저장 가능

#### **주요 기법**

1. **LoRA (Low-Rank Adaptation)**
   - 가중치 행렬에 저차원 분해 적용
   - W = W₀ + BA (B: d×r, A: r×k, r≪min(d,k))
   - 1-10%의 파라미터만 훈련

2. **Prefix Tuning**
   - 입력에 학습 가능한 prefix 추가
   - 모델 파라미터는 고정

3. **Adapter Layers**
   - Transformer 블록 사이에 작은 네트워크 삽입
   - Down-projection → 활성화 → Up-projection

### 💻 실습 9: LoRA 구현

```python
# examples/lora_implementation.py
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 레이어 구현
    """
    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # 저차원 행렬 A, B 초기화
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

    def forward(self, x):
        # x @ A @ B * scaling
        return (x @ self.lora_A @ self.lora_B) * self.scaling

class LinearWithLoRA(nn.Module):
    """
    기존 Linear 레이어에 LoRA 추가
    """
    def __init__(self, linear, rank=4, alpha=1.0):
        super().__init__()
        self.linear = linear  # 원본 레이어 (frozen)
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha
        )

        # 원본 레이어는 freeze
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 원본 출력 + LoRA 출력
        return self.linear(x) + self.lora(x)

def apply_lora_to_model(model, rank=4, alpha=1.0):
    """
    모델의 모든 Linear 레이어에 LoRA 적용
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Linear를 LinearWithLoRA로 교체
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]

            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model

            lora_layer = LinearWithLoRA(module, rank=rank, alpha=alpha)
            setattr(parent, child_name, lora_layer)

    # 훈련 가능한 파라미터 비율 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"전체 파라미터: {total_params:,}")
    print(f"훈련 가능 파라미터: {trainable_params:,}")
    print(f"비율: {trainable_params/total_params*100:.2f}%")

    return model

# 예제: BERT에 LoRA 적용
if __name__ == "__main__":
    from transformers import AutoModel

    print("원본 BERT 모델 로딩...")
    model = AutoModel.from_pretrained("bert-base-uncased")

    print("\nLoRA 적용 중...")
    model = apply_lora_to_model(model, rank=8, alpha=16.0)

    # 테스트
    test_input = torch.randn(2, 10, 768)  # (batch, seq_len, hidden_size)
    output = model.encoder.layer[0].attention.self.query(test_input)
    print(f"\n출력 shape: {output.shape}")
```

### 💻 실습 10: PEFT 라이브러리 사용

```python
# examples/peft_library.py
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset

def setup_lora_model(model_name, num_labels=2):
    """
    PEFT 라이브러리로 LoRA 모델 설정
    """
    # 기본 모델 로딩
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # LoRA 설정
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence Classification
        r=8,  # Rank
        lora_alpha=32,  # Scaling factor
        lora_dropout=0.1,
        target_modules=["query", "value"],  # Attention의 Q, V에만 적용
    )

    # LoRA 모델 생성
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer

def train_lora_model(model, tokenizer, dataset_name="imdb"):
    """
    LoRA 모델 간단 훈련 예제
    """
    from transformers import Trainer, TrainingArguments

    # 데이터셋 로딩
    print(f"데이터셋 로딩: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train[:1000]")  # 작은 subset

    # 토크나이제이션
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

    # 훈련 설정
    training_args = TrainingArguments(
        output_dir="./lora_model",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=3e-4,
        logging_steps=50,
        save_strategy="no",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("\n훈련 시작...")
    trainer.train()

    # LoRA 가중치 저장 (매우 작음!)
    model.save_pretrained("./lora_weights")
    print("\nLoRA 가중치 저장 완료: ./lora_weights")

    # 저장된 파일 크기 확인
    import os
    lora_size = sum(
        os.path.getsize(os.path.join("./lora_weights", f))
        for f in os.listdir("./lora_weights")
        if os.path.isfile(os.path.join("./lora_weights", f))
    ) / 1024**2
    print(f"LoRA 가중치 크기: {lora_size:.2f} MB")

# 예제 실행
if __name__ == "__main__":
    model, tokenizer = setup_lora_model("distilbert-base-uncased")
    train_lora_model(model, tokenizer)
```

### 💻 실습 11: QLoRA (Quantized LoRA)

```python
# examples/qlora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def setup_qlora_model(model_name):
    """
    QLoRA: 4-bit quantization + LoRA
    """
    # 4-bit quantization 설정
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # 모델 로딩
    print("4-bit 모델 로딩 중...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # K-bit 훈련 준비
    model = prepare_model_for_kbit_training(model)

    # LoRA 설정
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # PEFT 모델 생성
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

# 예제 실행
if __name__ == "__main__":
    # GPU 메모리 확인
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"사용 가능 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    model, tokenizer = setup_qlora_model("gpt2")

    print("\n✓ QLoRA 모델 설정 완료!")
    print("이제 큰 모델도 소형 GPU에서 fine-tuning 가능합니다!")
```

### ✅ TODO 5
- [ ] LoRA를 밑바닥부터 구현하고 동작 확인
- [ ] PEFT 라이브러리로 LoRA fine-tuning 수행
- [ ] QLoRA로 GPU 메모리 절감 효과 확인
- [ ] Full fine-tuning vs LoRA 성능 비교 실험

---

## 📊 Day 1-2 완료 체크리스트

### 이론 이해
- [ ] Model Card의 중요성과 구성 요소 이해
- [ ] Safetensors vs Pickle 차이 명확히 구분
- [ ] ONNX, TorchScript의 용도 이해
- [ ] Quantization의 원리와 trade-off 이해
- [ ] LoRA의 수학적 원리 이해

### 실습 완료
- [ ] Model Card 작성
- [ ] Safetensors 변환 및 속도 비교
- [ ] ONNX 변환 및 검증
- [ ] 다양한 precision 벤치마크
- [ ] INT8, INT4 quantization 적용
- [ ] LoRA 밑바닥 구현
- [ ] PEFT 라이브러리 사용
- [ ] QLoRA 설정

### 심화 과제
- [ ] 자신만의 모델에 대한 완전한 Model Card 작성
- [ ] Quantization 기법별 성능 비교 보고서
- [ ] Custom PEFT 기법 설계 및 구현

### 다음 단계
모든 항목을 완료했다면 [Day 3-4: 모델 허브 구조](./02-model-hub-structure.md)로 진행하세요!

---

## 📚 추가 학습 자료

### 논문
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

### 문서
- [HuggingFace Model Hub Documentation](https://huggingface.co/docs/hub/models)
- [Safetensors Documentation](https://huggingface.co/docs/safetensors/index)
- [PEFT Documentation](https://huggingface.co/docs/peft/index)
- [BitsAndBytes Documentation](https://github.com/TimDettmers/bitsandbytes)

### 블로그
- [Understanding LoRA](https://huggingface.co/blog/lora)
- [Making LLMs lighter with AutoGPTQ](https://huggingface.co/blog/gptq-integration)
