# vLLM Internals 상세 분석

vLLM의 내부 동작 메커니즘을 실제 코드 기반으로 상세히 분석한 한글 문서 모음입니다.

## 📚 문서 목록

### ✅ 완료된 문서

1. **[01_weight_loading.md](01_weight_loading.md)** - Weight 로딩 프로세스
   - 디스크 → CPU 메모리 → GPU 메모리 전체 과정
   - SafeTensors, PyTorch 포맷 비교
   - Multi-thread 로딩, 성능 최적화
   - 실제 vLLM 코드 기반 상세 분석 (~800 lines)

2. **[02_model_initialization.md](02_model_initialization.md)** - 모델 초기화 과정
   - GPUModelRunner 초기화
   - 모델 로딩 with DeviceMemoryProfiler
   - KV Cache 할당 (PagedAttention)
   - Attention Backend 선택 (FlashAttention, xFormers)
   - CUDA Graph 최적화 (~800 lines)

3. **[03_inference_process.md](03_inference_process.md)** - 추론 과정 상세 분석
   - LLMEngine, Scheduler, GPUModelRunner 전체 파이프라인
   - Continuous batching과 preemption
   - Forward pass 및 sampling
   - 성능 최적화 기법 (CUDA graphs, chunked prefill)
   - 실제 성능 측정 및 트러블슈팅 (~1500 lines)

4. **[04_paged_attention.md](04_paged_attention.md)** - PagedAttention 메커니즘
   - 전통적 Attention의 문제점 (메모리 낭비, fragmentation)
   - Block-based memory management
   - Block Pool과 Free Block Queue
   - Prefix Caching with block hashing
   - 7-10x 메모리 효율 개선 (~1300 lines)

5. **[05_transformer_llms.md](05_transformer_llms.md)** - Transformer LLMs (Llama 등)
   - Llama 아키텍처 (Self-attention + Feed-forward)
   - RoPE (Rotary Position Embedding) 상세 구현
   - GQA (Grouped Query Attention) - 64 Q heads, 8 KV heads
   - SwiGLU activation with fused operations
   - RMSNorm normalization
   - Tensor Parallelism과 Quantization
   - 실제 성능 측정 및 최적화 (~1400 lines)

### 🔄 작성 예정 문서

6. **06_moe_llms.md** - Mixture-of-Expert LLMs
   - Mixtral 아키텍처
   - Expert routing
   - Load balancing
   - Deepseek-V2/V3 차이점

7. **07_embedding_models.md** - Embedding Models
   - E5-Mistral 구조
   - Pooling 전략
   - Sentence embedding
   - 성능 최적화

8. **08_multimodal_llms.md** - Multi-modal LLMs
   - LLaVA 아키텍처
   - Vision encoder
   - Projection layer
   - Cross-modal attention

## 🎯 문서 작성 원칙

1. **실제 코드 기반**: 모든 설명은 vLLM 실제 코드를 인용
2. **한글 설명**: 명확한 한글로 작성
3. **코드 예시**: 핵심 코드 스니펫 포함
4. **시각화**: 다이어그램과 flow chart
5. **성능 데이터**: 실측 성능 수치 제공

## 🔧 문서 사용 방법

### 순서대로 읽기 (권장)
```
01 Weight Loading (기초)
  ↓
02 Model Initialization (구조 이해)
  ↓
03 Inference Process (동작 원리)
  ↓
04 PagedAttention (핵심 기술)
  ↓
05-08 모델별 특성 (심화)
```

### 특정 주제만 읽기
- **성능 최적화**: 01, 03, 04
- **모델 아키텍처**: 02, 05-08
- **PagedAttention 이해**: 03, 04
- **특정 모델 분석**: 05-08 중 해당 문서

## 📖 각 문서 구조

모든 문서는 다음 구조를 따릅니다:

```markdown
# 제목

## 목차
- 개요
- 상세 분석
- 실제 코드
- 성능 측정
- 트러블슈팅

## 개요
- 목적과 배경
- 전체 flow diagram

## 상세 분석
- 단계별 설명
- 실제 코드 인용
- 메모리/성능 분석

## 실제 코드
- vLLM 소스 코드 경로
- 핵심 함수 분석
- 코드 스니펫

## 성능 측정
- 벤치마크 결과
- 최적화 전/후 비교
- 권장 설정

## 트러블슈팅
- 흔한 문제
- 해결 방법
- 디버깅 팁
```

## 🔍 코드 참조

모든 문서는 다음 vLLM 코드를 기반으로 작성됩니다:

```
vllm-for-study/
├── vllm/
│   ├── model_executor/
│   │   ├── model_loader/        # Weight loading
│   │   ├── models/              # 모델 구현
│   │   └── layers/              # Layer 구현
│   ├── attention/               # Attention 구현
│   │   ├── backends/
│   │   └── ops/
│   ├── engine/                  # 추론 엔진
│   ├── worker/                  # Worker 프로세스
│   └── core/                    # Core 스케줄러
```

## 🚀 빠른 시작

### 1. Weight Loading 이해하기
```bash
# 문서 읽기
cat 01_weight_loading.md

# 실제 코드 확인
cd /home/user/vllm-for-study
cat vllm/model_executor/model_loader/default_loader.py
```

### 2. 직접 테스트하기
```python
# Weight loading 테스트
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    load_format="safetensors",
    model_loader_extra_config={
        "enable_multithread_load": True,
        "num_threads": 8,
    }
)
```

### 3. 성능 측정하기
```python
import time

start = time.time()
llm = LLM(model="meta-llama/Llama-2-7b-hf")
print(f"Loading time: {time.time() - start:.2f}s")
```

## 📊 예상 로딩 시간

| 모델 | 크기 | Single-thread | Multi-thread (8) |
|------|------|---------------|------------------|
| Llama-2-7B | 13GB | 8s | 3s |
| Llama-2-13B | 26GB | 15s | 5s |
| Llama-2-70B | 140GB | 90s | 18s |
| Mixtral-8x7B | 90GB | 60s | 12s |

## 🛠️ 디버깅 팁

### Weight Loading 디버깅
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 상세 로그 활성화
from vllm import LLM
llm = LLM(model="...", enforce_eager=True)
```

### 메모리 사용량 확인
```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

### 성능 프로파일링
```python
import torch.profiler

with torch.profiler.profile() as prof:
    llm = LLM(model="...")

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## 📚 참고 자료

### 공식 문서
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [Paper: PagedAttention](https://arxiv.org/abs/2309.06180)

### 관련 프로젝트
- [SafeTensors](https://github.com/huggingface/safetensors)
- [Transformers](https://github.com/huggingface/transformers)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)

### 논문
- vLLM: Efficient Memory Management for LLM Serving
- FlashAttention: Fast and Memory-Efficient Exact Attention
- Mixtral of Experts
- LLaVA: Visual Instruction Tuning

## 🤝 기여 방법

더 나은 문서를 위해 기여해주세요:

1. **오류 수정**: 잘못된 내용 발견 시 PR
2. **추가 내용**: 더 자세한 설명이 필요한 부분
3. **코드 예시**: 실용적인 코드 예시 추가
4. **성능 데이터**: 다양한 환경에서의 측정 결과

## 📝 작성자

이 문서는 vLLM 실제 코드를 기반으로 작성되었습니다.

**작성 일자**: 2025-11-06
**vLLM 버전**: Latest (as of 2025-11)
**Status**:
- ✅ 01_weight_loading.md (~800 lines)
- ✅ 02_model_initialization.md (~800 lines)
- ✅ 03_inference_process.md (~1500 lines)
- ✅ 04_paged_attention.md (~1300 lines)
- ✅ 05_transformer_llms.md (~1400 lines)
- 🔄 06-08 작성 중

## 📄 라이선스

이 문서는 vLLM의 Apache 2.0 라이선스를 따릅니다.

---

**다음 단계**: 각 문서를 순서대로 작성하여 vLLM의 전체 동작 원리를 완벽하게 이해할 수 있도록 합니다.
