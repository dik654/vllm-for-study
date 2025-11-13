# Phase 5: 최신 기술 통합

## 🎯 학습 목표

효율성과 프로덕션 배포를 위한 현대적 기법들을 마스터합니다.

1. ✅ Parameter-Efficient Fine-Tuning (PEFT) 기법
2. ✅ Quantization 기법 (GPTQ, AWQ, GGUF)
3. ✅ 모델 최적화 (Pruning, Distillation)
4. ✅ 프로덕션 배포 (ONNX, TensorRT, Serving)

## 📚 학습 내용

### Week 11: [효율화 기법](./01-efficient-methods.md)

#### PEFT (Parameter-Efficient Fine-Tuning)
- **LoRA (Low-Rank Adaptation)**
  - 저차원 분해: ΔW = BA (r≪d)
  - 밑바닥 구현 및 수학적 유도
  - Rank 선택 전략
- **QLoRA**
  - 4-bit + LoRA 조합
  - NormalFloat4 (NF4)
  - Double Quantization
- **Prefix Tuning**
  - Learnable prefix vectors
  - Virtual tokens
- **Adapter Layers**
  - Bottleneck architecture
  - Down → Up projection
- **IA³, BitFit, Prompt Tuning**

#### 비교 실험
- Full Fine-tuning vs PEFT 성능
- 메모리 사용량 측정
- 훈련 속도 비교
- 멀티태스크 학습

### Week 12: [양자화 & 배포](./02-quantization-deployment.md)

#### Quantization 심화
- **Post-Training Quantization (PTQ)**
  - Dynamic Quantization
  - Static Quantization
  - Calibration 전략
- **Quantization-Aware Training (QAT)**
  - Fake quantization
  - Straight-Through Estimator
- **GPTQ (Generative Pre-trained Transformer Quantization)**
  - Layer-wise quantization
  - Optimal Brain Quantization
- **AWQ (Activation-aware Weight Quantization)**
  - Channel-wise scaling
  - Salient weights 보호
- **GGUF/GGML**
  - llama.cpp 포맷
  - CPU 최적화

#### 모델 최적화
- **Pruning**
  - Unstructured pruning (magnitude-based)
  - Structured pruning (channel/layer)
  - Lottery Ticket Hypothesis
- **Knowledge Distillation**
  - Teacher-Student 프레임워크
  - Soft targets
  - Feature-based distillation
- **torch.compile()**
  - TorchDynamo 이해
  - Graph optimization

#### 프로덕션 배포
- **모델 변환**
  - ONNX: 프레임워크 독립성
  - TensorRT: NVIDIA GPU 최적화
  - OpenVINO: Intel 하드웨어
- **Model Serving**
  - TorchServe 설정
  - Triton Inference Server
  - FastAPI로 REST API 구축
  - gRPC for high performance
- **최적화 기법**
  - Batching strategies
  - Continuous batching
  - KV cache 관리
  - Speculative decoding
- **모니터링**
  - Latency tracking
  - Throughput monitoring
  - Resource utilization
  - Model versioning (DVC, MLflow)

## 🎓 완료 기준

- ✅ LoRA로 7B 모델을 소형 GPU에서 fine-tuning
- ✅ INT8 quantization으로 모델 크기 4배 감소
- ✅ ONNX로 2배 이상 inference 속도 향상
- ✅ TorchServe로 프로덕션 API 배포
- ✅ 모델 A/B 테스팅 인프라 구축

## 🛠️ 실전 프로젝트

### 프로젝트 1: LoRA Fine-tuning Pipeline
재사용 가능한 LoRA 훈련 템플릿

### 프로젝트 2: Model Optimization Suite
Quantization + Pruning + Distillation 자동화

### 프로젝트 3: Production ML System
전체 배포 파이프라인 (훈련 → 최적화 → 배포 → 모니터링)

## ⏭️ 다음 단계

[Phase 6: 실전 프로젝트](../phase6-project/)로 진행!
